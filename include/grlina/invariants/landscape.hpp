/** @file invariants/landscape.hpp @brief Diagonal barcode landscapes over F2, with in-memory PNG and CSV/SVG output. */
#pragma once
#include <grlina/module/module_base.hpp>
#include <grlina/invariants/hilbert_plot.hpp>
#include <grlina/invariants/landscape_vineyard.hpp>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

template <typename Matrix>
using landscape_point_t = std::array<double, std::tuple_size<typename Matrix::degree_type>::value>;

// Diagonal restriction: a degree a enters p+t*v at max_i((a_i-p_i)/v_i).
// The first map of a projective resolution is already a presentation.
template <typename Matrix>
vec<r2degree> landscape_barcode(const Module<Matrix>& module, const landscape_point_t<Matrix>& center,
                                landscape_point_t<Matrix> direction = [] {
                                    landscape_point_t<Matrix> v{}; v.fill(1); return v;
                                }()) {
    static_assert(std::tuple_size<landscape_point_t<Matrix>>::value > 0, "Landscape needs at least one parameter");
    for (std::size_t i = 0; i < center.size(); ++i)
        if (!std::isfinite(center[i]) || !std::isfinite(direction[i]) || direction[i] <= 0)
            throw std::invalid_argument("Landscape needs a finite center and a finite positive direction");
    if (!module.has_presentation()) throw std::logic_error("Landscape needs a projective presentation or resolution");
    const auto& presentation = module.presentation();
    auto time = [&](const auto& stored_degree) {
        const auto degree = geometric_degree(presentation, stored_degree);
        double t = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < center.size(); ++i) {
            const double entry = (static_cast<double>(degree[i]) - center[i]) / direction[i];
            if (!std::isfinite(entry)) throw std::invalid_argument("Landscape needs finite slice entry times");
            t = std::max(t, entry);
        }
        return t;
    };
    Matrix slice = presentation;
    slice.sort_compatibly([&](const auto& a, const auto& b) {
        const double ta = time(a), tb = time(b);
        return ta == tb ? a < b : ta < tb;
    });
    vec<r2degree> bars;
    vec<double> deaths;
    for (const auto& degree : slice.row_degrees) bars.push_back({time(degree), std::numeric_limits<double>::infinity()});
    for (const auto& degree : slice.col_degrees) deaths.push_back(time(degree));
    // Independent one-off reduction, also used as the reference for vineyard updates.
    slice.column_reduction_triangular();
    for (const auto& pivot : slice.pivots) bars[pivot.first][1] = deaths[pivot.second];
    bars.erase(std::remove_if(bars.begin(), bars.end(), [](const auto& bar) { return bar[0] >= bar[1]; }), bars.end());
    return bars;
}

inline vec<double> landscape_tents(const vec<r2degree>& bars, double t, std::size_t layers) {
    vec<double> values;
    for (const auto& bar : bars) values.push_back(std::max(0.0, std::min(t - bar[0], bar[1] - t)));
    std::partial_sort(values.begin(), values.begin() + std::min(layers, values.size()), values.end(), std::greater<double>{});
    values.resize(layers, 0);
    return values;
}

// Exact one-off query. Construct Landscape for repeated queries and export.
template <typename Matrix>
vec<double> landscape_at(const Module<Matrix>& module, const landscape_point_t<Matrix>& center,
                         std::size_t layers, landscape_point_t<Matrix> direction = [] {
                             landscape_point_t<Matrix> v{}; v.fill(1); return v;
                         }()) {
    return landscape_tents(landscape_barcode(module, center, direction), 0, layers);
}

// Vipond, Lemma 21 / Proposition 41: https://jmlr.org/papers/volume21/19-054/19-054.pdf
// Store full barcodes on parallel diagonals, including infinite bars; never clip to the box.
// In coordinates u=(x-xmin)/vx, v=(y-ymin)/vy, slices have offset s=u-v and time t=(u+v)/2.
// Evaluate each grid line's own barcode, as in Vipond's grid assembly. Never blend slices.
struct Landscape {
    using Box = std::array<double, 4>; // xmin,ymin,xmax,ymax
    Box bounds;
    r2degree direction;
    double resolution, spacing; // relative spacing: resolution * longest side in direction units
    array<r2degree> barcodes; // slice -> (birth,death) pairs; no sampled landscape tensor

    struct Grid {
        Box bounds;
        std::size_t layer;
        array<double> values; // values[x][y], increasing coordinates on both axes
        double maximum = 0;

        // Recolour/re-render an already computed grid, with no algebra work.
        vec<unsigned char> image(ColourMap colormap = ColourMap::blue, double colour_maximum = 0) const {
            if (!std::isfinite(colour_maximum) || colour_maximum < 0)
                throw std::invalid_argument("Landscape colour maximum must be finite and nonnegative");
            const auto xs = image_detail::uniform_axis(bounds[0], bounds[2], values.size());
            const auto ys = image_detail::uniform_axis(bounds[1], bounds[3], values.empty() ? 0 : values.front().size());
            return image_detail::heatmap_image(xs, ys, values,
                "Landscape layer " + std::to_string(layer), "Height", false, 0, colour_maximum, colormap);
        }
    };
    struct Image {
        vec<unsigned char> png;
        double maximum;
        array<double> values; // empty unless explicitly requested
    };

private:
    // Retain only the presentation, so new grid lines outlive the input module too.
    using SliceComputer = std::function<vec<r2degree>(double)>;
    std::function<SliceComputer()> make_slicer_;
    std::size_t max_layers_ = 0;

    double slice_tolerance() const {
        return 8 * std::numeric_limits<double>::epsilon() *
            std::max((bounds[2]-bounds[0])/direction[0], (bounds[3]-bounds[1])/direction[1]);
    }

    const vec<r2degree>* cached_slice(double s) const {
        const auto i = static_cast<std::size_t>(std::clamp(
            std::round((s-offset(0))/spacing), 0.0, double(barcodes.size()-1)));
        return std::abs(s-offset(i)) <= slice_tolerance() ? &barcodes[i] : nullptr;
    }

    // Group actual output points by their slice offset. This also handles
    // rectangular grids and explicit directions without changing the axes.
    // ponytail: sort grid points rather than add rational-stride machinery.
    template <typename Pixel, typename Slice>
    void visit_grid(std::size_t width, std::size_t height, Pixel pixel, Slice slice) const {
        const auto xs = image_detail::uniform_axis(bounds[0], bounds[2], width);
        const auto ys = image_detail::uniform_axis(bounds[1], bounds[3], height);
        struct Point { double s, t; std::size_t x, y; };
        std::vector<Point> points;
        points.reserve(width * height);
        for (std::size_t x=0; x<width; ++x) for (std::size_t y=0; y<height; ++y) {
            const double u = (xs[x]-bounds[0])/direction[0], v = (ys[y]-bounds[1])/direction[1];
            points.push_back({u-v, (u+v)/2, x, y});
        }
        std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.s < b.s; });
        auto compute_slice = make_slicer_(); // Local update state: const queries can run independently.
        std::size_t id = 0;
        for (std::size_t first=0; first<points.size(); ++id) {
            const double s = points[first].s;
            vec<r2degree> computed;
            auto* bars = cached_slice(s);
            if (!bars) { computed = compute_slice(s); bars = &computed; }
            slice(id, s, *bars);
            std::size_t end = first+1;
            while (end < points.size() && points[end].s-s <= slice_tolerance()) ++end;
            for (std::size_t i=first; i<end; ++i)
                pixel(points[i].x, points[i].y, points[i].t, *bars);
            first = end;
        }
    }

public:

    template <typename Matrix>
    static Box default_bounds(const Module<Matrix>& module) {
        if (!module.has_presentation()) throw std::logic_error("Landscape needs a projective presentation or resolution");
        Box box{0,0,1,1};
        if (module.num_of_gens() || module.num_of_rels()) {
            const auto degrees = module.bounding_box();
            box = {double(degrees.first[0]), double(degrees.first[1]), double(degrees.second[0]), double(degrees.second[1])};
        }
        for (std::size_t i = 0; i < 2; ++i) {
            if (box[i] == box[i+2]) { box[i] -= 0.5; box[i+2] += 0.5; }
            const double padding = 0.1 * (box[i+2] - box[i]); // 20% larger overall
            box[i] -= padding; box[i+2] += padding;
        }
        return box; // Betti-degree bounds when the supplied presentation is minimal
    }

    template <typename Matrix>
    explicit Landscape(const Module<Matrix>& module, double resolution = 0.01, r2degree direction = {1,1},
                       bool vineyard = true)
        : Landscape(module, default_bounds(module), resolution, direction, vineyard) {}

    template <typename Matrix>
    Landscape(const Module<Matrix>& module, Box box, double resolution = 0.01, r2degree direction = {1,1},
              bool vineyard = true)
        : bounds(box), direction(direction), resolution(resolution) {
        static_assert(std::tuple_size<landscape_point_t<Matrix>>::value == 2, "Landscape stores two parameters");
        if (!std::isfinite(resolution) || resolution <= 0 || resolution > 1)
            throw std::invalid_argument("Landscape resolution must be in (0,1]");
        r2degree span{};
        for (std::size_t i = 0; i < 2; ++i) {
            span[i] = (bounds[i+2] - bounds[i]) / direction[i];
            if (!std::isfinite(bounds[i]) || !std::isfinite(bounds[i+2]) ||
                !std::isfinite(direction[i]) || direction[i] <= 0 || !std::isfinite(span[i]) || span[i] <= 0)
                throw std::invalid_argument("Landscape needs finite increasing bounds and a finite positive direction");
        }
        // Avoid turning e.g. 998.0000000000001 into 999 intervals at resolution 1/499.
        const double count = std::ceil(std::nextafter(
            (span[0] + span[1]) / (resolution * std::max(span[0], span[1])), 0.0));
        if (!std::isfinite(count) || count >= std::numeric_limits<int>::max())
            throw std::length_error("Landscape resolution requests too many slices");
        spacing = (span[0] + span[1]) / count;
        if (!module.has_presentation()) throw std::logic_error("Landscape needs a projective presentation or resolution");
        max_layers_ = module.num_of_gens();
        make_slicer_ = [snapshot = std::make_shared<const Module<Matrix>>(module.presentation()), box, direction, vineyard]() -> SliceComputer {
            using Index = matrix_index_t<Matrix>;
            return [snapshot, box, direction, vineyard, state = std::optional<landscape_detail::Vineyard<Index>>{}](double s) mutable {
                const r2degree center{box[0] + direction[0]*s/2, box[1] - direction[1]*s/2};
                if (!vineyard) return landscape_barcode(*snapshot, center, direction);
                const auto& matrix = snapshot->presentation();
                auto times = [&](const auto& degrees) {
                    vec<double> result;
                    result.reserve(degrees.size());
                    for (const auto& stored : degrees) {
                        const auto degree = geometric_degree(matrix, stored);
                        const double x = (degree[0]-center[0])/direction[0], y = (degree[1]-center[1])/direction[1];
                        if (!std::isfinite(x) || !std::isfinite(y)) throw std::invalid_argument("Landscape needs finite slice entry times");
                        result.push_back(std::max(x,y));
                    }
                    return result;
                };
                const auto births = times(matrix.row_degrees), deaths = times(matrix.col_degrees);
                auto order = [](const auto& values) {
                    vec<Index> result(values.size());
                    std::iota(result.begin(), result.end(), 0);
                    std::sort(result.begin(), result.end(), [&](auto a, auto b) {
                        return values[a] == values[b] ? a < b : values[a] < values[b];
                    });
                    return result;
                };
                const auto rows = order(births), cols = order(deaths);
                if (!state) state.emplace(matrix, rows, cols);
                else state->update(rows, cols);
                vec<r2degree> bars;
                for (Index i=0; i<static_cast<Index>(births.size()); ++i) {
                    const auto death = state->death(i);
                    const double end = death < 0 ? std::numeric_limits<double>::infinity() : deaths[death];
                    if (births[i] < end) bars.push_back({births[i],end});
                }
                return bars;
            };
        };
        auto compute_slice = make_slicer_();
        for (int i = 0; i <= static_cast<int>(count); ++i) {
            barcodes.push_back(compute_slice(offset(i)));
        }
    }

    double offset(std::size_t slice) const { return -(bounds[3]-bounds[1])/direction[1] + slice*spacing; }
    double error_bound() const { return 0; } // No interpolation error; ordinary floating-point precision applies.

    vec<double> at(const r2degree& point, std::size_t layers = 1) const {
        for (std::size_t i = 0; i < 2; ++i)
            if (!std::isfinite(point[i]) || point[i] < bounds[i] || point[i] > bounds[i+2])
                throw std::out_of_range("Landscape query lies outside its bounding box");
        const double u = (point[0]-bounds[0])/direction[0], v = (point[1]-bounds[1])/direction[1];
        if (const auto* bars = cached_slice(u-v)) return landscape_tents(*bars, (u+v)/2, layers);
        return landscape_tents(make_slicer_()(u-v), (u+v)/2, layers);
    }

    // The numerical grid used by image(): direct values on the output grid's lines.
    Grid sample_grid(std::size_t layer, std::size_t width, std::size_t height) const {
        if (!layer) throw std::invalid_argument("Landscape image layer must be at least one");
        image_detail::validate_size(width, height);
        Grid grid{bounds, layer, array<double>(width, vec<double>(height, 0))};
        if (layer <= max_layers_)
            visit_grid(width, height, [&](auto x, auto y, double t, const auto& bars) {
                if (layer <= bars.size()) {
                    const double value = landscape_tents(bars, t, layer).back();
                    grid.values[x][y] = value;
                    grid.maximum = std::max(grid.maximum, value);
                }
            }, [](auto, double, const auto&) {});
        return grid;
    }

    array<double> sample(std::size_t layer, std::size_t width, std::size_t height) const {
        return sample_grid(layer, width, height).values;
    }

    Image image_with_data(std::size_t layer = 1, std::size_t width = 800, std::size_t height = 600,
                          ColourMap colormap = ColourMap::blue, double colour_maximum = 0,
                          bool include_values = false) const {
        auto grid = sample_grid(layer, width, height);
        auto png = grid.image(colormap, colour_maximum);
        return {std::move(png), grid.maximum, include_values ? std::move(grid.values) : array<double>{}};
    }

    /** PNG bytes for one layer (1-based), without creating a file.
     * Width/height count heatmap pixels; labels add 265 x 185 pixels.
     * Evaluate barcodes on lines through the pixels; reuse cached lines and update
     * persistence across additional lines. No cross-slice interpolation. Colours are linear in height.
     * A positive colour_maximum fixes the scale to [0,colour_maximum] for comparison;
     * zero selects the sampled maximum automatically.
     * Blue shares the Hilbert palette; inferno uses Matplotlib's colour table.
     * Define STB_IMAGE_WRITE_IMPLEMENTATION before any grlina include in one .cpp.
     */
    vec<unsigned char> image(std::size_t layer = 1, std::size_t width = 800,
                             std::size_t height = 600, ColourMap colormap = ColourMap::blue,
                             double colour_maximum = 0) const {
        return image_with_data(layer, width, height, colormap, colour_maximum).png;
    }

    // Export barcodes on the output grid's lines and their direct CSV/SVG samples.
    void save(const std::string& prefix, std::size_t layers = 1,
              ColourMap colormap = ColourMap::blue) const {
        if (!layers) throw std::invalid_argument("Landscape export needs at least one layer");
        std::ofstream slices;
        slices.exceptions(std::ios::failbit | std::ios::badbit);
        slices.open(prefix + ".barcodes.csv");
        slices << std::setprecision(17) << "# bounds=" << bounds[0] << ',' << bounds[1] << ',' << bounds[2] << ',' << bounds[3]
               << ";direction=" << direction[0] << ',' << direction[1] << ";resolution=" << resolution
               << ";sampling=grid\nslice,offset,birth,death\n";
        const int samples = static_cast<int>(std::ceil(std::nextafter(1/resolution, 0.0))) + 1;
        auto coordinate = [&](int i, std::size_t axis) {
            return i == samples - 1 ? bounds[axis + 2] :
                bounds[axis] + (bounds[axis + 2] - bounds[axis]) * (double(i) / (samples - 1));
        };
        array<double> values(static_cast<std::size_t>(samples) * samples);
        double maximum = 0;
        visit_grid(samples, samples, [&](auto x, auto y, double t, const auto& bars) {
            auto& value = values[x * samples + y];
            value = landscape_tents(bars, t, layers);
            maximum = std::max(maximum, value.front());
        }, [&](auto i, double s, const auto& bars) {
            if (bars.empty()) slices << i << ',' << s << ",,\n";
            for (const auto& bar : bars) slices << i << ',' << s << ',' << bar[0] << ',' << bar[1] << '\n';
        });
        slices.close();
        std::ofstream csv, svg;
        csv.exceptions(std::ios::failbit | std::ios::badbit);
        svg.exceptions(std::ios::failbit | std::ios::badbit);
        csv.open(prefix + ".csv"); svg.open(prefix + ".svg");
        csv << std::setprecision(17) << "x,y,layer,value\n";
        svg << std::setprecision(17) << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 "
            << 460.0 * layers << " 430\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/>"
            << "<g font-family=\"sans-serif\" font-size=\"13\" fill=\"#222\">";
        const double cell = 320.0 / samples;
        for (std::size_t k = 0; k < layers; ++k) {
            const double left = 70 + 460.0 * k;
            svg << "<text x=\"" << left + 160 << "\" y=\"25\" text-anchor=\"middle\">&#955;" << k + 1 << "</text>";
            for (int x = 0; x < samples; ++x) for (int y = 0; y < samples; ++y) {
                const double value = values[static_cast<std::size_t>(x) * samples + y][k];
                csv << coordinate(x, 0) << ',' << coordinate(y, 1) << ',' << k + 1 << ',' << value << '\n';
                const auto colour = value > 0 ? heatmap_colour(value / maximum, colormap)
                                              : heatmap_zero_colour(colormap);
                svg << "<rect x=\"" << left + x * cell << "\" y=\"" << 40 + (samples - 1 - y) * cell
                    << "\" width=\"" << cell << "\" height=\"" << cell << "\" fill=\"rgb("
                    << int(colour[0]) << ',' << int(colour[1]) << ',' << int(colour[2]) << ")\"/>";
            }
            svg << "<rect x=\"" << left << "\" y=\"40\" width=\"320\" height=\"320\" fill=\"none\" stroke=\"#222\"/>"
                << "<text x=\"" << left << "\" y=\"380\">" << bounds[0] << "</text>"
                << "<text x=\"" << left + 320 << "\" y=\"380\" text-anchor=\"end\">" << bounds[2] << "</text>"
                << "<text x=\"" << left - 8 << "\" y=\"48\" text-anchor=\"end\">" << bounds[3] << "</text>"
                << "<text x=\"" << left - 8 << "\" y=\"360\" text-anchor=\"end\">" << bounds[1] << "</text>"
                << "<text x=\"" << left + 160 << "\" y=\"405\">x</text><text x=\"" << left - 45 << "\" y=\"200\">y</text>"
                << "<text x=\"" << left + 160 << "\" y=\"425\" text-anchor=\"middle\">"
                << (colormap == ColourMap::blue ? "white: 0; black: " : "black: 0; pale yellow: ")
                << maximum << "</text>";
        }
        svg << "</g></svg>\n";
        csv.close(); svg.close();
    }
};

} // namespace graded_linalg
