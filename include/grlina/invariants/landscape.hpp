/** @file invariants/landscape.hpp @brief Diagonal barcode landscapes over F2, with in-memory PNG and CSV/SVG output. */
#pragma once
#include <grlina/module/module_base.hpp>
#include <grlina/invariants/hilbert_plot.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
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
    // ponytail: reduce each slice independently; vineyard updates can help for very large presentations.
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
// Values are exact on stored slices and linearly interpolated across them, with error <= spacing/4.
struct Landscape {
    using Box = std::array<double, 4>; // xmin,ymin,xmax,ymax
    Box bounds;
    r2degree direction;
    double resolution, spacing; // relative spacing: resolution * longest side in direction units
    array<r2degree> barcodes; // slice -> (birth,death) pairs; no sampled landscape tensor

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
    explicit Landscape(const Module<Matrix>& module, double resolution = 0.01, r2degree direction = {1,1})
        : Landscape(module, default_bounds(module), resolution, direction) {}

    template <typename Matrix>
    Landscape(const Module<Matrix>& module, Box box, double resolution = 0.01, r2degree direction = {1,1})
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
        const double count = std::ceil((span[0] + span[1]) / (resolution * std::max(span[0], span[1])));
        if (!std::isfinite(count) || count >= std::numeric_limits<int>::max())
            throw std::length_error("Landscape resolution requests too many slices");
        spacing = (span[0] + span[1]) / count;
        for (int i = 0; i <= static_cast<int>(count); ++i) {
            const double s = offset(i);
            barcodes.push_back(landscape_barcode(module,
                {bounds[0] + direction[0]*s/2, bounds[1] - direction[1]*s/2}, direction));
        }
    }

    double offset(std::size_t slice) const { return -(bounds[3]-bounds[1])/direction[1] + slice*spacing; }
    double error_bound() const { return spacing/4; }

    vec<double> at(const r2degree& point, std::size_t layers = 1) const {
        for (std::size_t i = 0; i < 2; ++i)
            if (!std::isfinite(point[i]) || point[i] < bounds[i] || point[i] > bounds[i+2])
                throw std::out_of_range("Landscape query lies outside its bounding box");
        const double u = (point[0]-bounds[0])/direction[0], v = (point[1]-bounds[1])/direction[1];
        const double position = std::clamp((u-v-offset(0))/spacing, 0.0, double(barcodes.size()-1));
        const auto i = std::min(static_cast<std::size_t>(position), barcodes.size()-2);
        auto values = landscape_tents(barcodes[i], (u+v)/2, layers);
        const auto next = landscape_tents(barcodes[i+1], (u+v)/2, layers);
        for (std::size_t k = 0; k < layers; ++k) values[k] += (position-i) * (next[k]-values[k]);
        return values;
    }

    /** PNG bytes for one layer (1-based), without creating a file.
     * Width/height count heatmap pixels; labels add 265 x 185 pixels.
     * Evaluate the cached landscape at each pixel, preserving its interpolation
     * error bound and performing no matrix reductions. Colours are linear in height.
     * Define STB_IMAGE_WRITE_IMPLEMENTATION before any grlina include in one .cpp.
     */
    vec<unsigned char> image(std::size_t layer = 1, std::size_t width = 800,
                             std::size_t height = 600) const {
        if (!layer) throw std::invalid_argument("Landscape image layer must be at least one");
        image_detail::validate_size(width, height);
        const auto xs = image_detail::uniform_axis(bounds[0], bounds[2], width);
        const auto ys = image_detail::uniform_axis(bounds[1], bounds[3], height);
        array<double> values(width, vec<double>(height, 0));
        // Layers beyond every stored barcode are identically zero.
        const bool available = std::any_of(barcodes.begin(), barcodes.end(),
            [layer](const auto& bars) { return bars.size() >= layer; });
        if (available)
            for (std::size_t x = 0; x < width; ++x)
                for (std::size_t y = 0; y < height; ++y)
                    values[x][y] = at({xs[x], ys[y]}, layer).back();
        return image_detail::heatmap_image(xs, ys, values,
            "Landscape layer " + std::to_string(layer), "Height", false);
    }

    // Export the cached barcodes and sampled CSV/SVG views.
    void save(const std::string& prefix, std::size_t layers = 1) const {
        if (!layers) throw std::invalid_argument("Landscape export needs at least one layer");
        std::ofstream slices;
        slices.exceptions(std::ios::failbit | std::ios::badbit);
        slices.open(prefix + ".barcodes.csv");
        slices << std::setprecision(17) << "# bounds=" << bounds[0] << ',' << bounds[1] << ',' << bounds[2] << ',' << bounds[3]
               << ";direction=" << direction[0] << ',' << direction[1] << ";resolution=" << resolution
               << ";spacing=" << spacing << "\nslice,offset,birth,death\n";
        for (std::size_t i = 0; i < barcodes.size(); ++i) {
            if (barcodes[i].empty()) slices << i << ',' << offset(i) << ",,\n";
            for (const auto& bar : barcodes[i]) slices << i << ',' << offset(i) << ',' << bar[0] << ',' << bar[1] << '\n';
        }
        slices.close();
        const int samples = static_cast<int>(std::ceil(std::nextafter(1/resolution, 0.0))) + 1;
        auto coordinate = [&](int i, std::size_t axis) {
            return i == samples - 1 ? bounds[axis + 2] :
                bounds[axis] + (bounds[axis + 2] - bounds[axis]) * (double(i) / (samples - 1));
        };
        array<double> values;
        double maximum = 0;
        for (int x = 0; x < samples; ++x) for (int y = 0; y < samples; ++y) {
            values.push_back(at({coordinate(x, 0), coordinate(y, 1)}, layers));
            maximum = std::max(maximum, values.back().front());
        }
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
                const int shade = maximum > 0 ? std::lround(255 * (1 - value / maximum)) : 255;
                svg << "<rect x=\"" << left + x * cell << "\" y=\"" << 40 + (samples - 1 - y) * cell
                    << "\" width=\"" << cell << "\" height=\"" << cell << "\" fill=\"rgb(" << shade << ',' << shade << ",255)\"/>";
            }
            svg << "<rect x=\"" << left << "\" y=\"40\" width=\"320\" height=\"320\" fill=\"none\" stroke=\"#222\"/>"
                << "<text x=\"" << left << "\" y=\"380\">" << bounds[0] << "</text>"
                << "<text x=\"" << left + 320 << "\" y=\"380\" text-anchor=\"end\">" << bounds[2] << "</text>"
                << "<text x=\"" << left - 8 << "\" y=\"48\" text-anchor=\"end\">" << bounds[3] << "</text>"
                << "<text x=\"" << left - 8 << "\" y=\"360\" text-anchor=\"end\">" << bounds[1] << "</text>"
                << "<text x=\"" << left + 160 << "\" y=\"405\">x</text><text x=\"" << left - 45 << "\" y=\"200\">y</text>"
                << "<text x=\"" << left + 160 << "\" y=\"425\" text-anchor=\"middle\">white: 0; blue: " << maximum << "</text>";
        }
        svg << "</g></svg>\n";
        csv.close(); svg.close();
    }
};

} // namespace graded_linalg
