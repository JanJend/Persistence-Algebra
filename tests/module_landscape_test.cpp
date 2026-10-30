#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/landscape.hpp>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <random>
#include <sstream>
#include <grlina/matrices/matrix_algorithms.hpp>

using namespace graded_linalg;
using Mat = R2Matrix<>;

struct CountedMatrix : CoordGradedSparseMatrix<double, 2, int, CountedMatrix> {
    using Base = CoordGradedSparseMatrix<double, 2, int, CountedMatrix>;
    using Base::Base;
    inline static int reductions = 0;
    void column_reduction_triangular(bool remove_zero = false) {
        ++reductions;
        Base::column_reduction_triangular(remove_zero);
    }
};

// Independent F2 rank calculation: dim(R_to + generators_from) - dim(R_to).
int transition_rank(const Mat& matrix, r2degree from, r2degree to) {
    vec<std::uint64_t> relations;
    for (int j = 0; j < matrix.get_num_cols(); ++j)
        if (Degree_traits<r2degree>::smaller_equal(matrix.col_degrees[j], to)) {
            std::uint64_t column = 0;
            for (auto row : matrix.get_col(j)) column ^= std::uint64_t{1} << row;
            relations.push_back(column);
        }
    auto rank = [](const auto& columns) {
        std::array<std::uint64_t, 64> basis{};
        int result = 0;
        for (auto column : columns) for (int row = 63; row >= 0; --row)
            if (column & (std::uint64_t{1} << row)) {
                if (basis[row]) column ^= basis[row];
                else { basis[row] = column; ++result; break; }
            }
        return result;
    };
    const int relations_rank = rank(relations);
    for (int i = 0; i < matrix.get_num_rows(); ++i)
        if (Degree_traits<r2degree>::smaller_equal(matrix.row_degrees[i], from))
            relations.push_back(std::uint64_t{1} << i);
    return rank(relations) - relations_rank;
}

template <typename F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main(int argc, char** argv) {
    // Deliberately unsorted: free quadrant born at (1/4,1/4), and a unit square.
    const Mat presentation(4, 2, {{1}, {1}, {1}, {}},
                           {{0,1}, {1,0}, {1,0}, {2,2}}, {{0.25,0.25}, {0,0}});
    const Module<Mat> module(presentation);
    assert(landscape_at(module, {0.5,0.5}, 3) == vec<double>({0.5,0.25,0}));
    assert(landscape_at(module, {0.5,0.5}, 3, {2,1}) == vec<double>({0.25,0.125,0}));
    assert(landscape_at(module, {1,1}, 3) == vec<double>({0.75,0,0}));
    assert(landscape_at(module, {-1,-1}, 3) == vec<double>({0,0,0}));
    assert(landscape_at(module, {0,0}, 0).empty());
    auto resolved = module;
    resolved.compute_projective_resolution();
    assert(resolved.proj_res_.size() > 1);
    assert(landscape_at(resolved, {0.5,0.5}, 3) == vec<double>({0.5,0.25,0}));
    assert(resolved.presentation().data == presentation.data); // queries preserve the representation
    Module<GridGradedSparseMatrix<double, 2, int>> grid{
        GridGradedSparseMatrix<double, 2, int>(presentation)};
    assert(landscape_at(grid, {0.5,0.5}, 3) == vec<double>({0.5,0.25,0}));
    Module<R3Matrix<>> higher(R3Matrix<>(1, 1, {{0}}, {{1,1,1}}, {{0,0,0}}));
    assert(landscape_at(higher, {0.5,0.5,0.5}, 2) == vec<double>({0.5,0}));
    const Module<Mat> zero(Mat(0, 0));
    assert(landscape_at(zero, {0,0}, 2) == vec<double>({0,0}));
    rejects([&] { landscape_at(Module<Mat>{}, {0,0}, 1); });
    rejects([&] { landscape_at(module, {0,0}, 1, {0,1}); });
    rejects([&] { landscape_at(module, {NAN,0}, 1); });
    rejects([&] { landscape_at(module, {0,0}, 1, {INFINITY,1}); });
    rejects([&] { landscape_at(Module<Mat>(Mat(0,1, {}, {}, {{NAN,0}})), {0,0}, 1); });

    const Landscape landscape(module, {0,0,1,1}, 0.25);
    assert(landscape.barcodes.size() == 9); // 5x5 grid: 9 diagonal barcodes, not 25
    assert(landscape.at({0.5,0.5}, 3) == vec<double>({0.5,0.25,0}));
    // Between cached slices, interpolation used to invent positive boundary values.
    assert(landscape.at({0,0.1}, 3) == vec<double>({0,0,0}));
    assert(landscape.at({0.1,0}, 3) == vec<double>({0,0,0}));
    assert(landscape.error_bound() == 0);
    const Landscape automatic(module);
    assert(automatic.bounds == Landscape::Box({-0.2,-0.2,2.2,2.2}));
    assert(automatic.resolution == 0.01 && automatic.barcodes.size() == 201);
    const Landscape grid_landscape(grid, {0,0,1,1}, 0.25);
    const Landscape resolved_landscape(resolved, {0,0,1,1}, 0.25);
    assert(grid_landscape.barcodes == landscape.barcodes && resolved_landscape.barcodes == landscape.barcodes);
    const Landscape free(Module<Mat>(Mat(0,1, {}, {}, {{0,0}})));
    assert(free.bounds == Landscape::Box({-0.6,-0.6,0.6,0.6}));
    assert(std::isinf(free.barcodes[100][0][1]));
    const Landscape empty(zero);
    assert(empty.at({0.5,0.5}, 3) == vec<double>({0,0,0}));
    auto snapshot = [&] { return Landscape(Module<Mat>(presentation), {0,0,1,1}, 0.25); }();
    assert(snapshot.at({0.5,0.5}, 3) == vec<double>({0.5,0.25,0})); // source lifetime is irrelevant
    assert(std::abs(snapshot.at({0.13,0.24}).front()-0.13) < 1e-12); // including new lines
    rejects([&] { Landscape invalid(module, {1,0,0,1}); });
    rejects([&] { Landscape invalid(module, 0); });
    rejects([&] { Landscape invalid(module, NAN); });
    rejects([&] { Landscape invalid(module, 1e-200); });
    rejects([&] { Landscape invalid(module, {0,0,1,1}, 0.1, {0,1}); });
    rejects([&] { landscape.at({-1,0}); });
    rejects([&] { landscape.at({NAN,0}); });

    Module<CountedMatrix> counted(CountedMatrix(2,1, {{0},{0}}, {{1,0},{0,1}}, {{0,0}}));
    const Landscape compressed(counted, {0,0,1,1}, 0.01, {1,1}, false);
    assert(CountedMatrix::reductions == 201 && compressed.barcodes.size() == 201);
    compressed.at({0.13,0.24}, 3);
    assert(CountedMatrix::reductions == 201);
    compressed.sample(1, 101, 101);
    assert(CountedMatrix::reductions == 201); // Reuse every cached line on an aligned grid.
    const auto png = compressed.image(1, 40, 30);
    const int after_png = CountedMatrix::reductions;
    assert(after_png > 201); // A different grid requires its own lines.
    assert(png.size() > 24);
    assert(std::string(png.begin(), png.begin()+8) == std::string("\x89PNG\r\n\x1a\n",8));
    auto image_dimension = [](const auto& bytes, int offset) {
        unsigned value = 0;
        for (int i = offset; i < offset+4; ++i) value = (value << 8) | bytes[i];
        return value;
    };
    assert(image_dimension(png,16) == 305 && image_dimension(png,20) == 215);
    assert(!compressed.image(2, 40, 30).empty()); // No second layer: a valid zero image.
    assert(!compressed.image(std::numeric_limits<std::size_t>::max(), 20, 20).empty());
    assert(CountedMatrix::reductions == after_png); // Absent layers require no reductions.
    assert(!empty.image(1, 20, 20).empty());
    assert(!free.image(1, 20, 20).empty());
    const auto values = landscape.sample(1, 5, 5);
    const auto sampled_grid = landscape.sample_grid(1, 5, 5);
    assert(sampled_grid.values == values && sampled_grid.maximum == 0.75);
    const auto image_data = landscape.image_with_data(1, 5, 5, ColourMap::blue, 2, true);
    assert(image_data.maximum == 0.75 && image_data.values == values);
    assert(image_data.png == sampled_grid.image(ColourMap::blue, 2));
    assert(landscape.image_with_data(1, 5, 5).values.empty());
    const auto reductions_before_render = CountedMatrix::reductions;
    const auto reusable = compressed.sample_grid(1, 101, 101);
    reusable.image(ColourMap::blue, 1);
    reusable.image(ColourMap::inferno, 2);
    assert(CountedMatrix::reductions == reductions_before_render); // Sampling reused the cache; renders do no algebra.
    for (std::size_t x = 0; x < 5; ++x)
        for (std::size_t y = 0; y < 5; ++y)
            assert(values[x][y] == landscape.at({x / 4.0, y / 4.0}).front());
    assert(compressed.image(1, 40, 30, ColourMap::blue, 2) != png);
    assert(CountedMatrix::reductions-after_png == after_png-201);
    rejects([&] { compressed.image(1, 20, 20, ColourMap::blue, -1); });
    rejects([&] { compressed.image(1, 20, 20, ColourMap::blue, NAN); });
    rejects([&] { compressed.image(1, 20, 20, ColourMap::blue, INFINITY); });
    rejects([&] { compressed.image(0); });
    rejects([&] { compressed.image(1, 0, 20); });
    rejects([&] { compressed.image(1, 20, 1); });
    rejects([&] { compressed.image(1, 50000, 50000); });

    // The notebook's 500x500 grid needs 999 lines. A floating-point ceil
    // previously made 1000, forcing interpolation at almost every pixel.
    CountedMatrix::reductions = 0;
    const Landscape notebook_grid(counted, {0,0,1,1}, 1.0/499, {1,1}, false);
    assert(notebook_grid.barcodes.size() == 999 && CountedMatrix::reductions == 999);
    const auto notebook_values = notebook_grid.sample(1, 500, 500);
    assert(CountedMatrix::reductions == 999);
    for (int x=0; x<500; ++x) for (int y=0; y<500; ++y)
        assert(std::abs(notebook_values[x][y]-std::min({x/499.0,y/499.0,1-x/499.0,1-y/499.0})) < 1e-12);

    std::mt19937 random(42);
    for (int trial = 0; trial < 30; ++trial) {
        vec<r2degree> births, deaths;
        array<int> columns;
        for (int i = 0; i < 6; ++i) births.push_back({double(random()%5)/4, double(random()%5)/4});
        for (int j = 0; j < 9; ++j) {
            vec<int> column;
            r2degree degree{};
            for (int row = 0; row < 6; ++row) if (random()%2) {
                column.push_back(row);
                degree = Degree_traits<r2degree>::join(degree, births[row]);
            }
            degree = degree + r2degree{double(random()%3)/4, double(random()%3)/4};
            columns.push_back(column); deaths.push_back(degree);
        }
        const Module<Mat> coupled(Mat(9, 6, columns, deaths, births));
        // Exercise both directions of adjacent swaps, including zero columns,
        // dependent relations and arbitrary rectangular (non-boundary) matrices.
        vec<int> row_order{0,1,2,3,4,5}, col_order{0,1,2,3,4,5,6,7,8};
        landscape_detail::Vineyard<int> vine(coupled.presentation(), row_order, col_order);
        for (int update=0; update<100; ++update) {
            std::shuffle(row_order.begin(), row_order.end(), random);
            std::shuffle(col_order.begin(), col_order.end(), random);
            vine.update(row_order, col_order);
            SparseMatrix<int> reference(9,6);
            reference.resize_data(9);
            vec<int> positions(6);
            for (int i=0; i<6; ++i) positions[row_order[i]] = i;
            for (int j=0; j<9; ++j) {
                auto column = columns[col_order[j]];
                for (auto& i : column) i = positions[i];
                std::sort(column.begin(), column.end());
                reference.set_col(j, column);
            }
            reference.column_reduction_triangular();
            for (int i=0; i<6; ++i) {
                auto found = reference.pivots.find(positions[i]);
                assert(vine.death(i) == (found == reference.pivots.end() ? -1 : col_order[found->second]));
            }
        }
        const Landscape sampled(coupled, {0,0,2,2}, 0.04, {1,2});
        const Landscape fresh(coupled, {0,0,2,2}, 0.04, {1,2}, false);
        for (std::size_t i=0; i<sampled.barcodes.size(); ++i) {
            auto updated = sampled.barcodes[i], independent = fresh.barcodes[i];
            std::sort(updated.begin(), updated.end());
            std::sort(independent.begin(), independent.end());
            assert(updated == independent);
        }
        // On stored diagonals, all layers are exact, including away from grid points.
        for (std::size_t i = 1; i+1 < sampled.barcodes.size(); ++i) {
            const double s = sampled.offset(i);
            const r2degree p{s >= 0 ? s : 0.0, s >= 0 ? 0.0 : -2*s};
            if (p[0] > 2 || p[1] > 2) continue;
            const auto exact = landscape_at(coupled, p, 8, {1,2}), values = sampled.at(p, 8);
            for (std::size_t k = 0; k < exact.size(); ++k) assert(std::abs(exact[k]-values[k]) < 1e-12);
        }
        // Rectangular grids and weighted directions must also use their actual lines.
        const auto pixels = sampled.sample(2, 9, 7);
        for (int x=0; x<9; ++x) for (int y=0; y<7; ++y)
            assert(std::abs(pixels[x][y]-landscape_at(coupled, {x/4.0,y/3.0}, 2, {1,2}).back()) < 1e-12);
        for (int query = 0; query < 12; ++query) {
            const r2degree center{double(random()%17)/8, double(random()%17)/8}, direction{1,2};
            const auto heights = sampled.at(center, 8);
            const auto exact = landscape_at(coupled, center, 8, direction);
            for (std::size_t k = 0; k < heights.size(); ++k)
                assert(std::abs(exact[k]-heights[k]) < 1e-12);
            for (std::size_t k = 0; k < heights.size(); ++k)
                // Avoid asking the rank oracle to resolve sub-roundoff perturbations.
                for (double epsilon : {heights[k]/2, heights[k]+1e-6}) if (epsilon > 1e-12)
                    assert((transition_rank(coupled.presentation(), center-epsilon*direction, center+epsilon*direction) >= int(k+1))
                           == (epsilon < heights[k]));
        }
    }

    const std::string prefix = argc > 1 ? argv[1] :
        (std::filesystem::temp_directory_path() / "grlina_module_landscape_test").string();
    landscape.save(prefix, 3);
    std::ifstream csv(prefix + ".csv");
    std::string line;
    std::getline(csv, line);
    assert(line == "x,y,layer,value");
    int count = 0;
    while (std::getline(csv, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        double x, y, value; int k;
        row >> x >> y >> k >> value;
        assert(row && k >= 1 && k <= 3);
        vec<double> expected{std::max(0.0, std::min({x,y,1-x,1-y})), std::max(0.0, std::min(x-0.25,y-0.25)), 0};
        std::sort(expected.begin(), expected.end(), std::greater<double>{});
        assert(value == expected[k-1]); ++count;
    }
    assert(count == 75);
    std::ifstream svg(prefix + ".svg");
    std::stringstream image; image << svg.rdbuf();
    assert(image.str().find("&#955;3") != std::string::npos);
    std::ifstream bars(prefix + ".barcodes.csv");
    std::getline(bars, line);
    assert(line == "# bounds=0,0,1,1;direction=1,1;resolution=0.25;sampling=grid");
    std::getline(bars, line);
    assert(line == "slice,offset,birth,death");
    array<r2degree> restored(landscape.barcodes.size());
    while (std::getline(bars, line)) {
        std::istringstream row(line);
        std::string id, offset, birth, death;
        std::getline(row, id, ','); std::getline(row, offset, ',');
        std::getline(row, birth, ','); std::getline(row, death);
        const auto i = std::stoul(id);
        assert(std::stod(offset) == landscape.offset(i));
        if (!birth.empty()) restored[i].push_back({std::stod(birth), std::stod(death)});
    }
    assert(restored == landscape.barcodes);
    const auto read = [](const std::string& path) {
        std::ifstream stream(path);
        return std::string(std::istreambuf_iterator<char>(stream), {});
    };
    const auto blue_csv = read(prefix + ".csv"), blue_barcodes = read(prefix + ".barcodes.csv");
    landscape.save(prefix, 3, ColourMap::inferno);
    assert(read(prefix + ".csv") == blue_csv && read(prefix + ".barcodes.csv") == blue_barcodes);
    const auto inferno_svg = read(prefix + ".svg");
    assert(inferno_svg.find("rgb(0,0,3)") != std::string::npos);
    assert(inferno_svg.find("rgb(252,254,164)") != std::string::npos);
    assert(inferno_svg.find("black: 0; pale yellow:") != std::string::npos);
    rejects([&] { landscape.save(prefix, 0); });
    rejects([&] { landscape.save(prefix + "/missing"); });
    if (argc == 1) for (const auto* extension : {".csv", ".svg", ".barcodes.csv"})
        std::filesystem::remove(prefix + extension);
}
