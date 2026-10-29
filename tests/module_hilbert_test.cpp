#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <grlina/invariants/hilbert_function.hpp>
#include "test_degree_matrices.hpp"
#include <grlina/module/module_base.hpp>
#include <cassert>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/graded_kernel.hpp>

using namespace graded_linalg;

// A genuine CRTP matrix that counts local reductions and kernel calls. This
// tests algorithm dispatch, not just equality between two output wrappers.
template <bool WithKernel>
struct ObservedMatrix : GradedSparseMatrix<r2degree, int, ObservedMatrix<WithKernel>> {
    using Base = GradedSparseMatrix<r2degree, int, ObservedMatrix<WithKernel>>;
    using Base::Base;
    explicit ObservedMatrix(SparseMatrix<int>&& matrix) : Base(std::move(matrix)) {}
    inline static int local_calls = 0, kernel_calls = 0;
    inline static bool forbid_local = false;
    auto map_at_degree_pair(r2degree degree, bool shifted = true) const {
        ++local_calls;
        if (forbid_local) throw std::logic_error("Unexpected local presentation evaluation");
        return Base::map_at_degree_pair(degree, shifted);
    }
    template <bool Enabled = WithKernel, std::enable_if_t<Enabled, int> = 0>
    ObservedMatrix graded_kernel() {
        ++kernel_calls;
        R2GradedSparseMatrix<int> source(this->get_num_cols(), this->get_num_rows(),
            this->data, this->col_degrees, this->row_degrees);
        const auto kernel = source.graded_kernel();
        return ObservedMatrix(kernel.get_num_cols(), kernel.get_num_rows(),
            kernel.data, kernel.col_degrees, kernel.row_degrees);
    }
};

template <typename Matrix> Matrix square() {
    // Unit square interval, deliberately unsorted, with a duplicate relation.
    return Matrix(3, 1, {{0}, {0}, {0}}, {{1,0}, {0,1}, {1,0}}, {{0,0}});
}

void test_explicit_resolution_and_incremental_queries() {
    using Matrix = ObservedMatrix<true>;
    using Mod = Module<Matrix>;
    static_assert(has_matrix_graded_kernel<Matrix>::value);
    Mod M(square<Matrix>());
    assert(!M.res_complete_);
    assert(Matrix::kernel_calls == 0 && M.proj_res_.size() == 1);
    assert(M.dimension_at({2,2}) == 0 && Matrix::local_calls == 1);
    assert(Matrix::kernel_calls == 0); // isolated queries do not build a resolution
    auto local_grid = M.hilbert_function();
    assert(local_grid.values == array<int>({{1,0}, {0,0}}));
    assert(!M.res_complete_ && Matrix::kernel_calls == 0 && Matrix::local_calls == 5);
    M.compute_projective_resolution();
    Matrix::forbid_local = true;
    auto grid = M.hilbert_function();
    assert(grid.x_grid == vec<double>({0,1}) && grid.y_grid == vec<double>({0,1}));
    assert(grid.values == array<int>({{1,0}, {0,0}}) && grid.maximum == 1);
    assert(M.res_complete_ && M.proj_res_.size() == 2);
    assert(M.presentation().col_degrees == square<Matrix>().col_degrees); // no ambient rebasing
    const auto calls = Matrix::kernel_calls;
    assert(calls >= 2); // compute the syzygies and establish terminal injectivity
    M.hilbert_function();
    assert(Matrix::kernel_calls == calls && Matrix::local_calls == 5);

    const vec<r2degree> queries{{2,2}, {0,0}, {-1,0}, {0.5,0.5}, {0,1}, {1,0}, {0,0}};
    const auto values = M.dimension_at(queries);
    const vec<int> expected{0,1,0,1,0,0,1};
    for (std::size_t i = 0; i < queries.size(); ++i) {
        assert(values[i] == expected[i]);
        assert(M.dimension_at(queries[i]) == expected[i]);
    }
    assert(M.dimension_at(vec<r2degree>{}).empty());
    // Axes omit the birth at zero and include points before and after support.
    grid = M.hilbert_function({-1,0.5,1,2}, {-1,0.5,1,2});
    array<int> expected_grid(4, vec<int>(4, 0));
    expected_grid[1][1] = 1;
    assert(grid.values == expected_grid && grid.maximum == 1);
    vec<double> fine_axis;
    for (int i = 0; i < 100; ++i) fine_axis.push_back(i / 50.0);
    grid = M.hilbert_function(fine_axis, fine_axis);
    for (int x = 0; x < 100; ++x)
        for (int y = 0; y < 100; ++y) assert(grid.values[x][y] == (x < 50 && y < 50 ? 1 : 0));
    assert(Matrix::kernel_calls == calls && Matrix::local_calls == 5);
    assert(M.hilbert_function({}, {0}).values.empty());
    assert(M.hilbert_function({0}, {}).values == array<int>({{}}));
    M.sort_compatibly();
    M.minimize_resolution();
    assert(M.res_complete_ && M.dimension_at({0,0}) == 1);
    M.shift({1,1});
    assert(M.res_complete_ && M.dimension_at({-1,-1}) == 1);
    auto copy = M;
    copy.minimize_presentation();
    assert(!copy.res_complete_);
    M.remove_relations({0});
    assert(!M.res_complete_ && M.proj_res_.size() == 1);
    Matrix::forbid_local = false;
    assert(M.dimension_at({-1,-1}) == 1);
}

void test_truncation_and_const_queries() {
    using Mat = R2GradedSparseMatrix<int>;
    using Mod = Module<Mat>;
    Mat P(2, 1, {{0}, {0}}, {{0,1}, {1,0}}, {{0,0}});
    Mat duplicated_syzygy(2, 2, {{0,1}, {0,1}}, {{1,1}, {1,1}}, P.col_degrees);
    Mod truncated(ChainComplex<Mat>({P, duplicated_syzygy}));
    assert(!truncated.res_complete_);
    // Naive Euler on the truncation would incorrectly give 1 here.
    assert(truncated.dimension_at({1,1}) == 0);
    truncated.hilbert_function();
    assert(!truncated.res_complete_);
    truncated.compute_projective_resolution();
    assert(truncated.res_complete_);
    assert(truncated.proj_res_.size() == 3);
    assert(truncated.proj_res_[2].data == array<int>({{0,1}}));
    assert(truncated.dimension_at({1,1}) == 0); // includes negative F3 contribution
    truncated.proj_res_ = ChainComplex<Mat>({P, duplicated_syzygy});
    truncated.res_complete_ = false;
    assert(!truncated.res_complete_);

    const Mod read_only(P);
    assert(read_only.hilbert_function().values == array<int>({{1,0}, {0,0}}));
    assert(read_only.proj_res_.size() == 1 && !read_only.res_complete_);

    Mat zero_terminal(0, 2, {}, {}, duplicated_syzygy.col_degrees);
    Mod unmarked(ChainComplex<Mat>({P, duplicated_syzygy, zero_terminal}));
    assert(unmarked.has_presentation() && !unmarked.res_complete_);
    assert(unmarked.dimension_at({1,1}) == 0); // Ignore the unjustified Euler value 1.

    Mod free(Mat(0, 1, {}, {}, {{2,3}}));
    assert(!free.res_complete_);
    free.compute_projective_resolution();
    assert(free.res_complete_);
    assert(free.proj_res_.size() == 1); // already a complete free resolution
    assert(free.dimension_at({1,3}) == 0 && free.dimension_at({2,3}) == 1);
    assert(free.hilbert_function().values == array<int>({{1}}));
    Mod zero(Mat(0, 0, {}, {}, {}));
    auto grid = zero.hilbert_function();
    assert(grid.values.empty() && grid.maximum == 0);
    free.proj_res_.clear();
    free.res_complete_ = false;
    assert(!free.res_complete_);
}

void test_no_kernel_and_explicit_completeness() {
    using Matrix = ObservedMatrix<false>;
    using Mod = Module<Matrix>;
    static_assert(!has_matrix_graded_kernel<Matrix>::value);
    Mod M(square<Matrix>());
    assert(M.hilbert_function().values == array<int>({{1,0}, {0,0}}));
    assert(Matrix::local_calls == 4 && !M.res_complete_);
    assert(Matrix::kernel_calls == 0);

    auto P = square<R2GradedSparseMatrix<int>>();
    auto K = P.graded_kernel();
    Matrix syzygy(K.get_num_cols(), K.get_num_rows(), K.data, K.col_degrees, K.row_degrees);
    auto supplied = Mod::from_projective_resolution(ChainComplex<Matrix>({square<Matrix>(), syzygy}),
                                                   ResolutionCompleteness::complete);
    Matrix::forbid_local = true;
    assert(supplied.dimension_at({1,1}) == 0);
    assert(supplied.hilbert_function().values == array<int>({{1,0}, {0,0}}));
    assert(supplied.dimension_at({{0,0}, {1,1}})[0] == 1);
    assert(Matrix::local_calls == 4);
    std::stringstream scc;
    supplied.to_stream(scc);
    Mod loaded(scc);
    assert(!loaded.res_complete_); // SCC has no completeness marker
    loaded.res_complete_ = true;
    assert(loaded.dimension_at({1,1}) == 0);
    Matrix::forbid_local = false;

    using R4 = NoKernelMatrix<4>;
    R4 injection(1, 1, {{0}}, {CoordinateDegree<double, 4>{1,0,0,0}}, {CoordinateDegree<double, 4>{0,0,0,0}});
    auto higher = Module<R4>::from_projective_resolution(ChainComplex<R4>({injection}),
                                                        ResolutionCompleteness::complete);
    assert((higher.dimension_at(CoordinateDegree<double, 4>{0,0,0,0}) == 1));
    assert((higher.dimension_at(CoordinateDegree<double, 4>{1,0,0,0}) == 0));
    assert((higher.dimension_at(vec<CoordinateDegree<double, 4>>{{0,0,0,0}})[0] == 1));
}

void test_mixed_handcrafted_summands() {
    using Mat = R2GradedSparseMatrix<int>;
    using GridMat = GridGradedSparseMatrix<double, 2, int>;
    using IntegerGrid = GridGradedSparseMatrix<int, 2, int>;
    static_assert(std::is_same_v<base_degree<Mat>, r2degree>);
    static_assert(std::is_same_v<Mat::degree_type, base_degree<Mat>>);
    static_assert(std::is_same_v<base_degree<GridMat>, CoordinateDegree<int, 2>>);
    static_assert(std::is_same_v<GridMat::degree_type, r2degree>);
    static_assert(std::is_same_v<GridGradedSparseMatrix<float, 2, int>::degree_type, CoordinateDegree<float, 2>>);
    static_assert(std::is_same_v<base_degree<IntegerGrid>, IntegerGrid::degree_type>);
    static_assert(matrix_grid_backed_v<IntegerGrid>);
    // Direct sum: rectangle [-1,1)x[-2,1), a quadrant born at zero killed
    // in degree (2,2), and a free summand born at (0.5,0.5).
    Module<Mat> M(Mat(3, 3, {{0}, {0}, {1}}, {{1,-2}, {-1,1}, {2,2}},
                     {{-1,-2}, {0,0}, {0.5,0.5}}));
    auto expected = [](r2degree p) {
        return int(p[0] >= -1 && p[0] < 1 && p[1] >= -2 && p[1] < 1) +
               int(p[0] >= 0 && p[1] >= 0 && !(p[0] >= 2 && p[1] >= 2)) +
               int(p[0] >= 0.5 && p[1] >= 0.5);
    };
    const auto grid = M.hilbert_function({-2,-1,-0.5,0,0.5,1,2,3}, {-3,-2,-1,0,0.5,1,2,3});
    assert(grid.maximum == 3);
    assert(M.hilbert_function().maximum == 3); // Thickness uses the full induced grid.
    Module<GridMat> gridded{GridMat(M.presentation())};
    assert((gridded.presentation().row_degree(0) == r2degree{-1,-2}));
    assert((gridded.presentation().col_degree(0) == r2degree{1,-2}));
    assert(gridded.presentation().row_degree(0) == M.presentation().row_degree(0));
    assert(gridded.presentation().col_degree(0) == M.presentation().col_degree(0));
    assert(support_degrees(gridded) == support_degrees(M));
    assert(gridded.hilbert_function(grid.x_grid, grid.y_grid).values == grid.values);
    gridded.compute_projective_resolution();
    const auto from_resolution = gridded.hilbert_function(grid.x_grid, grid.y_grid);
    assert(from_resolution.values == grid.values && from_resolution.maximum == grid.maximum);
    vec<r2degree> queries;
    for (std::size_t x = 0; x < grid.x_grid.size(); ++x)
        for (std::size_t y = 0; y < grid.y_grid.size(); ++y) {
            r2degree p{grid.x_grid[x], grid.y_grid[y]};
            assert(grid.values[x][y] == expected(p));
            queries.push_back(p);
        }
    std::reverse(queries.begin(), queries.end());
    const auto values = M.dimension_at(queries);
    for (std::size_t i = 0; i < queries.size(); ++i) assert(values[i] == expected(queries[i]));
    const auto grid_values = gridded.dimension_at(queries);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        assert(grid_values[i] == expected(queries[i]));
        assert(gridded.dimension_at(queries[i]) == grid_values[i]);
    }
}

void test_scalar_types_and_stored_axes() {
    using Matrix = GridGradedSparseMatrix<int, 2, long>;
    Matrix::Grid axes{{{-2,0,1,2,7}, {-3,0,1,2,8}}};
    Matrix matrix(1, 1, {{0}}, {{3,3}}, {{1,1}}, axes);
    Module<Matrix> module(matrix);
    const auto function = module.hilbert_function();
    static_assert(std::is_same_v<std::decay_t<decltype(function)>, HilbertFunction<int, long>>);
    assert(function.x_grid == axes[0] && function.y_grid == axes[1]);
    assert(function.values.size() == 5 && function.values[0].size() == 5);
    assert(function.values[0][0] == 0 && function.values[1][1] == 1);
    assert(function.values[3][3] == 0 && function.maximum == 1);
    assert((function.on_grid(vec<double>{-0.5,0.5,1.5,2.5}, vec<double>{0.5}).values ==
            array<long>{{0},{1},{1},{1}}));
    // The last query remains alive because y is below the relation's degree.
    const auto coarse = module.hilbert_function({-1,3}, {-1,3});
    assert(coarse.x_grid == vec<int>({-1,3}));
    assert(coarse.values == array<long>({{0,0},{0,0}}));
    assert(module.dimension_at({1,1}) == 1); // Changes omitted by coarse samples still exist.

    using FloatMatrix = CoordGradedSparseMatrix<float, 2, long>;
    Module<FloatMatrix> floating(FloatMatrix(1, 1, {{0}}, {{1.25f,2.5f}}, {{0.25f,0.5f}}));
    static_assert(std::is_same_v<decltype(floating.hilbert_function()), HilbertFunction<float,long>>);
    const auto expected = floating.hilbert_function().values;
    floating.compute_projective_resolution();
    assert(floating.hilbert_function().values == expected);
    assert(floating.dimension_at(vec<FloatMatrix::degree_type>{{0.25f,0.5f},{1.25f,2.5f}}) == vec<long>({1,0}));
}

void test_png_and_sampling_cost() {
    using Matrix = ObservedMatrix<false>;
    Module<Matrix> module(square<Matrix>());
    const auto calls = Matrix::local_calls;
    const auto function = module.hilbert_function();
    assert(Matrix::local_calls == calls + 4);
    const auto png = graded_linalg::hilbert_function_image(function, 40, 30);
    assert(Matrix::local_calls == calls + 4); // Rendering does no local reductions.
    assert(png.size() > 24 && png[0] == 137 && png[1] == 'P' && png[2] == 'N' && png[3] == 'G');
    auto dimension = [](const auto& bytes, int offset) {
        unsigned value = 0;
        for (int i = offset; i < offset + 4; ++i) value = (value << 8) | bytes[i];
        return value;
    };
    assert(dimension(png,16) == 305 && dimension(png,20) == 215);
    const auto file = std::filesystem::temp_directory_path() / ("grlina-hilbert-" +
        std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".png");
    save_png(png, file.string());
    std::ifstream saved(file, std::ios::binary);
    vec<unsigned char> saved_bytes{std::istreambuf_iterator<char>(saved), std::istreambuf_iterator<char>()};
    assert(saved_bytes == png);
    saved.close();
    std::filesystem::remove(file);
    const auto image = module.hilbert_function_image({0.25,0.75}, {0.25,0.75}, 60, 50);
    assert(dimension(image,16) == 325 && dimension(image,20) == 235);
    assert(Matrix::local_calls == calls + 8); // Four induced points, not 3,000 pixels.
    assert(!module.hilbert_function_image(20, 20).empty());
    assert(Matrix::local_calls == calls + 12);
    assert(!module.res_complete_ && Matrix::kernel_calls == 0);
    assert(!Module<R2GradedSparseMatrix<int>>().hilbert_function_image(20,20).empty());
    Module<R2GradedSparseMatrix<int>> free(R2GradedSparseMatrix<int>(0,1,{}, {}, {{2,3}}));
    assert(!free.hilbert_function_image(20,20).empty());
    const auto zero = Module<R2GradedSparseMatrix<int>>().hilbert_function();
    assert(!graded_linalg::hilbert_function_image(zero,20,20).empty());
    const double adjacent = std::nextafter(1.0, 2.0);
    const HilbertFunction<double,int> narrow{{1,adjacent},{0},{{1},{0}},1};
    assert(!graded_linalg::hilbert_function_image(narrow,40,30).empty());

    auto rejects = [](auto action) {
        bool rejected = false;
        try { action(); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    };
    rejects([&] { module.hilbert_function({1,0}, {0}); });
    assert(module.hilbert_function({0,0}, {0}).values == array<int>({{1},{1}}));
    rejects([&] { module.hilbert_function({NAN}, {0}); });
    rejects([&] { module.hilbert_function_image({}, {0}, 20,20); });
    rejects([&] { module.hilbert_function_image(1,20); });
    rejects([&] { module.hilbert_function_image(50000,50000); });
    rejects([&] { module.hilbert_function_image(20,20,-0.1); });
}

int main() {
    test_explicit_resolution_and_incremental_queries();
    test_truncation_and_const_queries();
    test_no_kernel_and_explicit_completeness();
    test_mixed_handcrafted_summands();
    test_scalar_types_and_stored_axes();
    test_png_and_sampling_cost();
}
