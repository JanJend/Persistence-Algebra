#include <grlina/modules.hpp>
#include <cassert>
#include <sstream>

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

template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

template <typename Matrix> Matrix square() {
    // Unit square interval, deliberately unsorted, with a duplicate relation.
    return Matrix(3, 1, {{0}, {0}, {0}}, {{1,0}, {0,1}, {1,0}}, {{0,0}});
}

void test_automatic_resolution_and_incremental_queries() {
    using Matrix = ObservedMatrix<true>;
    using Mod = Module<Matrix>;
    static_assert(has_matrix_graded_kernel<Matrix>::value);
    Mod M(square<Matrix>());
    assert(!M.has_complete_projective_resolution());
    rejects([&] { M.hilbert_function_on_grid({1,0}, {0}); });
    assert(Matrix::kernel_calls == 0 && M.projective_resolution().size() == 1);
    assert(M.dimension_at({2,2}) == 0 && Matrix::local_calls == 1);
    assert(Matrix::kernel_calls == 0); // isolated queries do not build a resolution
    Matrix::forbid_local = true;
    auto grid = M.hilbert_function_on_induced_grid();
    assert(grid.x_grid == vec<double>({0,1}) && grid.y_grid == vec<double>({0,1}));
    assert(grid.values == array<int>({{1,0}, {0,0}}) && grid.maximum == 1);
    assert(M.has_complete_projective_resolution() && M.projective_resolution().size() == 2);
    assert(M.presentation().col_degrees == square<Matrix>().col_degrees); // no ambient rebasing
    const auto calls = Matrix::kernel_calls;
    assert(calls >= 2); // compute the syzygies and establish terminal injectivity
    M.hilbert_function_on_induced_grid();
    assert(Matrix::kernel_calls == calls && Matrix::local_calls == 1);

    const vec<r2degree> queries{{2,2}, {0,0}, {-1,0}, {0.5,0.5}, {0,1}, {1,0}, {0,0}};
    const auto values = M.hilbert_function(queries);
    const vec<int> expected{0,1,0,1,0,0,1};
    for (std::size_t i = 0; i < queries.size(); ++i) {
        assert(values[i].degree == queries[i] && values[i].dimension == expected[i]);
        assert(M.dimension_at(queries[i]) == expected[i]);
    }
    assert(M.hilbert_function({}).empty());
    // Axes omit the birth at zero and include points before and after support.
    grid = M.hilbert_function_on_grid({-1,0.5,1,2}, {-1,0.5,1,2});
    array<int> expected_grid(4, vec<int>(4, 0));
    expected_grid[1][1] = 1;
    assert(grid.values == expected_grid && grid.maximum == 1);
    vec<double> fine_axis;
    for (int i = 0; i < 100; ++i) fine_axis.push_back(i / 50.0);
    grid = M.hilbert_function_on_grid(fine_axis, fine_axis);
    for (int x = 0; x < 100; ++x)
        for (int y = 0; y < 100; ++y) assert(grid.values[x][y] == (x < 50 && y < 50 ? 1 : 0));
    assert(Matrix::kernel_calls == calls && Matrix::local_calls == 1);
    assert(M.hilbert_function_on_grid({}, {0}).values.empty());
    assert(M.hilbert_function_on_grid({0}, {}).values == array<int>({{}}));
    rejects([&] { M.hilbert_function_on_grid({1,0}, {0}); });
    rejects([&] { M.hilbert_function_on_grid({0,0}, {0}); });
    M.sort_compatibly();
    M.minimize_resolution();
    assert(M.has_complete_projective_resolution() && M.dimension_at({0,0}) == 1);
    M.shift({1,1});
    assert(M.has_complete_projective_resolution() && M.dimension_at({-1,-1}) == 1);
    auto copy = M;
    copy.minimize_presentation();
    assert(!copy.has_complete_projective_resolution());
    M.mutable_presentation();
    assert(!M.has_complete_projective_resolution() && M.projective_resolution().size() == 1);
    Matrix::forbid_local = false;
    assert(M.dimension_at({-1,-1}) == 1);
}

void test_truncation_and_const_queries() {
    using Mat = R2GradedSparseMatrix<int>;
    using Mod = Module<Mat>;
    Mat P(2, 1, {{0}, {0}}, {{0,1}, {1,0}}, {{0,0}});
    Mat duplicated_syzygy(2, 2, {{0,1}, {0,1}}, {{1,1}, {1,1}}, P.col_degrees);
    Mod truncated(ChainComplex<Mat>({P, duplicated_syzygy}));
    assert(!truncated.has_complete_projective_resolution());
    // Naive Euler on the truncation would incorrectly give 1 here.
    assert(truncated.dimension_at({1,1}) == 0);
    truncated.hilbert_function_on_induced_grid();
    assert(truncated.has_complete_projective_resolution());
    assert(truncated.projective_resolution().size() == 3);
    assert(truncated.projective_resolution()[2].data == array<int>({{0,1}}));
    assert(truncated.dimension_at({1,1}) == 0); // includes negative F3 contribution
    truncated.set_projective_resolution(ChainComplex<Mat>({P, duplicated_syzygy}));
    assert(!truncated.has_complete_projective_resolution());

    const Mod read_only(P);
    assert(read_only.hilbert_function_on_induced_grid().values == array<int>({{1,0}, {0,0}}));
    assert(read_only.projective_resolution().size() == 1 && !read_only.has_complete_projective_resolution());

    Mod free(Mat(0, 1, {}, {}, {{2,3}}));
    assert(free.has_complete_projective_resolution());
    free.compute_projective_resolution();
    assert(free.projective_resolution().size() == 1); // already a complete free resolution
    assert(free.dimension_at({1,3}) == 0 && free.dimension_at({2,3}) == 1);
    assert(free.hilbert_function_on_induced_grid().values == array<int>({{1}}));
    Mod zero(Mat(0, 0, {}, {}, {}));
    auto grid = zero.hilbert_function_on_induced_grid();
    assert(grid.values.empty() && grid.maximum == 0);
    free.clear_projective_resolution();
    assert(!free.has_complete_projective_resolution());
}

void test_no_kernel_and_explicit_completeness() {
    using Matrix = ObservedMatrix<false>;
    using Mod = Module<Matrix>;
    static_assert(!has_matrix_graded_kernel<Matrix>::value);
    Mod M(square<Matrix>());
    assert(M.hilbert_function_on_induced_grid().values == array<int>({{1,0}, {0,0}}));
    assert(Matrix::local_calls == 4 && !M.has_complete_projective_resolution());
    assert(Matrix::kernel_calls == 0);

    auto P = square<R2GradedSparseMatrix<int>>();
    auto K = P.graded_kernel();
    Matrix syzygy(K.get_num_cols(), K.get_num_rows(), K.data, K.col_degrees, K.row_degrees);
    auto supplied = Mod::from_projective_resolution(ChainComplex<Matrix>({square<Matrix>(), syzygy}),
                                                   ResolutionCompleteness::complete);
    Matrix::forbid_local = true;
    assert(supplied.dimension_at({1,1}) == 0);
    assert(supplied.hilbert_function_on_induced_grid().values == array<int>({{1,0}, {0,0}}));
    assert(supplied.hilbert_function({{0,0}, {1,1}})[0].dimension == 1);
    assert(Matrix::local_calls == 4);
    std::stringstream scc;
    supplied.to_stream(scc);
    Mod loaded(scc);
    assert(!loaded.has_complete_projective_resolution()); // SCC has no completeness marker
    loaded.set_projective_resolution_completeness(ResolutionCompleteness::complete);
    assert(loaded.dimension_at({1,1}) == 0);
    Matrix::forbid_local = false;

    using R4 = R4GradedSparseMatrix<int>;
    R4 injection(1, 1, {{0}}, {r4degree(1,0,0,0)}, {r4degree(0,0,0,0)});
    auto higher = Module<R4>::from_projective_resolution(ChainComplex<R4>({injection}),
                                                        ResolutionCompleteness::complete);
    assert(higher.dimension_at(r4degree(0,0,0,0)) == 1);
    assert(higher.dimension_at(r4degree(1,0,0,0)) == 0);
    assert(higher.hilbert_function({r4degree(0,0,0,0)})[0].dimension == 1);
}

void test_mixed_handcrafted_summands() {
    using Mat = R2GradedSparseMatrix<int>;
    // Direct sum: rectangle [-1,1)x[-2,1), a quadrant born at zero killed
    // in degree (2,2), and a free summand born at (0.5,0.5).
    Module<Mat> M(Mat(3, 3, {{0}, {0}, {1}}, {{1,-2}, {-1,1}, {2,2}},
                     {{-1,-2}, {0,0}, {0.5,0.5}}));
    auto expected = [](r2degree p) {
        return int(p.first >= -1 && p.first < 1 && p.second >= -2 && p.second < 1) +
               int(p.first >= 0 && p.second >= 0 && !(p.first >= 2 && p.second >= 2)) +
               int(p.first >= 0.5 && p.second >= 0.5);
    };
    const auto grid = M.hilbert_function_on_grid({-2,-1,-0.5,0,0.5,1,2,3}, {-3,-2,-1,0,0.5,1,2,3});
    assert(grid.maximum == 3);
    vec<r2degree> queries;
    for (std::size_t x = 0; x < grid.x_grid.size(); ++x)
        for (std::size_t y = 0; y < grid.y_grid.size(); ++y) {
            r2degree p{grid.x_grid[x], grid.y_grid[y]};
            assert(grid.values[x][y] == expected(p));
            queries.push_back(p);
        }
    std::reverse(queries.begin(), queries.end());
    const auto values = M.hilbert_function(queries);
    for (const auto& value : values) assert(value.dimension == expected(value.degree));
}

int main() {
    test_automatic_resolution_and_incremental_queries();
    test_truncation_and_const_queries();
    test_no_kernel_and_explicit_completeness();
    test_mixed_handcrafted_summands();
}
