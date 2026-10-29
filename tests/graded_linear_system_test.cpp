#include <grlina/graded_linear_system.hpp>
#include <grlina/matrices/r2.hpp>
#include <grlina/matrices/coord.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/upset.hpp>
#include "test_degree_matrices.hpp"
#include "graded_linear_system_baselines.hpp"
#include <cassert>
#include <random>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/matrices/io.hpp>

using namespace graded_linalg;

template<class Matrix>
Matrix matrix(array<int> columns, vec<typename Matrix::degree_type> degrees,
              vec<typename Matrix::degree_type> rows) {
    auto result = GradedMatrixIO<Matrix>::make_matrix(
        static_cast<int>(columns.size()), static_cast<int>(rows.size()));
    result.resize_data(columns.size());
    set_geometric_degrees(result, degrees, rows);
    for (int j = 0; j < result.get_num_cols(); ++j) result.set_col(j, columns[j]);
    result.compatibly_sorted = false;
    return result;
}

template<class Matrix>
void check(const Matrix& A, const Matrix& B, bool solvable) {
    const auto original_a = A, original_b = B;
    auto X = solve_graded_linear_system(A, B);
    assert(X.has_value() == solvable);
    assert((solve_graded_linear_system<Matrix, false>(A, B) == solvable));
    assert(graded_linear_system_is_solvable(A, B) == solvable);
    if constexpr (std::tuple_size<typename Matrix::degree_type>::value == 2) {
        auto movable_a = A, movable_b = B;
        auto moved = solve_bigraded_linear_system(std::move(movable_a), std::move(movable_b));
        assert(moved.has_value() == solvable);
        if (moved) assert(moved->data == X->data);
    }
    auto compare = [&](const std::optional<Matrix>& other) {
        assert(other.has_value() == solvable);
        if (!other) return;

        assert(geometric_col_degrees(*other) == geometric_col_degrees(B));
        assert(geometric_row_degrees(*other) == geometric_col_degrees(A));
        auto image = A * *other;
        for (int j = 0; j < B.get_num_cols(); ++j) assert(image.get_col(j) == B.get_col(j));
        // Coefficients may differ for a nonunique solution, but their
        // difference must be a graded kernel vector at each RHS birth.
        Matrix difference = *X;
        for (int j = 0; j < B.get_num_cols(); ++j) {
            auto left = X->get_col(j), right = other->get_col(j);
            vec<int> sum;
            std::set_symmetric_difference(left.begin(), left.end(), right.begin(), right.end(),
                                         std::back_inserter(sum));
            difference.set_col(j, sum);
        }

        auto residual = A * difference;
        for (int j = 0; j < B.get_num_cols(); ++j) assert(residual.get_col(j).empty());
    };
    compare(test::column_by_column_solver(A, B));
    if constexpr (has_matrix_graded_kernel<Matrix>::value) compare(test::full_kernel_solver(A, B));
    assert(A.data == original_a.data && B.data == original_b.data);
    assert(A.col_degrees == original_a.col_degrees && A.row_degrees == original_a.row_degrees);
    assert(B.col_degrees == original_b.col_degrees && B.row_degrees == original_b.row_degrees);
    if constexpr (matrix_grid_backed_v<Matrix>) {
        assert(A.grid_ == original_a.grid_ && B.grid_ == original_b.grid_);
    }
    if (!X) return;

    assert(X->get_num_cols() == B.get_num_cols() && X->get_num_rows() == A.get_num_cols());
    assert(geometric_col_degrees(*X) == geometric_col_degrees(B));
    assert(geometric_row_degrees(*X) == geometric_col_degrees(A));
    const auto a_degrees = geometric_col_degrees(A), b_degrees = geometric_col_degrees(B);
    for (int j = 0; j < X->get_num_cols(); ++j)
        for (int k : X->column(j))
            assert(Degree_traits<typename Matrix::degree_type>::smaller_equal(a_degrees[k], b_degrees[j]));
    auto product = A * *X;
    for (int j = 0; j < B.get_num_cols(); ++j) assert(product.get_col(j) == B.get_col(j));
}

void bigraded_solver_without_matrix_copies() {
    struct MoveOnlyMatrix : GradedSparseMatrix<r2degree, int, MoveOnlyMatrix> {
        using Base = GradedSparseMatrix<r2degree, int, MoveOnlyMatrix>;
        using Base::Base;
        MoveOnlyMatrix(const MoveOnlyMatrix&) = delete;
        MoveOnlyMatrix(MoveOnlyMatrix&&) = default;
    };
    MoveOnlyMatrix A(2, 2, {{0,1}, {1}}, {{0,1}, {1,0}}, {{0,0}, {0,0}});
    MoveOnlyMatrix B(1, 2, {{0}}, {{1,1}}, A.row_degrees);
    auto X = solve_bigraded_linear_system(std::move(A), std::move(B));
    assert(X && X->get_col(0) == vec<int>({0,1}));
    assert(X->col_degrees == vec<r2degree>({{1,1}}));
    assert(X->row_degrees == vec<r2degree>({{0,1}, {1,0}}));
}

template<class Matrix>
void graded_checks() {
    using D = typename Matrix::degree_type;
    const vec<D> rows{{0,0}, {0,0}};
    // Late columns precede their early alternatives in the original basis.
    // The graded reduction must preserve that basis and the RHS order.
    auto A = matrix<Matrix>({{0,1}, {0,1}, {1}, {}},
        {{2,2}, {1,1}, {0,1}, {0,0}}, {{1,0}, {0,0}});
    auto B = matrix<Matrix>({{0}, {0,1}, {0}, {}},
        {{1,1}, {2,2}, {1,1}, {0,0}}, {{1,0}, {0,0}});
    if constexpr (matrix_grid_backed_v<Matrix>) B.include_real_degrees({{-1,3}});
    check(A, B, true);
    auto spanning = matrix<Matrix>({{0,1}, {1}}, {{1,0}, {0,1}}, rows);
    check(spanning, matrix<Matrix>({{0}}, {{1,1}}, rows), true);
    check(spanning, matrix<Matrix>({{0}}, {{1,0}}, rows), false);
    check(spanning, matrix<Matrix>({{0}}, {{0,1}}, rows), false);
    // The low-y column initially owns pivot 1. The later low-x column
    // displaces it, so its queued revisit at the join must reveal pivot 0.
    auto displaced = matrix<Matrix>({{0,1}, {1}}, {{0,1}, {1,0}}, rows);
    check(displaced, matrix<Matrix>({{0}, {1}, {0}},
                                  {{1,1}, {1,0}, {2,1}}, rows), true);
    // B columns must never become reducers: equal RHS images do not solve
    // either RHS if their common image lies outside im(A).
    check(matrix<Matrix>({{0}}, {{0,0}}, rows),
          matrix<Matrix>({{1}, {1}}, {{0,0}, {0,0}}, rows), false);
    check(spanning, matrix<Matrix>({}, {}, rows), true);
    auto empty = matrix<Matrix>({}, {}, rows);
    check(empty, matrix<Matrix>({{}, {}}, {{0,0}, {1,1}}, rows), true);
    check(empty, matrix<Matrix>({{0}}, {{1,1}}, rows), false);
    check(matrix<Matrix>({{}, {}}, {{1,0}, {0,1}}, {}),
          matrix<Matrix>({{}}, {{0,0}}, {}), true);

    // Independent oracle: enumerate every F2 combination of three columns.
    // Include repeated, nonadjacent RHS degrees in the shared traversal.
    const vec<D> degrees{{1,0}, {0,1}, {1,1}};
    for (int mask = 0; mask < 64; ++mask) {
        array<int> columns(3);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 2; ++i)
                if (mask & (1 << (2*j+i))) columns[j].push_back(i);
        auto source = matrix<Matrix>(columns, degrees, rows);
        for (const D& degree : vec<D>{{0,0}, {1,0}, {0,1}, {1,1}}) {
            for (int rhs = 0; rhs < 4; ++rhs) {
                bool solvable = false;
                for (int subset = 0; subset < 8; ++subset) {
                    bool admissible = true;
                    int image = 0;
                    for (int j = 0; j < 3; ++j) if (subset & (1 << j)) {
                        admissible &= Degree_traits<D>::smaller_equal(degrees[j], degree);
                        image ^= (mask >> (2*j)) & 3;
                    }
                    solvable |= admissible && image == rhs;
                }
                vec<int> target;
                for (int i = 0; i < 2; ++i) if (rhs & (1 << i)) target.push_back(i);
                check(source, matrix<Matrix>({target, {}, target},
                    {degree, {2,2}, degree}, rows), solvable);
            }
        }
    }
}

template<class Matrix>
void randomized_checks(std::size_t dimensions) {
    using D = typename Matrix::degree_type;
    D zero{};
    std::mt19937 random(20261009u + static_cast<unsigned>(dimensions));
    for (int sample = 0; sample < 200; ++sample) {
        const int a_count = random()%7, b_count = random()%7, row_count = random()%5;
        D bottom = zero;
        for (auto& coordinate : bottom) coordinate = -2;
        vec<D> rows(row_count, bottom), a_degrees(a_count, zero), b_degrees(b_count, zero);
        for (auto* degrees : {&a_degrees, &b_degrees})
            for (auto& degree : *degrees)
                for (auto& coordinate : degree) coordinate = int(random()%4)-1;
        vec<int> a_bits(a_count), b_bits(b_count);
        for (auto& bits : a_bits) bits = random() % (1 << row_count);
        bool solvable = true;
        for (int j = 0; j < b_count; ++j) {
            // Half the systems are constructed to be solvable; the rest also
            // exercise unsolvable RHS columns and incomparable helper grades.
            if (sample%2) b_bits[j] = random() % (1 << row_count);
            else for (int k = 0; k < a_count; ++k)
                if ((random()&1) && Degree_traits<D>::smaller_equal(a_degrees[k], b_degrees[j]))
                    b_bits[j] ^= a_bits[k];
            bool in_span = false;
            for (int subset = 0; subset < (1 << a_count); ++subset) {
                int image = 0;
                bool admissible = true;
                for (int k = 0; k < a_count; ++k) if (subset & (1 << k)) {
                    image ^= a_bits[k];
                    admissible &= Degree_traits<D>::smaller_equal(a_degrees[k], b_degrees[j]);
                }
                in_span |= admissible && image == b_bits[j];
            }
            solvable &= in_span;
        }
        auto columns = [&](const vec<int>& bits) {
            array<int> result(bits.size());
            for (std::size_t j = 0; j < bits.size(); ++j)
                for (int i = 0; i < row_count; ++i)
                    if (bits[j] & (1 << i)) result[j].push_back(i);
            return result;
        };
        check(matrix<Matrix>(columns(a_bits), a_degrees, rows),
              matrix<Matrix>(columns(b_bits), b_degrees, rows), solvable);
    }
}

void solvability_without_output_matrix() {
    struct NoOutputMatrix : GradedSparseMatrix<r2degree, int, NoOutputMatrix> {
        using Base = GradedSparseMatrix<r2degree, int, NoOutputMatrix>;
        using Base::Base;
        // The boolean solver must compile without a solution constructor.
        NoOutputMatrix(int, int) = delete;
    };
    const NoOutputMatrix A(2, 2, {{0,1}, {1}}, {{0,1}, {1,0}}, {{0,0}, {0,0}});
    const NoOutputMatrix B(1, 2, {{0}}, {{1,1}}, A.row_degrees);
    assert((solve_graded_linear_system<NoOutputMatrix, false>(A, B)));
    assert(graded_linear_system_is_solvable(A, B));
    const NoOutputMatrix too_early(1, 2, {{0}}, {{0,1}}, A.row_degrees);
    assert(!graded_linear_system_is_solvable(A, too_early));
    const NoOutputMatrix wrong_rows(1, 2, {{0}}, {{1,1}}, {{-1,0}, {0,0}});
    bool rejected = false;
    try { graded_linear_system_is_solvable(A, wrong_rows); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

void upset_solvability_checks() {
    using Matrix = UpsetGradedSparseMatrix<double, int>;
    using D = typename Matrix::degree_type;
    const vec<D> rows(2, D{{0,0}});
    const auto A = matrix<Matrix>({{0}, {1}}, {D{{1,0}}, D{{0,1}}}, rows);
    assert(graded_linear_system_is_solvable(A, matrix<Matrix>({{0,1}}, {D{{1,1}}}, rows)));
    assert(!graded_linear_system_is_solvable(A, matrix<Matrix>({{0,1}}, {D{{1,0}}}, rows)));
}

void reduction_stops_after_targets() {
    struct CountingMatrix : SparseMatrix<int> {
        using SparseMatrix<int>::SparseMatrix;
        int additions = 0;
        void col_op(int source, int target) {
            ++additions;
            SparseMatrix<int>::col_op(source, target);
        }
    } helper(3, 1, array<int>{{0}, {0}, {0}});
    // Column 0 is an early A generator, 1 is the only RHS, and 2 is a
    // late A duplicate which a full kernel computation would also reduce.
    SparseMatrix<int> slave(3, 2, array<int>{{0}, {}, {1}});
    struct View {
        vec<std::array<int, 2>> z2_col_degrees{{0,0}, {0,0}, {1,1}};
        vec<int> x_grid{0,1}, y_grid{0,1};
        int get_num_cols() const { return 3; }
    } view;
    Grid_scheduler<int> scheduler(view);
    vec<std::priority_queue<int, vec<int>, std::greater<int>>> queues(2);
    std::map<int, int> pivots;
    int visited = 0;
    visit_bigraded_columns(scheduler, queues, view.z2_col_degrees,
        [&](int column, auto, auto notify) {
            ++visited;
            reduce_graded_column(helper, slave, pivots, column, notify, true, column != 1);
            if (column == 1) {
                assert(helper.is_zero(column) && slave.get_col(column) == vec<int>{0});
                return false;
            }
            return true;
        });
    assert(visited == 2 && helper.additions == 1);
    assert(pivots.size() == 1 && pivots.at(0) == 0);
    assert(helper.get_col(2) == vec<int>{0} && slave.get_col(2) == vec<int>{1});
    // Revisiting a pivot owner must not schedule that column itself.
    int notifications = 0;
    reduce_graded_column(helper, slave, pivots, 0, [&](int) { ++notifications; });
    assert(notifications == 0 && helper.additions == 1);
}

int main() {
    bigraded_solver_without_matrix_copies();
    solvability_without_output_matrix();
    upset_solvability_checks();
    reduction_stops_after_targets();
    // The existing batched helper must initialize the caller's solution matrix.
    SparseMatrix<int> helper(2, 2, array<int>{{0,1}, {1}}), rhs(2, 2, array<int>{{0}, {1}}), solution;
    assert(helper.solve_col_reduction(rhs, std::move(solution), true));
    assert(solution.get_num_cols() == 2 && solution.get_num_rows() == 2);
    assert(solution.data == array<int>({{0,1}, {1}}) && rhs.is_zero());
    graded_checks<R2GradedSparseMatrix<int>>();
    // The solver reuses the reduction primitives, never graded_kernel().
    static_assert(!has_matrix_graded_kernel<NoKernelMatrix<2>>::value);
    graded_checks<NoKernelMatrix<2>>();
    graded_checks<CoordGradedSparseMatrix<double, 2, int>>();
    graded_checks<GridGradedSparseMatrix<double, 2, int>>();
    randomized_checks<R2GradedSparseMatrix<int>>(2);
    randomized_checks<NoKernelMatrix<2>>(2);
    randomized_checks<NoKernelMatrix<3>>(3);
    randomized_checks<NoKernelMatrix<4>>(4);
    randomized_checks<CoordGradedSparseMatrix<double, 0, int>>(0);
    randomized_checks<CoordGradedSparseMatrix<double, 1, int>>(1);
    randomized_checks<CoordGradedSparseMatrix<double, 2, int>>(2);
    randomized_checks<CoordGradedSparseMatrix<double, 3, int>>(3);
    randomized_checks<CoordGradedSparseMatrix<double, 4, int>>(4);
    randomized_checks<GridGradedSparseMatrix<double, 0, int>>(0);
    randomized_checks<GridGradedSparseMatrix<double, 1, int>>(1);
    randomized_checks<GridGradedSparseMatrix<double, 2, int>>(2);
    randomized_checks<GridGradedSparseMatrix<double, 3, int>>(3);
    randomized_checks<GridGradedSparseMatrix<double, 4, int>>(4);
}
