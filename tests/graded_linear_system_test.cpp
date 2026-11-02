#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/graded_linear_system.hpp>
#include <grlina/MPFREE_clone.hpp>
#include <grlina/matrices/r2.hpp>
#include <grlina/matrices/coord.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/upset.hpp>
#include "test_degree_matrices.hpp"
#include "graded_linear_system_baselines.hpp"
#include <cassert>
#include <numeric>
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
void basis_and_joint_reduction_checks() {
    using D = typename Matrix::degree_type;
    // Equal-grade columns must keep their original order, and both permutation
    // directions must agree with the permuted coefficients and degrees.
    auto tied = matrix<Matrix>({{0}, {1}, {2}, {0,2}},
                              {{2,0}, {0,1}, {0,1}, {1,0}}, vec<D>(3, D{0,0}));
    using StoredDegree = typename decltype(tied.col_degrees)::value_type;
    auto permutation = tied.sort_columns_with_permutation(Degree_traits<StoredDegree>::colex_order);
    assert(permutation.new_to_old == vec<int>({3,0,1,2}));
    assert(permutation.old_to_new == vec<int>({1,2,3,0}));
    assert(tied.get_col(0) == vec<int>({0,2}) && tied.get_col(1) == vec<int>({0}));
    assert(tied.get_col(2) == vec<int>({1}) && tied.get_col(3) == vec<int>({2}));
    assert(geometric_col_degrees(tied) == vec<D>({{1,0},{2,0},{0,1},{0,1}}));
    auto identity_order = tied.sort_columns_with_permutation(Degree_traits<StoredDegree>::colex_order);
    assert(identity_order.new_to_old == vec<int>({0,1,2,3}));
    assert(identity_order.old_to_new == identity_order.new_to_old);
    // A projection can reorder surviving rows; filter before sorting their new
    // indices, including columns that lose all their entries.
    auto projected_rows = project_rows(matrix<Matrix>({{0,1,2}, {1,2}, {1}, {}},
        vec<D>(4, D{1,1}), {{0,0},{0,1},{1,0}}), vec<int>{2,0});
    assert(geometric_row_degrees(projected_rows) == vec<D>({{1,0},{0,0}}));
    assert(projected_rows.get_col(0) == vec<int>({0,1}));
    assert(projected_rows.get_col(1) == vec<int>{0});
    assert(projected_rows.is_zero(2) && projected_rows.is_zero(3));
    auto no_rows = project_rows(std::move(projected_rows), vec<int>{});
    assert(no_rows.get_num_rows() == 0 && no_rows.get_num_cols() == 4 && no_rows.is_zero());
    auto A = matrix<Matrix>({{0,2}, {0,1}}, {{1,0}, {1,1}}, {{1,0},{0,1},{0,0}});
    auto B = matrix<Matrix>({{1,2}, {0,2}, {}}, {{1,1},{1,0},{0,0}}, geometric_row_degrees(A));
    auto original = A;
    auto X = solve_graded_system_by_back_substitution(A, B);
    assert(X && X->get_col(0) == vec<int>({0,1}) && X->get_col(1) == vec<int>({0}));
    assert((A * *X).data == B.data && A.data == original.data);
    // A prepared basis owns its data and can solve different batches without
    // reordering the caller's matrices or retaining query state between calls.
    const PreparedGradedBasis<Matrix> prepared(A);
    A.clear_col(0);
    assert(prepared.solve(B)->data == X->data);
    A = original;
    auto identity_coordinates = prepared.solve(A);
    assert(identity_coordinates && identity_coordinates->get_col(0) == vec<int>{0});
    assert(identity_coordinates->get_col(1) == vec<int>{1});
    auto late = matrix<Matrix>({{1,2}}, {{1,0}}, geometric_row_degrees(A));
    assert(!solve_graded_system_by_back_substitution(A, late));
    assert(!prepared.solve(late) && prepared.solve(B)->data == X->data);
    auto wrong_rows = B;
    auto changed_rows = geometric_row_degrees(B);
    changed_rows[0][0] += 1;
    set_geometric_degrees(wrong_rows, geometric_col_degrees(B), changed_rows);
    bool wrong_basis_rejected = false;
    try { prepared.solve(wrong_rows); }
    catch (const std::invalid_argument&) { wrong_basis_rejected = true; }
    assert(wrong_basis_rejected);
    auto units = matrix<Matrix>({{1}, {0}}, {{0,1},{1,0}}, {{1,0},{0,1}});
    auto rhs = matrix<Matrix>({{0,1}}, {{1,1}}, geometric_row_degrees(units));
    assert(solve_graded_system_by_back_substitution(units, rhs)->get_col(0) == vec<int>({0,1}));
    auto dependent = matrix<Matrix>({{0},{0}}, {{0,0},{1,1}}, {{0,0}});
    bool rejected = false;
    try { solve_graded_system_by_back_substitution(dependent, matrix<Matrix>({{0}}, {{1,1}}, {{0,0}})); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && solve_graded_linear_system(dependent, matrix<Matrix>({{0}}, {{1,1}}, {{0,0}})));

    // Include incomparable births, equal births, displaced pivots, empty columns
    // and noninjective payloads. Compare projected spans to full kernel images.
    std::mt19937 random(129);
    for (int trial = 0; trial < 80; ++trial) {
        array<int> columns(5), payload(5);
        vec<D> births{{0,1},{1,0},{1,1},{0,1},{2,2}};
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 3; ++i) if (random()%2) columns[j].push_back(i);
            for (int i = 0; i < 2; ++i) if (random()%2) payload[j].push_back(i);
        }
        auto differential = matrix<Matrix>(columns, births, {{0,0},{0,0},{0,0}});
        auto P = matrix<Matrix>(payload, births, {{0,0},{0,0}});
        if constexpr (matrix_grid_backed_v<Matrix>) P.include_real_degrees({{-1,3}});
        auto copy = differential;
        auto K = copy.graded_kernel();
        if constexpr (std::is_same_v<Matrix, R2GradedSparseMatrix<int, matrix_storage_t<Matrix>>>) {
            auto legacy_input = differential;
            auto legacy = legacy_input.graded_kernel_MPFREE();
            assert(legacy.col_degrees == K.col_degrees && legacy.row_degrees == K.row_degrees);
            for (int j = 0; j < K.get_num_cols(); ++j) assert(legacy.get_col(j) == K.get_col(j));
        }
        auto expected = P * K;
        auto joint = bigraded_reduce(differential, P, true);
        assert(graded_linear_system_is_solvable(joint.kernel_image, expected));
        assert(graded_linear_system_is_solvable(expected, joint.kernel_image));
        assert(graded_linear_system_is_solvable(joint.image, differential));
        assert(graded_linear_system_is_solvable(differential, joint.image));
        copy = differential;
        auto projected = projected_bigraded_kernel(copy);
        assert(copy.data.empty());
        array<int> selected_columns(K.get_num_cols());
        for (int j = 0; j < K.get_num_cols(); ++j)
            for (int row : K.column(j)) {
                auto at = std::lower_bound(projected.coordinates.begin(), projected.coordinates.end(), row);
                if (at != projected.coordinates.end() && *at == row)
                    selected_columns[j].push_back(static_cast<int>(at-projected.coordinates.begin()));
            }
        auto selected = matrix<Matrix>(selected_columns, geometric_col_degrees(K), geometric_row_degrees(projected.basis));
        assert(graded_linear_system_is_solvable(projected.basis, selected));
        assert(graded_linear_system_is_solvable(selected, projected.basis));
        assert(solve_graded_system_by_back_substitution(projected.basis, selected));
        copy = differential;
        auto payload_copy = P;
        auto with_image = projected_bigraded_kernel_image(copy, payload_copy);
        assert(copy.data.empty() && payload_copy.data.empty());
        auto coefficients = solve_graded_system_by_back_substitution(with_image.basis, selected);
        assert(coefficients);
        assert((with_image.image * *coefficients).data == expected.data);
        assert(graded_linear_system_is_solvable(selected, with_image.basis));
    }
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
        auto moved = solve_graded_linear_system(std::move(movable_a), std::move(movable_b));
        if constexpr (std::is_same_v<Matrix, R2Matrix<>>)
            assert(movable_a.data.empty() && movable_b.data.empty());
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
    SparseMatrix<int> helper(3, 1, array<int>{{0}, {0}, {0}});
    // Column 0 is an early A generator, 1 is the only RHS, and 2 is a
    // late A duplicate which a full kernel computation would also reduce.
    SparseMatrix<int> slave(3, 2, array<int>{{0}, {}, {1}});
    vec<std::array<int, 2>> degrees{{0,0}, {0,0}, {1,1}};
    std::map<int, int> pivots;
    int visited = 0;
    visit_bigraded_columns(degrees, helper.get_num_cols(),
        [&](int column, auto, auto notify) {
            ++visited;
            reduce_graded_column(helper, slave, pivots, column, notify, true, column != 1);
            if (column == 1) {
                assert(helper.is_zero(column) && slave.get_col(column) == vec<int>{0});
                return false;
            }
            return true;
        });
    assert(visited == 2);
    assert(pivots.size() == 1 && pivots.at(0) == 0);
    assert(helper.get_col(2) == vec<int>{0} && slave.get_col(2) == vec<int>{1});
    // Revisiting a pivot owner must not schedule that column itself.
    int notifications = 0;
    reduce_graded_column(helper, slave, pivots, 0, [&](int) { ++notifications; });
    assert(notifications == 0);
}

template<class Storage>
void adaptive_column_reduction() {
    using Mat = SparseMatrix<int, Storage>;
    GradedReductionWorkspace<int> workspace;
    for (bool fill_in : {false, true}) for (bool finish_zero : {false, true})
    for (bool track : {false, true}) for (bool update : {false, true}) {
        array<int> columns;
        for (int row = 4095; row >= 4032; --row) columns.push_back({row});
        vec<int> dense(4096);
        std::iota(dense.begin(), dense.end(), 0);
        if (fill_in) columns[0] = dense;
        if (finish_zero) columns.emplace_back(dense.begin(), dense.begin() + 4032);
        const int column = columns.size();
        columns.push_back(fill_in ? vec<int>{4095} : dense);
        Mat actual(columns.size(), 4096, columns);
        Mat reference(columns.size(), 4096, columns);
        Mat operations(columns.size(), columns.size(), "Identity"), expected_ops = operations;
        std::map<int, int> pivots;
        for (int j = 0; j < column; ++j) pivots[reference.col_last(j)] = j;
        // A later owner must be displaced and notified when the column survives.
        if (!finish_zero) pivots[4031] = column + 1;
        auto expected_pivots = pivots;
        vec<int> notified, expected_notifications;
        auto pivot = reference.col_last(column);
        while (pivot != -1) {
            auto owner = expected_pivots.find(pivot);
            if (owner == expected_pivots.end() || owner->second >= column) break;
            reference.col_op(owner->second, column);
            if (track) expected_ops.col_op(owner->second, column);
            pivot = reference.col_last(column);
        }
        if (pivot != -1 && update) {
            auto owner = expected_pivots.find(pivot);
            if (owner != expected_pivots.end() && owner->second > column)
                expected_notifications.push_back(owner->second);
            expected_pivots[pivot] = column;
        }
        auto notify = [&](int owner) { notified.push_back(owner); };
        if (track) reduce_graded_column(actual, operations, pivots, column, notify, true, update, &workspace);
        else reduce_graded_column(actual, actual, pivots, column, notify, false, update, &workspace);
        assert(!workspace.packed.empty()); // Exercise packed reduction with reused storage.
        assert(actual.data == reference.data && operations.data == expected_ops.data);
        assert(pivots == expected_pivots && notified == expected_notifications);

        // Reuse the dense workspace for sparse reduction: stale bits and payload
        // entries must not affect the next column, including a zero result.
        Mat sparse(2, 2, array<int>{{0,1}, {1}}), payload(2, 2, "Identity");
        std::map<int, int> sparse_pivots{{1,0}};
        reduce_graded_column(sparse, payload, sparse_pivots, 1, [](int){}, true, false, &workspace);
        assert(workspace.packed.empty() && sparse.get_col(1) == vec<int>{0});
        assert(payload.get_col(1) == vec<int>({0,1}));
        sparse.set_col(1, {0,1});
        payload.set_col(1, {1});
        reduce_graded_column(sparse, payload, sparse_pivots, 1, [](int){}, true, false, &workspace);
        assert(sparse.is_zero(1) && payload.get_col(1) == vec<int>({0,1}));
    }
}

int main() {
    basis_and_joint_reduction_checks<R2Matrix<>>();
    basis_and_joint_reduction_checks<R2GradedSparseMatrix<int, CSCStorage<int>>>();
    basis_and_joint_reduction_checks<R2GradedSparseMatrix<int, CompactStorage<int>>>();
    basis_and_joint_reduction_checks<CoordGradedSparseMatrix<double, 2, int>>();
    basis_and_joint_reduction_checks<GridGradedSparseMatrix<double, 2, int>>();
    basis_and_joint_reduction_checks<GridGradedSparseMatrix<double, 2, int, CompactStorage<int>>>();
    adaptive_column_reduction<vec<vec<int>>>();
    adaptive_column_reduction<CSCStorage<int>>();
    adaptive_column_reduction<CompactStorage<int>>();
    // The matrix index type need not match an integer-literal column argument.
    vec<long long> dense(512);
    std::iota(dense.begin(), dense.end(), 0);
    SparseMatrix<long long> wide(4,512,array<long long>{{511},{510},{509},dense});
    std::map<long long,long long> pivots{{511,0},{510,1},{509,2}};
    reduce_graded_column(wide,wide,pivots,3,[](long long){},false,false);
    assert(wide.column(3).size()==509 && wide.col_last(3)==508);
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
