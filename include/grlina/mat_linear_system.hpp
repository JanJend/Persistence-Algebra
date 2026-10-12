/** Graded linear systems over the sparse matrix coefficient field F2. */
#pragma once
#include <algorithm>
#include <array>
#include <numeric>
#include <map>
#include <optional>
#include <grlina/mat_graded.hpp>
#include <grlina/mat_kernel.hpp>
#include <grlina/mat_grid.hpp>

namespace graded_linalg {

/** Test solvability without computing coefficients. Right-hand sides at the
 * same degree share one reduction; neither input basis is reordered.
 */
template <typename Matrix>
bool graded_linear_system_is_solvable(const Matrix& A, const Matrix& B) {
    using index = typename Matrix::index_type;
    using D = matrix_geometry_degree_t<Matrix>;

    GRLINA_DEBUG_CHECK(if (!same_geometric_rows(A, B))
        throw std::invalid_argument("Graded linear system has different target bases"));
    auto less = Degree_traits<D>::lex_lambda();
    std::map<D, array<index>, decltype(less)> by_degree(less);
    for (index j = 0; j < B.get_num_cols(); ++j)
        by_degree[geometric_degree(B, B.col_degrees[j])].push_back(B.get_col(j));
    for (const auto& [degree, columns] : by_degree) {
        array<index> available;
        for (index k = 0; k < A.get_num_cols(); ++k)
            if (Degree_traits<D>::smaller_equal(geometric_degree(A, A.col_degrees[k]), degree))
                available.push_back(A.get_col(k));
        SparseMatrix<index> local(static_cast<index>(available.size()), A.get_num_rows(), available);
        SparseMatrix<index> rhs(static_cast<index>(columns.size()), B.get_num_rows(), columns);
        if (!local.solve_col_reduction(rhs)) return false;
    }
    return true;
}


// Reparameterization with an arbitrary left block: maintain A's helper basis
// with the kernel's queued reduction, and reduce only B as target columns.
template<class Matrix>
bool solve_bigraded_slice(const Matrix& A, const Matrix& B,
                         const vec<matrix_geometry_degree_t<Matrix>>& a_degrees,
                         const vec<matrix_geometry_degree_t<Matrix>>& b_degrees,
                         const vec<typename Matrix::index_type>& sources,
                         const vec<typename Matrix::index_type>& targets,
                         std::size_t frozen_axes, Matrix& X) {
    using index = typename Matrix::index_type;
    using Scalar = typename matrix_geometry_degree_t<Matrix>::value_type;
    const index a_count = static_cast<index>(sources.size());
    const index count = a_count + static_cast<index>(targets.size());
    const std::size_t dimension = b_degrees[targets.front()].size();
    vec<std::array<Scalar, 2>> projected(count);
    for (index i = 0; i < count; ++i) {
        const auto& degree = i < a_count ? a_degrees[sources[i]] : b_degrees[targets[i-a_count]];
        for (std::size_t axis = 0; axis < 2; ++axis)
            if (frozen_axes + axis < dimension) projected[i][axis] = degree[frozen_axes + axis];
    }
    vec<index> order(count);
    std::iota(order.begin(), order.end(), index{0});
    std::stable_sort(order.begin(), order.end(), [&](index left, index right) {
        return projected[left][1] < projected[right][1] ||
              (projected[left][1] == projected[right][1] && projected[left][0] < projected[right][0]);
    });
    std::array<vec<Scalar>, 2> axes;
    for (const auto& degree : projected)
        for (std::size_t axis = 0; axis < 2; ++axis) axes[axis].push_back(degree[axis]);
    for (auto& axis : axes) {
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    }
    vec<std::array<index, 2>> grades(count);
    array<index> columns(count), operations(count);
    for (index i = 0; i < count; ++i) {
        const index original = order[i];
        for (std::size_t axis = 0; axis < 2; ++axis)
            grades[i][axis] = static_cast<index>(std::lower_bound(
                axes[axis].begin(), axes[axis].end(), projected[original][axis]) - axes[axis].begin());
        columns[i] = original < a_count ? A.get_col(sources[original]) : B.get_col(targets[original-a_count]);
        // A's slave columns start as identity coordinates in the original A
        // basis. B's start empty, exactly as in MPfree's reparameterize.
        if (original < a_count) operations[i].push_back(sources[original]);
    }
    SparseMatrix<index> helper(count, A.get_num_rows(), std::move(columns));
    SparseMatrix<index> slave(count, A.get_num_cols(), std::move(operations));
    struct GridView {
        index count;
        const vec<std::array<index, 2>>& z2_col_degrees;
        const vec<Scalar>& x_grid;
        const vec<Scalar>& y_grid;
        index get_num_cols() const { return count; }
    } view{count, grades, axes[0], axes[1]};
    Grid_scheduler<index> scheduler(view);
    vec<std::priority_queue<index, vec<index>, std::greater<index>>> queues(axes[1].size());
    std::map<index, index> pivots;
    std::size_t remaining = targets.size();
    visit_bigraded_columns(scheduler, queues, grades, [&](index column, auto, auto notify) {
        const index original = order[column];
        const bool target = original >= a_count;
        reduce_graded_column(helper, slave, pivots, column, notify, true, !target);
        if (target) {
            if (!helper.is_zero(column)) return false;
            X.set_col(targets[original-a_count], slave.get_col(column));
            slave.clear_col(column);
            // No A-only kernel generators are built, and no later A grades
            // are processed after the last requested RHS has been solved.
            return --remaining != 0;
        }
        if (helper.is_zero(column)) slave.clear_col(column);
        return true;
    });
    return remaining == 0;
}


/** Solve A X = B in the original bases, with degree-admissible coefficients.
 * Uses the same column reduction and queued grade traversal as graded_kernel,
 * without computing a kernel. A columns update the helper; B columns never
 * become reducers, and their slave columns give X as soon as they reach zero.
 * Two-parameter systems share one traversal, with no per-degree fallback.
 * Other coordinate dimensions reuse the kernel's two-parameter restrictions
 * at the frozen prefixes requested by B, without enumerating a full kernel.
 * No solution is reported as nullopt. Input compatibility is a caller
 * precondition, checked automatically in diagnostic builds.
 */
template <typename Matrix>
std::optional<Matrix> solve_graded_linear_system(const Matrix& A, const Matrix& B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using index = typename Matrix::index_type;
    using D = matrix_geometry_degree_t<Matrix>;
    using Scalar = typename D::value_type;
    static_assert(has_degree_join_v<D>, "The coordinate graded solver requires Degree_traits::join(a, b)");

    GRLINA_DEBUG_CHECK(if (!same_geometric_rows(A, B))
        throw std::invalid_argument("Graded linear system has different target bases"));
    Matrix X = empty_matrix_like(A, B.get_num_cols(), A.get_num_cols());
    auto a_degrees = geometric_col_degrees(A);
    auto b_degrees = geometric_col_degrees(B);
    set_geometric_degrees(X, b_degrees, a_degrees);
    std::map<vec<Scalar>, vec<index>> by_prefix;
    std::size_t frozen_axes = 0;
    if (!b_degrees.empty() && b_degrees.front().size() > 2) frozen_axes = b_degrees.front().size()-2;
    for (index j = 0; j < B.get_num_cols(); ++j) {
        if (B.get_col(j).empty()) continue;
        vec<Scalar> prefix(b_degrees[j].begin(), b_degrees[j].begin()+frozen_axes);
        by_prefix[std::move(prefix)].push_back(j);
    }
    for (const auto& [prefix, targets] : by_prefix) {
        // Helpers outside the requested RHS box cannot contribute. The first
        // frozen coordinates are equal throughout this group; the remaining
        // coordinates vary during the shared two-parameter traversal.
        D required = b_degrees[targets.front()];
        for (index j : targets) required = Degree_traits<D>::join(required, b_degrees[j]);
        auto sources = columns_in_bigraded_slice<index>(a_degrees, required, required.size());
        if (!solve_bigraded_slice(A, B, a_degrees, b_degrees, sources, targets, frozen_axes, X))
            return std::nullopt;
    }
    X.compatibly_sorted = false;
    return X;
}

} // namespace graded_linalg
