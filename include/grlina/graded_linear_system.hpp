/** Graded linear systems over the sparse matrix coefficient field F2. */
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <grlina/matrices/graded.hpp>
#include <grlina/graded_kernel.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

namespace detail {

// MPfree reduction on integer bidegrees, already sorted by (y, x).
// original_columns records indices in the original [A | B], even for a slice.
template<class Matrix, bool StoreOperations, class Index, class Storage>
bool reduce_bigraded_system(SparseMatrix<Index, Storage>& helper,
                            const vec<std::array<Index, 2>>& degrees,
                            const vec<Index>& original_columns, Index a_count,
                            std::size_t remaining,
                            std::conditional_t<StoreOperations, Matrix, std::nullptr_t>& X) {
    auto coefficients = [&] {
        if constexpr (StoreOperations) {
            SparseMatrix<Index, Storage> result(helper.get_num_cols(), a_count);
            result.resize_data(helper.get_num_cols());
            for (Index i = 0; i < helper.get_num_cols(); ++i)
                if (original_columns[i] < a_count) result.set_col(i, vec<Index>{original_columns[i]});
            return result;
        } else return nullptr;
    }();
    Grid_scheduler<Index> scheduler(degrees, helper.get_num_cols());
    using Queue = std::priority_queue<Index, vec<Index>, std::greater<Index>>;
    std::map<Index, Queue> queues;
    std::map<Index, Index> pivots;
    visit_bigraded_columns(scheduler, queues, degrees, [&](Index column, auto, auto notify) {
        const Index original = original_columns[column];
        const bool target = original >= a_count;
        if constexpr (StoreOperations)
            reduce_graded_column(helper, coefficients, pivots, column, notify, true, !target);
        else
            reduce_graded_column(helper, helper, pivots, column, notify, false, !target);
        if (target) {
            if (!helper.is_zero(column)) return false;
            if constexpr (StoreOperations) X.move_col_from(original - a_count, coefficients, column);
            return --remaining != 0;
        }
        if constexpr (StoreOperations)
            if (helper.is_zero(column)) coefficients.clear_col(column);
        return true;
    });
    return remaining == 0;
}

} // namespace detail

/** Two-parameter solver: align grids, then run MPfree directly on integer degrees.
 * Lvalues are copied; pass std::move(A), std::move(B) to consume their storage.
 * The result has the input matrix type and retains both original domain bases.
 * Coordinate inputs build a single common grid on the working [A | B] matrix.
 * Grid inputs reuse their integer degrees after merge_grids (a no-op if equal).
 */
template<class Matrix, bool StoreOperations = true>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_bigraded_linear_system(Matrix A, Matrix B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using Index = matrix_index_t<Matrix>;
    using GeometricDegree = typename Matrix::degree_type;
    using Coordinate = typename GeometricDegree::value_type;
    using StoredDegree = base_degree<Matrix>;
    static_assert(std::tuple_size<GeometricDegree>::value == 2, "Expected two-parameter matrices");

    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(A, B);
    GRLINA_DEBUG_CHECK(if (A.get_num_rows() != B.get_num_rows() || A.row_degrees != B.row_degrees)
        throw std::invalid_argument("Graded linear system has different target bases"));
    const Index a_count = A.get_num_cols();
    auto X = [&] {
        if constexpr (StoreOperations) {
            auto result = empty_matrix_like(A, B.get_num_cols(), a_count);
            result.row_degrees = A.col_degrees;
            result.col_degrees = B.col_degrees;
            return result;
        } else return nullptr;
    }();

    // Ignore zero RHS columns and A columns outside the targets' bounding box.
    vec<Index> targets;
    StoredDegree required{};
    for (Index j = 0; j < B.get_num_cols(); ++j) {
        if (B.column(j).empty()) continue;
        required = targets.empty() ? B.col_degrees[j] : Degree_traits<StoredDegree>::join(required, B.col_degrees[j]);
        targets.push_back(j);
    }
    if (!targets.empty()) {
        auto order = columns_in_bigraded_slice<Index>(A.col_degrees, required, 2);
        for (Index j : targets) order.push_back(a_count + j);
        auto degree = [&](Index original) -> const StoredDegree& {
            return original < a_count ? A.col_degrees[original] : B.col_degrees[original - a_count];
        };
        // Stable sorting keeps A before B at equal degrees.
        std::stable_sort(order.begin(), order.end(), [&](Index left, Index right) {
            return Degree_traits<StoredDegree>::colex_order(degree(left), degree(right));
        });
        vec<StoredDegree> degrees;
        degrees.reserve(order.size());
        for (Index original : order) degrees.push_back(degree(original));

        GridGradedSparseMatrix<Coordinate, 2, Index, matrix_reduction_storage_t<Matrix>> helper(static_cast<Index>(order.size()), A.get_num_rows());
        if constexpr (matrix_grid_backed_v<Matrix>) {
            helper.grid_ = std::move(A.grid_);
            helper.row_degrees = std::move(A.row_degrees);
            helper.col_degrees = std::move(degrees);
        } else {
            helper.set_real_degrees(degrees, A.row_degrees);
        }
        for (Index i = 0; i < helper.get_num_cols(); ++i) {
            const Index original = order[i];
            if (original < a_count) helper.move_col_from(i, A, original);
            else helper.move_col_from(i, B, original - a_count);
        }
        if (!detail::reduce_bigraded_system<Matrix, StoreOperations>(
                helper, helper.col_degrees, order, a_count, targets.size(), X)) {
            if constexpr (StoreOperations) return std::nullopt;
            else return false;
        }
    }
    if constexpr (StoreOperations) {
        X.compatibly_sorted = false;
        return X;
    } else return true;
}

// Reparameterization with an arbitrary left block: maintain A's helper basis
// with the kernel's queued reduction, and reduce only B as target columns.
template<class Matrix, bool StoreOperations = true>
bool solve_bigraded_slice(const Matrix& A, const Matrix& B,
                         const vec<typename Matrix::degree_type>& a_degrees,
                         const vec<typename Matrix::degree_type>& b_degrees,
                         const vec<matrix_index_t<Matrix>>& sources,
                         const vec<matrix_index_t<Matrix>>& targets,
                         std::size_t frozen_axes,
                         std::conditional_t<StoreOperations, Matrix, std::nullptr_t>& X) {
    using index = matrix_index_t<Matrix>;
    using Scalar = typename Matrix::degree_type::value_type;
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
    array<index> columns(count);
    for (index i = 0; i < count; ++i) {
        const index original = order[i];
        for (std::size_t axis = 0; axis < 2; ++axis)
            grades[i][axis] = static_cast<index>(std::lower_bound(
                axes[axis].begin(), axes[axis].end(), projected[original][axis]) - axes[axis].begin());
        columns[i] = original < a_count ? A.get_col(sources[original]) : B.get_col(targets[original-a_count]);
    }
    SparseMatrix<index> helper(count, A.get_num_rows());
    helper.assign_data(std::move(columns));
    for (auto& original : order)
        original = original < a_count ? sources[original] : A.get_num_cols() + targets[original - a_count];
    return detail::reduce_bigraded_system<Matrix, StoreOperations>(
        helper, grades, order, A.get_num_cols(), targets.size(), X);
}


/** Solve A X = B in the original bases, with degree-admissible coefficients.
 * Uses the same column reduction and queued grade traversal as graded_kernel,
 * without computing a kernel. A columns update the helper; B columns never
 * become reducers, and their tracked coefficients give X when they reach zero.
 * Two-parameter systems consume owned arguments; move inputs to avoid copying.
 * They dispatch directly to solve_bigraded_linear_system.
 * Other coordinate dimensions reuse the kernel's two-parameter restrictions
 * at the frozen prefixes requested by B, without enumerating a full kernel.
 * No solution is reported as nullopt. Input compatibility is a caller
 * precondition, checked automatically in diagnostic builds.
 * With StoreOperations=false, return only bool using the same traversal;
 * neither a coefficient matrix nor a solution matrix is constructed.
 */
template <typename Matrix, bool StoreOperations = true,
          std::enable_if_t<std::tuple_size<typename Matrix::degree_type>::value == 2, int> = 0>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_graded_linear_system(Matrix A, Matrix B) {
    return solve_bigraded_linear_system<Matrix, StoreOperations>(std::move(A), std::move(B));
}

template <typename Matrix, bool StoreOperations = true,
          std::enable_if_t<std::tuple_size<typename Matrix::degree_type>::value != 2, int> = 0>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_graded_linear_system(const Matrix& A, const Matrix& B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    using Scalar = typename D::value_type;
    static_assert(has_degree_join_v<D>, "The coordinate graded solver requires Degree_traits::join(a, b)");

    GRLINA_DEBUG_CHECK(if (!same_geometric_rows(A, B))
        throw std::invalid_argument("Graded linear system has different target bases"));
    auto X = [&] {
        if constexpr (StoreOperations) return empty_matrix_like(A, B.get_num_cols(), A.get_num_cols());
        else return nullptr;
    }();
    auto a_degrees = geometric_col_degrees(A);
    auto b_degrees = geometric_col_degrees(B);
    if constexpr (StoreOperations) set_geometric_degrees(X, b_degrees, a_degrees);
    std::map<vec<Scalar>, vec<index>> by_prefix;
    std::size_t frozen_axes = 0;
    if (!b_degrees.empty() && b_degrees.front().size() > 2) frozen_axes = b_degrees.front().size()-2;
    for (index j = 0; j < B.get_num_cols(); ++j) {
        if (B.column(j).empty()) continue;
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
        if (!solve_bigraded_slice<Matrix, StoreOperations>(
                A, B, a_degrees, b_degrees, sources, targets, frozen_axes, X)) {
            if constexpr (StoreOperations) return std::nullopt;
            else return false;
        }
    }
    if constexpr (StoreOperations) {
        X.compatibly_sorted = false;
        return X;
    } else return true;
}

namespace detail {
template <typename D> struct is_coordinate_solver_degree : std::false_type {};
template <typename Scalar, std::size_t Dimension>
struct is_coordinate_solver_degree<std::array<Scalar, Dimension>> : std::true_type {};
} // namespace detail

/** Test solvability without computing coefficients or reordering either basis.
 * Coordinate degrees use the solver's shared traversal without a slave matrix.
 * General degree posets retain the per-degree span test.
 */
template <typename Matrix>
bool graded_linear_system_is_solvable(const Matrix& A, const Matrix& B) {
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    if constexpr (detail::is_coordinate_solver_degree<D>::value) {
        return solve_graded_linear_system<Matrix, false>(A, B);
    } else {
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
}

} // namespace graded_linalg
