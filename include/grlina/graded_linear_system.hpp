/** Graded linear systems over the sparse matrix coefficient field F2. */
#pragma once
#include <map>
#include <optional>
#include <grlina/graded_matrix.hpp>

namespace graded_linalg {

/** Test solvability without computing coefficients. Right-hand sides at the
 * same degree share one reduction; neither input basis is reordered.
 */
template <typename Matrix>
bool graded_linear_system_is_solvable(const Matrix& A, const Matrix& B) {
    using index = typename Matrix::index_type;
    using D = typename Matrix::degree_type;
    GRLINA_DEBUG_CHECK(A.validate());
    GRLINA_DEBUG_CHECK(B.validate());
    GRLINA_DEBUG_CHECK(if (A.row_degrees != B.row_degrees)
        throw std::invalid_argument("Graded linear system has different target bases"));
    auto less = Degree_traits<D>::lex_lambda();
    std::map<D, array<index>, decltype(less)> by_degree(less);
    for (index j = 0; j < B.get_num_cols(); ++j)
        by_degree[B.col_degrees[j]].push_back(B.data[j]);
    for (const auto& [degree, columns] : by_degree) {
        array<index> available;
        for (index k = 0; k < A.get_num_cols(); ++k)
            if (Degree_traits<D>::smaller_equal(A.col_degrees[k], degree))
                available.push_back(A.data[k]);
        SparseMatrix<index> local(static_cast<index>(available.size()), A.get_num_rows(), available);
        SparseMatrix<index> rhs(static_cast<index>(columns.size()), B.get_num_rows(), columns);
        if (!local.solve_col_reduction(rhs)) return false;
    }
    return true;
}

/** Solve A X = B with degree-admissible coordinates, preserving A's basis.
 * No solution is reported as nullopt. Input compatibility is a caller
 * precondition, checked automatically in diagnostic builds.
 */
template <typename Matrix>
std::optional<Matrix> solve_graded_linear_system(const Matrix& A, const Matrix& B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using index = typename Matrix::index_type;
    using D = typename Matrix::degree_type;
    GRLINA_DEBUG_CHECK(A.validate());
    GRLINA_DEBUG_CHECK(B.validate());
    GRLINA_DEBUG_CHECK(if (A.row_degrees != B.row_degrees)
        throw std::invalid_argument("Graded linear system has different target bases"));
    Matrix X(B.get_num_cols(), A.get_num_cols());
    X.data.resize(B.get_num_cols());
    X.col_degrees = B.col_degrees;
    X.row_degrees = A.col_degrees;
    for (index j = 0; j < B.get_num_cols(); ++j) {
        vec<index> selected;
        array<index> columns;
        for (index k = 0; k < A.get_num_cols(); ++k) {
            if (Degree_traits<D>::smaller_equal(A.col_degrees[k], B.col_degrees[j])) {
                selected.push_back(k);
                columns.push_back(A.data[k]);
            }
        }
        SparseMatrix<index> local(static_cast<index>(columns.size()), A.get_num_rows(), columns);
        vec<index> rhs = B.data[j], solution;
        if (!local.solve_col_reduction(rhs, solution, true, true, true)) return std::nullopt;
        for (index k : solution) X.data[j].push_back(selected.at(k));
        std::sort(X.data[j].begin(), X.data[j].end());
    }
    X.refresh_compatible_sorted();
    return X;
}

} // namespace graded_linalg
