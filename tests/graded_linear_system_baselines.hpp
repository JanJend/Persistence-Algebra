/** Reference solvers for correctness comparisons and timing; not library code. */
#pragma once
#include <grlina/graded_linear_system.hpp>
#include <chrono>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/graded_kernel.hpp>

namespace graded_linalg::test {

// The original per-column implementation, retained as a benchmark baseline.
template<class Matrix>
std::optional<Matrix> column_by_column_solver(const Matrix& A, const Matrix& B) {
    using index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    Matrix X = empty_matrix_like(A, B.get_num_cols(), A.get_num_cols());
    set_geometric_degrees(X, geometric_col_degrees(B), geometric_col_degrees(A));
    for (index j = 0; j < B.get_num_cols(); ++j) {
        vec<index> selected;
        array<index> columns;
        for (index k = 0; k < A.get_num_cols(); ++k) {
            if (Degree_traits<Degree>::smaller_equal(geometric_degree(A, A.col_degrees[k]),
                                                     geometric_degree(B, B.col_degrees[j]))) {
                selected.push_back(k);
                columns.push_back(A.get_col(k));
            }
        }
        SparseMatrix<index> local(static_cast<index>(columns.size()), A.get_num_rows(), columns);
        vec<index> rhs = B.get_col(j), solution;
        if (!local.solve_col_reduction(rhs, solution, true, true, true)) return std::nullopt;
        vec<index> column;
        for (index k : solution) column.push_back(selected.at(k));
        std::sort(column.begin(), column.end());
        X.set_col(j, column);
    }
    X.compatibly_sorted = false;
    return X;
}

// The former full-kernel approach. kernel_ms optionally records the kernel
// call alone; callers also time construction and extraction for a fair solve.
template<class Matrix>
std::optional<Matrix> full_kernel_solver(const Matrix& A, const Matrix& B, double* kernel_ms = nullptr) {
    using index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    const index a_count = A.get_num_cols();
    auto a_degrees = geometric_col_degrees(A);
    auto b_degrees = geometric_col_degrees(B);
    Matrix augmented = empty_matrix_like(A, a_count + B.get_num_cols(), A.get_num_rows());
    auto degrees = a_degrees;
    degrees.insert(degrees.end(), b_degrees.begin(), b_degrees.end());
    set_geometric_degrees(augmented, degrees, geometric_row_degrees(A));
    for (index k = 0; k < a_count; ++k) augmented.set_col(k, A.get_col(k));
    for (index j = 0; j < B.get_num_cols(); ++j) augmented.set_col(a_count+j, B.get_col(j));
    const auto start = std::chrono::steady_clock::now();
    auto kernel = augmented.graded_kernel();
    if (kernel_ms) *kernel_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-start).count();
    Matrix X = empty_matrix_like(A, B.get_num_cols(), a_count);
    set_geometric_degrees(X, b_degrees, a_degrees);
    std::vector<bool> solved(B.get_num_cols(), false);
    for (index k = 0; k < kernel.get_num_cols(); ++k) {
        auto column = kernel.get_col(k);
        auto rhs = std::lower_bound(column.begin(), column.end(), a_count);
        if (rhs == column.end() || rhs+1 != column.end()) continue;
        const index j = *rhs-a_count;
        if (solved[j] || !Degree_traits<Degree>::smaller_equal(
                geometric_degree(kernel, kernel.col_degrees[k]), b_degrees[j])) continue;
        column.erase(rhs);
        X.set_col(j, column);
        solved[j] = true;
    }
    if (std::find(solved.begin(), solved.end(), false) != solved.end()) return std::nullopt;
    X.compatibly_sorted = false;
    return X;
}

} // namespace graded_linalg::test
