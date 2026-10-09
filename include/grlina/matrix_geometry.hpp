/** @file matrix_geometry.hpp
 * @brief NEW: geometric degrees at module boundaries, independent of matrix storage.
 */
#pragma once
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <grlina/graded_matrix_io.hpp>

namespace graded_linalg {

template <typename Matrix, typename = void>
struct MatrixGeometryStorage {
    using degree_type = typename Matrix::degree_type;
    static constexpr bool grid_backed = false;
};

template <typename Matrix>
struct MatrixGeometryStorage<Matrix, std::void_t<decltype(Matrix::grid_backed), typename Matrix::real_degree_type>> {
    using degree_type = typename Matrix::real_degree_type;
    static constexpr bool grid_backed = Matrix::grid_backed;
};

// Keep degree aliases a substitution failure for non-matrix overload candidates.
template <typename Matrix, typename = void>
struct MatrixGeometry {};
template <typename Matrix>
struct MatrixGeometry<Matrix, std::void_t<typename Matrix::degree_type>> : MatrixGeometryStorage<Matrix> {};

template <typename Matrix>
using matrix_geometry_degree_t = typename MatrixGeometry<Matrix>::degree_type;
template <typename Matrix>
inline constexpr bool matrix_grid_backed_v = MatrixGeometry<Matrix>::grid_backed;

namespace detail {

template <typename Matrix>
Matrix empty_matrix_like(const Matrix& reference, typename Matrix::index_type columns,
                         typename Matrix::index_type rows) {
    Matrix result = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return reference.empty_like(columns, rows);
        else return GradedMatrixIO<Matrix>::make_matrix(columns, rows, GradedMatrixIO<Matrix>::matrix_context(reference));
    }();
    result.resize_data(static_cast<std::size_t>(columns));
    return result;
}

template <typename Matrix, typename Degree>
matrix_geometry_degree_t<Matrix> geometric_degree(const Matrix& matrix, const Degree& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_degree(degree);
    else return typename Matrix::degree_type(degree);
}

template <typename Matrix>
std::vector<matrix_geometry_degree_t<Matrix>> geometric_row_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_row_degrees();
    else return {matrix.row_degrees.begin(), matrix.row_degrees.end()};
}

template <typename Matrix>
std::vector<matrix_geometry_degree_t<Matrix>> geometric_col_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_col_degrees();
    else return {matrix.col_degrees.begin(), matrix.col_degrees.end()};
}

template <typename Matrix>
void set_geometric_degrees(Matrix& matrix,
                          const std::vector<matrix_geometry_degree_t<Matrix>>& columns,
                          const std::vector<matrix_geometry_degree_t<Matrix>>& rows) {
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto axes = matrix.grids;
        matrix.set_real_degrees(columns, rows);
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            axes[axis].insert(axes[axis].end(), matrix.grids[axis].begin(), matrix.grids[axis].end());
        matrix.reindex_grid(std::move(axes));
    }
    else {
        matrix.col_degrees = columns;
        matrix.row_degrees = rows;
    }
}

template <typename Matrix>
auto query_matrix_degree(const Matrix& matrix, const matrix_geometry_degree_t<Matrix>& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.query_degree(degree);
    else return degree;
}

template <typename Matrix>
bool same_geometric_rows(const Matrix& left, const Matrix& right) {
    if constexpr (matrix_grid_backed_v<Matrix>) return left.same_row_degrees(right);
    else return left.row_degrees == right.row_degrees;
}

template <typename Matrix>
void require_nonnegative_shift(const Matrix& matrix, const matrix_geometry_degree_t<Matrix>& amount,
                               const char* message) {
    if constexpr (GradedMatrixIO<Matrix>::runtime_dimension) {
        if (amount.size() != matrix.parameter_count()) throw std::invalid_argument("Shift has incompatible parameter count");
    }
    matrix_geometry_degree_t<Matrix> zero{};
    if constexpr (GradedMatrixIO<Matrix>::runtime_dimension) {
        zero = amount;
        for (auto& coordinate : zero) coordinate = {};
    }
    if (!Degree_traits<matrix_geometry_degree_t<Matrix>>::smaller_equal(zero, amount))
        throw std::invalid_argument(message);
}

} // namespace detail
} // namespace graded_linalg
