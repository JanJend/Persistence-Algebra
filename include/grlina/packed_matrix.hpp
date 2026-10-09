/** @file packed_matrix.hpp
 * @brief NEW: runtime matrices exchanged as packed CSC and flat degree buffers.
 */
#pragma once
#include <grlina/csc_matrix.hpp>
#include <grlina/dynamic_grid_matrix.hpp>
#include <grlina/matrix_geometry.hpp>

namespace graded_linalg {

/** Construct without changing the supplied bases. Grid matrices take stored
 * index degrees and the complete scalar axes; direct matrices take real degrees.
 * Buffers are owned values, so callers can move them into the matrix.
 */
template<class Matrix>
Matrix from_packed_csc(typename Matrix::index_type columns,
                      typename Matrix::index_type rows, std::size_t parameters,
                      vec<std::size_t> offsets, vec<typename Matrix::index_type> entries,
                      vec<typename Matrix::degree_type::value_type> column_coordinates,
                      vec<typename Matrix::degree_type::value_type> row_coordinates,
                      vec<vec<typename Matrix::scalar_type>> grids = {}) {
    using Index = typename Matrix::index_type;
    using Coordinate = typename Matrix::degree_type::value_type;
    Matrix result(columns, rows, parameters);
    CSCStorage<Index> packed(std::move(offsets), std::move(entries));
    if (packed.size() != static_cast<std::size_t>(columns))
        throw std::invalid_argument("CSC column count does not match the matrix");
    if constexpr (std::is_same_v<typename Matrix::storage_type, CSCStorage<Index>>) {
        result.data = std::move(packed);
    } else {
        array<Index> data;
        data.reserve(packed.size());
        for (auto column : packed) data.emplace_back(column.begin(), column.end());
        result.assign_data(std::move(data));
    }
    result.col_degrees = FlatDegreeTable<Coordinate>(columns, parameters, std::move(column_coordinates));
    result.row_degrees = FlatDegreeTable<Coordinate>(rows, parameters, std::move(row_coordinates));
    if constexpr (matrix_grid_backed_v<Matrix>) result.grids = std::move(grids);
    else if (!grids.empty()) throw std::invalid_argument("Direct coordinate matrices do not use grids");
    result.refresh_compatible_sorted();
    result.validate();
    return result;
}

/** An owning packed snapshot. Read data.offsets(), data.entries(), the flat
 * degree-table coordinates(), and (for grid matrices) grids from this object.
 */
template<class Matrix>
auto to_packed_csc(const Matrix& matrix) {
    return matrix.template to_storage<CSCStorage<typename Matrix::index_type>>();
}

} // namespace graded_linalg
