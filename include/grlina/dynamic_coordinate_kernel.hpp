/** @file dynamic_coordinate_kernel.hpp
 * @brief NEW: direct-coordinate kernel through an exact ordinal grid embedding.
 */
#pragma once
#include <grlina/dynamic_grid_matrix.hpp>

namespace graded_linalg {

template<class Scalar, class Index, class Storage>
auto DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>::graded_kernel() const
    -> DynamicCoordinateGradedSparseMatrix {
    // Grid compression preserves order and joins without scalar narrowing.
    auto editable = this->editable_copy();
    DynamicGridGradedSparseMatrix<Scalar, Index, vec<vec<Index>>> grid(editable);
    auto kernel = grid.graded_kernel();
    DynamicCoordinateGradedSparseMatrix result(kernel.get_num_cols(), kernel.get_num_rows(), this->parameter_count());
    result.assign_data(std::move(kernel.data));
    result.col_degrees = kernel.real_col_degrees();
    result.row_degrees = kernel.real_row_degrees();
    result.refresh_compatible_sorted();
    return result;
}

} // namespace graded_linalg
