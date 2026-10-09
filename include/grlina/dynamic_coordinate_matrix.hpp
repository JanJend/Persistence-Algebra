/**
 * @file dynamic_coordinate_matrix.hpp
 * @brief NEW: runtime-dimensional direct coordinate grading.
 */
#pragma once
#include <grlina/dynamic_coordinate_matrix_base.hpp>

namespace graded_linalg {
template<class Scalar, class Index, class Storage> class DynamicGridGradedSparseMatrix;

template<class Scalar, class Index, class Storage = vec<vec<Index>>>
class DynamicCoordinateGradedSparseMatrix
    : public DynamicCoordinateMatrixBase<Scalar, Index,
          DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>, Storage> {
    using Core = DynamicCoordinateMatrixBase<Scalar, Index, DynamicCoordinateGradedSparseMatrix, Storage>;
public:
    using Core::Core;
    DynamicCoordinateGradedSparseMatrix() = default;
    DynamicCoordinateGradedSparseMatrix graded_kernel() const;
    template<class TargetStorage>
    DynamicCoordinateGradedSparseMatrix<Scalar, Index, TargetStorage> to_storage() const {
        return this->template copy_storage_as<DynamicCoordinateGradedSparseMatrix<Scalar, Index, TargetStorage>>();
    }
    DynamicCoordinateGradedSparseMatrix<Scalar, Index> editable_copy() const {
        return to_storage<vec<vec<Index>>>();
    }
    friend DynamicCoordinateGradedSparseMatrix operator*(
        const DynamicCoordinateGradedSparseMatrix& lhs, const DynamicCoordinateGradedSparseMatrix& rhs) {
        return Core::multiply_coordinates(lhs, rhs);
    }
};

template<class Scalar, class Index, class Storage>
struct is_graded_sparse_matrix<DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>, void>
    : std::true_type {};

} // namespace graded_linalg

#include <grlina/dynamic_coordinate_kernel.hpp>
