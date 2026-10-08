/** @file z2graded_matrix.hpp @brief Sparse matrices graded by the product poset Z^2. */
#pragma once

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {

using z2degree = CoordinateDegree<long long, 2>;

template <typename index, typename MatrixBase = SparseMatrix<index>>
struct Z2GradedSparseMatrix
    : CoordinateGradedSparseMatrix<long long, 2, index, Z2GradedSparseMatrix<index, MatrixBase>, MatrixBase> {
    using Base = CoordinateGradedSparseMatrix<long long, 2, index, Z2GradedSparseMatrix<index, MatrixBase>, MatrixBase>;
    using Base::Base;
    Z2GradedSparseMatrix() = default;
    explicit Z2GradedSparseMatrix(MatrixBase&& other) : Base(std::move(other)) {}
};

} // namespace graded_linalg
