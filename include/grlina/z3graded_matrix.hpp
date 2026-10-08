/** @file z3graded_matrix.hpp @brief Sparse matrices graded by the product poset Z^3. */
#pragma once

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {

using z3degree = CoordinateDegree<long long, 3>;

template <typename index, typename MatrixBase = SparseMatrix<index>>
struct Z3GradedSparseMatrix
    : CoordinateGradedSparseMatrix<long long, 3, index, Z3GradedSparseMatrix<index, MatrixBase>, MatrixBase> {
    using Base = CoordinateGradedSparseMatrix<long long, 3, index, Z3GradedSparseMatrix<index, MatrixBase>, MatrixBase>;
    using Base::Base;
    Z3GradedSparseMatrix() = default;
    explicit Z3GradedSparseMatrix(MatrixBase&& other) : Base(std::move(other)) {}
};

} // namespace graded_linalg
