/** @file r4graded_matrix.hpp @brief Sparse matrices graded by the product poset R^4. */
#pragma once

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {

using r4degree = CoordinateDegree<double, 4>;

template <typename index>
struct R4GradedSparseMatrix
    : CoordinateGradedSparseMatrix<double, 4, index, R4GradedSparseMatrix<index>> {
    using Base = CoordinateGradedSparseMatrix<double, 4, index, R4GradedSparseMatrix<index>>;
    using Base::Base;
    R4GradedSparseMatrix() = default;
    explicit R4GradedSparseMatrix(SparseMatrix<index>&& other) : Base(std::move(other)) {}
};

} // namespace graded_linalg
