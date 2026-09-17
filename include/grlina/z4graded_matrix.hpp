/** @file z4graded_matrix.hpp @brief Sparse matrices graded by the product poset Z^4. */
#pragma once

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {

using z4degree = CoordinateDegree<long long, 4>;

template <typename index>
struct Z4GradedSparseMatrix
    : CoordinateGradedSparseMatrix<long long, 4, index, Z4GradedSparseMatrix<index>> {
    using Base = CoordinateGradedSparseMatrix<long long, 4, index, Z4GradedSparseMatrix<index>>;
    using Base::Base;
    Z4GradedSparseMatrix() = default;
    explicit Z4GradedSparseMatrix(SparseMatrix<index>&& other) : Base(std::move(other)) {}
};

} // namespace graded_linalg
