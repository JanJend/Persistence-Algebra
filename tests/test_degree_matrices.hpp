/** Custom degree matrices without a graded kernel, for capability tests. */
#pragma once
#include <grlina/deg_coordinate.hpp>
#include <grlina/mat_graded.hpp>

namespace graded_linalg {

template <std::size_t Dimension>
struct NoKernelMatrix : GradedSparseMatrix<CoordinateDegree<double, Dimension>, int,
                                          NoKernelMatrix<Dimension>> {
    using Base = GradedSparseMatrix<CoordinateDegree<double, Dimension>, int, NoKernelMatrix>;
    using Base::Base;
    NoKernelMatrix() = default;
    explicit NoKernelMatrix(SparseMatrix<int>&& matrix) : Base(std::move(matrix)) {}
};

} // namespace graded_linalg
