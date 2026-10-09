/** @file chain_basis_maps.hpp
 * @brief NEW: basis maps transported by the existing chain operations.
 */
#pragma once
#include <grlina/matrix_geometry.hpp>

namespace graded_linalg {
namespace detail {
template<class Matrix, class Degrees>
Matrix identity_on_degrees(const Matrix& reference, const Degrees& degrees) {
    using Index = typename Matrix::index_type;
    const auto count = static_cast<Index>(degrees.size());
    Matrix result = empty_matrix_like(reference, count, count);
    typename Matrix::storage_type columns;
    columns.reserve(degrees.size());
    for (Index i = 0; i < count; ++i) columns.push_back(vec<Index>{i});
    result.assign_data(std::move(columns));
    result.col_degrees = degrees;
    result.row_degrees = degrees;
    result.refresh_compatible_sorted();
    return result;
}
} // namespace detail

/** Initialize with identities on every chain group before passing to chain
 * sorting or minimization. The original bases remain fixed throughout.
 */
template<class Matrix>
struct ChainBasisMaps {
    vec<Matrix> to_original;   // current -> original
    vec<Matrix> from_original; // original -> current
};
} // namespace graded_linalg
