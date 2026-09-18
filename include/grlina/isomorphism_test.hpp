/** @file isomorphism_test.hpp
 * Exact isomorphism decisions for finitely presented persistence modules over F2.
 * No decomposition, field extension arithmetic, or random sampling is needed.
 */
#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <grlina/homomorphisms.hpp>

namespace graded_linalg {

enum class IsomorphismHomMethod { optimised, full_restriction };

namespace isomorphism_detail {

// The columns of space span a space of b-by-b matrices, flattened columnwise.
// Keep only an echelon basis: at most b*b columns, regardless of dim Hom.
template <typename I>
void extend_span(SparseMatrix<I>& space, vec<I> column) {
    std::sort(column.begin(), column.end());
    vec<I> unused;
    if (!space.solve_col_reduction(column, unused, true, false, false)) {
        space.data.push_back(std::move(column));
        space.compute_num_cols();
    }
}

template <typename I>
bool contains_invertible(SparseMatrix<I>& space, I b) {
    if (space.get_num_cols() == 0) return false;
    auto contains = [&](vec<I> column) {
        vec<I> unused;
        return space.solve_col_reduction(column, unused, true, false, false);
    };
    vec<I> identity;
    for (I i = 0; i < b; ++i) identity.push_back(i * b + i);
    if (contains(identity)) return true;
    if (b == 1) return false;

    if (b == 2) {
        // The other five elements of GL(2,F2); identity was tested above.
        for (const auto& entries : array<I>{{0, 2, 3}, {0, 1, 3},
                                           {1, 2}, {1, 2, 3}, {0, 1, 2}})
            if (contains(entries)) return true;
        return false;
    }

    // ponytail: exhaustive local-span search for larger repeated degrees.
    // Cost is 2^dim(space), dim(space)<=b*b; never exponential in the number
    // of degree blocks. A matrix-space nonsingularity solver can replace this
    // helper if large degree multiplicities become common. No size cutoff
    // returns a spurious negative answer.
    std::vector<bool> selected(space.data.size(), false);
    vec<I> combination;
    while (true) {
        std::size_t i = 0;
        while (i < selected.size() && selected[i]) {
            selected[i] = false;
            Column_traits<vec<I>, I>::add_to(space.data[i++], combination);
        }
        if (i == selected.size()) return false;
        selected[i] = true;
        Column_traits<vec<I>, I>::add_to(space.data[i], combination);
        array<I> columns(b);
        for (I entry : combination) columns[entry / b].push_back(entry % b);
        SparseMatrix<I> candidate(b, b, columns);
        if (candidate.is_invertible()) return true;
    }
}

template <typename D, typename I>
bool diagonal_blocks_pass(const SparseMatrix<I>& hom,
                         const vec<std::pair<I, I>>& positions,
                         const vec<D>& degrees) {
    vec<I> starts, sizes, block_of(degrees.size());
    std::vector<SparseMatrix<I>> spaces;
    for (I first = 0; first < static_cast<I>(degrees.size());) {
        I last = first + 1;
        while (last < static_cast<I>(degrees.size()) &&
               Degree_traits<D>::equals(degrees[first], degrees[last])) ++last;
        I b = last - first;
        if (b > std::numeric_limits<I>::max() / b)
            throw std::length_error("Isomorphism degree block exceeds the matrix index range");
        std::fill(block_of.begin() + first, block_of.begin() + last,
                  static_cast<I>(spaces.size()));
        starts.push_back(first);
        sizes.push_back(b);
        spaces.emplace_back(0, b * b);
        first = last;
    }

    // Packed Hom coordinates are (source generator, target generator).
    // Project all blocks in one pass through each Hom vector; do not expand
    // the vectors to full generator matrices or rescan them for every degree.
    vec<std::pair<I, I>> coordinates(positions.size(), {I(-1), I(0)});
    for (std::size_t v = 0; v < positions.size(); ++v) {
        const auto [column, row] = positions[v];
        I block = block_of[column];
        if (block == block_of[row])
            coordinates[v] = {block, (column - starts[block]) * sizes[block]
                                       + row - starts[block]};
    }
    array<I> projected(spaces.size());
    vec<I> touched;
    for (const auto& column : hom.data) {
        for (I v : column) {
            const auto [block, entry] = coordinates[v];
            if (block < 0) continue;
            if (projected[block].empty()) touched.push_back(block);
            projected[block].push_back(entry);
        }
        for (I block : touched) {
            extend_span(spaces[block], std::move(projected[block]));
            projected[block].clear();
        }
        touched.clear();
    }
    for (std::size_t block = 0; block < spaces.size(); ++block)
        if (!contains_invertible(spaces[block], sizes[block])) return false;
    return true;
}

} // namespace isomorphism_detail

/** Decide degree-preserving module isomorphism from presentation matrices.
 *
 * Copies are sorted and fully minimized unless assume_minimal=true. In that
 * case FULL minimality (including removal of redundant relations) is a trusted
 * precondition; cancelling equal-degree generator/relation pairs alone is not
 * sufficient. Inputs are never mutated. Exact degree equality is used.
 *
 * Usage: is_isomorphic(P, Q), or is_isomorphic(P, Q, true) for minimal inputs.
 * The default uses hom_space_optimised and its packed basis. The alternative
 * hom_space_full_restriction additionally needs the degree type's graded kernel.
 *
 * Correctness: matching minimal generator and relation degrees imply that a
 * Hom lift is an isomorphism iff all equal-degree generator blocks invert.
 * Each block may be tested INDEPENDENTLY: if every projected Hom space has a
 * unit, each block determinant is a nonzero polynomial in the Hom coordinates.
 * Their product is nonzero, giving an isomorphism over some finite extension K.
 * Restricting scalars gives M^[K:F2] ~= N^[K:F2]; Krull-Schmidt implies M ~= N.
 * This is a decision test; it does not assemble an isomorphism witness.
 *
 * After Hom, only projected spaces of dimension <= b*b are kept for a degree
 * with b generators. Singleton tests are linear and double-degree tests check
 * the six elements of GL(2,F2). Larger blocks use an exact exponential search
 * in their own projected space, not across combinations of degree blocks.
 */
template <typename Matrix>
bool is_isomorphic(Matrix A, Matrix B, bool assume_minimal = false,
                   IsomorphismHomMethod method = IsomorphismHomMethod::optimised,
                   bool info = false) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded presentation matrices");
    using I = typename Matrix::index_type;
    A.validate();
    B.validate();
    A.sort_compatibly();
    B.sort_compatibly();
    if (!assume_minimal) {
        A.minimize();
        B.minimize();
    }
    if (A.row_degrees != B.row_degrees || A.col_degrees != B.col_degrees) return false;
    if (A.data == B.data) return true; // Includes zero modules and free modules.

    A.compute_rows_forward();
    std::pair<SparseMatrix<I>, vec<std::pair<I, I>>> hom;
    switch (method) {
    case IsomorphismHomMethod::optimised:
        hom = hom_space_optimised(A, B, vec<I>{}, vec<I>{}, info);
        break;
    case IsomorphismHomMethod::full_restriction:
        if constexpr (has_matrix_graded_kernel<Matrix>::value)
            hom = hom_space_full_restriction(A, B, vec<I>{}, vec<I>{}, info);
        else
            throw std::logic_error("Full-restriction Hom requires a graded kernel");
        break;
    default:
        throw std::invalid_argument("Unknown isomorphism Hom method");
    }
    return isomorphism_detail::diagonal_blocks_pass(hom.first, hom.second, A.row_degrees);
}

} // namespace graded_linalg
