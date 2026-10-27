// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright 2025 TU Graz. Author: Jan Jendrysiak.
/** @file matrix_algorithms.hpp
 * @brief Free algorithms for matrix families and column-space comparison.
 */
#pragma once

#include <grlina/matrices/graded.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/r2.hpp>
// Preserve the aggregate includes for existing callers of this header.
#include <grlina/graded_kernel.hpp>

namespace graded_linalg {

// Single-matrix operations are defined in their owning classes. These free
// functions act on matrix families or compare two matrices as peers.

/**
 * @brief Performs reduction on a vector of matrices
 *
 * @tparam COLUMN
 * @tparam index
 * @tparam DERIVED
 * @param matrices
 * @return vec<index>
 */
template<typename index, typename T>
vec<index> general_reduction(vec< T > matrices) {
    // Ensure all matrices have the same dimensions
    GRLINA_ASSERT(!matrices.empty());
    index num_cols = matrices[0].get_num_cols();
    index num_rows = matrices[0].get_num_rows();
    for (const T& matrix : matrices) {
        GRLINA_ASSERT(matrix.get_num_cols() == num_cols);
        GRLINA_ASSERT(matrix.get_num_rows() == num_rows);
    }

    vec<index> non_zero_indices;

    for (index j = 0; j < matrices.size(); ++j) {
        T& matrix = matrices[j];
        auto [col, row] = matrix.last_entry();
        if (row != -1) {
            for (index k = j; k < matrices.size(); k++) {
                if (j != k && matrices[k].is_nonzero_entry(col, row)) {
                    matrix.add_matrix_to(matrices[k]);
                }
            }
            non_zero_indices.push_back(j);
        }
    }

    return non_zero_indices;
}

/**
 * @brief Treats all matrices in N_map[all_blocks] as stacked on top and column reduces them diagonally
 * wrt. the parts given by blocks to reduce on the columns from "support".
 *
 * @tparam index
 * @tparam DERIVED
 * @param N_map
 * @param blocks_to_reduce
 * @param all_blocks
 * @param support
 * @param zero_cols
 */
template <typename index, typename DERIVED>
bitset simultaneous_column_reduction(std::unordered_map<index, DERIVED>& N_map,
        vec<index>& blocks_to_reduce, vec<index>& all_blocks, bitset& support){


    index num_cols = N_map[all_blocks[0]].get_num_cols();
    GRLINA_ASSERT(support.size() == num_cols);
    bitset non_zero_cols = bitset(num_cols, false);

    for(index col = support.find_first(); col != bitset::npos; col = support.find_next(col)){
        for(index i = blocks_to_reduce.size()-1; i >= 0 ; i--){
            index b = blocks_to_reduce[i];
            DERIVED& N_b = N_map[b];
            index p = N_b.col_last(col);
            if(p >= 0){
                N_b.pivots[p] = col;
                for(index col_target = support.find_first(); col_target != bitset::npos; col_target = support.find_next(col_target)){
                    if(col_target != col && N_b.is_nonzero_entry(col_target, p)){
                        for(index j = 0; j < all_blocks.size(); j++){
                            N_map[all_blocks[j]].col_op(col, col_target);
                        }
                    }
                }
                non_zero_cols.set(col);
                break;
            }
        }
    }
    return non_zero_cols;
}

/**
 * @brief Treats all matrices in N_map[all_blocks] as stacked on top and column reduces them diagonally
 * wrt. the parts given by blocks to reduce on the columns from "support".
 *
 * @tparam index
 * @tparam DERIVED
 * @param N_map
 * @param blocks_to_reduce
 * @param all_blocks
 * @param support
 * @param zero_cols
 */
template <typename index, typename DERIVED>
bitset simultaneous_column_reduction_full_support(std::unordered_map<index, DERIVED>& N_map,
        vec<index>& blocks_to_reduce, vec<index>& all_blocks){

    index num_cols = N_map[all_blocks[0]].get_num_cols();
    bitset non_zero_cols = bitset(num_cols, false);

    for(index col = 0; col < num_cols; col++){
        for(index i = blocks_to_reduce.size()-1; i >= 0 ; i--){
            index b = blocks_to_reduce[i];
            DERIVED& N_b = N_map[b];
            index p = N_b.col_last(col);
            if(p >= 0){
                N_b.pivots[p] = col;
                for(index col_target = 0; col_target < num_cols; col_target ++){
                    if(col_target != col && N_b.is_nonzero_entry(col_target, p)){
                        for(index j = 0; j < all_blocks.size(); j++){
                            N_map[all_blocks[j]].col_op(col, col_target);
                        }
                    }
                }
                non_zero_cols.set(col);
                break;
            }
        }
    }
    return non_zero_cols;
}

/**
 * @brief Pushes all non-zero columns after support to the left.
 *
 * @param N_map
 * @param all_blocks
 * @param support
 * @param non_zero_cols
 */
template <typename index, typename DERIVED>
void simultaneous_align(std::unordered_map<index, DERIVED>& N_map, vec<index>& all_blocks, bitset& support, bitset& non_zero_cols){
    index num_cols = N_map[all_blocks[0]].num_cols;
    auto non_zero = non_zero_cols.find_first();
    for(index col = support.find_first(); non_zero != bitset::npos; col = support.find_next(col)){
        if(col == non_zero){
            non_zero = non_zero_cols.find_next(non_zero);
        } else {
            for(index b : all_blocks){
                N_map[b].swap_cols(col, non_zero);
                non_zero = non_zero_cols.find_next(non_zero);
            }
        }
    }
}



/**
 * @brief returns true if the column spaces of two matrices of equal dimension are isomorphic.
 *
 * @tparam DERIVED
 * @param A
 * @param B
 * @return true
 * @return false
 */
template <typename DERIVED>
bool compare_col_space(const DERIVED& A,const DERIVED& B){
    if(A.get_num_cols() != B.get_num_cols()){
        return false;
    }
    DERIVED copy_A = DERIVED(A);
    DERIVED copy_B = DERIVED(B);
    DERIVED A_inv = DERIVED(A.get_num_cols(), "Identity");
    DERIVED B_inv = DERIVED(B.get_num_cols(), "Identity");
    copy_A.column_gauss_jordan(A_inv);
    copy_B.column_gauss_jordan(B_inv);

    return copy_A.equals(copy_B);
}

/** Reduce A and replace B with a basis of (span(A) + span(B))/span(A) over F2.
 * A keeps its length, including zero results. Only coefficients are used;
 * matrices must share shapes and coordinates, and A and B must be distinct.
 */
template <typename Matrix>
void reduce_matrix_family_modulo(std::vector<Matrix>& A, std::vector<Matrix>& B) {
    using index = matrix_index_t<Matrix>;
    using position = std::pair<index, index>;
    std::map<position, Matrix*> reducers;
    auto reduce = [&](Matrix& matrix) {
        for (;;) {
            auto p = matrix.last_entry();
            if (p.second == -1) break;
            auto [it, inserted] = reducers.emplace(p, &matrix);
            if (inserted) break;
            it->second->add_matrix_to(matrix);
        }
    };
    for (auto& matrix : A) reduce(matrix);
    for (auto& matrix : B) reduce(matrix);
    // Compact only after reduction, while reducer pointers are no longer used.
    B.erase(std::remove_if(B.begin(), B.end(), [](Matrix& matrix) {
        return matrix.is_zero();
    }), B.end());
}



/**
 * @brief If all matrices N_map[blocks_to_reduce] contain exactly one row,
 * then this performs row reduction with these rows for entries after "support".
 *
 * @tparam index
 * @tparam DERIVED
 * @param N_map
 * @param blocks
 * @param support
 */
template<typename index, typename Storage>
void simultaneous_row_reduction(std::unordered_map<index, SparseMatrix<index, Storage>>& N_map, vec<index>& blocks, bitset& support){
    std::unordered_map<index, index> pivots;
    for(index b : blocks){
        vec<index>& row_b = N_map[b]._rows[0];
        for(index p : row_b){
            if(support.test(p)){
                pivots[p] = b;
                for(index c : blocks){
                    if( b == c){
                        continue;
                    } else {
                        vec<index>& row_c = N_map[c]._rows[0];
                        if(std::find(row_c.begin(), row_c.end(), p) != row_c.end()){
                            add_to(row_b, row_c);
                        }
                    }
                }
                break;
            }
        }
    }
}

/**
 * @brief If all matrices N_map[blocks_to_reduce] contain exactly one row,
 * then this performs row reduction with these rows for entries after "support".
 *
 * @tparam index
 * @tparam DERIVED
 * @param N_map
 * @param blocks
 * @param support
 */
template<typename index, typename Storage>
void simultaneous_row_reduction_on_submatrix(std::unordered_map<index, SparseMatrix<index, Storage>>& N_map, vec<index>& blocks, bitset& support, SparseMatrix<index, Storage>& A){
    std::unordered_map<index, index> pivots;
    for(index b : blocks){
        vec<index>& row_b = N_map[b]._rows[0];
        for(index p : row_b){
            if(support.test(p)){
                pivots[p] = b;
                for(index c : blocks){
                    if( b == c){
                        continue;
                    } else {
                        vec<index>& row_c = N_map[c]._rows[0];
                        if(std::find(row_c.begin(), row_c.end(), p) != row_c.end()){
                            Column_traits<vec<index>, index>::add_to(row_b, row_c);
                            A.fast_rev_row_op(b, c);
                        }
                    }
                }
                break;
            }
        }
    }
}

// Existing minor-extraction draft, retained for the matrix-operations audit.

/**
 * @brief Gets the Minor of M with respect to the row and column indices and saves the result as a DenseMatrix.
 *  Not tested yet!
 * @tparam index
 * @param M
 * @param col_indices
 * @param row_indices
 * @return DenseMatrix
 */
/**
template <typename index>
DenseMatrix restricted_dense_copy(const SparseMatrix<index>& M, vec<index> col_indices, vec<index> row_indices) {
    DenseMatrix<int> result(col_indices.size(), row_indices.size());
    for(index i = 0; i < col_indices.size(); i++){
        auto c = col_indices[i];
        auto it = std::lower_bound(M.data[c].begin(), M.data[c].end(), row_indices[0]);
        index j = 0;
        while(it != M.data[c].end() && j < row_indices.size()){
            auto r = row_indices[j];
            if(*it == r){
                result.data[i].set(j);
                j++;
                it++;
            } else if(*it < r){
                it++;
            } else {
                j++;
            }
        }
    }
}
*/

} // namespace graded_linalg
