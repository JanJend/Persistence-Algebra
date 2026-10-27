/** Dense/sparse F2 matrix conversions. */
#pragma once

#include <grlina/matrices/dense.hpp>
#include <grlina/matrices/sparse.hpp>

namespace graded_linalg {

/**
 * @brief Converts a dense matrix to a sparse matrix
 *
 * @tparam index
 * @param matrix
 * @return SparseMatrix<index>
 */
template <typename index>
SparseMatrix<index> sparse_from_dense(DenseMatrix& matrix){
    SparseMatrix<index> result = SparseMatrix<index>(matrix.get_num_cols(), matrix.get_num_rows());
    for(index i = 0; i < matrix.get_num_cols(); i++){
        result.data.push_back(vec<index>());
        for(index j = 0; j < matrix.get_num_rows(); j++){
            if(matrix.data[i][j] == 1){
                result.data[i].push_back(j);
            }
        }
    }
    return result;
}

template <typename index>
DenseMatrix from_sparse (SparseMatrix<index>& M) {
    DenseMatrix result(M.get_num_cols(), M.get_num_rows());
    result.data = vec<bitset>(M.get_num_cols(), bitset(M.get_num_rows()));
    for(int i = 0; i < M.get_num_cols(); i++){
        for(index j : M.data[i]){
            result.data[i].set(j);
        }
    }
    return result;
}

} // namespace graded_linalg
