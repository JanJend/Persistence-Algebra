/** Matrix conversions, subspace enumeration and matrix-stream helpers. */
#pragma once
#include <grlina/mat_dense.hpp>
#include <grlina/mat_sparse.hpp>
#include <grlina/mat_r2.hpp>

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
vec<vec<SparseMatrix<index>>> all_sparse_proper_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){
    
        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}

template <typename index>
vec<vec<SparseMatrix<index>>> all_sparse_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){
    
        vec<DenseMatrix> i_spaces = all_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}


template <typename index>
void fill_up_subspaces (vec<vec<SparseMatrix<index>>>& subspaces, index k ){

    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<SparseMatrix<index>>());
        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            subspaces[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> sparse_seperated_grassmannians(index k){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k, vec<vec<SparseMatrix<index>>>());
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_seperated_grassmannians(vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> all_sparse_grassmannians(index k, index n){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k);
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_grassmannians (vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k, index n){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

/**
 * @brief Constructs graded matrices from an input stream.
 * 
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index, typename Storage = vec<vec<index>>>
std::vector<R2GradedSparseMatrix<index, Storage>> get_matrices_from_stream(std::ifstream& file_stream, bool lex_sort = false) {
    std::vector<R2GradedSparseMatrix<index, Storage>> matrices;

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index, Storage> matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));

        // Check if the stream is exhausted
        if (file_stream.eof()) {
            break;
        }
    }

    return matrices;
}

/**
 * @brief Constructs graded matrices from an input stream.
 * 
 * @param matrices vector to store the constructed matrices
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename Matrix, typename InputStream>
void construct_matrices_from_stream(std::vector<Matrix>& matrices, InputStream& file_stream, bool lex_sort = false) {

    while (file_stream) {
        // Construct a new matrix from the stream
        Matrix matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));

        // Skip empty lines
        std::string line;
        while (std::getline(file_stream, line)) {
            if (!line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
                file_stream.seekg(-static_cast<int>(line.length()) - 1, std::ios_base::cur);
                break;
            }
        }

        if (file_stream.eof()) {
            break;
        }
    }

}

/**
 * @brief Constructs matrices from stream with header and type annotations
 *
 * @param matrices vector to store the constructed matrices
 * @param file_stream input stream with format: header, count, then (type, matrix) pairs
 * @param lex_sort whether to sort lexicographically
 * @param matrix_types optional vector to store the type of each matrix (e.g., "cyclic")
 */
template <typename Matrix, typename InputStream>
void read_sccsum(
    std::vector<Matrix>& matrices,
    InputStream& file_stream, 
    bool lex_sort = false, 
    std::vector<std::string>* matrix_types = nullptr) {
    
    // Read and validate header
    std::string header;
    std::getline(file_stream, header);
    if (header != "scc2020sum") {
        throw std::runtime_error("Invalid header: expected 'scc2020sum', got '" + header + "'");
    }
    
    // Read matrix count
    size_t count;
    file_stream >> count;
    
    matrices.reserve(count);
    if (matrix_types) {
        matrix_types->reserve(count);
    }
    
    for (size_t i = 0; i < count; ++i) {
        // Skip empty lines and read type
        std::string type;
        while (std::getline(file_stream, type) && type.empty()) {}
        
        if (matrix_types) {
            matrix_types->push_back(type);
        }
        
        // Construct matrix from stream
        Matrix matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));
    }
}

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

