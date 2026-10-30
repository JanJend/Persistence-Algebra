/**
 * @file sparse.hpp
 * @author Jan Jendrysiak
 * @brief Sparse F2 matrix representation and basic operations.
 * @version 2.0.0
 * @date 2025-03-13
 *
 * @copyright 2025 TU Graz
    This file is part of the AIDA library.
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once
#include <grlina/general_checks.hpp>
#include <grlina/storage/compact.hpp>


#include "grlina/storage/columns.hpp"
#ifndef GRLINA_MATRICES_SPARSE_HPP
#define GRLINA_MATRICES_SPARSE_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <grlina/matrices/base.hpp>
#include <grlina/matrices/dense.hpp>
#include <grlina/storage/csc.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <boost/dynamic_bitset.hpp>
#include <cmath>

namespace graded_linalg {



// Helperfunctions for index vectors and sparse matrices

    /**
     * @brief Attention: This might be slower then just counting parallelized! Returns the xor of all vectors whose indices are in the mask, generalizing the approach of add_to/ + operator.
     * In essence this computes the product of the matrix with the column vector given by the mask over GF(2)
     *
     * @tparam index
     * @param matrix
     * @param mask
     *
     * @return vec<index>
     */
    template <typename index, typename Storage>
    vec<index> vectorXORMulti(const Storage& matrix, const vec<index>& mask) {
        vec<index> result;

        int n = mask.size();
        if (n == 1){
            return vec<index>(matrix[mask[0]].begin(), matrix[mask[0]].end());
        }

        // Initialize iterators for all vectors
        std::vector<decltype(std::declval<const Storage&>()[0].begin())> iterators;
        for (index i = 0; i < n; ++i) {
            iterators.push_back(matrix[mask[i]].begin());
        }

        // Iterate through all vectors simultaneously
        while (true) {
            bool allEnd = true;
            index minVal = std::numeric_limits<index>::max();
            int minCount = 0;

            // Find the minimum value among all iterators and count occurrences
            for (size_t i = 0; i < n; ++i) {
                if (iterators[i] != matrix[mask[i]].end()) {
                    allEnd = false;
                    index val = *iterators[i];
                    if (val < minVal) {
                        minVal = val;
                        minCount = 1;
                    } else if (val == minVal) {
                        ++minCount;
                    }
                }
            }

            if (allEnd) {
                break;
            }

            // If the number of iterators pointing to the minimum value is odd, include it in the result
            if (minCount % 2 == 1) {
                result.push_back(minVal);
            }

            // Increment iterators pointing to the minimum value
            for (size_t i = 0; i < n; ++i) {
                if (iterators[i] != matrix[mask[i]].end() && *iterators[i] == minVal) {
                    ++iterators[i];
                }
            }
        }

        return result;
    }

    /**
     * @brief Computes A*b over F_2. Only somewhat fast if A is sparse.
     * Returns the xor of all vectors whose indices are in the mask, generalizing the approach of Michaels "+" operator.
     *
     * @tparam index
     * @param mask
     * @return vec<index>
     */
    template <typename index, typename Storage>
    vec<index> vectorXORMulti(const Storage& matrix, const bitset& mask) {
        vec<index> result;

        GRLINA_ASSERT(mask.size() == matrix.size());

        // Initialize iterators for all vectors
        std::vector<decltype(std::declval<const Storage&>()[0].begin())> iterators;
        std::vector<decltype(std::declval<const Storage&>()[0].begin())> endIterators;
        for (int i = mask.find_first(); i != -1; i = mask.find_next(i)){
            iterators.push_back(matrix[i].begin());
            endIterators.push_back(matrix[i].end());
        }
        // Then count mod 2
        while (true) {
            bool allEnd = true;
            index minVal = std::numeric_limits<index>::max();
            int minCount = 0;

            // Find the minimum value among all iterators and count occurrences
            for (int i = 0; i < iterators.size(); ++i) {
                if (iterators[i] != endIterators[i]) {
                    allEnd = false;
                    index val = *iterators[i];
                    if (val < minVal) {
                        minVal = val;
                        minCount = 1;
                    } else if (val == minVal) {
                        ++minCount;
                    }
                }
            }

            if (allEnd) {
                break;
            }

            // If the number of iterators pointing to the minimum value is odd, include it in the result
            if (minCount % 2 == 1) {
                result.push_back(minVal);
            }

            // Increment iterators pointing to the minimum value
            for (int i = 0; i < iterators.size(); ++i) {
                if (iterators[i] != endIterators[i] && *iterators[i] == minVal) {
                    ++iterators[i];
                }
            }
        }

        return result;
    }


// Preserve index deduction for the original vector-backed bitset overload.
template <typename index>
vec<index> vectorXORMulti(const vec<vec<index>>& matrix, const bitset& mask) {
    return vectorXORMulti<index, vec<vec<index>>>(matrix, mask);
}

/**
 * @brief This function is used to delete rows in a LOC sparse matrix.
 * It creates a map which maps the old indices to the new indices.
 *
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index>
 */
template <typename index>
std::unordered_map<index, index> shiftIndicesMap(const vec<index>& indices) {
    std::unordered_map<index, index> indexMap;
    indexMap.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indexMap[indices[i]] = i;
    }
    return indexMap;
}

/**
 * @brief Expand sorted, distinct surviving indices into a dense old-to-new map.
 * Missing indices map to -1. Moving indices into this overload reuses its buffer.
 */
template <typename index>
vec<index> shiftIndicesMap(vec<index> indices, index num_rows) {
    GRLINA_ASSERT(num_rows >= 0);
    GRLINA_ASSERT(std::is_sorted(indices.begin(), indices.end()));
    GRLINA_ASSERT(std::adjacent_find(indices.begin(), indices.end()) == indices.end());
    GRLINA_ASSERT(indices.empty() || (indices.front() >= 0 && indices.back() < num_rows));
    std::size_t remaining = indices.size();
    indices.resize(num_rows, index(-1));
    // Walk backwards so writing the map cannot overwrite unread surviving indices.
    for (index row = num_rows; row-- > 0;) {
        if (remaining && indices[remaining - 1] == row)
            indices[row] = static_cast<index>(--remaining);
        else indices[row] = index(-1);
    }
    return indices;
}

/**
 * @brief Apply an index map in place, optionally deleting unmapped entries in one pass.
 *
 * @param target
 * @param indexMap
 * @param needsNoDeletion If the target vector only contains indices which are in the indexMap, this can be set to true.
 */
template <typename index, typename Column>
void apply_transformation(Column& target, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion = false) {
    if (needsNoDeletion) {
        for (index& entry : target) {
            auto mapped = indexMap.find(entry);
            GRLINA_ASSERT(mapped != indexMap.end());
            entry = mapped->second;
        }
    } else {
        std::size_t write = 0;
        for (index entry : target) {
            auto mapped = indexMap.find(entry);
            if (mapped != indexMap.end()) target[write++] = mapped->second;
        }
        target.resize(write);
    }
}

/**
 * @brief Apply a transformation given by a vector to a vector of indices.
 *
 * @param target
 * @param index_vector
 * @param needsNoDeletion Defaults to mapping only. If false, discard entries mapped to -1.
 */
template <typename index, typename Column>
void apply_transformation(Column& target, const vec<index>& index_vector, bool needsNoDeletion = true) {
    if (needsNoDeletion) {
        for (index& entry : target) entry = index_vector[entry];
    } else {
        std::size_t write = 0;
        for (index entry : target) {
            const index mapped = index_vector[entry];
            if (mapped != index(-1)) target[write++] = mapped;
        }
        target.resize(write);
    }
}

/**
 * @brief Re-indxes the entries of v by subtracting the number of entries in w which are smaller than v[i].
 *
 * @tparam index
 * @param v
 * @param w
 */
template<typename index, typename Column>
void re_index_non_ocurrences(Column& v, const std::vector<index>& w) {
    GRLINA_ASSERT(std::is_sorted(v.begin(), v.end()));
    GRLINA_ASSERT(std::is_sorted(w.begin(), w.end()));
    index j = 0;
    index m = w.size();
    for (index& a : v) {
        while (j < m && w[j] < a) {
            ++j;
        }
        a -= j;
    }
}

/**
 * @brief Re-indexes the entries of S after having deleted the entries in indices_to_delete.
 *
 * @tparam index
 * @param S
 * @param indices_to_delete
 */
template<typename index>
void re_index_matrix(array<index>& S, const vec<index>& indices_to_delete){
    #pragma omp parallel for
    for(index i = 0; i < S.size(); ++i){
        re_index_non_ocurrences(S[i], indices_to_delete);
    }
}

/**
 * @brief Parallelised function to change a sparse matrix by applying the indexMap to each entry.
 *
 * @param S
 * @param indexMap
 */
template <typename index>
void transform_matrix(array<index>& S, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion) {
#pragma omp parallel for
    for (index i = 0; i < S.size(); ++i) {
        apply_transformation(S[i], indexMap, needsNoDeletion);
    }
}

/**
 * @brief Parallelised function to re-index a sparse matrix. An entry j is changed to index_vector[j].
 *
 * @param S
 * @param indexMap
 */
template <typename index>
void transform_matrix(array<index>& S, const vec<index>& index_vector) {
#pragma omp parallel for
    for (index i = 0; i < S.size(); ++i) {
        apply_transformation(S[i], index_vector);
    }
}

/**
 * @brief Every column is stored as a list of non-zero entries.
 *
 */
template <typename index, typename Storage = vec<vec<index>>>
struct SparseMatrix : public MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>{

    template<typename OtherIndex>
    using rebind_index = SparseMatrix<OtherIndex,
        typename MatrixStorageTraits<vec<index>, index, Storage>::template rebind_index<OtherIndex>>;

    // is_identity is inherited so the shared coefficient edits can invalidate it.
    // Stores caller-managed rows (sometimes in reverse order!).
    vec<vec<index>> _rows;
    // So that the rows are not computed multiple times by accident.
    bool rows_computed = false;

    protected:
        SparseMatrix& assign(const SparseMatrix& other) {
            MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>::assign(other); // Call base
            _rows = other._rows;
            rows_computed = other.rows_computed;
            return *this;
        }

        SparseMatrix& assign(SparseMatrix&& other) {
            MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>::assign(std::move(other));
            _rows = std::move(other._rows);
            rows_computed = other.rows_computed;
            return *this;
        }

    public:
        SparseMatrix& operator=(const SparseMatrix& other) {
            if (this != &other)
                assign(other);
            return *this;
        }

        SparseMatrix& operator=(SparseMatrix&& other) {
            if (this != &other)
                assign(std::move(other));
            return *this;
        }


    SparseMatrix() : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>() { rows_computed = false; }

    SparseMatrix(index m) : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(m) {rows_computed = false;}

    SparseMatrix(index m, index n) : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(m, n) {rows_computed = false;}

    SparseMatrix(const SparseMatrix& other)
        : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(other),
          _rows(other._rows), rows_computed(other.rows_computed) {}

    SparseMatrix(SparseMatrix&& other) noexcept
        : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(std::move(other)),
          _rows(std::move(other._rows)), rows_computed(other.rows_computed) {}

    SparseMatrix(index m, index n, const array<index>& data) : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(m, n, data) {rows_computed = false;}

    SparseMatrix(index m, index n, const std::string& type, const index percent = -1) : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(m, n, type, percent) {rows_computed = false;}

    SparseMatrix(index n, vec<index> indicator) : MatrixUtil<vec<index>, index, SparseMatrix<index, Storage>, Storage>(n, indicator) {rows_computed = false;}

    /**
     *   To treat a sparse matrix as a big vector for reduction we will need this function.
     *   In this way SparseMatrix becomes itself a sparse column type.
     */
    index last_entry_index() const{
        for(index i = this->get_num_cols()-1; i >= 0; i-- ){
            index p = this->col_last(i);
            if( p != -1){
                return i*this->get_num_rows() + p;
            }
        }
        return -1;
    }

    bool is_nonzero_at(index p){
        index col = p / this->get_num_cols();
        index row = p % this->get_num_cols();
        return this->is_nonzero_entry(col, row);
    }

    /**
     * @brief Sorts the entries of a column
     *
     * @param i
     */
    void sort_column(index i){
        this->edit_col(i, [](auto& column) {
            std::sort(column.begin(), column.end());
        });
    }

    /**
     * @brief Sorts all columns
     *
     */
    void sort_data(){
        GRLINA_ASSERT(this->num_cols == this->data.size());
        for(index i = 0; i < this->num_cols; i++){
            sort_column(i);
        }
    }

    index num_of_entries() const {
        index num = 0;
        for(index i = 0; i < this->get_num_cols(); i++){
            num+= this->data[i].size();
        }
        return num;
    }

    /**
     * @brief Computes all rows from column data in reverse order.
     *
     */
    void compute_revrows() {
        _rows.clear();
        for(index i = this->num_cols - 1; i >= 0 ; i--) {
            for(index j : this->data[i]) {
                if(j >= _rows.size()) {
                    _rows.resize(j+1);
                }
                _rows[j].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Compute all rows in forward order.
     *
     */
    void compute_rows_forward(){
        GRLINA_ASSERT(this->num_rows >= 0);
        _rows.clear();
        _rows = vec<vec<index>>(this->num_rows, vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[j].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Computes all rows in forward order given a list of row indices which can appear shifted by the col_indices.
     *
     * @param row_indices
     * @param col_indices
     *
     */
    void compute_rows_forward(const vec<index>& row_indices, const vec<index>& col_indices) {

        auto row_map = shiftIndicesMap(row_indices);
        _rows.clear();
        _rows = vec<vec<index>>(row_indices.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                index r = row_map[j];
                _rows[r].push_back(col_indices[i]);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a list of row indices which can appear without shift.
     *
     * @param row_indices
     */
    void compute_rows_forward(const vec<index>& row_indices) {

        auto row_map = shiftIndicesMap(row_indices);
        _rows.clear();
        _rows = vec<vec<index>>(row_indices.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a map from entries to row indices as a vector
     *
     * @param row_indices
     */
    void compute_rows_forward_map(const vec<index>& row_map){
        _rows = vec<vec<index>>(_rows.size(), vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief   Computes all rows in forward order given a map from entries to row indices as a vector
     *
     * @param row_indices
     */
    void compute_rows_forward_map(const vec<index>& row_map, index size){
        _rows = vec<vec<index>>(size, vec<index>());
        for(index i = 0; i < this->num_cols ; i++) {
            for(index j : this->data[i]) {
                _rows[row_map[j]].push_back(i);
            }
        }
        rows_computed = true;
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only 0..num_cols .
     *
     * @param col_indices
     */
    void compute_columns_from_rows() {
        this->assign_transpose(_rows, this->num_cols);
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only indices in col_indices.
     *
     * @param col_indices
     */
    void compute_columns_from_rows(vec<index>& col_indices, vec<index>& row_indices) {
        auto shiftMap = shiftIndicesMap(col_indices);
        auto rows = _rows;
        for(auto& row : rows) {
            for(auto& entry : row) {
                entry = shiftMap[entry];
            }
        }
        this->assign_transpose(rows, this->num_cols);
        transform_data(row_indices);
    }

    /**
     * @brief Computes the columns from the rows if the rows contain only 0..num_cols .
     *
     * @param col_indices
     */
    void compute_columns_from_rows(vec<index>& row_indices) {
        this->assign_transpose(_rows, this->num_cols);
        transform_data(row_indices);
    }


    /**
     * @brief If the entries of data are not from 1..n, but lie in row_indices, then this function
     * normalises those entries back to 1..n.
     *
     * @param col_indices
     * @param row_indices
     */
    void compute_normalisation(const vec<index>& row_indices) {
        auto row_map = shiftIndicesMap(row_indices);
        transform_data(row_map);
    }

    /**
     * @brief Deletes sorted rows and renumbers entries in O(num_rows + num_cols + nnz).
     *
     * @param row_indices
     */
    void delete_rows(const vec<index>& row_indices_to_remove) {
        GRLINA_ASSERT(std::is_sorted(row_indices_to_remove.begin(), row_indices_to_remove.end()));
        if(row_indices_to_remove.empty()){
            return;
        }
        vec<index> row_map(this->num_rows);
        std::iota(row_map.begin(), row_map.end(), index(0));
        remove_intersection(row_map, row_indices_to_remove);
        const index remaining_rows = static_cast<index>(row_map.size());
        row_map = shiftIndicesMap(std::move(row_map), this->num_rows);
        this->transform_columns([&](auto& column) {
            apply_transformation(column, row_map, false);
        });
        this->num_rows = remaining_rows;
    }

    /**
     * @brief Deletes the columns given by col_indices; col_indices must be sorted!
     *
     * @param col_indices_to_remove
     */
    void delete_columns(const vec<index>& col_indices_to_remove) {
        GRLINA_ASSERT(std::is_sorted(col_indices_to_remove.begin(), col_indices_to_remove.end()));
        if(col_indices_to_remove.empty()){
            return;
        }
        this->erase_columns(col_indices_to_remove);
        this->num_cols = this->data.size();
    }


    void compute_normalisation_with_pivots(const vec<index>& row_indices) {
        GRLINA_ASSERT(row_indices.size() == this->num_rows);
        auto row_map = shiftIndicesMap(row_indices);
        transform_data(row_map);
        // Create a new map with updated keys
        this->set_pivots_without_reducing();
    }

    void transform_data(const std::unordered_map<index, index>& indexMap) {
        this->transform_columns([&](auto& column) {
            apply_transformation(column, indexMap, true);
        }, true);
    }


    void transform_data(const vec<index>& index_vector){
        this->transform_columns([&](auto& column) {
            apply_transformation(column, index_vector);
        }, true);
    }

    /**
     * @brief Prints the _row list.
     *
     */
    void print_rows(){
        for(index i = 0; i < _rows.size(); i++){
            std::cout << "Row " << i << ": ";
            for(index j : _rows[i]){
                std::cout << j << " ";
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief Computes a row from column data in reverse order.
     *
     * @param i
     * @param row
     */
    void compute_row_rev(index i, vec<index>& row) {
        row.clear();
        for(index j = this->num_cols - 1; j >= 0; j--) {
            if(this->is_nonzero_entry(j, i)) {
                row.push_back(j);
            }
        }
    }

    /**
     * @brief Copy a row from _rows
     *
     * @param i
     * @param row
     */
    void get_row_copy(index i, vec<index>& row) {
      row.clear();
      std::copy(this->_rows[i].begin(),this->_rows[i].end(),std::back_inserter(row));
      std::sort(row.begin(),row.end());
    }

    bool is_sorted_sparse() const {
        for(index i = 0; i < this->num_cols; ++i){
            if(!graded_linalg::is_sorted(this->column(i))){
                return false;
            }
        }
        return true;
    }

    // Adds the i-th row to the j-th.
    void fast_row_op(index i, index j){
        Column_traits<vec<index>, index>::add_to(this->_rows[i], this->_rows[j]);
    }

    void row_op_with_update(index i, index j){
        fast_row_op(i, j);
        for(index k : this->_rows[i]){
            this->set_entry(k, j);
        }
    }

    // Adds the i-th row to the j-th when rows are stored in reverse order.
    // also updates the columns, but without sorting or removing duplicates.
    void fast_rev_row_op(index i, index j){
        GRLINA_ASSERT(i != j);
        for(index k : this->_rows[i]){
            this->append_entry(k, j);
        }
        rev_add_to(this->_rows[i], this->_rows[j]);
    }

    // Adds the i-th row to the j-th. This is so expensive because of the reindexing, we might want to use some list type instead.
    // Not finished yet.
    void row_op_on_cols(index i, index j) {
        for(index k = 0; k < this->num_cols; k++) {
            if( this->is_nonzero_entry(k,i) ) {
                this->set_entry(k,j);
            }
        }
    }

    /**
     * @brief Deletes the last entry of each column. This is used in the cokernel computation.
     *
     */
    void delete_last_entries(){
        this->transform_columns([](auto& column) {
            if (!column.empty()) {
                column.pop_back();
            }
        }, true);
    }

    /**
     * @brief Deletes either the last threshold rows of the matrix or everything after index threshold
     *
     * @param threshold
     */
    void cull_columns(const index& threshold, bool from_end){
        if(from_end){
            this->set_num_rows(this->get_num_rows() - threshold);
        } else {
            this->set_num_rows(threshold);
        }

        this->transform_columns([&](auto& column) {
            while(!column.empty() && column.back() >= this->num_rows){
                column.pop_back();
            }
        });

    };

    /**
    * @brief Returns a copy with only the columns at the indices given in colIndices.
    *
    * @param colIndices
    * @return sparseMatrix
    */
    SparseMatrix restricted_domain_copy(vec<index>& colIndices) const {
        for(index i : colIndices){
            GRLINA_ASSERT(i < this->num_cols);
        }
        SparseMatrix result(colIndices.size(), this->num_rows);
        for(index i = 0; i < colIndices.size(); i++){
            result.append_col(this->get_col(colIndices[i]));
        }
        return result;
    }



    /**
    * @brief Returns the transpose.
    *
    * @return sparseMatrix
    */
    SparseMatrix transposed_copy() const {
        SparseMatrix result(this->num_rows, this->num_cols);
        result.assign_transpose(this->data, this->num_rows);
        return result;
    }


    /**
     * @brief Brings Matrix in *non*-completely reduced Column Echelon Form and returns the performed operations.
     *
     * @param performed_ops Applies all column operations to this matrix.
     * @param zero_cols Stores the indices of the columns which are completely zero.
     */
    template<typename smaller_index, typename OtherStorage>
    void column_reduction_triangular_with_memory_int(SparseMatrix<smaller_index, OtherStorage>& performed_ops, vec<smaller_index>& zero_cols) {
        for(index j=0; j < this->num_cols; j++) {
            index p = this->col_last(j);
            while( p >= 0) {
                if(this->pivots.count(p)) {
                    index i = this->pivots[p];
                    this->col_op(i, j);
                    performed_ops.col_op(i, j);
                    auto new_p = this->col_last(j);
                    GRLINA_ASSERT( new_p < p);
                    p = new_p;
                } else {
                    this->pivots[p]=j;
                    break;
                }
            }
            if(p == -1){
                zero_cols.push_back(j);
            }
        }
    }

    /**
     * @brief ! Compute num_cols ! Computes the Kernel of the matrix by column-reduction. Warning: Changes the matrix!
     *
     *
     * @return DERIVED
     */
    template< typename smaller_index>
    rebind_index<smaller_index> get_kernel_int() {
        rebind_index<smaller_index> col_operations(this->num_cols, this->num_cols, "Identity");
        vec<smaller_index> zero_cols;
        this->column_reduction_triangular_with_memory_int<smaller_index>(col_operations, zero_cols);
        return col_operations.restricted_domain_copy(zero_cols);
    }

    /** Return a kernel basis in the original column coordinates, preserving
     * this matrix and ignoring any cached pivots from previous reductions.
     */
    SparseMatrix get_kernel() const {
        SparseMatrix copy(*this);
        copy.pivots.clear();
        return copy.kernel();
    }

    /**
     * @brief Computes the cokernel of a sparse matrix over F_2 by column reducing the matrix first
     * Notice that the result must be a cokernel to the non-reduced matrix, too, so we can also use a copy instead, if we want to keep the original matrix.
    *  Careful, this may not be working correctly right now, because it uses another column_reduction algorithm
    * @param S
    * @param isReduced
    * @return sparseMatrix
    */
    SparseMatrix coKernel(bool isReduced = false, vec<index>* basisLift = nullptr) {
        vec<index> quotientBasis;
        try {
            // If isRedcued is true, make sure that pivots are set!
            quotientBasis = this->coKernel_basis(isReduced);
        } catch (std::out_of_range& e) {
            std::cerr << "Error in coKernel Computation: " << e.what() << std::endl;
            this->print();
            std::abort();
        }

        auto indexMap = shiftIndicesMap(quotientBasis);
        SparseMatrix trunc(*this);
        trunc.delete_last_entries();
        trunc.transform_data(indexMap);

        index newRows = quotientBasis.size();
        index newCols = this->get_num_rows();
        SparseMatrix result(newCols, newRows);

        index j = 0;
        for(index i = 0; i < newCols; i++){
            // Construct Identity Matrix on the generators which descend to the quotient basis.
            if(j < quotientBasis.size() && quotientBasis[j] == i){
                result.append_col(vec<index>{j});
                j++;
            } else {
                // Locate the unqiue column with the last entry at i.
                index colForPivot = this->pivots[i];
                result.append_col(trunc.get_col(colForPivot));
            }
        }
        GRLINA_ASSERT(j == quotientBasis.size() && "Not all quotient basis elements were used");
        if(basisLift){
            *basisLift = quotientBasis;
        }
        return std::move(result);
    }

    /**
     * @brief Assumes that matrix is reduced and a basislift is computed
     * Computes the cokernel of a sparse matrix over F_2
    * @return sparseMatrix
    */
    SparseMatrix coKernel_without_prelim(vec<index>& quotientBasis, vec<index>& row_indices) {
        // TO-DO: This could be done without computing, storing and then deleting trunc. Need to see if we're too slow.

        auto indexMap = shiftIndicesMap(row_indices);  // Do we need this?
        SparseMatrix trunc(*this);

        // Since the matrix is fully reduced, every pivot row has a unique last entry in some column
        // and this is still stored in the pivot map. Therefore for computation this re-indexed matrix works better.
        trunc.delete_last_entries();
        trunc.transform_data(indexMap);

        index newRows = quotientBasis.size();
        index newCols = this->num_rows;
        SparseMatrix result(newCols, newRows);

        index j = 0;
        for(index i = 0; i < newCols; i++){

        // Construct Identity Matrix on the generators which descend to the quotient basis.
            if(j < quotientBasis.size() && quotientBasis[j] == i){
                result.append_col(vec<index>{j});
                j++;
            } else {
                // If were in a non-basis-column, compute the entries directly:
                // Locate the unqiue column in the input matrix with the last entry at i.
                // Observe that the matrix trunc is exactly the non-identity part of the matrix as long as we work over F_2
                result.append_col(trunc.get_col(this->pivots[i]));
            }
        }
        GRLINA_ASSERT(j == quotientBasis.size() && "Not all quotient basis elements were used");
        return std::move(result);
    }


     /**
     * @brief Assumes that matrix is reduced and a basislift was computed
     * Computes the transpose of a cokernel of a sparse matrix over F_2
    * @return sparseMatrix
    */
    SparseMatrix coKernel_transposed_without_prelim(vec<index>& basislift) {
        // TO-DO: Needs testing

        auto index_map = shiftIndicesMap(basislift);

        index new_rows_t = this->num_rows;
        index new_cols_t = basislift.size();
        SparseMatrix result_t(new_cols_t, new_rows_t);

        SparseMatrix result(new_rows_t, new_cols_t);
        index j = 0;
        for(index i = 0; i < new_rows_t; ++i){
            vec<index> column;
            if(j < basislift.size() && basislift[j] == i){
                column.push_back(j++);
            } else {
                index k = this->pivots[i];
                const auto& source = this->data[k];
                for(std::size_t l = 0; l + 1 < source.size(); ++l){
                    column.push_back(index_map[source[l]]);
                }
            }
            result.append_col(std::move(column));
        }
        result_t.assign_transpose(result.data, new_cols_t);

        return result_t;
    }

    /**
     * @brief Returns the number of entries in the matrix.
     *
     * @return long
     */
    long number_of_entries() const {
        long result = 0;
        for(index i=0; i<this->get_num_cols(); i++) {
            // Why was this copied in the original code?
            result+=this->data[i].size();
        }
        return result;
    }

    /**
     * @brief Returns the columns indexed by col_indices, but transformed by a right-multiplication with B.
     *
     * @param B
     * @param col_indices
     */
    SparseMatrix transformed_restricted_copy(const DenseMatrix& B, vec<index>& col_indices){
        index m = B.get_num_cols();
        index n = B.get_num_rows();
        GRLINA_ASSERT(n == col_indices.size());
        SparseMatrix result(m, this->num_rows);
        for(const bitset& v : B.data){
            vec<index> w;
            for(index i = 0; i < n; i++){
                if(v[i]){
                    w.push_back(col_indices[i]);
                }
            }
            result.append_col(vectorXORMulti(this->data, w));
        }
        return result;
    }

    /**
     * @brief Multiplies from the right with a dense matrix.
     *
     * @param D
     */
    void multiply_dense(DenseMatrix& D){
        GRLINA_ASSERT(this->num_cols == D.get_num_rows());
        auto copy = this->data;
        for(index i = 0; i < D.get_num_cols(); i++){
            this->set_col(i, vectorXORMulti<index>(copy, D.data[i]));
        }
    }

    /**
     * @brief Multiplies with D + ID if D is on one side of the diagonal.
     *         Not well optimised.
     * TO-DO: In practice D will often have at most one entry per column so we should probably
     * add single vectors iteratively to reduce unnecessary copying
     *
     * @param D
     */
    void multiply_id_triangular(DenseMatrix& D){
        GRLINA_ASSERT(this->num_cols == D.get_num_rows());
        for(index i = 0; i < D.get_num_cols(); i++){
            if(D.data[i].any()){
                this->add_to_col(i, vectorXORMulti<index>(this->data, D.data[i]));
            }
        }
    }

    /**
     * @brief Multiplies from the right with a square dense matrix, but checks if nothing needs to be done.
     *
     * @param D
     */
    void multiply_dense_with_e_check(DenseMatrix& D, vec<bitset>& e){
        GRLINA_ASSERT(this->num_cols == D.get_num_rows());
        GRLINA_ASSERT(D.get_num_cols() == e.size());
        auto copy = this->data;
        for(index i = 0; i < D.get_num_cols(); i++){
            if(D.data[i] != e[i]){
                this->set_col(i, vectorXORMulti<index>(copy, D.data[i]));
            }
        }
    }

    /**
     * @brief Multiplies the columns given by col_indices
     *          with a square dense matrix.
     *
     * @param D
     */
    void multiply_dense_with_e_check(DenseMatrix& D, vec<bitset>& e_vec, const bitset& col_indices){
        GRLINA_ASSERT(col_indices.count() == D.get_num_rows());
        // D.print();
        // this->print();
        array<index> copy;
        int counter = 0;
        for(auto i = col_indices.find_first(); i != -1; i = col_indices.find_next(i)){
            copy.push_back(this->get_col(i));
        }
        for(auto i = col_indices.find_first(); i != -1; i = col_indices.find_next(i)){
            if(D.data[counter] != e_vec[counter]){
                this->set_col(i, vectorXORMulti<index>(copy, D.data[counter]));
            }
            counter++;
        }
        // this->print();
    }



    vec<index> multiply_with_sparse_vector(vec<index>& v){
        return vectorXORMulti<index>(this->data, v);
    }

    vec<index> multiply_with_dense_vector(bitset& v){
        return vectorXORMulti<index>(this->data, v);
    }

    SparseMatrix multiply_right(SparseMatrix& N);

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     *
     * @param N
     */

    bool reduce_fully(SparseMatrix& N, bool is_diagonal = false) {
        if(!is_diagonal){
            this->column_reduction();
        }
        vec<index> col_ops;
        for(index i = 0; i < N.get_num_cols(); i++){
            col_ops.clear();
            for(auto it = N.data[i].rbegin(); it != N.data[i].rend(); it++){
                if(this->pivots.count(*it)){
                    col_ops.push_back(this->pivots[*it]);
                }
            }
            N.add_to_col(i, vectorXORMulti(this->data, col_ops));
        }
        return N.is_zero();
    }

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     * Alternative version using different addition scheme.
     *
     * @param N
     */

    bool reduce_fully_alt(SparseMatrix& N, bool is_diagonal = false) {
        if(!is_diagonal){
            this->column_reduction_triangular(true);
        }
        N.pivots.clear();
        for(index j=0; j < N.num_cols; j++) {
            index p = N.col_last(j);
            while( p >= 0) {
                if(this->pivots.count(p)) {
                    index i = this->pivots[p];
                     N.add_to_col(j, this->column(i));
                    auto new_p = N.col_last(j);
                    GRLINA_ASSERT( new_p < p);
                    p = new_p;
                } else if ( N.pivots.count(p)){
                    index i = N.pivots[p];
                    N.col_op(i, j);
                    auto new_p = N.col_last(j);
                    GRLINA_ASSERT( new_p < p);
                    p = new_p;
                } else {
                    N.pivots[p]=j;
                    break;
                }
            }
            if (p == -1){
                N.swap_cols(j, N.get_num_cols()-1);
                N.resize_data(N.data.size()-1);
                N.num_cols--;
                j--;
            }
        }
        return N.is_zero();
    }

    /**
     * @brief Reduces N as much as possible using column reduction to get a quotient space representation.
     *
     * @param N
     */
    void reduce_fully(vec<index>& N) {
        this->column_reduction();
        vec<index> col_ops;
        for(index j: N){
            if(this->pivots.count(j)){
                col_ops.push_back(this->pivots[j]);
            }
        }
        Column_traits<vec<index>, index>::add_to(vectorXORMulti(this->data, col_ops), N);
    }

    //destructor
    ~SparseMatrix(){
        _rows.clear();
    }

}; // SparseMatrix;

/**
 * SparseMatrix with compact CSC buffers. Copies share buffers; writes detach.
 * Costly column edits warn at compile time and once at runtime by default.
 */
template<class Index>
using CSCMatrix = SparseMatrix<Index, CSCStorage<Index>>;

template<typename index, typename Storage>
void add_to (const SparseMatrix<index, Storage>& A, SparseMatrix<index, Storage>& B){
    GRLINA_ASSERT(A.get_num_cols() == B.get_num_cols());
    GRLINA_ASSERT(A.get_num_rows() == B.get_num_rows());
    for(index i = 0; i< A.get_num_cols(); i++){
        B.add_to_col(i, A.column(i));
    }
}

/**
 * @brief Sparse F2 product, accumulating only columns selected by N.
 * Works directly with vector-backed and CSC columns; preserves storage type.
 *
 */
template<typename index, typename Storage>
SparseMatrix<index, Storage> multiply(const SparseMatrix<index, Storage>& M, const SparseMatrix<index, Storage>& N){
    GRLINA_ASSERT(M.get_num_cols() == N.get_num_rows());
    // Square identities leave coefficients unchanged.
    if (M.is_identity && M.get_num_rows() == M.get_num_cols()) return N;
    if (N.is_identity && N.get_num_rows() == N.get_num_cols()) return M;
    // A rectangular identity inserts coordinates in its selected standard basis.
    if (M.is_identity) {
        auto result = N;
        result.transform_columns([&](auto& column) {
            for (auto& row : column) row = M.col_last(row);
        });
        result.set_num_rows(M.get_num_rows());
        result.pivots.clear();
        result.is_identity = N.is_identity;
        return result;
    }
    // On the right, an inclusion simply selects columns.
    if (N.is_identity) {
        SparseMatrix<index, Storage> result(N.get_num_cols(), M.get_num_rows());
        for (index i = 0; i < N.get_num_cols(); ++i)
            result.append_col(M.get_col(N.col_last(i)));
        return result;
    }
    SparseMatrix<index, Storage> result(N.get_num_cols(), M.get_num_rows());
    // One byte per output row: 0 = untouched, 1 = odd, 2 = even.
    // Only touched rows are reset between columns; no dense row-column scan.
    vec<unsigned char> parity(M.get_num_rows(), 0);
    for (index j = 0; j < N.get_num_cols(); ++j) {
        vec<index> column;
        for (index k : N.column(j)) {
            for (index row : M.column(k)) {
                if (parity[row] == 0) column.push_back(row);
                parity[row] = parity[row] == 1 ? 2 : 1;
            }
        }
        column.erase(std::remove_if(column.begin(), column.end(), [&](index row) {
            const bool even = parity[row] == 2;
            parity[row] = 0;
            return even;
        }), column.end());
        std::sort(column.begin(), column.end());
        result.append_col(std::move(column));
    }
    return result;
}

/** Compute M^T N through the shared sparse product. Legacy callers may omit
 * row counts; missing trailing rows are zero, so infer the common row extent.
 */
template<typename index, typename Storage>
SparseMatrix<index, Storage> multiply_transpose(const SparseMatrix<index, Storage>& M,
                                               const SparseMatrix<index, Storage>& N) {
    index rows = std::max(M.get_num_rows(), N.get_num_rows());
    for (const auto* matrix : {&M, &N})
        for (index j = 0; j < matrix->get_num_cols(); ++j)
            if (!matrix->column(j).empty()) rows = std::max(rows, index(matrix->col_last(j) + 1));
    auto left = M, right = N;
    left.set_num_rows(rows);
    right.set_num_rows(rows);
    return multiply(left.transposed_copy(), right);
}

template<typename index, typename Storage>
SparseMatrix<index, Storage>  operator*(const SparseMatrix<index, Storage>& M, const SparseMatrix<index, Storage>& N){
    return multiply(M, N);
}


template<typename index, typename Storage>
SparseMatrix<index, Storage> SparseMatrix<index, Storage>::multiply_right(SparseMatrix<index, Storage>& N) {
    return multiply(*this, N);
}



} // namespace graded_linalg

#endif
