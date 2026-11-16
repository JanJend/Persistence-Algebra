// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright 2025 TU Graz. Author: Jan Jendrysiak.
/** @file matrix_algorithms.hpp
 * @brief Free algorithms for matrix families and column-space comparison.
 */
#pragma once

#include <limits>
#include <unordered_map>
#include <cstddef>
#include <grlina/matrices/graded.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/r2.hpp>
// Preserve the aggregate includes for existing callers of this header.
#include <grlina/graded_linear_system.hpp>

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

/** Shared grading and relation normal forms for a family of maps. A map is
 * stored as one sparse column, encoding (source column, target row) as c*n+r.
 * Neither empty source columns nor grades are stored separately for every map.
 */
template<class Matrix>
class MapCoefficientContext {
    using Index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    vec<Degree> degrees_;
    std::map<Degree, SparseMatrix<Index>> relations_;
    std::ptrdiff_t rows_;

public:
    MapCoefficientContext(const Matrix& source, const Matrix& target)
        : degrees_(geometric_row_degrees(source)),
          relations_(prepare_target_relation_reducers(target, degrees_)),
          rows_(target.get_num_rows()) {
        if (rows_ && degrees_.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max() / rows_))
            throw std::overflow_error("Flattened map coordinates exceed ptrdiff_t");
    }
    std::ptrdiff_t rows() const { return rows_; }
    std::ptrdiff_t coefficient_rows() const { return rows_ * std::ptrdiff_t(degrees_.size()); }

    template<class Output>
    void append_normal_column(std::size_t column, vec<Index> entries, Output& output) const {
        const auto& reducer = relations_.at(degrees_.at(column));
        const auto original = entries;
        for (Index row : original) {
            const auto pivot = reducer.pivots.find(row);
            if (pivot != reducer.pivots.end())
                Column_traits<vec<Index>, Index>::add_to(reducer.column(pivot->second), entries);
        }
        for (Index row : entries) output.push_back(std::ptrdiff_t(column) * rows_ + row);
    }

    template<class Entries>
    vec<std::ptrdiff_t> normal(const Entries& entries) const {
        vec<std::ptrdiff_t> result;
        auto it = entries.begin();
        while (it != entries.end()) {
            if (*it < 0 || *it >= coefficient_rows())
                throw std::invalid_argument("Map coefficient outside the common presentation coordinates");
            const auto column = *it / rows_;
            vec<Index> values;
            do { values.push_back(Index(*it % rows_)); ++it; }
            while (it != entries.end() && *it / rows_ == column);
            append_normal_column(column, std::move(values), result);
        }
        return result;
    }

    template<class Storage>
    void normalize(SparseMatrix<std::ptrdiff_t, Storage>& maps) const {
        if (maps.get_num_rows() != coefficient_rows())
            throw std::invalid_argument("Map family has incompatible presentation coordinates");
        for (std::ptrdiff_t j = 0; j < maps.get_num_cols(); ++j) maps.set_col(j, normal(maps.column(j)));
        maps.pivots.clear();
        maps.rows_computed = false;
        maps._rows.clear();
    }
};

/** Reduce W first, then H modulo W and its preceding columns. No map-family
 * copies or concatenation; only nonzero sparse columns survive. Inputs must be
 * normalized against the SAME target relations and source grades beforehand.
 */
template<class Index, class Storage>
void reduce_quotient_in_place(SparseMatrix<Index, Storage>& W, SparseMatrix<Index, Storage>& H) {
    if (&W == &H) throw std::invalid_argument("Quotient families must be distinct");
    if (W.get_num_rows() != H.get_num_rows())
        throw std::invalid_argument("Quotient families require common coefficient coordinates");
    using Column = typename Storage::value_type;
    std::unordered_map<Index, Column*> pivots;
    for (auto* family : {&W, &H}) {
        for (auto& column : family->data) {
            while (!column.empty()) {
                auto [it, inserted] = pivots.emplace(column.back(), &column);
                if (inserted) break;
                Column_traits<Column, Index>::add_to(*it->second, column);
            }
        }
    }
    // Moving columns earlier would invalidate pointers used during reduction.
    for (auto* family : {&W, &H}) {
        auto& columns = family->data;
        columns.erase(std::remove_if(columns.begin(), columns.end(), [](const auto& c) { return c.empty(); }), columns.end());
        family->set_num_cols(columns.size());
        family->pivots.clear();
        family->rows_computed = false;
        family->_rows.clear();
    }
}

/** One coefficient column per graded map, after identifying lifts of the same
 * map into coker(target_relations). Entry (row,column) of a lift is flattened
 * to column * number_of_target_generators + row. This is ordinary F2 linear
 * algebra on maps, not a graded span of their individual generator columns.
 */
template <typename Matrix>
SparseMatrix<std::ptrdiff_t> matrix_family_coefficients_modulo_target_relations(
    std::vector<Matrix> maps, const Matrix& target_relations, int threads = 1) {
    if (maps.empty()) return SparseMatrix<std::ptrdiff_t>(0, 0);
    const std::vector<typename Matrix::degree_type> source_degrees =
        geometric_col_degrees(maps.front());
    const std::map<typename Matrix::degree_type, SparseMatrix<matrix_index_t<Matrix>>>
        relation_reducers = prepare_target_relation_reducers(target_relations, source_degrees, threads);
    SparseMatrix<std::ptrdiff_t> coefficients(
        maps.size(), std::ptrdiff_t(maps.front().get_num_cols()) * target_relations.get_num_rows());
    coefficients.resize_data(maps.size());
    parallel_for(maps.size(), threads, [&](std::size_t i) {
        Matrix& map = maps[i];
        GRLINA_DEBUG_CHECK(if (geometric_col_degrees(map) != source_degrees)
            throw std::invalid_argument("Map families require common source generator grades"));
        normalize_map_modulo_target_relations(map, relation_reducers);
        // Each worker owns a whole map and output column; setters also edit metadata.
        auto& column = coefficients.data[i];
        for (std::ptrdiff_t j = 0; j < map.get_num_cols(); ++j)
            for (std::ptrdiff_t row : map.column(j))
                column.push_back(j * map.get_num_rows() + row);
    });
    return coefficients;
}

/** Indices of an independent subfamily of maps modulo target relations.
 * Preserve input order and select original representatives, rather than
 * replacing them by linear combinations.
 */
template <typename Matrix>
std::vector<std::size_t> independent_matrix_family_indices_modulo_target_relations(
    std::vector<Matrix> maps, const Matrix& target_relations, int threads = 1) {
    SparseMatrix<std::ptrdiff_t> coefficients =
        matrix_family_coefficients_modulo_target_relations(std::move(maps), target_relations, threads);
    coefficients.column_reduction_triangular();
    std::vector<std::size_t> selected;
    for (std::ptrdiff_t j = 0; j < coefficients.get_num_cols(); ++j)
        if (!coefficients.is_zero(j)) selected.push_back(std::size_t(j));
    return selected;
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

/** Reduce families of graded maps modulo A and the target relations.
 * B represents a basis of (span(A) + span(B) + Z)/(span(A) + Z), where Z
 * consists of lifts inducing zero maps into coker(target_relations).
 * All maps must share source generator grades and target generator coordinates.
 * Only relations available at each source column's grade may reduce that column.
 * A keeps its length; B is compacted, as in the two-argument overload.
 * The relation matrix is read-only and must not alias a matrix in either family.
 */
template <typename Matrix>
void reduce_matrix_family_modulo(std::vector<Matrix>& A, std::vector<Matrix>& B,
                                const Matrix& target_relations) {
    if (A.empty() && B.empty()) return;
    const auto source_degrees = geometric_col_degrees(A.empty() ? B.front() : A.front());
    const auto reducers = prepare_target_relation_reducers(target_relations, source_degrees);
    for (auto* family : {&A, &B}) {
        for (auto& matrix : *family) {
            GRLINA_DEBUG_CHECK(if (matrix.get_num_rows() != target_relations.get_num_rows() ||
                geometric_col_degrees(matrix) != source_degrees)
                throw std::invalid_argument("Map families require common source grades and target coordinates"));
            normalize_map_modulo_target_relations(matrix, reducers);
        }
    }
    reduce_matrix_family_modulo(A, B);
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
template<typename index, typename Storage, class Matrix>
void simultaneous_row_reduction_on_submatrix(std::unordered_map<index, SparseMatrix<index, Storage>>& N_map, vec<index>& blocks, bitset& support, Matrix& A){
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
