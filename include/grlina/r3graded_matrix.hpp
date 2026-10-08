/**
 * @file r3graded_matrix.hpp
 * @author Jan Jendrysiak
 * @brief 
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

#ifndef R3GRADED_MATRIX_HPP
#define R3GRADED_MATRIX_HPP

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {


using r3degree = CoordinateDegree<double, 3>;
using triple = r3degree;

template <typename index, typename MatrixBase = SparseMatrix<index>>
struct R3GradedSparseMatrix : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase> {

    R3GradedSparseMatrix() : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>() {}
    R3GradedSparseMatrix(index m, index n) : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(m, n) {}
    R3GradedSparseMatrix(index n, vec<index> indicator) : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(n, indicator) {}
    R3GradedSparseMatrix(MatrixBase&& other) : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(std::move(other)) {}
    R3GradedSparseMatrix(index m, index n, vec<triple> c_degrees, vec<triple> r_degrees)
        : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(m, n, std::move(c_degrees), std::move(r_degrees)) {}
    R3GradedSparseMatrix(index m, index n, const array<index>& data, vec<triple> c_degrees, vec<triple> r_degrees)
        : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(m, n, data, std::move(c_degrees), std::move(r_degrees)) {}

    void sort_rows_colexicographically() {
        this->sort_rows(TraitLinearOrder<triple>{Degree_traits<triple>::colex_lambda()});
    }

    vec<index> sort_rows_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<triple, index>(
            this->row_degrees, Degree_traits<triple>::colex_lambda());
        vec<index> reverse(permutation.size());
        for (index i = 0; i < static_cast<index>(permutation.size()); ++i)
            reverse[permutation[i]] = i;
        this->transform_data(reverse);
        this->sort_data();
        this->invalidate_cached_rows();
        this->compatible_order_ = Degree_traits<triple>::colex_lambda();
        this->compatibly_sorted = std::is_sorted(this->col_degrees.begin(), this->col_degrees.end(), this->compatible_order_);
        return permutation;
    }

    void sort_columns_colexicographically() {
        this->sort_columns(TraitLinearOrder<triple>{Degree_traits<triple>::colex_lambda()});
    }

    vec<index> sort_columns_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<triple, index>(
            this->col_degrees, Degree_traits<triple>::colex_lambda());
        this->permute_columns(permutation);
        this->invalidate_cached_rows();
        this->compatible_order_ = Degree_traits<triple>::colex_lambda();
        this->compatibly_sorted = std::is_sorted(this->row_degrees.begin(), this->row_degrees.end(), this->compatible_order_);
        return permutation;
    }

    void sort_colexicographically() {
        this->sort_compatibly(TraitLinearOrder<triple>{Degree_traits<triple>::colex_lambda()});
    }

    /**
     * @brief Constructs an R^3 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R3GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) 
        : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(filepath, lex_sort, compute_batches) {
    } // Constructor from file

    /**
     * @brief Constructs an R^3 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R3GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<triple, index, R3GradedSparseMatrix<index, MatrixBase>, MatrixBase>(file_stream, lex_sort, compute_batches ) {
    }


    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r3(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "3" << std::endl;
        output_stream << this->num_cols << " " << this->num_rows << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->num_cols; ++i) {
            Degree_traits<triple>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->num_rows; ++i) {
            Degree_traits<triple>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
            output_stream << std::endl;
        }
    }

    /**
     * @brief Returns a basis for the kernel of a 3-parameter graded matrix.
     * 
     * @return SparseMatrix<index> 
     */
    MatrixBase graded_kernel()  {
        // Implement
        return MatrixBase();
    }


}; // R3GradedSparseMatrix

} // namespace graded_linalg

#endif // R3GRADED_MATRIX_HPP
