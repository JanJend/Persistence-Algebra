/**
 * @file dense.hpp
 * @author Jan Jendrysiak
 * @brief Dense F2 matrices backed by bitset columns.
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

#ifndef GRLINA_MATRICES_DENSE_HPP
#define GRLINA_MATRICES_DENSE_HPP

#include "grlina/matrices/base.hpp"
#include "grlina/storage/bitset_algebra.hpp"


// Subspace enumeration and decomposition data: <grlina/subspaces.hpp>.

namespace graded_linalg {


struct DenseMatrix : public MatrixUtil<bitset, int, DenseMatrix>{

    bool rowReduced = false;
    bool completeRowReduced = false;
    boost::dynamic_bitset<> pivot_vector;


    /**
     * @brief Deletes the last i rows of the matrix.
     *
     * @param i
     */
    void cull_columns(int i){
        this->num_rows = this->num_rows - i;
        for (auto& column : this->data){
            column.resize(this->num_rows);
        }
    };

    DenseMatrix() : MatrixUtil<bitset, int, DenseMatrix>() {}

    DenseMatrix(int m, int n) : MatrixUtil<bitset, int, DenseMatrix>(m, n) {
        data = vec<bitset>(m, bitset(n, 0));
    }

    DenseMatrix(const vec<bitset>& data) : MatrixUtil<bitset, int, DenseMatrix>(data.size(), data[0].size(), data) {}

    DenseMatrix(const DenseMatrix& other) : MatrixUtil<bitset, int, DenseMatrix>(other)  {}

    DenseMatrix(int m, const std::string& type) : MatrixUtil<bitset, int, DenseMatrix>(m, m) {
        if(type == "Identity") {
            for(int i = 0; i < m; i++) {
                this->data.emplace_back(bitset(m, 0).set(i));
            }
        } else {
            throw std::invalid_argument("Unknown matrix type: " + type);
        }
    }

    /**
     * @brief Construct a new Dense Matrix object from an ifstream if it has the bitsets as 1-0-strings.
     *
     * @param file
     */
    DenseMatrix(std::ifstream& file) : MatrixUtil<bitset, int, DenseMatrix>(){
        // deserialize matrix from a file
        num_cols = 0;
        num_rows = 0;
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(int));
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(int));
        data.reserve(num_cols);
        for (int i = 0; i < num_cols; ++i) {
            unsigned long bs_data;
            file.read(reinterpret_cast<char*>(&bs_data), sizeof(bs_data));
            data.emplace_back(bitset(num_rows, bs_data));
        }
    }

    /**
     * @brief Construct a new Dense Matrix consisting of only elementary basis vectors.
     *
     * @param i_pivots Marks the elementary basis vectors that are to be included in the matrix.
     */
    DenseMatrix(const boost::dynamic_bitset<>& i_pivots) :
        MatrixUtil<bitset, int, DenseMatrix>(i_pivots.count(), i_pivots.size(), std::vector<bitset>(i_pivots.count(), bitset(i_pivots.size(), 0)) ),
        rowReduced(true),       // By construction, the matrix is in reduced form
        pivot_vector(i_pivots){
        // Populate the matrix to form an identity matrix on the pivot positions
        int pivotint = 0; // To keep track of which pivot we're on
        for (int r = 0; r < num_cols; ++r) {
            // Find the next set bit in i_pivots, which will be our next pivot
            while (!i_pivots[pivotint]) {
                ++pivotint; // Skip over unset bits
            }

            // Set the pivot position to 1
            data[r][pivotint] = 1;
            ++pivotint; // Move to the next pivot for the next row
        }
    }

    /**
     * @brief Construct a Dense Matrix in reduced form from a set of pivots_ and a number.
     * The binary representation of this number fills the spots in the reduced matrix which are not necessarily zero.
     *
     * @param pivots_
     * @param positions
     */
    DenseMatrix(const bitset& pivots_, std::vector<std::pair<int,int>> &positions, int filler) : DenseMatrix(pivots_) {
        size_t mul = positions.size();
        for (int j=0; j < mul; j++){
            if (filler & (1 << j)) {
                data[positions[j].first][positions[j].second] = 1;
            }
        }
    }



    /**
     * @brief Add this bitset to the i-th column.
     *
     * @param i
     * @param bitset
     */
    void addOutsideRow(int i, const boost::dynamic_bitset<>& bitset) {
        data[i] ^= bitset; // XOR the i-th row with the bitset
    }

    /**
     * @brief This is a second algorithm to reduce column-wise. It finds a lower triangular reduced form of the matrix involving swaps.
     *
     * @param complete
     */
    void colReduce(bool complete = false) {
        int lead = 0;
        pivot_vector = boost::dynamic_bitset<>(num_rows, 0);
        for (int r = 0; r < num_cols; ++r) {
            if (lead >= num_rows) {
                break; // No more columns to work on, exit the loop
            }

            int i = r;
            // Find the first non-zero entry in the row(potential pivot)
            while (i < num_cols && !data[i][lead]) {
                ++i;
            }

            if (i < num_cols) {
                // Found a non-zero entry, so this row does have a pivot
                // If the pivot is not in the current col, swap the cols
                if (i != r) {
                    swap_cols(i, r);
                }
                pivot_vector[lead] = true; // Mark this row as having a pivot after confirming pivot

                // Eliminate all non-zero entries below this pivot
                for (int j = r + 1; j < num_cols; ++j) {
                    if (data[j][lead]) {
                        data[j] ^= data[r];
                    }
                }

                if (complete) {
                    // Eliminate all non-zero entries above this pivot
                    for (int j = 0; j < r; ++j) {
                        if (data[j][lead]) {
                            data[j] ^= data[r];
                        }
                    }
                }

                ++lead; // Move to the next row
            } else {
                // No pivot in this row, so we move to the next row without incrementing the col int
                ++lead;
                --r; // Stay on the same col for the next iteration
            }
        }
        rowReduced = true;
        if(complete){completeRowReduced = true;}
    }

    /**
     * @brief Writes the matrix to an ofstream by converting it to a string.
     *
     * @param file
     */
    void serialize(std::ofstream& file) const {
        // Write the size of the vector
        GRLINA_ASSERT(data.size() == num_cols);
        file.write(reinterpret_cast<const char*>(&num_cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(&num_rows), sizeof(int));
        // Serialize each dynamic_bitset in the vector

        for (const auto& bs : data) {
            const auto& bs_data = bs.to_ulong();
            file.write(reinterpret_cast<const char*>(&bs_data), sizeof(bs_data));
        }
    }

    /**
     * @brief Computes this*other. Maybe a bit slow.
     *
     * @param other
     */
    DenseMatrix multiply_right(const DenseMatrix& other) const  {
        GRLINA_ASSERT(this->get_num_cols() == other.get_num_rows());
        DenseMatrix result(other.get_num_cols(), this->get_num_rows());

        for (int i = 0; i < other.get_num_cols(); ++i) {
            for (int j = 0; j < this->get_num_rows(); ++j) {
                for (int k = 0; k < this->get_num_cols(); ++k) {
                    if (this->data[k][j] && other.data[i][k]) {
                        result.data[i].flip(j);
                    }
                }
            }
        }

        return result;
    }

    /**
     * @brief Returns a transpose of the matrix.
     *
     * @return DenseMatrix
     */
    DenseMatrix transposed_copy() const {
        DenseMatrix result(this->get_num_rows(), this->get_num_cols());
        for (int i = 0; i < this->get_num_cols(); i++){
            for (int j = 0; j < this->get_num_rows(); j++){
                result.data[j][i] = this->data[i][j];
            }
        }
        return result;
    }

    DenseMatrix divide_left(DenseMatrix& other) const {
        //TO-DO: Implement row-reduction for dense-matrices and use it here instead of applying column-reduction to get an inverse
        DenseMatrix inverse = other.inverse_nocopy();
        DenseMatrix result = inverse.multiply_right(*this); // Unnecessary copy *and* multiplication
        return result;
    }

}; // end of DenseMatrix


} // namespace graded_linalg

#endif // GRLINA_MATRICES_DENSE_HPP
