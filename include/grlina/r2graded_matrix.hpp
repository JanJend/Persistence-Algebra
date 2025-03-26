/**
 * @file r2graded_matrix.hpp
 * @author Jan Jendrysiak
 * @brief 
 * @version 0.1
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

#ifndef R2GRADED_MATRIX_HPP
#define R2GRADED_MATRIX_HPP

#include <grlina/graded_matrix.hpp>
#include <grlina/grid_scheduler.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>



namespace graded_linalg {

using r2degree = std::pair<double, double>;
template<>
struct Degree_traits<r2degree> {
    static bool equals(const r2degree& lhs, const r2degree& rhs) {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }

    static bool smaller(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first < rhs.first) {
            return (lhs.second <= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second < rhs.second;
        } else {
            return false;
        }
    }

    static bool greater(const r2degree& lhs, const r2degree& rhs) {
        if(lhs.first > rhs.first) {
            return (lhs.second >= rhs.second);
        } else if (lhs.first == rhs.first) {
            return lhs.second > rhs.second;
        } else {
            return false;
        }
    }

    static bool greater_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first >= rhs.first) && (lhs.second >= rhs.second);
    }

    static bool smaller_equal(const r2degree& lhs, const r2degree& rhs) {
        return (lhs.first <= rhs.first) && (lhs.second <= rhs.second);
    }

    static bool lex_order(const r2degree& a, const r2degree& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        } else {
            return a.second < b.second;
        }
    }

    static bool colex_order(const r2degree& a, const r2degree& b) {
        if (a.second != b.second) {
            return a.second < b.second;
        } else {
            return a.first < b.first;
        }
    }

    static std::function<bool(const r2degree&, const r2degree&)> lex_lambda;

    static std::function<bool(const r2degree&, const r2degree&)> colex_lambda;

    static vec<double> position(const r2degree& a)  {
        return {a.first, a.second};
    }

    static void print_degree(const r2degree& a) {
        std::cout << "(" << a.first << ", " << a.second << ")";
    }

    static r2degree join(const r2degree& a, const r2degree& b)  {
        return {std::max(a.first, b.first), std::max(a.second, b.second)};
    }

    static r2degree meet(const r2degree& a, const r2degree& b) {
        return {std::min(a.first, b.first), std::min(a.second, b.second)};
    }

    
    /**
     * @brief Writes the r2degree to an output stream.
     */
    template <typename OutputStream>
    static void write_degree(OutputStream& os, const r2degree& a) {
        os << a.first << " " << a.second;
    }

    template <typename InputStream>
    static r2degree from_stream(InputStream& iss){
        r2degree deg;
        iss >> deg.first >> deg.second;
        return deg;
    }

}; //Degree_traits<r2degree>

/**
 * @brief Lambda function to compare lexicographically for sorting.
 */
std::function<bool(const r2degree&, const r2degree&)> Degree_traits<r2degree>::lex_lambda = [](const r2degree& a, const r2degree& b) {
    return Degree_traits<r2degree>::lex_order(a, b);
};

/**
 * @brief Lambda function to compare colexicographically for sorting.
 */
std::function<bool(const r2degree&, const r2degree&)> Degree_traits<r2degree>::colex_lambda = [](const r2degree& a, const r2degree& b) {
    return Degree_traits<r2degree>::colex_order(a, b);
};



/**
 * @brief A graded matrix with degrees in R^2.
 * 
 * @tparam index 
 */
template <typename index>
struct R2GradedSparseMatrix : GradedSparseMatrix<r2degree, index> {

    // For kernel computation we will need to compute a grid, i.e. a function Z^2 -> R^2, 
    // such that all degrees of columsn and rows are in the image of this function.

    vec<double> x_grid;
    vec<double> y_grid;

    std::unordered_map<double, index> x_to_index;
    std::unordered_map<double, index> y_to_index;

    vec<pair<index>> z2_col_degrees;
    vec<pair<index>> z2_row_degrees;

    // This is also used in kernel computation
    typedef std::priority_queue<index,std::vector<index>,std::greater<index>> PQ;

    Grid_scheduler<index> grid_scheduler;
    std::vector<PQ> pq_row;


    R2GradedSparseMatrix() : GradedSparseMatrix<r2degree, index>() {}
    R2GradedSparseMatrix(index m, index n) : GradedSparseMatrix<r2degree, index>(m, n) {}
    

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     * 
     * @param filepath path to the scc or firep file
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) 
        : GradedSparseMatrix<r2degree, index>(filepath, lex_sort, compute_batches) {
    } // Constructor from file


    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     * 
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     * @param compute_batches whether to compute the column batches and k_max
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : GradedSparseMatrix<r2degree, index>(file_stream, lex_sort, compute_batches) {
    } // Constructor from ifstream

    /**
     * @brief Sets up the grid scheduler for kernel computation
     * 
     */
    void initialise_grid_scheduler() {
        this->grid_scheduler = Grid_scheduler<index>(*this);
    }

    /**
     * @brief Sorts the row degrees, stores the permutation used and transforms the entries of the sparse col vectors accordingly.
     * 
     */
    void sort_rows_colexicographically() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->row_degrees, Degree_traits<r2degree>::colex_lambda);
        vec<index> reverse = vec<index>(permutation.size());
        for (index i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
    }

    /**
     * @brief Sorts the column degrees, stores the permutation used and then reorders the date with the same permutation.
     * 
     */
    void sort_columns_colexicographically() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->col_degrees, Degree_traits<r2degree>::colex_lambda);
        array<index> new_data = array<index>(this->data.size());
        for(index i = 0; i < this->data.size(); i++) {
            new_data[i] = this->data[permutation[i]];
        }
        this->data = new_data;
    }


    /**
     * @brief Stores all appearing unique x and y values of column and row degrees 
     * in an ordered way in the x_grid and y_grid vectors ordered. 
     * 
     */
    void compute_grid_representation(bool sort_lexicographically = false) {

        if(sort_lexicographically) {
            this->sort_columns_lexicographically();
            this->sort_rows_lexicographically();
        }

        x_grid.clear();
        y_grid.clear();
        z2_col_degrees.clear();
        z2_row_degrees.clear();

        x_to_index.clear();
        y_to_index.clear();

        // Reserve space to avoid repeated reallocation
        x_grid.reserve(this->get_num_cols() + this->get_num_rows());
        y_grid.reserve(this->get_num_cols() + this->get_num_rows());
        z2_col_degrees.reserve(this->get_num_cols());
        z2_row_degrees.reserve(this->get_num_rows());

        
        auto itc = this->col_degrees.begin();
        auto itr = this->row_degrees.begin();

        double last_x = -1;
        
        // Store all unique x values
        while(itc != this->col_degrees.end() || itr != this->row_degrees.end()) {
            if(itc == this->col_degrees.end() ){
  
                if(itr->first != last_x) {
                    x_grid.push_back(itr->first);
                    last_x = itr->first;
                }
                itr++;

            } else if(itr == this->row_degrees.end()) {

                if(itc->first != last_x) {
                    x_grid.push_back(itc->first);
                    last_x = itc->first;
                }
                itc++;

            } else if (itc->first < itr->first) {
                if (itc->first != last_x) {
                    x_grid.push_back(itc->first);
                    last_x = itc->first;
                }
                ++itc;
            } else if (itr->first < itc->first) {
                if (itr->first != last_x) {
                    x_grid.push_back(itr->first);
                    last_x = itr->first;
                }
                ++itr;
            } else { // Both values are equal
                if (itc->first != last_x) {
                    x_grid.push_back(itc->first);
                    last_x = itc->first;
                }
                ++itc;
                ++itr;
            }
        }

        // Store all unique y values
        std::vector<double> temp_y;
        temp_y.reserve( this->get_num_cols() + this->get_num_rows() );

        for (const auto& pair : this->col_degrees) {
            temp_y.push_back(pair.second);
        }
        for (const auto& pair :this-> row_degrees) {
            temp_y.push_back(pair.second);
        }

        // Sort and remove duplicates
        std::sort(temp_y.begin(), temp_y.end());
        temp_y.erase(std::unique(temp_y.begin(), temp_y.end()), temp_y.end());

        // Store in y_grid
        y_grid = std::move(temp_y);

        for(index i = 0; i < x_grid.size(); i++) {
            x_to_index[x_grid[i]] = i;
        }

        for(index i = 0; i < y_grid.size(); i++) {
            y_to_index[y_grid[i]] = i;
        }
        
        // Compute Z^2 representation of degrees

        for (const auto& pair : this->col_degrees) {
            z2_col_degrees.emplace_back(x_to_index[pair.first], y_to_index[pair.second]);
        }

        for (const auto& pair : this->row_degrees) {
            z2_row_degrees.emplace_back(x_to_index[pair.first], y_to_index[pair.second]);
        }

    }

    void print_grid(){
        std::cout << "x_grid: ";
        for (const auto& x : x_grid) {
            std::cout << x << " ";
        }
        std::cout << std::endl;

        std::cout << "y_grid: ";
        for (const auto& y : y_grid) {
            std::cout << y << " ";
        }
        std::cout << std::endl;
    }

    void print_grid_representation(){
        std::cout << "Z^2 Column Degrees: ";
        for (const auto& pair : z2_col_degrees) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;

        std::cout << "Z^2 Row Degrees: ";
        for (const auto& pair : z2_row_degrees) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief Writes the R^2 graded matrix to an output stream.
     * // print_to_stream works more generally in every dimension.
     * 
     * @param output_stream output stream to write the matrix data
     */
    template <typename Outputstream>
    void to_stream_r2(Outputstream& output_stream) const {
        
        output_stream << std::fixed << std::setprecision(17);

        // Write the header lines
        output_stream << "scc2020" << std::endl;
        output_stream << "2" << std::endl;
        output_stream << this->num_cols << " " << this->num_rows << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->num_cols; ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->num_rows; ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
            output_stream << std::endl;
        }
    }

    /**
     * @brief used in graded_kernel
     * 
     */
    void kernel_column_reduction(index i, pair<index>& curr_gr, SparseMatrix<index>& column_operations, bool store_col_ops=false, bool notify_pq=false){

        index p = this->col_last(i);
        
        // Reduction loop
        while (p != -1 && this->pivots[p] != -1 && this->pivots[p] < i) {
            index k = this->pivots[p];  // Get the pivot 'k'
            this->col_op(k, i);
            if (store_col_ops) {
                column_operations.col_op(k, i);
            }
            p = this->col_last(i);
        }


        if (notify_pq && p != -1 && this->pivots[p] > i) {
            index j = this->pivots[p];  
            index gr_y_index = this->z2_col_degrees[j].second;  

            this->pq_row[gr_y_index].push(j);
            index gr_x_index = curr_gr.first;  
            this->grid_scheduler.notify(gr_x_index, gr_y_index);
        }

        if (p != -1 && (this->pivots[p] == -1 || this->pivots[p] > i)) {
            this->pivots[p] = i;
        }
    }
    

    /**
     * @brief Returns a basis for the kernel of a 2d graded matrix as another 2d graded matrix.
     * Assumes that the columns are sorted lexicographically.
     * 
     * @return SparseMatrix<index> 
     */
    R2GradedSparseMatrix graded_kernel() {
        this->compute_grid_representation();
        this->initialise_grid_scheduler();
        pq_row.resize(this->y_grid.size());
        // "slave" matrix in mpfree
        SparseMatrix<index> col_operations = SparseMatrix<index>(this->get_num_cols(), this->get_num_cols(), "Identity");

        std::vector<r2degree> new_degrees; // Basis for the free module which is part of the kernel
        std::vector<std::vector<index>> new_cols; // representing the matrix given by the kernel
        
        std::vector<bool> indices_in_kernel(this->get_num_cols(), false);

        // Initialize grid scheduler for processing degrees in order
        Grid_scheduler<index>& grid = this->grid_scheduler;

        while (!grid.at_end()) {

            auto new_degree = grid.next_grade();
            index x = new_degree.first;
            index y = new_degree.second;

            auto& pq = this->pq_row[y];  // Priority queue for row reduction
            auto range_xy = grid.index_range_at(x, y);

            index start_xy = range_xy.first;
            index end_xy = range_xy.second;
            assert(start_xy <= end_xy);

            // Add indices in the range to the priority queue
            for (index i = start_xy; i < end_xy; ++i) {
                pq.push(i);
            }

            while (!pq.empty()) {
                index i = pq.top();

                // Remove duplicates
                while (!pq.empty() && i == pq.top()) {
                    pq.pop();
                }

                assert(z2_col_degrees[i].first <= x);
                assert(z2_col_degrees[i].second == y);

                // Reduce the column and check if it's part of the kernel
                kernel_column_reduction(i, new_degree, col_operations, true, true);

                if (!indices_in_kernel[i] && this->is_zero(i)) {
                    std::vector<index> col = col_operations.get_col(i);
                    new_cols.push_back(std::move(col));
                    new_degrees.emplace_back(this->x_grid[x], this->y_grid[y]);
                    indices_in_kernel[i] = true;
                    // what is this for?
                    this->data[i].clear();
                    col_operations.data[i].clear();
                }
            }
        }

        // Build the resulting kernel matrix
        R2GradedSparseMatrix<index> result(new_cols.size(), this->get_num_cols());
        result.data = std::move(new_cols);
        result.col_degrees = std::move(new_degrees);
        result.row_degrees = this->col_degrees;
        return result;

    }




}; // R2GradedSparseMatrix




} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP
