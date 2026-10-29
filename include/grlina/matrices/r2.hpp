/**
 * @file r2.hpp
 * @author Jan Jendrysiak
 * @brief R2GradedSparseMatrix representation and grid metadata.
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

#ifndef R2GRADED_MATRIX_HPP
#define R2GRADED_MATRIX_HPP

#include "grlina/posets/poset_operations.hpp"
#include <grlina/posets/coordinate.hpp>
#include <grlina/matrices/graded.hpp>
#include <grlina/graded_kernel_scheduler.hpp>
#include <iostream>
#include <vector>
#include <string>

// Graded kernel method definitions: <grlina/graded_kernel.hpp>.

namespace graded_linalg {

/**
 * @brief A graded matrix with degrees in R^2.
 *
 * @tparam index
 */
template <typename index, typename Storage = vec<vec<index>>>
struct R2GradedSparseMatrix : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage> {

    using SparseBase = SparseMatrix<index, Storage>;

    // For kernel computation we will need to compute a grid, i.e. a function Z^2 -> R^2,
    // such that all degrees of columsn and rows are in the image of this function.

    vec<double> x_grid;
    vec<double> y_grid;

    std::unordered_map<double, index> x_to_index;
    std::unordered_map<double, index> y_to_index;

    vec<CoordinateDegree<index, 2>> z2_col_degrees;
    vec<CoordinateDegree<index, 2>> z2_row_degrees;

    // This is also used in kernel computation
    typedef std::priority_queue<index,std::vector<index>,std::greater<index>> PQ;

    Grid_scheduler<index> grid_scheduler;
    std::vector<PQ> pq_row;


    R2GradedSparseMatrix() : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>() {}


    R2GradedSparseMatrix( SparseBase&& other) :  GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(std::move(other)) {}


    public:

    R2GradedSparseMatrix& operator= (const R2GradedSparseMatrix& other) {
        if (this != &other)
            GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>::assign(other);
        return *this;
    }

    R2GradedSparseMatrix& operator= (R2GradedSparseMatrix&& other) {
        if (this != &other)
            GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>::assign(std::move(other));
        return *this;
    }

    R2GradedSparseMatrix& operator = (const GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>& other) {
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>::assign(other);
        return *this;
    }

    R2GradedSparseMatrix& operator = (GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>&& other) {
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>::assign(std::move(other));
        return *this;
    }



    R2GradedSparseMatrix(index cols, index rows) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(cols, rows) {}
    explicit R2GradedSparseMatrix(const vec<r2degree>& degrees) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(degrees) {}
    R2GradedSparseMatrix(index n, vec<index> indicator) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(n, indicator) {}
    R2GradedSparseMatrix(index cols, index rows, std::string type, const index percent = -1) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(cols, rows, type, percent) {} // Constructor with type
    R2GradedSparseMatrix(index m, index n, vec<r2degree> c_degrees, vec<r2degree> r_degrees) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(m, n, c_degrees, r_degrees) {} // Constructor with degrees
    R2GradedSparseMatrix(index m, index n, const array<index>& data, vec<r2degree> c_degrees, vec<r2degree> r_degrees) :
        GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(m, n, data, c_degrees, r_degrees) {} // Constructor with data and degrees

    R2GradedSparseMatrix(const R2GradedSparseMatrix& other)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(other) {
    } // Copy constructor

    R2GradedSparseMatrix(R2GradedSparseMatrix&& other) noexcept
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(std::move(other)) {
    } // Move constructor

    R2GradedSparseMatrix(const GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix, Storage>& other) {
        this->assign(other);
    } // Copy constructor from GradedSparseMatrix
    R2GradedSparseMatrix( const SparseBase& other) : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(other.get_num_cols(), other.get_num_rows()) {
        this->data = other.data;
        this->is_identity = other.is_identity;
    } // Copy constructor from SparseMatrix

    /**
     * @brief Constructs an R^2 graded matrix from an scc or firep data file.
     *
     * @param filepath path to the scc or firep file
     */
    R2GradedSparseMatrix(const std::string& filepath, bool lex_sort = false)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(filepath, lex_sort) {
    } // Constructor from file


    /**
     * @brief Constructs an R^2 graded matrix from an input file stream.
     *
     * @param file_stream input file stream containing the scc or firep data
     * @param lex_sort whether to sort lexicographically
     */
    R2GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false)
        : GradedSparseMatrix<r2degree, index, R2GradedSparseMatrix<index, Storage>, Storage>(file_stream, lex_sort) {
    } // Constructor from ifstream

    /**
     * @brief Sets up the grid scheduler for kernel computation
     *
     */
    void initialise_grid_scheduler();

    /**
     * @brief Sorts the row degrees, stores the permutation used and transforms the entries of the sparse col vectors accordingly.
     *
     */
    void sort_rows_colexicographically(R2GradedSparseMatrix* elements = nullptr, const vec<R2GradedSparseMatrix*>& images = {}) {
        this->sort_rows(TraitLinearOrder<r2degree>{Degree_traits<r2degree>::colex_lambda()}, elements, images);
    }

    vec<index> sort_rows_colexicographically_with_output(R2GradedSparseMatrix* elements = nullptr,
                                                       const vec<R2GradedSparseMatrix*>& images = {}) {
        return this->sort_rows_with_permutation(TraitLinearOrder<r2degree>{Degree_traits<r2degree>::colex_lambda()},
                                                elements, images).new_to_old;
    }

    void sort_columns_colexicographically() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->col_degrees, Degree_traits<r2degree>::colex_lambda());
        this->permute_columns(permutation);

        this->compatibly_sorted = this->row_degrees.size() <= 1;
    }

    vec<index> sort_columns_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<r2degree, index>(this->col_degrees, Degree_traits<r2degree>::colex_lambda());
        this->permute_columns(permutation);

        this->compatibly_sorted = this->row_degrees.size() <= 1;
        return permutation;
    }

    void sort_colexicographically(R2GradedSparseMatrix* elements = nullptr, const vec<R2GradedSparseMatrix*>& images = {}) {
        this->sort_compatibly(TraitLinearOrder<r2degree>{Degree_traits<r2degree>::colex_lambda()}, elements, images);
    }

    private:

    template <typename T>
    void merge_unique_elements(const std::vector<CoordinateDegree<T, 2>>& vec1,
                           const std::vector<CoordinateDegree<T, 2>>& vec2,
                           std::vector<T>& out,
                           bool useFirst = true) {
        auto it1 = vec1.begin();
        auto it2 = vec2.begin();
        T last_x = T();
        bool has_last_x = false;

        while (it1 != vec1.end() || it2 != vec2.end()) {
            T value;

            if (it1 == vec1.end()) {
                value = useFirst ? (*it2)[0] : (*it2)[1];
                if (!has_last_x || value != last_x) {
                    out.push_back(value);
                    last_x = value;
                    has_last_x = true;
                }
                ++it2;
            }
            else if (it2 == vec2.end()) {
                value = useFirst ? (*it1)[0] : (*it1)[1];
                if (!has_last_x || value != last_x) {
                    out.push_back(value);
                    last_x = value;
                    has_last_x = true;
                }
                ++it1;
            }
            else {
                T val1 = useFirst ? (*it1)[0] : (*it1)[1];
                T val2 = useFirst ? (*it2)[0] : (*it2)[1];

                if (val1 < val2) {
                    if (!has_last_x || val1 != last_x) {
                        out.push_back(val1);
                        last_x = val1;
                        has_last_x = true;
                    }
                    ++it1;
                }
                else if (val2 < val1) {
                    if (!has_last_x || val2 != last_x) {
                        out.push_back(val2);
                        last_x = val2;
                        has_last_x = true;
                    }
                    ++it2;
                }
                else {
                    if (!has_last_x || val1 != last_x) {
                        out.push_back(val1);
                        last_x = val1;
                        has_last_x = true;
                    }
                    ++it1;
                    ++it2;
                }
            }
        }
    }


    public:

    /**
     * @brief Stores all appearing unique x and y values of column and row degrees
     * in an ordered way in the x_grid and y_grid vectors ordered.
     * After applying the function, the columns are ordered co(!)lexicographically and the permutation used is returned.
     *
     */
    vec<index> compute_grid_representation() {

        auto column_degrees_lex = this->col_degrees;
        auto row_degrees_lex = this->row_degrees;
        std::sort(column_degrees_lex.begin(), column_degrees_lex.end(), Degree_traits<r2degree>::lex_lambda());
        std::sort(row_degrees_lex.begin(), row_degrees_lex.end(), Degree_traits<r2degree>::lex_lambda());

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

        auto itc = column_degrees_lex.begin();
        auto itr = row_degrees_lex.begin();

        double last_x = -1;
        // Store all unique x values
        merge_unique_elements<double>(column_degrees_lex, row_degrees_lex, x_grid, true);

        auto rows_degrees_colex = this->row_degrees;
        std::sort(rows_degrees_colex.begin(), rows_degrees_colex.end(), Degree_traits<r2degree>::colex_lambda());
        vec<index> column_permutation = this->sort_columns_colexicographically_with_output();
        merge_unique_elements<double>(this->col_degrees, rows_degrees_colex, y_grid, false);

        for(index i = 0; i < x_grid.size(); i++) {
            x_to_index[x_grid[i]] = i;
        }

        for(index i = 0; i < y_grid.size(); i++) {
            y_to_index[y_grid[i]] = i;
        }

        // Compute Z^2 representation of degrees

        for (const auto& pair : this->col_degrees) {
            z2_col_degrees.push_back({x_to_index[pair[0]], y_to_index[pair[1]]});
        }

        for (const auto& pair : this->row_degrees) {
            z2_row_degrees.push_back({x_to_index[pair[0]], y_to_index[pair[1]]});
        }

        return column_permutation;
    }

    private:

        pair<index> snap_degree_to_grid_upper(const r2degree& degree, const vec<double>& x_grid, const vec<double>& y_grid) {
            auto it_x = std::lower_bound(x_grid.begin(), x_grid.end(), degree[0]);
            auto it_y = std::lower_bound(y_grid.begin(), y_grid.end(), degree[1]);
            index x_index = std::distance(x_grid.begin(), it_x);
            index y_index = std::distance(y_grid.begin(), it_y);
            return {x_index, y_index};
        }

        pair<index> snap_degree_to_grid_lower(const r2degree& degree, const vec<double>& x_grid, const vec<double>& y_grid) {
            auto it_x = std::upper_bound(x_grid.begin(), x_grid.end(), degree[0]);
            auto it_y = std::upper_bound(y_grid.begin(), y_grid.end(), degree[1]);
            index x_index = std::distance(x_grid.begin(), it_x)-1;
            index y_index = std::distance(y_grid.begin(), it_y)-1;
            return {x_index, y_index};
        }

    public:

    /**
     * @brief Returns the indices of the closest *smaller* grid point or -1 if to the left/bottom of the grid.
     *
     * @param degree
     * @return pair<index>
     */
    pair<index> get_closest_smaller_grid_point(r2degree& degree){
        if(x_grid.empty() || y_grid.empty()){
            std::cerr << "Grid representation was not computed. Calling compute_grid_representation()." << std::endl;
            compute_grid_representation();
        }
        return snap_degree_to_grid_lower(degree, x_grid, y_grid);
    }

    /** for any grid i:G \into R^2, if this is a presentation,
    * computes a presentation of i_! i^* X
    * by snapping all degrees to the next larger degree in x_grid x y_grid
    * (Highly inefficient implementation)
     */
    void snap_to_grid(const vec<double>& new_x_grid, const vec<double>& new_y_grid,
                      const vec<R2GradedSparseMatrix*>& images = {}) {
        GRLINA_ASSERT(!new_x_grid.empty() && !new_y_grid.empty());
        this->compatibly_sorted = false;
        index m = new_x_grid.size();
        index n = new_y_grid.size();
        vec<index> columns_to_remove = vec<index>();
        vec<index> rows_to_remove = vec<index>();
        for(index i = 0; i < this->get_num_cols(); i++){
            auto snapped = snap_degree_to_grid_upper(this->col_degrees[i], new_x_grid, new_y_grid);
            if(snapped.first == m || snapped.second == n){
                columns_to_remove.push_back(i);
            } else {
                this->col_degrees[i] = {new_x_grid[snapped.first], new_y_grid[snapped.second]};
            }
        }

        for(index i = 0; i < this->get_num_rows(); i++){
            auto snapped = snap_degree_to_grid_upper(this->row_degrees[i], new_x_grid, new_y_grid);
            if(snapped.first == m ||snapped.second == n){
                rows_to_remove.push_back(i);
            } else {
                this->row_degrees[i] = {new_x_grid[snapped.first], new_y_grid[snapped.second]};
            }
        }
        for (auto* image : images) if (image) image->col_degrees = this->row_degrees;
        this->delete_columns(columns_to_remove);
        this->delete_rows(rows_to_remove, nullptr, images);
        this->sort_compatibly(nullptr, images);
        this->minimize(nullptr, images);
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
            std::cout << "(" << pair[0] << ", " << pair[1] << ") ";
        }
        std::cout << std::endl;

        std::cout << "Z^2 Row Degrees: ";
        for (const auto& pair : z2_row_degrees) {
            std::cout << "(" << pair[0] << ", " << pair[1] << ") ";
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
        output_stream << this->get_num_cols() << " " << this->get_num_rows() << " 0" << std::endl;

        // Write the column degrees and data
        for (index i = 0; i < this->get_num_cols(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->col_degrees[i]);
            output_stream << " ; ";
            for (const auto& val : this->data[i]) {
                output_stream << val << " ";
            }
            output_stream << std::endl;
        }

        // Write the row degrees
        for (index i = 0; i < this->get_num_rows(); ++i) {
            Degree_traits<r2degree>::write_degree(output_stream, this->row_degrees[i]);
            output_stream << " ;" << std::endl;
        }
    }

    /**
     * @brief used in graded_kernel. Adapted from MPfree
     *
     */
    void kernel_column_reduction(index i, pair<index>& curr_gr, SparseBase& column_operations, bool store_col_ops=false, bool notify_pq=false);


    /**
     * @brief Returns a basis for the kernel of a 2d graded matrix as another 2d graded matrix.
     * Assumes that the columns are sorted lexicographically.
     * Adapted from MPfree.
     *
     * @return SparseMatrix<index>
     */
    R2GradedSparseMatrix graded_kernel();


    pair<r2degree> bounding_box() const{
        r2degree min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        r2degree max = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};

        for (const auto& degree : this->col_degrees) {
            min = Degree_traits<r2degree>::meet(min, degree);
            max = Degree_traits<r2degree>::join(max, degree);
        }

        for (const auto& degree : this->row_degrees) {
            min = Degree_traits<r2degree>::meet(min, degree);
            max = Degree_traits<r2degree>::join(max, degree);
        }

        return {min, max};
    }



    /**
     * @brief Computes an equidistant grid of n x n points in the bounding box of the degrees of the matrix.
     *
     * @param n
     * @return vec<r2degree>
     */
    vec<r2degree> get_equidistant_grid(const int& n) const {
        pair<r2degree> box = this->bounding_box();
        if(n == 0){
            return {};
        }
        if(n == 1){
            return {box.first};
        }
        double x_step = (box.second[0] - box.first[0]) / (n-1);
        double y_step = (box.second[1] - box.first[1]) / (n-1);
        vec<r2degree> grid;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                grid.push_back({box.first[0] + i * x_step, box.first[1] + j * y_step});
            }
        }
        return grid;
    }

    void snap_to_equidistant_grid(int n, bool end_inclusive = false, const vec<R2GradedSparseMatrix*>& images = {}) {
        pair<r2degree> box = this->bounding_box();
        if(n == 0){
            std::cerr << "Error: Cannot snap to an equidistant grid with 0 points." << std::endl;
        } else {
            r2degree step;
            if(end_inclusive){
                step = (box.second - box.first) / (n - 1);
            } else {
                step = (box.second - box.first) / (n);
            }
            vec<double> new_x_grid;
            vec<double> new_y_grid;
            for(int i = 0; i < n; i++){
                new_x_grid.push_back(box.first[0] + i * step[0]);
                new_y_grid.push_back(box.first[1] + i * step[1]);
            }
            this->snap_to_grid(new_x_grid, new_y_grid, images);
        }
    }

    void cut_above(double x_cutoff, double y_cutoff, const vec<R2GradedSparseMatrix*>& images = {}){
        auto box = this->bounding_box();
        r2degree diagonal = box.second - box.first;
        auto max_degree_x = box.first[0] + x_cutoff * diagonal[0];
        auto max_degree_y = box.first[1] + y_cutoff * diagonal[1];
        vec<index> rows_to_remove;
        vec<index> cols_to_remove;
        for(index i = 0; i < this->get_num_cols(); i++){
            if(this->col_degrees[i][0] > max_degree_x || this->col_degrees[i][1] > max_degree_y){
                cols_to_remove.push_back(i);
            }
        }

        for(index i = 0; i < this->get_num_rows(); i++){
            if(this->row_degrees[i][0] > max_degree_x || this->row_degrees[i][1] > max_degree_y){
                rows_to_remove.push_back(i);
            }
        }

        this->delete_columns(cols_to_remove);
        this->delete_rows(rows_to_remove, nullptr, images);
    };

    void bound_support(r2degree bound, const vec<R2GradedSparseMatrix*>& images = {}) {
        vec<index> rows_to_remove;
        for(index i = 0; i < this->get_num_rows(); i++){
            if(this->row_degrees[i][0] > bound[0] || this->row_degrees[i][1] > bound[1]){
                rows_to_remove.push_back(i);
            } else {
                this->append_column(std::vector<index>({i}),
                                    r2degree{bound[0], this->row_degrees[i][1]});
                this->append_column(std::vector<index>({i}),
                                    r2degree{this->row_degrees[i][0], bound[1]});
            }
        }
        this->delete_rows(rows_to_remove, nullptr, images);
        this->sort_compatibly(nullptr, images);
        this->minimize(nullptr, images);
    }

}; // R2GradedSparseMatrix

} // namespace graded_linalg

#endif // R2GRADED_MATRIX_HPP
