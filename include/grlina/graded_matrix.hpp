/**
 * @file graded_matrix.hpp
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

#ifndef GRADED_MATRIX_HPP
#define GRADED_MATRIX_HPP

#include <iostream>
#include <grlina/sparse_matrix.hpp>
#include <grlina/orders_and_graphs.hpp>
#include <grlina/to_quiver.hpp>
#include <string>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace graded_linalg {

template<typename index>
using Hom_space_temp = std::pair< SparseMatrix<index>, vec<std::pair<index,index>> >;

template <typename D, typename index, typename DERIVED>
struct GradedSparseMatrix;

/** True when Matrix uses the graded-matrix CRTP contract expected by modules. */
template <typename Matrix, typename = void>
struct is_graded_sparse_matrix : std::false_type {};

template <typename Matrix>
struct is_graded_sparse_matrix<Matrix, std::void_t<typename Matrix::degree_type,
                                                   typename Matrix::index_type>>
    : std::is_base_of<GradedSparseMatrix<typename Matrix::degree_type,
                                         typename Matrix::index_type,
                                         Matrix>, Matrix> {};

template <typename Matrix>
inline constexpr bool is_graded_sparse_matrix_v = is_graded_sparse_matrix<Matrix>::value;


/**
 * @brief A graded matrix with generic degree-type.
 *
 * @tparam D
 * @tparam index
 */
template <typename D, typename index, typename DERIVED>
struct GradedSparseMatrix : public SparseMatrix<index> {

    using degree_type = D;
    using index_type = index;
    using derived_type = DERIVED;

    vec<D> col_degrees;
    vec<D> row_degrees;

    /**
     * True exactly when both degree lists have been certified against the
     * comparator stored in compatible_order_. Explicit-degree construction
     * and parsing check the mandatory Degree_traits order without reordering.
     */
    bool compatibly_sorted = false;

protected:
    /** Comparator which certified compatibly_sorted; kept to detect stale flags. */
    std::function<bool(const D&, const D&)> compatible_order_;

public:

    // Unclear if we really need the following:
    // admissible_col[i] stores to what column i can be added
    array<index> admissible_col;
    // I actually want to store the indices of the columns that can be added to i strictly, i.e. without the batch
    array<index> admissible_col_strict_dual;
    // admissible_row[i] stores what can be added to i
    array<index> admissible_row;
    // Same here, but not strict.
    array<index> admissible_row_dual;

    // This is only relevant for AIDA
    vec<vec<index>> col_batches;
    vec<index> rel_k = vec<index>(100, 0);
	vec<index> gen_k= vec<index>(100, 0);
    index k_max = 1;


    protected:
        GradedSparseMatrix(SparseMatrix<index>&& other) : SparseMatrix<index>(std::move(other)) {
            this->col_degrees = vec<D>(other.get_num_cols());
            this->row_degrees = vec<D>(other.get_num_rows());
        }

        GradedSparseMatrix& assign(const GradedSparseMatrix& other) {
            SparseMatrix<index>::assign(other);
            this->col_degrees = other.col_degrees;
            this->row_degrees = other.row_degrees;
            this->col_batches = other.col_batches;
            this->k_max = other.k_max;
            this->compatibly_sorted = other.compatibly_sorted;
            this->compatible_order_ = other.compatible_order_;
            return *this;
        }

        GradedSparseMatrix& assign(GradedSparseMatrix&& other) {
            SparseMatrix<index>::assign(std::move(other));
            this->col_degrees = std::move(other.col_degrees);
            this->row_degrees = std::move(other.row_degrees);
            this->col_batches = std::move(other.col_batches);
            this->k_max = other.k_max;
            this->compatibly_sorted = other.compatibly_sorted;
            this->compatible_order_ = std::move(other.compatible_order_);
            return *this;
        }


    public:

    GradedSparseMatrix& operator= (const GradedSparseMatrix& other) {
        if (this != &other)
            assign(other);
        return *this;
    }

    GradedSparseMatrix& operator= (GradedSparseMatrix&& other) {
        if (this != &other)
            assign(std::move(other));
        return *this;
    }

    GradedSparseMatrix() : SparseMatrix<index>() {};

    GradedSparseMatrix(const GradedSparseMatrix& other) : SparseMatrix<index>(other), col_degrees(other.col_degrees), row_degrees(other.row_degrees), compatibly_sorted(other.compatibly_sorted), compatible_order_(other.compatible_order_), col_batches(other.col_batches), k_max(other.k_max) {}

    GradedSparseMatrix(GradedSparseMatrix&& other) : SparseMatrix<index>(std::move(other)), col_degrees(std::move(other.col_degrees)), row_degrees(std::move(other.row_degrees)), compatibly_sorted(other.compatibly_sorted), compatible_order_(std::move(other.compatible_order_)), col_batches(std::move(other.col_batches)), k_max(other.k_max) {}

    GradedSparseMatrix(index m, index n) : SparseMatrix<index>(m, n), col_degrees(vec<D>(m)), row_degrees(vec<D>(n)) {}

    GradedSparseMatrix(index m, index n, vec<D> c_degrees, vec<D> r_degrees) : SparseMatrix<index>(m, n), col_degrees(c_degrees), row_degrees(r_degrees) {
        assert(col_degrees.size() == m);
        assert(row_degrees.size() == n);
        refresh_compatible_sorted();
    }

    GradedSparseMatrix(index m, index n, const array<index>& data, vec<D> c_degrees, vec<D> r_degrees) : SparseMatrix<index>(m, n, data), col_degrees(c_degrees), row_degrees(r_degrees) {
        assert(col_degrees.size() == m);
        assert(row_degrees.size() == n);
        refresh_compatible_sorted();
    }

    GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false, bool compute_batches = false)
        : SparseMatrix<index>() {
        this->parse_stream(file_stream, lex_sort, compute_batches);
    }

    GradedSparseMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false)
        : SparseMatrix<index>() {
        std::ifstream file = create_ifstream(filepath);
        this->parse_stream(file, lex_sort, compute_batches);
    }

    GradedSparseMatrix(index n, vec<index> indicator)
        : SparseMatrix<index>(n, indicator), col_degrees(vec<D>(indicator.size())), row_degrees(vec<D>(n)) {}
    GradedSparseMatrix(index cols, index rows, std::string type, const index percent = -1)
        : SparseMatrix<index>(cols, rows, type, percent), col_degrees(vec<D>(cols)), row_degrees(vec<D>(rows)) {}

    template <typename Compare>
    bool degrees_are_sorted(Compare compare) const {
        return std::is_sorted(col_degrees.begin(), col_degrees.end(), compare) &&
               std::is_sorted(row_degrees.begin(), row_degrees.end(), compare);
    }

    /** Check using the mandatory Degree_traits linear extension. */
    bool degrees_are_compatibly_sorted() const {
        return degrees_are_sorted(Degree_traits<D>::lex_lambda());
    }

    /** Recompute the cached flag from the actual poset order. */
    bool refresh_compatible_sorted() {
        compatible_order_ = Degree_traits<D>::lex_lambda();
        compatibly_sorted = degrees_are_sorted(compatible_order_);
        return compatibly_sorted;
    }

    /** Recompute the cached flag for a particular compatible linear order. */
    template <typename Compare>
    bool refresh_compatible_sorted(Compare compare) {
        compatible_order_ = compare;
        compatibly_sorted = degrees_are_sorted(compatible_order_);
        return compatibly_sorted;
    }

    void invalidate_compatible_sorting() noexcept {
        compatibly_sorted = false;
        compatible_order_ = {};
    }

    void invalidate_cached_rows() noexcept {
        this->_rows.clear();
        this->rows_computed = false;
    }

    bool compatible_sorting_is_verified() const {
        return compatibly_sorted && compatible_order_ && degrees_are_sorted(compatible_order_);
    }

    void require_compatibly_sorted(const std::string& operation) {
        if (!compatible_sorting_is_verified()) {
            invalidate_compatible_sorting();
            throw std::invalid_argument(operation +
                " requires rows and columns sorted by one certified compatible linear order");
        }
    }

    bool is_admissible_column_operation(index i, index j) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], col_degrees[j]) && i != j;
    }

    bool is_admissible_column_operation(index i, const D d) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], d);
    }

    bool is_admissible_row_operation(index i, index j) const {
        assert(i != j);
        return Degree_traits<D>::greater_equal( row_degrees[i], row_degrees[j] );
    }

    bool is_admissible_row_operation(index i, D d) const {
        return Degree_traits<D>::greater_equal( row_degrees[i], d);
    }

    bool is_admissible_row_operation(D d, index i) const {
        return Degree_traits<D>::greater_equal( d, row_degrees[i]);
    }

    bool is_strictly_admissible_column_operation(index i, index j) const {
        return Degree_traits<D>::smaller( col_degrees[i], col_degrees[j]);
    }

    /**
     * @brief Checks if the matrix is actually graded.
     */
    bool is_graded_matrix(bool output = false) const {
        for(index i = 0; i < this->num_cols; i++){
            for(index j : this->data[i]){
                if(!Degree_traits<D>::greater_equal(this->col_degrees[i], this->row_degrees[j])){
                    if(output){
                        std::cout << "Column " << i << " has degree ";
                        Degree_traits<D>::print_degree(this->col_degrees[i]);
                        std::cout << " but row " << j << " has degree ";
                        Degree_traits<D>::print_degree(this->row_degrees[j]);
                        std::cout << std::endl;
                    }
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Deletes all entries which violate the gradedness condition.
     *
     */
    void make_graded(){
        this->invalidate_cached_rows();
        for(index i = 0; i < this->num_cols; i++){
            for(index j : this->data[i]){
                if(!Degree_traits<D>::greater_equal(this->col_degrees[i], this->row_degrees[j])){
                    this->set_entry(i, j);
                }
            }
        }
    }

    private:
    static std::ifstream create_ifstream(const std::string& filepath) {
        check_extension(filepath);

        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Unable to open file " << filepath << std::endl;
            std::abort();
        }
        return file;
    }

    static void check_extension(const std::string& filepath) {
        size_t dotPosition = filepath.find_last_of('.');
        bool no_file_extension = false;
        if (dotPosition == std::string::npos) {
           // No dot found, invalid file format
           no_file_extension = true;
            std::cout << " File does not have an extension (.scc .firep .txt)?" << std::endl;
        }

        std::string extension;
        if(!no_file_extension) {
            extension=filepath.substr(dotPosition);
        }

        // Check the file extension and perform actions accordingly
        if (extension == ".scc" || extension == ".firep" || extension == ".txt" || no_file_extension) {
            // std::cout << "Reading presentation file: " << filepath << std::endl;
        } else {
            // Invalid file extension
            std::cout << "Warning, extension does not match .scc, .firep, .txt, or no extension." << std::endl;
        }
    }

    static std::pair<D, std::vector<index>> parse_line(const std::string& line,  const bool& hasEntries = false) {
        std::istringstream iss(line);
        std::vector<index> rel;

        D deg = Degree_traits<D>::from_stream(iss);

        // Consume the semicolon
        std::string tmp;
        iss >> tmp;
        if(tmp != ";"){
            std::cerr << "Error: Expecting a semicolon. Invalid format in the following line: " << line << std::endl;
            std::abort();
        }

        // Parse relation
        if(hasEntries){
            index num;
            while (iss >> num) {
                rel.push_back(num);
            }
        }

        return std::move(std::make_pair(deg, rel));
    }

    void parse_stream(std::istream& file_stream, bool lex_sort, bool compute_batches) {
        std::string line;

        // Read the first line to determine the file type
        std::getline(file_stream, line);
        if (line.find("firep") != std::string::npos) {
            // Skip 2 lines for FIREP
            std::getline(file_stream, line);
            std::getline(file_stream, line);
        } else if (line.find("scc2020") != std::string::npos) {
            // Skip 1 line for SCC2020
            std::getline(file_stream, line);
        } else {
            // Invalid file type
            std::cerr << "Error: Unsupported file format. The first line must contain firep or scc2020." << std::endl;
            std::abort();
        }

        // Parse the first line after skipping
        std::getline(file_stream, line);
        std::istringstream iss(line);
        index num_rel, num_gen, thirdNumber;

        // read the two mandatory numbers
        if (!(iss >> num_rel >> num_gen)) {
            std::cerr << "Error: Invalid format. Expecting at least 2 numbers in the third line." << std::endl;
            std::abort();
        }

        // try to read a third number
        if (iss >> thirdNumber) {
            if (thirdNumber != 0) {
                std::cerr << "Error: A third nonzero number is present in the third line, indicating that this is not just a presentation matrix, but a longer chain complex." << std::endl;
                std::abort();
            }
        }

        this->num_cols = num_rel;
        this->num_rows = num_gen;

        this->col_degrees.reserve(num_rel);
        this->row_degrees.reserve(num_gen);
        this->data.reserve(num_gen);

        index rel_counter = 0;

        bool first_pass = true;
        D last_degree;
        index k_counter = 1;
        index j = 0;
        if (compute_batches) {
            this->col_batches.reserve(num_rel);
            if(num_rel > 0){
                this->col_batches.push_back(vec<index>());
            }
        }

        while (rel_counter < num_rel + num_gen) {
            if(!std::getline(file_stream, line)){
                std::cout << "Error: Unexpected end of file. \n Make sure that the dimensions of the file are correctly given at the beginning of the file." << std::endl;
            } else if (line.empty()) {
                std::cout << "Error: Encountered an empty line in the file." << std::endl;
            }
            std::pair<D, std::vector<index>> line_data;
            if (rel_counter < num_rel) {
                line_data = parse_line(line, true);
                if (compute_batches && !lex_sort) {
                    if (first_pass) {
                        last_degree = line_data.first;
                        first_pass = false;
                    } else if (line_data.first == last_degree) {
                        k_counter++;
                        if (k_counter > this->k_max) {
                            this->k_max = k_counter;
                        }
                    } else {
                        last_degree = line_data.first;
                        j++;
                        this->col_batches.push_back(vec<index>());
                        k_counter = 1;
                    }
                    this->col_batches[j].push_back(rel_counter);
                }
                this->col_degrees.push_back(line_data.first);
                this->data.push_back(line_data.second);
                rel_counter++;
            } else {
                line_data = parse_line(line, false);
                this->row_degrees.push_back(line_data.first);
                rel_counter++;
            }
        }

        if (compute_batches && !lex_sort) {
            // std::cout << "Loaded graded Matrix with k_max: " << this->k_max << std::endl;
        }

        if (lex_sort) {
            std::cout << "Sorting the matrix lexicographically" << std::endl;
            this->sort_columns_lexicographically();
            this->sort_rows_lexicographically();
            if (compute_batches) {
                this->compute_col_batches();
                // std::cout << "Loaded graded Matrix with k_max: " << this->k_max << std::endl;
            }
        } else {
            this->refresh_compatible_sorted();
        }

        if (!compute_batches) {
            // std::cout << "Loaded graded Matrix without computing k_max" << std::endl;
        }
    } // Constructor from ifstream

    public:

    /**
     * @brief Infer the number of rows from the list of row degrees.
     *
     */
    void compute_num_rows_from_degres(){
        this->num_rows = this->row_degrees.size();
    }

    void cull_columns(const index& threshold, bool from_end) {

        SparseMatrix<index>::cull_columns(threshold, from_end);
        this->invalidate_cached_rows();
        this->row_degrees.resize(this->num_rows);
        if (this->compatibly_sorted &&
            (!this->compatible_order_ || !this->degrees_are_sorted(this->compatible_order_)))
            this->invalidate_compatible_sorting();
    }

    /**
     * @brief Prints the content in scc format to a stream. Partially from MPP_UTILS print_in_rivet_format in Graded_matrix.h
     *
     */
    template <typename OutStream>
    void to_stream(OutStream& out, bool header=true) const {

        // Set the precision for floating-point output
        out << std::fixed << std::setprecision(17);

        if(row_degrees.size() == 0){
            std::cerr << "No rows in this matrix." << std::endl;
            return;
        }

        int dimension = Degree_traits<D>::position(this->row_degrees[0]).size();
        if(header) {
            out << "scc2020\n" << dimension << std::endl;
            out << this->get_num_cols() << " " << this->get_num_rows() << " 0" << std::endl;
        }
        for(index i = 0; i < this->get_num_cols(); i++){
            out << Degree_traits<D>::position(this->col_degrees[i]) << " ;";
            for(index j : this->data[i]){
                out << " " << j;
            }
            out << std::endl;
        }
        for(index j = 0; j < this-> row_degrees.size(); j++){
            out << Degree_traits<D>::position(this->row_degrees[j]) << " ;" << std::endl;
        }
    }


    /**
     * @brief computes the linear map induced at a single degree by cutting all columns and rows of a higher degree.
     *      First output is the matrix, second is a list of generators.
     * @param d
     * @param shifted if true then this reshifts to normalise the entries
     * @return std::pair<SparseMatrix, vec<index>>
     */
    std::pair<SparseMatrix<index>, vec<index>> map_at_degree_pair(D d, bool shifted = true) const {
        vec<index> selectedRowDegrees;

        // assert(row_degrees.size() == this->get_num_rows());
        // assert(col_degrees.size() == this->get_num_cols());
        for(index i = 0; i < this->num_rows; i++) {
            if( Degree_traits<D>::smaller_equal(row_degrees[i], d) ) {
                selectedRowDegrees.push_back(i);
            }
        }
        index new_row = selectedRowDegrees.size();
        SparseMatrix<index> result;
        result.set_num_rows(new_row);
        if(new_row == 0){
            result.set_num_cols(0);
            return std::move(std::make_pair(result, selectedRowDegrees));
        }
        for(index i = 0; i < this->num_cols; i++) {
            if( Degree_traits<D>::smaller_equal(col_degrees[i], d) ) {
                result.data.emplace_back(this->data[i]);
            }
        }

        result.compute_num_cols();

        if(shifted){
            transform_matrix(result.data, shiftIndicesMap(selectedRowDegrees), true);
        }

        return std::move(std::make_pair(result, selectedRowDegrees));
    }

    /**
     * @brief computes the linear map induced at a single degree by cutting all columns of higher degrees.
     *
     * @param d
     * @return std::pair<SparseMatrix, vec<index>>
     */
    SparseMatrix<index> map_at_degree(D d, vec<index>& local_admissible_columns) const  {
        // local_data = std::make_shared<Sparse_Matrix>(Sparse_Matrix(0,0));
        for(index i = 0; i < this->num_cols; i++){
            if(is_admissible_column_operation(i, d)){
                // std::cout << "  found an addmisible col op from column " << i << ": ";
                // std::cout << A.col_degrees[columns[i]].first << " " << A.col_degrees[columns[i]].second << " to " <<
                //     A.col_degrees[target].first << " " << A.col_degrees[target].second << std::endl;
                local_admissible_columns.push_back(i);
            }
        }
        return this->restricted_domain_copy(local_admissible_columns);
    }


    /**
     * @brief Returns all row indices whose degree is smaller or equal to d.
     *
     * @param d
     * @return vec<index>
     */
    vec<index> admissible_row_indices(D d) {
        vec<index> result;
        for(index i = 0; i < this->num_rows; i++){
            if(is_admissible_row_operation(d, i)){
                result.push_back(i);
            }
        }
        return result;
    }

    /**
     * @brief Stores the admissible column and row operations when we expect to use these multiple times.
     *
     */
    void precompute_admissible() {
        admissible_col.resize(this->get_num_cols());
        for(index j=0; j<this->get_num_cols(); j++) {
            for(index i=0; i< j; i++) {
                if(this->is_strictly_admissible_column_operation(i,j)) {
                    this->admissible_col_strict_dual[j].push_back(i);
                }
            }
        }
        admissible_row.resize(this->get_num_rows());
        for(index j=0; j<this->get_num_rows(); j++) {
            for(index i=0; i<this->get_num_rows(); i++) {
                if(this->is_admissible_row_operation(i,j)) {
                    this->admissible_row_dual[j].push_back(i);
                }
            }
        }
    }

    /**
     * @brief Returns a vector containing the degrees of the columns and rows.
     *
     * @return degree_list
     */
    vec<D> discrete_support() const {
        assert(this->col_degrees.size() == this->num_cols);
        assert(this->row_degrees.size() == this->num_rows);
        vec<D> result = col_degrees;
        result.insert(result.end(), row_degrees.begin(), row_degrees.end());
        std::sort(result.begin(), result.end(), Degree_traits<D>::lex_lambda());
        // Remove duplicates (works only after sorting)
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }



    /**
     * @brief Counts the number of columns with degree smaller or equal to d.
     *
     * @param d
     * @return index
     */
    index num_cols_before (D d) {
        index result = 0;
        for(index i = 0; i < this->num_cols; i++){
            if( Degree_traits<D>::smaller_equal(col_degrees[i], d) ){
                result++;
            }
        }
        return result;
    }

    /**
     * @brief Counts the number of rows with degree smaller or equal to d.
     *
     * @param d
     * @return index
     */
    index num_rows_before (D d) {
        index result = 0;
        for(index i = 0; i < this->num_rows; i++){
            if( Degree_traits<D>::smaller_equal(row_degrees[i], d) ){
                result++;
            }
        }
        return result;
    }

    /**
     * @brief Prints the matrix as well as the column and row degrees.
     *
     * @param suppress_description
     */
    void print_graded(bool suppress_description = false) const {
        this->print(suppress_description);
        std::cout << "Column Degrees: " ;
        for(D d : col_degrees) {
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << "\n Row Degrees: ";
        for(D d : row_degrees) {
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    /**
     * @brief groups the columns by degree.
     *
     * @param get_statistics
     */
    void compute_col_batches(bool get_statistics = false){
        if(this->get_num_cols() == 0) {
            this->col_batches.clear();
            this->k_max = 0;
            this->rel_k.clear();
            return;
        }
        this->col_batches.clear();
        this->col_batches.reserve(this->get_num_cols());
        D last_degree = col_degrees[0];
        index j = 0;
        this->col_batches.push_back(vec<index>(1, 0));
        index counter = 1;
        for(index i = 1; i < this->get_num_cols(); i++) {
            if( Degree_traits<D>::equals(col_degrees[i], last_degree) ) {
                counter++;
                if(counter > k_max) {
                    k_max = counter;
                }
            } else {
                col_batches.push_back(vec<index>());
                last_degree = col_degrees[i];
                j++;
                counter = 1;
                if(get_statistics) {
                    rel_k[counter]++;
                }
            }
            this->col_batches[j].push_back(i);
        }
        if(get_statistics) {
            rel_k[counter]++;
        }

        if(get_statistics) {
            counter = 1;
            last_degree  = row_degrees[0];
            for(index i = 1; i < this->get_num_rows(); i++){
                if( Degree_traits<D>::equals(row_degrees[i], last_degree) ){
                    counter++;
                } else {
                    gen_k[counter]++;
                    counter = 1;
                    last_degree = row_degrees[i];
                }
            }
            gen_k[counter]++;
        }
    }

    /**
     * @brief Count the number of repeating degrees. Assumes the degree lists to be sorted.
     *
     */
    void get_k_statistics(){
		D tmp = col_degrees[0];
		index counter = 1;
		for(index i = 1; i < this->num_cols; i++){
			if( Degree_traits<D>::equals(col_degrees[i], tmp) ){
				counter++;
			} else {
				rel_k[counter]++;
				counter = 1;
				tmp = col_degrees[i];
			}
		}
        rel_k[counter]++;

        counter = 1;
        tmp = row_degrees[0];
        for(index i = 1; i < this->num_rows; i++){
            if( Degree_traits<D>::equals(row_degrees[i], tmp) ){
                counter++;
            } else {
                gen_k[counter]++;
                counter = 1;
                tmp = row_degrees[i];
            }
        }

        gen_k[counter]++;
	}

    /**
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the columns.
     *
     * @return array<index>
     */
    array<index> get_column_graph() {
        return minimal_directed_graph<D, index>(col_degrees);
    }

    /**
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the rows.
     *
     * @return array<index>
     */
    array<index> get_row_graph() {
        return minimal_directed_graph<D, index>(row_degrees);
    }

    array<index> get_support_graph() {
        vec<D> support = discrete_support();
        return minimal_directed_graph<D, index>(support);
    }

    
    /**
     * @brief Graded version of column deletion.
     *
     * @param indices
     */
    void delete_columns(vec<index>& indices) {
        SparseMatrix<index>::delete_columns(indices);
        vec_deletion(col_degrees, indices);
        this->invalidate_cached_rows();
    }

    /**
     * @brief Graded version of row deletion.
     *
     * @param indices
     */
    void delete_rows(vec<index>& indices) {
        SparseMatrix<index>::delete_rows(indices);
        vec_deletion(row_degrees, indices);
        this->invalidate_cached_rows();
    }


    /**
     * @brief Sorts the columns lexicographically by degree,
     *  using a pointer which points two both the column degrees and the data.
     *
     */
    void sort_columns_lexicographically_with_pointers() {
        sort_columns(Degree_traits<D>::lex_lambda());
    }

    /** Sort columns by an arbitrary compatible linear order. */
    template <typename Compare>
    void sort_columns(Compare compare) {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->col_degrees, compare);
        array<index> new_data(this->data.size());
        for(index i = 0; i < static_cast<index>(this->data.size()); i++) {
            new_data[i] = std::move(this->data[permutation[i]]);
        }
        this->data = std::move(new_data);
        this->invalidate_cached_rows();
        this->refresh_compatible_sorted(compare);
    }

    /** Sort rows by an arbitrary compatible linear order. */
    template <typename Compare>
    void sort_rows(Compare compare) {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->row_degrees, compare);
        vec<index> reverse(permutation.size());
        for (index i = 0; i < static_cast<index>(permutation.size()); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
        this->sort_data();
        this->invalidate_cached_rows();
        this->refresh_compatible_sorted(compare);
    }

    /** Sort rows and columns by the same compatible linear order. */
    template <typename Compare>
    void sort_compatibly(Compare compare) {
        sort_columns(compare);
        sort_rows(compare);
        if (!this->refresh_compatible_sorted(compare)) {
            throw std::invalid_argument("The supplied comparator is not a linear extension of the degree poset");
        }
    }

    /** Sort with the mandatory linear order supplied by Degree_traits. */
    void sort_compatibly() {
        sort_compatibly(Degree_traits<D>::lex_lambda());
    }

    /**
     * @brief Sorts the columns lexicographically by degree,
     * saves the permutation used to do so and applies it to the data.
     *
     */
    void sort_columns_lexicographically() {
        sort_columns(Degree_traits<D>::lex_lambda());
    }

    /**
     * @brief Sorts the columns lexicographically by degree,
     * saves the permutation used to do so, applies it also on the data and returns its inverse.
     *
     * @return vec<index>
     */
    vec<index> sort_columns_lexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->col_degrees, Degree_traits<D>::lex_lambda());
        array<index> new_data = array<index>(this->data.size());
        for(index i = 0; i < this->data.size(); i++) {
            new_data[i] = this->data[permutation[i]];
        }
        this->data = new_data;
        this->invalidate_cached_rows();
        vec<index> reverse = vec<index>(permutation.size());
        for (int i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->refresh_compatible_sorted(Degree_traits<D>::lex_lambda());
        return reverse;
    }

    /**
     * @brief Sorts the rows lexicographically by degree, then transforms the data accordingly.
     *
     */
    void sort_rows_lexicographically(){
        sort_rows(Degree_traits<D>::lex_lambda());
    }

    /**
     * @brief Sorts the rows lexicographically by degree, then transforms the data accordingly and returns the permutation used to sort.
     *
     * @return vec<index>
     */
    vec<index> sort_rows_lexicographically_with_output(){
        vec<index> permutation = sort_and_get_permutation<D, index>(this->row_degrees, Degree_traits<D>::lex_lambda());
        // Need inverse of permutation
        vec<index> reverse = vec<index>(permutation.size());
        for (int i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }
        this->transform_data(reverse);
        this->sort_data();
        this->invalidate_cached_rows();
        this->refresh_compatible_sorted(Degree_traits<D>::lex_lambda());
        return permutation;
    }

    void permute_rows_graded(const vec<index>& permutation) {
        assert(permutation.size() == this->get_num_rows());
        this->transform_data(permutation);
        this->sort_data();
        vec<D> new_row_degrees(this->get_num_rows());
        for(index i = 0; i < (index)permutation.size(); i++) {
            new_row_degrees[permutation[i]] = this->row_degrees[i];
        }
        this->row_degrees = new_row_degrees;
        this->invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }

    DERIVED restricted_domain_copy(vec<index>& colIndices) const {
        DERIVED result( this->SparseMatrix<index>::restricted_domain_copy(colIndices) );
        result.col_degrees = vec<D>(colIndices.size());
        for(index i = 0; i < (index)colIndices.size(); i++) {
            result.col_degrees[i] = this->col_degrees[colIndices[i]];
        }
        result.row_degrees = this->row_degrees;
        result.refresh_compatible_sorted();
        return result;
    }

    bool are_columns_sorted_lexicographically() const {
        for(index i = 1; i < this->num_cols; i++) {
            if( !Degree_traits<D>::lex_lambda()(this->col_degrees[i-1], this->col_degrees[i]) && !Degree_traits<D>::equals(this->col_degrees[i-1], this->col_degrees[i]) ) {
                return false;
            }
        }
        return true;
    }

    bool are_rows_sorted_lexicographically() const {
        for(index i = 1; i < this->num_rows; i++) {
            if( !Degree_traits<D>::lex_lambda()(this->row_degrees[i-1], this->row_degrees[i]) && !Degree_traits<D>::equals(this->row_degrees[i-1], this->row_degrees[i]) ) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Outputs the lists of generators and relations
     *
     */
    void print_degrees() {
        std::cout << "Generators at: ";
            for(D d : row_degrees) {
                Degree_traits<D>::print_degree(d);
                std::cout << " ";
            }
            std::cout << "\nRelations at: ";
            for(D d : col_degrees) {
                Degree_traits<D>::print_degree(d);
                std::cout << " ";
            }
    }

    /**
     * Compute a minimal presentation using only graded basis operations.
     *
     * Redundant relation columns are first reduced to zero and removed.  A
     * unit entry (a relation and generator of equal degree) is cancelled only
     * after its row has been cleared from every other relation column.  This
     * substitution step is essential: deleting the pair before it changes the
     * presented module whenever another relation contains that generator.
     */
    void minimize(){
        this->require_compatibly_sorted("minimize");
        if (this->get_num_cols() != static_cast<index>(this->col_degrees.size()) ||
            this->get_num_rows() != static_cast<index>(this->row_degrees.size()) ||
            this->get_num_cols() != static_cast<index>(this->data.size())) {
            throw std::invalid_argument("minimize received inconsistent matrix dimensions");
        }
        if (!this->is_graded_matrix()) {
            throw std::invalid_argument("minimize requires a graded presentation matrix");
        }

        while (true) {
            // Replace the relation columns by an admissibly reduced generating
            // set. Columns which reduce to zero are redundant relations.
            this->column_reduction_graded();
            vec<index> zero_columns;
            for (index column = 0; column < this->num_cols; ++column) {
                if (this->data[column].empty()) zero_columns.push_back(column);
            }
            this->delete_columns(zero_columns);

            bool cancelled_any_unit = false;
            while (true) {
                // A unit can occur anywhere in a sparse column, not only at
                // its last row index.
                index pivot_column = static_cast<index>(-1);
                index pivot_row = static_cast<index>(-1);
                for (index column = 0;
                     column < this->num_cols && pivot_column == static_cast<index>(-1);
                     ++column) {
                    for (index row : this->data[column]) {
                        if (Degree_traits<D>::equals(this->col_degrees[column],
                                                      this->row_degrees[row])) {
                            pivot_column = column;
                            pivot_row = row;
                            break;
                        }
                    }
                }
                if (pivot_column == static_cast<index>(-1)) break;

                // Substitute the pivot relation in all other relations so
                // that the pivot generator occurs only in pivot_column.
                for (index column = 0; column < this->num_cols; ++column) {
                    if (column == pivot_column) continue;
                    if (std::binary_search(this->data[column].begin(),
                                           this->data[column].end(), pivot_row)) {
                        if (!this->is_admissible_column_operation(pivot_column, column)) {
                            throw std::logic_error(
                                "Internal error: unit cancellation required a non-graded column operation");
                        }
                        this->col_op(pivot_column, column);
                    }
                }
                for (index column = 0; column < this->num_cols; ++column) {
                    if (column != pivot_column &&
                        std::binary_search(this->data[column].begin(),
                                           this->data[column].end(), pivot_row)) {
                        throw std::logic_error(
                            "Internal error: unit pivot row was not cleared before cancellation");
                    }
                }

                vec<index> cancelled_column{pivot_column};
                vec<index> cancelled_row{pivot_row};
                this->delete_columns(cancelled_column);
                this->delete_rows(cancelled_row);
                cancelled_any_unit = true;
            }
            if (!cancelled_any_unit) break;
        }

        if (!this->compatible_sorting_is_verified()) {
            this->invalidate_compatible_sorting();
            throw std::logic_error(
                "Internal error: minimization destroyed compatible degree ordering");
        }
    }

    /** Backwards-compatible name for the correctness-first minimizer. */
    void minimize_variant(){
        this->minimize();
    }

    void shift (D d){
        for(index i = 0; i < this->num_cols; i++){
            Degree_traits<D>::subtract(d, this->col_degrees[i]);
        }
        for(index i = 0; i < this->num_rows; i++){
            Degree_traits<D>::subtract(d, this->row_degrees[i]);
        }
        if (this->compatibly_sorted &&
            (!this->compatible_order_ || !this->degrees_are_sorted(this->compatible_order_)))
            this->invalidate_compatible_sorting();
    }

    void shift_generators (D d){
        for(index i = 0; i < this->num_rows; i++){
            Degree_traits<D>::subtract(d, this->row_degrees[i]);
        }
        if (this->compatibly_sorted &&
            (!this->compatible_order_ || !this->degrees_are_sorted(this->compatible_order_)))
            this->invalidate_compatible_sorting();
    }

    /**
     * @brief Computes a basis of coker_alpha and lifts it to a set of elements in the 0-chains given as a list of rows.
     *
     * @param alpha
     * @return vec<index>
     */
    vec<index> basislift_at (D alpha) const {
        auto [B_alpha, rows_alpha] = this->map_at_degree_pair(alpha);
        // B_alpha.compute_normalisation(rows_alpha); Should be done in map_at_degree_pair ("shifted" is true by default    )
        vec<index> basislift_to_rows_alpha = B_alpha.coKernel_basis();
        vec<index> basis_lift = vec_restriction(rows_alpha, basislift_to_rows_alpha);
        return basis_lift;
    }

    /**
     * @brief Crude computation to get the dimension of the presented module at alpha.
     *
     * @param alpha
     * @return index
     */
    index dim_at(D alpha){
        vec<index> basislift = this->basislift_at(alpha);
        return basislift.size();
    }

    /**
     * @brief Performs column reduction whenever grades make it possible.
     * Needs columns to be sorted in a linear extension of N^2
     *
     */
    vec<index> column_reduction_graded(){
        this->require_compatibly_sorted("column_reduction_graded");
        this->invalidate_cached_rows();
        array<index> multi_pivots = array<index>(this->num_rows, vec<index>());
        vec<index> non_zero_columns;
        for(index i = 0; i < this->num_cols; i++){
            index p = this->col_last(i);
            bool found = true;
            while( p != -1 && multi_pivots[p].size() != 0 && found){
                found = false;
                for(index j : multi_pivots[p]){
                    if( is_admissible_column_operation(j, i) ){
                        this->col_op(j, i);
                        found = true;
                        p = this->col_last(i);
                        break;
                    }
                }
            }
            if(p != -1){
                multi_pivots[p].push_back(i);
                non_zero_columns.push_back(i);
            }
        }

        return non_zero_columns;
    }

    void column_reduction_graded_w_deletion (){
        auto nzc = this->column_reduction_graded();
        this->delete_all_but_columns_alt(nzc);
    }



        /**
     * @brief Sets all generators to be at the degree d and shifts the relations accordingly.s
     * 
     * @param d 
     */
    void set_all_generator_degrees(D d) {
        this->invalidate_compatible_sorting();
        for(index i = 0; i < this->get_num_rows(); i++){
            this->row_degrees[i] = d;
        }
        for(index i = 0; i < this->get_num_cols(); i++){
            if( ! Degree_traits<D>::smaller_equal(d, this->col_degrees[i]) ){
                this->col_degrees[i] = Degree_traits<D>::join(d, this->col_degrees[i]);
            }
        }
    }
    
    /** Backwards-compatible name for the correctness-first minimizer. */
    void semi_minimize(){
        this->minimize();
    }


    /**
     * @brief Checks if the matrix is minimal as a presentation by minimizing a copy.
     *
     * @return true
     * @return false
     */
    bool is_minimal() const {
        DERIVED copy = static_cast<const DERIVED&>(*this);
        copy.minimize();
        return (copy.get_num_cols() == this->get_num_cols()) && (copy.get_num_rows() == this->get_num_rows());
    }

    void append_column(const vec<index>& column_data, const D& column_degree) {
        this->data.push_back(column_data);
        this->col_degrees.push_back(column_degree);
        this->num_cols += 1;
        this->invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }

     /**
     * @brief Appends the columns and column degrees of another graded matrix.
     *
     * @param other
     */
    void append_matrix(const GradedSparseMatrix& other) {
        assert(this->num_rows == other.num_rows);
        assert(this->row_degrees == other.row_degrees);
        for(index i = 0; i < other.num_cols; i++) {
            this->data.push_back(other.data[i]);
            this->col_degrees.push_back(other.col_degrees[i]);
        }
        this->num_cols += other.num_cols;
        this->invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }

    void append_move_matrix(GradedSparseMatrix&& other) {
        assert(this->num_rows == other.num_rows);
        for(index i = 0; i < other.num_cols; i++) {
            this->data.push_back(std::move(other.data[i]));
            this->col_degrees.push_back(std::move(other.col_degrees[i]));
        }
        this->num_cols += other.num_cols;
        this->invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }

    /**
     * @brief Computes a presentation for the quotient by a submodule.
     * The submodule is given by a graded matrix denoting a map from the generators of the submodule to the generators of this module.
     * This method appends the input matrix to the current matrix and minimises it.
     *
     * @param Y
     */
    void quotient_by (GradedSparseMatrix<D, index, DERIVED>& Y) {
        this->append_matrix(Y);
        this->sort_compatibly();
        this->minimize();
    }

    /** 
     * @brief Same but does not change this matrix.
     *
     * @param Y
     * @return DERIVED
     */
    DERIVED quotient_by_copy (DERIVED& Y) const {
        DERIVED copy = static_cast<const DERIVED&>(*this);
        copy.append_matrix(Y);
        copy.sort_compatibly();
        copy.minimize();
        return copy;
    }

    /**
     * @brief Let M present a module and S a submodule of M via a map from generators to generators.
     * If this object presents a map f from the generators, say G, of some module to the one presented by M, 
     * then this method computes the submodule f^{-1}(Im S) as a map to G.
     * 
     *
     * @param
     */
    DERIVED inverse_image(const DERIVED& M, const DERIVED& S) {
        index row_temp = this->num_cols;
        this->append_matrix(S);
        this->append_matrix(M);
        auto K = static_cast<DERIVED&>(*this).graded_kernel();
        K.cull_columns(row_temp, false);
        
        K.column_reduction_graded_w_deletion();
         // TO-DO: If we want to fully reduce, we need to compute a kernel in the derived class.
        return K;
    }

    /**
     * @brief Same as above but does not change this matrix.
     *
     * @param M
     * @param S
     * @return DERIVED
     */
    DERIVED inverse_image_copy (const DERIVED& M, const DERIVED& S) const {

        DERIVED copy = static_cast<const DERIVED&>(*this);
        index row_temp = copy.num_cols;
        copy.append_matrix(S);
        copy.append_matrix(M);
        auto K = copy.graded_kernel();
        K.cull_columns(row_temp, false);
        K.column_reduction_graded_w_deletion();
        // TO-DO: could also fully minimize if we want to?
        return K;
    }


    DERIVED submodule_intersection(const DERIVED& l, const DERIVED& r) const {
        assert(l.row_degrees == r.row_degrees);
        auto k = l.inverse_image_copy(static_cast<const DERIVED&>(*this), r);
        auto i = l*k;
        i.column_reduction_graded_w_deletion();
        return i;
    };

    /**
     * @brief Returns a presentation of the submodule generated by the input, if *this is a presentation.
     *  Does not minimize!
     * @param new_generators
     * @return DERIVED
     */
    DERIVED submodule_generated_by(const DERIVED& new_generators) const {
        assert(this->get_num_rows() == new_generators.get_num_rows());
        assert(this->row_degrees == new_generators.row_degrees);
        DERIVED copy = new_generators;
        index num_new_gens = copy.get_num_cols();
        copy.append_matrix(*this);
        // Obsolete if append_matrix works correctly: copy.col_degrees.insert(copy.col_degrees.end(), this->col_degrees.begin(), this->col_degrees.end());
        // A kernel of this map is the pullback of this presentation along the injection
        DERIVED presentation = copy.graded_kernel();
        // To get the map to the basis, forget all the rows which correspong to relations
        presentation.cull_columns(num_new_gens, false);
        return presentation;
    }

    /**
     * @brief Computes a presentation for the submodule generated at the given degree.
     *
     * @param alpha
     * @return DERIVED
     */
    DERIVED submodule_generated_at(D alpha) const {
        vec<index> basis_lift = this->basislift_at(alpha);
        DERIVED basis_injection(this->get_num_rows(), basis_lift);
        basis_injection.row_degrees = this->row_degrees;
        basis_injection.col_degrees = vec<D>(basis_lift.size(), alpha);
        basis_injection.append_matrix(*this);
        DERIVED presentation = basis_injection.graded_kernel();
        presentation.cull_columns(basis_lift.size(), false);
        presentation.sort_compatibly();
        presentation.minimize();
        return presentation;
    }

    /**
     * @brief Returns a presentation of the submodule generated by this, if M is the presentation of the super-module.
     *  Does not minimize!
     * @param new_generators
     * @return DERIVED
     */
    DERIVED presentation_of_submodule (const DERIVED& M) {
        assert(this->get_num_rows() == M.get_num_rows());
        assert(this->row_degrees == M.row_degrees);
        index num_new_gens = this->get_num_cols();
        this->append_matrix(M);
        // A kernel of this map is the pullback of this presentation along the injection
        DERIVED presentation = static_cast<DERIVED&>(*this).graded_kernel();
        // To get the map to the basis, forget all the rows which correspong to relations
        presentation.cull_columns(num_new_gens, false);
        return presentation;
    }

    /**
     * @brief TO-DO: erase() is linear in the size of the vector, so this is quadratic. Instead we should use
     * std::remove_if !
     *
     * @param cs
     */
    void delete_all_but_columns(vec<index> cs){
        this->invalidate_cached_rows();
        auto i = cs.rbegin();
        for(index j = this->get_num_cols() - 1; j >= 0; j--){ // caution: will not work if index is unsigned
            if(i < cs.rend() && j == *i){
                i++;
            } else {
                this->data.erase(this->data.begin() + j);
                this->col_degrees.erase(this->col_degrees.begin() + j);
                this->num_cols--;
            }
        }
    }

    void delete_all_but_columns_alt(vec<index> cs) {
        this->invalidate_cached_rows();
        decltype(this->data) new_data;
        decltype(this->col_degrees) new_col_degrees;
        
        auto cs_it = cs.begin();
        
        for (index i = 0; i < this->get_num_cols(); ++i) {
            // If current column index matches next column to keep
            if (cs_it != cs.end() && i == *cs_it) {
                new_data.push_back(this->data[i]);
                new_col_degrees.push_back(this->col_degrees[i]);
                ++cs_it;
            }
            // Otherwise skip this column (don't add to new vectors)
        }
        
        this->data = std::move(new_data);
        this->col_degrees = std::move(new_col_degrees);
        this->num_cols = cs.size();
    }

    DERIVED transposed_copy() const {
        DERIVED result;
        result.set_num_rows(this->num_cols);
        result.set_num_cols(this->num_rows);
        result.col_degrees = this->row_degrees;
        result.row_degrees = this->col_degrees;
        result.data.resize(this->num_rows);
        result.compatible_order_ = this->compatible_order_;
        result.compatibly_sorted = this->compatibly_sorted && result.compatible_order_ &&
                                     result.degrees_are_sorted(result.compatible_order_);

        for(index i = 0; i < this->num_cols; i++){
            for(index j : this->data[i]){
                result.data[j].push_back(i);
            }
        }
        return result;
    }


    // The following functions are actually obsolete, tere are functional duplicates in other headers.
    /**
    * @brief Extracts the elements of a vector at the indices given in the second vector.
    * 
    * @param target 
    * @param indices 
    * @return vec<T> 
    */
    template <typename T>
    vec<T> extractElements(const vec<T>& target, const vec<index>& indices) {
    vec<T> result;
    result.reserve(indices.size());
    for (index i : indices) {
        assert(i >= 0 && i < target.size() && "Index out of bounds");
        result.push_back(target[i]);
    }
    return result;
    }

    /**
    * @brief Get all indices for which an element in the first vector is contained in the second vector. Both inputs should be ordered.
    * 
    * @param target 
    * @param subset 
    * @return vec<index> 
    */
    vec<index> getIndicatorVector(vec<index> target, vec<index> subset){
    vec<index> result;
    auto itS = subset.begin();
    for(index i = 0; i < target.size() ; i++){
        if(itS != subset.end()){
        if(*itS == target[i]){
        result.push_back(i);
        itS++;
        }
        }
    }
    assert(itS == subset.end() && "Not all elements of the subset were found in the target");
    return result;
    }
    /**
    * @brief The first vector in each argument is a set of row indices of the original matrix telling us which generators form a basis of the cokernel
    * The second vector is a set of column indices of the cokernel matrix defining a section of the cokernel.
    * 
    * @param source 
    * @param target 
    * @return vec<index> 
    */
    vec<index> basisLifting(std::pair<vec<index>, vec<index>>& source, std::pair<vec<index>, vec<index>>& target){
        //sanity check, comment out in real run
        vec<index> subsetIndicator1 = GradedSparseMatrix::getIndicatorVector(target.first, source.first);
        // Probably this doesnt need to be its own function
        vec<index> result = GradedSparseMatrix::getIndicatorVector(target.first, GradedSparseMatrix::extractElements(source.first, source.second));
        return result;
    }

	/**
	 * @brief This function computes a quiver representation on the poset of unique degrees 
     * appearing for the columns and rows of the matrix.
	 */
	QuiverRepresentation<index, D> induced_quiver_rep(
        const vec<D> vertices = vec<D>(), const array<index> edges = array<index>()) {

        assert(vertices.size() == edges.size());

        if(vertices.size() == 0){
            array<index> support_graph = get_support_graph();
        }
		
        QuiverRepresentation<index, D> rep;
        rep.degrees = vertices;
        for(index i = 0; i < rep.degrees.size(); i++) {
            for(index j : edges[i]){
                rep.edges.push_back(std::make_pair(i, j));
            }
        }
            
		index num_vert = rep.degrees.size();
		index num_edges = rep.edges.size();

		// For each degree we want to store the cokernel, 
		// the row-indices of the generators which form its domain 
		// and a section of the cokernel given by column indices which are mapped to a basis
		vec< SparseMatrix<index> > pointwise_Presentations;
        vec< std::pair< vec<index> , vec<index>> > pointwise_base;
		pointwise_Presentations.reserve(num_vert);
        pointwise_base.reserve(num_vert);
		rep.dimensionVector.reserve(num_vert);

        // #pragma omp parallel for
		for (index i = 0; i < num_vert; i++) {

            SparseMatrix<index> S;
            vec<index> gens;
            std::tie(S, gens) = this->map_at_degree_pair(rep.degrees[i]);
            S.column_reduction();
            vec<index> basisLift;
            pointwise_Presentations.emplace_back(S.coKernel(true, &basisLift));
            pointwise_base.emplace_back(std::make_pair(gens, basisLift));
			rep.dimensionVector.emplace_back(basisLift.size());
		}

        rep.matrices.reserve(num_edges);
		for (index i = 0; i < num_edges; i++) {
            index source = rep.edges[i].first;
            index target = rep.edges[i].second;

            auto sourceBasis = pointwise_base[source];
            auto targetBasis = pointwise_base[target];

            assert(pointwise_Presentations[source].get_num_cols() == sourceBasis.first.size());
            assert(pointwise_Presentations[target].get_num_cols() == targetBasis.first.size());

            vec<index> lift_of_basis = GradedSparseMatrix::basisLifting(sourceBasis, targetBasis);
            rep.matrices.emplace_back( pointwise_Presentations[target].restricted_domain_copy(lift_of_basis) );
        }
    
		for (index j = 0; j < num_edges; j++) { 
            if(rep.matrices[j].get_num_cols() != rep.dimensionVector[rep.edges[j].first] || rep.matrices[j].get_num_rows() != rep.dimensionVector[rep.edges[j].second]){
                throw std::runtime_error("Dimension mismatch in path action");
            }
        }      
        return rep;
	}

}; // GradedSparseMatrix


template <typename D, typename index, typename DERIVED>
DERIVED operator*(const GradedSparseMatrix<D, index, DERIVED>& A, const GradedSparseMatrix<D, index, DERIVED>& B) {
    assert(A.col_degrees == B.row_degrees);
    SparseMatrix<index> product = static_cast<const SparseMatrix<index>&>(A) * static_cast<const SparseMatrix<index>&>(B);
    DERIVED result(std::move(product));
    result.row_degrees = A.row_degrees;
    result.col_degrees = B.col_degrees;
    result.refresh_compatible_sorted();
    return result;
}



template <typename D, typename DERIVED>
DERIVED shifted_identity( vec<D>& generators, const D& epsilon) {
    DERIVED result(generators.size(),generators.size(), "Identity");
    result.col_degrees = generators;
    result.row_degrees = generators;
    for( D& d : result.col_degrees ) {
        Degree_traits<D>::add(epsilon, d);
    }
    result.refresh_compatible_sorted();
    return result;
}

/**
 * @brief Compares two graded matrices by their degrees.
 *
 * @tparam D
 * @tparam index
 */
template <typename D, typename index, typename DERIVED>
struct Compare_by_degrees {

    /**
     * @brief -1 if a<b, 0 if a=b, 1 if a>b
     *
     * @param a
     * @param b
     * @return int
     */
    static int compare_three_way(const GradedSparseMatrix<D, index, DERIVED>& a, const GradedSparseMatrix<D, index, DERIVED>& b) {
        // Compare row degrees
        for (size_t i = 0; i < std::min(a.row_degrees.size(), b.row_degrees.size()); ++i) {
            if (Degree_traits<D>::smaller(a.row_degrees[i], b.row_degrees[i])) {
                return -1;
            }
            if (Degree_traits<D>::smaller(b.row_degrees[i], a.row_degrees[i])) {
                return 1;
            }
        }
        if (a.row_degrees.size() != b.row_degrees.size()) {
            return a.row_degrees.size() < b.row_degrees.size() ? -1 : 1;
        }

        // Compare column degrees
        for (size_t i = 0; i < std::min(a.col_degrees.size(), b.col_degrees.size()); ++i) {
            if (Degree_traits<D>::smaller(a.col_degrees[i], b.col_degrees[i])) {
                return -1;
            }
            if (Degree_traits<D>::smaller(b.col_degrees[i], a.col_degrees[i])) {
                return 1;
            }
        }
        if (a.col_degrees.size() != b.col_degrees.size()) {
            return a.col_degrees.size() < b.col_degrees.size() ? -1 : 1;
        }

        return 0;
    }

    bool operator()(const GradedSparseMatrix<D, index, DERIVED>& a, const GradedSparseMatrix<D, index, DERIVED>& b) const {
        return compare_three_way(a, b) == -1;
    }
};


/**
 * @brief Can be used to recognise file extension, not really needed right now.
 *
 * @param filepath
 * @return std::ifstream
 */
inline std::ifstream check_matrix_file(const std::string& filepath) {
    size_t dotPosition = filepath.find_last_of('.');
    bool no_file_extension = false;
    if (dotPosition == std::string::npos) {
        // No dot found, invalid file format
        no_file_extension = true;
        std::cout << " File does not have an extension (.scc .firep .txt)?" << std::endl;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << " Error: Unable to open file " << filepath << std::endl;
        std::abort();
    }

    std::string extension;
    if(!no_file_extension) {
        extension=filepath.substr(dotPosition);
    }

    // Check the file extension and perform actions accordingly
    if (extension == ".scc" || extension == ".firep" || extension == ".txt" || no_file_extension) {
        // std::cout << "Reading presentation file: " << filepath << std::endl;
    } else {
        // Invalid file extension
        std::cout << "Warning, extension does not match .scc, .firep, .txt, or no extension." << std::endl;
    }
    return file;
}



} // namespace graded_linalg

#endif // GRADED_MATRIX_HPP
