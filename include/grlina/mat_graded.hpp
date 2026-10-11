/**
 * @file mat_graded.hpp
 * @author Jan Jendrysiak
 * @brief GradedSparseMatrix base and degree-admissible matrix operations.
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

#ifndef GRLINA_MAT_GRADED_HPP
#define GRLINA_MAT_GRADED_HPP

#include <iostream>
#include <grlina/mat_sparse.hpp>
#include <grlina/deg_poset_operations.hpp>
#include <grlina/deg_traits.hpp>
#include <grlina/quiver_representations.hpp>
#include <string>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace graded_linalg {

template<typename index>
using Hom_space_temp = std::pair< SparseMatrix<index>, vec<std::pair<index,index>> >;

template <typename D, typename index, typename DERIVED, typename Storage = vec<vec<index>>>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
struct GradedSparseMatrix;

/** Compile-time check for the graded-matrix CRTP contract used by module code. */
template <typename Matrix, typename = void>
struct is_graded_sparse_matrix : std::false_type {};

template <typename Matrix>
struct is_graded_sparse_matrix<Matrix, std::void_t<typename Matrix::degree_type,
                                                   typename Matrix::index_type,
                                                   typename Matrix::storage_type,
                                                   std::enable_if_t<is_degree_v<typename Matrix::degree_type>>>>
    : std::is_base_of<GradedSparseMatrix<typename Matrix::degree_type,
                                         typename Matrix::index_type,
                                         Matrix, typename Matrix::storage_type>, Matrix> {};

template <typename Matrix>
inline constexpr bool is_graded_sparse_matrix_v = is_graded_sparse_matrix<Matrix>::value;

template <typename Matrix, typename = void>
struct has_matrix_graded_kernel : std::false_type {};
template <typename Matrix>
struct has_matrix_graded_kernel<Matrix, std::void_t<decltype(std::declval<Matrix&>().graded_kernel())>>
    : std::is_same<std::decay_t<decltype(std::declval<Matrix&>().graded_kernel())>, Matrix> {};


/**
 * @brief A graded matrix with generic degree-type.
 *
 * @tparam D
 * @tparam index
 * @tparam Storage Column storage for the SparseMatrix base.
 */
template <typename D, typename index, typename DERIVED, typename Storage>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
struct GradedSparseMatrix : public SparseMatrix<index, Storage> {

    using SparseBase = SparseMatrix<index, Storage>;

    static_assert(is_degree_v<D>, "Degree_traits must provide poset comparisons and a linear extension");

    static_assert(std::is_integral_v<index> && std::is_signed_v<index> &&
                  static_cast<index>(-1) < 0,
                  "Graded matrix indices must represent the -1 sentinel");

    using degree_type = D;

    vec<D> col_degrees;
    vec<D> row_degrees;

    /** Both degree lists are sorted by a compatible order. Sorting sets this
     * flag; direct degree edits must update it. No degree scans are performed.
     */
    bool compatibly_sorted = false;

    protected:
        GradedSparseMatrix(SparseBase&& other) : SparseBase(std::move(other)) {
            this->col_degrees = vec<D>(other.get_num_cols());
            this->row_degrees = vec<D>(other.get_num_rows());
        }

        GradedSparseMatrix& assign(const GradedSparseMatrix& other) {
            SparseBase::assign(other);
            this->col_degrees = other.col_degrees;
            this->row_degrees = other.row_degrees;
            this->compatibly_sorted = other.compatibly_sorted;

            return *this;
        }

        GradedSparseMatrix& assign(GradedSparseMatrix&& other) {
            SparseBase::assign(std::move(other));
            this->col_degrees = std::move(other.col_degrees);
            this->row_degrees = std::move(other.row_degrees);
            this->compatibly_sorted = other.compatibly_sorted;

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

    GradedSparseMatrix() : SparseBase() {};

    GradedSparseMatrix(const GradedSparseMatrix& other) : SparseBase(other), col_degrees(other.col_degrees), row_degrees(other.row_degrees), compatibly_sorted(other.compatibly_sorted) {}

    GradedSparseMatrix(GradedSparseMatrix&& other) : SparseBase(std::move(other)), col_degrees(std::move(other.col_degrees)), row_degrees(std::move(other.row_degrees)), compatibly_sorted(other.compatibly_sorted) {}

    GradedSparseMatrix(index m, index n) : SparseBase(m, n), col_degrees(vec<D>(m)), row_degrees(vec<D>(n)) {}

    // Identity on the supplied degrees, preserving their order and repetitions.
    explicit GradedSparseMatrix(const vec<D>& degrees)
        : SparseBase(static_cast<index>(degrees.size()), static_cast<index>(degrees.size()), "Identity"),
          col_degrees(degrees), row_degrees(degrees) {}

    GradedSparseMatrix(index m, index n, vec<D> c_degrees, vec<D> r_degrees) : SparseBase(m, n), col_degrees(c_degrees), row_degrees(r_degrees) {
        GRLINA_ASSERT(col_degrees.size() == m);
        GRLINA_ASSERT(row_degrees.size() == n);
    }

    GradedSparseMatrix(index m, index n, const array<index>& data, vec<D> c_degrees, vec<D> r_degrees) : SparseBase(m, n, data), col_degrees(c_degrees), row_degrees(r_degrees) {
        GRLINA_ASSERT(col_degrees.size() == m);
        GRLINA_ASSERT(row_degrees.size() == n);
    }

    GradedSparseMatrix(std::istream& file_stream, bool lex_sort = false)
        : SparseBase() {
        this->parse_stream(file_stream, lex_sort);
    }

    GradedSparseMatrix(const std::string& filepath, bool lex_sort = false)
        : SparseBase() {
        std::ifstream file = create_ifstream(filepath);
        this->parse_stream(file, lex_sort);
    }

    GradedSparseMatrix(index n, vec<index> indicator)
        : SparseBase(n, indicator), col_degrees(vec<D>(indicator.size())), row_degrees(vec<D>(n)) {}
    GradedSparseMatrix(index cols, index rows, std::string type, const index percent = -1)
        : SparseBase(cols, rows, type, percent), col_degrees(vec<D>(cols)), row_degrees(vec<D>(rows)) {}

    bool is_admissible_column_operation(index i, index j) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], col_degrees[j]) && i != j;
    }

    bool is_admissible_column_operation(index i, const D d) const {
        return Degree_traits<D>::smaller_equal( col_degrees[i], d);
    }

    bool is_admissible_row_operation(index i, index j) const {
        GRLINA_ASSERT(i != j);
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
                        std::cout << "Column " << i << " is not admissible in row " << j;
                        if constexpr (has_degree_print_v<D>) {
                            std::cout << ": column degree ";
                            Degree_traits<D>::print_degree(this->col_degrees[i]);
                            std::cout << ", row degree ";
                            Degree_traits<D>::print_degree(this->row_degrees[j]);
                        }
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
        if (!iss) throw std::runtime_error("Invalid SCC degree: " + line);

        // Consume the semicolon
        std::string tmp;
        iss >> tmp;
        if(tmp != ";"){
            throw std::runtime_error("Expected SCC degree separator: " + line);
        }

        // Parse relation
        if(hasEntries){
            index num;
            while (iss >> num) {
                rel.push_back(num);
            }
            if (!iss.eof()) throw std::runtime_error("Invalid SCC entry: " + line);
        }

        return std::move(std::make_pair(deg, rel));
    }

    void parse_stream(std::istream& file_stream, bool lex_sort) {
        static_assert(has_degree_io_v<D>, "Matrix input requires Degree_traits::poset_id, write_degree and from_stream");
        std::string line;

        // Read the first line to determine the file type
        std::getline(file_stream, line);
        if (line.find("firep") != std::string::npos) {
            // Skip 2 lines for FIREP
            std::getline(file_stream, line);
            std::getline(file_stream, line);
        } else if (line.find("scc2020") != std::string::npos) {
            std::getline(file_stream, line);
            std::istringstream poset_line(line);
            std::string id, extra;
            if (!(poset_line >> id) || (poset_line >> extra) ||
                id != std::string(Degree_traits<D>::poset_id))
                throw std::runtime_error("SCC poset identifier does not match matrix degree type");
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
        this->reserve_data(num_rel);

        index rel_counter = 0;

        while (rel_counter < num_rel + num_gen) {
            if(!std::getline(file_stream, line)){
                std::cout << "Error: Unexpected end of file. \n Make sure that the dimensions of the file are correctly given at the beginning of the file." << std::endl;
            } else if (line.empty()) {
                std::cout << "Error: Encountered an empty line in the file." << std::endl;
            }
            std::pair<D, std::vector<index>> line_data;
            if (rel_counter < num_rel) {
                line_data = parse_line(line, true);
                this->col_degrees.push_back(line_data.first);
                this->append_col(std::move(line_data.second));
                rel_counter++;
            } else {
                line_data = parse_line(line, false);
                this->row_degrees.push_back(line_data.first);
                rel_counter++;
            }
        }

        if (lex_sort) {
            std::cout << "Sorting the matrix lexicographically" << std::endl;
            this->sort_compatibly();
        } else {
            this->compatibly_sorted = false;
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

        SparseBase::cull_columns(threshold, from_end);
        this->invalidate_cached_rows();
        this->row_degrees.resize(this->num_rows);
    }

    /**
     * @brief Prints the content in scc format to a stream. Partially from MPP_UTILS print_in_rivet_format in Graded_matrix.h
     *
     */
    template <typename OutStream>
    void to_stream(OutStream& out, bool header=true) const {
        static_assert(has_degree_position_v<D>, "Coordinate matrix output requires Degree_traits::position(degree)");

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
    std::pair<SparseBase, vec<index>> map_at_degree_pair(D d, bool shifted = true) const {
        vec<index> selectedRowDegrees;

        // GRLINA_ASSERT(row_degrees.size() == this->get_num_rows());
        // GRLINA_ASSERT(col_degrees.size() == this->get_num_cols());
        for(index i = 0; i < this->num_rows; i++) {
            if( Degree_traits<D>::smaller_equal(row_degrees[i], d) ) {
                selectedRowDegrees.push_back(i);
            }
        }
        index new_row = selectedRowDegrees.size();
        SparseBase result;
        result.set_num_rows(new_row);
        if(new_row == 0){
            result.set_num_cols(0);
            return std::move(std::make_pair(result, selectedRowDegrees));
        }
        for(index i = 0; i < this->num_cols; i++) {
            if( Degree_traits<D>::smaller_equal(col_degrees[i], d) ) {
                result.append_col(this->get_col(i));
            }
        }

        result.compute_num_cols();

        if(shifted){
            result.transform_data(shiftIndicesMap(selectedRowDegrees));
        }

        return std::move(std::make_pair(result, selectedRowDegrees));
    }

    /**
     * @brief computes the linear map induced at a single degree by cutting all columns of higher degrees.
     *
     * @param d
     * @return std::pair<SparseMatrix, vec<index>>
     */
    SparseBase map_at_degree(D d, vec<index>& local_admissible_columns) const  {
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
     * @brief Returns a vector containing the degrees of the columns and rows.
     *
     * @return degree_list
     */
    vec<D> discrete_support() const {
        GRLINA_ASSERT(this->col_degrees.size() == this->num_cols);
        GRLINA_ASSERT(this->row_degrees.size() == this->num_rows);
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
        static_assert(has_degree_print_v<D>, "Degree printing requires Degree_traits::print_degree(degree)");
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
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the columns.
     *
     * @return array<index>
     */
    array<index> get_column_graph() {
        GRLINA_DEBUG_CHECK(if (!compatibly_sorted)
            throw std::invalid_argument("get_column_graph requires compatibly sorted rows and columns"));
        return minimal_directed_graph<D, index>(col_degrees);
    }

    /**
     * @brief Returns a list of directed edges of the Hasse Diagram of the induced partial order on the rows.
     *
     * @return array<index>
     */
    array<index> get_row_graph() {
        GRLINA_DEBUG_CHECK(if (!compatibly_sorted)
            throw std::invalid_argument("get_row_graph requires compatibly sorted rows and columns"));
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
        SparseBase::delete_columns(indices);
        vec_deletion(col_degrees, indices);
        this->invalidate_cached_rows();
    }

    /**
     * @brief Graded version of row deletion.
     *
     * @param indices
     */
    void delete_rows(vec<index>& indices) {
        SparseBase::delete_rows(indices);
        vec_deletion(row_degrees, indices);
        this->invalidate_cached_rows();
    }


    /**
     * @brief Sorts the columns lexicographically by degree,
     *  using a pointer which points two both the column degrees and the data.
     *
     */
    void sort_columns_lexicographically_with_pointers() {
        sort_columns(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
    }

    /** Sort columns by an arbitrary compatible linear order. */
    template <typename Compare>
    void sort_columns(Compare compare) {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->col_degrees, compare);
        this->permute_columns(permutation);
        this->invalidate_cached_rows();

        compatibly_sorted = row_degrees.size() <= 1;
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

        compatibly_sorted = col_degrees.size() <= 1;
    }

    /** Sort rows and columns by the same compatible linear order. */
    template <typename Compare>
    void sort_compatibly(Compare compare) {
        auto columns = sort_and_get_permutation<D, index>(col_degrees, compare);
        this->permute_columns(columns);
        auto rows = sort_and_get_permutation<D, index>(row_degrees, compare);
        vec<index> old_to_new(rows.size());
        for (index i = 0; i < this->num_rows; ++i) old_to_new[rows[i]] = i;
        this->transform_data(old_to_new);
        this->sort_data();
        this->invalidate_cached_rows();
        compatibly_sorted = true;
    }

    /** Sort with the mandatory linear order supplied by Degree_traits. */
    void sort_compatibly() {
        sort_compatibly(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
    }

    struct SortingPermutation {
        vec<index> old_to_new;
        vec<index> new_to_old;
    };

    /** Unambiguous alternative to the historical *_with_output conventions. */
    template <typename Compare>
    SortingPermutation sort_columns_with_permutation(Compare compare) {
        auto degrees = col_degrees;
        auto new_to_old = sort_and_get_permutation<D, index>(degrees, compare);
        vec<index> old_to_new(new_to_old.size());
        for (index i = 0; i < static_cast<index>(new_to_old.size()); ++i)
            old_to_new[new_to_old[i]] = i;
        sort_columns(compare);
        return {std::move(old_to_new), std::move(new_to_old)};
    }
    SortingPermutation sort_columns_with_permutation() {
        return sort_columns_with_permutation(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
    }
    template <typename Compare>
    SortingPermutation sort_rows_with_permutation(Compare compare) {
        auto degrees = row_degrees;
        auto new_to_old = sort_and_get_permutation<D, index>(degrees, compare);
        vec<index> old_to_new(new_to_old.size());
        for (index i = 0; i < static_cast<index>(new_to_old.size()); ++i)
            old_to_new[new_to_old[i]] = i;
        permute_rows_graded(old_to_new);

        compatibly_sorted = col_degrees.size() <= 1;
        return {std::move(old_to_new), std::move(new_to_old)};
    }
    SortingPermutation sort_rows_with_permutation() {
        return sort_rows_with_permutation(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
    }

    /**
     * @brief Sorts the columns lexicographically by degree,
     * saves the permutation used to do so and applies it to the data.
     *
     */
    void sort_columns_lexicographically() {
        sort_columns(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
    }

    /**
     * @brief Sorts the columns lexicographically by degree,
     * saves the permutation used to do so, applies it also on the data and returns its inverse.
     *
     * @return vec<index>
     */
    vec<index> sort_columns_lexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<D, index>(this->col_degrees, Degree_traits<D>::lex_lambda());
        this->permute_columns(permutation);
        this->invalidate_cached_rows();
        vec<index> reverse = vec<index>(permutation.size());
        for (int i = 0; i < permutation.size(); ++i) {
            reverse[permutation[i]] = i;
        }

        compatibly_sorted = row_degrees.size() <= 1;
        return reverse;
    }

    /**
     * @brief Sorts the rows lexicographically by degree, then transforms the data accordingly.
     *
     */
    void sort_rows_lexicographically(){
        sort_rows(TraitLinearOrder<D>{Degree_traits<D>::lex_lambda()});
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

        compatibly_sorted = col_degrees.size() <= 1;
        return permutation;
    }

    void permute_rows_graded(const vec<index>& permutation) {
        GRLINA_ASSERT(permutation.size() == this->get_num_rows());
        this->transform_data(permutation);
        this->sort_data();
        vec<D> new_row_degrees(this->get_num_rows());
        for(index i = 0; i < (index)permutation.size(); i++) {
            new_row_degrees[permutation[i]] = this->row_degrees[i];
        }
        this->row_degrees = new_row_degrees;
        this->invalidate_cached_rows();
        this->compatibly_sorted = false;
    }

    DERIVED restricted_domain_copy(vec<index>& colIndices) const {
        DERIVED result( this->SparseBase::restricted_domain_copy(colIndices) );
        result.col_degrees = vec<D>(colIndices.size());
        for(index i = 0; i < (index)colIndices.size(); i++) {
            result.col_degrees[i] = this->col_degrees[colIndices[i]];
        }
        result.row_degrees = this->row_degrees;
        result.compatibly_sorted = false;
        return result;
    }

    /**
     * @brief Outputs the lists of generators and relations
     *
     */
    void print_degrees() const {
        static_assert(has_degree_print_v<D>, "Degree printing requires Degree_traits::print_degree(degree)");
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

    /** Cancel same-degree pairs by local pivot reduction and batch compaction.
     * An optional matrix of elements in the ambient generators is transported
     * through every cancellation, preserving its column degrees and order.
     * Compatibly sorted matrices use column tails directly. Unsorted matrices
     * use a temporary row ordering, preserving the order of surviving bases.
     * This is valid for every degree poset and needs no kernel algorithm.
     */
    void cancel_local_pairs(DERIVED* elements = nullptr) {
        cancel_local_pairs(elements, nullptr, [](auto&, auto&) {});
    }

    /** Also transport source basis images and report paired indices once.
     * source_basis has one column per source generator, in a fixed ambient
     * basis. on_cancel receives sorted original column and row indices after
     * compaction, so adjacent differentials can delete them in a single pass.
     */
    template<class OnCancellation>
    void cancel_local_pairs(DERIVED* elements, DERIVED* source_basis, OnCancellation on_cancel) {

        GRLINA_DEBUG_CHECK(if (elements && (elements == static_cast<DERIVED*>(this) ||
            elements->get_num_rows() != this->num_rows || elements->row_degrees != row_degrees))
            throw std::invalid_argument("Cancellation requires a separate matrix with the same ambient generators"));
        GRLINA_DEBUG_CHECK(if (source_basis && (source_basis == static_cast<DERIVED*>(this) ||
            source_basis == elements || source_basis->get_num_cols() != this->num_cols ||
            source_basis->col_degrees != col_degrees))
            throw std::invalid_argument("Cancellation requires separate source images with the same source generators"));
        if (this->num_cols == 0 || this->num_rows == 0) return;
        vec<index> row_order, row_rank;
        if (!compatibly_sorted) {
            auto degrees = row_degrees;
            row_order = sort_and_get_permutation<D, index>(degrees, Degree_traits<D>::lex_lambda());
            row_rank.resize(this->num_rows);
            for (index i = 0; i < this->num_rows; ++i) row_rank[row_order[i]] = i;
        }
        auto ordered_column = [&](const auto& matrix, index c) {
            auto column = matrix.get_col(c);
            if (!row_rank.empty()) {
                for (auto& r : column) r = row_rank[r];
                std::sort(column.begin(), column.end());
            }
            return column;
        };
        vec<index> local_pivots(this->num_rows, -1), removed_columns, removed_rows;
        array<index> reducers, reducer_images;
        // Fixed reducers always lower the ordered row pivot; reuse scratch space.
        auto add_column = [](const vec<index>& reducer, vec<index>& column, vec<index>& scratch) {
            scratch.resize(reducer.size() + column.size());
            auto end = std::set_symmetric_difference(reducer.begin(), reducer.end(),
                column.begin(), column.end(), scratch.begin());
            scratch.erase(end, scratch.end());
            column.swap(scratch);
        };
        vec<index> scratch, image_scratch;
        for (index c = 0; c < this->num_cols; ++c) {
            index p = this->col_last(c);
            if (!row_rank.empty()) {
                p = -1;
                for (index r : this->data[c]) p = std::max(p, row_rank[r]);
            }
            if (p == -1 || !Degree_traits<D>::equals(col_degrees[c],
                    row_degrees[row_order.empty() ? p : row_order[p]])) continue;
            auto column = ordered_column(*this, c);
            auto image = source_basis ? source_basis->get_col(c) : vec<index>{};
            while (!column.empty()) {
                p = column.back();
                if (!Degree_traits<D>::equals(col_degrees[c],
                        row_degrees[row_order.empty() ? p : row_order[p]])) break;
                if (local_pivots[p] == -1) {
                    local_pivots[p] = static_cast<index>(reducers.size());
                    reducers.push_back(std::move(column));
                    if (source_basis) reducer_images.push_back(std::move(image));
                    removed_columns.push_back(c);
                    break;
                }
                add_column(reducers[local_pivots[p]], column, scratch);
                if (source_basis) add_column(reducer_images[local_pivots[p]], image, image_scratch);
            }
        }
        if (removed_columns.empty()) return;

        // Keep original row order; remap surviving entries during compression.
        vec<index> row_map(this->num_rows, -1);
        vec<D> surviving_degrees;
        surviving_degrees.reserve(this->num_rows - reducers.size());
        for (index r = 0; r < this->num_rows; ++r) {
            index p = row_rank.empty() ? r : row_rank[r];
            if (local_pivots[p] == -1) {
                row_map[p] = static_cast<index>(surviving_degrees.size());
                surviving_degrees.push_back(row_degrees[r]);
            } else removed_rows.push_back(r);
        }
        auto compress_column = [&](const auto& matrix, index c, vec<index>* image = nullptr) {
            auto column = ordered_column(matrix, c);
            vec<index> result, scratch;
            while (!column.empty()) {
                index p = column.back();
                if (local_pivots[p] != -1) {
                    add_column(reducers[local_pivots[p]], column, scratch);
                    if (image) add_column(reducer_images[local_pivots[p]], *image, image_scratch);
                } else {
                    result.push_back(row_map[p]);
                    column.pop_back();
                }
            }
            if (row_rank.empty()) std::reverse(result.begin(), result.end());
            else std::sort(result.begin(), result.end());
            return result;
        };
        array<index> columns, images;
        columns.reserve(this->num_cols - removed_columns.size());
        if (source_basis) images.reserve(this->num_cols - removed_columns.size());
        auto removed = removed_columns.begin();
        for (index c = 0; c < this->num_cols; ++c) {
            if (removed != removed_columns.end() && c == *removed) { ++removed; continue; }
            auto image = source_basis ? source_basis->get_col(c) : vec<index>{};
            columns.push_back(compress_column(*this, c, source_basis ? &image : nullptr));
            if (source_basis) images.push_back(std::move(image));
        }
        if (elements) {
            array<index> transported;
            transported.reserve(elements->get_num_cols());
            for (index c = 0; c < elements->get_num_cols(); ++c)
                transported.push_back(compress_column(*elements, c));
            elements->assign_data(std::move(transported));
            elements->row_degrees = surviving_degrees;
            elements->set_num_rows(static_cast<index>(surviving_degrees.size()));
            elements->pivots.clear();
            elements->invalidate_cached_rows();
        }
        if (source_basis) {
            source_basis->assign_data(std::move(images));
            vec_deletion(source_basis->col_degrees, removed_columns);
            source_basis->set_num_cols(static_cast<index>(source_basis->col_degrees.size()));
            source_basis->pivots.clear();
            source_basis->invalidate_cached_rows();
        }
        this->assign_data(std::move(columns));
        vec_deletion(col_degrees, removed_columns);
        row_degrees = std::move(surviving_degrees);
        this->num_cols = static_cast<index>(col_degrees.size());
        this->num_rows = static_cast<index>(row_degrees.size());
        this->pivots.clear();
        this->invalidate_cached_rows();
        on_cancel(removed_columns, removed_rows);
    }

    /** Compute a kernel of this graded matrix and quotient out everything possible while preserving the image.
     */
    void remove_redundant_relations() {
        remove_redundant_relations([](auto, const auto&) {});
    }
    template<class OnDelete>
    void remove_redundant_relations(OnDelete on_delete) {

        if constexpr (has_matrix_graded_kernel<DERIVED>::value) {
            DERIVED source = static_cast<const DERIVED&>(*this);
            DERIVED syzygies = source.graded_kernel();
            while (true) {
                index c = -1, r = -1;
                for (index j = 0; j < syzygies.get_num_cols() && c == -1; ++j)
                    for (index i : syzygies.data[j])
                        if (Degree_traits<D>::equals(syzygies.col_degrees[j], syzygies.row_degrees[i])) {
                            c = j; r = i; break;
                        }
                if (c == -1) break;
                for (index j = 0; j < syzygies.get_num_cols(); ++j)
                    if (j != c && std::binary_search(syzygies.data[j].begin(), syzygies.data[j].end(), r))
                        syzygies.col_op(c, j);
                on_delete(r, syzygies.get_col(c));
                vec<index> redundant{r}, syzygy{c};
                delete_columns(redundant);
                syzygies.delete_rows(redundant);
                syzygies.delete_columns(syzygy);
            }
        } else {
            // Jan: implement the poset-specific graded_kernel() returning DERIVED.
            // Local cancellation is available as semi_minimize() in the meantime.
            throw std::logic_error("Relation minimization requires a poset-specific graded_kernel");
        }
    }

    /** Standard minimization: local cancellation, then graded-kernel relations. */
    void minimize() {
        if constexpr (!has_matrix_graded_kernel<DERIVED>::value) {
            // Jan: supply the graded kernel to enable full presentation minimization.
            throw std::logic_error("minimize requires a poset-specific graded_kernel; use semi_minimize for local pairs");
        } else {
            cancel_local_pairs();
            remove_redundant_relations();
        }
    }

    /** Optional cheap reduction before the same complete minimization. */
    void minimize_variant() {
        GRLINA_DEBUG_CHECK(if (!compatibly_sorted)
            throw std::invalid_argument("minimize_variant requires compatibly sorted rows and columns"));
        if constexpr (!has_matrix_graded_kernel<DERIVED>::value) {
            throw std::logic_error("minimize_variant requires a poset-specific graded_kernel");
        } else {
            cancel_local_pairs();
            column_reduction_graded_w_deletion();
            remove_redundant_relations();
        }
    }

    void shift (D d){
        static_assert(has_degree_subtract_v<D>, "Matrix shifts require Degree_traits::subtract(amount, degree)");
        for(index i = 0; i < this->num_cols; i++){
            Degree_traits<D>::subtract(d, this->col_degrees[i]);
        }
        for(index i = 0; i < this->num_rows; i++){
            Degree_traits<D>::subtract(d, this->row_degrees[i]);
        }
    }

    void shift_generators (D d){
        static_assert(has_degree_subtract_v<D>, "Generator shifts require Degree_traits::subtract(amount, degree)");
        for(index i = 0; i < this->num_rows; i++){
            Degree_traits<D>::subtract(d, this->row_degrees[i]);
        }
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
        GRLINA_DEBUG_CHECK(if (!compatibly_sorted)
            throw std::invalid_argument("column_reduction_graded requires compatibly sorted rows and columns"));
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
        static_assert(has_degree_join_v<D>, "Adjusting relation degrees requires Degree_traits::join(a, b)");
        this->compatibly_sorted = false;
        for(index i = 0; i < this->get_num_rows(); i++){
            this->row_degrees[i] = d;
        }
        for(index i = 0; i < this->get_num_cols(); i++){
            if( ! Degree_traits<D>::smaller_equal(d, this->col_degrees[i]) ){
                this->col_degrees[i] = Degree_traits<D>::join(d, this->col_degrees[i]);
            }
        }
    }
    
    /** Backwards-compatible name for local pair cancellation. */
    void semi_minimize(){
        this->cancel_local_pairs();
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
        this->append_col(column_data);
        this->col_degrees.push_back(column_degree);
        this->num_cols += 1;
        this->invalidate_cached_rows();
        this->compatibly_sorted = false;
    }

     /**
     * @brief Appends the columns and column degrees of another graded matrix.
     *
     * @param other
     */
    void append_matrix(const GradedSparseMatrix& other) {
        GRLINA_ASSERT(this->num_rows == other.num_rows);
        GRLINA_ASSERT(this->row_degrees == other.row_degrees);
        for(index i = 0; i < other.num_cols; i++) {
            this->append_col(other.get_col(i));
            this->col_degrees.push_back(other.col_degrees[i]);
        }
        this->num_cols += other.num_cols;
        this->invalidate_cached_rows();
        this->compatibly_sorted = false;
    }

    void append_move_matrix(GradedSparseMatrix&& other) {
        GRLINA_ASSERT(this->num_rows == other.num_rows);
        for(index i = 0; i < other.num_cols; i++) {
            this->append_col(other.take_col(i));
            this->col_degrees.push_back(std::move(other.col_degrees[i]));
        }
        this->num_cols += other.num_cols;
        this->invalidate_cached_rows();
        this->compatibly_sorted = false;
    }

    /**
     * @brief Computes a presentation for the quotient by a submodule.
     * The submodule is given by a graded matrix denoting a map from the generators of the submodule to the generators of this module.
     * This method appends the input matrix to the current matrix and minimises it.
     *
     * @param Y
     */
    void quotient_by (GradedSparseMatrix<D, index, DERIVED, Storage>& Y) {
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
        
        // Keep the original domain basis: sorting these rows would invalidate
        // the returned coordinates. Full generator minimization is optional.
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
        // Preserve the original domain coordinates, even when unsorted.
        // TO-DO: could also fully minimize if we want to?
        return K;
    }


    DERIVED submodule_intersection(const DERIVED& l, const DERIVED& r) const {
        GRLINA_ASSERT(l.row_degrees == r.row_degrees);
        auto k = l.inverse_image_copy(static_cast<const DERIVED&>(*this), r);
        auto i = l*k;
        // Rows are still the parent's fixed generator basis.
        return i;
    };

    /**
     * @brief Returns a presentation of the submodule generated by the input, if *this is a presentation.
     *  Does not minimize!
     * @param new_generators
     * @return DERIVED
     */
    DERIVED submodule_generated_by(const DERIVED& new_generators) const {
        GRLINA_ASSERT(this->get_num_rows() == new_generators.get_num_rows());
        GRLINA_ASSERT(this->row_degrees == new_generators.row_degrees);
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
        GRLINA_ASSERT(this->get_num_rows() == M.get_num_rows());
        GRLINA_ASSERT(this->row_degrees == M.row_degrees);
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
        vec<index> remove;
        auto keep = cs.begin();
        for (index j = 0; j < this->get_num_cols(); ++j) {
            if (keep != cs.end() && *keep == j) ++keep;
            else remove.push_back(j);
        }
        this->delete_columns(remove);
    }

    void delete_all_but_columns_alt(vec<index> cs) {
        delete_all_but_columns(std::move(cs));
    }

    DERIVED transposed_copy() const {
        DERIVED result(SparseBase::transposed_copy());
        result.col_degrees = this->row_degrees;
        result.row_degrees = this->col_degrees;

        result.compatibly_sorted = this->compatibly_sorted;

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
        GRLINA_ASSERT(i >= 0 && i < target.size() && "Index out of bounds");
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
    GRLINA_ASSERT(itS == subset.end() && "Not all elements of the subset were found in the target");
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
        // Probably this doesnt need to be its own function
        vec<index> result = GradedSparseMatrix::getIndicatorVector(target.first, GradedSparseMatrix::extractElements(source.first, source.second));
        return result;
    }

	/**
	 * @brief This function computes a quiver representation on the poset of unique degrees 
     * appearing for the columns and rows of the matrix.
	 */
	QuiverRepresentation<index, D, SparseBase> induced_quiver_rep(
        vec<D> vertices = vec<D>(), array<index> edges = array<index>()) {

        if (vertices.empty() && edges.empty()) vertices = discrete_support();
        if (edges.empty() && !vertices.empty()) {
            edges = minimal_directed_graph<D, index>(vertices);
        }
        GRLINA_DEBUG_CHECK(if (vertices.size() != edges.size())
            throw std::invalid_argument("Quiver adjacency must match its vertices");
        for (index i = 0; i < static_cast<index>(vertices.size()); ++i)
            for (index j : edges[i])
                if (j < 0 || j >= static_cast<index>(vertices.size()) ||
                    !Degree_traits<D>::smaller_equal(vertices[i], vertices[j]))
                    throw std::invalid_argument("Quiver edge must follow the degree order"));
		
        QuiverRepresentation<index, D, SparseBase> rep;
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
		vec< SparseBase > pointwise_Presentations;
        vec< std::pair< vec<index> , vec<index>> > pointwise_base;
		pointwise_Presentations.reserve(num_vert);
        pointwise_base.reserve(num_vert);
		rep.dimensionVector.reserve(num_vert);

        // #pragma omp parallel for
		for (index i = 0; i < num_vert; i++) {

            SparseBase S;
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

            GRLINA_ASSERT(pointwise_Presentations[source].get_num_cols() == sourceBasis.first.size());
            GRLINA_ASSERT(pointwise_Presentations[target].get_num_cols() == targetBasis.first.size());

            vec<index> lift_of_basis = GradedSparseMatrix::basisLifting(sourceBasis, targetBasis);
            rep.matrices.emplace_back( pointwise_Presentations[target].restricted_domain_copy(lift_of_basis) );
        }
    
        GRLINA_DEBUG_CHECK(for (index j = 0; j < num_edges; j++) {
            if(rep.matrices[j].get_num_cols() != rep.dimensionVector[rep.edges[j].first] || rep.matrices[j].get_num_rows() != rep.dimensionVector[rep.edges[j].second]){
                throw std::runtime_error("Dimension mismatch in path action");
            }
        });
        return rep;
	}

}; // GradedSparseMatrix


template <typename D, typename index, typename DERIVED, typename Storage>
DERIVED operator*(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B) {
    using SparseBase = SparseMatrix<index, Storage>;
    GRLINA_ASSERT(A.col_degrees == B.row_degrees);
    SparseBase product = static_cast<const SparseBase&>(A) * static_cast<const SparseBase&>(B);
    DERIVED result(std::move(product));
    result.row_degrees = A.row_degrees;
    result.col_degrees = B.col_degrees;
    result.compatibly_sorted = false;
    return result;
}



template <typename D, typename DERIVED>
DERIVED shifted_identity( vec<D>& generators, const D& epsilon) {
    static_assert(is_degree_v<D>, "D must implement the Degree_traits interface");
    static_assert(has_degree_add_v<D>, "Shifted identities require Degree_traits::add(amount, degree)");
    DERIVED result(generators.size(),generators.size(), "Identity");
    result.col_degrees = generators;
    result.row_degrees = generators;
    for( D& d : result.col_degrees ) {
        Degree_traits<D>::add(epsilon, d);
    }
    result.compatibly_sorted = false;
    return result;
}

/**
 * @brief Compares two graded matrices by their degrees.
 *
 * @tparam D
 * @tparam index
 */
template <typename D, typename index, typename DERIVED, typename Storage = typename DERIVED::storage_type>
struct Compare_by_degrees {
    static_assert(is_degree_v<D>, "D must implement the Degree_traits interface");

    /**
     * @brief -1 if a<b, 0 if a=b, 1 if a>b
     *
     * @param a
     * @param b
     * @return int
     */
    static int compare_three_way(const GradedSparseMatrix<D, index, DERIVED, Storage>& a, const GradedSparseMatrix<D, index, DERIVED, Storage>& b) {
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

    bool operator()(const GradedSparseMatrix<D, index, DERIVED, Storage>& a, const GradedSparseMatrix<D, index, DERIVED, Storage>& b) const {
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

#endif // GRLINA_MAT_GRADED_HPP
