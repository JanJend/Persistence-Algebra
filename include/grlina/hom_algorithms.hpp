/**
 * @file hom_algorithms.hpp
 * @author Jan Jendrysiak
 * @brief Full and optimised Hom-space algorithms for persistence presentations.
 * @version 2.0.0
 * @date 2026-02-05
 * 
 * @copyright 2026 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once
#include <grlina/general_checks.hpp>


#ifndef GRLINA_HOM_ALGORITHMS_HPP
#define GRLINA_HOM_ALGORITHMS_HPP

#include <grlina/matrices/graded.hpp>
#include <grlina/graded_linear_system.hpp>
#include <grlina/matrices/grid.hpp>
#include "grlina/matrices/r2.hpp"
#include "grlina/matrices/sparse.hpp"
#include <cstdlib>
#include <map>
#include <utility>
#include <random>
#include <vector>
#include <boost/timer/timer.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

/** Algorithms A-1/2, A and B of https://arxiv.org/abs/2604.10381. */
enum class HomAlgorithm {
    optimised,        // A-1/2: restrict Q using local target cokernels.
    full_restriction, // A: also restrict P using the target's graded kernel.
    hom_exactness    // B: use exactness of Hom(-, target).
};

inline const char* hom_algorithm_name(HomAlgorithm algorithm) {
    switch (algorithm) {
    case HomAlgorithm::optimised: return "optimised";
    case HomAlgorithm::full_restriction: return "full-restriction";
    case HomAlgorithm::hom_exactness: return "hom-exactness";
    }
    throw std::invalid_argument("Unknown Hom algorithm");
}

template<typename index>
using Hom_space_temp = std::pair< SparseMatrix<index>, vec<std::pair<index,index>> >;


/** Optional check for manually supplied f0; trusted constructors need not call it.
 * Returns f1 with target_presentation*f1 = f0*source_presentation, or nullopt.
 */
template <typename Matrix>
std::optional<Matrix> lift_to_relations(const Matrix& source_presentation,
                                      const Matrix& target_presentation,
                                      const Matrix& f0) {
    GRLINA_DEBUG_CHECK(if (geometric_col_degrees(f0) != geometric_row_degrees(source_presentation) ||
            geometric_row_degrees(f0) != geometric_row_degrees(target_presentation))
            throw std::invalid_argument("Generator lift has incompatible source or target bases"));
    return solve_graded_linear_system(target_presentation, f0 * source_presentation);
}

// Explicit test diagnostic; never called during normal computation.
template <typename Matrix>
bool is_homomorphism(const Matrix& source_presentation,
                    const Matrix& target_presentation, const Matrix& f0) {
    return lift_to_relations(source_presentation, target_presentation, f0).has_value();
}

    template <typename index>
double avg_entries_per_column(const SparseMatrix<index>& S){
    double total_entries = 0.0;
    for(const auto& col : S.data){
        total_entries += col.size();
    }
    return total_entries / (double) S.get_num_cols();
}

template <typename index>
void system_stats(boost::timer::cpu_timer& timer, std::string alg_name, SparseMatrix<index>& S, const int rows = -1){
    std::cout << alg_name << ":" << std::endl;
    timer.stop();
    std::cout << "  Time to set up: " << timer.elapsed().wall * 1e-6 << "ms " << std::endl;
    std::cout << "  System size: " << S.get_num_cols() << " cols and ";
    if(rows == -1){
        std::cout << S.get_num_rows() << " rows. " << std::endl;
    } else {
        std::cout << rows << " rows." << std::endl;
    }
    std::cout << "  Average #entries per column: " << avg_entries_per_column(S) << std::endl;
    timer.start();
}

template <typename index>
vec<index>index_pair_to_position(index row_index,
    index j_start, 
    const vec<index>& B_data_j, 
    const vec<std::pair<index,index>>& variable_positions) {

    vec<index> result;
    auto it = B_data_j.begin();
    for(index j = j_start; j < variable_positions.size() && it != B_data_j.end(); j++){
        auto& index_pair = variable_positions[j];
        if(index_pair.first == row_index){
            if(*it == index_pair.second){
                result.push_back(j);
                it++;
            }
        }
    }
    return result;
}

/**
 * @brief Returns the basis of Hom(A, B) as a vector of graded matrices.
 *
 * @tparam D
 * @tparam index
 * @tparam DERIVED
 * @param A
 * @param B
 * @return vec<DERIVED>
 */
template <typename D, typename index, typename DERIVED, typename Storage>
vec<DERIVED> hom_space_basis_new(
    const GradedSparseMatrix<D, index, DERIVED, Storage>& A,
    const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    bool use_hom_exactness = false,
    const bool info = false,
    const bool reduce_lift_duplicates = false,
    const std::size_t max_basis_size = 100000, // Zero permits any dimension; caller enforces resource limits.
    Hom_space_temp<index>* coefficient_basis = nullptr // Exactness kernel without expanding its maps.
    ) {
    if (coefficient_basis && !use_hom_exactness)
        throw std::invalid_argument("Compressed legacy Hom output requires the exactness algorithm");

    if constexpr (matrix_grid_backed_v<DERIVED>) {
        const auto& grid_A = static_cast<const DERIVED&>(A);
        const auto& grid_B = static_cast<const DERIVED&>(B);
        auto source = grid_A.to_coordinates();
        auto target = grid_B.to_coordinates();
        source.compute_rows_forward();
        auto lifts = hom_space_basis_new(source, target, use_hom_exactness, info,
                                         reduce_lift_duplicates, max_basis_size, coefficient_basis);
        auto axes = grid_A.grid_;
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            axes[axis].insert(axes[axis].end(), grid_B.grid_[axis].begin(), grid_B.grid_[axis].end());
        vec<DERIVED> result;
        result.reserve(lifts.size());
        for (const auto& lift : lifts) {
            DERIVED embedded(lift);
            embedded.reindex_grid(axes);
            result.push_back(std::move(embedded));
        }
        return result;
    } else {
    boost::timer::cpu_timer timer;
    if(info){
        timer.start();
    }
    vec<DERIVED> result = vec<DERIVED>();
    GRLINA_DEBUG_CHECK(if(!A.rows_computed){
        std::cerr << "Warning: Rows of A must be computed before usage." << std::endl;
        std::abort();
    });

    if(! use_hom_exactness){

    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    vec<index> variable_positions_separator = vec<index>(A.get_num_rows(), 0);
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        variable_positions_separator[i] = S_index;
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            }
        }
    }

    index row_op_threshold = S_index;
    GRLINA_ASSERT( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return result;
    }


    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            }
        }
    }

    

    S.compute_num_cols();
    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();
    K.column_reduction_triangular(true);

    if(reduce_lift_duplicates){
        SparseMatrix<index> N_bar = SparseMatrix<index>(0,K.get_num_cols());
        for(index i = 0; i < A.get_num_rows(); i++){
            for(index j = 0; j < B.get_num_cols(); j++){
                if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.col_degrees[j])){
                    // Add a new homotopy
                    vec<index> h = index_pair_to_position(i,variable_positions_separator[i], B.get_col(j), variable_positions);
                    N_bar.data.push_back(h);
                }
            }
        }
        N_bar.compute_num_cols();
        N_bar.reduce_fully(K);
        K.column_reduction_triangular(true);
    }

    for(index i = 0; i < K.get_num_cols(); i++){
        auto& current_map = K.data[i];
        DERIVED Q(A.get_num_rows(), B.get_num_rows(),
            array<index>(A.get_num_rows()),
            A.row_degrees, B.row_degrees);
        for(index j : current_map){
            auto& index_pairs = variable_positions[j];
            Q.append_entry(index_pairs.first, index_pairs.second);
        }
        GRLINA_ASSERT(Q.is_sorted_sparse());
        result.push_back(std::move(Q));
    } 

    //  If the bool is true, next use Hom( -, coker B) on the presentation A to get Hom(coker A, coker B)_0 as the kernel of 
    //  Hom(A.target, coker B)_0 -(A^t)-> Hom(A.domain, coker B)_0
    } else {

        vec<SparseMatrix<index, Storage>> B_local_spaces;
        B_local_spaces.reserve(A.get_num_rows() + A.get_num_cols());
        vec< std::pair<vec<index>, vec<index>> > B_local_basislifts =  vec< std::pair<vec<index>, vec<index>> >();
        B_local_basislifts.reserve(A.get_num_rows() + A.get_num_cols());

        // Setting up all structure maps for coker B
        array<index> S_column_partition = array<index>();
        index S_num_col = 0;
        for(size_t p = 0; p < A.get_num_rows() + A.get_num_cols(); p++){
            SparseMatrix<index, Storage> presentation;
            vec<index> gens; // the set of generators which appear before the selected degree.
            if(p < A.get_num_rows()){
                std::tie(presentation, gens) = B.map_at_degree_pair(A.row_degrees[p]);
            } else {
                std::tie(presentation, gens) = B.map_at_degree_pair(A.col_degrees[p - A.get_num_rows()]);             
            }
            vec<index> basisLift; // Will contain a set of indices, which indicates a minimal set of (local!) generators that maps to a basis of the local vector space
            B_local_spaces.emplace_back(presentation.coKernel(false, &basisLift));
            B_local_basislifts.push_back(std::make_pair(gens, basisLift));
            if(p < A.get_num_rows()){
                S_num_col += basisLift.size();
            }
        }
        SparseMatrix<index> S(S_num_col, 0, array<index>(S_num_col, vec<index>()));
        index S_column_counter = 0;
        index S_row_counter = 0;
        
        // building the matrix A^t \otimes B_*
        for(size_t i = 0; i < A.get_num_rows(); i++){
            S_row_counter = 0;
            std::pair< vec<index>, vec<index> >& sourceBasis = B_local_basislifts[i];
            GRLINA_ASSERT(B_local_spaces[i].get_num_cols() == sourceBasis.first.size());
            GRLINA_ASSERT(B_local_spaces[i].get_num_rows() == sourceBasis.second.size());
            auto itA = A._rows[i].begin();
            // The following will store the global indices corresponding to a basis of Y_(deg g_i)
            S_column_partition.push_back(vec_restriction(sourceBasis.first, sourceBasis.second));
            for(size_t j = 0; j < A.get_num_cols(); j++){
                if(itA == A._rows[i].end()){
                    break;
                } else if (*itA > j) {
                    index j_shift = A.get_num_rows()+j;
				    auto& targetBasis = B_local_basislifts[j_shift];
                    S_row_counter += targetBasis.second.size();
                    continue;
                } else {
                GRLINA_ASSERT(*itA == j);
                itA++;
                index j_shift = A.get_num_rows()+j;
				auto& targetBasis = B_local_basislifts[j_shift];
                auto& targetSpace = B_local_spaces[j_shift];
				GRLINA_ASSERT(targetSpace.get_num_cols() == targetBasis.first.size());
                GRLINA_ASSERT(targetSpace.get_num_rows() == targetBasis.second.size());
				vec<index> image_of_basis_indices = get_index_vector<index, index>(targetBasis.first, S_column_partition[i]);
                // ^ First computes the actual indices of the genertors which form a basis of the source vector space.
                // Then computes which subset of the generators of the target this corresponds to.
                
                GRLINA_ASSERT(image_of_basis_indices.size() == sourceBasis.second.size());
				// auto local_map = targetSpace.restricted_domain_copy(image_of_basis_indices);
                // The map (coker B)_{gen[i]->rel[j]} is now given by targetSpace restricted to image_of_basis_indices
                // Since we need to translate this in S, add column and row counter
                for(index k = 0; k < image_of_basis_indices.size(); k++){
                    auto& basis_vector = image_of_basis_indices[k];
                    for(index entry : targetSpace.data[basis_vector]){
                        S.data[S_column_counter + k].push_back(entry+S_row_counter);
                    }
                }
                S_row_counter += targetBasis.second.size();
                }
            }
            S_column_counter += S_column_partition[i].size();
        }
        S.set_num_rows(S_row_counter);

        if(info){
            system_stats(timer, "Alg_B", S);
        }

        SparseMatrix<index> K = S.kernel();

        if(info){
            timer.stop();
            std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
            // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
        }

        if (coefficient_basis) {
            vec<std::pair<index, index>> positions;
            for (index c = 0; c < A.get_num_rows(); ++c)
                for (index r : S_column_partition[c]) positions.emplace_back(c, r);
            *coefficient_basis = {std::move(K), std::move(positions)};
            return result;
        }
        if(max_basis_size && std::size_t(K.get_num_cols()) > max_basis_size){
            std::cout << " Hom-space is too large (" << K.get_num_cols() << ") to convert to a vector of matrices." << std::endl;
            return result;
        }
        // Translating the vectors to matrices.
        // TO-DO: The following could easily be parallelised
        for(auto f_vec : K.data){
            DERIVED new_Q(A.get_num_rows(), B.get_num_rows(), A.row_degrees, B.row_degrees);
            result.emplace_back(std::move(new_Q));
            auto& Q = result.back();
            Q.resize_data(A.get_num_rows());
            auto it = f_vec.begin();
            index column_counter = 0;
            // We need to advance the iterator for the next block of size S_column_partition[i]
            for(index i = 0; i < A.get_num_rows(); i++){
                index block_end = column_counter + S_column_partition[i].size();
                while(it != f_vec.end() && *it < block_end){
                    Q.append_entry(i, S_column_partition[i][*it - column_counter]);
                    it++;
                }
                if( it == f_vec.end()){
                    break;
                }
                column_counter += S_column_partition[i].size();
            }
        }
    }

    return result;
    }
}

/**
 * @brief Backwards-compatible name for the Hom-space basis routine.
 *
 * This historical entry point returns actual module morphisms, so generator
 * lifts that differ by a factorisation through target relations are removed.
 * Existing users of hom_space_basis_new retain its original default.
 */
template <typename D, typename index, typename DERIVED, typename Storage>
vec<DERIVED> hom_space_basis(
    const GradedSparseMatrix<D, index, DERIVED, Storage>& A,
    const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    bool use_hom_exactness = false,
    const bool info = false
    ) {
    return hom_space_basis_new(A, B, use_hom_exactness, info, true);
}

/**
 * @brief Returns the basis of Hom(A, B) as a vector of graded matrices.
 *
 * @tparam D
 * @tparam index
 * @tparam DERIVED
 * @param A
 * @param B
 * @return vec<DERIVED>
 */
template <typename D, typename index, typename DERIVED, typename Storage>
SparseMatrix<index> Alg_B_test(
    const GradedSparseMatrix<D, index, DERIVED, Storage>& A,
    const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const bool info = false
    ) {

    boost::timer::cpu_timer timer;
    if(info){
        timer.start();
    }
    vec<DERIVED> result = vec<DERIVED>();
    GRLINA_DEBUG_CHECK(if(!A.rows_computed){
        std::cerr << "Warning: Rows of A must be computed before usage." << std::endl;
        std::abort();
    });

        vec<SparseMatrix<index, Storage>> B_local_spaces;
        B_local_spaces.reserve(A.get_num_rows() + A.get_num_cols());
        vec< std::pair<vec<index>, vec<index>> > B_local_basislifts =  vec< std::pair<vec<index>, vec<index>> >();
        B_local_basislifts.reserve(A.get_num_rows() + A.get_num_cols());

        // Setting up all structure maps for coker B
        array<index> S_column_partition = array<index>();
        index S_num_col = 0;
        for(size_t p = 0; p < A.get_num_rows() + A.get_num_cols(); p++){
            SparseMatrix<index, Storage> presentation;
            vec<index> gens; // the set of generators which appear before the selected degree.
            if(p < A.get_num_rows()){
                std::tie(presentation, gens) = B.map_at_degree_pair(A.row_degrees[p]);
            } else {
                std::tie(presentation, gens) = B.map_at_degree_pair(A.col_degrees[p - A.get_num_rows()]);             
            }
            vec<index> basisLift; // Will contain a set of indices, which indicates a minimal set of (local!) generators that maps to a basis of the local vector space
            B_local_spaces.emplace_back(presentation.coKernel(false, &basisLift));
            B_local_basislifts.push_back(std::make_pair(gens, basisLift));
            if(p < A.get_num_rows()){
                S_num_col += basisLift.size();
            }
        }
        SparseMatrix<index> S(S_num_col, 0, array<index>(S_num_col, vec<index>()));
        index S_column_counter = 0;
        index S_row_counter = 0;
        
        // building the matrix A^t \otimes B_*
        for(size_t i = 0; i < A.get_num_rows(); i++){
            S_row_counter = 0;
            std::pair< vec<index>, vec<index> >& sourceBasis = B_local_basislifts[i];
            GRLINA_ASSERT(B_local_spaces[i].get_num_cols() == sourceBasis.first.size());
            GRLINA_ASSERT(B_local_spaces[i].get_num_rows() == sourceBasis.second.size());
            auto itA = A._rows[i].begin();
            // The following will store the global indices corresponding to a basis of Y_(deg g_i)
            S_column_partition.push_back(vec_restriction(sourceBasis.first, sourceBasis.second));
            for(size_t j = 0; j < A.get_num_cols(); j++){
                if(itA == A._rows[i].end()){
                    break;
                } else if (*itA > j) {
                    index j_shift = A.get_num_rows()+j;
				    auto& targetBasis = B_local_basislifts[j_shift];
                    S_row_counter += targetBasis.second.size();
                    continue;
                } else {
                GRLINA_ASSERT(*itA == j);
                itA++;
                index j_shift = A.get_num_rows()+j;
				auto& targetBasis = B_local_basislifts[j_shift];
                auto& targetSpace = B_local_spaces[j_shift];
				GRLINA_ASSERT(targetSpace.get_num_cols() == targetBasis.first.size());
                GRLINA_ASSERT(targetSpace.get_num_rows() == targetBasis.second.size());
				vec<index> image_of_basis_indices = get_index_vector<index, index>(targetBasis.first, S_column_partition[i]);
                // ^ First computes the actual indices of the genertors which form a basis of the source vector space.
                // Then computes which subset of the generators of the target this corresponds to.
                
                GRLINA_ASSERT(image_of_basis_indices.size() == sourceBasis.second.size());
				// auto local_map = targetSpace.restricted_domain_copy(image_of_basis_indices);
                // The map (coker B)_{gen[i]->rel[j]} is now given by targetSpace restricted to image_of_basis_indices
                // Since we need to translate this in S, add column and row counter
                for(index k = 0; k < image_of_basis_indices.size(); k++){
                    auto& basis_vector = image_of_basis_indices[k];
                    for(index entry : targetSpace.data[basis_vector]){
                        S.data[S_column_counter + k].push_back(entry+S_row_counter);
                    }
                }
                S_row_counter += targetBasis.second.size();
                }
            }
            S_column_counter += S_column_partition[i].size();
        }
        S.set_num_rows(S_row_counter);

        if(info){
            system_stats(timer, "Alg_B", S);
        }

        SparseMatrix<index> K = S.kernel();

        if(info){
            timer.stop();
            std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
            // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
        }

    return K;
}

    
/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 *  make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_optimised(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const vec<index>& row_indices_A = vec<index>(), const vec<index>& row_indices_B = vec<index>(),
    const bool info = false);


/**
 * @brief Algorithm A of "Homomorphisms of Persistence Modules"
 * Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 * make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_full_restriction(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const vec<index>& row_indices_A = vec<index>(), const vec<index>& row_indices_B = vec<index>(),
    const bool info = false)  {
    
    GRLINA_ASSERT(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();

    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    // TO-DO: Right now we compute map_at_degree possibly multiple times! This could be optimised.
    for(index i = 0; i < A.get_num_rows(); i++) {
        // Compute the target space B_alpha for each generator of A to minimise the number of variables.
        auto [B_alpha, rows_alpha] = B.map_at_degree_pair(A.row_degrees[i], false);
        vec<index> basislift;
        if(row_indices_B.size() != 0){
            basislift = B_alpha.coKernel_basis(rows_alpha, row_indices_B );
        } else {
            basislift = B_alpha.coKernel_basis_local(rows_alpha);
        }

        // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
    	    variable_positions.push_back(std::make_pair(i, j));
            for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                auto& column_index = *rit;
                S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
            }
            S_index++;
        }
    }
    
    index row_op_threshold = S_index;
    GRLINA_ASSERT( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }
    DERIVED B_copy = B;
    DERIVED O = B_copy.graded_kernel();
    // Then all column-operations from B to A, again restricted by the basislift
    for(index i = A.get_num_cols()-1; i > -1; i--){

        // Compute the target space O_alpha for each relation of A to minimise the number of variables.
        auto [O_alpha, rows_alpha] = O.map_at_degree_pair(A.col_degrees[i], false);
        vec<index> basislift = O_alpha.coKernel_basis_local(rows_alpha);
        
         // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
            for(index row_index : B.data[j]){
                if(row_indices_B.size() != 0){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                } else {
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));
                }
            }
            S_index++;
        }
    }

    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Alg_A", S, equation_counter);
    }


    auto K = S.kernel();

    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        timer.start();
    }

    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();

    return std::make_pair(K, variable_positions);
}

template <typename D, typename index, typename DERIVED, typename Storage>
vec<index> no_opt_system_info(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B){
    GRLINA_ASSERT(A.rows_computed);
    boost::timer::cpu_timer timer;
    
    timer.start();
    vec<index> result = vec<index>();

    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(Degree_traits<D>::greater_equal(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            } 
        }
    }
    
    index row_op_threshold = S_index;
    GRLINA_ASSERT( variable_positions.size() == S_index );


    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(B.is_admissible_column_operation(j, A.col_degrees[i])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                }
                S_index++;
            }
        }
    }


    S.compute_num_cols();


        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(Degree_traits<D>::greater_equal(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        result.push_back(S.get_num_cols());
        result.push_back(equation_counter);
        system_stats(timer, "Naive", S, equation_counter);
    return result;
}

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(A, B), where Q is a map on the generators. 
 *  make sure that the rows of A are computed.
 * if row_indices
 * @param A 
 * @param B 
 * @param row_indices If the row indices of B are shifted, this vector contains the shift.
 * @return vec<SparseMatrix<index>> 
 */
template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_no_opt(
    const GradedSparseMatrix<D, index, DERIVED, Storage>& A,
    const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const bool reduce = true,
    const vec<index>& row_indices_A = vec<index>(), 
    const vec<index>& row_indices_B = vec<index>(), 
    const bool info = false);

/**
 * @brief Returns a vector of matrices Q which form a basis of Hom(C, B), where Q is a map on the generators. 
 * C, B are both blocks in the large matrix A. Used in AIDA.
*/
template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index, index> > > block_hom_space_without_optimisation(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& C, const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
        vec<index>& C_rows, vec<index>& B_rows, bool system_size = false)  { 
    vec<std::pair<index, index>> row_ops; // we store the matrices Q_i which form the basis of hom(C, B) as vectors
    // This translates from entries of the vector to entries of the matrix.
    SparseMatrix K(0,0);
    SparseMatrix S(0,0);
    S.data.reserve( C_rows.size() + B_rows.size() + 1);
    index S_index = 0;
    // First add all row-operations from C to B
    for(index i = 0; i < C_rows.size(); i++){
        for(index j = 0; j < B_rows.size(); j++){
            auto source_row_index = C_rows[i];
            auto target_row_index = B_rows[j];
            if(A.is_admissible_row_operation(source_row_index, target_row_index)){
                row_ops.push_back({source_row_index, target_row_index});
                S.data.push_back(vec<index>());
                for(auto rit = C._rows[i].rbegin(); rit != C._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(A.linearise_position_reverse(column_index, target_row_index));
                }
                S_index++;
            }
        }
    }

    index row_op_threshold = S_index;
    GRLINA_ASSERT( row_ops.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return {SparseMatrix<index>(), row_ops};
    }

    // Then all column-operations from B to C
    for(index i = 0; i < B.columns.size(); i++){
        for(index j = 0; j < C.columns.size(); j++){
            if(A.is_admissible_column_operation(B.columns[i], C.columns[j])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[i]){
                    S.data[S_index].emplace_back(A.linearise_position_reverse(C.columns[j], row_index));
                }
                S_index++;
            }
        }
    }

    S.compute_num_cols();

    if(system_size){
        std::cout << "System size: " << S.get_num_cols() << std::endl;
    }

    // If M, N present the modules, then the following computes Hom(M,N), i.e. pairs of matrices st. QM = NP.
    K = S.kernel();
    // To see how much the following reduces K: index K_size = K.data.size();
    // Now we need to delete the entries of K which correspond to the row-operations.
    K.cull_columns(row_op_threshold, false);
    
    // Last we need to quotient out those Q where for every i the column Q_i - with its degree alpha_i -
    // lies in the image of N, that is, it lies in the image of N|alpha_i.
    // That is equivalent to locally reducing every column of Q.
    
    for(index i = 0; i < C_rows.size(); i++){
        D alpha = C.row_degrees[i];
        vec<index> local_admissible_columns;
        auto B_alpha = B.map_at_degree(alpha, local_admissible_columns);
        std::unordered_map<index, index> shiftIndicesMap;
        for(index j : B_rows){
            shiftIndicesMap[j] = A.linearise_position_reverse(C_rows[i], j);
        }
        B_alpha.transform_data(shiftIndicesMap);
        B_alpha.reduce_fully(K);
    }
    
    //  delete possible linear dependencies.
    K.column_reduction_triangular(true);
    return std::make_pair(K, row_ops);
}

/**
 * @brief Input, together with the hom space, is a set of indices A and a matrix B = coker_B.
 *  Computes for every matrix Q in the hom space
 * B*Q*A and reduce the whole space generated by these.
 * Outputs the indices which span the quotient space.
 * 
 * @tparam D 
 * @tparam index 
 * @param full_space 
 * @param coKer_B 
 * @param basislift_C 
 * @return Hom_space_temp<index> 
 */
template <typename index>
vec<index> hom_quotient( const Hom_space_temp<index>& full_space, 
    SparseMatrix<index>& coKer_B, const vec<index>& basislift_C, vec<index>& C_admissible_rows, vec<index>& B_admissible_rows, vec<index>& C_rows){

    // For each map in full_space, we need to compute the induced map at alpha.
    // It is given by first restricting the domain to the rows in basislift_C, 
    // then composing from the left with the coKer_B map, 

    // Because the basislift is given as a subset of the admissible rows, 
    // we need to re-index the map Q wth the help of the admissible_rows.

    vec<SparseMatrix<index>> induced_maps = vec<SparseMatrix<index>>();

    auto& B_row_index_map = shiftIndicesMap(B_admissible_rows);

        for(index i = 0; i < full_space.first.num_cols; i++){
            const vec<index>& current_map = full_space.first.data[i];
            auto& index_pairs = full_space.second; // This is a pair (i,j) where i is the column index, j, the row index, and it is sorted first by i, then by j.
            SparseMatrix<index> Q_basislift_C = SparseMatrix<index>(0,0);
            index k = 0;
            for(index j : basislift_C){
                Q_basislift_C.data.push_back(vec<index>());

                if(k != current_map.size()){
     
                    while( C_rows[ index_pairs[current_map[k]].first ] < C_admissible_rows[j] ){
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }

                    while( C_rows[ index_pairs[current_map[k]].first ] == C_admissible_rows[j]){
                        Q_basislift_C.data.back().push_back( B_row_index_map[ index_pairs[current_map[k]].second ]);
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                }
            }
            Q_basislift_C.compute_num_cols();
            // Now we have the restriction to the basislift_C, we need to compose with the coKer_B map.
            //TO-DO: Compute the rows of coKer_B beforehand and then multiply to avoid transposition.
            induced_maps.emplace_back(multiply_transpose(coKer_B, Q_basislift_C));
        }

        // Reduce the vector space of induced homomorphisms
        return general_reduction<index, SparseMatrix<index>> (induced_maps);
}

/**
 * @brief Just as hom_quotient, but returns only "true" if the quotient space is zero.
 * 
 * @tparam index 
 * @param full_space 
 * @param coKer_B 
 * @param basislift_C 
 * @param C_admissible_rows 
 * @param B_admissible_rows 
 * @return true 
 * @return false 
 */
template <typename index>
bool hom_quotient_zero( const Hom_space_temp<index>& full_space, 
    SparseMatrix<index>& coKer_B, const vec<index>& basislift_C, vec<index>& C_admissible_rows, vec<index>& B_admissible_rows, vec<index>& C_rows){

    // For each map in full_space, we need to compute the induced map at alpha.
    // It is given by first restricting the domain to the rows in basislift_C, 
    // then composing from the left with the coKer_B map, 

    // Because the basislift is given as a subset of the admissible rows, 
    // we need to re-index the map Q wth the help of the admissible_rows.

    auto B_row_index_map = shiftIndicesMap(B_admissible_rows);

        for(index i = 0; i < full_space.first.get_num_cols(); i++){
            const vec<index>& current_map = full_space.first.data[i];
            auto& index_pairs = full_space.second; // This is a pair (i,j) where i is the column index, j, the row index, and it is sorted first by i, then by j.
            SparseMatrix<index> Q_basislift_C = SparseMatrix<index>(0,0);
            index k = 0;
            for(index j : basislift_C){
                Q_basislift_C.data.push_back(vec<index>());

                if(k < current_map.size()){
     
                    while( C_rows[ index_pairs[current_map[k]].first ] < C_admissible_rows[j] ){
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                    if(k == current_map.size()){
                        break;
                    }
                    
                    while( C_rows[ index_pairs[current_map[k]].first ] == C_admissible_rows[j]){
                        Q_basislift_C.data.back().push_back( B_row_index_map[ index_pairs[current_map[k]].second ]);
                        k++;
                        if(k == current_map.size()){
                            break;
                        }
                    }
                }
            }
            // How many rows does this matrix have?
            Q_basislift_C.compute_num_cols();
            // Now we have the restriction to the basislift_C, we need to compose with the coKer_B map.
            //TO-DO: Compute the rows of coKer_B beforehand and then multiply to avoid transposition.
            SparseMatrix<index> Q_alpha = multiply_transpose(coKer_B, Q_basislift_C);
            if ( Q_alpha.is_nonzero() ) {
                return false;
            }
        }

        // Reduce the vector space of induced homomorphisms
        return true;
}

/**
 * @brief 
 * 
 * @param A 
 * @param B 
 * @param alpha 
 * @return Hom_space_temp<index> 
 */
template <typename D, typename index, typename DERIVED, typename Storage>
Hom_space_temp<index> hom_alpha(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B, Hom_space_temp<index>& full_hom_space, const D alpha) {
    
    Hom_space_temp<index> result;
    auto [B_alpha, B_alpha_gens] = B.map_at_degree_pair(alpha);
    auto [A_alpha, A_alpha_gens] = A.map_at_degree_pair(alpha);

    vec<index> B_alpha_basis = B_alpha.coKernel_basis(B_alpha_gens);
    vec<index> A_alpha_basis = A_alpha.coKernel_basis(A_alpha_gens);

    // What should the indexing for this be? TO-DO: Check this.
    auto coker_B_alpha = B_alpha.coKernel_without_prelim(B_alpha_basis, B_alpha_gens);

    //TO-DO: Finish this.
}   


/** Convert the existing compressed Hom encoding into common flattened map
 * coordinates. Only sparse entries are materialized, never full graded maps.
 */
template<class Storage = DefaultSparseStorage<std::ptrdiff_t>, class Matrix>
SparseMatrix<std::ptrdiff_t, Storage> hom_basis_coefficients(
    const Hom_space_temp<matrix_index_t<Matrix>>& hom,
    const Matrix& source, const Matrix& target) {
    const MapCoefficientContext<Matrix> context(source, target);
    SparseMatrix<std::ptrdiff_t, Storage> result(hom.first.get_num_cols(), context.coefficient_rows());
    result.resize_data(hom.first.get_num_cols());
    for (matrix_index_t<Matrix> j = 0; j < hom.first.get_num_cols(); ++j) {
        vec<std::ptrdiff_t> entries;
        for (auto variable : hom.first.column(j)) {
            const auto [c, r] = hom.second[variable];
            entries.push_back(std::ptrdiff_t(c) * context.rows() + r);
        }
        std::sort(entries.begin(), entries.end());
        result.set_col(j, context.normal(entries));
    }
    result.column_reduction_triangular(true);
    return result;
}

/** Actual Hom classes, as one sparse coefficient column per map. Source and
 * target presentations provide the common grades. The source row cache must
 * be computed. Storage is selectable independently of presentation storage.
 */
template<class Storage = DefaultSparseStorage<std::ptrdiff_t>, class Matrix>
SparseMatrix<std::ptrdiff_t, Storage> hom_space_coefficients(
    const Matrix& source, const Matrix& target,
    HomAlgorithm algorithm = HomAlgorithm::optimised, bool info = false) {
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto A = source.to_coordinates();
        const auto B = target.to_coordinates();
        A.compute_rows_forward();
        return hom_space_coefficients<Storage>(A, B, algorithm, info);
    } else {
        using Index = matrix_index_t<Matrix>;
        Hom_space_temp<Index> hom;
        switch (algorithm) {
        case HomAlgorithm::optimised:
            hom = hom_space_optimised(source, target, vec<Index>{}, vec<Index>{}, info);
            break;
        case HomAlgorithm::full_restriction:
            if constexpr (has_matrix_graded_kernel<Matrix>::value)
                hom = hom_space_full_restriction(source, target, vec<Index>{}, vec<Index>{}, info);
            else throw std::logic_error("Full-restriction Hom requires a graded kernel");
            break;
        case HomAlgorithm::hom_exactness:
            hom_space_basis_new(source, target, true, info, true, 0, &hom);
            break;
        default: throw std::invalid_argument("Unknown Hom algorithm");
        }
        return hom_basis_coefficients<Storage>(hom, source, target);
    }
}

template<class Storage = DefaultSparseStorage<std::ptrdiff_t>, class Matrix>
SparseMatrix<std::ptrdiff_t, Storage> hom_space_shift_image_coefficients(
    const Matrix& source, const Matrix& target, const Matrix& shifted_target, bool info = false) {
    return hom_basis_coefficients<Storage>(
        hom_space_shift_image(source, target, shifted_target, info), source, shifted_target);
}

/** Matrix-only adapter, with no per-map grades or module pointers. Large
 * quotient/composition computations should keep hom_space_coefficients instead
 * to avoid allocating empty source columns for every map.
 */
template<class Matrix, class Storage = DefaultSparseStorage<matrix_index_t<Matrix>>>
std::vector<SparseMatrix<matrix_index_t<Matrix>, Storage>> hom_space_basis_ungraded(
    const Matrix& source, const Matrix& target,
    HomAlgorithm algorithm = HomAlgorithm::optimised, bool info = false) {
    using Index = matrix_index_t<Matrix>;
    using PlainMatrix = SparseMatrix<Index, Storage>;
    auto coefficients = hom_space_coefficients(source, target, algorithm, info);
    std::vector<PlainMatrix> maps;
    maps.reserve(coefficients.get_num_cols());
    const auto rows = target.get_num_rows();
    for (const auto& entries : coefficients.data) {
        PlainMatrix map(source.get_num_rows(), rows);
        map.resize_data(source.get_num_rows());
        for (auto entry : entries) map.append_entry(entry / rows, entry % rows);
        maps.push_back(std::move(map));
    }
    return maps;
}

} // namespace graded_linalg


// Optimised Hom-space algorithms share the same matrix-level interface.
namespace graded_linalg {
template<class Matrix>
auto hom_space_optimised_impl(const Matrix& A, const Matrix& B,
    const vec<matrix_index_t<Matrix>>& row_indices_A = vec<matrix_index_t<Matrix>>(), const vec<matrix_index_t<Matrix>>& row_indices_B = vec<matrix_index_t<Matrix>>(),
    const bool info = false, const Matrix* image_target = nullptr)  {
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    using StoredDT = Degree_traits<base_degree<Matrix>>;
    const bool shared_grid = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return A.grid_ == B.grid_;
        else return true;
    }();
    auto source_above_target = [&](const auto& source_degree, const auto& target_degree) {
        if (shared_grid) return StoredDT::greater_equal(source_degree, target_degree);
        return Degree_traits<D>::greater_equal(geometric_degree(A, source_degree),
                                              geometric_degree(B, target_degree));
    };


    
    GRLINA_ASSERT(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();

    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    // TO-DO: Right now we compute map_at_degree possibly multiple times! This could be optimised.
    for(index i = 0; i < A.get_num_rows(); i++) {
        // Compute the target space B_alpha for each generator of A to minimise the number of variables.
        auto [B_alpha, rows_alpha] = [&] {
            if constexpr (matrix_grid_backed_v<Matrix>)
                return B.map_at_grid_degree_pair(shared_grid ? base_degree<Matrix>(A.row_degrees[i])
                    : query_matrix_degree(B, geometric_degree(A, A.row_degrees[i])), false);
            else return B.map_at_degree_pair(A.row_degrees[i], false);
        }();
        vec<index> basislift;
        if(row_indices_B.size() != 0){
            basislift = B_alpha.coKernel_basis(rows_alpha, row_indices_B );
        } else {
            basislift = B_alpha.coKernel_basis_local(rows_alpha);
        }

        // Then add the effect of all row-operations from A to B (modulo the image of B).
        for(index j : basislift) {
            S.data.push_back(vec<index>());
    	    variable_positions.push_back(std::make_pair(i, j));
            for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                auto& column_index = *rit;
                S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
            }
            S_index++;
        }
    }
    
    index row_op_threshold = S_index;
    GRLINA_ASSERT( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    // For a canonical target shift, carry only the shifted, normalized Q values
    // through elimination, rather than all coordinates of a kernel basis.
    SparseMatrix<index> projection(0, 0);
    vec<std::pair<index, index>> image_positions;
    if (image_target) {
        GRLINA_DEBUG_CHECK(if (!row_indices_A.empty() || !row_indices_B.empty())
            throw std::invalid_argument("Shift image requires original presentation coordinates"));
        const auto degrees = geometric_row_degrees(A);
        const auto reducers = prepare_target_relation_reducers(*image_target, degrees);
        std::map<std::pair<index, index>, index> coordinates;
        for (const auto& [column, row] : variable_positions) {
            vec<index> normal{row};
            const auto& reducer = reducers.at(degrees[column]);
            const auto pivot = reducer.pivots.find(row);
            if (pivot != reducer.pivots.end())
                Column_traits<vec<index>, index>::add_to(reducer.column(pivot->second), normal);
            for (index entry : normal) coordinates.try_emplace({column, entry}, 0);
            projection.append_col(std::move(normal));
        }
        if (coordinates.empty())
            return std::make_pair(SparseMatrix<index>(0, 0), image_positions);
        for (auto& [position, coordinate] : coordinates) {
            coordinate = static_cast<index>(image_positions.size());
            image_positions.push_back(position);
        }
        for (index j = 0; j < row_op_threshold; ++j)
            for (auto& row : projection.data[j])
                row = coordinates.at({variable_positions[j].first, row});
        projection.set_num_rows(static_cast<index>(image_positions.size()));
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }

    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(source_above_target(A.col_degrees[i], B.col_degrees[j])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    if(row_indices_B.size() != 0){
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                    } else {
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                    }
                }
                S_index++;
            }
        }
    }

    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(source_above_target(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Mixed", S, equation_counter);
    }

    SparseMatrix<index> K;
    if (image_target) {
        // The P variables have zero payload. Zero columns of the reduced system
        // carry precisely T(ker(QA-BP)), including corrections killed by eta.
        projection.resize_data(S.get_num_cols());
        projection.compute_num_cols();
        vec<index> zero_columns;
        S.column_reduction_triangular_with_memory(projection, zero_columns);
        K = projection.restricted_domain_copy(zero_columns);
        variable_positions = std::move(image_positions);
    } else {
        K = S.kernel();
        K.cull_columns(row_op_threshold, false);
        K.compute_num_cols();
    }

    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        timer.start();
    }

    K.column_reduction_triangular(true);

    if(info){
        timer.stop();
        std::cout << "  Time to reduce: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        // std::cout << "Dimension of hom-space: " << K.get_num_cols() << std::endl;
    }

    return std::make_pair(K, variable_positions);
}

template<class Matrix>
auto hom_space_no_opt_impl(
    const Matrix& A,
    const Matrix& B,
    const bool reduce = true,
    const vec<matrix_index_t<Matrix>>& row_indices_A = vec<matrix_index_t<Matrix>>(),
    const vec<matrix_index_t<Matrix>>& row_indices_B = vec<matrix_index_t<Matrix>>(),
    const bool info = false)  {
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    using StoredDT = Degree_traits<base_degree<Matrix>>;
    const bool shared_grid = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return A.grid_ == B.grid_;
        else return true;
    }();
    auto source_above_target = [&](const auto& source_degree, const auto& target_degree) {
        if (shared_grid) return StoredDT::greater_equal(source_degree, target_degree);
        return Degree_traits<D>::greater_equal(geometric_degree(A, source_degree),
                                              geometric_degree(B, target_degree));
    };


    
    GRLINA_ASSERT(A.rows_computed);
    boost::timer::cpu_timer timer;
    if(info)
        timer.start();
    vec<SparseMatrix<index>> result;
    vec<std::pair<index,index>> variable_positions; // Stores the position of the variables in the matrix Q
    vec<index> variable_positions_separator = vec<index>(A.get_num_rows(), 0); // To not have to search through the entire variable_positions vector
    SparseMatrix<index> S(0,0);
    S.data.reserve( A.get_num_rows() + B.get_num_rows() + 1);
    index S_index = 0;

    for(index i = 0; i < A.get_num_rows(); i++) {
        if(reduce){
            variable_positions_separator[i] = S_index;
        }
        for(index j = 0; j < B.get_num_rows(); j++) {
            if(source_above_target(A.row_degrees[i], B.row_degrees[j])){
                S.data.push_back(vec<index>());
                
                variable_positions.push_back(std::make_pair(i, j));
                for(auto rit = A._rows[i].rbegin(); rit != A._rows[i].rend(); rit++){
                    auto& column_index = *rit;
                    S.data[S_index].emplace_back(linearise_position_reverse_ext(column_index, j, A.get_num_cols(), B.get_num_rows()));
                }
                S_index++;
            } 
        }
    }
    
    index row_op_threshold = S_index;
    GRLINA_ASSERT( variable_positions.size() == S_index );

    if(row_op_threshold == 0){
        // If there are no row-operations, then the hom-space is zero.
        return std::make_pair( SparseMatrix<index>(0,0), variable_positions);
    }

    std::unordered_map<index, index> row_map;
    if(row_indices_B.size() != 0){
        row_map = shiftIndicesMap(row_indices_B );
    }

    // Then all column-operations from B to A
    for(index i = A.get_num_cols()-1; i > -1; i--){
        for(index j = 0; j < B.get_num_cols(); j++){
            if(source_above_target(A.col_degrees[i], B.col_degrees[j])){
                S.data.push_back(vec<index>());
                for(index row_index : B.data[j]){
                    if(row_indices_B.size() != 0){
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_map[row_index], A.get_num_cols(), B.get_num_rows()));
                    } else {
                        S.data[S_index].emplace_back(linearise_position_reverse_ext(i, row_index, A.get_num_cols(), B.get_num_rows()));        
                    }
                }
                S_index++;
            }
        }
    }


    S.compute_num_cols();

    if(info){
        index equation_counter = 0;
        for(index i = 0; i < A.get_num_cols(); i++){
            for(index j = 0; j < B.get_num_rows(); j++){
                if(source_above_target(A.col_degrees[i], B.row_degrees[j])){
                    equation_counter++;
                }
            }
        }
        system_stats(timer, "Naive", S, equation_counter);
    }

    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();
    K.column_reduction_triangular(true);
    if(info){
        timer.stop();
        std::cout << "  time to solve: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        // std::cout << "Dimension of hom-space before reduction: " << K.get_num_cols() << std::endl;
        timer.start();
    }
    if(reduce){
        SparseMatrix<index> N_bar = SparseMatrix<index>(0,K.get_num_cols());
        for(index i = 0; i < A.get_num_rows(); i++){
            for(index j = 0; j < B.get_num_cols(); j++){
                if(source_above_target(A.row_degrees[i], B.col_degrees[j])){
                    // Add a new homotopy
                    auto column = B.get_col(j);
                    if (!row_indices_B.empty()) for (auto& row : column) {
                        const auto mapped = row_map.find(row);
                        GRLINA_ASSERT(mapped != row_map.end());
                        row = mapped->second;
                    }
                    vec<index> h = index_pair_to_position(i, variable_positions_separator[i], column, variable_positions);
                    N_bar.data.push_back(h);
                }
            }
        }
        N_bar.compute_num_cols();
        N_bar.reduce_fully_alt(K);
        if(info){
            timer.stop();
            std::cout << "  Time to reduce to basis: " << timer.elapsed().wall * 1e-6 << "ms" << std::endl;
        }
    }
    return std::make_pair(K, variable_positions);
}

template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_optimised(const GradedSparseMatrix<D, index, DERIVED, Storage>& A, const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const vec<index>& row_indices_A, const vec<index>& row_indices_B,
    const bool info) {
    return hom_space_optimised_impl(static_cast<const DERIVED&>(A),
                                            static_cast<const DERIVED&>(B),
                                            row_indices_A, row_indices_B, info);
}

template <typename D, typename index, typename DERIVED, typename Storage>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_no_opt(
    const GradedSparseMatrix<D, index, DERIVED, Storage>& A,
    const GradedSparseMatrix<D, index, DERIVED, Storage>& B,
    const bool reduce,
    const vec<index>& row_indices_A, 
    const vec<index>& row_indices_B, 
    const bool info) {
    return hom_space_no_opt_impl(static_cast<const DERIVED&>(A),
                                         static_cast<const DERIVED&>(B),
                                         reduce, row_indices_A, row_indices_B, info);
}


/** Compute actual module-map representatives with an explicit Hom algorithm.
 * The source row cache must be computed. All three algorithms return the full
 * basis, in the original generator coordinates, without a conversion-size cap.
 * The historical boolean overload and hom_space_basis_new remain available.
 */
template<class Matrix>
std::vector<Matrix> hom_space_basis(const Matrix& source, const Matrix& target,
                                  HomAlgorithm algorithm, bool info = false) {
    using Index = matrix_index_t<Matrix>;
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto coordinate_source = source.to_coordinates();
        const auto coordinate_target = target.to_coordinates();
        coordinate_source.compute_rows_forward();
        auto lifts = hom_space_basis(coordinate_source, coordinate_target, algorithm, info);
        std::vector<Matrix> result;
        result.reserve(lifts.size());
        for (const auto& lift : lifts) result.emplace_back(lift);
        return result;
    } else {
        if (algorithm == HomAlgorithm::hom_exactness)
            return hom_space_basis_new(source, target, true, info, true, 0);
        Hom_space_temp<Index> hom;
        switch (algorithm) {
        case HomAlgorithm::optimised:
            hom = hom_space_optimised(source, target, vec<Index>{}, vec<Index>{}, info);
            break;
        case HomAlgorithm::full_restriction:
            if constexpr (has_matrix_graded_kernel<Matrix>::value)
                hom = hom_space_full_restriction(source, target, vec<Index>{}, vec<Index>{}, info);
            else
                throw std::logic_error("Full-restriction Hom requires a graded kernel");
            hom.first.column_reduction_triangular(true);
            break;
        default:
            throw std::invalid_argument("Unknown Hom algorithm");
        }
        const auto& [basis, positions] = hom;
        std::vector<Matrix> result;
        result.reserve(basis.get_num_cols());
        for (Index j = 0; j < basis.get_num_cols(); ++j) {
            auto map = empty_matrix_like(target, source.get_num_rows(), target.get_num_rows());
            set_geometric_degrees(map, geometric_row_degrees(source), geometric_row_degrees(target));
            for (Index entry : basis.column(j)) {
                const auto [column, row] = positions[entry];
                map.append_entry(column, row);
            }
            result.push_back(std::move(map));
        }
        return result;
    }
}

/** Compact basis of eta Hom(coker(source), coker(target)).
 * Trusted input: shifted_target is target shifted by a nonnegative amount,
 * with the same generator/relation coefficients and order (eta is the identity
 * on free coordinates). The source row cache must be computed.
 *
 * Use the sharp restriction systems of Section 3.2 / Algorithm A-1/2 of
 * poset_hom.pdf at the ORIGINAL target grades. If L(Q,P)=Q source-target P
 * and T normalizes Q modulo shifted_target at each source generator grade,
 * the required image is T(ker L). Carry T through column elimination on L;
 * its columns over the zero columns of L span exactly this image. This avoids
 * constructing either a full kernel basis or a basis of the unshifted Hom.
 * In particular, do not discard original Q variables whose T-image is zero:
 * they may be needed to satisfy the original relation equations.
 *
 * The returned pair has the same encoding as hom_space_optimised: each matrix
 * column is a map, with entries indexing (source generator, target generator)
 * pairs. Representatives are normalized modulo the shifted target relations.
 */
template<class Matrix>
Hom_space_temp<matrix_index_t<Matrix>> hom_space_shift_image(
    const Matrix& source, const Matrix& target, const Matrix& shifted_target,
    bool info = false) {
    using Index = matrix_index_t<Matrix>;
    GRLINA_DEBUG_CHECK(if (target.get_num_rows() != shifted_target.get_num_rows() ||
                          target.get_num_cols() != shifted_target.get_num_cols())
        throw std::invalid_argument("Shift image requires the same target presentation bases"));
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto coordinate_source = source.to_coordinates();
        const auto coordinate_target = target.to_coordinates();
        const auto coordinate_shift = shifted_target.to_coordinates();
        coordinate_source.compute_rows_forward();
        return hom_space_shift_image(coordinate_source, coordinate_target, coordinate_shift, info);
    } else {
        return hom_space_optimised_impl(source, target, vec<Index>{}, vec<Index>{}, info,
                                        &shifted_target);
    }
}

/** Generator-lift representatives of hom_space_shift_image.
 * For eta^d End_0(X), pass (X, X, X_d), where X_d = X; X_d.shift(d).
 * Only independent, nonzero shifted maps are materialized.
 */
template<class Matrix>
std::vector<Matrix> hom_space_shift_image_basis(
    const Matrix& source, const Matrix& target, const Matrix& shifted_target,
    bool info = false) {
    using Index = matrix_index_t<Matrix>;
    const auto [basis, positions] = hom_space_shift_image(source, target, shifted_target, info);
    std::vector<Matrix> result;
    result.reserve(basis.get_num_cols());
    for (Index j = 0; j < basis.get_num_cols(); ++j) {
        auto map = empty_matrix_like(shifted_target, source.get_num_rows(), target.get_num_rows());
        set_geometric_degrees(map, geometric_row_degrees(source), geometric_row_degrees(shifted_target));
        for (Index entry : basis.column(j)) {
            const auto [column, row] = positions[entry];
            map.append_entry(column, row);
        }
        result.push_back(std::move(map));
    }
    return result;
}

} // namespace graded_linalg


#endif // GRLINA_HOM_ALGORITHMS_HPP
