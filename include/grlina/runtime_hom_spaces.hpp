/** @file runtime_hom_spaces.hpp
 * @brief NEW: shared hom-space algorithms for legacy and runtime degree storage.
 * AIDA blocks may contain global source row indices until normalisation.
 */
#pragma once
#include <grlina/homomorphisms.hpp>
#include <grlina/matrix_geometry.hpp>

namespace graded_linalg {
namespace detail {
template<class Matrix>
auto hom_space_optimised_impl(const Matrix& A, const Matrix& B,
    const vec<typename Matrix::index_type>& row_indices_A = vec<typename Matrix::index_type>(), const vec<typename Matrix::index_type>& row_indices_B = vec<typename Matrix::index_type>(),
    const bool info = false)  {
    using index = typename Matrix::index_type;
    using D = matrix_geometry_degree_t<Matrix>;
    using StoredDT = Degree_traits<typename Matrix::degree_type>;
    const bool shared_grid = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return A.grids == B.grids;
        else return true;
    }();
    auto source_above_target = [&](const auto& source_degree, const auto& target_degree) {
        if (shared_grid) return StoredDT::greater_equal(source_degree, target_degree);
        return Degree_traits<D>::greater_equal(detail::geometric_degree(A, source_degree),
                                              detail::geometric_degree(B, target_degree));
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
        auto [B_alpha, rows_alpha] = B.map_at_degree_pair(shared_grid ? typename Matrix::degree_type(A.row_degrees[i])
            : detail::query_matrix_degree(B, detail::geometric_degree(A, A.row_degrees[i])), false);
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

    auto K = S.kernel();
    K.cull_columns(row_op_threshold, false);
    K.compute_num_cols();

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
    const vec<typename Matrix::index_type>& row_indices_A = vec<typename Matrix::index_type>(), 
    const vec<typename Matrix::index_type>& row_indices_B = vec<typename Matrix::index_type>(), 
    const bool info = false)  {
    using index = typename Matrix::index_type;
    using D = matrix_geometry_degree_t<Matrix>;
    using StoredDT = Degree_traits<typename Matrix::degree_type>;
    const bool shared_grid = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return A.grids == B.grids;
        else return true;
    }();
    auto source_above_target = [&](const auto& source_degree, const auto& target_degree) {
        if (shared_grid) return StoredDT::greater_equal(source_degree, target_degree);
        return Degree_traits<D>::greater_equal(detail::geometric_degree(A, source_degree),
                                              detail::geometric_degree(B, target_degree));
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
                    if (!row_indices_B.empty()) for (auto& row : column) row = row_map.at(row);
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
} // namespace detail

template <typename D, typename index, typename DERIVED, typename MatrixBase>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_optimised(const GradedSparseMatrix<D, index, DERIVED, MatrixBase>& A, const GradedSparseMatrix<D, index, DERIVED, MatrixBase>& B,
    const vec<index>& row_indices_A, const vec<index>& row_indices_B,
    const bool info) {
    return detail::hom_space_optimised_impl(A, B, row_indices_A, row_indices_B, info);
}

template<class Matrix, std::enable_if_t<Matrix::runtime_dimension, int> = 0>
auto hom_space_optimised(const Matrix& A, const Matrix& B,
    const vec<typename Matrix::index_type>& row_indices_A = {},
    const vec<typename Matrix::index_type>& row_indices_B = {}, bool info = false) {
    return detail::hom_space_optimised_impl(A, B, row_indices_A, row_indices_B, info);
}

template <typename D, typename index, typename DERIVED, typename MatrixBase>
std::pair< SparseMatrix<index>, vec<std::pair<index,index>> > hom_space_no_opt(
    const GradedSparseMatrix<D, index, DERIVED, MatrixBase>& A,
    const GradedSparseMatrix<D, index, DERIVED, MatrixBase>& B,
    const bool reduce,
    const vec<index>& row_indices_A, 
    const vec<index>& row_indices_B, 
    const bool info) {
    return detail::hom_space_no_opt_impl(A, B, reduce, row_indices_A, row_indices_B, info);
}

template<class Matrix, std::enable_if_t<Matrix::runtime_dimension, int> = 0>
auto hom_space_no_opt(const Matrix& A, const Matrix& B, bool reduce = true,
    const vec<typename Matrix::index_type>& row_indices_A = {},
    const vec<typename Matrix::index_type>& row_indices_B = {}, bool info = false) {
    return detail::hom_space_no_opt_impl(A, B, reduce, row_indices_A, row_indices_B, info);
}

/** NEW: convert the shared linear-system basis to runtime graded generator lifts. */
template<class Matrix, std::enable_if_t<Matrix::runtime_dimension, int> = 0>
vec<Matrix> hom_space_basis_new(const Matrix& source, const Matrix& target,
                              bool use_hom_exactness = false, bool info = false,
                              bool reduce_lift_duplicates = false) {
    source.validate(); target.validate();
    auto [basis, positions] = detail::hom_space_no_opt_impl(
        source, target, use_hom_exactness || reduce_lift_duplicates, {}, {}, info);
    vec<Matrix> result;
    result.reserve(basis.get_num_cols());
    for (typename Matrix::index_type i = 0; i < basis.get_num_cols(); ++i) {
        auto lift = detail::empty_matrix_like(target, source.get_num_rows(), target.get_num_rows());
        detail::set_geometric_degrees(lift, detail::geometric_row_degrees(source),
                                     detail::geometric_row_degrees(target));
        for (auto position : basis.column(i))
            lift.append_entry(positions[position].first, positions[position].second);
        lift.refresh_compatible_sorted();
        result.push_back(std::move(lift));
    }
    return result;
}

} // namespace graded_linalg
