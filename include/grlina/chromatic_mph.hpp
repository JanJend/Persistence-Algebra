#pragma once

#include <grlina/proj_chain_complex.hpp>
#include <grlina/hom_core_class.hpp>

namespace graded_linalg {

// Input stores d1, d2, ... and has third coordinate 0 or 1 in every degree
// L,K are the graded chain complexes over R^2
// L is the supcomplex where the third coordinate is 0, K is the whole complex
template <typename Index>
void compute_chromatic_mph(ChainComplex<R3Matrix<Index>>& input, std::size_t dimension) {
    // Track both directions while minimizing the input in place.
    auto basis_K = identity_basis_maps(input);
    input.minimize(true, &basis_K);
    // L's maps keep the original simplex rows and their three-coordinate degrees.
    auto basis_L = basis_K.to_original;
    ChainComplex<R2Matrix<Index>> L, K;
    vec<R2Matrix<Index>> inclusion_matrices;

    for (const auto& differential : input.differentials()) {
        // Forget the third coordinate, keeping the boundary and its basis order.
        R2Matrix<Index> d_K(differential.get_num_cols(), differential.get_num_rows());
        d_K.assign_data(differential.data);
        vec<Index> columns, rows;
        for (Index i = 0; i < differential.get_num_cols(); ++i) {
            const auto& degree = differential.col_degrees[i];
            d_K.col_degrees[i] = {degree[0], degree[1]};
            if (degree[2] == 1) columns.push_back(i);
        }
        for (Index i = 0; i < differential.get_num_rows(); ++i) {
            const auto& degree = differential.row_degrees[i];
            d_K.row_degrees[i] = {degree[0], degree[1]};
            if (degree[2] == 1) rows.push_back(i);
        }

        // Each inclusion column keeps its original basis index in K.
        // C0 occurs in the rows of d1; higher groups occur in the columns.
        if (inclusion_matrices.empty()) {
            basis_L[0].delete_columns(rows);
            R2Matrix<Index> i_0(d_K.row_degrees);
            i_0.compatibly_sorted = d_K.compatibly_sorted;
            i_0.delete_columns(rows);
            inclusion_matrices.push_back(std::move(i_0));
        }
        basis_L[K.size() + 1].delete_columns(columns);
        R2Matrix<Index> i_n(d_K.col_degrees);
        i_n.compatibly_sorted = d_K.compatibly_sorted;
        i_n.delete_columns(columns);
        inclusion_matrices.push_back(std::move(i_n));

        // Restrict to the zero subcomplex; deletion renumbers surviving faces.
        auto d_L = d_K;
        d_L.delete_columns(columns);
        d_L.delete_rows(rows);
        L.push_differential(std::move(d_L));
        K.push_differential(std::move(d_K));
    }
    const ChainMap<R2Matrix<Index>> iota(L, K, std::move(inclusion_matrices));

    // basis_K.to_original[n] and basis_L[n] send cycles to original simplex coordinates.

    // Use the outgoing boundary when it exists, without changing L or K.
    // Otherwise use the identity on C0, or the 0x0 identity for an absent group.
    R2Matrix<Index> kernel_L(0, 0);
    if (dimension > 0 && dimension <= L.size()) {
        kernel_L = std::as_const(L).differential(dimension).graded_kernel();
    } else if (dimension == 0 && !L.empty()) {
        kernel_L = R2Matrix<Index>(L[0].row_degrees);
        kernel_L.compatibly_sorted = L[0].compatibly_sorted;
    }

    R2Matrix<Index> kernel_K(0, 0);
    if (dimension > 0 && dimension <= K.size()) {
        kernel_K = std::as_const(K).differential(dimension).graded_kernel();
    } else if (dimension == 0 && !K.empty()) {
        kernel_K = R2Matrix<Index>(K[0].row_degrees);
        kernel_K.compatibly_sorted = K[0].compatibly_sorted;
    }

    // L=C and K=D in the diagram. Boundaries are cycles, so these lifts exist:
    // kernel_L * p_L = d_L(n+1), kernel_K * p_K = d_K(n+1).
    R2Matrix<Index> p_L(0, kernel_L.get_num_cols(), {}, kernel_L.col_degrees);
    if (dimension < L.size()) {
        p_L = std::move(*solve_graded_linear_system(
            kernel_L, std::as_const(L).differential(dimension + 1)));
    }
    R2Matrix<Index> p_K(0, kernel_K.get_num_cols(), {}, kernel_K.col_degrees);
    if (dimension < K.size()) {
        p_K = std::move(*solve_graded_linear_system(
            kernel_K, std::as_const(K).differential(dimension + 1)));
    }

    // Lift iota_n through the kernel inclusion: kernel_K * alpha = iota_n * kernel_L.
    R2Matrix<Index> alpha(0, 0);
    if (dimension < iota.matrices().size()) {
        // iota's identity flag makes this product a direct row reindexing.
        alpha = std::move(*solve_graded_linear_system(kernel_K, iota[dimension] * kernel_L));
    }

    // Present H_n(L) -> H_n(K) by alpha on generators and iota_(n+1) on relations.
    // The outer square commutes: alpha * p_L = p_K * iota_(n+1).
    const auto H_L = std::make_shared<const Module<R2Matrix<Index>>>(std::move(p_L));
    const auto H_K = std::make_shared<const Module<R2Matrix<Index>>>(std::move(p_K));
    vec<R2Matrix<Index>> lifts(2);
    lifts[0] = std::move(alpha);
    if (dimension < K.size()) lifts[1] = iota[dimension + 1];
    const Homomorphism<R2Matrix<Index>> H_iota(H_L, H_K, std::move(lifts));
}

} // namespace graded_linalg
