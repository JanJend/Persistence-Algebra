/**
 * @file modules.hpp
 * @brief Aggregate include for persistence modules, submodules, and maps.
 *
 * The implementation is split into focused headers so clients may include
 * only what they use.  This replaces the former incomplete Interval_module
 * stub without removing any presentation-matrix API.
 */
#pragma once

#include <grlina/chain_complex.hpp>
#include <grlina/module.hpp>
#include <grlina/submodule.hpp>
#include <grlina/homomorphism.hpp>
#include <grlina/module_homomorphisms.hpp>
#include <grlina/module_operations.hpp>
#include <grlina/r3graded_matrix.hpp>
#include <grlina/z2graded_matrix.hpp>
#include <grlina/z3graded_matrix.hpp>
#include <grlina/r4graded_matrix.hpp>
#include <grlina/z4graded_matrix.hpp>

namespace graded_linalg {

template <typename index> using R3Module = Module<R3GradedSparseMatrix<index>>;
template <typename index> using Z2Module = Module<Z2GradedSparseMatrix<index>>;
template <typename index> using Z3Module = Module<Z3GradedSparseMatrix<index>>;
template <typename index> using R4Module = Module<R4GradedSparseMatrix<index>>;
template <typename index> using Z4Module = Module<Z4GradedSparseMatrix<index>>;

} // namespace graded_linalg
