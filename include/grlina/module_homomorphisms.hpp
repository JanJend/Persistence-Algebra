/** @file module_homomorphisms.hpp @brief Module-level adapters for legacy Hom algorithms. */
#pragma once

#include <memory>
#include <vector>

#include <grlina/homomorphisms.hpp>
#include <grlina/homomorphism.hpp>

namespace graded_linalg {

template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_hom_space_basis(
    std::shared_ptr<const Module<Matrix>> domain,
    std::shared_ptr<const Module<Matrix>> target,
    bool use_hom_exactness = false, bool info = false) {
    if (!domain || !target)
        throw std::invalid_argument("Hom basis requires non-null domain and target modules");
    Matrix source = domain->presentation();
    Matrix destination = target->presentation();
    source.compute_rows_forward();
    // Module homomorphisms are equivalence classes of lifts, so remove maps that
    // differ by a factorisation through the target relations.
    auto lifts = hom_space_basis_new(source, destination, use_hom_exactness, info, true);
    std::vector<Homomorphism<Matrix>> result;
    result.reserve(lifts.size());
    for (auto& lift : lifts) result.emplace_back(domain, target, std::move(lift));
    return result;
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> module_endomorphism_basis(
    std::shared_ptr<const Module<Matrix>> module,
    bool use_hom_exactness = false, bool info = false) {
    return module_hom_space_basis(module, module, use_hom_exactness, info);
}

} // namespace graded_linalg
