/** @file module_transformations.hpp
 * @brief NEW: nonmutating module transformations with maps and homology cycles.
 */
#pragma once
#include <grlina/csc_matrix.hpp>
#include <grlina/submodule.hpp>
#include <grlina/module_storage.hpp>

namespace graded_linalg {

/** forward: original -> transformed; backward: transformed -> original.
 * They are inverse module maps (their generator lifts need not be inverse
 * matrices after cancelling relations). All stored projective lifts are kept.
 */
template<class Matrix>
struct ModuleTransformation {
    std::shared_ptr<const Module<Matrix>> original, transformed;
    Homomorphism<Matrix> forward, backward;
};

namespace detail {
template<class Matrix>
ChainBasisMaps<Matrix> identity_basis_maps(const ChainComplex<Matrix>& complex) {
    ChainBasisMaps<Matrix> maps;
    for (std::size_t group = 0; group <= complex.size(); ++group) {
        const Matrix& reference = complex[group == 0 ? 0 : group - 1];
        maps.to_original.push_back(identity_on_degrees(reference,
            group == 0 ? reference.row_degrees : reference.col_degrees));
    }
    maps.from_original = maps.to_original;
    return maps;
}

template<class Matrix>
std::shared_ptr<const Module<Matrix>> owning_module(const Module<Matrix>& module) {
    auto owner = module.weak_from_this().lock();
    return owner ? owner : std::make_shared<Module<Matrix>>(module);
}

template<class Matrix, class Operation>
ModuleTransformation<Matrix> transform_module(
    std::shared_ptr<const Module<Matrix>> original, Operation operation) {
    if (!original || !original->has_presentation())
        throw std::invalid_argument("Module transformation requires a presentation");
    // Runtime CSC modules use editable scratch once, including every lift.
    if constexpr (GradedMatrixIO<Matrix>::runtime_dimension &&
                  std::is_same_v<typename Matrix::storage_type, CSCStorage<typename Matrix::index_type>>) {
        using EditableModule = decltype(convert_module_storage<array<typename Matrix::index_type>>(*original));
        auto scratch = std::make_shared<const EditableModule>(
            convert_module_storage<array<typename Matrix::index_type>>(*original));
        auto change = transform_module(scratch, operation);
        auto transformed = std::make_shared<const Module<Matrix>>(
            convert_module_storage<typename Matrix::storage_type>(*change.transformed));
        vec<Matrix> forward, backward;
        for (const auto& lift : change.forward.lifts()) forward.push_back(lift.template to_storage<typename Matrix::storage_type>());
        for (const auto& lift : change.backward.lifts()) backward.push_back(lift.template to_storage<typename Matrix::storage_type>());
        return {original, transformed, Homomorphism<Matrix>(original, transformed, std::move(forward)),
                Homomorphism<Matrix>(transformed, original, std::move(backward))};
    } else {
        auto transformed = std::make_shared<Module<Matrix>>(*original);
        auto complex = original->projective_resolution();
        auto maps = identity_basis_maps(complex);
        operation(complex, maps);
        transformed->set_projective_resolution(std::move(complex),
            original->has_complete_projective_resolution() ? ResolutionCompleteness::complete
                                                         : ResolutionCompleteness::truncated);
        for (auto& map : maps.to_original) map.refresh_compatible_sorted();
        for (auto& map : maps.from_original) map.refresh_compatible_sorted();
        return {original, transformed, Homomorphism<Matrix>(original, transformed, std::move(maps.from_original)),
                Homomorphism<Matrix>(transformed, original, std::move(maps.to_original))};
    }
}
} // namespace detail

template<class Matrix, class Compare = TraitLinearOrder<typename Matrix::degree_type>>
ModuleTransformation<Matrix> sort_module_with_maps(
    std::shared_ptr<const Module<Matrix>> module,
    Compare compare = {Degree_traits<typename Matrix::degree_type>::lex_lambda()}) {
    return detail::transform_module(std::move(module), [compare](auto& complex, auto& maps) {
        complex.sort_compatibly(compare, &maps);
    });
}

template<class Matrix>
ModuleTransformation<Matrix> minimize_module_with_maps(
    std::shared_ptr<const Module<Matrix>> module, bool sort_if_needed = true) {
    return detail::transform_module(std::move(module), [sort_if_needed](auto& complex, auto& maps) {
        complex.minimize(sort_if_needed, &maps);
        auto& terminal = complex.differentials().back();
        if (terminal.get_num_cols() != 0) terminal.remove_redundant_relations([&](auto redundant, const auto& expression) {
            for (auto row : expression) if (row != redundant)
                maps.from_original.back().row_op_on_cols(redundant, row);
            vec<typename std::decay_t<decltype(terminal)>::index_type> removed{redundant};
            maps.to_original.back().delete_columns(removed);
            maps.from_original.back().delete_rows(removed);
        });
    });
}

template<class Matrix, class Compare = TraitLinearOrder<typename Matrix::degree_type>>
ModuleTransformation<Matrix> sort_module_with_maps(
    const Module<Matrix>& module,
    Compare compare = {Degree_traits<typename Matrix::degree_type>::lex_lambda()}) {
    return sort_module_with_maps(detail::owning_module(module), compare);
}

template<class Matrix>
ModuleTransformation<Matrix> minimize_module_with_maps(const Module<Matrix>& module, bool sort_if_needed = true) {
    return minimize_module_with_maps(detail::owning_module(module), sort_if_needed);
}

template<class Matrix, class Compare = TraitLinearOrder<typename Matrix::degree_type>>
ModuleTransformation<Matrix> sort_module_with_maps(
    std::shared_ptr<Module<Matrix>> module,
    Compare compare = {Degree_traits<typename Matrix::degree_type>::lex_lambda()}) {
    return sort_module_with_maps(std::shared_ptr<const Module<Matrix>>(std::move(module)), compare);
}

template<class Matrix>
ModuleTransformation<Matrix> minimize_module_with_maps(std::shared_ptr<Module<Matrix>> module, bool sort_if_needed = true) {
    return minimize_module_with_maps(std::shared_ptr<const Module<Matrix>>(std::move(module)), sort_if_needed);
}

template<class Matrix>
struct HomologyWithCycles {
    std::shared_ptr<const Module<Matrix>> module;
    Matrix cycles; // module generators -> the original C_q coordinates
};

/** Cycle representatives use the returned module's actual generator basis,
 * including after minimization. This suffices to lift a chain map to homology.
 */
template<class Matrix>
HomologyWithCycles<Matrix> homology_with_cycles(const ChainComplex<Matrix>& complex,
                                               std::size_t degree = 1, bool minimize = true) {
    auto result = detail::homology_presentation_and_cycles(complex, degree);
    auto module = std::make_shared<const Module<Matrix>>(std::move(result.first));
    if (!minimize) return {std::move(module), std::move(result.second)};
    auto change = minimize_module_with_maps(module);
    Matrix cycles = result.second * change.backward.generator_lift();
    return {std::move(change.transformed), std::move(cycles)};
}
} // namespace graded_linalg
