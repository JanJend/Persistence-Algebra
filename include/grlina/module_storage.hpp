/** @file module_storage.hpp
 * @brief NEW: convert every stored differential of a runtime module together.
 */
#pragma once
#include <grlina/module.hpp>

namespace graded_linalg {

template<class Storage, class Matrix>
auto convert_module_storage(const Module<Matrix>& module) {
    using TargetMatrix = decltype(std::declval<const Matrix&>().template to_storage<Storage>());
    using TargetModule = Module<TargetMatrix>;
    auto convert = [](const auto& resolution) {
        vec<TargetMatrix> maps;
        maps.reserve(resolution.size());
        for (const auto& map : resolution.differentials())
            maps.push_back(map.template to_storage<Storage>());
        if constexpr (GradedMatrixIO<Matrix>::runtime_dimension) {
            if (maps.empty()) return ChainComplex<TargetMatrix>(
                GradedMatrixIO<Matrix>::parse_poset_identifier(resolution.runtime_poset_identifier()));
        }
        return ChainComplex<TargetMatrix>(std::move(maps));
    };
    TargetModule result;
    result.set_projective_resolution(convert(module.projective_resolution()),
            module.has_complete_projective_resolution() ? ResolutionCompleteness::complete
                                                       : ResolutionCompleteness::truncated);
    result.set_injective_resolution(convert(module.injective_resolution()));
    return result;
}

} // namespace graded_linalg
