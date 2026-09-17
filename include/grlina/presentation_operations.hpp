/** Presentation/generator-matrix adapters for reusable module constructions.
 * Prefer Submodule/Homomorphism when parent objects are already available.
 */
#pragma once
#include <grlina/submodule.hpp>
#include <grlina/matrix_family.hpp>

namespace graded_linalg {

/** Exact inclusion of generated submodules of the SAME FREE target.
 * For a presented target use Submodule::contains, which includes its relations.
 */
template <typename Matrix>
bool image_contained_in_image(const Matrix& contained, const Matrix& containing) {
    return graded_linear_system_is_solvable(containing, contained);
}

template <typename Matrix>
bool present_same_submodule(const Matrix& presentation, const Matrix& A, const Matrix& B) {
    auto parent = std::make_shared<const Module<Matrix>>(presentation);
    return Submodule<Matrix>(parent, A).equals(Submodule<Matrix>(parent, B));
}

template <typename Matrix>
Matrix zero_submodule(const Matrix& presentation) {
    return Submodule<Matrix>::zero(std::make_shared<const Module<Matrix>>(presentation)).generator_map().generator_lift();
}

template <typename Matrix>
Matrix all_submodule(const Matrix& presentation) {
    return Submodule<Matrix>::whole(std::make_shared<const Module<Matrix>>(presentation)).generator_map().generator_lift();
}

template <typename Matrix>
Matrix reduce_submodule(const Matrix& presentation, const Matrix& generators,
                        bool lazy_preprocessing = true) {
    Submodule<Matrix> result(std::make_shared<const Module<Matrix>>(presentation), generators);
    result.minimize_generators(lazy_preprocessing);
    return result.generator_map().generator_lift();
}

/** Sum in the free target, with the ambient basis unchanged. */
template <typename Matrix>
Matrix submodule_sum(const Matrix& A, const Matrix& B) {
    Matrix free_presentation(0, A.get_num_rows());
    free_presentation.row_degrees = A.row_degrees;
    auto parent = std::make_shared<const Module<Matrix>>(std::move(free_presentation));
    return Submodule<Matrix>(parent, A).sum(Submodule<Matrix>(parent, B)).generator_map().generator_lift();
}

} // namespace graded_linalg
