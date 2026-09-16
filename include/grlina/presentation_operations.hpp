/** Presentation/generator-matrix adapters for reusable module constructions.
 * Prefer Submodule/Homomorphism when parent objects are already available.
 */
#pragma once
#include <grlina/homomorphism.hpp>
#include <grlina/matrix_family.hpp>

namespace graded_linalg {

/** Exact inclusion of generated submodules of the SAME FREE target.
 * For a presented target use Submodule::contains, which includes its relations.
 */
template <typename Matrix>
bool image_contained_in_image(const Matrix& contained, const Matrix& containing) {
    contained.validate();
    containing.validate();
    if (!contained.is_graded_matrix() || !containing.is_graded_matrix())
        throw std::invalid_argument("Image containment requires graded generator matrices");
    return solve_graded_linear_system(containing, contained).has_value();
}

template <typename Matrix>
bool present_same_submodule(const Matrix& presentation, const Matrix& A, const Matrix& B) {
    auto parent = std::make_shared<const Module<Matrix>>(presentation);
    return Submodule<Matrix>(parent, A).equals(Submodule<Matrix>(parent, B));
}

template <typename Matrix>
Matrix zero_submodule(const Matrix& presentation) {
    return Submodule<Matrix>::zero(std::make_shared<const Module<Matrix>>(presentation)).generators();
}

template <typename Matrix>
Matrix all_submodule(const Matrix& presentation) {
    return Submodule<Matrix>::whole(std::make_shared<const Module<Matrix>>(presentation)).generators();
}

template <typename Matrix>
Matrix reduce_submodule(const Matrix& presentation, const Matrix& generators,
                        bool lazy_preprocessing = true) {
    Submodule<Matrix> result(std::make_shared<const Module<Matrix>>(presentation), generators);
    result.minimize_generators(lazy_preprocessing);
    return result.generators();
}

/** Sum in the free target, with the ambient basis unchanged. */
template <typename Matrix>
Matrix submodule_sum(const Matrix& A, const Matrix& B) {
    Matrix free_presentation(0, A.get_num_rows());
    free_presentation.row_degrees = A.row_degrees;
    auto parent = std::make_shared<const Module<Matrix>>(std::move(free_presentation));
    return Submodule<Matrix>(parent, A).sum(Submodule<Matrix>(parent, B)).generators();
}

template <typename Matrix>
Matrix canonical_shift_lift(const Matrix& presentation, const typename Matrix::degree_type& amount) {
    return Homomorphism<Matrix>::canonical_shift(
        std::make_shared<const Module<Matrix>>(presentation), amount).generator_lift();
}

/** Basis of the space of generator lifts, without identifying lifts differing
 * by target relations. Use module_hom_space_basis for actual Hom classes.
 */
template <typename Matrix>
std::vector<Matrix> homomorphism_lift_basis(const Matrix& domain, const Matrix& target) {
    Matrix source = domain;
    source.compute_rows_forward();
    return hom_space_basis_new(source, target);
}

/** Additional generator lifts to M(amount), modulo the coefficient span of
 * endomorphism lifts transported along M -> M(amount). This preserves the
 * legacy pruning convention: these are LIFT spaces, not Hom classes modulo
 * target relations. The shift must admit the canonical structure map.
 */
template <typename Matrix>
std::vector<Matrix> shifted_endomorphism_lift_complement(
    const Matrix& presentation, const typename Matrix::degree_type& amount,
    bool info = false) {
    (void)canonical_shift_lift(presentation, amount); // degree-admissibility
    Matrix shifted = presentation;
    shifted.shift(amount);
    auto additional = homomorphism_lift_basis(presentation, shifted);
    auto inherited = homomorphism_lift_basis(presentation, presentation);
    const auto shifted_dimension = additional.size();
    for (auto& lift : inherited) {
        lift.row_degrees = shifted.row_degrees;
        lift.refresh_compatible_sorted();
    }
    reduce_matrix_family_modulo(inherited, additional);
    if (info)
        std::cout << "dim additional lifts = " << additional.size()
                  << " vs dim shifted lifts = " << shifted_dimension
                  << " and dim endomorphism lifts = " << inherited.size() << std::endl;
    return additional;
}

} // namespace graded_linalg
