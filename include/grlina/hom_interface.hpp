/** @file hom_interface.hpp @brief Matrix and module interfaces for Hom algorithms. */
#pragma once

#include <memory>
#include <vector>

#include <grlina/homomorphisms.hpp>
#include <grlina/hom_operations.hpp>
#include <grlina/matrix_family.hpp>

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
std::vector<Matrix> End_2d_0(
    const Matrix& presentation, const typename Matrix::degree_type& amount,
    bool info = false) {
    (void)canonical_shift_lift(presentation, amount); // degree-admissibility
    Matrix shifted = presentation;
    shifted.shift(amount);
    auto additional = homomorphism_lift_basis(presentation, shifted);
    auto inherited = homomorphism_lift_basis(presentation, presentation);
    const auto shifted_dimension = additional.size();
    // Reduction uses coefficients only; the canonical identity changes none.
    reduce_matrix_family_modulo(inherited, additional);
    if (info)
        std::cout << "dim additional lifts = " << additional.size()
                  << " vs dim shifted lifts = " << shifted_dimension
                  << " and dim endomorphism lifts = " << inherited.size() << std::endl;
    return additional;
}

/** Additional lifts using an existing canonical structure map M -> M(amount).
 * Trusted input: canonical_shift must be the canonical shift with identity
 * coefficients in the same presentation coordinates (as returned by
 * Homomorphism<Matrix>::canonical_shift), not an arbitrary homomorphism.
 * Quotient by transported endomorphism LIFTS, not by target relations.
 * Every result retains exactly the supplied domain and target objects, so
 * submodules of canonical_shift.target() can be used directly with preimage.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_2d_0(
    const Homomorphism<Matrix>& canonical_shift, bool info = false) {
    const auto& domain = canonical_shift.domain();
    const auto& target = canonical_shift.target();
    if (!domain || !target)
        throw std::invalid_argument("End_2d_0 requires domain and target modules");
    const Matrix& presentation = domain->presentation();
    if (presentation.get_num_rows() == 0) {
        if (info)
            std::cout << "dim additional lifts = 0 vs dim shifted lifts = 0 and dim endomorphism lifts = 0" << std::endl;
        return {};
    }

    Matrix source = presentation;
    source.compute_rows_forward();
    auto additional = hom_space_basis_new(source, target->presentation());
    auto inherited = hom_space_basis_new(source, presentation);
    const auto shifted_dimension = additional.size();
    // The canonical identity leaves the transported coefficients unchanged.
    reduce_matrix_family_modulo(inherited, additional);
    if (info)
        std::cout << "dim additional lifts = " << additional.size()
                  << " vs dim shifted lifts = " << shifted_dimension
                  << " and dim endomorphism lifts = " << inherited.size() << std::endl;

    std::vector<Homomorphism<Matrix>> result;
    result.reserve(additional.size());
    for (auto& lift : additional)
        result.emplace_back(domain, target, std::move(lift));
    return result;
}

/** Additional lifts M -> M(amount), wrapped as homomorphisms. As in the
 * presentation overload, quotient by transported endomorphism LIFTS, not by
 * target relations. All results retain domain and share a presentation-only
 * shifted target. Computed lift matrices are moved into the result.
 */
template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_2d_0(
    std::shared_ptr<const Module<Matrix>> domain,
    const typename Matrix::degree_type& amount, bool info = false) {
    if (!domain) throw std::invalid_argument("End_2d_0 requires a module");
    const Matrix& presentation = domain->presentation();
    Matrix shifted = presentation;
    shifted.shift(amount);
    using Degree = typename Matrix::degree_type;
    if (!Degree_traits<Degree>::smaller_equal(Degree{}, amount))
        throw std::invalid_argument("Canonical shift requires a nonnegative amount");
    if (presentation.get_num_rows() == 0) {
        if (info)
            std::cout << "dim additional lifts = 0 vs dim shifted lifts = 0 and dim endomorphism lifts = 0" << std::endl;
        return {};
    }

    // Both Hom computations use the same source row cache.
    Matrix source = presentation;
    source.compute_rows_forward();
    auto additional = hom_space_basis_new(source, shifted);
    auto inherited = hom_space_basis_new(source, presentation);
    const auto shifted_dimension = additional.size();
    // Reduction uses coefficients only; transporting inherited lifts along
    // the canonical identity changes no coefficients.
    reduce_matrix_family_modulo(inherited, additional);
    if (info)
        std::cout << "dim additional lifts = " << additional.size()
                  << " vs dim shifted lifts = " << shifted_dimension
                  << " and dim endomorphism lifts = " << inherited.size() << std::endl;

    std::vector<Homomorphism<Matrix>> result;
    if (additional.empty()) return result;
    std::vector<Matrix> differentials;
    differentials.reserve(1);
    differentials.push_back(std::move(shifted));
    auto target = std::make_shared<const Module<Matrix>>(
        ChainComplex<Matrix>(std::move(differentials)));
    result.reserve(additional.size());
    for (auto& lift : additional)
        result.emplace_back(domain, target, std::move(lift));
    return result;
}

template <typename Matrix>
std::vector<Homomorphism<Matrix>> End_2d_0(
    std::shared_ptr<Module<Matrix>> domain,
    const typename Matrix::degree_type& amount, bool info = false) {
    return End_2d_0<Matrix>(std::shared_ptr<const Module<Matrix>>(std::move(domain)), amount, info);
}

} // namespace graded_linalg
