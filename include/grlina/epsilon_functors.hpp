/** @file epsilon_functors.hpp
 * @brief Object parts of the image and kernel functors for diagonal shifts.
 */
#pragma once

#include <cmath>
#include <grlina/hom_operations.hpp>
#include <grlina/coordinate_degree.hpp>
#include <grlina/r3graded_matrix.hpp>

namespace graded_linalg {
namespace detail {

template <typename Degree> struct EpsilonDiagonal;

template <> struct EpsilonDiagonal<r2degree> {
    static r2degree make(double epsilon) { return {epsilon, epsilon}; }
};

template <> struct EpsilonDiagonal<triple> {
    static triple make(double epsilon) { return {epsilon, epsilon, epsilon}; }
};

template <std::size_t Dimension>
struct EpsilonDiagonal<CoordinateDegree<double, Dimension>> {
    static CoordinateDegree<double, Dimension> make(double epsilon) {
        CoordinateDegree<double, Dimension> result;
        result.coordinates.fill(epsilon);
        return result;
    }
};

inline void validate_functor_epsilon(double epsilon) {
    if (!std::isfinite(epsilon) || epsilon < 0)
        throw std::invalid_argument("Epsilon must be finite and nonnegative");
}

} // namespace detail

/** I_epsilon(X) = im(X[-epsilon] -> X).
 * Each input generator of degree a gives a generator of degree a + epsilon.
 * The returned matrix uses the original generator coordinates and retains
 * the exact input parent. Generators are not minimized modulo its relations.
 * Only the action on objects is implemented.
 */
template <typename Matrix>
Submodule<Matrix> epsilon_image(std::shared_ptr<const Module<Matrix>> module,
                                double epsilon) {
    detail::validate_functor_epsilon(epsilon);
    auto image = Submodule<Matrix>::whole(std::move(module));
    image.shift_generators(detail::EpsilonDiagonal<typename Matrix::degree_type>::make(epsilon));
    return image;
}

/** K_epsilon(X) = ker(X -> X[epsilon]).
 * Returns generators in the original input generator coordinates, retaining
 * the exact input parent. Requires the matrix type's graded-kernel algorithm.
 * Generators are not minimized. Only the action on objects is implemented.
 */
template <typename Matrix>
Submodule<Matrix> epsilon_kernel(std::shared_ptr<const Module<Matrix>> module,
                                 double epsilon) {
    detail::validate_functor_epsilon(epsilon);
    const auto amount = detail::EpsilonDiagonal<typename Matrix::degree_type>::make(epsilon);
    return Homomorphism<Matrix>::canonical_shift(std::move(module), amount).kernel(false);
}

} // namespace graded_linalg
