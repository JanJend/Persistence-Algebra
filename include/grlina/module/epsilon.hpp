/** @file module/epsilon.hpp
 * @brief Object parts of the image and kernel functors for diagonal shifts.
 */
#pragma once

#include <cmath>
#include <limits>
#include <grlina/module/submodule.hpp>
#include <grlina/posets/coordinate.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

template <typename Degree> struct EpsilonDiagonal;

template <typename Scalar, std::size_t Dimension>
struct EpsilonDiagonal<CoordinateDegree<Scalar, Dimension>> {
    static CoordinateDegree<Scalar, Dimension> make(double epsilon) {
        if constexpr (std::is_integral_v<Scalar>) {
            GRLINA_DEBUG_CHECK(if (!std::isfinite(epsilon) || std::trunc(epsilon) != epsilon ||
                epsilon < static_cast<long double>(std::numeric_limits<Scalar>::lowest()) ||
                epsilon > static_cast<long double>(std::numeric_limits<Scalar>::max()))
                throw std::invalid_argument("Epsilon cannot be represented by the degree scalar type"));
        }
        CoordinateDegree<Scalar, Dimension> result;
        result.fill(static_cast<Scalar>(epsilon));
        return result;
    }
};

/** Use physical coordinates for grid matrices. */
template <typename Matrix>
matrix_geometry_degree_t<Matrix> epsilon_diagonal(const Matrix& matrix, double epsilon) {
    return EpsilonDiagonal<matrix_geometry_degree_t<Matrix>>::make(epsilon);
}


/** I_epsilon(X) = im(X[-epsilon] -> X).
 * Each input generator of degree a gives a generator of degree a + epsilon.
 * The returned matrix uses the original generator coordinates and retains
 * the exact input parent. Generators are not minimized modulo its relations.
 * Only the action on objects is implemented.
 */
template <typename Matrix>
Submodule<Matrix> epsilon_image(std::shared_ptr<const Module<Matrix>> module,
                                double epsilon) {
    GRLINA_DEBUG_CHECK(if (!module) throw std::invalid_argument("Epsilon image requires a module"));
    const auto amount = epsilon_diagonal(module->presentation(), epsilon);
    auto image = Submodule<Matrix>::whole(std::move(module));
    image.shift_generators(amount);
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
    GRLINA_DEBUG_CHECK(if (!module) throw std::invalid_argument("Epsilon kernel requires a module"));
    const auto amount = epsilon_diagonal(module->presentation(), epsilon);
    return Homomorphism<Matrix>::canonical_shift(std::move(module), amount).kernel(false);
}

} // namespace graded_linalg
