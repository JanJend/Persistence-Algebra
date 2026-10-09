/** @file epsilon_functors.hpp
 * @brief Object parts of the image and kernel functors for diagonal shifts.
 */
#pragma once

#include <cmath>
#include <limits>
#include <grlina/submodule.hpp>
#include <grlina/coordinate_degree.hpp>
#include <grlina/r3graded_matrix.hpp>

namespace graded_linalg {
namespace detail {

template <typename Degree> struct EpsilonDiagonal;

template <std::size_t Dimension>
struct EpsilonDiagonal<CoordinateDegree<double, Dimension>> {
    static CoordinateDegree<double, Dimension> make(double epsilon) {
        CoordinateDegree<double, Dimension> result;
        result.fill(epsilon);
        return result;
    }
};

/** Geometry-aware diagonal, retaining the matrix's runtime parameter count. */
template <typename Matrix>
matrix_geometry_degree_t<Matrix> epsilon_diagonal(const Matrix& matrix, double epsilon) {
    if constexpr (GradedMatrixIO<Matrix>::runtime_dimension) {
        using Degree = matrix_geometry_degree_t<Matrix>;
        using Scalar = typename Degree::value_type;
        if constexpr (std::is_integral_v<Scalar>) {
            if (std::trunc(epsilon) != epsilon || epsilon > static_cast<long double>(std::numeric_limits<Scalar>::max()))
                throw std::invalid_argument("Epsilon cannot be represented by the grid scalar type");
        }
        return Degree(vec<Scalar>(matrix.parameter_count(), static_cast<Scalar>(epsilon)));
    } else return EpsilonDiagonal<typename Matrix::degree_type>::make(epsilon);
}

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
    if (!module) throw std::invalid_argument("Epsilon image requires a module");
    const auto amount = detail::epsilon_diagonal(module->presentation(), epsilon);
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
    detail::validate_functor_epsilon(epsilon);
    if (!module) throw std::invalid_argument("Epsilon kernel requires a module");
    const auto amount = detail::epsilon_diagonal(module->presentation(), epsilon);
    return Homomorphism<Matrix>::canonical_shift(std::move(module), amount).kernel(false);
}

} // namespace graded_linalg
