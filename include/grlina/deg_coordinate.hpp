/**
 * @file deg_coordinate.hpp
 * @brief Degree definitions and arithmetic for product orders on R^n and Z^n.
 */
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#include <grlina/deg_traits.hpp>
#include <grlina/general.hpp>

namespace graded_linalg {

/** Only explicitly supported scalar domains receive a grid identifier.
 * Custom coordinate domains must specialize this trait (or Degree_traits).
 */
template <typename Scalar> struct CoordinatePosetDomain;
template <> struct CoordinatePosetDomain<double> { static constexpr const char* suffix = ""; };
template <> struct CoordinatePosetDomain<float> { static constexpr const char* suffix = ""; };
template <> struct CoordinatePosetDomain<long double> { static constexpr const char* suffix = ""; };
template <> struct CoordinatePosetDomain<int> { static constexpr const char* suffix = "Z"; };
template <> struct CoordinatePosetDomain<long> { static constexpr const char* suffix = "Z"; };
template <> struct CoordinatePosetDomain<long long> { static constexpr const char* suffix = "Z"; };

template <typename Scalar, std::size_t Dimension>
using CoordinateDegree = std::array<Scalar, Dimension>;

template <typename Scalar, std::size_t Dimension>
std::istream& operator>>(std::istream& in, CoordinateDegree<Scalar, Dimension>& degree) {
    for (auto& value : degree) in >> value;
    return in;
}

template <typename Scalar, std::size_t Dimension>
struct Degree_traits<CoordinateDegree<Scalar, Dimension>,
                     std::void_t<decltype(CoordinatePosetDomain<Scalar>::suffix)>> {
    using degree_type = CoordinateDegree<Scalar, Dimension>;

    inline static const std::string poset_id =
        std::to_string(Dimension) + CoordinatePosetDomain<Scalar>::suffix;

    static constexpr degree_type zero() { return {}; }

    static bool equals(const degree_type& lhs, const degree_type& rhs) {
        return lhs == rhs;
    }

    static bool smaller_equal(const degree_type& lhs, const degree_type& rhs) {
        for (std::size_t i = 0; i < Dimension; ++i) {
            if (!(lhs[i] <= rhs[i])) return false;
        }
        return true;
    }

    static bool greater_equal(const degree_type& lhs, const degree_type& rhs) {
        return smaller_equal(rhs, lhs);
    }

    static bool smaller(const degree_type& lhs, const degree_type& rhs) {
        return smaller_equal(lhs, rhs) && !equals(lhs, rhs);
    }

    static bool greater(const degree_type& lhs, const degree_type& rhs) {
        return smaller(rhs, lhs);
    }

    static bool lex_order(const degree_type& lhs, const degree_type& rhs) {
        return lhs < rhs;
    }

    static bool colex_order(const degree_type& lhs, const degree_type& rhs) {
        for (std::size_t i = Dimension; i-- > 0;) {
            if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
        }
        return false;
    }

    static std::function<bool(const degree_type&, const degree_type&)> lex_lambda() {
        return lex_order;
    }

    static std::function<bool(const degree_type&, const degree_type&)> colex_lambda() {
        return colex_order;
    }

    static vec<double> position(const degree_type& degree) {
        vec<double> result;
        result.reserve(Dimension);
        for (const auto& value : degree) {
            result.push_back(static_cast<double>(value));
        }
        return result;
    }

    static void print_degree(const degree_type& degree) {
        std::cout << degree;
    }

    static degree_type join(const degree_type& lhs, const degree_type& rhs) {
        degree_type result{};
        for (std::size_t i = 0; i < Dimension; ++i) {
            result[i] = std::max(lhs[i], rhs[i]);
        }
        return result;
    }

    static degree_type meet(const degree_type& lhs, const degree_type& rhs) {
        degree_type result{};
        for (std::size_t i = 0; i < Dimension; ++i) {
            result[i] = std::min(lhs[i], rhs[i]);
        }
        return result;
    }

    template <typename OutputStream>
    static void write_degree(OutputStream& out, const degree_type& degree) {
        for (std::size_t i = 0; i < Dimension; ++i) {
            if (i != 0) out << " ";
            out << degree[i];
        }
    }

    template <typename InputStream>
    static degree_type from_stream(InputStream& in) {
        degree_type result{};
        for (std::size_t i = 0; i < Dimension; ++i) in >> result[i];
        return result;
    }

    static void add(const degree_type& amount, degree_type& degree) {
        for (std::size_t i = 0; i < Dimension; ++i) degree[i] += amount[i];
    }

    static void subtract(const degree_type& amount, degree_type& degree) {
        for (std::size_t i = 0; i < Dimension; ++i) degree[i] -= amount[i];
    }
};

using r2degree = CoordinateDegree<double, 2>;

// Vector addition
inline r2degree operator+(const r2degree& a,
    const r2degree& b) {
return {a[0] + b[0], a[1] + b[1]};
}

inline vec<r2degree> operator+(const vec<r2degree>& a,
    const r2degree& b) {
    vec<r2degree> result = a;
    for (auto& elem : result) {
        elem = elem + b;
    }
    return result;
}

// Vector subtraction
inline r2degree operator-(const r2degree& a,
    const r2degree& b) {
return {a[0] - b[0], a[1] - b[1]};
}


// Scalar multiplication (scalar * degree)
inline r2degree operator*(double scalar, const r2degree& p) {
return {scalar * p[0], scalar * p[1]};
}

// Scalar multiplication (degree * scalar)
inline r2degree operator*(const r2degree& p, double scalar) {
return {p[0] * scalar, p[1] * scalar};
}

inline r2degree operator/(const r2degree& p, double scalar) {
    return {p[0] / scalar, p[1] / scalar};
}

} // namespace graded_linalg
