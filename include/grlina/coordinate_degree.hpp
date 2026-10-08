/**
 * @file coordinate_degree.hpp
 * @brief Degree and matrix building blocks for product orders on R^n and Z^n.
 */
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

#include <grlina/graded_matrix.hpp>

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

template <typename Scalar, std::size_t Dimension, typename index, typename Derived,
          typename MatrixBase = SparseMatrix<index>>
struct CoordinateGradedSparseMatrix
    : GradedSparseMatrix<CoordinateDegree<Scalar, Dimension>, index, Derived, MatrixBase> {
    using degree_type = CoordinateDegree<Scalar, Dimension>;
    using Base = GradedSparseMatrix<degree_type, index, Derived, MatrixBase>;
    using Base::Base;

    CoordinateGradedSparseMatrix() = default;
    explicit CoordinateGradedSparseMatrix(MatrixBase&& other)
        : Base(std::move(other)) {}

    void sort_rows_colexicographically() {
        this->sort_rows(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<index> sort_rows_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<degree_type, index>(
            this->row_degrees, Degree_traits<degree_type>::colex_lambda());
        vec<index> reverse(permutation.size());
        for (index i = 0; i < static_cast<index>(permutation.size()); ++i)
            reverse[permutation[i]] = i;
        this->transform_data(reverse);
        this->sort_data();
        this->invalidate_cached_rows();
        this->compatible_order_ = Degree_traits<degree_type>::colex_lambda();
        this->compatibly_sorted = std::is_sorted(this->col_degrees.begin(), this->col_degrees.end(), this->compatible_order_);
        return permutation;
    }

    void sort_columns_colexicographically() {
        this->sort_columns(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<index> sort_columns_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<degree_type, index>(
            this->col_degrees, Degree_traits<degree_type>::colex_lambda());
        this->permute_columns(permutation);
        this->invalidate_cached_rows();
        this->compatible_order_ = Degree_traits<degree_type>::colex_lambda();
        this->compatibly_sorted = std::is_sorted(this->row_degrees.begin(), this->row_degrees.end(), this->compatible_order_);
        return permutation;
    }

    void sort_colexicographically() {
        this->sort_compatibly(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }
};

} // namespace graded_linalg
