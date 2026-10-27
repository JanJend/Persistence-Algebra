/** Finitely generated upsets of a two-coordinate product order, by reverse inclusion. */
#pragma once

#include <algorithm>
#include <iterator>
#include <grlina/posets/coordinate.hpp>

namespace graded_linalg {

/** Generators are lexicographically sorted by x, then y, with redundancy removed.
 * Thus x strictly increases and y strictly decreases along the list.
 * The empty vector represents the empty upset, the greatest element.
 */
template <typename Scalar>
using r2upset = vec<CoordinateDegree<Scalar, 2>>;

template <typename Scalar>
struct Degree_traits<r2upset<Scalar>,
                     std::void_t<decltype(CoordinatePosetDomain<Scalar>::suffix)>> {
    inline static constexpr const char* poset_id = "2";

    /** Remove duplicates and dominated generators in O(n), preserving order. */
    static void remove_redundant(r2upset<Scalar>& degree) {
        std::size_t kept = 0;
        for (const auto& point : degree) {
            if (kept == 0 || point[1] < degree[kept - 1][1]) {
                degree[kept++] = point;
            }
        }
        degree.resize(kept);
    }

    /** Every lhs generator lies above a rhs generator; O(n + m). */
    static bool greater_equal(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        auto next = rhs.begin();
        const CoordinateDegree<Scalar, 2>* lowest = nullptr;
        for (const auto& point : lhs) {
            while (next != rhs.end() && (*next)[0] <= point[0]) {
                lowest = &*next;
                ++next;
            }
            if (!lowest || !Degree_traits<CoordinateDegree<Scalar, 2>>::smaller_equal(*lowest, point)) return false;
        }
        return true;
    }

    static bool smaller_equal(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        return greater_equal(rhs, lhs);
    }

    static bool equals(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        return lhs == rhs;
    }

    static bool smaller(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        return smaller_equal(lhs, rhs) && !equals(lhs, rhs);
    }

    static bool greater(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        return smaller(rhs, lhs);
    }

    /** Compare minimal generators lexicographically, with longer prefixes first.
     * This puts containing upsets first.
     */
    static bool lex_order(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        auto a = lhs.begin(), b = rhs.begin();
        while (a != lhs.end() && b != rhs.end()) {
            if (*a != *b) return Degree_traits<CoordinateDegree<Scalar, 2>>::lex_order(*a, *b);
            ++a;
            ++b;
        }
        return a != lhs.end();
    }

    /** Analogous extension from the smallest y, comparing y before x. */
    static bool colex_order(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        auto i = lhs.rbegin(), j = rhs.rbegin();
        while (i != lhs.rend() && j != rhs.rend()) {
            if (*i != *j) return Degree_traits<CoordinateDegree<Scalar, 2>>::colex_order(*i, *j);
            ++i;
            ++j;
        }
        return i != lhs.rend();
    }

    static std::function<bool(const r2upset<Scalar>&, const r2upset<Scalar>&)> lex_lambda() {
        return lex_order;
    }

    static std::function<bool(const r2upset<Scalar>&, const r2upset<Scalar>&)> colex_lambda() {
        return colex_order;
    }

    /** Union is the greatest lower bound for reverse inclusion; O(n + m). */
    static r2upset<Scalar> meet(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        r2upset<Scalar> result;
        result.reserve(lhs.size() + rhs.size());
        std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                   std::back_inserter(result));
        remove_redundant(result);
        return result;
    }

    /** Intersection is the least upper bound. Sweep the two staircase boundaries
     * in O(n + m), emitting a generator whenever their maximum height drops.
     */
    static r2upset<Scalar> join(const r2upset<Scalar>& lhs, const r2upset<Scalar>& rhs) {
        r2upset<Scalar> result;
        if (lhs.empty() || rhs.empty()) return result;
        auto a = lhs.begin(), b = rhs.begin();
        const CoordinateDegree<Scalar, 2>* left = nullptr;
        const CoordinateDegree<Scalar, 2>* right = nullptr;
        while (a != lhs.end() || b != rhs.end()) {
            const Scalar x = a == lhs.end() ? (*b)[0]
                : b == rhs.end() ? (*a)[0] : std::min((*a)[0], (*b)[0]);
            // Consume both boundaries at the same x before emitting a corner.
            if (a != lhs.end() && (*a)[0] == x) {
                left = &*a;
                ++a;
            }
            if (b != rhs.end() && (*b)[0] == x) {
                right = &*b;
                ++b;
            }
            if (left && right) {
                const auto point = Degree_traits<CoordinateDegree<Scalar, 2>>::join(*left, *right);
                if (result.empty() || point[1] < result.back()[1]) result.push_back(point);
            }
        }
        return result;
    }

    static void print_degree(const r2upset<Scalar>& degree) {
        std::cout << degree;
    }

    /** Extended SCC degree format: x/y pairs, terminated by the caller's ';'. */
    template <typename OutputStream>
    static void write_degree(OutputStream& out, const r2upset<Scalar>& degree) {
        for (std::size_t i = 0; i < degree.size(); ++i) {
            if (i != 0) out << ' ';
            Degree_traits<CoordinateDegree<Scalar, 2>>::write_degree(out, degree[i]);
        }
    }

    template <typename InputStream>
    static r2upset<Scalar> from_stream(InputStream& in) {
        r2upset<Scalar> result;
        // Leave the separator for the matrix/chain-complex reader.
        while (in >> std::ws && in.peek() != ';') {
            CoordinateDegree<Scalar, 2> point;
            if (!(in >> point[0] >> point[1])) return {};
            result.push_back(point);
        }
        if (!in || in.peek() != ';') {
            in.setstate(std::ios::failbit);
            return {};
        }
        std::sort(result.begin(), result.end());
        remove_redundant(result);
        return result;
    }
};

} // namespace graded_linalg
