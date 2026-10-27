/** Degree-poset customization, interface checks and compatible sorting. */
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace graded_linalg {

/** Specialize this for a degree poset; the primary template supplies no operations. */
template <typename D, typename = void>
struct Degree_traits {};

/** The comparison interface used by this library's poset algorithms.
 * Each of equals, smaller, greater, smaller_equal and greater_equal returns
 * bool. Mathematically equality and <= suffice; the other comparisons are
 * required convenience operations in the existing Degree_traits protocol.
 * Equality and <= must satisfy the partial-order laws.
 * All checks below inspect signatures, not function bodies or algebraic laws.
 */
template <typename D, typename = void>
struct is_poset_degree : std::false_type {};

template <typename D>
struct is_poset_degree<D, std::void_t<std::enable_if_t<
    std::is_same_v<decltype(Degree_traits<D>::equals(std::declval<const D&>(), std::declval<const D&>())), bool> &&
    std::is_same_v<decltype(Degree_traits<D>::smaller(std::declval<const D&>(), std::declval<const D&>())), bool> &&
    std::is_same_v<decltype(Degree_traits<D>::greater(std::declval<const D&>(), std::declval<const D&>())), bool> &&
    std::is_same_v<decltype(Degree_traits<D>::smaller_equal(std::declval<const D&>(), std::declval<const D&>())), bool> &&
    std::is_same_v<decltype(Degree_traits<D>::greater_equal(std::declval<const D&>(), std::declval<const D&>())), bool>
>>> : std::true_type {};

template <typename D>
inline constexpr bool is_poset_degree_v = is_poset_degree<D>::value;

/** Minimum interface for generic graded matrices: a poset with a supplied
 * linear extension for sorting. lex_order returns bool and lex_lambda returns
 * a comparator convertible to std::function<bool(const D&, const D&)>.
 * No lattice, arithmetic, serialization, printing or coordinates are required.
 * Colexicographic order remains optional.
 */
template <typename D, typename = void>
struct is_degree : std::false_type {};

template <typename D>
struct is_degree<D, std::void_t<std::enable_if_t<
    is_poset_degree_v<D> &&
    std::is_same_v<decltype(Degree_traits<D>::lex_order(std::declval<const D&>(), std::declval<const D&>())), bool> &&
    std::is_convertible_v<decltype(Degree_traits<D>::lex_lambda()), std::function<bool(const D&, const D&)>>
>>> : std::true_type {};

template <typename D>
inline constexpr bool is_degree_v = is_degree<D>::value;

// Optional operations are checked independently so algorithms can request only
// what they use (e.g. join without meet, or subtraction without addition).
template <typename D, typename = void>
struct has_degree_join : std::false_type {};
template <typename D>
struct has_degree_join<D, std::void_t<decltype(Degree_traits<D>::join(
    std::declval<const D&>(), std::declval<const D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::join(std::declval<const D&>(), std::declval<const D&>())), D> {};
template <typename D>
inline constexpr bool has_degree_join_v = has_degree_join<D>::value;

template <typename D, typename = void>
struct has_degree_meet : std::false_type {};
template <typename D>
struct has_degree_meet<D, std::void_t<decltype(Degree_traits<D>::meet(
    std::declval<const D&>(), std::declval<const D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::meet(std::declval<const D&>(), std::declval<const D&>())), D> {};
template <typename D>
inline constexpr bool has_degree_meet_v = has_degree_meet<D>::value;

/** join and meet must be the least upper and greatest lower bounds. */
template <typename D>
struct is_lattice_degree : std::bool_constant<
    is_poset_degree_v<D> && has_degree_join_v<D> && has_degree_meet_v<D>> {};
template <typename D>
inline constexpr bool is_lattice_degree_v = is_lattice_degree<D>::value;

template <typename D, typename = void>
struct has_degree_add : std::false_type {};
template <typename D>
struct has_degree_add<D, std::void_t<decltype(Degree_traits<D>::add(
    std::declval<const D&>(), std::declval<D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::add(std::declval<const D&>(), std::declval<D&>())), void> {};
template <typename D>
inline constexpr bool has_degree_add_v = has_degree_add<D>::value;

template <typename D, typename = void>
struct has_degree_subtract : std::false_type {};
template <typename D>
struct has_degree_subtract<D, std::void_t<decltype(Degree_traits<D>::subtract(
    std::declval<const D&>(), std::declval<D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::subtract(std::declval<const D&>(), std::declval<D&>())), void> {};
template <typename D>
inline constexpr bool has_degree_subtract_v = has_degree_subtract<D>::value;

template <typename D, typename = void>
struct has_degree_zero : std::false_type {};
template <typename D>
struct has_degree_zero<D, std::void_t<decltype(Degree_traits<D>::zero())>>
    : std::is_same<decltype(Degree_traits<D>::zero()), D> {};
template <typename D>
inline constexpr bool has_degree_zero_v = has_degree_zero<D>::value;

/** add(amount, value) mutates value to value + amount; subtract analogously.
 * zero() returns the identity, which need not be D{}. The operations must form
 * an abelian group and preserve order under translation:
 * a <= b implies a + c <= b + c. These are semantic requirements, not proofs
 * provided by the signature check.
 */
template <typename D>
struct is_ordered_abelian_group_degree : std::bool_constant<
    is_poset_degree_v<D> && has_degree_zero_v<D> &&
    has_degree_add_v<D> && has_degree_subtract_v<D>> {};
template <typename D>
inline constexpr bool is_ordered_abelian_group_degree_v = is_ordered_abelian_group_degree<D>::value;

/** A lattice-ordered abelian group (abelian l-group). Total order is not
 * required; every pair has a join and meet, and translations preserve order.
 * Sorting, I/O and geometric coordinates remain separate capabilities.
 */
template <typename D>
struct is_lattice_ordered_abelian_group_degree : std::bool_constant<
    is_lattice_degree_v<D> && is_ordered_abelian_group_degree_v<D>> {};
template <typename D>
inline constexpr bool is_lattice_ordered_abelian_group_degree_v =
    is_lattice_ordered_abelian_group_degree<D>::value;

/** Optional SCC serialization interface, independent of the algebraic checks. */
template <typename D, typename = void>
struct has_degree_io : std::false_type {};
template <typename D>
struct has_degree_io<D, std::void_t<std::enable_if_t<
    std::is_constructible_v<std::string, decltype(Degree_traits<D>::poset_id)> &&
    std::is_same_v<decltype(Degree_traits<D>::write_degree(std::declval<std::ostream&>(), std::declval<const D&>())), void> &&
    std::is_same_v<decltype(Degree_traits<D>::from_stream(std::declval<std::istream&>())), D>
>>> : std::true_type {};
template <typename D>
inline constexpr bool has_degree_io_v = has_degree_io<D>::value;

template <typename D, typename = void>
struct has_degree_position : std::false_type {};
template <typename D>
struct has_degree_position<D, std::void_t<decltype(Degree_traits<D>::position(std::declval<const D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::position(std::declval<const D&>())), std::vector<double>> {};
template <typename D>
inline constexpr bool has_degree_position_v = has_degree_position<D>::value;

template <typename D, typename = void>
struct has_degree_print : std::false_type {};
template <typename D>
struct has_degree_print<D, std::void_t<decltype(Degree_traits<D>::print_degree(std::declval<const D&>()))>>
    : std::is_same<decltype(Degree_traits<D>::print_degree(std::declval<const D&>())), void> {};
template <typename D>
inline constexpr bool has_degree_print_v = has_degree_print<D>::value;

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
template <typename D>
concept PosetDegree = is_poset_degree_v<D>;
template <typename D>
concept Degree = is_degree_v<D>;
template <typename D>
concept LatticeDegree = is_lattice_degree_v<D>;
template <typename D>
concept OrderedAbelianGroupDegree = is_ordered_abelian_group_degree_v<D>;
template <typename D>
concept LatticeOrderedAbelianGroupDegree = is_lattice_ordered_abelian_group_degree_v<D>;
#endif

// Orders supplied by Degree_traits are part of the poset contract. Only a
// caller-supplied comparator needs the quadratic finite-poset validation.
template <typename D> struct TraitLinearOrder {
    static_assert(is_degree_v<D>, "Degree_traits must provide poset comparisons and a linear extension");
    std::function<bool(const D&, const D&)> compare;
    bool operator()(const D& a, const D& b) const { return compare(a, b); }
};

} // namespace graded_linalg
