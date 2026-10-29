#include <grlina/invariants/hilbert_function.hpp>
#include <grlina/matrices/coord.hpp>
#include <grlina/hom_interface.hpp>
#include <grlina/graded_linear_system.hpp>

#include <array>
#include <cassert>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <grlina/matrices/matrix_algorithms.hpp>

using namespace graded_linalg;
using Matrix = R2GradedSparseMatrix<int>;
using TestModule = R2Module<int>;

static_assert(is_graded_sparse_matrix_v<Matrix>);
static_assert(std::is_base_of_v<GradedSparseMatrix<r2degree, int, Matrix>, Matrix>);
static_assert(std::is_same_v<r2degree, std::array<double, 2>>);
static_assert(is_degree<r2degree>::value);
static_assert(is_degree_v<CoordinateDegree<float, 1>>);
static_assert(is_degree_v<CoordinateDegree<long double, 4>>);
static_assert(is_degree_v<CoordinateDegree<int, 3>>);
static_assert(is_degree_v<CoordinateDegree<long, 5>>);
static_assert(!is_degree_v<int>);
static_assert(!is_degree_v<CoordinateDegree<std::string, 2>>);

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
static_assert(Degree<r2degree>);
static_assert(!Degree<int>);
#endif

struct NotAGradedMatrix {
    vec<r2degree> col_degrees;
    array<int> data;
    int get_num_rows() const;
};
static_assert(!is_graded_sparse_matrix_v<NotAGradedMatrix>);

struct InvalidMatrixDegree {
    vec<int> col_degrees;
    array<int> data;
    int get_num_rows() const;
};
static_assert(!is_graded_sparse_matrix_v<InvalidMatrixDegree>);

struct NamedDegree {
    int rank = 0;
    bool operator==(const NamedDegree& other) const { return rank == other.rank; }
    friend std::ostream& operator<<(std::ostream& out, const NamedDegree& degree) {
        return out << degree.rank;
    }
};

struct MissingDegreeTraits {};
struct IncompleteDegreeTraits {};
struct WrongDegreeResults : NamedDegree {};
struct LatticeOnlyDegree : NamedDegree {};
struct GroupOnlyDegree : NamedDegree {};

// root <= left and root <= right; the incomparable branches have no join.
// No coordinates, serialization, printing, arithmetic or lattice operations.
enum class ForkDegree { root, left, right };
struct UnsortedForkDegree { ForkDegree value; };

namespace graded_linalg {

template <>
struct Degree_traits<NamedDegree> {
    inline static constexpr const char* poset_id = "named-poset";
    static NamedDegree zero() { return {0}; }
    static bool equals(const NamedDegree& a, const NamedDegree& b) { return a == b; }
    static bool smaller_equal(const NamedDegree& a, const NamedDegree& b) { return a.rank <= b.rank; }
    static bool greater_equal(const NamedDegree& a, const NamedDegree& b) { return a.rank >= b.rank; }
    static bool smaller(const NamedDegree& a, const NamedDegree& b) { return a.rank < b.rank; }
    static bool greater(const NamedDegree& a, const NamedDegree& b) { return a.rank > b.rank; }
    static bool lex_order(const NamedDegree& a, const NamedDegree& b) { return a.rank < b.rank; }
    static std::function<bool(const NamedDegree&, const NamedDegree&)> lex_lambda() {
        return lex_order;
    }
    static vec<double> position(const NamedDegree& degree) { return {double(degree.rank)}; }
    static void print_degree(const NamedDegree& degree) { std::cout << degree.rank; }
    static NamedDegree join(const NamedDegree& a, const NamedDegree& b) {
        return {std::max(a.rank, b.rank)};
    }
    static NamedDegree meet(const NamedDegree& a, const NamedDegree& b) {
        return {std::min(a.rank, b.rank)};
    }
    template <typename OutputStream>
    static void write_degree(OutputStream& out, const NamedDegree& degree) { out << degree.rank; }
    template <typename InputStream>
    static NamedDegree from_stream(InputStream& in) {
        NamedDegree degree;
        in >> degree.rank;
        return degree;
    }
    static void add(const NamedDegree& amount, NamedDegree& degree) { degree.rank += amount.rank; }
    static void subtract(const NamedDegree& amount, NamedDegree& degree) { degree.rank -= amount.rank; }
};

template <>
struct Degree_traits<IncompleteDegreeTraits> {
    static bool equals(const IncompleteDegreeTraits&, const IncompleteDegreeTraits&) { return true; }
};

// Every operation is present and callable, but the lattice and input functions
// return NamedDegree instead of the degree type being checked.
template <>
struct Degree_traits<WrongDegreeResults> : Degree_traits<NamedDegree> {};

template <>
struct Degree_traits<LatticeOnlyDegree> : Degree_traits<NamedDegree> {
    static LatticeOnlyDegree join(const LatticeOnlyDegree& a, const LatticeOnlyDegree& b) {
        return {{std::max(a.rank, b.rank)}};
    }
    static LatticeOnlyDegree meet(const LatticeOnlyDegree& a, const LatticeOnlyDegree& b) {
        return {{std::min(a.rank, b.rank)}};
    }
    static void add(const LatticeOnlyDegree&, LatticeOnlyDegree&) = delete;
    static void subtract(const LatticeOnlyDegree&, LatticeOnlyDegree&) = delete;
    static LatticeOnlyDegree zero() = delete;
};

template <>
struct Degree_traits<GroupOnlyDegree> : Degree_traits<NamedDegree> {
    static GroupOnlyDegree zero() { return {{0}}; }
    static GroupOnlyDegree join(const GroupOnlyDegree&, const GroupOnlyDegree&) = delete;
    static GroupOnlyDegree meet(const GroupOnlyDegree&, const GroupOnlyDegree&) = delete;
};

template <>
struct Degree_traits<ForkDegree> {
    static bool equals(ForkDegree a, ForkDegree b) { return a == b; }
    static bool smaller_equal(ForkDegree a, ForkDegree b) { return a == b || a == ForkDegree::root; }
    static bool greater_equal(ForkDegree a, ForkDegree b) { return smaller_equal(b, a); }
    static bool smaller(ForkDegree a, ForkDegree b) { return a != b && smaller_equal(a, b); }
    static bool greater(ForkDegree a, ForkDegree b) { return smaller(b, a); }
    static bool lex_order(ForkDegree a, ForkDegree b) { return a < b; }
    static auto lex_lambda() { return [](ForkDegree a, ForkDegree b) { return lex_order(a, b); }; }
};

template <>
struct Degree_traits<UnsortedForkDegree> {
    static bool equals(const UnsortedForkDegree& a, const UnsortedForkDegree& b) {
        return Degree_traits<ForkDegree>::equals(a.value, b.value);
    }
    static bool smaller_equal(const UnsortedForkDegree& a, const UnsortedForkDegree& b) {
        return Degree_traits<ForkDegree>::smaller_equal(a.value, b.value);
    }
    static bool greater_equal(const UnsortedForkDegree& a, const UnsortedForkDegree& b) {
        return smaller_equal(b, a);
    }
    static bool smaller(const UnsortedForkDegree& a, const UnsortedForkDegree& b) {
        return !equals(a, b) && smaller_equal(a, b);
    }
    static bool greater(const UnsortedForkDegree& a, const UnsortedForkDegree& b) { return smaller(b, a); }
};

struct ForkGradedMatrix : GradedSparseMatrix<ForkDegree, int, ForkGradedMatrix> {
    using Base = GradedSparseMatrix<ForkDegree, int, ForkGradedMatrix>;
    using Base::Base;
    ForkGradedMatrix() = default;
    explicit ForkGradedMatrix(SparseMatrix<int>&& matrix) : Base(std::move(matrix)) {}
};

template <typename index>
struct NamedGradedMatrix
    : GradedSparseMatrix<NamedDegree, index, NamedGradedMatrix<index>> {
    using Base = GradedSparseMatrix<NamedDegree, index, NamedGradedMatrix<index>>;
    using Base::Base;
    NamedGradedMatrix() = default;
    explicit NamedGradedMatrix(SparseMatrix<index>&& matrix) : Base(std::move(matrix)) {}
};

} // namespace graded_linalg

static_assert(is_degree_v<NamedDegree>);
static_assert(!is_degree_v<MissingDegreeTraits>);
static_assert(!is_degree_v<IncompleteDegreeTraits>);
static_assert(is_degree_v<WrongDegreeResults>); // Optional results do not invalidate comparisons.
static_assert(!is_lattice_degree_v<WrongDegreeResults>);
static_assert(!is_ordered_abelian_group_degree_v<WrongDegreeResults>);
static_assert(!has_degree_io_v<WrongDegreeResults>);
static_assert(is_degree_v<ForkDegree> && is_poset_degree_v<ForkDegree>);
static_assert(!is_lattice_degree_v<ForkDegree> && !is_ordered_abelian_group_degree_v<ForkDegree>);
static_assert(!has_degree_io_v<ForkDegree> && !has_degree_position_v<ForkDegree> && !has_degree_print_v<ForkDegree>);
static_assert(is_poset_degree_v<UnsortedForkDegree> && !is_degree_v<UnsortedForkDegree>);
static_assert(is_lattice_degree_v<LatticeOnlyDegree> && !is_ordered_abelian_group_degree_v<LatticeOnlyDegree>);
static_assert(is_ordered_abelian_group_degree_v<GroupOnlyDegree> && !is_lattice_degree_v<GroupOnlyDegree>);
static_assert(is_lattice_ordered_abelian_group_degree_v<NamedDegree>);
static_assert(is_lattice_ordered_abelian_group_degree_v<r2degree>);
static_assert(is_lattice_ordered_abelian_group_degree_v<CoordinateDegree<long, 5>>);
static_assert(!is_poset_degree_v<IncompleteDegreeTraits> && !is_lattice_ordered_abelian_group_degree_v<int>);

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
static_assert(PosetDegree<UnsortedForkDegree> && !Degree<UnsortedForkDegree>);
static_assert(Degree<ForkDegree> && !LatticeDegree<ForkDegree>);
static_assert(LatticeDegree<LatticeOnlyDegree> && !OrderedAbelianGroupDegree<LatticeOnlyDegree>);
static_assert(OrderedAbelianGroupDegree<GroupOnlyDegree> && !LatticeDegree<GroupOnlyDegree>);
static_assert(LatticeOrderedAbelianGroupDegree<r2degree>);
#endif

static void test_poset_without_optional_operations() {
    using D = ForkDegree;
    vec<D> support{D::root, D::left, D::right};
    assert((minimal_directed_graph<D, int>(support) == array<int>{{1, 2}, {}, {}}));
    vec<UnsortedForkDegree> unsorted_support{{D::root}, {D::left}, {D::right}};
    assert((minimal_directed_graph<UnsortedForkDegree, int>(unsorted_support) == array<int>{{1, 2}, {}, {}}));

    ForkGradedMatrix matrix(2, 1, {{0}, {0}}, {D::right, D::left}, {D::root});
    assert(matrix.is_graded_matrix());
    matrix.sort_compatibly();
    assert(matrix.compatibly_sorted);
    assert((matrix.col_degrees == vec<D>{D::left, D::right}));
    Module<ForkGradedMatrix> module(matrix);
    assert(support_degrees(module) == support);
    const auto values = module.dimension_at(support);
    assert(values[0] == 1 && values[1] == 0 && values[2] == 0);
    matrix.cancel_local_pairs();
    assert(matrix.get_num_cols() == 2); // No relation has the same degree as a generator.

    ForkGradedMatrix cancellable(1, 1, {{0}}, {D::left}, {D::left});
    cancellable.cancel_local_pairs();
    assert(cancellable.get_num_cols() == 0 && cancellable.get_num_rows() == 0);

    ForkGradedMatrix lhs(1, 1, {{0}}, {D::left}, {D::root});
    ForkGradedMatrix rhs(1, 1, {{0}}, {D::right}, {D::root});
    assert(graded_linear_system_is_solvable(lhs, lhs));
    assert(!graded_linear_system_is_solvable(lhs, rhs));
    // Invalid-degree diagnostics also work when degree printing is absent.
    lhs.row_degrees = {D::right};
    std::ostringstream diagnostic;
    auto* previous = std::cout.rdbuf(diagnostic.rdbuf());
    const bool valid = lhs.is_graded_matrix(true);
    std::cout.rdbuf(previous);
    assert(!valid && diagnostic.str().find("not admissible") != std::string::npos);
}

static void test_array_degree_traits() {
    using D = CoordinateDegree<long, 5>;
    using Traits = Degree_traits<D>;
    assert(Traits::zero() == D{});
    const D a{1, 4, -2, 0, 3}, b{2, 3, -2, 0, 1};
    vec<D> degrees{a, b};
    assert(degrees.data() + 1 == &degrees[1]);
    assert(a.data() + 4 == &a[4]);
    assert(a.size() == 5);
    assert(Traits::poset_id == "5Z");
    assert(!Traits::smaller_equal(a, b) && !Traits::greater_equal(a, b));
    assert(!Traits::smaller(a, b) && !Traits::greater(a, b));
    assert(Traits::lex_order(a, b) && Traits::lex_lambda()(a, b));
    assert(Traits::colex_order(b, a) && Traits::colex_lambda()(b, a));
    assert(Traits::equals(a, a) && !Traits::smaller(a, a));
    const D upper = Traits::join(a, b), lower = Traits::meet(a, b);
    assert((upper == D{2, 4, -2, 0, 3}));
    assert((lower == D{1, 3, -2, 0, 1}));
    assert(Traits::smaller(a, upper) && Traits::greater(a, lower));
    assert(Traits::position(a) == vec<double>({1, 4, -2, 0, 3}));
    D shifted = a;
    Traits::add(b, shifted);
    assert((shifted == D{3, 7, -4, 0, 4}));
    Traits::subtract(b, shifted);
    assert(shifted == a);
    std::stringstream encoded;
    Traits::write_degree(encoded, a);
    assert(encoded.str() == "1 4 -2 0 3");
    assert(Traits::from_stream(encoded) == a);
    std::ostringstream containers;
    containers << vec<r2degree>{{1, 2}, {3, 4}} << std::set<r2degree>{{1, 2}};
    assert(containers.str() == "(1, 2) (3, 4) {(1, 2)}");

    using FloatTraits = Degree_traits<CoordinateDegree<float, 1>>;
    assert(FloatTraits::poset_id == "1");
    assert(FloatTraits::smaller({0.5f}, {1.25f}));
    assert(FloatTraits::join({0.5f}, {1.25f})[0] == 1.25f);
    std::stringstream floating;
    FloatTraits::write_degree(floating, {-0.5f});
    assert(FloatTraits::from_stream(floating)[0] == -0.5f);
    const CoordinateDegree<float, 1> nan{std::numeric_limits<float>::quiet_NaN()};
    assert(!FloatTraits::smaller_equal(nan, {0.0f}));
    assert(!FloatTraits::smaller_equal({0.0f}, nan));
}

static Matrix interval_presentation() {
    Matrix result(1, 1);
    result.col_degrees = {{1.0, 1.0}};
    result.row_degrees = {{0.0, 0.0}};
    result.data = {{0}};
    return result;
}

static Matrix cancellation_presentation() {
    // r0 = g0 + g1 has the same degree as both generators.  The later
    // relation r1 = x*y*g0 must become x*y*g1 before r0/g0 are deleted.
    return Matrix(2, 2, array<int>{{0, 1}, {0}},
                  vec<r2degree>{{0.0, 0.0}, {1.0, 1.0}},
                  vec<r2degree>{{0.0, 0.0}, {0.0, 0.0}});
}

static void test_sort_state_and_degree_traits() {
    Matrix matrix = interval_presentation();
    assert(!matrix.compatibly_sorted);
    matrix.sort_columns_lexicographically();
    assert(matrix.compatibly_sorted); // the one row was already sorted
    matrix.sort_compatibly();
    assert(matrix.compatibly_sorted);
    assert(std::string(Degree_traits<r2degree>::poset_id) == "2");
    assert((Degree_traits<CoordinateDegree<double, 4>>::poset_id == "4"));
    assert((Degree_traits<CoordinateDegree<long long, 2>>::poset_id == "2Z"));
    assert((Degree_traits<CoordinateDegree<long long, 3>>::poset_id == "3Z"));
    assert((Degree_traits<CoordinateDegree<long long, 4>>::poset_id == "4Z"));

    Matrix checked_constructor(1, 1, array<int>{{0}},
                               vec<r2degree>{{1.0, 1.0}},
                               vec<r2degree>{{0.0, 0.0}});
    assert(!checked_constructor.compatibly_sorted);
    checked_constructor.append_column({0}, {2.0, 2.0});
    assert(!checked_constructor.compatibly_sorted);
    checked_constructor.sort_compatibly();
    assert(checked_constructor.compatibly_sorted);

    Matrix reordered(1, 2, array<int>{{0}},
                     vec<r2degree>{{1.0, 1.0}},
                     vec<r2degree>{{0.0, 0.0}, {1.0, 1.0}});
    reordered.sort_compatibly();
    assert(reordered.compatibly_sorted);
    reordered.permute_rows_graded({1, 0});
    assert(!reordered.compatibly_sorted);

    Matrix colex(2, 1, array<int>{{0}, {0}},
                 vec<r2degree>{{0.0, 1.0}, {1.0, 0.0}},
                 vec<r2degree>{{0.0, 0.0}});
    colex.sort_colexicographically();
    assert(colex.compatibly_sorted);
    assert((colex.col_degrees.front() == r2degree{1.0, 0.0}));
    colex.minimize();
    assert(colex.compatibly_sorted);
    assert(colex.get_num_cols() == 2); // incomparable relations are both needed
}

static void test_checked_chain_complex_sorting() {
    const std::string unsorted_scc =
        "scc2020\n2\n2 1 0\n"
        "2 2 ; 0\n"
        "1 1 ; 0\n"
        "0 0 ;\n";
    std::stringstream unchecked_input(unsorted_scc);
    auto unchecked = ChainComplex<Matrix>::from_stream(unchecked_input);
    assert(!unchecked[0].compatibly_sorted);

    std::stringstream sorted_input(unsorted_scc);
    auto sorted = ChainComplex<Matrix>::from_stream(sorted_input, true);
    assert(sorted[0].compatibly_sorted);
    assert((sorted[0].col_degrees.front() == r2degree{1.0, 1.0}));
}

static void assert_cancellation_result(const Matrix& minimized) {
    assert(minimized.get_num_rows() == 1);
    assert(minimized.get_num_cols() == 1);
    assert((minimized.row_degrees == vec<r2degree>{{0.0, 0.0}}));
    assert((minimized.col_degrees == vec<r2degree>{{1.0, 1.0}}));
    assert(minimized.data == array<int>{{0}});
    assert(minimized.compatibly_sorted);
}

static void test_correct_minimization() {
    Matrix original = cancellation_presentation();
    original.sort_compatibly();
    assert(original.compatibly_sorted);
    assert(original.dim_at({0.0, 0.0}) == 1);
    assert(original.dim_at({1.0, 1.0}) == 0);
    original.compute_rows_forward();
    assert(original.rows_computed);

    Matrix minimized = original;
    minimized.minimize();
    assert_cancellation_result(minimized);
    assert(minimized.rows_computed && minimized._rows == original._rows);
    minimized.compute_rows_forward();
    assert(minimized._rows == array<int>{{0}});
    assert(minimized.dim_at({0.0, 0.0}) == 1);
    assert(minimized.dim_at({1.0, 1.0}) == 0);
    assert(minimized.is_minimal());
    Matrix idempotent = minimized;
    idempotent.minimize();
    assert_cancellation_result(idempotent);

    Matrix variant = original;
    variant.minimize_variant();
    assert_cancellation_result(variant);
    Matrix semi = original;
    semi.semi_minimize();
    assert_cancellation_result(semi);

    // The degree-(2,2) relation is generated by the degree-(1,1)
    // relation and must disappear by an admissible column operation.
    Matrix redundant(2, 1, array<int>{{0}, {0}},
                     vec<r2degree>{{1.0, 1.0}, {2.0, 2.0}},
                     vec<r2degree>{{0.0, 0.0}});
    redundant.sort_compatibly();
    redundant.minimize();
    assert(redundant.get_num_cols() == 1);
    assert((redundant.col_degrees.front() == r2degree{1.0, 1.0}));
    assert(redundant.data == array<int>{{0}});

    Matrix two_cancellations(
        3, 3, array<int>{{0, 1}, {1, 2}, {0}},
        vec<r2degree>{{0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}},
        vec<r2degree>{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
    two_cancellations.sort_compatibly();
    two_cancellations.minimize();
    assert_cancellation_result(two_cancellations);

    Matrix unsorted(2, 1, array<int>{{0}, {0}},
                    vec<r2degree>{{2.0, 2.0}, {1.0, 1.0}},
                    vec<r2degree>{{0.0, 0.0}});
    assert(!unsorted.compatibly_sorted);
    Matrix unordered_copy = unsorted;
    unordered_copy.minimize(); // cancellation and redundancy removal do not require ordering
    assert(unordered_copy.dim_at({1,1}) == unsorted.dim_at({1,1}));
    bool reduction_rejected_unsorted = false;
    try {
        unsorted.column_reduction_graded();
    } catch (const std::invalid_argument&) {
        reduction_rejected_unsorted = true;
    }
    assert(reduction_rejected_unsorted);

    Matrix stale(2, 1, array<int>{{0}, {0}},
                 vec<r2degree>{{1.0, 1.0}, {2.0, 2.0}},
                 vec<r2degree>{{0.0, 0.0}});
    stale.sort_compatibly();
    assert(stale.compatibly_sorted);
    std::swap(stale.col_degrees[0], stale.col_degrees[1]);
    assert(stale.compatibly_sorted); // caller maintains metadata after direct edits
    stale.compatibly_sorted = false;
    assert(!stale.compatibly_sorted);

    TestModule auto_sorting(unsorted);
    auto_sorting.minimize();
    assert(auto_sorting.presentation().compatibly_sorted);
    TestModule strict(unsorted);
    bool strict_rejected = false;
    try {
        strict.minimize(false);
    } catch (const std::invalid_argument&) {
        strict_rejected = true;
    }
    assert(strict_rejected);
}

static void test_chain_complex_round_trip() {
    Matrix d1 = interval_presentation();
    Matrix d2(1, 1);
    d2.col_degrees = {{2.0, 2.0}};
    d2.row_degrees = d1.col_degrees;
    d2.data = {{}};
    ChainComplex<Matrix> complex({d1, d2});
    assert(complex.size() == 2);
    assert(!complex[0].compatibly_sorted);
    assert(!complex[1].compatibly_sorted);
    assert(complex.is_chain_complex());

    std::stringstream encoded;
    complex.to_stream(encoded);
    ChainComplex<Matrix> decoded(encoded);
    assert(decoded.size() == 2);
    assert(decoded[0].data == d1.data);
    assert(decoded[1].col_degrees == d2.col_degrees);
    assert(decoded.is_chain_complex());

    Matrix noncomplex_d1 = interval_presentation();
    Matrix noncomplex_d2(1, 1);
    noncomplex_d2.col_degrees = noncomplex_d1.col_degrees;
    noncomplex_d2.row_degrees = noncomplex_d1.col_degrees;
    noncomplex_d2.data = {{0}};
    ChainComplex<Matrix> noncomplex({noncomplex_d1, noncomplex_d2});
    assert(!noncomplex.is_chain_complex());

    // A large ambient group must not cause a scan of every product entry.
    Matrix sparse_lower(2, 100000, {{0, 99999}, {0, 99999}},
                        vec<r2degree>(2, {0, 0}), vec<r2degree>(100000, {0, 0}));
    Matrix sparse_upper(10000, 2, array<int>(10000, {0, 1}),
                        vec<r2degree>(10000, {0, 0}), vec<r2degree>(2, {0, 0}));
    assert(ChainComplex<Matrix>({sparse_lower, sparse_upper}).is_chain_complex());
    sparse_lower.set_col(1, {0});
    assert(!ChainComplex<Matrix>({sparse_lower, sparse_upper}).is_chain_complex());

    using Compact = R2GradedSparseMatrix<int, CSCStorage<int>>;
    Compact compact_lower(2, 2, {{0, 1}, {0, 1}},
                          vec<r2degree>(2, {0, 0}), vec<r2degree>(2, {0, 0}));
    Compact compact_upper(1, 2, {{0, 1}}, {{0, 0}}, vec<r2degree>(2, {0, 0}));
    assert(ChainComplex<Compact>({compact_lower, compact_upper}).is_chain_complex());
}

static void test_named_poset_io() {
    NamedGradedMatrix<int> matrix(1, 1);
    matrix.col_degrees = {{1}};
    matrix.row_degrees = {{0}};
    matrix.data = {{0}};
    std::stringstream encoded;
    ChainComplex<NamedGradedMatrix<int>>({matrix}).to_stream(encoded);
    std::string line;
    std::getline(encoded, line);
    std::getline(encoded, line);
    assert(line == "named-poset");
    encoded.clear();
    encoded.seekg(0);
    auto decoded = ChainComplex<NamedGradedMatrix<int>>::from_stream(encoded);
    assert(decoded[0].col_degrees == matrix.col_degrees);

    NamedGradedMatrix<int> cancellable(
        2, 2, array<int>{{0, 1}, {0}},
        vec<NamedDegree>{{0}, {1}}, vec<NamedDegree>{{0}, {0}});
    cancellable.sort_compatibly();
    cancellable.semi_minimize(); // named poset has no graded-kernel implementation
    assert(cancellable.get_num_cols() == 1);
    assert(cancellable.get_num_rows() == 1);
    assert(cancellable.data == array<int>{{0}});
    assert(cancellable.col_degrees.front() == NamedDegree{1});
}

static void test_existing_scc_files() {
    const auto base = std::filesystem::path(__FILE__).parent_path() / "../test_presentations";
    ChainComplex<Matrix> presentation((base / "toy_example_2.scc").string());
    assert(presentation.size() == 1);
    assert(presentation[0].get_num_rows() != 0);

    // The old fixture has an incorrect parameter identifier and must be rejected.
    const auto path = base / "full_rips_size_1_instance_5_min_pres_resolution.scc";
    bool rejected = false;
    try { ChainComplex<Matrix> wrong(path.string()); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    // Correct only the in-memory header to exercise its actual R2 data.
    std::ifstream input(path);
    std::string text{std::istreambuf_iterator<char>(input), {}};
    text.replace(text.find('\n') + 1, 1, "2");
    std::stringstream corrected(text);
    ChainComplex<Matrix> resolution(corrected);
    assert(resolution.size() == 2);
    std::stringstream canonical;
    resolution.to_stream(canonical);
    std::string header;
    std::getline(canonical, header);
    assert(header == "scc2020");
    std::getline(canonical, header);
    assert(header == "2");
}

static void test_module_hilbert_function_and_editing() {
    TestModule module(interval_presentation());
    assert(module.num_of_gens() == 1);
    assert(module.num_of_rels() == 1);
    assert(module.dimension_at({0.0, 0.0}) == 1);
    assert(module.dimension_at({1.0, 1.0}) == 0);
    auto grid = module.hilbert_function();
    assert(grid.x_grid.size() == 2);
    assert(grid.y_grid.size() == 2);
    assert(grid.maximum == 1);

    module.compute_projective_resolution();
    assert(module.proj_res_.size() == 2);
    module.shift({1.0, 1.0});
    assert(module.proj_res_.size() == 2);
    assert((module.presentation().row_degrees.front() == r2degree{-1.0, -1.0}));
    module.add_relation({0}, {1.0, 1.0});
    assert(module.proj_res_.size() == 1);
}

static void test_homology_module() {
    Matrix injective = interval_presentation();
    Matrix no_incoming(0, 1, array<int>{}, vec<r2degree>{}, injective.col_degrees);
    auto zero_homology = homology_module(ChainComplex<Matrix>({injective, no_incoming}));
    assert(zero_homology.num_of_gens() == 0);
    assert(zero_homology.num_of_rels() == 0);

    Matrix zero_differential(1, 1, array<int>{{}},
                             vec<r2degree>{{0.0, 0.0}},
                             vec<r2degree>{{0.0, 0.0}});
    auto free_homology = homology_module(ChainComplex<Matrix>({zero_differential}));
    assert(free_homology.num_of_gens() == 1);
    assert(free_homology.num_of_rels() == 0);
    assert(free_homology.dimension_at({0.0, 0.0}) == 1);

    Matrix identity_boundary(1, 1, array<int>{{0}},
                             vec<r2degree>{{0.0, 0.0}},
                             zero_differential.col_degrees);
    auto killed_homology = homology_module(
        ChainComplex<Matrix>({zero_differential, identity_boundary}));
    assert(killed_homology.num_of_gens() == 0);
    assert(killed_homology.num_of_rels() == 0);

    Matrix fold(2, 1, array<int>{{0}, {0}},
                vec<r2degree>{{0.0, 0.0}, {0.0, 0.0}},
                vec<r2degree>{{0.0, 0.0}});
    auto one_dimensional_kernel = homology_module(ChainComplex<Matrix>({fold}));
    assert(one_dimensional_kernel.num_of_gens() == 1);
    assert(one_dimensional_kernel.num_of_rels() == 0);
    Matrix kernel_boundary(1, 2, array<int>{{0, 1}},
                           vec<r2degree>{{0.0, 0.0}}, fold.col_degrees);
    auto killed_nontrivial_kernel = homology_module(
        ChainComplex<Matrix>({fold, kernel_boundary}));
    assert(killed_nontrivial_kernel.num_of_gens() == 0);
    assert(killed_nontrivial_kernel.num_of_rels() == 0);
}

static void test_submodules_and_morphisms() {
    auto module = std::make_shared<TestModule>(interval_presentation());
    auto zero = Submodule<Matrix>::zero(module);
    auto whole = Submodule<Matrix>::whole(module);
    assert(zero.is_zero());
    assert(whole.num_of_gens() == 1);
    auto sum = zero.sum(whole);
    assert(sum.num_of_gens() == 1);
    assert(whole.presented_module().dimension_at({0.0, 0.0}) == 1);
    assert(zero.quotient_module().dimension_at({0.0, 0.0}) == 1);
    assert(whole.quotient_module().dimension_at({0.0, 0.0}) == 0);
    auto intersection = whole.intersection(sum);
    assert(intersection.num_of_gens() == 1);
    assert(intersection.presented_module().dimension_at({0.0, 0.0}) == 1);

    Matrix identity(1, 1, "Identity");
    identity.row_degrees = module->presentation().row_degrees;
    identity.col_degrees = module->presentation().row_degrees;
    ModuleMorphism<Matrix> map(module, module, identity);
    assert(map.image().num_of_gens() == 1);
    assert(map.kernel().is_zero());
    auto composite = map.compose(map);
    assert(composite.generator_lift().data == identity.data);

    auto other_parent = std::make_shared<TestModule>(interval_presentation());
    bool rejected_different_parent = false;
    try {
        (void)zero.sum(Submodule<Matrix>::zero(other_parent));
    } catch (const std::invalid_argument&) {
        rejected_different_parent = true;
    }
    assert(rejected_different_parent);

    auto resolved = std::make_shared<TestModule>(interval_presentation());
    resolved->compute_projective_resolution();
    Matrix relation_lift(1, 1, "Identity");
    relation_lift.row_degrees = resolved->presentation().col_degrees;
    relation_lift.col_degrees = resolved->presentation().col_degrees;
    ModuleMorphism<Matrix> lifted_map(resolved, resolved, {identity, relation_lift});
    assert(lifted_map.resolution_maps().size() == 2);
}

static void test_module_hom_adapter() {
    auto module = std::make_shared<TestModule>(interval_presentation());
    module->sort_compatibly();
    auto endomorphisms = module_endomorphism_basis<Matrix>(module, true);
    assert(endomorphisms.size() == 1);
    assert(endomorphisms.front().domain().get() == module.get());
    assert(endomorphisms.front().target().get() == module.get());
}

static void test_coordinate_colex_sorting() {
    using Runtime = CoordGradedSparseMatrix<double, 3, int>;
    using D = Runtime::degree_type;
    Runtime matrix(2, 1, {{0}, {0}}, {{1, 0, 2}, {0, 1, 1}}, {{0, 0, 0}});
    matrix.sort_colexicographically();
    assert(matrix.compatibly_sorted);
    assert(matrix.col_degrees[0] == D({0, 1, 1}));
    Module<Runtime> module(matrix);
    module.shift({1, 1, 1});
    assert(module.presentation().col_degrees[0] == D({-1, 0, 0}));

    Runtime cancellable(2, 2, {{0, 1}, {0}},
                        {{0, 0, 0}, {1, 1, 1}}, {{0, 0, 0}, {0, 0, 0}});
    cancellable.sort_compatibly();
    cancellable.semi_minimize();
    assert(cancellable.get_num_cols() == 1 && cancellable.get_num_rows() == 1);
    assert(cancellable.data == array<int>{{0}});
}

static void test_coordinate_io() {
    using Real = CoordGradedSparseMatrix<double, 4, int>;
    Real real_matrix(1, 1, {{0}}, {{1.0, 2.0, 3.0, 4.0}}, {{0.0, 0.0, 0.0, 0.0}});
    real_matrix.sort_colexicographically();
    assert(real_matrix.compatibly_sorted);
    std::stringstream real_scc;
    ChainComplex<Real>({real_matrix}).to_stream(real_scc);
    auto real_round_trip = ChainComplex<Real>::from_stream(real_scc);
    assert(real_round_trip[0].col_degrees == real_matrix.col_degrees);

    using Integer = CoordGradedSparseMatrix<long long, 4, int>;
    Integer discrete_matrix(1, 1, {{0}}, {{1, 2, 3, 4}}, {{0, 0, 0, 0}});
    std::stringstream discrete_scc;
    ChainComplex<Integer>({discrete_matrix}).to_stream(discrete_scc);
    auto discrete_round_trip = ChainComplex<Integer>::from_stream(discrete_scc);
    assert(discrete_round_trip[0].row_degrees == discrete_matrix.row_degrees);

    using Integer3 = CoordGradedSparseMatrix<long long, 3, int>;
    Integer3 three(1, 1, {{0}}, {{2, 3, 4}}, {{1, 1, 1}});
    three.sort_colexicographically();
    assert(three.compatibly_sorted);
    using Integer2 = CoordGradedSparseMatrix<long long, 2, int>;
    Module<Integer2> two(Integer2(1, 1, {{0}}, {{2, 3}}, {{1, 1}}));
    two.shift({1, 1});
    assert(two.presentation().col_degrees[0] == Integer2::degree_type({1, 2}));
}

int main() {
    test_poset_without_optional_operations();
    test_array_degree_traits();
    test_sort_state_and_degree_traits();
    test_checked_chain_complex_sorting();
    test_correct_minimization();
    test_chain_complex_round_trip();
    test_named_poset_io();
    test_existing_scc_files();
    test_module_hilbert_function_and_editing();
    test_homology_module();
    test_submodules_and_morphisms();
    test_module_hom_adapter();
    test_coordinate_colex_sorting();
    test_coordinate_io();
}
