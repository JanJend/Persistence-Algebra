#include <grlina/mat_upset.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;
using Point = CoordinateDegree<double, 2>;
using Upset = r2upset<double>;
using Traits = Degree_traits<Upset>;

static_assert(std::is_same_v<Upset, vec<Point>>);
static_assert(is_degree_v<Upset> && is_lattice_degree_v<Upset>);
static_assert(has_degree_io_v<Upset> && has_degree_print_v<Upset>);
static_assert(!has_degree_add_v<Upset> && !has_degree_subtract_v<Upset>);
static_assert(!has_degree_zero_v<Upset> && !has_degree_position_v<Upset>);
static_assert(!is_ordered_abelian_group_degree_v<Upset>);
static_assert(!is_degree_v<r2upset<std::string>>);

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
static_assert(Degree<Upset> && LatticeDegree<Upset>);
static_assert(!OrderedAbelianGroupDegree<Upset>);
#endif

// Direct definitions, independent of the linear sweeps under test.
static bool above(const Point& a, const Point& b) {
    return a[0] >= b[0] && a[1] >= b[1];
}

static bool naive_greater_equal(const Upset& a, const Upset& b) {
    for (const auto& point : a) {
        bool found = false;
        for (const auto& other : b) found |= above(point, other);
        if (!found) return false;
    }
    return true;
}

static Upset naive_minimal(Upset degree) {
    std::sort(degree.begin(), degree.end());
    degree.erase(std::unique(degree.begin(), degree.end()), degree.end());
    Upset result;
    for (const auto& point : degree) {
        bool redundant = false;
        for (const auto& other : degree) redundant |= point != other && above(point, other);
        if (!redundant) result.push_back(point);
    }
    return result;
}

static void check_pairs(vec<Upset> degrees) {
    for (auto& degree : degrees) {
        const auto expected = naive_minimal(degree);
        Traits::remove_redundant(degree);
        assert(degree == expected);
    }
    for (const auto& a : degrees) {
        for (const auto& b : degrees) {
            const bool ge = naive_greater_equal(a, b);
            const bool le = naive_greater_equal(b, a);
            assert(Traits::greater_equal(a, b) == ge);
            assert(Traits::smaller_equal(a, b) == le);
            assert(Traits::equals(a, b) == (le && ge));
            assert(Traits::smaller(a, b) == (le && !ge));
            assert(Traits::greater(a, b) == (ge && !le));
            if (le && !ge) {
                assert(Traits::lex_order(a, b) && !Traits::lex_order(b, a));
                assert(Traits::colex_order(a, b) && !Traits::colex_order(b, a));
            } else if (le && ge) {
                assert(!Traits::lex_order(a, b) && !Traits::lex_order(b, a));
                assert(!Traits::colex_order(a, b) && !Traits::colex_order(b, a));
            }

            auto all = a;
            all.insert(all.end(), b.begin(), b.end());
            const auto meet = Traits::meet(a, b);
            assert(meet == naive_minimal(all));
            Upset overlap;
            for (const auto& left : a) {
                for (const auto& right : b) {
                    overlap.push_back({std::max(left[0], right[0]), std::max(left[1], right[1])});
                }
            }
            const auto join = Traits::join(a, b);
            assert(join == naive_minimal(overlap));
            assert(Traits::smaller_equal(meet, a) && Traits::smaller_equal(meet, b));
            assert(Traits::greater_equal(join, a) && Traits::greater_equal(join, b));
            assert(Traits::meet(a, join) == a);
            assert(Traits::join(a, meet) == a);
        }
    }
}

static void check_laws() {
    const Upset grid{{-1, -2}, {-1, 3}, {0, -2}, {0, 3}, {1, -2}, {1, 3}};
    vec<Upset> degrees;
    for (unsigned mask = 0; mask < (1u << grid.size()); ++mask) {
        Upset degree;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (mask & (1u << i)) degree.push_back(grid[i]);
        }
        degrees.push_back(degree);
        if (!degree.empty()) {
            degree.push_back(degree.front());
            degree.push_back({2, 4});
            std::sort(degree.begin(), degree.end());
            degrees.push_back(degree);
        }
    }
    check_pairs(degrees);
    check_pairs({{}, {{0, 2}, {1, 1}, {2, 0}}, {{0, 3}, {1, 1}, {2, -1}},
                 {{0.5, 2.5}, {1, 1.5}, {1, 4}, {2, 0.5}}, {{0, 0}},
                 {{-std::numeric_limits<double>::infinity(), 1}},
                 {{1, -std::numeric_limits<double>::infinity()}},
                 {{std::numeric_limits<double>::infinity(), 1}}});

    for (auto& degree : degrees) Traits::remove_redundant(degree);
    std::sort(degrees.begin(), degrees.end(), Traits::lex_lambda());
    degrees.erase(std::unique(degrees.begin(), degrees.end()), degrees.end());
    for (const auto& a : degrees) for (const auto& b : degrees) for (const auto& c : degrees) {
        const auto join = Traits::join(a, b), meet = Traits::meet(a, b);
        if (Traits::smaller_equal(a, c) && Traits::smaller_equal(b, c))
            assert(Traits::smaller_equal(join, c));
        if (Traits::smaller_equal(c, a) && Traits::smaller_equal(c, b))
            assert(Traits::smaller_equal(c, meet));
        assert(Traits::join(join, c) == Traits::join(a, Traits::join(b, c)));
        assert(Traits::meet(meet, c) == Traits::meet(a, Traits::meet(b, c)));
        assert(Traits::join(a, Traits::meet(b, c)) == Traits::meet(Traits::join(a, b), Traits::join(a, c)));
        if (Traits::lex_order(a, b) && Traits::lex_order(b, c)) assert(Traits::lex_order(a, c));
        if (Traits::colex_order(a, b) && Traits::colex_order(b, c)) assert(Traits::colex_order(a, c));
    }
}

template <typename Scalar>
static void check_io_and_matrices() {
    using Upset = r2upset<Scalar>;
    using Traits = Degree_traits<Upset>;
    using Matrix = UpsetGradedSparseMatrix<Scalar, int>;
    const Upset degree{{0, 2}, {1, 1}, {2, 0}};
    std::stringstream stream;
    Traits::write_degree(stream, degree);
    stream << " ;";
    assert(Traits::from_stream(stream) == degree);
    std::istringstream redundant_input("0 2 0 4 1 1 1 1 2 0 ;");
    assert(Traits::from_stream(redundant_input) == degree);
    std::string separator;
    stream >> separator;
    assert(separator == ";");
    std::istringstream empty(";");
    assert(Traits::from_stream(empty).empty() && empty);
    for (const char* invalid : {"-1 ;", "2 0 0 ;", "2 1", "word ;"}) {
        std::istringstream input(invalid);
        assert(Traits::from_stream(input).empty() && input.fail());
    }

    const Upset row{{0, 2}, {2, 0}}, col{{1, 3}, {3, 1}};
    Matrix matrix(1, 1, {{0}}, {col}, {row});
    assert(matrix.is_graded_matrix());
    matrix.sort_compatibly();
    std::stringstream scc;
    ChainComplex<Matrix> complex({matrix});
    complex.to_stream(scc);
    auto parsed_complex = ChainComplex<Matrix>::from_stream(scc);
    const auto& parsed = parsed_complex.differentials().front();
    assert(parsed.is_graded_matrix() && parsed.data == matrix.data);
    assert(parsed.col_degrees == matrix.col_degrees && parsed.row_degrees == matrix.row_degrees);
    matrix.col_degrees = {row};
    matrix.row_degrees = {col};
    assert(!matrix.is_graded_matrix());

    // Equal upset degrees cancel without a graded kernel. A strict incoming
    // boundary survives, with its cycle lifted through dependent reducers.
    const Upset later{{4, 4}};
    Matrix lower(4, 3, {{0,2}, {1,2}, {0,1}, {0}}, {col, col, col, later}, {row, col, col});
    Matrix upper(1, 4, {{0,1,2}}, {later}, lower.col_degrees);
    ChainComplex<Matrix> chain({lower, upper});
    auto maps = identity_basis_maps(chain);
    chain.minimize(false, &maps);
    assert(chain.is_chain_complex() && chain[0].is_graded_matrix() && chain[1].is_graded_matrix());
    assert(chain[0].data == array<int>({{}, {0}}) && chain[1].data == array<int>({{0}}));
    assert(maps.to_original[1].get_col(0) == vec<int>({0,1,2}));
    assert((lower * maps.to_original[1]).data == (maps.to_original[0] * chain[0]).data);
    assert((upper * maps.to_original[2]).data == (maps.to_original[1] * chain[1]).data);
    for (std::size_t group = 0; group <= chain.size(); ++group) {
        assert(maps.to_original[group].is_graded_matrix() && maps.from_original[group].is_graded_matrix());
        const auto identity = maps.from_original[group] * maps.to_original[group];
        for (int c = 0; c < identity.get_num_cols(); ++c) assert(identity.get_col(c) == vec<int>{c});
    }
}

template <typename Scalar>
static void check_scalar_type() {
    using Upset = r2upset<Scalar>;
    using Traits = Degree_traits<Upset>;
    static_assert(std::is_same_v<Upset, vec<std::array<Scalar, 2>>>);
    static_assert(is_degree_v<Upset> && is_lattice_degree_v<Upset> && has_degree_io_v<Upset>);
    static_assert(!has_degree_add_v<Upset> && !has_degree_subtract_v<Upset>);
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    static_assert(Degree<Upset> && LatticeDegree<Upset> && !OrderedAbelianGroupDegree<Upset>);
#endif
    assert(std::string(Traits::poset_id) == "2");
    const Upset a{{0, 3}, {2, 1}}, b{{1, 2}, {3, 0}};
    const auto join = Traits::join(a, b), meet = Traits::meet(a, b);
    assert((join == Upset{{1, 3}, {2, 2}, {3, 1}}));
    assert((meet == Upset{{0, 3}, {1, 2}, {2, 1}, {3, 0}}));
    assert(Traits::smaller(a, join) && Traits::greater(a, meet));
    assert(Traits::lex_order(meet, a) && Traits::colex_order(meet, a));
    auto redundant = a;
    redundant.push_back({2, 4});
    Traits::remove_redundant(redundant);
    assert(redundant == a);
    check_io_and_matrices<Scalar>();
}

static void check_exact_coordinates() {
    // A double cannot distinguish these consecutive integer coordinates.
    using IntegerUpset = r2upset<long long>;
    using IntegerTraits = Degree_traits<IntegerUpset>;
    constexpr long long large = 9007199254740992LL;
    const IntegerUpset a{{large, 3}}, b{{large + 1, 2}};
    assert((IntegerTraits::join(a, b) == IntegerUpset{{large + 1, 3}}));
    assert((IntegerTraits::meet(a, b) == IntegerUpset{{large, 3}, {large + 1, 2}}));
    std::stringstream stream;
    IntegerTraits::write_degree(stream, b);
    stream << " ;";
    assert(IntegerTraits::from_stream(stream) == b);

    using LongUpset = r2upset<long double>;
    const long double x = std::nextafter(1.0L, 2.0L);
    assert((Degree_traits<LongUpset>::join({{x, 0}}, {{0, x}}) == LongUpset{{x, x}}));
}

int main() {
    check_laws();
    check_scalar_type<double>();
    check_scalar_type<float>();
    check_scalar_type<long double>();
    check_scalar_type<int>();
    check_scalar_type<long>();
    check_scalar_type<long long>();
    check_exact_coordinates();
}
