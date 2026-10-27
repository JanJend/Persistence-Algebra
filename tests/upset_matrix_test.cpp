#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/matrices/upset.hpp>
#include <grlina/storage/csc.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <type_traits>

#ifdef GRLINA_MULTICRITICAL_SOURCE_DIR
#include <scc/Scc.h>
#include <multi_critical/Grade.h>
#endif

using namespace graded_linalg;
using Matrix = UpsetGradedSparseMatrix<double, int>;
static_assert(is_graded_sparse_matrix_v<Matrix>);
static_assert(!has_matrix_graded_kernel<Matrix>::value);

template <typename Scalar, typename Storage = vec<vec<int>>>
static void round_trip() {
    using M = UpsetGradedSparseMatrix<Scalar, int, Storage>;
    const M matrix(2, 2, {{0, 1}, {1}},
                   {{{1, 1}, {2, 0}}, {{2, 1}}},
                   {{{0, 0}}, {{0, 1}, {1, 0}}});
    assert(matrix.is_graded_matrix());
    std::stringstream stream;
    matrix.to_stream(stream);
    assert(stream.str().rfind("scc2020\n2\n2 2 0\n", 0) == 0);
    const M parsed(stream);
    assert(parsed.col_degrees == matrix.col_degrees && parsed.row_degrees == matrix.row_degrees);
    assert(parsed.get_col(0) == matrix.get_col(0) && parsed.get_col(1) == matrix.get_col(1));
    assert(parsed.is_graded_matrix());

    std::stringstream sorted_input;
    matrix.to_stream(sorted_input);
    M sorted(sorted_input, true);
    assert(sorted.compatibly_sorted && sorted.is_graded_matrix());
    M identity(2, 2, "Identity");
    identity.col_degrees = identity.row_degrees = matrix.col_degrees;
    const auto product = matrix * identity;
    assert(product.col_degrees == matrix.col_degrees && product.row_degrees == matrix.row_degrees);
    assert(product.get_col(0) == matrix.get_col(0) && product.get_col(1) == matrix.get_col(1));

    M empty;
    std::stringstream zero;
    empty.to_stream(zero);
    const M parsed_zero(zero);
    assert(parsed_zero.get_num_cols() == 0 && parsed_zero.get_num_rows() == 0);
    M no_rows(1, 0, {{{1, 2}}}, {});
    std::stringstream no_rows_stream;
    no_rows.to_stream(no_rows_stream);
    const M parsed_no_rows(no_rows_stream);
    assert(parsed_no_rows.get_num_cols() == 1 && parsed_no_rows.get_num_rows() == 0);
    assert(parsed_no_rows.get_col(0).empty());
}

static void format_and_errors() {
    std::stringstream input(
        "# Multicritical input\n\nscc2020 # header\n2\n2 2 0\n"
        "2 0 0 2 2 4 ; 0 1:1 # unsorted, redundant generators\n"
        "1 1;0:2 1:3\n0 0 ;\n1 0 0 1 ;\n");
    const Matrix matrix(input);
    assert((matrix.col_degrees[0] == r2upset<double>{{0, 2}, {2, 0}}));
    assert((matrix.row_degrees[1] == r2upset<double>{{0, 1}, {1, 0}}));
    assert(matrix.get_col(0) == vec<int>({0, 1}) && matrix.get_col(1) == vec<int>({1}));
    assert(matrix.is_graded_matrix());

    const auto path = std::filesystem::temp_directory_path() /
        ("grlina-upset-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".scc");
    matrix.to_file(path.string());
    const Matrix from_file(path.string());
    std::filesystem::remove(path);
    assert(from_file.col_degrees == matrix.col_degrees && from_file.row_degrees == matrix.row_degrees);
    assert(from_file.get_col(1) == matrix.get_col(1));

    for (const char* invalid : {
        "scc2020\n3\n0 0\n",
        "scc2020\n2\n-1 0\n",
        "scc2020\n2\n1 1\n0 0 1 ; 0\n0 0 ;\n",
        "scc2020\n2\n1 1\n0 0 ; 1\n0 0 ;\n",
        "scc2020\n2\n1 1\n0 0 ; 0:\n0 0 ;\n",
        "scc2020\n2\n1 1\n0 0 ; 0\n",
        "scc2020\n2\n1 1\n0 0 0\n0 0 ;\n"}) {
        std::stringstream bad(invalid);
        bool rejected = false;
        try { Matrix parsed(bad); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }

    // Changes to shared SCC framing also preserve ordinary coordinate matrix I/O.
    using CoordinateMatrix = CoordGradedSparseMatrix<double, 2, int>;
    std::stringstream coordinate("# comment\nscc2020\n2\n1 1 0\n1 1 ; 0:1\n0 0 ;\n");
    auto complex = ChainComplex<CoordinateMatrix>::from_stream(coordinate);
    assert(complex.size() == 1 && complex[0].is_graded_matrix());
}

static void exact_scalar_io() {
    using IntegerMatrix = UpsetGradedSparseMatrix<long long, int>;
    constexpr long long large = 9007199254740993LL;
    const IntegerMatrix integer(1, 1, {{0}}, {{{large, large}}}, {{{large - 1, large - 1}}});
    std::stringstream integer_stream;
    integer.to_stream(integer_stream);
    assert(IntegerMatrix(integer_stream).col_degrees == integer.col_degrees);

    using LongMatrix = UpsetGradedSparseMatrix<long double, int>;
    const long double next = std::nextafter(1.0L, 2.0L);
    const LongMatrix matrix(1, 1, {{0}}, {{{next, next}}}, {{{1, 1}}});
    std::stringstream stream;
    ChainComplex<LongMatrix>({matrix}).to_stream(stream);
    const auto parsed = ChainComplex<LongMatrix>::from_stream(stream);
    assert(parsed[0].col_degrees == matrix.col_degrees);
}

#ifdef GRLINA_MULTICRITICAL_SOURCE_DIR
static void compare_with_multicritical(const std::string& text, const ChainComplex<Matrix>& complex) {
    std::stringstream input(text);
    scc::Scc<> parser(input);
    assert(parser.number_of_parameters() == 2);
    for (int level = 1; level <= static_cast<int>(complex.size()) + 1; ++level) {
        const bool last = level == static_cast<int>(complex.size()) + 1;
        const auto& matrix = complex.differentials()[last ? 0 : complex.size() - level];
        const auto& degrees = last ? matrix.row_degrees : matrix.col_degrees;
        assert(parser.number_of_generators(level) == static_cast<int>(degrees.size()));
        for (int column = 0; column < parser.number_of_generators(level); ++column) {
            std::vector<double> coordinates;
            std::vector<std::pair<scc::index, int>> boundary;
            parser.next_column(level, std::back_inserter(coordinates), std::back_inserter(boundary));
            std::vector<multi_critical::Grade> minimal;
            if (!coordinates.empty()) multi_critical::get_incomparable_grades_in_lex_order(coordinates, minimal);
            r2upset<double> expected;
            for (const auto& point : minimal) expected.push_back({point.x, point.y});
            assert(degrees[column] == expected);
            if (!last) {
                vec<int> entries;
                for (const auto& entry : boundary) {
                    assert(entry.second == 1);
                    entries.push_back(static_cast<int>(entry.first));
                }
                assert(matrix.get_col(column) == entries);
            }
        }
    }
}

static void sample_files() {
    for (const char* name : {"full_deg_rips_20_points.scc", "compressed_deg_rips_100_points.scc"}) {
        const auto path = std::filesystem::path(GRLINA_MULTICRITICAL_SOURCE_DIR) / "sample_files" / name;
        std::ifstream input(path);
        assert(input);
        const std::string text{std::istreambuf_iterator<char>(input), {}};
        auto complex = ChainComplex<Matrix>::from_file(path.string());
        assert(complex.size() == 3);
        for (const auto& matrix : complex.differentials()) assert(matrix.is_graded_matrix());
        compare_with_multicritical(text, complex);
        std::stringstream output;
        complex.to_stream(output);
        compare_with_multicritical(output.str(), complex);
        const auto roundtrip = ChainComplex<Matrix>::from_stream(output);
        for (std::size_t i = 0; i < complex.size(); ++i) {
            assert(roundtrip[i].col_degrees == complex[i].col_degrees);
            assert(roundtrip[i].row_degrees == complex[i].row_degrees);
            assert(roundtrip[i].data == complex[i].data);
        }
    }
}
#endif

int main() {
    round_trip<double>();
    round_trip<float>();
    round_trip<long double>();
    round_trip<int>();
    round_trip<long long>();
    round_trip<double, CSCStorage<int>>();
    format_and_errors();
    exact_scalar_io();
#ifdef GRLINA_MULTICRITICAL_SOURCE_DIR
    sample_files();
#endif
}
