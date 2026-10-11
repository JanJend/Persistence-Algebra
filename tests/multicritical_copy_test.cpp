// Define GRLINA_MULTICRITICAL_ORIGINAL to compare against the source checkout.
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
#define MULTI_CRITICAL_TIMERS 1
#include <multi_critical/boost_timers.h>
#include <multi_critical/free_resolution.h>
#include <mpp_utils/Graded_matrix.h>
#include <scc/Scc.h>
#else
#include <multicritical/free_resolution.hpp>
#endif

#include <grlina/mat_coord.hpp>
#include <grlina/mat_csc.hpp>
#include <grlina/mat_upset.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

using UpsetMatrix = graded_linalg::UpsetGradedSparseMatrix<double, int>;
using UpsetComplex = graded_linalg::ChainComplex<UpsetMatrix>;
using Matrix = graded_linalg::CoordGradedSparseMatrix<double, 2, int>;
using Complex = graded_linalg::ChainComplex<Matrix>;

#ifndef GRLINA_MULTICRITICAL_ORIGINAL
static_assert(std::is_same_v<multi_critical::Degree, Matrix::degree_type>);
static_assert(std::is_same_v<decltype(multi_critical::Input_simplex{}.grades), UpsetMatrix::degree_type>);
static_assert(std::is_same_v<decltype(multi_critical::free_resolution(std::declval<const UpsetComplex&>())), Complex>);
#endif

static void require(bool condition) {
    if (!condition) throw std::runtime_error("Multicritical conversion check failed");
}

static void check_complex(const Complex& complex) {
    for (const auto& matrix : complex.differentials()) {
        require(matrix.is_graded_matrix());
        require(std::is_sorted(matrix.col_degrees.begin(), matrix.col_degrees.end()));
        require(std::is_sorted(matrix.row_degrees.begin(), matrix.row_degrees.end()));
        for (int column = 0; column < matrix.get_num_cols(); ++column) {
            const auto& entries = matrix.get_col(column);
            require(std::is_sorted(entries.begin(), entries.end()));
            require(std::adjacent_find(entries.begin(), entries.end()) == entries.end());
        }
    }
    for (std::size_t i = 1; i < complex.size(); ++i) {
        require(complex[i - 1].get_num_cols() == complex[i].get_num_rows());
        require(complex[i - 1].col_degrees == complex[i].row_degrees);
        for (int column = 0; column < complex[i].get_num_cols(); ++column) {
            std::vector<int> composed;
            for (int row : complex[i].get_col(column))
                graded_linalg::Column_traits<std::vector<int>, int>::add_to(complex[i - 1].get_col(row), composed);
            require(composed.empty());
        }
    }
}

static void check_roundtrip(const Complex& complex) {
    if (complex.empty()) return;
    std::stringstream output;
    complex.to_stream(output);
    const auto reread = Complex::from_stream(output);
    require(reread.size() == complex.size());
    for (std::size_t i = 0; i < complex.size(); ++i) {
        require(reread[i].get_num_rows() == complex[i].get_num_rows());
        require(reread[i].data == complex[i].data);
        require(reread[i].col_degrees == complex[i].col_degrees);
        require(reread[i].row_degrees == complex[i].row_degrees);
    }
    check_complex(reread);
}

static Complex convert(std::istream& input, bool logpath) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
    // Only the comparison executable uses the original parser and matrices.
    scc::Scc<> parser(input);
    std::vector<mpp_utils::Graded_matrix<phat::vector_vector>> original;
    multi_critical::free_resolution(parser, original, logpath);
    std::vector<Matrix> matrices;
    for (std::size_t level = original.size() - 1; level > 0; --level) {
        const auto& source = original[level - 1];
        Matrix matrix(static_cast<int>(source.get_num_cols()), static_cast<int>(source.num_rows));
        for (int column = 0; column < matrix.get_num_cols(); ++column) {
            matrix.col_degrees[column] = {source.grades[column].at[0], source.grades[column].at[1]};
            phat::column entries;
            source.get_col(column, entries);
            matrix.set_col(column, std::vector<int>(entries.begin(), entries.end()));
        }
        for (int row = 0; row < matrix.get_num_rows(); ++row)
            matrix.row_degrees[row] = {source.row_grades[row].at[0], source.row_grades[row].at[1]};
        matrix.compatibly_sorted = true;
        matrices.push_back(std::move(matrix));
    }
    Complex result(std::move(matrices));
#else
    const auto source = UpsetComplex::from_stream(input);
    auto result = multi_critical::free_resolution(source, logpath);
#endif
    check_complex(result);
    return result;
}

int main(int argc, char** argv) {
#if MULTI_CRITICAL_TIMERS
    multi_critical::initialize_timers();
#endif
    const bool dump = argc > 1 && std::string(argv[1]) == "--dump";
    for (bool logpath : {false, true}) {
        // C0 has one upset with three incomparable generators; C1 is empty.
        std::istringstream input("scc2020\n2\n0 1\n0 2 1 1 2 0 ;\n");
        auto result = convert(input, logpath);
        require(result.size() == 3);
        require(result[0].get_num_rows() == 3);
        require(result[0].get_num_cols() == (logpath ? 3 : 2));
        require(result[1].get_num_cols() == (logpath ? 1 : 0));
        require(result[2].get_num_cols() == 0);
        require(result[0].row_degrees == std::vector<Matrix::degree_type>({{0, 2}, {1, 1}, {2, 0}}));
        if (logpath) {
            require(result[1].col_degrees[0] == Matrix::degree_type({2, 2}));
            require(result[1].get_col(0) == std::vector<int>({0, 1, 2}));
        }
        check_roundtrip(result);
        if (dump) result.to_stream(std::cout);

        // An ordinary filtered interval needs no additional generators.
        std::istringstream interval("scc2020\n2\n1 2\n1 1 ; 0 1\n0 0 ;\n0 0 ;\n");
        auto free_interval = convert(interval, logpath);
        require(free_interval[0].data == graded_linalg::array<int>({{0, 1}}));
        require(free_interval[0].col_degrees == std::vector<Matrix::degree_type>({{1, 1}}));
        require(free_interval[0].get_num_rows() == 2);
        check_roundtrip(free_interval);
        if (dump) free_interval.to_stream(std::cout);

        for (int i = dump ? 2 : 1; i < argc; ++i) {
            std::ifstream file(argv[i]);
            require(bool(file));
            auto converted = convert(file, logpath);
            check_roundtrip(converted);
            if (dump) {
                std::cout << (logpath ? "logpath " : "path ") << argv[i] << '\n';
                converted.to_stream(std::cout);
            }
        }
    }
#ifndef GRLINA_MULTICRITICAL_ORIGINAL
    require(multi_critical::free_resolution(UpsetComplex{}).empty());
    // Our parser handles coefficients in F2 before the conversion sees them.
    std::istringstream coefficients("scc2020\n2\n1 2\n1 1 ; 0:2 1:3\n0 0 ;\n0 0 ;\n");
    require(convert(coefficients, true)[0].get_col(0) == std::vector<int>({1}));

    // Reading a CSC column view must work without materializing a scratch column.
    using CSCUpsetMatrix = graded_linalg::UpsetGradedSparseMatrix<double, int, graded_linalg::CSCStorage<int>>;
    using CSCUpsetComplex = graded_linalg::ChainComplex<CSCUpsetMatrix>;
    CSCUpsetMatrix csc_matrix(1, 2);
    csc_matrix.col_degrees[0] = {{1, 1}};
    csc_matrix.row_degrees = {{{0, 0}}, {{0, 0}}};
    csc_matrix.assign_data(graded_linalg::array<int>{{0, 1}});
    const CSCUpsetComplex csc({std::move(csc_matrix)});
    const auto csc_result = multi_critical::free_resolution(csc);
    require(csc_result[0].get_col(0) == std::vector<int>({0, 1}));
    require(csc[0].get_col(0) == std::vector<int>({0, 1}));
    check_complex(csc_result);
    check_roundtrip(csc_result);

    // Keep the caller's input and index type, including 64-bit indices.
    using WideUpsetMatrix = graded_linalg::UpsetGradedSparseMatrix<double, long long>;
    using WideUpsetComplex = graded_linalg::ChainComplex<WideUpsetMatrix>;
    std::istringstream wide_input("scc2020\n2\n0 1\n0 2 1 1 2 0 ;\n");
    const auto wide = WideUpsetComplex::from_stream(wide_input);
    const auto wide_result = multi_critical::free_resolution(wide);
    static_assert(std::is_same_v<typename decltype(wide_result)::index_type, long long>);
    require(wide_result[0].get_num_rows() == 3);
    require(wide_result[0].get_num_cols() == 2); // The default is path.
    require(wide_result[1].get_num_cols() == 0);
    require(wide[0].get_num_cols() == 0 && wide[0].get_num_rows() == 1);
    require(wide[0].row_degrees == std::vector<WideUpsetMatrix::degree_type>({{{0, 2}, {1, 1}, {2, 0}}}));

    // Expansion must reject dimensions that do not fit the output index type.
    using TinyUpsetMatrix = graded_linalg::UpsetGradedSparseMatrix<double, signed char>;
    TinyUpsetMatrix tiny(0, 1);
    for (int i = 0; i < 128; ++i) tiny.row_degrees[0].push_back({double(i), double(128 - i)});
    bool overflow = false;
    try {
        multi_critical::free_resolution(graded_linalg::ChainComplex<TinyUpsetMatrix>({tiny}), false);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    require(overflow);
#endif
    if (!dump) std::cout << "Native multicritical conversion checks passed\n";
}
