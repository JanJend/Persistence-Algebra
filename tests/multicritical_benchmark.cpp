// Compile twice, defining GRLINA_MULTICRITICAL_ORIGINAL for the source checkout.
// Define MULTI_CRITICAL_TIMERS=0 to measure without instrumentation in either.
#ifndef MULTI_CRITICAL_TIMERS
#define MULTI_CRITICAL_TIMERS 1
#endif
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
#if MULTI_CRITICAL_TIMERS
#include <multi_critical/boost_timers.h>
#else
// Upstream calls these two timers even when its timing macro is disabled.
// Supply no-ops in the benchmark, leaving all original headers untouched.
namespace multi_critical {
struct Benchmark_noop_timer { void resume() {} void stop() {} };
inline Benchmark_noop_timer test_timer_1, test_timer_2;
}
#endif
#include <multi_critical/free_resolution.h>
#include <mpp_utils/Graded_matrix.h>
#include <scc/Scc.h>
#else
#include <multicritical/free_resolution.hpp>
#endif

#include <grlina/matrices/coord.hpp>
#include <grlina/matrices/upset.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
using UpsetComplex = graded_linalg::ChainComplex<graded_linalg::UpsetGradedSparseMatrix<double, int>>;
using Matrix = graded_linalg::CoordGradedSparseMatrix<double, 2, int>;
using Complex = graded_linalg::ChainComplex<Matrix>;
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
using Result = std::vector<mpp_utils::Graded_matrix<phat::vector_vector>>;
#else
using Result = Complex;
#endif

template <typename Function>
static void visit_matrices(const Result& result, Function function) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
    // Match grlina's d1, d2, ... ordering, omitting the explicit C0 -> 0.
    for (std::size_t level = result.size() - 1; level > 0; --level)
        function(result[level - 1]);
#else
    for (const auto& matrix : result.differentials()) function(matrix);
#endif
}

struct Signature {
    std::uint64_t hash = 14695981039346656037ULL;
    std::uint64_t generators = 0, nonzeros = 0;

    void append(std::uint64_t value) {
        for (int byte = 0; byte < 8; ++byte) {
            hash = (hash ^ (value & 255)) * 1099511628211ULL;
            value >>= 8;
        }
    }

    void degree(double x, double y) {
        for (double coordinate : {x, y}) {
            std::uint64_t bits;
            static_assert(sizeof(bits) == sizeof(coordinate));
            std::memcpy(&bits, &coordinate, sizeof(bits));
            append(bits);
        }
    }
};

// Compare logical matrix data without translating the original matrix storage.
static Signature signature(const Result& result) {
    Signature digest;
    bool first = true;
    visit_matrices(result, [&](const auto& matrix) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
        const auto rows = matrix.num_rows;
#else
        const auto rows = matrix.get_num_rows();
#endif
        const auto columns = matrix.get_num_cols();
        digest.append(rows);
        digest.append(columns);
        digest.generators += columns + (first ? rows : 0);
        first = false;
        for (std::int64_t row = 0; row < rows; ++row) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
            const auto& degree = matrix.row_grades[row].at;
#else
            const auto& degree = matrix.row_degrees[row];
#endif
            digest.degree(degree[0], degree[1]);
        }
        for (std::int64_t column = 0; column < columns; ++column) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
            const auto& degree = matrix.grades[column].at;
            phat::column entries;
            matrix.get_col(column, entries);
#else
            const auto& degree = matrix.col_degrees[column];
            const auto& entries = matrix.get_col(column);
#endif
            digest.degree(degree[0], degree[1]);
            digest.append(entries.size());
            for (auto row : entries) digest.append(row);
            digest.nonzeros += entries.size();
        }
    });
    return digest;
}

static void dump(const Result& result) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
    std::vector<Matrix> matrices;
    visit_matrices(result, [&](const auto& source) {
        Matrix matrix(source.get_num_cols(), source.num_rows);
        for (int row = 0; row < matrix.get_num_rows(); ++row)
            matrix.row_degrees[row] = {source.row_grades[row].at[0], source.row_grades[row].at[1]};
        for (int column = 0; column < matrix.get_num_cols(); ++column) {
            matrix.col_degrees[column] = {source.grades[column].at[0], source.grades[column].at[1]};
            phat::column entries;
            source.get_col(column, entries);
            matrix.set_col(column, std::vector<int>(entries.begin(), entries.end()));
        }
        matrices.push_back(std::move(matrix));
    });
    Complex(std::move(matrices)).to_stream(std::cout);
#else
    result.to_stream(std::cout);
#endif
}

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::runtime_error("Usage: multicritical_benchmark input.scc path|logpath repeats [--dump]");
        const std::string mode = argv[2];
        if (mode != "path" && mode != "logpath") throw std::runtime_error("Unknown mode");
        const int repeats = std::stoi(argv[3]);
        if (repeats < 1) throw std::runtime_error("Repeats must be positive");
        if (argc == 5 && std::string(argv[4]) != "--dump") throw std::runtime_error("Unknown option");
#if MULTI_CRITICAL_TIMERS
        multi_critical::initialize_timers();
#endif
        std::ifstream file(argv[1]);
        if (!file) throw std::runtime_error("Cannot open input");
        const auto parse_start = Clock::now();
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
        scc::Scc<> input(file);
        constexpr auto implementation = "original";
#else
        const auto input = UpsetComplex::from_stream(file);
        constexpr auto implementation = "native";
#endif
        const double parse_seconds = std::chrono::duration<double>(Clock::now() - parse_start).count();
        auto convert = [&]() -> Result {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
            Result result;
            multi_critical::free_resolution(input, result, mode == "logpath");
            return result;
#else
            return multi_critical::free_resolution(input, mode == "logpath");
#endif
        };
        Signature expected;
        double load_and_convert_seconds;
        {
            // One untimed warmup; release its output before measuring.
            const auto warmup = convert();
            load_and_convert_seconds = std::chrono::duration<double>(Clock::now() - parse_start).count();
            if (argc == 5) { dump(warmup); return 0; }
            expected = signature(warmup);
        }
        std::cout << "implementation,timers,input,mode,repeat,conversion_seconds,parse_seconds,load_and_convert_seconds,prepare_input_seconds,build_matrices_seconds,generators,nonzeros,hash\n";
        std::cout << std::setprecision(12);
        for (int repeat = 0; repeat < repeats; ++repeat) {
#ifdef GRLINA_MULTICRITICAL_ORIGINAL
            input.reset_all();
#endif
            const auto start = Clock::now();
            const auto result = convert();
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            const auto observed = signature(result);
            if (observed.hash != expected.hash || observed.generators != expected.generators || observed.nonzeros != expected.nonzeros)
                throw std::runtime_error("Output changed between repetitions");
            double prepare_seconds = 0, build_seconds = 0;
#if MULTI_CRITICAL_TIMERS
            prepare_seconds = multi_critical::prepare_input_complex_timer.elapsed().wall / 1e9;
            build_seconds = multi_critical::convert_to_graded_matrices_timer.elapsed().wall / 1e9;
#endif
            std::cout << implementation << ',' << MULTI_CRITICAL_TIMERS << ',' << argv[1] << ',' << mode << ',' << repeat << ','
                      << seconds << ',' << parse_seconds << ',' << load_and_convert_seconds << ',' << prepare_seconds << ',' << build_seconds << ','
                      << observed.generators << ',' << observed.nonzeros << ',' << observed.hash << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
