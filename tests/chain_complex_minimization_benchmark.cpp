// Usage: chain_complex_minimization_benchmark [MAX_PAIRS=4000] [REPEATS=5]
//                                           [path|independent|all=all]
//        chain_complex_minimization_benchmark file ORDINARY_SCC [REPEATS=5]
// Setup, sorting the shared input, conversion and checks are outside timers.
// Both reducers use one thread and do not track maps. Report medians after a
// warm-up, rotating execution order to reduce cache and clock bias.
#include <grlina/mat_coord.hpp>
#include <grlina/proj_chain_complex.hpp>
#include <multi_chunk/multi_chunk.h>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace graded_linalg;
using Matrix = CoordGradedSparseMatrix<double, 2, int>;
using Degree = Matrix::degree_type;
using Complex = ChainComplex<Matrix>;
using NativeMatrix = mpp_utils::Graded_matrix<>;
using NativeComplex = std::vector<NativeMatrix>;
using Clock = std::chrono::steady_clock;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Complex load_file(const std::string& path) {
    std::ifstream input(path);
    require(bool(input), "Cannot open SCC file");
    std::ostringstream normalized;
    std::string line;
    while (std::getline(input, line)) {
        line = line.substr(0, line.find('#'));
        const auto separator = line.find(';');
        if (separator == std::string::npos) { normalized << line << '\n'; continue; }
        // SCC samples contain unordered entries. Normalize coefficients and
        // duplicate indices over F2 before using grlina's sorted-column reader.
        std::istringstream boundary(line.substr(separator + 1));
        std::map<int, bool> coefficients;
        int row;
        while (boundary >> row) {
            long long coefficient = 1;
            if (boundary.peek() == ':') {
                boundary.get();
                require(bool(boundary >> coefficient), "Invalid SCC coefficient");
            }
            coefficients[row] = coefficients[row] != (coefficient % 2 != 0);
        }
        require(boundary.eof(), "Invalid SCC boundary entry");
        normalized << line.substr(0, separator + 1);
        // Keep explicit zeros until the reader validates their row indices.
        for (const auto& [index, nonzero] : coefficients) normalized << ' ' << index << ':' << int(nonzero);
        normalized << '\n';
    }
    require(input.eof(), "Failed to read SCC file");
    std::istringstream stream(normalized.str());
    return Complex::from_stream(stream, true);
}

Complex fixture(int n, bool path) {
    const int extra = n / 4;
    array<int> columns(n + extra);
    vec<Degree> source(n + extra, Degree{1,1}), target(n + 1, Degree{1,1});
    target[0] = {0,0};
    for (int c = 0; c < n; ++c) columns[c] = path ? vec<int>{c, c + 1} : vec<int>{c + 1};
    for (int c = n; c < n + extra; ++c) {
        columns[c] = {0, n / 2, n};
        source[c] = {2,2};
    }
    Matrix lower(n + extra, n + 1, columns, source, target);
    columns.assign(extra - 1, {});
    for (int c = 0; c < extra - 1; ++c) columns[c] = {n + c, n + c + 1};
    Matrix upper(extra - 1, n + extra, columns, vec<Degree>(extra - 1, Degree{3,3}), source);
    Complex result({lower, upper});
    result.sort_compatibly();
    return result;
}

NativeComplex to_native(const Complex& input) {
    NativeComplex result(input.size() + 1);
    for (std::size_t level = 0; level < input.size(); ++level) {
        const auto& d = input[input.size() - 1 - level];
        auto& native = result[level];
        native.number_of_parameters = 2;
        native.num_rows = d.get_num_rows();
        native.set_dimensions(d.get_num_rows(), d.get_num_cols());
        for (int c = 0; c < d.get_num_cols(); ++c) {
            const auto column = d.get_col(c);
            native.set_col(c, phat::column(column.begin(), column.end()));
            native.grades.emplace_back(d.col_degrees[c][0], d.col_degrees[c][1]);
        }
        for (const auto& degree : d.row_degrees) native.row_grades.emplace_back(degree[0], degree[1]);
    }
    // Multi-chunk skips its final matrix. Supply C0 -> 0 so every actual
    // differential, including d1, is graded and eligible for cancellation.
    auto& terminal = result.back();
    terminal.number_of_parameters = 2;
    terminal.num_rows = 0;
    terminal.set_dimensions(0, input[0].get_num_rows());
    for (const auto& degree : input[0].row_degrees) terminal.grades.emplace_back(degree[0], degree[1]);
    return result;
}

Complex from_native(const NativeComplex& input) {
    std::vector<Matrix> result;
    for (std::size_t level = input.size() - 1; level > 0; --level) {
        const auto& native = input[level - 1];
        array<int> columns(native.get_num_cols());
        vec<Degree> source, target;
        for (int c = 0; c < native.get_num_cols(); ++c) {
            phat::column column;
            native.get_col(c, column);
            columns[c].assign(column.begin(), column.end());
            source.push_back({native.grades[c].at[0], native.grades[c].at[1]});
        }
        for (const auto& degree : input[level].grades) target.push_back({degree.at[0], degree.at[1]});
        require(target.size() == static_cast<std::size_t>(native.num_rows), "Multi-chunk returned inconsistent dimensions");
        result.emplace_back(native.get_num_cols(), native.num_rows, columns, source, target);
    }
    return Complex(std::move(result));
}

void check_complex(const Complex& complex, bool minimal) {
    require(!complex.empty(), "Expected at least one differential");
    for (std::size_t level = 0; level < complex.size(); ++level) {
        const auto& d = complex[level];
        require(d.col_degrees.size() == static_cast<std::size_t>(d.get_num_cols()) &&
                d.row_degrees.size() == static_cast<std::size_t>(d.get_num_rows()), "Inconsistent degree counts");
        if (level) require(d.row_degrees == complex[level - 1].col_degrees, "Adjacent chain-group bases disagree");
        for (int c = 0; c < d.get_num_cols(); ++c) {
            const auto column = d.get_col(c);
            require(std::is_sorted(column.begin(), column.end()) &&
                    std::adjacent_find(column.begin(), column.end()) == column.end(), "Unsorted or repeated matrix entries");
            for (int r : column) {
                require(r >= 0 && r < d.get_num_rows(), "Matrix row is outside its chain group");
                require(Degree_traits<Degree>::smaller_equal(d.row_degrees[r], d.col_degrees[c]), "Differential violates grading");
                require(!minimal || !Degree_traits<Degree>::equals(d.row_degrees[r], d.col_degrees[c]), "Equal-degree pair remains");
            }
        }
    }
    require(complex.is_chain_complex(), "Chain equation failed");
}

auto graded_counts(const Complex& complex) {
    std::vector<std::map<Degree, std::size_t>> result(complex.size() + 1);
    for (const auto& degree : complex[0].row_degrees) ++result[0][degree];
    for (std::size_t level = 0; level < complex.size(); ++level)
        for (const auto& degree : complex[level].col_degrees) ++result[level + 1][degree];
    return result;
}

bool same_matrices(const Complex& a, const Complex& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t level = 0; level < a.size(); ++level)
        if (a[level].data != b[level].data || a[level].col_degrees != b[level].col_degrees ||
            a[level].row_degrees != b[level].row_degrees) return false;
    return true;
}

std::size_t generators(const Complex& complex) {
    std::size_t count = complex[0].get_num_rows();
    for (const auto& d : complex.differentials()) count += d.get_num_cols();
    return count;
}

std::size_t entries(const Complex& complex) {
    std::size_t count = 0;
    for (const auto& d : complex.differentials())
        for (int c = 0; c < d.get_num_cols(); ++c) count += d.data[c].size();
    return count;
}

double median(vec<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

void run(const Complex& original, const std::string& label, int repeats, int expected_pairs = 0) {
    check_complex(original, false);
    const auto native_original = to_native(original);
    std::array<vec<double>, 3> times;
    bool exact_match = true;
    Complex reduced;
    for (int repetition = -1; repetition < repeats; ++repetition) {
        std::array<Complex, 3> results;
        for (int offset = 0; offset < 3; ++offset) {
            const int method = (offset + repetition + 1) % 3;
            double ms;
            if (method == 2) {
                auto input = native_original;
                const auto start = Clock::now();
                multi_chunk::compress(input);
                ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                results[method] = from_native(input);
            } else {
                auto input = original;
                const auto start = Clock::now();
                input.minimize(method == 1);
                ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                results[method] = std::move(input);
            }
            if (repetition >= 0) times[method].push_back(ms);
            check_complex(results[method], true);
            if (expected_pairs) {
                const int survivors = expected_pairs / 4;
                require(results[method][0].get_num_rows() == 1 && results[method][0].get_num_cols() == survivors &&
                        results[method][1].get_num_rows() == survivors && results[method][1].get_num_cols() == survivors - 1,
                        "Synthetic reduction has the wrong dimensions");
                for (int c = 0; c < survivors; ++c)
                    require(results[method][0].get_col(c) == vec<int>{0}, "Synthetic lower differential differs from reference");
                for (int c = 0; c < survivors - 1; ++c)
                    require(results[method][1].get_col(c) == vec<int>({c, c + 1}), "Synthetic upper differential differs from reference");
            }
        }
        require(same_matrices(results[0], results[1]), "Sorting changed the already sorted reduction");
        require(graded_counts(results[0]) == graded_counts(results[2]), "Reducers disagree on generators per degree and chain group");
        exact_match &= same_matrices(results[0], results[2]);
        if (expected_pairs) require(exact_match, "Synthetic outputs differ");
        reduced = std::move(results[0]);
    }
    const double grlina = median(times[0]), with_sort = median(times[1]), multi = median(times[2]);
    std::cout << label << ',' << generators(original) << ',' << generators(reduced) << ','
              << (generators(original) - generators(reduced)) / 2 << ',' << entries(original) << ','
              << entries(reduced) << ',' << repeats << ',' << grlina << ',' << with_sort << ',' << multi << ','
              << multi / grlina << ',' << exact_match << ",1,1\n" << std::flush;
}

int main(int argc, char** argv) {
    try {
        require(argc <= 4, "Usage: chain_complex_minimization_benchmark [MAX_PAIRS] [REPEATS] [path|independent|all] | file PATH [REPEATS]");
        const bool file = argc > 1 && std::string(argv[1]) == "file";
        require(!file || argc >= 3, "File mode requires an ordinary two-parameter SCC path");
        const int repeats = file ? (argc > 3 ? std::stoi(argv[3]) : 5) : (argc > 2 ? std::stoi(argv[2]) : 5);
        require(repeats >= 1 && repeats <= 100, "REPEATS must be 1..100");
        multi_chunk::verbose = mpp_utils::verbose = scc::verbose = false;
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        std::cout << std::setprecision(9)
                  << "case,input_generators,output_generators,cancelled_pairs,input_entries,output_entries,repeats,"
                     "grlina_presorted_ms,grlina_with_sort_ms,multi_chunk_ms,multi_over_grlina,exact_matrix_match,"
                     "graded_counts_match,multi_chunk_threads\n";
        if (file) {
            auto original = load_file(argv[2]);
            run(original, "file", repeats);
        } else {
            const int maximum = argc > 1 ? std::stoi(argv[1]) : 4000;
            const std::string family = argc > 3 ? argv[3] : "all";
            require(maximum >= 4 && maximum <= 64000, "MAX_PAIRS must be 4..64000");
            require(family == "path" || family == "independent" || family == "all", "Unknown synthetic family");
            for (const std::string label : {"independent", "path"}) {
                if (family != "all" && family != label) continue;
                for (int n = std::min(500, maximum);; n = std::min(2 * n, maximum)) {
                    run(fixture(n, label == "path"), label, repeats, n);
                    if (n == maximum) break;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
