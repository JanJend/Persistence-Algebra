// Dense square systems with n RHS columns: repeated O(n^3) reductions give
// O(n^4); sharing the graded-kernel reduction gives O(n^3) in these 2D
// families. Timings describe these cases, not a general bound in dimension d.
// Usage: graded_linear_system_benchmark [MAX_N=256] [REPEATS=3]
//                                      [uniform|mixed|late_basis|nonunique|all=all]
//                                      [OLD_MAX_N=MAX_N]
#include "graded_linear_system_baselines.hpp"
#include <grlina/mat_grid.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace graded_linalg;
using Matrix = GridGradedSparseMatrix<double, 2, int>;
using Degree = matrix_geometry_degree_t<Matrix>;
using Clock = std::chrono::steady_clock;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

array<int> dense_invertible(int n) {
    std::mt19937 random(20261009u + static_cast<unsigned>(n));
    for (int attempt = 0; attempt < 100; ++attempt) {
        array<int> columns(n);
        for (auto& column : columns)
            for (int row = 0; row < n; ++row)
                if (random() & 1u) column.push_back(row);
        SparseMatrix<int> rank_test(n, n, columns);
        rank_test.column_reduction_triangular();
        if (rank_test.pivots.size() == static_cast<std::size_t>(n)) return columns;
    }
    throw std::runtime_error("Could not generate a full-rank dense matrix");
}

double median(vec<double> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size()/2;
    return values.size()%2 ? values[middle] : (values[middle-1]+values[middle])/2;
}

enum Method { column_method, kernel_method, direct_method };

struct Timings {
    std::array<double, 3> milliseconds;
    double kernel_only;
    int column_exact_match; // -1 when the column-by-column baseline was skipped
    bool kernel_exact_match;
};

Timings run(int n, int repeats, const std::string& family, int old_maximum) {
    auto columns = dense_invertible(n);
    const auto rhs_columns = columns;
    const bool late_basis = family == "late_basis";
    const bool nonunique = family == "nonunique";
    vec<Degree> a_degrees(n, Degree{0,0}), b_degrees;
    if (late_basis || nonunique) {
        // Late duplicates precede the early source basis. The late block is
        // inadmissible in late_basis, and admissible as well in nonunique.
        columns.insert(columns.end(), rhs_columns.begin(), rhs_columns.end());
        a_degrees.insert(a_degrees.begin(), n, Degree{2,2});
    }
    for (int j = 0; j < n; ++j) {
        const double t = family == "uniform" ? 0.0 : double(j+1)/double(n+1);
        const double birth = nonunique ? 3.0 : 1.0;
        b_degrees.push_back({birth+t, birth-t});
    }
    const Matrix A(static_cast<int>(columns.size()), n, columns, a_degrees, vec<Degree>(n, Degree{0,0}));
    const Matrix B(n, n, rhs_columns, b_degrees, vec<Degree>(n, Degree{0,0}));
    const auto original_a = A.data, original_b = B.data;
    std::array<vec<double>, 3> times;
    vec<double> kernel_only_times;
    bool column_exact_match = true, kernel_exact_match = true;
    for (int repetition = 0; repetition < repeats; ++repetition) {
        std::array<std::optional<Matrix>, 3> solutions;
        vec<Method> methods{kernel_method, direct_method};
        if (n <= old_maximum) methods.push_back(column_method);
        // Rotate order to reduce bias from warm caches and clock changes.
        for (std::size_t offset = 0; offset < methods.size(); ++offset) {
            const Method method = methods[(offset+repetition)%methods.size()];
            double kernel_only = 0;
            const auto start = Clock::now();
            solutions[method] = method == column_method ? test::column_by_column_solver(A, B) :
                                method == kernel_method ? test::full_kernel_solver(A, B, &kernel_only) :
                                solve_graded_linear_system(A, B);
            times[method].push_back(std::chrono::duration<double, std::milli>(Clock::now()-start).count());
            if (method == kernel_method) kernel_only_times.push_back(kernel_only);
            // Setup, rank testing, and every correctness check are outside the timer.
            const auto& solution = solutions[method];
            require(solution.has_value(), "Solvable system was rejected");

            require(solution->get_num_cols() == n && solution->get_num_rows() == A.get_num_cols(), "Wrong solution shape");
            require(geometric_col_degrees(*solution) == b_degrees &&
                    geometric_row_degrees(*solution) == a_degrees, "Basis degrees changed");
            // Unique cases must agree exactly. In the nonunique case, the
            // original solver chooses late coordinates; the others choose early.
            const int expected_offset = late_basis || (nonunique && method != column_method) ? n : 0;
            for (int j = 0; j < n; ++j)
                require(solution->get_col(j) == vec<int>{expected_offset+j}, "Wrong reference solution");
            const auto product = A * *solution;
            for (int j = 0; j < n; ++j) require(product.get_col(j) == B.get_col(j), "A X differs from B");
            require(A.data == original_a && B.data == original_b, "Solver changed an input");
        }
        const bool same_kernel = solutions[kernel_method]->data == solutions[direct_method]->data;
        require(same_kernel, "Full-kernel and direct solutions differ on the benchmark family");
        kernel_exact_match &= same_kernel;
        if (n <= old_maximum) {
            const bool same_column = solutions[column_method]->data == solutions[direct_method]->data;
            require(same_column == !nonunique, "Unexpected coefficient comparison");
            column_exact_match &= same_column;
        }
    }
    Timings result{{}, median(kernel_only_times), n <= old_maximum ? int(column_exact_match) : -1, kernel_exact_match};
    for (Method method : {column_method, kernel_method, direct_method})
        result.milliseconds[method] = times[method].empty() ? std::numeric_limits<double>::quiet_NaN() : median(times[method]);
    return result;
}

int main(int argc, char** argv) {
    try {
        require(argc <= 5, "Usage: graded_linear_system_benchmark [MAX_N] [REPEATS] [uniform|mixed|late_basis|nonunique|all] [OLD_MAX_N]");
        const int maximum = argc > 1 ? std::stoi(argv[1]) : 256;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
        const std::string selection = argc > 3 ? argv[3] : "all";
        const int old_maximum = argc > 4 ? std::stoi(argv[4]) : maximum;
        require(maximum >= 32 && maximum <= 4096, "MAX_N must be 32..4096");
        require(repeats >= 1 && repeats <= 100, "REPEATS must be 1..100");
        require(old_maximum >= 0 && old_maximum <= maximum, "OLD_MAX_N must be 0..MAX_N");
        require(selection == "uniform" || selection == "mixed" || selection == "late_basis" ||
                selection == "nonunique" || selection == "all", "Unknown family");
        std::cout << std::setprecision(9)
                  << "case,n,a_cols,b_cols,repeats,column_ms,kernel_ms,direct_ms,kernel_only_ms,"
                     "column_speedup,kernel_speedup,column_growth,kernel_growth,direct_growth,"
                     "column_exponent,kernel_exponent,direct_exponent,column_ms_per_n4,"
                     "kernel_ms_per_n3,direct_ms_per_n3,column_exact_match,kernel_exact_match,ax_matches\n";
        for (const std::string family : {"uniform", "mixed", "late_basis", "nonunique"}) {
            if (selection != "all" && selection != family) continue;
            std::array<double, 3> previous{};
            for (int n = 32; n <= maximum; n *= 2) {
                const auto timings = run(n, repeats, family, old_maximum);
                const auto& t = timings.milliseconds;
                std::array<double, 3> growth;
                for (Method method : {column_method, kernel_method, direct_method})
                    growth[method] = previous[method] ? t[method]/previous[method] : std::numeric_limits<double>::quiet_NaN();
                std::cout << family << ',' << n << ',' << (family == "late_basis" || family == "nonunique" ? 2*n : n)
                          << ',' << n << ',' << repeats << ',' << t[column_method] << ',' << t[kernel_method] << ','
                          << t[direct_method] << ',' << timings.kernel_only << ',' << t[column_method]/t[direct_method]
                          << ',' << t[kernel_method]/t[direct_method];
                for (Method method : {column_method, kernel_method, direct_method}) std::cout << ',' << growth[method];
                for (Method method : {column_method, kernel_method, direct_method}) std::cout << ',' << std::log2(growth[method]);
                std::cout << ',' << t[column_method]/std::pow(double(n),4) << ',' << t[kernel_method]/std::pow(double(n),3)
                          << ',' << t[direct_method]/std::pow(double(n),3) << ',' << timings.column_exact_match << ','
                          << timings.kernel_exact_match << ",1" << std::endl;
                previous = t;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
