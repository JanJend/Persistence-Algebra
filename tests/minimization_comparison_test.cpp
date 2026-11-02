#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/graded_linalg.hpp>
#include "test_degree_matrices.hpp"
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

using namespace graded_linalg;

template<class Matrix>
void same_image(const Matrix& normal, const Matrix& alternative) {
    assert(normal.row_degrees == alternative.row_degrees);
    assert(normal.get_num_cols() == alternative.get_num_cols());
    assert(graded_linear_system_is_solvable(normal, alternative));
    assert(graded_linear_system_is_solvable(alternative, normal));
    for (int column = 0; column < alternative.get_num_cols(); ++column)
        for (int row : alternative.column(column))
            assert(alternative.col_degrees[column] != alternative.row_degrees[row]);
}

template<class Matrix>
void compare(const Matrix& input) {
    auto normal = input, alternative = input;
    normal.minimize();
    alternative.minimize_by_image_generators();
    same_image(normal, alternative);
}

template<class Matrix>
void correctness() {
    using Degree = typename Matrix::degree_type;
    constexpr auto dimension = std::tuple_size_v<Degree>;
    Degree zero{}, one{}, two{};
    one.fill(1); two.fill(2);
    // Unit cancellation, duplicate relations and a zero relation.
    compare(Matrix(4, 3, {{0,1}, {0,1,2}, {0,1,2}, {}},
        {one, two, two, zero}, {zero, one, zero}));
    if constexpr (dimension >= 2) {
        Degree x{}, y{}, join{};
        x[0] = 1; y[1] = 1; join[0] = join[1] = 1;
        // The last column is redundant only after combining incomparable births.
        Matrix example(3, 2, {{0,1}, {1}, {0}}, {x, y, join}, {zero, zero});
        compare(example);
        example.minimize_by_image_generators();
        assert(example.get_num_cols() == 2);
    }
    std::mt19937 random(731 + dimension);
    for (int trial = 0; trial < 24; ++trial) {
        vec<Degree> columns(7), rows(4);
        for (auto* degrees : {&columns, &rows})
            for (auto& degree : *degrees)
                for (auto& value : degree) value = random() % 3;
        array<int> entries(7);
        for (int j = 0; j < 7; ++j)
            for (int i = 0; i < 4; ++i)
                if ((random() & 1) && Degree_traits<Degree>::smaller_equal(rows[i], columns[j]))
                    entries[j].push_back(i);
        compare(Matrix(7, 4, entries, columns, rows));
    }
}

// Instrument kernel construction outside timing. Both paths have the same
// presentation input; only the standard minimiser should request a kernel.
template<std::size_t Dimension>
struct CountedMatrix : GradedSparseMatrix<CoordinateDegree<int, Dimension>, int, CountedMatrix<Dimension>> {
    using Base = GradedSparseMatrix<CoordinateDegree<int, Dimension>, int, CountedMatrix>;
    using Base::Base;
    inline static int kernel_calls = 0;
    CountedMatrix graded_kernel() {
        ++kernel_calls;
        CoordGradedSparseMatrix<int, Dimension, int> copy(this->get_num_cols(), this->get_num_rows(),
            this->data, this->col_degrees, this->row_degrees);
        auto result = copy.graded_kernel();
        return CountedMatrix(result.get_num_cols(), result.get_num_rows(), result.data,
            result.col_degrees, result.row_degrees);
    }
};

template<std::size_t Dimension>
void no_kernel_vectors() {
    using Degree = CoordinateDegree<int, Dimension>;
    Degree zero{}, one{}, two{};
    one.fill(1); two.fill(2);
    CountedMatrix<Dimension> input(2, 1, {{0}, {0}}, {one, two}, {zero});
    auto normal = input, alternative = input;
    CountedMatrix<Dimension>::kernel_calls = 0;
    normal.minimize();
    assert(CountedMatrix<Dimension>::kernel_calls == 1);
    CountedMatrix<Dimension>::kernel_calls = 0;
    alternative.minimize_by_image_generators();
    assert(CountedMatrix<Dimension>::kernel_calls == 0);
    same_image(normal, alternative);
    // A supplied graded_kernel() is not needed at all by the alternative.
    using WithoutKernel = NoKernelMatrix<Dimension>;
    using RealDegree = typename WithoutKernel::degree_type;
    RealDegree real_zero{}, real_one{}, real_two{};
    real_one.fill(1); real_two.fill(2);
    WithoutKernel independent(2, 1, {{0}, {0}}, {real_one, real_two}, {real_zero});
    independent.minimize_by_image_generators();
    assert(independent.get_num_cols() == 1);
}

template<std::size_t Dimension>
auto benchmark_input(int surviving_rows, const std::string& family) {
    using Matrix = CoordGradedSparseMatrix<int, Dimension, int>;
    using Degree = typename Matrix::degree_type;
    const int local = family == "local_pairs" ? surviving_rows : 0;
    Degree zero{}, one{}; one.fill(1);
    vec<Degree> columns, rows(local + surviving_rows, zero);
    array<int> entries;
    for (int row = 0; row < local; ++row) {
        columns.push_back(zero);
        entries.push_back({row});
    }
    for (int row = 0; row < surviving_rows; ++row) {
        const int repetitions = family == "minimal" ? 1 : 4;
        for (int repeat = 0; repeat < repetitions; ++repeat) {
            Degree birth = one;
            if (family == "incomparable" && repeat < int(Dimension)) {
                birth = zero;
                birth[repeat] = 1;
            } else for (auto& value : birth) value += repeat;
            vec<int> column;
            if (local) column.push_back(row % local);
            column.push_back(local + row);
            if (row) column.push_back(local + row - 1);
            std::sort(column.begin(), column.end());
            columns.push_back(birth);
            entries.push_back(std::move(column));
        }
    }
    return Matrix(static_cast<int>(entries.size()), static_cast<int>(rows.size()), entries, columns, rows);
}

template<class Matrix>
void same_output(const Matrix& result, const Matrix& expected) {
    assert(result.data == expected.data);
    assert(result.col_degrees == expected.col_degrees && result.row_degrees == expected.row_degrees);
}

double median(vec<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void write_samples(const vec<double>& samples) {
    std::cout << '"';
    for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? ";" : "") << samples[i];
    std::cout << '"';
}

template<std::size_t Dimension>
void benchmark(int rows, const std::string& family, int repeats) {
    const auto input = benchmark_input<Dimension>(rows, family);
    auto normal_reference = input, alternative_reference = input, cancelled = input;
    normal_reference.minimize();
    alternative_reference.minimize_by_image_generators();
    same_image(normal_reference, alternative_reference);
    cancelled.cancel_local_pairs();
    vec<double> normal_samples, alternative_samples;
    for (int repetition = 0; repetition < repeats; ++repetition) {
        // Copies, validation and output stay outside the timed public calls.
        auto normal = input, alternative = input;
        auto run_normal = [&] {
            const auto start = std::chrono::steady_clock::now();
            normal.minimize();
            normal_samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        };
        auto run_alternative = [&] {
            const auto start = std::chrono::steady_clock::now();
            alternative.minimize_by_image_generators();
            alternative_samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        };
        if (repetition % 2) { run_alternative(); run_normal(); }
        else { run_normal(); run_alternative(); }
        same_output(normal, normal_reference);
        same_output(alternative, alternative_reference);
    }
    std::cout << Dimension << ',' << family << ',' << input.get_num_rows() << ',' << input.get_num_cols()
        << ',' << cancelled.get_num_rows() << ',' << cancelled.get_num_cols()
        << ',' << median(normal_samples) << ',' << median(alternative_samples) << ',' << repeats << ",pass,";
    write_samples(normal_samples);
    std::cout << ',';
    write_samples(alternative_samples);
    std::cout << '\n';
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 9;
        if (repeats < 1 || repeats > 1000) return 2;
        std::cout << std::setprecision(9)
            << "parameters,family,rows,columns,rows_after_cancellation,columns_after_cancellation,normal_ms,image_generators_ms,repeats,verified,normal_samples_ms,image_generators_samples_ms\n";
        for (const std::string family : {"minimal", "duplicates", "incomparable", "local_pairs"}) {
            benchmark<2>(128, family, repeats);
            benchmark<3>(64, family, repeats);
        }
        return 0;
    }
    correctness<CoordGradedSparseMatrix<int, 1, int>>();
    correctness<CoordGradedSparseMatrix<int, 2, int>>();
    correctness<CoordGradedSparseMatrix<int, 3, int>>();
    correctness<CoordGradedSparseMatrix<int, 4, int>>();
    correctness<R2GradedSparseMatrix<int>>();
    correctness<CoordGradedSparseMatrix<double, 2, int, void, CSCStorage<int>>>();
    correctness<CoordGradedSparseMatrix<double, 3, int, void, CSCStorage<int>>>();
    correctness<GridGradedSparseMatrix<double, 2, int>>();
    correctness<GridGradedSparseMatrix<double, 3, int>>();
    no_kernel_vectors<2>();
    no_kernel_vectors<3>();
    no_kernel_vectors<4>();
}
