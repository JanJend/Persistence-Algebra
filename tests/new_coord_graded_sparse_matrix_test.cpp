#include <grlina/mat_coord.hpp>
#include <grlina/mat_r2.hpp>
#include <cassert>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>

using namespace graded_linalg;

static int binary_rank(const vec<std::uint64_t>& columns) {
    std::uint64_t pivots[64]{};
    int rank = 0;
    for (auto column : columns)
        for (int row = 63; row >= 0; --row) {
            if (!(column & (std::uint64_t{1} << row))) continue;
            if (pivots[row]) column ^= pivots[row];
            else { pivots[row] = column; ++rank; break; }
        }
    return rank;
}

template<class Matrix>
static auto column_degrees(const Matrix& matrix) {
    if constexpr (Matrix::grid_backed) return matrix.real_col_degrees();
    else return matrix.col_degrees;
}

template<class Matrix>
static void check_kernel(const Matrix& source, const Matrix& kernel) {
    using Degree = typename decltype(column_degrees(source))::value_type;
    using DT = Degree_traits<Degree>;
    auto original = column_degrees(source), degrees = column_degrees(kernel);
    assert(kernel.get_num_rows() == source.get_num_cols());
    if constexpr (Matrix::grid_backed) {
        assert(kernel.grids == source.grids);
        assert(kernel.real_row_degrees() == original);
    } else assert(kernel.row_degrees == original);

    auto bits = [](const auto& column) {
        std::uint64_t value = 0;
        for (int row : column) value ^= std::uint64_t{1} << row;
        return value;
    };
    for (int i = 0; i < kernel.get_num_cols(); ++i) {
        std::uint64_t image = 0;
        for (int row : kernel.get_col(i)) {
            assert(DT::smaller_equal(original[row], degrees[i]));
            image ^= bits(source.get_col(row));
        }
        assert(image == 0);
        vec<std::uint64_t> others;
        for (int j = 0; j < kernel.get_num_cols(); ++j)
            if (j != i && DT::smaller_equal(degrees[j], degrees[i])) others.push_back(bits(kernel.get_col(j)));
        int rank = binary_rank(others);
        others.push_back(bits(kernel.get_col(i)));
        assert(binary_rank(others) == rank + 1);
    }
    Degree grade{};
    auto visit = [&](auto&& self, std::size_t axis) -> void {
        if (axis == grade.size()) {
            vec<std::uint64_t> images, relations;
            for (int i = 0; i < source.get_num_cols(); ++i)
                if (DT::smaller_equal(original[i], grade)) images.push_back(bits(source.get_col(i)));
            for (int i = 0; i < kernel.get_num_cols(); ++i)
                if (DT::smaller_equal(degrees[i], grade)) relations.push_back(bits(kernel.get_col(i)));
            assert(binary_rank(relations) == static_cast<int>(images.size()) - binary_rank(images));
            return;
        }
        vec<typename Degree::value_type> values;
        for (const auto& degree : original) values.push_back(degree[axis]);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (values.empty()) values.push_back(0);
        values.insert(values.begin(), values.front() - 1);
        values.push_back(values.back() + 1);
        for (auto value : values) { grade[axis] = value; self(self, axis + 1); }
    };
    visit(visit, 0);
}

template<class Scalar, std::size_t Dimension>
static void randomized_kernels() {
    using Coordinates = CoordGradedSparseMatrix<Scalar, Dimension, int>;
    using Grid = GridGradedSparseMatrix<Scalar, Dimension, int>;
    using Degree = typename Coordinates::degree_type;
    static_assert(has_matrix_graded_kernel<Coordinates>::value);
    static_assert(has_matrix_graded_kernel<Grid>::value);
    std::mt19937 random(20261009 + Dimension);
    for (int sample = 0; sample < 16; ++sample) {
        int columns = random() % 8, rows = random() % 5;
        vec<Degree> c(columns), r(rows);
        auto fill = [&](auto& degrees) {
            for (auto& degree : degrees)
                for (std::size_t a = 0; a < Dimension; ++a) {
                    // Integer coordinates deliberately exceed the matrix index range.
                    if constexpr (std::is_integral_v<Scalar>) degree[a] = (Scalar(random() % 3) - 1) * Scalar(10000000000LL + a);
                    else degree[a] = (Scalar(random() % 3) - 1) * Scalar(1.25 + a);
                }
        };
        fill(c); fill(r);
        array<int> data(columns);
        for (int i = 0; i < columns; ++i)
            for (int j = 0; j < rows; ++j)
                if ((random() & 1) && Degree_traits<Degree>::smaller_equal(r[j], c[i])) data[i].push_back(j);
        Coordinates source(columns, rows, data, c, r);
        check_kernel(source, static_cast<const Coordinates&>(source).graded_kernel());
        Grid grid(source);
        Degree unused{};
        unused.fill(Scalar(99));
        grid.include_real_degrees({unused});
        check_kernel(grid, static_cast<const Grid&>(grid).graded_kernel());
    }
}

// A general graded matrix: no dimension query, grid conversion, scheduler,
// row queues, kernel-specific reduction methods or empty_like factory.
struct PlainZ2Matrix : GradedSparseMatrix<CoordinateDegree<long long, 2>, int, PlainZ2Matrix> {
    using Base = GradedSparseMatrix<CoordinateDegree<long long, 2>, int, PlainZ2Matrix>;
    using Base::Base;
};

static void legacy_kernel_comparison() {
    using Coordinates = CoordGradedSparseMatrix<long long, 2, int>;
    using Grid = GridGradedSparseMatrix<long long, 2, int>;
    using Degree = Coordinates::degree_type;
    auto real_degrees = [](const auto& degrees) {
        vec<r2degree> result;
        for (const auto& degree : degrees)
            result.push_back({static_cast<double>(degree[0]), static_cast<double>(degree[1])});
        return result;
    };
    auto compare = [&](const Coordinates& source) {
        auto native = source;
        auto native_kernel = native.graded_kernel();
        PlainZ2Matrix plain(source.get_num_cols(), source.get_num_rows(), source.data,
                            source.col_degrees, source.row_degrees);
        // A stale matrix-owned pivot must not participate in this computation.
        plain.pivots[0] = 1000;
        auto plain_kernel = z2_graded_kernel(plain);
        assert(plain_kernel.data == native_kernel.data);
        assert(plain_kernel.col_degrees == native_kernel.col_degrees);
        assert(plain_kernel.row_degrees == native_kernel.row_degrees);
        assert(plain.data == native.data && plain.col_degrees == native.col_degrees);
        Grid original_grid(source);
        // Unused grid points change ordinal labels without changing the order.
        original_grid.include_real_degrees({{-10000000001LL, -10000000001LL}, {1, 1}});
        auto grid = original_grid;
        auto grid_kernel = grid.graded_kernel();
        R2GradedSparseMatrix<int> legacy(source.get_num_cols(), source.get_num_rows(), source.data,
                                        real_degrees(source.col_degrees), real_degrees(source.row_degrees));
        auto legacy_kernel = legacy.graded_kernel();

        check_kernel(source, native_kernel);
        check_kernel(original_grid, grid_kernel);
        assert(native_kernel.data == legacy_kernel.data);
        assert(grid_kernel.data == legacy_kernel.data);
        assert(real_degrees(native_kernel.col_degrees) == legacy_kernel.col_degrees);
        assert(real_degrees(grid_kernel.real_col_degrees()) == legacy_kernel.col_degrees);
        assert(real_degrees(native_kernel.row_degrees) == legacy_kernel.row_degrees);
        assert(real_degrees(grid_kernel.real_row_degrees()) == legacy_kernel.row_degrees);
        assert(native.data == legacy.data && grid.data == legacy.data);
        assert(real_degrees(native.col_degrees) == legacy.col_degrees);
        assert(real_degrees(grid.real_col_degrees()) == legacy.col_degrees);
        assert(real_degrees(native.row_degrees) == legacy.row_degrees);
        assert(real_degrees(grid.real_row_degrees()) == legacy.row_degrees);
    };
    // All 2x3 binary matrices and all assignments of four incomparable/comparable
    // grades: ties, unsorted domains, zero columns and delayed kernel births.
    const vec<Degree> grades{{-1, -1}, {-1, 2}, {2, -1}, {2, 2}};
    for (unsigned entries = 0; entries < 64; ++entries)
        for (unsigned labels = 0; labels < 64; ++labels) {
            array<int> data(3);
            vec<Degree> degrees(3);
            for (int c = 0; c < 3; ++c) {
                degrees[c] = grades[(labels >> (2 * c)) & 3];
                for (int r = 0; r < 2; ++r)
                    if ((entries >> (2 * c + r)) & 1) data[c].push_back(r);
            }
            compare(Coordinates(3, 2, data, degrees, {{-2, -3}, {-3, -2}}));
        }
    std::mt19937 random(20261011);
    const vec<long long> values{-10000000000LL, -3, 0, 2, 11, 10000000000LL};
    for (int sample = 0; sample < 1000; ++sample) {
        int columns = random() % 9, rows = random() % 6;
        vec<Degree> c(columns), r(rows);
        for (auto* degrees : {&c, &r})
            for (auto& degree : *degrees)
                for (auto& coordinate : degree) coordinate = values[random() % values.size()];
        array<int> data(columns);
        for (int i = 0; i < columns; ++i)
            for (int j = 0; j < rows; ++j)
                if ((random() & 1) && Degree_traits<Degree>::smaller_equal(r[j], c[i])) data[i].push_back(j);
        compare(Coordinates(columns, rows, data, c, r));
    }
    // Native integer grades must remain exact even beyond double's precision.
    const long long exact = 9007199254740993LL;
    Coordinates precise(2, 1, {{0}, {0}}, {{exact - 1, 0}, {exact, 0}}, {{0, 0}});
    auto kernel = precise.graded_kernel();
    assert((kernel.data == array<int>{{0, 1}}));
    assert((kernel.col_degrees == vec<Degree>{{exact, 0}}));
}

template<std::size_t Dimension>
static void koszul_kernels() {
    using Matrix = CoordGradedSparseMatrix<double, Dimension, int>;
    using Degree = typename Matrix::degree_type;
    vec<Degree> degrees(Dimension);
    for (std::size_t a = 0; a < Dimension; ++a) degrees[a][a] = 1;
    Matrix differential(Dimension, 1, array<int>(Dimension, {0}), degrees, {Degree{}});
    vec<int> counts = Dimension == 3 ? vec<int>{3, 1, 0} : vec<int>{6, 4, 1, 0};
    for (int count : counts) {
        auto next = static_cast<const Matrix&>(differential).graded_kernel();
        check_kernel(differential, next);
        assert(next.get_num_cols() == count);
        differential = std::move(next);
    }
}

static void muphasa_three_parameter_kernels() {
    using Coordinates = CoordGradedSparseMatrix<long long, 3, int>;
    using Grid = GridGradedSparseMatrix<long long, 3, int>;
    using Degree = Coordinates::degree_type;
    auto check = [](const Coordinates& source) {
        auto working = source;
        auto kernel = working.graded_kernel();
        check_kernel(source, kernel);
        assert(working.data == source.data && working.col_degrees == source.col_degrees);
        assert(working.row_degrees == source.row_degrees);
        Grid grid(source);
        const auto original_grid = grid;
        auto grid_kernel = grid.graded_kernel();
        check_kernel(original_grid, grid_kernel);
        assert(grid.data == original_grid.data && grid.col_degrees == original_grid.col_degrees);
        assert(grid.row_degrees == original_grid.row_degrees && grid.grids == original_grid.grids);
        assert(grid_kernel.data == kernel.data && grid_kernel.real_col_degrees() == kernel.col_degrees);
        return kernel;
    };
    // The first pivot collision creates a new image pivot whose next collision
    // reveals the relation at the three-way join. Rows of K must still refer
    // to the original, non-colexicographically ordered columns.
    auto completed = check(Coordinates(3, 3, {{0,2}, {1,2}, {0,1}},
        {{1,0,0}, {0,1,0}, {0,0,1}}, {{0,0,0}, {0,0,0}, {0,0,0}}));
    assert((completed.data == array<int>{{0,1,2}}));
    assert((completed.col_degrees == vec<Degree>{{1,1,1}}));
    // A kernel Groebner basis has a third, redundant generator at (2,2,0).
    auto minimal = check(Coordinates(3, 1, {{0}, {0}, {0}},
        {{0,2,0}, {2,0,0}, {1,1,0}}, {{0,0,0}}));
    assert(minimal.get_num_cols() == 2);
    assert((minimal.col_degrees == vec<Degree>{{1,2,0}, {2,1,0}}));
    // Empty event queues and zero input columns must not index pivot -1.
    check(Coordinates(0, 0));
    auto zero_columns = check(Coordinates(3, 1, {{}, {0}, {}},
        {{2,0,1}, {0,0,0}, {0,2,1}}, {{0,0,0}}));
    assert((zero_columns.data == array<int>{{2}, {0}}));
    check(Coordinates(3, 0, {{}, {}, {}}, {{1,0,0}, {0,1,0}, {0,0,1}}, {}));
    // The difference of these coordinates exceeds signed 64-bit range;
    // scheduling comparisons must use joins rather than coordinate subtraction.
    const long long low = std::numeric_limits<long long>::lowest() + 1;
    const long long high = std::numeric_limits<long long>::max() - 1;
    check(Coordinates(3, 1, {{0}, {0}, {0}},
        {{low,high,0}, {high,low,0}, {0,0,high}}, {{low,low,low}}));

    // Dense random maps exercise pivot joins without the many zero columns
    // produced when random row degrees exclude most entries. The checker uses
    // independent binary ranks at every grade, including minimality of K.
    std::mt19937 random(20261011);
    for (int sample = 0; sample < 256; ++sample) {
        int columns = random() % 9, rows = random() % 5;
        vec<Degree> degrees(columns), row_degrees(rows, Degree{-1,-1,-1});
        array<int> entries(columns);
        for (int c = 0; c < columns; ++c) {
            for (auto& coordinate : degrees[c]) coordinate = random() % 4;
            for (int r = 0; r < rows; ++r)
                if (random() & 1) entries[c].push_back(r);
        }
        check(Coordinates(columns, rows, entries, degrees, row_degrees));
    }
}

static void local_cancellation() {
    using Matrix = CoordGradedSparseMatrix<double, 2, int>;
    using Degree = Matrix::degree_type;
    using DT = Degree_traits<Degree>;
    auto bits = [](const auto& column) {
        std::uint64_t result = 0;
        for (int r : column) result ^= std::uint64_t{1} << r;
        return result;
    };
    // A reducer can contain an earlier paired row; transporting elements must
    // follow both substitutions, including when an element becomes zero.
    Matrix example(3, 3, {{0, 1}, {1, 2}, {0, 1, 2}},
                   {{1, 1}, {1, 1}, {2, 2}}, {{0, 0}, {1, 1}, {1, 1}});
    example.sort_compatibly();
    Matrix elements(4, 3, {{2}, {0, 2}, {1, 2}, {1}},
                    {{4, 4}, {3, 3}, {1, 1}, {2, 2}}, example.row_degrees);
    const auto element_degrees = elements.col_degrees;
    example.compute_rows_forward();
    elements.compute_rows_forward();
    example.cancel_local_pairs(&elements);
    assert((example.data == array<int>{{0}}));
    assert((example.col_degrees == vec<Degree>{{2, 2}}));
    assert((example.row_degrees == vec<Degree>{{0, 0}}));
    assert((elements.data == array<int>{{0}, {}, {}, {0}}));
    assert(elements.col_degrees == element_degrees);
    assert(!example.rows_computed && !elements.rows_computed);
    assert(example.compatibly_sorted);

    // The equal-degree row is not the physical column tail. Surviving rows
    // must keep their original order rather than the temporary degree order.
    Matrix unsorted(1, 3, {{0, 1, 2}}, {{2, 2}}, {{1, 1}, {2, 2}, {0, 0}});
    Matrix transported(3, 3, "Identity");
    transported.col_degrees = transported.row_degrees = unsorted.row_degrees;
    unsorted.cancel_local_pairs(&transported);
    assert(unsorted.get_num_cols() == 0);
    assert((unsorted.row_degrees == vec<Degree>{{1, 1}, {0, 0}}));
    assert((transported.data == array<int>{{0}, {0, 1}, {1}}));

    std::mt19937 random(20261011);
    for (int sample = 0; sample < 300; ++sample) {
        int rows = random() % 10, columns = random() % 13;
        vec<Degree> r(rows), c(columns);
        for (auto& degree : r) degree = {double(random() % 3), double(random() % 3)};
        for (auto& degree : c)
            degree = rows && (random() & 1) ? r[random() % rows]
                : Degree{double(random() % 4), double(random() % 4)};
        array<int> data(columns);
        for (int j = 0; j < columns; ++j)
            for (int i = 0; i < rows; ++i)
                if ((random() & 1) && DT::smaller_equal(r[i], c[j])) data[j].push_back(i);
        Matrix reduced(columns, rows, data, c, r);
        if (sample % 3 == 1) reduced.sort_compatibly();
        if (sample % 3 == 2) reduced.sort_colexicographically();
        const auto original = reduced;
        Matrix projection(rows, rows, "Identity");
        projection.col_degrees = projection.row_degrees = original.row_degrees;
        auto without_elements = reduced;
        without_elements.cancel_local_pairs();
        reduced.cancel_local_pairs(&projection);
        assert(reduced.data == without_elements.data);
        assert(reduced.col_degrees == without_elements.col_degrees);
        assert(reduced.row_degrees == without_elements.row_degrees);
        assert(projection.col_degrees == original.row_degrees);
        assert(projection.row_degrees == reduced.row_degrees);
        assert(reduced.is_graded_matrix() && projection.is_graded_matrix());
        for (int j = 0; j < reduced.get_num_cols(); ++j)
            for (int i : reduced.data[j]) assert(!DT::equals(reduced.col_degrees[j], reduced.row_degrees[i]));
        // Independent binary ranks check kernel/cokernel dimensions at every
        // grade, and that the transported projection induces a module map.
        for (int x = 0; x < 4; ++x) for (int y = 0; y < 4; ++y) {
            Degree degree{double(x), double(y)};
            vec<std::uint64_t> old_images, new_images, projected_generators;
            int old_rows = 0, new_rows = 0;
            for (const auto& d : original.row_degrees) old_rows += DT::smaller_equal(d, degree);
            for (const auto& d : reduced.row_degrees) new_rows += DT::smaller_equal(d, degree);
            for (int j = 0; j < original.get_num_cols(); ++j)
                if (DT::smaller_equal(original.col_degrees[j], degree)) old_images.push_back(bits(original.data[j]));
            for (int j = 0; j < reduced.get_num_cols(); ++j)
                if (DT::smaller_equal(reduced.col_degrees[j], degree)) new_images.push_back(bits(reduced.data[j]));
            int old_rank = binary_rank(old_images), new_rank = binary_rank(new_images);
            assert(old_rows - old_rank == new_rows - new_rank);
            assert(int(old_images.size()) - old_rank == int(new_images.size()) - new_rank);
            for (int j = 0; j < projection.get_num_cols(); ++j)
                if (DT::smaller_equal(projection.col_degrees[j], degree)) projected_generators.push_back(bits(projection.data[j]));
            assert(binary_rank(projected_generators) == new_rows);
            for (int j = 0; j < original.get_num_cols(); ++j) {
                if (!DT::smaller_equal(original.col_degrees[j], degree)) continue;
                std::uint64_t image = 0;
                for (int i : original.data[j]) image ^= bits(projection.data[i]);
                new_images.push_back(image);
            }
            assert(binary_rank(new_images) == new_rank);
        }
        auto second = reduced;
        second.cancel_local_pairs();
        assert(second.data == reduced.data && second.row_degrees == reduced.row_degrees);
    }
}

int main() {
    using Matrix = CoordGradedSparseMatrix<double, 2, int>;
    using Degree = Matrix::degree_type;
    static_assert(is_graded_sparse_matrix_v<Matrix>);

    Matrix matrix(2, 3, {{0, 1}, {0, 2}},
                  {{3, 1}, {1, 1}}, {{0, 0}, {1, 0}, {0, 1}});
    matrix.sort_compatibly();

    assert((matrix.col_degrees == vec<Degree>{{1, 1}, {3, 1}}));
    assert((matrix.data == array<int>{{0, 1}, {0, 2}}));
    auto at_degree = matrix.map_at_degree_pair(Degree{1, 1}).first;
    assert(at_degree.get_num_cols() == 1 && at_degree.get_num_rows() == 3);
    assert(matrix.dim_at(Degree{1, 1}) == 2);

    Matrix identity(2, 2, "Identity");
    identity.row_degrees = identity.col_degrees = matrix.col_degrees;
    identity.compatibly_sorted = false;
    Matrix product = matrix * identity;
    assert(product.data == matrix.data);
    assert(product.row_degrees == matrix.row_degrees);
    assert(product.col_degrees == matrix.col_degrees);

    vec<int> columns{1};
    Matrix restricted = matrix.restricted_domain_copy(columns);
    assert((restricted.data == array<int>{{0, 2}}));
    assert((restricted.col_degrees == vec<Degree>{{3, 1}}));
    assert(restricted.row_degrees == matrix.row_degrees);

    Matrix transposed = matrix.transposed_copy();
    assert((transposed.data == array<int>{{0, 1}, {0}, {1}}));
    assert(transposed.row_degrees == matrix.col_degrees);
    assert(transposed.col_degrees == matrix.row_degrees);

    std::stringstream stream;
    matrix.to_stream(stream);
    Matrix loaded(stream);
    assert(loaded.data == matrix.data);
    assert(loaded.row_degrees == matrix.row_degrees);
    assert(loaded.col_degrees == matrix.col_degrees);

    randomized_kernels<long long, 0>();
    randomized_kernels<long long, 1>();
    randomized_kernels<long long, 2>();
    randomized_kernels<long long, 3>();
    randomized_kernels<long long, 4>();
    randomized_kernels<double, 0>();
    randomized_kernels<double, 1>();
    randomized_kernels<double, 2>();
    randomized_kernels<double, 3>();
    randomized_kernels<double, 4>();
    koszul_kernels<3>();
    koszul_kernels<4>();
    muphasa_three_parameter_kernels();
    legacy_kernel_comparison();
    local_cancellation();
}
