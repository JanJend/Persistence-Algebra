// NEW: independent fibre-rank oracle for runtime coordinate-graded kernels.
// Run directly with the same C++17 include flags as dynamic_grid_matrix_test.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/dynamic_grid_matrix.hpp>
#include <grlina/dynamic_coordinate_matrix.hpp>
#include <cassert>
#include <cstdint>
#include <random>

using namespace graded_linalg;

namespace {

using Bits = std::uint64_t;
using Degree = DynamicDegree<double>;

template<class Scalar, class Index, class Storage>
Degree column_degree(const DynamicGridGradedSparseMatrix<Scalar, Index, Storage>& matrix, int column) {
    return matrix.real_col_degree(column);
}

template<class Scalar, class Index, class Storage>
Degree column_degree(const DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>& matrix, int column) {
    return Degree(matrix.col_degree(column));
}

template<class Scalar, class Index, class Storage>
vec<double> axis_values(const DynamicGridGradedSparseMatrix<Scalar, Index, Storage>& matrix, std::size_t axis) {
    return matrix.grid(axis);
}

template<class Scalar, class Index, class Storage>
vec<double> axis_values(const DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>& matrix, std::size_t axis) {
    vec<double> values;
    for (const auto& degree : matrix.col_degrees) values.push_back(degree[axis]);
    for (const auto& degree : matrix.row_degrees) values.push_back(degree[axis]);
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

template<class Scalar, class Index, class Storage>
void check_embedding(const DynamicGridGradedSparseMatrix<Scalar, Index, Storage>& source,
                     const DynamicGridGradedSparseMatrix<Scalar, Index, Storage>& kernel) {
    assert(kernel.grids == source.grids);
    assert(kernel.real_row_degrees() == source.real_col_degrees());
}

template<class Scalar, class Index, class Storage>
void check_embedding(const DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>& source,
                     const DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>& kernel) {
    assert(kernel.row_degrees == source.col_degrees);
}

template<class Scalar, class Index, class Storage>
void include_unused(DynamicGridGradedSparseMatrix<Scalar, Index, Storage>& matrix, const Degree& degree) {
    matrix.include_real_degrees({degree});
}

template<class Scalar, class Index, class Storage>
void include_unused(DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>&, const Degree&) {}

int binary_rank(const vec<Bits>& columns) {
    Bits pivots[64]{};
    int rank = 0;
    for (Bits column : columns) {
        for (int row = 63; row >= 0; --row) {
            if (!(column & (Bits{1} << row))) continue;
            if (pivots[row]) column ^= pivots[row];
            else { pivots[row] = column; ++rank; break; }
        }
    }
    return rank;
}

template<class Column> Bits bit_column(const Column& column) {
    Bits result = 0;
    for (int row : column) {
        assert(row >= 0 && row < 64);
        result ^= Bits{1} << row;
    }
    return result;
}

bool available(const Degree& degree, const Degree& grade) {
    assert(degree.size() == grade.size());
    for (std::size_t axis = 0; axis < grade.size(); ++axis)
        if (degree[axis] > grade[axis]) return false;
    return true;
}

template<class Matrix>
void check_fibre(const Matrix& source, const Matrix& kernel, const Degree& grade) {
    vec<Bits> images, relations;
    for (int column = 0; column < source.get_num_cols(); ++column)
        if (available(column_degree(source, column), grade))
            images.push_back(bit_column(source.get_col(column)));
    for (int column = 0; column < kernel.get_num_cols(); ++column) {
        if (!available(column_degree(kernel, column), grade)) continue;
        Bits image = 0;
        for (int domain : kernel.get_col(column)) {
            assert(available(column_degree(source, domain), grade));
            image ^= bit_column(source.get_col(domain));
        }
        assert(image == 0);
        relations.push_back(bit_column(kernel.get_col(column)));
    }
    // Containment plus matching nullity proves equality of the two fibres.
    assert(binary_rank(relations) == static_cast<int>(images.size()) - binary_rank(images));
}

template<class Matrix>
void check_minimal_generators(const Matrix& kernel) {
    for (int candidate = 0; candidate < kernel.get_num_cols(); ++candidate) {
        vec<Bits> earlier;
        const auto grade = column_degree(kernel, candidate);
        for (int column = 0; column < kernel.get_num_cols(); ++column)
            if (column != candidate && available(column_degree(kernel, column), grade))
                earlier.push_back(bit_column(kernel.get_col(column)));
        const int rank = binary_rank(earlier);
        earlier.push_back(bit_column(kernel.get_col(candidate)));
        assert(binary_rank(earlier) == rank + 1);
    }
}

template<class Matrix>
void check_kernel(const Matrix& source, const Matrix& kernel) {
    kernel.validate();
    assert(kernel.parameter_count() == source.parameter_count());
    assert(kernel.get_num_rows() == source.get_num_cols());
    check_embedding(source, kernel);
    check_minimal_generators(kernel);
    Degree grade(source.parameter_count());
    auto visit = [&](auto&& self, std::size_t axis) -> void {
        if (axis == source.parameter_count()) { check_fibre(source, kernel, grade); return; }
        // Grid values are critical grades; also check both exterior regions.
        vec<double> values = axis_values(source, axis);
        if (values.empty()) values.push_back(0.0);
        values.insert(values.begin(), values.front() - 1.0);
        values.push_back(values.back() + 1.0);
        for (double value : values) { grade[axis] = value; self(self, axis + 1); }
    };
    visit(visit, 0);
}

double coordinate(std::size_t axis, int level) {
    return -0.75 + 0.5 * axis + (1.25 + 0.5 * axis) * level;
}

template<template<class, class, class> class Family, class Storage>
void randomized_fibres() {
    using Matrix = Family<double, int, Storage>;
    std::mt19937 random(20261009);
    for (std::size_t dimensions = 0; dimensions <= 4; ++dimensions) {
        for (int sample = 0; sample < 24; ++sample) {
            const int rows = random() % 5, columns = random() % 8;
            vec<Degree> row_degrees(rows, Degree(dimensions));
            vec<Degree> col_degrees(columns, Degree(dimensions));
            for (auto& degree : row_degrees)
                for (std::size_t axis = 0; axis < dimensions; ++axis)
                    degree[axis] = coordinate(axis, random() % 2);
            for (auto& degree : col_degrees)
                for (std::size_t axis = 0; axis < dimensions; ++axis)
                    degree[axis] = coordinate(axis, random() % 3);
            array<int> data(columns);
            for (int column = 0; column < columns; ++column)
                for (int row = 0; row < rows; ++row)
                    if ((random() & 1) && available(row_degrees[row], col_degrees[column]))
                        data[column].push_back(row);
            Matrix source(columns, rows, dimensions, data, col_degrees, row_degrees);
            Degree unused(dimensions);
            for (std::size_t axis = 0; axis < dimensions; ++axis) unused[axis] = coordinate(axis, 4);
            include_unused(source, unused);
            auto working = source;
            const auto kernel = working.graded_kernel();
            check_kernel(source, kernel);
        }
    }
}

template<template<class, class, class> class Family, class Storage>
void pair_completion_and_koszul() {
    using Matrix = Family<double, int, Storage>;
    // The first S-pair creates an image column with a new row pivot. Its pair
    // with the third input reveals the dependence at a three-way join.
    Matrix source(3, 3, 3, {{0,2}, {1,2}, {0,1}},
                  {{1,0,0}, {0,1,0}, {0,0,1}}, {{0,0,0}, {0,0,0}, {0,0,0}});
    auto working = source;
    const auto kernel = working.graded_kernel();
    check_kernel(source, kernel);
    assert(kernel.get_num_cols() == 1 && kernel.get_col(0) == vec<int>({0,1,2}));
    assert(column_degree(kernel, 0) == Degree({1,1,1}));

    // Two earlier relations share the domain pivot 2 at incomparable grades.
    // Their cancellation spans {0,1} at (2,2,0); retaining that third pair
    // would give a generating kernel but an incorrect minimal generator count.
    Matrix cancellation(3, 1, 3, {{0}, {0}, {0}},
                        {{0,2,0}, {2,0,0}, {1,1,0}}, {{0,0,0}});
    auto cancellation_working = cancellation;
    const auto minimal = cancellation_working.graded_kernel();
    check_kernel(cancellation, minimal);
    assert(minimal.get_num_cols() == 2);

    for (int dimensions : {3,4}) {
        vec<Degree> degrees(dimensions, Degree(dimensions));
        for (int axis = 0; axis < dimensions; ++axis) degrees[axis][axis] = 1;
        Matrix differential(dimensions, 1, dimensions, array<int>(dimensions, {0}),
                            degrees, {Degree(dimensions)});
        const vec<int> expected = dimensions == 3 ? vec<int>{3,1,0} : vec<int>{6,4,1,0};
        for (int count : expected) {
            auto input = differential;
            auto next = differential.graded_kernel();
            check_kernel(input, next);
            assert(next.get_num_cols() == count);
            differential = std::move(next);
        }
    }
}

} // namespace

int main() {
    randomized_fibres<DynamicGridGradedSparseMatrix, vec<vec<int>>>();
    randomized_fibres<DynamicGridGradedSparseMatrix, CSCStorage<int>>();
    pair_completion_and_koszul<DynamicGridGradedSparseMatrix, vec<vec<int>>>();
    pair_completion_and_koszul<DynamicGridGradedSparseMatrix, CSCStorage<int>>();
    randomized_fibres<DynamicCoordinateGradedSparseMatrix, vec<vec<int>>>();
    randomized_fibres<DynamicCoordinateGradedSparseMatrix, CSCStorage<int>>();
    pair_completion_and_koszul<DynamicCoordinateGradedSparseMatrix, vec<vec<int>>>();
    pair_completion_and_koszul<DynamicCoordinateGradedSparseMatrix, CSCStorage<int>>();
}
