#include <grlina/storage/compact.hpp>
#include <grlina/graded_linear_system.hpp>
#include <grlina/module/module_base.hpp>
#include <grlina/quiver_representations.hpp>
#include <cassert>
#include <random>
#include <sstream>
#include <iostream>

using namespace graded_linalg;

template<class Index>
void columns() {
    std::mt19937 random(42);
    for (int repeat = 0; repeat < 500; ++repeat) {
        vec<Index> a, b;
        for (int row = 0; row < 80; ++row) {
            if (random() % 9 == 0) a.push_back(row);
            if (random() % 9 == 0) b.push_back(row);
        }
        CompactColumn<Index> left(a), right(b);
        auto expected = a;
        Column_traits<vec<Index>, Index>::add_to(b, expected);
        left.xor_with(right);
        assert(vec<Index>(left.begin(), left.end()) == expected);
        auto copy = left;
        auto moved = std::move(left);
        assert(moved == copy && left.empty());
        moved.xor_with(moved);
        assert(moved.empty() && moved.capacity() == 3);
    }
    CompactColumn<Index> column{0, 1, 2};
    assert(column.capacity() == 3);
    column.push_back(3);
    auto* heap = column.data();
    auto moved = std::move(column);
    assert(moved.data() == heap && column.empty());
    moved.pop_back();
    assert(moved.capacity() == 3);
    moved.assign(moved.begin() + 1, moved.end());
    Column_traits<CompactColumn<Index>, Index>::set_entry(moved, 0);
    Column_traits<CompactColumn<Index>, Index>::set_entry(moved, 1);
    assert((moved == CompactColumn<Index>{0, 2}));
    moved = moved;
    assert((moved == CompactColumn<Index>{0, 2}));
    if constexpr (sizeof(Index) > 4) {
        moved = CompactColumn<Index>{Index{1} << 40};
        assert(moved.back() == (Index{1} << 40));
    }
}

template<class Matrix>
void algebra() {
    using Index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    const vec<Degree> vertices(2, Degree{0, 0}), edges{{1, 0}, {0, 1}, {1, 1}};
    Matrix matrix(3, 2, {{0, 1}, {0, 1}, {0, 1}}, edges, vertices);
    auto working = matrix;
    auto kernel = working.graded_kernel();
    assert((matrix * kernel).is_zero());
    assert(kernel.get_num_cols() == 2);
    auto solution = solve_graded_linear_system(matrix, matrix);
    assert(solution && (matrix * *solution).data == matrix.data);
    auto doubled = matrix + matrix;
    assert(doubled.is_zero());
    auto appended = matrix;
    appended.append_matrix(matrix);
    assert(appended.get_num_cols() == 6);
    auto removed = matrix;
    vec<Index> rows{1};
    removed.delete_rows(rows);
    assert(removed.get_num_rows() == 1 && removed.get_col(0) == vec<Index>{0});
    Module<Matrix> module(matrix);
    module.compute_projective_resolution();
    module.minimize_resolution();
    assert(module.presentation().is_minimal());
    ChainComplex<Matrix> complex{matrix, kernel};
    assert(complex.is_chain_complex());
    auto endpoint = std::make_shared<ChainComplex<Matrix>>(std::move(complex));
    ChainMap<Matrix> identity(endpoint, endpoint,
        {Matrix(vertices), Matrix(edges), Matrix(geometric_col_degrees(kernel))});
    identity.minimize();
    assert(identity.mapping_cone().is_chain_complex());
    std::stringstream stream;
    matrix.to_stream(stream);
    Matrix reloaded(stream);
    assert(reloaded.data == matrix.data);
    if constexpr (matrix_grid_backed_v<Matrix>)
        assert(matrix.to_coordinates().data == matrix.data);
}

void dense_reduction_and_conversion() {
    using Compact = R2Matrix<int, CompactStorage<int>>;
    using Vectors = R2Matrix<int, vec<vec<int>>>;
    std::mt19937 random(123);
    array<int> data(70);
    for (int j = 0; j < 60; ++j)
        for (int row = 0; row < 512; ++row)
            if (random() % 2) data[j].push_back(row);
    for (int j = 60; j < 70; ++j) data[j] = data[j - 60];
    const vec<r2degree> columns(70, {0, 0}), rows(512, {0, 0});
    Compact compact(70, 512, data, columns, rows);
    Vectors vectors(70, 512, data, columns, rows);
    auto compact_kernel = compact.graded_kernel();
    auto vector_kernel = vectors.graded_kernel();
    assert(compact_kernel.data == vector_kernel.data);
    assert(compact_kernel.col_degrees == vector_kernel.col_degrees);
    CSCMatrix<int> csc(70, 512);
    csc.assign_data(compact.data);
    compact.assign_data(csc.data);
    vectors.assign_data(compact.data);
    assert(compact.data == vectors.data);
    auto source = std::move(compact);
    const auto* allocation = source.column(0).data();
    compact.resize_data(70);
    compact.move_col_from(0, source, 0);
    assert(compact.column(0).data() == allocation && source.column(0).empty());
}

// Run each storage separately under /usr/bin/time -l to measure peak memory.
template<class Storage>
void memory_example() {
    Storage data(1000000);
    for (auto& column : data) column = {0, 1, 2};
    std::size_t bytes = data.capacity() * sizeof(typename Storage::value_type), checksum = 0;
    for (const auto& column : data) {
        checksum += column.back();
        if constexpr (is_compact_storage_v<Storage>) {
            if (column.capacity() > 3) bytes += column.capacity() * sizeof(int);
        } else bytes += column.capacity() * sizeof(int);
    }
    std::cout << "columns=" << data.size() << " owned_bytes=" << bytes << " checksum=" << checksum << '\n';
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--memory-compact") { memory_example<CompactStorage<int>>(); return 0; }
    if (argc == 2 && std::string(argv[1]) == "--memory-vector") { memory_example<vec<vec<int>>>(); return 0; }
    columns<int>();
    columns<long long>();
    dense_reduction_and_conversion();
    algebra<R2Matrix<int, CompactStorage<int>>>();
    algebra<R3Matrix<int, CompactStorage<int>>>();
    algebra<CoordGradedSparseMatrix<double, 4, int, void, CompactStorage<int>>>();
    algebra<GridGradedSparseMatrix<double, 2, int, CompactStorage<int>>>();
    algebra<R2GradedSparseMatrix<int, CompactStorage<int>>>();
}
