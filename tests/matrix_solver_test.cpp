#include <grlina/mat_sparse.hpp>
#include <grlina/mat_dense.hpp>
#include <cassert>
#include <set>

using namespace graded_linalg;

template <typename Column>
Column column_from_mask(unsigned mask, int size) {
    if constexpr (std::is_same_v<Column, bitset>) return bitset(size, mask);
    else {
        Column result;
        for (int i = 0; i < size; ++i)
            if (mask & (1u << i)) result.push_back(i);
        return result;
    }
}

template <typename Column>
unsigned column_mask(const Column& column) {
    if constexpr (std::is_same_v<Column, bitset>) return column.to_ulong();
    else {
        unsigned mask = 0;
        for (int entry : column) mask |= 1u << entry;
        return mask;
    }
}

// Independent F2 oracle: each stored column is one equation in A^T x = b.
unsigned evaluate_equations(const array<int>& columns, unsigned x) {
    unsigned rhs = 0;
    for (std::size_t j = 0; j < columns.size(); ++j) {
        bool parity = false;
        for (int i : columns[j]) parity ^= bool(x & (1u << i));
        if (parity) rhs |= 1u << j;
    }
    return rhs;
}

template <typename Matrix>
void check_solver(Matrix matrix, const array<int>& columns) {
    const auto before = matrix.data;
    const auto pivots = matrix.pivots;
    const Matrix& input = matrix;
    for (unsigned b = 0; b < (1u << matrix.get_num_cols()); ++b) {
        bool solvable = false;
        for (unsigned x = 0; x < (1u << matrix.get_num_rows()); ++x)
            solvable |= evaluate_equations(columns, x) == b;
        auto rhs = column_from_mask<typename Matrix::column_type>(b, matrix.get_num_cols());
        const auto original_rhs = rhs;
        bool solved = false;
        try { input.column_gauss_solver(rhs); solved = true; }
        catch (const std::invalid_argument&) { assert(rhs == original_rhs); }
        assert(solved == solvable);
        if (solved) {
            assert(column_mask(rhs) < (1u << matrix.get_num_rows()));
            assert(evaluate_equations(columns, column_mask(rhs)) == b);
        }
        assert(matrix.data == before && matrix.pivots == pivots);
    }
}

template <typename Matrix>
void check_kernel(Matrix matrix, const array<int>& columns) {
    // A cached pivot from an unrelated reduction must not affect the answer.
    matrix.pivots[0] = 0;
    const auto before = matrix.data;
    const auto pivots = matrix.pivots;
    const Matrix& input = matrix;
    auto basis = input.get_kernel();
    static_assert(std::is_same_v<decltype(basis), Matrix>);
    assert(basis.get_num_rows() == matrix.get_num_cols());
    assert(basis.get_num_cols() <= matrix.get_num_cols());
    std::set<unsigned> expected, span;
    for (unsigned x = 0; x < (1u << matrix.get_num_cols()); ++x) {
        unsigned image = 0;
        for (int j = 0; j < matrix.get_num_cols(); ++j)
            if (x & (1u << j)) image ^= column_mask(columns[j]);
        if (!image) expected.insert(x);
    }
    for (unsigned selection = 0; selection < (1u << basis.get_num_cols()); ++selection) {
        unsigned x = 0;
        for (int j = 0; j < basis.get_num_cols(); ++j)
            if (selection & (1u << j)) x ^= column_mask(basis.get_col(j));
        span.insert(x);
    }
    assert(span == expected && span.size() == (1u << basis.get_num_cols()));
    assert(matrix.data == before && matrix.pivots == pivots);
}

template <typename Matrix>
void check_matrix_family() {
    auto make = [](unsigned first, unsigned second) {
        Matrix matrix(2, 3);
        matrix.resize_data(2);
        matrix.set_col(0, column_from_mask<typename Matrix::column_type>(first, 3));
        matrix.set_col(1, column_from_mask<typename Matrix::column_type>(second, 3));
        return matrix;
    };
    const Matrix xy = make(1, 4), y = make(0, 4), x = make(1, 0), z = make(0, 1);
    std::vector<Matrix> A{xy, xy}, B{y, x, z, z};
    if constexpr (std::is_same_v<Matrix, SparseMatrix<int>>) {
        for (auto& matrix : A) matrix.compute_rows_forward();
        for (auto& matrix : B) matrix.compute_rows_forward();
    }
    reduce_matrix_family_modulo(A, B);
    assert(A.size() == 2 && A[1].is_zero());
    assert(B.size() == 2 && B[0].data == x.data && B[1].data == z.data);
    if constexpr (std::is_same_v<Matrix, SparseMatrix<int>>) {
        assert(A[0].rows_computed && !A[1].rows_computed);
        assert(!B[0].rows_computed && B[1].rows_computed);
        B[0].compute_rows_forward();
        assert(B[0]._rows == array<int>({{0}, {}, {}}));
    }
    std::vector<Matrix> empty;
    reduce_matrix_family_modulo(empty, B);
    assert(B.size() == 2);
    reduce_matrix_family_modulo(B, empty);
    assert(empty.empty());
}

int main() {
    check_matrix_family<SparseMatrix<int>>();
    check_matrix_family<DenseMatrix>();
    // Exhaust every matrix and RHS through 3x3, including rectangular, singular,
    // inconsistent and empty systems. Verify complete independent kernel spans.
    for (int cols = 0; cols <= 3; ++cols)
        for (int rows = 0; rows <= 3; ++rows)
            for (unsigned entries = 0; entries < (1u << (cols * rows)); ++entries) {
                array<int> columns(cols);
                DenseMatrix dense(cols, rows);
                for (int j = 0; j < cols; ++j) {
                    columns[j] = column_from_mask<vec<int>>(entries >> (j * rows), rows);
                    dense.set_col(j, bitset(rows, entries >> (j * rows)));
                }
                SparseMatrix<int> sparse(cols, rows, columns);
                check_solver(sparse, columns);
                check_solver(dense, columns);
                check_kernel(sparse, columns);
            }

    SparseMatrix<int> sparse(1, 1, array<int>{{0}});
    vec<int> invalid{1};
    bool rejected = false;
    try { sparse.column_gauss_solver(invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && invalid == vec<int>{1});
    DenseMatrix dense(1, 1);
    bitset wrong_size(2);
    rejected = false;
    try { dense.column_gauss_solver(wrong_size); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && wrong_size.size() == 2);
}
