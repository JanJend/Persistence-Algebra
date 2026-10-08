// These tests deliberately exercise expensive operations; warning behavior is
// tested separately so this differential suite can run without diagnostic noise.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/csc_matrix.hpp>
#include <cassert>
#include <random>
#include <type_traits>

using namespace graded_linalg;

template<class Index>
void assert_csc_invariants(const CSCMatrix<Index>& matrix) {
    const auto& offsets = matrix.data.offsets();
    const auto& entries = matrix.data.entries();
    assert(offsets.size() == static_cast<std::size_t>(matrix.get_num_cols()) + 1);
    assert(offsets.front() == 0);
    assert(offsets.back() == entries.size());
    assert(std::is_sorted(offsets.begin(), offsets.end()));
    for (Index i = 0; i < matrix.get_num_cols(); ++i) {
        const auto column = matrix.column(i);
        assert(column.size() == offsets[i + 1] - offsets[i]);
        assert(std::equal(column.begin(), column.end(), entries.begin() + offsets[i]));
    }
}

template<class Left, class Index>
void assert_same(const Left& left, const CSCMatrix<Index>& right) {
    assert(left.get_num_cols() == right.get_num_cols());
    assert(left.get_num_rows() == right.get_num_rows());
    assert(left.data.size() == right.data.size());
    for (Index i = 0; i < right.get_num_cols(); ++i)
        assert(left.get_col(i) == right.get_col(i));
    assert_csc_invariants(right);
}

void test_shared_algorithms() {
    std::mt19937 generator(419);
    for (int trial = 0; trial < 50; ++trial) {
        const int cols = 1 + generator() % 9;
        const int rows = 1 + generator() % 9;
        array<int> data(cols);
        for (int col = 0; col < cols; ++col)
            for (int row = 0; row < rows; ++row)
                if (generator() % 3 == 0) data[col].push_back(row);
        SparseMatrix<int> original(cols, rows, data);
        CSCMatrix<int> compressed(cols, rows, data);
        assert_same(original, compressed);
        assert_same(original.transposed_copy(), compressed.transposed_copy());
        assert_same(original + original, compressed + compressed);
        assert((compressed + compressed).is_zero());
        auto brace_edit = compressed;
        auto vector_brace_edit = original;
        brace_edit.add_to_col(0, {0});
        vector_brace_edit.add_to_col(0, {0});
        assert_same(vector_brace_edit, brace_edit);

        auto ordinary = original;
        auto packed = compressed;
        ordinary.col_op(0, cols - 1);
        packed.col_op(0, cols - 1);
        assert_same(ordinary, packed);
        assert_same(original, compressed);

        ordinary = original;
        packed = compressed;
        ordinary.column_reduction_triangular();
        packed.column_reduction_triangular();
        assert_same(ordinary, packed);

        ordinary = original;
        packed = compressed;
        auto kernel = ordinary.kernel();
        auto packed_kernel = packed.kernel();
        static_assert(std::is_same_v<decltype(packed_kernel), CSCMatrix<int>>);
        assert_same(ordinary, packed);
        assert_same(kernel, packed_kernel);
        auto composition = multiply(compressed, packed_kernel);
        assert_same(multiply(original, kernel), composition);
        assert(composition.is_zero());

        ordinary = original;
        packed = compressed;
        auto narrow_kernel = ordinary.get_kernel_int<short>();
        auto packed_narrow_kernel = packed.get_kernel_int<short>();
        static_assert(std::is_same_v<decltype(packed_narrow_kernel), CSCMatrix<short>>);
        assert_same(narrow_kernel, packed_narrow_kernel);

        ordinary = original;
        packed = compressed;
        ordinary.delete_rows(vec<int>{0});
        packed.delete_rows(vec<int>{0});
        assert_same(ordinary, packed);
        ordinary = original;
        packed = compressed;
        ordinary.delete_columns(vec<int>{0});
        packed.delete_columns(vec<int>{0});
        assert_same(ordinary, packed);
        ordinary = original;
        packed = compressed;
        ordinary.delete_last_entries();
        packed.delete_last_entries();
        assert_same(ordinary, packed);

        ordinary = original;
        packed = compressed;
        ordinary.compute_rows_forward();
        packed.compute_rows_forward();
        ordinary.compute_columns_from_rows();
        packed.compute_columns_from_rows();
        assert_same(ordinary, packed);
        assert_same(original, packed);

        vec<int> indices;
        for (int i = 0; i < cols; i += 2) indices.push_back(i);
        assert_same(original.restricted_domain_copy(indices),
                    compressed.restricted_domain_copy(indices));
        vec<int> mask{0};
        assert(original.multiply_with_sparse_vector(mask)
               == compressed.multiply_with_sparse_vector(mask));
        bitset bits(cols);
        bits.set(0);
        assert(original.multiply_with_dense_vector(bits)
               == compressed.multiply_with_dense_vector(bits));

        ordinary = original;
        packed = compressed;
        add_to(original, ordinary);
        add_to(compressed, packed);
        assert_same(ordinary, packed);
        assert(packed.is_zero());

        ordinary = original;
        packed = compressed;
        vec<int> basis, packed_basis;
        auto cokernel = ordinary.coKernel(false, &basis);
        auto packed_cokernel = packed.coKernel(false, &packed_basis);
        assert_same(cokernel, packed_cokernel);
        assert(basis == packed_basis);
        auto quotient_composition = packed_cokernel.multiply_right(compressed);
        assert_same(cokernel.multiply_right(original), quotient_composition);
        assert(quotient_composition.is_zero());
    }
}

void test_copy_on_write_and_boundaries() {
    const array<int> columns{{0, 3}, {}, {1, 2, 4}, {}};
    CSCMatrix<int> original(4, 5, columns);
    auto equal_size_edit = original;
    assert(original.data.shares_storage_with(equal_size_edit.data));
    auto owned_column = equal_size_edit.get_col(0);
    owned_column[0] = 1;
    assert(original.data.shares_storage_with(equal_size_edit.data));
    equal_size_edit.set_col(0, owned_column);
    assert(!original.data.shares_storage_with(equal_size_edit.data));
    assert(original.get_col(0) == columns[0]);
    assert(equal_size_edit.get_col(0) == (vec<int>{1, 3}));
    assert_csc_invariants(original);
    assert_csc_invariants(equal_size_edit);

    auto grow = original;
    grow.col_op(2, 1);
    assert(grow.get_col(1) == columns[2]);
    assert(original.get_col(1).empty());
    assert_csc_invariants(grow);
    grow.col_op(1, 1);
    assert(grow.get_col(1).empty());
    assert_csc_invariants(grow);

    auto permuted = original;
    permuted.permute_columns(vec<int>{3, 2, 0, 1});
    assert(permuted.get_col(0).empty());
    assert(permuted.get_col(1) == columns[2]);
    assert(permuted.get_col(2) == columns[0]);
    assert_csc_invariants(permuted);
    assert_same(SparseMatrix<int>(4, 5, columns), original);

    CSCMatrix<int> unsorted(3, 3, array<int>{{2}, {1}, {0, 2}});
    unsorted.append_entry(0, 0);
    assert(unsorted.get_col(0) == (vec<int>{2, 0}));
    unsorted.sort_data();
    assert(unsorted.is_sorted_sparse());
    assert(unsorted.get_col(0) == (vec<int>{0, 2}));
    assert_csc_invariants(unsorted);
    DenseMatrix identity(3, "Identity");
    auto before = unsorted;
    unsorted.multiply_dense(identity);
    assert_same(before, unsorted);

    for (const auto dimensions : {std::pair<int, int>{0, 0}, {3, 0}, {0, 4}, {4, 3}}) {
        const int cols = dimensions.first;
        const int rows = dimensions.second;
        CSCMatrix<int> empty(cols, rows, array<int>(cols));
        assert_csc_invariants(empty);
        assert(empty.is_zero());
        auto transposed = empty.transposed_copy();
        assert(transposed.get_num_rows() == cols);
        assert(transposed.get_num_cols() == rows);
        assert_csc_invariants(transposed);
        assert_same(empty, transposed.transposed_copy());
    }

    for (const auto& invalid : vec<vec<std::size_t>>{{}, {1}, {0, 2, 1}, {0, 3}}) {
        bool rejected = false;
        try { CSCStorage<int> storage(invalid, vec<int>{0}); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
}

int main() {
    test_shared_algorithms();
    test_copy_on_write_and_boundaries();
}
