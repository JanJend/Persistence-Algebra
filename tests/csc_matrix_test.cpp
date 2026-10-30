// These tests deliberately exercise expensive operations; warning behavior is
// tested separately so this differential suite can run without diagnostic noise.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/matrices/sparse.hpp>
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

template<class Storage>
void test_identity_shortcuts() {
    using Matrix = SparseMatrix<int, Storage>;
    auto check = [](Matrix left, Matrix right) {
        const auto actual = left * right;
        left.is_identity = right.is_identity = false;
        const auto expected = left * right;
        assert(actual.get_num_rows() == expected.get_num_rows());
        assert(actual.get_num_cols() == expected.get_num_cols());
        for (int i = 0; i < actual.get_num_cols(); ++i)
            assert(actual.get_col(i) == expected.get_col(i));
    };
    Matrix A(3, 2, {{0}, {1}, {0, 1}});
    Matrix I2(2, 2, "Identity"), I3(3, 3, "Identity");
    assert(I2.is_identity && I3.is_identity);
    check(I2, A);
    check(A, I3);
    if constexpr (std::is_same_v<Storage, CSCStorage<int>>) {
        assert((I2 * A).data.shares_storage_with(A.data));
        assert((A * I3).data.shares_storage_with(A.data));
    }
    Matrix inclusion(5, vec<int>{1, 3});
    assert(inclusion.is_identity);
    A.compute_rows_forward();
    const auto embedded = inclusion * A;
    assert(embedded.rows_computed && embedded._rows == A._rows && embedded.get_num_rows() == 5);
    assert((embedded.get_col(2) == vec<int>{1, 3}));
    check(inclusion, A);
    vec<int> removed{1};
    I3.delete_columns(removed);
    assert(I3.is_identity); // The retained unit columns are an ordered inclusion.
    check(A, I3);
    Matrix outer(7, vec<int>{0, 2, 3, 5, 6});
    const auto composite = outer * inclusion;
    assert(composite.is_identity);
    assert((composite.get_col(0) == vec<int>{2}) && (composite.get_col(1) == vec<int>{5}));
    check(outer, inclusion);
    Matrix empty_inclusion(3, vec<int>{});
    check(A, empty_inclusion);
    check(Matrix(0, 0, "Identity"), Matrix(2, 0, array<int>{{}, {}}));
    check(Matrix(3, vec<int>{2, 0}), Matrix(1, 2, array<int>{{0, 1}})); // Unordered: ordinary product.
    check(Matrix(3, vec<int>{1, 1}), Matrix(1, 2, array<int>{{0, 1}})); // Repeated images cancel over F2.
    Matrix changed = I2;
    changed.col_op(0, 1);
    assert(!changed.is_identity);
    check(changed, A);
    changed = I2;
    assert(changed.is_identity);
    changed.set_col(0, vec<int>{1});
    assert(!changed.is_identity);
    changed = I2;
    changed.assign_data(I2.data);
    assert(!changed.is_identity);
    Matrix moved(std::move(I2));
    assert(moved.is_identity && !I2.is_identity);
}

template<class Storage>
void test_sparse_products() {
    using Matrix = SparseMatrix<int, Storage>;
    std::mt19937 random(20261029);
    for (int trial = 0; trial < 100; ++trial) {
        const int rows = trial % 11, inner = (trial / 11) % 7, cols = random() % 10;
        array<int> a(inner), b(cols);
        const unsigned density = trial % 4; // Includes zero, sparse, and dense products.
        for (auto& column : a)
            for (int row = 0; row < rows; ++row)
                if (random() % 3 < density) column.push_back(row);
        for (auto& column : b)
            for (int row = 0; row < inner; ++row)
                if (random() % 3 < density) column.push_back(row);
        const Matrix left(inner, rows, a), right(cols, inner, b);
        const auto product = left * right;
        assert(multiply_transpose(left.transposed_copy(), right).data == product.data);
        auto legacy_left = left.transposed_copy(), legacy_right = right;
        legacy_left.set_num_rows(0);
        legacy_right.set_num_rows(0);
        assert(multiply_transpose(legacy_left, legacy_right).data == product.data);
        static_assert(std::is_same_v<std::decay_t<decltype(product)>, Matrix>);
        assert(product.get_num_rows() == rows && product.get_num_cols() == cols);
        // Independent dense dot-product oracle, including cancellation to zero.
        for (int j = 0; j < cols; ++j) {
            vec<int> expected;
            for (int row = 0; row < rows; ++row) {
                bool odd = false;
                for (int k = 0; k < inner; ++k)
                    odd ^= std::binary_search(a[k].begin(), a[k].end(), row) &&
                           std::binary_search(b[j].begin(), b[j].end(), k);
                if (odd) expected.push_back(row);
            }
            assert(product.get_col(j) == expected);
            assert(right.get_col(j) == b[j]);
        }
        for (int k = 0; k < inner; ++k) assert(left.get_col(k) == a[k]);
        if constexpr (std::is_same_v<Storage, CSCStorage<int>>) assert_csc_invariants(product);
    }
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

void test_dense_index_transformation() {
    const vec<int> row_map{-1,0,-1,1,2,-1};
    vec<int> mapped{4,0,3,5,1,2};
    apply_transformation(mapped, row_map);
    assert(mapped == (vec<int>{2,-1,1,-1,0,-1}));
    vec<int> filtered{4,0,3,5,1,2};
    apply_transformation(filtered, row_map, false);
    assert(filtered == (vec<int>{2,1,0}));
}

void test_row_index_maps() {
    for (const vec<int>& removed : array<int>{{}, {0,2,5}, {0,1,2,3,4,5}}) {
        vec<int> kept(6);
        std::iota(kept.begin(), kept.end(), 0);
        const auto buffer = kept.data();
        remove_intersection(kept, removed);
        const auto sparse_map = shiftIndicesMap(kept);
        const auto dense_map = shiftIndicesMap(std::move(kept), 6);
        assert(dense_map.data() == buffer);
        for (int row = 0; row < 6; ++row) {
            const auto it = sparse_map.find(row);
            assert(dense_map[row] == (it == sparse_map.end() ? -1 : it->second));
        }
        vec<int> sparse_result{4,0,3,5,1,2}, dense_result = sparse_result;
        apply_transformation(sparse_result, sparse_map);
        apply_transformation(dense_result, dense_map, false);
        assert(sparse_result == dense_result);
    }
    assert(shiftIndicesMap(vec<int>{}, 0).empty());
    assert(shiftIndicesMap(vec<int>{1,3}, 5) == (vec<int>{-1,0,-1,1,-1}));
    const std::unordered_map<int,int> sparse_map{{1000000,7}, {3,-1}};
    vec<int> filtered{3,2,1000000,3};
    apply_transformation(filtered, sparse_map);
    assert(filtered == (vec<int>{-1,7,-1}));
    vec<int> mapped{1000000,3};
    apply_transformation(mapped, sparse_map, true);
    assert(mapped == (vec<int>{7,-1}));
}

void test_batch_row_deletion() {
    const array<int> columns{{0,2,5}, {}, {1,3,4,5}, {0,1,2,3,4,5}};
    for (const vec<int>& removed : array<int>{{}, {0,2,5}, {0,1,2,3,4,5}}) {
        array<int> expected(columns.size());
        for (std::size_t c = 0; c < columns.size(); ++c)
            for (int row : columns[c]) {
                auto it = std::lower_bound(removed.begin(), removed.end(), row);
                if (it == removed.end() || *it != row)
                    expected[c].push_back(row - static_cast<int>(it - removed.begin()));
            }
        SparseMatrix<int> ordinary(columns.size(), 6, columns);
        CSCMatrix<int> packed(columns.size(), 6, columns);
        ordinary.compute_rows_forward();
        packed.compute_rows_forward();
        const auto stored_rows = ordinary._rows;
        ordinary.delete_rows(removed);
        packed.delete_rows(removed);
        const SparseMatrix<int> reference(columns.size(), 6 - removed.size(), expected);
        assert_same(reference, packed);
        assert(ordinary.data == expected && ordinary.get_num_rows() == reference.get_num_rows());
        assert(ordinary.rows_computed && packed.rows_computed);
        assert(ordinary._rows == stored_rows && packed._rows == stored_rows);
    }
    SparseMatrix<int> empty(0, 6);
    CSCMatrix<int> packed_empty(0, 6);
    empty.delete_rows(vec<int>{0,2,5});
    packed_empty.delete_rows(vec<int>{0,2,5});
    assert(empty.get_num_rows() == 3);
    assert_same(empty, packed_empty);
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
}

void test_borrowed_column_sorting() {
    const CSCMatrix<int> empty(0, 0);
    assert(empty.is_sorted_sparse());
    CSCMatrix<int> matrix(4, 6, array<int>{{}, {3}, {1, 4}, {0, 2, 5}});
    const auto& read_only = matrix;
    const auto column = read_only.column(3);
    assert(&*column.begin() == &read_only.data.entries()[read_only.data.offsets()[3]]);
    assert(read_only.is_sorted_sparse()); // A descent across column boundaries is valid.
    assert(graded_linalg::is_sorted(column));
    matrix.set_col(3, vec<int>{0, 5, 2});
    assert(!read_only.is_sorted_sparse());
    matrix.set_col(3, vec<int>{0, 2, 2});
    assert(!read_only.is_sorted_sparse());
}

void test_offset_check_policy() {
    bool rejected = false;
    try {
        // Invalid buffers are only constructed here, never accessed.
        CSCStorage<int> invalid(vec<std::size_t>{1, 1}, vec<int>{});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected == (GRLINA_ENABLE_CHECKS != 0));
}

int main() {
    test_identity_shortcuts<vec<vec<int>>>();
    test_identity_shortcuts<CSCStorage<int>>();
    test_sparse_products<vec<vec<int>>>();
    test_sparse_products<CSCStorage<int>>();
    test_shared_algorithms();
    test_dense_index_transformation();
    test_row_index_maps();
    test_batch_row_deletion();
    test_copy_on_write_and_boundaries();
    test_borrowed_column_sorting();
    test_offset_check_policy();
}
