#include <grlina/csc_matrix.hpp>

// Compiled in several modes by csc_warnings.cmake. Read-only code must remain
// warning-free even when deprecated declarations are treated as errors.
int main() {
    using namespace graded_linalg;
    CSCMatrix<int> matrix(2, 3, array<int>{{0, 2}, {1}});
    const auto snapshot = matrix;
    if (!matrix.data.shares_storage_with(snapshot.data)) return 1;
    if (snapshot.col_last(0) != 2 || snapshot.get_col(1) != vec<int>{1}) return 2;
#if CSC_WARNING_TEST_MUTATE == 1
    matrix.col_op(0, 1);
    matrix.col_op(0, 1);
    if (matrix.data.shares_storage_with(snapshot.data)) return 3;
    if (matrix.get_col(1) != vec<int>{1} || snapshot.get_col(1) != vec<int>{1}) return 4;
#elif CSC_WARNING_TEST_MUTATE == 2
    // Appending is normally cheap, but a copied matrix must detach its buffers.
    matrix.append_col(vec<int>{2});
    matrix.compute_num_cols();
    if (matrix.data.shares_storage_with(snapshot.data)) return 5;
    if (matrix.get_num_cols() != 3 || snapshot.get_num_cols() != 2) return 6;
#endif
}
