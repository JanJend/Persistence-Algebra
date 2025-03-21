
#include "grlina/graded_linalg.hpp"

#include <boost/timer/timer.hpp>

using namespace graded_linalg;
using namespace boost::timer;

void functionality_demo(){
    std::cout << "Basic functionality of SparseMatrix: \n" 
    "When initialising, the format is always (columns, rows)! \n" 

    "You can create the identity matrix:" << std::endl;
    SparseMatrix<int> I(4, 4, "Identity");
    I.print();

    std::cout << "Or a random matrix with log(rows) many entries per column:" << std::endl;
    SparseMatrix<int> R(4, 4, "Random");
    R.print();

    std::cout << "Even with your own (say 80%) fill percentage:" << std::endl;
    SparseMatrix<int> R2(4, 4, "Random", 80);
    R2.print();

    std::cout << "Or part of an identity matrix / a sub set of the standard basis:" << std::endl;
    vec<int> indicator = {0, 2};
    SparseMatrix<int> P(4, indicator);
    P.print();

    std::cout << "Or a matrix A from a list of non-zero entries:" << std::endl;
    vec<vec<int>> data = {{0, 3, 4}, {1, 3, 5}, {0, 2, 3, 4}, {0, 1, 2, 4, 5}};
    SparseMatrix<int> A(4, 6, data);
    vec<vec<int>> data2 = {{0}, {}, {1}, {0,1}, {}, {0,1}};
    SparseMatrix<int> B(6, 2, data2);
    A.print();
    auto A_copy1 = A;

    std::cout << "Convert to a dense matrix:" << std::endl;
    auto dense = from_sparse(A);
    dense.print();

    std::cout << " Perform column operations, which is fast:" << std::endl;
    A_copy1.col_op(0, 1);
    A_copy1.print();

    std::cout << " Or perform row operations, which is slow:" << std::endl;
    A_copy1.row_op_on_cols(0, 1);

    std::cout << " If we want the row operations to be fast, we need to compute the rows first:" << std::endl;
    A_copy1.compute_rows_forward();
    A_copy1.print_rows();
    std::cout << " Now we can perform a row operation fast on the rows, without touching the columns" << std::endl;
    A_copy1.fast_row_op(0, 1);
    A_copy1.print_rows();
    std::cout << " But this does not change the columns yet, so we need to recompute them:" << std::endl;
    A_copy1.compute_columns_from_rows();
    A_copy1.print();
    std::cout << " We can also perform a row operation on the rows and update the columns instantly, if we only do it a couple of times:" << std::endl;
    A_copy1.row_op_with_update(0, 1);
    A_copy1.print();

    std::cout << " Column reduction is fast:" << std::endl;
    A_copy1.column_reduction_triangular();
    A_copy1.print();

    std::cout << " So to do row reduction, we want to transpose and then reduce: " << std::endl;
    auto AT = A.transposed_copy();
    AT.print();
    AT.column_reduction_triangular();
    AT.print();
    auto A_row_reduced = AT.transposed_copy();
    A_row_reduced.print();

    std::cout << "Compute Kernel (Careful, this reduces the matrix, so make copy):" << std::endl;
    auto A_copy2 = A;
    auto K = A_copy2.kernel();
    K.print();

    std::cout << "Compute Cokernel (Also make copy!).\n" 
    "Pass a pointer to an empty vector, if you need a lift of the basis:" << std::endl;
    auto A_copy3 = A;
    vec<int> basis_lift;
    auto coker = A_copy3.coKernel(false, &basis_lift);
    coker.print();

    std::cout << "Check that the coker is really a coker:" << std::endl;
    auto check = coker.multiply_right(A);
    check.print();

    std::cout << "Check that the standard basis vectors given by the basis lift really map to a basis of the quotient:" << std::endl;
    auto basis_lift_matrix = SparseMatrix<int>(6, basis_lift);
    auto induced_basis = coker.multiply_right(basis_lift_matrix);
    induced_basis.print();
    
    std::cout << "Consider another matrix B:" << std::endl;
    B.print();

    std::cout << "Multiplying: B*A:" << std::endl;
    auto C = B.multiply_right(A);
    C.print();

    std::cout << "Consider a square matrix D:" << std::endl;    
    vec<vec<int>> data3 = {{0}, {0, 1}, {2}};
    SparseMatrix<int> D(3, 3, data3);
    D.print();

    std::cout << "Is it invertible?" << std::endl;
    std::cout << "It is: " << ( D.is_invertible() ? "yes" : "no")  << std::endl;
    std::cout << "Compute its inverse:" << std::endl;
    auto D_inv = D.inverse();
    D_inv.print();

    std::cout << "Shift row indices via any map given by a vector:" << std::endl;
    vec<int> row_indices = {2, 6, 7};

    std::cout << "New indices " << row_indices << std::endl;
    D.transform_data(row_indices);
    D.print();

    std::cout << "Maybe this new matrix is a submatrix of a larger one, and we need to restrict to a set of row indices. \n" 
    "say we are interested in {2, 3, 6, 7}. When restricting we need to reindex the columns (now 3 is not hit, so 6 denotes the third entry):" << std::endl;
    vec<int> new_row_indices = {2, 3, 6, 7};
    D.compute_normalisation(new_row_indices);
    D.print();

    std::cout << "Test done. \n " << std::endl;

}


void row_test(){
    vec<vec<int>> data = {{0, 1, 2}, {1, 3}, {0, 2, 3}, {2}};
    SparseMatrix<int> A(4, 4, data);
    std::cout << "Testing fast row operations. Example Matrix is:" << std::endl;
    A.print();
    std::cout << "For this we compute the rows in reverse direction. \n"
     "The idea is that we're only updating the last columns of the matrix very late" << std::endl;
    A.compute_revrows();
    std::cout << "The rows are:" << std::endl;
    A.print_rows();
    std::cout << "Adding 0th to first row. This updates the columns in the fastest possible way by only appending an index" << std::endl;
    A.fast_rev_row_op(0, 1);
    A.print_rows();
    std::cout << "We can see that this did not correctly compute the columns:" << std::endl;
    A.print();
    std::cout << "To rectify, we need to sort and remove duplicates:" << std::endl;
    for(int i = 0; i < A.get_num_cols(); i++){
        A.sort_column(i);
        convert_mod_2(A.data[i]);
    }
    A.print();
}


void compare_vec_set_test(const vec<int>& test_sizes = {2000}){
    std::cout << "Comparing speed of reduction algorithms for sparse vector and set implementation:" << std::endl;
    for(int n : test_sizes){
        cpu_timer vec_timer;
        cpu_timer set_timer;
        auto M = SparseMatrix<int>(n, n, "Random");
        auto N = SparseMatrix_set<int>(M);
        vec_timer.start();
        M.column_reduction_triangular();
        vec_timer.stop();
        std::cout << "Vector time for n = " << n << ": " << vec_timer.format() << std::endl;


        set_timer.start();
        N.column_reduction_triangular();
        set_timer.stop();
        std::cout << "Set time for n = " << n << ": " << set_timer.format() << std::endl;
    }
}

void test_kernel(){
    int n = 4;
    int perc = 20;
    auto M = SparseMatrix<int>(n, n, "Random", perc);
    std::cout << "M is nonzero: " << !M.is_zero() << std::endl;
    M.print();
    auto copy = M;
    auto K = copy.kernel();
    std::cout << "K is nonzero: " << !K.is_zero() << ". Size:" << K.get_num_cols() <<  std::endl;
    K.print();
    auto Zero = M.multiply_right(K);
    assert(Zero.is_zero());
}

int main() {

    functionality_demo();
    row_test();
    compare_vec_set_test();
    test_kernel();

    return 0;
}
