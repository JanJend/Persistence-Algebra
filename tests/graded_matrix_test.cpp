

#include "grlina/graded_linalg.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace graded_linalg; 

/**
 * @brief This function demonstrates the basic functionality needed to work with Graded Matrices.
 * 
 */
void functionality_demo() {
    std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path toy_example_path = current_path / "../test_presentations/toy_example_2.scc";
    std::filesystem::path full_rips_path = current_path / "../test_presentations/full_rips_size_1_instance_5_min_pres.scc";
    std::filesystem::path non_example_path = current_path / "../test_presentations/non_example.scc";


    std::cout << "  Basic functionality of R2GradedSparseMatrix: \n"

    // Loading and printing matrices:
    "   We can load them from files: " << std::endl;
    R2GradedSparseMatrix<int> M(toy_example_path.string());
    R2GradedSparseMatrix<int> N(full_rips_path.string());
    R2GradedSparseMatrix<int> O(non_example_path.string());

    std::cout << "  .. and print them with the degrees: " << std::endl;
    M.print_graded();
    std::cout << "  Next one is larger, so we dont print the degrees:" << std::endl;
    N.print();
    std::cout << "  And a small example" << std::endl;
    O.print_graded();
    std::cout << "  We can copy it to a dense matrix, but lose the degrees:" << std::endl;
    DenseMatrix O_dense = from_sparse(O);
    O_dense.print();

    // Creating graded matrices:
    std::cout << "  We can create a graded matrix from scratch: \n The data type of the degrees is a pair of doubles, r2degree = std::pair<double, double>"
        << std::endl;
    array<int> test_data = {{0}, {0, 1}, {2}};
    vec<r2degree> col_degrees = {{1.0, 2.0}, {2.0, 1.0}, {2.0, 2.0}};
    vec<r2degree> row_degrees = {{1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}};

    R2GradedSparseMatrix<int> P(3, 3);
    P.data = test_data;
    P.col_degrees = col_degrees;
    P.row_degrees = row_degrees;
    std::cout << "   or use the constructor with degrees: " << std::endl;
    R2GradedSparseMatrix<int> Pa(3, 3, col_degrees, row_degrees);
    Pa.data = {{0}, {0, 1}, {2}};
    std::cout << "   or use the constructor with degrees and the data: " << std::endl;
    R2GradedSparseMatrix<int> Pb(3, 3, test_data, col_degrees, row_degrees);
    
    std::cout << " Create diagonal matrices by passing a vector of row indices: " << std::endl;
    vec<int> row_indices = {0, 3, 4};
    R2GradedSparseMatrix<int> diag(5, row_indices);
    diag.col_degrees = {{1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}};
    diag.row_degrees = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}, {1.0, 1.0}};
    diag.print_graded();

    std::cout << "  Are these matrices actually graded? " << std::endl;
    std::cout << "  M is graded: " << M.is_graded_matrix() << std::endl;
    std::cout << "  N is graded: " << N.is_graded_matrix() << std::endl;
    std::cout << "  O is not graded: " << O.is_graded_matrix() << std::endl;
    std::cout << "  We can make it graded by deleting the violating entries: " << std::endl;
    O.make_graded();
    O.print_graded();

    std::cout << "  We can sort the columns by degree:" << std::endl;
    O.sort_columns_lexicographically();
    O.print_graded();
    std::cout << "  And sort the rows (Notice the swaps):" << std::endl;
    O.sort_rows_lexicographically();
    O.print_graded();

    std::cout << "  We can ask if an operation is admissible:" << std::endl;
    std::cout << "  Can we add the first to the second column?" << ( O.is_admissible_column_operation(0, 1) ? "Yes" : "No") << std::endl;
    if(O.is_admissible_column_operation(0, 1)){
        O.col_op(0, 1);
    }
    O.print_graded();
    std::cout << "  We can also ask if a row operation is admissible:" << std::endl;
    std::cout << "  Can we add the first to the second row?" << ( O.is_admissible_row_operation(0, 1) ? " Yes" : " No") << std::endl;
    if(O.is_admissible_row_operation(0, 1)){
        // This shouldnt happen, because degrees dont match
        O.row_op_with_update(0, 1);
    }

    std::cout << "  Write the matrix to a file when we're done: " << std::endl;
    std::filesystem::path output_path = current_path / "../test_presentations/non_example_rectified.scc";
    std::ofstream output_file(output_path.string());
    O.to_stream(output_file);
    output_file.close();  

    std::cout << "  Get the Hasse diagram of the row degrees / generators" << std::endl;
    array<int> generator_graph = O.get_row_graph();
    print_edge_list(generator_graph);
    std::cout << "  .. of the row degrees / relations" << std::endl;
        array<int> relation_graph = O.get_column_graph();
    print_edge_list(relation_graph);
    std::cout << "  .. or of both" << std::endl;
    array<int> full_graph = O.get_support_graph();
    print_edge_list(full_graph);

    std::cout << "  Transform this to a boost graph:" << std::endl;
    Graph G = boost_graph_from_edge_list(full_graph);
    print_graph(G);

}

/**
 * @brief Demonstrates algebraic manipulation of graded matrices.
 * 
 */
void algebraic_functionality_demo() {
     std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path toy_example_path = current_path / "../test_presentations/toy_example_2.scc";
    std::filesystem::path full_rips_path = current_path / "../test_presentations/full_rips_size_1_instance_5_min_pres.scc";
    

    std::cout << "  Algebraic functionality of Graded Matrices: \n"
    "   Load the same matrices: " << std::endl;
    R2GradedSparseMatrix<int> M(toy_example_path.string());
    R2GradedSparseMatrix<int> N(full_rips_path.string());

    // Minimality:

    std::cout << "  Are they minimal?" << std::endl;
    std::cout << "  M is not minimal: " << M.is_minimal() << std::endl;
    std::cout << "  N is minimal: " << N.is_minimal() << std::endl;
    M.print_graded();
    std::cout << " We can minimise them, but the algorithm only works if the columns are sorted:" << std::endl;
    M.sort_columns_lexicographically();
    M.print_graded();
    M.minimize();
    M.print_graded();
    std::cout << "  M is now minimal: " << M.is_minimal() << std::endl;
    // Vector space at a concrete degree:

    std::cout << "  We can compute the presentation at any degree:" << std::endl;
    std::cout << "  By this we mean a presentation of the vector space at this degree \n "
    "I.e. a non-graded matrix whose cokernel is that vector space." << std::endl;
    r2degree deg = {0.45, -0.15};
    // auto P = N.map_at_degree(deg);
    auto [P, rows] = N.map_at_degree_pair(deg);
    std::cout << "  The presentation at degree " << deg << " is: " << std::endl;
    P.print();
    std::cout << "  The set of generators at this degree is given by:" << std::endl;
    std::cout <<  rows << std::endl;
    std::cout << " This matrix is not normalised, in so far as its entries are in the range given by the generators." << std::endl;
    std::cout << " To compute its cokernel we need to normalise it, which can be done directly by adding <true> to the map_at_degree function." << std::endl;
    auto [Q, rows_new] = N.map_at_degree_pair(deg, true);
    Q.print();
    std::cout << "Or normalise directly:" << std::endl;
    auto P_normalised = P;
    P_normalised.compute_normalisation(rows);
    P_normalised.print();
    std::cout << "  rows_new still stores the original generators, so that we do not lose this information" << std::endl;
    std::cout <<  rows_new << std::endl;

    std::cout << "  Then the dimension of the presented module is given by a basis for the cokernel at this point:" << std::endl;
    vec<int> basis = Q.coKernel_basis();
    std::cout << "  The dimension is: " << basis.size() <<  std::endl;

    std::cout << "  Compare this with computing this for the non-normalised matrix:" << std::endl;
    vec<int> basis2 = P.coKernel_basis();
    std::cout << "  The dimension is: " << basis2.size() << ", which is clearly different, so this computation is wrong if the matrix-entries are not normalised" << std::endl;

    // Kernel computation:

    std::cout << " We can compute the kernel of any graded matrix-  which must be free over R^2 by Hilbert's syzygy thm - using the Lesnick, Wright, Kerber, Rolle Algorithm:" << std::endl;
    auto K = M.graded_kernel();
    std::cout << "  The kernel is:" << std::endl;
    K.print_graded();
    std::cout << "  The result should be minimal again, with zero kernel itself:" << std::endl;
    std::cout << "  Is the kernel minimal? " << K.is_minimal() << std::endl;
    auto Kkernel = K.graded_kernel();
    std::cout << "  is its kernel zero? " << Kkernel.is_zero() << std::endl;

    // Shift:

    std::cout << "  We can shift all columns and rows by a given degree:" << std::endl;
    K.shift({0.5, 0.5});
    K.print_graded();

}   


void test_reordering(){
    const size_t num_rows = 3;
    const size_t num_cols = 3;
    // Initialize an R2GradedSparseMatrix with the specified dimensions
    R2GradedSparseMatrix<int> M(num_cols, num_rows);
    
    M.col_degrees = {
        {2.5, 3}, {3, 2}, {2, 1}  // Example column degrees
    };

    M.row_degrees = {
        {2, 1}, {1.5 , 1.7}, {2.5, 0}  // Example row degrees
    };

    M.data = {
        {0, 1}, {0, 2}, {0} // Example data
    };

    assert(M.is_graded_matrix());
    M.print_graded();
    vec<int> permutation = M.sort_rows_lexicographically_with_output();
    M.print_graded();
    M.permute_rows_graded(permutation);
    M.print_graded();
}

void test_grid(){
    // Create a 4x5 matrix with float degrees
    const size_t num_rows = 4;
    const size_t num_cols = 5;

    // Initialize an R2GradedSparseMatrix with the specified dimensions
    R2GradedSparseMatrix<int> M(num_cols, num_rows);
    
    M.col_degrees = {
        {1.5 , 2}, {2, 1.7}, {2, 2}, {3, 3}, {3, 3}  // Example column degrees
    };

    M.row_degrees = {
        {2, 0}, {1.5 , 1.7}, {0, 2}, { 1.5 , 3}  // Example row degrees
    };

    M.data = {
        {1, 2}, {0, 1}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}  // Example data
    };

    M.print_graded();

    std::cout << "Sorting lexicographically" << std::endl;
    M.sort_columns_lexicographically();
    M.sort_rows_lexicographically();

    M.print_graded();

    vec<int> permutation = M.compute_grid_representation();

    M.print_grid();

    M.print_grid_representation();

    M.initialise_grid_scheduler();
    // Additional checks could be added here to verify kernel correctness
    std::cout << "The priority queue has size " << M.grid_scheduler.size() << std::endl;

}

void test_kernel(){
    R2GradedSparseMatrix<int> matrix(2, 1);
    matrix.col_degrees = {{1.3, 5}, {4, 1.7}};
    matrix.row_degrees = {{1, 1}};
    matrix.data = {{0}, {0}};
    matrix.print_graded();
    auto K = matrix.graded_kernel();
    K.print_graded();

    std::string path1 = "Persistence-Algebra/test_presentations/toy_example_1.scc";
    std::string path2 = "Persistence-Algebra/test_presentations/full_rips_size_1_instance_5_min_pres.scc";
    std::string path3 = "Persistence-Algebra/test_presentations/toy_example_2.scc";

    vec<std::string> paths = {path1, path2, path3};
    for(auto path : paths){
        R2GradedSparseMatrix<int> matrix_from_file(path);
        if(!matrix_from_file.is_graded_matrix()){
            std::cerr << "Test Matrix is not a valid graded matrix. \n"  <<
            "We're making it graded by deleting the violating entries." << std::endl;
            matrix_from_file.make_graded();
        }
        matrix_from_file.print_graded();
        auto matrix_copy = matrix_from_file;
        auto K_real_data = matrix_copy.graded_kernel();
        assert(K.is_graded_matrix());
        std::cout << "The kernel is" << std::endl;
        K_real_data.print_graded();
        auto product = matrix_from_file.multiply_right(K_real_data);
        if(!product.is_zero()){
            std::cerr << "The product is not zero!" << std::endl;
        }
    }
}

void test_submodule_generated_at(){
    std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path example_path = current_path / "../test_presentations/full_rips_size_1_instance_5_min_pres.scc";
    std::filesystem::path example_path2 = current_path / "../test_presentations/toy_example_3.scc";
    std::filesystem::path example_path3 = current_path / "../test_presentations/multi_cover_050_10_1_min_pres.scc";
    
    
    R2GradedSparseMatrix<int> M(example_path3.string());

    vec<r2degree> support = M.discrete_support();
    r2degree average = {0, 0};
    r2degree specific = {0.333631, -0.0675645};
    for(auto d : support){
        average.first += d.first;
        average.second += d.second;
    }
    average.first /= 2*support.size();
    average.second /= 2*support.size();
    std::cout << "The half average degree is: " << average << std::endl;
    vec<int> basislift = M.basislift_at(average);
    auto M_induced_a = M.submodule_generated_at(average);
    std::cout << "The dimension of the submodule generated at " << average << " is: "
     << basislift.size() << " and the following generators map to a basis \n" << basislift << std::endl;

    // assert(M_induced_a.get_num_rows() == dim_comparison);
    // std::cout << "The submodule generated here is presented by: " << std::endl;
    M_induced_a.print_graded();
}

int main() {
    functionality_demo();
    algebraic_functionality_demo();
    return 0;
}
