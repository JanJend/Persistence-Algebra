

#include "grlina/graded_linalg.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace graded_linalg; 


void functionality_demo() {
    std::filesystem::path current_path = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path toy_example_path = current_path / "../test_presentations/toy_example_2.scc";
    std::filesystem::path full_rips_path = current_path / "../test_presentations/full_rips_size_1_instance_5_min_pres.scc";
    std::filesystem::path non_example_path = current_path / "../test_presentations/non_example.scc";

    std::cout << "  Basic functionality of R2GradedSparseMatrix: \n"
    "   We can load them from files: " << std::endl;
    R2GradedSparseMatrix<int> M(toy_example_path.string());
    R2GradedSparseMatrix<int> N(full_rips_path.string());
    R2GradedSparseMatrix<int> O(non_example_path.string());
    std::cout << "  We can print them with the degrees: " << std::endl;
    M.print_graded();
    std::cout << "  Next one is larger, so we dont print the degrees:" << std::endl;
    N.print();
    std::cout << "  And a small example" << std::endl;
    O.print_graded();
    std::cout << "  We can copy it to a dense matrix, but lose the degrees:" << std::endl;
    DenseMatrix O_dense = from_sparse(O);
    O_dense.print();

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

    std::cout << "  We can compute the presentation at any degree:" << std::endl;
    r2degree deg = {0.45, -0.15};
    // auto P = N.map_at_degree(deg);
    auto [P, rows] = N.map_at_degree_pair(deg);
    std::cout << "  The presentation at degree " << deg << " is: " << std::endl;
    P.print();
    std::cout << "  or shift indices directly to normalise the entries:" << std::endl;
    auto [Q, rows_new] = N.map_at_degree_pair(deg, true);
    Q.print();

    std::cout << "  Then the dimension of the presented module is given by a basis for the cokernel at this point:" << std::endl;
    vec<int> basis = Q.coKernel_basis();
    std::cout << "  The dimension is: " << basis.size() << std::endl;
    /** 
     * Bug in the code, will fix asap.
    std::cout << "At last, compute the Hom-space between two graded matrices\n"
     "(Warning this is the version optimised for pointwise rather flat modules):" << std::endl;
    Hom_space_temp<int> End_N = hom_space(N, N);
    SparseMatrix<int> hom_matrix = End_N.first;
    vec<pair<int>> index_map = End_N.second;

    std::cout << "The hom_matrix stores every matrix as a sparse vector. \n"
    "For every entry i in such a vector, index_map[i] stores the indices of the corresponding row operation" << std::endl;
    std::cout << "The dimension of End(N, N) is: " << hom_matrix.get_num_cols() << std::endl;
    */

}

int main() {
    functionality_demo();
    return 0;
}
