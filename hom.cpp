#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_hom_space(std::filesystem::path input_path_A, std::filesystem::path input_path_B, int type,  bool compare = false) {
    
    using aida_result = std::pair< SparseMatrix<int>, vec<std::pair<int,int>> >;

    R2GradedSparseMatrix<int> A = R2GradedSparseMatrix<int>(input_path_A .string());
    R2GradedSparseMatrix<int> B = R2GradedSparseMatrix<int>(input_path_B .string());

    A.sort_columns_lexicographically();
    B.sort_columns_lexicographically();
    A.sort_rows_lexicographically();
    B.sort_rows_lexicographically();

    A.compute_rows_forward();

    aida_result hom_space_full;
    aida_result hom_space_semi_restriction;
    aida_result hom_space_full_rest;
    vec<R2GradedSparseMatrix<int>> hom_space_new;

    // Full linear system
    if(type == 0|| compare){
        hom_space_new = hom_space_basis_new<r2degree, int, R2GradedSparseMatrix<int>>(A,B, true);
        std::cout << "After hom-exact, dim hom: " << hom_space_new.size() << std::endl;
    }
    if(type == 1 || compare){
        hom_space_full = hom_space_no_opt<r2degree, int, R2GradedSparseMatrix<int>>(A, B);
        std::cout << "After full linear system, dim hom: " << hom_space_full.first.get_num_cols() << std::endl;
    }  
    if(type == 2 || compare){
        hom_space_semi_restriction = hom_space_optimised<r2degree, int, R2GradedSparseMatrix<int>>(A,B);
        std::cout << "After semi restriction, dim hom: " << hom_space_semi_restriction.first.get_num_cols() << std::endl;
    } 
    if(type == 3 || compare){
        hom_space_full_rest = hom_space_full_restriction<r2degree, int, R2GradedSparseMatrix<int>>(A,B);
        std::cout << "After full restriction, dim hom: " << hom_space_full_rest.first.get_num_cols() << std::endl;
    } 
    
        
}


int main(int argc, char** argv) {
    
    std::string filepath_A;
    std::string filepath_B;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path_A> <file_path_B>" << std::endl;
        filepath_A = "/home/wsljan/MP-Workspace/Persistence-Algebra/test_presentations/toy_example_5_cut_A.scc";
        filepath_B = "/home/wsljan/MP-Workspace/Persistence-Algebra/test_presentations/toy_example_5_cut.scc";
    } else {
        filepath_A = argv[1];
        filepath_B = argv[2];
    }

    std::filesystem::path input_path_A(filepath_A);
    std::filesystem::path input_path_B(filepath_B);
    
    compute_hom_space(input_path_A, input_path_B, 0, true);

    return 0;
} // main