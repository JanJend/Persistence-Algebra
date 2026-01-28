#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>
#include <boost/timer/timer.hpp>


using namespace graded_linalg;


int compute_hom_space(R2GradedSparseMatrix<int> A, R2GradedSparseMatrix<int> B, int type, bool info = false) {
    
    using aida_result = std::pair< SparseMatrix<int>, vec<std::pair<int,int>> >;

    int result_full;
    int result_A;
    int result_B;
    int result_semi;

    A.sort_columns_lexicographically();
    B.sort_columns_lexicographically();
    A.sort_rows_lexicographically();
    B.sort_rows_lexicographically();

    A.compute_rows_forward();

    aida_result hom_space_full;
    aida_result hom_space_semi_restriction;
    aida_result hom_space_full_rest;
    SparseMatrix<int> hom_space_new;

    // Algorithm B
    if(type == 0|| type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_new = Alg_B_test<r2degree, int, R2GradedSparseMatrix<int>>(A,B, info);
        std::cout << "  Alg B time: " << (timer.elapsed().wall * 1e-6) << "ms" << std::endl;
        result_B = hom_space_new.get_num_cols();
    }
    // Full linear system
    if(type == 1 || type == -1){
        boost::timer::cpu_timer timer;
        hom_space_full = hom_space_no_opt<r2degree, int, R2GradedSparseMatrix<int>>(A, B, false, vec<int>(), vec<int>(), info);
        std::cout << "  Naive time: " << (timer.elapsed().wall * 1e-6) << "ms" <<  std::endl;
        result_full = hom_space_full.first.get_num_cols();
    } else if ( (type == -2) || (type == -3)){
         vec<int> system_size = no_opt_system_info<r2degree, int, R2GradedSparseMatrix<int>>(A, B);
    }
    // Semi-restriction
    if(type == 2 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_semi_restriction = hom_space_optimised<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        std::cout << "  Mixed time: " << (timer.elapsed().wall * 1e-6) << "ms" << std::endl;
        result_semi = hom_space_semi_restriction.first.get_num_cols();
    }
    // Algorithm A
    if(type == 3 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_full_rest = hom_space_full_restriction<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        std::cout << "  Alg A time: " << (timer.elapsed().wall * 1e-6) << "ms" <<std::endl;
        result_A = hom_space_full_rest.first.get_num_cols();
    }
    
    if(type == -2 || type == -1){
        if(result_A != result_B || result_A != result_semi){
            std::cerr << "Warning: Different results for different methods: " 
                      << "Mixed: " << result_semi 
                      << ", Alg A: " << result_A << ", Alg B: " << result_B << std::endl;
        } else {
        }
        if( type == -1){
            if(result_full != result_A){
                std::cerr << "Warning: Full linear system yields different result: " << result_full << std::endl;
            }
        }
        return result_B;
    } else {
        if(type == 0){
            return result_B;
        } else if(type == 1){
            return result_full;
        } else if(type == 2){
            return result_semi;
        } else if(type == 3){
            return result_A;
        } else {
            std::cerr << "Error: Unknown type " << type << std::endl;
            return -1;
        }
    }
 
}

int main(int argc, char** argv) {
    
    std::string filepath_A;
    std::string filepath_B;

    int type = 0;
    bool info = false;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path_A> <file_path_B> <type> <info>" << std::endl;
        if(true){
            filepath_A = "/home/wsljan/MP-Workspace/data/1.5mmRegions/CD8_scc/T_A_ROI_10_locations_CD8_H1.scc";
            filepath_B = "/home/wsljan/MP-Workspace/data/1.5mmRegions/CD8_scc/T_A_ROI_10_locations_CD8_H1.scc";
            type = 1;
            info = true;
        } else {
            return 1;
        }
    } else {
        filepath_A = argv[1];
        filepath_B = argv[2];
        if( argc > 4){
            info = std::stoi(argv[4]) != 0;
        }
        if( argc > 3){
            type = std::stoi(argv[3]);
        }
    }

    std::filesystem::path input_path_A(filepath_A);
    std::filesystem::path input_path_B(filepath_B);
    
    R2GradedSparseMatrix<int> A(input_path_A.string());
    R2GradedSparseMatrix<int> B(input_path_B.string());

    int dim = compute_hom_space(A, B, type, info);
    std::cout << "Dimension: " << dim << std::endl;

    return 0;
} // main