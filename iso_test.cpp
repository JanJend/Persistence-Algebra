#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>
#include <boost/timer/timer.hpp>


using namespace graded_linalg;


int main(int argc, char** argv) {
    
    std::string filepath_A;
    std::string filepath_B;

    int type = 0;
    bool info = false;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path_A> <file_path_B> <type> <info>" << std::endl;
        if(false){
            filepath_A = "";
            filepath_B = "";
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
    if(info){
        std::cout << "Dimensions of A: " << A.get_num_rows() << " x " << A.get_num_cols() << std::endl;
        std::cout << "Dimensions of B: " << B.get_num_rows() << " x " << B.get_num_cols() << std::endl;
    }
    if(is_isomorphic<R2GradedSparseMatrix<int>>(A,B)){
        std::cout << "The modules at " << input_path_A << " and " << input_path_A << " are isomorphic." << std::endl;
        return 1;
    } else {
        std::cout << "The modules at " << input_path_A << " and " << input_path_A << " are not isomorphic." << std::endl;
        return 0;
    }
    return 1;
} // main
