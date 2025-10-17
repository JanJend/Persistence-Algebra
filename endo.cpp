#include "grlina/homomorphisms.hpp"
#include "grlina/r2graded_matrix.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void endomorphism_sizes(std::filesystem::path input_path) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    std::cout << presentation.get_num_rows() << " x " << presentation.get_num_cols() << std::endl;
    for(int i = 0; i < 5; ++i) {
        double eps = 0.005 * i;
        auto pres_shift = presentation;
        presentation.sort_columns_lexicographically();
        presentation.sort_rows_lexicographically();
        pres_shift.shift({eps, eps});
        presentation.compute_rows_forward();
        auto endos = hom_space_basis(presentation, pres_shift);
        std::cout << "Epsilon: " << eps << " Number of endomorphisms: " << endos.size() << std::endl;
    }
}


int main(int argc, char** argv) {
    
    std::string filepath;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        filepath = "/home/wsljan/AIDA/Persistence-Algebra/test_presentations/points_wo_density_20_dim2_k_fold_10_min_pres.scc";
    } else {
        filepath = argv[1];
    }

    std::filesystem::path input_path(filepath);

    endomorphism_sizes(input_path);
    

    
    
    return 0;
} // main