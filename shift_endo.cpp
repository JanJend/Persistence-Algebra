#include "grlina/homomorphisms.hpp"
#include "grlina/r2graded_matrix.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void endomorphism_sizes(std::filesystem::path input_path) {
    auto module = std::make_shared<R2Module<int>>(input_path.string());
    module->sort_compatibly();
    std::cout << module->number_of_generators() << " x " << module->number_of_relations() << std::endl;
    for(int i = 0; i < 5; ++i) {
        double eps = 0.005 * i;
        auto shifted = std::make_shared<R2Module<int>>(*module);
        shifted->shift({eps, eps});
        auto endos = module_hom_space_basis< R2GradedSparseMatrix<int> >(
            module, shifted, true);
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
