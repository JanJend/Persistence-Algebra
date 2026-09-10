#include "grlina/graded_linalg.hpp"
#include "grlina/matrix_base.hpp"
#include "grlina/r2graded_matrix.hpp"
#include "grlina/sparse_matrix.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace graded_linalg; 
using Mat = R2GradedSparseMatrix<int>;

void hom_space_test(){
    const auto test_directory = std::filesystem::path(__FILE__).parent_path();
    std::string path1 = (test_directory / "../test_presentations/hom_test_domain.scc").string();
    std::string path2 = (test_directory / "../test_presentations/hom_test_target.scc").string();
    Mat domain = Mat(path1);
    Mat target = Mat(path2);
    domain.print_graded();
    target.print_graded();
    domain.compute_rows_forward();
    auto result1 = hom_space_basis(domain, target, false);
    auto result2 = hom_space_basis(domain, target, true);
    
    
    auto non_zero1 = general_reduction<int, Mat>(result1);
    auto non_zero2 = general_reduction<int, Mat>(result2);
    assert(non_zero2.size() == result2.size());

    // Reduce first hom space.
    auto result1_basis = vec_restriction<int, Mat>(result1, non_zero1);
    assert(result1_basis.size() == result2.size());
    
}

int main() {
    hom_space_test();
    return 0;
}
