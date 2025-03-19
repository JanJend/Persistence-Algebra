

#include "grlina/r2graded_matrix.hpp"  
#include "grlina/r3graded_matrix.hpp"  
#include <iostream>
#include <cassert>

using namespace graded_linalg; 


void test_order() {
    R2GradedSparseMatrix<int> M("/home/wsljan/AIDA/test_presentations/toy_example_1.scc");
    M.print_graded();
    M.data[1].push_back(2);
    M.print_graded();
}

int main() {
    test_order();
    return 0;
}
