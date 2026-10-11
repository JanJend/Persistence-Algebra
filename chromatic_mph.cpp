#include <grlina/chromatic_mph.hpp>
#include <cstdlib>

int main(int, char** argv) {
    // Load the three-parameter complex using the library's existing SCC/FIREP reader.
    graded_linalg::ChainComplex<graded_linalg::R3Matrix<>> input(argv[1]);
    graded_linalg::compute_chromatic_mph(input, std::atoi(argv[2]));
}
