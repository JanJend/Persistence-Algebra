#include <grlina/mat_upset.hpp>
#include <cassert>
#include <iostream>

using namespace graded_linalg;
using Matrix = UpsetGradedSparseMatrix<double, int>;

static void check_complex(const ChainComplex<Matrix>& complex) {
    assert(complex.is_chain_complex());
    for (const auto& matrix : complex.differentials()) assert(matrix.is_graded_matrix());
    for (std::size_t i = 1; i < complex.size(); ++i) {
        assert(complex[i - 1].get_num_cols() == complex[i].get_num_rows());
        assert(complex[i - 1].col_degrees == complex[i].row_degrees);
    }
}

int main(int argc, char* argv[]) {
    const char* path = argc > 1 ? argv[1]
        : "../multicritical/sample_files/compressed_deg_rips_100_points.scc";
    auto complex = ChainComplex<Matrix>::from_file(path);
    assert(complex.size() == 3);
    assert(complex[0].get_num_rows() == 100);
    assert(complex[0].get_num_cols() == 483);
    assert(complex[1].get_num_cols() == 995);
    assert(complex[2].get_num_cols() == 1128);
    check_complex(complex);

    complex.minimize();
    check_complex(complex);
    for (const auto& matrix : complex.differentials()) {
        assert(matrix.compatibly_sorted);
        for (int column = 0; column < matrix.get_num_cols(); ++column)
            for (int row : matrix.get_col(column))
                assert(!Degree_traits<r2upset<double>>::equals(
                    matrix.col_degrees[column], matrix.row_degrees[row]));
    }

    std::cout << "Minimized chain-group dimensions (C0 through C3): "
              << complex[0].get_num_rows();
    for (const auto& matrix : complex.differentials()) std::cout << ' ' << matrix.get_num_cols();
    std::cout << '\n';
}
