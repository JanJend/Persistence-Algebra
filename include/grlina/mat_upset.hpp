/** Upset-graded sparse matrices with multicritical's extended SCC I/O. */
#pragma once

#include <grlina/deg_upset.hpp>
#include <grlina/proj_chain_complex.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace graded_linalg {

template <typename Scalar, typename Index, typename Storage = vec<vec<Index>>>
struct UpsetGradedSparseMatrix
    : GradedSparseMatrix<r2upset<Scalar>, Index, UpsetGradedSparseMatrix<Scalar, Index, Storage>, Storage> {
    using Base = GradedSparseMatrix<r2upset<Scalar>, Index, UpsetGradedSparseMatrix, Storage>;
    using Base::Base;

    UpsetGradedSparseMatrix() : Base(0, 0) {}
    UpsetGradedSparseMatrix(Index columns, Index rows) : Base(columns, rows) {
        this->resize_data(columns);
    }
    UpsetGradedSparseMatrix(Index columns, Index rows,
                           vec<r2upset<Scalar>> column_degrees, vec<r2upset<Scalar>> row_degrees)
        : Base(columns, rows, std::move(column_degrees), std::move(row_degrees)) {
        this->resize_data(columns);
    }
    explicit UpsetGradedSparseMatrix(SparseMatrix<Index, Storage>&& matrix) : Base(std::move(matrix)) {}

    /** Read one differential; use ChainComplex for an entire multi-matrix file. */
    explicit UpsetGradedSparseMatrix(std::istream& input, bool lex_sort = false) {
        auto complex = ChainComplex<UpsetGradedSparseMatrix>::from_stream(input, lex_sort);
        if (complex.size() != 1)
            throw std::runtime_error("Expected one upset-graded matrix; use ChainComplex for multiple differentials");
        *this = std::move(complex.differentials().front());
    }

    explicit UpsetGradedSparseMatrix(const std::string& path, bool lex_sort = false) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Unable to open SCC input file: " + path);
        *this = UpsetGradedSparseMatrix(input, lex_sort);
    }

    template <typename Output>
    void to_stream(Output& output, bool header = true) const {
        output << std::defaultfloat << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
        if (header) output << "scc2020\n2\n" << this->get_num_cols() << ' '
                           << this->get_num_rows() << " 0\n";
        for (Index i = 0; i < this->get_num_cols(); ++i) {
            Degree_traits<r2upset<Scalar>>::write_degree(output, this->col_degrees[i]);
            output << " ;";
            for (Index row : this->get_col(i)) output << ' ' << row;
            output << '\n';
        }
        for (const auto& upset : this->row_degrees) {
            Degree_traits<r2upset<Scalar>>::write_degree(output, upset);
            output << " ;\n";
        }
    }

    void to_file(const std::string& path) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Unable to open SCC output file: " + path);
        to_stream(output);
    }
};

// Preserve the scalar's precision when the generic chain-complex writer is used.
template <typename Scalar, typename Index, typename Storage>
struct GradedMatrixIO<UpsetGradedSparseMatrix<Scalar, Index, Storage>>
    : MatrixIOOperations<UpsetGradedSparseMatrix<Scalar, Index, Storage>> {
    static constexpr int output_precision() noexcept { return std::numeric_limits<Scalar>::max_digits10; }
    static UpsetGradedSparseMatrix<Scalar, Index, Storage> make_matrix(Index columns, Index rows) {
        return {columns, rows};
    }
};

} // namespace graded_linalg
