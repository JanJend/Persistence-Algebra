/**
 * @file graded_matrix_io.hpp
 * @brief NEW: SCC metadata adapter for fixed and runtime-dimensional matrices.
 */
#pragma once

#include <cstddef>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <grlina/graded_matrix.hpp>

namespace graded_linalg {

template<class Matrix, class = void>
struct MatrixSerializedDegree { using type = typename Matrix::degree_type; static constexpr bool grid = false; };
template<class Matrix>
struct MatrixSerializedDegree<Matrix, std::void_t<typename Matrix::real_degree_type, decltype(Matrix::grid_backed)>> {
    using type = typename Matrix::real_degree_type; static constexpr bool grid = Matrix::grid_backed;
};

template<class Matrix>
struct MatrixIOOperations {
    using serialized_degree_type = typename MatrixSerializedDegree<Matrix>::type;
    static void assign_degrees(Matrix& matrix, const std::vector<serialized_degree_type>& columns,
                               const std::vector<serialized_degree_type>& rows) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) matrix.set_real_degrees(columns, rows);
        else { matrix.col_degrees = columns; matrix.row_degrees = rows; }
    }
    template<class Output> static void write_column_degree(Output& out, const Matrix& matrix, typename Matrix::index_type i) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) Degree_traits<serialized_degree_type>::write_degree(out, matrix.real_col_degree(i));
        else Degree_traits<serialized_degree_type>::write_degree(out, matrix.col_degrees[i]);
    }
    template<class Output> static void write_row_degree(Output& out, const Matrix& matrix, typename Matrix::index_type i) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) Degree_traits<serialized_degree_type>::write_degree(out, matrix.real_row_degree(i));
        else Degree_traits<serialized_degree_type>::write_degree(out, matrix.row_degrees[i]);
    }
    static bool matching_chain_group(const Matrix& lower, const Matrix& higher) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) return lower.real_col_degrees() == higher.real_row_degrees();
        else return lower.col_degrees == higher.row_degrees;
    }
    static void normalize_complex(std::vector<Matrix>& matrices) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) {
            if (matrices.empty()) return;
            Matrix merged = matrices.front();
            for (auto& matrix : matrices) merged.merge_grids(matrix);
            for (auto& matrix : matrices) matrix.reindex_grid(merged.grids);
        }
    }
};

/** Keep the SCC poset context separate from the owning degree type. */
template <typename Matrix, typename = void>
struct GradedMatrixIO : MatrixIOOperations<Matrix> {
    using degree_type = typename MatrixSerializedDegree<Matrix>::type;
    using index_type = typename Matrix::index_type;
    using context_type = std::size_t;
    static constexpr bool runtime_dimension = false;

    static constexpr int output_precision() noexcept { return 17; }

    static context_type matrix_context(const Matrix&) noexcept { return 0; }

    static std::string identifier(context_type = 0) {
        return std::string(Degree_traits<degree_type>::poset_id);
    }

    static std::string identifier(const Matrix&) { return identifier(); }

    static context_type parse_poset_identifier(const std::string& identifier_in) {
        if (identifier_in != identifier())
            throw std::runtime_error("SCC poset identifier '" + identifier_in +
                                     "' does not match matrix poset '" + identifier() + "'");
        return 0;
    }

    static degree_type parse_degree(std::istream& input, context_type) {
        return Degree_traits<degree_type>::from_stream(input);
    }

    static Matrix make_matrix(index_type columns, index_type rows, context_type) {
        return Matrix(columns, rows);
    }
};

template <typename Matrix>
struct GradedMatrixIO<Matrix, std::enable_if_t<Matrix::runtime_dimension>> : MatrixIOOperations<Matrix> {
    using degree_type = typename MatrixSerializedDegree<Matrix>::type;
    using index_type = typename Matrix::index_type;
    using context_type = std::size_t;
    static constexpr bool runtime_dimension = true;

    static constexpr int output_precision() noexcept {
        constexpr int digits = std::numeric_limits<typename Matrix::scalar_type>::max_digits10;
        return digits > 0 ? digits : 17;
    }

    static context_type matrix_context(const Matrix& matrix) noexcept {
        return matrix.parameter_count();
    }

    static std::string identifier(context_type parameters) {
        return Degree_traits<degree_type>::poset_identifier(parameters);
    }

    static std::string identifier(const Matrix& matrix) {
        return identifier(matrix_context(matrix));
    }

    static context_type parse_poset_identifier(const std::string& identifier_in) {
        return Degree_traits<degree_type>::parse_poset_identifier(identifier_in);
    }

    static degree_type parse_degree(std::istream& input, context_type parameters) {
        return Degree_traits<degree_type>::from_stream(input, parameters);
    }

    static Matrix make_matrix(index_type columns, index_type rows, context_type parameters) {
        return Matrix(columns, rows, parameters);
    }
};

} // namespace graded_linalg
