/**
 * @file scc_io.hpp
 * @brief SCC metadata adapter for coordinate and grid matrices.
 */
#pragma once

#include <type_traits>
#include <vector>

#include <grlina/mat_graded.hpp>

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
    static void normalize_complex(std::vector<Matrix>& matrices) {
        if constexpr (MatrixSerializedDegree<Matrix>::grid) {
            if (matrices.empty()) return;
            Matrix merged = matrices.front();
            for (auto& matrix : matrices) merged.merge_grids(matrix);
            for (auto& matrix : matrices) matrix.reindex_grid(merged.grids);
        }
    }
};

/** SCC operations using the matrix's serialized degree type. */
template <typename Matrix, typename = void>
struct GradedMatrixIO : MatrixIOOperations<Matrix> {
    using degree_type = typename MatrixSerializedDegree<Matrix>::type;
    using index_type = typename Matrix::index_type;
    static constexpr int output_precision() noexcept { return 17; }

    static Matrix make_matrix(index_type columns, index_type rows) {
        return Matrix(columns, rows);
    }
};


} // namespace graded_linalg
