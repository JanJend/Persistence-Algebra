/**
 * @file io.hpp
 * @brief SCC metadata adapters and matrix stream helpers.
 */
#pragma once

#include <type_traits>
#include <vector>

#include <grlina/matrices/graded.hpp>
#include <grlina/matrices/r2.hpp>

namespace graded_linalg {

template<class Matrix>
struct MatrixIOOperations {
    using serialized_degree_type = typename Matrix::degree_type;
    static void assign_degrees(Matrix& matrix, const std::vector<serialized_degree_type>& columns,
                               const std::vector<serialized_degree_type>& rows) {
        if constexpr (matrix_grid_backed_v<Matrix>) matrix.set_real_degrees(columns, rows);
        else { matrix.col_degrees = columns; matrix.row_degrees = rows; }
    }
    template<class Output> static void write_column_degree(Output& out, const Matrix& matrix, matrix_index_t<Matrix> i) {
        Degree_traits<serialized_degree_type>::write_degree(out, matrix.col_degree(i));
    }
    template<class Output> static void write_row_degree(Output& out, const Matrix& matrix, matrix_index_t<Matrix> i) {
        Degree_traits<serialized_degree_type>::write_degree(out, matrix.row_degree(i));
    }
    static void normalize_complex(std::vector<Matrix>& matrices) {
        if constexpr (matrix_grid_backed_v<Matrix>) {
            if (matrices.empty()) return;
            Matrix merged = matrices.front();
            for (auto& matrix : matrices) merge_grids(merged, matrix);
            for (auto& matrix : matrices) matrix.reindex_grid(merged.grid_);
        }
    }
};

/** SCC operations using the matrix's serialized degree type. */
template <typename Matrix, typename = void>
struct GradedMatrixIO : MatrixIOOperations<Matrix> {
    using degree_type = typename Matrix::degree_type;
    using index_type = matrix_index_t<Matrix>;
    static constexpr int output_precision() noexcept { return 17; }

    static Matrix make_matrix(index_type columns, index_type rows) {
        return Matrix(columns, rows);
    }
};


/**
 * @brief Can be used to recognise file extension, not really needed right now.
 *
 * @param filepath
 * @return std::ifstream
 */
inline std::ifstream check_matrix_file(const std::string& filepath) {
    size_t dotPosition = filepath.find_last_of('.');
    bool no_file_extension = false;
    if (dotPosition == std::string::npos) {
        // No dot found, invalid file format
        no_file_extension = true;
        std::cout << " File does not have an extension (.scc .firep .txt)?" << std::endl;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << " Error: Unable to open file " << filepath << std::endl;
        std::abort();
    }

    std::string extension;
    if(!no_file_extension) {
        extension=filepath.substr(dotPosition);
    }

    // Check the file extension and perform actions accordingly
    if (extension == ".scc" || extension == ".firep" || extension == ".txt" || no_file_extension) {
        // std::cout << "Reading presentation file: " << filepath << std::endl;
    } else {
        // Invalid file extension
        std::cout << "Warning, extension does not match .scc, .firep, .txt, or no extension." << std::endl;
    }
    return file;
}



/**
 * @brief Constructs graded matrices from an input stream.
 *
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename index, typename Storage = vec<vec<index>>>
std::vector<R2GradedSparseMatrix<index, Storage>> get_matrices_from_stream(std::ifstream& file_stream, bool lex_sort = false) {
    std::vector<R2GradedSparseMatrix<index, Storage>> matrices;

    while (file_stream) {
        // Construct a new matrix from the stream
        R2GradedSparseMatrix<index, Storage> matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));

        // Check if the stream is exhausted
        if (file_stream.eof()) {
            break;
        }
    }

    return matrices;
}

/**
 * @brief Constructs graded matrices from an input stream.
 *
 * @param matrices vector to store the constructed matrices
 * @param file_stream input file stream containing multiple matrices
 * @param lex_sort whether to sort lexicographically
 * @return std::vector<R2GradedSparseMatrix> vector of R2GradedSparseMatrix objects
 */
template <typename Matrix, typename InputStream>
void construct_matrices_from_stream(std::vector<Matrix>& matrices, InputStream& file_stream, bool lex_sort = false) {

    while (file_stream) {
        // Construct a new matrix from the stream
        Matrix matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));

        // Skip empty lines
        std::string line;
        while (std::getline(file_stream, line)) {
            if (!line.empty() && line.find_first_not_of(" \t\n\r\f\v") != std::string::npos) {
                file_stream.seekg(-static_cast<int>(line.length()) - 1, std::ios_base::cur);
                break;
            }
        }

        if (file_stream.eof()) {
            break;
        }
    }

}

/**
 * @brief Constructs matrices from stream with header and type annotations
 *
 * @param matrices vector to store the constructed matrices
 * @param file_stream input stream with format: header, count, then (type, matrix) pairs
 * @param lex_sort whether to sort lexicographically
 * @param matrix_types optional vector to store the type of each matrix (e.g., "cyclic")
 */
template <typename Matrix, typename InputStream>
void read_sccsum(
    std::vector<Matrix>& matrices,
    InputStream& file_stream,
    bool lex_sort = false,
    std::vector<std::string>* matrix_types = nullptr) {

    // Read and validate header
    std::string header;
    std::getline(file_stream, header);
    if (header != "scc2020sum") {
        throw std::runtime_error("Invalid header: expected 'scc2020sum', got '" + header + "'");
    }

    // Read matrix count
    size_t count;
    file_stream >> count;

    matrices.reserve(count);
    if (matrix_types) {
        matrix_types->reserve(count);
    }

    for (size_t i = 0; i < count; ++i) {
        // Skip empty lines and read type
        std::string type;
        while (std::getline(file_stream, type) && type.empty()) {}

        if (matrix_types) {
            matrix_types->push_back(type);
        }

        // Construct matrix from stream
        Matrix matrix(file_stream, lex_sort);
        matrices.push_back(std::move(matrix));
    }
}


} // namespace graded_linalg
