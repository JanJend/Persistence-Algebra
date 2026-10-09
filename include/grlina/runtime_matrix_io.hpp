/** @file runtime_matrix_io.hpp
 * @brief NEW: presentation and SCC-sum loaders for runtime grid matrices.
 */
#pragma once
#include <grlina/dynamic_grid_matrix.hpp>
#include <istream>
#include <stdexcept>
#include <string>

namespace graded_linalg {

template<class Scalar, class Index, class Storage, class InputStream>
void read_sccsum(vec<DynamicGridGradedSparseMatrix<Scalar, Index, Storage>>& matrices,
                 InputStream& input, bool lex_sort = false, bool compute_batches = false,
                 vec<std::string>* matrix_types = nullptr) {
    input >> std::ws;
    std::string header;
    std::getline(input, header);
    if (!header.empty() && header.back() == '\r') header.pop_back();
    if (header != "scc2020sum") throw std::runtime_error("Expected scc2020sum header");
    std::size_t count;
    if (!(input >> count)) throw std::runtime_error("Invalid SCC sum matrix count");
    for (std::size_t i = 0; i < count; ++i) {
        input >> std::ws;
        std::string type;
        if (!std::getline(input, type)) throw std::runtime_error("Missing SCC sum matrix type");
        if (!type.empty() && type.back() == '\r') type.pop_back();
        if (matrix_types) matrix_types->push_back(type);
        DynamicGridGradedSparseMatrix<Scalar, Index, Storage> matrix(input, lex_sort);
        if (compute_batches) matrix.compute_col_batches();
        matrices.push_back(std::move(matrix));
    }
}

template<class Scalar, class Index, class Storage, class InputStream>
void construct_matrices_from_stream(vec<DynamicGridGradedSparseMatrix<Scalar, Index, Storage>>& matrices,
                                    InputStream& input, bool lex_sort = false,
                                    bool compute_batches = false) {
    while (input >> std::ws && input.peek() != std::char_traits<char>::eof()) {
        const auto position = input.tellg();
        std::string header;
        std::getline(input, header);
        input.seekg(position);
        if (!input) throw std::runtime_error("Presentation loading requires a seekable stream");
        if (header == "scc2020sum" || header == "scc2020sum\r") {
            read_sccsum(matrices, input, lex_sort, compute_batches);
        } else {
            DynamicGridGradedSparseMatrix<Scalar, Index, Storage> matrix(input, lex_sort);
            if (compute_batches) matrix.compute_col_batches();
            matrices.push_back(std::move(matrix));
        }
    }
}

} // namespace graded_linalg
