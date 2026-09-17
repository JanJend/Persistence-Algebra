/** Linear algebra of coefficient-matrix families over F2. */
#pragma once
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include <grlina/graded_matrix.hpp>

namespace graded_linalg {

/** Reduce A, then replace B with representatives of
 * (span(A) + span(B))/span(A). A keeps its length (possibly zero entries).
 * Matrices are vectors of coefficients here: shapes must agree, but degree
 * labels may differ. The caller must supply the common interpretation, e.g.
 * transport endomorphism lifts along a canonical shift before taking a quotient.
 * This is NOT a quotient of homomorphisms modulo target relations.
 * Neither family is reordered in its ambient row/column coordinates.
 */
template <typename Matrix>
void reduce_matrix_family_modulo(std::vector<Matrix>& A, std::vector<Matrix>& B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    if (&A == &B) throw std::invalid_argument("Matrix families must not alias");
    if (A.empty() && B.empty()) return;
    using index = typename Matrix::index_type;
    using position = std::pair<index, index>;
    const Matrix& model = A.empty() ? B.front() : A.front();
    for (const auto* family : {&A, &B})
        for (const auto& matrix : *family) {
            GRLINA_DEBUG_CHECK(matrix.validate());
            if (matrix.get_num_cols() != model.get_num_cols() ||
                matrix.get_num_rows() != model.get_num_rows())
                throw std::invalid_argument("Matrix family shapes differ");
        }
    // A pair avoids flattened-index overflow and works for rectangular matrices.
    auto pivot = [](const Matrix& matrix) -> std::optional<position> {
        for (index c = matrix.get_num_cols(); c > 0; --c)
            if (matrix.col_last(c - 1) != -1)
                return position{c - 1, matrix.col_last(c - 1)};
        return std::nullopt;
    };
    // Neither input vector is resized until reduction ends; reducer pointers
    // remain stable and avoid copying potentially large lift matrices.
    std::map<position, Matrix*> reducers;
    auto reduce = [&](Matrix& matrix) {
        matrix.invalidate_cached_rows();
        while (auto p = pivot(matrix)) {
            auto found = reducers.find(*p);
            if (found == reducers.end()) {
                reducers.emplace(*p, &matrix);
                return true;
            }
            for (index c = 0; c < matrix.get_num_cols(); ++c)
                matrix.add_to_col(c, found->second->data[c]);
        }
        return false;
    };
    for (auto& matrix : A) reduce(matrix);
    std::vector<std::size_t> survivors;
    for (std::size_t i = 0; i < B.size(); ++i)
        if (reduce(B[i])) survivors.push_back(i);
    std::vector<Matrix> representatives;
    representatives.reserve(survivors.size());
    for (auto i : survivors) representatives.push_back(std::move(B[i]));
    B = std::move(representatives);
}

} // namespace graded_linalg
