/** @file sliced_coordinate_kernel.hpp
 * @brief NEW: arbitrary-dimensional kernels by slices of the existing 2D routine.
 *
 * Freeze the first d-2 coordinates at source-column thresholds; use the last
 * two as the bigrading. Lift each slice generator to the original full domain.
 * Prefix-lex slices followed by lex slice generators form a linear extension,
 * so ordinary fibre span checks remove redundant generators as they arrive.
 */
#pragma once
namespace graded_linalg { namespace detail {
template<class Matrix> Matrix sliced_grid_kernel(const Matrix& source);
} }
#include <grlina/dynamic_grid_matrix.hpp>
#include <map>

namespace graded_linalg {
namespace detail {

template<class Matrix>
Matrix sliced_grid_kernel(const Matrix& source) {
    using Index = typename Matrix::index_type;
    using Degree = typename Matrix::degree_type;
    using DT = Degree_traits<Degree>;
    const std::size_t dimension = source.parameter_count();
    const std::size_t frozen = dimension > 2 ? dimension-2 : 0;
    Matrix result = source.empty_like(0, source.get_num_cols());
    result.row_degrees = source.col_degrees;
    if (source.get_num_cols() == 0) return result;
    vec<vec<Index>> thresholds(frozen);
    for (auto degree : source.col_degrees)
        for (std::size_t a = 0; a < frozen; ++a) thresholds[a].push_back(degree[a]);
    for (auto& axis : thresholds) {
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    }
    auto add = [](const vec<Index>& left, const vec<Index>& right) {
        vec<Index> sum;
        std::set_symmetric_difference(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(sum));
        return sum;
    };
    auto generated = [&](vec<Index> column, const Degree& degree) {
        // Graded generators need not be a Groebner basis. Reduce the entire
        // eligible coefficient span before testing this candidate.
        std::map<Index, vec<Index>> pivots;
        for (Index i = 0; i < result.get_num_cols(); ++i) {
            if (!DT::smaller_equal(result.col_degree(i), degree)) continue;
            auto current = result.get_col(i);
            while (!current.empty() && pivots.count(current.back())) current = add(current, pivots.at(current.back()));
            if (!current.empty()) { Index pivot = current.back(); pivots[pivot] = std::move(current); }
        }
        while (!column.empty() && pivots.count(column.back())) column = add(column, pivots.at(column.back()));
        return column.empty();
    };
    Degree grade(dimension);
    auto run_slice = [&] {
        vec<Index> selected;
        for (Index i = 0; i < source.get_num_cols(); ++i) {
            bool active = true;
            for (std::size_t a = 0; a < frozen; ++a) if (source.col_degree(i)[a] > grade[a]) { active = false; break; }
            if (active) selected.push_back(i);
        }
        if (selected.empty()) return;
        Matrix slice(static_cast<Index>(selected.size()), source.get_num_rows(), 2);
        for (std::size_t a = 0; a < 2; ++a)
            if (frozen+a < dimension) slice.grids[a] = source.grids[frozen+a];
        auto projected = [&](auto original) {
            Degree degree(2);
            for (std::size_t a = 0; a < 2; ++a) if (frozen+a < dimension) degree[a] = original[frozen+a];
            return degree;
        };
        for (Index i = 0; i < source.get_num_rows(); ++i) slice.row_degrees.set(i, projected(source.row_degree(i)));
        for (Index i = 0; i < static_cast<Index>(selected.size()); ++i) {
            slice.col_degrees.set(i, projected(source.col_degree(selected[i])));
            slice.set_col(i, source.get_col(selected[i]));
        }
        auto kernel = slice.graded_kernel();
        for (Index i = 0; i < kernel.get_num_cols(); ++i) {
            for (std::size_t a = 0; a < 2; ++a) if (frozen+a < dimension) grade[frozen+a] = kernel.col_degree(i)[a];
            auto column = kernel.get_col(i);
            for (auto& row : column) row = selected[row];
            if (!generated(column, grade)) result.append_column_at_grid_degree(column, grade);
        }
    };
    auto enumerate = [&](auto&& self, std::size_t a) -> void {
        if (a == frozen) { run_slice(); return; }
        for (Index value : thresholds[a]) { grade[a] = value; self(self, a+1); }
    };
    // ponytail: at most n^(d-2) slices; event/signature scheduling can replace
    // independent slices when higher-dimensional workloads need it.
    enumerate(enumerate, 0);
    result.sort_columns_lexicographically();
    return result;
}

} // namespace detail
} // namespace graded_linalg
