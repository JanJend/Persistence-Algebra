/**
 * @file graded_kernel.hpp
 * @author Michael Kerber - adapted to graded_linalg by Jan Jendrysiak
 * @brief MPfree two-parameter and muphasa three-parameter kernel reductions.

   mpfree is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   mpfree is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with mpfree.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <grlina/general_checks.hpp>
#include <grlina/posets/traits.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/graded_kernel_scheduler.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/r2.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace graded_linalg {

  // MPfree's reduce_column: a helper may displace a later pivot owner and
  // reschedule it; a right-hand side only consumes the current helper basis.
  template<class Matrix, class Slave, class Pivots, class Index, class Notify>
  void reduce_graded_column(Matrix& matrix, Slave& slave, Pivots& pivots,
                            Index column, Notify notify,
                            bool store_operations = true, bool update_pivot = true) {
      auto pivot = matrix.col_last(column);
      while (pivot != -1) {
          auto owner = pivots.find(pivot);
          if (owner == pivots.end() || owner->second >= column) break;
          matrix.col_op(owner->second, column);
          if (store_operations) slave.col_op(owner->second, column);
          pivot = matrix.col_last(column);
      }
      if (pivot != -1 && update_pivot) {
          auto owner = pivots.find(pivot);
          if (owner == pivots.end()) pivots[pivot] = column;
          else if (owner->second > column) {
              notify(owner->second);
              pivots[pivot] = column;
          }
      }
  }

  // Shared Kerber--Rolle traversal. The visitor decides whether to emit a
  // kernel column or a solution, and returns false to stop work immediately.
  template<class Index, class Coordinate, class Queues, class Degrees, class Visit>
  void visit_bigraded_columns(Grid_scheduler<Index, Coordinate>& scheduler, Queues& queues,
                             const Degrees& degrees, Visit visit) {
      while (!scheduler.at_end()) {
          auto grade = scheduler.next_grade();
          auto& queue = queues[grade.second];
          auto range = scheduler.index_range_at(grade.first, grade.second);
          for (Index column = range.first; column < range.second; ++column) queue.push(column);
          auto notify = [&](Index owner) {
              auto y = degrees[owner][1];
              queues[y].push(owner);
              scheduler.notify(grade.first, y);
          };
          while (!queue.empty()) {
              Index column = queue.top();
              while (!queue.empty() && queue.top() == column) queue.pop();
              GRLINA_ASSERT(degrees[column][0] <= grade.first);
              GRLINA_ASSERT(degrees[column][1] == grade.second);
              if (!visit(column, grade, notify)) return;
          }
      }
  }

  // General-dimensional kernels freeze all but their last two coordinates.
  // Solvers use the same restriction, only at frozen grades requested by B.
  template<class Index, class Degrees, class Prefix>
  std::vector<Index> columns_in_bigraded_slice(const Degrees& degrees,
                                              const Prefix& prefix,
                                              std::size_t frozen_axes) {
      std::vector<Index> columns;
      for (Index column = 0; column < static_cast<Index>(degrees.size()); ++column) {
          bool active = true;
          for (std::size_t axis = 0; axis < frozen_axes; ++axis)
              if (degrees[column][axis] > prefix[axis]) { active = false; break; }
          if (active) columns.push_back(column);
      }
      return columns;
  }


  } // namespace graded_linalg

namespace graded_linalg {

template <class Matrix, class = void>
struct has_kernel_empty_like : std::false_type {};
template <class Matrix>
struct has_kernel_empty_like<Matrix, std::void_t<
    decltype(std::declval<const Matrix&>().empty_like(0, 0))>> : std::true_type {};

// The input already has Z^2 degrees. Only column order changes; no coordinates
// are compressed or converted. The returned rows retain the original domain.
template <class Matrix>
Matrix z2_graded_kernel(Matrix& matrix) {
    using Index = matrix_index_t<Matrix>;
    using Degree = matrix_degree_t<Matrix>;
    using Coordinate = typename Degree::value_type;
    static_assert(std::tuple_size_v<Degree> == 2, "Kernel degrees must have two coordinates");
    static_assert(std::is_integral_v<Coordinate>, "Kernel degree coordinates must be integers");

    // Reduction requires colex order: compare y first, then x. Save the
    // sorted-to-original permutation for the coordinates of the kernel vectors.
    auto permutation = matrix.sort_columns_with_permutation(
        [](const Degree& left, const Degree& right) {
            return left[1] < right[1] || (left[1] == right[1] && left[0] < right[0]);
        }).new_to_old;
    // All reduction state belongs to this call, rather than the matrix object.
    Grid_scheduler<Index, Coordinate> scheduler(matrix.col_degrees, matrix.get_num_cols());
    using Queue = std::priority_queue<Index, vec<Index>, std::greater<Index>>;
    std::map<Coordinate, Queue> queues;
    std::map<Index, Index> pivots;

    SparseMatrix<Index> operations(matrix.get_num_cols(), matrix.get_num_cols(), "Identity");
    array<Index> columns;
    vec<Degree> degrees;
    std::vector<bool> emitted(matrix.get_num_cols(), false);
    visit_bigraded_columns(scheduler, queues, matrix.col_degrees,
        [&](Index column, auto grade, auto notify) {
            reduce_graded_column(matrix, operations, pivots, column, notify);
            if (!emitted[column] && matrix.is_zero(column)) {
                columns.push_back(operations.get_col(column));
                degrees.push_back({grade.first, grade.second});
                emitted[column] = true;
                matrix.clear_col(column);
                operations.clear_col(column);
            }
            return true;
        });

    // Kernel vectors live in the input domain (one row per original column),
    // so they form a separate matrix. empty_like preserves optional grid metadata.
    Matrix result = [&] {
        auto count = static_cast<Index>(columns.size());
        if constexpr (has_kernel_empty_like<Matrix>::value)
            return matrix.empty_like(count, matrix.get_num_cols());
        else return Matrix(count, matrix.get_num_cols());
    }();
    result.assign_data(std::move(columns));
    result.col_degrees = std::move(degrees);
    result.row_degrees = matrix.col_degrees;
    result.permute_rows_graded(permutation);
    // Preserve the legacy routine's lexicographic order of kernel generators.
    result.sort_columns_lexicographically();
    return result;
}

/* Three-parameter adaptation of muphasa's computeGroebnerBases_gradeopt_min
 * (src/cpp/groebner.cpp), retaining its signature histories, pivot-grade joins
 * and minimal join scheduling. Only the kernel part of its output is needed.
 *
 * MIT License
 * Copyright (c) 2021 Oliver Gäfvert, Matías R. Bender, Michael Lesnick
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
template <class Matrix>
Matrix z3_graded_kernel(const Matrix& matrix) {
    using Index = matrix_index_t<Matrix>;
    using Degree = matrix_degree_t<Matrix>;
    using DT = Degree_traits<Degree>;
    using CT = Column_traits<vec<Index>, Index>;
    static_assert(std::tuple_size_v<Degree> == 3, "Kernel degrees must have three coordinates");
    static_assert(std::is_integral_v<typename Degree::value_type>, "Kernel coordinates must be integers");

    // Signatures use colex order; operation vectors keep original domain indices.
    vec<Index> order(matrix.get_num_cols());
    for (Index i = 0; i < matrix.get_num_cols(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
        return DT::colex_order(matrix.col_degrees[a], matrix.col_degrees[b]);
    });
    struct ImageColumn {
        vec<Index> image, operations;
        Degree degree;
        std::size_t last_updated = 0;
    };
    vec<ImageColumn> image_basis;
    vec<vec<std::size_t>> image_signatures(order.size());
    vec<vec<Degree>> syzygy_signatures(order.size()), pivot_grades(matrix.get_num_rows());
    std::priority_queue<Degree, vec<Degree>, std::greater<Degree>> grades;
    std::set<Degree> visited;
    auto schedule = [&](const Degree& grade) {
        if (visited.insert(grade).second) grades.push(grade);
    };
    for (const auto& degree : matrix.col_degrees) schedule(degree);

    auto schedule_joins = [&](Index pivot, const Degree& degree) {
        auto& previous = pivot_grades[pivot];
        if (!previous.empty() && previous.back() == degree) return;
        vec<Degree> minimal_joins;
        for (const auto& old : previous) {
            auto joined = DT::join(old, degree);
            // For fixed degree, ordering joins is equivalent to muphasa's
            // m_ji ordering, without subtracting possibly large coordinates.
            if (std::any_of(minimal_joins.begin(), minimal_joins.end(),
                    [&](const Degree& earlier) { return DT::smaller_equal(earlier, joined); })) continue;
            minimal_joins.push_back(joined);
            schedule(joined);
        }
        previous.push_back(degree);
    };

    constexpr auto no_column = std::numeric_limits<std::size_t>::max();
    vec<std::size_t> pivots(matrix.get_num_rows(), no_column);
    array<Index> kernel_columns;
    vec<Degree> kernel_degrees;
    std::size_t iteration = 0;
    while (!grades.empty()) {
        Degree grade = grades.top();
        grades.pop();
        ++iteration;
        for (std::size_t signature = 0; signature < order.size(); ++signature) {
            Index original = order[signature];
            const auto& birth = matrix.col_degrees[original];
            if (birth[2] > grade[2]) break;
            const auto& syzygies = syzygy_signatures[signature];
            if (std::any_of(syzygies.begin(), syzygies.end(),
                    [&](const Degree& degree) { return DT::smaller_equal(degree, grade); })) continue;

            auto& history = image_signatures[signature];
            std::size_t column = no_column;
            for (auto it = history.rbegin(); it != history.rend(); ++it)
                if (DT::smaller_equal(image_basis[*it].degree, grade)) { column = *it; break; }
            bool is_new = history.empty() && birth == grade;
            if (column == no_column && !is_new) continue;
            ImageColumn working = is_new ? ImageColumn{matrix.get_col(original), {original}, birth}
                                         : image_basis[column];
            bool changed = false;
            while (!working.image.empty()) {
                auto owner = pivots[working.image.back()];
                if (owner == no_column || image_basis[owner].last_updated != iteration) break;
                const auto& reducer = image_basis[owner];
                CT::add_to(reducer.image, working.image);
                CT::add_to(reducer.operations, working.operations);
                working.degree = DT::join(working.degree, reducer.degree);
                changed = true;
            }
            if (working.image.empty()) {
                kernel_columns.push_back(std::move(working.operations));
                kernel_degrees.push_back(working.degree);
                syzygy_signatures[signature].push_back(working.degree);
                continue;
            }
            if (is_new || changed) {
                column = image_basis.size();
                image_basis.push_back(std::move(working));
                history.push_back(column);
            }
            auto& reduced = image_basis[column];
            reduced.last_updated = iteration;
            Index pivot = reduced.image.back();
            pivots[pivot] = column;
            if ((is_new || changed) && reduced.degree == grade) schedule_joins(pivot, reduced.degree);
        }
    }

    Matrix result = [&] {
        auto count = static_cast<Index>(kernel_columns.size());
        if constexpr (has_kernel_empty_like<Matrix>::value)
            return matrix.empty_like(count, matrix.get_num_cols());
        else return Matrix(count, matrix.get_num_cols());
    }();
    result.assign_data(std::move(kernel_columns));
    result.col_degrees = std::move(kernel_degrees);
    result.row_degrees = matrix.col_degrees;
    result.sort_columns_lexicographically();

    // Muphasa returns a kernel Groebner basis. Preserve graded_kernel's minimal
    // generating-set contract by discarding generators in the eligible span.
    array<Index> minimal_columns;
    vec<Degree> minimal_degrees;
    for (Index c = 0; c < result.get_num_cols(); ++c) {
        std::map<Index, vec<Index>> span;
        auto reduce = [&](vec<Index>& column) {
            while (!column.empty()) {
                auto it = span.find(column.back());
                if (it == span.end()) break;
                CT::add_to(it->second, column);
            }
        };
        for (std::size_t i = 0; i < minimal_columns.size(); ++i) {
            if (!DT::smaller_equal(minimal_degrees[i], result.col_degrees[c])) continue;
            auto column = minimal_columns[i];
            reduce(column);
            if (!column.empty()) span.emplace(column.back(), std::move(column));
        }
        auto column = result.get_col(c);
        reduce(column);
        if (!column.empty()) {
            minimal_columns.push_back(result.get_col(c));
            minimal_degrees.push_back(result.col_degrees[c]);
        }
    }
    result.assign_data(std::move(minimal_columns));
    result.col_degrees = std::move(minimal_degrees);
    result.set_num_cols(static_cast<Index>(result.col_degrees.size()));
    return result;
}

} // namespace graded_linalg

namespace graded_linalg {

template<typename Scalar, std::size_t Dimension, typename Index, typename Derived>
auto CoordGradedSparseMatrix<Scalar, Dimension, Index, Derived>::graded_kernel() -> Self {
    if constexpr (!std::is_integral_v<Scalar>) {
        GridGradedSparseMatrix<Scalar, Dimension, Index> grid(*this);
        auto kernel = grid.graded_kernel();
        this->assign_data(std::move(grid.data));
        this->col_degrees = grid.real_col_degrees();
        this->compatibly_sorted = false;
        Self result = static_cast<const Self&>(*this).empty_like(kernel.get_num_cols(), kernel.get_num_rows());
        result.assign_data(std::move(kernel.data));
        result.col_degrees = kernel.real_col_degrees();
        result.row_degrees = kernel.real_row_degrees();
        result.compatibly_sorted = false;
        return result;
    } else if constexpr (Dimension == 2) {
        return z2_graded_kernel(static_cast<Self&>(*this));
    } else if constexpr (Dimension == 3) {
        return z3_graded_kernel(static_cast<const Self&>(*this));
    } else {
        constexpr std::size_t frozen = Dimension > 2 ? Dimension - 2 : 0;
        Self result = static_cast<const Self&>(*this).empty_like(0, this->get_num_cols());
        result.row_degrees = this->col_degrees;
        std::array<vec<Scalar>, frozen> thresholds;
        for (const auto& degree : this->col_degrees)
            for (std::size_t a = 0; a < frozen; ++a) thresholds[a].push_back(degree[a]);
        for (auto& axis : thresholds) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        }
        auto generated = [&](vec<Index> column, const degree_type& degree) {
            SparseMatrix<Index> basis(result.get_num_cols() + 1, result.get_num_rows());
            Index count = 0;
            for (Index i = 0; i < result.get_num_cols(); ++i)
                if (Degree_traits<degree_type>::smaller_equal(result.col_degrees[i], degree)) {
                    basis.append_col(result.get_col(i)); ++count;
                }
            basis.append_col(column);
            basis.set_num_cols(count + 1);
            basis.column_reduction();
            return basis.is_zero(count);
        };
        degree_type grade{};
        auto run_slice = [&] {
            auto selected = columns_in_bigraded_slice<Index>(this->col_degrees, grade, frozen);
            if (selected.empty()) return;
            CoordGradedSparseMatrix<Scalar, 2, Index> slice(static_cast<Index>(selected.size()), this->get_num_rows());
            auto project = [&](const degree_type& original) {
                CoordinateDegree<Scalar, 2> degree{};
                for (std::size_t a = 0; a < 2; ++a) if (frozen + a < Dimension) degree[a] = original[frozen + a];
                return degree;
            };
            for (Index i = 0; i < this->get_num_rows(); ++i) slice.row_degrees[i] = project(this->row_degrees[i]);
            for (Index i = 0; i < static_cast<Index>(selected.size()); ++i) {
                slice.col_degrees[i] = project(this->col_degrees[selected[i]]);
                slice.set_col(i, this->get_col(selected[i]));
            }
            auto kernel = slice.graded_kernel();
            for (Index i = 0; i < kernel.get_num_cols(); ++i) {
                for (std::size_t a = 0; a < 2; ++a) if (frozen + a < Dimension) grade[frozen + a] = kernel.col_degrees[i][a];
                auto column = kernel.get_col(i);
                for (auto& row : column) row = selected[row];
                if (!generated(column, grade)) static_cast<Base&>(result).append_column(column, grade);
            }
        };
        auto enumerate = [&](auto&& self, std::size_t a) -> void {
            if (a == frozen) { run_slice(); return; }
            for (auto value : thresholds[a]) { grade[a] = value; self(self, a + 1); }
        };
        // ponytail: higher dimensions use up to n^(Dimension-2) slices; event scheduling can replace them if needed.
        enumerate(enumerate, 0);
        result.sort_columns_lexicographically();
        return result;
    }
}

template<typename Scalar, std::size_t Dimension, typename Index, typename Derived>
auto CoordGradedSparseMatrix<Scalar, Dimension, Index, Derived>::graded_kernel() const -> Self {
    Self copy = static_cast<const Self&>(*this);
    return copy.graded_kernel();
}

template<typename index, typename Storage>
auto R2GradedSparseMatrix<index, Storage>::initialise_grid_scheduler() -> void {
    this->grid_scheduler = Grid_scheduler<index>(*this);
}

template<typename index, typename Storage>
auto R2GradedSparseMatrix<index, Storage>::kernel_column_reduction(index i, pair<index>& curr_gr, SparseBase& column_operations, bool store_col_ops, bool notify_pq) -> void {
    reduce_graded_column(*this, column_operations, this->pivots, i,
        [&](index owner) {
            if (notify_pq) {
                index y = this->z2_col_degrees[owner][1];
                this->pq_row[y].push(owner);
                this->grid_scheduler.notify(curr_gr.first, y);
            }
        }, store_col_ops);
}

template<typename index, typename Storage>
auto R2GradedSparseMatrix<index, Storage>::graded_kernel() -> R2GradedSparseMatrix {
    this->pivots.clear();
    this->pq_row.clear();
    GRLINA_ASSERT(this->col_degrees.size() == this->get_num_cols());
    GRLINA_ASSERT(this->row_degrees.size() == this->get_num_rows());
    vec<index> column_permutation = this->compute_grid_representation();
    this->initialise_grid_scheduler();

    pq_row.resize(this->y_grid.size());
    // This is the "slave" matrix in mpfree
    SparseBase col_operations = SparseBase(this->get_num_cols(), this->get_num_cols(), "Identity");

    std::vector<r2degree> new_degrees; // Basis for the free module which is part of the kernel
    Storage new_cols; // representing the matrix given by the kernel

    std::vector<bool> indices_in_kernel(this->get_num_cols(), false);

    // Initialize grid scheduler for processing degrees in order
    Grid_scheduler<index>& grid = this->grid_scheduler;

    visit_bigraded_columns(grid, pq_row, z2_col_degrees,
        [&](index i, pair<index> grade, auto notify) {
            reduce_graded_column(*this, col_operations, this->pivots, i, notify);
            if (!indices_in_kernel[i] && this->is_zero(i)) {
                new_cols.push_back(col_operations.get_col(i));
                new_degrees.push_back({this->x_grid[grade.first], this->y_grid[grade.second]});
                indices_in_kernel[i] = true;
                this->clear_col(i);
                col_operations.clear_col(i);
            }
            return true;
        });

    // Build the resulting kernel matrix
    R2GradedSparseMatrix<index, Storage> result(new_cols.size(), this->get_num_cols());
    result.assign_data(std::move(new_cols));
    result.col_degrees = std::move(new_degrees);
    result.row_degrees = this->col_degrees;

    result.permute_rows_graded(column_permutation);
    // Rows are coordinates in the ORIGINAL input domain: never sort them
    // independently. Pullbacks project these rows by their original index.
    result.sort_columns_lexicographically();

    return result;
}

} // namespace graded_linalg
