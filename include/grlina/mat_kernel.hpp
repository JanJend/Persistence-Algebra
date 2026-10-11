/**
 * @file mat_kernel.hpp
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
#include <grlina/deg_traits.hpp>
#include <grlina/mat_sparse.hpp>
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

  template<typename index, typename Coordinate = index>
  struct Sort_grades {
      using index_pair = std::pair<Coordinate, Coordinate>;
  
      bool operator()(const index_pair& c1, const index_pair& c2) const noexcept {
          return (c1.first > c2.first) || (c1.first == c2.first && c1.second > c2.second);
      }
  };
  
  template<typename index, typename Coordinate = index>
  class Grid_scheduler {
      using grade_pair = std::pair<Coordinate, Coordinate>;
      using index_pair = std::pair<index, index>;
      
      private:
      std::priority_queue<grade_pair, std::vector<grade_pair>, Sort_grades<index, Coordinate>> grades;
      std::map<grade_pair, index_pair> index_range;
      grade_pair curr_grade = {std::numeric_limits<Coordinate>::lowest(), std::numeric_limits<Coordinate>::lowest()};

      public:
          Grid_scheduler() = default;
      
          // Compatibility with legacy R2's already-compressed coordinate fields.
          // This constructor does not perform coordinate compression.
          template<typename GradedMatrix>
          explicit Grid_scheduler(GradedMatrix& M) {
              initialize(M);
          }

          // Group existing column births into event ranges. Native integer
          // coordinates need no dense grid or coordinate compression.
          template<typename Degrees>
          Grid_scheduler(const Degrees& degrees, index count) {
              for (index first = 0; first < count;) {
                  grade_pair grade{degrees[first][0], degrees[first][1]};
                  index last = first + 1;
                  while (last < count && degrees[last] == degrees[first]) ++last;
                  index_range[grade] = {first, last};
                  grades.push(grade);
                  first = last;
              }
          }
      
          int size() const noexcept {
              return grades.size();
          }
      
          bool at_end() const noexcept {
              return grades.empty();
          }
      
          grade_pair next_grade() noexcept {
              if (grades.empty()) {
                  return {0, 0};
              }
      
              grade_pair result = std::move(grades.top());
              grades.pop();
      
              // Skip duplicate entries
              while (!grades.empty() && grades.top() == result) {
                  grades.pop();
              }
      
              curr_grade = result;
              return result;
          }
      
          index_pair index_range_at(Coordinate x, Coordinate y) const noexcept {
              auto it = index_range.find({x, y});
              return (it != index_range.end()) ? it->second : index_pair{0, 0};
          }
      
          void notify(Coordinate x, Coordinate y) noexcept {
              if (curr_grade != grade_pair{x, y}) {
                  grades.push({x, y});
              }
          }
      
      
      private:
          template<typename GradedMatrix>
          void initialize(GradedMatrix& M) {
              index_pair last_pair = {std::numeric_limits<index>::min(), std::numeric_limits<index>::min()};
              index curr_start = std::numeric_limits<index>::min();
      
              for (index i = 0; i < M.get_num_cols(); ++i) {
                  index curr_x = M.z2_col_degrees[i][0];
                  index curr_y = M.z2_col_degrees[i][1];
      
                  GRLINA_ASSERT(curr_x < static_cast<index>(M.x_grid.size()));
                  GRLINA_ASSERT(curr_y < static_cast<index>(M.y_grid.size()));
      
                  if (curr_x != last_pair.first || curr_y != last_pair.second) {
                      // New grade encountered
                      if (curr_start != std::numeric_limits<index>::min()) {
                          index_range[last_pair] = {curr_start, i};
                      }
      
                      curr_start = i;
                      last_pair = {curr_x, curr_y};
                      grades.push(last_pair);
                  }
              }
      
              // Store the final range
              if (curr_start != std::numeric_limits<index>::min()) {
                  index_range[last_pair] = {curr_start, M.get_num_cols()};
              }
          }
  };


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
    using Index = typename Matrix::index_type;
    using Degree = typename Matrix::degree_type;
    using Coordinate = typename Degree::value_type;
    static_assert(std::tuple_size_v<Degree> == 2, "Kernel degrees must have two coordinates");
    static_assert(std::is_integral_v<Coordinate>, "Kernel degree coordinates must be integers");

    // Reduction requires colex order: compare y first, then x. Save the
    // sorted-to-original permutation for the coordinates of the kernel vectors.
    auto permutation = matrix.sort_columns_with_permutation(
        [](const Degree& left, const Degree& right) {
            return left[1] < right[1] || (left[1] == right[1] && left[0] < right[0]);
        }).new_to_old;
    matrix.invalidate_cached_rows();
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
    using Index = typename Matrix::index_type;
    using Degree = typename Matrix::degree_type;
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
