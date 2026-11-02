/**
 * @file MPFREE_clone.hpp
 * @author Michael Kerber - adapted to graded_linalg by Jan Jendrysiak
 * @brief Preserved MPfree graded-kernel clone with its original grid scheduler.
 *
 * Borrowed from MPfree by Michael Kerber and Alexander Rolle:
 * https://bitbucket.org/mkerber/mpfree
 * Adapted to graded_linalg by Jan Jendrysiak. Restores the implementation
 * predating the graded-header consolidation: grid compression, grade heap,
 * per-y column queues and kernel reduction. This is separate from the newer
 * streamed traversal in graded_linear_system.hpp; it remains directly usable.

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
#include <grlina/matrices/r2.hpp>
#include <algorithm>
#include <limits>
#include <map>
#include <queue>
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
  void reduce_graded_column_MPFREE(Matrix& matrix, Slave& slave, Pivots& pivots,
                            Index column, Notify notify,
                            bool store_operations = true, bool update_pivot = true) {
      auto pivot = matrix.col_last(column);
      bitset packed;
      std::size_t additions = 0;
      while (pivot != -1) {
          auto owner = pivots.find(pivot);
          if (owner == pivots.end() || owner->second >= column) break;
          // Keep short/sparse reductions sparse. After two additions, a column
          // with >=128 entries and >=1/16 density can avoid repeated merging
          // and allocation by accumulating in bits. Only this column is packed;
          // reducer columns and the operation matrix retain their own storage.
          if (packed.empty() && additions >= 2) {
              const auto& entries = matrix.column(column);
              if (entries.size() >= 128 && entries.size() >= (std::size_t(pivot) + 16) / 16) {
                  packed.resize(std::size_t(pivot) + 1);
                  for (auto row : entries) packed.set(row);
              }
          }
          if (packed.empty()) {
              matrix.col_op(owner->second, column);
              pivot = matrix.col_last(column);
          } else {
              for (auto row : matrix.column(owner->second)) packed.flip(row);
              // The pivot only moves down, so total scanning is linear in
              // the packed column's span over the entire reduction.
              while (pivot != -1 && !packed.test(pivot)) --pivot;
          }
          if (store_operations) slave.col_op(owner->second, column);
          ++additions;
      }
      if (!packed.empty()) {
          matrix_column_t<Matrix> entries;
          entries.reserve(packed.count());
          for (auto row = packed.find_first(); row != bitset::npos; row = packed.find_next(row))
              entries.push_back(static_cast<matrix_index_t<Matrix>>(row));
          matrix.set_col(column, std::move(entries));
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
  void visit_bigraded_columns_MPFREE(Grid_scheduler<Index, Coordinate>& scheduler, Queues& queues,
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

// Full kernel representatives in the original input domain. Mutable input is
// reduced in place and its legacy compressed-grid fields are populated.
template<typename index, typename Storage>
auto R2GradedSparseMatrix<index, Storage>::graded_kernel_MPFREE() -> R2GradedSparseMatrix {
    this->pivots.clear();
    GRLINA_ASSERT(this->col_degrees.size() == this->get_num_cols());
    GRLINA_ASSERT(this->row_degrees.size() == this->get_num_rows());
    vec<index> column_permutation = this->compute_grid_representation();
    Grid_scheduler<index> grid(*this);

    vec<std::priority_queue<index, vec<index>, std::greater<index>>> pq_row(this->y_grid.size());
    // This is the "slave" matrix in mpfree
    SparseBase col_operations = SparseBase(this->get_num_cols(), this->get_num_cols(), "Identity");

    std::vector<r2degree> new_degrees; // Basis for the free module which is part of the kernel
    Storage new_cols; // representing the matrix given by the kernel

    std::vector<bool> indices_in_kernel(this->get_num_cols(), false);

    // Initialize grid scheduler for processing degrees in order

    visit_bigraded_columns_MPFREE(grid, pq_row, z2_col_degrees,
        [&](index i, pair<index> grade, auto notify) {
            reduce_graded_column_MPFREE(*this, col_operations, this->pivots, i, notify);
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
