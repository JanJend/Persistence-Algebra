/**
 * @file graded_linear_system.hpp
 * @author Michael Kerber - adapted to graded_linalg by Jan Jendrysiak
 * @brief Graded kernels, linear systems, images and projected reductions over F2.

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

#include <grlina/matrices/graded.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/r2.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace graded_linalg {

// All graded reduction algorithms live here; matrix headers only declare their
// graded_kernel() entry points. Shared scheduling and column reduction serve
// both kernel generation and linear-system solves. Sections:
//   1. Shared reduction primitives and bigraded event traversal
//   2. Full graded kernels and matrix-method adapters
//   3. Linear systems: known-basis substitution and general graded reduction
//   4. Joint image/projected-kernel reductions and coordinate projections
// Two-parameter routines follow MPfree; the three-parameter kernel below also
// retains the MIT notice for its muphasa-derived algorithm.


// ---------------------------------------------------------------------------
// 1. Shared reduction primitives and bigraded event traversal
// ---------------------------------------------------------------------------

/** Visit colex-sorted bigraded columns using a sorted birth stream and a heap
 * only for join notifications. One deduplicated column heap replaces the map
 * of per-y heaps. Adapted from mpfree's fast Scheduler (LGPL-3.0-or-later).
 */
template<class Index, class Degrees, class Visit>
void visit_bigraded_columns(const Degrees& degrees, Index count, Visit visit) {
    using Coordinate = typename Degrees::value_type::value_type;
    using Grade = std::pair<Coordinate, Coordinate>;
    struct Birth { Grade grade; Index first, last; };
    std::vector<Birth> births;
    std::vector<Index> y_rank(count);
    Index y_count = 0;
    for (Index first = 0; first < count;) {
        Index last = first + 1;
        while (last < count && degrees[last] == degrees[first]) ++last;
        if (first == 0 || degrees[first][1] != degrees[first-1][1]) ++y_count;
        for (Index i = first; i < last; ++i) y_rank[i] = y_count-1;
        births.push_back({{degrees[first][0], degrees[first][1]}, first, last});
        first = last;
    }
    std::sort(births.begin(), births.end(), [](const auto& a, const auto& b) { return a.grade < b.grade; });
    std::priority_queue<Grade, std::vector<Grade>, std::greater<Grade>> joins;
    std::priority_queue<Index, std::vector<Index>, std::greater<Index>> columns;
    std::vector<bool> pending(count, false), notified(y_count, false);
    std::vector<Coordinate> notified_x(y_count);
    auto push = [&](Index column) {
        if (!pending[column]) { pending[column] = true; columns.push(column); }
    };
    std::size_t next = 0;
    while (next < births.size() || !joins.empty()) {
        Grade grade = next == births.size() ? joins.top() : births[next].grade;
        if (!joins.empty() && joins.top() < grade) grade = joins.top();
        if (next < births.size() && births[next].grade == grade) {
            const auto& birth = births[next++];
            for (Index i = birth.first; i < birth.last; ++i) push(i);
        }
        if (!joins.empty() && joins.top() == grade) joins.pop();
        auto notify = [&](Index owner) {
            push(owner);
            if (degrees[owner][1] == grade.second) return;
            GRLINA_ASSERT(degrees[owner][1] > grade.second);
            Index y = y_rank[owner];
            if (!notified[y] || notified_x[y] != grade.first) {
                notified[y] = true;
                notified_x[y] = grade.first;
                joins.push({grade.first, degrees[owner][1]});
            }
        };
        while (!columns.empty() && degrees[columns.top()][1] == grade.second) {
            Index column = columns.top();
            columns.pop();
            pending[column] = false;
            if (!visit(column, grade, notify)) return;
        }
    }
}

// One workspace per reduction traversal, shared by main and payload reductions.
template<class Index>
struct GradedReductionWorkspace {
    vec<Index> column, scratch, operations, operation_scratch;
    bitset packed;
    CompactColumn<Index> compact_operations, compact_scratch, compact_operation_scratch;
};

// MPfree's reduce_column: a helper may displace a later pivot owner and
// reschedule it; a right-hand side only consumes the current helper basis.
template<class Matrix, class Slave, class Pivots, class Index, class Notify>
void reduce_graded_column(Matrix& matrix, Slave& slave, Pivots& pivots,
                          Index column, Notify notify,
                          bool store_operations = true, bool update_pivot = true,
                          GradedReductionWorkspace<matrix_index_t<Matrix>>* workspace = nullptr) {
    if (!workspace) {
        GradedReductionWorkspace<matrix_index_t<Matrix>> local;
        return reduce_graded_column(matrix, slave, pivots, column, notify,
                                    store_operations, update_pivot, &local);
    }
    auto& work = *workspace;
    auto& packed = work.packed;
    packed.clear();
    using Owner = matrix_index_t<Matrix>;
    constexpr bool in_place_matrix = std::is_same_v<matrix_storage_t<Matrix>, array<Owner>> ||
                                     is_compact_storage_v<matrix_storage_t<Matrix>>;
    constexpr bool in_place_slave = std::is_same_v<matrix_storage_t<Slave>, array<Owner>> ||
                                    is_compact_storage_v<matrix_storage_t<Slave>>;
    auto add = [](const auto& source, auto& target, auto& scratch) {
        if (source.empty()) return;
        if constexpr (std::is_same_v<std::decay_t<decltype(target)>, CompactColumn<Owner>>) {
            // Short XORs stay inline; longer ones reuse owning scratch storage.
            if (source.size() <= 3 && target.size() <= 3) target.xor_with(source);
            else {
                scratch.resize(source.size() + target.size());
                auto end = std::set_symmetric_difference(source.begin(), source.end(),
                    target.begin(), target.end(), scratch.begin());
                scratch.resize(static_cast<std::size_t>(end - scratch.begin()));
                target.swap(scratch);
            }
        } else {
            scratch.clear();
            scratch.reserve(source.size() + target.size());
            std::set_symmetric_difference(source.begin(), source.end(), target.begin(), target.end(),
                                         std::back_inserter(scratch));
            target.swap(scratch);
        }
    };
    auto pivot = matrix.col_last(column);
    // Vector and compact columns reduce in place; CSC unpacks only once.
    auto& entries = [&]() -> decltype(auto) {
        if constexpr (in_place_matrix) return (matrix.data[column]);
        else return (work.column);
    }();
    auto& operations = [&]() -> decltype(auto) {
        if constexpr (is_compact_storage_v<matrix_storage_t<Slave>>) {
            if (store_operations) return (slave.data[column]);
            return (work.compact_operations);
        } else {
            if constexpr (in_place_slave)
                if (store_operations) return (slave.data[column]);
            return (work.operations);
        }
    }();
    auto& reduction_scratch = [&]() -> decltype(auto) {
        if constexpr (is_compact_storage_v<matrix_storage_t<Matrix>>) return (work.compact_scratch);
        else return (work.scratch);
    }();
    auto& operation_scratch = [&]() -> decltype(auto) {
        if constexpr (is_compact_storage_v<matrix_storage_t<Slave>>) return (work.compact_operation_scratch);
        else return (work.operation_scratch);
    }();
    auto pivot_owner = [&](auto row) -> Owner {
        if constexpr (std::is_same_v<Pivots, vec<Owner>>) return pivots[row];
        else {
            auto at = pivots.find(row);
            return at == pivots.end() ? Owner(-1) : at->second;
        }
    };
    bool active = false;
    std::size_t additions = 0;
    while (pivot != -1) {
        auto owner = pivot_owner(pivot);
        if (owner == Owner(-1) || owner >= column) break;
        if (!active) {
            if constexpr (in_place_matrix) matrix.is_identity = false;
            else entries = matrix.take_col(column);
            if (store_operations) {
                if constexpr (in_place_slave) slave.is_identity = false;
                else operations = slave.take_col(column);
            }
            active = true;
        }
        // Keep short/sparse reductions sparse. After two additions, a column
        // with >=128 entries and >=1/16 density can avoid repeated merging
        // and allocation by accumulating in bits. Only this column is packed;
        // reducer columns and the operation matrix retain their own storage.
        if (packed.empty() && additions >= 2) {
            if (entries.size() >= 128 && entries.size() >= (std::size_t(pivot) + 16) / 16) {
                packed.resize(std::size_t(pivot) + 1);
                for (auto row : entries) packed.set(row);
            }
        }
        if (packed.empty()) {
            add(matrix.column(owner), entries, reduction_scratch);
            pivot = entries.empty() ? Owner(-1) : entries.back();
        } else {
            for (auto row : matrix.column(owner)) packed.flip(row);
            // The pivot only moves down, so total scanning is linear in
            // the packed column's span over the entire reduction.
            while (pivot != -1 && !packed.test(pivot)) --pivot;
        }
        if (store_operations) add(slave.column(owner), operations, operation_scratch);
        ++additions;
    }
    if (!packed.empty()) {
        entries.clear();
        entries.reserve(packed.count());
        for (auto row = packed.find_first(); row != bitset::npos; row = packed.find_next(row))
            entries.push_back(static_cast<matrix_index_t<Matrix>>(row));
    }
    if (active) {
        if constexpr (!in_place_matrix)
            matrix.set_col(column, std::move(entries));
        if constexpr (!in_place_slave)
            if (store_operations) slave.set_col(column, std::move(operations));
    }
    if (pivot != -1 && update_pivot) {
        auto owner = pivot_owner(pivot);
        if (owner == Owner(-1)) pivots[pivot] = column;
        else if (owner > column) {
            notify(owner);
            pivots[pivot] = column;
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

// ---------------------------------------------------------------------------
// 2. Full graded kernels and matrix-method adapters
// ---------------------------------------------------------------------------

// Two-coordinate kernel reduction (the legacy z2 name is retained). Integer
// and real degrees use the same events without coordinate compression. Only
// column order changes; returned rows retain the original input domain.
template <class Matrix, std::enable_if_t<!std::is_const_v<Matrix>, int> = 0>
Matrix z2_graded_kernel(Matrix& matrix, bool dense_pivots = false) {
    using Index = matrix_index_t<Matrix>;
    using Degree = base_degree<Matrix>;
    static_assert(std::tuple_size_v<Degree> == 2, "Kernel degrees must have two coordinates");

    // Reduction requires colex order: compare y first, then x. Save the
    // sorted-to-original permutation for the coordinates of the kernel vectors.
    auto permutation = matrix.sort_columns_with_permutation(Degree_traits<Degree>::colex_order).new_to_old;
    // All reduction state belongs to this call, rather than the matrix object.
    SparseMatrix<Index, matrix_reduction_storage_t<Matrix>> operations(matrix.get_num_cols(), matrix.get_num_cols(), "Identity");
    matrix_reduction_storage_t<Matrix> columns;
    vec<Degree> degrees;
    std::vector<bool> emitted(matrix.get_num_cols(), false);
    GradedReductionWorkspace<Index> workspace;
    auto traverse = [&](auto& pivots) {
      visit_bigraded_columns(matrix.col_degrees, matrix.get_num_cols(),
        [&](Index column, auto grade, auto notify) {
            reduce_graded_column(matrix, operations, pivots, column, notify, true, true, &workspace);
            if (!emitted[column] && matrix.is_zero(column)) {
                columns.push_back(std::move(operations.data[column]));
                degrees.push_back({grade.first, grade.second});
                emitted[column] = true;
                matrix.clear_col(column);
                operations.clear_col(column);
            }
            return true;
        });
    };
    if (dense_pivots) {
        vec<Index> pivots(matrix.get_num_rows(), Index(-1));
        traverse(pivots);
    } else {
        std::map<Index, Index> pivots;
        traverse(pivots);
    }

    // Kernel vectors live in the input domain (one row per original column),
    // so they form a separate matrix. empty_like preserves optional grid metadata.
    Matrix result = empty_matrix_like(matrix, static_cast<Index>(columns.size()), matrix.get_num_cols());
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
template <class Matrix, std::enable_if_t<!std::is_const_v<Matrix>, int> = 0>
Matrix z3_graded_kernel(Matrix& matrix) {
    using Index = matrix_index_t<Matrix>;
    using Degree = base_degree<Matrix>;
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
            ImageColumn working = is_new ? ImageColumn{matrix.take_col(original), {original}, birth}
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

    Matrix result = empty_matrix_like(matrix, static_cast<Index>(kernel_columns.size()), matrix.get_num_cols());
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



template<typename Scalar, std::size_t Dimension, typename Index, typename Derived, typename Storage>
auto CoordGradedSparseMatrix<Scalar, Dimension, Index, Derived, Storage>::graded_kernel() -> Self {
    Self source = std::move(static_cast<Self&>(*this));
    if constexpr (!std::is_integral_v<Scalar>) {
        GridGradedSparseMatrix<Scalar, Dimension, Index, Storage> grid(source.get_num_cols(), source.get_num_rows());
        grid.set_real_degrees(source.col_degrees, source.row_degrees);
        grid.assign_data(std::move(source.data));
        grid.compatibly_sorted = source.compatibly_sorted;
        grid.is_identity = source.is_identity;
        auto kernel = grid.graded_kernel();
        Self result = source.empty_like(kernel.get_num_cols(), kernel.get_num_rows());
        result.assign_data(std::move(kernel.data));
        result.col_degrees = kernel.real_col_degrees();
        result.row_degrees = kernel.real_row_degrees();
        result.compatibly_sorted = false;
        return result;
    } else if constexpr (Dimension == 2) {
        return z2_graded_kernel(source);
    } else if constexpr (Dimension == 3) {
        return z3_graded_kernel(source);
    } else {
        constexpr std::size_t frozen = Dimension > 2 ? Dimension - 2 : 0;
        Self result = source.empty_like(0, source.get_num_cols());
        result.row_degrees = source.col_degrees;
        std::array<vec<Scalar>, frozen> thresholds;
        for (const auto& degree : source.col_degrees)
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
            auto selected = columns_in_bigraded_slice<Index>(source.col_degrees, grade, frozen);
            if (selected.empty()) return;
            CoordGradedSparseMatrix<Scalar, 2, Index, void, Storage> slice(static_cast<Index>(selected.size()), source.get_num_rows());
            auto project = [&](const degree_type& original) {
                CoordinateDegree<Scalar, 2> degree{};
                for (std::size_t a = 0; a < 2; ++a) if (frozen + a < Dimension) degree[a] = original[frozen + a];
                return degree;
            };
            for (Index i = 0; i < source.get_num_rows(); ++i) slice.row_degrees[i] = project(source.row_degrees[i]);
            for (Index i = 0; i < static_cast<Index>(selected.size()); ++i) {
                slice.col_degrees[i] = project(source.col_degrees[selected[i]]);
                slice.set_col(i, source.get_col(selected[i]));
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

// Legacy real-valued R2 matrices use the same two-coordinate kernel reduction.
template<typename index, typename Storage>
auto R2GradedSparseMatrix<index, Storage>::graded_kernel() -> R2GradedSparseMatrix {
    return z2_graded_kernel(*this);
}

// Optional direct indexing for two-coordinate kernels; other posets retain
// their own kernel algorithm. The default follows the existing matrix entry point.
template<class Matrix>
Matrix graded_kernel_with_pivots(Matrix& matrix, bool dense_pivots = false) {
    if constexpr (is_bigraded_coordinate_degree<base_degree<Matrix>>::value) {
        if (dense_pivots) return z2_graded_kernel(matrix, true);
    }
    return matrix.graded_kernel();
}

// ---------------------------------------------------------------------------
// 3. Linear systems: known-basis substitution and general graded reduction
// ---------------------------------------------------------------------------

/** Prepare fully reduced target-relation bases at the requested source grades.
 * At grade g, use precisely the relation columns born at degrees <= g.
 * The returned bases retain the target row coordinates and can be shared by
 * maps into the same presented target. Repeated grades are prepared only once.
 */
template<class Matrix>
auto prepare_target_relation_reducers(const Matrix& target_relations,
                                     const vec<typename Matrix::degree_type>& source_degrees,
                                     int threads = 1) {
    using Index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    const auto relation_degrees = geometric_col_degrees(target_relations);
    std::map<Degree, SparseMatrix<Index>> reducers;
    vec<typename decltype(reducers)::iterator> jobs;
    for (const auto& grade : source_degrees) {
        auto [it, inserted] = reducers.try_emplace(grade, 0, target_relations.get_num_rows());
        if (inserted) jobs.push_back(it);
    }
    parallel_for(jobs.size(), threads, [&](std::size_t i) {
        auto& [grade, reducer] = *jobs[i];
        for (Index j = 0; j < target_relations.get_num_cols(); ++j)
            if (Degree_traits<Degree>::smaller_equal(relation_degrees[j], grade))
                reducer.append_col(target_relations.get_col(j));
        reducer.compute_num_cols();
        reducer.column_reduction();
    });
    return reducers;
}

/** Normalize a map using bases from prepare_target_relation_reducers.
 * The bases must cover every source column grade and use the map's target
 * coordinates. Reuse them to give a whole family consistent normal forms.
 */
template<class Matrix>
void normalize_map_modulo_target_relations(Matrix& map,
        const std::map<typename Matrix::degree_type, SparseMatrix<matrix_index_t<Matrix>>>& reducers) {
    using Index = matrix_index_t<Matrix>;
    const auto source_degrees = geometric_col_degrees(map);
    for (Index c = 0; c < map.get_num_cols(); ++c) {
        const auto& reducer = reducers.at(source_degrees[c]);
        GRLINA_DEBUG_CHECK(if (map.get_num_rows() != reducer.get_num_rows())
            throw std::invalid_argument("Map and target relations require common target coordinates"));
        auto residual = map.get_col(c);
        // Fully reduced columns contain no other pivot coordinates, so
        // the original column determines all required cancellations.
        for (auto row : map.column(c)) {
            auto pivot = reducer.pivots.find(row);
            if (pivot != reducer.pivots.end())
                Column_traits<vec<Index>, Index>::add_to(reducer.column(pivot->second), residual);
        }
        map.set_col(c, std::move(residual));
    }
    map.rows_computed = false;
    map._rows.clear();
    map.pivots.clear();
}

/** Replace a graded map in place by its normal form modulo target relations.
 * For each column of grade g, reduce modulo relation columns of degree <= g.
 * The difference induces the zero map into coker(target_relations), so the
 * induced map is unchanged. Shape, grades and target coordinates are preserved.
 * With fixed target relations this is linear and idempotent; lifts of the same
 * map have the same normal form. Source-relation compatibility is not checked.
 * Both matrices must use the same target generator coordinates and must not alias.
 */
template<class Matrix>
void normalize_map_modulo_target_relations(Matrix& map, const Matrix& target_relations) {
    GRLINA_DEBUG_CHECK(if (map.get_num_rows() != target_relations.get_num_rows())
        throw std::invalid_argument("Map and target relations require common target coordinates"));
    const auto reducers = prepare_target_relation_reducers(target_relations, geometric_col_degrees(map));
    normalize_map_modulo_target_relations(map, reducers);
}

/** An owned, immutable graded basis prepared for repeated coordinate solves.
 * Ambient rows are normalised once; coefficients retain the original basis
 * column order. Each solve still checks the query's row basis and birth grades.
 */
template<class Matrix>
class PreparedGradedBasis {
    using Index = matrix_index_t<Matrix>;
    using Degree = typename Matrix::degree_type;
    Matrix basis_;
    vec<Degree> row_degrees_, column_degrees_;
    vec<Index> rank_, pivots_;
    bool units_ = true;

public:
    explicit PreparedGradedBasis(Matrix basis)
        : basis_(std::move(basis)), row_degrees_(geometric_row_degrees(basis_)),
          column_degrees_(geometric_col_degrees(basis_)), rank_(basis_.get_num_rows()),
          pivots_(basis_.get_num_rows(), Index(-1)) {
        std::iota(rank_.begin(), rank_.end(), Index(0));
        if (!std::is_sorted(row_degrees_.begin(), row_degrees_.end(), Degree_traits<Degree>::colex_order)) {
            auto rows = row_degrees_;
            auto order = sort_and_get_permutation<Degree, Index>(rows, Degree_traits<Degree>::colex_order);
            for (Index i = 0; i < static_cast<Index>(order.size()); ++i) rank_[order[i]] = i;
            basis_.permute_rows_graded(rank_);
        }
        for (Index j = 0; j < basis_.get_num_cols(); ++j) {
            const auto& column = basis_.column(j);
            if (column.empty() || pivots_[column.back()] != Index(-1))
                throw std::invalid_argument("Graded basis solve requires distinct nonzero pivots");
            pivots_[column.back()] = j;
            units_ = units_ && column.size() == 1;
        }
    }

    std::optional<Matrix> solve(Matrix B) const {
        if (geometric_row_degrees(B) != row_degrees_)
            throw std::invalid_argument("Graded basis solve has different target bases");
        auto result = empty_matrix_like(basis_, B.get_num_cols(), basis_.get_num_cols());
        auto b_degrees = geometric_col_degrees(B);
        set_geometric_degrees(result, b_degrees, column_degrees_);
        vec<Index> residual, coordinates, scratch;
        for (Index j = 0; j < B.get_num_cols(); ++j) {
            residual.clear();
            for (Index row : B.column(j)) residual.push_back(rank_[row]);
            coordinates.clear();
            if (units_) {
                for (Index row : residual) {
                    Index owner = pivots_[row];
                    if (owner == Index(-1) || !Degree_traits<Degree>::smaller_equal(column_degrees_[owner], b_degrees[j]))
                        return std::nullopt;
                    coordinates.push_back(owner);
                }
            } else {
                std::sort(residual.begin(), residual.end());
                while (!residual.empty()) {
                    Index owner = pivots_[residual.back()];
                    if (owner == Index(-1) || !Degree_traits<Degree>::smaller_equal(column_degrees_[owner], b_degrees[j]))
                        return std::nullopt;
                    coordinates.push_back(owner);
                    const auto& reducer = basis_.column(owner);
                    if (reducer.size() == 1) residual.pop_back();
                    else {
                        scratch.clear();
                        std::set_symmetric_difference(reducer.begin(), reducer.end(),
                            residual.begin(), residual.end(), std::back_inserter(scratch));
                        residual.swap(scratch);
                    }
                }
            }
            std::sort(coordinates.begin(), coordinates.end());
            result.set_col(j, coordinates);
        }
        result.compatibly_sorted = false;
        return result;
    }
};

/** One-shot coordinate solve. Move a basis into PreparedGradedBasis to prepare
 * it once for several solves. A query outside its graded span returns nullopt.
 */
template<class Matrix>
std::optional<Matrix> solve_graded_system_by_back_substitution(const Matrix& A, Matrix B) {
    return PreparedGradedBasis<Matrix>(A).solve(std::move(B));
}

// MPfree reduction on integer bidegrees, already sorted by (y, x).
// original_columns records indices in the original [A | B], even for a slice.
template<class Matrix, bool StoreOperations, class Index, class Storage>
bool reduce_bigraded_system(SparseMatrix<Index, Storage>& helper,
                            const vec<std::array<Index, 2>>& degrees,
                            const vec<Index>& original_columns, Index a_count,
                            std::size_t remaining,
                            std::conditional_t<StoreOperations, Matrix, std::nullptr_t>& X) {
    auto coefficients = [&] {
        if constexpr (StoreOperations) {
            SparseMatrix<Index, Storage> result(helper.get_num_cols(), a_count);
            result.resize_data(helper.get_num_cols());
            for (Index i = 0; i < helper.get_num_cols(); ++i)
                if (original_columns[i] < a_count) result.set_col(i, vec<Index>{original_columns[i]});
            return result;
        } else return nullptr;
    }();
    std::map<Index, Index> pivots;
    GradedReductionWorkspace<Index> workspace;
    visit_bigraded_columns(degrees, helper.get_num_cols(), [&](Index column, auto, auto notify) {
        const Index original = original_columns[column];
        const bool target = original >= a_count;
        if constexpr (StoreOperations)
            reduce_graded_column(helper, coefficients, pivots, column, notify, true, !target, &workspace);
        else
            reduce_graded_column(helper, helper, pivots, column, notify, false, !target, &workspace);
        if (target) {
            if (!helper.is_zero(column)) return false;
            if constexpr (StoreOperations) X.move_col_from(original - a_count, coefficients, column);
            return --remaining != 0;
        }
        if constexpr (StoreOperations)
            if (helper.is_zero(column)) coefficients.clear_col(column);
        return true;
    });
    return remaining == 0;
}

/** Two-parameter solver: align grids, then run MPfree directly on integer degrees.
 * Lvalues are copied; pass std::move(A), std::move(B) to consume their storage.
 * The result has the input matrix type and retains both original domain bases.
 * Coordinate inputs build a single common grid on the working [A | B] matrix.
 * Grid inputs reuse their integer degrees after merge_grids (a no-op if equal).
 */
template<class Matrix, bool StoreOperations = true>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_bigraded_linear_system(Matrix A, Matrix B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using Index = matrix_index_t<Matrix>;
    using GeometricDegree = typename Matrix::degree_type;
    using Coordinate = typename GeometricDegree::value_type;
    using StoredDegree = base_degree<Matrix>;
    static_assert(std::tuple_size<GeometricDegree>::value == 2, "Expected two-parameter matrices");

    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(A, B);
    GRLINA_DEBUG_CHECK(if (A.get_num_rows() != B.get_num_rows() || A.row_degrees != B.row_degrees)
        throw std::invalid_argument("Graded linear system has different target bases"));
    const Index a_count = A.get_num_cols();
    auto X = [&] {
        if constexpr (StoreOperations) {
            auto result = empty_matrix_like(A, B.get_num_cols(), a_count);
            result.row_degrees = A.col_degrees;
            result.col_degrees = B.col_degrees;
            return result;
        } else return nullptr;
    }();

    // Ignore zero RHS columns and A columns outside the targets' bounding box.
    vec<Index> targets;
    StoredDegree required{};
    for (Index j = 0; j < B.get_num_cols(); ++j) {
        if (B.column(j).empty()) continue;
        required = targets.empty() ? B.col_degrees[j] : Degree_traits<StoredDegree>::join(required, B.col_degrees[j]);
        targets.push_back(j);
    }
    if (!targets.empty()) {
        auto order = columns_in_bigraded_slice<Index>(A.col_degrees, required, 2);
        for (Index j : targets) order.push_back(a_count + j);
        auto degree = [&](Index original) -> const StoredDegree& {
            return original < a_count ? A.col_degrees[original] : B.col_degrees[original - a_count];
        };
        // Stable sorting keeps A before B at equal degrees.
        std::stable_sort(order.begin(), order.end(), [&](Index left, Index right) {
            return Degree_traits<StoredDegree>::colex_order(degree(left), degree(right));
        });
        vec<StoredDegree> degrees;
        degrees.reserve(order.size());
        for (Index original : order) degrees.push_back(degree(original));

        GridGradedSparseMatrix<Coordinate, 2, Index, matrix_reduction_storage_t<Matrix>> helper(static_cast<Index>(order.size()), A.get_num_rows());
        if constexpr (matrix_grid_backed_v<Matrix>) {
            helper.grid_ = std::move(A.grid_);
            helper.row_degrees = std::move(A.row_degrees);
            helper.col_degrees = std::move(degrees);
        } else {
            helper.set_real_degrees(degrees, A.row_degrees);
        }
        for (Index i = 0; i < helper.get_num_cols(); ++i) {
            const Index original = order[i];
            if (original < a_count) helper.move_col_from(i, A, original);
            else helper.move_col_from(i, B, original - a_count);
        }
        if (!reduce_bigraded_system<Matrix, StoreOperations>(
                helper, helper.col_degrees, order, a_count, targets.size(), X)) {
            if constexpr (StoreOperations) return std::nullopt;
            else return false;
        }
    }
    if constexpr (StoreOperations) {
        X.compatibly_sorted = false;
        return X;
    } else return true;
}

// Reparameterization with an arbitrary left block: maintain A's helper basis
// with the kernel's queued reduction, and reduce only B as target columns.
template<class Matrix, bool StoreOperations = true>
bool solve_bigraded_slice(const Matrix& A, const Matrix& B,
                         const vec<typename Matrix::degree_type>& a_degrees,
                         const vec<typename Matrix::degree_type>& b_degrees,
                         const vec<matrix_index_t<Matrix>>& sources,
                         const vec<matrix_index_t<Matrix>>& targets,
                         std::size_t frozen_axes,
                         std::conditional_t<StoreOperations, Matrix, std::nullptr_t>& X) {
    using index = matrix_index_t<Matrix>;
    using Scalar = typename Matrix::degree_type::value_type;
    const index a_count = static_cast<index>(sources.size());
    const index count = a_count + static_cast<index>(targets.size());
    const std::size_t dimension = b_degrees[targets.front()].size();
    vec<std::array<Scalar, 2>> projected(count);
    for (index i = 0; i < count; ++i) {
        const auto& degree = i < a_count ? a_degrees[sources[i]] : b_degrees[targets[i-a_count]];
        for (std::size_t axis = 0; axis < 2; ++axis)
            if (frozen_axes + axis < dimension) projected[i][axis] = degree[frozen_axes + axis];
    }
    vec<index> order(count);
    std::iota(order.begin(), order.end(), index{0});
    std::stable_sort(order.begin(), order.end(), [&](index left, index right) {
        return projected[left][1] < projected[right][1] ||
              (projected[left][1] == projected[right][1] && projected[left][0] < projected[right][0]);
    });
    std::array<vec<Scalar>, 2> axes;
    for (const auto& degree : projected)
        for (std::size_t axis = 0; axis < 2; ++axis) axes[axis].push_back(degree[axis]);
    for (auto& axis : axes) {
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    }
    vec<std::array<index, 2>> grades(count);
    array<index> columns(count);
    for (index i = 0; i < count; ++i) {
        const index original = order[i];
        for (std::size_t axis = 0; axis < 2; ++axis)
            grades[i][axis] = static_cast<index>(std::lower_bound(
                axes[axis].begin(), axes[axis].end(), projected[original][axis]) - axes[axis].begin());
        columns[i] = original < a_count ? A.get_col(sources[original]) : B.get_col(targets[original-a_count]);
    }
    SparseMatrix<index> helper(count, A.get_num_rows());
    helper.assign_data(std::move(columns));
    for (auto& original : order)
        original = original < a_count ? sources[original] : A.get_num_cols() + targets[original - a_count];
    return reduce_bigraded_system<Matrix, StoreOperations>(
        helper, grades, order, A.get_num_cols(), targets.size(), X);
}


/** Solve A X = B in the original bases, with degree-admissible coefficients.
 * Uses the same column reduction and queued grade traversal as graded_kernel,
 * without computing a kernel. A columns update the helper; B columns never
 * become reducers, and their tracked coefficients give X when they reach zero.
 * Two-parameter systems consume owned arguments; move inputs to avoid copying.
 * They dispatch directly to solve_bigraded_linear_system.
 * Other coordinate dimensions reuse the kernel's two-parameter restrictions
 * at the frozen prefixes requested by B, without enumerating a full kernel.
 * No solution is reported as nullopt. Input compatibility is a caller
 * precondition, checked automatically in diagnostic builds.
 * With StoreOperations=false, return only bool using the same traversal;
 * neither a coefficient matrix nor a solution matrix is constructed.
 */
template <typename Matrix, bool StoreOperations = true,
          std::enable_if_t<std::tuple_size<typename Matrix::degree_type>::value == 2, int> = 0>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_graded_linear_system(Matrix A, Matrix B) {
    return solve_bigraded_linear_system<Matrix, StoreOperations>(std::move(A), std::move(B));
}

template <typename Matrix, bool StoreOperations = true,
          std::enable_if_t<std::tuple_size<typename Matrix::degree_type>::value != 2, int> = 0>
std::conditional_t<StoreOperations, std::optional<Matrix>, bool>
solve_graded_linear_system(const Matrix& A, const Matrix& B) {
    static_assert(is_graded_sparse_matrix_v<Matrix>, "Expected graded CRTP matrices");
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    using Scalar = typename D::value_type;
    static_assert(has_degree_join_v<D>, "The coordinate graded solver requires Degree_traits::join(a, b)");

    GRLINA_DEBUG_CHECK(if (!same_geometric_rows(A, B))
        throw std::invalid_argument("Graded linear system has different target bases"));
    auto X = [&] {
        if constexpr (StoreOperations) return empty_matrix_like(A, B.get_num_cols(), A.get_num_cols());
        else return nullptr;
    }();
    auto a_degrees = geometric_col_degrees(A);
    auto b_degrees = geometric_col_degrees(B);
    if constexpr (StoreOperations) set_geometric_degrees(X, b_degrees, a_degrees);
    std::map<vec<Scalar>, vec<index>> by_prefix;
    std::size_t frozen_axes = 0;
    if (!b_degrees.empty() && b_degrees.front().size() > 2) frozen_axes = b_degrees.front().size()-2;
    for (index j = 0; j < B.get_num_cols(); ++j) {
        if (B.column(j).empty()) continue;
        vec<Scalar> prefix(b_degrees[j].begin(), b_degrees[j].begin()+frozen_axes);
        by_prefix[std::move(prefix)].push_back(j);
    }
    for (const auto& [prefix, targets] : by_prefix) {
        // Helpers outside the requested RHS box cannot contribute. The first
        // frozen coordinates are equal throughout this group; the remaining
        // coordinates vary during the shared two-parameter traversal.
        D required = b_degrees[targets.front()];
        for (index j : targets) required = Degree_traits<D>::join(required, b_degrees[j]);
        auto sources = columns_in_bigraded_slice<index>(a_degrees, required, required.size());
        if (!solve_bigraded_slice<Matrix, StoreOperations>(
                A, B, a_degrees, b_degrees, sources, targets, frozen_axes, X)) {
            if constexpr (StoreOperations) return std::nullopt;
            else return false;
        }
    }
    if constexpr (StoreOperations) {
        X.compatibly_sorted = false;
        return X;
    } else return true;
}

template <typename D> struct is_coordinate_solver_degree : std::false_type {};
template <typename Scalar, std::size_t Dimension>
struct is_coordinate_solver_degree<std::array<Scalar, Dimension>> : std::true_type {};

/** Test solvability without computing coefficients or reordering either basis.
 * Coordinate degrees use the solver's shared traversal without a slave matrix.
 * General degree posets retain the per-degree span test.
 */
template <typename Matrix>
bool graded_linear_system_is_solvable(const Matrix& A, const Matrix& B) {
    using index = matrix_index_t<Matrix>;
    using D = typename Matrix::degree_type;
    if constexpr (is_coordinate_solver_degree<D>::value) {
        return solve_graded_linear_system<Matrix, false>(A, B);
    } else {
        GRLINA_DEBUG_CHECK(if (!same_geometric_rows(A, B))
            throw std::invalid_argument("Graded linear system has different target bases"));
        auto less = Degree_traits<D>::lex_lambda();
        std::map<D, array<index>, decltype(less)> by_degree(less);
        for (index j = 0; j < B.get_num_cols(); ++j)
            by_degree[geometric_degree(B, B.col_degrees[j])].push_back(B.get_col(j));
        for (const auto& [degree, columns] : by_degree) {
            array<index> available;
            for (index k = 0; k < A.get_num_cols(); ++k)
                if (Degree_traits<D>::smaller_equal(geometric_degree(A, A.col_degrees[k]), degree))
                    available.push_back(A.get_col(k));
            SparseMatrix<index> local(static_cast<index>(available.size()), A.get_num_rows(), available);
            SparseMatrix<index> rhs(static_cast<index>(columns.size()), B.get_num_rows(), columns);
            if (!local.solve_col_reduction(rhs)) return false;
        }
        return true;
    }
}

// ---------------------------------------------------------------------------
// 4. Joint image/projected-kernel reductions and coordinate projections
// ---------------------------------------------------------------------------

template<class Matrix>
struct BigradedReduction {
    Matrix image, kernel_image;
    vec<matrix_index_t<Matrix>> kernel_pivots;
    vec<base_degree<Matrix>> kernel_degrees;
    bool pivot_revisited = false;
};

/** Reduction of already colex-ordered columns. original_columns maps their
 * positions to the caller's domain, so discovery and replay can share an order.
 */
template<class Matrix>
BigradedReduction<Matrix> bigraded_reduce_in_colex_order(Matrix D, Matrix P,
        const vec<matrix_index_t<Matrix>>& original_columns, bool want_image = false,
        bool dense_pivots = true) {
    using Index = matrix_index_t<Matrix>;
    using Degree = base_degree<Matrix>;
    static_assert(std::tuple_size_v<typename Matrix::degree_type> == 2, "Expected two parameters");
    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(D, P);
    if (D.get_num_cols() != P.get_num_cols() || D.col_degrees != P.col_degrees)
        throw std::invalid_argument("Joint graded reduction requires the same domain");
    if (original_columns.size() != static_cast<std::size_t>(D.get_num_cols()))
        throw std::invalid_argument("Joint graded reduction has an invalid column order");
    GRLINA_DEBUG_CHECK(if (!std::is_sorted(D.col_degrees.begin(), D.col_degrees.end(), Degree_traits<Degree>::colex_order))
        throw std::invalid_argument("Joint graded reduction requires colex-ordered columns"));
    BigradedReduction<Matrix> result{empty_matrix_like(D, Index(0), D.get_num_rows()),
                                     empty_matrix_like(P, Index(0), P.get_num_rows())};
    result.image.row_degrees = D.row_degrees;
    result.kernel_image.row_degrees = P.row_degrees;
    vec<Index> pending(D.get_num_cols(), Index(-1));
    vec<bool> emitted(D.get_num_cols(), false);
    GradedReductionWorkspace<Index> workspace;
    auto traverse = [&](auto& pivots) {
      visit_bigraded_columns(D.col_degrees, D.get_num_cols(),
        [&](Index column, auto grade, auto notify) {
            if (emitted[column]) return true;
            Degree birth{grade.first, grade.second};
            if (want_image && D.col_degrees[column] != birth) {
                if (pending[column] != Index(-1)) {
                    result.image.set_col(pending[column], D.get_col(column));
                    pending[column] = Index(-1);
                }
            }
            reduce_graded_column(D, P, pivots, column, [&](Index owner) {
                result.pivot_revisited = true;
                notify(owner);
            }, P.get_num_rows() != 0, true, &workspace);
            if (D.is_zero(column)) {
                emitted[column] = true;
                result.kernel_pivots.push_back(original_columns[column]);
                result.kernel_degrees.push_back(birth);
                if (!P.is_zero(column)) {
                    if constexpr (matrix_grid_backed_v<Matrix>)
                        result.kernel_image.append_column_at_grid_degree({}, birth);
                    else result.kernel_image.append_column({}, birth);
                    result.kernel_image.move_col_from(result.kernel_image.get_num_cols() - 1, P, column);
                    P.clear_col(column);
                }
            } else if (want_image && D.col_degrees[column] == birth && pending[column] == Index(-1)) {
                pending[column] = result.image.get_num_cols();
                if constexpr (matrix_grid_backed_v<Matrix>) result.image.append_column_at_grid_degree({}, birth);
                else result.image.append_column({}, birth);
            }
            return true;
        });
    };
    if (dense_pivots) {
        vec<Index> pivots(D.get_num_rows(), Index(-1));
        traverse(pivots);
    } else {
        std::map<Index, Index> pivots;
        traverse(pivots);
    }
    for (Index j = 0; j < D.get_num_cols(); ++j)
        if (pending[j] != Index(-1)) result.image.move_col_from(pending[j], D, j);
    result.image.compatibly_sorted = std::is_sorted(result.image.row_degrees.begin(),
        result.image.row_degrees.end(), Degree_traits<Degree>::lex_order);
    result.kernel_image.compatibly_sorted = std::is_sorted(result.kernel_image.row_degrees.begin(),
        result.kernel_image.row_degrees.end(), Degree_traits<Degree>::lex_order);
    return result;
}

/** Reduce D while carrying P through the same operations. Return generators of
 * P(ker D), optionally also semi-minimal generators of im D. Inputs have the
 * same graded domain and independent targets. Both are consumed; output rows
 * retain their original target bases.
 */
template<class Matrix>
BigradedReduction<Matrix> bigraded_reduce(Matrix D, Matrix P, bool want_image = false,
                                        bool dense_pivots = true) {
    using Degree = base_degree<Matrix>;
    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(D, P);
    if (D.get_num_cols() != P.get_num_cols() || D.col_degrees != P.col_degrees)
        throw std::invalid_argument("Joint graded reduction requires the same domain");
    auto order = D.sort_columns_with_permutation(Degree_traits<Degree>::colex_order).new_to_old;
    P.permute_columns_graded(order);
    return bigraded_reduce_in_colex_order(std::move(D), std::move(P), order, want_image, dense_pivots);
}

/** Semi-minimal generators of a bigraded matrix image, in its original rows. */
template<class Matrix>
Matrix bigraded_image_generators(Matrix matrix) {
    auto auxiliary = empty_matrix_like(matrix, matrix.get_num_cols(), matrix_index_t<Matrix>(0));
    auxiliary.col_degrees = matrix.col_degrees;
    return bigraded_reduce(std::move(matrix), std::move(auxiliary), true).image;
}

/** Reduce [left right], projecting its kernel onto one domain summand. */
template<class Matrix>
BigradedReduction<Matrix> bigraded_reduce_pair(Matrix left, Matrix right,
                                              bool project_left, bool want_image = false,
                                              bool dense_pivots = true) {
    using Index = matrix_index_t<Matrix>;
    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(left, right);
    if (left.get_num_rows() != right.get_num_rows() || left.row_degrees != right.row_degrees)
        throw std::invalid_argument("Mixed reduction requires the same target");
    auto selected_degrees = project_left ? left.col_degrees : right.col_degrees;
    Index offset = project_left ? 0 : left.get_num_cols();
    left.append_move_matrix(std::move(right));
    auto projection = empty_matrix_like(left, left.get_num_cols(), static_cast<Index>(selected_degrees.size()));
    projection.col_degrees = left.col_degrees;
    projection.row_degrees = std::move(selected_degrees);
    for (Index j = 0; j < projection.get_num_rows(); ++j) projection.set_col(offset+j, vec<Index>{j});
    return bigraded_reduce(std::move(left), std::move(projection), want_image, dense_pivots);
}

/** Generators of map^-1(im submodule), modulo the supplied target relations.
 * Over F2, D = [map submodule presentation]. Only the map-domain coordinates
 * of ker D are needed. Carrying their projection through the same reduction
 * avoids constructing the other coordinates of a full kernel basis.
 */
template<class Matrix>
Matrix inverse_image_generators(const Matrix& map, const Matrix& presentation,
                                const Matrix& submodule, bool projected = false,
                                bool dense_pivots = false) {
    if constexpr (is_bigraded_coordinate_degree<base_degree<Matrix>>::value) {
        if (projected) {
            Matrix right = submodule;
            right.append_matrix(presentation);
            return bigraded_reduce_pair(map, std::move(right), true, false, dense_pivots).kernel_image;
        }
    }
    if (dense_pivots) {
        Matrix combined = map;
        const auto domain = combined.get_num_cols();
        combined.append_matrix(submodule);
        combined.append_matrix(presentation);
        auto result = graded_kernel_with_pivots(combined, true);
        result.cull_columns(domain, false);
        return result;
    } else {
        return map.inverse_image_copy(presentation, submodule);
    }
}

template<class Matrix>
struct ProjectedBigradedKernel {
    // If Z is the kernel basis chosen by this reduction and pi_S retains the
    // domain coordinates S, basis = pi_S * Z. These are compact coordinates,
    // not full columns in the original domain of D.
    Matrix basis;
    // Original domain coordinates retained as rows of basis, in original order.
    vec<matrix_index_t<Matrix>> coordinates;
};

/** Retain the given ambient coordinates in their supplied order. */
template<class Matrix>
Matrix project_rows(Matrix matrix, const vec<matrix_index_t<Matrix>>& coordinates) {
    using Index = matrix_index_t<Matrix>;
    vec<Index> rank(matrix.get_num_rows(), Index(-1));
    vec<base_degree<Matrix>> degrees;
    for (Index i = 0; i < static_cast<Index>(coordinates.size()); ++i) {
        Index row = coordinates[i];
        if (row < 0 || row >= matrix.get_num_rows() || rank[row] != Index(-1))
            throw std::invalid_argument("Row projection requires distinct valid coordinates");
        rank[row] = i;
        degrees.push_back(matrix.row_degrees[row]);
    }
    matrix.transform_columns([&](auto& column) {
        std::size_t kept = 0;
        for (Index row : column) if (rank[row] != Index(-1)) column[kept++] = rank[row];
        column.resize(kept);
        std::sort(column.begin(), column.end());
    });
    matrix.set_num_rows(static_cast<Index>(coordinates.size()));
    matrix.row_degrees = std::move(degrees);
    matrix.compatibly_sorted = false;
    matrix.pivots.clear();
    return matrix;
}

/** Represent ker(D) using a subset S of its original domain coordinates.
 * Returns basis = pi_S*Z and coordinates = S, where columns of Z form a graded
 * kernel basis. pi_S is injective on ker(D), so these compact vectors determine
 * every kernel vector uniquely; arbitrary coordinate subsets would not work.
 * Full Z is not constructed. Discover kernel signatures without operation
 * vectors, then replay with only the selected coordinates.
 * If no pivot was displaced, the projected basis is a permutation identity.
 * Used for compact target-cycle coordinates in chromatic's homology map.
 * Consumes D. Copy explicitly beforehand to retain the input; afterwards only
 * destruction or reassignment of D is supported.
 */
template<class Matrix, std::enable_if_t<!std::is_const_v<Matrix>, int> = 0>
ProjectedBigradedKernel<Matrix> projected_bigraded_kernel(Matrix& D) {
    using Index = matrix_index_t<Matrix>;
    Matrix differential = std::move(D);
    auto order = differential.sort_columns_with_permutation(Degree_traits<base_degree<Matrix>>::colex_order);
    auto auxiliary = empty_matrix_like(differential, differential.get_num_cols(), Index(0));
    auxiliary.col_degrees = differential.col_degrees;
    auto sorted_degrees = differential.col_degrees;
    // Discovery may request a replay. Keep one explicit working copy for it;
    // the caller's input has already been consumed.
    auto discovery = bigraded_reduce_in_colex_order(differential, std::move(auxiliary), order.new_to_old);
    auto coordinates = discovery.kernel_pivots;
    std::sort(coordinates.begin(), coordinates.end());
    auto projection = empty_matrix_like(differential, differential.get_num_cols(), static_cast<Index>(coordinates.size()));
    projection.col_degrees = std::move(sorted_degrees);
    projection.row_degrees.clear();
    vec<Index> selected(differential.get_num_cols(), Index(-1));
    for (Index i = 0; i < static_cast<Index>(coordinates.size()); ++i) {
        selected[coordinates[i]] = i;
        projection.row_degrees.push_back(differential.col_degrees[order.old_to_new[coordinates[i]]]);
        projection.set_col(order.old_to_new[coordinates[i]], vec<Index>{i});
    }
    Matrix basis = [&] {
        if (discovery.pivot_revisited) {
            return bigraded_reduce_in_colex_order(std::move(differential), std::move(projection), order.new_to_old).kernel_image;
        }
        auto identity = empty_matrix_like(projection, static_cast<Index>(coordinates.size()), projection.get_num_rows());
        identity.row_degrees = projection.row_degrees;
        identity.col_degrees = std::move(discovery.kernel_degrees);
        for (Index j = 0; j < identity.get_num_cols(); ++j)
            identity.set_col(j, vec<Index>{selected[discovery.kernel_pivots[j]]});
        return identity;
    }();
    return {std::move(basis), std::move(coordinates)};
}

template<class Matrix>
struct ProjectedBigradedKernelImage {
    // Both matrices refer to the SAME kernel basis Z: basis = pi_S*Z and
    // image = F*Z. Image columns stay paired, including zero/dependent images.
    Matrix basis, image;
    vec<matrix_index_t<Matrix>> coordinates;
};

/** Compute compact coordinates for ker(D), and apply F to the same cycles.
 * F and D have the same domain. Returns basis = pi_S*Z, image = F*Z and
 * coordinates = S. This returns the images of Z's basis columns, not an
 * independently minimised generating set of im(F|ker(D)).
 * Carry [pi_S; F] through D's column operations instead of carrying a full
 * identity matrix and multiplying F*Z afterwards. Full Z is not constructed.
 * In chromatic, F = pi_T*f_n: the output is used to express mapped source
 * cycles in the compact target kernel basis and obtain the induced map.
 * Consumes both D and F. Copy explicitly beforehand to retain either input;
 * afterwards only destruction or reassignment of the inputs is supported.
 */
template<class Matrix, std::enable_if_t<!std::is_const_v<Matrix>, int> = 0>
ProjectedBigradedKernelImage<Matrix> projected_bigraded_kernel_image(Matrix& D, Matrix& map) {
    using Index = matrix_index_t<Matrix>;
    auto differential = std::move(D);
    auto F = std::move(map);
    if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(differential, F);
    if (differential.col_degrees != F.col_degrees || differential.get_num_cols() != F.get_num_cols())
        throw std::invalid_argument("Kernel image requires maps with the same domain");
    auto order = differential.sort_columns_with_permutation(Degree_traits<base_degree<Matrix>>::colex_order);
    F.permute_columns_graded(order.new_to_old);
    auto auxiliary = empty_matrix_like(differential, differential.get_num_cols(), Index(0));
    auxiliary.col_degrees = differential.col_degrees;
    auto discovery = bigraded_reduce_in_colex_order(differential, std::move(auxiliary), order.new_to_old);
    auto coordinates = discovery.kernel_pivots;
    std::sort(coordinates.begin(), coordinates.end());
    Index rank = static_cast<Index>(coordinates.size());
    auxiliary = empty_matrix_like(F, F.get_num_cols(), rank + F.get_num_rows());
    auxiliary.col_degrees = F.col_degrees;
    auxiliary.row_degrees.clear();
    vec<Index> selected(F.get_num_cols(), Index(-1));
    for (Index i = 0; i < rank; ++i) {
        selected[coordinates[i]] = i;
        auxiliary.row_degrees.push_back(F.col_degrees[order.old_to_new[coordinates[i]]]);
    }
    auxiliary.row_degrees.insert(auxiliary.row_degrees.end(), F.row_degrees.begin(), F.row_degrees.end());
    for (Index j = 0; j < F.get_num_cols(); ++j) {
        vec<Index> column;
        if (selected[order.new_to_old[j]] != Index(-1)) column.push_back(selected[order.new_to_old[j]]);
        for (Index row : F.column(j)) column.push_back(rank + row);
        auxiliary.set_col(j, column);
    }
    auto joined = bigraded_reduce_in_colex_order(std::move(differential), std::move(auxiliary), order.new_to_old).kernel_image;
    vec<Index> source_rows(rank), target_rows(F.get_num_rows());
    std::iota(source_rows.begin(), source_rows.end(), Index(0));
    std::iota(target_rows.begin(), target_rows.end(), rank);
    auto basis = project_rows(joined, source_rows);
    auto image = project_rows(std::move(joined), target_rows);
    return {std::move(basis), std::move(image), std::move(coordinates)};
}

} // namespace graded_linalg
