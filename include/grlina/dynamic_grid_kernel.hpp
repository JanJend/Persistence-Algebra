/** @file dynamic_grid_kernel.hpp
 * @brief NEW: runtime grid kernels; reuses the existing two-axis scheduler.
 */
#pragma once
#include <grlina/sliced_coordinate_kernel.hpp>

namespace graded_linalg {

template<class Scalar, class Index, class Storage>
auto DynamicGridGradedSparseMatrix<Scalar, Index, Storage>::graded_kernel() -> Self {
    validate();
    if constexpr (!std::is_same_v<Storage, vec<vec<Index>>>) {
        auto editable = this->editable_copy();
        return editable.graded_kernel().template to_storage<Storage>();
    } else {
    if (this->parameter_count() != 2) return detail::sliced_grid_kernel(*this);
    this->invalidate_cached_rows();
    auto permutation = compute_grid_representation();
    struct View {
        const Self& matrix;
        const FlatDegreeTable<Index>& z2_col_degrees;
        const vec<Scalar>& x_grid;
        const vec<Scalar>& y_grid;
        Index get_num_cols() const { return matrix.get_num_cols(); }
    } view{*this, this->col_degrees, grids[0], grids[1]};
    Grid_scheduler<Index> scheduler(view);
    vec<std::priority_queue<Index, vec<Index>, std::greater<Index>>> queues(grids[1].size());
    SparseMatrix<Index> operations(this->get_num_cols(), this->get_num_cols(), "Identity");
    std::map<Index, Index> pivots;
    vec<degree_type> degrees;
    array<Index> columns;
    std::vector<bool> in_kernel(this->get_num_cols(), false);
    while (!scheduler.at_end()) {
        auto d = scheduler.next_grade(); auto& queue = queues[d.second];
        auto range = scheduler.index_range_at(d.first, d.second);
        for (Index i = range.first; i < range.second; ++i) queue.push(i);
        while (!queue.empty()) {
            Index i = queue.top(); while (!queue.empty() && queue.top() == i) queue.pop();
            Index p = this->col_last(i);
            while (p != -1 && pivots.count(p)) {
                Index k = pivots.at(p);
                if (k < i) { this->col_op(k, i); operations.col_op(k, i); p = this->col_last(i); }
                else {
                    Index y = this->col_degree(k)[1]; queues[y].push(k); scheduler.notify(d.first, y); break;
                }
            }
            if (p != -1) pivots[p] = i;
            if (!in_kernel[i] && this->is_zero(i)) {
                columns.push_back(operations.get_col(i)); degrees.push_back({d.first, d.second}); in_kernel[i] = true;
                this->clear_col(i); operations.clear_col(i);
            }
        }
    }
    Self result = empty_like(static_cast<Index>(degrees.size()), this->get_num_cols());
    result.assign_data(std::move(columns)); result.col_degrees = degrees; result.row_degrees = this->col_degrees;
    result.permute_rows_graded(permutation); result.sort_columns_lexicographically(); return result;
    }
}

} // namespace graded_linalg
