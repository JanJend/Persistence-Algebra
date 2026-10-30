/** @file landscape_vineyard.hpp @brief Incremental F2 reduction for parallel presentation slices. */
#pragma once
#include <grlina/matrices/sparse.hpp>

namespace graded_linalg::landscape_detail {

// R = D V, with R reduced and V upper triangular in the current orders.
// Keep coefficients in original row/column IDs: changing an order then only
// changes the affected pivots, not every entry in the swapped rows/columns.
// Adjacent transpositions repair the factorization with at most two additions.
// Reference: Cohen-Steiner, Edelsbrunner, Morozov, "Vines and Vineyards" (2006).
template <typename Index>
class Vineyard {
    SparseMatrix<Index> reduced_, basis_;
    vec<Index> pivots_, lows_, rows_, cols_, row_positions_, col_positions_;

    void add(Index source, Index target) {
        reduced_.col_op(source, target);
        basis_.col_op(source, target);
    }
    void swap_rows(Index position) {
        const auto a = rows_[position], b = rows_[position+1];
        auto left = pivots_[a], right = pivots_[b];
        if (right < 0 || !reduced_.is_nonzero_entry(right, a)) return;
        if (left < 0) {
            pivots_[a] = right; pivots_[b] = -1; lows_[right] = a;
        } else {
            if (col_positions_[left] > col_positions_[right]) std::swap(left, right);
            add(left, right);
            pivots_[a] = left; pivots_[b] = right;
            lows_[left] = a; lows_[right] = b;
        }
    }
    void swap_columns(Index position) {
        const auto a = cols_[position], b = cols_[position+1];
        // Remove the entry that would move below V's diagonal.
        if (!basis_.is_nonzero_entry(b, a)) return;
        add(a, b);
        const auto low_a = lows_[a], low_b = lows_[b];
        const auto rank_a = low_a < 0 ? -1 : row_positions_[low_a];
        const auto rank_b = low_b < 0 ? -1 : row_positions_[low_b];
        if (rank_a > rank_b) {
            // After the transposition b precedes a; this restores distinct lows.
            add(b, a);
            lows_[a] = low_b; lows_[b] = low_a;
            pivots_[low_a] = b;
            if (low_b >= 0) pivots_[low_b] = a;
        }
    }
    template <typename Swap>
    static void reorder(const vec<Index>& target, vec<Index>& order, vec<Index>& positions, Swap swap) {
        for (Index i=0; i<static_cast<Index>(target.size()); ++i)
            for (Index j=positions[target[i]]; j>i; --j) {
                swap(j-1);
                std::swap(order[j-1], order[j]);
                positions[order[j-1]] = j-1; positions[order[j]] = j;
            }
    }

public:
    template <typename Presentation>
    Vineyard(const Presentation& matrix, vec<Index> rows, vec<Index> cols)
        : reduced_(matrix.get_num_cols(), matrix.get_num_rows()),
          basis_(matrix.get_num_cols(), matrix.get_num_cols()), pivots_(rows.size(), -1), lows_(cols.size(), -1),
          rows_(std::move(rows)), cols_(std::move(cols)), row_positions_(rows_.size()), col_positions_(cols_.size()) {
        reduced_.resize_data(cols_.size()); basis_.resize_data(cols_.size());
        for (Index i=0; i<static_cast<Index>(rows_.size()); ++i) row_positions_[rows_[i]] = i;
        SparseMatrix<Index> reduced(matrix.get_num_cols(), matrix.get_num_rows());
        SparseMatrix<Index> basis(matrix.get_num_cols(), matrix.get_num_cols());
        reduced.resize_data(cols_.size()); basis.resize_data(cols_.size());
        for (Index j=0; j<static_cast<Index>(cols_.size()); ++j) {
            col_positions_[cols_[j]] = j;
            auto column = matrix.get_col(cols_[j]);
            for (auto& i : column) i = row_positions_[i];
            std::sort(column.begin(), column.end());
            reduced.set_col(j, std::move(column));
            basis.set_col(j, vec<Index>{j});
        }
        // One initial reduction; all subsequent slices update this factorization.
        reduced.column_reduction_triangular_with_memory(basis);
        for (Index j=0; j<static_cast<Index>(cols_.size()); ++j) {
            const auto p = reduced.col_last(j);
            if (p >= 0) { lows_[cols_[j]] = rows_[p]; pivots_[rows_[p]] = cols_[j]; }
            auto r = reduced.take_col(j), v = basis.take_col(j);
            for (auto& i : r) i = rows_[i];
            for (auto& i : v) i = cols_[i];
            std::sort(r.begin(), r.end()); std::sort(v.begin(), v.end());
            reduced_.set_col(cols_[j], std::move(r)); basis_.set_col(cols_[j], std::move(v));
        }
    }
    void update(const vec<Index>& rows, const vec<Index>& cols) {
        reorder(rows, rows_, row_positions_, [this](Index i) { swap_rows(i); });
        reorder(cols, cols_, col_positions_, [this](Index i) { swap_columns(i); });
    }
    // Original generator -> original relation, with -1 for an infinite death.
    Index death(Index original_row) const { return pivots_[original_row]; }
};

} // namespace graded_linalg::landscape_detail
