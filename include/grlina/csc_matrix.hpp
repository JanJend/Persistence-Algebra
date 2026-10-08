/**
 * @file csc_matrix.hpp
 * @brief CSC implementation of the shared sparse matrix storage primitives.
 *
 * Include this header to opt into CSCMatrix. Existing SparseMatrix users do
 * not include CSC storage or its performance diagnostics.
 */
#pragma once

#include <grlina/csc_storage.hpp>
#include <grlina/matrix_storage.hpp>
#include <numeric>

namespace graded_linalg {

template<class Index>
struct MatrixStorageTraits<vec<Index>, Index, CSCStorage<Index>> {
    using Storage = CSCStorage<Index>;
    using Column = vec<Index>;
    template<class NewIndex> using rebind_index = CSCStorage<NewIndex>;

    static auto column(const Storage& data, Index i) { return data.column(i); }
    static Column get_col(const Storage& data, Index i) { return data.get_col(i); }

    GRLINA_CSC_SLOW_OPERATION
    static Column take_col(Storage& data, Index i) {
        auto result = data.get_col(i);
        data.clear_col(i);
        return result;
    }

    GRLINA_CSC_SLOW_OPERATION
    static void set_col(Storage& data, Index i, const Column& values) {
        data.set_col(i, values);
    }

    GRLINA_CSC_SLOW_OPERATION
    static void set_col(Storage& data, Index i, Column&& values) { data.set_col(i, values); }

    static void append_col(Storage& data, const Column& values) { data.append_col(values); }
    static void append_col(Storage& data, Column&& values) { data.append_col(values); }

    GRLINA_CSC_SLOW_OPERATION
    static void clear_col(Storage& data, Index i) { data.clear_col(i); }

    GRLINA_CSC_SLOW_OPERATION
    static void append_entry(Storage& data, Index i, Index value) { data.append_entry(i, value); }

    GRLINA_CSC_SLOW_OPERATION
    static void pop_entry(Storage& data, Index i) { data.pop_entry(i); }

    GRLINA_CSC_SLOW_OPERATION
    static void swap_cols(Storage& data, Index i, Index j) { data.swap_cols(i, j); }

    GRLINA_CSC_SLOW_OPERATION
    static void col_op(Storage& data, Index source, Index target) {
        const auto a = data.column(source);
        const auto b = data.column(target);
        Column sum;
        sum.reserve(a.size() + b.size());
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                      std::back_inserter(sum));
        data.set_col(target, sum);
    }

    template<class Range>
    GRLINA_CSC_SLOW_OPERATION
    static void add_to_col(Storage& data, Index i, const Range& values) {
        const auto original = data.column(i);
        Column sum;
        sum.reserve(original.size() + values.size());
        std::set_symmetric_difference(original.begin(), original.end(), values.begin(), values.end(),
                                      std::back_inserter(sum));
        data.set_col(i, sum);
    }

    static void add_column_to(const Storage& data, Index i, Column& scratch) {
        const auto source = data.column(i);
        Column sum;
        sum.reserve(source.size() + scratch.size());
        std::set_symmetric_difference(source.begin(), source.end(), scratch.begin(), scratch.end(),
                                      std::back_inserter(sum));
        scratch.swap(sum);
    }

    GRLINA_CSC_SLOW_OPERATION
    static void set_entry(Storage& data, Index i, Index value) {
        auto changed = data.get_col(i);
        Column_traits<Column, Index>::set_entry(changed, value);
        data.set_col(i, changed);
    }

    static bool is_nonzero_entry(const Storage& data, Index i, Index value) {
        const auto values = data.column(i);
        return std::binary_search(values.begin(), values.end(), value);
    }
    static Index col_last(const Storage& data, Index i) {
        const auto values = data.column(i);
        return values.empty() ? Index(-1) : values.back();
    }
    static bool is_zero(const Storage& data, Index i) { return data.column(i).empty(); }
    static bool columns_equal(const Storage& a, Index i, const Storage& b, Index j) {
        const auto ca = a.column(i);
        const auto cb = b.column(j);
        return ca.size() == cb.size() && std::equal(ca.begin(), ca.end(), cb.begin());
    }

    template<class Function>
    GRLINA_CSC_SLOW_OPERATION
    static void edit_col(Storage& data, Index i, Function&& function) {
        auto changed = data.get_col(i);
        function(changed);
        data.set_col(i, changed);
    }

    // Bulk edits rebuild once, avoiding a shift of all following entries per column.
    template<class Function>
    static void transform_columns(Storage& data, Function&& function, bool parallel = false) {
        (void)parallel; // A compact buffer is rebuilt serially; columns cannot resize independently.
        vec<std::size_t> offsets;
        Column entries;
        offsets.reserve(data.size() + 1);
        entries.reserve(data.entries().size());
        offsets.push_back(0);
        Column scratch;
        for (const auto values : data) {
            scratch.assign(values.begin(), values.end());
            function(scratch);
            entries.insert(entries.end(), scratch.begin(), scratch.end());
            offsets.push_back(entries.size());
        }
        data = Storage(std::move(offsets), std::move(entries));
    }

    GRLINA_CSC_SLOW_OPERATION
    static void permute_columns(Storage& data, const vec<Index>& new_to_old) {
        detail::warn_csc_slow_operation();
        data.select_columns(new_to_old);
    }

    GRLINA_CSC_SLOW_OPERATION
    static void erase_columns(Storage& data, const vec<Index>& sorted) {
        if (sorted.empty()) return;
        detail::warn_csc_slow_operation();
        vec<Index> kept;
        kept.reserve(data.size() - sorted.size());
        auto next = sorted.begin();
        for (std::size_t i = 0; i < data.size(); ++i) {
            if (next != sorted.end() && static_cast<std::size_t>(*next) == i) ++next;
            else kept.push_back(static_cast<Index>(i));
        }
        data.select_columns(kept);
    }

    template<class SourceStorage>
    static void assign_transpose(Storage& target, const SourceStorage& source, Index output_cols) {
        vec<std::size_t> offsets(static_cast<std::size_t>(output_cols) + 1, 0);
        for (std::size_t i = 0; i < source.size(); ++i)
            for (Index row : source[i]) ++offsets[static_cast<std::size_t>(row) + 1];
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        Column entries(offsets.back());
        auto next = offsets;
        for (std::size_t i = 0; i < source.size(); ++i)
            for (Index row : source[i]) entries[next[row]++] = static_cast<Index>(i);
        target = Storage(std::move(offsets), std::move(entries));
    }
};

} // namespace graded_linalg

#include <grlina/sparse_matrix.hpp>

namespace graded_linalg {

/**
 * SparseMatrix algorithms with compact CSC buffers. Copying shares the buffers;
 * writes detach before modifying them. Degrees and matrix workspaces have their
 * own ordinary copy semantics. Costly column edits remain supported and warn by
 * default at compile time and once at runtime.
 */
template<class Index>
using CSCMatrix = SparseMatrix<Index, CSCStorage<Index>>;

} // namespace graded_linalg
