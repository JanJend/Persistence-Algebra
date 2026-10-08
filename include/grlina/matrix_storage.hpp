#pragma once

#include <grlina/column_types.hpp>
#include <utility>

namespace graded_linalg {

// Storage primitives used by the shared matrix algorithms. COLUMN is the owning
// scratch-column type; a storage backend may return a different read-only view.
template<class COLUMN, class Index, class Storage>
struct MatrixStorageTraits;

template<class COLUMN, class Index>
struct MatrixStorageTraits<COLUMN, Index, vec<COLUMN>> {
    using Storage = vec<COLUMN>;
    using CT = Column_traits<COLUMN, Index>;
    template<class NewIndex> using rebind_index = vec<vec<NewIndex>>;

    static const COLUMN& column(const Storage& s, Index i) { return s[i]; }
    static COLUMN get_col(const Storage& s, Index i) { return s[i]; }
    static COLUMN take_col(Storage& s, Index i) { return std::move(s[i]); }
    static void set_col(Storage& s, Index i, const COLUMN& c) { s[i] = c; }
    static void set_col(Storage& s, Index i, COLUMN&& c) { s[i] = std::move(c); }
    static void append_col(Storage& s, const COLUMN& c) { s.push_back(c); }
    static void append_col(Storage& s, COLUMN&& c) { s.push_back(std::move(c)); }
    static void clear_col(Storage& s, Index i) { s[i].clear(); }
    static void append_entry(Storage& s, Index i, Index entry) { s[i].push_back(entry); }
    static void pop_entry(Storage& s, Index i) { s[i].pop_back(); }
    static void swap_cols(Storage& s, Index i, Index j) { std::swap(s[i], s[j]); }
    static void col_op(Storage& s, Index source, Index target) { CT::add_to(s[source], s[target]); }
    template<class Range>
    static void add_to_col(Storage& s, Index i, const Range& c) { CT::add_to(c, s[i]); }
    static void add_column_to(const Storage& s, Index i, COLUMN& c) { CT::add_to(s[i], c); }
    static void set_entry(Storage& s, Index i, Index entry) { CT::set_entry(s[i], entry); }
    static bool is_nonzero_entry(const Storage& s, Index i, Index entry) { return CT::is_nonzero_at(s[i], entry); }
    static Index col_last(const Storage& s, Index i) { return CT::last_entry_index(s[i]); }
    static bool is_zero(const Storage& s, Index i) { return CT::is_zero(s[i]); }
    static bool columns_equal(Storage& a, Index i, Storage& b, Index j) { return CT::is_equal(a[i], b[j]); }
    template<class Function>
    static void edit_col(Storage& s, Index i, Function&& f) { f(s[i]); }
    template<class Function>
    static void transform_columns(Storage& s, Function&& f, bool parallel = false) {
        if (parallel) {
#pragma omp parallel for
            for (std::size_t i = 0; i < s.size(); ++i) f(s[i]);
        } else {
            for (auto& c : s) f(c);
        }
    }
    static void permute_columns(Storage& s, const vec<Index>& order) {
        Storage result;
        result.reserve(order.size());
        for(auto i : order) result.push_back(std::move(s[i]));
        s = std::move(result);
    }
    static void erase_columns(Storage& s, const vec<Index>& indices) {
        std::size_t next = 0, out = 0;
        for(std::size_t i = 0; i < s.size(); ++i) {
            if(next < indices.size() && i == static_cast<std::size_t>(indices[next])) { ++next; continue; }
            if(out != i) s[out] = std::move(s[i]);
            ++out;
        }
        s.resize(out);
    }
    template<class SourceStorage>
    static void assign_transpose(Storage& s, const SourceStorage& source, Index count) {
        Storage result(count);
        for(std::size_t i = 0; i < source.size(); ++i)
            for(auto row : source[i]) result[row].push_back(static_cast<Index>(i));
        s = std::move(result);
    }
};

} // namespace graded_linalg
