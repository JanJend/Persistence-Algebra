/**
 * @file csc_storage.hpp
 * @brief Compact sparse-column storage with copy-on-write value semantics.
 */
#pragma once

#include <grlina/column_types.hpp>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <utility>

// Define either switch to 0 before including this header to silence that channel.
#ifndef GRLINA_CSC_COMPILE_WARNINGS
#define GRLINA_CSC_COMPILE_WARNINGS 1
#endif
#ifndef GRLINA_CSC_RUNTIME_WARNINGS
#define GRLINA_CSC_RUNTIME_WARNINGS 1
#endif

#if GRLINA_CSC_COMPILE_WARNINGS
#define GRLINA_CSC_SLOW_OPERATION \
    [[deprecated("CSC performance warning: editing columns may move all entries; " \
                 "prefer SparseMatrix for repeated column operations. " \
                 "Define GRLINA_CSC_COMPILE_WARNINGS=0 to acknowledge this cost.")]]
#else
#define GRLINA_CSC_SLOW_OPERATION
#endif

namespace graded_linalg {
namespace detail {

// One notification across all CSC index types and translation units.
inline void warn_csc_slow_operation() {
#if GRLINA_CSC_RUNTIME_WARNINGS
    static const bool warned = [] {
        std::cerr << "CSC performance warning: editing columns may move all entries "
                     "or detach shared storage. Prefer SparseMatrix for repeated "
                     "column operations. Define GRLINA_CSC_RUNTIME_WARNINGS=0 "
                     "to acknowledge this cost.\n";
        return true;
    }();
    (void)warned;
#endif
}

} // namespace detail

/**
 * Borrowed read-only column. Invalidated by mutation or destruction of its
 * storage, like a reference into a vector. Use get_col() for an owning copy.
 */
template<class Index>
class CSCColumnView {
public:
    using const_iterator = typename vec<Index>::const_iterator;
    using value_type = Index;

    CSCColumnView(const_iterator first, const_iterator last)
        : first_(first), last_(last) {}

    const_iterator begin() const { return first_; }
    const_iterator end() const { return last_; }
    auto rbegin() const { return std::reverse_iterator<const_iterator>(last_); }
    auto rend() const { return std::reverse_iterator<const_iterator>(first_); }
    std::size_t size() const { return static_cast<std::size_t>(last_ - first_); }
    bool empty() const { return first_ == last_; }
    const Index& operator[](std::size_t i) const { return first_[i]; }
    const Index& front() const { return *first_; }
    const Index& back() const { return *(last_ - 1); }

private:
    const_iterator first_;
    const_iterator last_;
};

template<class Index>
std::ostream& operator<<(std::ostream& out, CSCColumnView<Index> column) {
    for (Index entry : column) out << entry << ' ';
    return out;
}

/**
 * One offsets vector and one entries vector, shared until a write occurs.
 * Columns may temporarily be unsorted, just as in vector-backed matrices.
 * Operations which require sorted columns retain the usual precondition.
 */
template<class Index>
class CSCStorage {
    struct Buffers {
        vec<std::size_t> offsets{0};
        vec<Index> entries;
    };

    static const std::shared_ptr<Buffers>& empty_buffers() {
        static const auto empty = std::make_shared<Buffers>();
        return empty;
    }

    std::shared_ptr<Buffers> buffers_;

    void detach() {
        if (buffers_.use_count() != 1) {
            if (!buffers_->entries.empty()) detail::warn_csc_slow_operation();
            buffers_ = std::make_shared<Buffers>(*buffers_);
        }
    }

    void replace_column(std::size_t column, const vec<Index>& values) {
        // entries() may have been supplied as the source; consume it before writing.
        if (&values == &buffers_->entries) {
            const vec<Index> copy(values);
            replace_column(column, copy);
            return;
        }
        const std::size_t start = buffers_->offsets[column];
        const std::size_t end = buffers_->offsets[column + 1];
        const std::size_t old_size = end - start;
        detach();
        auto& entries = buffers_->entries;
        if (values.size() > old_size) {
            const std::size_t extra = values.size() - old_size;
            entries.insert(entries.begin() + end, extra, Index{});
            for (std::size_t i = column + 1; i < buffers_->offsets.size(); ++i)
                buffers_->offsets[i] += extra;
        } else if (values.size() < old_size) {
            const std::size_t removed = old_size - values.size();
            entries.erase(entries.begin() + start + values.size(), entries.begin() + end);
            for (std::size_t i = column + 1; i < buffers_->offsets.size(); ++i)
                buffers_->offsets[i] -= removed;
        }
        std::copy(values.begin(), values.end(), entries.begin() + start);
    }

public:
    using value_type = CSCColumnView<Index>;
    using size_type = std::size_t;

    class const_iterator {
        const CSCStorage* storage_ = nullptr;
        std::size_t column_ = 0;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = CSCColumnView<Index>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = void;

        const_iterator() = default;
        const_iterator(const CSCStorage* storage, std::size_t column)
            : storage_(storage), column_(column) {}
        reference operator*() const { return (*storage_)[column_]; }
        reference operator[](difference_type n) const { return *(*this + n); }
        const_iterator& operator++() { ++column_; return *this; }
        const_iterator operator++(int) { auto old = *this; ++*this; return old; }
        const_iterator& operator--() { --column_; return *this; }
        const_iterator operator--(int) { auto old = *this; --*this; return old; }
        const_iterator& operator+=(difference_type n) { column_ += n; return *this; }
        const_iterator& operator-=(difference_type n) { column_ -= n; return *this; }
        friend const_iterator operator+(const_iterator it, difference_type n) { return it += n; }
        friend const_iterator operator+(difference_type n, const_iterator it) { return it += n; }
        friend const_iterator operator-(const_iterator it, difference_type n) { return it -= n; }
        friend difference_type operator-(const_iterator a, const_iterator b) {
            return static_cast<difference_type>(a.column_) - static_cast<difference_type>(b.column_);
        }
        friend bool operator==(const_iterator a, const_iterator b) {
            return a.storage_ == b.storage_ && a.column_ == b.column_;
        }
        friend bool operator!=(const_iterator a, const_iterator b) { return !(a == b); }
        friend bool operator<(const_iterator a, const_iterator b) { return a.column_ < b.column_; }
        friend bool operator>(const_iterator a, const_iterator b) { return b < a; }
        friend bool operator<=(const_iterator a, const_iterator b) { return !(b < a); }
        friend bool operator>=(const_iterator a, const_iterator b) { return !(a < b); }
    };

    CSCStorage() : buffers_(empty_buffers()) {}
    explicit CSCStorage(std::size_t columns) : CSCStorage() { resize(columns); }
    CSCStorage(std::size_t columns, const vec<Index>& fill) : CSCStorage() { resize(columns, fill); }
    CSCStorage(const vec<vec<Index>>& columns) : CSCStorage() { assign_data(columns); }
    CSCStorage(const CSCStorage&) = default;
    CSCStorage& operator=(const CSCStorage&) = default;
    // The already initialized empty buffer leaves moved-from storage valid.
    CSCStorage(CSCStorage&& other) noexcept : buffers_(std::move(other.buffers_)) {
        other.buffers_ = empty_buffers();
    }
    CSCStorage& operator=(CSCStorage&& other) noexcept {
        if (this != &other) {
            buffers_ = std::move(other.buffers_);
            other.buffers_ = empty_buffers();
        }
        return *this;
    }
    CSCStorage& operator=(const vec<vec<Index>>& columns) {
        assign_data(columns);
        return *this;
    }

    CSCStorage(vec<std::size_t> offsets, vec<Index> entries) : CSCStorage() {
        if (offsets.empty() || offsets.front() != 0 || offsets.back() != entries.size()
            || !std::is_sorted(offsets.begin(), offsets.end()))
            throw std::invalid_argument("Invalid CSC column offsets");
        buffers_ = std::make_shared<Buffers>(Buffers{std::move(offsets), std::move(entries)});
    }

    std::size_t size() const { return buffers_->offsets.size() - 1; }
    bool empty() const { return size() == 0; }
    const vec<std::size_t>& offsets() const { return buffers_->offsets; }
    const vec<Index>& entries() const { return buffers_->entries; }
    bool shares_storage_with(const CSCStorage& other) const { return buffers_ == other.buffers_; }
    friend bool operator==(const CSCStorage& a, const CSCStorage& b) {
        return a.shares_storage_with(b)
            || (a.offsets() == b.offsets() && a.entries() == b.entries());
    }
    friend bool operator!=(const CSCStorage& a, const CSCStorage& b) { return !(a == b); }

    CSCColumnView<Index> operator[](std::size_t i) const {
        return {buffers_->entries.cbegin() + buffers_->offsets[i],
                buffers_->entries.cbegin() + buffers_->offsets[i + 1]};
    }
    CSCColumnView<Index> column(std::size_t i) const { return (*this)[i]; }
    CSCColumnView<Index> front() const { return (*this)[0]; }
    CSCColumnView<Index> back() const { return (*this)[size() - 1]; }
    vec<Index> get_col(std::size_t i) const {
        auto c = (*this)[i];
        return {c.begin(), c.end()};
    }
    const_iterator begin() const { return {this, 0}; }
    const_iterator end() const { return {this, size()}; }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

    void reserve(std::size_t columns) {
        if (buffers_->offsets.capacity() >= columns + 1) return;
        detach();
        buffers_->offsets.reserve(columns + 1);
    }
    void reserve_entries(std::size_t count) {
        if (buffers_->entries.capacity() >= count) return;
        detach();
        buffers_->entries.reserve(count);
    }
    void clear() noexcept { buffers_ = empty_buffers(); }
    void resize(std::size_t columns) {
        if (columns == size()) return;
        detach();
        if (columns < size()) buffers_->entries.resize(buffers_->offsets[columns]);
        buffers_->offsets.resize(columns + 1, buffers_->entries.size());
    }
    void resize(std::size_t columns, const vec<Index>& fill) {
        if (columns <= size() || fill.empty()) { resize(columns); return; }
        reserve(columns);
        while (size() < columns) append_col(fill);
    }
    void pop_back() { resize(size() - 1); }

    void append_col(const vec<Index>& values) {
        if (&values == &buffers_->entries) {
            const vec<Index> copy(values);
            append_col(copy);
            return;
        }
        detach();
        // Reserve the offset first, so a failed allocation cannot leave orphaned entries.
        if (buffers_->offsets.size() == buffers_->offsets.capacity())
            buffers_->offsets.reserve(std::max<std::size_t>(2, 2 * buffers_->offsets.size()));
        buffers_->entries.insert(buffers_->entries.end(), values.begin(), values.end());
        buffers_->offsets.push_back(buffers_->entries.size());
    }
    void append_col(CSCColumnView<Index> values) {
        // A view may refer to our entries, which append can reallocate.
        append_col(vec<Index>(values.begin(), values.end()));
    }
    void push_back(const vec<Index>& values) { append_col(values); }
    void push_back(CSCColumnView<Index> values) { append_col(values); }
    void emplace_back() { append_col(vec<Index>{}); }
    void emplace_back(const vec<Index>& values) { append_col(values); }
    void emplace_back(CSCColumnView<Index> values) { append_col(values); }

    void assign_data(const vec<vec<Index>>& columns) {
        CSCStorage fresh;
        std::size_t total = 0;
        for (const auto& column : columns) total += column.size();
        fresh.reserve(columns.size());
        fresh.reserve_entries(total);
        for (const auto& column : columns) fresh.append_col(column);
        buffers_ = std::move(fresh.buffers_);
    }

    void set_col(std::size_t column, const vec<Index>& values) {
        detail::warn_csc_slow_operation();
        replace_column(column, values);
    }
    void set_col(std::size_t column, CSCColumnView<Index> values) {
        set_col(column, vec<Index>(values.begin(), values.end()));
    }
    void clear_col(std::size_t column) {
        if (!(*this)[column].empty()) set_col(column, vec<Index>{});
    }
    void append_entry(std::size_t column, Index value) {
        if (column + 1 != size()) detail::warn_csc_slow_operation();
        detach();
        buffers_->entries.insert(buffers_->entries.begin() + buffers_->offsets[column + 1], value);
        for (std::size_t i = column + 1; i < buffers_->offsets.size(); ++i)
            ++buffers_->offsets[i];
    }
    void pop_entry(std::size_t column) {
        detail::warn_csc_slow_operation();
        detach();
        buffers_->entries.erase(buffers_->entries.begin() + buffers_->offsets[column + 1] - 1);
        for (std::size_t i = column + 1; i < buffers_->offsets.size(); ++i)
            --buffers_->offsets[i];
    }
    void sort_col(std::size_t column) {
        detach();
        std::sort(buffers_->entries.begin() + buffers_->offsets[column],
                  buffers_->entries.begin() + buffers_->offsets[column + 1]);
    }
    template<class Function>
    void transform_col(std::size_t column, Function&& function) {
        detach();
        for (std::size_t i = buffers_->offsets[column]; i < buffers_->offsets[column + 1]; ++i)
            buffers_->entries[i] = function(buffers_->entries[i]);
    }
    void swap_cols(std::size_t a, std::size_t b) {
        if (a == b) return;
        detail::warn_csc_slow_operation();
        auto first = get_col(a);
        auto second = get_col(b);
        replace_column(a, second);
        replace_column(b, first);
    }

    // new_to_old may be a permutation or a subset; build compact buffers once.
    template<class Indices>
    void select_columns(const Indices& new_to_old) {
        auto result = std::make_shared<Buffers>();
        std::size_t total = 0;
        for (auto i : new_to_old) total += (*this)[i].size();
        result->offsets.reserve(new_to_old.size() + 1);
        result->entries.reserve(total);
        for (auto i : new_to_old) {
            const auto values = (*this)[i];
            result->entries.insert(result->entries.end(), values.begin(), values.end());
            result->offsets.push_back(result->entries.size());
        }
        buffers_ = std::move(result);
    }
    const_iterator erase(const_iterator first, const_iterator last) {
        const std::size_t from = first - begin();
        const std::size_t to = last - begin();
        if (from == to) return {this, from};
        if (to == size()) { resize(from); return end(); }
        detail::warn_csc_slow_operation();
        detach();
        const std::size_t entry_from = buffers_->offsets[from];
        const std::size_t entry_to = buffers_->offsets[to];
        buffers_->entries.erase(buffers_->entries.begin() + entry_from,
                                buffers_->entries.begin() + entry_to);
        buffers_->offsets.erase(buffers_->offsets.begin() + from,
                                buffers_->offsets.begin() + to);
        for (std::size_t i = from; i < buffers_->offsets.size(); ++i)
            buffers_->offsets[i] -= entry_to - entry_from;
        return {this, from};
    }
    const_iterator erase(const_iterator position) { return erase(position, position + 1); }
};

} // namespace graded_linalg
