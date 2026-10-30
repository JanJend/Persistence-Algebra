/** Small owning F2 columns. Three entries fit inline; longer columns own an array.
 * LGPL-3.0-or-later. Layout inspired by mpfree/fast/column.h; indices remain signed.
 */
#pragma once
#include <grlina/storage/traits.hpp>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <ostream>
#include <type_traits>

namespace graded_linalg {

template<class Index>
class CompactColumn {
    static_assert(std::is_integral_v<Index> && std::is_signed_v<Index>);
    static constexpr std::uint32_t heap_flag = std::uint32_t{1} << 31;
    struct Small { std::uint32_t size; Index entries[3]; };
    struct Heap { std::uint32_t size, capacity; Index* entries; };
    union Storage { Small small; Heap heap; Storage() : small{} {} } storage_;
    bool heap() const { return (storage_.small.size & heap_flag) != 0; }

public:
    using value_type = Index;
    using iterator = Index*;
    using const_iterator = const Index*;
    CompactColumn() = default;
    template<class Iterator>
    CompactColumn(Iterator first, Iterator last) { assign(first, last); }
    CompactColumn(std::initializer_list<Index> entries) : CompactColumn(entries.begin(), entries.end()) {}
    CompactColumn(const vec<Index>& entries) : CompactColumn(entries.begin(), entries.end()) {}
    CompactColumn(const CompactColumn& other) : CompactColumn(other.begin(), other.end()) {}
    CompactColumn(CompactColumn&& other) noexcept {
        storage_ = other.storage_;
        other.storage_.small = Small{};
    }
    ~CompactColumn() { if (heap()) delete[] storage_.heap.entries; }
    CompactColumn& operator=(CompactColumn other) noexcept { swap(other); return *this; }
    void swap(CompactColumn& other) noexcept { std::swap(storage_, other.storage_); }
    std::size_t size() const { return storage_.small.size & ~heap_flag; }
    bool empty() const { return size() == 0; }
    std::size_t capacity() const { return heap() ? storage_.heap.capacity : 3; }
    Index* data() { return heap() ? storage_.heap.entries : storage_.small.entries; }
    const Index* data() const { return heap() ? storage_.heap.entries : storage_.small.entries; }
    iterator begin() { return data(); }
    iterator end() { return data() + size(); }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data() + size(); }
    auto rbegin() const { return std::reverse_iterator<const_iterator>(end()); }
    auto rend() const { return std::reverse_iterator<const_iterator>(begin()); }
    Index& operator[](std::size_t i) { return data()[i]; }
    const Index& operator[](std::size_t i) const { return data()[i]; }
    Index& front() { return data()[0]; }
    const Index& front() const { return data()[0]; }
    Index& back() { return data()[size() - 1]; }
    const Index& back() const { return data()[size() - 1]; }
    void reserve(std::size_t count) {
        if (count <= capacity()) return;
        if (count >= heap_flag) throw std::length_error("Compact column exceeds 31-bit entry count");
        auto* entries = new Index[count];
        const auto length = size();
        std::copy(begin(), end(), entries);
        if (heap()) delete[] storage_.heap.entries;
        storage_.heap = Heap{static_cast<std::uint32_t>(length) | heap_flag,
                             static_cast<std::uint32_t>(count), entries};
    }
    void resize(std::size_t count) {
        const auto old_size = size();
        if (heap() && count <= 3 && count <= old_size) {
            Small small{};
            small.size = static_cast<std::uint32_t>(count);
            std::copy_n(begin(), std::min(count, old_size), small.entries);
            delete[] storage_.heap.entries;
            storage_.small = small;
        } else {
            reserve(count);
            if (count > old_size) std::fill(data() + old_size, data() + count, Index{});
            if (heap()) storage_.heap.size = static_cast<std::uint32_t>(count) | heap_flag;
            else storage_.small.size = static_cast<std::uint32_t>(count);
        }
    }
    void clear() { resize(0); }
    void push_back(Index entry) {
        const auto length = size();
        if (length == capacity()) reserve(std::min(std::size_t(heap_flag - 1), capacity() * 2));
        resize(length + 1);
        data()[length] = entry;
    }
    void pop_back() { resize(size() - 1); }
    template<class Iterator>
    void assign(Iterator first, Iterator last) {
        CompactColumn result;
        result.resize(static_cast<std::size_t>(std::distance(first, last)));
        std::copy(first, last, result.begin());
        swap(result);
    }
    iterator erase(iterator first, iterator last) {
        const auto offset = first - begin(), count = last - first;
        std::move(last, end(), first);
        resize(size() - count);
        return begin() + offset;
    }
    iterator erase(iterator position) { return erase(position, position + 1); }
    iterator insert(iterator position, Index entry) {
        const auto offset = position - begin();
        push_back(entry);
        std::move_backward(begin() + offset, end() - 1, end());
        data()[offset] = entry;
        return begin() + offset;
    }
    template<class Range>
    void xor_with(const Range& other) {
        if (size() <= 3 && other.size() <= 3) {
            Index entries[6];
            auto last = std::set_symmetric_difference(begin(), end(), other.begin(), other.end(), entries);
            assign(entries, last);
        } else {
            CompactColumn result;
            result.reserve(size() + other.size());
            std::set_symmetric_difference(begin(), end(), other.begin(), other.end(), std::back_inserter(result));
            result.resize(result.size()); // Return short results to inline storage.
            swap(result);
        }
    }
    friend bool operator==(const CompactColumn& a, const CompactColumn& b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    }
    friend bool operator!=(const CompactColumn& a, const CompactColumn& b) { return !(a == b); }
};
static_assert(sizeof(CompactColumn<int>) == 16, "32-bit compact columns must occupy 16 bytes");

template<class Index>
std::ostream& operator<<(std::ostream& out, const CompactColumn<Index>& column) {
    for (Index row : column) out << row << ' ';
    return out;
}

template<class Index>
using CompactStorage = vec<CompactColumn<Index>>;

template<class Index>
bool operator==(const CompactStorage<Index>& compact, const vec<vec<Index>>& vectors) {
    return compact.size() == vectors.size() && std::equal(compact.begin(), compact.end(), vectors.begin(),
        [](const auto& a, const auto& b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); });
}
template<class Index>
bool operator==(const vec<vec<Index>>& vectors, const CompactStorage<Index>& compact) { return compact == vectors; }

#if defined(GRLINA_USE_COMPACT_COLUMNS) && GRLINA_USE_COMPACT_COLUMNS
template<class Index> using DefaultSparseStorage = CompactStorage<Index>;
#else
template<class Index> using DefaultSparseStorage = vec<vec<Index>>;
#endif

template<class Storage> inline constexpr bool is_compact_storage_v = false;
template<class Index> inline constexpr bool is_compact_storage_v<CompactStorage<Index>> = true;

// Mutable reducers use compact storage only when the input requests it. CSC
// retains vector scratch storage, avoiding expensive edits of contiguous buffers.
template<class Matrix>
using matrix_reduction_storage_t = std::conditional_t<is_compact_storage_v<decltype(std::declval<Matrix>().data)>,
    decltype(std::declval<Matrix>().data), vec<vec<decltype(std::declval<Matrix>().get_num_rows())>>>;

template<class Index>
struct Column_traits<CompactColumn<Index>, Index> {
    template<class Range>
    static void add_to(const Range& source, CompactColumn<Index>& target) { target.xor_with(source); }
    static void set_entry(CompactColumn<Index>& column, Index row) {
        auto position = std::lower_bound(column.begin(), column.end(), row);
        if (position != column.end() && *position == row) column.erase(position);
        else column.insert(position, row);
    }
    static Index last_entry_index(const CompactColumn<Index>& column) { return column.empty() ? -1 : column.back(); }
    static bool is_zero(const CompactColumn<Index>& column) { return column.empty(); }
    static bool is_nonzero_at(const CompactColumn<Index>& column, Index row) {
        return std::binary_search(column.begin(), column.end(), row);
    }
    static bool is_equal(const CompactColumn<Index>& a, const CompactColumn<Index>& b) { return a == b; }
};

// MatrixUtil keeps its existing vector-valued owning interface. Repeated XORs
// operate directly on compact columns; conversions are limited to explicit
// owning access and callbacks that specifically require std::vector.
template<class Index>
struct MatrixStorageTraits<vec<Index>, Index, CompactStorage<Index>>
    : MatrixStorageTraits<CompactColumn<Index>, Index, CompactStorage<Index>> {
    using Storage = CompactStorage<Index>;
    using Base = MatrixStorageTraits<CompactColumn<Index>, Index, Storage>;
    using Base::append_col;
    template<class NewIndex> using rebind_index = CompactStorage<NewIndex>;
    static vec<Index> get_col(const Storage& s, Index i) { return {s[i].begin(), s[i].end()}; }
    static vec<Index> take_col(Storage& s, Index i) { auto result = get_col(s, i); s[i].clear(); return result; }
    static void set_col(Storage& s, Index i, const vec<Index>& column) { s[i].assign(column.begin(), column.end()); }
    static void set_col(Storage& s, Index i, vec<Index>&& column) { set_col(s, i, column); }
    static void append_col(Storage& s, const vec<Index>& column) { s.emplace_back(column); }
    static void append_col(Storage& s, vec<Index>&& column) { append_col(s, column); }
    static void add_column_to(const Storage& s, Index i, vec<Index>& column) { Column_traits<vec<Index>, Index>::add_to(s[i], column); }
    template<class Function>
    static void edit_col(Storage& s, Index i, Function&& f) {
        if constexpr (std::is_invocable_v<Function, CompactColumn<Index>&>) f(s[i]);
        else {
            auto column = get_col(s, i);
            f(column);
            set_col(s, i, column);
        }
    }
    template<class Function>
    static void transform_columns(Storage& s, Function&& f, bool parallel = false) {
        if (parallel) {
#pragma omp parallel for
            for (std::size_t i = 0; i < s.size(); ++i) edit_col(s, static_cast<Index>(i), f);
        } else {
            for (std::size_t i = 0; i < s.size(); ++i) edit_col(s, static_cast<Index>(i), f);
        }
    }
};
} // namespace graded_linalg
