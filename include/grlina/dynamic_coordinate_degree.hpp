/**
 * @file dynamic_coordinate_degree.hpp
 * @brief NEW: runtime coordinate degrees and contiguous degree tables (C++17).
 *
 * Owning degrees are values; table entries are read-only borrowed views.
 * A table mutation may invalidate its views and iterators. Runtime dimensions
 * belong to each table/matrix, never to a static Degree_traits::poset_id.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <grlina/coordinate_degree.hpp>

namespace graded_linalg {

namespace dynamic_coordinate_detail {

template <typename Scalar>
void validate_coordinate(const Scalar& value) {
    if constexpr (std::is_floating_point_v<Scalar>) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Coordinate degrees require finite coordinates");
    }
}

template <typename Range>
void validate_coordinates(const Range& degree) {
    for (const auto& coordinate : degree) validate_coordinate(coordinate);
}

template <typename Left, typename Right>
void validate_pair(const Left& lhs, const Right& rhs) {
    if (lhs.size() != rhs.size())
        throw std::invalid_argument("Coordinate degrees have different parameter counts");
    validate_coordinates(lhs);
    validate_coordinates(rhs);
}

template <typename Index>
std::size_t checked_index(Index index, std::size_t count) {
    static_assert(std::is_integral_v<Index>, "Degree indices must be integral");
    if constexpr (std::is_signed_v<Index>) {
        if (index < 0) throw std::out_of_range("Negative degree index");
    }
    if (static_cast<std::make_unsigned_t<Index>>(index) >= count)
        throw std::out_of_range("Degree index is outside its table");
    return static_cast<std::size_t>(index);
}

template <typename Scalar>
Scalar sum(const Scalar& lhs, const Scalar& rhs) {
    if constexpr (std::is_integral_v<Scalar>) {
        if constexpr (std::is_signed_v<Scalar>) {
            if ((rhs > 0 && lhs > std::numeric_limits<Scalar>::max() - rhs) ||
                (rhs < 0 && lhs < std::numeric_limits<Scalar>::lowest() - rhs))
                throw std::overflow_error("Coordinate addition overflow");
        } else if (lhs > std::numeric_limits<Scalar>::max() - rhs) {
            throw std::overflow_error("Coordinate addition overflow");
        }
    }
    Scalar result = lhs + rhs;
    validate_coordinate(result);
    return result;
}

template <typename Scalar>
Scalar difference(const Scalar& lhs, const Scalar& rhs) {
    if constexpr (std::is_integral_v<Scalar>) {
        if constexpr (std::is_signed_v<Scalar>) {
            if ((rhs > 0 && lhs < std::numeric_limits<Scalar>::lowest() + rhs) ||
                (rhs < 0 && lhs > std::numeric_limits<Scalar>::max() + rhs))
                throw std::overflow_error("Coordinate subtraction overflow");
        } else if (lhs < rhs) {
            throw std::overflow_error("Coordinate subtraction overflow");
        }
    }
    Scalar result = lhs - rhs;
    validate_coordinate(result);
    return result;
}

} // namespace dynamic_coordinate_detail

template <typename Scalar> class DynamicDegree;

/** NEW: a read-only C++17 view of one contiguous coordinate degree. */
template <typename Scalar>
class CoordinateDegreeView {
    const Scalar* data_ = nullptr;
    std::size_t size_ = 0;

public:
    using value_type = Scalar;
    using const_iterator = const Scalar*;

    CoordinateDegreeView() = default;
    CoordinateDegreeView(const Scalar* data, std::size_t size) : data_(data), size_(size) {
        if (size && !data) throw std::invalid_argument("Null coordinate storage");
    }
    CoordinateDegreeView(const vec<Scalar>& coordinates)
        : CoordinateDegreeView(coordinates.data(), coordinates.size()) {}
    CoordinateDegreeView(const DynamicDegree<Scalar>& degree)
        : CoordinateDegreeView(degree.data(), degree.size()) {}
    // Prevent an immediately dangling view from an owning temporary.
    CoordinateDegreeView(vec<Scalar>&&) = delete;
    CoordinateDegreeView(const vec<Scalar>&&) = delete;
    CoordinateDegreeView(DynamicDegree<Scalar>&&) = delete;
    CoordinateDegreeView(const DynamicDegree<Scalar>&&) = delete;

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    const Scalar* data() const noexcept { return data_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return size_ ? data_ + size_ : data_; }
    const Scalar& operator[](std::size_t coordinate) const {
        if (coordinate >= size_) throw std::out_of_range("Coordinate index is outside its degree");
        return data_[coordinate];
    }
    DynamicDegree<Scalar> to_degree() const;
};

/** NEW: an independently owned coordinate degree with a runtime dimension. */
template <typename Scalar>
class DynamicDegree {
    static_assert(!std::is_same_v<Scalar, bool>, "Boolean vector storage is not contiguous");
    vec<Scalar> coordinates_;

public:
    using value_type = Scalar;
    using iterator = typename vec<Scalar>::iterator;
    using const_iterator = typename vec<Scalar>::const_iterator;

    DynamicDegree() = default;
    explicit DynamicDegree(std::size_t parameters) : coordinates_(parameters) {}
    DynamicDegree(std::initializer_list<Scalar> coordinates) : coordinates_(coordinates) {
        dynamic_coordinate_detail::validate_coordinates(coordinates_);
    }
    explicit DynamicDegree(vec<Scalar> coordinates) : coordinates_(std::move(coordinates)) {
        dynamic_coordinate_detail::validate_coordinates(coordinates_);
    }
    template<std::size_t N>
    DynamicDegree(const std::array<Scalar, N>& coordinates)
        : coordinates_(coordinates.begin(), coordinates.end()) {
        dynamic_coordinate_detail::validate_coordinates(coordinates_);
    }
    // Copy coordinates, so storing a view in an owning-degree container is safe.
    DynamicDegree(CoordinateDegreeView<Scalar> degree) {
        coordinates_.reserve(degree.size());
        for (const auto& value : degree) coordinates_.push_back(value);
        dynamic_coordinate_detail::validate_coordinates(coordinates_);
    }

    std::size_t size() const noexcept { return coordinates_.size(); }
    bool empty() const noexcept { return coordinates_.empty(); }
    const Scalar* data() const noexcept { return coordinates_.data(); }
    const vec<Scalar>& coordinates() const noexcept { return coordinates_; }
    iterator begin() noexcept { return coordinates_.begin(); }
    iterator end() noexcept { return coordinates_.end(); }
    const_iterator begin() const noexcept { return coordinates_.begin(); }
    const_iterator end() const noexcept { return coordinates_.end(); }
    Scalar& operator[](std::size_t coordinate) { return coordinates_.at(coordinate); }
    const Scalar& operator[](std::size_t coordinate) const { return coordinates_.at(coordinate); }
    bool operator==(const DynamicDegree& rhs) const { return coordinates_ == rhs.coordinates_; }
    bool operator!=(const DynamicDegree& rhs) const { return !(*this == rhs); }
};

template <typename Scalar>
DynamicDegree<Scalar> CoordinateDegreeView<Scalar>::to_degree() const {
    return DynamicDegree<Scalar>(*this);
}

/**
 * NEW: degree-major contiguous storage. Degree i begins at i * parameter_count.
 * Zero-dimensional tables retain a logical degree count without coordinates.
 * Write through set/push_back; operator[] deliberately returns a read-only view.
 */
template <typename Scalar>
class FlatDegreeTable {
    static_assert(!std::is_same_v<Scalar, bool>, "Boolean vector storage is not contiguous");
    std::size_t parameters_ = 0;
    std::size_t count_ = 0;
    vec<Scalar> coordinates_;

    std::size_t coordinate_count(std::size_t count) const {
        if (parameters_ && count > vec<Scalar>().max_size() / parameters_)
            throw std::length_error("Coordinate degree table size overflow");
        return count * parameters_;
    }

    template <typename Range>
    vec<Scalar> copy_degree(const Range& degree) const {
        if (degree.size() != parameters_)
            throw std::invalid_argument("Coordinate degree has the wrong parameter count");
        dynamic_coordinate_detail::validate_coordinates(degree);
        vec<Scalar> result;
        result.reserve(parameters_);
        for (const auto& value : degree) result.push_back(value);
        dynamic_coordinate_detail::validate_coordinates(result);
        return result;
    }

public:
    using value_type = CoordinateDegreeView<Scalar>;

    FlatDegreeTable() = default;
    FlatDegreeTable(const FlatDegreeTable&) = default;
    FlatDegreeTable& operator=(const FlatDegreeTable&) = default;
    FlatDegreeTable(FlatDegreeTable&& other) noexcept
        : parameters_(other.parameters_), count_(std::exchange(other.count_, 0)),
          coordinates_(std::move(other.coordinates_)) {}
    FlatDegreeTable& operator=(FlatDegreeTable&& other) noexcept {
        if (this != &other) {
            parameters_ = other.parameters_;
            count_ = std::exchange(other.count_, 0);
            coordinates_ = std::move(other.coordinates_);
        }
        return *this;
    }
    FlatDegreeTable(std::size_t count, std::size_t parameters)
        : parameters_(parameters), count_(count), coordinates_(coordinate_count(count)) {}
    FlatDegreeTable(std::size_t parameters, const vec<DynamicDegree<Scalar>>& degrees)
        : parameters_(parameters) {
        reserve(degrees.size());
        for (const auto& degree : degrees) push_back(degree);
    }
    FlatDegreeTable(std::size_t count, std::size_t parameters, vec<Scalar> coordinates)
        : parameters_(parameters), count_(count), coordinates_(std::move(coordinates)) {
        if (coordinates_.size() != coordinate_count(count))
            throw std::invalid_argument("Flat coordinate buffer has the wrong size");
        dynamic_coordinate_detail::validate_coordinates(coordinates_);
    }
    FlatDegreeTable& operator=(const vec<DynamicDegree<Scalar>>& degrees) {
        FlatDegreeTable replacement(parameters_, degrees);
        *this = std::move(replacement);
        return *this;
    }

    std::size_t size() const noexcept { return count_; }
    bool empty() const noexcept { return count_ == 0; }
    std::size_t parameter_count() const noexcept { return parameters_; }
    const vec<Scalar>& coordinates() const noexcept { return coordinates_; }
    value_type operator[](std::size_t degree) const {
        if (degree >= count_) throw std::out_of_range("Degree index is outside its table");
        return value_type(parameters_ ? coordinates_.data() + degree * parameters_ : nullptr,
                          parameters_);
    }

    template <typename Range>
    void set(std::size_t degree, const Range& value) {
        if (degree >= count_) throw std::out_of_range("Degree index is outside its table");
        // Copy first: value may refer to an overlapping degree in this table.
        const auto copy = copy_degree(value);
        if (parameters_)
            std::copy(copy.begin(), copy.end(), coordinates_.begin() + degree * parameters_);
    }
    void set(std::size_t degree, std::initializer_list<Scalar> value) {
        set(degree, DynamicDegree<Scalar>(value));
    }
    template <typename Range>
    void push_back(const Range& degree) {
        if (count_ == std::numeric_limits<std::size_t>::max())
            throw std::length_error("Coordinate degree count overflow");
        coordinate_count(count_ + 1);
        const auto copy = copy_degree(degree);
        coordinates_.insert(coordinates_.end(), copy.begin(), copy.end());
        ++count_;
    }
    void push_back(std::initializer_list<Scalar> degree) { push_back(DynamicDegree<Scalar>(degree)); }
    void reserve(std::size_t count) { coordinates_.reserve(coordinate_count(count)); }
    void resize(std::size_t count) {
        coordinates_.resize(coordinate_count(count));
        count_ = count;
    }
    void clear() noexcept { coordinates_.clear(); count_ = 0; }

    /** new_to_old[i] is the old degree placed at new index i. */
    template <typename Indices>
    void permute(const Indices& new_to_old) {
        if (new_to_old.size() != count_)
            throw std::invalid_argument("Degree permutation has the wrong size");
        vec<bool> seen(count_, false);
        for (const auto index : new_to_old) {
            const auto old = dynamic_coordinate_detail::checked_index(index, count_);
            if (seen[old]) throw std::invalid_argument("Degree permutation contains duplicate indices");
            seen[old] = true;
        }
        *this = select(new_to_old);
    }

    template <typename Indices>
    void erase_indices(const Indices& indices) {
        std::size_t previous = 0;
        bool first = true;
        for (const auto index : indices) {
            const auto current = dynamic_coordinate_detail::checked_index(index, count_);
            if (!first && current <= previous)
                throw std::invalid_argument("Erased degree indices must be sorted and unique");
            previous = current;
            first = false;
        }
        FlatDegreeTable replacement(0, parameters_);
        replacement.reserve(count_ - indices.size());
        auto erased = indices.begin();
        for (std::size_t i = 0; i < count_; ++i) {
            if (erased != indices.end() && i == static_cast<std::size_t>(*erased)) ++erased;
            else replacement.push_back((*this)[i]);
        }
        *this = std::move(replacement);
    }

    template <typename Indices>
    FlatDegreeTable select(const Indices& indices) const {
        FlatDegreeTable result(0, parameters_);
        result.reserve(indices.size());
        for (const auto index : indices)
            result.push_back((*this)[dynamic_coordinate_detail::checked_index(index, count_)]);
        return result;
    }

    vec<DynamicDegree<Scalar>> to_vector() const {
        vec<DynamicDegree<Scalar>> result;
        result.reserve(count_);
        for (std::size_t i = 0; i < count_; ++i) result.push_back((*this)[i].to_degree());
        return result;
    }

    class const_iterator {
        const FlatDegreeTable* owner_ = nullptr;
        std::size_t index_ = 0;
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = CoordinateDegreeView<Scalar>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = value_type;
        const_iterator() = default;
        const_iterator(const FlatDegreeTable* owner, std::size_t index) : owner_(owner), index_(index) {}
        value_type operator*() const { return (*owner_)[index_]; }
        const_iterator& operator++() { ++index_; return *this; }
        const_iterator operator++(int) { auto old = *this; ++*this; return old; }
        bool operator==(const const_iterator& rhs) const {
            return owner_ == rhs.owner_ && index_ == rhs.index_;
        }
        bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
    };
    const_iterator begin() const noexcept { return const_iterator(this, 0); }
    const_iterator end() const noexcept { return const_iterator(this, count_); }
    bool operator==(const FlatDegreeTable& rhs) const {
        return parameters_ == rhs.parameters_ && count_ == rhs.count_ && coordinates_ == rhs.coordinates_;
    }
    bool operator!=(const FlatDegreeTable& rhs) const { return !(*this == rhs); }
};

/** NEW: product-order operations accepting owned degrees or borrowed views. */
template <typename Scalar>
struct Degree_traits<DynamicDegree<Scalar>,
                     std::void_t<decltype(CoordinatePosetDomain<Scalar>::suffix)>> {
    using degree_type = DynamicDegree<Scalar>;

    static std::string poset_identifier(std::size_t parameters) {
        return std::to_string(parameters) + CoordinatePosetDomain<Scalar>::suffix;
    }
    static std::size_t parse_poset_identifier(const std::string& identifier) {
        const std::string suffix = CoordinatePosetDomain<Scalar>::suffix;
        if (identifier.size() <= suffix.size() ||
            identifier.compare(identifier.size() - suffix.size(), suffix.size(), suffix) != 0)
            throw std::invalid_argument("Coordinate poset identifier has the wrong scalar domain");
        std::size_t parameters = 0;
        for (std::size_t i = 0; i < identifier.size() - suffix.size(); ++i) {
            const char digit = identifier[i];
            if (digit < '0' || digit > '9')
                throw std::invalid_argument("Coordinate poset identifier requires a numeric dimension");
            const auto value = static_cast<std::size_t>(digit - '0');
            if (parameters > (std::numeric_limits<std::size_t>::max() - value) / 10)
                throw std::out_of_range("Coordinate poset dimension overflow");
            parameters = parameters * 10 + value;
        }
        return parameters;
    }

    template <typename Left, typename Right>
    static bool equals(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_coordinates(lhs);
        dynamic_coordinate_detail::validate_coordinates(rhs);
        if (lhs.size() != rhs.size()) return false;
        for (std::size_t i = 0; i < lhs.size(); ++i) if (lhs[i] != rhs[i]) return false;
        return true;
    }
    template <typename Left, typename Right>
    static bool smaller_equal(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_pair(lhs, rhs);
        for (std::size_t i = 0; i < lhs.size(); ++i) if (!(lhs[i] <= rhs[i])) return false;
        return true;
    }
    template <typename Left, typename Right>
    static bool greater_equal(const Left& lhs, const Right& rhs) { return smaller_equal(rhs, lhs); }
    template <typename Left, typename Right>
    static bool smaller(const Left& lhs, const Right& rhs) { return smaller_equal(lhs, rhs) && !equals(lhs, rhs); }
    template <typename Left, typename Right>
    static bool greater(const Left& lhs, const Right& rhs) { return smaller(rhs, lhs); }
    template <typename Left, typename Right>
    static bool lex_order(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_pair(lhs, rhs);
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
        }
        return false;
    }
    template <typename Left, typename Right>
    static bool colex_order(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_pair(lhs, rhs);
        for (std::size_t i = lhs.size(); i-- > 0;) {
            if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
        }
        return false;
    }
    static std::function<bool(const degree_type&, const degree_type&)> lex_lambda() {
        return [](const degree_type& lhs, const degree_type& rhs) { return lex_order(lhs, rhs); };
    }
    static std::function<bool(const degree_type&, const degree_type&)> colex_lambda() {
        return [](const degree_type& lhs, const degree_type& rhs) { return colex_order(lhs, rhs); };
    }
    template <typename Range>
    static vec<double> position(const Range& degree) {
        dynamic_coordinate_detail::validate_coordinates(degree);
        vec<double> result;
        result.reserve(degree.size());
        for (const auto& value : degree) result.push_back(static_cast<double>(value));
        return result;
    }
    template <typename Range>
    static void print_degree(const Range& degree) { write_degree(std::cout, degree); }
    template <typename Left, typename Right>
    static degree_type join(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_pair(lhs, rhs);
        degree_type result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) result[i] = std::max(lhs[i], rhs[i]);
        return result;
    }
    template <typename Left, typename Right>
    static degree_type meet(const Left& lhs, const Right& rhs) {
        dynamic_coordinate_detail::validate_pair(lhs, rhs);
        degree_type result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) result[i] = std::min(lhs[i], rhs[i]);
        return result;
    }
    template <typename OutputStream, typename Range>
    static void write_degree(OutputStream& out, const Range& degree) {
        dynamic_coordinate_detail::validate_coordinates(degree);
        for (std::size_t i = 0; i < degree.size(); ++i) {
            if (i) out << ' ';
            out << degree[i];
        }
    }
    /** Parse all coordinates before ';', leaving the separator unread. */
    template <typename InputStream>
    static degree_type from_stream(InputStream& in) {
        vec<Scalar> coordinates;
        while (in >> std::ws) {
            if (in.peek() == ';') return degree_type(std::move(coordinates));
            Scalar coordinate{};
            if (!(in >> coordinate)) return degree_type();
            if constexpr (std::is_floating_point_v<Scalar>) {
                if (!std::isfinite(coordinate)) {
                    in.setstate(std::ios::failbit);
                    return degree_type();
                }
            }
            coordinates.push_back(coordinate);
        }
        in.setstate(std::ios::failbit); // a degree line requires its separator
        return degree_type();
    }
    template <typename InputStream>
    static degree_type from_stream(InputStream& in, std::size_t parameters) {
        degree_type result = from_stream(in);
        if (result.size() != parameters) in.setstate(std::ios::failbit);
        return result;
    }
    template <typename Range>
    static void add(const Range& amount, degree_type& degree) {
        dynamic_coordinate_detail::validate_pair(amount, degree);
        degree_type result(degree.size());
        for (std::size_t i = 0; i < degree.size(); ++i)
            result[i] = dynamic_coordinate_detail::sum(degree[i], amount[i]);
        degree = std::move(result);
    }
    template <typename Range>
    static void subtract(const Range& amount, degree_type& degree) {
        dynamic_coordinate_detail::validate_pair(amount, degree);
        degree_type result(degree.size());
        for (std::size_t i = 0; i < degree.size(); ++i)
            result[i] = dynamic_coordinate_detail::difference(degree[i], amount[i]);
        degree = std::move(result);
    }
};

// The legacy detector requires a static poset_id. Runtime degrees instead use
// the instance dimension and poset_identifier(parameters), while providing all
// the degree operations required by the generic graded algorithms.
template <typename Scalar>
struct is_degree<DynamicDegree<Scalar>,
                 std::void_t<decltype(CoordinatePosetDomain<Scalar>::suffix)>> : std::true_type {};

template <typename Scalar>
std::ostream& operator<<(std::ostream& out, const DynamicDegree<Scalar>& degree) {
    Degree_traits<DynamicDegree<Scalar>>::write_degree(out, degree);
    return out;
}
template <typename Scalar>
std::ostream& operator<<(std::ostream& out, CoordinateDegreeView<Scalar> degree) {
    Degree_traits<DynamicDegree<Scalar>>::write_degree(out, degree);
    return out;
}

} // namespace graded_linalg
