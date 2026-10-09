/**
 * @file graded_matrix_algorithms.hpp
 * @brief NEW: graded algorithms using degree accessors, without owning degrees.
 *
 * This additive CRTP layer is independent of the legacy GradedSparseMatrix's
 * vec<D> storage. Derived supplies row_degree/col_degree, degree setters and
 * permutations, and its sparse backend. No virtual degree access is needed.
 */
#pragma once

#include <grlina/graded_matrix.hpp>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace graded_linalg {

template<class D, class Index, class Derived, class DegreeView>
class GradedMatrixAlgorithms {
    template<class, class, class, class> friend class GradedMatrixAlgorithms;
    using DT = Degree_traits<D>;
    Derived& matrix() { return static_cast<Derived&>(*this); }
    const Derived& matrix() const { return static_cast<const Derived&>(*this); }

    struct LexOrder {
        bool operator()(DegreeView a, DegreeView b) const { return DT::lex_order(a, b); }
    };
    struct ColexOrder {
        bool operator()(DegreeView a, DegreeView b) const { return DT::colex_order(a, b); }
    };

    template<class Compare>
    vec<Index> sorting_permutation(bool rows, Compare compare) const {
        const auto& m = matrix();
        vec<Index> order(rows ? m.get_num_rows() : m.get_num_cols());
        std::iota(order.begin(), order.end(), Index(0));
        std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) {
            return rows ? compare(m.row_degree(a), m.row_degree(b))
                        : compare(m.col_degree(a), m.col_degree(b));
        });
        return order;
    }

    static vec<Index> inverse(const vec<Index>& permutation) {
        vec<Index> result(permutation.size());
        for (Index i = 0; i < static_cast<Index>(permutation.size()); ++i)
            result[permutation[i]] = i;
        return result;
    }

protected:
    std::function<bool(DegreeView, DegreeView)> compatible_order_;

public:
    // Certification is checked against actual degree views on reads, including
    // edits made through the public flat degree tables.
    bool compatibly_sorted = false;
    array<Index> col_batches;
    Index k_max = 0;
    vec<Index> rel_k, gen_k;

    void compute_col_batches(bool statistics = false) {
        require_compatibly_sorted("compute_col_batches");
        col_batches.clear(); k_max = 0; rel_k.clear(); gen_k.clear();
        for (Index i = 0; i < matrix().get_num_cols(); ++i) {
            if (i == 0 || !DT::equals(matrix().col_degree(i-1), matrix().col_degree(i)))
                col_batches.push_back({});
            col_batches.back().push_back(i);
            k_max = std::max(k_max, static_cast<Index>(col_batches.back().size()));
        }
        if (statistics) {
            rel_k.resize(static_cast<std::size_t>(k_max) + 1);
            for (const auto& batch : col_batches) ++rel_k[batch.size()];
        }
    }

    struct SortingPermutation {
        vec<Index> old_to_new;
        vec<Index> new_to_old;
    };

    template<class Compare>
    bool degrees_are_sorted(Compare compare) const {
        const auto& m = matrix();
        for (Index i = 1; i < m.get_num_rows(); ++i)
            if (compare(m.row_degree(i), m.row_degree(i - 1))) return false;
        for (Index i = 1; i < m.get_num_cols(); ++i)
            if (compare(m.col_degree(i), m.col_degree(i - 1))) return false;
        return true;
    }
    bool degrees_are_compatibly_sorted() const { return degrees_are_sorted(LexOrder{}); }

    template<class Compare>
    void require_linear_extension(Compare compare) const {
        if constexpr (std::is_same_v<Compare, LexOrder> ||
                      std::is_same_v<Compare, ColexOrder> ||
                      std::is_same_v<Compare, TraitLinearOrder<D>>) return;
        const auto& m = matrix();
        vec<DegreeView> degrees;
        auto append = [&](DegreeView degree) {
            for (auto existing : degrees) {
                if (DT::equals(degree, existing)) {
                    if (compare(degree, existing) || compare(existing, degree))
                        throw std::invalid_argument("Degree order distinguishes equal degrees");
                    return;
                }
            }
            degrees.push_back(degree);
        };
        for (Index i = 0; i < m.get_num_rows(); ++i) append(m.row_degree(i));
        for (Index i = 0; i < m.get_num_cols(); ++i) append(m.col_degree(i));
        vec<std::size_t> predecessors(degrees.size(), 0);
        for (std::size_t i = 0; i < degrees.size(); ++i) {
            if (compare(degrees[i], degrees[i]))
                throw std::invalid_argument("Degree order is not strict");
            for (std::size_t j = 0; j < degrees.size(); ++j) {
                if (i != j && compare(degrees[i], degrees[j]) == compare(degrees[j], degrees[i]))
                    throw std::invalid_argument("Degree comparator is not a strict total order");
                if (i != j && DT::smaller_equal(degrees[i], degrees[j]) &&
                    !compare(degrees[i], degrees[j]))
                    throw std::invalid_argument("Degree order is not a linear extension");
                if (compare(degrees[j], degrees[i])) ++predecessors[i];
            }
        }
        vec<bool> ranks(degrees.size(), false);
        for (auto rank : predecessors) {
            if (rank >= ranks.size() || ranks[rank])
                throw std::invalid_argument("Degree order is not transitive");
            ranks[rank] = true;
        }
    }

    void invalidate_compatible_sorting() noexcept {
        compatibly_sorted = false;
        compatible_order_ = {};
    }
    bool refresh_compatible_sorted() { return refresh_compatible_sorted(LexOrder{}); }
    template<class Compare>
    bool refresh_compatible_sorted(Compare compare) {
        require_linear_extension(compare);
        compatible_order_ = compare;
        compatibly_sorted = degrees_are_sorted(compare);
        return compatibly_sorted;
    }
    template<class Compare>
    void certify_compatible_sorted(Compare compare) {
        require_linear_extension(compare);
        if (!degrees_are_sorted(compare))
            throw std::invalid_argument("Cannot certify unsorted degree lists");
        compatible_order_ = compare;
        compatibly_sorted = true;
    }
    bool compatible_sorting_is_verified() const {
        return compatibly_sorted && compatible_order_ && degrees_are_sorted(compatible_order_);
    }
    void require_compatibly_sorted(const std::string& operation) {
        if (!compatible_sorting_is_verified()) {
            invalidate_compatible_sorting();
            throw std::invalid_argument(operation + " requires compatibly sorted degrees");
        }
    }
    template<class OtherDerived>
    void inherit_compatible_sorting(const GradedMatrixAlgorithms<D, Index, OtherDerived, DegreeView>& other) {
        compatible_order_ = other.compatible_order_;
        compatibly_sorted = other.compatibly_sorted;
        if (compatibly_sorted && !compatible_sorting_is_verified()) invalidate_compatible_sorting();
    }

    template<class Compare>
    SortingPermutation sort_columns_with_permutation(Compare compare) {
        require_linear_extension(compare);
        auto order = sorting_permutation(false, compare);
        auto reverse = inverse(order);
        matrix().permute_columns_graded(order);
        compatible_order_ = compare;
        compatibly_sorted = degrees_are_sorted(compare);
        return {std::move(reverse), std::move(order)};
    }
    SortingPermutation sort_columns_with_permutation() {
        return sort_columns_with_permutation(LexOrder{});
    }
    template<class Compare>
    SortingPermutation sort_rows_with_permutation(Compare compare) {
        require_linear_extension(compare);
        auto order = sorting_permutation(true, compare);
        auto reverse = inverse(order);
        matrix().permute_rows_graded(reverse);
        compatible_order_ = compare;
        compatibly_sorted = degrees_are_sorted(compare);
        return {std::move(reverse), std::move(order)};
    }
    SortingPermutation sort_rows_with_permutation() { return sort_rows_with_permutation(LexOrder{}); }
    template<class Compare> void sort_columns(Compare compare) { sort_columns_with_permutation(compare); }
    template<class Compare> void sort_rows(Compare compare) { sort_rows_with_permutation(compare); }
    template<class Compare>
    void sort_compatibly(Compare compare) {
        sort_columns(compare);
        sort_rows(compare);
    }
    void sort_compatibly() { sort_compatibly(LexOrder{}); }
    void sort_columns_lexicographically() { sort_columns(LexOrder{}); }
    void sort_rows_lexicographically() { sort_rows(LexOrder{}); }
    void sort_columns_colexicographically() { sort_columns(ColexOrder{}); }
    void sort_rows_colexicographically() { sort_rows(ColexOrder{}); }
    void sort_colexicographically() { sort_compatibly(ColexOrder{}); }
    // New API: *_with_output consistently returns new_to_old.
    vec<Index> sort_columns_lexicographically_with_output() {
        return sort_columns_with_permutation(LexOrder{}).new_to_old;
    }
    vec<Index> sort_rows_lexicographically_with_output() {
        return sort_rows_with_permutation(LexOrder{}).new_to_old;
    }
    vec<Index> sort_columns_colexicographically_with_output() {
        return sort_columns_with_permutation(ColexOrder{}).new_to_old;
    }
    vec<Index> sort_rows_colexicographically_with_output() {
        return sort_rows_with_permutation(ColexOrder{}).new_to_old;
    }

    bool is_admissible_column_operation(Index source, Index target) const {
        return source != target && DT::smaller_equal(matrix().col_degree(source), matrix().col_degree(target));
    }
    bool is_admissible_column_operation(Index column, const D& degree) const {
        matrix().check_degree(degree);
        return DT::smaller_equal(matrix().col_degree(column), degree);
    }
    bool is_strictly_admissible_column_operation(Index source, Index target) const {
        return DT::smaller(matrix().col_degree(source), matrix().col_degree(target));
    }
    bool is_admissible_row_operation(Index source, Index target) const {
        return DT::greater_equal(matrix().row_degree(source), matrix().row_degree(target));
    }
    bool is_admissible_row_operation(Index row, const D& degree) const {
        matrix().check_degree(degree);
        return DT::greater_equal(matrix().row_degree(row), degree);
    }
    bool is_admissible_row_operation(const D& degree, Index row) const {
        matrix().check_degree(degree);
        return DT::smaller_equal(matrix().row_degree(row), degree);
    }
    bool is_graded_matrix(bool /*output*/ = false) const {
        const auto& m = matrix();
        for (Index c = 0; c < m.get_num_cols(); ++c)
            for (Index r : m.column(c))
                if (r < 0 || r >= m.get_num_rows() ||
                    !DT::smaller_equal(m.row_degree(r), m.col_degree(c))) return false;
        return true;
    }

    vec<Index> admissible_row_indices(const D& degree) const {
        matrix().check_degree(degree);
        vec<Index> rows;
        for (Index i = 0; i < matrix().get_num_rows(); ++i)
            if (DT::smaller_equal(matrix().row_degree(i), degree)) rows.push_back(i);
        return rows;
    }
    auto map_at_degree_pair(const D& degree, bool shifted = true) const {
        const auto& m = matrix();
        m.validate();
        auto rows = admissible_row_indices(degree);
        vec<Index> columns;
        for (Index i = 0; i < m.get_num_cols(); ++i)
            if (DT::smaller_equal(m.col_degree(i), degree)) columns.push_back(i);
        typename Derived::sparse_matrix_type result(static_cast<Index>(columns.size()),
            shifted ? static_cast<Index>(rows.size()) : m.get_num_rows());
        vec<Index> row_map(m.get_num_rows(), Index(-1));
        for (Index i = 0; i < static_cast<Index>(rows.size()); ++i) row_map[rows[i]] = i;
        for (Index column : columns) {
            auto values = m.get_col(column);
            if (shifted) for (auto& value : values) value = row_map[value];
            result.append_col(std::move(values));
        }
        return std::make_pair(std::move(result), std::move(rows));
    }
    auto map_at_degree(const D& degree, vec<Index>& selected_columns) const {
        const auto& m = matrix();
        m.check_degree(degree);
        selected_columns.clear();
        for (Index i = 0; i < m.get_num_cols(); ++i)
            if (DT::smaller_equal(m.col_degree(i), degree)) selected_columns.push_back(i);
        typename Derived::sparse_matrix_type result(static_cast<Index>(selected_columns.size()), m.get_num_rows());
        for (auto i : selected_columns) result.append_col(m.get_col(i));
        return result;
    }
    Index num_cols_before(const D& degree) const {
        matrix().check_degree(degree);
        Index count = 0;
        for (Index i = 0; i < matrix().get_num_cols(); ++i)
            if (DT::smaller_equal(matrix().col_degree(i), degree)) ++count;
        return count;
    }
    Index num_rows_before(const D& degree) const {
        return static_cast<Index>(admissible_row_indices(degree).size());
    }
    vec<Index> basislift_at(const D& degree) const {
        auto local = map_at_degree_pair(degree);
        auto basis = local.first.coKernel_basis();
        for (auto& index : basis) index = local.second[index];
        return basis;
    }
    Index dim_at(const D& degree) const { return static_cast<Index>(basislift_at(degree).size()); }
    vec<D> discrete_support() const {
        const auto& m = matrix();
        vec<D> support;
        support.reserve(m.get_num_rows() + m.get_num_cols());
        for (Index i = 0; i < m.get_num_rows(); ++i) support.emplace_back(m.row_degree(i));
        for (Index i = 0; i < m.get_num_cols(); ++i) support.emplace_back(m.col_degree(i));
        std::sort(support.begin(), support.end(), DT::lex_lambda());
        support.erase(std::unique(support.begin(), support.end()), support.end());
        return support;
    }
    std::pair<D, D> bounding_box() const {
        auto support = discrete_support();
        if (support.empty()) throw std::invalid_argument("Empty matrix has no degree bounds");
        D low = support.front(), high = low;
        for (const auto& degree : support) {
            low = DT::meet(low, degree);
            high = DT::join(high, degree);
        }
        return {std::move(low), std::move(high)};
    }

    vec<Index> column_reduction_graded() {
        auto& m = matrix();
        require_compatibly_sorted("column_reduction_graded");
        m.validate();
        vec<vec<Index>> pivots(m.get_num_rows());
        vec<Index> nonzero;
        for (Index i = 0; i < m.get_num_cols(); ++i) {
            Index pivot = m.col_last(i);
            while (pivot != -1) {
                auto found = std::find_if(pivots[pivot].begin(), pivots[pivot].end(),
                    [&](Index source) { return is_admissible_column_operation(source, i); });
                if (found == pivots[pivot].end()) break;
                m.col_op(*found, i);
                pivot = m.col_last(i);
            }
            if (pivot != -1) { pivots[pivot].push_back(i); nonzero.push_back(i); }
        }
        m.invalidate_cached_rows();
        return nonzero;
    }
    void column_reduction_graded_w_deletion() {
        auto nonzero = column_reduction_graded();
        vec<Index> remove;
        std::size_t keep = 0;
        for (Index i = 0; i < matrix().get_num_cols(); ++i)
            if (keep < nonzero.size() && nonzero[keep] == i) ++keep;
            else remove.push_back(i);
        matrix().delete_columns(remove);
    }
    void cancel_local_pairs(Derived* elements = nullptr) {
        auto& m = matrix();
        require_compatibly_sorted("cancel_local_pairs");
        m.validate();
        if (elements) {
            elements->validate();
            if (elements == &m || elements->row_degrees != m.row_degrees)
                throw std::invalid_argument("Cancellation requires separate elements with the same generator degrees");
        }
        while (true) {
            Index column = -1, row = -1;
            for (Index c = 0; c < m.get_num_cols() && column == -1; ++c)
                for (Index r : m.column(c))
                    if (DT::equals(m.col_degree(c), m.row_degree(r))) { column = c; row = r; break; }
            if (column == -1) break;
            auto pivot_column = m.get_col(column);
            for (Index c = 0; c < m.get_num_cols(); ++c) {
                auto values = m.column(c);
                if (c != column && std::binary_search(values.begin(), values.end(), row)) m.col_op(column, c);
            }
            if (elements) for (Index c = 0; c < elements->get_num_cols(); ++c) {
                auto values = elements->column(c);
                if (std::binary_search(values.begin(), values.end(), row)) elements->add_to_col(c, pivot_column);
            }
            m.delete_columns(vec<Index>{column});
            m.delete_rows(vec<Index>{row});
            if (elements) elements->delete_rows(vec<Index>{row});
        }
        m.invalidate_cached_rows();
    }
    void semi_minimize() { cancel_local_pairs(); }
    Derived submodule_generated_by(const Derived& generators) const {
        Derived injection = generators;
        const auto count = injection.get_num_cols();
        injection.append_matrix(matrix());
        auto result = injection.graded_kernel();
        result.cull_columns(count, false);
        return result;
    }
    Derived inverse_image_copy(const Derived& presentation, const Derived& submodule) const {
        Derived injection = matrix();
        const auto count = injection.get_num_cols();
        injection.append_matrix(submodule);
        injection.append_matrix(presentation);
        auto result = injection.graded_kernel();
        result.cull_columns(count, false);
        return result;
    }
    Derived submodule_intersection(const Derived& left, const Derived& right) const {
        return left * left.inverse_image_copy(matrix(), right);
    }
    void remove_redundant_relations() {
        remove_redundant_relations([](auto, const auto&) {});
    }
    template<class OnDelete>
    void remove_redundant_relations(OnDelete on_delete) {
        if constexpr (!has_matrix_graded_kernel<Derived>::value) {
            throw std::logic_error("Relation minimization requires a graded kernel; use semi_minimize for local pairs");
        } else {
            auto& m = matrix();
            require_compatibly_sorted("remove_redundant_relations");
            m.validate();
            Derived source = m;
            Derived syzygies = source.graded_kernel();
            while (true) {
                Index c = -1, r = -1;
                for (Index j = 0; j < syzygies.get_num_cols() && c == -1; ++j)
                    for (Index i : syzygies.column(j))
                        if (DT::equals(syzygies.col_degree(j), syzygies.row_degree(i))) { c = j; r = i; break; }
                if (c == -1) break;
                for (Index j = 0; j < syzygies.get_num_cols(); ++j) {
                    auto values = syzygies.column(j);
                    if (j != c && std::binary_search(values.begin(), values.end(), r)) syzygies.col_op(c, j);
                }
                on_delete(r, syzygies.get_col(c));
                m.delete_columns(vec<Index>{r});
                syzygies.delete_rows(vec<Index>{r});
                syzygies.delete_columns(vec<Index>{c});
            }
        }
    }
    void minimize() {
        if constexpr (!has_matrix_graded_kernel<Derived>::value)
            throw std::logic_error("Minimization requires a graded kernel; use semi_minimize for local pairs");
        else { cancel_local_pairs(); remove_redundant_relations(); }
    }
    void minimize_variant() { minimize(); }
    bool is_minimal() const {
        Derived copy = matrix();
        copy.minimize();
        return copy.get_num_cols() == matrix().get_num_cols() && copy.get_num_rows() == matrix().get_num_rows();
    }
    void print_degrees() const {
        std::cout << "Generators at: ";
        for (Index i = 0; i < matrix().get_num_rows(); ++i) {
            DT::write_degree(std::cout, matrix().row_degree(i)); std::cout << "; ";
        }
        std::cout << "\nRelations at: ";
        for (Index i = 0; i < matrix().get_num_cols(); ++i) {
            DT::write_degree(std::cout, matrix().col_degree(i)); std::cout << "; ";
        }
        std::cout << '\n';
    }
    void print_graded(bool suppress_description = false) const {
        matrix().print(suppress_description);
        print_degrees();
    }
};

} // namespace graded_linalg
