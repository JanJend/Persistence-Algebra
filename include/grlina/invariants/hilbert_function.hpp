/** @file invariants/hilbert_function.hpp
 * @brief Module dimensions, induced-grid Hilbert functions and PNG images.
 *
 * Numerical queries never build or change a resolution. A stored complete
 * resolution enables Euler counting; otherwise we reduce local presentations.
 * Images reuse the induced-grid table and sample it at equidistant pixels.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <grlina/module/module_base.hpp>
#include <grlina/invariants/hilbert_plot.hpp>

namespace graded_linalg {
namespace hilbert_detail {

template <typename Degree> inline constexpr bool two_parameters = false;
template <typename Scalar>
inline constexpr bool two_parameters<CoordinateDegree<Scalar, 2>> = true;

} // namespace hilbert_detail

/** A two-parameter Hilbert function tabulated on a Cartesian grid.
 * values[x][y] is the integer dimension at (x_grid[x], y_grid[y]).
 * For an induced/stored grid this represents the full function: use the
 * greatest grid coordinate <= the query on each axis, or zero below an axis.
 * A table on explicitly supplied axes stores only those samples; omitted
 * change coordinates cannot be recovered from that table alone.
 */
template <typename Scalar, typename Index>
struct HilbertFunction {
    vec<Scalar> x_grid;
    vec<Scalar> y_grid;
    array<Index> values;
    Index maximum = 0; // Thickness when this is the full induced-grid table.

    /** Resample by coordinate predecessors, without algebraic computation.
     * Query coordinates can have another scalar type (e.g. fractional pixels
     * for a module with integer degrees). Empty axes produce empty dimensions.
     */
    template <typename QueryScalar>
    HilbertFunction<QueryScalar, Index> on_grid(const vec<QueryScalar>& xs,
                                               const vec<QueryScalar>& ys) const {
        hilbert_detail::validate_axis(xs);
        hilbert_detail::validate_axis(ys);
        hilbert_detail::validate_axis(x_grid);
        hilbert_detail::validate_axis(y_grid);
        if (values.size() != x_grid.size() ||
            std::any_of(values.begin(), values.end(), [&](const auto& row) { return row.size() != y_grid.size(); }))
            throw std::invalid_argument("Hilbert dimensions must match the coordinate axes");
        HilbertFunction<QueryScalar, Index> result{xs, ys};
        result.values.assign(xs.size(), vec<Index>(ys.size(), 0));
        // One binary search per axis coordinate, rather than per pixel.
        vec<std::size_t> x_ends, y_ends;
        for (auto x : xs)
            x_ends.push_back(std::upper_bound(x_grid.begin(), x_grid.end(), x) - x_grid.begin());
        for (auto y : ys)
            y_ends.push_back(std::upper_bound(y_grid.begin(), y_grid.end(), y) - y_grid.begin());
        for (std::size_t x = 0; x < xs.size(); ++x)
            for (std::size_t y = 0; y < ys.size(); ++y)
                if (x_ends[x] && y_ends[y]) {
                    const auto value = values[x_ends[x] - 1][y_ends[y] - 1];
                    result.values[x][y] = value;
                    result.maximum = std::max(result.maximum, value);
                }
        return result;
    }
};

/** Enumerate free-generator births with signs +F0, -F1, +F2, ... .
 * This does not sum at a query point. Matrix accessors decode grid indices.
 */
template <typename Matrix, typename Visitor>
void visit_euler_degrees(const ChainComplex<Matrix>& resolution, Visitor visit) {
    using index = matrix_index_t<Matrix>;
    GRLINA_DEBUG_CHECK(if (resolution.empty()) throw std::logic_error("Euler characteristic requires a resolution"));
    for (index i = 0; i < resolution[0].get_num_rows(); ++i)
        visit(resolution[0].row_degree(i), index{1});
    index sign = -1;
    for (const auto& differential : resolution.differentials()) {
        for (index j = 0; j < differential.get_num_cols(); ++j)
            visit(differential.col_degree(j), sign);
        sign = -sign;
    }
}

/** Evaluate a complete, exact two-parameter resolution on sorted Cartesian axes.
 * Return values[x][y] = dim M(xs[x], ys[y]). Deposit each signed birth at
 * the first sample >= it, then take a 2D prefix sum. Births need not be samples.
 * O(B(log X + log Y) + XY), for B free summands and X, Y axis lengths.
 */
template <typename Matrix, typename Scalar>
array<matrix_index_t<Matrix>> euler_grid_r2(const ChainComplex<Matrix>& resolution,
                                          const vec<Scalar>& xs, const vec<Scalar>& ys) {
    using index = matrix_index_t<Matrix>;
    array<index> values(xs.size(), vec<index>(ys.size(), 0));
    visit_euler_degrees(resolution, [&](const auto& degree, index sign) {
        const auto x = std::lower_bound(xs.begin(), xs.end(), degree[0]) - xs.begin();
        const auto y = std::lower_bound(ys.begin(), ys.end(), degree[1]) - ys.begin();
        if (x < static_cast<std::ptrdiff_t>(xs.size()) && y < static_cast<std::ptrdiff_t>(ys.size()))
            values[x][y] += sign;
    });
    for (std::size_t x = 0; x < xs.size(); ++x)
        for (std::size_t y = 0; y < ys.size(); ++y) {
            if (x) values[x][y] += values[x - 1][y];
            if (y) values[x][y] += values[x][y - 1];
            if (x && y) values[x][y] -= values[x - 1][y - 1];
        }
    return values;
}

/** Sweep arbitrary two-parameter queries in x, maintaining signed y-prefix
 * counts in a Fenwick tree. Preserve query order and duplicates.
 * O((B+Q) log(B+Q)) time, O(B+Q) memory, without a Cartesian table.
 */
template <typename Matrix>
vec<matrix_index_t<Matrix>> euler_queries_r2(const ChainComplex<Matrix>& resolution,
    const vec<typename Matrix::degree_type>& locations) {
    using index = matrix_index_t<Matrix>;
    using degree_type = typename Matrix::degree_type;
    vec<index> result(locations.size(), 0);
    if (locations.empty()) return result;
    vec<std::pair<degree_type, index>> events;
    visit_euler_degrees(resolution, [&](const auto& degree, index sign) { events.emplace_back(degree, sign); });
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.first[0] < b.first[0]; });
    vec<typename degree_type::value_type> ys;
    for (const auto& location : locations) ys.push_back(location[1]);
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    vec<std::size_t> order(locations.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return locations[a][0] < locations[b][0]; });
    vec<index> tree(ys.size() + 1, 0);
    std::size_t next = 0;
    for (auto q : order) {
        while (next < events.size() && events[next].first[0] <= locations[q][0]) {
            auto i = static_cast<std::size_t>(std::lower_bound(ys.begin(), ys.end(), events[next].first[1]) - ys.begin()) + 1;
            for (; i < tree.size(); i += i & -i) tree[i] += events[next].second;
            ++next;
        }
        auto i = static_cast<std::size_t>(std::lower_bound(ys.begin(), ys.end(), locations[q][1]) - ys.begin()) + 1;
        for (; i; i -= i & -i) result[q] += tree[i];
    }
    return result;
}

template <typename Matrix>
auto Module<Matrix>::dimension_at(const degree_type& degree) const -> index_type {
    if (!has_presentation()) return 0;
    if (res_complete_) {
        index_type result = 0;
        visit_euler_degrees(proj_res_, [&](const auto& birth, auto sign) {
            if (Degree_traits<degree_type>::smaller_equal(birth, degree)) result += sign;
        });
        return result;
    }
    auto local = local_presentation_at(degree).first;
    return static_cast<index_type>(local.coKernel_basis().size());
}

template <typename Matrix>
auto Module<Matrix>::dimension_at(const vec<degree_type>& degrees) const -> vec<index_type> {
    if constexpr (hilbert_detail::two_parameters<degree_type>)
        if (res_complete_ && has_presentation()) return euler_queries_r2(proj_res_, degrees);
    vec<index_type> result;
    result.reserve(degrees.size());
    for (const auto& degree : degrees) result.push_back(dimension_at(degree));
    return result;
}

/** Distinct presentation degrees (not the entire mathematical support). */
template <typename Matrix>
vec<typename Matrix::degree_type> support_degrees(const Module<Matrix>& module) {
    using degree_type = typename Matrix::degree_type;
    vec<degree_type> result;
    if (!module.has_presentation()) return result;
    const auto& presentation = module.presentation();
    result.reserve(presentation.get_num_rows() + presentation.get_num_cols());
    for (matrix_index_t<Matrix> i = 0; i < presentation.get_num_rows(); ++i)
        result.push_back(presentation.row_degree(i));
    for (matrix_index_t<Matrix> j = 0; j < presentation.get_num_cols(); ++j)
        result.push_back(presentation.col_degree(j));
    std::sort(result.begin(), result.end(), Degree_traits<degree_type>::lex_lambda());
    result.erase(std::unique(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return Degree_traits<degree_type>::equals(lhs, rhs);
    }), result.end());
    return result;
}

/** Cartesian axes on which the module can change. Grid matrices retain their
 * full stored axes, including coordinates unused by the presentation degrees.
 */
template <typename Matrix>
auto induced_grid(const Module<Matrix>& module) {
    using degree_type = typename Matrix::degree_type;
    static_assert(hilbert_detail::two_parameters<degree_type>, "Hilbert grids require two coordinate parameters");
    using scalar = typename degree_type::value_type;
    std::array<vec<scalar>, 2> axes;
    if (!module.has_presentation()) return axes;
    if constexpr (matrix_grid_backed_v<Matrix>) {
        return module.presentation().coordinate_grid();
    } else {
        for (const auto& degree : support_degrees(module)) {
            axes[0].push_back(degree[0]);
            axes[1].push_back(degree[1]);
        }
        for (auto& axis : axes) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        }
        return axes;
    }
}

template <typename Matrix>
auto Module<Matrix>::hilbert_function() const {
    const auto axes = induced_grid(*this);
    HilbertFunction<typename degree_type::value_type, index_type> result{axes[0], axes[1]};
    if (res_complete_ && has_presentation()) {
        result.values = euler_grid_r2(proj_res_, axes[0], axes[1]);
    } else {
        result.values.assign(axes[0].size(), vec<index_type>(axes[1].size(), 0));
        for (std::size_t x = 0; x < axes[0].size(); ++x)
            for (std::size_t y = 0; y < axes[1].size(); ++y)
                result.values[x][y] = dimension_at(degree_type{axes[0][x], axes[1][y]});
    }
    for (const auto& row : result.values)
        for (auto value : row) result.maximum = std::max(result.maximum, value);
    return result;
}

template <typename Matrix>
template <typename Degree>
auto Module<Matrix>::hilbert_function(const vec<typename Degree::value_type>& xs,
                                     const vec<typename Degree::value_type>& ys) const {
    hilbert_detail::validate_axis(xs);
    hilbert_detail::validate_axis(ys);
    if (xs.empty() || ys.empty()) {
        HilbertFunction<typename Degree::value_type, index_type> result{xs, ys};
        result.values.assign(xs.size(), vec<index_type>(ys.size(), 0));
        return result;
    }
    return hilbert_function().on_grid(xs, ys);
}

template <typename Matrix>
vec<unsigned char> Module<Matrix>::hilbert_function_image(std::size_t width, std::size_t height,
                                                         double padding) const {
    if (!std::isfinite(padding) || padding < 0)
        throw std::invalid_argument("Hilbert image padding must be finite and nonnegative");
    const auto function = hilbert_function();
    auto bounds = [padding](const auto& axis) {
        const long double low = axis.empty() ? 0 : axis.front();
        const long double high = axis.empty() ? 0 : axis.back();
        const auto margin = (high > low ? high - low : 2) * padding;
        vec<long double> result{low - margin};
        if (high + margin > result.front()) result.push_back(high + margin);
        return result;
    };
    return graded_linalg::hilbert_function_image(function, bounds(function.x_grid),
                                                 bounds(function.y_grid), width, height);
}

template <typename Matrix>
template <typename Degree>
vec<unsigned char> Module<Matrix>::hilbert_function_image(const vec<typename Degree::value_type>& xs,
    const vec<typename Degree::value_type>& ys, std::size_t width, std::size_t height) const {
    return graded_linalg::hilbert_function_image(hilbert_function(), xs, ys, width, height);
}

} // namespace graded_linalg
