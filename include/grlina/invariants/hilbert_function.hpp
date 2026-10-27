/** @file invariants/hilbert_function.hpp @brief Hilbert values, grids and Euler evaluation. */
#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>
#include <grlina/module/module_base.hpp>

namespace graded_linalg {

template <typename Degree, typename Index>
struct HilbertValue { Degree degree; Index dimension; };

template <typename Index>
struct R2HilbertGrid {
    vec<double> x_grid;
    vec<double> y_grid;
    array<Index> values;
    Index maximum = 0;
};

template <typename Matrix, typename Visitor>
void visit_euler_degrees(const ChainComplex<Matrix>& resolution, Visitor visit) {
    using index = matrix_index_t<Matrix>;
    GRLINA_DEBUG_CHECK(if (resolution.empty()) throw std::logic_error("Euler characteristic requires a resolution"));
    for (const auto& degree : resolution[0].row_degrees) visit(geometric_degree(resolution[0], degree), index{1});
    index sign = -1;
    for (const auto& differential : resolution.differentials()) {
        for (const auto& degree : differential.col_degrees) visit(geometric_degree(differential, degree), sign);
        sign = -sign;
    }
}

template <typename Matrix>
matrix_index_t<Matrix> euler_dimension_at(
    const ChainComplex<Matrix>& resolution, const matrix_geometry_degree_t<Matrix>& location) {
    matrix_index_t<Matrix> result = 0;
    visit_euler_degrees(resolution, [&](const auto& degree, auto sign) {
        if (Degree_traits<matrix_geometry_degree_t<Matrix>>::smaller_equal(degree, location)) result += sign;
    });
    return result;
}

/** Signed birth histogram followed by a two-dimensional prefix sum.
 * O(B(log X + log Y) + XY), with B the total number of free summands.
 * lower_bound also handles grids that omit some resolution birth coordinates.
 */
template <typename Matrix>
std::vector<std::vector<matrix_index_t<Matrix>>> euler_grid_r2(
    const ChainComplex<Matrix>& resolution, const std::vector<double>& xs,
    const std::vector<double>& ys) {
    using index = matrix_index_t<Matrix>;
    std::vector<std::vector<index>> values(xs.size(), std::vector<index>(ys.size(), 0));
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

/** Sweep arbitrary R2 query points in x and keep signed y-prefix counts in
 * a Fenwick tree. Preserve the caller's order and repeated query points.
 * O((B+Q) log(B+Q)) time, O(B+Q) memory; no Cartesian grid is materialized.
 */
template <typename Matrix>
std::vector<matrix_index_t<Matrix>> euler_queries_r2(
    const ChainComplex<Matrix>& resolution,
    const std::vector<matrix_geometry_degree_t<Matrix>>& locations) {
    using index = matrix_index_t<Matrix>;
    using degree_type = matrix_geometry_degree_t<Matrix>;
    std::vector<index> result(locations.size(), 0);
    if (locations.empty()) return result;
    std::vector<std::pair<degree_type, index>> events;
    visit_euler_degrees(resolution, [&](const auto& degree, index sign) { events.emplace_back(degree, sign); });
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.first[0] < b.first[0];
    });
    std::vector<double> ys;
    for (const auto& location : locations) ys.push_back(location[1]);
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    std::vector<std::size_t> order(locations.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return locations[a][0] < locations[b][0]; });
    std::vector<index> tree(ys.size() + 1, 0);
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

/** Dimension of the module fibre, using a complete resolution when available. */
template <typename Matrix>
matrix_index_t<Matrix> dimension_at(
    const Module<Matrix>& module, const matrix_geometry_degree_t<Matrix>& degree) {
    if (module.res_complete_)
        return euler_dimension_at(module.proj_res_, degree);
    const auto& presentation = module.presentation();
    auto local = presentation.map_at_degree_pair(degree, true).first;
    return static_cast<matrix_index_t<Matrix>>(local.coKernel_basis().size());
}

template <typename Matrix>
std::vector<HilbertValue<matrix_geometry_degree_t<Matrix>, matrix_index_t<Matrix>>>
hilbert_function(const Module<Matrix>& module,
                 const std::vector<matrix_geometry_degree_t<Matrix>>& locations) {
    using degree_type = matrix_geometry_degree_t<Matrix>;
    std::vector<HilbertValue<degree_type, matrix_index_t<Matrix>>> result;
    result.reserve(locations.size());
    if constexpr (std::is_same_v<degree_type, r2degree>) {
        if (module.res_complete_) {
            const auto dimensions = euler_queries_r2(module.proj_res_, locations);
            for (std::size_t i = 0; i < locations.size(); ++i) result.push_back({locations[i], dimensions[i]});
            return result;
        }
    }
    for (const auto& degree : locations) result.push_back({degree, dimension_at(module, degree)});
    return result;
}

template <typename Matrix>
std::vector<matrix_geometry_degree_t<Matrix>> support_degrees(const Module<Matrix>& module) {
    using degree_type = matrix_geometry_degree_t<Matrix>;
    auto result = geometric_row_degrees(module.presentation());
    const auto columns = geometric_col_degrees(module.presentation());
    result.insert(result.end(), columns.begin(), columns.end());
    std::sort(result.begin(), result.end(), Degree_traits<degree_type>::lex_lambda());
    result.erase(std::unique(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return Degree_traits<degree_type>::equals(lhs, rhs);
    }), result.end());
    return result;
}

template <typename Matrix>
std::vector<HilbertValue<matrix_geometry_degree_t<Matrix>, matrix_index_t<Matrix>>>
hilbert_function_on_support(const Module<Matrix>& module) {
    return hilbert_function(module, support_degrees(module));
}

/** Query sorted, unique Cartesian axes, which need not contain all births. */
template <typename Matrix>
R2HilbertGrid<matrix_index_t<Matrix>> hilbert_function_on_grid(
    const Module<Matrix>& module, const vec<double>& xs, const vec<double>& ys) {
    using degree_type = matrix_geometry_degree_t<Matrix>;
    using index_type = matrix_index_t<Matrix>;
    static_assert(std::is_same_v<degree_type, r2degree> || matrix_grid_backed_v<Matrix>,
                  "This grid helper is only available for R2 modules");
    if constexpr (matrix_grid_backed_v<Matrix>)
        static_assert(std::tuple_size_v<degree_type> == 2, "Hilbert grid requires two parameters");
    R2HilbertGrid<index_type> result;
    result.x_grid = xs;
    result.y_grid = ys;
    if (module.res_complete_) {
        result.values = euler_grid_r2(module.proj_res_, xs, ys);
        for (const auto& row : result.values)
            for (auto value : row) result.maximum = std::max(result.maximum, value);
        return result;
    }
    // Evaluate the presentation when no complete resolution is stored.
    result.values.assign(xs.size(), vec<index_type>(ys.size(), 0));
    for (std::size_t x = 0; x < xs.size(); ++x) {
        for (std::size_t y = 0; y < ys.size(); ++y) {
            degree_type degree{static_cast<typename degree_type::value_type>(xs[x]),
                               static_cast<typename degree_type::value_type>(ys[y])};
            result.values[x][y] = dimension_at(module, degree);
            result.maximum = std::max(result.maximum, result.values[x][y]);
        }
    }
    return result;
}

/** Hilbert function on the full presentation-induced Cartesian grid. */
template <typename Matrix>
R2HilbertGrid<matrix_index_t<Matrix>> hilbert_function_on_induced_grid(const Module<Matrix>& module) {
    using degree_type = matrix_geometry_degree_t<Matrix>;
    static_assert(std::is_same_v<degree_type, r2degree> || matrix_grid_backed_v<Matrix>,
                  "This helper is only available for R2 modules");
    if constexpr (matrix_grid_backed_v<Matrix>)
        static_assert(std::tuple_size_v<degree_type> == 2, "Hilbert grid requires two parameters");
    vec<double> xs, ys;
    for (const auto& degree : support_degrees(module)) {
        xs.push_back(degree[0]);
        ys.push_back(degree[1]);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    return hilbert_function_on_grid(module, xs, ys);
}

} // namespace graded_linalg
