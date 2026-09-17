/** Euler-characteristic Hilbert queries for complete projective resolutions.
 * Completeness/exactness must be established by the owning Module. These
 * routines use chain-group degrees only, never local differential matrices.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <grlina/chain_complex.hpp>

namespace graded_linalg::detail {

template <typename Matrix, typename Visitor>
void visit_euler_degrees(const ChainComplex<Matrix>& resolution, Visitor visit) {
    using index = typename Matrix::index_type;
    if (resolution.empty()) throw std::logic_error("Euler characteristic requires a resolution");
    for (const auto& degree : resolution[0].row_degrees) visit(degree, index{1});
    index sign = -1;
    for (const auto& differential : resolution.differentials()) {
        for (const auto& degree : differential.col_degrees) visit(degree, sign);
        sign = -sign;
    }
}

template <typename Matrix>
typename Matrix::index_type euler_dimension_at(
    const ChainComplex<Matrix>& resolution, const typename Matrix::degree_type& location) {
    typename Matrix::index_type result = 0;
    visit_euler_degrees(resolution, [&](const auto& degree, auto sign) {
        if (Degree_traits<typename Matrix::degree_type>::smaller_equal(degree, location)) result += sign;
    });
    return result;
}

/** Signed birth histogram followed by a two-dimensional prefix sum.
 * O(B(log X + log Y) + XY), with B the total number of free summands.
 * lower_bound also handles grids that omit some resolution birth coordinates.
 */
template <typename Matrix>
std::vector<std::vector<typename Matrix::index_type>> euler_grid_r2(
    const ChainComplex<Matrix>& resolution, const std::vector<double>& xs,
    const std::vector<double>& ys) {
    using index = typename Matrix::index_type;
    std::vector<std::vector<index>> values(xs.size(), std::vector<index>(ys.size(), 0));
    visit_euler_degrees(resolution, [&](const auto& degree, index sign) {
        const auto x = std::lower_bound(xs.begin(), xs.end(), degree.first) - xs.begin();
        const auto y = std::lower_bound(ys.begin(), ys.end(), degree.second) - ys.begin();
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
std::vector<typename Matrix::index_type> euler_queries_r2(
    const ChainComplex<Matrix>& resolution,
    const std::vector<typename Matrix::degree_type>& locations) {
    using index = typename Matrix::index_type;
    using degree_type = typename Matrix::degree_type;
    std::vector<index> result(locations.size(), 0);
    if (locations.empty()) return result;
    std::vector<std::pair<degree_type, index>> events;
    visit_euler_degrees(resolution, [&](const auto& degree, index sign) { events.emplace_back(degree, sign); });
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        return a.first.first < b.first.first;
    });
    std::vector<double> ys;
    for (const auto& location : locations) ys.push_back(location.second);
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    std::vector<std::size_t> order(locations.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return locations[a].first < locations[b].first; });
    std::vector<index> tree(ys.size() + 1, 0);
    std::size_t next = 0;
    for (auto q : order) {
        while (next < events.size() && events[next].first.first <= locations[q].first) {
            auto i = static_cast<std::size_t>(std::lower_bound(ys.begin(), ys.end(), events[next].first.second) - ys.begin()) + 1;
            for (; i < tree.size(); i += i & -i) tree[i] += events[next].second;
            ++next;
        }
        auto i = static_cast<std::size_t>(std::lower_bound(ys.begin(), ys.end(), locations[q].second) - ys.begin()) + 1;
        for (; i; i -= i & -i) result[q] += tree[i];
    }
    return result;
}

} // namespace graded_linalg::detail
