/** @file hom_generators.hpp @brief Budgeted, certified F2 composition generators. */
#pragma once

#include <grlina/hom_interface.hpp>
#include <algorithm>
#include <array>
#include <map>
#include <limits>
#include <string>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace graded_linalg {

struct CompositionGeneratorOptions {
    bool sparse_nilpotent = true;
    bool complement_orbit = true;
    // Sparse-nilpotent method (nilpotent samples plus small sparse seeds).
    std::size_t max_products = 8192;       // Per side; half reserved for the nilpotent stage.
    std::size_t max_pair_visits = 131072; // Includes zero, duplicate and over-budget products.
    std::size_t max_product_terms = 65536; // Symbolic scalar terms, before F2 cancellation.
    std::size_t nilpotent_rounds = 8;
    std::size_t max_seeds = 8;
    std::size_t max_seed_basis = 8;
    std::size_t max_seed_terms = 256;     // Sum of input nnz, so also bounds the resulting nnz.
    // Complement-orbit: broad sums, random subsets, or alternating broad/random seeds.
    std::size_t orbit_rounds = 12;
    std::string orbit_seed_mode = "hybrid"; // sum | random | hybrid
    std::size_t orbit_min_gain = 1; // Net maps saved; smaller gains count toward patience.
    std::size_t orbit_patience = 3; // Consecutive low-gain attempts; 0 disables early stopping.
    std::size_t orbit_random_seed = 923407;
    std::size_t orbit_max_pair_visits = 0; // 0 = rounds * dim End, with overflow saturation.
    std::size_t orbit_max_product_terms = 65536;
    std::size_t orbit_max_seed_terms = 65536; // Sum of input nnz; zero disables seed construction.
};

struct CompositionGeneratorStats {
    std::size_t nilpotent_endomorphisms = 0, nilpotent_rank = 0;
    std::size_t pair_visits = 0, products = 0, oversized_products = 0;
    std::size_t rank = 0, seeds = 0, sparse_generators = 0;
    std::size_t orbit_pair_visits = 0, orbit_products = 0, orbit_oversized_products = 0;
    std::size_t orbit_seeds = 0, largest_orbit_seed = 0;
    std::size_t orbit_attempts = 0, orbit_low_gain_attempts = 0;
};

/** Sparse F2 echelon columns, optionally carrying coordinates in an input basis.
 * Existing columns never change on insertion, which also permits rollback of
 * a trial by erasing only its newly inserted pivots.
 */
struct CompositionCoefficientBasis {
    using Vector = std::vector<std::size_t>;
    struct Column { Vector entries, coordinates; };
    std::map<std::size_t, Column> pivots;

    Vector normal(Vector entries, Vector* coordinates = nullptr) const {
        Vector remainder;
        while (!entries.empty()) {
            const auto found = pivots.find(entries.back());
            if (found == pivots.end()) {
                remainder.push_back(entries.back());
                entries.pop_back();
            } else {
                Column_traits<Vector, std::size_t>::add_to(found->second.entries, entries);
                if (coordinates)
                    Column_traits<Vector, std::size_t>::add_to(found->second.coordinates, *coordinates);
            }
        }
        std::reverse(remainder.begin(), remainder.end());
        return remainder;
    }

    bool insert(Vector entries, Vector coordinates = {}) {
        entries = normal(std::move(entries), &coordinates);
        if (entries.empty()) return false;
        const auto pivot = entries.back();
        pivots.emplace(pivot, Column{std::move(entries), std::move(coordinates)});
        return true;
    }
};

/** Prepare relation normal forms, W, and coordinates in H/W once for both sides.
 * No new lifting solve is needed for products of valid generator lifts.
 */
template<class Matrix>
class PreparedCompositionCoordinates {
    using Index = matrix_index_t<Matrix>;
    using Vector = CompositionCoefficientBasis::Vector;
    MapCoefficientContext<Matrix> context_;
    CompositionCoefficientBasis denominator_, coordinates_;

public:
    PreparedCompositionCoordinates(const std::vector<Homomorphism<Matrix>>& basis,
                                   const std::vector<Homomorphism<Matrix>>& subspace)
        : context_(basis.at(0).domain()->presentation(), basis.at(0).target()->presentation()) {
        for (const auto& map : subspace) denominator_.insert(coefficients(map.generator_lift()));
        for (std::size_t j = 0; j < basis.size(); ++j)
            if (!coordinates_.insert(denominator_.normal(coefficients(basis[j].generator_lift())), {j}))
                throw std::invalid_argument("Composition input is not a basis modulo the supplied subspace");
    }
    template<class Storage>
    PreparedCompositionCoordinates(const SparseMatrix<std::ptrdiff_t, Storage>& basis,
                                   const SparseMatrix<std::ptrdiff_t, Storage>& subspace,
                                   const Matrix& source, const Matrix& target)
        : context_(source, target) {
        for (const auto& entries : subspace.data) {
            auto normal = context_.normal(entries);
            denominator_.insert(Vector(normal.begin(), normal.end()));
        }
        for (std::ptrdiff_t j = 0; j < basis.get_num_cols(); ++j) {
            auto normal = context_.normal(basis.column(j));
            if (!coordinates_.insert(denominator_.normal(Vector(normal.begin(), normal.end())), {std::size_t(j)}))
                throw std::invalid_argument("Composition input is not a basis modulo the supplied subspace");
        }
    }
    PreparedCompositionCoordinates(const PreparedCompositionCoordinates&) = delete;
    PreparedCompositionCoordinates& operator=(const PreparedCompositionCoordinates&) = delete;

    void append_normal_column(std::size_t column, std::vector<Index> entries, Vector& result) const {
        context_.append_normal_column(column, std::move(entries), result);
    }

    Vector coefficients(const Matrix& map) const {
        Vector result;
        for (Index c = 0; c < map.get_num_cols(); ++c)
            if (!map.column(c).empty()) append_normal_column(c, map.get_col(c), result);
        return result;
    }

    Vector coordinates(Vector normalized_coefficients) const {
        Vector result;
        if (!coordinates_.normal(denominator_.normal(std::move(normalized_coefficients)), &result).empty())
            throw std::invalid_argument("Composition lies outside the supplied quotient span");
        return result;
    }
};

/** Certified left/right generators of H/W for E=End_0(X).
 * Each coefficient column encodes one map; source/target supply common grades.
 * basis consists of X -> X(delta) maps; W is invariant on both sides, and end
 * consists of X -> X maps in those same presentation coordinates. W=eta E is
 * the pruning case; W=0 gives generators of H. Shifts change grades, not the
 * coefficient products. All input maps and the supplied quotient basis are trusted.
 *
 * U is contained in NQ + EG (or QN + GE), where N is the nilpotent ideal of
 * maps admitting strictly birth-degree-lowering lifts. Return G and the original
 * nonpivot basis vectors. Nilpotence proves generation even when the budget
 * stops the search before U is invariant or before all products are visited.
 * Two independent methods share U: sparse_nilpotent, then complement_orbit.
 * The latter retains sums/subsets of current nonpivot basis maps and samples their
 * orbits. Existing sparse candidates remain retained. Never reinterpret the
 * reduced generating family as a vector-space basis of Q.
 * An optional prepared coordinate solver must use this exact basis and subspace.
 * No minimum-cardinality or runtime improvement is promised. Dense products are
 * skipped; budgets bound attempts and symbolic multiplication, not relation fill-in.
 */
template<class Matrix, class Storage>
std::pair<SparseMatrix<std::ptrdiff_t, Storage>, SparseMatrix<std::ptrdiff_t, Storage>>
budgeted_composition_generators(
    const SparseMatrix<std::ptrdiff_t, Storage>& basis,
    const SparseMatrix<std::ptrdiff_t, Storage>& end,
    const SparseMatrix<std::ptrdiff_t, Storage>& subspace,
    const Matrix& source, const Matrix& target,
    const CompositionGeneratorOptions& options = {},
    std::array<CompositionGeneratorStats, 2>* statistics = nullptr,
    const PreparedCompositionCoordinates<Matrix>* prepared = nullptr) {
    using Index = matrix_index_t<Matrix>;
    using Vector = CompositionCoefficientBasis::Vector;
    using Column = typename Storage::value_type;
    using CT = Column_traits<Column, std::ptrdiff_t>;
    const std::size_t dimension = basis.get_num_cols(), end_count = end.get_num_cols();
    if (statistics) *statistics = {};
    if (options.orbit_seed_mode != "sum" && options.orbit_seed_mode != "random" && options.orbit_seed_mode != "hybrid")
        throw std::invalid_argument("Orbit seed mode must be sum, random, or hybrid");
    if (dimension == 0) {
        SparseMatrix<std::ptrdiff_t, Storage> empty(0, basis.get_num_rows());
        return {empty, std::move(empty)};
    }
    const bool sparse_enabled = options.sparse_nilpotent && options.max_products && options.max_pair_visits;
    const bool orbit_enabled = options.complement_orbit && options.orbit_rounds;
    const auto orbit_visit_limit = options.orbit_max_pair_visits ? options.orbit_max_pair_visits :
        (end_count && options.orbit_rounds > std::numeric_limits<std::size_t>::max() / end_count
         ? std::numeric_limits<std::size_t>::max() : options.orbit_rounds * end_count);
    if (!sparse_enabled && !orbit_enabled) {
        if (statistics) for (auto& side : *statistics) side.sparse_generators = dimension;
        return {basis, basis};
    }
    const auto degrees = geometric_row_degrees(source);
    const std::size_t generators = degrees.size();
    if (target.get_num_rows() != generators || basis.get_num_rows() != std::ptrdiff_t(generators) * generators ||
        end.get_num_rows() != basis.get_num_rows() || subspace.get_num_rows() != basis.get_num_rows())
        throw std::invalid_argument("Composition families require common square map coordinates");

    // Cache supports, not shifted copies of every matrix or a full action table.
    struct Support { Vector columns, rows; std::size_t terms = 0; };
    auto support = [&](const Column& entries) {
        Support result;
        result.terms = entries.size();
        for (auto entry : entries) {
            result.columns.push_back(entry / generators);
            result.rows.push_back(entry % generators);
        }
        result.columns.erase(std::unique(result.columns.begin(), result.columns.end()), result.columns.end());
        std::sort(result.rows.begin(), result.rows.end());
        result.rows.erase(std::unique(result.rows.begin(), result.rows.end()), result.rows.end());
        return result;
    };
    auto column_range = [&](const Column& entries, std::size_t c) {
        const auto first = std::lower_bound(entries.begin(), entries.end(), std::ptrdiff_t(c * generators));
        return std::make_pair(first, std::lower_bound(first, entries.end(), std::ptrdiff_t((c + 1) * generators)));
    };
    std::vector<Support> end_support, basis_support;
    std::vector<bool> lowering;
    for (const auto& e : end.data) {
        end_support.push_back(support(e));
        bool strict = true;
        if (sparse_enabled) for (auto entry : e)
            if (degrees[entry % generators] == degrees[entry / generators]) strict = false;
        lowering.push_back(strict);
    }
    for (const auto& q : basis.data) basis_support.push_back(support(q));
    Vector order(dimension);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return basis_support[a].terms < basis_support[b].terms;
    });
    std::unique_ptr<PreparedCompositionCoordinates<Matrix>> owned_coordinates;
    if (!prepared) {
        owned_coordinates = std::make_unique<PreparedCompositionCoordinates<Matrix>>(basis, subspace, source, target);
        prepared = owned_coordinates.get();
    }
    const auto& coordinates = *prepared;

    auto select = [&](bool left, CompositionGeneratorStats& stats) {
        CompositionCoefficientBasis span;
        std::vector<Column> selected;
        std::mt19937 random(left ? 923407 : 923408);
        std::seed_seq orbit_seed{static_cast<unsigned>(options.orbit_random_seed),
            static_cast<unsigned>(options.orbit_random_seed / 4294967296ULL), unsigned(left ? 0 : 1)};
        std::mt19937 orbit_random(orbit_seed);
        // Index only structural compatibility. F2 cancellations are still checked
        // by the exact product/coordinate path; no potentially nonzero action is omitted.
        std::vector<Vector> by_middle(generators);
        for (std::size_t e = 0; e < end_count; ++e)
            for (auto middle : left ? end_support[e].columns : end_support[e].rows)
                by_middle[middle].push_back(e);
        auto supported_actions = [&](const Support& seed_support) {
            Vector actions;
            std::vector<bool> seen(end_count, false);
            for (auto middle : left ? seed_support.rows : seed_support.columns)
                for (auto e : by_middle[middle]) if (!seen[e]) {
                    seen[e] = true; actions.push_back(e);
                }
            std::sort(actions.begin(), actions.end()); // Preserve basis order under partial budgets.
            return actions;
        };
        auto product = [&](const Column& a, const Column& b, const Vector& b_columns, Vector& result, bool orbit) {
            const auto limit = orbit ? options.orbit_max_product_terms : options.max_product_terms;
            auto& oversized = orbit ? stats.orbit_oversized_products : stats.oversized_products;
            std::size_t terms = 0;
            for (auto c : b_columns) {
                const auto [first, last] = column_range(b, c);
                for (auto it = first; it != last; ++it) {
                    const auto [a_first, a_last] = column_range(a, *it % generators);
                    const auto count = std::size_t(a_last - a_first);
                    if (count > limit - terms) { ++oversized; return false; }
                    terms += count;
                }
            }
            if (!terms) return false;
            ++(orbit ? stats.orbit_products : stats.products);
            Vector coefficients;
            for (auto c : b_columns) {
                std::vector<Index> column, summand;
                const auto [first, last] = column_range(b, c);
                for (auto it = first; it != last; ++it) {
                    const auto [a_first, a_last] = column_range(a, *it % generators);
                    summand.clear();
                    for (auto entry = a_first; entry != a_last; ++entry) summand.push_back(*entry % generators);
                    Column_traits<std::vector<Index>, Index>::add_to(summand, column);
                }
                if (!column.empty()) coordinates.append_normal_column(c, std::move(column), coefficients);
            }
            result = coordinates.coordinates(std::move(coefficients));
            return true;
        };
        auto action = [&](std::size_t e, const Column& q, const Support& q_support, Vector& result, bool orbit = false) {
            return left ? product(end.column(e), q, q_support.columns, result, orbit)
                        : product(q, end.column(e), end_support[e].columns, result, orbit);
        };

        if (sparse_enabled) {
            // Only basis endomorphisms with zero same-degree blocks are used here.
            // ponytail: this can miss kernel combinations; compute that kernel only
            // if the cheap pass proves insufficient on measured workloads.
            std::vector<Vector> nil_by_middle(generators), middles(dimension);
            for (std::size_t e = 0; e < end_count; ++e) if (lowering[e]) {
                ++stats.nilpotent_endomorphisms;
                for (auto middle : left ? end_support[e].columns : end_support[e].rows)
                    nil_by_middle[middle].push_back(e);
            }
            for (std::size_t q = 0; q < dimension; ++q)
                for (auto middle : left ? basis_support[q].rows : basis_support[q].columns)
                    if (!nil_by_middle[middle].empty()) middles[q].push_back(middle);
            std::set<std::pair<std::size_t, std::size_t>> visited;
            const auto nil_limit = options.max_products / 2;
            for (std::size_t round = 0; round < options.nilpotent_rounds && stats.products < nil_limit &&
                    stats.pair_visits < options.max_pair_visits / 2; ++round) {
                for (auto q : order) {
                    if (stats.products >= nil_limit || stats.pair_visits >= options.max_pair_visits / 2) break;
                    if (middles[q].empty()) continue;
                    ++stats.pair_visits;
                    const auto& bucket = nil_by_middle[middles[q][random() % middles[q].size()]];
                    const auto e = bucket[random() % bucket.size()];
                    if (!visited.emplace(e, q).second) continue;
                    Vector result;
                    if (action(e, basis.column(q), basis_support[q], result)) span.insert(std::move(result));
                }
            }
            stats.nilpotent_rank = span.pivots.size();

            // Small sums with disjoint row/column supports favor partial permutation
            // blocks instead of the often rank-one sum of every matrix unit.
            std::vector<bool> tried(dimension, false);
            for (std::size_t trial = 0; trial < options.max_seeds && stats.products < options.max_products &&
                    stats.pair_visits < options.max_pair_visits; ++trial) {
                Vector candidate;
                std::vector<bool> rows(generators, false), columns(generators, false);
                std::size_t terms = 0;
                for (auto q : order) {
                    if (tried[q] || span.pivots.count(q) ||
                        basis_support[q].terms > options.max_seed_terms - terms) continue;
                    bool overlaps = false;
                    for (auto r : basis_support[q].rows) if (rows[r]) overlaps = true;
                    for (auto c : basis_support[q].columns) if (columns[c]) overlaps = true;
                    if (overlaps) continue;
                    if (candidate.size() == options.max_seed_basis) break;
                    candidate.push_back(q); tried[q] = true;
                    terms += basis_support[q].terms;
                    for (auto r : basis_support[q].rows) rows[r] = true;
                    for (auto c : basis_support[q].columns) columns[c] = true;
                }
                if (candidate.empty()) break;
                Column seed = basis.column(candidate.front());
                for (std::size_t j = 1; j < candidate.size(); ++j) CT::add_to(basis.column(candidate[j]), seed);
                const auto seed_support = support(seed);
                std::sort(candidate.begin(), candidate.end());
                Vector added_pivots;
                auto insert_trial = [&](Vector vector) {
                    vector = span.normal(std::move(vector));
                    if (!vector.empty()) { added_pivots.push_back(vector.back()); span.insert(std::move(vector)); }
                };
                insert_trial(candidate); // Identity action, without a matrix product.
                for (auto e : supported_actions(seed_support)) {
                    if (stats.products >= options.max_products || stats.pair_visits >= options.max_pair_visits ||
                        span.pivots.size() == dimension) break;
                    ++stats.pair_visits;
                    Vector result;
                    if (action(e, seed, seed_support, result)) insert_trial(std::move(result));
                }
                if (added_pivots.size() > 1) selected.push_back(std::move(seed));
                else for (auto pivot : added_pivots) span.pivots.erase(pivot);
            }
        }
        stats.sparse_generators = selected.size() + dimension - span.pivots.size();

        // Complement-orbit: varied seeds with a round cap and low-gain patience.
        // Independently budgeted. It extends the SAME certificate when following the sparse
        // method; previously retained seeds remain in the returned family.
        std::size_t low_gain_streak = 0;
        for (std::size_t round = 0; orbit_enabled && round < options.orbit_rounds &&
                stats.orbit_pair_visits < orbit_visit_limit && span.pivots.size() < dimension; ++round) {
            ++stats.orbit_attempts;
            const bool random_subset = options.orbit_seed_mode == "random" ||
                (options.orbit_seed_mode == "hybrid" && (round % 2) == 1);
            auto low_gain = [&] {
                ++stats.orbit_low_gain_attempts;
                return ++low_gain_streak >= options.orbit_patience && options.orbit_patience != 0;
            };
            Vector candidate;
            std::size_t terms = 0;
            bool affordable = true;
            for (std::size_t q = 0; q < dimension; ++q) if (!span.pivots.count(q)) {
                if (random_subset && !(orbit_random() & 1u)) continue;
                if (basis_support[q].terms > options.orbit_max_seed_terms - terms) {
                    affordable = false; break;
                }
                candidate.push_back(q);
                terms += basis_support[q].terms;
            }
            if (!affordable || candidate.empty()) {
                if (low_gain()) break;
                continue;
            }
            Column seed = basis.column(candidate.front());
            for (std::size_t j = 1; j < candidate.size(); ++j) CT::add_to(basis.column(candidate[j]), seed);
            const auto seed_support = support(seed);
            stats.largest_orbit_seed = std::max(stats.largest_orbit_seed, seed_support.terms);
            Vector added_pivots;
            auto insert_trial = [&](Vector vector) {
                vector = span.normal(std::move(vector));
                if (!vector.empty()) { added_pivots.push_back(vector.back()); span.insert(std::move(vector)); }
            };
            insert_trial(candidate);
            for (auto e : supported_actions(seed_support)) {
                if (stats.orbit_pair_visits >= orbit_visit_limit || span.pivots.size() == dimension) break;
                ++stats.orbit_pair_visits;
                Vector result;
                if (action(e, seed, seed_support, result, true)) insert_trial(std::move(result));
            }
            if (added_pivots.size() > 1) {
                selected.push_back(std::move(seed));
                ++stats.orbit_seeds;
            } else {
                for (auto pivot : added_pivots) span.pivots.erase(pivot);
            }
            const auto saved = added_pivots.size() > 1 ? added_pivots.size() - 1 : 0;
            // Positive but small savings remain certified and retained. Patience
            // controls further work, not acceptance. Zero gain always counts as low.
            if (!saved || saved < options.orbit_min_gain) {
                if (low_gain()) break;
            } else low_gain_streak = 0;
        }
        stats.rank = span.pivots.size(); stats.seeds = selected.size();
        for (std::size_t j = 0; j < dimension; ++j)
            if (!span.pivots.count(j)) selected.push_back(basis.column(j));
        SparseMatrix<std::ptrdiff_t, Storage> result(selected.size(), basis.get_num_rows());
        result.assign_data(std::move(selected));
        return result;
    };
    std::array<CompositionGeneratorStats, 2> stats;
    auto left = select(true, stats[0]);
    auto right = select(false, stats[1]);
    if (statistics) *statistics = stats;
    return {std::move(left), std::move(right)};
}

template<class Matrix>
std::pair<std::vector<Homomorphism<Matrix>>, std::vector<Homomorphism<Matrix>>>
budgeted_composition_generators(
    const std::vector<Homomorphism<Matrix>>& basis,
    const std::vector<Homomorphism<Matrix>>& end,
    const std::vector<Homomorphism<Matrix>>& subspace = {},
    const CompositionGeneratorOptions& options = {},
    std::array<CompositionGeneratorStats, 2>* statistics = nullptr,
    const PreparedCompositionCoordinates<Matrix>* prepared = nullptr) {
    if (basis.empty()) { if (statistics) *statistics = {}; return {}; }
    const auto H = homomorphism_coefficients(basis);
    auto E = homomorphism_coefficients(end);
    auto W = homomorphism_coefficients(subspace);
    if (end.empty()) E.set_num_rows(H.get_num_rows());
    if (subspace.empty()) W.set_num_rows(H.get_num_rows());
    auto selected = budgeted_composition_generators(H, E, W,
        basis.front().domain()->presentation(), basis.front().target()->presentation(), options, statistics, prepared);
    return {homomorphisms_from_coefficients(selected.first, basis.front().domain(), basis.front().target()),
            homomorphisms_from_coefficients(selected.second, basis.front().domain(), basis.front().target())};
}

} // namespace graded_linalg
