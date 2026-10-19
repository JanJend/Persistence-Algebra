/** @file mod_module.hpp @brief Modules backed by optional projective/injective resolutions. */
#pragma once

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <grlina/mod_isomorphism.hpp>
#include <grlina/proj_chain_complex.hpp>
#include <grlina/mat_r2.hpp>
#include <grlina/mat_grid.hpp>

namespace graded_linalg {

template <typename Matrix, typename Visitor>
void visit_euler_degrees(const ChainComplex<Matrix>& resolution, Visitor visit) {
    using index = typename Matrix::index_type;
    GRLINA_DEBUG_CHECK(if (resolution.empty()) throw std::logic_error("Euler characteristic requires a resolution"));
    for (const auto& degree : resolution[0].row_degrees) visit(geometric_degree(resolution[0], degree), index{1});
    index sign = -1;
    for (const auto& differential : resolution.differentials()) {
        for (const auto& degree : differential.col_degrees) visit(geometric_degree(differential, degree), sign);
        sign = -sign;
    }
}

template <typename Matrix>
typename Matrix::index_type euler_dimension_at(
    const ChainComplex<Matrix>& resolution, const matrix_geometry_degree_t<Matrix>& location) {
    typename Matrix::index_type result = 0;
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
std::vector<std::vector<typename Matrix::index_type>> euler_grid_r2(
    const ChainComplex<Matrix>& resolution, const std::vector<double>& xs,
    const std::vector<double>& ys) {
    using index = typename Matrix::index_type;
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
std::vector<typename Matrix::index_type> euler_queries_r2(
    const ChainComplex<Matrix>& resolution,
    const std::vector<matrix_geometry_degree_t<Matrix>>& locations) {
    using index = typename Matrix::index_type;
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

} // namespace graded_linalg

namespace graded_linalg {

enum class ResolutionKind { projective, injective };
enum class ResolutionCompleteness { truncated, complete };

template <typename Matrix>
class Submodule;

template <typename Matrix>
class Module : public std::enable_shared_from_this<Module<Matrix>> {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Module<Matrix> requires the graded sparse matrix interface");
    using degree_type = matrix_geometry_degree_t<Matrix>;
    using stored_degree_type = typename Matrix::degree_type;
    using index_type = typename Matrix::index_type;
    using chain_complex_type = ChainComplex<Matrix>;

    struct HilbertValue { degree_type degree; index_type dimension; };
    struct R2HilbertGrid {
        vec<double> x_grid;
        vec<double> y_grid;
        array<index_type> values;
        index_type maximum = 0;
    };

private:
    chain_complex_type projective_resolution_;
    chain_complex_type injective_resolution_; // not used at the moment

    bool projective_resolution_complete_ = false;

    static void normalize_indices(vec<index_type>& indices, index_type size) {
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        GRLINA_DEBUG_CHECK(if (!indices.empty() && (indices.front() < 0 || indices.back() >= size))
            throw std::out_of_range("Presentation index out of range"));
    }

    void require_presentation() const {
        GRLINA_DEBUG_CHECK(if (projective_resolution_.empty())
            throw std::logic_error("This module has no projective presentation"));
    }

    void ensure_resolution_for_hilbert_grid() {
        if constexpr (has_matrix_graded_kernel<Matrix>::value)
            if (!has_complete_projective_resolution()) compute_projective_resolution();
    }

    static void validate_hilbert_axes(const vec<double>& xs, const vec<double>& ys) {
        for (const auto* axis : {&xs, &ys}) {
            if (!std::is_sorted(axis->begin(), axis->end()) ||
                std::adjacent_find(axis->begin(), axis->end()) != axis->end() ||
                std::any_of(axis->begin(), axis->end(), [](double x) { return x != x; }))
                throw std::invalid_argument("Hilbert grid axes must be sorted and unique, without NaN");
        }
    }

public:
    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = default;
    Module(Module&&) = default;
    Module& operator=(const Module&) = default;
    Module& operator=(Module&&) = default;
    explicit Module(Matrix presentation)
        : projective_resolution_(std::vector<Matrix>{std::move(presentation)}) {}
    explicit Module(chain_complex_type resolution,
                    ResolutionKind kind = ResolutionKind::projective,
                    ResolutionCompleteness completeness = ResolutionCompleteness::truncated) {
        if (kind == ResolutionKind::projective) {
            projective_resolution_ = std::move(resolution);
            projective_resolution_complete_ = completeness == ResolutionCompleteness::complete;
        }
        else injective_resolution_ = std::move(resolution);
    }
    Module(chain_complex_type projective, chain_complex_type injective)
        : projective_resolution_(std::move(projective)), injective_resolution_(std::move(injective)) {
    }
    explicit Module(const std::string& path, bool sort_if_needed = false)
        : projective_resolution_(chain_complex_type::from_file(path, sort_if_needed)) {}
    explicit Module(std::istream& input, bool sort_if_needed = false)
        : projective_resolution_(chain_complex_type::from_stream(input, sort_if_needed)) {}

    static Module from_presentation(Matrix presentation) {
        return Module(std::move(presentation));
    }
    static Module from_projective_resolution(chain_complex_type resolution,
        ResolutionCompleteness completeness = ResolutionCompleteness::truncated) {
        return Module(std::move(resolution), ResolutionKind::projective, completeness);
    }
    static Module from_injective_resolution(chain_complex_type resolution) {
        return Module(std::move(resolution), ResolutionKind::injective);
    }

    bool has_projective_resolution() const noexcept { return !projective_resolution_.empty(); }
    bool has_injective_resolution() const noexcept { return !injective_resolution_.empty(); }
    bool has_presentation() const noexcept { return has_projective_resolution(); }
    /** A nonempty sequence alone is not evidence of completeness. An explicit
     * zero terminal group certifies the endpoint, assuming supplied exactness.
     */
    bool has_complete_projective_resolution() const noexcept {
        return has_projective_resolution() && (projective_resolution_complete_ ||
            projective_resolution_[projective_resolution_.size() - 1].get_num_cols() == 0);
    }
    /** Trust the caller's guarantee, just as for exactness of supplied resolutions.
     * SCC has no completeness marker; call this after loading a known full resolution.
     */
    void set_projective_resolution_completeness(ResolutionCompleteness completeness) {
        require_presentation();
        projective_resolution_complete_ = completeness == ResolutionCompleteness::complete;
    }
    const chain_complex_type& projective_resolution() const noexcept { return projective_resolution_; }
    const chain_complex_type& injective_resolution() const noexcept { return injective_resolution_; }

    void set_projective_resolution(chain_complex_type resolution,
        ResolutionCompleteness completeness = ResolutionCompleteness::truncated) {
        projective_resolution_ = std::move(resolution);
        projective_resolution_complete_ = completeness == ResolutionCompleteness::complete;
    }
    void set_injective_resolution(chain_complex_type resolution) {
        injective_resolution_ = std::move(resolution);
    }
    void clear_projective_resolution() noexcept {
        projective_resolution_.clear();
        projective_resolution_complete_ = false;
    }
    void clear_injective_resolution() noexcept { injective_resolution_.clear(); }

    const Matrix& presentation() const { require_presentation(); return projective_resolution_[0]; }

    /** Canonical submodules retaining this exact parent. Requires shared_ptr
     * ownership (otherwise throws std::bad_weak_ptr) and a presentation.
     * Include mod_submodule.hpp or mod_modules.hpp for the definitions.
     */
    Submodule<Matrix> whole_submodule() const;
    Submodule<Matrix> zero_submodule() const;

    /** Obtain an explicit presentation in this object. Plain modules already
     * require a supplied presentation; derived representations (e.g. Submodule)
     * override this hook to construct one from their defining data.
     */
    virtual void compute_presentation(bool minimize = false) {
        require_presentation();
        if (minimize) minimize_presentation();
    }

    /** Arbitrary edits invalidate higher projective maps and the injective model. */
    Matrix& mutable_presentation() {
        require_presentation();
        projective_resolution_complete_ = false;
        if (projective_resolution_.size() > 1) {
            Matrix d1 = projective_resolution_[0];
            projective_resolution_ = chain_complex_type(std::vector<Matrix>{std::move(d1)});
        }
        injective_resolution_.clear();
        return projective_resolution_[0];
    }

    template <typename Editor>
    void edit_presentation(Editor&& editor) {
        Matrix& d1 = mutable_presentation();
        std::forward<Editor>(editor)(d1);

    }

    index_type number_of_generators() const { return presentation().get_num_rows(); }
    index_type number_of_relations() const { return presentation().get_num_cols(); }
    std::size_t number_of_entries() const {
        std::size_t result = 0;
        for (const auto& column : presentation().data) result += column.size();
        return result;
    }

    /** Fibre presentation and the original generator indices of its rows.
     * Its cokernel is the module fibre; row coordinates are normalized locally.
     */
    auto local_presentation_at(const degree_type& degree) const {
        return presentation().map_at_degree_pair(query_matrix_degree(presentation(), degree));
    }

    /** Relations available at degree, retaining ALL original generator rows.
     * relation_indices is replaced with their original column indices.
     */
    auto relations_at(const degree_type& degree,
                                         vec<index_type>& relation_indices) const {
        relation_indices.clear();
        return presentation().map_at_degree(query_matrix_degree(presentation(), degree), relation_indices);
    }

    void print_presentation(bool suppress_description = false) const {
        presentation().print_graded(suppress_description);
    }
    void print_degrees() const { presentation().print_degrees(); }

    /** Remove chosen relations from this presentation; may enlarge the module. */
    void remove_relations(vec<index_type> indices) {
        normalize_indices(indices, number_of_relations());
        edit_presentation([&](Matrix& matrix) { matrix.delete_columns(indices); });
    }

    /** Kill the chosen presentation generators (take the corresponding quotient). */
    void quotient_by_generators(vec<index_type> indices) {
        normalize_indices(indices, number_of_generators());
        edit_presentation([&](Matrix& matrix) { matrix.delete_rows(indices); });
    }

    /** Keep the first count generators and kill the remaining ones.
     * Equivalent to the matrix cull_columns(count, false), which truncates rows.
     */
    void quotient_by_tail_generators(index_type count) {
        GRLINA_DEBUG_CHECK(if (count < 0 || count > number_of_generators())
            throw std::out_of_range("Number of retained generators out of range"));
        edit_presentation([&](Matrix& matrix) { matrix.cull_columns(count, false); });
    }

    /** Add a homogeneous relation in the current generator coordinates. */
    void add_relation(const vec<index_type>& coefficients, const degree_type& degree) {
        edit_presentation([&](Matrix& matrix) {
            if constexpr (matrix_grid_backed_v<Matrix>) {
                matrix.include_real_degrees({degree});
                matrix.append_column(coefficients, matrix.grid_degree(degree));
            } else matrix.append_column(coefficients, degree);
        });
    }

    /** Cancel local generator/relation pairs without requiring a graded kernel. */
    void semi_minimize_presentation() {
        Matrix matrix = presentation();
        matrix.sort_compatibly();
        matrix.semi_minimize();
        projective_resolution_ = chain_complex_type(std::vector<Matrix>{std::move(matrix)});
        projective_resolution_complete_ = false;
    }

    bool is_presentation_minimal() const {
        Matrix matrix = presentation();
        matrix.sort_compatibly();
        return matrix.is_minimal();
    }

    /** Bounds of presentation degrees, NOT bounds of the module's support. */
    auto presentation_degree_bounds() const {
        GRLINA_DEBUG_CHECK(if (number_of_generators() == 0 && number_of_relations() == 0)
            throw std::invalid_argument("Empty presentation has no degree bounds"));
        return presentation().bounding_box();
    }

    /** An n-by-n sampling grid over the R2 presentation degree bounds. */
    vec<degree_type> equidistant_presentation_grid(int n) const {
        GRLINA_DEBUG_CHECK(if (n < 0) throw std::invalid_argument("Grid size must be nonnegative"));
        if (n == 0 || (number_of_generators() == 0 && number_of_relations() == 0)) return {};
        return presentation().get_equidistant_grid(n);
    }

    /** Restrict to a finite degree poset; by default use presentation degrees. */
    auto to_quiver(vec<degree_type> vertices = {}, array<index_type> edges = {}) const {
        Matrix matrix = presentation();
        return matrix.induced_quiver_rep(std::move(vertices), std::move(edges));
    }

    virtual void sort_compatibly() {
        projective_resolution_.sort_compatibly();
        if (!injective_resolution_.empty()) injective_resolution_.sort_compatibly();
    }
    template <typename Compare>
    void sort_compatibly(Compare compare) {
        projective_resolution_.sort_compatibly(compare);
        if (!injective_resolution_.empty()) injective_resolution_.sort_compatibly(compare);
    }

    /** Use the complete stored projective resolution whenever available. */
    void minimize(bool sort_if_needed = true) {
        if (!has_presentation() && has_injective_resolution()) {
            minimize_injective_resolution(sort_if_needed);
            return;
        }
        require_presentation();
        if (projective_resolution_.size() > 1)
            minimize_resolution(sort_if_needed);
        else minimize_presentation(sort_if_needed);
    }

    /** Minimize the stored projective resolution, which may be truncated.
     * Exactness below the truncation is the caller's guarantee. After chain
     * cancellation, minimize the terminal map's generating set via its kernel.
     * This extra step preserves the resolved module, but may change terminal
     * homology of a truncation; it does not belong to ChainComplex::minimize.
     * Keep the original resolution intact if either step fails.
     */
    virtual void minimize_resolution(bool sort_if_needed = true) {
        require_presentation();
        chain_complex_type working = projective_resolution_;
        working.minimize(sort_if_needed);
        auto& terminal = working[working.size() - 1];
        if (terminal.get_num_cols() != 0) terminal.remove_redundant_relations();
        projective_resolution_ = std::move(working);
    }

    /** Remove redundant relations via the graded kernel, without changing
     * ambient generator coordinates. Higher projective maps are discarded;
     * the independent injective resolution remains valid and is preserved.
     * Sorting is temporary on rows, so existing submodule embeddings remain
     * valid. This does not cancel generator/relation pairs.
     */
    void remove_extra_rels(bool sort_if_needed = true) {
        Matrix reduced = presentation();
        vec<index_type> restore_rows;
        if (sort_if_needed) {
            restore_rows = reduced.sort_rows_with_permutation().new_to_old;
            reduced.sort_columns_lexicographically();
            reduced.compatibly_sorted = true;
        }
        reduced.remove_redundant_relations();
        if (sort_if_needed) {
            reduced.permute_rows_graded(restore_rows);
        }
        projective_resolution_ = chain_complex_type(std::vector<Matrix>{std::move(reduced)});
        projective_resolution_complete_ = false;
    }

    /** Presentation-only minimization: discard higher projective maps while
     * preserving the independently stored injective resolution.
     */
    virtual void minimize_presentation(bool sort_if_needed = true) {
        const bool was_complete_presentation = projective_resolution_.size() == 1 &&
            has_complete_projective_resolution();
        Matrix minimized = presentation();
        if (sort_if_needed) minimized.sort_compatibly();
        minimized.column_reduction_graded_w_deletion();
        minimized.cancel_local_pairs();
        Module working(std::move(minimized));
        working.remove_extra_rels(false);
        projective_resolution_ = std::move(working.projective_resolution_);
        projective_resolution_complete_ = was_complete_presentation;
    }

    void minimize_injective_resolution(bool /*sort_if_needed*/ = true) {
        // Jan: define the injective/cochain degree convention and implement its
        // dual cancellation. Projective cancellation must not be applied blindly.
        throw std::logic_error("Injective-resolution minimization is not implemented");
    }

    /** Complete the stored (possibly truncated) resolution, preserving existing
     * bases. Repeated kernels stop at an injective terminal map. This requires
     * a terminating graded-kernel implementation (supplied for R2 and the
     * runtime-dimensional coordinate/grid matrices).
     */
    void compute_projective_resolution() {
        if (!has_presentation()) compute_presentation();
        require_presentation();
        if (has_complete_projective_resolution()) return;
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            chain_complex_type working = projective_resolution_;

            while (true) {
                Matrix source = working[working.size() - 1]; // kernel computation is destructive
                Matrix kernel = source.graded_kernel();
                const bool terminal = kernel.get_num_cols() == 0;
                // Preserve the established d1,d2 shape for a presentation,
                // even when d2 has zero columns. No further zero maps are needed.
                if (!terminal || working.size() == 1) working.push_differential(std::move(kernel));
                if (terminal) break;
            }
            projective_resolution_ = std::move(working);
            projective_resolution_complete_ = true;
        } else {
            throw std::logic_error("This graded matrix type does not implement a graded kernel");
        }
    }

    index_type dimension_at(const degree_type& degree) const {
        const auto matrix_degree = query_matrix_degree(presentation(), degree);
        if (has_complete_projective_resolution())
            return euler_dimension_at(projective_resolution_, degree);
        auto local = presentation().map_at_degree_pair(matrix_degree, true).first;
        return static_cast<index_type>(local.coKernel_basis().size());
    }
    std::vector<HilbertValue> hilbert_function(const std::vector<degree_type>& locations) const {
        std::vector<HilbertValue> result;
        result.reserve(locations.size());
        if constexpr (std::is_same_v<degree_type, r2degree>) {
            if (has_complete_projective_resolution()) {
                const auto dimensions = euler_queries_r2(projective_resolution_, locations);
                for (std::size_t i = 0; i < locations.size(); ++i) result.push_back({locations[i], dimensions[i]});
                return result;
            }
        }
        for (const auto& degree : locations) result.push_back({degree, dimension_at(degree)});
        return result;
    }
    std::vector<degree_type> support_degrees() const {
        auto result = geometric_row_degrees(presentation());
        const auto columns = geometric_col_degrees(presentation());
        result.insert(result.end(), columns.begin(), columns.end());
        std::sort(result.begin(), result.end(), Degree_traits<degree_type>::lex_lambda());
        result.erase(std::unique(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            return Degree_traits<degree_type>::equals(lhs, rhs);
        }), result.end());
        return result;
    }
    std::vector<HilbertValue> hilbert_function_on_support() const {
        return hilbert_function(support_degrees());
    }

    /** Mutable grid queries retain a computed complete resolution for reuse.
     * Const queries compute on a private copy if needed, leaving shared modules
     * and references to their stored bases unchanged.
     */
    R2HilbertGrid hilbert_function_on_induced_grid() {
        ensure_resolution_for_hilbert_grid();
        return std::as_const(*this).hilbert_function_on_induced_grid();
    }

    /** Hilbert function on the full presentation-induced Cartesian grid. */
    R2HilbertGrid hilbert_function_on_induced_grid() const {
        static_assert(std::is_same<degree_type, r2degree>::value || matrix_grid_backed_v<Matrix>,
                      "This helper is only available for R2 modules");
        if constexpr (matrix_grid_backed_v<Matrix>)
            static_assert(std::tuple_size_v<degree_type> == 2, "Hilbert grid requires two parameters");
        vec<double> xs, ys;
        for (const auto& degree : support_degrees()) {
            xs.push_back(degree[0]);
            ys.push_back(degree[1]);
        }
        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
        std::sort(ys.begin(), ys.end());
        ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
        return hilbert_function_on_grid(xs, ys);
    }

    /** Query sorted, unique Cartesian axes, which need not contain all births. */
    R2HilbertGrid hilbert_function_on_grid(const vec<double>& xs, const vec<double>& ys) {
        GRLINA_DEBUG_CHECK(validate_hilbert_axes(xs, ys));
        ensure_resolution_for_hilbert_grid();
        return std::as_const(*this).hilbert_function_on_grid(xs, ys);
    }

    R2HilbertGrid hilbert_function_on_grid(const vec<double>& xs, const vec<double>& ys) const {
        static_assert(std::is_same_v<degree_type, r2degree> || matrix_grid_backed_v<Matrix>, "This grid helper is only available for R2 modules");
        require_presentation();
        if constexpr (matrix_grid_backed_v<Matrix>)
            static_assert(std::tuple_size_v<degree_type> == 2, "Hilbert grid requires two parameters");
        GRLINA_DEBUG_CHECK(validate_hilbert_axes(xs, ys));
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            if (!has_complete_projective_resolution()) {
                Module working = *this;
                working.compute_projective_resolution();
                return std::as_const(working).hilbert_function_on_grid(xs, ys);
            }
        }
        R2HilbertGrid result;
        result.x_grid = xs;
        result.y_grid = ys;
        if (has_complete_projective_resolution()) {
            result.values = euler_grid_r2(projective_resolution_, xs, ys);
            for (const auto& row : result.values)
                for (auto value : row) result.maximum = std::max(result.maximum, value);
            return result;
        }
        // No graded kernel is available: retain the local-presentation path.
        result.values.assign(result.x_grid.size(), vec<index_type>(result.y_grid.size(), 0));
        for (std::size_t x = 0; x < result.x_grid.size(); ++x) {
            for (std::size_t y = 0; y < result.y_grid.size(); ++y) {
                degree_type degree{static_cast<typename degree_type::value_type>(result.x_grid[x]),
                                   static_cast<typename degree_type::value_type>(result.y_grid[y])};
                result.values[x][y] = dimension_at(degree);
                result.maximum = std::max(result.maximum, result.values[x][y]);
            }
        }
        return result;
    }

    void shift(const degree_type& amount) {
        for (auto& differential : projective_resolution_.differentials())
            differential.shift(amount);
        for (auto& differential : injective_resolution_.differentials())
            differential.shift(amount);
    }
    template <typename... Args> void snap_to_grid(Args&&... args) {
        edit_presentation([&](Matrix& matrix) { matrix.snap_to_grid(std::forward<Args>(args)...); });
    }
    template <typename... Args> void snap_to_equidistant_grid(Args&&... args) {
        edit_presentation([&](Matrix& matrix) { matrix.snap_to_equidistant_grid(std::forward<Args>(args)...); });
    }
    template <typename... Args> void bound_support(Args&&... args) {
        edit_presentation([&](Matrix& matrix) { matrix.bound_support(std::forward<Args>(args)...); });
    }
    template <typename... Args> void cut_above(Args&&... args) {
        edit_presentation([&](Matrix& matrix) { matrix.cut_above(std::forward<Args>(args)...); });
    }

    template <typename OutputStream>
    void to_stream(OutputStream& output, ResolutionKind kind = ResolutionKind::projective) const {
        const auto& resolution = kind == ResolutionKind::projective
            ? projective_resolution_ : injective_resolution_;
        GRLINA_DEBUG_CHECK(if (resolution.empty()) throw std::logic_error("Requested resolution is empty"));
        resolution.to_stream(output);
    }
    void to_file(const std::string& path, ResolutionKind kind = ResolutionKind::projective) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Unable to open module output file: " + path);
        to_stream(output, kind);
    }
};

// Compatibility spelling from the first module API.
template <typename Matrix> using PersistenceModule = Module<Matrix>;
template <typename index, typename Storage = vec<vec<index>>>
using R2Module = Module<R2GradedSparseMatrix<index, Storage>>;

/**
 * Present H_k = ker(d_k) / im(d_{k+1}) from a chain complex.
 *
 * graded_kernel() returns kernel generators as columns in the coordinates of
 * C_k.  Each incoming-boundary column is therefore solved against that kernel
 * matrix at its own degree; those solution coordinates are the relations of
 * the homology presentation.
 */
template <typename Matrix>
std::pair<Module<Matrix>, Matrix> homology_presentation_and_cycles(
    const ChainComplex<Matrix>& complex, std::size_t homological_degree) {
    if (homological_degree == 0) {
        Matrix presentation = complex.empty()
            ? GradedMatrixIO<Matrix>::make_matrix(0, 0)
            : complex.differential(1);
        Matrix cycles(geometric_row_degrees(presentation));
        if constexpr (matrix_grid_backed_v<Matrix>) {
            cycles.reindex_grid(presentation.grids);
        }
        cycles.compatibly_sorted = presentation.compatibly_sorted;
        return {Module<Matrix>(std::move(presentation)), std::move(cycles)};
    }
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "homology_module requires the graded sparse matrix interface");
    if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
        throw std::logic_error("This graded matrix type does not implement a graded kernel");
    } else {
        GRLINA_DEBUG_CHECK(if (homological_degree > complex.size())
            throw std::out_of_range("No outgoing differential at the requested homological degree"));

        Matrix kernel_source = complex.differential(homological_degree);
        Matrix kernel_generators = kernel_source.graded_kernel();
        kernel_generators.sort_columns_lexicographically();

        using index_type = typename Matrix::index_type;
        using degree_type = typename Matrix::degree_type;
        typename Matrix::storage_type relation_coordinates;
        vec<degree_type> relation_degrees;

        if (homological_degree < complex.size()) {
            Matrix aligned_incoming;
            const Matrix* incoming_ptr = &complex.differential(homological_degree + 1);
            if constexpr (matrix_grid_backed_v<Matrix>) {
                aligned_incoming = *incoming_ptr;
                kernel_generators.merge_grids(aligned_incoming);
                incoming_ptr = &aligned_incoming;
            }
            const Matrix& incoming = *incoming_ptr;
            relation_coordinates.reserve(static_cast<std::size_t>(incoming.get_num_cols()));
            relation_degrees.assign(incoming.col_degrees.begin(), incoming.col_degrees.end());

            for (index_type column = 0; column < incoming.get_num_cols(); ++column) {
                const degree_type degree = incoming.col_degrees[column];
                auto local_pair = kernel_generators.map_at_degree_pair(degree, true);
                auto local_kernel = std::move(local_pair.first);
                const vec<index_type>& selected_rows = local_pair.second;

                vec<index_type> selected_kernel_columns;
                for (index_type candidate = 0;
                     candidate < kernel_generators.get_num_cols(); ++candidate) {
                    if (Degree_traits<degree_type>::smaller_equal(
                            kernel_generators.col_degrees[candidate], degree)) {
                        selected_kernel_columns.push_back(candidate);
                    }
                }

                vec<index_type> local_boundary;
                local_boundary.reserve(incoming.data[column].size());
                for (index_type row : incoming.data[column]) {
                    auto position = std::lower_bound(selected_rows.begin(), selected_rows.end(), row);
                    GRLINA_DEBUG_CHECK(if (position == selected_rows.end() || *position != row) {
                        throw std::logic_error(
                            "An incoming boundary contains a row unavailable at its degree");
                    });
                    local_boundary.push_back(
                        static_cast<index_type>(std::distance(selected_rows.begin(), position)));
                }

                vec<index_type> local_solution;
                if (!local_kernel.solve_col_reduction(
                        local_boundary, local_solution, true, true, true)) {
                    GRLINA_DEBUG_CHECK(throw std::invalid_argument(
                        "An incoming boundary is not contained in the computed graded kernel"));
                }

                vec<index_type> global_solution;
                global_solution.reserve(local_solution.size());
                for (index_type local_index : local_solution) {
                    GRLINA_DEBUG_CHECK(if (local_index < 0 ||
                        local_index >= static_cast<index_type>(selected_kernel_columns.size())) {
                        throw std::logic_error("Kernel-coordinate solver returned an invalid index");
                    });
                    global_solution.push_back(selected_kernel_columns[local_index]);
                }
                relation_coordinates.push_back(std::move(global_solution));
            }
        }

        // A graded kernel need not be free: include its own syzygies as well
        // as lifts of incoming boundaries. Both use the aligned kernel basis.
        Matrix kernel_copy = kernel_generators;
        Matrix kernel_relations = kernel_copy.graded_kernel();
        for (index_type column = 0; column < kernel_relations.get_num_cols(); ++column) {
            const auto entries = kernel_relations.get_col(column);
            relation_coordinates.push_back(vec<index_type>(entries.begin(), entries.end()));
            relation_degrees.push_back(kernel_relations.col_degrees[column]);
        }
        Matrix presentation = empty_matrix_like(kernel_generators,
            static_cast<index_type>(relation_coordinates.size()), kernel_generators.get_num_cols());
        presentation.col_degrees = relation_degrees;
        presentation.row_degrees = kernel_generators.col_degrees;
        presentation.assign_data(std::move(relation_coordinates));
        presentation.compatibly_sorted = false;
        return {Module<Matrix>(std::move(presentation)), std::move(kernel_generators)};
    }
}

template <typename Matrix>
Module<Matrix> homology_module(const ChainComplex<Matrix>& complex,
                              std::size_t homological_degree = 1, bool minimize = true) {
    auto result = homology_presentation_and_cycles(complex, homological_degree);
    if (minimize) result.first.minimize();
    return std::move(result.first);
}

} // namespace graded_linalg
