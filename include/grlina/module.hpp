/** @file module.hpp @brief Modules backed by optional projective/injective resolutions. */
#pragma once

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <grlina/chain_complex.hpp>
#include <grlina/hilbert_euler.hpp>
#include <grlina/r2graded_matrix.hpp>

namespace graded_linalg {

enum class ResolutionKind { projective, injective };
enum class ResolutionCompleteness { truncated, complete };

template <typename Matrix>
class Module {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Module<Matrix> requires Matrix to inherit "
                  "GradedSparseMatrix<D, index, Matrix> via CRTP");
    using matrix_type = Matrix;
    using degree_type = typename Matrix::degree_type;
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
    chain_complex_type injective_resolution_;
    bool projective_resolution_complete_ = false;

    void require_presentation() const {
        if (projective_resolution_.empty())
            throw std::logic_error("This module has no projective presentation");
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
        resolution.validate_structure();
        if (kind == ResolutionKind::projective) {
            projective_resolution_ = std::move(resolution);
            projective_resolution_complete_ = completeness == ResolutionCompleteness::complete;
        }
        else injective_resolution_ = std::move(resolution);
    }
    Module(chain_complex_type projective, chain_complex_type injective)
        : projective_resolution_(std::move(projective)), injective_resolution_(std::move(injective)) {
        projective_resolution_.validate_structure();
        injective_resolution_.validate_structure();
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
        resolution.validate_structure();
        projective_resolution_ = std::move(resolution);
        projective_resolution_complete_ = completeness == ResolutionCompleteness::complete;
    }
    void set_injective_resolution(chain_complex_type resolution) {
        resolution.validate_structure();
        injective_resolution_ = std::move(resolution);
    }
    void clear_projective_resolution() noexcept {
        projective_resolution_.clear();
        projective_resolution_complete_ = false;
    }
    void clear_injective_resolution() noexcept { injective_resolution_.clear(); }

    const Matrix& presentation() const { require_presentation(); return projective_resolution_[0]; }

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
        projective_resolution_.validate_structure();
    }

    index_type number_of_generators() const { return presentation().get_num_rows(); }
    index_type number_of_relations() const { return presentation().get_num_cols(); }
    std::size_t number_of_entries() const {
        std::size_t result = 0;
        for (const auto& column : presentation().data) result += column.size();
        return result;
    }

    void sort_compatibly() {
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
    void minimize_resolution(bool sort_if_needed = true) {
        require_presentation();
        chain_complex_type working = projective_resolution_;
        working.minimize(sort_if_needed);
        auto& terminal = working[working.size() - 1];
        if (terminal.get_num_cols() != 0) terminal.remove_redundant_relations();
        if (!working.squares_to_zero())
            throw std::logic_error("Resolution minimization broke d*d=0");
        projective_resolution_ = std::move(working);
    }

    /** Explicit presentation-only operation; discard higher projective maps. */
    void minimize_presentation(bool sort_if_needed = true) {
        const bool was_complete_presentation = projective_resolution_.size() == 1 &&
            has_complete_projective_resolution();
        Matrix minimized = presentation();
        if (sort_if_needed) minimized.sort_compatibly();
        minimized.minimize();
        projective_resolution_ = chain_complex_type(std::vector<Matrix>{std::move(minimized)});
        projective_resolution_complete_ = was_complete_presentation;
    }

    void minimize_injective_resolution(bool /*sort_if_needed*/ = true) {
        // Jan: define the injective/cochain degree convention and implement its
        // dual cancellation. Projective cancellation must not be applied blindly.
        throw std::logic_error("Injective-resolution minimization is not implemented");
    }

    /** Complete the stored (possibly truncated) resolution, preserving existing
     * bases. Repeated kernels stop at an injective terminal map. This requires
     * a terminating graded-kernel implementation (currently supplied for R2).
     */
    void compute_projective_resolution() {
        if (!has_presentation()) compute_presentation();
        require_presentation();
        if (has_complete_projective_resolution()) return;
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            chain_complex_type working = projective_resolution_;
            if (!working.squares_to_zero()) throw std::invalid_argument("Resolution does not square to zero");
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
        if (has_complete_projective_resolution())
            return detail::euler_dimension_at(projective_resolution_, degree);
        auto local = presentation().map_at_degree_pair(degree, true).first;
        return static_cast<index_type>(local.coKernel_basis().size());
    }
    std::vector<HilbertValue> hilbert_function(const std::vector<degree_type>& locations) const {
        std::vector<HilbertValue> result;
        result.reserve(locations.size());
        if constexpr (std::is_same_v<degree_type, r2degree>) {
            if (has_complete_projective_resolution()) {
                const auto dimensions = detail::euler_queries_r2(projective_resolution_, locations);
                for (std::size_t i = 0; i < locations.size(); ++i) result.push_back({locations[i], dimensions[i]});
                return result;
            }
        }
        for (const auto& degree : locations) result.push_back({degree, dimension_at(degree)});
        return result;
    }
    std::vector<degree_type> support_degrees() const {
        std::vector<degree_type> result = presentation().row_degrees;
        result.insert(result.end(), presentation().col_degrees.begin(), presentation().col_degrees.end());
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
        static_assert(std::is_same<degree_type, r2degree>::value,
                      "This helper is only available for R2 modules");
        vec<double> xs, ys;
        for (const auto& degree : support_degrees()) {
            xs.push_back(degree.first);
            ys.push_back(degree.second);
        }
        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
        std::sort(ys.begin(), ys.end());
        ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
        return hilbert_function_on_grid(xs, ys);
    }

    /** Query sorted, unique Cartesian axes, which need not contain all births. */
    R2HilbertGrid hilbert_function_on_grid(const vec<double>& xs, const vec<double>& ys) {
        validate_hilbert_axes(xs, ys);
        ensure_resolution_for_hilbert_grid();
        return std::as_const(*this).hilbert_function_on_grid(xs, ys);
    }

    R2HilbertGrid hilbert_function_on_grid(const vec<double>& xs, const vec<double>& ys) const {
        static_assert(std::is_same_v<degree_type, r2degree>, "This grid helper is only available for R2 modules");
        require_presentation();
        validate_hilbert_axes(xs, ys);
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
            result.values = detail::euler_grid_r2(projective_resolution_, xs, ys);
            for (const auto& row : result.values)
                for (auto value : row) result.maximum = std::max(result.maximum, value);
            return result;
        }
        // No graded kernel is available: retain the local-presentation path.
        result.values.assign(result.x_grid.size(), vec<index_type>(result.y_grid.size(), 0));
        for (std::size_t x = 0; x < result.x_grid.size(); ++x) {
            for (std::size_t y = 0; y < result.y_grid.size(); ++y) {
                degree_type degree{result.x_grid[x], result.y_grid[y]};
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
        projective_resolution_.validate_structure();
        injective_resolution_.validate_structure();
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
        if (resolution.empty()) throw std::logic_error("Requested resolution is empty");
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
template <typename index> using R2Module = Module<R2GradedSparseMatrix<index>>;

/**
 * Present H_k = ker(d_k) / im(d_{k+1}) from a chain complex.
 *
 * graded_kernel() returns kernel generators as columns in the coordinates of
 * C_k.  Each incoming-boundary column is therefore solved against that kernel
 * matrix at its own degree; those solution coordinates are the relations of
 * the homology presentation.
 */
template <typename Matrix>
Module<Matrix> homology_module(const ChainComplex<Matrix>& complex,
                                          std::size_t homological_degree = 1,
                                          bool minimize = true) {
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "homology_module requires the GradedSparseMatrix CRTP contract");
    if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
        throw std::logic_error("This graded matrix type does not implement a graded kernel");
    } else {
        complex.validate_structure();
        if (!complex.squares_to_zero())
            throw std::invalid_argument("homology_module requires d_k d_(k+1) = 0");
        if (homological_degree == 0 || homological_degree > complex.size())
            throw std::out_of_range("No outgoing differential at the requested homological degree");

        Matrix kernel_source = complex.differential(homological_degree);
        Matrix kernel_generators = kernel_source.graded_kernel();
        kernel_generators.sort_columns_lexicographically();

        using index_type = typename Matrix::index_type;
        using degree_type = typename Matrix::degree_type;
        array<index_type> relation_coordinates;
        vec<degree_type> relation_degrees;

        if (homological_degree < complex.size()) {
            const Matrix& incoming = complex.differential(homological_degree + 1);
            relation_coordinates.reserve(static_cast<std::size_t>(incoming.get_num_cols()));
            relation_degrees = incoming.col_degrees;

            for (index_type column = 0; column < incoming.get_num_cols(); ++column) {
                const degree_type& degree = incoming.col_degrees[column];
                auto local_pair = kernel_generators.map_at_degree_pair(degree, true);
                SparseMatrix<index_type> local_kernel = std::move(local_pair.first);
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
                    if (position == selected_rows.end() || *position != row) {
                        throw std::logic_error(
                            "An incoming boundary contains a row unavailable at its degree");
                    }
                    local_boundary.push_back(
                        static_cast<index_type>(std::distance(selected_rows.begin(), position)));
                }

                vec<index_type> local_solution;
                if (!local_kernel.solve_col_reduction(
                        local_boundary, local_solution, true, true, true)) {
                    throw std::invalid_argument(
                        "An incoming boundary is not contained in the computed graded kernel");
                }

                vec<index_type> global_solution;
                global_solution.reserve(local_solution.size());
                for (index_type local_index : local_solution) {
                    if (local_index < 0 ||
                        local_index >= static_cast<index_type>(selected_kernel_columns.size())) {
                        throw std::logic_error("Kernel-coordinate solver returned an invalid index");
                    }
                    global_solution.push_back(selected_kernel_columns[local_index]);
                }
                relation_coordinates.push_back(std::move(global_solution));
            }
        }

        Matrix presentation(
            static_cast<index_type>(relation_coordinates.size()),
            kernel_generators.get_num_cols(), relation_coordinates,
            relation_degrees, kernel_generators.col_degrees);
        Module<Matrix> result(std::move(presentation));
        if (minimize) result.minimize();
        return result;
    }
}

} // namespace graded_linalg
