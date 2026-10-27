/** @file module/module_base.hpp @brief Modules backed by optional projective/injective resolutions. */
#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <grlina/proj_chain_complex.hpp>
#include <grlina/matrices/r2.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/quiver_representations.hpp>
#include <grlina/graded_kernel.hpp>
#include <grlina/matrices/io.hpp>

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
    using stored_degree_type = matrix_degree_t<Matrix>;
    using index_type = matrix_index_t<Matrix>;
    using chain_complex_type = ChainComplex<Matrix>;

    chain_complex_type proj_res_;
    chain_complex_type inj_res_; // not used at the moment
    bool res_complete_ = false;

    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = default;
    Module(Module&&) = default;
    Module& operator=(const Module&) = default;
    Module& operator=(Module&&) = default;
    explicit Module(Matrix presentation)
        : proj_res_(std::vector<Matrix>{std::move(presentation)}) {}
    explicit Module(chain_complex_type resolution,
                    ResolutionKind kind = ResolutionKind::projective,
                    ResolutionCompleteness completeness = ResolutionCompleteness::truncated) {
        if (kind == ResolutionKind::projective) {
            proj_res_ = std::move(resolution);
            res_complete_ = completeness == ResolutionCompleteness::complete;
        }
        else inj_res_ = std::move(resolution);
    }
    Module(chain_complex_type projective, chain_complex_type injective)
        : proj_res_(std::move(projective)), inj_res_(std::move(injective)) {
    }
    explicit Module(const std::string& path, bool sort_if_needed = false)
        : proj_res_(chain_complex_type::from_file(path, sort_if_needed)) {}
    explicit Module(std::istream& input, bool sort_if_needed = false)
        : proj_res_(chain_complex_type::from_stream(input, sort_if_needed)) {}

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

    bool has_presentation() const noexcept { return !proj_res_.empty(); }
    Matrix& presentation() { return proj_res_[0]; }
    const Matrix& presentation() const { return proj_res_[0]; }

    /** Canonical submodules retaining this exact parent. Requires shared_ptr
     * ownership (otherwise throws std::bad_weak_ptr) and a presentation.
     * Include grlina/module/submodule.hpp for the definitions.
     */
    Submodule<Matrix> whole_submodule() const;
    Submodule<Matrix> zero_submodule() const;

    index_type num_of_gens() const { return presentation().get_num_rows(); }
    index_type num_of_rels() const { return presentation().get_num_cols(); }

    /** Restrict to generators and relations born by degree.
     * Return the fibre presentation with rows renumbered locally, together
     * with their original generator indices. Its cokernel is the module fibre.
     */
    std::pair<SparseMatrix<index_type, matrix_storage_t<Matrix>>, vec<index_type>>
    local_presentation_at(const degree_type& degree) const {
        return presentation().map_at_degree_pair(degree);
    }

    void print_presentation(bool suppress_description = false) const {
        presentation().print_graded(suppress_description);
    }
    void print_degrees() const { presentation().print_degrees(); }

    /** Remove chosen relations from this presentation; may enlarge the module. */
    void remove_relations(vec<index_type> indices) {
        normalize_indices(indices, num_of_rels());
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().delete_columns(indices);
    }

    /** Kill the chosen presentation generators (take the corresponding quotient). */
    void quotient_by_generators(vec<index_type> indices) {
        normalize_indices(indices, num_of_gens());
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().delete_rows(indices);
    }

    /** Keep the first count generators and kill the remaining ones.
     * Equivalent to the matrix cull_columns(count, false), which truncates rows.
     */
    void quotient_by_tail_generators(index_type count) {
        GRLINA_DEBUG_CHECK(if (count < 0 || count > num_of_gens())
            throw std::out_of_range("Number of retained generators out of range"));
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().cull_columns(count, false);
    }

    /** Add a homogeneous relation in the current generator coordinates. */
    void add_relation(const vec<index_type>& coefficients, const degree_type& degree) {
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().append_column(coefficients, degree);
    }

    /** Cancel local generator/relation pairs without requiring a graded kernel. */
    void semi_minimize_presentation() {
        Matrix matrix = presentation();
        matrix.sort_compatibly();
        matrix.semi_minimize();
        proj_res_ = chain_complex_type(std::vector<Matrix>{std::move(matrix)});
        res_complete_ = false;
    }

    bool is_presentation_minimal() const {
        return presentation().is_minimal();
    }

    /** Bounds of presentation degrees, NOT bounds of the module's support. */
    std::pair<degree_type, degree_type> bounding_box() const {
        GRLINA_DEBUG_CHECK(if (num_of_gens() == 0 && num_of_rels() == 0)
            throw std::invalid_argument("Empty presentation has no degree bounds"));
        return presentation().bounding_box();
    }

    /** An n-by-n sampling grid over the R2 presentation degree bounds. */
    vec<degree_type> get_equidistant_grid(int n) const {
        GRLINA_DEBUG_CHECK(if (n < 0) throw std::invalid_argument("Grid size must be nonnegative"));
        if (n == 0 || (num_of_gens() == 0 && num_of_rels() == 0)) return {};
        return presentation().get_equidistant_grid(n);
    }

    /** Restrict to a finite degree poset; by default use presentation degrees. */
    QuiverRepresentation<index_type, degree_type, SparseMatrix<index_type, matrix_storage_t<Matrix>>>
    to_quiver(vec<degree_type> vertices = {}, array<index_type> edges = {}) const {
        Matrix matrix = presentation();
        return matrix.induced_quiver_rep(std::move(vertices), std::move(edges));
    }

    virtual void sort_compatibly() {
        proj_res_.sort_compatibly();
        if (!inj_res_.empty()) inj_res_.sort_compatibly();
    }
    template <typename Compare>
    void sort_compatibly(Compare compare) {
        proj_res_.sort_compatibly(compare);
        if (!inj_res_.empty()) inj_res_.sort_compatibly(compare);
    }

    /** Minimize all stored projective maps, or just the presentation if there is one map. */
    void minimize(bool sort_if_needed = true) {
        if (proj_res_.size() > 1)
            minimize_resolution(sort_if_needed);
        else minimize_presentation(sort_if_needed);
    }

    /** Minimize the stored projective resolution in place.
     * Extend a truncation by one kernel map before cancellation, then remove
     * the added tail. Only the last differential is copied for its kernel.
     */
    virtual void minimize_resolution(bool sort_if_needed = true) {
        if (res_complete_) {
            proj_res_.minimize(sort_if_needed);
            return;
        }
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            Matrix terminal = proj_res_.differentials().back();
            proj_res_.push_differential(terminal.graded_kernel());
            try {
                proj_res_.minimize(sort_if_needed);
            } catch (...) {
                proj_res_.differentials().pop_back();
                throw;
            }
            proj_res_.differentials().pop_back();
        } else {
            throw std::logic_error("Minimizing a truncated resolution requires a poset-specific graded_kernel");
        }
    }


    /** Presentation-only minimization: discard higher projective maps while
     * preserving the independently stored injective resolution.
     */
    virtual void minimize_presentation(bool sort_if_needed = true) {
        const bool was_complete_presentation = proj_res_.size() == 1 &&
            res_complete_;
        Matrix minimized = presentation();
        if (sort_if_needed) minimized.sort_compatibly();
        minimized.column_reduction_graded_w_deletion();
        minimized.cancel_local_pairs();
        minimized.remove_redundant_relations();
        proj_res_ = chain_complex_type(std::vector<Matrix>{std::move(minimized)});
        res_complete_ = was_complete_presentation;
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
    virtual void compute_projective_resolution() {
        if (res_complete_) return;
        if (proj_res_[proj_res_.size() - 1].get_num_cols() == 0) {
            res_complete_ = true;
            return;
        }
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            chain_complex_type working = proj_res_;

            while (true) {
                Matrix source = working[working.size() - 1]; // kernel computation is destructive
                Matrix kernel = source.graded_kernel();
                const bool terminal = kernel.get_num_cols() == 0;
                // Preserve the established d1,d2 shape for a presentation,
                // even when d2 has zero columns. No further zero maps are needed.
                if (!terminal || working.size() == 1) working.push_differential(std::move(kernel));
                if (terminal) break;
            }
            proj_res_ = std::move(working);
            res_complete_ = true;
        } else {
            throw std::logic_error("This graded matrix type does not implement a graded kernel");
        }
    }

    void shift(const degree_type& amount) {
        for (auto& differential : proj_res_.differentials())
            differential.shift(amount);
        for (auto& differential : inj_res_.differentials())
            differential.shift(amount);
    }
    /** Snap every differential in both stored resolutions independently.
     * They retain storage and the completeness flag; compatible adjacent
     * bases and exactness remain the caller's responsibility.
     */
    template <typename Scalar>
    void snap_to_grid(const vec<Scalar>& x_grid, const vec<Scalar>& y_grid) {
        for (auto& differential : proj_res_.differentials())
            differential.snap_to_grid(x_grid, y_grid);
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_grid(x_grid, y_grid);
    }
    template <typename Scalar, std::size_t Dimension>
    void snap_to_grid(const std::array<vec<Scalar>, Dimension>& axes) {
        for (auto& differential : proj_res_.differentials())
            differential.snap_to_grid(axes);
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_grid(axes);
    }
    void snap_to_equidistant_grid(int n, bool end_inclusive = false) {
        for (auto& differential : proj_res_.differentials())
            differential.snap_to_equidistant_grid(n, end_inclusive);
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_equidistant_grid(n, end_inclusive);
    }
    /** Add boundary relations to the presentation and discard stale resolutions. */
    void bound_support(const degree_type& bound) {
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().bound_support(bound);
    }
    void cut_above(double x_cutoff, double y_cutoff) {
        for (auto& differential : proj_res_.differentials())
            differential.cut_above(x_cutoff, y_cutoff);
        for (auto& differential : inj_res_.differentials())
            differential.cut_above(x_cutoff, y_cutoff);
    }
    template <std::size_t Dimension>
    void cut_above(const std::array<double, Dimension>& cutoffs) {
        for (auto& differential : proj_res_.differentials())
            differential.cut_above(cutoffs);
        for (auto& differential : inj_res_.differentials())
            differential.cut_above(cutoffs);
    }

    template <typename OutputStream>
    void to_stream(OutputStream& output, ResolutionKind kind = ResolutionKind::projective) const {
        const auto& resolution = kind == ResolutionKind::projective
            ? proj_res_ : inj_res_;
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
            cycles.reindex_grid(presentation.grid_);
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

        using index_type = matrix_index_t<Matrix>;
        using degree_type = matrix_degree_t<Matrix>;
        matrix_storage_t<Matrix> relation_coordinates;
        vec<degree_type> relation_degrees;

        if (homological_degree < complex.size()) {
            Matrix aligned_incoming;
            const Matrix* incoming_ptr = &complex.differential(homological_degree + 1);
            if constexpr (matrix_grid_backed_v<Matrix>) {
                aligned_incoming = *incoming_ptr;
                merge_grids(kernel_generators, aligned_incoming);
                incoming_ptr = &aligned_incoming;
            }
            const Matrix& incoming = *incoming_ptr;
            relation_coordinates.reserve(static_cast<std::size_t>(incoming.get_num_cols()));
            relation_degrees.assign(incoming.col_degrees.begin(), incoming.col_degrees.end());

            for (index_type column = 0; column < incoming.get_num_cols(); ++column) {
                const degree_type degree = incoming.col_degrees[column];
                auto local_pair = [&] {
                    if constexpr (matrix_grid_backed_v<Matrix>)
                        return kernel_generators.map_at_grid_degree_pair(degree, true);
                    else return kernel_generators.map_at_degree_pair(degree, true);
                }();
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
