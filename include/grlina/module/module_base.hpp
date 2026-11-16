/** @file module/module_base.hpp @brief Modules backed by optional projective/injective resolutions. */
#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <list>
#include <memory>
#include <optional>
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
#include <grlina/graded_linear_system.hpp>
#include <grlina/matrices/io.hpp>

// AIDA remains optional: include aida_interface.hpp when calling decompose().
namespace aida {
template <typename Index> struct AIDA_functor;
template <class Matrix> struct Block_t;
}

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
    using degree_type = typename Matrix::degree_type;
    using stored_degree_type = base_degree<Matrix>;
    using index_type = matrix_index_t<Matrix>;
    using chain_complex_type = ChainComplex<Matrix>;

    chain_complex_type proj_res_;
    chain_complex_type inj_res_; // not used at the moment
    bool res_complete_ = false;
    // Columns lift the current presentation generators to cycles in C_q.
    // Retained by homology with keep_cycles=true and transported to summands.
    std::optional<Matrix> cycles;
    // nullopt means not computed; an empty vector is the zero module's decomposition.
    std::optional<vec<Module>> decomposition;

protected:
    void include_cycles(vec<Matrix*>& images) {
        if (cycles && std::find(images.begin(), images.end(), &*cycles) == images.end())
            images.push_back(&*cycles);
    }
public:

    Module() = default;
    virtual ~Module() = default;
    Module(const Module&) = default;
    Module(Module&&) = default;
    Module& operator=(const Module&) = default;
    Module& operator=(Module&&) = default;
    explicit Module(Matrix presentation)
        : proj_res_(std::move(presentation)) {}
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

    /** Decompose a minimal, sorted copy of the presentation using AIDA.
     * Include aida_interface.hpp and link AIDA to instantiate this member.
     * The dependent default keeps ordinary Module users independent of AIDA.
     * With keep_cycles, existing representatives are transported to each
     * summand; absent cycles are not synthesized. The parent's presentation,
     * cycles and resolutions are unchanged.
     * No presentation leaves decomposition unset. Direct edits through public
     * matrices require the caller to reset decomposition themselves.
     */
    template <class Decomposer = aida::AIDA_functor<index_type>>
    void decompose(bool keep_cycles = false) {
        if (!has_presentation()) {
            decomposition.reset();
            return;
        }
        Module working(presentation());
        if (keep_cycles) working.cycles = cycles;
        working.minimize_presentation();
        vec<Module> summands;
        if (working.num_of_gens() == 0) {
            decomposition = std::move(summands);
            return;
        }

        // R2 (including CSC storage) uses editable coordinate columns in AIDA.
        constexpr bool r2 = std::is_same_v<Matrix,
            R2GradedSparseMatrix<index_type, matrix_storage_t<Matrix>>>;
        using WorkingMatrix = std::conditional_t<r2,
            CoordGradedSparseMatrix<double, 2, index_type>, Matrix>;
        WorkingMatrix matrix = [&]() {
            if constexpr (r2) {
                const auto& p = working.presentation();
                array<index_type> columns;
                columns.reserve(p.get_num_cols());
                for (index_type j = 0; j < p.get_num_cols(); ++j)
                    columns.push_back(p.get_col(j));
                return WorkingMatrix(p.get_num_cols(), p.get_num_rows(), columns,
                                     p.col_degrees, p.row_degrees);
            } else return working.presentation();
        }();
        matrix.compatibly_sorted = working.presentation().compatibly_sorted;
        Decomposer decomposer;
        decomposer.config.save_base_change = working.cycles.has_value();
        std::list<aida::Block_t<WorkingMatrix>> blocks;
        decomposer(matrix, blocks);

        if (working.cycles) {
            // A row addition source -> target changes the generator basis by
            // the inverse column addition target -> source, in recorded order.
            for (const auto& op : decomposer.base_changes.back()->performed_row_ops)
                working.cycles->col_op(op.second, op.first);
        }
        summands.reserve(blocks.size());
        for (auto& block : blocks) {
            if constexpr (r2) {
                Matrix p(block.get_num_cols(), block.get_num_rows(), block.data,
                         block.col_degrees, block.row_degrees);
                p.compatibly_sorted = block.compatibly_sorted;
                summands.emplace_back(std::move(p));
            } else {
                summands.emplace_back(std::move(static_cast<Matrix&>(block)));
            }
            if (working.cycles)
                summands.back().cycles = working.cycles->restricted_domain_copy(block.rows);
        }
        decomposition = std::move(summands);
    }

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

    /** Dimension at one degree, or dimensions in the supplied query order. */
    index_type dimension_at(const degree_type& degree) const;
    vec<index_type> dimension_at(const vec<degree_type>& degrees) const;

    /** Numerical Hilbert function on the induced (or stored) two-parameter grid. */
    auto hilbert_function() const;

    /** Numerical values on sorted Cartesian axes, retaining their scalar type.
     * The dependent Degree keeps non-coordinate Module specializations usable.
     */
    template <typename Degree = degree_type>
    auto hilbert_function(const vec<typename Degree::value_type>& xs,
                          const vec<typename Degree::value_type>& ys) const;

    /** PNG bytes in memory. Width/height count heatmap pixels; labels add margins.
     * Automatic bounds include padding (a fraction of each axis span).
     * Axis captions are empty unless supplied explicitly.
     */
    vec<unsigned char> hilbert_function_image(std::size_t width = 800,
        std::size_t height = 600, double padding = 0.05,
        const std::string& x_label = "", const std::string& y_label = "") const;

    /** PNG restricted to the supplied axes' range, without padding.
     * All induced-grid changes inside that range remain visible.
     */
    template <typename Degree = degree_type>
    vec<unsigned char> hilbert_function_image(const vec<typename Degree::value_type>& xs,
        const vec<typename Degree::value_type>& ys,
        std::size_t width = 800, std::size_t height = 600,
        const std::string& x_label = "", const std::string& y_label = "") const;

    void print_presentation(bool suppress_description = false) const {
        presentation().print_graded(suppress_description);
    }
    void print_degrees() const { presentation().print_degrees(); }

    /** Remove chosen relations from this presentation; may enlarge the module. */
    void remove_relations(vec<index_type> indices) {
        decomposition.reset();
        normalize_indices(indices, num_of_rels());
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().delete_columns(indices);
    }

    /** Kill the chosen presentation generators (take the corresponding quotient). */
    void quotient_by_generators(vec<index_type> indices, vec<Matrix*> images = {}) {
        decomposition.reset();
        normalize_indices(indices, num_of_gens());
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        include_cycles(images);
        presentation().delete_rows(indices, nullptr, images);
    }

    /** Keep the first count generators and kill the remaining ones.
     * Equivalent to the matrix cull_columns(count, false), which truncates rows.
     */
    void quotient_by_tail_generators(index_type count, vec<Matrix*> images = {}) {
        decomposition.reset();
        GRLINA_DEBUG_CHECK(if (count < 0 || count > num_of_gens())
            throw std::out_of_range("Number of retained generators out of range"));
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        include_cycles(images);
        presentation().cull_columns(count, false, nullptr, images);
    }

    /** Add a homogeneous relation in the current generator coordinates. */
    void add_relation(const vec<index_type>& coefficients, const degree_type& degree) {
        decomposition.reset();
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        presentation().append_column(coefficients, degree);
    }

    /** Cancel local generator/relation pairs without requiring a graded kernel. */
    void semi_minimize_presentation(Matrix* elements = nullptr, vec<Matrix*> images = {}) {
        include_cycles(images);
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        auto& matrix = presentation();
        matrix.sort_compatibly(elements, images);
        matrix.semi_minimize(elements, images);
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

    /** Incoming elements use generator rows; outgoing images use generator columns.
     * Initialize an outgoing image to identity to obtain the new-to-old basis map.
     * Existing cycles are always updated directly, without matrix multiplication.
     */
    virtual void sort_compatibly(Matrix* elements = nullptr, vec<Matrix*> images = {}) {
        sort_compatibly(TraitLinearOrder<stored_degree_type>{Degree_traits<stored_degree_type>::lex_lambda()},
                        elements, std::move(images));
    }
    template <typename Compare, std::enable_if_t<std::is_convertible_v<Compare,
        std::function<bool(const stored_degree_type&, const stored_degree_type&)>>, int> = 0>
    void sort_compatibly(Compare compare, Matrix* elements = nullptr, vec<Matrix*> images = {}) {
        include_cycles(images);
        proj_res_.sort_compatibly(compare, nullptr, elements, images);
        if (!inj_res_.empty()) inj_res_.sort_compatibly(compare);
    }

    /** Minimize all stored projective maps, or just the presentation if there is one map. */
    void minimize(bool sort_if_needed = true, Matrix* elements = nullptr, vec<Matrix*> images = {}) {
        if (proj_res_.size() > 1)
            minimize_resolution(sort_if_needed, elements, std::move(images));
        else minimize_presentation(sort_if_needed, elements, std::move(images));
    }

    /** Minimize the stored projective resolution in place.
     * Extend a truncation by one kernel map before cancellation, then remove
     * the added tail. Only the last differential is copied for its kernel.
     */
    virtual void minimize_resolution(bool sort_if_needed = true, Matrix* elements = nullptr,
                                     vec<Matrix*> images = {}) {
        include_cycles(images);
        if (res_complete_) {
            proj_res_.minimize(sort_if_needed, nullptr, elements, images);
            return;
        }
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            Matrix terminal = proj_res_.differentials().back();
            proj_res_.push_differential(terminal.graded_kernel());
            try {
                proj_res_.minimize(sort_if_needed, nullptr, elements, images);
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
    virtual void minimize_presentation(bool sort_if_needed = true, Matrix* elements = nullptr,
                                       vec<Matrix*> images = {}) {
        include_cycles(images);
        const bool was_complete_presentation = proj_res_.size() == 1 &&
            res_complete_;
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        auto& minimized = presentation();
        if (sort_if_needed) minimized.sort_compatibly(elements, images);
        GRLINA_DEBUG_CHECK(if (!minimized.compatibly_sorted)
            throw std::invalid_argument("Presentation minimization requires compatibly sorted rows and columns"));
        minimized.minimize_variant(elements, images);
        res_complete_ = was_complete_presentation;
    }

    /** Minimise a presentation by cancellation followed by image generation.
     * Delegates to the alternative beside Matrix::minimize(), supporting all
     * parameter counts. Cycles and supplied generator maps are transported.
     * Requires exactly one stored projective map. Use minimize_resolution()
     * when higher maps are already available and should be transported.
     */
    virtual void minimize_presentation_by_image_generators(bool sort_if_needed = true,
            Matrix* elements = nullptr, vec<Matrix*> images = {}) {
        if (proj_res_.size() != 1)
            throw std::invalid_argument("Presentation-only minimization requires one presentation and no higher projective maps");
        include_cycles(images);
        auto& minimized = presentation();
        if (sort_if_needed) minimized.sort_compatibly(elements, images);
        GRLINA_DEBUG_CHECK(if (!minimized.compatibly_sorted)
            throw std::invalid_argument("Presentation minimization requires compatibly sorted rows and columns"));
        minimized.minimize_by_image_generators(elements, images);
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
            while (true) {
                Matrix source = proj_res_.differentials().back(); // Preserve the stored differential during destructive reduction.
                Matrix kernel = source.graded_kernel();
                const bool terminal = kernel.get_num_cols() == 0;
                // Preserve the established d1,d2 shape for a presentation,
                // even when d2 has zero columns. No further zero maps are needed.
                if (!terminal || proj_res_.size() == 1) proj_res_.push_differential(std::move(kernel));
                if (terminal) break;
            }
            res_complete_ = true;
        } else {
            throw std::logic_error("This graded matrix type does not implement a graded kernel");
        }
    }

    void shift(const degree_type& amount) {
        if (decomposition)
            for (auto& summand : *decomposition) summand.shift(amount);
        if (cycles) cycles->shift(amount);
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
    void snap_to_grid(const vec<Scalar>& x_grid, const vec<Scalar>& y_grid, vec<Matrix*> images = {}) {
        decomposition.reset();
        include_cycles(images);
        for (std::size_t i = 0; i < proj_res_.size(); ++i)
            proj_res_[i].snap_to_grid(x_grid, y_grid, i == 0 ? images : vec<Matrix*>{});
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_grid(x_grid, y_grid);
    }
    template <typename Scalar, std::size_t Dimension>
    void snap_to_grid(const std::array<vec<Scalar>, Dimension>& axes, vec<Matrix*> images = {}) {
        decomposition.reset();
        include_cycles(images);
        for (std::size_t i = 0; i < proj_res_.size(); ++i)
            proj_res_[i].snap_to_grid(axes, i == 0 ? images : vec<Matrix*>{});
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_grid(axes);
    }
    void snap_to_equidistant_grid(int n, bool end_inclusive = false, vec<Matrix*> images = {}) {
        decomposition.reset();
        include_cycles(images);
        for (std::size_t i = 0; i < proj_res_.size(); ++i)
            proj_res_[i].snap_to_equidistant_grid(n, end_inclusive, i == 0 ? images : vec<Matrix*>{});
        for (auto& differential : inj_res_.differentials())
            differential.snap_to_equidistant_grid(n, end_inclusive);
    }
    /** Add boundary relations to the presentation and discard stale resolutions. */
    void bound_support(const degree_type& bound, vec<Matrix*> images = {}) {
        decomposition.reset();
        proj_res_.differentials().resize(1);
        res_complete_ = false;
        inj_res_.clear();
        include_cycles(images);
        presentation().bound_support(bound, images);
    }
    void cut_above(double x_cutoff, double y_cutoff, vec<Matrix*> images = {}) {
        decomposition.reset();
        include_cycles(images);
        for (std::size_t i = 0; i < proj_res_.size(); ++i)
            proj_res_[i].cut_above(x_cutoff, y_cutoff, i == 0 ? images : vec<Matrix*>{});
        for (auto& differential : inj_res_.differentials())
            differential.cut_above(x_cutoff, y_cutoff);
    }
    template <std::size_t Dimension>
    void cut_above(const std::array<double, Dimension>& cutoffs, vec<Matrix*> images = {}) {
        decomposition.reset();
        include_cycles(images);
        for (std::size_t i = 0; i < proj_res_.size(); ++i)
            proj_res_[i].cut_above(cutoffs, i == 0 ? images : vec<Matrix*>{});
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

/** Construct the homology output and optionally retain cycle representatives.
 * Kept outside the algorithms so every path applies the same output options.
 */
template <typename Matrix>
Module<Matrix> finish_homology_group(Matrix presentation, Matrix representatives,
                                    bool minimize, bool keep_cycles) {
    Module<Matrix> result(std::move(presentation));
    if (keep_cycles) result.cycles = std::move(representatives);
    if (minimize) {
        if constexpr (is_bigraded_coordinate_degree<typename Matrix::degree_type>::value)
            result.minimize_presentation_by_image_generators();
        else result.minimize();
    }
    return result;
}

/** H_0 = C_0 / im(d_1): no kernel computation is needed. */
template <typename Matrix>
Module<Matrix> homology_group_degree_zero(const ChainComplex<Matrix>& complex,
                                         bool minimize = true, bool keep_cycles = false) {
    Matrix presentation = complex.empty()
        ? GradedMatrixIO<Matrix>::make_matrix(0, 0) : complex.differential(1);
    Matrix representatives;
    if (keep_cycles) {
        representatives = Matrix(geometric_row_degrees(presentation));
        if constexpr (matrix_grid_backed_v<Matrix>) representatives.reindex_grid(presentation.grid_);
        representatives.compatibly_sorted = presentation.compatibly_sorted;
    }
    return finish_homology_group(std::move(presentation), std::move(representatives), minimize, keep_cycles);
}

/** Two-coordinate kernels are free. Express all incoming boundaries in this
 * triangular kernel basis; their coordinates are the homology relations.
 * There are no relations among these kernel generators to compute separately.
 */
template <typename Matrix>
Module<Matrix> homology_group_bigraded(const ChainComplex<Matrix>& complex,
                                      std::size_t homological_degree = 1,
                                      bool minimize = true, bool keep_cycles = false) {
    static_assert(is_bigraded_coordinate_degree<typename Matrix::degree_type>::value,
                  "Bigraded homology requires exactly two coordinate parameters");
    if (homological_degree == 0) return homology_group_degree_zero(complex, minimize, keep_cycles);
    GRLINA_DEBUG_CHECK(if (homological_degree > complex.size())
        throw std::out_of_range("No outgoing differential at the requested homological degree"));
    Matrix outgoing = complex.differential(homological_degree);
    Matrix kernel_generators = outgoing.graded_kernel();
    kernel_generators.sort_columns_lexicographically();
    Matrix incoming;
    if (homological_degree < complex.size()) {
        incoming = complex.differential(homological_degree + 1);
        if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(kernel_generators, incoming);
    }
    Matrix presentation = empty_matrix_like(kernel_generators,
        matrix_index_t<Matrix>(0), kernel_generators.get_num_cols());
    presentation.row_degrees = kernel_generators.col_degrees;
    if (homological_degree < complex.size()) {
        auto coordinates = solve_graded_system_by_back_substitution(
            kernel_generators, std::move(incoming));
        if (!coordinates) throw std::invalid_argument("Incoming boundaries are outside the graded kernel span");
        presentation = std::move(*coordinates);
    }
    return finish_homology_group(std::move(presentation), std::move(kernel_generators), minimize, keep_cycles);
}

/** Express incoming boundaries in possibly dependent graded cycle generators.
 * Coordinate degrees use the shared graded solver. For a general degree poset,
 * solve each column using just the generators born by that boundary's grade.
 */
template <typename Matrix>
Matrix homology_boundary_coordinates(const Matrix& kernel_generators, Matrix incoming) {
    if constexpr (is_coordinate_solver_degree<typename Matrix::degree_type>::value) {
        auto coordinates = solve_graded_linear_system(kernel_generators, std::move(incoming));
        if (!coordinates) throw std::invalid_argument("Incoming boundaries are outside the graded kernel span");
        return std::move(*coordinates);
    } else {
        Matrix coordinates = empty_matrix_like(kernel_generators, incoming.get_num_cols(), kernel_generators.get_num_cols());
        set_geometric_degrees(coordinates, geometric_col_degrees(incoming), geometric_col_degrees(kernel_generators));
        for (matrix_index_t<Matrix> column = 0; column < incoming.get_num_cols(); ++column) {
            vec<matrix_index_t<Matrix>> selected;
            array<matrix_index_t<Matrix>> eligible_columns;
            for (matrix_index_t<Matrix> candidate = 0; candidate < kernel_generators.get_num_cols(); ++candidate) {
                if (Degree_traits<base_degree<Matrix>>::smaller_equal(
                        kernel_generators.col_degrees[candidate], incoming.col_degrees[column])) {
                    selected.push_back(candidate);
                    eligible_columns.push_back(kernel_generators.get_col(candidate));
                }
            }
            SparseMatrix<matrix_index_t<Matrix>> local(static_cast<matrix_index_t<Matrix>>(selected.size()),
                kernel_generators.get_num_rows(), eligible_columns);
            auto boundary = incoming.get_col(column);
            vec<matrix_index_t<Matrix>> solution;
            if (!local.solve_col_reduction(boundary, solution, true, true, true))
                throw std::invalid_argument("Incoming boundaries are outside the graded kernel span");
            for (auto& index : solution) index = selected.at(index);
            coordinates.set_col(column, solution);
        }
        return coordinates;
    }
}

/** Present H_k = ker(d_k) / im(d_{k+1}).
 * Degree zero and two-coordinate kernels have their own implementations above.
 * In the general case kernel generators can have relations. Include both those
 * relations and coordinates of incoming boundaries in the presentation.
 * With keep_cycles, Module::cycles retains representatives in the original C_k.
 */
template <typename Matrix>
Module<Matrix> homology_group(const ChainComplex<Matrix>& complex,
                              std::size_t homological_degree = 1, bool minimize = true,
                              bool keep_cycles = false) {
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "homology_group requires the graded sparse matrix interface");
    if (homological_degree == 0) return homology_group_degree_zero(complex, minimize, keep_cycles);
    if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
        throw std::logic_error("This graded matrix type does not implement a graded kernel");
    } else if constexpr (is_bigraded_coordinate_degree<typename Matrix::degree_type>::value) {
        return homology_group_bigraded(complex, homological_degree, minimize, keep_cycles);
    } else {
        GRLINA_DEBUG_CHECK(if (homological_degree > complex.size())
            throw std::out_of_range("No outgoing differential at the requested homological degree"));
        Matrix outgoing = complex.differential(homological_degree);
        Matrix kernel_generators = outgoing.graded_kernel();
        kernel_generators.sort_columns_lexicographically();
        Matrix incoming;
        if (homological_degree < complex.size()) {
            incoming = complex.differential(homological_degree + 1);
            if constexpr (matrix_grid_backed_v<Matrix>) merge_grids(kernel_generators, incoming);
        }
        Matrix presentation = empty_matrix_like(kernel_generators,
            matrix_index_t<Matrix>(0), kernel_generators.get_num_cols());
        presentation.row_degrees = kernel_generators.col_degrees;
        if (homological_degree < complex.size()) {
            presentation = homology_boundary_coordinates(kernel_generators, std::move(incoming));
        }
        // Quotient by dependencies among the cycle generators as well as by
        // incoming boundaries. These dependencies matter beyond two parameters.
        Matrix kernel_copy = kernel_generators;
        presentation.append_move_matrix(kernel_copy.graded_kernel());
        return finish_homology_group(std::move(presentation), std::move(kernel_generators), minimize, keep_cycles);
    }
}

} // namespace graded_linalg

// Hilbert members are defined separately, after Module is complete.
#include <grlina/invariants/hilbert_function.hpp>
