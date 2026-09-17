/** @file submodule.hpp @brief Submodules with an explicit parent module. */
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include <grlina/homomorphism_core.hpp>
#include <grlina/graded_linear_system.hpp>

namespace graded_linalg {

template <typename Matrix>
class Submodule : public Module<Matrix> {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Submodule<Matrix> requires the GradedSparseMatrix CRTP contract");
    using module_type = Module<Matrix>;
    using index_type = typename Matrix::index_type;
    using homomorphism_type = Homomorphism<Matrix>;

private:
    homomorphism_type generator_map_;

    const Matrix& generator_matrix() const noexcept { return generator_map_.generator_lift(); }

    static homomorphism_type make_generator_map(std::shared_ptr<const module_type> parent,
                                               Matrix matrix, bool id_matrix = false) {
        return homomorphism_type::from_image_generators(std::move(parent), std::move(matrix), id_matrix);
    }

    void replace_generator_map(Matrix matrix, bool id_matrix = false) {
        generator_map_ = make_generator_map(parent(), std::move(matrix), id_matrix);
        invalidate_projective_representation();
    }

    void invalidate_projective_representation() {
        this->clear_projective_resolution();
    }

    // Work on copies and commit together: no other submodule/map sharing the
    // old parent is silently changed, and exceptions leave this object intact.
    void minimize_parent_impl(bool remove_extra_relations) {
        Matrix presentation = parent()->presentation();
        Matrix generators = generator_matrix();
        auto rows = presentation.sort_rows_with_permutation();
        generators.permute_rows_graded(rows.old_to_new);
        presentation.sort_columns_lexicographically();
        presentation.cancel_local_pairs(&generators);
        presentation.column_reduction_graded_w_deletion();
        auto minimized_parent = std::make_shared<module_type>(std::move(presentation));
        if (remove_extra_relations) minimized_parent->remove_extra_rels(false);
        if (parent()->has_injective_resolution())
            minimized_parent->set_injective_resolution(parent()->injective_resolution());
        generator_map_ = make_generator_map(std::move(minimized_parent), std::move(generators));
        invalidate_projective_representation();
    }

public:
    /** Explicit structural validation, available in every build. */
    void validate() const {
        if (!parent()) throw std::invalid_argument("A submodule requires a parent module");
        generator_matrix().validate();
        const Matrix& parent_presentation = parent()->presentation();
        if (generator_matrix().get_num_rows() != parent_presentation.get_num_rows() ||
            generator_matrix().row_degrees != parent_presentation.row_degrees) {
            throw std::invalid_argument(
                "Submodule target degrees must equal the parent module's generator degrees");
        }

    }

    Submodule(std::shared_ptr<const module_type> parent, Matrix generators)
        : Submodule(make_generator_map(std::move(parent), std::move(generators))) {}

private:
    explicit Submodule(homomorphism_type generator_map)
        : generator_map_(std::move(generator_map)) {
        GRLINA_DEBUG_CHECK(validate());
    }

public:
    const std::shared_ptr<const module_type>& parent() const noexcept { return generator_map_.target(); }
    /** Inclusion of the represented submodule into its parent. The source is
     * materialized lazily in the defining generator basis, independently of
     * any minimized presentation cached in this object's Module base.
     * Copies retain stable endpoints and survive later submodule mutations.
     */
    const homomorphism_type& generator_map() const noexcept { return generator_map_; }

    /** Move generator degrees forward by a nonnegative amount, keeping the
     * parent fixed. This is multiplication by x^amount, not a module twist.
     */
    void shift_generators(const typename Matrix::degree_type& amount) {
        using Degree = typename Matrix::degree_type;
        if (!Degree_traits<Degree>::smaller_equal(Degree{}, amount))
            throw std::invalid_argument("Submodule generator shift must be nonnegative");
        Matrix generators = generator_matrix();
        for (auto& degree : generators.col_degrees)
            Degree_traits<Degree>::add(amount, degree);
        replace_generator_map(std::move(generators), generator_map_.id_matrix());
        this->clear_injective_resolution(); // This changes the represented submodule.
    }

    /** Size of the stored module presentation, or the defining generating
     * family if no presentation has been computed yet.
     */
    index_type number_of_generators() const {
        return this->has_presentation() ? module_type::number_of_generators()
                                       : number_of_embedding_generators();
    }
    index_type number_of_embedding_generators() const noexcept { return generator_matrix().get_num_cols(); }
    /** Every supplied vector must vanish modulo the parent's relations. */
    bool is_zero() const {
        return generator_matrix().get_num_cols() == 0 ||
            solve_graded_linear_system(parent()->presentation(), generator_matrix()).has_value();
    }

    static Submodule zero(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix generators(0, presentation.get_num_rows());
        generators.row_degrees = presentation.row_degrees;
        generators.col_degrees.clear();
        generators.data.clear();
        generators.inherit_compatible_sorting(presentation);
        return Submodule(std::move(parent), std::move(generators));
    }

    static Submodule whole(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix identity(presentation.get_num_rows(), presentation.get_num_rows(), "Identity");
        identity.row_degrees = presentation.row_degrees;
        identity.col_degrees = presentation.row_degrees;
        identity.inherit_compatible_sorting(presentation);
        // Both the identity coefficients and the exact source endpoint are
        // known here; neither needs to be discovered by inspecting the matrix.
        return Submodule(homomorphism_type(parent, parent, std::move(identity), true));
    }

    /** Cheap reduction modulo the supplied parent relations, without a kernel.
     * Only matching pivots are considered, and a relation is added only when
     * its degree is <= the generator degree. The pivot strictly decreases.
     * Parent relations are neither reduced nor changed; dependencies requiring
     * combinations of their nonmatching pivots may therefore be missed.
     * No degree sorting is required: every operation checks admissibility and
     * ambient rows stay fixed. Remove zero generators in one batch at the end.
     */
    void reduce_generators_lazy() {
        GRLINA_DEBUG_CHECK(validate());
        const Matrix& presentation = parent()->presentation();
        GRLINA_DEBUG_CHECK(presentation.validate());
        Matrix generators = generator_matrix();
        array<index_type> relations_by_pivot(presentation.get_num_rows());
        for (index_type j = 0; j < presentation.get_num_cols(); ++j) {
            const auto pivot = presentation.col_last(j);
            if (pivot != -1) relations_by_pivot[pivot].push_back(j);
        }
        generators.invalidate_cached_rows();
        vec<index_type> zero_generators;
        for (index_type g = 0; g < generators.get_num_cols(); ++g) {
            auto pivot = generators.col_last(g);
            while (pivot != -1) {
                index_type reducer = -1;
                for (auto relation : relations_by_pivot[pivot])
                    if (Degree_traits<typename Matrix::degree_type>::smaller_equal(
                            presentation.col_degrees[relation], generators.col_degrees[g])) {
                        reducer = relation;
                        break;
                    }
                if (reducer == -1) break;
                generators.add_to_col(g, presentation.data[reducer]);
                pivot = generators.col_last(g);
            }
            if (pivot == -1) zero_generators.push_back(g);
        }
        if (!zero_generators.empty()) generators.delete_columns(zero_generators);
        replace_generator_map(std::move(generators));
    }

    /** Exact minimization modulo the parent relations, with optional cheap
     * preprocessing (enabled by default). Pass false to use syzygies directly.
     */
    void minimize_generators(bool lazy_preprocessing = true) {
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            invalidate_projective_representation();
            if (lazy_preprocessing) reduce_generators_lazy();
            if (generator_matrix().get_num_cols() == 0) return;
            Matrix generators = generator_matrix();
            Matrix ambient = parent()->presentation();
            const index_type relations = ambient.get_num_cols();
            ambient.append_matrix(generators);
            Matrix syzygies = ambient.graded_kernel();
            vec<index_type> redundant_generators;
            while (true) {
                index_type c = -1, r = -1;
                for (index_type j = 0; j < syzygies.get_num_cols() && c == -1; ++j)
                    for (index_type i : syzygies.data[j])
                        if (i >= relations && Degree_traits<typename Matrix::degree_type>::equals(
                                syzygies.col_degrees[j], syzygies.row_degrees[i])) {
                            c = j; r = i; break;
                        }
                if (c == -1) break;
                for (index_type j = 0; j < syzygies.get_num_cols(); ++j)
                    if (j != c && std::binary_search(syzygies.data[j].begin(), syzygies.data[j].end(), r))
                        syzygies.col_op(c, j);
                // Row r is now zero in every other syzygy. Clearing column c
                // removes its dependency without shifting any indices. Row r
                // can never reappear under later column additions, so physical
                // syzygy row/column deletion is unnecessary.
                syzygies.data[c].clear();
                redundant_generators.push_back(r - relations);
            }
            std::sort(redundant_generators.begin(), redundant_generators.end());
            if (!redundant_generators.empty()) generators.delete_columns(redundant_generators);
            replace_generator_map(std::move(generators));
            GRLINA_DEBUG_CHECK(validate());
        } else {
            // Jan: supply Matrix::graded_kernel() to enable this construction.
            throw std::logic_error("Submodule minimization requires a poset-specific graded_kernel");
        }
    }

    /** Exact containment modulo the parent's relations; no graded kernel or
     * sorted basis is needed. Parent identity fixes the ambient coordinates.
     */
    bool contains(const Submodule& other) const {
        if (parent().get() != other.parent().get())
            throw std::invalid_argument("Submodule containment requires the same parent object");
        GRLINA_DEBUG_CHECK(validate());
        GRLINA_DEBUG_CHECK(other.validate());
        Matrix spanning = parent()->presentation();
        spanning.append_matrix(generator_matrix());
        return solve_graded_linear_system(spanning, other.generator_matrix()).has_value();
    }

    bool is_contained_in(const Submodule& other) const { return other.contains(*this); }

    bool equals(const Submodule& other) const {
        return contains(other) && other.contains(*this);
    }

    Submodule sum(const Submodule& other) const {
        if (parent().get() != other.parent().get())
            throw std::invalid_argument("Submodule sum requires the same parent object");
        Matrix combined = generator_matrix();
        combined.append_matrix(other.generator_matrix());
        Submodule result(parent(), std::move(combined));
        result.minimize_generators();
        return result;
    }

    /** Intersection computed by the shared CRTP pullback/kernel construction. */
    Submodule intersection(const Submodule& other, bool minimize = true) const {
        if (parent().get() != other.parent().get())
            throw std::invalid_argument("Submodule intersection requires the same parent object");
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Intersection requires a poset-specific graded_kernel");
        } else {
            Matrix intersection_generators = parent()->presentation().submodule_intersection(
                generator_matrix(), other.generator_matrix());
            Submodule result(parent(), std::move(intersection_generators));
            if (minimize) result.minimize_generators();
            return result;
        }
    }

    /** Store this submodule's own presentation in the Module base, not in its
     * parent. Without minimization its F0 basis is exactly the generator map's columns.
     * Minimizing the stored module can change that basis; generator_map_ retains
     * the separate defining family in the parent's coordinates.
     * Explicit recomputation always uses that defining family, not an old cache.
     */
    void compute_presentation(bool minimize = false) override {
        GRLINA_DEBUG_CHECK(validate());
        if (generator_matrix().get_num_cols() == 0) {
            module_type result(Matrix(0, 0, {}, {}, {}));
            if (this->has_injective_resolution())
                result.set_injective_resolution(this->injective_resolution());
            module_type::operator=(std::move(result));
            return;
        }
        Matrix presentation = generator_map_.domain()->presentation();
        module_type result(std::move(presentation));
        if (minimize) result.minimize();
        if (this->has_injective_resolution())
            result.set_injective_resolution(this->injective_resolution());
        // Commit only after construction/minimization succeeds. Parent and
        // defining generators are unchanged; older stored resolutions are replaced.
        module_type::operator=(std::move(result));
    }

    /** Compatibility API: compute/store in place, then return a standalone copy. */
    module_type presented_module(bool minimize = true) {
        compute_presentation(minimize);
        return static_cast<const module_type&>(*this);
    }

    /** Const compatibility calls cannot populate this object's storage. */
    module_type presented_module(bool minimize = true) const {
        const bool empty = generator_matrix().get_num_cols() == 0;
        module_type result(empty ? Matrix(0, 0, {}, {}, {}) : generator_map_.domain()->presentation());
        if (minimize && !empty) result.minimize();
        if (this->has_injective_resolution())
            result.set_injective_resolution(this->injective_resolution());
        return result;
    }

    /** Image of this submodule I in X/K, with K sharing the same parent X.
     * This is I/(I intersect K), hence I/K when K is contained in I.
     * The original I, K and X are unchanged. By default no reduction occurs;
     * full minimization takes precedence if both flags are true.
     */
    Submodule submodule_quotient(const Submodule& K,
                                bool lazy_minimize = false, bool minimize = false) const {
        if (parent().get() != K.parent().get())
            throw std::invalid_argument("Submodule quotient requires the same parent object");
        auto quotient = std::make_shared<const module_type>(K.quotient_module(false));
        // The quotient projection has identity coefficients and unchanged row
        // degrees: its image is just this copy, with no matrix multiplication.
        Submodule result(std::move(quotient), generator_matrix());
        if (minimize) result.minimize_parent();
        else if (lazy_minimize) result.lazy_minimize_parent();
        return result;
    }

    /** Cancel local pairs and reduce relation columns without a graded kernel.
     * Replaces this submodule's parent and transports its defining generators.
     */
    void lazy_minimize_parent() { minimize_parent_impl(false); }

    /** As above, then remove extra relations using the graded kernel. */
    void minimize_parent() { minimize_parent_impl(true); }

    module_type quotient_module(bool minimize = true) const {
        Matrix quotient = parent()->presentation();
        quotient.append_matrix(generator_matrix());
        module_type result(std::move(quotient));
        if (minimize) result.minimize();
        return result;
    }
};

template <typename Matrix>
Submodule<Matrix> Module<Matrix>::whole_submodule() const {
    return Submodule<Matrix>::whole(this->shared_from_this());
}

template <typename Matrix>
Submodule<Matrix> Module<Matrix>::zero_submodule() const {
    return Submodule<Matrix>::zero(this->shared_from_this());
}

/** Submodule generated by the fibre at degree, with its original parent.
 * The shared parent makes the returned embedding's lifetime explicit.
 */
template <typename Matrix>
Submodule<Matrix> submodule_generated_at(std::shared_ptr<const Module<Matrix>> parent,
                                       const typename Matrix::degree_type& degree) {
    if (!parent) throw std::invalid_argument("A generated submodule requires a parent");
    const Matrix& presentation = parent->presentation();
    auto basis = presentation.basislift_at(degree);
    Matrix generators(presentation.get_num_rows(), basis);
    generators.row_degrees = presentation.row_degrees;
    generators.col_degrees.assign(basis.size(), degree);
    return Submodule<Matrix>(std::move(parent), std::move(generators));
}

} // namespace graded_linalg
