/** @file submodule.hpp @brief Submodules with an explicit parent module. */
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include <grlina/module.hpp>
#include <grlina/graded_linear_system.hpp>

namespace graded_linalg {

template <typename Matrix>
class Submodule : public Module<Matrix> {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Submodule<Matrix> requires the GradedSparseMatrix CRTP contract");
    using module_type = Module<Matrix>;
    using index_type = typename Matrix::index_type;

private:
    std::shared_ptr<const module_type> parent_;
    Matrix generators_;

    void invalidate_module_representation() {
        this->clear_projective_resolution();
        this->clear_injective_resolution();
    }

    void validate() const {
        if (!parent_) throw std::invalid_argument("A submodule requires a parent module");
        generators_.validate();
        const Matrix& parent_presentation = parent_->presentation();
        if (generators_.get_num_rows() != parent_presentation.get_num_rows() ||
            generators_.row_degrees != parent_presentation.row_degrees) {
            throw std::invalid_argument(
                "Submodule target degrees must equal the parent module's generator degrees");
        }
        if (generators_.col_degrees.size() != static_cast<std::size_t>(generators_.get_num_cols()) ||
            generators_.data.size() != static_cast<std::size_t>(generators_.get_num_cols())) {
            throw std::invalid_argument("Submodule generator matrix has inconsistent dimensions");
        }
        if (!generators_.is_graded_matrix())
            throw std::invalid_argument("Submodule generator matrix is not graded");
    }

public:
    Submodule(std::shared_ptr<const module_type> parent, Matrix generators)
        : parent_(std::move(parent)), generators_(std::move(generators)) {
        validate();
        if (!generators_.compatible_sorting_is_verified()) generators_.refresh_compatible_sorted();
    }

    const std::shared_ptr<const module_type>& parent() const noexcept { return parent_; }
    const Matrix& generators() const noexcept { return generators_; }
    /** Size of the stored module presentation, or the defining generating
     * family if no presentation has been computed yet.
     */
    index_type number_of_generators() const {
        return this->has_presentation() ? module_type::number_of_generators()
                                       : number_of_embedding_generators();
    }
    index_type number_of_embedding_generators() const noexcept { return generators_.get_num_cols(); }
    /** Every supplied vector must vanish modulo the parent's relations. */
    bool is_zero() const {
        return generators_.get_num_cols() == 0 ||
            solve_graded_linear_system(parent_->presentation(), generators_).has_value();
    }

    static Submodule zero(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix generators(0, presentation.get_num_rows());
        generators.row_degrees = presentation.row_degrees;
        generators.col_degrees.clear();
        generators.data.clear();
        generators.refresh_compatible_sorted();
        return Submodule(std::move(parent), std::move(generators));
    }

    static Submodule whole(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix identity(presentation.get_num_rows(), presentation.get_num_rows(), "Identity");
        identity.row_degrees = presentation.row_degrees;
        identity.col_degrees = presentation.row_degrees;
        identity.refresh_compatible_sorted();
        return Submodule(std::move(parent), std::move(identity));
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
        validate();
        const Matrix& presentation = parent_->presentation();
        presentation.validate();
        invalidate_module_representation();
        array<index_type> relations_by_pivot(presentation.get_num_rows());
        for (index_type j = 0; j < presentation.get_num_cols(); ++j) {
            const auto pivot = presentation.col_last(j);
            if (pivot != -1) relations_by_pivot[pivot].push_back(j);
        }
        generators_.invalidate_cached_rows();
        vec<index_type> zero_generators;
        for (index_type g = 0; g < generators_.get_num_cols(); ++g) {
            auto pivot = generators_.col_last(g);
            while (pivot != -1) {
                index_type reducer = -1;
                for (auto relation : relations_by_pivot[pivot])
                    if (Degree_traits<typename Matrix::degree_type>::smaller_equal(
                            presentation.col_degrees[relation], generators_.col_degrees[g])) {
                        reducer = relation;
                        break;
                    }
                if (reducer == -1) break;
                generators_.add_to_col(g, presentation.data[reducer]);
                pivot = generators_.col_last(g);
            }
            if (pivot == -1) zero_generators.push_back(g);
        }
        if (!zero_generators.empty()) generators_.delete_columns(zero_generators);
        if (!generators_.compatible_sorting_is_verified()) generators_.refresh_compatible_sorted();
    }

    /** Exact minimization modulo the parent relations, with optional cheap
     * preprocessing (enabled by default). Pass false to use syzygies directly.
     */
    void minimize_generators(bool lazy_preprocessing = true) {
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            invalidate_module_representation();
            if (lazy_preprocessing) reduce_generators_lazy();
            if (generators_.get_num_cols() == 0) return;
            Matrix ambient = parent_->presentation();
            const index_type relations = ambient.get_num_cols();
            ambient.append_matrix(generators_);
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
            if (!redundant_generators.empty()) generators_.delete_columns(redundant_generators);
            if (!generators_.compatible_sorting_is_verified()) generators_.refresh_compatible_sorted();
            validate();
        } else {
            // Jan: supply Matrix::graded_kernel() to enable this construction.
            throw std::logic_error("Submodule minimization requires a poset-specific graded_kernel");
        }
    }

    /** Exact containment modulo the parent's relations; no graded kernel or
     * sorted basis is needed. Parent identity fixes the ambient coordinates.
     */
    bool contains(const Submodule& other) const {
        if (parent_.get() != other.parent_.get())
            throw std::invalid_argument("Submodule containment requires the same parent object");
        validate();
        other.validate();
        Matrix spanning = parent_->presentation();
        spanning.append_matrix(generators_);
        return solve_graded_linear_system(spanning, other.generators_).has_value();
    }

    bool is_contained_in(const Submodule& other) const { return other.contains(*this); }

    bool equals(const Submodule& other) const {
        return contains(other) && other.contains(*this);
    }

    Submodule sum(const Submodule& other) const {
        if (parent_.get() != other.parent_.get())
            throw std::invalid_argument("Submodule sum requires the same parent object");
        Matrix combined = generators_;
        combined.append_matrix(other.generators_);
        Submodule result(parent_, std::move(combined));
        result.minimize_generators();
        return result;
    }

    /** Intersection computed by the shared CRTP pullback/kernel construction. */
    Submodule intersection(const Submodule& other, bool minimize = true) const {
        if (parent_.get() != other.parent_.get())
            throw std::invalid_argument("Submodule intersection requires the same parent object");
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Intersection requires a poset-specific graded_kernel");
        } else {
            Matrix intersection_generators = parent_->presentation().submodule_intersection(
                generators_, other.generators_);
            Submodule result(parent_, std::move(intersection_generators));
            if (minimize) result.minimize_generators();
            return result;
        }
    }

    /** Store this submodule's own presentation in the Module base, not in its
     * parent. Without minimization its F0 basis is exactly generators_' columns.
     * Minimizing the stored module can change that basis; generators_ remains
     * the separate defining family in the parent's coordinates.
     * Explicit recomputation always uses that defining family, not an old cache.
     */
    void compute_presentation(bool minimize = false) override {
        validate();
        if (generators_.get_num_cols() == 0) {
            module_type::operator=(module_type(Matrix(0, 0, {}, {}, {})));
            return;
        }
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Submodule presentation requires a poset-specific graded_kernel");
        } else {
            Matrix presentation = parent_->presentation().submodule_generated_by(generators_);
            module_type result(std::move(presentation));
            if (minimize) result.minimize();
            // Commit only after construction/minimization succeeds. Parent and
            // defining generators are unchanged; older stored resolutions are replaced.
            module_type::operator=(std::move(result));
        }
    }

    /** Compatibility API: compute/store in place, then return a standalone copy. */
    module_type presented_module(bool minimize = true) {
        compute_presentation(minimize);
        return static_cast<const module_type&>(*this);
    }

    /** Const compatibility calls cannot populate this object's storage. */
    module_type presented_module(bool minimize = true) const {
        Submodule working(parent_, generators_);
        working.compute_presentation(minimize);
        return static_cast<const module_type&>(working);
    }

    module_type quotient_module(bool minimize = true) const {
        Matrix quotient = parent_->presentation();
        quotient.append_matrix(generators_);
        module_type result(std::move(quotient));
        if (minimize) result.minimize();
        return result;
    }
};

} // namespace graded_linalg
