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
class Submodule {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Submodule<Matrix> requires the GradedSparseMatrix CRTP contract");
    using module_type = Module<Matrix>;
    using index_type = typename Matrix::index_type;

private:
    std::shared_ptr<const module_type> parent_;
    Matrix generators_;

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
    index_type number_of_generators() const noexcept { return generators_.get_num_cols(); }
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

    /** Minimize generators modulo the parent's relations using graded syzygies. */
    void minimize_generators() {
        if constexpr (has_matrix_graded_kernel<Matrix>::value) {
            Matrix ambient = parent_->presentation();
            const index_type relations = ambient.get_num_cols();
            ambient.append_matrix(generators_);
            Matrix syzygies = ambient.graded_kernel();
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
                vec<index_type> row{r}, column{c}, generator{r - relations};
                syzygies.delete_rows(row);
                syzygies.delete_columns(column);
                generators_.delete_columns(generator);
            }
            generators_.refresh_compatible_sorted();
            validate();
        } else {
            // Jan: supply Matrix::graded_kernel() to enable this construction.
            throw std::logic_error("Submodule minimization requires a poset-specific graded_kernel");
        }
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

    module_type presented_module(bool minimize = true) const {
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Submodule presentation requires a poset-specific graded_kernel");
        } else {
            Matrix presentation = parent_->presentation().submodule_generated_by(generators_);
            module_type result(std::move(presentation));
            if (minimize) result.minimize();
            return result;
        }
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
