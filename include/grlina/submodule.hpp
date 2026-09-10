/** @file submodule.hpp @brief Submodules with an explicit parent module. */
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include <grlina/module.hpp>

namespace graded_linalg {

template <typename Matrix>
class Submodule {
public:
    using module_type = PersistenceModule<Matrix>;
    using index_type = typename Matrix::index_type;

private:
    std::shared_ptr<const module_type> parent_;
    Matrix generators_;

    void validate() const {
        if (!parent_) throw std::invalid_argument("A submodule requires a parent module");
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
        : parent_(std::move(parent)), generators_(std::move(generators)) { validate(); }

    const std::shared_ptr<const module_type>& parent() const noexcept { return parent_; }
    const Matrix& generators() const noexcept { return generators_; }
    index_type number_of_generators() const noexcept { return generators_.get_num_cols(); }
    bool is_zero() const noexcept { return generators_.get_num_cols() == 0; }

    static Submodule zero(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix generators(0, presentation.get_num_rows());
        generators.row_degrees = presentation.row_degrees;
        generators.col_degrees.clear();
        generators.data.clear();
        return Submodule(std::move(parent), std::move(generators));
    }

    static Submodule whole(std::shared_ptr<const module_type> parent) {
        if (!parent) throw std::invalid_argument("A submodule requires a parent module");
        const Matrix& presentation = parent->presentation();
        Matrix identity(presentation.get_num_rows(), presentation.get_num_rows(), "Identity");
        identity.row_degrees = presentation.row_degrees;
        identity.col_degrees = presentation.row_degrees;
        return Submodule(std::move(parent), std::move(identity));
    }

    /** Stable-Decomposition's relation-aware generator reduction. */
    void minimize_generators() {
        Matrix ambient = parent_->presentation();
        const index_type relation_count = ambient.get_num_cols();
        ambient.append_matrix(generators_);
        auto old_to_new = ambient.sort_columns_lexicographically_with_output();
        std::vector<index_type> generator_positions;
        generator_positions.reserve(static_cast<std::size_t>(generators_.get_num_cols()));
        for (index_type column = relation_count;
             column < relation_count + generators_.get_num_cols(); ++column) {
            generator_positions.push_back(old_to_new[column]);
        }
        std::sort(generator_positions.begin(), generator_positions.end());
        auto nonzero_columns = ambient.column_reduction_graded();
        nonzero_columns.erase(
            std::remove_if(nonzero_columns.begin(), nonzero_columns.end(),
                           [&](index_type column) {
                               return !std::binary_search(generator_positions.begin(),
                                                          generator_positions.end(), column);
                           }),
            nonzero_columns.end());
        if (nonzero_columns.empty()) {
            *this = zero(parent_);
            return;
        }
        ambient.delete_all_but_columns_alt(nonzero_columns);
        generators_ = std::move(ambient);
        validate();
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

    module_type presented_module(bool minimize = true) const {
        Matrix presentation = parent_->presentation().submodule_generated_by(generators_);
        module_type result(std::move(presentation));
        if (minimize) result.minimize();
        return result;
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
