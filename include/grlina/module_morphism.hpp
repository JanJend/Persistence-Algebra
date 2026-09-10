/** @file module_morphism.hpp @brief Module maps represented by resolution lifts. */
#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <grlina/submodule.hpp>

namespace graded_linalg {

template <typename Matrix>
class ModuleMorphism {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "ModuleMorphism<Matrix> requires the GradedSparseMatrix CRTP contract");
    using module_type = PersistenceModule<Matrix>;
    using submodule_type = Submodule<Matrix>;
    using chain_complex_type = ChainComplex<Matrix>;

private:
    std::shared_ptr<const module_type> domain_;
    std::shared_ptr<const module_type> target_;
    // Reuse the generic chain storage; lifts are a chain-map sequence rather
    // than adjacent boundary maps, so ChainComplex structural validation is off.
    chain_complex_type lifts_;

    static const std::vector<typename Matrix::degree_type>& chain_group_degrees(
        const module_type& module, std::size_t degree) {
        const auto& resolution = module.projective_resolution();
        if (resolution.empty()) throw std::invalid_argument("A module map requires projective resolutions");
        if (degree == 0) return resolution[0].row_degrees;
        if (degree <= resolution.size()) return resolution[degree - 1].col_degrees;
        throw std::invalid_argument("Lift exceeds the available projective resolution");
    }

    void validate() const {
        if (!domain_ || !target_)
            throw std::invalid_argument("A module map requires domain and target modules");
        if (lifts_.empty()) throw std::invalid_argument("A module map requires a generator lift");
        for (std::size_t i = 0; i < lifts_.size(); ++i) {
            const auto& lift = lifts_[i];
            const auto& source = chain_group_degrees(*domain_, i);
            const auto& destination = chain_group_degrees(*target_, i);
            if (lift.col_degrees != source || lift.row_degrees != destination ||
                lift.get_num_cols() != static_cast<typename Matrix::index_type>(source.size()) ||
                lift.get_num_rows() != static_cast<typename Matrix::index_type>(destination.size()))
                throw std::invalid_argument("A lift has incompatible source or target generators");
            if (!lift.is_graded_matrix()) throw std::invalid_argument("A module-map lift is not graded");
        }
    }

public:
    ModuleMorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, Matrix generator_lift)
        : domain_(std::move(domain)), target_(std::move(target)),
          lifts_(std::vector<Matrix>{std::move(generator_lift)}, false) { validate(); }

    ModuleMorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, std::vector<Matrix> lifts)
        : domain_(std::move(domain)), target_(std::move(target)),
          lifts_(std::move(lifts), false) { validate(); }

    const std::shared_ptr<const module_type>& domain() const noexcept { return domain_; }
    const std::shared_ptr<const module_type>& target() const noexcept { return target_; }
    const chain_complex_type& lifts() const noexcept { return lifts_; }
    const Matrix& generator_lift() const { return lifts_[0]; }

    submodule_type image(bool minimize = true) const {
        submodule_type result(target_, generator_lift());
        if (minimize) result.minimize_generators();
        return result;
    }

    submodule_type preimage(const submodule_type& submodule, bool minimize = true) const {
        if (submodule.parent().get() != target_.get())
            throw std::invalid_argument("Preimage submodule belongs to a different target module");
        Matrix generators = generator_lift().inverse_image_copy(
            target_->presentation(), submodule.generators());
        submodule_type result(domain_, std::move(generators));
        if (minimize) result.minimize_generators();
        return result;
    }

    submodule_type kernel(bool minimize = true) const {
        return preimage(submodule_type::zero(target_), minimize);
    }

    ModuleMorphism then(const ModuleMorphism& after_this) const {
        if (target_.get() != after_this.domain_.get())
            throw std::invalid_argument("Module-map composition has incompatible middle module");
        Matrix composite = after_this.generator_lift() * generator_lift();
        return ModuleMorphism(domain_, after_this.target_, std::move(composite));
    }
};

template <typename Matrix> using ModuleFunction = ModuleMorphism<Matrix>;

} // namespace graded_linalg
