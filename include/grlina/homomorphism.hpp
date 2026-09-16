/** @file homomorphism.hpp @brief Trusted module homomorphisms and their lifts. */
#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <grlina/submodule.hpp>
#include <grlina/homomorphisms.hpp>

namespace graded_linalg {

template <typename Matrix>
class Homomorphism {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Homomorphism<Matrix> requires the GradedSparseMatrix CRTP contract");
    using module_type = Module<Matrix>;
    using submodule_type = Submodule<Matrix>;
    using chain_complex_type = ChainComplex<Matrix>;

private:
    std::shared_ptr<const module_type> domain_;
    std::shared_ptr<const module_type> target_;
    // These maps form a lift sequence, not a chain complex on their own.
    std::vector<Matrix> lifts_;

    static const std::vector<typename Matrix::degree_type>& chain_group_degrees(
        const module_type& module, std::size_t degree) {
        const auto& resolution = module.projective_resolution();
        if (resolution.empty()) throw std::invalid_argument("A module map requires projective resolutions");
        if (degree == 0) return resolution[0].row_degrees;
        if (degree <= resolution.size()) return resolution[degree - 1].col_degrees;
        throw std::invalid_argument("Lift exceeds the available projective resolution");
    }

    void validate() {
        if (!domain_ || !target_)
            throw std::invalid_argument("A module map requires domain and target modules");
        if (lifts_.empty()) throw std::invalid_argument("A module map requires a generator lift");
        for (std::size_t i = 0; i < lifts_.size(); ++i) {
            auto& lift = lifts_[i];
            lift.validate();
            if (!lift.compatible_sorting_is_verified()) lift.refresh_compatible_sorted();
            const auto& source = chain_group_degrees(*domain_, i);
            const auto& destination = chain_group_degrees(*target_, i);
            if (lift.col_degrees != source || lift.row_degrees != destination ||
                lift.get_num_cols() != static_cast<typename Matrix::index_type>(source.size()) ||
                lift.get_num_rows() != static_cast<typename Matrix::index_type>(destination.size()))
                throw std::invalid_argument("A lift has incompatible source or target generators");
        }
    }

public:
    /** Caller guarantees that a manually supplied lift induces a homomorphism.
     * Shape/degree validation is cheap and unconditional. Use lift_to_relations
     * or check_lifts explicitly for untrusted input, e.g. after reading a file.
     */
    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, Matrix generator_lift)
        : domain_(std::move(domain)), target_(std::move(target)),
          lifts_{std::move(generator_lift)} { validate(); }

    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, std::vector<Matrix> lifts)
        : domain_(std::move(domain)), target_(std::move(target)),
          lifts_(std::move(lifts)) { validate(); }

    const std::shared_ptr<const module_type>& domain() const noexcept { return domain_; }
    const std::shared_ptr<const module_type>& target() const noexcept { return target_; }
    const std::vector<Matrix>& lifts() const noexcept { return lifts_; }
    const Matrix& generator_lift() const { return lifts_[0]; }

    bool check_lifts() const {
        if (!is_homomorphism(domain_->presentation(), target_->presentation(), lifts_[0])) return false;
        for (std::size_t i = 1; i < lifts_.size(); ++i) {
            auto left = target_->projective_resolution()[i - 1] * lifts_[i];
            auto right = lifts_[i - 1] * domain_->projective_resolution()[i - 1];
            if (left.data != right.data) return false;
        }
        return true;
    }

    /** Extend through the common stored part of the two resolutions. */
    void lift_to_resolution() {
        auto result = lifts_;
        const auto& source = domain_->projective_resolution();
        const auto& destination = target_->projective_resolution();
        const auto length = std::min(source.size(), destination.size());
        while (result.size() <= length) {
            const std::size_t i = result.size() - 1;
            auto next = solve_graded_linear_system(destination[i], result.back() * source[i]);
            if (!next) throw std::invalid_argument("Homomorphism cannot be lifted through the supplied resolutions");
            result.push_back(std::move(*next));
        }
        lifts_ = std::move(result);
    }

    static Homomorphism identity(std::shared_ptr<const module_type> module) {
        if (!module) throw std::invalid_argument("Identity requires a module");
        std::vector<Matrix> lifts;
        for (std::size_t i = 0; i <= module->projective_resolution().size(); ++i) {
            const auto& degrees = chain_group_degrees(*module, i);
            Matrix identity(degrees.size(), degrees.size(), "Identity");
            identity.row_degrees = identity.col_degrees = degrees;
            identity.refresh_compatible_sorted();
            lifts.push_back(std::move(identity));
        }
        return Homomorphism(module, module, std::move(lifts));
    }

    static Homomorphism zero(std::shared_ptr<const module_type> domain,
                             std::shared_ptr<const module_type> target) {
        if (!domain || !target) throw std::invalid_argument("Zero homomorphism requires modules");
        Matrix zero(domain->number_of_generators(), target->number_of_generators());
        zero.data.resize(domain->number_of_generators());
        zero.col_degrees = domain->presentation().row_degrees;
        zero.row_degrees = target->presentation().row_degrees;
        zero.refresh_compatible_sorted();
        return Homomorphism(std::move(domain), std::move(target), std::move(zero));
    }

    /** Translate both modules and every stored lift, preserving resolutions.
     * Matrix::shift uses M(amount)_a = M_(a+amount), hence subtracts degrees.
     */
    Homomorphism shifted(const typename Matrix::degree_type& amount) const {
        auto source = std::make_shared<module_type>(*domain_);
        source->shift(amount);
        auto destination = source;
        if (domain_.get() != target_.get()) {
            destination = std::make_shared<module_type>(*target_);
            destination->shift(amount);
        }
        auto translated = lifts_;
        for (auto& lift : translated) {
            lift.shift(amount);
            if (!lift.is_graded_matrix())
                throw std::invalid_argument("Shift does not preserve the grading of this homomorphism");
        }
        return Homomorphism(source, destination, std::move(translated));
    }

    /** Canonical structure homomorphism M -> M(amount), including identity
     * lifts on all stored projective groups. Reject shifts for which these
     * identities are not degree-admissible. Exactness is inherited, not retested.
     */
    static Homomorphism canonical_shift(std::shared_ptr<const module_type> module,
                                       const typename Matrix::degree_type& amount) {
        if (!module) throw std::invalid_argument("Canonical shift requires a module");
        auto shifted_module = std::make_shared<module_type>(*module);
        shifted_module->shift(amount);
        auto result = identity(module);
        result.target_ = shifted_module;
        for (std::size_t i = 0; i < result.lifts_.size(); ++i) {
            auto& lift = result.lifts_[i];
            lift.row_degrees = chain_group_degrees(*shifted_module, i);
            lift.refresh_compatible_sorted();
            if (!lift.is_graded_matrix())
                throw std::invalid_argument("Canonical shift is not degree-admissible");
        }
        result.validate();
        return result;
    }

    module_type cokernel(bool minimize = true) const {
        return image(false).quotient_module(minimize);
    }

    module_type coimage(bool minimize = true) const {
        return kernel(false).quotient_module(minimize);
    }

    Homomorphism operator+(const Homomorphism& other) const {
        if (domain_.get() != other.domain_.get() || target_.get() != other.target_.get())
            throw std::invalid_argument("Homomorphism addition requires the same domain and target");
        std::vector<Matrix> result;
        for (std::size_t i = 0; i < std::min(lifts_.size(), other.lifts_.size()); ++i) {
            Matrix sum = lifts_[i];
            for (std::size_t j = 0; j < sum.data.size(); ++j)
                Column_traits<vec<typename Matrix::index_type>, typename Matrix::index_type>::add_to(
                    other.lifts_[i].data[j], sum.data[j]);
            sum.invalidate_cached_rows();
            result.push_back(std::move(sum));
        }
        return Homomorphism(domain_, target_, std::move(result));
    }

    submodule_type image(bool minimize = true) const {
        submodule_type result(target_, generator_lift());
        if (minimize) result.minimize_generators();
        return result;
    }

    /** Image of a submodule under this homomorphism. */
    submodule_type image(const submodule_type& submodule, bool minimize = true) const {
        if (submodule.parent().get() != domain_.get())
            throw std::invalid_argument("Image submodule belongs to a different domain module");
        submodule_type result(target_, generator_lift() * submodule.generators());
        if (minimize) result.minimize_generators();
        return result;
    }

    submodule_type preimage(const submodule_type& submodule, bool minimize = true) const {
        if (submodule.parent().get() != target_.get())
            throw std::invalid_argument("Preimage submodule belongs to a different target module");
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Preimage requires a poset-specific graded_kernel");
        } else {
            Matrix generators = generator_lift().inverse_image_copy(
                target_->presentation(), submodule.generators());
            submodule_type result(domain_, std::move(generators));
            if (minimize) result.minimize_generators();
            return result;
        }
    }

    submodule_type kernel(bool minimize = true) const {
        return preimage(submodule_type::zero(target_), minimize);
    }

    Homomorphism then(const Homomorphism& after_this) const {
        if (target_.get() != after_this.domain_.get())
            throw std::invalid_argument("Module-map composition has incompatible middle module");
        std::vector<Matrix> composite;
        for (std::size_t i = 0; i < std::min(lifts_.size(), after_this.lifts_.size()); ++i)
            composite.push_back(after_this.lifts_[i] * lifts_[i]);
        return Homomorphism(domain_, after_this.target_, std::move(composite));
    }
};

template <typename Matrix> using ModuleMorphism = Homomorphism<Matrix>;
template <typename Matrix> using ModuleFunction = Homomorphism<Matrix>;

} // namespace graded_linalg
