/** @file hom_core_class.hpp @brief Trusted module homomorphisms and their lifts. */
#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <grlina/module/module_base.hpp>
#include <grlina/hom_algorithms.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

template <typename Matrix>
class Homomorphism {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "Homomorphism<Matrix> requires the graded sparse matrix interface");
    using module_type = Module<Matrix>;
    using submodule_type = Submodule<Matrix>;
    using chain_complex_type = ChainComplex<Matrix>;

private:
    friend class Submodule<Matrix>;

    std::shared_ptr<const module_type> domain_;
    std::shared_ptr<const module_type> target_;
    // Components of a possibly partial chain map between projective resolutions.
    std::vector<Matrix> resolution_maps_;
    bool id_matrix_ = false;
    // Only this prefix is known to have identity coefficients. Extending a
    // quotient map to relations need not produce further identity lifts.
    std::size_t identity_lift_count_ = 0;


    static const auto& chain_group_degrees(
        const module_type& module, std::size_t degree) {
        const auto& resolution = module.proj_res_;
        GRLINA_DEBUG_CHECK(
            if (resolution.empty()) throw std::invalid_argument("A module map requires projective resolutions");
            if (degree > resolution.size()) throw std::invalid_argument("Lift exceeds the available projective resolution"));
        if (degree == 0) return resolution[0].row_degrees;
        return resolution[degree - 1].col_degrees;
    }

    static std::vector<matrix_geometry_degree_t<Matrix>> geometric_chain_group_degrees(
        const module_type& module, std::size_t degree) {
        const auto& resolution = module.proj_res_;
        if (degree == 0) return geometric_row_degrees(resolution[0]);
        return geometric_col_degrees(resolution[degree - 1]);
    }

public:
    /** Trusted construction. Identity coefficients are never inferred from
     * supplied matrices: only identity-producing factories set the flag.
     */
    Homomorphism(std::shared_ptr<const module_type> domain,
                 std::shared_ptr<const module_type> target, Matrix generator_lift)
        : Homomorphism(std::move(domain), std::move(target), std::move(generator_lift), false) {}

    Homomorphism(std::shared_ptr<const module_type> domain,
                 std::shared_ptr<const module_type> target, std::vector<Matrix> lifts)
        : Homomorphism(std::move(domain), std::move(target), std::move(lifts), false) {}

private:
    /** Trusted construction: caller guarantees valid endpoints, compatible
     * graded lifts, and the homomorphism equations. No validation or sorting
     * scans are performed.
     */
    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, Matrix generator_lift,
                   bool id_matrix)
        : domain_(std::move(domain)), target_(std::move(target)),
          id_matrix_(id_matrix), identity_lift_count_(id_matrix ? 1 : 0) {
        resolution_maps_.push_back(std::move(generator_lift));
    }

    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, std::vector<Matrix> lifts,
                   bool id_matrix)
        : domain_(std::move(domain)), target_(std::move(target)),
          resolution_maps_(std::move(lifts)), id_matrix_(id_matrix),
          identity_lift_count_(id_matrix ? resolution_maps_.size() : 0) {}

public:
    /** Access endpoints without computing a presentation. A submodule's own
     * inclusion borrows its Module base; it does not own a second module.
     */
    const std::shared_ptr<const module_type>& domain() const noexcept { return domain_; }
    const std::shared_ptr<const module_type>& target() const noexcept { return target_; }
    /** Stored components f_i : F_i(domain) -> F_i(target), starting at i = 0.
     * Entry 0 maps generators, entry 1 maps relations, and later entries map
     * higher syzygies. This only exposes the stored prefix; lift_to_resolution()
     * computes missing components through the common stored resolution length.
     */
    const std::vector<Matrix>& resolution_maps() const noexcept { return resolution_maps_; }
    const Matrix& generator_lift() const { return resolution_maps_[0]; }
    /** True means identity coefficients, not necessarily an identity map:
     * endpoints and generator degrees may differ. This is a construction
     * invariant, never checked by scanning matrix coefficients.
     */
    bool id_matrix() const noexcept { return id_matrix_; }

    bool check_lifts() const {
        if (!is_homomorphism(domain()->presentation(), target_->presentation(), resolution_maps_[0])) return false;
        for (std::size_t i = 1; i < resolution_maps_.size(); ++i) {
            auto left = target_->proj_res_[i - 1] * resolution_maps_[i];
            auto right = resolution_maps_[i - 1] * domain()->proj_res_[i - 1];
            if (left.data != right.data) return false;
        }
        return true;
    }

    /** Extend through the common stored part of the two resolutions. */
    void lift_to_resolution() {
        auto result = resolution_maps_;
        const auto& source = domain()->proj_res_;
        const auto& destination = target_->proj_res_;
        const auto length = std::min(source.size(), destination.size());
        while (result.size() <= length) {
            const std::size_t i = result.size() - 1;
            auto next = solve_graded_linear_system(destination[i], result.back() * source[i]);
            if (!next) throw std::invalid_argument("Homomorphism cannot be lifted through the supplied resolutions");
            result.push_back(std::move(*next));
        }
        resolution_maps_ = std::move(result);
    }

    static Homomorphism identity(std::shared_ptr<const module_type> module) {
        GRLINA_DEBUG_CHECK(
            if (!module) throw std::invalid_argument("Identity requires a module");
            if (module->proj_res_.empty())
                throw std::invalid_argument("A module map requires projective resolutions"));
        std::vector<Matrix> lifts;
        for (std::size_t i = 0; i <= module->proj_res_.size(); ++i) {
            std::size_t differential = 0;
            if (i > 0) differential = i - 1;
            const Matrix& reference = module->proj_res_[differential];
            Matrix identity(geometric_chain_group_degrees(*module, i));
            if constexpr (matrix_grid_backed_v<Matrix>) {
                identity.reindex_grid(reference.grid_);
            }
            identity.compatibly_sorted = reference.compatibly_sorted;
            lifts.push_back(std::move(identity));
        }
        return Homomorphism(module, module, std::move(lifts), true);
    }

    /** Unminimized X -> X/K. The retained ambient basis makes this lift
     * identity by construction, with unchanged ordering of its degrees.
     */
    static Homomorphism quotient_projection(const submodule_type& K) {
        auto quotient = std::make_shared<const module_type>(K.quotient_module(false));
        const auto& parent = K.parent();
        Matrix lift(geometric_row_degrees(parent->presentation()));
        if constexpr (matrix_grid_backed_v<Matrix>) {
            lift.reindex_grid(parent->presentation().grid_);
        }
        lift.compatibly_sorted = parent->presentation().compatibly_sorted;
        return Homomorphism(parent, std::move(quotient), std::move(lift), true);
    }

    static Homomorphism zero(std::shared_ptr<const module_type> domain,
                             std::shared_ptr<const module_type> target) {
        GRLINA_DEBUG_CHECK(if (!domain || !target) throw std::invalid_argument("Zero homomorphism requires modules"));
        Matrix zero = empty_matrix_like(domain->presentation(), domain->num_of_gens(), target->num_of_gens());
        set_geometric_degrees(zero, geometric_row_degrees(domain->presentation()),
                                           geometric_row_degrees(target->presentation()));
        return Homomorphism(std::move(domain), std::move(target), std::move(zero));
    }

    /** Translate both modules and every stored lift, preserving resolutions.
     * Matrix::shift uses M(amount)_a = M_(a+amount), hence subtracts degrees.
     */
    Homomorphism shifted(const matrix_geometry_degree_t<Matrix>& amount) const {
        auto source = std::make_shared<module_type>(*domain());
        source->shift(amount);
        auto destination = source;
        if (domain().get() != target_.get()) {
            destination = std::make_shared<module_type>(*target_);
            destination->shift(amount);
        }
        auto translated = resolution_maps_;
        for (auto& lift : translated) {
            lift.shift(amount);
        }
        auto result = Homomorphism(source, destination, std::move(translated));
        result.id_matrix_ = id_matrix_;
        result.identity_lift_count_ = identity_lift_count_;
        return result;
    }

    /** Canonical structure homomorphism M -> M(amount) for nonnegative amount,
     * including identity lifts on all stored projective groups. Translation
     * preserves degree order; grading follows from the sign of the shift.
     */
    static Homomorphism canonical_shift(std::shared_ptr<const module_type> module,
                                       const matrix_geometry_degree_t<Matrix>& amount) {
        GRLINA_DEBUG_CHECK(if (!module) throw std::invalid_argument("Canonical shift requires a module"));
        auto shifted_module = std::make_shared<module_type>(*module);
        shifted_module->shift(amount);
        auto result = identity(module);
        result.target_ = shifted_module;
        auto lifts = result.resolution_maps();
        for (std::size_t i = 0; i < lifts.size(); ++i) {
            auto& lift = lifts[i];
            set_geometric_degrees(lift, geometric_chain_group_degrees(*module, i),
                                               geometric_chain_group_degrees(*shifted_module, i));
        }
        result.resolution_maps_ = std::move(lifts);
        return result;
    }

    /** Const-reference convenience overload preserving the source identity.
     * As with Module::whole_submodule(), module must already be owned by a
     * shared_ptr; otherwise shared_from_this() throws std::bad_weak_ptr.
     */
    static Homomorphism canonical_shift(const module_type& module,
                                       const matrix_geometry_degree_t<Matrix>& amount) {
        return canonical_shift(module.shared_from_this(), amount);
    }

    module_type cokernel(bool minimize = true) const {
        return image(false).quotient_module(minimize);
    }

    module_type coimage(bool minimize = true) const {
        return kernel(false).quotient_module(minimize);
    }

    Homomorphism operator+(const Homomorphism& other) const {
        GRLINA_DEBUG_CHECK(if (domain_.get() != other.domain_.get() ||
            target_.get() != other.target_.get())
            throw std::invalid_argument("Homomorphism addition requires the same domain and target"));
        std::vector<Matrix> result;
        for (std::size_t i = 0; i < std::min(resolution_maps_.size(), other.resolution_maps_.size()); ++i) {
            Matrix sum = resolution_maps_[i];
            for (std::size_t j = 0; j < sum.data.size(); ++j)
                sum.add_to_col(j, other.resolution_maps_[i].column(j));
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
        GRLINA_DEBUG_CHECK(if (submodule.parent().get() != domain().get())
            throw std::invalid_argument("Image submodule belongs to a different domain module"));
        Matrix generators = id_matrix_ ? submodule.generator_map().generator_lift()
                                       : generator_lift() * submodule.generator_map().generator_lift();
        if (id_matrix_) {
            // Factory-built identity lifts only preserve or translate degrees;
            // the copied ordering certificate remains valid.
            set_geometric_degrees(generators, geometric_col_degrees(generators),
                                                    geometric_row_degrees(generator_lift()));
        }
        submodule_type result(target_, std::move(generators));
        if (minimize) result.minimize_generators();
        return result;
    }

    submodule_type preimage(const submodule_type& submodule, bool minimize = true) const {
        GRLINA_DEBUG_CHECK(if (submodule.parent().get() != target_.get())
            throw std::invalid_argument("Preimage submodule belongs to a different target module"));
        if constexpr (!has_matrix_graded_kernel<Matrix>::value) {
            // Jan: implement the poset-specific graded kernel.
            throw std::logic_error("Preimage requires a poset-specific graded_kernel");
        } else {
            Matrix generators = generator_lift().inverse_image_copy(
                target_->presentation(), submodule.generator_map().generator_lift());
            submodule_type result(domain(), std::move(generators));
            if (minimize) result.minimize_generators();
            return result;
        }
    }

    submodule_type kernel(bool minimize = true) const {
        return preimage(submodule_type::zero(target_), minimize);
    }

    /** Forward composition: f.compose(g) is g o f (apply f, then g). */
    Homomorphism compose(const Homomorphism& after_this) const {
        GRLINA_DEBUG_CHECK(if (target_.get() != after_this.domain().get())
            throw std::invalid_argument("Module-map composition has incompatible middle module"));
        std::vector<Matrix> composite;
        for (std::size_t i = 0; i < std::min(resolution_maps_.size(), after_this.resolution_maps_.size()); ++i) {
            if (i < after_this.identity_lift_count_) {
                Matrix lift = resolution_maps_[i];
                set_geometric_degrees(lift, geometric_col_degrees(lift),
                                                   geometric_row_degrees(after_this.resolution_maps_[i]));
                composite.push_back(std::move(lift));
            } else if (i < identity_lift_count_) {
                Matrix lift = after_this.resolution_maps_[i];
                set_geometric_degrees(lift, geometric_col_degrees(resolution_maps_[i]),
                                                   geometric_row_degrees(lift));
                composite.push_back(std::move(lift));
            } else {
                composite.push_back(after_this.resolution_maps_[i] * resolution_maps_[i]);
            }
        }
        auto result = Homomorphism(domain_, after_this.target_, std::move(composite));
        result.identity_lift_count_ = std::min(identity_lift_count_, after_this.identity_lift_count_);
        result.id_matrix_ = result.identity_lift_count_ != 0;
        return result;
    }

    /** Compatibility spelling; prefer compose(). */
    Homomorphism then(const Homomorphism& after_this) const {
        return compose(after_this);
    }
};

template <typename Matrix> using ModuleMorphism = Homomorphism<Matrix>;
template <typename Matrix> using ModuleFunction = Homomorphism<Matrix>;

} // namespace graded_linalg
