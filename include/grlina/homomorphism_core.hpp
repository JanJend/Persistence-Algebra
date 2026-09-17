/** @file homomorphism_core.hpp @brief Trusted module homomorphisms and their lifts. */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <grlina/module.hpp>
#include <grlina/homomorphisms.hpp>
#include <grlina/matrix_family.hpp>

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
    friend class Submodule<Matrix>;

    // Copies of a map share this endpoint, including before lazy materialization.
    // The factory owns only matrix storage and the parent, never the Submodule.
    struct Domain {
        mutable std::shared_ptr<const module_type> module;
        mutable std::function<std::shared_ptr<const module_type>()> factory;
        mutable std::once_flag initialized;
        explicit Domain(std::shared_ptr<const module_type> value) : module(std::move(value)) {}
        const std::shared_ptr<const module_type>& get() const {
            std::call_once(initialized, [&] {
                if (factory) {
                    module = factory();
                    factory = {};
                }
            });
            return module;
        }
    };
    std::shared_ptr<Domain> domain_;
    std::shared_ptr<const module_type> target_;
    // These maps form a lift sequence, not a chain complex on their own.
    std::shared_ptr<const std::vector<Matrix>> lifts_;
    bool id_matrix_ = false;
    // Only this prefix is known to have identity coefficients. Extending a
    // quotient map to relations need not produce further identity lifts.
    std::size_t identity_lift_count_ = 0;


    Homomorphism with_same_domain(std::shared_ptr<const module_type> target,
                                 std::vector<Matrix> lifts) const {
        Homomorphism result(std::shared_ptr<const module_type>{}, std::move(target), std::move(lifts));
        result.domain_ = domain_;
        return result;
    }

    static Homomorphism from_image_generators(std::shared_ptr<const module_type> parent,
                                             Matrix generators, bool id_matrix = false) {
        Homomorphism result(std::shared_ptr<const module_type>{}, parent, std::move(generators), id_matrix);
        auto storage = result.lifts_;
        result.domain_->factory = [parent = std::move(parent), storage] {
            const Matrix& matrix = (*storage)[0];
            if (matrix.get_num_cols() == 0)
                return std::make_shared<const module_type>(Matrix(0, 0, {}, {}, {}));
            if constexpr (has_matrix_graded_kernel<Matrix>::value) {
                return std::make_shared<const module_type>(parent->presentation().submodule_generated_by(matrix));
            } else {
                throw std::logic_error("Submodule presentation requires a poset-specific graded_kernel");
            }
        };
        return result;
    }

    static const std::vector<typename Matrix::degree_type>& chain_group_degrees(
        const module_type& module, std::size_t degree) {
        const auto& resolution = module.projective_resolution();
        if (resolution.empty()) throw std::invalid_argument("A module map requires projective resolutions");
        if (degree == 0) return resolution[0].row_degrees;
        if (degree <= resolution.size()) return resolution[degree - 1].col_degrees;
        throw std::invalid_argument("Lift exceeds the available projective resolution");
    }

public:
    /** Explicit structural check for untrusted input. Does not check the
     * homomorphism equations; use check_lifts() for those.
     */
    void validate() const {
        if (!domain() || !target_)
            throw std::invalid_argument("A module map requires domain and target modules");
        if (lifts_->empty()) throw std::invalid_argument("A module map requires a generator lift");
        for (std::size_t i = 0; i < lifts_->size(); ++i) {
            const auto& lift = (*lifts_)[i];
            lift.validate();
            const auto& source = chain_group_degrees(*domain(), i);
            const auto& destination = chain_group_degrees(*target_, i);
            if (lift.col_degrees != source || lift.row_degrees != destination ||
                lift.get_num_cols() != static_cast<typename Matrix::index_type>(source.size()) ||
                lift.get_num_rows() != static_cast<typename Matrix::index_type>(destination.size()))
                throw std::invalid_argument("A lift has incompatible source or target generators");
        }
    }

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
     * scans are performed. For untrusted input, call validate() explicitly,
     * then check_lifts() to check the equations.
     */
    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, Matrix generator_lift,
                   bool id_matrix)
        : domain_(std::make_shared<Domain>(std::move(domain))), target_(std::move(target)),
          id_matrix_(id_matrix), identity_lift_count_(id_matrix ? 1 : 0) {
        std::vector<Matrix> lifts;
        lifts.push_back(std::move(generator_lift));
        lifts_ = std::make_shared<const std::vector<Matrix>>(std::move(lifts));
    }

    Homomorphism(std::shared_ptr<const module_type> domain,
                   std::shared_ptr<const module_type> target, std::vector<Matrix> lifts,
                   bool id_matrix)
        : domain_(std::make_shared<Domain>(std::move(domain))), target_(std::move(target)),
          lifts_(std::make_shared<const std::vector<Matrix>>(std::move(lifts))), id_matrix_(id_matrix),
          identity_lift_count_(id_matrix ? lifts_->size() : 0) {}

public:
    /** Accessing a submodule inclusion's source materializes its presentation
     * on first use. Copies share that source; coefficient access stays lazy.
     */
    const std::shared_ptr<const module_type>& domain() const { return domain_->get(); }
    const std::shared_ptr<const module_type>& target() const noexcept { return target_; }
    const std::vector<Matrix>& lifts() const noexcept { return *lifts_; }
    const Matrix& generator_lift() const { return (*lifts_)[0]; }
    /** True means identity coefficients, not necessarily an identity map:
     * endpoints and generator degrees may differ. This is a construction
     * invariant, never checked by scanning matrix coefficients.
     */
    bool id_matrix() const noexcept { return id_matrix_; }

    bool check_lifts() const {
        if (!is_homomorphism(domain()->presentation(), target_->presentation(), (*lifts_)[0])) return false;
        for (std::size_t i = 1; i < lifts_->size(); ++i) {
            auto left = target_->projective_resolution()[i - 1] * (*lifts_)[i];
            auto right = (*lifts_)[i - 1] * domain()->projective_resolution()[i - 1];
            if (left.data != right.data) return false;
        }
        return true;
    }

    /** Extend through the common stored part of the two resolutions. */
    void lift_to_resolution() {
        auto result = *lifts_;
        const auto& source = domain()->projective_resolution();
        const auto& destination = target_->projective_resolution();
        const auto length = std::min(source.size(), destination.size());
        while (result.size() <= length) {
            const std::size_t i = result.size() - 1;
            auto next = solve_graded_linear_system(destination[i], result.back() * source[i]);
            if (!next) throw std::invalid_argument("Homomorphism cannot be lifted through the supplied resolutions");
            result.push_back(std::move(*next));
        }
        lifts_ = std::make_shared<const std::vector<Matrix>>(std::move(result));
    }

    static Homomorphism identity(std::shared_ptr<const module_type> module) {
        if (!module) throw std::invalid_argument("Identity requires a module");
        std::vector<Matrix> lifts;
        for (std::size_t i = 0; i <= module->projective_resolution().size(); ++i) {
            const auto& degrees = chain_group_degrees(*module, i);
            Matrix identity(degrees.size(), degrees.size(), "Identity");
            identity.row_degrees = identity.col_degrees = degrees;
            identity.inherit_compatible_sorting(module->projective_resolution()[i == 0 ? 0 : i - 1]);
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
        Matrix lift(parent->number_of_generators(), parent->number_of_generators(), "Identity");
        lift.col_degrees = lift.row_degrees = parent->presentation().row_degrees;
        lift.inherit_compatible_sorting(parent->presentation());
        return Homomorphism(parent, std::move(quotient), std::move(lift), true);
    }

    static Homomorphism zero(std::shared_ptr<const module_type> domain,
                             std::shared_ptr<const module_type> target) {
        if (!domain || !target) throw std::invalid_argument("Zero homomorphism requires modules");
        Matrix zero(domain->number_of_generators(), target->number_of_generators());
        zero.data.resize(domain->number_of_generators());
        zero.col_degrees = domain->presentation().row_degrees;
        zero.row_degrees = target->presentation().row_degrees;
        return Homomorphism(std::move(domain), std::move(target), std::move(zero));
    }

    /** Translate both modules and every stored lift, preserving resolutions.
     * Matrix::shift uses M(amount)_a = M_(a+amount), hence subtracts degrees.
     */
    Homomorphism shifted(const typename Matrix::degree_type& amount) const {
        auto source = std::make_shared<module_type>(*domain());
        source->shift(amount);
        auto destination = source;
        if (domain().get() != target_.get()) {
            destination = std::make_shared<module_type>(*target_);
            destination->shift(amount);
        }
        auto translated = *lifts_;
        for (auto& lift : translated) {
            lift.shift(amount);
            GRLINA_DEBUG_CHECK(if (!lift.is_graded_matrix())
                throw std::invalid_argument("Shift does not preserve the grading of this homomorphism"));
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
                                       const typename Matrix::degree_type& amount) {
        if (!module) throw std::invalid_argument("Canonical shift requires a module");
        using Degree = typename Matrix::degree_type;
        if (!Degree_traits<Degree>::smaller_equal(Degree{}, amount))
            throw std::invalid_argument("Canonical shift requires a nonnegative amount");
        auto shifted_module = std::make_shared<module_type>(*module);
        shifted_module->shift(amount);
        auto result = identity(module);
        result.target_ = shifted_module;
        auto lifts = result.lifts();
        for (std::size_t i = 0; i < lifts.size(); ++i) {
            auto& lift = lifts[i];
            lift.row_degrees = chain_group_degrees(*shifted_module, i);
        }
        result.lifts_ = std::make_shared<const std::vector<Matrix>>(std::move(lifts));
        return result;
    }

    /** Const-reference convenience overload preserving the source identity.
     * As with Module::whole_submodule(), module must already be owned by a
     * shared_ptr; otherwise shared_from_this() throws std::bad_weak_ptr.
     */
    static Homomorphism canonical_shift(const module_type& module,
                                       const typename Matrix::degree_type& amount) {
        return canonical_shift(module.shared_from_this(), amount);
    }

    module_type cokernel(bool minimize = true) const {
        return image(false).quotient_module(minimize);
    }

    module_type coimage(bool minimize = true) const {
        return kernel(false).quotient_module(minimize);
    }

    Homomorphism operator+(const Homomorphism& other) const {
        if ((domain_ != other.domain_ && domain().get() != other.domain().get()) ||
            target_.get() != other.target_.get())
            throw std::invalid_argument("Homomorphism addition requires the same domain and target");
        std::vector<Matrix> result;
        for (std::size_t i = 0; i < std::min(lifts_->size(), other.lifts_->size()); ++i) {
            Matrix sum = (*lifts_)[i];
            for (std::size_t j = 0; j < sum.data.size(); ++j)
                Column_traits<vec<typename Matrix::index_type>, typename Matrix::index_type>::add_to(
                    (*other.lifts_)[i].data[j], sum.data[j]);
            sum.invalidate_cached_rows();
            result.push_back(std::move(sum));
        }
        return with_same_domain(target_, std::move(result));
    }

    submodule_type image(bool minimize = true) const {
        submodule_type result(target_, generator_lift());
        if (minimize) result.minimize_generators();
        return result;
    }

    /** Image of a submodule under this homomorphism. */
    submodule_type image(const submodule_type& submodule, bool minimize = true) const {
        if (submodule.parent().get() != domain().get())
            throw std::invalid_argument("Image submodule belongs to a different domain module");
        Matrix generators = id_matrix_ ? submodule.generator_map().generator_lift()
                                       : generator_lift() * submodule.generator_map().generator_lift();
        if (id_matrix_) {
            // Factory-built identity lifts only preserve or translate degrees;
            // the copied ordering certificate remains valid.
            generators.row_degrees = generator_lift().row_degrees;
        }
        submodule_type result(target_, std::move(generators));
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
        if (target_.get() != after_this.domain().get())
            throw std::invalid_argument("Module-map composition has incompatible middle module");
        std::vector<Matrix> composite;
        for (std::size_t i = 0; i < std::min(lifts_->size(), after_this.lifts_->size()); ++i) {
            if (i < after_this.identity_lift_count_) {
                Matrix lift = (*lifts_)[i];
                lift.row_degrees = (*after_this.lifts_)[i].row_degrees;
                composite.push_back(std::move(lift));
            } else if (i < identity_lift_count_) {
                Matrix lift = (*after_this.lifts_)[i];
                lift.col_degrees = (*lifts_)[i].col_degrees;
                composite.push_back(std::move(lift));
            } else {
                composite.push_back((*after_this.lifts_)[i] * (*lifts_)[i]);
            }
        }
        auto result = with_same_domain(after_this.target_, std::move(composite));
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
