/** Finite categorical constructions, with canonical homomorphisms over F2.
 * Presentations are deliberately not minimized: their bases identify the
 * canonical maps. Call Module::minimize() on a COPY if only the object is needed.
 */
#pragma once
#include <grlina/submodule.hpp>

namespace graded_linalg {

template <typename Matrix>
struct Biproduct {
    std::shared_ptr<const Module<Matrix>> module;
    Homomorphism<Matrix> inclusion_left, inclusion_right;
    Homomorphism<Matrix> projection_left, projection_right;
};

/** Finite direct sum, simultaneously the product and coproduct. */
template <typename Matrix>
Biproduct<Matrix> direct_sum(std::shared_ptr<const Module<Matrix>> left,
                            std::shared_ptr<const Module<Matrix>> right) {
    if (!left || !right) throw std::invalid_argument("Direct sum requires two modules");
    using index = typename Matrix::index_type;
    const auto& A = left->presentation();
    const auto& B = right->presentation();
    Matrix P(A.get_num_cols() + B.get_num_cols(), A.get_num_rows() + B.get_num_rows());
    P.data.resize(P.get_num_cols());
    P.col_degrees = A.col_degrees;
    P.col_degrees.insert(P.col_degrees.end(), B.col_degrees.begin(), B.col_degrees.end());
    P.row_degrees = A.row_degrees;
    P.row_degrees.insert(P.row_degrees.end(), B.row_degrees.begin(), B.row_degrees.end());
    for (index j = 0; j < A.get_num_cols(); ++j) P.data[j] = A.data[j];
    for (index j = 0; j < B.get_num_cols(); ++j)
        for (index i : B.data[j]) P.data[A.get_num_cols() + j].push_back(A.get_num_rows() + i);
    auto sum = std::make_shared<const Module<Matrix>>(std::move(P));
    auto injection = [&](const auto& domain, index offset) {
        Matrix I(domain->number_of_generators(), sum->number_of_generators());
        I.data.resize(I.get_num_cols());
        I.col_degrees = domain->presentation().row_degrees;
        I.row_degrees = sum->presentation().row_degrees;
        for (index j = 0; j < I.get_num_cols(); ++j) I.data[j] = {offset + j};
        return Homomorphism<Matrix>(domain, sum, std::move(I));
    };
    auto projection = [&](const auto& target, index offset) {
        Matrix Q(sum->number_of_generators(), target->number_of_generators());
        Q.data.resize(Q.get_num_cols());
        Q.col_degrees = sum->presentation().row_degrees;
        Q.row_degrees = target->presentation().row_degrees;
        for (index i = 0; i < Q.get_num_rows(); ++i) Q.data[offset + i] = {i};
        return Homomorphism<Matrix>(sum, target, std::move(Q));
    };
    return {sum, injection(left, 0), injection(right, A.get_num_rows()),
            projection(left, 0), projection(right, A.get_num_rows())};
}

template <typename Matrix>
Biproduct<Matrix> product(std::shared_ptr<const Module<Matrix>> a,
                          std::shared_ptr<const Module<Matrix>> b) { return direct_sum<Matrix>(a, b); }
template <typename Matrix>
Biproduct<Matrix> coproduct(std::shared_ptr<const Module<Matrix>> a,
                            std::shared_ptr<const Module<Matrix>> b) { return direct_sum<Matrix>(a, b); }

template <typename Matrix>
struct Subobject {
    std::shared_ptr<const Module<Matrix>> module;
    Homomorphism<Matrix> inclusion;
};

template <typename Matrix>
Subobject<Matrix> as_subobject(Submodule<Matrix> submodule) {
    // This adapter explicitly requests a presented, owning subobject. The
    // inclusion's source is that same Submodule, not a separate Module cache.
    auto object = std::make_shared<Submodule<Matrix>>(std::move(submodule));
    object->compute_presentation();
    Homomorphism<Matrix> inclusion(object, object->parent(), object->generator_map().generator_lift());
    return {std::move(object), std::move(inclusion)};
}

template <typename Matrix>
Subobject<Matrix> kernel(const Homomorphism<Matrix>& f) { return as_subobject(f.kernel(false)); }
template <typename Matrix>
Subobject<Matrix> image(const Homomorphism<Matrix>& f) { return as_subobject(f.image(false)); }

template <typename Matrix>
struct QuotientObject {
    std::shared_ptr<const Module<Matrix>> module;
    Homomorphism<Matrix> projection;
};

template <typename Matrix>
QuotientObject<Matrix> as_quotient(const Submodule<Matrix>& submodule) {
    auto projection = Homomorphism<Matrix>::quotient_projection(submodule);
    return {projection.target(), std::move(projection)};
}

template <typename Matrix>
QuotientObject<Matrix> cokernel(const Homomorphism<Matrix>& f) { return as_quotient(f.image(false)); }
template <typename Matrix>
QuotientObject<Matrix> coimage(const Homomorphism<Matrix>& f) { return as_quotient(f.kernel(false)); }

template <typename Matrix>
Subobject<Matrix> equalizer(const Homomorphism<Matrix>& f, const Homomorphism<Matrix>& g) {
    return kernel(f + g); // subtraction equals addition over F2
}
template <typename Matrix>
QuotientObject<Matrix> coequalizer(const Homomorphism<Matrix>& f, const Homomorphism<Matrix>& g) {
    return cokernel(f + g);
}

template <typename Matrix>
struct Pullback {
    std::shared_ptr<const Module<Matrix>> module;
    Homomorphism<Matrix> to_left, to_right;
};

/** A x_C B = ker([f,-g] : A + B -> C). */
template <typename Matrix>
Pullback<Matrix> pullback(const Homomorphism<Matrix>& f, const Homomorphism<Matrix>& g) {
    if (f.target().get() != g.target().get())
        throw std::invalid_argument("Pullback requires a common target");
    auto sum = direct_sum<Matrix>(f.domain(), g.domain());
    auto difference = sum.projection_left.compose(f) + sum.projection_right.compose(g);
    auto K = kernel(difference);
    return {K.module, K.inclusion.compose(sum.projection_left), K.inclusion.compose(sum.projection_right)};
}

template <typename Matrix>
struct Pushout {
    std::shared_ptr<const Module<Matrix>> module;
    Homomorphism<Matrix> from_left, from_right;
};

/** A +_C B = coker((f,-g) : C -> A + B). */
template <typename Matrix>
Pushout<Matrix> pushout(const Homomorphism<Matrix>& f, const Homomorphism<Matrix>& g) {
    if (f.domain().get() != g.domain().get())
        throw std::invalid_argument("Pushout requires a common domain");
    auto sum = direct_sum<Matrix>(f.target(), g.target());
    auto difference = f.compose(sum.inclusion_left) + g.compose(sum.inclusion_right);
    auto Q = cokernel(difference);
    return {Q.module, sum.inclusion_left.compose(Q.projection), sum.inclusion_right.compose(Q.projection)};
}

} // namespace graded_linalg
