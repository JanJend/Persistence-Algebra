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
    Matrix P = detail::empty_matrix_like(A, A.get_num_cols() + B.get_num_cols(), A.get_num_rows() + B.get_num_rows());
    auto columns = detail::geometric_col_degrees(A), rows = detail::geometric_row_degrees(A);
    const auto other_columns = detail::geometric_col_degrees(B), other_rows = detail::geometric_row_degrees(B);
    columns.insert(columns.end(), other_columns.begin(), other_columns.end());
    rows.insert(rows.end(), other_rows.begin(), other_rows.end());
    detail::set_geometric_degrees(P, columns, rows);
    for (index j = 0; j < A.get_num_cols(); ++j) P.set_col(j, A.get_col(j));
    for (index j = 0; j < B.get_num_cols(); ++j) {
        auto column = B.get_col(j);
        for (auto& i : column) i += A.get_num_rows();
        P.set_col(A.get_num_cols() + j, column);
    }
    auto sum = std::make_shared<const Module<Matrix>>(std::move(P));
    auto injection = [&](const auto& domain, index offset) {
        Matrix I = detail::empty_matrix_like(sum->presentation(), domain->number_of_generators(), sum->number_of_generators());
        detail::set_geometric_degrees(I, detail::geometric_row_degrees(domain->presentation()),
                                        detail::geometric_row_degrees(sum->presentation()));
        for (index j = 0; j < I.get_num_cols(); ++j) I.set_col(j, {offset + j});
        return Homomorphism<Matrix>(domain, sum, std::move(I));
    };
    auto projection = [&](const auto& target, index offset) {
        Matrix Q = detail::empty_matrix_like(sum->presentation(), sum->number_of_generators(), target->number_of_generators());
        detail::set_geometric_degrees(Q, detail::geometric_row_degrees(sum->presentation()),
                                        detail::geometric_row_degrees(target->presentation()));
        for (index i = 0; i < Q.get_num_rows(); ++i) Q.set_col(offset + i, {i});
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
