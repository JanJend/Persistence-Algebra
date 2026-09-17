#include <grlina/modules.hpp>
#include <cassert>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;
using Sub = Submodule<Mat>;
static_assert(std::is_base_of_v<Mod, Sub>);
static_assert(std::has_virtual_destructor_v<Mod>);

template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

void test_in_place_module_storage() {
    auto parent = std::make_shared<Mod>(Mat(1, 2, {{0}}, {{2,2}}, {{0,0}, {0,0}}));
    parent->compute_projective_resolution();
    const auto parent_presentation = parent->presentation();
    const auto parent_resolution_size = parent->projective_resolution().size();
    Mat G(2, 2, {{0}, {0}}, {{1,1}, {1,1}}, parent->presentation().row_degrees);
    auto S = std::make_shared<Sub>(parent, G);
    assert(!S->has_presentation() && S->number_of_generators() == 2);
    Mod& as_module = *S;
    // Virtual presentation hook works through a Module reference.
    as_module.compute_presentation();
    assert(as_module.has_presentation() && S->has_projective_resolution());
    assert(S->presentation().row_degrees == G.col_degrees);
    assert(S->number_of_generators() == as_module.number_of_generators());
    assert(S->dimension_at({0,0}) == 0 && S->dimension_at({1,1}) == 1);
    assert(S->dimension_at({2,2}) == 0);
    as_module.compute_projective_resolution();
    assert(S->has_complete_projective_resolution());
    assert(S->projective_resolution().squares_to_zero());
    assert(S->generator_map().generator_lift().data == G.data && S->parent().get() == parent.get());
    assert(S->hilbert_function_on_induced_grid().maximum == 1);

    std::shared_ptr<const Mod> module_pointer = S;
    assert(Homomorphism<Mat>::identity(module_pointer).check_lifts());
    // A computed submodule can itself be the parent of another submodule.
    Sub nested(module_pointer, Mat(1, 2, {{0}}, {{1,1}}, S->presentation().row_degrees));
    nested.compute_projective_resolution(); // automatically constructs its presentation
    assert(nested.has_complete_projective_resolution());
    assert(nested.dimension_at({1,1}) == 1 && nested.dimension_at({2,2}) == 0);

    std::stringstream encoded;
    S->to_stream(encoded);
    Mod decoded(encoded);
    assert(decoded.dimension_at({1,1}) == 1 && decoded.dimension_at({2,2}) == 0);
    Sub copied = *S;
    Sub moved = std::move(copied);
    assert(moved.has_complete_projective_resolution() && moved.parent() == S->parent());
    assert(moved.dimension_at({1,1}) == 1);
    assert(parent->presentation().data == parent_presentation.data);
    assert(parent->projective_resolution().size() == parent_resolution_size);
    assert(parent->has_complete_projective_resolution());
}

void test_canonical_submodule_members() {
    auto parent = std::make_shared<Mod>(Mat(1, 2, {{0}}, {{2,2}}, {{0,0}, {0,0}}));
    std::weak_ptr<const Mod> lifetime = parent;
    {
        const Mod& module = *parent;
        auto whole = module.whole_submodule();
        auto zero = module.zero_submodule();
        assert(whole.parent().get() == parent.get());
        assert(zero.parent().get() == parent.get());
        assert(whole.equals(Sub::whole(parent)));
        assert(zero.equals(Sub::zero(parent)));
        assert(whole.contains(zero) && !zero.contains(whole));
        parent.reset();
        assert(!lifetime.expired());
        assert(whole.presented_module().dimension_at({0,0}) == 2);
        assert(whole.presented_module().dimension_at({2,2}) == 1);
        assert(zero.is_zero());
    }
    assert(lifetime.expired());

    Mod stack(Mat(0, 0, {}, {}, {}));
    for (bool whole : {false, true}) {
        bool threw = false;
        try {
            (void)(whole ? stack.whole_submodule() : stack.zero_submodule());
        } catch (const std::bad_weak_ptr&) { threw = true; }
        assert(threw);
    }

    auto empty = std::make_shared<const Mod>(Mat(0, 0, {}, {}, {}));
    assert(empty->whole_submodule().equals(empty->zero_submodule()));
    auto missing = std::make_shared<Mod>();
    rejects([&] { missing->whole_submodule(); });
    rejects([&] { missing->zero_submodule(); });

    // Derived modules retain their own shared ownership after presentation updates.
    auto nested_parent = std::make_shared<Sub>(empty->zero_submodule());
    nested_parent->compute_presentation();
    assert(nested_parent->whole_submodule().parent().get() == nested_parent.get());
}

void test_shared_generator_basis() {
    auto parent = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0,0}}));
    Mat G(2, 1, {{0}, {0}}, {{1,1}, {1,1}}, {{0,0}});
    Sub S(parent, G);
    auto standalone = S.presented_module();
    assert(S.has_presentation() && S.number_of_generators() == 1);
    assert(S.number_of_embedding_generators() == 1);
    assert(S.generator_map().domain().get() == static_cast<const Mod*>(&S));
    assert(S.generator_map().check_lifts());
    assert(standalone.presentation().data == S.presentation().data);
    auto object = as_subobject(S);
    assert(object.module->number_of_generators() == 1 && object.inclusion.check_lifts());
    assert(object.inclusion.domain() == object.module);
    S.compute_presentation();
    assert(S.number_of_generators() == 1);
    S.compute_projective_resolution();
    Mod& base = S;
    base.minimize();
    assert(S.number_of_generators() == S.number_of_embedding_generators());
    assert(S.generator_map().check_lifts() && S.has_complete_projective_resolution());
    assert(S.dimension_at({1,1}) == 1);

    const Sub fresh(parent, G);
    auto value = fresh.presented_module(false);
    assert(value.number_of_generators() == 2 && !fresh.has_presentation());
}

void test_invalidation_and_automatic_resolution() {
    auto parent = std::make_shared<Mod>(Mat(1, 2, {{0}}, {{1,1}}, {{0,0}, {0,0}}));
    Mat G(2, 2, {{0}, {1}}, {{1,1}, {0,0}}, parent->presentation().row_degrees);
    Sub S(parent, G);
    Mod& base = S;
    base.compute_projective_resolution(); // virtual hook obtains a presentation first
    assert(S.has_complete_projective_resolution());
    S.set_injective_resolution(S.projective_resolution()); // storage-only fixture
    S.reduce_generators_lazy();
    assert(!S.has_projective_resolution() && S.has_injective_resolution());
    assert(S.generator_map().generator_lift().data == array<int>({{1}}));
    S.compute_projective_resolution();
    assert(S.has_complete_projective_resolution() && S.dimension_at({2,2}) == 1);
    S.minimize_generators(false);
    assert(!base.has_presentation());
    base.compute_presentation();
    assert(base.number_of_generators() == 1 && base.dimension_at({0,0}) == 1);
    // Existing Module behavior stays valid: its presentation hook is a no-op
    // unless explicit minimization is requested.
    Mod ordinary(Mat(1, 1, {{0}}, {{0,0}}, {{0,0}}));
    ordinary.compute_presentation(true);
    assert(ordinary.number_of_generators() == 0);
    Mod missing;
    rejects([&] { missing.compute_presentation(); });
    rejects([&] { missing.compute_projective_resolution(); });
}

void test_zero_without_kernel_and_polymorphic_destruction() {
    using Higher = R4GradedSparseMatrix<int>;
    auto parent = std::make_shared<Module<Higher>>(Higher(0, 1, {}, {}, {r4degree(0,0,0,0)}));
    auto zero = Submodule<Higher>::zero(parent);
    zero.compute_projective_resolution();
    assert(zero.has_presentation() && zero.has_complete_projective_resolution());
    assert(zero.number_of_generators() == 0 && zero.dimension_at(r4degree(0,0,0,0)) == 0);
    auto whole = Submodule<Higher>::whole(parent);
    whole.compute_presentation(); // The whole submodule reuses the known parent.
    assert(whole.generator_map().domain().get() == static_cast<const Module<Higher>*>(&whole));
    Submodule<Higher> nontrivial(parent, Higher(1, 1, {{0}},
        {r4degree(1,1,1,1)}, parent->presentation().row_degrees));
    rejects([&] { nontrivial.compute_presentation(); }); // nontrivial kernel still unsupported
    assert(!nontrivial.has_presentation());

    struct DestructionProbe : Sub {
        bool& destroyed;
        DestructionProbe(std::shared_ptr<const Mod> p, Mat g, bool& flag)
            : Sub(std::move(p), std::move(g)), destroyed(flag) {}
        ~DestructionProbe() override { destroyed = true; }
    };
    auto free = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0,0}}));
    bool destroyed = false;
    {
        std::unique_ptr<Mod> owned = std::make_unique<DestructionProbe>(
            free, Mat(1, 1, {{0}}, {{0,0}}, {{0,0}}), destroyed);
        owned->compute_projective_resolution();
        assert(owned->dimension_at({0,0}) == 1);
    }
    assert(destroyed);
}

void test_generator_map_homomorphism() {
    using Hom = Homomorphism<Mat>;
    static_assert(std::is_same_v<decltype(std::declval<const Sub&>().generator_map()), const Hom&>);
    auto parent = std::make_shared<const Mod>(Mat(1, 2, {{0}}, {{2,2}}, {{0,0}, {0,0}}));
    Mat matrix(3, 2, {{0}, {0}, {1}}, {{1,1}, {1,1}, {0,0}}, parent->presentation().row_degrees);
    Sub S(parent, matrix);
    const auto& inclusion = S.generator_map();
    assert(inclusion.domain().get() == static_cast<const Mod*>(&S));
    assert(inclusion.target() == parent);
    assert(!inclusion.domain()->has_presentation());
    inclusion.validate();
    rejects([&] { (void)inclusion.domain()->presentation(); });
    assert(!S.has_presentation());
    S.compute_presentation();
    inclusion.validate();
    assert(inclusion.check_lifts() && inclusion.kernel(false).is_zero());
    assert(inclusion.domain()->number_of_generators() == 3);
    assert(inclusion.domain()->dimension_at({1,1}) == 2);
    assert(inclusion.domain()->dimension_at({2,2}) == 1);
    assert(inclusion.image(false).equals(S));

    auto eta = Hom::canonical_shift(parent, {1,1});
    auto composite = inclusion.compose(eta);
    composite.validate();
    assert(composite.domain().get() == &S && composite.target() == eta.target());
    assert(composite.check_lifts());
    assert(composite.image(false).equals(eta.image(S, false)));
    auto copied_map = inclusion;
    copied_map.lift_to_resolution();
    assert(copied_map.lifts().size() > inclusion.lifts().size());
    assert(copied_map.check_lifts() && inclusion.lifts().size() == 1);

    S.compute_presentation(true);
    assert(S.number_of_generators() == 2 && S.number_of_embedding_generators() == 2);
    assert(S.generator_map().domain().get() == &S);
    assert(S.generator_map().check_lifts());
    // Earlier copies of a map must not be used after the source basis changes.
    Sub changed = S;
    changed.shift_generators({1,1});
    assert(!changed.has_presentation());
    assert(changed.generator_map().domain().get() == &changed);
    changed.compute_presentation();
    assert(changed.generator_map().check_lifts());
    S.minimize_generators();
    assert(!S.has_presentation() && S.generator_map().domain().get() == &S);
    S.compute_presentation();
    assert(S.generator_map().check_lifts());
    S.lazy_minimize_parent();
    assert(S.generator_map().target() == S.parent() && S.parent() != parent);
    S.compute_presentation();
    assert(S.generator_map().check_lifts());
    auto quotient = S.submodule_quotient(Sub::zero(S.parent()));
    assert(quotient.generator_map().domain().get() == &quotient);
    assert(!quotient.has_presentation());
    quotient.compute_presentation();
    assert(quotient.generator_map().check_lifts());
    auto whole = Sub::whole(parent);
    assert(whole.generator_map().id_matrix());
    assert(whole.generator_map().domain().get() == &whole && !whole.has_presentation());
    whole.compute_presentation();
    assert(whole.generator_map().check_lifts());
    auto zero = Sub::zero(parent);
    assert(zero.generator_map().domain().get() == &zero && !zero.has_presentation());
    zero.compute_presentation();
    zero.generator_map().validate();
    assert(zero.generator_map().check_lifts() && zero.number_of_generators() == 0);
}

void test_self_domain_copy_move_and_explicit_computation() {
    auto parent = std::make_shared<const Mod>(Mat(0, 1, {}, {}, {{0,0}}));
    Sub original(parent, Mat(2, 1, {{0}, {0}}, {{1,1}, {1,1}}, {{0,0}}));
    auto check_self = [](const Sub& S) {
        assert(S.generator_map().domain().get() == static_cast<const Mod*>(&S));
    };
    check_self(original);
    Sub copied = original;
    check_self(copied);
    assert(!copied.has_presentation());
    Sub moved = std::move(copied);
    check_self(moved);
    copied = original;
    check_self(copied);
    copied = std::move(moved);
    check_self(copied);
    copied.compute_presentation();
    assert(!original.has_presentation());
    Sub assigned = Sub::zero(parent);
    assigned = copied;
    check_self(assigned);
    assert(assigned.generator_map().check_lifts());
    assigned = std::move(copied);
    check_self(assigned);
    assert(assigned.generator_map().check_lifts());
    std::vector<Sub> relocated;
    relocated.push_back(original);
    relocated.push_back(original);
    for (const auto& S : relocated) check_self(S);

    // An owning categorical result keeps its actual Submodule alive; there
    // is no domain cache or implicit work on endpoint access.
    auto object = as_subobject(Sub(parent, Mat(1, 1, {{0}}, {{1,1}}, {{0,0}})));
    assert(dynamic_cast<const Sub*>(object.module.get()));
    assert(object.inclusion.domain() == object.module && object.inclusion.check_lifts());

    // R4 has no kernel implementation, so even this accessor must still work.
    using Four = R4GradedSparseMatrix<int>;
    using FourHom = Homomorphism<Four>;
    auto four_parent = std::make_shared<const Module<Four>>(Four(0, 1, {}, {}, {r4degree(0,0,0,0)}));
    Submodule<Four> S(four_parent, Four(1, 1, {{0}}, {r4degree(1,1,1,1)},
                                        four_parent->presentation().row_degrees));
    const auto& inclusion = S.generator_map();
    assert(inclusion.domain().get() == &S && !inclusion.domain()->has_presentation());
    auto composed = inclusion.compose(FourHom::identity(four_parent));
    assert(composed.domain().get() == &S);
    assert(composed.image(false).equals(S));
    rejects([&] { S.compute_presentation(); });
    assert(!S.has_presentation());
}

void test_sorting_transports_the_inclusion() {
    auto parent = std::make_shared<const Mod>(Mat(0, 2, {}, {}, {{0,0}, {0,0}}));
    Sub S(parent, Mat(2, 2, {{0}, {1}}, {{1,0}, {0,1}}, parent->presentation().row_degrees));
    S.compute_presentation();
    Mod& base = S;
    base.sort_compatibly();
    assert(S.presentation().row_degrees == S.generator_map().generator_lift().col_degrees);
    assert(S.generator_map().generator_lift().data == array<int>({{1}, {0}}));
    assert(S.generator_map().check_lifts());
    S.sort_compatibly(Degree_traits<r2degree>::colex_lambda());
    assert(S.generator_map().generator_lift().data == array<int>({{0}, {1}}));
    assert(S.generator_map().check_lifts());
}

int main() {
    test_sorting_transports_the_inclusion();
    test_generator_map_homomorphism();
    test_self_domain_copy_move_and_explicit_computation();
    test_canonical_submodule_members();
    test_in_place_module_storage();
    test_shared_generator_basis();
    test_invalidation_and_automatic_resolution();
    test_zero_without_kernel_and_polymorphic_destruction();
}
