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
    assert(S->generators().data == G.data && S->parent().get() == parent.get());
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

void test_basis_distinction_and_compatibility() {
    auto parent = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0,0}}));
    Mat G(2, 1, {{0}, {0}}, {{1,1}, {1,1}}, {{0,0}});
    Sub S(parent, G);
    auto standalone = S.presented_module(); // old API now stores on mutable S too
    assert(S.has_presentation() && S.number_of_generators() == 1);
    assert(S.number_of_embedding_generators() == 2 && S.generators().data == G.data);
    assert(standalone.presentation().data == S.presentation().data);
    const Sub& constant = S;
    // Canonical maps use the defining generator basis, not a potentially
    // minimized/reordered stored presentation. The compatibility path rebuilds it.
    auto object = as_subobject(constant);
    assert(object.module->number_of_generators() == 2 && object.inclusion.check_lifts());
    assert(S.number_of_generators() == 1); // const adapter has not overwritten S
    S.compute_presentation();
    assert(S.number_of_generators() == 2 && S.presentation().row_degrees == G.col_degrees);
    S.compute_projective_resolution();
    S.minimize(); // inherited module minimization operates on the stored resolution
    assert(S.number_of_generators() == 1 && S.number_of_embedding_generators() == 2);
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
    assert(!S.has_projective_resolution() && !S.has_injective_resolution());
    assert(S.generators().data == array<int>({{1}}));
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
    rejects([&] { whole.compute_presentation(); }); // nontrivial kernel still unsupported
    assert(!whole.has_presentation());

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

int main() {
    test_in_place_module_storage();
    test_basis_distinction_and_compatibility();
    test_invalidation_and_automatic_resolution();
    test_zero_without_kernel_and_polymorphic_destruction();
}
