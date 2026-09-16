#include <grlina/modules.hpp>
#include <cassert>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;
using Hom = Homomorphism<Mat>;
using Ptr = std::shared_ptr<const Mod>;
static_assert(std::is_same_v<Mod, PersistenceModule<Mat>>);
static_assert(std::is_same_v<Hom, ModuleMorphism<Mat>>);

template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

Mat identity_lift(const Mod& source, const Mod& target) {
    Mat I(source.number_of_generators(), target.number_of_generators(), "Identity");
    I.col_degrees = source.presentation().row_degrees;
    I.row_degrees = target.presentation().row_degrees;
    return I;
}

void test_homomorphisms() {
    Ptr free = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0, 0}}));
    Ptr interval = std::make_shared<Mod>(Mat(1, 1, {{0}}, {{1, 1}}, {{0, 0}}));
    Hom quotient(free, interval, identity_lift(*free, *interval));
    assert(quotient.check_lifts());
    assert(quotient.generator_lift().compatibly_sorted);
    Submodule<Mat> zero_from_relations(interval, interval->presentation());
    assert(zero_from_relations.number_of_generators() == 1 && zero_from_relations.is_zero());
    assert(zero_from_relations.generators().compatibly_sorted);
    assert(!Submodule<Mat>::whole(interval).is_zero());
    auto relation_lift = lift_to_relations(free->presentation(), interval->presentation(), quotient.generator_lift());
    assert(relation_lift && relation_lift->get_num_cols() == 0);
    Hom manual(interval, free, identity_lift(*interval, *free)); // intentionally trusted, invalid input
    assert(!manual.check_lifts());
    assert(!lift_to_relations(interval->presentation(), free->presentation(), manual.generator_lift()));
    rejects([&] { manual.lift_to_resolution(); });
    rejects([&] { module_hom_space_basis<Mat>(nullptr, interval); });
    rejects([&] { module_endomorphism_basis<Mat>(nullptr); });

    Mat A(1, 2, {{0, 1}}, {{2, 2}}, {{0, 0}, {0, 0}});
    Mat B(2, 2, {{0}, {1}}, {{1, 1}, {2, 2}}, A.row_degrees);
    Mat f0(2, 2, {{0}, {1}}, A.row_degrees, B.row_degrees);
    auto f1 = lift_to_relations(A, B, f0);
    assert(f1 && f1->data == array<int>({{0, 1}}));
    B.col_degrees[1] = {3, 3}; // ungraded solution exists, but graded solution does not
    assert(!lift_to_relations(A, B, f0));

    auto resolved = std::make_shared<Mod>(interval->presentation());
    resolved->compute_projective_resolution();
    Hom I = Hom::identity(resolved);
    assert(I.lifts().size() == 3 && I.check_lifts());
    Hom J(resolved, resolved, I.generator_lift());
    J.lift_to_resolution();
    assert(J.lifts().size() == 3 && J.check_lifts());
    auto composed = I.then(J);
    assert(composed.lifts().size() == 3 && composed.check_lifts());
    auto zero = I + J;
    assert(zero.check_lifts() && zero.generator_lift().data == array<int>({{}}));

    auto K = kernel(quotient);
    assert(K.inclusion.check_lifts());
    assert(K.module->dimension_at({0, 0}) == 0);
    assert(K.module->dimension_at({1, 1}) == 1);
    assert(K.inclusion.then(quotient).check_lifts());
    auto Im = image(quotient);
    assert(Im.inclusion.check_lifts() && Im.module->dimension_at({0, 0}) == 1);
    auto Co = cokernel(quotient);
    assert(Co.projection.check_lifts() && Co.module->dimension_at({0, 0}) == 0);
    auto Ci = coimage(quotient);
    assert(Ci.projection.check_lifts());
    assert(Ci.module->dimension_at({0, 0}) == 1 && Ci.module->dimension_at({1, 1}) == 0);
    assert(quotient.cokernel().dimension_at({0, 0}) == 0);
    assert(quotient.coimage().dimension_at({1, 1}) == 0);

    auto D = direct_sum<Mat>(free, interval);
    assert(D.module->dimension_at({0, 0}) == 2 && D.module->dimension_at({1, 1}) == 1);
    for (const auto* map : {&D.inclusion_left, &D.inclusion_right, &D.projection_left, &D.projection_right})
        assert(map->check_lifts());
    assert(D.inclusion_left.then(D.projection_left).generator_lift().data == array<int>({{0}}));
    assert(D.inclusion_left.then(D.projection_right).generator_lift().data == array<int>({{}}));
    auto reconstructed = D.projection_left.then(D.inclusion_left) + D.projection_right.then(D.inclusion_right);
    assert(reconstructed.generator_lift().data == array<int>({{0}, {1}}));
    assert(product<Mat>(free, interval).module->dimension_at({0, 0}) == 2);
    assert(coproduct<Mat>(free, interval).module->dimension_at({1, 1}) == 1);

    auto E = equalizer(quotient, Hom::zero(free, interval));
    assert(E.inclusion.check_lifts() && E.module->dimension_at({1, 1}) == 1);
    auto CE = coequalizer(quotient, quotient);
    assert(CE.projection.check_lifts() && CE.module->dimension_at({0, 0}) == 1);
    auto PB = pullback(quotient, quotient);
    assert(PB.to_left.check_lifts() && PB.to_right.check_lifts());
    assert(PB.module->dimension_at({0, 0}) == 1 && PB.module->dimension_at({1, 1}) == 2);
    auto difference = PB.to_left.then(quotient) + PB.to_right.then(quotient);
    assert(difference.image().is_zero());
    auto PO = pushout(quotient, quotient);
    assert(PO.from_left.check_lifts() && PO.from_right.check_lifts());
    assert(PO.module->dimension_at({0, 0}) == 1 && PO.module->dimension_at({1, 1}) == 0);
    assert((quotient.then(PO.from_left) + quotient.then(PO.from_right)).image().is_zero());
    rejects([&] { pullback(quotient, Hom::identity(free)); });
    rejects([&] { pushout(quotient, Hom::identity(interval)); });
}

void test_kernel_minimization() {
    // a+b=c at their join. Ordinary graded column reduction misses this:
    // a,b have the same pivot and incomparable degrees; c has a smaller pivot.
    Mat P(3, 3, {{0, 2}, {1, 2}, {0, 1}}, {{0, 1}, {1, 0}, {1, 1}},
          {{-1, -1}, {-1, -1}, {-1, -1}});
    Mat cheap = P;
    cheap.column_reduction_graded_w_deletion();
    assert(cheap.get_num_cols() == 3);
    Mat minimized = P;
    minimized.minimize();
    assert(minimized.get_num_cols() == 2 && minimized.get_num_rows() == 3);
    for (const r2degree d : {r2degree{-1,-1}, {0,1}, {1,0}, {1,1}, {2,2}})
        assert(Mod(P).dimension_at(d) == Mod(minimized).dimension_at(d));
    cheap.minimize_variant();
    assert(cheap.data == minimized.data && cheap.col_degrees == minimized.col_degrees);
    auto parent = std::make_shared<Mod>(Mat(0, 3, {}, {}, P.row_degrees));
    Submodule<Mat> S(parent, P);
    S.minimize_generators();
    assert(S.number_of_generators() == 2);
    Mat unsorted = P;
    unsorted.permute_rows_graded({2, 0, 1});
    auto source = unsorted;
    auto K = source.graded_kernel();
    assert(K.row_degrees == unsorted.col_degrees);
    assert((unsorted * K).is_zero());
}

void test_resolution_minimization_and_sorting() {
    Mat d1(2, 2, {{0, 1}, {0, 1}}, {{0, 0}, {0, 0}}, {{0, 0}, {0, 0}});
    Mat d2 = d1;
    Mat d3(1, 2, {{0, 1}}, {{0, 0}}, d2.col_degrees);
    ChainComplex<Mat> resolution({d1, d2, d3});
    assert(resolution.squares_to_zero());
    Mod M(resolution);
    M.minimize();
    assert(M.projective_resolution().size() == 3 && M.projective_resolution().squares_to_zero());
    assert(M.number_of_generators() == 1 && M.number_of_relations() == 0);
    assert(M.projective_resolution()[1].get_num_cols() == 0);
    assert(M.projective_resolution()[2].get_num_cols() == 0);
    Mod presentation_only(resolution);
    presentation_only.minimize_presentation();
    assert(presentation_only.projective_resolution().size() == 1);
    Mod editable(resolution, resolution); // storage compatibility, exactness trusted
    editable.mutable_presentation();
    assert(editable.projective_resolution().size() == 1 && !editable.has_injective_resolution());

    Mat triangle(3, 3, {{0,2}, {1,2}, {0,1}}, {{0,1}, {1,0}, {1,1}},
                 {{-1,-1}, {-1,-1}, {-1,-1}});
    Mat syzygy(1, 3, {{0,1,2}}, {{1,1}}, triangle.col_degrees);
    Mod T(ChainComplex<Mat>({triangle, syzygy}));
    T.minimize();
    assert(T.number_of_relations() == 2 && T.projective_resolution()[1].get_num_cols() == 0);
    assert(T.projective_resolution().squares_to_zero());

    // Repeated grades with distinct basis elements must be permuted identically.
    Mat fold(3, 1, {{0}, {0}, {0}}, {{2,2}, {0,0}, {0,0}}, {{0,0}});
    Mat boundary(2, 3, {{0,1}, {1,2}}, {{2,2}, {0,0}}, fold.col_degrees);
    ChainComplex<Mat> C({fold, boundary});
    C.sort_compatibly();
    assert(C.squares_to_zero());
    assert(C[1].row_degrees == C[0].col_degrees);
    assert(C[1].data == array<int>({{0,1}, {0,2}}));
    C.sort_compatibly(Degree_traits<r2degree>::colex_lambda());
    assert(C.squares_to_zero());
    Mat stale(2, 1, {{0}, {0}}, {{1,1}, {2,2}}, {{0,0}});
    stale.col_degrees[0] = {3,3};
    assert(!stale.compatibly_sorted);
    rejects([&] { stale.minimize(); });
    rejects([&] { stale.sort_compatibly([](auto a, auto b) { return a > b; }); });
    Mat antichain(3, 0, {{}, {}, {}}, {{0,2}, {1,1}, {2,0}}, {});
    rejects([&] { antichain.sort_compatibly([](auto a, auto b) {
        return (static_cast<int>(a.first) + 1) % 3 == static_cast<int>(b.first);
    }); });
    stale.sort_compatibly();
    Mat copied = stale;
    stale.col_degrees[0] = {4,4};
    assert(copied.compatibly_sorted && !stale.compatibly_sorted);
    Mat moved = std::move(copied);
    assert(moved.compatibly_sorted);
    Mat permutation_input(3, 3, {{0}, {1}, {2}}, {{2,2}, {0,0}, {1,1}},
                          {{-3,-3}, {-2,-2}, {-1,-1}});
    Mat historical = permutation_input;
    auto permutation = permutation_input.sort_columns_with_permutation();
    assert(permutation.old_to_new == vec<int>({2,0,1}));
    assert(permutation.new_to_old == vec<int>({1,2,0}));
    assert(historical.sort_columns_lexicographically_with_output() == permutation.old_to_new);
    assert(historical.data == permutation_input.data);
    historical.permute_rows_graded({2,0,1});
    auto row_permutation = historical.sort_rows_with_permutation();
    assert(row_permutation.old_to_new == vec<int>({1,2,0}));
    assert(historical.data == permutation_input.data);
    R4GradedSparseMatrix<int> higher(1, 1, {{0}}, {r4degree(1,1,1,1)}, {r4degree(0,0,0,0)});
    rejects([&] { higher.minimize(); }); // deliberate kernel implementation frame
    higher.semi_minimize();
    assert(higher.get_num_cols() == 1);
    Module<R4GradedSparseMatrix<int>> injective(ChainComplex<R4GradedSparseMatrix<int>>({higher}),
                                               ResolutionKind::injective);
    rejects([&] { injective.minimize(); });
}

void test_chain_vs_resolution_minimization() {
    // [x x] is minimal as a complex, but not as a presentation of S/(x).
    // Its nonzero H1 is free on (1,1) in degree (1,0).
    Mat duplicate(2, 1, {{0}, {0}}, {{1,0}, {1,0}}, {{0,0}});
    ChainComplex<Mat> C({duplicate});
    C.minimize();
    assert(C[0].data == duplicate.data && C[0].col_degrees == duplicate.col_degrees);
    assert(homology_module(C, 1).dimension_at({1,0}) == 1);
    assert(homology_module(C, 1).dimension_at({0,0}) == 0);
    Mod presentation(C);
    presentation.minimize_resolution(); // explicit method also accepts one map
    assert(presentation.number_of_relations() == 1);
    assert(presentation.dimension_at({0,0}) == 1 && presentation.dimension_at({1,0}) == 0);

    // A truncated resolution of the square interval with duplicate top syzygies.
    // It is exact at F1, but H2 is free in degree (1,1).
    Mat square(2, 1, {{0}, {0}}, {{0,1}, {1,0}}, {{0,0}});
    Mat syzygies(2, 2, {{0,1}, {0,1}}, {{1,1}, {1,1}}, square.col_degrees);
    ChainComplex<Mat> truncated({square, syzygies});
    truncated.minimize();
    assert(truncated[1].data == syzygies.data);
    assert(homology_module(truncated, 2).dimension_at({1,1}) == 1);
    Mod M(truncated);
    M.minimize(); // dispatches to module resolution minimization
    assert(M.projective_resolution().size() == 2);
    assert(M.projective_resolution()[1].data == array<int>({{0,1}}));
    assert(M.projective_resolution().squares_to_zero());
    assert(homology_module(M.projective_resolution(), 2).dimension_at({1,1}) == 0);
    assert(M.dimension_at({0,0}) == 1 && M.dimension_at({1,0}) == 0);

    // Completing the resolution exposes a contractible pair: ordinary chain
    // minimization then removes both a duplicate syzygy and its dependency.
    Mat dependency(1, 2, {{0,1}}, {{1,1}}, syzygies.col_degrees);
    ChainComplex<Mat> complete({square, syzygies, dependency});
    complete.minimize();
    assert(complete[1].data == array<int>({{0,1}}));
    assert(complete[2].get_num_cols() == 0 && complete.squares_to_zero());

    // A zero differential is homology, not a contractible summand. The R4
    // example also proves that chain minimization needs no graded_kernel.
    using Higher = R4GradedSparseMatrix<int>;
    Higher local(2, 2, {{0}, {}}, {r4degree(0,0,0,0), r4degree(1,1,1,1)},
                                {r4degree(0,0,0,0), r4degree(0,0,0,0)});
    ChainComplex<Higher> general({local});
    general.minimize();
    assert(general[0].get_num_rows() == 1 && general[0].data == array<int>({{}}));
    assert(general[0].col_degrees == vec<r4degree>({r4degree(1,1,1,1)}));
    Module<Higher> unsupported(ChainComplex<Higher>({local}));
    rejects([&] { unsupported.minimize_resolution(); });
    assert(unsupported.presentation().data == local.data); // strong exception guarantee

    ChainComplex<Mat> empty;
    empty.minimize();
    assert(empty.empty());
    Mat unit(1, 1, {{0}}, {{0,0}}, {{0,0}});
    ChainComplex<Mat> invalid({unit, unit});
    rejects([&] { invalid.minimize(); });
    assert(invalid[0].data == unit.data && invalid[1].data == unit.data);
    duplicate.col_degrees = {{2,0}, {1,0}};
    ChainComplex<Mat> unsorted({duplicate});
    rejects([&] { unsorted.minimize(false); });
    assert(unsorted[0].col_degrees == duplicate.col_degrees);
}

void test_scc() {
    std::stringstream zero("scc2020\n2\n0 0 0\n");
    Mod Z(zero);
    assert(Z.has_presentation() && Z.number_of_generators() == 0 && Z.dimension_at({0,0}) == 0);
    std::stringstream encoded;
    Z.to_stream(encoded);
    Mod round_trip(encoded);
    assert(round_trip.has_presentation() && round_trip.number_of_relations() == 0);
    rejects([] { std::stringstream s("scc2020\n2\n1 1 0\n1 1 1 ; 0\n0 0 0 ;\n");
        ChainComplex<R3GradedSparseMatrix<int>> wrong(s); });
    rejects([] { std::stringstream s("scc2020\n2\n0 0 0\n"); R3GradedSparseMatrix<int> wrong(s); });
    rejects([] { std::stringstream s("scc2020\n2\n1 1 rubbish\n"); Mod wrong(s); });
    rejects([] { std::stringstream s("scc2020\n2\n1 1 0\n1 1 ; 0 garbage\n0 0 ;\n"); Mod wrong(s); });
    rejects([] { std::stringstream s("scc2020\n3\n1 1 0\n1 1 ; 0\n0 0 ;\n");
        ChainComplex<R3GradedSparseMatrix<int>> wrong(s); });
}

int main() {
    test_homomorphisms();
    test_kernel_minimization();
    test_resolution_minimization_and_sorting();
    test_chain_vs_resolution_minimization();
    test_scc();
}
