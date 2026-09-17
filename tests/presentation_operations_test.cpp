#include <grlina/submodule.hpp>
#include <grlina/hom_interface.hpp>
#include <grlina/modules.hpp>
#include <cassert>
using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;
using Sub = Submodule<Mat>;
using Hom = Homomorphism<Mat>;

template <typename Action> void rejects(Action action) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

void containment() {
    // e0=(e0+e1)+e1 only after BOTH incomparable generators are available.
    Mat spanning(2, 2, {{0,1}, {1}}, {{1,0}, {0,1}}, {{0,0}, {0,0}});
    Mat late(1, 2, {{0}}, {{1,1}}, spanning.row_degrees);
    Mat early(1, 2, {{0}}, {{1,0}}, spanning.row_degrees);
    assert(image_contained_in_image(late, spanning));
    assert(!image_contained_in_image(early, spanning));
    assert(!image_contained_in_image(spanning, late));
    Mat different(0, 1, {}, {}, {{0,0}});
    rejects([&] { image_contained_in_image(late, different); });
    Mat relation(1, 2, {{0,1}}, {{0,0}}, {{0,0}, {0,0}});
    auto parent = std::make_shared<const Mod>(relation);
    Mat e0(1, 2, {{0}}, {{0,0}}, relation.row_degrees);
    Mat e1(1, 2, {{1}}, {{0,0}}, relation.row_degrees);
    Sub A(parent, e0), B(parent, e1);
    assert(!image_contained_in_image(e0, e1));
    assert(A.contains(B) && A.is_contained_in(B) && A.equals(B));
    assert(present_same_submodule(relation, e0, e1));
    auto zero = Sub::zero(parent);
    assert(A.contains(zero) && !zero.contains(A) && !A.equals(zero));
    Sub foreign(std::make_shared<const Mod>(relation), e0);
    rejects([&] { A.contains(foreign); });
    rejects([&] { A.equals(foreign); });
    // No kernel implementation is needed, even in R4.
    using Four = R4GradedSparseMatrix<int>;
    auto p4 = std::make_shared<const Module<Four>>(Four(0, 1, {}, {}, {r4degree(0,0,0,0)}));
    auto z4 = Submodule<Four>::zero(p4), w4 = Submodule<Four>::whole(p4);
    assert(w4.contains(z4) && !z4.contains(w4) && w4.equals(w4));

    // Compare batched containment with the coefficient solver, including
    // repeated, nonadjacent degrees and incomparable available columns.
    for (int mask = 0; mask < 64; ++mask) {
        array<int> columns(3);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 2; ++i)
                if (mask & (1 << (2*j+i))) columns[j].push_back(i);
        Mat A(3, 2, columns, {{1,0}, {0,1}, {1,1}}, {{0,0}, {0,0}});
        for (r2degree degree : {r2degree{0,0}, {1,0}, {0,1}, {1,1}}) {
            Mat B(4, 2, {{0}, {1}, {0,1}, {}},
                  {degree, {1,1}, degree, {0,0}}, A.row_degrees);
            assert(image_contained_in_image(B, A) == solve_graded_linear_system(A, B).has_value());
            assert(A.data == columns && B.data == array<int>({{0}, {1}, {0,1}, {}}));
        }
    }
}

void presentation_adapters() {
    // Non-sorted ambient rows: no helper may permute them.
    Mat free(0, 2, {}, {}, {{1,0}, {0,0}});
    auto zero = zero_submodule(free), whole = all_submodule(free);
    assert(zero.get_num_cols() == 0 && zero.row_degrees == free.row_degrees);
    assert(whole.data == array<int>({{0}, {1}}));
    assert(whole.row_degrees == free.row_degrees && whole.col_degrees == free.row_degrees);
    Mat first(1, 2, {{0}}, {{1,0}}, free.row_degrees);
    Mat second(1, 2, {{1}}, {{0,0}}, free.row_degrees);
    auto sum = submodule_sum(first, second);
    assert(sum.row_degrees == free.row_degrees && sum.get_num_cols() == 2);
    assert(present_same_submodule(free, sum, whole));
    Mat duplicates(2, 2, {{0}, {0}}, {{1,0}, {1,0}}, free.row_degrees);
    auto reduced = reduce_submodule(free, duplicates);
    assert(reduced.data == array<int>({{0}}) && reduced.row_degrees == free.row_degrees);
    assert(reduce_submodule(free, duplicates, false).data == reduced.data);
    Mat line(0, 1, {}, {}, {{0,0}});
    auto basis = homomorphism_lift_basis(line, line);
    assert(basis.size() == 1 && basis[0].data == array<int>({{0}}));
    assert(!line.rows_computed);
    Mat torsion(1, 1, {{0}}, {{1,1}}, {{0,0}});
    assert(homomorphism_lift_basis(torsion, line).empty());
    assert(End_2d_0(line, {0,0}).empty());
    // Two free generators born at 0 and 1: End has 3 allowed matrix entries;
    // shifting by 1 admits the fourth (early generator -> late generator).
    Mat births(0, 2, {}, {}, {{0,0}, {1,1}});
    auto extra = End_2d_0(births, {1,1});
    assert(extra.size() == 1 && extra[0].data == array<int>({{1}, {}}));
    assert(extra[0].is_graded_matrix());
    auto domain = std::make_shared<const Mod>(births);
    auto target_value = births;
    target_value.shift({1,1});
    auto target = std::make_shared<const Mod>(target_value);
    assert(Hom(domain, target, extra[0]).check_lifts());
    rejects([&] { End_2d_0(births, {-1,-1}); });
}

void matrix_families() {
    // Rectangular: 3 rows, 2 columns. x=E00, y=E12, z=E10 are independent.
    Mat xy(2, 3, {{0}, {2}}, {{0,0}, {0,0}}, {{0,0}, {0,0}, {0,0}});
    Mat y(2, 3, {{}, {2}}, xy.col_degrees, xy.row_degrees);
    Mat x(2, 3, {{0}, {}}, xy.col_degrees, xy.row_degrees);
    Mat z(2, 3, {{}, {0}}, xy.col_degrees, xy.row_degrees);
    std::vector<Mat> A{xy}, B{y, x, z, z};
    reduce_matrix_family_modulo(A, B);
    assert(A.size() == 1 && A[0].data == xy.data);
    assert(B.size() == 2 && B[0].data == x.data && B[1].data == z.data);
    assert(B[0].row_degrees == y.row_degrees);
    std::vector<Mat> empty, also_empty;
    reduce_matrix_family_modulo(empty, also_empty);
    rejects([&] { reduce_matrix_family_modulo(empty, empty); });
    reduce_matrix_family_modulo(empty, B);
    assert(B.size() == 2);
    std::vector<Mat> wrong{Mat(0, 0, {}, {}, {})};
    rejects([&] { reduce_matrix_family_modulo(A, wrong); });
}

void homomorphism_shifts() {
    Mod square(Mat(2, 1, {{0}, {0}}, {{1,0}, {0,1}}, {{0,0}}));
    square.compute_projective_resolution();
    auto source = std::make_shared<const Mod>(square);
    auto can = Hom::canonical_shift(source, {0.5,0.5});
    const Mod& source_ref = *source;
    auto from_ref = Hom::canonical_shift(source_ref, {0.5,0.5});
    assert(from_ref.domain() == source);
    assert(from_ref.check_lifts());
    assert(from_ref.image(source_ref.whole_submodule(), false).parent() == from_ref.target());
    bool unowned_threw = false;
    try { (void)Hom::canonical_shift(square, {0.5,0.5}); }
    catch (const std::bad_weak_ptr&) { unowned_threw = true; }
    assert(unowned_threw);
    assert(can.domain() == source && can.check_lifts());
    assert(can.target()->has_complete_projective_resolution());
    assert(can.lifts().size() == source->projective_resolution().size() + 1);
    assert(can.generator_lift().row_degrees == vec<r2degree>({{-0.5,-0.5}}));
    assert(can.generator_lift().col_degrees == source->presentation().row_degrees);
    assert(source->presentation().row_degrees == vec<r2degree>({{0,0}}));
    assert(canonical_shift_lift(source->presentation(), {0.5,0.5}).data == can.generator_lift().data);
    rejects([&] { Hom::canonical_shift(source, {-1,-1}); });
    rejects([&] { Hom::canonical_shift(nullptr, {1,1}); });
    auto translated = can.shifted({1,2});
    assert(translated.check_lifts() && translated.lifts().size() == can.lifts().size());
    assert(translated.domain()->presentation().row_degrees == vec<r2degree>({{-1,-2}}));
    auto identity = Hom::identity(source).shifted({-1,-2});
    assert(identity.domain() == identity.target() && identity.check_lifts());
    auto restricted = can.image(Sub::whole(source), false);
    assert(restricted.parent() == can.target());
    assert(restricted.generator_map().generator_lift().data == can.generator_lift().data);
    assert(can.image(Sub::zero(source)).is_zero());
    rejects([&] { can.image(Sub::whole(can.target())); });
}

void typed_additional_lifts() {
    for (const Mat& presentation : {
            Mat(0, 3, {}, {}, {{0,0}, {1,1}, {2,2}}),
            Mat(2, 3, {{0}, {1,2}}, {{3,3}, {4,4}}, {{0,0}, {1,1}, {2,2}})}) {
        auto domain = std::make_shared<Mod>(presentation);
        for (const r2degree amount : {r2degree{0,0}, r2degree{1,1}, r2degree{2,2}}) {
            auto expected = End_2d_0(presentation, amount);
            auto actual = End_2d_0(domain, amount);
            auto eta = Hom::canonical_shift(domain, amount);
            auto shared = End_2d_0(eta);
            assert(actual.size() == expected.size());
            assert(shared.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                assert(shared[i].domain() == eta.domain());
                assert(shared[i].target() == eta.target());
                assert(shared[i].generator_lift().data == expected[i].data);
                shared[i].validate();
                assert(shared[i].check_lifts());
                assert(shared[i].preimage(Sub::whole(eta.target()), false).equals(Sub::whole(domain)));
                assert(eta.preimage(shared[i].image(false), false).parent() == domain);
                assert(actual[i].domain().get() == domain.get());
                assert(actual[i].target() == actual[0].target());
                assert(actual[i].generator_lift().data == expected[i].data);
                assert(actual[i].generator_lift().row_degrees == expected[i].row_degrees);
                assert(actual[i].generator_lift().col_degrees == expected[i].col_degrees);
                assert(actual[i].check_lifts());
            }
        }
        assert(domain->presentation().data == presentation.data);
        assert(domain->presentation().row_degrees == presentation.row_degrees);
        assert(!domain->presentation().rows_computed);
    }
    auto domain = std::make_shared<const Mod>(Mat(0, 2, {}, {}, {{0,0}, {1,1}}));
    assert(End_2d_0(std::make_shared<const Mod>(Mat(0, 0, {}, {}, {})), {1,1}).empty());
    assert(End_2d_0(Hom::canonical_shift(
        std::make_shared<const Mod>(Mat(0, 0, {}, {}, {})), {1,1})).empty());
    rejects([&] { End_2d_0(Hom(nullptr, domain, Mat{})); });
    rejects([&] { End_2d_0(Hom(domain, nullptr, Mat{})); });
    auto result = End_2d_0(domain, {1,1});
    assert(result.size() == 1);
    std::weak_ptr<const Mod> lifetime = domain;
    rejects([&] { End_2d_0(domain, {-1,-1}); });
    rejects([&] { End_2d_0(std::shared_ptr<const Mod>{}, {1,1}); });
    domain.reset();
    assert(!lifetime.expired() && result[0].check_lifts());

    // Moving into a homomorphism must preserve the sparse storage allocation.
    Mat lift = result[0].generator_lift();
    const auto* storage = lift.data.data();
    const auto* entries = lift.data[0].data();
    Hom moved(result[0].domain(), result[0].target(), std::move(lift));
    assert(moved.generator_lift().data.data() == storage);
    assert(moved.generator_lift().data[0].data() == entries);
}

void identity_shortcuts() {
    auto X = std::make_shared<const Mod>(Mat(1, 2, {{0,1}}, {{3,3}}, {{0,0}, {1,1}}));
    auto eta = Hom::canonical_shift(X, {1,1});
    assert(eta.id_matrix() && Hom::identity(X).id_matrix());
    assert(eta.shifted({2,2}).id_matrix());
    Hom ordinary(eta.domain(), eta.target(), eta.lifts());
    assert(!ordinary.id_matrix());
    Sub I(X, Mat(2, 2, {{0,1}, {1}}, {{2,2}, {1,1}}, X->presentation().row_degrees));
    auto fast = eta.image(I, false);
    auto slow = ordinary.image(I, false);
    fast.validate();
    assert(fast.parent() == eta.target() && fast.generator_map().generator_lift().data == slow.generator_map().generator_lift().data);
    assert(fast.generator_map().generator_lift().col_degrees == I.generator_map().generator_lift().col_degrees);
    assert(fast.generator_map().generator_lift().row_degrees == eta.target()->presentation().row_degrees);

    auto next = Hom::canonical_shift(eta.target(), {2,2});
    auto composite = eta.compose(next);
    assert(composite.id_matrix() && composite.check_lifts());
    composite.validate();
    for (std::size_t i = 0; i < composite.lifts().size(); ++i) {
        auto expected = next.lifts()[i] * eta.lifts()[i];
        assert(composite.lifts()[i].data == expected.data);
        assert(composite.lifts()[i].row_degrees == expected.row_degrees);
        assert(composite.lifts()[i].col_degrees == expected.col_degrees);
    }
    Hom unmarked_next(next.domain(), next.target(), next.lifts());
    auto left_identity = eta.compose(unmarked_next);
    auto right_identity = ordinary.compose(next);
    assert(!left_identity.id_matrix() && !right_identity.id_matrix());
    assert(left_identity.check_lifts() && right_identity.check_lifts());
    left_identity.validate();
    right_identity.validate();
    assert(!((eta + ordinary).id_matrix()));
    assert((eta + ordinary).image(false).is_zero());
    auto free_two = std::make_shared<const Mod>(Mat(0, 2, {}, {}, {{0,0}, {1,1}}));
    auto free_three = std::make_shared<const Mod>(Mat(0, 3, {}, {}, {{-2,-2}, {-2,-2}, {-2,-2}}));
    auto before = Hom::canonical_shift(free_two, {1,1});
    Hom rectangular(before.target(), free_three, Mat(2, 3, {{0,2}, {1}},
        before.target()->presentation().row_degrees, free_three->presentation().row_degrees));
    auto after = Hom::canonical_shift(free_three, {1,1});
    auto first = before.compose(rectangular);
    auto second = rectangular.compose(after);
    for (auto* map : {&first, &second}) {
        map->validate();
        assert(map->check_lifts() && !map->id_matrix());
        assert(map->generator_lift().data == rectangular.generator_lift().data);
    }
    assert(first.generator_lift().col_degrees == free_two->presentation().row_degrees);
    assert(second.generator_lift().row_degrees == after.target()->presentation().row_degrees);
    rejects([&] { eta.compose(eta); });
    rejects([&] { eta.image(Sub::whole(eta.target()), false); });
    static_assert(!std::is_constructible_v<Hom, std::shared_ptr<const Mod>,
        std::shared_ptr<const Mod>, Mat, bool>);
    static_assert(!std::is_constructible_v<Sub, std::shared_ptr<const Mod>, Mat, bool>);

    // A quotient is identity on F0, but has a rectangular, nonidentity F1 map.
    auto quotient = as_quotient(I);
    assert(quotient.projection.id_matrix());
    quotient.projection.lift_to_resolution();
    assert(quotient.projection.id_matrix() && quotient.projection.check_lifts());
    auto shifted_quotient = Hom::canonical_shift(quotient.module, {1,1});
    auto composed_quotient = quotient.projection.compose(shifted_quotient);
    composed_quotient.validate();
    assert(composed_quotient.check_lifts());
    assert(composed_quotient.lifts()[1].data ==
           (shifted_quotient.lifts()[1] * quotient.projection.lifts()[1]).data);
    auto twice = quotient.projection.compose(Hom::identity(quotient.module));
    // Only F0 of twice is known to be identity; F1 still must be multiplied.
    Hom unmarked_identity(quotient.module, quotient.module, Hom::identity(quotient.module).lifts());
    auto three = twice.compose(unmarked_identity);
    three.validate();
    assert(three.check_lifts());
    auto quotient_image = quotient.projection.image(I, false);
    assert(quotient_image.is_zero());
    auto zero_module = std::make_shared<const Mod>(Mat(0,0,{}, {}, {}));
    auto zero_identity = Hom::identity(zero_module);
    zero_identity.compose(zero_identity).validate();
    assert(zero_identity.image(Sub::whole(zero_module), false).is_zero());
}

void submodule_quotients_and_parent_minimization() {
    // Repeated, unsorted ambient degrees. The quotient cancels e0 via e0+e1.
    Mat P(2, 3, {{1,2}, {1,2}}, {{2,2}, {3,3}}, {{1,1}, {0,0}, {0,0}});
    auto X = std::make_shared<const Mod>(P);
    Sub I(X, Mat(3, 3, {{0,2}, {0}, {2}}, {{1,1}, {1,1}, {0,0}}, P.row_degrees));
    Sub K(X, Mat(1, 3, {{0,1}}, {{1,1}}, P.row_degrees));
    auto raw = I.submodule_quotient(K);
    assert(raw.parent() != X && I.parent() == X && K.parent() == X);
    assert(raw.parent()->number_of_generators() == 3 && raw.parent()->number_of_relations() == 3);
    assert(raw.generator_map().generator_lift().data == I.generator_map().generator_lift().data);
    assert(raw.generator_map().generator_lift().row_degrees == I.generator_map().generator_lift().row_degrees);
    auto lazy = I.submodule_quotient(K, true);
    auto full = I.submodule_quotient(K, false, true);
    auto both = I.submodule_quotient(K, true, true);
    for (auto* result : {&lazy, &full, &both}) {
        result->validate();
        assert(result->parent()->number_of_generators() == 2);
        assert(result->parent()->number_of_relations() == 1);
        assert(result->generator_map().generator_lift().data == array<int>({{0,1}, {0}, {1}}));
        assert(result->generator_map().generator_lift().col_degrees == I.generator_map().generator_lift().col_degrees);
        auto presented = result->presented_module(true);
        auto expected = raw.presented_module(true);
        for (const r2degree d : {r2degree{0,0}, {1,1}, {2,2}, {4,4}})
            assert(presented.dimension_at(d) == expected.dimension_at(d));
    }
    assert(X->presentation().data == P.data && X->presentation().row_degrees == P.row_degrees);
    auto saved_parent = raw.parent();
    raw.compute_presentation();
    assert(raw.has_presentation());
    raw.lazy_minimize_parent();
    assert(!raw.has_presentation() && raw.parent() != saved_parent);
    assert(saved_parent->number_of_generators() == 3);
    rejects([&] { I.submodule_quotient(Sub::whole(std::make_shared<const Mod>(P))); });
    assert(I.submodule_quotient(Sub::whole(X), true).is_zero());
    auto zero = Sub::zero(X).submodule_quotient(K, true);
    zero.validate();
    assert(zero.is_zero());

    // Chained cancellations must substitute into every accompanying column.
    auto chain = std::make_shared<const Mod>(Mat(2, 3, {{0,1}, {1,2}},
        {{2,2}, {1,1}}, {{2,2}, {1,1}, {0,0}}));
    Sub single(chain, Mat(1, 3, {{0}}, {{2,2}}, chain->presentation().row_degrees));
    single.lazy_minimize_parent();
    assert(single.parent()->number_of_generators() == 1);
    assert(single.generator_map().generator_lift().data == array<int>({{0}}));
    assert(single.generator_map().generator_lift().row_degrees == vec<r2degree>({{0,0}}));
    assert(single.generator_map().generator_lift().col_degrees == vec<r2degree>({{2,2}}));

    // Lazy reduction misses this relation; the graded kernel finds it.
    Mat redundant(3, 3, {{0,2}, {1,2}, {0,1}}, {{0,1}, {1,0}, {1,1}},
                  {{-1,-1}, {-1,-1}, {-1,-1}});
    auto R = std::make_shared<Mod>(redundant);
    auto whole = Sub::whole(R);
    whole.lazy_minimize_parent();
    assert(whole.parent()->number_of_relations() == 3);
    whole.minimize_parent();
    assert(whole.parent()->number_of_relations() == 2);
    whole.validate();
    // Removing only relations preserves original ambient coordinates and caches
    // belonging to submodules, even when the parent rows need sorting.
    Mat unsorted(3, 3, {{0,2}, {1,2}, {0,1}}, {{2,3}, {3,2}, {3,3}},
                 {{1,1}, {-1,-1}, {0,0}});
    auto U = std::make_shared<Mod>(unsorted);
    auto retained = Sub::whole(U);
    retained.compute_presentation();
    U->compute_projective_resolution();
    U->remove_extra_rels();
    assert(U->number_of_relations() == 2);
    assert(U->presentation().row_degrees == unsorted.row_degrees);
    assert(U->projective_resolution().size() == 1);
    retained.validate();
    assert(retained.equals(Sub::whole(U)));

    // Lazy cancellation also works where no graded kernel is implemented.
    using Four = R4GradedSparseMatrix<int>;
    auto four = std::make_shared<const Module<Four>>(Four(1, 1, {{0}},
        {r4degree(0,0,0,0)}, {r4degree(0,0,0,0)}));
    auto four_sub = Submodule<Four>::whole(four);
    four_sub.lazy_minimize_parent();
    assert(four_sub.parent()->number_of_generators() == 0);
    auto empty = std::make_shared<const Mod>(Mat(0,0,{}, {}, {}));
    auto empty_sub = Sub::whole(empty).submodule_quotient(Sub::zero(empty), false, true);
    empty_sub.validate();
    assert(empty_sub.is_zero());
}

int main() { containment(); presentation_adapters(); matrix_families(); homomorphism_shifts(); typed_additional_lifts(); identity_shortcuts(); submodule_quotients_and_parent_minimization(); }
