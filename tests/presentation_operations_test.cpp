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
    assert(shifted_endomorphism_lift_complement(line, {0,0}).empty());
    // Two free generators born at 0 and 1: End has 3 allowed matrix entries;
    // shifting by 1 admits the fourth (early generator -> late generator).
    Mat births(0, 2, {}, {}, {{0,0}, {1,1}});
    auto extra = shifted_endomorphism_lift_complement(births, {1,1});
    assert(extra.size() == 1 && extra[0].data == array<int>({{1}, {}}));
    assert(extra[0].is_graded_matrix());
    auto domain = std::make_shared<const Mod>(births);
    auto target_value = births;
    target_value.shift({1,1});
    auto target = std::make_shared<const Mod>(target_value);
    assert(Hom(domain, target, extra[0]).check_lifts());
    rejects([&] { shifted_endomorphism_lift_complement(births, {-1,-1}); });
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
    assert(restricted.generators().data == can.generator_lift().data);
    assert(can.image(Sub::zero(source)).is_zero());
    rejects([&] { can.image(Sub::whole(can.target())); });
}

int main() { containment(); presentation_adapters(); matrix_families(); homomorphism_shifts(); }
