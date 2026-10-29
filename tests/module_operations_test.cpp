#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/invariants/hilbert_function.hpp>
#include "test_degree_matrices.hpp"
#include <grlina/storage/csc.hpp>
#include <grlina/module/constructions.hpp>
#include <grlina/hom_interface.hpp>
#include <cassert>
#include <cstdint>
#include <random>
#include <sstream>
#include <type_traits>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/quiver_representations.hpp>
#include <grlina/graded_kernel.hpp>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;
using Hom = Homomorphism<Mat>;
using Ptr = std::shared_ptr<const Mod>;
static_assert(std::is_same_v<Mod, PersistenceModule<Mat>>);
static_assert(std::is_same_v<Hom, ModuleMorphism<Mat>>);
static_assert(std::is_nothrow_move_constructible_v<Mat>);
static_assert(std::is_nothrow_move_constructible_v<R2GradedSparseMatrix<int, CSCStorage<int>>>);

struct CopyCountingMatrix : GradedSparseMatrix<CoordinateDegree<double, 2>, int, CopyCountingMatrix> {
    using Base = GradedSparseMatrix<CoordinateDegree<double, 2>, int, CopyCountingMatrix>;
    using Base::Base;
    inline static int copies = 0;
    inline static int kernel_calls = 0;
    CopyCountingMatrix() = default;
    CopyCountingMatrix(const CopyCountingMatrix& other) : Base(other) { ++copies; }
    CopyCountingMatrix(CopyCountingMatrix&&) = default;
    CopyCountingMatrix& operator=(const CopyCountingMatrix& other) {
        Base::assign(other);
        ++copies;
        return *this;
    }
    CopyCountingMatrix& operator=(CopyCountingMatrix&&) = default;
    CopyCountingMatrix graded_kernel() {
        ++kernel_calls;
        Mat source(this->get_num_cols(), this->get_num_rows(),
                   this->data, this->col_degrees, this->row_degrees);
        Mat kernel = source.graded_kernel();
        return CopyCountingMatrix(kernel.get_num_cols(), kernel.get_num_rows(),
                                  kernel.data, kernel.col_degrees, kernel.row_degrees);
    }
};

template <typename F> void rejects(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

Mat identity_lift(const Mod& source, const Mod& target) {
    Mat I(source.num_of_gens(), target.num_of_gens(), "Identity");
    I.col_degrees = source.presentation().row_degrees;
    I.row_degrees = target.presentation().row_degrees;
    return I;
}

void test_homomorphisms() {
    Ptr free = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0, 0}}));
    Ptr interval = std::make_shared<Mod>(Mat(1, 1, {{0}}, {{1, 1}}, {{0, 0}}));
    Hom quotient(free, interval, identity_lift(*free, *interval));

    assert(quotient.check_lifts());
    assert(!quotient.generator_lift().compatibly_sorted); // no implicit sorting certification
    Submodule<Mat> zero_from_relations(interval, interval->presentation());
    assert(zero_from_relations.num_of_gens() == 1 && zero_from_relations.is_zero());
    assert(!zero_from_relations.generator_map().generator_lift().compatibly_sorted);
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
    assert(I.resolution_maps().size() == 3 && I.check_lifts());
    Hom J(resolved, resolved, I.generator_lift());
    J.lift_to_resolution();
    assert(J.resolution_maps().size() == 3 && J.check_lifts());
    auto composed = I.compose(J);
    assert(composed.resolution_maps().size() == 3 && composed.check_lifts());
    auto zero = I + J;
    assert(zero.check_lifts() && zero.generator_lift().data == array<int>({{}}));

    auto K = kernel(quotient);
    assert(K.inclusion.check_lifts());
    assert(K.module->dimension_at({0, 0}) == 0);
    assert(K.module->dimension_at({1, 1}) == 1);
    assert(K.inclusion.compose(quotient).check_lifts());
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
    assert(D.inclusion_left.compose(D.projection_left).generator_lift().data == array<int>({{0}}));
    assert(D.inclusion_left.compose(D.projection_right).generator_lift().data == array<int>({{}}));
    auto reconstructed = D.projection_left.compose(D.inclusion_left) + D.projection_right.compose(D.inclusion_right);
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
    auto difference = PB.to_left.compose(quotient) + PB.to_right.compose(quotient);
    assert(difference.image().is_zero());
    auto PO = pushout(quotient, quotient);
    assert(PO.from_left.check_lifts() && PO.from_right.check_lifts());
    assert(PO.module->dimension_at({0, 0}) == 1 && PO.module->dimension_at({1, 1}) == 0);
    assert((quotient.compose(PO.from_left) + quotient.compose(PO.from_right)).image().is_zero());
    rejects([&] { pullback(quotient, Hom::identity(free)); });
    rejects([&] { pushout(quotient, Hom::identity(interval)); });
}

void test_kernel_minimization() {
    // a+b=c at their join. Ordinary graded column reduction misses this:
    // a,b have the same pivot and incomparable degrees; c has a smaller pivot.
    Mat P(3, 3, {{0, 2}, {1, 2}, {0, 1}}, {{0, 1}, {1, 0}, {1, 1}},
          {{-1, -1}, {-1, -1}, {-1, -1}});
    P.sort_compatibly();
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
    // Module preprocessing must retain exact kernel-based minimization, even
    // when ordinary column reduction misses a relation at a join degree.
    Mod module(P);
    module.minimize();
    assert(module.num_of_gens() == 3 && module.num_of_rels() == 2);
    for (const r2degree d : {r2degree{-1,-1}, {0,1}, {1,0}, {1,1}, {2,2}})
        assert(module.dimension_at(d) == Mod(P).dimension_at(d));
    auto parent = std::make_shared<Mod>(Mat(0, 3, {}, {}, P.row_degrees));
    Submodule<Mat> S(parent, P);
    S.minimize_generators();
    assert(S.num_of_gens() == 2);
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
    assert(resolution.is_chain_complex());
    Mod M(resolution);
    M.minimize();
    assert(M.proj_res_.size() == 3 && M.proj_res_.is_chain_complex());
    assert(M.num_of_gens() == 1 && M.num_of_rels() == 0);
    assert(M.proj_res_[1].get_num_cols() == 0);
    assert(M.proj_res_[2].get_num_cols() == 0);
    Mod presentation_only(resolution);
    presentation_only.minimize_presentation();
    assert(presentation_only.proj_res_.size() == 1);
    Mod editable(resolution, resolution); // storage compatibility, exactness trusted
    editable.remove_relations({0});
    assert(editable.proj_res_.size() == 1 && editable.inj_res_.empty());

    Mat triangle(3, 3, {{0,2}, {1,2}, {0,1}}, {{0,1}, {1,0}, {1,1}},
                 {{-1,-1}, {-1,-1}, {-1,-1}});
    Mat syzygy(1, 3, {{0,1,2}}, {{1,1}}, triangle.col_degrees);
    Mod T(ChainComplex<Mat>({triangle, syzygy}));
    T.minimize();
    assert(T.num_of_rels() == 2 && T.proj_res_[1].get_num_cols() == 0);
    assert(T.proj_res_.is_chain_complex());

    // Repeated grades with distinct basis elements must be permuted identically.
    Mat fold(3, 1, {{0}, {0}, {0}}, {{2,2}, {0,0}, {0,0}}, {{0,0}});
    Mat boundary(2, 3, {{0,1}, {1,2}}, {{2,2}, {0,0}}, fold.col_degrees);
    ChainComplex<Mat> C({fold, boundary});
    C.sort_compatibly();
    assert(C.is_chain_complex());
    assert(C[1].row_degrees == C[0].col_degrees);
    assert(C[1].data == array<int>({{0,1}, {0,2}}));
    C.sort_compatibly(Degree_traits<r2degree>::colex_lambda());
    assert(C.is_chain_complex());
    Mat stale(2, 1, {{0}, {0}}, {{1,1}, {2,2}}, {{0,0}});
    stale.sort_compatibly();
    stale.col_degrees[0] = {3,3};
    assert(stale.compatibly_sorted); // no hidden scans after direct degree edits
    stale.get_column_graph(); // trusts the flag
    stale.compatibly_sorted = false;
    rejects([&] { stale.get_column_graph(); });
    stale.sort_compatibly();
    Mat copied = stale;
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
    NoKernelMatrix<4> higher(1, 1, {{0}}, {CoordinateDegree<double, 4>{1,1,1,1}}, {CoordinateDegree<double, 4>{0,0,0,0}});
    rejects([&] { higher.minimize(); }); // deliberate kernel implementation frame
    higher.sort_compatibly();
    higher.semi_minimize();
    assert(higher.get_num_cols() == 1);
    auto complete_without_kernel = Module<NoKernelMatrix<4>>::from_projective_resolution(
        ChainComplex<NoKernelMatrix<4>>({higher}), ResolutionCompleteness::complete);
    complete_without_kernel.minimize_resolution();
    assert(complete_without_kernel.res_complete_ && complete_without_kernel.num_of_rels() == 1);
    Module<NoKernelMatrix<4>> injective(ChainComplex<NoKernelMatrix<4>>({higher}),
                                               ResolutionKind::injective);
    rejects([&] { injective.minimize_injective_resolution(); });
}

void test_chain_vs_resolution_minimization() {
    // A truncation needs one extra kernel map to expose terminal redundant
    // relations as cancellable pairs in the extended chain complex.
    // Its nonzero H1 is free on (1,1) in degree (1,0).
    Mat duplicate(2, 1, {{0}, {0}}, {{1,0}, {1,0}}, {{0,0}});
    ChainComplex<Mat> C({duplicate});
    C.minimize();
    assert(C[0].data == duplicate.data && C[0].col_degrees == duplicate.col_degrees);
    assert(homology_module(C, 1).dimension_at({1,0}) == 1);
    assert(homology_module(C, 1).dimension_at({0,0}) == 0);
    Mod presentation(C);
    presentation.minimize_resolution(); // explicit method also accepts one map
    assert(presentation.proj_res_.size() == 1 && presentation.num_of_rels() == 1);
    assert(!presentation.res_complete_);
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
    assert(M.proj_res_.size() == 2);
    assert(M.proj_res_[1].data == array<int>({{0,1}}));
    assert(M.proj_res_.is_chain_complex());
    assert(homology_module(M.proj_res_, 2).dimension_at({1,1}) == 0);
    assert(!M.res_complete_);
    assert(M.dimension_at({0,0}) == 1 && M.dimension_at({1,0}) == 0);

    // Completing the resolution exposes a contractible pair: ordinary chain
    // minimization then removes both a duplicate syzygy and its dependency.
    Mat dependency(1, 2, {{0,1}}, {{1,1}}, syzygies.col_degrees);
    ChainComplex<Mat> complete({square, syzygies, dependency});
    complete.minimize();
    assert(complete[1].data == array<int>({{0,1}}));
    assert(complete[2].get_num_cols() == 0 && complete.is_chain_complex());

    // A zero differential is homology, not a contractible summand. The custom
    // example also proves that chain minimization needs no graded_kernel.
    using Higher = NoKernelMatrix<4>;
    Higher local(2, 2, {{0}, {}}, {CoordinateDegree<double, 4>{0,0,0,0}, CoordinateDegree<double, 4>{1,1,1,1}},
                                {CoordinateDegree<double, 4>{0,0,0,0}, CoordinateDegree<double, 4>{0,0,0,0}});
    ChainComplex<Higher> general({local});
    general.minimize();
    assert(general[0].get_num_rows() == 1 && general[0].data == array<int>({{}}));
    assert((general[0].col_degrees == vec<CoordinateDegree<double, 4>>({CoordinateDegree<double, 4>{1,1,1,1}})));
    Module<Higher> unsupported(ChainComplex<Higher>({local}));
    rejects([&] { unsupported.minimize_resolution(); });
    assert(unsupported.presentation().data == local.data); // rejected before minimization

    ChainComplex<Mat> empty;
    empty.minimize();
    assert(empty.empty());
    Mat unit(1, 1, {{0}}, {{0,0}}, {{0,0}});
    ChainComplex<Mat> invalid({unit, unit});
    assert(!invalid.is_chain_complex());
    assert(invalid[0].data == unit.data && invalid[1].data == unit.data);
    duplicate.col_degrees = {{2,0}, {1,0}};
    ChainComplex<Mat> unsorted({duplicate});
    unsorted.minimize(false); // chain cancellation does not require degree ordering
    assert(unsorted[0].col_degrees == duplicate.col_degrees);
}

void test_chain_minimization_does_not_copy() {
    using Matrix = CopyCountingMatrix;
    for (bool sort : {false, true}) for (bool track : {false, true}) {
        Matrix lower(2, 1, {{0}, {}}, {{0,0}, {1,1}}, {{0,0}});
        Matrix upper(1, 2, {{1}}, {{1,1}}, lower.col_degrees);
        ChainComplex<Matrix> complex({lower, upper});
        auto maps = identity_basis_maps(complex);
        Matrix::copies = 0;
        complex.minimize(sort, track ? &maps : nullptr);
        assert(Matrix::copies == 0);
        assert(complex[0].get_num_rows() == 0 && complex[0].get_num_cols() == 0);
        assert(complex[1].get_num_rows() == 0 && complex[1].get_num_cols() == 0);

        Module<Matrix> module(ChainComplex<Matrix>({lower, upper}));
        module.res_complete_ = true;
        Matrix::copies = 0;
        Matrix::kernel_calls = 0;
        module.minimize_resolution(sort);
        assert(Matrix::copies == 0 && Matrix::kernel_calls == 0);
        assert(module.proj_res_[0].get_num_rows() == 0 && module.proj_res_[0].get_num_cols() == 0);
        assert(module.proj_res_[1].get_num_rows() == 0 && module.proj_res_[1].get_num_cols() == 0);

        Module<Matrix> truncated(ChainComplex<Matrix>({lower, upper}));
        Matrix::copies = 0;
        Matrix::kernel_calls = 0;
        truncated.minimize_resolution(sort);
        assert(Matrix::copies == 1 && Matrix::kernel_calls == 1);
        assert(truncated.proj_res_.size() == 2 && !truncated.res_complete_);
        assert(truncated.proj_res_[0].get_num_rows() == 0 && truncated.proj_res_[0].get_num_cols() == 0);
        assert(truncated.proj_res_[1].get_num_rows() == 0 && truncated.proj_res_[1].get_num_cols() == 0);
    }
}

template<class Matrix>
void test_batched_chain_minimization() {
    using D = base_degree<Matrix>;
    using Complex = ChainComplex<Matrix>;
    auto check = [](const Complex& original, bool sort) {
        std::stringstream before, after;
        original.to_stream(before);
        auto reduced = original;
        for (auto& d : reduced.differentials()) d.compute_rows_forward();
        auto maps = identity_basis_maps(original);
        reduced.minimize(sort, &maps);
        assert(reduced.is_chain_complex());
        for (std::size_t level = 0; level < reduced.size(); ++level) {
            const auto& d = reduced[level];
            assert(d.is_graded_matrix() && d.rows_computed);
            for (int c = 0; c < d.get_num_cols(); ++c)
                for (int r : d.data[c]) assert(!Degree_traits<D>::equals(d.col_degrees[c], d.row_degrees[r]));
            if (level) assert(d.row_degrees == reduced[level - 1].col_degrees);
            auto inclusion_left = original[level] * maps.to_original[level + 1];
            auto inclusion_right = maps.to_original[level] * d;
            assert(inclusion_left.data == inclusion_right.data);
            auto projection_left = d * maps.from_original[level + 1];
            auto projection_right = maps.from_original[level] * original[level];
            assert(projection_left.data == projection_right.data);
        }
        for (std::size_t group = 0; group <= reduced.size(); ++group) {
            const auto& degrees = group ? reduced[group - 1].col_degrees : reduced[0].row_degrees;
            const auto& old_degrees = group ? original[group - 1].col_degrees : original[0].row_degrees;
            const auto& inclusion = maps.to_original[group];
            const auto& projection = maps.from_original[group];
            assert(inclusion.is_graded_matrix() && projection.is_graded_matrix());
            assert(inclusion.col_degrees == degrees && inclusion.row_degrees == old_degrees);
            assert(projection.col_degrees == old_degrees && projection.row_degrees == degrees);
            auto identity = projection * inclusion;
            assert(identity.get_num_cols() == degrees.size() && identity.get_num_rows() == degrees.size());
            for (int c = 0; c < identity.get_num_cols(); ++c) assert(identity.get_col(c) == vec<int>{c});
        }
        // Independent binary elimination checks homology at every grid point.
        auto rank_at = [](const Matrix& d, D point) {
            std::uint64_t pivots[8]{};
            int rank = 0;
            for (int c = 0; c < d.get_num_cols(); ++c) {
                if (!Degree_traits<D>::smaller_equal(d.col_degrees[c], point)) continue;
                std::uint64_t column = 0;
                for (int r : d.data[c]) column ^= std::uint64_t{1} << r;
                for (int r = 7; r >= 0; --r) if (column & (std::uint64_t{1} << r)) {
                    if (pivots[r]) column ^= pivots[r];
                    else { pivots[r] = column; ++rank; break; }
                }
            }
            return rank;
        };
        auto betti_at = [&](const Complex& complex, std::size_t group, D point) {
            const auto& degrees = group ? complex[group - 1].col_degrees : complex[0].row_degrees;
            int dimension = 0;
            for (const auto& degree : degrees) dimension += Degree_traits<D>::smaller_equal(degree, point);
            if (group) dimension -= rank_at(complex[group - 1], point);
            if (group < complex.size()) dimension -= rank_at(complex[group], point);
            return dimension;
        };
        for (int x = -1; x <= 3; ++x) for (int y = -1; y <= 3; ++y)
            for (std::size_t group = 0; group <= reduced.size(); ++group)
                assert(betti_at(original, group, D{double(x), double(y)}) ==
                       betti_at(reduced, group, D{double(x), double(y)}));
        auto untracked = original;
        untracked.minimize(sort);
        auto once = reduced;
        auto once_maps = maps;
        reduced.minimize(false, &maps);
        for (std::size_t level = 0; level < reduced.size(); ++level) {
            assert(reduced[level].data == untracked[level].data && reduced[level].data == once[level].data);
            assert(reduced[level].col_degrees == once[level].col_degrees);
            assert(reduced[level].row_degrees == once[level].row_degrees);
        }
        for (std::size_t group = 0; group <= reduced.size(); ++group) {
            assert(maps.to_original[group].data == once_maps.to_original[group].data);
            assert(maps.from_original[group].data == once_maps.from_original[group].data);
        }
        original.to_stream(after);
        assert(before.str() == after.str());
        return maps;
    };
    // The second reducer uses the first, while the third column reduces to a
    // surviving cycle. Its inclusion must be computed once, not twice.
    Matrix lower(4, 3, {{0,2}, {1,2}, {0,1}, {0}},
                 {D{1,1}, D{1,1}, D{1,1}, D{2,2}}, {D{0,0}, D{1,1}, D{1,1}});
    Matrix upper(1, 4, {{0,1,2}}, {D{2,2}}, lower.col_degrees);
    Complex dependency({lower, upper});
    const auto maps = check(dependency, false);
    assert(maps.to_original[1].get_col(0) == vec<int>({0,1,2}));

    vec<D> degrees[] = {
        {D{0,0}, D{1,0}, D{0,0}, D{0,1}, D{1,1}, D{0,1}, D{2,2}, D{1,0}},
        {D{0,0}, D{1,0}, D{1,1}, D{0,1}, D{1,1}, D{0,1}, D{1,1}, D{1,0}},
        {D{1,1}, D{0,1}, D{2,2}, D{1,0}, D{0,0}, D{1,0}, D{0,0}, D{1,1}},
        {D{0,0}, D{1,0}, D{0,1}, D{1,1}, D{1,1}, D{0,1}, D{2,2}, D{1,0}}
    };
    std::mt19937 random(4719);
    for (int trial = 0; trial < 80; ++trial) {
        std::vector<Matrix> differentials;
        for (int level = 0; level < 3; ++level) {
            array<int> columns(8);
            for (int c = 0; c < 4; ++c) columns[c] = {c + (level ? 4 : 0)};
            differentials.emplace_back(8, 8, columns, degrees[level + 1], degrees[level]);
        }
        Complex complex(std::move(differentials));
        // Admissible basis changes mix contractible pairs with strict
        // differentials and zero cycles, updating both adjacent maps.
        for (int change = 0; change < 100; ++change) {
            int group = random() % 4, source = random() % 8, target = random() % 8;
            if (source == target || !Degree_traits<D>::smaller_equal(degrees[group][source], degrees[group][target])) continue;
            if (group) complex[group - 1].col_op(source, target);
            if (group < 3) complex[group].row_op_on_cols(target, source);
        }
        assert(complex.is_chain_complex());
        check(complex, false);
        check(complex, true);
        complex.sort_compatibly(Degree_traits<D>::colex_lambda());
        check(complex, false);
    }
}

void test_scc() {
    std::stringstream zero("scc2020\n2\n0 0 0\n");
    Mod Z(zero);
    assert(Z.has_presentation() && Z.num_of_gens() == 0 && Z.dimension_at({0,0}) == 0);
    std::stringstream encoded;
    Z.to_stream(encoded);
    Mod round_trip(encoded);
    assert(round_trip.has_presentation() && round_trip.num_of_rels() == 0);
    rejects([] { std::stringstream s("scc2020\n2\n1 1 rubbish\n"); Mod wrong(s); });
    rejects([] { std::stringstream s("scc2020\n2\n1 1 0\n1 1 ; 0 garbage\n0 0 ;\n"); Mod wrong(s); });
    rejects([] { std::stringstream s("scc2020\n3\n1 1 0\n1 1 ; 0\n0 0 ;\n");
        ChainComplex<NoKernelMatrix<3>> wrong(s); });
}

void test_resolution_degree_operations() {
    Mat d1(2, 1, {{0}, {0}}, {{0, 2}, {2, 0}}, {{0, 0}});
    Mat d2(1, 2, {{0, 1}}, {{2, 2}}, d1.col_degrees);
    ChainComplex<Mat> projective({d1, d2});
    ChainComplex<Mat> injective = projective;
    for (auto& differential : injective.differentials()) differential.shift({-4, -4});
    const vec<double> axis{0, 3, 4, 7};

    for (bool complete : {false, true}) {
        for (int operation = 0; operation < 4; ++operation) {
            Mod module(projective, injective);
            module.res_complete_ = complete;
            switch (operation) {
                case 0: module.snap_to_grid(axis, axis); break;
                case 1: module.snap_to_equidistant_grid(2); break;
                case 2: module.bound_support({1, 1}); break;
                case 3: module.cut_above(0.5, 0.5); break;
            }
            if (operation == 2) {
                assert(module.proj_res_.size() == 1 && module.inj_res_.empty());
                assert(!module.res_complete_);
                assert(module.presentation().col_degrees == vec<r2degree>({{0, 1}, {1, 0}}));
                assert(module.dimension_at({0, 0}) == 1);
                assert(module.dimension_at({1, 0}) == 0 && module.dimension_at({0, 1}) == 0);
                module.compute_projective_resolution();
                assert(module.res_complete_ && module.proj_res_.size() == 2);
                assert(module.proj_res_.is_chain_complex());
                assert(module.proj_res_[1].col_degrees == vec<r2degree>({{1, 1}}));
                continue;
            }
            assert(module.proj_res_.size() == 2 && module.inj_res_.size() == 2);
            assert(module.res_complete_ == complete);
            if (operation == 0) {
                assert(module.proj_res_[0].col_degrees == vec<r2degree>({{0, 3}, {3, 0}}));
                assert(module.proj_res_[1].col_degrees == vec<r2degree>({{3, 3}}));
                assert(module.inj_res_[0].col_degrees == vec<r2degree>({{4, 7}, {7, 4}}));
                assert(module.inj_res_[1].col_degrees == vec<r2degree>({{7, 7}}));
            } else {
                assert(module.proj_res_[1].get_num_rows() == 0);
                assert(module.proj_res_[1].get_num_cols() == 0);
                assert(module.inj_res_[1].get_num_rows() == 0);
                assert(module.inj_res_[1].get_num_cols() == 0);
                assert(module.proj_res_[0].get_num_cols() == 0);
                assert(module.inj_res_[0].get_num_cols() == 0);
                assert(module.inj_res_[0].row_degrees == vec<r2degree>({{4, 4}}));
            }
        }
    }

    Mod injective_only = Mod::from_injective_resolution(injective);
    injective_only.snap_to_equidistant_grid(2, true);
    assert(injective_only.proj_res_.empty() && injective_only.inj_res_.size() == 2);
    assert(injective_only.inj_res_[0].col_degrees == injective[0].col_degrees);

    using Grid = GridGradedSparseMatrix<float, 3, int>;
    Grid grid(1, 1, {{0}}, {{1, 1, 1}}, {{0, 0, 0}});
    Module<Grid> higher(ChainComplex<Grid>({grid}), ChainComplex<Grid>({grid}));
    const std::array<vec<float>, 3> axes{{{0, 2}, {0, 2}, {0, 2}}};
    higher.snap_to_grid(axes);
    assert(higher.proj_res_[0].real_col_degrees() == vec<Grid::GridD>({{2, 2, 2}}));
    assert(higher.inj_res_[0].real_col_degrees() == vec<Grid::GridD>({{2, 2, 2}}));
    higher.cut_above(std::array<double, 3>{0.5, 0.5, 0.5});
    assert(higher.proj_res_[0].get_num_cols() == 0 && higher.inj_res_[0].get_num_cols() == 0);

    // Integral physical coordinates must not be interpreted as grid indices.
    using IntegralGrid = GridGradedSparseMatrix<int, 2, int>;
    IntegralGrid free(0, 1, {}, {}, {{0, 0}});
    Module<IntegralGrid> integral(ChainComplex<IntegralGrid>({free}), ChainComplex<IntegralGrid>({free}));
    integral.bound_support({2, 3});
    const vec<IntegralGrid::GridD> deaths{{0, 3}, {2, 0}};
    assert(integral.proj_res_[0].real_col_degrees() == deaths);
    assert(integral.inj_res_.empty() && !integral.res_complete_);
}

void test_presentation_adapters() {
    // Deliberately unsorted generators: the local row map must retain original indices.
    Mod module(Mat(2, 3, {{1, 2}, {0}}, {{2, 2}, {4, 4}}, {{3, 0}, {0, 0}, {1, 1}}));
    auto [local, rows] = module.local_presentation_at({2, 2});
    assert(rows == vec<int>({1, 2}));
    assert(local.get_num_rows() == 2 && local.data == array<int>({{0, 1}}));
    vec<int> relations;
    auto available = module.presentation().map_at_degree({2, 2}, relations);
    assert(relations == vec<int>({0}));
    assert(available.get_num_rows() == 3 && available.data == array<int>({{1, 2}}));
    auto [before_birth, no_rows] = module.local_presentation_at({-1, -1});
    assert(no_rows.empty() && before_birth.get_num_rows() == 0);

    auto bounds = module.bounding_box();
    assert((bounds.first == r2degree{0, 0} && bounds.second == r2degree{4, 4}));
    assert(module.get_equidistant_grid(2) == vec<r2degree>({{0,0}, {0,4}, {4,0}, {4,4}}));
    assert(module.get_equidistant_grid(1) == vec<r2degree>({{0,0}}));
    assert(module.get_equidistant_grid(0).empty());
    rejects([&] { module.get_equidistant_grid(-1); });
    Mod empty(Mat(0, 0));
    rejects([&] { empty.bounding_box(); });
    assert(empty.get_equidistant_grid(3).empty());

    module.compute_projective_resolution();
    module.inj_res_ = module.proj_res_;
    module.remove_relations({1, 0, 1});
    assert(module.num_of_rels() == 0 && module.dimension_at({4,4}) == 3);
    assert(module.proj_res_.size() == 1);
    assert(module.inj_res_.empty()); // No relations now: a complete free presentation.
    module.add_relation({1, 2}, {2, 2});
    assert(!module.res_complete_);
    assert(module.dimension_at({2,2}) == 1);
    rejects([&] { module.remove_relations({-1}); });
    rejects([&] { module.quotient_by_generators({3}); });
    assert(module.num_of_rels() == 1);
    module.quotient_by_generators({2, 0, 2});
    assert(module.num_of_gens() == 1);
    assert(module.presentation().row_degrees == vec<r2degree>({{0,0}}));
    assert(module.dimension_at({1,1}) == 1 && module.dimension_at({2,2}) == 0);

    Mod tail(Mat(1, 3, {{0, 2}}, {{3,3}}, {{0,0}, {1,1}, {2,2}}));
    tail.quotient_by_tail_generators(2);
    assert(tail.num_of_gens() == 2 && tail.presentation().data == array<int>({{0}}));
    rejects([&] { tail.quotient_by_tail_generators(3); });
    rejects([&] { tail.quotient_by_tail_generators(-1); });
    tail.quotient_by_tail_generators(0);
    assert(tail.dimension_at({5,5}) == 0);

    Mod nonminimal(Mat(2, 2, {{0}, {1}}, {{2,2}, {0,0}}, {{2,2}, {0,0}}));
    assert(!nonminimal.is_presentation_minimal());
    assert(nonminimal.num_of_gens() == 2); // Query did not mutate.
    nonminimal.semi_minimize_presentation();
    assert(nonminimal.num_of_gens() == 0 && nonminimal.is_presentation_minimal());
    // Partial cancellation is also available without a graded-kernel implementation.
    Module<NoKernelMatrix<3>> three(
        NoKernelMatrix<3>(1, 1, {{0}}, {{1,1,1}}, {{1,1,1}}));
    three.semi_minimize_presentation();
    assert(three.num_of_gens() == 0);

    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    std::as_const(module).print_presentation();
    std::as_const(module).print_degrees();
    std::cout.rdbuf(previous);
    assert(output.str().find("Generators at:") != std::string::npos);
}

void test_generated_fibre_and_quiver() {
    Ptr parent = std::make_shared<Mod>(Mat(1, 2, {{0,1}}, {{2,2}}, {{1,1}, {0,0}}));
    auto sub = submodule_generated_at(parent, r2degree{1,1});
    assert(sub.parent() == parent);
    assert(sub.generator_map().generator_lift().row_degrees == parent->presentation().row_degrees);
    auto presented = sub.presented_module();
    assert(presented.dimension_at({0,0}) == 0);
    assert(presented.dimension_at({1,1}) == 2 && presented.dimension_at({2,2}) == 1);
    assert(submodule_generated_at(parent, r2degree{-1,-1}).is_zero());
    assert(submodule_generated_at(parent, r2degree{2,2}).number_of_embedding_generators() == 1);
    rejects([] { submodule_generated_at<Mat>(nullptr, {0,0}); });

    auto quiver = parent->to_quiver();
    assert(quiver.degrees == vec<r2degree>({{0,0}, {1,1}, {2,2}}));
    assert(quiver.dimensionVector == vec<int>({1,2,1}));
    assert(quiver.edges.size() == 2 && quiver.matrices.size() == 2);
    auto path = quiver.matrices[1] * quiver.matrices[0];
    auto direct = parent->to_quiver({{0,0}, {2,2}}, {{1}, {}});
    assert(path.equals(direct.matrices[0]));
    assert(path.data == array<int>({{0}})); // The surviving generator maps nontrivially.
    auto discrete = parent->to_quiver({{0,0}, {2,2}}, {{}, {}});
    assert(discrete.edges.empty());
    rejects([&] { parent->to_quiver({{0,0}}, {{1}}); });
    rejects([&] { parent->to_quiver({{0,0}, {2,2}}, {{}, {0}}); });
    rejects([&] { parent->to_quiver({{0,0}}, {{}, {}}); });
    assert(Mod(Mat(0,0)).to_quiver().degrees.empty());
    assert(parent->presentation().row_degrees == vec<r2degree>({{1,1}, {0,0}}));
}

void test_minimization_preserves_injective_storage() {
    // Distinct storage fixture: exactness is trusted by the framework. This
    // regression checks that projective basis changes never touch this model.
    ChainComplex<Mat> injective({Mat(1, 2, {{0,1}}, {{4,4}}, {{3,3}, {3,3}}),
                                Mat(0, 1, {}, {}, {{4,4}})});
    auto unchanged = [&](const Mod& module) {
        assert(!module.inj_res_.empty());
        const auto& actual = module.inj_res_;
        assert(actual.size() == injective.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            assert(actual[i].data == injective[i].data);
            assert(actual[i].row_degrees == injective[i].row_degrees);
            assert(actual[i].col_degrees == injective[i].col_degrees);
        }
    };
    Mat P(3, 2, {{0}, {1}, {1}}, {{0,0}, {2,2}, {3,3}}, {{0,0}, {0,0}});
    for (int operation = 0; operation < 5; ++operation) {
        Mod module(P);
        if (operation == 4) module.compute_projective_resolution();
        module.inj_res_ = injective;
        const auto* storage = module.inj_res_[0].data.data();
        if (operation == 0) module.minimize_presentation();
        if (operation == 1) module.semi_minimize_presentation();
        if (operation == 2) module.presentation().remove_redundant_relations();
        if (operation >= 3) module.minimize();
        unchanged(module);
        assert(module.inj_res_[0].data.data() == storage);
        assert(operation == 2 ? module.num_of_rels() < 3 : module.num_of_gens() == 1);
    }
    auto parent = std::make_shared<Mod>(P);
    parent->inj_res_ = injective;
    for (bool full : {false, true}) {
        auto S = Submodule<Mat>::whole(parent);
        S.inj_res_ = injective;
        if (full) S.minimize_parent(); else S.lazy_minimize_parent();
        unchanged(*S.parent());
        unchanged(S);
        unchanged(*parent);
        unchanged(std::as_const(S).presented_module(true));
        S.compute_presentation(true);
        unchanged(S);
        S.minimize_generators();
        unchanged(S);
        S.shift_generators({1,1});
        assert(S.inj_res_.empty());
    }
    auto zero = Submodule<Mat>::zero(parent);
    zero.inj_res_ = injective;
    zero.compute_presentation(true);
    unchanged(zero);
    auto quotient = Submodule<Mat>::whole(parent).submodule_quotient(Submodule<Mat>::whole(parent));
    assert(quotient.parent()->inj_res_.empty());
    Mod edited(P);
    edited.inj_res_ = injective;
    edited.add_relation({1}, {1,1});
    assert(edited.inj_res_.empty());
}

int main() {
    test_resolution_degree_operations();
    test_minimization_preserves_injective_storage();
    test_presentation_adapters();
    test_generated_fibre_and_quiver();
    test_homomorphisms();
    test_kernel_minimization();
    test_resolution_minimization_and_sorting();
    test_chain_vs_resolution_minimization();
    test_chain_minimization_does_not_copy();
    test_batched_chain_minimization<Mat>();
    test_batched_chain_minimization<R2GradedSparseMatrix<int, CSCStorage<int>>>();
    test_scc();
}
