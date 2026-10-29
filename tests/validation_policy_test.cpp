#include <grlina/invariants/hilbert_function.hpp>
#include <grlina/module/epsilon.hpp>
#include <grlina/module/submodule.hpp>
#include <grlina/hom_interface.hpp>
#include <grlina/matrices/grid.hpp>
#include <sstream>
#include <stdexcept>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/quiver_representations.hpp>

using namespace graded_linalg;
using Mat = R2GradedSparseMatrix<int>;
using Mod = Module<Mat>;

static_assert(GRLINA_ENABLE_CHECKS == EXPECT_GRLINA_CHECKS);

// Deliberately independent of both NDEBUG and the library policy.
void expect(bool condition) {
    if (!condition) throw std::runtime_error("Validation policy regression");
}
template <typename Action> void rejects(Action action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    expect(threw);
}

void check_policy_and_sorting_cost() {
    int evaluated = 0;
    GRLINA_DEBUG_CHECK(++evaluated);
    expect(evaluated == EXPECT_GRLINA_CHECKS);
    GRLINA_ASSERT(++evaluated > 0);
    expect(evaluated == 2 * EXPECT_GRLINA_CHECKS);

    Mat matrix(0, 3, {}, {}, {{0,0}, {1,1}, {2,2}});
    int comparisons = 0;
    auto compare = [&](const r2degree& a, const r2degree& b) {
        ++comparisons;
        return Degree_traits<r2degree>::lex_lambda()(a, b);
    };
    expect(!matrix.compatibly_sorted); // constructors do not scan supplied degrees
    matrix.sort_compatibly(compare);
    comparisons = 0;
    expect(matrix.compatibly_sorted);
    matrix.get_row_graph();
    expect(comparisons == 0);
    std::swap(matrix.row_degrees[0], matrix.row_degrees[2]);
    expect(matrix.compatibly_sorted); // direct edits leave flag maintenance to the caller
    matrix.get_row_graph();
    expect(comparisons == 0);
    matrix.compatibly_sorted = false;
    if constexpr (EXPECT_GRLINA_CHECKS) {
        rejects([&] { matrix.get_row_graph(); });
    } else {
        matrix.get_row_graph();
    }
    matrix.sort_compatibly();
    matrix.shift({1,1});
    expect(matrix.compatibly_sorted);
}

void trusted_data_and_input_boundaries() {
    Mat malformed(1, 1, {{2}}, {{0,0}}, {{0,0}});
    ChainComplex<Mat> unchecked(std::vector<Mat>{malformed});
    expect(unchecked.size() == 1); // no constructor validation in either mode
    // File loading parses data without a full grading/chain-structure scan.
    std::stringstream ungraded("scc2020\n2\n1 1 0\n0 0 ; 0\n1 1 ;\n");
    Mod loaded(ungraded);
    expect(loaded.presentation().data == array<int>{{0}});
    rejects([&] {
        std::stringstream stream("scc2020\n2\n1 1 0\n1 1 ; 2\n0 0 ;\n");
        Mod module(stream);
    });
    Mat identity(1, 1, {{0}}, {{0,0}}, {{0,0}});
    expect(!ChainComplex<Mat>({identity, identity}).is_chain_complex());
    Mat zero(1, 1, {{}}, {{0,0}}, {{0,0}});
    expect(ChainComplex<Mat>({identity, zero}).is_chain_complex());
    expect(ChainComplex<Mat>().is_chain_complex());
}

void same_algebra_in_both_modes() {
    // One cancellable pair plus an interval [0,2) along the diagonal.
    Mat presentation(2, 2, {{0}, {1}}, {{1,1}, {2,2}}, {{1,1}, {0,0}});
    Mod module(presentation);
    module.minimize();
    expect(module.num_of_gens() == 1 && module.num_of_rels() == 1);
    expect(module.dimension_at({0,0}) == 1 && module.dimension_at({2,2}) == 0);
    module.compute_projective_resolution();
    expect(module.proj_res_.is_chain_complex());

    auto parent = std::make_shared<Mod>(Mat(0, 2, {}, {}, {{0,0}, {1,1}}));
    auto whole = parent->whole_submodule();
    auto zero = parent->zero_submodule();
    expect(whole.contains(zero) && !zero.contains(whole));
    // Parent identity is required even when diagnostic checks are disabled.
    auto foreign_parent = std::make_shared<Mod>(parent->presentation());
    auto foreign_zero = foreign_parent->zero_submodule();
    rejects([&] { (void)whole.contains(foreign_zero); });
    rejects([&] { (void)whole.equals(foreign_zero); });
    whole.compute_presentation();
    expect(whole.dimension_at({1,1}) == 2);
    auto maps = End_2d_0(parent, {1,1});
    expect(maps.size() == 1 && maps[0].check_lifts());
    expect(maps[0].generator_lift().data == array<int>({{1}, {}}));
    auto canonical = Homomorphism<Mat>::canonical_shift(parent, {1,1});
    expect(canonical.check_lifts());
    auto sum = zero.sum(parent->whole_submodule());
    expect(sum.equals(parent->whole_submodule()));

    Mat a(1, 2, {{0,1}}, {{1,1}}, {{0,0}, {0,0}});
    auto solution = solve_graded_linear_system(a, a);
    expect(solution.has_value() && solution->data == array<int>({{0}}));
    std::stringstream encoded;
    module.to_stream(encoded);
    Mod decoded(encoded);
    expect(decoded.dimension_at({0,0}) == 1 && decoded.dimension_at({2,2}) == 0);
}

void caller_preconditions_follow_policy() {
    Mat a(0, 1, {}, {}, {{0,0}}), b(0, 1, {}, {}, {{1,1}});
    if constexpr (EXPECT_GRLINA_CHECKS) {
        rejects([&] { (void)solve_graded_linear_system(a, b); });
        rejects([&] { (void)graded_linear_system_is_solvable(a, b); });
        rejects([&] { (void)lift_to_relations(a, b, a); });
    } else {
        // Empty maps have no coefficients to access, so these deliberate
        // precondition violations exercise disabled checks without invalid reads.
        expect(solve_graded_linear_system(a, b).has_value());
        expect(graded_linear_system_is_solvable(a, b));
        expect(lift_to_relations(a, b, a).has_value());
    }
    ChainComplex<Mat> chain({a});
    chain.push_differential(a);
    expect(chain.size() == 2);
    expect(epsilon_image<Mat>(std::make_shared<Mod>(a), -1).generator_map().generator_lift().get_num_cols() == 1);
    // Inconsistency is an algebraic result, not a caller precondition.
    SparseMatrix<int> inconsistent(1, 1, array<int>{{}});
    vec<int> rhs{0};
    rejects([&] { inconsistent.column_gauss_solver(rhs); });
    expect(rhs == vec<int>{0});
}

void restored_adapters_work_in_both_modes() {
    using Coord = CoordGradedSparseMatrix<double, 2, int>;
    using Grid = GridGradedSparseMatrix<double, 2, int>;
    Grid matrix(Coord(1, 1, {{0}}, {{3,5}}, {{1,2}}));
    auto owner = std::make_shared<const Module<Grid>>(matrix);
    expect(owner->dimension_at({1,2}) == 1 && owner->dimension_at({3,5}) == 0);
    auto image = epsilon_image(owner, 0.5);
    expect(image.generator_map().generator_lift().real_col_degrees() == vec<r2degree>{{1.5,2.5}});
    std::stringstream stream;
    ChainComplex<Grid>({matrix}).to_stream(stream);
    auto decoded = ChainComplex<Grid>::from_stream(stream);
    expect(decoded[0].real_col_degrees() == matrix.real_col_degrees());
    expect(decoded[0].real_row_degrees() == matrix.real_row_degrees());
    auto grid = (*owner).hilbert_function({1,3}, {2,5});
    expect(grid.values == array<int>({{1,1}, {1,0}}));

    auto unsorted = std::make_shared<const Mod>(
        Mat(2, 1, {{0}, {0}}, {{2,2}, {1,1}}, {{0,0}}));
    Mod changed = *unsorted;
    Mat incoming(changed.presentation().row_degrees), outgoing = incoming;
    changed.sort_compatibly(&incoming, {&outgoing});
    changed.minimize(true, &incoming, {&outgoing});
    expect(changed.num_of_rels() == 1);
    auto owner_changed = std::make_shared<const Mod>(changed);
    expect(Homomorphism<Mat>(unsorted, owner_changed, incoming).check_lifts());
    expect(Homomorphism<Mat>(owner_changed, unsorted, outgoing).check_lifts());

}

int main() {
    check_policy_and_sorting_cost();
    trusted_data_and_input_boundaries();
    same_algebra_in_both_modes();
    caller_preconditions_follow_policy();
    restored_adapters_work_in_both_modes();
}
