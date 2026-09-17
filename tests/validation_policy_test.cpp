#include <grlina/modules.hpp>
#include <sstream>
#include <stdexcept>

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
    matrix.refresh_compatible_sorted(compare); // explicit: always performs a check
    comparisons = 0;
    expect(matrix.compatible_sorting_is_verified());
    expect(EXPECT_GRLINA_CHECKS ? comparisons > 0 : comparisons == 0);

    // An explicit refresh still detects stale metadata in optimized builds.
    std::swap(matrix.row_degrees[0], matrix.row_degrees[2]);
    expect(!matrix.refresh_compatible_sorted());
    matrix.sort_compatibly();
    expect(matrix.compatible_sorting_is_verified());
    matrix.shift({1,1});
    expect(matrix.compatible_sorting_is_verified());
    matrix.validate();
}

void explicit_validation_and_input_boundaries() {
    Mat malformed(1, 1, {{0}}, {{0,0}}, {{0,0}});
    malformed.data[0] = {2};
    rejects([&] { malformed.validate(); });
    rejects([&] { ChainComplex<Mat> checked(std::vector<Mat>{malformed}, true); });
    ChainComplex<Mat> unchecked(std::vector<Mat>{malformed}, false);
    rejects([&] { unchecked.validate_structure(); });
    if constexpr (EXPECT_GRLINA_CHECKS) {
        rejects([&] { ChainComplex<Mat> implicit(std::vector<Mat>{malformed}); });
    } else {
        ChainComplex<Mat> implicit(std::vector<Mat>{malformed});
        expect(implicit.size() == 1);
    }
    // Invalid grading and invalid sparse indices must still be rejected by I/O.
    for (const auto* input : {
            "scc2020\n2\n1 1 0\n0 0 ; 0\n1 1 ;\n",
            "scc2020\n2\n1 1 0\n1 1 ; 2\n0 0 ;\n"}) {
        rejects([&] { std::stringstream stream(input); Mod module(stream); });
    }
}

void same_algebra_in_both_modes() {
    // One cancellable pair plus an interval [0,2) along the diagonal.
    Mat presentation(2, 2, {{0}, {1}}, {{1,1}, {2,2}}, {{1,1}, {0,0}});
    Mod module(presentation);
    module.minimize();
    expect(module.number_of_generators() == 1 && module.number_of_relations() == 1);
    expect(module.dimension_at({0,0}) == 1 && module.dimension_at({2,2}) == 0);
    module.compute_projective_resolution();
    expect(module.projective_resolution().squares_to_zero());

    auto parent = std::make_shared<Mod>(Mat(0, 2, {}, {}, {{0,0}, {1,1}}));
    auto whole = parent->whole_submodule();
    auto zero = parent->zero_submodule();
    expect(whole.contains(zero) && !zero.contains(whole));
    whole.compute_presentation();
    expect(whole.dimension_at({1,1}) == 2);
    auto maps = End_2d_0(parent, {1,1});
    expect(maps.size() == 1 && maps[0].check_lifts());
    expect(maps[0].generator_lift().data == array<int>({{1}, {}}));
    auto canonical = Homomorphism<Mat>::canonical_shift(parent, {1,1});
    canonical.validate();
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

int main() {
    check_policy_and_sorting_cost();
    explicit_validation_and_input_boundaries();
    same_algebra_in_both_modes();
}
