// These tests intentionally exercise expensive CSC edits. Diagnostics are
// checked separately; acknowledge their cost here to keep algebra tests quiet.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/mat_csc.hpp>
#include <grlina/mod_modules.hpp>
#include <cassert>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;
using Base = CSCMatrix<int>;
using Mat = R2GradedSparseMatrix<int, CSCStorage<int>>;
using Mod = R2Module<int, CSCStorage<int>>;

static_assert(is_graded_sparse_matrix_v<Mat>);
static_assert(std::is_base_of_v<Base, Mat>);
static_assert(std::is_base_of_v<SparseMatrix<int>, R2GradedSparseMatrix<int>>);
static_assert(std::is_same_v<decltype(std::declval<const Mod&>().presentation()), const Mat&>);
static_assert(std::is_same_v<decltype(std::declval<Mat>().graded_kernel()), Mat>);

void storage_and_const_io() {
    vec<std::size_t> offsets{0, 2, 2, 3};
    vec<int> entries{0, 2, 1};
    const auto* original_entries = entries.data();
    Base native(3, 3);
    native.assign_data(CSCStorage<int>(std::move(offsets), std::move(entries)));
    assert(native.data.entries().data() == original_entries);
    Mat graded(std::move(native));
    const Base& coefficients = graded;
    assert(&coefficients.data == &graded.data);
    assert(coefficients.get_col(0) == vec<int>({0, 2}));

    auto shifted = graded;
    assert(shifted.data.shares_storage_with(graded.data));
    shifted.shift({1, 2});
    assert(shifted.data.shares_storage_with(graded.data));
    assert((graded.row_degrees[0] == r2degree{0, 0}));
    assert(shifted.row_degrees[0] != graded.row_degrees[0]);
    assert((Compare_by_degrees<r2degree, int, Mat>::compare_three_way(graded, shifted) == 1));
    shifted.col_op(0, 1);
    assert(!shifted.data.shares_storage_with(graded.data));
    assert(graded.get_col(1).empty());
    assert(shifted.get_col(1) == vec<int>({0, 2}));

    std::stringstream direct_scc;
    graded.to_stream(direct_scc);
    const Mat parsed(direct_scc);
    assert(parsed.data == graded.data);
    assert(parsed.data.offsets() == vec<std::size_t>({0, 2, 2, 3}));

    const Mod module(parsed);
    const Mod copy = module;
    assert(module.presentation().data.shares_storage_with(copy.presentation().data));
    assert(module.number_of_entries() == 3);
    assert(module.number_of_generators() == 3);
    std::stringstream module_scc;
    module.to_stream(module_scc);
    const Mod reloaded(module_scc);
    assert(reloaded.presentation().data == module.presentation().data);
    assert(reloaded.presentation().col_degrees == module.presentation().col_degrees);
    assert(reloaded.presentation().row_degrees == module.presentation().row_degrees);
}

void shared_graded_algorithms() {
    Mat original(2, 1, {{0}, {0}}, {{1, 0}, {0, 1}}, {{0, 0}});
    Mat identity(2, 2, "Identity");
    identity.row_degrees = identity.col_degrees = original.col_degrees;
    const auto product = original * identity;
    assert(product.data == original.data);
    assert(product.row_degrees == original.row_degrees);
    assert(product.col_degrees == original.col_degrees);
    auto source = original;
    source.sort_compatibly();
    auto kernel = source.graded_kernel();
    assert(kernel.get_num_cols() == 1);
    assert(kernel.get_col(0) == vec<int>({0, 1}));

    Mod module(original);
    assert(module.dimension_at({0, 0}) == 1);
    assert(module.dimension_at({1, 1}) == 0);
    auto fibre = module.local_presentation_at({1, 1});
    static_assert(std::is_same_v<decltype(fibre.first), Base>);
    vec<int> which;
    auto relations = module.relations_at({1, 1}, which);
    assert(relations.get_num_cols() == 2);
    module.compute_projective_resolution();
    std::stringstream stream;
    module.to_stream(stream);
    Mod roundtrip(stream);
    assert(roundtrip.dimension_at({0, 0}) == 1);

    auto transpose = original.transposed_copy();
    static_assert(std::is_same_v<decltype(transpose), Mat>);
    assert(transpose.get_col(0) == vec<int>({0, 1}));
    auto sorted = original;
    sorted.sort_columns_colexicographically();
    sorted.sort_columns_colexicographically_with_output();
    sorted.sort_columns_lexicographically_with_output();
    sorted.sort_rows_colexicographically();
    auto copy = sorted;
    sorted.append_move_matrix(std::move(copy));
    assert(sorted.get_num_cols() == 4);
    sorted.delete_all_but_columns({1, 3});
    assert(sorted.get_num_cols() == 2);
    sorted.delete_all_but_columns_alt({0});
    assert(sorted.get_num_cols() == 1);

    Mat contractible(1, 1, {{0}}, {{0, 0}}, {{0, 0}});
    ChainComplex<Mat> chain(std::vector<Mat>{contractible});
    chain.minimize();
    assert(chain[0].get_num_cols() == 0);
    Mat cancellable(3, 3, {{0, 1}, {1, 2}, {0, 1, 2}},
                    {{1, 1}, {1, 1}, {2, 2}}, {{0, 0}, {1, 1}, {1, 1}});
    cancellable.sort_compatibly();
    const auto unchanged = cancellable;
    Mat elements(2, 3, {{2}, {1, 2}}, {{3, 3}, {2, 2}}, cancellable.row_degrees);
    const auto degrees = elements.col_degrees;
    cancellable.cancel_local_pairs(&elements);
    assert((cancellable.get_col(0) == vec<int>{0}));
    assert(cancellable.get_num_cols() == 1 && cancellable.get_num_rows() == 1);
    assert((elements.get_col(0) == vec<int>{0}) && elements.get_col(1).empty());
    assert(elements.col_degrees == degrees && elements.row_degrees == cancellable.row_degrees);
    assert(unchanged.get_num_rows() == 3 && (unchanged.get_col(0) == vec<int>{0, 1}));
    auto quiver = original.induced_quiver_rep({{0, 0}, {1, 1}});
    std::stringstream quiver_text;
    quiver.to_stream_simple(quiver_text);
    assert(!quiver_text.str().empty());
}

void module_constructions() {
    using Hom = Homomorphism<Mat>;
    using Ptr = std::shared_ptr<const Mod>;
    Ptr free = std::make_shared<Mod>(Mat(0, 1, {}, {}, {{0, 0}}));
    Ptr interval = std::make_shared<Mod>(Mat(1, 1, {{0}}, {{1, 1}}, {{0, 0}}));
    Mat identity(1, 1, {{0}}, {{0, 0}}, {{0, 0}});
    Hom quotient(free, interval, identity);
    assert(quotient.check_lifts());
    assert(module_hom_space_basis<Mat>(free, interval).size() == 1);
    auto sum = direct_sum<Mat>(free, interval);
    assert(sum.module->dimension_at({0, 0}) == 2);
    assert(sum.module->dimension_at({1, 1}) == 1);
    assert(sum.inclusion_left.check_lifts() && sum.projection_right.check_lifts());
    auto twice = quotient + quotient;
    assert(twice.generator_lift().get_col(0).empty());
    assert(twice.check_lifts());
    auto k = kernel(quotient);
    assert(k.inclusion.check_lifts());
    assert(k.module->dimension_at({0, 0}) == 0);
    assert(k.module->dimension_at({1, 1}) == 1);
}

int main() {
    storage_and_const_io();
    shared_graded_algorithms();
    module_constructions();
}
