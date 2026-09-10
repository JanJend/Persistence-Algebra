#include <grlina/modules.hpp>

#include <cassert>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace graded_linalg;
using Matrix = R2GradedSparseMatrix<int>;
using TestModule = R2Module<int>;

struct NamedDegree {
    int rank = 0;
    bool operator==(const NamedDegree& other) const { return rank == other.rank; }
    friend std::ostream& operator<<(std::ostream& out, const NamedDegree& degree) {
        return out << degree.rank;
    }
};

namespace graded_linalg {

template <>
struct Degree_traits<NamedDegree> {
    inline static constexpr const char* poset_id = "named-poset";
    static bool equals(const NamedDegree& a, const NamedDegree& b) { return a == b; }
    static bool smaller_equal(const NamedDegree& a, const NamedDegree& b) { return a.rank <= b.rank; }
    static bool greater_equal(const NamedDegree& a, const NamedDegree& b) { return a.rank >= b.rank; }
    static bool smaller(const NamedDegree& a, const NamedDegree& b) { return a.rank < b.rank; }
    static bool greater(const NamedDegree& a, const NamedDegree& b) { return a.rank > b.rank; }
    static bool lex_order(const NamedDegree& a, const NamedDegree& b) { return a.rank < b.rank; }
    static std::function<bool(const NamedDegree&, const NamedDegree&)> lex_lambda() {
        return lex_order;
    }
    static vec<double> position(const NamedDegree& degree) { return {double(degree.rank)}; }
    static void print_degree(const NamedDegree& degree) { std::cout << degree.rank; }
    static NamedDegree join(const NamedDegree& a, const NamedDegree& b) {
        return {std::max(a.rank, b.rank)};
    }
    static NamedDegree meet(const NamedDegree& a, const NamedDegree& b) {
        return {std::min(a.rank, b.rank)};
    }
    template <typename OutputStream>
    static void write_degree(OutputStream& out, const NamedDegree& degree) { out << degree.rank; }
    template <typename InputStream>
    static NamedDegree from_stream(InputStream& in) {
        NamedDegree degree;
        in >> degree.rank;
        return degree;
    }
    static void add(const NamedDegree& amount, NamedDegree& degree) { degree.rank += amount.rank; }
    static void subtract(const NamedDegree& amount, NamedDegree& degree) { degree.rank -= amount.rank; }
};

template <typename index>
struct NamedGradedMatrix
    : GradedSparseMatrix<NamedDegree, index, NamedGradedMatrix<index>> {
    using Base = GradedSparseMatrix<NamedDegree, index, NamedGradedMatrix<index>>;
    using Base::Base;
    NamedGradedMatrix() = default;
    explicit NamedGradedMatrix(SparseMatrix<index>&& matrix) : Base(std::move(matrix)) {}
};

} // namespace graded_linalg

static Matrix interval_presentation() {
    Matrix result(1, 1);
    result.col_degrees = {{1.0, 1.0}};
    result.row_degrees = {{0.0, 0.0}};
    result.data = {{0}};
    return result;
}

static void test_sort_state_and_degree_traits() {
    Matrix matrix = interval_presentation();
    assert(!matrix.compatibly_sorted);
    matrix.sort_columns_lexicographically();
    assert(matrix.compatibly_sorted); // the one row was already sorted
    matrix.sort_compatibly();
    assert(matrix.compatibly_sorted);
    assert(std::string(Degree_traits<r2degree>::poset_id) == "2");
    assert(Degree_traits<r4degree>::poset_id == "4");
    assert(Degree_traits<z2degree>::poset_id == "2Z");
    assert(Degree_traits<z3degree>::poset_id == "3Z");
    assert(Degree_traits<z4degree>::poset_id == "4Z");
}

static void test_chain_complex_round_trip() {
    Matrix d1 = interval_presentation();
    Matrix d2(1, 1);
    d2.col_degrees = {{2.0, 2.0}};
    d2.row_degrees = d1.col_degrees;
    d2.data = {{}};
    ChainComplex<Matrix> complex({d1, d2});
    assert(complex.size() == 2);
    assert(complex.is_chain_complex());

    std::stringstream encoded;
    complex.to_stream(encoded);
    ChainComplex<Matrix> decoded(encoded);
    assert(decoded.size() == 2);
    assert(decoded[0].data == d1.data);
    assert(decoded[1].col_degrees == d2.col_degrees);
    assert(decoded.is_chain_complex());

    Matrix noncomplex_d1 = interval_presentation();
    Matrix noncomplex_d2(1, 1);
    noncomplex_d2.col_degrees = noncomplex_d1.col_degrees;
    noncomplex_d2.row_degrees = noncomplex_d1.col_degrees;
    noncomplex_d2.data = {{0}};
    ChainComplex<Matrix> noncomplex({noncomplex_d1, noncomplex_d2});
    assert(!noncomplex.is_chain_complex());
}

static void test_named_poset_io() {
    NamedGradedMatrix<int> matrix(1, 1);
    matrix.col_degrees = {{1}};
    matrix.row_degrees = {{0}};
    matrix.data = {{0}};
    std::stringstream encoded;
    ChainComplex<NamedGradedMatrix<int>>({matrix}).to_stream(encoded);
    std::string line;
    std::getline(encoded, line);
    std::getline(encoded, line);
    assert(line == "named-poset");
    encoded.clear();
    encoded.seekg(0);
    auto decoded = ChainComplex<NamedGradedMatrix<int>>::from_stream(encoded);
    assert(decoded[0].col_degrees == matrix.col_degrees);
}

static void test_existing_scc_files() {
    const auto base = std::filesystem::path(__FILE__).parent_path() / "../test_presentations";
    ChainComplex<Matrix> presentation((base / "toy_example_2.scc").string());
    assert(presentation.size() == 1);
    assert(presentation[0].get_num_rows() != 0);

    // This legacy file uses "3" as chain length on line two.  The new reader
    // accepts it and writes the canonical R^2 poset identifier "2".
    ChainComplex<Matrix> resolution(
        (base / "full_rips_size_1_instance_5_min_pres_resolution.scc").string());
    assert(resolution.size() == 2);
    std::stringstream canonical;
    resolution.to_stream(canonical);
    std::string header;
    std::getline(canonical, header);
    assert(header == "scc2020");
    std::getline(canonical, header);
    assert(header == "2");
}

static void test_module_hilbert_function_and_editing() {
    TestModule module(interval_presentation());
    assert(module.number_of_generators() == 1);
    assert(module.number_of_relations() == 1);
    assert(module.dimension_at({0.0, 0.0}) == 1);
    assert(module.dimension_at({1.0, 1.0}) == 0);
    auto grid = module.hilbert_function_on_induced_grid();
    assert(grid.x_grid.size() == 2);
    assert(grid.y_grid.size() == 2);
    assert(grid.maximum == 1);

    module.compute_projective_resolution();
    assert(module.projective_resolution().size() == 2);
    module.shift({1.0, 1.0});
    assert(module.projective_resolution().size() == 2);
    assert(module.presentation().row_degrees.front() == r2degree(-1.0, -1.0));
    (void)module.mutable_presentation();
    assert(module.projective_resolution().size() == 1);
}

static void test_submodules_and_morphisms() {
    auto module = std::make_shared<TestModule>(interval_presentation());
    auto zero = Submodule<Matrix>::zero(module);
    auto whole = Submodule<Matrix>::whole(module);
    assert(zero.is_zero());
    assert(whole.number_of_generators() == 1);
    auto sum = zero.sum(whole);
    assert(sum.number_of_generators() == 1);
    assert(whole.presented_module().dimension_at({0.0, 0.0}) == 1);
    assert(zero.quotient_module().dimension_at({0.0, 0.0}) == 1);
    assert(whole.quotient_module().dimension_at({0.0, 0.0}) == 0);

    Matrix identity(1, 1, "Identity");
    identity.row_degrees = module->presentation().row_degrees;
    identity.col_degrees = module->presentation().row_degrees;
    ModuleMorphism<Matrix> map(module, module, identity);
    assert(map.image().number_of_generators() == 1);
    assert(map.kernel().is_zero());
    auto composite = map.then(map);
    assert(composite.generator_lift().data == identity.data);

    auto other_parent = std::make_shared<TestModule>(interval_presentation());
    bool rejected_different_parent = false;
    try {
        (void)zero.sum(Submodule<Matrix>::zero(other_parent));
    } catch (const std::invalid_argument&) {
        rejected_different_parent = true;
    }
    assert(rejected_different_parent);

    auto resolved = std::make_shared<TestModule>(interval_presentation());
    resolved->compute_projective_resolution();
    Matrix relation_lift(1, 1, "Identity");
    relation_lift.row_degrees = resolved->presentation().col_degrees;
    relation_lift.col_degrees = resolved->presentation().col_degrees;
    ModuleMorphism<Matrix> lifted_map(resolved, resolved, {identity, relation_lift});
    assert(lifted_map.lifts().size() == 2);
}

static void test_module_hom_adapter() {
    auto module = std::make_shared<TestModule>(interval_presentation());
    module->sort_compatibly();
    auto endomorphisms = module_endomorphism_basis<Matrix>(module, true);
    assert(endomorphisms.size() == 1);
    assert(endomorphisms.front().domain().get() == module.get());
    assert(endomorphisms.front().target().get() == module.get());
}

static void test_r3_colex_sorting() {
    R3GradedSparseMatrix<int> matrix(2, 1);
    matrix.col_degrees = {{1, 0, 2}, {0, 1, 1}};
    matrix.row_degrees = {{0, 0, 0}};
    matrix.data = {{0}, {0}};
    matrix.sort_colexicographically();
    assert(matrix.compatibly_sorted);
    assert(matrix.col_degrees.front() == triple(0, 1, 1));
    R3Module<int> r3_module(matrix);
    r3_module.shift(triple(1, 1, 1));
    assert(r3_module.presentation().col_degrees.front() == triple(-1, 0, 0));
}

static void test_four_parameter_io() {
    R4GradedSparseMatrix<int> real_matrix(1, 1);
    real_matrix.col_degrees = {r4degree(1.0, 2.0, 3.0, 4.0)};
    real_matrix.row_degrees = {r4degree(0.0, 0.0, 0.0, 0.0)};
    real_matrix.data = {{0}};
    real_matrix.sort_colexicographically();
    assert(real_matrix.compatibly_sorted);
    std::stringstream real_scc;
    ChainComplex<R4GradedSparseMatrix<int>>({real_matrix}).to_stream(real_scc);
    auto real_round_trip = ChainComplex<R4GradedSparseMatrix<int>>::from_stream(real_scc);
    assert(real_round_trip[0].col_degrees == real_matrix.col_degrees);

    Z4GradedSparseMatrix<int> discrete_matrix(1, 1);
    discrete_matrix.col_degrees = {z4degree(1, 2, 3, 4)};
    discrete_matrix.row_degrees = {z4degree(0, 0, 0, 0)};
    discrete_matrix.data = {{0}};
    std::stringstream discrete_scc;
    ChainComplex<Z4GradedSparseMatrix<int>>({discrete_matrix}).to_stream(discrete_scc);
    auto discrete_round_trip = ChainComplex<Z4GradedSparseMatrix<int>>::from_stream(discrete_scc);
    assert(discrete_round_trip[0].row_degrees == discrete_matrix.row_degrees);

    Z3GradedSparseMatrix<int> z3_matrix(1, 1);
    z3_matrix.col_degrees = {z3degree(2, 3, 4)};
    z3_matrix.row_degrees = {z3degree(1, 1, 1)};
    z3_matrix.data = {{0}};
    z3_matrix.sort_colexicographically();
    assert(z3_matrix.compatibly_sorted);

    Z2GradedSparseMatrix<int> z2_matrix(1, 1);
    z2_matrix.col_degrees = {z2degree(2, 3)};
    z2_matrix.row_degrees = {z2degree(1, 1)};
    z2_matrix.data = {{0}};
    Z2Module<int> z2_module(z2_matrix);
    z2_module.shift(z2degree(1, 1));
    assert(z2_module.presentation().col_degrees.front() == z2degree(1, 2));
}

int main() {
    test_sort_state_and_degree_traits();
    test_chain_complex_round_trip();
    test_named_poset_io();
    test_existing_scc_files();
    test_module_hilbert_function_and_editing();
    test_submodules_and_morphisms();
    test_module_hom_adapter();
    test_r3_colex_sorting();
    test_four_parameter_io();
}
