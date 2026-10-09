// NEW: runtime coordinate degrees and their integration with graded modules.
// Exercise both storage backends without the diagnostics for costly CSC edits.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/csc_matrix.hpp>
#include <grlina/dynamic_coordinate_matrix.hpp>
#include <grlina/module.hpp>
#include <cassert>
#include <limits>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;

template <typename Action>
void rejects(Action action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    assert(threw);
}

template <typename Matrix>
void assert_columns(const Matrix& matrix, const array<int>& expected) {
    assert(matrix.get_num_cols() == static_cast<int>(expected.size()));
    for (int column = 0; column < matrix.get_num_cols(); ++column)
        assert(matrix.get_col(column) == expected[column]);
}

void degree_values_and_tables() {
    using Degree = DynamicDegree<double>;
    using Traits = Degree_traits<Degree>;
    using View = CoordinateDegreeView<double>;
    static_assert(!std::is_constructible_v<View, Degree&&>);
    static_assert(!std::is_constructible_v<View, const Degree&&>);
    static_assert(!std::is_constructible_v<View, vec<double>&&>);
    static_assert(!std::is_constructible_v<View, const vec<double>&&>);
    Degree a{1, 3}, b{2, 1};
    FlatDegreeTable<double> table(2, vec<Degree>{a, b});
    assert(!Traits::smaller_equal(a, b) && !Traits::smaller_equal(b, a));
    const Degree join = Traits::join(table[0], table[1]);
    const Degree meet = Traits::meet(table[0], table[1]);
    assert(join == Degree({2, 3}) && meet == Degree({1, 1}));
    assert(Traits::smaller_equal(a, join) && Traits::smaller_equal(meet, b));
    assert(Traits::lex_order(a, b) && !Traits::colex_order(a, b));
    // Appending a degree from the same table must survive buffer reallocation.
    table.push_back(table[0]);
    assert(table.coordinates() == vec<double>({1, 3, 2, 1, 1, 3}));
    table.set(1, table[0]);
    assert(table[1].to_degree() == a);
    table.set(0, Degree{0, 0});
    assert(join == Degree({2, 3}) && a == Degree({1, 3}));
    assert(Traits::join(Degree{}, Degree{}).empty());
    assert(Traits::meet(Degree{}, Degree{}).empty());
    assert(Traits::smaller_equal(Degree{}, Degree{}));
    assert(!Traits::lex_order(Degree{}, Degree{}));
    rejects([&] { Traits::smaller_equal(a, Degree{1, 3, 0}); });
    assert(FlatDegreeTable<double>(0, 2) != FlatDegreeTable<double>(0, 3));
}

void scalar_domains_and_precision() {
    using IntegerMatrix = DynamicCoordinateGradedSparseMatrix<int, long, CSCStorage<long>>;
    IntegerMatrix integer(1, 1, 3, {{0}}, {{1, 2, 3}}, {{0, 0, 0}});
    std::stringstream integer_text;
    integer.to_stream(integer_text);
    assert(integer_text.str().find("\n3Z\n") != std::string::npos);
    IntegerMatrix integer_roundtrip(integer_text);
    assert(integer_roundtrip.col_degree(0).to_degree() == DynamicDegree<int>({1, 2, 3}));
    assert(integer_roundtrip.get_col(0) == vec<long>({0}));
    IntegerMatrix overflow(1, 2, 1, {{0}}, {{1}}, {{0}, {std::numeric_limits<int>::lowest()}});
    rejects([&] { overflow.shift(DynamicDegree<int>{1}); });
    assert(overflow.col_degree(0).to_degree() == DynamicDegree<int>({1}));
    assert(overflow.row_degree(0).to_degree() == DynamicDegree<int>({0}));
    assert(overflow.row_degree(1)[0] == std::numeric_limits<int>::lowest());
    overflow.validate();

    using PreciseMatrix = DynamicCoordinateGradedSparseMatrix<long double, int>;
    const long double coordinate = 1.234567890123456789L;
    PreciseMatrix precise(0, 1, 1, {}, {}, {{coordinate}});
    Module<PreciseMatrix> module(precise);
    std::stringstream module_text;
    module.to_stream(module_text);
    Module<PreciseMatrix> module_roundtrip(module_text);
    assert(module_roundtrip.presentation().row_degree(0)[0] == coordinate);
    ChainComplex<PreciseMatrix> chain({precise});
    std::stringstream chain_text;
    chain.to_stream(chain_text);
    ChainComplex<PreciseMatrix> chain_roundtrip(chain_text);
    assert(chain_roundtrip[0].row_degree(0)[0] == coordinate);
}

template <typename Matrix>
void storage_and_runtime_dimensions() {
    using Degree = typename Matrix::degree_type;
    static_assert(is_graded_sparse_matrix_v<Matrix>);
    static_assert(has_matrix_graded_kernel<Matrix>::value);
    static_assert(std::is_same_v<typename Matrix::degree_type, DynamicDegree<double>>);
    Matrix two(2, 2, 2, {{0, 1}, {1}}, {{2, 1}, {2, 2}}, {{0, 0}, {1, 0}});
    Matrix three(0, 0, 3), five(0, 0, 5);
    assert(two.parameter_count() == 2 && three.parameter_count() == 3);
    assert(five.parameter_count() == 5);
    two.validate();
    assert(two.row_degrees.coordinates() == vec<double>({0, 0, 1, 0}));
    assert(two.col_degrees.coordinates() == vec<double>({2, 1, 2, 2}));
    assert(two.row_degree(1).data() == two.row_degrees.coordinates().data() + 2);
    const auto saved_degree = two.row_degree(0).to_degree();
    auto copy = two;
    copy.set_row_degree(0, Degree{-1, -2});
    assert(saved_degree == Degree({0, 0}));
    assert(two.row_degree(0).to_degree() == saved_degree);
    assert(copy.row_degree(0).to_degree() == Degree({-1, -2}));
    copy.shift(Degree{1, 2});
    assert(copy.row_degree(0).to_degree() == Degree({-2, -4}));
    assert(copy.col_degree(0).to_degree() == Degree({1, -1}));
    assert_columns(copy, {{0, 1}, {1}});
    rejects([&] { two.set_row_degree(0, Degree{0, 0, 0}); });
    rejects([&] { two.set_col_degree(0, Degree{std::numeric_limits<double>::infinity(), 0}); });
    rejects([&] { two.row_degree(2); });
    rejects([&] { two.col_degree(-1); });
    rejects([&] { three.shift(Degree{0, 0}); });

    // Empty tuples are distinct logical degrees even though their buffers are empty.
    Matrix zero_parameters(2, 2, 0, {{0}, {1}}, {Degree{}, Degree{}}, {Degree{}, Degree{}});
    zero_parameters.validate();
    assert(zero_parameters.row_degrees.size() == 2);
    assert(zero_parameters.col_degrees.size() == 2);
    assert(zero_parameters.row_degrees.coordinates().empty());
    assert(zero_parameters.dim_at(Degree{}) == 0);
    zero_parameters.sort_compatibly();
    zero_parameters.semi_minimize();
    assert(zero_parameters.get_num_cols() == 0 && zero_parameters.get_num_rows() == 0);
    assert(zero_parameters.parameter_count() == 0);
}

template <typename Matrix>
void permutations_and_edits() {
    using Degree = typename Matrix::degree_type;
    const Matrix original(3, 3, 2, {{0, 2}, {1}, {0, 1}},
                          {{3, 2}, {2, 1}, {2, 1}}, {{1, 0}, {0, 0}, {1, 0}});
    auto sorted = original;
    const auto rows = sorted.sort_rows_with_permutation();
    assert(rows.new_to_old == vec<int>({1, 0, 2}));
    assert(rows.old_to_new == vec<int>({1, 0, 2}));
    const auto columns = sorted.sort_columns_with_permutation();
    assert(columns.new_to_old == vec<int>({1, 2, 0}));
    assert(columns.old_to_new == vec<int>({2, 0, 1}));
    assert_columns(sorted, {{0}, {0, 1}, {1, 2}});
    assert(sorted.col_degree(0).to_degree() == Degree({2, 1}));
    assert(sorted.row_degree(0).to_degree() == Degree({0, 0}));
    sorted.validate();

    Matrix incomparable(2, 2, 2, {{0}, {1}}, {{2, 1}, {1, 2}}, {{1, 0}, {0, 1}});
    incomparable.sort_rows_lexicographically();
    incomparable.sort_columns_lexicographically();
    assert_columns(incomparable, {{0}, {1}});
    assert(incomparable.row_degree(0).to_degree() == Degree({0, 1}));
    incomparable.sort_rows_colexicographically();
    incomparable.sort_columns_colexicographically();
    assert_columns(incomparable, {{0}, {1}});
    assert(incomparable.row_degree(0).to_degree() == Degree({1, 0}));
    incomparable.sort_compatibly();
    incomparable.validate();

    vec<int> selected{2, 0};
    const auto restricted = original.restricted_domain_copy(selected);
    assert(restricted.parameter_count() == 2 && restricted.get_num_rows() == 3);
    assert_columns(restricted, {{0, 1}, {0, 2}});
    assert(restricted.col_degree(0).to_degree() == original.col_degree(2).to_degree());
    auto edited = original;
    vec<int> removed_rows{0}, removed_columns{1};
    edited.delete_rows(removed_rows);
    edited.delete_columns(removed_columns);
    assert_columns(edited, {{1}, {0}});
    assert(edited.row_degree(0).to_degree() == Degree({0, 0}));
    edited.append_column({0, 1}, Degree{4, 4});
    const auto first_copy = edited;
    edited.append_matrix(first_copy);
    assert_columns(edited, {{1}, {0}, {0, 1}, {1}, {0}, {0, 1}});
    assert(edited.parameter_count() == 2);
    edited.validate();
    rejects([&] { vec<int> duplicate{1, 1}; edited.delete_columns(duplicate); });
    rejects([&] { vec<int> unordered{1, 0}; edited.delete_rows(unordered); });
    rejects([&] { edited.append_column({2}, Degree{4, 4}); });
    rejects([&] { edited.append_column({0}, Degree{4, 4, 4}); });
    rejects([&] { edited.append_matrix(Matrix(0, 2, 3)); });
}

template <typename Matrix>
void algebra_and_local_queries() {
    using Degree = typename Matrix::degree_type;
    Matrix left(2, 2, 3, {{0}, {0, 1}}, {{1, 0, 0}, {0, 1, 0}},
                {{0, 0, 0}, {0, 0, 0}});
    Matrix right(1, 2, 3, {{0, 1}}, {{1, 1, 0}}, {{1, 0, 0}, {0, 1, 0}});
    const auto product = left * right;
    static_assert(std::is_same_v<std::decay_t<decltype(product)>, Matrix>);
    assert(product.parameter_count() == 3 && product.get_num_rows() == 2);
    assert_columns(product, {{1}});
    assert(product.row_degrees == left.row_degrees && product.col_degrees == right.col_degrees);
    product.validate();
    const auto transpose = left.transposed_copy();
    assert(transpose.parameter_count() == 3 && transpose.get_num_rows() == 2);
    assert_columns(transpose, {{0, 1}, {1}});
    assert(transpose.row_degrees == left.col_degrees && transpose.col_degrees == left.row_degrees);
    rejects([&] { Matrix(0, 0, 2) * Matrix(0, 0, 3); });
    rejects([&] { left * Matrix(1, 1, 3); });

    Matrix dynamic(2, 3, 2, {{1}, {1}}, {{1, 0}, {0, 1}}, {{0.5, 0.5}, {0, 0}, {0, 0}});
    R2GradedSparseMatrix<int> fixed(2, 3, {{1}, {1}}, {{1, 0}, {0, 1}},
                                   {{0.5, 0.5}, {0, 0}, {0, 0}});
    for (const r2degree query : vec<r2degree>{{-1, 0}, {0, 0}, {0.75, 0.75}, {1, 0}, {1, 1}}) {
        const Degree degree{query[0], query[1]};
        assert(dynamic.dim_at(degree) == fixed.dim_at(query));
    }
    const auto local = dynamic.map_at_degree_pair(Degree{1, 0});
    assert(local.second == vec<int>({1, 2}));
    assert(local.first.get_num_rows() == 2);
    assert_columns(local.first, {{0}});
    assert_columns(dynamic.map_at_degree_pair(Degree{1, 0}, false).first, {{1}});
    vec<int> selected;
    const auto relations = dynamic.map_at_degree(Degree{1, 0}, selected);
    assert(selected == vec<int>({0}));
    assert(relations.get_num_rows() == 3);
    assert_columns(relations, {{1}});
    rejects([&] { dynamic.map_at_degree_pair(Degree{1, 0, 0}); });

    Matrix local_pair(2, 2, 2, {{0}, {0, 1}}, {{0, 0}, {1, 1}}, {{0, 0}, {0, 0}});
    local_pair.sort_compatibly();
    local_pair.cancel_local_pairs();
    assert(local_pair.get_num_rows() == 1 && local_pair.get_num_cols() == 1);
    assert_columns(local_pair, {{0}});
    assert(local_pair.row_degree(0).to_degree() == Degree({0, 0}));
    assert(local_pair.col_degree(0).to_degree() == Degree({1, 1}));
    local_pair.minimize();
    local_pair.validate();
    assert(local_pair.get_num_rows() == 1 && local_pair.get_num_cols() == 1);
    assert_columns(local_pair, {{0}});
}

template <typename Matrix>
void module_and_chain_complex() {
    using Degree = typename Matrix::degree_type;
    using Mod = Module<Matrix>;
    Matrix presentation(2, 1, 2, {{0}, {0}}, {{1, 0}, {0, 1}}, {{0, 0}});
    Mod module(presentation);
    assert(module.dimension_at(Degree{0, 0}) == 1);
    assert(module.dimension_at(Degree{1, 1}) == 0);
    assert(module.support_degrees() == vec<Degree>({{0, 0}, {0, 1}, {1, 0}}));
    std::stringstream module_text;
    module.to_stream(module_text);
    Mod restored(module_text);
    assert(restored.presentation().parameter_count() == 2);
    assert(restored.dimension_at(Degree{0, 0}) == 1);
    restored.compute_projective_resolution();
    assert(restored.has_complete_projective_resolution());
    const auto hilbert_values = restored.hilbert_function({Degree{0, 0}, Degree{1, 1}});
    assert(hilbert_values[0].dimension == 1 && hilbert_values[1].dimension == 0);
    restored.add_relation({0}, Degree{0.5, 0.5});
    assert(restored.number_of_relations() == 3);
    assert(restored.dimension_at(Degree{0.75, 0.75}) == 0);
    rejects([&] { restored.add_relation({0}, Degree{1, 1, 1}); });
    rejects([&] { restored.dimension_at(Degree{1, 1, 1}); });
    const auto complete_empty = Mod::from_projective_resolution(
        ChainComplex<Matrix>({Matrix(0, 0, 3)}), ResolutionCompleteness::complete);
    assert(complete_empty.has_complete_projective_resolution());
    assert(complete_empty.dimension_at(Degree{0, 0, 0}) == 0);
    rejects([&] { complete_empty.dimension_at(Degree{0, 0}); });

    // The repeated degree in C1 must receive the same stable permutation in d1 and d2.
    Matrix d1(3, 2, 2, {{0}, {1}, {0}}, {{1, 0}, {0, 1}, {1, 0}}, {{0, 0}, {0, 0}});
    Matrix d2(1, 3, 2, {{0, 2}}, {{2, 2}}, {{1, 0}, {0, 1}, {1, 0}});
    ChainComplex<Matrix> chain({d1, d2});
    assert(chain.squares_to_zero());
    chain.sort_compatibly();
    assert_columns(chain[0], {{1}, {0}, {0}});
    assert_columns(chain[1], {{1, 2}});
    assert(chain.squares_to_zero());
    std::stringstream chain_text;
    chain.to_stream(chain_text);
    const ChainComplex<Matrix> roundtrip(chain_text);
    assert(roundtrip.size() == 2 && roundtrip.squares_to_zero());
    assert(roundtrip[0].data == chain[0].data && roundtrip[1].data == chain[1].data);
    assert(roundtrip[0].col_degrees == chain[0].col_degrees);

    Matrix contractible(1, 1, 2, {{0}}, {{0, 0}}, {{0, 0}});
    Matrix terminal_cycle(1, 1, 2, {{}}, {{1, 1}}, {{0, 0}});
    ChainComplex<Matrix> cancellation({contractible, terminal_cycle});
    cancellation.minimize();
    assert(cancellation[0].get_num_cols() == 0 && cancellation[0].get_num_rows() == 0);
    assert(cancellation[1].get_num_cols() == 1 && cancellation[1].get_num_rows() == 0);
    assert(cancellation[1].col_degree(0).to_degree() == Degree({1, 1}));
    assert(cancellation.squares_to_zero());

    ChainComplex<Matrix> empty_map({Matrix(0, 0, 5)});
    std::stringstream empty_text;
    empty_map.to_stream(empty_text);
    ChainComplex<Matrix> empty_roundtrip(empty_text);
    assert(empty_roundtrip.size() == 1 && empty_roundtrip[0].parameter_count() == 5);
    rejects([&] { ChainComplex<Matrix> mismatched({Matrix(0, 0, 2), Matrix(0, 0, 3)}); });
    ChainComplex<Matrix> no_differentials(std::size_t{5});
    assert(no_differentials.runtime_poset_identifier() == "5");
    no_differentials.push_differential(Matrix(0, 0, 5));
    no_differentials.clear();
    assert(no_differentials.runtime_poset_identifier() == "5");
    rejects([&] { no_differentials.push_differential(Matrix(0, 0, 2)); });
}

template <typename Matrix>
void three_parameter_kernels_and_resolution() {
    using Degree = typename Matrix::degree_type;
    Matrix presentation(3, 1, 3, {{0}, {0}, {0}},
                        {{0.25, -2, -3}, {-1, 0.5, -3}, {-1, -2, 1.75}},
                        {{-1, -2, -3}});
    auto source = presentation;
    auto second = source.graded_kernel();
    second.validate();
    assert(second.parameter_count() == 3 && second.get_num_cols() == 3);
    assert(second.row_degrees == presentation.col_degrees);
    assert((presentation * second).is_zero());
    const auto original_second = second;
    auto third = second.graded_kernel();
    third.validate();
    assert(third.get_num_cols() == 1);
    assert(third.col_degree(0).to_degree() == Degree({0.25, 0.5, 1.75}));
    assert((original_second * third).is_zero());
    auto terminal = third.graded_kernel();
    terminal.validate();
    assert(terminal.get_num_cols() == 0 && terminal.parameter_count() == 3);

    Module<Matrix> module(presentation);
    assert(module.dimension_at(Degree{-1, -2, -3}) == 1);
    assert(module.dimension_at(Degree{0.25, 0.5, 1.75}) == 0);
    module.compute_projective_resolution();
    assert(module.has_complete_projective_resolution());
    assert(module.projective_resolution().size() == 3);
    assert(module.projective_resolution().squares_to_zero());
    assert(module.dimension_at(Degree{-1, -2, -3}) == 1);
    assert(module.dimension_at(Degree{0.25, 0.5, 1.75}) == 0);
}

template <typename Matrix>
void checked_scc_input() {
    const Matrix original(1, 1, 3, {{0}}, {{1, 2, 3}}, {{0, 0, 0}});
    std::stringstream valid;
    original.to_stream(valid);
    const Matrix parsed(valid);
    assert(parsed.parameter_count() == 3 && parsed.row_degrees == original.row_degrees);
    assert_columns(parsed, {{0}});
    for (const auto* invalid : {
             "scc2020\n2Z\n0 0 0\n", // incompatible scalar domain
             "scc2020\n-2\n0 0 0\n", // invalid runtime dimension
             "scc2020\n3\n1 1 0\n1 2 ; 0\n0 0 0 ;\n", // missing coordinate
             "scc2020\n2\n1 1 0\n1 1 ; 1\n0 0 ;\n", // out-of-range row
             "scc2020\n2\n1 1 0\n1 1 ; 0 0\n0 0 ;\n", // duplicate row
             "scc2020\n2\n1 1 0\n-1 -1 ; 0\n0 0 ;\n"}) { // nonhomogeneous column
        rejects([&] { std::stringstream input(invalid); Matrix malformed(input); });
    }
    Matrix zero(1, 1, 0, {{0}}, {DynamicDegree<double>{}}, {DynamicDegree<double>{}});
    std::stringstream zero_text;
    zero.to_stream(zero_text);
    Matrix zero_roundtrip(zero_text);
    assert(zero_roundtrip.parameter_count() == 0 && zero_roundtrip.row_degrees.size() == 1);
    assert_columns(zero_roundtrip, {{0}});
}

template <typename Storage>
void suite() {
    using Matrix = DynamicCoordinateGradedSparseMatrix<double, int, Storage>;
    storage_and_runtime_dimensions<Matrix>();
    permutations_and_edits<Matrix>();
    algebra_and_local_queries<Matrix>();
    module_and_chain_complex<Matrix>();
    three_parameter_kernels_and_resolution<Matrix>();
    checked_scc_input<Matrix>();
}

int main() {
    degree_values_and_tables();
    suite<vec<vec<int>>>();
    suite<CSCStorage<int>>();
    scalar_domains_and_precision();
    using Matrix = DynamicCoordinateGradedSparseMatrix<double, int, CSCStorage<int>>;
    Matrix packed(2, 1, 3, {{0}, {0}}, {{1,0,0}, {0,1,0}}, {{0,0,0}});
    packed.sort_compatibly(); packed.compute_col_batches(); packed.compute_rows_forward();
    auto editable = packed.editable_copy();
    static_assert(std::is_same_v<typename decltype(editable)::storage_type, vec<vec<int>>>);
    assert(editable.parameter_count() == 3 && editable.col_degrees == packed.col_degrees);
    assert(editable.row_degrees == packed.row_degrees && editable.col_batches == packed.col_batches);
    assert(editable.rows_computed && editable._rows == packed._rows);
    auto restored = editable.to_storage<CSCStorage<int>>();
    restored.validate(); assert(restored.col_degrees == packed.col_degrees);
    editable.set_col(0, {}); assert(packed.get_col(0) == vec<int>({0}));
}
