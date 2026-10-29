#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/graded_linalg.hpp>
#include <cassert>

using namespace graded_linalg;

template<class Matrix>
void check_representatives(const Module<Matrix>& module, const Matrix& original_relations) {
    assert(module.cycles);
    const auto& cycles = *module.cycles;
    assert(cycles.get_num_cols() == module.num_of_gens());
    assert(geometric_col_degrees(cycles) == geometric_row_degrees(module.presentation()));
    assert(cycles.is_graded_matrix());
    // Relations among the new representatives must still be old boundaries.
    assert(graded_linear_system_is_solvable(original_relations, cycles * module.presentation()));
}

template<class Matrix>
void presentation_cycles() {
    using D = typename Matrix::degree_type;
    Matrix relations(3, 3, {{0,1}, {0,1,2}, {0,1,2}},
                     {D{1,1}, D{2,2}, D{2,2}}, {D{0,0}, D{1,1}, D{0,0}});
    const ChainComplex<Matrix> complex({relations});
    auto raw = homology_module(complex, 0, false, true);
    assert(!homology_module(complex, 0, false).cycles);
    assert(!homology_module(complex, 0).cycles);
    check_representatives(raw, relations);

    for (int method = 0; method < 7; ++method) {
        auto module = raw;
        Matrix incoming(geometric_row_degrees(relations));
        Matrix outgoing = incoming;
        // The tracked map is deliberately nonidentity, with repeated columns.
        Matrix map(3, 3, {{0,1}, {0,1,2}, {0,1,2}},
                   {D{2,2}, D{2,2}, D{2,2}}, geometric_row_degrees(relations));
        Matrix saved_map = map;
        if (method == 0) module.minimize(true, &map, {&outgoing});
        if (method == 1) module.semi_minimize_presentation(&map, {&outgoing});
        if (method == 2) module.minimize_resolution(true, &map, {&outgoing});
        if (method == 3) {
            module.compute_projective_resolution();
            module.minimize_resolution(true, &map, {&outgoing});
        }
        if (method == 4) {
            module.sort_compatibly(TraitLinearOrder<base_degree<Matrix>>{
                Degree_traits<base_degree<Matrix>>::colex_lambda()}, &map, {&outgoing});
            module.minimize_presentation(false, &map, {&outgoing});
        }
        if (method >= 5) {
            auto& p = module.presentation();
            p.sort_compatibly(&map, {&outgoing, &*module.cycles});
            if (method == 5) p.minimize(&map, {&outgoing, &*module.cycles});
            else p.minimize_variant(&map, {&outgoing, &*module.cycles});
        }
        check_representatives(module, relations);
        assert(((*raw.cycles) * outgoing).data == module.cycles->data);
        Matrix difference = outgoing * map;
        for (int j = 0; j < difference.get_num_cols(); ++j)
            difference.add_to_col(j, saved_map.get_col(j));
        assert(graded_linear_system_is_solvable(relations, difference));
        assert(module.num_of_gens() == 2);
        assert(module.cycles->get_col(0) == vec<int>{0});
        assert(module.cycles->get_col(1) == vec<int>{2});
        for (const D& point : {D{0,0}, D{1,1}, D{2,2}})
            assert(module.dimension_at(point) == raw.dimension_at(point));
    }
    auto minimized = homology_module(complex, 0, true, true);
    assert(minimized.num_of_gens() == 2 && minimized.num_of_rels() == 1);
    check_representatives(minimized, relations);
    Matrix unit(1, 1, {{0}}, {D{0,0}}, {D{0,0}});
    auto zero = homology_module(ChainComplex<Matrix>({unit}), 0, true, true);
    assert(zero.cycles && zero.cycles->get_num_cols() == 0);
    assert(zero.cycles->get_num_rows() == 1);
    assert(homology_module(ChainComplex<Matrix>{}, 0, true, true).cycles);

    // A submodule uses parent cycles through its inclusion, never another cache.
    auto parent = std::make_shared<const Module<Matrix>>(raw);
    Matrix generators(1, 3, {{1}}, {D{1,1}}, geometric_row_degrees(relations));
    Submodule<Matrix> submodule(parent, generators);
    submodule.minimize_parent();
    assert(!submodule.cycles && submodule.parent()->cycles);
    auto representatives = *submodule.parent()->cycles * submodule.generator_map().generator_lift();
    Matrix difference = representatives;
    difference.add_to_col(0, ((*raw.cycles) * generators).get_col(0));
    assert(graded_linear_system_is_solvable(relations, difference));
}

template<class Matrix>
void degree_and_generator_edits() {
    using D = typename Matrix::degree_type;
    Matrix free(0, 3, {}, {}, {D{2,2}, D{0,0}, D{1,1}});
    const auto original = homology_module(ChainComplex<Matrix>({free}), 0, false, true);
    for (int operation = 0; operation < 6; ++operation) {
        auto module = original;
        if (operation == 0) module.quotient_by_generators({0});
        if (operation == 1) module.quotient_by_tail_generators(2);
        if (operation == 2) module.cut_above(0.5, 0.5);
        if (operation == 3) module.snap_to_grid(vec<double>{0,2,4}, vec<double>{0,2,4});
        if (operation == 4) module.bound_support(D{1,1});
        if (operation == 5) module.snap_to_equidistant_grid(3, true);
        assert(module.cycles->get_num_cols() == module.num_of_gens());
        assert(geometric_col_degrees(*module.cycles) == geometric_row_degrees(module.presentation()));
        assert(module.cycles->is_graded_matrix());
        assert(module.cycles->get_num_rows() == 3);
        if (operation == 0 || operation == 2)
            assert(module.cycles->get_col(0) == vec<int>{1});
        if (operation == 3) {
            assert(module.cycles->get_col(0) == vec<int>{1});
            assert(module.cycles->get_col(1) == vec<int>{0});
            assert(module.cycles->get_col(2) == vec<int>{2});
        }
    }
    auto shifted = original;
    shifted.shift(D{1,1});
    assert(geometric_col_degrees(*shifted.cycles) == geometric_row_degrees(shifted.presentation()));
    auto before = shifted.cycles->data;
    shifted.add_relation({0}, D{3,3});
    shifted.remove_relations({0});
    shifted.compute_projective_resolution();
    assert(shifted.cycles->data == before);
}

template<class Matrix>
void nonfree_cycles() {
    using D = typename Matrix::degree_type;
    Matrix outgoing(3, 1, {{0},{0},{0}}, {D{1,0,0}, D{0,1,0}, D{0,0,1}}, {D{0,0,0}});
    Matrix incoming(1, 3, {{0,1}}, {D{1,1,0}}, geometric_col_degrees(outgoing));
    ChainComplex<Matrix> complex({outgoing, incoming});
    auto raw = homology_module(complex, 1, false, true);
    assert(raw.num_of_gens() == 3 && raw.num_of_rels() == 2);
    auto module = homology_module(complex, 1, true, true);
    assert(module.num_of_gens() == 2 && module.num_of_rels() == 1);
    assert((outgoing * *module.cycles).is_zero());
    assert(graded_linear_system_is_solvable(incoming, *module.cycles * module.presentation()));
    assert(module.dimension_at(D{1,1,1}) == 1);
    assert(module.dimension_at(D{1,1,0}) == 0);
    module.compute_projective_resolution();
    module.minimize_resolution();
    assert((outgoing * *module.cycles).is_zero());
    assert(geometric_col_degrees(*module.cycles) == geometric_row_degrees(module.presentation()));
}

int main() {
    presentation_cycles<R2GradedSparseMatrix<int>>();
    presentation_cycles<R2GradedSparseMatrix<int, CSCStorage<int>>>();
    presentation_cycles<CoordGradedSparseMatrix<double, 2, int>>();
    presentation_cycles<GridGradedSparseMatrix<double, 2, int>>();
    degree_and_generator_edits<R2GradedSparseMatrix<int>>();
    degree_and_generator_edits<CoordGradedSparseMatrix<double, 2, int>>();
    degree_and_generator_edits<GridGradedSparseMatrix<double, 2, int>>();
    nonfree_cycles<CoordGradedSparseMatrix<double, 3, int>>();
    nonfree_cycles<GridGradedSparseMatrix<double, 3, int>>();
}
