/** Standalone AIDA integration check. Build with AIDA/include, AIDA/src and
 * grlina's include directory; link libaida, Boost.Timer and Boost.Chrono.
 */
#ifdef NDEBUG
#undef NDEBUG
#endif
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/module/module_base.hpp> // Also check this include order.
#include <aida_interface.hpp>
#include <cassert>

using namespace graded_linalg;

template<class Matrix>
void same_matrix(const Matrix& a, const Matrix& b) {
    assert(a.get_num_rows() == b.get_num_rows());
    assert(a.get_num_cols() == b.get_num_cols());
    assert(geometric_row_degrees(a) == geometric_row_degrees(b));
    assert(geometric_col_degrees(a) == geometric_col_degrees(b));
    for (matrix_index_t<Matrix> j = 0; j < a.get_num_cols(); ++j)
        assert(a.get_col(j) == b.get_col(j));
    if constexpr (matrix_grid_backed_v<Matrix>) assert(a.grid_ == b.grid_);
}

template<class Matrix>
void check_summands(const Module<Matrix>& module, const Matrix& boundaries,
                    const Matrix* outgoing = nullptr) {
    using Index = matrix_index_t<Matrix>;
    assert(module.decomposition);
    for (const auto& summand : *module.decomposition) {
        assert(summand.cycles);
        assert(summand.cycles->get_num_cols() == summand.num_of_gens());
        assert(geometric_col_degrees(*summand.cycles) == geometric_row_degrees(summand.presentation()));
        assert(summand.cycles->is_graded_matrix());
        assert(summand.presentation().is_minimal());
        assert(graded_linear_system_is_solvable(boundaries, *summand.cycles * summand.presentation()));
        if (outgoing) assert((*outgoing * *summand.cycles).is_zero());
    }
    // The cycle images span the parent modulo its original boundaries.
    Matrix generated = boundaries;
    for (const auto& summand : *module.decomposition)
        generated.append_matrix(*summand.cycles);
    assert(graded_linear_system_is_solvable(generated, *module.cycles));
    Matrix parent_generators = boundaries;
    parent_generators.append_matrix(*module.cycles);
    for (const auto& summand : *module.decomposition)
        assert(graded_linear_system_is_solvable(parent_generators, *summand.cycles));
    for (const auto& degree : {typename Matrix::degree_type{0,0},
                              typename Matrix::degree_type{1,2},
                              typename Matrix::degree_type{3,3}}) {
        Index dimension = 0;
        for (const auto& summand : *module.decomposition)
            dimension += summand.dimension_at(degree);
        assert(dimension == module.dimension_at(degree));
    }
}

template<class Matrix>
void triangular_decomposition() {
    using D = typename Matrix::degree_type;
    using Index = matrix_index_t<Matrix>;
    Matrix relations(3, 3, {{0,1,2}, {1,2}, {2}},
                     {D{0,3}, D{1,2}, D{2,1}}, {D{0,0}, D{0,0}, D{0,0}});
    if constexpr (matrix_grid_backed_v<Matrix>) relations.include_real_degrees({D{7,9}});
    auto module = homology_module(ChainComplex<Matrix>({relations}), 0, false, true);
    module.compute_projective_resolution();
    const auto original = module;
    assert(!module.decomposition);
    module.decompose();
    assert(module.decomposition->size() == 3);
    for (const auto& summand : *module.decomposition) assert(!summand.cycles);
    module.decompose(true);
    assert(module.decomposition->size() == 3);
    check_summands(module, relations);

    // AIDA records 0->2, 0->1, 1->2 here. Reversing the replay or swapping
    // the column-operation direction gives incorrect representatives.
    for (const auto& summand : *module.decomposition) {
        const auto degree = geometric_col_degrees(summand.presentation()).front();
        const Index column = static_cast<Index>(degree[0]);
        assert(summand.num_of_gens() == 1 && summand.num_of_rels() == 1);
        assert(summand.cycles->get_col(0) == relations.get_col(column));
        if constexpr (matrix_grid_backed_v<Matrix>) assert(summand.presentation().grid_ == relations.grid_);
    }
    same_matrix(*module.cycles, *original.cycles);
    assert(module.res_complete_ == original.res_complete_);
    assert(module.proj_res_.size() == original.proj_res_.size());
    for (std::size_t i = 0; i < module.proj_res_.size(); ++i)
        same_matrix(module.proj_res_[i], original.proj_res_[i]);

    // Basis changes preserve the cached summands, while edits to the module
    // invalidate them. Uniform shifts transport the summands and their cycles.
    module.sort_compatibly();
    module.minimize();
    assert(module.decomposition);
    module.shift(D{1,1});
    auto shifted_relations = relations;
    shifted_relations.shift(D{1,1});
    check_summands(module, shifted_relations);
    module.add_relation({0}, D{4,4});
    assert(!module.decomposition);

    Module<Matrix> no_cycles(relations);
    no_cycles.decompose(true);
    for (const auto& summand : *no_cycles.decomposition) assert(!summand.cycles);
    Module<Matrix> no_presentation;
    no_presentation.decompose(true);
    assert(!no_presentation.decomposition);
    Matrix unit(1, 1, {{0}}, {D{0,0}}, {D{0,0}});
    Module<Matrix> zero(unit);
    zero.decompose(true);
    assert(zero.decomposition && zero.decomposition->empty());

    Matrix free(0, 2, {}, {}, {D{2,2}, D{0,0}});
    auto free_module = homology_module(ChainComplex<Matrix>({free}), 0, false, true);
    free_module.decompose(true);
    assert(free_module.decomposition->size() == 2);
    assert(free_module.decomposition->front().cycles->get_col(0) == vec<Index>{1});
    assert(free_module.decomposition->back().cycles->get_col(0) == vec<Index>{0});

    // Sorting and cancelling a local pair must transport representatives before AIDA.
    Matrix nonminimal(3, 3, {{0,1,2}, {0,1}, {0,1,2}},
                      {D{2,2}, D{1,1}, D{2,2}}, {D{0,0}, D{1,1}, D{0,0}});
    auto minimized = homology_module(ChainComplex<Matrix>({nonminimal}), 0, false, true);
    minimized.decompose(true);
    check_summands(minimized, nonminimal);
}

void homology_cycles() {
    using M = CoordGradedSparseMatrix<double, 2, int>;
    M outgoing(4, 1, {{0},{0},{0},{0}}, {{0,0},{0,0},{0,0},{0,0}}, {{0,0}});
    // A nonidentity cycle basis followed by the same triangular relation basis.
    M cycles(3, 4, {{0,1}, {1,2}, {2,3}}, {{0,0},{0,0},{0,0}}, outgoing.col_degrees);
    M coordinates(3, 3, {{0,1,2}, {1,2}, {2}}, {{0,3},{1,2},{2,1}}, cycles.col_degrees);
    M boundaries = cycles * coordinates;
    auto module = homology_module(ChainComplex<M>({outgoing, boundaries}), 1, true, true);
    module.decompose(true);
    assert(module.decomposition->size() == 3);
    check_summands(module, boundaries, &outgoing);
}

int main() {
    triangular_decomposition<R2GradedSparseMatrix<int>>();
    triangular_decomposition<R2GradedSparseMatrix<int, CSCStorage<int>>>();
    triangular_decomposition<CoordGradedSparseMatrix<double, 2, long>>();
    triangular_decomposition<GridGradedSparseMatrix<long double, 2, long>>();
    triangular_decomposition<CoordGradedSparseMatrix<double, 3, long long>>();
    triangular_decomposition<GridGradedSparseMatrix<double, 3, long long>>();
    homology_cycles();
}
