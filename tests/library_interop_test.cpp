// NEW: packed exchange, ownership, basis transport and nonfree homology.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/packed_matrix.hpp>
#include <grlina/module_transformations.hpp>
#include <grlina/dynamic_coordinate_matrix.hpp>
#include <cassert>

using namespace graded_linalg;
using Degree = DynamicDegree<double>;

template<class Matrix>
Matrix matrix(int columns, int rows, array<int> entries,
              vec<Degree> column_degrees, vec<Degree> row_degrees, std::size_t parameters = 3) {
    Matrix result = GradedMatrixIO<Matrix>::make_matrix(columns, rows, parameters);
    GradedMatrixIO<Matrix>::assign_degrees(result, column_degrees, row_degrees);
    typename Matrix::storage_type storage;
    for (auto& column : entries) storage.push_back(std::move(column));
    result.assign_data(std::move(storage));
    result.refresh_compatible_sorted();
    return result;
}

template<class Matrix>
void check_change(const ModuleTransformation<Matrix>& change) {
    change.forward.validate();
    change.backward.validate();
    assert(change.forward.check_lifts());
    assert(change.backward.check_lifts());
    auto identity = change.backward.compose(change.forward);
    for (const auto& lift : identity.lifts())
        for (int i = 0; i < lift.get_num_cols(); ++i) assert(lift.get_col(i) == vec<int>{i});
    auto old_identity = change.forward.compose(change.backward);
    // After minimization the other composite is identity modulo relations.
    auto difference = old_identity + Homomorphism<Matrix>::identity(change.original);
    assert(difference.image(false).is_zero());
}

template<class Matrix>
void runtime_interop() {
    const Degree zero{0,0,0}, one{1,1,1}, two{2,2,2};
    // A unit needs both row and column operations; duplicate terminal
    // relations also test substitution in the returned relation lifts.
    auto d1 = matrix<Matrix>(3, 3, {{0,1}, {0,1,2}, {0,1,2}},
        {one,two,two}, {zero,one,zero});
    if constexpr (matrix_grid_backed_v<Matrix>) d1.include_real_degrees({{-0.5,0.5,2.5}});
    auto original = std::make_shared<const Module<Matrix>>(d1);
    auto sorted = sort_module_with_maps(original);
    check_change(sorted);
    auto change = minimize_module_with_maps(original);
    check_change(change);
    assert(change.original.get() == original.get());
    assert(original->presentation().data == d1.data);
    assert(change.transformed->number_of_generators() == 2);
    assert(change.transformed->number_of_relations() == 1);
    for (const auto& grade : {zero,one,two})
        assert(original->dimension_at(grade) == change.transformed->dimension_at(grade));

    auto packed = to_packed_csc(d1);
    auto restored = from_packed_csc<Matrix>(3,3,3,packed.data.offsets(),packed.data.entries(),
        packed.col_degrees.coordinates(),packed.row_degrees.coordinates(), [&] {
            if constexpr (matrix_grid_backed_v<Matrix>) return packed.grids;
            else return vec<vec<double>>{};
        }());
    assert(restored.data == d1.data);
    assert(detail::geometric_row_degrees(restored) == detail::geometric_row_degrees(d1));
    assert(detail::geometric_col_degrees(restored) == detail::geometric_col_degrees(d1));
    if constexpr (matrix_grid_backed_v<Matrix>) assert(restored.grids == d1.grids);

    // Export an inclusion from a stack object, then destroy that object.
    auto inclusion = [&] {
        auto submodule = Submodule<Matrix>::whole(original);
        return submodule.inclusion();
    }();
    inclusion.validate();
    assert(inclusion.check_lifts());
    auto shared_submodule = std::make_shared<Submodule<Matrix>>(Submodule<Matrix>::whole(original));
    auto shared_inclusion = shared_submodule->inclusion();
    assert(shared_inclusion.domain().get() == shared_submodule.get());
    auto const_inclusion = static_cast<const Submodule<Matrix>&>(*shared_submodule).inclusion();
    assert(const_inclusion.domain().get() == shared_submodule.get());
    shared_submodule.reset();
    assert(shared_inclusion.check_lifts());

    // The two-parameter kernel path is destructive on mutable input.
    auto two_parameter = matrix<Matrix>(2,1,{{0},{0}}, {{1,0},{0,1}}, {{0,0}}, 2);
    const auto two_parameter_saved = two_parameter;
    auto two_kernel = static_cast<const Matrix&>(two_parameter).graded_kernel();
    assert(two_kernel.get_num_cols() == 1);
    assert(two_parameter.data == two_parameter_saved.data);
    assert(two_parameter.col_degrees == two_parameter_saved.col_degrees);

    // Koszul first differential: three kernel generators with one relation.
    auto outgoing = matrix<Matrix>(3,1,{{0},{0},{0}},
        {{1,0,0},{0,1,0},{0,0,1}}, {zero});
    const auto saved = outgoing;
    auto kernel = static_cast<const Matrix&>(outgoing).graded_kernel();
    assert(outgoing.data == saved.data && outgoing.col_degrees == saved.col_degrees);
    ChainComplex<Matrix> complex({outgoing});
    auto homology = homology_with_cycles(complex,1,false);
    assert(homology.module->number_of_generators() == 3);
    assert(homology.module->number_of_relations() == 1);
    assert(homology.module->dimension_at(one) == 2);
    assert(homology.module->dimension_at({1,1,0}) == 1);
    assert((outgoing * homology.cycles).is_zero());
    auto incoming = matrix<Matrix>(1,3,{{0,1}}, {{1,1,0}},
        {{1,0,0},{0,1,0},{0,0,1}});
    ChainComplex<Matrix> with_boundary({outgoing,incoming});
    auto quotient = homology_with_cycles(with_boundary,1);
    assert(quotient.module->dimension_at(one) == 1);
    assert(quotient.module->dimension_at({1,1,0}) == 0);
    assert(quotient.cycles.get_num_cols() == quotient.module->number_of_generators());
    assert((outgoing * quotient.cycles).is_zero());
    // Identity chain map induces identity after choosing the returned bases.
    Matrix target_cycles = quotient.cycles;
    target_cycles.append_matrix(incoming);
    auto induced_lift = solve_graded_linear_system(target_cycles, quotient.cycles);
    assert(induced_lift);
    induced_lift->cull_columns(quotient.module->number_of_generators(), false);
    Homomorphism<Matrix> induced(quotient.module, quotient.module, std::move(*induced_lift));
    assert(induced.check_lifts());
    assert(homology_module(complex,0).dimension_at(zero) == 1);
    assert(homology_module(complex,0).dimension_at(one) == 0);

    // Every stored differential and the completeness marker survive conversion.
    Module<Matrix> resolved(outgoing);
    resolved.compute_projective_resolution();
    resolved.set_injective_resolution(with_boundary);
    auto converted = convert_module_storage<CSCStorage<int>>(resolved);
    assert(converted.has_complete_projective_resolution());
    assert(converted.projective_resolution().size() == resolved.projective_resolution().size());
    assert(converted.injective_resolution().size() == 2);
    auto converted_back = convert_module_storage<array<int>>(converted);
    assert(converted_back.projective_resolution()[0].data == outgoing.data);
    check_change(sort_module_with_maps(resolved));
    check_change(minimize_module_with_maps(resolved));

    // Equal-degree pairs in higher groups transport operations to both
    // neighboring differentials, rather than just changing the presentation.
    auto lower = matrix<Matrix>(3,1,{{0},{0},{0}}, {one,one,one}, {zero});
    auto middle = matrix<Matrix>(3,3,{{0,1},{1,2},{0,2}}, {one,one,two}, {one,one,one});
    auto upper = matrix<Matrix>(1,3,{{0,1,2}}, {two}, {one,one,two});
    auto contractible = Module<Matrix>::from_projective_resolution(
        ChainComplex<Matrix>({lower,middle,upper}), ResolutionCompleteness::complete);
    auto cancelled = minimize_module_with_maps(contractible);
    check_change(cancelled);
    assert(cancelled.transformed->number_of_generators() == 1);
    assert(cancelled.transformed->number_of_relations() == 1);
    assert(cancelled.transformed->has_complete_projective_resolution());
    assert(cancelled.transformed->projective_resolution()[1].get_num_cols() == 0);
}

int main() {
    runtime_interop<DynamicCoordinateGradedSparseMatrix<double,int>>();
    runtime_interop<DynamicCoordinateGradedSparseMatrix<double,int,CSCStorage<int>>>();
    runtime_interop<DynamicGridGradedSparseMatrix<double,int>>();
    runtime_interop<DynamicGridGradedSparseMatrix<double,int,CSCStorage<int>>>();
    using Legacy = R2GradedSparseMatrix<int>;
    Module<Legacy> legacy(Legacy(3,3,{{0,1},{0,1,2},{0,1,2}},
        {{1,1},{2,2},{2,2}}, {{0,0},{1,1},{0,0}}));
    check_change(sort_module_with_maps(legacy));
    check_change(minimize_module_with_maps(legacy));
    using Direct = DynamicCoordinateGradedSparseMatrix<double,int>;
    auto empty = from_packed_csc<Direct>(0,0,3,{0},{},{},{});
    assert(empty.parameter_count() == 3);
    empty.compute_rows_forward();
    assert(empty.rows_computed);
    assert(homology_module(ChainComplex<Direct>(3),0).presentation().parameter_count() == 3);
    auto empty_module = convert_module_storage<CSCStorage<int>>(Module<Direct>(ChainComplex<Direct>(3)));
    assert(empty_module.projective_resolution().runtime_poset_identifier() == "3");
    auto mutable_owner = std::make_shared<Module<Direct>>(empty);
    assert(sort_module_with_maps(mutable_owner).original.get() == mutable_owner.get());
    assert(minimize_module_with_maps(mutable_owner).original.get() == mutable_owner.get());
    auto dimension_zero = from_packed_csc<Direct>(2,1,0,{0,1,2},{0,0},{},{});
    auto zero_kernel = static_cast<const Direct&>(dimension_zero).graded_kernel();
    assert(zero_kernel.get_num_cols() == 1 && zero_kernel.parameter_count() == 0);
}
