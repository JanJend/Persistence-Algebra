// NEW: packed exchange, ownership, basis transport and nonfree homology.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/invariants/hilbert_function.hpp>
#include <grlina/runtime_matrices/packed_csc.hpp>
#include <grlina/module/submodule.hpp>
#include <grlina/runtime_matrices/class_coord_graded_sparse_matrix.hpp>
#include <cassert>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/graded_kernel.hpp>
#include <grlina/matrices/io.hpp>

using namespace graded_linalg;
using Degree = DynamicDegree<double>;

template<class Matrix>
Matrix matrix(int columns, int rows, array<int> entries,
              vec<Degree> column_degrees, vec<Degree> row_degrees, std::size_t parameters = 3) {
    Matrix result(columns, rows, parameters);
    GradedMatrixIO<Matrix>::assign_degrees(result, column_degrees, row_degrees);
    matrix_storage_t<Matrix> storage;
    for (auto& column : entries) storage.push_back(std::move(column));
    result.assign_data(std::move(storage));
    result.compatibly_sorted = false;
    return result;
}

template<class Matrix>
void check_change(const Module<Matrix>& before, bool minimize = true) {
    auto original = std::make_shared<const Module<Matrix>>(before);
    auto changed = std::make_shared<Module<Matrix>>(before);
    Matrix incoming(geometric_row_degrees(before.presentation()));
    Matrix outgoing = incoming;
    if (minimize) changed->minimize(true, &incoming, {&outgoing});
    else changed->sort_compatibly(&incoming, {&outgoing});
    Homomorphism<Matrix> forward(original, changed, incoming), backward(changed, original, outgoing);
    assert(forward.check_lifts() && backward.check_lifts());
    assert((forward.compose(backward) + Homomorphism<Matrix>::identity(original)).image(false).is_zero());
    assert((backward.compose(forward) + Homomorphism<Matrix>::identity(changed)).image(false).is_zero());
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
    check_change(*original, false);
    check_change(*original);
    Module<Matrix> minimized = *original;
    minimized.minimize();
    assert(original->presentation().data == d1.data);
    assert(minimized.num_of_gens() == 2 && minimized.num_of_rels() == 1);
    for (const auto& grade : {zero,one,two})
        assert(original->dimension_at(grade) == minimized.dimension_at(grade));

    auto packed = to_packed_csc(d1);
    auto restored = from_packed_csc<Matrix>(3,3,3,packed.data.offsets(),packed.data.entries(),
        packed.col_degrees.coordinates(),packed.row_degrees.coordinates(), [&] {
            if constexpr (matrix_grid_backed_v<Matrix>) return packed.grid_;
            else return vec<vec<double>>{};
        }());
    assert(restored.data == d1.data);
    assert(geometric_row_degrees(restored) == geometric_row_degrees(d1));
    assert(geometric_col_degrees(restored) == geometric_col_degrees(d1));
    if constexpr (matrix_grid_backed_v<Matrix>) assert(restored.grid_ == d1.grid_);

    // Export an inclusion from a stack object, then destroy that object.
    auto inclusion = [&] {
        auto submodule = Submodule<Matrix>::whole(original);
        return submodule.inclusion();
    }();

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
    auto homology = homology_module(complex,1,false,true);
    assert(homology.num_of_gens() == 3);
    assert(homology.num_of_rels() == 1);
    assert(homology.dimension_at(one) == 2);
    assert(homology.dimension_at({1,1,0}) == 1);
    assert((outgoing * *homology.cycles).is_zero());
    auto incoming = matrix<Matrix>(1,3,{{0,1}}, {{1,1,0}},
        {{1,0,0},{0,1,0},{0,0,1}});
    ChainComplex<Matrix> with_boundary({outgoing,incoming});
    auto quotient = homology_module(with_boundary,1,true,true);
    assert(quotient.dimension_at(one) == 1);
    assert(quotient.dimension_at({1,1,0}) == 0);
    assert(quotient.cycles->get_num_cols() == quotient.num_of_gens());
    assert((outgoing * *quotient.cycles).is_zero());
    // Identity chain map induces identity after choosing the returned bases.
    Matrix target_cycles = *quotient.cycles;
    target_cycles.append_matrix(incoming);
    auto induced_lift = solve_graded_linear_system(target_cycles, *quotient.cycles);
    assert(induced_lift);
    induced_lift->cull_columns(quotient.num_of_gens(), false);
    auto owner = std::make_shared<const Module<Matrix>>(quotient);
    Homomorphism<Matrix> induced(owner, owner, std::move(*induced_lift));
    assert(induced.check_lifts());
    assert(homology_module(complex,0).dimension_at(zero) == 1);
    assert(homology_module(complex,0).dimension_at(one) == 0);

    // Every stored differential and the completeness marker survive conversion.
    Module<Matrix> resolved(outgoing);
    resolved.compute_projective_resolution();
    resolved.inj_res_ = with_boundary;
    auto converted = convert_module_storage<CSCStorage<int>>(resolved);
    assert(converted.res_complete_);
    assert(converted.proj_res_.size() == resolved.proj_res_.size());
    assert(converted.inj_res_.size() == 2);
    auto converted_back = convert_module_storage<array<int>>(converted);
    assert(converted_back.proj_res_[0].data == outgoing.data);
    check_change(resolved, false);
    check_change(resolved);

    // Equal-degree pairs in higher groups transport operations to both
    // neighboring differentials, rather than just changing the presentation.
    auto lower = matrix<Matrix>(3,1,{{0},{0},{0}}, {one,one,one}, {zero});
    auto middle = matrix<Matrix>(3,3,{{0,1},{1,2},{0,2}}, {one,one,two}, {one,one,one});
    auto upper = matrix<Matrix>(1,3,{{0,1,2}}, {two}, {one,one,two});
    auto contractible = Module<Matrix>::from_projective_resolution(
        ChainComplex<Matrix>({lower,middle,upper}), ResolutionCompleteness::complete);
    check_change(contractible);
    auto cancelled = contractible;
    cancelled.minimize();
    assert(cancelled.num_of_gens() == 1);
    assert(cancelled.num_of_rels() == 1);
    assert(cancelled.res_complete_);
    assert(cancelled.proj_res_[1].get_num_cols() == 0);
}

int main() {
    runtime_interop<CoordGradedSparseMatrix<double,int>>();
    runtime_interop<CoordGradedSparseMatrix<double,int,CSCStorage<int>>>();
    runtime_interop<GridGradedSparseMatrix<double,int>>();
    runtime_interop<GridGradedSparseMatrix<double,int,CSCStorage<int>>>();
    using Legacy = R2GradedSparseMatrix<int>;
    Module<Legacy> legacy(Legacy(3,3,{{0,1},{0,1,2},{0,1,2}},
        {{1,1},{2,2},{2,2}}, {{0,0},{1,1},{0,0}}));
    check_change(legacy, false);
    check_change(legacy);
    using Direct = CoordGradedSparseMatrix<double,int>;
    auto empty = from_packed_csc<Direct>(0,0,3,{0},{},{},{});
    assert(empty.parameter_count() == 3);
    empty.compute_rows_forward();
    assert(empty.rows_computed);
    auto mutable_owner = std::make_shared<Module<Direct>>(empty);
    check_change(*mutable_owner, false);
    check_change(*mutable_owner);
    auto dimension_zero = from_packed_csc<Direct>(2,1,0,{0,1,2},{0,0},{},{});
    auto zero_kernel = static_cast<const Direct&>(dimension_zero).graded_kernel();
    assert(zero_kernel.get_num_cols() == 1 && zero_kernel.parameter_count() == 0);
}
