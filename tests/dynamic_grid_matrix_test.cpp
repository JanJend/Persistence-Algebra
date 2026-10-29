// NEW: runtime grids preserve real geometry across storage and algebra operations.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/runtime_matrices/class_grid_graded_sparse_matrix.hpp>
#include <grlina/module/module_base.hpp>
#include <cassert>
#include <sstream>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/graded_kernel.hpp>

using namespace graded_linalg;

template<class Storage> void grid_checks() {
    using M = GridGradedSparseMatrix<double, int, Storage>;
    using D = typename M::real_degree_type;
    using I = base_degree<M>;
    M matrix(3, 2, 3, {{0}, {1}, {0,1}},
             {{0.5,4,1.25}, {2.75,1.5,2}, {2.75,4,2}},
             {{-1.5,1.5,1.25}, {-1.5,1.5,1.25}});

    assert(matrix.row_degrees.coordinates().size() == 6);
    assert(matrix.col_degrees.coordinates().size() == 9);
    static_assert(std::is_same_v<typename M::degree_storage_type, FlatDegreeTable<int>>);
    assert(matrix.grid(0) == vec<double>({-1.5,0.5,2.75}));
    assert(matrix.real_col_degree(0) == D({0.5,4,1.25}));
    assert(matrix.dim_at(D{-2,4,2}) == 0);
    assert(matrix.dim_at(D{-1.5,1.5,1.25}) == 2);
    assert(matrix.dim_at(D{1,4,2}) == 1);
    assert(matrix.dim_at(D{3,4,2}) == 0);
    matrix.include_real_degrees({{9,7,8}});
    auto original = matrix;
    original.sort_compatibly(); original.compute_rows_forward();
    auto editable = original.editable_copy();
    static_assert(std::is_same_v<matrix_storage_t<decltype(editable)>, vec<vec<int>>>);
    assert(editable.grid_ == original.grid_ && editable.col_degrees == original.col_degrees);
    assert(editable.row_degrees == original.row_degrees);
    assert(editable.rows_computed && editable._rows == original._rows && editable.compatibly_sorted);
    auto packed = editable.template to_storage<CSCStorage<int>>();
    assert(packed.grid_ == original.grid_ && packed.real_col_degrees() == original.real_col_degrees());
    for (int i = 0; i < packed.get_num_cols(); ++i) assert(packed.get_col(i) == original.get_col(i));
    editable.set_col(0, {});
    assert(!original.get_col(0).empty() && !packed.get_col(0).empty());
    // Keep the original order for the remaining explicit degree assertions.
    original = matrix;
    auto kernel = matrix.graded_kernel();
    assert(kernel.get_num_cols() == 1 && kernel.get_num_rows() == 3);
    assert(kernel.real_col_degree(0) == D({2.75,4,2}));
    assert(kernel.get_col(0) == vec<int>({0,1,2}));
    assert(kernel.grid_ == original.grid_);
    assert((original * kernel).is_zero());
    auto terminal = kernel.graded_kernel();
    assert(terminal.get_num_cols() == 0 && terminal.grid_ == original.grid_);
    auto restriction = original.restricted_domain_copy(vec<int>{2,0});
    assert(restriction.grid_ == original.grid_ && restriction.real_col_degree(0) == D({2.75,4,2}));
    auto shifted = original; shifted.shift(D{0.25,0.5,0.75});
    assert(shifted.real_col_degree(0) == D({0.25,3.5,0.5}));
    assert(shifted.col_degrees == original.col_degrees);

    std::stringstream text; original.to_stream(text); M roundtrip(text);
    assert(roundtrip.real_col_degrees() == original.real_col_degrees());
    assert(roundtrip.real_row_degrees() == original.real_row_degrees());
    assert(roundtrip.get_col(2) == vec<int>({0,1}));
    assert(text.str().find("\n3\n") != std::string::npos);
    auto first = original; auto second = original;
    second.include_real_degrees({{-7,0,0}});
    first.append_matrix(second);
    assert(first.get_num_cols() == 6 && first.real_col_degree(3) == original.real_col_degree(0));
    Module<M> module(original);
    assert(module.dimension_at(D{1,4,2}) == 1);
    auto final_map = original.empty_like(0, original.get_num_cols());
    final_map.row_degrees = original.col_degrees;
    ChainComplex<M> chain({original, final_map});

    std::stringstream chain_text; chain.to_stream(chain_text);
    auto parsed = ChainComplex<M>::from_stream(chain_text);
    assert(parsed[0].real_col_degrees() == original.real_col_degrees());
}

template<class Storage> void two_parameter_kernels() {
    using M = GridGradedSparseMatrix<double, int, Storage>;
    using D = typename M::real_degree_type;
    // The dependence is born at the join, absent from the input's column grades.
    M matrix(2,1,2,{{0},{0}},{{0.5,4.25},{2.75,1.5}},{{-1,0}});
    matrix.include_real_degrees({{8,9}});
    auto source = matrix; auto kernel = matrix.graded_kernel();
    assert(kernel.get_num_cols() == 1 && kernel.get_col(0) == vec<int>({0,1}));
    assert(kernel.real_col_degree(0) == D({2.75,4.25}));
    assert(kernel.real_row_degrees() == source.real_col_degrees());
    assert(kernel.grid_ == source.grid_ && (source * kernel).is_zero());
    auto zero = kernel.graded_kernel(); assert(zero.get_num_cols() == 0 && zero.grid_ == source.grid_);
    std::stringstream firep("firep\nx\ny\n2 1 0\n0.5 4.25 ; 0\n2.75 1.5 ; 0\n-1 0 ;\n");
    M from_firep(firep);
    assert(from_firep.real_col_degrees() == source.real_col_degrees());
    source.sort_compatibly(); source.minimize(); assert(source.get_num_cols() == 2);
    M redundant(3,1,2,{{0},{0},{0}},{{0,0},{1,1},{2,2}},{{0,0}});
    redundant.sort_compatibly(); redundant.minimize();
    assert(redundant.get_num_cols() == 0 && redundant.get_num_rows() == 0);
}

template<class Storage> void three_parameter_resolution() {
    using M = GridGradedSparseMatrix<double, int, Storage>;
    using D = typename M::real_degree_type;
    M differential(3, 1, 3, {{0},{0},{0}},
                   {{1,0,0},{0,1,0},{0,0,1}}, {{0,0,0}});
    auto first = differential;
    auto second = first.graded_kernel();
    assert(second.get_num_cols() == 3 && (differential * second).is_zero());
    auto original_second = second;
    auto third = second.graded_kernel();
    assert(third.get_num_cols() == 1 && third.real_col_degree(0) == D({1,1,1}));
    assert((original_second * third).is_zero());
    auto fourth = third.graded_kernel(); assert(fourth.get_num_cols() == 0);
    Module<M> module(differential);
    assert(module.dimension_at(D{0,0,0}) == 1 && module.dimension_at(D{1,1,1}) == 0);
    module.compute_projective_resolution();
    const auto& resolution = module.proj_res_;
    assert(resolution.size() == 3 && resolution.is_chain_complex());
}

int main() {
    grid_checks<vec<vec<int>>>(); grid_checks<CSCStorage<int>>();
    two_parameter_kernels<vec<vec<int>>>(); two_parameter_kernels<CSCStorage<int>>();
    three_parameter_resolution<vec<vec<int>>>(); three_parameter_resolution<CSCStorage<int>>();
    GridGradedSparseMatrix<int,long,CSCStorage<long>> integral(1,1,1,{{0}},{{7}},{{-2}});
    assert(integral.real_col_degree(0) == DynamicDegree<int>({7}));
    auto kernel = integral.graded_kernel(); assert(kernel.get_num_cols() == 0);
    GridGradedSparseMatrix<double,int> dimension_zero(2,1,0,{{0},{0}},{{},{}},{{}});
    auto zero_kernel = dimension_zero.graded_kernel(); assert(zero_kernel.get_num_cols() == 1);
    GridGradedSparseMatrix<double,int> empty_two(0,0,2), empty_three(0,0,3);
    assert(!empty_two.same_row_degrees(empty_three));
}
