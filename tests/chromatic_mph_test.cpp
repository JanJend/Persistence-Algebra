#include <grlina/chromatic_mph.hpp>
#include <cassert>
#include <sstream>

using namespace graded_linalg;

template<class Matrix>
void check_identity_constructor(const vec<matrix_geometry_degree_t<Matrix>>& degrees) {
    Matrix identity(degrees);
    assert(identity.is_identity);
    assert(identity.get_num_cols() == degrees.size() && identity.get_num_rows() == degrees.size());
    assert(geometric_col_degrees(identity) == degrees && geometric_row_degrees(identity) == degrees);
    assert(identity.is_graded_matrix());
    Matrix copy = identity;
    assert(copy.is_identity);
    for (typename Matrix::index_type i = 0; i < identity.get_num_cols(); ++i) {
        assert(identity.get_col(i) == vec<typename Matrix::index_type>{i});
    }
    Matrix empty(vec<matrix_geometry_degree_t<Matrix>>{});
    assert(empty.is_identity);
    assert(empty.get_num_cols() == 0 && empty.get_num_rows() == 0);
}

int main() {
    check_identity_constructor<R2Matrix<>>({{3, 1}, {0, 2}, {3, 1}});
    check_identity_constructor<R3Matrix<>>({{3, 1, 0}, {0, 2, 1}, {3, 1, 0}});
    check_identity_constructor<GridGradedSparseMatrix<double, 2, int>>({{3, 1}, {0, 2}, {3, 1}});
    using Grid = GridGradedSparseMatrix<double, 2, int>;
    ChainComplex<Grid> grid_complex{
        Grid(1, 2, {{0, 1}}, {{4, 5}}, {{1, 1}, {2, 1}})};
    auto grid_maps = identity_basis_maps(grid_complex);
    for (const auto& identity : grid_maps.to_original) {
        assert(identity.grids == grid_complex[0].grids);
    }
    assert((grid_complex[0] * grid_maps.to_original[1]).data ==
           (grid_maps.to_original[0] * grid_complex[0]).data);
    Grid grid_identity(vec<Grid::real_degree_type>{{3, 1}, {0, 2}});
    assert(grid_identity.to_coordinates().is_identity);
    ChainComplex<R3Matrix<>> input(
        std::string(GRLINA_SOURCE_DIR) + "/tests/fixtures/chromatic_triangle.scc");
    for (std::size_t dimension : {0, 1, 2, 3}) compute_chromatic_mph(input, dimension);
    // An absent chain group has the 0x0 identity as its kernel.
    ChainComplex<R3Matrix<>> empty;
    for (std::size_t dimension : {0, 1})
        compute_chromatic_mph(empty, dimension);

    // Minimization acts on the caller's complex: an edge collapses to a vertex.
    ChainComplex<R3Matrix<>> edge{
        R3Matrix<>(1, 2, {{0, 1}}, {{0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}})};
    compute_chromatic_mph(edge, 0);
    assert(edge[0].get_num_cols() == 0 && edge[0].get_num_rows() == 1);

    // Cancelling two zero-layer edges must still lift the surviving L-cycle
    // to its three original edges, with the excluded edge interleaved.
    ChainComplex<R3Matrix<>> graph{
        R3Matrix<>(4, 4, {{0, 2}, {0, 1}, {0, 3}, {2, 3}},
            {{0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 0}},
            {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}})};
    const auto original_graph = graph;
    auto basis_K = identity_basis_maps(graph);
    graph.minimize(true, &basis_K);
    auto basis_L = basis_K.to_original;
    vec<int> excluded{1};
    basis_L[0].delete_columns(excluded);
    basis_L[1].delete_columns(excluded);
    const R2Matrix<> d_L(1, 1, {{}}, {{1, 1}}, {{0, 0}});
    const auto cycles = basis_L[1] * d_L.graded_kernel();
    assert(basis_K.to_original[1].get_num_cols() == 2);
    assert(basis_L[0].get_num_cols() == 1 && basis_L[0].get_num_rows() == 4);
    assert(cycles.get_num_cols() == 1 && cycles.get_num_rows() == 4);
    assert((cycles.get_col(0) == vec<int>{0, 2, 3}));
    assert((original_graph[0] * cycles).is_zero());

    // Inclusion of a triangle boundary and an isolated vertex, with the
    // excluded simplices interleaved in K's bases and an empty top group in L.
    using Matrix = R2Matrix<>;
    const vec<Matrix::degree_type> vertices_K(5, {0, 0}), vertices_L(4, {0, 0});
    const vec<Matrix::degree_type> edges_K{{0, 2}, {3, 0}, {1, 0}, {2, 1}, {1, 1}};
    const vec<Matrix::degree_type> edges_L{{0, 2}, {1, 0}, {2, 1}};
    const ChainComplex<Matrix> K{
        Matrix(5, 5, {{0, 2}, {1, 2}, {0, 3}, {2, 3}, {0, 1}}, edges_K, vertices_K),
        Matrix(1, 5, {{0, 2, 3}}, {{3, 3}}, edges_K)};
    const ChainComplex<Matrix> L{
        Matrix(3, 4, {{0, 1}, {0, 2}, {1, 2}}, edges_L, vertices_L),
        Matrix(0, 3, {}, {}, edges_L)};
    const ChainMap<Matrix> inclusion(L, K, {
        Matrix(4, 5, {{0}, {2}, {3}, {4}}, vertices_L, vertices_K),
        Matrix(3, 5, {{0}, {2}, {3}}, edges_L, edges_K),
        Matrix(0, 1, {}, {}, {{3, 3}})});
    assert(&inclusion.domain() == &L && &inclusion.target() == &K);
    assert(inclusion.matrices().size() == K.size() + 1);
    for (const auto& matrix : inclusion.matrices()) assert(matrix.is_graded_matrix());
    for (std::size_t n = 1; n <= K.size(); ++n) {
        const auto left = inclusion.target().differential(n) * inclusion[n];
        const auto right = inclusion[n - 1] * inclusion.domain().differential(n);
        assert(left.data == right.data);
        assert(left.col_degrees == right.col_degrees && left.row_degrees == right.row_degrees);
    }

    // Fill L's triangle too, so the outer square has a nonzero incoming boundary.
    auto filled_L = L;
    filled_L[1] = Matrix(1, 3, {{0, 1, 2}}, {{3, 3}}, edges_L);
    const auto kernel_L = L[0].graded_kernel();
    const auto kernel_K = K[0].graded_kernel();
    auto p_L = solve_graded_linear_system(kernel_L, filled_L[1]);
    auto p_K = solve_graded_linear_system(kernel_K, K[1]);
    Matrix included_cycles(kernel_L.get_num_cols(), inclusion[1].get_num_rows(),
        kernel_L.data, kernel_L.col_degrees, inclusion[1].row_degrees);
    included_cycles.transform_columns([&](auto& column) {
        for (auto& row : column) row = inclusion[1].col_last(row);
    });
    const auto product = inclusion[1] * kernel_L;
    assert(included_cycles.data == product.data);
    assert(included_cycles.col_degrees == product.col_degrees &&
           included_cycles.row_degrees == product.row_degrees);
    auto alpha = solve_graded_linear_system(kernel_K, included_cycles);
    assert(p_L && p_K && alpha);
    const Matrix phi_next(vec<Matrix::degree_type>{{3, 3}});
    assert((kernel_L * *p_L).data == filled_L[1].data);
    assert((kernel_K * *p_K).data == K[1].data);
    assert((kernel_K * *alpha).data == (inclusion[1] * kernel_L).data);
    assert((*alpha * *p_L).data == (*p_K * phi_next).data);
    const auto H_L = std::make_shared<const Module<Matrix>>(std::move(*p_L));
    const auto H_K = std::make_shared<const Module<Matrix>>(std::move(*p_K));
    const Homomorphism<Matrix> H_iota(H_L, H_K, vec<Matrix>{std::move(*alpha), phi_next});
    assert(H_iota.check_lifts());

    // The same computation also accepts a complex loaded from three-coordinate FIREP.
    std::istringstream firep("firep\nradius\ndensity\nclass\n1 2 0\n1 1 1 ; 0 1\n0 0 0 ;\n0 0 1 ;\n");
    ChainComplex<R3Matrix<>> firep_input(firep);
    compute_chromatic_mph(firep_input, 1);
}
