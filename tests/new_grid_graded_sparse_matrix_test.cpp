#include <grlina/matrices/grid.hpp>
#include <grlina/graded_linear_system.hpp>
#include <cassert>
#include <sstream>
#include <grlina/matrices/matrix_algorithms.hpp>
#include <grlina/quiver_representations.hpp>

using namespace graded_linalg;

int main() {
    using Coordinates = CoordGradedSparseMatrix<double, 2, int>;
    using Grid = GridGradedSparseMatrix<double, 2, int>;
    using GridD = Grid::GridD;
    static_assert(is_graded_sparse_matrix_v<Grid>);
    static_assert(matrix_grid_backed_v<Grid> && !matrix_grid_backed_v<Coordinates>);
    static_assert(std::is_same_v<Grid::D, CoordinateDegree<int, 2>>);
    static_assert(std::is_same_v<Grid::Grid, std::array<vec<double>, 2>>);

    const SparseMatrix<int> sparse(1, 1);
    for (const auto& initialized : {Grid(), Grid(1, 1), Grid(1, 1, "Identity"),
                                    Grid(1, vec<int>{0}), Grid(sparse), Grid(SparseMatrix<int>(1, 1))}) {
        for (const auto& axis : initialized.grid_) assert(axis == vec<double>{0});
        for (const auto& degree : initialized.real_col_degrees()) assert((degree == GridD{0, 0}));
        for (const auto& degree : initialized.real_row_degrees()) assert((degree == GridD{0, 0}));
    }

    Coordinates coordinates(2, 3, {{0, 1}, {0, 2}},
                            {{3, 1}, {1, 1}}, {{0, 0}, {1, 0}, {0, 1}});
    Grid matrix(coordinates);
    assert(matrix.to_coordinates().data == coordinates.data);
    assert(matrix.real_col_degrees() == coordinates.col_degrees);
    assert(matrix.real_row_degrees() == coordinates.row_degrees);
    matrix.sort_compatibly();
    coordinates.sort_compatibly();

    assert(matrix.to_coordinates().data == coordinates.data);
    assert((matrix.grid_degree({3, 1}) == Grid::D{2, 1}));
    assert((matrix.query_degree({2, 0.5}) == Grid::D{1, 0}));
    assert(matrix.dim_at_grid_degree(matrix.query_degree({1, 1})) == coordinates.dim_at({1, 1}));
    assert(matrix.dim_at_grid_degree(matrix.query_degree({-1, 2})) == 0);
    assert(matrix.dim_at_grid_degree(matrix.query_degree({100, 100})) == 1);

    matrix.include_real_degrees({{99, 99}});
    Grid restricted = matrix.restricted_domain_copy({1});
    assert(restricted.grid_ == matrix.grid_);
    assert((restricted.real_col_degree(0) == GridD{3, 1}));
    Grid transposed = matrix.transposed_copy();
    assert(transposed.grid_ == matrix.grid_);
    assert(transposed.real_row_degrees() == matrix.real_col_degrees());
    assert(transposed.real_col_degrees() == matrix.real_row_degrees());
    auto empty = matrix.empty_like(2, 3);

    assert(empty.grid_ == matrix.grid_ && empty.data.size() == 2);

    Grid identity(2, 2, "Identity");
    identity.set_real_degrees(matrix.real_col_degrees(), matrix.real_col_degrees());
    const auto identity_grids = identity.grid_;
    Grid product = matrix * identity;
    assert(product.grid_ == matrix.grid_ && product.data == matrix.data);
    assert(product.real_col_degrees() == matrix.real_col_degrees());
    assert(identity.grid_ == identity_grids);

    Grid extra(Coordinates(1, 3, {{0}}, {{2, 1}}, coordinates.row_degrees));
    auto appended = matrix;
    appended.append_matrix(extra);

    assert(appended.get_num_cols() == 3);
    assert((appended.real_col_degree(2) == GridD{2, 1}));
    assert(appended.real_row_degrees() == matrix.real_row_degrees());
    auto moved = matrix;
    auto extra_copy = extra;
    moved.append_move_matrix(std::move(extra_copy));
    assert(moved.real_col_degrees() == appended.real_col_degrees());

    auto shifted = matrix;
    shifted.shift({0.25, 0.5});

    assert(shifted.col_degrees == matrix.col_degrees);
    assert((shifted.real_col_degree(0) == GridD{0.75, 0.5}));
    shifted = matrix;
    shifted.shift_generators({2, 2});

    assert(shifted.real_col_degrees() == matrix.real_col_degrees());
    assert((shifted.real_row_degree(0) == GridD{-2, -2}));

    Grid rhs(Coordinates(1, 3, {{0, 1}}, {{4, 2}}, coordinates.row_degrees));
    auto solution = solve_graded_linear_system(matrix, rhs);
    assert(solution);
    assert((matrix * *solution).data == rhs.data);
    assert(solution->real_row_degrees() == matrix.real_col_degrees());
    assert(solution->real_col_degrees() == rhs.real_col_degrees());

    Coordinates presentation_coordinates(1, 2, {{0, 1}}, {{1, 1}}, {{0, 0}, {1, 1}});
    Grid presentation(presentation_coordinates);
    presentation.include_real_degrees({{0.5, 0.5}});
    Grid elements(Coordinates(1, 2, {{1}}, {{3, 3}}, presentation_coordinates.row_degrees));
    presentation.sort_compatibly();
    presentation.cancel_local_pairs(&elements);


    assert(presentation.get_num_cols() == 0 && presentation.get_num_rows() == 1);
    assert((elements.data == array<int>{{0}}));
    assert((elements.real_col_degree(0) == GridD{3, 3}));

    std::stringstream stream;
    matrix.to_stream(stream);
    Grid loaded(stream);

    assert(loaded.data == matrix.data);
    assert(loaded.real_col_degrees() == matrix.real_col_degrees());
    assert(loaded.real_row_degrees() == matrix.real_row_degrees());

    using IntegerCoordinates = CoordGradedSparseMatrix<long long, 3, int>;
    using IntegerGrid = GridGradedSparseMatrix<long long, 3, int>;
    IntegerGrid integral(IntegerCoordinates(1, 1, {{0}}, {{10, 20, 30}}, {{-2, 0, 4}}));
    std::stringstream integer_stream;
    integral.to_stream(integer_stream);
    IntegerGrid integer_loaded(integer_stream);
    assert(integer_loaded.real_col_degrees() == integral.real_col_degrees());
    assert(integer_loaded.real_row_degrees() == integral.real_row_degrees());
    std::stringstream compatible_stream;
    integral.to_stream_r3(compatible_stream);
    assert(compatible_stream.str() == integer_stream.str());

    Grid free_map(Coordinates(1, 0, {{}}, {{1, 2}}, {}));
    std::stringstream free_stream;
    free_map.to_stream(free_stream);
    Grid free_loaded(free_stream);
    assert(free_loaded.get_num_rows() == 0);
    assert(free_loaded.real_col_degrees() == free_map.real_col_degrees());

    assert(matrix.dim_at({1, 1}) == coordinates.dim_at({1, 1}));
    assert(matrix.num_cols_before({2, 1}) == 1);
    assert(matrix.real_column_degrees() == matrix.real_col_degrees());
    assert((matrix.get_closest_smaller_grid_point({2, 0.5}) == std::pair<int, int>{1, 0}));
    assert(matrix.discrete_support() == coordinates.discrete_support());
    auto quiver = matrix.induced_quiver_rep();
    assert(quiver.degrees == coordinates.discrete_support());

    Grid redundant(2, 1, {{0}, {0}}, {{1, 1}, {2, 2}}, {{0, 0}});
    redundant.include_real_degrees({{0.5, 0.5}});
    auto axes = redundant.grid_;
    redundant.sort_compatibly();
    redundant.minimize();
    assert(redundant.get_num_cols() == 1 && redundant.grid_ == axes);
    assert(redundant.is_minimal());
    redundant.minimize_variant();

    Grid free(0, 1, {}, {}, {{0, 0}});
    Grid generator(1, 1, {{0}}, {{1, 1}}, {{0, 0}});
    free.include_real_degrees({{0.5, 0.5}});
    auto generated = free.submodule_generated_by(generator);
    assert(generated.get_num_cols() == 0 && generated.get_num_rows() == 1);
    assert((generated.real_row_degree(0) == GridD{1, 1}));
    auto presented = generator.presentation_of_submodule(free);
    assert(presented.get_num_cols() == 0 && presented.get_num_rows() == 1);
    auto at = free.submodule_generated_at({0.75, 0.75});
    assert(at.get_num_cols() == 0 && (at.real_row_degree(0) == GridD{0.75, 0.75}));
    auto quotient = free.quotient_by_copy(generator);
    assert(quotient.dim_at({0, 0}) == 1 && quotient.dim_at({1, 1}) == 0);

    Grid left(1, 1, {{0}}, {{1, 0}}, {{0, 0}});
    Grid right(1, 1, {{0}}, {{0, 1}}, {{0, 0}});
    auto intersection = free.submodule_intersection(left, right);
    assert((intersection.data == array<int>{{0}}));
    assert((intersection.real_col_degree(0) == GridD{1, 1}));
    auto pullback = left.inverse_image_copy(free, right);
    assert((pullback.data == array<int>{{0}}));
    assert((pullback.real_row_degree(0) == GridD{1, 0}));
    assert((pullback.real_col_degree(0) == GridD{1, 1}));

    Grid doubled = left + left;
    assert(doubled.get_num_cols() == 1 && doubled.is_zero(0));
    assert(doubled.real_col_degrees() == left.real_col_degrees());
    auto appended_grid = free;
    appended_grid.append_column({0}, {0.75, 0.75});
    assert((appended_grid.real_col_degree(0) == GridD{0.75, 0.75}));
    auto degree = appended_grid.grid_degree({0.75, 0.75});
    appended_grid.append_column_at_grid_degree({0}, degree);
    assert(appended_grid.real_col_degree(1) == appended_grid.real_col_degree(0));

    auto bounded = free;
    bounded.bound_support({2, 3});
    assert(bounded.get_num_cols() == 2);
    assert(bounded.dim_at({1, 1}) == 1 && bounded.dim_at({2, 1}) == 0);
    Grid snapped(1, 1, {{0}}, {{1.25, 0.75}}, {{0.25, 0.25}});
    snapped.snap_to_grid({0, 1, 2}, {0, 1, 2});
    assert((snapped.real_col_degree(0) == GridD{2, 1}));
    assert((snapped.real_row_degree(0) == GridD{1, 1}));
    Grid cut(2, 1, {{0}, {0}}, {{2, 1}, {4, 3}}, {{0, 0}});
    cut.cut_above(0.5, 0.5);
    assert(cut.get_num_cols() == 1 && (cut.real_col_degree(0) == GridD{2, 1}));

    using IntegralGrid = GridGradedSparseMatrix<int, 2, int>;
    IntegralGrid same_scalar(1, 1, {{0}}, {{20, 30}}, {{10, 10}});
    auto ordinal = same_scalar.query_degree({15, 15});
    assert(same_scalar.dim_at_grid_degree(ordinal) == 1);
    assert(same_scalar.dim_at({15, 15}) == 1);
    const IntegralGrid::GridD point{15, 15};
    assert(same_scalar.dim_at(point) == same_scalar.dim_at({15, 15}));
    assert(same_scalar.map_at_degree_pair(point).first.get_num_cols() == 0);
    assert(same_scalar.map_at_grid_degree_pair(ordinal).first.get_num_cols() == 0);
    same_scalar.append_column({0}, {25, 35});
    assert((same_scalar.real_col_degree(1) == IntegralGrid::GridD{25, 35}));
    auto grid_degree = same_scalar.grid_degree({25, 35});
    same_scalar.append_column_at_grid_degree({0}, grid_degree);
    assert(same_scalar.real_col_degree(2) == same_scalar.real_col_degree(1));
    auto physical_generated = same_scalar.submodule_generated_at({15, 15});
    assert((physical_generated.real_row_degree(0) == IntegralGrid::GridD{15, 15}));
    auto generated_at_grid_degree = same_scalar.submodule_generated_at_grid_degree(ordinal);
    assert((generated_at_grid_degree.real_row_degree(0) == IntegralGrid::GridD{10, 10}));
    std::stringstream same_scalar_stream;
    same_scalar.to_stream(same_scalar_stream);
    IntegralGrid same_scalar_loaded(same_scalar_stream);
    assert(same_scalar_loaded.real_col_degrees() == same_scalar.real_col_degrees());
    assert(same_scalar_loaded.real_row_degrees() == same_scalar.real_row_degrees());

    Coordinates degrees_only(1, 1, {{2, 3}}, {{0, 0}});
    degrees_only.set_col(0, {0});
    Grid imported(degrees_only);
    Grid copied(static_cast<const SparseMatrix<int>&>(degrees_only));
    assert(copied.data == degrees_only.data);
    Grid assigned;
    assigned = degrees_only;
    assert(assigned.real_col_degrees() == degrees_only.col_degrees);
    GridGradedSparseMatrix<long double, 2, int> widened(imported);
    assert(widened.grid_[0].size() == imported.grid_[0].size());
    Grid embedded(1, 1, vec<Grid::D>{{1, 1}}, vec<Grid::D>{{0, 0}}, {{{0, 2}, {0, 3}}});
    embedded.set_col(0, {0});
    assert(embedded.real_col_degrees() == imported.real_col_degrees());
}
