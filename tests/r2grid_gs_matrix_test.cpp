// CSC algebra remains supported; acknowledge its existing slow-edit diagnostics.
#define GRLINA_CSC_COMPILE_WARNINGS 0
#define GRLINA_CSC_RUNTIME_WARNINGS 0
#include <grlina/csc_matrix.hpp>
#include <grlina/r2grid_gs_matrix.hpp>
#include <grlina/r2graded_matrix.hpp>
#include <grlina/z2graded_matrix.hpp>
#include <cassert>
#include <random>
#include <sstream>
#include <type_traits>

using namespace graded_linalg;

template <typename F>
void rejects(F f) {
    bool rejected = false;
    try { f(); }
    catch (const std::invalid_argument&) { rejected = true; }
    catch (const std::out_of_range&) { rejected = true; }
    assert(rejected);
}

template <typename Storage>
void assert_same(const R2GridGSMatrix<int, double, Storage>& a,
                 const R2GradedSparseMatrix<int, Storage>& b) {
    assert(a.get_num_rows() == b.get_num_rows());
    assert(a.get_num_cols() == b.get_num_cols());
    assert(a.real_column_degrees() == b.col_degrees);
    assert(a.real_row_degrees() == b.row_degrees);
    for (int i = 0; i < a.get_num_cols(); ++i) assert(a.get_col(i) == b.get_col(i));
    a.validate();
}

template <typename Storage>
void grid_and_kernel() {
    using Mat = R2GridGSMatrix<int, double, Storage>;
    using Old = R2GradedSparseMatrix<int, Storage>;
    static_assert(is_graded_sparse_matrix_v<Mat>);
    static_assert(has_matrix_graded_kernel<Mat>::value);
    static_assert(std::is_same_v<typename Mat::degree_type, CoordinateDegree<int, 2>>);
    static_assert(!std::is_constructible_v<Mat, Z2GradedSparseMatrix<int, Storage>>);
    Old original(3, 2, {{0}, {1}, {0, 1}}, {{10, 0}, {0, 10}, {10, 10}}, {{0, 0}, {0, 0}});
    Mat matrix(original);
    assert_same(matrix, original);
    assert(matrix.x_grid == vec<double>({0, 10}));
    assert((matrix.col_degrees[0] == typename Mat::degree_type{1, 0}));
    assert((matrix.col_degrees[1] == typename Mat::degree_type{0, 1}));

    // Keep unused points through two successive kernels. No grid compaction.
    matrix.reindex_grid({-7, 0, 3, 10, 25}, {-5, 0, 8, 10, 30});
    auto xs = matrix.x_grid, ys = matrix.y_grid;
    const auto* input_grid = matrix.x_grid.data();
    auto expected = original.graded_kernel();
    auto kernel = matrix.graded_kernel();
    assert_same(kernel, expected);
    assert(matrix.x_grid.data() == input_grid);
    assert(kernel.x_grid == xs && kernel.y_grid == ys);
    auto expected_second = expected.graded_kernel();
    auto second = kernel.graded_kernel();
    assert_same(second, expected_second);
    assert(second.x_grid == xs && second.y_grid == ys);

    Mat unsorted(3, 2, {{0}, {1}, {0, 1}}, {{10, 0}, {0, 10}, {10, 10}}, {{0, 0}, {0, 0}});
    auto source = unsorted;
    auto syzygies = source.graded_kernel();
    assert((unsorted * syzygies).is_zero());
    assert(syzygies.real_row_degrees() == unsorted.real_column_degrees());

    // Kernels agree with the existing implementation on assorted valid matrices.
    std::mt19937 rng(128);
    const vec<double> coordinates{-3.5, -0.25, 0, 2.75, 19};
    for (int trial = 0; trial < 48; ++trial) {
        int ncols = rng()%7, nrows = rng()%5;
        vec<r2degree> cs, rs;
        for (int r = 0; r < nrows; ++r) rs.push_back({coordinates[rng()%5], coordinates[rng()%5]});
        array<int> data(ncols);
        for (int c = 0; c < ncols; ++c) {
            cs.push_back({coordinates[rng()%5], coordinates[rng()%5]});
            for (int r = 0; r < nrows; ++r)
                if (Degree_traits<r2degree>::smaller_equal(rs[r], cs[c]) && rng()%2) data[c].push_back(r);
        }
        Old old(ncols, nrows, data, cs, rs);
        Mat fresh(ncols, nrows, data, cs, rs);
        Mat unchanged = fresh;
        auto old_kernel = old.graded_kernel();
        auto new_kernel = fresh.graded_kernel();
        assert_same(new_kernel, old_kernel);
        assert((unchanged * new_kernel).is_zero());
    }
}

template <typename Storage>
void combining_and_copies() {
    using Mat = R2GridGSMatrix<int, double, Storage>;
    Mat a(1, 1, {{0}}, {{10, 10}}, {{0, 0}});
    Mat b(1, 1, {{0}}, {{5, 20}}, {{0, 0}});
    const auto original_b = b;
    a.append_matrix(b);
    a.validate();
    assert(a.x_grid == vec<double>({0, 5, 10}));
    assert(a.y_grid == vec<double>({0, 10, 20}));
    assert((a.col_degrees == vec<typename Mat::degree_type>{{2, 1}, {1, 2}}));
    assert(b.x_grid == original_b.x_grid && b.col_degrees == original_b.col_degrees);
    a.append_move_matrix(std::move(b));
    a.validate();
    assert(a.get_num_cols() == 3);
    a.append_matrix(a);
    a.validate();
    assert(a.get_num_cols() == 6);

    auto copied = a;
    Mat assigned;
    assigned = a;
    auto moved = std::move(copied);
    assigned = std::move(moved);
    assigned.validate();
    assert(assigned.x_to_index == a.x_to_index && assigned.y_to_index == a.y_to_index);
    vec<int> keep{1};
    auto restricted = assigned.restricted_domain_copy(keep);
    restricted.validate();
    assert(restricted.x_grid == a.x_grid && restricted.y_grid == a.y_grid);
    assert(restricted.real_column_degree(0) == a.real_column_degree(1));
    auto transposed = a.transposed_copy();
    assert(transposed.x_grid == a.x_grid && transposed.y_to_index == a.y_to_index);
    assert(transposed.real_column_degrees() == a.real_row_degrees());

    R2GradedSparseMatrix<int, Storage> real(1, 1, {{0}}, {{7, 12}}, {{0, 0}});
    Mat from_real;
    from_real = real;
    assert_same(from_real, real);
    R2GridGSMatrix<int, long double, Storage> widened(assigned);
    widened.validate();
    assert(widened.x_grid == vec<long double>(assigned.x_grid.begin(), assigned.x_grid.end()));

    Mat left(1, 1, {{0}}, {{10, 10}}, {{0, 0}});
    Mat right(1, 1, {{0}}, {{25, 25}}, {{10, 10}});
    right.reindex_grid({-9, 5, 10, 25}, {-4, 10, 25});
    auto product = left * right;
    product.validate();
    assert((product.real_column_degree(0) == typename Mat::real_degree_type{25, 25}));
    assert((product.real_row_degree(0) == typename Mat::real_degree_type{0, 0}));
    assert(product.get_col(0) == vec<int>{0});
    auto same = left;
    same.reindex_grid({0, 5, 10}, {0, 10, 30});
    auto sum = left + same;
    sum.validate();
    assert(sum.get_col(0).empty());
    Mat incompatible(1, 1, {{0}}, {{10, 10}}, {{1, 1}});
    rejects([&] { left.append_matrix(incompatible); });
    rejects([&] { (void)(left * incompatible); });
}

template <typename Storage>
void real_operations_and_io() {
    using Mat = R2GridGSMatrix<int, double, Storage>;
    using Old = R2GradedSparseMatrix<int, Storage>;
    Mat matrix(2, 1, {{0}, {0}}, {{10, 0}, {0, 20}}, {{0, 0}});
    assert(matrix.num_cols_before({5, 12}) == 0);
    assert(matrix.num_cols_before({12, 5}) == 1);
    assert(matrix.num_cols_before({1, 1}) == 0);
    assert(matrix.num_cols_before(typename Mat::degree_type{1, 1}) == 2);
    assert(matrix.num_rows_before({-1, 50}) == 0);
    assert(matrix.dim_at({5, 12}) == 1);
    assert(matrix.dim_at({12, 5}) == 0);
    assert((matrix.get_closest_smaller_grid_point({5, 12}) == pair<int>{0, 0}));
    vec<int> columns;
    auto local = matrix.map_at_degree({12, 5}, columns);
    assert(columns == vec<int>{0} && local.get_col(0) == vec<int>{0});
    vec<int> grid_columns;
    auto grid_local = matrix.map_at_degree(typename Mat::degree_type{1, 0}, grid_columns);
    assert(grid_columns == columns && grid_local.get_col(0) == local.get_col(0));
    assert(!matrix.is_admissible_row_operation(0, typename Mat::real_degree_type{1, 0}));
    assert(matrix.is_admissible_row_operation(typename Mat::real_degree_type{1, 0}, 0));
    auto before_shift = matrix.col_degrees;
    matrix.shift({1.5, -2.25});
    assert(matrix.col_degrees == before_shift);
    assert((matrix.real_column_degree(0) == typename Mat::real_degree_type{8.5, 2.25}));
    assert(matrix.x_to_index.at(-1.5) == 0);
    matrix.shift_generators({2, 3});
    assert((matrix.real_row_degree(0) == typename Mat::real_degree_type{-3.5, -0.75}));
    assert((matrix.real_column_degree(0) == typename Mat::real_degree_type{8.5, 2.25}));
    matrix.validate();
    matrix.append_column({0}, {7, 8});
    matrix.validate();
    matrix.set_all_generator_degrees({0, 1});
    matrix.validate();
    assert((matrix.real_row_degree(0) == typename Mat::real_degree_type{0, 1}));
    auto box = matrix.bounding_box();
    assert((box.first == typename Mat::real_degree_type{0, 1}));
    assert(matrix.get_equidistant_grid(1) == vec<typename Mat::real_degree_type>{box.first});

    Old original(2, 1, {{0}, {0}}, {{10, 0}, {0, 20}}, {{0, 0}});
    Mat snapped(original);
    vec<double> xs{0, 12}, ys{0, 25};
    original.snap_to_grid(xs, ys);
    snapped.snap_to_grid(xs, ys);
    assert_same(snapped, original);
    snapped.snap_to_equidistant_grid(1, true);
    snapped.validate();
    Mat swapped(2, 1, {{0}, {0}}, {{10, 0}, {0, 20}}, {{0, 0}});
    swapped.snap_to_grid(swapped.y_grid, swapped.x_grid);
    swapped.validate();
    assert(swapped.get_num_cols() == 1);
    assert((swapped.real_column_degree(0) == typename Mat::real_degree_type{20, 0}));
    Mat cut(2, 1, {{0}, {0}}, {{10, 0}, {0, 20}}, {{0, 0}});
    cut.cut_above(.5, 1);
    cut.validate();
    assert(cut.get_num_cols() == 1 && cut.real_column_degree(0)[1] == 20);
    Mat free(0, 1, {}, {}, {{-1, -2}});
    free.bound_support({4, 7});
    free.validate();
    assert(free.dim_at({0, 0}) == 1 && free.dim_at({4, 0}) == 0 && free.dim_at({0, 7}) == 0);

    std::stringstream stream;
    matrix.to_stream_r2(stream);
    Mat parsed(stream);
    parsed.validate();
    assert(parsed.real_column_degrees() == matrix.real_column_degrees());
    assert(parsed.real_row_degrees() == matrix.real_row_degrees());
    for (int i = 0; i < matrix.get_num_cols(); ++i) assert(parsed.get_col(i) == matrix.get_col(i));
    std::stringstream empty_stream;
    Mat empty(0, 0);
    empty.to_stream(empty_stream);
    Mat empty_parsed(empty_stream);
    empty_parsed.validate();
    assert(empty_parsed.get_num_rows() == 0);
    auto quiver = matrix.induced_quiver_rep(matrix.discrete_support());
    assert(quiver.degrees == matrix.discrete_support());

    Mat explicit_grid(1, 1, array<int>{{0}}, vec<typename Mat::degree_type>{{2, 1}},
                      vec<typename Mat::degree_type>{{0, 0}}, {-1, 3, 20}, {-5, 10});
    explicit_grid.validate();
    explicit_grid.minimize();
    assert((explicit_grid.real_column_degree(0) == typename Mat::real_degree_type{20, 10}));
    Z2GradedSparseMatrix<int, Storage> integer_matrix(1, 1, {{0}}, {{2, 1}}, {{0, 0}});
    Mat from_integer(integer_matrix, {-1, 3, 20}, {-5, 10});
    from_integer.validate();
    assert(from_integer.col_degrees == explicit_grid.col_degrees);
    assert(from_integer.real_column_degrees() == explicit_grid.real_column_degrees());

    auto bad = matrix;
    bad.x_grid[0] -= 1;
    rejects([&] { bad.validate(); });
    bad.rebuild_grid_maps();
    bad.validate();
    rejects([&] { matrix.reindex_grid({0, 0}, {0}); });
    rejects([&] { matrix.reindex_grid({999}, {999}); });
    rejects([&] { matrix.snap_to_equidistant_grid(0); });
    rejects([&] { matrix.append_column({0}, {std::numeric_limits<double>::quiet_NaN(), 0}); });
}

template <typename Storage>
void composite_operations() {
    using Mat = R2GridGSMatrix<int, double, Storage>;
    Mat presentation(1, 1, {{0}}, {{10, 10}}, {{0, 0}});
    Mat generators(1, 1, {{0}}, {{5, 0}}, {{0, 0}});
    auto submodule = presentation.submodule_generated_by(generators);
    submodule.validate();
    assert(submodule.get_num_rows() == 1);
    assert((submodule.real_row_degree(0) == typename Mat::real_degree_type{5, 0}));
    auto at = presentation.submodule_generated_at({3, 4});
    at.validate();
    assert((at.real_row_degree(0) == typename Mat::real_degree_type{3, 4}));
    auto copy = generators;
    auto inverse = copy.inverse_image(presentation, generators);
    inverse.validate();
    auto pres = generators;
    auto pulled = pres.presentation_of_submodule(presentation);
    pulled.validate();
    auto intersection = presentation.submodule_intersection(generators, generators);
    intersection.validate();
    auto quotient = presentation.quotient_by_copy(generators);
    quotient.validate();
    assert(quotient.dim_at({0, 0}) == 1 && quotient.dim_at({5, 0}) == 0);
    auto minimal = quotient;
    minimal.sort_compatibly(); minimal.minimize_variant(); minimal.validate();

    Mat contractible(1, 1, {{0}}, {{0, 0}}, {{0, 0}});
    Mat elements(1, 1, {{0}}, {{5, 0}}, {{0, 0}});
    contractible.sort_compatibly();
    contractible.cancel_local_pairs(&elements);
    contractible.validate(); elements.validate();
    assert(elements.get_num_rows() == 0 && elements.get_col(0).empty());
}

int main() {
    grid_and_kernel<SparseMatrix<int>>();
    grid_and_kernel<CSCMatrix<int>>();
    combining_and_copies<SparseMatrix<int>>();
    combining_and_copies<CSCMatrix<int>>();
    real_operations_and_io<SparseMatrix<int>>();
    real_operations_and_io<CSCMatrix<int>>();
    composite_operations<SparseMatrix<int>>();
    composite_operations<CSCMatrix<int>>();

    // Use values that would lose precision if converted through double.
    using Precise = R2GridGSMatrix<long long, long double>;
    const long double tiny = 1e-30L;
    const long double exact = std::nextafter(1.0L, 2.0L);
    Precise precise(1, 1, {{0}}, {{exact, tiny}}, {{1.0L, 0.0L}});
    std::stringstream stream;
    precise.to_stream(stream);
    Precise reloaded(stream);
    reloaded.validate();
    assert(reloaded.real_column_degrees() == precise.real_column_degrees());
    assert(reloaded.real_row_degrees() == precise.real_row_degrees());
    if (exact != static_cast<long double>(static_cast<double>(exact)))
        rejects([&] { R2GridGSMatrix<long long, double> narrowed(precise); });
    R2GridGSMatrix<int, float> single(1, 1, {{0}}, {{2.5f, 7.f}}, {{0.f, 0.f}});
    auto k = single.graded_kernel();
    k.validate();
    assert(k.x_grid == vec<float>({0, 2.5f}));
}
