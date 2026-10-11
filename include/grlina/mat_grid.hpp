/** Integer coordinate degrees with grids recording their physical coordinates. */
#pragma once

#include <grlina/mat_coord.hpp>
#include <grlina/mat_io.hpp>
#include <algorithm>
#include <array>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace graded_linalg {

template <typename Scalar, std::size_t Dimension, typename Index>
class GridGradedSparseMatrix
    : public CoordGradedSparseMatrix<Index, Dimension, Index,
                                     GridGradedSparseMatrix<Scalar, Dimension, Index>> {
public:
    using Self = GridGradedSparseMatrix;
    using Base = CoordGradedSparseMatrix<Index, Dimension, Index, Self>;
    using GradedBase = typename Base::GradedBase;
    using degree_type = CoordinateDegree<Index, Dimension>;
    using real_degree_type = CoordinateDegree<Scalar, Dimension>;
    using coordinate_matrix_type = CoordGradedSparseMatrix<Scalar, Dimension, Index>;
    using grid_type = std::array<vec<Scalar>, Dimension>;
    // Distinguish physical braces from typed grid indices even when Scalar == Index.
    struct PhysicalDegree : real_degree_type {
        PhysicalDegree(const real_degree_type& degree) : real_degree_type(degree) {}
        PhysicalDegree(std::initializer_list<Scalar> values) : real_degree_type{} {
            std::copy(values.begin(), values.end(), this->begin());
        }
    };
    using point_type = std::conditional_t<std::is_same_v<Scalar, Index>, PhysicalDegree, real_degree_type>;
    template <typename D>
    using EnableGridDegree = std::enable_if_t<std::is_same_v<std::decay_t<D>, degree_type>, int>;
    static constexpr bool grid_backed = true;

    // Stored coordinate d[a] denotes grids[a][d[a]]. Axes are strictly increasing.
    grid_type grids = [] {
        grid_type axes;
        for (auto& axis : axes) axis.push_back(Scalar{});
        return axes;
    }();

    GridGradedSparseMatrix() = default;
    GridGradedSparseMatrix(const Self&) = default;
    GridGradedSparseMatrix(Self&&) = default;
    Self& operator=(const Self&) = default;
    Self& operator=(Self&&) = default;
    GridGradedSparseMatrix(Index columns, Index rows) : Base(columns, rows) {}
    GridGradedSparseMatrix(Index columns, Index rows, const std::string& type, Index percent = -1)
        : Base(columns, rows, type, percent) {}
    explicit GridGradedSparseMatrix(const vec<real_degree_type>& degrees)
        : GridGradedSparseMatrix(static_cast<Index>(degrees.size()), static_cast<Index>(degrees.size()), "Identity") {
        set_real_degrees(degrees, degrees);
    }
    GridGradedSparseMatrix(Index rows, vec<Index> indicator) : Base(rows, std::move(indicator)) {}
    explicit GridGradedSparseMatrix(SparseMatrix<Index>&& matrix) : Base(std::move(matrix)) {}
    explicit GridGradedSparseMatrix(const SparseMatrix<Index>& matrix) : Base(matrix) {}

    GridGradedSparseMatrix(Index columns, Index rows, vec<real_degree_type> c, vec<real_degree_type> r)
        : Base(columns, rows) { set_real_degrees(c, r); }
    GridGradedSparseMatrix(Index columns, Index rows, const array<Index>& entries,
                          vec<real_degree_type> c, vec<real_degree_type> r)
        : Base(columns, rows, entries, vec<degree_type>(columns), vec<degree_type>(rows)) { set_real_degrees(c, r); }
    GridGradedSparseMatrix(Index columns, Index rows, vec<degree_type> c, vec<degree_type> r, grid_type axes)
        : Base(columns, rows, std::move(c), std::move(r)), grids(std::move(axes)) {}
    GridGradedSparseMatrix(Index columns, Index rows, const array<Index>& entries,
                          vec<degree_type> c, vec<degree_type> r, grid_type axes)
        : Base(columns, rows, entries, std::move(c), std::move(r)), grids(std::move(axes)) {}

    template <typename OtherScalar, typename Derived>
    explicit GridGradedSparseMatrix(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived>& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows()) {
        this->assign_data(matrix.data);
        set_real_degrees(convert_degrees(matrix.col_degrees), convert_degrees(matrix.row_degrees));
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar, typename Derived>
    GridGradedSparseMatrix(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived>& matrix, grid_type axes)
        : Base(matrix.get_num_cols(), matrix.get_num_rows()), grids(std::move(axes)) {
        this->assign_data(matrix.data);
        for (Index i = 0; i < matrix.get_num_cols(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) this->col_degrees[i][a] = static_cast<Index>(matrix.col_degrees[i][a]);
        for (Index i = 0; i < matrix.get_num_rows(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) this->row_degrees[i][a] = static_cast<Index>(matrix.row_degrees[i][a]);
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar>
    explicit GridGradedSparseMatrix(const GridGradedSparseMatrix<OtherScalar, Dimension, Index>& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows(), matrix.data, matrix.col_degrees, matrix.row_degrees) {
        for (std::size_t a = 0; a < Dimension; ++a) grids[a].assign(matrix.grids[a].begin(), matrix.grids[a].end());
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar, typename Derived>
    Self& operator=(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived>& matrix) {
        return *this = Self(matrix);
    }
    template <typename OtherScalar>
    Self& operator=(const GridGradedSparseMatrix<OtherScalar, Dimension, Index>& matrix) { return *this = Self(matrix); }

    explicit GridGradedSparseMatrix(std::istream& input, bool sort = false)
        : GridGradedSparseMatrix(coordinate_matrix_type(input, sort)) {}
    explicit GridGradedSparseMatrix(const std::string& path, bool sort = false)
        : GridGradedSparseMatrix(coordinate_matrix_type(path, sort)) {}

    coordinate_matrix_type to_coordinates() const {
        coordinate_matrix_type result(this->get_num_cols(), this->get_num_rows(),
                                      this->data, real_col_degrees(), real_row_degrees());
        result.compatibly_sorted = this->compatibly_sorted;
        result.is_identity = this->is_identity;
        return result;
    }
    Self empty_like(Index columns, Index rows) const {
        Self result(columns, rows);
        result.grids = grids;
        return result;
    }
    Self restricted_domain_copy(vec<Index> columns) const {
        Self result = Base::restricted_domain_copy(columns);
        result.grids = grids;
        return result;
    }
    Self transposed_copy() const {
        Self result = Base::transposed_copy();
        result.grids = grids;
        return result;
    }
    friend Self operator*(Self left, Self right) {
        left.merge_grids(right);
        Self result = static_cast<const GradedBase&>(left) * static_cast<const GradedBase&>(right);
        result.grids = std::move(left.grids);
        return result;
    }
    friend Self operator+(Self left, Self right) {
        left.merge_grids(right);
        Self result = left.empty_like(left.get_num_cols(), left.get_num_rows());
        result.assign_data((static_cast<const SparseMatrix<Index>&>(left) + static_cast<const SparseMatrix<Index>&>(right)).data);
        result.col_degrees = left.col_degrees;
        result.row_degrees = left.row_degrees;
        result.compatibly_sorted = left.compatibly_sorted;
        return result;
    }

    void append_matrix(const Self& other) {
        Self aligned = other;
        merge_grids(aligned);
        Base::append_matrix(aligned);
    }
    void append_move_matrix(Self&& other) {
        if (this == &other) { append_matrix(other); return; }
        merge_grids(other);
        Base::append_move_matrix(std::move(other));
    }
    void append_column(const vec<Index>& column, const point_type& degree) {
        include_real_degrees({degree});
        Base::append_column(column, grid_degree(degree));
    }
    void append_column_at_grid_degree(const vec<Index>& column, const degree_type& degree) {
        Base::append_column(column, degree);
    }
    template <typename D, EnableGridDegree<D> = 0>
    void append_column(const vec<Index>& column, const D& degree) { append_column_at_grid_degree(column, degree); }

    void quotient_by(const Self& other) {
        append_matrix(other);
        this->sort_compatibly();
        this->minimize();
    }
    Self quotient_by_copy(const Self& other) const {
        Self result = *this;
        result.quotient_by(other);
        return result;
    }
    Self inverse_image(const Self& presentation, const Self& submodule) {
        Index domain = this->get_num_cols();
        append_matrix(submodule);
        append_matrix(presentation);
        auto result = this->graded_kernel();
        result.cull_columns(domain, false);
        return result;
    }
    Self inverse_image_copy(const Self& presentation, const Self& submodule) const {
        Self result = *this;
        return result.inverse_image(presentation, submodule);
    }
    Self submodule_intersection(const Self& left, const Self& right) const {
        return left * left.inverse_image_copy(*this, right);
    }
    Self submodule_generated_by(const Self& generators) const {
        Self injection = generators;
        Index count = injection.get_num_cols();
        injection.append_matrix(*this);
        auto result = injection.graded_kernel();
        result.cull_columns(count, false);
        return result;
    }
    Self presentation_of_submodule(const Self& presentation) {
        Index count = this->get_num_cols();
        append_matrix(presentation);
        auto result = this->graded_kernel();
        result.cull_columns(count, false);
        return result;
    }
    Self submodule_generated_at(const point_type& degree) const {
        auto basis = basislift_at(degree);
        Self injection(this->get_num_rows(), basis);
        injection.grids = grids;
        injection.row_degrees = this->row_degrees;
        injection.include_real_degrees({degree});
        injection.col_degrees.assign(basis.size(), injection.grid_degree(degree));
        injection.append_matrix(*this);
        auto result = injection.graded_kernel();
        result.cull_columns(static_cast<Index>(basis.size()), false);
        result.sort_compatibly();
        result.minimize();
        return result;
    }
    template <typename D, EnableGridDegree<D> = 0>
    Self submodule_generated_at(const D& degree) const { return submodule_generated_at(point_type(real_degree(degree))); }

    using Base::cancel_local_pairs;
    void cancel_local_pairs(Self* elements = nullptr) {
        if (elements) merge_grids(*elements);
        Base::cancel_local_pairs(elements);
    }
    void shift(const real_degree_type& amount) {
        for (std::size_t a = 0; a < Dimension; ++a)
            for (auto& coordinate : grids[a]) coordinate -= amount[a];
    }
    void shift_generators(const real_degree_type& amount) {
        auto rows = real_row_degrees();
        for (auto& degree : rows)
            for (std::size_t a = 0; a < Dimension; ++a) degree[a] -= amount[a];
        include_real_degrees(rows);
        this->row_degrees = encode_degrees(rows, grids);
        this->compatibly_sorted = false;
    }
    void set_all_generator_degrees(const point_type& degree) {
        include_real_degrees({degree});
        Base::set_all_generator_degrees(grid_degree(degree));
    }
    template <typename D, EnableGridDegree<D> = 0>
    void set_all_generator_degrees(const D& degree) { Base::set_all_generator_degrees(degree); }

    std::pair<SparseMatrix<Index>, vec<Index>> map_at_degree_pair(const point_type& degree, bool shifted = true) const {
        return Base::map_at_degree_pair(query_degree(degree), shifted);
    }
    SparseMatrix<Index> map_at_degree(const point_type& degree, vec<Index>& columns) const {
        return Base::map_at_degree(query_degree(degree), columns);
    }
    Index num_cols_before(const point_type& degree) const { return count_before(this->col_degrees, query_degree(degree)); }
    Index num_rows_before(const point_type& degree) const { return count_before(this->row_degrees, query_degree(degree)); }
    vec<Index> basislift_at(const point_type& degree) const { return Base::basislift_at(query_degree(degree)); }
    Index dim_at(const point_type& degree) const { return static_cast<Index>(basislift_at(degree).size()); }
    vec<Index> admissible_row_indices(const point_type& degree) const {
        vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i)
            if (is_admissible_row_operation(degree, i)) rows.push_back(i);
        return rows;
    }
    bool is_admissible_column_operation(Index i, Index j) const { return Base::is_admissible_column_operation(i, j); }
    bool is_admissible_column_operation(Index i, const point_type& degree) const {
        return Degree_traits<real_degree_type>::smaller_equal(real_col_degree(i), degree);
    }
    bool is_admissible_row_operation(Index i, Index j) const { return Base::is_admissible_row_operation(i, j); }
    bool is_admissible_row_operation(Index i, const point_type& degree) const {
        return Degree_traits<real_degree_type>::greater_equal(real_row_degree(i), degree);
    }
    bool is_admissible_row_operation(const point_type& degree, Index i) const {
        return Degree_traits<real_degree_type>::greater_equal(degree, real_row_degree(i));
    }

    // Typed integer arrays denote grid indices; braces select physical coordinates.
    template <typename D, EnableGridDegree<D> = 0>
    auto map_at_degree_pair(const D& degree, bool shifted = true) const { return Base::map_at_degree_pair(degree, shifted); }
    template <typename D, EnableGridDegree<D> = 0>
    auto map_at_degree(const D& degree, vec<Index>& columns) const { return Base::map_at_degree(degree, columns); }
    template <typename D, EnableGridDegree<D> = 0>
    Index num_cols_before(const D& degree) const { return count_before(this->col_degrees, degree); }
    template <typename D, EnableGridDegree<D> = 0>
    Index num_rows_before(const D& degree) const { return count_before(this->row_degrees, degree); }
    template <typename D, EnableGridDegree<D> = 0>
    vec<Index> basislift_at(const D& degree) const { return Base::basislift_at(degree); }
    template <typename D, EnableGridDegree<D> = 0>
    Index dim_at(const D& degree) const { return static_cast<Index>(Base::basislift_at(degree).size()); }
    template <typename D, EnableGridDegree<D> = 0>
    vec<Index> admissible_row_indices(const D& degree) const {
        vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i)
            if (Base::is_admissible_row_operation(degree, i)) rows.push_back(i);
        return rows;
    }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_column_operation(Index i, const D& degree) const { return Base::is_admissible_column_operation(i, degree); }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_row_operation(Index i, const D& degree) const { return Base::is_admissible_row_operation(i, degree); }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_row_operation(const D& degree, Index i) const { return Base::is_admissible_row_operation(degree, i); }

    vec<real_degree_type> discrete_support() const { return decode_degrees(Base::discrete_support()); }
    auto induced_quiver_rep(vec<real_degree_type> vertices = {}, array<Index> edges = {}) {
        return to_coordinates().induced_quiver_rep(std::move(vertices), std::move(edges));
    }
    std::pair<real_degree_type, real_degree_type> bounding_box() const { return to_coordinates().bounding_box(); }
    vec<real_degree_type> get_equidistant_grid(const int& n) const { return to_coordinates().get_equidistant_grid(n); }
    void snap_to_grid(grid_type axes) {
        auto coordinates = to_coordinates();
        coordinates.snap_to_grid(axes);
        *this = Self(coordinates);
        reindex_grid(std::move(axes));
    }
    template <std::size_t D = Dimension, std::enable_if_t<D == 2, int> = 0>
    void snap_to_grid(const vec<Scalar>& x, const vec<Scalar>& y) { snap_to_grid(grid_type{x, y}); }
    void snap_to_equidistant_grid(int n, bool end_inclusive = false) {
        if (n <= 0) return;
        auto box = bounding_box();
        grid_type axes;
        for (std::size_t a = 0; a < Dimension; ++a) {
            Scalar step = n == 1 ? Scalar{} : (box.second[a] - box.first[a]) / Scalar(end_inclusive ? n - 1 : n);
            for (int i = 0; i < n; ++i) axes[a].push_back(box.first[a] + Scalar(i) * step);
        }
        snap_to_grid(std::move(axes));
    }
    void cut_above(const std::array<double, Dimension>& cutoffs) {
        auto axes = grids;
        auto coordinates = to_coordinates();
        coordinates.cut_above(cutoffs);
        *this = Self(coordinates);
        reindex_grid(std::move(axes));
    }
    template <std::size_t D = Dimension, std::enable_if_t<D == 2, int> = 0>
    void cut_above(double x, double y) { cut_above(std::array<double, Dimension>{x, y}); }
    void bound_support(const point_type& bound) {
        include_real_degrees({bound});
        Base::bound_support(grid_degree(bound));
    }
    template <typename D, EnableGridDegree<D> = 0>
    void bound_support(const D& bound) { Base::bound_support(bound); }

    void set_real_degrees(const vec<real_degree_type>& columns, const vec<real_degree_type>& rows) {
        grid_type axes;
        for (const auto* degrees : {&columns, &rows})
            for (const auto& degree : *degrees)
                for (std::size_t a = 0; a < Dimension; ++a) axes[a].push_back(degree[a]);
        normalize_grids(axes);
        auto c = encode_degrees(columns, axes), r = encode_degrees(rows, axes);
        grids = std::move(axes);
        this->col_degrees = std::move(c);
        this->row_degrees = std::move(r);
        this->compatibly_sorted = false;
    }
    const vec<Scalar>& grid(std::size_t axis) const { return grids[axis]; }
    grid_type coordinate_grid() const { return grids; }
    real_degree_type real_degree(const degree_type& degree) const {
        real_degree_type result{};
        for (std::size_t a = 0; a < Dimension; ++a) result[a] = grids[a][degree[a]];
        return result;
    }
    real_degree_type real_col_degree(Index column) const { return real_degree(this->col_degrees[column]); }
    real_degree_type real_column_degree(Index column) const { return real_col_degree(column); }
    real_degree_type real_row_degree(Index row) const { return real_degree(this->row_degrees[row]); }
    vec<real_degree_type> real_col_degrees() const { return decode_degrees(this->col_degrees); }
    vec<real_degree_type> real_column_degrees() const { return real_col_degrees(); }
    vec<real_degree_type> real_row_degrees() const { return decode_degrees(this->row_degrees); }
    degree_type grid_degree(const real_degree_type& degree) const { return encode_degree(degree, grids); }
    degree_type query_degree(const real_degree_type& point) const {
        degree_type result{};
        for (std::size_t a = 0; a < Dimension; ++a)
            result[a] = static_cast<Index>(std::upper_bound(grids[a].begin(), grids[a].end(), point[a]) - grids[a].begin()) - 1;
        return result;
    }
    auto get_closest_smaller_grid_point(const real_degree_type& point) const {
        auto result = query_degree(point);
        if constexpr (Dimension == 2) return std::make_pair(result[0], result[1]);
        else return result;
    }
    bool same_row_degrees(const Self& other) const { return real_row_degrees() == other.real_row_degrees(); }
    void reindex_grid(grid_type axes) {
        normalize_grids(axes);
        auto c = encode_degrees(real_col_degrees(), axes), r = encode_degrees(real_row_degrees(), axes);
        grids = std::move(axes);
        this->col_degrees = std::move(c);
        this->row_degrees = std::move(r);
        this->pivots.clear();
    }
    template <std::size_t D = Dimension, std::enable_if_t<D == 2, int> = 0>
    void reindex_grid(const vec<Scalar>& x, const vec<Scalar>& y) { reindex_grid(grid_type{x, y}); }
    void rebuild_grid_maps() { this->pivots.clear(); }
    void include_real_degrees(const vec<real_degree_type>& degrees) {
        auto axes = grids;
        for (const auto& degree : degrees)
            for (std::size_t a = 0; a < Dimension; ++a) axes[a].push_back(degree[a]);
        reindex_grid(std::move(axes));
    }
    void merge_grids(Self& other) {
        if (grids == other.grids) return;
        auto axes = grids;
        for (std::size_t a = 0; a < Dimension; ++a)
            axes[a].insert(axes[a].end(), other.grids[a].begin(), other.grids[a].end());
        reindex_grid(axes);
        other.reindex_grid(std::move(axes));
    }
    void print_grid() const {
        for (std::size_t a = 0; a < Dimension; ++a) std::cout << "grid[" << a << "]: " << grids[a] << '\n';
    }
    void print_grid_representation() const {
        std::cout << "Grid Column Degrees: " << this->col_degrees << '\n';
        std::cout << "Grid Row Degrees: " << this->row_degrees << '\n';
    }
    void print_degrees() const { to_coordinates().print_degrees(); }
    void print_graded(bool suppress_description = false) const { to_coordinates().print_graded(suppress_description); }
    void parse_stream(std::istream& input, bool sort = false) {
        *this = Self(coordinate_matrix_type(input, sort));
    }
    template <typename Output>
    void to_stream(Output& output, bool header = true) const { to_coordinates().to_stream(output, header); }

private:
    template <typename OtherScalar>
    static vec<real_degree_type> convert_degrees(const vec<CoordinateDegree<OtherScalar, Dimension>>& degrees) {
        vec<real_degree_type> result(degrees.size());
        for (std::size_t i = 0; i < degrees.size(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) result[i][a] = static_cast<Scalar>(degrees[i][a]);
        return result;
    }
    static void normalize_grids(grid_type& axes) {
        for (auto& axis : axes) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
            if (axis.empty()) axis.push_back(Scalar{});
        }
    }
    static degree_type encode_degree(const real_degree_type& degree, const grid_type& axes) {
        degree_type result{};
        for (std::size_t a = 0; a < Dimension; ++a)
            result[a] = static_cast<Index>(std::lower_bound(axes[a].begin(), axes[a].end(), degree[a]) - axes[a].begin());
        return result;
    }
    static vec<degree_type> encode_degrees(const vec<real_degree_type>& degrees, const grid_type& axes) {
        vec<degree_type> result;
        result.reserve(degrees.size());
        for (const auto& degree : degrees) result.push_back(encode_degree(degree, axes));
        return result;
    }
    vec<real_degree_type> decode_degrees(const vec<degree_type>& degrees) const {
        vec<real_degree_type> result;
        result.reserve(degrees.size());
        for (const auto& degree : degrees) result.push_back(real_degree(degree));
        return result;
    }
    static Index count_before(const vec<degree_type>& degrees, const degree_type& point) {
        return static_cast<Index>(std::count_if(degrees.begin(), degrees.end(),
            [&](const auto& degree) { return Degree_traits<degree_type>::smaller_equal(degree, point); }));
    }
};

// Shared algorithms use physical degrees; grid matrices store axis indices.
template <typename Matrix, typename = void>
struct MatrixGeometryStorage {
    using degree_type = typename Matrix::degree_type;
    static constexpr bool grid_backed = false;
};

template <typename Matrix>
struct MatrixGeometryStorage<Matrix, std::void_t<decltype(Matrix::grid_backed), typename Matrix::real_degree_type>> {
    using degree_type = typename Matrix::real_degree_type;
    static constexpr bool grid_backed = Matrix::grid_backed;
};

// Keep degree aliases a substitution failure for non-matrix overload candidates.
template <typename Matrix, typename = void>
struct MatrixGeometry {};
template <typename Matrix>
struct MatrixGeometry<Matrix, std::void_t<typename Matrix::degree_type>> : MatrixGeometryStorage<Matrix> {};

template <typename Matrix>
using matrix_geometry_degree_t = typename MatrixGeometry<Matrix>::degree_type;
template <typename Matrix>
inline constexpr bool matrix_grid_backed_v = MatrixGeometry<Matrix>::grid_backed;


template <typename Matrix>
Matrix empty_matrix_like(const Matrix& reference, typename Matrix::index_type columns,
                         typename Matrix::index_type rows) {
    Matrix result = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return reference.empty_like(columns, rows);
        else return GradedMatrixIO<Matrix>::make_matrix(columns, rows);
    }();
    result.resize_data(static_cast<std::size_t>(columns));
    return result;
}

template <typename Matrix, typename Degree>
matrix_geometry_degree_t<Matrix> geometric_degree(const Matrix& matrix, const Degree& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_degree(degree);
    else return typename Matrix::degree_type(degree);
}

template <typename Matrix>
std::vector<matrix_geometry_degree_t<Matrix>> geometric_row_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_row_degrees();
    else return {matrix.row_degrees.begin(), matrix.row_degrees.end()};
}

template <typename Matrix>
std::vector<matrix_geometry_degree_t<Matrix>> geometric_col_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_col_degrees();
    else return {matrix.col_degrees.begin(), matrix.col_degrees.end()};
}

template <typename Matrix>
void set_geometric_degrees(Matrix& matrix,
                          const std::vector<matrix_geometry_degree_t<Matrix>>& columns,
                          const std::vector<matrix_geometry_degree_t<Matrix>>& rows) {
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto axes = matrix.grids;
        matrix.set_real_degrees(columns, rows);
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            axes[axis].insert(axes[axis].end(), matrix.grids[axis].begin(), matrix.grids[axis].end());
        matrix.reindex_grid(std::move(axes));
    }
    else {
        matrix.col_degrees = columns;
        matrix.row_degrees = rows;
    }
}

template <typename Matrix>
auto query_matrix_degree(const Matrix& matrix, const matrix_geometry_degree_t<Matrix>& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.query_degree(degree);
    else return degree;
}

template <typename Matrix>
bool same_geometric_rows(const Matrix& left, const Matrix& right) {
    if constexpr (matrix_grid_backed_v<Matrix>) return left.same_row_degrees(right);
    else return left.row_degrees == right.row_degrees;
}

} // namespace graded_linalg
