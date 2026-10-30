/** Integer-graded matrices with a product grid mapping degrees into real space. */
#pragma once

#include <grlina/matrices/coord.hpp>
#include <grlina/matrices/io.hpp>
#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Graded kernel method definitions: <grlina/graded_kernel.hpp>.
// Quiver method definitions: <grlina/quiver_representations.hpp>.

namespace graded_linalg {

// Put both matrices on the union grid, preserving their real degrees.
template <typename Scalar, std::size_t Dimension, typename Index, typename Storage>
void merge_grids(GridGradedSparseMatrix<Scalar, Dimension, Index, Storage>& left,
                 GridGradedSparseMatrix<Scalar, Dimension, Index, Storage>& right) {
    if (left.grid_ == right.grid_) return;
    auto axes = left.grid_;
    for (std::size_t a = 0; a < Dimension; ++a)
        axes[a].insert(axes[a].end(), right.grid_[a].begin(), right.grid_[a].end());
    left.reindex_grid(axes);
    right.reindex_grid(std::move(axes));
}

template <typename Scalar, std::size_t Dimension, typename Index, typename Storage>
struct GridGradedSparseMatrix
    : CoordGradedSparseMatrix<Index, Dimension, Index,
                             GridGradedSparseMatrix<Scalar, Dimension, Index, Storage>, Storage> {
    using Self = GridGradedSparseMatrix;
    using Base = CoordGradedSparseMatrix<Index, Dimension, Index, Self, Storage>;
    using D = CoordinateDegree<Index, Dimension>;
    using GridD = CoordinateDegree<Scalar, Dimension>;
    using degree_type = GridD;
    using Grid = std::array<vec<Scalar>, Dimension>;

    // D is an integer degree; GridD is its image under the grid.
    // grid_[a][d[a]] is the image of d[a] on axis a. Axes are strictly increasing.
    Grid grid_;

    // A zero on each axis gives default degree 0 an image under the grid.
    GridGradedSparseMatrix() { grid_.fill({Scalar{}}); }
    GridGradedSparseMatrix(const Self&) = default;
    GridGradedSparseMatrix(Self&&) = default;
    Self& operator=(const Self&) = default;
    Self& operator=(Self&&) = default;
    GridGradedSparseMatrix(Index columns, Index rows) : Base(columns, rows) { grid_.fill({Scalar{}}); }
    GridGradedSparseMatrix(Index columns, Index rows, const std::string& type, Index percent = -1)
        : Base(columns, rows, type, percent) { grid_.fill({Scalar{}}); }
    explicit GridGradedSparseMatrix(const vec<GridD>& degrees)
        : GridGradedSparseMatrix(static_cast<Index>(degrees.size()), static_cast<Index>(degrees.size()), "Identity") {
        set_real_degrees(degrees, degrees);
    }
    GridGradedSparseMatrix(Index rows, vec<Index> indicator) : Base(rows, std::move(indicator)) { grid_.fill({Scalar{}}); }
    explicit GridGradedSparseMatrix(SparseMatrix<Index, Storage>&& matrix) : Base(std::move(matrix)) { grid_.fill({Scalar{}}); }
    explicit GridGradedSparseMatrix(const SparseMatrix<Index, Storage>& matrix) : Base(matrix) { grid_.fill({Scalar{}}); }

    GridGradedSparseMatrix(Index columns, Index rows, vec<GridD> c, vec<GridD> r)
        : Base(columns, rows) { set_real_degrees(c, r); }
    GridGradedSparseMatrix(Index columns, Index rows, const array<Index>& entries,
                          vec<GridD> c, vec<GridD> r)
        : Base(columns, rows, entries, vec<D>(columns), vec<D>(rows)) { set_real_degrees(c, r); }
    GridGradedSparseMatrix(Index columns, Index rows, vec<D> c, vec<D> r, Grid axes)
        : Base(columns, rows, std::move(c), std::move(r)), grid_(std::move(axes)) {}
    GridGradedSparseMatrix(Index columns, Index rows, const array<Index>& entries,
                          vec<D> c, vec<D> r, Grid axes)
        : Base(columns, rows, entries, std::move(c), std::move(r)), grid_(std::move(axes)) {}

    template <typename OtherScalar, typename Derived>
    explicit GridGradedSparseMatrix(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived, Storage>& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows()) {
        this->assign_data(matrix.data);
        set_real_degrees(convert_degrees(matrix.col_degrees), convert_degrees(matrix.row_degrees));
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar, typename Derived>
    GridGradedSparseMatrix(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived, Storage>& matrix, Grid axes)
        : Base(matrix.get_num_cols(), matrix.get_num_rows()), grid_(std::move(axes)) {
        this->assign_data(matrix.data);
        for (Index i = 0; i < matrix.get_num_cols(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) this->col_degrees[i][a] = static_cast<Index>(matrix.col_degrees[i][a]);
        for (Index i = 0; i < matrix.get_num_rows(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) this->row_degrees[i][a] = static_cast<Index>(matrix.row_degrees[i][a]);
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar>
    explicit GridGradedSparseMatrix(const GridGradedSparseMatrix<OtherScalar, Dimension, Index, Storage>& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows(), matrix.col_degrees, matrix.row_degrees) {
        this->assign_data(matrix.data);
        for (std::size_t a = 0; a < Dimension; ++a) grid_[a].assign(matrix.grid_[a].begin(), matrix.grid_[a].end());
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    template <typename OtherScalar, typename Derived>
    Self& operator=(const GradedSparseMatrix<CoordinateDegree<OtherScalar, Dimension>, Index, Derived, Storage>& matrix) {
        return *this = Self(matrix);
    }
    template <typename OtherScalar>
    Self& operator=(const GridGradedSparseMatrix<OtherScalar, Dimension, Index, Storage>& matrix) { return *this = Self(matrix); }

    explicit GridGradedSparseMatrix(std::istream& input, bool sort = false)
        : GridGradedSparseMatrix(CoordGradedSparseMatrix<Scalar, Dimension, Index, void, Storage>(input, sort)) {}
    explicit GridGradedSparseMatrix(const std::string& path, bool sort = false)
        : GridGradedSparseMatrix(CoordGradedSparseMatrix<Scalar, Dimension, Index, void, Storage>(path, sort)) {}

    CoordGradedSparseMatrix<Scalar, Dimension, Index, void, Storage> to_coordinates() const {
        CoordGradedSparseMatrix<Scalar, Dimension, Index, void, Storage> result(this->get_num_cols(), this->get_num_rows(),
                                                               real_col_degrees(), real_row_degrees());
        result.assign_data(this->data);
        result.compatibly_sorted = this->compatibly_sorted;
        result.is_identity = this->is_identity;
        return result;
    }
    Self empty_like(Index columns, Index rows) const {
        Self result(columns, rows);
        result.grid_ = grid_;
        return result;
    }
    Self restricted_domain_copy(vec<Index> columns) const {
        Self result = Base::restricted_domain_copy(columns);
        result.grid_ = grid_;
        return result;
    }
    Self transposed_copy() const {
        Self result = Base::transposed_copy();
        result.grid_ = grid_;
        return result;
    }
    friend Self operator*(const Self& left, const Self& right) {
        GRLINA_ASSERT(left.real_col_degrees() == right.real_row_degrees());
        Self result(static_cast<const SparseMatrix<Index, Storage>&>(left) * static_cast<const SparseMatrix<Index, Storage>&>(right));
        result.grid_ = left.grid_;
        for (std::size_t a = 0; a < Dimension; ++a)
            result.grid_[a].insert(result.grid_[a].end(), right.grid_[a].begin(), right.grid_[a].end());
        normalize_grids(result.grid_);
        result.row_degrees = encode_degrees(left.real_row_degrees(), result.grid_);
        result.col_degrees = encode_degrees(right.real_col_degrees(), result.grid_);
        result.compatibly_sorted = false;
        return result;
    }
    friend Self operator+(const Self& left, const Self& right) {
        Self result(static_cast<const SparseMatrix<Index, Storage>&>(left) + static_cast<const SparseMatrix<Index, Storage>&>(right));
        result.grid_ = left.grid_;
        for (std::size_t a = 0; a < Dimension; ++a)
            result.grid_[a].insert(result.grid_[a].end(), right.grid_[a].begin(), right.grid_[a].end());
        normalize_grids(result.grid_);
        result.col_degrees = encode_degrees(left.real_col_degrees(), result.grid_);
        result.row_degrees = encode_degrees(left.real_row_degrees(), result.grid_);
        result.compatibly_sorted = left.compatibly_sorted;
        return result;
    }

    void append_matrix(const Self& other) {
        if (grid_ == other.grid_) { Base::append_matrix(other); return; }
        GRLINA_ASSERT(this->real_row_degrees() == other.real_row_degrees());
        auto axes = grid_;
        for (std::size_t a = 0; a < Dimension; ++a)
            axes[a].insert(axes[a].end(), other.grid_[a].begin(), other.grid_[a].end());
        reindex_grid(std::move(axes));
        for (Index j = 0; j < other.get_num_cols(); ++j)
            Base::append_column(other.column(j), grid_degree(other.real_col_degree(j)));
    }
    void append_move_matrix(Self&& other) {
        if (this == &other) { append_matrix(other); return; }
        merge_grids(*this, other);
        Base::append_move_matrix(std::move(other));
    }
    void append_column(const vec<Index>& column, const GridD& degree) {
        include_real_degrees({degree});
        Base::append_column(column, grid_degree(degree));
    }
    void append_column_at_grid_degree(const vec<Index>& column, const D& degree) {
        Base::append_column(column, degree);
    }

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
    Self submodule_generated_at(const GridD& degree) const {
        auto basis = basislift_at(degree);
        Self injection(this->get_num_rows(), basis);
        injection.grid_ = grid_;
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
    Self submodule_generated_at_grid_degree(const D& degree) const {
        return submodule_generated_at(real_degree(degree));
    }

    using Base::cancel_local_pairs;
    void cancel_local_pairs(Self* elements = nullptr, const vec<Self*>& images = {}) {
        if (elements) merge_grids(*this, *elements);
        Base::cancel_local_pairs(elements, images);
    }
    void shift(const GridD& amount) {
        for (std::size_t a = 0; a < Dimension; ++a)
            for (auto& coordinate : grid_[a]) coordinate -= amount[a];
    }
    void shift_generators(const GridD& amount) {
        auto rows = real_row_degrees();
        for (auto& degree : rows)
            for (std::size_t a = 0; a < Dimension; ++a) degree[a] -= amount[a];
        include_real_degrees(rows);
        this->row_degrees = encode_degrees(rows, grid_);
        this->compatibly_sorted = false;
    }
    void set_all_generator_degrees(const GridD& degree) {
        include_real_degrees({degree});
        Base::set_all_generator_degrees(grid_degree(degree));
    }
    void set_all_generator_degrees_at_grid_degree(const D& degree) {
        Base::set_all_generator_degrees(degree);
    }

    std::pair<SparseMatrix<Index, Storage>, vec<Index>> map_at_degree_pair(const GridD& degree, bool shifted = true) const {
        return Base::map_at_degree_pair(query_degree(degree), shifted);
    }
    SparseMatrix<Index, Storage> map_at_degree(const GridD& degree, vec<Index>& columns) const {
        return Base::map_at_degree(query_degree(degree), columns);
    }
    Index num_cols_before(const GridD& degree) const { return count_before(this->col_degrees, query_degree(degree)); }
    Index num_rows_before(const GridD& degree) const { return count_before(this->row_degrees, query_degree(degree)); }
    vec<Index> basislift_at(const GridD& degree) const { return Base::basislift_at(query_degree(degree)); }
    Index dim_at(const GridD& degree) const { return static_cast<Index>(basislift_at(degree).size()); }
    vec<Index> admissible_row_indices(const GridD& degree) const {
        vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i)
            if (is_admissible_row_operation(degree, i)) rows.push_back(i);
        return rows;
    }
    bool is_admissible_column_operation(Index i, Index j) const { return Base::is_admissible_column_operation(i, j); }
    bool is_admissible_column_operation(Index i, const GridD& degree) const {
        return Degree_traits<GridD>::smaller_equal(real_col_degree(i), degree);
    }
    bool is_admissible_row_operation(Index i, Index j) const { return Base::is_admissible_row_operation(i, j); }
    bool is_admissible_row_operation(Index i, const GridD& degree) const {
        return Degree_traits<GridD>::greater_equal(real_row_degree(i), degree);
    }
    bool is_admissible_row_operation(const GridD& degree, Index i) const {
        return Degree_traits<GridD>::greater_equal(degree, real_row_degree(i));
    }

    // Coordinate-valued methods above also accept braces. Grid indices are explicit.
    auto map_at_grid_degree_pair(const D& degree, bool shifted = true) const {
        return Base::map_at_degree_pair(degree, shifted);
    }
    auto map_at_grid_degree(const D& degree, vec<Index>& columns) const {
        return Base::map_at_degree(degree, columns);
    }
    Index num_cols_before_grid_degree(const D& degree) const {
        return count_before(this->col_degrees, degree);
    }
    Index num_rows_before_grid_degree(const D& degree) const {
        return count_before(this->row_degrees, degree);
    }
    vec<Index> basislift_at_grid_degree(const D& degree) const { return Base::basislift_at(degree); }
    Index dim_at_grid_degree(const D& degree) const {
        return static_cast<Index>(Base::basislift_at(degree).size());
    }
    vec<Index> admissible_row_indices_at_grid_degree(const D& degree) const {
        vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i)
            if (Base::is_admissible_row_operation(degree, i)) rows.push_back(i);
        return rows;
    }
    bool is_admissible_column_operation_at_grid_degree(Index i, const D& degree) const {
        return Base::is_admissible_column_operation(i, degree);
    }
    bool is_admissible_row_operation_at_grid_degree(Index i, const D& degree) const {
        return Base::is_admissible_row_operation(i, degree);
    }
    bool is_admissible_row_operation_at_grid_degree(const D& degree, Index i) const {
        return Base::is_admissible_row_operation(degree, i);
    }

    vec<GridD> discrete_support() const { return decode_degrees(Base::discrete_support()); }
    auto induced_quiver_rep(vec<GridD> vertices = {}, array<Index> edges = {});
    std::pair<GridD, GridD> bounding_box() const { return to_coordinates().bounding_box(); }
    vec<GridD> get_equidistant_grid(const int& n) const { return to_coordinates().get_equidistant_grid(n); }
    // Coordinate conversion must carry outgoing images as well as the presentation.
    template <typename Operation>
    void apply_in_coordinates(Operation operation, const vec<Self*>& images) {
        auto coordinates = to_coordinates();
        using Coordinates = decltype(coordinates);
        vec<Coordinates> image_copies;
        image_copies.reserve(images.size());
        for (auto* image : images) if (image) image_copies.push_back(image->to_coordinates());
        vec<Coordinates*> pointers;
        for (auto& image : image_copies) pointers.push_back(&image);
        operation(coordinates, pointers);
        *this = Self(coordinates);
        std::size_t i = 0;
        for (auto* image : images) if (image) *image = Self(image_copies[i++]);
    }
    void snap_to_grid(Grid axes, const vec<Self*>& images = {}) {
        apply_in_coordinates([&](auto& coordinates, const auto& outgoing) {
            coordinates.snap_to_grid(axes, outgoing);
        }, images);
        reindex_grid(std::move(axes));
    }
    template <std::size_t N = Dimension, std::enable_if_t<N == 2, int> = 0>
    void snap_to_grid(const vec<Scalar>& x, const vec<Scalar>& y, const vec<Self*>& images = {}) {
        snap_to_grid(Grid{x, y}, images);
    }
    void snap_to_equidistant_grid(int n, bool end_inclusive = false, const vec<Self*>& images = {}) {
        if (n <= 0) return;
        auto box = bounding_box();
        Grid axes;
        for (std::size_t a = 0; a < Dimension; ++a) {
            Scalar step = n == 1 ? Scalar{} : (box.second[a] - box.first[a]) / Scalar(end_inclusive ? n - 1 : n);
            for (int i = 0; i < n; ++i) axes[a].push_back(box.first[a] + Scalar(i) * step);
        }
        snap_to_grid(std::move(axes), images);
    }
    void cut_above(const std::array<double, Dimension>& cutoffs, const vec<Self*>& images = {}) {
        auto axes = grid_;
        apply_in_coordinates([&](auto& coordinates, const auto& outgoing) {
            coordinates.cut_above(cutoffs, outgoing);
        }, images);
        reindex_grid(std::move(axes));
    }
    template <std::size_t N = Dimension, std::enable_if_t<N == 2, int> = 0>
    void cut_above(double x, double y, const vec<Self*>& images = {}) {
        cut_above(std::array<double, Dimension>{x, y}, images);
    }
    void bound_support(const GridD& bound, const vec<Self*>& images = {}) {
        include_real_degrees({bound});
        Base::bound_support(grid_degree(bound), images);
    }
    void bound_support_at_grid_degree(const D& bound, const vec<Self*>& images = {}) {
        Base::bound_support(bound, images);
    }

    void set_real_degrees(const vec<GridD>& columns, const vec<GridD>& rows) {
        Grid axes;
        for (const auto* degrees : {&columns, &rows})
            for (const auto& degree : *degrees)
                for (std::size_t a = 0; a < Dimension; ++a) axes[a].push_back(degree[a]);
        normalize_grids(axes);
        auto c = encode_degrees(columns, axes), r = encode_degrees(rows, axes);
        grid_ = std::move(axes);
        this->col_degrees = std::move(c);
        this->row_degrees = std::move(r);
        this->compatibly_sorted = false;
    }
    const vec<Scalar>& grid(std::size_t axis) const { return grid_[axis]; }
    Grid coordinate_grid() const { return grid_; }
    GridD real_degree(const D& degree) const {
        GridD result{};
        for (std::size_t a = 0; a < Dimension; ++a) result[a] = grid_[a][degree[a]];
        return result;
    }
    /** Actual coordinates; row_degrees and col_degrees continue to store indices. */
    degree_type row_degree(Index row) const { return real_degree(this->row_degrees[row]); }
    degree_type col_degree(Index column) const { return real_degree(this->col_degrees[column]); }
    GridD real_col_degree(Index column) const { return col_degree(column); }
    GridD real_column_degree(Index column) const { return real_col_degree(column); }
    GridD real_row_degree(Index row) const { return row_degree(row); }
    vec<GridD> real_col_degrees() const { return decode_degrees(this->col_degrees); }
    vec<GridD> real_column_degrees() const { return real_col_degrees(); }
    vec<GridD> real_row_degrees() const { return decode_degrees(this->row_degrees); }
    D grid_degree(const GridD& degree) const { return encode_degree(degree, grid_); }
    D query_degree(const GridD& point) const {
        D result{};
        for (std::size_t a = 0; a < Dimension; ++a)
            result[a] = static_cast<Index>(std::upper_bound(grid_[a].begin(), grid_[a].end(), point[a]) - grid_[a].begin()) - 1;
        return result;
    }
    auto get_closest_smaller_grid_point(const GridD& point) const {
        auto result = query_degree(point);
        if constexpr (Dimension == 2) return std::make_pair(result[0], result[1]);
        else return result;
    }
    bool same_row_degrees(const Self& other) const { return real_row_degrees() == other.real_row_degrees(); }
    void reindex_grid(Grid axes) {
        normalize_grids(axes);
        auto c = encode_degrees(real_col_degrees(), axes), r = encode_degrees(real_row_degrees(), axes);
        grid_ = std::move(axes);
        this->col_degrees = std::move(c);
        this->row_degrees = std::move(r);
        this->pivots.clear();
    }
    template <std::size_t N = Dimension, std::enable_if_t<N == 2, int> = 0>
    void reindex_grid(const vec<Scalar>& x, const vec<Scalar>& y) { reindex_grid(Grid{x, y}); }
    void rebuild_grid_maps() { this->pivots.clear(); }
    void include_real_degrees(const vec<GridD>& degrees) {
        auto axes = grid_;
        for (const auto& degree : degrees)
            for (std::size_t a = 0; a < Dimension; ++a) axes[a].push_back(degree[a]);
        reindex_grid(std::move(axes));
    }
    void print_grid() const {
        for (std::size_t a = 0; a < Dimension; ++a) std::cout << "grid[" << a << "]: " << grid_[a] << '\n';
    }
    void print_grid_representation() const {
        std::cout << "Grid Column Degrees: " << this->col_degrees << '\n';
        std::cout << "Grid Row Degrees: " << this->row_degrees << '\n';
    }
    void print_degrees() const { to_coordinates().print_degrees(); }
    void print_graded(bool suppress_description = false) const { to_coordinates().print_graded(suppress_description); }
    void parse_stream(std::istream& input, bool sort = false) {
        *this = Self(CoordGradedSparseMatrix<Scalar, Dimension, Index, void, Storage>(input, sort));
    }
    template <typename Output>
    void to_stream(Output& output, bool header = true) const { to_coordinates().to_stream(output, header); }

private:
    template <typename OtherScalar>
    static vec<GridD> convert_degrees(const vec<CoordinateDegree<OtherScalar, Dimension>>& degrees) {
        vec<GridD> result(degrees.size());
        for (std::size_t i = 0; i < degrees.size(); ++i)
            for (std::size_t a = 0; a < Dimension; ++a) result[i][a] = static_cast<Scalar>(degrees[i][a]);
        return result;
    }
    static void normalize_grids(Grid& axes) {
        for (auto& axis : axes) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
            if (axis.empty()) axis.push_back(Scalar{});
        }
    }
    static D encode_degree(const GridD& degree, const Grid& axes) {
        D result{};
        for (std::size_t a = 0; a < Dimension; ++a)
            result[a] = static_cast<Index>(std::lower_bound(axes[a].begin(), axes[a].end(), degree[a]) - axes[a].begin());
        return result;
    }
    static vec<D> encode_degrees(const vec<GridD>& degrees, const Grid& axes) {
        vec<D> result;
        result.reserve(degrees.size());
        for (const auto& degree : degrees) result.push_back(encode_degree(degree, axes));
        return result;
    }
    vec<GridD> decode_degrees(const vec<D>& degrees) const {
        vec<GridD> result;
        result.reserve(degrees.size());
        for (const auto& degree : degrees) result.push_back(real_degree(degree));
        return result;
    }
    static Index count_before(const vec<D>& degrees, const D& point) {
        return static_cast<Index>(std::count_if(degrees.begin(), degrees.end(),
            [&](const auto& degree) { return Degree_traits<D>::smaller_equal(degree, point); }));
    }
};

// Free helpers for algorithms that accept both grid and ordinary graded matrices.
// "Geometric" means the actual degree: decode grid indices, or use an ordinary
// matrix's stored degree unchanged. For example, a grid matrix can store {2, 1}
// for the actual degree {10.0, 7.0}. Indices from different grids cannot be compared
// directly, so shared algorithms (e.g. the graded linear solver) use these helpers.

// Create a zero matrix of the requested shape, preserving the reference grid.
template <typename Matrix>
Matrix empty_matrix_like(const Matrix& reference, matrix_index_t<Matrix> columns,
                         matrix_index_t<Matrix> rows) {
    Matrix result = [&] {
        if constexpr (matrix_grid_backed_v<Matrix>) return reference.empty_like(columns, rows);
        else return GradedMatrixIO<Matrix>::make_matrix(columns, rows);
    }();
    result.resize_data(static_cast<std::size_t>(columns));
    return result;
}

// Convert one stored degree to its actual value.
template <typename Matrix, typename Degree>
typename Matrix::degree_type geometric_degree(const Matrix& matrix, const Degree& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_degree(degree);
    else return base_degree<Matrix>(degree);
}

// Return a copy of all row degrees as actual values.
template <typename Matrix>
std::vector<typename Matrix::degree_type> geometric_row_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_row_degrees();
    else return {matrix.row_degrees.begin(), matrix.row_degrees.end()};
}

// Return a copy of all column degrees as actual values.
template <typename Matrix>
std::vector<typename Matrix::degree_type> geometric_col_degrees(const Matrix& matrix) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.real_col_degrees();
    else return {matrix.col_degrees.begin(), matrix.col_degrees.end()};
}

// Assign actual degrees, encoding them as indices for a grid matrix.
// Keep existing grid points as well as any new points needed by these degrees.
template <typename Matrix>
void set_geometric_degrees(Matrix& matrix,
                          const std::vector<typename Matrix::degree_type>& columns,
                          const std::vector<typename Matrix::degree_type>& rows) {
    if constexpr (matrix_grid_backed_v<Matrix>) {
        auto axes = matrix.grid_;
        matrix.set_real_degrees(columns, rows);
        for (std::size_t axis = 0; axis < axes.size(); ++axis)
            axes[axis].insert(axes[axis].end(), matrix.grid_[axis].begin(), matrix.grid_[axis].end());
        matrix.reindex_grid(std::move(axes));
    }
    else {
        matrix.col_degrees = columns;
        matrix.row_degrees = rows;
    }
}

// Convert an actual query point to stored coordinates. On a grid, each coordinate
// becomes the last axis index at or below the point (-1 below the first point).
template <typename Matrix>
auto query_matrix_degree(const Matrix& matrix, const typename Matrix::degree_type& degree) {
    if constexpr (matrix_grid_backed_v<Matrix>) return matrix.query_degree(degree);
    else return degree;
}

// Compare actual row degrees, even when the matrices use different grids.
template <typename Matrix>
bool same_geometric_rows(const Matrix& left, const Matrix& right) {
    if constexpr (matrix_grid_backed_v<Matrix>) return left.same_row_degrees(right);
    else return left.row_degrees == right.row_degrees;
}

} // namespace graded_linalg
