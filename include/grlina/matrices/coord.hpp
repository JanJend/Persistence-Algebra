/** Coordinate-graded sparse matrices over product orders on R^n and Z^n. */
#pragma once

#include <grlina/posets/coordinate.hpp>
#include <grlina/matrices/graded.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <type_traits>
#include <utility>

// Graded kernel method definitions: <grlina/graded_kernel.hpp>.

namespace graded_linalg {

template <typename Scalar, std::size_t Dimension, typename Index>
struct GridGradedSparseMatrix;

/** Sparse matrix with array-valued degrees in Scalar^Dimension, ordered coordinatewise.
 * Derived lets grid matrices keep their own return type in inherited operations.
 */
template <typename Scalar, std::size_t Dimension, typename Index, typename Derived = void>
struct CoordGradedSparseMatrix
    : GradedSparseMatrix<CoordinateDegree<Scalar, Dimension>, Index,
          std::conditional_t<std::is_void_v<Derived>,
              CoordGradedSparseMatrix<Scalar, Dimension, Index>, Derived>> {
    // Inherited attributes; this struct adds no stored data:
    // data holds the sparse columns; col_degrees and row_degrees hold their degrees.
    // Matrix dimensions, compatibly_sorted and reduction state also live in the base.

    using Self = std::conditional_t<std::is_void_v<Derived>, CoordGradedSparseMatrix, Derived>;
    using Base = GradedSparseMatrix<CoordinateDegree<Scalar, Dimension>, Index, Self>;
    using degree_type = CoordinateDegree<Scalar, Dimension>;
    using Grid = std::array<vec<Scalar>, Dimension>;

    using Base::Base;
    CoordGradedSparseMatrix() : Base(0, 0) {}
    CoordGradedSparseMatrix(Index columns, Index rows) : Base(columns, rows) {
        this->resize_data(columns);
    }
    CoordGradedSparseMatrix(Index columns, Index rows, vec<degree_type> c, vec<degree_type> r)
        : Base(columns, rows, std::move(c), std::move(r)) { this->resize_data(columns); }
    CoordGradedSparseMatrix(const CoordGradedSparseMatrix&) = default;
    CoordGradedSparseMatrix(CoordGradedSparseMatrix&&) = default;
    CoordGradedSparseMatrix& operator=(const CoordGradedSparseMatrix&) = default;
    CoordGradedSparseMatrix& operator=(CoordGradedSparseMatrix&&) = default;

    explicit CoordGradedSparseMatrix(SparseMatrix<Index>&& matrix) : Base(std::move(matrix)) {}
    explicit CoordGradedSparseMatrix(const SparseMatrix<Index>& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows()) {
        this->assign_data(matrix.data);
        this->is_identity = matrix.is_identity;
    }
    CoordGradedSparseMatrix(const Base& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows(), matrix.data,
               matrix.col_degrees, matrix.row_degrees) {
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    CoordGradedSparseMatrix(Base&& matrix) : CoordGradedSparseMatrix(matrix) {
        this->assign_data(std::move(matrix.data));
        this->is_identity = std::exchange(matrix.is_identity, false);
    }
    CoordGradedSparseMatrix& operator=(const Base& matrix) {
        this->assign(matrix);
        return *this;
    }
    CoordGradedSparseMatrix& operator=(Base&& matrix) {
        this->assign(std::move(matrix));
        return *this;
    }

    static constexpr std::size_t parameter_count() { return Dimension; }

    Self empty_like(Index columns, Index rows) const { return Self(columns, rows); }

    void sort_rows_colexicographically() {
        this->sort_rows(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<Index> sort_rows_colexicographically_with_output() {
        vec<Index> permutation = sort_and_get_permutation<degree_type, Index>(
            this->row_degrees, Degree_traits<degree_type>::colex_lambda());
        vec<Index> reverse(permutation.size());
        for (Index i = 0; i < static_cast<Index>(permutation.size()); ++i)
            reverse[permutation[i]] = i;
        this->transform_data(reverse);
        this->sort_data();

        this->compatibly_sorted = this->col_degrees.size() <= 1;
        return permutation;
    }

    void sort_columns_colexicographically() {
        this->sort_columns(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<Index> sort_columns_colexicographically_with_output() {
        vec<Index> permutation = sort_and_get_permutation<degree_type, Index>(
            this->col_degrees, Degree_traits<degree_type>::colex_lambda());
        this->permute_columns(permutation);

        this->compatibly_sorted = this->row_degrees.size() <= 1;
        return permutation;
    }

    void sort_colexicographically() {
        this->sort_compatibly(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    Self graded_kernel();
    Self graded_kernel() const;

    Grid coordinate_grid() const {
        Grid axes;
        for (const auto* degrees : {&this->col_degrees, &this->row_degrees})
            for (const auto& degree : *degrees)
                for (std::size_t a = 0; a < Dimension; ++a) axes[a].push_back(degree[a]);
        for (auto& axis : axes) {
            std::sort(axis.begin(), axis.end());
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        }
        return axes;
    }

    auto get_closest_smaller_grid_point(const degree_type& point) const {
        auto axes = coordinate_grid();
        CoordinateDegree<Index, Dimension> result{};
        for (std::size_t a = 0; a < Dimension; ++a)
            result[a] = static_cast<Index>(std::upper_bound(axes[a].begin(), axes[a].end(), point[a]) - axes[a].begin()) - 1;
        if constexpr (Dimension == 2) return std::make_pair(result[0], result[1]);
        else return result;
    }

    std::pair<degree_type, degree_type> bounding_box() const {
        degree_type lower, upper;
        lower.fill(std::numeric_limits<Scalar>::max());
        upper.fill(std::numeric_limits<Scalar>::lowest());
        for (const auto* degrees : {&this->col_degrees, &this->row_degrees})
            for (const auto& degree : *degrees) {
                lower = Degree_traits<degree_type>::meet(lower, degree);
                upper = Degree_traits<degree_type>::join(upper, degree);
            }
        return {lower, upper};
    }

    vec<degree_type> get_equidistant_grid(const int& n) const {
        vec<degree_type> result;
        if (n <= 0) return result;
        auto axes = equidistant_axes(n, true);
        degree_type point{};
        auto enumerate = [&](auto&& self, std::size_t a) -> void {
            if (a == Dimension) { result.push_back(point); return; }
            for (auto value : axes[a]) { point[a] = value; self(self, a + 1); }
        };
        enumerate(enumerate, 0);
        return result;
    }

    void snap_to_grid(const Grid& axes) {
        vec<Index> columns, rows;
        auto snap = [&](auto& degrees, auto& remove) {
            for (Index i = 0; i < static_cast<Index>(degrees.size()); ++i)
                for (std::size_t a = 0; a < Dimension; ++a) {
                    auto position = std::lower_bound(axes[a].begin(), axes[a].end(), degrees[i][a]);
                    if (position == axes[a].end()) { remove.push_back(i); break; }
                    degrees[i][a] = *position;
                }
        };
        this->compatibly_sorted = false;
        snap(this->col_degrees, columns);
        snap(this->row_degrees, rows);
        this->delete_columns(columns);
        this->delete_rows(rows);
        this->sort_compatibly();
        this->minimize();
    }

    template <std::size_t D = Dimension, std::enable_if_t<D == 2, int> = 0>
    void snap_to_grid(const vec<Scalar>& x, const vec<Scalar>& y) { snap_to_grid(Grid{x, y}); }

    void snap_to_equidistant_grid(int n, bool end_inclusive = false) {
        if (n > 0) snap_to_grid(equidistant_axes(n, end_inclusive));
    }

    void cut_above(const std::array<double, Dimension>& cutoffs) {
        auto box = bounding_box();
        auto outside = [&](const degree_type& degree) {
            for (std::size_t a = 0; a < Dimension; ++a)
                if (degree[a] > box.first[a] + cutoffs[a] * (box.second[a] - box.first[a])) return true;
            return false;
        };
        vec<Index> columns, rows;
        for (Index i = 0; i < this->get_num_cols(); ++i) if (outside(this->col_degrees[i])) columns.push_back(i);
        for (Index i = 0; i < this->get_num_rows(); ++i) if (outside(this->row_degrees[i])) rows.push_back(i);
        this->delete_columns(columns);
        this->delete_rows(rows);
    }

    template <std::size_t D = Dimension, std::enable_if_t<D == 2, int> = 0>
    void cut_above(double x, double y) { cut_above(std::array<double, Dimension>{x, y}); }

    void bound_support(const degree_type& bound) {
        vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i) {
            if (!Degree_traits<degree_type>::smaller_equal(this->row_degrees[i], bound)) rows.push_back(i);
            else for (std::size_t a = 0; a < Dimension; ++a) {
                auto degree = this->row_degrees[i];
                degree[a] = bound[a];
                this->append_column(vec<Index>{i}, degree);
            }
        }
        this->delete_rows(rows);
        this->sort_compatibly();
        this->minimize();
    }

    void print_grid() const {
        auto axes = coordinate_grid();
        for (std::size_t a = 0; a < Dimension; ++a) std::cout << "grid[" << a << "]: " << axes[a] << '\n';
    }

    void print_grid_representation() const {
        auto axes = coordinate_grid();
        auto print = [&](const auto& degrees) {
            for (const auto& degree : degrees) {
                CoordinateDegree<Index, Dimension> ordinal{};
                for (std::size_t a = 0; a < Dimension; ++a)
                    ordinal[a] = static_cast<Index>(std::lower_bound(axes[a].begin(), axes[a].end(), degree[a]) - axes[a].begin());
                std::cout << ordinal << ' ';
            }
            std::cout << '\n';
        };
        std::cout << "Grid Column Degrees: "; print(this->col_degrees);
        std::cout << "Grid Row Degrees: "; print(this->row_degrees);
    }

    template <typename Output>
    void to_stream(Output& output, bool header = true) const {
        output << std::defaultfloat << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
        if (header) output << "scc2020\n" << Degree_traits<degree_type>::poset_id << '\n'
                           << this->get_num_cols() << ' ' << this->get_num_rows() << " 0\n";
        for (Index i = 0; i < this->get_num_cols(); ++i) {
            Degree_traits<degree_type>::write_degree(output, this->col_degrees[i]);
            output << " ;";
            for (Index row : this->get_col(i)) output << ' ' << row;
            output << '\n';
        }
        for (const auto& degree : this->row_degrees) {
            Degree_traits<degree_type>::write_degree(output, degree);
            output << " ;\n";
        }
    }

    template <typename Output, std::size_t D = Dimension, std::enable_if_t<D == 3, int> = 0>
    void to_stream_r3(Output& output) const { static_cast<const Self&>(*this).to_stream(output); }

    friend CoordGradedSparseMatrix operator+(const CoordGradedSparseMatrix& left,
                                            const CoordGradedSparseMatrix& right) {
        CoordGradedSparseMatrix result(left.get_num_cols(), left.get_num_rows());
        result.assign_data((static_cast<const SparseMatrix<Index>&>(left) +
                            static_cast<const SparseMatrix<Index>&>(right)).data);
        result.col_degrees = left.col_degrees;
        result.row_degrees = left.row_degrees;
        result.compatibly_sorted = left.compatibly_sorted;
        return result;
    }

private:
    Grid equidistant_axes(int n, bool inclusive) const {
        auto box = bounding_box();
        Grid axes;
        for (std::size_t a = 0; a < Dimension; ++a) {
            Scalar step = n == 1 ? Scalar{} : (box.second[a] - box.first[a]) / Scalar(inclusive ? n - 1 : n);
            for (int i = 0; i < n; ++i) axes[a].push_back(box.first[a] + Scalar(i) * step);
        }
        return axes;
    }

};

// Real coordinate matrices in two and three parameters.
template <typename Index = int>
using R2Matrix = CoordGradedSparseMatrix<double, 2, Index>;

template <typename Index = int>
using R3Matrix = CoordGradedSparseMatrix<double, 3, Index>;

} // namespace graded_linalg

// A real-coordinate kernel is computed on the integer degrees of this class.
#include <grlina/matrices/grid.hpp>
