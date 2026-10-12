/** Coordinate-graded sparse matrices over product orders on R^n and Z^n. */
#pragma once

#include <grlina/deg_coordinate.hpp>
#include <grlina/mat_graded.hpp>
#include <grlina/mat_kernel.hpp>
#include <array>
#include <cstddef>
#include <iomanip>
#include <type_traits>
#include <utility>

namespace graded_linalg {

template <typename Scalar, std::size_t Dimension, typename index, typename Derived,
          typename Storage = vec<vec<index>>>
struct CoordinateGradedSparseMatrix
    : GradedSparseMatrix<CoordinateDegree<Scalar, Dimension>, index, Derived, Storage> {
    using degree_type = CoordinateDegree<Scalar, Dimension>;
    using Base = GradedSparseMatrix<degree_type, index, Derived, Storage>;
    using Base::Base;

    CoordinateGradedSparseMatrix() = default;
    explicit CoordinateGradedSparseMatrix(SparseMatrix<index, Storage>&& other)
        : Base(std::move(other)) {}

    void sort_rows_colexicographically() {
        this->sort_rows(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<index> sort_rows_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<degree_type, index>(
            this->row_degrees, Degree_traits<degree_type>::colex_lambda());
        vec<index> reverse(permutation.size());
        for (index i = 0; i < static_cast<index>(permutation.size()); ++i)
            reverse[permutation[i]] = i;
        this->transform_data(reverse);
        this->sort_data();
        this->invalidate_cached_rows();

        this->compatibly_sorted = this->col_degrees.size() <= 1;
        return permutation;
    }

    void sort_columns_colexicographically() {
        this->sort_columns(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }

    vec<index> sort_columns_colexicographically_with_output() {
        vec<index> permutation = sort_and_get_permutation<degree_type, index>(
            this->col_degrees, Degree_traits<degree_type>::colex_lambda());
        this->permute_columns(permutation);
        this->invalidate_cached_rows();

        this->compatibly_sorted = this->row_degrees.size() <= 1;
        return permutation;
    }

    void sort_colexicographically() {
        this->sort_compatibly(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::colex_lambda()});
    }
};

template <typename Scalar, std::size_t Dimension, typename Index>
class GridGradedSparseMatrix;

template <typename Scalar, std::size_t Dimension, typename Index, typename Derived = void>
class CoordGradedSparseMatrix
    : public CoordinateGradedSparseMatrix<Scalar, Dimension, Index,
          std::conditional_t<std::is_void_v<Derived>,
              CoordGradedSparseMatrix<Scalar, Dimension, Index>, Derived>> {
public:
    using Self = std::conditional_t<std::is_void_v<Derived>, CoordGradedSparseMatrix, Derived>;
    using Base = CoordinateGradedSparseMatrix<Scalar, Dimension, Index, Self>;
    using GradedBase = typename Base::Base;
    using degree_type = CoordinateDegree<Scalar, Dimension>;
    using grid_type = std::array<vec<Scalar>, Dimension>;
    static constexpr bool grid_backed = false;
    static constexpr std::size_t parameter_count() { return Dimension; }

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
    CoordGradedSparseMatrix(const GradedBase& matrix)
        : Base(matrix.get_num_cols(), matrix.get_num_rows(), matrix.data,
               matrix.col_degrees, matrix.row_degrees) {
        this->compatibly_sorted = matrix.compatibly_sorted;
        this->is_identity = matrix.is_identity;
    }
    CoordGradedSparseMatrix(GradedBase&& matrix) : CoordGradedSparseMatrix(matrix) {
        this->assign_data(std::move(matrix.data));
        this->is_identity = std::exchange(matrix.is_identity, false);
    }
    CoordGradedSparseMatrix& operator=(const GradedBase& matrix) {
        this->assign(matrix);
        return *this;
    }
    CoordGradedSparseMatrix& operator=(GradedBase&& matrix) {
        this->assign(std::move(matrix));
        return *this;
    }

    Self empty_like(Index columns, Index rows) const { return Self(columns, rows); }

    Self graded_kernel() {
        if constexpr (std::is_integral_v<Scalar>) {
            return integer_kernel();
        } else {
            GridGradedSparseMatrix<Scalar, Dimension, Index> grid(*this);
            auto kernel = grid.graded_kernel();
            this->assign_data(std::move(grid.data));
            this->col_degrees = grid.real_col_degrees();
            this->invalidate_cached_rows();
            this->compatibly_sorted = false;
            Self result = static_cast<const Self&>(*this).empty_like(kernel.get_num_cols(), kernel.get_num_rows());
            result.assign_data(std::move(kernel.data));
            result.col_degrees = kernel.real_col_degrees();
            result.row_degrees = kernel.real_row_degrees();
            result.compatibly_sorted = false;
            return result;
        }
    }

    Self graded_kernel() const {
        Self copy = static_cast<const Self&>(*this);
        return copy.graded_kernel();
    }

    grid_type coordinate_grid() const {
        grid_type axes;
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

    void snap_to_grid(const grid_type& axes) {
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
    void snap_to_grid(const vec<Scalar>& x, const vec<Scalar>& y) { snap_to_grid(grid_type{x, y}); }

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
    grid_type equidistant_axes(int n, bool inclusive) const {
        auto box = bounding_box();
        grid_type axes;
        for (std::size_t a = 0; a < Dimension; ++a) {
            Scalar step = n == 1 ? Scalar{} : (box.second[a] - box.first[a]) / Scalar(inclusive ? n - 1 : n);
            for (int i = 0; i < n; ++i) axes[a].push_back(box.first[a] + Scalar(i) * step);
        }
        return axes;
    }

    Self integer_kernel() {
        if constexpr (Dimension == 2) {
            return z2_graded_kernel(static_cast<Self&>(*this));
        } else if constexpr (Dimension == 3) {
            return z3_graded_kernel(static_cast<const Self&>(*this));
        } else {
            constexpr std::size_t frozen = Dimension > 2 ? Dimension - 2 : 0;
            Self result = static_cast<const Self&>(*this).empty_like(0, this->get_num_cols());
            result.row_degrees = this->col_degrees;
            std::array<vec<Scalar>, frozen> thresholds;
            for (const auto& degree : this->col_degrees)
                for (std::size_t a = 0; a < frozen; ++a) thresholds[a].push_back(degree[a]);
            for (auto& axis : thresholds) {
                std::sort(axis.begin(), axis.end());
                axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
            }
            auto generated = [&](vec<Index> column, const degree_type& degree) {
                SparseMatrix<Index> basis(result.get_num_cols() + 1, result.get_num_rows());
                Index count = 0;
                for (Index i = 0; i < result.get_num_cols(); ++i)
                    if (Degree_traits<degree_type>::smaller_equal(result.col_degrees[i], degree)) {
                        basis.append_col(result.get_col(i)); ++count;
                    }
                basis.append_col(column);
                basis.set_num_cols(count + 1);
                basis.column_reduction();
                return basis.is_zero(count);
            };
            degree_type grade{};
            auto run_slice = [&] {
                auto selected = columns_in_bigraded_slice<Index>(this->col_degrees, grade, frozen);
                if (selected.empty()) return;
                CoordGradedSparseMatrix<Scalar, 2, Index> slice(static_cast<Index>(selected.size()), this->get_num_rows());
                auto project = [&](const degree_type& original) {
                    CoordinateDegree<Scalar, 2> degree{};
                    for (std::size_t a = 0; a < 2; ++a) if (frozen + a < Dimension) degree[a] = original[frozen + a];
                    return degree;
                };
                for (Index i = 0; i < this->get_num_rows(); ++i) slice.row_degrees[i] = project(this->row_degrees[i]);
                for (Index i = 0; i < static_cast<Index>(selected.size()); ++i) {
                    slice.col_degrees[i] = project(this->col_degrees[selected[i]]);
                    slice.set_col(i, this->get_col(selected[i]));
                }
                auto kernel = slice.graded_kernel();
                for (Index i = 0; i < kernel.get_num_cols(); ++i) {
                    for (std::size_t a = 0; a < 2; ++a) if (frozen + a < Dimension) grade[frozen + a] = kernel.col_degrees[i][a];
                    auto column = kernel.get_col(i);
                    for (auto& row : column) row = selected[row];
                    if (!generated(column, grade)) static_cast<GradedBase&>(result).append_column(column, grade);
                }
            };
            auto enumerate = [&](auto&& self, std::size_t a) -> void {
                if (a == frozen) { run_slice(); return; }
                for (auto value : thresholds[a]) { grade[a] = value; self(self, a + 1); }
            };
            // ponytail: higher dimensions use up to n^(Dimension-2) slices; event scheduling can replace them if needed.
            enumerate(enumerate, 0);
            result.sort_columns_lexicographically();
            return result;
        }
    }
};

// Real coordinate matrices in two and three parameters.
template <typename Index = int>
using R2Matrix = CoordGradedSparseMatrix<double, 2, Index>;

template <typename Index = int>
using R3Matrix = CoordGradedSparseMatrix<double, 3, Index>;

} // namespace graded_linalg

// A real-coordinate kernel is computed on the integer degrees of this class.
#include <grlina/mat_grid.hpp>
