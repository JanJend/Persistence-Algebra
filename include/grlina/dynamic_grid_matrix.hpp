/** @file dynamic_grid_matrix.hpp
 * @brief NEW: runtime-dimensional, index-graded sparse matrices with a real grid.
 *
 * Degrees occupy two contiguous buffers of Index. Each grid axis owns its
 * scalar coordinates. Entry storage remains a separate template parameter.
 */
#pragma once
#include <grlina/dynamic_coordinate_matrix.hpp>
#include <grlina/csc_matrix.hpp>
#include <grlina/grid_scheduler.hpp>
#include <map>
#include <queue>

namespace graded_linalg {

template<class Scalar, class Index, class Storage = vec<vec<Index>>>
class DynamicGridGradedSparseMatrix : public DynamicCoordinateMatrixBase<Index, Index,
        DynamicGridGradedSparseMatrix<Scalar, Index, Storage>, Storage> {
    using Self = DynamicGridGradedSparseMatrix;
    using Core = DynamicCoordinateMatrixBase<Index, Index, Self, Storage>;
    using DT = Degree_traits<DynamicDegree<Index>>;
    using RealDT = Degree_traits<DynamicDegree<Scalar>>;
    template<class D> using EnableIndexDegree = std::enable_if_t<
        std::is_same_v<std::decay_t<D>, DynamicDegree<Index>> && !std::is_same_v<Scalar, Index>, int>;

    static std::size_t infer(const vec<DynamicDegree<Scalar>>& columns,
                             const vec<DynamicDegree<Scalar>>& rows) {
        if (!columns.empty()) return columns.front().size();
        if (!rows.empty()) return rows.front().size();
        return 2; // Historical two-parameter constructor, explicit dimensions remain available.
    }
    static void sort_unique(vec<Scalar>& axis) {
        dynamic_coordinate_detail::validate_coordinates(axis);
        std::sort(axis.begin(), axis.end());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
        if (axis.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
            throw std::length_error("Grid axis exceeds the index type");
    }
    void check_real(const DynamicDegree<Scalar>& d) const {
        if (d.size() != this->parameter_count()) throw std::invalid_argument("Wrong grid dimension");
        dynamic_coordinate_detail::validate_coordinates(d);
    }
    void require_same_rows(const Self& other) const {
        if (!same_row_degrees(other)) throw std::invalid_argument("Different graded generator bases");
    }
public:
    using scalar_type = Scalar;
    using degree_type = DynamicDegree<Index>;
    using real_degree_type = DynamicDegree<Scalar>;
    using index_type = Index;
    using sparse_matrix_type = SparseMatrix<Index, Storage>;
    static constexpr bool grid_backed = true;
    static constexpr bool runtime_dimension = true;
    vec<vec<Scalar>> grids;

    DynamicGridGradedSparseMatrix() : DynamicGridGradedSparseMatrix(0, 0, 0) {}
    DynamicGridGradedSparseMatrix(Index columns, Index rows, std::size_t parameters = 2)
        : Core(columns, rows, parameters), grids(parameters, vec<Scalar>{Scalar{}}) {}
    DynamicGridGradedSparseMatrix(Index columns, Index rows, std::size_t parameters,
            const array<Index>& entries, vec<real_degree_type> degrees, vec<real_degree_type> generators)
        : DynamicGridGradedSparseMatrix(columns, rows, parameters) {
        this->assign_data(entries); set_real_degrees(degrees, generators); validate();
    }
    DynamicGridGradedSparseMatrix(Index columns, Index rows, const array<Index>& entries,
            vec<real_degree_type> degrees, vec<real_degree_type> generators)
        : DynamicGridGradedSparseMatrix(columns, rows, infer(degrees, generators), entries, degrees, generators) {}
    DynamicGridGradedSparseMatrix(Index columns, Index rows,
            vec<real_degree_type> degrees, vec<real_degree_type> generators)
        : DynamicGridGradedSparseMatrix(columns, rows, infer(degrees, generators)) {
        set_real_degrees(degrees, generators);
    }
    explicit DynamicGridGradedSparseMatrix(const DynamicCoordinateGradedSparseMatrix<Scalar, Index, Storage>& other)
        : DynamicGridGradedSparseMatrix(other.get_num_cols(), other.get_num_rows(), other.parameter_count()) {
        this->assign_data(other.data);
        set_real_degrees(other.col_degrees.to_vector(), other.row_degrees.to_vector()); validate();
    }
    template<class Matrix, class = decltype(std::declval<const Matrix&>().col_degrees),
             std::enable_if_t<!std::is_base_of_v<Self, Matrix>, int> = 0>
    explicit DynamicGridGradedSparseMatrix(const Matrix& other)
        : DynamicGridGradedSparseMatrix(other.get_num_cols(), other.get_num_rows(), 2) {
        this->assign_data(other.data);
        vec<real_degree_type> columns, rows;
        for (auto d : other.col_degrees) columns.emplace_back(d);
        for (auto d : other.row_degrees) rows.emplace_back(d);
        set_real_degrees(columns, rows); validate();
    }
    explicit DynamicGridGradedSparseMatrix(std::istream& input, bool sort = false) {
        auto complex = ChainComplex<Self>::from_stream(input, sort);
        if (complex.size() != 1) throw std::runtime_error("Expected one presentation differential");
        *this = std::move(complex[0]);
    }
    explicit DynamicGridGradedSparseMatrix(const std::string& path, bool sort = false) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot open SCC file: " + path);
        auto complex = ChainComplex<Self>::from_stream(input, sort);
        if (complex.size() != 1) throw std::runtime_error("Expected one presentation differential");
        *this = std::move(complex[0]);
    }
    Self empty_like(Index columns, Index rows) const {
        Self result(columns, rows, this->parameter_count()); result.grids = grids; return result;
    }
    template<class TargetStorage>
    DynamicGridGradedSparseMatrix<Scalar, Index, TargetStorage> to_storage() const {
        auto result = this->template copy_storage_as<DynamicGridGradedSparseMatrix<Scalar, Index, TargetStorage>>();
        result.grids = grids;
        return result;
    }
    DynamicGridGradedSparseMatrix<Scalar, Index> editable_copy() const {
        return to_storage<vec<vec<Index>>>();
    }
    const vec<Scalar>& grid(std::size_t axis) const { return grids.at(axis); }
    void copy_embedding(const Self& other) {
        if (this->parameter_count() != other.parameter_count()) throw std::invalid_argument("Different grid dimensions");
        grids = other.grids;
    }
    std::string poset_identifier() const { return RealDT::poset_identifier(this->parameter_count()); }
    void validate_index_degree(const degree_type& degree) const {
        Core::check_degree(degree);
        for (std::size_t a = 0; a < degree.size(); ++a)
            dynamic_coordinate_detail::checked_index(degree[a], grids.at(a).size());
    }
    void validate() const {
        if (grids.size() != this->parameter_count()) throw std::invalid_argument("Wrong number of grid axes");
        for (const auto& axis : grids) {
            dynamic_coordinate_detail::validate_coordinates(axis);
            if (!std::is_sorted(axis.begin(), axis.end()) || std::adjacent_find(axis.begin(), axis.end()) != axis.end())
                throw std::invalid_argument("Grid axes must be strictly increasing");
        }
        for (auto d : this->col_degrees) validate_index_degree(d);
        for (auto d : this->row_degrees) validate_index_degree(d);
        Core::validate();
    }
    void set_row_degree(Index row, const degree_type& d) { validate_index_degree(d); Core::set_row_degree(row, d); }
    void set_col_degree(Index column, const degree_type& d) { validate_index_degree(d); Core::set_col_degree(column, d); }
    template<class Range> real_degree_type real_degree(const Range& d) const {
        if (d.size() != this->parameter_count()) throw std::invalid_argument("Wrong grid dimension");
        real_degree_type result(this->parameter_count());
        for (std::size_t a = 0; a < d.size(); ++a)
            result[a] = grids.at(a).at(dynamic_coordinate_detail::checked_index(d[a], grids.at(a).size()));
        return result;
    }
    real_degree_type real_row_degree(Index i) const { return real_degree(this->row_degree(i)); }
    real_degree_type real_col_degree(Index i) const { return real_degree(this->col_degree(i)); }
    real_degree_type real_column_degree(Index i) const { return real_col_degree(i); }
    vec<real_degree_type> real_row_degrees() const {
        vec<real_degree_type> result; result.reserve(this->row_degrees.size());
        for (auto d : this->row_degrees) result.push_back(real_degree(d)); return result;
    }
    vec<real_degree_type> real_col_degrees() const {
        vec<real_degree_type> result; result.reserve(this->col_degrees.size());
        for (auto d : this->col_degrees) result.push_back(real_degree(d)); return result;
    }
    vec<real_degree_type> real_column_degrees() const { return real_col_degrees(); }
    bool same_row_degrees(const Self& other) const {
        return this->parameter_count() == other.parameter_count() && real_row_degrees() == other.real_row_degrees();
    }
    degree_type grid_degree(const real_degree_type& d) const {
        check_real(d); degree_type result(d.size());
        for (std::size_t a = 0; a < d.size(); ++a) {
            auto it = std::lower_bound(grids[a].begin(), grids[a].end(), d[a]);
            if (it == grids[a].end() || *it != d[a]) throw std::out_of_range("Coordinate is absent from grid");
            result[a] = static_cast<Index>(it - grids[a].begin());
        }
        return result;
    }
    degree_type query_degree(const real_degree_type& d) const {
        check_real(d); degree_type result(d.size());
        for (std::size_t a = 0; a < d.size(); ++a)
            result[a] = static_cast<Index>(std::upper_bound(grids[a].begin(), grids[a].end(), d[a]) - grids[a].begin()) - 1;
        return result;
    }
    void set_real_degrees(const vec<real_degree_type>& columns, const vec<real_degree_type>& rows) {
        if (columns.size() != static_cast<std::size_t>(this->get_num_cols()) || rows.size() != static_cast<std::size_t>(this->get_num_rows()))
            throw std::invalid_argument("Degree counts differ from matrix ranks");
        vec<vec<Scalar>> axes(this->parameter_count());
        for (const auto* degrees : {&columns, &rows}) for (const auto& d : *degrees) {
            check_real(d); for (std::size_t a = 0; a < d.size(); ++a) axes[a].push_back(d[a]);
        }
        for (auto& axis : axes) { sort_unique(axis); if (axis.empty()) axis.push_back(Scalar{}); }
        auto old = grids; grids = std::move(axes);
        try {
            vec<degree_type> c, r;
            for (const auto& d : columns) c.push_back(grid_degree(d));
            for (const auto& d : rows) r.push_back(grid_degree(d));
            this->col_degrees = c; this->row_degrees = r;
        } catch (...) { grids = std::move(old); throw; }
        this->refresh_compatible_sorted(); this->invalidate_cached_rows();
    }
    void reindex_grid(vec<vec<Scalar>> axes) {
        if (axes.size() != grids.size()) throw std::invalid_argument("Different grid dimensions");
        auto columns = real_col_degrees(), rows = real_row_degrees();
        for (std::size_t a = 0; a < axes.size(); ++a) {
            sort_unique(axes[a]);
            for (Scalar value : grids[a]) if (!std::binary_search(axes[a].begin(), axes[a].end(), value))
                throw std::invalid_argument("Reindexing must preserve every grid point");
        }
        grids = std::move(axes);
        for (std::size_t i = 0; i < columns.size(); ++i) this->col_degrees.set(i, grid_degree(columns[i]));
        for (std::size_t i = 0; i < rows.size(); ++i) this->row_degrees.set(i, grid_degree(rows[i]));
        this->invalidate_cached_rows();
    }
    void include_real_degrees(const vec<real_degree_type>& degrees) {
        auto axes = grids;
        for (const auto& d : degrees) { check_real(d); for (std::size_t a = 0; a < d.size(); ++a) axes[a].push_back(d[a]); }
        reindex_grid(std::move(axes));
    }
    void merge_grids(Self& other) {
        if (grids.size() != other.grids.size()) throw std::invalid_argument("Different grid dimensions");
        auto axes = grids;
        for (std::size_t a = 0; a < axes.size(); ++a) axes[a].insert(axes[a].end(), other.grids[a].begin(), other.grids[a].end());
        reindex_grid(axes); other.reindex_grid(std::move(axes));
    }
    using Core::map_at_degree_pair;
    using Core::map_at_degree;
    using Core::basislift_at;
    using Core::dim_at;
    template<class D = Scalar, std::enable_if_t<!std::is_same_v<D, Index>, int> = 0>
    auto map_at_degree_pair(const real_degree_type& d, bool shifted = true) const { return Core::map_at_degree_pair(query_degree(d), shifted); }
    template<class D = Scalar, std::enable_if_t<!std::is_same_v<D, Index>, int> = 0>
    auto map_at_degree(const real_degree_type& d, vec<Index>& selected) const { return Core::map_at_degree(query_degree(d), selected); }
    template<class D = Scalar, std::enable_if_t<!std::is_same_v<D, Index>, int> = 0>
    auto basislift_at(const real_degree_type& d) const { return Core::basislift_at(query_degree(d)); }
    template<class D = Scalar, std::enable_if_t<!std::is_same_v<D, Index>, int> = 0>
    auto dim_at(const real_degree_type& d) const { return Core::dim_at(query_degree(d)); }
    void append_column_at_grid_degree(const vec<Index>& entries, const degree_type& d) { validate_index_degree(d); Core::append_column(entries, d); }
    void append_column(const vec<Index>& entries, const real_degree_type& d) { include_real_degrees({d}); append_column_at_grid_degree(entries, grid_degree(d)); }
    template<class D, EnableIndexDegree<D> = 0>
    void append_column(const vec<Index>& entries, const D& d) { append_column_at_grid_degree(entries, d); }
    void append_matrix(const Self& other) { require_same_rows(other); Self aligned = other; merge_grids(aligned); Core::append_matrix(aligned); }
    void append_move_matrix(Self&& other) { append_matrix(other); }
    vec<Index> compute_grid_representation() { return this->sort_columns_colexicographically_with_output(); }
    Self graded_kernel();
    /** The same kernel routine on a copy; the non-const overload may reduce this matrix. */
    Self graded_kernel() const { Self copy = *this; return copy.graded_kernel(); }
    Self submodule_generated_by(const Self& generators) const {
        require_same_rows(generators); Self injection = generators; Index n = injection.get_num_cols();
        injection.append_matrix(*this); auto result = injection.graded_kernel(); result.cull_columns(n, false); return result;
    }
    Self inverse_image_copy(const Self& presentation, const Self& submodule) const {
        Self injection = *this; Index n = injection.get_num_cols();
        injection.append_matrix(submodule); injection.append_matrix(presentation);
        auto result = injection.graded_kernel(); result.cull_columns(n, false); return result;
    }
    Self submodule_intersection(const Self& left, const Self& right) const {
        require_same_rows(left); require_same_rows(right);
        auto kernel = left.inverse_image_copy(*this, right);
        return left * kernel;
    }
    Self presentation_of_submodule(const Self& presentation) {
        Index n = this->get_num_cols(); append_matrix(presentation); auto result = graded_kernel(); result.cull_columns(n, false); return result;
    }
    void quotient_by(Self& other) { append_matrix(other); this->sort_compatibly(); this->minimize(); }
    void cancel_local_pairs(Self* elements = nullptr) {
        if (elements) {
            if (elements == this) throw std::invalid_argument("Elements must be a separate matrix");
            require_same_rows(*elements); merge_grids(*elements);
        }
        Core::cancel_local_pairs(elements);
    }
    Self quotient_by_copy(Self& other) const { Self result = *this; result.quotient_by(other); return result; }
    Self submodule_generated_at(const real_degree_type& alpha) const {
        auto basis = Core::basislift_at(query_degree(alpha)); Self injection = empty_like(static_cast<Index>(basis.size()), this->get_num_rows());
        injection.row_degrees = this->row_degrees; injection.include_real_degrees({alpha});
        vec<degree_type> degrees(basis.size(), injection.grid_degree(alpha)); injection.col_degrees = degrees;
        for (Index i = 0; i < static_cast<Index>(basis.size()); ++i) injection.set_col(i, {basis[i]});
        injection.append_matrix(*this); auto result = injection.graded_kernel(); result.cull_columns(static_cast<Index>(basis.size()), false);
        result.sort_compatibly(); result.minimize(); return result;
    }
    template<class D, EnableIndexDegree<D> = 0>
    Self submodule_generated_at(const D& d) const { return submodule_generated_at(real_degree(d)); }
    void shift(const real_degree_type& amount) {
        check_real(amount); auto axes = grids;
        for (std::size_t a = 0; a < axes.size(); ++a) {
            for (auto& value : axes[a]) value = dynamic_coordinate_detail::difference(value, amount[a]);
            if (std::adjacent_find(axes[a].begin(), axes[a].end(), std::greater_equal<Scalar>{}) != axes[a].end())
                throw std::overflow_error("Grid translation cannot preserve distinct coordinates in this scalar type");
        }
        grids = std::move(axes);
    }
    void shift_generators(const real_degree_type& amount) {
        check_real(amount); auto columns = real_col_degrees(), rows = real_row_degrees();
        for (auto& d : rows) for (std::size_t a = 0; a < d.size(); ++a) d[a] = dynamic_coordinate_detail::difference(d[a], amount[a]);
        include_real_degrees(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) this->row_degrees.set(i, grid_degree(rows[i]));
        this->invalidate_compatible_sorting();
    }
    void set_all_generator_degrees(const real_degree_type& degree) { include_real_degrees({degree}); Core::set_all_generator_degrees(grid_degree(degree)); }
    template<class D, EnableIndexDegree<D> = 0>
    void set_all_generator_degrees(const D& d) { validate_index_degree(d); Core::set_all_generator_degrees(d); }
    std::pair<real_degree_type, real_degree_type> bounding_box() const {
        auto bounds = Core::bounding_box(); return {real_degree(bounds.first), real_degree(bounds.second)};
    }
    void cut_off_at(const real_degree_type& bound) {
        check_real(bound); vec<Index> columns, rows;
        for (Index i = 0; i < this->get_num_cols(); ++i) if (!RealDT::smaller_equal(real_col_degree(i), bound)) columns.push_back(i);
        for (Index i = 0; i < this->get_num_rows(); ++i) if (!RealDT::smaller_equal(real_row_degree(i), bound)) rows.push_back(i);
        this->delete_columns(columns); this->delete_rows(rows); this->sort_compatibly(); this->minimize();
    }
    void bound_support(const real_degree_type& bound) {
        include_real_degrees({bound}); auto b = grid_degree(bound); vec<Index> rows;
        for (Index i = 0; i < this->get_num_rows(); ++i) {
            degree_type d = this->row_degree(i);
            if (!DT::smaller_equal(d, b)) rows.push_back(i);
            else for (std::size_t a = 0; a < d.size(); ++a) { auto relation = d; relation[a] = b[a]; append_column_at_grid_degree({i}, relation); }
        }
        this->delete_rows(rows); this->sort_compatibly(); this->minimize();
    }
    template<class OutputStream> void to_stream(OutputStream& out, bool header = true) const {
        validate(); out << std::defaultfloat << std::setprecision(std::numeric_limits<Scalar>::max_digits10 > 0 ? std::numeric_limits<Scalar>::max_digits10 : 17);
        if (header) out << "scc2020\n" << poset_identifier() << '\n' << this->get_num_cols() << ' ' << this->get_num_rows() << " 0\n";
        for (Index i = 0; i < this->get_num_cols(); ++i) {
            RealDT::write_degree(out, real_col_degree(i)); out << " ;";
            for (Index r : this->column(i)) out << ' ' << r; out << '\n';
        }
        for (Index i = 0; i < this->get_num_rows(); ++i) { RealDT::write_degree(out, real_row_degree(i)); out << " ;\n"; }
    }
    template<class OutputStream> void to_stream_r2(OutputStream& out) const { to_stream(out); }
    void to_file(const std::string& path) const { std::ofstream out(path); if (!out) throw std::runtime_error("Cannot open SCC output: " + path); to_stream(out); }
    friend Self operator*(const Self& lhs, const Self& rhs) {
        if (lhs.real_col_degrees() != rhs.real_row_degrees()) throw std::invalid_argument("Different intermediate graded bases");
        Self a = lhs, b = rhs; a.merge_grids(b); return Core::multiply_coordinates(a, b);
    }
};

template<class S, class I, class Storage>
struct is_graded_sparse_matrix<DynamicGridGradedSparseMatrix<S,I,Storage>> : std::true_type {};

} // namespace graded_linalg
#include <grlina/dynamic_grid_kernel.hpp>

#include <grlina/runtime_matrix_io.hpp>
