/**
 * @file dynamic_coordinate_matrix_base.hpp
 * @brief NEW: shared runtime coordinate storage with CRTP result factories.
 *
 * Derived::empty_like preserves geometry when graded operations create maps.
 * The public direct-coordinate and grid-backed types each expose three template
 * parameters; this implementation detail does not add a dimension parameter.
 */
#pragma once

#include <grlina/dynamic_coordinate_degree.hpp>
#include <grlina/graded_matrix_algorithms.hpp>
#include <grlina/chain_complex.hpp>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace graded_linalg {

template<class Scalar, class Index, class Derived, class Storage>
class DynamicCoordinateMatrixBase
    : public SparseMatrix<Index, Storage>,
      public GradedMatrixAlgorithms<DynamicDegree<Scalar>, Index,
          Derived, CoordinateDegreeView<Scalar>> {
    using Self = Derived;
    using Base = SparseMatrix<Index, Storage>;
    using Algorithms = GradedMatrixAlgorithms<DynamicDegree<Scalar>, Index, Self, CoordinateDegreeView<Scalar>>;
    using DT = Degree_traits<DynamicDegree<Scalar>>;
    std::size_t parameters_ = 0;

    static Index checked_rank(Index rank) {
        if (rank < 0) throw std::invalid_argument("Negative matrix rank");
        return rank;
    }
    static std::size_t inferred_parameters(const vec<DynamicDegree<Scalar>>& columns,
                                           const vec<DynamicDegree<Scalar>>& rows) {
        if (!columns.empty()) return columns.front().size();
        if (!rows.empty()) return rows.front().size();
        throw std::invalid_argument("Empty degree lists require an explicit parameter count");
    }
    void check_column(const vec<Index>& column, const DynamicDegree<Scalar>& degree) const {
        check_degree(degree);
        if (!std::is_sorted(column.begin(), column.end()) ||
            std::adjacent_find(column.begin(), column.end()) != column.end())
            throw std::invalid_argument("Sparse columns must contain sorted unique row indices");
        for (Index row : column) {
            if (row < 0 || row >= this->get_num_rows())
                throw std::out_of_range("Sparse entry outside matrix rows");
            if (!DT::smaller_equal(row_degree(row), degree))
                throw std::invalid_argument("Column is not homogeneous at its degree");
        }
    }
    static FlatDegreeTable<Scalar> shifted_table(const FlatDegreeTable<Scalar>& table,
                                                const DynamicDegree<Scalar>& amount) {
        auto coordinates = table.coordinates();
        for (std::size_t i = 0; i < coordinates.size(); ++i)
            coordinates[i] = dynamic_coordinate_detail::difference(coordinates[i], amount[i % amount.size()]);
        return FlatDegreeTable<Scalar>(table.size(), table.parameter_count(), std::move(coordinates));
    }

protected:
    /** Copy entries in one batch; degree tables and sorting do not depend on storage. */
    template<class Target>
    Target copy_storage_as() const {
        static_cast<const Self&>(*this).validate();
        Target result(this->get_num_cols(), this->get_num_rows(), parameters_);
        array<Index> columns;
        columns.reserve(static_cast<std::size_t>(this->get_num_cols()));
        for (Index i = 0; i < this->get_num_cols(); ++i) columns.push_back(this->get_col(i));
        result.assign_data(std::move(columns));
        result.col_degrees = col_degrees;
        result.row_degrees = row_degrees;
        result.inherit_compatible_sorting(*this);
        result.col_batches = this->col_batches;
        result.k_max = this->k_max;
        result.rel_k = this->rel_k;
        result.gen_k = this->gen_k;
        result._rows = this->_rows;
        result.rows_computed = this->rows_computed;
        return result;
    }

public:
    static_assert(std::is_integral_v<Index> && std::is_signed_v<Index>,
                  "Graded matrix indices must represent the -1 sentinel");
    static_assert(is_degree_v<DynamicDegree<Scalar>>, "Unsupported coordinate scalar domain");
    using scalar_type = Scalar;
    using degree_type = DynamicDegree<Scalar>;
    using degree_view = CoordinateDegreeView<Scalar>;
    using index_type = Index;
    using derived_type = Self;
    using sparse_matrix_type = Base;
    using degree_storage_type = FlatDegreeTable<Scalar>;
    static constexpr bool runtime_dimension = true;

    // Public for compatibility with modules/complexes. Reads borrow contiguous
    // coordinates; writes use table.set or the matrix setters, never proxies.
    degree_storage_type col_degrees;
    degree_storage_type row_degrees;

    DynamicCoordinateMatrixBase() : DynamicCoordinateMatrixBase(0, 0, 0) {}
    DynamicCoordinateMatrixBase(Index columns, Index rows, std::size_t parameters)
        : Base(checked_rank(columns), checked_rank(rows)), parameters_(parameters),
          col_degrees(static_cast<std::size_t>(columns), parameters),
          row_degrees(static_cast<std::size_t>(rows), parameters) {
        this->resize_data(static_cast<std::size_t>(columns));
        this->refresh_compatible_sorted();
    }
    DynamicCoordinateMatrixBase(Index columns, Index rows, std::size_t parameters,
        vec<degree_type> column_degrees, vec<degree_type> generator_degrees)
        : DynamicCoordinateMatrixBase(columns, rows, parameters) {
        if (column_degrees.size() != static_cast<std::size_t>(columns) ||
            generator_degrees.size() != static_cast<std::size_t>(rows))
            throw std::invalid_argument("Degree counts do not match matrix ranks");
        col_degrees = column_degrees;
        row_degrees = generator_degrees;
        this->refresh_compatible_sorted();
    }
    DynamicCoordinateMatrixBase(Index columns, Index rows, std::size_t parameters,
        const array<Index>& entries, vec<degree_type> column_degrees, vec<degree_type> generator_degrees)
        : DynamicCoordinateMatrixBase(columns, rows, parameters, std::move(column_degrees), std::move(generator_degrees)) {
        this->assign_data(entries);
        validate();
    }
    DynamicCoordinateMatrixBase(Index columns, Index rows,
        vec<degree_type> column_degrees, vec<degree_type> generator_degrees)
        : DynamicCoordinateMatrixBase(columns, rows, inferred_parameters(column_degrees, generator_degrees),
               column_degrees, generator_degrees) {}
    DynamicCoordinateMatrixBase(Index columns, Index rows, const array<Index>& entries,
        vec<degree_type> column_degrees, vec<degree_type> generator_degrees)
        : DynamicCoordinateMatrixBase(columns, rows, inferred_parameters(column_degrees, generator_degrees),
               entries, column_degrees, generator_degrees) {}
    DynamicCoordinateMatrixBase(Index columns, Index rows, std::size_t parameters,
                                        const std::string& type, Index percent = -1)
        : Base(checked_rank(columns), checked_rank(rows), type, percent), parameters_(parameters),
          col_degrees(static_cast<std::size_t>(columns), parameters),
          row_degrees(static_cast<std::size_t>(rows), parameters) {
        validate();
        this->refresh_compatible_sorted();
    }
    explicit DynamicCoordinateMatrixBase(Base&& matrix, std::size_t parameters)
        : DynamicCoordinateMatrixBase(matrix.get_num_cols(), matrix.get_num_rows(), parameters) {
        Base::operator=(std::move(matrix));
    }
    explicit DynamicCoordinateMatrixBase(std::istream& input, bool sort_if_needed = false) {
        auto complex = ChainComplex<Self>::from_stream(input, sort_if_needed);
        if (complex.size() != 1)
            throw std::runtime_error("Expected an SCC presentation with one differential");
        static_cast<Self&>(*this) = std::move(complex[0]);
    }
    explicit DynamicCoordinateMatrixBase(const std::string& path, bool sort_if_needed = false) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot open SCC file: " + path);
        auto complex = ChainComplex<Self>::from_stream(input, sort_if_needed);
        if (complex.size() != 1)
            throw std::runtime_error("Expected an SCC presentation with one differential");
        static_cast<Self&>(*this) = std::move(complex[0]);
    }

    std::size_t parameter_count() const noexcept { return parameters_; }
    Self empty_like(Index columns, Index rows) const { return Self(columns, rows, parameters_); }
    std::string poset_identifier() const { return DT::poset_identifier(parameters_); }
    degree_view row_degree(Index row) const {
        return row_degrees[dynamic_coordinate_detail::checked_index(row, row_degrees.size())];
    }
    degree_view col_degree(Index column) const {
        return col_degrees[dynamic_coordinate_detail::checked_index(column, col_degrees.size())];
    }
    void check_degree(const degree_type& degree) const {
        if (degree.size() != parameters_)
            throw std::invalid_argument("Degree has the wrong number of parameters");
        dynamic_coordinate_detail::validate_coordinates(degree);
    }
    void set_row_degree(Index row, const degree_type& degree) {
        check_degree(degree);
        row_degrees.set(dynamic_coordinate_detail::checked_index(row, row_degrees.size()), degree);
        this->invalidate_compatible_sorting();
    }
    void set_col_degree(Index column, const degree_type& degree) {
        check_degree(degree);
        col_degrees.set(dynamic_coordinate_detail::checked_index(column, col_degrees.size()), degree);
        this->invalidate_compatible_sorting();
    }
    void validate() const {
        if (this->get_num_cols() < 0 || this->get_num_rows() < 0 ||
            col_degrees.parameter_count() != parameters_ || row_degrees.parameter_count() != parameters_ ||
            col_degrees.size() != static_cast<std::size_t>(this->get_num_cols()) ||
            row_degrees.size() != static_cast<std::size_t>(this->get_num_rows()) ||
            this->data.size() != static_cast<std::size_t>(this->get_num_cols()))
            throw std::invalid_argument("Inconsistent graded matrix dimensions");
        for (Index column = 0; column < this->get_num_cols(); ++column) {
            decltype(auto) entries = this->column(column);
            if (!std::is_sorted(entries.begin(), entries.end()) ||
                std::adjacent_find(entries.begin(), entries.end()) != entries.end())
                throw std::invalid_argument("Sparse columns must contain sorted unique row indices");
            for (Index row : entries)
                if (row < 0 || row >= this->get_num_rows())
                    throw std::out_of_range("Sparse entry outside matrix rows");
        }
        if (!this->is_graded_matrix()) throw std::invalid_argument("Matrix is not graded");
    }
    void invalidate_cached_rows() noexcept {
        this->_rows.clear();
        this->rows_computed = false;
        this->pivots.clear();
    }
    void set_col(Index column, const vec<Index>& entries) {
        dynamic_coordinate_detail::checked_index(column, col_degrees.size());
        Base::set_col(column, entries);
        invalidate_cached_rows();
    }
    void col_op(Index source, Index target) {
        dynamic_coordinate_detail::checked_index(source, col_degrees.size());
        dynamic_coordinate_detail::checked_index(target, col_degrees.size());
        Base::col_op(source, target);
        invalidate_cached_rows();
    }
    template<class Range>
    void add_to_col(Index column, const Range& entries) {
        dynamic_coordinate_detail::checked_index(column, col_degrees.size());
        Base::add_to_col(column, entries);
        invalidate_cached_rows();
    }
    void row_op_on_cols(Index source, Index target) {
        dynamic_coordinate_detail::checked_index(source, row_degrees.size());
        dynamic_coordinate_detail::checked_index(target, row_degrees.size());
        Base::row_op_on_cols(source, target);
        invalidate_cached_rows();
    }
    void permute_columns_graded(const vec<Index>& new_to_old) {
        auto degrees = col_degrees;
        degrees.permute(new_to_old); // validate before touching sparse storage
        Base::permute_columns(new_to_old);
        col_degrees = std::move(degrees);
        invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }
    /** old_to_new[old] is the new row index, matching the existing matrix API. */
    void permute_rows_graded(const vec<Index>& old_to_new) {
        if (old_to_new.size() != row_degrees.size())
            throw std::invalid_argument("Row permutation has the wrong size");
        vec<Index> new_to_old(old_to_new.size(), Index(-1));
        for (Index old = 0; old < this->get_num_rows(); ++old) {
            auto next = dynamic_coordinate_detail::checked_index(old_to_new[old], old_to_new.size());
            if (new_to_old[next] != -1) throw std::invalid_argument("Row permutation contains duplicates");
            new_to_old[next] = old;
        }
        auto degrees = row_degrees;
        degrees.permute(new_to_old);
        this->transform_data(old_to_new);
        this->sort_data();
        row_degrees = std::move(degrees);
        invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }
    void delete_columns(const vec<Index>& indices) {
        auto degrees = col_degrees;
        degrees.erase_indices(indices);
        Base::delete_columns(indices);
        col_degrees = std::move(degrees);
        invalidate_cached_rows();
    }
    void delete_rows(const vec<Index>& indices) {
        auto degrees = row_degrees;
        degrees.erase_indices(indices);
        Base::delete_rows(indices);
        row_degrees = std::move(degrees);
        invalidate_cached_rows();
    }
    /** Historical spelling: this operation truncates generator rows. */
    void cull_columns(Index threshold, bool from_end) {
        if (threshold < 0 || threshold > this->get_num_rows())
            throw std::out_of_range("Generator truncation outside matrix rows");
        const Index keep = from_end ? this->get_num_rows() - threshold : threshold;
        vec<Index> remove;
        for (Index i = keep; i < this->get_num_rows(); ++i) remove.push_back(i);
        delete_rows(remove);
        this->refresh_compatible_sorted();
    }
    void append_column(const vec<Index>& entries, const degree_type& degree) {
        check_column(entries, degree);
        if (this->get_num_cols() == std::numeric_limits<Index>::max())
            throw std::length_error("Sparse column count overflow");
        const auto old_count = col_degrees.size();
        col_degrees.push_back(degree);
        try {
            Base::append_col(entries);
        } catch (...) {
            col_degrees.resize(old_count);
            throw;
        }
        ++this->num_cols;
        invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }
    void append_matrix(const Self& other) {
        if (&other == this) { Self copy(other); append_matrix(copy); return; }
        validate(); other.validate();
        if (parameters_ != other.parameters_ || row_degrees != other.row_degrees)
            throw std::invalid_argument("Appended matrix has incompatible generator degrees");
        if (other.get_num_cols() > std::numeric_limits<Index>::max() - this->get_num_cols())
            throw std::length_error("Sparse column count overflow");
        auto degrees = col_degrees;
        degrees.reserve(col_degrees.size() + other.col_degrees.size());
        for (auto degree : other.col_degrees) degrees.push_back(degree);
        for (Index i = 0; i < other.get_num_cols(); ++i) Base::append_col(other.get_col(i));
        this->num_cols += other.get_num_cols();
        col_degrees = std::move(degrees);
        invalidate_cached_rows();
        this->invalidate_compatible_sorting();
    }
    void append_move_matrix(Self&& other) { append_matrix(other); }
    Self restricted_domain_copy(const vec<Index>& columns) const {
        auto degrees = col_degrees.select(columns);
        Self result = static_cast<const Self&>(*this).empty_like(static_cast<Index>(columns.size()), this->get_num_rows());
        for (Index i = 0; i < static_cast<Index>(columns.size()); ++i)
            result.set_col(i, this->get_col(columns[i]));
        result.col_degrees = std::move(degrees);
        result.row_degrees = row_degrees;
        result.inherit_compatible_sorting(*this);
        return result;
    }
    Self transposed_copy() const {
        Self result = static_cast<const Self&>(*this).empty_like(this->get_num_rows(), this->get_num_cols());
        result.assign_data(Base::transposed_copy().data);
        result.col_degrees = row_degrees;
        result.row_degrees = col_degrees;
        result.inherit_compatible_sorting(*this);
        return result;
    }
    void shift(const degree_type& amount) {
        check_degree(amount);
        auto columns = shifted_table(col_degrees, amount);
        auto rows = shifted_table(row_degrees, amount);
        col_degrees = std::move(columns);
        row_degrees = std::move(rows);
        // Translation preserves both compatible orders; overflow was checked
        // before replacing either coordinate buffer.
    }
    void shift_generators(const degree_type& amount) {
        check_degree(amount);
        row_degrees = shifted_table(row_degrees, amount);
    }
    void set_all_generator_degrees(const degree_type& degree) {
        check_degree(degree);
        auto rows = row_degrees, columns = col_degrees;
        for (std::size_t i = 0; i < rows.size(); ++i) rows.set(i, degree);
        for (std::size_t i = 0; i < columns.size(); ++i) columns.set(i, DT::join(degree, columns[i]));
        row_degrees = std::move(rows); col_degrees = std::move(columns);
        this->invalidate_compatible_sorting();
    }
    template<class OutputStream>
    void to_stream(OutputStream& output, bool header = true) const {
        validate();
        output << std::setprecision(std::numeric_limits<Scalar>::max_digits10 > 0
            ? std::numeric_limits<Scalar>::max_digits10 : 17);
        if (header)
            output << "scc2020\n" << poset_identifier() << '\n'
                   << this->get_num_cols() << ' ' << this->get_num_rows() << " 0\n";
        for (Index column = 0; column < this->get_num_cols(); ++column) {
            DT::write_degree(output, col_degree(column));
            output << " ;";
            for (Index row : this->column(column)) output << ' ' << row;
            output << '\n';
        }
        for (Index row = 0; row < this->get_num_rows(); ++row) {
            DT::write_degree(output, row_degree(row));
            output << " ;\n";
        }
    }
    void to_file(const std::string& path) const {
        validate();
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Cannot open SCC output: " + path);
        to_stream(output);
    }
    static Self multiply_coordinates(const Self& lhs, const Self& rhs) {
        lhs.validate(); rhs.validate();
        if (lhs.parameters_ != rhs.parameters_ || lhs.col_degrees != rhs.row_degrees)
            throw std::invalid_argument("Product has incompatible intermediate generator degrees");
        Base product = static_cast<const Base&>(lhs) * static_cast<const Base&>(rhs);
        Self result = lhs.empty_like(product.get_num_cols(), product.get_num_rows());
        result.assign_data(std::move(product.data));
        result.row_degrees = lhs.row_degrees;
        result.col_degrees = rhs.col_degrees;
        result.refresh_compatible_sorted();
        return result;
    }
};

} // namespace graded_linalg
