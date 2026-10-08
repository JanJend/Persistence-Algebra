/**
 * @file r2grid_gs_matrix.hpp
 * @brief R^2-graded matrices storing integer grid coordinates as their degrees.
 *
 * Kernel reduction is adapted from MPfree, as in r2graded_matrix.hpp.
 * Copyright 2026 TU Graz. Distributed under the GNU LGPL, version 3 or later.
 */
#pragma once

#include <grlina/coordinate_degree.hpp>
#include <grlina/grid_scheduler.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace graded_linalg {

/** An order-preserving grid embedding Z^2 -> R^2 carried by a graded matrix.
 *
 * row_degrees and col_degrees are grid indices: {i,j} represents
 * {x_grid[i],y_grid[j]}. Public geometric methods take real coordinates.
 * Queries also accept typed degree_type arguments as integer grid points;
 * brace-initialized queries use real coordinates.
 * Integer degree arithmetic must never be used for geometric translations.
 *
 * Grids are strictly increasing and coordinates must be finite. Unused grid
 * points are retained by kernels, restrictions, deletion and minimization.
 * Direct edits to the public grids require rebuild_grid_maps(); inserting a
 * coordinate should instead use reindex_grid() to preserve existing degrees.
 * As with GradedSparseMatrix, direct degree edits invalidate sorting caches.
 */
template <typename index, typename real_n = double, typename MatrixBase = SparseMatrix<index>>
struct R2GridGSMatrix
    : CoordinateGradedSparseMatrix<index, 2, index,
          R2GridGSMatrix<index, real_n, MatrixBase>, MatrixBase> {
    using Self = R2GridGSMatrix;
    using degree_type = CoordinateDegree<index, 2>;
    using real_degree_type = CoordinateDegree<real_n, 2>;
    using real_type = real_n;
    using Base = CoordinateGradedSparseMatrix<index, 2, index, Self, MatrixBase>;
    using PQ = std::priority_queue<index, vec<index>, std::greater<index>>;
    template <typename D>
    using EnableGridDegree = std::enable_if_t<std::is_same_v<std::decay_t<D>, degree_type>, int>;
    static_assert(CoordinatePosetDomain<real_n>::suffix[0] == '\0',
                  "real_n must be a real coordinate domain");

    // Matrices constructed without degrees start at the real origin.
    vec<real_n> x_grid{real_n(0)};
    vec<real_n> y_grid{real_n(0)};
    std::unordered_map<real_n, index> x_to_index{{real_n(0), index(0)}};
    std::unordered_map<real_n, index> y_to_index{{real_n(0), index(0)}};
    Grid_scheduler<index> grid_scheduler;
    vec<PQ> pq_row;

    R2GridGSMatrix() = default;
    R2GridGSMatrix(const Self&) = default;
    R2GridGSMatrix(Self&&) = default;
    Self& operator=(const Self&) = default;
    Self& operator=(Self&&) = default;

    R2GridGSMatrix(index cols, index rows) : Base(cols, rows) {}
    R2GridGSMatrix(index n, vec<index> indicator) : Base(n, std::move(indicator)) {}
    R2GridGSMatrix(index cols, index rows, const std::string& type, index percent = -1)
        : Base(cols, rows, type, percent) {}

    template <typename Matrix, std::enable_if_t<std::is_same_v<std::decay_t<Matrix>, MatrixBase>, int> = 0>
    explicit R2GridGSMatrix(Matrix&& other)
        : Base(other.get_num_cols(), other.get_num_rows()) {
        MatrixBase::operator=(std::forward<Matrix>(other));
    }

    R2GridGSMatrix(index cols, index rows, vec<real_degree_type> columns,
                   vec<real_degree_type> generators) : Base(cols, rows) {
        set_real_degrees(columns, generators);
    }
    R2GridGSMatrix(index cols, index rows, const array<index>& data,
                   vec<real_degree_type> columns, vec<real_degree_type> generators)
        : Base(cols, rows, data, vec<degree_type>(cols), vec<degree_type>(rows)) {
        set_real_degrees(columns, generators);
    }

    /** Construct from integer degrees and their explicit embedding. */
    R2GridGSMatrix(index cols, index rows, vec<degree_type> columns,
                   vec<degree_type> generators, vec<real_n> xs, vec<real_n> ys)
        : Base(cols, rows, std::move(columns), std::move(generators)),
          x_grid(std::move(xs)), y_grid(std::move(ys)) {
        rebuild_grid_maps();
        this->refresh_compatible_sorted();
    }

    /** Import Z^2 grid indices only when their real embedding is supplied. */
    template <typename Integer, typename Derived, std::enable_if_t<std::is_integral_v<Integer>, int> = 0>
    R2GridGSMatrix(const GradedSparseMatrix<CoordinateDegree<Integer, 2>, index, Derived, MatrixBase>& other,
                   vec<real_n> xs, vec<real_n> ys)
        : Base(other.get_num_cols(), other.get_num_rows()), x_grid(std::move(xs)), y_grid(std::move(ys)) {
        validate_grid(x_grid); validate_grid(y_grid);
        auto convert = [&](const auto& source, auto& degrees) {
            for (std::size_t i = 0; i < source.size(); ++i) {
                const auto& d = source[i];
                if (d[0] < 0 || d[1] < 0 || static_cast<std::uintmax_t>(d[0]) >= x_grid.size() ||
                    static_cast<std::uintmax_t>(d[1]) >= y_grid.size())
                    throw std::invalid_argument("Degree outside the supplied grid embedding");
                degrees.at(i) = {static_cast<index>(d[0]), static_cast<index>(d[1])};
            }
        };
        if (other.col_degrees.size() != this->col_degrees.size() || other.row_degrees.size() != this->row_degrees.size())
            throw std::invalid_argument("Degree counts must match matrix dimensions");
        convert(other.col_degrees, this->col_degrees); convert(other.row_degrees, this->row_degrees);
        rebuild_grid_maps();
        MatrixBase::operator=(static_cast<const MatrixBase&>(other));
        this->refresh_compatible_sorted();
        this->col_batches = other.col_batches; this->k_max = other.k_max;
    }
    R2GridGSMatrix(index cols, index rows, const array<index>& data,
                   vec<degree_type> columns, vec<degree_type> generators,
                   vec<real_n> xs, vec<real_n> ys)
        : Base(cols, rows, data, std::move(columns), std::move(generators)),
          x_grid(std::move(xs)), y_grid(std::move(ys)) {
        rebuild_grid_maps();
        this->refresh_compatible_sorted();
    }

    /** Import an ordinary coordinate-graded matrix, including R2GradedSparseMatrix. */
    template <typename Scalar, typename Derived,
              std::enable_if_t<!std::is_integral_v<Scalar>, int> = 0>
    explicit R2GridGSMatrix(const GradedSparseMatrix<CoordinateDegree<Scalar, 2>,
                           index, Derived, MatrixBase>& other)
        : Base(other.get_num_cols(), other.get_num_rows()) {
        vec<real_degree_type> columns, generators;
        for (const auto& d : other.col_degrees)
            columns.push_back({static_cast<real_n>(d[0]), static_cast<real_n>(d[1])});
        for (const auto& d : other.row_degrees)
            generators.push_back({static_cast<real_n>(d[0]), static_cast<real_n>(d[1])});
        set_real_degrees(columns, generators);
        MatrixBase::operator=(static_cast<const MatrixBase&>(other));
        this->col_batches = other.col_batches;
        this->k_max = other.k_max;
    }

    template <typename Scalar, typename Derived,
              std::enable_if_t<!std::is_integral_v<Scalar>, int> = 0>
    Self& operator=(const GradedSparseMatrix<CoordinateDegree<Scalar, 2>,
                    index, Derived, MatrixBase>& other) {
        Self result(other);
        return *this = std::move(result);
    }

    /** Converting grid matrices retains the embedding, including unused points. */
    template <typename OtherReal>
    explicit R2GridGSMatrix(const R2GridGSMatrix<index, OtherReal, MatrixBase>& other)
        : Base(other.get_num_cols(), other.get_num_rows(), other.col_degrees, other.row_degrees),
          x_grid(other.x_grid.begin(), other.x_grid.end()), y_grid(other.y_grid.begin(), other.y_grid.end()) {
        rebuild_grid_maps(); // Reject a narrowing conversion which collapses distinct grid points.
        this->refresh_compatible_sorted();
        MatrixBase::operator=(static_cast<const MatrixBase&>(other));
        this->col_batches = other.col_batches;
        this->k_max = other.k_max;
    }

    explicit R2GridGSMatrix(std::istream& stream, bool lex_sort = false, bool compute_batches = false) {
        parse_stream(stream, lex_sort, compute_batches);
    }
    explicit R2GridGSMatrix(const std::string& filepath, bool lex_sort = false, bool compute_batches = false) {
        auto stream = Base::create_ifstream(filepath);
        parse_stream(stream, lex_sort, compute_batches);
    }

    real_degree_type real_degree(const degree_type& d) const {
        return {x_grid.at(static_cast<std::size_t>(d[0])),
                y_grid.at(static_cast<std::size_t>(d[1]))};
    }
    real_degree_type real_column_degree(index i) const { return real_degree(this->col_degrees.at(i)); }
    real_degree_type real_row_degree(index i) const { return real_degree(this->row_degrees.at(i)); }
    vec<real_degree_type> real_column_degrees() const { return embedded_degrees(this->col_degrees); }
    vec<real_degree_type> real_row_degrees() const { return embedded_degrees(this->row_degrees); }

    degree_type grid_degree(const real_degree_type& d) const {
        return {x_to_index.at(d[0]), y_to_index.at(d[1])};
    }

    /** Replace real degrees and build their grid once, preserving matrix entries. */
    void set_real_degrees(const vec<real_degree_type>& columns,
                          const vec<real_degree_type>& generators) {
        if (columns.size() != static_cast<std::size_t>(this->get_num_cols()) ||
            generators.size() != static_cast<std::size_t>(this->get_num_rows()))
            throw std::invalid_argument("Degree counts must match matrix dimensions");
        vec<real_n> xs, ys;
        for (const auto* degrees : {&columns, &generators})
            for (const auto& d : *degrees) {
                check_coordinate(d[0]); check_coordinate(d[1]);
                xs.push_back(d[0]); ys.push_back(d[1]);
            }
        sort_unique(xs); sort_unique(ys);
        auto xm = make_index_map(xs), ym = make_index_map(ys);
        auto convert = [&](const vec<real_degree_type>& degrees) {
            vec<degree_type> result;
            result.reserve(degrees.size());
            for (const auto& d : degrees) result.push_back({xm.at(d[0]), ym.at(d[1])});
            return result;
        };
        auto cs = convert(columns), rs = convert(generators);
        x_grid = std::move(xs); y_grid = std::move(ys);
        x_to_index = std::move(xm); y_to_index = std::move(ym);
        this->col_degrees = std::move(cs); this->row_degrees = std::move(rs);
        invalidate_grid_workspaces();
        this->refresh_compatible_sorted();
    }

    /** Rebuild maps after an intentional edit to the embedding itself. */
    void rebuild_grid_maps() {
        auto xm = make_index_map(x_grid), ym = make_index_map(y_grid);
        validate_degree_indices();
        x_to_index = std::move(xm); y_to_index = std::move(ym);
        invalidate_grid_workspaces();
    }

    void validate() const {
        validate_embedding();
        Base::validate();
    }

    /** Change to a larger grid without changing any embedded degree. */
    void reindex_grid(vec<real_n> xs, vec<real_n> ys) {
        validate_embedding();
        if (xs == x_grid && ys == y_grid) return;
        auto xm = make_index_map(xs), ym = make_index_map(ys);
        vec<index> xmap, ymap;
        for (const auto& x : x_grid) xmap.push_back(xm.at(x));
        for (const auto& y : y_grid) ymap.push_back(ym.at(y));
        for (auto* degrees : {&this->col_degrees, &this->row_degrees})
            for (auto& d : *degrees) d = {xmap[d[0]], ymap[d[1]]};
        x_grid = std::move(xs); y_grid = std::move(ys);
        x_to_index = std::move(xm); y_to_index = std::move(ym);
        invalidate_grid_workspaces();
        this->refresh_compatible_sorted();
    }

    /** Put both matrices on their common grid. Matrix entries remain unchanged. */
    void merge_grids(Self& other) {
        validate_embedding(); other.validate_embedding();
        if (this == &other || (x_grid == other.x_grid && y_grid == other.y_grid)) return;
        auto xs = grid_union(x_grid, other.x_grid), ys = grid_union(y_grid, other.y_grid);
        reindex_grid(xs, ys);
        other.reindex_grid(std::move(xs), std::move(ys));
    }

    /** Compatibility name: only order columns; the grid is already stored. */
    vec<index> compute_grid_representation() {
        return this->sort_columns_colexicographically_with_output();
    }

    void initialise_grid_scheduler() {
        // The scheduler needs a z2_col_degrees name, not a second degree vector.
        struct View {
            const Self& matrix;
            const vec<degree_type>& z2_col_degrees;
            const vec<real_n>& x_grid;
            const vec<real_n>& y_grid;
            index get_num_cols() const { return matrix.get_num_cols(); }
        } view{*this, this->col_degrees, x_grid, y_grid};
        grid_scheduler = Grid_scheduler<index>(view);
    }

    void kernel_column_reduction(index i, pair<index>& curr_gr, MatrixBase& column_operations,
                                 bool store_col_ops = false, bool notify_pq = false) {
        index p = this->col_last(i);
        while (p != -1 && this->pivots.count(p)) {
            index k = this->pivots[p];
            if (k < i) {
                this->col_op(k, i);
                if (store_col_ops) column_operations.col_op(k, i);
                p = this->col_last(i);
            } else if (notify_pq) {
                index y = this->col_degrees[k][1];
                pq_row[y].push(k);
                grid_scheduler.notify(curr_gr.first, y);
                break;
            } else break;
        }
        if (p != -1) this->pivots[p] = i;
    }

    /** Destructive kernel reduction, matching R2GradedSparseMatrix's convention. */
    Self graded_kernel() {
        GRLINA_DEBUG_CHECK(validate());
        this->pivots.clear(); pq_row.clear(); this->invalidate_cached_rows();
        auto permutation = compute_grid_representation();
        initialise_grid_scheduler();
        pq_row.resize(y_grid.size());
        MatrixBase operations(this->get_num_cols(), this->get_num_cols(), "Identity");
        vec<degree_type> degrees;
        typename MatrixBase::storage_type columns;
        std::vector<bool> in_kernel(this->get_num_cols(), false);
        while (!grid_scheduler.at_end()) {
            auto d = grid_scheduler.next_grade();
            auto& pq = pq_row[d.second];
            auto range = grid_scheduler.index_range_at(d.first, d.second);
            for (index i = range.first; i < range.second; ++i) pq.push(i);
            while (!pq.empty()) {
                index i = pq.top();
                while (!pq.empty() && pq.top() == i) pq.pop();
                GRLINA_ASSERT(this->col_degrees[i][0] <= d.first && this->col_degrees[i][1] == d.second);
                kernel_column_reduction(i, d, operations, true, true);
                if (!in_kernel[i] && this->is_zero(i)) {
                    columns.push_back(operations.get_col(i));
                    degrees.push_back({d.first, d.second});
                    in_kernel[i] = true;
                    this->clear_col(i); operations.clear_col(i);
                }
            }
        }
        Self result(static_cast<index>(degrees.size()), this->get_num_cols());
        result.assign_data(std::move(columns));
        result.col_degrees = std::move(degrees);
        result.row_degrees = this->col_degrees;
        result.copy_embedding(*this);
        // Kernel rows refer to the original domain basis, including its order.
        result.permute_rows_graded(permutation);
        result.sort_columns_lexicographically();
        return result;
    }

    Self restricted_domain_copy(vec<index>& columns) const {
        auto result = Base::restricted_domain_copy(columns);
        result.copy_embedding(*this);
        return result;
    }
    Self transposed_copy() const {
        auto result = Base::transposed_copy();
        result.copy_embedding(*this);
        return result;
    }
    void cull_columns(const index& threshold, bool from_end) {
        if (threshold < 0 || threshold > this->get_num_rows())
            throw std::invalid_argument("Row culling threshold outside the matrix");
        Base::cull_columns(threshold, from_end);
    }

    void append_matrix(const Self& other) {
        require_same_rows(other);
        Self aligned = other; // Also handles appending the matrix to itself.
        merge_grids(aligned);
        Base::append_matrix(aligned);
    }
    void append_move_matrix(Self&& other) {
        if (this == &other) { append_matrix(other); return; }
        require_same_rows(other);
        merge_grids(other);
        Base::append_move_matrix(std::move(other));
    }
    void append_column(const vec<index>& column, const real_degree_type& d) {
        include_real_degrees({d});
        Base::append_column(column, grid_degree(d));
    }
    /** Explicit alternative when the caller already has integer grid indices. */
    void append_column_at_grid_degree(const vec<index>& column, const degree_type& d) {
        validate_index_degree(d);
        Base::append_column(column, d);
    }
    template <typename D, EnableGridDegree<D> = 0>
    void append_column(const vec<index>& column, const D& d) { append_column_at_grid_degree(column, d); }

    void quotient_by(Self& other) {
        append_matrix(other); this->sort_compatibly(); this->minimize();
    }
    Self quotient_by_copy(Self& other) const {
        Self result = *this; result.quotient_by(other); return result;
    }
    Self inverse_image(const Self& presentation, const Self& submodule) {
        index domain_size = this->get_num_cols();
        require_same_rows(presentation); require_same_rows(submodule);
        append_matrix(submodule); append_matrix(presentation);
        auto result = graded_kernel();
        result.cull_columns(domain_size, false);
        return result;
    }
    Self inverse_image_copy(const Self& presentation, const Self& submodule) const {
        Self result = *this; return result.inverse_image(presentation, submodule);
    }
    Self submodule_intersection(const Self& left, const Self& right) const {
        require_same_rows(left); require_same_rows(right);
        auto k = left.inverse_image_copy(*this, right);
        return left * k;
    }
    Self submodule_generated_by(const Self& generators) const {
        require_same_rows(generators);
        Self injection = generators;
        index n = injection.get_num_cols();
        injection.append_matrix(*this);
        auto result = injection.graded_kernel();
        result.cull_columns(n, false);
        return result;
    }
    Self presentation_of_submodule(const Self& presentation) {
        require_same_rows(presentation);
        index n = this->get_num_cols();
        append_matrix(presentation);
        auto result = graded_kernel();
        result.cull_columns(n, false);
        return result;
    }
    void cancel_local_pairs(Self* elements = nullptr) {
        if (elements) {
            if (elements == this) throw std::invalid_argument("Elements must be a separate matrix");
            require_same_rows(*elements);
            auto order = this->compatible_order_;
            bool certified = this->compatibly_sorted;
            merge_grids(*elements);
            if (certified) this->refresh_compatible_sorted(order);
        }
        GRLINA_DEBUG_CHECK(validate());
        Base::cancel_local_pairs(elements);
    }

    void minimize() { GRLINA_DEBUG_CHECK(validate()); Base::minimize(); }
    void minimize_variant() { GRLINA_DEBUG_CHECK(validate()); Base::minimize_variant(); }

    /** Greatest grid point <= a real query, with -1 below either grid axis. */
    pair<index> get_closest_smaller_grid_point(const real_degree_type& d) const {
        return {floor_index(x_grid, d[0]), floor_index(y_grid, d[1])};
    }
    std::pair<MatrixBase, vec<index>> map_at_degree_pair(real_degree_type d, bool shifted = true) const {
        return Base::map_at_degree_pair(query_degree(d), shifted);
    }
    MatrixBase map_at_degree(real_degree_type d, vec<index>& columns) const {
        auto grid_d = query_degree(d);
        for (index i = 0; i < this->get_num_cols(); ++i)
            if (Degree_traits<degree_type>::smaller_equal(this->col_degrees[i], grid_d)) columns.push_back(i);
        return MatrixBase::restricted_domain_copy(columns);
    }
    index num_cols_before(real_degree_type d) const { return count_before(this->col_degrees, query_degree(d)); }
    index num_rows_before(real_degree_type d) const { return count_before(this->row_degrees, query_degree(d)); }
    vec<index> basislift_at(real_degree_type d) const { return Base::basislift_at(query_degree(d)); }
    index dim_at(real_degree_type d) const { return static_cast<index>(basislift_at(d).size()); }

    // Templates let typed integer degrees use the inherited grid algorithms
    // without making brace-initialized real queries ambiguous.
    template <typename D, EnableGridDegree<D> = 0>
    std::pair<MatrixBase, vec<index>> map_at_degree_pair(const D& d, bool shifted = true) const {
        return Base::map_at_degree_pair(d, shifted);
    }
    template <typename D, EnableGridDegree<D> = 0>
    MatrixBase map_at_degree(const D& d, vec<index>& columns) const { return Base::map_at_degree(d, columns); }
    template <typename D, EnableGridDegree<D> = 0>
    index num_cols_before(const D& d) const { return count_before(this->col_degrees, d); }
    template <typename D, EnableGridDegree<D> = 0>
    index num_rows_before(const D& d) const { return count_before(this->row_degrees, d); }
    template <typename D, EnableGridDegree<D> = 0>
    vec<index> basislift_at(const D& d) const { return Base::basislift_at(d); }
    template <typename D, EnableGridDegree<D> = 0>
    index dim_at(const D& d) const { return static_cast<index>(Base::basislift_at(d).size()); }

    bool is_admissible_column_operation(index i, index j) const {
        return Base::is_admissible_column_operation(i, j);
    }
    bool is_admissible_column_operation(index i, real_degree_type d) const {
        return Degree_traits<real_degree_type>::smaller_equal(real_column_degree(i), d);
    }
    bool is_admissible_row_operation(index i, index j) const { return Base::is_admissible_row_operation(i, j); }
    bool is_admissible_row_operation(index i, real_degree_type d) const {
        return Degree_traits<real_degree_type>::greater_equal(real_row_degree(i), d);
    }
    bool is_admissible_row_operation(real_degree_type d, index i) const {
        return Degree_traits<real_degree_type>::greater_equal(d, real_row_degree(i));
    }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_column_operation(index i, const D& d) const {
        return Base::is_admissible_column_operation(i, d);
    }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_row_operation(index i, const D& d) const { return Base::is_admissible_row_operation(i, d); }
    template <typename D, EnableGridDegree<D> = 0>
    bool is_admissible_row_operation(const D& d, index i) const { return Base::is_admissible_row_operation(d, i); }
    vec<index> admissible_row_indices(real_degree_type d) const {
        vec<index> result;
        for (index i = 0; i < this->get_num_rows(); ++i)
            if (is_admissible_row_operation(d, i)) result.push_back(i);
        return result;
    }
    template <typename D, EnableGridDegree<D> = 0>
    vec<index> admissible_row_indices(const D& d) const {
        vec<index> result;
        for (index i = 0; i < this->get_num_rows(); ++i)
            if (Base::is_admissible_row_operation(d, i)) result.push_back(i);
        return result;
    }

    vec<real_degree_type> discrete_support() const {
        auto support = Base::discrete_support();
        return embedded_degrees(support);
    }
    QuiverRepresentation<index, real_degree_type, MatrixBase> induced_quiver_rep(
        vec<real_degree_type> vertices = {}, array<index> edges = {}) {
        auto matrix = real_matrix_copy();
        return matrix.induced_quiver_rep(std::move(vertices), std::move(edges));
    }
    Self submodule_generated_at(real_degree_type alpha) const {
        auto basis = basislift_at(alpha);
        Self injection(this->get_num_rows(), basis);
        injection.copy_embedding(*this);
        injection.row_degrees = this->row_degrees;
        injection.include_real_degrees({alpha});
        injection.col_degrees.assign(basis.size(), injection.grid_degree(alpha));
        injection.append_matrix(*this);
        auto result = injection.graded_kernel();
        result.cull_columns(static_cast<index>(basis.size()), false);
        result.sort_compatibly(); result.minimize();
        return result;
    }
    template <typename D, EnableGridDegree<D> = 0>
    Self submodule_generated_at(const D& d) const { return submodule_generated_at(real_degree(d)); }

    /** Match shift()'s existing convention: subtract the given real vector. */
    void shift(real_degree_type d) {
        check_coordinate(d[0]); check_coordinate(d[1]);
        auto xs = x_grid, ys = y_grid;
        for (auto& x : xs) x -= d[0];
        for (auto& y : ys) y -= d[1];
        auto xm = make_index_map(xs), ym = make_index_map(ys);
        x_grid = std::move(xs); y_grid = std::move(ys);
        x_to_index = std::move(xm); y_to_index = std::move(ym);
        // Integer degrees and their order do not change under a translation.
    }
    void shift_generators(real_degree_type d) {
        check_coordinate(d[0]); check_coordinate(d[1]);
        auto rows = real_row_degrees();
        for (auto& r : rows) { r[0] -= d[0]; r[1] -= d[1]; }
        include_real_degrees(rows);
        for (std::size_t i = 0; i < rows.size(); ++i) this->row_degrees[i] = grid_degree(rows[i]);
        invalidate_grid_workspaces();
    }
    void set_all_generator_degrees(real_degree_type d) {
        include_real_degrees({d});
        Base::set_all_generator_degrees(grid_degree(d));
        invalidate_grid_workspaces();
    }
    template <typename D, EnableGridDegree<D> = 0>
    void set_all_generator_degrees(const D& d) {
        validate_index_degree(d);
        Base::set_all_generator_degrees(d);
        invalidate_grid_workspaces();
    }

    pair<real_degree_type> bounding_box() const {
        if (this->col_degrees.empty() && this->row_degrees.empty())
            throw std::invalid_argument("An empty matrix has no degree bounding box");
        degree_type low{std::numeric_limits<index>::max(), std::numeric_limits<index>::max()};
        degree_type high{-1, -1};
        for (const auto* degrees : {&this->col_degrees, &this->row_degrees})
            for (const auto& d : *degrees) {
                low = Degree_traits<degree_type>::meet(low, d);
                high = Degree_traits<degree_type>::join(high, d);
            }
        return {real_degree(low), real_degree(high)};
    }
    vec<real_degree_type> get_equidistant_grid(const int& n) const {
        if (n < 0) throw std::invalid_argument("Grid size must be nonnegative");
        if (n == 0) return {};
        auto box = bounding_box();
        if (n == 1) return {box.first};
        auto step = grid_step(box, n - 1);
        vec<real_degree_type> result;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result.push_back({box.first[0] + real_n(i)*step[0], box.first[1] + real_n(j)*step[1]});
        return result;
    }
    void snap_to_equidistant_grid(int n, bool end_inclusive = false) {
        if (n <= 0) throw std::invalid_argument("Snapping requires a positive grid size");
        auto box = bounding_box();
        auto step = n == 1 ? real_degree_type{0, 0} : grid_step(box, end_inclusive ? n-1 : n);
        vec<real_n> xs, ys;
        for (int i = 0; i < n; ++i) {
            xs.push_back(box.first[0] + real_n(i)*step[0]);
            ys.push_back(box.first[1] + real_n(i)*step[1]);
        }
        // Include the endpoint exactly, despite floating point rounding.
        if (end_inclusive && n > 1) { xs.back() = box.second[0]; ys.back() = box.second[1]; }
        sort_unique(xs); sort_unique(ys);
        snap_to_grid(xs, ys);
    }
    void snap_to_grid(vec<real_n> xs, vec<real_n> ys) {
        if (xs.empty() || ys.empty()) throw std::invalid_argument("Snapping requires nonempty grid axes");
        auto xm = make_index_map(xs), ym = make_index_map(ys);
        GRLINA_DEBUG_CHECK(validate());
        auto columns = real_column_degrees(), rows = real_row_degrees();
        vec<index> remove_columns, remove_rows;
        auto snap = [&](const auto& real_degrees, auto& degrees, auto& remove) {
            for (std::size_t i = 0; i < degrees.size(); ++i) {
                index x = ceiling_index(xs, real_degrees[i][0]), y = ceiling_index(ys, real_degrees[i][1]);
                if (static_cast<std::size_t>(x) == xs.size() || static_cast<std::size_t>(y) == ys.size())
                    remove.push_back(static_cast<index>(i));
                else degrees[i] = {x, y};
            }
        };
        snap(columns, this->col_degrees, remove_columns);
        snap(rows, this->row_degrees, remove_rows);
        this->delete_columns(remove_columns); this->delete_rows(remove_rows);
        x_grid = std::move(xs); y_grid = std::move(ys);
        x_to_index = std::move(xm); y_to_index = std::move(ym);
        invalidate_grid_workspaces();
        this->sort_compatibly(); this->minimize();
    }
    void cut_above(real_n x_cutoff, real_n y_cutoff) {
        auto box = bounding_box();
        real_degree_type bound{box.first[0] + x_cutoff*(box.second[0]-box.first[0]),
                               box.first[1] + y_cutoff*(box.second[1]-box.first[1])};
        vec<index> columns, rows;
        for (index i = 0; i < this->get_num_cols(); ++i)
            if (!Degree_traits<real_degree_type>::smaller_equal(real_column_degree(i), bound)) columns.push_back(i);
        for (index i = 0; i < this->get_num_rows(); ++i)
            if (!Degree_traits<real_degree_type>::smaller_equal(real_row_degree(i), bound)) rows.push_back(i);
        this->delete_columns(columns); this->delete_rows(rows);
    }
    void bound_support(real_degree_type bound) {
        include_real_degrees({bound});
        auto grid_bound = grid_degree(bound);
        vec<index> rows;
        for (index i = 0; i < this->get_num_rows(); ++i) {
            const auto d = this->row_degrees[i];
            if (!Degree_traits<degree_type>::smaller_equal(d, grid_bound)) rows.push_back(i);
            else {
                Base::append_column(vec<index>{i}, degree_type{grid_bound[0], d[1]});
                Base::append_column(vec<index>{i}, degree_type{d[0], grid_bound[1]});
            }
        }
        this->delete_rows(rows); this->sort_compatibly(); this->minimize();
    }
    template <typename D, EnableGridDegree<D> = 0>
    void bound_support(const D& d) { bound_support(real_degree(d)); }

    template <typename OutputStream>
    void to_stream(OutputStream& out, bool header = true) const {
        out << std::defaultfloat << std::setprecision(std::numeric_limits<real_n>::max_digits10);
        if (header) out << "scc2020\n2\n" << this->get_num_cols() << " " << this->get_num_rows() << " 0\n";
        for (index i = 0; i < this->get_num_cols(); ++i) {
            Degree_traits<real_degree_type>::write_degree(out, real_column_degree(i));
            out << " ;";
            for (index r : this->data[i]) out << " " << r;
            out << '\n';
        }
        for (index i = 0; i < this->get_num_rows(); ++i) {
            Degree_traits<real_degree_type>::write_degree(out, real_row_degree(i));
            out << " ;\n";
        }
    }
    template <typename OutputStream>
    void to_stream_r2(OutputStream& out) const { to_stream(out); }
    void print_grid() const {
        std::cout << "x_grid: "; for (const auto& x : x_grid) std::cout << x << " ";
        std::cout << "\ny_grid: "; for (const auto& y : y_grid) std::cout << y << " ";
        std::cout << '\n';
    }
    void print_grid_representation() const {
        std::cout << "Z^2 Column Degrees: ";
        for (const auto& d : this->col_degrees) std::cout << "(" << d[0] << ", " << d[1] << ") ";
        std::cout << "\nZ^2 Row Degrees: ";
        for (const auto& d : this->row_degrees) std::cout << "(" << d[0] << ", " << d[1] << ") ";
        std::cout << '\n';
    }
    void print_degrees() const { real_matrix_copy().print_degrees(); }
    void print_graded(bool suppress_description = false) const { real_matrix_copy().print_graded(suppress_description); }

    friend Self operator*(const Self& left, const Self& right) {
        if (left.get_num_cols() != right.get_num_rows() ||
            left.real_column_degrees() != right.real_row_degrees())
            throw std::invalid_argument("Composition requires the same intermediate graded basis");
        Self a = left, b = right;
        a.merge_grids(b);
        MatrixBase data = static_cast<const MatrixBase&>(a) * static_cast<const MatrixBase&>(b);
        Self result(std::move(data));
        result.row_degrees = a.row_degrees; result.col_degrees = b.col_degrees;
        result.copy_embedding(a); result.refresh_compatible_sorted();
        return result;
    }
    friend Self operator+(const Self& left, const Self& right) {
        left.require_same_rows(right);
        if (left.real_column_degrees() != right.real_column_degrees())
            throw std::invalid_argument("Addition requires the same graded domain basis");
        Self a = left, b = right;
        a.merge_grids(b);
        MatrixBase data = static_cast<const MatrixBase&>(a) + static_cast<const MatrixBase&>(b);
        Self result(std::move(data));
        result.col_degrees = a.col_degrees; result.row_degrees = a.row_degrees;
        result.copy_embedding(a); result.refresh_compatible_sorted();
        return result;
    }

private:
    // Reuse the existing real-coordinate parser and quiver implementation.
    struct RealMatrix : CoordinateGradedSparseMatrix<real_n, 2, index, RealMatrix, MatrixBase> {
        using RealBase = CoordinateGradedSparseMatrix<real_n, 2, index, RealMatrix, MatrixBase>;
        using RealBase::RealBase;
    };

    static void check_coordinate(const real_n& value) {
        if constexpr (std::is_floating_point_v<real_n>)
            if (!std::isfinite(value)) throw std::invalid_argument("Grid coordinates must be finite");
    }
    static void validate_grid(const vec<real_n>& grid) {
        if (grid.size() > static_cast<std::size_t>(std::numeric_limits<index>::max()))
            throw std::length_error("Grid exceeds the index range");
        for (std::size_t i = 0; i < grid.size(); ++i) {
            check_coordinate(grid[i]);
            if (i && !(grid[i-1] < grid[i]))
                throw std::invalid_argument("Grid coordinates must be strictly increasing");
        }
    }
    static std::unordered_map<real_n, index> make_index_map(const vec<real_n>& grid) {
        validate_grid(grid);
        std::unordered_map<real_n, index> result;
        result.reserve(grid.size());
        for (std::size_t i = 0; i < grid.size(); ++i) result.emplace(grid[i], static_cast<index>(i));
        return result;
    }
    static void sort_unique(vec<real_n>& grid) {
        std::sort(grid.begin(), grid.end());
        grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    }
    static vec<real_n> grid_union(const vec<real_n>& a, const vec<real_n>& b) {
        vec<real_n> result;
        result.reserve(a.size() + b.size());
        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
        return result;
    }
    void validate_index_degree(const degree_type& d) const {
        if (d[0] < 0 || d[1] < 0 || static_cast<std::size_t>(d[0]) >= x_grid.size() ||
            static_cast<std::size_t>(d[1]) >= y_grid.size())
            throw std::invalid_argument("Degree outside the grid embedding");
    }
    void validate_degree_indices() const {
        if (this->col_degrees.size() != static_cast<std::size_t>(this->get_num_cols()) ||
            this->row_degrees.size() != static_cast<std::size_t>(this->get_num_rows()))
            throw std::invalid_argument("Degree counts must match matrix dimensions");
        for (const auto* degrees : {&this->col_degrees, &this->row_degrees})
            for (const auto& d : *degrees) validate_index_degree(d);
    }
    void validate_embedding() const {
        validate_grid(x_grid); validate_grid(y_grid); validate_degree_indices();
        auto check_map = [](const auto& grid, const auto& map) {
            if (grid.size() != map.size()) throw std::invalid_argument("Grid index map is stale");
            for (std::size_t i = 0; i < grid.size(); ++i) {
                auto found = map.find(grid[i]);
                if (found == map.end() || found->second != static_cast<index>(i))
                    throw std::invalid_argument("Grid index map is stale");
            }
        };
        check_map(x_grid, x_to_index); check_map(y_grid, y_to_index);
    }
    void copy_embedding(const Self& other) {
        x_grid = other.x_grid; y_grid = other.y_grid;
        x_to_index = other.x_to_index; y_to_index = other.y_to_index;
    }
    void invalidate_grid_workspaces() {
        grid_scheduler = Grid_scheduler<index>(); pq_row.clear();
        this->invalidate_cached_rows(); this->invalidate_compatible_sorting();
        this->col_batches.clear();
    }
    vec<real_degree_type> embedded_degrees(const vec<degree_type>& degrees) const {
        vec<real_degree_type> result;
        result.reserve(degrees.size());
        for (const auto& d : degrees) result.push_back(real_degree(d));
        return result;
    }
    void require_same_rows(const Self& other) const {
        if (this->get_num_rows() != other.get_num_rows() || real_row_degrees() != other.real_row_degrees())
            throw std::invalid_argument("Matrices must have the same embedded ambient generators");
    }
    void include_real_degrees(const vec<real_degree_type>& degrees) {
        vec<real_n> xs = x_grid, ys = y_grid;
        for (const auto& d : degrees) {
            check_coordinate(d[0]); check_coordinate(d[1]);
            xs.push_back(d[0]); ys.push_back(d[1]);
        }
        sort_unique(xs); sort_unique(ys);
        reindex_grid(std::move(xs), std::move(ys));
    }
    static index floor_index(const vec<real_n>& grid, const real_n& coordinate) {
        if constexpr (std::is_floating_point_v<real_n>)
            if (std::isnan(coordinate)) throw std::invalid_argument("A degree query cannot contain NaN");
        return static_cast<index>(std::upper_bound(grid.begin(), grid.end(), coordinate) - grid.begin()) - 1;
    }
    static index ceiling_index(const vec<real_n>& grid, const real_n& coordinate) {
        return static_cast<index>(std::lower_bound(grid.begin(), grid.end(), coordinate) - grid.begin());
    }
    degree_type query_degree(const real_degree_type& d) const {
        auto point = get_closest_smaller_grid_point(d);
        return {point.first, point.second};
    }
    static index count_before(const vec<degree_type>& degrees, const degree_type& d) {
        return static_cast<index>(std::count_if(degrees.begin(), degrees.end(), [&](const auto& a) {
            return Degree_traits<degree_type>::smaller_equal(a, d);
        }));
    }
    static real_degree_type grid_step(const pair<real_degree_type>& box, int divisor) {
        return {(box.second[0]-box.first[0])/real_n(divisor), (box.second[1]-box.first[1])/real_n(divisor)};
    }
    RealMatrix real_matrix_copy() const {
        RealMatrix result(this->get_num_cols(), this->get_num_rows(),
                          real_column_degrees(), real_row_degrees());
        static_cast<MatrixBase&>(result) = static_cast<const MatrixBase&>(*this);
        return result;
    }

public:
    void parse_stream(std::istream& stream, bool lex_sort = false, bool compute_batches = false) {
        RealMatrix parsed(stream, lex_sort, compute_batches);
        Self result(parsed);
        *this = std::move(result);
    }
};

} // namespace graded_linalg
