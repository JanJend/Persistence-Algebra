/**
 * @file chain_complex.hpp
 * @brief Poset-independent storage and SCC I/O for graded chain complexes.
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <grlina/matrices/graded.hpp>
#include <grlina/matrices/io.hpp>
#include <grlina/matrices/grid.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

/** Basis changes tracked by ChainComplex::sort_compatibly and minimize.
 * Initialize with identities on each chain group; original bases stay fixed.
 */
template<class Matrix>
struct ChainBasisMaps {
    vec<Matrix> to_original;   // current -> original
    vec<Matrix> from_original; // original -> current
};

/**
 * A finite chain complex whose entries all have one concrete graded-matrix
 * type.  Differentials are stored in homological order: differentials()[0]
 * is d1 : C1 -> C0, differentials()[1] is d2 : C2 -> C1, and so on.
 *
 * The class is generic in Matrix and therefore has no knowledge of the
 * underlying degree type or poset. Matrix supplies those through its members,
 * Degree_traits, and GradedMatrixIO.
 */
template <typename Matrix>
class ChainComplex {
public:
    static_assert(is_graded_sparse_matrix_v<Matrix>,
                  "ChainComplex<Matrix> requires the graded sparse matrix interface");
    using degree_type = base_degree<Matrix>;
    using index_type = matrix_index_t<Matrix>;

private:
    using matrix_io = GradedMatrixIO<Matrix>;
    using serialized_degree_type = typename matrix_io::serialized_degree_type;
    std::vector<Matrix> differentials_;

    static std::string trim(std::string value) {
        auto not_space = [](unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
        value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
        return value;
    }

    static void next_scc_line(std::istream& input, std::string& line) {
        while (std::getline(input, line)) {
            line = trim(line.substr(0, line.find('#')));
            if (!line.empty()) return;
        }
        throw std::runtime_error("Unexpected end of SCC data");
    }

    static std::pair<serialized_degree_type, std::vector<index_type>> parse_degree_line(
        const std::string& line, bool with_entries, index_type row_count) {
        std::istringstream input(line);
        serialized_degree_type degree = Degree_traits<serialized_degree_type>::from_stream(input);
        if (!input) {
            throw std::runtime_error("Invalid degree in SCC line: " + line);
        }

        char separator = '\0';
        input >> separator;
        if (separator != ';') {
            throw std::runtime_error("Expected ';' in SCC line: " + line);
        }

        std::vector<index_type> entries;
        index_type entry;
        while (true) {
            input >> std::ws;
            if (input.eof()) break;
            if (!(input >> entry)) throw std::runtime_error("Invalid token in SCC entries: " + line);
            if (!with_entries) {
                throw std::runtime_error("Entries are not allowed on the final SCC chain group");
            }
            if (entry < 0 || entry >= row_count) {
                throw std::runtime_error("SCC matrix entry is outside its target chain group");
            }
            long long coefficient = 1;
            if (input.peek() == ':') {
                input.get();
                if (!(input >> coefficient)) throw std::runtime_error("Invalid SCC coefficient: " + line);
            }
            if (coefficient % 2 != 0) entries.push_back(entry);
        }
        if (!input.eof()) throw std::runtime_error("Invalid token in SCC entries: " + line);
        if (!std::is_sorted(entries.begin(), entries.end()) ||
            std::adjacent_find(entries.begin(), entries.end()) != entries.end()) {
            throw std::runtime_error("SCC columns must contain sorted, unique row indices");
        }
        return {std::move(degree), std::move(entries)};
    }

public:
    ChainComplex() = default;

    explicit ChainComplex(std::vector<Matrix> differentials)
        : differentials_(std::move(differentials)) { matrix_io::normalize_complex(differentials_); }

    // Unlike an initializer_list, forwarding preserves moves of owned matrices.
    template<class... Matrices, std::enable_if_t<(sizeof...(Matrices) > 0) &&
        (std::is_same_v<Matrix, std::decay_t<Matrices>> && ...), int> = 0>
    explicit ChainComplex(Matrices&&... differentials) {
        differentials_.reserve(sizeof...(Matrices));
        (differentials_.emplace_back(std::forward<Matrices>(differentials)), ...);
        matrix_io::normalize_complex(differentials_);
    }

    explicit ChainComplex(std::istream& input, bool sort_if_needed = false) {
        *this = from_stream(input, sort_if_needed);
    }

    explicit ChainComplex(const std::string& path, bool sort_if_needed = false) {
        *this = from_file(path, sort_if_needed);
    }

    bool empty() const noexcept { return differentials_.empty(); }
    std::size_t size() const noexcept { return differentials_.size(); }

    const std::vector<Matrix>& differentials() const noexcept { return differentials_; }
    std::vector<Matrix>& differentials() noexcept { return differentials_; }

    const Matrix& differential(std::size_t homological_degree) const {
        GRLINA_DEBUG_CHECK(if (homological_degree == 0 || homological_degree > differentials_.size()) {
            throw std::out_of_range("Differentials are numbered d1, d2, ...");
        });
        return differentials_[homological_degree - 1];
    }

    Matrix& differential(std::size_t homological_degree) {
        return const_cast<Matrix&>(std::as_const(*this).differential(homological_degree));
    }

    const Matrix& operator[](std::size_t zero_based_index) const {
        GRLINA_ASSERT(zero_based_index < differentials_.size());
        return differentials_[zero_based_index];
    }

    Matrix& operator[](std::size_t zero_based_index) {
        GRLINA_ASSERT(zero_based_index < differentials_.size());
        return differentials_[zero_based_index];
    }

    /** Test the chain equations d_i d_(i+1) = 0. */
    bool is_chain_complex() const {
        for (std::size_t i = 1; i < differentials_.size(); ++i) {
            const auto& lower = differentials_[i - 1];
            const auto& upper = differentials_[i];
            GRLINA_ASSERT(lower.get_num_cols() == upper.get_num_rows());
            // Over F2 each product column is the XOR of its selected boundaries.
            // Avoid testing every unrelated row/column pair or storing d^2.
            for (index_type column = 0; column < upper.get_num_cols(); ++column)
                if (!vectorXORMulti<index_type>(lower.data, upper.get_col(column)).empty())
                    return false;
        }
        return true;
    }

    void push_differential(Matrix differential) {
        differentials_.push_back(std::move(differential));
        matrix_io::normalize_complex(differentials_);
    }

    void clear() noexcept { differentials_.clear(); }

    /** Actual (decoded for grid matrices) degrees of C_q; absent groups are zero. */
    vec<typename Matrix::degree_type> group_degrees(std::size_t q) const {
        if (empty() || q > size()) return {};
        return q == 0 ? geometric_row_degrees(differentials_[0])
                      : geometric_col_degrees(differentials_[q - 1]);
    }

    void sort_compatibly() {
        sort_compatibly(TraitLinearOrder<degree_type>{Degree_traits<degree_type>::lex_lambda()});
    }

    template <typename Compare>
    void sort_compatibly(Compare compare, ChainBasisMaps<Matrix>* basis_maps = nullptr,
                         Matrix* elements = nullptr, const vec<Matrix*>& generator_images = {},
                         vec<Matrix>* incoming = nullptr, vec<Matrix>* outgoing = nullptr) {
        if (empty()) return;
        GRLINA_ASSERT(!basis_maps || (basis_maps->to_original.size() > size() &&
                                     basis_maps->from_original.size() > size()));
        GRLINA_ASSERT(!incoming || incoming->size() > size());
        GRLINA_ASSERT(!outgoing || outgoing->size() > size());

        // Each group is sorted ONCE; its basis permutation is shared by both
        // adjacent maps. This includes stable handling of repeated degrees.
        for (std::size_t group = 0; group <= size(); ++group) {
            const auto& stored_degrees = group == 0 ? differentials_[0].row_degrees
                                                   : differentials_[group - 1].col_degrees;
            std::vector<degree_type> degrees(stored_degrees.begin(), stored_degrees.end());
            auto new_to_old = sort_and_get_permutation<degree_type, index_type>(degrees, compare);
            vec<index_type> old_to_new(new_to_old.size());
            for (index_type i = 0; i < static_cast<index_type>(new_to_old.size()); ++i)
                old_to_new[new_to_old[i]] = i;
            if (incoming) (*incoming)[group].permute_rows_graded(old_to_new);
            if (outgoing) (*outgoing)[group].permute_columns_graded(new_to_old);
            if (basis_maps) {
                auto& inclusion = basis_maps->to_original[group];
                inclusion.permute_columns(new_to_old);
                inclusion.col_degrees = degrees;
                inclusion.compatibly_sorted = false;
                basis_maps->from_original[group].permute_rows_graded(old_to_new);
            }
            if (group > 0) {
                auto& outgoing = differentials_[group - 1];
                outgoing.permute_columns(new_to_old);
                outgoing.col_degrees = degrees;
                outgoing.compatibly_sorted = false;
            }
            if (group < size()) {
                if (group == 0) differentials_[group].permute_rows_graded(old_to_new, elements, generator_images);
                else differentials_[group].permute_rows_graded(old_to_new);
            }
        }
        for (auto& d : differentials_) d.compatibly_sorted = true;

    }

    /** Remove contractible equal-degree pairs by pivot reduction and batch
     * compaction, preserving chain-homotopy type. No exactness assumption or
     * graded kernel is needed. Adjacent maps share the surviving bases; their
     * paired indices are deleted once per differential. Terminal cycles are
     * retained: deleting redundant terminal generators can change homology.
     * Runs in place from higher to lower homological degree, updating supplied
     * basis maps directly and discarding paired lower columns before reduction.
     * Alternatively, incoming[q] lands in C_q and outgoing[q] starts in C_q.
     * They must be compatible chain maps, with an explicit matrix for every
     * group (including C0); they are transported directly, without basis maps.
     */
    void minimize(bool sort_if_needed = true, ChainBasisMaps<Matrix>* basis_maps = nullptr,
                  Matrix* elements = nullptr, const vec<Matrix*>& generator_images = {},
                  vec<Matrix>* incoming = nullptr, vec<Matrix>* outgoing = nullptr) {
        if (empty()) return;
        GRLINA_DEBUG_CHECK(if (basis_maps && elements)
            throw std::invalid_argument("Use either full chain basis maps or an incoming generator map"));
        GRLINA_DEBUG_CHECK(if ((incoming || outgoing) && (basis_maps || elements || !generator_images.empty()))
            throw std::invalid_argument("Use either attached chain maps or the existing basis/generator tracking arguments"));
        GRLINA_ASSERT(!basis_maps || (basis_maps->to_original.size() > size() &&
                                     basis_maps->from_original.size() > size()));
        GRLINA_ASSERT(!incoming || incoming->size() > size());
        GRLINA_ASSERT(!outgoing || outgoing->size() > size());
        if constexpr (matrix_grid_backed_v<Matrix>) {
            if (incoming || outgoing) {
                auto common = empty_matrix_like(differentials_[0], 0, 0);
                for (auto* collection : {&differentials_, incoming, outgoing})
                    if (collection) for (auto& matrix : *collection) merge_grids(common, matrix);
                for (auto* collection : {&differentials_, incoming, outgoing})
                    if (collection) for (auto& matrix : *collection) matrix.reindex_grid(common.grid_);
            }
        }
        if (sort_if_needed) sort_compatibly(
            TraitLinearOrder<degree_type>{Degree_traits<degree_type>::lex_lambda()},
            basis_maps, elements, generator_images, incoming, outgoing);

        auto& maps = differentials_;
        for (std::size_t remaining = maps.size(); remaining > 0; --remaining) {
            const std::size_t level = remaining - 1;
            auto& d = maps[level];
            if constexpr (matrix_grid_backed_v<Matrix>) {
                if (level == 0 && elements) {
                    merge_grids(d, *elements);
                    GradedMatrixIO<Matrix>::normalize_complex(maps);
                }
            }
            d.cancel_local_pairs(basis_maps ? &basis_maps->from_original[level] :
                    (incoming ? &(*incoming)[level] : (level == 0 ? elements : nullptr)),
                basis_maps ? &basis_maps->to_original[level + 1] : (outgoing ? &(*outgoing)[level + 1] : nullptr),
                [&](auto& columns, auto& rows) {
                // Target survivors retain their basis; source survivors change
                // only by paired columns. The inverse changes in both adjacent
                // differentials therefore affect only indices being removed.
                if (level > 0) maps[level - 1].delete_columns(rows);
                if (level + 1 < maps.size()) maps[level + 1].delete_rows(columns);
                if (level == 0)
                    for (auto* images : generator_images) if (images) images->delete_columns(rows);
                if (basis_maps) {
                    basis_maps->to_original[level].delete_columns(rows);
                    basis_maps->from_original[level + 1].delete_rows(columns);
                }
                if (outgoing) (*outgoing)[level].delete_columns(rows);
                if (incoming) (*incoming)[level + 1].delete_rows(columns);
            });
        }
    }

    template <typename OutputStream>
    void to_stream(OutputStream& output) const {
        static_assert(has_degree_io_v<serialized_degree_type>,
                      "Chain SCC I/O requires Degree_traits::poset_id, write_degree and from_stream");
        output << std::setprecision(matrix_io::output_precision());
        output << "scc2020\n" << Degree_traits<serialized_degree_type>::poset_id << "\n";

        if (differentials_.empty()) {
            output << "0 0 0\n";
            return;
        }

        output << differentials_.back().get_num_cols();
        for (auto it = differentials_.rbegin(); it != differentials_.rend(); ++it) {
            output << " " << it->get_num_rows();
        }
        if (differentials_.size() == 1) output << " 0";
        output << "\n";

        for (auto it = differentials_.rbegin(); it != differentials_.rend(); ++it) {
            for (index_type column = 0; column < it->get_num_cols(); ++column) {
                matrix_io::write_column_degree(output, *it, column);
                output << " ;";
                for (const auto row : it->data[column]) output << " " << row;
                output << "\n";
            }
        }
        for (index_type row = 0; row < differentials_.front().get_num_rows(); ++row) {
            matrix_io::write_row_degree(output, differentials_.front(), row);
            output << " ;\n";
        }
    }

    void to_file(const std::string& path) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Unable to open SCC output file: " + path);
        to_stream(output);
    }

    static ChainComplex from_stream(std::istream& input, bool sort_if_needed = false) {
        static_assert(has_degree_io_v<serialized_degree_type>,
                      "Chain SCC I/O requires Degree_traits::poset_id, write_degree and from_stream");
        std::string line;
        std::string file_poset;
        next_scc_line(input, line);
        const bool firep = trim(line) == "firep";
        if (firep) {
            // FIREP carries axis labels; the matrix type supplies the degree domain.
            next_scc_line(input, line);
            next_scc_line(input, line);
            file_poset = std::string(Degree_traits<serialized_degree_type>::poset_id);
        } else if (trim(line) == "scc2020") {
            next_scc_line(input, file_poset);
        } else {
            throw std::runtime_error("Expected scc2020 or firep header");
        }
        file_poset = trim(file_poset);
        const std::string expected_poset(Degree_traits<serialized_degree_type>::poset_id);
        if (file_poset != expected_poset)
            throw std::runtime_error("SCC poset identifier '" + file_poset +
                                     "' does not match matrix poset '" + expected_poset + "'");

        next_scc_line(input, line);
        if (firep) {
            // Three-coordinate FIREP may include a third axis label.
            std::istringstream probe(line);
            index_type first_rank;
            if (!(probe >> first_rank)) next_scc_line(input, line);
        }
        std::istringstream dimensions_stream(line);
        std::vector<index_type> ranks;
        index_type rank;
        while (dimensions_stream >> rank) {
            if (rank < 0) throw std::runtime_error("Negative SCC chain-group dimension");
            ranks.push_back(rank);
        }
        if (!dimensions_stream.eof()) throw std::runtime_error("Invalid SCC chain dimensions");
        if (ranks.size() < 2) throw std::runtime_error("SCC requires at least two chain-group dimensions");

        // A presentation's final zero is the conventional dummy chain group.
        // Always retain at least two ranks: 0 0 0 represents the zero module.
        while (ranks.size() > 2 && ranks.back() == 0) ranks.pop_back();

        struct Group {
            std::vector<serialized_degree_type> degrees;
            matrix_storage_t<Matrix> columns;
        };
        std::vector<Group> groups(ranks.size());

        for (std::size_t group_index = 0; group_index < ranks.size(); ++group_index) {
            const bool with_entries = group_index + 1 < ranks.size();
            const index_type target_rank = with_entries ? ranks[group_index + 1] : 0;
            groups[group_index].degrees.reserve(static_cast<std::size_t>(ranks[group_index]));
            if (with_entries) groups[group_index].columns.reserve(static_cast<std::size_t>(ranks[group_index]));

            for (index_type element = 0; element < ranks[group_index]; ++element) {
                next_scc_line(input, line);
                auto parsed = parse_degree_line(line, with_entries, target_rank);
                groups[group_index].degrees.push_back(std::move(parsed.first));
                if (with_entries) groups[group_index].columns.push_back(std::move(parsed.second));
            }
        }

        std::vector<Matrix> high_to_low;
        high_to_low.reserve(groups.size() - 1);
        for (std::size_t i = 0; i + 1 < groups.size(); ++i) {
            Matrix differential = matrix_io::make_matrix(ranks[i], ranks[i + 1]);
            matrix_io::assign_degrees(differential, groups[i].degrees, groups[i + 1].degrees);
            differential.assign_data(std::move(groups[i].columns));
            differential.compatibly_sorted = false;
            high_to_low.push_back(std::move(differential));
        }

        std::reverse(high_to_low.begin(), high_to_low.end());
        ChainComplex result(std::move(high_to_low));
        if (sort_if_needed) result.sort_compatibly();
        return result;
    }

    static ChainComplex from_file(const std::string& path, bool sort_if_needed = false) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Unable to open SCC input file: " + path);
        return from_stream(input, sort_if_needed);
    }
};

template<class Matrix>
ChainBasisMaps<Matrix> identity_basis_maps(const ChainComplex<Matrix>& complex) {
    ChainBasisMaps<Matrix> maps;
    if (complex.empty()) return maps;
    for (std::size_t group = 0; group <= complex.size(); ++group) {
        std::size_t differential = 0;
        if (group > 0) differential = group - 1;
        const Matrix& reference = complex[differential];
        auto degrees = geometric_row_degrees(reference);
        if (group > 0) degrees = geometric_col_degrees(reference);
        Matrix identity(degrees);
        if constexpr (matrix_grid_backed_v<Matrix>) {
            identity.reindex_grid(reference.grid_);
        }
        identity.compatibly_sorted = reference.compatibly_sorted;
        maps.to_original.push_back(std::move(identity));
    }
    maps.from_original = maps.to_original;
    return maps;
}

// matrices[n] maps domain Cn to target Cn. Shared ownership keeps the endpoints
// alive. Mutating an endpoint invalidates other maps using its old basis.
// An explicit copy of a ChainMap copies its endpoints; moving transfers ownership.
template <typename Matrix>
class ChainMap {
public:
    using chain_complex_type = ChainComplex<Matrix>;

private:
    std::shared_ptr<chain_complex_type> domain_;
    std::shared_ptr<chain_complex_type> target_;
    std::vector<Matrix> matrices_;

public:
    ChainMap(std::shared_ptr<chain_complex_type> domain,
             std::shared_ptr<chain_complex_type> target,
             std::vector<Matrix> matrices)
        : domain_(std::move(domain)), target_(std::move(target)), matrices_(std::move(matrices)) {}

    ChainMap(const ChainMap& other)
        : domain_(std::make_shared<chain_complex_type>(other.domain())),
          target_(other.domain_ == other.target_ ? domain_ :
                  std::make_shared<chain_complex_type>(other.target())), matrices_(other.matrices_) {}
    ChainMap(ChainMap&&) noexcept = default;
    ChainMap& operator=(ChainMap&&) noexcept = default;
    ChainMap& operator=(const ChainMap& other) {
        if (this != &other) *this = ChainMap(other);
        return *this;
    }

    chain_complex_type& domain() noexcept { return *domain_; }
    chain_complex_type& target() noexcept { return *target_; }
    const chain_complex_type& domain() const noexcept { return *domain_; }
    const chain_complex_type& target() const noexcept { return *target_; }
    const std::vector<Matrix>& matrices() const noexcept { return matrices_; }
    const Matrix& operator[](std::size_t degree) const { return matrices_[degree]; }

    /** Cancel endpoint pairs and transport f directly as p_target*f*i_source.
     * Requires a coherent graded chain map. Missing components are allowed
     * only if their source or target group is zero. Changes this map in place;
     * endpoint complexes are mutated, invalidating other incident maps.
     */
    void minimize() {
        const auto has_pairs = [](const chain_complex_type& complex) {
            for (const auto& d : complex.differentials())
                for (matrix_index_t<Matrix> j = 0; j < d.get_num_cols(); ++j)
                    for (auto row : d.column(j))
                        if (Degree_traits<base_degree<Matrix>>::equals(d.col_degrees[j], d.row_degrees[row]))
                            return true;
            return false;
        };
        const bool reduce_source = has_pairs(domain());
        const bool reduce_target = domain_ == target_ ? reduce_source : has_pairs(target());
        if (!reduce_source && !reduce_target) return;
        auto& maps = matrices_;
        for (std::size_t q = maps.size(); q <= std::max(domain().size(), target().size()); ++q) {
            const auto from = domain().group_degrees(q), to = target().group_degrees(q);
            Matrix zero = GradedMatrixIO<Matrix>::make_matrix(from.size(), to.size());
            zero.resize_data(from.size());
            set_geometric_degrees(zero, from, to);
            maps.push_back(std::move(zero));
        }
        if (domain_ == target_) {
            domain_->minimize(true, nullptr, nullptr, {}, &maps, &maps);
        } else {
            if (reduce_source) domain_->minimize(true, nullptr, nullptr, {}, nullptr, &maps);
            if (reduce_target) target_->minimize(true, nullptr, nullptr, {}, &maps);
        }
    }

    /** Cone(f)_q = target_q + domain_(q-1), with d = [d_target f; 0 d_source]
     * over F2. Keeps this original basis; call minimize() separately if desired.
     * Requires a coherent graded chain map, as for minimize().
     */
    chain_complex_type mapping_cone() const {
        using Index = matrix_index_t<Matrix>;
        const auto top = std::max(target().size(), domain().empty() ? 0 : domain().size() + 1);
        vec<Matrix> differentials;
        auto rows = target().group_degrees(0);
        for (std::size_t q = 1; q <= top; ++q) {
            auto columns = target().group_degrees(q);
            const auto source_columns = domain().group_degrees(q - 1);
            columns.insert(columns.end(), source_columns.begin(), source_columns.end());
            if (columns.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
                rows.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()))
                throw std::length_error("Mapping-cone group exceeds matrix index range");
            Matrix d = GradedMatrixIO<Matrix>::make_matrix(columns.size(), rows.size());
            d.clear_data();
            set_geometric_degrees(d, columns, rows);
            if (q <= target().size())
                for (Index j = 0; j < target().differential(q).get_num_cols(); ++j)
                    d.append_col(target().differential(q).get_col(j));
            const auto offset = target().group_degrees(q - 1).size();
            for (std::size_t j = 0; j < source_columns.size(); ++j) {
                vec<Index> column;
                if (q - 1 < matrices_.size()) column = matrices_[q - 1].get_col(j);
                if (q > 1)
                    for (Index row : domain().differential(q - 1).column(j)) column.push_back(offset + row);
                d.append_col(std::move(column));
            }
            differentials.push_back(std::move(d));
            rows = std::move(columns);
        }
        return chain_complex_type(std::move(differentials));
    }

    /** Brutal truncation to [low, high], preserving absolute homological indices.
     * Mutates the endpoints and invalidates other incident maps. Boundary homology
     * can change; the caller must choose a sufficient window for its computation.
     */
    void truncate(std::size_t low, std::size_t high) {
        if (low > high) throw std::invalid_argument("Truncation requires low <= high");
        const auto cut = [&](chain_complex_type& complex) {
            const auto top = std::min(high, complex.size());
            if (complex.empty() || low > top) { complex.clear(); return; }
            // Keep retained matrices in their existing allocations. A lone C0
            // is represented by its zero incoming differential d1.
            complex.differentials().resize(std::max(std::size_t{1}, top));
            for (std::size_t q = 1; q <= complex.size(); ++q) {
                if (q > low && q <= top) continue;
                auto columns = q >= low && q <= top ? complex.group_degrees(q) : vec<typename Matrix::degree_type>{};
                auto rows = q - 1 >= low && q - 1 <= top ? complex.group_degrees(q - 1) : vec<typename Matrix::degree_type>{};
                Matrix d = GradedMatrixIO<Matrix>::make_matrix(columns.size(), rows.size());
                d.resize_data(columns.size());
                set_geometric_degrees(d, columns, rows);
                complex.differential(q) = std::move(d);
            }
        };
        cut(domain());
        if (domain_ != target_) cut(target());
        const auto previous_size = matrices_.size();
        matrices_.resize(std::max(domain().size(), target().size()) + 1);
        for (std::size_t q = 0; q < matrices_.size(); ++q) {
            if (q >= low && q <= high && q < previous_size) continue;
            const auto from = domain().group_degrees(q), to = target().group_degrees(q);
            Matrix zero = GradedMatrixIO<Matrix>::make_matrix(from.size(), to.size());
            zero.resize_data(from.size());
            set_geometric_degrees(zero, from, to);
            matrices_[q] = std::move(zero);
        }
    }
};

} // namespace graded_linalg
