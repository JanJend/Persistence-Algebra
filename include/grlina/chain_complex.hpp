/**
 * @file chain_complex.hpp
 * @brief Poset-independent storage and SCC I/O for graded chain complexes.
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <grlina/graded_matrix.hpp>

namespace graded_linalg {

/**
 * A finite chain complex whose entries all have one concrete graded-matrix
 * type.  Differentials are stored in homological order: differentials()[0]
 * is d1 : C1 -> C0, differentials()[1] is d2 : C2 -> C1, and so on.
 *
 * The class is generic in Matrix and therefore has no knowledge of the
 * underlying degree type or poset.  Matrix supplies those through its
 * inherited degree_type/index_type aliases and Degree_traits.
 */
template <typename Matrix>
class ChainComplex {
public:
    using matrix_type = Matrix;
    using degree_type = typename Matrix::degree_type;
    using index_type = typename Matrix::index_type;

private:
    std::vector<Matrix> differentials_;

    static std::string trim(std::string value) {
        auto not_space = [](unsigned char c) { return !std::isspace(c); };
        value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
        value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
        return value;
    }

    static bool is_unsigned_integer(const std::string& value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
    }

    static std::pair<degree_type, std::vector<index_type>> parse_degree_line(
        const std::string& line, bool with_entries, index_type row_count) {
        std::istringstream input(line);
        degree_type degree = Degree_traits<degree_type>::from_stream(input);
        if (!input) {
            throw std::runtime_error("Invalid degree in SCC line: " + line);
        }

        std::string separator;
        input >> separator;
        if (separator != ";") {
            throw std::runtime_error("Expected ';' in SCC line: " + line);
        }

        std::vector<index_type> entries;
        index_type entry;
        while (input >> entry) {
            if (!with_entries) {
                throw std::runtime_error("Entries are not allowed on the final SCC chain group");
            }
            if (entry < 0 || entry >= row_count) {
                throw std::runtime_error("SCC matrix entry is outside its target chain group");
            }
            entries.push_back(entry);
        }
        if (!std::is_sorted(entries.begin(), entries.end()) ||
            std::adjacent_find(entries.begin(), entries.end()) != entries.end()) {
            throw std::runtime_error("SCC columns must contain sorted, unique row indices");
        }
        return {std::move(degree), std::move(entries)};
    }

public:
    ChainComplex() = default;

    explicit ChainComplex(std::vector<Matrix> differentials, bool validate = true)
        : differentials_(std::move(differentials)) {
        if (validate) validate_structure();
    }

    ChainComplex(std::initializer_list<Matrix> differentials)
        : differentials_(differentials) {
        validate_structure();
    }

    explicit ChainComplex(std::istream& input) {
        *this = from_stream(input);
    }

    explicit ChainComplex(const std::string& path) {
        *this = from_file(path);
    }

    bool empty() const noexcept { return differentials_.empty(); }
    std::size_t size() const noexcept { return differentials_.size(); }

    const std::vector<Matrix>& differentials() const noexcept { return differentials_; }
    std::vector<Matrix>& differentials() noexcept { return differentials_; }

    const Matrix& differential(std::size_t homological_degree) const {
        if (homological_degree == 0 || homological_degree > differentials_.size()) {
            throw std::out_of_range("Differentials are numbered d1, d2, ...");
        }
        return differentials_[homological_degree - 1];
    }

    Matrix& differential(std::size_t homological_degree) {
        return const_cast<Matrix&>(std::as_const(*this).differential(homological_degree));
    }

    const Matrix& operator[](std::size_t zero_based_index) const {
        return differentials_.at(zero_based_index);
    }

    Matrix& operator[](std::size_t zero_based_index) {
        return differentials_.at(zero_based_index);
    }

    static std::string poset_identifier() {
        return std::string(Degree_traits<degree_type>::poset_id);
    }

    void validate_structure() const {
        for (std::size_t i = 0; i < differentials_.size(); ++i) {
            const auto& matrix = differentials_[i];
            if (matrix.col_degrees.size() != static_cast<std::size_t>(matrix.get_num_cols()) ||
                matrix.row_degrees.size() != static_cast<std::size_t>(matrix.get_num_rows()) ||
                matrix.data.size() != static_cast<std::size_t>(matrix.get_num_cols())) {
                throw std::invalid_argument("A chain differential has inconsistent dimensions");
            }
            if (!matrix.is_graded_matrix()) {
                throw std::invalid_argument("A chain differential is not graded");
            }
            if (i != 0) {
                const auto& previous = differentials_[i - 1];
                if (previous.get_num_cols() != matrix.get_num_rows() ||
                    previous.col_degrees != matrix.row_degrees) {
                    throw std::invalid_argument("Adjacent chain differentials have incompatible chain groups");
                }
            }
        }
    }

    bool squares_to_zero() const {
        validate_structure();
        for (std::size_t i = 1; i < differentials_.size(); ++i) {
            auto composition = differentials_[i - 1] * differentials_[i];
            if (!composition.is_zero()) return false;
        }
        return true;
    }

    bool is_chain_complex() const {
        try {
            return squares_to_zero();
        } catch (const std::exception&) {
            return false;
        }
    }

    void push_differential(Matrix differential) {
        if (!differentials_.empty()) {
            const auto& previous = differentials_.back();
            if (previous.get_num_cols() != differential.get_num_rows() ||
                previous.col_degrees != differential.row_degrees) {
                throw std::invalid_argument("New differential has the wrong target chain group");
            }
        }
        differentials_.push_back(std::move(differential));
        validate_structure();
    }

    void clear() noexcept { differentials_.clear(); }

    void sort_compatibly() {
        for (auto& differential : differentials_) differential.sort_compatibly();
        validate_structure();
    }

    template <typename Compare>
    void sort_compatibly(Compare compare) {
        for (auto& differential : differentials_) differential.sort_compatibly(compare);
        validate_structure();
    }

    template <typename OutputStream>
    void to_stream(OutputStream& output) const {
        validate_structure();
        output << std::setprecision(17);
        output << "scc2020\n" << poset_identifier() << "\n";

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
                Degree_traits<degree_type>::write_degree(output, it->col_degrees[column]);
                output << " ;";
                for (const auto row : it->data[column]) output << " " << row;
                output << "\n";
            }
        }
        for (const auto& degree : differentials_.front().row_degrees) {
            Degree_traits<degree_type>::write_degree(output, degree);
            output << " ;\n";
        }
    }

    void to_file(const std::string& path) const {
        std::ofstream output(path);
        if (!output) throw std::runtime_error("Unable to open SCC output file: " + path);
        to_stream(output);
    }

    static ChainComplex from_stream(std::istream& input) {
        std::string line;
        if (!std::getline(input, line) || trim(line) != "scc2020") {
            throw std::runtime_error("Expected scc2020 header");
        }

        std::string file_poset;
        if (!std::getline(input, file_poset)) throw std::runtime_error("Missing SCC poset identifier");
        file_poset = trim(file_poset);

        if (!std::getline(input, line)) throw std::runtime_error("Missing SCC chain dimensions");
        std::istringstream dimensions_stream(line);
        std::vector<index_type> ranks;
        index_type rank;
        while (dimensions_stream >> rank) {
            if (rank < 0) throw std::runtime_error("Negative SCC chain-group dimension");
            ranks.push_back(rank);
        }
        if (ranks.size() < 2) throw std::runtime_error("SCC requires at least two chain-group dimensions");

        std::size_t effective_rank_count = ranks.size();
        while (effective_rank_count != 0 && ranks[effective_rank_count - 1] == 0)
            --effective_rank_count;
        const std::string expected_poset = poset_identifier();
        const bool legacy_chain_length = is_unsigned_integer(file_poset) &&
            static_cast<std::size_t>(std::stoull(file_poset)) == effective_rank_count;
        if (file_poset != expected_poset && !legacy_chain_length) {
            throw std::runtime_error(
                "SCC poset identifier '" + file_poset + "' does not match matrix poset '" + expected_poset + "'");
        }

        while (!ranks.empty() && ranks.back() == 0) ranks.pop_back();
        if (ranks.empty()) return ChainComplex();
        if (ranks.size() < 2) throw std::runtime_error("SCC contains only one nonzero chain group");

        struct Group {
            std::vector<degree_type> degrees;
            std::vector<std::vector<index_type>> columns;
        };
        std::vector<Group> groups(ranks.size());

        for (std::size_t group_index = 0; group_index < ranks.size(); ++group_index) {
            const bool with_entries = group_index + 1 < ranks.size();
            const index_type target_rank = with_entries ? ranks[group_index + 1] : 0;
            groups[group_index].degrees.reserve(static_cast<std::size_t>(ranks[group_index]));
            if (with_entries) groups[group_index].columns.reserve(static_cast<std::size_t>(ranks[group_index]));

            for (index_type element = 0; element < ranks[group_index]; ++element) {
                if (!std::getline(input, line)) throw std::runtime_error("Unexpected end of SCC data");
                auto parsed = parse_degree_line(line, with_entries, target_rank);
                groups[group_index].degrees.push_back(std::move(parsed.first));
                if (with_entries) groups[group_index].columns.push_back(std::move(parsed.second));
            }
        }

        std::vector<Matrix> high_to_low;
        high_to_low.reserve(groups.size() - 1);
        for (std::size_t i = 0; i + 1 < groups.size(); ++i) {
            Matrix differential(ranks[i], ranks[i + 1]);
            differential.col_degrees = groups[i].degrees;
            differential.row_degrees = groups[i + 1].degrees;
            differential.data = groups[i].columns;
            high_to_low.push_back(std::move(differential));
        }

        std::reverse(high_to_low.begin(), high_to_low.end());
        return ChainComplex(std::move(high_to_low));
    }

    static ChainComplex from_file(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Unable to open SCC input file: " + path);
        return from_stream(input);
    }
};

} // namespace graded_linalg
