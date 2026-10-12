/** General container, file and console utilities, independent of the algebra. */
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <grlina/general_checks.hpp>

namespace graded_linalg {

template <typename T>
using vec = std::vector<T>;
template <typename T>
using array = vec<vec<T>>;
template <typename T>
using pair = std::pair<T, T>;
template <typename T>
using set = std::set<T>;

// Declare array formatting before the container formatters so nested degree
// containers also find it through ordinary lookup (std::array has no grlina ADL).
template <typename Scalar, std::size_t Dimension>
std::ostream& operator<<(std::ostream& out, const std::array<Scalar, Dimension>& degree) {
    out << "(";
    for (std::size_t i = 0; i < Dimension; ++i) {
        if (i != 0) out << ", ";
        out << degree[i];
    }
    return out << ")";
}


template <typename T>
inline std::ostream& operator<< (std::ostream& ostr, const vec<T>& c) {
    for(T i:c) {
      ostr << i << " ";
    }
    return ostr;
}

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << *it;
        if (std::next(it) != v.end()) {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

template<typename T1, typename T2>
std::ostream& operator<<(std::ostream& os, const std::pair<T1, T2>& p) {
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}

template <typename index>
vec<index> sort_by_permutation(const vec<index>& compare_against){
    vec<index> indices(compare_against.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto comparator = [&compare_against](index a, index b) {
        return compare_against[a] < compare_against[b]; // Change < to > if you want descending order
    };
    std::sort(indices.begin(), indices.end(), comparator);
    return indices;
}

/**
 * @brief Sorts the input vector using a custom comparator and returns a vector of indices representing the permutation used to sort.
 *
 * @tparam T Type of the elements in the vector.
 * @tparam index Type of the indices.
 * @param vec The vector to be sorted.
 * @param comp The comparison function to use for sorting.
 * @return std::vector<index> A vector of indices representing the permutation used to sort.
 */
template <typename T, typename index>
std::vector<index> sort_and_get_permutation(std::vector<T>& vec, std::function<bool(const T&, const T&)> comp) {
    // Initialize indices
    std::vector<index> indices(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices based on the custom comparator
    std::stable_sort(indices.begin(), indices.end(), [&vec, &comp](index i1, index i2) {
        return comp(vec[i1], vec[i2]);
    });

    // Create a copy of the original vector
    std::vector<T> sorted_vec(vec.size());
    for (index i = 0; i < vec.size(); ++i) {
        sorted_vec[i] = vec[indices[i]];
    }

    // Assign the sorted vector back to the original vector
    vec = sorted_vec;

    return indices;
}

    template <typename index>
    struct pair_hash {
        std::size_t operator()(const std::pair<index, index>& p) const {
            auto hash1 = std::hash<index>{}(p.first);
            auto hash2 = std::hash<index>{}(p.second);
            return hash1 ^ (hash2 << 1); // or use another method?
        }
    };


    /**
     * @brief Returns a copy of the target vector with the entries at the indices given in the mask.
     *
     * @tparam index
     * @tparam T
     * @param target
     * @param mask
     * @return vec<T>
     */
    template <typename index, typename T>
    vec<T> vec_restriction(vec<T>& target, vec<index>& mask){
        vec<T> result;
        for(index i : mask){
            result.push_back(target[i]);
        }
        return result;
    }

    /**
     * @brief Inverse to vec_restriction. Returns the indices in "target" of the elements in subset.
     * Only works if both are sorted.
     * @tparam index
     * @tparam T
     * @return vec<T>
     */
    template <typename index, typename T>
    vec<index> get_index_vector(vec<T> target, vec<T> subset){
        vec<index> result = vec<index>();
        auto itS = subset.begin();
        for(index i = 0; i < target.size() ; i++){
            if(itS != subset.end() && *itS == target[i]){
                result.push_back(i);
                itS++;
            }
        }
        GRLINA_ASSERT(itS == subset.end() && "Not all elements of the subset were found in the target");
        return result;
    }

    /**
     * @brief Delets all entries of v at the indices given while preserving the order of v.
     *
     * @tparam T
     * @tparam index
     * @param v
     * @param indices
     */
    template<typename T, typename index>
    void vec_deletion(std::vector<T>& v, const std::vector<index>& indices) {

        GRLINA_ASSERT(std::is_sorted(indices.begin(), indices.end()));

        index read = 0;
        index write = 0;
        index j = 0;
        index n = v.size();
        index m = indices.size();

        while (read < n) {
            if (j < m && read == indices[j]) {
                ++read;
                ++j;
            } else {
                if (read != write) {
                    v[write] = std::move(v[read]);
                }
                ++read;
                ++write;
            }
        }

        v.resize(write);
    }

/**
 * @brief Erases i from v and returns true if i was found and erased.
 *
 * @tparam index
 * @param v
 * @param i
 * @return true
 * @return false
 */
template <typename index>
bool erase_from_sorted_vector_bool(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    if(it != v.end() && *it == i){
        v.erase(it);
        return true;
    }
    return false;
}

/**
 * @brief Tries to delete i from v.
 *
 * @tparam index
 * @param v
 * @param i
 */
template <typename index>
void erase_from_sorted_vector(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    if(it != v.end() && *it == i){
        v.erase(it);
    }
}

/**
 * @brief Removes the intersection of v and w from v. Both must be sorted
 *
 * @tparam index
 * @param v
 * @param w
 */
template <typename index>
void remove_intersection(vec<index>& v, const vec<index>& w) {
    GRLINA_ASSERT(std::is_sorted(v.begin(), v.end()));
    GRLINA_ASSERT(std::is_sorted(w.begin(), w.end()));
    index i = 0; // Pointer for v
    index j = 0; // Pointer for w
    index k = 0; // Position to write in v

    while (i < v.size() && j < w.size()) {
        if (v[i] < w[j]) {
            v[k++] = v[i++];
        } else if (v[i] > w[j]) {
            ++j;
        } else { // v[i] == w[j], skip v[i]
            ++i;
            ++j;
        }
    }

    // Copy any remaining elements from v (which are greater than all in w)
    while (i < v.size()) {
        v[k++] = v[i++];
    }

    v.resize(k); // Shrink v to keep only the unique values
}

template <typename index>
void insert_into_sorted_vector(vec<index>& v, index i){
    auto it = std::lower_bound(v.begin(), v.end(), i);
    v.insert(it, i);
}


/**
 * @brief This function is used to delete rows in a LOC sparse matrix.
 * It creates a map which maps the old indices to the new indices.
 *
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index>
 */
template <typename index>
std::unordered_map<index, index> shiftIndicesMap(const vec<index>& indices) {
    std::unordered_map<index, index> indexMap;
    indexMap.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        indexMap[indices[i]] = i;
    }
    return indexMap;
}

/**
 * @brief Expand sorted, distinct surviving indices into a dense old-to-new map.
 * Missing indices map to -1. Moving indices into this overload reuses its buffer.
 */
template <typename index>
vec<index> shiftIndicesMap(vec<index> indices, index num_rows) {
    GRLINA_ASSERT(num_rows >= 0);
    GRLINA_ASSERT(std::is_sorted(indices.begin(), indices.end()));
    GRLINA_ASSERT(std::adjacent_find(indices.begin(), indices.end()) == indices.end());
    GRLINA_ASSERT(indices.empty() || (indices.front() >= 0 && indices.back() < num_rows));
    std::size_t remaining = indices.size();
    indices.resize(num_rows, index(-1));
    // Walk backwards so writing the map cannot overwrite unread surviving indices.
    for (index row = num_rows; row-- > 0;) {
        if (remaining && indices[remaining - 1] == row)
            indices[row] = static_cast<index>(--remaining);
        else indices[row] = index(-1);
    }
    return indices;
}

/**
 * @brief This function is used to delete rows in a LOC sparse matrix.
 * It creates a map which maps the old indices to the new indices.
 *
 * @param indices Holds the indices of the rows which should stay in the matrix.
 * @return std::unordered_map<index, index>
 */
template <typename index>
std::unordered_map< std::pair<index,index>, index, pair_hash<index> > pair_to_index_map(const vec<std::pair<index, index>>& index_pairs) {
    std::unordered_map<std::pair<index, index>, index, pair_hash<index> > indexMap;
    for (std::size_t i = 0; i < index_pairs.size(); ++i) {
        indexMap[index_pairs[i]] = i;
    }
    return indexMap;
}

/**
 * @brief Apply an index map in place, optionally deleting unmapped entries in one pass.
 *
 * @param target
 * @param indexMap
 * @param needsNoDeletion If the target vector only contains indices which are in the indexMap, this can be set to true.
 */
template <typename index>
void apply_transformation(vec<index>& target, const std::unordered_map<index, index>& indexMap, const bool& needsNoDeletion = false) {
    if (needsNoDeletion) {
        for (index& entry : target) {
            auto mapped = indexMap.find(entry);
            GRLINA_ASSERT(mapped != indexMap.end());
            entry = mapped->second;
        }
    } else {
        std::size_t write = 0;
        for (index entry : target) {
            auto mapped = indexMap.find(entry);
            if (mapped != indexMap.end()) target[write++] = mapped->second;
        }
        target.resize(write);
    }
}

/**
 * @brief Apply a transformation given by a vector to a vector of indices.
 *
 * @param target
 * @param index_vector
 * @param needsNoDeletion Defaults to mapping only. If false, discard entries mapped to -1.
 */
template <typename index>
void apply_transformation(vec<index>& target, const vec<index>& index_vector, bool needsNoDeletion = true) {
    if (needsNoDeletion) {
        for (index& entry : target) entry = index_vector[entry];
    } else {
        std::size_t write = 0;
        for (index entry : target) {
            const index mapped = index_vector[entry];
            if (mapped != index(-1)) target[write++] = mapped;
        }
        target.resize(write);
    }
}

/**
 * @brief Re-indxes the entries of v by subtracting the number of entries in w which are smaller than v[i].
 *
 * @tparam index
 * @param v
 * @param w
 */
template<typename index>
void re_index_non_ocurrences(std::vector<index>& v, const std::vector<index>& w) {
    GRLINA_ASSERT(std::is_sorted(v.begin(), v.end()));
    GRLINA_ASSERT(std::is_sorted(w.begin(), w.end()));
    index j = 0;
    index m = w.size();
    for (index& a : v) {
        while (j < m && w[j] < a) {
            ++j;
        }
        a -= j;
    }
}

/**
 * @brief Writes a vector of numbers (arithmetic type) to a .txt file.
 *
 * @tparam T
 * @param vec
 * @param folder
 * @param filename
 */
template <typename T>
void write_vector_to_file(const std::vector<T>& vec, const std::string& folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    // Construct the full path by combining the folder and filename
    std::string full_path = folder + "/" + filename;

    std::ofstream outfile(full_path);

    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (size_t i = 0; i < vec.size(); ++i) {
        outfile << vec[i];
        if (i != vec.size() - 1) {
            outfile << " ";  // Use space as a separator
        }
    }

    outfile.close();

    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}

/**
 * @brief Writes a vector of sets of numbers to a .txt file.
 *
 * @tparam T
 * @param vec
 * @param relative_folder
 * @param filename
 */
template <typename T>
void write_vector_of_sets_to_file(const std::vector<std::set<T>>& vec, const std::string& relative_folder, const std::string& filename) {
    static_assert(std::is_arithmetic<T>::value, "Template parameter T must be a numeric type");

    std::cout << "Writing vector of length " << vec.size() << " to " << filename << " in folder " << relative_folder << std::endl;

    // Construct the full path by combining the relative folder and filename
    std::string full_path = relative_folder + "/" + filename;

    std::ofstream outfile(full_path);

    if (!outfile) {
        std::cerr << "Error: Could not open the file " << full_path << " for writing." << std::endl;
        return;
    }

    for (const auto& s : vec) {
        outfile << "{";
        for (auto it = s.begin(); it != s.end(); ++it) {
            outfile << *it;
            if (std::next(it) != s.end()) {
                outfile << ", ";
            }
        }
        outfile << "}\n";
    }

    outfile.close();

    if (!outfile) {
        std::cerr << "Error: Could not close the file " << full_path << " after writing." << std::endl;
    }
}


/** Console timing utility; joins its worker even if the computation throws.
 * A condition variable avoids an unconditional one-second wait on fast calls.
 */
template <typename Func, typename... Args>
auto timed_with_progress(const std::string& label, Func&& func, Args&&... args)
    -> std::invoke_result_t<Func, Args...> {
    const auto start = std::chrono::steady_clock::now();
    std::mutex mutex;
    std::condition_variable changed;
    bool done = false;
    std::thread worker([&] {
        std::unique_lock<std::mutex> lock(mutex);
        while (!changed.wait_for(lock, std::chrono::seconds(1), [&] { return done; })) {
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << '\r' << label << ": " << seconds << "s" << std::flush;
        }
    });
    auto finish = [&] {
        { std::lock_guard<std::mutex> lock(mutex); done = true; }
        changed.notify_one();
        worker.join();
    };
    auto report = [&] {
        const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << '\r' << label << " completed in: " << seconds << "s" << std::endl;
    };
    try {
        if constexpr (std::is_void_v<std::invoke_result_t<Func, Args...>>) {
            std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
            finish();
            report();
        } else {
            decltype(auto) result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
            finish();
            report();
            return result;
        }
    } catch (...) {
        if (worker.joinable()) finish();
        throw;
    }
}

} // namespace graded_linalg

inline std::string insert_suffix_before_extension(const std::string& filepath, const std::string& suffix, const std::string& new_extension = "") {
    std::filesystem::path path(filepath);
    std::string stem = path.stem().string();
    std::string extension;      // filename without extension
    if (!new_extension.empty()) {
      extension = new_extension;
    } else {
        extension =  path.extension().string();
    }
    std::filesystem::path new_path = path.parent_path() / (stem + suffix + extension);
    return new_path.string();
}


// Preserve the global name without creating a competing overload under ADL.
using graded_linalg::timed_with_progress;
