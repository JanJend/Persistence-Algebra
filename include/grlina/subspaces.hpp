// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright 2025 TU Graz. Author: Jan Jendrysiak.
/** @file subspaces.hpp @brief F2 subspace enumeration, decomposition data and serialization. */
#pragma once

#include <grlina/matrices/conversions.hpp>
#include <grlina/matrices/dense.hpp>
#include <grlina/storage/bitset_algebra.hpp>

namespace graded_linalg {

inline std::vector<DenseMatrix > pivotsToEchelon(const boost::dynamic_bitset<> &pivots, std::vector<std::pair<int,int>> &positions ){
    std::vector<DenseMatrix> reducedMatrices;

    size_t n = pivots.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        reducedMatrices.emplace_back( DenseMatrix(pivots, positions, i) );
    }
    return reducedMatrices;
}

inline void pivotsToEchelon(std::vector<DenseMatrix>& result, const boost::dynamic_bitset<> &pivots, std::vector<std::pair<int,int>> &positions ){

    size_t n = pivots.size();
    size_t mul = positions.size();
    size_t subsetCount = static_cast<size_t>(std::pow(2, mul));

    for (size_t i = 0; i < subsetCount; ++i) {
        result.emplace_back( DenseMatrix(pivots, positions, i) );
    }
}




// For a given bitset returns all non-set positions in a col-echelon matrix whose pivots are given by the input.
inline std::vector<std::pair<int, int>> getEchelonPositions(const boost::dynamic_bitset<> &bitset) {
    size_t countOnes = 0;
    std::vector<std::pair<int, int>> positions;

    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i) {
        if (!bitset.test(i)) {
            for (int j = 0; j < countOnes; j++){
                positions.push_back(std::make_pair(j, i));
            }
        } else {
            countOnes++;
        }
    }
    return positions;
}

inline std::vector<DenseMatrix> all_proper_subspaces(int k){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 1; i < (1ULL << k) - 1; ++i) {
        pivots.emplace_back(bitset(k, i));
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> all_subspaces(int k){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        pivots.emplace_back(bitset(k, i));
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> grassmannian(int k, int n){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        bitset b(k, i);
        if(b.count() == n){
            pivots.emplace_back(b);
        }
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

inline std::vector<DenseMatrix> grassmannian_union(int k, int n){
    vec<bitset> pivots;
    vec<DenseMatrix> subspaces;
    for (int i = 0; i < (1ULL << k); ++i) {
        bitset b(k, i);
        if(b.count() <= n){
            pivots.emplace_back(b);
        }
    }
    for( bitset& b : pivots){
        auto positions = getEchelonPositions(b);
        pivotsToEchelon(subspaces, b, positions);
    }
    return subspaces;
}

template <typename index>
vec<vec<SparseMatrix<index>>> all_sparse_proper_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){

        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}

template <typename index>
vec<vec<SparseMatrix<index>>> all_sparse_subspaces(index k){
    vec<vec<SparseMatrix<index>>> result = vec<vec<SparseMatrix<index>>>(k);
    for(index i = 0; i < k; i++){

        vec<DenseMatrix> i_spaces = all_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            result[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
    return result;
}


template <typename index>
void fill_up_subspaces (vec<vec<SparseMatrix<index>>>& subspaces, index k ){

    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<SparseMatrix<index>>());
        vec<DenseMatrix> i_spaces = all_proper_subspaces(i+1);
        for(DenseMatrix matrix : i_spaces){
            subspaces[i].emplace_back(sparse_from_dense<index>(matrix));
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> sparse_seperated_grassmannians(index k){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k, vec<vec<SparseMatrix<index>>>());
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_seperated_grassmannians(vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

template <typename index>
vec<vec<vec<SparseMatrix<index>>>> all_sparse_grassmannians(index k, index n){
    vec<vec<vec<SparseMatrix<index>>>> result = vec<vec<vec<SparseMatrix<index>>>>(k);
    for(index i = 0; i < k; i++){
        for(index j = 0; j <= i+1; j++){
            result[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                result[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
    return result;
}

template <typename index>
void fill_up_grassmannians (vec<vec<vec<SparseMatrix<index>>>>& subspaces, index k, index n){
    for(index i = subspaces.size(); i < k; i++){
        subspaces.push_back(vec<vec<SparseMatrix<index>>>());
        for(index j = 0; j <= i+1; j++){
            subspaces[i].emplace_back(vec<SparseMatrix<index>>());
            if(j > n){
                continue;
            }
            vec<DenseMatrix> Gr_i_j = grassmannian(i+1, j);
            for(DenseMatrix matrix : Gr_i_j){
                subspaces[i][j].emplace_back(sparse_from_dense<index>(matrix));
            }
        }
    }
}

// A pair of DenseMatrices - the two subspaces of a decomposition.
using VecDecomp = std::pair<DenseMatrix, DenseMatrix>;

// Associates to each integer the subspace whos Plücker coordinates have the integer as a binary representation.
// Then to each subspace there can be a list of complements.
using DecompBranch = std::vector< std::vector<VecDecomp> >;

// associates to each bitset the subspaces whose reduced form has the entries of the bitset as pivots.
using DecompTree = std::unordered_map<boost::dynamic_bitset<>, DecompBranch , BitsetHash>;

// A transition is an invertible matrix with 1s on the diagonal and a subset of the columns given as a bitset.
// The matrix stores the necessary column-operations to transform a decomposition into another one.
// The nonzero entries of the bitset give the column-indices associated to the first subspace,
// the zero entries to the second subspace.
using transition = std::pair<DenseMatrix, bitset>;

/**
 * @brief Writes the DecompTree to a file.
 *
 * @param tree
 * @param filename
 */
inline void saveDecompTree(const DecompTree& tree, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "File opened successfully for writing: " << filename << std::endl;
    }


    size_t treeSize = tree.size();
    file.write(reinterpret_cast<const char*>(&treeSize), sizeof(treeSize));

    for ( auto& [key, branch] : tree) {

        serializeDynamicBitset(key, file);
        // key gives pivots

        size_t branchSize = branch.size();
        file.write(reinterpret_cast<const char*>(&branchSize), sizeof(branchSize));

        for (size_t p_coord = 0; p_coord< branch.size(); p_coord++) {

            // p_coord is the integer representation of the Plücker coordinate

            size_t vectorSize = branch[p_coord].size();
            file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

            for (const VecDecomp& vecs : branch[p_coord]) {
                // Careful, we are serializing the second file first, because the reader will reverse this again!
                vecs.second.serialize(file);

                vecs.first.serialize(file);
            }
        }
    }

    file.close();
}

/**
 * @brief Loads a DecompTree from a file.
 *
 * @param filename
 * @return DecompTree
 */
inline DecompTree loadDecompTree(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file '" + filename + "' for reading.");
    } else {
        std::cout << "File opened successfully for reading: " << filename << std::endl;
    }

    size_t treeSize;
    file.read(reinterpret_cast<char*>(&treeSize), sizeof(treeSize));

    DecompTree tree;
    for (size_t i = 0; i < treeSize; ++i) {
        boost::dynamic_bitset<> key = deserializeDynamicBitset(file);

        size_t branchSize;
        file.read(reinterpret_cast<char*>(&branchSize), sizeof(branchSize));

        DecompBranch branch;
        for (size_t j = 0; j < branchSize; ++j) {

            size_t vectorSize;
            file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));

            std::vector<VecDecomp> VecDecomps;
            VecDecomps.reserve(vectorSize);
            for (size_t k = 0; k < vectorSize; ++k) {
                //This reverses the order of the pair. Works because the writer has already reversed it, too.
                VecDecomps.emplace_back(DenseMatrix(file), DenseMatrix(file));
            }

            branch.emplace_back(std::move(VecDecomps));
        }

        tree.emplace(std::move(key), std::move(branch));
    }

    file.close();
    return tree;
}

/**
 * @brief Prints the whole content of the DecompTree to the console.
 *
 * @param tree
 */
inline void print_tree(DecompTree& tree){
    int num_bits = 1;
    std::cout << "Printing tree with the following branches: " << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << pivots_ << ", ";
    }
    std::cout << std::endl;
    for(auto& [pivots_, branch] : tree){
        std::cout << "Pivots: ";
        print_bitset(pivots_);
        for(int i = 0; i < branch.size(); i++){
            num_bits = static_cast<int>(std::log2(branch.size()));
            bitset binary(num_bits, i);
            std::cout << "Pluecker Coordinate as integer is " << i << ". And as binary is ";
            print_bitset(binary);
            for(auto& [first, second] : branch[i]){
                std::cout << "First: " << std::endl;
                first.print(true);
                std::cout << "Second: " << std::endl;
                second.print(true);
            }
        }
    }
}

/**
 * @brief Saves a transition list to a file.
 *
 * @param transitions
 * @param filename
 */
inline void save_transition_list(const std::vector<transition>& transitions, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    } else {
        std::cout << "Writing transitions to: " << filename << std::endl;
    }

    // Write the size of the vector
    size_t vectorSize = transitions.size();
    file.write(reinterpret_cast<const char*>(&vectorSize), sizeof(vectorSize));

    for (const auto& [matrix, bitset] : transitions) {
        matrix.serialize(file);
        serializeDynamicBitset(bitset, file);
    }

    file.close();
}

/**
 * @brief Loads a transition list from a file.
 *
 * @param filename
 * @return std::vector<transition>
 */
inline std::vector<transition> load_transition_list(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename + " for reading.");
    } else {

    }

    size_t vectorSize;
    file.read(reinterpret_cast<char*>(&vectorSize), sizeof(vectorSize));
    // std::cout << "loading #matrices: " << vectorSize << std::endl;
    std::vector<transition> transitions;
    transitions.reserve(vectorSize);
    for (size_t i = 0; i < vectorSize; ++i) {
        auto T = DenseMatrix(file);
        auto bs = deserializeDynamicBitset(file);
        transitions.emplace_back(std::move(T), std::move(bs));
    }

    file.close();
    return transitions;
}

} // namespace graded_linalg
