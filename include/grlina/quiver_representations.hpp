// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright 2025 TU Graz. Author: Jan Jendrysiak.
#pragma once
#include <grlina/general_checks.hpp>

#include <grlina/posets/poset_operations.hpp>
#include <grlina/matrices/sparse.hpp>
#include <grlina/matrices/matrix_algorithms.hpp>

namespace graded_linalg {

template <typename index>
array<index> equidistant_grid_edges(int m, index n) {
    array<index> result(m * n);
    
    for(index i = 0; i < m; i++) {
        for(index j = 0; j < n; j++) {
            index idx = i * n + j;
            
            // Right neighbor (j+1)
            if(j + 1 < n) {
                result[idx].push_back(i * n + (j + 1));
            }
            
            // Top neighbor (i+1)
            if(i + 1 < m) {
                result[idx].push_back((i + 1) * n + j);
            }
        }
    }
    
    return result;
}

template <typename index, typename D, typename Matrix = SparseMatrix<index>>
struct QuiverRepresentation {
    static_assert(is_degree_v<D>, "D must implement the Degree_traits interface");
	vec<D> degrees;
    vec<index> dimensionVector;
	edge_list<index> edges;
	vec<Matrix> matrices;
    
    QuiverRepresentation(){
        degrees = vec<D>();
        dimensionVector = vec<index>();
        edges = edge_list<index>();
        matrices = vec<Matrix>();
    }

    void print(){
        std::cout << "Degrees: ";
        for(auto d : degrees){
            Degree_traits<D>::print_degree(d);
            std::cout << " ";
        }
        std::cout << std::endl;
        print_edge_list(edges);
        index size = degrees.size();
        GRLINA_ASSERT(size == dimensionVector.size());
        index length = matrices.size();
        GRLINA_ASSERT(length == edges.size());
        for(index i = 0; i < length; i++){
            std::cout << "Matrix at edge " << i << std::endl;
            matrices[i].print();
        }
    }

    QuiverRepresentation(vec<D> degrees, const vec<index>& dimensionVector, const edge_list<index>& edges, const vec<Matrix>& matrices) :
    degrees(degrees), dimensionVector(dimensionVector), edges(edges), matrices(matrices) {}

    template <typename OutStream>
    void to_stream_simple(OutStream& outfile, const std::string& quiverName = "Q") const {
        index num_vert = degrees.size();
        index num_edges = edges.size();     
        // Define the quiver
        
        outfile << quiverName << " := Quiver(" << num_vert << ", [ ";
        // Write edges
        for (index j = 0; j < num_edges; j++) {
            index source = edges[j].first;
            index target = edges[j].second;
            // Write edge
            outfile << "["  << source + 1 << "," << target + 1 << ", \"E" << j << "\"" << "], ";
        }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << "]);" << std::endl;

        // Define module
        // syntax: M, [1,2,2, ..], [ .. ["b", [[1,0], [1,1]] ], ..] )
        outfile << quiverName << ", [";
        // dimension vector
        for (index i = 0; i < num_vert; i++) {
            outfile << dimensionVector[i] << ",";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "], [ ";
        // matrices at edges
        bool remove_comma = false;
        for (index j = 0; j < num_edges; j++) {
            auto source = edges[j].first;
            auto target = edges[j].second;
            
            const Matrix* current_mat_ptr = &matrices[j];
            // We write our matrices column wise. If you want the row format, uncomment the next lines.
            // SparseMatrix transposed = matrices[j].transposed_copy();
            // current_mat_ptr = &transposed;
            const auto& current_mat = *current_mat_ptr;

            GRLINA_DEBUG_CHECK(
            if (current_mat.get_num_cols() != dimensionVector[source]) {
                throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
            }
            if (current_mat.get_num_rows() != dimensionVector[target]) {
                throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
            }
            );

            if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
                continue;
            } else {
                remove_comma = true;
            }
            // Write matrix at arrow; where each vector is a row

            
            outfile << "[\"E" << j << "\", [";
            for (index i = 0; i < current_mat.get_num_cols(); i++) {
                outfile << "[";
                auto it = current_mat.data[i].begin();

                for (index k = 0; k < current_mat.get_num_rows(); k++) {
                    if (it != current_mat.data[i].end()) {
                        if(*it == k){
                            outfile << "1" << ",";
                            ++it;
                        } else {
                            outfile << "0" << ",";
                        }
                    } else {
                        outfile << "0" << ",";
                    }
                }
                outfile.seekp(-1, std::ios::end);  // Remove trailing comma
                outfile << "],";
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "] ], ";
        }
        if(remove_comma){
            outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        }
        outfile << " ];" << std::endl;
    }

    template <typename OutStream>
    void to_streamQPA(OutStream& outfile, const std::string& quiverName = "Q") {
        
        index dimension = 0;
        for(index i = 0; i < dimensionVector.size(); i++){
            dimension += dimensionVector[i];
        }

        index num_vert = degrees.size();
        index num_edges = edges.size();

        // Write header information -  May need to be changed
        // outfile << "LoadPackage(\"qpa\");\n\n";

        // Define the quiver
        outfile << "# Dimension of Module: " << dimension << "\n";
        outfile << quiverName << " := Quiver(" << num_vert << ", [ ";
        

        // Write arrows (edges)
        for (index j = 0; j < num_edges; j++) {
            index source = edges[j].first;
            index target = edges[j].second;

            // Write arrow
            outfile << "["  << source + 1 << "," << target + 1 << ", \"E" << j << "\"" << "], ";

            
            }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << "]);" << std::endl;

        // Define path algebra
        outfile << "A" << quiverName << " := PathAlgebra(GF(2), " << quiverName<< ");" << std::endl;

        // Define module
        // syntax: N := RightModuleOverPathAlgebra( A, [1,2,2], [ .. ["b", [[1,0], [-1,0]] ], ..] )
        // Use Z(2)^0 and 0*Z(2) for 1 and 0; (Need Z(p^k) for general finite field)

        outfile << "M" << quiverName << " := RightModuleOverPathAlgebra(A" << quiverName << ", [";
        for (index i = 0; i < num_vert; i++) {
            outfile << dimensionVector[i] << ",";
        }
        outfile.seekp(-1, std::ios::end);  // Remove trailing comma
        outfile << "], [ ";

        for (index j = 0; j < num_edges; j++) {
            auto source = edges[j].first;
            auto target = edges[j].second;
            // No need for this, qpa wants the transposed matrices in row format, so we just write our matrices column wise instead of row wise
            // No need for: SparseMatrix edgeMatrix = matrices[j].transposed_copy();

            GRLINA_DEBUG_CHECK(
            if (matrices[j].get_num_cols() != dimensionVector[source]) {
                throw std::runtime_error("Dimension mismatch for source vertex " + std::to_string(source));
            }
            if (matrices[j].get_num_rows() != dimensionVector[target]) {
                throw std::runtime_error("Dimension mismatch for target vertex " + std::to_string(target));
            }
            );

            if(dimensionVector[source] == 0 || dimensionVector[target] == 0){
                continue;
            }
            // Write matrix at arrow; where each vector is a row

            
            outfile << "[\"E" << j << "\", [";
            for (index i = 0; i < matrices[j].get_num_cols(); i++) {
                outfile << "[";
                auto it = matrices[j].data[i].begin();

                for (index k = 0; k < matrices[j].get_num_rows(); k++) {
                    if (it != matrices[j].data[i].end()) {
                        if(*it == k){
                            outfile << "Z(2)^0" << ",";
                            ++it;
                        } else {
                            outfile << "0*Z(2)" << ",";
                        }
                    } else {
                        outfile << "0*Z(2)" << ",";
                    }
                }
                outfile.seekp(-1, std::ios::end);  // Remove trailing comma
                outfile << "],";
            }
            outfile.seekp(-1, std::ios::end);  // Remove trailing comma
            outfile << "] ], ";
        }

        outfile.seekp(-2, std::ios::end);  // Remove trailing comma
        outfile << " ]);" << std::endl;
    }

    void write_QPA_decomp(const std::string& fullFilePath, const std::string& quiverName = "Q") {
        
        std::string scriptPath = fullFilePath + ".g";
        std::string logFilePath = fullFilePath + "_log.txt";

        std::ofstream outfile(scriptPath);
        if (!outfile.is_open()) {
            std::cerr << "Error opening file at path: " << scriptPath << std::endl;

        }
        outfile << "LoadPackage(\"qpa\");\n";
        outfile << "start_read := Runtime();  # Record the starting time\n";

        to_streamQPA(outfile, quiverName);
        
        outfile << "List" << quiverName << " := BlockDecompositionOfModule(M" << quiverName << ");" << std::endl;
        outfile << "end_read := Runtime(); \n";
        outfile << "time_taken := end_read - start_read; \n";
        outfile << "PrintTo( \"" << logFilePath << " \" , \"CPU time taken to read file and compute decomposition: \", time_taken, \" milliseconds\", List" << quiverName << "); \n";
        outfile << "QUIT;";
        std::cout << "Quiver file written to: " << fullFilePath << std::endl;

    }
};


// Conversion from graded presentations to their induced quiver representations.

template<typename D, typename index, typename DERIVED, typename Storage>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
template <typename T>
auto GradedSparseMatrix<D, index, DERIVED, Storage>::extractElements(const vec<T>& target, const vec<index>& indices) -> vec<T> {
    vec<T> result;
    result.reserve(indices.size());
    for (index i : indices) {
        GRLINA_ASSERT(i >= 0 && i < target.size() && "Index out of bounds");
        result.push_back(target[i]);
    }
    return result;
}

template<typename D, typename index, typename DERIVED, typename Storage>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
auto GradedSparseMatrix<D, index, DERIVED, Storage>::getIndicatorVector(vec<index> target, vec<index> subset) -> vec<index> {
    vec<index> result;
    auto itS = subset.begin();
    for(index i = 0; i < target.size() ; i++){
        if(itS != subset.end()){
        if(*itS == target[i]){
        result.push_back(i);
        itS++;
        }
        }
    }
    GRLINA_ASSERT(itS == subset.end() && "Not all elements of the subset were found in the target");
    return result;
}

template<typename D, typename index, typename DERIVED, typename Storage>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
auto GradedSparseMatrix<D, index, DERIVED, Storage>::basisLifting(std::pair<vec<index>, vec<index>>& source, std::pair<vec<index>, vec<index>>& target) -> vec<index> {
    // Probably this doesnt need to be its own function
    vec<index> result = GradedSparseMatrix::getIndicatorVector(target.first, GradedSparseMatrix::extractElements(source.first, source.second));
    return result;
}

template<typename D, typename index, typename DERIVED, typename Storage>
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    requires Degree<D>
#endif
auto GradedSparseMatrix<D, index, DERIVED, Storage>::induced_quiver_rep(vec<D> vertices, array<index> edges) -> QuiverRepresentation<index, D, SparseMatrix<index, Storage>> {
            if (vertices.empty() && edges.empty()) vertices = discrete_support();
            if (edges.empty() && !vertices.empty()) {
                edges = minimal_directed_graph<D, index>(vertices);
            }
            GRLINA_DEBUG_CHECK(if (vertices.size() != edges.size())
                throw std::invalid_argument("Quiver adjacency must match its vertices");
            for (index i = 0; i < static_cast<index>(vertices.size()); ++i)
                for (index j : edges[i])
                    if (j < 0 || j >= static_cast<index>(vertices.size()) ||
                        !Degree_traits<D>::smaller_equal(vertices[i], vertices[j]))
                        throw std::invalid_argument("Quiver edge must follow the degree order"));

            QuiverRepresentation<index, D, SparseMatrix<index, Storage>> rep;
            rep.degrees = vertices;
            for(index i = 0; i < rep.degrees.size(); i++) {
                for(index j : edges[i]){
                    rep.edges.push_back(std::make_pair(i, j));
                }
            }

            index num_vert = rep.degrees.size();
            index num_edges = rep.edges.size();

            // For each degree we want to store the cokernel,
            // the row-indices of the generators which form its domain
            // and a section of the cokernel given by column indices which are mapped to a basis
            vec< SparseMatrix<index, Storage> > pointwise_Presentations;
            vec< std::pair< vec<index> , vec<index>> > pointwise_base;
            pointwise_Presentations.reserve(num_vert);
            pointwise_base.reserve(num_vert);
            rep.dimensionVector.reserve(num_vert);

            // #pragma omp parallel for
            for (index i = 0; i < num_vert; i++) {

                SparseMatrix<index, Storage> S;
                vec<index> gens;
                std::tie(S, gens) = this->map_at_degree_pair(rep.degrees[i]);
                S.column_reduction();
                vec<index> basisLift;
                pointwise_Presentations.emplace_back(S.coKernel(true, &basisLift));
                pointwise_base.emplace_back(std::make_pair(gens, basisLift));
                rep.dimensionVector.emplace_back(basisLift.size());
            }

            rep.matrices.reserve(num_edges);
            for (index i = 0; i < num_edges; i++) {
                index source = rep.edges[i].first;
                index target = rep.edges[i].second;

                auto sourceBasis = pointwise_base[source];
                auto targetBasis = pointwise_base[target];

                GRLINA_ASSERT(pointwise_Presentations[source].get_num_cols() == sourceBasis.first.size());
                GRLINA_ASSERT(pointwise_Presentations[target].get_num_cols() == targetBasis.first.size());

                vec<index> lift_of_basis = GradedSparseMatrix::basisLifting(sourceBasis, targetBasis);
                rep.matrices.emplace_back( pointwise_Presentations[target].restricted_domain_copy(lift_of_basis) );
            }

            GRLINA_DEBUG_CHECK(for (index j = 0; j < num_edges; j++) {
                if(rep.matrices[j].get_num_cols() != rep.dimensionVector[rep.edges[j].first] || rep.matrices[j].get_num_rows() != rep.dimensionVector[rep.edges[j].second]){
                    throw std::runtime_error("Dimension mismatch in path action");
                }
            });
            return rep;
}

template<typename Scalar, std::size_t Dimension, typename Index, typename Storage>
auto GridGradedSparseMatrix<Scalar, Dimension, Index, Storage>::induced_quiver_rep(vec<GridD> vertices, array<Index> edges) {
    return to_coordinates().induced_quiver_rep(std::move(vertices), std::move(edges));
}

} // namespace graded_linalg
