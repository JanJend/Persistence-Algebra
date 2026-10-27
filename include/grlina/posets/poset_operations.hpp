/** Hasse diagrams and CSV export for degree posets. */
#pragma once
#include <grlina/posets/traits.hpp>
#include <grlina/graphs.hpp>

namespace graded_linalg {

/**
 * @brief Computes the Hasse Diagram of a sorted list of degrees forming a Poset
 * 
 * @tparam D needs to suport functions "smaller" and "equals"
 * @param degrees needs to be sorted in any linear extension of the order
 * @return edgeList 
 */
template <typename D, typename index>
array<index> minimal_directed_graph(vec<D>& degrees, vec<index> support = vec<index>()) {
    static_assert(is_poset_degree_v<D>, "Degree_traits must provide the poset comparison interface");

    array<index> edges(degrees.size(), vec<index>());
    Degree_traits<D> D_traits;
    // writeDegreeListToCSV("degreeList.csv", vertices);
    
    // Add directed edges based on the relationships in the degreeList
    // This computes a transitive reduction of the full poset graph

    index batch_start = 0;
    bool in_batch = false;
    bool batch_ends = false;

    if(support.size() == 0) {
        // Don't check the last element
        for (index i = 0; i < degrees.size() - 1; i++) {

            D& d = degrees[i];
            // Skip batch of equal degrees
            if( D_traits.equals(d, degrees[i+1]) ) {
                if(!in_batch) {
                    in_batch = true;
                    batch_start = i;
                }
                edges[i].push_back(i+1);
                continue;
            } else if (in_batch) {
                batch_ends = true;
                in_batch = false;
                edges[i].push_back(batch_start);
            }

            for (index j = i + 1; j < degrees.size(); j++) {
                bool comesAfterSuccessor = false;
                D& nextd = degrees[j];
                if( D_traits.smaller(d, nextd) ){
                    // If a batch has ended, then the first element in edges is the first element of the batch
                    // and so has the same degree, always making comesAfterSuccessor true
                    for(index k = !batch_ends ? 0 : 1; k < edges[i].size(); k++){
                        if( D_traits.smaller(degrees[edges[i][k]], nextd) ) {
                            comesAfterSuccessor = true;
                            break;
                        }
                    }
                    if(!comesAfterSuccessor){
                        edges[i].push_back(j);
                    }
                }
            }

            batch_ends = false;
        }
    } else {
        // Don't check the last element
        for (index s = 0; s < support.size() - 1; s++) {

            D d = degrees[support[s]];

            // Skip batch of equal degrees
            if( D_traits.equals(d, degrees[support[s+1]]) ) {
                if(!in_batch) {
                    in_batch = true;
                    batch_start = s;
                }
                edges[s].push_back(s+1);
                continue;
            } else if (in_batch) {
                in_batch = false;
                edges[s].push_back(batch_start);
            }

            for (index t = s + 1; t < support.size(); t++) {
            bool comesAfterSuccessor = false;
            D& nextd = degrees[support[t]];
                if( D_traits.smaller(d, nextd) ){
                    // TO-DO: This is a bit inefficient, but it should be fine for non-very large lists
                    for(auto u : edges[s]){
                        if( D_traits.smaller(degrees[support[u]], nextd) ) {
                            comesAfterSuccessor = true;
                            break;
                        }
                    }
                    if(!comesAfterSuccessor){
                        edges[s].push_back(t);
                    }
                }
            }
        }
    }

    return edges;
}

template <typename D>
void writeDegreeListToCSV(const std::string& filename, const vec<D>& degrees) {
    static_assert(is_degree_v<D>, "D must implement the Degree_traits interface");

    Degree_traits<D> D_traits;
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(15);

    int dim = position(degrees[0]).size();
    for (size_t i = 0; i < dim; ++i) {
        outfile << "Column_" << i << ",";
    }
    outfile << std::endl;

    for (const auto& element : degrees) {
        const vec<std::string>& pos = D_traits.position(element);
        for (const std::string& coord : pos) {
            outfile << coord << ",";
        }
        outfile << std::endl;
    }

    std::cout << "Degree list written to: " << filename << std::endl;
}

} // namespace graded_linalg
