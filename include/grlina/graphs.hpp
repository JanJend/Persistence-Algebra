/**
 * @file graphs.hpp
 * @author Jan Jendrysiak
 * @brief Graph construction, traversal, condensation and export.
 * @version 2.0.0
 * @date 2025-03-13
 * 
 * @copyright 2025 TU Graz
    This file is part of the AIDA library. 
   You can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 */

#pragma once

#ifndef GRLINA_GENERAL_GRAPHS_HPP
#define GRLINA_GENERAL_GRAPHS_HPP

#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <iomanip>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/topological_sort.hpp>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <map>
#include <grlina/general.hpp>
#include <stack>

namespace graded_linalg {

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;

template <typename index>
using edge_list = vec<std::pair<index,index>>;

template <typename index>
void print_edge_list(const edge_list<index>& edges){
    for(std::pair<index,index> e : edges){
        std::cout << e.first << " -> " << e.second << std::endl;
    }
}

/**
 * @brief Creates an induced subgraph on a subset of the vertices.
 * 
 * @tparam index 
 * @param g 
 * @param vertices 
 * @return Graph 
 */
template <typename index>
Graph induced_subgraph (const Graph& g, const vec<index>& vertices) {
    Graph subgraph;
    std::unordered_map<index, Vertex> vertex_map;

    // Add vertices to the subgraph and create a mapping
    for (auto v : vertices) {
        Vertex new_v = boost::add_vertex(subgraph);
        vertex_map[v] = new_v;
    }

    // Add edges to the subgraph
    for (auto v : vertices) {
        for (auto it = boost::out_edges(v, g); it.first != it.second; ++it.first) {
            Vertex u = boost::target(*it.first, g);
            if ( std::binary_search(vertices.begin(), vertices.end(), u) ) {
                boost::add_edge(vertex_map[v], vertex_map[u], subgraph);
            }
        }
    }

    return subgraph;
}

inline void delete_incoming_edges(Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    std::tie(in_begin, in_end) = boost::in_edges(v, g);
    for (auto it = in_begin; it != in_end; ++it) {
        boost::remove_edge(*it, g);
    }
}



// Function to print the graph with labels
template <typename T>
void print_graph_with_labels(const Graph& g, const std::vector<T>& labels) {

    for (Vertex v = 0; v < boost::num_vertices(g); ++v) {
        std::cout << labels[v] << " -> ";
        out_edge_iterator out_i, out_end;
        bool first = true;
        for (boost::tie(out_i, out_end) = boost::out_edges(v, g); out_i != out_end; ++out_i) {
            Vertex u = boost::target(*out_i, g);
            if (!first) {
                std::cout << ", ";
            }
            std::cout << labels[u];
            first = false;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief prints the content of a graph
 * 
 */
inline void print_graph(const Graph& g){
    boost::print_graph(g); //For now
}


/**
 * @brief Checks if a vertex has incoming edges.
 * 
 * @param g The graph.
 * @param v The vertex.
 * @return true if the vertex has incoming edges, false otherwise.
 */
inline bool has_incoming_edges(const Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    // Get the range of incoming edges for vertex v
    std::tie(in_begin, in_end) = boost::in_edges(v, g);

    // Check if the range is empty
    return in_begin != in_end;
}

template <typename index>
std::vector<index> incoming_edges(const Graph& g, const Vertex& v) {
    in_edge_iterator in_begin, in_end;
    std::tie(in_begin, in_end) = boost::in_edges(v, g);
    std::vector<index> sources;
    for (auto it = in_begin; it != in_end; ++it) {
        sources.push_back(static_cast<index>(boost::source(*it, g)));
    }
    return sources;
}

/**
 * @brief Constructs a directed boost graph from a list of labels and a function EdgeChecker which returns true if there is a directed edge between two labels.
 * 
 * @tparam index 
 * @tparam EdgeChecker 
 * @param labels 
 * @param has_edge 
 * @param visualise 
 * @return Graph 
 */
template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const vec<index>& labels, EdgeChecker has_edge, bool visualise = false) {
    Graph g;

    for (index i = 0; i < labels.size(); i++) {
        boost::add_vertex(g);
    }
    for (index i = 0; i < labels.size(); i++) {
        for (index j = 0; j < labels.size(); j++) {
            if (has_edge(labels[i], labels[j]) && labels[i] != labels[j]) {
                boost::add_edge(i, j, g);
            }
        }
    }

    if(visualise) {
        print_graph(g);
    }

    return g;
}

template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const index& num_vertices, EdgeChecker has_edge){
    Graph g;
    for (index i = 0; i < num_vertices; i++) {
        boost::add_vertex(g);
    }
    for (index i = 0; i < num_vertices; i++) {
        for (index j = 0; j < num_vertices; j++) {
            if (has_edge(i, j) && i != j) {
                boost::add_edge(i, j, g);
            }
        }
    }
    return g;
}

/**
 * @brief Constructs a directed boost graph from a list of labels and a function EdgeChecker which returns true if there is a directed edge between two labels.
 * 
 * @tparam index 
 * @tparam EdgeChecker 
 * @param labels 
 * @param has_edge 
 * @param visualise 
 * @return Graph 
 */
template <typename index, typename EdgeChecker>
Graph construct_boost_graph(const std::set<index>& labels, EdgeChecker has_edge, bool visualise = false) {
    Graph g;

    std::map<index, Graph::vertex_descriptor> label_to_vertex;
    
    for (auto label : labels) {
        Vertex v = boost::add_vertex(label, g);
        label_to_vertex[label] = v;
    }


    for (auto label1 : labels) {
        for (auto label2 : labels) {
            if (has_edge(label1, label2) && label1 != label2) {
                boost::add_edge(label_to_vertex[label1], label_to_vertex[label2], g);
            }
        }
    }

    if(visualise) {
        print_graph(g);
    }

    return g;
}



/**
 * @brief Given a graph G computes its condenstion.
 * 
 * @param g 
 * @param component will store the index of the component a vertex belongs to
 * @return std::pair<std::vector<std::set<int>>, Graph> returns a vector containing the components as sets and the condensation graph
 */
template <typename index>
Graph compute_scc_and_condensation(const Graph& g, std::vector<index>& component, std::vector<std::vector<index>>& scc_sets) {
    int num_vertices = boost::num_vertices(g);

    // Vector to store the component index of each vertex
    component.assign(num_vertices, -1);
    int num_components = boost::strong_components(g, boost::make_iterator_property_map(component.begin(), boost::get(boost::vertex_index, g)));

    // Create a vector of sets to represent SCCs
    scc_sets = vec<vec<index>>(num_components);
    for (int i = 0; i < num_vertices; ++i) {
        scc_sets[component[i]].push_back(i);
    }

    // Construct the condensation graph - This is super inefficient, in fact 
    // the condensation is implicitly computed by tarjans algorithm which is used by boost

    Graph condensation(num_components);

    for (int u = 0; u < num_vertices; ++u) {
        for (auto it = boost::out_edges(u, g); it.first != it.second; ++it.first) {
            int v = boost::target(*it.first, g);
            if (component[u] != component[v] && !boost::edge(component[u], component[v], condensation).second) {
                boost::add_edge(component[u], component[v], condensation);
            }
        }
    }
    return condensation;
}

/**
 * @brief Returns a topological order for the input directed graph.
 * 
 * @tparam index 
 * @param g 
 * @return std::vector<index> 
 */
template <typename index>
std::vector<index> compute_topological_order(const Graph& g) {
    std::vector<index> topological_order;
    boost::topological_sort(g, std::back_inserter(topological_order));
    return topological_order;
}

template <typename index>
Graph boost_graph_from_edge_list(const array<index>& edges) {
    Graph g;
    for (index i = 0; i < edges.size(); i++) {
        boost::add_vertex(g);
    }
    for (index i = 0; i < edges.size(); i++) {
        for (index j : edges[i]) {
            boost::add_edge(i, j, g);
        }
    }
    return g;
}


/**
 * @brief Writes a list of edges to a CSV file
 * 
 * @tparam D 
 * @param filename 
 * @param degrees 
 */
template <typename index>
void writeEdgeListToCSV(const std::string& filename, const array<index>& edges) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write header if needed
    outfile << "Source: Targets" << std::endl;

    // Write integer pairs to the file
    for (index source = 0; source < edges.size(); source++) {
        outfile << source << ": " << std::endl;
        for(auto target : edges[source]){
            outfile << target << ", " << std::endl;
        }
    }

    std::cout << "Integer pairs written to: " << filename << std::endl;
}

/**
 * @brief Print the edges of a directed graph.
 * 
 * @tparam index 
 * @param edges 
 */
template <typename index>
void print_edge_list(const array<index>& edges){
    for (index source = 0; source < edges.size(); source++) {
        std::cout << source << " -> ";
        for(auto target : edges[source]){
            std::cout << target << ", ";
        }
        std::cout << std::endl;
    }
}



// TO-DO: This is AI generated, make sure it works. Maybe change to Kosaraju's algorithm.
// Function to perform DFS and find SCCs using Tarjan's algorithm
template <typename index>
void tarjanDFS(index u, const std::unordered_map<index, vec<index>>& outgoing_edges, vec<index>& disc, vec<index>& low, std::stack<index>& st, vec<bool>& inStack, array<index>& scc, index& time) {
    disc[u] = low[u] = ++time;
    st.push(u);
    inStack[u] = true;

    if (auto outgoing = outgoing_edges.find(u); outgoing != outgoing_edges.end()) {
        for (int v : outgoing->second) {
            if (disc[v] == -1) {
                tarjanDFS(v, outgoing_edges, disc, low, st, inStack, scc, time);
                low[u] = std::min(low[u], low[v]);
            } else if (inStack[v]) {
                low[u] = std::min(low[u], disc[v]);
            }
        }
    }

    if (low[u] == disc[u]) {
        vec<index> component;
        while (st.top() != u) {
            int v = st.top();
            st.pop();
            inStack[v] = false;
            component.push_back(v);
        }
        st.pop();
        inStack[u] = false;
        component.push_back(u);
        scc.push_back(component);
    }
}

template <typename index>
array<index> condensation(const set<index>& vertex_labels, const std::unordered_map<index, vec<index>>& outgoing_edges, array<index>& scc) {
    int n = vertex_labels.size();
    vec<int> disc(n, -1), low(n, -1);
    vec<bool> inStack(n, false);
    std::stack<int> st;
    int time = 0;

    for (int u : vertex_labels) {
        if (disc[u] == -1) {
            tarjanDFS(u, outgoing_edges, disc, low, st, inStack, scc, time);
        }
    }

    array<index> dag(scc.size());

    for (int u : vertex_labels) {
        if (auto outgoing = outgoing_edges.find(u); outgoing != outgoing_edges.end()) {
            for (int v : outgoing->second) {
                int scc_u = -1, scc_v = -1;
                for (int i = 0; i < scc.size(); ++i) {
                    if (find(scc[i].begin(), scc[i].end(), u) != scc[i].end()) scc_u = i;
                    if (find(scc[i].begin(), scc[i].end(), v) != scc[i].end()) scc_v = i;
                }
                if (scc_u != -1 && scc_v != -1 && scc_u != scc_v) {
                    dag[scc_v].push_back(scc_u);  // Store predecessors
                }
            }
        }
    }

    return dag;
}


/**
 * @brief Copies an array into an analogous hash map.
 * 
 * @tparam index 
 * @param adj_matrix 
 * @return std::unordered_map<index, vec<index>> 
 */
template <typename index>
std::unordered_map<index, vec<index>> convert_to_map(const array<index>& arr) {
    std::unordered_map<index, vec<index>> map;
    for (index i = 0; i < arr.size(); ++i) {
        map[i] = arr[i];
    }
    return map;
}


} // namespace graded_linalg

#endif // GRLINA_GENERAL_GRAPHS_HPP
