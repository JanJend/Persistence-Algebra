#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_quiver_rep(std::filesystem::path input_path, std::filesystem::path output_path, const int optional_value = 0) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());

    vec<r2degree> vertices;
    array<int> edges;
    if(optional_value == 0){
        vertices = presentation.discrete_support();
        edges = minimal_directed_graph<r2degree, int>(vertices);
    } else {
        vertices = presentation.get_equidistant_grid(optional_value);
        edges = equidistant_grid_edges<int>(optional_value, optional_value);
        // edges = minimal_directed_graph<r2degree, int>(vertices); could change to this
    }
    QuiverRepresentation<int, r2degree> rep = presentation.induced_quiver_rep(vertices, edges);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        rep.to_streamQPA(output_file);
        output_file.close();
        std::cout << "Quiver representation computed and saved to: " << output_path << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string filepath;
    int optional_value = 0;  // default value
    
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [optional_integer]" << std::endl;
        filepath = "/home/wsljan/MP-Workspace/Skyscraper-Invariant/example_files/indecomps/torus_dim2_2.scc";
    } else {
        filepath = argv[1];
    }
    
    if (argc == 3) {
        try {
            optional_value = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid integer argument" << std::endl;
            return 1;
        }
    } else {
        optional_value = 3;
    }

    std::string suffix;
    if (optional_value != 0){
        suffix = "_" + std::to_string(optional_value) + "x" + std::to_string(optional_value);
    } else {
        suffix = "_fullsupport";
    }

    std::filesystem::path input_path(filepath);
    std::string modified_path = insert_suffix_before_extension(filepath, suffix, ".g");
    std::filesystem::path output_path(modified_path);

    compute_quiver_rep(input_path, output_path, optional_value);

    return 0;
}