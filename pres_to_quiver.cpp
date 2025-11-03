#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_quiver_rep(std::filesystem::path input_path, std::filesystem::path output_path, const int optional_value = 0, const bool gap_output = false) {
    
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
        if(gap_output) {
            rep.to_streamQPA(output_file, "Q");
        } else {
            rep.to_stream_simple(output_file, "Q");
        }
        output_file.close();
        std::cout << "Quiver representation computed and saved to: " << output_path << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string filepath;
    int optional_value = 0;  // default value
    bool gap_output = false;
    std::filesystem::path output_path;
    std::string suffix;
    std::filesystem::path input_path;

    if (argc < 2 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [optional_grid_size] [bool_gap] [output_path] \n";
        std::cerr << "  optional_grid_size = 0 computes full support\n";
        std::cerr << "  bool_gap = 1 for gap output, 0 (default) for simple output." << std::endl;
        return 1;
    } else {
        filepath = argv[1];
        input_path = std::filesystem::path(filepath);
    }
    if (argc >= 3) {
        try {
            optional_value = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid integer argument" << std::endl;
            return 1;
        }
    }
    if (argc >= 4) {
        try {
            int output_option = std::stoi(argv[3]);
            if (output_option == 1) {
                gap_output = true;
            } else if (output_option != 0) {
                std::cerr << "Error: output_option must be 0 or 1" << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid output_option argument" << std::endl;
            return 1;
        }
    }
    
    if(argc >= 5) {
        try {
            output_path = std::filesystem::path(argv[4]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid output path argument" << std::endl;
            return 1;
        }
    } else {
        std::string suffix;
        std::string extension = gap_output ? ".g" : ".txt";
        if (optional_value != 0){
            suffix = "_" + std::to_string(optional_value) + "x" + std::to_string(optional_value);
        } else {
            suffix = "_fullsupport";
        }   
        std::string modified_path = insert_suffix_before_extension(filepath, suffix, extension);
        output_path = std::filesystem::path(modified_path);
    }
    
    compute_quiver_rep(input_path, output_path, optional_value, gap_output);

    return 0;
}