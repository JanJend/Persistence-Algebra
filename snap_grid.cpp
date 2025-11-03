#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void snap_presentation(std::filesystem::path input_path, std::filesystem::path output_path, const int grid_size = 5) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());

   presentation.snap_to_equidistant_grid(grid_size);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        presentation.to_stream(output_file);
        output_file.close();
        std::cout << "Restricted module to " << grid_size << "x" << grid_size << " grid and saved to: " << output_path << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string filepath;
    int grid_size = 5;  // default value
    std::filesystem::path output_path;
    std::string suffix;
    std::filesystem::path input_path;

    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [grid_size] [output_path] \n";
        std::cerr << "  output_path = optional output file path." << std::endl;
    } else {
        filepath = argv[1];
        input_path = std::filesystem::path(filepath);
        try {
            grid_size = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid integer argument" << std::endl;
            return 1;
        }
    }
    if(argc >= 4) {
        try {
            output_path = std::filesystem::path(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid output path argument" << std::endl;
            return 1;
        }
    } else {
        std::string suffix = "_" + std::to_string(grid_size) + "x" + std::to_string(grid_size) + "_snapped";
        std::string modified_path = insert_suffix_before_extension(filepath, suffix);
        output_path = std::filesystem::path(modified_path);
    }
    input_path = "/home/wsljan/MP-Workspace/Skyscraper-Invariant/example_files/indecomps/torus_dim4_1.scc";
    snap_presentation(input_path, output_path, grid_size);
    return 0;
}