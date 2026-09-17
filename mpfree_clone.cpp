#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;

void get_minimal_presentation(std::filesystem::path input_path, std::filesystem::path output_path) {
    
    ChainComplex<R2GradedSparseMatrix<int>> complex(input_path.string());
    if (complex.size() < 2) {
        throw std::runtime_error("Homology computation requires d1 and d2");
    }
    R2Module<int> module = homology_module(complex, 1, true);
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return;
    }
    module.to_stream(output_file);
    output_file.close();

}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <resolution.scc> [output.scc]" << std::endl;
        return 1;
    }
    std::string filepath = argv[1];
    std::filesystem::path input_path(filepath);
    std::filesystem::path output_path = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::path(insert_suffix_before_extension(filepath, "_minpres"));

    get_minimal_presentation(input_path, output_path);
    std::cout << "Minimal presentation saved to: " << output_path << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
} // main
