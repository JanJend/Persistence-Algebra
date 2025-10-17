#include "grlina/r2graded_matrix.hpp"
#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void  compute_submodule_at(std::filesystem::path input_path, std::filesystem::path output_path, r2degree degree) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    R2GradedSparseMatrix<int> submodule = presentation.submodule_generated_at(degree);
    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        submodule.to_stream(output_file);
        output_file.close();
        std::cout << "Submodule at degree (" << degree.first << ", " << degree.second << ") computed and saved to: " << output_path << std::endl;
    }
}

r2degree string_to_r2degree(const std::string& str) {
    size_t comma_pos = str.find(',');
    if (comma_pos == std::string::npos) {
        throw std::invalid_argument("Invalid format for r2degree. Expected format: x,y");
    }
    double x = std::stod(str.substr(0, comma_pos));
    double y = std::stod(str.substr(comma_pos + 1));
    return {x, y};
}

int main(int argc, char** argv) {
    
    std::string filepath;
    r2degree degree;
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << " <degree>" << std::endl;
    } else {
        filepath = argv[1];
        degree = string_to_r2degree(argv[2]);
    }

    std::filesystem::path input_path(filepath);
    
    std::string modified_path = insert_suffix_before_extension(filepath, "_min");
    std::filesystem::path output_path(modified_path);
    
    compute_submodule_at(input_path, output_path, degree);
    return 0;
} // main