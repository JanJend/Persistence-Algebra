#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void cut_presentation(std::filesystem::path input_path, std::filesystem::path output_path, double x_cutoff = 1.0, double y_cutoff = 1.0) {
    
    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());

    presentation.cut_above(x_cutoff, y_cutoff);

    std::ofstream output_file(output_path);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to open output file " << output_path << std::endl;
        return;
    } else {
        presentation.to_stream(output_file);
        output_file.close();
        std::cout << "Restricted module with cutoffs " << x_cutoff << " and " << y_cutoff << " and saved to: " << output_path << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string filepath;
    double x_cutoff = 1.0; // default value
    double y_cutoff = 1.0; // default value
    std::filesystem::path output_path;
    std::string suffix;
    std::filesystem::path input_path;

    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0] << " <file_path> [x cutoff] [y cutoff] [output_path] \n";
        std::cerr << "  output_path = optional output file path." << std::endl;
        return 1;
    } else {
        filepath = argv[1];
        input_path = std::filesystem::path(filepath);
        try {
            x_cutoff = std::stod(argv[2]);
            y_cutoff = std::stod(argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid double argument" << std::endl;
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
        std::string suffix = "_cut_after_" + std::to_string(x_cutoff) + "_" + std::to_string(y_cutoff);
        std::string modified_path = insert_suffix_before_extension(filepath, suffix);
        output_path = std::filesystem::path(modified_path);
    }
    cut_presentation(input_path, output_path, x_cutoff, y_cutoff);
    return 0;
}