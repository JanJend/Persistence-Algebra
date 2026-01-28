#include "grlina/r2graded_matrix.hpp"
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_thickness(std::filesystem::path input_path) {

    R2GradedSparseMatrix<int> presentation = R2GradedSparseMatrix<int>(input_path.string());
    R2Resolution<int> res(presentation, true);
    int thickness = 0;
    auto v = res.dimension_vector_non_opt(thickness);
    std::cout << presentation.get_num_rows() << " x " << presentation.get_num_cols() << ": ";
    std::cout << "thickness: " << thickness << std::endl;
}

bool is_decomp_file(const std::filesystem::path& filepath) {
    return filepath.extension() == ".sccsum";
}

void compute_decomp_thickness(std::filesystem::path input_path) {

    vec<int> thicks;
    vec<int> sizes;

    std::ifstream input_file(input_path);
    
    if (!input_file.is_open()) {
        std::cerr << "Error opening input file" << std::endl;
        return;
    }
    
    std::string line;
    
    // Read and verify header
    if (!std::getline(input_file, line) || line != "scc2020sum") {
        std::cerr << "Error: Expected 'scc2020sum' as first line" << std::endl;
        return;
    }
    
    // Read number of sections
    int declared_sections;
    if (!(input_file >> declared_sections)) {
        std::cerr << "Error: Could not read number of sections" << std::endl;
        return;
    }
    input_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // <-- consume the rest of the line
    
    int processed_sections = 0;
    int section_index = 0;

    // Process sections until EOF or declared count reached
    while (section_index < declared_sections && !input_file.eof()) {
            // Step 1: read the guaranteed empty line
        if (!std::getline(input_file, line)) {
            std::cerr << "Unexpected EOF while expecting blank line before section "
                    << section_index << std::endl;
            break;
        }
        if (!line.empty()) {
            std::cerr << "Warning: expected blank line before section "
                    << section_index << ", got '" << line << "'\n";
        }

        // Step 2: now read the type line
        if (!std::getline(input_file, line)) {
            std::cerr << "Unexpected EOF while reading type for section "
                    << section_index << std::endl;
            break;
        }
        if (line.empty()) {
            std::cerr << "Warning: type line is empty in section "
                    << section_index << std::endl;
        }
        std::string type = line;
        

        // Update progress
        section_index++;
        // std::cout << "\rProcessing section " << section_index << "/" << declared_sections 
        //          << " (" << type << ")..." << std::flush;
                  
        // Check if next line is scc2020 or firep
        std::streampos pos_before_header = input_file.tellg();
        if (!std::getline(input_file, line) || (line != "scc2020" && line != "firep")) {
            std::cerr << "Warning: Expected 'scc2020' or 'firep' after type: " << type 
                     << " in section " << section_index << std::endl;
            continue;
        }
        
        try {
            // Reset to position before header and let constructor handle parsing
            input_file.seekg(pos_before_header);

            if(type == "free" || type == "cyclic" || type == "interval"){
                thicks.push_back(1);
                R2GradedSparseMatrix<int> A(input_file);
            } else {
                R2GradedSparseMatrix<int> A(input_file);
                R2Resolution<int> res(A, true);
                int thickness = 0;
                auto v = res.dimension_vector_non_opt(thickness);
                thicks.push_back(thickness);
                sizes.push_back(A.get_num_cols()+A.get_num_rows());
            }
            
            processed_sections++;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << type << " section " << section_index 
                     << ": " << e.what() << std::endl;
        }
    }
    
    // Check for remaining content after declared sections
    if (section_index >= declared_sections && !input_file.eof()) {
        std::string remaining;
        while (std::getline(input_file, remaining)) {
            if (!remaining.empty()) {
                std::cerr << "Warning: Found additional content after " << declared_sections 
                         << " declared sections. File may contain more sections than declared." << std::endl;
                break;
            }
        }
    }
    std::cout << std::flush;
    // Final report
    if (processed_sections != declared_sections) {
        std::cerr << "Warning: Declared " << declared_sections << " sections, but successfully processed " 
                 << processed_sections << " sections." << std::endl;
    }

    int max = 1;
    int max_index = -1;
    for (size_t i = 0; i < thicks.size(); i++) {
        if (thicks[i] > max) {
            max = thicks[i];
            max_index = i;
        }
    }
    std::cout << "layer thickness:" << max << ", ind: " << max_index << ", size: " << sizes[max_index] << std::endl;
}




int main(int argc, char** argv) {
    
    std::string filepath;
    std::filesystem::path output_path;

    if (argc < 2 || argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    } else {
        filepath = argv[1];
    }

    std::filesystem::path input_path(filepath);
    
    if (is_decomp_file(input_path)) {
        // std::cout << "Analysing decomposed sccsum file." << std::endl;
        compute_decomp_thickness(input_path);
    } else {
        // std::cout << "Analysing single scc file of unknown type."  << std::endl;
        compute_thickness(input_path);
    }
    
    return 0;
} // main