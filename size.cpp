#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;



bool is_decomp_file(const std::filesystem::path& filepath) {
    return filepath.extension() == ".sccsum";
}

void get_size_decomp(std::filesystem::path input_path) {
    std::ifstream input_file(input_path);

    
    if (!input_file.is_open()) {
        std::cerr << "Error opening files" << std::endl;
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
        std::cout << "\rProcessing section " << section_index << "/" << declared_sections 
                  << " (" << type << ")..." << std::flush;
                  
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
            
            R2GradedSparseMatrix<int> minimal_presentation(input_file);
            std::cout << minimal_presentation.num_of_entries() << std::endl;
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
    
    std::cout << "Processed " << processed_sections << std::endl;
}

void get_size(std::filesystem::path input_path) {
    
    R2GradedSparseMatrix<int> minimal_presentation = R2GradedSparseMatrix<int>(input_path.string());
    std::cout << minimal_presentation.num_of_entries() << std::endl;
}


int main(int argc, char** argv) {
    
    std::string filepath;

    if (argc < 2 || argc > 2) {
        std::cerr << "Usage: " << argv[0] << " <file_path>" << std::endl;
        return 1;
    } else {
        filepath = argv[1];
    }
    std::cout << "Size of " << filepath << std::endl;
    std::filesystem::path input_path(filepath);
    
    if (is_decomp_file(input_path)) {
        get_size_decomp(input_path);
    } else {
        get_size(input_path);
    }
    
    return 0;
} // main