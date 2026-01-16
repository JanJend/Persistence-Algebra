#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


void compute_hom_space(R2GradedSparseMatrix<int> A, R2GradedSparseMatrix<int> B, int type, bool info = false) {
    
    using aida_result = std::pair< SparseMatrix<int>, vec<std::pair<int,int>> >;

    A.sort_columns_lexicographically();
    B.sort_columns_lexicographically();
    A.sort_rows_lexicographically();
    B.sort_rows_lexicographically();

    A.compute_rows_forward();

    aida_result hom_space_full;
    aida_result hom_space_semi_restriction;
    aida_result hom_space_full_rest;
    vec<R2GradedSparseMatrix<int>> hom_space_new;

    // Algorithm B
    if(type == 0|| type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_new = hom_space_basis_new<r2degree, int, R2GradedSparseMatrix<int>>(A,B,true, info);
        std::cout << "After hom-exact, dim hom: " << hom_space_new.size() 
                  << ", time: " << timer.format() << std::endl;
    }
    // Full linear system
    if(type == 1 || type == -1){
        boost::timer::cpu_timer timer;
        hom_space_full = hom_space_no_opt<r2degree, int, R2GradedSparseMatrix<int>>(A, B, true, vec<int>(), vec<int>(), info);
        std::cout << "After full linear system, dim hom: " << hom_space_full.first.get_num_cols() 
                  << ", time: " << timer.format() << std::endl;
    } else if ( (type == -2) || (type = -3)){
        vec<int> system_size = no_opt_system_info<r2degree, int, R2GradedSparseMatrix<int>>(A, B);
    }
    // Semi-restriction
    if(type == 2 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_semi_restriction = hom_space_optimised<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        std::cout << "After semi restriction, dim hom: " << hom_space_semi_restriction.first.get_num_cols() 
                  << ", time: " << timer.format() << std::endl;
    }
    // Algorithm A
    if(type == 3 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_full_rest = hom_space_full_restriction<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        std::cout << "After full restriction, dim hom: " << hom_space_full_rest.first.get_num_cols() 
                  << ", time: " << timer.format() << std::endl;
    }
    
        
}

bool is_decomp_file(const std::filesystem::path& filepath) {
    return filepath.extension() == ".sccsum";
}

void compute_end(std::filesystem::path input_path) {
    std::cout << "Type of module unknown." << std::endl;
    R2GradedSparseMatrix<int> A(input_path.string());
    R2GradedSparseMatrix<int> B(input_path.string());
    compute_hom_space(A, B, -1, true);
}

void compute_decomp_end(std::filesystem::path input_path) {
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
    
    output_file << "scc2020sum\n" << declared_sections << "\n\n";

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
            
            R2GradedSparseMatrix<int> A(input_file);
            R2GradedSparseMatrix<int> B(input_file);
            compute_hom_space(A, B, std::stoi(type), true);
            
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
    
    std::cout << "Processed " << processed_sections
              << " sections, saved to: " << output_path << std::endl;
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
        compute_decomp_end(input_path);
    } else {
        compute_end(input_path);
    }
    
    return 0;
} // main