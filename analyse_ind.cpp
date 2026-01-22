#include <grlina/graded_linalg.hpp>
#include <iostream>
#include <filesystem>

using namespace graded_linalg;


int compute_hom_space(R2GradedSparseMatrix<int> A, R2GradedSparseMatrix<int> B, int type, bool info = false) {
    
    using aida_result = std::pair< SparseMatrix<int>, vec<std::pair<int,int>> >;

    int result_full;
    int result_A;
    int result_B;
    int result_semi;

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
        // std::cout << "After hom-exact, time: " << (timer.elapsed().wall * 1e-6) << std::endl;
        result_B = hom_space_new.size();
    }
    // Full linear system
    if(type == 1 || type == -1){
        boost::timer::cpu_timer timer;
        hom_space_full = hom_space_no_opt<r2degree, int, R2GradedSparseMatrix<int>>(A, B, true, vec<int>(), vec<int>(), info);
        // std::cout << "After full linear system, time: " << (timer.elapsed().wall * 1e-6) << "ms" <<  std::endl;
        result_full = hom_space_full.first.get_num_cols();
    } else if ( (type == -2) || (type == -3)){
        // vec<int> system_size = no_opt_system_info<r2degree, int, R2GradedSparseMatrix<int>>(A, B);
    }
    // Semi-restriction
    if(type == 2 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_semi_restriction = hom_space_optimised<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        // std::cout << "After semi restriction, time: " << (timer.elapsed().wall * 1e-6) << std::endl;
        result_semi = hom_space_semi_restriction.first.get_num_cols();
    }
    // Algorithm A
    if(type == 3 || type == -1 || type == -2){
        boost::timer::cpu_timer timer;
        hom_space_full_rest = hom_space_full_restriction<r2degree, int, R2GradedSparseMatrix<int>>(A,B, vec<int>(), vec<int>(), info);
        // std::cout << "After full restriction, time: " << (timer.elapsed().wall * 1e-6) << std::endl;
        result_A = hom_space_full_rest.first.get_num_cols();
    }
    
    if(type == -2 || type == -1){
        if(result_A != result_B || result_A != result_semi){
            std::cerr << "Warning: Different results for different methods: " 
                      << "Semi: " << result_semi 
                      << ", Alg A: " << result_A << ", Alg B: " << result_B << std::endl;
        } else {
        }
        if( type == -1){
            if(result_full != result_A){
                std::cerr << "Warning: Full linear system: " << result_full << std::endl;
            }
        }
        return result_B;
    } else {
        if(type == 0){
            return result_B;
        } else if(type == 1){
            return result_full;
        } else if(type == 2){
            return result_semi;
        } else if(type == 3){
            return result_A;
        } else {
            std::cerr << "Error: Unknown type " << type << std::endl;
            return -1;
        }
    }
 
}

bool is_decomp_file(const std::filesystem::path& filepath) {
    return filepath.extension() == ".sccsum";
}

void compute_end(std::filesystem::path input_path) {
    R2GradedSparseMatrix<int> A(input_path.string());
    R2GradedSparseMatrix<int> B = A;
    int dim = compute_hom_space(A, B, -2, false);
    std::cout << "Dimension of hom-space: " << dim << std::endl;
}

void compute_decomp_end(std::filesystem::path input_path) {

    vec<pair<int>> non_int_dims;

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
            
            if(type == "free" || type == "cyclic" || type == "interval"){
                R2GradedSparseMatrix<int> A(input_file);
            } else {
                R2GradedSparseMatrix<int> A(input_file);
                R2GradedSparseMatrix<int> B = A;
                non_int_dims.push_back(std::make_pair(A.get_num_cols() + A.get_num_rows(), 0));
                int dim;
                if(A.get_num_rows() < 100){
                    dim = compute_hom_space(A, B, 0, false);
                } else {
                    dim = compute_hom_space(A, B, -2, false);
                }
                non_int_dims.back().second = dim;
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

    std::cout << "Non-interval size and hom-space dimensions:" << std::endl;
    for(auto& p : non_int_dims){
        std::cout << "Size: " << p.first << ", dim hom: " << p.second << std::endl;
    }
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
        compute_decomp_end(input_path);
    } else {
        // std::cout << "Analysing single scc file of unknown type."  << std::endl;
        compute_end(input_path);
    }
    
    return 0;
} // main