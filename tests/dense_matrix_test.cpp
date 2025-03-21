#include <iostream>
#include <cassert>
#include <vector>
#include <dense_matrix.hpp>
#include <matrix_base.hpp>
#include <filesystem>

using namespace graded_linalg;

void test_bitset(){
    bitset bs = bitset(5, 11);
    bitset bs2 = bitset(4, 11);
    print_bitset(bs);
    print_bitset(bs2);
    std::string bitsetString = bitsetToString_alt(bs);
    int length = bitsetString.length();
    std::cout << bitsetString << " of length " << length << std::endl;
    auto number = bs.to_ulong();
    auto number2 = bs2.to_ulong();
    std::cout << "Numbers: " << number << " " << number2 << std::endl;
    std::cout << bs.npos << " " << bs.size() << std::endl;
    int i = bs.find_first();
    int j = bs.find_next(bs.find_next(bs.find_next(bs.find_next(i))));
    std::cout << "First: " << i << " Next: " << j << std::endl;
    auto e = compute_standard_vectors(3);
    std::cout << "Standard vectors: " << e[0] << " " <<  e[1] << " " << e[2] << std::endl;
    for(auto it = bs.find_first(); it != bs.npos; it = bs.find_next(it)){
        std::cout << it << " ";
    }
    std::cout << std::endl;
    std::cout << bitset(1, 1) << std::endl;
    auto count_vec = compute_sum_of_standard_vectors(3);
    std::cout << "Full Bitsets: ";
    for(auto it = count_vec.begin(); it != count_vec.end(); it++){
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

void test_matrix_serialization(){
    DenseMatrix A = DenseMatrix(3, "Identity");
    std::ofstream out("test_matrix.bin", std::ios::binary);
    A.serialize(out);
    out.close();
    std::ifstream in("test_matrix.bin", std::ios::binary);
    DenseMatrix B = DenseMatrix(in);
    assert(A.equals(B));
}


void test_load_transition(){
    auto filename = "decompositions_4.bin";
    auto filename2 = "transitions_4.bin";
    if (std::filesystem::exists(filename2)) {
        std::cout << "File '" << filename2 << "' exists." << std::endl;
    } else {
        std::cout << "Error, file '" << filename2 << "' does not exist." << std::endl;
    }
    DecompTree tree = loadDecompTree(filename);
    vec<transition> matrices = load_transition_list(filename2);
}



int main(){

    return 0;
}