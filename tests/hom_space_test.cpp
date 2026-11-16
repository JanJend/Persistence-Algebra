#include "grlina/graded_linalg.hpp"
#include "grlina/matrices/base.hpp"
#include "grlina/matrices/r2.hpp"
#include "grlina/matrices/sparse.hpp"
#include <iostream>
#include <cassert>
#include <filesystem>
#include <random>
#include <grlina/matrices/matrix_algorithms.hpp>

using namespace graded_linalg; 
using Mat = R2GradedSparseMatrix<int>;

void hom_space_test(){
    const auto test_directory = std::filesystem::path(__FILE__).parent_path();
    std::string path1 = (test_directory / "../test_presentations/hom_test_domain.scc").string();
    std::string path2 = (test_directory / "../test_presentations/hom_test_target.scc").string();
    Mat domain = Mat(path1);
    Mat target = Mat(path2);
    domain.print_graded();
    target.print_graded();
    domain.compute_rows_forward();
    auto result1 = hom_space_basis(domain, target, false);
    auto result2 = hom_space_basis(domain, target, true);
    
    
    auto non_zero1 = general_reduction<int, Mat>(result1);
    auto non_zero2 = general_reduction<int, Mat>(result2);
    assert(non_zero2.size() == result2.size());

    // Reduce first hom space.
    auto result1_basis = vec_restriction<int, Mat>(result1, non_zero1);
    assert(result1_basis.size() == result2.size());
    
}

template<class Storage, class Matrix>
void check_hom_coefficients(const Matrix& source, const Matrix& target,
                            HomAlgorithm algorithm, const std::vector<Matrix>& expected) {
    auto coefficients = hom_space_coefficients<Storage>(source, target, algorithm);
    auto actual = graded_maps_from_coefficients(coefficients, source, target);
    assert(actual.size() == expected.size());
    for (const auto& map : actual) assert(is_homomorphism(source, target, map));
    auto reference = expected;
    reduce_matrix_family_modulo(reference, actual, target);
    assert(actual.empty());
}

template<class Storage, class Matrix>
void check_quotient_coefficients(const Matrix& source, const typename Matrix::degree_type& shift) {
    auto parent = std::make_shared<const Module<Matrix>>(source);
    const auto eta = Homomorphism<Matrix>::canonical_shift(parent, shift);
    auto E = hom_space_coefficients<Storage>(source, source, HomAlgorithm::optimised);
    const auto original_E = E;
    auto reference_E = hom_space_basis_new(source, source, true, false, true, 0);
    auto reference_H = hom_space_basis_new(source, eta.target()->presentation(), true, false, true, 0);
    const auto shifted_dimension = reference_H.size();
    for (auto& map : reference_E)
        set_geometric_degrees(map, geometric_row_degrees(source), geometric_row_degrees(eta.target()->presentation()));
    reduce_matrix_family_modulo(reference_E, reference_H, eta.target()->presentation());
    for (auto algorithm : {HomAlgorithm::optimised, HomAlgorithm::full_restriction, HomAlgorithm::hom_exactness}) {
        for (bool supplied : {false, true}) {
            auto result = supplied ? End_d_0_coefficients(eta, E, algorithm)
                                   : End_d_0_coefficients<Storage>(eta, algorithm);
            assert(std::size_t(result.quotient.get_num_cols()) == reference_H.size());
            assert(result.quotient.get_num_cols() + result.subspace.get_num_cols() == shifted_dimension);
            auto actual = graded_maps_from_coefficients(result.quotient, source, eta.target()->presentation());
            auto reference = reference_H;
            // Compare modulo W as well as target relations, using the independent
            // original matrix-family reduction rather than the new column reducer.
            auto W = graded_maps_from_coefficients(result.subspace, source, eta.target()->presentation());
            auto image_check = W;
            auto image_reference = reference_E;
            reduce_matrix_family_modulo(image_reference, image_check, eta.target()->presentation());
            assert(image_check.empty());
            reference.insert(reference.end(), W.begin(), W.end());
            reduce_matrix_family_modulo(reference, actual, eta.target()->presentation());
            assert(actual.empty());
            assert(E.data == original_E.data); // Supplied End_0 is never consumed.
        }
    }
}

template<class Matrix>
void compare_hom_algorithms() {
    using Degree = typename Matrix::degree_type;
    std::mt19937 random(260410381);
    auto presentation = [&](int rows, int columns) {
        Matrix result(columns, rows);
        vec<Degree> row_degrees(rows), column_degrees(columns);
        for (auto& degree : row_degrees) degree = {double(random() % 3), double(random() % 3)};
        for (auto& degree : column_degrees) degree = {double(random() % 4), double(random() % 4)};
        array<int> data(columns);
        for (int c = 0; c < columns; ++c)
            for (int r = 0; r < rows; ++r)
                if (Degree_traits<Degree>::smaller_equal(row_degrees[r], column_degrees[c]) && random() % 2)
                    data[c].push_back(r);
        result.assign_data(std::move(data));
        set_geometric_degrees(result, column_degrees, row_degrees);
        return result;
    };
    for (int example = 0; example < 32; ++example) {
        auto source = presentation(example % 4, example % 3);
        const auto target = presentation((example + 1) % 4, (example + 2) % 4);
        source.compute_rows_forward();
        // Independent direct computation, including its quotient by zero lifts.
        const auto expected = hom_space_basis_new(source, target, false, false, true, 0);
        for (auto algorithm : {HomAlgorithm::optimised, HomAlgorithm::full_restriction, HomAlgorithm::hom_exactness}) {
            check_hom_coefficients<CompactStorage<std::ptrdiff_t>>(source, target, algorithm, expected);
            check_hom_coefficients<vec<vec<std::ptrdiff_t>>>(source, target, algorithm, expected);
            const auto plain = hom_space_basis_ungraded(source, target, algorithm);
            assert(plain.size() == expected.size());
            auto actual = hom_space_basis(source, target, algorithm);
            assert(actual.size() == expected.size());
            for (const auto& map : actual) {
                assert(map.is_graded_matrix());
                assert(is_homomorphism(source, target, map));
                assert(geometric_col_degrees(map) == geometric_row_degrees(source));
                assert(geometric_row_degrees(map) == geometric_row_degrees(target));
            }
            auto reference = expected;
            reduce_matrix_family_modulo(reference, actual, target);
            assert(actual.empty());
        }
        // Compare direct image elimination against the independent full-Hom
        // computation, followed by transport and reduction in the shifted target.
        for (Degree shift : {Degree{0,0}, Degree{0.5,1}, Degree{4,4}}) {
            check_quotient_coefficients<CompactStorage<std::ptrdiff_t>>(source, shift);
            check_quotient_coefficients<vec<vec<std::ptrdiff_t>>>(source, shift);
            auto shifted_target = target;
            shifted_target.shift(shift);
            auto transported = expected;
            for (auto& map : transported)
                set_geometric_degrees(map, geometric_row_degrees(source), geometric_row_degrees(shifted_target));
            std::vector<Matrix> empty;
            reduce_matrix_family_modulo(empty, transported, shifted_target);
            auto image = hom_space_shift_image_basis(source, target, shifted_target);
            assert(image.size() == transported.size());
            for (const auto& map : image) {
                assert(map.is_graded_matrix());
                assert(is_homomorphism(source, shifted_target, map));
            }
            reduce_matrix_family_modulo(transported, image, shifted_target);
            assert(image.empty());
        }
    }
}

void shift_image_obstructions() {
    // X = [0,3) + [1,2). The shifted second-to-first map uses an
    // already-born target generator, but does not lift to an endomorphism.
    Mat X(2, 2, {{0}, {1}}, {{3,0}, {2,0}}, {{0,0}, {1,0}});
    X.compute_rows_forward();
    auto shifted = X;
    shifted.shift({1,0});
    Mat bad(2, 2, {{}, {0}}, X.row_degrees, shifted.row_degrees);
    assert(is_homomorphism(X, shifted, bad));
    set_geometric_degrees(bad, X.row_degrees, X.row_degrees);
    assert(!is_homomorphism(X, X, bad));
    assert(hom_space_shift_image_basis(X, X, shifted).size() == 1);
    for (auto algorithm : {HomAlgorithm::optimised, HomAlgorithm::full_restriction, HomAlgorithm::hom_exactness})
        assert(End_d_0(X, {1,0}, false, algorithm).size() == 2);

    // The image of the only early generator is represented by a late generator
    // in the shifted sharp restriction system. Filtering that system loses it.
    Mat source(0, 1, {}, {}, {{0,1}});
    source.compute_rows_forward();
    Mat target(1, 2, {{0,1}}, {{2,2}}, {{1,0}, {0,1}});
    shifted = target;
    shifted.shift({2,2});
    auto image = hom_space_shift_image_basis(source, target, shifted);
    assert(image.size() == 1 && image[0].get_col(0) == vec<int>{0});

    // A Q variable killed by eta is still needed to satisfy the ORIGINAL
    // relation q0 + q1 = 0. It must remain in the system with zero payload.
    source = Mat(1, 2, {{0,1}}, {{2,0}}, {{0,0}, {1,0}});
    source.compute_rows_forward();
    target = Mat(1, 1, {{0}}, {{3,0}}, {{0,0}});
    shifted = target;
    shifted.shift({2,0});
    image = hom_space_shift_image_basis(source, target, shifted);
    assert(image.size() == 1);
    assert(image[0].get_col(0) == vec<int>{0} && image[0].get_col(1).empty());
}

template<class Storage>
void coefficient_quotient_in_place() {
    SparseMatrix<std::ptrdiff_t, Storage> W(1, 3, array<std::ptrdiff_t>{{0}});
    SparseMatrix<std::ptrdiff_t, Storage> H(3, 3, {{0}, {1}, {0,1}});
    const auto* w_storage = W.data.data();
    const auto* h_storage = H.data.data();
    H.compute_rows_forward();
    reduce_quotient_in_place(W, H);
    assert(W.get_num_cols() == 1 && H.get_num_cols() == 1);
    assert(W.data.data() == w_storage && H.data.data() == h_storage);
    assert(H.get_col(0) == vec<std::ptrdiff_t>{1});
    assert(!H.rows_computed && H._rows.empty());
    bool rejected = false;
    try { reduce_quotient_in_place(H, H); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main() {
    coefficient_quotient_in_place<CompactStorage<std::ptrdiff_t>>();
    coefficient_quotient_in_place<vec<vec<std::ptrdiff_t>>>();
    hom_space_test();
    shift_image_obstructions();
    compare_hom_algorithms<Mat>();
    compare_hom_algorithms<R2GradedSparseMatrix<int, CompactStorage<int>>>();
    compare_hom_algorithms<GridGradedSparseMatrix<double, 2, int>>();
    return 0;
}
