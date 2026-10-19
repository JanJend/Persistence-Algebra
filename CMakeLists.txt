cmake_minimum_required(VERSION 3.15)
project(Persistence-Algebra VERSION 2.0.0 LANGUAGES CXX)

# NEW: header-only library target, usable without building CLI programs/tests.
set(_grlina_standalone OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_grlina_standalone ON)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
    endif()
    include(CTest)
endif()
option(GRLINA_BUILD_PROGRAMS "Build Persistence-Algebra command line programs" ${_grlina_standalone})
set(_grlina_tests_default OFF)
if(_grlina_standalone AND BUILD_TESTING)
    set(_grlina_tests_default ON)
endif()
option(GRLINA_BUILD_TESTS "Build Persistence-Algebra tests" ${_grlina_tests_default})

# Find Boost. Boost.System is header-only since 1.69 and newer distributions
# (Boost >= 1.90) no longer ship a boost_system CMake config, so do not request it.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
    cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED COMPONENTS timer)

# OpenMP is optional: used when available, ignored otherwise.
# Apple clang ships no OpenMP runtime, so point CMake at Homebrew's libomp if present.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE _brew_libomp
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_brew_libomp AND EXISTS "${_brew_libomp}")
        set(OpenMP_ROOT "${_brew_libomp}")
    endif()
endif()
find_package(OpenMP)

add_library(grlina INTERFACE)
add_library(grlina::grlina ALIAS grlina)
target_compile_features(grlina INTERFACE cxx_std_17)
target_include_directories(grlina INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(grlina INTERFACE Boost::timer)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grlina INTERFACE OpenMP::OpenMP_CXX)
endif()

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/grlinaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/grlinaConfig.cmake
    INSTALL_DESTINATION lib/cmake/grlina)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/grlinaConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(TARGETS grlina EXPORT grlinaTargets)
install(EXPORT grlinaTargets NAMESPACE grlina:: DESTINATION lib/cmake/grlina)
install(DIRECTORY include/grlina DESTINATION include)
install(DIRECTORY include/multicritical DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/grlinaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/grlinaConfigVersion.cmake DESTINATION lib/cmake/grlina)

if(GRLINA_BUILD_PROGRAMS)
add_executable(shift_endo shift_endo.cpp)
target_link_libraries(shift_endo grlina::grlina)
set_target_properties(shift_endo PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(hom hom.cpp)
target_link_libraries(hom grlina::grlina)
set_target_properties(hom PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(minimize minimize.cpp)
target_link_libraries(minimize grlina::grlina)
set_target_properties(minimize PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(pres_to_quiver pres_to_quiver.cpp)
target_link_libraries(pres_to_quiver grlina::grlina)
set_target_properties(pres_to_quiver PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(resolution resolution.cpp)
target_link_libraries(resolution grlina::grlina)
set_target_properties(resolution PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(snap_grid snap_grid.cpp)
target_link_libraries(snap_grid grlina::grlina)
set_target_properties(snap_grid PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(submodule_at submodule_at.cpp)
target_link_libraries(submodule_at grlina::grlina)
set_target_properties(submodule_at PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(analyse_ind analyse_ind.cpp)
target_link_libraries(analyse_ind grlina::grlina)
set_target_properties(analyse_ind PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(thickness thickness.cpp)
target_link_libraries(thickness grlina::grlina)
set_target_properties(thickness PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(size size.cpp)
target_link_libraries(size grlina::grlina)
set_target_properties(size PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(cut_module cut_module.cpp)
target_link_libraries(cut_module grlina::grlina)
set_target_properties(cut_module PROPERTIES DEBUG_POSTFIX "_debug")

add_executable(iso_test iso_test.cpp)
target_link_libraries(iso_test grlina::grlina)
set_target_properties(iso_test PROPERTIES DEBUG_POSTFIX "_debug")


foreach(_module_cli delete_gens_and_rels minimize_pres mpfree_clone)
    add_executable(${_module_cli} ${_module_cli}.cpp)
    target_link_libraries(${_module_cli} grlina::grlina)
    set_target_properties(${_module_cli} PROPERTIES DEBUG_POSTFIX "_debug")
endforeach()

install(TARGETS shift_endo cut_module hom minimize pres_to_quiver resolution snap_grid submodule_at analyse_ind thickness size
                delete_gens_and_rels minimize_pres mpfree_clone
    RUNTIME DESTINATION bin
)

endif() # GRLINA_BUILD_PROGRAMS

if(GRLINA_BUILD_TESTS)
    enable_testing()
    add_executable(isomorphism_test tests/isomorphism_test.cpp)
    target_link_libraries(isomorphism_test grlina::grlina)
    add_test(NAME isomorphism_test COMMAND isomorphism_test)
    set_tests_properties(isomorphism_test PROPERTIES TIMEOUT 120)
    add_executable(isomorphism_benchmark tests/isomorphism_benchmark.cpp)
    target_link_libraries(isomorphism_benchmark grlina::grlina)
    add_test(NAME isomorphism_points COMMAND isomorphism_benchmark file
        ${CMAKE_CURRENT_SOURCE_DIR}/test_presentations/points_wo_density_20_dim2_k_fold_10_min_pres.scc 1001)
    add_test(NAME isomorphism_annulus COMMAND isomorphism_benchmark file
        ${CMAKE_CURRENT_SOURCE_DIR}/test_presentations/presentation/noisy_annulus_socg_bifilt.scc 1001)
    set_tests_properties(isomorphism_points isomorphism_annulus PROPERTIES TIMEOUT 120)
    add_executable(epsilon_functors_test tests/epsilon_functors_test.cpp)
    target_link_libraries(epsilon_functors_test grlina::grlina)
    add_test(NAME epsilon_functors_test COMMAND epsilon_functors_test)
    add_executable(presentation_operations_test tests/presentation_operations_test.cpp)
    target_link_libraries(presentation_operations_test grlina::grlina)
    add_test(NAME presentation_operations_test COMMAND presentation_operations_test)
    add_executable(submodule_module_test tests/submodule_module_test.cpp)
    target_link_libraries(submodule_module_test grlina::grlina)
    add_test(NAME submodule_module_test COMMAND submodule_module_test)
    add_executable(submodule_reduction_test tests/submodule_reduction_test.cpp)
    target_link_libraries(submodule_reduction_test grlina::grlina)
    add_test(NAME submodule_reduction_test COMMAND submodule_reduction_test)
    add_executable(module_hilbert_test tests/module_hilbert_test.cpp)
    target_link_libraries(module_hilbert_test grlina::grlina)
    add_test(NAME module_hilbert_test COMMAND module_hilbert_test)
    add_executable(module_landscape_test tests/module_landscape_test.cpp)
    target_link_libraries(module_landscape_test grlina::grlina)
    add_test(NAME module_landscape_test COMMAND module_landscape_test)
    add_executable(module_operations_test tests/module_operations_test.cpp)
    target_link_libraries(module_operations_test grlina::grlina)
    add_test(NAME module_operations_test COMMAND module_operations_test)
    add_executable(module_framework_test tests/modules_test.cpp)
    target_link_libraries(module_framework_test grlina::grlina)
    add_test(NAME module_framework_test COMMAND module_framework_test)
    set_tests_properties(module_framework_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(upset_degree_test tests/upset_degree_test.cpp)
    target_link_libraries(upset_degree_test grlina::grlina)
    add_test(NAME upset_degree_test COMMAND upset_degree_test)
    add_executable(upset_matrix_test tests/upset_matrix_test.cpp)
    target_link_libraries(upset_matrix_test grlina::grlina)
    add_test(NAME upset_matrix_test COMMAND upset_matrix_test)
    # Cross-check against multicritical's actual parser when its checkout is available.
    set(_multicritical_source "${CMAKE_CURRENT_SOURCE_DIR}/../multicritical")
    if(EXISTS "${_multicritical_source}/scc_mod/include/scc/Scc.h")
        target_include_directories(upset_matrix_test PRIVATE "${_multicritical_source}/scc_mod/include"
                                                            "${_multicritical_source}/mpp_utils_mod/include"
                                                            "${_multicritical_source}/phat_mod/include"
                                                            "${_multicritical_source}/include")
        target_compile_definitions(upset_matrix_test PRIVATE GRLINA_MULTICRITICAL_SOURCE_DIR="${_multicritical_source}")
    endif()
    add_executable(upset_chain_complex_test tests/upset_chain_complex_test.cpp)
    target_link_libraries(upset_chain_complex_test grlina::grlina)
    if(EXISTS "${_multicritical_source}/sample_files/compressed_deg_rips_100_points.scc")
        add_test(NAME upset_chain_complex_test COMMAND upset_chain_complex_test
            "${_multicritical_source}/sample_files/compressed_deg_rips_100_points.scc")
        set_tests_properties(upset_chain_complex_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    add_executable(multicritical_copy_test tests/multicritical_copy_test.cpp)
    target_link_libraries(multicritical_copy_test grlina::grlina)
    add_test(NAME multicritical_copy_test COMMAND multicritical_copy_test)
    foreach(_sample compressed_deg_rips_100_points full_deg_rips_20_points wheel)
        if(EXISTS "${_multicritical_source}/sample_files/${_sample}.scc")
            add_test(NAME multicritical_copy_${_sample} COMMAND multicritical_copy_test
                "${_multicritical_source}/sample_files/${_sample}.scc")
        endif()
    endforeach()

    if(GRLINA_BUILD_PROGRAMS)
    add_executable(cli_programs_test tests/cli_programs_test.cpp)
    add_test(NAME cli_programs_test COMMAND cli_programs_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures
        $<TARGET_FILE:minimize>
        $<TARGET_FILE:hom>
        $<TARGET_FILE:shift_endo>
        $<TARGET_FILE:pres_to_quiver>
        $<TARGET_FILE:resolution>
        $<TARGET_FILE:snap_grid>
        $<TARGET_FILE:submodule_at>
        $<TARGET_FILE:analyse_ind>
        $<TARGET_FILE:thickness>
        $<TARGET_FILE:size>
        $<TARGET_FILE:cut_module>
        $<TARGET_FILE:delete_gens_and_rels>
        $<TARGET_FILE:minimize_pres>
        $<TARGET_FILE:mpfree_clone>)
    target_link_libraries(cli_programs_test grlina::grlina)
    endif()

    add_executable(new_coord_graded_sparse_matrix_test tests/new_coord_graded_sparse_matrix_test.cpp)
    target_link_libraries(new_coord_graded_sparse_matrix_test grlina::grlina)
    add_test(NAME new_coord_graded_sparse_matrix_test COMMAND new_coord_graded_sparse_matrix_test)
    add_executable(new_grid_graded_sparse_matrix_test tests/new_grid_graded_sparse_matrix_test.cpp)
    target_link_libraries(new_grid_graded_sparse_matrix_test grlina::grlina)
    add_test(NAME new_grid_graded_sparse_matrix_test COMMAND new_grid_graded_sparse_matrix_test)
    foreach(_csc_test csc_matrix_test csc_graded_test)
        add_executable(${_csc_test} tests/${_csc_test}.cpp)
        target_link_libraries(${_csc_test} grlina::grlina)
        add_test(NAME ${_csc_test} COMMAND ${_csc_test})
    endforeach()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(_csc_cxx20_supported OFF)
        if("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
            set(_csc_cxx20_supported ON)
        endif()
        add_test(NAME csc_warning_test COMMAND ${CMAKE_COMMAND}
            "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DBOOST_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}"
            "-DCXX20_SUPPORTED=${_csc_cxx20_supported}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/csc_warnings.cmake)
    endif()
    # Runtime-dimensional suites still require the removed runtime_matrices classes.
    add_executable(graded_linear_system_benchmark tests/graded_linear_system_benchmark.cpp)
    target_link_libraries(graded_linear_system_benchmark grlina::grlina)
    # Measure production reduction costs even in a diagnostic build. The
    # benchmark explicitly validates every result outside its timed region.
    target_compile_definitions(graded_linear_system_benchmark PRIVATE GRLINA_ENABLE_CHECKS=0)
    if(MSVC)
        target_compile_options(graded_linear_system_benchmark PRIVATE /O2)
    else()
        target_compile_options(graded_linear_system_benchmark PRIVATE -O2)
    endif()
    add_test(NAME graded_linear_system_benchmark COMMAND graded_linear_system_benchmark 64 1 all)
    set_tests_properties(graded_linear_system_benchmark PROPERTIES TIMEOUT 120)

    # Optional comparison with the external multi-chunk checkout; no download
    # or dependency on it is introduced for library clients or other tests.
    set(GRLINA_MULTI_CHUNK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../multi-chunk" CACHE PATH
        "multi-chunk checkout used by the chain-complex minimization benchmark")
    if(EXISTS "${GRLINA_MULTI_CHUNK_ROOT}/include/multi_chunk/multi_chunk.h")
        add_executable(chain_complex_minimization_benchmark tests/chain_complex_minimization_benchmark.cpp)
        target_link_libraries(chain_complex_minimization_benchmark PRIVATE grlina::grlina)
        target_include_directories(chain_complex_minimization_benchmark SYSTEM PRIVATE
            "${GRLINA_MULTI_CHUNK_ROOT}/include" "${GRLINA_MULTI_CHUNK_ROOT}/mpp_utils_mod/include"
            "${GRLINA_MULTI_CHUNK_ROOT}/phat_mod/include" "${GRLINA_MULTI_CHUNK_ROOT}/scc_mod/include")
        target_compile_definitions(chain_complex_minimization_benchmark PRIVATE
            GRLINA_ENABLE_CHECKS=0 MULTI_CHUNK_TIMERS=0 NDEBUG)
        if(MSVC)
            target_compile_options(chain_complex_minimization_benchmark PRIVATE /O2)
        else()
            target_compile_options(chain_complex_minimization_benchmark PRIVATE -O3)
        endif()
        add_test(NAME chain_complex_minimization_benchmark COMMAND chain_complex_minimization_benchmark 64 1 all)
        add_test(NAME chain_complex_minimization_benchmark_scc COMMAND chain_complex_minimization_benchmark
            file "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/cancellation.scc" 1)
        add_test(NAME chain_complex_minimization_benchmark_nonzero_terminal COMMAND chain_complex_minimization_benchmark
            file "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/interval_resolution_expected.scc" 1)
        add_test(NAME chain_complex_minimization_benchmark_unsorted_scc COMMAND chain_complex_minimization_benchmark
            file "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chain_minimization_unsorted.scc" 1)
        set_tests_properties(chain_complex_minimization_benchmark chain_complex_minimization_benchmark_scc
            chain_complex_minimization_benchmark_nonzero_terminal chain_complex_minimization_benchmark_unsorted_scc
            PROPERTIES TIMEOUT 120)
    endif()

    # Keep the established matrix-level API under test while the module API is
    # introduced alongside it.
    foreach(_legacy_test dense_matrix_test sparse_matrix_test matrix_solver_test graded_matrix_test hom_space_test graded_linear_system_test)
        add_executable(${_legacy_test} tests/${_legacy_test}.cpp)
        target_link_libraries(${_legacy_test} grlina::grlina)
        add_test(NAME ${_legacy_test} COMMAND ${_legacy_test})
        set_tests_properties(${_legacy_test} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
    # These tests use assert; keep checks active in Release builds as well.
    foreach(_test epsilon_functors_test presentation_operations_test submodule_module_test submodule_reduction_test module_hilbert_test module_landscape_test module_operations_test module_framework_test upset_degree_test upset_matrix_test upset_chain_complex_test multicritical_copy_test cli_programs_test
                  dense_matrix_test sparse_matrix_test matrix_solver_test graded_matrix_test hom_space_test graded_linear_system_test new_coord_graded_sparse_matrix_test new_grid_graded_sparse_matrix_test csc_matrix_test csc_graded_test)
        if(NOT TARGET ${_test})
            continue()
        endif()
        # These suites exercise rejection of invalid input as well as algebra.
        # Keep diagnostic checks on even when optimizing their test binaries.
        target_compile_definitions(${_test} PRIVATE GRLINA_ENABLE_CHECKS=1)
        if(MSVC)
            target_compile_options(${_test} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${_test} PRIVATE -UNDEBUG)
        endif()
    endforeach()

    # Exercise the compiler-selected policy and both explicit overrides. Test
    # assertions remain active independently of the library's check policy.
    foreach(_mode debug optimized force_on force_off)
        set(_target validation_policy_${_mode}_test)
        add_executable(${_target} tests/validation_policy_test.cpp)
        target_link_libraries(${_target} grlina::grlina)
        add_test(NAME ${_target} COMMAND ${_target})
        if(_mode STREQUAL "debug" OR _mode STREQUAL "force_off")
            if(MSVC)
                target_compile_options(${_target} PRIVATE /Od /UNDEBUG)
            else()
                target_compile_options(${_target} PRIVATE -O0 -UNDEBUG)
            endif()
        else()
            if(MSVC)
                target_compile_options(${_target} PRIVATE /O2)
                target_compile_definitions(${_target} PRIVATE NDEBUG)
            else()
                target_compile_options(${_target} PRIVATE -O2 -UNDEBUG)
            endif()
        endif()
        if(_mode STREQUAL "force_on")
            target_compile_definitions(${_target} PRIVATE GRLINA_ENABLE_CHECKS=1)
        elseif(_mode STREQUAL "force_off")
            target_compile_definitions(${_target} PRIVATE GRLINA_ENABLE_CHECKS=0)
        endif()
        if(_mode STREQUAL "debug" OR _mode STREQUAL "force_on")
            target_compile_definitions(${_target} PRIVATE EXPECT_GRLINA_CHECKS=1)
        else()
            target_compile_definitions(${_target} PRIVATE EXPECT_GRLINA_CHECKS=0)
        endif()
    endforeach()
endif()
