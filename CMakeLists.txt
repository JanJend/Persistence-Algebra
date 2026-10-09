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
    add_executable(module_operations_test tests/module_operations_test.cpp)
    target_link_libraries(module_operations_test grlina::grlina)
    add_test(NAME module_operations_test COMMAND module_operations_test)
    add_executable(module_framework_test tests/modules_test.cpp)
    target_link_libraries(module_framework_test grlina::grlina)
    add_test(NAME module_framework_test COMMAND module_framework_test)
    set_tests_properties(module_framework_test PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

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

    foreach(_csc_test csc_matrix_test csc_graded_test)
        add_executable(${_csc_test} tests/${_csc_test}.cpp)
        target_link_libraries(${_csc_test} grlina::grlina)
        add_test(NAME ${_csc_test} COMMAND ${_csc_test})
    endforeach()
    add_executable(r2grid_gs_matrix_test tests/r2grid_gs_matrix_test.cpp)
    target_link_libraries(r2grid_gs_matrix_test grlina::grlina)
    add_test(NAME r2grid_gs_matrix_test COMMAND r2grid_gs_matrix_test)
    # NEW: opt-in runtime-dimensional coordinate matrix architecture.
    add_executable(dynamic_coordinate_matrix_test tests/dynamic_coordinate_matrix_test.cpp)
    target_link_libraries(dynamic_coordinate_matrix_test grlina::grlina)
    add_test(NAME dynamic_coordinate_matrix_test COMMAND dynamic_coordinate_matrix_test)
    add_executable(dynamic_grid_matrix_test tests/dynamic_grid_matrix_test.cpp)
    target_link_libraries(dynamic_grid_matrix_test grlina::grlina)
    add_test(NAME dynamic_grid_matrix_test COMMAND dynamic_grid_matrix_test)
    add_executable(general_coordinate_kernel_test tests/general_coordinate_kernel_test.cpp)
    target_link_libraries(general_coordinate_kernel_test grlina::grlina)
    add_test(NAME general_coordinate_kernel_test COMMAND general_coordinate_kernel_test)
    add_executable(library_interop_test tests/library_interop_test.cpp)
    target_link_libraries(library_interop_test grlina::grlina)
    add_test(NAME library_interop_test COMMAND library_interop_test)
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

    # Keep the established matrix-level API under test while the module API is
    # introduced alongside it.
    foreach(_legacy_test dense_matrix_test sparse_matrix_test graded_matrix_test hom_space_test)
        add_executable(${_legacy_test} tests/${_legacy_test}.cpp)
        target_link_libraries(${_legacy_test} grlina::grlina)
        add_test(NAME ${_legacy_test} COMMAND ${_legacy_test})
        set_tests_properties(${_legacy_test} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
    # These tests use assert; keep checks active in Release builds as well.
    foreach(_test epsilon_functors_test presentation_operations_test submodule_module_test submodule_reduction_test module_hilbert_test module_operations_test module_framework_test cli_programs_test
                  dense_matrix_test sparse_matrix_test graded_matrix_test hom_space_test csc_matrix_test csc_graded_test r2grid_gs_matrix_test dynamic_coordinate_matrix_test dynamic_grid_matrix_test general_coordinate_kernel_test library_interop_test)
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
