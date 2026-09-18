cmake_minimum_required(VERSION 3.15)
project(Persistence-Algebra VERSION 2.0.0 LANGUAGES CXX)

# Set C++ standard (overridable on the command line for testing newer standards)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Find Boost. Boost.System is header-only since 1.69 and newer distributions
# (Boost >= 1.90) no longer ship a boost_system CMake config, so do not request it.
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

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set optimization flags for Release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Add _debug suffix to executables in Debug mode
set(CMAKE_DEBUG_POSTFIX "_debug")


# Include directories
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# Applies to every target defined below, when OpenMP was found.
if(OpenMP_CXX_FOUND)
    link_libraries(OpenMP::OpenMP_CXX)
endif()


add_executable(shift_endo shift_endo.cpp)
target_link_libraries(shift_endo ${Boost_LIBRARIES})
set_target_properties(shift_endo PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(hom hom.cpp)
target_link_libraries(hom ${Boost_LIBRARIES})
set_target_properties(hom PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(minimize minimize.cpp)
target_link_libraries(minimize ${Boost_LIBRARIES})
set_target_properties(minimize PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(pres_to_quiver pres_to_quiver.cpp)
target_link_libraries(pres_to_quiver ${Boost_LIBRARIES})
set_target_properties(pres_to_quiver PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(resolution resolution.cpp)
target_link_libraries(resolution ${Boost_LIBRARIES})
set_target_properties(resolution PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(snap_grid snap_grid.cpp)
target_link_libraries(snap_grid ${Boost_LIBRARIES})
set_target_properties(snap_grid PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(submodule_at submodule_at.cpp)
target_link_libraries(submodule_at ${Boost_LIBRARIES})
set_target_properties(submodule_at PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(analyse_ind analyse_ind.cpp)
target_link_libraries(analyse_ind ${Boost_LIBRARIES})
set_target_properties(analyse_ind PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(thickness thickness.cpp)
target_link_libraries(thickness ${Boost_LIBRARIES})
set_target_properties(thickness PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(size size.cpp)
target_link_libraries(size ${Boost_LIBRARIES})
set_target_properties(size PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(cut_module cut_module.cpp)
target_link_libraries(cut_module ${Boost_LIBRARIES})
set_target_properties(cut_module PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

foreach(_module_cli birth_death delete_gens_and_rels minimize_pres mpfree_clone)
    add_executable(${_module_cli} ${_module_cli}.cpp)
    target_link_libraries(${_module_cli} ${Boost_LIBRARIES})
    set_target_properties(${_module_cli} PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
endforeach()

install(TARGETS shift_endo cut_module hom minimize pres_to_quiver resolution snap_grid submodule_at analyse_ind thickness size
                birth_death delete_gens_and_rels minimize_pres mpfree_clone
    RUNTIME DESTINATION bin
)

include(CTest)
if(BUILD_TESTING)
    add_executable(isomorphism_test tests/isomorphism_test.cpp)
    target_link_libraries(isomorphism_test ${Boost_LIBRARIES})
    add_test(NAME isomorphism_test COMMAND isomorphism_test)
    set_tests_properties(isomorphism_test PROPERTIES TIMEOUT 120)
    add_executable(isomorphism_benchmark tests/isomorphism_benchmark.cpp)
    target_link_libraries(isomorphism_benchmark ${Boost_LIBRARIES})
    add_test(NAME isomorphism_points COMMAND isomorphism_benchmark file
        ${CMAKE_CURRENT_SOURCE_DIR}/test_presentations/points_wo_density_20_dim2_k_fold_10_min_pres.scc 1001)
    add_test(NAME isomorphism_annulus COMMAND isomorphism_benchmark file
        ${CMAKE_CURRENT_SOURCE_DIR}/test_presentations/presentation/noisy_annulus_socg_bifilt.scc 1001)
    set_tests_properties(isomorphism_points isomorphism_annulus PROPERTIES TIMEOUT 120)
    add_executable(epsilon_functors_test tests/epsilon_functors_test.cpp)
    target_link_libraries(epsilon_functors_test ${Boost_LIBRARIES})
    add_test(NAME epsilon_functors_test COMMAND epsilon_functors_test)
    add_executable(presentation_operations_test tests/presentation_operations_test.cpp)
    target_link_libraries(presentation_operations_test ${Boost_LIBRARIES})
    add_test(NAME presentation_operations_test COMMAND presentation_operations_test)
    add_executable(submodule_module_test tests/submodule_module_test.cpp)
    target_link_libraries(submodule_module_test ${Boost_LIBRARIES})
    add_test(NAME submodule_module_test COMMAND submodule_module_test)
    add_executable(submodule_reduction_test tests/submodule_reduction_test.cpp)
    target_link_libraries(submodule_reduction_test ${Boost_LIBRARIES})
    add_test(NAME submodule_reduction_test COMMAND submodule_reduction_test)
    add_executable(module_hilbert_test tests/module_hilbert_test.cpp)
    target_link_libraries(module_hilbert_test ${Boost_LIBRARIES})
    add_test(NAME module_hilbert_test COMMAND module_hilbert_test)
    add_executable(module_operations_test tests/module_operations_test.cpp)
    target_link_libraries(module_operations_test ${Boost_LIBRARIES})
    add_test(NAME module_operations_test COMMAND module_operations_test)
    add_executable(module_framework_test tests/modules_test.cpp)
    target_link_libraries(module_framework_test ${Boost_LIBRARIES})
    add_test(NAME module_framework_test COMMAND module_framework_test)
    set_tests_properties(module_framework_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(cli_programs_test tests/cli_programs_test.cpp)
    add_test(NAME cli_programs_test COMMAND cli_programs_test
        ${CMAKE_SOURCE_DIR}/tests/fixtures
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
        $<TARGET_FILE:birth_death>
        $<TARGET_FILE:delete_gens_and_rels>
        $<TARGET_FILE:minimize_pres>
        $<TARGET_FILE:mpfree_clone>)

    # Keep the established matrix-level API under test while the module API is
    # introduced alongside it.
    foreach(_legacy_test dense_matrix_test sparse_matrix_test graded_matrix_test hom_space_test)
        add_executable(${_legacy_test} tests/${_legacy_test}.cpp)
        target_link_libraries(${_legacy_test} ${Boost_LIBRARIES})
        add_test(NAME ${_legacy_test} COMMAND ${_legacy_test})
        set_tests_properties(${_legacy_test} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endforeach()
    # These tests use assert; keep checks active in Release builds as well.
    foreach(_test epsilon_functors_test presentation_operations_test submodule_module_test submodule_reduction_test module_hilbert_test module_operations_test module_framework_test cli_programs_test
                  dense_matrix_test sparse_matrix_test graded_matrix_test hom_space_test)
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
        target_link_libraries(${_target} ${Boost_LIBRARIES})
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
