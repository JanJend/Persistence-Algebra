cmake_minimum_required(VERSION 3.15)
project(Persistence-Algebra VERSION 1.0 LANGUAGES CXX)

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

install(TARGETS shift_endo cut_module hom minimize pres_to_quiver resolution snap_grid submodule_at analyse_ind thickness size
    RUNTIME DESTINATION bin
)

