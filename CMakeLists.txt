cmake_minimum_required(VERSION 3.15)
project(Persistence-Algebra VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Find Boost
find_package(Boost REQUIRED COMPONENTS timer system)

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

# add_executable(analyse_ind analyse_ind.cpp)
# target_link_libraries(analyse_ind ${Boost_LIBRARIES})
# set_target_properties(analyse_ind PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

add_executable(thickness thickness.cpp)
target_link_libraries(thickness ${Boost_LIBRARIES})
set_target_properties(thickness PROPERTIES DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")

install(TARGETS shift_endo hom minimize pres_to_quiver resolution snap_grid submodule_at thickness
    RUNTIME DESTINATION bin
)

