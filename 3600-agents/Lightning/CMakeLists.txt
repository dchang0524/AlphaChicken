cmake_minimum_required(VERSION 3.12)
project(CPPHikaru3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build a shared library (not a Python module)
add_library(cpphikaru3 SHARED
    src/cpphikaru3.cpp
    src/agent_wrapper.cpp
    src/search.cpp
    src/evaluation.cpp
    src/voronoi.cpp
    src/bfs.cpp
    src/hmm.cpp
    src/zobrist.cpp
    src/game_rules.cpp
)

# Optimization flags
target_compile_options(cpphikaru3 PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -fPIC
)

# Include directories
target_include_directories(cpphikaru3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set output name
set_target_properties(cpphikaru3 PROPERTIES
    OUTPUT_NAME "libcpphikaru3"
    PREFIX ""
)

# On macOS, set install name for the library
if(APPLE)
    set_target_properties(cpphikaru3 PROPERTIES
        INSTALL_NAME_DIR "@rpath"
    )
endif()
