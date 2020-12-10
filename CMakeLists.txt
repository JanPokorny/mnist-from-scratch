cmake_minimum_required(VERSION 3.16)
project(pv021_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_executable(pv021_project src/main.cpp)

# Debug only
# add_compile_options(-O0)

# GOTTA GO FAST
add_compile_options(-ffast-math -funsafe-math-optimizations -O3)

# OpenMP
find_package(OpenMP)
add_compile_options(-fopenmp)
target_link_libraries(pv021_project PRIVATE OpenMP::OpenMP_CXX)
