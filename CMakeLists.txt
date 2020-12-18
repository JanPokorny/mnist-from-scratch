cmake_minimum_required(VERSION 3.16)
project(pv021_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Ofast -ffp-model=fast -funsafe-math-optimizations -march=native -fopenmp -msse2 -mfpmath=sse -ftree-vectorize)
add_executable(pv021_project src/main.cpp)

