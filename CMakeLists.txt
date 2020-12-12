cmake_minimum_required(VERSION 3.16)
project(pv021_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pv021_project src/main.cpp)

# Debug only
# add_compile_options(-O0)

# GOTTA GO FAST
add_compile_options(-Ofast -ffp-model=fast -funsafe-math-optimizations)

# Threads
find_package(Threads REQUIRED)
target_link_libraries(pv021_project PRIVATE Threads::Threads)
