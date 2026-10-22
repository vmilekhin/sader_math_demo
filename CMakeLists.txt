cmake_minimum_required(VERSION 3.20)
project(sader_math LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sader_math src/module.cpp)
target_compile_options(sader_math PRIVATE 
    -O3 
    -march=native 
    -fopt-info-vec-optimized
    -Wall 
    -Wextra 
    -Wpedantic
)
