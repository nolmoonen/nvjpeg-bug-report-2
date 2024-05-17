cmake_minimum_required(VERSION 3.22)

project(nvjpeg-bug-report LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(repro main.cpp)
target_link_libraries(repro PRIVATE CUDA::cudart_static CUDA::nvjpeg_static)
