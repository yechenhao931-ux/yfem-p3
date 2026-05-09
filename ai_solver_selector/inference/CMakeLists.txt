# AI Solver Selector - inference CMake.
#
# Builds ai_solver_demo, linking against MFEM and libtorch.
#
#   cd ai_solver_selector/inference
#   cmake -B build -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
#                  -DMFEM_DIR=../../build
#   cmake --build build
#
# Tested with libtorch >= 2.0 (download the CXX11-ABI binary distribution from
# pytorch.org and point Torch_DIR at <libtorch>/share/cmake/Torch).

cmake_minimum_required(VERSION 3.18)
project(ai_solver_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(MFEM REQUIRED)

add_executable(ai_solver_demo demo_select.cpp)
target_include_directories(ai_solver_demo PRIVATE
   ${MFEM_INCLUDE_DIRS}
   ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(ai_solver_demo PRIVATE
   ${MFEM_LIBRARIES}
   ${TORCH_LIBRARIES})
target_compile_options(ai_solver_demo PRIVATE ${TORCH_CXX_FLAGS})
