# SPDX-License-Identifier: MIT
# wronggpu-convert: the install-time model converter (spec 08). C++17, standard library only.
cmake_minimum_required(VERSION 3.16)
project(wronggpu_convert LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

set(WGC_LIB_SOURCES
  src/sha256.cpp src/json.cpp src/numeric.cpp src/map.cpp src/locate.cpp src/lz4block.cpp
  src/convert.cpp src/convert5.cpp src/ops5.cpp src/output.cpp src/pe_version.cpp src/api.cpp)

add_library(wgc_core STATIC ${WGC_LIB_SOURCES})
target_include_directories(wgc_core PUBLIC src)
target_link_libraries(wgc_core PUBLIC Threads::Threads)

# The decode operations must be bit-exact: strict IEEE semantics, no contraction into FMA.
if(MSVC)
  target_compile_options(wgc_core PUBLIC /W4 /permissive- /utf-8 /EHsc /fp:precise)
  target_compile_definitions(wgc_core PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(wgc_core PUBLIC -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
endif()

add_executable(wronggpu-convert src/main.cpp)
target_link_libraries(wronggpu-convert PRIVATE wgc_core)
if(MINGW)
  target_link_options(wronggpu-convert PRIVATE -municode -static)
endif()

add_executable(wgc-unit-tests tests/unit_tests.cpp)
target_link_libraries(wgc-unit-tests PRIVATE wgc_core)
if(MINGW)
  target_link_options(wgc-unit-tests PRIVATE -static)
endif()

enable_testing()
add_test(NAME unit COMMAND wgc-unit-tests --work ${CMAKE_CURRENT_BINARY_DIR}/unit-work)
# 08-T7: the spec team's decompression vectors (spec 08 section 4.4), data/lz4-block-test.json of this repository.
set(WGC_LZ4_VECTORS "${CMAKE_CURRENT_SOURCE_DIR}/../../data/lz4-block-test.json")
if(EXISTS "${WGC_LZ4_VECTORS}")
  add_test(NAME decompression-vectors COMMAND wgc-unit-tests --lz4-vectors ${WGC_LZ4_VECTORS})
endif()
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME synthetic
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
                   --exe $<TARGET_FILE:wronggpu-convert> --work ${CMAKE_CURRENT_BINARY_DIR}/test-work)
  # Map version 5 (spec 57, preset F); its spec-data and synthetic-library paths: see the script's --help.
  add_test(NAME synthetic5
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests5.py
                   --exe $<TARGET_FILE:wronggpu-convert> --unit $<TARGET_FILE:wgc-unit-tests>
                   --work ${CMAKE_CURRENT_BINARY_DIR}/test-work5)
endif()
