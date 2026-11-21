# SPDX-License-Identifier: MIT
# Frame generation runtime groundwork (issue #96, F3a + F4a): the versioned FG contract, the network grid table, the
# local model check against data/fg.v1.map.json, net1's fused D3D12 plan and recorder, frame ingest and the history
# state machine, and the create-time worker. CPU only: the recorder is checked through a fake command recorder.
# Stand-alone:   cmake -S clean/fg/runtime -B build/fg-runtime && cmake --build build/fg-runtime --config Release
#                ctest --test-dir build/fg-runtime -C Release --output-on-failure
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED PROJECT_NAME)
  project(wronggpu_fg_runtime LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
endif()
find_package(Threads REQUIRED)

set(WG_FG_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR})
add_library(wg_fg_runtime STATIC
  ${WG_FG_RUNTIME_DIR}/fg_contract.cpp
  ${WG_FG_RUNTIME_DIR}/fg_model.cpp
  ${WG_FG_RUNTIME_DIR}/fg_net1_plan.cpp
  ${WG_FG_RUNTIME_DIR}/fg_history.cpp
  ${WG_FG_RUNTIME_DIR}/fg_worker.cpp
  ${WG_FG_RUNTIME_DIR}/../../converter/src/sha256.cpp)
target_include_directories(wg_fg_runtime PUBLIC ${WG_FG_RUNTIME_DIR})
target_link_libraries(wg_fg_runtime PUBLIC Threads::Threads)

add_executable(test_fg_runtime ${WG_FG_RUNTIME_DIR}/tests/test_fg_runtime.cpp)
target_link_libraries(test_fg_runtime PRIVATE wg_fg_runtime)

foreach(t wg_fg_runtime test_fg_runtime)
  if(MSVC)
    set_target_properties(${t} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # CI's flags; /permissive- turns on /Zc:lambda, which the worker's failure-path tests depend on
    target_compile_options(${t} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

add_test(NAME fg_runtime COMMAND test_fg_runtime)
