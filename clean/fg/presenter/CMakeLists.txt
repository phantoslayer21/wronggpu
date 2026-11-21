# SPDX-License-Identifier: MIT
# Frame generation presentation boundary (clean/fg/presenter): IFgPresenter, the target-time scheduler and the
# deterministic fake presenter, with their CPU tests. No GPU, no window, no platform SDK.
# Stand-alone:   cmake -S clean/fg/presenter -B build/fg-presenter && cmake --build build/fg-presenter --config Release
#                ctest --test-dir build/fg-presenter -C Release --output-on-failure
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED PROJECT_NAME)
  project(wronggpu_fg_presenter LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
endif()

add_library(wg_fg_presenter STATIC
  ${CMAKE_CURRENT_LIST_DIR}/fg_presenter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fg_target_scheduler.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fg_fake_presenter.cpp)
target_include_directories(wg_fg_presenter PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(wg_fg_presenter PUBLIC cxx_std_17)

add_executable(test_fg_presenter ${CMAKE_CURRENT_LIST_DIR}/tests/test_fg_presenter.cpp)
target_link_libraries(test_fg_presenter PRIVATE wg_fg_presenter)

foreach(t wg_fg_presenter test_fg_presenter)
  set_target_properties(${t} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MSVC)
    # the CI flags: /permissive- (with /Zc:lambda) is what game code is built with
    target_compile_options(${t} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

add_test(NAME fg_presenter COMMAND test_fg_presenter)
