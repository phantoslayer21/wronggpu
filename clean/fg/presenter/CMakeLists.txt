# SPDX-License-Identifier: GPL-3.0-only
# Frame generation presentation boundary (clean/fg/presenter): IFgPresenter, the target-time scheduler, the
# deterministic fake presenter and the shared display-slot engine, with their CPU tests (no GPU, no window, no platform
# SDK). On Windows also the production back ends (composition / hardware flip queue and DXGI flip-discard), the
# back-end selector and their tests: CPU checks by default; `test_fg_win_presenters --gpu` adds headless GPU checks
# (no window is shown) and runs only on request.
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
  ${CMAKE_CURRENT_LIST_DIR}/fg_fake_presenter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/fg_present_engine.cpp)
target_include_directories(wg_fg_presenter PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(wg_fg_presenter PUBLIC cxx_std_17)

add_executable(test_fg_presenter ${CMAKE_CURRENT_LIST_DIR}/tests/test_fg_presenter.cpp)
target_link_libraries(test_fg_presenter PRIVATE wg_fg_presenter)

set(WG_FG_PRESENTER_TARGETS wg_fg_presenter test_fg_presenter)

if(WIN32)
  add_library(wg_fg_win_presenter STATIC
    ${CMAKE_CURRENT_LIST_DIR}/fg_win_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fg_win_comp_presenter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fg_dxgi_presenter.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fg_presenter_factory.cpp)
  target_link_libraries(wg_fg_win_presenter PUBLIC wg_fg_presenter
    d3d11.lib d3d12.lib dxgi.lib gdi32.lib user32.lib mincore.lib)
  target_compile_definitions(wg_fg_win_presenter PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

  add_executable(test_fg_win_presenters ${CMAKE_CURRENT_LIST_DIR}/tests/test_fg_win_presenters.cpp)
  target_link_libraries(test_fg_win_presenters PRIVATE wg_fg_win_presenter)

  add_executable(test_fg_hwfq_window ${CMAKE_CURRENT_LIST_DIR}/tests/test_fg_hwfq_window.cpp)
  target_link_libraries(test_fg_hwfq_window PRIVATE wg_fg_win_presenter dwmapi.lib)

  list(APPEND WG_FG_PRESENTER_TARGETS wg_fg_win_presenter test_fg_win_presenters test_fg_hwfq_window)
endif()

foreach(t ${WG_FG_PRESENTER_TARGETS})
  set_target_properties(${t} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MSVC)
    # the CI flags: /permissive- (with /Zc:lambda) is what game code is built with
    target_compile_options(${t} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

add_test(NAME fg_presenter COMMAND test_fg_presenter)
if(WIN32)
  add_test(NAME fg_win_presenters COMMAND test_fg_win_presenters)
endif()
