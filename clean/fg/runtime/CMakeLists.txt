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
# F9: developer test builds only (see clean/fg/fg_limits.h); public and CI builds keep it OFF.
option(WRONGGPU_FG_DEV_QUALIFIED "Developer test builds only: FG modes taken without the release gate" OFF)
if(WRONGGPU_FG_DEV_QUALIFIED)
  add_compile_definitions(WRONGGPU_FG_DEV_QUALIFIED=1)
  message(WARNING "fg runtime: WRONGGPU_FG_DEV_QUALIFIED is ON (developer test build; never publish this build)")
endif()

set(WG_FG_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR})
add_library(wg_fg_runtime STATIC
  ${WG_FG_RUNTIME_DIR}/fg_contract.cpp
  ${WG_FG_RUNTIME_DIR}/fg_model.cpp
  ${WG_FG_RUNTIME_DIR}/fg_net1_plan.cpp
  ${WG_FG_RUNTIME_DIR}/fg_exec_plan.cpp
  ${WG_FG_RUNTIME_DIR}/fg_history.cpp
  ${WG_FG_RUNTIME_DIR}/fg_worker.cpp
  ${WG_FG_RUNTIME_DIR}/../../converter/src/sha256.cpp)
target_include_directories(wg_fg_runtime PUBLIC ${WG_FG_RUNTIME_DIR})
target_link_libraries(wg_fg_runtime PUBLIC Threads::Threads)

add_executable(test_fg_runtime ${WG_FG_RUNTIME_DIR}/tests/test_fg_runtime.cpp)
target_link_libraries(test_fg_runtime PRIVATE wg_fg_runtime)

# F7a: the FG session (fg_session.h) joins the runtime with the presenter boundary (clean/fg/presenter) and the game
# adapter core (clean/fg/proxy). Their sources are compiled into a dependency library here, so this folder still builds
# on its own; their own suites stay in their folders. On Windows the session also gets the presenter source over the
# back-end factory (fg_session_win.h).
set(WG_FG_PRESENTER_DIR ${WG_FG_RUNTIME_DIR}/../presenter)
set(WG_FG_PROXY_DIR ${WG_FG_RUNTIME_DIR}/../proxy)
add_library(wg_fg_session_deps STATIC
  ${WG_FG_PRESENTER_DIR}/fg_presenter.cpp
  ${WG_FG_PRESENTER_DIR}/fg_target_scheduler.cpp
  ${WG_FG_PRESENTER_DIR}/fg_fake_presenter.cpp
  ${WG_FG_PRESENTER_DIR}/fg_present_engine.cpp
  ${WG_FG_PROXY_DIR}/fg_proxy_core.cpp
  ${WG_FG_PROXY_DIR}/fg_settings.cpp)
target_include_directories(wg_fg_session_deps PUBLIC ${WG_FG_RUNTIME_DIR} ${WG_FG_PRESENTER_DIR} ${WG_FG_PROXY_DIR})
target_link_libraries(wg_fg_session_deps PUBLIC wg_fg_runtime)
set(WG_FG_SESSION_SOURCES ${WG_FG_RUNTIME_DIR}/fg_session.cpp)
if(WIN32)
  target_sources(wg_fg_session_deps PRIVATE
    ${WG_FG_PRESENTER_DIR}/fg_win_common.cpp
    ${WG_FG_PRESENTER_DIR}/fg_win_comp_presenter.cpp
    ${WG_FG_PRESENTER_DIR}/fg_dxgi_presenter.cpp
    ${WG_FG_PRESENTER_DIR}/fg_presenter_factory.cpp)
  target_link_libraries(wg_fg_session_deps PUBLIC
    d3d11.lib d3d12.lib dxgi.lib dcomp.lib gdi32.lib user32.lib mincore.lib)
  target_compile_definitions(wg_fg_session_deps PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN WG_FG_SESSION_WIN=1)
  list(APPEND WG_FG_SESSION_SOURCES ${WG_FG_RUNTIME_DIR}/fg_session_win.cpp)
endif()
add_library(wg_fg_session STATIC ${WG_FG_SESSION_SOURCES})
target_link_libraries(wg_fg_session PUBLIC wg_fg_session_deps)

# F7b: the D3D12 executor (fg_d3d12_executor.h) and its end-to-end test. ctest runs the CPU checks;
# `test_fg_d3d12_e2e --gpu --dxil <build>/dx12` adds the headless GPU leg (kernels from clean/dx12/build_dx12.py --only).
if(WIN32)
  add_library(wg_fg_d3d12 STATIC ${WG_FG_RUNTIME_DIR}/fg_d3d12_executor.cpp)
  target_link_libraries(wg_fg_d3d12 PUBLIC wg_fg_session)
  add_executable(test_fg_d3d12_e2e ${WG_FG_RUNTIME_DIR}/tests/test_fg_d3d12_e2e.cpp)
  target_link_libraries(test_fg_d3d12_e2e PRIVATE wg_fg_d3d12)
  list(APPEND WG_FG_D3D12_TARGETS wg_fg_d3d12 test_fg_d3d12_e2e)
endif()

add_executable(test_fg_session ${WG_FG_RUNTIME_DIR}/tests/test_fg_session.cpp)
target_link_libraries(test_fg_session PRIVATE wg_fg_session)
set(WG_FG_SESSION_TARGETS wg_fg_session_deps wg_fg_session test_fg_session)

# Mutants: each breaks one session rule (fg_session.cpp, WG_FG_SESSION_MUTANT); the suite must catch every one.
#  1 shared work per index   2 menu latch dropped   3 wrong intermediate time   4 silent continuation after a refusal
#  5 history advanced per index   6 a count change resets history
set(WG_FG_SESSION_MUTANTS 1 2 3 4 5 6)
foreach(m ${WG_FG_SESSION_MUTANTS})
  add_executable(test_fg_session_mutant_${m} ${WG_FG_RUNTIME_DIR}/tests/test_fg_session.cpp ${WG_FG_SESSION_SOURCES})
  target_link_libraries(test_fg_session_mutant_${m} PRIVATE wg_fg_session_deps)
  target_compile_definitions(test_fg_session_mutant_${m} PRIVATE WG_FG_SESSION_MUTANT=${m})
  list(APPEND WG_FG_SESSION_TARGETS test_fg_session_mutant_${m})
endforeach()

foreach(t wg_fg_runtime test_fg_runtime ${WG_FG_SESSION_TARGETS} ${WG_FG_D3D12_TARGETS})
  if(MSVC)
    set_target_properties(${t} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # CI's flags; /permissive- turns on /Zc:lambda, which the worker's failure-path tests depend on
    target_compile_options(${t} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endforeach()

add_test(NAME fg_runtime COMMAND test_fg_runtime)
add_test(NAME fg_session COMMAND test_fg_session)
if(WIN32)
  add_test(NAME fg_d3d12_e2e COMMAND test_fg_d3d12_e2e WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
foreach(m ${WG_FG_SESSION_MUTANTS})
  add_test(NAME fg_session_mutant_${m} COMMAND test_fg_session_mutant_${m})
  set_tests_properties(fg_session_mutant_${m} PROPERTIES WILL_FAIL TRUE)
endforeach()
