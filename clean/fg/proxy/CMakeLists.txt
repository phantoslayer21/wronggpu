# SPDX-License-Identifier: MIT
# Frame generation game adapter groundwork (issue #96, F8a): the host-neutral adapter contract, the adapter core
# (install-off gate, feature-scoped routing, capability, tag-to-packet assembly, single present owner) and the per-game
# FG settings keys. CPU only: tests run against a fake game interface, a fake host and a fake executor.
# Stand-alone:   cmake -S clean/fg/proxy -B build/fg-proxy && cmake --build build/fg-proxy --config Release
#                ctest --test-dir build/fg-proxy -C Release --output-on-failure
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED PROJECT_NAME)
  project(wronggpu_fg_proxy LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
endif()

set(WG_FG_PROXY_DIR ${CMAKE_CURRENT_LIST_DIR})
# The FG runtime contract (clean/fg/runtime/fg_contract.h): when it is in the tree, the test also checks the adapter's
# packets and sessions through fg_contract_bridge.h with the runtime's own validation.
set(WG_FG_CONTRACT_DIR "${WG_FG_PROXY_DIR}/../runtime" CACHE PATH "Folder of fg_contract.h / fg_contract.cpp")
set(WG_FG_PROXY_SOURCES ${WG_FG_PROXY_DIR}/fg_proxy_core.cpp ${WG_FG_PROXY_DIR}/fg_settings.cpp)
set(WG_FG_TEST_SOURCES ${WG_FG_PROXY_DIR}/tests/test_fg_proxy.cpp)
set(WG_FG_HAVE_CONTRACT OFF)
if(EXISTS "${WG_FG_CONTRACT_DIR}/fg_contract.h" AND EXISTS "${WG_FG_CONTRACT_DIR}/fg_contract.cpp")
  set(WG_FG_HAVE_CONTRACT ON)
  list(APPEND WG_FG_TEST_SOURCES ${WG_FG_CONTRACT_DIR}/fg_contract.cpp)
endif()
message(STATUS "fg proxy: runtime contract bridge test ${WG_FG_HAVE_CONTRACT}")

function(wg_fg_flags t)
  if(MSVC)
    set_target_properties(${t} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # CI's flags; /permissive- turns on /Zc:lambda
    target_compile_options(${t} PRIVATE /W4 /WX /permissive- /utf-8 /EHsc)
  else()
    target_compile_options(${t} PRIVATE -Wall -Wextra -Werror)
  endif()
endfunction()

add_library(wg_fg_proxy STATIC ${WG_FG_PROXY_SOURCES})
target_include_directories(wg_fg_proxy PUBLIC ${WG_FG_PROXY_DIR})
wg_fg_flags(wg_fg_proxy)

function(wg_fg_test name mutant)
  if(mutant EQUAL 0)
    add_executable(${name} ${WG_FG_TEST_SOURCES})
    target_link_libraries(${name} PRIVATE wg_fg_proxy)
  else()
    # a deliberately broken core, compiled into the test itself
    add_executable(${name} ${WG_FG_TEST_SOURCES} ${WG_FG_PROXY_SOURCES})
    target_include_directories(${name} PRIVATE ${WG_FG_PROXY_DIR})
    target_compile_definitions(${name} PRIVATE WG_FG_PROXY_MUTANT=${mutant})
  endif()
  # the per-game settings file's parser (Windows only: wg_state.h)
  target_include_directories(${name} PRIVATE ${WG_FG_PROXY_DIR}/../../wgshared/include)
  if(WG_FG_HAVE_CONTRACT)
    target_include_directories(${name} PRIVATE ${WG_FG_CONTRACT_DIR})
    target_compile_definitions(${name} PRIVATE WG_FG_HAVE_RUNTIME_CONTRACT=1)
  endif()
  if(WIN32)
    target_link_libraries(${name} PRIVATE advapi32)
  endif()
  wg_fg_flags(${name})
endfunction()

wg_fg_test(test_fg_proxy 0)
add_test(NAME fg_proxy COMMAND test_fg_proxy)

# Mutants: each breaks one rule (fg_proxy_core.cpp / .h, fg_settings.cpp); the suite must catch every one.
#  1 install switch ignored   2 every feature taken as FG   3 present owner not checked   4 an interval presented twice
#  5 adapter LUID not matched 6 case-folded settings value   7 modes gate ignored          8 stale tags reported as missing
foreach(m 1 2 3 4 5 6 7 8)
  wg_fg_test(test_fg_proxy_mutant_${m} ${m})
  add_test(NAME fg_proxy_mutant_${m} COMMAND test_fg_proxy_mutant_${m})
  set_tests_properties(fg_proxy_mutant_${m} PROPERTIES WILL_FAIL TRUE)
endforeach()
