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
    # CI's flags; /permissive- turns on /Zc:lambda (C++ sources only: a resource script takes none of them)
    target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /WX /permissive- /utf-8 /EHsc>)
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

# F8b: the Win32 hook transport (WrongGPU-FG.asi, the RR plugin's hook-mode delivery: IAT / delay-load redirects, the
# genuine interface file untouched) and the DXGI swap-chain hook seam. Windows only. Tests run against a stand-in
# interface DLL (synthetic export names and version), two stand-in game modules that import it by name (static and
# delay-load import tables), an exe that delay-loads it, and fake swap-chain / factory COM objects.
if(WIN32)
  if(MSVC)
    enable_language(RC)
  endif()
  set(WG_RR_PROXY_SRC_DIR "${WG_FG_PROXY_DIR}/../../rr_f/proxy/src")
  # The qualified title bindings (bindings/*.json) become headers of macros in the build tree, so the interface's names
  # stay in the binding data (bindings/fg_binding_gen.cmake; string(JSON) needs CMake 3.19).
  if(CMAKE_VERSION VERSION_LESS 3.19)
    message(FATAL_ERROR "fg proxy: the FG title bindings need CMake 3.19 or newer")
  endif()
  include(${WG_FG_PROXY_DIR}/bindings/fg_binding_gen.cmake)
  set(WG_FG_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
  wg_fg_generate_binding("${WG_FG_PROXY_DIR}/bindings/cp2077-2.14.1.json" "${WG_FG_GEN_DIR}/fg_binding_cp2077.h" CP2077)
  # the bounded PE view, safe reads and scan budget are the RR guard's own (rr_guard.cpp), reused unchanged
  set(WG_FG_WIN32_SOURCES ${WG_FG_PROXY_DIR}/win32/fg_asi_transport.cpp ${WG_FG_PROXY_DIR}/win32/fg_swapchain_hook.cpp
                          ${WG_FG_PROXY_DIR}/win32/fg_sl214_codec.cpp ${WG_FG_PROXY_DIR}/win32/fg_title_bindings.cpp
                          ${WG_RR_PROXY_SRC_DIR}/rr_guard.cpp)
  add_library(wg_fg_win32 STATIC ${WG_FG_WIN32_SOURCES})
  target_include_directories(wg_fg_win32 PUBLIC ${WG_FG_GEN_DIR})
  target_link_libraries(wg_fg_win32 PUBLIC wg_fg_proxy version)
  wg_fg_flags(wg_fg_win32)

  add_library(wg_fg_asi SHARED ${WG_FG_PROXY_DIR}/win32/fg_asi_entry.cpp)
  target_link_libraries(wg_fg_asi PRIVATE wg_fg_win32)
  target_include_directories(wg_fg_asi PRIVATE ${WG_FG_PROXY_DIR}/../../wgshared/include)
  set_target_properties(wg_fg_asi PROPERTIES OUTPUT_NAME "WrongGPU-FG" PREFIX "" SUFFIX ".asi")
  wg_fg_flags(wg_fg_asi)

  add_library(fg_mock_interposer SHARED ${WG_FG_PROXY_DIR}/tests/fg_mock_interposer.cpp
                                        ${WG_FG_PROXY_DIR}/tests/fg_mock_interposer.rc)
  wg_fg_flags(fg_mock_interposer)
  add_library(fg_mock_game SHARED ${WG_FG_PROXY_DIR}/tests/fg_mock_game.cpp)
  target_link_libraries(fg_mock_game PRIVATE fg_mock_interposer)
  wg_fg_flags(fg_mock_game)
  add_library(fg_mock_game_delay SHARED ${WG_FG_PROXY_DIR}/tests/fg_mock_game.cpp)
  target_link_libraries(fg_mock_game_delay PRIVATE fg_mock_interposer delayimp)
  target_link_options(fg_mock_game_delay PRIVATE /DELAYLOAD:fg_mock_interposer.dll)
  wg_fg_flags(fg_mock_game_delay)

  add_executable(test_fg_win32_transport ${WG_FG_PROXY_DIR}/tests/test_fg_win32_transport.cpp)
  target_link_libraries(test_fg_win32_transport PRIVATE wg_fg_win32 fg_mock_interposer)
  add_dependencies(test_fg_win32_transport fg_mock_game fg_mock_game_delay wg_fg_asi)
  wg_fg_flags(test_fg_win32_transport)
  add_executable(test_fg_win32_delay_exe ${WG_FG_PROXY_DIR}/tests/test_fg_win32_transport.cpp)
  target_compile_definitions(test_fg_win32_delay_exe PRIVATE FG_TEST_DELAY_EXE=1)
  target_link_libraries(test_fg_win32_delay_exe PRIVATE wg_fg_win32 fg_mock_interposer delayimp)
  target_link_options(test_fg_win32_delay_exe PRIVATE /DELAYLOAD:fg_mock_interposer.dll)
  wg_fg_flags(test_fg_win32_delay_exe)

  add_test(NAME fg_win32_transport COMMAND test_fg_win32_transport)
  add_test(NAME fg_win32_delay_exe COMMAND test_fg_win32_delay_exe)
  # the shipped plugin, one process per settings case (the settings are read once, at InitializeASI)
  set(WG_FG_ASI_CASES "${CMAKE_CURRENT_BINARY_DIR}/fg-asi-cases")
  file(WRITE "${WG_FG_ASI_CASES}/off/wronggpu-setup/settings.ini" "fg=off\nfg_multiplier=4\n")
  file(WRITE "${WG_FG_ASI_CASES}/on/wronggpu-setup/settings.ini" "fg=on\nfg_preset=B\nfg_multiplier=6\n")
  file(WRITE "${WG_FG_ASI_CASES}/invalid/wronggpu-setup/settings.ini" "fg=On\n")
  file(MAKE_DIRECTORY "${WG_FG_ASI_CASES}/default")
  foreach(leg off on invalid default)
    add_test(NAME fg_win32_asi_${leg} COMMAND test_fg_win32_transport --asi ${leg})
    set_tests_properties(fg_win32_asi_${leg} PROPERTIES ENVIRONMENT "WG_STATE_GAME_DIR=${WG_FG_ASI_CASES}/${leg}")
  endforeach()

  # Mutants of the transport and the seam: each breaks one rule; the in-process suite must catch every one.
  #  1 install-off gate ignored        2 a forwarded call loses an argument  3 scan budget ignored
  #  4 a retry begins a new interval   5 a test present reaches the callback 6 the present owner is not claimed
  #  7 restore leaves slots redirected 8 a failing callback stays attached
  foreach(m 1 2 3 4 5 6 7 8)
    add_executable(test_fg_win32_mutant_${m} ${WG_FG_PROXY_DIR}/tests/test_fg_win32_transport.cpp
                   ${WG_FG_PROXY_DIR}/win32/fg_asi_transport.cpp ${WG_FG_PROXY_DIR}/win32/fg_swapchain_hook.cpp
                   ${WG_RR_PROXY_SRC_DIR}/rr_guard.cpp)
    target_compile_definitions(test_fg_win32_mutant_${m} PRIVATE WG_FG_WIN32_MUTANT=${m})
    target_link_libraries(test_fg_win32_mutant_${m} PRIVATE wg_fg_proxy version fg_mock_interposer)
    add_dependencies(test_fg_win32_mutant_${m} fg_mock_game fg_mock_game_delay)
    wg_fg_flags(test_fg_win32_mutant_${m})
    add_test(NAME fg_win32_mutant_${m} COMMAND test_fg_win32_mutant_${m})
    set_tests_properties(fg_win32_mutant_${m} PROPERTIES WILL_FAIL TRUE TIMEOUT 60)
  endforeach()

  # The Cyberpunk 2077 title binding (interface build 2.14.1) and the 2.14 native codec: binding data, layouts, chain
  # walk, options / state / tags / constants, and a whole game session through the stand-in interface (the binding's
  # own GUIDs, buffer types and feature-function names; the stand-in's export names), the factory hook and the present
  # bridge to the core and an executor. Synthetic values only.
  add_executable(test_fg_cp2077_binding ${WG_FG_PROXY_DIR}/tests/test_fg_cp2077_binding.cpp)
  target_link_libraries(test_fg_cp2077_binding PRIVATE wg_fg_win32 fg_mock_interposer)
  add_dependencies(test_fg_cp2077_binding fg_mock_game fg_mock_game_delay)
  wg_fg_flags(test_fg_cp2077_binding)
  add_test(NAME fg_cp2077_binding COMMAND test_fg_cp2077_binding)

  # the shipped plugin in a process with the game's exe name: the binding is selected; no interface module is loaded
  # there, so nothing is patched (fg=on), or nothing is read at all (fg=off)
  add_executable(fg_cp2077_asi_host ${WG_FG_PROXY_DIR}/tests/fg_cp2077_asi_host.cpp)
  set_target_properties(fg_cp2077_asi_host PROPERTIES OUTPUT_NAME "Cyberpunk2077"
                        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cp2077-host")
  target_link_libraries(fg_cp2077_asi_host PRIVATE wg_fg_win32)  # result words only
  add_dependencies(fg_cp2077_asi_host wg_fg_asi)
  wg_fg_flags(fg_cp2077_asi_host)
  file(WRITE "${WG_FG_ASI_CASES}/cp2077-off/wronggpu-setup/settings.ini" "fg=off\n")
  file(WRITE "${WG_FG_ASI_CASES}/cp2077-on/wronggpu-setup/settings.ini" "fg=on\nfg_multiplier=4\n")
  foreach(leg off on)
    add_test(NAME fg_cp2077_asi_${leg} COMMAND fg_cp2077_asi_host $<TARGET_FILE:wg_fg_asi> ${leg})
    set_tests_properties(fg_cp2077_asi_${leg} PROPERTIES ENVIRONMENT "WG_STATE_GAME_DIR=${WG_FG_ASI_CASES}/cp2077-${leg}")
  endforeach()

  # Mutants of the codec and the present path: each breaks one rule; the suite must catch every one.
  #  1 a generated count of 0 accepted        2 a count above the maximum accepted   3 state written past its version
  #  4 the chain walk ignores the GUID        5 the tag lifetime gate ignored        6 the factory hook never installed
  #  7 the present bridge never submits       8 the frame clock never advances
  foreach(m 1 2 3 4 5 6 7 8)
    add_executable(test_fg_cp2077_mutant_${m} ${WG_FG_PROXY_DIR}/tests/test_fg_cp2077_binding.cpp ${WG_FG_WIN32_SOURCES})
    target_compile_definitions(test_fg_cp2077_mutant_${m} PRIVATE WG_FG_CP2077_MUTANT=${m})
    target_include_directories(test_fg_cp2077_mutant_${m} PRIVATE ${WG_FG_GEN_DIR})
    target_link_libraries(test_fg_cp2077_mutant_${m} PRIVATE wg_fg_proxy version fg_mock_interposer)
    add_dependencies(test_fg_cp2077_mutant_${m} fg_mock_game fg_mock_game_delay)
    wg_fg_flags(test_fg_cp2077_mutant_${m})
    add_test(NAME fg_cp2077_mutant_${m} COMMAND test_fg_cp2077_mutant_${m})
    set_tests_properties(fg_cp2077_mutant_${m} PROPERTIES WILL_FAIL TRUE TIMEOUT 60)
  endforeach()
endif()
