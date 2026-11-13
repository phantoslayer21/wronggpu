# SPDX-License-Identifier: MIT
# WrongGPU FSR front end (spec 20): amd_fidelityfx_dx12.dll replacement (integration point 1), the FSR 3.0 component
# forwarders ffx_fsr3_x64.dll and ffx_fsr3upscaler_x64.dll (integration point 2), and their tests.
#
# Windows, Visual Studio 2022 (x64), from a fresh clone:
#   cmake -S clean/fsr -B build/fsr -G "Visual Studio 17 2022" -A x64
#   cmake --build build/fsr --config Release
#   ctest --test-dir build/fsr -C Release --output-on-failure
# Linux, MinGW-w64 cross build (plus the host unit test and Wine tests): clean/fsr/build-mingw.sh
# Linux host unit test only: cmake -S clean/fsr -B build/host && cmake --build build/host && ctest --test-dir build/host
cmake_minimum_required(VERSION 3.16)
project(wronggpu_fsr LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(FFX_SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fidelityfx-sdk-1.1.4)
set(FFX_SDK30_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/fidelityfx-sdk-fsr3-v3.0.4)
set(WG_BUILD_ID "local build" CACHE STRING "Build identification shown in the log (e.g. the git commit)")
option(WG_HOSTED_DESKTOP_BADGE_QA "Private Windows-composed badge validation; release hold remains" OFF)
option(WG_SPLIT_PROGRESS_QA "Private read-only split fence diagnostics; no production enablement" OFF)
# Model K in the front end: its entry in the hosted version list, its id, model=K, its names in the overlay and the
# provider label. OFF (a release that ships Model M only): none of that; the list is "3.1.5 WrongGPU (Model M)" and
# AMD's versions, the Model K id is an unknown override like any other, model=K from wronggpu.ini or
# WG_RUNTIME_OPTIONS is dropped with one log line and Model M runs (src/model_k.h).
option(WG_MODEL_K "Offer Model K (hosted list entry, model=K); OFF = Model M only" ON)
if(WG_MODEL_K)
  add_compile_definitions(WRONGGPU_MODEL_K=1)
else()
  add_compile_definitions(WRONGGPU_MODEL_K=0)
endif()

enable_testing()
# clean/wgshared: the registry, the shared settings block and wg_settings_io.h (with their own tests).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../wgshared ${CMAKE_BINARY_DIR}/wgshared)
# The retired OptiScaler menu-patching plugin (asi/, WrongGPU.asi; not needed with OptiScaler 10's custom upscaler
# folder, spec 65.6): not built unless asked for. Its header-only unit tests still run.
option(WG_BUILD_ASI "Build the retired OptiScaler menu-patching plugin WrongGPU.asi" OFF)
add_executable(test_provider_label tests/test_provider_label.cpp)
if(MSVC)
  set_property(TARGET test_provider_label PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
add_test(NAME fsr_provider_label_unit COMMAND test_provider_label)

# Front-end logic without Windows or D3D12 (routing, forwarding, parameter mapping, log text, settings).
add_library(wg_fsr_core STATIC
  src/amd_compare.cpp
  src/amd_provider.cpp
  src/async_create.cpp
  src/config.cpp
  src/ffx_describe.cpp
  src/frontend.cpp
  src/fsr30_hook.cpp
  src/fsr30_map.cpp
  src/param_map.cpp)
target_include_directories(wg_fsr_core SYSTEM PUBLIC ${FFX_SDK_DIR})
target_include_directories(wg_fsr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(wg_fsr_core PUBLIC wg_shared_headers)
set_target_properties(wg_fsr_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  target_compile_options(wg_fsr_core PRIVATE /Zi)
  target_compile_options(wg_fsr_core PUBLIC /W4 $<$<COMPILE_LANGUAGE:CXX>:/permissive-> /utf-8 /wd4100)
  target_compile_definitions(wg_fsr_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(wg_fsr_core PUBLIC -Wall -Wextra)
endif()
if(MINGW)
  # Struct-returning COM methods (GetDesc, GetCPUDescriptorHandleForHeapStart) need the MSVC calling convention.
  target_compile_definitions(wg_fsr_core PUBLIC WIDL_EXPLICIT_AGGREGATE_RETURNS)
endif()

# Host unit tests (any platform).
add_executable(test_amd_provider tests/test_amd_provider.cpp)
target_link_libraries(test_amd_provider PRIVATE wg_fsr_core)
add_test(NAME fsr_amd_provider_unit COMMAND test_amd_provider)
add_executable(test_frontend tests/test_frontend.cpp)
target_link_libraries(test_frontend PRIVATE wg_fsr_core)
add_test(NAME fsr_frontend_unit COMMAND test_frontend)
add_executable(test_fsr30 tests/test_fsr30.cpp)
target_link_libraries(test_fsr30 PRIVATE wg_fsr_core)
add_test(NAME fsr30_hook_unit COMMAND test_fsr30)

# WrongGPU OptiScaler ASI plugin: thunk logic (any OS) and the offline anchor scan of an OptiScaler binary
# (any OS; skips when WRONGGPU_OPTISCALER_BIN is not set). Header-only resolver/thunk, no extra libraries.
add_executable(test_asi_thunk tests/test_asi_thunk.cpp)
add_test(NAME fsr_asi_thunk_unit COMMAND test_asi_thunk)
add_executable(test_asi_patch tests/test_asi_patch.cpp)
add_test(NAME fsr_asi_patch_unit COMMAND test_asi_patch)
add_executable(test_asi_lifetime tests/test_asi_lifetime.cpp)
add_test(NAME fsr_asi_lifetime_unit COMMAND test_asi_lifetime)
add_executable(test_asi_scan tests/test_asi_scan.cpp)
add_test(NAME fsr_asi_scan_unit COMMAND test_asi_scan)
set_tests_properties(fsr_asi_scan_unit PROPERTIES SKIP_RETURN_CODE 77)
if(MSVC)
  set_property(TARGET test_asi_thunk test_asi_scan test_asi_patch test_asi_lifetime PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(WIN32)
  enable_language(RC)
  if(MINGW)
    set(WG_STATIC_RUNTIME -static -static-libgcc -static-libstdc++)
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET wg_fsr_core test_frontend test_fsr30 PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
  if(MINGW)
    target_link_options(test_frontend PRIVATE ${WG_STATIC_RUNTIME})
    target_link_options(test_fsr30 PRIVATE ${WG_STATIC_RUNTIME})
    target_link_options(test_asi_thunk PRIVATE ${WG_STATIC_RUNTIME})
    target_link_options(test_asi_scan PRIVATE ${WG_STATIC_RUNTIME})
    target_link_options(test_asi_patch PRIVATE ${WG_STATIC_RUNTIME})
    target_link_options(test_asi_lifetime PRIVATE ${WG_STATIC_RUNTIME})
  endif()

  # Windows parts shared by the three DLLs: settings, loading AMD's originals, D3D12 queries, the stand-in.
  add_library(wg_fsr_win STATIC src/win_util.cpp src/standin_d3d12.cpp src/d3d12_hooks.cpp src/statistics_query.cpp src/split_progress.cpp src/overlay.cpp src/asi_bridge.cpp
              src/settings_bridge.cpp)
  if(WG_SPLIT_PROGRESS_QA)
    target_compile_definitions(wg_fsr_win PRIVATE WRONGGPU_SPLIT_PROGRESS_QA=1)
  endif()
  target_include_directories(wg_fsr_win PUBLIC src)
  target_link_libraries(wg_fsr_win PUBLIC wg_fsr_core d3d12 d3dcompiler advapi32)
  if(MSVC)
    target_compile_options(wg_fsr_win PRIVATE /Zi)
  endif()
  set_target_properties(wg_fsr_win PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  add_executable(test_vtable_clone tests/test_vtable_clone.cpp)
  target_link_libraries(test_vtable_clone PRIVATE wg_fsr_win d3d12 dxgi d3dcompiler user32)
  set_target_properties(test_vtable_clone PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_test(NAME fsr_vtable_clone COMMAND test_vtable_clone)
  add_executable(test_vtable_growth tests/test_vtable_growth.cpp)
  target_link_libraries(test_vtable_growth PRIVATE wg_fsr_win d3d12 dxgi d3dcompiler user32)
  set_target_properties(test_vtable_growth PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_test(NAME fsr_vtable_growth COMMAND test_vtable_growth)
  # A list with another hook layer's own vtable is never patched or split (fake COM objects, no device).
  add_executable(test_wrapped_list tests/test_wrapped_list.cpp)
  target_link_libraries(test_wrapped_list PRIVATE wg_fsr_win d3d12 dxgi d3dcompiler user32)
  set_target_properties(test_wrapped_list PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # case F: another layer's module holding the class vtables of its wrapper objects (loaded by the test from its folder)
  add_library(test_wrapper_layer SHARED tests/wrapper_layer_dll.cpp)
  set_target_properties(test_wrapper_layer PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_dependencies(test_wrapped_list test_wrapper_layer)
  add_test(NAME fsr_wrapped_list_unit COMMAND test_wrapped_list WORKING_DIRECTORY $<TARGET_FILE_DIR:test_wrapped_list>)
  add_executable(test_statistics_integer tests/test_statistics_integer.cpp)
  target_include_directories(test_statistics_integer SYSTEM PRIVATE ${FFX_SDK_DIR})
  target_link_libraries(test_statistics_integer PRIVATE d3d12 d3dcompiler dxguid)
  set_target_properties(test_statistics_integer PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_test(NAME fsr_statistics_integer COMMAND test_statistics_integer)
  add_executable(test_split_progress tests/test_split_progress.cpp src/split_progress.cpp)
  target_link_libraries(test_split_progress PRIVATE d3d12 dxgi)
  set_target_properties(test_split_progress PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(test_split_progress PRIVATE NOMINMAX)
  add_test(NAME fsr_split_progress COMMAND test_split_progress)
  set_tests_properties(fsr_statistics_integer PROPERTIES SKIP_RETURN_CODE 77)
  target_sources(wg_fsr_win PRIVATE src/hosted_badge_win.cpp)
  add_executable(test_hosted_badge tests/test_hosted_badge.cpp)
  target_link_libraries(test_hosted_badge PRIVATE wg_fsr_win user32 gdi32 dxgi)
  set_target_properties(test_hosted_badge PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_test(NAME fsr_hosted_badge_unit COMMAND test_hosted_badge)
  add_executable(test_diagnostics_win tests/test_diagnostics_win.cpp)
  add_library(fake_diagnostic_hip SHARED tests/fake_diagnostic_hip.cpp)
  set_target_properties(fake_diagnostic_hip PROPERTIES PREFIX "" OUTPUT_NAME amdhip64_7
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/diagnostictest/$<CONFIG>"
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set_target_properties(test_diagnostics_win PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/diagnostictest/$<CONFIG>")
  add_dependencies(test_diagnostics_win fake_diagnostic_hip)
  target_link_libraries(test_diagnostics_win PRIVATE advapi32)
  target_include_directories(test_diagnostics_win PRIVATE src)
  set_target_properties(test_diagnostics_win PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_test(NAME fsr_diagnostics_win COMMAND test_diagnostics_win)

  # wg_add_dll(<target/file name> <sources...>): one of WrongGPU's replacement DLLs.
  function(wg_add_dll name)
    add_library(${name} SHARED ${ARGN})
    set_target_properties(${name} PROPERTIES PREFIX "" OUTPUT_NAME ${name} MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    target_compile_definitions(${name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:WRONGGPU_FSR_BUILD="${WG_BUILD_ID}">)
    if(WG_HOSTED_DESKTOP_BADGE_QA AND name STREQUAL "amd_fidelityfx_dx12")
      target_compile_definitions(${name} PRIVATE WRONGGPU_HOSTED_DESKTOP_BADGE_QA=1)
    endif()
    target_link_libraries(${name} PRIVATE wg_fsr_win dxguid)
    if(MSVC)
      target_compile_options(${name} PRIVATE /Zi)
      target_link_options(${name} PRIVATE /DEBUG:FULL /INCREMENTAL:NO)
    endif()
    if(MINGW)
      target_link_options(${name} PRIVATE ${WG_STATIC_RUNTIME})
    endif()
    # The release's notices file (licence of the FidelityFX headers used as declarations) goes next to the DLLs.
    add_custom_command(TARGET ${name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/NOTICES.txt $<TARGET_FILE_DIR:${name}>/wronggpu-notices.txt)
  endfunction()

  # Integration point 1: the replacement amd_fidelityfx_dx12.dll.
  wg_add_dll(amd_fidelityfx_dx12 src/dllmain.cpp src/exports.def src/version.rc)

  # Integration point 2: forwarders for the FSR 3.0 component DLLs.
  wg_add_dll(ffx_fsr3upscaler_x64 src/dll_fsr3upscaler.cpp src/exports_fsr3upscaler.def src/version_fsr3upscaler.rc)
  wg_add_dll(ffx_fsr3_x64 src/dll_fsr3.cpp src/exports_fsr3.def src/version_fsr3.rc)

  # The SR front end as OptiScaler 10's custom upscaler (spec 65.1, DESIGN-i15 8.1): the same DLL under the FFX DX12
  # upscaler file name in custom\WrongGPU\ffx\. Built into <build>/package (target wg_package, part of ALL) and
  # installed by `cmake --install <build> --prefix <dir>` into <dir>/package/custom/WrongGPU/ffx/.
  set(WG_FFX_SLOT_FILE amd_fidelityfx_upscaler_dx12.dll CACHE STRING "File name of the FFX DX12 upscaler slot in OptiScaler's custom folder")
  set(WG_PACKAGE_SLOT ${CMAKE_BINARY_DIR}/package/custom/WrongGPU/ffx)
  add_custom_target(wg_package ALL
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WG_PACKAGE_SLOT}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:amd_fidelityfx_dx12> ${WG_PACKAGE_SLOT}/${WG_FFX_SLOT_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/NOTICES.txt ${WG_PACKAGE_SLOT}/wronggpu-notices.txt
      COMMENT "custom upscaler package: package/custom/WrongGPU/ffx/${WG_FFX_SLOT_FILE}")
  add_dependencies(wg_package amd_fidelityfx_dx12)
  install(FILES $<TARGET_FILE:amd_fidelityfx_dx12> DESTINATION package/custom/WrongGPU/ffx RENAME ${WG_FFX_SLOT_FILE})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NOTICES.txt DESTINATION package/custom/WrongGPU/ffx RENAME wronggpu-notices.txt)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../wgshared/ini/wronggpu.default.ini DESTINATION package/WrongGPU)

  if(WG_BUILD_ASI)
  # WrongGPU OptiScaler ASI plugin (WrongGPU.asi): our own code only, no third-party hooking library, no OptiScaler
  # code. Loaded by OptiScaler; adds WrongGPU to its main upscaler menu and answers the front end's model query.
  # RETIRED (OptiScaler 10 custom upscaler folder); built only with -DWG_BUILD_ASI=ON.
  add_library(wronggpu_asi SHARED asi/wronggpu_asi.cpp asi/exports_asi.def asi/version_asi.rc)
  set_target_properties(wronggpu_asi PROPERTIES PREFIX "" OUTPUT_NAME WrongGPU SUFFIX ".asi"
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_include_directories(wronggpu_asi PRIVATE src)
  target_compile_definitions(wronggpu_asi PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
  if(MSVC)
    target_compile_options(wronggpu_asi PRIVATE /W4 /wd4100 /Zi)
    target_link_options(wronggpu_asi PRIVATE /DEBUG:FULL /INCREMENTAL:NO)
  endif()
  if(MINGW)
    target_link_options(wronggpu_asi PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  # The plugin's module identification in a process (asi/opti_module.h), with the built plugin: an OptiScaler binary
  # (WRONGGPU_OPTISCALER_BIN; unset: SKIP) mapped as dxgi.dll without running it -> hooked; only the system dxgi.dll
  # -> not hooked. Either way the plugin never scans its own image.
  add_executable(test_asi_module tests/test_asi_module.cpp)
  set_target_properties(test_asi_module PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_dependencies(test_asi_module wronggpu_asi)
  add_test(NAME fsr_asi_module_system COMMAND test_asi_module $<TARGET_FILE:wronggpu_asi> system)
  add_test(NAME fsr_asi_module_opti COMMAND test_asi_module $<TARGET_FILE:wronggpu_asi> opti)
  set_tests_properties(fsr_asi_module_opti PROPERTIES SKIP_RETURN_CODE 77)
  if(MINGW)
    target_link_options(test_asi_module PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  # Own-DLL query test only: never invokes InitializeASI or opens a foreign host.
  add_executable(test_asi_status tests/test_asi_status.cpp)
  set_target_properties(test_asi_status PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_dependencies(test_asi_status wronggpu_asi)
  add_test(NAME fsr_asi_status_unit COMMAND test_asi_status $<TARGET_FILE:wronggpu_asi>)
  if(MINGW)
    target_link_options(test_asi_status PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  endif()  # WG_BUILD_ASI


  # Compile-time checks of WrongGPU's mirrors against AMD's headers that contain code (ffx_api_dx12.h: inline helpers;
  # the fsr3-v3.0.4 host headers: a static variable), and of the D3D12 vtable positions the hooks patch against the
  # SDK's C declarations (d3d12_vtbl_check.c). Compiled with every build, linked into nothing, so no code from AMD's
  # headers can reach a release binary (README.md, owner's rule). The DLLs see only the four declaration-only
  # FidelityFX API headers; tests/audit-release.sh checks both.
  add_library(wg_sdk_layout_check OBJECT src/ffx_api_layout_check.cpp src/fsr30_layout_check.cpp src/d3d12_vtbl_check.c)
  target_link_libraries(wg_sdk_layout_check PRIVATE wg_fsr_core)
  target_include_directories(wg_sdk_layout_check PRIVATE src)
  set_source_files_properties(src/fsr30_layout_check.cpp PROPERTIES
                              INCLUDE_DIRECTORIES "${FFX_SDK30_DIR};${CMAKE_CURRENT_SOURCE_DIR}/src/sdk30_stub")
  if(MSVC)
    # The Windows SDK's C declarations are not /W4-clean in C (nameless struct/union, type definitions in parameter
    # lists, non-int bit fields).
    set_source_files_properties(src/d3d12_vtbl_check.c PROPERTIES COMPILE_OPTIONS "/wd4201;/wd4115;/wd4214")
    set_source_files_properties(src/fsr30_layout_check.cpp PROPERTIES COMPILE_OPTIONS "/wd4365;/wd4505")
  else()
    set_source_files_properties(src/fsr30_layout_check.cpp PROPERTIES COMPILE_OPTIONS "-Wno-unknown-pragmas;-Wno-unused-variable")
  endif()

  # Test doubles and the D3D12 end-to-end test (runs on Windows, or under Wine with a Vulkan driver).
  add_library(fake_amd_fidelityfx SHARED tests/fake_original.cpp tests/fake_original.def)
  set_target_properties(fake_amd_fidelityfx PROPERTIES PREFIX "" OUTPUT_NAME amd_fidelityfx_dx12.wronggpu-original
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_include_directories(fake_amd_fidelityfx SYSTEM PRIVATE ${FFX_SDK_DIR})
  add_executable(test_hosted_dll tests/test_hosted_dll.cpp)
  target_link_libraries(test_hosted_dll PRIVATE wg_fsr_core)
  set_target_properties(test_hosted_dll PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hostedtest/$<CONFIG>"
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_dependencies(test_hosted_dll amd_fidelityfx_dx12 fake_amd_fidelityfx)
  add_custom_command(TARGET test_hosted_dll POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:amd_fidelityfx_dx12> $<TARGET_FILE_DIR:test_hosted_dll>/wronggpu_ffx_dx12.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:fake_amd_fidelityfx> $<TARGET_FILE_DIR:test_hosted_dll>/wronggpu-amd-compare.dll)
  add_test(NAME fsr_hosted_dll COMMAND test_hosted_dll WORKING_DIRECTORY $<TARGET_FILE_DIR:test_hosted_dll>)
  # Hosted with AMD's versions from the game's own amd_fidelityfx_dx12.dll only (folder game), and with no AMD DLL at
  # all (folder none): the test, WrongGPU under its hosted name and, in game, the test double under the game's name.
  add_executable(test_hosted_provider tests/test_hosted_provider.cpp)
  target_link_libraries(test_hosted_provider PRIVATE wg_fsr_win)
  set(WG_PROVIDER_GAME ${CMAKE_BINARY_DIR}/hostedprovider/game/$<CONFIG>)
  set(WG_PROVIDER_NONE ${CMAKE_BINARY_DIR}/hostedprovider/none/$<CONFIG>)
  set_target_properties(test_hosted_provider PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hostedprovider/game/$<CONFIG>"
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_dependencies(test_hosted_provider amd_fidelityfx_dx12 fake_amd_fidelityfx)
  add_custom_command(TARGET test_hosted_provider POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:amd_fidelityfx_dx12> ${WG_PROVIDER_GAME}/wronggpu_ffx_dx12.dll
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:fake_amd_fidelityfx> ${WG_PROVIDER_GAME}/amd_fidelityfx_dx12.dll
      COMMAND ${CMAKE_COMMAND} -E make_directory ${WG_PROVIDER_NONE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test_hosted_provider> ${WG_PROVIDER_NONE}/
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:amd_fidelityfx_dx12> ${WG_PROVIDER_NONE}/wronggpu_ffx_dx12.dll)
  add_test(NAME fsr_hosted_provider_game COMMAND test_hosted_provider game WORKING_DIRECTORY ${WG_PROVIDER_GAME})
  add_test(NAME fsr_hosted_provider_none COMMAND ${WG_PROVIDER_NONE}/test_hosted_provider.exe none WORKING_DIRECTORY ${WG_PROVIDER_NONE})
  if(MINGW)
    target_link_options(fake_amd_fidelityfx PRIVATE ${WG_STATIC_RUNTIME})
  endif()

  # Overlay test: draws through the hooked Present on swap chains of every supported format and reads them back.
  add_executable(test_overlay tests/test_overlay.cpp)
  target_link_libraries(test_overlay PRIVATE wg_fsr_win d3d12 dxgi dxguid)
  set_target_properties(test_overlay PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MINGW)
    target_compile_definitions(test_overlay PRIVATE WIDL_EXPLICIT_AGGREGATE_RETURNS)
    target_link_options(test_overlay PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  add_test(NAME fsr_overlay COMMAND test_overlay WORKING_DIRECTORY $<TARGET_FILE_DIR:test_overlay>)

  # Test binaries go to their own folder (dlltest/<config>) with a copy of the front end, so the test double never
  # sits next to the real payload.
  set(WG_TEST_BIN ${CMAKE_BINARY_DIR}/dlltest/$<CONFIG>)
  set_target_properties(fake_amd_fidelityfx PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${WG_TEST_BIN} LIBRARY_OUTPUT_DIRECTORY ${WG_TEST_BIN})
  add_executable(test_dll_d3d12 tests/test_dll_d3d12.cpp)
  set_target_properties(test_dll_d3d12 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${WG_TEST_BIN})
  foreach(variant fsr3upscaler fsr3)
    if(variant STREQUAL "fsr3upscaler")
      set(fake_name ffx_fsr3upscaler_x64.wronggpu-original)
      set(fake_define FAKE_UPSCALER)
    else()
      set(fake_name ffx_fsr3_x64.wronggpu-original)
      set(fake_define FAKE_FSR3)
    endif()
    add_library(fake_${variant} SHARED tests/fake_fsr30.cpp tests/fake_${variant}.def)
    target_compile_definitions(fake_${variant} PRIVATE ${fake_define})
    if(variant STREQUAL "fsr3")
      target_link_libraries(fake_${variant} PRIVATE ffx_fsr3upscaler_x64)  # imports WrongGPU's forwarder by name
    endif()
    set_target_properties(fake_${variant} PROPERTIES PREFIX "" OUTPUT_NAME ${fake_name}
                          RUNTIME_OUTPUT_DIRECTORY ${WG_TEST_BIN} LIBRARY_OUTPUT_DIRECTORY ${WG_TEST_BIN}
                          MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    if(MINGW)
      target_link_options(fake_${variant} PRIVATE ${WG_STATIC_RUNTIME})
    endif()
  endforeach()
  # Test double of the network runtime (upscaler = network), named like the real one, in the test folder only.
  add_library(fake_runtime SHARED tests/fake_runtime.cpp)
  set_target_properties(fake_runtime PROPERTIES PREFIX "" OUTPUT_NAME wronggpu_runtime
                        RUNTIME_OUTPUT_DIRECTORY ${WG_TEST_BIN} LIBRARY_OUTPUT_DIRECTORY ${WG_TEST_BIN}
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MINGW)
    target_link_options(fake_runtime PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  add_custom_command(TARGET test_dll_d3d12 POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:amd_fidelityfx_dx12>
                             $<TARGET_FILE:ffx_fsr3upscaler_x64> $<TARGET_FILE:ffx_fsr3_x64> ${WG_TEST_BIN}/)
  target_include_directories(test_dll_d3d12 SYSTEM PRIVATE ${FFX_SDK_DIR})
  set_target_properties(test_dll_d3d12 PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MINGW)
    target_compile_definitions(test_dll_d3d12 PRIVATE WIDL_EXPLICIT_AGGREGATE_RETURNS)
    target_link_options(test_dll_d3d12 PRIVATE ${WG_STATIC_RUNTIME})
  endif()
  target_link_libraries(test_dll_d3d12 PRIVATE dxguid)

  # The custom upscaler package end to end (DESIGN-i15 10.1): our own OptiScaler-style loader (tests/fake_opti_loader.*)
  # loads the built package's DLL from custom\<folder>\ffx\ in a game folder the test lays out in its own folder
  # (customslot/<config>), on WARP.
  add_executable(test_custom_slot tests/test_custom_slot.cpp tests/fake_opti_loader.cpp)
  target_link_libraries(test_custom_slot PRIVATE wg_fsr_core d3d12 dxgi dxguid)
  set_target_properties(test_custom_slot PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/customslot/$<CONFIG>"
                        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(test_custom_slot PRIVATE NOMINMAX UNICODE _UNICODE)
  add_dependencies(test_custom_slot wg_package fake_amd_fidelityfx fake_runtime)
  if(MINGW)
    target_link_options(test_custom_slot PRIVATE ${WG_STATIC_RUNTIME} -municode)
  endif()
  add_test(NAME fsr_custom_slot COMMAND test_custom_slot ${WG_PACKAGE_SLOT}/${WG_FFX_SLOT_FILE}
           $<TARGET_FILE:fake_amd_fidelityfx> $<TARGET_FILE:fake_runtime>
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/customslot/$<CONFIG>)
  set_tests_properties(fsr_custom_slot PROPERTIES SKIP_RETURN_CODE 77)
  add_dependencies(test_dll_d3d12 amd_fidelityfx_dx12 ffx_fsr3upscaler_x64 ffx_fsr3_x64 fake_amd_fidelityfx fake_fsr3upscaler fake_fsr3
                   fake_runtime)
  if(NOT CMAKE_CROSSCOMPILING)
    foreach(mode upscale upscale_mips lanczos warpdiff motionsign forward amd view_motion view_depth view_jitter srgb
                 fsr30_log fsr30_upscaler fsr30_upscaler_shared fsr30_fsr3_inner fsr30_fsr3_static
                 fsr30_fsr3_upscaling_only hooks_trace split_standin split_fsr30 split_release split_control split_cyberpunk hooks_trace_fsr3
                 network_missing network_create_fail network_dispatch_fail network_split network_split_stats network_split_stats_nested network_split_stats_twice network_v1 network_bad
                 network_cyberpunk_fail network_park)
      add_test(NAME fsr_dll_d3d12_${mode} COMMAND test_dll_d3d12 ${mode}
               WORKING_DIRECTORY $<TARGET_FILE_DIR:test_dll_d3d12>)
      set_tests_properties(fsr_dll_d3d12_${mode} PROPERTIES SKIP_RETURN_CODE 77)  # no D3D12 device
    endforeach()
  endif()
endif()
