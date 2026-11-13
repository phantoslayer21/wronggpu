# SPDX-License-Identifier: MIT
# WrongGPU shared pieces (clean/wgshared): tests of the registry generator and of the shared settings block.
# Stand-alone:   cmake -S clean/wgshared -B build/wgshared && cmake --build build/wgshared --config Release
#                ctest --test-dir build/wgshared -C Release --output-on-failure
# Also pulled into the SR front end's build (clean/fsr/CMakeLists.txt adds this folder), so its ctest runs these too.
cmake_minimum_required(VERSION 3.16)
if(NOT DEFINED PROJECT_NAME)
  project(wronggpu_wgshared LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  enable_testing()
endif()

add_library(wg_shared_headers INTERFACE)
# Extension proof builds only: a folder with generated headers of an extended registry (gen_registry.py --extend X
# --out-dir D; pass D/include) found before the committed ones. Never set for a release.
set(WG_REGISTRY_INCLUDE "" CACHE PATH "Test only: include folder of an extended registry (gen_registry.py --extend)")
if(WG_REGISTRY_INCLUDE)
  message(STATUS "wgshared: EXTENDED registry headers from ${WG_REGISTRY_INCLUDE} (test build)")
  target_include_directories(wg_shared_headers BEFORE INTERFACE ${WG_REGISTRY_INCLUDE})
endif()
target_include_directories(wg_shared_headers INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)

if(WIN32)
  add_executable(test_shared_settings ${CMAKE_CURRENT_LIST_DIR}/tests/test_shared_settings.cpp)
  target_link_libraries(test_shared_settings PRIVATE wg_shared_headers)
  set_target_properties(test_shared_settings PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MSVC)
    target_compile_options(test_shared_settings PRIVATE /W4 /WX /permissive- /utf-8)
  else()
    target_compile_options(test_shared_settings PRIVATE -Wall -Wextra)
    target_link_options(test_shared_settings PRIVATE -static)
  endif()
  add_test(NAME wgshared_settings_block COMMAND test_shared_settings)

  # The SR side's implementation against the RR side's independent reader (RR plugin / RR runtime), in one process.
  add_executable(test_cross_reader ${CMAKE_CURRENT_LIST_DIR}/tests/test_cross_reader.cpp)
  target_include_directories(test_cross_reader PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../rr_f/proxy/include)
  target_link_libraries(test_cross_reader PRIVATE wg_shared_headers)
  set_target_properties(test_cross_reader PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  if(MSVC)
    target_compile_options(test_cross_reader PRIVATE /W4 /WX /permissive- /utf-8)
  else()
    target_compile_options(test_cross_reader PRIVATE -Wall -Wextra)
    target_link_options(test_cross_reader PRIVATE -static)
  endif()
  add_test(NAME wgshared_cross_reader COMMAND test_cross_reader)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  foreach(t test_gen_registry test_dummy_extension)
    add_test(NAME wgshared_${t} COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/tests/${t}.py)
  endforeach()
endif()
