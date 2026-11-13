@rem SPDX-License-Identifier: MIT
@echo off
rem Plain build without CMake for Windows with MSVC. Run from a "x64 Native Tools Command Prompt
rem for VS". Output in .\build-plain\.  Usage: build.bat [test]
setlocal
cd /d "%~dp0"
if not exist build-plain mkdir build-plain
if not exist build-plain\obj mkdir build-plain\obj
set FLAGS=/nologo /std:c++17 /O2 /EHsc /W4 /permissive- /utf-8 /fp:precise /D_CRT_SECURE_NO_WARNINGS /Isrc /Fobuild-plain\obj\
set SRCS=src\sha256.cpp src\json.cpp src\numeric.cpp src\map.cpp src\locate.cpp src\lz4block.cpp src\convert.cpp src\convert5.cpp src\ops5.cpp src\output.cpp src\pe_version.cpp src\api.cpp
cl %FLAGS% %SRCS% src\main.cpp /Fe:build-plain\wronggpu-convert.exe || exit /b 1
cl %FLAGS% %SRCS% tests\unit_tests.cpp /Fe:build-plain\wgc-unit-tests.exe || exit /b 1
echo built build-plain\wronggpu-convert.exe and build-plain\wgc-unit-tests.exe
if /i "%1"=="test" (
  build-plain\wgc-unit-tests.exe --work build-plain\unit-work || exit /b 1
  if exist ..\..\data\lz4-block-test.json (
    build-plain\wgc-unit-tests.exe --lz4-vectors ..\..\data\lz4-block-test.json || exit /b 1
  )
  python tests\run_tests.py --exe build-plain\wronggpu-convert.exe --work build-plain\test-work || exit /b 1
  python tests\run_tests5.py --exe build-plain\wronggpu-convert.exe --unit build-plain\wgc-unit-tests.exe --work build-plain\test-work5 || exit /b 1
)
endlocal
