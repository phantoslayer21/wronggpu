#!/bin/sh
# SPDX-License-Identifier: MIT
# Plain build without CMake (Linux, macOS, MinGW). Output in ./build-plain/.
# Usage: ./build.sh            build wronggpu-convert and the unit tests
#        ./build.sh test       also run the unit tests and the synthetic end-to-end tests
set -e
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
FLAGS="-std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math -Isrc"
mkdir -p build-plain
LIB="src/sha256.cpp src/json.cpp src/numeric.cpp src/map.cpp src/locate.cpp src/lz4block.cpp src/convert.cpp src/convert5.cpp src/ops5.cpp src/output.cpp src/pe_version.cpp src/api.cpp"
$CXX $FLAGS $LIB src/main.cpp -o build-plain/wronggpu-convert -pthread
$CXX $FLAGS $LIB tests/unit_tests.cpp -o build-plain/wgc-unit-tests -pthread
echo "built build-plain/wronggpu-convert and build-plain/wgc-unit-tests"
if [ "$1" = "test" ]; then
  build-plain/wgc-unit-tests --work build-plain/unit-work
  if [ -f ../../data/lz4-block-test.json ]; then
    build-plain/wgc-unit-tests --lz4-vectors ../../data/lz4-block-test.json
  fi
  python3 tests/run_tests.py --exe build-plain/wronggpu-convert --work build-plain/test-work
  python3 tests/run_tests5.py --exe build-plain/wronggpu-convert --unit build-plain/wgc-unit-tests --work build-plain/test-work5
fi
