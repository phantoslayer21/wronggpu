#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Offline tests of the runtime's D3D12 interop (spec 10 sections 1, 6, 8):
#   1. MinGW-w64 cross build of wronggpu_runtime.dll (core runtime + interop) and of the test program;
#   2. under Wine (vkd3d on a Vulkan driver such as Mesa lavapipe; an X display or xvfb-run): the D3D12 test with the
#      DLL's "d3d12-test" executor in place of the network (copy/convert passes, split protocol, failure paths).
# The MinGW build is for these tests only; releases are MSVC /MT builds (clean/runner/build.py --interop on Windows).
# Exit status 0 when everything that ran passed.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$here/../../.." && pwd)
work=${WG_TEST_DIR:-$repo/build/interop-tests}
mkdir -p "$work"
cxx=${MINGW_CXX:-x86_64-w64-mingw32-g++-posix}
if ! command -v "$cxx" >/dev/null; then echo "SKIP: $cxx not installed"; exit 0; fi

rt="$repo/clean/runtime"
flags=(-std=c++17 -O2 -Wall -Wextra -static -static-libgcc -static-libstdc++ -D_CRT_SECURE_NO_WARNINGS)
echo "== MinGW build"
"$cxx" "${flags[@]}" -shared -o "$work/wronggpu_runtime.dll" \
  "$rt/device.cpp" "$rt/embedded.cpp" "$rt/model.cpp" "$rt/runtime.cpp" "$repo/clean/k/runtime/model_k.cpp" \
  "$repo/clean/k/runtime/k_pack.cpp" "$repo/clean/k/runtime/model_k_load.cpp" \
  "$repo/clean/k/runtime/k_filter_table.cpp" "$repo/clean/k/runtime/runtime_k.cpp" "$repo/clean/converter/src/sha256.cpp" \
  "$repo/clean/converter/src/json.cpp" "$rt/device_dx12.cpp" "$rt/device_d3d12.cpp" "$here/convert.cpp" \
  "$here/executor_hip.cpp" "$here/executor_test.cpp" "$here/executor_d3d12.cpp" "$here/backend.cpp" \
  "$here/native_candidate.cpp" || exit 1
"$cxx" "${flags[@]}" -o "$work/test_interop_d3d12.exe" "$here/tests/test_interop_d3d12.cpp" || exit 1
x86_64-w64-mingw32-objdump -p "$work/wronggpu_runtime.dll" | grep -E "DLL Name|wgGetUpscalerBackend" | sed 's/^/  /'

wine=$(command -v wine64 || command -v wine || true)
if [ -z "$wine" ]; then echo "== Wine tests skipped (no Wine)"; exit 0; fi
echo "== Wine tests"
export WINEDEBUG=${WINEDEBUG:--all} WINEPREFIX=${WINEPREFIX:-$work/wineprefix}
run=("$wine")
if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then run=(xvfb-run -a "$wine"); fi
(cd "$work" && "${run[@]}" test_interop_d3d12.exe wronggpu_runtime.dll "$@" 2>&1 | tr -d "\r" |
  grep -v -E "X connection|XIO|requests \(") | tee "$work/result.txt"
grep -q "ALL TESTS PASSED\|^SKIP" "$work/result.txt"
