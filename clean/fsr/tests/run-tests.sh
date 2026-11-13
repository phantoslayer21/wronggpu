#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Runs every offline test of the FSR front end that this machine can run:
#   1. the D3D12-free unit test on the host (with AddressSanitizer and UBSan when the compiler has them);
#   2. the MinGW-w64 cross build and the release audit (tests/audit-release.sh);
#   3. under Wine (if installed): the unit test as a Windows program and the end-to-end DLL tests, which need a
#      D3D12 device (Wine's vkd3d on a Vulkan driver, e.g. Mesa lavapipe; an X display or xvfb-run).
# Exit status 0 when everything that ran passed.
set -uo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$here/../.." && pwd)
work=${WG_TEST_DIR:-$repo/build/fsr-tests}
mkdir -p "$work"
failed=0

echo "== host unit tests"
cxx=${CXX:-g++}
san="-fsanitize=address,undefined"
echo 'int main(){}' > "$work/san.cpp"
$cxx $san "$work/san.cpp" -o "$work/san" 2>/dev/null || san=""
core="$here/src/frontend.cpp $here/src/param_map.cpp $here/src/ffx_describe.cpp $here/src/config.cpp $here/src/fsr30_hook.cpp $here/src/fsr30_map.cpp $here/src/amd_compare.cpp $here/src/async_create.cpp"
for t in test_frontend test_fsr30; do
  if $cxx -std=c++17 -Wall -Wextra -O1 -g $san -I"$here/../third_party/fidelityfx-sdk-1.1.4" \
        "$here/tests/$t.cpp" $core -o "$work/$t"; then
    "$work/$t" || failed=1
  else
    failed=1
  fi
done

echo "== MinGW-w64 build"
if command -v x86_64-w64-mingw32-g++-posix >/dev/null; then
  "$here/build-mingw.sh" "$work/mingw" >/dev/null || failed=1
  echo "== release audit (owner's rule: no third-party code in the DLLs)"
  "$here/tests/audit-release.sh" "$work/mingw" || failed=1
else
  echo "skipped: x86_64-w64-mingw32-g++-posix not installed"
fi

wine=$(command -v wine64 || command -v wine || ls /usr/lib/wine/wine64 2>/dev/null || true)
tbin="$work/mingw/dlltest/Release"
if [ -n "$wine" ] && [ -f "$tbin/test_dll_d3d12.exe" ]; then
  echo "== Wine tests"
  export WINEDEBUG=${WINEDEBUG:--all} WINEPREFIX=${WINEPREFIX:-$work/wineprefix}
  run=("$wine")
  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null; then run=(xvfb-run -a "$wine"); fi
  for t in test_frontend test_fsr30; do
    (cd "$work/mingw" && "${run[@]}" $t.exe 2>&1 | tr -d "\r" | grep -E "checks|FAIL") || failed=1
  done
  for mode in forward amd upscale lanczos warpdiff motionsign view_motion view_depth view_jitter srgb \
              fsr30_log fsr30_upscaler fsr30_upscaler_shared fsr30_fsr3_inner fsr30_fsr3_static fsr30_fsr3_upscaling_only \
              hooks_trace split_standin split_fsr30 split_release split_control split_cyberpunk hooks_trace_fsr3 \
              network_missing network_create_fail network_dispatch_fail network_split network_v1 network_bad \
              network_cyberpunk_fail; do
    out=$(cd "$tbin" && "${run[@]}" test_dll_d3d12.exe "$mode" 2>&1 | tr -d "\r" | grep -v -E "X connection|XIO|requests \(")
    status=$(printf '%s\n' "$out" | grep -E "^$mode: (passed|FAILED)|^SKIP" | tail -1)
    printf '%s\n' "$out" | grep -E "max |mean |worst " || true
    echo "${status:-$mode: no result}"
    case "$status" in *passed|SKIP*) ;; *) failed=1; printf '%s\n' "$out" | tail -40 ;; esac
  done
else
  echo "== Wine tests skipped (no Wine or no MinGW build)"
fi

pwsh=$(command -v pwsh || command -v powershell || true)
if [ -n "$pwsh" ]; then
  echo "== install script test (mock game folders)"
  "$pwsh" -NoProfile -File "$here/tests/test-install.ps1" || failed=1
else
  echo "== install script test skipped (no PowerShell)"
fi

[ $failed -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $failed
