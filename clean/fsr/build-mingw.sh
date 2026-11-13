#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Cross-builds the FSR front end for 64-bit Windows with MinGW-w64 and collects the files the spec team needs.
#   clean/fsr/build-mingw.sh [build-dir]        (default: <repo>/build/fsr-mingw)
# Needs: cmake, x86_64-w64-mingw32-g++-posix (Debian/Ubuntu package g++-mingw-w64-x86-64-posix).
# For development and the Wine tests only, never a release: MinGW-w64 links GCC's runtime (libstdc++, libgcc) and
# winpthreads statically into the DLLs, which is third-party code (README.md, owner's rule; QUESTIONS.md spec 20
# item 15). Releases and in-game tests use the MSVC build.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
build=${1:-$repo/build/fsr-mingw}
commit=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)
dirty=$(git -C "$repo" status --porcelain -- clean/fsr clean/third_party 2>/dev/null | head -c1)
id="mingw ${commit}${dirty:+-dirty}"
cmake -S "$here" -B "$build" -DCMAKE_TOOLCHAIN_FILE="$here/cmake/mingw-w64-x86_64.cmake" -DCMAKE_BUILD_TYPE=Release \
      -DWG_BUILD_ID="$id" >/dev/null
cmake --build "$build" -j"$(nproc)"
payload="$build/payload"
rm -rf "$payload"
mkdir -p "$payload"
cp "$build/amd_fidelityfx_dx12.dll" "$build/ffx_fsr3_x64.dll" "$build/ffx_fsr3upscaler_x64.dll" \
   "$here/install/install-wronggpu-fsr.ps1" "$here/TESTPLAN-20.md" "$payload/"
cp "$here/NOTICES.txt" "$payload/wronggpu-notices.txt"
(cd "$payload" && sha256sum * > SHA256SUMS.txt)
echo "built ($id, development build - not for release): $payload"
cat "$payload/SHA256SUMS.txt"
