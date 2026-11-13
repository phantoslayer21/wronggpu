#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Checks a MinGW-w64 build of the FSR front end against the owner's release rule (README.md): no third-party code in
# WrongGPU's DLLs; AMD's FidelityFX headers only as declarations.
#   clean/fsr/tests/audit-release.sh <build-dir>      (a build made by build-mingw.sh)
# Checks:
#   1. each AMD header the DLLs may include defines nothing (compiled alone, keeping inline functions and constants);
#   2. the DLLs' own objects include no other AMD header (from the compiler's dependency files);
#   3. the layout-check objects (which do include AMD headers with code) are not linked into any DLL;
#   4. the DLLs import only Windows system DLLs.
# It also reports what the MinGW toolchain links in statically (GCC runtime, winpthreads); that is why the MinGW build
# is for development only (QUESTIONS.md, spec 20 item 15).
set -uo pipefail
# The link line of a target, with the response files that list its objects.
linkinfo() { cat "$build/CMakeFiles/$1.dir/link.txt" "$build/CMakeFiles/$1.dir"/objects*.rsp 2>/dev/null; }
here=$(cd "$(dirname "$0")/.." && pwd)
build=${1:?usage: audit-release.sh <build-dir>}
sdk="$here/../third_party/fidelityfx-sdk-1.1.4"
cxx=${MINGW_CXX:-x86_64-w64-mingw32-g++-posix}
dlls="amd_fidelityfx_dx12 ffx_fsr3_x64 ffx_fsr3upscaler_x64"
allowed="ffx_api/ffx_api.h ffx_api/ffx_api_types.h ffx_api/ffx_framegeneration.h ffx_api/ffx_upscale.h"
failed=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "1. AMD headers allowed in the DLLs define no code or data"
for h in $allowed; do
  printf '#include "%s"\n' "$h" > "$tmp/h.cpp"
  if ! $cxx -std=c++17 -O0 -fkeep-inline-functions -fkeep-static-consts -I"$sdk" -c "$tmp/h.cpp" -o "$tmp/h.o"; then
    echo "   FAIL: $h does not compile alone"; failed=1; continue
  fi
  defs=$(x86_64-w64-mingw32-nm --defined-only "$tmp/h.o" | grep -v -E ' [tTdDrRbB] \.|\.file|@feat' || true)
  if [ -n "$defs" ]; then echo "   FAIL: $h defines:"; echo "$defs" | sed 's/^/     /'; failed=1; else echo "   ok   $h"; fi
done

echo "2. the DLLs' objects include no other AMD header"
# Only the objects each target is linked from now (a reused build folder may hold stale ones).
deps=""
for t in $dlls wg_fsr_core wg_fsr_win; do
  for o in $(linkinfo "$t" | grep -o -E "CMakeFiles/$t\.dir/[^ ]+\.obj" | sort -u); do
    deps="$deps $build/$o.d"
  done
done
deps=$(printf '%s\n' $deps)
if [ -z "$deps" ]; then echo "   FAIL: no dependency files under $build/CMakeFiles"; failed=1; fi
for d in $deps; do
  [ -f "$d" ] || { echo "   FAIL: $d missing"; failed=1; continue; }
  for f in $(tr -s ' \\\n' '\n' < "$d" | grep 'third_party/' || true); do
    rel=${f#*third_party/fidelityfx-sdk-1.1.4/}
    case " $allowed " in
      *" $rel "*) ;;
      *) echo "   FAIL: ${d#$build/} includes $f"; failed=1 ;;
    esac
  done
done
n=$(echo "$deps" | wc -l)
if [ "$n" -lt 11 ]; then echo "   FAIL: only $n objects found (expected at least 11)"; failed=1; fi
[ "$failed" = 0 ] && echo "   ok   $n objects"

echo "3. layout checks are linked into no DLL"
bad=0
for t in $dlls; do
  if linkinfo "$t" | grep -q -E 'layout_check'; then
    echo "   FAIL: $t links a layout check"; bad=1
  fi
done
for lib in "$build"/libwg_fsr_core.a "$build"/libwg_fsr_win.a; do
  if [ -f "$lib" ] && x86_64-w64-mingw32-ar t "$lib" | grep -q layout_check; then echo "   FAIL: $lib holds a layout check"; bad=1; fi
done
[ "$bad" = 0 ] && echo "   ok" || failed=1

echo "4. imports: Windows system DLLs only"
system="kernel32.dll msvcrt.dll user32.dll advapi32.dll ole32.dll version.dll d3d12.dll dxgi.dll"
for t in $dlls; do
  f="$build/$t.dll"
  if [ ! -f "$f" ]; then echo "   FAIL: $f missing"; failed=1; continue; fi
  imports=$(x86_64-w64-mingw32-objdump -p "$f" | sed -n 's/^\tDLL Name: //p' | tr 'A-Z' 'a-z' | tr '\n' ' ')
  for i in $imports; do
    case " $system " in *" $i "*) ;; *) echo "   FAIL: $t.dll imports $i"; failed=1 ;; esac
  done
  echo "   $t.dll: $imports"
done

echo "note: statically linked by the MinGW-w64 toolchain (why this build is not a release):"
for t in $dlls; do
  marks=$(strings -a "$build/$t.dll" | grep -o -E 'GCC: \(GNU\) [^ ]+|winpthreads|Mingw-w64 runtime' | sort -u | tr '\n' ';')
  echo "   $t.dll: ${marks:-none found}"
done

[ "$failed" = 0 ] && echo "audit-release: passed" || echo "audit-release: FAILED"
exit $failed
