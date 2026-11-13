#!/bin/sh
# SPDX-License-Identifier: MIT
# Cross-builds the launcher with MinGW-w64 (Linux): WrongGPU-Restore.exe (the per-game restorer), then WrongGPU.exe
# with the restorer and the payload embedded. Usage:
#   ./build-mingw.sh <payload dir> [map.json] [version] [out dir]
# Default map: data/preset-M.v1.map.json; default out dir: build-mingw/.
set -e
cd "$(dirname "$0")"
PAYLOAD="${1:?payload folder}"
MAP="${2:-../../data/preset-M.v1.map.json}"
VERSION="${3:-0.3.0}"
OUT="${4:-build-mingw}"
CXX="${CXX:-x86_64-w64-mingw32-g++}"
WINDRES="${WINDRES:-x86_64-w64-mingw32-windres}"
GEN="$OUT/gen"
mkdir -p "$GEN"
python3 tools/make_payload.py --payload "$PAYLOAD" --map "$MAP" --version "$VERSION" --manifest src/setup.manifest --out "$GEN"
"$WINDRES" -O coff -i "$GEN/resources.rc" -o "$GEN/resources.o"
CONV=../converter/src
CONV_SRC="$CONV/sha256.cpp $CONV/json.cpp $CONV/numeric.cpp $CONV/map.cpp $CONV/locate.cpp $CONV/lz4block.cpp $CONV/convert.cpp $CONV/output.cpp $CONV/pe_version.cpp $CONV/api.cpp"
FLAGS="-std=c++17 -O2 -Wall -Wextra -ffp-contract=off -municode -mwindows -static -I$GEN -I$CONV -Isrc"
LIBS="-lcomctl32 -lshell32 -lole32 -luuid -ladvapi32 -luser32 -lgdi32 -lgdiplus -ldwmapi -luxtheme -lwinhttp"
# the restorer: wg_core without payload or UI
printf '1 ICON "app.ico"\n' > "$GEN/restore_app.rc"
"$WINDRES" -O coff -I"$GEN" -i "$GEN/restore_app.rc" -o "$GEN/restore_app.o"
$CXX $FLAGS "-DWRONGGPU_VERSION_STR=L\"$VERSION\"" src/app/restore_main.cpp src/wg_core/*.cpp $CONV_SRC "$GEN/restore_app.o" $LIBS -o "$OUT/WrongGPU-Restore.exe"
cp "$OUT/WrongGPU-Restore.exe" "$GEN/restorer.bin"
printf '960 RCDATA "restorer.bin"\n' > "$GEN/restore.rc"
"$WINDRES" -O coff -I"$GEN" -i "$GEN/restore.rc" -o "$GEN/restore.o"
# the launcher
$CXX $FLAGS src/app/main.cpp src/app/wizard.cpp src/app/cli.cpp src/app/launcher_ui.cpp src/app/ui_*.cpp src/app/payload_embed.cpp \
  src/wg_core/*.cpp src/wg_ui/*.cpp $CONV_SRC "$GEN/resources.o" "$GEN/restore.o" $LIBS -ld2d1 -ld3d11 -ldxgi -ldwrite -lwindowscodecs -ldxguid -luiautomationcore -loleaut32 -o "$OUT/WrongGPU.exe"
echo "built $OUT/WrongGPU.exe and $OUT/WrongGPU-Restore.exe"
