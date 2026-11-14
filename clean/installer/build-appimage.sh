#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Builds WrongGPU-<version>-x86_64.AppImage, the Linux launcher, from this checkout.
#
#   build-appimage.sh --payload <release payload folder> [--version x.y.z] [--out <dir>]
#   build-appimage.sh --test-payload [--version x.y.z] [--out <dir>]     smoke build with the test stand-in payload; never ship it
#
# Run it inside the pinned Ubuntu 22.04 container (build-appimage-container.sh does that from any host). The AppImage runs
# on every system whose glibc is as old as the build system's or newer; building on a newer system raises that floor
# silently, so the script refuses other hosts and step 9 fails on any file that needs a newer glibc.
#
# The script does not know the launcher's source files or screens: CMakeLists.txt's install() rules decide what ships and
# linuxdeploy follows the binary's shared-library dependencies. The only libraries it names are the host libraries it
# expects from the user's system (HOST_LIBS below); step 9 fails on any other dependency the AppDir does not carry.
# New launcher code needs no edit here.
set -euo pipefail

# ---- 0. Pinned inputs ---------------------------------------------------------------------------------------------------
# Every download is checked against its SHA-256 before use. Exact versions on purpose: a "latest" URL changes under the
# same checksum. To update a pin, change the version and the checksum together (README-linux.md, "Updating a pin").
CONTAINER_IMAGE="docker.io/library/ubuntu:22.04@sha256:74b12b11ffa550fba1929ad4d8d16533c34f6612728469671afba1b03b66881c"

SDL_VERSION="3.4.18"
SDL_URL="https://github.com/libsdl-org/SDL/releases/download/release-${SDL_VERSION}/SDL3-${SDL_VERSION}.tar.gz"
SDL_SHA256="9c75cf16330322c217dedd2e0609f1124f1b54b8633e763467b4684d0f4334a3"

LINUXDEPLOY_TAG="1-alpha-20251107-1"
LINUXDEPLOY_URL="https://github.com/linuxdeploy/linuxdeploy/releases/download/${LINUXDEPLOY_TAG}/linuxdeploy-x86_64.AppImage"
LINUXDEPLOY_SHA256="c20cd71e3a4e3b80c3483cef793cda3f4e990aca14014d23c544ca3ce1270b4d"

APPIMAGETOOL_TAG="1.9.1"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/${APPIMAGETOOL_TAG}/appimagetool-x86_64.AppImage"
APPIMAGETOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"

# appimagetool fetches "the newest runtime" unless it is given one; that would tie the build to the day it runs.
RUNTIME_TAG="20251108"
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/${RUNTIME_TAG}/runtime-x86_64"
RUNTIME_SHA256="2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"

MAX_GLIBC="2.35"   # Ubuntu 22.04; decided by the maintainer
# The newest libstdc++ symbol versions Ubuntu 22.04's libstdc++6 (12.x) provides. The binary links libstdc++ statically, so
# nothing in the AppDir should need either; step 9 reports anything that does.
MAX_GLIBCXX="3.4.30"
MAX_CXXABI="1.3.13"

# The shared libraries the AppImage deliberately does not carry: it takes them from the user's system. Step 9 fails when a
# file in the AppDir needs any other library that the AppDir lacks. Each entry is here for a reason:
#  - the glibc family: a second copy cannot work, and MAX_GLIBC bounds what the AppImage asks of it. libdl, libpthread and
#    librt are folded into libc since glibc 2.34, kept as empty stubs for programs linked before that;
#  - libX11.so.6 and libxcb.so.1: the X11 client libraries. Cairo's X11 backend and SDL3 open the user's own, and linuxdeploy
#    excludes them on purpose: a second copy clashes with the one the desktop's other programs load. Every graphical desktop
#    has them, Wayland ones through XWayland;
#  - libz.so.1 and libuuid.so.1: base libraries of every Debian, Ubuntu, Fedora and Arch install (zlib, util-linux), which
#    Cairo, FreeType, GIO and libpng (zlib) and fontconfig (uuid) need. linuxdeploy's exclude list leaves them out too.
# What the build adds beyond the glibc family was read from the AppDir's own NEEDED entries (readelf -d on every ELF file).
HOST_LIBS=(
  ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1
  libX11.so.6 libxcb.so.1
  libz.so.1 libuuid.so.1
)

# From Ubuntu 22.04's archive, unpinned: apt versions follow Ubuntu's security updates, the container digest bounds the drift.
APT_PACKAGES=(
  ca-certificates curl git binutils file python3 cmake ninja-build pkg-config build-essential
  libcairo2-dev libpango1.0-dev libfontconfig-dev                                  # the launcher's drawing
  libx11-dev libxext-dev libxcursor-dev libxi-dev libxfixes-dev libxrandr-dev libxtst-dev  # SDL3's X11 backend
  libxkbcommon-dev libwayland-dev wayland-protocols libdbus-1-dev libudev-dev      # SDL3's Wayland, D-Bus and udev
  libasound2-dev libpulse-dev                                                      # SDL3's audio
)

# ---- 1. Arguments and paths ---------------------------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # clean/installer, where CMakeLists.txt is
PAYLOAD="" ; TEST_PAYLOAD=0 ; VERSION="" ; OUT=""
usage() { sed -n '3,6p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --payload)      [ $# -ge 2 ] || usage; PAYLOAD="$(cd "$2" && pwd)"; shift 2 ;;
    --test-payload) TEST_PAYLOAD=1; shift ;;
    --version)      [ $# -ge 2 ] || usage; VERSION="$2"; shift 2 ;;
    --out)          [ $# -ge 2 ] || usage; mkdir -p "$2"; OUT="$(cd "$2" && pwd)"; shift 2 ;;
    *) usage ;;
  esac
done
# exactly one of --payload and --test-payload
if [ -n "$PAYLOAD" ] && [ "$TEST_PAYLOAD" = 1 ]; then usage; fi
if [ -z "$PAYLOAD" ] && [ "$TEST_PAYLOAD" = 0 ]; then usage; fi
# One source of the version: the default in CMakeLists.txt, as on Windows.
[ -n "$VERSION" ] || VERSION="$(sed -n 's/^set(WRONGGPU_VERSION "\([^"]*\)".*/\1/p' "$HERE/CMakeLists.txt")"
[ -n "$VERSION" ] || { echo "cannot read WRONGGPU_VERSION from CMakeLists.txt; pass --version" >&2; exit 2; }

WORK="${WG_WORK:-$HERE/../../build/appimage}"                # scratch inside the project, never /tmp (a reboot wipes it)
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
OUT="${OUT:-$WORK/out}"; mkdir -p "$OUT"
CACHE="$WORK/cache" ; DEPS="$WORK/deps" ; BUILD="$WORK/cmake" ; APPDIR="$WORK/AppDir"
mkdir -p "$CACHE" "$DEPS"
# The same commit gives the same file times in the squashfs.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -c safe.directory='*' -C "$HERE" log -1 --format=%ct 2>/dev/null || date +%s)}"
step() { printf '\n==== %s\n' "$*"; }
START=$SECONDS

# ---- 2. Host check ------------------------------------------------------------------------------------------------------
step "2. Host"
# os-release defines VERSION, which is also this script's variable: read it in a subshell.
HOST="$( [ -r /etc/os-release ] && . /etc/os-release && echo "${ID:-unknown} ${VERSION_ID:-}" || echo "unknown" )"
if [ "$HOST" != "ubuntu 22.04" ]; then
  [ "${WG_ALLOW_OTHER_HOST:-0}" = 1 ] || {
    echo "This is not Ubuntu 22.04 ($HOST): the AppImage would need a newer glibc than 22.04's." >&2
    echo "Use build-appimage-container.sh, or set WG_ALLOW_OTHER_HOST=1 for a local test build that must not be shipped." >&2
    exit 1; }
fi

# ---- 3. Packages --------------------------------------------------------------------------------------------------------
step "3. apt packages"
if [ "$(id -u)" = 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}"
else
  echo "not root: assuming the packages are installed (${APT_PACKAGES[*]})"
fi

# ---- 4. Pinned tools ----------------------------------------------------------------------------------------------------
step "4. Pinned tools"
fetch() {  # fetch <url> <sha256> <dest>: checks before anything runs, so a changed file stops the build
  if [ -f "$3" ] && echo "$2  $3" | sha256sum -c --status -; then return 0; fi
  curl --fail --location --silent --show-error --retry 3 --output "$3.part" "$1"
  echo "$2  $3.part" | sha256sum -c --status - || { echo "checksum mismatch for $1" >&2; rm -f "$3.part"; exit 1; }
  mv "$3.part" "$3"
}
fetch "$SDL_URL"          "$SDL_SHA256"          "$CACHE/SDL3-$SDL_VERSION.tar.gz"
fetch "$LINUXDEPLOY_URL"  "$LINUXDEPLOY_SHA256"  "$CACHE/linuxdeploy-$LINUXDEPLOY_TAG.AppImage"
fetch "$APPIMAGETOOL_URL" "$APPIMAGETOOL_SHA256" "$CACHE/appimagetool-$APPIMAGETOOL_TAG.AppImage"
fetch "$RUNTIME_URL"      "$RUNTIME_SHA256"      "$CACHE/runtime-$RUNTIME_TAG"
chmod +x "$CACHE"/*.AppImage
# Containers have no FUSE: unpack each AppImage once and run its AppRun.
unpack() {  # unpack <AppImage> <dir>
  if [ ! -x "$2/AppRun" ]; then
    rm -rf "$2" "$2.tmp"; mkdir -p "$2.tmp"
    (cd "$2.tmp" && "$1" --appimage-extract >/dev/null) && mv "$2.tmp/squashfs-root" "$2" && rmdir "$2.tmp"
  fi
}
unpack "$CACHE/linuxdeploy-$LINUXDEPLOY_TAG.AppImage"  "$CACHE/linuxdeploy-$LINUXDEPLOY_TAG"
unpack "$CACHE/appimagetool-$APPIMAGETOOL_TAG.AppImage" "$CACHE/appimagetool-$APPIMAGETOOL_TAG"
LINUXDEPLOY="$CACHE/linuxdeploy-$LINUXDEPLOY_TAG/AppRun"
APPIMAGETOOL="$CACHE/appimagetool-$APPIMAGETOOL_TAG/AppRun"

# ---- 5. SDL3, built static ----------------------------------------------------------------------------------------------
# Ubuntu 22.04 has no SDL3. Static, so there is no SDL soname to match on the user's system. SDL still opens the user's own
# X11, Wayland, audio and udev libraries at run time. The app draws its own title bar, so no libdecor.
step "5. SDL3 $SDL_VERSION (static)"
if [ ! -f "$DEPS/.sdl-$SDL_VERSION" ]; then
  rm -rf "$DEPS/sdl-src" "$DEPS/sdl-build"; mkdir -p "$DEPS/sdl-src"
  tar -xf "$CACHE/SDL3-$SDL_VERSION.tar.gz" -C "$DEPS/sdl-src" --strip-components=1
  cmake -S "$DEPS/sdl-src" -B "$DEPS/sdl-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$DEPS" \
        -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF -DSDL_X11_XSCRNSAVER=OFF \
        -DSDL_WAYLAND_LIBDECOR=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  # one command per line: in "a && b", set -e ignores a failing a, and the stamp would then claim a build that never finished
  cmake --build "$DEPS/sdl-build"
  cmake --install "$DEPS/sdl-build"
  touch "$DEPS/.sdl-$SDL_VERSION"
fi

# ---- 6. The launcher ----------------------------------------------------------------------------------------------------
# -static-libstdc++/-static-libgcc: the C++ runtime is the other thing that raises the glibc/libstdc++ floor.
step "6. Target WrongGPU $VERSION"
CMAKE_PAYLOAD=(-DWRONGGPU_PAYLOAD_DIR="$PAYLOAD" -DWRONGGPU_TESTING=OFF)
# Without a payload folder, a WRONGGPU_TESTING build makes its own stand-in payload; the app is the same code.
[ "$TEST_PAYLOAD" = 1 ] && CMAKE_PAYLOAD=(-DWRONGGPU_TESTING=ON)
cmake -S "$HERE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEPS" \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" -DWRONGGPU_VERSION="$VERSION" "${CMAKE_PAYLOAD[@]}"
cmake --build "$BUILD" --target WrongGPU

# ---- 7. AppDir ----------------------------------------------------------------------------------------------------------
# cmake --install is the one place that lists what ships (the binary, the fonts, the menu entry, the icon).
step "7. AppDir"
rm -rf "$APPDIR"
cmake --install "$BUILD" --prefix "$APPDIR/usr"
# linuxdeploy copies in the shared libraries the binary needs (Cairo, Pango, GLib, ...), but its built-in exclude list leaves
# out fontconfig, FreeType, HarfBuzz, FriBidi and expat, which would then come from the user's system: text could then look
# different from one distribution to the next, or the program fail to start on a minimal one. They are forced in. The X11
# libraries stay on the exclude list: SDL opens the user's own, and a second copy in the AppImage would clash with them. Libraries SDL opens at run time are not found
# by following the binary either; EXTRA_LIBS="/path/a.so /path/b.so" adds one that turns out missing.
FORCE=()
for soname in libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libfribidi.so.0 libexpat.so.1; do
  path="$(ldconfig -p | awk -v s="$soname" '$1 == s && /x86-64/ { print $NF; exit }')"
  [ -n "$path" ] || { echo "$soname is not installed in this container" >&2; exit 1; }
  FORCE+=(--library "$path")
done
for l in ${EXTRA_LIBS:-}; do FORCE+=(--library "$l"); done
"$LINUXDEPLOY" --appdir "$APPDIR" --executable "$APPDIR/usr/bin/WrongGPU" \
  --desktop-file "$APPDIR/usr/share/applications/wronggpu.desktop" \
  --icon-file "$APPDIR/usr/share/icons/hicolor/256x256/apps/wronggpu.png" "${FORCE[@]}"

# ---- 8. Pack ------------------------------------------------------------------------------------------------------------
step "8. AppImage"
NAME="WrongGPU-$VERSION-x86_64"; [ "$TEST_PAYLOAD" = 1 ] && NAME="$NAME-testpayload"
ARCH=x86_64 "$APPIMAGETOOL" --runtime-file "$CACHE/runtime-$RUNTIME_TAG" "$APPDIR" "$OUT/$NAME.AppImage"
(cd "$OUT" && sha256sum "$NAME.AppImage" > "$NAME.AppImage.sha256")

# ---- 9. Checks on the AppDir --------------------------------------------------------------------------------------------
# The ways an AppImage breaks far from the build machine: a file that needs a newer glibc (or libstdc++) than the user's, a
# shared library that is neither in the AppDir nor one the system is expected to have (HOST_LIBS), a drawing library left
# out, and missing fonts (the text would fall back to whatever the user has, and the icons would be blank).
step "9. Checks"
fail=0
declare -A host_lib=() in_appdir=()
for l in "${HOST_LIBS[@]}"; do host_lib[$l]=1; done
# find -L: a symbolic link that leads nowhere does not count as carried
while IFS= read -r -d '' f; do in_appdir[${f##*/}]=1; done < <(find -L "$APPDIR" -type f -name '*.so*' -print0)
newest() {  # newest <symbol-version prefix> <dynamic symbols>: the highest version of that prefix, empty when none
  { grep -o "$1_[0-9][0-9.]*" <<<"$2" || true; } | sed "s/^$1_//" | sort -V | tail -1
}
too_new() {  # too_new <version> <limit>: true when the version is above the limit
  [ -n "$1" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$2" ]
}
while IFS= read -r -d '' f; do
  rel="${f#"$APPDIR"/}"
  magic="$(file -b "$f")"
  case "$magic" in ELF*) ;; *) continue ;; esac
  # a failing readelf or objdump fails the step: it must never read as "nothing needed"
  dyn="$(readelf -d "$f" 2>&1)" || { echo "FAIL: readelf -d failed on $rel: $dyn" >&2; fail=1; continue; }
  syms=""
  if ! grep -q 'There is no dynamic section' <<<"$dyn"; then
    syms="$(objdump -T "$f" 2>&1)" || { echo "FAIL: objdump -T failed on $rel: $syms" >&2; fail=1; continue; }
  fi
  v="$(newest GLIBC "$syms")"
  if too_new "$v" "$MAX_GLIBC"; then echo "FAIL: $rel needs GLIBC_$v (limit $MAX_GLIBC)" >&2; fail=1; fi
  cxx="$(newest GLIBCXX "$syms")"
  if too_new "$cxx" "$MAX_GLIBCXX"; then echo "FAIL: $rel needs GLIBCXX_$cxx (Ubuntu 22.04 has up to $MAX_GLIBCXX)" >&2; fail=1; fi
  abi="$(newest CXXABI "$syms")"
  if too_new "$abi" "$MAX_CXXABI"; then echo "FAIL: $rel needs CXXABI_$abi (Ubuntu 22.04 has up to $MAX_CXXABI)" >&2; fail=1; fi
  while IFS= read -r need; do
    [ -n "$need" ] || continue
    if [ -z "${host_lib[$need]:-}" ] && [ -z "${in_appdir[$need]:-}" ]; then
      echo "FAIL: $rel needs $need, which is neither in the AppDir nor in HOST_LIBS" >&2; fail=1
    fi
  done < <(sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' <<<"$dyn")
  echo "glibc ${v:-none}: $rel"
done < <(find "$APPDIR" -type f -print0)
for lib in libcairo.so.2 libpango-1.0.so.0 libpangocairo-1.0.so.0 libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libfribidi.so.0 libexpat.so.1; do
  [ -e "$APPDIR/usr/lib/$lib" ] || { echo "FAIL: $lib is not in the AppDir" >&2; fail=1; }
done
for font in selawik/selawk.ttf fluent/FluentSystemIcons-Regular.ttf; do
  [ -s "$APPDIR/usr/share/wronggpu/assets/fonts/$font" ] || { echo "FAIL: font $font is not in the AppDir" >&2; fail=1; }
done
[ "$fail" = 0 ] || exit 1
echo "glibc, libstdc++, dependencies and fonts: ok (limit GLIBC_$MAX_GLIBC; host libraries: ${HOST_LIBS[*]})"
echo
echo "built $OUT/$NAME.AppImage ($(du -h "$OUT/$NAME.AppImage" | cut -f1), $((SECONDS - START)) s)"
