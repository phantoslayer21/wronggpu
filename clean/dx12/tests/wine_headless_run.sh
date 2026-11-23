#!/bin/bash
# SPDX-License-Identifier: MIT
# Runs a Windows program under Wine with no window, ever: for the headless D3D12 tests on Linux (gfx11 helpers, PSO sweep).
#   WG_WINE_PREFIX    a Wine prefix whose [Software\Wine\Drivers] "Graphics" is "null" (no X11 / Wayland driver is loaded)
#   WG_WINE_SCRATCH   a scratch folder (home, caches, a private XDG_RUNTIME_DIR); never /tmp, never /run/user/*
#   WG_GPU_LOCK       lock file shared with other GPU users (default: none; the card is shared, set it)
#   WG_WINE_MEM       memory cap of the scope (default 10G)
# usage: wine_headless_run.sh [VAR=value ...] -- <program.exe> [args]
# Wine itself never sees the login session's runtime folder (its Wayland driver opens real windows through it); only the
# systemd-run client does. Debugger dialogs are off (winedbg.exe=d, ShowCrashDialog=0).
set -u
: "${WG_WINE_PREFIX:?set WG_WINE_PREFIX}" "${WG_WINE_SCRATCH:?set WG_WINE_SCRATCH}"
case "$WG_WINE_SCRATCH" in /tmp|/tmp/*|/run/user|/run/user/*) echo "wine_headless_run: refusing scratch $WG_WINE_SCRATCH" >&2; exit 9;; esac
grep -q '"Graphics"="null"' "$WG_WINE_PREFIX/user.reg" || { echo "wine_headless_run: $WG_WINE_PREFIX has no null graphics driver" >&2; exit 9; }
REAL_XDG=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
S=$WG_WINE_SCRATCH
mkdir -p "$S/xdgrun" "$S/home" && chmod 700 "$S/xdgrun"
envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ $# -gt 0 ] && shift
wine_env=(HOME="$S/home" XDG_CACHE_HOME="$S/home/.cache" XDG_CONFIG_HOME="$S/home/.config" XDG_DATA_HOME="$S/home/.local/share"
  XDG_RUNTIME_DIR="$S/xdgrun" SDL_VIDEO_DRIVER=offscreen WINEPREFIX="$WG_WINE_PREFIX" WINEDEBUG=-all
  WINEDLLOVERRIDES="d3d12,d3d12core=n;winemenubuilder.exe=d;winedbg.exe=d;mscoree,mshtml=d"
  VKD3D_DEBUG="${VKD3D_DEBUG:-warn}" VKD3D_LOG_FILE="${VKD3D_LOG_FILE:-$S/vkd3d.log}")
lock=(); [ -n "${WG_GPU_LOCK:-}" ] && lock=(flock "$WG_GPU_LOCK")
export XDG_RUNTIME_DIR=$REAL_XDG
exec "${lock[@]}" systemd-run --user --scope -q -p MemoryMax="${WG_WINE_MEM:-10G}" -p MemorySwapMax=0 -- nice \
  env -u DISPLAY -u WAYLAND_DISPLAY -u DXIL_SPIRV_CONFIG "${wine_env[@]}" "${envs[@]}" wine "$@"
