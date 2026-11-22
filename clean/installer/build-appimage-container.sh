#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Runs build-appimage.sh in the pinned Ubuntu 22.04 container with podman (or docker), from any Linux host.
#   ./build-appimage-container.sh --payload <folder> [--version x.y.z] [--out <dir>]
#   ./build-appimage-container.sh --test-payload
# The repository is mounted read-write at /src; the payload folder and --out must be inside it. Output: build/appimage/out.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
IMAGE="$(sed -n 's/^CONTAINER_IMAGE="\(.*\)"/\1/p' "$HERE/build-appimage.sh")"
ENGINE="${CONTAINER_ENGINE:-$(command -v podman || command -v docker)}"
[ -n "$ENGINE" ] || { echo "install podman or docker" >&2; exit 1; }
# Paths on the command line are repository-relative inside the container.
args=(); while [ $# -gt 0 ]; do
  case "$1" in
    --payload|--out) p="$(mkdir -p "$2" && cd "$2" && pwd)"; case "$p" in "$REPO"/*) ;; *) echo "$1 must be inside $REPO" >&2; exit 2 ;; esac
                     args+=("$1" "/src/${p#"$REPO"/}"); shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
# Rootless podman maps the invoking user to root inside, so files written to the mount belong to the caller. Docker needs
# --user, but apt needs root: it runs as root and the files are handed back at the end.
OWNER=""; [ "$(basename "$ENGINE")" = docker ] && OWNER="chown -R $(id -u):$(id -g) /src/build/appimage;"
# WG_WORK (scratch outside the repository) is mounted at the same path.
MOUNTS=(-v "$REPO:/src"); [ -z "${WG_WORK:-}" ] || { mkdir -p "$WG_WORK"; MOUNTS+=(-v "$WG_WORK:$WG_WORK" -e WG_WORK); }
exec "$ENGINE" run --rm "${MOUNTS[@]}" -w /src "$IMAGE" bash -c \
  "set +e; /src/clean/installer/build-appimage.sh \"\$@\"; rc=\$?; $OWNER exit \$rc" bash "${args[@]}"
