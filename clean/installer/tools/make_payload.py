#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generates the installer's embedded resources (replaces build-installer.ps1 of the 0.2.0 installer).

    make_payload.py --payload <dir> --map <map.json> --version <x.y.z> [--manifest <setup.manifest>] [--hosted-only]
                    --out <dir>

Writes <out>/payload_table.h, <out>/resources.rc and <out>/app.ico. The payload directory mirrors the install folder
(the folder that holds the game's amd_fidelityfx_dx12.dll): WrongGPU's amd_fidelityfx_dx12.dll (the FSR front end of
spec 20) at its root, plus the clean runtime and its kernel files, all from one release build. Every file is embedded,
hashed (SHA-256) and verified by the installer. The converter map (spec 08) is embedded with its hash.

Written for this project; Python standard library only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys

FRONT_END = "amd_fidelityfx_dx12.dll"
FSR30 = ("ffx_fsr3_x64.dll", "ffx_fsr3upscaler_x64.dll")  # WrongGPU's FSR 3.0 component DLLs: the direct route only
FORBIDDEN_PREFIXES = ("nvngx", "nvapi", "dlss")         # WrongGPU never ships or imitates NVIDIA files
RESERVED = ("amd_fidelityfx_dx12.wronggpu-original.dll", "wronggpu-setup")
MAP_ID = 900
MAP_F_ID = 902  # the RR converter map (preset F)
MAP_L_ID = 903  # Model L's converter map (preset L, #89)
RR_FIRST_ID = 1601  # the Ray Reconstruction files (--rr)
RR_FILES = ("plugins\\wronggpu-rr.asi",)  # what an RR folder holds (lower case)
# release 0.6.0 (owner 2026-11-02: one runtime DLL): the RR runtime is part of the payload's wronggpu_runtime.dll; an RR folder
# with a runtime of its own (the first 0.6.0 build's layout) is refused
RR_LEGACY_RUNTIME = "wronggpu_rr_runtime.dll"
# what it may hold besides: builds of the plugin for one version of the game interface it hooks (plugins\<major>.<minor>\),
# and interposer.json (the RR proxy's binding file naming the game interface's file: read here and embedded as a name, not
# as a file)
RR_VARIANT = re.compile(r"^plugins\\[0-9]+\.[0-9]+\\wronggpu-rr\.asi$")
RR_INTERFACE = "interposer.json"


# Never OptiScaler or any file of it (owner 2026-11-02: "The launcher shouldn't include ANY of the optiscaler fork in the
# bundle", the fork is GPL): the launcher downloads WrongGPU's edition of OptiScaler from its own project
# (src/wg_core/optiedition.h). A file named OptiScaler*, under a folder named so, or whose version resource's ProductName
# is OptiScaler stops the build (an earlier build embedded the edition as optiscaler-edition\OptiScaler.dll).
def optiscaler_file(rel, data):
    if any("optiscaler" in p.lower() for p in rel.replace("/", "\\").split("\\")):
        return True
    key, name = "ProductName".encode("utf-16-le"), "OptiScaler".encode("utf-16-le")
    i = data.find(key)
    while i >= 0:
        if name in data[i:i + 64]:
            return True
        i = data.find(key, i + 1)
    return False
LOGO_ID = 950   # the wordmark PNG (assets/wronggpu-wordmark.png), drawn in the wizard's sidebar
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def fail(msg):
    sys.exit("make_payload: " + msg)


def payload_files(root, hosted_only=False):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("/", "\\")
            parts = rel.split("\\")
            low = [p.lower() for p in parts]
            if any(p.startswith(FORBIDDEN_PREFIXES) for p in low):
                fail("payload file %s has a name WrongGPU must not use" % rel)
            if hosted_only and len(low) == 1 and low[0] in FSR30:
                print("payload: %s is not built in (--hosted-only)" % rel)
                continue
            if low[0] in RESERVED:
                fail("payload file %s uses a name reserved for the installer" % rel)
            if any(p in ("", ".", "..") or ":" in p for p in parts):
                fail("bad payload path %s" % rel)
            with open(full, "rb") as f:
                data = f.read()
            if optiscaler_file(rel, data):
                fail("payload file %s is OptiScaler: WrongGPU never ships OptiScaler (the launcher downloads its edition)" % rel)
            rows.append((rel, hashlib.sha256(data).hexdigest(), len(data), full))
    rows.sort(key=lambda r: r[0].lower())
    if not any(r[0].lower() == FRONT_END for r in rows):
        fail("the payload has no %s (the FSR front end) at its root" % FRONT_END)
    return rows


def rr_files(root):
    """The RR plugin and runtime of one RR release build (rr.h): RR_FILES, and the plugin's builds for one interface version
    and the OptiScaler edition when there, hashed like the payload; with the game interface's file name (interposer.json)."""
    rows = []
    interface = ""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("/", "\\")
            if rel.lower() == RR_INTERFACE:
                with open(full, encoding="utf-8") as f:
                    interface = str(json.load(f).get("file_name", "")).strip()
                if not re.match(r"^[A-Za-z0-9_.-]+\.dll$", interface):
                    fail("%s names no DLL file (file_name)" % rel)
                continue
            with open(full, "rb") as f:
                data = f.read()
            if optiscaler_file(rel, data):
                fail("RR folder file %s is OptiScaler: WrongGPU never ships OptiScaler (the launcher downloads its edition)" % rel)
            if os.path.basename(rel).lower() == RR_LEGACY_RUNTIME:
                fail("%s: the RR runtime is part of wronggpu_runtime.dll (release 0.6.0: one runtime DLL); no RR runtime of its own" % rel)
            if rel.lower() not in RR_FILES and not RR_VARIANT.match(rel.lower()):
                fail("RR folder file %s is not one of %s, plugins\\<major>.<minor>\\WrongGPU-RR.asi or %s" %
                     (rel, ", ".join(RR_FILES), RR_INTERFACE))
            if any(p.lower().startswith(FORBIDDEN_PREFIXES) for p in rel.split("\\")):
                fail("RR file %s has a name WrongGPU must not use" % rel)
            rows.append((rel, hashlib.sha256(data).hexdigest(), len(data), full))
    rows.sort(key=lambda r: r[0].lower())
    if not set(RR_FILES) <= set(r[0].lower() for r in rows):
        fail("the RR folder needs %s" % " and ".join(RR_FILES))
    if any(RR_VARIANT.match(r[0].lower()) for r in rows) and not interface:
        fail("the RR folder has plugin builds for one interface version but no %s naming the interface's file" % RR_INTERFACE)
    return rows, interface


def icon_bmp(n):
    """One 32-bit BMP icon image (BITMAPINFOHEADER, BGRA rows bottom-up, AND mask): a dark rounded square with an
    orange bar and a white W, drawn here so the repository holds no binary art."""
    dark, orange, white = (41, 27, 22, 255), (4, 93, 232, 255), (255, 255, 255, 255)
    rad = max(2, int(n * 0.22))
    px = [[(0, 0, 0, 0)] * n for _ in range(n)]

    def inside(x, y):
        cx = min(max(x, rad), n - 1 - rad)
        cy = min(max(y, rad), n - 1 - rad)
        return (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad

    for y in range(n):
        for x in range(n):
            if inside(x + 0.5, y + 0.5):
                px[y][x] = orange if y >= int(n * 0.80) else dark
    # W: four strokes between five points.
    pts = [(0.18, 0.22), (0.34, 0.68), (0.50, 0.36), (0.66, 0.68), (0.82, 0.22)]
    width = max(1.0, n * 0.075)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        x0, y0, x1, y1 = x0 * n, y0 * n, x1 * n, y1 * n
        for y in range(n):
            for x in range(n):
                px_, py_ = x + 0.5, y + 0.5
                dx, dy = x1 - x0, y1 - y0
                t = max(0.0, min(1.0, ((px_ - x0) * dx + (py_ - y0) * dy) / (dx * dx + dy * dy)))
                if (px_ - x0 - t * dx) ** 2 + (py_ - y0 - t * dy) ** 2 <= (width / 2) ** 2:
                    px[y][x] = white
    header = struct.pack("<IiiHHIIiiII", 40, n, 2 * n, 1, 32, 0, n * n * 4, 0, 0, 0, 0)
    color = b"".join(bytes(px[y][x]) for y in range(n - 1, -1, -1) for x in range(n))
    mask_row = ((n + 31) // 32) * 4
    mask = b"\0" * (mask_row * n)
    return header + color + mask


def write_icon(path):
    sizes = [16, 32, 48]
    images = [icon_bmp(s) for s in sizes]
    out = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    for s, img in zip(sizes, images):
        out += struct.pack("<BBBBHHII", s, s, 0, 0, 1, 32, len(img), offset)
        offset += len(img)
    with open(path, "wb") as f:
        f.write(out + b"".join(images))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--map-k", help="Preset K converter map; omit for an installer without Preset K "
                    "(CMake WRONGGPU_PRESET_K=OFF): no Preset K resource, no kKMap* constants")
    ap.add_argument("--version", required=True)
    ap.add_argument("--build-id", default="", help="the build's own name (a pre-release such as 0.5.5-launcher-dev); "
                    "default: the version. Shown next to the version; the version resource's ProductVersion")
    ap.add_argument("--payload-name", default="", help="the release build the payload comes from (e.g. 0.5.5-ea), for logs")
    ap.add_argument("--manifest", help="application manifest to embed in the resource script (MinGW); omit when the "
                    "linker embeds it (MSVC: CMake lists the manifest as a source file)")
    ap.add_argument("--hosted-only", action="store_true", help="an installer without the direct FSR route (CMake "
                    "WRONGGPU_DIRECT_ROUTE=OFF): the FSR 3.0 component DLLs at the payload root are left out")
    ap.add_argument("--rr", help="folder with the Ray Reconstruction plugin (plugins\\WrongGPU-RR.asi) and runtime "
                    "(wronggpu_rr_runtime.dll), optionally the plugin's builds for one interface version "
                    "(plugins\\<major>.<minor>\\WrongGPU-RR.asi, with interposer.json); never OptiScaler; omit for an "
                    "installer without them")
    ap.add_argument("--map-f", help="the RR converter map (preset F); omit for an installer without it")
    ap.add_argument("--map-l", help="Model L's converter map (preset L); omit for an installer without Model L")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.abspath(a.payload)
    if not os.path.isdir(root):
        fail("payload folder not found: " + root)
    rows = payload_files(root, a.hosted_only)
    with open(a.map, "rb") as f:
        map_bytes = f.read()
    map_k_bytes = None
    if a.map_k:
        with open(a.map_k, "rb") as f:
            map_k_bytes = f.read()
    rr_rows, rr_interface = rr_files(os.path.abspath(a.rr)) if a.rr else ([], "")
    map_f_bytes = None
    if a.map_f:
        with open(a.map_f, "rb") as f:
            map_f_bytes = f.read()
    map_l_bytes = None
    if a.map_l:
        with open(a.map_l, "rb") as f:
            map_l_bytes = f.read()
    os.makedirs(a.out, exist_ok=True)

    h = ["// Generated by tools/make_payload.py - do not edit.", "#pragma once",
         '#define SETUP_VERSION L"%s"' % a.version,
         '#define SETUP_BUILD_ID L"%s"' % (a.build_id or a.version),
         '#define SETUP_PAYLOAD_NAME L"%s"' % (a.payload_name or a.version),
         '#include "wg_core/context.h"  // wg::PayloadFile',
         "static const wg::PayloadFile kPayload[] = {"]
    for i, (rel, sha, size, _) in enumerate(rows):
        h.append('    {%d, L"%s", L"%s", %dull},' % (1001 + i, rel.replace("\\", "\\\\"), sha, size))
    logo = os.path.join(ASSETS, "wronggpu-wordmark.png")
    h += ["};"]
    if map_k_bytes is not None:
        h += ["static const int kKMapResourceId = 901;",
              'static const wchar_t kKMapSha256[] = L"%s";' % hashlib.sha256(map_k_bytes).hexdigest()]
    h.append("static const wg::PayloadFile kRrPayload[] = {")
    for i, (rel, sha, size, _) in enumerate(rr_rows):
        h.append('    {%d, L"%s", L"%s", %dull},' % (RR_FIRST_ID + i, rel.replace("\\", "\\\\"), sha, size))
    h += ['    {0, L"", L"", 0ull},  // (an array is never empty)', "};", "static const size_t kRrPayloadCount = %d;" % len(rr_rows),
          'static const wchar_t kRrInterfaceFile[] = L"%s";' % rr_interface,
          "static const int kFMapResourceId = %d;" % (MAP_F_ID if map_f_bytes is not None else 0),
          'static const wchar_t kFMapSha256[] = L"%s";' % (hashlib.sha256(map_f_bytes).hexdigest() if map_f_bytes is not None else ""),
          "static const int kLMapResourceId = %d;" % (MAP_L_ID if map_l_bytes is not None else 0),
          'static const wchar_t kLMapSha256[] = L"%s";' % (hashlib.sha256(map_l_bytes).hexdigest() if map_l_bytes is not None else "")]
    h += ["static const int kMapResourceId = %d;" % MAP_ID,
          "static const int kLogoResourceId = %d;" % (LOGO_ID if os.path.isfile(logo) else 0),
          'static const wchar_t kMapSha256[] = L"%s";' % hashlib.sha256(map_bytes).hexdigest(), ""]
    with open(os.path.join(a.out, "payload_table.h"), "w", newline="\n") as f:
        f.write("\n".join(h))

    if os.path.isfile(os.path.join(ASSETS, "wronggpu.ico")):  # the logo (brand asset); else the drawn fallback
        shutil.copyfile(os.path.join(ASSETS, "wronggpu.ico"), os.path.join(a.out, "app.ico"))
    else:
        write_icon(os.path.join(a.out, "app.ico"))
    fwd = lambda p: os.path.abspath(p).replace("\\", "/")
    parts = [p for p in a.version.replace("-", ".").split(".") if p.isdigit()][:4]
    parts += ["0"] * (4 - len(parts))
    vcomma = ",".join(parts)
    rc = ["#include <windows.h>",
          '1 ICON "%s"' % fwd(os.path.join(a.out, "app.ico")),
          "1 VERSIONINFO", " FILEVERSION %s" % vcomma, " PRODUCTVERSION %s" % vcomma,
          " FILEOS VOS_NT_WINDOWS32", " FILETYPE VFT_APP", "BEGIN", ' BLOCK "StringFileInfo"', " BEGIN",
          '  BLOCK "040904B0"', "  BEGIN",
          '   VALUE "ProductName", "WrongGPU"', '   VALUE "FileDescription", "WrongGPU Installer"',
          '   VALUE "FileVersion", "%s"' % a.version, '   VALUE "ProductVersion", "%s"' % (a.build_id or a.version),
          '   VALUE "OriginalFilename", "WrongGPU-Setup.exe"', "  END", " END",
          ' BLOCK "VarFileInfo"', " BEGIN", '  VALUE "Translation", 0x409, 1200', " END", "END",
          '%d RCDATA "%s"' % (MAP_ID, fwd(a.map))]
    if a.map_k:
        rc.append('901 RCDATA "%s"' % fwd(a.map_k))
    if a.map_f:
        rc.append('%d RCDATA "%s"' % (MAP_F_ID, fwd(a.map_f)))
    if a.map_l:
        rc.append('%d RCDATA "%s"' % (MAP_L_ID, fwd(a.map_l)))
    for i, (_, _, _, full) in enumerate(rr_rows):
        rc.append('%d RCDATA "%s"' % (RR_FIRST_ID + i, fwd(full)))
    if os.path.isfile(logo):
        rc.append('%d RCDATA "%s"' % (LOGO_ID, fwd(logo)))
    if a.manifest:
        rc.insert(2, '1 24 "%s"' % fwd(a.manifest))
    for i, (_, _, _, full) in enumerate(rows):
        rc.append('%d RCDATA "%s"' % (1001 + i, fwd(full)))
    with open(os.path.join(a.out, "resources.rc"), "w", newline="\n") as f:
        f.write("\n".join(rc) + "\n")
    print("payload: %d files, %.1f MB from %s; map %s" % (len(rows), sum(r[2] for r in rows) / 1048576.0, root, a.map))
    print("Ray Reconstruction: %s; RR converter map %s" % ("%d files from %s" % (len(rr_rows), a.rr) if rr_rows else "none", a.map_f or "none"))
    print("Model L converter map: %s" % (a.map_l or "none"))


if __name__ == "__main__":
    main()
