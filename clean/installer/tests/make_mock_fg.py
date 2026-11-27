#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generated stand-ins for the frame generation files of a WRONGGPU_TESTING build with WRONGGPU_FG (CMake runs this when no
WRONGGPU_FG_DIR is given): plugins\\WrongGPU-FG.asi as a minimal PE file with a version resource and a marker, and two kernel
files in the release stage folder's layout (fg-dxil\\mock.dxil, fg-dxil\\mock_b.dxil) with the kernel build's list beside them
(fg-dxil\\manifest.json, which the payload step reads past). They are never loaded, only embedded, installed, hashed and removed.

    python make_mock_fg.py <out folder>

Written for this project; Python standard library only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mockpe  # noqa: E402


def main():
    out = os.path.abspath(sys.argv[1])
    files = {
        os.path.join("plugins", "WrongGPU-FG.asi"): mockpe.make_dll(
            {"ProductName": "WrongGPU", "FileDescription": "WrongGPU frame generation plugin (test stand-in)"}, (0, 7, 0, 0),
            extra=b"WrongGPU FG plugin stand-in (synthetic, never loaded)", dll_name="WrongGPU-FG.asi"),
        os.path.join("fg-dxil", "mock.dxil"): b"DXBC synthetic frame generation kernel stand-in A (never loaded)\n" * 8,
        os.path.join("fg-dxil", "mock_b.dxil"): b"DXBC synthetic frame generation kernel stand-in B (never loaded)\n" * 8,
        os.path.join("fg-dxil", "manifest.json"): b'{\n "format": 1,\n "test_only": true,\n "kernels": []\n}\n',
    }
    for rel, data in files.items():
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        old = open(p, "rb").read() if os.path.isfile(p) else None
        if old != data:  # unchanged bytes keep their time (no rebuild of the payload)
            with open(p, "wb") as f:
                f.write(data)
    print("mock FG files in", out)


if __name__ == "__main__":
    main()
