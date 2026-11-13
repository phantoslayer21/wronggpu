#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generated stand-ins for the Ray Reconstruction files of a WRONGGPU_TESTING build (CMake runs this when no
WRONGGPU_RR_DIR is given): plugins\\WrongGPU-RR.asi, one build for the game interfaces 2.7, 2.8 and 2.14 as the release has
it since 0.6.0 (the RR runtime is part of wronggpu_runtime.dll since 0.6.0) as minimal PE files with a version resource and a marker (never OptiScaler: WrongGPU does not ship it; the launcher
downloads its edition), and interposer.json naming a made-up interface file. They are never loaded, only installed, hashed and removed.

    python make_mock_rr.py <out folder>

Written for this project; Python standard library only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mockpe  # noqa: E402


def main():
    out = os.path.abspath(sys.argv[1])
    files = {
        os.path.join("plugins", "WrongGPU-RR.asi"): mockpe.make_dll(
            {"ProductName": "WrongGPU", "FileDescription": "WrongGPU Ray Reconstruction plugin (test stand-in)"}, (0, 5, 5, 0),
            extra=b"WrongGPU RR plugin stand-in (synthetic, never loaded)", dll_name="WrongGPU-RR.asi"),
        # the game interface's file name, as the RR proxy's binding file gives it (a made-up name for the tests)
        "interposer.json": b'{\n "file_name": "wg_test_interface.dll"\n}\n',
    }
    old28 = os.path.join(out, "plugins", "2.8", "WrongGPU-RR.asi")  # an earlier stand-in (one plugin build since 0.6.0)
    if os.path.isfile(old28):
        os.remove(old28)
        try:
            os.rmdir(os.path.dirname(old28))
        except OSError:
            pass
    if os.path.isfile(os.path.join(out, "wronggpu_rr_runtime.dll")):  # an earlier stand-in (one runtime DLL since 0.6.0)
        os.remove(os.path.join(out, "wronggpu_rr_runtime.dll"))
    stale = os.path.join(out, "optiscaler-edition", "OptiScaler.dll")  # an earlier stand-in (make_payload.py refuses it now)
    if os.path.isfile(stale):
        os.remove(stale)
        try:
            os.rmdir(os.path.dirname(stale))
        except OSError:
            pass
    for rel, data in files.items():
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        old = open(p, "rb").read() if os.path.isfile(p) else None
        if old != data:  # unchanged bytes keep their time (no rebuild of the payload)
            with open(p, "wb") as f:
                f.write(data)
    print("mock RR files in", out)


if __name__ == "__main__":
    main()
