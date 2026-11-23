# SPDX-License-Identifier: MIT
"""Cross-builds the headless test hosts on Linux with llvm-mingw (static, runs under Wine):
  kernel_replay.exe   clean/dx12/tests/kernel_replay.cpp + the runtime's device sources (runs launches on the native
                      D3D12 device; the device refuses a card that is not RDNA 4 until the family gate is opened)
  native_probe_test.exe  the borrowed-device path with its driver-channel probe (a device made here, handed to the runtime);
                      -DWG_DEV_GFX11 in --cxx makes it (and kernel_replay) accept an RX 7900 series board (runtime/rdna4.h)
  pso_sweep.exe       pso_sweep.cpp (PSO creation per kernel with the runtime's root signatures; no runtime code)
  gfx11_probe_host.exe  the host of test_gfx11_helpers.py
  frame_host.exe      a whole frame on the runtime library's native D3D12 executor (the host of frame_parity.py; loads
                      wronggpu_runtime.dll, itself a Windows build of clean/runtime)
    python3 build_kernel_replay.py --cxx <x86_64-w64-mingw32-g++ [flags]> --out <dir> [--only pso_sweep,...]
"""
import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runtime")
CONV = os.path.join(os.path.dirname(os.path.dirname(HERE)), "converter", "src")
TARGETS = {
    "kernel_replay": [os.path.join(HERE, "kernel_replay.cpp"), *(os.path.join(RT, f) for f in
                      ("device.cpp", "device_d3d12.cpp", "embedded.cpp")), os.path.join(CONV, "json.cpp"),
                      os.path.join(CONV, "sha256.cpp")],
    "native_probe_test": [os.path.join(HERE, "native_probe_test.cpp"), *(os.path.join(RT, f) for f in
                          ("device.cpp", "device_d3d12.cpp", "embedded.cpp")), os.path.join(CONV, "json.cpp"),
                          os.path.join(CONV, "sha256.cpp")],
    "pso_sweep": [os.path.join(HERE, "pso_sweep.cpp")],
    "gfx11_probe_host": [os.path.join(HERE, "gfx11_probe_host.cpp")],
    "frame_host": [os.path.join(HERE, "frame_host.cpp")],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cxx", required=True, help="llvm-mingw g++ command (flags allowed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="comma list of: " + ", ".join(TARGETS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in (a.only.split(",") if a.only else TARGETS):
        cmd = [*shlex.split(a.cxx), "-std=c++17", "-O2", "-fms-extensions", "-static", "-static-libgcc", "-static-libstdc++",
               "-D_CRT_SECURE_NO_WARNINGS", *TARGETS[name], "-o", os.path.join(a.out, name + ".exe")]
        print(" ".join(shlex.quote(c) for c in cmd[:1]), "...", name, flush=True)
        subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
