# SPDX-License-Identifier: MIT
"""Build the model K code objects (and a model K-only host emulator library for tests).

clean/runner/build.py calls build_device() for every normal build, and adds SOURCES to its emulator library, so
`python clean/runner/build.py --rocm <ROCm> --out build` builds model K too. Standalone:

    python clean/k/runner/k_build.py --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --out build      # code objects
    python clean/k/runner/k_build.py --out build --no-device-libs --clang clang --report          # compile check
    python clean/k/runner/k_build.py --out build --emu-only                                       # wg_emu_k.so

Code objects (spec 90 compile command, -I clean/kernels for wg_platform.h):
    k_linear.co        linear / merge / expand kernels, 32 tokens per wave (default)
    k_linear__g1.co    the same with 16 tokens per wave (timing variant "@g1")
    k_block.co         the spec 41 block kernels
    k_block_wide.co    wide stage kernels for levels 3..5 (one window per 2 or 4 groups of 4 waves)
    k_window.co        window gather / scatter (K3 tests)
    k_act.co           activation tables (K2 test on the GPU)
    k_io.co            input stage (spec 42) and output stage (spec 43), wave64 by default
                       (standalone --io-wave32 retains the comparison build; neural kernels stay wave32)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

K_KERNEL_DIR = os.path.join(os.path.dirname(HERE), "kernels")
M_KERNEL_DIR = os.path.join(ROOT, "clean", "kernels")
SOURCES = ["k_linear.hip", "k_block.hip", "k_block_wide.hip", "k_window.hip", "k_act.hip", "k_io.hip"]
# (code object, source, defines)
BUILDS = [("k_linear.co", "k_linear.hip", {}), ("k_linear__g1.co", "k_linear.hip", {"WGK_LIN_G": "1"}),
          ("k_block.co", "k_block.hip", {}), ("k_block_wide.co", "k_block_wide.hip", {}),
          ("k_window.co", "k_window.hip", {}), ("k_act.co", "k_act.hip", {}),
          ("k_io.co", "k_io.hip", {}), ("k_io_exact.co", "k_io.hip", {"WGK_IN_EXACT": "1"})]
EMU_LIBRARY = "wg_emu_k.so"


def source_paths():
    return [os.path.join(K_KERNEL_DIR, s) for s in SOURCES]


def build_device(clang, rocm, out_dir, device_libs=True, report=False, run=None, io_wave64=True):
    from clean.runner.build import device_command
    from clean.runner.build import run as default_run
    run = run or default_run
    for co, src, defines in BUILDS:
        extra = ["-I", M_KERNEL_DIR]
        if src == "k_io.hip" and io_wave64:
            extra.append("-mwavefrontsize64")
        run(device_command(clang, rocm, os.path.join(K_KERNEL_DIR, src), os.path.join(out_dir, co),
                           device_libs=device_libs, report=report, defines=defines, extra=extra))


def emu_command(cxx, out):
    return [cxx, "-std=c++17", "-O2", "-g", "-Wno-psabi", "-shared", "-fPIC", "-DWG_EMU", "-I", M_KERNEL_DIR,
            os.path.join(M_KERNEL_DIR, "emu", "wg_emu.cpp"), "-x", "c++", *source_paths(), "-o", out]


def main(argv=None):
    from clean.runner.build import find_clang, run
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--clang")
    ap.add_argument("--no-device-libs", action="store_true")
    ap.add_argument("--report", action="store_true", help="print VGPR / spill / occupancy per kernel")
    ap.add_argument("--io-wave32", action="store_true", help="A/B baseline: compile only K I/O in wave32")
    ap.add_argument("--emu-only", action="store_true", help="only the host emulator library " + EMU_LIBRARY)
    ap.add_argument("--emu", action="store_true", help="also the host emulator library " + EMU_LIBRARY)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if not a.emu_only:
        if not a.rocm and not a.no_device_libs:
            raise SystemExit("--rocm is required (or --no-device-libs for a compile check)")
        build_device(a.clang or find_clang(a.rocm), a.rocm, a.out, not a.no_device_libs, a.report, io_wave64=not a.io_wave32)
    if a.emu or a.emu_only:
        run(emu_command(a.cxx, os.path.join(a.out, EMU_LIBRARY)))


if __name__ == "__main__":
    main()
