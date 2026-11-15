# SPDX-License-Identifier: MIT
"""The HIP SDK of every build and HIP process of this project: AMD's official HIP SDK in its standard install folder,
never one taken from the environment (ROCM_PATH, HIP_PATH, LLVM_PATH, HIP_DEVICE_LIB_PATH or PATH). A script's --rocm
may name only that folder; another one needs --force-sdk. Every build prints the SDK, compiler and device-library
folders it uses (banner), so a build log shows where its code objects came from."""
import os
import sys

OFFICIAL_ROCM = r"C:\Program Files\AMD\ROCm\7.2" if sys.platform == "win32" else "/opt/rocm"


def _norm(path):
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def is_official(path):
    return bool(path) and _norm(path) == _norm(OFFICIAL_ROCM)


def resolve(rocm=None, force=False):
    """The SDK folder for a --rocm value: empty or None = the official SDK; another folder only with force."""
    if not rocm:
        return OFFICIAL_ROCM
    if is_official(rocm) or force:
        return rocm
    raise SystemExit(f"--rocm {rocm!r} is not the official HIP SDK ({OFFICIAL_ROCM}); pass --force-sdk to build with "
                     "it anyway")


def device_lib_dir(rocm):
    return os.path.join(rocm, "amdgcn", "bitcode")


def add_arguments(ap, help_suffix=""):
    """--rocm (default: the official SDK, never an environment variable) and --force-sdk."""
    ap.add_argument("--rocm", default="", help=f"HIP SDK folder (default and only accepted value without --force-sdk: "
                                               f"{OFFICIAL_ROCM}){help_suffix}")
    ap.add_argument("--force-sdk", action="store_true", help="accept a --rocm other than the official HIP SDK")


def banner(rocm, clang=None, cxx=None, device_libs=True):
    """One line naming the SDK, the compilers and the device-library folder of a build."""
    parts = [f"HIP SDK: {rocm} ({'official' if is_official(rocm) else 'NOT the official SDK: forced'})"]
    if clang:
        parts.append(f"clang: {clang}")
    if cxx:
        parts.append(f"host C++: {cxx}")
    parts.append(f"device libs: {device_lib_dir(rocm) if device_libs else 'none (-nogpulib)'}")
    return "; ".join(parts)
