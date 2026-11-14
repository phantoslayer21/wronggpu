# SPDX-License-Identifier: MIT
"""Thin ctypes wrapper over the HIP runtime C API (amdhip64), written from the public HIP API documentation.

Only the calls the runner needs: device memory, module loading, kernel launch, events.
"""
import ctypes
import glob
import os
import sys
import warnings

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2


class HipError(RuntimeError):
    pass


# The HIP SDK every HIP / clang process of this project uses, and the environment cleanup applied before the HIP
# runtime is loaded (and so inherited by every child process): every variable whose name contains HIP, LLVM, ROCM or
# HSA is removed, except HIP_PLATFORM and HIP_PATH_72, and ROCM_PATH is set to the SDK. A stray inherited variable
# can point the vendor runtime at an unrelated folder (an LLVM_PATH outside the SDK made hipModuleLoadData crash).
# Per process only: the user's and the machine's environment are not changed.
DEFAULT_ROCM = r"C:\Program Files\AMD\ROCm\7.2"
HIP_ENV_KEEP = ("HIP_PLATFORM", "HIP_PATH_72")
HIP_ENV_PARTS = ("HIP", "LLVM", "ROCM", "HSA")


def hip_child_environment(env=None, rocm_dir=None):
    """A copy of `env` (default os.environ) cleaned for HIP / clang processes, and the names removed (never values).
    ROCM_PATH = rocm_dir, else DEFAULT_ROCM."""
    src = dict(os.environ if env is None else env)
    keep = {k.upper() for k in HIP_ENV_KEEP}
    out, removed = {}, []
    for k, v in src.items():
        u = k.upper()
        if u not in keep and any(part in u for part in HIP_ENV_PARTS):
            removed.append(k)
        else:
            out[k] = v
    out["ROCM_PATH"] = rocm_dir or DEFAULT_ROCM
    return out, sorted(removed)


def clean_process_environment(rocm_dir=None):
    """Applies hip_child_environment to this process's os.environ (so the HIP runtime loaded afterwards and every child
    process see the cleaned environment). Returns the names removed."""
    cleaned, removed = hip_child_environment(os.environ, rocm_dir)
    for k in list(os.environ):
        if k not in cleaned:
            del os.environ[k]
    os.environ["ROCM_PATH"] = cleaned["ROCM_PATH"]
    return removed


def _check_environment(rocm_dir):
    """Warn about a stray LLVM_PATH: one pointing outside the HIP SDK made hipModuleLoadData crash for every code
    object on the spec team's machine (results/2026-10-05-swin-selftest.md)."""
    llvm = os.environ.get("LLVM_PATH")
    if not llvm:
        return
    root = os.path.normcase(os.path.abspath(rocm_dir)) if rocm_dir else ""
    path = os.path.normcase(os.path.abspath(llvm))
    if not root or os.path.commonpath([root, path]) != root:
        warnings.warn(f"LLVM_PATH is set to {llvm!r}, outside the HIP SDK ({rocm_dir!r}); an unrelated LLVM there "
                      "can crash hipModuleLoadData. Unset it if module loading fails.", RuntimeWarning, stacklevel=3)


def _find_library(rocm_dir):
    if sys.platform == "win32":
        bindir = os.path.join(rocm_dir, "bin")
        if os.path.isdir(bindir) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(bindir)
        names = ["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"]
        candidates = [os.path.join(bindir, n) for n in names] + names
    else:
        candidates = [os.path.join(rocm_dir, "lib", "libamdhip64.so")]
        candidates += sorted(glob.glob(os.path.join(rocm_dir, "lib", "libamdhip64.so.*")))
        candidates += ["libamdhip64.so"]
    errors = []
    for c in candidates:
        if os.path.isabs(c) and not os.path.exists(c):
            continue
        try:
            return ctypes.CDLL(c)
        except OSError as e:
            errors.append(f"{c}: {e}")
    raise HipError("cannot load the HIP runtime library; tried:\n  " + "\n  ".join(errors or candidates))


class Hip:
    def __init__(self, rocm_dir, device=0):
        clean_process_environment(rocm_dir)
        _check_environment(rocm_dir)
        lib = self.lib = _find_library(rocm_dir)
        self._images = []
        vp, u, i, sz = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_size_t
        P = ctypes.POINTER
        sig = {
            "hipInit": [u],
            "hipSetDevice": [i],
            "hipDeviceGetName": [ctypes.c_char_p, i, i],
            "hipMalloc": [P(vp), sz],
            "hipFree": [vp],
            "hipMemcpy": [vp, vp, sz, i],
            "hipMemset": [vp, i, sz],
            "hipDeviceSynchronize": [],
            "hipModuleLoadData": [P(vp), vp],
            "hipModuleUnload": [vp],
            "hipModuleGetFunction": [P(vp), vp, ctypes.c_char_p],
            "hipModuleGetGlobal": [P(vp), P(sz), vp, ctypes.c_char_p],
            "hipModuleLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)],
            "hipEventCreate": [P(vp)],
            "hipEventDestroy": [vp],
            "hipEventRecord": [vp, vp],
            "hipEventSynchronize": [vp],
            "hipEventElapsedTime": [P(ctypes.c_float), vp, vp],
            "hipGetLastError": [],
        }
        for name, args in sig.items():
            f = getattr(lib, name)
            f.argtypes = args
            f.restype = ctypes.c_int
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        self.check(lib.hipInit(0), "hipInit")
        self.check(lib.hipSetDevice(device), "hipSetDevice")
        name = ctypes.create_string_buffer(256)
        self.check(lib.hipDeviceGetName(name, 256, device), "hipDeviceGetName")
        self.device_name = name.value.decode(errors="replace")

    def check(self, err, what):
        if err != 0:
            msg = self.lib.hipGetErrorString(err)
            raise HipError(f"{what} failed: error {err} ({msg.decode(errors='replace') if msg else '?'})")

    # memory
    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(p), max(int(nbytes), 1)), "hipMalloc")
        return p.value

    def free(self, ptr):
        if ptr:
            self.check(self.lib.hipFree(ptr), "hipFree")

    def upload(self, ptr, data):
        data = bytes(data)
        self.check(self.lib.hipMemcpy(ptr, data, len(data), HIP_MEMCPY_HOST_TO_DEVICE), "hipMemcpy H2D")

    def download(self, ptr, nbytes):
        buf = ctypes.create_string_buffer(int(nbytes))
        self.check(self.lib.hipMemcpy(buf, ptr, int(nbytes), HIP_MEMCPY_DEVICE_TO_HOST), "hipMemcpy D2H")
        return buf.raw

    def memset(self, ptr, value, nbytes):
        self.check(self.lib.hipMemset(ptr, value, nbytes), "hipMemset")

    def sync(self):
        self.check(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")

    # modules and launches
    def load_module(self, image):
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(bytes(image), len(image))
        self._images.append(buf)  # keep the image alive for the module's lifetime
        self.check(self.lib.hipModuleLoadData(ctypes.byref(mod), buf), "hipModuleLoadData")
        return mod.value

    def get_function(self, module, name):
        f = ctypes.c_void_p()
        self.check(self.lib.hipModuleGetFunction(ctypes.byref(f), module, name.encode()),
                   f"hipModuleGetFunction({name})")
        return f.value

    def get_global(self, module, name):
        """(device pointer, size in bytes) of a global variable of a loaded code object."""
        p, n = ctypes.c_void_p(), ctypes.c_size_t()
        self.check(self.lib.hipModuleGetGlobal(ctypes.byref(p), ctypes.byref(n), module, name.encode()),
                   f"hipModuleGetGlobal({name})")
        return p.value, n.value

    def launch(self, func, grid, block, args, shared=0):
        """args: list of ctypes values (c_void_p for device pointers, c_int, ...)."""
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        self.check(self.lib.hipModuleLaunchKernel(func, int(grid), 1, 1, int(block), 1, 1, int(shared), None,
                                                  params, None), "hipModuleLaunchKernel")

    # events
    def event(self):
        e = ctypes.c_void_p()
        self.check(self.lib.hipEventCreate(ctypes.byref(e)), "hipEventCreate")
        return e.value

    def record(self, ev):
        self.check(self.lib.hipEventRecord(ev, None), "hipEventRecord")

    def elapsed_ms(self, start, stop):
        self.check(self.lib.hipEventSynchronize(stop), "hipEventSynchronize")
        ms = ctypes.c_float()
        self.check(self.lib.hipEventElapsedTime(ctypes.byref(ms), start, stop), "hipEventElapsedTime")
        return ms.value

    def destroy_event(self, ev):
        self.check(self.lib.hipEventDestroy(ev), "hipEventDestroy")
