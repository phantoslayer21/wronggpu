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
