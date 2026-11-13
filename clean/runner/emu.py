# SPDX-License-Identifier: MIT
"""Backend with the same interface as hip.Hip that runs the kernels in the host lane emulator (test-only).

The emulator library is built by `build.py --emu` from the same kernel sources as the GPU code object.
"""
import ctypes
import os
import time

import numpy as np


class Emu:
    device_name = "host emulator"

    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        self.lib.emu_launch.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.emu_launch.restype = ctypes.c_int
        self.lib.emu_has_kernel.argtypes = [ctypes.c_char_p]
        self.lib.emu_set_layout.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.emu_wmma_count.restype = ctypes.c_longlong
        self._bufs = {}

    def set_layout(self, kvariant, dvariant):
        """Switch the emulated WMMA K-slot permutation and accumulator row map (see emu/wg_emu.cpp)."""
        self.lib.emu_set_layout(kvariant, dvariant)

    def wmma_count(self):
        return self.lib.emu_wmma_count()

    def malloc(self, nbytes):
        raw = np.zeros(int(nbytes) + 512, dtype=np.uint8)
        addr = raw.ctypes.data
        aligned = (addr + 255) & ~255
        self._bufs[aligned] = raw
        return aligned

    def free(self, ptr):
        self._bufs.pop(ptr, None)

    def upload(self, ptr, data):
        data = bytes(data)
        ctypes.memmove(ptr, data, len(data))

    def download(self, ptr, nbytes):
        return ctypes.string_at(ptr, int(nbytes))

    def memset(self, ptr, value, nbytes):
        ctypes.memset(ptr, value, int(nbytes))

    def sync(self):
        pass

    def load_module(self, image):
        return None

    def get_function(self, module, name):
        if not self.lib.emu_has_kernel(name.encode()):
            raise RuntimeError(f"emulator has no kernel {name}")
        return name

    def get_global(self, module, name, nbytes=8 * 64):
        """(host address, size) of an extern "C" global of the emulator library (size: given by the caller)."""
        return ctypes.addressof(ctypes.c_ubyte.in_dll(self.lib, name)), nbytes

    def launch(self, func, grid, block, args, shared=0):
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        if self.lib.emu_launch(func.encode(), int(grid), int(block), params) != 0:
            raise RuntimeError(f"emu_launch({func}) failed")

    def event(self):
        return [0.0]

    def record(self, ev):
        ev[0] = time.perf_counter()

    def elapsed_ms(self, start, stop):
        return (stop[0] - start[0]) * 1e3

    def destroy_event(self, ev):
        pass
