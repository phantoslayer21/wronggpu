# SPDX-License-Identifier: MIT
"""Host emulation of D3D12 kernel text (test-only; never shipped). WrongGPU's own code.

build(source, names) compiles the @wg-kernel declarations `names` of one .hlsl file (clean/dx12/kernels or
clean/k/dx12/kernels) as C++ against hlsl_emu.h, which stands in for the platform layer, and links them with the lane
emulator of the HIP kernels (clean/kernels/emu): one shared library per set of defines, every lane of a workgroup a
coroutine running the kernel text. HlslEmu is a device with the interface of clean/runner/emu.Emu: memory is one
arena whose byte offsets are the pointers (0 = null, as the D3D12 runtime passes them), and a launch flattens its
ctypes arguments into the root constants the way the manifest's argument tokens lay them out (a pointer one constant,
the other fields their 32-bit words). So the spec 90 Runner and the model K stage plans run on the kernel text.

The text is made C++ by: the headers and the kernel file copied into a stage folder with hlsl_emu.h as
wg_platform.hlsli (and an empty wg_ags.hlsli); HLSL attributes ([unroll], [loop], ...) removed; `out` / `inout`
parameters made references (arrays stay arrays); the file's text after WG_ARGS_END closed as the body of a struct over
the argument block. Defines as build_dx12.py: WG_BLOCK, WG_WAVE, WG_GFX (11 only), the declaration's defines.
"""
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
CLEAN = os.path.dirname(DX12)
sys.path.insert(0, DX12)
import build_dx12 as B  # noqa: E402

HEADER_DIRS = [os.path.join(DX12, "hlsl"), os.path.join(CLEAN, "k", "dx12", "hlsl")]
EMU_DIR = os.path.join(CLEAN, "kernels", "emu")
ATTR = re.compile(r"\[\s*(unroll|loop|branch|flatten|fastopt|allow_uav_condition)\s*(\(\s*\d*\s*\))?\s*\]")
OUTP = re.compile(r"\b(out|inout)\s+(\w+)\s+(\w+)(\s*\[)?")
CONST_TABLE = re.compile(r"\bstatic const (int|uint|float)(\s+\w+\s*\[)")   # a table in a class body needs constexpr in C++


def _cxx_text(text):
    text = CONST_TABLE.sub(r"static constexpr \1\2", ATTR.sub("", text))
    return OUTP.sub(lambda m: f"{m.group(2)} {m.group(3)}{m.group(4)}" if m.group(4) else f"{m.group(2)}& {m.group(3)}",
                    text)


def _stage(src, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for d in HEADER_DIRS:
        for fn in os.listdir(d):
            if fn.endswith((".hlsli", ".inc")):   # .inc: the head row table of k_block.hlsli (main's #288)
                with open(os.path.join(d, fn)) as f:
                    txt = f.read()
                with open(os.path.join(out_dir, fn), "w") as f:
                    f.write(_cxx_text(txt))
    shutil.copyfile(os.path.join(HERE, "hlsl_emu.h"), os.path.join(out_dir, "wg_platform.hlsli"))
    with open(os.path.join(out_dir, "wg_ags.hlsli"), "w") as f:
        f.write("#pragma once\n// (hlsl_emu.h has the channel)\n")
    with open(src) as f:
        txt = _cxx_text(f.read())
    return out_dir, txt


def compiler():
    cxx = os.environ.get("CXX") or shutil.which("clang++")
    if not cxx or "clang" not in os.path.basename(cxx):
        raise RuntimeError("the HLSL emulation needs clang++ (ext_vector_type and _Float16 of the lane emulator)")
    return cxx


def build(src, names, out_dir, gfx=12, extra_defines=None):
    """Compile the declarations `names` of `src` (all declarations with the same defines) into one library in out_dir.
    Returns (library path, {name: entry}). extra_defines: test-only defines (negative controls)."""
    decls = {d["name"]: d for d in B.kernel_decls(src)}
    sel = [decls[n] for n in names]
    key = {(d["group"], d["wave"], tuple(sorted(d["defines"].items()))) for d in sel}
    if len(key) != 1:
        raise ValueError(f"{names}: one library per set of defines (got {key})")
    group, wave, defs = next(iter(key))
    defines = {"WG_BLOCK": group, "WG_WAVE": wave, **({"WG_GFX": gfx} if gfx != 12 else {}), **dict(defs),
               **(extra_defines or {})}
    tag = hashlib.sha256(repr((os.path.basename(src), sorted(names), sorted(defines.items()))).encode()).hexdigest()[:12]
    stage_dir, text = _stage(src, os.path.join(out_dir, f"stage_{tag}"))
    if "WG_ARGS_END" not in text:
        raise ValueError(f"{src}: no WG_ARGS_END")
    entries = {d["name"]: d["entry"] for d in sel}
    tail = ["};  // hemu_scope", "static hemu_scope& hemu_self() { return *reinterpret_cast<hemu_scope*>(hemu_consts); }"]
    for name, entry in entries.items():
        tail.append(f"static void hemu_t_{entry}() {{ hemu_self().{entry}(); }}")
    tail.append('extern "C" int hemu_launch_fn(void (*fn)(), unsigned gx, unsigned block);')
    tail.append('extern "C" int hemu_run(const char* e, unsigned gx, unsigned block) {')
    for entry in sorted(set(entries.values())):
        tail.append(f'  if (!__builtin_strcmp(e, "{entry}")) return hemu_launch_fn(&hemu_t_{entry}, gx, block);')
    tail.append("  return -2;\n}")
    kfile = os.path.join(stage_dir, "kernel.cpp")
    with open(kfile, "w") as f:
        f.write(text + "\n" + "\n".join(tail) + "\n")
    lib = os.path.join(out_dir, f"hemu_{os.path.basename(src)[:-5]}_{tag}.so")
    cxx = compiler()
    common = [cxx, "-std=c++20", "-O1", "-fPIC", "-ffp-contract=off", "-fno-fast-math", "-w"]
    objs = []
    for s, extra in ((kfile, ["-fno-operator-names", "-I", stage_dir, *[f"-D{k}={v}" for k, v in defines.items()]]),
                     (os.path.join(HERE, "hlsl_emu_bridge.cpp"), ["-DWG_EMU", "-I", os.path.join(CLEAN, "kernels")]),
                     (os.path.join(EMU_DIR, "wg_emu.cpp"), ["-DWG_EMU", "-I", os.path.join(CLEAN, "kernels")])):
        o = os.path.join(stage_dir, os.path.basename(s) + ".o")
        r = subprocess.run([*common, *extra, "-c", "-x", "c++", s, "-o", o], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"hlsl_emu: compiling {s} failed:\n{r.stderr[-6000:]}")
        objs.append(o)
    r = subprocess.run([cxx, "-shared", "-o", lib, *objs], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"hlsl_emu: link failed:\n{r.stderr[-3000:]}")
    return lib, entries


def _words(arg):
    """32-bit root constants of one launch argument (ctypes): a pointer is one constant (the arena offset)."""
    if isinstance(arg, ctypes.Structure):
        out = []
        for name, ty in arg._fields_:
            out += _words(ty(getattr(arg, name)) if not isinstance(getattr(arg, name), ctypes._SimpleCData) else
                          getattr(arg, name))
        return out
    if isinstance(arg, ctypes.c_void_p):
        v = arg.value or 0
        if v >= 1 << 32:
            raise ValueError("a pointer outside the arena")
        return [v]
    if isinstance(arg, (ctypes.c_float,)):
        return [int(np.float32(arg.value).view(np.uint32))]
    if isinstance(arg, (ctypes.c_int, ctypes.c_uint, ctypes.c_int32, ctypes.c_uint32)):
        return [arg.value & 0xffffffff]
    raise TypeError(f"launch argument {arg!r} not supported")


class HlslEmu:
    """Device with clean/runner/emu.Emu's interface over hlsl_emu libraries. kernels: {launch name: (library, entry)}."""
    device_name = "host emulator (D3D12 kernel text)"

    def __init__(self, kernels, arena_bytes=96 << 20):
        self.arena = np.zeros(arena_bytes, np.uint8)
        self._libs = {}
        self.kernels = {}
        for name, (lib, entry) in kernels.items():
            if lib not in self._libs:
                L = ctypes.CDLL(os.path.abspath(lib))
                L.hemu_run.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
                L.hemu_set_arena.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
                L.hemu_set_consts.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
                L.hemu_oob_count.restype = ctypes.c_ulonglong
                L.hemu_last_error.restype = ctypes.c_char_p
                self._libs[lib] = L
            self.kernels[name] = (self._libs[lib], entry)
        self.top = 4096
        self.launches = 0

    def malloc(self, nbytes):
        p = self.top
        self.top = (self.top + int(nbytes) + 64 + 255) & ~255
        if self.top > len(self.arena):
            raise MemoryError("hlsl_emu arena full")
        return p

    def free(self, ptr):
        pass

    def reset(self):
        self.top = 4096

    def upload(self, ptr, data):
        data = np.frombuffer(bytes(data), np.uint8)
        self.arena[ptr:ptr + len(data)] = data

    def download(self, ptr, nbytes):
        return self.arena[ptr:ptr + int(nbytes)].tobytes()

    def memset(self, ptr, value, nbytes):
        self.arena[ptr:ptr + int(nbytes)] = value

    def sync(self):
        pass

    def load_module(self, image):
        return None

    def get_function(self, module, name):
        if name not in self.kernels:
            raise RuntimeError(f"hlsl_emu has no kernel {name}")
        return name

    def launch(self, func, grid, block, args, shared=0):
        lib, entry = self.kernels[func]
        words = [0, int(grid)]
        for a in args:
            words += _words(a)
        c = (ctypes.c_uint * len(words))(*words)
        lib.hemu_set_consts(c, len(words))
        lib.hemu_set_arena(self.arena.ctypes.data, len(self.arena))
        r = lib.hemu_run(entry.encode(), int(grid), int(block))
        if r != 0:
            raise RuntimeError(f"hlsl_emu: launch of {func} ({entry}) failed ({r})")
        err = lib.hemu_last_error()
        if err:
            raise RuntimeError(f"hlsl_emu: {func}: {err.decode()}")
        oob = lib.hemu_oob_count()
        if oob:
            raise RuntimeError(f"hlsl_emu: {func}: {oob} accesses outside the arena")
        self.launches += 1

    def event(self):
        return [0.0]

    def record(self, ev):
        pass

    def elapsed_ms(self, start, stop):
        return 0.0
