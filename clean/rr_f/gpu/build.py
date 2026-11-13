# SPDX-License-Identifier: MIT
"""Build and run helpers of the spec 51 D3D12 kernels (clean/rr_f/kernels) and their test host (rr_host.cpp).

Kernels are compiled with clean/dx12/build_dx12.py's exact arithmetic mode (DXC -Gis, intrinsics rewritten, dxv).
The host is a standalone D3D12 executor (no HIP): one arena buffer, root constants, the driver channel table.

Default build directory: <WG_BUILD_ROOT>/rr-f/kbuild (WG_BUILD_ROOT defaults to <repo>/build; override with WG_RR_BUILD)."""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RR = os.path.dirname(HERE)
KDIR = os.path.join(RR, "kernels")
REPO = os.path.dirname(os.path.dirname(RR))
sys.path.insert(0, os.path.join(REPO, "clean", "dx12"))
import build_dx12  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BUILD = os.environ.get("WG_RR_BUILD", os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(_REPO, "build")), "rr-f", "kbuild"))
# Optional GPU lock (WG_GPU_LOCK: a script run as `python <lock> --timeout <s> -- <command>`, e.g. to keep
# GPU jobs of several shells apart); unset: the command runs directly.
GPU_LOCK = os.environ.get("WG_GPU_LOCK", "")


def _locked(cmd, timeout="3600"):
    return ([sys.executable, GPU_LOCK, "--timeout", str(timeout), "--"] if GPU_LOCK else []) + list(cmd)


def build_host(out=BUILD):
    exe = os.path.join(out, "rr_host.exe")
    src = os.path.join(HERE, "rr_host.cpp")
    if os.path.exists(exe) and os.path.getmtime(exe) >= os.path.getmtime(src):
        return exe
    os.makedirs(out, exist_ok=True)
    cmd = (f'"{build_dx12.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /EHsc /std:c++17 "{src}" '
           f'/Fe"{exe}" /Fo"{os.path.join(out, "rr_host.obj")}" /link d3d12.lib dxgi.lib')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise SystemExit("building rr_host failed:\n" + r.stdout + r.stderr)
    return exe


def _key(src, entry, defines):
    h = hashlib.sha256()
    for f in sorted(os.listdir(KDIR)):
        if f.endswith((".hlsl", ".hlsli")):
            with open(os.path.join(KDIR, f), "rb") as fh:
                h.update(f.encode() + fh.read())
    for f in sorted(os.listdir(build_dx12.HLSL_DIR)):
        with open(os.path.join(build_dx12.HLSL_DIR, f), "rb") as fh:
            h.update(f.encode() + fh.read())
    h.update(json.dumps([os.path.basename(src), entry, sorted(defines.items())]).encode())
    return h.hexdigest()[:16]


def _pp_key(src, entry, defines):
    """Cache key of a kernel from its PREPROCESSED source (dxc -P, #line markers dropped), entry, defines and the
    compiler arguments: an edit of one kernel file no longer invalidates every other kernel (2026-10-22)."""
    dxc = os.path.join(build_dx12.DXC_DIR, "dxc.exe")
    with tempfile.TemporaryDirectory(prefix="wgpp_") as tmp:
        pp = os.path.join(tmp, "p.hlsl")
        cmd = [dxc, "-T", "cs_6_6", "-HV", "2021", "-enable-16bit-types", "-P", "-Fi", pp, "-I", build_dx12.HLSL_DIR,
               "-I", KDIR, *[f"-D{k}={v}" for k, v in defines.items()], os.path.join(KDIR, src)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"dxc -P failed for {src}:\n{r.stdout}{r.stderr}")
        with open(pp, "rb") as f:
            text = b"\n".join(l.strip() for l in f.read().splitlines() if l.strip() and not l.startswith(b"#line"))
    h = hashlib.sha256(text)
    h.update(json.dumps([entry, sorted(defines.items()), build_dx12.DXC_ARGS]).encode())
    with open(os.path.join(os.path.dirname(build_dx12.__file__), "build_dx12.py"), "rb") as f:
        h.update(f.read())                                   # the listing rewrite of the exact mode
    return "p" + h.hexdigest()[:15]


def compile_kernel(src, entry, defines, out=BUILD, isa=False, tag=None):
    """Compile one entry of clean/rr_f/kernels/<src> -> dxil path (cached by the preprocessed source + defines; a DXIL
    cached under the older whole-directory key is taken over)."""
    import shutil
    os.makedirs(os.path.join(out, "dxil"), exist_ok=True)
    defs = {"WG_BLOCK": 128, "WG_WAVE": 32}
    defs.update(defines)
    tag = tag or entry
    path = os.path.join(out, "dxil", f"{tag}_{_pp_key(src, entry, defs)}.dxil")
    if not os.path.exists(path):
        old = os.path.join(out, "dxil", f"{tag}_{_key(src, entry, defs)}.dxil")
        if os.path.exists(old):
            shutil.copyfile(old, path)
        else:
            asm = build_dx12.build_asm_tool(out)
            build_dx12.compile_hlsl(os.path.join(KDIR, src), entry, path, defs, includes=(KDIR,), asm_tool=asm)
    if isa:
        stem = os.path.basename(path)[:-5]
        isa_dir = os.path.join(out, "isa")
        if not os.path.exists(os.path.join(isa_dir, stem + ".stats.csv")):
            build_dx12.rga_isa(path, isa_dir, stem)
    return path


def isa_stats(dxil, out=BUILD):
    """RGA statistics (VGPRs, LDS, scratch) of a compiled kernel."""
    stem = os.path.basename(dxil)[:-5]
    isa_dir = os.path.join(out, "isa")
    p = os.path.join(isa_dir, stem + ".stats.csv")
    if not os.path.exists(p):
        build_dx12.rga_isa(dxil, isa_dir, stem)
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def isa_text(dxil, out=BUILD):
    stem = os.path.basename(dxil)[:-5]
    p = os.path.join(out, "isa", stem + ".isa.txt")
    if not os.path.exists(p):
        build_dx12.rga_isa(dxil, os.path.join(out, "isa"), stem)
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


class Arena:
    """Byte arena laid out on the host; every allocation 256-byte aligned (offset 0 reserved: null)."""

    def __init__(self):
        self.parts = []
        self.size = 256

    def add(self, data, align=256):
        data = np.ascontiguousarray(data).view(np.uint8).ravel()
        off = (self.size + align - 1) // align * align
        self.parts.append((off, data))
        self.size = off + max(len(data), 4)
        return off

    def reserve(self, nbytes, align=256, fill=0):
        return self.add(np.full(nbytes, fill, np.uint8), align)

    def image(self):
        img = np.zeros((self.size + 255) // 256 * 256, np.uint8)
        for off, d in self.parts:
            img[off:off + len(d)] = d
        return img


def run(arena, kernels, dispatches, lock=True, reps=0, workdir=None, timeout=15000, require_double=False,
        checkpoints=(), time_first=0):
    """kernels: {id: (dxil, nargs)}; dispatches: [(id, groups, [args])]. Returns (arena bytes after the first run,
    host stdout). checkpoints: dispatch counts after which the host submits, waits and writes the whole arena to
    <workdir>/checkpoint_<k>.bin (the arena stays resident); their paths are in run.last_checkpoints."""
    exe = build_host()
    img = arena.image()
    wd = workdir or tempfile.mkdtemp(prefix="rrrun_", dir=os.path.join(BUILD))
    os.makedirs(wd, exist_ok=True)
    init = os.path.join(wd, "arena_in.bin")
    outp = os.path.join(wd, "arena_out.bin")
    img.tofile(init)
    lines = [f"arena {len(img)} {init}"]
    if require_double:
        lines.append("require_double")
    for k, (dx, na) in kernels.items():
        lines.append(f"kernel {k} {dx} {na}")
    cps = sorted(checkpoints)
    run.last_checkpoints = [os.path.join(wd, f"checkpoint_{k}.bin") for k in range(len(cps))]
    ci = 0
    for di, (k, g, args) in enumerate(dispatches):
        while ci < len(cps) and cps[ci] == di:
            lines.append(f"checkpoint {run.last_checkpoints[ci]}")
            ci += 1
        lines.append(f"dispatch {k} {g} " + " ".join(str(int(a) & 0xffffffff) for a in args))
    while ci < len(cps):
        lines.append(f"checkpoint {run.last_checkpoints[ci]}")
        ci += 1
    lines.append(f"readback {outp}")
    if reps:
        lines.append(f"time {reps} {int(time_first)}")
    job = os.path.join(wd, "job.txt")
    with open(job, "w") as f:
        f.write("\n".join(lines) + "\n")
    cmd = [exe, job]
    if lock:
        cmd = _locked(cmd, "14400")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rr_host failed ({r.returncode}):\n{r.stdout}{r.stderr}")
    return np.fromfile(outp, np.uint8), r.stdout
