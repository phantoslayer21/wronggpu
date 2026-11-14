# SPDX-License-Identifier: MIT
"""Model L (issue #89): the folded decoder block l_swin_block_c32_dec_coef12_nc (D3D12 only; clean/dx12/kernels/l_swin.hlsl,
swin_block.hlsl SW_OUT_COEF12) against the sequence it replaces, l_swin_block_c32_dec_nc -> l_coef_pack12, byte for byte.

    python clean/l/runner/l_fold_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_fold_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

--prepare (CPU) builds into <dir>: l_swin.co, l_ops.co and swin.co (clean/l/runner/l_c32_check.py compile_kernels), the
D3D12 kernels (clean/dx12/build_dx12.py --only) and kernel_replay.exe.
--run (GPU, one GPU slot): every case is a decoder block grid H x W (both window offsets of the decoder's kind), a render
rectangle rw x rh inside it, a random block blob and coefficient scalars. On the same input bytes:
- the sequence on D3D12 (kernel_replay, two passes: the block writes y2, then l_coef_pack12 reads that file);
- the sequence on HIP (in this process);
- the folded kernel on D3D12.
Rules (as clean/l/runner/l_d3d12_check.py):
- the RAW downloaded bytes are compared (nothing is decoded and encoded again);
- every output buffer (A, C, latent) has 64 bytes of slack after the rectangle's pixels and starts filled with its run's
  own sentinel (HIP 0xA5, the D3D12 sequence 0x5A, the folded kernel 0x3C): every rectangle byte must be identical
  across the three, and every slack byte must have kept its own run's sentinel;
- a negative control (one case of the folded kernel again with half its workgroups) must be caught;
- Model M's swin_block_c96_nc (swin.co, its port already proven) runs through the same replay as a control.
Output: counts only, no timing.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_d3d12_check as LD  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402

BLOCK, COEF, FOLD = "l_swin_block_c32_dec_nc", "l_coef_pack12", "l_swin_block_c32_dec_coef12_nc"
CONTROL = "swin_block_c96_nc"
HIP_SENTINEL, D3D12_SENTINEL, FOLD_SENTINEL = LD.HIP_SENTINEL, LD.D3D12_SENTINEL, 0x3C
SLACK = 64
# Decoder block grids (H, W): sides multiples of 4 (the runtime's fh x fw), fewer windows than a workgroup's 8 in the
# last two. Each grid runs with the render rectangles below.
GRIDS = LD.L_BLOCK_GRIDS


def rects(H, W):
    """Render rectangles (rw, rh) inside the grid: all of it, odd sides short of it, about half of it."""
    return sorted({(W, H), (max(1, W - 3), max(1, H - 1)), (max(1, W // 2 + 1), max(1, H // 2 - 1))})


# FP8 codes the network does not produce but the per-pixel functions must treat as the pair does: both NaN codes,
# +-448, +-0, subnormals
SPECIAL = np.array([0x7f, 0xff, 0x7e, 0xfe, 0x00, 0x80, 0x01, 0x81, 0x07, 0x87], np.uint8)


def cases(rng):
    """(H, W, offset, rw, rh, y bytes, block blob, coefficient blob, LDecParams bytes) of every case. The cases cycle
    through four kinds: the network's range; large inputs (outputs past the +-20 clamp and tanh's saturation); special
    FP8 codes in a tenth of y (NaN outputs); extreme coefficient scalars S, T (FP16 products from subnormal to
    infinite)."""
    from clean.runner import dec0
    kind = "second"
    n = 0
    for (H, W) in GRIDS:
        for parity in (0, 1):
            off = T.block_offset(T.TABLE, kind, parity)
            for (rw, rh) in rects(H, W):
                blob = LD._params_c32(rng)
                if not LD._nc(blob):
                    continue                                   # the folded kernel is the _nc form only
                mode, n = n % 4, n + 1
                scale = 60.0 if mode == 1 else float(rng.choice([1.5, 4.0]))
                y = fp8_encode(q8(rng.normal(0, scale, (H, W, 32)))).astype(np.uint8)
                if mode == 2:
                    pick = rng.random(y.shape) < 0.1
                    y[pick] = rng.choice(SPECIAL, int(pick.sum()))
                yb = y.tobytes()
                S, T_ = float(rng.uniform(0.5, 4.0)), float(rng.uniform(0.3, 3.0))
                if mode == 3:
                    S, T_ = float(rng.choice([1e-3, 3e-3, 300.0, 1000.0])), float(rng.choice([1e-3, 50.0]))
                cw = dec0.pack({"W": np.zeros((68, 64), np.float32), "b": np.zeros(68, np.float32), "S": S, "T": T_})
                P = bytes(dec0.DecParams(rw=rw, rh=rh, fw=W, fh=H, reset=0, pad0=0, dw=2 * rw, dh=2 * rh, kx=0.5,
                                         ky=0.5, jx=0.25, jy=-0.25, pjx=0.0, pjy=0.0))
                yield H, W, off, rw, rh, yb, blob, cw, P


def nwin_of(H, W, off):
    return -(-(W + off[0]) // 4) * -(-(H + off[1]) // 4)


def out_sizes(rw, rh):
    n = rw * rh
    return [n * 8 + SLACK, n * 4 + SLACK, n * 4 + SLACK]   # A (pack12 image A), C (image C), latent


def hip_sequence(g, ops, c):
    """The block, then l_coef_pack12, on HIP: the three outputs' raw bytes (each buffer starts at HIP_SENTINEL)."""
    H, W, off, rw, rh, yb, blob, cw, P = c
    h = g.hip
    owned = []

    def buf(data):
        p = h.malloc(max(len(data), 4))
        h.upload(p, data)
        owned.append(p)
        return p

    try:
        px, pw, pcw = buf(yb), buf(blob), buf(cw)
        py2 = h.malloc(H * W * 64)
        owned.append(py2)
        h.memset(py2, HIP_SENTINEL, H * W * 64)
        outs = []
        for n in out_sizes(rw, rh):
            p = h.malloc(n)
            owned.append(p)
            h.memset(p, HIP_SENTINEL, n)
            outs.append((p, n))
        ox, oy = off
        h.launch(h.get_function(g.mods["l_swin.co"], BLOCK), -(-nwin_of(H, W, off) // 8), 256,
                 [ctypes.c_void_p(px), ctypes.c_void_p(0), ctypes.c_void_p(py2), ctypes.c_void_p(pw), ctypes.c_int(H),
                  ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)])
        ntok = (H // 2) * (W // 2)
        (pa, _), (pc, _), (pl, _) = outs
        h.launch(h.get_function(ops, COEF), -(-ntok // 128), 256,
                 [ctypes.c_void_p(py2), ctypes.c_void_p(pa), ctypes.c_void_p(0), ctypes.c_void_p(pc), ctypes.c_void_p(pl),
                  ctypes.c_void_p(pcw), (ctypes.c_ubyte * len(P)).from_buffer_copy(P)])
        h.sync()
        return [bytes(h.download(p, n)) for p, n in outs]
    finally:
        for p in owned:
            h.free(p)


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def replay(build, cdir, lines, tag):
    path = os.path.join(cdir, f"cases_{tag}.txt")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, path], env=env, capture_output=True, text=True,
                       timeout=150, creationflags=0x08000000)
    return r.returncode, r.stdout[-600:] + r.stderr[-300:]


def fold_lines(name, c, prefix="", grid=None):
    H, W, off, rw, rh = c[:5]
    nA, nC, nL = out_sizes(rw, rh)
    return [f"case {prefix}{name}_fold", f"kernel {FOLD} {grid or -(-nwin_of(H, W, off) // 8)} 256",
            f"arg buf {name}.y", "arg null", "arg null", f"arg buf {name}.w", f"arg u32 {H}", f"arg u32 {W}",
            f"arg u32 {off[0]}", f"arg u32 {off[1]}", f"arg out {nA} {prefix}{name}.fA {FOLD_SENTINEL}",
            f"arg out {nC} {prefix}{name}.fC {FOLD_SENTINEL}", f"arg out {nL} {prefix}{name}.fL {FOLD_SENTINEL}",
            f"arg buf {name}.cw", f"arg u32 {rw}", f"arg u32 {rh}", ""]


def compare(got, want, defined, sent_got, sent_want):
    """Raw bytes: identical over the first `defined` bytes; after them each buffer keeps its own sentinel."""
    a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
    if a.size != b.size or a.size < defined:
        return {"bytes": b.size, "identical": False, "differ": -1}
    d = int(np.sum(a[:defined] != b[:defined]))
    bad = int(np.sum(a[defined:] != sent_got) + np.sum(b[defined:] != sent_want))
    return {"bytes": int(b.size), "differ": d, "slack_bad": bad, "identical": d == 0 and bad == 0}


def run(build, rocm, seeds):
    g = C32.Gpu(build, rocm)
    with open(os.path.join(build, "l_ops.co"), "rb") as f:
        ops = g.hip.load_module(f.read())
    cdir = os.path.join(build, "replay-fold")
    os.makedirs(cdir, exist_ok=True)
    pass1, pass2, all_cases, neg = [], [], {}, None
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ci, c in enumerate(cases(rng)):
            name = f"s{si}_{ci}"
            H, W, off, rw, rh, yb, blob, cw, P = c
            all_cases[name] = (c, hip_sequence(g, ops, c))
            write(os.path.join(cdir, name + ".y"), yb)
            write(os.path.join(cdir, name + ".w"), blob)
            write(os.path.join(cdir, name + ".cw"), cw)
            write(os.path.join(cdir, name + ".P"), P)
            nA, nC, nL = out_sizes(rw, rh)
            grid = -(-nwin_of(H, W, off) // 8)
            # pass 1: the D3D12 block (y2) and the folded kernel
            pass1 += [f"case {name}_block", "module l_swin.co", f"kernel {BLOCK} {grid} 256", f"arg buf {name}.y",
                      "arg null", f"arg out {H * W * 64} {name}.y2 {D3D12_SENTINEL}", f"arg buf {name}.w",
                      f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {off[0]}", f"arg u32 {off[1]}", ""]
            pass1 += fold_lines(name, c)
            if neg is None and grid >= 4 and (rw, rh) == (W, H):   # the dropped windows hold rectangle pixels
                neg = name
                pass1 += fold_lines(name, c, prefix="negative_", grid=grid // 2)
            # pass 2: l_coef_pack12 on the D3D12 block's y2
            pass2 += [f"case {name}_coef", "module l_ops.co", f"kernel {COEF} {-(-((H // 2) * (W // 2)) // 128)} 256",
                      f"arg buf {name}.y2", f"arg out {nA} {name}.sA {D3D12_SENTINEL}", "arg null",
                      f"arg out {nC} {name}.sC {D3D12_SENTINEL}", f"arg out {nL} {name}.sL {D3D12_SENTINEL}",
                      f"arg buf {name}.cw", f"arg bytes {name}.P", ""]
    # control: Model M's swin_block_c96_nc through the same replay, against HIP
    rng = np.random.default_rng(seeds[0])
    ctl = []
    for ci, (H, W, off, xb, nout, blob) in enumerate(LD.block_cases(CONTROL, rng)):
        name = f"control_{ci}"
        ctl.append((name, LD.hip_block_raw(g, "swin.co", CONTROL, xb, blob, H, W, off, nout)))
        write(os.path.join(cdir, name + ".x"), xb)
        write(os.path.join(cdir, name + ".w"), blob)
        pass1 += [f"case {name}", "module swin.co", f"kernel {CONTROL} {-(-nwin_of(H, W, off) // 8)} 256",
                  f"arg buf {name}.x", "arg null", f"arg out {nout} {name}.out {D3D12_SENTINEL}", f"arg buf {name}.w",
                  f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {off[0]}", f"arg u32 {off[1]}", ""]
    e1, t1 = replay(build, cdir, pass1, "pass1")
    e2, t2 = replay(build, cdir, pass2, "pass2") if e1 == 0 else (-1, "pass 1 failed")
    res = {"replay_exit": [e1, e2], "replay_tail": [t1, t2], "cases": []}

    def read(n):
        p = os.path.join(cdir, n)
        return open(p, "rb").read() if os.path.exists(p) else b""

    for name, (c, hip) in all_cases.items():
        H, W, off, rw, rh = c[:5]
        n = rw * rh
        defined = [n * 8, n * 4, n * 4]
        r = {"case": name, "grid": [H, W], "offset": list(off), "rect": [rw, rh], "fold_vs_d3d12_seq": [],
             "fold_vs_hip_seq": [], "d3d12_seq_vs_hip_seq": []}
        for k, s in enumerate("ACL"):
            fold, seq = read(f"{name}.f{s}"), read(f"{name}.s{s}")
            r["fold_vs_d3d12_seq"].append(compare(fold, seq, defined[k], FOLD_SENTINEL, D3D12_SENTINEL))
            r["fold_vs_hip_seq"].append(compare(fold, hip[k], defined[k], FOLD_SENTINEL, HIP_SENTINEL))
            r["d3d12_seq_vs_hip_seq"].append(compare(seq, hip[k], defined[k], D3D12_SENTINEL, HIP_SENTINEL))
        r["identical"] = all(o["identical"] for key in ("fold_vs_d3d12_seq", "fold_vs_hip_seq", "d3d12_seq_vs_hip_seq")
                             for o in r[key])
        res["cases"].append(r)
    caught = None
    if neg:
        c = all_cases[neg][0]
        n = c[3] * c[4]
        outs = [compare(read(f"negative_{neg}.f{s}"), read(f"{neg}.s{s}"), d, FOLD_SENTINEL, D3D12_SENTINEL)
                for s, d in zip("ACL", (n * 8, n * 4, n * 4))]
        caught = all(o["differ"] >= 0 for o in outs) and any(not o["identical"] for o in outs)
        res["negative_control"] = {"case": neg, "half_grid": True, "caught": caught,
                                   "differ": sum(max(o["differ"], 0) for o in outs)}
    ctl_ok = [read(f"{name}.out") == want for name, want in ctl]
    res["control"] = {"kernel": CONTROL, "cases": len(ctl_ok), "identical": sum(ctl_ok)}
    res["summary"] = {"cases": len(res["cases"]), "identical": sum(c["identical"] for c in res["cases"]),
                      "rect_bytes": sum(c["rect"][0] * c["rect"][1] * 16 for c in res["cases"])}
    res["pass"] = (e1 == 0 and e2 == 0 and caught is True and res["summary"]["cases"] > 0 and
                   res["summary"]["identical"] == res["summary"]["cases"] and len(ctl_ok) > 0 and all(ctl_ok))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    os.makedirs(a.build, exist_ok=True)
    if a.prepare:
        C32.compile_kernels(a.build, a.rocm, with_swin=True)
        print("built", LD.prepare_replay(a.build, a.rocm, [BLOCK, COEF, FOLD, CONTROL]))
    if a.run:
        res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        print({k: v for k, v in res.items() if k not in ("cases", "replay_tail")})
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
