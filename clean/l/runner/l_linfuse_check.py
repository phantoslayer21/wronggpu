# SPDX-License-Identifier: MIT
"""Model L (issue #89): the fused input linears l_linear_32_32_fused (D3D12 only; clean/dx12/kernels/l_ops_linear.hlsl)
against the sequence it replaces, l_linear_32_32_relu -> l_linear_32_32_f16, byte for byte.

    python clean/l/runner/l_linfuse_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_linfuse_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

--prepare (CPU) builds into <dir>: l_ops.co and swin.co (clean/l/runner/l_c32_check.py compile_kernels), the D3D12
kernels (clean/dx12/build_dx12.py --only) and kernel_replay.exe.
--run (GPU, one GPU slot): every case is a pixel map (npix pixels, partly filled waves and workgroups among them), an
FP8 input map q and the two blobs (enc0.Win1, enc0.Win2), at three weight scales (results past 448 and past FP16's
range, results in FP16's subnormal range) and with special FP8 codes in q (NaN, +-448, +-0, subnormals): the fused
kernel's conversions are the driver's, the pair's software. On the same input bytes:
- the sequence on D3D12 (kernel_replay, two passes: the relu linear writes x1, then the f16 linear reads that file);
- the sequence on HIP (in this process);
- the fused kernel on D3D12.
The rules are clean/l/runner/l_fold_check.py's: raw bytes; the output (x2, 64 bytes a pixel) has 64 bytes of slack and
starts filled with its run's own sentinel (HIP 0xA5, the D3D12 sequence 0x5A, the fused kernel 0x3C): every pixel byte
identical across the three, every slack byte its own run's sentinel; a negative control (the fused kernel with half its
workgroups) must be caught; Model M's swin_block_c96_nc runs through the same replay as a control.
Output: counts only, no timing.
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_d3d12_check as LD  # noqa: E402
from clean.l.runner import l_d3d12_ops_check as OPS  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402
from clean.l.runner.l_fold_check import (CONTROL, D3D12_SENTINEL, FOLD_SENTINEL, HIP_SENTINEL, SLACK, compare,  # noqa: E402
                                         nwin_of, replay, write)

RELU, F16, FUSED = "l_linear_32_32_relu", "l_linear_32_32_f16", "l_linear_32_32_fused"
# (rows, cols) of the maps: the ops check's (a 35-pixel map among them: a partly filled wave), the halo maps of the
# runtime's smallest frames, 1 x 1, and one more workgroup than a whole number (129 pixels)
MAPS = OPS.MAPS + [(36, 36), (1, 1), (3, 43), (68, 132)]


def _blob(rng, scale, bias_scale):
    """enc0.Win1 / Win2 layout (bias tile vector of 32 floats, 4 fragments of 256 FP8 bytes) at a weight scale."""
    bias = rng.normal(0, bias_scale, 32).astype(np.float32).tobytes()
    return bias + fp8_encode(q8(rng.normal(0, scale, 4 * 256))).tobytes()


# FP8 codes the network does not produce but the conversions must treat as the pair does: both NaN codes, +-448, +-0,
# subnormals
SPECIAL = np.array([0x7f, 0xff, 0x7e, 0xfe, 0x00, 0x80, 0x01, 0x81, 0x07, 0x87], np.uint8)
# weight / bias scales: the ops check's; large (sums past 448 and past FP16's 65504: infinities); tiny (FP16 subnormal
# and zero results)
SCALES = [(0.25, 0.5), (24.0, 400.0), (2e-3, 1e-4)]


def cases(rng):
    """(npix, q bytes, Win1 blob, Win2 blob) of every case: each map at each weight scale, every other case with
    special FP8 codes in a tenth of q's bytes."""
    for ci, (rows, cols) in enumerate(MAPS):
        n = rows * cols
        for si, (s, b) in enumerate(SCALES):
            q = np.frombuffer(OPS._fp8_map(rng, n * 32), np.uint8).copy()
            if (ci + si) % 2:
                pick = rng.random(q.size) < 0.1
                q[pick] = rng.choice(SPECIAL, int(pick.sum()))
            yield n, q.tobytes(), _blob(rng, s, b), _blob(rng, s, b)


def hip_sequence(g, ops, c):
    """relu, then f16, on HIP: x2's raw bytes (the output starts at HIP_SENTINEL)."""
    n, q, w1, w2 = c
    h = g.hip
    owned = []

    def buf(data):
        p = h.malloc(max(len(data), 4))
        h.upload(p, data)
        owned.append(p)
        return p

    try:
        pq, pw1, pw2 = buf(q), buf(w1), buf(w2)
        px1, py = h.malloc(n * 32), h.malloc(n * 64 + SLACK)
        owned += [px1, py]
        h.memset(px1, HIP_SENTINEL, n * 32)
        h.memset(py, HIP_SENTINEL, n * 64 + SLACK)
        grid = -(-n // 128)
        for k, a, b, wb in ((RELU, pq, px1, pw1), (F16, px1, py, pw2)):
            h.launch(h.get_function(ops, k), grid, 256,
                     [ctypes.c_void_p(a), ctypes.c_void_p(b), ctypes.c_void_p(wb), ctypes.c_int(n)])
        h.sync()
        return bytes(h.download(py, n * 64 + SLACK))
    finally:
        for p in owned:
            h.free(p)


def fused_lines(name, n, prefix="", grid=None):
    return [f"case {prefix}{name}_fused", f"kernel {FUSED} {grid or -(-n // 128)} 256", f"arg buf {name}.q",
            f"arg out {n * 64 + SLACK} {prefix}{name}.fy {FOLD_SENTINEL}", f"arg buf {name}.w1", f"arg buf {name}.w2",
            f"arg u32 {n}", ""]


def run(build, rocm, seeds):
    g = C32.Gpu(build, rocm)
    with open(os.path.join(build, "l_ops.co"), "rb") as f:
        ops = g.hip.load_module(f.read())
    cdir = os.path.join(build, "replay-linfuse")
    os.makedirs(cdir, exist_ok=True)
    pass1, pass2, all_cases, neg = [], [], {}, None
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ci, c in enumerate(cases(rng)):
            name = f"s{si}_{ci}"
            n, q, w1, w2 = c
            all_cases[name] = (n, hip_sequence(g, ops, c))
            write(os.path.join(cdir, name + ".q"), q)
            write(os.path.join(cdir, name + ".w1"), w1)
            write(os.path.join(cdir, name + ".w2"), w2)
            grid = -(-n // 128)
            # pass 1: the D3D12 relu linear (x1) and the fused kernel
            pass1 += [f"case {name}_relu", "module l_ops.co", f"kernel {RELU} {grid} 256", f"arg buf {name}.q",
                      f"arg out {n * 32} {name}.x1 {D3D12_SENTINEL}", f"arg buf {name}.w1", f"arg u32 {n}", ""]
            pass1 += fused_lines(name, n)
            if neg is None and grid >= 4:
                neg = name
                pass1 += fused_lines(name, n, prefix="negative_", grid=grid // 2)
            # pass 2: the D3D12 f16 linear on that x1
            pass2 += [f"case {name}_f16", "module l_ops.co", f"kernel {F16} {grid} 256", f"arg buf {name}.x1",
                      f"arg out {n * 64 + SLACK} {name}.sy {D3D12_SENTINEL}", f"arg buf {name}.w2", f"arg u32 {n}", ""]
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

    for name, (n, hip) in all_cases.items():
        fused, seq = read(f"{name}.fy"), read(f"{name}.sy")
        r = {"case": name, "npix": n,
             "fused_vs_d3d12_seq": compare(fused, seq, n * 64, FOLD_SENTINEL, D3D12_SENTINEL),
             "fused_vs_hip_seq": compare(fused, hip, n * 64, FOLD_SENTINEL, HIP_SENTINEL),
             "d3d12_seq_vs_hip_seq": compare(seq, hip, n * 64, D3D12_SENTINEL, HIP_SENTINEL)}
        r["identical"] = all(r[k]["identical"] for k in ("fused_vs_d3d12_seq", "fused_vs_hip_seq", "d3d12_seq_vs_hip_seq"))
        res["cases"].append(r)
    caught = None
    if neg:
        n = all_cases[neg][0]
        o = compare(read(f"negative_{neg}.fy"), read(f"{neg}.sy"), n * 64, FOLD_SENTINEL, D3D12_SENTINEL)
        caught = o["differ"] >= 0 and not o["identical"]
        res["negative_control"] = {"case": neg, "half_grid": True, "caught": caught, "differ": max(o["differ"], 0)}
    ctl_ok = [read(f"{name}.out") == want for name, want in ctl]
    res["control"] = {"kernel": CONTROL, "cases": len(ctl_ok), "identical": sum(ctl_ok)}
    res["summary"] = {"cases": len(res["cases"]), "identical": sum(c["identical"] for c in res["cases"]),
                      "pixel_bytes": sum(c["npix"] * 64 for c in res["cases"])}
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
        print("built", LD.prepare_replay(a.build, a.rocm, [RELU, F16, FUSED, CONTROL]))
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
