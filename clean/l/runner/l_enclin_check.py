# SPDX-License-Identifier: MIT
"""Model L (issue #89): the encoder block with the two input linears as its prologue, l_swin_block_c32_enc_lin_nc
(D3D12 only; clean/dx12/kernels/swin_block.hlsl SW_SRC 3, declared in l_swin.hlsl), against the sequence it replaces,
the linears -> l_swin_block_c32_enc_nc, byte for byte.

    python clean/l/runner/l_enclin_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_enclin_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

--prepare (CPU) builds into <dir>: l_ops.co, l_swin.co and swin.co (clean/l/runner/l_c32_check.py compile_kernels), the
D3D12 kernels (clean/dx12/build_dx12.py --only) and kernel_replay.exe.
--run (GPU, one GPU slot): every case is an encoder block case of clean/l/runner/l_d3d12_check.py (its grids, fewer
windows than a workgroup's 8 among them, both window offsets, the block blob) with an FP8 q halo map [H + 4][W + 4][32]
in place of x2 and the two linear blobs (enc0.Win1, enc0.Win2) at clean/l/runner/l_linfuse_check.py's weight scales,
special FP8 codes in q in every other case; q's bytes 20..31 zero (the map's format: 20 values, 12 zero bytes), and
the fused kernel reads q's 20-byte form (lev/input_encoder_q20's layout: bytes 0..15 of every pixel, then bytes 16..19
of every pixel). On the same input bytes:
- the sequence on D3D12 (kernel_replay, two passes: l_linear_32_32_fused writes x2, then l_swin_block_c32_enc_nc reads
  that file);
- the sequence on HIP (in this process: l_linear_32_32_relu, l_linear_32_32_f16, l_swin_block_c32_enc_nc);
- the fused kernel on D3D12.
The rules are clean/l/runner/l_fold_check.py's: raw bytes; the output (x3, 32 bytes a pixel) has 64 bytes of slack and
starts filled with its run's own sentinel (HIP 0xA5, the D3D12 sequence 0x5A, the fused kernel 0x3C): every pixel byte
identical across the three, every slack byte its own run's sentinel; a negative control (the fused kernel with half
its workgroups) must be caught; Model M's swin_block_c96_nc runs through the same replay as a control.
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
from clean.l.runner import l_linfuse_check as LF  # noqa: E402
from clean.l.runner.l_fold_check import (CONTROL, D3D12_SENTINEL, FOLD_SENTINEL, HIP_SENTINEL, SLACK, compare,  # noqa: E402
                                         nwin_of, replay, write)

RELU, F16, LIN = LF.RELU, LF.F16, LF.FUSED
ENC, ENCLIN = "l_swin_block_c32_enc_nc", "l_swin_block_c32_enc_lin_nc"


def cases(rng):
    """(H, W, offset, q bytes, Win1 blob, Win2 blob, block blob, x3 bytes) of every case."""
    for ci, (H, W, off, _x2, nout, blob) in enumerate(LD.block_cases(ENC, rng)):
        npix = (H + 4) * (W + 4)
        s, b = LF.SCALES[ci % len(LF.SCALES)]
        q = np.frombuffer(OPS._fp8_map(rng, npix * 32), np.uint8).copy()
        if ci % 2:
            pick = rng.random(q.size) < 0.1
            q[pick] = rng.choice(LF.SPECIAL, int(pick.sum()))
        q = q.reshape(npix, 32)
        q[:, 20:] = 0
        yield H, W, off, q.tobytes(), LF._blob(rng, s, b), LF._blob(rng, s, b), blob, nout


def q20(q):
    """The 20-byte form of a 32-byte q map: bytes 0..15 of every pixel, then bytes 16..19 of every pixel; 12 bytes of
    slack after it (the last pixel's upper lanes load 16 bytes at its place in the second plane)."""
    a = np.frombuffer(q, np.uint8).reshape(-1, 32)
    return a[:, :16].tobytes() + a[:, 16:20].tobytes() + bytes(12)


def hip_sequence(g, ops, c):
    """relu, f16, then the encoder block, on HIP: x3's raw bytes (the output starts at HIP_SENTINEL)."""
    H, W, off, q, w1, w2, blob, nout = c
    npix = (H + 4) * (W + 4)
    h = g.hip
    owned = []

    def buf(data):
        p = h.malloc(max(len(data), 4))
        h.upload(p, data)
        owned.append(p)
        return p

    try:
        pq, pw1, pw2 = buf(q), buf(w1), buf(w2)
        px1, px2 = h.malloc(npix * 32), h.malloc(npix * 64)
        owned += [px1, px2]
        grid = -(-npix // 128)
        for k, a, o, wb in ((RELU, pq, px1, pw1), (F16, px1, px2, pw2)):
            h.launch(h.get_function(ops, k), grid, 256,
                     [ctypes.c_void_p(a), ctypes.c_void_p(o), ctypes.c_void_p(wb), ctypes.c_int(npix)])
        h.sync()
        x2 = bytes(h.download(px2, npix * 64))
    finally:
        for p in owned:
            h.free(p)
    return LD.hip_block_raw(g, "l_swin.co", ENC, x2, blob, H, W, off, nout + SLACK)


def enclin_lines(name, c, prefix="", grid=None):
    H, W, off, _q, _w1, _w2, _blob, nout = c
    return [f"case {prefix}{name}_enclin", f"kernel {ENCLIN} {grid or -(-nwin_of(H, W, off) // 8)} 256",
            f"arg buf {name}.q20", "arg null", f"arg out {nout + SLACK} {prefix}{name}.fx {FOLD_SENTINEL}",
            f"arg buf {name}.w", f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {off[0]}", f"arg u32 {off[1]}",
            f"arg buf {name}.w1", f"arg buf {name}.w2", ""]


def run(build, rocm, seeds):
    g = C32.Gpu(build, rocm)
    with open(os.path.join(build, "l_ops.co"), "rb") as f:
        ops = g.hip.load_module(f.read())
    cdir = os.path.join(build, "replay-enclin")
    os.makedirs(cdir, exist_ok=True)
    pass1, pass2, all_cases, neg = [], [], {}, None
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ci, c in enumerate(cases(rng)):
            name = f"s{si}_{ci}"
            H, W, off, q, w1, w2, blob, nout = c
            npix = (H + 4) * (W + 4)
            all_cases[name] = (nout, hip_sequence(g, ops, c))
            for ext, data in ((".q", q), (".q20", q20(q)), (".w1", w1), (".w2", w2), (".w", blob)):
                write(os.path.join(cdir, name + ext), data)
            grid = -(-nwin_of(H, W, off) // 8)
            # pass 1: the D3D12 fused linears (x2) and the fused block
            pass1 += [f"case {name}_lin", f"kernel {LIN} {-(-npix // 128)} 256", f"arg buf {name}.q",
                      f"arg out {npix * 64} {name}.x2 {D3D12_SENTINEL}", f"arg buf {name}.w1", f"arg buf {name}.w2",
                      f"arg u32 {npix}", ""]
            pass1 += enclin_lines(name, c)
            if neg is None and grid >= 4:
                neg = name
                pass1 += enclin_lines(name, c, prefix="negative_", grid=grid // 2)
            # pass 2: the D3D12 encoder block on that x2
            pass2 += [f"case {name}_enc", "module l_swin.co", f"kernel {ENC} {grid} 256", f"arg buf {name}.x2",
                      "arg null", f"arg out {nout + SLACK} {name}.sx {D3D12_SENTINEL}", f"arg buf {name}.w",
                      f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {off[0]}", f"arg u32 {off[1]}", ""]
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

    for name, (nout, hip) in all_cases.items():
        fused, seq = read(f"{name}.fx"), read(f"{name}.sx")
        r = {"case": name, "bytes": nout,
             "fused_vs_d3d12_seq": compare(fused, seq, nout, FOLD_SENTINEL, D3D12_SENTINEL),
             "fused_vs_hip_seq": compare(fused, hip, nout, FOLD_SENTINEL, HIP_SENTINEL),
             "d3d12_seq_vs_hip_seq": compare(seq, hip, nout, D3D12_SENTINEL, HIP_SENTINEL)}
        r["identical"] = all(r[k]["identical"] for k in ("fused_vs_d3d12_seq", "fused_vs_hip_seq", "d3d12_seq_vs_hip_seq"))
        res["cases"].append(r)
    caught = None
    if neg:
        nout = all_cases[neg][0]
        o = compare(read(f"negative_{neg}.fx"), read(f"{neg}.sx"), nout, FOLD_SENTINEL, D3D12_SENTINEL)
        caught = o["differ"] >= 0 and not o["identical"]
        res["negative_control"] = {"case": neg, "half_grid": True, "caught": caught, "differ": max(o["differ"], 0)}
    ctl_ok = [read(f"{name}.out") == want for name, want in ctl]
    res["control"] = {"kernel": CONTROL, "cases": len(ctl_ok), "identical": sum(ctl_ok)}
    res["summary"] = {"cases": len(res["cases"]), "identical": sum(c["identical"] for c in res["cases"]),
                      "output_bytes": sum(c["bytes"] for c in res["cases"])}
    res["pass"] = (e1 == 0 and e2 == 0 and caught is True and res["summary"]["cases"] > 0 and
                   res["summary"]["identical"] == res["summary"]["cases"] and len(ctl_ok) > 0 and all(ctl_ok))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    from clean.runner import sdk  # the HIP SDK policy: --rocm, --force-sdk
    ap.add_argument("--rocm", required=True)
    sdk.add_force_argument(ap)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    sdk.apply(a)   # --rocm: an official SDK, or a custom one with --force-sdk
    os.makedirs(a.build, exist_ok=True)
    if a.prepare:
        C32.compile_kernels(a.build, a.rocm, with_swin=True)
        print("built", LD.prepare_replay(a.build, a.rocm, [LIN, ENC, ENCLIN, CONTROL]))
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
