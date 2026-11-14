# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the matrix kernels of clean/l/kernels/l_ops.hip against the oracle steps
(clean/l/reference/oracle.py), synthetic data only.

    python clean/l/runner/l_ops_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--seeds N]
        --json out.json

<dir>/l_ops.co comes from clean/l/runner/l_c32_check.py --compile-only. The GPU part runs in a GPU slot. Output:
counts only, no timing.

Per kernel and case: the values that differ from the oracle step, and Model M's per-stage rule (clean/l/runner/
l_c32_check.py passes) on the FP8 values (on the FP8-rounded values for the FP16 output, whose FP16 differences are
reported too):
- l_linear_32_32_relu: q8(max(0, RN16(b + Win1 q))) on the value map (20 values, 12 zero channels);
- l_linear_32_32_f16: RN16(b + Win2 x), FP16;
- l_project_gather: the 2 x 2 gather and the 128 -> 64 projection (running FP16 sum per group of 32, clamp, FP8);
- l_dec_project: the 64 -> 4 x 32 projection scattered to the pixel map;
- l_coef_pack12 (pack12 store form): its images, unpacked, byte-identical to l_coef's A, B, C and latent, with
  sentinel-filled buffers (no byte past a buffer written, every pixel inside R written) and a negative control;
- l_coef: Model L's per-pixel functions on an FP16 map against oracle.coefficients_l: stored codes of A, B and the
  latent within one step, C within 4 FP16 ulp (the oracle's exponential and tanh are exact; the kernel's use the
  hardware operations), with the counts that differ at all.
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

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402

CASES = [(16, 24), (32, 64), (8, 8), (64, 96)]      # pixel maps (rows, cols), even sides


class Ops:
    def __init__(self, build, rocm):
        from clean.runner.hip import Hip
        self.hip = Hip(rocm)
        with open(os.path.join(build, "l_ops.co"), "rb") as f:
            self.mod = self.hip.load_module(f.read())

    def run(self, kernel, xb, out_bytes, blob, ints, n):
        h = self.hip
        px, pw, po = h.malloc(len(xb)), h.malloc(len(blob)), h.malloc(out_bytes)
        h.upload(px, xb)
        h.upload(pw, blob)
        h.memset(po, 0, out_bytes)
        args = [ctypes.c_void_p(px), ctypes.c_void_p(po), ctypes.c_void_p(pw)] + [ctypes.c_int(v) for v in ints]
        h.launch(h.get_function(self.mod, kernel), -(-n // 128), 256, args)
        h.sync()
        out = h.download(po, out_bytes)
        for p in (px, pw, po):
            h.free(p)
        return out


COEF_CASES = [(32, 64, 30, 51), (16, 32, 16, 32), (64, 96, 61, 77)]   # (fh, fw, rh, rw)


def coef_case(g, rng, dec, fh, fw, rh, rw, y2=None, ref_y2=None):
    """l_coef on an FP16 map y2 [fh][fw][32] (default: random) against oracle.coefficients_l of ref_y2 (default: y2):
    stored codes that differ. g: an object with .hip and .mod (l_ops.co)."""
    from clean.runner.dec0 import DecParams
    if y2 is None:
        y2 = L.f16(rng.normal(0, 2.0, (fh, fw, 32)))
    ref_y2 = y2 if ref_y2 is None else ref_y2
    h = g.hip
    yb = y2.astype(np.float16).tobytes()
    blob = LP.pack_coef(dec)
    npx = rh * rw
    py, pw = h.malloc(len(yb)), h.malloc(len(blob))
    outs = [h.malloc(npx * n) for n in (4, 4, 8, 4)]
    h.upload(py, yb)
    h.upload(pw, blob)
    for p, n in zip(outs, (4, 4, 8, 4)):
        h.memset(p, 0, npx * n)
    P = DecParams(rw=rw, rh=rh, fw=fw, fh=fh)
    args = [ctypes.c_void_p(py)] + [ctypes.c_void_p(p) for p in outs] + [ctypes.c_void_p(pw), P]
    h.launch(h.get_function(g.mod, "l_coef"), -(-((fh // 2) * (fw // 2)) // 128), 256, args)
    h.sync()
    A, B, C, Lt = (h.download(p, npx * n) for p, n in zip(outs, (4, 4, 8, 4)))
    for p in [py, pw] + outs:
        h.free(p)
    a = np.asarray(ref_y2)[:rh, :rw][..., list(L._table.L_DEC_VALUES)].astype(np.float32)
    ref = L.coefficients_l(a, dec["S"], dec["T"])
    got = {"codes_A": np.frombuffer(A, np.uint8).reshape(rh, rw, 4)[..., :3],
           "codes_B": np.frombuffer(B, np.uint8).reshape(rh, rw, 4)[..., :3],
           "C": np.frombuffer(C, np.float16).reshape(rh, rw, 4)[..., :3],
           "codes_L": np.frombuffer(Lt, np.int8).reshape(rh, rw, 4)}
    m = {"case": [fh, fw, rh, rw]}
    for k in ("codes_A", "codes_B", "codes_L"):
        d = np.abs(got[k].astype(int) - np.asarray(ref[k]).astype(int))
        m[k] = {"values": int(d.size), "differ": int(np.sum(d > 0)), "max": int(d.max())}
    s = C32.f16_steps(got["C"], np.asarray(ref["C"]).astype(np.float16))
    m["C"] = {"values": int(s.size), "differ": int(np.sum(s > 0)), "max_f16_steps": int(s.max())}
    # Codes within one step of the oracle and FP16 C within a few ulp (the oracle's exp / tanh are exact in float64).
    m["pass"] = all(m[k]["max"] <= 1 for k in ("codes_A", "codes_B", "codes_L")) and m["C"]["max_f16_steps"] <= 4
    return m


def _fp16_ties(rng, n):
    """Values exactly halfway between neighbouring FP16 numbers over the whole exponent range, both signs."""
    e = rng.integers(-14, 16, n)
    m = rng.integers(0, 1024, n)
    v = (1.0 + (m + 0.5) / 1024.0) * np.exp2(e.astype(np.float64))
    return np.where(rng.random(n) < 0.5, -v, v)


def _fp8_ties_and_saturation():
    """Midpoints between neighbouring positive FP8 E4M3 values, values above 448, negatives and zero."""
    pos = np.unique(fp8_decode(np.arange(0, 127, dtype=np.uint8)))
    mids = (pos[:-1] + pos[1:]) / 2.0
    extra = np.array([448.0, 449.0, 464.0, 480.0, 1000.0, 65504.0, -0.25, -448.0, -1000.0, 0.0])
    return np.concatenate([mids, extra])


def boundary_cases(g, rng, enc):
    """Rounding boundaries of the two input linears, driven through the bias with zero input (acc = b exactly): FP16
    ties and overflow (l_linear_32_32_f16), FP8 ties and saturation after RN16 (l_linear_32_32_relu); then the FP16
    linear on extreme inputs (+-448) with large weights. Output bits must equal the oracle's (one rounding each)."""
    out = []
    H, W = 4, 32
    npix = H * W
    zero = bytes(npix * 32)
    W0 = np.zeros((32, 32))
    for kernel, values, mode, nbytes in (
            ("l_linear_32_32_f16", np.concatenate([_fp16_ties(rng, 256 - 3), [65520.0, 70000.0, -1.0e5]]), "none", 64),
            ("l_linear_32_32_relu", _fp8_ties_and_saturation(), "relu_q8", 32)):
        bad = 0
        vals = np.concatenate([values, np.zeros((-len(values)) % 32)])
        for k in range(0, len(vals), 32):
            b = vals[k:k + 32].astype(np.float32)
            raw = g.run(kernel, zero, npix * nbytes, LP.pack_linear(W0, b, 32, 32), [npix], npix)
            ref = L.linear(np.zeros((32,)), W0, b.astype(np.float64), "f16", mode)
            if mode == "none":
                got = np.frombuffer(raw, np.float16).reshape(npix, 32)
                want = np.broadcast_to(ref.astype(np.float16), got.shape)
            else:
                got = np.frombuffer(raw, np.uint8).reshape(npix, 32)
                want = np.broadcast_to(fp8_encode(ref), got.shape)
            bad += int(np.sum(got.view(np.uint8 if mode != "none" else np.uint16)
                              != want.view(np.uint8 if mode != "none" else np.uint16)))
        out.append({"case": kernel + " bias boundaries", "values": int(len(values)) * npix, "differ_bits": bad,
                    "pass": bad == 0})
    x = np.where(rng.random((H, W, 32)) < 0.5, -448.0, 448.0)
    Wx = q8(rng.normal(0, 4.0, (32, 32)))
    bx = L.f16(rng.normal(0, 100.0, 32))
    raw = g.run("l_linear_32_32_f16", fp8_encode(x).tobytes(), npix * 64, LP.pack_linear(Wx, bx, 32, 32), [npix], npix)
    got = np.frombuffer(raw, np.float16).reshape(H, W, 32).astype(np.float64)
    ref = L.linear(x, Wx, bx, "f16", "none")
    m = C32.f16_metrics(got, ref)
    m.update({"case": "l_linear_32_32_f16 extreme inputs", "overflow_values": int(np.sum(~np.isfinite(ref)))})
    m["pass"] = C32.f16_passes(m, max_steps=1)
    out.append(m)
    return out


SENTINEL = 0xA5
SLACK = 64


def _launch_coef(g, kernel, yb, blob, sizes, P, grid, null_b=False):
    """Run an l_coef form with output buffers of the given byte sizes (A, B, C, latent), each followed by SLACK bytes,
    all filled with SENTINEL first. Returns the downloaded buffers including their slack."""
    h = g.hip
    py, pw = h.malloc(len(yb)), h.malloc(len(blob))
    h.upload(py, yb)
    h.upload(pw, blob)
    outs = []
    for n in sizes:
        if n is None:
            outs.append(None)
            continue
        p = h.malloc(n + SLACK)
        h.memset(p, SENTINEL, n + SLACK)
        outs.append(p)
    args = [ctypes.c_void_p(py)] + [ctypes.c_void_p(p or 0) for p in outs] + [ctypes.c_void_p(pw), P]
    h.launch(h.get_function(g.mod, kernel), grid, 256, args)
    h.sync()
    res = [None if p is None else np.frombuffer(h.download(p, n + SLACK), np.uint8) for p, n in zip(outs, sizes)]
    for p in [py, pw] + [p for p in outs if p]:
        h.free(p)
    return res


def unpack12(a8, c4):
    """pack12 images (image A: 2 words per pixel, image C: 1 word) -> (A codes word, B codes word, C words [C0C1,
    C2]) in the base form's layout (padding bytes 0)."""
    w0, w1 = a8[0::2], a8[1::2]
    A = w0 & 0xFFFFFF
    B = (w0 >> 24) | ((w1 & 0xFFFF) << 8)
    C = np.stack([c4, w1 >> 16], axis=-1).reshape(-1)
    return A, B, C


def coef_pack12_case(g, rng, dec, fh, fw, rh, rw):
    """l_coef and l_coef_pack12 on the same FP16 map: the pack12 images, unpacked, must equal the base form's A, B,
    C and latent byte for byte; no byte past any buffer may be written; every pixel inside R must be written (the base
    form's padding bytes are 0, not the sentinel); and a one-bit change in the pack12 image must be caught (negative
    control)."""
    from clean.runner.dec0 import DecParams
    y2 = L.f16(rng.normal(0, 2.0, (fh, fw, 32)))
    yb = y2.astype(np.float16).tobytes()
    blob = LP.pack_coef(dec)
    npx = rh * rw
    P = DecParams(rw=rw, rh=rh, fw=fw, fh=fh)
    grid = -(-((fh // 2) * (fw // 2)) // 128)
    base = _launch_coef(g, "l_coef", yb, blob, (4 * npx, 4 * npx, 8 * npx, 4 * npx), P, grid)
    p12 = _launch_coef(g, "l_coef_pack12", yb, blob, (8 * npx, None, 4 * npx, 4 * npx), P, grid)
    m = {"case": [fh, fw, rh, rw], "pixels": npx}
    m["slack_untouched"] = all(bool((b[-SLACK:] == SENTINEL).all()) for b in base + [p for p in p12 if p is not None])
    w = lambda b, n: b[:n].view(np.uint32)  # noqa: E731
    bA, bB, bC, bL = w(base[0], 4 * npx), w(base[1], 4 * npx), w(base[2], 8 * npx), base[3][:4 * npx]
    m["base_all_written"] = bool(((bA >> 24) == 0).all() and ((bB >> 24) == 0).all() and ((bC[1::2] >> 16) == 0).all())
    a8, c4, pL = w(p12[0], 8 * npx), w(p12[2], 4 * npx), p12[3][:4 * npx]
    uA, uB, uC = unpack12(a8, c4)
    m["A_identical"] = bool(np.array_equal(uA, bA))
    m["B_identical"] = bool(np.array_equal(uB, bB))
    m["C_identical"] = bool(np.array_equal(uC, bC))
    m["latent_identical"] = bool(np.array_equal(pL, bL))
    bad = a8.copy()
    k = int(rng.integers(0, npx))
    bad[2 * k + 1] ^= np.uint32(1 << 20)                              # one bit of C2 of pixel k
    nA, nB, nC = unpack12(bad, c4)
    m["negative_control_caught"] = not np.array_equal(nC, bC)
    m["pass"] = all(m[k_] for k_ in ("slack_untouched", "base_all_written", "A_identical", "B_identical", "C_identical",
                                     "latent_identical", "negative_control_caught"))
    return m


def _fp8_metrics(got, ref, case):
    m = C32.metrics(got, ref)
    m.update({"case": case, "pass": C32.passes(m)})
    return m


def run(build, rocm, seeds=(89,)):
    g = Ops(build, rocm)
    res = {k: [] for k in ("relu", "f16", "project", "dec_project")}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        params = L.random_params(rng)
        enc, dec = params["enc0"], params["dec0"]
        b_w1 = LP.pack_linear(enc["Win1"], enc["bin1"], 32, 32)
        b_w2 = LP.pack_linear(enc["Win2"], enc["bin2"], 32, 32)
        b_pj = LP.pack_project(enc["W"], enc["b"], (32, 32, 32, 32))
        b_dp = LP.pack_dec_project(dec["W"], dec["b"])
        for H, W in CASES:
            npix = H * W
            q = np.zeros((H, W, 32))
            q[..., :20] = q8(rng.normal(0, 2.0, (H, W, 20)))
            raw = g.run("l_linear_32_32_relu", fp8_encode(q).tobytes(), npix * 32, b_w1, [npix], npix)
            got = fp8_decode(np.frombuffer(raw, np.uint8).reshape(H, W, 32))
            ref = L.linear(q[..., :20], enc["Win1"], enc["bin1"], "f16", "relu_q8")
            res["relu"].append(_fp8_metrics(got, ref, [H, W]))

            x1 = q8(np.abs(rng.normal(0, 1.5, (H, W, 32))))
            raw = g.run("l_linear_32_32_f16", fp8_encode(x1).tobytes(), npix * 64, b_w2, [npix], npix)
            got = np.frombuffer(raw, np.float16).reshape(H, W, 32).astype(np.float64)
            ref = L.linear(x1, enc["Win2"], enc["bin2"], "f16", "none")
            m = _fp8_metrics(q8(got), q8(ref), [H, W])
            m.update(C32.f16_metrics(got, ref))
            m["pass"] = m["pass"] and C32.f16_passes(m, max_steps=1)      # one rounding of the accumulator
            res["f16"].append(m)

            x3 = q8(rng.normal(0, 1.5, (H, W, 32)))
            tw, th = W // 2, H // 2
            raw = g.run("l_project_gather", fp8_encode(x3).tobytes(), tw * th * 64, b_pj, [tw, th], tw * th)
            got = fp8_decode(np.frombuffer(raw, np.uint8).reshape(th, tw, 64))
            ref = L.project_groups(L.space_to_depth(x3), enc["W"], enc["b"], (32, 32, 32, 32), 6.28125)
            res["project"].append(_fp8_metrics(got, ref, [H, W]))

            z = q8(rng.normal(0, 1.5, (th, tw, 64)))
            raw = g.run("l_dec_project", fp8_encode(z).tobytes(), npix * 32, b_dp, [tw, th], tw * th)
            got = fp8_decode(np.frombuffer(raw, np.uint8).reshape(H, W, 32))
            ref = L.depth_to_space(L.dec_project(z, dec["W"], dec["b"]))
            res["dec_project"].append(_fp8_metrics(got, ref, [H, W]))
        res.setdefault("boundary", []).extend(boundary_cases(g, rng, enc))
        for case in COEF_CASES:
            res.setdefault("coef", []).append(coef_case(g, rng, dec, *case))
            res.setdefault("coef_pack12", []).append(coef_pack12_case(g, rng, dec, *case))
    res["coef_total"] = {k: {"values": sum(m[k]["values"] for m in res["coef"]),
                             "differ": sum(m[k]["differ"] for m in res["coef"])}
                         for k in ("codes_A", "codes_B", "C", "codes_L")}
    for k in ("relu", "f16", "project", "dec_project"):
        res[k + "_total"] = {"cases": len(res[k]), "values": sum(m["values"] for m in res[k]),
                             "differ": sum(m["differ"] for m in res[k]),
                             "max_steps": max(m["max_steps"] for m in res[k])}
    res["f16_total"]["f16_differ"] = sum(m["f16_differ"] for m in res["f16"])
    kinds = ("relu", "f16", "project", "dec_project", "coef", "boundary", "coef_pack12")
    p12 = res["coef_pack12"]
    res["coef_pack12_total"] = {"cases": len(p12), "pixels": sum(m["pixels"] for m in p12),
                                "all_pass": all(m["pass"] for m in p12)}
    res["pass"] = all(m["pass"] for k in kinds for m in res[k])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--seeds", type=int, default=1, help="run the case list with this many seeds (89, 90, ...)")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k.endswith("_total") or k == "pass"})
    for k in ("relu", "f16", "project", "dec_project", "coef", "boundary", "coef_pack12"):
        for m in res[k]:
            if not m["pass"]:
                print("FAIL", k, json.dumps(m))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
