# SPDX-License-Identifier: MIT
"""Probe P1b: data for the accumulation model of v_wmma_f32_16x16x16_fp8_fp8 and v_wmma_f32_16x16x16_f16 (same
products as f16 operands). Sparse and dense products, controlled exponent gaps, C from 0 to wide-range f32 values.
The raw results are saved (npz) for offline model fitting (probe_p1b_fit.py)."""
import argparse
import os
import sys

import numpy as np

from clean.rr_f.gpu import build, probe_p1
from clean.rr_f.reference.fmt import D8_TABLE

FINITE = probe_p1.FINITE
NMODES = 8


def gen(rng, nw):
    modes = np.arange(nw) % NMODES
    A = np.zeros((nw, 32, 8), np.uint8)
    B = np.zeros((nw, 32, 8), np.uint8)
    C = np.zeros((nw, 32, 8), np.float32)
    for w in range(nw):
        m = modes[w]
        if m in (0, 1, 2):          # sparse: about 1-3 nonzero products per output
            dens = (0.25, 0.35, 0.5)[m]
            A[w] = np.where(rng.random((32, 8)) < dens, rng.choice(FINITE, (32, 8)), 0)
            B[w] = np.where(rng.random((32, 8)) < dens, rng.choice(FINITE, (32, 8)), 0)
        elif m == 3:                # dense, all magnitudes
            A[w] = rng.choice(FINITE, (32, 8))
            B[w] = rng.choice(FINITE, (32, 8))
        elif m == 4:                # dense, one exponent band (no alignment loss) with many mantissa bits
            e = rng.integers(12, 15, (2, 32, 8))
            mm = rng.integers(0, 8, (2, 32, 8))
            s = rng.integers(0, 2, (2, 32, 8))
            A[w] = (s[0] * 128 + e[0] * 8 + mm[0]).astype(np.uint8)
            B[w] = (s[1] * 128 + e[1] * 8 + mm[1]).astype(np.uint8)
        elif m == 5:                # one big and one tiny product per output (two-level sparse)
            A[w] = np.where(rng.random((32, 8)) < 0.3, rng.choice(FINITE, (32, 8)), 0)
            B[w] = np.where(rng.random((32, 8)) < 0.3, rng.choice(FINITE, (32, 8)), 0)
        else:                       # sparse or dense with network-like magnitudes
            dens = 0.3 if m == 6 else 1.0
            e = rng.integers(1, 10, (2, 32, 8))
            mm = rng.integers(0, 8, (2, 32, 8))
            s = rng.integers(0, 2, (2, 32, 8))
            A[w] = np.where(rng.random((32, 8)) < dens, s[0] * 128 + e[0] * 8 + mm[0], 0).astype(np.uint8)
            B[w] = np.where(rng.random((32, 8)) < dens, s[1] * 128 + e[1] * 8 + mm[1], 0).astype(np.uint8)
        # C: zero for a quarter of the lanes, else f32 with random exponent and full mantissa, or f16-like values
        kind = rng.integers(0, 4, (32, 8))
        ex = rng.integers(-24, 22, (32, 8))
        mant = rng.random((32, 8)) + 1.0
        sg = np.where(rng.random((32, 8)) < 0.5, -1.0, 1.0)
        full = (sg * mant * np.exp2(ex)).astype(np.float32)
        f16v = full.astype(np.float16).astype(np.float32)
        C[w] = np.where(kind == 0, 0.0, np.where(kind == 1, full, np.where(kind == 2, f16v, full * 1024)))
    return A, B, C, modes


def to_f16_frag(codes):
    return D8_TABLE[codes].astype(np.float16)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=4096)
    ap.add_argument("--out", default=os.path.join(build.BUILD, "p1b.npz"))
    ap.add_argument("--nolock", action="store_true")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(11)
    nw = a.waves - a.waves % 4
    A, B, C, modes = gen(rng, nw)
    inp = np.zeros((nw, 32, 80), np.uint8)
    inp[:, :, 0:8] = A
    inp[:, :, 8:16] = B
    inp[:, :, 16:32] = to_f16_frag(A).view(np.uint8).reshape(nw, 32, 16)
    inp[:, :, 32:48] = to_f16_frag(B).view(np.uint8).reshape(nw, 32, 16)
    inp[:, :, 48:80] = C.view(np.uint8).reshape(nw, 32, 32)
    ar = build.Arena()
    pin = ar.add(inp)
    pout = ar.reserve(nw * 32 * 64)
    dx = build.compile_kernel("rr_probe.hlsl", "rr_p1b_wmma", {})
    img, _ = build.run(ar, {"p": (dx, 3)}, [("p", nw // 4, [pin, pout, nw * 32])], lock=not a.nolock,
                       workdir=os.path.join(build.BUILD, "runs", "p1b"))
    out = img[pout:pout + nw * 32 * 64].reshape(nw, 32, 64)
    D8 = out[:, :, :32].copy().view(np.float32).reshape(nw, 32, 8)
    D16 = out[:, :, 32:].copy().view(np.float32).reshape(nw, 32, 8)
    np.savez(a.out, A=A, B=B, C=C, modes=modes, D8=D8, D16=D16)
    print("saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
