# SPDX-License-Identifier: MIT
"""Probe P1 (DESIGN-51.md 7.2): how v_wmma_f32_16x16x16_fp8_fp8 accumulates, and how often the 51.1 matrix step done
as two WMMAs + one f16 conversion differs from the exact step R(C + sum of 32 products).

    python -m clean.rr_f.gpu.probe_p1 [--waves N] [--nolock]

Models checked per WMMA: RN (exact sum of the 16 products and C, rounded once to f32, nearest even)."""
import argparse
import sys

import numpy as np

from clean.rr_f.gpu import build
from clean.rr_f.reference.fmt import D8_TABLE, round_f16, two_sum

FINITE = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)


def rn32(s, e):
    """Exact RN to f32 of s + e (float64 s, exact error e; no overflow)."""
    r = s.astype(np.float32)
    r64 = r.astype(np.float64)
    up = np.nextafter(r, np.float32(np.inf)).astype(np.float64)
    dn = np.nextafter(r, np.float32(-np.inf)).astype(np.float64)
    other = np.where(s > r64, up, dn)
    mid = (s != r64) & ((r64 + other) / 2 == s) & (e != 0)
    # at a midpoint with a nonzero error: the neighbour on the side of e
    pick_other = mid & (np.sign(e) == np.sign(other - r64))
    return np.where(pick_other, other, r64)


def gen(rng, nw):
    """Per wave: codes A1, B1, A2, B2 [nw, 32, 8], C [nw, 32, 8] f32, and a mode per wave."""
    modes = np.arange(nw) % 6
    A = np.empty((4, nw, 32, 8), np.uint8)
    C = np.zeros((nw, 32, 8), np.float32)
    for w in range(nw):
        m = modes[w]
        if m == 0:      # full range, C = 0
            A[:, w] = rng.choice(FINITE, size=(4, 32, 8))
        elif m == 1:    # full range, C = f16 values
            A[:, w] = rng.choice(FINITE, size=(4, 32, 8))
            C[w] = rng.standard_normal((32, 8)).astype(np.float16).astype(np.float32) * 64
        elif m == 2:    # network-like: |values| in [2^-6, 4], C f16 (bias-like)
            e = rng.integers(1, 10, size=(4, 32, 8))
            mm = rng.integers(0, 8, size=(4, 32, 8))
            s = rng.integers(0, 2, size=(4, 32, 8))
            A[:, w] = (s * 128 + e * 8 + mm).astype(np.uint8)
            C[w] = (rng.standard_normal((32, 8)) * 4).astype(np.float16).astype(np.float32)
        elif m == 3:    # mixed magnitudes: big and tiny products, C large
            big = rng.integers(12, 15, size=(4, 32, 8))
            tiny = rng.integers(0, 3, size=(4, 32, 8))
            pick = rng.integers(0, 2, size=(4, 32, 8))
            e = np.where(pick, big, tiny)
            mm = rng.integers(0, 8, size=(4, 32, 8))
            s = rng.integers(0, 2, size=(4, 32, 8))
            A[:, w] = (s * 128 + e * 8 + mm).astype(np.uint8)
            C[w] = (rng.standard_normal((32, 8)) * 2048).astype(np.float16).astype(np.float32)
        elif m == 4:    # C with full f32 mantissas
            A[:, w] = rng.choice(FINITE, size=(4, 32, 8))
            C[w] = (rng.standard_normal((32, 8)) * 1000).astype(np.float32)
        else:           # cancellation: B2 = -B1 pattern, tiny C
            A[:, w] = rng.choice(FINITE, size=(4, 32, 8))
            A[3, w] = A[1, w] ^ 0x80
            A[2, w] = A[0, w]
            C[w] = (rng.standard_normal((32, 8)) * 1e-3).astype(np.float16).astype(np.float32)
    return A, C, modes


def frag_mats(a, b):
    """A frag [nw, 32, 8] codes -> A matrix [nw, 16 rows, 16 k]; B frag -> B [nw, 16 k, 16 cols] (float64)."""
    nw = a.shape[0]
    av = D8_TABLE[a].reshape(nw, 2, 16, 8)            # [w, h, r, j]
    Am = av.transpose(0, 2, 1, 3).reshape(nw, 16, 16)  # [w, r, k = 8h + j]
    bv = D8_TABLE[b].reshape(nw, 2, 16, 8)            # [w, h, c, j]
    Bm = bv.transpose(0, 1, 3, 2).reshape(nw, 16, 16)  # [w, k, c]
    return Am, Bm


def acc_to_mat(d):
    """Accumulator [nw, 32, 8] -> [nw, 16 rows, 16 cols]: lane 16h + c, element v = row 8h + v."""
    nw = d.shape[0]
    return d.reshape(nw, 2, 16, 8).transpose(0, 1, 3, 2).reshape(nw, 16, 16)


def mat_to_acc(m):
    nw = m.shape[0]
    return m.reshape(nw, 2, 8, 16).transpose(0, 1, 3, 2).reshape(nw, 32, 8)


def prepare(ar, waves=3072, seed=1):
    """Adds P1 to an arena: returns (kernels, dispatches, analyze(img) -> ok)."""
    rng = np.random.default_rng(seed)
    nw = waves - waves % 4
    A, C, modes = gen(rng, nw)
    inp = np.zeros((nw, 32, 64), np.uint8)
    for k in range(4):
        inp[:, :, 8 * k:8 * k + 8] = A[k]
    inp[:, :, 32:64] = C.view(np.uint8).reshape(nw, 32, 32)
    pin = ar.add(inp)
    pout = ar.reserve(nw * 32 * 80)
    dx = build.compile_kernel("rr_probe.hlsl", "rr_p1_wmma", {})

    def analyze(img):
        return _analyze(img, pout, nw, A, C, modes)
    return {"p1": (dx, 3)}, [("p1", nw // 4, [pin, pout, nw * 32])], analyze


def _analyze(img, pout, nw, A, C, modes):
    out = img[pout:pout + nw * 32 * 80].reshape(nw, 32, 80)
    D1 = out[:, :, 0:32].copy().view(np.float32).reshape(nw, 32, 8).astype(np.float64)
    D2 = out[:, :, 32:64].copy().view(np.float32).reshape(nw, 32, 8).astype(np.float64)
    H = out[:, :, 64:80].copy().view(np.float16).reshape(nw, 32, 8).astype(np.float64)
    A1, B1 = frag_mats(A[0], A[1])
    A2, B2 = frag_mats(A[2], A[3])
    S1 = mat_to_acc(np.matmul(A1, B1))           # exact (multiples of 2^-18 below 2^22)
    S2 = mat_to_acc(np.matmul(A2, B2))
    Cd = C.astype(np.float64)
    s, e = two_sum(S1, Cd)
    m1 = rn32(s, e)
    s, e = two_sum(S2, D1)
    m2 = rn32(s, e)
    s, e = two_sum(S1 + S2, Cd)                  # S1 + S2 exact
    ex16 = round_f16(s, e)
    ok = True
    print("P1 mode  values   D1!=RN   D2!=RN(gpuD1)   step f16 != exact R(C + S1 + S2)   (D1 inexact)")
    for md in range(6):
        sel = modes == md
        n = int(sel.sum()) * 256
        b1 = int((D1[sel] != m1[sel]).sum())
        b2 = int((D2[sel] != m2[sel]).sum())
        fin = np.isfinite(ex16[sel])
        b3 = int(((H[sel] != ex16[sel]) & fin).sum())
        inex = int((D1[sel] != (S1[sel] + Cd[sel])).sum())
        print(f"   {md:4d} {n:9d} {b1:8d} {b2:10d} {b3:16d} {inex:22d}")
        if b1 or b2:
            ok = False
            i = np.argwhere((D1[sel] != m1[sel]))[:4]
            for w, l, v in i:
                print("     e.g. D1", repr(D1[sel][w, l, v]), "RN", repr(m1[sel][w, l, v]), "S", repr(S1[sel][w, l, v]),
                      "C", repr(Cd[sel][w, l, v]))
    print("P1 model RN (one rounding per WMMA):", "holds" if ok else "FAILS")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=3072)
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    ar = build.Arena()
    k, d, an = prepare(ar, a.waves, a.seed)
    img, _ = build.run(ar, k, d, lock=not a.nolock)
    an(img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
