# SPDX-License-Identifier: MIT
"""Root-cause tracer for the step-E head (track r3-head): runs the D0-R window block of block.py twice on the same
inputs, once with the exact 51.1 step (exact sum, one f16 rounding) and once with an emulation of variant E (two
16-input partial sums, each added to an f32 accumulator with one round-to-nearest f32, then the f16 rounding; the
score step takes the bias B as the first WMMA's start value), keeping every intermediate, and reports for each
window whose y2h differs the first intermediate that differs and the step values around it.

    python -m clean.rr_f.head.trace_e <inputs.pkl> [--gpu-y2h y2h.npy]

inputs.pkl: the arguments of head.d0r_stage (dumped by runtime/tests/e2e.py --dump-head)."""
import argparse
import pickle

import numpy as np

from . import block as B
from . import fp

f64 = np.float64


def rn32(x):
    return np.asarray(x, f64).astype(np.float32).astype(f64)


def chain_exact(xv, wv, c0, trace=None):
    acc = np.asarray(c0, f64)
    with np.errstate(all="ignore"):
        for k0 in range(0, xv.shape[-1], 32):
            part = np.matmul(xv[..., k0:k0 + 32], wv[..., k0:k0 + 32, :])
            s = acc + part
            if trace is not None:
                trace.append(dict(exact=s))
            acc = fp.r16(s)
    return acc


def chain_e(xv, wv, c0, trace=None):
    acc = np.asarray(c0, f64)
    with np.errstate(all="ignore"):
        for k0 in range(0, xv.shape[-1], 32):
            p0 = np.matmul(xv[..., k0:k0 + 16], wv[..., k0:k0 + 16, :])
            p1 = np.matmul(xv[..., k0 + 16:k0 + 32], wv[..., k0 + 16:k0 + 32, :])
            d2 = rn32(rn32(acc + p0) + p1)
            if trace is not None:
                trace.append(dict(exact=acc + p0 + p1, f32=d2))
            acc = fp.r16(d2)
    return acc


def block_trace(xw, W, k, ra, m, mode):
    """block.window_block with every intermediate kept; mode 'exact' or 'E'."""
    chain = chain_exact if mode == "exact" else chain_e
    H = W["Wq"].shape[0]
    C = xw.shape[-1]
    x = fp.d8(xw)
    T = {}
    O = []
    for h in range(H):
        q = fp.d8(fp.q8(chain(x, fp.d8(W["Wq"][h]), 0.0)))
        v = fp.d8(fp.q8(chain(x, fp.d8(W["Wv"][h]), 0.0)))
        Bm = B.bias_matrix(np.asarray(W["Tb"][h], f64))
        with np.errstate(all="ignore"):
            if mode == "exact":
                s = fp.r16(np.matmul(q, np.swapaxes(q, -1, -2)) + Bm)
            else:
                p0 = np.matmul(q[..., :16], np.swapaxes(q[..., :16], -1, -2))
                p1 = np.matmul(q[..., 16:], np.swapaxes(q[..., 16:], -1, -2))
                s = fp.r16(rn32(rn32(Bm + p0) + p1))
        w = B.attention_weight(s, k)
        Z = fp.r16(np.sum(w, axis=-1))
        with np.errstate(all="ignore"):
            r = fp.r16(1.0 / Z)
        P = fp.d8(fp.q8(fp.mul16(w, r[..., None])))
        tr = []
        o = fp.d8(fp.q8(chain(P, v, 0.0, tr)))
        O.append(o)
        T.update({f"q{h}": q, f"v{h}": v, f"s{h}": s, f"w{h}": w, f"Z{h}": Z, f"P{h}": P, f"O{h}": o,
                  f"PV{h}.step0": fp.r16(tr[0].get("f32", tr[0]["exact"]))})
    Ocat = np.concatenate(O, axis=-1)
    Wo = fp.d8(np.asarray(W["Wo"]).reshape(H * 32, C))
    tr = []
    Dp = chain(Ocat, Wo, W["bo"], tr)
    T["Dp.step0"] = fp.r16(tr[0].get("f32", tr[0]["exact"]))
    T["Dp"] = Dp
    y1h = fp.fma16(Dp, ra, x)
    y1q = fp.d8(fp.q8(y1h))
    tr = []
    Du = chain(y1q, fp.d8(W["W1"]), W["b1"], tr)
    Au = fp.d8(fp.q8(B.act(Du, k)))
    tr2 = []
    Dd = chain(Au, fp.d8(W["W2"]), W["b2"], tr2)
    for i, t in enumerate(tr2[:-1]):
        T[f"Dd.step{i}"] = fp.r16(t.get("f32", t["exact"]))
    T["Dd.trace"] = tr2
    y2h = fp.fma16(Dd, m, y1h)
    T.update(y1h=y1h, y1q=y1q, Du=Du, Au=Au, Dd=Dd, y2h=y2h)
    return T


ORDER = ["q0", "v0", "q1", "v1", "s0", "s1", "w0", "w1", "Z0", "Z1", "P0", "P1", "PV0.step0", "PV1.step0", "O0",
         "O1", "Dp.step0", "Dp", "y1h", "y1q", "Du", "Au", "Dd.step0", "Dd.step1", "Dd.step2", "Dd", "y2h"]


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("--max", type=int, default=5)
    a = ap.parse_args(argv)
    d = pickle.load(open(a.inputs, "rb"))
    x, W, k, ra, m, shift = d["x"], d["W"], d["k"], d["ra"], d["m"], d["shift"]
    ny, nx, C = x.shape
    Xw, Yw = B.window_coords(nx, ny, *shift)
    xw = x[B.reflect(Yw, ny), B.reflect(Xw, nx)]
    ex = block_trace(xw, W, k, ra, m, "exact")
    ee = block_trace(xw, W, k, ra, m, "E")
    inside = (Xw >= 0) & (Xw < nx) & (Yw >= 0) & (Yw < ny)
    bad = ~same(ex["y2h"], ee["y2h"]) & inside[..., None]
    wins = np.nonzero(bad.any(axis=(1, 2)))[0]
    print(f"windows {xw.shape[0]}, y2h values differing (E vs exact) {int(bad.sum())} in {len(wins)} windows; "
          f"tokens {int(bad.any(axis=2).sum())}")
    for name in ORDER:
        nd = int((~same(ex[name], ee[name])).sum())
        print(f"  {name}: {nd} differing values ({ex[name].size} total)")
    for wi in wins[:a.max]:
        t, c = np.argwhere(bad[wi])[0]
        print(f"window {wi} token {t} (x {Xw[wi, t]}, y {Yw[wi, t]}) channel {c}: y2h exact {ex['y2h'][wi, t, c]} "
              f"E {ee['y2h'][wi, t, c]}")
        for name in ORDER:
            dd = ~same(ex[name][wi], ee[name][wi])
            if dd.any():
                idx = np.argwhere(dd)
                print(f"    first differing intermediate in this window: {name}, {len(idx)} values, e.g. index "
                      f"{tuple(idx[0])}: exact {ex[name][wi][tuple(idx[0])]!r} E {ee[name][wi][tuple(idx[0])]!r}")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
