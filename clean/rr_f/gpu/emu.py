# SPDX-License-Identifier: MIT
"""Model of variant E's matrix step (DESIGN-51.md 7.2) for predicting its differences from the reference.

Variant E computes a 51.1 step as two 16-deep WMMAs on an f32 accumulator: D1 = RN32(C + S0), D2 = RN32(D1 + S1),
then R(D2) (S0 / S1 = the exact sums of the products of inputs 32 s .. 32 s + 15 and 32 s + 16 .. 32 s + 31). Under
probe P1's model RN (each WMMA = exact sum rounded once to f32) this is the hardware's result. This module runs the
reference stages with that step (monkeypatched into the reference) so that GPU results of variant E can be compared
code for code with the model, separating step-precision effects from kernel errors.

    python -m clean.rr_f.gpu.emu [--stages E2,...]   # how many outputs of each vector case variant E changes"""
import argparse
import contextlib
import sys

import numpy as np

from clean.rr_f.reference import block, fmt, stages
from clean.rr_f.reference.fmt import round_f16


def _rn32(x):
    return np.asarray(x, np.float64).astype(np.float32).astype(np.float64)


def mm_e(x, W, c0=None):
    """MM(x; W; c0) with variant E's step (see the module doc)."""
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    K = x.shape[-1]
    acc = None
    with np.errstate(invalid="ignore", over="ignore"):
        for k0 in range(0, K, 32):
            p0 = np.matmul(x[..., k0:k0 + 16], W[..., k0:k0 + 16, :]) + 0.0
            p1 = np.matmul(x[..., k0 + 16:k0 + 32], W[..., k0 + 16:k0 + 32, :]) + 0.0
            if acc is None:
                acc = np.zeros(p0.shape) if c0 is None else np.broadcast_to(np.asarray(c0, dtype=np.float64), p0.shape)
            d1 = _rn32(acc + p0)          # acc + p0 exact in float64 (f16 start value, products of e4m3)
            d2 = _rn32(d1 + p1)           # f32 + exact partial sum: exact in float64 for these magnitudes
            acc = round_f16(d2)
    return acc


@contextlib.contextmanager
def variant_e():
    """Within the block, the reference uses variant E's step (also for the scores: one step with start value B)."""
    saved = (fmt.mm, block.mm, stages.mm, block.window_block, stages.window_block)
    fmt.mm = block.mm = stages.mm = mm_e
    block.window_block = stages.window_block = _window_block_e
    try:
        yield
    finally:
        fmt.mm, block.mm, stages.mm, block.window_block, stages.window_block = saved


def _window_block_e(x, xr, W, ra, m, wc, Z_override=None, trace=False):
    """block.window_block with the score step as variant E computes it (first WMMA on features 0-15 with the bias as
    start value, second on features 16-31)."""
    from clean.rr_f.reference.block import score_bias
    from clean.rr_f.reference.fmt import d8, fma16, mul16, q8
    from clean.rr_f.reference.scalar import activation, attention_weight, recip16
    Nw, T, C = x.shape
    H = W["Wq"].shape[0]
    xv = d8(x)
    Zs, Os = [], []
    for h in range(H):
        qc = q8(mm_e(xv, d8(W["Wq"][h])))
        vc = q8(mm_e(xv, d8(W["Wv"][h])))
        qv, vv = d8(qc), d8(vc)
        B = score_bias(W["Tb"][h])
        with np.errstate(invalid="ignore", over="ignore"):
            p0 = np.einsum("wai,wbi->wab", qv[..., :16], qv[..., :16]) + 0.0
            p1 = np.einsum("wai,wbi->wab", qv[..., 16:], qv[..., 16:]) + 0.0
            s = round_f16(_rn32(_rn32(B[None] + p0) + p1))
        w = attention_weight(s, wc)
        Z = round_f16(w.sum(axis=-1))
        if Z_override is not None:
            Z = np.asarray(Z_override, dtype=np.float64)[:, h, :]
        r = recip16(Z)
        P = q8(mul16(w, r[..., None]))
        O = q8(mm_e(d8(P), vv))
        Zs.append(Z)
        Os.append(O)
    info = {"Z": np.stack(Zs, axis=1)}
    Ocat = np.concatenate([d8(o) for o in Os], axis=-1)
    Dp = mm_e(Ocat, d8(W["Wo"]).reshape(H * 32, C), W["bo"])
    y1h = fma16(Dp, ra, xr)
    y1q = q8(y1h)
    Du = mm_e(d8(y1q), d8(W["W1"]), W["b1"])
    Au = q8(activation(Du, wc))
    Dd = mm_e(d8(Au), d8(W["W2"]), W["b2"])
    y2h = fma16(Dd, m, y1h)
    return q8(y2h), info


def main(argv=None):
    from clean.rr_f.gpu.stage import compare, f16b as stage_f16b
    from clean.rr_f.tests.vectors_io import load_case, load_manifest
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages")
    a = ap.parse_args(argv)
    man = load_manifest()
    st = set(a.stages.split(",")) if a.stages else None
    for case in man["cases"]:
        if st and case["stage"] not in st:
            continue
        name, inputs, W, consts, wc, shift_row, expected, inter = load_case(man, case)
        with variant_e():
            out, _ = stages.run_stage(name, inputs, W, consts, wc, shift_row, Z_override=inter["Z"])
        r = {k: compare(k, out[k] if out[k].dtype == np.uint8 else stage_f16b(out[k]), expected[k]) for k in out}
        print(f"{case['id']:22s} " + " ".join(f"{k}:{v['diff']}/{v['n']}" for k, v in r.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
