# SPDX-License-Identifier: MIT
"""Probes of the exact integer step "I" (kernels/rr_step_i.hlsli, lever L4 of results/2026-10-21-body-levers.md):

H1'  v_wmma_i32_16x16x16_iu8 (driver channel op 0x28, format 0xC) returns A B + C exactly in int32 for signed int8 A,
     B and partial sums inside int32: random and extreme operands (+-127, -128, all-max chains near 2^31) against
     int64 NumPy.
FIN  ri_fin = R(c + N 2^G) of int32 sums N, f16 start values c (subnormals, +-0, +-inf, NaN), both regimes (|N| < 2^24:
     one v_fma_mix; |N| >= 2^24: RNE32 + TwoSum + one v_fma_mix, G <= -8; G > -8 must set the flag), adversarial N
     placed at and next to every kind of f16 rounding boundary.
STEP whole steps (digits, bases, Horner, bound, rounding) against the exact 51.1 step R(c + sum of 32 products):
     mode 0 both operands activations (ri_step_aa), 1 host weight planes x activations (ri_step_w), 2 activations x
     host planes (ri_step_t); data: full range, network-like, every exponent span 0..14, sums at f16 midpoints (ties)
     and next to them, |N| >= 2^24, the int32 bound (must flag), NaN codes (must flag), +-inf / NaN / signed-zero
     starts, all-zero blocks. A value counts only when its flag is clear; flagged values go to the fixup (step X).

    python -m clean.rr_f.gpu.probe_i8 [--waves N] [--seed S] [--json out.json] [--nolock]"""
import argparse
import json
import os
import sys

import numpy as np

from clean.rr_f.gpu import build, layout
from clean.rr_f.reference.fmt import D8_TABLE, round_f16

FINITE = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)
VAL = D8_TABLE.astype(np.float64)


def acc_to_mat(d):
    nw = d.shape[0]
    return d.reshape(nw, 2, 16, 8).transpose(0, 1, 3, 2).reshape(nw, 16, 16)


def mat_to_acc(m):
    nw = m.shape[0]
    return m.reshape(nw, 2, 8, 16).transpose(0, 1, 3, 2).reshape(nw, 32, 8)


def a_frag_to_mat(f):
    """A fragment bytes [nw, 32, 8] -> [nw, 16 rows, 16 k]."""
    nw = f.shape[0]
    return f.reshape(nw, 2, 16, 8).transpose(0, 2, 1, 3).reshape(nw, 16, 16)


def b_frag_to_mat(f):
    """B fragment bytes [nw, 32, 8] -> [nw, 16 k, 16 cols]."""
    nw = f.shape[0]
    return f.reshape(nw, 2, 16, 8).transpose(0, 1, 3, 2).reshape(nw, 16, 16)


def run_kernel(entry, defines, inp, out_bytes, nlanes, lock):
    ar = build.Arena()
    pin = ar.add(inp)
    pout = ar.reserve(out_bytes)
    dx = build.compile_kernel("rr_probe_i8.hlsl", entry, defines, isa=True)
    img, _ = build.run(ar, {"k": (dx, 3)}, [("k", nlanes // 128, [pin, pout, nlanes])], lock=lock)
    return img[pout:pout + out_bytes], build.isa_text(dx)


# ---- H1' ------------------------------------------------------------------------------------------------------------
def run_wmma(nw, seed, lock):
    rng = np.random.default_rng(seed)
    A = rng.integers(-128, 128, (nw, 32, 8)).astype(np.int8)
    B = rng.integers(-128, 128, (nw, 32, 8)).astype(np.int8)
    C = rng.integers(-2 ** 30, 2 ** 30, (nw, 32, 8)).astype(np.int64)
    kinds = np.arange(nw) % 4
    for w in np.flatnonzero(kinds == 1):                 # extremes: +-127 / -128 everywhere
        A[w] = rng.choice(np.array([-128, -127, 127], np.int8), (32, 8))
        B[w] = rng.choice(np.array([-128, -127, 127], np.int8), (32, 8))
    for w in np.flatnonzero(kinds == 2):                 # all products 16384 (-128 x -128), C at 2^31 - 1 - 16 * 16384
        A[w] = -128
        B[w] = -128
        C[w] = 2 ** 31 - 1 - 16 * 16384
    for w in np.flatnonzero(kinds == 3):                 # most negative: 16 x (-128 x 127), C near -2^31
        A[w] = -128
        B[w] = 127
        C[w] = -2 ** 31 + 16 * 128 * 127
    inp = np.zeros((nw, 32, 48), np.uint8)
    inp[..., :8] = A.view(np.uint8)
    inp[..., 8:16] = B.view(np.uint8)
    inp[..., 16:] = C.astype(np.int32).view(np.uint8).reshape(nw, 32, 32)
    out, isa = run_kernel("rr_pi_wmma", {}, inp, nw * 32 * 32, nw * 32, lock)
    got = out.view(np.int32).reshape(nw, 32, 8).astype(np.int64)
    Am = a_frag_to_mat(A).astype(np.int64)
    Bm = b_frag_to_mat(B).astype(np.int64)
    want = mat_to_acc(np.einsum("wrk,wkc->wrc", Am, Bm) + acc_to_mat(C))
    bad = int((got != want).sum())
    n_wmma = len([l for l in isa.splitlines() if "v_wmma_i32_16x16x16_iu8" in l])
    return bad == 0 and n_wmma >= 1, dict(values=int(got.size), bad=bad, isa_wmma_iu8=n_wmma)


# ---- FIN ------------------------------------------------------------------------------------------------------------
F16_ALL = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
F16_FIN = F16_ALL[np.isfinite(F16_ALL)]


def exact_r16(c, N, G):
    """R(c + N 2^G) exactly (c f16 values as float64, N int64, G int): the float64 sum may round, so the exact error is
    carried as in reference/fmt.round_f16 (two-sum of the two float64 terms)."""
    from clean.rr_f.reference.fmt import two_sum
    p = np.ldexp(N.astype(np.float64), G)                 # exact: |N| < 2^31
    with np.errstate(invalid="ignore"):
        s, e = two_sum(p, c)
        r = round_f16(s, e)
    return np.where(np.isfinite(c), r, c + p)


def bits16(v):
    v = np.asarray(v, np.float64)
    nan = np.isnan(v)
    b = np.where(nan, 0.0, v).astype(np.float16).view(np.uint16)
    return np.where(nan, 0x7E00, b).astype(np.uint16), nan


def run_fin(nw, seed, lock):
    rng = np.random.default_rng(seed + 11)
    N = np.zeros((nw, 32, 8), np.int64)
    c = np.zeros((nw, 32, 8), np.float64)
    G = np.zeros(nw, np.int64)
    kinds = ("small", "big", "tie", "tiebig", "cfine", "edge", "special", "gpos")
    names = []
    for w in range(nw):
        k = kinds[w % len(kinds)]
        names.append(k)
        G[w] = rng.integers(-18, -7) if k != "gpos" else rng.integers(-7, 11)
        if k in ("small", "special", "gpos"):
            N[w] = rng.integers(-2 ** 24 + 1, 2 ** 24, (32, 8))
            c[w] = rng.choice(F16_FIN, (32, 8))
        elif k == "big":
            N[w] = rng.integers(2 ** 24, 2 ** 31 - 512, (32, 8)) * rng.choice([-1, 1], (32, 8))
            c[w] = rng.choice(F16_FIN, (32, 8))
        elif k in ("tie", "tiebig"):                       # c + N g at an f16 midpoint, or one unit of N away
            lo = 2 ** 24 if k == "tiebig" else 1
            Nm = rng.integers(lo, 2 ** 30, (32, 8)) * rng.choice([-1, 1], (32, 8))
            v = np.ldexp(Nm.astype(np.float64), int(G[w]))
            cand = rng.choice(F16_FIN[np.abs(F16_FIN) < 60000], (32, 8))
            tgt = (v + cand).astype(np.float16).astype(np.float64)       # a nearby f16 value
            up = np.nextafter(tgt.astype(np.float16), np.float16(np.inf)).astype(np.float64)
            mid = (tgt + up) / 2                                         # its upper midpoint (exact in float64)
            dN = np.round(np.ldexp(mid - cand - v, -int(G[w])))          # N + dN: as close to the midpoint as N allows
            N[w] = np.clip(Nm + dN.astype(np.int64) + rng.integers(-1, 2, (32, 8)), -2 ** 31 + 512, 2 ** 31 - 512)
            c[w] = cand
        elif k == "cfine":                                 # c with bits far below g, big N
            N[w] = rng.integers(2 ** 24, 2 ** 31 - 512, (32, 8)) * rng.choice([-1, 1], (32, 8))
            c[w] = rng.choice(F16_FIN[np.abs(F16_FIN) < 2 ** -10], (32, 8))
        elif k == "edge":                                  # |N| at 2^24 - 1, 2^24, 2^31 - 512
            N[w] = rng.choice(np.array([2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 31 - 512, 2 ** 31 - 513]), (32, 8))
            N[w] *= rng.choice([-1, 1], (32, 8))
            c[w] = rng.choice(F16_FIN, (32, 8))
        if k == "special":
            sp = rng.random((32, 8)) < 0.3
            c[w] = np.where(sp, rng.choice(np.array([np.inf, -np.inf, np.nan, 0.0, -0.0]), (32, 8)), c[w])
            N[w] = np.where(rng.random((32, 8)) < 0.2, 0, N[w])
    inp = np.zeros((nw, 32, 80), np.uint8)
    inp[..., :32] = N.astype(np.int32).view(np.uint8).reshape(nw, 32, 32)
    inp[..., 32:64] = c.astype(np.float32).view(np.uint8).reshape(nw, 32, 32)
    inp[..., 64:68] = np.broadcast_to(G.astype(np.int32)[:, None, None], (nw, 32, 1)).view(np.uint8).reshape(nw, 32, 4)
    out, isa = run_kernel("rr_pi_fin", {}, inp, nw * 32 * 20, nw * 32, lock)
    o = out.reshape(nw, 32, 20)
    got = o[..., :16].copy().view(np.uint16).reshape(nw, 32, 8)
    flag = o[..., 16:20].copy().view(np.uint32).reshape(nw, 32)
    want = exact_r16(c, N, G[:, None, None])
    wb, wnan = bits16(want)
    gnan = ((got & 0x7C00) == 0x7C00) & ((got & 0x3FF) != 0)
    ok_val = (got == wb) | (wnan & gnan)
    zero = (wb & 0x7FFF) == 0
    fl = np.broadcast_to(flag[..., None] != 0, got.shape)
    big = np.abs(N) >= 2 ** 24
    wave_big = big.reshape(nw, -1).any(axis=1)
    expect_flag = np.broadcast_to((wave_big & (G > -8))[:, None, None], got.shape)
    rep = dict(values=int(got.size), unflagged=int((~fl).sum()), bad=int((~ok_val & ~fl & ~zero).sum()),
               zero_sign_differs=int((~ok_val & ~fl & zero).sum()), flag_mismatch=int((fl != expect_flag).sum()),
               per_kind={k: int(((~ok_val & ~fl) & np.array([names[w] == k for w in range(nw)])[:, None, None]).sum())
                         for k in kinds},
               isa_fma_mix=len([l for l in isa.splitlines() if "v_fma_mix" in l]),
               isa_fma_f32=len([l for l in isa.splitlines() if "v_fma_f32" in l or "v_fmac_f32" in l]))
    return rep["bad"] == 0 and rep["zero_sign_differs"] == 0 and rep["flag_mismatch"] == 0, rep


# ---- STEP -----------------------------------------------------------------------------------------------------------
STEP_KINDS = ("full", "net", "span", "tie", "bigN", "bound", "nan", "special", "zero")


def gen_block(rng, kind, shape, role):
    """e4m3 codes of one operand block [..., 16, 32] (rows x k) for a step test kind."""
    if kind == "full":
        return rng.choice(FINITE, shape)
    if kind == "bigN":                                   # |N| >= 2^24 inside the int32 bound: E = beta + 9, a few at beta
        lo = int(rng.integers(1, 7))
        e = np.where(rng.random(shape) < 0.05, lo, lo + 9)
        code = (e << 3) | rng.integers(0, 8, shape)
        return (code | (rng.integers(0, 2, shape) << 7)).astype(np.uint8)
    if kind in ("net", "tie", "special"):
        v = rng.standard_normal(shape) * (0.25 if role == "w" else 2.0)
        return np.asarray(_q8(v), np.uint8)
    if kind == "span":                                   # exponents in a window of random span 0..14
        span = int(rng.integers(0, 15))
        lo = int(rng.integers(1, 16 - span))
        e = rng.integers(lo, lo + span + 1, shape)
        m = rng.integers(0, 8, shape)
        code = np.where(e == 1, rng.integers(1, 16, shape), (e << 3) | m)   # E = 1: normal 1.x or subnormal
        code = np.where(rng.random(shape) < 0.1, 0, code)
        return (code | (rng.integers(0, 2, shape) << 7)).astype(np.uint8)
    if kind == "bound":                                  # widest span, largest values: the int32 bound fails
        code = np.where(rng.random(shape) < 0.5, 0x7E, 0x01)
        return (code | (rng.integers(0, 2, shape) << 7)).astype(np.uint8)
    if kind == "nan":
        code = rng.choice(FINITE, shape)
        return np.where(rng.random(shape) < 0.02, 0x7F, code).astype(np.uint8)
    if kind == "zero":
        return np.where(rng.random(shape) < 0.5, 0x80, 0).astype(np.uint8)
    raise ValueError(kind)


def _q8(v):
    from clean.rr_f.reference.fmt import q8
    return q8(v)


def run_step(nw, seed, mode, lock):
    rng = np.random.default_rng(seed + 100 * mode)
    Am = np.zeros((nw, 16, 32), np.uint8)               # A [rows][k]
    Bt = np.zeros((nw, 16, 32), np.uint8)               # B^T [cols][k]
    C = np.zeros((nw, 16, 16), np.float64)
    names = []
    for w in range(nw):
        k = STEP_KINDS[w % len(STEP_KINDS)]
        names.append(k)
        Am[w] = gen_block(rng, k, (16, 32), "w" if mode == 1 else "a")
        Bt[w] = gen_block(rng, k if k != "nan" or mode != 1 else "full", (16, 32), "w" if mode == 2 else "a")
        if mode == 1 and k == "nan":
            Am[w] = gen_block(rng, "full", (16, 32), "w")
            Bt[w] = gen_block(rng, "nan", (16, 32), "a")
        if mode == 2 and k == "nan":
            Bt[w] = gen_block(rng, "full", (16, 32), "w")
            Am[w] = gen_block(rng, "nan", (16, 32), "a")
        C[w] = rng.choice(F16_FIN, (16, 16))
        if k == "special":
            sp = rng.random((16, 16)) < 0.3
            C[w] = np.where(sp, rng.choice(np.array([np.inf, -np.inf, np.nan, 0.0, -0.0]), (16, 16)), C[w])
        if k == "zero":
            C[w] = rng.choice(np.array([0.0, -0.0, 1.0, -2.0 ** -24]), (16, 16))
    av = np.where((Am & 0x7F) == 0x7F, np.nan, VAL[Am])
    bv = np.where((Bt & 0x7F) == 0x7F, np.nan, VAL[Bt])
    S = np.einsum("wrk,wck->wrc", np.nan_to_num(av), np.nan_to_num(bv))          # exact in float64
    for w in range(nw):
        if names[w] == "tie":                           # move C so that C + S sits at an f16 midpoint where possible
            tgt = (S[w] + C[w]).astype(np.float16)
            up = np.nextafter(tgt, np.float16(np.inf)).astype(np.float64)
            mid = (tgt.astype(np.float64) + up) / 2
            cand = (mid - S[w]).astype(np.float16).astype(np.float64)
            C[w] = np.where(np.isfinite(cand), cand, C[w])
    with np.errstate(invalid="ignore"):
        want = round_f16(C + S + 0.0)                  # exact: S on 2^-18, C on 2^-24, |C + S| < 2^24
        nanop = np.isnan(av).any(axis=2)[:, :, None] | np.isnan(bv).any(axis=2)[:, None, :]
    # fragments (two K-blocks)
    def afr(M, kb):                                     # [nw, 16, 32] rows x k -> A fragment of K-block kb [nw, 32, 8]
        blk = M[:, :, 16 * kb:16 * kb + 16].reshape(len(M), 16, 2, 8)
        return blk.transpose(0, 2, 1, 3).reshape(len(M), 32, 8)
    inp = np.zeros((nw, 32, 112), np.uint8)
    inp[..., 0:8], inp[..., 8:16] = afr(Am, 0), afr(Am, 1)
    inp[..., 16:24], inp[..., 24:32] = afr(Bt, 0), afr(Bt, 1)
    inp[..., 32:64] = mat_to_acc(C).astype(np.float32).view(np.uint8).reshape(nw, 32, 32)
    if mode in (1, 2):
        Wm = Am if mode == 1 else Bt
        hok = np.ones(nw, bool)
        for w in range(nw):
            pl, tb, ok = layout.int_planes(Wm[w])
            hok[w] = ok
            if not ok:
                continue
            p0, p1 = pl[0].reshape(1, 16, 32), pl[1].reshape(1, 16, 32)
            inp[w, :, 64:72], inp[w, :, 72:80] = afr(p0, 0)[0], afr(p1, 0)[0]
            inp[w, :, 80:88], inp[w, :, 88:96] = afr(p0, 1)[0], afr(p1, 1)[0]
            inp[w, :, 96:104] = tb[0, 0].astype(np.uint32).view(np.uint8)
    out, isa = run_kernel("rr_pi_step", {"RR_PI_MODE": mode}, inp, nw * 32 * 20, nw * 32, lock)
    o = out.reshape(nw, 32, 20)
    got = acc_to_mat(o[..., :16].copy().view(np.uint16).reshape(nw, 32, 8))
    flag = o[..., 16:20].copy().view(np.uint32).reshape(nw, 32)
    wflag = (flag != 0).any(axis=1)                    # the kernel marks whole windows (any lane)
    if mode in (1, 2):
        wflag = wflag | ~hok                           # host check failed: the stage would run step X
    wb, wnan = bits16(want)
    gnan = ((got & 0x7C00) == 0x7C00) & ((got & 0x3FF) != 0)
    okv = (got == wb) | (wnan & gnan)
    zero = (wb & 0x7FFF) == 0
    fl = np.broadcast_to(wflag[:, None, None], got.shape)
    per = {}
    for k in STEP_KINDS:
        m = np.array([names[w] == k for w in range(nw)])
        per[k] = dict(waves=int(m.sum()), flagged=int(wflag[m].sum()),
                      bad=int((~okv & ~fl & ~zero)[m].sum()), zero_sign=int((~okv & ~fl & zero)[m].sum()))
    must = np.array([names[w] in ("nan", "bound") for w in range(nw)])
    nan_wave = nanop.reshape(nw, -1).any(axis=1)
    rep = dict(mode=mode, values=int(got.size), unflagged_values=int((~fl).sum()),
               bad=int((~okv & ~fl & ~zero).sum()), zero_sign_differs=int((~okv & ~fl & zero).sum()),
               nan_waves_not_flagged=int((nan_wave & ~wflag).sum()), must_flag_missed=int((must & ~wflag).sum()),
               per_kind=per, isa_wmma=len([l for l in isa.splitlines() if "v_wmma_i32_16x16x16_iu8" in l]),
               isa_fma_mix=len([l for l in isa.splitlines() if "v_fma_mix" in l]))
    good = rep["bad"] == 0 and rep["zero_sign_differs"] == 0 and rep["nan_waves_not_flagged"] == 0
    if os.environ.get("RR_PI_DUMP"):
        np.savez(os.environ["RR_PI_DUMP"] + f"_m{mode}.npz", A=Am, Bt=Bt, C=C, got=got, want=wb, wflag=wflag,
                 names=np.array(names), flag=flag)
    return good, rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--waves", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--only", default="wmma,fin,step0,step1,step2")
    a = ap.parse_args(argv)
    nw = a.waves - a.waves % 4
    res, allok = {}, True
    for part in a.only.split(","):
        if part == "wmma":
            ok, rep = run_wmma(nw, a.seed, not a.nolock)
        elif part == "fin":
            ok, rep = run_fin(nw, a.seed, not a.nolock)
        else:
            ok, rep = run_step(nw, a.seed, int(part[-1]), not a.nolock)
        res[part] = dict(ok=bool(ok), **rep)
        allok &= ok
        print(part, "PASS" if ok else "FAIL", json.dumps(rep), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
