# SPDX-License-Identifier: MIT
"""Per-stage check of the HYBRID C32 kernels (the hybrid check design's per-stage leg): the dump variants
l_swin_block_c32_enc_hyb_dump / _dec_hyb_dump (SW_HYB_DUMP) write every stage's values; each stage is recomputed by
the oracle's FP16 form from the kernel's OWN inputs to that stage and compared:

  element-wise (norm_q7, softmax_q7, the gather and its mirrors, the output point): identical bits (NaN = NaN);
  chains (q/k, v, scores, P V, Wp, W1, W2): identical to RN16 of the f32chain oracle, except where the exact value E
    lies within delta = bound(hardware) + bound(oracle) of an FP16 rounding boundary; the kernel's value must always lie
    in [RN16(E - bound_hw), RN16(E + bound_hw)] (the measured probe-1 rule, never widened): per 8-product step of the
    kernel's K order, 9 g + ulp32(step result) / 2 with g = 2^(e - 24), e the exponent of the step's largest term (the
    incoming exact partial included); the oracle's bound is ulp32 / 2 of each of its sequential partials.
  Pass (oversight 2026-11-15 16:00Z): no bound exceedance, every difference inside the allowance, actual difference
  rate <= 1e-4 per stage; the allowance share (<= 1e-3) counts only on the real-weights run (--real).
  NaN (oversight decision on the NaN finding, permanent until the 0.6.6 fix): the random legs draw finite codes only
  (FINITE_FP8 / FINITE_F16: no FP8 NaN code, no FP16 NaN); ONE dedicated NaN-input leg (one extra seed, every case with
  the full special sets, NaN codes included) is reported separately in the record's "nan_input": listed, neither pass
  nor fail (the original keeps a NaN through enc0's ReLU + FP8 step where ours gives 0; spec 03 maps NaN inputs to 0).

Layouts (gfx12, packing.py with D rows v + 8 h): lane (t, hh), element v:
  X, Y1, Z, Y2, OUT: token t, channel hh C/2 + 8 k + v (tile k)       UQ: token t, dim 16 mt + 8 hh + v
  V: value dim 16 mt + t, token 8 hh + v                               S, P: query t, key 8 hh + v
  O: token t, dim 16 dt + 8 hh + v                                     H: token t, hidden 16 j + 8 hh + v

    python clean/l/runner/l_hyb_stage_check.py --build <dir> [--prepare] [--run] [--seeds N] [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as O  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_d3d12_check as LD  # noqa: E402
from clean.l.runner.l_hyb_pack import pack_block_hyb  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.l.runner.l_enclin_check import q20  # noqa: E402
from clean.l.runner.l_ops_check import unpack12  # noqa: E402
from clean.l.runner.l_c32_check import f16_steps  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402


# the random legs' special codes: finite only (the NaN codes 0x7f / 0xff and the FP16 NaN go to the NaN leg alone)
FINITE_FP8 = np.array([c for c in LD.SPECIAL_FP8 if c & 0x7F != 0x7F], np.uint8)
FINITE_F16 = LD.SPECIAL_F16[~np.isnan(LD.SPECIAL_F16)]


def configure(c, heads, mlp):
    """The block shape and the dump layout (swin_block.hlsl HD_*) of the kernel being checked."""
    global C, HEADS, MLP, KC, MTH, HD_HEAD, HD_Z, HD_Y2, HD_H, HD_OUT, HD_WORDS
    C, HEADS, MLP = c, heads, mlp
    KC, MTH = C // 16, MLP // 16
    HD_HEAD = 16 * KC
    HD_Z = HD_HEAD + 60 * HEADS
    HD_Y2 = HD_Z + 8 * KC
    HD_H = HD_Y2 + 8 * KC
    HD_OUT = HD_H + 8 * MTH
    HD_WORDS = HD_OUT + 8 * KC


configure(32, 2, 128)
# name: (kind, C, heads, mlp). Kinds: enc = FP16 halo map in, FP8 out (Model L's enc0); dec = RN16(E + S) in (E coarse
# mirror, S fine), FP16 out (dec0); blk = FP8 map in (fine mirror), FP8 out (enc1, enc2, b0..b5); add = RN16(E + S) in,
# FP8 out (dec2, dec1).
KERNELS = {"l_swin_block_c32_enc_hyb_dump": ("enc", 32, 2, 128), "l_swin_block_c32_dec_hyb_dump": ("dec", 32, 2, 128)}
S3_KERNELS = {"swin_block_c64_hyb_dump": ("blk", 64, 2, 256), "swin_block_c64_add_hyb_dump": ("add", 64, 2, 256),
              "swin_block_c96_hyb_dump": ("blk", 96, 4, 384), "swin_block_c96_add_hyb_dump": ("add", 96, 4, 384),
              "swin_block_c128_hyb_dump": ("blk", 128, 4, 512)}
# --set fused (speed lever 3): the hybrid twins of Model L's fusions. enclin = the encoder block with the two input
# linears as its prologue (q map in its 20-byte form + enc0.Win1 / Win2 in; its X stage = x2 = the linears' output,
# checked as the chain lin2 from the oracle's x1); deccoef = the decoder block with l_coef_pack12 as its epilogue (no
# FP16 map out: the pack12 images A / C / latent, checked against oracle.coefficients_l of the block's own output
# stage, l_ops_check's rule: codes within 1, C within 4 FP16 steps)
FUSED_KERNELS = {"l_swin_block_c32_enc_lin_hyb_dump": ("enclin", 32, 2, 128),
                 "l_swin_block_c32_dec_coef12_hyb_dump": ("deccoef", 32, 2, 128)}
FUSED_MUT = {"l_swin_block_c32_enc_lin_hyb_mutdump": ("enclin", 32, 2, 128),
             "l_swin_block_c32_dec_coef12_hyb_mutdump": ("deccoef", 32, 2, 128)}
SHARE_MAX = 1e-3     # the allowance share: judged on the real-weights recorded-frame run only (--real)
DIFF_MAX = 1e-4      # actual one-step differences per stage (oversight 2026-11-15 16:00Z: hard rule)


def f16(x):
    return O.f16(np.asarray(x, np.float64))


def same(a, b):
    """Elementwise identical FP16 values (NaN equals NaN; +0 and -0 differ)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    eq = (a == b) & (np.signbit(a) == np.signbit(b))
    return np.where(na | nb, na & nb, eq)


# ---- the dump in token / channel order ---------------------------------------------------------------------------

def lane_words(dump, win):
    """[32 lanes, HD_WORDS] float32 values of one window."""
    w = np.frombuffer(dump, np.uint32).reshape(-1, 32, HD_WORDS)[win]
    return w.view(np.float32).astype(np.float64), w


def layout_l(f, base):
    """[16 tokens, C] from a layout-L field at word `base` (KC tiles of 8)."""
    out = np.empty((16, C))
    for hh in range(2):
        for t in range(16):
            for k in range(KC):
                out[t, hh * (C // 2) + 8 * k:hh * (C // 2) + 8 * k + 8] = f[16 * hh + t, base + 8 * k:base + 8 * k + 8]
    return out


def stages(dump, win):
    f, u = lane_words(dump, win)
    d = {"X": layout_l(f, 0), "Y1": layout_l(f, 8 * KC), "Z": layout_l(f, HD_Z), "Y2": layout_l(f, HD_Y2),
         "OUT": layout_l(f, HD_OUT), "UQ": [], "V": [], "S": [], "P": [], "O": []}
    for hd in range(HEADS):
        b = HD_HEAD + 60 * hd
        uq, v, s, p, o = np.empty((16, 32)), np.empty((16, 32)), np.empty((16, 16)), np.empty((16, 16)), np.empty((16, 32))
        for hh in range(2):
            for t in range(16):
                ln = 16 * hh + t
                for mt in range(2):
                    uq[t, 16 * mt + 8 * hh:16 * mt + 8 * hh + 8] = f[ln, b + 8 * mt:b + 8 * mt + 8]
                    v[8 * hh:8 * hh + 8, 16 * mt + t] = f[ln, b + 16 + 8 * mt:b + 24 + 8 * mt]   # [token, dim]
                    o[t, 16 * mt + 8 * hh:16 * mt + 8 * hh + 8] = f[ln, b + 44 + 8 * mt:b + 52 + 8 * mt]
                s[t, 8 * hh:8 * hh + 8] = f[ln, b + 32:b + 40]
                pw = u[ln, b + 40:b + 44]
                h16 = np.stack([pw & 0xFFFF, pw >> 16], axis=-1).reshape(8).astype(np.uint16)
                p[t, 8 * hh:8 * hh + 8] = h16.view(np.float16).astype(np.float64)
        for n, a in (("UQ", uq), ("V", v), ("S", s), ("P", p), ("O", o)):
            d[n].append(a)
    hmap = np.empty((16, MLP))
    for hh in range(2):
        for t in range(16):
            for j in range(MTH):
                hmap[t, 16 * j + 8 * hh:16 * j + 8 * hh + 8] = f[16 * hh + t, HD_H + 8 * j:HD_H + 8 * j + 8]
    d["H"] = hmap
    return d


# ---- chain comparison --------------------------------------------------------------------------------------------

class Tally:
    def __init__(self):
        self.s = {}

    def add(self, stage, **kw):
        t = self.s.setdefault(stage, {"outputs": 0, "near": 0, "differ": 0, "differ_outside": 0, "violations": 0,
                                      "max_steps": 0})
        for k, v in kw.items():
            t[k] = max(t[k], v) if k == "max_steps" else t[k] + v


def order_l(K):
    """The kernel's K order of a layout-L operand (q/k, v, W1 chains): instruction kc, step 1 = channels 8 kc + j, step 2
    = channels K/2 + 8 kc + j (K slot (h, j) of group kc holds channel h K/2 + 8 kc + j)."""
    return np.array([h * (K // 2) + 8 * kc + j for kc in range(K // 16) for h in range(2) for j in range(8)])


def _half_ulp32(x):
    """ulp32(|x|) / 2 for float64 values (0 for 0)."""
    m, e = np.frexp(np.abs(x))
    return np.where(m == 0, 0.0, np.ldexp(1.0, e - 25))


def _hw_bound(s0, pk):
    """The probe-1 bound of one chain: start s0 [...], products pk [..., K] in the kernel's order."""
    K = pk.shape[-1]
    steps = pk.reshape(pk.shape[:-1] + (K // 8, 8))
    acc_in = s0[..., None] + np.concatenate([np.zeros(steps.shape[:-2] + (1,)),
                                             np.cumsum(steps.sum(axis=-1), axis=-1)[..., :-1]], axis=-1)
    big = np.maximum(np.abs(acc_in), np.abs(steps).max(axis=-1))
    return (9.0 * _half_ulp32(big) + _half_ulp32(acc_in + steps.sum(axis=-1))).sum(axis=-1)


def chain(tally, stage, start, a, Wm, got, order=None, oracle_exact=False, split=None):
    """start [..., n] (or [n]), a [..., K], Wm [n, K] (values; products exact in float64), got [..., n] (the kernel's
    RN16 chain end); order: the kernel's K order (default ascending). Records the outcome under `stage`.
    Hardware bound (probe 1): per 8-product step, 9 g (g = half ulp32 of the step's largest term, the incoming exact
    partial included) + half ulp32 of the step result. Oracle bound: half ulp32 of every sequential partial (K order)."""
    a = np.asarray(a, np.float64)
    Wm = np.asarray(Wm, np.float64)
    K = a.shape[-1]
    st = np.broadcast_to(np.asarray(start, np.float64), a.shape[:-1] + (Wm.shape[0],))
    with np.errstate(over="ignore", invalid="ignore"):
        prods = a[..., None, :] * Wm                                   # [..., n, K]
        E = st + prods.sum(axis=-1)
        fin = np.isfinite(E) & np.all(np.isfinite(prods), axis=-1) & np.isfinite(st)
        pk = np.where(fin[..., None], prods, 0.0)
        s0 = np.where(fin, st, 0.0)
        if split is None:
            b_hw = _hw_bound(s0, pk if order is None else pk[..., order])
        else:   # the order mutant 3: split[0] from the start, split[1] from 0, one float32 add at the end
            b_hw = (_hw_bound(s0, pk[..., split[0]]) + _hw_bound(np.zeros_like(s0), pk[..., split[1]])
                    + _half_ulp32(E))
        b_or = 0.0 if oracle_exact else _half_ulp32(s0[..., None] + np.cumsum(pk, axis=-1)).sum(axis=-1)
        oracle = f16(E) if oracle_exact else O.mm8(st.astype(np.float32), a, Wm, "f32chain")
        lo, hi = f16(E - b_hw), f16(E + b_hw)
        near = fin & ~same(f16(E - b_hw - b_or), f16(E + b_hw + b_or))
        got = np.asarray(got, np.float64)
        differ = ~same(got, oracle)
        inside = (got >= lo) & (got <= hi)
        viol = (fin & ~inside) | (~fin & differ)
        steps = np.where(fin & differ & np.isfinite(got) & np.isfinite(oracle),
                         np.abs(got.astype(np.float16).view(np.int16).astype(np.int64)
                                - np.asarray(oracle, np.float64).astype(np.float16).view(np.int16).astype(np.int64)), 0)
    tally.add(stage, outputs=int(got.size), near=int(near.sum()), differ=int(differ.sum()),
              differ_outside=int((differ & ~near).sum()), violations=int(viol.sum()), max_steps=int(steps.max(initial=0)))


def elementwise(tally, stage, want, got):
    ok = same(want, got)
    tally.add(stage, outputs=int(np.size(got)), differ=int((~ok).sum()), differ_outside=int((~ok).sum()),
              violations=int((~ok).sum()))


# ---- one window --------------------------------------------------------------------------------------------------

def check_window(tally, d, p, x_tokens, out_tokens, kind, mutant=False, lin=None):
    """d: stages(); p: the block's random parameters (natural values); x_tokens [16, C]: the oracle's gather of the
    block input (FP16 values); out_tokens: the kernel's output for the window's real tokens (dict token -> values).
    lin (kind enclin): (x1 tokens [16, 32], Win2 [32, 32], bin2): the X stage is the second linear's chain."""
    W = {n: q8(np.asarray(p[n], np.float64)) for n in ("Wqk", "Wv", "Wp", "W1", "W2")}
    V = {n: f16(p[n]) for n in ("bp", "b1", "b2", "B")}
    if lin is not None:
        x1t, w2, b2 = lin
        chain(tally, "lin2", f16(np.asarray(b2, np.float64)), x1t, q8(np.asarray(w2, np.float64)), d["X"],
              order=order_l(32))
    else:
        elementwise(tally, "gather", x_tokens, d["X"])
    elementwise(tally, "norm1", O.norm_q7(d["X"], O.gamma_q7(p["gamma1"])), d["Y1"])
    a8 = q8(d["Y1"])
    for hd in range(HEADS):
        chain(tally, "qk", 0.0, a8, W["Wqk"][hd], d["UQ"][hd], order=order_l(C))
        chain(tally, "v", 0.0, a8, W["Wv"][hd], d["V"][hd], order=order_l(C))
        u8 = q8(d["UQ"][hd])
        chain(tally, "scores", V["B"][hd], u8, u8, d["S"][hd])          # S[t, s] = B[t, s] + u_t . u_s
        elementwise(tally, "softmax", O.softmax_q7(d["S"][hd]), d["P"][hd])
        chain(tally, "pv", 0.0, d["P"][hd], d["V"][hd].T, d["O"][hd], oracle_exact=True)
    o8 = q8(np.concatenate([d["O"][hd] for hd in range(HEADS)], axis=-1))
    chain(tally, "wp", f16(V["bp"] + d["X"]), o8, W["Wp"], d["Z"])
    elementwise(tally, "norm2", O.norm_q7(d["Z"], O.gamma_q7(p["gamma2"])), d["Y2"])
    chain(tally, "w1", V["b1"], q8(d["Y2"]), W["W1"], d["H"], order=order_l(C))
    split = None
    if mutant == 3:   # SW_HYB_MUT_ORDER 3 (kept for the record; the probe kernels now build mutant 4): hidden groups of 32 (GROUP 2 tiles), odd groups in the second accumulator
        idx = np.arange(MLP)
        split = (idx[(idx // 32) % 2 == 0], idx[(idx // 32) % 2 == 1])
    chain(tally, "w2", f16(V["b2"] + d["Z"]), q8(O.gelu_q7(d["H"])), W["W2"], d["OUT"], split=split)
    if out_tokens:
        idx = sorted(out_tokens)
        got = np.stack([out_tokens[t] for t in idx])
        want = d["OUT"][idx]
        if kind not in ("dec", "deccoef"):
            elementwise(tally, "output_fp8", fp8_decode(fp8_encode(q8(want))), fp8_decode(got.astype(np.uint8)))
        else:
            elementwise(tally, "output_f16", want, got)


def gather(kind, xin, H, W, off, win, skip=None):
    ox, oy = off
    nwx = (W + ox + 3) // 4
    wy, wx = divmod(win, nwx)
    xt, toks = np.empty((16, C)), {}
    for t in range(16):
        pr, pc = 4 * wy - oy + t // 4, 4 * wx - ox + t % 4
        if kind == "enc":                 # FP16 halo map [H + 4][W + 4][C] at (pr + 2, pc + 2)
            xt[t] = xin[pr + 2, pc + 2].astype(np.float64)
        elif kind == "enclin":            # xin = the oracle's x1 halo map [H + 4][W + 4][32] (lin2's input)
            xt[t] = xin[pr + 2, pc + 2]
        elif kind == "blk":               # FP8 map, the fine mirror
            xt[t] = fp8_decode(xin[int(O._swin.mirror(pr, H)), int(O._swin.mirror(pc, W))]).astype(np.float64)
        else:                             # RN16(E + S): E by the coarse mirror (sub-pixel kept), S by the fine one
            m = lambda q, n: 2 * int(O._swin.mirror(q >> 1, n >> 1)) + (q & 1)  # noqa: E731
            fm = lambda q, n: int(O._swin.mirror(q, n))  # noqa: E731
            xt[t] = f16(fp8_decode(xin[m(pr, H), m(pc, W)]).astype(np.float64)
                        + fp8_decode(skip[fm(pr, H), fm(pc, W)]).astype(np.float64))
        if 0 <= pr < H and 0 <= pc < W:
            toks[t] = (pr, pc)
    return xt, toks


# ---- the fused twins' inputs and outputs (--set fused) -----------------------------------------------------------

def lin_extra(qb, lp):
    """kind enclin: the oracle's x1 halo map (hybrid form: f32 chain, RN16 at its end, ReLU, FP8) from the q codes
    qb [H + 4][W + 4][32] (bytes 20..31 zero) and the linears lp (Win1 [32, 20], bin1, Win2, bin2); the 20-byte q form
    and the two blobs for the kernel."""
    qv = fp8_decode(qb).astype(np.float64)
    x1 = O.linear(qv[..., :20], lp["Win1"], lp["bin1"], "f16", "relu_q8", "f32chain")
    return {"x1": np.asarray(x1, np.float64), "q20": q20(qb), "w2": lp["Win2"], "b2": lp["bin2"],
            "blob1": LP.pack_linear(lp["Win1"], lp["bin1"], 32, 32), "blob2": LP.pack_linear(lp["Win2"], lp["bin2"], 32, 32)}


def case_lines(cdir, n, kernel, kind, H, W, ox, oy, nwin, xin, skip, nbytes, p, ex):
    """The kernel_replay lines of one case (and its input files): the block kinds as before; enclin with the q20 map
    and the two linear blobs; deccoef with the pack12 images (A: 8, C: 4, latent: 4 bytes a pixel of the rw x rh
    rectangle) in place of the FP16 map and its coefficient blob."""
    files = [(f"w{n}.bin", pack_block_hyb(p, C, HEADS, MLP))]
    files.append((f"x{n}.bin", ex["q20"] if kind == "enclin" else xin.tobytes()))
    if skip is not None:
        files.append((f"s{n}.bin", skip.tobytes()))
    if kind == "enclin":
        files += [(f"l1{n}.bin", ex["blob1"]), (f"l2{n}.bin", ex["blob2"])]
    if kind == "deccoef":
        files.append((f"cw{n}.bin", LP.pack_coef(ex["coef"])))
    for nm, data in files:
        with open(os.path.join(cdir, nm), "wb") as f:
            f.write(data)
    out = (["arg null"] if kind == "deccoef" else [f"arg out {nbytes} o{n}.bin {LD.D3D12_SENTINEL}"])
    ln = [f"case c{n}", f"kernel {kernel} {-(-nwin // 8)} 256", f"arg buf x{n}.bin",
          f"arg buf s{n}.bin" if skip is not None else "arg null"] + out + \
         [f"arg buf w{n}.bin", f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {ox}", f"arg u32 {oy}"]
    if kind == "deccoef":
        npx = ex["rw"] * ex["rh"]
        ln += [f"arg out {8 * npx} cA{n}.bin 0", f"arg out {4 * npx} cC{n}.bin 0", f"arg out {4 * npx} cL{n}.bin 0",
               f"arg buf cw{n}.bin", f"arg u32 {ex['rw']}", f"arg u32 {ex['rh']}"]
    if kind == "enclin":
        ln += [f"arg buf l1{n}.bin", f"arg buf l2{n}.bin"]
    return ln + [f"arg out {nwin * 32 * HD_WORDS * 4} d{n}.bin 0", ""]


def coef_check(tally, cdir, n, y2, ex):
    """kind deccoef: the pack12 images against oracle.coefficients_l of the block's own output stage y2 (the dumped
    RN16 outputs, every real pixel; checked against the oracle per stage), l_ops_check coef_case's rule."""
    rw, rh = ex["rw"], ex["rh"]
    npx = rw * rh
    a8 = np.fromfile(os.path.join(cdir, f"cA{n}.bin"), np.uint32)[:2 * npx]
    c4 = np.fromfile(os.path.join(cdir, f"cC{n}.bin"), np.uint32)[:npx]
    lt = np.fromfile(os.path.join(cdir, f"cL{n}.bin"), np.int8)[:4 * npx].reshape(rh, rw, 4)
    uA, uB, uC = unpack12(a8, c4)
    ga = np.stack([(uA >> (8 * i)) & 0xFF for i in range(3)], -1).reshape(rh, rw, 3)
    gb = np.stack([(uB >> (8 * i)) & 0xFF for i in range(3)], -1).reshape(rh, rw, 3)
    cw = uC.reshape(-1, 2)
    gc = np.stack([cw[:, 0] & 0xFFFF, cw[:, 0] >> 16, cw[:, 1] & 0xFFFF], -1).astype(np.uint16).view(np.float16)
    gc = gc.reshape(rh, rw, 3)
    a = y2[:rh, :rw][..., list(T.L_DEC_VALUES)].astype(np.float32)
    ref = O.coefficients_l(a, ex["coef"]["S"], ex["coef"]["T"])
    for got, k in ((ga, "codes_A"), (gb, "codes_B"), (lt, "codes_L")):
        dd = np.abs(got.astype(int) - np.asarray(ref[k]).astype(int))
        tally.add("coef12_" + k, outputs=int(dd.size), differ=int(np.sum(dd > 0)), differ_outside=int(np.sum(dd > 1)),
                  violations=int(np.sum(dd > 1)), max_steps=int(dd.max(initial=0)))
    st = f16_steps(gc, np.asarray(ref["C"]).astype(np.float16))
    tally.add("coef12_C", outputs=int(st.size), differ=int(np.sum(st > 0)), differ_outside=int(np.sum(st > 4)),
              violations=int(np.sum(st > 4)), max_steps=int(st.max(initial=0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.path.join(r"C:\Program Files\AMD\ROCm", "7.2"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--replay-only", action="store_true", help="the GPU replay only (run it inside the GPU slot)")
    ap.add_argument("--no-replay", action="store_true", help="analyse the outputs of an earlier --replay-only (CPU)")
    ap.add_argument("--real", action="store_true", help="real L weights / recorded frames: the allowance share counts")
    ap.add_argument("--real-inputs", help="a directory of <fixture>-<frame>.npz (real L weights and recorded-frame block "
                    "inputs: x2, x3, y, parity, enc0_block, dec0_block; private, never in the repository); implies --real")
    ap.add_argument("--set", choices=("c32", "s3", "fused"), default="c32", help="c32: Model L's two blocks; s3: the "
                    "shared stages' hybrid blocks (C64, C96 with and without the skip, C128); fused: the twins of the "
                    "fusions le / lc (encoder with the linears, decoder with the pack12 epilogue)")
    ap.add_argument("--mutant", action="store_true",
                    help="the order mutant's dump kernel (SW_HYB_MUT_ORDER=2, encoder): the tolerance leg must PASS it")
    a = ap.parse_args(argv)
    if a.set == "fused":
        kernels = FUSED_MUT if a.mutant else FUSED_KERNELS
    else:
        kernels = ({"l_swin_block_c32_enc_hyb_mutdump": ("enc", 32, 2, 128)} if a.mutant
                   else S3_KERNELS if a.set == "s3" else KERNELS)
    if a.prepare:
        LD.prepare_replay(a.build, a.rocm, list(kernels))
    if not a.run:
        return 0
    rng = np.random.default_rng(13)
    cdir = os.path.join(a.build, "hyb_stage")
    os.makedirs(cdir, exist_ok=True)
    lines, meta, nan_leg = [], [], set()
    n = 0
    # the random legs (finite codes), then the one NaN-input leg (seed a.seeds, every case special, NaN codes included)
    for seed in range(a.seeds + 1):
        is_nan = seed == a.seeds
        sp8 = LD.SPECIAL_FP8 if is_nan else FINITE_FP8
        sp16 = LD.SPECIAL_F16 if is_nan else FINITE_F16
        for kernel, spec in kernels.items():
            kind = spec[0]
            configure(*spec[1:])
            tkind = "first" if kind in ("enc", "blk", "enclin") else "second"
            for gi, (H, W) in enumerate([(8, 12), (12, 8), (16, 16)]):
                for parity in (0, 1):
                    p = O.random_block_params(rng, C, HEADS, MLP)
                    off = T.block_offset(T.TABLE, tkind, parity)
                    special = is_nan or (gi + parity + seed) % 2 == 1
                    scale = 60.0 if special else 1.5
                    ex = {}
                    if kind == "enclin":   # q (FP8, 20 values + 12 zero bytes), the linears at the oracle's scales
                        lp = O.random_params(rng)["enc0"]
                        qv = np.zeros((H + 4, W + 4, 32))
                        qv[..., :20] = q8(rng.normal(0, 30.0 if special else 1.5, (H + 4, W + 4, 20)))
                        qb = fp8_encode(qv).astype(np.uint8)
                        if special:
                            pick = rng.random((H + 4, W + 4, 20)) < 0.05
                            sub = qb[..., :20]
                            sub[pick] = rng.choice(sp8, int(pick.sum()))
                            qb[..., :20] = sub
                        ex = lin_extra(qb, lp)
                        xin, nbytes, skip = ex.pop("x1"), H * W * C, None
                    elif kind == "deccoef":
                        cp = O.random_params(rng)["dec0"]
                        ex = {"coef": cp, "rw": W - (gi & 1), "rh": H - parity}
                    if kind == "enc":
                        xin = rng.normal(0, scale, (H + 4, W + 4, C)).astype(np.float16)
                        if special:
                            pick = rng.random(xin.shape) < 0.05
                            xin[pick] = rng.choice(sp16, int(pick.sum()))
                        nbytes, skip = H * W * C, None
                    elif kind == "enclin":
                        pass
                    elif kind == "blk":
                        xin = fp8_encode(q8(rng.normal(0, scale, (H, W, C)))).astype(np.uint8)
                        if special:
                            pick = rng.random(xin.shape) < 0.05
                            xin[pick] = rng.choice(sp8, int(pick.sum()))
                        nbytes, skip = H * W * C, None
                    else:
                        xin = fp8_encode(q8(rng.normal(0, scale, (H, W, C)))).astype(np.uint8)
                        skip = fp8_encode(q8(rng.normal(0, scale, (H, W, C)))).astype(np.uint8)
                        if special:
                            pick = rng.random(xin.shape) < 0.05
                            xin[pick] = rng.choice(sp8, int(pick.sum()))
                            pick = rng.random(skip.shape) < 0.05
                            skip[pick] = rng.choice(sp8, int(pick.sum()))
                        nbytes = H * W * C * (2 if kind == "dec" else 1)
                    ox, oy = off
                    nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
                    lines += case_lines(cdir, n, kernel, kind, H, W, ox, oy, nwin, xin, skip, nbytes, p, ex)
                    meta.append((n, spec, H, W, off, nwin, p, xin, skip, ex))
                    if is_nan:
                        nan_leg.add(n)
                    n += 1
    if a.real_inputs:   # the real cases replace the random ones
        a.real = True
        lines, meta, n, nan_leg = [], [], 0, set()
        for fn in sorted(f for f in os.listdir(a.real_inputs) if f.endswith(".npz")):
            z = np.load(os.path.join(a.real_inputs, fn), allow_pickle=True)
            parity = int(z["parity"])
            x3 = fp8_encode(q8(z["x3"].astype(np.float64))).astype(np.uint8)
            for kernel, spec in kernels.items():
                kind = spec[0]
                configure(*spec[1:])
                tkind = "first" if kind in ("enc", "enclin") else "second"
                p = z["enc0_block" if kind in ("enc", "enclin") else "dec0_block"][0]
                off = T.block_offset(T.TABLE, tkind, parity)
                H, W = x3.shape[:2]
                ex = {}
                if kind == "enc":
                    xin, skip, nbytes = z["x2"].astype(np.float16), None, H * W * C
                elif kind == "enclin":   # the real q halo map (20 values) and the real linears
                    qv = np.zeros(z["q"].shape[:2] + (32,))
                    qv[..., :20] = q8(z["q"].astype(np.float64))
                    ex = lin_extra(fp8_encode(qv).astype(np.uint8), z["enc0"][0])
                    xin, skip, nbytes = ex.pop("x1"), None, H * W * C
                else:
                    xin, skip, nbytes = fp8_encode(q8(z["y"].astype(np.float64))).astype(np.uint8), x3, H * W * C * 2
                    if kind == "deccoef":
                        ex = {"coef": z["dec0"][0], "rw": W, "rh": H}
                ox, oy = off
                nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
                lines += case_lines(cdir, n, kernel, kind, H, W, ox, oy, nwin, xin, skip, nbytes, p, ex)
                meta.append((n, spec, H, W, off, nwin, p, xin, skip, ex))
                n += 1
    with open(os.path.join(cdir, "cases.txt"), "w") as f:
        f.write("\n".join(lines))
    if not a.no_replay:
        r = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, os.path.join(cdir, "cases.txt")],
                           env=dict(os.environ, WG_L_FORM="hybrid"), capture_output=True, text=True, timeout=150,
                           creationflags=0x08000000)
        if r.returncode:
            raise SystemExit(f"kernel_replay failed: {r.stdout[-800:]}{r.stderr[-400:]}")
        # the default leg (standing rule): no form variable set == WG_L_FORM=hybrid, every output and dump byte
        dl = LD.replay_default_leg(a.build, os.path.join(cdir, "cases.txt"), dict(os.environ, WG_L_FORM="hybrid"))
        with open(os.path.join(cdir, "default_leg.json"), "w") as f:
            json.dump(dl, f)
        print("default leg", "PASS" if dl["pass"] else "FAIL", dl["identical"], "/", dl["outputs"])
    if a.replay_only:   # the GPU part only (inside a GPU slot); the analysis runs afterwards with --no-replay
        print("replayed", len(meta), "cases into", cdir)
        return 0
    tally = Tally()
    # the NaN-input leg (oversight decision on the NaN finding, permanent until the 0.6.6 fix): its cases go to
    # tally_nan, listed in the record, neither pass nor fail; every random-leg case (finite codes) must pass
    tally_nan, nan_cases = Tally(), {}
    for n, spec, H, W, off, nwin, p, xin, skip, ex in meta:
        kind = spec[0]
        configure(*spec[1:])
        dump = open(os.path.join(cdir, f"d{n}.bin"), "rb").read()
        if kind == "deccoef":
            out, y2 = None, np.full((H, W, C), np.nan)
        else:
            outb = open(os.path.join(cdir, f"o{n}.bin"), "rb").read()
            out = (np.frombuffer(outb, np.float16).reshape(H, W, C).astype(np.float64) if kind == "dec"
                   else np.frombuffer(outb, np.uint8).reshape(H, W, C))
        for win in range(nwin):
            xt, toks = gather(kind, xin, H, W, off, win, skip)
            d = stages(dump, win)
            lin = (xt, ex["w2"], ex["b2"]) if kind == "enclin" else None
            tw_ = tally_nan if n in nan_leg else tally
            if n in nan_leg and np.isnan(np.asarray(xt, np.float64)).any():
                nan_cases.setdefault(n, {"kernel": spec[0], "H": H, "W": W, "windows": 0})["windows"] += 1
            if kind == "deccoef":
                for t, rc in toks.items():
                    y2[rc] = d["OUT"][t]
                check_window(tw_, d, p, xt, {}, kind, a.mutant)
            else:
                check_window(tw_, d, p, xt, {t: out[rc] for t, rc in toks.items()}, kind, a.mutant, lin=lin)
        if kind == "deccoef":
            coef_check(tally_nan if n in nan_leg else tally, cdir, n, y2, ex)
            if n in nan_leg:
                nan_cases.setdefault(n, {"kernel": spec[0], "H": H, "W": W, "windows": 0})["pack12_pixels_nan"] = \
                    int(np.isnan(y2[:ex["rh"], :ex["rw"]]).any(axis=-1).sum())
    res = {"stages": tally.s}
    if nan_leg:   # reported, neither pass nor fail
        for t in tally_nan.s.values():
            t["differ_rate"] = t["differ"] / max(1, t["outputs"])
        res["nan_input"] = {"rule": "oversight decision on the NaN finding (until the 0.6.6 fix): the dedicated NaN-input "
                            "leg, listed, neither pass nor fail", "leg_cases": len(nan_leg),
                            "cases_with_nan": {str(k): v for k, v in sorted(nan_cases.items())}, "stages": tally_nan.s}
    dlp = os.path.join(cdir, "default_leg.json")
    res["default_leg"] = json.load(open(dlp)) if os.path.exists(dlp) else {"pass": False, "missing": True}
    for st, t in tally.s.items():
        t["near_share"] = t["near"] / max(1, t["outputs"])
    for t in tally.s.values():
        t["differ_rate"] = t["differ"] / max(1, t["outputs"])
    # oversight 2026-11-15 ~18:20Z: the allowance share is reported only; in the --real run (local files, read at run
    # time) every FP8 chain stage had 0 actual differences (P.V, the one chain whose float32 sum is not exact, keeps the
    # 1e-4 rate)
    fp8_chains = ("qk", "v", "scores", "wp", "w1", "w2")
    fp8_chains = fp8_chains + ("lin2",)
    res["pass"] = all(t["violations"] == 0 and t["differ_outside"] == 0
                      and (st.startswith("coef12_") or t["differ_rate"] <= DIFF_MAX)
                      and not (a.real and st in fp8_chains and t["differ"] > 0) for st, t in tally.s.items())
    res["pass"] = res["pass"] and res["default_leg"]["pass"]   # the default leg (standing rule)
    res["rules"] = {"bound_exceedance": 0, "differ_rate_max": DIFF_MAX, "allowance_share": "reported only",
                    "random_legs": "finite codes only; one NaN-input leg reported separately",
                    "fp8_chain_differences": 0 if a.real else "not required (random weights)"}
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
