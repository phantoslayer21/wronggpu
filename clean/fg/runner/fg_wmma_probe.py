# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): probe of gfx12's 16 x 16 x 16 matrix instructions against net1's chunk arithmetic.

    python clean/fg/runner/fg_wmma_probe.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_wmma_probe.py --build <dir> --rocm ... --run [--backends hip,d3d12] [--scale 1]
        [--json record.json]

The spec's chunk is acc' = RN16(acc + sum of 16 float16 products) with ONE rounding. A WMMA has K = 16 and adds C in
the same instruction, so each of its 256 outputs is a chunk when C is the accumulator. Two forms are probed
(clean/fg/kernels/fg_probe.hip, D3D12 port clean/dx12/kernels/fg_probe.hlsl):
- f16: v_wmma_f16_16x16x16_f16 (float16 accumulator): tier A of the fast-path note if it always equals the exact chunk;
- f32: v_wmma_f32_16x16x16_f16 (float32 accumulator): is it RN32 of the exact sum? And for tier B: is RN16 of its
  result the exact chunk wherever that result is not exactly on a float16 midpoint (fg_probe_mid.h)?
The expected values come from the exact integer chunk on HIP (fg_exact16.h, proven against the CPU reference in #183 to
#191); a sample of every class is re-checked on the CPU here (fp16acc.acc_dot16 and Python integers).

Classes: millions of random chunks (several distributions, net1-like values, the zero channels of net1_01's input),
adversarial ones (cancellation down to the smallest terms, ties at the rounding boundary decided by a 2^-48 term or
by nothing, overflow to inf and huge intermediate sums that cancel, subnormal results, signed zeros incl. the zero
channels, NaN / inf inputs), and every net1 layer's chunk chain (its number of chunks, D fed back as C). Counts per
form and class, with up to three mismatching examples (inputs, expected, got). On D3D12 the outputs are also compared
byte for byte with HIP (sentinels 0xA5 / 0x5A and 64 slack bytes). Every launch is one short dispatch.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.fg import layout as LY  # noqa: E402
from clean.fg.reference import fp16acc as FA  # noqa: E402

CO = "fg_probe.co"
KERNELS = ["fg_probe_mma16", "fg_probe_mma32"]
HIP_SENTINEL, D3D12_SENTINEL, SLACK = 0xA5, 0x5A, 64
TILES = 8192                      # tiles (256 chunks each) per batch
F16 = np.float16


def f16(x):
    return np.asarray(x, dtype=F16)


def pow2(rng, lo, hi, shape):
    """Random signed powers of two 2^e, e integer in [lo, hi]."""
    s = rng.choice([-1.0, 1.0], shape)
    return f16(s * np.exp2(rng.integers(lo, hi + 1, shape).astype(np.float64)))


def rbits(rng, shape, emax=30, emin=0):
    """Random finite float16 values: exponent field in [emin, emax], any fraction, random sign."""
    e = rng.integers(emin, emax + 1, shape).astype(np.uint16)
    b = (e << 10) | rng.integers(0, 1024, shape).astype(np.uint16) | (rng.integers(0, 2, shape).astype(np.uint16) << 15)
    return b.view(F16)


def blank(T, L=1):
    return np.zeros((T, L, 16, 16), F16), np.zeros((T, L, 16, 16), F16), np.zeros((T, 16, 16), F16)


# ---- single-chunk classes (L = 1): A [T, 1, m, k], B [T, 1, k, n], C [T, m, n] -------------------------------------

def c_random(rng, T):
    return f16(rng.uniform(-2, 2, (T, 1, 16, 16))), f16(rng.uniform(-2, 2, (T, 1, 16, 16))), f16(rng.uniform(-4, 4, (T, 16, 16)))


def c_random_bits(rng, T):
    """Any finite float16 (overflow to inf frequent)."""
    return rbits(rng, (T, 1, 16, 16)), rbits(rng, (T, 1, 16, 16)), rbits(rng, (T, 16, 16))


def c_bits_small(rng, T):
    """Any finite float16 below 2^4 (subnormals included): the widest exponent spread without overflow."""
    return rbits(rng, (T, 1, 16, 16), 18), rbits(rng, (T, 1, 16, 16), 18), rbits(rng, (T, 16, 16), 18)


def _layer_x(rng, shape):
    x = rng.uniform(0, 3, shape)
    neg = rng.random(shape) < 0.2
    x[neg] = -rng.uniform(0, 0.03, int(neg.sum()))
    return f16(x)


def c_layer(rng, T):
    """net1-like: activations (leaky outputs), weights, biases."""
    return f16(rng.uniform(-0.25, 0.25, (T, 1, 16, 16))), _layer_x(rng, (T, 1, 16, 16)), f16(rng.uniform(-0.5, 0.5, (T, 16, 16)))


def c_zero_channels(rng, T):
    """net1_01's third channel block: input channels 40..47 of the cells are +0 (k = 8..15 of the chunk)."""
    A, B, C = c_layer(rng, T)
    B[:, :, 8:, :] = 0
    return A, B, C


def c_cancel_pairs(rng, T):
    """Pairs X * Y + X * Y' with Y' = h(-Y + d): products up to 2^16 cancelling down to tiny residues."""
    A, B, C = blank(T)
    X = pow2(rng, 0, 15, (T, 16, 8)) * f16(rng.uniform(1, 2, (T, 16, 8)))
    Y = f16(rng.uniform(1, 2, (T, 8, 16)) * rng.choice([-1, 1], (T, 8, 16)) * np.exp2(-rng.integers(0, 9, (T, 8, 16))))
    d = pow2(rng, -24, -10, (T, 8, 16))
    A[:, 0, :, 0::2] = X
    A[:, 0, :, 1::2] = X
    B[:, 0, 0::2, :] = Y
    B[:, 0, 1::2, :] = f16(-Y.astype(np.float64) + d.astype(np.float64))
    C[:] = np.where(rng.random((T, 16, 16)) < 0.5, pow2(rng, -24, -10, (T, 16, 16)), 0)
    return A, B, C


def c_cancel_acc(rng, T):
    """The accumulator cancels a big product exactly (C = V, -1 * V); the 15 other products are tiny (2^-28 .. 2^-12):
    the result is their exact sum."""
    A, B, C = blank(T)
    V = f16(rng.uniform(1024, 65504, (T, 16)) * rng.choice([-1, 1], (T, 16)))
    A[:, 0, :, 0] = -1
    B[:, 0, 0, :] = V
    C[:] = V[:, None, :]
    A[:, 0, :, 1:] = pow2(rng, -14, -6, (T, 16, 15)) * f16(rng.uniform(1, 2, (T, 16, 15)))
    B[:, 0, 1:, :] = pow2(rng, -14, -6, (T, 15, 16)) * f16(rng.uniform(1, 2, (T, 15, 16)))
    return A, B, C


def c_ties(rng, T):
    """C = v and products adding exactly half a float16 step of v (one term, or four quarter terms), plus a sticky
    term of +-2^-48 / +-2^-44 / 0 and, in half of the columns, a big pair that cancels exactly."""
    A, B, C = blank(T)
    E = rng.integers(-11, 15, (T, 16))                                       # binade of row m
    frac = rng.integers(0, 1024, (T, 16, 16))
    C[:] = f16((1024 + frac) * np.exp2(E[:, :, None] - 10.0) * rng.choice([-1, 1], (T, 16, 16)))
    hstep = np.exp2(E - 11.0)
    split = rng.random((T, 16)) < 0.3                                        # column n: four quarter terms
    A[:, 0, :, 0] = f16(hstep)
    A[:, 0, :, 4:8] = f16(hstep / 4)[:, :, None]
    B[:, 0, 0, :] = np.where(split, 0, 1)
    B[:, 0, 4:8, :] = np.where(split, 1, 0)[:, None, :]
    A[:, 0, :, 1] = 2.0 ** -24
    B[:, 0, 1, :] = f16(rng.choice([2.0 ** -24, -2.0 ** -24, 0, 0, 2.0 ** -20, -2.0 ** -20], (T, 16)))
    X = pow2(rng, 5, 15, (T, 16))
    A[:, 0, :, 2] = X
    A[:, 0, :, 3] = X
    Y = f16(rng.uniform(1, 2, (T, 16)) * (rng.random((T, 16)) < 0.5))
    B[:, 0, 2, :] = Y
    B[:, 0, 3, :] = -Y
    return A, B, C


def c_overflow(rng, T):
    """Around 65504 / 65520 (the overflow boundary), sums of same-sign big products, and products of 2^30 that cancel
    (the exact result is finite)."""
    A, B, C = blank(T)
    C[:] = f16(rng.choice([65504, 65472, 65440, 32768, 1], (T, 16, 16)) * rng.choice([-1, 1], (T, 16, 16)))
    A[:, 0, :, 0] = f16(rng.choice([8, 15.9921875, 16, 16.015625, 31.984375, 32, 0], (T, 16)) * rng.choice([-1, 1], (T, 16)))
    B[:, 0, 0, :] = 1
    A[:, 0, :, 1] = f16(rng.choice([32768, 65504, 1024], (T, 16)))
    A[:, 0, :, 2] = A[:, 0, :, 1]
    big = f16(rng.choice([32768, 65504, 4096], (T, 16)) * rng.choice([-1, 1], (T, 16)))
    on = rng.random((T, 16)) < 0.6
    B[:, 0, 1, :] = np.where(on, big, 0)
    B[:, 0, 2, :] = np.where(on & (rng.random((T, 16)) < 0.7), -big, np.where(on, big, 0))
    A[:, 0, :, 3] = 2.0 ** -24
    B[:, 0, 3, :] = f16(rng.choice([2.0 ** -24, -2.0 ** -24, 0], (T, 16)))
    A[:, 0, :, 4:] = f16(rng.uniform(-300, 300, (T, 16, 12)))
    B[:, 0, 4:, :] = f16(rng.uniform(-300, 300, (T, 12, 16)) * (rng.random((T, 1, 16)) < 0.3))
    return A, B, C


def c_subnormal(rng, T):
    """Tiny operands (float16 subnormals and the lowest binades) and accumulators: results in or near the float16
    subnormal range, sums crossing 2^-14."""
    A = rbits(rng, (T, 1, 16, 16), 8)
    big = rng.random((T, 1, 16, 1)) < 0.25                                    # some rows of ordinary size
    A = np.where(big, rbits(rng, (T, 1, 16, 16), 16, 12), A)
    B = rbits(rng, (T, 1, 16, 16), 4)
    C = rbits(rng, (T, 16, 16), 3)
    return A, B, C


def c_signed_zero(rng, T):
    """Exact zero sums: products that are +-0 or cancel in pairs, accumulators +-0. The reference rounds an exact zero
    to +0 (fg_round16 / fp16acc); hardware may follow IEEE (-0 only when every term is -0)."""
    A, B, C = blank(T)
    a = f16(rng.choice([0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0 ** -24], (T, 16, 8)))
    A[:, 0, :, 0::2] = a
    A[:, 0, :, 1::2] = a
    y = f16(rng.choice([1.0, -2.0, 0.25, 65504.0], (T, 8, 16)))
    zero_col = rng.random((T, 1, 16)) < 0.5
    zb = f16(rng.choice([0.0, -0.0], (T, 16, 16)))
    B[:, 0, 0::2, :] = np.where(zero_col, zb[:, 0::2, :], y)
    B[:, 0, 1::2, :] = np.where(zero_col, zb[:, 1::2, :], -y)
    C[:] = f16(rng.choice([0.0, -0.0], (T, 16, 16)))
    return A, B, C


def c_zero_channels_signed(rng, T):
    """The zero-channel case with an exact zero: k = 0..7 cancel in pairs, k = 8..15 are +0 inputs (cells 40..47) times
    weights of either sign, the accumulator +-0."""
    A, B, C = blank(T)
    a = f16(rng.uniform(-0.3, 0.3, (T, 16, 4)))
    A[:, 0, :, 0:8:2] = a
    A[:, 0, :, 1:8:2] = a
    A[:, 0, :, 8:] = f16(rng.uniform(-0.3, 0.3, (T, 16, 8)))
    y = f16(rng.uniform(0, 3, (T, 4, 16)))
    B[:, 0, 0:8:2, :] = y
    B[:, 0, 1:8:2, :] = -y
    C[:] = f16(rng.choice([0.0, -0.0], (T, 16, 16)))
    return A, B, C


def c_specials(rng, T):
    """net1-like values with NaN and +-inf in A, B and C, and zeros next to infinities (0 * inf)."""
    A, B, C = c_layer(rng, T)
    for X, vals in ((A, [np.nan, np.inf, -np.inf]), (B, [np.nan, np.inf, -np.inf, 0.0]), (C, [np.nan, np.inf, -np.inf])):
        m = rng.random(X.shape) < 0.03
        X[m] = f16(rng.choice(vals, int(m.sum())))
    return A, B, C


SINGLE = [("random", c_random, 8), ("random_bits", c_random_bits, 8), ("bits_small", c_bits_small, 8),
          ("layer_like", c_layer, 8), ("zero_channels", c_zero_channels, 4), ("cancel_pairs", c_cancel_pairs, 2),
          ("cancel_acc", c_cancel_acc, 2), ("ties", c_ties, 2), ("overflow", c_overflow, 2),
          ("subnormal", c_subnormal, 2), ("signed_zero", c_signed_zero, 1),
          ("zero_channels_signed", c_zero_channels_signed, 1), ("specials", c_specials, 2)]


# ---- chain classes: every net1 layer's chunk chain ------------------------------------------------------------------

def chain_case(rng, layer, T):
    k = 3 if layer.taps == 9 else 1
    nb = layer.cin // 16
    L = nb * k * k
    s = 1.5 / np.sqrt(layer.cin * k * k)
    A = f16(rng.uniform(-s, s, (T, L, 16, 16)))
    B = _layer_x(rng, (T, L, 16, 16))
    if layer.name == "net1_01":                       # block_kx_ky: channel block 2 is chunks 18..26; cells 40..47 = 0
        B[:, 2 * k * k:, 8:, :] = 0
    C = f16(rng.uniform(-0.1, 0.1, (T, 16, 16)))
    return A, B, C


def chain_classes(scale):
    out = []
    for l in LY.NET1:
        L = (l.cin // 16) * l.taps
        T = max(64, int(scale * (256 if L >= 32 else 1024)))
        out.append(("chain_" + l.name, (lambda rng, T, l=l: chain_case(rng, l, T)), 1, T, L))
    return out


# ---- comparison -----------------------------------------------------------------------------------------------------

def nan16(b):
    return ((b & 0x7C00) == 0x7C00) & ((b & 0x3FF) != 0)


def nan32(b):
    return ((b & 0x7F800000) == 0x7F800000) & ((b & 0x7FFFFF) != 0)


def compare16(got, want):
    """(equal mask, zero-sign-only mask): bits equal or both NaN; a mismatch that is only +0 against -0."""
    eq = (got == want) | (nan16(got) & nan16(want))
    zs = ~eq & ((got ^ want) == 0x8000) & ((want & 0x7FFF) == 0)
    return eq, zs


def compare32(got, want):
    eq = (got == want) | (nan32(got) & nan32(want))
    zs = ~eq & ((got ^ want) == 0x80000000) & ((want & 0x7FFFFFFF) == 0)
    return eq, zs


def hexs(a):
    return ["%04x" % int(v) for v in np.asarray(a).view(np.uint16).reshape(-1)]


def example(A, B, C, idx, want, got, L, width=16):
    t, m, n = idx // 256, (idx // 16) % 16, idx % 16
    ex = {"tile": int(t), "m": int(m), "n": int(n), "acc_c": hexs(C[t, m, n]),
          "expected": ("%04x" if width == 16 else "%08x") % int(want[idx]), "got": ("%04x" if width == 16 else "%08x") % int(got[idx])}
    if L == 1:
        ex["a_row"] = hexs(A[t, 0, m, :])
        ex["b_col"] = hexs(B[t, 0, :, n])
    else:
        ex["chunks"] = int(L)
    return ex


class Tally:
    def __init__(self):
        self.n = 0
        self.c = {}
        self.ex = {}

    def add(self, key, mask, A, B, C, want, got, L, width=16, keep=True):
        self.c[key] = self.c.get(key, 0) + int(mask.sum())
        if keep and mask.any() and len(self.ex.setdefault(key, [])) < 3:
            for i in np.flatnonzero(mask)[:3 - len(self.ex[key])]:
                self.ex[key].append(example(A, B, C, int(i), want, got, L, width))

    def out(self):
        r = {"chunks": self.n, **self.c}
        if any(self.ex.values()):
            r["examples"] = {k: v for k, v in self.ex.items() if v}
        return r


def steps(got, want):
    """Distance in float16 steps between finite values (the order of the bit patterns; +0 and -0 are 0 apart); -1 where
    either is not finite (then only equality counts)."""
    def key(b):
        b = b.astype(np.int64)
        return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)
    fin = ((got & 0x7C00) != 0x7C00) & ((want & 0x7C00) != 0x7C00)
    return np.where(fin, np.abs(key(got) - key(want)), -1)


def step_stats(tally, prefix, got, want, eq):
    d = steps(got, want)
    miss = ~eq
    tally.c[prefix + "steps_max"] = max(tally.c.get(prefix + "steps_max", 0), int(d.max()) if d.size else 0)
    tally.c[prefix + "over_1_step"] = tally.c.get(prefix + "over_1_step", 0) + int(np.sum(d > 1))
    tally.c[prefix + "over_4_steps"] = tally.c.get(prefix + "over_4_steps", 0) + int(np.sum(d > 4))
    tally.c[prefix + "nonfinite_mismatch"] = tally.c.get(prefix + "nonfinite_mismatch", 0) + int(np.sum(miss & (d < 0)))


def score(tal, A, B, C, L, e16, e32, d16, d32, f):
    """Adds one batch's results to tal = {"f16": Tally, "f32": Tally}."""
    n = e16.size
    t16, t32 = tal["f16"], tal["f32"]
    t16.n += n
    t32.n += n
    eq, zs = compare16(d16, e16)
    t16.add("equal", eq, A, B, C, e16, d16, L, keep=False)
    t16.add("mismatch", ~eq & ~zs, A, B, C, e16, d16, L)
    t16.add("zero_sign_only", zs, A, B, C, e16, d16, L)
    step_stats(t16, "", d16, e16, eq | zs)
    acc16 = (f & 0xFFFF).astype(np.uint16)
    flag = (f >> 16) != 0
    eqh, zsh = compare16(acc16, e16)
    step_stats(t32, "rn16_", acc16, e16, eqh | zsh)
    if L == 1:
        eq32, zs32 = compare32(d32, e32)
        t32.add("rn32_equal", eq32, A, B, C, e32, d32, L, 32, keep=False)
        t32.add("rn32_mismatch", ~eq32 & ~zs32, A, B, C, e32, d32, L, 32)
        t32.add("rn32_zero_sign_only", zs32, A, B, C, e32, d32, L, 32)
    t32.add("rn16_equal", eqh, A, B, C, e16, acc16, L, keep=False)
    t32.add("rn16_mismatch_flagged", ~eqh & ~zsh & flag, A, B, C, e16, acc16, L)
    t32.add("rn16_mismatch_unflagged", ~eqh & ~zsh & ~flag, A, B, C, e16, acc16, L)
    t32.add("rn16_zero_sign_only", zsh, A, B, C, e16, acc16, L)
    t32.add("flagged", flag, A, B, C, e16, acc16, L, keep=False)


def mid16_py(x):
    """Python twin of fg_probe_mid.h's fg_mid16 (float32 bits -> 1 when exactly halfway between two float16 values)."""
    a = x & 0x7FFFFFFF
    e = a >> 23
    if e == 255 or e == 0:
        return 0
    if e >= 113:
        return int(e <= 142 and (a & 0x1FFF) == 0x1000)
    s = 125 - e
    if s > 23:
        return 0
    M = (a & 0x7FFFFF) | 0x800000
    return int((M & ((2 << s) - 1)) == (1 << s))


# ---- CPU re-check of the GPU's exact reference ---------------------------------------------------------------------

def _m(b):
    e, f = (b >> 10) & 31, b & 1023
    m = f if e == 0 else (f | 1024) << (e - 1)
    return -m if b & 0x8000 else m


def rn32_py(acc_bits, a_bits, b_bits):
    """RN32 of acc + sum a * b (finite float16 bits) with Python integers (units 2^-48); exact zero -> +0."""
    t = _m(acc_bits) << 24
    for a, b in zip(a_bits, b_bits):
        t += _m(a) * _m(b)
    if t == 0:
        return 0
    neg, mag = t < 0, abs(t)
    e = mag.bit_length() - 1
    if e <= 23:
        q, ex = mag << (23 - e), e
    else:
        sh = e - 23
        q, rem, half = mag >> sh, mag & ((1 << sh) - 1), 1 << (sh - 1)
        if rem > half or (rem == half and q & 1):
            q += 1
        ex = e
        if q == 1 << 24:
            q >>= 1
            ex += 1
    return (int(neg) << 31) | ((ex - 48 + 127) << 23) | (q & 0x7FFFFF)


def cpu_check(rng, A, B, C, L, e16, e32, n=2048):
    """GPU reference against fp16acc.acc_dot16 (chained) on n sampled outputs, and against rn32_py on finite ones."""
    T = C.shape[0]
    idx = rng.choice(T * 256, min(n, T * 256), replace=False)
    t, m, nn = idx // 256, (idx // 16) % 16, idx % 16
    acc = C[t, m, nn].astype(np.float64)
    last_acc = acc
    for l in range(L):
        a = A[t, l, m, :].astype(np.float64)
        b = B[t, l, :, nn].astype(np.float64)
        last_acc = acc
        with np.errstate(invalid="ignore", over="ignore"):
            acc = FA.acc_dot16(acc, a, b, axis=1)
    want = acc.astype(F16).view(np.uint16)
    eq, _ = compare16(e16[idx], want)
    bad32 = 0
    checked32 = 0
    lb = last_acc.astype(F16).view(np.uint16)
    for j, i in enumerate(idx[:256]):
        ab = A[t[j], L - 1, m[j], :].view(np.uint16)
        bb = B[t[j], L - 1, :, nn[j]].view(np.uint16)
        bits = np.concatenate([ab, bb, lb[j:j + 1]])
        if np.any((bits & 0x7C00) == 0x7C00):
            continue
        checked32 += 1
        bad32 += int(rn32_py(int(lb[j]), [int(x) for x in ab], [int(x) for x in bb]) != int(e32[i]))
    return {"sampled16": int(idx.size), "differ16": int((~eq).sum()), "sampled32": checked32, "differ32": bad32}


# ---- HIP and D3D12 --------------------------------------------------------------------------------------------------

def hip_batch(h, fn, A, B, C, L):
    T = C.shape[0]
    n = T * 256
    bufs = {}
    try:
        for k, arr in (("A", A), ("B", B), ("C", C)):
            bufs[k] = h.malloc(arr.nbytes)
            h.upload(bufs[k], np.ascontiguousarray(arr).tobytes())
        for k, nb in (("e16", 2 * n), ("e32", 4 * n), ("d16", 2 * n), ("d32", 4 * n), ("f", 4 * n)):
            bufs[k] = h.malloc(nb)
            h.memset(bufs[k], HIP_SENTINEL, nb)
        p = {k: ctypes.c_void_p(v) for k, v in bufs.items()}
        ci = [ctypes.c_int(T), ctypes.c_int(L)]
        h.launch(fn["fg_probe_exact"], T, 256, [p["A"], p["B"], p["C"], p["e16"], p["e32"]] + ci)
        g = -(-T // 8)
        h.launch(fn["fg_probe_mma16"], g, 256, [p["A"], p["B"], p["C"], p["d16"], p["f"]] + ci)
        h.launch(fn["fg_probe_mma32"], g, 256, [p["A"], p["B"], p["C"], p["d32"], p["f"]] + ci)
        h.sync()
        rd = lambda k, nb, dt: np.frombuffer(bytes(h.download(bufs[k], nb)), dt)
        return rd("e16", 2 * n, np.uint16), rd("e32", 4 * n, np.uint32), rd("d16", 2 * n, np.uint16), \
            rd("d32", 4 * n, np.uint32), rd("f", 4 * n, np.uint32)
    finally:
        for v in bufs.values():
            h.free(v)


def replay_case_lines(name, A, B, C, L, cdir, written):
    T = C.shape[0]
    n = T * 256
    files = {}
    for k, arr in (("A", A), ("B", B), ("C", C)):
        fn = "%s.%s" % (name, k)
        with open(os.path.join(cdir, fn), "wb") as f:
            f.write(np.ascontiguousarray(arr).tobytes())
        files[k] = fn
        written.append(fn)
    g = -(-T // 8)
    lines = []
    for kern, outs in (("fg_probe_mma16", [("d16", 2 * n)]), ("fg_probe_mma32", [("d32", 4 * n), ("f", 4 * n)])):
        lines += ["case %s_%s" % (name, kern), "module " + CO, "kernel %s %d 256" % (kern, g)]
        lines += ["arg buf " + files[k] for k in ("A", "B", "C")]
        for k, nb in outs:
            fn = "%s.%s" % (name, k)
            lines.append("arg out %d %s %d" % (nb + SLACK, fn, D3D12_SENTINEL))
            written.append(fn)
        if kern == "fg_probe_mma16":
            lines.append("arg null")
        lines += ["arg u32 %d" % T, "arg u32 %d" % L, ""]
    return lines


def run(build, rocm, backends, scale, seed=20261114):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    with open(os.path.join(build, CO), "rb") as f:
        mod = h.load_module(f.read())
    fn = {k: h.get_function(mod, k) for k in ("fg_probe_exact", "fg_probe_mma16", "fg_probe_mma32")}
    rng = np.random.default_rng(seed)
    classes = [(name, gen, max(1, int(round(nb * scale))), TILES, 1) for name, gen, nb in SINGLE] + chain_classes(scale)
    res = {"seed": seed, "scale": scale, "tiles_per_batch": TILES, "hip": {}, "cpu_check": {}}
    d3d = "d3d12" in backends
    cdir = os.path.join(build, "replay-probe")
    os.makedirs(cdir, exist_ok=True)
    lines, written, keep = [], [], {}
    t0 = time.time()
    for name, gen, nb, T, L in classes:
        tal = {"f16": Tally(), "f32": Tally()}
        cc = None
        for b in range(nb):
            A, B, C = gen(rng, T)
            e16, e32, d16, d32, f = hip_batch(h, fn, A, B, C, L)
            score(tal, A, B, C, L, e16, e32, d16, d32, f)
            if b == 0:
                cc = cpu_check(rng, A, B, C, L, e16, e32)
                if d3d:
                    cname = "probe_" + name
                    lines += replay_case_lines(cname, A, B, C, L, cdir, written)
                    keep[name] = (cname, A, B, C, L, e16, e32, d16, d32, f)
        res["hip"][name] = {"chunks_per_output": L, "f16": tal["f16"].out(), "f32": tal["f32"].out()}
        res["cpu_check"][name] = cc
    res["hip_seconds"] = round(time.time() - t0, 1)
    if d3d:
        res["d3d12"] = run_d3d12(build, cdir, lines, written, keep)
    return res


def run_d3d12(build, cdir, lines, written, keep):
    path = os.path.join(cdir, "cases.txt")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "512"
    try:
        r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, path], env=env, capture_output=True,
                           text=True, timeout=120, creationflags=0x08000000)
        out = {"replay_exit": r.returncode, "replay_tail": r.stdout[-300:] + r.stderr[-200:], "classes": {}}

        def read(fn, dt, n):
            p = os.path.join(cdir, fn)
            raw = open(p, "rb").read() if os.path.exists(p) else b""
            body, slack = raw[:n], raw[n:]
            ok = len(raw) == n + SLACK and all(c == D3D12_SENTINEL for c in slack)
            return (np.frombuffer(body, dt) if len(body) == n else None), ok

        for name, (cname, A, B, C, L, e16, e32, d16, d32, f) in keep.items():
            n = C.shape[0] * 256
            g16, s1 = read(cname + ".d16", np.uint16, 2 * n)
            g32, s2 = read(cname + ".d32", np.uint32, 4 * n)
            gf, s3 = read(cname + ".f", np.uint32, 4 * n)
            if g16 is None or g32 is None or gf is None:
                out["classes"][name] = {"missing_output": True}
                continue
            tal = {"f16": Tally(), "f32": Tally()}
            score(tal, A, B, C, L, e16, e32, g16, g32, gf)
            out["classes"][name] = {"chunks_per_output": L, "f16": tal["f16"].out(), "f32": tal["f32"].out(),
                                    "raw_equal_hip": {"d16": int(np.sum(g16 != d16)) == 0,
                                                      "d32": int(np.sum(g32 != d32)) == 0,
                                                      "f": int(np.sum(gf != f)) == 0},
                                    "slack_ok": bool(s1 and s2 and s3)}
        return out
    finally:
        for fn in written:
            p = os.path.join(cdir, fn)
            if os.path.isfile(p):
                os.remove(p)


def verdict(res):
    """Tier A: the float16 form equals every chunk (zero-sign-only differences reported separately); tier B: the float32
    form's RN16 equals every chunk wherever it is not flagged; else tier C. Per back end."""
    out = {}
    for be in ("hip", "d3d12"):
        cls = res.get(be) if be == "hip" else (res.get("d3d12") or {}).get("classes")
        if not cls:
            continue
        m16 = sum(c["f16"].get("mismatch", 0) for c in cls.values() if "f16" in c)
        z16 = sum(c["f16"].get("zero_sign_only", 0) for c in cls.values() if "f16" in c)
        un = sum(c["f32"].get("rn16_mismatch_unflagged", 0) for c in cls.values() if "f32" in c)
        r32 = sum(c["f32"].get("rn32_mismatch", 0) for c in cls.values() if "f32" in c)
        chunks = sum(c["f16"]["chunks"] * c["chunks_per_output"] for c in cls.values() if "f16" in c)
        tier = "A" if m16 == 0 else ("B" if un == 0 else "C")
        out[be] = {"tier": tier, "chunks": chunks, "f16_mismatch": m16, "f16_zero_sign_only": z16,
                   "f32_rn32_mismatch": r32, "f32_rn16_unflagged_mismatch": un}
        # how far the misses are (float16 steps from the exact chunk), per form: ordinary data (random and net1-like
        # classes and the layer chains) apart from the adversarial classes
        ordinary = [k for k in cls if k in ("random", "bits_small", "layer_like", "zero_channels") or k.startswith("chain_")]
        for form, pre in (("f16", ""), ("f32", "rn16_")):
            for group, keys in (("ordinary", ordinary), ("all", list(cls))):
                cs = [cls[k][form] for k in keys if form in cls[k]]
                n = sum(cls[k]["f16"]["chunks"] for k in keys if "f16" in cls[k])
                out[be]["%s_%s" % (form, group)] = {
                    "outputs": n, "steps_max": max(c.get(pre + "steps_max", 0) for c in cs),
                    "over_1_step": sum(c.get(pre + "over_1_step", 0) for c in cs),
                    "over_4_steps": sum(c.get(pre + "over_4_steps", 0) for c in cs),
                    "nonfinite_mismatch": sum(c.get(pre + "nonfinite_mismatch", 0) for c in cs)}
    return out


def prepare(build, rocm):
    from clean.l.runner.l_d3d12_check import prepare_replay
    from clean.runner import build as BLD
    from clean.runner.hip import clean_process_environment
    os.makedirs(build, exist_ok=True)
    clean_process_environment(rocm)
    clang = BLD.find_clang(rocm)
    BLD.run(BLD.device_command(clang, rocm, os.path.join(ROOT, "clean", "fg", "kernels", "fg_probe.hip"),
                               os.path.join(build, CO)) + ["-I", os.path.join(ROOT, "clean", "kernels")])
    return prepare_replay(build, rocm, KERNELS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backends", default="hip,d3d12")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.prepare:
        prepare(a.build, a.rocm)
    if a.run:
        res = run(a.build, a.rocm, a.backends.split(","), a.scale)
        res["verdict"] = verdict(res)
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res["verdict"], indent=1))
        bad_cpu = {k: v for k, v in res["cpu_check"].items() if v and (v["differ16"] or v["differ32"])}
        print("cpu re-check of the GPU reference:", "all equal" if not bad_cpu else json.dumps(bad_cpu))
        ok = not bad_cpu
        if "d3d12" in res:
            d = res["d3d12"]
            same = d["replay_exit"] == 0 and all(
                all(c.get("raw_equal_hip", {}).values()) and c.get("raw_equal_hip") and c.get("slack_ok")
                for c in d["classes"].values())
            print("d3d12 replay exit", d["replay_exit"], "| byte-identical to HIP with slack sentinels:", same)
            ok = ok and same
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
