# SPDX-License-Identifier: MIT
"""Adversarial cases for Model K's softmax weight rounding (hlsl/k_smx.hlsli, kernels/k_mix_probe.hlsl). WrongGPU's own
code.

A softmax weight is A = f16(w inv), ONE round-to-nearest-even rounding of the exact product of two positive FP32 values.
A form that rounds twice (to FP32 first, then to FP16) differs exactly where p = RN32(w inv) is an FP16 rounding
midpoint and the residual e = w inv - p is nonzero: the exact product lies on one side of the midpoint, the double
rounding sees a tie. These cases put such products in every lane, with both residual signs, in the FP16 normal and
subnormal ranges, plus exact midpoints (e = 0: ties to even) and ordinary products.

One record per lane (k_mix_probe's layout): 12 FP32 values, inv, w0 .. w7, three zeros; the expected output is 8 FP16
patterns, A_j = RN16(w_j inv) computed exactly here (Fraction arithmetic: no floating-point rounding in the oracle).
Products lie in [2^-26, 1] (a softmax weight is at most 1; below 2^-25 the FP16 result is 0 or the smallest subnormal).

    records, expected, kinds = cases(seed, lanes)
"""
from fractions import Fraction

import numpy as np

KINDS = ("mid+", "mid-", "tie", "sub+", "sub-", "plain")


def rn16_bits(x):
    """RN16 of a positive exact value (Fraction or int/float converted exactly), as the FP16 bit pattern."""
    x = Fraction(x)
    if x <= 0:
        raise ValueError("positive values only")
    # exponent of x: 2^e <= x < 2^(e+1)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    q_exp = max(e, -14) - 10                       # quantum: 2^(e-10) for normal FP16, 2^-24 below 2^-14
    q = x / (Fraction(2) ** q_exp)
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if e < -14:                                     # subnormal (n may reach 1024 = the smallest normal)
        return int(n)
    if n == 2048:
        n, e = 1024, e + 1
    if e + 15 >= 31:
        return 0x7C00
    return ((e + 15) << 10) | (int(n) - 1024)


def _midpoint(p):
    """float32 array p > 0: True where p is an FP16 rounding midpoint (normal: low 13 significand bits 0x1000;
    subnormal FP16 range: p / 2^-24 has the fraction 1/2)."""
    b = p.view(np.uint32)
    normal = p >= np.float32(2.0 ** -14)
    mid_n = (b & np.uint32(0x1FFF)) == np.uint32(0x1000)
    s = p.astype(np.float64) * 2.0 ** 24
    mid_s = (s - np.floor(s)) == 0.5
    return np.where(normal, mid_n, mid_s), normal


def cases(seed, lanes):
    rng = np.random.default_rng(seed)
    records = np.zeros((lanes, 12), np.float32)
    kinds = []
    searched = [k for k in KINDS if k != "tie"]
    for lane in range(lanes):
        # every fourth lane holds exact midpoints (its inv is a power of two, chosen with them); the others the searched
        # kinds for their own inv
        want = ["tie"] * 8 if lane % 4 == 3 else [searched[(lane + j) % len(searched)] for j in range(8)]
        # inv: wide range; for subnormal-result lanes small enough that products below 2^-14 are common
        inv = np.float32(2.0 ** rng.uniform(-30.0, 0.0))
        got = {}
        for _ in range(64):
            if all(k in got and len(got[k]) >= want.count(k) for k in set(want)):
                break
            w = (2.0 ** rng.uniform(-14.4, 14.4, 1 << 16)).astype(np.float32)
            p = w * inv                                                   # RN32 product (numpy float32)
            x = w.astype(np.float64) * np.float64(inv)                    # exact (48-bit product)
            ok = (x <= 1.0) & (x >= 2.0 ** -26)
            mid, normal = _midpoint(p)
            e = x - p.astype(np.float64)                                  # exact residual
            sel = {"mid+": ok & mid & normal & (e > 0), "mid-": ok & mid & normal & (e < 0),
                   "sub+": ok & mid & ~normal & (e > 0), "sub-": ok & mid & ~normal & (e < 0),
                   "plain": ok & ~mid}
            for k, m in sel.items():
                idx = np.nonzero(m)[0]
                got.setdefault(k, []).extend(w[idx[:8]].tolist())
        ws = []
        for j, k in enumerate(want):
            if k == "tie":
                # an exact FP16 midpoint as the product: w = (2n + 1) 2^(t - 11) with 2^t in the normal FP16 range,
                # times inv' = a power of two (the lane's inv is replaced by it when the lane has a tie: e = 0)
                ws.append(None)
                continue
            pool = got.get(k) or got.get("plain")
            ws.append(np.float32(pool[j % len(pool)]))
        if any(v is None for v in ws):
            inv = np.float32(2.0 ** -int(rng.integers(0, 10)))
            for j, v in enumerate(ws):
                if v is None:
                    t = int(rng.integers(-12, 0))                         # product in [2^-12, 1)
                    n = int(rng.integers(0, 1024))
                    prod = Fraction(2 * (1024 + n) + 1, 2 ** 11) * Fraction(2) ** t
                    ws[j] = np.float32(float(prod / Fraction(float(inv))))
                elif not (2.0 ** -26 <= float(v) * float(inv) <= 1.0):
                    ws[j] = np.float32(0.5 / float(inv))                  # keep the product in range
        records[lane, 0] = inv
        records[lane, 1:9] = ws
        kinds.append(want)
    expected = np.zeros((lanes, 8), np.uint16)
    for lane in range(lanes):
        inv = Fraction(float(records[lane, 0]))
        for j in range(8):
            expected[lane, j] = rn16_bits(Fraction(float(records[lane, 1 + j])) * inv)
    return records, expected, kinds


def double_rounding_bits(records):
    """What a second rounding through FP32 gives (the K_SMX_MUT control): RN16(RN32(w inv)) per element."""
    out = np.zeros((records.shape[0], 8), np.uint16)
    for lane in range(records.shape[0]):
        for j in range(8):
            p = np.float32(records[lane, 1 + j]) * np.float32(records[lane, 0])
            out[lane, j] = rn16_bits(Fraction(float(p)))
    return out
