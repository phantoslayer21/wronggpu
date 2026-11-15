# SPDX-License-Identifier: MIT
"""Writes clean/runtime/fold_selftest.h: the colour image of the native D3D12 device's differential self-test of Model
L's post (device_d3d12.cpp fold_selftest). WrongGPU's own code.

    python clean/runtime/gen_fold_selftest.py [out.h]

The self-test runs the REAL post shader twice on one tiny frame: the fold port of l_post_recon_p12 (its history encode
through the driver's conversion fold, hlsl/l_post_encode.hlsli) and the no-fold port (round to odd, exact without the
fold), and compares the raw output bytes. Equal: the device runs the fold port; any difference or error: the no-fold
port. The frame (kW x kH, render size = output size, jitter 0, reset, exposure 1, post scale s = 0, every code and
kernel-shape value 0) makes the full history H^ of pixel (x, y) channel i exactly the render colour c_i, so the history
code is q = (_Float16)(kEncK x) with x = log2(max(fma(100, c, 1), 1e-4)) evaluated by the post shader itself
(enc_log): one wave of the shader's six encode sites (three channels, two rows of each row pair) sees each class:
- FP32-product ties: c whose x (predicted with the correctly rounded log2) makes the FP32 product RN32(kEncK x) land
  exactly on an FP16 midpoint while the exact product does not, the exact product above and below the midpoint, most
  of them on the side where the second rounding (to even) goes the wrong way, in four ranges of x and among the
  FP16-subnormal products (the ones there are: x near 0 is sparse);
- exact-product ties do not exist: kEncK's significand has the odd factor 0x377c61 (22 bits) and an FP16 midpoint's
  has at most 12 bits, so kEncK x is never exactly a midpoint unless it is 0;
- zero products: c = +-0, +-2^-149, +-1e-30 (fma(100, c, 1) = 1, x = +0, the code +0);
- FP16-subnormal products of both signs (fma(100, c, 1) = 1 +- a few ulps); the exhaustive search over every FP32
  value of fma(100, c, 1) in that range finds no FP32-product tie there;
- the floor: c whose fma(100, c, 1) is just below, at or near 1e-4 (the max picks 1e-4), and far below it (0, -1, -1e30).
Which x the GPU's log2 gives is not assumed: an x that differs from the prediction just tests another value. The
fold-removed mutant (LP_MUT 3, two roundings) must make the self-test fail (GPU record of the PR).
Every predicted value is checked with exact rationals (fractions.Fraction, decimal logarithms to 60 digits).
"""
import decimal
import os
import struct
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_fold_probe import K_BITS, bits32, f32, rn16_bits  # noqa: E402

W, H = 32, 16
SITES = 6        # (row of the pair, channel)
PER_SITE = W * H * 3 // SITES
SEED = 20261115
OFFSETS = range(-8, 9)   # the device's log2 may differ from the correctly rounded one by up to 8 ulp
TIES_PER_SITE = 13       # each with all 17 offsets: N = 13 detecting values per site for every uniform offset
K = Fraction(f32(K_BITS))
DEC = decimal.Context(prec=60)


def rn(fr, p, emin):
    """Round a Fraction to p significant bits, nearest even, exponents below emin on the emin grid (no overflow)."""
    if fr == 0:
        return Fraction(0)
    s = -1 if fr < 0 else 1
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, emin)
    q = Fraction(2) ** (e - p + 1)
    n = a / q
    f = n.numerator // n.denominator
    r = n - f
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and f % 2 == 1):
        f += 1
    return s * f * q


def rn32(fr):
    return rn(fr, 24, -126)


def m_of(c):
    """RN32(fma(100, c, 1)) for the FP32 value c (exact)."""
    return rn32(100 * Fraction(c) + 1)


def log2_exact(m):
    """(the correctly rounded FP32 log2 of the FP32 value m > 0, the distance of the exact log2 from an FP32 midpoint in
    units of that FP32 ulp) from 60-digit decimal logarithms."""
    lm = DEC.divide(DEC.ln(decimal.Decimal(m.numerator) / decimal.Decimal(m.denominator)), DEC.ln(decimal.Decimal(2)))
    v = Fraction(lm)
    x = rn32(v)
    if x == 0:
        return x, None
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    ulp = Fraction(2) ** (e - 23)
    frac = (a / ulp) - (a / ulp).numerator // (a / ulp).denominator
    return x, abs(frac - Fraction(1, 2))


def fp16_midpoint(p):
    """Whether the FP32 value p lies exactly halfway between two adjacent FP16 values (finite range)."""
    a = abs(p)
    if a == 0 or a >= 65504:
        return False
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    ulp = Fraction(2) ** (max(e, -14) - 10)
    n = a / ulp
    return n.denominator == 2


def classify(c):
    """(x, product class) of the colour c, checked exactly: None unless kEncK x is an FP32-product tie; else
    ('up' / 'down' = the exact product's side, detecting = the second rounding goes to the other FP16 value)."""
    m = m_of(c)
    if m < Fraction(f32(bits32(1e-4))):
        return None
    x, margin = log2_exact(m)
    if x == 0 or margin is None or margin < Fraction(1, 10**12):
        return None
    pe = x * K
    p32 = rn32(pe)
    if not fp16_midpoint(p32) or pe == p32:
        return None
    detecting = rn16_bits(pe) != rn16_bits(p32)
    return x, ("up" if pe > p32 else "down"), detecting


def search(rng, lo, hi, want, sub=False):
    """want ties (half the exact products above the midpoint, half below; detecting, plus a sixth non-detecting) from
    c whose fma(100, c, 1) lies in [lo, hi], log-uniform; sub: every m = 1 +- k ulp in the range (exhaustive)."""
    k32 = np.float64(f32(K_BITS))
    if sub:
        ks = np.arange(1, int((hi - 1) * 2**23) + 1)
        mt = np.concatenate([1.0 + ks * 2.0**-23, 1.0 - np.arange(1, int((1 - lo) * 2**24) + 1) * 2.0**-24])
        rounds = [mt]
    else:
        rounds = (np.exp2(rng.uniform(np.log2(lo), np.log2(hi), 1 << 20)) for _ in range(400))
    picks = {("up", True): [], ("down", True): [], ("up", False): [], ("down", False): []}
    quota = {("up", True): want // 2, ("down", True): want // 2, ("up", False): want // 12, ("down", False): want // 12}
    seen = set()
    for mt in rounds:
        c = ((mt - 1.0) / 100.0).astype(np.float32)
        m = (100.0 * c.astype(np.float64) + 1.0).astype(np.float32)
        ok = (m >= lo) & (m <= hi)
        x = np.log2(m.astype(np.float64)).astype(np.float32)
        pe = x.astype(np.float64) * k32
        p32 = pe.astype(np.float32).astype(np.float64)
        a = np.abs(p32)
        normal = a >= 2.0**-14
        pb = np.abs(p32.astype(np.float32)).view(np.uint32)
        tie_n = normal & ((pb & 0x1fff) == 0x1000)
        t = a * 2.0**25
        tie_s = ~normal & (t == np.round(t)) & (np.fmod(t, 2.0) == 1.0) & (a > 0)
        for i in np.nonzero(ok & (tie_n | tie_s) & (pe != p32))[0]:
            cb = int(c[i].view(np.uint32))
            if cb in seen:
                continue
            seen.add(cb)
            r = classify(f32(cb))
            if r is None:
                continue
            key = (r[1], r[2])
            if len(picks[key]) < quota[key]:
                picks[key].append(cb)
        if all(len(picks[k]) >= quota[k] for k in picks):
            break
    return picks


def floor_values():
    """c around the floor: fma(100, c, 1) the nearest reachable values below / at / above 1e-4, and far below."""
    one_em4 = Fraction(f32(bits32(1e-4)))
    c0 = bits32(np.float32((1e-4 - 1.0) / 100.0))
    below, above = [], []
    for d in range(-40, 41):
        cb = c0 + d   # negative c: a larger bit pattern is a more negative value
        m = m_of(f32(cb))
        (below if m < one_em4 else above).append((m, cb))
    below = [cb for _, cb in sorted(below, reverse=True)[:3]]
    above = [cb for _, cb in sorted(above)[:3]]
    far = [bits32(0.0) | 0, bits32(-0.01), bits32(-1.0), bits32(-1e30)]
    return below + above + far


def zero_values():
    return [0x00000000, 0x80000000, 0x00000001, 0x80000001, bits32(1e-30), bits32(-1e-30)]


def subnormal_values():
    """c with fma(100, c, 1) = 1 + k 2^-23 or 1 - k 2^-24: FP16-subnormal products of both signs (no tie among them)."""
    out = []
    for m in [1.0 + k * 2.0**-23 for k in (1, 2, 7, 100, 3000)] + [1.0 - k * 2.0**-24 for k in (1, 3, 50, 6000)]:
        cb = bits32(np.float32((m - 1.0) / 100.0))
        assert m_of(f32(cb)) == Fraction(m)
        out.append(cb)
    return out


def x_step(x, d):
    """The FP32 value d ulps from the positive FP32 value x (same binade)."""
    return Fraction(f32(bits32(float(x)) + d))


def detecting_tie(x):
    """(side) when kEncK x is an FP32-product tie that a second rounding gets wrong, else None (exact)."""
    pe = x * K
    p32 = rn32(pe)
    if not fp16_midpoint(p32) or pe == p32 or rn16_bits(pe) == rn16_bits(p32):
        return None
    return "up" if pe > p32 else "down"


def c_for_log(xd):
    """A colour c (FP32 bits) whose fma(100, c, 1) = m has the correctly rounded log2 xd (exact checks), or None."""
    mt = 2.0 ** float(xd)
    m0 = bits32(np.float32(mt))
    for dm in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        m = Fraction(f32(m0 + dm))
        x, margin = log2_exact(m)
        if x != xd or margin is None or margin < Fraction(1, 10**12):
            continue
        c0 = bits32(np.float32(float((m - 1) / 100)))
        for dc in (0, 1, -1, 2, -2, 3, -3):
            if m_of(f32(c0 + dc)) == m:
                return c0 + dc
    return None


def offset_ties(rng, count):
    """count ties x in [8, 100) (exact products alternately above and below the midpoint), each with the colours whose
    predicted log2 is x - d ulp for every d in OFFSETS: a device log2 that is uniformly d ulp off still lands one of
    them on x. [(side, x, [c bits for d in OFFSETS])]."""
    out, want_up = [], True
    while len(out) < count:
        x = Fraction(f32(bits32(np.float32(rng.uniform(8.0, 100.0)))))
        side = detecting_tie(x)
        if side is None or (side == "up") != want_up:
            continue
        if bits32(float(x)) >> 23 != (bits32(float(x)) + 8) >> 23 or bits32(float(x)) >> 23 != (bits32(float(x)) - 8) >> 23:
            continue   # all offsets in the same binade
        cs = [c_for_log(x_step(x, -d)) for d in OFFSETS]
        if any(c is None for c in cs):
            continue
        out.append((side, x, cs))
        want_up = not want_up
    return out


def shifted_detecting(colour, e):
    """Per site, the number of colours whose correctly rounded log2 moved by e ulp is a detecting tie (the CPU model
    of a device log2 uniformly e ulp off)."""
    per = [0] * SITES
    for y in range(H):
        for xx in range(W):
            for ch in range(3):
                m = m_of(f32(colour[3 * (y * W + xx) + ch]))
                if m < Fraction(f32(bits32(1e-4))):
                    continue
                x, _ = log2_exact(m)
                if x == 0:
                    continue
                xs = Fraction(f32(bits32(float(x)) + (e if x > 0 else -e)))
                if detecting_tie(xs):
                    per[3 * (y % 2) + ch] += 1
    return per


def build():
    rng = np.random.default_rng(SEED)
    specials = floor_values() + zero_values() + subnormal_values()
    groups = offset_ties(rng, TIES_PER_SITE * SITES)
    extra = search(rng, 1.0e-4, 1.0 - 3.9e-4, 72)   # negative products (sparse x: no offsets), predicted ties
    extra_cbs = extra[("up", True)] + extra[("down", True)]
    per_site = [list(specials) for _ in range(SITES)]
    for i, (side, x, cs) in enumerate(groups):
        per_site[i % SITES] += cs
    for i, cb in enumerate(extra_cbs):
        if len(per_site[i % SITES]) < PER_SITE:
            per_site[i % SITES].append(cb)
    fill = [bits32(1.0 + 0.25 * k) for k in range(PER_SITE)]   # ordinary values for any slot left
    for s_ in per_site:
        assert len(s_) <= PER_SITE
        s_ += fill[:PER_SITE - len(s_)]
    colour = [0] * (W * H * 3)
    for site in range(SITES):
        row, ch = divmod(site, 3)
        pixels = [(x, 2 * j + row) for j in range(H // 2) for x in range(W)]
        for (x, y), cb in zip(pixels, per_site[site]):
            colour[3 * (y * W + x) + ch] = cb
    counts = {"offset_ties_per_site": TIES_PER_SITE, "offsets": f"{OFFSETS.start}..{OFFSETS.stop - 1}",
              "up": sum(g[0] == "up" for g in groups), "down": sum(g[0] == "down" for g in groups),
              "negative_extra": sum(cb in set(extra_cbs) for cb in colour)}
    return colour, counts, len(specials)


def write(path):
    colour, counts, nspec = build()
    lines = ["// SPDX-License-Identifier: MIT",
             "// Generated by clean/runtime/gen_fold_selftest.py (do not edit): the render colour image of the native D3D12",
             "// device's differential self-test of Model L's post (fold port against no-fold port, device_d3d12.cpp).",
             "// FP32 bits, slot 3 (y kW + x) + channel. Synthetic values from exact rational arithmetic; no model data.",
             "// Per site (row of the pair, channel): %d special values (floor, zero and FP16-subnormal products), then" % nspec,
             ("// per site %d FP32-product ties a second rounding gets wrong, each at predicted-log2 offsets %s ulp (%d exact\n"
              "// products above the midpoint, %d below), then %d predicted ties of negative products over all sites.") % (
                 counts["offset_ties_per_site"], counts["offsets"], counts["up"], counts["down"], counts["negative_extra"])]
    lines += ["#pragma once", "#include <cstdint>", "", "namespace fold_selftest {", "",
              f"constexpr int kW = {W}, kH = {H};", f"constexpr uint32_t kColour[{W * H * 3}] = {{"]
    for i in range(0, len(colour), 8):
        lines.append("    " + ", ".join(f"0x{v:08x}u" for v in colour[i:i + 8]) + ",")
    lines += ["};", "", "}  // namespace fold_selftest", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return counts


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fold_selftest.h")
    print(write(out))
