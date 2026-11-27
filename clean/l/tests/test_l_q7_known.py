# SPDX-License-Identifier: GPL-3.0-only
"""Known answers for the FP16 block form of Model L's oracle (clean/l/reference/oracle.py, form "q7"), with nonzero
weights. The expected values are worked out here independently of the oracle: by hand (the block scenario) or with an
exact rational reference (fractions.Fraction with its own round-to-nearest-even to FP16 and FP8). Every check has
negative controls: the oracle mutated in one way (chunk order, a missing chunk, a rounding moved across a chunk
boundary, the wrong W2 slice, per-product accumulation, ties away from zero, subnormals flushed, overflow saturated,
the attention-weight and GELU constants) must fail it.

    python -m unittest clean.l.tests.test_l_q7_known
"""
import contextlib
import inspect
import math
import textwrap
import unittest
from fractions import Fraction as Fr

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.reference.fp8 import f16 as F16

# ---- independent exact reference --------------------------------------------------------------------------------------


def _floor_log2(a):
    """floor(log2 a) of a positive Fraction."""
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fr(2) ** e > a:
        e -= 1
    while Fr(2) ** (e + 1) <= a:
        e += 1
    return e


def rn16(v):
    """Round a Fraction to FP16, nearest even, subnormals kept, overflow to infinity (as a float)."""
    v = Fr(v)
    if v == 0:
        return 0.0
    a = abs(v)
    if a >= 65520:                       # halfway between 65504 and 2^16 and above: infinity
        return math.copysign(math.inf, v)
    q = Fr(2) ** (max(_floor_log2(a), -14) - 10)
    return float(math.copysign(round(a / q) * q, v))     # Fraction round(): half to even


def rq8(v):
    """Round to FP8 E4M3FN, nearest even, saturating at +-448 (as a float)."""
    v = Fr(v)
    if v == 0:
        return 0.0
    a = abs(v)
    q = Fr(2) ** (max(_floor_log2(a), -6) - 3)
    return float(math.copysign(min(round(a / q) * q, Fr(448)), v))


def bits16(x):
    return int(np.array([x], np.float16).view(np.uint16)[0])


def from_bits16(b):
    return float(np.array([b], np.uint16).view(np.float16)[0])


def ref_weights(s):
    """Attention weights of one row of 16 FP16 scores, written out step by step."""
    c, lim = Fr(from_bits16(0x2466)), Fr(from_bits16(0x3873))
    t0, p0 = Fr(from_bits16(0x3B6B)), Fr(from_bits16(0x3D80))
    e = []
    for si in s:
        u = Fr(rn16(Fr(si) * c))
        u = min(max(u, -lim), lim)
        t = Fr(rn16(u * -u + t0))                       # one rounding (fused)
        p = rn16(u * t + p0)
        e.append(Fr(from_bits16((bits16(p) & 0x03FF) << 5)))
    a = [Fr(rn16(e[2 * k] + e[8 + 2 * k])) for k in range(4)]
    b = [Fr(rn16(e[2 * k + 1] + e[9 + 2 * k])) for k in range(4)]
    A = Fr(rn16(Fr(rn16(a[0] + a[1])) + Fr(rn16(a[2] + a[3]))))
    B = Fr(rn16(Fr(rn16(b[0] + b[1])) + Fr(rn16(b[2] + b[3]))))
    total = rn16(A + B)
    w = Fr(rn16(Fr(float(np.float32(1) / np.float32(total)))))   # (the reciprocal's last bit is free)
    return [rn16(ei * w) for ei in e]


def ref_gelu(h):
    h = Fr(h)
    y = min(max(h, Fr(-2)), Fr(2))
    a, b = Fr(from_bits16(0x3698)), Fr(from_bits16(0x2D30))
    inner = Fr(rn16(a - Fr(rn16(b * abs(y)))))
    return rn16(h * Fr(rn16(Fr(1, 2) + Fr(rn16(y * inner)))))


def ref_norm(x, gamma):
    """RMS norm of one pixel (FP16 values x, C channels) as the tree: lane t, half (even / odd channel), tiles."""
    C = len(x)
    q = [Fr(rn16(Fr(v) * Fr(v))) for v in x]
    tiles = C // 8

    def grp(ps):
        if len(ps) == 1:
            return ps[0]
        if len(ps) == 2:
            return Fr(rn16(ps[0] + ps[1]))
        if len(ps) == 3:
            return Fr(rn16(Fr(rn16(ps[0] + ps[1])) + ps[2]))
        return Fr(rn16(Fr(rn16(ps[0] + ps[1])) + Fr(rn16(ps[2] + ps[3]))))

    def lane(t, odd):
        Q = [q[8 * j + 2 * t + odd] for j in range(tiles)]
        left = grp([Fr(rn16(Q[j] + Q[j + 2])) for j in range(0, tiles, 4)])
        right = grp([Fr(rn16(Q[j] + Q[j + 2])) for j in range(1, tiles, 4)])
        return Fr(rn16(left + right))

    def half(odd):
        e = [lane(t, odd) for t in range(4)]
        return Fr(rn16(Fr(rn16(e[0] + e[1])) + Fr(rn16(e[2] + e[3]))))

    s = rn16(Fr(rn16(half(0) + half(1))) + Fr(1, 2 ** 13))
    r = Fr(rn16(Fr(float(np.float32(1) / np.sqrt(np.float32(s))))))
    g = [Fr(rn16(Fr(float(np.float32(gm) * np.sqrt(np.float32(C)))))) for gm in gamma]
    return [rn16(Fr(v) * Fr(rn16(r * gi))) for v, gi in zip(x, g)]


# ---- mutation of the oracle -------------------------------------------------------------------------------------------

@contextlib.contextmanager
def mutated(name, old, new):
    """The oracle's function `name` with `old` replaced by `new` in its source (exactly one occurrence)."""
    f = getattr(L, name)
    src = textwrap.dedent(inspect.getsource(f))
    if src.count(old) != 1:
        raise AssertionError(f"mutation of {name}: {old!r} occurs {src.count(old)} times")
    ns = dict(L.__dict__)
    exec(compile(src.replace(old, new), f"<mutant {name}>", "exec"), ns)
    setattr(L, name, ns[name])
    try:
        yield
    finally:
        setattr(L, name, f)


@contextlib.contextmanager
def patched(name, value):
    old = getattr(L, name)
    setattr(L, name, value)
    try:
        yield
    finally:
        setattr(L, name, old)


def f16_ties_away(x):
    x = np.asarray(x, np.float64)
    r = F16(x)
    lo, hi = np.nextafter(r.astype(np.float16), np.float16(-np.inf)), np.nextafter(r.astype(np.float16), np.float16(np.inf))
    lo, hi = lo.astype(np.float64), hi.astype(np.float64)
    tie_lo = np.isfinite(r) & (x - lo == r - x) & (np.abs(lo) > np.abs(r))
    tie_hi = np.isfinite(r) & (hi - x == x - r) & (np.abs(hi) > np.abs(r))
    return np.where(tie_lo, lo, np.where(tie_hi, hi, r))


def f16_flush(x):
    r = F16(x)
    return np.where(np.abs(r) < 2.0 ** -14, 0.0 * r, r)


def f16_saturate(x):
    r = F16(x)
    return np.where(np.isinf(r) & ~np.isinf(np.asarray(x, np.float64)), np.copysign(65504.0, r), r)


# ---- the checks (each returns True when the oracle gives the known answer) ---------------------------------------------

def chain_case():
    """K = 64 (two chunks). Start 1024 (FP16 spacing 1): chunk 0 adds 0.75 -> RN16(1024.75) = 1025, chunk 1 adds 0.5 ->
    RN16(1025.5) = 1026 (tie, even). Reversed: 1024.5 -> 1024 (tie, even), 1024.75 -> 1025. Exact across chunks:
    1025.25 -> 1025. Chunk 1 missing: 1025."""
    a = np.zeros(64)
    a[3], a[40] = 0.75, 0.5
    W = np.zeros((2, 64))
    W[:, 3], W[:, 40] = 1.0, 1.0
    W[1] *= -1.0
    got = L.mm8(np.array([1024.0, -1024.0]), a, W)
    return list(got) == [1026.0, -1026.0]


def op_case():
    """One 32-term op, start 2048 (FP16 spacing 2), four products of 0.5: the exact sum 2050 is FP16. A rounding after
    each product (2048.5 -> 2048, tie even) gives 2048. Sixteen products of 0.125 (2050 exactly) too; truncating the
    terms to 13 bits below 2^11 (multiples of 0.25) would drop them (2048)."""
    a = np.zeros(32)
    a[:4] = 0.5
    W = np.ones((1, 32))
    b = np.zeros(32)
    b[:16] = 0.125
    return (L.mm8(np.array([2048.0]), a, W)[0] == 2050.0 and
            L.mm8(np.array([2048.0]), b, W)[0] == 2050.0)


def trunc13_case():
    """The truncating model on the same op: 0.125 is below the quantum 2^(11 - 13) = 0.25, so 2048 stays; 0.25 terms
    survive (sixteen give 2052); block 2's start is block 1's FP16 result."""
    b = np.zeros(32)
    b[:16] = 0.125
    c = np.zeros(32)
    c[:16] = 0.25
    W = np.ones((1, 32))
    return (L.mm8(np.array([2048.0]), b, W, "trunc13")[0] == 2048.0 and
            L.mm8(np.array([2048.0]), c, W, "trunc13")[0] == 2052.0)


PROBES = (   # (k, a, b) products, start, result bits (exact model), result bits (trunc13 model)
    (((0, 2.0 ** -2, 2.0 ** -9), (1, 2.0 ** -9, 2.0 ** -9)), 1.0, 0x3C01, 0x3C00),   # 1 + 2^-11 + 2^-18
    (((0, -2.0 ** -3, 2.0 ** -9), (1, -2.0 ** -9, 2.0 ** -9)), 1.0, 0x3BFF, 0x3C00),  # 1 - 2^-12 - 2^-18
    (((0, 1.0, 1.0), (1, 2.0 ** -2, 2.0 ** -9), (16, 2.0 ** -2, 2.0 ** -9)), 0.0, 0x3C01, 0x3C00),  # across the blocks
    # 1 + 2^-11 (a tie) + 0.75 x 2^-13: truncated to 0 the tie goes to even (1.0); rounded it would pass the tie
    (((0, 2.0 ** -2, 2.0 ** -9), (1, 1.5 * 2.0 ** -6, 2.0 ** -8)), 1.0, 0x3C01, 0x3C00),
)


def probe_case(model):
    """Single ops that tell the two models apart (worked out by hand: the exact sum is just past / short of an FP16 tie;
    truncation to 13 bits below 2^0 drops the 2^-18 term, and block 1 rounds 1 + 2^-11 to 1 before block 2)."""
    ok = True
    for prods, start, b_exact, b_trunc in PROBES:
        a, W = np.zeros(32), np.zeros((1, 32))
        for k, x, w in prods:
            a[k], W[0, k] = x, w
        got = L.mm8(np.array([start]), a, W, model)[0]
        ok = ok and bits16(got) == (b_exact if model == "exact" else b_trunc)
    return ok


def rounding_edges_case():
    """FP16 ties to even (1024.5 -> 1024, 1025.5 -> 1026), subnormals kept (2.5 x 2^-24 -> 2 x 2^-24, 1.5 x 2^-24 -> 2^-23)
    and overflow to infinity (65504 + 32): through one FP8 op and through the GELU (h -> h / 2 for tiny h)."""
    W = np.ones((1, 32))
    a = np.zeros(32)
    a[0] = 0.5
    t1 = L.mm8(np.array([1024.0]), a, W)[0] == 1024.0 and L.mm8(np.array([1025.0]), a, W)[0] == 1026.0
    a2 = np.zeros(32)
    a2[0], a2[1] = 8.0, 4.0                          # 32
    t2 = np.isposinf(L.mm8(np.array([65504.0]), a2, np.array([[4.0, 0.0] + [0.0] * 30]))[0])
    t2 = t2 and np.isposinf(L.mm8(np.array([65504.0]), np.r_[8.0, 4.0, np.zeros(30)], np.r_[2.0, 4.0, np.zeros(30)][None])[0])
    g = L.gelu_q7(np.array([5 * 2.0 ** -24, 3 * 2.0 ** -24]))
    t3 = list(g) == [rn16(Fr(5, 2 ** 25)), rn16(Fr(3, 2 ** 25))] == [2 * 2.0 ** -24, 2.0 ** -23]
    return bool(t1 and t2 and t3)


def block_case():
    """A whole block (C 32, 2 heads, MLP 128) with nonzero weights, worked out by hand:
    x = 1 everywhere, gamma1 = gamma2 = 1: the norm gives 1 (sum 32, r = RN16(rsqrt 32) = 0.1767578125,
    RN16(r x 5.65625) = 1.0); Wqk = 0 and B = 0: every score 0, every weight 1/16; Wv[h][0][0] = 1: v = 1 in channel 0 of
    each head, o = v; Wp[:, 0] = 0.75 (head 0), Wp[:, 32] = 0.5 (head 1), bp = 1023: z = 1024 -> 1025 -> 1026 (head order;
    reversed 1025, exact 1025); norm 2 of 1026: the squares overflow, r = 0, the MLP input is 0; b1 slices 2, 1, 2, -2:
    GELU 2, 0.8310546875 -> FP8 0.8125, 2, 0; W2[:, 0] = 0.375 (slice 0), W2[:, 64] = 0.25 (slice 2), b2 = -2: the chain
    from RN16(-2 + 1026) = 1024 adds 0.75 -> 1025, 0, 0.5 -> 1026 (tie, even), 0. Output 1026 in every channel."""
    p = {k: np.zeros_like(v, dtype=np.float64) for k, v in L.random_block_params(np.random.default_rng(0), 32, 2, 128).items()}
    p["gamma1"] = p["gamma2"] = np.ones(32)
    p["Wv"][0, 0, 0] = p["Wv"][1, 0, 0] = 1.0
    p["Wp"][:, 0], p["Wp"][:, 32] = 0.75, 0.5
    p["bp"] = np.full(32, 1023.0)
    p["b1"] = np.repeat([2.0, 1.0, 2.0, -2.0], 32)
    p["W2"][:, 0], p["W2"][:, 64] = 0.375, 0.25
    p["b2"] = np.full(32, -2.0)
    x = np.ones((2, 16, 32))
    got = L._block_tokens(x, p, 2, T.block_consts(T.TABLE, "dec0_block"))
    return bool(np.all(got == 1026.0))


def weights_case():
    rng = np.random.default_rng(11)
    rows = [np.zeros(16), np.r_[60.0, -60.0, np.zeros(14)], L.f16(rng.normal(0, 8, 16)), L.f16(rng.normal(0, 20, 16)),
            np.linspace(-31, 31, 16)]
    return all(list(L.softmax_q7(r)) == ref_weights(r) for r in rows)


def gelu_case():
    h = np.r_[L.f16(np.linspace(-2.5, 2.5, 41)), 1.0, 0.3, -0.7, 1.9]
    h = L.f16(h)
    return list(L.gelu_q7(h)) == [ref_gelu(v) for v in h] and L.gelu_q7(np.array([1.0]))[0] == 0.8310546875


def norm_case():
    rng = np.random.default_rng(12)
    ok = True
    for C in (32, 64, 96, 128):
        for scale in (0.5, 4.0, 16.0):           # (sums of squares below the FP16 range)
            x = L.f16(rng.normal(0, scale, C))
            g = L.f16(rng.uniform(0.5, 1.5, C))
            ok = ok and list(L.norm_q7(x, L.gamma_q7(g))) == ref_norm(x, g)
    return ok


# ---- tests ------------------------------------------------------------------------------------------------------------

REV_CHUNKS = ("for c0 in range(0, W8.shape[1], 32):", "for c0 in reversed(range(0, W8.shape[1], 32)):")
MISS_CHUNK = ("for c0 in range(0, W8.shape[1], 32):", "for c0 in range(0, max(32, W8.shape[1] - 32), 32):")
EXACT_ACROSS = ("    acc = start\n    for c0 in range(0, W8.shape[1], 32):\n"
                "        acc = _op8(acc, a8[..., c0:c0 + 32], W8[:, c0:c0 + 32], model)\n    return acc",
                "    return f16(np.asarray(start, np.float64) + a8 @ W8.T)")
PER_PRODUCT = ("            return f16(start + a8 @ W8.T)",
               "            acc = start\n            for i in range(a8.shape[-1]):\n"
               "                acc = f16(acc + a8[..., i:i + 1] @ W8[:, i:i + 1].T)\n            return acc")
W2_SLICE = ('acc = mm8(acc, q8(gelu_q7(hs)), W["W2"][:, r], model, final=(model != "f32chain" or sl == nsl - 1))',
            'acc = mm8(acc, q8(gelu_q7(hs)), W["W2"][:, slice(32 * ((sl + 1) % 4), 32 * ((sl + 1) % 4) + 32)], model, '
            'final=(model != "f32chain" or sl == nsl - 1))')
MLP_REVERSED = ('for sl in range(nsl):', 'for sl in reversed(range(nsl)):')


class Q7KnownAnswerTest(unittest.TestCase):
    def test_reference_rounding(self):
        """The rational reference itself: ties to even, subnormals, overflow, FP8 saturation."""
        self.assertEqual((rn16(Fr(2049, 2)), rn16(Fr(2051, 2))), (1024.0, 1026.0))
        self.assertEqual((rn16(Fr(5, 2 ** 25)), rn16(Fr(3, 2 ** 25))), (2 * 2.0 ** -24, 2.0 ** -23))
        self.assertEqual((rn16(65519), rn16(65520)), (65504.0, math.inf))
        self.assertEqual((rq8(Fr(17, 16)), rq8(Fr(19, 16)), rq8(500)), (1.0, 1.25, 448.0))
        self.assertEqual(L.f16(np.array([1024.5, 65520.0]))[0], 1024.0)

    def test_chunk_chain(self):
        self.assertTrue(chain_case())
        for name, mut in (("mm8", REV_CHUNKS), ("mm8", MISS_CHUNK), ("mm8", EXACT_ACROSS)):
            with self.subTest(mutant=mut[1][:40]), mutated(name, *mut):
                self.assertFalse(chain_case())

    def test_one_op_accumulation(self):
        self.assertTrue(op_case())
        with mutated("_op8", *PER_PRODUCT):
            self.assertFalse(op_case())
        with mutated("mm8", 'model="exact"', 'model="trunc13"'):      # the other model as the default
            self.assertFalse(op_case())

    def test_trunc13_model(self):
        self.assertTrue(trunc13_case())
        with mutated("_trunc13_block", "emax - 13", "emax - 16"):
            self.assertFalse(trunc13_case())

    def test_model_probes(self):
        self.assertTrue(probe_case("exact") and probe_case("trunc13"))
        with mutated("_trunc13_block", "np.trunc(terms / quantum)", "np.round(terms / quantum)"):
            self.assertFalse(probe_case("trunc13"))                  # rounding instead of truncation
        with mutated("_op8", "            for b0 in (0, 16):\n                acc = _trunc13_block(acc, prods[..., b0:b0 + 16])",
                     "            acc = _trunc13_block(acc, prods)"):      # one block of 32 instead of two of 16
            self.assertFalse(probe_case("trunc13"))
        with mutated("_trunc13_block", "emax - 13", "emax - 20"):
            self.assertFalse(probe_case("trunc13"))

    def test_rounding_edges(self):
        self.assertTrue(rounding_edges_case())
        for f in (f16_ties_away, f16_flush, f16_saturate):
            with self.subTest(mutant=f.__name__), patched("f16", f):
                self.assertFalse(rounding_edges_case())

    def test_block_by_hand(self):
        self.assertTrue(block_case())
        for name, mut in (("mm8", REV_CHUNKS), ("mm8", MISS_CHUNK), ("mm8", EXACT_ACROSS),
                          ("_block_tokens_q7", W2_SLICE), ("_block_tokens_q7", MLP_REVERSED)):
            with self.subTest(mutant=mut[1][:40]), mutated(name, *mut):
                self.assertFalse(block_case())

    def test_attention_weights(self):
        self.assertTrue(weights_case())
        for name, value in (("Q7_POLY_T", L._h(0x3B6C)), ("Q7_POLY_P", L._h(0x3D81)), ("Q7_SCORE_SCALE", L._h(0x2467)),
                            ("Q7_SCORE_CLAMP", L._h(0x3800))):
            with self.subTest(mutant=name), patched(name, value):
                self.assertFalse(weights_case())
        with mutated("softmax_q7", "(pb & 0x03FF) << 5", "(pb & 0x03FF) << 4"):
            self.assertFalse(weights_case())
        with mutated("softmax_q7", "e[..., lo + 8 + 2 * t]", "e[..., lo + 1 + 2 * t]"):
            self.assertFalse(weights_case())

    def test_gelu(self):
        self.assertTrue(gelu_case())
        for name, value in (("Q7_GELU_A", L._h(0x3699)), ("Q7_GELU_B", L._h(0x2D31))):
            with self.subTest(mutant=name), patched(name, value):
                self.assertFalse(gelu_case())
        with mutated("gelu_q7", "f16(h * f16(0.5", "f16(y * f16(0.5"):     # the clamped y outside instead of h
            self.assertFalse(gelu_case())

    def test_norm(self):
        self.assertTrue(norm_case())
        with mutated("norm_q7", "range(1, tiles, 4)", "range(0, tiles, 4)"):
            self.assertFalse(norm_case())
        with mutated("gamma_q7", "np.sqrt(np.float32(C))", "np.sqrt(np.float32(32.0))"):
            self.assertFalse(norm_case())
        with mutated("norm_q7", "return f16(f16(e[0] + e[1]) + f16(e[2] + e[3]))",
                     "return f16(f16(f16(e[0] + e[1]) + e[2]) + e[3])"):
            self.assertFalse(norm_case())


if __name__ == "__main__":
    unittest.main()
