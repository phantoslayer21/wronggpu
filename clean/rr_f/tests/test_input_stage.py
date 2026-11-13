# SPDX-License-Identifier: MIT
"""Spec 52 (input stage) reference: scalar rules, exhaustive quantisation, sampler, per-frame constants, camera maps,
per-token features against the independent slow oracle (tests/slow_input.py), front-end MLP, motion surfaces,
reset independence and the stage call.

    python -m unittest clean.rr_f.tests.test_input_stage
RR_INPUT_TOKENS (default 24) = tokens per feature case checked against the slow oracle (corners and padding
tokens always included)."""
import math
import os
import unittest

import numpy as np

from clean.rr_f.reference import f32 as F
from clean.rr_f.reference import input_stage as IS
from clean.rr_f.reference.fmt import ALL_F16, d8, from_f16_bits, q8, round_f16, to_f16_bits
from clean.rr_f.reference.stages import run_stage
from clean.rr_f.tests import slow, slow_input as SL, synth, synth_input as SI

NTOK = int(os.environ.get("RR_INPUT_TOKENS", "24"))


def same(a, b):
    """Bitwise f32 equality, NaN == NaN (any payload), +0 != -0."""
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    nan = np.isnan(a) & np.isnan(b)
    return (a.view(np.uint32) == b.view(np.uint32)) | nan


def special_f32(rng, n):
    edges = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.0, -1.0, 2.0 ** -126, -2.0 ** -126, 2.0 ** -127,
                      3e38, -3e38, 65504.0, 65520.0, 1e-30, 0.5, 2.0 ** -149], np.float32)
    v = (rng.normal(0, 1, n) * np.exp2(rng.integers(-130, 130, n))).astype(np.float32)
    m = rng.random(n) < 0.15
    v[m] = rng.choice(edges, m.sum())
    # values near the smallest normal and products that round to it
    k = rng.random(n) < 0.05
    v[k] = (np.float32(2.0 ** -63) * rng.uniform(0.99, 1.01, k.sum())).astype(np.float32)
    return v


class TestF32(unittest.TestCase):
    def test_binary_ops(self):
        rng = np.random.default_rng(520)
        a, b, c = special_f32(rng, 4000), special_f32(rng, 4000), special_f32(rng, 4000)
        # products of pairs around 2^-63 land next to the smallest normal (round-up-then-keep rule)
        for name, vf, sf in (("add", F.add, SL.add), ("sub", F.sub, SL.sub), ("mul", F.mul, SL.mul),
                             ("div", F.div, SL.div)):
            got = vf(a, b)
            exp = np.array([sf(float(x), float(y)) for x, y in zip(a, b)], np.float32)
            bad = ~same(got, exp)
            self.assertFalse(bad.any(), (name, a[bad][:3], b[bad][:3], got[bad][:3], exp[bad][:3]))
        got = F.fma(a, b, c)
        exp = np.array([SL.fma(float(x), float(y), float(z)) for x, y, z in zip(a, b, c)], np.float32)
        bad = ~same(got, exp)
        self.assertFalse(bad.any(), (a[bad][:3], b[bad][:3], c[bad][:3], got[bad][:3], exp[bad][:3]))

    def test_fma_midpoints(self):
        # products whose sum with c lies exactly on an f32 midpoint plus a tiny residual
        rng = np.random.default_rng(5201)
        a = (1 + rng.integers(1, 2 ** 23, 3000) * 2.0 ** -23).astype(np.float32)
        b = (1 + rng.integers(1, 2 ** 12, 3000) * 2.0 ** -23).astype(np.float32)
        c = -(a.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
        got = F.fma(a, b, c)
        exp = np.array([SL.fma(float(x), float(y), float(z)) for x, y, z in zip(a, b, c)], np.float32)
        self.assertTrue(same(got, exp).all())

    def test_unary(self):
        rng = np.random.default_rng(5202)
        x = special_f32(rng, 1500)
        x = np.concatenate([x, np.abs(x), (1 + rng.integers(0, 2 ** 23, 500) * 2.0 ** -23).astype(np.float32)])
        for name, vf, sf in (("sqrt", F.sqrt, SL.fsqrt), ("log2", F.log2, SL.flog2), ("rcp", F.rcp, SL.rcp)):
            got = vf(x)
            exp = np.array([sf(float(v)) for v in x], np.float32)
            bad = ~same(got, exp)
            self.assertFalse(bad.any(), (name, x[bad][:4], got[bad][:4], exp[bad][:4]))
        t = np.concatenate([rng.uniform(-140, 130, 1500), rng.uniform(-1, 1, 500),
                            [0, -0.0, 128, 127.99999, -126, -126.0001, -149, -150, np.nan, np.inf, -np.inf]])
        t = t.astype(np.float32)
        got = F.exp2(t)
        exp = np.array([SL.fexp2(float(v)) for v in t], np.float32)
        self.assertTrue(same(got, exp).all())

    def test_constants_and_minmax(self):
        self.assertEqual(F.LOG10_2, np.float32(SL.LOG10_2))
        self.assertEqual(F.LOG2_10, np.float32(SL.LOG2_10))
        self.assertEqual(F.LN_2, np.float32(SL.LN_2))
        v = np.float32([np.nan, 1.0, np.nan, -np.inf])
        w = np.float32([2.0, np.nan, np.nan, 3.0])
        self.assertTrue(same(F.fmin(v, w), [2.0, 1.0, np.nan, -np.inf]).all())
        self.assertTrue(same(F.fmax(v, w), [2.0, 1.0, np.nan, 3.0]).all())


def quant_int(xbits, d):
    """Independent integer form of 52.4 for f16 inputs given as bit patterns (vectorised, no float arithmetic):
    returns codes."""
    xbits = xbits.astype(np.int64)
    d = d.astype(np.int64)
    sign = xbits >> 15
    E = (xbits >> 10) & 31
    M = xbits & 1023
    nan = (E == 31) & (M != 0)
    inf = (E == 31) & (M == 0)
    # magnitudes in units of 2^-24: |x| = (E ? 1024 + M : M) << max(E - 1, 0)
    sh = np.maximum(E - 1, 0)
    xm = np.where(E > 0, 1024 + M, M) << sh
    fbm = np.where(E > 0, 1024 << sh, 0)
    fam = np.where(E > 0, (1024 + d) << sh, d)
    def round16(m):
        # round an integer magnitude (units 2^-24) to binary16 (RNE); returns magnitude units, or -1 for inf
        out = np.empty_like(m)
        for i in range(m.size):
            v = int(m.flat[i])
            if v < 2048:
                out.flat[i] = v
                continue
            e = v.bit_length() - 11
            q, r = v >> e, v & ((1 << e) - 1)
            h = 1 << (e - 1)
            if r > h or (r == h and q & 1):
                q += 1
            v2 = q << e
            out.flat[i] = -1 if v2 >= 65520 * 2 ** 24 else v2
        return out
    s1 = round16(xm + fam)
    s2 = np.where(s1 < 0, -1, s1 - fbm)                  # s1 - fb is exact here (same binade or one above)
    code = np.zeros_like(xm)
    for i in range(code.size):
        v = int(s2.flat[i])
        if v < 0:
            code.flat[i] = 0x7E
            continue
        if v == 0:
            continue
        e = v.bit_length() - 1                          # value exponent (units 2^-24): 2^(e-24)
        keep = max(e - 3, 0)
        tv = (v >> keep) << keep                         # truncated to 4 significant bits
        if tv > 448 * 2 ** 24:
            code.flat[i] = 0x7E
        elif tv >= 2 ** 18:                              # >= 2^-6: e4m3 normal
            ee = tv.bit_length() - 1
            code.flat[i] = ((ee - 24 + 7) << 3) | ((tv >> (ee - 3)) & 7)
        else:                                            # subnormal grid 2^-9 = 2^15 units
            q, r = tv >> 15, tv & (2 ** 15 - 1)
            if r > 2 ** 14 or (r == 2 ** 14 and q & 1):
                q += 1
            code.flat[i] = q
    code = np.where(inf, 0x7E, code)
    code = np.where(nan, 0x7F, code)
    return (code | (sign << 7)).astype(np.uint8)


class TestQuantise(unittest.TestCase):
    def test_exhaustive_f16(self):
        """Every f16 pattern (as an f32 feature) with every dither value 0..127, against the integer form."""
        bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
        vals = F.f32(from_f16_bits(bits))
        rng = np.random.default_rng(5204)
        dsel = list(range(128)) if os.environ.get("RR_INPUT_EXHAUSTIVE") else \
            sorted(set([0, 1, 63, 64, 127] + list(rng.integers(0, 128, 6))))
        for d in dsel:
            got = IS.quantise(vals, np.full(vals.shape, d))
            exp = quant_int(bits, np.full(bits.shape, d))
            bad = got != exp
            self.assertFalse(bad.any(), (d, bits[bad][:5], got[bad][:5], exp[bad][:5]))

    def test_f32_values_vs_exact(self):
        rng = np.random.default_rng(5205)
        v = np.concatenate([special_f32(rng, 3000),
                            (rng.normal(0, 1, 3000) * np.exp2(rng.integers(-26, 17, 3000))).astype(np.float32),
                            np.float32([65519.99, 65520.0, -65520.0, 480.0, 479.9, 448.0, 2.0 ** -15, 2.0 ** -10,
                                        3 * 2.0 ** -11, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26])])
        d = rng.integers(0, 128, v.size)
        got = IS.quantise(v, d)
        exp = np.array([SL.s_quant(float(x), int(dd)) for x, dd in zip(v, d)], np.uint8)
        bad = got != exp
        self.assertFalse(bad.any(), (v[bad][:5], d[bad][:5], got[bad][:5], exp[bad][:5]))

    def test_rules(self):
        # zero keeps the sign of x; NaN by sign; +-inf saturate
        self.assertEqual(int(IS.quantise(np.float32(-0.0), 5)), 0x80)
        self.assertEqual(int(IS.quantise(np.float32(0.0), 127)), 0x00)
        self.assertEqual(int(IS.quantise(np.float32(np.inf), 0)), 0x7E)
        self.assertEqual(int(IS.quantise(np.float32(-np.inf), 0)), 0xFE)
        self.assertEqual(int(IS.quantise(-np.float32(np.nan), 0)), 0xFF)
        self.assertEqual(int(IS.quantise(np.float32(1e6), 0)), 0x7E)
        # truncation: 1 + 15/128 (d = 0) truncates to 1 + 1/8 - nothing rounds up
        self.assertEqual(int(IS.quantise(np.float32(1 + 15 / 128), 0)), 0x38)
        # dither 127 pushes 1 + 1/8 - 2^-10 up to 1 + 1/8
        self.assertEqual(int(IS.quantise(np.float32(1 + 1 / 8 - 2 ** -10), 127)), 0x39)

    def test_dither_index(self):
        D = np.arange(2048, dtype=np.uint16) * 32
        X, Y = np.meshgrid(np.arange(40), np.arange(35))
        for fc in range(8):
            d = IS.dither_values(D, fc, X, Y)
            i = (fc * 16 + Y % 16) * 16 + X % 16
            self.assertTrue(np.array_equal(d, (i * 32) >> 9))

    def test_nu_permutation(self):
        perm = [IS.nu(k) for k in range(64)]
        self.assertEqual(sorted(perm[:32]), list(range(32)))
        self.assertEqual(sorted(perm[32:]), list(range(32, 64)))
        zero_k = {k for k in range(64) if k not in IS.FEATURE_NAMES or k in (33, 53)}
        zero_rows = {k for k in range(64) if IS.nu(k) >= 43}
        self.assertTrue(zero_rows <= zero_k)
        self.assertEqual(len(zero_k - zero_rows), 1)        # 56.3.2: one constant-zero position has a row
        self.assertEqual(len(IS.FEATURE_NAMES), 44)


class TestBilinear(unittest.TestCase):
    def test_vs_slow(self):
        rng = np.random.default_rng(5206)
        for Wi, Hi in ((7, 5), (1, 1), (13, 2)):
            img = rng.uniform(-3, 3, (Hi, Wi, 2)).astype(np.float32)
            img[rng.random(img.shape) < 0.05] = np.inf
            img[rng.random(img.shape) < 0.03] = np.nan
            u = np.concatenate([rng.uniform(-0.2, 1.2, 300), [np.nan, np.inf, -np.inf, 0, 1, 1e30, -0.0]])
            v = np.concatenate([rng.uniform(-0.2, 1.2, 300), [0.5, np.nan, 1e-45, 1, 0, -1e30, 0.25]])
            u, v = u.astype(np.float32), v.astype(np.float32)
            got = IS.bilinear(img, u, v)
            for n in range(u.size):
                exp = SL.bil(lambda i, j: [SL.ftz(float(t)) for t in img[j, i]], Wi, Hi, float(u[n]), float(v[n]))
                self.assertTrue(same(got[n], exp).all(), (Wi, Hi, u[n], v[n], got[n], exp))


def random_params(rng):
    Wo, Ho = int(rng.integers(16, 3900)), int(rng.integers(16, 2200))
    Wr = int(rng.integers(max(1, round(Wo / 3)), Wo + 1))
    Hr = int(rng.integers(max(1, round(Ho / 3)), Ho + 1))
    j = lambda: np.float32(rng.uniform(-1, 1) if rng.random() < 0.9 else rng.choice([0.0, -0.5, 0.5, 1e-40]))
    return IS.FrameParams(render_size=(Wr, Hr), output_size=(Wo, Ho), jitter=(j(), j()), prev_jitter=(j(), j()),
                          mv_scale_raw=(np.float32(rng.choice([1.0, -1.0, 0.5]) / rng.integers(1, 4000)),
                                        np.float32(rng.uniform(-1, 1) / rng.integers(1, 4000))),
                          mv_jittered=bool(rng.random() < 0.5), E=np.float32(1), Ep=np.float32(1),
                          X3=np.float32(0), reset=bool(rng.random() < 0.3), par=int(rng.integers(0, 2)),
                          fc=int(rng.integers(0, 8)), SH=SI.shift_table(rng))


class TestFrameConstants(unittest.TestCase):
    def test_vs_slow(self):
        rng = np.random.default_rng(5207)
        for _ in range(300):
            p = random_params(rng)
            k = IS.frame_constants(p)
            c = SL.slow_constants(p)
            for name in ("Wr", "Hr", "Wt", "Ht", "Wh", "Hh", "sx", "sy", "rev"):
                self.assertEqual(getattr(k, name), c[name], name)
            for name in ("rWhWr", "rHhHr", "rWrWh", "rHrHh", "invWr", "invHr", "invWh", "invHh", "zfar"):
                self.assertTrue(same(getattr(k, name), c[name]).all(), name)
            for name, cname in (("ms", "ms"), ("mo", "mo"), ("msh", "msh"), ("hc_scale", "hcs"), ("hoff", "hoff")):
                self.assertTrue(same(np.float32(getattr(k, name)), np.float32(c[cname])).all(), (name, p))

    def test_sizes(self):
        p = random_params(np.random.default_rng(1))
        p.render_size, p.output_size = (1706, 960), (2560, 1440)
        k = IS.frame_constants(p)
        self.assertEqual((k.Wt, k.Ht, k.Wh, k.Hh), (1728, 960, 3840, 2160))
        p.output_size = (2561, 1441)
        k = IS.frame_constants(p)
        self.assertEqual((k.Wh, k.Hh), (3843, 2163))
        p.render_size = (0, 5)
        with self.assertRaises(ValueError):
            IS.frame_constants(p)


class TestCamera(unittest.TestCase):
    def test_vs_slow(self):
        rng = np.random.default_rng(5208)
        kinds = ["perspective", "reverse", "ortho", "identity", "zero", "random", "nonfinite", "static"]
        n_ok = n_fail = 0
        for i in range(48):
            kind = kinds[i % len(kinds)]
            V2C, W2V, V2Cp, W2Vp = SI.camera_kind(rng, kind)
            if i % 11 == 5:
                W2V = W2V.copy()
                W2V[:, 1] = W2V[:, 0]                      # singular
            wr, hr = int(rng.integers(8, 2000)), int(rng.integers(8, 1200))
            got = IS.camera_maps(V2C, W2V, V2Cp, W2Vp, wr, hr)
            exp = SL.slow_camera(V2C, W2V, V2Cp, W2Vp, wr, hr)
            self.assertEqual(got is None, exp is None, kind)
            if got is None:
                n_fail += 1
                continue
            n_ok += 1
            for g, e in zip(got, exp):
                self.assertTrue(same(g, np.float32(e)).all(), (kind, g, np.float32(e)))
        self.assertGreater(n_ok, 20)
        self.assertGreater(n_fail, 5)

    def test_inverse_quality(self):
        rng = np.random.default_rng(5209)
        for _ in range(20):
            A = SI.world_to_view(rng)
            inv = IS.inv4(A)
            self.assertTrue(np.allclose(inv.astype(np.float64) @ A.astype(np.float64), np.eye(4), atol=5e-4))

    def test_static_camera_zero_motion(self):
        rng = np.random.default_rng(5210)
        for kind in ("static",):
            fr, pa, S, IN0, c, wc = SI.make_case(rng, smv="Hd", camera=kind)
            k = IS.frame_constants(pa)
            out = IS.compute_features(fr, k, S, IN0["D"], trace=True)
            m2 = np.stack(out["m2"], -1)
            ok = ~out["exited"]
            self.assertTrue(np.all(np.abs(m2[ok]) < 1e-2), np.abs(m2[ok]).max())


def token_sample(rng, k, n):
    pts = {(0, 0), (k.Wt - 1, k.Ht - 1), (k.Wr - 1, k.Hr - 1), (k.Wr, 0), (0, k.Hr), (k.Wt - 1, 0),
           (min(k.Wr, k.Wt - 1), min(k.Hr, k.Ht - 1)), (k.Wr - 1, 0)}
    for _ in range(n):
        if rng.random() < 0.25:
            pts.add((int(rng.integers(min(k.Wr, k.Wt - 1), k.Wt)), int(rng.integers(0, k.Ht))))
        else:
            pts.add((int(rng.integers(0, k.Wt)), int(rng.integers(0, k.Ht))))
    return sorted((x, y) for x, y in pts if 0 <= x < k.Wt and 0 <= y < k.Ht)


FEATURE_CASES = [
    dict(),
    dict(reset=True),
    dict(edge=0.06),
    dict(rev=True, camera="reverse", nrm_dec=True),
    dict(rgh_packed=True, guides=["B", "G1", "G2", "Q"]),
    dict(smv="SV", edge=0.03, mv_jittered=True),
    dict(smv="Hd"),
    dict(smv="Hd", b2=True),
    dict(smv="Hd", hw_depth=False, edge=0.03),
    dict(smv="Hd", hw_depth=False, b2=True, camera="ortho"),
    dict(smv="Hd", camera="zero"),
    dict(smv="Hd", camera="random", b2=True, edge=0.05),
    dict(smv="Hd", camera="nonfinite"),
    dict(smv="Hd", specular_from_v=True),
    dict(Wr=19, Hr=11, tex_margin=(0, 0), flavour=1),
    dict(Wr=33, Hr=32, origins=False, flavour=1, reset=True, smv="Hd", b2=True),
    dict(hot=True, edge=0.02, guides=["B", "G1"]),
    dict(tiny=True, smv="Hd", guides=["B", "G1", "G2"]),
]


class TestFeaturesVsSlow(unittest.TestCase):
    def test_cases(self):
        for i, kw in enumerate(FEATURE_CASES):
            with self.subTest(case=i, kw=kw):
                self.check(i, kw)

    def check(self, i, kw):
        rng = np.random.default_rng(5300 + i)
        kw = dict(kw)
        kw.setdefault("Wr", int(rng.integers(9, 31)))
        kw.setdefault("Hr", int(rng.integers(7, 30)))
        fr, pa, S, IN0, consts, wc = SI.make_case(rng, **kw)
        k = IS.frame_constants(pa)
        out = IS.compute_features(fr, k, S, IN0["D"])
        c = SL.slow_constants(pa)
        self.assertEqual(c["maps"] is None, not k.camera_usable)
        Sd = dict(S.values)
        for (X, Y) in token_sample(rng, k, NTOK):
            t = SL.token(fr, pa, c, Sd, IN0["D"], X, Y, b2=S.b2)
            got = out["feat"][Y, X]
            ok = same(got, np.float32(t["f"]))
            self.assertTrue(ok.all(), (X, Y, np.flatnonzero(~ok), got[~ok], np.float32(t["f"])[~ok]))
            self.assertTrue(np.array_equal(out["codes"][Y, X], np.array(t["codes"], np.uint8)), (X, Y))
            self.assertTrue(same(out["hold"][Y, X], np.float32(t["hold"])).all(), (X, Y))
            for name in ("mO", "mN", "mF", "m2"):
                self.assertTrue(same([out[name][0][Y, X], out[name][1][Y, X]], np.float32(t[name])).all(),
                                (name, X, Y))

    def test_reset_regression(self):
        """52.3.2 / 52.11 item 10: on reset frames every history tap uses the token's own colour texel."""
        rng = np.random.default_rng(5320)
        fr, pa, S, IN0, consts, wc = SI.make_case(rng, reset=True, Wr=29, Hr=21)
        k = IS.frame_constants(pa)
        out = IS.compute_features(fr, k, S, IN0["D"])
        C = IS.fetch(fr, "C", np.minimum(np.arange(k.Wt)[None, :], k.Wr - 1) + 0 * np.arange(k.Ht)[:, None],
                     np.minimum(np.arange(k.Ht)[:, None], k.Hr - 1) + 0 * np.arange(k.Wt)[None, :])[..., :3]
        hf = IS.clip(F.mul(S.Gh, IS.ccf(C, k.E, S)), S)
        for kk, ch in ((5, 0), (8, 1), (9, 2), (24, 0), (25, 1), (28, 2), (12, 0), (17, 0), (37, 0), (41, 2)):
            self.assertTrue(same(out["feat"][..., kk], hf[..., ch]).all(), kk)
        for kk in (19, 22, 23, 26, 27, 44, 45, 48, 49, 52, 36):
            self.assertTrue(np.all(out["feat"][..., kk] == 0), kk)


class TestResetIndependence(unittest.TestCase):
    """53.12 item 1: with reset the temporal images do not matter; without, they do."""
    def test(self):
        for reset in (True, False):
            rng = np.random.default_rng(5330)
            fr, pa, S, IN0, consts, wc = SI.make_case(rng, reset=reset, Wr=27, Hr=20, smv="Hd")
            a = IS.input_stage(fr, pa, S, IN0, consts, wc)
            rng2 = np.random.default_rng(99)
            fr.PH1 = rng2.integers(0, 1024, fr.PH1.shape).astype(np.uint16)
            fr.PH2 = rng2.integers(0, 1024, fr.PH2.shape).astype(np.uint16)
            fr.HID = rng2.integers(-128, 128, fr.HID.shape).astype(np.int8)
            fr.PM_read = round_f16(rng2.uniform(-9, 9, fr.PM_read.shape)).astype(np.float16)
            b = IS.input_stage(fr, pa, S, IN0, consts, wc)
            eq = all(np.array_equal(a[n], b[n]) for n in ("codes", "x0", "y2", "hnew", "hmerged"))
            eq = eq and all(np.array_equal(a["surfaces"][n], b["surfaces"][n]) for n in ("PM", "Pe", "Pn"))
            self.assertEqual(eq, reset)


class TestFrontEnd(unittest.TestCase):
    def test_vs_slow(self):
        rng = np.random.default_rng(5340)
        wc, _ = synth.window_consts(rng, "A")
        for scale in (1.0, 6.0):
            FE = SI.fe_weights(rng, scale=scale)
            codes = synth.e4m3(rng, (5, 7, 64), 2.0, edge_small=0.05, edge_large=0.02)
            x0, tr = IS.front_end(codes, FE, wc)
            rows = d8(codes).reshape(-1, 64).tolist()
            A = []
            for br in (0, 1):
                H = slow.mm_rows(rows, d8(FE[f"W1_{br}"]).tolist(), list(FE[f"b1_{br}"]))
                A.append([[slow.q8_value(slow.activation(h, wc)) for h in row] for row in H])
            Z = slow.mm_rows(A[0], d8(FE["W2_0"]).tolist(), list(FE["b2"]))
            out = [slow.mm_rows([A[1][r]], d8(FE["W2_1"]).tolist(), Z[r])[0] for r in range(len(rows))]
            exp = np.array([[slow.q8_value(v) for v in row] for row in out])
            got = d8(x0).reshape(-1, 32)
            self.assertTrue(((got == exp) | (np.isnan(got) & np.isnan(exp))).all())


class TestSurfaces(unittest.TestCase):
    def test_origin_writes(self):
        rng = np.random.default_rng(5350)
        for o in ((0, 0), (3, 2), (-2, 1), (40, 0), (5, -3)):
            fr, pa, S, IN0, consts, wc = SI.make_case(rng, Wr=21, Hr=14, smv="SV")
            fr.textures["V"].origin = (max(o[0], 0), max(o[1], 0))
            ox, oy = fr.textures["V"].origin
            fr.textures["V"].data = np.pad(fr.textures["V"].data, ((0, 50), (0, 50), (0, 0)))
            fr.textures["SV"].data = np.pad(fr.textures["SV"].data, ((0, 50), (0, 50), (0, 0)))
            k = IS.frame_constants(pa)
            out = IS.compute_features(fr, k, S, IN0["D"])
            s = out["surfaces"]
            PM = fr.PM_write.copy()
            for y in range(k.Hr):
                for x in range(k.Wr):
                    tx, ty = x + ox, y + oy
                    if tx < k.Wr and ty < k.Hr:
                        PM[ty, tx] = [np.float16(out["mN"][0][y, x]), np.float16(out["mN"][1][y, x]),
                                      np.float16(out["m2"][0][y, x]), np.float16(out["m2"][1][y, x])]
            self.assertTrue(np.array_equal(PM.view(np.uint16), s["PM"].view(np.uint16)), o)
            untouched = np.ones((k.Hr, k.Wr), bool)
            untouched[oy:, ox:] = False
            self.assertTrue(np.array_equal(s["Pe"][untouched].view(np.uint16), fr.Pe[untouched].view(np.uint16)))


class TestStageCall(unittest.TestCase):
    def test_in0_and_hold(self):
        rng = np.random.default_rng(5360)
        fr, pa, S, IN0, consts, wc = SI.make_case(rng, Wr=37, Hr=29)
        r = IS.input_stage(fr, pa, S, IN0, consts, wc)
        k = r["k"]
        self.assertEqual(r["x0"].shape, (k.Ht, k.Wt, 32))
        W = {n: IN0[n] for n in ("Wq", "Wv", "Tb", "Wo", "bo", "W1", "b1", "W2", "b2", "Wh", "bh")}
        out, _ = run_stage("IN0", {"x": r["x0"], "hold": r["hold"]}, W, consts, wc, tuple(int(v) for v in pa.SH[k.par][0]))
        for n in ("y2", "hnew", "hmerged"):
            self.assertTrue(np.array_equal(np.asarray(out[n]), np.asarray(r[n]), equal_nan=True), n)
        self.assertTrue(np.all(r["hold"][..., 3] == 0))
        # converter groups and InputSymbols.from_groups give the same symbols
        c, b2 = SI.symbol_groups(S)
        S2 = IS.InputSymbols.from_groups(c, b2)
        self.assertEqual({n: float(v) for n, v in S2.values.items()}, {n: float(v) for n, v in S.values.items()})


if __name__ == "__main__":
    unittest.main()


class TestPrepareRoles(unittest.TestCase):
    def test_mapping(self):
        t = lambda o=(0, 0): IS.Texture(np.zeros((4, 4, 4), np.float32), o)
        base = {3: t((1, 2)), 1: t((3, 0)), 7: t(), 8: t(), 5: t(), 6: t()}
        with self.assertRaises(IS.MissingInput):
            IS.prepare_roles(base, packing=0, normals_format="R8G8B8A8_UNORM")
        tags = {**base, 0: t(), 49: t((5, 5)), 41: t((7, 7)), 58: t(), 10: t(), 42: t()}
        roles, flags = IS.prepare_roles(tags, packing=0, normals_format="B8G8R8A8_UNORM")
        self.assertIs(roles["Z"], tags[49])
        self.assertIs(roles["G1"], tags[41])
        self.assertEqual(flags, dict(hw_depth=False, rgh_packed=False, nrm_dec=False, smv=True, dis=False))
        frame = IS.InputFrame(roles)
        self.assertEqual(IS.origin_of(frame, "G1"), (1, 2))   # colour-before-fog read with C's origin
        self.assertEqual(IS.origin_of(frame, "SV"), (3, 0))
        self.assertEqual(IS.origin_of(frame, "Hd"), (0, 0))
        with self.assertRaises(IS.MissingInput):
            IS.prepare_roles({k: v for k, v in tags.items() if k != 5}, packing=0, normals_format="R8_UNORM")
        roles, flags = IS.prepare_roles({**base, 14: t(), 0: t(), 58: t()}, packing=1,
                                        normals_format="R10G10B10A2_UNORM")
        self.assertTrue(flags["hw_depth"] and flags["rgh_packed"] and flags["nrm_dec"])
        self.assertNotIn("R", roles)
        for f, v in (("R8_UNORM", True), ("R8G8_UNORM", True), ("R8G8B8A8_UNORM", True), ("R16G16B16A16_UNORM", True),
                     ("R10G10B10A2_UNORM", True), ("B8G8R8A8_UNORM", False), ("R16G16B16A16_FLOAT", False),
                     ("R8G8B8A8_SNORM", False), ("R16G16_UNORM", False), ("R8G8B8A8_UNORM_SRGB", False)):
            self.assertEqual(IS.nrm_dec_for(f), v, f)
