# SPDX-License-Identifier: MIT
"""Exhaustive / dense checks of the scalar rules of pages 55 and 51 used by the D0-R head reference.

Run: python -m unittest clean.rr_f.head.tests.test_scalar_rules -v   (about a minute)
"""
import math
import unittest

import numpy as np

from .. import block as B
from .. import exactfp as X
from .. import fp
from .. import head as Hd
from .. import vectors as V
from . import slow_oracle as S

np.seterr(all="ignore")
ALL = fp.all_f16()
FINITE = ALL[np.isfinite(ALL)]
f32 = np.float32
f64 = np.float64
REPORT = {}


def same16(a, b):
    """Bit-exact on f16 values except that zero signs and NaN payloads are not compared."""
    return np.asarray(fp.step_diff16(a, b) == 0)


class Rounding(unittest.TestCase):
    def test_r16_against_integer_rounding(self):
        rng = np.random.default_rng(1)
        v = np.concatenate([
            rng.normal(0, 1, 20000) * np.exp2(rng.integers(-30, 20, 20000)),
            rng.uniform(65400, 65600, 2000), -rng.uniform(65400, 65600, 2000)])
        # midpoints of neighbouring f16 values, exactly and +- a tiny amount (catches double rounding)
        a = FINITE[(FINITE >= 0) & (FINITE < 65504)]
        nxt = fp.from_bits16(fp.bits16(a) + 1)
        mid = (a + nxt) / 2
        v = np.concatenate([v, mid, -mid, np.nextafter(mid, 0), np.nextafter(mid, np.inf), [65520.0, 65519.99]])
        got = fp.r16(v)
        want = np.array([X.r16(float(t)) for t in v])
        self.assertTrue(np.all(same16(got, want)))
        self.assertTrue(np.all(np.signbit(got) == np.signbit(v)))

    def test_fma16_against_exact(self):
        rng = np.random.default_rng(2)
        n = 30000
        r = lambda s: fp.r16(rng.normal(0, 1, n) * np.exp2(rng.integers(-24, 16, n) * s))
        a, b, c = r(1), r(1), r(1)
        # cancellation-heavy triples: c close to -a*b
        k = n // 3
        c[:k] = fp.r16(-(a[:k] * b[:k]) * (1 + rng.normal(0, 2 ** -12, k)))
        sp = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 2 ** -24])
        a = np.concatenate([a, np.repeat(sp, 49)])
        b = np.concatenate([b, np.tile(np.repeat(sp, 7), 7)])
        c = np.concatenate([c, np.tile(sp, 49)])
        got = fp.fma16(a, b, c)
        want = np.array([X.fma(float(p), float(q), float(t)) for p, q, t in zip(a, b, c)])
        self.assertTrue(np.all(same16(got, want)))

    def test_recip_Z_all_f16(self):
        Z = FINITE[FINITE > 0]
        got = fp.r16(1.0 / Z)
        want = np.array([X.div(1.0, float(z), X.rnd16) for z in Z])
        self.assertTrue(np.all(same16(got, want)))


class E4M3(unittest.TestCase):
    def test_q8_all_f16(self):
        got = fp.q8(ALL)
        fast = np.array([S.q8_fast(float(v)) for v in ALL], np.uint8)
        self.assertTrue(np.array_equal(got, fast))
        specials = [0x7C00, 0xFC00, 0x7E00, 0xFE00]                     # +-inf, NaN
        specials += [0x8000, 0x8001, 0x5F00, 0x5F80, 0x5FFF]            # -0, -tiny, 448 .. 511.75
        sample = list(range(0, 65536, 7)) + specials
        brute = np.array([X.q8(float(ALL[i])) for i in sample], np.uint8)
        self.assertTrue(np.array_equal(got[sample], brute))

    def test_q8_specials(self):
        c = fp.q8(np.array([np.inf, -np.inf, np.nan, -np.nan, -0.0, -2 ** -11, 2 ** -11, 464.0, 500.0, 65504.0]))
        self.assertEqual(list(c), [0x7E, 0xFE, 0x7F, 0xFF, 0x80, 0x80, 0x00, 0x7E, 0x7E, 0x7E])

    def test_d8(self):
        self.assertEqual(fp.d8(np.array([0x7E]))[0], 448.0)
        self.assertTrue(np.isnan(fp.d8(np.array([0x7F, 0xFF]))).all())
        self.assertEqual(fp.d8(np.array([0x01]))[0], 2 ** -9)
        self.assertTrue(np.signbit(fp.d8(np.array([0x80]))[0]))
        rt = fp.q8(fp.d8(np.arange(256)))
        ok = np.array([c in (0x7F, 0xFF) or rt[c] == c for c in range(256)])
        self.assertTrue(ok.all())


class Sigmoid(unittest.TestCase):
    """Page 51.4 / 55.1 sigmoid: exhaustive over all f16 inputs."""

    def test_table_against_numpy_f32_path(self):
        x32 = ALL.astype(f32)
        with np.errstate(all="ignore"):
            t = x32 * f32(-X.L32)                        # fl32(x * (-L))
            q64 = np.exp2(t.astype(f64))
            q = q64.astype(f32)
            undecided = np.isfinite(q64) & (
                (q64 * (1 - 1e-12)).astype(f32) != (q64 * (1 + 1e-12)).astype(f32))
            d = f32(1) + q
            s = f32(1) / d
        got = fp.r16_from_f32(s)
        tab = fp.sig_table()
        REPORT["sig_numpy_undecided"] = int(undecided.sum())
        self.assertTrue(np.all(same16(got[~undecided], tab[~undecided])))
        self.assertTrue(np.all(np.isnan(tab) == np.isnan(ALL)))
        self.assertEqual(list(fp.sig16(np.array([np.inf, -np.inf, 0.0, -0.0]))), [1.0, 0.0, 0.5, 0.5])

    def test_equals_expf_form_on_every_finite_input(self):
        tab = fp.sig_table()[np.isfinite(ALL)]
        alt = np.array([X.sig_expf_form(float(v)) for v in FINITE])
        diff = np.nonzero(~same16(tab, alt))[0]
        REPORT["sig_vs_expf_form_differences"] = int(diff.size)
        self.assertEqual(diff.size, 0, "inputs: %s" % [float(FINITE[i]) for i in diff[:10]])

    def test_against_correctly_rounded_logistic(self):
        tab = fp.sig_table()[np.isfinite(ALL)]
        cr = np.array([X.logistic_cr(float(v), X.F16) for v in FINITE])
        steps = fp.step_diff16(tab, cr)
        idx = np.nonzero(steps)[0]
        REPORT["sig_vs_exact_logistic"] = [(float(FINITE[i]), float(tab[i]), float(cr[i])) for i in idx]
        self.assertTrue(np.all(steps <= 1))
        self.assertEqual(idx.size, 7, "spec 55.1 says 7 of 63,488; found %d: %s" % (idx.size,
                                                                                REPORT["sig_vs_exact_logistic"]))


class Tanh(unittest.TestCase):
    def test_tanh16_table(self):
        tab = fp.tanh16_table()
        with np.errstate(all="ignore"):
            t64 = np.tanh(ALL)
        approx = fp._guarded(t64, fp.r16, lambda v: X.tanh_cr(v, X.F16), ALL)
        self.assertTrue(np.all(same16(tab, approx)))
        # equals rounding a correctly rounded f32 tanh to f16 (page 55.3)
        via32 = fp.r16_from_f32(fp.tanh32(ALL.astype(f32)))
        self.assertTrue(np.all(same16(tab, via32)))
        self.assertEqual(list(fp.tanh16(np.array([np.inf, -np.inf]))), [1.0, -1.0])
        self.assertTrue(np.signbit(fp.tanh16(np.array([-0.0]))[0]))

    def test_tanh32_rho(self):
        rng = np.random.default_rng(3)
        a = (rng.normal(0, 1, 20000) * np.exp2(rng.integers(-20, 6, 20000))).astype(f32)
        a = np.concatenate([a, np.array([0, np.inf, -np.inf, 9.0, -9.0, 1e-30], f32)])
        got = fp.tanh32(a)
        want = np.array([X.tanh_cr(float(v), X.F32) for v in a], f32)
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)))


class Stores(unittest.TestCase):
    def test_unorm8_all_f16(self):
        got = fp.unorm8_store(ALL)
        want = np.array([X.unorm8_code(float(v)) for v in ALL], np.uint8)
        self.assertTrue(np.array_equal(got, want))
        sp = fp.unorm8_store(np.array([0.5, np.nan, -0.0, np.inf, -np.inf, -1.0, 2.0, 0.0]))
        self.assertEqual(list(sp), [128, 0, 0, 255, 0, 0, 255, 0])
        # the only half-way value is 0.5
        c = np.clip(FINITE, 0, 1) * 255
        self.assertEqual(sorted(set(FINITE[(c - np.floor(c)) == 0.5])), [0.5])

    def test_snorm8_all_f16(self):
        got = fp.snorm8_store(ALL)
        want = np.array([X.snorm8_code(float(v)) for v in ALL], np.int8)
        self.assertTrue(np.array_equal(got, want))
        sp = fp.snorm8_store(np.array([0.5, -0.5, np.nan, np.inf, -np.inf, -0.0, 1.0, -1.0]))
        self.assertEqual(list(sp), [64, -64, 0, 127, -127, 0, 127, -127])
        c = np.clip(FINITE, -1, 1) * 127
        self.assertEqual(sorted(set(FINITE[np.abs(c - np.trunc(c)) == 0.5])), [-0.5, 0.5])

    def test_readback_all_codes(self):
        got = fp.unorm8_read(np.arange(256))
        want = np.array([X.unorm8_read(k) for k in range(256)])
        self.assertTrue(np.all(same16(got, want)))
        via32 = fp.r16_from_f32(np.arange(256, dtype=f32) / f32(255))
        self.assertTrue(np.all(same16(got, via32)))

    def test_two_step_rule_differs_from_direct_conversion(self):
        rng = np.random.default_rng(4)
        v = rng.random(200000).astype(f32)
        two = fp.unorm8_store(fp.r16_from_f32(v))
        direct = np.rint(v.astype(f64) * 255).astype(np.uint8)
        frac = float(np.mean(two != direct))
        REPORT["unorm8_two_step_vs_direct"] = frac
        self.assertTrue(0.01 < frac < 0.04, frac)


class BlockScalars(unittest.TestCase):
    """Attention weight (51.3 step 3) and activation (step 8) for every f16 input, both synthetic constant sets."""

    def _blk(self, kind):
        k = V.const_set(kind)
        W = {"Wq": np.zeros((1, 32, 32), np.uint8), "Wv": np.zeros((1, 32, 32), np.uint8),
             "Wo": np.zeros((1, 32, 32), np.uint8), "W1": np.zeros((32, 128), np.uint8),
             "W2": np.zeros((128, 32), np.uint8), "Tb": np.zeros((1, 12, 16, 16)), "bo": np.zeros(32),
             "b1": np.zeros(128), "b2": np.zeros(32)}
        return k, S.Block(W, k, 1.0, 1.0)

    def test_attention_weight_all_f16(self):
        for kind in ("base", "zovf"):
            k, blk = self._blk(kind)
            got = B.attention_weight(ALL, k)
            want = np.array([blk.attn(float(s)) for s in ALL])
            self.assertTrue(np.array_equal(got, want), kind)
            self.assertTrue(np.all(np.isfinite(got)) and np.all(got >= 2 ** -14) and np.all(got < 2 ** 16))
            self.assertTrue(np.all(fp.r16(got) == got))          # exactly representable

    def test_activation_all_f16(self):
        k, blk = self._blk("base")
        got = B.act(ALL, k)
        want = np.array([blk.act(float(x)) for x in ALL])
        self.assertTrue(np.all(same16(got, want)))

    def test_constant_check(self):
        ok, wmax, ovf = B.check_attention_constants(V.const_set("base"))
        self.assertTrue(ok)
        self.assertFalse(ovf)
        ok, wmax, ovf = B.check_attention_constants(V.const_set("zovf"))
        self.assertTrue(ok and ovf)
        bad = dict(V.const_set("base"), A_P0=0.5)
        self.assertFalse(B.check_attention_constants(bad)[0])
        with self.assertRaises(B.OutOfRange):
            B.attention_weight(np.array([0.0]), bad)


class Candw4(unittest.TestCase):
    """55-ERRATUM-2: the vectorised binary16 chain (head.candw4_values) against the scalar exact oracle."""
    def test_candw4_random_and_edges(self):
        rng = np.random.default_rng(55)
        vals = np.concatenate([rng.normal(0, 3, 3000), rng.normal(0, 20, 1000)])
        z = fp.r16(vals.reshape(-1, 4))
        edge = np.array([[0.5, 0.5, 0.5, 0.5], [1.0, 2.0, -1.0, 0.0], [3.0, -np.inf, 3.0, 3.0], [np.nan, 1.0, 1.0, 1.0],
                         [np.inf, 1.0, 0.0, 0.0], [-np.inf] * 4, [65504.0, -65504.0, 0.0, 6e-8], [2.0, 2.0, -30.0, -30.0]])
        z = np.concatenate([z, edge])
        w = np.stack(Hd.candw4_values(*z.T), -1)
        ref = np.array([S.candw4(r) for r in z])
        self.assertTrue(np.all(same16(w, ref)))
        self.assertTrue(np.all(w[-8] == 0.25))
        self.assertEqual(list(fp.unorm8_store(w[-8])), [64, 64, 64, 64])

    def test_tables_correctly_rounded(self):
        v = fp.all_f16()
        e, r = fp.exp2_16_table(), fp.rcp16_table()
        for i in range(0, 65536, 97):
            x = float(v[i])
            self.assertTrue(same16(np.array([e[i]]), np.array([X.exp2_cr(x, X.F16)]))[0], x)


class Candw(unittest.TestCase):
    def test_candw_dense(self):
        for y24 in (0.0, 1.5):
            y20 = ALL
            w, w2 = Hd.candw_values(y20, np.full_like(y20, y24))
            ref = np.array([S.candw(a, y24) for a in y20])
            self.assertTrue(np.all(same16(w, ref[:, 0])) and np.all(same16(w2, ref[:, 1])), y24)

    def test_candw_exact_requirements(self):
        w, w2 = Hd.candw_values(FINITE, FINITE)
        self.assertTrue(np.all(w == 0.25) and np.all(w2 == 0.25))
        inf, nan = np.inf, np.nan
        a = np.array([inf, -inf, inf, 3.0, -inf, 3.0, inf, -inf, nan, 1.0, nan])
        b = np.array([-inf, inf, 3.0, inf, 3.0, -inf, inf, -inf, 1.0, nan, nan])
        w, w2 = Hd.candw_values(a, b)
        exp_w = [0.5, 0.0, 0.5, 0.0, 0.0, 0.5, nan, nan, nan, nan, nan]
        exp_w2 = [0.0, 0.5, 0.0, 0.5, 0.5, 0.0, nan, nan, nan, nan, nan]
        self.assertTrue(np.all(same16(w, exp_w)) and np.all(same16(w2, exp_w2)))
        # values in [0, 1/2]; codes
        self.assertEqual(list(fp.unorm8_store(np.array([0.25, 0.5, 0.0]))), [64, 128, 0])


class Converter(unittest.TestCase):
    def test_expf_all_f16(self):
        got = fp.expf32(FINITE.astype(f32))
        want = np.array([X.exp_cr(float(h), X.F32) for h in FINITE], f32)
        self.assertTrue(np.array_equal(got.view(np.uint32), want.view(np.uint32)))

    def test_scalar_rules(self):
        s1, s2 = Hd.derive_scalars(100.0, 1.0, f32(20.5))           # e0 = +inf -> S1CAP
        self.assertEqual(s1, f32(20.5))
        s1, s2 = Hd.derive_scalars(0.0, 0.0, f32(20.5))
        self.assertEqual((s1, s2), (f32(1), f32(1)))
        with self.assertRaises(ValueError):
            Hd.derive_scalars(0.0, 100.0, f32(20.5))
        self.assertEqual(s1.dtype, f32)


class Geometry(unittest.TestCase):
    def test_reflect(self):
        for n in range(1, 21):
            for v in range(-60, 61):
                self.assertEqual(int(B.reflect(np.array([v]), n)[0]), S.reflect_slow(v, n), (v, n))
        self.assertEqual(list(B.reflect(np.array([-1, 8, -6, 14]), 8)), [1, 6, 6, 0])

    def test_windows_cover_each_token_once(self):
        for nx, ny in ((32, 32), (64, 32), (32, 96)):
            for sx in (0, 2, 4, 6):
                for sy in (0, 2, 4, 6):
                    Xw, Yw = B.window_coords(nx, ny, sx, sy)
                    inside = (Xw >= 0) & (Xw < nx) & (Yw >= 0) & (Yw < ny)
                    cnt = np.zeros((ny, nx), int)
                    np.add.at(cnt, (Yw[inside], Xw[inside]), 1)
                    self.assertTrue(np.all(cnt == 1))

    def test_recurrent_shift(self):
        self.assertEqual(B.recurrent_shift((2, 6)), (6, 2))
        self.assertEqual(B.recurrent_shift((4, 0)), (0, 4))
        with self.assertRaises(ValueError):
            B.recurrent_shift((3, 0))

    def test_bilinear_against_slow(self):
        rng = np.random.default_rng(5)
        for wt, ht in ((32, 32), (64, 32)):
            hc = fp.r16(rng.normal(0, 50, (ht // 2, wt // 2, 4)) * np.exp2(rng.integers(-12, 8, (ht // 2, wt // 2, 4))))
            hc[0, 0, 0] = np.inf
            hc[1, 1, 1] = np.nan
            hc[2, 3, 2] = 65504.0
            hc[2, 4, 2] = 65504.0
            u = Hd.upsample_coarse(hc, ht, wt)
            for Y in range(ht):
                for Xa in range(wt):
                    xm, ym = (Xa - 1) // 2, (Y - 1) // 2
                    fx = 0.75 if Xa % 2 == 0 else 0.25
                    fy = 0.75 if Y % 2 == 0 else 0.25
                    rd = lambda yy, xx, j: float(hc[S.reflect_slow(yy, ht // 2)][S.reflect_slow(xx, wt // 2)][j])
                    for j in range(3):
                        A, Bv, Cc, Dc = rd(ym, xm + 1, j), rd(ym, xm, j), rd(ym + 1, xm, j), rd(ym + 1, xm + 1, j)
                        t = X.mul(X.mul(A, fx), 1 - fy)
                        t = X.fma(X.mul(Bv, 1 - fx), 1 - fy, t)
                        t = X.fma(X.mul(Cc, 1 - fx), fy, t)
                        t = X.fma(X.mul(Dc, fx), fy, t)
                        self.assertTrue(same16(u[Y, Xa, j], t), (Xa, Y, j))


def tearDownModule():
    for k, v in REPORT.items():
        print("REPORT %s: %s" % (k, v))


if __name__ == "__main__":
    unittest.main()
