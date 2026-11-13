# SPDX-License-Identifier: MIT
"""Comparison tool of the equivalence tests (clean/equiv/compare.py): synthetic pairs built to land in each verdict
(equal, 1-step rounding noise, a biased rule, a wrong rule in one channel / border band / window class, a large
difference), the first-difference finder, temporal trend, perturbation and sequence verdicts, and value-free printing.
Synthetic data only.

Run: python -m unittest clean.tests.test_equiv_compare
"""
import contextlib
import io
import json
import os
import re
import tempfile
import unittest

import numpy as np

from clean.equiv import compare as K
from clean.equiv import steps as S

H, W, C = 128, 128, 8
N = H * W * C


def base(seed=1, shape=(H, W, C), lo=0.25, hi=4.0):
    """Positive FP16 values spread over four binades."""
    r = np.random.default_rng(seed)
    return r.uniform(lo, hi, size=shape).astype(np.float16)


def from_ord(o):
    o = np.asarray(o, np.int64)
    return np.where(o < 0, 0x8000 | (-o), o).astype(np.uint16).view(np.float16)


def shifted(a, mask, k):
    """a with the values under mask moved by k steps (k may be an array)."""
    o, _ = S.ordinals(a, "fp16")
    return from_ord(np.where(mask, o + k, o))


def pick(rng, shape, count, where=None):
    m = np.zeros(int(np.prod(shape)), bool)
    cand = np.arange(m.size) if where is None else np.flatnonzero(np.asarray(where).reshape(-1))
    m[rng.choice(cand, size=count, replace=False)] = True
    return m.reshape(shape)


def pm1(rng, shape):
    return np.where(rng.random(shape) < 0.5, -1, 1)


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.a = base()
        self.r = np.random.default_rng(7)

    def cmp(self, ours, **kw):
        return K.compare_stage(ours, self.a, "fp16", "HWC", **kw)

    def test_equal(self):
        st = self.cmp(self.a.copy())
        self.assertEqual(st["verdict"], K.MATCHES)
        self.assertIsNone(st["first_difference"])
        self.assertEqual(st["histogram"]["0"], N)

    def test_rounding_ties_match(self):
        m = pick(self.r, self.a.shape, 12)
        k = np.zeros(self.a.shape, np.int64)
        k[m] = np.array([1, -1] * 6)
        st = self.cmp(shifted(self.a, m, k))
        self.assertEqual(st["verdict"], K.MATCHES)
        self.assertEqual((st["positive"], st["negative"]), (6, 6))

    def test_biased_ties_are_close(self):
        st = self.cmp(shifted(self.a, pick(self.r, self.a.shape, 12), 1))
        self.assertEqual(st["verdict"], K.CLOSE)
        self.assertIn("1-step differences biased to one sign", st["reasons"])

    def test_one_step_noise_close(self):
        m = pick(self.r, self.a.shape, N // 200)
        st = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)))
        self.assertEqual(st["verdict"], K.CLOSE, st["reasons"])
        self.assertEqual(st["concentrated"], [])
        self.assertEqual(st["max_step"], 1)

    def test_two_step_noise_close(self):
        m = pick(self.r, self.a.shape, N // 300)
        k = pm1(self.r, self.a.shape) * self.r.integers(1, 3, size=self.a.shape)
        st = self.cmp(shifted(self.a, m, k))
        self.assertEqual(st["verdict"], K.CLOSE, st["reasons"])
        self.assertEqual(st["max_step"], 2)

    def test_three_steps_not_close(self):
        m = pick(self.r, self.a.shape, 20)
        st = self.cmp(shifted(self.a, m, 3 * pm1(self.r, self.a.shape)))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["reasons"], ["differences over 2 steps"])
        self.assertEqual(st["size"], "minor")

    def test_biased_rule_different(self):
        # 0.8 % of the values 2 steps up: equal share, step size and classes pass; the bias (0.016 steps) does not
        st = self.cmp(shifted(self.a, pick(self.r, self.a.shape, N * 8 // 1000), 2))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(len(st["reasons"]), 1)
        self.assertIn("mean signed difference", st["reasons"][0])
        self.assertEqual(st["size"], "minor")
        self.assertAlmostEqual(st["mean_signed"], 2 * (N * 8 // 1000) / N)

    def test_wrong_rule_in_one_channel(self):
        ch3 = np.zeros(self.a.shape, bool)
        ch3[..., 3] = True
        m = pick(self.r, self.a.shape, (N // C) * 4 // 100, where=ch3)    # 4 % of channel 3, 0.5 % overall
        st = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["reasons"], ["differences concentrated in channel: channel 3"])
        self.assertEqual(st["size"], "minor")
        self.assertEqual(st["max_step"], 1)

    def test_wrong_rule_in_border_band(self):
        y = np.arange(H).reshape(H, 1, 1)
        x = np.arange(W).reshape(1, W, 1)
        band = np.broadcast_to((y < 8) | (y >= H - 8) | (x < 8) | (x >= W - 8), self.a.shape)
        m = pick(self.r, self.a.shape, N // 200, where=band)
        st = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["reasons"], ["differences concentrated in border band: border band (8 px)"])

    def test_wrong_rule_in_value_range(self):
        top = self.a.astype(np.float64) >= 2.0                              # binade [2, 4): about 57 % of values
        low = self.a.astype(np.float64) < 0.5                               # binade [0.25, 0.5): about 7 %
        m = pick(self.r, self.a.shape, N // 300, where=low)
        st = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["reasons"], ["differences concentrated in value range: |value| in [2^-2, 2^-1)"])
        self.assertTrue(np.any(top))

    def test_wrong_rule_in_border_windows(self):
        sy = sx = 4
        wy = ((np.arange(H) - sy) % H) // 8
        wx = ((np.arange(W) - sx) % W) // 8
        bw = ((wy == 0) | (wy == 15)).reshape(H, 1, 1) | ((wx == 0) | (wx == 15)).reshape(1, W, 1)
        bw = np.broadcast_to(bw, self.a.shape) & ~self._band()
        m = pick(self.r, self.a.shape, N // 300, where=bw)
        st = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)), window=[8, 8], shift=[sy, sx])
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertIn("differences concentrated in window: border windows", st["reasons"])
        self.assertEqual([c["family"] for c in st["concentrated"]], ["window"])
        same = self.cmp(shifted(self.a, m, pm1(self.r, self.a.shape)))       # without the window family
        self.assertEqual(same["verdict"], K.CLOSE, same["reasons"])

    def _band(self):
        y = np.arange(H).reshape(H, 1, 1)
        x = np.arange(W).reshape(1, W, 1)
        return np.broadcast_to((y < 8) | (y >= H - 8) | (x < 8) | (x >= W - 8), self.a.shape)

    def test_large_difference_massive(self):
        st = self.cmp(base(seed=2))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["size"], "massive")
        self.assertGreater(st["over_16"] * 10, N)
        self.assertGreater(st["max_step"], 1000)

    def test_medium_difference_major(self):
        m = pick(self.r, self.a.shape, N * 2 // 100)
        st = self.cmp(shifted(self.a, m, 5 * pm1(self.r, self.a.shape)))
        self.assertEqual((st["verdict"], st["size"]), (K.DIFFERENT, "major"))
        self.assertEqual(st["histogram"]["5-8"], N * 2 // 100)

    def test_nonfinite(self):
        o = self.a.copy()
        o[5, 6, 1] = np.nan
        st = self.cmp(o)
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertIn("non-finite on one side", st["reasons"])
        self.assertIsNone(st["max_step"])
        self.assertEqual(st["histogram"][">16"], 1)
        both = self.a.copy()
        both[5, 6, 1] = np.nan
        st = K.compare_stage(both.copy(), both, "fp16", "HWC")
        self.assertEqual(st["verdict"], K.MATCHES)
        self.assertEqual(st["nonfinite_either"], 1)
        inf = both.copy()
        inf[5, 6, 1] = np.inf
        self.assertEqual(K.compare_stage(inf, both, "fp16", "HWC")["nonfinite_mismatch"], 1)

    def test_shape_mismatch_is_error(self):
        st = K.compare_stage(self.a[:-1], self.a, "fp16", "HWC")
        self.assertEqual(K.stage_verdict(st)[0], K.ERROR)


class StepTest(unittest.TestCase):
    def test_fp8_codes(self):
        o, nf = S.ordinals(np.array([0x00, 0x80, 0x01, 0x81, 0x7E, 0x7F, 0xFF], np.uint8), "fp8")
        self.assertEqual(list(o[:5]), [0, 0, 1, -1, 126])
        self.assertEqual(list(nf), [0, 0, 0, 0, 0, S.NF_NAN, S.NF_NAN])
        a = np.array([0x01, 0x80, 0x38], np.uint8)
        b = np.array([0x81, 0x00, 0x39], np.uint8)
        st = K.compare_stage(a, b, "fp8", "W")
        self.assertEqual(st["histogram"]["0"], 1)     # -0 == +0
        self.assertEqual(st["histogram"]["2"], 1)     # +-smallest subnormal: 2 steps across zero
        self.assertEqual(st["histogram"]["1"], 1)

    def test_fp8_from_floats(self):
        codes = np.arange(0, 0x7F, dtype=np.uint8)
        vals = S.decoded(codes, "fp8")
        o, _ = S.ordinals(vals, "fp8")
        self.assertEqual(list(o), list(range(0x7F)))
        o, _ = S.ordinals(-vals[1:], "fp8")
        self.assertEqual(list(o), [-i for i in range(1, 0x7F)])
        st = K.compare_stage(np.array([0.3]), np.array([0.25]), "fp8", "W")
        self.assertIn("not representable", st["error"])

    def test_fp16_fp32_neighbours(self):
        for ft, ut in ((np.float16, np.uint16), (np.float32, np.uint32)):
            fmt = "fp16" if ft is np.float16 else "fp32"
            x = np.array([1.0, -1.0, 0.0, -0.0], ft)
            up = np.nextafter(x, ft(np.inf))
            o0, _ = S.ordinals(x, fmt)
            o1, _ = S.ordinals(up, fmt)
            self.assertEqual(list(o1 - o0), [1, 1, 1, 1])
            self.assertEqual(o0[2], o0[3])
            self.assertEqual(list(S.ordinals(x.view(ut), fmt)[0]), list(o0))

    def test_grids(self):
        o, _ = S.ordinals(np.array([0, 255, 1023], np.int64), "int")
        self.assertEqual(list(o), [0, 255, 1023])
        o, _ = S.ordinals(np.array([0.0, 1.0, 128 / 255]), "unorm8")
        self.assertEqual(list(o), [0, 255, 128])
        o, _ = S.ordinals(np.array([-1.0, 0.0, 1.0]), "snorm10")
        self.assertEqual(list(o), [-511, 0, 511])
        with self.assertRaises(S.FormatError):
            S.ordinals(np.array([0.5]), "unorm8")
        key, label = S.range_classes(np.array([0, 1, 3, 512], np.int64), "int")
        self.assertEqual([label(int(k)) for k in key], ["code 0", "|code| in [2^0, 2^1)", "|code| in [2^1, 2^2)",
                                                        "|code| in [2^9, 2^10)"])


class FirstDifferenceTest(unittest.TestCase):
    def test_compute_order_from_any_layout(self):
        a = base(shape=(3, 16, 20, 4))                       # F H W C
        o = a.copy()
        o = shifted(o, np.zeros(o.shape, bool), 0)
        m = np.zeros(o.shape, bool)
        m[2, 0, 0, 0] = True                                 # later frame
        m[1, 9, 3, 2] = True                                 # expected first: frame 1, row 9, column 3, channel 2
        m[1, 9, 5, 0] = True
        m[1, 12, 0, 0] = True
        o = shifted(o, m, 1)
        want = {"frame": 1, "row": 9, "column": 3, "channel": 2}
        self.assertEqual(K.compare_stage(o, a, "fp16", "FHWC")["first_difference"], want)
        t = (0, 3, 1, 2)                                     # F C H W order
        self.assertEqual(K.compare_stage(o.transpose(t), a.transpose(t), "fp16", "FCHW")["first_difference"], want)


class SeriesTest(unittest.TestCase):
    def make(self, frames, grow):
        r = np.random.default_rng(3)
        a = base(shape=(frames, 32, 32, 4))
        k = np.zeros(a.shape, np.int64)
        for f in range(frames):
            count = (2 + f // 4) if grow else 10        # at most 31 of 4096 per frame: every frame CLOSE
            m = pick(r, a.shape[1:], count)
            k[f][m] = pm1(r, (count,))
        return shifted(a, k != 0, k), a

    def test_steady_noise_close(self):
        o, a = self.make(120, False)
        st = K.compare_series(o, a, "fp16", "FHWC", trend=True)
        self.assertEqual(st["verdict"], K.CLOSE)
        self.assertTrue(st["trend"]["evaluated"])
        self.assertFalse(st["trend"]["grows"])
        self.assertEqual(st["first_difference"]["frame"], 1)

    def test_growing_difference(self):
        o, a = self.make(120, True)
        st = K.compare_series(o, a, "fp16", "FHWC", trend=True)
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["reasons"], ["difference grows over time"])
        self.assertTrue(st["trend"]["grows"])
        self.assertEqual(K.worst(f["verdict"] for f in st["per_frame"]), K.CLOSE)

    def test_short_series_trend_not_evaluated(self):
        o, a = self.make(20, True)
        st = K.compare_series(o, a, "fp16", "FHWC", trend=True)
        self.assertFalse(st["trend"]["evaluated"])
        probe = K.compare_series(o, a, "fp16", "FHWC", trend=False)
        self.assertIsNone(probe["trend"])
        self.assertEqual(set(probe["numbers_at_frames"]), {1, 10, probe["worst_frame"]})

    def test_one_bad_frame_decides(self):
        a = base(shape=(12, 16, 16, 2))
        m = np.zeros(a.shape, bool)
        m[7] = True
        st = K.compare_series(shifted(a, m, 40), a, "fp16", "FHWC")
        self.assertEqual((st["verdict"], st["size"], st["worst_frame"]), (K.DIFFERENT, "massive", 8))
        self.assertEqual(st["first_difference"], {"frame": 8, "row": 0, "column": 0, "channel": 0})


class PerturbationTest(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(5)
        self.before = r.uniform(0.1, 1.0, size=(64, 64, 3)).astype(np.float32)
        y, x = np.mgrid[0:64, 0:64]
        bump = np.clip(1.0 - np.hypot(y - 30, x - 20) / 6.0, 0, None)[..., None]
        self.change = (bump * np.array([0.05, 0.03, 0.04])).astype(np.float32)

    def run_case(self, change_ours):
        b = self.before
        return K.compare_perturbation(b, b + change_ours, b, b + self.change, "fp32", "HWC", (30, 20))

    def test_equal_change(self):
        st = self.run_case(self.change)
        self.assertEqual(st["verdict"], K.MATCHES)
        self.assertEqual(st["radius_ours"], st["radius_orig"])

    def test_slightly_scaled_change_close(self):
        st = self.run_case((self.change * np.float32(1.02)).astype(np.float32))
        self.assertEqual(st["verdict"], K.CLOSE, st["reasons"])
        self.assertGreater(st["correlation"], 0.999)

    def test_moved_change_different(self):
        st = self.run_case(np.roll(self.change, 9, axis=1))
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertIn("99 % radius differs", st["reasons"])

    def test_opposite_change_massive(self):
        st = self.run_case(-self.change)
        self.assertEqual((st["verdict"], st["size"]), (K.DIFFERENT, "massive"))


class SequenceTest(unittest.TestCase):
    def test_sequences(self):
        a = [{"call": "init", "result": 0}, {"call": "optimal", "result": 0, "render_w": 1, "render_h": 2}]
        self.assertEqual(K.compare_sequence(a, json.loads(json.dumps(a)))["verdict"], K.MATCHES)
        b = json.loads(json.dumps(a))
        b[1]["render_h"] = 3
        st = K.compare_sequence(a, b)
        self.assertEqual(st["verdict"], K.DIFFERENT)
        self.assertEqual(st["first_difference"], {"call": 1, "field": "render_h"})
        st = K.compare_sequence(a[:1], a)
        self.assertEqual(st["differing_calls"], 1)
        self.assertEqual(st["first_difference"]["call"], 1)
        self.assertIn("missing", st["first_difference"]["field"])


def odd_fp16(shape, seed):
    """FP16 values in [1/4, 1/2) whose last mantissa bit is set: their decimal form needs 12 decimals."""
    r = np.random.default_rng(seed)
    m = 2 * r.integers(0, 512, size=shape) + 1
    return ((1024 + m) / 4096.0).astype(np.float16)


class ChannelVerdictTest(unittest.TestCase):
    """Protocol question 5 (project decision 2026-11-04): one verdict row per channel, so one wrong channel of three
    cannot hide under the 3-times rule (a class with a third of the values can never hold more than 3 times its share)."""

    def one_wrong_of_three(self):
        r = np.random.default_rng(21)
        a = base(seed=4, shape=(64, 64, 3))
        ch1 = np.zeros(a.shape, bool)
        ch1[..., 1] = True
        m = pick(r, a.shape, (64 * 64) * 24 // 1000, where=ch1)          # 2.4 % of channel 1, 0.8 % overall
        return shifted(a, m, pm1(r, a.shape)), a

    def test_stage_alone_misses_it(self):
        o, a = self.one_wrong_of_three()
        st = K.compare_stage(o, a, "fp16", "HWC")
        self.assertEqual(st["verdict"], K.CLOSE, st["reasons"])
        self.assertEqual(st["concentrated"], [])

    def test_channel_rows_catch_it(self):
        o, a = self.one_wrong_of_three()
        fn = lambda x, y, lay: K.compare_stage(x, y, "fp16", lay)
        chans = K.compare_channels(o, a, "HWC", fn)
        self.assertEqual([c["verdict"] for c in chans], [K.MATCHES, K.DIFFERENT, K.MATCHES])
        self.assertEqual(chans[1]["reasons"], ["under 99 % equal"])
        self.assertEqual(chans[1]["size"], "minor")
        self.assertEqual(chans[1]["first_difference"]["channel"], 1)
        t = o.transpose(2, 0, 1)                                         # C H W order gives the same rows
        chans2 = K.compare_channels(t, a.transpose(2, 0, 1), "CHW", fn)
        self.assertEqual([c["verdict"] for c in chans2], [c["verdict"] for c in chans])
        self.assertEqual(K.compare_channels(o[..., :1], a[..., :1], "HWC", fn), [])
        self.assertEqual(K.compare_channels(o[..., 0], a[..., 0], "HW", fn), [])

    def test_manifest_rows_table_and_chaining(self):
        o, a = self.one_wrong_of_three()
        s2 = base(seed=9, shape=(32, 32, 2))
        r = np.random.default_rng(4)
        m2 = pick(r, s2.shape, 10)
        o2 = shifted(s2, m2, pm1(r, s2.shape))                           # 10 of 2048 values 1 step off: CLOSE
        tser = base(seed=10, shape=(5, 8, 8, 3))
        mt = np.zeros(tser.shape, bool)
        mt[2, :, :, 2] = True
        with tempfile.TemporaryDirectory() as td:
            ours, orig = os.path.join(td, "ours"), os.path.join(td, "orig")
            os.mkdir(ours)
            os.mkdir(orig)
            for name, x, y in (("s1.npy", o, a), ("s2.npy", o2, s2), ("t.npy", shifted(tser, mt, 50), tser)):
                np.save(os.path.join(ours, name), x)
                np.save(os.path.join(orig, name), y)
            man = {"model": "synthetic", "run": "isolated",
                   "stages": [{"name": "s1", "file": "s1.npy", "format": "fp16", "layout": "HWC"},
                              {"name": "s2", "file": "s2.npy", "format": "fp16", "layout": "HWC"}],
                   "probes": [{"name": "p", "file": "t.npy", "format": "fp16", "layout": "FHWC"}]}
            rep = K.run_manifest(man, ours, orig)
        rows = {r["name"]: r for r in rep["rows"]}
        self.assertEqual(rows["s1"]["verdict"], K.CLOSE)
        self.assertEqual([rows[f"s1 / ch {c}"]["verdict"] for c in range(3)], [K.MATCHES, K.DIFFERENT, K.MATCHES])
        self.assertEqual(rows["s2"]["verdict"], K.CLOSE)
        self.assertEqual(rows["p"]["verdict"], K.DIFFERENT)
        self.assertEqual([rows[f"p / ch {c}"]["verdict"] for c in range(3)], [K.MATCHES, K.MATCHES, K.DIFFERENT])
        self.assertEqual(rows["p / ch 2"]["numbers"]["worst_frame"], 3)
        self.assertEqual(rep["model_verdict"], K.DIFFERENT)
        text = K.format_report(rep)
        table = text[text.index("VERDICTS"):text.index("NUMBERS")]
        self.assertRegex(table, r"channel\s+s1 / channels 0, 2\s+MATCHES")
        self.assertRegex(table, r"channel\s+s1 / ch 1\s+DIFFERENT \(minor\)")
        self.assertRegex(table, r"channel\s+p / channels 0-1\s+MATCHES")
        self.assertIn("channel s1 / ch 1: DIFFERENT (minor); reasons: under 99 % equal", text)
        self.assertNotIn("channel s1 / ch 0:", text)                    # MATCHES channels: table only

    def test_ranges(self):
        self.assertEqual(K._ranges([0, 1, 2, 4, 6, 7]), "0-2, 4, 6-7")


class BandAndChainedTest(unittest.TestCase):
    """Brief 5 (d): per-stage CLOSE bands; chained intermediates get numbers only; the spec team's non-finite rules."""

    def three_step(self):
        a = base(seed=12, shape=(32, 32, 4))
        m = pick(np.random.default_rng(2), a.shape, 20)
        return shifted(a, m, 3 * pm1(np.random.default_rng(3), a.shape)), a

    def test_band_turns_three_steps_close(self):
        o, a = self.three_step()
        self.assertEqual(K.compare_stage(o, a, "fp16", "HWC")["verdict"], K.DIFFERENT)
        st = K.compare_stage(o, a, "fp16", "HWC", rules=K.with_band(K.Rules, 3))
        self.assertEqual(st["verdict"], K.CLOSE, st["reasons"])
        self.assertEqual((st["band"], st["over_band"], st["over_2"]), (3, 0, 20))
        st = K.compare_stage(o, a, "fp16", "HWC", rules=K.with_band(K.Rules, 1))
        self.assertEqual(st["reasons"], ["differences over 1 steps"])
        self.assertIs(K.with_band(K.Rules, None), K.Rules)
        self.assertIs(K.with_band(K.Rules, 2), K.Rules)

    def test_band_lookup(self):
        man = {"close_band": {"enc0.features": {"fp8": 3}, "b*": 4, "*": {"unorm8": 1}, "post.*": {"fp16": 5}}}
        self.assertEqual(K.band_for(man, "enc0.features", "fp8"), 3)
        self.assertIsNone(K.band_for(man, "enc0.features", "fp16"))     # no fp16 entry for it, "*" has unorm8 only
        self.assertEqual(K.band_for(man, "b3.out", "fp8"), 4)
        self.assertEqual(K.band_for(man, "dec0.A", "unorm8"), 1)
        self.assertEqual(K.band_for(man, "post.history", "fp16"), 5)
        self.assertIsNone(K.band_for({}, "x", "fp8"))

    def test_manifest_bands_and_chained_numbers_only(self):
        o, a = self.three_step()
        with tempfile.TemporaryDirectory() as td:
            ours, orig = os.path.join(td, "ours"), os.path.join(td, "orig")
            os.mkdir(ours)
            os.mkdir(orig)
            for n in ("a", "b", "f"):
                np.save(os.path.join(ours, n + ".npy"), o)
                np.save(os.path.join(orig, n + ".npy"), a)
            stages = [{"name": "enc1.skip", "file": "a.npy", "format": "fp16", "layout": "HWC"},
                      {"name": "b0.out", "file": "b.npy", "format": "fp16", "layout": "HWC", "close_steps": 3},
                      {"name": "filter.out", "file": "f.npy", "format": "fp16", "layout": "HWC", "final": True}]
            iso = K.run_manifest({"run": "isolated", "close_band": {"enc1.*": {"fp16": 3}}, "stages": stages},
                                 ours, orig)
            v = {r["name"]: r["verdict"] for r in iso["rows"] if r["kind"] == "stage"}
            self.assertEqual(v, {"enc1.skip": K.CLOSE, "b0.out": K.CLOSE, "filter.out": K.DIFFERENT})
            ch = K.run_manifest({"run": "chained", "stages": stages}, ours, orig)
            rows = {r["name"]: r for r in ch["rows"]}
            self.assertEqual(rows["enc1.skip"]["verdict"], K.NUMBERS_ONLY)
            self.assertEqual(rows["enc1.skip / ch 0"]["verdict"], K.NUMBERS_ONLY)
            self.assertEqual(rows["b0.out"]["verdict"], K.NUMBERS_ONLY)
            self.assertEqual(rows["enc1.skip"]["numbers"]["over_2"], 20)                 # numbers stay
            self.assertEqual((rows["filter.out"]["verdict"], rows["filter.out"]["note"]), (K.DIFFERENT, "final output"))
            self.assertEqual(ch["model_verdict"], K.DIFFERENT)                          # only the final output counts
            self.assertEqual(ch["first_difference"]["stage"], "enc1.skip")
            text = K.format_report(ch)
            self.assertRegex(text, r"stage\s+enc1\.skip\s+numbers only")
            only = K.run_manifest({"run": "chained", "stages": stages[:2]}, ours, orig)
            self.assertEqual(only["model_verdict"], K.NUMBERS_ONLY)
            forced = K.run_manifest({"run": "chained", "stages": [dict(stages[0], verdict=True)]}, ours, orig)
            self.assertEqual(forced["rows"][0]["verdict"], K.DIFFERENT)

    def test_nonfinite_rules_confirmed(self):
        # NaN vs NaN (any sign, any FP8 NaN code) and the same infinity are equal; shares reported separately
        a = np.array([np.nan, np.inf, -np.inf, 1.0], np.float16)
        b = np.array([-np.nan, np.inf, -np.inf, 1.0], np.float16)
        st = K.compare_stage(a, b, "fp16", "W")
        self.assertEqual((st["verdict"], st["nonfinite_either"], st["nonfinite_mismatch"]), (K.MATCHES, 3, 0))
        st = K.compare_stage(np.array([np.inf], np.float16), np.array([-np.inf], np.float16), "fp16", "W")
        self.assertEqual((st["verdict"], st["nonfinite_mismatch"]), (K.DIFFERENT, 1))
        st = K.compare_stage(np.array([0x7F, 0x00], np.uint8), np.array([0xFF, 0x80], np.uint8), "fp8", "W")
        self.assertEqual((st["verdict"], st["nonfinite_either"]), (K.MATCHES, 1))      # NaN codes; +0 / -0
        st = K.compare_stage(np.array([0x7F], np.uint8), np.array([0x7E], np.uint8), "fp8", "W")
        self.assertEqual((st["max_step"], st["nonfinite_mismatch"]), (None, 1))         # NaN code is not a step
        st = K.compare_stage(np.array([0x01], np.uint8), np.array([0x81], np.uint8), "fp8", "W")
        self.assertEqual(st["max_step"], 2)                                            # across zero, +-0 one value


class ManifestAndPrintingTest(unittest.TestCase):
    def test_end_to_end_value_free(self):
        r = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as td:
            ours, orig = os.path.join(td, "ours"), os.path.join(td, "orig")
            os.mkdir(ours)
            os.mkdir(orig)
            arrays = []

            def put(name, a_ours, a_orig):
                np.save(os.path.join(ours, name), a_ours)
                np.save(os.path.join(orig, name), a_orig)
                arrays.extend([a_ours, a_orig])

            s0 = odd_fp16((24, 24, 4), 1)
            put("s0.npy", s0.copy(), s0)                                          # MATCHES
            s1 = odd_fp16((24, 24, 4), 2)
            m = np.zeros(s1.shape, bool)
            m[3, 4, 1] = True
            put("s1.npy", shifted(s1, m, 2).astype(np.float16), s1)              # 2 steps once: CLOSE? 1 of 2304
            s2 = odd_fp16((24, 24, 4), 3)
            put("s2.npy", odd_fp16((24, 24, 4), 4), s2)                           # DIFFERENT
            s3 = odd_fp16((24, 24, 4), 5)
            m3 = pick(r, s3.shape, 30)
            put("s3.npy", shifted(s3, m3, pm1(r, s3.shape)).astype(np.float16), s3)  # 30 of 2304: DIFFERENT
            t = odd_fp16((6, 8, 8, 2), 6)
            put("t.npy", t.copy(), t)
            pb = odd_fp16((16, 16, 3), 7)
            put("pb.npy", pb, pb)
            pa = odd_fp16((16, 16, 3), 8)
            put("pa.npy", pa, pa)
            seq = [{"call": "x", "result": 0}]
            for d in (ours, orig):
                with open(os.path.join(d, "seq.json"), "w") as f:
                    json.dump(seq, f)
            man = {"model": "synthetic", "run": "isolated",
                   "stages": [{"name": "s0", "file": "s0.npy", "format": "fp16", "layout": "HWC"},
                              {"name": "s1", "file": "s1.npy", "format": "fp16", "layout": "HWC"},
                              {"name": "s2", "file": "s2.npy", "format": "fp16", "layout": "HWC"},
                              {"name": "s3", "file": "s3.npy", "format": "fp16", "layout": "HWC"},
                              {"name": "missing", "file": "nope.npy", "format": "fp16", "layout": "HWC"}],
                   "temporal": [{"name": "t", "case": "steady", "file": "t.npy", "format": "fp16", "layout": "FHWC"}],
                   "probes": [{"name": "p", "file": "t.npy", "format": "fp16", "layout": "FHWC"}],
                   "perturbations": [{"name": "q", "before": "pb.npy", "after": "pa.npy", "format": "fp16",
                                      "layout": "HWC", "center": [8, 8]}],
                   "sequences": [{"name": "seq", "file": "seq.json"}]}
            mp = os.path.join(td, "m.json")
            with open(mp, "w") as f:
                json.dump(man, f)
            rep = K.run_manifest(man, ours, orig)
            got = {r["name"]: (r["verdict"], r["note"]) for r in rep["rows"]}
            self.assertEqual(got["s0"], (K.MATCHES, None))
            self.assertEqual(got["s1"][0], K.CLOSE)
            self.assertEqual(got["s2"][0], K.DIFFERENT)
            self.assertEqual(got["s3"], (K.DIFFERENT, None))                       # 30 of 2304: under 99 % equal
            self.assertEqual(got["missing"][0], K.ERROR)
            self.assertEqual(got["t / steady"][0], K.MATCHES)
            self.assertEqual(got["q"][0], K.MATCHES)
            self.assertEqual(got["seq"][0], K.MATCHES)
            self.assertEqual(rep["model_verdict"], K.ERROR)
            self.assertEqual(rep["first_difference"], {"stage": "s1", "frame": 0, "row": 3, "column": 4, "channel": 1})
            text = K.format_report(rep)
            # the verdict table comes first
            self.assertLess(text.index("VERDICTS"), text.index("NUMBERS"))
            self.assertRegex(text.splitlines()[3], r"^  model\s+ERROR$")
            # no value of either side appears in the text or the JSON report
            vals = set(float(v) for a in arrays for v in np.asarray(a, np.float64).reshape(-1))
            toks = [float(t) for t in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
            self.assertEqual([t for t in toks if t in vals], [])
            jp = os.path.join(td, "r.json")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(K.main(["--manifest", mp, "--ours", ours, "--orig", orig, "--json", jp]), 1)
            self.assertEqual(out.getvalue(), text)
            with open(jp) as f:
                blob = f.read()
            jtoks = [float(t) for t in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", blob)]
            self.assertEqual([t for t in jtoks if t in vals], [])
            for v in list(vals)[:50]:
                self.assertNotIn(repr(v), blob)
                self.assertNotIn(repr(v), text)


if __name__ == "__main__":
    unittest.main()
