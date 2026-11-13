# SPDX-License-Identifier: MIT
"""Stages post and filter (spec 06): the real kernel source (kernels/post.hip) run in the host lane emulator through
the spec 90 runner, compared with the NumPy reference (reference/post.py). Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import unittest

import numpy as np

from clean.reference import post as P
from clean.runner import post
from clean.tests.test_kernels_emu import EmuTest
from clean.tests.test_post_reference import bright


def f16_steps(a, b):
    """Distance in FP16 steps between FP16-exact float arrays (ordered integer view of the bit patterns)."""
    ia = np.asarray(a, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFF), ib)
    return np.abs(ia - ib)


def spec09_fp16(got, ref):
    finite = np.isfinite(got)
    d = np.abs(np.where(finite, got, 0.0).astype(np.float64) - ref)
    return {"nonfinite": int(np.sum(~finite)), "frac_over_1": float(np.mean(f16_steps(got, ref) > 1)),
            "frac_diff": float(np.mean(f16_steps(got, ref) > 0)),
            "mean_rel_rms": float(d.mean() / max(np.sqrt(np.mean(np.square(ref, dtype=np.float64))), 1e-30))}


def with_nans(rng, x):
    x = dict(x)
    col = x["color"].copy()
    col[rng.random(col.shape) < 0.05] = np.nan
    col[rng.random(col.shape) < 0.05] = np.inf
    col[rng.random(col.shape) < 0.02] = -np.inf
    x["color"] = col
    C = x["C"].copy()
    C[rng.random(C.shape) < 0.02] = np.nan
    x["C"] = C
    h = x["history"].copy()
    h[rng.random(h.shape) < 0.02] = np.nan
    h[rng.random(h.shape) < 0.02] = np.inf
    x["history"] = h
    for k in ("m_near", "m_center", "m_far"):
        m = x[k].copy()
        m[rng.random(m.shape) < 0.03] = np.nan
        m[rng.random(m.shape) < 0.01] = np.inf
        x[k] = m
    return x


class TestPostEmu(EmuTest):
    # (R, D or None for the Quality-like ratio, input overrides, special)
    CASES = [
        ((40, 24), None, {}, None),
        ((40, 24), None, {"reset": True}, None),
        ((26, 15), (59, 33), {}, None),                       # odd D, partial tiles
        ((36, 20), (36, 20), {}, None),                       # D = R: footprint too large for LDS, global path
        ((20, 12), (90, 54), {}, None),                       # 3x ratio
        ((30, 18), None, {}, "far"),                          # many reprojections outside D
        ((30, 18), None, {"exposure": 0.0, "exposure_prev": 0.0}, None),
        ((30, 18), None, {"exposure": 4.0, "exposure_prev": 0.25}, None),
        ((30, 18), None, {}, "nan"),                          # NaN / inf inputs (06-T3)
        ((30, 18), None, {}, "edge"),                         # reprojected positions exactly on 0 and D
        ((30, 18), None, {}, "flat_scores"),                  # equal scores: tie rules
        ((30, 18), None, {}, "bright"),                       # highlights: display limit 2.45 (spec 06 draft 2)
        ((3, 2), (5, 1), {}, None),                           # tiny: one D row, empty reduced history
        ((2, 3), (1, 4), {}, None),                           # one D column
    ]

    def make_case(self, i, R, D, xo, special):
        rng = np.random.default_rng(600 + i)
        D = D or (None, None)
        x = P.random_inputs(rng, R[0], R[1], dw=D[0], dh=D[1], reset=xo.get("reset", False),
                            motion_scale=40.0 if special == "far" else 3.0)
        x.update({k: v for k, v in xo.items() if k != "reset"})
        dw, dh = x["recon_size"]
        if special == "nan":
            x = with_nans(rng, x)
        elif special == "edge":
            rh, rw = x["color"].shape[:2]
            px, py = P.target_positions(rw, rh, dw, dh, *x["jitter"])
            fy, fx = P.point_index(px, py, rw, rh)
            m = np.zeros((rh, rw, 2), np.float32)
            # every D pixel whose point read lands on render pixel (fy, fx) gets that pixel's motion; pick motions
            # that put the leftmost / topmost such D pixel exactly on 0 and the rightmost / lowest exactly on D
            for ry in range(rh):
                for rx in range(rw):
                    cols = np.nonzero(fx[0] == rx)[0]
                    rows = np.nonzero(fy[:, 0] == ry)[0]
                    if cols.size and rows.size:
                        m[ry, rx, 0] = -(cols[0] + 0.5) if rx % 2 == 0 else dw - (cols[-1] + 0.5)
                        m[ry, rx, 1] = -(rows[0] + 0.5) if ry % 2 == 0 else dh - (rows[-1] + 0.5)
            x["m_near"] = x["m_far"] = P.f16(m)
            self.assertTrue(np.array_equal(P.f16(m), m))
        elif special == "bright":
            x = bright(x)
        elif special == "flat_scores":
            B = x["B"].copy()
            B[..., 1] = B[..., 0]
            B[: B.shape[0] // 2, :, 2] = B[: B.shape[0] // 2, :, 0]
            x["B"] = B
        return x, float(rng.uniform(0.2, 2.0))

    def test_matches_reference(self):
        """Spec 09 acceptance for FP16 and 8/10-bit outputs, per case. Differences come from FP32 (kernel) vs FP64
        (reference) rounding of values near an FP16 or 10-bit rounding midpoint."""
        for i, (R, D, xo, special) in enumerate(self.CASES):
            with self.subTest(case=i, R=R, D=D, x=xo, special=special):
                x, s = self.make_case(i, R, D, xo, special)
                ref = P.reconstruct(x, s)
                got = self.runner.run("post", self.runner.pack_weights("post", {"s": np.array(s)}), x)
                self.assertEqual(set(got), {"history", "reduced", "display"})
                for k in ("history", "display"):
                    self.assertEqual(got[k].shape, ref[k].shape)
                    m = spec09_fp16(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0, k)
                    self.assertLessEqual(m["frac_over_1"], 1e-4, (k, m))
                    self.assertLessEqual(m["mean_rel_rms"], 1e-3, (k, m))
                    self.assertLessEqual(m["frac_diff"], 0.01, (k, m))
                if special == "bright":   # spec 06 draft 2: the display image stops at 2.45 (FP16: 2.4492)
                    self.assertEqual(float(got["display"].max()), float(np.float16(2.45)))
                    self.assertGreater(float(got["history"].max()), 2.45)
                self.assertEqual(got["reduced"].shape, ref["reduced"].shape)
                dc = np.abs(np.rint(got["reduced"] * 1023.0) - np.rint(ref["reduced"] * 1023.0))
                self.assertLessEqual(int(dc.max(initial=0)), 1)
                self.assertLessEqual(float(np.mean(dc > 0)) if dc.size else 0.0, 0.01)

    def test_reset_ignores_history(self):
        x, s = self.make_case(1, (40, 24), None, {"reset": True}, None)
        w = self.runner.pack_weights("post", {"s": np.array(s)})
        a = self.runner.run("post", w, x)
        b = self.runner.run("post", w, dict(x, history=np.full_like(x["history"], np.nan)))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


class TestFilterEmu(EmuTest):
    # odd and even O, O not a multiple of 4 (partial column groups), several workgroups and strips, odd D
    SIZES = [(8, 6), (7, 5), (1, 1), (2, 3), (130, 37), (260, 70), (515, 33)]

    def check(self, disp, e, O, stage="filter"):
        ref = P.output_filter(disp, e, O)
        got = self.runner.run(stage, b"", {"display": disp, "exposure": e, "output_size": O})["out"]
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(np.all(np.isfinite(got)))
        # FP32 rounding of the filtered E value (~3e-7) is amplified by Dec: d(ln Dec)/dq = ln(10) / 0.36 = 6.4
        # (and an absolute FP32 error ~1e-7 of near-zero filtered values: atol)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=2e-9)
        return got

    def test_random(self):
        rng = np.random.default_rng(77)
        for O in self.SIZES:
            disp = P.f16(rng.uniform(-0.2, 3.2, (P.recon_size_for(O[1]), P.recon_size_for(O[0]), 3)))
            e = float(rng.uniform(0.3, 3.0))
            outs = {}
            for k in post.FILTER_VARIANTS:
                with self.subTest(O=O, kernel=k):
                    outs[k] = self.check(disp, e, O, stage="filter" if k == post.KERNEL_FILTER else f"filter:{k}")
            with self.subTest(O=O, variants="identical"):
                for k, v in outs.items():
                    np.testing.assert_array_equal(v, outs[post.KERNEL_FILTER], err_msg=k)

    def test_impulse_and_edges(self):
        O = (40, 34)
        D = (P.recon_size_for(O[0]), P.recon_size_for(O[1]))
        imp = np.zeros((D[1], D[0], 3), np.float32)
        imp[17, 23] = [2.0, 1.0, 0.5]
        imp[0, 0] = 1.5
        imp[-1, -1] = 2.5
        self.check(imp, 1.0, O)
        edge = np.zeros_like(imp)
        edge[:, D[0] // 2:] = 2.0
        edge[D[1] // 3:, :, 1] = 1.25
        self.check(edge, 0.0, O)   # exposure 0 -> 1


if __name__ == "__main__":
    unittest.main()
