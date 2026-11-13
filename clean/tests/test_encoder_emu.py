# SPDX-License-Identifier: MIT
"""Stage enc0 (spec 03): the real kernel source (kernels/encoder.hip) run in the host lane emulator through the
spec 90 runner, compared with the NumPy reference (reference/encoder.py). Synthetic data only.

Run: python -m unittest discover -s clean/tests -t .
"""
import unittest

import numpy as np

from clean.reference import encoder as E
from clean.reference import fp8
from clean.runner import enc0
from clean.tests.test_kernels_emu import EmuTest, build_dir, spec09_metrics, ulp_distance


def f16_ulps(a, b):
    """Distance in FP16 steps between FP16-exact float arrays (same sign assumed where both are non-zero)."""
    ia = np.asarray(a, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float32).astype(np.float16).view(np.int16).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFF), ib)
    return np.abs(ia - ib)


class TestEncoderEmu(EmuTest):
    # (render size, padded size or None for ceil32, inputs overrides, params overrides)
    CASES = [
        ((40, 24), None, {}, {}),
        ((40, 24), None, {"reset": True}, {}),
        ((40, 24), None, {"depth_inverted": True, "frame": 7}, {}),
        ((22, 18), (26, 20), {"frame": 0}, {}),                            # partial wave and workgroup
        ((70, 40), None, {}, {}),                                           # 12 workgroups
        ((40, 24), None, {"exposure": 0.0, "exposure_prev": 0.0}, {}),
        ((40, 24), None, {"exposure_prev": -1.0}, {}),
        ((40, 24), None, {"exposure_prev": float("nan")}, {}),
        ((40, 24), None, {}, {"w_scale": 0.6}),                             # feature clamp at +-6.28125
        ((40, 24), None, {"exposure": float("nan")}, {}),                  # every colour/history value 0
        ((40, 24), None, {"exposure": 1.25, "exposure_prev": 1.25}, {}),  # e == e_prev: G(h) = q, no exp / log
        ((40, 24), None, {"motion_special": True}, {}),                     # NaN / inf motion vectors
        ((37, 21), None, {"frame": 3}, {}),                                 # odd width: reversed right padding
        ((4, 6), (6, 8), {}, {}),                                           # smallest width (depth window edges)
    ]

    def make_case(self, i, R, F, xo, po):
        rng = np.random.default_rng(300 + i)
        kw = {k: v for k, v in xo.items() if k in ("reset", "frame", "depth_inverted")}
        fw, fh = F if F else (None, None)
        x = E.random_inputs(rng, rw=R[0], rh=R[1], fw=fw, fh=fh, **kw)
        x.update({k: v for k, v in xo.items() if k not in kw and k != "motion_special"})
        if xo.get("motion_special"):
            m = x["motion"]
            m[3, 5] = np.nan
            m[7, 2, 1] = np.inf
            m[11, 30, 0] = -np.inf
            m[20, 0] = np.nan
        p = E.random_params(rng, **po)
        return x, p

    def test_matches_reference(self):
        """Features: spec 09 acceptance per case, and the stricter: >= 99 % of values bit-identical per case.
        The remaining differences come from values within float32 rounding error of an FP16 rounding midpoint or a
        dither truncation boundary (kernel FP32, reference FP64). Motion surfaces: at most one FP16 step, >= 99.5 %
        identical (same cause). Layout or formula mistakes change far more values."""
        clamp_hit = False
        n_diff = n_all = 0
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i, R=R, F=F, x=xo, p=po):
                x, p = self.make_case(i, R, F, xo, po)
                ref = E.encode(x, p)
                got = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
                self.assertEqual(set(got), {"features", "m_near", "m_far", "m_center"})
                g, r = got["features"], ref["features"]
                self.assertEqual(g.shape, r.shape)
                self.assertTrue(fp8.is_fp8_exact(g))
                m = spec09_metrics(g, r)
                self.assertEqual(m["nonfinite"], 0)
                self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
                self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)
                self.assertGreaterEqual(float(np.mean(ulp_distance(g, r) == 0)), 0.99, m)
                clamp_hit |= bool(np.any(np.abs(r) == 6.5))
                n_diff += int(np.sum(g != r))
                n_all += g.size
                for k in ("m_near", "m_far", "m_center"):
                    self.assertEqual(got[k].shape, ref[k].shape)
                    fin = np.isfinite(ref[k])   # NaN / inf motion vectors give non-finite surfaces in both
                    np.testing.assert_array_equal(got[k][~fin], ref[k][~fin], err_msg=k)
                    self.assertTrue(np.all(np.isfinite(got[k][fin])), k)
                    u = f16_ulps(got[k][fin], ref[k][fin])
                    self.assertLessEqual(int(u.max()), 1, k)
                    self.assertGreaterEqual(float(np.mean(u == 0)), 0.995, k)
        self.assertTrue(clamp_hit, "no case exercised the feature clamp")
        # Over all cases at most 0.01 % of the features may differ from the FP64 reference at all. Measured on these
        # cases: this kernel 10 of 467,840 (0.002 %), the first version (kernels/baseline/encoder_v1.hip) 48 (0.010 %);
        # a constant wrong in its 6th digit in the history path gave 0.016 %, which the per-case checks above let
        # through.
        self.assertLessEqual(n_diff, 1e-4 * n_all, (n_diff, n_all))

    def test_reset_03_T2(self):
        """On reset the kernel must not read history, latent or motion history: fill them with garbage."""
        rng = np.random.default_rng(11)
        x = E.random_inputs(rng, rw=40, rh=24, reset=True)
        p = E.random_params(rng)
        a = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
        x["reduced_history"] = np.ones_like(x["reduced_history"])
        x["latent"] = -np.ones_like(x["latent"])
        x["motion_history"] = np.full_like(x["motion_history"], 1000.0)
        b = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["features"], E.encode(x, p)["features"])


class TestEncoderLayout(EmuTest):
    def run_once(self, runner=None, seed=21):
        rng = np.random.default_rng(seed)
        x = E.random_inputs(rng, rw=40, rh=24)
        p = E.random_params(rng)
        r = runner or self.runner
        return r.run("enc0", r.pack_weights("enc0", p), x, None)

    def test_k_slot_permutation_does_not_matter(self):
        a = self.run_once()
        self.runner.dev.set_layout(1, 0)
        b = self.run_once()
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_accumulator_layout_is_absorbed_by_packing(self):
        a = self.run_once()
        self.runner.dev.set_layout(1, 1)
        r11 = self.Runner(build_dir(), "", backend="emu", d_row="gfx11")
        b = self.run_once(runner=r11)
        np.testing.assert_array_equal(a["features"], b["features"])
        c = self.run_once()  # packing for the wrong accumulator layout must break the result
        self.assertGreater(float(np.mean(a["features"] != c["features"])), 0.2)


class TestEncoderPacking(unittest.TestCase):
    def test_k_order_covers_each_column_once(self):
        K = enc0.kord_encoder()
        used = K[K != enc0.PAD]
        self.assertEqual(sorted(used.tolist()), list(range(80)))
        for p in range(4):  # sub-pixel p only in its own two K blocks
            blk = K[32 * p:32 * p + 32]
            self.assertTrue(np.all((blk == enc0.PAD) | ((blk >= 20 * p) & (blk < 20 * p + 20))))

    def test_blob(self):
        rng = np.random.default_rng(1)
        p = E.random_params(rng)
        blob = enc0.pack(p)
        self.assertEqual(len(blob), enc0.BLOB_BYTES)
        T = np.frombuffer(blob, np.uint8, 2048, enc0.OFF_DITHER).reshape(8, 16, 16)
        np.testing.assert_array_equal(T, p["T"])
        s = np.frombuffer(blob, np.float32, 2, enc0.OFF_SIGMA)
        self.assertEqual(s[0], p["sigma1"])
        self.assertEqual(s[1], p["sigma2"])
        bad = dict(p, T=p["T"] + 0.5)
        with self.assertRaises(ValueError):
            enc0.pack(bad)


if __name__ == "__main__":
    unittest.main()
