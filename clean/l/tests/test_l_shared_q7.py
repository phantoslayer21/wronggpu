# SPDX-License-Identifier: MIT
"""Model L: the shared stages in their FP16 form (oracle.run_stage_q7, table shared_form "q7"; Q8 Part B, Q10).

Run: python -m unittest discover -s clean/l/tests -t .
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.reference import swin as MS
from clean.reference.fp8 import f16, q8


def _params(stage, seed):
    return {k: np.asarray(v, np.float64) for k, v in MS.random_params(stage, np.random.default_rng(seed)).items()}


def _k(stage):
    k = T.block_consts(T.TABLE, stage)
    k["mm8"] = "exact"
    return k


class SharedStagesQ7(unittest.TestCase):
    def test_encoder_stage_merge_by_hand(self):
        """enc1: skip = FP8(block); merged = FP8(clamp(chain of 8 ops over the 2 x 2 patch from bm)), checked against
        a per-pixel loop of single ops."""
        p = _params("enc1", 1)
        rng = np.random.default_rng(2)
        x = q8(rng.normal(0, 2, (8, 8, 64)))
        o = L.run_stage_q7("enc1", p, x, (2, 2))
        y = q8(L.swin_block(x, p, (2, 2), 2, _k("enc1"), "mirror"))
        self.assertTrue(np.array_equal(o["skip"], y))
        want = np.empty((4, 4, 96))
        Wm = q8(p["Wm"])
        for r in range(4):
            for c in range(4):
                z = np.concatenate([y[2 * r, 2 * c], y[2 * r, 2 * c + 1], y[2 * r + 1, 2 * c], y[2 * r + 1, 2 * c + 1]])
                acc = f16(p["bm"])                                    # the FP16 bias the device holds
                for c0 in range(0, 256, 32):
                    acc = f16(acc + Wm[:, c0:c0 + 32] @ z[c0:c0 + 32])
                want[r, c] = q8(np.clip(acc, -6.28125, 6.28125))
        self.assertTrue(np.array_equal(o["merged"], want))

    def test_merge_clamp(self):
        p = _params("enc2", 3)
        p["Wm"] = np.zeros_like(p["Wm"])
        p["bm"] = np.where(np.arange(128) % 2 == 0, 100.0, -100.0)
        x = q8(np.random.default_rng(4).normal(0, 1, (8, 8, 96)))
        m = L.run_stage_q7("enc2", p, x, (0, 2))["merged"]
        self.assertTrue(np.all(m[..., 0::2] == 6.5) and np.all(m[..., 1::2] == -6.5))     # FP8(6.28125) = 6.5

    def test_decoder_stage_zero_expansion_is_the_skip_block(self):
        """dec1 with We = 0, be = 0: E = 0, the block input is S everywhere (fine mirror outside the grid), so the stage
        equals the block on S with the mirror border, whatever shared_dec_border says."""
        p = _params("dec1", 5)
        p["We"] = np.zeros_like(p["We"])
        p["be"] = np.zeros_like(p["be"])
        rng = np.random.default_rng(6)
        S = q8(rng.normal(0, 2, (8, 12, 64)))
        xl = q8(rng.normal(0, 2, (4, 6, 96)))
        want = q8(L.swin_block(S, p, (2, 0), 2, _k("dec1"), "mirror"))
        for border in ("coarse", "mirror"):
            t = T.table(shared_dec_border=border)
            self.assertTrue(np.array_equal(L.run_stage_q7("dec1", p, xl, (2, 0), S, t)["out"], want))

    def test_decoder_stage_expansion_layout_and_border(self):
        """dec2 with S = 0: inside the grid the block input is E at the member's own sub-pixel (rows 32 s + c of the
        coarse pixel, s = dx + 2 dy); the two border rules differ only in windows that reach outside the grid."""
        p = _params("dec2", 7)
        rng = np.random.default_rng(8)
        xl = q8(rng.normal(0, 2, (4, 4, 128)))
        S = np.zeros((8, 8, 96))
        e = q8(L.mm8(p["be"], xl, p["We"]))
        E = L.depth_to_space(e)
        self.assertTrue(np.array_equal(E[3, 2], e[1, 1, 96 * 2:96 * 3]))     # dy 1, dx 0: s = dx + 2 dy = 2
        self.assertTrue(np.array_equal(E[2, 3], e[1, 1, 96 * 1:96 * 2]))     # dy 0, dx 1: s = 1
        a = L.run_stage_q7("dec2", p, xl, (0, 0), S, T.table(shared_dec_border="coarse"))["out"]
        b = L.run_stage_q7("dec2", p, xl, (0, 0), S, T.table(shared_dec_border="mirror"))["out"]
        self.assertTrue(np.array_equal(a, b))                                 # offset (0, 0): no member outside
        a = L.run_stage_q7("dec2", p, xl, (2, 0), S, T.table(shared_dec_border="coarse"))["out"]
        b = L.run_stage_q7("dec2", p, xl, (2, 0), S, T.table(shared_dec_border="mirror"))["out"]
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a[:, 2:6], b[:, 2:6]))                 # windows fully inside agree

    def test_bottleneck_stage(self):
        p = _params("b3", 9)
        x = q8(np.random.default_rng(10).normal(0, 2, (4, 8, 128)))
        o = L.run_stage_q7("b3", p, x, (0, 0))["out"]
        self.assertTrue(np.array_equal(o, q8(L.swin_block(x, p, (0, 0), 4, _k("b3"), "mirror"))))

    def test_form_switch(self):
        self.assertEqual(T.TABLE["shared_form"], "q7")                        # the default since the q7 kernels run them
        p = _params("enc1", 11)
        x = q8(np.random.default_rng(12).normal(0, 2, (8, 8, 64)))
        m = L.run_shared_stage("enc1", p, x, (2, 2), t=T.table(shared_form="m"))
        q = L.run_shared_stage("enc1", p, x, (2, 2))
        self.assertTrue(np.array_equal(m["skip"], MS.run_stage("enc1", p, x, (2, 2))["skip"]))
        self.assertFalse(np.array_equal(m["skip"], q["skip"]))


if __name__ == "__main__":
    unittest.main()
