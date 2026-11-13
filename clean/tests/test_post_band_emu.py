# SPDX-License-Identifier: MIT
"""Banded two-kernel path (post_recon2_r_band + post_filter_c1_s<S>_band over n horizontal bands, runner/post.py:
band_plan, prepare_banded) in the host lane emulator: bit-identical to post_recon2_r + post_filter_c1_s<S> for several
band counts on the cases of test_post_emu; the band plan covers every tile once and launches each filter row only
after its input rows are reconstructed.

Run: python -m unittest discover -s clean/tests -t .
"""
import numpy as np

from clean.runner import post as PS
from clean.tests.test_kernels_emu import EmuTest
from clean.tests import test_post_emu as PE


class TestBandEmu(EmuTest):
    def test_plan(self):
        for (dw, dh, ow, oh) in ((3840, 2160, 2560, 1440), (90, 54, 60, 36), (59, 33, 39, 22), (1621, 913, 1081, 609)):
            for n in (1, 2, 3, 4, 8, 100):
                for s in (4, 2):
                    plan = PS.band_plan(dw, dh, ow, oh, n, s)
                    tx, fx = PS._cdiv(dw, 32), PS._cdiv(ow, 40)
                    rrows, frows = PS._cdiv(PS._cdiv(dh, 2), 8), PS._cdiv(PS._cdiv(oh, 2), 2 * s)
                    seen = {"recon": 0, "filter": 0}
                    d_done = 0
                    for kind, t0, nt in plan:
                        with self.subTest(D=(dw, dh), n=n, s=s):
                            self.assertEqual(t0, seen[kind])   # contiguous, in order, no gaps or repeats
                            self.assertGreater(nt, 0)
                            seen[kind] += nt
                            if kind == "recon":
                                d_done = min(16 * ((t0 + nt) // tx), dh)
                            else:
                                last_pair = min(((t0 + nt) // fx) * 2 * s, PS._cdiv(oh, 2)) - 1
                                self.assertLess(min(3 * last_pair + 4, dh - 1), d_done)
                    self.assertEqual(seen, {"recon": tx * rrows, "filter": fx * frows})

    def test_identical_to_two_kernel_path(self):
        maker = PE.TestPostEmu("test_matches_reference")
        maker.runner = self.runner
        checked = 0
        for i, (R, D, xo, special) in enumerate(PE.TestPostEmu.CASES):
            x, s = maker.make_case(i, R, D, xo, special)
            rh, rw = x["color"].shape[:2]
            dw, dh = x["recon_size"]
            try:
                O = PS.output_size_for(dw, dh)
            except ValueError:
                continue
            if not PS.recon2_supported(rw, rh, dw, dh):
                continue
            w = self.runner.pack_weights("post", {"s": np.array(s)})
            a = PS.run_plan(self.runner, PS.prepare_post(self.runner.dev, w, x, "post_recon2_r"))
            y = {"display": a["display"], "exposure": x["exposure"], "output_size": O}
            base = PS.run_plan(self.runner, PS.prepare_filter(self.runner.dev, b"", y, "post_filter_c1_s4"))["out"]
            for rk, fk in [(PS.BAND_RECON, f) for f in PS.BAND_FILTERS] + [
                    (k, "post_filter_c1_s4_band") for k in list(PS.BAND_RECONS)[1:]]:
                for n in (1, 2, 3):
                    with self.subTest(case=i, special=special, recon=rk, filter=fk, n=n):
                        g = PS.run_plan(self.runner, PS.prepare_banded(self.runner.dev, w, x, n, fk, rk))
                        np.testing.assert_array_equal(g["history"], a["history"])
                        np.testing.assert_array_equal(g["reduced"], a["reduced"])
                        np.testing.assert_array_equal(g["out"], base)
                        checked += 1
        self.assertGreaterEqual(checked, 15 * 8)

    def test_pack12_layout(self):
        """pack12_inputs = dec0's pack12 store (decoder.hip, 469572c): A8 = (A | B << 24, (B >> 8) | (C1 << 16)),
        C4 = C0, with A, B the code words and C0, C1 the two words of C."""
        rng = np.random.default_rng(6)
        n = 37
        A = rng.integers(0, 1 << 24, n, dtype=np.uint32)
        B = rng.integers(0, 1 << 24, n, dtype=np.uint32)
        C = rng.integers(0, 1 << 32, (n, 2), dtype=np.uint64).astype(np.uint32)
        C[:, 1] &= 0xFFFF   # the fourth half of C is the zero padding
        got = PS.pack12_inputs({"A": A.astype("<u4").tobytes(), "B": B.astype("<u4").tobytes(),
                                "C": C.astype("<u4").tobytes()})
        a8 = np.frombuffer(got["A"], "<u4").reshape(n, 2)
        np.testing.assert_array_equal(a8[:, 0], A | (B << 24))
        np.testing.assert_array_equal(a8[:, 1], (B >> 8) | (C[:, 1] << 16))
        np.testing.assert_array_equal(np.frombuffer(got["C"], "<u4"), C[:, 0])

    def test_planar_codec(self):
        h = np.random.default_rng(4).uniform(0, 3, (7, 9, 3)).astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(PS.decode_history_planar(PS.encode_history_planar(h), h.shape), h)
