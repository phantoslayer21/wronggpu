# SPDX-License-Identifier: MIT
"""Spec43 soft limiting: thresholds/nonfinite taps, compared with the clean reference."""
import unittest
import numpy as np
from clean.k.reference.outstage import output_stage
from clean.k.tests.test_k_io_emu import frame, same_bits
from clean.k.tests.test_k_kernels_emu import runner


class TestOutputSoftLimit(unittest.TestCase):
    def test_thresholds_and_nonfinite_history(self):
        r = runner()
        rng = np.random.default_rng(72)
        values = np.array([-np.inf, -1., -0., 0., .98974609375, .990234375, 1., np.inf, np.nan], np.float16)
        for hdr in (False, True):
            for mvlo in (False, True):
                for index in range(len(values) + 1):
                    with self.subTest(hdr=hdr, mvlo=mvlo, history=index), np.errstate(all="ignore"):
                        x = frame(rng, hdr, mvlo, True, base=(16, 16))
                        x["base"] = (16, 16)  # small unit-test override, no production geometry change
                        if index < len(values):
                            x["L"].fill(values[index])
                        else:
                            x["L"][:] = rng.choice(values, x["L"].shape)
                        expected, got = output_stage(x), r.run("output", b"", x)
                        for name in ("out", "L", "Hc", "Hh"):
                            self.assertTrue(same_bits(got[name], expected[name]).all(), name)

    def test_bounded_colour_and_history_fallbacks(self):
        r = runner()
        rng = np.random.default_rng(271)
        values = np.array([-np.inf, -65504., -1., -0., 0.,
                           np.nextafter(np.float16(0), np.float16(1)),
                           .9990234375, 1., 65504., np.inf, np.nan], np.float16)
        for hdr in (False, True):
            for index in range(len(values) + 4):
                with self.subTest(hdr=hdr, case=index), np.errstate(all="ignore"):
                    x = frame(rng, hdr, True, True, base=(16, 16))
                    x["base"] = (16, 16)
                    if index < len(values):
                        x["Hc"].fill(values[index])
                        x["L"].fill(values[index])
                    else:
                        # Arbitrary half histories challenge the finite-colour
                        # bound; mixed negative/large luma forces batch fallback.
                        for key in ("Hc", "L"):
                            x[key] = rng.integers(0, 65536, x[key].shape,
                                                  dtype=np.uint16).view(np.float16)
                        x["E_cur"] = (-1.5, 0., np.inf, np.nan)[index-len(values)]
                    expected, got = output_stage(x), r.run("output", b"", x)
                    for name in ("out", "L", "Hc", "Hh"):
                        self.assertTrue(same_bits(got[name], expected[name]).all(), name)


if __name__ == "__main__":
    unittest.main()
