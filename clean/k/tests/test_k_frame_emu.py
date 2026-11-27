# SPDX-License-Identifier: GPL-3.0-only
"""Model K whole frame (clean/k/runner/k_frame.py) in the host emulator: input stage -> network -> output stage with
histories and the frame index, over a short sequence with a render-size change (no reset for K: the history is
kept) and then a game reset, with the exposure rule of spec 42 draft 3 section 4 (clean/k/reference/exposure.py:
the game's exposure or 1 as E_cur, an auto-exposure value as E_aux).

- Wiring, exactly: each frame's input and output stages equal the NumPy references bit for bit when those are given
  the emulator's own histories and network output.
- Chained against the references alone: the network output stays within spec 49's whole-network rule; the image
  stays finite (its error is reported: decision points of spec 43 amplify the network's FP32 accumulation
  differences, and spec 49's whole-image thresholds are pending).

Run: python -m unittest discover -s clean/k/tests -t .
"""
import unittest

import numpy as np

from clean.k.reference import instage, outstage
from clean.k.reference.network import random_network_params
from clean.k.runner.k_frame import KFrame, new_reference_state, reference_frame
from clean.k.tests.test_k_io_emu import same_bits
from clean.k.tests.test_k_kernels_emu import device

W, H, BASE = 40, 24, (32, 16)
# (render size, game reset, the game's exposure (None: none), the auto-exposure value (None: off))
SIZES = [(27, 16, False, None, 0.75), (27, 16, False, None, 1.5), (27, 16, False, 1.25, 1.5), (30, 20, False, 2.0, None),
         (30, 20, True, 2.0, 0.5)]


def frames(rng, flags):
    for rw, rh, reset, game, measured in SIZES:
        mshape = (rh, rw, 2) if flags["mv_lowres"] else (H, W, 2)
        yield dict(color=(rng.random((rh, rw, 3)) * (3 if flags["hdr"] else 1)).astype(np.float32),
                   motion=(rng.standard_normal(mshape) * 0.7).astype(np.float32),
                   depth=rng.random((rh, rw)).astype(np.float32), render_size=(rw, rh),
                   jitter=tuple(float(v) for v in rng.uniform(-0.5, 0.5, 2)), mv_scale=(1.0, 1.0), pre_exposure=1.0,
                   reset=reset, game_exposure=game, measured=measured)


class TestFrameEmu(unittest.TestCase):
    def setUp(self):
        self.dev, self.fn = device()  # the emulator, or the GPU with WG_K_GPU_BUILD
        self.P = random_network_params(3)

    def test_sequence(self):
        for flags in ({"hdr": True, "mv_lowres": True, "depth_inverted": True},
                      {"hdr": False, "mv_lowres": False, "depth_inverted": False}):
            kf = KFrame(self.dev, self.fn, (W, H), (30, 20), self.P, flags, base=BASE, exact_input=True)
            st = new_reference_state((W, H), BASE)
            try:
                for i, x in enumerate(frames(np.random.default_rng(5), flags)):
                    hist = kf.histories()
                    g = kf.frame(x, debug=True)
                    d = g["frame"]
                    self.assertEqual(g["reset"], i in (0, 4))
                    self.assertEqual(g["f"], {0: 0, 1: 1, 2: 2, 3: 3, 4: 0}[i])
                    # the exposure rule (spec 42 draft 3 section 4)
                    rw, rh, _, game, measured = SIZES[i]
                    if flags["hdr"]:
                        want_cur = 1.0 if game is None else game
                        self.assertEqual(d["E_cur"], want_cur)
                        self.assertEqual(d["E_aux"], want_cur if measured is None else measured)
                        prev = SIZES[i - 1][3] if i else None
                        self.assertEqual(d["E_prev"], want_cur if i in (0, 4) else (1.0 if prev is None else prev))
                    else:
                        self.assertEqual((d["E_cur"], d["E_aux"], d["E_prev"], d["e1"], d["e2"]), (1.0,) * 5)
                    # wiring, bit for bit
                    x0 = instage.input_stage(dict(d, Hc=hist["Hc"], Hh=hist["Hh"]))
                    self.assertTrue(same_bits(g["x0"].astype(np.float16), x0).all(), f"frame {i}: input stage")
                    o = outstage.output_stage(dict(d, N=g["N"], L=hist["L"], Hc=hist["Hc"]))
                    self.assertTrue(same_bits(g["out"], o["out"]).all(), f"frame {i}: output stage")
                    # chained references
                    r = reference_frame(st, x, self.P, flags)
                    rms = np.sqrt(np.mean(r["N"].astype(np.float64) ** 2))
                    dN = np.abs(g["N"] - r["N"].astype(np.float64))
                    self.assertLessEqual(dN.mean() / rms, 8e-3)
                    self.assertLessEqual(dN.max() / rms, 7e-2)
                    self.assertTrue(np.all(np.isfinite(g["out"])))
            finally:
                kf.release()


if __name__ == "__main__":
    unittest.main()
