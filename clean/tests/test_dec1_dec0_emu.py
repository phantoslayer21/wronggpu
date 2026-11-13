# SPDX-License-Identifier: MIT
"""Fused dec1 + dec0 (kernels/dec1_dec0.hip) against the separate kernels, in the host emulator: bit-identical A, B, C
and latent. Also checks that the composite base stage equals running stage dec1 and then stage dec0 through the runner.

Run: python -m unittest clean.tests.test_dec1_dec0_emu
"""
import unittest

import numpy as np

from clean.reference import decoder as Dc
from clean.reference import swin as S
from clean.tests.test_kernels_emu import EmuTest

# (render size, padded size or None for ceil32, window offset, reset)
CASES = [((40, 24), (40, 24), (2, 0), False),
         ((37, 21), (40, 24), (0, 2), False),     # odd render size: padding tokens, partial pixel pairs
         ((70, 40), None, (2, 2), False),         # map 32 x 48: several workgroups, border windows
         ((40, 24), None, (0, 0), True)]          # reset frame


def make(i, R, F, offset, reset):
    rng = np.random.default_rng(7300 + i)
    fw, fh = F if F else (-(-R[0] // 32) * 32, -(-R[1] // 32) * 32)
    H, W = fh // 2, fw // 2
    p1 = S.random_params("dec1", rng)
    y, skip = S.random_inputs("dec1", H, W, rng)
    xd = Dc.random_inputs(rng, rw=R[0], rh=R[1], fw=fw, fh=fh, reset=reset)
    p0 = Dc.random_params(rng)
    x = {k: v for k, v in xd.items() if k != "features"}
    x.update(y=y, skip=skip, offset=offset)
    return x, {"dec1": p1, "dec0": p0}


class TestDec1Dec0Emu(EmuTest):
    def test_fused_equals_separate(self):
        for i, (R, F, offset, reset) in enumerate(CASES):
            x, t = make(i, R, F, offset, reset)
            w = bytearray(self.runner.pack_weights("dec1dec0", t))
            for flag in (0, 1):  # FLAGS bit 0 of the dec1 blob selects swin_block_*_nc in both paths
                w[256] = (w[256] & 0xFE) | flag
                with self.subTest(case=i, R=R, offset=offset, reset=reset, nc=flag):
                    a = self.runner.run("dec1dec0", bytes(w), x, None)
                    b = self.runner.run("dec1dec0@fused", bytes(w), x, None)
                    self.assertEqual(set(a), {"A", "B", "C", "latent"})
                    for k in a:
                        np.testing.assert_array_equal(b[k], a[k], err_msg=k)

    def test_base_equals_stages(self):
        x, t = make(0, *CASES[0])
        r = self.runner
        z = r.run("dec1", r.pack_weights("dec1", t["dec1"]), x["y"], x["offset"], x["skip"])["out"]
        xd = dict({k: v for k, v in x.items() if k not in ("y", "skip", "offset")}, features=z)
        want = r.run("dec0", r.pack_weights("dec0", t["dec0"]), xd, None)
        got = r.run("dec1dec0", r.pack_weights("dec1dec0", t), x, None)
        for k in want:
            np.testing.assert_array_equal(got[k], want[k], err_msg=k)


if __name__ == "__main__":
    unittest.main()
