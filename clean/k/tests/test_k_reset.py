# SPDX-License-Identifier: MIT
"""Reset input specialization: exact old-kernel equivalence and no dependency on discarded inputs.

Run with WG_K_GPU_BUILD under the shared GPU lock, or on the host lane emulator.
Synthetic frames only; the general kernel remains the differential baseline.
"""
import unittest

import numpy as np

from clean.k.runner import k_io
from clean.k.tests.test_k_io_emu import frame
from clean.k.tests.test_k_kernels_emu import runner


class TestResetPlan(unittest.TestCase):
    def test_selects_only_on_reset(self):
        bufs = dict.fromkeys(("Hc", "Hh", "x0", "cpre", "cpre2"), 0)
        for exact in (False, True):
            for reset in (False, True):
                fin = k_io.KFrameIn(rw=27, rh=16, Bw=33, Bh=17, flags=k_io.RESET if reset else 0)
                plan = k_io.input_launches(fin, bufs, exact=exact)
                prefix = k_io.EXACT_CODE_OBJECT + ":" if exact else ""
                self.assertEqual(plan[1][0], prefix + ("k_in_reset" if reset else "k_in_base"))
                self.assertEqual(plan[1][1], (4 * 33 * 17 + 127) // 128)


class TestResetInput(unittest.TestCase):
    def test_general_kernel_equivalence(self):
        r = runner()
        rng = np.random.default_rng(91)
        for exact in (False, True):
            for hdr in (False, True):
                for mvlo in (False, True):
                    for inv in (False, True):
                        with self.subTest(exact=exact, hdr=hdr, mvlo=mvlo, inv=inv):
                            x = frame(rng, hdr, mvlo, inv, reset=True, base=(33, 17))
                            x["base"] = (33, 17)  # partial final workgroup and padded output positions
                            x["jitter"] = (-0.5, 0.5)
                            # Nonzero subrect origin and row padding; prepass writes dense scratch.
                            x["color"] = np.pad(x["color"], ((2, 1), (3, 4), (0, 0)), constant_values=-7)
                            x["color_base"] = (3, 2)
                            stage = "input@exact" if exact else "input"
                            old = r._prepare(stage, b"", x, None, None)
                            name, grid, threads, args = old["launches"][-1]
                            old["launches"][-1] = (name.replace("k_in_reset", "k_in_base"), grid, threads, args)
                            want = r._run_plan(old)["x0"]
                            new = r._prepare(stage, b"", x, None, None)
                            name, grid, threads, args = new["launches"][-1]
                            prm = k_io.KInParams.from_buffer_copy(bytes(args[0]))
                            # These buffers must never be read by the reset kernel. Its prepass retains
                            # its original parameters, so still has valid cpre storage.
                            prm.Hc = prm.Hh = prm.cpre = None
                            prm.f.motion = prm.f.depth = prm.f.xs = None
                            new["launches"][-1] = (name, grid, threads, [prm])
                            got = r._run_plan(new)["x0"]
                            np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))
                            np.testing.assert_array_equal(got[..., 0::4], got[..., 1::4])
                            self.assertTrue(np.all(got[..., 2::4] == 0))
                            self.assertTrue(np.all(got[..., 3::4] == 0))


if __name__ == "__main__":
    unittest.main()
