# SPDX-License-Identifier: MIT
"""Complete-frame tiled color ownership, fallback planning, and GPU equality."""
import ctypes
import os
import unittest

import numpy as np

from clean.k.runner import k_io


class TestTiledColorPlan(unittest.TestCase):
    def test_complete_plan_and_fallbacks(self):
        bufs = {name: 256 * (i + 1) for i, name in enumerate(
            ("Hc", "Hh", "x0", "cpre", "cpre2", "ctap", "ctapT"))}
        fin = k_io.KFrameIn(W=2560, H=1440, rw=1707, rh=960, Bw=640, Bh=384,
                           s=.5, jx=.5, jy=-.5)
        plan, fused = k_io.complete_input_launches(fin, bufs, tiled=True)
        self.assertTrue(fused)
        self.assertEqual(plan[0][:3], ("k_io.co:k_in_tiled_color", 80 * 48, 256))
        prm = plan[0][3][0]
        self.assertEqual(prm.ctap, bufs["ctap"])
        self.assertEqual(prm.ctapT, bufs["ctapT"])
        self.assertEqual(prm.input.x0, bufs["x0"])
        self.assertFalse(prm.input.cpre)
        self.assertFalse(prm.input.cpre2)
        self.assertEqual(ctypes.sizeof(prm), 216)
        # Independent stage calls still prepare only their original outputs.
        self.assertEqual(k_io.input_launches(fin, bufs, tiled=True)[0][0], "k_io.co:k_in_tiled")
        for exact, tiled in ((True, True), (False, False)):
            self.assertFalse(k_io.complete_input_launches(fin, bufs, exact=exact, tiled=tiled)[1])
        for field, value in (("flags", k_io.RESET), ("Bw", 639), ("Bh", 359), ("rw", 2560),
                             ("s", 1.), ("jx", .501), ("jy", float("nan")), ("W", 16385)):
            with self.subTest(field=field):
                bad = k_io.KFrameIn.from_buffer_copy(bytes(fin))
                setattr(bad, field, value)
                plan, fused = k_io.complete_input_launches(bad, bufs, tiled=True)
                self.assertFalse(fused)
                self.assertEqual([x[0] for x in plan], [x[0] for x in k_io.input_launches(bad, bufs, tiled=True)])


class TestTiledColorGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("WG_K_GPU_BUILD"):
            raise unittest.SkipTest("GPU only: WG_K_GPU_BUILD and shared GPU lock required")

    def test_complete_frames_and_taps(self):
        from clean.runner.hip import Hip
        from clean.k.runner.k_bench import KFuncs
        from clean.k.runner.k_frame import KFrame
        from clean.k.reference.network import random_network_params
        from clean.k.tests.test_k_runtime_gpu import inputs
        dev = Hip("")
        fn = KFuncs(dev, os.environ["WG_K_GPU_BUILD"], "hip")
        params = random_network_params(17)
        rng = np.random.default_rng(43)
        for hdr in (False, True):
            for inv in (False, True):
                flags = dict(hdr=hdr, mv_lowres=True, depth_inverted=inv)
                a = KFrame(dev, fn, (65, 49), (43, 33), params, flags, base=(32, 16), input_tiled=False)
                b = KFrame(dev, fn, (65, 49), (43, 33), params, flags, base=(32, 16), input_tiled=True)
                try:
                    for i in range(3):
                        x = inputs(rng, 43, 33, hdr, reset=i == 0, jitter=((i-1)*.5, (1-i)*.5))
                        x['pre_exposure'] = 1 + i*.25
                        oa, ob = a.frame(x, debug=True), b.frame(x, debug=True)
                        for key in ('out', 'x0', 'N'):
                            np.testing.assert_array_equal(oa[key].view(np.uint32), ob[key].view(np.uint32))
                        ha, hb = a.histories(), b.histories()
                        for key in ha:
                            np.testing.assert_array_equal(ha[key].view(np.uint32), hb[key].view(np.uint32))
                        for key, size in k_io.output_scratch_bytes(43, 33).items():
                            self.assertEqual(dev.download(a.bufs[key], size), dev.download(b.bufs[key], size))
                finally:
                    a.release()
                    b.release()
