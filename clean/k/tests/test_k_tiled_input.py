# SPDX-License-Identifier: MIT
"""Tiled input producer: complete-plan fallbacks and exact legacy/reference equality.

Standalone input plans remain canonical. GPU tests opt into fusion explicitly;
run under the machine shared GPU lock with WG_K_GPU_BUILD.
"""
import os
import unittest

import numpy as np

from clean.k.runner import k_io
from clean.k.reference.instage import input_stage
from clean.k.tests.test_k_io_emu import frame, same_bits


class TestTiledInputPlan(unittest.TestCase):
    def test_selection_and_complete_fallback(self):
        bufs = dict.fromkeys(("Hc", "Hh", "x0", "cpre", "cpre2"), 256)
        fin = k_io.KFrameIn(W=2560, H=1440, rw=1707, rh=960, Bw=640, Bh=384, s=.5, jx=.5, jy=-.5)
        for exact in (False, True):
            self.assertEqual(len(k_io.input_launches(fin, bufs, exact=exact)), 2)
            fast = k_io.input_launches(fin, bufs, exact=exact, tiled=True)
            self.assertEqual(len(fast), 1)
            self.assertEqual(fast[0][0], ("k_io_exact.co" if exact else "k_io.co") + ":k_in_tiled")
            self.assertEqual(fast[0][1:3], (80 * 48, 256))
            self.assertFalse(fast[0][3][0].cpre)
            self.assertFalse(fast[0][3][0].cpre2)
            for name, value in (("s", 1.), ("rw", 2560), ("jx", .75), ("jy", float("nan")),
                                ("W", 16385), ("Bw", 4097)):
                bad = k_io.KFrameIn.from_buffer_copy(bytes(fin))
                setattr(bad, name, value)
                plan = k_io.input_launches(bad, bufs, exact=exact, tiled=True)
                self.assertEqual(len(plan), 2)
                self.assertEqual([p[0].split(":")[-1] for p in plan], ["k_in_color", "k_in_base"])
                self.assertTrue(plan[0][3][0].cpre)
                self.assertTrue(plan[0][3][0].cpre2)


    def test_non_hip_frame_cannot_force_gpu_only_kernel(self):
        from unittest.mock import patch
        from clean.k.runner.k_frame import KFrame

        class HostDevice:
            def malloc(self, size):
                return 256
            def upload(self, *args):
                pass
            def memset(self, *args):
                pass
            def free(self, *args):
                pass

        class NoNetwork:
            def __init__(self, *args, **kwargs):
                self.bufs = {}
            def release(self):
                pass

        with patch("clean.k.runner.k_frame.KNet", NoNetwork):
            for request in (None, False, True):
                frame = KFrame(HostDevice(), lambda name: None, (40, 24), (27, 16), {}, {},
                               base=(1, 1), input_tiled=request)
                try:
                    self.assertFalse(frame.input_tiled)
                finally:
                    frame.release()


class TestTiledInputGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = os.environ.get("WG_K_GPU_BUILD")
        if not build:
            raise unittest.SkipTest("GPU only: set WG_K_GPU_BUILD and use shared GPU lock")
        from clean.runner.stage_runner import Runner
        cls.runner = Runner(build, os.environ.get("ROCM_PATH", ""), backend="hip", model="K")

    def run_input(self, x, exact, tiled, device_scalars=False):
        stage = "input@exact" if exact else "input"
        plan = self.runner._prepare(stage, b"", x, None, None)
        prm = plan["launches"][-1][3][0]
        if device_scalars:
            xs = np.array([prm.f.kappa, prm.f.rho, prm.f.lam, 1. / prm.f.kappa], np.float32)
            ptr = self.runner.dev.malloc(xs.nbytes)
            self.runner.dev.upload(ptr, xs.tobytes())
            plan["bufs"]["tiled_test_xs"], plan["sizes"]["tiled_test_xs"] = ptr, xs.nbytes
            prm.f.xs = ptr
            # Wrong host values ensure the producer really uses GPU-resident exposure.
            prm.f.kappa, prm.f.rho, prm.f.lam = -17., 7., 9.
        bufs = {name: getattr(prm, name) for name in ("Hc", "Hh", "x0", "cpre", "cpre2")}
        plan["launches"] = k_io.input_launches(prm.f, bufs, exact=exact, tiled=tiled)
        if tiled and k_io.input_tiled_supported(prm.f):
            self.assertEqual(len(plan["launches"]), 1)
            if x["reset"]:
                # Reset must not consume any discarded history, motion or depth.
                actual = plan["launches"][0][3][0]
                actual.Hc = actual.Hh = None
                actual.f.motion = actual.f.depth = None
        else:
            self.assertEqual(len(plan["launches"]), 2)
        # The runner rounds tiny allocations up to256bytes. Decode only logical
        # values, and check that inactive lanes left the padded tail untouched.
        logical_bytes = prm.f.Bw * prm.f.Bh * 16 * 2
        decode = plan["decoders"]["x0"]
        def decode_x0(raw, shape):
            tail = raw[logical_bytes:]
            self.assertEqual(tail, bytes([0x7F]) * len(tail))
            value = decode(raw[:logical_bytes], shape)
            self.assertTrue(np.isfinite(value).all(), "nonfinite or unwritten input value")
            return value
        plan["decoders"]["x0"] = decode_x0
        return self.runner._run_plan(plan)["x0"]

    def test_flags_resets_odd_base_subrect_and_reference(self):
        rng = np.random.default_rng(610)
        for exact in (False, True):
            for hdr in (False, True):
                for mvlo in (False, True):
                    for inv in (False, True):
                        for reset in (False, True):
                            with self.subTest(exact=exact, hdr=hdr, mvlo=mvlo, inv=inv, reset=reset):
                                x = frame(rng, hdr, mvlo, inv, reset=reset, base=(33, 17))
                                x["base"] = (33, 17)
                                x["color"] = np.pad(x["color"], ((2, 1), (3, 4), (0, 0)), constant_values=-7)
                                x["color_base"] = (3, 2)
                                want = self.run_input(x, exact, False)
                                got = self.run_input(x, exact, True)
                                self.assertTrue(same_bits(want, got).all())
                                if exact:
                                    self.assertTrue(same_bits(input_stage(x), got).all())

    def test_gpu_scalars_small_edges_and_nonfinite_histories(self):
        rng = np.random.default_rng(927)
        for base in ((1, 1), (7, 9), (8, 8), (9, 7), (33, 17)):
            for exact in (False, True):
                for reset in (False, True):
                    with self.subTest(base=base, exact=exact, reset=reset):
                        x = frame(rng, True, True, True, reset=reset, base=base)
                        x["base"] = base
                        x.update(E_cur=1.5, E_prev=1., E_aux=.75, pre_exposure=2.)
                        x["Hh"].reshape(-1)[::7] = np.nan
                        x["Hh"].reshape(-1)[3::13] = np.inf
                        x["Hc"].reshape(-1)[::17] = -np.inf
                        want = self.run_input(x, exact, False, device_scalars=True)
                        got = self.run_input(x, exact, True, device_scalars=True)
                        self.assertTrue(same_bits(want, got).all())

    def test_unsupported_frames_execute_complete_fallback(self):
        rng = np.random.default_rng(813)
        for exact in (False, True):
            for change in (dict(jitter=(.75, 0.)), dict(s=1.), dict(render_size=(40, 24))):
                x = frame(rng, True, True, True, base=(33, 17))
                x["base"] = (33, 17)
                if "render_size" in change:
                    x["color"] = np.pad(x["color"], ((0, 8), (0, 13), (0, 0)))
                    x["depth"] = np.pad(x["depth"], ((0, 8), (0, 13)))
                    x["motion"] = np.pad(x["motion"], ((0, 8), (0, 13), (0, 0)))
                x.update(change)
                self.assertTrue(same_bits(self.run_input(x, exact, False), self.run_input(x, exact, True)).all())


if __name__ == "__main__":
    unittest.main()
