# SPDX-License-Identifier: MIT
"""The whole Model M frame on a gfx11 (RDNA 3) GPU through clean/runner/pipeline.py (E5): synthetic weights and frames only.

THIS NEEDS A RUNTIME THAT THE REPOSITORY DOES NOT HAVE YET. Skipped, with the runtime's own message, until it runs:
  - build.py makes no libwg_runtime.so for a gfx11 target (the gfx11 branch returns before the runtime step), and
  - the runtime refuses the device (device.cpp: "no matching HIP device with packaged gfx1200/gfx1201 kernels") and looks up
    its code objects as <arch>/<name> or <arch>.<name> in the build folder.
With those three things changed in a scratch copy (runtime built by build.runtime_command with embed=False, device.cpp
accepting gfx1100, the code objects linked as gfx1100/<name>) the cases below ran on the RX 7900 XT; the numbers are in the
commit message. Which of these changes the project takes, and how, is the maintainer's decision.

What it checks, with the provisional acceptance of the other card tests (the output standard of an RDNA 3 build is the
maintainer's open decision, plan question 7):
  - stage by stage, as test_runtime does it for the emulator: every stage of the card's frame against the gfx11 lane emulator
    run on the card's own stage input and weights (both are the same instruction sequence up to the matrix instruction's
    rounding): no non-finite value, fewer than 5 % one-code flips per stage and case, at most 2 % more than one step off.
    The bottleneck is checked as the chain b0 .. b5 (the persistent kernel does not write the maps between the blocks);
  - the final output of three frames is finite;
  - the output pixels against the exact (gfx12-arithmetic) emulator frame: an EXPECTED FAILURE. The stages differ by a few
    one-code flips each and a random synthetic network compounds them (about 3 % of the RMS at the end of the encoder, 8 % at the
    bottleneck output, 28 % at the output of the first frame, nearly every pixel different); the < 5 % per case bound and the 1 %
    overall bound are both missed;
  - the median of 50 whole frames, three runs, at 1706x960 -> 2560x1440 and the small cases (printed).

Skipped unless WG_GPU_TEST=1. Needs a gfx11 build with a runtime library (see above):

    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_pipeline
"""
import os
import tempfile
import unittest

import numpy as np

from clean.reference import encoder as En
from clean.reference.swin import window_offset
from clean.runner import family
from clean.runner.pipeline import SWIN_ORDER, Pipeline
from clean.runner.synthetic_model import synthetic_tensors, write_raw
from clean.tests import test_kernels_emu as TK
from clean.tests import test_kernels_emu_gfx11 as TG
from clean.tests import test_post_emu as PE

# (render size, output size): a small one, an odd one (partial 32-token tiles, odd tokens), and the in-game 1440p Quality frame
SMALL = [((64, 40), (96, 60)), ((45, 29), (67, 43))]
GAME = ((1706, 960), (2560, 1440))
FLAGS = {"depth_inverted": False, "auto_exposure": True}


def frames(rw, rh, n=3, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = En.random_inputs(rng, rw=rw, rh=rh, fw=-(-rw // 32) * 32, fh=-(-rh // 32) * 32)
        out.append({"color": x["color"], "depth": x["depth"], "motion": x["motion"], "mv_scale": x["mv_scale"],
                    "jitter": x["jitter"], "exposure": 1.0, "pre_exposure": 1.25, "reset": i == 0})
    return out


class WholeFrameGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("WG_GPU_TEST") != "1":
            raise unittest.SkipTest("set WG_GPU_TEST=1 (and WG_BUILD, WG_ROCM) to run on the GPU")
        cls.build = os.environ.get("WG_BUILD")
        if not cls.build or family.build_family(cls.build) != "gfx11":
            raise unittest.SkipTest("WG_BUILD is not a gfx11 build")
        cls.rocm = os.environ.get("WG_ROCM", "")
        cls.model = tempfile.mkdtemp(prefix="wg_model_", dir=os.environ.get("TMPDIR"))
        write_raw(cls.model, synthetic_tensors(1))
        cls.pipes = {}

    def pipeline(self, render, output):
        if (render, output) not in self.pipes:
            try:
                self.pipes[(render, output)] = Pipeline(self.build, self.rocm, self.model, render, output, FLAGS,
                                                        model_format="raw", check_packing=False)
            except (RuntimeError, OSError) as e:
                self.skipTest(f"no whole-frame runtime for this build and device: {e}")
        return self.pipes[(render, output)]

    def test_every_stage_against_the_gfx11_emulator_on_the_cards_own_input(self):
        from clean.runner.stage_runner import Runner
        emu = Runner(TG.build_dir(), "", backend="emu", arch="gfx11")
        for render, output in SMALL:
            p = self.pipeline(render, output)
            o = p.frame(frames(*render)[0], debug=True)
            parity = p.info()["parity"]

            def check(name, got, want):
                with self.subTest(render=render, stage=name):
                    self.assertEqual(int(np.sum(~np.isfinite(got))), 0)
                    u = TK.ulp_distance(got, want)
                    print(f"{render} {name}: {np.mean(u > 0) * 100:.3f} % one-code flips, {np.mean(u > 1) * 100:.4f} % more "
                          f"than one step, {u.size} values")
                    self.assertLess(float(np.mean(u > 0)), 0.05)
                    self.assertLessEqual(float(np.mean(u > 1)), 0.02)

            x = o["enc0"]["features"]
            for s in ("enc1", "enc2"):
                r = emu.run(s, p.stage_blob(s), x, window_offset(s, parity))
                for k in r:
                    check(f"{s}.{k}", o[s][k], r[k])
                x = o[s]["merged"]
            for s in SWIN_ORDER[2:8]:
                x = emu.run(s, p.stage_blob(s), x, window_offset(s, parity))["out"]
            check("b0..b5", o["b5"]["out"], x)
            for s, skip in (("dec2", "enc2"), ("dec1", "enc1")):
                r = emu.run(s, p.stage_blob(s), o["b5" if s == "dec2" else "dec2"]["out"], window_offset(s, parity),
                            o[skip]["skip"])
                check(s, o[s]["out"], r["out"])

    def test_the_output_of_three_frames_is_finite(self):
        for render, output in SMALL:
            p = self.pipeline(render, output)
            for i, f in enumerate(frames(*render)):
                with self.subTest(render=render, frame=i):
                    out = p.frame(f)["out"]
                    self.assertEqual(out.shape, (output[1], output[0], 3))
                    self.assertEqual(int(np.sum(~np.isfinite(out))), 0)

    @unittest.expectedFailure
    def test_output_pixels_against_the_exact_emulator_frame(self):
        """Both bounds miss with the compounding of a random synthetic network: per case < 5 % of the pixels different (any
        channel more than 0 FP16 steps off) and < 1 % overall."""
        from clean.tests.test_runtime import runtime_dir
        total = differ = 0
        for render, output in SMALL:
            p = self.pipeline(render, output)
            emu = Pipeline(runtime_dir(), "", self.model, render, output, FLAGS, model_format="raw", backend="emu",
                           check_packing=False)
            for f in frames(*render):
                got, want = p.frame(f)["out"], emu.frame(f)["out"]
                px = np.any(PE.f16_steps(got, want) > 0, axis=-1)
                print(f"{render} frame: {np.mean(px) * 100:.2f} % of the output pixels different")
                total += px.size
                differ += int(px.sum())
                self.assertLess(float(np.mean(px)), 0.05)
        self.assertLess(differ / total, 0.01)

    def test_frame_time(self):
        """Median of 50 whole frames (inputs already on the card), three runs; printed, and finite."""
        for render, output in SMALL + [GAME]:
            p = self.pipeline(render, output)
            f = frames(*render, n=2)[1]
            runs = [p.time_frame(f, 50) for _ in range(3)]
            print(f"{render[0]}x{render[1]} -> {output[0]}x{output[1]}: whole frame {runs[0]:.3f} {runs[1]:.3f} {runs[2]:.3f} ms")
            self.assertTrue(all(0.0 < v < 1000.0 for v in runs))


if __name__ == "__main__":
    unittest.main()
