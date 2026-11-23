# SPDX-License-Identifier: MIT
"""The patch merge fused into the encoder blocks (swin_block_c64_mg, swin_block_c96_mg) and the expand fused into the decoder
blocks (swin_block_c64_add_ex, swin_block_c96_add_ex) of swin.hip on a gfx11 (RDNA 3) GPU. The runtime runs the `_ex` kernels
for dec1 and dec2 (WG_SWIN_FUSE_STAGES, default "dec1,dec2"); the `_mg` ones are the stage runner's `fuse` geometry.

v_wmma and v_permlanex16 need every lane executing, and the scan of test_exec_scan finds them under a lane mask in these four
kernels and in fused_merge / fused_expand, which they call. The mask comes from the entry test of the loop over the output tiles
(`for (mt = wave; mt < tiles; mt += 8)`): `wave` is a vector value to the compiler, so it tests `tid < 32 * tiles` per lane. The
bound is a multiple of 32, so every lane of a wave32 is on one side of it and the mask is all or nothing at run time: the
pattern is carried, not live (test_exec_scan.FUSED_GUARDED pins it as a known, explained exception). These tests are the card
evidence for that, and the first card runs of these kernels: the fused kernels must give the bytes of the separate ones.

Each case runs the fused kernels and the separate ones (block, then patch_merge / expand) on the card and compares every output
byte. The cases have windows that stick out of the grid (window offset 2: lanes of a wave whose merged token is past the grid
edge) and a window count that is not a multiple of the 8 windows of a workgroup. The unwritten bytes of an output show as
NaN (the runner fills it with 0x7F first).

Skipped unless WG_GPU_TEST=1. Needs a build for the card with swin.co:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_swin_fused

WG_VARIANT=<name> runs a variant of the build (build.py --variants).
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from clean.reference import swin as S
from clean.runner import variants as V
from clean.tests import test_gpu_enc1 as T1
from clean.tests import test_kernels_emu as TK

# (stage, rows, columns, window offset): the grids of the emulator test of the fused kernels, then small and odd ones: a last
# row / column of windows that sticks out of the grid, and a workgroup of 8 windows only partly filled
CASES = [("enc1", 12, 16, (0, 0)), ("enc1", 20, 24, (2, 2)), ("enc1", 16, 12, (0, 2)), ("enc1", 4, 8, (2, 2)),
         ("enc1", 8, 4, (2, 0)), ("enc1", 36, 28, (2, 2)), ("enc1", 4, 4, (0, 0)),
         ("enc2", 12, 16, (2, 2)), ("enc2", 16, 20, (0, 0)), ("enc2", 8, 8, (2, 0)), ("enc2", 4, 12, (2, 2)),
         ("enc2", 28, 20, (2, 2)),
         ("dec1", 12, 16, (0, 0)), ("dec1", 20, 24, (2, 0)), ("dec1", 8, 12, (0, 2)), ("dec1", 4, 8, (2, 2)),
         ("dec1", 8, 4, (0, 2)), ("dec1", 36, 28, (2, 2)),
         ("dec2", 12, 16, (2, 0)), ("dec2", 16, 20, (0, 0)), ("dec2", 8, 8, (2, 2)), ("dec2", 4, 12, (2, 2)),
         ("dec2", 28, 20, (2, 2))]


class SwinFusedGpu(T1.Enc1Gpu, TK.EmuTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        build, rocm = os.environ["WG_BUILD"], os.environ.get("WG_ROCM", "")
        cls.variant_name = os.environ.get("WG_VARIANT", "base")
        # the `fuse` geometry of the variant's code object: a build folder of its own with a manifest that names it
        cls.folder = tempfile.mkdtemp(prefix="wg_fuse_", dir=os.environ.get("TMPDIR"))
        cls.addClassCleanup(shutil.rmtree, cls.folder, True)
        defines = {**V.all_builds().get(cls.variant_name, {}), **V.VARIANTS["fuse"]}
        code = V.code_object(cls.variant_name)
        os.symlink(os.path.join(build, code), os.path.join(cls.folder, code))
        shutil.copy(os.path.join(build, "target.json"), cls.folder)
        with open(os.path.join(cls.folder, V.MANIFEST), "w") as f:
            json.dump({"fuse": {"file": code, "defines": defines, "geometry": V.geometry(defines)}}, f)
        cls.base = cls.Runner(build, rocm, variant=cls.variant_name)
        cls.fused = cls.Runner(cls.folder, rocm, variant="fuse")

    def test_the_fused_geometry_launches_the_fused_kernels(self):
        self.assertTrue(self.fused.geo["fuse"])
        self.assertFalse(self.base.geo["fuse"])

    def test_fused_kernels_give_the_bytes_of_the_separate_ones(self):
        for i, (stage, H, W, offset) in enumerate(CASES):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                rng = np.random.default_rng(700 + i)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                w = self.base.pack_weights(stage, p)
                want = self.base.run(stage, w, x, offset, skip)
                got = self.fused.run(stage, w, x, offset, skip)
                self.assertEqual(set(got), set(want))
                for k in want:
                    self.assertEqual(int(np.sum(~np.isfinite(got[k]))), int(np.sum(~np.isfinite(want[k]))), k)
                    np.testing.assert_array_equal(got[k], want[k], err_msg=k)

    def test_fused_kernels_against_the_reference(self):
        """The same cases against the float64 reference with the spec 09 limits of the stage tests (no non-finite value, at most
        2 % of the values more than one step off, mean absolute error at most 1 % of the reference RMS; < 5 % different where an
        output has 2000 values or more)."""
        for i, (stage, H, W, offset) in enumerate(CASES):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                rng = np.random.default_rng(700 + i)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                got = self.fused.run(stage, self.base.pack_weights(stage, p), x, offset, skip)
                ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
                for k in ref:
                    m = TK.spec09_metrics(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0, k)
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (k, m))
                    if ref[k].size >= 2000:   # a fraction of a few hundred values is one or two values
                        self.assertLess(float(np.mean(TK.ulp_distance(got[k], ref[k]) > 0)), 0.05, k)


if __name__ == "__main__":
    unittest.main()
