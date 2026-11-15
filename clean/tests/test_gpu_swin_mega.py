# SPDX-License-Identifier: MIT
"""The persistent Swin kernels of Model M on a gfx11 (RDNA 3) GPU, the FP16 route: swin_bneck6 (b0 to b5 in one launch) and
swin_mid (enc2 with its merge, b0 to b5, dec2 with its expand in one launch), the build swin__bmegahg.co (kernels/swin_v2.hip with
swin.hip's arithmetic; the runtime's default for b0 to b5 on gfx12). They are the six or ten launches of the plain kernels in
one ordered queue with per-window-row completion counters in global memory; the sync-word protocol is the one of gfx12.

What is checked, on the card:
  * the persistent launch gives exactly the bytes of the plain launches of the default build, for both window parities, several
    grid sizes (fewer workgroups than items and more), the clamping kernels and the clamp-free ones, maps with partial windows
    (a window whose members are partly outside the map) and one or two window rows; the sync words are zero after every launch
    (the runner checks that);
  * the plain launches are correct against the float64 reference block by block, each block fed the card's own output of the
    block before it (a chain fed back to the reference compounds the one-code flips of the F16 matrix instruction: printed, not
    bounded, see test_chain_against_the_reference_chain), with the acceptance of test_gpu_enc1: spec 09 per case and output,
    fewer than 5 % of the values different per case;
  * times at the in-game sizes: median of 50 launches, three runs, against the plain kernels (printed).

Skipped unless WG_GPU_TEST=1. Needs a build for the card with swin.co and swin__bmegahg.co (the default build of build.py):

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime [--variants f16wm]
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_swin_mega
"""
import os
import unittest

import numpy as np

from clean.reference import swin as S
from clean.runner import family
from clean.runner import variants as V
from clean.runner.packing import MID_STAGES, split_mid
from clean.tests import test_kernels_emu as TK
from clean.tests.test_gpu_stages import GpuCase

# (rows, columns) of the bottleneck blocks (an eighth of the render size) and of enc2 / dec2 (a quarter): 1706x960 and 2560x1440
BNECK_SIZES = {"1706x960": (120, 216), "2560x1440": (180, 320)}
MID_SIZES = {"1706x960": (240, 432), "2560x1440": (360, 640)}
RUNS = 3


def flags_cleared(w):
    """The packed weights with flags bit 0 cleared: the runner then launches the clamping kernels."""
    return bytes([w[0] & 0xFE]) + bytes(w[1:])


def same(a, b):
    """Bit-identical decoded maps (NaN equals NaN)."""
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


class MegaGpu(GpuCase):
    """Runners of the default build (the plain launches), of the persistent build and, when it is built, of f16wm (the plain
    launches of the fastest block kernels)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        build, rocm = os.environ["WG_BUILD"], os.environ.get("WG_ROCM", "")
        if family.build_family(build) != "gfx11":
            raise unittest.SkipTest("the build is not for a gfx11 target")
        if not os.path.exists(os.path.join(build, "swin.co")) or not os.path.exists(os.path.join(build, "swin__bmegahg.co")):
            raise unittest.SkipTest("the build has no swin.co or swin__bmegahg.co")
        from clean.runner.stage_runner import Runner
        cls.plain = Runner(build, rocm)
        cls.mega = Runner(build, rocm, variant="bmegahg")
        cls.fast = Runner(build, rocm, variant="f16wm") if os.path.exists(os.path.join(build, "swin__f16wm.co")) else None

    def check_block(self, label, got, ref):
        """spec 09 and the per-case 5 % bound of one block output against the reference output on the same input."""
        self.assertEqual(got.shape, ref.shape)
        m = TK.spec09_metrics(got, ref)
        differ = float(np.mean(TK.ulp_distance(got, ref) > 0))
        self.assertEqual(m["nonfinite"], 0, label)
        self.assertLessEqual(m["frac_over_1_step"], 0.02, (label, m))
        self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (label, m))
        self.assertLess(differ, 0.05, label)
        return m, differ

    def runs(self, fn):
        """RUNS measurements (each a median of 50 launches): the list and its median."""
        t = [fn() for _ in range(RUNS)]
        return t, float(np.median(t))


class TestBottleneckMegakernelGpu(MegaGpu, unittest.TestCase):
    """swin_bneck6 against six launches of swin_block_c128 on the card."""

    # (H, W): partial windows at the borders with the window offsets (2, 2) / (0, 2), 8x12 and 36x32 several items per block,
    # 4 x 20 and 8 x 4 one or two window rows (every item depends on every row of the block before it)
    MAPS = ((12, 16), (8, 12), (4, 20), (8, 4), (36, 32))

    def bneck_case(self, H, W, seed):
        rng = np.random.default_rng(seed)
        params = [S.random_params(f"b{k}", rng) for k in range(6)]
        x, _ = S.random_inputs("b0", H, W, rng)
        return params, [self.plain.pack_weights(f"b{k}", params[k]) for k in range(6)], x

    def test_the_megakernel_is_launched(self):
        params, ws, x = self.bneck_case(8, 12, 1)
        plan = self.mega._prepare("bneck", self.plain.pack_weights("bneck", dict(zip("b0 b1 b2 b3 b4 b5".split(), params))),
                                  x, S.window_offset("b0", 0), None)
        try:
            self.assertEqual([launch[0] for launch in plan["launches"]], ["swin_bneck6_nc"])
        finally:
            self.mega._release(plan)

    def test_bit_identical_to_six_plain_launches(self):
        """Both parities, 1 / 3 workgroups and the build's grid (more workgroups than items on the small maps)."""
        for i, (H, W) in enumerate(self.MAPS):
            _, ws, x = self.bneck_case(H, W, 600 + i)
            for parity in (0, 1):
                want = self.plain.run_bneck(ws, x, parity, mega=False)
                self.assertEqual(int(np.sum(~np.isfinite(want))), 0)
                for groups in (1, 3, None):
                    with self.subTest(H=H, W=W, parity=parity, groups=groups):
                        self.assertTrue(same(self.mega.run_bneck(ws, x, parity, mega=True, groups=groups), want))

    def test_clamping_kernels_bit_identical_to_six_plain_launches(self):
        """The weights without the flags bit: swin_bneck6 (clamping) against six swin_block_c128 launches."""
        _, ws, x = self.bneck_case(12, 16, 650)
        wc = [flags_cleared(w) for w in ws]
        plan = self.mega._prepare_bneck(wc, x, 0, True, None)
        try:
            self.assertEqual([launch[0] for launch in plan["launches"]], ["swin_bneck6"])
        finally:
            self.mega._release(plan)
        for parity in (0, 1):
            with self.subTest(parity=parity):
                self.assertTrue(same(self.mega.run_bneck(wc, x, parity, mega=True, groups=3),
                                     self.plain.run_bneck(wc, x, parity, mega=False)))

    def test_blocks_against_the_reference_block_by_block(self):
        """Each block of the chain on the card, fed the card's own output of the block before it, against the float64 reference
        on that same input (so the one-code flips of the F16 matrix instruction do not compound); the megakernel's output is the
        last block's, bit for bit (test above)."""
        for i, (H, W) in enumerate(((12, 16), (36, 32))):
            params, ws, x = self.bneck_case(H, W, 600 + i if H == 12 else 604)
            for parity in (0, 1):
                y = x
                for k in range(6):
                    off = S.window_offset(f"b{k}", parity)
                    got = self.plain.run(f"b{k}", ws[k], y, off)["out"]
                    ref = S.run_stage(f"b{k}", params[k], y, off, gelu="sigmoid")["out"]
                    with self.subTest(H=H, W=W, parity=parity, block=k):
                        m, differ = self.check_block(f"b{k}", got, ref)
                        print(f"b{k} {H}x{W} parity {parity}: {m['frac_over_1_step'] * 100:.3f} % more than one step off, mean abs "
                              f"error {m['mean_abs_err'] / m['ref_rms'] * 100:.4f} % of RMS, {differ * 100:.3f} % different")
                    y = got
                with self.subTest(H=H, W=W, parity=parity, what="megakernel = the chain"):
                    self.assertTrue(same(self.mega.run_bneck(ws, x, parity, mega=True), y))

    def test_chain_against_the_reference_chain(self):
        """Printed, not bounded: the six blocks on the card against the six blocks of the float64 reference, each fed the
        reference's output. Per block the card is within the limits (test above); chained, the one-code flips of the matrix
        instruction feed the next block, and the blocks amplify them: the same numbers for the plain launches and the
        megakernel, which are the same bytes."""
        params, ws, x = self.bneck_case(12, 16, 600)
        ref = x
        for k in range(6):
            ref = S.run_stage(f"b{k}", params[k], ref, S.window_offset(f"b{k}", 0), gelu="sigmoid")["out"]
        got = self.mega.run_bneck(ws, x, 0, mega=True)
        m = TK.spec09_metrics(got, ref)
        print(f"b0..b5 chain 12x16: {m['frac_over_1_step'] * 100:.2f} % more than one step off, mean abs error "
              f"{m['mean_abs_err'] / m['ref_rms'] * 100:.2f} % of RMS, "
              f"{float(np.mean(TK.ulp_distance(got, ref) > 0)) * 100:.1f} % different, {m['nonfinite']} non-finite")
        self.assertEqual(m["nonfinite"], 0)

    def test_time(self):
        """Median of 50 launches, three runs, at the in-game sizes: six plain launches (default build, and f16wm when built)
        against the megakernel; inputs already on the card."""
        for name, (H, W) in BNECK_SIZES.items():
            _, ws, x = self.bneck_case(H, W, 5)
            row = {}
            for label, fn in (("plain", lambda: self.plain.time_bneck(ws, x, 0, mega=False, repeats=50)),
                              ("mega", lambda: self.mega.time_bneck(ws, x, 0, mega=True, repeats=50)),
                              *([("plain f16wm", lambda: self.fast.time_bneck(ws, x, 0, mega=False, repeats=50))]
                                if self.fast else [])):
                row[label] = self.runs(fn)
            print(f"bneck {name} render ({H}x{W} tokens): " + "; ".join(
                f"{label} {' '.join(f'{v:.3f}' for v in t)} ({m:.3f}) ms" for label, (t, m) in row.items()))
            self.assertTrue(all(0.0 < m < 1000.0 for _, m in row.values()))


class TestResidentBlockGpu(MegaGpu, unittest.TestCase):
    """The C = 64 block kernels of the persistent build (swin_block_c64[_add], clamping and clamp-free; weights resident in LDS,
    one wave per window) launched alone, with window counts that leave the last workgroup with one to seven idle waves and the
    last wave of a 2-window launch with one window.

    v_wmma and v_permlanex16 need every lane executing. The loop over a wave's windows ends with `if (win0 + i >= nwin) break;`,
    and with win0 a vector value the compiler could not see that the test is the same for the whole wave: it put the whole loop
    body under a lane mask. The mask is all or nothing at run time (win0 comes from the wave's index), so the old code gave the
    right bytes on the card too: this test cannot go red from the old code, test_exec_scan (the check of the instruction stream) does.
    What it holds: the kernels with the scalar window index give the bytes of the plain build's kernels and stay within the
    reference limits of test_gpu_enc1 on every window count."""

    # (stage, H, W, offset): the window count nwin = ceil((W + ox) / 4) * ceil((H + oy) / 4) takes 1, 2, 3, 5, 7, 8, 9, 15, 17
    CASES = (("enc1", 4, 4, (0, 0)), ("enc1", 4, 8, (0, 0)), ("enc1", 4, 12, (0, 0)), ("enc1", 8, 12, (2, 2)),
             ("enc1", 12, 24, (0, 2)), ("enc1", 8, 28, (0, 0)), ("enc1", 36, 32, (2, 2)),
             ("dec1", 4, 4, (0, 0)), ("dec1", 4, 8, (0, 0)), ("dec1", 12, 20, (2, 0)), ("dec1", 28, 36, (0, 0)),
             ("dec1", 36, 32, (2, 0)))

    def run_both(self, stage, H, W, offset, seed, clamp=False):
        rng = np.random.default_rng(seed)
        p = S.random_params(stage, rng)
        x, skip = S.random_inputs(stage, H, W, rng)
        w = self.plain.pack_weights(stage, p)
        if clamp:
            w = flags_cleared(w)
        want = self.plain.run(stage, w, x, offset, skip)
        got = self.mega.run(stage, w, x, offset, skip)
        return p, x, skip, want, got

    def test_the_block_kernels_of_the_persistent_build_are_launched(self):
        for stage, kernel in (("enc1", "swin_block_c64_nc"), ("dec1", "swin_block_c64_add_nc")):
            rng = np.random.default_rng(1)
            x, skip = S.random_inputs(stage, 12, 12, rng)
            w = self.mega.pack_weights(stage, S.random_params(stage, rng))
            plan = self.mega._prepare(stage, w, x, S.window_offset(stage, 0), skip)
            try:
                self.assertIn(kernel, [launch[0] for launch in plan["launches"]])
            finally:
                self.mega._release(plan)

    def test_bit_identical_to_the_plain_build_and_within_the_reference_limits(self):
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            for clamp in (False, True):
                p, x, skip, want, got = self.run_both(stage, H, W, offset, 900 + i, clamp)
                with self.subTest(stage=stage, H=H, W=W, offset=offset, clamp=clamp):
                    for k in want:
                        self.assertTrue(same(got[k], want[k]), k)
                    if not clamp and H * W >= 96:  # a per-case fraction of a few hundred values is too coarse for 5 %
                        ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
                        for k in ref:
                            self.check_block(f"{stage} {H}x{W} {k}", got[k], ref[k])


class TestMidGpu(MegaGpu, unittest.TestCase):
    """swin_mid against the ten launches (enc2, patch merge, b0 to b5, expand, dec2) on the card."""

    MAPS = ((16, 24), (24, 40), (8, 8), (8, 24), (40, 56))

    def mid_case(self, H, W, seed):
        rng = np.random.default_rng(seed)
        params = {st: S.random_params(st, rng) for st in MID_STAGES}
        x, _ = S.random_inputs("enc2", H, W, rng)
        return params, self.plain.pack_weights("mid", params), x

    def test_the_persistent_kernel_is_launched(self):
        _, w, x = self.mid_case(16, 24, 1)
        plan = self.mega._prepare("mid", w, x, S.window_offset("enc2", 0), None)
        try:
            self.assertEqual([launch[0] for launch in plan["launches"]], ["swin_mid_nc"])
        finally:
            self.mega._release(plan)

    def test_bit_identical_to_ten_plain_launches(self):
        """Both parities, 1 / 3 workgroups and the build's grid; clamp-free and clamping kernels."""
        for i, (H, W) in enumerate(self.MAPS):
            _, w, x = self.mid_case(H, W, 700 + i)
            blobs = (("nc", w), ("clamp", b"".join(flags_cleared(b) for b in split_mid(w)))) if i < 2 else (("nc", w),)
            for kind, wk in blobs:
                for parity in (0, 1):
                    want = self.plain.run_mid(wk, x, parity, mega=False)
                    self.assertEqual(int(np.sum(~np.isfinite(want))), 0)
                    for groups in (1, 3, None):
                        with self.subTest(H=H, W=W, kernels=kind, parity=parity, groups=groups):
                            self.assertTrue(same(self.mega.run_mid(wk, x, parity, groups=groups), want))

    def test_stages_against_the_reference_stage_by_stage(self):
        """enc2 (skip and merged), b0 to b5 and dec2 on the card, each fed the card's own output of the stage before, against
        the float64 reference on that same input; swin_mid's output is dec2's, bit for bit."""
        for H, W in ((16, 24), (24, 40)):
            params, w, x = self.mid_case(H, W, 720)
            for parity in (0, 1):
                off = S.window_offset("enc2", parity)
                blob = {st: b for st, b in zip(MID_STAGES, split_mid(w))}
                o = self.plain.run("enc2", blob["enc2"], x, off)
                ref = S.run_stage("enc2", params["enc2"], x, off, gelu="sigmoid")
                for key in ref:
                    with self.subTest(H=H, W=W, parity=parity, stage="enc2", output=key):
                        self.check_block(f"enc2 {key}", o[key], ref[key])
                y = o["merged"]
                for k in range(6):
                    st, boff = f"b{k}", S.window_offset(f"b{k}", parity)
                    got = self.plain.run(st, blob[st], y, boff)["out"]
                    with self.subTest(H=H, W=W, parity=parity, stage=st):
                        self.check_block(st, got, S.run_stage(st, params[st], y, boff, gelu="sigmoid")["out"])
                    y = got
                doff = S.window_offset("dec2", parity)
                d = self.plain.run("dec2", blob["dec2"], y, doff, o["skip"])["out"]
                with self.subTest(H=H, W=W, parity=parity, stage="dec2"):
                    self.check_block("dec2", d, S.run_stage("dec2", params["dec2"], y, doff, o["skip"], gelu="sigmoid")["out"])
                with self.subTest(H=H, W=W, parity=parity, what="swin_mid = the stage chain"):
                    self.assertTrue(same(self.mega.run_mid(w, x, parity), d))

    def test_time(self):
        """Median of 50 launches, three runs, at the in-game sizes: ten plain launches (default build, and f16wm when built)
        against swin_mid; inputs already on the card."""
        for name, (H, W) in MID_SIZES.items():
            _, w, x = self.mid_case(H, W, 5)
            row = {}
            for label, fn in (("plain", lambda: self.plain.time_mid(w, x, 0, mega=False, repeats=50)),
                              ("mid", lambda: self.mega.time_mid(w, x, 0, mega=True, repeats=50)),
                              *([("plain f16wm", lambda: self.fast.time_mid(w, x, 0, mega=False, repeats=50))]
                                if self.fast else [])):
                row[label] = self.runs(fn)
            print(f"mid {name} render ({H}x{W} tokens): " + "; ".join(
                f"{label} {' '.join(f'{v:.3f}' for v in t)} ({m:.3f}) ms" for label, (t, m) in row.items()))
            self.assertTrue(all(0.0 < m < 1000.0 for _, m in row.values()))


if __name__ == "__main__":
    unittest.main()
