# SPDX-License-Identifier: MIT
"""Model M stages enc0 (kernels/encoder.hip), dec0 (kernels/decoder.hip) and the fused dec1 + dec0 (kernels/dec1_dec0.hip) on a
gfx11 (RDNA 3) GPU: the FP16 route, with the matrix wrappers of kernels/wg_platform.h. The cases are those of the emulator
tests (test_encoder_emu, test_decoder_emu, test_dec1_dec0_emu), against the same float64 / NumPy references.

Acceptance (provisional, the maintainer's open decision; the same as test_gpu_enc1 and test_gpu_swin_stages): spec 09 per case
and output (no non-finite value, at most 2 % of the values more than one step off, mean absolute error at most 1 % of the
reference RMS) and fewer than 5 % of the values different per case; the overall fraction different (the 1 % bound of the
emulator tests) is printed, and the emulator tests' own stricter limits run as expected failures: RDNA 3's F16 matrix
instruction loses about a unit in the last place with mixed signs, so a few values land on the other side of a rounding step.
"Step" is an FP8 code for the features, an 8-bit code of its own scale for dec0's A, B and latent, an FP16 step for C.

The cases include token counts that are not a multiple of 16 (22x18 padded to 26x20, 37x21, 4x6): a 16-token tile of a wave is
only partly valid. v_wmma and v_permlanex16 need every lane executing; an encoder store behind `if (tok < ntok)` let the
compiler move the whole projection into that branch (test_exec_scan finds it in the instruction stream; on the card the valid
tokens of a partly valid tile got NaN or wrong values).

Skipped unless WG_GPU_TEST=1. Needs a build for the card with encoder.co, decoder.co and dec1_dec0.co (the default build):

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_GPU_TEST=1 WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_gpu_e4
"""
import os
import unittest

import numpy as np

from clean.reference import decoder as Dc
from clean.reference import encoder as E
from clean.reference import fp8
from clean.reference import swin as S
from clean.runner import family
from clean.tests import test_dec1_dec0_emu as TD1
from clean.tests import test_decoder_emu as TD
from clean.tests import test_encoder_emu as TE
from clean.tests import test_kernels_emu as TK
from clean.tests import test_kernels_emu_gfx11 as TG
from clean.tests.test_gpu_stages import GpuCase

# the in-game sizes (runner/sweep.py GAME_SIZES: 1440p Quality, render 1706x960; the same stages for a 2560x1440 render)
TIMED_SIZES = {"1706x960": (1706, 960), "2560x1440": (2560, 1440)}
RUNS = 3


class E4Gpu(GpuCase):
    CODE_OBJECTS = ("encoder.co", "decoder.co", "dec1_dec0.co")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        build = os.environ["WG_BUILD"]
        if family.build_family(build) != "gfx11":
            raise unittest.SkipTest("the build is not for a gfx11 target")
        for co in cls.CODE_OBJECTS:
            if not os.path.exists(os.path.join(build, co)):
                raise unittest.SkipTest(f"the build has no {co}")
        from clean.runner.stage_runner import Runner
        cls.Runner = Runner


def code_steps(a, b, scale):
    return np.abs(np.rint(np.asarray(a, np.float64) * scale) - np.rint(np.asarray(b, np.float64) * scale))


def judge(case, label, got, ref, steps):
    """The provisional acceptance of one case and output. steps: the distance of every value to the reference in steps.
    Returns (values, values different)."""
    case.assertEqual(got.shape, ref.shape, label)
    case.assertEqual(int(np.sum(~np.isfinite(got))), 0, label)
    over, differ = float(np.mean(steps > 1)), float(np.mean(steps > 0))
    rms = float(np.sqrt(np.mean(np.asarray(ref, np.float64) ** 2)))
    mean_abs = float(np.mean(np.abs(np.asarray(got, np.float64) - ref)))
    print(f"{label}: {over * 100:.3f} % more than one step off, mean abs error {mean_abs / rms * 100:.4f} % of RMS, "
          f"{differ * 100:.3f} % different")
    case.assertLessEqual(over, 0.02, label)
    case.assertLessEqual(mean_abs, 0.01 * rms, label)
    case.assertLess(differ, 0.05, label)
    return steps.size, int(np.sum(steps > 0))


def dec0_steps(got, ref):
    """{output: (got, ref, steps)} of dec0's outputs: A, B (8-bit codes, 255), latent (127), C in FP16 steps."""
    out = {}
    for k, scale in (("A", 255), ("B", 255), ("latent", 127)):
        out[k] = (got[k], ref[k], code_steps(got[k], ref[k], scale))
    out["C"] = (got["C"], ref["C"], TE.f16_ulps(got["C"], ref["C"]))
    return out


class TestEncoderGpu(E4Gpu, TE.TestEncoderEmu):
    """enc0 on the card, the cases of test_encoder_emu."""

    def test_matches_reference(self):
        total = differ = 0
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i, R=R, F=F, x=xo, p=po):
                x, p = self.make_case(i, R, F, xo, po)
                ref = E.encode(x, p)
                got = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
                self.assertTrue(fp8.is_fp8_exact(got["features"]))
                n, d = judge(self, f"enc0 case {i} {R} features", got["features"], ref["features"],
                             TK.ulp_distance(got["features"], ref["features"]))
                total += n
                differ += d
                for k in ("m_near", "m_far", "m_center"):  # before the matrix: the same as the emulator tests ask
                    self.assertEqual(got[k].shape, ref[k].shape)
                    fin = np.isfinite(ref[k])
                    np.testing.assert_array_equal(got[k][~fin], ref[k][~fin], err_msg=k)
                    self.assertTrue(np.all(np.isfinite(got[k][fin])), k)
                    u = TE.f16_ulps(got[k][fin], ref[k][fin])
                    self.assertLessEqual(int(u.max()), 1, k)
                    self.assertGreaterEqual(float(np.mean(u == 0)), 0.995, k)
        print(f"enc0 all cases: {differ / total * 100:.4f} % of the features differ from the reference ({total} values)")

    @unittest.expectedFailure
    def test_stricter_limits_of_the_emulator_test(self):
        """At most 0.01 % of the features different over all cases, and 99 % bit-identical per case."""
        TE.TestEncoderEmu.test_matches_reference(self)

    def test_reset_03_T2(self):
        rng = np.random.default_rng(11)
        x = E.random_inputs(rng, rw=40, rh=24, reset=True)
        p = E.random_params(rng)
        a = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
        x["reduced_history"] = np.ones_like(x["reduced_history"])
        x["latent"] = -np.ones_like(x["latent"])
        x["motion_history"] = np.full_like(x["motion_history"], 1000.0)
        b = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        judge(self, "enc0 reset features", a["features"], E.encode(x, p)["features"],
              TK.ulp_distance(a["features"], E.encode(x, p)["features"]))

    def test_partly_valid_tiles_have_no_garbage(self):
        """The cases with a token count that is not a multiple of 16 (130 and 126 tokens, 13 x 10 and 12 x 11): every feature of
        the last, partly valid tile is finite and within the limits above (the hazard gave NaN)."""
        for i, (R, F, xo, po) in enumerate(self.CASES):
            Wt = (F[0] if F else -(-R[0] // 32) * 32) // 2
            Ht = (F[1] if F else -(-R[1] // 32) * 32) // 2
            if (Wt * Ht) % 16 == 0:
                continue
            with self.subTest(case=i, tokens=Wt * Ht):
                x, p = self.make_case(i, R, F, xo, po)
                got = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)["features"]
                self.assertTrue(np.all(np.isfinite(got)))
                ref = E.encode(x, p)["features"]
                self.assertLess(float(np.mean(TK.ulp_distance(got, ref) > 0)), 0.05)

    def test_one_code_flips_against_the_gfx11_emulator(self):
        """The same inputs through the gfx11 emulator (exact sums): reported per case, bounded at 5 %."""
        emu = self.Runner(TG.build_dir(), "", backend="emu", arch="gfx11")
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i):
                x, p = self.make_case(i, R, F, xo, po)
                got = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
                want = emu.run("enc0", emu.pack_weights("enc0", p), x, None)
                u = TK.ulp_distance(got["features"], want["features"])
                print(f"enc0 case {i} {R}: {np.mean(u > 0) * 1e6:.0f} per million one-code flips, {np.mean(u > 1) * 1e6:.0f} per "
                      f"million more than one step (card vs gfx11 emulator)")
                self.assertLess(float(np.mean(u > 0)), 0.05)
                for k in ("m_near", "m_far", "m_center"):
                    np.testing.assert_array_equal(got[k], want[k], err_msg=k)

    def test_the_row_table_matters(self):
        a = self.run_once()
        wrong = self.Runner(os.environ["WG_BUILD"], os.environ.get("WG_ROCM", ""), variant=self.variant, d_row="gfx12")
        b = self.run_once(wrong)
        self.assertGreater(float(np.mean(a["features"] != b["features"])), 0.2)

    def run_once(self, runner=None):
        rng = np.random.default_rng(21)
        x = E.random_inputs(rng, rw=40, rh=24)
        p = E.random_params(rng)
        r = runner or self.runner
        return r.run("enc0", r.pack_weights("enc0", p), x, None)

    def test_stage_time(self):
        """Median of 50 launches, three runs, at the in-game sizes (inputs already on the card): printed, finite."""
        r = self.runner
        for name, (rw, rh) in TIMED_SIZES.items():
            rng = np.random.default_rng(5)
            x = E.random_inputs(rng, rw=rw, rh=rh)
            w = r.pack_weights("enc0", E.random_params(rng))
            t = [r.time("enc0", w, x, None, repeats=50) for _ in range(RUNS)]
            print(f"enc0 {name} render: median of 50 launches {' '.join(f'{v:.3f}' for v in t)} ({np.median(t):.3f}) ms on "
                  f"{r.dev.device_name}")
            self.assertTrue(all(0.0 < v < 1000.0 for v in t))


class TestDecoderGpu(E4Gpu, TD.TestDecoderEmu):
    """dec0 on the card, the cases of test_decoder_emu."""

    def judge_all(self, label, got, ref, totals=None):
        self.assertEqual(set(got), {"A", "B", "C", "latent"})
        for k, (g, r, steps) in dec0_steps(got, ref).items():
            n, d = judge(self, f"{label} {k}", g, r, steps)
            if totals is not None:
                totals[0] += n
                totals[1] += d

    def test_matches_reference(self):
        totals = [0, 0]
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i, R=R, F=F, x=xo, p=po):
                rng = np.random.default_rng(500 + i)
                fw, fh = F if F else (None, None)
                x = Dc.random_inputs(rng, rw=R[0], rh=R[1], fw=fw, fh=fh, **xo)
                p = Dc.random_params(rng, **po)
                self.judge_all(f"dec0 case {i} {R}", self.run_dec0(x, p), Dc.decode(x, p), totals)
        print(f"dec0 all cases: {totals[1] / totals[0] * 100:.4f} % of the values differ from the reference ({totals[0]} values)")

    @unittest.expectedFailure
    def test_stricter_limits_of_the_emulator_test(self):
        """>= 99.9 % of the values identical per output and case, at most 1e-4 more than one step off."""
        TD.TestDecoderEmu.test_matches_reference(self)

    def test_reset_05_T2(self):
        rng = np.random.default_rng(12)
        x = Dc.random_inputs(rng, reset=True)
        p = Dc.random_params(rng)
        a = self.run_dec0(x, p)
        b = self.run_dec0(dict(x, latent_prev=-x["latent_prev"]), p)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.judge_all("dec0 reset", a, Dc.decode(x, p))

    def test_border_05_T3(self):
        from clean.tests.test_decoder_reference import border_case
        x, p, expect = border_case()
        got = self.run_dec0(x, p)
        nores = self.run_dec0(dict(x, reset=True), p)["latent"]
        for (yy, xx), ok in expect.items():
            self.assertEqual(bool(np.any(got["latent"][yy, xx] != nores[yy, xx])), ok, (yy, xx))
        self.judge_all("dec0 border", got, Dc.decode(x, p))

    def test_one_code_flips_against_the_gfx11_emulator(self):
        emu = self.Runner(TG.build_dir(), "", backend="emu", arch="gfx11")
        for i, (R, F, xo, po) in enumerate(self.CASES):
            with self.subTest(case=i):
                rng = np.random.default_rng(500 + i)
                fw, fh = F if F else (None, None)
                x = Dc.random_inputs(rng, rw=R[0], rh=R[1], fw=fw, fh=fh, **xo)
                p = Dc.random_params(rng, **po)
                got = self.run_dec0(x, p)
                want = self.run_dec0(x, p, runner=emu)
                for k, (g, r, steps) in dec0_steps(got, want).items():
                    print(f"dec0 case {i} {R} {k}: {np.mean(steps > 0) * 1e6:.0f} per million flips, {np.mean(steps > 1) * 1e6:.0f} "
                          f"per million more than one step (card vs gfx11 emulator)")
                    self.assertLess(float(np.mean(steps > 0)), 0.05, k)

    def test_stage_time(self):
        r = self.runner
        for name, (rw, rh) in TIMED_SIZES.items():
            rng = np.random.default_rng(5)
            x = Dc.random_inputs(rng, rw=rw, rh=rh)
            w = r.pack_weights("dec0", Dc.random_params(rng))
            t = [r.time("dec0", w, x, None, repeats=50) for _ in range(RUNS)]
            print(f"dec0 {name} render: median of 50 launches {' '.join(f'{v:.3f}' for v in t)} ({np.median(t):.3f}) ms on "
                  f"{r.dev.device_name}")
            self.assertTrue(all(0.0 < v < 1000.0 for v in t))


class TestDec1Dec0Gpu(E4Gpu, TD1.TestDec1Dec0Emu):
    """The fused dec1 + dec0 on the card: the same bytes as the two stages one after the other (the fused kernel's contract),
    and against the reference: dec0's reference fed the card's own dec1 output."""

    def test_fused_equals_separate(self):
        super().test_fused_equals_separate()

    def test_against_the_reference_fed_the_card_s_own_dec1_output(self):
        r = self.runner
        totals = [0, 0]
        for i, (R, F, offset, reset) in enumerate(TD1.CASES):
            x, t = TD1.make(i, R, F, offset, reset)
            with self.subTest(case=i, R=R, offset=offset, reset=reset):
                got = r.run("dec1dec0", r.pack_weights("dec1dec0", t), x, None)
                z = r.run("dec1", r.pack_weights("dec1", t["dec1"]), x["y"], x["offset"], x["skip"])["out"]
                zref = S.run_stage("dec1", t["dec1"], x["y"], x["offset"], x["skip"], gelu="sigmoid")["out"]
                judge(self, f"dec1 case {i} out", z, zref, TK.ulp_distance(z, zref))
                xd = dict({k: v for k, v in x.items() if k not in ("y", "skip", "offset")}, features=z)
                ref = Dc.decode(xd, t["dec0"])
                self.assertEqual(set(got), {"A", "B", "C", "latent"})
                for k, (g, rf, steps) in dec0_steps(got, ref).items():
                    n, d = judge(self, f"dec1dec0 case {i} {R} {k}", g, rf, steps)
                    totals[0] += n
                    totals[1] += d
        print(f"dec1dec0 all cases: {totals[1] / totals[0] * 100:.4f} % of the dec0 values differ from the reference "
              f"({totals[0]} values)")

    def test_stage_time(self):
        """Median of 50 launches, three runs: the fused kernel against dec1 then dec0 as two launches."""
        r = self.runner
        for name, (rw, rh) in TIMED_SIZES.items():
            rng = np.random.default_rng(5)
            fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
            H, W = fh // 2, fw // 2
            p1, p0 = S.random_params("dec1", rng), Dc.random_params(rng)
            y, skip = S.random_inputs("dec1", H, W, rng)
            xd = Dc.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh)
            x = {k: v for k, v in xd.items() if k != "features"}
            x.update(y=y, skip=skip, offset=S.window_offset("dec1", 0))
            w = r.pack_weights("dec1dec0", {"dec1": p1, "dec0": p0})
            row = {}
            for label, stage in (("fused", "dec1dec0@fused"), ("two launches", "dec1dec0")):
                t = [r.time(stage, w, x, None, repeats=50) for _ in range(RUNS)]
                row[label] = (t, float(np.median(t)))
            print(f"dec1dec0 {name} render: " + "; ".join(
                f"{k} {' '.join(f'{v:.3f}' for v in t)} ({m:.3f}) ms" for k, (t, m) in row.items()))
            self.assertTrue(all(0.0 < m < 1000.0 for _, m in row.values()))


if __name__ == "__main__":
    unittest.main()
