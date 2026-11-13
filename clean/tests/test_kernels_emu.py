# SPDX-License-Identifier: MIT
"""Run the real kernel sources in the host lane emulator and compare against the NumPy reference.

Needs a host C++ compiler (clang++ or g++, env CXX) on Linux; the emulator library is built into a temporary
directory. Run: python -m unittest discover -s clean/tests -t .
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clean.reference import fp8
from clean.reference import swin as S
from clean.runner import build

_BUILD = None


def host_cxx():
    """The host C++ compiler for emulator builds; skips (as build_dir does) off Linux or without a compiler."""
    if not sys.platform.startswith("linux"):
        raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        raise unittest.SkipTest("no host C++ compiler")
    return cxx


def build_dir():
    global _BUILD
    if _BUILD is None:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("no host C++ compiler")
        d = tempfile.mkdtemp(prefix="wg_emu_")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu.so")), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("emulator build failed:\n" + r.stderr)
        _BUILD = d
    return _BUILD


def ulp_distance(a, b):
    ca, cb = fp8.fp8_encode(a).astype(int), fp8.fp8_encode(b).astype(int)
    oa = np.where(ca & 0x80, -(ca & 0x7F), ca)
    ob = np.where(cb & 0x80, -(cb & 0x7F), cb)
    return np.abs(oa - ob)


def spec09_metrics(got, ref):
    """The per-stage measures of spec/09-tolerances.md."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    finite = np.isfinite(got)
    d = np.abs(np.where(finite, got, 0.0) - ref)
    return {
        "nonfinite": int(np.sum(~finite)),
        "frac_over_1_step": float(np.mean(ulp_distance(np.where(finite, got, 0.0), ref) > 1)),
        "mean_abs_err": float(d.mean()),
        "max_abs_err": float(d.max()),
        "ref_rms": float(np.sqrt(np.mean(ref * ref))),
    }


class EmuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from clean.runner.stage_runner import Runner
        cls.Runner = Runner
        cls.runner = Runner(build_dir(), "", backend="emu")

    def setUp(self):
        self.runner.dev.set_layout(0, 0)

    def run_case(self, stage, H, W, offset, seed, runner=None):
        rng = np.random.default_rng(seed)
        p = S.random_params(stage, rng)
        x, skip = S.random_inputs(stage, H, W, rng)
        r = runner or self.runner
        got = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
        return got, (p, x, skip)


class TestStagesEmu(EmuTest):
    CASES = [
        ("enc1", 8, 12, (2, 2)), ("enc1", 12, 20, (0, 2)), ("enc1", 20, 24, (2, 2)),
        ("enc2", 8, 8, (2, 2)), ("enc2", 12, 8, (0, 2)),
        ("b0", 8, 8, (2, 2)), ("b1", 12, 8, (0, 0)), ("b3", 8, 12, (2, 0)),
        ("dec2", 8, 12, (0, 0)), ("dec2", 12, 8, (2, 0)),
        ("dec1", 8, 8, (0, 0)), ("dec1", 12, 20, (2, 0)),
        ("enc1", 36, 32, (2, 2)), ("dec1", 36, 32, (2, 0)),  # several workgroups in every kernel
    ]

    def test_stages_match_reference(self):
        """Kernels (default sigmoid GELU) vs the reference with the same GELU formula (spec 02, FP16 attention weights).

        Nearly all values are bit-identical. An intermediate value that lands on an FP8 rounding midpoint can round
        differently in FP32 (kernel) and FP64 (reference); through the softmax that changes a few outputs of the
        same window by several FP8 steps (on these small maps one window is up to 25% of the tokens). Checked:
        spec 09 acceptance (no non-finite values, <= 2% more than one FP8 step off, mean abs error <= 1% of the
        reference RMS) per case, and the stricter: < 5% of values differ per case, < 1% over all cases.
        Layout or indexing bugs change most values.
        """
        total = differ = 0
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=100 + i)
                ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
                self.assertEqual(set(got), set(ref))
                for k in ref:
                    self.assertEqual(got[k].shape, ref[k].shape)
                    m = spec09_metrics(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0, k)
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (k, m))
                    u = ulp_distance(got[k], ref[k])
                    self.assertLess(float(np.mean(u > 0)), 0.05, k)
                    total += u.size
                    differ += int(np.sum(u > 0))
        self.assertLess(differ / total, 0.01)


    def test_default_vs_exact_gelu(self):
        """The default GELU is an approximation of the exact one (accepted on the whole-frame rule of spec 09 draft 2,
        not on the per-stage limits). Here: block outputs stay within the per-stage limits on synthetic data; the enc
        *merged* outputs are only reported (random networks amplify the differences: 3-4 % more than one step)."""
        for i, (stage, H, W, offset) in enumerate((("b1", 12, 16, (0, 0)), ("enc1", 20, 24, (2, 2)),
                                                   ("dec1", 20, 24, (2, 0)), ("dec2", 12, 16, (2, 0)))):
            with self.subTest(stage=stage):
                got, (p, x, skip) = self.run_case(stage, H, W, offset, seed=700 + i)
                ref = S.run_stage(stage, p, x, offset, skip, gelu="exact")
                for k in ref:
                    m = spec09_metrics(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0)
                    if k != "merged":
                        self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))
                        self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (k, m))
                    else:
                        self.assertLessEqual(m["frac_over_1_step"], 0.10, (k, m))


class TestLayoutIndependence(EmuTest):
    def test_k_slot_permutation_does_not_matter(self):
        a, _ = self.run_case("enc2", 8, 8, (2, 2), seed=7)
        self.runner.dev.set_layout(1, 0)
        b, _ = self.run_case("enc2", 8, 8, (2, 2), seed=7)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

    def test_accumulator_layout_is_absorbed_by_packing(self):
        a, _ = self.run_case("dec1", 8, 8, (2, 0), seed=8)
        self.runner.dev.set_layout(1, 1)
        r11 = self.Runner(build_dir(), "", backend="emu", d_row="gfx11")
        b, _ = self.run_case("dec1", 8, 8, (2, 0), seed=8, runner=r11)
        np.testing.assert_array_equal(a["out"], b["out"])
        # sanity: packing for the wrong accumulator layout must break the result
        c, _ = self.run_case("dec1", 8, 8, (2, 0), seed=8)
        self.assertGreater(float(np.mean(a["out"] != c["out"])), 0.2)


class TestVariantsEmu(unittest.TestCase):
    """Build variants (variants.py) change workgroup shapes and work per wave but never the result."""

    CASES = [("enc1", 20, 24, (2, 2)), ("b1", 12, 16, (0, 0)), ("enc2", 12, 16, (0, 2)), ("dec1", 20, 24, (2, 0)),
             ("dec2", 12, 16, (2, 0))]

    def test_variants_bit_identical_to_base(self):
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        base = Runner(build_dir(), "", backend="emu")
        for name in ("t128", "t512", "c64w2", "nt3", "g128", "pair", "forceclamp", "pf1", "pf2"):
            d = tempfile.mkdtemp(prefix="wg_emu_v_")
            r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.VARIANTS[name]),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, V.MANIFEST), "w") as f:  # what build.py --variants writes
                json.dump({name: {"file": V.code_object(name), "defines": V.VARIANTS[name],
                                  "geometry": V.geometry(V.VARIANTS[name])}}, f)
            variant = Runner(d, "", backend="emu", variant=name)
            for i, (stage, H, W, offset) in enumerate(self.CASES):
                with self.subTest(variant=name, stage=stage):
                    rng = np.random.default_rng(300 + i)
                    p = S.random_params(stage, rng)
                    x, skip = S.random_inputs(stage, H, W, rng)
                    want = base.run(stage, base.pack_weights(stage, p), x, offset, skip)
                    got = variant.run(stage, base.pack_weights(stage, p), x, offset, skip)
                    for k in want:
                        np.testing.assert_array_equal(got[k], want[k])


    def test_fused_merge_and_expand_bit_identical_to_base(self):
        """Variant `fuse` (patch merge inside the encoder block, expand inside the decoder block): same bytes as the
        separate kernels for every window offset, border windows included; odd offsets fall back to the separate path."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = host_cxx()
        name = "fuse"
        d = tempfile.mkdtemp(prefix="wg_emu_f_")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.VARIANTS[name]),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({name: {"file": V.code_object(name), "defines": V.VARIANTS[name],
                              "geometry": V.geometry(V.VARIANTS[name])}}, f)
        fused = Runner(d, "", backend="emu", variant=name)
        base = Runner(build_dir(), "", backend="emu")
        cases = [("enc1", 12, 16, (0, 0)), ("enc1", 20, 24, (2, 2)), ("enc1", 16, 12, (0, 2)), ("enc1", 12, 12, (1, 0)),
                 ("enc2", 12, 16, (2, 2)), ("enc2", 16, 20, (0, 0)), ("enc2", 8, 8, (2, 0)),
                 ("dec1", 12, 16, (0, 0)), ("dec1", 20, 24, (2, 0)), ("dec1", 8, 12, (0, 2)), ("dec1", 12, 8, (3, 0)),
                 ("dec2", 12, 16, (2, 0)), ("dec2", 16, 20, (0, 0)), ("dec2", 8, 8, (2, 2))]
        for i, (stage, H, W, offset) in enumerate(cases):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                rng = np.random.default_rng(700 + i)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                w = base.pack_weights(stage, p)
                want = base.run(stage, w, x, offset, skip)
                got = fused.run(stage, w, x, offset, skip)
                self.assertEqual(set(got), set(want))
                for k in want:
                    np.testing.assert_array_equal(got[k], want[k])

    def test_dot4norm_changes_only_the_summation_order(self):
        """WG_NORM_DOT4 (first RMSNorm statistics from FP8 dot products): same results up to rare one-step flips."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = host_cxx()
        name = "dot4norm"
        d = tempfile.mkdtemp(prefix="wg_emu_d_")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.VARIANTS[name]),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({name: {"file": V.code_object(name), "defines": V.VARIANTS[name],
                              "geometry": V.geometry(V.VARIANTS[name])}}, f)
        var = Runner(d, "", backend="emu", variant=name)
        base = Runner(build_dir(), "", backend="emu")
        for i, (stage, H, W, offset) in enumerate((("enc1", 12, 16, (2, 2)), ("b1", 12, 16, (0, 0)), ("dec1", 12, 16, (0, 0)))):
            with self.subTest(stage=stage):
                rng = np.random.default_rng(800 + i)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                w = base.pack_weights(stage, p)
                want = base.run(stage, w, x, offset, skip)
                got = var.run(stage, w, x, offset, skip)
                for k in want:
                    self.assertLess(float(np.mean(got[k] != want[k])), 0.01)
                    if stage == "dec1":  # decoder blocks do not take this path at all
                        np.testing.assert_array_equal(got[k], want[k])

    def test_gelu_variants_match_their_references(self):
        """The GELU variants (not bit-identical to base) against the reference with the matching GELU formula, and
        against the exact reference with the per-stage limits where they hold (exact, A&S 7.1.25)."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = host_cxx()
        refname = {"geluexact": "exact", "gelu725": "as725", "gelu16": "poly16"}
        for name in sorted(n for n in V.INEXACT if n in refname):
            d = tempfile.mkdtemp(prefix="wg_emu_x_")
            r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.VARIANTS[name]),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, V.MANIFEST), "w") as f:
                json.dump({name: {"file": V.code_object(name), "defines": V.VARIANTS[name],
                                  "geometry": V.geometry(V.VARIANTS[name])}}, f)
            variant = Runner(d, "", backend="emu", variant=name)
            for i, (stage, H, W, offset) in enumerate((("b1", 12, 16, (0, 0)), ("enc2", 12, 16, (0, 2)),
                                                       ("dec1", 20, 24, (2, 0)))):
                with self.subTest(variant=name, stage=stage):
                    rng = np.random.default_rng(500 + i)
                    p = S.random_params(stage, rng)
                    x, skip = S.random_inputs(stage, H, W, rng)
                    got = variant.run(stage, variant.pack_weights(stage, p), x, offset, skip)
                    ref = S.run_stage(stage, p, x, offset, skip, gelu=refname[name])
                    for k in ref:
                        m = spec09_metrics(got[k], ref[k])
                        self.assertEqual(m["nonfinite"], 0)
                        self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
                        self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)
                    if name != "gelu16":    # gelu16 is judged on the whole-frame rule only (2-6 % FP8 flips)
                        exact = S.run_stage(stage, p, x, offset, skip, gelu="exact")
                        for k in exact:
                            m = spec09_metrics(got[k], exact[k])
                            self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))

    def test_attention_variants_run_and_stay_finite(self):
        """av8 / smax16 trade accuracy for little speed (they are sweep variants only): they must run and be finite."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = host_cxx()
        for name in ("av8", "smax16", "vec16", "gelumix1", "gelumix2", "gelumix3"):
            d = tempfile.mkdtemp(prefix="wg_emu_att_")
            r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.VARIANTS[name]),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, V.MANIFEST), "w") as f:
                json.dump({name: {"file": V.code_object(name), "defines": V.VARIANTS[name],
                                  "geometry": V.geometry(V.VARIANTS[name])}}, f)
            variant = Runner(d, "", backend="emu", variant=name)
            rng = np.random.default_rng(950)
            p = S.random_params("b1", rng)
            x, skip = S.random_inputs("b1", 8, 12, rng)
            out = variant.run("b1", variant.pack_weights("b1", p), x, (0, 0), skip)
            ref = S.run_stage("b1", p, x, (0, 0), skip, gelu="sigmoid")
            m = spec09_metrics(out["out"], ref["out"])
            self.assertEqual(m["nonfinite"], 0)
            self.assertLess(m["mean_abs_err"], 0.1 * m["ref_rms"], (name, m))

    def test_ablation_builds_compile_and_run(self):
        """Ablations (timing-only builds with wrong output) must still build and run to completion."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        cxx = host_cxx()
        for name in V.ABLATIONS:
            d = tempfile.mkdtemp(prefix="wg_emu_a_")
            r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.ABLATIONS[name]),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(os.path.join(d, V.MANIFEST), "w") as f:
                json.dump({name: {"file": V.code_object(name), "defines": V.ABLATIONS[name],
                                  "geometry": V.geometry(V.ABLATIONS[name]), "ablation": True}}, f)
            runner = Runner(d, "", backend="emu", variant=name)
            for stage, H, W, offset in (("b1", 8, 12, (2, 0)), ("dec1", 8, 8, (0, 0))):
                with self.subTest(ablation=name, stage=stage):
                    rng = np.random.default_rng(400)
                    p = S.random_params(stage, rng)
                    x, skip = S.random_inputs(stage, H, W, rng)
                    out = runner.run(stage, runner.pack_weights(stage, p), x, offset, skip)
                    self.assertEqual(out["out"].shape, (H, W, S.STAGES[stage]["C"]))


    def test_profile_builds_are_exact_and_report_per_wave_words(self):
        """The instrumented builds compute the same output as the base build and fill one 16-word record per wave."""
        from clean.runner import variants as V
        from clean.runner.stage_runner import Runner
        from clean.runner import profile_stages
        cxx = host_cxx()
        name = "prof"
        d = tempfile.mkdtemp(prefix="wg_emu_p_")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{name}.so"), V.PROFILES[name]),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({name: {"file": V.code_object(name), "defines": V.PROFILES[name],
                              "geometry": V.geometry(V.PROFILES[name]), "ablation": False, "profile": True}}, f)
        prof = Runner(d, "", backend="emu", variant=name)
        base = Runner(build_dir(), "", backend="emu")
        for stage, H, W, offset in (("b1", 8, 12, (2, 0)), ("dec1", 8, 8, (0, 0))):
            with self.subTest(stage=stage):
                rng = np.random.default_rng(500)
                p = S.random_params(stage, rng)
                x, skip = S.random_inputs(stage, H, W, rng)
                a = prof.run(stage, prof.pack_weights(stage, p), x, offset, skip)
                b = base.run(stage, base.pack_weights(stage, p), x, offset, skip)
                for k in b:
                    np.testing.assert_array_equal(a[k], b[k])
                words, info = prof.profile(stage, prof.pack_weights(stage, p), x, offset, skip)
                self.assertEqual(words.shape, (info["waves"], V.PROF_WORDS))
                self.assertTrue((words[:, 12] > 0).all())
                res = profile_stages.analyse(words, info, stage)
                self.assertEqual(res["waves"], info["waves"])
                self.assertIn("gelu+pack", profile_stages.format_stage(res))


class TestSafeFlag(EmuTest):
    def test_flag_set_for_normal_weights_and_clear_for_huge_ones(self):
        import struct
        rng = np.random.default_rng(900)
        p = S.random_params("b1", rng)
        self.assertEqual(struct.unpack("<I", self.runner.pack_weights("b1", p)[:4])[0] & 1, 1)
        q = dict(p)
        q["gamma1"] = (np.asarray(p["gamma1"]) * 800.0).astype(np.float32)
        self.assertEqual(struct.unpack("<I", self.runner.pack_weights("b1", q)[:4])[0] & 1, 0)

    def test_unsafe_weights_still_match_the_reference(self):
        """With a huge gamma1 the clamp-free path must not be used: values saturate exactly like the reference."""
        rng = np.random.default_rng(901)
        for stage, H, W, offset in (("b1", 8, 12, (0, 0)), ("enc1", 8, 12, (2, 2))):
            p = S.random_params(stage, rng)
            p["gamma1"] = (np.asarray(p["gamma1"]) * 300.0).astype(np.float32)
            x, skip = S.random_inputs(stage, H, W, rng)
            got = self.runner.run(stage, self.runner.pack_weights(stage, p), x, offset, skip)
            ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
            for k in ref:
                self.assertTrue(np.isfinite(got[k]).all())
                m = spec09_metrics(got[k], ref[k])
                self.assertLessEqual(m["frac_over_1_step"], 0.05, (stage, k, m))


class TestSelftestEmu(unittest.TestCase):
    def test_selftest_passes_in_emulator(self):
        from clean.runner import selftest
        report = selftest.selftest(build_dir(), backend="emu")
        self.assertTrue(report["pass"], report)
        self.assertEqual(report["wmma_fp8"]["d_row_name"], "gfx12")


if __name__ == "__main__":
    unittest.main()
