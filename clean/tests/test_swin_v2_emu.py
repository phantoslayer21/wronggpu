# SPDX-License-Identifier: MIT
"""The second Swin block design (kernels/swin_v2.hip, build variants "v2*") in the host lane emulator.

Same checks as test_kernels_emu.py for the first design: the exact-GELU build against the NumPy reference with the
same strict criteria, the default (gelusig) build within spec 09 on the block outputs, all v2 geometry / schedule
variants bit-identical to v2, layout independence, ablation builds, the conversion probe, and the bmega builds
(base arithmetic and the bottleneck megakernel) bit-identical to the base build.
Run: python -m unittest discover -s clean/tests -t .
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from clean.reference import swin as S
from clean.runner import build
from clean.runner import variants as V
from clean.tests.test_kernels_emu import spec09_metrics, ulp_distance

_DIR = None


def v2_build_dir():
    """Emulator libraries of the base build and of every v2 build (variants and ablations) in one directory, with its
    variants.json."""
    global _DIR
    if _DIR is None:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("no host C++ compiler")
        d = tempfile.mkdtemp(prefix="wg_emu_v2_")
        names = list(V.V2_VARIANTS) + list(V.V2_ABLATIONS)
        names += sorted(set(V.MEGA_REFERENCE.values()) - {"base"})  # swin.hip builds the bmega* builds must match

        def one(n):
            lib = "wg_emu.so" if n == "base" else f"wg_emu__{n}.so"
            cmd = build.emu_command(cxx, os.path.join(d, lib), V.all_builds()[n], V.source(n))
            return n, subprocess.run(cmd, capture_output=True, text=True)

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            for n, r in ex.map(one, ["base"] + names):
                if r.returncode != 0:
                    raise RuntimeError(f"emulator build of {n} failed:\n" + r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({n: {"file": V.code_object(n), "defines": V.all_builds()[n],
                           "geometry": V.variant_geometry(n), "ablation": V.is_ablation(n)} for n in ["base"] + names},
                      f)
        _DIR = d
    return _DIR


def runner(variant, d_row=None):
    from clean.runner.stage_runner import Runner
    return Runner(v2_build_dir(), "", backend="emu", variant=variant, d_row=d_row)


def case(stage, H, W, seed):
    rng = np.random.default_rng(seed)
    p = S.random_params(stage, rng)
    x, skip = S.random_inputs(stage, H, W, rng)
    return p, x, skip


# Several workgroups and partly filled last workgroups for every kernel (C=64: 512-lane groups, 2 windows per wave).
CASES = [
    ("enc1", 8, 12, (2, 2)), ("enc1", 12, 20, (0, 2)), ("enc1", 20, 24, (2, 2)),
    ("enc2", 8, 8, (2, 2)), ("enc2", 12, 8, (0, 2)),
    ("b0", 8, 8, (2, 2)), ("b1", 12, 8, (0, 0)), ("b3", 8, 12, (2, 0)),
    ("dec2", 8, 12, (0, 0)), ("dec2", 12, 8, (2, 0)),
    ("dec1", 8, 8, (0, 0)), ("dec1", 12, 20, (2, 0)),
    ("enc1", 36, 32, (2, 2)), ("dec1", 36, 32, (2, 0)), ("b2", 40, 36, (2, 2)), ("enc2", 36, 40, (0, 2)),
]


class TestV2Exact(unittest.TestCase):
    def test_v2exact_matches_reference(self):
        """v2 with the exact GELU: the criteria of test_kernels_emu.TestStagesEmu (nearly all values bit-identical
        to the reference; the rest are FP8 rounding-midpoint cases)."""
        r = runner("v2exact")
        total = differ = 0
        for i, (stage, H, W, offset) in enumerate(CASES):
            with self.subTest(stage=stage, H=H, W=W, offset=offset):
                p, x, skip = case(stage, H, W, 100 + i)
                got = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
                ref = S.run_stage(stage, p, x, offset, skip)
                self.assertEqual(set(got), set(ref))
                for k in ref:
                    m = spec09_metrics(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0, k)
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (k, m))
                    u = ulp_distance(got[k], ref[k])
                    self.assertLess(float(np.mean(u > 0)), 0.05, k)
                    total += u.size
                    differ += int(np.sum(u > 0))
        self.assertLess(differ / total, 0.01)


class TestV2Default(unittest.TestCase):
    def test_v2_within_spec09(self):
        """Default v2 (GELU table): block outputs within the spec 09 per-stage measures on synthetic data (the merged
        outputs of random networks are not, as for swin.hip's gelusig; reported by the spec team on real data)."""
        for name in ("v2", "v2gsig", "v2g7"):
            self._within_spec09(name)

    def _within_spec09(self, name):
        r = runner(name)
        for i, (stage, H, W, offset) in enumerate(CASES[:12]):
            with self.subTest(variant=name, stage=stage, H=H, W=W, offset=offset):
                p, x, skip = case(stage, H, W, 200 + i)
                got = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
                ref = S.run_stage(stage, p, x, offset, skip)
                for k in ref:
                    m = spec09_metrics(got[k], ref[k])
                    self.assertEqual(m["nonfinite"], 0, k)
                    if k == "merged":
                        continue
                    self.assertLessEqual(m["frac_over_1_step"], 0.02, (k, m))
                    self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], (k, m))

    def test_nan_in_input_stays_visible(self):
        """A NaN in a token's input shows up as a non-finite value of that token's output."""
        r = runner("v2")
        for stage, H, W, offset in (("enc1", 8, 12, (2, 2)), ("b1", 8, 8, (0, 0)), ("enc2", 8, 8, (0, 2))):
            with self.subTest(stage=stage):
                p, x, skip = case(stage, H, W, 7)
                x = x.copy()
                x[3, 5, 7] = np.nan
                got = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
                out = got["skip" if "skip" in got else "out"]
                self.assertFalse(np.all(np.isfinite(out[3, 5])))


class TestV2Variants(unittest.TestCase):
    # Same arithmetic in the same order as v2: bit-identical outputs.
    EXACT_FAMILY = ["v2nopipe", "v2c64t256", "v2c64w1", "v2c64w4", "v2c64s", "v2t256", "v2w2", "v2nofence", "v2prof"]
    CASES = [("enc1", 20, 24, (2, 2)), ("b1", 12, 16, (0, 0)), ("enc2", 12, 16, (0, 2)), ("dec1", 20, 24, (2, 0)),
             ("dec2", 12, 16, (2, 0)), ("dec1", 36, 32, (0, 0))]

    def test_variants_bit_identical_to_v2(self):
        """Geometry and scheduling variants of v2 change only the speed: outputs bit-identical to v2."""
        base = runner("v2")
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            p, x, skip = case(stage, H, W, 300 + i)
            w = base.pack_weights(stage, p)
            want = base.run(stage, w, x, offset, skip)
            for name in self.EXACT_FAMILY:
                with self.subTest(variant=name, stage=stage):
                    got = runner(name).run(stage, w, x, offset, skip)
                    for k in want:
                        np.testing.assert_array_equal(got[k], want[k])

    def test_lds32_matches_g7(self):
        """v2lds32 (header from global memory, 8 KB units, 8 KB table) computes exactly as v2g7."""
        g7, lds32 = runner("v2g7"), runner("v2lds32")
        for i, (stage, H, W, offset) in enumerate(self.CASES):
            p, x, skip = case(stage, H, W, 350 + i)
            w = g7.pack_weights(stage, p)
            c, d = g7.run(stage, w, x, offset, skip), lds32.run(stage, w, x, offset, skip)
            for k in c:
                with self.subTest(stage=stage, k=k):
                    np.testing.assert_array_equal(d[k], c[k])

    def test_ablations_run(self):
        for name in list(V.V2_ABLATIONS) + ["v2gbyte"]:
            r = runner(name)
            for stage, H, W, offset in (("b1", 8, 12, (2, 0)), ("dec1", 8, 8, (0, 0)), ("enc1", 8, 8, (2, 2))):
                with self.subTest(ablation=name, stage=stage):
                    p, x, skip = case(stage, H, W, 400)
                    out = r.run(stage, r.pack_weights(stage, p), x, offset, skip)
                    self.assertEqual(out["skip" if "skip" in out else "out"].shape, (H, W, S.STAGES[stage]["C"]))

    def test_layout_independence(self):
        """K-slot permutation does not matter; the accumulator layout is absorbed by the packing (as for swin.hip)."""
        r = runner("v2exact")
        try:
            p, x, skip = case("enc2", 8, 8, 7)
            a = r.run("enc2", r.pack_weights("enc2", p), x, (2, 2), skip)
            r.dev.set_layout(1, 0)
            b = r.run("enc2", r.pack_weights("enc2", p), x, (2, 2), skip)
            for k in a:
                np.testing.assert_array_equal(a[k], b[k])
            r.dev.set_layout(0, 0)
            p, x, skip = case("dec1", 8, 8, 8)
            a = r.run("dec1", r.pack_weights("dec1", p), x, (2, 0), skip)
            r11 = runner("v2exact", d_row="gfx11")
            r11.dev.set_layout(1, 1)
            b = r11.run("dec1", r11.pack_weights("dec1", p), x, (2, 0), skip)
            np.testing.assert_array_equal(a["out"], b["out"])
        finally:
            r.dev.set_layout(0, 0)


def same(a, b):
    """Bit-identical FP8-decoded maps (NaN equals NaN)."""
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def flags_cleared(w):
    """The packed weights with flags bit 0 cleared: the runner then launches the clamping kernels."""
    return bytes([w[0] & 0xFE]) + bytes(w[1:])


def bneck_case(H, W, seed):
    rng = np.random.default_rng(seed)
    params = [S.random_params(f"b{k}", rng) for k in range(6)]
    x, _ = S.random_inputs("b0", H, W, rng)
    base = runner("base")
    return base, params, [base.pack_weights(f"b{k}", params[k]) for k in range(6)], x


class TestBmega(unittest.TestCase):
    """bmega builds: swin_v2.hip with WG_V2_BASEMATH, which must give exactly the bytes of the base build (swin.hip) in
    every kernel, and the bottleneck megakernel swin_bneck6 (six blocks in one launch)."""

    def test_bneck_stage(self):
        """Stage "bneck" of the runner: pack_weights("bneck", ...) and run("bneck", ..., b0's offset) launch the
        megakernel in a bmega* build and six block kernels in the base build; same bytes."""
        rng = np.random.default_rng(630)
        params = {f"b{k}": S.random_params(f"b{k}", rng) for k in range(6)}
        x, _ = S.random_inputs("b0", 8, 12, rng)
        base, r = runner("base"), runner("bmegahg")
        w = base.pack_weights("bneck", params)
        self.assertEqual(len(w), 6 * len(base.pack_weights("b0", params["b0"])))
        for parity in (0, 1):
            off = S.window_offset("b0", parity)
            plan = r._prepare("bneck", w, x, off, None)
            try:
                self.assertEqual([l[0] for l in plan["launches"]], ["swin_bneck6_nc"])
            finally:
                r._release(plan)
            with self.subTest(parity=parity):
                self.assertTrue(same(r.run("bneck", w, x, off)["out"], base.run("bneck", w, x, off)["out"]))

    def test_stage_kernels_bit_identical_to_base(self):
        base, r = runner("base"), runner("bmega")
        cases = [("enc1", 20, 24, (2, 2)), ("enc2", 12, 16, (0, 2)), ("b0", 12, 16, (2, 2)), ("b1", 12, 8, (0, 0)),
                 ("dec2", 12, 16, (2, 0)), ("dec1", 20, 24, (0, 0))]
        for i, (stage, H, W, offset) in enumerate(cases):
            p, x, skip = case(stage, H, W, 500 + i)
            w = base.pack_weights(stage, p)
            for name, wk in (("_nc", w), ("clamp", flags_cleared(w))):
                with self.subTest(stage=stage, kernels=name):
                    want, got = base.run(stage, wk, x, offset, skip), r.run(stage, wk, x, offset, skip)
                    for k in want:
                        self.assertTrue(same(got[k], want[k]), (stage, k))

    def test_megakernel_bit_identical_to_six_base_launches(self):
        """Both parities, several grid sizes (fewer workgroups than items, and more), clamp-free and clamping kernels;
        the sync words are back to zero after every launch (checked by the runner). Each build against six launches
        of its swin.hip reference build (base; vec16 for bmegahg16)."""
        _, _, ws, x = bneck_case(12, 16, 600)
        for parity in (0, 1):
            want = {ref: runner(ref).run_bneck(ws, x, parity, mega=False) for ref in set(V.MEGA_REFERENCE.values())}
            for name in V.V2_MEGA:
                r = runner(name)
                for g in ((1, 3, 40) if name == "bmegahg" else (2,)):
                    with self.subTest(variant=name, parity=parity, groups=g):
                        got = r.run_bneck(ws, x, parity, mega=True, groups=g)
                        self.assertTrue(same(got, want[V.MEGA_REFERENCE[name]]))
        base = runner("base")
        wc = [flags_cleared(w) for w in ws]
        with self.subTest(kernels="clamp"):
            self.assertTrue(same(runner("bmegahg").run_bneck(wc, x, 0, mega=True, groups=3),
                                 base.run_bneck(wc, x, 0, mega=False)))

    def test_megakernel_matches_reference(self):
        """Against the NumPy reference with the kernels' GELU (gelusig) run block by block: within spec 09 (on these
        synthetic cases even identical, as for the base kernels)."""
        _, params, ws, x = bneck_case(8, 12, 610)
        got = runner("bmegahg").run_bneck(ws, x, 1, mega=True)
        y = x
        for k in range(6):
            y = S.run_stage(f"b{k}", params[k], y, S.window_offset(f"b{k}", 1), gelu="sigmoid")["out"]
        m = spec09_metrics(got, y)
        self.assertEqual(m["nonfinite"], 0)
        self.assertLessEqual(m["frac_over_1_step"], 0.02, m)
        self.assertLessEqual(m["mean_abs_err"], 0.01 * m["ref_rms"], m)

    def test_small_maps(self):
        """One or two window rows (every item depends on every row of the previous block), non-square."""
        for H, W in ((4, 20), (8, 4)):
            base, _, ws, x = bneck_case(H, W, 620 + H)
            for parity in (0, 1):
                with self.subTest(H=H, W=W, parity=parity):
                    self.assertTrue(same(runner("bmegahg").run_bneck(ws, x, parity, mega=True, groups=2),
                                         base.run_bneck(ws, x, parity, mega=False)))


def mid_case(H, W, seed):
    from clean.runner.packing import MID_STAGES
    rng = np.random.default_rng(seed)
    params = {st: S.random_params(st, rng) for st in MID_STAGES}
    x, _ = S.random_inputs("enc2", H, W, rng)
    return params, runner("base").pack_weights("mid", params), x


class TestMid(unittest.TestCase):
    """Persistent kernel swin_mid (stage "mid" = enc2 with its merge, b0..b5, dec2 with its expand)."""

    def test_mid_bit_identical_to_ten_base_launches(self):
        """Both parities, 1 to 40 workgroups, against the ten launches of the base build (and bmegahg16 against
        vec16); the sync words are back to zero after every launch (checked by the runner)."""
        _, w, x = mid_case(16, 24, 700)
        for parity in (0, 1):
            want = runner("base").run_mid(w, x, parity, mega=False)
            for g in (1, 3, 40):
                with self.subTest(parity=parity, groups=g):
                    self.assertTrue(same(runner("bmegahg").run_mid(w, x, parity, groups=g), want))
        with self.subTest(variant="bmegahg16"):
            self.assertTrue(same(runner("bmegahg16").run_mid(w, x, 1, groups=3),
                                 runner("vec16").run_mid(w, x, 1, mega=False)))

    def test_mid_clamping_kernels(self):
        from clean.runner.packing import split_mid
        _, w, x = mid_case(16, 24, 710)
        wc = b"".join(flags_cleared(b) for b in split_mid(w))
        self.assertTrue(same(runner("bmegahg").run_mid(wc, x, 0, groups=3), runner("base").run_mid(wc, x, 0, mega=False)))

    def test_mid_stage_matches_stage_chain(self):
        """run("mid") in a bmega* build launches swin_mid_nc, and gives the result of running enc2, b0..b5 and dec2
        one stage after another (enc2's merged output and skip connection feeding the next stages)."""
        params, w, x = mid_case(16, 24, 720)
        base, r = runner("base"), runner("bmegahg")
        off = S.window_offset("enc2", 1)
        plan = r._prepare("mid", w, x, off, None)
        try:
            self.assertEqual([l[0] for l in plan["launches"]], ["swin_mid_nc"])
        finally:
            r._release(plan)
        o = base.run("enc2", base.pack_weights("enc2", params["enc2"]), x, off)
        y = o["merged"]
        for k in range(6):
            y = base.run(f"b{k}", base.pack_weights(f"b{k}", params[f"b{k}"]), y, S.window_offset(f"b{k}", 1))["out"]
        want = base.run("dec2", base.pack_weights("dec2", params["dec2"]), y, S.window_offset("dec2", 1), o["skip"])
        self.assertTrue(same(r.run("mid", w, x, off)["out"], want["out"]))

    def test_mid_larger_map(self):
        """Several items per window row, rows of different phases depending on each other across resolutions."""
        _, w, x = mid_case(40, 56, 730)
        self.assertTrue(same(runner("bmegahg").run_mid(w, x, 0, groups=7), runner("base").run_mid(w, x, 0, mega=False)))


class TestGeluTable(unittest.TestCase):
    def test_header_matches_generator(self):
        from clean.runner import gelu_table
        self.assertEqual(gelu_table.main(["--check"]), 0)

    def test_table_accuracy(self):
        """Through the tables, q8(GELU(x)) is never more than one FP8 step from the exact value (model of the kernel
        arithmetic: FP16 round toward zero for the index, FP32 subtraction for the correction tables)."""
        import math
        from clean.reference.fp8 import DECODE, fp8_encode
        from clean.runner import gelu_table
        rng = np.random.default_rng(3)
        xs = np.concatenate([np.linspace(-14, 14, 40001), rng.standard_normal(40000) * 2]).astype(np.float32)
        ex = np.array([0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in xs.astype(np.float64)])

        def rtz_bits(v):
            h = v.astype(np.float16)
            h = np.where(np.abs(h.astype(np.float32)) > np.abs(v), np.nextafter(h, np.float16(0)), h)
            return h.view(np.uint16).astype(np.int64)

        for m, limit in ((8, 0.008), (7, 0.012)):
            c = gelu_table.ctable(m).view(np.float16).astype(np.float32)[(rtz_bits(xs) & 0x7FFF) >> (10 - m)]
            y = np.clip(xs, 0, 448) - c
            u = ulp_distance(DECODE[fp8_encode(y.astype(np.float64))], ex)
            self.assertEqual(int(np.sum(u > 1)), 0)
            self.assertLess(float(np.mean(u > 0)), limit, m)
        y = DECODE[gelu_table.table(8)[rtz_bits(xs) >> 2]]
        u = ulp_distance(y, ex)
        self.assertEqual(int(np.sum(u > 1)), 0)
        self.assertLess(float(np.mean(u > 0)), 0.01)


class TestProfile(unittest.TestCase):
    def test_profile_counts_phases(self):
        """v2prof: every wave visits the phases of its kernel the expected number of times."""
        from clean.runner import profile_v2
        rep = profile_v2.profile(v2_build_dir(), "", backend="emu", stages=["b1", "enc1", "dec2"],
                                 sizes={"b1": (12, 16, 0), "enc1": (16, 24, 0), "dec2": (12, 16, 0)})
        b1 = rep["stages"]["b1"]["phases"]
        waves = b1[22]["visits"]                      # norm1: once per wave (streamed C=128 kernel)
        self.assertEqual(b1[27]["visits"], 4 * waves)  # 4 heads
        self.assertEqual(b1[29]["visits"], 8 * waves)  # 8 MLP units of 4 hidden tiles
        enc1 = rep["stages"]["enc1"]["phases"]
        self.assertEqual(enc1[23]["visits"], rep["stages"]["enc1"]["windows"])  # attention: once per window
        self.assertEqual(enc1[20]["visits"], enc1[21]["visits"])  # weight copy: once per wave, one barrier

    def test_profile_bneck(self):
        """bmegaprof: per item, each of its 8 waves claims / waits once, runs one window (4 heads, 8 MLP units) and
        ends the item once. 8x12 map: 2 items per even block, 1 per odd block (8 windows per item)."""
        from clean.runner import profile_v2
        rep = profile_v2.profile(v2_build_dir(), "", backend="emu", variant="bmegaprof", stages=["bneck"],
                                 sizes={"bneck": (8, 12)})
        ph = rep["stages"]["bneck"]["phases"]
        waves = 8 * 3 * (2 + 1)
        self.assertEqual(ph[2]["visits"], waves)
        self.assertEqual(ph[3]["visits"], waves)
        self.assertEqual(ph[22]["visits"], waves)
        self.assertEqual(ph[27]["visits"], 4 * waves)
        self.assertEqual(ph[29]["visits"], 8 * waves)

    def test_profile_mid(self):
        """bmegaprof, stage mid at 16x24: 5 enc2 items (fused merge), 9 b items, 3 dec2 items (fused expand)."""
        from clean.runner import profile_v2
        rep = profile_v2.profile(v2_build_dir(), "", backend="emu", variant="bmegaprof", stages=["mid"],
                                 sizes={"mid": (16, 24)})
        ph = rep["stages"]["mid"]["phases"]
        self.assertEqual(ph[2]["visits"], 8 * 17)
        self.assertEqual(ph[4]["visits"], 8 * 3)
        self.assertEqual(ph[5]["visits"], 8 * 5)


class TestV2Probe(unittest.TestCase):
    def test_conversion_probe(self):
        from clean.runner import selftest
        dev = selftest._Dev("emu", v2_build_dir(), "", emu_library="wg_emu__v2.so")
        res = selftest.probe_v2(dev)
        self.assertTrue(res["pass"], res)


if __name__ == "__main__":
    unittest.main()
