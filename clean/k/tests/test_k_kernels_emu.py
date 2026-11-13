# SPDX-License-Identifier: MIT
"""Model K kernels in the host lane emulator, through the spec 90 Runner (model="K"), against the NumPy reference.

Builds a model K-only emulator library (clean/k/runner/k_build.py) into a temporary directory and opens it as the
Runner's emulator backend. Needs Linux and a host C++ compiler (clang++ or g++, env CXX).
Run: python -m unittest discover -s clean/k/tests -t .

Acceptance as spec 49 (relative to the RMS of the reference output): mean |error| <= 2.5e-3, max |error| <= 3e-2,
no non-finite values. The emulator sums each WMMA in double precision, so these runs mostly check layouts, addressing,
borders and the order of the FP16 steps; the FP32 accumulation of the GPU is measured by the spec team.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clean.k.reference import linear as L
from clean.k.reference import network as N
from clean.k.reference.layout import STAGES, cdiv, window_partition
from clean.k.reference.f16 import f16
from clean.k.runner import k_build

_BUILD = None


def build_dir():
    global _BUILD
    if _BUILD is None:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux (ucontext)")
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("no host C++ compiler")
        d = tempfile.mkdtemp(prefix="wg_emu_k_")
        r = subprocess.run(k_build.emu_command(cxx, os.path.join(d, "wg_emu.so")), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("emulator build failed:\n" + r.stderr)
        _BUILD = d
    return _BUILD


def runner():
    """The emulator Runner; with WG_K_GPU_BUILD=<kernel build dir> (and ROCM_PATH) the same tests run on the GPU
    instead (any OS; run them under the machine's GPU lock)."""
    from clean.runner.stage_runner import Runner
    gpu = os.environ.get("WG_K_GPU_BUILD")
    if gpu:
        return Runner(gpu, os.environ.get("ROCM_PATH", ""), backend="hip", model="K")
    return Runner(build_dir(), "", backend="emu", model="K")


def device():
    """(device, KFuncs) for the k_bench / k_frame harness: the emulator, or the GPU with WG_K_GPU_BUILD."""
    from clean.k.runner.k_bench import KFuncs, open_device
    gpu = os.environ.get("WG_K_GPU_BUILD")
    backend, build = ("hip", gpu) if gpu else ("emu", build_dir())
    dev = open_device(backend, build, os.environ.get("ROCM_PATH", ""))
    return dev, KFuncs(dev, build, backend)


class Checks(unittest.TestCase):
    def assert_close(self, got, ref, what, mean_tol=2.5e-3, max_tol=3e-2):
        got = np.asarray(got, np.float64)
        ref = np.asarray(ref, np.float64)
        self.assertEqual(got.shape, ref.shape, what)
        self.assertTrue(np.all(np.isfinite(got)), f"{what}: non-finite outputs")
        rms = np.sqrt(np.mean(ref ** 2))
        d = np.abs(got - ref)
        self.assertLessEqual(d.mean() / rms, mean_tol, what)
        self.assertLessEqual(d.max() / rms, max_tol, what)


class TestActivationEmu(unittest.TestCase):
    def test_exhaustive(self):
        out = runner().run("act", b"", None)
        self.assertEqual(out["mismatch_scalar"], 0)
        self.assertEqual(out["mismatch_packed"], 0)


class TestLinearEmu(Checks):
    CASES = [("plain", 16, 32, (9, 7), dict(relu=True)), ("plain", 32, 40, (5, 6), {}),
             ("plain", 64, 192, (11, 3), dict(act=True)), ("plain", 640, 160, (4, 5), {}),
             ("merge", 128, 64, (9, 7), {}), ("merge", 512, 160, (4, 6), {}),
             ("expand", 64, 128, (5, 4), {}), ("expand", 160, 512, (3, 3), {})]

    def test_shapes(self):
        r = runner()
        rng = np.random.default_rng(0)
        for variant in ("", "@g1"):
            for mode, k, n, (w, h), opts in self.CASES:
                W = f16(rng.standard_normal((n, k)) / np.sqrt(k))
                b = f16(rng.standard_normal(n) * 0.3)
                stage = f"lin:{mode}:{k}x{n}{variant}"
                blob = r.pack_weights(stage, {"W": W, "b": b})
                cin = k // 4 if mode == "merge" else k
                x = f16(rng.standard_normal((h, w, cin)))
                if mode == "plain":
                    got = r.run(stage, blob, {"x": x, **opts})["out"]
                    ref = L.linear(x, W, b, relu=opts.get("relu", False), use_act=opts.get("act", False))
                elif mode == "merge":
                    got, ref = r.run(stage, blob, x)["out"], L.merge(x, W, b)
                else:
                    size = (2 * w - 1, 2 * h)  # odd output width: the last column of expand outputs is dropped
                    skip = f16(rng.standard_normal((size[1], size[0], n // 4)))
                    got = r.run(stage, blob, {"x": x, "out_size": size}, skip=skip)["out"]
                    ref = L.expand(x, W, b, size, skip=skip)
                self.assert_close(got, ref, stage, mean_tol=1e-4, max_tol=2e-3)


class TestWindowEmu(unittest.TestCase):
    def test_gather_scatter(self):
        r = runner()
        x = f16(np.random.default_rng(1).standard_normal((13, 21, 16)))
        for offset, margin, border in (((4, 4), 0, "mirror"), ((7, 1), 2, "mirror"), ((3, 6), 1, "clamp"),
                                       ((0, 0), 1, "zero")):
            out = r.run("win", b"", {"x": x, "offset": offset, "margin": margin, "border": border})
            ref = window_partition(x, *offset, margin, border)
            np.testing.assert_array_equal(out["windows"], ref.astype(np.float32))
            np.testing.assert_array_equal(out["back"], x.astype(np.float32))


class TestStagesEmu(Checks):
    """Every stage of spec 41 on small odd maps (border rule) with random spec-44-style offsets."""
    SIZES = {0: (21, 13), 1: (11, 7), 2: (9, 8), 3: (8, 9), 4: (8, 8), 5: (8, 8)}

    def run_stages(self, rng, offset_of, suffix=""):
        r = runner()
        P = N.random_network_params(5)
        for st in STAGES:
            name, p, lv = st["name"], P[st["name"]], self.SIZES[st["level"]]
            off = offset_of(st)
            blob = r.pack_weights(name, p)
            if st["kind"] == "dec":
                x = f16(rng.standard_normal((cdiv(lv[1], 2), cdiv(lv[0], 2), st["C_prev"])) * 2)
                skip = f16(rng.standard_normal((lv[1], lv[0], st["C"])))
                ref, got = N.run_stage(st, p, x, off, skip, lv), r.run(name + suffix, blob, x, off, skip=skip)
            else:
                cin = 16 if st.get("embed") else st["C"]
                x = f16(rng.standard_normal((lv[1], lv[0], cin)) * 2)
                ref, got = N.run_stage(st, p, x, off), r.run(name + suffix, blob, x, off)
            self.assertEqual(set(got), set(ref), name)
            for k in ref:
                self.assert_close(got[k], ref[k], f"{name}{suffix} {k} offset {off}")

    def test_all_stages_fused(self):
        """Even offsets (as spec 44 gives): one fused kernel per stage."""
        rng = np.random.default_rng(3)
        self.run_stages(rng, lambda st: (2 * int(rng.integers(0, 4)), 2 * int(rng.integers(0, 4))))

    def test_all_stages_fused16(self):
        """The FP16-accumulating fused kernels (k_stageh_*, spec 49 speed approximation): within spec 49's per-stage
        limits."""
        rng = np.random.default_rng(3)
        self.run_stages(rng, lambda st: (2 * int(rng.integers(0, 4)), 2 * int(rng.integers(0, 4))), "@f16")

    def test_two_windows_per_workgroup(self):
        """e0 and d0 with two windows per workgroup (also an odd window count: the last group has one)."""
        r = runner()
        rng = np.random.default_rng(8)
        P = N.random_network_params(5)
        for name in ("e0", "d0"):
            st = [s for s in STAGES if s["name"] == name][0]
            p, lv = P[name], self.SIZES[0]
            blob = r.pack_weights(name, p)
            for suffix in ("@w2", "@w2@f16"):
                if name == "d0":
                    x = f16(rng.standard_normal((cdiv(lv[1], 2), cdiv(lv[0], 2), 64)) * 2)
                    skip = f16(rng.standard_normal((lv[1], lv[0], 32)))
                    ref = N.run_stage(st, p, x, (2, 6), skip, lv)
                    got = r.run(name + suffix, blob, x, (2, 6), skip=skip)
                    one = r.run(name + suffix.replace("@w2", ""), blob, x, (2, 6), skip=skip) if "f16" in suffix else None
                else:
                    x = f16(rng.standard_normal((lv[1], lv[0], 16)) * 2)
                    ref, got = N.run_stage(st, p, x, (2, 6)), r.run(name + suffix, blob, x, (2, 6))
                    one = r.run(name + suffix.replace("@w2", ""), blob, x, (2, 6)) if "f16" in suffix else None
                for k in ref:
                    self.assert_close(got[k], ref[k], f"{name}{suffix} {k}")
                    if one is not None:  # the same FP16 accumulator chains as one window per workgroup
                        np.testing.assert_array_equal(got[k], one[k], f"{name}{suffix} {k} vs {name}@f16")

    def test_wide_stages(self):
        """The wide kernels (k_block_wide.hip) of levels 3..5 with NG = 2 and 4: within the limits, and bit for bit
        the fused kernel's results (same FP32 accumulation order)."""
        from clean.k.runner.k_launch import WIDE_BUILDS
        r = runner()
        rng = np.random.default_rng(11)
        P = N.random_network_params(5)
        for st in STAGES:
            name = st["name"]
            if name not in WIDE_BUILDS:
                continue
            p, lv = P[name], self.SIZES[st["level"]]
            blob = r.pack_weights(name, p)
            off = (2 * int(rng.integers(0, 4)), 2 * int(rng.integers(0, 4)))
            if st["kind"] == "dec":
                x = f16(rng.standard_normal((cdiv(lv[1], 2), cdiv(lv[0], 2), st["C_prev"])) * 2)
                skip = f16(rng.standard_normal((lv[1], lv[0], st["C"])))
            else:
                x, skip = f16(rng.standard_normal((lv[1], lv[0], st["C"])) * 2), None
            ref = N.run_stage(st, p, x, off, skip, lv) if skip is not None else N.run_stage(st, p, x, off)
            for f in ("", "@f16"):  # FP32 and FP16 accumulation: each bit for bit its fused kernel
                fused = r.run(name + f, blob, x, off, skip=skip)
                for ng in WIDE_BUILDS[name]:
                    got = r.run(f"{name}@wide{ng}{f}", blob, x, off, skip=skip)
                    self.assertEqual(set(got), set(ref), name)
                    for k in ref:
                        self.assert_close(got[k], ref[k], f"{name}@wide{ng}{f} {k} offset {off}")
                        np.testing.assert_array_equal(got[k], fused[k], f"{name}@wide{ng}{f} {k} vs {name}{f}")

    def test_all_stages_split(self):
        """The separate kernels; decoder stages also with odd offsets (the padded expand)."""
        rng = np.random.default_rng(7)

        def off(st):
            if st["kind"] == "enc":
                return 2 * int(rng.integers(0, 4)), 2 * int(rng.integers(0, 4))
            return int(rng.integers(0, 8)), int(rng.integers(0, 8))
        self.run_stages(rng, off, "@split")

    def test_norm_edge_cases(self):
        """Spec 41 draft 2: an all-zero token and a token whose sum of squares overflows FP16 both get n = 0."""
        r = runner()
        rng = np.random.default_rng(6)
        st = [s for s in STAGES if s["name"] == "e2"][0]
        p = N.random_stage_params(st, rng)
        x = f16(rng.standard_normal((16, 16, 64)))
        x[3, 5] = 0.0
        x[9, 9] = 50.0
        x[0, 15] = -60.0
        ref = N.run_stage(st, p, x, (4, 4))
        got = r.run("e2", r.pack_weights("e2", p), x, (4, 4))
        for k in ref:
            self.assert_close(got[k], ref[k], f"e2 {k} with zero / overflow tokens")

    def test_block_entry_and_g1(self):
        r = runner()
        rng = np.random.default_rng(4)
        st = [s for s in STAGES if s["name"] == "e2"][0]
        p = N.random_stage_params(st, rng)
        x = f16(rng.standard_normal((10, 12, 64)))
        blob = r.pack_weights("block:64:2:softmax", p)
        ref = N.block(x, p, "softmax", 2, (3, 5))
        self.assert_close(r.run("block:64:2:softmax", blob, x, (3, 5))["out"], ref, "block entry")
        got = r.run("e2@g1", r.pack_weights("e2", p), x, (2, 4))
        self.assert_close(got["merged"], N.run_stage(st, p, x, (2, 4))["merged"], "e2@g1 merged")


class TestNetworkEmu(unittest.TestCase):
    """The whole chained network (KNet, the timing harness's wiring) vs the chained reference, spec 49 draft 2's
    whole-network rule, on a small odd base map (levels down to 2 x 1: repeated mirror reflections)."""

    def test_chain(self):
        from clean.k.runner import k_bench
        dev, fn = device()
        for variant in ("fused", "fused2", "fused1", "auto", "wide2", "wide4", "g2", "fused16", "fused2@f16",
                        "fused1@f16", "wide2@f16", "wide4@f16"):
            r = k_bench.check(dev, fn, None, 1.0, variant, 5, base=(37, 29))
            bad = {k: v for k, v in r["stages"].items() if v["nonfinite"] or v["mean_rel"] > k_bench.WHOLE_MEAN or
                   v["max_rel"] > k_bench.WHOLE_MAX}
            self.assertTrue(r["pass"], (variant, bad))


if __name__ == "__main__":
    unittest.main()
