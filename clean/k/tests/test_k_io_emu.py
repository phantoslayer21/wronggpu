# SPDX-License-Identifier: MIT
"""Model K input stage (spec 42) and output stage (spec 43): emulated kernels vs the NumPy references, bit for bit,
all eight flag variants, on frames built to reach the rare branches (very bright pixels: the encode branch Y >= 1000;
luma history above 0.999: the decode branch; resets; history inside and outside the frame; blend logits beyond +-3;
every filter-table selector). Also the filter-shape table formula and the exact arithmetic helpers.

Run: python -m unittest discover -s clean/k/tests -t .
"""
import unittest

import numpy as np

from clean.k.reference import fp
from clean.k.reference.instage import input_stage
from clean.k.reference.layout import base_size
from clean.k.reference.outstage import ceil16, filter_level, filter_table, output_stage
from clean.k.tests.test_k_kernels_emu import runner

W, H, RW, RH = 40, 24, 27, 16


def frame(rng, hdr, mvlo, inv, reset=False, base=None):
    Bw, Bh = base or base_size(W, H)
    W16, H16 = ceil16(W), ceil16(H)
    color = rng.random((RH, RW, 4)).astype(np.float32) * (4 if hdr else 1)
    if hdr:
        bright = rng.random((RH, RW)) < 0.08
        color[bright, :3] *= 3000.0  # kappa Y >= 1000: the encode branch
    color[rng.random((RH, RW)) < 0.03, 1] = -0.5  # negative channels (max(RGB, 0))
    mshape = (RH, RW, 2) if mvlo else (H, W, 2)
    motion = (rng.standard_normal(mshape) * 0.8).astype(np.float32)
    motion[rng.random(mshape[:2]) < 0.1] *= 40.0  # some history from outside the frame
    N = (rng.standard_normal((Bh, Bw, 40)) * 1.2).astype(np.float16)
    N[..., 9::10] = (rng.standard_normal((Bh, Bw, 4)) * 5).astype(np.float16)  # blend logits beyond +-3 (t = n/2)
    L = (rng.random((2 * H16, 2 * W16)) * 1.05).astype(np.float16)  # some above 0.999: the decode branch
    Hc = (rng.random((H16, W16, 4)) * 0.6).astype(np.float16)
    Hh = (rng.standard_normal((Bh, Bw, 4)) * 0.3).astype(np.float16)
    return dict(N=N, color=color, motion=motion, depth=rng.random((RH, RW)).astype(np.float32), render_size=(RW, RH),
                output_size=(W, H), hdr=hdr, mv_lowres=mvlo, depth_inverted=inv, jitter=(0.13, -0.27),
                mv_scale=(1.0, 0.9), mv_offset=(0.1, 0.0), reset=reset, E_cur=1.5, E_prev=1.25, E_aux=1.0, e1=1.0,
                e2=1.1, pre_exposure=0.8, L=L, Hc=Hc, Hh=Hh)


def same_bits(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


class TestHelpers(unittest.TestCase):
    def test_fma_exact(self):
        from fractions import Fraction
        rng = np.random.default_rng(0)
        a = (rng.standard_normal(3000) * 1e3).astype(np.float32)
        b = rng.standard_normal(3000).astype(np.float32)
        c = (-(a.astype(np.float64) * b) * (1 + rng.standard_normal(3000) * 1e-7)).astype(np.float32)
        got = fp.fma32(a, b, c)
        for i in range(0, 3000, 7):
            q = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
            x = np.float32(float(q))
            cands = [np.nextafter(x, np.float32(-np.inf)), x, np.nextafter(x, np.float32(np.inf))]
            best = min(cands, key=lambda v: (abs(Fraction(float(v)) - q), int(np.float32(v).view(np.uint32)) & 1))
            self.assertEqual(got[i], best)

    def test_round_half_away_and_trunc16(self):
        np.testing.assert_array_equal(fp.R(np.float32([0.49999997, 0.5, -0.5, 2.5, -2.4999998])), [0, 1, -1, 3, -2])
        t = fp.trunc16(np.float32([1.0009766 * 1.0004, -2.3, 70000.0, 65519.0, 1e-8]))
        self.assertEqual(float(t[0]), 1.0009765625)
        self.assertEqual(float(t[1]), -2.298828125)
        self.assertTrue(np.isinf(t[2]))
        self.assertTrue(np.isinf(t[3]) and t[3] > 0)  # spec 43 d2 7.8: magnitude above 65504 -> infinity
        self.assertEqual(float(t[4]), 0.0)


class TestFilterTable(unittest.TestCase):
    def test_levels(self):
        # 7 (sqrt(1 / ((r/2)(r/2))) - 1) / 3 with r = render / output: native 7/3 -> 2, 2x upscale 7 -> 7,
        # 1.5x (Quality) 14/3 -> 5
        self.assertEqual(filter_level(2560, 1440, 2560, 1440), 2)
        self.assertEqual(filter_level(1280, 720, 2560, 1440), 7)
        self.assertEqual(filter_level(1706, 960, 2560, 1440), 5)
        lv = [filter_level(int(2560 / s), int(1440 / s), 2560, 1440) for s in (1.0, 1.5, 1.7, 2.0, 3.0)]
        self.assertEqual(lv, sorted(lv))
        self.assertTrue(all(0 <= v <= 7 for v in lv))

    def test_table_shape_and_flags(self):
        t = filter_table(3)
        self.assertEqual(t.shape, (2048, 8))
        o = t[:, 4].view(np.uint16)
        self.assertTrue(set(np.unique(o)) <= {0, 1})
        self.assertTrue(np.all(t[:, 7] == 0))
        self.assertTrue(np.all(np.isfinite(t)))


class TestOutputStageEmu(unittest.TestCase):
    def test_bit_exact_all_variants(self):
        r = runner()
        rng = np.random.default_rng(11)
        for hdr in (True, False):
            for mvlo in (True, False):
                for inv in (True, False):
                    for reset in (False, True):
                        if reset and not (hdr and mvlo):
                            continue
                        x = frame(rng, hdr, mvlo, inv, reset)
                        ref = output_stage(x)
                        got = r.run("output", b"", x)
                        for k in ("out", "L", "Hc", "Hh"):
                            ok = same_bits(got[k], ref[k])
                            self.assertTrue(ok.all(), f"hdr={hdr} mvlo={mvlo} inv={inv} reset={reset} {k}: "
                                                      f"{int((~ok).sum())} values differ")


class TestInputStageEmu(unittest.TestCase):
    def test_bit_exact_all_variants(self):
        r = runner()
        rng = np.random.default_rng(12)
        for hdr in (True, False):
            for mvlo in (True, False):
                for inv in (True, False):
                    x = frame(rng, hdr, mvlo, inv, reset=(hdr and not mvlo and inv))
                    ref = input_stage(x)
                    got = r.run("input@exact", b"", x)["x0"]
                    ok = same_bits(got.astype(np.float16), ref)
                    self.assertTrue(ok.all(), f"hdr={hdr} mvlo={mvlo} inv={inv}: {int((~ok).sum())} values differ")


class TestFp32Functions(unittest.TestCase):
    """GPU only (WG_K_GPU_BUILD), exhaustive: the FP32 double-word kf_log2 / kf_ln (every normal positive binary32
    input), kf_exp2 (every input in [-125, 127)) and kf_rsqrt (every input >= 1) equal their binary64 path bit for bit (k_fp_probe, k_act.co).
    About two seconds."""

    def test_exhaustive(self):
        import ctypes
        import os
        import struct
        if not os.environ.get("WG_K_GPU_BUILD"):
            self.skipTest("GPU only: set WG_K_GPU_BUILD")
        from clean.k.tests.test_k_kernels_emu import device

        class KFpProbe(ctypes.Structure):
            _fields_ = [("base", ctypes.c_uint32), ("count", ctypes.c_uint32), ("fn", ctypes.c_uint32),
                        ("pad", ctypes.c_uint32), ("out", ctypes.c_void_p)]

        dev, fn = device()
        k = fn("k_fp_probe")
        out = dev.malloc(256)
        bits = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]  # noqa: E731
        ranges = {0: [(0x00800000, 0x7F800000)], 1: [(0x00800000, 0x7F800000)],
                  2: [(0, bits(127.0)), (0x80000000, 0x80000000 + bits(125.0) + 1)], 3: [(bits(1.0), 0x7F800000)]}
        try:
            for f, rs in ranges.items():
                bad, examples = 0, []
                for lo, hi in rs:
                    for b in range(lo, hi, 1 << 24):
                        n = min(1 << 24, hi - b)
                        dev.memset(out, 0, 256)
                        dev.launch(k, (n + 255) // 256, 256, [KFpProbe(base=b, count=n, fn=f, pad=0, out=out)])
                        dev.sync()
                        r = struct.unpack("<64I", dev.download(out, 256))
                        bad += r[0]
                        examples += [tuple(hex(v) for v in r[4 + 3 * i: 7 + 3 * i]) for i in range(min(r[2], 4))]
                self.assertEqual(bad, 0, f"function {('log2', 'ln', 'exp2', 'rsqrt')[f]}: {bad} differences, e.g. {examples}")
        finally:
            dev.free(out)


class TestScaleFactor(unittest.TestCase):
    """Spec 44's base-size factor s (draft 3: 1/2) flows through both stages: s = 0.5, 1, 0.75 and 1.5 against the
    references, bit for bit (input stage: the exact build)."""

    def test_s(self):
        r = runner()
        rng = np.random.default_rng(31)
        for s in (0.5, 1.0, 0.75, 1.5):  # 0.5: spec 44 draft 3 (also the default); 1: drafts before
            for hdr, mvlo in ((True, True), (False, False)):
                x = frame(rng, hdr, mvlo, True, base=base_size(W, H, s))
                x["s"] = s
                ref, got = output_stage(x), r.run("output", b"", x)
                for k in ("out", "L", "Hc", "Hh"):
                    ok = same_bits(got[k], ref[k])
                    self.assertTrue(ok.all(), f"s={s} hdr={hdr} {k}: {int((~ok).sum())} values differ")
                ok = same_bits(r.run("input@exact", b"", x)["x0"].astype(np.float16), input_stage(x))
                self.assertTrue(ok.all(), f"s={s} hdr={hdr} input: {int((~ok).sum())} values differ")


def fp16_steps(a, b):
    """|a - b| in FP16 steps (distance of the ordered bit patterns), for float16 arrays."""
    def ordered(x):
        u = np.asarray(x, np.float16).view(np.uint16).astype(np.int32)
        return np.where(u & 0x8000, 0x8000 - (u & 0x7FFF), 0x8000 + u)
    return np.abs(ordered(a) - ordered(b))


def fast_input_stats(got, ref):
    """42-T1 measures of the fast input stage vs the reference: fraction within one FP16 step, max steps, non-finite."""
    g = np.asarray(got, np.float32).astype(np.float16)
    r = np.asarray(ref, np.float16)
    both_nan = np.isnan(g) & np.isnan(r)
    st = np.where(both_nan, 0, fp16_steps(g, r))
    return {"within1": float(np.mean(st <= 1)), "max_steps": int(st.max()), "nonfinite": int((~np.isfinite(g)).sum()),
            "exact": float(np.mean(st == 0))}


class TestInputFastGPU(unittest.TestCase):
    """GPU only: the default (fast) input stage against the reference with spec 42 section 9's acceptance (42-T1):
    every value finite, >= 99.9 % within one FP16 step (the rest only at decision points), all eight variants with
    steady and reset frames, and at 2560 x 1440 output."""

    def test_acceptance(self):
        import os
        if not os.environ.get("WG_K_GPU_BUILD"):
            self.skipTest("GPU only: set WG_K_GPU_BUILD")
        global W, H, RW, RH
        r = runner()
        rng = np.random.default_rng(21)
        worst = None
        for size in ((40, 24, 27, 16), (2560, 1440, 1707, 960)):
            saved = (W, H, RW, RH)
            W, H, RW, RH = size
            try:
                cases = [(h_, m_, i_, rs) for h_ in (True, False) for m_ in (True, False) for i_ in (True, False)
                         for rs in (False, True)] if size[0] == 40 else [(True, True, True, False),
                                                                          (False, False, False, False)]
                for hdr, mvlo, inv, reset in cases:
                    x = frame(rng, hdr, mvlo, inv, reset)
                    s = fast_input_stats(r.run("input", b"", x)["x0"], input_stage(x))
                    print(f"fast input {size[0]}x{size[1]} hdr={hdr} mvlo={mvlo} inv={inv} reset={reset}: "
                          f"within 1 step {100 * s['within1']:.4f} %, exact {100 * s['exact']:.2f} %, "
                          f"max {s['max_steps']} steps, non-finite {s['nonfinite']}", flush=True)
                    self.assertEqual(s["nonfinite"], 0)
                    self.assertGreaterEqual(s["within1"], 0.999, (size, hdr, mvlo, inv, reset, s))
                    worst = s if worst is None or s["within1"] < worst["within1"] else worst
            finally:
                W, H, RW, RH = saved


class TestExp2Half(unittest.TestCase):
    """GPU only: the output stage's 2^E (v_exp_f16) equals h(2^E) (spec 43 5.4) for all 65536 FP16 inputs."""

    def test_all_inputs(self):
        import ctypes
        import os
        if not os.environ.get("WG_K_GPU_BUILD"):
            self.skipTest("GPU only: set WG_K_GPU_BUILD")
        from clean.k.runner.k_io import exp2_table
        from clean.k.tests.test_k_kernels_emu import device
        dev, fn = device()
        out = dev.malloc(65536 * 2)
        try:
            dev.launch(fn("k_exp2h_probe"), 256, 256, [ctypes.c_void_p(out)])
            dev.sync()
            got = np.frombuffer(dev.download(out, 65536 * 2), np.uint16)
        finally:
            dev.free(out)
        ref = exp2_table().view(np.uint16)
        nan = lambda a: ((a & 0x7C00) == 0x7C00) & ((a & 0x3FF) != 0)  # noqa: E731
        bad = (got != ref) & ~(nan(got) & nan(ref))
        self.assertEqual(int(bad.sum()), 0, f"differ at {np.flatnonzero(bad)[:8]}")


class TestRealSizeGPU(unittest.TestCase):
    """GPU only (WG_K_GPU_BUILD): both stages bit for bit at 2560 x 1440 output (render 1707 x 960). Rare rounding
    cases need many pixels: a compiler fusion of (h16)(a * b) into one rounding (kh_round, k_fp.h) changed 46 of
    3.7 M colour-history pixels and never showed at the small size above. About a minute per variant (the NumPy
    references)."""

    def test_1440p(self):
        import os
        if not os.environ.get("WG_K_GPU_BUILD"):
            self.skipTest("GPU only: set WG_K_GPU_BUILD")
        global W, H, RW, RH
        saved = (W, H, RW, RH)
        W, H, RW, RH = 2560, 1440, 1707, 960
        try:
            r = runner()
            rng = np.random.default_rng(5)
            for hdr, mvlo, inv, reset in ((True, True, True, False), (False, False, False, False)):
                x = frame(rng, hdr, mvlo, inv, reset)
                ref, got = output_stage(x), r.run("output", b"", x)
                for k in ("out", "L", "Hc", "Hh"):
                    ok = same_bits(got[k], ref[k])
                    self.assertTrue(ok.all(), f"hdr={hdr} mvlo={mvlo} {k}: {int((~ok).sum())} values differ")
                ok = same_bits(r.run("input@exact", b"", x)["x0"].astype(np.float16), input_stage(x))
                self.assertTrue(ok.all(), f"input hdr={hdr} mvlo={mvlo}: {int((~ok).sum())} values differ")
        finally:
            W, H, RW, RH = saved


if __name__ == "__main__":
    unittest.main()
