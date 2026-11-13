# SPDX-License-Identifier: MIT
"""Model K reference self-checks (no GPU, no compiler): activation (K2), layout and windows (K3), packing.

Run: python -m unittest discover -s clean/k/tests -t .
"""
import unittest
from fractions import Fraction

import numpy as np

from clean.k.reference import layout as LY
from clean.k.reference import network as N
from clean.k.reference.act import act, act_exact_bits
from clean.k.reference.f16 import all_f16, bits, f16, round_f16_exact
from clean.k.runner import k_pack


class TestActivation(unittest.TestCase):
    def test_exhaustive_against_exact_arithmetic(self):
        """K2: the NumPy oracle equals an exact-rational evaluation of spec 40's formula on all 65536 inputs."""
        got = bits(act(all_f16()))
        ref = np.array([act_exact_bits(b) for b in range(65536)], dtype=np.uint16)
        nan_got = np.isnan(act(all_f16()))
        nan_ref = ((ref & 0x7C00) == 0x7C00) & ((ref & 0x3FF) != 0)
        np.testing.assert_array_equal(nan_got, nan_ref)
        np.testing.assert_array_equal(got[~nan_got], ref[~nan_ref])

    def test_edge_values(self):
        x = np.float16([-np.inf, -3, -2, -0.0, 0.0, 2, 3, np.inf, np.nan])
        y = act(x)
        self.assertTrue(np.isnan(y[0]))  # -inf * 0
        self.assertEqual(bits(y[1]), 0x8000)  # -0
        self.assertEqual(bits(y[2]), 0x8000)
        self.assertEqual(bits(y[3]), 0x8000)
        self.assertEqual(bits(y[4]), 0x0000)
        self.assertEqual(float(y[5]), 2.0)
        self.assertEqual(float(y[6]), 3.0)
        self.assertEqual(float(y[7]), np.inf)
        self.assertTrue(np.isnan(y[8]))

    def test_rounding_helper(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(5000) * np.exp(rng.uniform(-20, 12, 5000))
        with np.errstate(over="ignore"):
            ref = bits(v.astype(np.float16))
        got = np.array([round_f16_exact(Fraction(float(q))) for q in v], dtype=np.uint16)
        np.testing.assert_array_equal(got, ref)


class TestLayout(unittest.TestCase):
    def test_base_size(self):
        self.assertEqual(LY.base_size(1920, 1080, 1.0), (960, 544))
        self.assertEqual(LY.base_size(2560, 1440, 1.0), (1280, 736))
        self.assertEqual(LY.base_size(3840, 2160, 1.0), (1920, 1088))
        self.assertEqual(LY.base_size(640, 360, 1.0), (320, 256))  # spec 44: at least 256
        # s = 0.5, the default (owner's choice)
        self.assertEqual(LY.DEFAULT_S, 0.5)
        self.assertEqual(LY.base_size(2560, 1440), (640, 384))
        self.assertEqual(LY.base_size(1920, 1080), (480, 288))
        self.assertEqual(LY.base_size(3840, 2160), (960, 544))
        self.assertEqual(LY.level_size(1280, 736, 5), (40, 23))

    def test_offsets(self):
        for f in range(16):
            off = LY.stage_offsets(f, LY.SYNTHETIC_OFFSETS)
            self.assertEqual(off["m"], LY.SYNTHETIC_OFFSETS[f % 8])
            for s in ("e0", "e1", "e2", "e3", "e4"):
                self.assertEqual(off[s][0] % 2, 0)  # spec 44: merging stages have even offsets
                self.assertEqual(off[s][1] % 2, 0)
            self.assertEqual(off["e4"], off["d4"])

    def test_mirror(self):
        np.testing.assert_array_equal(LY.border_index([-2, -1, 0, 9, 10, 11], 10), [2, 1, 0, 9, 8, 7])
        np.testing.assert_array_equal(LY.border_index([-7, 13], 8), [7, 1])

    def test_partition_roundtrip(self):
        rng = np.random.default_rng(1)
        for (h, w) in ((8, 8), (13, 21), (9, 10), (3, 5)):
            x = rng.standard_normal((h, w, 4))
            for ox, oy in ((0, 0), (4, 4), (7, 3)):
                for margin in (0, 2):
                    for border in ("mirror", "clamp", "zero"):
                        win = LY.window_partition(x, ox, oy, margin, border)
                        np.testing.assert_array_equal(LY.window_reverse(win, w, h, ox, oy, margin), x)

    def test_partition_values(self):
        x = np.arange(10 * 12).reshape(10, 12, 1).astype(np.float64)
        win = LY.window_partition(x, 4, 4).reshape(2, 2, 8, 8)[0, 0]  # window (0, 0): rows/cols -4..3
        self.assertEqual(win[0, 0], 4 * 12 + 4)  # (-4, -4) -> (4, 4)
        self.assertEqual(win[4, 4], 0)
        self.assertEqual(win[3, 7], 1 * 12 + 3)  # row -1 -> 1, col 3

    def test_buffer_plan(self):
        plan = LY.buffer_plan(1280, 736)
        self.assertEqual(set(plan), {"X0", "A", "B", "S0", "S1", "S2", "S3", "S4", "OUT"})
        self.assertEqual(plan["S0"], 1280 * 736 * 32 * 2)
        self.assertEqual(plan["OUT"], 1280 * 736 * 40 * 2)

    def test_linear_shapes_cover_stages(self):
        shapes = LY.linear_shapes()
        self.assertIn(("plain", 16, 32), shapes)
        self.assertIn(("plain", 32, 40), shapes)
        self.assertIn(("plain", 160, 768), shapes)
        self.assertEqual(len([s for s in shapes if s[0] == "merge"]), 5)
        self.assertEqual(len([s for s in shapes if s[0] == "expand"]), 5)


class TestPacking(unittest.TestCase):
    def test_tile_roundtrip(self):
        rng = np.random.default_rng(2)
        W = f16(rng.standard_normal((48, 32)))
        for d_row in ("gfx12", [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]):
            for order in ("nk", "kn"):
                t = k_pack.pack_tiles(W, d_row, order)
                np.testing.assert_array_equal(k_pack.unpack_tiles(t, d_row, order), W)

    def test_tile_layout(self):
        W = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 2048
        t = k_pack.pack_tiles(W)  # [1, 2, 32, 8]
        self.assertEqual(t[0, 1, 17, 3], W[1, 16 + 8 + 3])  # lane 17: row 1, inputs 16 kt + 8 + j

    def test_stage_blob_sizes(self):
        P = N.random_network_params(0)
        for st in LY.STAGES:
            blob = k_pack.pack_stage(st["name"], P[st["name"]])
            self.assertEqual(len(blob), k_pack.stage_sections(st)["_size"])

    def test_head_padding(self):
        rng = np.random.default_rng(3)
        W = f16(rng.standard_normal((40, 32)))
        blob = k_pack.pack_linear(W, f16(rng.standard_normal(40)))
        self.assertEqual(len(blob), k_pack.linear_size(40, 32))
        t = np.frombuffer(blob[:48 * 32 * 2], np.float16).reshape(3, 2, 32, 8)
        np.testing.assert_array_equal(k_pack.unpack_tiles(t)[:40], W)
        self.assertFalse(np.any(k_pack.unpack_tiles(t)[40:]))

    def test_rejects_non_fp16(self):
        with self.assertRaises(ValueError):
            k_pack.pack_linear(np.full((16, 16), 0.1, np.float32))


class TestSpec48Tensors(unittest.TestCase):
    def converter_form(self, stage, p):
        """Canonical tensors -> the converter's shapes of spec 48 section 7.2."""
        st = LY.STAGE[stage]
        q = dict(p)
        for nm in ("B", "M"):
            if nm in q:
                q[nm] = q[nm].reshape(st["heads"], 8, 8, 8, 8)
        if "Wm" in q:
            q["Wm"] = q["Wm"].reshape(st["C_next"], 4, st["C"])
        if "Wx" in q:
            q["Wx"] = q["Wx"].reshape(4, st["C"], st["C_prev"])
            q["bx"] = q["bx"].reshape(4, st["C"])
        return q

    def test_pack_accepts_converter_shapes(self):
        P = N.random_network_params(1)
        for st in LY.STAGES:
            p = P[st["name"]]
            self.assertEqual(k_pack.pack_stage(st["name"], self.converter_form(st["name"], p)),
                             k_pack.pack_stage(st["name"], p))

    def test_map_shapes(self):
        from clean.k.reference.params import expected_shapes
        from clean.k.runner.k_model import map_tensors
        groups = map_tensors()
        self.assertEqual(groups["net"], [("window_offsets", (8, 2))])  # spec 48 draft 3 (int32)
        for stage, tensors in groups.items():
            if stage == "net":
                continue
            want = expected_shapes(stage)
            self.assertEqual({n for n, _ in tensors}, set(want), stage)
            for n, shape in tensors:
                self.assertEqual(int(np.prod(shape)), int(np.prod(want[n])), (stage, n))

    def test_raw_roundtrip(self):
        import tempfile
        from clean.k.runner.k_model import load_raw, write_raw
        P = N.random_network_params(2)
        d = tempfile.mkdtemp(prefix="wg_k_raw_")
        write_raw(d, P)
        Q = load_raw(d)
        np.testing.assert_array_equal(Q["net"]["window_offsets"], P["net"]["window_offsets"])
        for st in LY.STAGES:
            self.assertEqual(k_pack.pack_stage(st["name"], Q[st["name"]]), k_pack.pack_stage(st["name"], P[st["name"]]))


class TestSpec41Draft3(unittest.TestCase):
    def test_norm_edge_cases(self):
        x = np.zeros((3, 64), np.float16)
        x[1] = 50.0  # sum of squares 160000: FP16 overflow -> n = 0
        x[2, 0] = 1.0
        n = N.norm(x, np.ones(64, np.float16))
        self.assertTrue(np.all(n[0] == 0) and np.all(n[1] == 0))
        self.assertEqual(float(n[2, 0]), 1.0)

    def test_threshold_windows(self):
        """Spec 49 draft 3: a window with a token whose exact sum of squares is near 65504..65520 is flagged."""
        w = np.zeros((3, 64, 16), np.float16)
        w[0, 5, 0] = 255.875  # 65472.0: inside [0.995 x 65504, 1.005 x 65520]
        w[1, 0, :] = 10.0     # 1600: far below
        w[2, :, 0] = 300.0    # 90000: far above (plain overflow, tested normally)
        np.testing.assert_array_equal(LY.threshold_windows(w), [True, False, False])
        m = LY.window_flags_to_map(np.array([[True, False], [False, False]]), 12, 10, 4, 4)
        self.assertEqual(m.shape, (10, 12))
        self.assertTrue(m[:4, :4].all() and not m[4:, :].any() and not m[:, 4:].any())

    def test_decoder_border_example(self):
        """Spec 41 draft 3: row -1 takes the expanded part from coarse row 1 (sub-row 1) and the skip from row 1."""
        rng = np.random.default_rng(5)
        C, Cp = 16, 16
        y = f16(rng.standard_normal((8, 8, Cp)))
        skip = f16(rng.standard_normal((16, 16, C)))
        Wx = f16(rng.standard_normal((4 * C, Cp)) / 4)
        bx = f16(rng.standard_normal(4 * C))
        win = N.decoder_windows(y, skip, Wx, bx, (4, 4)).reshape(3, 3, 8, 8, C)
        from clean.k.reference.linear import matmul_f16
        e = matmul_f16(y, Wx, bx).reshape(8, 8, 4, C)
        # window (0, 0): member (row 3, col 4) is map position (-1, 0): coarse (1, 0), k = 2 * 1 + 0
        want = (e[1, 0, 2] + skip[1, 0]).astype(np.float16)
        np.testing.assert_array_equal(win[0, 0, 3, 4], want)
        # member (row 0, col 0) is (-4, -4): coarse (m'(-2), m'(-2)) = (2, 2), k = 0, skip (4, 4)
        np.testing.assert_array_equal(win[0, 0, 0, 0], (e[2, 2, 0] + skip[4, 4]).astype(np.float16))
        # inside the map the windows hold the plain expand + skip
        from clean.k.reference.linear import expand
        inside = expand(y, Wx, bx, (16, 16), skip=skip)
        np.testing.assert_array_equal(N.decoder_input(y, skip, Wx, bx), inside)


class TestNetwork(unittest.TestCase):
    def test_runs_finite(self):
        P = N.random_network_params(0)
        outs = N.run_network(N.random_input(64, 48), P, f=3)
        self.assertEqual(outs["d0"]["head"].shape, (48, 64, 40))
        for o in outs.values():
            for a in o.values():
                self.assertTrue(np.all(np.isfinite(a)))

    def test_softmax_rows_sum_to_one(self):
        S = f16(np.random.default_rng(4).standard_normal((5, 64)) * 10)
        A = N.softmax_weights(S).astype(np.float64)
        np.testing.assert_allclose(A.sum(-1), 1.0, atol=64 * 2.0 ** -11)


if __name__ == "__main__":
    unittest.main()
