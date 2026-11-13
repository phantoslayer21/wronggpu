# SPDX-License-Identifier: MIT
"""Checks of the NumPy reference against slow, independently written loop versions of spec 02.

Run: python -m unittest discover -s clean/tests -t .
"""
import math
import unittest

import numpy as np

from clean.reference import fp8
from clean.reference import swin as S


def q8_slow(v):
    """Nearest E4M3FN value by exhaustive search, ties to the even code, saturating at +-448."""
    if math.isnan(v):
        return math.nan
    a = min(abs(v), 448.0)
    best, best_code = None, None
    for code in range(0x7F):
        d = abs(fp8.DECODE[code] - a)
        if best is None or d < best or (d == best and code % 2 == 0):
            best, best_code = d, code
    return math.copysign(fp8.DECODE[best_code], v)


def mirror_slow(p, n):
    if p < 0:
        return -p
    if p >= n:
        return 2 * (n - 1) - p
    return p


def block_slow(x, p, offset, heads):
    """Spec 02 "One Swin block", written window by window and token by token."""
    H, W, C = x.shape
    ox, oy = offset
    out = np.full_like(x, np.nan)
    eps = 2.0 ** -13 / C

    def rms(v, g):
        return g * v / math.sqrt(float(np.mean(v * v)) + eps)

    for wy in range((H + oy + 3) // 4):
        for wx in range((W + ox + 3) // 4):
            pos = [(4 * wy - oy + t // 4, 4 * wx - ox + t % 4) for t in range(16)]
            xs = [x[mirror_slow(r, H), mirror_slow(c, W)] for r, c in pos]
            a = [fp8.q8(rms(v, p["gamma1"])) for v in xs]
            o = np.zeros((16, heads * 32))
            for h in range(heads):
                u = [fp8.q8(p["Wqk"][h] @ at) for at in a]
                vv = [fp8.f16(p["Wv"][h] @ at) for at in a]
                for t in range(16):
                    logits = []
                    for s in range(16):
                        sc = float(u[t] @ u[s]) + p["B"][h, t, s]
                        g = sc - sc ** 3 / (3 * S.CAP ** 2) if abs(sc) <= S.CAP else math.copysign(2 * S.CAP / 3, sc)
                        logits.append(S.TAU * g)
                    m = max(logits)
                    e = [math.exp(z - m) for z in logits]
                    A = [float(fp8.f16(ei / sum(e))) for ei in e]  # spec 02 step 5: rounded to FP16
                    o[t, h * 32:(h + 1) * 32] = sum(A[s] * vv[s] for s in range(16))
            for t, (r, c) in enumerate(pos):
                if 0 <= r < H and 0 <= c < W:
                    x1 = xs[t] + p["Wp"] @ fp8.q8(o[t]) + p["bp"]
                    m = fp8.q8(rms(x1, p["gamma2"]))
                    pre = p["W1"] @ m + p["b1"]
                    hid = fp8.q8(np.array([0.5 * z * (1 + math.erf(z / math.sqrt(2))) for z in pre]))
                    out[r, c] = x1 + p["W2"] @ hid + p["b2"]
    return out


class TestFp8(unittest.TestCase):
    def test_q8_matches_exhaustive_search(self):
        rng = np.random.default_rng(0)
        vals = list(fp8.DECODE[:0x7F]) + list(-fp8.DECODE[:0x7F])
        mids = [(fp8.DECODE[i] + fp8.DECODE[i + 1]) / 2 for i in range(0x7E)]
        vals += mids + [-m for m in mids] + [m * (1 + 1e-9) for m in mids] + [m * (1 - 1e-9) for m in mids]
        vals += [448.0, 455.9, 464.0, 470.0, 1e6, -1e6, 2.0 ** -10, 2.0 ** -11, 0.0, -0.0]
        vals += list(rng.standard_normal(2000) * 10) + list(rng.standard_normal(500) * 0.01)
        got = fp8.q8(np.array(vals))
        want = np.array([q8_slow(v) for v in vals])
        np.testing.assert_array_equal(got, want)

    def test_special_values(self):
        self.assertEqual(fp8.q8(np.inf), 448.0)
        self.assertEqual(fp8.q8(-np.inf), -448.0)
        self.assertTrue(np.isnan(fp8.q8(np.nan)))

    def test_encode_decode_roundtrip(self):
        codes = np.array([c for c in range(256) if c & 0x7F != 0x7F], dtype=np.uint8)
        np.testing.assert_array_equal(fp8.fp8_encode(fp8.fp8_decode(codes)), codes)
        self.assertEqual(fp8.fp8_encode(np.array([np.nan]))[0] & 0x7F, 0x7F)


class TestWindows(unittest.TestCase):
    def test_mirror(self):
        n = 10
        for p in range(-2, n + 2):
            self.assertEqual(int(S.mirror(p, n)), mirror_slow(p, n))
        self.assertEqual([int(S.mirror(p, 10)) for p in (-1, -2, 10, 11)], [1, 2, 8, 7])

    def test_every_token_in_exactly_one_window(self):
        for H, W in [(8, 12), (12, 20)]:
            for offset in [(0, 0), (2, 2), (0, 2), (2, 0), (1, 3)]:
                rows, cols = S.window_grid(H, W, offset)
                count = np.zeros((H, W), int)
                for r in rows:
                    for c in cols:
                        for rr in r:
                            for cc in c:
                                if 0 <= rr < H and 0 <= cc < W:
                                    count[rr, cc] += 1
                self.assertTrue((count == 1).all(), (H, W, offset))

    def test_offset_table(self):
        self.assertEqual(S.window_offset("enc1", 0), (2, 2))
        self.assertEqual(S.window_offset("b4", 1), (0, 2))
        self.assertEqual(S.window_offset("b5", 0), (0, 0))
        self.assertEqual(S.window_offset("dec1", 1), (2, 0))


class TestStages(unittest.TestCase):
    def test_block_matches_slow_version(self):
        rng = np.random.default_rng(3)
        for stage, H, W, offset in [("enc1", 8, 8, (2, 2)), ("b1", 4, 8, (2, 0)), ("dec2", 8, 4, (0, 2))]:
            cfg = S.STAGES[stage]
            p = S.random_params(stage, rng)
            x = fp8.q8(rng.standard_normal((H, W, cfg["C"])))
            fast = S.swin_block(x, p, offset, cfg["heads"])
            slow = block_slow(x, {k: np.asarray(v, np.float64) for k, v in p.items()}, offset, cfg["heads"])
            np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)

    def test_merge_and_expand(self):
        rng = np.random.default_rng(4)
        s = fp8.q8(rng.standard_normal((4, 6, 64)))
        Wm = fp8.q8(rng.standard_normal((96, 256)) / 16)
        bm = rng.standard_normal(96)
        m = S.patch_merge(s, Wm, bm)
        for i in range(2):
            for j in range(3):
                z = np.concatenate([s[2 * i, 2 * j], s[2 * i, 2 * j + 1], s[2 * i + 1, 2 * j], s[2 * i + 1, 2 * j + 1]])
                np.testing.assert_array_equal(m[i, j], fp8.q8(Wm @ z + bm))
        y = fp8.q8(rng.standard_normal((2, 3, 96)))
        skip = fp8.q8(rng.standard_normal((4, 6, 64)))
        We = fp8.q8(rng.standard_normal((256, 96)) / 10)
        be = rng.standard_normal(256)
        xe = S.expand(y, skip, We, be)
        for r in range(4):
            for c in range(6):
                e = fp8.q8(We @ y[r // 2, c // 2] + be)
                k = 2 * (r % 2) + c % 2
                np.testing.assert_array_equal(xe[r, c], e[k * 64:(k + 1) * 64] + skip[r, c])

    def test_decoder_border_member_is_mirrored_token(self):
        """Spec 02 "Expand" / ANSWERS.md 4: a window member outside the map is x(mirror(r), mirror(c)), with expand
        and skip evaluated at that mirrored position. swin_block gathers x[mirror(r), mirror(c)] from expand()."""
        rng = np.random.default_rng(6)
        H, W, C, CL = 8, 12, 64, 96
        y = fp8.q8(rng.standard_normal((H // 2, W // 2, CL)))
        skip = fp8.q8(rng.standard_normal((H, W, C)))
        We = fp8.q8(rng.standard_normal((4 * C, CL)) / 10)
        be = rng.standard_normal(4 * C)
        x = S.expand(y, skip, We, be)
        for r in (-2, -1, 0, H - 1, H, H + 1):
            for c in (-2, -1, 0, W - 1, W, W + 1):
                mr, mc = mirror_slow(r, H), mirror_slow(c, W)
                e = fp8.q8(We @ y[mr // 2, mc // 2] + be)
                k = 2 * (mr % 2) + mc % 2
                member = e[k * C:(k + 1) * C] + skip[mr, mc]
                np.testing.assert_array_equal(x[int(S.mirror(r, H)), int(S.mirror(c, W))], member)

    def test_run_stage_outputs(self):
        rng = np.random.default_rng(5)
        for stage in S.STAGES:
            p = S.random_params(stage, rng)
            x, skip = S.random_inputs(stage, 8, 8, rng)
            out = S.run_stage(stage, p, x, S.window_offset(stage, 0), skip)
            for v in out.values():
                self.assertTrue(np.isfinite(v).all())
                self.assertTrue(fp8.is_fp8_exact(v))


if __name__ == "__main__":
    unittest.main()
