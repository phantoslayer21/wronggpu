# SPDX-License-Identifier: MIT
"""Reference of "Motion at output resolution" (spec 03 draft 4, spec 06 section 3): clean/reference/motion_hi.py and
its use in reference/encoder.py and reference/post.py. Synthetic data and the spec team's vectors
(data/motion-hi-vectors.json, tests 03-T4 and 06-T6).

- texel selection of the input encoder against a scalar version written from the spec text (jitter, clamps, NaN);
- texel selection of the reconstruction against exact rational arithmetic, and the either-texel rule: off the pixels
  whose centre lies exactly between two texels the rule equals the exact floor; on them it is one of the two
  neighbours;
- s = RN32(mv_scale / RN32(M / D)) and the fused q;
- the encoder with three equal motions, and the single-motion reconstruction (against the render-resolution
  reference in the cases where the two must agree, and against a per-pixel loop);
- 03-T4: the three encoder vectors, bit for bit; 06-T6: the two reconstruction vectors with the measures of spec 09.

Run: python -m unittest discover -s clean/tests -t .
"""
import json
import math
import os
import struct
import unittest
from fractions import Fraction

import numpy as np

from clean.reference import encoder as En
from clean.reference import motion_hi as MH
from clean.reference import post as Po

VECTORS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                       "motion-hi-vectors.json")


def f32(v):
    """One FP32 rounding of a Python float (IEEE double)."""
    return struct.unpack("f", struct.pack("f", v))[0]


def vector_array(a):
    v = np.array(a["data"])
    t = a["type"]
    if t == "float32_bits":
        v = v.astype(np.uint32).view(np.float32)
    elif t == "float16_bits":
        v = v.astype(np.uint16).view(np.float16)
    else:
        v = v.astype(np.uint8 if t == "uint8" else np.uint16)
    return v.reshape(a["shape"])


def load_vectors():
    with open(VECTORS) as f:
        return [dict(c, arrays={k: vector_array(v) for k, v in c["arrays"].items()}) for c in json.load(f)["cases"]]


def recon_inputs(c, **kw):
    A = c["arrays"]
    x = {"color": A["color"], "A": A["A"] / 255.0, "B": A["B"] / 255.0, "C": A["C"].astype(np.float32),
         "history": A["history_prev"].astype(np.float32), "exposure": c["e"], "exposure_prev": c["e_prev"],
         "jitter": c["jitter"], "reset": c["reset"], "recon_size": tuple(c["D"]), "motion_output_res": True,
         "motion": A["motion_texture"], "motion_size": tuple(c["M"]), "motion_origin": tuple(c["motion_origin"]),
         "mv_scale": c["mv_scale"], "mv_bias": c["b_mv"]}
    x.update(kw)
    return x


def fp16_steps(a, b):
    """Distance in FP16 steps between two float16 arrays (finite values)."""
    def order(v):
        u = np.asarray(v, np.float16).view(np.uint16).astype(np.int32)
        return np.where(u & 0x8000, -(u & 0x7FFF), u)
    return np.abs(order(a) - order(b))


def spec09(expected, got):
    """Spec 09 "Stages with 8-bit and FP16 outputs", FP16 measures: (fraction more than one step and more than 0.001
    away, mean absolute error / RMS of the expected values)."""
    e, g = np.asarray(expected, np.float64), np.asarray(got, np.float64)
    far = (fp16_steps(expected, got) > 1) & (np.abs(e - g) > 1e-3)
    return float(far.mean()), float(np.abs(e - g).mean() / math.sqrt((e * e).mean()))


class TexelTest(unittest.TestCase):
    def test_encoder_texels_scalar(self):
        rng = np.random.default_rng(1)
        for rw, mw in ((32, 48), (31, 47), (24, 48), (1705, 2560), (960, 1440), (1280, 3840), (7, 5)):
            for j in list(rng.uniform(-0.5, 0.5, 4)) + [0.0, 0.5, -0.5, 300.0, -300.0, float("inf"), float("nan")]:
                got = MH.encoder_texels(rw, mw, j)
                k = f32(mw / rw)                                   # RN32(M / R)
                for x in list(range(0, rw, max(1, rw // 97))) + [rw - 1]:
                    t = f32(f32((x + 0.5) - f32(j)) * k)           # the difference, then the product
                    want = 0 if math.isnan(t) else int(min(max(math.floor(t) if math.isfinite(t) else t, 0), mw - 1))
                    self.assertEqual(int(got[x]), want, (rw, mw, j, x))

    def test_encoder_texels_jitter_and_clamps(self):
        # without jitter and M = 1.5 R: render pixel x reads texel floor((x + 1/2) 1.5)
        self.assertEqual(list(MH.encoder_texels(8, 12, 0.0)), [0, 2, 3, 5, 6, 8, 9, 11])
        # a jitter of +1/2 moves the sample half a render pixel to the left: floor(x 1.5)
        self.assertEqual(list(MH.encoder_texels(8, 12, 0.5)), [0, 1, 3, 4, 6, 7, 9, 10])
        self.assertEqual(list(MH.encoder_texels(4, 6, 100.0)), [0, 0, 0, 0])      # clamp below
        self.assertEqual(list(MH.encoder_texels(4, 6, -100.0)), [5, 5, 5, 5])     # clamp above

    def test_recon_texels_and_boundary(self):
        for dw, mw in ((72, 48), (71, 47), (42, 28), (41, 27), (3840, 2560), (2160, 1440), (5760, 3840), (2880, 1920),
                       (1617, 1078), (1000, 667)):
            got = MH.recon_texels(dw, mw)
            bd = MH.recon_boundary(dw, mw)
            lo, hi = MH.recon_texel_alternatives(dw, mw)
            exact = [Fraction(2 * x + 1, 2 * dw) * mw for x in range(dw)]
            for x in range(dw):
                whole = exact[x].denominator == 1
                self.assertEqual(bool(bd[x]), whole, (dw, mw, x))
                # the rule itself, one FP32 rounding per operation
                self.assertEqual(int(got[x]), min(int(f32(f32((x + 0.5) / dw) * mw)), mw - 1), (dw, mw, x))
                if whole:   # exactly between two texels: either neighbour
                    self.assertIn(int(got[x]), (int(exact[x]) - 1, int(exact[x])), (dw, mw, x))
                    self.assertEqual((int(lo[x]), int(hi[x])), (max(int(exact[x]) - 1, 0), min(int(exact[x]), mw - 1)))
                else:
                    self.assertEqual(int(lo[x]), int(hi[x]))
            # D = 1.5 M exactly: every third pixel (x = 1 mod 3) is such a pixel, and no other
            if 2 * dw == 3 * mw:
                self.assertTrue(np.array_equal(bd, np.arange(dw) % 3 == 1))
            else:
                self.assertLess(bd.mean(), 0.34)
            # off those pixels the FP32 rule may still differ from the exact floor where the exact position is within
            # an FP32 rounding of a whole number; count them (information for the results page, not a failure)
            off = [x for x in range(dw) if not bd[x] and int(got[x]) != min(math.floor(exact[x]), mw - 1)]
            self.assertEqual(off, [], (dw, mw, off[:5]))

    def test_division_by_the_rounded_reciprocal(self):
        """The kernels (kernels/post_motion_outres.h mo_center) form RN32((x + 1/2) / n) from rn = RN32(1 / n):
        q0 = RN32(a rn), q = fma(fma(-n, q0, a), rn, q0). It must be the correctly rounded quotient for every pixel
        of every size (here: every n up to 700 and a list of real D sizes; all n <= 12000 were checked once)."""
        sizes = list(range(1, 701)) + [960, 1080, 1440, 1616, 1617, 2160, 2870, 2880, 3240, 3840, 4320, 5760, 6480,
                                       7680, 11520]
        for n in sizes:
            a = np.arange(n, dtype=np.float32) + np.float32(0.5)
            d = np.float32(n)
            rn = np.float32(1.0) / d
            q0 = (a * rn).astype(np.float32)
            q = MH.fma32(MH.fma32(-d, q0, a), rn, q0)
            self.assertTrue(np.array_equal(q, (a / d).astype(np.float32)), n)

    def test_scale_and_fma(self):
        self.assertEqual(MH.scale((1.0, 1.0), (2560, 1440), (3840, 2160)).tolist(), [1.5, 1.5])
        s = MH.scale((23.5, -13.5), (47, 27), (71, 41))
        self.assertEqual(s.tolist(), [f32(23.5 / f32(47 / 71)), f32(-13.5 / f32(27 / 41))])
        # fma32 against exact rational arithmetic (one rounding of the exact a b + c)
        rng = np.random.default_rng(3)
        a = rng.normal(0, 30, 2000).astype(np.float32)
        b = rng.normal(0, 3, 2000).astype(np.float32)
        c = (rng.integers(0, 4000, 2000) + 0.5).astype(np.float32)
        got = MH.fma32(a, b, c)
        for i in range(2000):
            exact = Fraction(float(a[i])) * Fraction(float(b[i])) + Fraction(float(c[i]))
            lo, hi = np.nextafter(got[i], np.float32(-np.inf)), np.nextafter(got[i], np.float32(np.inf))
            err = abs(Fraction(float(got[i])) - exact)
            self.assertLessEqual(err, abs(Fraction(float(lo)) - exact))
            self.assertLessEqual(err, abs(Fraction(float(hi)) - exact))
        # a case where "multiply, round, add, round" differs from the fused operation
        x, y, z = np.float32(1 + 2 ** -12), np.float32(1 + 2 ** -12), np.float32(-1.0)
        self.assertNotEqual(float(MH.fma32(x, y, z)), float(np.float32(x * y) + z))


class EncoderTest(unittest.TestCase):
    def test_three_equal_motions_and_everything_else_unchanged(self):
        rng = np.random.default_rng(5)
        x = En.random_inputs(rng, rw=40, rh=24, dw=90, dh=54)
        params = En.random_params(rng)
        mw, mh = 60, 36
        tex = rng.normal(0, 2, (mh + 3, mw + 4, 2)).astype(np.float32)
        hi = dict(x, motion_output_res=True, motion=tex, motion_size=(mw, mh), motion_origin=(2, 1),
                  mv_scale=np.array([1.0, -1.0], np.float32))
        mn, mf, mc = En.motion_surfaces(hi)
        self.assertTrue(np.array_equal(mn, mc) and np.array_equal(mf, mc))
        out = En.encode(hi, params)
        self.assertTrue(np.array_equal(out["m_near"], out["m_far"]) and np.array_equal(out["m_near"], out["m_center"]))
        # per pixel from the spec text
        s = MH.scale(hi["mv_scale"], (mw, mh), (90, 54))
        for (px, py) in ((0, 0), (39, 23), (17, 5), (3, 20)):
            tx = min(max(math.floor(f32(f32((px + 0.5) - float(x["jitter"][0])) * f32(mw / 40))), 0), mw - 1)
            ty = min(max(math.floor(f32(f32((py + 0.5) - float(x["jitter"][1])) * f32(mh / 24))), 0), mh - 1)
            mv = tex[1 + ty, 2 + tx]
            want = [f32(float(s[i]) * f32(float(x["mv_bias"][i]) + float(mv[i]))) for i in range(2)]
            self.assertEqual(mc[py, px].tolist(), want)
        # the same frame at render resolution with that motion on every pixel and a flat depth (near = far = centre)
        # is the same encoder input: every value and feature must be equal
        lo = dict(x, depth=np.zeros_like(x["depth"]))
        lo["motion"] = (mc / (np.asarray(x["mv_scale"], np.float64) * np.array([90 / 40, 54 / 24]))
                        - np.asarray(x["mv_bias"], np.float64))
        ref = En.encode(lo, params)
        self.assertLess(np.nanmax(np.abs(ref["values"] - out["values"])), 1e-4)

    def test_depth_is_not_read(self):
        rng = np.random.default_rng(6)
        x = En.random_inputs(rng, rw=40, rh=24, dw=90, dh=54)
        params = En.random_params(rng)
        hi = dict(x, motion_output_res=True, motion=rng.normal(0, 2, (36, 60, 2)).astype(np.float32),
                  motion_size=(60, 36))
        a = En.encode(hi, params)
        b = En.encode(dict(hi, depth=rng.uniform(0, 1, x["depth"].shape).astype(np.float32),
                           depth_inverted=not x["depth_inverted"]), params)
        self.assertTrue(np.array_equal(a["features"], b["features"]))


class ReconTest(unittest.TestCase):
    def make(self, seed, rw=20, rh=12, ow=30, oh=18):
        rng = np.random.default_rng(seed)
        dw, dh = Po.recon_size_for(ow), Po.recon_size_for(oh)
        x = Po.random_inputs(rng, rw, rh, dw=dw, dh=dh)
        x.update(motion_output_res=True, motion_size=(ow, oh), mv_scale=np.array([1.0, 1.0], np.float32),
                 mv_bias=(0.0, 0.0))
        return rng, x, (dw, dh)

    def test_uniform_motion_equals_render_resolution_path(self):
        # one motion everywhere: both paths reproject every pixel by the same vector (1.5 v, exact in FP16)
        rng, x, (dw, dh) = self.make(11)
        for v in ((0.0, 0.0), (2.0, -1.0), (-0.75, 0.5), (500.0, 0.0)):
            x["motion"] = np.broadcast_to(np.array(v, np.float32), (18, 30, 2)).copy()
            hi = Po.reconstruct(x, 1.3)
            m = np.broadcast_to(1.5 * np.array(v, np.float32), (12, 20, 2)).astype(np.float32)
            lo = Po.reconstruct(dict(x, motion_output_res=False, m_near=m, m_center=m, m_far=m), 1.3)
            for k in ("history", "reduced", "display"):
                self.assertTrue(np.array_equal(hi[k], lo[k]), (v, k))

    def test_single_motion_per_pixel_loop(self):
        rng, x, (dw, dh) = self.make(12)
        tex = rng.normal(0, 3, (18 + 2, 30 + 3, 2)).astype(np.float32)
        tex[4, 5] = np.nan
        tex[7, 9] = (1e6, 0.0)
        x.update(motion=tex, motion_origin=(3, 2), mv_scale=np.array([1.25, -0.5], np.float32), mv_bias=(0.125, -0.25))
        out = Po.reconstruct(x, 0.02)
        L = out["_lin"]
        s = MH.scale(x["mv_scale"], (30, 18), (dw, dh))
        hist = np.asarray(x["history"], np.float64)
        color = np.asarray(x["color"], np.float64)
        px, py = Po.target_positions(20, 12, dw, dh, *x["jitter"])
        e, ep = x["exposure"], x["exposure_prev"]
        # steps 1, 2 and the weights are the shared code (tested in test_post_reference); steps 3 to 5 per pixel here
        W, fbar = Po.gather(Po.render_features(color, x["A"], x["B"], e), x["C"], px, py, 20, 12)
        C_lin = Po.dec(fbar[..., 0:3])
        alpha = Po.clamp(0.02 * W * fbar[..., 6], 0.0, 1.0)
        bh = alpha * (1.0 - fbar[..., 7]) + fbar[..., 7]
        checked = {"valid": 0, "fallback": 0, "nan": 0}
        for y in range(dh):
            for xx in range(dw):
                tx = min(int(f32(f32((xx + 0.5) / dw) * 30)), 29)
                ty = min(int(f32(f32((y + 0.5) / dh) * 18)), 17)
                mv = tex[2 + ty, 3 + tx]
                v = [f32(float(x["mv_bias"][i]) + float(mv[i])) for i in range(2)]
                q = [float(MH.fma32(s[0], np.float32(v[0]), np.float32(xx + 0.5))),
                     float(MH.fma32(s[1], np.float32(v[1]), np.float32(y + 0.5)))]
                if math.isnan(q[0]) or math.isnan(q[1]):
                    kind = "nan"       # counts as valid; the NaN guard gives C_lin
                    want = C_lin[y, xx]
                else:
                    if 0 <= q[0] <= dw and 0 <= q[1] <= dh:
                        kind = "valid"
                        S = Po.history_sample(hist, np.array([[q[0]]]), np.array([[q[1]]]))[0, 0]
                    else:
                        kind = "fallback"
                        fx = min(max(math.floor(px[xx]), 0), 19)
                        fy = min(max(math.floor(py[y]), 0), 11)
                        S = Po.enc(ep * color[fy, fx])
                    w1, w2 = L["w1"][y, xx], L["w2"][y, xx]
                    H = w1 * S + (w2 * S if w2 != 0 else 0.0)     # one sample, both weights
                    H_lin = Po.dec(H) / ep * e
                    want = H_lin + (C_lin[y, xx] - H_lin) * bh[y, xx]
                checked[kind] += 1
                self.assertTrue(np.allclose(L["H_hat"][y, xx], want, rtol=1e-12, atol=1e-300), (xx, y, kind))
        self.assertTrue(all(v > 0 for v in checked.values()), checked)

    def test_either_texel_rule(self):
        # D = 1.5 M: the two acceptable texel choices give different results only on the pixels of recon_boundary
        rng, x, (dw, dh) = self.make(13)
        x["motion"] = rng.normal(0, 3, (18, 30, 2)).astype(np.float32)
        bx, by = MH.recon_boundary(dw, 30), MH.recon_boundary(dh, 18)
        self.assertTrue(np.array_equal(bx, np.arange(dw) % 3 == 1))
        (lox, hix), (loy, hiy) = MH.recon_texel_alternatives(dw, 30), MH.recon_texel_alternatives(dh, 18)
        base = Po.reconstruct(x, 0.02)["history"]
        lo = Po.reconstruct(dict(x, motion_texels=(lox, loy)), 0.02)["history"]
        hi = Po.reconstruct(dict(x, motion_texels=(hix, hiy)), 0.02)["history"]
        mixed = [Po.reconstruct(dict(x, motion_texels=t), 0.02)["history"] for t in ((lox, hiy), (hix, loy))]
        bd = by[:, None] | bx[None, :]
        self.assertTrue(np.array_equal(lo[~bd], hi[~bd]))
        self.assertTrue(np.array_equal(base[~bd], lo[~bd]))
        self.assertGreater(int((lo[bd] != hi[bd]).any(-1).sum()), bd.sum() // 2)
        # the rule's own result is one of the acceptable ones at every pixel
        ok = np.zeros(bd.shape, bool)
        for cand in (lo, hi, *mixed):
            ok |= (cand == base).all(-1)
        self.assertTrue(ok.all())

    def test_reset_and_missing_surfaces(self):
        rng, x, _ = self.make(14)
        x["motion"] = rng.normal(0, 3, (18, 30, 2)).astype(np.float32)
        for k in ("m_near", "m_center", "m_far"):
            x.pop(k)                                             # the three surfaces are not read
        a = Po.reconstruct(dict(x, reset=True), 1.0)
        b = Po.reconstruct(dict(x, reset=True, motion=np.full((18, 30, 2), np.nan, np.float32)), 1.0)
        self.assertTrue(np.array_equal(a["history"], b["history"]))   # on reset the motion does not matter


class VectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_vectors()

    def test_03_T4_encoder_vectors_bit_for_bit(self):
        n = 0
        for c in self.cases:
            if not c["case"].startswith("enc/"):
                continue
            A = c["arrays"]
            m = MH.encoder_motion(A["motion_texture"], c["R"], c["M"], c["D"], c["jitter"], c["mv_scale"], c["b_mv"],
                                  c["motion_origin"])
            with np.errstate(over="ignore"):
                got = m.astype(np.float16)
            self.assertTrue(np.array_equal(got.view(np.uint16), A["motion"].view(np.uint16)), c["case"])
            n += 1
        self.assertEqual(n, 3)

    def test_06_T6_reconstruction_vectors(self):
        """This tree's reference is FP64 with the operation order of spec 06 draft 3; the vectors are draft 4 (FP16
        kernel exponent and accumulators). So the comparison is spec 09's, not bit for bit; the motion-specific part
        (texel, q, single sample, NaN position) is what decides whether it passes: with the neighbouring texel on the
        non-boundary pixels, or the fallback at NaN positions, it fails by a wide margin (checked below)."""
        n = 0
        for c in self.cases:
            if not c["case"].startswith("recon/"):
                continue
            A = c["arrays"]
            dw, dh = c["D"]
            mw, mh = c["M"]
            self.assertTrue(np.array_equal(A["boundary"].astype(bool),
                                           MH.recon_boundary(dh, mh)[:, None] | MH.recon_boundary(dw, mw)[None, :]))
            out = Po.reconstruct(recon_inputs(c), c["s"])
            got = out["history"].astype(np.float16)
            far, mae = spec09(A["full_history"], got)
            self.assertLessEqual(mae, 1e-3, c["case"])
            self.assertLessEqual(far, 5e-3, c["case"])            # draft 3 arithmetic against draft 4 vectors
            code = np.rint(out["reduced"].astype(np.float64) * 1023.0).astype(np.int64)
            exp = A["reduced_history"].astype(np.int64)[:code.shape[0], :code.shape[1]]   # (draft 4: ceil(D / 2))
            self.assertLessEqual(float((np.abs(code - exp) > 1).mean()), 5e-3, c["case"])
            # a wrong texel (the next one on every pixel) or the fallback at NaN positions is far outside
            tx, ty = MH.recon_texels(dw, mw), MH.recon_texels(dh, mh)
            wrong = Po.reconstruct(recon_inputs(c, motion_texels=(np.minimum(tx + 1, mw - 1), ty)), c["s"])
            self.assertGreater(spec09(A["full_history"], wrong["history"].astype(np.float16))[0], 20 * max(far, 1e-3))
            if np.isnan(A["motion_texture"]).any():
                old = Po.reconstruct(recon_inputs(c, nan_position_valid=False), c["s"])
                self.assertGreater(spec09(A["full_history"], old["history"].astype(np.float16))[0], 2 * far)
            n += 1
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
