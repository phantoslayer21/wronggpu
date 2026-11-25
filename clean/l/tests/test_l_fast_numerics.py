# SPDX-License-Identifier: MIT
"""CPU accuracy checks of the arithmetic choices of Model L's l_mode=fast kernels (no GPU).

    python -m unittest clean.l.tests.test_l_fast_numerics

Each test models one choice of clean/dx12/kernels/l_post_tf_fast.hlsl / l_ops_project.hlsl in NumPy (float32 / float16
arithmetic as the instructions give it) and bounds its error against the exact form:
  * Dec as exp2(q kDecK) - 1 for every lane against the 7-term polynomial the hybrid route uses below kDecPolyMax;
  * the round-to-nearest bias folded into a multiplier (kRn) and the half-FP16-step integer add (rn16) in front of the
    truncating pack v_cvt_pk_rtz_f16_f32;
  * the 16-bit horizontal Lanczos ring (two FP16 per dword) against the FP32 ring;
  * the FP16 tap weight (v_exp_f16) with the total weight summed in FP16 from the same weights, against the FP32 gather;
  * the MLP GELU forms (the arithmetic chain of the M-form kernels and the plain tanh form) against the erf GELU;
  * the FP32 accumulation of the projections' K blocks against the stepwise FP16-rounded sum.
Numbers are printed with -v; the bounds are the ones the fast route is gated on (tone-mapped PSNR >= 48 dB per choice).
"""
import math
import unittest

import numpy as np

f32 = np.float32
f16 = np.float16


def bits(x):
    return np.asarray(x, dtype=f32).view(np.uint32)


def from_bits(u):
    return np.asarray(u, dtype=np.uint32).view(f32)


def rtz16(x):
    """v_cvt_pk_rtz_f16_f32: round toward zero (the plain f32tof16 / (half) cast on this driver)."""
    x = np.asarray(x, dtype=f32)
    h = x.astype(f16)  # nearest even
    back = h.astype(f32)
    # where the nearest value is larger in magnitude than x, step one FP16 ulp toward zero
    over = np.abs(back) > np.abs(x)
    hb = h.view(np.uint16).astype(np.int32)
    hb = np.where(over, hb - 1, hb)
    return hb.astype(np.uint16).view(f16)


def rn16(x):
    """l_post_tf_fast.hlsl rn16: + half an FP16 step (bit pattern + 0x1000), then the truncating pack."""
    return rtz16(from_bits(bits(x) + np.uint32(0x1000)))


class DecFast(unittest.TestCase):
    kDecK = f32(3.32192809488736235) / f32(0.36)
    kDecLn = f32(2.30258509299404568) / f32(0.36)
    kDecPolyMax = f32(0.35) / kDecLn

    def poly(self, q):
        y = q * self.kDecLn
        t = y * f32(1 / 5040) + f32(1 / 720)
        for c in (f32(1 / 120), f32(1 / 24), f32(1 / 6), f32(0.5), f32(1.0)):
            t = t * y + c
        return t * y

    def test_dec_plain_vs_poly(self):
        q = np.linspace(1e-4, float(self.kDecPolyMax), 200001, dtype=f32)
        plain = np.exp2(q * self.kDecK).astype(f32) - f32(1.0)
        ref = np.expm1(q.astype(np.float64) * float(self.kDecLn))
        err_plain = np.abs(plain - ref).max()
        err_poly = np.abs(self.poly(q) - ref).max()
        # Dec's output is 0.01 times this value times dec_scale: the absolute error in units of the output
        print("Dec plain max abs err %.3g (poly %.3g), x 0.01 = %.3g" % (err_plain, err_poly, err_plain * 0.01))
        self.assertLess(err_plain, 1.5e-6)  # the polynomial form's own error is 1e-7 order; both are below 1e-5
        # relative at the darkest clamp (q = 1e-4): the plain form keeps 3 digits, below the display's FP16 resolution
        self.assertLess(abs(plain[0] - ref[0]) / ref[0], 5e-4)

    def test_dec_fold_g_k_hist(self):
        # (exp2(.) - 1) * (0.01 e / e_prev) against ((exp2(.) - 1) * 0.01) / e_prev * e (the hybrid order): <= 2 ulp
        rng = np.random.default_rng(1)
        q = rng.uniform(1e-4, 3.0, 100000).astype(f32)
        e, ep = f32(0.37), f32(1.9)
        v = np.exp2(q * self.kDecK).astype(f32) - f32(1.0)
        a = v * f32(f32(0.01) * e / ep)
        b = (v * f32(0.01)) * (f32(1.0) / ep) * e
        self.assertLess(np.abs(a / b - 1).max(), 3e-7)


class RoundingFold(unittest.TestCase):
    kRn = f32(1.00034)

    def test_rn16_matches_rne_except_ties(self):
        rng = np.random.default_rng(2)
        x = (rng.uniform(0.02, 3.0, 400000).astype(f32)) * rng.choice([-1, 1], 400000).astype(f32)
        want = x.astype(f16)  # nearest even
        got = rn16(x)
        diff = (got.view(np.uint16) != want.view(np.uint16)).mean()
        print("rn16 differs from RNE in %.5f%% of random inputs (exact ties go away from zero)" % (100 * diff))
        self.assertLess(diff, 1e-4)
        self.assertLessEqual(np.abs(got.astype(f32) - x).max(), np.abs(want.astype(f32) - x).max() + 1e-12)

    def test_kRn_fold_unbiased(self):
        rng = np.random.default_rng(3)
        x = np.exp(rng.uniform(math.log(0.01), math.log(3.0), 400000)).astype(f32)
        bias_plain = float(np.mean((rtz16(x).astype(np.float64) - x) / x))
        got = rtz16(x * self.kRn).astype(np.float64)
        bias_fold = float(np.mean((got - x) / x))
        worst = float(np.max(np.abs(got - x) / x))
        print("mean relative error of the truncating pack %.3g; with the kRn fold %.3g (worst %.3g; RNE worst 4.9e-4)" %
              (bias_plain, bias_fold, worst))
        self.assertLess(abs(bias_fold), 0.15 * abs(bias_plain))
        self.assertLess(abs(bias_fold), 6e-5)
        self.assertLess(worst, 1.1e-3)


# post_filter_c1_s4's analytic Lanczos taps (l_post_tf.hlsl kLz0 .. kLz4)
KLZ = from_bits([0xbdba3b4b, 0x3de71ca9, 0x3f23b70f, 0x3ed182b5, 0xbd90a4ac])


def lz5a(t, w):
    v = w[0] * t[0]
    for k in range(1, 5):
        v = np.float32(w[k]) * t[k] + v
    lo = np.minimum.reduce(t)
    hi = np.maximum.reduce(t)
    return np.minimum(np.maximum(v, lo), hi)


class Ring16(unittest.TestCase):
    def test_fp16_ring_psnr(self):
        rng = np.random.default_rng(4)
        H, W = 60, 192
        # display-domain values (E-encoded, [0, 2.45]): smooth gradients, texture and edges, already FP16 on entry
        yy, xx = np.mgrid[0:H, 0:W].astype(f32)
        img = 1.2 + 0.7 * np.sin(xx / 17.0) * np.cos(yy / 11.0) + 0.15 * rng.standard_normal((H, W)).astype(f32)
        img[:, 70:76] = 0.1
        img[20:30, :] += 0.8
        img = np.clip(img, 0.0, 2.45).astype(f16)
        wev = [KLZ[0], KLZ[1], KLZ[3], KLZ[4]]
        w_even = [KLZ[0], KLZ[1], KLZ[2], KLZ[3], KLZ[4]]
        w_odd = [KLZ[4], KLZ[3], KLZ[2], KLZ[1], KLZ[0]]
        t0 = img.astype(f32)
        # horizontal 3:2 decimation as the kernel: output columns 4g .. 4g + 3 from display columns 6g .. 6g + 9
        cols = []
        for g in range(W // 6 - 1):
            tp = [t0[:, 6 * g + i] for i in range(10)]
            cols += [lz5a(tp[0:5], w_even), lz5a(tp[2:7], w_odd), lz5a(tp[3:8], w_even), lz5a(tp[5:10], w_odd)]
        hz = np.stack(cols, axis=1).astype(f32)

        def vertical(ring):
            # 3:2 vertical: output row pairs from 5 ring rows (even phase / odd phase), as the vertical pass
            outs = []
            for p in range(2, H // 3 - 2):
                for dy in (0, 1):
                    r0 = 3 * p - 2 + 2 * dy
                    outs.append(lz5a([ring[r0 + k] for k in range(5)], w_even if dy == 0 else w_odd))
            return np.stack(outs)

        ref = vertical(hz)
        alt = vertical(np.array([rn16(row).astype(f32) for row in hz]))
        dec = lambda q: (np.exp2(np.clip(q, 1e-4, 3.0) * np.float32(9.2275)) - 1) * 0.01
        # tone-mapped (gamma 1/2.2 of the linear value, relative to a 10-unit white) PSNR of the linear output
        a, b = dec(ref), dec(alt)
        tm = lambda v: np.clip(v / 10.0, 0, 1) ** (1 / 2.2)
        mse = np.mean((tm(a) - tm(b)) ** 2)
        psnr = 10 * math.log10(1.0 / mse)
        print("FP16 ring vs FP32 ring: tone-mapped PSNR %.1f dB, max |dq| %.3g" % (psnr, np.abs(ref - alt).max()))
        self.assertGreater(psnr, 55.0)


class GatherHalf(unittest.TestCase):
    def test_half_weights_fbar(self):
        rng = np.random.default_rng(5)
        n = 20000
        # eps = kA u^2 + kB v^2 + kC u v per tap (4 x 4 window), kernel shapes like the staged ones (negative definite)
        u = (rng.uniform(-0.5, 0.5, (n, 1)) + np.arange(-1, 3)[None, :]).astype(f16)   # columns
        v = (rng.uniform(-0.5, 0.5, (n, 1)) + np.arange(-1, 3)[None, :]).astype(f16)   # rows
        kA = (-0.7213 * rng.uniform(0.3, 1.6, n)).astype(f16)
        kB = (-0.7213 * rng.uniform(0.3, 1.6, n)).astype(f16)
        kC = (-1.4427 * rng.uniform(-0.4, 0.4, n)).astype(f16)
        feat = (rng.uniform(0, 1, (n, 4, 4, 9)) * 0.5 + 0.3 * np.sin(np.arange(16).reshape(4, 4)[None, :, :, None])).astype(f16)
        # reference: FP32 weights, FP32 sums
        U = u.astype(np.float64)[:, None, :] * np.ones((1, 4, 1))
        V = v.astype(np.float64)[:, :, None] * np.ones((1, 1, 4))
        eps32 = kA.astype(np.float64)[:, None, None] * U**2 + kB.astype(np.float64)[:, None, None] * V**2 + \
            kC.astype(np.float64)[:, None, None] * U * V
        w32 = 2.0 ** eps32
        ref = (w32[..., None] * feat.astype(np.float64)).sum((1, 2)) / w32.sum((1, 2))[:, None]
        # fast: FP16 eps (fma chain), v_exp_f16 (nearest, +- 1 ulp), FP16 sums of w f and w
        Uh, Vh = U.astype(f16), V.astype(f16)
        uu = (Uh * Uh)
        eps16 = (Uh * Vh * kC[:, None, None]) + ((Vh * Vh) * kB[:, None, None] + uu * kA[:, None, None])
        w16 = np.exp2(eps16.astype(f32)).astype(f16)
        ulp = rng.integers(-1, 2, w16.shape)   # the hardware exponential is faithful, not always nearest
        w16 = (w16.view(np.uint16).astype(np.int32) + ulp).clip(0, 0x7bff).astype(np.uint16).view(f16)
        F = np.zeros((n, 9), dtype=f16)
        W = np.zeros(n, dtype=f16)
        for i in range(4):
            for j in range(4):
                w = w16[:, j, i]
                W = W + w
                for k in range(9):
                    F[:, k] = w * feat[:, j, i, k] + F[:, k]
        got = F.astype(np.float64) / np.maximum(W.astype(np.float64), 2.0**-24)[:, None]
        err = np.abs(got - ref)
        print("half gather: fbar max abs err %.3g, 99.9%% %.3g, rms %.3g (features in [0, 1])" %
              (err.max(), np.quantile(err, 0.999), np.sqrt((err**2).mean())))
        self.assertLess(np.quantile(err, 0.999), 3e-3)
        self.assertLess(np.sqrt((err**2).mean()), 5e-4)


class GeluForms(unittest.TestCase):
    def test_gelu_chain_and_tanh(self):
        x = np.linspace(-10, 10, 400001, dtype=np.float64)
        erf = np.vectorize(math.erf)
        ref = 0.5 * x * (1 + erf(x / math.sqrt(2)))
        # the M-form kernels' chain (gelu_group): x / (1 + 2^(x (b x^2 + a)))
        a, b = f32(-2.30816186), f32(-0.10029031)
        xf = x.astype(f32)
        chain = xf / (f32(1.0) + np.exp2(xf * (xf * xf * b + a)))
        # plain tanh GELU as x sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)) in exp2 form
        t = xf * (f32(1.0) + f32(0.044715) * xf * xf)
        tanh = xf / (f32(1.0) + np.exp2(f32(-2 * math.sqrt(2 / math.pi) / math.log(2)) * t))
        ec, et = np.abs(chain - ref).max(), np.abs(tanh - ref).max()
        print("GELU max abs err over [-10, 10]: M-form chain %.3g, tanh form %.3g" % (ec, et))
        self.assertLess(ec, 1e-3)
        self.assertLess(et, 1.1e-3)
        # the constant of the exp2 form: 2 sqrt(2 / pi) / ln 2 = 2.3022082 (not 2.3022227)
        self.assertAlmostEqual(2 * math.sqrt(2 / math.pi) / math.log(2), 2.3022082, places=6)


class ProjectAccumulation(unittest.TestCase):
    def test_fp32_accumulation_vs_stepwise_fp16(self):
        rng = np.random.default_rng(6)
        n, k = 4000, 64
        x = rng.uniform(-3, 3, (n, k)).astype(f32)
        w = rng.uniform(-0.3, 0.3, (n, k)).astype(f32)
        bias = rng.uniform(-1, 1, n).astype(f32)
        exact = bias.astype(np.float64) + (x.astype(np.float64) * w).sum(1)
        acc32 = bias.copy()
        acc16 = bias.copy()
        for g in range(k // 32):
            blk = (x[:, 32 * g:32 * g + 32] * w[:, 32 * g:32 * g + 32]).sum(1, dtype=f32)
            acc32 = acc32 + blk
            acc16 = (acc16 + blk).astype(f16).astype(f32)
        e32, e16 = np.abs(acc32 - exact).max(), np.abs(acc16 - exact).max()
        print("projection accumulation: FP32 max err %.3g, stepwise FP16 %.3g" % (e32, e16))
        self.assertLess(e32, e16)


if __name__ == "__main__":
    unittest.main()
