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
  * the FP32 accumulation of the projections' K blocks against the stepwise FP16-rounded sum;
  * the centred 3 x 3 gather of l_post_tf_p12[_pf]_fast3 (LP_WIN3) against the 4 x 4 window, by kernel width;
  * the two-windows-per-wave Swin blocks (l_swin_c32_fast.hlsl, also the c64 / c96 twins): the clamped-cubic GELU
    (x + |x| min(P(|x|), 1)) / 2 against the model's GELU (and its FP8 codes, the doubled residual accumulator), the softmax
    without the maximum against the max-subtracted one, and the coefficient store in FP32
    (tanh as 1 - 2 / (2^(2 t log2 e) + 1), no RN16 points, the truncating FP16 pack with the kRn fold) against
    l_coef_px.hlsli's arithmetic.
Numbers are printed with -v; the bounds are the ones the fast route is gated on (tone-mapped PSNR >= 48 dB per choice).
"""
import math
import os
import unittest

import numpy as np

KERNELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dx12", "kernels")
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



def gelu_chain(x):
    """The model's GELU as the M-form kernels compute it (and the arena tables hold): x / (1 + 2^(x (b x^2 + a)))."""
    a, b = from_bits(0xc013b8ed), from_bits(0xbdcd6502)
    x = np.asarray(x, dtype=f32)
    with np.errstate(over="ignore"):
        return x / (f32(1.0) + np.exp2(x * (x * x * b + a)))


def fp8_e4m3(x):
    """Round to the nearest FP8 E4M3 value (v_cvt_pk_fp8_f32: nearest even; |x| <= 448 here)."""
    x = np.asarray(x, dtype=np.float64)
    _, e = np.frexp(np.abs(x))          # |x| = m 2^e, m in [0.5, 1)
    e = np.maximum(e, -5)               # below 2^-6: the subnormal step 2^-9
    step = np.ldexp(1.0, e - 4)         # 3 mantissa bits
    return np.sign(x) * np.round(np.abs(x) / step) * step


class PolyGelu(unittest.TestCase):
    """The clamped-cubic GELU of l_swin_c32_fast.hlsl (LF_GELU 1: the "_w2p_fast" width-32 blocks, the c64 / c96 twins):
    2 GELU(x) = x + |x| min(P(|x|), 1), P(t) = c0 + c1 t + c2 t^2 + c3 t^3, in the kernel's float32 operations (|x| by an
    AND, three fmas, the min, one fma; the halving is the doubled residual accumulator's, exact)."""
    C = (0xbd6c7ea1, 0x3f878bed, 0xbeb6e5c7, 0x3d25ab15)   # kLfP0 ... kLfP3
    K_QUICK = from_bits(0xc021eb85)                        # 2B-3's quick GELU constant, for comparison

    def setUp(self):
        self.c = [from_bits(v) for v in self.C]

    def g2(self, x):
        """2 GELU(x) as the kernel computes it (float32, fused multiply-adds)."""
        x = np.asarray(x, dtype=f32)
        t = np.abs(x)
        c0, c1, c2, c3 = (np.float64(v) for v in self.c)
        fma = lambda a, b, c: (np.asarray(a, np.float64) * np.asarray(b, np.float64) + c).astype(f32)
        p = fma(c3, t, c2)
        p = fma(p, t, c1)
        p = np.minimum(fma(p, t, c0), f32(1.0))
        return fma(t, p, x)

    def poly(self, x):
        return self.g2(x).astype(np.float64) * 0.5

    def test_constants_are_the_kernels(self):
        src = open(os.path.join(KERNELS, "l_swin_c32_fast.hlsl"), encoding="utf-8").read()
        for i, v in enumerate(self.C):
            self.assertIn("kLfP%d = asfloat(0x%08xu)" % (i, v), src)

    def test_error_bounds(self):
        x = np.linspace(-12, 12, 2400001, dtype=f32)
        err = np.abs(self.poly(x) - gelu_chain(x))
        rng = np.random.default_rng(11)
        xs = rng.normal(0, 1.5, 2000000).astype(f32)
        rms = np.sqrt(np.mean((self.poly(xs) - gelu_chain(xs)) ** 2))
        quick = xs * (f32(1.0) / (f32(1.0) + np.exp2(xs * self.K_QUICK)))
        rms_q = np.sqrt(np.mean((quick.astype(np.float64) - gelu_chain(xs)) ** 2))
        print("clamped-cubic GELU: max abs err %.5f, RMS %.5f on N(0, 1.5) (quick GELU RMS %.4f)" % (err.max(), rms, rms_q))
        self.assertLess(err.max(), 0.0021)
        self.assertLess(rms, 0.0015)
        self.assertLess(rms * 5, rms_q)
        # far out: GELU(x) = x for large x, 0 for large negative x (P stays >= 1, the min keeps it at 1)
        big = np.array([3.5, 10.0, 200.0, -3.5, -10.0, -200.0], dtype=f32)
        np.testing.assert_array_equal(self.g2(big), np.array([7.0, 20.0, 400.0, 0.0, 0.0, 0.0], dtype=f32))

    def test_polynomial_shape(self):
        # P' > 0 everywhere (no real root of c1 + 2 c2 t + 3 c3 t^2): P crosses 1 once (the knee) and never comes back;
        # P < 0 only below t = 0.06, where the missing lower clamp costs less than the bound above
        c0, c1, c2, c3 = (float(v) for v in self.c)
        self.assertLess(4 * c2 * c2 - 12 * c1 * c3, 0.0)
        t = np.linspace(0, 1000, 2000001)
        p = ((c3 * t + c2) * t + c1) * t + c0
        knee = t[np.argmax(p >= 1.0)]
        print("knee t = %.3f, P(0) = %.4f" % (knee, c0))
        self.assertTrue(3.0 < knee < 3.6)
        self.assertTrue(np.all(p[t >= knee] >= 1.0))
        self.assertLess(t[np.argmax(p >= 0.0)], 0.06)

    def test_doubled_accumulator_is_exact(self):
        # q8(2 g) = 2 q8(g) for every g whose FP8 code is normal on both sides: the W2 products accumulate onto 2 x the
        # residual and the output halves it, both exact power-of-two scalings
        rng = np.random.default_rng(14)
        g = rng.normal(0, 2.0, 200000)
        g = g[(np.abs(g) >= 2.0 ** -6) & (np.abs(g) <= 200.0)]
        np.testing.assert_array_equal(fp8_e4m3(2 * g), 2 * fp8_e4m3(g))

    def test_fp8_codes(self):
        # the W2 input is the FP8 code of GELU(h): the share of codes that change, and the mean signed error
        rng = np.random.default_rng(12)
        h = rng.normal(0, 1.5, 1000000).astype(f32)
        a = fp8_e4m3(gelu_chain(h))
        b = fp8_e4m3(self.g2(h)) * 0.5   # the kernel packs 2 GELU
        same = np.mean(a == b)
        bias = np.mean(b - a)
        print("clamped-cubic GELU FP8 codes: %.1f%% unchanged, mean signed error %.2e (mean |GELU| %.3f)" %
              (100 * same, bias, np.mean(np.abs(a))))
        self.assertGreater(same, 0.84)
        self.assertLess(abs(bias), 0.0005)


class SoftmaxNoMax(unittest.TestCase):
    def test_no_max_needed(self):
        rng = np.random.default_rng(13)
        n = 200000
        cap, k3, kt = from_bits(0x420179db), from_bits(0xb9a6cc38), from_bits(0x3f028297)
        s = np.concatenate([rng.normal(0, 4, (n, 16)), rng.uniform(-80, 80, (n // 10, 16))]).astype(f32)
        c = np.clip(s, -cap, cap)
        t = (c * c * c * k3 + c).astype(f32)
        # the M-form: z = t k - max(t) k; the fast form: z = t k (|z| <= 11: no overflow, no denormal)
        z = (t * kt).astype(f32)
        self.assertLess(np.abs(z).max(), 11.1)
        e_fast = np.exp2(z)
        p_fast = (e_fast * (f32(1.0) / e_fast.sum(1, dtype=f32, keepdims=True))).astype(f16)
        e_ref = np.exp2((t * kt - (t.max(1, keepdims=True) * kt)).astype(f32))
        p_ref = (e_ref * (f32(1.0) / e_ref.sum(1, dtype=f32, keepdims=True))).astype(f16)
        d = np.abs(p_fast.view(np.uint16).astype(np.int32) - p_ref.view(np.uint16).astype(np.int32))
        print("softmax without the max: FP16 weights equal %.3f%%, at most %d ulp apart" % (100 * np.mean(d == 0), d.max()))
        self.assertLessEqual(d.max(), 2)
        self.assertGreater(np.mean(d == 0), 0.97)


def gather(feat, kA, kB, kC, px, py, win3):
    """The normalized gather of the D pixels at (px, py) (render pixels, texel centres at i + 1/2) over the 4 x 4 window
    floor(p - 1/2) - 1 ... + 2 or the centred 3 x 3 window floor(p) - 1 ... + 1, float64. feat[y, x, k], shapes per texel."""
    if win3:
        cx, cy, taps = np.floor(px).astype(int), np.floor(py).astype(int), range(-1, 2)
    else:
        cx, cy, taps = np.floor(px - 0.5).astype(int), np.floor(py - 0.5).astype(int), range(-1, 3)
    num = np.zeros(px.shape + (feat.shape[2],))
    den = np.zeros(px.shape)
    for j in taps:
        for i in taps:
            X, Y = cx + i, cy + j
            u, v = X + 0.5 - px, Y + 0.5 - py
            Xc, Yc = np.clip(X, 0, feat.shape[1] - 1), np.clip(Y, 0, feat.shape[0] - 1)
            w = 2.0 ** (kA[Yc, Xc] * u * u + kB[Yc, Xc] * v * v + kC[Yc, Xc] * u * v)
            num += w[..., None] * feat[Yc, Xc]
            den += w
    return num / den[..., None]


class CentredGather(unittest.TestCase):
    def test_3x3_vs_4x4_by_width(self):
        rng = np.random.default_rng(14)
        H, W = 96, 96
        yy, xx = np.mgrid[0:H, 0:W]
        smooth = np.stack([0.5 + 0.4 * np.sin(xx * 0.21 + k) * np.cos(yy * 0.17 + 2 * k) for k in range(3)], -1)
        noise = rng.uniform(0, 1, (H, W, 3))
        r = 1707.0 / 3840.0   # render pixels per D pixel at 1440p Quality
        dy, dx = np.mgrid[0:int((H - 8) / r), 0:int((W - 8) / r)]
        px, py = (dx + 0.5) * r + 4.0, (dy + 0.5) * r + 4.0
        res = []
        for k0 in (4.0, 2.0, 1.0, 0.5):   # w = exp(-k0 u^2 / 2): sigma = 1 / sqrt(k0) render pixels
            kA = np.full((H, W), -0.721325 * k0)
            kB = kA * rng.uniform(0.7, 1.3, (H, W))
            kC = -1.44265 * k0 * rng.uniform(-0.2, 0.2, (H, W))
            out = []
            for feat in (smooth, noise):
                a = gather(feat, kA, kB, kC, px, py, False)
                b = gather(feat, kA, kB, kC, px, py, True)
                out.append(10 * math.log10(1.0 / max(np.mean((a - b) ** 2), 1e-20)))
            res.append((k0, out[0], out[1]))
            print("centred 3 x 3 vs 4 x 4 gather, sigma %.2f render px: PSNR %.1f dB (smooth field), %.1f dB (texel noise)" %
                  (1 / math.sqrt(k0), out[0], out[1]))
        # narrow kernels (sigma 0.5 render px): the dropped ring carries little weight
        self.assertGreater(res[0][1], 45.0)
        self.assertGreater(res[0][2], 35.0)
        # the difference grows with the kernel width: the A/B on the GPU decides (lever 256)
        self.assertGreater(res[0][2], res[-1][2])


def lcoef_exact(a, S, T):
    """l_coef_px.hlsli's values (FP32 arithmetic, the RN16 points) of one pixel's 13 channels a (FP16 inputs)."""
    a = [np.asarray(v, dtype=f16).astype(f32) for v in a]
    sig = lambda x: (f32(1.0) / (f32(1.0) + np.exp2(x * from_bits(0xBFB8AA3B)))).astype(f32)
    rn = lambda x: np.asarray(x, dtype=f32).astype(f16).astype(f32)
    unorm = lambda v: np.round(v * f32(255.0)).astype(np.int64)
    snorm = lambda v: np.round(np.clip(v, -1, 1) * f32(127.0)).astype(np.int64)
    tanh = lambda x: np.tanh(x.astype(np.float64)).astype(f32)
    A = [unorm(rn(sig(a[i]))) for i in range(3)]
    m = np.maximum(np.maximum(a[3], a[4]), a[5])
    e = [np.exp2((a[i] - m) * from_bits(0x3FB8AA3B)) for i in (3, 4, 5)]
    rs = f32(1.0) / (e[2] + (e[0] + e[1]))
    B = [unorm(x * rs) for x in e]
    g0, g1 = rn(sig(np.clip(a[6], -20, 20))), rn(sig(np.clip(a[7], -20, 20)))
    sx, sy = S * g0, S * g1
    C = [(sx * sx).astype(f16), (sy * sy).astype(f16), ((sx * sy) * tanh(a[8] * T)).astype(f16)]
    L = [snorm(tanh(a[i])) for i in (9, 10, 11, 12)]
    return A, B, C, L


def lcoef_fast(a, S, T):
    """l_swin_c32_fast.hlsl lf_coef_px: FP32 throughout from the block's FP32 output, the fast tanh, the truncating pack."""
    a = [np.asarray(v, dtype=f32) for v in a]
    sig = lambda x: (f32(1.0) / (f32(1.0) + np.exp2(x * from_bits(0xBFB8AA3B)))).astype(f32)
    tanh = lambda x: (f32(1.0) - f32(2.0) / (np.exp2(x * from_bits(0x4038AA3B)) + f32(1.0))).astype(f32)
    unorm = lambda v: np.floor(v * f32(255.0) + f32(0.5)).astype(np.int64)
    snorm = lambda v: np.round(v * f32(127.0)).astype(np.int64)
    A = [unorm(sig(a[i])) for i in range(3)]
    m = np.maximum(np.maximum(a[3], a[4]), a[5])
    e = [np.exp2((a[i] - m) * from_bits(0x3FB8AA3B)) for i in (3, 4, 5)]
    rs = f32(1.0) / (e[2] + (e[0] + e[1]))
    B = [unorm(x * rs) for x in e]
    g0, g1 = sig(np.clip(a[6], -20, 20)), sig(np.clip(a[7], -20, 20))
    S2 = f32(S) * f32(S) * f32(1.00034)
    sg0 = S2 * g0
    C = [rtz16(sg0 * g0), rtz16((S2 * g1) * g1), rtz16((sg0 * g1) * tanh(a[8] * T))]
    L = [snorm(tanh(a[i])) for i in (9, 10, 11, 12)]
    return A, B, C, L


class CoefFast(unittest.TestCase):
    def test_fast_tanh(self):
        t = np.linspace(-20, 20, 400001, dtype=f32)
        with np.errstate(over="ignore"):
            fast = f32(1.0) - f32(2.0) / (np.exp2(t * from_bits(0x4038AA3B)) + f32(1.0))
        err = np.abs(fast.astype(np.float64) - np.tanh(t.astype(np.float64)))
        print("fast tanh: max abs err %.3g" % err.max())
        self.assertLess(err.max(), 3e-7)
        self.assertTrue(np.all(np.abs(fast) <= 1.0))

    def test_codes_and_products(self):
        rng = np.random.default_rng(15)
        n = 300000
        a = [rng.normal(0, 2.5, n).astype(f32) for _ in range(13)]
        S, T = f32(1.7), f32(0.8)
        with np.errstate(over="ignore"):
            ex, fa = lcoef_exact(a, S, T), lcoef_fast(a, S, T)
        worst = 0
        for grp in (0, 1, 3):   # A, B, L: 8-bit codes
            for x, y in zip(ex[grp], fa[grp]):
                worst = max(worst, int(np.abs(x - y).max()))
        # the FP16 products in units of their scale S^2 (the exact form rounds its inputs to FP16 first: near rho = 0 the
        # relative difference of two tiny values says nothing)
        rel = max(float(np.quantile(np.abs(x.astype(np.float64) - y.astype(np.float64)), 0.999)) for x, y in zip(ex[2], fa[2]))
        rel /= float(S) * float(S)
        print("fast coefficient store: 8-bit codes within %d LSB, FP16 products 99.9%% within %.2e of S^2" % (worst, rel))
        self.assertLessEqual(worst, 1)
        self.assertLess(rel, 2.0 ** -9)


if __name__ == "__main__":
    unittest.main()
