# SPDX-License-Identifier: MIT
"""Model K: the output alpha follows the colour input's alpha (issue #26). Host checks (no GPU, no HIP run).

The rule (the spec side's answer, in our terms): the output alpha is purely spatial, from the current frame's colour
input only.

- Carry: the colour input's fourth channel at render resolution, this frame's, exactly as the game supplies it (no
  pre-exposure division, no exposure scale, no clamp: negative values and values above 1 pass through). No history,
  nothing from the network. A colour format without an alpha channel: 1, the value a texture fetch gives a missing
  component.
- The output stage reads it at the same four render taps as the Y / Co / Cg current-colour 2 x 2 bilinear of spec 43
  section 4.3, with the same edge rule (tap positions clamped to the render rectangle) and the same four FP16 weights.
- Rounding: each tap's alpha is rounded to FP16 first; RN16(w0 a0); then the other three taps with FP16 fused
  multiply-adds in the tap order of the colour bilinear ((0,0), (1,0), (0,1), (1,1)).
- The result is the output's fourth channel, unchanged; HDR and LDR alike.

Checked here:
- alpha_reference (a straightforward CPU form of the rule, FP32 / FP16 steps with NumPy) against exact rational
  arithmetic on sampled pixels, and against the fourth channel of the spec 43 reference clean/k/reference/outstage.py
  (bit for bit: gradient, random, negative, above-1 and out-of-FP16-range alphas; HDR and LDR; several sizes and
  jitters; the render-rectangle edges);
- spatial only: the reference's alpha does not change with the histories, the network output, the motion, the exposure
  or the pre-exposure, and over two chained frames with different alpha the second frame's alpha is that of its own
  colour;
- no clamp: negative and above-1 results come out;
- sources: every K output path stores cbar3 (D3D12 k_out.hlsl: RGBA16F rows, the direct texture store, dense RGBA
  f32; HIP k_io.hip), the direct colour read keeps the texel's alpha, the interop's passes carry it for model K only
  (input passes, output pass on an RGBA f32 result, output pass on the result texture, fallback), the runtime's
  RGBA f32 result (FrameIo.out_rgba32f) and the test path's alpha.
The GPU checks are clean/k/tests/test_k_alpha_gpu.py (opt-in).

Run: python -m unittest clean.k.tests.test_k_alpha
"""
import os
import unittest
from fractions import Fraction

import numpy as np

from clean.k.reference.f16 import decode_exact, round_f16_exact
from clean.k.reference.fp import F32, R, f32, fma16, fma32, h, sat

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CLEAN = os.path.join(ROOT, "clean")
TAPS = ((0, 0), (1, 0), (0, 1), (1, 1))   # (dx, dy), the colour bilinear's order


def read(*p):
    with open(os.path.join(CLEAN, *p), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def alpha_taps(render, output, jitter):
    """Per output pixel [H, W] of spec 43 sections 4.1 / 4.3: the four taps' render coordinates (clamped to the render
    rectangle) and FP16 weights, in the tap order TAPS. -> (ty [4, H, W], tx [4, H, W], w [4, H, W] float16)."""
    rw, rh = (int(v) for v in render)
    W, H = (int(v) for v in output)
    jx, jy = F32(jitter[0]), F32(jitter[1])
    with np.errstate(all="ignore"):
        rho_x, rho_y = F32(rw) * (F32(1) / F32(W)), F32(rh) * (F32(1) / F32(H))
        jmx, jmy = jx - F32(0.5), jy - F32(0.5)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        px, py = f32(xx) + F32(0.5), f32(yy) + F32(0.5)
        # 4.1 local origin of the pixel's 16 x 16 block
        o_x = R(fma32(f32(16 * (xx // 16)) + F32(0.5), rho_x, jmx)) - F32(4)
        o_y = R(fma32(f32(16 * (yy // 16)) + F32(0.5), rho_y, jmy)) - F32(4)
        Ux, Uy = h(fma32(px, rho_x, jmx) - o_x), h(fma32(py, rho_y, jmy) - o_y)
        # 4.3 the bilinear's FP16 fractions and weights
        fUx, fUy = np.floor(Ux), np.floor(Uy)
        fx, fy = sat(h(Ux - fUx)), sat(h(Uy - fUy))
        one = np.float16(1)
        w = np.stack([h(h(one - fx) * h(one - fy)), h(fx * h(one - fy)), h(h(one - fx) * fy), h(fx * fy)])
        bx = (o_x + fUx.astype(np.float32)).astype(np.int64)
        by = (o_y + fUy.astype(np.float32)).astype(np.int64)
    tx = np.stack([np.clip(bx + dx, 0, rw - 1) for dx, _ in TAPS])
    ty = np.stack([np.clip(by + dy, 0, rh - 1) for _, dy in TAPS])
    return ty, tx, w.astype(np.float16)


def alpha_reference(alpha, render, output, jitter):
    """The rule: the output alpha [H, W] (float16) of a colour alpha [rh, rw] (any float type; None: a colour without
    an alpha channel, 1)."""
    rw, rh = (int(v) for v in render)
    a = np.ones((rh, rw), np.float32) if alpha is None else np.asarray(alpha, np.float32)
    assert a.shape == (rh, rw)
    ty, tx, w = alpha_taps(render, output, jitter)
    with np.errstate(all="ignore"):
        a16 = h(a)                                  # each tap's alpha rounded to FP16 first
        acc = h(a16[ty[0], tx[0]].astype(np.float64) * w[0].astype(np.float64))   # RN16(w0 a0): the product is exact
        for q in (1, 2, 3):                         # then three FP16 fused multiply-adds
            acc = fma16(a16[ty[q], tx[q]], w[q], acc)
    return acc.astype(np.float16)


def same16(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return (a.view(np.uint16) == b.view(np.uint16)) | (np.isnan(a) & np.isnan(b))


def make_alpha(rng, rw, rh, t=0):
    """A colour alpha [rh, rw] float32 that changes with t: gradient, random block, negative band, band above 1, far
    values (one beyond the FP16 range), other values on the first / last row and column."""
    y, x = np.mgrid[0:rh, 0:rw].astype(np.float32)
    a = np.mod((x + 2.0 * y + 5.0 * t) / np.float32(rw + 2 * rh), 1.0).astype(np.float32)
    a[rh // 4: rh // 2, rw // 4: rw // 2] = rng.random((rh // 2 - rh // 4, rw // 2 - rw // 4), np.float32)
    a[rh // 2, :] = -rng.random(rw, np.float32) * 3.0 - 0.01
    a[:, rw // 2] = 1.0 + rng.random(rh, np.float32) * 3.0
    a[rh // 3, rw // 3] = -37.5
    a[(2 * rh) // 3, (2 * rw) // 3] = 1000.0
    a[0, :] = 0.25 + 0.5 * (t % 2)
    a[-1, :] = 2.5
    a[:, 0] = -0.5
    a[:, -1] = 0.875
    return a


def exact16(v):
    kind = decode_exact(np.float16(v).view(np.uint16))
    assert kind[0] == "num"
    return kind[1] * kind[2]


class TestAlphaRule(unittest.TestCase):
    CASES = [((27, 16), (40, 24), (0.13, -0.27)), ((27, 16), (40, 24), (-0.5, 0.5)), ((20, 12), (40, 24), (0.0, 0.0)),
             ((85, 48), (128, 72), (0.31, 0.07)), ((13, 9), (48, 32), (-0.44, 0.21))]

    def test_exact_rational(self):
        """alpha_reference's FP16 chain against exact rational arithmetic with one rounding per step."""
        rng = np.random.default_rng(3)
        for render, output, jitter in self.CASES:
            rw, rh = render
            a = make_alpha(rng, rw, rh)
            got = alpha_reference(a, render, output, jitter)
            ty, tx, w = alpha_taps(render, output, jitter)
            a16 = a.astype(np.float16)
            H, W = got.shape
            for _ in range(400):
                y, x = int(rng.integers(H)), int(rng.integers(W))
                taps = [a16[ty[q, y, x], tx[q, y, x]] for q in range(4)]
                bits = round_f16_exact(exact16(taps[0]) * exact16(w[0, y, x]))
                for q in (1, 2, 3):
                    acc = exact16(np.uint16(bits).view(np.float16))
                    sgn = -1 if bits & 0x8000 else 1
                    bits = round_f16_exact(exact16(taps[q]) * exact16(w[q, y, x]) + acc, sign_if_zero=sgn)
                want = np.uint16(bits).view(np.float16)
                # (a zero sum's sign is not modelled exactly above: compare values, zeros of either sign equal)
                self.assertTrue(got[y, x] == want, f"{render} {output} {jitter} ({x}, {y}): {got[y, x]} != {want}")
            # the weights are those of a bilinear: each in [0, 1], and every tap inside the render rectangle
            self.assertTrue(np.all((w >= 0) & (w <= 1)))
            self.assertTrue(np.all((tx >= 0) & (tx < rw) & (ty >= 0) & (ty < rh)))
            # both edges of the rectangle are reached (the clamp rule is exercised)
            self.assertEqual((int(tx.min()), int(ty.min())), (0, 0))
            self.assertEqual((int(tx.max()), int(ty.max())), (rw - 1, rh - 1))

    def _frame(self, rng, hdr, render, output, jitter, alpha, reset=False):
        from clean.k.reference.layout import base_size
        from clean.k.reference.outstage import ceil16
        (rw, rh), (W, H) = render, output
        Bw, Bh = base_size(W, H)
        W16, H16 = ceil16(W), ceil16(H)
        color = rng.random((rh, rw, 4)).astype(np.float32) * (4 if hdr else 1)
        color[rng.random((rh, rw)) < 0.03, 1] = -0.5
        if alpha is None:
            color = color[..., :3]
        else:
            color[..., 3] = alpha
        N = (rng.standard_normal((Bh, Bw, 40)) * 1.2).astype(np.float16)
        return dict(N=N, color=color, motion=(rng.standard_normal((rh, rw, 2)) * 0.8).astype(np.float32),
                    depth=rng.random((rh, rw)).astype(np.float32), render_size=render, output_size=output, hdr=hdr,
                    mv_lowres=True, depth_inverted=False, jitter=jitter, mv_scale=(1.0, 0.9), mv_offset=(0.1, 0.0),
                    reset=reset, E_cur=1.5, E_prev=1.25, E_aux=1.0, e1=1.0, e2=1.1, pre_exposure=0.8,
                    L=(rng.random((2 * H16, 2 * W16)) * 1.05).astype(np.float16),
                    Hc=(rng.random((H16, W16, 4)) * 0.6).astype(np.float16),
                    Hh=(rng.standard_normal((Bh, Bw, 4)) * 0.3).astype(np.float16))

    def test_against_spec43_reference(self):
        """The fourth channel of clean/k/reference/outstage.py's output, bit for bit, HDR and LDR."""
        from clean.k.reference.outstage import output_stage
        rng = np.random.default_rng(5)
        for render, output, jitter in self.CASES[:4]:
            for hdr in (True, False):
                a = make_alpha(rng, *render)
                a[1, 1] = 70000.0   # beyond FP16: RN16 gives +inf (inf times a zero weight is NaN, as IEEE says)
                x = self._frame(rng, hdr, render, output, jitter, a)
                out = output_stage(x)["out"]
                want = alpha_reference(a, render, output, jitter)
                ok = same16(out[..., 3].astype(np.float16), want)
                self.assertTrue(ok.all(), f"{render} {output} hdr={hdr}: {int((~ok).sum())} alpha values differ")
                # the reference's float32 fourth channel is the exact widening of the FP16 result
                fin = np.isfinite(want)
                np.testing.assert_array_equal(out[..., 3][fin], want.astype(np.float32)[fin])
                # no clamp: negative and above-1 alphas come out
                self.assertLess(float(np.nanmin(want[fin])), -1.0)
                self.assertGreater(float(np.nanmax(want[fin])), 2.0)

    def test_no_alpha_channel_is_one(self):
        from clean.k.reference.outstage import output_stage
        rng = np.random.default_rng(6)
        render, output, jitter = self.CASES[0]
        x = self._frame(rng, True, render, output, jitter, None)
        out = output_stage(x)["out"]
        want = alpha_reference(None, render, output, jitter)
        self.assertTrue(same16(out[..., 3].astype(np.float16), want).all())
        # the four FP16 weights of a pixel sum to 1 up to FP16 rounding of the chain: alpha 1 comes out as 1 within
        # one FP16 step below 1 (the rule stores the chain's result unchanged)
        self.assertTrue(np.all((want >= np.float16(0.9990234375)) & (want <= np.float16(1.0))))

    def test_spatial_only(self):
        """Same colour alpha, everything else different (histories, network output, motion, depth, exposure,
        pre-exposure, RGB, reset): the same output alpha. Over two chained frames with different alpha the second
        frame's alpha is that of its own colour (no temporal carry)."""
        from clean.k.reference.outstage import output_stage
        rng = np.random.default_rng(7)
        render, output, jitter = self.CASES[3]
        a = make_alpha(rng, *render, t=0)
        want = alpha_reference(a, render, output, jitter)
        outs = []
        for k in range(3):
            x = self._frame(rng, bool(k % 2), render, output, jitter, a, reset=k == 2)
            x.update(E_cur=1.5 + k, E_prev=0.5 + k, pre_exposure=0.8 / (k + 1), e1=1.0 + 0.25 * k)
            outs.append(output_stage(x)["out"][..., 3].astype(np.float16))
            self.assertTrue(same16(outs[-1], want).all(), f"variant {k}")
        # two chained frames: frame 1 reads frame 0's histories; its alpha is its own colour's
        x0 = self._frame(rng, True, render, output, jitter, a, reset=True)
        o0 = output_stage(x0)
        b = make_alpha(rng, *render, t=1)
        self.assertGreater(int(np.count_nonzero(a != b)), a.size // 2)
        x1 = self._frame(rng, True, render, output, (0.21, -0.4), b)
        x1.update(L=o0["L"], Hc=o0["Hc"], Hh=o0["Hh"])
        o1 = output_stage(x1)["out"][..., 3].astype(np.float16)
        self.assertTrue(same16(o1, alpha_reference(b, render, output, (0.21, -0.4))).all())
        self.assertFalse(same16(o1, alpha_reference(a, render, output, (0.21, -0.4))).all())

    def test_fp32_carry_equals_fp16_carry(self):
        """Carrying the alpha as FP32 and rounding at the tap gives the bits of carrying RN16 of it (the buffer forms'
        FP32 colour against the FP16 colour of the raw-buffer route and the stored taps)."""
        rng = np.random.default_rng(8)
        render, output, jitter = self.CASES[3]
        a = (make_alpha(rng, *render) * np.float32(1.000123)).astype(np.float32)   # not FP16 values
        self.assertGreater(int(np.count_nonzero(a.astype(np.float16).astype(np.float32) != a)), a.size // 2)
        self.assertTrue(same16(alpha_reference(a, render, output, jitter),
                               alpha_reference(a.astype(np.float16), render, output, jitter)).all())


class TestAlphaSources(unittest.TestCase):
    def test_d3d12_output_kernel(self):
        if not all(os.path.exists(os.path.join(ROOT, "clean", "k", "dx12", "kernels", f)) for f in ("k_in.hlsl", "k_out.hlsl")):
            self.skipTest("the D3D12 input and output ports are not in this tree")
        o, oi = read("k", "dx12", "kernels", "k_out.hlsl"), read("k", "dx12", "hlsl", "k_out.hlsli")
        self.assertNotIn("KO_OUT_ALPHA0", o)
        self.assertNotIn("KO_ALPHA", o)
        # the stored tap's A: RN16 of the colour's fourth value; the bilinear: RN16(w0 a0), then three FP16 fmas
        self.assertIn("tap = uint2(hY | (hCo << 16), hCg | (ko_f2h(c.w) << 16));", oi)
        self.assertIn("hi = q == 0 ? kp_mul_s(t.y, w[q]) : kp_fma_s(t.y, w[q], hi);", o)
        self.assertIn("cbar3 = kp_hi(hi);", o)
        # every output form stores cbar3 unchanged
        self.assertIn("ko_f2h(B) | (kh_u(cbar3) << 16)", o)                                   # RGBA16F rows
        self.assertIn("ko_store_out(x, y, R, G, B, cbar3);", o)                               # the direct texture
        self.assertIn("native_direct_output[uint2((uint)x, (uint)y)] = float4(v, (float)A);", o)
        self.assertIn("asuint(float4(R, G, B, (float)cbar3))", o)                             # dense RGBA f32
        # the direct colour read keeps the texel's alpha (not model M's nd_raw_color_xy, alpha +0)
        tex = oi[oi.index("float4 kio_texel(int x, int y)"):oi.index("float ko_depth_at")]
        self.assertIn("return native_direct_color_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0));", tex)
        self.assertNotIn("return nd_raw_color_xy", tex)
        # model M's shared header is untouched: its raw colour stays alpha +0
        nd = read("dx12", "hlsl", "native_direct.hlsli")
        self.assertIn("return float4(native_direct_color_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).rgb, 0.0f);", nd)

    def test_hip_output_kernel(self):
        k = read("k", "kernels", "k_io.hip")
        self.assertNotIn("KIO_OUT_ALPHA", k)
        self.assertIn("t.A = kh_round(c[3]);", k)
        self.assertIn("const v4h o = {kh_round(R), kh_round(G), kh_round(B), cbar[3]};", k)
        self.assertIn("o[3] = (float)cbar[3];", k)
        self.assertIn("hi = q == 0 ? kh2_mul(vh, w) : kh2_fma(vh, w, hi);", k)
        self.assertIn("out[3] = hi[1];", k)

    def test_runtime(self):
        r, rh, core, api = (read("k", "runtime", "runtime_k.cpp"), read("k", "runtime", "runtime_k.h"),
                            read("runtime", "runtime_core.h"), read("runtime", "wg_runtime.h"))
        self.assertIn("bool out_rgba32f = false;", core)
        self.assertIn("} else if (io && io->out_rgba32f) {\n    op.omode = OUT_RGBA32F;\n    op.opitch = W_;", r)
        self.assertIn("if (!external_io_ && !alloc(&out_, (size_t)W_ * H_ * 16, true)) return false;", r)
        # the test path: an RGB colour has alpha 1, an RGBA colour its own
        self.assertIn("std::vector<float> c(n * 4, 1.0f);", r)
        self.assertIn("int channels = 3);", rh)
        self.assertIn("WG_RT_API int wg_rt_upload_inputs_rgba(", api)

    def test_interop(self):
        c, ch, b, eh, ex = (read("runtime", "interop", "convert.cpp"), read("runtime", "interop", "convert.h"),
                            read("runtime", "interop", "backend.cpp"), read("runtime", "interop", "executor_hip.cpp"),
                            read("runtime", "interop", "executor.h"))
        shader = c[c.index('R"HLSL(') + 7:c.index(')HLSL"')]
        # input passes: the texel's alpha for model K (flag 8), +0 for model M; the compact pass (model K only) keeps it
        self.assertIn("float4 ct = colorTex.Load(p);\n  colorBuf[i] = float4(ct.rgb, (flags & 8u) ? ct.a : 0.0);", shader)
        self.assertIn("colorBuf[i] = colorTex.Load(p);", shader)
        # output passes: model K (flag 4) carries the alpha, model M writes 1
        self.assertIn("uint i = (ka ? 4 : 3) * (id.y * ow + id.x);", shader)
        self.assertIn("[branch] if (ka) a = resultBuf[i + 3];", shader)
        self.assertIn("outTex[id.xy] = float4(c, (flags & 4u) ? t.a : 1.0);", shader)
        self.assertIn("outTex[id.xy] = float4(c, (flags & 4u) ? b.a : 1.0);", shader)
        self.assertNotIn("? 0.0 : 1.0", shader)
        self.assertIn("constexpr uint32_t kMotionOutputRes = 1, kExposureTexture = 2;", c)
        self.assertIn("constexpr uint32_t kOutputAlpha = 4, kInputAlpha = 8;", c)
        self.assertIn("(model_k_ ? kInputAlpha : 0);", c)
        self.assertEqual(c.count("c.flags = model_k_ ? kOutputAlpha : 0;"), 2)   # record_output, record_output_tex
        self.assertIn("d.Buffer.NumElements = (model_k_ ? 4 : 3) * f.output_width * f.output_height;", c)
        self.assertIn("void set_model_k(bool on) { model_k_ = on; }", ch)
        self.assertIn("bool model_k_ = false;", ch)   # model M (and every other Converter user): as before
        self.assertEqual(b.count("conv_.set_model_k("), 1)
        self.assertIn("conv_.set_model_k(model_k_);", b)
        self.assertIn("job.out_rgba32f = model_k_ && !halfOut;", b)
        self.assertIn("sizes[4] = std::max(O * (model_k_ ? 16 : 12), po * oh_);", b)
        self.assertIn("io.out_rgba32f = job.out_rgba32f;", eh)
        self.assertIn("bool out_rgba32f = false;", ex)


if __name__ == "__main__":
    unittest.main()
