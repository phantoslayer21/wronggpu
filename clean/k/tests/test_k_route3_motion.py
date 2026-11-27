# SPDX-License-Identifier: GPL-3.0-only
"""Model K on the interop's FP32 input-conversion pass (the HIP executor's route when the colour or motion texture is
not FP16): the pass writes the raw motion texels for model K and the backend passes motion_scale and the raw-unit
offset in WgRtFrame, so K folds the scale into its motion factor as the reference / buffer form (issue #26).
Host checks (no GPU, no HIP):

- arithmetic, in FP32: the previous route (the pass's per-texel ms raw + mo, as one mad or two roundings, then K with
  scale 1) differs from the buffer form for a share of the raw values; the new route (the pass with scale 1, offset 0:
  mad(1, raw, 0); K with mv_scale ms, mv_bias RN(mo / ms)) gives the buffer form's bits on every value (signed zeros,
  infinities and NaN included: 1 raw is exact, + 0 maps -0 to +0, which K's raw + b maps to the same value);
- sources: the Converter's run-time HLSL carries the alpha for model K only (issue #26: every change is under a flag
  only model K sets or in a pass only model K runs; model M's passes compute what they did, with the same constants),
  the raw-motion constants are model K's only, and the backend's K-only
  branch;
- the HIP executor job's WgRtFrame reaches K's frame input (fin.sx = mv_scale, fin.mox = mv_bias).

Run: python -m unittest clean.k.tests.test_k_route3_motion
"""
import os
import subprocess
import unittest

import numpy as np

from clean.k.reference.fp import F32, fma32
from clean.k.tests.test_k_direct_motion import in_buffer, out_buffer, rn_div

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CLEAN = os.path.join(ROOT, "clean")


def read(*p):
    with open(os.path.join(CLEAN, *p), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def k_out(m, px, s, b, W, rw):   # K's output stage on the dense motion m with WgRtFrame (s, b): the buffer form
    return out_buffer(m, px, s, b, W, rw)


def k_in(m, s, b, W, rw):
    return in_buffer(m, s, b, W, rw)


class TestKRoute3Motion(unittest.TestCase):
    CASES = [(1706, 2560, 1706.0, 0.0), (1280, 2560, 1280.0, 0.0), (340, 512, 0.3, 0.0), (340, 512, 0.7, 0.0),
             (960, 1440, -960.0, 0.0), (1920, 3840, 0.5, 0.0), (340, 512, 0.3, 0.1), (192, 288, 0.7, -0.2)]

    def test_arithmetic(self):
        rng = np.random.default_rng(23)
        n = 200000
        old_diff = 0
        for rw, W, s, b in self.CASES:
            # raw FP32 motion (RG32F textures: the FP32 route), with signed zeros, infinities and NaN mixed in
            raw = (rng.standard_normal(n) * 40.0 / abs(s)).astype(np.float32)
            raw[:64] = -0.0
            raw[64:72] = np.float32(np.inf)
            raw[72:80] = -np.float32(np.inf)
            raw[80:88] = np.float32(np.nan)
            px = (rng.integers(0, W, n) + 0.5).astype(np.float32)
            mo = float(F32(F32(s) * F32(b)))                     # the game's offset in render pixels after the scale
            bk = 0.0 if mo == 0 else float(rn_div(mo, s))        # the backend's raw-unit offset RN(mo / ms)
            # new route: the pass with scale 1, offset 0 (one mad; 1 raw exact), then K with (ms, bk)
            m_new = fma32(F32(1), raw, F32(0))
            with np.errstate(invalid="ignore"):
                ob, ib = k_out(raw, px, s, bk, W, rw), k_in(raw, s, bk, W, rw)
                on, inn = k_out(m_new, px, s, bk, W, rw), k_in(m_new, s, bk, W, rw)
                self.assertTrue(np.array_equal(ob.view(np.uint32), on.view(np.uint32)), (rw, W, s, b))
                self.assertTrue(np.array_equal(ib.view(np.uint32), inn.view(np.uint32)), (rw, W, s, b))
                if b == 0 or bk == float(F32(b)):   # and the buffer drive's own b (when RN(mo / ms) recovers it)
                    self.assertTrue(np.array_equal(out_buffer(raw, px, s, b, W, rw).view(np.uint32), on.view(np.uint32)))
                # old route: the pass's per-texel conversion (one mad, d3dcompiler's msx * m + mox; or two roundings),
                # then K with scale 1, offset 0
                fin = np.isfinite(raw)
                for m_old in (fma32(F32(s), raw, F32(mo)), ((F32(s) * raw).astype(np.float32) + F32(mo)).astype(np.float32)):
                    old_diff += int(np.count_nonzero((ob != k_out(m_old, px, 1.0, 0.0, W, rw)) & fin))
                    old_diff += int(np.count_nonzero((ib != k_in(m_old, 1.0, 0.0, W, rw)) & fin))
        self.assertGreater(old_diff, 0)

    # issue #26 (the output alpha follows the colour input's alpha): the only changes of the run-time HLSL, as (old text, new text, occurrences). Flags 4 / 8 are set for model K only (Converter::set_model_k);
    # the compact pass inputs16 is model K's own.
    ALPHA_CARRY = [
        ("  colorBuf[i] = float4(colorTex.Load(p).rgb, 0.0);\n  depthBuf[i] = depthTex.Load(p);\n  int3 mp = p;\n"
         "  if (flags & 1u) {  // motion at output size: the texel under the render pixel's centre\n"
         "    mp = int3(min(uint((id.x + 0.5) * mw / rw), mw - 1), min(uint((id.y + 0.5) * mh / rh), mh - 1), 0);\n"
         "  }\n  float2 m = motionTex.Load(mp).xy;\n",
         "  // flags & 8u (model K): the texel's alpha, unmodified, as the colour's fourth value (model K's output alpha "
         "follows\n  // it; a format without an alpha channel loads 1). Else +0 (model M).\n"
         "  float4 ct = colorTex.Load(p);\n  colorBuf[i] = float4(ct.rgb, (flags & 8u) ? ct.a : 0.0);\n"
         "  depthBuf[i] = depthTex.Load(p);\n  int3 mp = p;\n"
         "  if (flags & 1u) {  // motion at output size: the texel under the render pixel's centre\n"
         "    mp = int3(min(uint((id.x + 0.5) * mw / rw), mw - 1), min(uint((id.y + 0.5) * mh / rh), mh - 1), 0);\n"
         "  }\n  float2 m = motionTex.Load(mp).xy;\n", 1),
        ("// exposure as in inputs. The network side applies m = ms raw + mo when it reads the motion.\n",
         "// exposure as in inputs. The colour keeps the texel's alpha, unmodified (model K's output alpha follows it; "
         "R11G11B10\n// has none and loads 1). The network side applies m = ms raw + mo when it reads the motion.\n", 1),
        ("  colorBuf[i] = float4(colorTex.Load(p).rgb, 0.0);\n", "  colorBuf[i] = colorTex.Load(p);\n", 1),
        ("  uint i = 3 * (id.y * ow + id.x);\n  float3 c = float3(resultBuf[i], resultBuf[i + 1], resultBuf[i + 2]);\n"
         "  if (srgb) c = to_srgb(c);\n  outTex[id.xy] = float4(c, (flags & 4u) ? 0.0 : 1.0);\n",
         "  // flags & 4u (model K): the result holds 4 x f32 per pixel, the fourth being the output alpha (the colour "
         "input's\n  // alpha through model K's output stage), stored as it is. Else 3 x f32 and alpha 1 (model M).\n"
         "  bool ka = (flags & 4u) != 0;\n  uint i = (ka ? 4 : 3) * (id.y * ow + id.x);\n"
         "  float3 c = float3(resultBuf[i], resultBuf[i + 1], resultBuf[i + 2]);\n  float a = 1.0;\n"
         "  [branch] if (ka) a = resultBuf[i + 3];\n  if (srgb) c = to_srgb(c);\n  outTex[id.xy] = float4(c, a);\n", 1),
        ("  float3 c = proxyTex.Load(int3(id.xy, 0)).rgb;\n  if (srgb) c = to_srgb(c);\n"
         "  outTex[id.xy] = float4(c, (flags & 4u) ? 0.0 : 1.0);\n",
         "  float4 t = proxyTex.Load(int3(id.xy, 0));\n  float3 c = t.rgb;\n  if (srgb) c = to_srgb(c);\n"
         "  outTex[id.xy] = float4(c, (flags & 4u) ? t.a : 1.0);  // model K: the result texture's alpha, as it is\n", 1),
        ("  float3 c00 = colorTex.Load(int3(p0.x, p0.y, 0)).rgb, c10 = colorTex.Load(int3(p1.x, p0.y, 0)).rgb;\n"
         "  float3 c01 = colorTex.Load(int3(p0.x, p1.y, 0)).rgb, c11 = colorTex.Load(int3(p1.x, p1.y, 0)).rgb;\n"
         "  float3 c = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);\n  if (srgb) c = to_srgb(c);\n"
         "  outTex[id.xy] = float4(c, (flags & 4u) ? 0.0 : 1.0);\n",
         "  float4 c00 = colorTex.Load(int3(p0.x, p0.y, 0)), c10 = colorTex.Load(int3(p1.x, p0.y, 0));\n"
         "  float4 c01 = colorTex.Load(int3(p0.x, p1.y, 0)), c11 = colorTex.Load(int3(p1.x, p1.y, 0));\n"
         "  float4 b = lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);\n  float3 c = b.rgb;\n"
         "  if (srgb) c = to_srgb(c);\n"
         "  // model K (flags & 4u): the colour's alpha through the same bilinear (a frame without a network result "
         "follows the\n  // colour input's alpha too); else alpha 1 (model M)\n"
         "  outTex[id.xy] = float4(c, (flags & 4u) ? b.a : 1.0);\n", 1),
    ]

    def test_converter_hlsl_carries_the_alpha_for_model_k_only(self):
        """The run-time HLSL has each alpha-carry text above (every one under a flag only model K sets, or in a pass only
        model K runs), none of the texts it replaces (model M's constants: colour alpha +0 in, alpha 1 out), and the
        motion pass still computes m = ms raw + mo."""
        c = read("runtime", "interop", "convert.cpp")
        shader = c[c.index('R"HLSL(') + 7:c.index(')HLSL"')]
        for old, new, n in self.ALPHA_CARRY:
            self.assertEqual(shader.count(new), n, new)
            self.assertNotIn(old, shader)
        self.assertIn("motionBuf[i] = float2(msx * m.x + mox, msy * m.y + moy);", shader)

    def test_sources(self):
        c, h, b = read("runtime", "interop", "convert.cpp"), read("runtime", "interop", "convert.h"), \
            read("runtime", "interop", "backend.cpp")
        for line in ("c.msx = raw_motion ? 1.0f : f.motion_scale[0];", "c.msy = raw_motion ? 1.0f : f.motion_scale[1];",
                     "c.mox = raw_motion ? 0.0f : f.motion_offset[0];", "c.moy = raw_motion ? 0.0f : f.motion_offset[1];",
                     "record_inputs_pass(list, f, create_flags, b, false, raw_motion);"):
            self.assertIn(line, c)
        self.assertIn("const InputBuffers& b, bool compact, bool raw_motion = false);", h)
        self.assertIn("bool raw_motion = false);", h)
        self.assertIn("const bool kraw = model_k_ && f.motion_scale[0] != 0.0f && f.motion_scale[1] != 0.0f;", b)
        self.assertIn("job.frame.mv_scale[k] = f.motion_scale[k];", b)
        self.assertIn("job.frame.mv_bias[k] = f.motion_offset[k] == 0.0f ? 0.0f : f.motion_offset[k] / f.motion_scale[k];", b)
        self.assertIn("conv_.record_inputs(list, f, flags_, in, kraw);", b)
        self.assertEqual(b.count("conv_.record_inputs("), 1)
        # the frame the HIP executor passes reaches K's frame input
        rt = read("k", "runtime", "runtime_k.cpp")
        self.assertIn("fin.sx = fr.mv_scale[0];", rt)
        self.assertIn("fin.mox = fr.mv_bias[0];", rt)


if __name__ == "__main__":
    unittest.main()
