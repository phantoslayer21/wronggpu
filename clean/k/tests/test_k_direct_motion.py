# SPDX-License-Identifier: GPL-3.0-only
"""The direct (in-game, texture) forms of model K's input and output stages apply the motion scale and offset as the
buffer forms and the reference do (issue #26). Host checks (no GPU):

- cause, reproduced in FP32: with a non-unit motion scale, the previous direct arithmetic (every texel scaled, RN(ms r)
  + mo in the input stage, one fused mad(ms, r, mo) in the output stage, then the factor of scale 1) gives a different
  warp position than the reference / buffer form (sp = RN(s / RN(RN(1 / n) r)) with the scale folded in, times
  RN(r + b)) for a share of the raw values; the new direct arithmetic (the scale folded into the factor from b1's ms,
  the offset RN(mo / ms) in raw units) gives the buffer form's bits for every value when mo = 0, and whenever RN(mo / ms)
  recovers the buffer drive's raw-unit offset b (mo = RN(ms b)) otherwise;
- sources: the direct ko_motion / kio_motion return the raw texel, the direct factor / offset helpers mirror the host's
  expressions (runtime_k.cpp out uniforms, k_input_uniforms.h, k_frame.hlsli ki_host), the buffer forms are unchanged;
- the full-head direct output stage k_out_pixel_direct is declared (the stage tool's --io direct --head full);
- the output alpha: the colour input's alpha through the current-colour bilinear (issue #26, test_k_alpha).

Run: python -m unittest clean.k.tests.test_k_direct_motion
"""
import os
import re
import unittest

import numpy as np

from clean.k.reference.fp import F32, fma32

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
K = os.path.join(ROOT, "clean", "k")


def read(*p):
    with open(os.path.join(K, *p), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def rn_div(a, b):   # correctly rounded FP32 division (wg_div / ko_div / the host's '/')
    return (np.asarray(a, np.float32) / np.asarray(b, np.float32)).astype(np.float32)


def out_buffer(raw, px, s, b, W, rw):
    """Output stage, buffer form: rhox = RN(rw RN(1 / W)), spx = RN(s / rhox), wx = fma(spx, RN(r + b), px)."""
    rhox = F32(F32(rw) * F32(F32(1) / F32(W)))
    spx = F32(F32(s) / rhox)
    return fma32(spx, (raw + F32(b)).astype(np.float32), px)


def out_direct_old(raw, px, ms, mo, W, rw):
    rhox = F32(F32(rw) * F32(F32(1) / F32(W)))
    spx = F32(F32(1) / rhox)
    return fma32(spx, fma32(F32(ms), raw, F32(mo)) + F32(0), px)


def out_direct_new(raw, px, ms, mo, W, rw):
    rhox = F32(F32(rw) * F32(F32(1) / F32(W)))
    spx = rn_div(F32(1) * F32(ms), rhox)
    off = F32(0) if mo == 0 else F32(F32(0) + rn_div(mo, ms))
    return fma32(spx, (raw + off).astype(np.float32), px)


def in_buffer(raw, s, b, W, rw):
    """Input stage, buffer form: sp = RN(s / RN(RN(1 / W) rw)), mx = RN(sp RN(r + b))."""
    sp = rn_div(s, F32(F32(F32(1) / F32(W)) * F32(rw)))
    return (sp * (raw + F32(b)).astype(np.float32)).astype(np.float32)


def in_direct_old(raw, ms, mo, W, rw):
    sp = rn_div(F32(1), F32(F32(F32(1) / F32(W)) * F32(rw)))
    m = ((F32(ms) * raw).astype(np.float32) + F32(mo)).astype(np.float32)
    return (sp * (m + F32(0))).astype(np.float32)


def in_direct_new(raw, ms, mo, W, rw):
    sp = rn_div(F32(1) * F32(ms), F32(F32(F32(1) / F32(W)) * F32(rw)))
    off = F32(0) if mo == 0 else F32(F32(0) + rn_div(mo, ms))
    return (sp * (raw + off).astype(np.float32)).astype(np.float32)


class TestKDirectMotion(unittest.TestCase):
    CASES = [  # (render w, output W, scale, raw-unit offset b)
        (1706, 2560, 1706.0, 0.0), (1280, 2560, 1280.0, 0.0), (340, 512, 0.3, 0.0), (340, 512, 0.7, 0.0),
        (960, 1440, -960.0, 0.0), (1920, 3840, 0.5, 0.0), (340, 512, 0.3, 0.1), (192, 288, 0.7, -0.2)]

    def test_cause_and_fix_arithmetic(self):
        rng = np.random.default_rng(7)
        n = 200000
        old_diff_total = 0
        for rw, W, s, b in self.CASES:
            # raw motion as a game's texture holds it: about +-40 render px / s, FP16 values (16-bit textures)
            raw = (rng.standard_normal(n) * 40.0 / abs(s)).astype(np.float16).astype(np.float32)
            px = (rng.integers(0, W, n) + 0.5).astype(np.float32)
            mo = float(F32(F32(s) * F32(b)))   # the interop's b1 offset: mo = RN(ms b)
            ob, onew, oold = out_buffer(raw, px, s, b, W, rw), out_direct_new(raw, px, s, mo, W, rw), \
                out_direct_old(raw, px, s, mo, W, rw)
            ib, inew, iold = in_buffer(raw, s, b, W, rw), in_direct_new(raw, s, mo, W, rw), in_direct_old(raw, s, mo, W, rw)
            recovered = b == 0 or float(rn_div(mo, s)) == float(F32(b))
            if recovered:
                self.assertTrue(np.array_equal(ob.view(np.uint32), onew.view(np.uint32)), (rw, W, s, b))
                self.assertTrue(np.array_equal(ib.view(np.uint32), inew.view(np.uint32)), (rw, W, s, b))
            if s not in (1.0, 0.5):   # a power-of-two scale commutes with the roundings
                old_diff_total += int(np.count_nonzero(ob != oold)) + int(np.count_nonzero(ib != iold))
        self.assertGreater(old_diff_total, 0)   # the previous direct arithmetic differs (the cause)

    def test_scale_one_unchanged(self):
        # ms = 1, mo = 0 (the executor's motion pass, output-size motion): the new direct arithmetic is the old one
        rng = np.random.default_rng(3)
        raw = (rng.standard_normal(50000) * 10).astype(np.float32)
        px = (rng.integers(0, 2560, 50000) + 0.5).astype(np.float32)
        self.assertTrue(np.array_equal(out_direct_new(raw, px, 1.0, 0.0, 2560, 1706),
                                       out_direct_old(raw, px, 1.0, 0.0, 2560, 1706)))
        self.assertTrue(np.array_equal(in_direct_new(raw, 1.0, 0.0, 2560, 1706), in_direct_old(raw, 1.0, 0.0, 2560, 1706)))

    def test_sources(self):
        if not all(os.path.exists(os.path.join(ROOT, "clean", "k", "dx12", "kernels", f)) for f in ("k_in.hlsl", "k_out.hlsl")):
            self.skipTest("the D3D12 input and output ports are not in this tree")
        out_i, out_k = read("dx12", "hlsl", "k_out.hlsli"), read("dx12", "kernels", "k_out.hlsl")
        fr = read("dx12", "hlsl", "k_frame.hlsli")
        # the input kernels' shared stage code (k_in_common.hlsli) is included by k_in.hlsl and k_in_legacy.hlsl
        kin = read("dx12", "kernels", "k_in.hlsl") + read("dx12", "hlsl", "k_in_common.hlsli")
        # direct reads: the raw texel (no per-texel scale; not model M's nd_motion_xy)
        self.assertNotIn("nd_motion_xy(", re.sub(r"//[^\n]*", "", out_i))
        for src in (out_i, fr):
            self.assertIn("mvx = r.x;\n    mvy = r.y;", src)
        # factor / offset helpers mirror the host expressions
        self.assertIn("float ko_dsp(float s, float ms, float rho, bool mvlo) { const float sm = s * ms; return mvlo ? ko_div(sm, rho) : sm; }", out_i)
        self.assertIn("float ko_dmo(float o, float mo, float ms) { return mo == 0.0f ? o : o + ko_div(mo, ms); }", out_i)
        self.assertIn("float ki_dsp(float sx, float ms, int n, int r) { return wg_div(sx * ms, wg_div(1.0f, (float)n) * (float)r); }", fr)
        self.assertIn("float ki_dmo(float o, float mo, float ms) { return mo == 0.0f ? o : o + wg_div(mo, ms); }", fr)
        self.assertIn("k.sp_x = wg_div(F.sx, wg_div(1.0f, (float)F.W) * (float)(mvlo ? F.rw : F.W));", fr)   # ki_host
        rt = read("runtime", "runtime_k.cpp")
        self.assertIn("u.rhox = (float)rw * u.invW;", rt)
        self.assertIn("u.spx = fin.sx / u.rhox;  // [MVlo]", rt)
        self.assertIn("k.sp_x = f.sx / ((1.0f / (float)f.W) * (float)((f.flags & 2) ? f.rw : f.W));",
                      read("runtime", "k_input_uniforms.h"))
        # direct forms: b1's ms / mo folded (the KIO_M16 route shares the code with rmsx ..; test_k_m16_alpha)
        for src in (out_k, kin):
            self.assertIn("const bool dfold = true;", src)
            self.assertIn("const float dmsx = nd_msx, dmsy = nd_msy, dmox = nd_mox, dmoy = nd_moy;", src)
        self.assertIn("spx = ko_dsp(f_sx, dmsx, u_rhox, mvlo);", out_k)
        self.assertIn("ofx = ko_dmo(f_mox, dmox, dmsx);", out_k)
        self.assertIn("const float wx = mad(spx, mvx + ofx, px), wy = mad(spy, mvy + ofy, py);", out_k)
        self.assertIn("spx = ki_dsp(F.sx, dmsx, F.W, dmvlo ? F.rw : F.W);", kin)
        self.assertIn("ofx = ki_dmo(F.mox, dmox, dmsx);", kin)
        # the plain buffer form (no KIO_M16): the host's uniforms and WgRtFrame's offset
        self.assertIn("float spx = u_spx, spy = u_spy, ofx = f_mox, ofy = f_moy;", out_k)
        self.assertIn("float spx = k.sp_x, spy = k.sp_y, ofx = F.mox, ofy = F.moy;", kin)

    def test_full_head_direct_port_and_alpha(self):
        if not all(os.path.exists(os.path.join(ROOT, "clean", "k", "dx12", "kernels", f)) for f in ("k_in.hlsl", "k_out.hlsl")):
            self.skipTest("the D3D12 input and output ports are not in this tree")
        out_k = read("dx12", "kernels", "k_out.hlsl")
        line = [l for l in out_k.splitlines() if l.startswith("// @wg-kernel k_out_pixel_direct ")]
        self.assertEqual(len(line), 1)   # declared since the native executor runs model K on D3D12
        if line:
            self.assertEqual(len(line), 1)
            for need in ("entry=k_out_pixel_c", "rs=direct", "KO_COMPACT=0", "WG_DIRECT_TEXTURES=1"):
                self.assertIn(need, line[0])
            c = [l for l in out_k.splitlines() if l.startswith("// @wg-kernel k_out_pixel_c_direct ")][0]
            self.assertEqual(c.replace("k_out_pixel_c_direct", "X").replace("KO_COMPACT=1", "C"),
                             line[0].replace("k_out_pixel_direct", "X").replace("KO_COMPACT=0", "C"))
            # the runtime names a direct form by the suffix (runtime_k.cpp kname): the head form follows k_head
            self.assertIn('kname_buf += "_direct"', read("runtime", "runtime_k.cpp"))
        # the output alpha is cbar3 (the colour input's alpha, issue #26) in both head forms: no constant alpha
        self.assertNotIn("KO_OUT_ALPHA0", out_k)
        self.assertIn("native_direct_output[uint2((uint)x, (uint)y)] = float4(v, (float)A);", out_k)
        self.assertIn("ko_f2h(B) | (kh_u(cbar3) << 16)", out_k)
        self.assertEqual(len(re.findall(r"float4\(v, [01]\.0f\)|0x3c00u << 16|0x0000u << 16", out_k)), 0)


if __name__ == "__main__":
    unittest.main()
