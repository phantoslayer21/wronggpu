# SPDX-License-Identifier: MIT
"""Model K: the interop's raw-buffer motion route (KIO_M16) applies the motion scale and offset as the buffer forms and
the reference do (issue #26). Host checks (no GPU, no HIP run):

- KIO_M16 arithmetic, in FP32: the previous form (every texel ms r + mo, two roundings, then the factor of scale 1 that
  WgRtFrame's mv_scale 1 gives) differs from the reference / buffer form (sp = RN(s / RN(RN(1/n) r)) with the scale
  folded in, times RN(r + b)) for a share of the raw values; the new form (the raw texel, the scale folded into the
  factor, the offset RN(mo / ms) in raw units) gives the buffer form's bits whenever that offset is recovered (always
  for mo = 0). The same FP32 operations in HIP (k_io.hip, contract off, correctly rounded division) and D3D12;
- sources: HIP k_io.hip and the D3D12 ports read the raw FP16 texel on KIO_M16 and fold ms / mo with the host's
  expressions at both use sites (input ki_fine, output k_out_pixel);
- the Converter's run-time HLSL compiles with d3dcompiler_47 (cs_5_0), every pass. (The output alpha follows the colour input's alpha, issue #26; its checks are
  clean/k/tests/test_k_alpha.py.)

Run: python -m unittest clean.k.tests.test_k_m16_alpha
"""
import ctypes
import os
import re
import sys
import unittest

import numpy as np

from clean.k.reference.fp import F32, fma32
from clean.k.tests.test_k_direct_motion import (in_buffer, in_direct_new, in_direct_old, out_buffer, out_direct_new,
                                               rn_div)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CLEAN = os.path.join(ROOT, "clean")


def read(*p):
    with open(os.path.join(CLEAN, *p), encoding="utf-8") as f:
        return f.read().replace("\r\n", "\n")


def out_m16_old(raw, px, ms, mo, W, rw):
    """Output stage, previous KIO_M16 form: m = RN(RN(ms r) + mo) per texel (HIP contract off: two roundings), the factor
    of WgRtFrame's scale 1: spx = RN(1 / rhox), wx = fma(spx, RN(m + 0), px)."""
    rhox = F32(F32(rw) * F32(F32(1) / F32(W)))
    spx = F32(F32(1) / rhox)
    m = ((F32(ms) * raw).astype(np.float32) + F32(mo)).astype(np.float32)
    return fma32(spx, (m + F32(0)).astype(np.float32), px)


def in_m16_old(raw, ms, mo, W, rw):
    return in_direct_old(raw, ms, mo, W, rw)   # the same two roundings, then sp of scale 1


class TestKM16Motion(unittest.TestCase):
    CASES = [(1706, 2560, 1706.0, 0.0), (1280, 2560, 1280.0, 0.0), (340, 512, 0.3, 0.0), (340, 512, 0.7, 0.0),
             (960, 1440, -960.0, 0.0), (1920, 3840, 0.5, 0.0), (340, 512, 0.3, 0.1), (192, 288, 0.7, -0.2)]

    def test_arithmetic(self):
        rng = np.random.default_rng(11)
        n = 200000
        old_diff = 0
        for rw, W, s, b in self.CASES:
            # raw FP16 motion as the route's RG16F copies hold it
            raw = (rng.standard_normal(n) * 40.0 / abs(s)).astype(np.float16).astype(np.float32)
            px = (rng.integers(0, W, n) + 0.5).astype(np.float32)
            mo = float(F32(F32(s) * F32(b)))
            ob, ib = out_buffer(raw, px, s, b, W, rw), in_buffer(raw, s, b, W, rw)
            if b == 0 or float(rn_div(mo, s)) == float(F32(b)):
                # the new KIO_M16 form is the direct forms' arithmetic with (rmsx, rmox) for (ms, mo)
                self.assertTrue(np.array_equal(ob.view(np.uint32), out_direct_new(raw, px, s, mo, W, rw).view(np.uint32)))
                self.assertTrue(np.array_equal(ib.view(np.uint32), in_direct_new(raw, s, mo, W, rw).view(np.uint32)))
            old_diff += int(np.count_nonzero(ob != out_m16_old(raw, px, s, mo, W, rw)))
            old_diff += int(np.count_nonzero(ib != in_m16_old(raw, s, mo, W, rw)))
        self.assertGreater(old_diff, 0)

    def test_hip_sources(self):
        h = read("k", "kernels", "k_io.hip")
        self.assertIn("#pragma clang fp contract(off)", h)
        self.assertIn("WG_FN float kio_m16_sp(float s, float ms, float den) { return (s * ms) / den; }", h)
        self.assertIn("WG_FN float kio_m16_mo(float o, float mo, float ms) { return mo == 0.0f ? o : o + mo / ms; }", h)
        # raw reads on KIO_M16 (input kio_motion, output ko_motion): no per-texel scale left
        self.assertEqual(len(re.findall(r"\*mvx = \(float\)r\[0\];", h)), 2)
        self.assertNotIn("F.rmsx * (float)r[0]", h)
        # input: ki_host's sp_x with s = sx ms; output: out_uniforms' spx with s = sx ms
        self.assertIn("k.sp_x = F.sx / ((1.0f / (float)F.W) * (float)(mvlo ? F.rw : F.W));", h)
        self.assertIn("spx = kio_m16_sp(F.sx, F.rmsx, (1.0f / (float)F.W) * (float)(mvlo ? F.rw : F.W));", h)
        self.assertIn("ofx = kio_m16_mo(F.mox, F.rmox, F.rmsx);", h)
        self.assertIn("const float mx = spx * (mvx + ofx), my = spy * (mvy + ofy);", h)
        self.assertIn("spx = mvlo ? kio_m16_sp(F.sx, F.rmsx, U.rhox) : F.sx * F.rmsx;", h)
        self.assertIn("const float wx = kf_fma(spx, mvx + ofx, px), wy = kf_fma(spy, mvy + ofy, py);", h)
        rt = read("k", "runtime", "runtime_k.cpp")
        self.assertIn("u.spx = fin.sx / u.rhox;  // [MVlo]", rt)
        self.assertIn("u.rhox = (float)rw * u.invW;", rt)

    def test_d3d12_sources(self):
        if not all(os.path.exists(os.path.join(ROOT, "clean", "k", "dx12", "kernels", f)) for f in ("k_in.hlsl", "k_out.hlsl")):
            self.skipTest("the D3D12 input and output ports are not in this tree")
        o, oi = read("k", "dx12", "kernels", "k_out.hlsl"), read("k", "dx12", "hlsl", "k_out.hlsli")
        fr = read("k", "dx12", "hlsl", "k_frame.hlsli")
        ki = read("k", "dx12", "kernels", "k_in.hlsl") + read("k", "dx12", "hlsl", "k_in_common.hlsli")
        self.assertNotIn("f_rmsx * r0", oi)
        self.assertNotIn("F.rmsx * wg_h2f_lo(r)", fr)
        self.assertIn("mvx = wg_h2f(wg_ld16(f_motion, i * 2u));", oi)
        self.assertIn("mvx = wg_h2f_lo(r);", fr)
        self.assertIn("const bool dfold = (f_flags & KIO_M16) != 0;", o)
        self.assertIn("const float dmsx = f_rmsx, dmsy = f_rmsy, dmox = f_rmox, dmoy = f_rmoy;", o)
        self.assertIn("const float wx = mad(spx, mvx + ofx, px), wy = mad(spy, mvy + ofy, py);", o)
        self.assertIn("const bool dfold = (F.flags & KIO_M16) != 0;", ki)
        self.assertIn("const float dmsx = F.rmsx, dmsy = F.rmsy, dmox = F.rmox, dmoy = F.rmoy;", ki)
        self.assertIn("const float mx = spx * (mvx + ofx), my = spy * (mvy + ofy);", ki)


class TestConverterShader(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "d3dcompiler_47 (Windows)")
    def test_converter_hlsl_compiles(self):
        c = read("runtime", "interop", "convert.cpp")
        shader = c[c.index('R"HLSL(') + 7:c.index(')HLSL"')].encode()
        d3dc = ctypes.WinDLL("d3dcompiler_47.dll")

        class Macro(ctypes.Structure):
            _fields_ = [("Name", ctypes.c_char_p), ("Definition", ctypes.c_char_p)]

        passes = dict(re.findall(r'"(\w+)", "(PASS_\w+)"', c.replace("\n", " ")))
        entries = re.search(r"const char\* entries\[\d+\] = \{([^}]*)\}", c).group(1)
        names = re.findall(r'"(\w+)"', entries)
        pass_defs = re.findall(r'"(PASS_\w+)"', re.search(r"const char\* passes\[\d+\] = \{([^}]*)\}", c).group(1))
        todo = list(zip(names, pass_defs)) + [("outtex", "PASS_OUTTEX")]
        self.assertIn(("output", "PASS_OUTPUT"), todo)
        self.assertIn(("fallback", "PASS_FALLBACK"), todo)
        for entry, define in todo:
            macros = (Macro * 2)(Macro(define.encode(), b"1"), Macro(None, None))
            code, err = ctypes.c_void_p(), ctypes.c_void_p()
            hr = d3dc.D3DCompile(shader, len(shader), b"convert", macros, None, entry.encode(), b"cs_5_0",
                                 ctypes.c_uint(1 << 15), 0, ctypes.byref(code), ctypes.byref(err))
            msg = ""
            if err.value:
                vt = ctypes.cast(ctypes.cast(err, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vt[3])(err)
                size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vt[4])(err)
                msg = ctypes.string_at(ptr, size).decode(errors="replace")
                ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vt[2])(err)
            if code.value:
                vt = ctypes.cast(ctypes.cast(code, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vt[2])(code)
            self.assertEqual(hr, 0, f"{entry} ({define}): {msg}")
        _ = passes


if __name__ == "__main__":
    unittest.main()
