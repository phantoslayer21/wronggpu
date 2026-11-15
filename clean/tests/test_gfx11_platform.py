# SPDX-License-Identifier: MIT
"""The gfx11 (RDNA 3) branch of kernels/wg_platform.h, compiled for gfx1100 (no GPU needed; needs a ROCm clang, from
WG_ROCM, else ROCM_PATH): the family macro, the half-wave exchange as one v_permlanex16_b32, and clear errors for what
gfx11 does not have. The same sources must still compile for gfx1201 with the old ds_bpermute exchange.
"""
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from clean.runner import build

ROCM = os.environ.get("WG_ROCM") or os.environ.get("ROCM_PATH", "")

PRELUDE = '#include "wg_platform.h"\n'
KERNEL = 'WG_KERNEL(k)(%s) { const u32 i = wg_lane_id(); %s }\n'


def compile_source(arch, body, args="u32* out, v8f* d", asm=False):
    """(return code, stderr, assembly or "") of one kernel `k` made from `body`, built for `arch`."""
    try:
        clang = build.find_clang(ROCM)
    except SystemExit:
        raise unittest.SkipTest("no ROCm clang")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        src, out = os.path.join(tmp, "t.hip"), os.path.join(tmp, "t.s" if asm else "t.co")
        with open(src, "w") as f:
            f.write(PRELUDE + KERNEL % (args, body))
        with patch.object(build, "ARCH", arch):
            cmd = build.device_command(clang, ROCM, src, out, extra=["-I", build.KERNEL_DIR] + (["-S"] if asm else []))
        r = subprocess.run(cmd, capture_output=True, text=True)
        text = ""
        if asm and r.returncode == 0:
            with open(out) as f:
                text = f.read()
        return r.returncode, r.stderr, text


class Gfx11Platform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rc, err, _ = compile_source("gfx1100", "out[i] = i;")
        if rc != 0:
            raise unittest.SkipTest("cannot compile for gfx1100: " + err[:200])

    def test_family_macro_is_exact(self):
        guard = "\n#ifndef WG_GFX11\n#error not gfx11\n#endif\n"
        self.assertEqual(compile_source("gfx1100", guard + "out[i] = i;")[0], 0)
        self.assertEqual(compile_source("gfx1101", guard + "out[i] = i;")[0], 0)
        rc, err, _ = compile_source("gfx1201", "\n#ifdef WG_GFX11\n#error gfx11\n#endif\nout[i] = i;")
        self.assertEqual(rc, 0, err)

    def test_software_fp8_is_forced_on_gfx11(self):
        rc, err, text = compile_source("gfx1100", "out[i] = wg_f32_to_e4m3((float)i); d[i] = wg_unpack_fp8(wg_pack_q8(d[i]));",
                                       asm=True)
        self.assertEqual(rc, 0, err)
        self.assertNotIn("v_cvt_pk_fp8", text)
        self.assertNotIn("v_cvt_f32_fp8", text)

    def test_exchange_is_one_permlane_on_gfx11_and_bpermute_on_gfx12(self):
        body = "out[i] = wg_xchg16_u(out[i]); d[i][0] = wg_xor16(d[i][0]);"
        rc, err, text = compile_source("gfx1100", body, asm=True)
        self.assertEqual(rc, 0, err)
        self.assertEqual(text.count("v_permlanex16_b32"), 2)
        self.assertNotIn("ds_bpermute", text)
        rc, err, text = compile_source("gfx1201", body, asm=True)
        self.assertEqual(rc, 0, err)
        self.assertEqual(text.count("ds_bpermute_b32"), 2)
        self.assertNotIn("permlane", text)

    def test_what_gfx11_lacks_is_a_clear_error(self):
        cases = {"wg_wmma_fp8": ("d[i] = wg_wmma_fp8(v2i{1, 2}, v2i{3, 4}, d[i]);", "no FP8 matrix instruction"),
                 "wg_wmma_f16": ("d[i] = wg_wmma_f16(v8h{}, v8h{}, d[i]);", "F16 matrix operand layout"),
                 "wg_dot4_fp8": ("d[i][0] = wg_dot4_fp8(1, 2, d[i][0]);", "WG_NORM_DOT4"),
                 "wg_realtime": ("out[i] = (u32)wg_realtime();", "WG_PROFILE"),
                 "wg_cycles": ("out[i] = wg_cycles();", "WG_PROFILE"),
                 "wg_hwid": ("out[i] = wg_hwid();", "WG_PROFILE"),
                 "wg_drain": ("wg_drain(); out[i] = 1;", "WG_PROFILE")}
        for name, (body, message) in cases.items():
            with self.subTest(name=name):
                rc, err, _ = compile_source("gfx1100", body)
                self.assertNotEqual(rc, 0)
                self.assertIn("declared with 'error' attribute", err)
                self.assertIn(message, err)
                rc, err, _ = compile_source("gfx1201", body)   # the same call is fine where the instruction exists
                self.assertEqual(rc, 0, err)

    def test_dead_code_with_those_calls_still_compiles(self):
        """swin.hip keeps `if (WG_NORM_DOT4 && ...)` branches with the macro 0: the error is only for a surviving call."""
        rc, err, _ = compile_source("gfx1100", "if (0 && out[i] == 7u) d[i][0] = wg_dot4_fp8(1, 2, d[i][0]); out[i] = i;")
        self.assertEqual(rc, 0, err)


if __name__ == "__main__":
    unittest.main()
