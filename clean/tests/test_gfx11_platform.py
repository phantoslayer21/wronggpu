# SPDX-License-Identifier: MIT
"""The gfx11 (RDNA 3) branch of kernels/wg_platform.h, compiled for gfx1100 (no GPU needed; needs a ROCm clang, from
WG_ROCM, else ROCM_PATH): the family macro, the half-wave exchange as one v_permlanex16_b32, the matrix wrappers made
of it and the F16 matrix instruction, clear errors for what gfx11 does not have, and the register gate of the Swin block
kernels. The same sources must still compile for gfx1201 with the old ds_bpermute exchange and the FP8 matrix instruction;
the gfx11-only helpers (wg_xchg16_u) do not exist there.
"""
import os
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from clean.runner import build

ROCM = os.environ.get("WG_ROCM") or os.environ.get("ROCM_PATH", "")

PRELUDE = '#include "wg_platform.h"\n'
KERNEL = 'WG_KERNEL(k)(%s) { const u32 i = wg_lane_id(); %s }\n'


def compile_source(arch, body, args="u32* out, v8f* d", asm=False, flags=()):
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
            cmd = build.device_command(clang, ROCM, src, out, extra=["-I", build.KERNEL_DIR] + list(flags) + (["-S"] if asm else []))
        r = subprocess.run(cmd, capture_output=True, text=True)
        text = ""
        if asm and r.returncode == 0:
            with open(out) as f:
                text = f.read()
        return r.returncode, r.stderr, text


def resource_report(stderr):
    """{kernel: {item: value}} from the compiler's -Rpass-analysis=kernel-resource-usage remarks."""
    report, name = {}, None
    for line in stderr.splitlines():
        m = re.search(r"remark:\s+(.*?) \[-Rpass", line)
        if not m:
            continue
        key, _, value = m.group(1).partition(":")
        if key == "Function Name":
            name = value.strip()
            report[name] = {}
        elif name:
            report[name][key.strip()] = value.strip()
    return report


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
        # gfx12 keeps main's platform text: wg_xor16 is the ds_bpermute exchange, and the gfx11 exchange wg_xchg16_u
        # does not exist there at all (it is gfx11 / emulator only, so the gfx12 source stays main's).
        rc, err, text = compile_source("gfx1201", "d[i][0] = wg_xor16(d[i][0]);", asm=True)
        self.assertEqual(rc, 0, err)
        self.assertEqual(text.count("ds_bpermute_b32"), 1)
        self.assertNotIn("permlane", text)
        rc, err, _ = compile_source("gfx1201", "out[i] = wg_xchg16_u(out[i]);")
        self.assertNotEqual(rc, 0)
        self.assertIn("wg_xchg16_u", err)
        self.assertIn("undeclared identifier", err)

    def test_matrix_wrappers_are_the_f16_instruction_on_gfx11_and_unchanged_on_gfx12(self):
        fp8 = "d[i] = wg_wmma_fp8(v2i{(int)out[i], (int)out[i + 1]}, v2i{(int)out[i + 2], (int)out[i + 3]}, d[i]);"
        f16 = ("d[i] = wg_wmma_f16(__builtin_bit_cast(v8h, v4u{out[i], out[i + 1], out[i + 2], out[i + 3]}),"
               " __builtin_bit_cast(v8h, v4u{out[i + 4], out[i + 5], out[i + 6], out[i + 7]}), d[i]);")
        for name, body, dwords in (("wg_wmma_fp8", fp8, 2), ("wg_wmma_f16", f16, 4)):
            with self.subTest(name=name):
                rc, err, text = compile_source("gfx1100", body, asm=True)
                self.assertEqual(rc, 0, err)
                self.assertEqual(text.count("v_wmma_f32_16x16x16_f16"), 1)
                self.assertEqual(text.count("v_permlanex16_b32"), 2 * dwords)   # a and b, one per dword of the lane
                rc, err, text = compile_source("gfx1201", body, asm=True)
                self.assertEqual(rc, 0, err)
                self.assertEqual(text.count("v_wmma_f32_16x16x16_"), 1)
                self.assertNotIn("permlane", text)

    def test_what_gfx11_lacks_is_a_clear_error(self):
        cases = {"wg_dot4_fp8": ("d[i][0] = wg_dot4_fp8(1, 2, d[i][0]);", "WG_NORM_DOT4"),
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


class SwinGfx11Registers(unittest.TestCase):
    """The recorded gate of the naive FP16 route: the five Swin block kernels (and their clamp-free twins) keep their
    registers, no spills, with the wrappers and the build's own flags (build.py --arch gfx1100 --report). Numbers of
    clang 20: VGPRs 175 / 175 / 206 / 206 / 250 (c64, c64_add, c96, c96_add, c128), 8 / 8 / 7 / 7 / 5 waves per SIMD."""

    KERNELS = ("swin_block_c64", "swin_block_c64_add", "swin_block_c96", "swin_block_c96_add", "swin_block_c128")

    @classmethod
    def setUpClass(cls):
        try:
            clang = build.find_clang(ROCM)
        except SystemExit:
            raise unittest.SkipTest("no ROCm clang")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            out = os.path.join(tmp, "swin.co")
            with patch.object(build, "ARCH", "gfx1100"):
                cmd = build.device_command(clang, ROCM, os.path.join(build.KERNEL_DIR, "swin.hip"), out, report=True,
                                           defines=build.FAMILY_DEFINES["gfx11"]["swin.hip"])
            r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest("cannot compile swin.hip for gfx1100: " + r.stderr[:300])
        cls.report = resource_report(r.stderr)

    def test_no_spills_in_the_block_kernels(self):
        for kernel in self.KERNELS + tuple(k + "_nc" for k in self.KERNELS):
            with self.subTest(kernel=kernel):
                r = self.report[kernel]
                self.assertEqual(int(r["VGPRs Spill"]), 0, r)
                self.assertEqual(int(r["ScratchSize [bytes/lane]"]), 0, r)
                self.assertLessEqual(int(r["VGPRs"]), 256)

    def test_the_faster_quantizer_is_what_spills(self):
        """Why the gfx11 swin.hip build asks for -DWG_SLOW_Q8 (build.FAMILY_DEFINES): without it c128 spills."""
        clang = build.find_clang(ROCM)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            with patch.object(build, "ARCH", "gfx1100"):
                cmd = build.device_command(clang, ROCM, os.path.join(build.KERNEL_DIR, "swin.hip"),
                                           os.path.join(tmp, "swin.co"), report=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:300])
        self.assertGreater(int(resource_report(r.stderr)["swin_block_c128"]["VGPRs Spill"]), 0)


class FastE4m3(unittest.TestCase):
    """The gfx11 FP32 -> E4M3 conversion rounds with (|x| + C) - C and needs IEEE arithmetic (the exhaustive check of the
    result is runner/q8_exhaustive.py, on a GPU)."""

    BODY = "d[i][0] = (float)wg_f32_to_e4m3(d[i][0]); out[i] = (u32)__builtin_bit_cast(u32, wg_q8_to_f16(d[i]));"
    UNSAFE = ("-ffast-math", "-ffinite-math-only", "-fno-honor-nans", "-fassociative-math", "-fno-signed-zeros",
              "-funsafe-math-optimizations", "-Ofast", "-ffp-model=fast", "-freciprocal-math")

    @classmethod
    def setUpClass(cls):
        rc, err, _ = compile_source("gfx1100", "out[i] = i;")
        if rc != 0:
            raise unittest.SkipTest("cannot compile for gfx1100: " + err[:200])

    def test_the_repos_flags_are_ieee(self):
        with patch.object(build, "ARCH", "gfx1100"):
            cmd = build.device_command("clang", ROCM, "a.hip", "a.co", soft_fp8=True)
        for flag in self.UNSAFE:
            self.assertNotIn(flag, cmd)

    def test_fast_math_is_refused(self):
        for flag in ("-ffast-math", "-ffinite-math-only"):
            rc, err, _ = compile_source("gfx1100", self.BODY, flags=[flag])
            self.assertNotEqual(rc, 0, flag)
            self.assertIn("needs IEEE arithmetic", err)
        # gfx12 keeps its own conversion and does not mind
        self.assertEqual(compile_source("gfx1201", "out[i] = wg_f32_to_e4m3(d[i][0]);", flags=["-ffast-math"])[0], 0)

    def test_the_rounding_sum_survives_the_compiler(self):
        """With reassociation allowed the add of the constant would be folded away and the conversion would not round."""
        body = "const v4u r = __builtin_bit_cast(v4u, wg_q8_to_f16(d[i])); out[i] = r[0] ^ r[1] ^ r[2] ^ r[3];"
        rc, err, text = compile_source("gfx1100", body, asm=True)
        self.assertEqual(rc, 0, err)
        self.assertGreaterEqual(text.count("v_add_f32"), 8)
        rc, err, text = compile_source("gfx1100", body, asm=True, flags=["-fassociative-math", "-fno-signed-zeros"])
        self.assertEqual(rc, 0, err)
        self.assertLess(text.count("v_add_f32"), 8)   # what the guard above would catch: the sums are gone

    def test_plain_form_is_still_there(self):
        """-DWG_SLOW_Q8 selects the plain software conversion (the timing baseline); it has no F16 forms."""
        rc, err, _ = compile_source("gfx1100", "out[i] = wg_f32_to_e4m3(d[i][0]);", flags=["-DWG_SLOW_Q8"])
        self.assertEqual(rc, 0, err)
        rc, err, _ = compile_source("gfx1100", self.BODY, flags=["-DWG_SLOW_Q8"])
        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
