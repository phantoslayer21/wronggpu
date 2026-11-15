# SPDX-License-Identifier: MIT
"""CPU checks of the conversion-fold guard: clean/runtime/fold_probe.h and fold_selftest.h are what their generators
write (inputs and expected bytes from exact rationals), the self-test frame holds its value classes on every encode
site, and build_dx12.py's needs= / isa= declarations parse (and refuse bad values), the literal-tied ISA count included.
The GPU side (the probe kernel against these bytes, the self-test of the real post ports, the two-roundings mutant) is
a GPU-slot check."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.dx12 import build_dx12 as B  # noqa: E402


class FoldProbeTest(unittest.TestCase):
    def test_header_is_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fold_probe.h")
            subprocess.run([sys.executable, os.path.join(ROOT, "clean", "runtime", "gen_fold_probe.py"), out], check=True,
                           capture_output=True)
            with open(out, encoding="utf-8") as f:
                want = f.read()
        with open(os.path.join(ROOT, "clean", "runtime", "fold_probe.h"), encoding="utf-8") as f:
            self.assertEqual(f.read(), want)

    def test_selftest_header_is_generated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "fold_selftest.h")
            subprocess.run([sys.executable, os.path.join(ROOT, "clean", "runtime", "gen_fold_selftest.py"), out],
                           check=True, capture_output=True)
            with open(out, encoding="utf-8") as f:
                want = f.read()
        with open(os.path.join(ROOT, "clean", "runtime", "fold_selftest.h"), encoding="utf-8") as f:
            self.assertEqual(f.read(), want)

    def test_selftest_frame_classes(self):
        """Every one of the six encode sites (row of the pair, channel) holds FP32-product ties of both sides that a
        second rounding gets wrong, the zero product and the floor."""
        sys.path.insert(0, os.path.join(ROOT, "clean", "runtime"))
        import gen_fold_selftest as G
        colour, _, _ = G.build()
        floor = G.Fraction(G.f32(G.bits32(1e-4)))
        for site in range(G.SITES):
            row, ch = divmod(site, 3)
            vals = [colour[3 * (y * G.W + x) + ch] for y in range(row, G.H, 2) for x in range(G.W)]
            sides = {"up": 0, "down": 0}
            for cb in vals:
                r = G.classify(G.f32(cb))
                if r and r[2]:
                    sides[r[1]] += 1
            self.assertGreaterEqual(min(sides.values()), 40, (site, sides))
            ms = [G.m_of(G.f32(cb)) for cb in vals]
            self.assertIn(G.Fraction(1), ms)                          # x = 0: the zero product
            self.assertTrue(any(m < floor for m in ms) and any(floor <= m < 2 * floor for m in ms))

    def test_isa_rules(self):
        self.assertEqual(B.parse_isa_rules("v_fma_mixlo_f16+v_fma_mixhi_f16>=6,s_round_mode==0", "t"),
                         [([("v_fma_mixlo_f16", None), ("v_fma_mixhi_f16", None)], ">=", 6),
                          ([("s_round_mode", None)], "==", 0)])
        self.assertEqual(B.parse_isa_rules("v_fma_mixlo_f16@3dddf184+v_fma_mixhi_f16@3dddf184==6", "t"),
                         [([("v_fma_mixlo_f16", "3dddf184"), ("v_fma_mixhi_f16", "3dddf184")], "==", 6)])
        with self.assertRaises(SystemExit):
            B.parse_isa_rules("v_fma_mixlo_f16>6", "t")
        with self.assertRaises(SystemExit):
            B.parse_isa_rules("v_fma_mixlo_f16@3dddf18==6", "t")

    def test_isa_count(self):
        lines = ["	v_fma_mixlo_f16 v3, v3, 0xf184, 0                          // 000000008224: CC210003 0201FF03 3DDDF184",
                 "	v_fma_mixhi_f16 v10, v2, 0xf184, 0                         // 000000008244: CC22000A 0201FF02 3DDDF184",
                 "	v_fma_mixlo_f16 v3, v3, 0xf184, v4                         // 000000008254: CC210003 0411FF03 3DDDF184",
                 "	v_fma_mixlo_f16 v3, v3, 0xf184, 0 op_sel_hi:[1,0,0]        // 000000008264: CC210003 0201FF03 3DDDF184",
                 "	v_fma_mixlo_f16 v3, v3, 0xa8c1, 0                          // 000000008274: CC210003 0201FF03 BF38A8C1",
                 "	v_mul_f32_e32 v1, 0x3dddf184, v1                           // 0000000009DC: 100202FF 3DDDF184"]
        self.assertEqual(B.isa_count(lines, "v_fma_mixlo_f16", None), 4)
        self.assertEqual(B.isa_count(lines, "v_fma_mixlo_f16", "3dddf184"), 1)   # no other addend, no op_sel, literal
        self.assertEqual(B.isa_count(lines, "v_fma_mixhi_f16", "3dddf184"), 1)

    def test_post_ports(self):
        decls = [d for d in B.kernel_decls(os.path.join(ROOT, "clean", "dx12", "kernels", "l_post.hlsl"))
                 if d["name"] == "l_post_recon_p12"]
        self.assertEqual(sorted(d["needs"] for d in decls), ["fold", "nofold"])
        fold = [d for d in decls if d["needs"] == "fold"][0]
        self.assertEqual(fold["defines"].get("LP_ENCFOLD"), "1")
        self.assertIn("@3dddf184", fold["isa"])   # the fold's sites: kEncK as the literal, addend 0
        nofold = [d for d in decls if d["needs"] == "nofold"][0]
        self.assertNotIn("LP_ENCFOLD", nofold["defines"])
        self.assertEqual(nofold["file"], "l_post_recon_p12__nofold")

    def test_bad_needs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "k.hlsl")
            with open(p, "w") as f:
                f.write("// @wg-kernel x co=- block=32 wave=32 args=p needs=sometimes\n")
            with self.assertRaises(SystemExit):
                B.kernel_decls(p)


if __name__ == "__main__":
    unittest.main()
