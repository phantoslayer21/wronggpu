# SPDX-License-Identifier: MIT
"""CPU checks of the conversion-fold guard: clean/runtime/fold_probe.h is what clean/runtime/gen_fold_probe.py writes
(inputs and expected bytes from exact rationals), and build_dx12.py's needs= / isa= declarations parse (and refuse bad
values). The GPU side (the probe kernel against these bytes, its two-roundings mutant) is a GPU-slot check."""
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

    def test_isa_rules(self):
        self.assertEqual(B.parse_isa_rules("v_fma_mixlo_f16+v_fma_mixhi_f16>=6,s_round_mode==0", "t"),
                         [(["v_fma_mixlo_f16", "v_fma_mixhi_f16"], ">=", 6), (["s_round_mode"], "==", 0)])
        with self.assertRaises(SystemExit):
            B.parse_isa_rules("v_fma_mixlo_f16>6", "t")

    def test_post_ports(self):
        decls = [d for d in B.kernel_decls(os.path.join(ROOT, "clean", "dx12", "kernels", "l_post.hlsl"))
                 if d["name"] == "l_post_recon_p12"]
        self.assertEqual(sorted(d["needs"] for d in decls), ["fold", "nofold"])
        fold = [d for d in decls if d["needs"] == "fold"][0]
        self.assertEqual(fold["defines"].get("LP_ENCFOLD"), "1")
        self.assertTrue(fold["isa"])
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
