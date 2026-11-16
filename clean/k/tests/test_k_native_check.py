# SPDX-License-Identifier: MIT
"""clean/k/dx12/k_native_check.py on the CPU: --prepare writes kernel_replay chains for both forms over the same input
files, --emulate runs them through the emulation of the kernel text (clean/dx12/emu), --compare finds them equal and
inside the limits, catches the negative control, and fails on one flipped output byte. A subset of the stages and
sizes (the full set is the GPU run's). Needs Linux and clang++ (as test_k_native_g11_emu.py)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

from clean.k.dx12 import k_native_check as KC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "clean", "dx12"))
from emu import hlsl_emu as H  # noqa: E402


class TestNativeCheck(unittest.TestCase):
    def test_prepare_emulate_compare(self):
        if not sys.platform.startswith("linux"):
            self.skipTest("the lane emulator needs Linux (ucontext)")
        try:
            H.compiler()
        except RuntimeError as e:
            self.skipTest(str(e))
        d = tempfile.mkdtemp(prefix="wg_k_native_check_", dir=os.environ.get("WG_TEST_TMP"))
        sub = {k: KC.KERNELS[k] for k in ("k_stage_e2", "k_stage2_d1", "k_stage2_d0_hp", "k_stage_e2__rodd")}
        with mock.patch.object(KC, "KERNELS", sub), mock.patch.object(KC, "SIZES", KC.SIZES[:3]):
            idx = KC.prepare(d, [0])
            self.assertEqual(len(idx), 12)
            for form in ("native", "shortcut"):
                with open(os.path.join(d, f"{form}.txt")) as fh:
                    txt = fh.read()
                self.assertEqual(txt.count("chain "), 12 + (form == "native"))   # + the negative control
            with open(os.path.join(d, "native.txt")) as fh:
                self.assertIn("k_stage2_d0_hp__g11", fh.read())
            KC.emulate(d, os.path.join(d, "emu"))
        r = KC.compare(d)
        self.assertTrue(r["pass"], r["failures"])
        self.assertTrue(r["negative_caught"])
        self.assertEqual(r["differ"], 0)
        # a single flipped bit in one native output must fail the comparison
        f = os.path.join(d, "native_s0_k_stage_e2_1.block.d3d12")
        with open(f, "rb") as fh:
            data = bytearray(fh.read())
        data[10] ^= 1
        with open(f, "wb") as fh:
            fh.write(bytes(data))
        r = KC.compare(d)
        self.assertFalse(r["pass"])
        self.assertEqual(r["differ"], 1)


if __name__ == "__main__":
    unittest.main()
