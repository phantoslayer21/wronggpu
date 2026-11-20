# SPDX-License-Identifier: MIT
"""native_driver_probe_f16 (kernels/native_game_helpers.hlsl), the RDNA 3 form of the native device's driver-channel
probe, built by build_dx12.py --gfx 11 and run headless on the card (same environment as test_gfx11_helpers.py, plus
WG_GFX11_NATIVE_CXX = a C++17 compiler for the host, default c++): its operands come from runtime/wmma_probe.h
(namespace gfx11, wmma_probe_dump), and the results must equal the CPU model's within its tolerance (the hardware's sum differs from the exact one in the last bit). The probe with the gfx12
layout in the buffer must fail. The gfx12 probe kernel (native_driver_probe, FP16 and FP8) is not built for gfx11."""
import json
import os
import shutil
import subprocess
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_gfx11_helpers as H  # noqa: E402  (also puts clean/dx12 on the path)
import build_dx12 as B  # noqa: E402

RUNTIME = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runtime")
CXX = os.environ.get("WG_GFX11_NATIVE_CXX", "c++")


@unittest.skipUnless(H.HAVE_CARD and shutil.which(CXX), "no card / C++ compiler configured (see test_gfx11_helpers.py)")
class TestGfx11Probe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        H.Card.setup()
        dump = os.path.join(H.WORK, "wmma_probe_dump")
        subprocess.run([CXX, "-std=c++17", "-O1", "-o", dump, os.path.join(RUNTIME, "tests", "wmma_probe_dump.cpp")], check=True)
        for mode in ("gfx11", "gfx12"):
            r = subprocess.run([dump, mode, os.path.join(H.WORK, f"probe_{mode}.in"), os.path.join(H.WORK, f"probe_{mode}.exp")],
                               check=True, capture_output=True, text=True)
            if mode == "gfx11":
                cls.tolerance = float(r.stdout.split()[1])
        # the kernel as build_dx12.py --gfx 11 builds it (the declaration decides its defines and file)
        out = os.path.join(H.WORK, "probe_build")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.join(out, "dx12", "tools"))
        asm = os.path.join(out, "dx12", "tools", "dxil_asm.exe")
        shutil.copyfile(H.ASM, asm)
        os.chmod(asm, 0o755)
        os.utime(asm, None)
        assert B.main(["--out", out, "--gfx", "11", "--only", "native_driver_probe_f16", "-j", "1"]) == 0
        with open(os.path.join(out, "dx12-gfx11", "manifest.json")) as f:
            cls.manifest = json.load(f)
        cls.dxil = os.path.join(out, "dx12-gfx11", cls.manifest["kernels"][0]["file"])
        H.Card.dxil["probe"] = cls.dxil

    def run_probe(self, mode):
        with open(os.path.join(H.WORK, f"probe_{mode}.in"), "rb") as f:
            buf = f.read()
        out, txt = H.Card.run("probe", buf, [0], tag=f"probe_{mode}")
        self.assertIsNotNone(out, txt[-400:])
        return out[:1024].view(np.float32)

    def test_build_writes_the_gfx11_manifest(self):
        self.assertEqual(self.manifest["gfx"], 11)
        self.assertEqual([k["name"] for k in self.manifest["kernels"]], ["native_driver_probe_f16"])
        self.assertEqual(self.manifest["kernels"][0]["rs"], "buffer")

    def test_probe_matches_the_cpu_model(self):
        want = np.fromfile(os.path.join(H.WORK, "probe_gfx11.exp"), np.float32)
        got = self.run_probe("gfx11")
        self.assertLessEqual(float(np.abs(got - want).max()), self.tolerance)   # the last bit of the hardware's sum may differ

    def test_probe_fails_with_the_gfx12_layout(self):
        want = np.fromfile(os.path.join(H.WORK, "probe_gfx11.exp"), np.float32)
        got = self.run_probe("gfx12")
        self.assertGreater(int(np.sum(np.abs(got - want) > self.tolerance)), 200)


if __name__ == "__main__":
    unittest.main()
