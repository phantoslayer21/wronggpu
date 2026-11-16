# SPDX-License-Identifier: MIT
"""build_dx12.py --gfx: the GPU family of a build (declarations, output folder, manifest key, defines). Nothing is
compiled: compile_hlsl and the assembler tool are replaced, so the test runs everywhere in a second."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_dx12 as B  # noqa: E402

SRC = """// @wg-kernel old_kernel co=a.co block=32 wave=32 args=p
// @wg-kernel both_kernel co=b.co block=32 wave=32 args=p gfx=11,12
// @wg-kernel rdna3_kernel co=- block=32 wave=32 args=p,u gfx=11 entry=k11
"""


class TestBuildGfx(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kdir = os.path.join(self.tmp.name, "kernels")
        os.makedirs(self.kdir)
        with open(os.path.join(self.kdir, "t.hlsl"), "w") as f:
            f.write(SRC)
        self.out = os.path.join(self.tmp.name, "out")
        self.calls = []

        def fake_compile(src, entry, out_dxil, defines=None, **kw):
            self.calls.append((entry, dict(defines or {})))
            with open(out_dxil, "wb") as f:
                f.write(b"DXIL" + entry.encode())
            return out_dxil

        for p in (mock.patch.object(B, "KERNEL_DIR", self.kdir), mock.patch.object(B, "K_KERNEL_DIR", "/nonexistent"),
                  mock.patch.object(B, "build_asm_tool", lambda out: "asm"),
                  mock.patch.object(B, "compile_hlsl", fake_compile),
                  mock.patch.object(B, "vgpr_check", lambda built, required=(): self.calls.append(("vgpr_check", len(built))))):
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, folder):
        with open(os.path.join(self.out, folder, "manifest.json")) as f:
            return json.load(f)

    def build(self, *args):
        with mock.patch("sys.stdout"):
            self.assertEqual(B.main(["--out", self.out, *args]), 0)

    def test_declarations_default_to_gfx12(self):
        decls = {d["name"]: d["gfx"] for d in B.kernel_decls(os.path.join(self.kdir, "t.hlsl"))}
        self.assertEqual(decls, {"old_kernel": {12}, "both_kernel": {11, 12}, "rdna3_kernel": {11}})

    def test_bad_family_in_a_declaration_is_refused(self):
        path = os.path.join(self.kdir, "bad.hlsl")
        with open(path, "w") as f:
            f.write("// @wg-kernel x co=a.co block=32 wave=32 args=p gfx=13\n")
        with self.assertRaises(SystemExit):
            B.kernel_decls(path)

    def test_kernel_sets_by_family(self):
        self.assertEqual(B.kernel_set(), [("both_kernel", "b.co"), ("old_kernel", "a.co")])
        self.assertEqual(B.kernel_set(gfx=12), B.kernel_set())
        self.assertEqual(B.kernel_set(gfx=11), [("both_kernel", "b.co"), ("rdna3_kernel", "-")])

    def test_gfx12_build_is_as_before(self):
        self.build()
        man = self.manifest("dx12")
        self.assertEqual(list(man), ["format", "dxc", "test_only", "kernels"])   # no gfx key: a gfx12 manifest keeps its bytes
        self.assertEqual([k["name"] for k in man["kernels"]], ["both_kernel", "old_kernel"])
        for _, defs in [c for c in self.calls if c[0] != "vgpr_check"]:
            self.assertNotIn("WG_GFX", defs)
        self.assertIn(("vgpr_check", 2), self.calls)
        self.assertFalse(os.path.exists(os.path.join(self.out, "dx12-gfx11")))

    def test_gfx11_build(self):
        self.build("--gfx", "11")
        man = self.manifest("dx12-gfx11")
        self.assertEqual(man["gfx"], 11)
        self.assertEqual([k["name"] for k in man["kernels"]], ["both_kernel", "rdna3_kernel"])
        self.assertEqual(sorted(os.listdir(os.path.join(self.out, "dx12-gfx11"))),
                         ["both_kernel.dxil", "manifest.json", "rdna3_kernel.dxil"])
        self.assertEqual(sorted(e for e, _ in self.calls), ["both_kernel", "k11"])
        for _, defs in self.calls:
            self.assertEqual(defs["WG_GFX"], 11)
        self.assertNotIn(("vgpr_check", 2), self.calls)   # RGA is not used for gfx11
        self.assertFalse(os.path.exists(os.path.join(self.out, "dx12", "manifest.json")))

    def test_gfx11_only_picks_gfx11_kernels(self):
        with self.assertRaises(SystemExit):
            self.build("--gfx", "11", "--only", "old_kernel")   # a gfx12-only kernel is not declared for gfx11

    def test_isa_is_refused_for_gfx11(self):
        with self.assertRaises(SystemExit):
            self.build("--gfx", "11", "--isa")

    def test_list_follows_the_family(self):
        with mock.patch("builtins.print") as pr:
            B.main(["--list", "--gfx", "11"])
        self.assertEqual(json.loads(pr.call_args[0][0]), [["both_kernel", "b.co"], ["rdna3_kernel", "-"]])


if __name__ == "__main__":
    unittest.main()
