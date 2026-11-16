# SPDX-License-Identifier: MIT
"""Model K's softmax self-check reference is bound to its sources (host checks, no GPU).

clean/k/runtime/k_smx_selfcheck.h holds the HIP reference's hashes of the self-check cases and kSourceDigest, the digest
of the shader sources, @wg-kernel definitions, runtime regions and extra defines they were generated from
(clean/k/dx12/smx_selfcheck_digest.py). Checked here:
- the committed header belongs to the committed sources (a source change without a regenerated header fails here);
- a copy of the sources with one changed byte in a shader source, a definition line or a runtime region, or with an
  extra define, gives another digest and is reported stale;
- clean/dx12/build_dx12.py refuses to build a self-check kernel with a stale header (WG_DX12_SMX_DIGEST=0 lets a
  negative-control build through only into a test-only output, --test-only), and ignores builds without one.

Run: python -m unittest clean.k.tests.test_k_smx_selfcheck_digest
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from clean.k.dx12 import smx_selfcheck_digest as D

ROOT = D.ROOT


def copy_sources(dst):
    """The files the digest reads, at the same relative paths under dst."""
    for d in [D.KERNEL_DIR] + D.INCLUDE_DIRS:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(dst, d))
    for rel in D.RUNTIME_FILES + [D.GENERATED_BY, D.HEADER]:
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), os.path.join(dst, rel))


def flip(path, needle):
    """Change one byte of the first occurrence of `needle` in the file."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    at = data.find(needle.encode())
    assert at >= 0, (path, needle)
    data[at] = ord("#") if data[at] != ord("#") else ord("!")
    with open(path, "wb") as f:
        f.write(bytes(data))


class TestSmxSelfcheckDigest(unittest.TestCase):
    def test_committed_header_is_current(self):
        self.assertIsNotNone(D.header_digest(), "k_smx_selfcheck.h has no kSourceDigest")
        self.assertIsNone(D.stale(), "regenerate clean/k/runtime/k_smx_selfcheck.h (gen_k_smx_selfcheck.py)")

    def test_changes_are_caught(self):
        cases = [(os.path.join("clean", "k", "dx12", "hlsl", "k_smx.hlsli"), "ro.v[j] = p + e;"),
                 (os.path.join("clean", "k", "dx12", "hlsl", "k_wide.hlsli"), "kw_softpoly8"),
                 (os.path.join("clean", "dx12", "hlsl", "wg_math.hlsli"), "float"),
                 (os.path.join("clean", "k", "dx12", "kernels", "k_block_wide.hlsl"), "KW_RD=12"),
                 (os.path.join("clean", "k", "runtime", "runtime_k.cpp"), "kSmxSeed = 0x6b5eu"),
                 (os.path.join("clean", "k", "runtime", "model_k.cpp"), "(2 * oxm - 4)")]
        for rel, needle in cases:
            with self.subTest(file=rel):
                tmp = tempfile.mkdtemp(prefix="smx_digest_")
                try:
                    copy_sources(tmp)
                    self.assertIsNone(D.stale(tmp), "an unchanged copy must match")
                    flip(os.path.join(tmp, rel), needle)
                    why = D.stale(tmp)
                    self.assertIsNotNone(why, "a changed byte in %s was not caught" % rel)
                    self.assertIn("regenerate", why)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
        self.assertIsNotNone(D.stale(defines=["K_SMX_MUT=1"]), "an extra define was not caught")

    def test_build_refuses_a_stale_header(self):
        from clean.dx12 import build_dx12 as B
        with mock.patch.dict(os.environ, {"WG_DX12_SMX_DIGEST": "1"}):
            with self.assertRaises(SystemExit) as cm:
                B.smx_digest_check({"k_stage_e2"}, ["K_SMX_MUT=1"])
            self.assertIn("regenerate", str(cm.exception))
            B.smx_digest_check({"k_stage_e2"}, [])             # current header, no extra define: builds
            B.smx_digest_check({"k_out_color"}, ["K_SMX_MUT=1"])   # no self-check kernel: not concerned
        with mock.patch.dict(os.environ, {"WG_DX12_SMX_DIGEST": "0"}):
            with self.assertRaises(SystemExit) as cm:   # the bypass alone: refused (never into a normal output)
                B.smx_digest_check({"k_stage_e2"}, ["K_SMX_MUT=1"])
            self.assertIn("--test-only", str(cm.exception))
            # a negative-control build into a test-only output: through, and the bypassed kernels are reported
            self.assertEqual(B.smx_digest_check({"k_stage_e2", "k_out_color"}, ["K_SMX_MUT=1"], test_only=True),
                             {"k_stage_e2"})
            self.assertEqual(B.smx_digest_check({"k_stage_e2"}, [], test_only=True), set())   # current: no bypass


if __name__ == "__main__":
    unittest.main()
