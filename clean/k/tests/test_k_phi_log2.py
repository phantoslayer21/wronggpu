# SPDX-License-Identifier: MIT
"""Check the certified HDR phi logarithm with the generated D3D12 tables."""
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


K_ROOT = Path(__file__).resolve().parents[1]
PROOF_INPUT_SHA256 = "4006deb20e730e6084c6ba42cc0014633b40ccfc3f3974526ad0400696aec176"


def proof_input_digest(source, shader):
    from clean.k.tests.prove_ki_phi_log2 import table

    digest = hashlib.sha256()
    for name in ("KF_LOG_INV", "KF_LOG_H", "KF_LOG_L"):
        digest.update(table(name, source).view("uint32").astype("<u4").tobytes())
    digest.update(shader.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


class TestPhiLog2(unittest.TestCase):
    def test_production_tables_and_shader(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy is required to generate and prove the D3D12 tables")
        dx12 = K_ROOT / "dx12"
        with tempfile.TemporaryDirectory(prefix="wg_k_phi_log2_") as tmp:
            tables = Path(tmp) / "k_fp_tables.hlsli"
            self.run_checked(dx12 / "gen_k_fp_tables.py", tables)

            proof = Path(__file__).with_name("prove_ki_phi_log2.py")
            # Set WG_K_PHI_LOG2_FULL=1 to re-prove all binary32 mantissas.
            if os.environ.get("WG_K_PHI_LOG2_FULL") == "1":
                self.run_checked(proof, tables)
            else:
                self.run_checked(proof, tables, "65536")

            digest = proof_input_digest(tables.read_text(), dx12 / "hlsl" / "k_phi_log2.hlsli")
            self.assertEqual(digest, PROOF_INPUT_SHA256,
                             "Proof inputs changed. Re-run with WG_K_PHI_LOG2_FULL=1 before updating the hash.")

    def run_checked(self, *args):
        run = subprocess.run([sys.executable, *args], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
