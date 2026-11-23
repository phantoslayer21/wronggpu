# SPDX-License-Identifier: MIT
"""Prove the production S transfer table with a CPU-only dump."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class TestSBreakpoints(unittest.TestCase):
    def test_production_table(self):
        with tempfile.TemporaryDirectory(prefix="wg_k_s_breakpoints_") as tmp:
            supplied = os.environ.get("WG_K_S_BREAKPOINTS_DUMP")
            if supplied:
                dump = Path(supplied).resolve()
                if not dump.is_file():
                    self.skipTest("WG_K_S_BREAKPOINTS_DUMP does not name a table dump")
            else:
                cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
                if not cxx or not shutil.which(cxx):
                    self.skipTest("no host C++ compiler or WG_K_S_BREAKPOINTS_DUMP")
                probe = Path(tmp) / ("ki_s_breakpoints_cpu.exe" if sys.platform == "win32" else "ki_s_breakpoints_cpu")
                source = Path(__file__).with_name("ki_s_breakpoints_cpu.cpp")
                dump = Path(tmp) / "table.bin"
                self.run_checked(cxx, "-std=c++17", "-O2", "-ffp-contract=off", source, "-o", probe)
                self.run_checked(probe, dump)
            repo = Path(__file__).resolve().parents[3]
            self.run_checked(sys.executable, "-m", "clean.k.tests.prove_ki_s_breakpoints", dump, cwd=repo)

    def run_checked(self, *args, cwd=None):
        run = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
