# SPDX-License-Identifier: MIT
"""GPU/D3D12 regression: preserve HIP timing events until completed-frame harvest.

Set WG_K_GPU_BUILD and WG_K_INTEROP_EXE; run under the shared GPU lock. Generates
only synthetic parameters and runs the hipbench case with four frames in flight.
"""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import gpu_build, write_k_model


class TestInteropTiming(unittest.TestCase):
    def run_benchmark(self, stage_events=True):
        exe = os.environ.get("WG_K_INTEROP_EXE")
        if not exe:
            self.skipTest("set WG_K_INTEROP_EXE to the clean native interop test")
        build = Path(gpu_build()).resolve()
        with tempfile.TemporaryDirectory(prefix="k-timing-") as td:
            model, modelmap = write_k_model(td, random_network_params(11))
            opts = ";".join(("model=K", "network=fused", "timing_log=1", "done_mode=fence",
                             f"kernels_dir={build}", f"model_dir={td}",
                             f"model_k_dir={Path(model).parent}", f"model_k_map={modelmap}",
                             "timing_stages=1" if stage_events else "timing_stages=0"))
            env = os.environ.copy()
            env.pop("WG_RUNTIME_OPTIONS", None)
            p = subprocess.run([exe, str(build / "wronggpu_runtime.dll"), "hipbench", opts, "900"],
                               capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            rows = [line for line in p.stdout.splitlines() if "timing, last 300 dispatches:" in line]
            self.assertGreaterEqual(len(rows), 3, p.stdout)
            for row in rows[1:]:
                total = re.search(r"TOTAL\(t3-t0\) ([0-9.]+)", row)
                self.assertIsNotNone(total, row)
                self.assertGreater(float(total[1]), 0, row)
                if stage_events:
                    self.assertNotIn("network GPU n/a", row)
                    match = re.search(r"\] \((\d+) frames\); TOTAL", row)
                    self.assertIsNotNone(match, row)
                    self.assertGreaterEqual(int(match[1]), 250, row)
                else:
                    self.assertIn("network GPU n/a", row)
            if not stage_events:
                self.assertIn("HIP kernel events disabled by timing_stages=0", p.stdout)
            self.assertIn("nonfinite=0", p.stdout)
            self.assertTrue(p.stdout.rstrip().endswith("PASS"), p.stdout)
            return p.stdout

    def test_inflight_stage_timings_survive(self):
        self.run_benchmark()

    def test_full_span_only_matches_output(self):
        traced = self.run_benchmark()
        full_span_only = self.run_benchmark(stage_events=False)
        pattern = r"hipbench final output: (\d+) values, nonfinite=0, fnv1a64=([0-9a-f]+)"
        a, b = re.search(pattern, traced), re.search(pattern, full_span_only)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(a.groups(), b.groups())


if __name__ == "__main__":
    unittest.main()
