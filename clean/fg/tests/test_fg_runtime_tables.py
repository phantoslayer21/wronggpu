# SPDX-License-Identifier: GPL-3.0-only
"""The C++ FG runtime's generated headers (clean/fg/runtime/fg_map_table.h and tests/net1_plan_golden.h) are current:
they match data/fg.v1.map.json and clean/fg/runner/fg_net1_mma.py's fused plan and weight fragments. The C++ tests
(clean/fg/runtime, ctest fg_runtime) check the runtime against these headers, so together they tie the C++ plan to the
Python plan that the D3D12 replay checks."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "tools"))

import gen_fg_runtime_tables as T  # noqa: E402


class GeneratedHeadersTest(unittest.TestCase):
    def test_headers_are_current(self):
        for path, text in T.outputs().items():
            with open(path, "rb") as f:
                have = f.read().decode("utf-8")
            self.assertEqual(have, text, "%s is not current: run python clean/fg/tools/gen_fg_runtime_tables.py"
                             % os.path.relpath(path, ROOT))

    def test_golden_covers_the_size_table_rows(self):
        # net1 cells of 1280x720, 1920x1080 / 3840x2160, 2560x1440, 3440x1440 and 7680x4320 (FG-PART3 section 1.3)
        for cells in ((192, 320), (288, 480), (384, 640), (384, 864), (544, 960)):
            self.assertIn(cells, T.CASES)

    def test_synth_bits_are_finite_float16(self):
        b = T.synth_bits(1 << 16)
        self.assertTrue(((b & 0x7C00) != 0x7C00).all())
        self.assertTrue((b & 0x8000).any() and (~b & 0x8000).any())


if __name__ == "__main__":
    unittest.main()
