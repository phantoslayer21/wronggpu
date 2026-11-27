# SPDX-License-Identifier: GPL-3.0-only
"""runtime/hyb_tables.h (the arena tables the D3D12 hybrid kernels read) gives exactly the bytes of
clean/l/reference/hyb_tables.py: GELU16 -> FP8 for all 65,536 inputs, the rsqrt and rcp tables, and the GELU16 table
with 16-bit entries."""
import os
import subprocess
import tempfile
import unittest

import numpy as np

from clean.l.reference import hyb_tables as T
from clean.tests import host_cxx


def python_bytes():
    out = np.zeros(65536 + 4096 + 2048 + 131072, np.uint8)
    out[:65536] = T.gelu16_q8()
    out[65536:69632] = T.rsq16_table().astype("<u2").reshape(-1).view(np.uint8)
    out[69632:71680] = T.rcp16_table().astype("<u2").view(np.uint8)
    out[71680:] = T.gelu16_q8().astype("<u2").view(np.uint8)
    return out


class HybTablesCpp(unittest.TestCase):
    def test_same_bytes(self):
        exe = host_cxx.build("hyb_dump", ["clean/tests/hyb_dump.cpp"],
                             include_dirs=[os.path.join(host_cxx.ROOT, "clean", "runtime")])
        want = python_bytes()
        # the default rounding mode, and the host thread in FE_DOWNWARD (hyb_fill sets round-to-nearest for the
        # generation and restores the caller's mode; the dump exits 3 if the mode was not restored)
        for mode in ([], ["downward"]):
            with self.subTest(mode=mode or "default"), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "t.bin")
                r = subprocess.run([exe, path] + mode)
                self.assertEqual(r.returncode, 0, "hyb_dump %s: exit %d (3 = rounding mode not restored)" % (mode, r.returncode))
                got = np.fromfile(path, np.uint8)
                self.assertEqual(got.size, want.size)
                bad = np.nonzero(got != want)[0]
                self.assertEqual(bad.size, 0, f"{bad.size} differing bytes, first at {bad[:8].tolist()}")

    def test_fill_needs_the_mode(self):
        """Negative control: the generation itself (hyb_fill_nearest) run under FE_DOWNWARD gives different bytes, so
        the mode handling is what keeps them equal."""
        src = os.path.join(host_cxx.ROOT, "clean", "tests", "hyb_dump.cpp")
        text = open(src, encoding="utf-8").read().replace("wg::hyb_fill(t.data());", "wg::hyb_fill_nearest(t.data());")
        with tempfile.TemporaryDirectory() as tmp:
            mut = os.path.join(tmp, "hyb_dump_nearest.cpp")
            open(mut, "w", encoding="utf-8").write(text)
            exe = host_cxx.build("hyb_dump_nearest", [mut], include_dirs=[os.path.join(host_cxx.ROOT, "clean", "runtime")])
            path = os.path.join(tmp, "t.bin")
            subprocess.run([exe, path, "downward"])
            got = np.fromfile(path, np.uint8)
        self.assertGreater(int(np.sum(got != python_bytes())), 0)


if __name__ == "__main__":
    unittest.main()
