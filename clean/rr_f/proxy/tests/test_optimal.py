# SPDX-License-Identifier: MIT
"""WrongGPU's optimal-settings rule (spec 61.5) checked against an independent oracle in exact rationals.

The C++ implementation (src/optimal.cpp) is dumped by test_proxy_core.exe --dump-optimal for modes 0..7 and
every output size 0..8192 (both axes use the same function). The oracle: r = the binary32 value nearest to the
exact ratio (Fractions, round-to-nearest-even), p = fl32(fl32(output) x r) by exact rounding of the exact product,
size = floor(p + 1/2) (round half up).
"""
import math
import os
import struct
import subprocess
import tempfile
import unittest
from fractions import Fraction as F

BUILD = os.environ.get("WG_RR_PROXY_BUILD", os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "build")), "rr-f-proxy", "out"))
EXE = os.path.join(BUILD, "core", "test_proxy_core.exe")
RATIOS = {1: F(1, 2), 2: F(58, 100), 3: F(2, 3), 4: F(1, 3), 6: F(1)}


def fl32(x):
    """Round an exact positive rational to binary32, nearest-even (normal range only, as needed here)."""
    x = F(x)
    if x == 0:
        return F(0)
    e = math.floor(math.log2(x))
    while F(2) ** e > x:
        e -= 1
    while F(2) ** (e + 1) <= x:
        e += 1
    ulp = F(2) ** (e - 23)
    q, r = divmod(x, ulp)
    q = int(q)
    if r * 2 > ulp or (r * 2 == ulp and q % 2 == 1):
        q += 1
    return q * ulp


def oracle(mode, o):
    if mode not in RATIOS:
        return 0, 0, 0, 0
    r = fl32(RATIOS[mode])
    third = fl32(F(1, 3))
    size = lambda rr: math.floor(fl32(fl32(o) * rr) + F(1, 2))
    return 1, size(r), size(third), o


class Optimal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(EXE):
            raise unittest.SkipTest("build test_proxy_core.exe first (tools/build.py)")
        fd, cls.path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        subprocess.run([EXE, "--dump-optimal", cls.path], check=True)
        with open(cls.path) as f:
            cls.rows = [tuple(int(x) for x in line.split(",")) for line in f]
        os.remove(cls.path)

    def test_ratio_constants(self):
        # the float nearest to the exact ratio, also via double rounding in the C++ (no midpoint case)
        for mode, r in RATIOS.items():
            via_double = F(struct.unpack("<f", struct.pack("<f", float(r)))[0])
            self.assertEqual(via_double, fl32(r), mode)

    def test_every_size_every_mode(self):
        self.assertEqual(len(self.rows), 8 * 8193)
        mismatches = [row for row in self.rows if row[2:] != oracle(row[0], row[1])]
        self.assertEqual(mismatches[:5], [])
        print(f"\noptimal settings: {len(self.rows) * 4} values compared, {len(mismatches)} mismatches")

    def test_exact_halves_round_up(self):
        # max performance r = 1/2: odd outputs land exactly on .5 and must round up
        halves = [row for row in self.rows if row[0] == 1 and row[1] % 2 == 1]
        self.assertTrue(halves)
        for mode, o, has, w, mn, mx in halves:
            self.assertEqual(w, (o + 1) // 2)

    def test_window(self):
        for mode, o, has, w, mn, mx in self.rows:
            if has:
                self.assertEqual(mx, o)
                self.assertLessEqual(mn, w)
                self.assertLessEqual(w, mx)


if __name__ == "__main__":
    unittest.main()
