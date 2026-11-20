# SPDX-License-Identifier: MIT
"""The runtime's target map (clean/runtime/target_family.h) is the map of clean/runner/family.py: the same exact names, the
same families, and every other name refused (no prefix match). No GPU; the host C++ compiler builds the small program
clean/tests/test_target_family.cpp (skipped without one).

Run: python -m unittest clean.tests.test_target_family
"""
import subprocess
import unittest

from clean.runner import family
from clean.tests import host_cxx

# refused by family.py, so by the runtime: LLVM's other gfx11 / gfx12 names, partial and decorated names, case, other families
REFUSED = ["gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1170", "gfx1171", "gfx1172", "gfx11", "gfx12", "gfx1100:xnack-",
           "gfx1201:sramecc+", "GFX1100", " gfx1100", "gfx900", "gfx90a", "unknown"]


class TargetFamily(unittest.TestCase):
    def test_runtime_map_equals_family_py(self):
        exe = host_cxx.build("test_target_family", ["clean/tests/test_target_family.cpp"])
        names = list(family.FAMILIES) + REFUSED
        r = subprocess.run([exe] + names, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        got = dict(line.rsplit(" ", 1) for line in r.stdout.splitlines())
        want = {}
        for n in names:
            try:
                want[n] = family.family(n)
            except ValueError:
                want[n] = "-"
        self.assertEqual(got, want)
        self.assertEqual({n: got[n] for n in family.FAMILIES}, family.FAMILIES)


if __name__ == "__main__":
    unittest.main()
