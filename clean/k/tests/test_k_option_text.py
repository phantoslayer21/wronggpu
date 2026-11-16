# SPDX-License-Identifier: MIT
"""Model K's player-visible option text (host check, no GPU).

The create note of the opt-in k_approx option (clean/k/runtime/runtime_k.cpp, approx_note_) reaches the player's log
through the front end. It names the option as a faster approximation and the default as the standard path; it makes no
claim of exactness (the project's wording rule). The note itself is checked on a device by
clean/k/tests/test_k_runtime_d3d12.py test_k_approx_each_tag.

Run: python -m unittest clean.k.tests.test_k_option_text
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestKOptionText(unittest.TestCase):
    def test_k_approx_note_wording(self):
        with open(os.path.join(ROOT, "clean", "k", "runtime", "runtime_k.cpp"), encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"approx_note_ = std::string\(\"model K: close approximation k_approx=\"\)(.*?);\n", src, re.S)
        self.assertIsNotNone(m, "approx_note_ assignment not found")
        stmt = m.group(1)
        fixed = stmt.split("(approx_a1_ ? std::string(")[0]   # the part every k_approx note carries
        self.assertIn('" (opt-in faster approximation; the default is the standard path)"', fixed)
        self.assertNotIn("exact", fixed)
        self.assertNotIn("default path is exact", src)


if __name__ == "__main__":
    unittest.main()
