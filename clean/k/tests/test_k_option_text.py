# SPDX-License-Identifier: MIT
"""Model K's player-visible option text (host check, no GPU).

The create note of the opt-in k_approx option (clean/k/runtime/runtime_k.cpp, approx_note_), its a1 legacy-input line
and the front end's K mode line (clean/runtime/interop/backend.cpp) reach the player's log. They name the option as a
faster approximation and the default as the standard path (input stage); they make no claim of exactness (the project's
wording rule). The note itself is checked on a device by
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
        a1 = stmt.split("(approx_a1_ ? std::string(")[1]   # the a1 part of the note
        self.assertIn("run the standard input stage", a1)
        self.assertNotIn("exact", a1)
        m = re.search(r'frame_note_ = "(model K: k_approx=a1 [^"]*)";', src)
        self.assertIsNotNone(m, "the a1 legacy-input line not found")
        self.assertIn("run the standard input stage", m.group(1))
        self.assertNotIn("exact", m.group(1))

    def test_k_mode_line_wording(self):
        """The front end's K mode line (clean/runtime/interop/backend.cpp) names the default as the standard path and an
        opt-in mode as an approximation, with no claim of exactness."""
        with open(os.path.join(ROOT, "clean", "runtime", "interop", "backend.cpp"), encoding="utf-8") as f:
            src = f.read()
        m = re.search(r'network model K mode: network=.*?\(exact \? "([^"]*)" : "([^"]*)"\)', src, re.S)
        self.assertIsNotNone(m, "the K mode line not found")
        self.assertEqual(m.groups(), (", the standard path", ", an approximation (opt-in)"))


if __name__ == "__main__":
    unittest.main()
