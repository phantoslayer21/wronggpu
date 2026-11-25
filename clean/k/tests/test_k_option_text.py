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

    def test_k_mode_option(self):
        """k_mode (0.7.0, the player's Fast / Exact from the SR front end): fast takes the k_approx forms the create
        supports and never refuses; exact is the standard path; a later k_approx item decides instead; neither given
        keeps the standard path (bare runtime calls stay exact for the unit, oracle and RADV tests)."""
        with open(os.path.join(ROOT, "clean", "k", "runtime", "runtime_k.cpp"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('const long kmode_at = option_pos(c, "k_mode"), kapprox_at = option_pos(c, "k_approx");', src)
        self.assertIn("if (kmode_at >= 0 && kmode_at > kapprox_at) {", src)
        m = re.search(r'if \(kmode == "fast"\) \{(.*?)\} else if \(kmode == "exact"\)', src, re.S)
        self.assertIsNotNone(m, "the k_mode=fast branch not found")
        fast = m.group(1)
        self.assertNotIn("return fail(", fast)                 # never a refusal
        self.assertIn("if (c.backend == 3) {", fast)            # the faster forms on the native D3D12 device only
        self.assertIn("approx_a1_ = in_exact_;", fast)
        self.assertIn("approx_a2_ = !fast_ && compact_head_;", fast)
        self.assertIn('approx_note_ = "model K: k_mode=exact (the standard path)";', src)
        # the approximation flags start off: without k_mode / k_approx nothing turns them on
        with open(os.path.join(ROOT, "clean", "k", "runtime", "runtime_k.h"), encoding="utf-8") as f:
            hdr = f.read()
        self.assertIn("bool approx_a1_ = false;", hdr)
        self.assertIn("bool approx_a2_ = false;", hdr)
        self.assertEqual(src.count("approx_a1_ = true;"), 1)   # only the explicit k_approx=a1 item
        self.assertEqual(src.count("approx_a2_ = true;"), 1)

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
