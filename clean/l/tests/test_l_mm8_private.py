# SPDX-License-Identifier: GPL-3.0-only
"""The synthetic-values rule for clean/l/runner/l_mm8_f32_check.py (its --real mode reads a local recorded run): that
mode's intermediates go only to a folder outside the repository, and every record it writes is aggregate-only.
Regression checks of both; no GPU, no model, no recorded data (every value here is constructed).
Run: python clean/l/tests/test_l_mm8_private.py
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_mm8_f32_check as M  # noqa: E402


class PrivateOutput(unittest.TestCase):
    def test_a_folder_inside_the_repository_is_refused(self):
        for p in (ROOT, os.path.join(ROOT, "clean", "l", "records"), os.path.join(ROOT, "out", "real-run")):
            with self.subTest(p=p), self.assertRaises(SystemExit) as cm:
                M.private_dir(p)
            self.assertIn("inside the repository", str(cm.exception))

    def test_a_folder_outside_it_is_taken(self):
        with tempfile.TemporaryDirectory() as d:
            try:
                inside = os.path.commonpath([os.path.realpath(d), os.path.realpath(ROOT)]) == os.path.realpath(ROOT)
            except ValueError:   # another drive
                inside = False
            if inside:
                self.skipTest("the temporary folder is inside the repository on this machine")
            self.assertEqual(M.private_dir(os.path.join(d, "x")), os.path.realpath(os.path.join(d, "x")))

    def test_the_real_mode_is_explicit_and_private(self):
        """A recorded run is read only with --real, and --real only with a private folder outside the repository: the
        command refuses before it reads anything (the paths below do not exist)."""
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            cases = (["--json", j, "--frames", os.path.join(d, "nobuild")],
                     ["--json", j, "--real", "--frames", os.path.join(d, "nobuild")],
                     ["--json", j, "--real", "--frames", os.path.join(d, "nobuild"), "--private-out",
                      os.path.join(ROOT, "out", "real-run")],
                     ["--json", j, "--wmma-prepare", d, "--real", "--frames", os.path.join(d, "nobuild"),
                      "--private-out", os.path.join(ROOT, "clean")])
            for argv in cases:
                with self.subTest(argv=argv[2:]), self.assertRaises(SystemExit):
                    M.main(argv)
            self.assertFalse(os.path.exists(j))


class AggregateOnly(unittest.TestCase):
    ROW = {"start_f16_bits": 15360, "a_fp8_codes": [56] * 32, "w_fp8_codes": [48] * 32, "exact_sum": 1.0,
           "instruction_f32": 1.0}   # constructed values (the shape of an example row)

    def test_operand_values_of_a_real_set_are_refused(self):
        for rec in ({"probe": {"real_frames_sample": {"ops": 10, "examples": [self.ROW]}}},
                    {"frames": {"ops": 3, "examples": [self.ROW]}},
                    {"probe": {"real_frames_sample": {"zero_start": {"a_fp8_codes": [1, 2]}}}}):
            with self.subTest(rec=list(rec)), self.assertRaises(SystemExit):
                M.assert_aggregate_only(rec)

    def test_aggregates_and_synthetic_examples_pass(self):
        M.assert_aggregate_only({"probe": {"real_frames_sample": {"ops": 10, "f32_not_exact_sum": 0, "examples": []},
                                           "near_tie": {"ops": 5, "examples": [self.ROW]}},
                                 "frames": {"ops": 3, "differ": 0, "min_margin": 2.5}})


if __name__ == "__main__":
    unittest.main()
