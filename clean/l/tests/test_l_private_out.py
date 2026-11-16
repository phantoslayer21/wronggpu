# SPDX-License-Identifier: MIT
"""The synthetic-values rule for the Model L checkers' real modes: l_hyb_stage_check --real-inputs, l_q7_check --real
and l_q7w_check --real write their real-data intermediates (case files, replay inputs and outputs) only to --private-out,
a resolved folder outside the repository. Each real-input path refuses a missing folder and a folder inside the
repository before it reads or writes anything; a cases file prepared with --real is read only from such a folder.
No GPU, no model, no recorded data (every value here is constructed; the paths given as real inputs do not exist).
Run: python clean/l/tests/test_l_private_out.py
"""
import os
import pickle
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_hyb_stage_check as HS  # noqa: E402
from clean.l.runner import l_private as P  # noqa: E402
from clean.l.runner import l_q7_check as Q7  # noqa: E402
from clean.l.runner import l_q7w_check as Q7W  # noqa: E402

IN_REPO = (ROOT, os.path.join(ROOT, "clean", "l", "records"), os.path.join(ROOT, "out", "real-run"))


def outside_tmp(test):
    d = tempfile.mkdtemp(prefix="wg_priv_")
    test.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
    if P.inside_repository(d):
        test.skipTest("the temporary folder is inside the repository on this machine")
    return d


def listing(d):
    return sorted(os.path.relpath(os.path.join(r, f), d) for r, _, fs in os.walk(d) for f in fs) + \
        sorted(os.path.relpath(os.path.join(r, x), d) for r, xs, _ in os.walk(d) for x in xs)


class Helper(unittest.TestCase):
    def test_inside_is_refused_outside_is_taken(self):
        for p in IN_REPO:
            with self.subTest(p=p), self.assertRaises(SystemExit) as cm:
                P.private_dir(p)
            self.assertIn("inside the repository", str(cm.exception))
        with self.assertRaises(SystemExit):
            P.private_dir(None)
        d = outside_tmp(self)
        self.assertEqual(P.private_dir(os.path.join(d, "x")), os.path.realpath(os.path.join(d, "x")))


class HybStageRealInputs(unittest.TestCase):
    def test_refused_before_anything_is_written(self):
        d = outside_tmp(self)
        build = os.path.join(d, "build")
        os.makedirs(build)
        nothing = os.path.join(d, "no-real-inputs")   # never read: the refusal comes first
        cases = ([], ["--private-out", os.path.join(ROOT, "out", "real-run")],
                 ["--private-out", os.path.join(ROOT, "clean")])
        for extra in cases:
            for mode in (["--run"], ["--run", "--replay-only"], ["--run", "--no-replay"]):
                argv = ["--build", build, "--real-inputs", nothing] + mode + extra
                with self.subTest(argv=argv[4:]), self.assertRaises(SystemExit):
                    HS.main(argv)
        self.assertEqual(listing(build), [])
        self.assertFalse(os.path.exists(os.path.join(ROOT, "out", "real-run")))

    def test_private_out_alone_is_refused(self):
        d = outside_tmp(self)
        with self.assertRaises(SystemExit):
            HS.main(["--build", d, "--run", "--private-out", os.path.join(d, "p")])


class Q7Real(unittest.TestCase):
    MODS = (("l_q7_check", Q7, "l_q7_cases.pkl"), ("l_q7w_check", Q7W, "l_q7w_cases.pkl"))

    def test_prepare_real_needs_a_private_folder_outside(self):
        for name, mod, _ in self.MODS:
            d = outside_tmp(self)
            build = os.path.join(d, "build")
            os.makedirs(build)
            real = os.path.join(d, "no-real-build")   # never read: the refusal comes first
            for extra in ([], ["--private-out", os.path.join(ROOT, "out", "real-run")],
                          ["--private-out", os.path.join(ROOT, "clean", "l")]):
                argv = ["--build", build, "--prepare", "--real", real] + extra
                with self.subTest(checker=name, argv=argv[4:]), self.assertRaises(SystemExit):
                    mod.main(argv)
            self.assertEqual(listing(build), [])
        self.assertFalse(os.path.exists(os.path.join(ROOT, "out", "real-run")))

    def test_run_with_a_private_folder_inside_is_refused(self):
        for name, mod, _ in self.MODS:
            d = outside_tmp(self)
            with self.subTest(checker=name), self.assertRaises(SystemExit):
                mod.main(["--build", d, "--run", "--d3d12", "--private-out", os.path.join(ROOT, "out", "real-run")])
        self.assertFalse(os.path.exists(os.path.join(ROOT, "out", "real-run")))

    def test_a_real_cases_file_is_read_only_from_its_private_folder(self):
        """A cases file marked as prepared with --real (constructed: no cases, a made-up digest) in the build folder
        is refused by --run without --private-out, before any GPU work."""
        for name, mod, fn in self.MODS:
            d = outside_tmp(self)
            p = os.path.join(d, fn)
            with open(p, "wb") as f:
                pickle.dump({"cases": [], "real_params_sha256": "0" * 64, "form": "q7"}, f)
            with self.subTest(checker=name), self.assertRaises(SystemExit) as cm:
                mod.main(["--build", d, "--run"])
            self.assertIn("--private-out", str(cm.exception))

    def test_real_only_with_prepare(self):
        for name, mod, _ in self.MODS:
            d = outside_tmp(self)
            with self.subTest(checker=name), self.assertRaises(SystemExit):
                mod.main(["--build", d, "--run", "--real", os.path.join(d, "r"), "--private-out", os.path.join(d, "p")])

    def test_the_private_folder_holds_the_cases_file(self):
        """data_dir resolves the private folder (outside the repository); load_cases reads a real file from it."""
        d = outside_tmp(self)
        priv = Q7.data_dir(os.path.join(d, "build"), os.path.join(d, "priv"))
        self.assertEqual(priv, os.path.realpath(os.path.join(d, "priv")))
        p = os.path.join(priv, "l_q7_cases.pkl")
        with open(p, "wb") as f:
            pickle.dump({"cases": [], "real_params_sha256": "0" * 64, "form": "q7"}, f)
        self.assertEqual(Q7.load_cases(p, priv)["cases"], [])
        self.assertEqual(Q7.data_dir(os.path.join(d, "build"), None), os.path.join(d, "build"))


if __name__ == "__main__":
    unittest.main()
