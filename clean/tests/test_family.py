# SPDX-License-Identifier: MIT
"""GPU family of a target (clean/runner/family.py): exact names only, and the family a build records (no GPU)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from clean.runner import family
from clean.runner.packing import D_ROW_LAYOUTS


class FamilyMap(unittest.TestCase):
    def test_known_targets(self):
        self.assertEqual({a: family.family(a) for a in ("gfx1200", "gfx1201", "gfx1100", "gfx1101")},
                         {"gfx1200": "gfx12", "gfx1201": "gfx12", "gfx1100": "gfx11", "gfx1101": "gfx11"})

    def test_everything_else_is_refused(self):
        # gfx1170-1172 are LLVM gfx11 names but carry the gfx12 matrix design: a prefix match would misfile them
        for arch in ("gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1170", "gfx1171", "gfx1172", "gfx11", "gfx12",
                     "gfx1201:xnack-", "GFX1100", "gfx900", ""):
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "unsupported GPU target"):
                family.family(arch)

    def test_every_family_has_an_accumulator_row_table(self):
        for fam in set(family.FAMILIES.values()):
            self.assertIn(fam, D_ROW_LAYOUTS)


class RecordedTarget(unittest.TestCase):
    def write(self, d, text):
        with open(os.path.join(d, family.TARGET_FILE), "w") as f:
            f.write(text)

    def test_no_record_is_gfx12(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(family.build_family(d), "gfx12")

    def test_record(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, json.dumps({"arch": "gfx1100"}))
            self.assertEqual(family.build_family(d), "gfx11")
            self.write(d, json.dumps({"arch": "gfx1201"}))
            self.assertEqual(family.build_family(d), "gfx12")

    def test_record_of_an_unsupported_target_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, json.dumps({"arch": "gfx1170"}))
            with self.assertRaises(ValueError):
                family.build_family(d)


class RunnerDefaultRows(unittest.TestCase):
    """Runner's default accumulator row table follows the build's target; WG_D_ROW and the argument still win."""

    def runner(self, arch, **kw):
        from clean.runner.stage_runner import Runner
        with tempfile.TemporaryDirectory() as d:
            if arch:
                with open(os.path.join(d, family.TARGET_FILE), "w") as f:
                    json.dump({"arch": arch}, f)
            with mock.patch("clean.runner.emu.Emu"):
                return Runner(d, "", backend="emu", **kw).d_row

    def test_default_follows_the_target(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("WG_D_ROW", None)
            self.assertEqual(self.runner(None), "gfx12")
            self.assertEqual(self.runner("gfx1201"), "gfx12")
            self.assertEqual(self.runner("gfx1100"), "gfx11")

    def test_explicit_choices_win(self):
        with mock.patch.dict(os.environ, {"WG_D_ROW": "gfx12"}):
            self.assertEqual(self.runner("gfx1100"), "gfx12")
        with mock.patch.dict(os.environ):
            os.environ.pop("WG_D_ROW", None)
            self.assertEqual(self.runner("gfx1100", d_row="gfx12"), "gfx12")


class PipelineDefaultRows(unittest.TestCase):
    """Pipeline's default accumulator row table follows the build's target as Runner's does (no runtime library needed:
    the constructor is stopped at the row table)."""

    class Stop(Exception):
        pass

    def default_rows(self, arch, env=None, **kw):
        from clean.runner import pipeline
        with tempfile.TemporaryDirectory() as d, mock.patch("ctypes.CDLL"), mock.patch.dict(os.environ, env or {}), \
                mock.patch.object(pipeline, "resolve_d_row", side_effect=self.Stop) as resolve:
            os.environ.pop("WG_D_ROW", None) if not env else None
            if arch:
                with open(os.path.join(d, family.TARGET_FILE), "w") as f:
                    json.dump({"arch": arch}, f)
            with self.assertRaises(self.Stop):
                pipeline.Pipeline(d, "", "", (8, 8), (8, 8), {}, **kw)
            return resolve.call_args.args[0]

    def test_default_follows_the_target(self):
        self.assertEqual(self.default_rows(None), "gfx12")
        self.assertEqual(self.default_rows("gfx1201"), "gfx12")
        self.assertEqual(self.default_rows("gfx1100"), "gfx11")

    def test_explicit_choices_win(self):
        self.assertEqual(self.default_rows("gfx1100", env={"WG_D_ROW": "gfx12"}), "gfx12")
        self.assertEqual(self.default_rows("gfx1100", d_row="gfx12"), "gfx12")


if __name__ == "__main__":
    unittest.main()
