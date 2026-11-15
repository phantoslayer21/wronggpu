# SPDX-License-Identifier: MIT
"""Packaging completeness uses tiny own fixtures, no GPU or model data."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from clean.runner import build


class BuildTargets(unittest.TestCase):
    def test_complete_and_missing_target(self):
        with tempfile.TemporaryDirectory() as d, patch.object(build, "EXTRA_ARCHS", ["gfx1200"]):
            root = Path(d)
            (root / "gfx1200").mkdir()
            for name in build.EMBED_FILES:
                (root / name).write_bytes(b"own synthetic fixture")
                if name.endswith(".co"):
                    (root / "gfx1200" / name).write_bytes(b"own gfx1200 fixture")
            table = Path(build.write_embedded(d)).read_text()
            for arch in ("gfx1200", "gfx1201"):
                for name in build.EMBED_FILES:
                    if name.endswith(".co"):
                        self.assertIn('"' + arch + "." + name + '"', table)
            (root / "gfx1200" / "post_w64.co").unlink()
            with self.assertRaisesRegex(SystemExit, "missing required gfx1200"):
                build.write_embedded(d)


class FamilyBuilds(unittest.TestCase):
    """build.py main() with the compiler replaced: which sources are built for which target."""

    def build(self, *args, arch="gfx1201"):
        calls = []
        with tempfile.TemporaryDirectory() as d, patch.object(build, "ARCH", arch), \
                patch.object(build, "run", calls.append), patch.object(build, "find_clang", lambda r: "clang"):
            build.main(["--rocm", d, "--out", os.path.join(d, "out"), *args])
            out = sorted(os.listdir(os.path.join(d, "out")))
            target = os.path.join(d, "out", "target.json")
            recorded = None
            if os.path.exists(target):
                with open(target) as f:
                    recorded = json.load(f)["arch"]
        return calls, out, recorded

    def test_gfx1100_default_sources(self):
        calls, out, recorded = self.build("--arch", "gfx1100", "--no-runtime")
        self.assertEqual(sorted(c[-1].rsplit(os.sep, 1)[-1] for c in calls), sorted(s[:-4] + ".co" for s in build.FAMILY_SOURCES["gfx11"]))
        self.assertTrue(all("--offload-arch=gfx1100" in c for c in calls))
        self.assertEqual(recorded, "gfx1100")
        self.assertFalse([n for n in out if n.startswith(("k_", "post_w64", "swin"))])

    def test_gfx1100_swin_takes_the_plain_conversion(self):
        """swin.hip for gfx11 asks for -DWG_SLOW_Q8 (FAMILY_DEFINES: the faster conversion makes the block kernels spill)."""
        calls, _, _ = self.build("--arch", "gfx1100", "--no-runtime")
        by_name = {c[-1].rsplit(os.sep, 1)[-1]: c for c in calls}
        self.assertIn("-DWG_SLOW_Q8=1", by_name["swin.co"])
        self.assertFalse([n for n, c in by_name.items() if n != "swin.co" and any("WG_SLOW_Q8" in a for a in c)])
        calls, _, _ = self.build("--arch", "gfx1100", "--no-runtime", "--sources", "swin.hip", "--define", "WG_GELU_MODE=1")
        self.assertTrue({"-DWG_SLOW_Q8=1", "-DWG_GELU_MODE=1"} <= set(calls[0]))

    def test_gfx1100_variants(self):
        """--variants builds swin__<name>.co next to the default code objects, with the variant's macros on top of the
        family's, and writes variants.json (what Runner(..., variant=name) reads); only gfx11 variants are accepted."""
        from clean.runner import variants as V
        calls, out, _ = self.build("--arch", "gfx1100", "--no-runtime", "--variants", "f16w")
        by_name = {c[-1].rsplit(os.sep, 1)[-1]: c for c in calls}
        self.assertIn("swin__f16w.co", by_name)
        self.assertTrue({"-DWG_SLOW_Q8=1", "-DWG_F16_W=1"} <= set(by_name["swin__f16w.co"]))
        self.assertNotIn("-DWG_F16_W=1", by_name["swin.co"])
        self.assertIn("variants.json", out)
        calls, _, _ = self.build("--arch", "gfx1100", "--no-runtime", "--variants")  # all of the family
        self.assertEqual(sorted(c[-1].rsplit(os.sep, 1)[-1] for c in calls if "swin" in c[-1].rsplit(os.sep, 1)[-1]),
                         sorted(["swin.co"] + [V.code_object(n) for n in V.FAMILY_VARIANTS["gfx11"]]))
        with self.assertRaisesRegex(SystemExit, "not a gfx11 variant"):
            self.build("--arch", "gfx1100", "--no-runtime", "--variants", "t128")

    def test_chosen_sources(self):
        calls, _, recorded = self.build("--arch", "gfx1101", "--no-runtime", "--sources", "post.hip", "frame.hip")
        self.assertEqual([c[-1].rsplit(os.sep, 1)[-1] for c in calls], ["post.co", "frame.co"])
        self.assertTrue(all("--offload-arch=gfx1101" in c for c in calls))
        self.assertEqual(recorded, "gfx1101")

    def test_chosen_sources_for_gfx12(self):
        calls, _, recorded = self.build("--sources", "exposure.hip")
        self.assertEqual([c[-1].rsplit(os.sep, 1)[-1] for c in calls], ["exposure.co"])
        self.assertIn("--offload-arch=gfx1201", calls[0])
        self.assertEqual(recorded, "gfx1201")

    def test_unknown_source_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no_such.hip"):
            self.build("--sources", "no_such.hip")

    def test_unsupported_targets_are_refused(self):
        for arch in ("gfx1102", "gfx1103", "gfx1151", "gfx1170"):
            with self.subTest(arch=arch), self.assertRaisesRegex(SystemExit, "unsupported GPU target"):
                self.build("--arch", arch, "--no-runtime")

    def test_gfx11_is_not_a_runtime_or_extra_target(self):
        for args, message in ((["--arch", "gfx1100", "--runtime-only"], "gfx11 targets build code objects only"),
                              (["--arch", "gfx1100", "--motion-outres-only"], "gfx11 targets build code objects only"),
                              (["--extra-archs", "gfx1100", "--no-runtime"], "--extra-archs takes gfx12 targets only")):
            with self.subTest(args=args), self.assertRaisesRegex(SystemExit, message):
                self.build(*args)

if __name__ == "__main__":
    unittest.main()
