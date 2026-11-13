# SPDX-License-Identifier: MIT
"""Packaging completeness uses tiny own fixtures, no GPU or model data."""
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


if __name__ == "__main__":
    unittest.main()
