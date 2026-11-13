# SPDX-License-Identifier: MIT
"""Runtime library without model K (clean/runner/build.py --no-model-k). No GPU, no model data.

  python -m unittest clean.tests.test_no_model_k                  build lists only (own fixtures)
  WG_NO_K_BUILD=<folder of a --no-model-k build> python -m unittest clean.tests.test_no_model_k
        also the built library: nothing of model K built in, model=K refused at create, model M still selected
  WG_K_BUILD=<folder of a default build>   optional positive control for the scan (finds model K there)
"""
import ctypes
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clean.runner import build

NO_K = os.environ.get("WG_NO_K_BUILD", "")
WITH_K = os.environ.get("WG_K_BUILD", "")
# what only a library with model K holds: names of its built-in files, texts of its runtime and of its interop branches
K_MARKS = [b"k_block.co", b"k_block_wide.co", b"k_io.co", b"preset-K.v1.map.json", b"model K:", b"network model K"]
K_NAME = re.compile(r"(^|\.)k_|preset-K")


def table_names(folder):
    text = (Path(folder) / "embedded_files.cpp").read_text()
    return re.findall(r'\{"([^"]+)", f\d+, \d+\}', text)


def create_error(folder, options):
    """wg_rt_create's error text for the runtime options (sizes 0: model M stops at its size check, before any
    device or file), or "" when it succeeds."""
    from clean.runner.pipeline import WgRtConfig
    lib = ctypes.CDLL(str(Path(folder) / build.runtime_library_name()))
    lib.wg_rt_create.argtypes = [ctypes.POINTER(WgRtConfig), ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                 ctypes.c_int]
    cfg = WgRtConfig()
    cfg.build_dir = str(folder).encode()
    cfg.runtime_options = options.encode()
    rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(512)
    return "" if lib.wg_rt_create(ctypes.byref(cfg), ctypes.byref(rt), err, 512) == 0 else err.value.decode()


class BuildLists(unittest.TestCase):
    def tearDown(self):
        build.set_model_k(True)

    def test_lists(self):
        on_src, on_files = list(build.RUNTIME_SOURCES), list(build.EMBED_FILES)
        self.assertEqual(len(build.K_RUNTIME_SOURCES), 5)
        build.set_model_k(False)
        self.assertEqual([s for s in build.RUNTIME_SOURCES if os.sep + "k" + os.sep in s], [])
        self.assertIn(build.NO_MODEL_K_SOURCE, build.RUNTIME_SOURCES)
        self.assertEqual([n for n in build.EMBED_FILES if K_NAME.search(n)], [])
        self.assertEqual(build.EMBED_FILES, [n for n in on_files if n not in build.K_EMBED_FILES])  # model M: the same
        self.assertIn("-DWG_NO_MODEL_K", build.runtime_command("c++", "x"))
        build.set_model_k(True)
        self.assertEqual((build.RUNTIME_SOURCES, build.EMBED_FILES), (on_src, on_files))
        self.assertNotIn("-DWG_NO_MODEL_K", build.runtime_command("c++", "x"))

    def test_table_without_k_objects(self):
        """The table builds from a folder that holds no model K file, for both targets; model K files lying there
        are left out."""
        build.set_model_k(False)
        with tempfile.TemporaryDirectory() as d, patch.object(build, "EXTRA_ARCHS", ["gfx1200"]):
            root = Path(d)
            (root / "gfx1200").mkdir()
            for name in build.EMBED_FILES:
                (root / name).write_bytes(b"own synthetic fixture")
                if name.endswith(".co"):
                    (root / "gfx1200" / name).write_bytes(b"own gfx1200 fixture")
            build.write_embedded(d)
            first = table_names(d)
            for name in build.K_EMBED_FILES:
                (root / name).write_bytes(b"own stray fixture")
                (root / "gfx1200" / name).write_bytes(b"own stray fixture")
            build.write_embedded(d)
            self.assertEqual(table_names(d), first)
            self.assertEqual([n for n in first if K_NAME.search(n)], [])


@unittest.skipUnless(NO_K, "WG_NO_K_BUILD not set")
class BuiltLibrary(unittest.TestCase):
    def test_nothing_of_model_k(self):
        names = table_names(NO_K)
        self.assertTrue("preset-M.v1.map.json" in names and "gfx1201.post_w64.co" in names)
        self.assertEqual([n for n in names if K_NAME.search(n)], [])
        data = (Path(NO_K) / build.runtime_library_name()).read_bytes()
        self.assertEqual([m.decode() for m in K_MARKS if m in data], [])
        self.assertIn(b"Model K is not part of this build", data)
        if WITH_K:  # the scan does find model K in a default library, whose model M files are the same list
            full = (Path(WITH_K) / build.runtime_library_name()).read_bytes()
            self.assertEqual([m.decode() for m in K_MARKS if m not in full], [])
            self.assertEqual([n for n in table_names(WITH_K) if not K_NAME.search(n)], names)

    def test_model_k_refused(self):
        old = os.environ.pop("WG_RUNTIME_OPTIONS", None)
        try:
            for options in ("model=K", "model=k", "graph=1;model=K"):
                self.assertIn("Model K is not part of this build", create_error(NO_K, options))
            for options in ("", "model=M", "model=K;model=M"):  # model M is selected: its own first check answers
                self.assertEqual(create_error(NO_K, options), "invalid render or output size")
            self.assertIn("unknown model", create_error(NO_K, "model=Q"))
        finally:
            if old is not None:
                os.environ["WG_RUNTIME_OPTIONS"] = old


if __name__ == "__main__":
    sys.exit(unittest.main())
