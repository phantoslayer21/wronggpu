# SPDX-License-Identifier: MIT
"""The build script's optional models (clean/runner/build.py OPTIONAL_MODELS, #89): model K's row is the former
--no-model-k switch, and a further optional model plugs in with its own row, its own --no-model-<name> option and its
own define, without changing what model M or model K get. No GPU, no compiler.

Run: python -m unittest clean.tests.test_build_models
"""
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from clean.runner import build


class OptionalModels(unittest.TestCase):
    def setUp(self):
        self.saved = (list(build.RUNTIME_SOURCES), list(build.EMBED_FILES), dict(build.MODELS_BUILT), build.MODEL_K)

    def tearDown(self):  # the module's lists exactly as before the test (other tests share them)
        build.OPTIONAL_MODELS.pop("T", None)
        build.RUNTIME_SOURCES[:], build.EMBED_FILES[:] = self.saved[0], self.saved[1]
        build.MODELS_BUILT.clear()
        build.MODELS_BUILT.update(self.saved[2])
        build.MODEL_K = self.saved[3]

    def test_k_row_is_the_old_switch(self):
        k = build.OPTIONAL_MODELS["K"]
        self.assertIs(k["sources"], build.K_RUNTIME_SOURCES)
        self.assertIs(k["embed"], build.K_EMBED_FILES)
        self.assertEqual((k["map"], k["define"], k["stand_in"]), (build.K_MAP_FILE, "WG_NO_MODEL_K",
                                                                   build.NO_MODEL_K_SOURCE))
        self.assertEqual(list(build.OPTIONAL_MODELS), ["K"])
        on = (list(build.RUNTIME_SOURCES), list(build.EMBED_FILES))
        build.set_model("K", False)
        self.assertFalse(build.model_built("K") or build.MODEL_K)
        off = (list(build.RUNTIME_SOURCES), list(build.EMBED_FILES))
        build.set_model_k(True)
        self.assertTrue(build.model_built("K") and build.MODEL_K)
        self.assertEqual((build.RUNTIME_SOURCES, build.EMBED_FILES), on)
        build.set_model_k(False)
        self.assertEqual((build.RUNTIME_SOURCES, build.EMBED_FILES), off)  # the same lists by either name

    def test_second_optional_model(self):
        t_src = [os.path.join("..", "t", "runtime", "model_t.cpp")]
        build.OPTIONAL_MODELS["T"] = {"sources": t_src, "embed": ["t_block.co"], "map": build.MAP_FILE,
                                      "define": "WG_NO_MODEL_T", "stand_in": "no_model_t.cpp",
                                      "build_device": lambda *a, **k: None, "help": "a library without model T"}
        build.MODELS_BUILT["T"] = False
        build.set_model("T", True)
        self.assertEqual(build.RUNTIME_SOURCES[-1:], t_src)
        self.assertIn("t_block.co", build.EMBED_FILES)
        k_on = list(build.K_RUNTIME_SOURCES)
        build.set_model("T", False)
        self.assertNotIn(t_src[0], build.RUNTIME_SOURCES)
        self.assertIn("no_model_t.cpp", build.RUNTIME_SOURCES)
        self.assertNotIn("t_block.co", build.EMBED_FILES)
        cmd = build.runtime_command("c++", "x")
        self.assertIn("-DWG_NO_MODEL_T", cmd)
        self.assertNotIn("-DWG_NO_MODEL_K", cmd)  # model K untouched
        self.assertTrue(all(s in build.RUNTIME_SOURCES for s in k_on))
        build.set_model("K", False)
        cmd = build.runtime_command("c++", "x")
        self.assertIn("-DWG_NO_MODEL_K", cmd)
        self.assertIn("-DWG_NO_MODEL_T", cmd)
        # its option is generated from the table
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit):
            build.main(["--help"])
        self.assertIn("--no-model-t", buf.getvalue())
        self.assertIn("--no-model-k", buf.getvalue())

    def test_maps_copied(self):
        copied = []
        with patch.object(build.shutil, "copyfile", lambda a, b: copied.append(os.path.basename(a))), \
                patch.object(build, "write_embedded", lambda d: None), patch.object(build, "run", lambda c: None):
            build.build_runtime("c++", "out", embed=False)
            self.assertEqual(copied, ["preset-M.v1.map.json", "preset-K.v1.map.json"])
            copied.clear()
            build.set_model("K", False)
            build.build_runtime("c++", "out", embed=False)
            self.assertEqual(copied, ["preset-M.v1.map.json"])


if __name__ == "__main__":
    unittest.main()
