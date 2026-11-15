# SPDX-License-Identifier: MIT
"""Frame generation (#96): the real-file check tool (clean/fg/tools/real_check.py) on made-up converter output and a
stand-in converter (a small Python script written by the test). No library file is read.

    python -m unittest clean.fg.tests.test_fg_real_check
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

from clean.fg.tools import real_check as R

MAP_SHA = "ab" * 32
LOG_OK = ("[wronggpu-convert] map sha256 %s, 42 stages, model: x\n"
          "[wronggpu-convert] size: 7000000 bytes\n"
          "[wronggpu-convert] file version: 310.9.1.0\n"
          "[wronggpu-convert] source A01 at offset 1234567\n"
          "[wronggpu-convert] all 42 stage digests match (0.27 s)\n") % MAP_SHA

STUB = r'''
import os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
mode = os.environ.get("FG_STUB_MODE", "ok")
sys.stderr.write(%r)
if mode == "ok":
    os.makedirs(out)
    open(os.path.join(out, "model.wgcm"), "wb").write(b"made-up model file")
    print("The model was converted successfully.")
    sys.exit(0)
print("A made-up failure message.")
sys.exit(13)
''' % LOG_OK


class Parse(unittest.TestCase):
    def test_pass(self):
        r = R.parse(0, R.SUCCESS + "\n", LOG_OK)
        self.assertEqual((r["verdict"], r["map_sha256"], r["file_version"], r["file_size"]),
                         ("pass", MAP_SHA, "310.9.1.0", 7000000))
        self.assertEqual((r["stages_in_map"], r["stage_digests_matched"]), (42, 42))

    def test_fail_forms(self):
        self.assertEqual(R.parse(13, "A made-up failure message.", LOG_OK)["verdict"], "fail")
        self.assertEqual(R.parse(0, R.SUCCESS, LOG_OK.replace("all 42 stage", "all 41 stage"))["verdict"], "fail")
        self.assertEqual(R.parse(0, R.SUCCESS, "")["verdict"], "fail")
        self.assertEqual(R.parse(0, "", LOG_OK)["verdict"], "fail")

    def test_record_holds_no_offsets(self):
        text = json.dumps(R.parse(0, R.SUCCESS, LOG_OK))
        self.assertNotIn("1234567", text)
        self.assertNotIn("offset", text)


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fg-rc-test-")
        self.stub = os.path.join(self.tmp, "stub.py")
        with open(self.stub, "w", encoding="utf-8") as f:
            f.write(STUB)
        self.map = os.path.join(self.tmp, "map.json")
        with open(self.map, "wb") as f:
            f.write(b"{}")

    def tearDown(self):
        os.environ.pop("FG_STUB_MODE", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pass_removes_the_model(self):
        work = os.path.join(self.tmp, "w1")
        r = R.run([sys.executable, "-I", self.stub], self.map, "C:/made/up/path.dll", work)
        self.assertEqual(r["verdict"], "pass")
        self.assertEqual(r["model_size"], len(b"made-up model file"))
        self.assertEqual(len(r["model_sha256"]), 64)
        self.assertFalse(os.path.exists(os.path.join(work, "out")))
        self.assertNotIn("1234567", json.dumps(r))

    def test_fail(self):
        os.environ["FG_STUB_MODE"] = "fail"
        r = R.run([sys.executable, "-I", self.stub], self.map, "C:/made/up/path.dll", os.path.join(self.tmp, "w2"))
        self.assertEqual((r["verdict"], r["exit_code"], r["model_sha256"]), ("fail", 13, None))

    def test_refuses_a_used_work_folder(self):
        work = os.path.join(self.tmp, "w3")
        os.makedirs(os.path.join(work, "out"))
        with self.assertRaises(SystemExit):
            R.run([sys.executable, "-I", self.stub], self.map, "x.dll", work)


if __name__ == "__main__":
    unittest.main()
