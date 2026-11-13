# SPDX-License-Identifier: MIT
"""The C++ model K layer table (clean/k/runtime/model_k.cpp) against the Python layout and packing, and the runtime's
model selection (runtime option model=K accepted, unknown models refused; model M is unchanged; without a converted
model K the runtime library's create fails with a plain message).

Run: python -m unittest discover -s clean/k/tests -t .
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

from clean.k.reference import layout as LY
from clean.k.runner import k_pack

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME_K = os.path.join(os.path.dirname(HERE), "runtime")
RUNTIME = os.path.join(os.path.dirname(os.path.dirname(HERE)), "runtime")

DUMP = r"""
#include <cstdio>
#include <string>
#include "model_k.h"
#include "wg_runtime.h"
using namespace wgrt;
int main() {
  std::printf("{\"stages\": [");
  const auto& t = k_stage_table();
  for (size_t i = 0; i < t.size(); ++i) {
    const KStageInfo& s = t[i];
    std::printf("%s{\"name\": \"%s\", \"kind\": \"%c\", \"level\": %d, \"C\": %d, \"heads\": %d, \"fixed\": %d, "
                "\"C_next\": %d, \"C_prev\": %d, \"embed\": %d, \"head\": %d, \"D\": %d, \"blob\": %zu}",
                i ? ", " : "", s.name, s.kind, s.level, s.C, s.heads, (int)s.fixed, s.C_next, s.C_prev,
                (int)s.embed, (int)s.head, s.D(), k_stage_blob_bytes(s));
  }
  std::printf("], \"base\": [");
  const int outs[][2] = {{1920, 1080}, {2560, 1440}, {3840, 2160}, {640, 360}, {1000, 777}, {7680, 4320}};
  for (int i = 0; i < 6; ++i) {
    int wb, hb;
    k_base_size(outs[i][0], outs[i][1], 1.0, &wb, &hb);
    std::printf("%s[%d, %d]", i ? ", " : "", wb, hb);
  }
  std::printf("], \"offsets\": [");
  const int table[8][2] = {{0, 7}, {1, 6}, {2, 5}, {3, 4}, {4, 3}, {5, 2}, {6, 1}, {7, 0}};  // SYNTHETIC_OFFSETS
  for (int f = 0; f < 16; ++f) {
    int off[11][2];
    k_stage_offsets(f, table, off);
    std::printf("%s[", f ? ", " : "");
    for (int i = 0; i < 11; ++i) std::printf("%s[%d, %d]", i ? ", " : "", off[i][0], off[i][1]);
    std::printf("]");
  }
  std::printf("], \"plans\": [");
  const int bases[][2] = {{960, 544}, {1280, 736}, {1920, 1088}, {256, 256}, {300, 270}};
  for (int i = 0; i < 5; ++i) {
    std::printf("%s{", i ? ", " : "");
    const auto plan = k_buffer_plan(bases[i][0], bases[i][1]);
    for (size_t j = 0; j < plan.size(); ++j)
      std::printf("%s\"%s\": %zu", j ? ", " : "", plan[j].name.c_str(), plan[j].bytes);
    std::printf("}");
  }
  std::printf("], \"select\": {");
  const char* opts[] = {"", "model=M", "graph=1;model=K", "model=X"};
  for (int i = 0; i < 4; ++i) {
    WgRtConfig c = {};
    c.runtime_options = opts[i];
    std::string err;
    const bool ok = model_k_create_check(c, &err);
    std::printf("%s\"%s\": %d", i ? ", " : "", opts[i], (int)ok);
  }
  std::printf("}}\n");
}
"""


def cxx():
    c = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not c:
        raise unittest.SkipTest("no host C++ compiler")
    return c


class TestRuntimeTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = tempfile.mkdtemp(prefix="wg_k_table_")
        src = os.path.join(d, "dump.cpp")
        with open(src, "w") as f:
            f.write(DUMP)
        exe = os.path.join(d, "dump")
        env = dict(os.environ)
        env.pop("WG_RUNTIME_OPTIONS", None)
        r = subprocess.run([cxx(), "-std=c++17", "-O1", "-I", RUNTIME_K, "-I", RUNTIME, src,
                            os.path.join(RUNTIME_K, "model_k.cpp"), "-o", exe], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        cls.data = json.loads(subprocess.run([exe], capture_output=True, text=True, check=True, env=env).stdout)

    def test_stages(self):
        got = self.data["stages"]
        self.assertEqual([s["name"] for s in got], LY.STAGE_NAMES)
        for g, st in zip(got, LY.STAGES):
            self.assertEqual(g["kind"], st["kind"][0])
            for k in ("level", "C", "heads", "D"):
                self.assertEqual(g[k], st[k], (st["name"], k))
            self.assertEqual(bool(g["fixed"]), st["mix"] == "fixed")
            self.assertEqual(g["C_next"], st.get("C_next", 0))
            self.assertEqual(g["C_prev"], st.get("C_prev", 0))
            self.assertEqual(bool(g["embed"]), bool(st.get("embed")))
            self.assertEqual(bool(g["head"]), bool(st.get("head")))
            self.assertEqual(g["blob"], k_pack.stage_sections(st)["_size"], st["name"])

    def test_sizes_and_offsets(self):
        outs = [(1920, 1080), (2560, 1440), (3840, 2160), (640, 360), (1000, 777), (7680, 4320)]
        self.assertEqual([tuple(b) for b in self.data["base"]], [LY.base_size(*o, 1.0) for o in outs])
        for f, offs in enumerate(self.data["offsets"]):
            ref = LY.stage_offsets(f, LY.SYNTHETIC_OFFSETS)
            self.assertEqual([tuple(o) for o in offs], [ref[n] for n in LY.STAGE_NAMES])

    def test_buffer_plans(self):
        bases = [(960, 544), (1280, 736), (1920, 1088), (256, 256), (300, 270)]
        for b, plan in zip(bases, self.data["plans"]):
            self.assertEqual(plan, LY.buffer_plan(*b))

    def test_model_selection(self):
        self.assertEqual(self.data["select"], {"": 1, "model=M": 1, "graph=1;model=K": 1, "model=X": 0})


class TestRuntimeLibrary(unittest.TestCase):
    """The real runtime library (host emulator backend): model=K without a converted model K refuses create with a
    message; model M runs."""

    def test_create(self):
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("the lane emulator needs Linux")
        from clean.tests import test_runtime as TR
        mdir = TR.model_dir()
        with self.assertRaises(RuntimeError) as cm:
            TR.make_pipeline(mdir, kernels={"options": "model=K"})
        self.assertIn("model K", str(cm.exception))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = TR.make_pipeline(mdir, kernels={"options": "model=M"})
        rng = np.random.default_rng(0)
        out = p.frame(TR.frame_inputs(rng, 24, 20, reset=True))
        self.assertTrue(np.all(np.isfinite(out["out"])))
        del p


if __name__ == "__main__":
    unittest.main()
