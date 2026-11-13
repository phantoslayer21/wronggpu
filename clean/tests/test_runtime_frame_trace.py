# SPDX-License-Identifier: MIT
"""The runtime's frame sequence on the CPU (#89): clean/tests/runtime_frame_trace.cpp runs the runtime
(clean/runtime/runtime.cpp) on a recording device, so whole frames run without a GPU and without HIP. Launches are
recorded, not executed: the test checks the sequence (code objects loaded, kernels launched per frame with grid and
block), not pixels. The base for checking other models' frames against Model M's (Model L: Model M's frame with the
encoder and decoder steps replaced).

- a synthetic Model M (converter raw layout): create, the code objects loaded, the launches of four frames;
- the same run twice gives the same trace;
- model=L is refused at create for now.

The runtime options of the environment (WG_*) are removed for the runs. Built with the host C++ compiler
(clean/tests/host_cxx.py; skipped without one).

Run: python -m unittest clean.tests.test_runtime_frame_trace
"""
import os
import shutil
import subprocess
import tempfile
import unittest

from clean.runner import synthetic_model as SM
from clean.tests import host_cxx

SOURCES = [os.path.join("clean", "tests", "runtime_frame_trace.cpp")]
SOURCES += [os.path.join("clean", "runtime", s) for s in ("device.cpp", "embedded.cpp", "model.cpp", "runtime.cpp",
                                                          "no_model_k.cpp")]
SOURCES += [os.path.join("clean", "converter", "src", s) for s in ("sha256.cpp", "json.cpp")]
_EXE = None
_MODEL = None


def trace_exe():
    global _EXE
    if _EXE is None:
        _EXE = host_cxx.build("runtime_frame_trace", SOURCES, ["WG_NO_MODEL_K"])
    return _EXE


def model_dir():
    global _MODEL
    if _MODEL is None:
        d = tempfile.mkdtemp(prefix="wg_ft_")
        SM.write_raw(d, SM.synthetic_tensors(1))
        _MODEL = d
    return _MODEL


def run_trace(rw=64, rh=64, ow=96, oh=96, frames=3, options=""):
    """The program's output lines. The build folder is empty: the recording device loads nothing."""
    build = tempfile.mkdtemp(prefix="wg_ftb_")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
    try:
        r = subprocess.run([trace_exe(), build, model_dir(), str(rw), str(rh), str(ow), str(oh), str(frames), options],
                           capture_output=True, text=True, env=env, timeout=600)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if r.returncode != 0:
        raise AssertionError("runtime_frame_trace failed (exit %d)\n%s%s" % (r.returncode, r.stdout, r.stderr))
    return r.stdout.splitlines()


def parse(lines):
    """(create line, [loaded code objects], [[(kernel, grid, block)] per frame], kernels line)."""
    loads, frames, kernels = [], [], None
    for ln in lines[1:]:
        if ln.startswith("load "):
            loads.append(ln[5:])
        elif ln.startswith("frame "):
            frames.append([])
        elif ln.startswith("launch "):
            k, g, b = ln[7:].split(" ")
            frames[-1].append((k, int(g), int(b)))
        elif ln.startswith("kernels "):
            kernels = dict(kv.split("=", 1) for kv in ln[8:].split(" ") if "=" in kv)
        else:
            raise AssertionError("unexpected line: " + ln)
    return lines[0], loads, frames, kernels


class ModelM(unittest.TestCase):
    def test_frames(self):
        create, loads, frames, kernels = parse(run_trace(frames=4))
        self.assertEqual(create, "create ok model M")
        for key in ("swin", "enc0", "dec0"):
            self.assertIn(kernels[key], loads)
        self.assertEqual(len(frames), 4)
        for f in frames:
            names = [k for k, _, _ in f]
            self.assertGreater(len(names), 10)
            self.assertTrue(all(g > 0 and b > 0 for _, g, b in f))
            self.assertEqual(names[0], "exposure_pair_diet")  # exposure first, then the encoder
            self.assertIn(kernels["post"], names)
        # the window offsets alternate from frame to frame (grids of the Swin stages change), so the launches repeat
        # every second frame
        self.assertNotEqual(frames[0], frames[1])
        self.assertEqual(frames[0], frames[2])
        self.assertEqual(frames[1], frames[3])

    def test_repeatable(self):
        self.assertEqual(run_trace(rw=80, rh=48, ow=120, oh=72), run_trace(rw=80, rh=48, ow=120, oh=72))

    def test_model_l_refused(self):
        lines = run_trace(options="model=L")
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("ERR create: ") and "Model L" in lines[0], lines[0])


if __name__ == "__main__":
    unittest.main()
