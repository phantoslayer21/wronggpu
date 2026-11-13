# SPDX-License-Identifier: MIT
"""Runtime model loading from the converter's model.wgcm (spec 10 section 7; clean/converter/OUTPUT-FORMAT.md).

The real converter (clean/converter) is built and run on a synthetic "DLL" with the layout of the shipped preset-M
map but random content (clean/converter/tests/synth.py: shipped_like_case), so model.wgcm and raw/ are genuine
converter output from synthetic data. The runtime must load model.wgcm (its packing must equal pack_weights on the
raw tensors) and must reject damaged files, files converted with another map, and missing data.

Run: python -m unittest discover -s clean/tests -t .
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

from clean.runner.pipeline import DEFAULT_MAP, Pipeline
from clean.tests.test_kernels_emu import EmuTest
from clean.tests.test_runtime import runtime_dir

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONV = os.path.join(ROOT, "clean", "converter")
CONV_SOURCES = ["sha256.cpp", "json.cpp", "numeric.cpp", "map.cpp", "locate.cpp", "convert.cpp", "output.cpp",
                "pe_version.cpp", "main.cpp"]
_CASE = None


def converted_case():
    """(work dir, map path, output dir) of one converter run on a shipped-like synthetic case."""
    global _CASE
    if _CASE is None:
        if not sys.platform.startswith("linux"):
            raise unittest.SkipTest("builds the converter with a Linux toolchain")
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        work = tempfile.mkdtemp(prefix="wg_conv_")
        exe = os.path.join(work, "wronggpu-convert")
        cmd = [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I", os.path.join(CONV, "src")]
        cmd += [os.path.join(CONV, "src", s) for s in CONV_SOURCES] + ["-o", exe, "-pthread"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("converter build failed:\n" + r.stderr)
        sys.path.insert(0, os.path.join(CONV, "tests"))
        import synth  # noqa: E402  (the converter's own test-case builder)
        case = synth.shipped_like_case(DEFAULT_MAP, seed=3)
        dll, mp, out = os.path.join(work, "fake.dll"), os.path.join(work, "map.json"), os.path.join(work, "out")
        with open(dll, "wb") as f:
            f.write(case.file)
        with open(mp, "w") as f:
            json.dump(case.map, f)
        r = subprocess.run([exe, "--map", mp, "--dll", dll, "--out", out, "--raw"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"converter failed ({r.returncode}): {r.stdout} {r.stderr}")
        _CASE = (work, mp, out)
    return _CASE


def make(model_dir, map_path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return Pipeline(runtime_dir(), "", model_dir, (24, 20), (16, 12), {}, backend="emu", map_path=map_path, **kw)


def wgcm_only(tag, mutate=None):
    """Copy of model.wgcm alone in a new folder, optionally modified (bytes -> bytes)."""
    _, _, out = converted_case()
    d = tempfile.mkdtemp(prefix=f"wg_{tag}_")
    with open(os.path.join(out, "model.wgcm"), "rb") as f:
        data = f.read()
    with open(os.path.join(d, "model.wgcm"), "wb") as f:
        f.write(mutate(data) if mutate else data)
    return d


def flip(data, offset):
    b = bytearray(data)
    b[offset] ^= 0x10
    return bytes(b)


def rehash(data):
    return data[:-32] + hashlib.sha256(data[:-32]).digest()


class WgcmTest(EmuTest):
    def test_loads_converter_output(self):
        _, mp, out = converted_case()
        p = make(out, mp)                     # check_packing: wgcm-loaded blobs == pack_weights(raw tensors)
        self.assertEqual(p.model_source, "model.wgcm")
        q = make(out, mp, model_format="raw")
        self.assertEqual(q.model_source, "raw")
        for stage in ("enc0", "b3", "dec0", "post"):
            self.assertEqual(p.stage_blob(stage), q.stage_blob(stage), stage)
        make(wgcm_only("plain"), mp)          # without raw/: the Python check reads model.wgcm itself

    def test_rejects_damage_and_wrong_map(self):
        _, mp, _ = converted_case()
        size = os.path.getsize(os.path.join(converted_case()[2], "model.wgcm"))
        cases = {
            "integrity": (lambda d: flip(d, size // 2), mp, "integrity check"),   # inside the large tensors
            "digest": (lambda d: rehash(flip(d, size // 2)), mp, "digest"),
            "truncated": (lambda d: d[:1000], mp, "integrity check|not a WrongGPU"),
            "magic": (lambda d: rehash(b"XXXX" + d[4:]), mp, "not a WrongGPU model file"),
            "other_map": (None, DEFAULT_MAP, "different map"),
            "no_map": (None, os.path.join(tempfile.gettempdir(), "no-such-map.json"), "converter map missing"),
        }
        for name, (mutate, map_path, msg) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(RuntimeError, msg):
                    make(wgcm_only(name, mutate), map_path, model_format="wgcm", check_packing=False)

    def test_missing_model(self):
        with self.assertRaisesRegex(RuntimeError, "model data missing"):
            make(tempfile.mkdtemp(prefix="wg_empty_"), converted_case()[1], check_packing=False)


if __name__ == "__main__":
    unittest.main()
