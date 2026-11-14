# SPDX-License-Identifier: MIT
"""Model L's raw model loading in the runtime (clean/l/runtime/l_model.cpp) on the CPU (#89):

- the tensor list against the preset-L map (data/preset-L.v1.map.json: stages, names, shapes, types, in order),
  against the structure table (clean/l/reference/table.py param_shapes: names, number of values) and, for the ten
  Swin stages and post, against Model M's raw tensors (clean/runner/synthetic_model.py);
- a synthetic Model L in the raw layout (clean/l/reference/oracle.py random_params) loads with every tensor's bytes
  unchanged;
- the refusals (missing tensor, wrong size, T above 127);
- model.wgcm with Model L's stages (synthetic file and map): the same tensors as from the raw layout; refused: a map
  shape not kept, a Model M file, a file whose digests the shipped preset-L map does not list.

No GPU: clean/l/runtime/tests/l_load_dump.cpp is built with the runtime's model.cpp by the host C++ compiler (CXX,
clang++ or g++; on Windows MSVC cl after them, from PATH or through vcvars64.bat; skipped without one).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_load.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.runner import synthetic_model as SM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
SOURCES = [os.path.join("clean", "l", "runtime", "tests", "l_load_dump.cpp"),
           os.path.join("clean", "l", "runtime", "l_model.cpp"),
           os.path.join("clean", "runtime", "model.cpp"), os.path.join("clean", "runtime", "embedded.cpp"),
           os.path.join("clean", "converter", "src", "sha256.cpp"),
           os.path.join("clean", "converter", "src", "json.cpp")]
SHARED = ("enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")
_EXE = None


def _vcvars64():
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    r = subprocess.run([vswhere, "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                       text=True)
    vcvars = os.path.join(r.stdout.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
    return vcvars if r.returncode == 0 and r.stdout.strip() and os.path.exists(vcvars) else None


def dump_exe():
    global _EXE
    if _EXE is None:
        tmp = tempfile.mkdtemp(prefix="wg_ll_")
        exe = os.path.join(tmp, "l_load_dump" + (".exe" if os.name == "nt" else ""))
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if cxx:
            cmd = [cxx, "-std=c++17", "-O1"] + (["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else [])
            r = subprocess.run(cmd + [os.path.join(ROOT, s) for s in SOURCES] + ["-o", exe], capture_output=True,
                               text=True)
        elif os.name == "nt" and (shutil.which("cl") or _vcvars64()):
            vcvars = None if shutil.which("cl") else _vcvars64()
            objdir = os.path.join(tmp, "obj")
            os.makedirs(objdir)
            cl = ('cl /nologo /std:c++17 /O1 /EHsc /utf-8 /D_CRT_SECURE_NO_WARNINGS %s /Fo"%s\\\\" /Fe"%s"'
                  % (" ".join('"%s"' % os.path.join(ROOT, s) for s in SOURCES), objdir, exe))
            bat = os.path.join(tmp, "build.bat")
            with open(bat, "w") as f:
                f.write("@echo off\r\n" + ('call "%s" >nul || exit /b 1\r\n' % vcvars if vcvars else "") +
                        cl + " || exit /b 1\r\n")
            r = subprocess.run(["cmd", "/d", "/c", bat], capture_output=True, text=True)
        else:
            raise unittest.SkipTest("no host C++ compiler (CXX, clang++, g++ or MSVC cl)")
        if r.returncode != 0:
            raise RuntimeError("build of l_load_dump failed:\n" + r.stdout + r.stderr)
        _EXE = exe
    return _EXE


def dump(*args):
    return subprocess.run([dump_exe(), *args], capture_output=True, text=True).stdout.strip().splitlines()


def specs():
    """{stage: {name: (shape, is_u8)}} in the C++ order (dicts keep it)."""
    out = {}
    for ln in dump("specs"):
        stage, name, dims, kind = ln.split(" ")
        out.setdefault(stage, {})[name] = (tuple(int(d) for d in dims.split("x")), kind == "u8")
    return out


def raw_model(seed):
    """A synthetic Model L as {stage: {name: array}} in the raw shapes and dtypes of the C++ list."""
    p = L.random_params(np.random.default_rng(seed))
    p["post"] = {"s": np.array([1.25], np.float32)}
    out = {}
    for stage, tensors in specs().items():
        out[stage] = {}
        for name, (shape, u8) in tensors.items():
            a = np.asarray(p[stage][name])
            out[stage][name] = a.astype(np.uint8).reshape(shape) if u8 else a.astype("<f4").reshape(shape)
    return out


def write_raw(model_dir, model):
    for stage, tensors in model.items():
        d = os.path.join(model_dir, "raw", stage)
        os.makedirs(d, exist_ok=True)
        for name, a in tensors.items():
            np.ascontiguousarray(a).tofile(os.path.join(d, name + ".bin"))


def numel(shape):
    return int(np.prod(shape)) if shape else 1


class Specs(unittest.TestCase):
    def test_map(self):
        """Stages, names, shapes and types of the preset-L map (data/preset-L.v1.map.json), in its order."""
        with open(os.path.join(ROOT, "data", "preset-L.v1.map.json"), encoding="utf-8") as f:
            m = json.load(f)
        want = {st["stage"]: {t["name"]: (tuple(t["shape"]), t["dtype"] == "uint8") for t in st["tensors"]}
                for st in m["stages"]}
        s = specs()
        self.assertEqual(list(s), list(want))
        for stage in want:
            self.assertEqual(list(s[stage].items()), list(want[stage].items()), stage)

    def test_table(self):
        """The structure table's parameters (table.param_shapes): same stages and names, same number of values."""
        s = specs()
        for stage, want in T.param_shapes().items():
            self.assertEqual(sorted(s[stage]), sorted(want), stage)
            for name, shape in want.items():
                self.assertEqual(numel(s[stage][name][0]), numel(shape), "%s/%s" % (stage, name))
        self.assertEqual(list(s)[0], "enc0")
        self.assertEqual(list(s["enc0"])[0], "W")  # the raw root is found by enc0/W.bin, as for Model M

    def test_shared_stages_are_model_m(self):
        s, m = specs(), SM.synthetic_tensors(1)
        for stage in SHARED + ("post",):
            self.assertEqual({k: v[0] for k, v in s[stage].items()}, {k: a.shape for k, a in m[stage].items()}, stage)
        for stage, names in (("enc0", ("b", "T", "sigma1", "sigma2")), ("dec0", ("S", "T"))):
            for name in names:  # Model M's raw shapes (L: projection 64 x 128, dec0 32 outputs per sub-pixel)
                self.assertEqual(s[stage][name][0], m[stage][name].shape, name)


class Load(unittest.TestCase):
    def test_bytes(self):
        model = raw_model(4)
        d = tempfile.mkdtemp(prefix="wg_ll_")
        try:
            write_raw(d, model)
            lines = dump("load", d)
            self.assertEqual(lines[0], "OK %d" % len(model))
            got = {tuple(ln.split(" ")[:2]): ln.split(" ")[2:] for ln in lines[1:]}
            want = {(st, n): [str(a.size), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()]
                    for st, ts in model.items() for n, a in ts.items()}
            self.assertEqual(got, want)
            # model_dir may also be raw/ itself
            self.assertEqual(dump("load", os.path.join(d, "raw")), lines)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_refusals(self):
        model = raw_model(5)
        d = tempfile.mkdtemp(prefix="wg_ll_")
        try:
            write_raw(d, model)
            raw = d + "/raw"  # as the loader names it
            f = os.path.join(d, "raw", "dec0_block", "W1.bin")
            os.remove(f)
            self.assertEqual(dump("load", d), ["ERR model data missing: " + raw + "/dec0_block/W1.bin"])
            model["dec0_block"]["W1"][:64].tofile(f)
            self.assertEqual(dump("load", d), ["ERR model data damaged: " + raw + "/dec0_block/W1.bin has 8192 bytes,"
                                                                              " expected 16384"])
            model["dec0_block"]["W1"].tofile(f)
            t = model["enc0"]["T"].copy()
            t.flat[7] = 200
            t.tofile(os.path.join(d, "raw", "enc0", "T.bin"))
            self.assertEqual(dump("load", d), ["ERR model data damaged: " + raw + "/enc0/T.bin has a value above 127"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Wgcm(unittest.TestCase):
    """model.wgcm (clean/converter/OUTPUT-FORMAT.md) with Model L's stages: a synthetic file and its synthetic map
    (synthetic_model.write_wgcm in the map's stage order), as the converter writes with the preset-L map."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="wg_lw_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write(self, model, sub="L"):
        """(model.wgcm path, synthetic map path) in <tmp>/<sub>."""
        d = os.path.join(self.d, sub)
        return os.path.join(d, "model.wgcm"), SM.write_wgcm(d, model, order=list(specs()))

    def test_bytes(self):
        model = raw_model(7)
        path, map_path = self.write(model)
        lines = dump("wgcm", path, map_path)
        self.assertEqual(lines[0], "OK %d" % len(model))
        got = {tuple(ln.split(" ")[:2]): ln.split(" ")[2:] for ln in lines[1:]}
        want = {(st, n): [str(a.size), hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()]
                for st, ts in model.items() for n, a in ts.items()}
        self.assertEqual(got, want)
        write_raw(os.path.join(self.d, "raw_copy"), model)  # the same tensors as from the raw layout
        self.assertEqual(dump("load", os.path.join(self.d, "raw_copy")), lines)

    def test_refusals(self):
        model = raw_model(8)
        bad = {st: dict(ts) for st, ts in model.items()}
        bad["dec0"]["W"] = bad["dec0"]["W"].reshape(8, 16, 64)  # same values, not the map's shape
        path, map_path = self.write(bad)
        self.assertEqual(dump("wgcm", path, map_path),
                         ["ERR model data damaged: %s: tensor dec0/W has the wrong shape or type" % path])
        m_dir = os.path.join(self.d, "M")  # a Model M file (with its map) is not a Model L file
        m_map, m_path = SM.write_wgcm(m_dir, SM.synthetic_tensors(1)), os.path.join(m_dir, "model.wgcm")
        self.assertEqual(dump("wgcm", m_path, m_map),
                         ["ERR model data damaged: %s: tensor enc0/W has the wrong shape or type" % m_path])
        path, _ = self.write(model)  # the shipped preset-L map: its digests are not those of synthetic data
        self.assertEqual(dump("wgcm", path, os.path.join(ROOT, "data", "preset-L.v1.map.json")),
                         ["ERR model data does not match this installation (converted with a different map): " + path])


if __name__ == "__main__":
    unittest.main()
