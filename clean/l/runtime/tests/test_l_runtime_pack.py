# SPDX-License-Identifier: MIT
"""Model L's runtime packer (clean/l/runtime/l_pack.cpp) against clean/l/runner/l_pack.py, byte for byte (#89):

- every blob of pack_model_l for random parameters (clean/l/reference/oracle.py random_params), three seeds: the
  encoder (values, the input linears Win1 / Win2, enc0_block, the projection) and the decoder (dec0.project,
  dec0_block, dec0.coef);
- the default table's encoder and decoder launch lists against l_pack.encoder_launches / decoder_launches, field by
  field;
- the refusals (missing stage, missing tensor, wrong size).

No GPU: clean/l/runtime/tests/l_pack_dump.cpp is built with the runtime's model.cpp by the host C++ compiler (CXX,
clang++ or g++; on Windows MSVC cl after them, from PATH or through vcvars64.bat; skipped without one).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_pack.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.l.runner import l_pack as LP

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
SOURCES = [os.path.join("clean", "l", "runtime", "tests", "l_pack_dump.cpp"),
           os.path.join("clean", "l", "runtime", "l_pack.cpp"),
           os.path.join("clean", "runtime", "model.cpp"), os.path.join("clean", "runtime", "embedded.cpp"),
           os.path.join("clean", "converter", "src", "sha256.cpp"),
           os.path.join("clean", "converter", "src", "json.cpp")]
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
        tmp = tempfile.mkdtemp(prefix="wg_lp_")
        exe = os.path.join(tmp, "l_pack_dump" + (".exe" if os.name == "nt" else ""))
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
            raise RuntimeError("build of l_pack_dump failed:\n" + r.stdout + r.stderr)
        _EXE = exe
    return _EXE


def dump(*args):
    return subprocess.run([dump_exe(), *args], capture_output=True, text=True).stdout.strip()


STAGES = ("enc0", "enc0_block", "dec0", "dec0_block")  # Model L's own stages (the map's names)


def write_params(root, params):
    """Model L's own stages as <stage>/<name>.f32 (and enc0/T.u8: the dither table as the runtime's loader keeps it)."""
    for stage in STAGES:
        if stage not in params:
            continue
        d = os.path.join(root, stage)
        os.makedirs(d)
        for name, a in params[stage].items():
            if stage == "enc0" and name == "T":
                np.asarray(a).astype(np.uint8).tofile(os.path.join(d, "T.u8"))
            else:
                np.ascontiguousarray(a, "<f4").reshape(-1).tofile(os.path.join(d, name + ".f32"))


def first_diff(got, want):
    if len(got) != len(want):
        return "length %d, expected %d" % (len(got), len(want))
    d = np.flatnonzero(np.frombuffer(got, np.uint8) != np.frombuffer(want, np.uint8))
    return "%d bytes differ, first at %d" % (len(d), d[0])


def pack_cpp(params):
    d = tempfile.mkdtemp(prefix="wg_lp_")
    write_params(d, params)
    out = os.path.join(d, "out")
    os.makedirs(out)
    res = dump("pack", d, out)
    blobs = {}
    if res.startswith("OK"):
        for f in os.listdir(out):
            with open(os.path.join(out, f), "rb") as fh:
                blobs[f[:-4]] = fh.read()
    return res, blobs


class Blobs(unittest.TestCase):
    def test_random_params(self):
        for seed in (0, 1, 2):
            params = L.random_params(np.random.default_rng(seed))
            want = LP.pack_model_l(params)
            res, got = pack_cpp(params)
            self.assertEqual(res, "OK %d" % len(want), "seed %d" % seed)
            self.assertEqual(sorted(got), sorted(want))
            for name, w in want.items():
                if got[name] != w:
                    self.fail("seed %d, %s: %s" % (seed, name, first_diff(got[name], w)))

    def test_refusals(self):
        params = L.random_params(np.random.default_rng(3))

        def without(stage, name=None, keep=None):
            p = {st: dict(params[st]) for st in STAGES}
            if name is None:
                del p[stage]
            elif keep is None:
                del p[stage][name]
            else:
                p[stage][name] = np.asarray(p[stage][name]).reshape(-1)[:keep]
            return pack_cpp(p)[0]

        self.assertEqual(without("enc0", "Win2"), "ERR enc0: missing tensor Win2")
        self.assertEqual(without("enc0_block"), "ERR missing stage enc0_block")
        self.assertEqual(without("dec0_block", "b1", 64), "ERR dec0_block: tensor b1 has 64 values, expected 128")
        self.assertEqual(without("enc0", "W", 4096), "ERR enc0: tensor W has 4096 values, expected 8192")
        self.assertEqual(without("dec0", "W", 4096), "ERR dec0: tensor W has 4096 values, expected 8192")


class Launches(unittest.TestCase):
    FIELDS = (("kind", ""), ("blob", ""), ("src", ""), ("dst", ""), ("C_in", 0), ("C_out", 0), ("w", ""),
              ("b", ""), ("acc", ""), ("out_round", ""), ("C", 0), ("heads", 0), ("mlp", 0), ("offsets", ""),
              ("params", ""), ("in_format", ""), ("out_format", ""), ("border", ""), ("dec_pixel", ""))

    def line(self, where, ln):
        f = [where] + [str(ln.get(k) if ln.get(k) is not None else d) for k, d in self.FIELDS]
        f.append(",".join(str(v) for v in ln.get("groups", ())))
        f.append("%.9g" % ln["clamp"] if ln.get("clamp") is not None else "")
        f.append(",".join(str(v) for v in ln.get("values", ())))
        return "|".join(f)

    def test_default_table(self):
        want = [self.line("enc", ln) for ln in LP.encoder_launches()]
        want += [self.line("dec", ln) for ln in LP.decoder_launches()]
        self.assertEqual(dump("launches").splitlines(), want)
        self.assertEqual(want[-1].split("|")[-1], "0,1,6,7,8,9,10,11,12,2,3,4,5")    # the exact 13-value sequence

    def test_map_identities(self):
        """Every launch reads and writes distinct maps (no kernel works in place), each map is written once, and every
        map read is written earlier (or is the stage input)."""
        for launches, inputs in ((LP.encoder_launches(), {"values"}), (LP.decoder_launches(), {"features"})):
            written = set(inputs)
            for ln in launches:
                src, dst = ln.get("src"), ln.get("dst")
                self.assertTrue(dst and dst != src, ln)
                self.assertNotIn(dst, written, ln)
                if src:
                    self.assertIn(src, written, ln)
                written.add(dst)
        fmts = [(ln["kind"], ln.get("in_format", ""), ln.get("out_format", ""), ln.get("border", ""))
                for ln in LP.encoder_launches() + LP.decoder_launches()]
        self.assertEqual(fmts, [("values", "", "fp8", ""), ("linear", "fp8", "fp8", ""), ("linear", "fp8", "fp16", ""),
                                ("swin", "fp16", "fp8", "input"), ("project", "fp8", "fp8", ""),
                                ("dec_project", "fp8", "fp8", ""), ("swin", "fp8", "fp16", "coarse"),
                                ("coef", "fp16", "", "")])


if __name__ == "__main__":
    unittest.main()
