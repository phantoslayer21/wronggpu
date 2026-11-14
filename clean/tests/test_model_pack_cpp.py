# SPDX-License-Identifier: MIT
"""The runtime's C++ packers (clean/runtime/model.cpp) against the Python packers, byte for byte, on the CPU (#89):

- every stage blob of a synthetic Model M (enc0, the ten Swin stages, dec0) loaded from the converter's raw layout:
  pack_stage_blob against clean/runner/enc0.py pack, clean/runner/packing.py pack_stage and clean/runner/dec0.py pack;
- the building block for other models' packers, pack_parts::pack_swin_block: at Model M's widths (equal to the block
  section of the stage blob) and at width 32 with a name prefix (Model L's blocks), against packing.pack_block, plus its
  refusals (missing tensor, wrong size).

No GPU and no model data: clean/tests/model_pack_dump.cpp is built with the host C++ compiler (CXX, clang++ or g++; on
Windows MSVC cl after them, from PATH or through vcvars64.bat; skipped without one).

Run: python -m unittest clean.tests.test_model_pack_cpp
"""
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clean.reference import swin as Sw
from clean.reference.fp8 import f16
from clean.runner import dec0 as D0
from clean.runner import enc0 as E0
from clean.runner import packing as P
from clean.runner import synthetic_model as SM
from clean.runner.pipeline import logical_tensors

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = [os.path.join("clean", "tests", "model_pack_dump.cpp"), os.path.join("clean", "runtime", "model.cpp"),
           os.path.join("clean", "runtime", "embedded.cpp"), os.path.join("clean", "converter", "src", "sha256.cpp"),
           os.path.join("clean", "converter", "src", "json.cpp")]
BLOCK_NAMES = ("gamma1", "gamma2", "Wqk", "Wv", "B", "Wp", "bp", "W1", "b1", "W2", "b2")
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
        tmp = tempfile.mkdtemp(prefix="wg_mp_")
        exe = os.path.join(tmp, "model_pack_dump" + (".exe" if os.name == "nt" else ""))
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
            raise RuntimeError("build of model_pack_dump failed:\n" + r.stdout + r.stderr)
        _EXE = exe
    return _EXE


def dump(*args):
    r = subprocess.run([dump_exe(), *args], capture_output=True, text=True)
    return r.stdout.strip()


def first_diff(got, want):
    if len(got) != len(want):
        return "length %d, expected %d" % (len(got), len(want))
    d = np.flatnonzero(np.frombuffer(got, np.uint8) != np.frombuffer(want, np.uint8))
    return "%d bytes differ, first at %d" % (len(d), d[0])


def block_tensors(C, heads, mlp, rng):
    """Synthetic block tensors at any width (FP8-exact matrices, FP16-exact vectors as the converter writes)."""
    D = heads * 32
    q8 = lambda a: np.asarray(Sw.q8(a), np.float32)  # noqa: E731
    t = {"gamma1": 0.5 + rng.random(C), "gamma2": 0.5 + rng.random(C),
         "Wqk": q8(rng.standard_normal((heads, 32, C)) / np.sqrt(C)), "Wv": q8(rng.standard_normal((heads, 32, C)) / np.sqrt(C)),
         "B": rng.standard_normal((heads, 16, 16)), "Wp": q8(rng.standard_normal((C, D)) / np.sqrt(D)),
         "bp": 0.1 * rng.standard_normal(C), "W1": q8(rng.standard_normal((mlp, C)) / np.sqrt(C)),
         "b1": 0.1 * rng.standard_normal(mlp), "W2": q8(rng.standard_normal((C, mlp)) / np.sqrt(mlp)),
         "b2": 0.1 * rng.standard_normal(C)}
    return {k: (np.asarray(v, np.float32) if k.startswith("W") else f16(v).astype(np.float32)) for k, v in t.items()}


def pack_block_py(t, C, heads, mlp):
    blob = P.pack_block(t, C, heads, mlp, P.resolve_d_row("gfx12"))
    return blob + bytes((-len(blob)) % P.ALIGN)


class StageBlobs(unittest.TestCase):
    def test_model_m_stages(self):
        for seed in (1, 2):
            t = SM.synthetic_tensors(seed)
            d = tempfile.mkdtemp(prefix="wg_mp_")
            SM.write_raw(d, t)
            out = os.path.join(d, "out")
            os.makedirs(out)
            self.assertEqual(dump("stages", d, out), "OK 12")
            lt = logical_tensors(t)  # the converter's raw shapes -> the Python packers' shapes (as the pipeline)
            want = {"enc0": E0.pack(lt["enc0"]), "dec0": D0.pack(lt["dec0"])}
            want.update({s: P.pack_stage(s, lt[s]) for s in Sw.STAGES})
            for s, w in want.items():
                with open(os.path.join(out, s + ".bin"), "rb") as f:
                    got = f.read()
                if got != w:
                    self.fail("seed %d, %s: %s" % (seed, s, first_diff(got, w)))


class SwinBlockBuildingBlock(unittest.TestCase):
    def run_block(self, t, prefix, C, heads, mlp):
        d = tempfile.mkdtemp(prefix="wg_mp_")
        for name, a in t.items():
            np.ascontiguousarray(a, "<f4").tofile(os.path.join(d, prefix + name + ".bin"))
        out = os.path.join(d, "block.bin")
        res = dump("block", d, prefix, str(C), str(heads), str(mlp), out)
        got = None
        if res.startswith("OK"):
            with open(out, "rb") as f:
                got = f.read()
        return res, got

    def test_model_m_widths_equal_the_stage_blocks(self):
        rng = np.random.default_rng(5)
        for s in ("enc1", "enc2", "b0", "dec2"):
            cfg = Sw.STAGES[s]
            p = {k: np.asarray(v, np.float32) for k, v in Sw.random_params(s, rng).items() if k in BLOCK_NAMES}
            res, got = self.run_block(p, "", cfg["C"], cfg["heads"], cfg["mlp"])
            want = pack_block_py(p, cfg["C"], cfg["heads"], cfg["mlp"])
            self.assertTrue(res.startswith("OK"), res)
            if got != want:
                self.fail("%s: %s" % (s, first_diff(got, want)))
            full = P.pack_stage(s, Sw.random_params(s, np.random.default_rng(9)))  # (another blob, same section size)
            self.assertEqual(len(want), P.stage_sections(s)[1])
            self.assertLessEqual(len(want), len(full))

    def test_width_32_with_prefix(self):
        """Model L's width-32 blocks (2 heads, MLP 128), tensors named "blk.<name>"."""
        for seed in (3, 4):
            t = block_tensors(32, 2, 128, np.random.default_rng(seed))
            res, got = self.run_block(t, "blk.", 32, 2, 128)
            want = pack_block_py(t, 32, 2, 128)
            self.assertEqual(res, "OK %d" % len(want))
            if got != want:
                self.fail(first_diff(got, want))

    def test_refusals(self):
        t = block_tensors(32, 2, 128, np.random.default_rng(6))
        t2 = dict(t)
        del t2["W2"]
        self.assertEqual(self.run_block(t2, "blk.", 32, 2, 128)[0], "ERR missing tensor blk.W2")
        t3 = dict(t)
        t3["b1"] = t3["b1"][:64]
        self.assertEqual(self.run_block(t3, "blk.", 32, 2, 128)[0],
                         "ERR tensor blk.b1 has 64 values, expected 128")
        self.assertTrue(self.run_block(t, "blk.", 40, 2, 128)[0].startswith("ERR block shape C 40"))


if __name__ == "__main__":
    unittest.main()
