# SPDX-License-Identifier: MIT
"""The C++ host packing of the input stage (clean/rr_f/host/rr_input_host.h) against the Python reference
(reference/input_stage.frame_constants + gpu/input_stage.scalar_words): per-frame constants, camera maps (every
camera kind of the synthetic set), flags and symbols, word for word. Builds the test driver with MSVC
(/fp:strict); skipped when Visual Studio is not found."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from clean.rr_f.reference import input_stage as IS
from clean.rr_f.tests import synth_input as SI
from clean.rr_f.tests.test_input_stage import random_params

HOST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "host")


def build_cli(out):
    sys.path.insert(0, os.path.join(os.path.dirname(HOST), "..", "dx12"))
    import build_dx12
    exe = os.path.join(out, "rr_input_host_cli.exe")
    cmd = (f'"{build_dx12.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /fp:strict /EHsc /std:c++17 '
           f'"{os.path.join(HOST, "rr_input_host_cli.cpp")}" /Fe"{exe}" /Fo"{os.path.join(out, "cli.obj")}"')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise RuntimeError(r.stdout + r.stderr)
    return exe


def hexf(v):
    return f"{int(np.asarray(v, np.float32).view(np.uint32)):08x}"


class TestHostPacking(unittest.TestCase):
    def test_vs_python(self):
        try:
            import build_dx12  # noqa: F401
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(HOST), "..", "dx12"))
        tmp = tempfile.mkdtemp(prefix="rr_input_host_")
        try:
            exe = build_cli(tmp)
        except Exception as e:      # no compiler on this machine
            self.skipTest(f"MSVC build failed: {e}")
        rng = np.random.default_rng(5400)
        kinds = ["perspective", "reverse", "ortho", "identity", "zero", "random", "nonfinite", "static", None]
        lines, expect = [], []
        for i in range(400):
            p = random_params(rng)
            b2 = bool(rng.random() < 0.5)
            S = SI.symbols(b2=b2, flavour=int(rng.integers(0, 2)))
            kind = kinds[i % len(kinds)]
            if kind is not None:
                p.V2C, p.W2V, p.V2Cp, p.W2Vp = SI.camera_kind(rng, kind)
            p.E, p.Ep, p.X3 = (np.float32(rng.uniform(0.1, 9)), np.float32(rng.uniform(0.1, 9)),
                               np.float32(rng.uniform(-1, 1)))
            p.hw_depth, p.depth_inverted = bool(rng.random() < 0.6), bool(rng.random() < 0.5)
            p.nrm_dec, p.rgh_packed, p.specular_from_v = (bool(rng.random() < 0.5) for _ in range(3))
            mask = int(rng.integers(0, 1 << 13)) | 0b1110111
            attached = {r for j, r in enumerate(IS.ROLES) if mask >> j & 1}
            k = IS.frame_constants(p)
            from clean.rr_f.gpu.input_stage import scalar_words
            expect.append((int(k.camera_usable), scalar_words(k, S, attached)))
            have = p.V2C is not None
            mats = [p.V2C, p.W2V, p.V2Cp, p.W2Vp] if have else [np.zeros((4, 4), np.float32)] * 4
            f = [p.render_size[0], p.render_size[1], p.output_size[0], p.output_size[1]]
            f += [hexf(v) for v in (*p.jitter, *p.prev_jitter, *p.mv_scale_raw)]
            f += [int(p.mv_jittered), hexf(p.E), hexf(p.Ep), hexf(p.X3), int(p.reset), p.par, p.fc,
                  int(p.hw_depth), int(p.depth_inverted), int(p.nrm_dec), int(p.rgh_packed),
                  int(p.specular_from_v), int(b2), int(have), mask]
            f += [hexf(v) for m in mats for v in np.asarray(m, np.float32).reshape(-1)]
            f += [int(v) for v in np.asarray(p.SH).reshape(-1)]
            from clean.rr_f.gpu.input_stage import SYMS
            f += [hexf(S.values.get(n, 0.0)) for n in SYMS]
            lines.append(" ".join(str(v) for v in f))
        path = os.path.join(tmp, "frames.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        out = subprocess.run([exe, path], capture_output=True, text=True, check=True).stdout.split("\n")
        n_usable = 0
        for i, (usable, words) in enumerate(expect):
            got = out[i].split()
            self.assertEqual(int(got[0]), usable, i)
            n_usable += usable
            gw = np.array([int(v, 16) for v in got[1:]], np.uint32)
            bad = np.flatnonzero(gw != words)
            self.assertEqual(bad.size, 0, (i, bad, gw[bad], words[bad]))
        self.assertGreater(n_usable, 100)
        # the float-float tables: c exactly, Th + Tl and Eh + El within 2^-48 of the Python (decimal) values
        from clean.rr_f.gpu import ffmath
        rows = subprocess.run([exe, "--tables"], capture_output=True, text=True, check=True).stdout.splitlines()
        val = lambda h: np.uint32(int(h, 16)).view(np.float32)
        for i in range(256):
            c, th, tl = (val(v) for v in rows[i].split())
            self.assertEqual(c, ffmath.LOGT[i, 0], i)
            ref = float(ffmath.LOGT[i, 1]) + float(ffmath.LOGT[i, 2])
            self.assertLessEqual(abs(float(th) + float(tl) - ref), 2.0 ** -48 * max(abs(ref), 2.0 ** -8), i)
        for j in range(65):
            eh, el = (val(v) for v in rows[256 + j].split())
            ref = float(ffmath.EXPT[j, 0]) + float(ffmath.EXPT[j, 1])
            self.assertLessEqual(abs(float(eh) + float(el) - ref), 2.0 ** -48 * ref, j)


if __name__ == "__main__":
    unittest.main()
