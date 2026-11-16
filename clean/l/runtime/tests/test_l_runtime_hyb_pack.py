# SPDX-License-Identifier: MIT
"""Model L's HYBRID weight values in the runtime (clean/l/runtime/l_hyb_pack.h, through pack_stage_blob and
l::pack_model) against clean/l/runner/l_hyb_stage_pack.py + the Python packers, byte for byte:

- every blob (the ten shared stages, packing.pack_stage; Model L's own launches, l_pack.pack_model_l) for random
  parameters (clean/l/reference/oracle.random_params), three seeds, plus one seed with the weights and biases scaled so
  that the largest weight is 1000 (FP8 saturation at +-448, FP16 rounding of large biases);
- negative control: the C++ "mutant" (the gammas without the sqrtf(C) fold) must differ in every gamma-bearing blob;
- with WG_TEST_L_WGCM=<model.wgcm> and WG_TEST_L_MAP=<map>: a local model file (never in the repository) loaded by the
  runtime's loader, its tensors exported by the tool and the Python reference built from them; only counts and a hash
  are printed, nothing of the model.

No GPU: clean/l/runtime/tests/l_hyb_pack_dump.cpp is built with the runtime's sources by the host C++ compiler (as
test_l_runtime_pack.py; skipped without one).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_hyb_pack.py
"""
import hashlib
import os
import tempfile
import unittest

import numpy as np

import test_l_runtime_pack as TP
from clean.l.reference import oracle as L
from clean.l.runner import l_hyb_stage_pack as HS
from clean.l.runner import l_pack as LP
from clean.runner import packing as P

ROOT = TP.ROOT
SOURCES = [os.path.join("clean", "l", "runtime", "tests", "l_hyb_pack_dump.cpp"),
           os.path.join("clean", "l", "runtime", "l_pack.cpp"), os.path.join("clean", "l", "runtime", "l_model.cpp"),
           os.path.join("clean", "runtime", "model.cpp"), os.path.join("clean", "runtime", "embedded.cpp"),
           os.path.join("clean", "converter", "src", "sha256.cpp"),
           os.path.join("clean", "converter", "src", "json.cpp")]
SHARED = ("enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")
GAMMA_BLOBS = SHARED + ("enc0_block", "dec0_block")
_EXE = None


def dump_exe():
    global _EXE
    if _EXE is None:
        saved, saved_exe = TP.SOURCES, TP._EXE
        try:
            TP.SOURCES, TP._EXE = SOURCES, None
            _EXE = TP.dump_exe()
        finally:
            TP.SOURCES, TP._EXE = saved, saved_exe
    return _EXE


def run(*args):
    import subprocess
    return subprocess.run([dump_exe(), *args], capture_output=True, text=True).stdout.strip()


def write_params(root, params):
    for stage, tensors in params.items():
        d = os.path.join(root, stage)
        os.makedirs(d)
        for name, a in tensors.items():
            if stage == "enc0" and name == "T":
                np.asarray(a).astype(np.uint8).tofile(os.path.join(d, "T.u8"))
            else:
                np.ascontiguousarray(a, "<f4").reshape(-1).tofile(os.path.join(d, name + ".f32"))


def read_dir(d):
    out = {}
    for f in os.listdir(d):
        with open(os.path.join(d, f), "rb") as fh:
            out[os.path.splitext(f)[0]] = fh.read()
    return out


def read_params(root):
    params = {}
    for stage in os.listdir(root):
        st = {}
        for f in os.listdir(os.path.join(root, stage)):
            path = os.path.join(root, stage, f)
            # enc0's dither table T: uint8 on disk, float32 integer values for the Python packers (as random_params)
            st[os.path.splitext(f)[0]] = (np.fromfile(path, np.uint8).astype(np.float32) if f.endswith(".u8")
                          else np.fromfile(path, "<f4"))
        params[stage] = st
    return params


def reference(params):
    """The Python packers on the hybrid values: {blob: bytes} as the runtime names them."""
    hp = HS.hyb_params(params)
    want = {s: P.pack_stage(s, hp[s], "gfx12", model="M") for s in SHARED}
    want.update(LP.pack_model_l(hp))
    return want


def shaped(params, like):
    """Flat tensors read back from disk, reshaped as the oracle's random parameters (for the Python packers)."""
    return {s: {n: np.asarray(v).reshape(np.shape(like[s][n])) if s in like and n in like[s] else v
                for n, v in t.items()} for s, t in params.items()}


def pack_cpp(params, mutant=False):
    d = tempfile.mkdtemp(prefix="wg_hyb_")
    write_params(d, params)
    out = os.path.join(d, "out")
    res = run("hyb", d, out, *(["mutant"] if mutant else []))
    return res, (read_dir(out) if res.startswith("OK") else {})


def scaled(params, k):
    out = {}
    for s, t in params.items():
        out[s] = dict(t)
        if s in HS.BLOCK_STAGES:
            for n in ("Wqk", "Wv", "Wp", "W1", "W2", "Wm", "We", "bp", "b1", "b2", "bm", "be"):
                if n in t:
                    out[s][n] = (np.asarray(t[n], np.float32) * np.float32(k)).astype(np.float32)
    return out


class HybBlobs(unittest.TestCase):
    def check(self, params, tag):
        want = reference(params)
        res, got = pack_cpp(params)
        self.assertEqual(res, "OK %d" % len(want), tag)
        self.assertEqual(sorted(got), sorted(want), tag)
        for name, w in want.items():
            if got[name] != w:
                self.fail("%s, %s: %s" % (tag, name, TP.first_diff(got[name], w)))
        return want

    def test_random_params(self):
        for seed in (0, 1, 2):
            self.check(L.random_params(np.random.default_rng(seed)), "seed %d" % seed)

    def test_saturation(self):
        p0 = L.random_params(np.random.default_rng(5))
        wmax = max(float(np.abs(np.asarray(p0[s][n])).max()) for s in HS.BLOCK_STAGES for n in ("Wqk", "W1", "W2"))
        p = scaled(p0, 1000.0 / wmax)                    # the largest weight at 1000: beyond FP8's 448
        w = np.concatenate([np.asarray(p[s][n]).reshape(-1) for s in HS.BLOCK_STAGES for n in ("Wqk", "W1", "W2")])
        self.assertTrue((np.abs(w) > 448).any(), "the scaled seed must reach FP8 saturation")
        self.check(p, "scaled seed 5")

    def test_mutant_differs(self):
        params = L.random_params(np.random.default_rng(0))
        want = reference(params)
        res, got = pack_cpp(params, mutant=True)
        self.assertTrue(res.startswith("OK"), res)
        for name in GAMMA_BLOBS:
            self.assertNotEqual(got[name], want[name], "mutant not caught in " + name)

    def test_hybrid_values_differ_from_today(self):
        # the hybrid blobs are not the M-form blobs (values changed in every block stage)
        params = L.random_params(np.random.default_rng(1))
        want = reference(params)
        today = {s: P.pack_stage(s, params[s], "gfx12", model="M") for s in SHARED}
        today.update(LP.pack_model_l(params))
        for name in GAMMA_BLOBS:
            self.assertNotEqual(today[name], want[name], name)

    @unittest.skipUnless(os.environ.get("WG_TEST_L_WGCM") and os.environ.get("WG_TEST_L_MAP"),
                         "WG_TEST_L_WGCM / WG_TEST_L_MAP not set (a local model file)")
    def test_local_model_file(self):
        d = tempfile.mkdtemp(prefix="wg_hyb_local_")
        res = run("wgcm", os.environ["WG_TEST_L_WGCM"], os.environ["WG_TEST_L_MAP"], d)
        self.assertTrue(res.startswith("OK"), res)
        like = L.random_params(np.random.default_rng(0))
        params = shaped(read_params(os.path.join(d, "tensors")), like)
        want = reference(params)
        got = read_dir(os.path.join(d, "blobs"))
        self.assertEqual(sorted(got), sorted(want))
        bad = [n for n in want if got[n] != want[n]]
        digest = hashlib.sha256(b"".join(want[n] for n in sorted(want))).hexdigest()
        print("local model file: %d hybrid blobs, %d differ, digest %s" % (len(want), len(bad), digest))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
