# SPDX-License-Identifier: MIT
"""The runtime's C++ model K weight packer (clean/k/runtime/k_pack.cpp k_pack_stage) against clean/k/runner/k_pack.py
pack_stage (d_row "gfx12"), byte for byte, all eleven stages:

- random parameters (clean/k/reference/network.py random_network_params) in the converter's --raw layout
  (clean/k/runner/k_model.py write_raw: spec 48 shapes);
- the same in the canonical shapes of clean/k/reference/params.py, with g1 named gamma1, an extra tensor and
  infinities;
- tensors the packer must refuse (missing, wrong shape, not FP16, NaN);
- a converted synthetic preset-K DLL (test_k_runtime_tables.synthetic_converted) loaded by the C++ loader;
- the FP16 rounding it uses (clean/runtime/fpconv.h f32_to_f16) against numpy's float32 -> float16.

Goes through clean/k/tests/k_runtime_tables_dump.cpp, built with the compiler discovery of test_k_runtime_tables.py.

Run: python -m unittest clean.k.tests.test_k_runtime_pack
"""
import os
import shutil
import unittest

import numpy as np

from clean.k.reference.layout import STAGE, STAGE_NAMES
from clean.k.reference.network import random_network_params
from clean.k.reference.params import normalize
from clean.k.runner import k_pack
from clean.k.runner.k_model import load_raw, write_raw
from clean.k.tests.test_k_runtime_tables import build_dir, run_dump, synthetic_converted


def write_dir(root, params, shapes=True):
    """{stage: {name: array}} -> root/<stage>/<name>.bin (float32) and, with shapes, <name>.shape."""
    if os.path.exists(root):
        shutil.rmtree(root)
    for stage, tensors in params.items():
        d = os.path.join(root, stage)
        os.makedirs(d)
        for name, a in tensors.items():
            a = np.asarray(a, dtype="<f4")
            a.tofile(os.path.join(d, name + ".bin"))
            if shapes:
                with open(os.path.join(d, name + ".shape"), "w") as f:
                    f.write(" ".join(str(n) for n in a.shape) + "\n")


def run_pack(*args):
    """Runs the dump's pack / packwgcm; returns ({stage: "ok" | error message}, out dir)."""
    out = args[-1]
    if os.path.exists(out):
        shutil.rmtree(out)
    r = run_dump(*args)
    if r.returncode != 0:
        raise AssertionError("dump %s failed (%d): %s%s" % (args[0], r.returncode, r.stdout, r.stderr))
    res = {}
    for line in r.stdout.splitlines():
        stage, rest = line.split(" ", 1)
        res[stage] = "ok" if rest.startswith("ok ") else rest
    return res, out


class TestKRuntimePack(unittest.TestCase):
    def compare(self, res, out, params, stages=STAGE_NAMES):
        for stage in stages:
            want = k_pack.pack_stage(stage, params[stage], d_row="gfx12")
            self.assertEqual(len(want), k_pack.stage_sections(STAGE[stage])["_size"])
            self.assertEqual(res.get(stage), "ok", stage)
            with open(os.path.join(out, stage + ".bin"), "rb") as f:
                got = f.read()
            self.assertEqual(len(got), len(want), stage)
            if got != want:
                diff = np.flatnonzero(np.frombuffer(got, np.uint8) != np.frombuffer(want, np.uint8))
                self.fail("%s: %d bytes differ, first at %d" % (stage, len(diff), diff[0]))

    def test_random_converter_layout(self):
        """random_network_params written with write_raw (spec 48 shapes), three seeds."""
        for seed in (0, 1, 2):
            params = random_network_params(seed)
            raw = os.path.join(build_dir(), "pack_raw_%d" % seed)
            if os.path.exists(raw):
                shutil.rmtree(raw)
            write_raw(raw, params)
            res, out = run_pack("pack", os.path.join(raw, "raw"), os.path.join(build_dir(), "pack_raw_out_%d" % seed))
            self.assertEqual(sorted(res), sorted(STAGE_NAMES))
            self.compare(res, out, params)

    def test_canonical_shapes(self):
        """Canonical shapes (params.py), gamma1 instead of g1, an ignored extra tensor, infinities and -0."""
        params = {s: normalize(s, p) for s, p in random_network_params(7).items() if s != "net"}
        params["e2"]["gamma1"] = params["e2"].pop("g1")
        params["m"]["extra"] = np.ones((3, 5), np.float32)
        params["d3"]["bo"][5] = np.inf
        params["d3"]["W1"][17, 3] = -np.inf
        params["e0"]["be"][0] = -0.0
        params["d0"]["Wh"][39, 31] = 65504.0
        params["e4"]["B"][1, 2, 3] = 2.0 ** -24  # smallest FP16 subnormal
        write_dir(os.path.join(build_dir(), "pack_canon"), params)
        res, out = run_pack("pack", os.path.join(build_dir(), "pack_canon"), os.path.join(build_dir(), "pack_canon_out"))
        self.compare(res, out, params)

    def test_refused(self):
        """Tensors k_pack.py refuses are refused by the C++ packer too."""
        base = random_network_params(3)
        cases = {}

        def case(stage, label, edit):
            p = {n: np.array(a, np.float32) for n, a in base[stage].items()}
            edit(p)
            cases[(stage, label)] = p

        case("e1", "missing", lambda p: p.pop("Wv"))
        case("e3", "shape", lambda p: p.__setitem__("Wo", p["Wo"].reshape(p["Wo"].shape[1], p["Wo"].shape[0])))
        case("d2", "bx shape", lambda p: p.__setitem__("bx", p["bx"].reshape(2, -1)))
        case("m", "not fp16", lambda p: p["W2"].__setitem__((3, 9), np.float32(1.0 + 2.0 ** -12)))
        case("d0", "nan", lambda p: p["M"].__setitem__((1, 2, 3), np.nan))
        case("d4", "subnormal not fp16", lambda p: p["b1"].__setitem__(4, np.float32(2.0 ** -25)))
        for (stage, label), p in cases.items():
            if label == "nan":
                continue  # k_pack.py keeps a NaN; the C++ packer refuses it (the loader never returns one)
            with self.assertRaises(ValueError, msg=label):
                k_pack.pack_stage(stage, p)
        for i, ((stage, label), p) in enumerate(cases.items()):
            root = os.path.join(build_dir(), "pack_bad_%d" % i)
            write_dir(root, {stage: p})
            res, _ = run_pack("pack", root, os.path.join(build_dir(), "pack_bad_out_%d" % i))
            self.assertIn(stage, res, label)
            self.assertTrue(res[stage].startswith("error: "), "%s %s: %s" % (stage, label, res[stage]))

    def test_synthetic_converted(self):
        """A converted synthetic preset-K DLL: load_k_model + k_pack_stage equals load_raw + pack_stage."""
        syn = synthetic_converted()
        res, out = run_pack("packwgcm", syn["wgcm"], syn["map"], os.path.join(build_dir(), "pack_wgcm_out"))
        params = load_raw(syn["out"], map_path=syn["map"], check=False)
        for stage in STAGE_NAMES:
            try:
                want = k_pack.pack_stage(stage, params[stage])
            except ValueError as e:  # values that are not FP16: both must refuse
                self.assertTrue(res[stage].startswith("error: "), "%s: Python refuses (%s), C++ %s" % (
                    stage, e, res[stage]))
                continue
            self.assertEqual(res[stage], "ok", stage)
            with open(os.path.join(out, stage + ".bin"), "rb") as f:
                self.assertEqual(f.read(), want, stage)
        self.assertTrue(any(v == "ok" for v in res.values()), res)

    def test_f16_rounding(self):
        """fpconv.h f32_to_f16 is numpy's float32 -> float16 (round to nearest even, subnormals, overflow)."""
        rng = np.random.default_rng(16)
        rand = rng.integers(0, 2 ** 32, 300_000, dtype=np.uint64).astype(np.uint32)
        near = (rng.integers(0x32800000, 0x48000000, 600_000, dtype=np.uint64).astype(np.uint32) |
                (rng.integers(0, 2, 600_000, dtype=np.uint64).astype(np.uint32) << np.uint32(31)))
        edges = np.array([0.0, -0.0, 65504.0, 65504.5, 65519.0, 65519.99, 65520.0, 1e9, np.inf, -np.inf,
                          2.0 ** -14, 2.0 ** -14 * (1 - 2 ** -20), 2.0 ** -24, 2.0 ** -24 * 0.999, 2.0 ** -25,
                          2.0 ** -25 * 1.001, 3 * 2.0 ** -25, 1e-45, 1.0, 1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11,
                          1.0 - 2 ** -24, -1.0 - 2 ** -11, 3.0e-5, -0.1], np.float32).view(np.uint32)
        bits = np.concatenate([edges, rand, near]).astype(np.uint32)
        src = os.path.join(build_dir(), "f16_in.bin")
        dst = os.path.join(build_dir(), "f16_out.bin")
        bits.tofile(src)
        r = run_dump("f16", src, dst)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = np.fromfile(dst, np.uint16)
        x = bits.view(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            ref = x.astype(np.float16).view(np.uint16)
        nan = np.isnan(x)
        self.assertTrue(np.all((got[nan] & 0x7c00) == 0x7c00) and np.all((got[nan] & 0x3ff) != 0), "NaN stays NaN")
        bad = np.flatnonzero((got != ref) & ~nan)
        self.assertEqual(len(bad), 0, "%d differ, first %08x -> %04x vs %04x" % (
            len(bad), *((int(bits[bad[0]]), int(got[bad[0]]), int(ref[bad[0]])) if len(bad) else (0, 0, 0))))


if __name__ == "__main__":
    unittest.main()
