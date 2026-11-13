# SPDX-License-Identifier: MIT
"""The optional model group filter / output_taps [5] (spec 08 draft 2 section 7.2; spec 06 draft 4): the runtime's
loaders (clean/runtime/model.cpp, raw layout and model.wgcm) and the synthetic model writer.

No GPU and no model data: clean/tests/test_model_filter.cpp is built with the host C++ compiler (CXX, clang++ or g++;
skipped without one) and loads synthetic models written by clean/runner/synthetic_model.py. The model.wgcm files come
from write_wgcm (the container of clean/converter/OUTPUT-FORMAT.md with a synthetic map), since the group is written
"like every other stage group"; the converter's own output is covered by clean/tests/test_model_wgcm.py.

Run: python -m unittest discover -s clean/tests -t .
"""
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from clean.reference import post as Po
from clean.runner import synthetic_model as SM
from clean.runner.pipeline import DEFAULT_MAP, read_model_tensors, read_wgcm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCES = [os.path.join("clean", "tests", "test_model_filter.cpp"), os.path.join("clean", "runtime", "model.cpp"),
           os.path.join("clean", "runtime", "embedded.cpp"), os.path.join("clean", "converter", "src", "sha256.cpp"),
           os.path.join("clean", "converter", "src", "json.cpp")]
_EXE = None


def loader_exe():
    global _EXE
    if _EXE is None:
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            raise unittest.SkipTest("no host C++ compiler (CXX, clang++, g++)")
        exe = os.path.join(tempfile.mkdtemp(prefix="wg_mf_"), "test_model_filter" + (".exe" if os.name == "nt" else ""))
        cmd = [cxx, "-std=c++17", "-O1"] + (["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else [])
        r = subprocess.run(cmd + [os.path.join(ROOT, s) for s in SOURCES] + ["-o", exe], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("build of test_model_filter failed:\n" + r.stderr)
        _EXE = exe
    return _EXE


def load(*args):
    r = subprocess.run([loader_exe(), *args], capture_output=True, text=True)
    return r.stdout.strip()


def bits(taps):
    return "OK " + " ".join("%08x" % v for v in np.asarray(taps, np.float32).view(np.uint32))


def rehash(data):
    return data[:-32] + hashlib.sha256(data[:-32]).digest()


class SyntheticWriter(unittest.TestCase):
    """No compiler needed: what synthetic_model.py writes and pipeline.py reads back."""

    def test_taps_kinds(self):
        analytic = SM.output_taps("analytic")
        self.assertEqual(analytic.dtype, np.float32)
        np.testing.assert_array_equal(analytic, Po.filter_weights(np.arange(1))[1][0].astype(np.float32))
        rnd = SM.output_taps("random", seed=5)
        self.assertAlmostEqual(float(rnd.astype(np.float64).sum()), 1.0, places=6)
        self.assertFalse(np.array_equal(rnd, analytic))
        np.testing.assert_array_equal(SM.output_taps("random", seed=5), rnd)          # seeded
        self.assertFalse(np.array_equal(SM.output_taps("random", seed=6), rnd))
        np.testing.assert_array_equal(SM.output_taps("0.1,0.2,0.4,0.2,0.1"),
                                      np.array([0.1, 0.2, 0.4, 0.2, 0.1], np.float32))
        for bad in ("lanczos", [1.0, 2.0], [0, 0, np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                SM.output_taps(bad)

    def test_group_is_optional_and_round_trips(self):
        d = tempfile.mkdtemp(prefix="wg_mf_")
        plain = SM.synthetic_tensors(3)
        self.assertNotIn("filter", plain)
        SM.write_raw(d, plain)
        self.assertFalse(os.path.exists(os.path.join(d, "raw", "filter")))
        self.assertNotIn("filter", SM.read_raw(d))
        d2 = tempfile.mkdtemp(prefix="wg_mf_")
        t = SM.synthetic_tensors(3, taps="random")
        self.assertEqual(SM.stage_order(t)[-2:], ["post", "filter"])
        SM.write_raw(d2, t)
        self.assertEqual(os.path.getsize(os.path.join(d2, "raw", "filter", "output_taps.bin")), 20)
        np.testing.assert_array_equal(SM.read_raw(d2)["filter"]["output_taps"], t["filter"]["output_taps"])
        for stage in SM.MAP_ORDER:                      # the other stages do not depend on the group
            for name, a in plain[stage].items():
                np.testing.assert_array_equal(a, t[stage][name])

    def test_wgcm_container(self):
        d = tempfile.mkdtemp(prefix="wg_mf_")
        t = SM.synthetic_tensors(4, taps="random")
        mp = SM.write_wgcm(d, t)
        self.assertTrue(os.path.exists(mp))
        back = read_wgcm(os.path.join(d, "model.wgcm"))
        self.assertEqual(list(back), SM.stage_order(t))
        for stage in back:
            for name, a in t[stage].items():
                np.testing.assert_array_equal(back[stage][name], a)
        np.testing.assert_array_equal(read_model_tensors(d)["filter"]["output_taps"], t["filter"]["output_taps"])
        with open(os.path.join(d, "model.wgcm"), "rb") as f:
            data = f.read()
        self.assertEqual(data[-32:], hashlib.sha256(data[:-32]).digest())


class LoaderRaw(unittest.TestCase):
    def test_absent(self):
        d = tempfile.mkdtemp(prefix="wg_mf_")
        SM.write_raw(d, SM.synthetic_tensors(1))
        self.assertEqual(load("raw", d), "OK none")
        self.assertEqual(load("raw", os.path.join(d, "raw")), "OK none")      # model_dir may be raw/ itself

    def test_present(self):
        for taps in ("analytic", "random", [0.5, -0.25, 1.0, -0.125, -0.125]):
            d = tempfile.mkdtemp(prefix="wg_mf_")
            t = SM.synthetic_tensors(1, taps=taps)
            SM.write_raw(d, t)
            self.assertEqual(load("raw", d), bits(t["filter"]["output_taps"]))
            self.assertEqual(load("raw", os.path.join(d, "raw")), bits(t["filter"]["output_taps"]))

    def test_damaged(self):
        d = tempfile.mkdtemp(prefix="wg_mf_")
        SM.write_raw(d, SM.synthetic_tensors(1, taps="analytic"))
        path = os.path.join(d, "raw", "filter", "output_taps.bin")
        np.zeros(4, "<f4").tofile(path)
        self.assertRegex(load("raw", d), r"^ERR model data damaged: .*output_taps\.bin has 16 bytes, expected 20")
        np.array([0.1, 0.2, np.inf, 0.2, 0.1], "<f4").tofile(path)
        self.assertRegex(load("raw", d), r"^ERR model data damaged: .*output_taps\.bin has a non-finite value")


class LoaderWgcm(unittest.TestCase):
    def case(self, tensors, mutate=None, **kw):
        d = tempfile.mkdtemp(prefix="wg_mf_")
        mp = SM.write_wgcm(d, tensors, **kw)
        path = os.path.join(d, "model.wgcm")
        if mutate:
            with open(path, "rb") as f:
                data = f.read()
            with open(path, "wb") as f:
                f.write(mutate(data))
        return load("wgcm", path, mp)

    def test_absent(self):
        # the map lists the group (a digest list), the file has none: loads, no taps
        self.assertEqual(self.case(SM.synthetic_tensors(2)), "OK none")
        self.assertEqual(self.case(SM.synthetic_tensors(2), other_filter_variants=2), "OK none")

    def test_present_any_variant(self):
        t = SM.synthetic_tensors(2, taps="random")
        want = bits(t["filter"]["output_taps"])
        for others, at in ((0, 0), (1, 0), (1, 1), (2, 1), (3, 3)):
            with self.subTest(others=others, at=at):
                self.assertEqual(self.case(t, other_filter_variants=others, filter_variant=at), want)

    def test_digest_must_be_in_the_maps_list(self):
        t = SM.synthetic_tensors(2, taps="random")
        d = tempfile.mkdtemp(prefix="wg_mf_")
        SM.write_wgcm(d, t)
        other = dict(t)
        other["filter"] = {"output_taps": SM.output_taps("analytic")}
        d2 = tempfile.mkdtemp(prefix="wg_mf_")
        mp2 = SM.write_wgcm(d2, other)
        # d's file with the map of d2 (which lists other weights): refused as "another map" by the map hash, so make
        # the file claim d2's map
        with open(os.path.join(d, "model.wgcm"), "rb") as f:
            data = f.read()
        with open(mp2, "rb") as f:
            claimed = hashlib.sha256(f.read()).digest()
        path = os.path.join(d, "model.claim.wgcm")
        with open(path, "wb") as f:
            f.write(rehash(data[:48] + claimed + data[80:]))
        self.assertRegex(load("wgcm", path, mp2), r"^ERR .*stage filter does not match the map's digest")
        self.assertRegex(load("wgcm", os.path.join(d, "model.wgcm"), mp2), r"^ERR .*different map")

    def test_wrong_shape_type_value(self):
        t = SM.synthetic_tensors(2)
        for name, taps, msg in (("shape", np.zeros(4, np.float32), "wrong shape or type"),
                                ("rank", np.zeros((1, 5), np.float32), "wrong shape or type"),
                                ("type", np.zeros(5, np.uint8), "wrong shape or type"),
                                ("nan", np.array([0, 0, np.nan, 0, 0], np.float32), "non-finite")):
            with self.subTest(case=name):
                bad = dict(t)
                bad["filter"] = {"output_taps": taps}
                self.assertRegex(self.case(bad), r"^ERR .*filter/output_taps.*" + msg)
        bad = dict(t)
        bad["filter"] = {"taps": SM.output_taps("analytic")}
        self.assertRegex(self.case(bad), r"^ERR .*tensor filter/output_taps missing")
        bad["filter"] = {"output_taps": SM.output_taps("analytic"), "extra": np.zeros(1, np.float32)}
        self.assertRegex(self.case(bad), r"^ERR .*filter/output_taps has the wrong shape or type")

    def test_damage_inside_the_group(self):
        t = SM.synthetic_tensors(2, taps="random")
        # the group's data is the last tensor: its 20 bytes end 32 bytes before the end of the file
        flip = lambda data: rehash(data[:-40] + bytes([data[-40] ^ 0x10]) + data[-39:])  # noqa: E731
        self.assertRegex(self.case(t, mutate=flip), r"^ERR .*stage filter does not match its stored digest")

    def test_file_of_an_earlier_map(self):
        # A file written with the map as it was before the group existed (thirteen stages, another map hash) keeps
        # loading with the current map: every stage equals this map's digest; no weights (spec 08 section 7.2: such a
        # file "counts as not present").
        plain = SM.synthetic_tensors(2)
        old = tempfile.mkdtemp(prefix="wg_mf_")
        old_map = SM.write_wgcm(old, plain, list_filter=False)
        new = tempfile.mkdtemp(prefix="wg_mf_")
        new_map = SM.write_wgcm(new, SM.synthetic_tensors(2, taps="random"))
        with open(old_map, "rb") as a, open(new_map, "rb") as b:
            self.assertNotEqual(a.read(), b.read())
        self.assertEqual(load("wgcm", os.path.join(old, "model.wgcm"), old_map), "OK none")
        self.assertEqual(load("wgcm", os.path.join(old, "model.wgcm"), new_map), "OK none other-map")
        # the other way round (a newer file, an older map that does not list the group): the group is skipped
        self.assertEqual(load("wgcm", os.path.join(new, "model.wgcm"), old_map), "OK none other-map")
        # other content under another map is still refused, whatever stage differs
        other = tempfile.mkdtemp(prefix="wg_mf_")
        SM.write_wgcm(other, SM.synthetic_tensors(3), list_filter=False)
        self.assertRegex(load("wgcm", os.path.join(other, "model.wgcm"), new_map), r"^ERR .*different map")
        self.assertRegex(load("wgcm", os.path.join(new, "model.wgcm"), DEFAULT_MAP), r"^ERR .*different map")

    def test_file_of_another_map_with_the_group(self):
        # the same weights listed at another position of another map's list: accepted, weights loaded
        t = SM.synthetic_tensors(2, taps="random")
        a = tempfile.mkdtemp(prefix="wg_mf_")
        SM.write_wgcm(a, t, other_filter_variants=1, filter_variant=1)
        b = tempfile.mkdtemp(prefix="wg_mf_")
        map_b = SM.write_wgcm(b, t, other_filter_variants=3, filter_variant=0)
        self.assertEqual(load("wgcm", os.path.join(a, "model.wgcm"), map_b),
                         bits(t["filter"]["output_taps"]) + " other-map")

    def test_shipped_map_is_readable(self):
        # the shipped map (version 4) gives the group a digest list; a synthetic file that claims it gets as far as
        # the digest comparison of its first stage (not "converter map damaged")
        t = SM.synthetic_tensors(2, taps="random")
        self.assertRegex(self.case(t, map_file=DEFAULT_MAP), r"^ERR .*stage enc0 does not match the map's digest")


if __name__ == "__main__":
    unittest.main()
