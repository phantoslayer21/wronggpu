# SPDX-License-Identifier: MIT
"""Model M equivalence tools (clean/equiv/model_m.py): storage formats, chained dumps (full and tiles), stage-input
files, isolated stage runs, replay with delete-after, input reader. CPU tests use stand-ins for spec 90's Pipeline and
Runner with synthetic values; the GPU smoke test (real Pipeline / Runner, synthetic model and inputs) runs only when
WG_EQUIV_GPU_BUILD (a clean/runner/build.py output) and WG_EQUIV_GPU_MODEL (a synthetic model folder) are set.

Run: python -m unittest clean.tests.test_equiv_model_m
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

from clean.equiv import compare as K
from clean.equiv import model_m as M
from clean.reference.fp8 import fp8_decode

RW, RH, OW, OH = 40, 24, 48, 27


def fp8_vals(r, shape):
    c = r.integers(0, 0x7F, size=shape).astype(np.uint8) | (r.integers(0, 2, size=shape).astype(np.uint8) << 7)
    return fp8_decode(c)


def grid(r, shape, sc, lo=0):
    return (r.integers(lo, sc + 1, size=shape) / sc).astype(np.float32)


class FakePipe:
    """Spec 90 Pipeline stand-in: a debug tree of the right shapes, values exactly on each storage grid."""

    def __init__(self, display=True):
        self.ow, self.oh = OW, OH
        self.dw, self.dh = (3 * OW + 1) // 2, (3 * OH + 1) // 2
        self.display = display
        self.count = 0
        self.trees = []

    def frame(self, x, debug=False):
        r = np.random.default_rng(100 + self.count)
        fw, fh = -(-RW // 32) * 32, -(-RH // 32) * 32
        F = (fh // 2, fw // 2)
        t = {"exposure": {"e": float(np.float16(0.75 + 0.125 * self.count)), "e_prev": 0.5},
             "enc0": {"features": fp8_vals(r, F + (64,))}}
        for k in ("m_near", "m_far", "m_center"):
            t["enc0"][k] = r.standard_normal((RH, RW, 2)).astype(np.float16).astype(np.float32)
        t["enc1"] = {"skip": fp8_vals(r, F + (64,)), "merged": fp8_vals(r, (F[0] // 2, F[1] // 2, 96))}
        t["enc2"] = {"skip": fp8_vals(r, (F[0] // 2, F[1] // 2, 96)), "merged": fp8_vals(r, (F[0] // 4, F[1] // 4, 128))}
        for i in range(6):
            t[f"b{i}"] = {"out": fp8_vals(r, (F[0] // 4, F[1] // 4, 128))}
        t["dec2"] = {"out": fp8_vals(r, (F[0] // 2, F[1] // 2, 96))}
        t["dec1"] = {"out": fp8_vals(r, F + (64,))}
        t["dec0"] = {"A": grid(r, (RH, RW, 3), 255), "B": grid(r, (RH, RW, 3), 255),
                     "C": r.random((RH, RW, 3)).astype(np.float16).astype(np.float32),
                     "latent": grid(r, (RH, RW, 4), 127, -127)}
        t["post"] = {"history": r.random((self.dh, self.dw, 3)).astype(np.float16).astype(np.float32),
                     "reduced": grid(r, (self.dh // 2, self.dw // 2, 3), 1023),
                     "display": r.random((self.dh, self.dw, 3)).astype(np.float16).astype(np.float32)
                     if self.display else None}
        t["filter"] = {"out": r.random((OH, OW, 3)).astype(np.float32)}
        t["state"] = {"motion_history": r.standard_normal((RH, RW, 2)).astype(np.float16).astype(np.float32)}
        self.parity = self.count % 2
        self.count += 1
        self.trees.append(t)
        return t if debug else {"out": t["filter"]["out"]}

    def info(self):
        from clean.reference.swin import window_offset
        return {"index": self.count - 1, "n": (self.count - 1) % 8, "parity": self.parity, "reset": self.count == 1,
                "render_size": (RW, RH), "jitter_prev": (0.25, -0.25),
                "offsets": {s: window_offset(s, self.parity) for s in M.SWIN}}

    def kernels(self):
        return {"post": "fake"}


def frames(n):
    return [{"jitter": np.array([0.125 * i, -0.125]), "color": None} for i in range(n)]


class FakeRunner:
    """Spec 90 Runner stand-in: checks the decoded inputs against the chained tree and returns the chained outputs."""

    def __init__(self, tree, test):
        self.tree, self.test, self.calls = tree, test, []

    def run(self, stage, weights, x, offset=None, skip=None):
        self.calls.append((stage, offset))
        if stage in M.SWIN_INPUTS:
            src = M.SWIN_INPUTS[stage]["x"].split(".")
            self.test.assertTrue(np.array_equal(x, self.tree[src[0]][src[1]]))
            if "skip" in M.SWIN_INPUTS[stage]:
                s = M.SWIN_INPUTS[stage]["skip"].split(".")
                self.test.assertTrue(np.array_equal(skip, self.tree[s[0]][s[1]]))
            return dict(self.tree[stage])
        if stage == "filter":
            self.test.assertTrue(np.array_equal(x["display"], self.tree["post"]["display"]))
            self.test.assertEqual(x["output_size"], (OW, OH))
            return {"out": self.tree["filter"]["out"]}
        if stage == "exposure":
            return {"e": 0.5}
        return {k: np.zeros((2, 2, 3), np.float32) for k in M.STAGE_OUTPUTS[stage]}


class FormatTest(unittest.TestCase):
    def test_round_trip(self):
        r = np.random.default_rng(1)
        vals = {"fp8": fp8_vals(r, (5, 7, 3)), "fp16": r.random((5, 7)).astype(np.float16).astype(np.float32),
                "unorm8": grid(r, (9,), 255), "snorm8": grid(r, (9,), 127, -127), "unorm10": grid(r, (9,), 1023),
                "fp32": r.random(4).astype(np.float32)}
        dt = {"fp8": np.uint8, "fp16": np.float16, "unorm8": np.uint8, "snorm8": np.int8, "unorm10": np.uint16,
              "fp32": np.float32}
        for fmt, v in vals.items():
            c = M.encode(v, fmt)
            self.assertEqual(c.dtype, np.dtype(dt[fmt]), fmt)
            self.assertTrue(np.array_equal(M.decode(c, fmt), v), fmt)
        self.assertEqual(int(M.encode(np.array([-1.0]), "snorm8")[0]), -127)

    def test_off_grid_raises(self):
        for v, fmt in ((np.array([0.3]), "fp8"), (np.array([1 / 3], np.float32), "fp16"),
                       (np.array([0.3]), "unorm8"), (np.array([2.0]), "unorm8"),
                       (np.array([np.nan]), "unorm10")):
            with self.assertRaises(M.EncodeError):
                M.encode(v, fmt)


class DumpTest(unittest.TestCase):
    def test_full_dump(self):
        p = FakePipe()
        with tempfile.TemporaryDirectory() as td:
            man = M.dump(p, frames(3), td, full=True, max_frames=2, state=True, backend="fake")
            self.assertEqual(len(man["frames"]), 2)
            self.assertEqual(man["missing"], [])
            self.assertEqual(len(man["frames"][0]["files"]), len(M.BOUNDARIES) + 1)
            for i in range(2):
                for name, path, page, fmt, gridk in M.BOUNDARIES:
                    stem = os.path.join(td, f"f{i:04d}", name)
                    codes = np.load(stem + ".npy")
                    with open(stem + ".json") as f:
                        d = json.load(f)
                    self.assertEqual((d["boundary"], d["format"], d["spec_page"], d["frame"]), (name, fmt, page, i))
                    self.assertEqual((d["dtype"], d["shape"]), (str(codes.dtype), list(codes.shape)))
                    self.assertEqual((d["parity"], d["n"], d["render_size"], d["output_size"]), (i % 2, i, [RW, RH],
                                                                                               [OW, OH]))
                    self.assertEqual(d["feature_size"], [64, 32])
                    self.assertEqual(d["recon_size"], [p.dw, p.dh])
                    want = p.trees[i][path[0]][path[1]]
                    self.assertTrue(np.array_equal(M.decode(codes, fmt).reshape(np.shape(want)),
                                                   np.asarray(want, np.float32)), name)
                    if name == "b1.out":
                        self.assertEqual(d["window_offsets"]["b1"], [0, 0] if i == 0 else [2, 0])
            self.assertEqual(np.load(os.path.join(td, "f0000", "enc0.features.npy")).dtype, np.uint8)
            self.assertEqual(np.load(os.path.join(td, "f0000", "dec0.latent.npy")).dtype, np.int8)
            self.assertEqual(np.load(os.path.join(td, "f0000", "post.reduced.npy")).dtype, np.uint16)
            # the comparison manifest runs as is: chained intermediates numbers only, the final output judged
            with open(os.path.join(td, "compare-manifest.json")) as f:
                cm = json.load(f)
            rep = K.run_manifest(cm, td, td)
            v = {r["name"]: r["verdict"] for r in rep["rows"] if r["kind"] == "stage"}
            self.assertEqual(v["enc1.skip f0"], K.NUMBERS_ONLY)
            self.assertEqual(v["filter.out f1"], K.MATCHES)
            self.assertEqual(rep["model_verdict"], K.MATCHES)

    def test_missing_display_is_recorded(self):
        with tempfile.TemporaryDirectory() as td:
            man = M.dump(FakePipe(display=False), frames(1), td, full=True)
            self.assertEqual([m["boundary"] for m in man["missing"]], ["post.display"])
            self.assertFalse(os.path.exists(os.path.join(td, "f0000", "post.display.npy")))

    def test_needs_full_or_tiles(self):
        with self.assertRaises(ValueError):
            M.dump(FakePipe(), frames(1), "unused")

    def test_map_rect(self):
        sizes = {"render_size": [RW, RH], "feature_size": [64, 32], "recon_size": [72, 41], "output_size": [OW, OH]}
        self.assertEqual(M.map_rect((5, 3, 6, 4), "render", (RH, RW), sizes), (5, 3, 11, 7))
        self.assertEqual(M.map_rect((5, 3, 6, 4), "feature", (16, 32), sizes), (2, 1, 6, 4))      # stride 2
        self.assertEqual(M.map_rect((5, 3, 6, 4), "feature", (4, 8), sizes), (0, 0, 2, 1))        # stride 8
        self.assertEqual(M.map_rect((5, 3, 6, 4), "recon", (41, 72), sizes), (9, 5, 20, 12))      # x 1.8, 1.708
        self.assertEqual(M.map_rect((36, 20, 10, 10), "output", (OH, OW), sizes), (43, 22, OW, OH))   # clamped

    def test_tile_dump(self):
        p = FakePipe()
        with tempfile.TemporaryDirectory() as td:
            tf = os.path.join(td, "tiles.json")
            with open(tf, "w") as f:
                json.dump({"units": "render pixels", "tiles": [[0, 0, 8, 8], [16, 8, 16, 8]]}, f)
            tiles = M.load_tiles(tf)
            out = os.path.join(td, "d")
            man = M.dump(p, frames(1), out, tiles=tiles)
            fdir = os.path.join(out, "f0000")
            self.assertFalse(os.path.exists(os.path.join(fdir, "enc1.skip.npy")))
            self.assertTrue(os.path.exists(os.path.join(fdir, "exposure.e.npy")))          # scalar: whole
            sizes = {"render_size": [RW, RH], "feature_size": [64, 32], "recon_size": [p.dw, p.dh],
                     "output_size": [OW, OH]}
            for name, path, _, fmt, gridk in M.BOUNDARIES:
                if gridk == "scalar":
                    continue
                full = M.encode(p.trees[0][path[0]][path[1]], fmt)
                for k, rect in enumerate(tiles):
                    x0, y0, x1, y1 = M.map_rect(rect, gridk, full.shape[:2], sizes)
                    got = np.load(os.path.join(fdir, f"{name}.t{k:02d}.npy"))
                    self.assertTrue(np.array_equal(got, full[y0:y1, x0:x1]), (name, k))
                    with open(os.path.join(fdir, f"{name}.t{k:02d}.json")) as f:
                        d = json.load(f)
                    self.assertEqual(d["tile"], {"index": k, "render_rect": list(rect),
                                                 "rect": [x0, y0, x1 - x0, y1 - y0]})
            self.assertEqual(len(man["frames"][0]["files"]), 1 + 2 * (len(M.BOUNDARIES) - 1))
            for bad in ({"tiles": []}, {"tiles": [[0, 0, 0, 4]]}, {"units": "output pixels", "tiles": [[0, 0, 1, 1]]}):
                with open(tf, "w") as f:
                    json.dump(bad, f)
                with self.assertRaises(ValueError):
                    M.load_tiles(tf)


class StageTest(unittest.TestCase):
    def test_isolated_equals_chained_for_dump_inputs(self):
        p = FakePipe()
        with tempfile.TemporaryDirectory() as td:
            dd = os.path.join(td, "dump")
            M.dump(p, frames(2), dd, full=True)
            tree = p.trees[1]
            fr = FakeRunner(tree, self)
            for stage in M.SWIN + ["filter"]:
                spec = M.stage_inputs_from_dump(dd, 1, stage)
                sp = os.path.join(dd, f"in_{stage}.json")
                with open(sp, "w") as f:
                    json.dump(spec, f)
                out = os.path.join(td, "iso", stage)
                stems = M.run_stage(fr, sp, out, weights=b"w")
                self.assertEqual(stems, [f"{stage}.{k}" for k in M.STAGE_OUTPUTS[stage]])
                for st in stems:
                    key = st.split(".", 1)[1]
                    chained = f"{'filter' if stage == 'filter' else stage}.{key}"
                    a = np.load(os.path.join(out, st + ".npy"))
                    b = np.load(os.path.join(dd, "f0001", chained + ".npy"))
                    self.assertEqual(K.compare_stage(a, b, M.STAGE_OUTPUTS[stage][key], "HWC")["verdict"], K.MATCHES)
                    with open(os.path.join(out, st + ".json")) as f:
                        d = json.load(f)
                    self.assertEqual((d["frame"], d["parity"]), (1, 1))
            offs = dict(fr.calls)
            self.assertEqual(offs["b1"], (2, 0))       # parity 1, from the dump's window offsets
            self.assertEqual(offs["enc1"], (0, 2))
            with self.assertRaises(ValueError):
                M.stage_inputs_from_dump(dd, 0, "dec0")

    def test_dict_stage_with_inline_formats_and_scalars(self):
        r = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as td:
            np.save(os.path.join(td, "color.npy"), r.random((4, 6, 3)).astype(np.float32))
            spec = {"stage": "exposure", "inputs": {"color": {"file": "color.npy", "format": "fp32"}},
                    "scalars": {"pre_exposure": 2.0, "alpha": 0.18}, "meta": {"frame": 7}}
            sp = os.path.join(td, "s.json")
            with open(sp, "w") as f:
                json.dump(spec, f)
            x, off, skip = M.stage_x(spec, td)
            self.assertEqual((x["pre_exposure"], off, skip), (2.0, None, None))
            stems = M.run_stage(FakeRunner({}, self), sp, os.path.join(td, "o"))
            self.assertEqual(stems, ["exposure.e"])
            e = np.load(os.path.join(td, "o", "exposure.e.npy"))
            self.assertEqual((e.dtype, e.shape, float(e[0])), (np.dtype(np.float16), (1,), 0.5))
            spec = {"stage": "dec0", "inputs": {"features": {"file": "color.npy", "format": "fp32"}},
                    "scalars": {"jitter": [0.25, 0.0], "reset": True, "feature_size": [64, 32]}}
            x, _, _ = M.stage_x(spec, td)
            self.assertEqual((x["jitter"], x["reset"], x["feature_size"]), ((0.25, 0.0), True, (64, 32)))
            with open(sp, "w") as f:
                json.dump({"stage": "nope"}, f)
            with self.assertRaises(ValueError):
                M.run_stage(FakeRunner({}, self), sp, os.path.join(td, "o"))


class ReplayTest(unittest.TestCase):
    def test_replay_keep_and_delete_after(self):
        with tempfile.TemporaryDirectory() as td:
            seen = []

            def cb(path, i):
                self.assertTrue(os.path.exists(path))
                a = np.load(path)
                self.assertEqual((a.dtype, a.shape), (np.dtype(np.float16), (OH, OW, 4)))
                self.assertTrue(np.all(a[..., 3] == 1))
                seen.append(i)
                return "ok"

            p = FakePipe()
            s = M.replay(p, frames(3), os.path.join(td, "k"), then=cb)
            self.assertEqual((s["frames"], seen, s["results"]), (3, [0, 1, 2], ["ok"] * 3))
            one = os.path.getsize(os.path.join(td, "k", "f0000.rgba16f.npy"))
            self.assertEqual(s["peak_bytes"], 3 * one)
            a = np.load(os.path.join(td, "k", "f0002.rgba16f.npy"))
            self.assertTrue(np.array_equal(a[..., :3], p.trees[2]["filter"]["out"].astype(np.float16)))
            seen.clear()
            s = M.replay(FakePipe(), frames(3), os.path.join(td, "d"), then=cb, delete_after=True)
            self.assertEqual((s["frames"], s["peak_bytes"], seen), (3, one, [0, 1, 2]))
            self.assertEqual(os.listdir(os.path.join(td, "d")), [])

    def test_command_callback(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.npy")
            np.save(path, np.zeros(2))
            ok = M.command_callback(f'"{sys.executable}" -c "import os,sys; sys.exit(0 if os.path.exists(r\'{{path}}\') '
                                    f'and {{frame}} == 4 else 3)"')
            self.assertEqual(ok(path, 4), 0)
            with self.assertRaises(RuntimeError):
                ok(path, 5)


class ReaderAndCliTest(unittest.TestCase):
    def test_npz_reader(self):
        with tempfile.TemporaryDirectory() as td:
            for i in (1, 0):
                np.savez(os.path.join(td, f"frame{i:03d}.npz"), color=np.full((2, 3, 3), i, np.float32),
                         depth=np.zeros((2, 3), np.float32), motion=np.zeros((2, 3, 2), np.float32),
                         mv_scale=np.ones(2), jitter=np.zeros(2), reset=np.array(i == 0), exposure=np.array(1.5))
            got = list(M.frame_reader("npz", td))
            self.assertEqual([float(g["color"][0, 0, 0]) for g in got], [0.0, 1.0])
            self.assertEqual((got[0]["reset"], got[1]["reset"], got[0]["exposure"]), (True, False, 1.5))
            with self.assertRaises(FileNotFoundError):
                list(M.frame_reader("npz", os.path.join(td, "none")))

    def test_inputs_command(self):
        with tempfile.TemporaryDirectory() as td:
            dd = os.path.join(td, "dump")
            M.dump(FakePipe(), frames(1), dd, full=True)
            out = os.path.join(td, "specs", "dec2.json")
            os.makedirs(os.path.dirname(out))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(M.main(["inputs", "--dump", dd, "--frame", "0", "--stage", "dec2", "--out", out]), 0)
            with open(out) as f:
                spec = json.load(f)
            self.assertEqual(spec["inputs"], {"x": "../dump/f0000/b5.out.npy", "skip": "../dump/f0000/enc2.skip.npy"})
            x, off, skip = M.stage_x(spec, os.path.dirname(out))
            self.assertEqual((x.shape, skip.shape, off), ((4, 8, 128), (8, 16, 96), (0, 0)))   # b5 -> dec2, parity 0


class ConventionsTest(unittest.TestCase):
    """Input conventions as flags (decided 2026-11-04): motion units / sign / bias, jitter units / sign, depth
    direction, auto-exposure, subrect origin; recorded in the manifests."""

    def frame(self, tw=50, th=30, **kw):
        x = {"color": np.arange(th * tw * 3, dtype=np.float32).reshape(th, tw, 3),
             "depth": np.arange(th * tw, dtype=np.float32).reshape(th, tw),
             "motion": np.zeros((th, tw, 2), np.float32), "jitter": np.array([0.25, -0.5])}
        x.update(kw)
        return x

    def test_defaults_keep_the_frame(self):
        c = M.Conventions()
        y = c.apply(self.frame(mv_scale=np.array([2.0, 3.0]), mv_bias=np.array([0.5, 0.0])), (RW, RH), (OW, OH))
        self.assertEqual(list(y["mv_scale"]), [2.0, 3.0])
        self.assertEqual(list(y["mv_bias"]), [0.5, 0.0])
        self.assertEqual(list(y["jitter"]), [0.25, -0.5])
        self.assertEqual(y["color"].shape, (30, 50, 3))
        self.assertEqual(list(c.apply(self.frame(), (RW, RH), (OW, OH))["mv_scale"]), [1.0, 1.0])
        self.assertEqual(c.flags(), {"depth_inverted": False, "auto_exposure": False})

    def test_units_and_signs(self):
        f = self.frame(mv_scale=np.array([9.0, 9.0]))
        y = M.Conventions(mv_units="uv", mv_sign=(1, -1)).apply(f, (64, 64), (100, 60))
        self.assertEqual(list(y["mv_scale"]), [50.0, -30.0])                    # render size 50x30
        y = M.Conventions(mv_units="output", mv_sign=(-1, -1)).apply(f, (64, 64), (100, 60))
        self.assertEqual(list(y["mv_scale"]), [-0.5, -0.5])
        y = M.Conventions(mv_units="render", mv_bias=(1.0, 2.0)).apply(f, (64, 64), (100, 60))
        self.assertEqual((list(y["mv_scale"]), list(y["mv_bias"])), ([1.0, 1.0], [1.0, 2.0]))
        y = M.Conventions(jitter_units="uv", jitter_sign=(-1, 1)).apply(f, (64, 64), (100, 60))
        self.assertEqual(list(y["jitter"]), [-12.5, -15.0])
        y = M.Conventions(jitter_units="output").apply(f, (64, 64), (100, 60))
        self.assertEqual(list(y["jitter"]), [0.125, -0.25])

    def test_subrect(self):
        f = self.frame()
        y = M.Conventions(subrect=(7, 4, 20, 10)).apply(f, (64, 64), (OW, OH))
        self.assertTrue(np.array_equal(y["color"], f["color"][4:14, 7:27]))
        self.assertTrue(np.array_equal(y["depth"], f["depth"][4:14, 7:27]))
        self.assertEqual(y["motion"].shape, (10, 20, 2))
        y = M.Conventions(subrect=(7, 4)).apply(f, (RW, RH), (OW, OH))                 # size from --render, clipped
        self.assertEqual(y["color"].shape, (RH, RW, 3))
        y = M.Conventions(subrect=(30, 20)).apply(f, (RW, RH), (OW, OH))
        self.assertEqual(y["color"].shape, (10, 20, 3))
        y = M.Conventions(mv_units="uv", subrect=(0, 0, 20, 10)).apply(f, (RW, RH), (OW, OH))
        self.assertEqual(list(y["mv_scale"]), [20.0, 10.0])                          # units use the cut size
        with self.assertRaises(ValueError):
            M.Conventions(subrect=(40, 0, 20, 10)).apply(f, (RW, RH), (OW, OH))

    def test_validation(self):
        for kw in ({"mv_units": "ndc"}, {"jitter_units": "frame"}, {"depth": "reversed"}, {"auto_exposure": "yes"},
                   {"mv_sign": (1, 2)}, {"jitter_sign": (0, 1)}, {"subrect": (1,)}, {"subrect": (0, 0, 0, 4)}):
            with self.assertRaises(ValueError, msg=str(kw)):
                M.Conventions(**kw)

    def test_cli_flags_reach_pipeline_and_manifest(self):
        made = {}

        def fake_pipeline(a, conv):
            made["flags"] = conv.flags()
            return FakePipe()

        with tempfile.TemporaryDirectory() as td:
            ind = os.path.join(td, "in")
            os.mkdir(ind)
            for i in range(2):
                np.savez(os.path.join(ind, f"{i:03d}.npz"), color=np.zeros((RH + 9, RW + 17, 3), np.float32),
                         depth=np.zeros((RH + 9, RW + 17), np.float32), motion=np.zeros((RH + 9, RW + 17, 2),
                                                                                       np.float32),
                         mv_scale=np.ones(2), jitter=np.array([0.5, 0.25]))
            orig = M._pipeline
            M._pipeline = fake_pipeline
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = M.main(["dump", "--build", "b", "--model", "m", "--render", f"{RW}x{RH}", "--output",
                                 f"{OW}x{OH}", "--inputs", ind, "--full", "--out", os.path.join(td, "d"),
                                 "--mv-units", "uv", "--mv-sign=-1,1", "--jitter-units", "uv", "--depth",
                                 "inverted", "--auto-exposure", "--subrect", "17,9"])
                    self.assertEqual(rc, 0)
                    rc = M.main(["replay", "--build", "b", "--model", "m", "--render", f"{RW}x{RH}", "--output",
                                 f"{OW}x{OH}", "--inputs", ind, "--out", os.path.join(td, "r"), "--delete-after"])
                    self.assertEqual(rc, 0)
            finally:
                M._pipeline = orig
            self.assertEqual(made["flags"], {"depth_inverted": False, "auto_exposure": False})   # the replay's
            with open(os.path.join(td, "d", "manifest.json")) as f:
                man = json.load(f)
            c = man["input_conventions"]
            self.assertEqual((c["mv_units"], c["mv_sign"], c["jitter_units"], c["depth"], c["auto_exposure"],
                              c["subrect"]), ("uv", [-1.0, 1.0], "uv", "inverted", "on", [17, 9]))
            self.assertEqual(man["frames"][0]["jitter"], [0.5 * RW, 0.25 * RH])          # converted to render px
            with self.assertRaises(ValueError):
                M.conventions_from_args(type("A", (), dict(mv_units="frame", mv_sign="1,0", mv_bias=None,
                                                           jitter_units="render", jitter_sign="1,1", depth="normal",
                                                           depth_inverted=False, auto_exposure="off",
                                                           subrect=None))())


GPU_BUILD, GPU_MODEL = os.environ.get("WG_EQUIV_GPU_BUILD"), os.environ.get("WG_EQUIV_GPU_MODEL")


@unittest.skipUnless(GPU_BUILD and GPU_MODEL, "GPU smoke test: set WG_EQUIV_GPU_BUILD and WG_EQUIV_GPU_MODEL")
class GpuSmokeTest(unittest.TestCase):
    """Real spec 90 Pipeline and Runner on a synthetic model with synthetic inputs (tiny size): every boundary dumps;
    each Swin stage and the filter run in isolation on our own dumped inputs give the chained bytes; the D3D12 replay
    deletes its outputs after each step."""
    ROCM = os.environ.get("WG_EQUIV_ROCM", r"C:\Program Files\AMD\ROCm\7.2")
    RENDER, OUTPUT = (128, 72), (192, 108)

    def synth_frames(self, n):
        r = np.random.default_rng(9)
        rw, rh = self.RENDER
        for i in range(n):
            yield {"color": r.random((rh, rw, 3)).astype(np.float32), "depth": r.random((rh, rw)).astype(np.float32),
                   "motion": (r.random((rh, rw, 2)) - 0.5).astype(np.float32), "mv_scale": np.array([1.0, 1.0]),
                   "jitter": np.array([0.25 * (i % 2) - 0.125, 0.125]), "exposure": 1.0, "pre_exposure": 1.0,
                   "reset": i == 0}

    def pipe(self, backend):
        from clean.runner.pipeline import Pipeline
        return Pipeline(GPU_BUILD, self.ROCM, GPU_MODEL, self.RENDER, self.OUTPUT,
                        {"depth_inverted": False, "auto_exposure": False}, backend=backend,
                        kernels={"post": "post_recon2_r"})

    def test_dump_isolated_replay(self):
        from clean.runner.stage_runner import Runner
        with tempfile.TemporaryDirectory() as td:
            dd = os.path.join(td, "dump")
            man = M.dump(self.pipe("hip"), self.synth_frames(2), dd, full=True, backend="hip")
            self.assertEqual(man["missing"], [])
            run = Runner(GPU_BUILD, self.ROCM)
            for stage in M.SWIN + ["filter"]:
                sp = os.path.join(dd, f"in_{stage}.json")
                with open(sp, "w") as f:
                    json.dump(M.stage_inputs_from_dump(dd, 1, stage), f)
                out = os.path.join(td, "iso", stage)
                for st in M.run_stage(run, sp, out, model_dir=GPU_MODEL):
                    key = st.split(".", 1)[1]
                    a = np.load(os.path.join(out, st + ".npy"))
                    b = np.load(os.path.join(dd, "f0001", f"{stage}.{key}.npy"))
                    v = K.compare_stage(a, b, M.STAGE_OUTPUTS[stage][key], "HWC")
                    self.assertEqual(v["verdict"], K.MATCHES, (stage, key, v["reasons"]))
            s = M.replay(self.pipe("d3d12"), self.synth_frames(3), os.path.join(td, "r"),
                         then=lambda p, i: float(np.load(p)[..., :3].astype(np.float32).mean()), delete_after=True)
            self.assertEqual(s["frames"], 3)
            self.assertTrue(all(np.isfinite(v) for v in s["results"]))
            self.assertEqual(os.listdir(os.path.join(td, "r")), [])


if __name__ == "__main__":
    unittest.main()
