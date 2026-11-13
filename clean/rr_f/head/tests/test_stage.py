# SPDX-License-Identifier: MIT
"""D0-R stage / head: the committed test vectors against (a) the vectorised reference recomputed from the vectors'
inputs, (b) the slow element-by-element oracle, (c) the stated properties of page 55.

Run: python -m unittest clean.rr_f.head.tests.test_stage -v   (about a minute)
"""
import os
import re
import unittest

import numpy as np

from .. import block as B
from .. import fp
from .. import head as Hd
from .. import vectors as V
from . import slow_oracle as S

np.seterr(all="ignore")
VDIR = V.DEFAULT_OUT
MAN = V.load_manifest(VDIR)
CASES = {c["name"]: c for c in MAN["cases"]}
IMG = ("HID", "GATES", "PREC", "CANDW")
CODES = ("HID_code", "GATES_code", "CANDW_code")


def load(case, group, name):
    return V.load_array(VDIR, case[group][name])


def same16(a, b):
    return np.asarray(fp.step_diff16(a, b) == 0)


def stage_args(case):
    W = {n: load(case, "inputs", n) for n in ("Wq", "Wv", "Wo", "W1", "W2", "Tb", "bo", "b1", "b2")}
    k = {n: V.scalar_value(case["constants"][n]) for n in B.CONST_NAMES}
    ra = V.scalar_value(case["constants"]["ra"])
    m = V.scalar_value(case["constants"]["m"])
    return W, k, ra, m, tuple(case["shift"]["recurrent_shift"])


def scalars(case):
    sc = case["scalars"]
    return (V.scalar_value(sc["g"]), V.scalar_value(sc["s1"]), V.scalar_value(sc["s2"]),
            case["render_size"]["Wr"], case["render_size"]["Hr"])


def recompute(case):
    g, s1, s2, wr, hr = scalars(case)
    hold, hc = load(case, "inputs", "hold"), load(case, "inputs", "Hc")
    diag = {}
    if case["kind"] == "stage":
        W, k, ra, m, shift = stage_args(case)
        out = Hd.d0r_stage(load(case, "inputs", "x"), W, k, ra, m, g, shift, hold, hc, s1, s2, wr, hr, diag)
    else:
        y2h = load(case, "inputs", "y2h")
        out = Hd.output_images(y2h, s1, s2, wr, hr)
        out["Cnet"] = Hd.hidden_state(y2h, hold, hc, g)
    return out, diag


class Files(unittest.TestCase):
    def test_vectors_are_reproducible(self):
        self.assertEqual(V.main(["--check", "--out", VDIR]), 0)

    def test_file_constraints(self):
        for root, _, files in os.walk(VDIR):
            for f in files:
                p = os.path.join(root, f)
                data = open(p, "rb").read()
                self.assertLess(len(data), 256 * 1024, p)
                self.assertNotIn(b"\0", data, p)
                self.assertIsNone(re.search(rb"0x[0-9a-fA-F]{4}", data), p)
                data.decode("ascii")

    def test_scalar_rows(self):
        for r in CASES["converter_scalars"]["rows"]:
            h0, h1, cap = (V.scalar_value(r[n]) for n in ("h0", "h1", "S1CAP"))
            if "refused" in r:
                with self.assertRaises(ValueError):
                    Hd.derive_scalars(h0, h1, cap)
                continue
            s1, s2 = Hd.derive_scalars(h0, h1, cap)
            self.assertEqual((s1, s2), (V.scalar_value(r["s1"]), V.scalar_value(r["s2"])))

    def test_tables(self):
        t = CASES["tables"]["arrays"]
        allv = fp.all_f16()
        self.assertTrue(np.all(same16(V.load_array(VDIR, t["sig"]), fp.sig_table())))
        self.assertTrue(np.all(same16(V.load_array(VDIR, t["tanh16"]), fp.tanh16_table())))
        self.assertTrue(np.array_equal(V.load_array(VDIR, t["unorm8_store"]), fp.unorm8_store(allv)))
        self.assertTrue(np.array_equal(V.load_array(VDIR, t["snorm8_store"]), fp.snorm8_store(allv)))
        self.assertTrue(np.array_equal(V.load_array(VDIR, t["q8"]), fp.q8(allv)))


class Recompute(unittest.TestCase):
    """The vectorised reference on the vectors' inputs reproduces the vectors' outputs bit for bit."""

    def test_all_cases(self):
        for name, case in CASES.items():
            if case["kind"] not in ("stage", "head"):
                continue
            out, diag = recompute(case)
            for n in ("Cnet",) + IMG:
                self.assertTrue(np.all(same16(out[n], load(case, "outputs", n))), (name, n))
            for n in CODES:
                self.assertTrue(np.array_equal(out[n], load(case, "outputs", n)), (name, n))
            if case["kind"] == "stage":
                self.assertTrue(np.all(same16(diag["y2h"], load(case, "diagnostics", "y2h"))), name)
                self.assertTrue(np.all(same16(diag["Z"], load(case, "diagnostics", "Z"))), name)


class SlowOracle(unittest.TestCase):
    """Vectors against the element-by-element oracle."""

    def _windows(self, case, full):
        wt, ht = case["tensor_extent"]["Wt"], case["tensor_extent"]["Ht"]
        sx, sy = case["shift"]["recurrent_shift"]
        nwx, nwy = -(-(wt + sx) // 8), -(-(ht + sy) // 8)
        if full:
            return None
        return {0, nwx - 1, nwx * (nwy - 1), nwx * nwy - 1, (nwy // 2) * nwx + nwx // 2}

    def _check_block(self, name, full=False):
        case = CASES[name]
        W, k, ra, m, shift = stage_args(case)
        x = load(case, "inputs", "x")
        y2h = load(case, "diagnostics", "y2h")
        Z = load(case, "diagnostics", "Z")
        got, zs = S.stage_y2h(x, W, k, ra, m, shift, self._windows(case, full))
        self.assertTrue(len(got) > 0)
        for (Xa, Ya), row in got.items():
            self.assertTrue(np.all(same16(np.array(row), y2h[Ya, Xa])), (name, Xa, Ya))
        for widx, zz in zs.items():
            self.assertTrue(np.all(same16(np.array(zz), Z[widx])), (name, widx))
        return len(got)

    def _check_head(self, name):
        case = CASES[name]
        g, s1, s2, wr, hr = scalars(case)
        y2h = load(case, "diagnostics" if case["kind"] == "stage" else "inputs", "y2h")
        hold, hc = load(case, "inputs", "hold"), load(case, "inputs", "Hc")
        outs = {n: load(case, "outputs", n) for n in ("Cnet",) + IMG + CODES}
        ht, wt = y2h.shape[:2]
        for Ya in range(ht):
            for Xa in range(wt):
                r = S.head_token([float(v) for v in y2h[Ya, Xa]], hold[Ya, Xa], hc, g, Xa, Ya, s1, s2)
                self.assertTrue(np.all(same16(np.array(r["Cnet"]), outs["Cnet"][Ya, Xa])), (name, Xa, Ya))
                if Xa >= wr or Ya >= hr:
                    continue
                for n in IMG:
                    self.assertTrue(np.all(same16(np.array(r[n]), outs[n][Ya, Xa])), (name, n, Xa, Ya))
                for n in CODES:
                    self.assertEqual(list(r[n]), list(outs[n][Ya, Xa]), (name, n, Xa, Ya))

    def test_block_basic_all_windows(self):
        self._check_block("stage_basic", full=True)

    def test_block_border_windows(self):
        for name in ("stage_border", "stage_shift", "stage_zovf", "stage_edge", "stage_nan"):
            self._check_block(name)

    def test_head_all_cases(self):
        for name, case in CASES.items():
            if case["kind"] in ("stage", "head"):
                self._check_head(name)


class Properties(unittest.TestCase):
    def test_render_crop(self):
        """Images are per token: the render-size images are the top-left part of the full-extent images (tokens
        X >= Wr or Y >= Hr write nothing, 55.3)."""
        case = CASES["head_edge"]
        g, s1, s2, wr, hr = scalars(case)
        y2h = load(case, "inputs", "y2h")
        full = Hd.output_images(y2h, s1, s2, y2h.shape[1], y2h.shape[0])
        crop = Hd.output_images(y2h, s1, s2, wr, hr)
        for n in IMG + CODES:
            self.assertEqual(crop[n].shape[:2], (hr, wr))
            self.assertTrue(np.array_equal(fp.bits16(crop[n]) if n in IMG else crop[n],
                                           fp.bits16(full[n][:hr, :wr]) if n in IMG else full[n][:hr, :wr]))

    def test_nan_case(self):
        case = CASES["stage_nan"]
        g, s1, s2, wr, hr = scalars(case)
        o = {n: load(case, "outputs", n) for n in ("Cnet",) + IMG + CODES}
        self.assertTrue(np.isnan(o["PREC"][..., :3]).all() and np.isnan(o["GATES"][..., :3]).all())
        self.assertTrue(np.isnan(o["CANDW"]).all() and np.isnan(o["HID"]).all())
        self.assertTrue((o["GATES_code"] == 0).all() and (o["CANDW_code"] == 0).all() and (o["HID_code"] == 0).all())
        self.assertFalse(np.any(o["PREC"][..., :2] == float(s1) ** 2))
        self.assertTrue(np.all(o["PREC"][..., 3] == 0) and np.all(o["GATES"][..., 3] == 0))
        self.assertTrue(np.isnan(load(case, "diagnostics", "y2h")).all())

    def test_codes_follow_store_rule(self):
        for name, case in CASES.items():
            if case["kind"] not in ("stage", "head"):
                continue
            self.assertTrue(np.array_equal(load(case, "outputs", "HID_code"),
                                           fp.snorm8_store(load(case, "outputs", "HID"))))
            for n in ("GATES", "CANDW"):
                self.assertTrue(np.array_equal(load(case, "outputs", n + "_code"),
                                               fp.unorm8_store(load(case, "outputs", n))))

    def test_candw_exact_requirements(self):
        for name in ("head_edge", "head_random", "stage_edge"):
            case = CASES[name]
            wr, hr = case["render_size"]["Wr"], case["render_size"]["Hr"]
            grp = "inputs" if case["kind"] == "head" else "diagnostics"
            y = load(case, grp, "y2h")[:hr, :wr]
            c = load(case, "outputs", "CANDW")
            if len(Hd.head_channels()["candw"]) == 4:     # 55-ERRATUM-2: four logits, binary16 chain
                z = np.stack([y[..., n] for n in Hd.head_channels()["candw"]], -1)
                eq = np.isfinite(z).all(-1) & (z == z[..., :1]).all(-1)
                self.assertTrue(np.all(c[eq] == 0.25))
                nanq = np.isnan(z).any(-1) | (z == np.inf).any(-1) | (z == -np.inf).all(-1)   # d = inf - inf
                self.assertTrue(np.isnan(c[nanq]).all())
                ok = ~nanq
                self.assertTrue(np.all((c[ok] >= 0) & (c[ok] <= 1)))
                if name == "head_edge":   # the edge values sit on y10 / y12; y11 / y13 stay random
                    self.assertGreater(int(nanq.sum()), 10)
                continue
            ca, cb = Hd.head_channels()["candw"]
            a, b = y[..., ca], y[..., cb]
            self.assertTrue(np.array_equal(fp.bits16(c[..., 0]), fp.bits16(c[..., 1])))
            self.assertTrue(np.array_equal(fp.bits16(c[..., 2]), fp.bits16(c[..., 3])))
            eq = np.isfinite(a) & (a == b)
            self.assertTrue(np.all(c[eq] == 0.25))
            nanq = np.isnan(a) | np.isnan(b) | (np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b)))
            self.assertTrue(np.isnan(c[nanq]).all())
            pos = ~nanq & ((a == np.inf) | (b == -np.inf))
            neg = ~nanq & ((a == -np.inf) | (b == np.inf))
            self.assertTrue(np.all(c[pos][:, :2] == 0.5) and np.all(c[pos][:, 2:] == 0))
            self.assertTrue(np.all(c[neg][:, :2] == 0) and np.all(c[neg][:, 2:] == 0.5))
            if name == "head_edge":
                self.assertGreater(int(eq.sum()), 10)
                self.assertGreater(int(pos.sum()) + int(neg.sum()), 10)
                self.assertGreater(int(nanq.sum()), 10)

    def test_head_edge_coverage(self):
        """The edge case reaches the half-way store values and the sigmoid / tanh saturation."""
        case = CASES["head_edge"]
        hid = load(case, "outputs", "HID_code")
        gates = load(case, "outputs", "GATES_code")
        self.assertTrue((hid == 64).any() and (hid == -64).any() and (hid == 127).any() and (hid == -127).any())
        self.assertTrue((gates[..., :3] == 128).any() and (gates[..., :3] == 255).any())
        self.assertTrue((gates[..., :3] == 0).any())
        cand = load(case, "outputs", "CANDW_code")
        self.assertTrue((cand == 64).any() and (cand == 128).any())

    def test_zovf_rows(self):
        case = CASES["stage_zovf"]
        Z = load(case, "diagnostics", "Z")
        self.assertTrue(np.isinf(Z).any() and np.isfinite(Z).any())


if __name__ == "__main__":
    unittest.main()
