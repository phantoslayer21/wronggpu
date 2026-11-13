# SPDX-License-Identifier: MIT
"""Cross-team reference agreement and spec-50 stage wiring, synthetic inputs only."""
import unittest
from unittest.mock import patch

import numpy as np

from clean.rr_f.reference import network
from clean.rr_f.reference.block import Grid, window_block
from clean.rr_f.reference.fmt import d8
from clean.rr_f.reference.stages import stage_info
from clean.rr_f.head.block import run_block_stage
from clean.rr_f.head import fp, head
from clean.rr_f.tests import synth


class ReferenceAgreement(unittest.TestCase):
    def test_independent_final_block_matches_common_block(self):
        for seed, shape, shift in ((51, (8, 8), (0, 0)), (52, (4, 6), (6, 2)), (53, (2, 2), (4, 6))):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                wc, _ = synth.window_consts(rng, "A")
                weights = synth.stage_weights(rng, "IN0")
                constants = synth.stage_consts(rng)
                x = synth.e4m3(rng, (*shape, 32), .2)
                grid = Grid(shape[1], shape[0], *shift)
                xw = grid.gather(x)
                _, trace = window_block(xw, d8(xw), weights, constants["ra"], constants["m"], wc, trace=True)
                common = grid.scatter(trace["y2h"], 0)
                head = run_block_stage(x, weights, wc.as_dict(), constants["ra"], constants["m"], shift)
                self.assertTrue(np.all((common == head) | (np.isnan(common) & np.isnan(head))))


class NetworkWiring(unittest.TestCase):
    def test_children_are_row_major(self):
        x = np.arange(4 * 4 * 2, dtype=np.uint8).reshape(4, 4, 2)
        y = network.regroup_children(x)
        np.testing.assert_array_equal(y[1, 0], np.concatenate([x[2, 0], x[2, 1], x[3, 0], x[3, 1]]))

    def test_all_skips_and_current_frame_state_bindings(self):
        rng = np.random.default_rng(51)
        wc, _ = synth.window_consts(rng, "A")
        tokens = np.zeros((32, 32, 32), np.uint8)
        initial = np.zeros((32, 32, 4), np.float64)
        weights = {name: {} for name in network.ORDER}
        constants = {name: {"ra": 1., "m": 1., "g": 1.} for name in network.ORDER}
        shifts = {name: (2*(i%4), 2*((i//4)%4)) for i,name in enumerate(network.ENCODERS)}
        shifts.update({f"D{k}": (2*(k%4), 2*((k+1)%4)) for k in range(5)})
        seen, called = {}, []

        def stage(name, inputs, W, c, windows, shift):
            called.append(name)
            self.assertIs(W, weights[name])
            self.assertIs(c, constants[name])
            self.assertIs(windows, wc)
            self.assertEqual(shift, shifts[name.split("-")[0]])
            info = stage_info(name)
            level, channels = info["level"], info["C"]
            extent = 32 >> level
            if name == "IN0":
                self.assertIs(inputs["x"], tokens)
                self.assertIs(inputs["hold"], initial)
            elif name.startswith("E"):
                parent = "IN0" if level == 1 else f"E{level-1}"
                self.assertIs(inputs["hold"], seen[parent]["hmerged"])
                if level == 1:
                    np.testing.assert_array_equal(inputs["in"], network.regroup_children(seen[parent]["y2"]))
                else:
                    self.assertIs(inputs["x"], seen[parent]["merged"])
            elif name.endswith("-F"):
                parent = "E5" if level == 4 else f"D{level+1}-R"
                encoder = "IN0" if level == 0 else f"E{level}"
                self.assertIs(inputs["cin"], seen[parent]["y2"])
                self.assertIs(inputs["skip"], seen[encoder]["y2"])
            else:
                parent = "E5" if level == 4 else f"D{level+1}-R"
                self.assertIs(inputs["x"], seen[f"D{level}-F"]["y2"])
                self.assertIs(inputs["hold"], seen[f"E{level}"]["hnew"])
                self.assertIs(inputs["Hc"], seen[parent]["hnew"])
            row = {"y2": np.full((extent, extent, channels), len(called), np.uint8)}
            if info["hidden"]:
                row["hnew"] = np.full((extent, extent, 4), len(called), np.float64)
            if info["merged_hidden"]:
                row["hmerged"] = np.full((extent//2, extent//2, 4), len(called)+.5, np.float64)
            if info["Cn"]:
                row["merged"] = np.full((extent//2, extent//2, info["Cn"]), len(called), np.uint8)
            seen[name] = row
            return row, {}

        def head(x, W, k, ra, m, g, shift, hold, hc, s1, s2, wr, hr):
            called.append("D0-R")
            self.assertIs(x, seen["D0-F"]["y2"])
            self.assertIs(hold, seen["IN0"]["hnew"])
            self.assertIs(hc, seen["D1-R"]["hnew"])
            self.assertIs(W, weights["D0-R"])
            self.assertEqual(k, wc.as_dict())
            self.assertEqual((ra,m,g), (1.,1.,1.))
            self.assertEqual((s1,s2), (1.,1.))
            self.assertEqual(shift, (4, 6))
            self.assertEqual((wr, hr), (29, 19))
            return {"Cnet": hold}

        with patch.object(network, "run_stage", stage), patch.object(network, "d0r_stage", head):
            out = network.run_body(tokens, initial, weights, constants, wc, shifts, s1=1., s2=1., render_size=(29, 19))
        self.assertEqual(tuple(called), network.ORDER)
        self.assertIs(out["head"], out["stages"]["D0-R"])

    def synthetic_body(self):
        """run_body on synthetic inputs; also returns the head's y2h (D0-R's block output, through d0r_stage's diag)."""
        seen = {}
        real = network.d0r_stage

        def d0r(*a, **kw):
            seen["diag"] = kw.setdefault("diag", {})
            return real(*a, **kw)

        rng = np.random.default_rng(551)
        wc, _ = synth.window_consts(rng, "A")
        weights = {name: synth.stage_weights(rng, "IN0" if name == "D0-R" else name) for name in network.ORDER}
        constants = {name: synth.stage_consts(rng) for name in network.ORDER}
        shifts = {name: (2, 6) for name in network.ENCODERS}
        shifts.update({f"D{k}": (0, 2) for k in range(5)})
        tokens = synth.e4m3(rng, (32, 32, 32), .1)
        state = rng.uniform(0, 1, (32, 32, 4)).astype(np.float16).astype(np.float64)
        with patch.object(network, "d0r_stage", d0r):
            out = network.run_body(tokens, state, weights, constants, wc, shifts, s1=1.25, s2=.75, render_size=(29, 19))
        self.assertEqual(tuple(out["stages"]), network.ORDER)
        self.assertEqual(out["head"]["Cnet"].shape, (32, 32, 4))
        for key in ("HID", "GATES", "PREC", "CANDW"):
            self.assertEqual(out["head"][key].shape, (19, 29, 4))
        return out, seen["diag"]["y2h"][:19, :29]

    def test_real_synthetic_body(self):
        # 55-ERRATUM-2 (head.py's default since 79924487, and the shipped runtime's reading, rr_runtime.cpp head18):
        # CANDW W4.r..a from the four logits y10, y11, y12, y13 by the binary16 chain, HID = tanh of y14 .. y17. Before
        # the erratum this test checked page 55's paired reading (CANDW channels 0 = 1, 2 = 3; test below).
        out, y = self.synthetic_body()
        want = np.stack(head.candw4_values(*(y[..., n] for n in (10, 11, 12, 13))), -1)
        np.testing.assert_array_equal(out["head"]["CANDW"], want)
        np.testing.assert_array_equal(out["head"]["CANDW_code"], fp.unorm8_store(want))
        np.testing.assert_array_equal(out["head"]["HID"], np.stack([fp.tanh16(y[..., n]) for n in (14, 15, 16, 17)], -1))

    def test_real_synthetic_body_page55_paired(self):
        # page 55's paired reading, kept in head.py for comparison (erratum2=False; the runtime's WG_RR_HEAD18=0)
        with patch.object(head, "P55_ERRATUM2", False):
            out, _ = self.synthetic_body()
        np.testing.assert_array_equal(out["head"]["CANDW_code"][..., 0], out["head"]["CANDW_code"][..., 1])
        np.testing.assert_array_equal(out["head"]["CANDW_code"][..., 2], out["head"]["CANDW_code"][..., 3])


if __name__ == "__main__":
    unittest.main()
