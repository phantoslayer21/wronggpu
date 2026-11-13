# SPDX-License-Identifier: MIT
"""Regression test (RR bluefix, 2026-10-21): an all-grey frame through a colour-symmetric network keeps the three colour
channels of every hidden state equal - IN0 / E1..E5 hnew, hmerged, the Hc each recurrent half reads, D4-R..D1-R hnew
and Cnet - on the CPU reference and (with WG_RR_GPU_TESTS=1; gpu/build.run takes the GPU lock (WG_GPU_LOCK, BUILDING.md) itself) on the GPU body + head.

The synthetic model is "colour-symmetric": every tensor that produces stream channels (Wo, bo, W2, b2, Wm, bm, Wa, ba,
Wup, bup) has identical output columns 0, 1, 2, and every hidden projection (Wh, bh) identical columns 0, 1, 2; the
IN0 tokens have identical channels 0, 1, 2 and the input hidden state is grey (hold_0 = hold_1 = hold_2, hold_3 = 0).
So the D0-R block outputs 0, 1, 2 are equal, while output 4 is not. Selecting y2h channel 4 as the blue gate logit
(page 55.2's c = 0, 1, 4, 5 taken in the converter's numbering) breaks the symmetry of Cnet only, which is how the
real-frame RR output became yellow with blue edges; the negative control below shows this test fails on that rule.

    python -m unittest clean.rr_f.tests.test_colour_symmetry                         # CPU
    WG_RR_GPU_TESTS=1 WG_RR_BUILD=<dir> python -m unittest clean.rr_f.tests.test_colour_symmetry   (takes the GPU lock itself)
"""
import functools
import os
import unittest
from unittest import mock

import numpy as np

from clean.rr_f.head import head as Hd
from clean.rr_f.reference import network
from clean.rr_f.reference.stages import stage_info

GPU = os.environ.get("WG_RR_GPU_TESTS") == "1"


def symmetric_fixture(seed=551):
    """The GPU harness's synthetic body fixture made colour-symmetric (module docstring)."""
    from clean.rr_f.gpu import network as gnet
    tokens, state0, weights, constants, wc, shifts = gnet.fixture(seed)
    tokens = tokens.copy()
    tokens[..., 1] = tokens[..., 2] = tokens[..., 0]
    state0 = state0.copy()
    state0[..., 1] = state0[..., 2] = state0[..., 0]
    state0[..., 3] = 0.0
    sym = {}
    for name, W in weights.items():
        W = {k: np.array(v, copy=True) for k, v in W.items()}
        C = stage_info("IN0" if name == "D0-R" else name)["C"]
        for k in ("Wo", "bo", "W2", "b2", "Wm", "bm", "Wa", "ba", "Wh", "bh"):
            if k in W:
                W[k][..., 1] = W[k][..., 0]
                W[k][..., 2] = W[k][..., 0]
        for k in ("Wup", "bup"):           # last axis n * C + c, n = child
            if k in W:
                for n in range(4):
                    W[k][..., n * C + 1] = W[k][..., n * C]
                    W[k][..., n * C + 2] = W[k][..., n * C]
        sym[name] = W
    return tokens, state0, sym, constants, wc, shifts


def hidden_tensors(stages, cnet):
    """(label, tensor) of every hidden-state tensor of one frame, in network order."""
    out = []
    for name in network.ENCODERS:
        out.append((f"{name}.hnew", stages[name]["hnew"]))
        if "hmerged" in stages[name]:
            out.append((f"{name}.hmerged", stages[name]["hmerged"]))
    coarse = "E5"
    for k in (4, 3, 2, 1):
        out.append((f"D{k}-R.Hc", stages[coarse]["hnew"]))
        out.append((f"D{k}-R.hnew", stages[f"D{k}-R"]["hnew"]))
        coarse = f"D{k}-R"
    out.append(("D0-R.Hc", stages[coarse]["hnew"]))
    out.append(("Cnet", cnet))
    return out


def asymmetric(tensors):
    """Labels whose channels 0, 1, 2 are not bit-identical."""
    bad = []
    for label, t in tensors:
        b = np.asarray(t, np.float64).astype(np.float16).view(np.uint16)
        if not (np.array_equal(b[..., 0], b[..., 1]) and np.array_equal(b[..., 0], b[..., 2])):
            bad.append(label)
    return bad


def run_cpu(data, render=(29, 19)):
    tokens, state0, W, K, wc, shifts = data
    return network.run_body(tokens, state0, W, K, wc, shifts, s1=1.25, s2=.75, render_size=render)


class ColourSymmetryCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = symmetric_fixture()
        cls.body = run_cpu(cls.data)

    def test_every_hidden_state_symmetric(self):
        t = hidden_tensors(self.body["stages"], self.body["head"]["Cnet"])
        self.assertEqual(asymmetric(t), [])
        for label, v in t:
            if label.startswith("E") or label.startswith("IN0"):
                self.assertTrue(np.all(np.asarray(v)[..., 3] == 0), label)

    def test_fixture_separates_the_gate_rules(self):
        """The D0-R outputs 0, 1, 2 are equal and output 4 differs, so the test decides between the two rules."""
        diag = {}
        tokens, state0, W, K, wc, shifts = self.data
        stages = self.body["stages"]
        c = K["D0-R"]
        Hd.d0r_stage(stages["D0-F"]["y2"], W["D0-R"], wc.as_dict(), c["ra"], c["m"], c["g"],
                     ((shifts["D0"][0] + 4) % 8, (shifts["D0"][1] + 4) % 8), stages["IN0"]["hnew"],
                     stages["D1-R"]["hnew"], 1.25, .75, 29, 19, diag)
        y = diag["y2h"]
        self.assertTrue(np.array_equal(y[..., 0], y[..., 2]))
        self.assertGreater(np.mean(y[..., 0] != y[..., 4]), 0.9)

    def test_negative_control_page_55_2_numbering(self):
        """With c = 0, 1, 4, 5 (the rule before the fix) Cnet loses the symmetry; every other tensor keeps it."""
        old = functools.partial(Hd.hidden_state, gate_ch=Hd.SPEC_55_2_GATE_CH)
        with mock.patch.object(Hd, "hidden_state", old):
            body = run_cpu(self.data)
        self.assertEqual(asymmetric(hidden_tensors(body["stages"], body["head"]["Cnet"])), ["Cnet"])


@unittest.skipUnless(GPU, "GPU test: set WG_RR_GPU_TESTS=1 and WG_RR_BUILD (not inside gpu_lock: build.run takes the lock)")
class ColourSymmetryGPU(unittest.TestCase):
    def test_gpu_body_and_head(self):
        from pathlib import Path
        from clean.rr_f.gpu import build, network as gnet
        data = symmetric_fixture()
        ar, kernels, dispatches, calls, ref, base, ha, _, regroup = gnet.prepare(data=data)
        img, _ = build.run(ar, kernels, dispatches, workdir=str(Path(build.BUILD) / "colour-symmetry"))
        report = gnet.analyze(img, calls, ref, base, ha, regroup)
        self.assertTrue(report["passed"], "GPU body / head differ from the CPU reference")
        ht, wt = data[0].shape[:2]
        off = base + ha.ptr["cnet"]
        cnet = img[off:off + ht * wt * 8].copy().view("<f2").reshape(ht, wt, 4).astype(np.float64)
        got = {name: call.outputs(img) for name, call in calls.items()}
        stages = {name: {k: (v.view(np.float16).astype(np.float64) if v.dtype == np.uint16 else v)
                         for k, v in out.items()} for name, out in got.items()}
        self.assertEqual(asymmetric(hidden_tensors(stages, cnet)), [])


if __name__ == "__main__":
    unittest.main()
