# SPDX-License-Identifier: MIT
"""Model-keyed stage tables of the reference and the packers (#89): Model M's rows are exactly the tables that were
hard-wired before, every function defaults to Model M, an unknown model is refused by name, and a second model's rows
plug in without touching Model M's (a synthetic model "T" made of Model M's own stages, renamed).

Run: python -m unittest clean.tests.test_model_tables
"""
import unittest

import numpy as np

from clean.reference import swin as S
from clean.runner import packing as P
from clean.runner import stage_runner as R

# Model M's tables as they were written before the tables became model-keyed (the literals of that code)
M_STAGES = {
    "enc1": dict(kind="enc", C=64, heads=2, mlp=256, C_out=96),
    "enc2": dict(kind="enc", C=96, heads=4, mlp=384, C_out=128),
    **{f"b{i}": dict(kind="b", C=128, heads=4, mlp=512) for i in range(6)},
    "dec2": dict(kind="dec", C=96, heads=4, mlp=384, C_low=128),
    "dec1": dict(kind="dec", C=64, heads=2, mlp=256, C_low=96),
}
M_SWIN_ORDER = ["enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"]


def m_window_offset(stage, parity):  # the rule as it was written before
    if stage in ("b1", "b3", "b5", "dec2", "dec1"):
        return (0, 0) if parity == 0 else (2, 0)
    return (2, 2) if parity == 0 else (0, 2)


class ModelMRows(unittest.TestCase):
    def test_stage_table(self):
        self.assertIs(S.stage_table(), S.STAGES)
        self.assertIs(S.MODEL_STAGES["M"], S.STAGES)
        self.assertEqual(S.STAGES, M_STAGES)
        self.assertEqual(list(S.STAGES), list(M_STAGES))  # the same order

    def test_window_offsets(self):
        for stage in list(M_STAGES) + ["enc0", "unknown"]:
            for parity in (0, 1):
                self.assertEqual(S.window_offset(stage, parity), m_window_offset(stage, parity), (stage, parity))
                self.assertEqual(S.window_offset(stage, parity, "M"), m_window_offset(stage, parity))

    def test_stage_lists(self):
        self.assertEqual(P.BNECK_STAGES, tuple(f"b{k}" for k in range(6)))
        self.assertEqual(P.MID_STAGES, ("enc2",) + P.BNECK_STAGES + ("dec2",))
        self.assertIs(P.bneck_stages(), P.BNECK_STAGES)
        self.assertIs(P.mid_stages(), P.MID_STAGES)
        from clean.runner import pipeline
        self.assertEqual(pipeline.SWIN_ORDER, M_SWIN_ORDER)
        self.assertIs(pipeline.swin_order(), pipeline.SWIN_ORDER)
        self.assertEqual(R.MODELS, ("M", "K"))

    def test_messages_unchanged_for_m(self):
        with self.assertRaisesRegex(ValueError, r"^bneck needs the tensors of the six blocks b0\.\.b5$"):
            P.pack_bneck([{}])
        with self.assertRaisesRegex(ValueError, r"^mid needs the tensors of enc2, b0\.\.b5 and dec2$"):
            P.pack_mid([{}])
        with self.assertRaisesRegex(ValueError, r"^a bneck blob holds six equal b-stage blobs$"):
            P.split_bneck(b"1234567")
        with self.assertRaisesRegex(ValueError, r"^bneck needs six blobs$"):
            P.split_bneck([b"a"])
        with self.assertRaisesRegex(ValueError, r"^mid needs eight blobs$"):
            P.split_mid([b"a"])

    def test_unknown_model(self):
        for f in (lambda: S.stage_table("Z"), lambda: S.window_offset("b0", 0, "Z"), lambda: P.bneck_stages("Z"),
                  lambda: P.mid_stages("Z"), lambda: P.stage_sections("b0", "Z")):
            with self.assertRaisesRegex(ValueError, r"for model 'Z'; known: M"):
                f()


class SecondModel(unittest.TestCase):
    """A synthetic model "T": Model M's enc2, b0 and dec2 under new names, with its own offsets and lists."""

    def setUp(self):
        self.rows = {"t_enc": S.STAGES["enc2"], "t_b": S.STAGES["b0"], "t_dec": S.STAGES["dec2"]}
        S.MODEL_STAGES["T"] = self.rows
        S.WINDOW_OFFSETS["T"] = {"stages": {"t_b": ((1, 1), (3, 3))}, "default": ((0, 0), (2, 2))}
        P.MODEL_BNECK_STAGES["T"] = ("t_b",)
        P.MODEL_MID_STAGES["T"] = ("t_enc", "t_b", "t_b", "t_dec")

    def tearDown(self):
        for t in (S.MODEL_STAGES, S.WINDOW_OFFSETS, P.MODEL_BNECK_STAGES, P.MODEL_MID_STAGES):
            t.pop("T", None)

    def test_rows_and_packing(self):
        self.assertEqual(S.window_offset("t_b", 1, "T"), (3, 3))
        self.assertEqual(S.window_offset("t_enc", 0, "T"), (0, 0))
        self.assertEqual(S.window_offset("b1", 0), (0, 0))  # Model M untouched
        rng = np.random.default_rng(7)
        params = {s: S.random_params(s, rng, model="T") for s in self.rows}
        # the same stage shapes as Model M's stages: the same bytes as packing those stages as Model M
        for s, m in (("t_enc", "enc2"), ("t_b", "b0"), ("t_dec", "dec2")):
            self.assertEqual(P.pack_stage(s, params[s], model="T"), P.pack_stage(m, params[s]))
            self.assertEqual(P.stage_sections(s, "T"), P.stage_sections(m))
            self.assertEqual(P.stage_blob_size(s, "T"), P.stage_blob_size(m))
        self.assertEqual(P.pack_bneck({"t_b": params["t_b"]}, model="T"), P.pack_stage("b0", params["t_b"]))
        mid = P.pack_mid({s: params[s] for s in self.rows}, model="T")
        self.assertEqual(len(P.split_mid(mid, "T")), 4)
        x, skip = S.random_inputs("t_dec", 8, 8, rng, model="T")
        a = S.run_stage("t_dec", params["t_dec"], x, (0, 0), skip, model="T")
        b = S.run_stage("dec2", params["t_dec"], x, (0, 0), skip)
        np.testing.assert_array_equal(a["out"], b["out"])
        with self.assertRaisesRegex(ValueError, r"^mid needs the tensors of t_enc, t_b\.\.t_b and t_dec$"):
            P.pack_mid([{}], model="T")


if __name__ == "__main__":
    unittest.main()
