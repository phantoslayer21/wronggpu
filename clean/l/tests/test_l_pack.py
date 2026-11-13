# SPDX-License-Identifier: MIT
"""Model L weight packing (clean/l/runner/l_pack.py), CPU only, synthetic parameters.

Run: python -m unittest discover -s clean/l/tests -t . -p test_l_pack.py
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.l.runner import l_pack as LP
from clean.reference import encoder as ME
from clean.reference import swin as MS
from clean.reference.fp8 import q8
from clean.runner import dec0 as D0
from clean.runner import enc0 as E0
from clean.runner import packing as P


class OrderTest(unittest.TestCase):
    def test_groups32_is_dec0_order(self):
        self.assertTrue(np.array_equal(LP.kord_groups32(64), D0.kord_decoder()))

    def test_groups32_blocks(self):
        K = LP.kord_groups32(128)
        self.assertEqual(sorted(K), list(range(128)))
        for g in range(4):
            self.assertEqual(sorted(K[32 * g:32 * g + 32]), list(range(32 * g, 32 * g + 32)))
        with self.assertRaises(ValueError):
            LP.kord_groups32(80)


class RoundTripTest(unittest.TestCase):
    def test_linear(self):
        rng = np.random.default_rng(1)
        for C_in, C_out, n, lo in ((80, 32, 20, 0), (80, 32, 20, 60), (32, 32, 32, 0), (64, 16, 16, 16)):
            W = q8(rng.normal(0, 0.5, (C_out, n)))
            b = rng.normal(0, 1, C_out).astype(np.float32)
            blob = LP.pack_linear(W, b, C_in, C_out, lo)
            self.assertEqual(len(blob), 4 * C_out + C_out * C_in)
            Wu, bu = LP.unpack_linear(blob, C_in, C_out)
            want = np.zeros((C_out, C_in))
            want[:, lo:lo + n] = W
            self.assertTrue(np.array_equal(Wu, want))
            self.assertTrue(np.array_equal(bu, b))
        with self.assertRaises(ValueError):
            LP.pack_linear(np.zeros((32, 20)), np.zeros(32), 80, 32, lo=70)

    def test_linear_is_merge_section_format(self):
        """The linear blob is the patch-merge section format when the K order is the merge's."""
        rng = np.random.default_rng(2)
        W = q8(rng.normal(0, 0.5, (96, 64)))
        b = rng.normal(0, 1, 96).astype(np.float32)
        drow = P.resolve_d_row("gfx12")
        RL = P.rows_layout_l(96, drow)
        self.assertEqual(LP.pack_linear(W, b, 64, 96),
                         P.pack_tile_vector(b, RL, drow) + P.pack_frags(W, RL, P.kord_layout_l(64)))

    def test_project(self):
        rng = np.random.default_rng(3)
        W = q8(rng.normal(0, 0.3, (64, 128)))
        b = rng.normal(0, 1, 64).astype(np.float32)
        blob = LP.pack_project(W, b, (32, 32, 32, 32))
        self.assertEqual(len(blob), 4 * 64 + 64 * 128)
        Wu, bu = LP.unpack_project(blob, 128, 64)
        self.assertTrue(np.array_equal(Wu, W))
        self.assertTrue(np.array_equal(bu, b))
        with self.assertRaises(ValueError):
            LP.pack_project(W, b, (20, 20, 20, 20, 48))

    def test_block_equals_model_m_packer(self):
        """The L block packer is packing.pack_block: Model M's stage blobs come out byte for byte, and C = 32 has
        the layout's size."""
        rng = np.random.default_rng(4)
        for stage in ("b0", "dec1"):
            cfg = MS.STAGES[stage]
            p = MS.random_params(stage, rng)
            want = P.pack_stage(stage, p)[:P.stage_sections(stage)[1]]
            got = LP.pack_block({"x." + k: v for k, v in p.items()}, "x.", cfg["C"], cfg["heads"], cfg["mlp"])
            self.assertEqual(got, want, stage)
        p = L.random_block_params(rng, 32, 2, 128, "blk.")
        blob = LP.pack_block(p, "blk.", 32, 2, 128)
        lay = P.block_layout(32, 2, 128)
        self.assertEqual(len(blob), lay["SIZE"] + (-lay["SIZE"]) % 256)
        self.assertEqual((lay["HPU"], lay["NMLP"], lay["MLP_BYTES"], lay["HEAD_BYTES"]), (8, 1, 8192, 3072))

    def test_values_blob(self):
        rng = np.random.default_rng(5)
        enc = L.random_params(rng)["enc0"]
        blob = LP.pack_values(enc)
        self.assertEqual(len(blob), E0.BLOB_BYTES)
        self.assertEqual(blob[:256], bytes(256))                         # zero bias
        self.assertEqual(blob[E0.OFF_DITHER:E0.OFF_DITHER + 2048], np.asarray(enc["T"]).astype(np.uint8).tobytes())


class ModelTest(unittest.TestCase):
    def test_launch_list(self):
        ln = LP.encoder_launches()
        self.assertEqual([x["kind"] for x in ln], ["values", "linear", "linear", "swin", "concat", "project"])
        self.assertEqual((ln[1]["src"], ln[1]["C_in"], ln[1]["lo"], ln[1]["C_out"]), ("raw", 80, 0, 32))
        self.assertEqual((ln[2]["src"], ln[2]["C_in"]), ("x", 32))

    def test_pack_model_l(self):
        rng = np.random.default_rng(6)
        params = L.random_params(rng)
        blobs = LP.pack_model_l(params)
        self.assertEqual(sorted(blobs), sorted(["enc0.values", "enc0.Wa", "enc0.Wb", "enc0.blk", "enc0.project",
                                                "dec0.blk", "dec0"]))
        self.assertEqual(len(blobs["enc0.Wa"]), 4 * 32 + 32 * 80)
        self.assertEqual(len(blobs["enc0.Wb"]), 4 * 32 + 32 * 32)
        self.assertEqual(len(blobs["dec0"]), D0.BLOB_BYTES)
        self.assertEqual(blobs["dec0"], D0.pack({k: params["dec0"][k] for k in ("W", "b", "S", "T")}))
        self.assertEqual(blobs, LP.pack_model_l(params))                  # deterministic

    def test_unpacked_weights_reproduce_the_oracle(self):
        """The weights decoded from the blobs, applied to the oracle's maps the way the kernels apply them (whole
        input maps, zero weights outside a select), give the oracle's maps byte for byte."""
        rng = np.random.default_rng(7)
        params = L.random_params(rng)
        x = ME.random_inputs(rng, rw=36, rh=20)
        out = L.encode(x, params["enc0"], parity=0, keep=True)
        m = out["maps"]
        blobs = LP.pack_model_l(params)
        enc = params["enc0"]
        Wa, ba = LP.unpack_linear(blobs["enc0.Wa"], 80, 32)
        Wb, bb = LP.unpack_linear(blobs["enc0.Wb"], 32, 32)
        h1 = q8(m["raw"] @ Wa.T + ba)
        self.assertTrue(np.array_equal(h1, q8(m["raw"][..., :20] @ np.asarray(enc["Wa"], np.float64).T + enc["ba"])))
        h2 = q8(h1 @ Wb.T + bb)
        bp = {k[4:]: v for k, v in enc.items() if k.startswith("blk.")}
        h3 = q8(L.swin_block(h2, bp, T.block_offset(T.TABLE, "first", 0), 2, T.M_BLOCK_CONSTS))
        self.assertTrue(np.array_equal(h3, m["x"]))
        Wp, bpj = LP.unpack_project(blobs["enc0.project"], 128, 64)
        feats = L.project_groups(m["z"], Wp, bpj, (32, 32, 32, 32))
        self.assertEqual(feats.astype(np.float32).tobytes(), out["features"].tobytes())


if __name__ == "__main__":
    unittest.main()
