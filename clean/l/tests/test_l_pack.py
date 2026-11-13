# SPDX-License-Identifier: MIT
"""Model L weight packing (clean/l/runner/l_pack.py), CPU only, synthetic parameters.

Run: python -m unittest discover -s clean/l/tests -t . -p test_l_pack.py
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.l.runner import l_pack as LP
from clean.reference import decoder as MD
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
        for C_in, C_out, n, lo in ((32, 32, 20, 0), (80, 32, 20, 60), (32, 32, 32, 0), (64, 16, 16, 16)):
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

    def test_dec_project(self):
        rng = np.random.default_rng(4)
        W = q8(rng.normal(0, 0.3, (4, 32, 64)))
        b = rng.normal(0, 1, (4, 32)).astype(np.float32)
        blob = LP.pack_dec_project(W, b)
        self.assertEqual(len(blob), 4 * 128 + 128 * 64)
        Wu, bu = LP.unpack_dec_project(blob)
        self.assertTrue(np.array_equal(Wu, W))
        self.assertTrue(np.array_equal(bu, b))

    def test_dec_project_is_expand_section_format(self):
        """Rows R_E(32): the expand section's row order at C = 32 (sub-pixel s, then layout L of its 32 channels)."""
        rng = np.random.default_rng(5)
        W = q8(rng.normal(0, 0.3, (4, 32, 64)))
        b = rng.normal(0, 1, (4, 32)).astype(np.float32)
        drow = P.resolve_d_row("gfx12")
        RE = P.rows_expand(32, drow)
        want = P.pack_tile_vector(b.reshape(128), RE, drow) + P.pack_frags(W.reshape(128, 64), RE,
                                                                            LP.kord_groups32(64))
        self.assertEqual(LP.pack_dec_project(W, b), want)

    def test_block_equals_model_m_packer(self):
        rng = np.random.default_rng(6)
        for stage in ("b0", "dec1"):
            cfg = MS.STAGES[stage]
            p = MS.random_params(stage, rng)
            want = P.pack_stage(stage, p)[:P.stage_sections(stage)[1]]
            got = LP.pack_block({k: v for k, v in p.items() if k not in ("We", "be")}, cfg["C"], cfg["heads"],
                                cfg["mlp"])
            self.assertEqual(got, want, stage)
        p = L.random_block_params(rng, 32, 2, 128)
        blob = LP.pack_block(p, 32, 2, 128)
        lay = P.block_layout(32, 2, 128)
        self.assertEqual(len(blob), lay["SIZE"] + (-lay["SIZE"]) % 256)
        self.assertEqual((lay["HPU"], lay["NMLP"], lay["MLP_BYTES"], lay["HEAD_BYTES"]), (8, 1, 8192, 3072))

    def test_values_and_coef_blobs(self):
        rng = np.random.default_rng(7)
        params = L.random_params(rng)
        blob = LP.pack_values(params["enc0"])
        self.assertEqual(len(blob), E0.BLOB_BYTES)
        self.assertEqual(blob[:256], bytes(256))                         # zero bias
        self.assertEqual(blob[E0.OFF_DITHER:E0.OFF_DITHER + 2048],
                         np.asarray(params["enc0"]["T"]).astype(np.uint8).tobytes())
        coef = LP.pack_coef(params["dec0"])
        self.assertEqual(len(coef), D0.BLOB_BYTES)
        full = D0.pack({"W": np.zeros((68, 64), np.float32), "b": np.zeros(68, np.float32),
                        "S": params["dec0"]["S"], "T": params["dec0"]["T"]})
        self.assertEqual(coef, full)


class ModelTest(unittest.TestCase):
    def test_launch_lists(self):
        ln = LP.encoder_launches()
        self.assertEqual([x["kind"] for x in ln], ["values", "linear", "linear", "swin", "project"])
        self.assertEqual((ln[1]["src"], ln[1]["C_in"], ln[1]["C_out"], ln[1]["out_round"]), ("q", 32, 32, "relu_q8"))
        self.assertEqual((ln[2]["src"], ln[2]["C_in"]), ("x", 32))
        self.assertEqual(ln[3]["blob"], "enc0_block")
        self.assertEqual([(x.get("in_format"), x.get("out_format")) for x in ln[1:4]],
                         [("fp8", "fp8"), ("fp8", "fp16"), ("fp16", "fp8")])
        self.assertEqual(ln[3]["border"], "input")
        dn = LP.decoder_launches()
        self.assertEqual([x["kind"] for x in dn], ["dec_project", "swin", "coef"])
        self.assertEqual((dn[0]["out_format"], dn[1]["in_format"], dn[1]["out_format"], dn[1]["border"]),
                         ("fp8", "fp8", "fp16", "coarse"))
        self.assertEqual((dn[2]["values"], dn[2]["dec_pixel"], dn[2]["in_format"]), (T.L_DEC_VALUES, "l", "fp16"))
        with self.assertRaises(ValueError):
            LP.map_format(None, "none")

    def test_pack_model_l(self):
        rng = np.random.default_rng(8)
        params = L.random_params(rng)
        blobs = LP.pack_model_l(params)
        self.assertEqual(sorted(blobs), sorted(["enc0.values", "enc0.Win1", "enc0.Win2", "enc0_block", "enc0.project",
                                                "dec0.project", "dec0_block", "dec0.coef"]))
        self.assertEqual(len(blobs["enc0.Win1"]), 4 * 32 + 32 * 32)
        self.assertEqual(len(blobs["enc0.Win2"]), 4 * 32 + 32 * 32)
        self.assertEqual(len(blobs["dec0.project"]), 4 * 128 + 128 * 64)
        self.assertEqual(blobs, LP.pack_model_l(params))                  # deterministic

    def test_unpacked_weights_reproduce_the_oracle(self):
        """The weights decoded from the blobs, applied the way the kernels apply them (whole 32-channel input maps,
        zero weights for the padding), give the oracle's maps byte for byte."""
        rng = np.random.default_rng(9)
        params = L.random_params(rng)
        x = ME.random_inputs(rng, rw=36, rh=20)
        out = L.encode(x, params, parity=0, keep=True)
        m = out["maps"]
        blobs = LP.pack_model_l(params)
        enc = params["enc0"]
        W1, b1 = LP.unpack_linear(blobs["enc0.Win1"], 32, 32)
        W2, b2 = LP.unpack_linear(blobs["enc0.Win2"], 32, 32)
        qpad = np.concatenate([m["q"], np.zeros(m["q"].shape[:-1] + (12,))], axis=-1)
        h1 = L.linear(qpad, W1, b1, "f16", "relu_q8")
        self.assertTrue(np.array_equal(h1, L.linear(m["q"], enc["Win1"], enc["bin1"], "f16", "relu_q8")))
        h2 = L.linear(h1, W2, b2, "f16", "none")
        h3 = q8(L.swin_block(h2, params["enc0_block"], T.block_offset(T.TABLE, "first", 0), 2, T.M_BLOCK_CONSTS,
                             "input", T.block_halo(T.TABLE, "first")))
        self.assertTrue(np.array_equal(h3, m["x"]))
        Wp, bpj = LP.unpack_project(blobs["enc0.project"], 128, 64)
        feats = L.project_groups(L.space_to_depth(h3), Wp, bpj, (32, 32, 32, 32), 6.28125)
        self.assertEqual(feats.astype(np.float32).tobytes(), out["features"].tobytes())
        xd = MD.random_inputs(rng, rw=36, rh=20)
        dout = L.decode(xd, params, 0, keep=True)
        Wd, bd = LP.unpack_dec_project(blobs["dec0.project"])
        y = L.dec_project(xd["features"], Wd, bd)
        self.assertTrue(np.array_equal(y, dout["maps"]["y"]))


if __name__ == "__main__":
    unittest.main()
