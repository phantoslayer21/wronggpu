# SPDX-License-Identifier: MIT
"""Model L oracle self-checks (no GPU, no compiler; synthetic data only).

Run: python -m unittest discover -s clean/l/tests -t .

The strongest checks reduce Model L to Model M through the table and require Model M's oracle results byte for byte:
the block with M's constants, the encoder with M's op list, the decoder with M's projection rows and no block, and
the whole network chain.
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.reference import decoder as MD
from clean.reference import encoder as ME
from clean.reference import swin as MS
from clean.reference.fp8 import is_fp8_exact, q8

# Model M's input encoder written as an L op list.
M_ENC_OPS = (
    {"op": "quant", "mode": "dither", "src": "values", "dst": "q"},
    {"op": "space_to_depth", "src": "q", "dst": "z"},
    {"op": "project", "src": "z", "w": "W", "b": "b", "groups": (20, 20, 20, 20), "n_out": 64, "clamp": 6.28125},
)
# Model M's decoder written as an L op list: the projection, then the value positions straight away (no block).
M_DEC_OPS = (
    {"op": "dec_project", "src": "features", "dst": "y", "w": "W", "b": "b", "groups": (32, 32)},
    {"op": "depth_to_space", "src": "y", "dst": "p"},
    {"op": "pick", "src": "p", "values": T.DEC0_VALUES},
)


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def m_dec_as_l(p):
    """Model M's dec0 tensors (W [68, 64], b [68]) as L's [4, 32, 64] / [4, 32] with zeros outside the value rows."""
    W = np.zeros((4, 32, 64), np.float32)
    b = np.zeros((4, 32), np.float32)
    Wm = np.asarray(p["W"], np.float32).reshape(4, 17, 64)
    bm = np.asarray(p["b"], np.float32).reshape(4, 17)
    for s in range(4):
        for k, c in enumerate(T.DEC0_VALUES):
            W[s, c] = Wm[s, k]
            b[s, c] = bm[s, k]
    return {"W": W, "b": b, "S": p["S"], "T": p["T"]}


class BlockTest(unittest.TestCase):
    def test_block_equals_model_m_oracle(self):
        """With Model M's constants the L block is clean/reference/swin.py swin_block, byte for byte."""
        rng = np.random.default_rng(1)
        for C, heads, mlp in ((32, 2, 128), (64, 2, 256), (96, 4, 384)):
            p = L.random_block_params(rng, C, heads, mlp)
            for H, W in ((9, 13), (16, 8), (5, 7)):
                x = q8(rng.normal(0, 1.5, (H, W, C)))
                for off in ((0, 0), (2, 0), (2, 2), (0, 2), (1, 3)):
                    want = MS.swin_block(x, p, off, heads)
                    got = L.swin_block(x, p, off, heads, T.M_BLOCK_CONSTS)
                    self.assertTrue(_same(got, want), (C, H, W, off))

    def test_constants_are_used(self):
        rng = np.random.default_rng(2)
        p = L.random_block_params(rng, 32, 2, 128)
        x = q8(rng.normal(0, 1.5, (8, 8, 32)))
        base = q8(L.swin_block(x, p, (2, 2), 2, T.M_BLOCK_CONSTS))
        for key, val in (("tau", 0.5), ("cap", 8.0), ("eps_num", 1e-2), ("gelu", "sigmoid")):
            k = dict(T.M_BLOCK_CONSTS)
            k[key] = val
            self.assertFalse(_same(q8(L.swin_block(x, p, (2, 2), 2, k)), base), key)

    def test_block_slow_loop(self):
        """Vectorised block vs a per-window, per-token loop (float64; outputs agree to rounding noise)."""
        rng = np.random.default_rng(3)
        C, heads = 32, 2
        p = {k: np.asarray(v, np.float64) for k, v in L.random_block_params(rng, C, heads, 128).items()}
        H, W, off = 6, 7, (2, 2)
        x = q8(rng.normal(0, 1.5, (H, W, C)))
        got = L.swin_block(x, p, off, heads, T.M_BLOCK_CONSTS)
        k = T.M_BLOCK_CONSTS
        want = np.full_like(got, np.nan)
        rows, cols = L.window_grid(H, W, off, 4)
        for r in rows:
            for c in cols:
                toks = [(MS.mirror(rr, H), MS.mirror(cc, W)) for rr in r for cc in c]
                xw = np.stack([x[a, b] for a, b in toks])
                a = q8(np.stack([L.rmsnorm(xw[i], p["gamma1"], C, k["eps_num"]) for i in range(16)]))
                o = np.zeros((16, 32 * heads))
                for h in range(heads):
                    u = q8(np.array([[np.dot(p["Wqk"][h, d], a[i]) for d in range(32)] for i in range(16)]))
                    v = L.f16(np.array([[np.dot(p["Wv"][h, d], a[i]) for d in range(32)] for i in range(16)]))
                    for i in range(16):
                        S = np.array([np.dot(u[i], u[j]) for j in range(16)]) + p["B"][h, i]
                        z = k["tau"] * L.soft_cap(S, k["cap"])
                        e = np.exp(z - z.max())
                        A = L.f16(e / e.sum())
                        o[i, 32 * h:32 * h + 32] = A @ v
                x1 = xw + q8(o) @ p["Wp"].T + p["bp"]
                m = q8(np.stack([L.rmsnorm(x1[i], p["gamma2"], C, k["eps_num"]) for i in range(16)]))
                hdn = q8(L.gelu(m @ p["W1"].T + p["b1"], "exact"))
                y = x1 + hdn @ p["W2"].T + p["b2"]
                for i, (rr, cc) in enumerate((rr, cc) for rr in r for cc in c):
                    if 0 <= rr < H and 0 <= cc < W:
                        want[rr, cc] = y[i]
        self.assertFalse(np.isnan(want).any())
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


class EncoderTest(unittest.TestCase):
    def test_model_m_op_list_equals_model_m_encoder(self):
        rng = np.random.default_rng(4)
        t = T.table(enc_ops=M_ENC_OPS)
        for reset in (False, True):
            x = ME.random_inputs(rng, rw=37, rh=21, reset=reset)
            p = ME.random_params(rng)
            want = ME.encode(x, p)
            got = L.encode(x, {"enc0": p}, t=t)
            for k in ("features", "m_near", "m_far", "m_center"):
                self.assertTrue(_same(got[k], want[k]), (k, reset))

    def test_default_encoder(self):
        rng = np.random.default_rng(5)
        x = ME.random_inputs(rng, rw=40, rh=24)
        params = L.random_params(rng)
        for stage, shapes in T.param_shapes().items():
            self.assertEqual(set(params[stage]), set(shapes), stage)
            for name, shape in shapes.items():
                self.assertEqual(np.shape(params[stage][name]), shape, (stage, name))
        out = L.encode(x, params, keep=True)
        fw, fh = x["feature_size"]
        self.assertEqual(out["features"].shape, (fh // 2, fw // 2, 64))
        self.assertTrue(np.isfinite(out["features"]).all())
        self.assertTrue(is_fp8_exact(out["features"]))
        maps = out["maps"]
        self.assertEqual(maps["q"].shape, (fh, fw, 20))
        self.assertEqual(maps["x"].shape, (fh, fw, 32))
        self.assertEqual(maps["z"].shape, (fh // 2, fw // 2, 128))
        self.assertTrue(_same(L.encode(x, params)["features"], out["features"]))

    def test_parts_by_hand(self):
        """The default op list recomputed step by step."""
        rng = np.random.default_rng(6)
        x = ME.random_inputs(rng, rw=33, rh=17)
        params = L.random_params(rng)
        enc = params["enc0"]
        out = L.encode(x, params, parity=1, keep=True)
        m = out["maps"]
        fw, fh = x["feature_size"]
        q = ME.dither_fp8(ME.pixel_values(x, enc), ME.dither_table(enc["T"], x["frame"], fw, fh)[..., None])
        self.assertTrue(_same(m["q"], q))
        h = q8(np.maximum(L.f16(q @ np.asarray(enc["Win1"], np.float64).T + enc["bin1"]), 0))
        h = q8(L.f16(h @ np.asarray(enc["Win2"], np.float64).T + enc["bin2"]))
        h = q8(L.swin_block(h, params["enc0_block"], T.block_offset(T.TABLE, "first", 1), 2, T.M_BLOCK_CONSTS))
        self.assertTrue(_same(m["x"], h))
        z = np.zeros((fh // 2, fw // 2, 128))
        for r in range(fh // 2):
            for c in range(fw // 2):
                for p in range(4):
                    z[r, c, 32 * p:32 * p + 32] = h[2 * r + p // 2, 2 * c + p % 2]
        self.assertTrue(_same(m["z"], z))
        W = np.asarray(enc["W"], np.float64)
        s = np.broadcast_to(np.asarray(enc["b"], np.float64), z.shape[:-1] + (64,))
        for g in range(4):
            s = L.f16(s + z[..., 32 * g:32 * g + 32] @ W[:, 32 * g:32 * g + 32].T)
        self.assertTrue(_same(out["features"], q8(np.clip(s, -6.28125, 6.28125)).astype(np.float32)))

    def test_table_edits(self):
        """Provisional entries are table edits: no ReLU, no FP16 accumulation, plain FP8 quantization, no clamp."""
        rng = np.random.default_rng(7)
        x = ME.random_inputs(rng, rw=24, rh=20)
        base = T.TABLE["enc_ops"]
        variants = {
            "no_relu": (base[0], dict(base[1], out_round="q8")) + tuple(base[2:]),
            "no_f16": (base[0], dict(base[1], acc=None), dict(base[2], acc=None)) + tuple(base[3:]),
            "plain_q8": (dict(base[0], mode="q8"),) + tuple(base[1:]),
            "no_clamp": tuple(base[:-1]) + (dict(base[-1], clamp=None),),
        }
        params = L.random_params(np.random.default_rng(8))
        ref = L.encode(x, params)["features"]
        for name, ops in variants.items():
            t = T.table(enc_ops=ops)
            out = L.encode(x, params, t=t)
            self.assertTrue(is_fp8_exact(out["features"]), name)
            if name != "no_clamp":
                self.assertFalse(_same(out["features"], ref), name)
        self.assertNotIn("T", T.param_shapes(T.table(enc_ops=variants["plain_q8"]))["enc0"])

    def test_plan_refuses_bad_lists(self):
        base = T.TABLE["enc_ops"]
        bad = [
            base[:-1],                                                         # no project at the end
            (base[0], dict(base[1], n_in=32)) + tuple(base[2:]),               # 20 channels into a 32-input linear
            (base[0], {"op": "frobnicate", "src": "q"}, base[-1]),
            tuple(base[:4]) + (base[5],),                                      # project on a pixel-level map
        ]
        for ops in bad:
            with self.assertRaises(ValueError):
                T.encoder_plan(T.table(enc_ops=ops))


class DecoderTest(unittest.TestCase):
    def test_model_m_op_list_equals_model_m_decoder(self):
        """With Model M's projection rows and no block, the L decoder is Model M's decoder byte for byte."""
        rng = np.random.default_rng(9)
        t = T.table(dec_ops=M_DEC_OPS)
        for reset in (False, True):
            x = MD.random_inputs(rng, rw=29, rh=19, reset=reset)
            p = MD.random_params(rng)
            want = MD.decode(x, p)
            got = L.decode(x, {"dec0": m_dec_as_l(p)}, 0, t)
            for k in ("A", "B", "C", "latent", "codes_A", "codes_B", "codes_L"):
                self.assertTrue(_same(got[k], want[k]), (k, reset))

    def test_default_decoder(self):
        rng = np.random.default_rng(10)
        x = MD.random_inputs(rng, rw=30, rh=18)
        params = L.random_params(rng)
        out = L.decode(x, params, 1, keep=True)
        fw, fh = x["feature_size"]
        m = out["maps"]
        self.assertEqual(m["y"].shape, (fh // 2, fw // 2, 128))
        self.assertEqual(m["p"].shape, (fh, fw, 32))
        self.assertEqual(out["a"].shape, (18, 30, 17))
        y = L.dec_project(x["features"], params["dec0"]["W"], params["dec0"]["b"])
        p = q8(L.swin_block(L.depth_to_space(y), params["dec0_block"], T.block_offset(T.TABLE, "second", 1), 2,
                            T.M_BLOCK_CONSTS))
        self.assertTrue(_same(m["p"], p))
        self.assertTrue(_same(out["a"], p[:18, :30][..., list(T.DEC0_VALUES)].astype(np.float32)))
        for k in ("A", "B", "C", "latent"):
            self.assertTrue(np.isfinite(out[k]).all(), k)

    def test_depth_to_space_inverts_space_to_depth(self):
        v = np.random.default_rng(11).normal(size=(6, 8, 5))
        self.assertTrue(_same(L.depth_to_space(L.space_to_depth(v)), v))

    def test_pending(self):
        with self.assertRaises(NotImplementedError):
            L.reconstruct()
        rng = np.random.default_rng(12)
        x = MD.random_inputs(rng, rw=20, rh=12)
        with self.assertRaises(NotImplementedError):
            L.decode(x, L.random_params(rng), 0, T.table(dec_pixel="other"))


class NetworkTest(unittest.TestCase):
    def _inputs(self, rng, rw, rh):
        xe = ME.random_inputs(rng, rw=rw, rh=rh)
        xd = MD.random_inputs(rng, rw=rw, rh=rh, fw=xe["feature_size"][0], fh=xe["feature_size"][1],
                              dw=xe["recon_size"][0], dh=xe["recon_size"][1])
        del xd["features"]
        return xe, xd

    def test_reduces_to_model_m_chain(self):
        """With Model M's encoder and decoder op lists, the L network is Model M's oracle chain."""
        rng = np.random.default_rng(13)
        xe, xd = self._inputs(rng, 40, 24)
        t = T.table(enc_ops=M_ENC_OPS, dec_ops=M_DEC_OPS)
        p = L.random_params(rng, t)
        p["enc0"] = ME.random_params(rng)
        pm_dec = MD.random_params(rng)
        p["dec0"] = m_dec_as_l(pm_dec)
        for parity in (0, 1):
            got = L.network(xe, xd, p, parity, t)
            x = ME.encode(xe, p["enc0"])["features"]
            skips = {}
            for s in T.SHARED_STAGES:
                off = MS.window_offset(s, parity)
                kind = MS.STAGES[s]["kind"]
                if kind == "enc":
                    o = MS.run_stage(s, p[s], x, off)
                    skips[s], x = o["skip"], o["merged"]
                elif kind == "dec":
                    o = MS.run_stage(s, p[s], x, off, skips["enc2" if s == "dec2" else "enc1"])
                    x = o["out"]
                else:
                    x = MS.run_stage(s, p[s], x, off)["out"]
                key = "merged" if kind == "enc" else "out"
                self.assertTrue(_same(got[s][key], x), (s, parity))
            xx = dict(xd)
            xx["features"] = x
            want = MD.decode(xx, pm_dec)
            for k in ("A", "B", "C", "latent"):
                self.assertTrue(_same(got["dec0"][k], want[k]), (k, parity))

    def test_default_network_runs(self):
        rng = np.random.default_rng(14)
        xe, xd = self._inputs(rng, 36, 20)
        p = L.random_params(rng)
        out = L.network(xe, xd, p, 0)
        fw, fh = xe["feature_size"]
        self.assertEqual(out["dec1"]["out"].shape, (fh // 2, fw // 2, 64))
        for k in ("A", "B", "C", "latent"):
            self.assertTrue(np.isfinite(out["dec0"][k]).all(), k)


class TableTest(unittest.TestCase):
    def test_shared_stages_are_model_m(self):
        self.assertEqual(tuple(T.SHARED_STAGES), tuple(k for k in MS.STAGES))

    def test_param_shapes_are_the_maps(self):
        """Model L's own stages with the shapes of the preset-L converter map's records."""
        blk = {"gamma1": (32,), "gamma2": (32,), "Wqk": (2, 32, 32), "Wv": (2, 32, 32), "B": (2, 16, 16),
               "Wp": (32, 64), "bp": (32,), "W1": (128, 32), "b1": (128,), "W2": (32, 128), "b2": (32,)}
        want = {"enc0": {"W": (64, 128), "b": (64,), "T": (8, 16, 16), "sigma1": (), "sigma2": (),
                         "Win1": (32, 20), "bin1": (32,), "Win2": (32, 32), "bin2": (32,)},
                "enc0_block": blk, "dec0_block": blk,
                "dec0": {"W": (4, 32, 64), "b": (4, 32), "S": (), "T": ()}, "post": {"s": ()}}
        self.assertEqual(T.param_shapes(), want)

    def test_macs(self):
        self.assertEqual(T.block_macs_per_token(64, 2, 256), 47104)
        self.assertEqual(T.block_macs_per_token(96, 4, 384), 114688)
        self.assertEqual(T.block_macs_per_token(128, 4, 512), 184320)
        self.assertEqual(T.block_macs_per_token(32, 2, 128), 16384)
        r = T.frame_macs(T.TABLE, 1706, 960)
        self.assertEqual(r["M"], 103089438720)
        self.assertEqual(round(r["L"] / 1e9), 163)

    def test_table_copy(self):
        t = T.table(dec_pixel="x")
        self.assertEqual(t["dec_pixel"], "x")
        self.assertEqual(T.TABLE["dec_pixel"], "m")
        with self.assertRaises(KeyError):
            T.table(nonsense=1)


if __name__ == "__main__":
    unittest.main()
