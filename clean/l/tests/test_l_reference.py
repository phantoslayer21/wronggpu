# SPDX-License-Identifier: MIT
"""Model L oracle self-checks (no GPU, no compiler; synthetic data only).

Run: python -m unittest discover -s clean/l/tests -t .

The strongest checks reduce Model L to Model M through the table and require the L oracle to give Model M's oracle
results byte for byte: the block with M's constants, the encoder with M's op list, the decoder without a block, and
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
    {"op": "project", "src": "z", "w": "W", "b": "b", "groups": (20, 20, 20, 20), "n_out": 64},
)


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


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
            got = L.encode(x, p, t=t)
            for k in ("features", "m_near", "m_far", "m_center"):
                self.assertTrue(_same(got[k], want[k]), (k, reset))

    def test_default_encoder(self):
        rng = np.random.default_rng(5)
        x = ME.random_inputs(rng, rw=40, rh=24)
        p = L.random_params(rng)["enc0"]
        self.assertEqual(set(p), set(T.encoder_shapes()))
        for name, shape in T.encoder_shapes().items():
            self.assertEqual(np.shape(p[name]), shape, name)
        out = L.encode(x, p, keep=True)
        fw, fh = x["feature_size"]
        self.assertEqual(out["features"].shape, (fh // 2, fw // 2, 64))
        self.assertTrue(np.isfinite(out["features"]).all())
        self.assertTrue(is_fp8_exact(out["features"]))
        maps = out["maps"]
        self.assertEqual(maps["raw"].shape, (fh // 2, fw // 2, 80))
        self.assertEqual(maps["x"].shape, (fh // 2, fw // 2, 32))
        self.assertEqual(maps["z"].shape, (fh // 2, fw // 2, 128))
        self.assertTrue(np.all(maps["z"][..., 112:] == 0))
        again = L.encode(x, p)
        self.assertTrue(_same(again["features"], out["features"]))

    def test_parts_by_hand(self):
        """linear, select, concat and project of the default op list recomputed directly."""
        rng = np.random.default_rng(6)
        x = ME.random_inputs(rng, rw=33, rh=17)
        p = L.random_params(rng)["enc0"]
        out = L.encode(x, p, parity=1, keep=True)
        m = out["maps"]
        fw, fh = x["feature_size"]
        q = ME.dither_fp8(ME.pixel_values(x, p), ME.dither_table(p["T"], x["frame"], fw, fh)[..., None])
        self.assertTrue(_same(m["q"], q))
        raw = np.zeros((fh // 2, fw // 2, 80))
        for r in range(fh // 2):
            for c in range(fw // 2):
                for pp in range(4):
                    raw[r, c, 20 * pp:20 * pp + 20] = q[2 * r + pp // 2, 2 * c + pp % 2]
        self.assertTrue(_same(m["raw"], raw))
        h = q8(raw[..., :20] @ np.asarray(p["Wa"], np.float64).T + p["ba"])
        h = q8(h @ np.asarray(p["Wb"], np.float64).T + p["bb"])
        bp = {k[4:]: v for k, v in p.items() if k.startswith("blk.")}
        h = q8(L.swin_block(h, bp, T.block_offset(T.TABLE, "first", 1), 2, T.M_BLOCK_CONSTS))
        self.assertTrue(_same(m["x"], h))
        z = np.concatenate([raw, h, np.zeros(h.shape[:-1] + (16,))], axis=-1)
        W = np.asarray(p["W"], np.float64)
        s = np.broadcast_to(np.asarray(p["b"], np.float64), z.shape[:-1] + (64,))
        for g in range(4):
            s = L.f16(s + z[..., 32 * g:32 * g + 32] @ W[:, 32 * g:32 * g + 32].T)
        self.assertTrue(_same(out["features"], q8(np.clip(s, -6.28125, 6.28125)).astype(np.float32)))

    def test_table_edits(self):
        """Provisional entries are table edits: other op orders, a pixel-level pre-network, plain q8 inputs."""
        rng = np.random.default_rng(7)
        x = ME.random_inputs(rng, rw=24, rh=20)
        base = T.TABLE["enc_ops"]
        variants = {
            "block_first": (base[0], base[1], base[2],
                            {**base[5], "C": 20, "heads": 1, "mlp": 80, "prefix": "blk."},
                            base[3], base[4], base[6], base[7]),
            "pixel_level": (
                base[0],
                {**base[3], "src": "q"}, base[4], base[5],
                {"op": "space_to_depth", "src": "x", "dst": "z"},
                {"op": "project", "src": "z", "w": "W", "b": "b", "groups": (32, 32, 32, 32), "n_out": 64}),
            "plain_q8": ({"op": "quant", "mode": "q8", "src": "values", "dst": "q"},) + tuple(base[1:]),
        }
        feats = {}
        for name, ops in variants.items():
            t = T.table(enc_ops=ops)
            plan = T.encoder_plan(t)
            p = L.random_params(np.random.default_rng(8), t)["enc0"]
            self.assertEqual(set(p), set(T.encoder_shapes(t)), name)
            out = L.encode(x, p, t=t)
            self.assertTrue(is_fp8_exact(out["features"]), name)
            feats[name] = out["features"]
            self.assertEqual(plan[-1], ("project", "token", 64))
        self.assertNotIn("T", T.encoder_shapes(T.table(enc_ops=variants["plain_q8"])))
        self.assertEqual(T.encoder_plan(T.table(enc_ops=variants["pixel_level"]))[1], ("linear", "pixel", 32))

    def test_plan_refuses_bad_lists(self):
        base = T.TABLE["enc_ops"]
        bad = [
            base[:-1],                                                     # no project at the end
            (base[0], {**base[3], "src": "q", "n_in": 32}) + tuple(base[4:]),   # 20 channels into a 32-input linear
            (base[0], base[1], {**base[2], "hi": 99}) + tuple(base[3:]),   # select outside the map
            (base[0], {"op": "frobnicate", "src": "q"}, base[-1]),
        ]
        for ops in bad:
            with self.assertRaises(ValueError):
                T.encoder_plan(T.table(enc_ops=ops))


class DecoderTest(unittest.TestCase):
    def test_no_block_equals_model_m_decoder(self):
        rng = np.random.default_rng(9)
        t = T.table(dec_connect=(("pass", 0, 64),))
        for reset in (False, True):
            x = MD.random_inputs(rng, rw=29, rh=19, reset=reset)
            p = MD.random_params(rng)
            want = MD.decode(x, p)
            got = L.decode(x, p, 0, t)
            for k in ("A", "B", "C", "latent", "codes_A", "codes_B", "codes_L", "h"):
                self.assertTrue(_same(got[k], want[k]), (k, reset))

    def test_default_decoder(self):
        rng = np.random.default_rng(10)
        x = MD.random_inputs(rng, rw=30, rh=18)
        p = L.random_params(rng)["dec0"]
        self.assertEqual(set(p), set(T.decoder_shapes()))
        out = L.decode(x, p, 1)
        front = out["front"]
        self.assertTrue(_same(front[..., 32:], x["features"][..., 32:]))
        bp = {k[4:]: v for k, v in p.items() if k.startswith("blk.")}
        y = q8(L.swin_block(x["features"][..., :32], bp, T.block_offset(T.TABLE, "second", 1), 2,
                            T.M_BLOCK_CONSTS)).astype(np.float32)
        self.assertTrue(_same(front[..., :32], y))
        xx = dict(x)
        xx["features"] = front
        want = MD.decode(xx, {k: p[k] for k in ("W", "b", "S", "T")})
        self.assertTrue(_same(out["A"], want["A"]))
        self.assertTrue(_same(out["latent"], want["latent"]))

    def test_other_segments(self):
        rng = np.random.default_rng(11)
        x = MD.random_inputs(rng, rw=20, rh=12)
        p = L.random_params(rng)["dec0"]
        t = T.table(dec_connect=(("pass", 0, 32), ("block", 32, 64)))
        out = L.decode(x, p, 0, t)
        self.assertTrue(_same(out["front"][..., :32], x["features"][..., :32]))
        with self.assertRaises(ValueError):
            L.decode(x, p, 0, T.table(dec_connect=(("block", 0, 48), ("pass", 48, 64))))
        with self.assertRaises(NotImplementedError):
            L.decode(x, p, 0, T.table(dec_block_level="pixel"))
        with self.assertRaises(NotImplementedError):
            L.reconstruct()


class NetworkTest(unittest.TestCase):
    def _inputs(self, rng, rw, rh):
        xe = ME.random_inputs(rng, rw=rw, rh=rh)
        xd = MD.random_inputs(rng, rw=rw, rh=rh, fw=xe["feature_size"][0], fh=xe["feature_size"][1],
                              dw=xe["recon_size"][0], dh=xe["recon_size"][1])
        del xd["features"]
        return xe, xd

    def test_reduces_to_model_m_chain(self):
        """With M's encoder op list and no decoder block, the L network is Model M's oracle chain."""
        rng = np.random.default_rng(12)
        xe, xd = self._inputs(rng, 40, 24)
        t = T.table(enc_ops=M_ENC_OPS, dec_connect=(("pass", 0, 64),))
        p = L.random_params(rng, t)
        p["enc0"] = ME.random_params(rng)
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
            want = MD.decode(xx, p["dec0"])
            for k in ("A", "B", "C", "latent"):
                self.assertTrue(_same(got["dec0"][k], want[k]), (k, parity))

    def test_default_network_runs(self):
        rng = np.random.default_rng(13)
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

    def test_macs(self):
        self.assertEqual(T.block_macs_per_token(64, 2, 256), 47104)
        self.assertEqual(T.block_macs_per_token(96, 4, 384), 114688)
        self.assertEqual(T.block_macs_per_token(128, 4, 512), 184320)
        self.assertEqual(T.block_macs_per_token(32, 2, 128), 16384)
        r = T.frame_macs(T.TABLE, 1706, 960)
        self.assertEqual(r["M"], 103089438720)
        self.assertGreater(r["L"], r["M"])

    def test_provisional_keys_cover_entries(self):
        self.assertEqual(set(T.PROVISIONAL), {f"I{i}" for i in range(1, 11)})

    def test_table_copy(self):
        t = T.table(recon_ratio=(3, 1))
        self.assertEqual(t["recon_ratio"], (3, 1))
        self.assertEqual(T.TABLE["recon_ratio"], (3, 2))
        with self.assertRaises(KeyError):
            T.table(nonsense=1)


if __name__ == "__main__":
    unittest.main()
