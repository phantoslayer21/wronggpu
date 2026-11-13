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


def _block_by_windows(value_at, H, W, off, p, heads=2):
    """A block written window by window: value_at(y, x) gives any window member's input (inside or outside the
    grid); outputs are kept on the H x W grid. Float64, not rounded."""
    p = {k: np.asarray(v, np.float64) for k, v in p.items()}
    ox, oy = off
    out = None
    for y0 in range(-oy, H, 4):
        for x0 in range(-ox, W, 4):
            pos = [(y, x) for y in range(y0, y0 + 4) for x in range(x0, x0 + 4)]
            y = L._block_tokens(np.stack([value_at(a, b) for a, b in pos]), p, heads, T.M_BLOCK_CONSTS)
            if out is None:
                out = np.full((H, W, y.shape[-1]), np.nan)
            for i, (a, b) in enumerate(pos):
                if 0 <= a < H and 0 <= b < W:
                    out[a, b] = y[i]
    assert not np.isnan(out).any()
    return out


def _render_mirror(q, n):
    """The input encoder's position rule about the render size n (|q|, else 2 n - |q| - 2)."""
    q = abs(q)
    return q if q < n else 2 * n - q - 2


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
        h = T.block_halo(T.TABLE, "first")
        self.assertEqual(h, 2)
        self.assertEqual(maps["q"].shape, (fh + 2 * h, fw + 2 * h, 20))         # per-pixel ops run on the halo
        self.assertEqual(maps["x"].shape, (fh, fw, 32))
        self.assertEqual(maps["z"].shape, (fh // 2, fw // 2, 128))
        self.assertTrue(_same(L.encode(x, params)["features"], out["features"]))

    def test_parts_by_hand(self):
        """The default op list recomputed step by step; the block's out-of-grid members evaluate the per-pixel chain
        at the mirrored render position with the dither of the unmirrored position."""
        rng = np.random.default_rng(6)
        rw, rh = 37, 19          # small sizes: positions fw + 1 and fh + 1 still mirror inside R
        x = ME.random_inputs(rng, rw=rw, rh=rh)
        params = L.random_params(rng)
        enc = params["enc0"]
        fw, fh = x["feature_size"]
        vals = ME.pixel_values(x, enc)
        Tn = np.asarray(enc["T"])[int(x["frame"])]
        Win1, Win2 = np.asarray(enc["Win1"], np.float64), np.asarray(enc["Win2"], np.float64)

        def chain(yy, xx):
            v = vals[_render_mirror(yy, rh), _render_mirror(xx, rw)]
            q = ME.dither_fp8(v, Tn[abs(yy) % 16, abs(xx) % 16])
            h1 = q8(np.maximum(L.f16(q @ Win1.T + enc["bin1"]), 0))
            return L.f16(h1 @ Win2.T + enc["bin2"])                            # FP16, not FP8, into the block

        for parity in (0, 1):
            out = L.encode(x, params, parity=parity, keep=True)
            m = out["maps"]
            q = np.stack([np.stack([ME.dither_fp8(vals[_render_mirror(a, rh), _render_mirror(b, rw)],
                                                  Tn[abs(a) % 16, abs(b) % 16])
                                    for b in range(-2, fw + 2)]) for a in range(-2, fh + 2)])
            self.assertTrue(_same(m["q"], q))
            h = q8(_block_by_windows(chain, fh, fw, T.block_offset(T.TABLE, "first", parity), params["enc0_block"]))
            np.testing.assert_array_equal(m["x"], h)
        self.assertFalse(is_fp8_exact(chain(3, 4)))
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
            tuple(base[:3]) + (dict(base[3], border="wrap"),) + tuple(base[4:]),
            tuple(base[:3]) + ({"op": "space_to_depth", "src": "x", "dst": "z"},
                               dict(base[3], src="z", C=128, border="input")) + tuple(base[4:]),
        ]
        for ops in bad:
            with self.assertRaises(ValueError):
                T.encoder_plan(T.table(enc_ops=ops))


class DecoderTest(unittest.TestCase):
    def test_model_m_op_list_equals_model_m_decoder(self):
        """With Model M's projection rows and no block, the L decoder is Model M's decoder byte for byte."""
        rng = np.random.default_rng(9)
        t = T.table(dec_ops=M_DEC_OPS, dec_pixel="m")
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
        self.assertEqual(out["a"].shape, (18, 30, 13))
        y = L.dec_project(x["features"], params["dec0"]["W"], params["dec0"]["b"])
        hw = fw // 2

        def member(yy, xx):                     # coarse mirror of the token, sub-pixel kept (rows stay inside)
            q = xx // 2
            if not 0 <= q < hw:
                q = min(abs(q), fw - abs(q) - 2)
            return y[yy // 2, q, 32 * ((xx % 2) + 2 * (yy % 2)):][:32]

        self.assertTrue(_same(member(0, -1), member(0, 3)) and _same(member(1, fw), member(1, fw - 4)))
        p = L.f16(_block_by_windows(member, fh, fw, T.block_offset(T.TABLE, "second", 1), params["dec0_block"]))
        np.testing.assert_array_equal(m["p"], p)                                 # FP16 block output, no FP8
        self.assertFalse(is_fp8_exact(p))
        fine = L.f16(L.swin_block(L.depth_to_space(y), params["dec0_block"], T.block_offset(T.TABLE, "second", 1),
                                  2, T.M_BLOCK_CONSTS))
        self.assertFalse(_same(fine, p))                                         # not a fine-pixel mirror
        self.assertTrue(_same(fine[:, 2:-2], p[:, 2:-2]))                        # ... only the edge windows differ
        a = p[:18, :30][..., [0, 1, 6, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5]].astype(np.float32)
        self.assertTrue(_same(out["a"], a))
        # The per-pixel functions value by value.
        S, Tc = np.float32(params["dec0"]["S"]), np.float32(params["dec0"]["T"])
        for (r, c) in ((0, 0), (5, 7), (17, 29)):
            v = a[r, c]
            sig = [np.float16(np.float32(1) / (np.float32(1) + np.exp(-v[i]))) for i in range(3)]
            self.assertEqual(list(out["codes_A"][r, c]), [int(np.rint(np.float32(s) * np.float32(255))) for s in sig])
            e = np.exp(v[3:6] - v[3:6].max())
            self.assertTrue(np.allclose(out["B"][r, c], e / e.sum(), atol=0.5 / 255 + 1e-6))
            g = [np.float32(np.float16(np.float32(1) / (np.float32(1) + np.exp(-np.clip(v[i], -20, 20)))))
                 for i in (6, 7)]
            sx, sy = S * g[0], S * g[1]
            rho = np.float32(np.tanh(np.float64(Tc * v[8])))
            want_c = [np.float16(sx * sx), np.float16(sy * sy), np.float16((sx * sy) * rho)]
            self.assertEqual([np.float16(q) for q in out["C"][r, c]], want_c)
            want_l = [int(np.rint(np.clip(np.float32(np.tanh(np.float64(v[9 + i]))), -1, 1) * np.float32(127)))
                      for i in range(4)]
            self.assertEqual(list(out["codes_L"][r, c]), want_l)
        # No previous latent, motion or jitter is read.
        x2 = dict(x, latent_prev=-np.asarray(x["latent_prev"]), motion=np.asarray(x["motion"]) + 5.0,
                  jitter=(0.3, -0.2), reset=not x["reset"])
        out2 = L.decode(x2, params, 1)
        for k in ("A", "B", "C", "latent"):
            self.assertTrue(np.isfinite(out[k]).all(), k)
            self.assertTrue(_same(out[k], out2[k]), k)
        with self.assertRaises(ValueError):                                      # 13 values for "l", 17 for "m"
            L.decode(x, params, 1, T.table(dec_pixel="m"))

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
        t = T.table(enc_ops=M_ENC_OPS, dec_ops=M_DEC_OPS, dec_pixel="m")
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
        self.assertEqual(T.TABLE["dec_pixel"], "l")
        with self.assertRaises(KeyError):
            T.table(nonsense=1)


if __name__ == "__main__":
    unittest.main()
