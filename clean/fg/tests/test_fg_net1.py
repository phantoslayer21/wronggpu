# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): net1 with the delivered wiring and arithmetic (clean/fg/reference/net1.py), on synthetic
weights and inputs, against versions written separately here: scalar loops for the upscale rule, exact rationals for
one convolution, and a hand-composed network (float64 chunk sums, which are exact for this data, numpy max for the
pool, the upscale tables from the scalar rule).

    python -m unittest clean.fg.tests.test_fg_net1
"""
import itertools
import os
import struct
import tempfile
import unittest
from fractions import Fraction

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import forward as F
from clean.fg.reference import net1 as N
from clean.fg.reference import ops
from clean.fg.tests.test_fg_fp16acc import rn16_exact

S = 0.0099945068359375   # the delivered slope (float16 0x211E)


def f16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def synthetic_params(rng, scale=0.05):
    return {name: (rng.uniform(-scale, scale, ops.weight_shape(l)).astype(np.float16),
                   rng.uniform(-0.1, 0.1, l.out).astype(np.float16)) for name, l in F.LAYERS.items()}


# ---- scalar / independent versions ----------------------------------------------------------------------------------

def axis_scalar(n_in, n_out):
    """The align-corners rule of part 3 for one axis, one index at a time with numpy float32 scalars."""
    r = np.float32(n_in - 1) / np.float32(n_out - 1) if n_out > 1 else np.float32(0.0)
    rows = []
    for o in range(n_out):
        s = np.float32(r * np.float32(o))
        x0 = int(s)                                     # trunc (s >= 0)
        x1 = int(np.float32(s + np.float32(1.0)))
        if x1 >= n_in:
            x0 = x1 = n_in - 1
        w1 = np.float16(np.float32(s - np.float32(x0)))
        w0 = np.float16(np.float32(np.float32(np.float32(x0) - s) + np.float32(1.0)))
        rows.append((x0, x1, float(w0), float(w1)))
    return rows


def upscale_scalar(p, wo, ho):
    c, hi, wi = p.shape
    ax, ay = axis_scalar(wi, wo), axis_scalar(hi, ho)
    out = np.zeros((c, ho, wo))
    for ch in range(c):
        for y, (y0, y1, wy0, wy1) in enumerate(ay):
            for x, (x0, x1, wx0, wx1) in enumerate(ax):
                top = np.float16(np.float16(wx0 * p[ch, y0, x0]) + np.float16(wx1 * p[ch, y0, x1]))
                bot = np.float16(np.float16(wx0 * p[ch, y1, x0]) + np.float16(wx1 * p[ch, y1, x1]))
                out[ch, y, x] = np.float16(np.float16(wy0 * float(top)) + np.float16(wy1 * float(bot)))
    return out


def chunk_list(order, k, nb):
    if order == "ascending":
        return [(0, 0, cb) for cb in range(nb)]
    if order == "block_kx_ky":
        return [(ky, kx, cb) for cb in range(nb) for kx in range(k) for ky in range(k)]
    return [(ky, kx, cb) for kx in range(k) for cb in range(nb) for ky in range(k)]


def conv_exact(x, w, b, order, act):
    """One convolution, element by element, with exact rational chunk sums and nearest-even float16 rounding."""
    cout, k, _, cin = w.shape
    _, H, W = x.shape
    p = k // 2
    out = np.zeros((cout, H, W))
    for o in range(cout):
        for yy in range(H):
            for xx in range(W):
                acc = float(b[o])
                for ky, kx, cb in chunk_list(order, k, cin // 16):
                    sy, sx = yy + ky - p, xx + kx - p
                    s = Fraction(acc)
                    if 0 <= sy < H and 0 <= sx < W:
                        for c in range(16 * cb, 16 * cb + 16):
                            s += Fraction(float(w[o, ky, kx, c])) * Fraction(float(x[c, sy, sx]))
                    acc = rn16_exact(s)
                if act == "leaky":
                    acc = max(acc, float(np.float16(S * acc)))
                out[o, yy, xx] = acc
    return out


def conv_hand(x, w, b, order, act):
    """Hand-composed convolution: per chunk a float64 sum (exact for small synthetic data), then float16."""
    cout, k, _, cin = w.shape
    _, H, W = x.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    acc = np.broadcast_to(np.asarray(b, dtype=np.float64)[:, None, None], (cout, H, W)).astype(np.float16)
    for ky, kx, cb in chunk_list(order, k, cin // 16):
        part = np.einsum("oc,chw->ohw", np.asarray(w[:, ky, kx, 16 * cb:16 * cb + 16], dtype=np.float64),
                         xp[16 * cb:16 * cb + 16, ky:ky + H, kx:kx + W])
        acc = (acc.astype(np.float64) + part).astype(np.float16)
    acc = acc.astype(np.float64)
    if act == "leaky":
        acc = np.maximum(acc, (np.float16(S) * acc.astype(np.float16)).astype(np.float64))
    return acc


def net1_hand(values, params):
    """The 32 steps written out by hand."""
    gh, gw = values.shape[1:]
    H0, W0 = (gh + 1) // 2, (gw + 1) // 2
    pad = np.zeros((10, 2 * H0, 2 * W0))
    pad[:, :gh, :gw] = f16(values)
    cells = np.zeros((48, H0, W0))
    for v in range(10):
        for dy in range(2):
            for dx in range(2):
                cells[4 * v + 2 * dy + dx] = pad[v, dy::2, dx::2]

    def C(name, x, act="leaky"):
        w, b = params[name]
        order = {"net1_01": "block_kx_ky", "net1_03": "block_kx_ky", "net1_04": "block_kx_ky",
                 "net1_05": "kx_block_ky"}.get(name, "ascending")
        return conv_hand(x, w, b, order, act)

    def P(x):
        c, h, w = x.shape
        return x[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))

    def U(x, like):
        c, hi, wi = x.shape
        ax, ay = axis_scalar(wi, like.shape[2]), axis_scalar(hi, like.shape[1])
        x0 = [r[0] for r in ax]; x1 = [r[1] for r in ax]
        wx0 = np.array([r[2] for r in ax]); wx1 = np.array([r[3] for r in ax])
        y0 = [r[0] for r in ay]; y1 = [r[1] for r in ay]
        wy0 = np.array([r[2] for r in ay])[None, :, None]; wy1 = np.array([r[3] for r in ay])[None, :, None]
        t = f16(f16(wx0 * x[:, y0][:, :, x0]) + f16(wx1 * x[:, y0][:, :, x1]))
        bt = f16(f16(wx0 * x[:, y1][:, :, x0]) + f16(wx1 * x[:, y1][:, :, x1]))
        return f16(f16(wy0 * t) + f16(wy1 * bt))

    def A(a, b):
        return f16(a + b)

    a0 = C("net1_01", cells)
    a1 = C("net1_02", P(a0))
    a2 = C("net1_03", P(a1))
    a3 = C("net1_04", P(a2))
    a4 = C("net1_05", P(a3))
    d = C("net1_07", C("net1_06", P(a4)))
    d = C("net1_09", C("net1_08", A(U(d, a4), a4)))
    d = C("net1_11", C("net1_10", A(U(d, a3), a3)))
    d = C("net1_13", C("net1_12", A(U(d, a2), a2)))
    d = C("net1_15", C("net1_14", A(U(d, a1), a1)))
    out = C("net1_17", C("net1_16", A(U(d, a0), a0)), act="none")
    px = np.zeros((4, 2 * H0, 2 * W0))
    for dy in range(2):
        for dx in range(2):
            for t in range(4):
                px[t, dy::2, dx::2] = out[4 * t + 2 * dy + dx]
    return px[:, :gh, :gw]


# ---- tests ----------------------------------------------------------------------------------------------------------

class Operations(unittest.TestCase):
    def test_leaky(self):
        self.assertEqual(N.LEAKY_SLOPE, 0.0099945068359375)
        self.assertEqual(int(np.float16(N.LEAKY_SLOPE).view(np.uint16)), 0x211E)
        x = f16(np.array([-65504, -100, -1, -2 ** -24, -0.0, 0, 2 ** -24, 1, 65504, np.inf, -np.inf]))
        want = [max(v, float(np.float16(S * v))) for v in x]
        np.testing.assert_array_equal(N.leaky(x), want)
        self.assertEqual(N.leaky(np.array([-1.0]))[0], float(np.float16(-S)))

    def test_maxpool_finite_and_floor(self):
        rng = np.random.default_rng(1)
        x = f16(rng.uniform(-5, 5, (3, 7, 9)))
        got = N.maxpool2(x)
        self.assertEqual(got.shape, (3, 3, 4))
        np.testing.assert_array_equal(got, x[:, :6, :8].reshape(3, 3, 2, 4, 2).max(axis=(2, 4)))

    def test_maxpool_specials_do_not_depend_on_tap_order(self):
        vals = [1.0, -2.0, np.inf, -np.inf, np.nan, 0.0, 65504.0, -65504.0]
        got_any = []
        for window in itertools.product(vals, repeat=4):
            x = np.array(window, dtype=np.float64).reshape(1, 2, 2)
            got = N.maxpool2(x)[0, 0, 0]
            results = set()
            for perm in itertools.permutations(window):
                m = perm[0]
                with np.errstate(invalid="ignore"):
                    for v in perm[1:]:
                        m = float(np.float16((v > m) * v + (v <= m) * m))
                results.add("nan" if np.isnan(m) else m)
            self.assertEqual(len(results), 1, window)
            want = results.pop()
            self.assertTrue((want == "nan" and np.isnan(got)) or got == want, (window, got, want))
            got_any.append(got)
        self.assertTrue(any(np.isnan(g) for g in got_any))

    def test_upscale_against_scalar_rule(self):
        rng = np.random.default_rng(2)
        for (hi, wi), (ho, wo) in (((1, 1), (2, 2)), ((2, 2), (4, 5)), ((3, 4), (7, 9)), ((5, 2), (11, 4)),
                                   ((1, 3), (1, 7)), ((4, 4), (8, 8)), ((2, 3), (2, 3))):
            p = f16(rng.uniform(-3, 3, (2, hi, wi)))
            np.testing.assert_array_equal(N.upscale_to(p, wo, ho), upscale_scalar(p, wo, ho),
                                          err_msg="%s -> %s" % ((hi, wi), (ho, wo)))

    def test_upscale_corners_and_extent_one(self):
        p = f16(np.array([[[1.0, 2.0], [3.0, 4.0]]]))
        y = N.upscale_to(p, 5, 5)
        self.assertEqual((y[0, 0, 0], y[0, 0, -1], y[0, -1, 0], y[0, -1, -1]), (1.0, 2.0, 3.0, 4.0))
        np.testing.assert_array_equal(N.upscale_to(p, 1, 1), [[[1.0]]])     # ratio 0: the first sample

    def test_conv_against_exact_rationals(self):
        rng = np.random.default_rng(3)
        for k, cin, order in ((3, 32, "block_kx_ky"), (3, 32, "kx_block_ky"), (1, 48, "ascending")):
            x = f16(rng.uniform(-2, 2, (cin, 3, 4)))
            w = f16(rng.uniform(-1, 1, (2, k, k, cin)))
            b = f16(rng.uniform(-1, 1, 2))
            for act in ("leaky", "none"):
                got = N.conv16(x, w, b, chunks=order, act=act)
                np.testing.assert_array_equal(got, conv_exact(x, w, b, order, act), err_msg="%s %s %s" % (k, order, act))

    def test_chunk_order_is_honoured(self):
        # a large partial sum that later cancels: the float16 rounding of intermediate sums depends on the order
        x = np.zeros((32, 1, 1)); w = np.zeros((1, 3, 3, 32))
        x[0, 0, 0] = 1.0; x[16, 0, 0] = 1.0
        w[0, 1, 1, 0] = 2048.0; w[0, 1, 1, 16] = 1.0          # centre tap, both channel blocks
        b = np.array([1.0])
        a = N.conv16(x, w, b, chunks="block_kx_ky", act="none")   # 1 + 2048 = 2049 -> 2048, then + 1 -> 2049 -> 2048
        self.assertEqual(a[0, 0, 0], 2048.0)
        w2 = np.zeros((1, 3, 3, 32)); w2[0, 1, 1, 16] = 2048.0; w2[0, 1, 1, 0] = -2048.0
        self.assertEqual(N.conv16(x, w2, b, chunks="kx_block_ky", act="none")[0, 0, 0], 1.0)
        with self.assertRaises(ValueError):
            N.conv16(x, w, b, chunks="ascending", act="none")       # 3 x 3 has no ascending order
        with self.assertRaises(ValueError):
            N.conv16(x, w, b, chunks="block_kx_ky", act="relu")

    def test_zero_border(self):
        x = np.ones((16, 2, 2)); w = np.ones((1, 3, 3, 16)) / 16; b = np.zeros(1)
        y = N.conv16(x, f16(w), b, chunks="block_kx_ky", act="none")
        np.testing.assert_array_equal(y[0], [[4.0, 4.0], [4.0, 4.0]])     # a 2 x 2 grid: 4 taps inside per output


class FoldAndWiring(unittest.TestCase):
    def test_fold_unfold(self):
        rng = np.random.default_rng(4)
        v = f16(rng.uniform(0, 1, (10, 5, 7)))
        cells = N.fold(v)
        self.assertEqual(cells.shape, (48, 3, 4))
        self.assertEqual(cells[4 * 6 + 2 * 1 + 0, 1, 2], v[6, 3, 4])      # value 6 of pixel (x=4, y=3): dx 0, dy 1
        np.testing.assert_array_equal(cells[40:], 0.0)
        self.assertEqual(cells[4 * 2 + 1, 2, 3], 0.0)                       # pixel (7, 4) is outside the 7 x 5 input
        out = np.zeros((16, 3, 4)); out[4 * 3 + 2 * 1 + 1] = 5.0
        px = N.unfold(out, 5, 7)
        self.assertEqual(px.shape, (4, 5, 7))
        self.assertEqual(px[3, 1, 1], 5.0)
        self.assertEqual(px[3, 0, 0], 0.0)

    def test_wiring_data(self):
        w = N.NET1_WIRING
        self.assertEqual(len(w), 32)
        self.assertEqual([s["op"] for s in w].count("conv"), 17)
        self.assertEqual([s["op"] for s in w].count("pool"), 5)
        self.assertEqual([s["op"] for s in w].count("upscale"), 5)
        self.assertEqual([s["op"] for s in w].count("add"), 5)
        self.assertTrue(N.check_wiring_layers())
        acts = [s["act"] for s in w if s["op"] == "conv"]
        self.assertEqual(acts, ["leaky"] * 16 + ["none"])
        chunks = {s["layer"]: s["chunks"] for s in w if s["op"] == "conv"}
        self.assertEqual({k for k, v in chunks.items() if v != "ascending"}, {"net1_01", "net1_03", "net1_04", "net1_05"})
        self.assertEqual(chunks["net1_05"], "kx_block_ky")
        self.assertEqual(w[-1]["out"], "out")

    def test_level_sizes(self):
        rng = np.random.default_rng(5)
        params = synthetic_params(rng)
        buf = N.run(N.fold(f16(rng.uniform(0, 1, (10, 70, 133)))), params)
        sizes = [buf["a%d" % i].shape[1:] for i in range(5)] + [buf["d5"].shape[1:]]
        self.assertEqual(sizes, [(35, 67), (17, 33), (8, 16), (4, 8), (2, 4), (1, 2)])
        self.assertEqual(buf["out"].shape, (16, 35, 67))
        with self.assertRaises(ValueError):
            N.forward(np.zeros((10, 63, 64)), params)


class EndToEnd(unittest.TestCase):
    def test_against_hand_composition(self):
        rng = np.random.default_rng(6)
        params = synthetic_params(rng)
        for gh, gw in ((64, 64), (66, 71)):
            v = f16(rng.uniform(0, 1, (10, gh, gw)))
            got = F.net1_forward(v, params)
            self.assertEqual(got.shape, (4, gh, gw))
            np.testing.assert_array_equal(got, net1_hand(v, params), err_msg="%dx%d" % (gh, gw))
            self.assertTrue(np.all(np.isfinite(got)))


def write_model_file(path, params, m):
    """A minimal model.wgcm (container version 1) holding stages dl1.lNN / dl2.lNN with W (natural map shape) and b,
    written here from the format description; digests and key are zeros (the reader does not check them)."""
    from clean.fg.reference import natural as Nat
    stages = []
    for name, l in F.LAYERS.items():
        w, b = params[name]
        nat = w.transpose(0, 3, 1, 2) if w.shape[1] == 3 else w[:, 0, 0, :]
        assert nat.shape == Nat.weight_shape(l, m)
        stages.append((Nat.stage_name(l), [("W", nat.astype("<f4")), ("b", np.asarray(b).astype("<f4"))]))
    index = bytearray()
    blobs = []
    for sname, tensors in stages:
        index += struct.pack("<H", len(sname)) + sname.encode() + bytes(32) + struct.pack("<I", len(tensors))
        for tname, arr in tensors:
            index += struct.pack("<H", len(tname)) + tname.encode() + bytes([0, arr.ndim])
            index += struct.pack("<%dI" % arr.ndim, *arr.shape)
            index += b"OFFSET__" + struct.pack("<Q", arr.nbytes)
            blobs.append(arr.tobytes())
    head = b"WGCMODEL" + struct.pack("<II", 1, len(stages)) + bytes(64) + struct.pack("<Q", len(index))
    pos = len(head) + len(index)
    data = bytearray()
    offsets = []
    for blob in blobs:
        pad = (-(pos + len(data))) % 64
        data += bytes(pad)
        offsets.append(pos + len(data))
        data += blob
    for off in offsets:
        i = index.index(b"OFFSET__")
        index[i:i + 8] = struct.pack("<Q", off)
    body = head + bytes(index) + bytes(data)
    import hashlib
    with open(path, "wb") as f:
        f.write(body + hashlib.sha256(body).digest())


class ModelFile(unittest.TestCase):
    def test_params_from_model_file_and_digest_tool(self):
        from clean.fg.reference import natural as Nat
        from clean.fg.tools import net1_digest
        m = Nat.load_map()
        params = synthetic_params(np.random.default_rng(7))
        fd, path = tempfile.mkstemp(suffix=".wgcm")
        os.close(fd)
        try:
            write_model_file(path, params, m)
            got = F.params_from_model_file(path, m)
            for name in F.LAYERS:
                np.testing.assert_array_equal(got[name][0].view("<u2"), params[name][0].view("<u2"))
                np.testing.assert_array_equal(got[name][1].view("<u2"), params[name][1].view("<u2"))
            rec = net1_digest.digest(path)
            self.assertEqual(rec["output_shape"], [4, net1_digest.SIZE, net1_digest.SIZE])
            self.assertEqual(rec["nonfinite"], 0)
            self.assertEqual(len(rec["output_sha256"]), 64)
            self.assertEqual(rec, net1_digest.digest(path))                    # deterministic
            with open(path, "r+b") as f:                                       # a damaged file is refused
                f.seek(200)
                f.write(b"\xff")
            with self.assertRaises(ValueError):
                F.read_model_file(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
