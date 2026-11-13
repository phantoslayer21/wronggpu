# SPDX-License-Identifier: MIT
"""Page 54b reference tests: vectorized NumPy reference vs the independent slow oracle (bit patterns), and the
exact rules (variant choice, layer selection, creation resets, half-way jitter, history copies, stores).

All symbols and data are invented for testing; none is a model constant.
"""
import math
import unittest

import numpy as np

from clean.rr_f.reference import f32x as fx
from clean.rr_f.reference import resolve as rr
from clean.rr_f.tests import slow_resolve as slow


def symbols(**kw):
    v = dict(RW_A=-0.0625, RW_B=-0.03125, RW_C=0.375, RW_D=0.125, RW_0=0.625, K1=0.0625, K2n=-1.25,
             K2x2n=-0.5, K4=4, K5=0, K7=1.5, K8=0.75, K10=6, K11=0.5, K12=0.25, K13=2.0 ** -20, C_HDEC=1.5,
             T_EFL=2.0 ** -10, TBW=0.3125)
    v.update(kw)
    return rr.ResolveSymbols(**v)


def bits32(v):
    v = np.asarray(v, np.float32)
    b = v.view(np.uint32).astype(np.int64)
    return np.where(np.isnan(v), -1, b)


def same32(a, b):
    return np.array_equal(bits32(a), bits32(b))


def rand_f32(rng, n, lo=-150, hi=130):
    """Random f32 values over many binades, plus specials and subnormals."""
    e = rng.uniform(lo, hi, n)
    v = (np.sign(rng.uniform(-1, 1, n)) * np.exp2(e)).astype(np.float32)
    extra = np.array([0, -0.0, np.inf, -np.inf, np.nan, 2.0 ** -149, -(2.0 ** -130), 1, -1, 0.5, 2.0 ** -126,
                      np.float32(3.4028235e38)], np.float32)
    return np.concatenate([v, extra])


class F32xTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(54)

    def check(self, fast, ref, *args):
        got = fast(*[np.asarray(a, np.float32) for a in args])
        for idx in range(len(args[0])):
            want = ref(*[float(a[idx]) for a in args])
            self.assertTrue(same32(got[idx], np.float32(want)),
                            f"{fast.__name__}{tuple(float(a[idx]) for a in args)}: {got[idx]!r} vs {want!r}")

    def test_basic_ops_match_exact(self):
        n = 3000
        a, b, c = (rand_f32(self.rng, n, -70, 70) for _ in range(3))
        self.check(fx.add, slow.add, a, b)
        self.check(fx.mul, slow.mul, a, b)
        self.check(fx.div, slow.div, a, b)
        self.check(fx.fma, slow.fma, a, b, c)

    def test_fma_rounding_edges(self):
        # products whose exact sum lies at / next to f32 midpoints and the underflow boundary
        rows = []
        for u in range(0x00FFFFF0, 0x01000010):
            rows.append((np.array(u, np.uint32).view(np.float32), 0.5, 0.0))
        for bb in (0x3F7FFFFC, 0x3F7FFFFD, 0x3F7FFFFE, 0x3F7FFFFF):
            rows.append((np.array(0x00800001, np.uint32).view(np.float32), np.array(bb, np.uint32).view(np.float32), 0.0))
        for k in range(200):
            x = np.float32(1 + k * 2.0 ** -12)
            rows.append((x, x, np.float32(-1.0)))
            rows.append((x, np.float32(1 + 2.0 ** -23), np.float32(2.0 ** -24)))
        a, b, c = (np.array(col, np.float32) for col in zip(*rows))
        self.check(fx.fma, slow.fma, a, b, c)
        self.check(fx.fma, slow.fma, -a, b, -c)

    def test_exp2_log2_correctly_rounded(self):
        x = np.concatenate([self.rng.uniform(-128, 129, 2500).astype(np.float32),
                            self.rng.uniform(-1, 1, 500).astype(np.float32),
                            np.float32([-126, -126.00001, -125.99999, 127.99999, 128, -150, 0, -0.0, 2.0 ** -140,
                                        np.nan, np.inf, -np.inf])])
        self.check(fx.exp2, slow.exp2, x)
        y = np.concatenate([np.abs(rand_f32(self.rng, 2500, -126, 128)),
                            np.float32(1 + self.rng.uniform(-2 ** -10, 2 ** -10, 300)),
                            np.float32([0, -0.0, -1, 2.0 ** -149, 1, 2, 0.5, np.inf, np.nan])])
        self.check(fx.log2, slow.log2, y)

    def test_exp2_hard_cases_use_the_exact_path(self):
        # Find binary64 results closest to f32 midpoints in a random sample; their f32 results must be exact.
        x = self.rng.uniform(-100, 100, 400000).astype(np.float32)
        y = np.exp2(x.astype(np.float64))
        low = (y.view(np.uint64) & np.uint64(0x1FFFFFFF)).astype(np.int64)
        worst = np.argsort(np.abs(low - (1 << 28)))[:40]
        self.check(fx.exp2, slow.exp2, x[worst])

    def test_rnd_halves_away_with_toward_zero_addition(self):
        v = np.concatenate([self.rng.uniform(-40, 40, 2000).astype(np.float32),
                            np.float32([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999997, -0.49999997,
                                        8388607.5, -8388607.5, 16777215, 0, -0.0])])
        got = fx.rnd_half_away(v)
        for g, x in zip(got, v):
            self.assertEqual(float(g), slow.rnd(float(x)), x)

    def test_min_max_signed_zero_and_nan(self):
        self.assertTrue(np.signbit(fx.fmin(0.0, -0.0)) and np.signbit(fx.fmin(-0.0, 0.0)))
        self.assertFalse(np.signbit(fx.fmax(0.0, -0.0)) or np.signbit(fx.fmax(-0.0, 0.0)))
        self.assertEqual(fx.fmin(np.nan, 3.0), 3.0)
        self.assertEqual(fx.fmax(2.0, np.nan), 2.0)
        self.assertTrue(np.isnan(fx.fmin(np.nan, np.nan)))
        self.assertTrue(np.isnan(fx.min_nan(np.nan, 1.0)) and np.isnan(fx.min_nan(1.0, np.nan)))


def random_q(rng, sz, specials=True):
    q = rng.uniform(-0.25, 3.0, (sz.Hh, sz.Wh, 3))
    if specials:
        m = rng.uniform(size=q.shape)
        q = np.where(m < 0.03, np.inf, q)
        q = np.where((m >= 0.03) & (m < 0.06), -np.inf, q)
        q = np.where((m >= 0.06) & (m < 0.09), np.nan, q)
        q = np.where((m >= 0.09) & (m < 0.11), 70000.0, q)      # beyond HMAX: binary16 infinity
        q = np.where((m >= 0.11) & (m < 0.13), 7.0, q)          # above K10 (decode clamp)
    return fx.f16(q)


class PlainResolveTests(unittest.TestCase):
    def test_plain_matches_slow_oracle(self):
        rng = np.random.default_rng(541)
        for (wo, ho), E in (((2, 2), 1.0), ((3, 5), 0.75), ((7, 4), 3.5), ((5, 1), 0.0625)):
            sz = rr.sizes((wo, ho))
            Q = random_q(rng, sz)
            s = symbols(K5=-0.125)
            E = np.float32(E)
            dest = rr.Surface(np.full((sz.He + 3, sz.We + 2, 4), 7, np.float32), "f32", (2, 1))
            res = rr.resolve(Q, output_size=(wo, ho), render_size=(2, 2), E=E, Ep=E, jitter=(0, 0),
                             reset=False, parity=0, symbols=s, colour_out=dest,
                             histories=rr.ResolveHistories((wo, ho)))
            self.assertEqual(res.variant, "plain")
            Ql = Q.tolist()
            for y in range(sz.He):
                for x in range(sz.We):
                    col = slow.plain_pixel(Ql, x, y, s)
                    want = [slow.mul(c, slow.rcp(float(E))) for c in col] + [1.0]
                    self.assertTrue(same32(res.colour.data[1 + y, 2 + x], np.float32(want)), (wo, ho, x, y))

    def test_weights_mirror_and_overshoot_clamp(self):
        s = symbols(RW_A=-0.5, RW_B=0.0, RW_C=1.0, RW_D=0.0, RW_0=0.5)
        sz = rr.sizes((2, 2))
        Q = np.zeros((sz.Hh, sz.Wh, 3))
        Q[:, :, 0] = np.arange(sz.Wh)[None, :]
        v = rr.resample(Q, sz, s)
        # even x: taps +1 (weight 1) and +2 (-0.5) => overshoot clamped to the tap range
        for y in range(sz.He):
            for x in range(sz.We):
                cx = math.floor((x + 0.5) * 1.5)
                taps = [min(max(cx + i, 0), sz.Wh - 1) for i in range(-2, 3)]
                self.assertGreaterEqual(v[y, x, 0], min(taps))
                self.assertLessEqual(v[y, x, 0], max(taps))
        s2 = symbols(RW_A=0.0, RW_B=0.0, RW_C=1.0, RW_D=0.0, RW_0=0.0)
        v2 = rr.resample(Q, sz, s2)
        for x in range(sz.We):
            cx = math.floor((x + 0.5) * 1.5)
            want = min(cx + (1 if x % 2 == 0 else -1), sz.Wh - 1)
            self.assertEqual(v2[0, x, 0], max(want, 0))


class StoreTests(unittest.TestCase):
    def run_plain(self, wo, ho, dest, bounded=False):
        sz = rr.sizes((wo, ho))
        Q = np.full((sz.Hh, sz.Wh, 3), 0.5)
        return rr.resolve(Q, output_size=(wo, ho), render_size=(1, 1), E=1, Ep=1, jitter=(0, 0), reset=True,
                          parity=1, symbols=symbols(), colour_out=dest, histories=rr.ResolveHistories((wo, ho)),
                          bounded_stores=bounded)

    def test_odd_size_default_stores_padded_row_and_column(self):
        dest = rr.Surface(np.full((6, 8, 4), -1, np.float32), "f32", (1, 1))
        out = self.run_plain(5, 3, dest).colour.data
        written = out[..., 3] == 1
        self.assertTrue(np.all(written[1:5, 1:7]))                 # We x He = 6 x 4, padded x = 5, y = 3
        self.assertEqual(int(written.sum()), 24)
        out_b = self.run_plain(5, 3, dest, bounded=True).colour.data
        self.assertEqual(int((out_b[..., 3] == 1).sum()), 15)       # Wo x Ho only
        self.assertTrue(np.all(out_b[4, :, 3] == -1) and np.all(out_b[:, 6, 3] == -1))

    def test_out_of_resource_stores_dropped(self):
        dest = rr.Surface(np.full((3, 4, 4), -1, np.float32), "f32", (2, 1))
        out = self.run_plain(4, 4, dest).colour.data
        self.assertTrue(np.all(out[1:, 2:, 3] == 1))
        self.assertTrue(np.all(out[0, :, 3] == -1) and np.all(out[:, :2, 3] == -1))

    def test_store_conversions(self):
        ties8 = (np.arange(5) + 0.5) / 255
        self.assertEqual(rr.encode(np.float32(ties8)[None, :].astype(float), "unorm8").tolist()[0],
                         [slow.unorm(float(np.float32(t)), 8) for t in ties8])
        v = np.array([[np.nan, -1, 2, 0.5 / 1023, 1.5 / 1023, 0.5, 1.0, 2 / 3]])
        got = rr.encode(np.float32(v).astype(float), "f16")
        self.assertTrue(np.isnan(got[0, 0]))
        codes = rr.encode(np.float32(v[:, :4]).astype(float), "rgb10a2")
        self.assertEqual(codes.tolist()[0], [0, 0, 1023, 0])          # NaN -> 0, clamp, 0.5 tie -> even 0 (alpha: 2 bits)
        self.assertEqual(rr.encode(np.float32([[1.5 / 1023, 1, 1, 0.5]]).astype(float), "rgb10a2").tolist()[0],
                         [slow.unorm(float(np.float32(1.5 / 1023)), 10), 1023, 1023, 2])
        self.assertTrue(np.isinf(rr.encode(np.array([[70000.0]]), "f16")[0, 0]))


class QuarterPoolTests(unittest.TestCase):
    def test_matches_slow_and_reset_keeps_content(self):
        rng = np.random.default_rng(545)
        s = symbols(K5=-0.0625, K10=1.25, C_HDEC=2.5)
        for out_size in ((2, 2), (5, 3), (9, 7), (4, 11)):
            (h1, w1), (h2, w2) = rr.pooled_sizes(out_size)
            PH1 = rng.integers(0, 1024, (h1, w1, 4)).astype(np.uint16)
            PH1[..., 3] = 3
            PH1[0, 0, :3] = [0, 1023, 512]
            PH2 = rng.integers(0, 1024, (h2 + 2, w2 + 1, 4)).astype(np.uint16)
            PH2[..., 3] = rng.integers(0, 4, PH2.shape[:2])
            same = rr.quarter_pool(PH1, PH2, output_size=out_size, reset=True, symbols=s)
            self.assertTrue(np.array_equal(same, PH2))
            got = rr.quarter_pool(PH1, PH2, output_size=out_size, reset=False, symbols=s)
            for y in range(h2):
                for x in range(w2):
                    self.assertEqual(got[y, x, :3].tolist(), slow.quarter_pixel(PH1, x, y, s), (out_size, x, y))
            self.assertTrue(np.all(got[:h2, :w2, 3] == 3))
            self.assertTrue(np.array_equal(got[h2:], PH2[h2:]) and np.array_equal(got[:, w2:], PH2[:, w2:]))


# ------------------------------------------------------------------------------------------------ extended

def make_frame(rng, out_size, render, *, layer=False, cbp=False, opacity=False, alpha_in=False, alpha_out=False,
               lorigin=(0, 0), fmt_layer="f16", jitter=None):
    sz = rr.sizes(out_size)
    wr, hr = render
    f = dict(Q=random_q(rng, sz, specials=False), E=np.float32(rng.choice([0.75, 1.0, 1.375])),
             Ep=np.float32(rng.choice([0.5, 1.0, 2.0 ** -12])),
             jitter=jitter or tuple(np.float32(rng.uniform(-0.5, 0.5, 2))))

    def surf(fmt, h, w, origin):
        if fmt == "f16":
            d = rng.uniform(0, 1.2, (h, w, 4)).astype(np.float16)
        elif fmt == "f32":
            d = rng.uniform(0, 1.2, (h, w, 4)).astype(np.float32)
        elif fmt == "unorm8":
            d = rng.integers(0, 256, (h, w, 4)).astype(np.uint8)
        else:
            d = rng.integers(0, 1024, (h, w, 4)).astype(np.uint16)
            d[..., 3] = rng.integers(0, 4, (h, w))
        return rr.Surface(d, fmt, origin)

    f["layer"] = surf(fmt_layer, hr + lorigin[1] + 1, wr + lorigin[0] + 2, lorigin) if layer else None
    f["cbp"] = surf("f32", hr + 1, wr, (3, 3)) if cbp else None
    f["opacity"] = surf("unorm8", hr + 2, wr + 1, (1, 2)) if opacity else None
    f["alpha_in"] = rr.Surface(rng.uniform(0, 1, (hr + 1, wr + 1, 1)).astype(np.float32), "f32", (1, 0)) \
        if alpha_in else None
    f["colour_in"] = surf("rgb10a2", hr + 1, wr + 2, (2, 1))
    f["alpha_out"] = rr.Surface(np.full((sz.He + 1, sz.We + 1, 1), 0.25, np.float16), "f16", (1, 1)) \
        if alpha_out else None
    f["motion"] = fx.f16(rng.uniform(-3, 3, (hr, wr, 4)))
    f["mv_origin"] = tuple(int(v) for v in rng.integers(0, 2, 2))
    f["P"] = fx.f16(np.stack([rng.uniform(0.2, 3, (hr, wr)), rng.uniform(0.2, 3, (hr, wr)),
                              rng.uniform(-0.5, 0.5, (hr, wr))], -1))
    f["G"] = rng.integers(0, 256, (hr, wr, 3)).astype(np.uint8)
    return f


def run_frame(f, out_size, render, hist, *, reset, parity, sym):
    sz = rr.sizes(out_size)
    dest = rr.Surface(np.zeros((sz.He + 2, sz.We + 1, 4), np.float32), "f32", (1, 2))
    return rr.resolve(f["Q"], output_size=out_size, render_size=render, E=f["E"], Ep=f["Ep"],
                      jitter=f["jitter"], reset=reset, parity=parity, symbols=sym, colour_out=dest,
                      histories=hist, alpha_out=f["alpha_out"], layer=f["layer"], cbp=f["cbp"],
                      opacity=f["opacity"], alpha_in=f["alpha_in"], colour_in=f["colour_in"],
                      motion=f["motion"], motion_origin=f["mv_origin"], P=f["P"], G=f["G"])


def slow_frame(f, out_size, render, hist_before, res, *, reset, parity, sym):
    """Slow oracle per pixel with the histories the reference read; returns mismatch descriptions."""
    sz = rr.sizes(out_size)
    wr, hr = render
    layer_src = f["cbp"] if f["cbp"] is not None else f["layer"]
    created_layer = (layer_src is not None) and (hist_before.TH is None or (f["opacity"] is not None
                                                                            and hist_before.OH is None))
    ctx = dict(sym=sym, We=sz.We, He=sz.He, Wh=sz.Wh, Hh=sz.Hh, Wr=wr, Hr=hr, E=float(f["E"]),
               Ep=float(f["Ep"]), jx=float(f["jitter"][0]), jy=float(f["jitter"][1]), MV=f["motion"],
               mv_origin=f["mv_origin"], reset=reset, layer=layer_src,
               layer_origin=f["layer"].origin if f["layer"] is not None else (0, 0),
               O=f["opacity"] if layer_src is not None else None,
               created=created_layer or hist_before.ah_created, ah_created=hist_before.ah_created,
               alpha=hist_before.alpha_upscaling, P=f["P"], G=f["G"])
    if layer_src is not None and hist_before.TH is not None:
        ctx["TH"] = hist_before.TH[parity]
    if ctx["O"] is not None and hist_before.OH is not None:
        ctx["OH"] = hist_before.OH[parity]
    if hist_before.alpha_upscaling:
        ctx["AH"] = hist_before.AH[parity]
        if f["alpha_in"] is not None:
            ctx.update(A=f["alpha_in"], A_origin=f["alpha_in"].origin, A_channel=0)
        else:
            ctx.update(A=f["colour_in"], A_origin=f["colour_in"].origin, A_channel=3)

    Ql = f["Q"].tolist()
    ctx["col"] = [[slow.plain_pixel(Ql, x, y, sym) for x in range(sz.We)] for y in range(sz.He)]
    bad = []
    v = res.values
    for y in range(sz.He):
        for x in range(sz.We):
            o = slow.extended_pixel(x, y, ctx)
            want = [slow.mul(slow.rcp(float(f["E"])), c) for c in o["col"]]
            if not same32(v["colour"][y, x, :3], np.float32(want)):
                bad.append(("colour", x, y, v["colour"][y, x, :3], want))
            if "T" in o and not same32(v["T"][y, x], np.float32(o["T"])):
                bad.append(("T", x, y, v["T"][y, x], o["T"]))
            if "Tp" in o and not same32(v["Tp"][y, x], np.float32(o["Tp"])):
                bad.append(("Tp", x, y, v["Tp"][y, x], o["Tp"]))
            for key in ("ah", "oa"):
                if key in o and not same32(v[key][y, x], np.float32(o[key])):
                    bad.append((key, x, y, v[key][y, x], o[key]))
            if "s" in o and bool(v["s"][y, x]) != o["s"]:
                bad.append(("s", x, y))
    return bad


CONFIGS = [
    dict(layer=True),
    dict(layer=True, opacity=True, lorigin=(2, 1)),
    dict(cbp=True),                                   # colour-before-particles alone acts as the layer
    dict(cbp=True, layer=True, lorigin=(1, 2)),       # Bp read at L's origin
    dict(cbp=True, opacity=True),
    dict(opacity=True),                               # opacity without a layer: no effect
    dict(layer=True, fmt_layer="unorm8"),
    dict(layer=True, fmt_layer="rgb10a2", opacity=True),
    dict(),
]


class ExtendedResolveTests(unittest.TestCase):
    def test_random_sequences_match_slow_oracle(self):
        rng = np.random.default_rng(5403)
        sym = symbols()
        cases = [((5, 3), (3, 2), True), ((4, 4), (3, 3), False), ((6, 3), (4, 2), True), ((3, 5), (2, 3), True)]
        for out_size, render, alpha in cases:
            for alpha_in in (False, True):
                if not alpha and alpha_in:
                    continue
                hist = rr.ResolveHistories(out_size, alpha_upscaling=alpha)
                parity = 0
                seq = rng.permutation(len(CONFIGS))[:6]
                for n, ci in enumerate(seq):
                    cfg = dict(CONFIGS[ci])
                    f = make_frame(rng, out_size, render, alpha_in=alpha_in, alpha_out=alpha and n % 2 == 1, **cfg)
                    reset = n == 0 or n == 4
                    res = run_frame(f, out_size, render, hist, reset=reset, parity=parity, sym=sym)
                    expect_ext = alpha or f["layer"] is not None or f["cbp"] is not None
                    self.assertEqual(res.variant, "extended" if expect_ext else "plain")
                    if expect_ext:
                        bad = slow_frame(f, out_size, render, hist, res, reset=reset, parity=parity, sym=sym)
                        self.assertEqual(bad[:3], [], (out_size, render, cfg, n))
                    hist = res.histories
                    parity = 1 - parity

    def test_extended_without_layer_gives_plain_colour_bits(self):
        rng = np.random.default_rng(7)
        out_size, render = (5, 3), (3, 2)
        f = make_frame(rng, out_size, render, alpha_out=True)
        sym = symbols()
        h_plain = rr.ResolveHistories(out_size)
        h_alpha = rr.ResolveHistories(out_size, alpha_upscaling=True)
        a = run_frame(f, out_size, render, h_plain, reset=False, parity=0, sym=sym)
        b = run_frame(f, out_size, render, h_alpha, reset=False, parity=0, sym=sym)
        self.assertEqual((a.variant, b.variant), ("plain", "extended"))
        self.assertTrue(same32(a.colour.data, b.colour.data))       # alpha goes to the separate output: colour alpha 1
        self.assertFalse(np.array_equal(b.alpha.data, f["alpha_out"].data))

    def test_layer_selection_and_origins(self):
        rng = np.random.default_rng(8)
        out_size, render = (4, 4), (3, 3)
        sym = symbols()
        f = make_frame(rng, out_size, render, cbp=True, layer=True, lorigin=(2, 1))
        h = run_frame(f, out_size, render, rr.ResolveHistories(out_size), reset=True, parity=1,
                      sym=sym).histories                       # TH exists: the neighbourhood (origin) is used
        with_both = run_frame(f, out_size, render, h, reset=False, parity=0, sym=sym)
        g = dict(f)
        g["layer"] = rr.Surface(np.full_like(f["layer"].data, np.float16(np.nan)), "f16", (2, 1))
        poisoned = run_frame(g, out_size, render, h, reset=False, parity=0, sym=sym)
        # Bp replaces every read of L; L contributes only its origin
        self.assertTrue(same32(with_both.values["colour"], poisoned.values["colour"]))
        g["layer"] = rr.Surface(f["layer"].data, "f16", (0, 0))
        moved = run_frame(g, out_size, render, h, reset=False, parity=0, sym=sym)
        alone = dict(f)
        alone["layer"] = None
        bp_alone = run_frame(alone, out_size, render, h, reset=False, parity=0, sym=sym)
        self.assertEqual(bp_alone.variant, "extended")
        self.assertTrue(same32(moved.values["colour"], bp_alone.values["colour"]))   # Bp alone: origin (0, 0)
        self.assertFalse(same32(moved.values["colour"], with_both.values["colour"]))

    def test_opacity_without_layer_has_no_effect(self):
        rng = np.random.default_rng(9)
        out_size, render = (4, 2), (2, 2)
        f = make_frame(rng, out_size, render, opacity=True)
        h = rr.ResolveHistories(out_size)
        res = run_frame(f, out_size, render, h, reset=False, parity=0, sym=symbols())
        self.assertEqual(res.variant, "plain")
        self.assertIsNone(res.histories.OH)
        self.assertIsNone(res.histories.TH)

    def test_creation_resets_and_history_copies(self):
        rng = np.random.default_rng(10)
        out_size, render = (4, 4), (3, 3)
        sym = symbols()
        h = rr.ResolveHistories(out_size)
        frames = [make_frame(rng, out_size, render, layer=True) for _ in range(6)]
        # frame 0: TH created on a non-reset frame -> hist = cur, i.e. T = cur exactly when w + w1 sum is exact
        r0 = run_frame(frames[0], out_size, render, h, reset=False, parity=0, sym=sym)
        self.assertEqual(r0.histories.log[0], "TH created")
        cur_only = run_frame(frames[0], out_size, render, rr.ResolveHistories(out_size), reset=True, parity=0, sym=sym)
        self.assertTrue(same32(r0.values["T"], cur_only.values["T"]))
        h1 = r0.histories                                              # TH[1] written
        # frame 1 (parity 1): reads TH[1]
        r1 = run_frame(frames[1], out_size, render, h1, reset=False, parity=1, sym=sym)
        h1b = h1.copy()
        h1b.TH[0][:] = np.float16(np.nan)                              # the copy not read must not matter
        r1b = run_frame(frames[1], out_size, render, h1b, reset=False, parity=1, sym=sym)
        self.assertTrue(same32(r1.values["T"], r1b.values["T"]))
        # frame 2 (parity 0): no layer -> plain; TH untouched
        nolayer = dict(frames[2])
        nolayer["layer"] = None
        r2 = run_frame(nolayer, out_size, render, r1.histories, reset=False, parity=0, sym=sym)
        self.assertEqual(r2.variant, "plain")
        self.assertTrue(all(np.array_equal(a, b, equal_nan=True) for a, b in zip(r2.histories.TH, r1.histories.TH)))
        # frame 3 (parity 1): reads TH[1] as last written (by frame 0), no reset
        r3 = run_frame(frames[3], out_size, render, r2.histories, reset=False, parity=1, sym=sym)
        alt = r2.histories.copy()
        alt.TH[1][:] = r0.histories.TH[1]
        r3b = run_frame(frames[3], out_size, render, alt, reset=False, parity=1, sym=sym)
        self.assertTrue(same32(r3.values["T"], r3b.values["T"]))
        alt.TH[1][:] = np.float16(0.0)
        r3c = run_frame(frames[3], out_size, render, alt, reset=False, parity=1, sym=sym)
        self.assertFalse(same32(r3.values["T"], r3c.values["T"]))
        # frame 4: opacity appears with an existing TH -> both histories reset
        f4 = make_frame(rng, out_size, render, layer=True, opacity=True)
        r4 = run_frame(f4, out_size, render, r3.histories, reset=False, parity=0, sym=sym)
        self.assertIn("OH created", r4.histories.log)
        ref = run_frame(f4, out_size, render, r3.histories, reset=True, parity=0, sym=sym)
        self.assertTrue(same32(r4.values["T"], ref.values["T"]) and same32(r4.values["Tp"], ref.values["Tp"]))
        self.assertTrue(np.all(r4.histories.OH[1][..., 3] == 1))

    def test_half_way_jitter_decides_blend_weight(self):
        rng = np.random.default_rng(11)
        out_size, render = (4, 4), (2, 2)            # We/Wr = 2: jx' = 2 * jx
        sym = symbols(TBW=0.375)
        for jx, jy, sx, sy in ((0.25, 0.0, False, True), (-0.25, 0.0, False, True), (0.0, 0.25, True, False),
                               (0.0, -0.25, True, False), (0.125, -0.125, True, True), (0.25, 0.25, False, False)):
            f = make_frame(rng, out_size, render, layer=True, jitter=(np.float32(jx), np.float32(jy)))
            res = run_frame(f, out_size, render, rr.ResolveHistories(out_size), reset=True, parity=0, sym=sym)
            s = res.values["s"]
            self.assertEqual(bool(np.all(s)), sx and sy, (jx, jy))
            self.assertEqual(bool(np.any(s)), sx and sy, (jx, jy))

    def test_alpha_reset_and_nan_replacement(self):
        rng = np.random.default_rng(12)
        out_size, render = (4, 2), (3, 2)
        sym = symbols()
        f = make_frame(rng, out_size, render, alpha_in=True)
        h = rr.ResolveHistories(out_size, alpha_upscaling=True)
        res = run_frame(f, out_size, render, h, reset=False, parity=0, sym=sym)   # creation frame: hc = 0
        v = res.values
        self.assertTrue(np.all(v["hc"] == 0))
        self.assertTrue(same32(v["ah"], fx.mul(v["a_up"], v["beta"])))
        self.assertTrue(same32(v["oa"], fx.mul(v["a_up"], v["gamma"])))
        g = dict(f)
        g["P"] = f["P"].copy()
        g["P"][0, 0] = [-30000.0, -30000.0, 0.0]      # growing exponent: weights overflow, sums become NaN
        res2 = run_frame(g, out_size, render, res.histories, reset=False, parity=1, sym=sym)
        v2 = res2.values
        nan_ah = np.isnan(fx.fma(v2["a_up"], v2["beta"], fx.mul(fx.sub(1, v2["beta"]), v2["hc"])))
        self.assertTrue(np.any(nan_ah))
        self.assertTrue(same32(v2["ah"][nan_ah], v2["a_up"][nan_ah]))


if __name__ == "__main__":
    unittest.main()


class BilinearAndIntegrationTests(unittest.TestCase):
    def test_bilinear_matches_slow_including_clamped_edge_fractions(self):
        rng = np.random.default_rng(5214)
        for _ in range(3000):
            h, w = (int(v) for v in rng.integers(1, 6, 2))
            tex = rng.uniform(-2, 2, (h, w, 1)).astype(np.float32)
            tex.reshape(-1)[:: 5] = rng.choice([np.inf, -np.inf, 0.0, -0.0])
            u, v = np.float32(rng.uniform(-0.2, 1.2, 2))
            got = rr.bilinear(tex.astype(np.float64), u, v)[0]
            want = slow.bilinear(lambda i, j: [float(tex[j, i, 0])], w, h, float(u), float(v))[0]
            self.assertTrue(same32(np.float32(got), np.float32(want)), (h, w, u, v, got, want))

    def test_resource_state_runs_ph2_and_resolve_on_its_own_images(self):
        from clean.rr_f.reference.resources import ResourceState
        from clean.rr_f.tests import test_resources as tr
        s = ResourceState((3, 2), (5, 3), alpha_upscaling=True)
        sym = symbols()
        frames = []
        for n in (1, 2):
            p = tr.begin(s, n, layer_attached=True)
            if p.reset:
                s.clear_exposure(p)
            else:
                ph1 = s.buffer("PH1")
                s.run_ph2(p, symbols=sym)
                want = rr.quarter_pool(ph1, np.zeros_like(s.buffer("PH2")), output_size=(5, 3), reset=False,
                                       symbols=sym)
                self.assertTrue(np.array_equal(s.buffer("PH2"), want))
            tr.expose(s, p, n)
            tr.motion(s, p, n)
            s.lifecycle.input_consumed_jitter(p)
            s.acknowledge_prepared_input(p)
            s.commit_head(p, tr.head(s, n))
            s.commit_output(p, tr.output(s, n))
            before = s.resolve_histories.copy()
            layer = rr.Surface(np.full((2, 3, 4), 0.5, np.float16), "f16")
            colour_in = rr.Surface(np.full((2, 3, 4), 0.75, np.float32), "f32")
            dest = rr.Surface(np.zeros((4, 6, 4), np.float32), "f32")
            got = s.run_resolve(p, symbols=sym, colour_out=dest, layer=layer, colour_in=colour_in)
            e = s.exposure_result
            want = rr.resolve(s.buffer("Q"), output_size=(5, 3), render_size=(3, 2), E=e.E, Ep=e.Ep,
                              jitter=p.current_jitter, reset=bool(p.reset), parity=p.parity, symbols=sym,
                              colour_out=dest, histories=before, layer=layer, colour_in=colour_in,
                              motion=s.buffer(f"PM[{p.history_write}]"), P=s.buffer("PREC")[..., :3],
                              G=s.buffer("GATES")[..., :3])
            self.assertTrue(same32(got.colour.data, want.colour.data))
            self.assertEqual(got.variant, "extended")
            s.lifecycle.complete_frame(p)
            frames.append(got)
        self.assertIn("TH created", s.resolve_histories.log)
        self.assertFalse(s.resolve_histories.ah_created)


class GpuHelperTests(unittest.TestCase):
    """CPU-only checks of the GPU runner's layouts; never launches the GPU."""

    def test_rgb10a2_pack_roundtrip_and_param_slots(self):
        from clean.rr_f.gpu import resolve as gres
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 1024, (3, 5, 4)).astype(np.uint16)
        codes[..., 3] &= 3
        self.assertTrue(np.array_equal(gres.unpack_rgb10a2(gres.pack_rgb10a2(codes)), codes))
        from pathlib import Path
        src = (Path(gres.build.KDIR) / "rr_resolve.hlsl").read_text(encoding="utf-8")
        for name, slot in gres.RES.items():
            self.assertIn(f"{slot} {name}", src.replace("\n//   ", " "))
        p = gres.params(symbols(), sizes=rr.sizes((5, 3)), render=(3, 2), flags=5)
        self.assertEqual(p[17:22].tolist(), [int(np.float16(v).view(np.uint16)) for v in symbols().even_weights()])
        self.assertEqual(p.size, gres.NPARAM)

    def test_generated_constants_header_is_current(self):
        import importlib
        import io
        from contextlib import redirect_stdout
        from pathlib import Path
        gen = importlib.import_module("clean.rr_f.gpu.gen_elem_consts")
        path = Path(gen.OUT)
        before = path.read_bytes()
        try:
            with redirect_stdout(io.StringIO()):
                gen.main()
            crlf, lf = bytes([13, 10]), bytes([10])
            self.assertEqual(path.read_bytes().replace(crlf, lf), before.replace(crlf, lf))
        finally:
            path.write_bytes(before)               # keep the checked-out line endings
