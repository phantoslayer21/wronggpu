# SPDX-License-Identifier: MIT
"""Frame generation (#96): the light path's schedule and blend (clean/fg/reference/pipeline.py, lite.py): grid sizes,
the once-per-game-frame / per-call split for 2x to 6x, history and swaps on the last index only, resets, frozen flags,
the generation-disabled word, the light blend's simple path, UI overlay (flag 1) and refined candidates (flag 2) against
scalar versions, and whole sequences on synthetic scenes.

    python -m unittest clean.fg.tests.test_fg_lite_pipeline
"""
import unittest

import numpy as np

from clean.fg.reference import lite as LITE
from clean.fg.reference import pipeline as P
from clean.fg.reference import scene as SC
from clean.fg.reference import stages_common as S
from clean.fg.tests.test_fg_common_stages import adiv, centre, fma, h, hb, sq, tex_linear

f = np.float32
ONCE = {"downsample", "mvec", "p2c_scatter", "p2c_unpack", "p2c_inpaint", "p2c_pull", "p2c_push", "hole_mask",
        "quality_init", "net1_input", "net1", "net1_output", "detect", "set_disable"}
PER_CALL = {"interm", "weights", "lite_blend", "menu_output"}


def zero_net1(cells):
    return np.zeros(cells.shape[:2] + (16,), np.float16)


class Geometry(unittest.TestCase):
    def test_table(self):
        # frame -> d, net1 image, net1 cells, pad left / top
        rows = {(1280, 720): (2, (640, 360), (320, 192), (0, 6)), (1920, 1080): (2, (960, 540), (480, 288), (0, 9)),
                (2560, 1440): (2, (1280, 720), (640, 384), (0, 12)), (3440, 1440): (2, (1720, 720), (864, 384), (2, 12)),
                (3840, 2160): (4, (960, 540), (480, 288), (0, 9)), (7680, 4320): (4, (1920, 1080), (960, 544), (0, 2))}
        for (w, hh), (d, img, cells, pad) in rows.items():
            g = P.Geometry(w, hh)
            self.assertEqual((g.d, (g.wd, g.hd), (g.W0, g.H0), (g.pad_l, g.pad_t)), (d, img, cells, pad), (w, hh))
            self.assertEqual(g.W0 % 32, 0)
        self.assertEqual((P.Geometry(1920, 1080).w1, P.Geometry(3840, 2880).dp, P.Geometry(3840, 2881).dp), (960, 2, 4))
        with self.assertRaises(ValueError):
            P.Geometry(3, 100)

    def test_refusals(self):
        g = P.Geometry(64, 36)
        with self.assertRaises(NotImplementedError):
            P.Host(P.Settings(lite=False), g)
        h_ = P.Host(P.Settings(), g)
        with self.assertRaises(NotImplementedError):
            h_.call(1, 1, distortion=True)
        with self.assertRaises(ValueError):
            h_.call(6, 1)                                               # 7x is not accepted
        with self.assertRaises(ValueError):
            h_.call(2, 3)


class Schedule(unittest.TestCase):
    def names(self, ops):
        return [n for n, _ in ops]

    def test_2x_every_call_is_a_game_frame(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        for k in range(3):
            ops, info = host.call(1, 1, hudless=True)
            n = self.names(ops)
            self.assertTrue(info["first"] and info["last"])
            self.assertEqual(info["t"], f(0.5))
            self.assertEqual(info["history"], k > 0)
            self.assertIn("downsample", n)
            self.assertEqual(n[-1], "swap")
            self.assertEqual("lite_blend" in n, k > 0)
            self.assertEqual("net1" in n, k > 0)
            self.assertEqual("quality_init" in n, k == 0)
            self.assertEqual("final_copy" in n, k == 0)
            self.assertEqual(sum(1 for x in n if x == "clear") > 0, k == 0)

    def test_6x_split(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        ts = []
        for frame in range(3):
            for i in range(1, 6):
                ops, info = host.call(5, i, hudless=True)
                n = set(self.names(ops))
                ts.append(int(np.array(info["t"]).view(np.uint32)))
                self.assertEqual(info["first"], i == 1)
                self.assertEqual(info["last"], i == 5)
                self.assertEqual(bool(n & ONCE), i == 1)
                self.assertEqual("swap" in n, i == 5)
                if frame > 0:
                    self.assertTrue(PER_CALL - {"menu_output"} <= n)
                else:
                    self.assertFalse(n & PER_CALL)
                    self.assertIn("final_copy", n)
        self.assertEqual(ts[:5], [0x3E2AAAAB, 0x3EAAAAAB, 0x3F000000, 0x3F2AAAAB, 0x3F555555])

    def test_missing_last_index_freezes_history(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        host.call(1, 1)
        for i in (1, 2, 3, 4, 1, 2):                                      # i = 5 never arrives
            ops, info = host.call(5, i)
            n = self.names(ops)
            self.assertNotIn("swap", n)
            self.assertEqual(bool(set(n) & ONCE), i == 1 and not host.since_last - 1)
        self.assertEqual(host.history, 1)

    def test_changing_n_resets_nothing(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        host.call(1, 1, hudless=True)
        for n in (5, 2, 3):
            for i in range(1, n + 1):
                _, info = host.call(n, i, hudless=True)
                self.assertTrue(info["history"])
        self.assertEqual(host.history, 4)

    def test_flag_change_and_reset(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        host.call(1, 1, hudless=True)
        _, info = host.call(1, 1, hudless=True)
        self.assertTrue(info["history"])
        _, info = host.call(1, 1, hudless=False)                         # Lite flag 2 drops: reset
        self.assertFalse(info["history"])
        _, info = host.call(1, 1, hudless=False)
        self.assertTrue(info["history"])
        _, info = host.call(1, 1, hudless=False, reset=True)
        self.assertFalse(info["history"])

    def test_flags_frozen_in_a_game_frame(self):
        host = P.Host(P.Settings(), P.Geometry(64, 36))
        host.call(1, 1, hudless=True)
        ops, _ = host.call(3, 1, hudless=True)
        ops2, _ = host.call(3, 2, hudless=False)                          # ignored until the next first call
        blend = [p for n, p in ops2 if n == "lite_blend"][0]
        self.assertTrue(blend["flag2"])
        ops3, info = host.call(3, 3, hudless=False)
        swap = [p for n, p in ops3 if n == "swap"][0]
        self.assertIn(("Lc", "Lp"), swap["pairs"])

    def test_derived_flags(self):
        st = P.Settings()
        self.assertEqual(P.derived_flags(st, True, False, False)["lite2"], True)
        self.assertEqual(P.derived_flags(st, False, True, False)["lite2"], False)
        st1 = P.Settings(lite_l1=True)
        fl = P.derived_flags(st1, True, False, True)
        self.assertTrue(fl["lite1"] and not fl["lite2"])
        self.assertFalse(fl["ui_recomp"])
        r3 = P.r3_flags(fl)
        self.assertTrue(r3["F1"] and r3["L1"])
        self.assertFalse(P.derived_flags(P.Settings(ignore_hudless=True), True, False, False)["lite2"])

    def test_swaps(self):
        host = P.Host(P.Settings(lite_l1=True), P.Geometry(64, 36))
        ops, _ = host.call(1, 1, hudless=True, ui4=True)
        pairs = [p for n, p in ops if n == "swap"][0]["pairs"]
        self.assertIn(("Uc", "Up"), pairs)
        self.assertNotIn(("Lc", "Lp"), pairs)
        self.assertNotIn(("Um", "Ump"), pairs)                            # case U: no UI mask
        clears = [p["role"] for n, p in ops if n == "clear"]
        self.assertEqual(clears, ["Cp", "Kp", "Zp", "m", "Uc", "Up"])


# ---- the light blend ------------------------------------------------------------------------------------------------

def blend_scalar(Cp, Cc, MA, MB, W, wm, hm, t, flag1, flag2, Up, Uc, Lp, Lc):
    ho, wo = Cc.shape[:2]
    out = np.zeros((ho, wo, 4), np.float32)
    lreg = S.Region(wo, ho)

    def correct(c, alpha, uvx, uvy, u, v, empty, P0, Q0, L, own):
        if not (all(np.isfinite(c)) and not empty and np.isfinite(uvx) and np.isfinite(uvy) and wo >= 2 and ho >= 2):
            return c
        inb = lambda px, py: 0 <= px < wo - 1 and 0 <= py < ho - 1
        if not inb(fma(uvx, wo, -0.5), fma(uvy, ho, -0.5)) or not inb(fma(u, wo, -0.5), fma(v, ho, -0.5)):
            return c
        s = adiv(f(alpha - f(0.5)), LITE.ALPHA_SPAN)
        s = f(0) if s <= 0 else (f(1) if s >= 1 else s)
        g = f(f(s * s) * f(f(3) - f(s + s)))
        if not g > 0:
            return c
        dx, dy = f(f(uvx - u) * f(wo)), f(f(uvy - v) * f(ho))
        if not sq(dx, dy) >= 4:
            return c
        if not (all(np.isfinite(P0[:3])) and all(np.isfinite(Q0[:3])) and P0[3] == 0 and Q0[3] == 0):
            return c
        X1, Y1 = lreg.coords(np.array([uvx]), np.array([uvy]))
        X0, Y0 = lreg.coords(np.array([u]), np.array([v]))
        d1, d0 = tex_linear(L, X1[0], Y1[0])[:3], tex_linear(L, X0[0], Y0[0])[:3]
        dl = (own[:3] - d0).astype(np.float32)
        if not (all(np.isfinite(d1)) and all(np.isfinite(d0)) and all(np.isfinite(dl))):
            return c
        if not np.max(np.abs(dl)) <= LITE.DETAIL_MAX:
            return c
        return np.array([fma(g, f(S.sat(f(d1[i] + dl[i]))[()] - c[i]), c[i]) for i in range(3)], np.float32)
    for y in range(ho):
        for x in range(wo):
            u, v = centre(x, wo), centre(y, ho)
            mA, eA = S.vec_lookup(MA, wm, hm, np.array([u]), np.array([v]))
            mB, eB = S.vec_lookup(MB, wm, hm, np.array([u]), np.array([v]))
            mA, eA, mB, eB = mA[0], bool(eA[0]), mB[0], bool(eB[0])
            P0, Q0 = Cp[y, x].astype(np.float32), Cc[y, x].astype(np.float32)
            uPx, uPy, uQx, uQy = f(u + mB[0]), f(v + mB[1]), f(u + mA[0]), f(v + mA[1])
            sP, sQ = tex_linear(Cp, uPx, uPy), tex_linear(Cc, uQx, uQy)
            cP, cQ = sP[:3], sQ[:3]
            w = tex_linear(W, u, v)
            if flag2:
                cP, cQ = (correct(cP, sP[3], uPx, uPy, u, v, eB, P0, Q0, Lp, P0),
                          correct(cQ, sQ[3], uQx, uQy, u, v, eA, P0, Q0, Lc, Q0))
            o = []
            for i in range(3):
                a = fma(1.0 if eB else 0.0, f(cQ[i] - cP[i]), cP[i])
                b = fma(1.0 if eA else 0.0, f(cP[i] - cQ[i]), cQ[i])
                rP = fma(f(a - P0[i]), w[0], P0[i])
                rQ = fma(f(b - Q0[i]), w[1], Q0[i])
                o.append(fma(f(rQ - rP), t, rP))
            if flag1:
                up, uc = Up[y, x].astype(np.float32), Uc[y, x].astype(np.float32)
                U = [fma(f(uc[k] - up[k]), t, up[k]) for k in range(4)]
                o = [fma(o[i], f(f(1) - U[3]), U[i]) for i in range(3)]
            out[y, x] = (o[0], o[1], o[2], 1.0)
    return hb(out)


def vec_buffer(rng, n, frac, scale, wm, hm):
    buf = np.full(n, S.EMPTY64, np.int64)
    for k in rng.choice(n, int(n * frac), replace=False):
        vx = np.float16(rng.uniform(-scale, scale) / wm)
        vy = np.float16(rng.uniform(-scale, scale) / hm)
        buf[k] = (0x3F800000 << 32) | (int(vx.view(np.uint16)) << 16) | int(vy.view(np.uint16))
    return buf


class LiteBlend(unittest.TestCase):
    def inputs(self, seed, wo=12, ho=9, wm=6, hm=5):
        rng = np.random.default_rng(seed)
        Cp = rng.uniform(0, 1, (ho, wo, 4)).astype(np.float16)
        Cc = rng.uniform(0, 1, (ho, wo, 4)).astype(np.float16)
        for C in (Cp, Cc):                                               # alpha: UI-free (0) with a UI patch (1)
            C[..., 3] = 0
            C[1:4, 6:10, 3] = 1
        MA = vec_buffer(rng, wm * hm, 0.7, 6.0, wm, hm)
        MB = vec_buffer(rng, wm * hm, 0.7, 6.0, wm, hm)
        Wimg = rng.uniform(0, 1, (hm, wm, 2)).astype(np.float16)
        Up = rng.uniform(0, 0.5, (ho, wo, 4)).astype(np.float16)
        Uc = rng.uniform(0, 0.5, (ho, wo, 4)).astype(np.float16)
        Lp = (Cp.astype(np.float32) + rng.uniform(-0.05, 0.05, Cp.shape)).astype(np.float16)
        Lc = (Cc.astype(np.float32) + rng.uniform(-0.05, 0.05, Cc.shape)).astype(np.float16)
        return Cp, Cc, MA, MB, Wimg, wm, hm, Up, Uc, Lp, Lc

    def check(self, seed, t, flag1, flag2):
        Cp, Cc, MA, MB, Wimg, wm, hm, Up, Uc, Lp, Lc = self.inputs(seed)
        out = np.full((Cc.shape[0] + 2, Cc.shape[1] + 3, 4), np.float16(203.25), np.float16)
        LITE.lite_blend(Cp, Cc, MA, MB, Wimg, wm, hm, t, flag1, flag2, Up, Uc, Lp, Lc, out, 2, 1)
        ref = blend_scalar(Cp, Cc, MA, MB, Wimg, wm, hm, f(t), flag1, flag2, Up, Uc, Lp, Lc)
        np.testing.assert_array_equal(out[1:-1, 2:-1].view(np.uint16), ref.view(np.uint16))
        self.assertTrue(np.all(out[0] == np.float16(203.25)) and np.all(out[:, :2] == np.float16(203.25)))
        self.assertTrue(np.all(out[1:-1, 2:-1, 3] == 1))
        return out[1:-1, 2:-1]

    def test_simple_path(self):
        for seed, t in ((1, 0.5), (2, 1 / 6), (3, 5 / 6)):
            self.check(seed, t, False, False)

    def test_ui_overlay(self):
        a = self.check(4, 0.25, True, False)
        b = self.check(4, 0.25, False, False)
        self.assertFalse(np.array_equal(a, b))

    def test_refined_path_changes_some_pixels(self):
        changed = 0
        for seed in range(5, 11):
            a = self.check(seed, 0.5, False, True)
            b = self.check(seed, 0.5, False, False)
            changed += int(np.sum(np.any(a != b, -1)))
        self.assertGreater(changed, 0)

    def test_refined_rules(self):
        # one pixel, a 3-pixel motion into a UI patch, both co-located pixels UI-free: corrected; each failed rule skips
        wo, ho = 16, 8
        Cp = np.full((ho, wo, 4), 0.4, np.float16)
        Cp[..., 3] = 0
        Cp[:, 10:, :3] = np.float16(0.9)
        Cp[:, 10:, 3] = 1
        Cc = Cp.copy()
        Lp = np.full((ho, wo, 4), 0.38, np.float16)
        Lp[..., 3] = 0
        vx = np.float16(3.0 / wo)
        MB = np.full(wo * ho, S.EMPTY64, np.int64)
        MB[:] = (0x3F800000 << 32) | (int(vx.view(np.uint16)) << 16)
        MA = MB.copy()
        Wimg = np.full((ho, wo, 2), 1.0, np.float16)
        z = np.zeros((ho, wo, 4), np.float16)

        def run(Cp_, Lp_, MB_):
            out = np.zeros((ho, wo, 4), np.float16)
            LITE.lite_blend(Cp_, Cc, MA, MB_, Wimg, wo, ho, 0.0, False, True, z, z, Lp_, Lp_, out)
            plain = np.zeros((ho, wo, 4), np.float16)
            LITE.lite_blend(Cp_, Cc, MA, MB_, Wimg, wo, ho, 0.0, False, False, z, z, Lp_, Lp_, plain)
            return out[4, 8, :3].astype(np.float32), plain[4, 8, :3].astype(np.float32)
        a, b = run(Cp, Lp, MB)
        self.assertFalse(np.array_equal(a, b))                          # corrected toward Lp + detail (0.4)
        self.assertLess(abs(float(a[0]) - 0.4), abs(float(b[0]) - 0.4))
        Cp2 = Cp.copy()
        Cp2[4, 8, 3] = np.float16(0.5)                                    # co-located pixel not UI-free
        a, b = run(Cp2, Lp, MB)
        self.assertTrue(np.array_equal(a, b))
        Lp2 = Lp.copy()
        Lp2[..., :3] = np.float16(0.2)                                    # detail 0.2 > 0.15
        a, b = run(Cp, Lp2, MB)
        self.assertTrue(np.array_equal(a, b))
        MB2 = MB.copy()
        MB2[:] = (0x3F800000 << 32) | (int(np.float16(1.5 / wo).view(np.uint16)) << 16)   # under 2 pixels
        a, b = run(Cp, Lp, MB2)
        self.assertTrue(np.array_equal(a, b))

    def test_empty_vectors_take_the_other_candidate(self):
        Cp, Cc, MA, MB, Wimg, wm, hm, Up, Uc, Lp, Lc = self.inputs(12)
        MB[:] = S.EMPTY64
        Wimg[...] = 1.0
        out = np.zeros(Cc.shape, np.float16)
        LITE.lite_blend(Cp, Cc, MA, MB, Wimg, wm, hm, 0.0, False, False, Up, Uc, Lp, Lc, out)
        # t = 0, w.x = 1: out = a = cQ (MB empty)
        ho, wo = Cc.shape[:2]
        u, v = S.grid_uv(wo, ho)
        mA, _ = S.vec_lookup(MA, wm, hm, np.broadcast_to(u, (ho, wo)), np.broadcast_to(v, (ho, wo)))
        cQ = S.read_linear(Cc, (u + mA[..., 0]).astype(np.float32), (v + mA[..., 1]).astype(np.float32))[..., :3]
        np.testing.assert_array_equal(out[..., :3], cQ.astype(np.float16))


# ---- whole sequences -----------------------------------------------------------------------------------------------

class Sequences(unittest.TestCase):
    def run_seq(self, settings, frames, wo=96, ho=54, net1=zero_net1, **scene):
        g = P.Geometry(wo, ho)
        pl = P.LitePipeline(settings, g, net1)
        res = []
        for fr, (n, over) in enumerate(frames):
            kw = dict(scene)
            kw.update(over)
            for i in range(1, n + 1):
                c = SC.make_call(wo, ho, fr, n=n, i=i, **kw)
                r = pl.evaluate(c)
                res.append((fr, i, c, r))
        return pl, res

    def test_2x_and_6x(self):
        for n in (1, 5):
            pl, res = self.run_seq(P.Settings(), [(n, {})] * 3, hudless=True)
            self.assertEqual(len(res), 3 * n)
            for fr, i, c, r in res:
                o = r["out"].astype(np.float32)
                self.assertTrue(np.all(np.isfinite(o)))
                if fr == 0:
                    np.testing.assert_array_equal(r["out"], c.colour)        # no history: the game frame itself
                    self.assertEqual(r["disable"], 1)
                else:
                    self.assertEqual(r["disable"], 0)
                    self.assertFalse(np.array_equal(r["out"], c.colour))
                    self.assertTrue(np.all(o[..., 3] == 1))
            outs = [r["out"] for fr, i, c, r in res if fr == 2]
            self.assertEqual(len({o.tobytes() for o in outs}), n)           # one distinct frame per index

    def test_flag2_corrects_in_the_scene(self):
        _, a = self.run_seq(P.Settings(), [(1, {})] * 3, ui="ui1", hudless=True)
        _, b = self.run_seq(P.Settings(lite_l2=False), [(1, {})] * 3, ui="ui1", hudless=True)
        self.assertFalse(np.array_equal(a[-1][3]["out"], b[-1][3]["out"]))
        self.assertIn("lite2", [k for k, v in P.derived_flags(P.Settings(), True, False, True).items() if v])

    def test_flag1_ui_composite(self):
        pl, res = self.run_seq(P.Settings(lite_l1=True), [(2, {})] * 3, ui="ui4", hudless=True)
        blends = [p for r in res for n, p in r[3]["ops"] if n == "lite_blend"]
        self.assertTrue(blends and all(p["flag1"] and not p["flag2"] for p in blends))
        self.assertTrue(np.all(np.isfinite(res[-1][3]["out"].astype(np.float32))))

    def test_menu_disables_generation(self):
        st = P.Settings(menu_detection=True)
        pl, res = self.run_seq(st, [(2, {}), (2, {}), (2, {"menu": True}), (2, {}), (2, {})], ui="both",
                               hudless=True)
        disabled = [r["disable"] for fr, i, c, r in res if i == 1]
        self.assertEqual(disabled, [1, 0, 1, 1, 0])                        # detected, then latched one more frame
        for fr, i, c, r in res:
            if fr in (2, 3):
                np.testing.assert_array_equal(r["out"], c.colour)            # exact copy of the game frame
        self.assertEqual([x[3] for x in pl.log], [False, False, True, False, False])

    def test_hudless_detector_runs_without_ui(self):
        pl, res = self.run_seq(P.Settings(menu_detection=True), [(1, {})] * 2, hudless=True)
        self.assertEqual([x[1] for x in pl.log], ["hudless", "hudless"])

    def test_exact_net1_runs(self):
        params = SC.synthetic_net1_params()
        pl, res = self.run_seq(P.Settings(), [(1, {})] * 2, wo=64, ho=36, net1=P.exact_net1(params), hudless=True)
        self.assertTrue(np.all(np.isfinite(res[-1][3]["out"].astype(np.float32))))
        m = pl.img["m"].astype(np.float32)
        self.assertTrue(np.all((m >= 0) & (m <= 1)))

    def test_output_offset_and_prefill(self):
        g = P.Geometry(64, 36)
        pl = P.LitePipeline(P.Settings(), g, zero_net1)
        for fr in range(2):
            c = SC.make_call(64, 36, fr, pad=2, out_size=(68, 40), out_offset=(2, 2), hudless=True)
            r = pl.evaluate(c)
            self.assertIn("prefill", [n for n, _ in r["ops"]])
            np.testing.assert_array_equal(r["out"][:2], c.colour[:2])         # outside the extent: the colour
            if fr == 0:
                np.testing.assert_array_equal(r["out"], c.colour)


if __name__ == "__main__":
    unittest.main()
