# SPDX-License-Identifier: GPL-3.0-only
"""CPU proofs of the exactness arguments behind frame generation's Turbo v2 kernels (clean/dx12/kernels/fg_lite_p2c.hlsl,
fg_lite_g.hlsl). Each class re-states what the old kernel computes and what the new kernel computes, step by step as the
HLSL does, and compares them on random and edge inputs. The kernels themselves are checked on a GPU
(clean/fg/runtime/tests/test_fg_d3d12_e2e.cpp --gpu, paired runs against the kernels they replace)."""
import unittest

import numpy as np

F32 = np.float32


# ---- fg_lite_p2c_inpaint_tiled == fg_lite_p2c_inpaint -------------------------------------------------------------------
def inpaint_old(fl, w1, h1):
    """fg_lite_p2c_inpaint: bits of the 5 x 5 neighbourhood by flat index and column mask, nine 3 x 3 windows."""
    size = w1 * h1
    out = np.zeros(size, dtype=np.uint32)
    for idx in range(size):
        x = idx % w1
        xmask = ((1 << min(w1 - x + 2, 5)) - 1) & (31 << max(2 - x, 0))
        bits = 0
        for j in range(5):
            p = idx + (j - 2) * w1 - 2
            row = 0
            for i in range(5):
                q = p + i
                if 0 <= q < size and (int(fl[q]) & 2):
                    row |= 1 << i
            bits |= (row & xmask) << (j * 5)
        inp = True
        for j in range(3):
            for i in range(3):
                mask = 0x1ce7 << (j * 5 + i)
                inp = inp and (bits & mask) != 0
        dyn = (int(fl[idx]) & 1) != 0
        out[idx] = (2 if (dyn and not inp) else 0) | (1 if inp else 0)
    return out


def inpaint_tiled(fl, w1, h1):
    """fg_lite_p2c_inpaint_tiled: 32 x 8 tiles, 36-bit row masks (columns x0 - 2 .. x0 + 33), dilate then erode."""
    out = np.zeros(w1 * h1, dtype=np.uint32)

    def flag(x, y):
        return 0 <= x < w1 and 0 <= y < h1 and (int(fl[y * w1 + x]) & 2) != 0

    for ty in range((h1 + 7) // 8):
        for tx in range((w1 + 31) // 32):
            x0, y0 = tx * 32, ty * 8
            row = [0] * 12
            for r in range(12):
                m = 0
                for c in range(36):
                    if flag(x0 - 2 + c, y0 - 2 + r):
                        m |= 1 << c
                row[r] = m
            dil = [b | (b >> 1) | (b >> 2) for b in row]
            dv = [dil[r] | dil[r + 1] | dil[r + 2] for r in range(10)]
            for wave in range(8):
                for lane in range(32):
                    x, y = x0 + lane, y0 + wave
                    if x >= w1 or y >= h1:
                        continue
                    e = dv[wave] & dv[wave + 1] & dv[wave + 2]
                    inp = ((e >> lane) & 7) == 7
                    dyn = (int(fl[y * w1 + x]) & 1) != 0
                    out[y * w1 + x] = (2 if (dyn and not inp) else 0) | (1 if inp else 0)
    return out


class InpaintTiledTest(unittest.TestCase):
    def test_equal_on_random_and_edge_grids(self):
        rng = np.random.default_rng(7)
        sizes = [(1, 1), (2, 3), (3, 2), (5, 9), (31, 8), (32, 8), (33, 9), (40, 17), (65, 7), (64, 16), (1, 20), (20, 1)]
        for w1, h1 in sizes:
            for density in (0.0, 0.05, 0.3, 0.7, 0.97, 1.0):
                fl = np.zeros(w1 * h1, dtype=np.uint32)
                fl |= (rng.random(w1 * h1) < density).astype(np.uint32) * 2
                fl |= (rng.random(w1 * h1) < 0.5).astype(np.uint32)
                np.testing.assert_array_equal(inpaint_old(fl, w1, h1), inpaint_tiled(fl, w1, h1),
                                              err_msg="%dx%d density %.2f" % (w1, h1, density))


# ---- fg_lite_p2c_fine_mask: the hole mask reads F at the pixel's own index when the grids are equal ---------------------
class HoleMaskGridTest(unittest.TestCase):
    def test_nearest_index_of_a_centre_is_the_pixel(self):
        """fg_centre(i, n) = sat(fl((i + 0.5) * (1 / n))); fg_nearest_index(X, n) = floor(X * n): i for every i, n."""
        for n in list(range(1, 300)) + [511, 512, 513, 640, 720, 1023, 1024, 1279, 1280, 1281, 1920, 2048, 2880, 3840,
                                        4095, 4096, 8191, 8192]:
            i = np.arange(n, dtype=F32)
            rcp = F32(1.0) / F32(n)
            u = np.clip((i + F32(0.5)) * rcp, F32(0.0), F32(1.0)).astype(F32)
            idx = np.clip(np.floor(u * F32(n)).astype(np.int64), 0, n - 1)
            np.testing.assert_array_equal(idx, np.arange(n), err_msg="n = %d" % n)


# ---- fg_lite_interm_both: LDS-binned minima == one global atomic minimum per scatter ----------------------------------
EMPTY = 0x7FFFFFFF7FFFFFFF


def to_i64(hi, lo):
    v = (hi << 32) | lo
    return v - (1 << 64) if v >= (1 << 63) else v


class BinnedMinimumTest(unittest.TestCase):
    def run_case(self, rng, w1, h1, tile_ox, tile_oy, disp, footprint, hi_range):
        # sources of one 16 x 16 tile aimed at targets around origin + displacement (+ per-pixel noise)
        scatters = []
        for sy in range(16):
            for sx in range(16):
                x, y = tile_ox + sx, tile_oy + sy
                if x >= w1 or y >= h1:
                    continue
                dx = disp[0] + int(rng.integers(-3, 4))
                dy = disp[1] + int(rng.integers(-3, 4))
                hi = int(rng.integers(*hi_range))
                lo = int(rng.integers(0, 1 << 32))
                v = to_i64(hi, lo)
                for jj in range(footprint):
                    for ii in range(footprint):
                        tx, ty = x + dx + ii, y + dy + jj
                        if 0 <= tx < w1 and 0 <= ty < h1:
                            scatters.append((tx, ty, v))
        init = np.full(w1 * h1, EMPTY, dtype=np.int64)
        direct = init.copy()
        for tx, ty, v in scatters:
            direct[ty * w1 + tx] = min(int(direct[ty * w1 + tx]), v)
        # the kernel: window around the centre displacement, cells start empty, outside targets go straight to global
        wx0, wy0 = tile_ox + disp[0] - 8, tile_oy + disp[1] - 8
        cell = np.full(32 * 32, EMPTY, dtype=np.int64)
        binned = init.copy()
        for tx, ty, v in scatters:
            cx, cy = tx - wx0, ty - wy0
            if 0 <= cx < 32 and 0 <= cy < 32:
                cell[cy * 32 + cx] = min(int(cell[cy * 32 + cx]), v)
            else:
                binned[ty * w1 + tx] = min(int(binned[ty * w1 + tx]), v)
        for c in range(1024):
            if int(cell[c]) < EMPTY:
                gx, gy = wx0 + (c & 31), wy0 + (c >> 5)
                binned[gy * w1 + gx] = min(int(binned[gy * w1 + gx]), int(cell[c]))
        np.testing.assert_array_equal(direct, binned)
        return len(scatters)

    def test_random_tiles(self):
        rng = np.random.default_rng(11)
        n = 0
        for _ in range(40):
            w1, h1 = int(rng.integers(20, 90)), int(rng.integers(20, 90))
            ox, oy = int(rng.integers(0, w1)) // 16 * 16, int(rng.integers(0, h1)) // 16 * 16
            disp = (int(rng.integers(-30, 30)), int(rng.integers(-30, 30)))
            footprint = int(rng.integers(2, 6))
            # keys: ordinary positive float bit patterns, and values above / at the empty element
            hi_range = [(0x3F000000, 0x40000000), (0x7FFFFFF0, 0x7FFFFFFF), (0, 0x7FFFFFFF)][int(rng.integers(0, 3))]
            n += self.run_case(rng, w1, h1, ox, oy, disp, footprint, hi_range)
        self.assertGreater(n, 1000)

    def test_a_value_above_the_empty_element_never_wins(self):
        # hi == 0x7FFFFFFF and a low word above 0x7FFFFFFF is larger than the empty element as a signed 64-bit value
        v = to_i64(0x7FFFFFFF, 0xFFFFFFFF)
        self.assertGreater(v, EMPTY)
        self.assertEqual(min(EMPTY, v), EMPTY)


# =====================================================================================================================
# Wave 1B (the fuse2 levers). Each class below states what the Wave 1 kernels compute and what the fused kernel computes,
# on the same inputs, and compares them; the HLSL itself is compared against the GPU by test_fg_d3d12_e2e --gpu
# (--only turbo: every lever on against every lever off, and Wave 1B alone against Wave 1).
# =====================================================================================================================
import re
from pathlib import Path

DX12 = Path(__file__).resolve().parents[2] / "dx12"


def _hlsl(rel):
    return (DX12 / rel).read_text()


def _stripped(lines):
    return [l.strip() for l in lines if l.strip()]


def _function_body(text, signature):
    """The lines between the braces of the function that starts with `signature`."""
    start = text.index(signature)
    open_ = text.index("{", start)
    depth, i = 0, open_
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[open_ + 1:i].splitlines()


# ---- A13 / A6 / A14: the copied bodies stay copies ------------------------------------------------------------------------
class CopiedBodiesTest(unittest.TestCase):
    """The new kernels share a few lines of arithmetic with kernels whose DXIL had to stay byte-identical (so those
    kernels were not rewritten to call a shared function): the copies are compared line by line."""

    def test_weight_texel_is_fg_lite_weights(self):
        kernel = _stripped(_function_body(_hlsl("kernels/fg_lite_g.hlsl"), "WG_KERNEL(fg_lite_weights)"))
        helper = _stripped(_function_body(_hlsl("hlsl/fg_weights.hlsli"), "uint fg_weight_texel("))
        math_k = [l for l in kernel if l.startswith("const float") or l.startswith("const uint mi") or l.startswith("const uint ji")]
        math_h = [l for l in helper if l.startswith("const float") or l.startswith("const uint mi") or l.startswith("const uint ji")]
        self.assertEqual(len(math_k), 8)
        self.assertEqual(math_k, math_h)
        # the store: fg_st_h2(wout, idx, sigma(A1, A0), sigma(B1, B0)) is the packed pair the helper returns
        self.assertIn("fg_st_h2(wout, idx, fg_sigmoid(A1, A0), fg_sigmoid(B1, B0));", kernel)
        self.assertIn("return fg_pack_h2(fg_sigmoid(A1, A0), fg_sigmoid(B1, B0));", helper)

    def test_net1_pixel_is_fg_lite_net1_input(self):
        kernel = _function_body(_hlsl("kernels/fg_lite_net1io.hlsl"), "WG_KERNEL(fg_lite_net1_input)")
        k = _stripped(kernel)
        a = next(i for i, l in enumerate(k) if l.startswith("if (Y < H0 && x >= 0"))
        b = next(i for i, l in enumerate(k) if l.startswith("if (hwc) {") and i > a)
        inner_k = [l for l in k[a + 2:b - 1]]          # without the `{` and the closing `}` of the pixel block
        helper = _stripped(_function_body(_hlsl("hlsl/fg_net1_pixel.hlsli"), "void fg_net1_pixel("))
        c = next(i for i, l in enumerate(helper) if l.startswith("if (active && x >= 0"))
        inner_h = [l for l in helper[c + 2:-1] if l not in ("if (store_side)", "{", "}")]
        inner_k = [l for l in inner_k if l not in ("{", "}")]
        self.assertGreater(len(inner_k), 20)
        self.assertEqual(inner_k, inner_h)
        # fg_warp is the same function
        wk = _stripped(_function_body(_hlsl("kernels/fg_lite_net1io.hlsl"), "float fg_warp("))
        wh = _stripped(_function_body(_hlsl("hlsl/fg_net1_pixel.hlsli"), "float fg_warp("))
        self.assertEqual(wk, wh)

    def test_up_taps_is_the_second_half_of_fgf_up(self):
        full = _stripped(_function_body(_hlsl("hlsl/fg_full.hlsli"), "float4 fgf_up("))
        taps = _stripped(_function_body(_hlsl("hlsl/fg_full_up.hlsli"), "float4 fgf_up_taps("))
        z = next(i for i, l in enumerate(full) if l.startswith("const float z = fgf_zero()"))
        self.assertEqual(full[z:], taps)
        # the tap indices
        idx = _stripped(_function_body(_hlsl("hlsl/fg_full_up.hlsli"), "void fgf_up_idx("))
        self.assertEqual([l for l in full if l.startswith("const int px") or l.startswith("const int nx") or l.startswith("const int ny")],
                         [l for l in idx if l.startswith("const int px") or l.startswith("const int nx") or l.startswith("const int ny")])
        loads = " ".join(full)
        for expr in ("fgf_ld4h(lv, py * wc + px)", "fgf_ld4h(lv, py * wc + nx)", "fgf_ld4h(lv, ny * wc + px)", "fgf_ld4h(lv, ny * wc + nx)"):
            self.assertIn(expr, loads)
        self.assertEqual([l for l in idx if l.startswith("iP") or l.startswith("iX") or l.startswith("iY") or l.startswith("iD")],
                         ["iP = py * wc + px;", "iX = py * wc + nx;", "iY = ny * wc + px;", "iD = ny * wc + nx;"])


# ---- A13: the W tile ------------------------------------------------------------------------------------------------------
def _geometry(wo, ho):
    dp = 4 if ho > 2880 else 2
    return wo // dp, ho // dp


def _taps(n_out, n_grid):
    """x0 / x1 of the bilinear taps (fgf_tap / fg_read_linear) of every pixel of an axis of n_out pixels onto n_grid texels."""
    i = np.arange(n_out, dtype=np.float64)
    u = np.clip(((i + 0.5).astype(F32) / F32(n_out)).astype(F32), 0, 1).astype(F32)          # fgf_pixel_centre
    t = (u.astype(np.float64) * n_grid - 0.5).astype(F32)                                          # mad(u, n, -0.5)
    f = np.floor(t)
    r = np.clip(f.astype(np.int64), -1, n_grid - 1)
    return np.maximum(r, 0), np.minimum(r + 1, n_grid - 1)


class WTileBoundTest(unittest.TestCase):
    """The blends' LDS tile holds 12 x 12 (Full, 16 x 16 pixels) or 20 x 8 (light, 32 x 8 pixels) W texels: the taps of the
    pixels of one tile span no more, for the frame sizes the executor takes (the scatter grid is a half or a quarter)."""

    SIZES = [(1920, 1080), (2560, 1440), (3840, 2160), (1280, 720), (640, 360), (3440, 1440), (2560, 1080), (5120, 2880),
             (7680, 4320), (8192, 7680), (131, 89), (4, 4), (17, 33), (1001, 777), (4096, 2304)]

    def span(self, n_out, n_grid, tile):
        x0, x1 = _taps(n_out, n_grid)
        worst = 0
        for s in range(0, n_out, tile):
            e = min(s + tile, n_out)
            worst = max(worst, int(x1[s:e].max() - x0[s:e].min() + 1))
        return worst

    def test_full_blend_tile_is_12_by_12(self):
        for wo, ho in self.SIZES:
            w1, h1 = _geometry(wo, ho)
            if w1 < 1 or h1 < 1:
                continue
            self.assertLessEqual(self.span(wo, w1, 16), 12, "%dx%d columns" % (wo, ho))
            self.assertLessEqual(self.span(ho, h1, 16), 12, "%dx%d rows" % (wo, ho))

    def test_light_blend_tile_is_20_by_8(self):
        for wo, ho in self.SIZES:
            w1, h1 = _geometry(wo, ho)
            if w1 < 1 or h1 < 1:
                continue
            self.assertLessEqual(self.span(wo, w1, 32), 20, "%dx%d columns" % (wo, ho))
            self.assertLessEqual(self.span(ho, h1, 8), 8, "%dx%d rows" % (wo, ho))


# ---- A12: X0 / X1 as RGB ------------------------------------------------------------------------------------------------
class X3LayoutTest(unittest.TestCase):
    def test_pair_store_and_window_read(self):
        rng = np.random.default_rng(5)
        n = 64
        rgb = rng.integers(0, 65536, size=(n, 3), dtype=np.uint16)
        a = rng.integers(0, 65536, size=n, dtype=np.uint16)
        h_x = rgb[:, 0].astype(np.uint32) | (rgb[:, 1].astype(np.uint32) << 16)           # r | g << 16
        h_y = rgb[:, 2].astype(np.uint32) | (a.astype(np.uint32) << 16)                    # b | a << 16
        out = np.zeros(n * 3, dtype=np.uint16)
        words = np.zeros(n * 3 // 2, dtype=np.uint32)
        for e in range(0, n, 2):                                                           # the even lane of a Morton pair
            px, pa = (h_x[e], h_y[e]), (h_x[e + 1], h_y[e + 1])                            # own and partner
            at = e * 6 // 4
            words[at] = px[0]
            words[at + 1] = (px[1] & 0xFFFF) | ((pa[0] << 16) & 0xFFFFFFFF)
            words[at + 2] = (pa[0] >> 16) | ((pa[1] << 16) & 0xFFFFFFFF)
        raw = words.tobytes()
        np.testing.assert_array_equal(np.frombuffer(raw, dtype=np.uint16), rgb.reshape(-1))
        # fgp_ld3h: the 8 bytes from the dword at or below the texel's first byte hold its 6 bytes
        buf = np.frombuffer(raw + bytes(8), dtype=np.uint32)
        for i in range(n):
            at = i * 6
            d = buf[(at & ~3) // 4], buf[(at & ~3) // 4 + 1]
            hi = (at & 2) != 0
            rg = ((int(d[0]) >> 16) | (int(d[1]) << 16)) & 0xFFFFFFFF if hi else int(d[0])
            b = (int(d[1]) >> 16) if hi else (int(d[1]) & 0xFFFF)
            self.assertEqual((rg & 0xFFFF, rg >> 16, b), tuple(int(v) for v in rgb[i]))


# ---- A9: L14's input tile from the upwarp ---------------------------------------------------------------------------------
class UpwarpStageLayoutTest(unittest.TestCase):
    def test_four_groups_are_y_padded_to_32_channels(self):
        rng = np.random.default_rng(9)
        val = rng.integers(0, 65536, size=8, dtype=np.uint16)           # the 8 warped values
        U = rng.integers(0, 65536, size=8, dtype=np.uint16)             # A4 (4), A1, A3 (3)
        iav = int(rng.integers(0, 65536))
        hf = int(rng.integers(0, 65536))
        # fg_net2_upwarp's y: val, U.x U.y (A4), A1 | Ia << 16, A3[0] | A3[1] << 16, A3[2] | h(f) << 16
        y = list(val) + [U[0], U[1], U[2], U[3], U[4], iav, U[5], U[6], U[7], hf]
        self.assertEqual(len(y), 18)
        # the staged groups, as fg_stage_upwarp writes them (halves, low first)
        g0 = list(val)
        g1 = [U[0], U[1], U[2], U[3], U[4], iav, U[5], U[6]]
        g2 = [U[7], hf, 0, 0, 0, 0, 0, 0]
        g3 = [0] * 8
        self.assertEqual(g0 + g1 + g2 + g3, y + [0] * 14)


# ---- A14: the pyramid's tail and level 1 on the fly ---------------------------------------------------------------------
def _levels(wo, ho):
    w, h, levels = wo, ho, 1
    while levels < 12 and (levels < 9 or w > 1 or h > 1):
        w, h = (w + 1) // 2, (h + 1) // 2
        levels += 1
    sizes = [((wo + (1 << l) - 1) >> l, (ho + (1 << l) - 1) >> l) for l in range(levels)]
    return sizes, (8 if levels > 9 else 4)


def _tail_levels(sizes, lc, cap=3584):
    cells = nlev = 0
    for l in range(lc, 1, -1):
        if nlev >= 8:
            break
        cells += sizes[l][0] * sizes[l][1]
        if cells > cap:
            break
        nlev = lc - l + 1
    return nlev


F = np.float32


def _h(x):
    with np.errstate(all="ignore"):
        return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def _up_taps(eP, eX, eY, eD):
    """fgf_up_taps on float32 4-vectors (the same function for both pipelines: this test is about which texel feeds which)."""
    with np.errstate(all="ignore"):
        ws = F(0.0625) * eD[3] + (F(0.1875) * eY[3] + (F(0.1875) * eX[3] + F(0.5625) * eP[3]))
        vs = [F(0.0625) * (eD[c] * eD[3]) + (F(0.1875) * (eY[c] * eY[3]) + (F(0.1875) * (eX[c] * eX[3]) + F(0.5625) * (eP[c] * eP[3])))
              for c in range(3)]
        den = min(max(ws, F(1e-12)), F(1.0))
        return np.array([vs[0] / den, vs[1] / den, vs[2] / den, ws], dtype=np.float32)


def _up(lv, wc, hc, qx, qy):
    px, py = qx >> 1, qy >> 1
    nx = min(max(px + (1 if qx & 1 else -1), 0), wc - 1)
    ny = min(max(py + (1 if qy & 1 else -1), 0), hc - 1)
    return _up_taps(lv[py * wc + px], lv[py * wc + nx], lv[ny * wc + px], lv[ny * wc + nx])


def _push(own, coarse, wf, wc, hc):
    """fg_out_push in place: own where its weight is not <= 0 (NaN keeps it), else the upsample of the filled coarse level."""
    out = own.copy()
    for i in range(len(own)):
        if not (own[i][3] <= 0.0):
            continue
        out[i] = _h(_up(coarse, wc, hc, i % wf, i // wf))
    return out


class OutputTailTest(unittest.TestCase):
    SIZES = [(1920, 1080), (2560, 1440), (3840, 2160), (131, 89), (200, 150), (64, 48), (7680, 4320), (4, 4), (255, 255), (256, 256)]

    def test_tail_levels_fit_and_stop_above_level_1(self):
        for wo, ho in self.SIZES:
            sizes, lc = _levels(wo, ho)
            nlev = _tail_levels(sizes, lc)
            if nlev >= 2:
                lt = lc - nlev + 1
                self.assertGreaterEqual(lt, 2)
                self.assertLessEqual(sum(sizes[l][0] * sizes[l][1] for l in range(lt, lc + 1)), 3584)
                self.assertLessEqual(nlev, 8)

    def test_tail_and_level1_on_the_fly_equal_the_push_chain(self):
        rng = np.random.default_rng(3)
        for wo, ho in [(131, 89), (200, 150), (64, 48), (17, 33)]:
            sizes, lc = _levels(wo, ho)
            lv = []
            for (w, h) in sizes[:lc + 1]:
                a = np.zeros((w * h, 4), dtype=np.float32)
                a[:, :3] = _h(rng.random((w * h, 3)))
                wgt = rng.choice([0.0, 0.0, 0.25, 1.0, -0.5, np.nan], size=w * h)
                a[:, 3] = _h(wgt.astype(np.float32))
                lv.append(a)
            # Wave 1: push levels lc - 1 .. 1 in place, then the fine pass reads filled level 1
            old = [a.copy() for a in lv]
            for l in range(lc - 1, 0, -1):
                old[l] = _push(old[l], old[l + 1], sizes[l][0], sizes[l + 1][0], sizes[l + 1][1])
            # Wave 1B: tail over the coarse levels, pushes down to level 2, level 1 pushed on the fly at the taps
            new = [a.copy() for a in lv]
            nlev = _tail_levels(sizes, lc)
            first = lc - 1
            if nlev >= 2:
                for l in range(lc - 1, lc - nlev, -1):
                    new[l] = _push(new[l], new[l + 1], sizes[l][0], sizes[l + 1][0], sizes[l + 1][1])
                first = lc - nlev
            for l in range(first, 1, -1):
                new[l] = _push(new[l], new[l + 1], sizes[l][0], sizes[l + 1][0], sizes[l + 1][1])
            w1, h1 = sizes[1]
            w2, h2 = sizes[2]

            def texel1(i):
                if not (new[1][i][3] <= 0.0):
                    return new[1][i]
                return _h(_up(new[2], w2, h2, i % w1, i // w1))

            for qy in range(ho):
                for qx in range(wo):
                    px, py = qx >> 1, qy >> 1
                    nx = min(max(px + (1 if qx & 1 else -1), 0), w1 - 1)
                    ny = min(max(py + (1 if qy & 1 else -1), 0), h1 - 1)
                    got = _up_taps(texel1(py * w1 + px), texel1(py * w1 + nx), texel1(ny * w1 + px), texel1(ny * w1 + nx))
                    want = _up(old[1], w1, h1, qx, qy)
                    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32), err_msg="%dx%d (%d, %d)" % (wo, ho, qx, qy))


# ---- A7: a residual pair in one launch ----------------------------------------------------------------------------------
def _conv3(x, w, b):
    """3 x 3 convolution, zero padding, float32 accumulation, float16 result (the same function for both pipelines)."""
    H, W, _ = x.shape
    out = np.zeros((H, W, w.shape[0]), dtype=np.float32)
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    for ky in range(3):
        for kx in range(3):
            out += xp[ky:ky + H, kx:kx + W, :] @ w[:, :, ky, kx].T
    return _h(out + b)


def _post(v, skip=None):
    v = np.where(v < 0, _h(F(0.01) * v), v).astype(np.float32)            # leaky ReLU
    v = np.clip(v, -6, 6)
    return _h(v + F(0.0)) if skip is None else _h(skip + v)               # h(v + 0) / the residual add


def _fused_pair(x, wa, ba, wb, bb, tile_w=14, tile_h=6):
    H, W, C = x.shape
    out = np.zeros_like(x)
    covered = np.zeros((H, W), dtype=np.int32)
    for ty in range((H + tile_h - 1) // tile_h):
        for tx in range((W + tile_w - 1) // tile_w):
            ix, iy = tx * tile_w, ty * tile_h
            # the input tile of 18 x 10 pixels from (ix - 2, iy - 2), zero outside the grid
            tile = np.zeros((tile_h + 4, tile_w + 4, C), dtype=np.float32)
            for r in range(tile_h + 4):
                for c in range(tile_w + 4):
                    gx, gy = ix - 2 + c, iy - 2 + r
                    if 0 <= gx < W and 0 <= gy < H:
                        tile[r, c] = x[gy, gx]
            # first layer on the 16 x 8 region (pixel (c, r) = conv pixel (ix - 1 + c, iy - 1 + r)), zero outside the grid
            region = np.zeros((tile_h + 2, tile_w + 2, C), dtype=np.float32)
            a = _post(_conv3(np.pad(tile, ((0, 0), (0, 0), (0, 0))), wa, ba))
            for r in range(tile_h + 2):
                for c in range(tile_w + 2):
                    gx, gy = ix - 1 + c, iy - 1 + r
                    if 0 <= gx < W and 0 <= gy < H:
                        region[r, c] = a[1 + r, 1 + c]          # the tile conv output at tile pixel (c + 1, r + 1)
            # second layer on the interior from the region
            b = _conv3(region, wb, bb)
            for r in range(tile_h):
                for c in range(tile_w):
                    gx, gy = ix + c, iy + r
                    if gx < W and gy < H:
                        out[gy, gx] = _post(b[1 + r, 1 + c], x[gy, gx])
                        covered[gy, gx] += 1
    return out, covered


class PairFusionTest(unittest.TestCase):
    def test_fused_pair_equals_two_launches(self):
        rng = np.random.default_rng(21)
        C = 4
        for H, W in [(12, 18), (17, 30), (3, 5), (6, 14), (7, 15), (34, 60), (1, 1), (9, 2)]:
            x = _h(rng.normal(0, 1.5, size=(H, W, C)))
            wa = (rng.normal(0, 0.3, size=(C, C, 3, 3))).astype(np.float32)
            wb = (rng.normal(0, 0.3, size=(C, C, 3, 3))).astype(np.float32)
            ba, bb = _h(rng.normal(0, 0.5, size=C)), _h(rng.normal(0, 0.5, size=C))
            a = _post(_conv3(x, wa, ba))
            want = _post(_conv3(a, wb, bb), x)
            got, covered = _fused_pair(x, wa, ba, wb, bb)
            np.testing.assert_array_equal(got, want, err_msg="%dx%d" % (H, W))
            self.assertTrue((covered == 1).all(), "every pixel is an interior pixel of exactly one workgroup")

    def test_region_and_tile_indexing_of_the_kernel(self):
        # lane (n, row) of the region reads the input tile at (n + kx, row + ky) (18 columns, 10 rows); the second layer's
        # interior lane (n < 14, row < 6) reads the region at (n + kx, row + ky): both inside what is staged
        for n in range(16):
            for row in range(8):
                for ky in range(3):
                    for kx in range(3):
                        self.assertLess(n + kx, 18)
                        self.assertLess(row + ky, 10)
        for n in range(14):
            for row in range(6):
                for ky in range(3):
                    for kx in range(3):
                        self.assertLess(n + kx, 16)
                        self.assertLess(row + ky, 8)


if __name__ == "__main__":
    unittest.main()
