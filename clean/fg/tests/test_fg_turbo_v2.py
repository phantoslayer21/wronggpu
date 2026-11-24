# SPDX-License-Identifier: MIT
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


if __name__ == "__main__":
    unittest.main()
