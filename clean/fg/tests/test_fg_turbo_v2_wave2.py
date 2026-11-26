# SPDX-License-Identifier: MIT
"""CPU proofs of the exactness arguments behind frame generation's Wave 2 kernels (the fuse2 bits 64 .. 1024):
A15 fg_lite_fill16_p2, A16 the bilinear row loads (FG_BIL128), A17 the light blend's 32 x 8 tiles with pair stores
(FG_TILE) and the output pull's division skip (FGP_POW2), A18 the z-sliced net2 convolutions (FG_ZB). Each class restates
what the Wave 1B kernel computes and what the Wave 2 kernel computes, as the HLSL does, and compares them on random and
edge inputs; the kernels themselves are checked on a GPU by test_fg_d3d12_e2e --gpu --only turbo (every lever on against
every lever off, default mask 2047)."""
import re
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

F32 = np.float32
DX12 = Path(__file__).resolve().parents[2] / "dx12"
RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def _text(rel):
    return (DX12 / rel).read_text()


def _kernel_lines(text):
    return [l for l in text.splitlines() if l.startswith("// @wg-kernel ")]


def _decl(line):
    m = re.match(r"// @wg-kernel (\S+) (.*)$", line)
    d = {"name": m.group(1)}
    for tok in m.group(2).split():
        k, v = tok.split("=", 1)
        d[k] = v
    return d


def _function_text(text, signature):
    start = text.index(signature)
    open_ = text.index("{", start)
    depth, i = 0, open_
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1


# ---- A15: fg_lite_fill16_p2 -----------------------------------------------------------------------------------------------
def fill_old(n16, period, p):
    """fg_lite_fill16: unit i = p[i % period]; each lane writes 8 units (one 16-byte store when all 8 are inside)."""
    out = np.zeros(n16, dtype=np.uint16)
    for lane in range((n16 + 7) // 8):
        first = lane * 8
        if first + 8 <= n16:
            for j in range(8):
                out[first + j] = p[(first + j) % period] & 0xFFFF
        else:
            for i in range(first, n16):
                out[i] = p[i % period] & 0xFFFF
    return out


def fill_new(n16, period, p):
    """fg_lite_fill16_p2: the same word for every full lane, the tail by i & (period - 1)."""
    u0 = p[0] & 0xFFFF
    u1 = (p[0] if period == 1 else p[1]) & 0xFFFF
    u2 = (p[2] if period == 4 else p[0]) & 0xFFFF
    u3 = (p[0] if period == 1 else (p[3] if period == 4 else p[1])) & 0xFFFF
    lo, hi = u0 | (u1 << 16), u2 | (u3 << 16)
    word = (lo, hi, lo, hi)
    out = np.zeros(n16, dtype=np.uint16)
    for lane in range((n16 + 7) // 8):
        first = lane * 8
        assert first % 8 == 0
        if first + 8 <= n16:
            for j in range(8):
                out[first + j] = (word[j // 2] >> (16 * (j % 2))) & 0xFFFF
        else:
            for i in range(first, n16):
                out[i] = p[i & (period - 1)] & 0xFFFF
    return out


class FillP2Test(unittest.TestCase):
    def test_same_units_for_every_period_and_size(self):
        rng = np.random.default_rng(15)
        for period in (1, 2, 4):
            for n16 in list(range(1, 40)) + [255, 256, 2047, 2048, 2049, 4097]:
                for _ in range(3):
                    p = [int(v) for v in rng.integers(0, 1 << 32, size=4, dtype=np.uint64)]   # the high bits are masked
                    np.testing.assert_array_equal(fill_old(n16, period, p), fill_new(n16, period, p),
                                                  err_msg="period %d n16 %d" % (period, n16))

    def test_the_runtime_patterns(self):
        # every Fill16 the executor records: clear {1.0}, clear {0,0,0,1}, a word as two units, fill64 (0xFFFF, 0x7FFF)
        for period, p in ((1, [0x3C00, 0, 0, 0]), (4, [0, 0, 0, 0x3C00]), (2, [0xFFFF, 0x7FFF, 0, 0]), (2, [0, 0, 0, 0]),
                          (1, [0x5A5A] * 4)):
            for n16 in (1, 2, 3, 7, 8, 9, 2457600, 1 << 20):
                if n16 > 5000:
                    n16 = n16 % 5000 + 3
                np.testing.assert_array_equal(fill_old(n16, period, p), fill_new(n16, period, p))

    def test_the_new_kernel_has_no_division(self):
        text = _text("kernels/fg_lite_util.hlsl")
        body = _function_text(text, "WG_KERNEL(fg_lite_fill16)")          # the first one is the FG_FILL_P2 branch
        self.assertIn("#if FG_FILL_P2", text)
        p2 = text[text.index("#if FG_FILL_P2"):text.index("#else", text.index("#if FG_FILL_P2"))]
        self.assertNotIn("%", p2.replace("// ", "//"), "no modulo in the FG_FILL_P2 kernel")
        self.assertNotIn(" / ", re.sub(r"//.*", "", p2))
        self.assertIn("fg_lite_fill16_p2", text)
        self.assertTrue(body)

    def test_first_is_a_multiple_of_eight(self):
        # first = (block * 256 + lane) * 8: the units of a lane start at a multiple of 8, so unit j of the lane has
        # i & (period - 1) == j & (period - 1) for every period that divides 8
        for first in range(0, 8 * 600, 8):
            for period in (1, 2, 4):
                for j in range(8):
                    self.assertEqual((first + j) % period, j & (period - 1))


# ---- A16: bilinear row loads ----------------------------------------------------------------------------------------------
def _tap_axis(t, n):
    """fgf_tap / fg_read_linear: x0, x1 for the texel position t on an axis of n texels."""
    x0f = np.floor(F32(t))
    xi = int(np.clip(int(x0f), -1, n - 1))
    return max(xi, 0), min(xi + 1, n - 1)


class BilinearRowLoadTest(unittest.TestCase):
    def test_second_texel_is_the_next_one_or_the_same(self):
        rng = np.random.default_rng(16)
        for n in (1, 2, 3, 5, 16, 17, 1920, 2560):
            ts = list(rng.uniform(-3.0, n + 3.0, size=400)) + [-0.5, 0.0, n - 1.0, n - 0.5, n - 1e-4, float(n), -1e-9]
            for t in ts:
                x0, x1 = _tap_axis(t, n)
                self.assertIn(x1 - x0, (0, 1), "n=%d t=%r" % (n, t))
                self.assertGreaterEqual(x0, 0)
                self.assertLessEqual(x1, n - 1)

    def test_row_load_gives_the_four_load_texels(self):
        # a buffer of 8-byte texels (W x H) with distinct values; the row load reads the texel at i00 and the one after it
        # (zero past the end of the buffer: robust buffer access), the second texel is the loaded one unless x1 == x0
        rng = np.random.default_rng(17)
        for W, H in ((1, 1), (1, 5), (2, 2), (5, 3), (17, 9), (64, 4)):
            tex = rng.integers(1, 1 << 63, size=W * H, dtype=np.uint64)

            def row_load(i):
                return int(tex[i]), (int(tex[i + 1]) if i + 1 < W * H else 0)

            for _ in range(300):
                tx, ty = rng.uniform(-2.5, W + 2.5), rng.uniform(-2.5, H + 2.5)
                x0, x1 = _tap_axis(tx, W)
                y0, y1 = _tap_axis(ty, H)
                i00, i10, i01, i11 = y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1
                a0, a1 = row_load(i00)
                b0, b1 = row_load(i01)
                got = (a0, a1 if i10 != i00 else a0, b0, b1 if i11 != i01 else b0)
                want = (int(tex[i00]), int(tex[i10]), int(tex[i01]), int(tex[i11]))
                self.assertEqual(got, want, "W=%d H=%d tap (%r, %r)" % (W, H, tx, ty))

    def test_two_half_texels_the_same_way(self):
        rng = np.random.default_rng(18)
        for W, H in ((1, 3), (2, 2), (9, 4), (33, 7)):
            tex = rng.integers(1, 1 << 31, size=W * H, dtype=np.uint64)
            for _ in range(300):
                tx, ty = rng.uniform(-2.5, W + 2.5), rng.uniform(-2.5, H + 2.5)
                x0, x1 = _tap_axis(tx, W)
                y0, y1 = _tap_axis(ty, H)
                i00, i10, i01, i11 = y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1
                a = (int(tex[i00]), int(tex[i00 + 1]) if i00 + 1 < W * H else 0)
                b = (int(tex[i01]), int(tex[i01 + 1]) if i01 + 1 < W * H else 0)
                got = (a[0], a[1] if i10 != i00 else a[0], b[0], b[1] if i11 != i01 else b[0])
                self.assertEqual(got, (int(tex[i00]), int(tex[i10]), int(tex[i01]), int(tex[i11])))

    def test_blend_arithmetic_is_untouched(self):
        # the mix is the same call on the same four values; the _b code only changes where the values come from
        full = _text("hlsl/fg_full.hlsli")
        self.assertEqual(full.count("return fgf_mix4(t, fgf_unpack4h(a.xy), fgf_unpack4h(c10), fgf_unpack4h(b.xy), fgf_unpack4h(c11));"), 1)
        self.assertEqual(full.count("return fgf_mix2(t, fgf_unpack2h(a.x), fgf_unpack2h(c10), fgf_unpack2h(b.x), fgf_unpack2h(c11));"), 1)
        lite = _text("hlsl/fg_lite_common.hlsli")
        tail = lite[lite.index("float4 fg_read_linear("):]
        self.assertIn("float4 r0 = mad(ax.xxxx, t10 - t00, t00);", tail)
        self.assertIn("return mad(ay.xxxx, r1 - r0, r0);", tail)

    def test_every_variant_is_declared_and_defaults_to_off(self):
        for rel, base, variants in (
            ("kernels/fg_lite_blend.hlsl", "fg_lite_blend", ("fg_lite_blend_b", "fg_lite_blend_pack_b", "fg_lite_blend_bt", "fg_lite_blend_pack_bt")),
            ("kernels/fg_full_blend.hlsl", "fg_full_blend", ("fg_full_blend_b", "fg_full_blend_merge_b")),
            ("kernels/fg_out_pull.hlsl", "fg_out_pull", ("fg_out_pull_b", "fg_out_pull_pack_b", "fg_out_pull_bq", "fg_out_pull_pack_bq")),
            ("kernels/fg_lite_g.hlsl", "fg_lite_weights", ("fg_lite_weights_b",)),
        ):
            decls = {d["name"]: d for d in map(_decl, _kernel_lines(_text(rel)))}
            for v in variants:
                self.assertIn(v, decls)
                self.assertIn("FG_BIL128=1", decls[v]["defines"])
            self.assertNotIn("FG_BIL128", decls[base].get("defines", ""))
        self.assertIn("#define FG_BIL128 0", _text("hlsl/fg_full.hlsli"))
        self.assertIn("#define FG_BIL128 0", _text("hlsl/fg_lite_common.hlsli"))


# ---- A17: 32 x 8 tiles with pair stores -----------------------------------------------------------------------------------
def _tile_map(wo, ho):
    tiles_x = (wo + 31) >> 5
    n_tiles = tiles_x * ((ho + 7) // 8)
    seen = {}
    stores = {}
    for tile in range(n_tiles):
        tile_y = tile // tiles_x
        for lane in range(256):
            xr = ((tile - tile_y * tiles_x) << 5) + (lane & 31)
            yr = (tile_y << 3) + (lane >> 5)
            live = xr < wo and yr < ho
            x, y = min(xr, wo - 1), min(yr, ho - 1)
            if live:
                seen[(x, y)] = seen.get((x, y), 0) + 1
            # the pair store: the even lane of a quad pair stores both pixels (its partner is the next lane, x + 1)
            if live and (xr & 1) == 0:
                if x + 1 < wo:
                    stores[(x, y)] = stores.get((x, y), 0) + 1
                    stores[(x + 1, y)] = stores.get((x + 1, y), 0) + 1
                else:
                    stores[(x, y)] = stores.get((x, y), 0) + 1
    return seen, stores


class LiteBlendTileTest(unittest.TestCase):
    def test_every_pixel_once_and_stored_once(self):
        for wo, ho in ((1, 1), (2, 3), (31, 9), (32, 8), (33, 17), (64, 8), (100, 37), (127, 5), (640, 360)):
            seen, stores = _tile_map(wo, ho)
            want = {(x, y) for x in range(wo) for y in range(ho)}
            self.assertEqual(set(seen), want, "%dx%d covered" % (wo, ho))
            self.assertTrue(all(v == 1 for v in seen.values()), "%dx%d no pixel twice" % (wo, ho))
            self.assertEqual(set(stores), want, "%dx%d stored" % (wo, ho))
            self.assertTrue(all(v == 1 for v in stores.values()), "%dx%d no pixel stored twice" % (wo, ho))

    def test_the_grid_is_not_smaller_than_the_scanline_grid(self):
        for wo, ho in ((2560, 1440), (1920, 1080), (3840, 2160), (1366, 768), (7, 3)):
            tiles = ((wo + 31) >> 5) * ((ho + 7) // 8)
            self.assertGreaterEqual(tiles * 256, wo * ho)
        self.assertEqual(((2560 + 31) >> 5) * ((1440 + 7) // 8), (2560 * 1440 + 255) // 256)   # 14400 both ways at 1440p

    def test_a_quad_pair_is_a_horizontal_pair(self):
        # a wave is one tile row (lane & 31 = x offset, lane >> 5 = row) and QuadReadAcrossX swaps lane with lane ^ 1
        for lane in range(256):
            partner = lane ^ 1
            self.assertEqual(lane >> 5, partner >> 5)
            self.assertEqual((lane & 31) ^ 1, partner & 31)

    def test_the_kernel_is_declared_and_division_free_per_pixel(self):
        decls = {d["name"]: d for d in map(_decl, _kernel_lines(_text("kernels/fg_lite_blend.hlsl")))}
        for v in ("fg_lite_blend_t", "fg_lite_blend_pack_t", "fg_lite_blend_bt", "fg_lite_blend_pack_bt"):
            self.assertIn("FG_TILE=1", decls[v]["defines"])
        text = _text("kernels/fg_lite_blend.hlsl")
        tile = text[text.index("#elif FG_TILE"):text.index("#else", text.index("#elif FG_TILE"))]
        self.assertEqual(len(re.findall(r"(?<![/])/(?![/])", re.sub(r"//.*", "", tile))), 1, "one division, per workgroup (tile / tiles_x)")
        self.assertNotIn("%", re.sub(r"//.*", "", tile))


# ---- A17: the output pull's division on a power of two ----------------------------------------------------------------------
def _round32(fr):
    """Round an exact rational to float32 (nearest even, subnormals kept)."""
    if fr == 0:
        return F32(0.0)
    neg = fr < 0
    a = abs(fr)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    elif Fraction(2) ** (e + 1) <= a:
        e += 1
    e = max(e, -126)
    q = Fraction(2) ** (e - 23)
    n = a / q
    k = n.numerator // n.denominator
    rem = n - k
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and k % 2 == 1):
        k += 1
    v = float(k * q)
    return F32(-v if neg else v)


def _fma(a, b, c):
    a, b, c = float(a), float(b), float(c)
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        neg = (np.signbit(a) != np.signbit(b)) and np.signbit(c) and (a * b == 0)   # -0 only for (-0) + (-0)
        return F32(-0.0) if neg else F32(0.0)
    return _round32(exact)


def _mul(a, b):
    return _fma(a, b, -0.0)


def _rcp(b):
    return _round32(Fraction(1) / Fraction(float(b)))     # v_rcp_f32 is exact on powers of two


def div_reciprocal(b):
    y = _rcp(b)
    return _fma(_fma(-b, y, 1.0), y, y)


def div_normal(a, b, y):
    q = _mul(a, y)
    q = _fma(_fma(-b, q, a), y, q)
    return _fma(_fma(-b, q, a), y, q)


def level_old(a, w, m):
    den = max(F32(w), F32(1.0))
    rgb = div_normal(a, den, div_reciprocal(den))
    return a if a == 0 else rgb


def level_new(a, w, m):
    scale = F32(np.uint32((127 - 2 * m) << 23).view(F32))
    rgb = _mul(a, scale)
    if F32(w) != F32(1 << (2 * m)):
        raise AssertionError("the division branch is the old code")
    return a if a == 0 else rgb


class PullPow2Test(unittest.TestCase):
    def test_division_by_a_power_of_two_is_the_scale(self):
        rng = np.random.default_rng(19)
        samples = [F32(0.0), F32(-0.0), F32(1.0), F32(0.5), F32(1e-30), F32(1e-38), F32(1.4e-45), F32(1.17549435e-38),
                   F32(0.9999999), F32(3.0e-39)]
        samples += [F32(v) for v in rng.uniform(0.0, 1.0, size=200)]
        samples += [F32(v) for v in 10.0 ** rng.uniform(-45, 0, size=200)]
        samples += [F32(v) * F32(16) for v in rng.uniform(0.0, 1.0, size=50)]   # sums of 16 saturated values
        for m in range(0, 5):
            w = 1 << (2 * m)
            for a in samples:
                old = level_old(a, w, m)
                new = level_new(a, w, m)
                self.assertEqual(np.asarray(old, dtype=F32).view(np.uint32), np.asarray(new, dtype=F32).view(np.uint32),
                                 "m=%d a=%r" % (m, a))

    def test_the_condition_is_exactly_the_power_of_two(self):
        # den = max(w, 1) is w itself for w = 4^m, so the reciprocal is exact; any other weight takes the division
        for m in range(0, 5):
            self.assertEqual(max(F32(1 << (2 * m)), F32(1.0)), F32(1 << (2 * m)))
            self.assertEqual(div_reciprocal(F32(1 << (2 * m))), F32(2.0 ** (-2 * m)))

    def test_declared_and_off_by_default(self):
        decls = {d["name"]: d for d in map(_decl, _kernel_lines(_text("kernels/fg_out_pull.hlsl")))}
        for v in ("fg_out_pull_q", "fg_out_pull_pack_q", "fg_out_pull_bq", "fg_out_pull_pack_bq"):
            self.assertIn("FGP_POW2=1", decls[v]["defines"])
        self.assertNotIn("FGP_POW2", decls["fg_out_pull"].get("defines", ""))
        self.assertNotIn("FGP_POW2", decls["fg_out_pull_pack"].get("defines", ""))
        self.assertIn("#define FGP_POW2 0", _text("kernels/fg_out_pull.hlsl"))


# ---- A18: z-sliced net2 convolutions ----------------------------------------------------------------------------------------
class ZSliceTest(unittest.TestCase):
    def setUp(self):
        self.decls = [_decl(l) for l in _kernel_lines(_text("kernels/fg_net2_mconv.hlsl"))]

    def test_every_plain_convolution_has_a_z_twin(self):
        base = {d["name"]: d for d in self.decls if d.get("entry") == "fg_net2_mconv" and "FG_PAIR" not in d.get("defines", "")
                and "FG_UPWARP_STAGE" not in d.get("defines", "") and not d["name"].endswith("_z")}
        z = {d["name"]: d for d in self.decls if d["name"].endswith("_z")}
        for name, d in base.items():
            if name == "fg_net2_mconv_h3b2u":    # not launched by the executor
                continue
            self.assertIn(name + "_z", z, name)
            t = z[name + "_z"]
            self.assertEqual(t["args"], d["args"] + ",u,u")        # zstride, zmask
            self.assertEqual(t["block"], d["block"])
            self.assertEqual(t["defines"], d["defines"] + ",FG_ZB=1,WG_Z_GRID=1")
        # the originals (and what their checkers launch) are unchanged
        for name, d in base.items():
            self.assertNotIn("FG_ZB", d.get("defines", ""))
            self.assertNotIn("WG_Z_GRID", d.get("defines", ""))

    def test_the_pointer_order_matches_the_mask_bits(self):
        text = _text("kernels/fg_net2_mconv.hlsl")
        decl = re.search(r"uint a_x, a_wf, a_b, a_y, a_skip, a_y1, a_y2, a_skip1, a_skip2;", text)
        self.assertTrue(decl)
        setup = text[text.index("#define FG_ZB_SETUP()"):text.index("#else", text.index("#define FG_ZB_SETUP()"))]
        order = ["x", "wf", "b", "y", "skip", "y1", "y2", "skip1", "skip2"]
        for bit, name in enumerate(order):
            self.assertIn("%s = a_%s + ((zmask & %du) != 0u ? zo_ : 0u)" % (name, name, 1 << bit), setup)

    def test_the_executor_maps_every_executor_variant(self):
        cpp = (RUNTIME / "fg_d3d12_executor.cpp").read_text()
        for v in ("m1b1", "m1b2", "m2b1", "m2b2", "m2b4", "m1b4s", "m1b2s", "m1b1u", "m1b2u", "m2b4u", "m3b2u", "m3b4u"):
            self.assertIn('"fg_net2_mconv_%s_z"' % v, cpp)
            self.assertIn("case kN2%s: return kN2z%s;" % (v, v), cpp)
        self.assertIn('"fg_net2_mconv_h3b1u_z"', cpp)
        self.assertIn('"fg_net2_mconv_h3b1u_s3_z"', cpp)

    def test_slice_addressing_is_the_frames_own_bank(self):
        # frame f of a batch records into bank f (stride bytes after bank f - 1), the slice adds z * stride to the banked
        # pointers: the address is the frame's own, in 32-bit arithmetic like the shader's
        stride = 0x1200000
        for f in range(5):
            for ptr in (0x400, 0x7FFFF00, 0xEF000000):
                own = (ptr + f * stride) & 0xFFFFFFFF
                z = f
                self.assertEqual(own, (ptr + z * stride) & 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
