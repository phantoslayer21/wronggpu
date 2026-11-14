# SPDX-License-Identifier: MIT
"""Geometry of the true-fusion kernels post_tf* (kernels/post.hip, post_tf_body), simulated on the host (no GPU):
for many output sizes and chunk choices, every D pixel's full history is written by exactly one workgroup and every
reduced-history code by exactly one lane; every row a vertical pass reads (3p - 2 ... 3p + 4, clamped) was
reconstructed in an earlier or the same step of the same workgroup and is still in its ring slot (not overwritten
by a later row); the mirrored ring slots and the wrap-free reads agree; the staged footprint of every step fits the
cap whenever tf_supported says so; and every output pixel is written exactly once.

Run: python -m unittest clean.tests.test_post_tf_geometry
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clean.runner import post as PS  # noqa: E402


def cdiv(a, b):
    return -(-a // b)


def walk(ow, oh, rw, rh, cpairs, k1=1 << 30, c2=None, nw=8, cap=352):
    """Simulates post_tf_body's control flow; returns (history owner counts, reduced owner counts, output counts,
    max staged texels). Raises AssertionError on a ring or ordering violation."""
    import numpy as np
    c2 = c2 or cpairs
    dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
    rx, ry = float(np.float32(rw / dw)), float(np.float32(rh / dh))
    pos_x = lambda x: float(np.float32(np.float32(np.float32(x) + np.float32(0.5)) * np.float32(rx)))  # noqa: E731
    pos_y = lambda y: float(np.float32(np.float32(np.float32(y) + np.float32(0.5)) * np.float32(ry)))  # noqa: E731
    import math
    fl = lambda v: int(math.floor(float(np.float32(np.float32(v) - np.float32(0.5)))))  # noqa: E731
    npo = cdiv(oh, 2)
    strips = cdiv(ow, 40)
    nchunks = (k1 + cdiv(npo - k1 * cpairs, c2)) if k1 * cpairs < npo else cdiv(npo, cpairs)
    ring_n = 2 * nw + 6
    hist = np.zeros((dh, dw), np.int32)
    red = np.zeros((dh // 2, dw // 2), np.int32)
    out = np.zeros((oh, ow), np.int32)
    max_stage = 0
    for chunk in range(nchunks):
        csize = cpairs if chunk < k1 else c2
        P0 = chunk * cpairs if chunk < k1 else k1 * cpairs + (chunk - k1) * c2
        if P0 >= npo:
            continue
        p_end = min(P0 + csize, npo)
        own_r0, own_r1 = 3 * P0, min(3 * (P0 + csize), dh)
        r_first = max(3 * P0 - 2, 0)
        assert r_first % 2 == 0
        r_need = min(3 * (p_end - 1) + 4, dh - 1)
        for strip in range(strips):
            X0 = 60 * strip
            c0 = fl(pos_x(min(max(X0 - 2, 0), dw - 1))) - 1
            c1 = fl(pos_x(min(max(X0 + 61, 0), dw - 1))) + 1
            ncol = c1 - c0 + 1
            slot_row = {}  # ring slot -> D row it holds
            p_next = P0
            y_step = r_first
            while y_step <= r_need:
                yl = min(y_step + 2 * (nw - 1), r_need) & ~1
                nrow = fl(pos_y(yl)) + 2 - (fl(pos_y(y_step)) - 1) + 1
                max_stage = max(max_stage, ncol * nrow)
                for w in range(nw):
                    yv = y_step + 2 * w
                    if yv > r_need:
                        continue
                    for k in range(2):
                        y = yv + k
                        if y >= dh:
                            continue
                        slot_row[(y - r_first) % ring_n] = y
                        if own_r0 <= y < own_r1:
                            hist[y, X0:min(X0 + 60, dw)] += 1
                    if yv + 1 < dh and own_r0 <= yv and yv + 1 < own_r1:
                        red[yv // 2, X0 // 2:min(X0 + 60, dw - (dw & 1)) // 2] += 1
                r_done = min(y_step + 2 * nw, r_need + 1)
                p_rdy = p_end if r_done > r_need else min(p_end, (r_done - 5) // 3 + 1)
                assert p_rdy >= p_next
                interior = 3 * p_next - 2 >= 0 and 3 * p_rdy + 1 <= dh - 1
                s0 = (3 * p_next - 2 - r_first) % ring_n if interior else 0
                for pp in range(p_next, p_rdy):
                    for dy in range(2):
                        y = 2 * pp + dy
                        if y >= oh:
                            continue
                        for kk in range(5):
                            row = min(max(3 * pp - 2 + 2 * dy + kk, 0), dh - 1)
                            assert row < r_done, (ow, oh, pp, row, r_done)
                            sl = (row - r_first) % ring_n
                            assert slot_row.get(sl) == row, ("ring slot overwritten", ow, oh, pp, row)
                            if interior:   # wrap-free read through the mirrored slots
                                u = (s0 + 3 * (pp - p_next) + 2 * dy) % ring_n
                                assert u + kk < ring_n + 6 and (u + kk) % ring_n == sl
                        out[y, 40 * strip:min(40 * strip + 40, ow)] += 1
                p_next = p_rdy
                y_step += 2 * nw
            assert p_next == p_end
    return hist, red, out, max_stage


class TestTfGeometry(unittest.TestCase):
    SIZES = [(1920, 1080, 1.5), (2560, 1440, 1.5), (3840, 2160, 2.0), (1920, 1080, 3.0), (2560, 1440, 1.7),
             (1913, 1077, 1.92), (101, 57, 1.58), (1000, 600, 1.5), (2560, 1440, 1.0), (63, 41, 1.0), (40, 2, 1.5),
             (41, 3, 1.5), (1281, 723, 2.2)]

    def test_coverage_and_ring(self):
        for ow, oh, ratio in self.SIZES:
            rw, rh = max(1, int(ow / ratio + 0.5)), max(1, int(oh / ratio + 0.5))
            dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
            for cpairs, k1, c2 in ((68, 1 << 30, None), (36, 1 << 30, None), (20, 1 << 30, None), (2, 1 << 30, None),
                                   (68, 3, 20), (52, 0, 20), (68, 1, 2)):
                with self.subTest(O=(ow, oh), R=(rw, rh), C=cpairs, k1=k1, c2=c2):
                    hist, red, out, stage = walk(ow, oh, rw, rh, cpairs, k1, c2)
                    self.assertTrue((hist == 1).all(), "every D pixel's history written exactly once")
                    self.assertTrue((red == 1).all(), "every reduced code written exactly once")
                    self.assertTrue((out == 1).all(), "every output pixel written exactly once")
                    for cap, kw in ((352, {}), (644, {"cap": 644})):
                        if PS.tf_supported(rw, rh, dw, dh, 8, kw.get("cap")):
                            self.assertLessEqual(stage, cap)

    # Brief 15: the D3D12 kernels post_tf4_dl_* (NW = 4, cap 448: native AA and every ratio between 1.0 and 1.5;
    # dx12/kernels/post.hlsl, runtime.cpp tf4_supported)
    SIZES4 = [(1920, 1080, 1.0), (2560, 1440, 1.0), (1281, 721, 1.0), (1920, 1080, 1.25), (2560, 1440, 1.33),
              (1920, 1080, 1.5), (3840, 2160, 1.0), (101, 57, 1.0), (63, 41, 1.1), (64, 36, 1.0), (1001, 563, 1.4),
              (41, 3, 1.0), (40, 2, 1.0), (1913, 1077, 1.2), (1280, 720, 1.5)]

    def test_coverage_and_ring_nw4(self):
        for ow, oh, ratio in self.SIZES4:
            rw, rh = max(1, int(ow / ratio + 0.5)), max(1, int(oh / ratio + 0.5))
            dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
            for cpairs, k1, c2 in ((68, 1 << 30, None), (36, 1 << 30, None), (2, 1 << 30, None), (68, 3, 20),
                                   (68, 1, 2)):
                with self.subTest(O=(ow, oh), R=(rw, rh), C=cpairs, k1=k1, c2=c2):
                    hist, red, out, stage = walk(ow, oh, rw, rh, cpairs, k1, c2, nw=4, cap=PS.TF4_DL_CAP)
                    self.assertTrue((hist == 1).all(), "every D pixel's history written exactly once")
                    self.assertTrue((red == 1).all(), "every reduced code written exactly once")
                    self.assertTrue((out == 1).all(), "every output pixel written exactly once")
                    if PS.tf_supported(rw, rh, dw, dh, 4, PS.TF4_DL_CAP):
                        self.assertLessEqual(stage, PS.TF4_DL_CAP)

    def test_nw4_takes_every_ratio_from_1_to_1_5(self):
        """Every size with render <= output on both axes that the banded path takes (recon2_supported) and that the
        cap-352 kernel refuses is taken by post_tf4_dl_* (the host bound), for every output size up to 4096."""
        import random
        rng = random.Random(15)
        outs = [(o, o) for o in range(4, 400)] + [(rng.randint(4, 4096), rng.randint(4, 4096)) for _ in range(3000)]
        outs += [(1920, 1080), (2560, 1440), (3840, 2160), (1281, 721), (3440, 1440)]
        n = 0
        for ow, oh in outs:
            dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
            lo_w, lo_h = int(ow / 1.5), int(oh / 1.5)
            for rw in sorted({ow, lo_w, lo_w + 1, (ow + lo_w) // 2, max(1, ow - 1)}):
                for rh in sorted({oh, lo_h, lo_h + 1, (oh + lo_h) // 2, max(1, oh - 1)}):
                    if not PS.recon2_supported(rw, rh, dw, dh) or PS.tf_supported(rw, rh, dw, dh):
                        continue
                    n += 1
                    self.assertTrue(PS.tf_supported(rw, rh, dw, dh, 4, PS.TF4_DL_CAP), (ow, oh, rw, rh))
        self.assertGreater(n, 1000)

    def test_nw4_declarations_agree(self):
        """runtime.cpp's tf4_supported (cap, NW = 4 footprint), runner/post.py's TF4_DL_CAP and the four post_tf4_dl_*
        kernels of dx12/kernels/post.hlsl (NW = 4, the same cap, 256 lanes) say the same. A name may be declared once per
        build it ports (the original filter's post_w64__orig.co, the analytic filter's post_w64.co): every declaration
        line is checked."""
        import re
        clean = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rt = open(os.path.join(clean, "runtime", "runtime.cpp"), encoding="utf-8").read()
        self.assertIn("constexpr int kTf4Cap = %d;" % PS.TF4_DL_CAP, rt)
        self.assertIn("((long long)(63.0 * rw / dw) + 4) * ((long long)(6.0 * rh / dh) + 5) <= kTf4Cap", rt)
        hlsl = open(os.path.join(clean, "dx12", "kernels", "post.hlsl"), encoding="utf-8").read()
        decl = re.findall(r"(?m)^// @wg-kernel (\S*post_tf4_dl\S*) .*$", hlsl)
        self.assertEqual(sorted(set(decl)), ["pmo/post_tf4_dl_f32_p12_mo", "pmo/post_tf4_dl_f32_p12_mo_direct",
                                        "w64/post_tf4_dl_f32_p12_diet_vsit", "w64/post_tf4_dl_f32_p12_diet_vsit_direct"])
        for line in re.findall(r"(?m)^// @wg-kernel \S*post_tf4_dl\S* .*$", hlsl):
            self.assertIn(" block=256 ", line)
            self.assertIn("WG_TF_NW=4,WG_TF_CAP=%d" % PS.TF4_DL_CAP, line)
        # 448 x 48 bytes of staged texels + (2 NW + 6 + 6) ring rows of 480 bytes fit D3D12's 32 KB of groupshared memory
        self.assertLessEqual(PS.TF4_DL_CAP * 48 + (2 * 4 + 6 + 6) * 480, 32768)


if __name__ == "__main__":
    unittest.main()
