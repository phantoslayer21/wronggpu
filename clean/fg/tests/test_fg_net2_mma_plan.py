# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (#96): net2's fast path (clean/fg/runner/fg_net2_mma.py, clean/dx12/kernels/fg_net2_mconv.hlsl) as
planned on the CPU: the weight swizzle and padded bias, the kernel variant per layer, the fast chain's launches against
the kernel declarations and buffers, and a CPU model of the kernel's addressing (A fragments read back from the swizzled
bytes, B fragments gathered with the kernel's channel masks and zero border, chunks in the kernel's order with exact
sums) that must give the CPU reference's convolution bit for bit. No GPU.

    python -m unittest clean.fg.tests.test_fg_net2_mma_plan
"""
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "fg", "runner"))

import fg_net2_d3d12_check as K  # noqa: E402
import fg_net2_mma as M  # noqa: E402
from clean.fg import layout as L  # noqa: E402
from clean.fg.reference import net2 as N  # noqa: E402
from clean.fg.reference.fp16acc import acc_matmul16  # noqa: E402

KDIR = os.path.join(ROOT, "clean", "dx12", "kernels")


def declarations():
    out = {}
    for fn in ("fg_net2_mconv.hlsl", "fg_net2_upwarp.hlsl") + tuple(k + ".hlsl" for k in K.KERNELS):
        with open(os.path.join(KDIR, fn), encoding="utf-8") as f:
            for m in re.finditer(r"@wg-kernel\s+(\S+)\s+co=(\S+)(.*)", f.read()):
                kv = dict(x.split("=", 1) for x in m.group(3).split() if "=" in x)
                out[m.group(1)] = (m.group(2), kv["args"].split(","), kv.get("defines", ""))
    return out


def model_conv(x, in_co, cin, H, W, wbytes, bbytes, cout):
    """The fast kernel's reads with exact chunk sums: x [H, W, in_cs] float16 values; returns [H, W, cout]."""
    in_cs = x.shape[2]
    tiles = -(-cout // 16)
    wf = np.frombuffer(wbytes, np.float16).astype(np.float64)
    nb = wf.size // (tiles * 9 * 32 * 8)
    wf = wf.reshape(tiles, 9, nb, 32, 8)
    weff = np.zeros((16 * tiles, 9, 16 * nb))
    for mt in range(tiles):
        for lane in range(32):
            o, c0 = 16 * mt + lane % 16, 8 * (lane // 16)
            for cb in range(nb):
                weff[o, :, 16 * cb + c0:16 * cb + c0 + 8] = wf[mt, :, cb, lane, :]
    bias = np.frombuffer(bbytes, np.float16).astype(np.float64)
    flat = x.reshape(-1)
    acc = np.broadcast_to(bias[None, :], (H * W, 16 * tiles)).copy()
    yy, xx = np.divmod(np.arange(H * W), W)
    for t in range(9):
        ky, kx = divmod(t, 3)
        sy, sx = yy + ky - 1, xx + kx - 1
        inb = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
        e = np.where(inb, (sy * W + sx) * in_cs + in_co, 0)
        for cb in range(nb):
            c = 16 * cb + np.arange(16)
            idx = np.minimum(e[:, None] + c[None, :], flat.size - 1)
            xs = np.where(inb[:, None] & (c[None, :] < cin), flat[idx], 0.0)
            acc = acc_matmul16(acc, xs, weff[:, t, 16 * cb:16 * cb + 16].T)
    return acc[:, :cout].reshape(H, W, cout)


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.p = K.synthetic_params(rng)
        cls.ib, cls.ic, cls.ia = (K._rand(rng, (32, 48, c), 8.0) for c in (4, 4, 1))
        cls.case = M.fast_chain_case("plan fast 48x32", 48, 32, cls.p, cls.ib, cls.ic, cls.ia, np.float32(0.5))
        cls.exact = K.chain_case("plan exact 48x32", 48, 32, cls.p, cls.ib, cls.ic, cls.ia, np.float32(0.5))
        cls.decl = declarations()

    def test_swizzle_layout(self):
        rng = np.random.default_rng(3)
        for cout, cw in ((8, 16), (16, 18), (32, 10), (48, 32), (96, 64)):
            w = K.f16(rng.uniform(-1, 1, (cout, 3, 3, cw)))
            a = np.frombuffer(M.swizzle(w), np.float16)
            nb, tiles = M.blocks(cw), -(-cout // 16)
            self.assertEqual(a.size, tiles * 9 * nb * 32 * 8)
            a = a.reshape(tiles, 9, nb, 32, 8)
            for _ in range(200):
                mt, t, cb, lane, j = (int(rng.integers(n)) for n in (tiles, 9, nb, 32, 8))
                o, c = 16 * mt + lane % 16, 16 * cb + 8 * (lane // 16) + j
                want = w[o, t // 3, t % 3, c] if o < cout and c < cw else 0.0
                self.assertEqual(float(a[mt, t, cb, lane, j]), want)
            b = np.frombuffer(M.bias_bytes(np.arange(cout)), np.float16)
            self.assertEqual(b.size, 16 * tiles)
            self.assertTrue(np.all(b[:cout] == np.arange(cout)) and np.all(b[cout:] == 0))

    def test_variants(self):
        def pick(l, up=0, pixels=None):
            return M.variant(l.out, l.cin, up, pixels)[0]
        n = {l.name: l for l in L.NET2}
        self.assertEqual(pick(n["net2_01"], pixels=480 * 272), "fg_net2_mconv_m2b1")
        self.assertEqual(pick(n["net2_04"], pixels=60 * 34), "fg_net2_mconv_m1b4s")
        self.assertEqual(pick(n["net2_04"], pixels=108 * 46), "fg_net2_mconv_m1b4s")
        self.assertEqual(pick(n["net2_04"], pixels=1 << 20), "fg_net2_mconv_m2b4")
        self.assertEqual(pick(n["net2_16"], pixels=160 * 92), "fg_net2_mconv_m1b2s")
        self.assertEqual(pick(n["net2_16"], pixels=216 * 92), "fg_net2_mconv_m2b2")
        self.assertEqual(pick(n["net2_12"], up=1), "fg_net2_mconv_m3b4u")
        self.assertEqual(pick(n["net2_13"], up=1), "fg_net2_mconv_m1b2u")
        self.assertEqual(pick(n["net2_14"], pixels=480 * 272), "fg_net2_mconv_m1b2")
        self.assertEqual(pick(n["net2_24"], up=1), "fg_net2_mconv_m3b2u")
        self.assertEqual(pick(n["net2_25"], up=1), "fg_net2_mconv_m1b1u")
        self.assertEqual(M.variant(96, 64, 1)[1:], (3, 2, 8))     # 3 tiles per wave, 2 groups, 128 lanes = 8 rows
        self.assertEqual(M.variant(64, 64, 1)[0], "fg_net2_mconv_m2b4u")
        for k, (mt, nb, up, block) in M.VARIANTS.items():
            co, args, defines = self.decl[k]
            self.assertEqual(co, M.CO_MMA)
            self.assertEqual(defines, "FG_MT=%d,FG_NB=%d%s" % (mt, nb, ",FG_UP=1" if up else ""))
            self.assertEqual(len(args), 28)
            self.assertEqual(args[:9], ["p"] * 9)
        for k, d in self.decl.items():
            if k in M.VARIANTS:
                with open(os.path.join(KDIR, "fg_net2_mconv.hlsl"), encoding="utf-8") as f:
                    self.assertIn("@wg-kernel %s co=%s entry=fg_net2_mconv block=%d " % (k, M.CO_MMA, M.VARIANTS[k][3]),
                                  f.read())
        with self.assertRaises(ValueError):
            M.variant(32, 48)

    def test_chain_against_exact_plan(self):
        """The fast chain: 30 launches; the merge, the three head upsamples and the warp-merge as in the exact chain;
        every convolution on a fast kernel with consistent tiling and weight data; the upsample staged in L12, L13, L24
        and L25, the heads of L13 and L25 in one launch each; buffers as the exact chain's minus the upsampled copies."""
        fast, exact = self.case.steps, self.exact.steps
        self.assertEqual(len(fast), 27)
        self.assertEqual(set(self.exact.outs) - set(self.case.outs), {"r4u", "gu", "s4u", "eu"})
        self.assertFalse(set(self.case.outs) - set(self.exact.outs))
        other = [s for s in fast if s[0] not in M.VARIANTS]
        self.assertEqual([s[0] for s in other], ["fg_net2_merge", M.UPWARP])
        self.assertIn(other[0], exact)
        warp = [s for s in exact if s[0] == "fg_net2_warp"][0][2]           # ib, ic, ia, u4, u1, u3, y, W, H, f
        ups = [s[2] for s in exact if s[0] == "fg_net2_up2"][2:5]          # the head upsamples: x, y, C, hi, wi
        _, g, a = other[1]
        self.assertEqual(a[:3], warp[:3])
        self.assertEqual(a[3:6], [u[0] for u in ups])
        self.assertEqual(a[6:9], [u[1] for u in ups])
        self.assertEqual(a[9:], warp[6:])
        self.assertEqual(g, -(-(int(a[10]) * int(a[11])) // 256))
        self.assertEqual(len(a), len(self.decl[M.UPWARP][1]))
        convs = [s for s in fast if s[0] in M.VARIANTS]
        self.assertEqual(len(convs), 25)
        for j, (k, g, a) in enumerate(convs):
            l = L.NET2[j]
            self.assertEqual(len(a), 28)
            mt, nb, up, block = M.VARIANTS[k]
            H, W, heads = int(a[12]), int(a[13]), int(a[27])
            self.assertEqual(up, 1 if j + 1 in (12, 13, 24, 25) else 0, l.name)
            self.assertEqual(heads, 1 if j + 1 in (13, 25) else 0, l.name)
            th = 2 * block // 32
            tx, ty = int(a[25]), int(a[26])
            self.assertEqual((tx, ty), (-(-W // 16), -(-H // th)))
            groups = 3 if heads else -(-l.out // 16) // mt
            self.assertEqual(g, tx * ty * groups)
            self.assertEqual(len(self.case.data[a[1][1:]]), -(-l.out // 16) * 9 * nb * 512)
            self.assertEqual(len(self.case.data[a[2][1:]]), 32 * -(-l.out // 16))
            self.assertEqual(nb, M.blocks(l.cin))
        for _, _, args in fast:
            for a in args:
                if a.startswith("@"):
                    self.assertTrue(a[1:] in self.case.data or a[1:] in self.case.outs, a)
        with self.assertRaises(ValueError):
            c = M.FastCase("bad", "x")
            c.conv(c.buf("x", np.zeros((4, 4, 32))), c.buf("w", np.zeros((16, 3, 3, 32))), c.buf("b", np.zeros(16)),
                   "@y", in_cs=32, cin=32, cw=32, co_n=16, H=4, W=4, heads=1)

    def test_model_matches_reference(self):
        """The kernel's addressing with exact sums equals conv3 bit for bit on every layer shape (L1's 10 of 16 stored
        channels, L14's 18, group reads from a wider tensor)."""
        rng = np.random.default_rng(5)
        for H, W, in_cs, in_co, cin, cw, cout in ((9, 21, 10, 0, 10, 16, 32), (7, 18, 18, 0, 18, 18, 16),
                                                  (6, 17, 96, 64, 32, 32, 8), (5, 19, 64, 0, 64, 64, 96),
                                                  (8, 16, 48, 16, 16, 16, 8), (6, 6, 32, 0, 32, 32, 48)):
            x = K.f16(rng.uniform(-4, 4, (H, W, in_cs)))
            w = K.f16(rng.uniform(-0.3, 0.3, (cout, 3, 3, cw)))
            b = K.f16(rng.uniform(-2, 2, cout))
            got = model_conv(x, in_co, cin, H, W, M.swizzle(w), M.bias_bytes(b), cout)
            want = N.conv3(x[:, :, in_co:in_co + cin], w, b)
            self.assertTrue(np.array_equal(got.astype(np.float16).view(np.uint16),
                                           want.astype(np.float16).view(np.uint16)), (H, W, in_cs, cin, cout))

    def test_exact_values_are_exact(self):
        """The exactval stand-in draws only values whose chunk sums stay exact (and the context restores K._rand)."""
        rng = np.random.default_rng(1)
        with M.exact_values():
            self.assertTrue(set(np.unique(K._rand(rng, 1000, 0.3))) <= {-0.5, 0.0, 0.5})
            self.assertTrue(set(np.unique(K._rand(rng, 1000, 8.0))) <= {-1.0, 0.0, 1.0})
        self.assertEqual(K._rand.__name__, "_rand")
        self.assertGreater(np.abs(K._rand(rng, 1000, 8.0)).max(), 1.0)


if __name__ == "__main__":
    unittest.main()
