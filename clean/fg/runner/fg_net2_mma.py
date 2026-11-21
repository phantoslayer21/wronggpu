# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): net2 on gfx12's matrix instructions (tier B of the fast path, D3D12), with the exact
path (clean/fg/runner/fg_net2_d3d12_check.py) selectable as the validation mode.

    python clean/fg/runner/fg_net2_mma.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_net2_mma.py --build <dir> --make
    python clean/fg/runner/fg_net2_mma.py --build <dir> --run [--json out.json]
    python clean/fg/runner/fg_net2_mma.py --build <dir> --timing --record out.json [--json timing.json]
    python clean/fg/runner/fg_net2_mma.py --build <dir> --profile "<case>,<case>" [--json profile.json]

Paths (same weights, same inputs):
- fast (27 launches): the input merge (exact kernel), the 25 convolutions as fg_net2_mconv_<variant>
  (clean/dx12/kernels/fg_net2_mconv.hlsl: one v_wmma_f32_16x16x16_f16 per spec chunk in net2's tap-major chunk order,
  the float16 accumulator from the bias as C, the result rounded to float16 after every chunk; the exact post-ops, pool,
  clamp rule and residual in the epilogue), with the 2x upsample of L12, L13, L24 and L25 staged exactly inside the
  convolution and the three heads of L13 and of L25 in one launch each, and launches 15 + 16 (head upsample and
  warp-merge) as one exact kernel (clean/dx12/kernels/fg_net2_upwarp.hlsl). Tier B: only the chunk sums differ from the
  CPU reference; every other operation has its bits;
- exact (38 launches): fg_net2_d3d12_check.py's chain, bit-exact to the CPU reference (the validation mode and the
  fallback).

--prepare (CPU) builds the D3D12 kernels of both paths and kernel_replay.exe into <dir>.
--make (CPU) writes every case into <dir>/replay-net2-mma. No GPU.
--run (GPU, one GPU slot) runs every chain through kernel_replay (one process, memory and time watchdog). Every chain
  runs twice, its output buffers filled with 0x5A and with 0xA5 and followed by 64 slack bytes. Gates:
- exactval: every fast convolution variant on every net2 layer shape (single launches, the small-grid variants and
  the large ones, the staged upsample, the heads) with values whose chunk sums are exact (inputs in {-1, 0, 1},
  weights in {-0.5, 0, 0.5}, biases and skips in {-1, 0, 1}: every sum is exact in single precision), so tier B and
  the exact arithmetic agree: byte-identical to the CPU reference (indexing, chunk tiling, channel masks, the staging,
  the post-ops, pool and stores), both fills, slack kept; and fg_net2_upwarp (exact) byte-identical, one case with
  adversarial images and displacements;
- probe (adversarial, every chunk sum exact so the result must equal the reference byte for byte; each with a mutant
  it must catch): the exact runner's multiply-add probe through the residual (two roundings); the staged upsample
  through identity convolutions on float16 midpoints, tiny addends, signed zeros, subnormals, values near 65504 and
  non-finite values, for every staging shape (every product rounded); leaky ReLU, clamp and the 2 x 2 average pool
  through identity convolutions on the same values (the average with one rounding); upwarp's adversarial case
  (every product rounded);
- negative controls: per fast kernel, its first exactval case on half of the first launch's workgroups must differ
  from the reference;
- kernel / chain / full (the 480 x 272 and 640 x 368 chains): tier B is not bit-exact, so the gate is that both fills
  give the same bytes (every element written, deterministic) and keep the slack; every buffer is reported against the
  CPU reference (values more than 1 / 4 float16 steps off, the largest step distance, the largest absolute difference
  and that over the buffer's largest reference magnitude; report only, FG-ACCEPTANCE.md);
- mutants on the fast chain: the last launch on half of its workgroups, L12's group 0 clamped, L24 clamped; each is
  caught when its buffer differs from the plain fast chain's (same data) while every buffer written before it is
  byte-identical to the plain run;
- exact: the exact chain at 48 x 32 and at the two full grids, byte-identical to the reference.
--timing (GPU, one GPU slot; information only) refuses unless --record is a passing --run record. Per path and grid:
  a warm-up chain, then one chain with `time <reps> each`: per run the wall time and the GPU time (device timestamps
  around the run's launches), mean and median of each; every timed process saves the output again and its bytes must
  equal the checked run's (fast) or the reference (exact).
--profile (GPU, diagnostic only, no byte check): the named chains under the device's dispatch profiler
  (WG_D3D12_PROFILE), the median GPU time of every launch.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from clean.fg import layout as L  # noqa: E402
from clean.fg.reference import net2 as N  # noqa: E402
import fg_net2_d3d12_check as K  # noqa: E402

CO_MMA = "fg_net2_mma.co"
# kernel -> (output tiles per wave, 16-channel blocks per tap, upsample staging, lanes per workgroup); the declarations
# of clean/dx12/kernels/fg_net2_mconv.hlsl
VARIANTS = {"fg_net2_mconv_m1b1": (1, 1, 0, 256), "fg_net2_mconv_m1b2": (1, 2, 0, 256),
            "fg_net2_mconv_m2b1": (2, 1, 0, 256), "fg_net2_mconv_m2b2": (2, 2, 0, 256),
            "fg_net2_mconv_m2b4": (2, 4, 0, 256), "fg_net2_mconv_m1b4s": (1, 4, 0, 128),
            "fg_net2_mconv_m1b2s": (1, 2, 0, 128), "fg_net2_mconv_m1b1u": (1, 1, 1, 256),
            "fg_net2_mconv_m1b2u": (1, 2, 1, 256), "fg_net2_mconv_m2b4u": (2, 4, 1, 128),
            "fg_net2_mconv_m3b2u": (3, 2, 1, 256), "fg_net2_mconv_m3b4u": (3, 4, 1, 128)}
MCONV = list(VARIANTS)
UPWARP = "fg_net2_upwarp"            # clean/dx12/kernels/fg_net2_upwarp.hlsl: launches 15 + 16, exact
FAST_KERNELS = MCONV + [UPWARP]
KERNELS = FAST_KERNELS + K.KERNELS
FILLS, SLACK = K.FILLS, K.SLACK
FULL_GRIDS = K.FULL_GRIDS
TILE = 16
SMALL_GRID = 16384     # conv grids of at most this many pixels use a 128-lane one-tile variant where there is one
HEAD_KEEP = [(0, 4), (4, 1), (5, 3)]   # the heads' kept outputs (fg_net2_mconv heads = 1)


# ---- plan ---------------------------------------------------------------------------------------------------------

def blocks(cw):
    """16-channel blocks per tap of a layer with cw stored weight channels (L1's 16, L14's 18 as 32)."""
    nb = -(-cw // 16)
    if nb not in (1, 2, 4):
        raise ValueError("%d weight channels: no fast kernel" % cw)
    return nb


def variant(cout, cw, up=0, pixels=None, small_grid=SMALL_GRID):
    """(kernel, output tiles per wave, weight tile groups, conv rows per workgroup) for a layer with cout outputs, cw
    weight channels, upsample staging up and a conv grid of `pixels` pixels."""
    nb, tiles = blocks(cw), -(-cout // 16)
    cands = [(k, v) for k, v in VARIANTS.items() if v[1] == nb and v[2] == up and tiles % v[0] == 0]
    small = [kv for kv in cands if kv[1][3] == 128 and kv[1][0] == 1]
    if not up and pixels is not None and pixels <= small_grid and small:
        k, v = small[0]
    else:
        big = sorted([kv for kv in cands if up or kv[1][3] == 256], key=lambda kv: -kv[1][0])
        if not big:
            raise ValueError("no fast kernel for %d outputs, %d weight channels, up %d" % (cout, cw, up))
        k, v = big[0]
    return k, v[0], tiles // v[0], 2 * v[3] // 32


def swizzle(w):
    """Canonical weights [cout, 3, 3, cw] -> A fragments [cout / 16][9 taps][nb][32 lanes][8] float16, zero-padded to
    whole 16-output tiles and 16-channel blocks; tap t = 3 ky + kx; lane l holds weight output 16 mt + l % 16 and input
    channels 16 cb + 8 (l // 16) + 0 .. 7."""
    w = np.asarray(w, np.float64).astype(np.float16)
    cout, kh, kw, cw = w.shape
    if (kh, kw) != (3, 3):
        raise ValueError("3 x 3 weights expected")
    nb, tiles = blocks(cw), -(-cout // 16)
    p = np.zeros((16 * tiles, 3, 3, 16 * nb), np.float16)
    p[:cout, :, :, :cw] = w
    a = p.reshape(tiles, 16, 9, nb, 2, 8)                       # [mt, m, t, cb, h, j]
    return np.ascontiguousarray(a.transpose(0, 2, 3, 4, 1, 5)).tobytes()   # [mt, t, cb, h, m, j]


def bias_bytes(b):
    """The bias padded with zeros to whole 16-output tiles."""
    b = np.asarray(b, np.float64).astype(np.float16).reshape(-1)
    p = np.zeros(16 * -(-b.size // 16), np.float16)
    p[:b.size] = b
    return p.tobytes()


class FastCase(K.Case):
    """A case whose 3 x 3 convolution launches run on the fast kernels; everything else as K.Case (same data, same
    expected buffers: the CPU reference). conv() takes fg_net2_conv's launch arguments plus up (the conv grid H x W is
    the 2x upsample of the input) and heads (the three heads in one launch: y, y1, y2 and skip, skip1, skip2)."""
    small_grid = SMALL_GRID

    def __init__(self, name, kernel):
        super().__init__(name, kernel)
        self.arrays = {}

    def buf(self, i, arr):
        self.arrays[i] = np.asarray(arr)
        return super().buf(i, arr)

    def conv(self, x, w, b, y, *, in_cs, in_co=0, cin, cw, co_lo=0, co_n, out_cs=None, out_co=0, H, W, pool=0,
             leaky=1, clamp_from=0, res=0, res_mul=K.F16_ONE, skip="null", skip_cs=0, skip_co=0, up=0, heads=0,
             y1="null", y2="null", skip1="null", skip2="null"):
        wa, ba = self.arrays[w[1:]], self.arrays[b[1:]]
        if wa.shape[3] != cw:
            raise ValueError("%s: %d weight channels, launch says %d" % (w, wa.shape[3], cw))
        if heads and (wa.shape[0] != 8 or pool or up and H % 2):
            raise ValueError("heads: one 8-output weight tile, no pool")
        if up and (H % 2 or W % 2 or pool):
            raise ValueError("up: an even conv grid, no pool")
        fw, fb = "f" + w[1:], "f" + b[1:]
        if fw not in self.data:
            self.data[fw], self.data[fb] = swizzle(wa), bias_bytes(ba)
        kernel, _, groups, th = variant(wa.shape[0], cw, up, H * W, self.small_grid)
        groups = 3 if heads else groups
        tx, ty = -(-W // TILE), -(-H // th)
        args = [x, "@" + fw, "@" + fb, y, skip, y1, y2, skip1, skip2, in_cs, in_co, cin, H, W, co_lo, co_n,
                co_n if out_cs is None else out_cs, out_co, pool, leaky, clamp_from, res, res_mul, skip_cs, skip_co,
                tx, ty, heads]
        self.steps.append((kernel, tx * ty * groups, [str(a) for a in args]))

    def upwarp(self, ib, ic, ia, a4, a1, a3, u4, u1, u3, y, W, H, f):
        """launches 15 + 16: U4 / U1 / U3 = the 2x upsample of A4 / A1 / A3, y = the warp-merge"""
        self.steps.append((UPWARP, max(1, -(-(W * H) // 256)),
                           [str(a) for a in (ib, ic, ia, a4, a1, a3, u4, u1, u3, y, W, H, K.f32_bits(f))]))


class FastCaseLarge(FastCase):
    """FastCase without the small-grid variants (the exactval cases cover every variant)."""
    small_grid = 0


# ---- cases --------------------------------------------------------------------------------------------------------

def _exact_rand(rng, shape, amp):
    """K._rand's stand-in for the exactval cases: weights (amplitude below 1) in {-0.5, 0, 0.5}, everything else in
    {-1, 0, 1}: every chunk sum of up to 64 channels x 9 taps (at most 288.5 in steps of 0.5) is exact in float16."""
    vals = np.array([-0.5, 0.0, 0.5]) if amp < 1.0 else np.array([-1.0, 0.0, 1.0])
    return rng.choice(vals, shape)


@contextlib.contextmanager
def exact_values():
    saved = K._rand
    K._rand = _exact_rand
    try:
        yield
    finally:
        K._rand = saved


def _single_launch_cases(rng, tag, cls):
    k = {"cls": cls}
    return [K.conv_case("ev%s L1 34x40 pool" % tag, rng, 34, 40, 10, 16, 32, pool=1, **k),
            K.conv_case("ev%s L2 18x36 pool" % tag, rng, 18, 36, 32, 32, 32, pool=1, **k),
            K.conv_case("ev%s L3 20x34 pool" % tag, rng, 20, 34, 32, 32, 64, pool=1, **k),
            K.conv_case("ev%s L4 flag0 17x30" % tag, rng, 17, 30, 64, 64, 64, res=1, **k),
            K.conv_case("ev%s L5 flag1 17x30" % tag, rng, 17, 30, 64, 64, 64, res=2, **k),
            K.conv_case("ev%s L12 split 20x34" % tag, rng, 20, 34, 64, 64, 96, clamp_from=32, **k),
            K.conv_case("ev%s L14 18ch 34x36 pool" % tag, rng, 34, 36, 18, 18, 16, pool=1, **k),
            K.conv_case("ev%s L15 18x36 pool" % tag, rng, 18, 36, 16, 16, 32, pool=1, **k),
            K.conv_case("ev%s L16 flag0 18x33" % tag, rng, 18, 33, 32, 32, 32, res=1, **k),
            K.conv_case("ev%s L17 flag1 18x33" % tag, rng, 18, 33, 32, 32, 32, res=2, **k),
            K.conv_case("ev%s L24 no clamp 20x34" % tag, rng, 20, 34, 32, 32, 48, clamp_from=K.NO_CLAMP, **k),
            K.conv_case("ev%s group read 19x21" % tag, rng, 19, 21, 32, 32, 8, in_cs=96, in_co=64, co_lo=3, co_n=4, **k),
            K.heads_case("ev%s L13 heads 20x34" % tag, rng, 20, 34, 32, 32, residual=False, **k),
            K.heads_case("ev%s L25 heads 34x36" % tag, rng, 34, 36, 16, 16, residual=True, **k)]


def fused_case(name, rng, h, w, cin_each, cout, *, heads=0, residual=False, clamp_from=0, leaky=1):
    """One fused launch: the conv on the 2x upsample of x (h x w -> 2h x 2w); heads: L13 / L25's three heads, with
    residual one 8-channel output plus fma(2, U4) / U1 / U3."""
    c = FastCase(name, "fg_net2_mconv")
    groups = 3 if heads else 1
    x = K._rand(rng, (h, w, groups * cin_each), 8.0)
    wt = K._rand(rng, (cout, 3, 3, cin_each), 2.5 / np.sqrt(9 * cin_each))
    b = K._rand(rng, cout, 8.0)
    xa, wa, ba = c.buf("x", x), c.buf("w", wt), c.buf("b", b)
    xu = N.up2(x)
    H, W = 2 * h, 2 * w
    if not heads:
        ya = c.out("y", (H, W, cout))
        c.conv(xa, wa, ba, ya, in_cs=cin_each, cin=cin_each, cw=cin_each, co_n=cout, H=H, W=W, up=1,
               clamp_from=clamp_from, leaky=leaky)
        c.expect["y"] = K.conv_cpu(xu, 0, cin_each, wt, b, 0, cout, leaky=leaky, clamp_from=clamp_from)
        return c
    if residual:
        us = [K._rand(rng, (H, W, n), 8.0) for _, n in HEAD_KEEP]
        ua = [c.buf("U%d" % n, u) for (_, n), u in zip(HEAD_KEEP, us)]
        ya = c.out("out", (H, W, 8))
        c.conv(xa, wa, ba, ya, in_cs=3 * cin_each, cin=cin_each, cw=cin_each, co_n=8, out_cs=8, H=H, W=W, up=1, heads=1,
               leaky=0, clamp_from=4, res=2, res_mul=K.F16_TWO, skip=ua[0], skip1=ua[1], skip2=ua[2])
        c.expect["out"] = np.concatenate(
            [K.conv_cpu(xu, j * cin_each, cin_each, wt, b, lo, n, leaky=0, clamp_from=4, res=2,
                        res_mul=2.0 if j == 0 else 1.0, skip=us[j]) for j, (lo, n) in enumerate(HEAD_KEEP)], 2)
        return c
    ys = [c.out(nm, (H, W, n)) for nm, (_, n) in zip(("h4", "h1", "h3"), HEAD_KEEP)]
    c.conv(xa, wa, ba, ys[0], in_cs=3 * cin_each, cin=cin_each, cw=cin_each, co_n=8, out_cs=0, H=H, W=W, up=1, heads=1,
           leaky=0, clamp_from=4, y1=ys[1], y2=ys[2])
    for j, (nm, (lo, n)) in enumerate(zip(("h4", "h1", "h3"), HEAD_KEEP)):
        c.expect[nm] = K.conv_cpu(xu, j * cin_each, cin_each, wt, b, lo, n, leaky=0, clamp_from=4)
    return c


@contextlib.contextmanager
def two_rounding_fma():
    """The reference's fma16 replaced by h(h(a b) + c) (every product rounded): the alternative the upsample probe must
    tell apart."""
    saved = N.fma16
    N.fma16 = lambda a, b, c: N.add16(N.mul16(a, b), c)
    try:
        yield
    finally:
        N.fma16 = saved


def probe_up_values(rng, shape):
    """Adversarial float16 inputs for the staged upsample: products at float16 midpoints (1 + k 2^-10, k odd, times
    0.75 / 0.25 / 1.25) next to tiny addends of both signs, ties, signed zeros, subnormals, values near 65504, mixed
    magnitudes; non-finite values only in the input's last 2 x 2 corner (a zero weight times inf is NaN, so they would
    otherwise cover the whole identity convolution)."""
    n = int(np.prod(shape))
    k = rng.integers(0, 512, n) * 2 + 1
    mids = (1.0 + k * 2.0 ** -10) * 2.0 ** rng.integers(-6, 8, n) * rng.choice([-1.0, 1.0], n)
    tiny = rng.choice([2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -2.0 ** -20, 3 * 2.0 ** -24, 2.0 ** -15, -2.0 ** -14,
                       (1023.0 / 1024.0) * 2.0 ** -14], n)
    zeros = rng.choice([0.0, -0.0], n)
    big = rng.choice([65504.0, -65504.0, 60000.0, 32768.0 + 16.0, -49152.0, 2048.0 + 2.0, 1025.0], n)
    mixed = rng.uniform(-1, 1, n) * 2.0 ** rng.integers(-24, 15, n)
    pick = rng.choice(5, n, p=[0.35, 0.2, 0.1, 0.1, 0.25])
    v = np.choose(pick, [mids, tiny, zeros, big, mixed]).reshape(shape)
    v[-2:, -2:, :] = rng.choice([np.inf, -np.inf, np.nan, 1.0, -3.0], (2, 2, shape[2]))
    return K.f16(v)


def probe_up_case(name, rng, h, w, cin, cout):
    """The staged upsample through an identity convolution (weight 1 at the centre tap from input channel o to output
    o, zero elsewhere, zero bias, no activation): every chunk sum is exact, so the output is the upsample itself (+0 for
    zeros) and must equal the reference byte for byte; the two-rounding alternative is saved as the mutant."""
    c = FastCase(name, "fg_net2_mconv")
    x = probe_up_values(rng, (h, w, cin))
    wt = np.zeros((cout, 3, 3, cin))
    for o in range(min(cin, cout)):
        wt[o, 1, 1, o] = 1.0
    b = np.zeros(cout)
    xa, wa, ba = c.buf("x", x), c.buf("w", wt), c.buf("b", b)
    ya = c.out("y", (2 * h, 2 * w, cout))
    c.conv(xa, wa, ba, ya, in_cs=cin, cin=cin, cw=cin, co_n=cout, H=2 * h, W=2 * w, up=1, leaky=0,
           clamp_from=K.NO_CLAMP)
    with np.errstate(invalid="ignore", over="ignore"):
        c.expect["y"] = K.conv_cpu(N.up2(x), 0, cin, wt, b, 0, cout, leaky=0, clamp_from=K.NO_CLAMP)
        with two_rounding_fma():
            c.alt["y"] = K.conv_cpu(N.up2(x), 0, cin, wt, b, 0, cout, leaky=0, clamp_from=K.NO_CLAMP)
    return c


@contextlib.contextmanager
def single_rounding_pool():
    """The reference's pool2 replaced by h(h(a00 + a01 + a10 + a11) * 0.25) (the four values summed exactly, one
    rounding): the alternative the pool probe must tell apart."""
    saved = N.pool2

    def pool1(a):
        a = N.h(a)
        H, W, _ = a.shape
        t = [a[dy:2 * (H // 2):2, dx:2 * (W // 2):2] for dy in (0, 1) for dx in (0, 1)]
        with np.errstate(invalid="ignore", over="ignore"):
            return N.mul16(N.h(t[0] + t[1] + t[2] + t[3]), 0.25)
    N.pool2 = pool1
    try:
        yield
    finally:
        N.pool2 = saved


def probe_pool_case(name, rng, H, W, c, cls=FastCase):
    """The epilogue's exact operations through an identity convolution (as probe_up_case): leaky ReLU, the clamp on the
    upper half of the outputs and the 2 x 2 average pool of the adversarial values, byte for byte; the single-rounding
    average is saved as the mutant."""
    cs = cls(name, "fg_net2_mconv")
    x = probe_up_values(rng, (H, W, c))
    wt = np.zeros((c, 3, 3, c))
    for o in range(c):
        wt[o, 1, 1, o] = 1.0
    b = np.zeros(c)
    xa, wa, ba = cs.buf("x", x), cs.buf("w", wt), cs.buf("b", b)
    ya = cs.out("y", (H // 2, W // 2, c))
    cs.conv(xa, wa, ba, ya, in_cs=c, cin=c, cw=c, co_n=c, H=H, W=W, pool=1, leaky=1, clamp_from=c // 2)
    with np.errstate(invalid="ignore", over="ignore"):
        cs.expect["y"] = K.conv_cpu(x, 0, c, wt, b, 0, c, pool=1, leaky=1, clamp_from=c // 2)
        with single_rounding_pool():
            cs.alt["y"] = K.conv_cpu(x, 0, c, wt, b, 0, c, pool=1, leaky=1, clamp_from=c // 2)
    return cs


def upwarp_case(name, rng, H, W, aamp, f, adversarial=False):
    """fg_net2_upwarp on its own: heads A4 / A1 / A3 at H / 2 x W / 2 (displacements up to about 2 aamp pixels after
    the doubling), images Ib / Ic / Ia; expected U4 / U1 / U3 (up2) and y (warp_merge) of the reference. adversarial:
    images with float16 midpoints, tiny addends, signed zeros, subnormals, values near 65504 and non-finite values in a
    corner, and displacements whose fractions are fine binary steps (weights at many float16 roundings); the
    two-rounding alternative is saved as the mutant."""
    c = FastCase(name, UPWARP)
    h, w = H // 2, W // 2
    if adversarial:
        ims = [probe_up_values(rng, (H, W, k)) for k in (4, 4, 1)]
        heads = [K.f16(rng.integers(-2048, 2048, (h, w, k)) * aamp / 2048.0 + rng.choice([0.0, 2.0 ** -24, -0.0], (h, w, k)))
                 for k in (4, 1, 3)]
    else:
        ims = [K._rand(rng, (H, W, k), 9.0) for k in (4, 4, 1)]
        heads = [K._sprinkle(rng, K._rand(rng, (h, w, k), aamp), 0.05,
                             np.array([0.5, -0.5, 0.0, -0.0, 2.0 ** -24, -2.0 ** -24, -0.2998046875])) for k in (4, 1, 3)]
    ia_ = [c.buf(i, v) for i, v in zip(("Ib", "Ic", "Ia"), ims)]
    ha = [c.buf(i, v) for i, v in zip(("h4", "h1", "h3"), heads)]
    outs = [c.out(i, (H, W, k)) for i, k in (("U4", 4), ("U1", 1), ("U3", 3))]
    ya = c.out("y", (H, W, 18))
    c.upwarp(*ia_, *ha, *outs, ya, W, H, np.float32(f))
    with np.errstate(invalid="ignore", over="ignore"):
        us = [N.up2(v) for v in heads]
        for i, u in zip(("U4", "U1", "U3"), us):
            c.expect[i] = u
        c.expect["y"] = N.warp_merge(*ims, *us, np.float32(f))
        if adversarial:
            with two_rounding_fma():
                us2 = [N.up2(v) for v in heads]
                c.alt["y"] = N.warp_merge(*ims, *us2, np.float32(f))
    return c


def upwarp_cases(rng):
    return [upwarp_case("upwarp 32x48", rng, 32, 48, 4.0, -0.8125),
            upwarp_case("upwarp 48x32 far", rng, 48, 32, 60.0, 0.4375),
            upwarp_case("upwarp 16x64 small", rng, 16, 64, 0.75, 3.0e-6),
            upwarp_case("upwarp 64x96 adversarial", rng, 64, 96, 12.0, 1.0 + 2.0 ** -11, adversarial=True)]


def exactval_cases(rng):
    """Every layer shape (grids with several workgroups) on exactly summable values, with the small-grid variants and
    without them, and the fused launches (upsample staging, heads)."""
    with exact_values():
        return _single_launch_cases(rng, "", FastCase) + _single_launch_cases(rng, " large", FastCaseLarge) + \
            [fused_case("ev L12 up split 10x17", rng, 10, 17, 64, 96, clamp_from=32),
             fused_case("ev up 64 to 64 9x18", rng, 9, 18, 64, 64),
             fused_case("ev L13 up heads 10x17", rng, 10, 17, 32, 8, heads=1),
             fused_case("ev L24 up no clamp 10x17", rng, 10, 17, 32, 48, clamp_from=K.NO_CLAMP),
             fused_case("ev L25 up heads 17x18", rng, 17, 18, 16, 8, heads=1, residual=True)]


def fast_chain_case(name, W2, H2, params, ib, ic, ia, f, *, mutant=None):
    """net2 as 30 fast launches: K.chain_case's chain with the upsample of L12, L13, L24 and L25 staged inside the
    convolution and each head layer as one launch. Expected buffers: the CPU reference's (no upsampled copies)."""
    ref = N.run(ib, ic, ia, f, params)
    c = FastCase(name, "chain")
    exp = {k: v for k, v in ref.items() if isinstance(v, np.ndarray) and k not in ("Ib", "Ic", "Ia")}
    for k in ("g0", "g1", "g2", "e0", "e1", "e2"):
        exp.pop(k)
    exp["g"] = np.concatenate([ref["g0"], ref["g1"], ref["g2"]], 2)
    exp["e"] = np.concatenate([ref["e0"], ref["e1"], ref["e2"]], 2)
    for k in ("A4", "A1", "A3"):        # saved files are named by buffer id: A1 and a1 would be one file on Windows
        exp["h" + k] = exp.pop(k)
    c.expect = exp
    ext = N.extents(W2, H2)
    A = {"Ib": c.buf("Ib", ib), "Ic": c.buf("Ic", ic), "Ia": c.buf("Ia", ia)}
    for l in L.NET2:
        w, b = params[l.name]
        A["w" + l.name[-2:]], A["b" + l.name[-2:]] = c.buf("w" + l.name[-2:], w), c.buf("b" + l.name[-2:], b)
    for i in exp:
        A[i] = c.out(i, exp[i].shape)

    def wb(n):
        return A["w%02d" % n], A["b%02d" % n]

    def conv(n, src, dst, cin, H, W, **kw):
        l = L.NET2[n - 1]
        c.conv(A[src], *wb(n), A[dst], in_cs=exp[src].shape[2] if src in exp else cin, cin=cin, cw=l.cin,
               co_n=kw.pop("co_n", l.out), H=H, W=W, **kw)

    def block(first, src, temps, outs, cin):
        H, W = exp[src].shape[:2]
        for u in range(4):
            t, r = "%s%d" % (temps, u + 1), "%s%d" % (outs, u + 1)
            conv(first + 2 * u, src, t, cin, H, W, res=1)
            conv(first + 2 * u + 1, t, r, cin, H, W, res=2, skip=A[src], skip_cs=cin)
            src = r

    c.merge(A["Ib"], A["Ic"], A["Ia"], A["x"], W2 * H2, f)
    conv(1, "x", "a1", 10, H2, W2, pool=1)
    conv(2, "a1", "a2", 32, *exp["a1"].shape[:2], pool=1)
    conv(3, "a2", "r0", 32, *exp["a2"].shape[:2], pool=1)
    block(4, "r0", "t", "r", 64)
    h4, w4 = exp["g"].shape[:2]
    conv(12, "r4", "g", 64, h4, w4, up=1, clamp_from=0 if mutant == "clamp_l12" else 32)
    h2, w2 = ext["A4"]
    conv(13, "g", "hA4", 32, h2, w2, up=1, heads=1, co_n=8, out_cs=0, leaky=0, clamp_from=4, y1=A["hA1"],
         y2=A["hA3"])
    c.upwarp(A["Ib"], A["Ic"], A["Ia"], A["hA4"], A["hA1"], A["hA3"], A["U4"], A["U1"], A["U3"], A["y"], W2, H2, f)
    conv(14, "y", "b1", 18, H2, W2, pool=1)
    conv(15, "b1", "s0", 16, *exp["b1"].shape[:2], pool=1)
    block(16, "s0", "tb", "s", 32)
    h2, w2 = exp["e"].shape[:2]
    conv(24, "s4", "e", 32, h2, w2, up=1, clamp_from=0 if mutant == "clamp_l24" else K.NO_CLAMP)
    conv(25, "e", "out", 16, H2, W2, up=1, heads=1, co_n=8, out_cs=8, leaky=0, clamp_from=4, res=2,
         res_mul=K.F16_TWO, skip=A["U4"], skip1=A["U1"], skip2=A["U3"])
    if mutant == "half_last":
        k, g, a = c.steps[-1]
        c.steps[-1] = (k, max(1, g // 2), a)
    return c


def smooth(rng, H, W, C, amp):
    """Synthetic smooth images: per channel four sinusoids of 1 to 4 cycles across the grid, peak about amp."""
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W]).reshape(2, 1, 1)
    out = np.zeros((H, W, C))
    for c in range(C):
        for _ in range(4):
            fy, fx, ph = rng.uniform(1, 4), rng.uniform(1, 4), rng.uniform(0, 2 * np.pi)
            out[:, :, c] += (amp / 4) * np.sin(2 * np.pi * (fy * yy + fx * xx) + ph)
    return K.f16(out)


def chain_pair(name, W2, H2, p, rng, f, fill_specials=False):
    ib, ic, ia = (K._rand(rng, (H2, W2, c), 8.0) for c in (4, 4, 1))
    if fill_specials:
        ib, ic, ia = (K._sprinkle(rng, v, 0.05, K.FINITE_SPECIALS) for v in (ib, ic, ia))
    return ib, ic, ia, np.float32(f)


@contextlib.contextmanager
def reference_memo():
    """The CPU reference run once per input set (the fast and exact chains of one grid share it)."""
    saved, memo = K.N.run, {}

    def run_once(ib, ic, ia, f, params, *a, **kw):
        key = hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(v, np.float64)).tobytes() for v in (ib, ic, ia)) +
                             np.float32(f).tobytes() + repr(id(params)).encode()).hexdigest()
        if key not in memo:
            memo[key] = saved(ib, ic, ia, f, params, *a, **kw)
        return memo[key]
    K.N.run = run_once
    try:
        yield
    finally:
        K.N.run = saved


def make(build, include_full=True):
    with reference_memo():
        return _make(build, include_full)


def _make(build, include_full):
    cdir = os.path.join(build, "replay-net2-mma")
    os.makedirs(cdir, exist_ok=True)
    rng = np.random.default_rng(20261122)
    groups = {"exactval": exactval_cases(rng) + upwarp_cases(rng),
              "probe": [K.probe_case(rng, cls=FastCase)] +
                       [probe_up_case("probe up %d to %d" % (ci, co), rng, h, w, ci, co)
                        for h, w, ci, co in ((19, 22, 16, 16), (11, 21, 32, 32), (9, 17, 32, 48), (10, 9, 64, 64),
                                             (9, 10, 64, 96))] +
                       [probe_pool_case("probe pool 16", rng, 36, 40, 16),
                        probe_pool_case("probe pool 32 small", rng, 22, 36, 32),
                        probe_pool_case("probe pool 32 large", rng, 22, 36, 32, FastCaseLarge)],
              "kernel": K.conv_cases(rng, cls=FastCase)}
    p = K.synthetic_params(rng)
    chains = []
    for (W2, H2), f, sp in (((48, 32), 0.4375, False), ((32, 64), -1.25, False), ((48, 32), 3.0e-6, True)):
        ib, ic, ia, ff = chain_pair("", W2, H2, p, rng, f, sp)
        chains.append(fast_chain_case("fast net2 %dx%d%s" % (W2, H2, " finite specials" if sp else ""), W2, H2, p, ib,
                                      ic, ia, ff))
    groups["chain"] = chains
    ib, ic, ia, ff = chain_pair("", 48, 32, p, rng, 0.4375)
    groups["mutant"] = [fast_chain_case("fast mutant base", 48, 32, p, ib, ic, ia, ff)] + \
        [fast_chain_case("fast mutant " + m, 48, 32, p, ib, ic, ia, ff, mutant=m) for m in K.MUTANT_TARGETS]
    groups["exact"] = [K.chain_case("exact net2 48x32", 48, 32, p, ib, ic, ia, ff)]
    if include_full:
        frng = np.random.default_rng(20261121)
        groups["full"] = []
        for label, (W2, H2) in FULL_GRIDS.items():
            ib, ic, ia = (K._rand(frng, (H2, W2, c), 8.0) for c in (4, 4, 1))
            args = ("net2 %s %dx%d" % (label, W2, H2), W2, H2, p, ib, ic, ia, np.float32(0.4375))
            groups["full"].append(fast_chain_case("fast " + args[0], *args[1:]))
            groups["exact"].append(K.chain_case("exact " + args[0], *args[1:]))
        for label, (W2, H2) in FULL_GRIDS.items():
            ib, ic, ia = (smooth(frng, H2, W2, c, 8.0) for c in (4, 4, 1))
            groups["full"].append(fast_chain_case("fast smooth net2 %s %dx%d" % (label, W2, H2), W2, H2, p, ib, ic, ia,
                                                  np.float32(0.4375)))
    man = {}
    for g, cases in groups.items():
        man[g] = []
        for c in cases:
            m = K.write_case(cdir, c)
            m["sizes"] = {i: len(b) for i, b in c.data.items()}
            man[g].append(m)
    with open(os.path.join(cdir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=0)
    print("made", json.dumps({g: len(v) for g, v in man.items()}), "in", cdir)
    return man


# ---- comparison ---------------------------------------------------------------------------------------------------

def step_report(got_bits, want):
    """got: float16 bits; want: float16-valued reference. Values over 1 / over 4 float16 steps, the largest step
    distance, the largest absolute difference (finite values) and that over the largest reference magnitude, and
    non-finite disagreements (NaN equals NaN)."""
    w = np.asarray(want, np.float64).astype(np.float16).view(np.uint16).reshape(-1)
    g = np.asarray(got_bits, np.uint16).reshape(-1)

    def key(b):
        b = b.astype(np.int64)
        return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)
    fin = ((g & 0x7C00) != 0x7C00) & ((w & 0x7C00) != 0x7C00)
    d = np.where(fin, np.abs(key(g) - key(w)), 0)
    gv, wv = g.view(np.float16).astype(np.float64), w.view(np.float16).astype(np.float64)
    ad = np.where(fin, np.abs(gv - wv), 0.0)
    mref = float(np.max(np.abs(wv[fin]))) if fin.any() else 0.0
    gn = ((g & 0x7C00) == 0x7C00) & ((g & 0x3FF) != 0)
    wn = ((w & 0x7C00) == 0x7C00) & ((w & 0x3FF) != 0)
    nonfin_bad = ~fin & ~((g == w) | (gn & wn))
    return {"values": int(g.size), "exact": int(np.sum((g == w) | (gn & wn))), "over_1_step": int(np.sum(d > 1)),
            "over_4_steps": int(np.sum(d > 4)), "steps_max": int(d.max()) if d.size else 0,
            "max_abs_diff": float(ad.max()) if ad.size else 0.0, "max_abs_ref": mref,
            "rel": float(ad.max() / mref) if mref else 0.0, "nonfinite_mismatch": int(np.sum(nonfin_bad))}


def _saved(cdir, m, tag, i):
    return K._read(os.path.join(cdir, "%s_%s.%s" % (m["id"], tag, i)))


def fills_agree(cdir, m, tags):
    """Both fills' runs of entry m: every output present, the same bytes, and each run's slack its own fill."""
    out, ok = {}, True
    for i, n in m["outs"].items():
        raws = [_saved(cdir, m, t, i) for t, _ in tags]
        present = all(len(r) == n + SLACK for r in raws)
        same = present and raws[0][:n] == raws[1][:n]
        slack = present and all(r[n:] == bytes([fl]) * SLACK for r, (_, fl) in zip(raws, tags))
        out[i] = {"present": present, "fills_agree": same, "slack_ok": slack}
        ok = ok and present and same and slack
    return ok, out


def report_entry(cdir, m, tag):
    rep = {}
    for i, n in m["outs"].items():
        raw = _saved(cdir, m, tag, i)
        exp = np.load(os.path.join(cdir, "%s.expect.%s.npy" % (m["id"], i)))
        if len(raw) < n:
            rep[i] = None
            continue
        rep[i] = step_report(np.frombuffer(raw[:n], np.uint16), exp.astype(np.float64))
    return rep


def total_report(reps):
    """The buffers' reports summed (counts) and maximised (distances)."""
    rs = [r for r in reps if r]
    if not rs:
        return None
    t = {k: sum(r[k] for r in rs) for k in ("values", "exact", "over_1_step", "over_4_steps", "nonfinite_mismatch")}
    t.update({k: max(r[k] for r in rs) for k in ("steps_max", "max_abs_diff", "rel")})
    t["share_over_1_step"] = t["over_1_step"] / t["values"] if t["values"] else 0.0
    return t


def chain_lines(m, fill, tag, **kw):
    """K.chain_lines with both code objects loaded and each step at its kernel's workgroup size."""
    out = []
    for ln in K.chain_lines(m, fill, tag, **kw):
        if ln == "module " + K.CO:
            out += [ln, "module " + CO_MMA]
            continue
        p = ln.split(" ")
        if p[0] == "step" and p[1] in VARIANTS:
            p[3] = str(VARIANTS[p[1]][3])
            ln = " ".join(p)
        out.append(ln)
    return out


def run(build, groups=("exactval", "probe", "kernel", "chain", "mutant", "exact", "full")):
    cdir = os.path.join(build, "replay-net2-mma")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    rid = time.strftime("%H%M%S")
    tags = [("f%02x_%s" % (f, rid), f) for f in FILLS]
    neg_tag = "neg_" + rid
    lines, negs = [], {}
    for g in groups:
        for m in man.get(g, []):
            if g == "mutant":
                lines += chain_lines(m, FILLS[0], tags[0][0])
                continue
            for tag, fill in tags:
                lines += chain_lines(m, fill, tag)
            k0 = m["steps"][0][0]
            if g == "exactval" and k0 not in negs and m["steps"][0][1] >= 2:
                negs[k0] = m
                lines += chain_lines(m, FILLS[0], neg_tag, half=0)
    used = [m for g in groups for m in man.get(g, [])]
    rep, out = K.replay(build, cdir, lines, K.arena_for(used))
    names = [ln.split()[1] for ln in lines if ln.startswith("chain ")]
    not_ok = [n for n in names if "%s: ok" % n not in out]
    rep["chains"], rep["chains_not_ok"] = len(names), not_ok
    res = {"replay": rep, "cases": [], "negative_controls": {}, "mutants": {}}
    ok = rep["exit"] == 0 and rep["killed"] is None and not not_ok
    base = None
    for g in groups:
        for m in man.get(g, []):
            if g == "mutant":
                if m["name"] == "fast mutant base":
                    base = m
                    good, bufs = fills_agree(cdir, m, [tags[0], tags[0]])
                    res["mutants"]["base"] = {"present": good, "report": total_report(report_entry(cdir, m,
                                                                                                   tags[0][0]).values())}
                    ok = ok and good
                    continue
                kind = m["name"].split()[-1]
                target, before = K.MUTANT_TARGETS[kind]
                diff = {}
                for i, n in m["outs"].items():
                    a, b = _saved(cdir, m, tags[0][0], i), _saved(cdir, base, tags[0][0], i)
                    diff[i] = (-1 if len(a) != n + SLACK or len(b) != n + SLACK else
                               int(np.sum(np.frombuffer(a[:n], np.uint16) != np.frombuffer(b[:n], np.uint16))))
                before = [i for i in before if i in diff] if before is not None else [i for i in diff if i != target]
                caught = diff[target] > 0
                clean = all(diff[i] == 0 for i in before)
                res["mutants"][m["name"]] = {"caught": caught and clean, "target": target, "target_differ": diff[target],
                                             "before_match_plain_run": clean}
                ok = ok and caught and clean
                continue
            entry = {"group": g, "case": m["name"], "kernels": m["launch_kernels"], "launches": len(m["steps"])}
            if g in ("exactval", "probe", "exact"):
                good, bufs = K.check_entry(cdir, m, tags)
                entry.update({"gate": "byte-identical to the CPU reference", "ok": good, "buffers": bufs})
                if m.get("alts"):
                    n = 0
                    for i in m["alts"]:
                        alt = np.load(os.path.join(cdir, "%s.alt.%s.npy" % (m["id"], i)))
                        r = K.compare(_saved(cdir, m, tags[0][0], i), alt, m["outs"][i], tags[0][1])
                        n += max(0, r["differ"])
                    res["mutants"]["two roundings: " + m["name"]] = {"caught": n > 0, "differ": n}
                    good = good and n > 0
            else:
                good, bufs = fills_agree(cdir, m, tags)
                reps = report_entry(cdir, m, tags[0][0])
                entry.update({"gate": "both fills identical, slack kept", "ok": good, "buffers": bufs,
                              "vs_cpu": reps, "vs_cpu_total": total_report(reps.values())})
                if "out" in m["outs"]:
                    entry["out_sha256"] = hashlib.sha256(_saved(cdir, m, tags[0][0], "out")[:m["outs"]["out"]]).hexdigest()
            ok = ok and good
            res["cases"].append(entry)
    for k in FAST_KERNELS:
        m = negs.get(k)
        if m is None:
            res["negative_controls"][k] = None
            ok = False
            continue
        _, bufs = K.check_entry(cdir, m, [(neg_tag, FILLS[0])])
        caught = any(b["differ"][0] > 0 for b in bufs.values()) and all(b["differ"][0] >= 0 for b in bufs.values())
        res["negative_controls"][k] = {"case": m["name"], "half_grid": True, "caught": caught,
                                       "differ": sum(b["differ"][0] for b in bufs.values())}
        ok = ok and caught
    cs = res["cases"]
    res["summary"] = {"cases": len(cs), "cases_ok": sum(c["ok"] for c in cs),
                      "byte_exact_cases": sum(1 for c in cs if c["group"] in ("exactval", "probe", "exact")),
                      "fills": [hex(f) for f in FILLS], "pass": bool(ok)}
    res["accuracy"] = {c["case"]: c["vs_cpu_total"] for c in cs if c.get("vs_cpu_total")}
    return res


# ---- timing (information only) ------------------------------------------------------------------------------------

def _time_of(out, name):
    for line in out.splitlines():
        if line.startswith("chain %s: time" % name):
            return float(line.split("time ")[1].split(" ms")[0])
    return None


def timing(build, record, reps=10):
    rec = json.load(open(record, encoding="utf-8"))
    if not rec.get("summary", {}).get("pass"):
        raise SystemExit("timing refused: %s is not a passing --run record" % record)
    sha = {c["case"]: c.get("out_sha256") for c in rec["cases"] if c["group"] == "full"}
    exact_ok = {c["case"] for c in rec["cases"] if c["group"] == "exact" and c["ok"]}
    cdir = os.path.join(build, "replay-net2-mma")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    res = {"reps": reps, "byte_check_record": os.path.abspath(record), "sizes": {}}
    todo = [m for m in man.get("full", []) if sha.get(m["name"]) and "smooth" not in m["name"]] + \
        [m for m in man.get("exact", []) if m["name"] in exact_ok and "x32" not in m["name"]]
    for m in todo:
        tag = "time_" + time.strftime("%H%M%S")
        # a warm-up chain (clocks up), then the timed chain: its first run is saved and checked, then `reps` runs
        lines = chain_lines(m, FILLS[0], tag + "w", saves=False, time_runs=3 * reps)
        lines += [ln + " each" if ln.startswith("time ") else ln
                  for ln in chain_lines(m, FILLS[0], tag + "m", saves=True, time_runs=reps)]
        rep, out = K.replay(build, cdir, lines, K.arena_for([m]), fname="time_%s.txt" % m["id"])
        raw = _saved(cdir, m, tag + "m", "out")[:m["outs"]["out"]]
        if m["name"].startswith("fast"):
            good = hashlib.sha256(raw).hexdigest() == sha[m["name"]]
        else:
            good, _ = K.check_entry(cdir, m, [(tag + "m", FILLS[0])])
        name = "%s_%sm" % (m["id"], tag)
        runs = {}
        for ln in out.splitlines():
            if ln.startswith("chain %s: run " % name):          # chain <name>: run <i> <wall> ms gpu <gpu> ms
                p = ln.split()
                runs[int(p[3])] = (float(p[4]), float(p[7]))
        samples = [runs.get(r) for r in range(reps)]
        mean = _time_of(out, name)
        e = {"launches": len(m["steps"]), "byte_check": bool(good),
             "replay": {k: rep[k] for k in ("exit", "secs", "peak_ws_mb", "killed")}}
        if good and mean is not None and None not in samples and all(s[1] > 0 for s in samples):
            wall, gpu = [s[0] for s in samples], [s[1] for s in samples]
            e.update({"wall_mean_ms": round(float(np.mean(wall)), 4), "wall_median_ms": round(float(np.median(wall)), 4),
                      "gpu_mean_ms": round(float(np.mean(gpu)), 4), "gpu_median_ms": round(float(np.median(gpu)), 4),
                      "chain_mean_ms": mean, "wall_samples_ms": wall, "gpu_samples_ms": gpu})
        else:
            e["withheld"] = "byte check failed or a run missing"
        res["sizes"][m["name"]] = e
    res["method"] = ("kernel_replay on the native D3D12 device: a warm-up chain of %d network runs, then one chain whose "
                     "first run is saved and byte-checked, then %d timed runs (`time %d each`): per run the wall time "
                     "(host: submission to completion, synchronised after each run) and the GPU time (device timestamps "
                     "before the first launch and after the last one, bottom of pipe); mean and median of each. "
                     "Synthetic weights, buffers resident. Information only." % (3 * reps, reps, reps))
    return res


def profile(build, names, runs=20):
    """Diagnostic only (no byte check): after a warm-up, `runs` timed runs of the chain under the device's dispatch
    profiler (WG_D3D12_PROFILE: bottom-of-pipe GPU timestamps around every dispatch); per launch the median over the
    runs, and per run the GPU span from the first dispatch's start to the last one's end."""
    cdir = os.path.join(build, "replay-net2-mma")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    res = {}
    for m in [m for g in man.values() for m in g if m["name"] in names]:
        lines = chain_lines(m, FILLS[0], "pw", saves=False, time_runs=100)      # warm-up (clocks up)
        lines += chain_lines(m, FILLS[0], "pt", saves=False, time_runs=runs)
        path = os.path.join(cdir, "profile_%s_%s.jsonl" % (m["id"], time.strftime("%H%M%S")))
        os.environ["WG_D3D12_PROFILE"] = path
        try:
            rep, out = K.replay(build, cdir, lines, K.arena_for([m]), fname="profile_%s.txt" % m["id"])
        finally:
            os.environ.pop("WG_D3D12_PROFILE", None)
        recs, freq = [], None
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            if r["type"] == "gpu":
                recs.append(r)
            elif r["type"] == "meta":
                freq = r["frequency_hz"]
        disp = sorted([r for r in recs if r["kind"] in ("dispatch", "fill")], key=lambda r: r["sequence"])
        n = len(m["steps"])
        last = disp[-runs * n:]
        ok = len(last) == runs * n and all(r["name"] == m["steps"][i % n][0] for i, r in enumerate(last))
        per, spans = [], []
        if ok:
            for j, (k, g, a) in enumerate(m["steps"]):
                per.append({"step": j, "kernel": k, "grid": g,
                            "ms": round(float(np.median([last[r * n + j]["ms"] for r in range(runs)])), 4)})
            spans = [(last[r * n + n - 1]["stop_tick"] - last[r * n]["start_tick"]) for r in range(runs)]
        res[m["name"]] = {"replay": {k: rep[k] for k in ("exit", "secs", "killed")}, "parsed": ok, "steps": per,
                          "sum_ms": round(sum(p["ms"] for p in per), 4),
                          "gpu_span_ms_median": round(float(np.median(spans)) * 1000.0 / freq, 4) if spans and freq
                          else None,
                          "wall_ms_mean": _time_of(out, "%s_pt" % m["id"]), "profile_file": path}
    return res


# ---- prepare ------------------------------------------------------------------------------------------------------

def prepare(build, rocm):
    from clean.l.runner.l_d3d12_check import prepare_replay
    from clean.runner.hip import clean_process_environment
    os.makedirs(build, exist_ok=True)
    clean_process_environment(rocm)
    return prepare_replay(build, rocm, KERNELS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--no-full", action="store_true", help="--make / --run without the 480 x 272 and 640 x 368 chains")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--groups", default="exactval,probe,kernel,chain,mutant,exact,full")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--record", help="--timing: the passing --run record")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--profile", help="diagnostic: comma-separated case names, every launch timed on its own")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.prepare:
        if not a.rocm:
            ap.error("--prepare needs --rocm")
        prepare(a.build, a.rocm)
    if a.make:
        make(a.build, include_full=not a.no_full)
    res = None
    if a.run:
        groups = [g for g in a.groups.split(",") if not (a.no_full and g == "full")]
        res = run(a.build, groups)
        print(json.dumps(res["summary"]))
        print("negative controls:", json.dumps({k: v and v["caught"] for k, v in res["negative_controls"].items()}),
              "mutants:", json.dumps({k: v.get("caught", v.get("present")) for k, v in res["mutants"].items()}))
        for c in res["cases"]:
            if not c["ok"]:
                print("FAIL", c["case"], json.dumps({i: b for i, b in c["buffers"].items()
                                                     if not b.get("fills_agree", True) or not b.get("slack_ok", True)
                                                     or any(b.get("differ", [0]))}))
        for k, v in res["accuracy"].items():
            print("vs cpu", k, json.dumps(v))
        print("replay", json.dumps({k: res["replay"][k] for k in ("exit", "secs", "peak_ws_mb", "killed")}))
    elif a.timing:
        if not a.record:
            ap.error("--timing needs --record")
        res = timing(a.build, a.record, a.reps)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.endswith("samples_ms")} for k, v in res["sizes"].items()},
                         indent=1))
    elif a.profile:
        res = profile(a.build, a.profile.split(","))
        for name, v in res.items():
            print(name, "sum of dispatches %.3f ms, GPU span %s ms, wall %s ms" % (v["sum_ms"], v["gpu_span_ms_median"],
                                                                                 v["wall_ms_mean"]),
                  json.dumps(v["replay"]), "parsed", v["parsed"])
            for s in v["steps"]:
                print("  %2d %-22s grid %6d  %s ms" % (s["step"], s["kernel"], s["grid"], s["ms"]))
    if res is not None and a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    if a.run:
        return 0 if res["summary"]["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
