# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): net2's D3D12 kernels (clean/dx12/kernels/fg_net2_*.hlsl) against the CPU reference
(clean/fg/reference/net2.py), byte for byte: every kernel on its own and the whole network chained on the device.

    python clean/fg/runner/fg_net2_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_net2_d3d12_check.py --build <dir> --make
    python clean/fg/runner/fg_net2_d3d12_check.py --build <dir> --run [--json out.json]
    python clean/fg/runner/fg_net2_d3d12_check.py --build <dir> --timing --record out.json [--json timing.json]

--prepare (CPU) builds the four D3D12 kernels (clean/dx12/build_dx12.py --only; the pinned DXC through WG_TOOLCHAINS,
BUILDING.md) and kernel_replay.exe (clean/dx12/tests/kernel_replay.cpp) into <dir>.
--make (CPU) writes every case into <dir>/replay-net2: input, weight and bias files, the kernel_replay chains and the
CPU reference's expected buffers. No GPU.
--run (GPU, inside one GPU slot) runs every chain through kernel_replay on the native D3D12 device (one process, with a
memory and time watchdog) and compares the saved buffers with the CPU reference, float16 bits (NaN equals NaN; the
reference's NaN payload is not specified):
- every chain runs twice, its output buffers filled with 0x5A and with 0xA5 and followed by 64 slack bytes; both runs
  must equal the reference and keep the slack, so an unwritten element or a write past the end shows in at least one;
- kernel cases: every net2 layer shape (input merge, the 3 x 3 convolution with pooling, both residual flags, the split
  and head forms with their clamp rules, L14's 18 channels, the output heads with their residuals), the 2x upsample, the
  warp-merge, on small grids with amplitudes above 6 and special values (NaN, +-inf, subnormals, +-65504, -0);
- the whole network (38 launches for the 28 of the reference: the upsample inside L12, L13, L24 and L25 runs as its own
  launch, and L13 / L25 run one launch per head) on synthetic weights, every buffer compared, at two small grids, with
  special values, and at the grids of 1080p / 4K (480 x 272) and 1440p (640 x 368);
- negative controls: per kernel, its first case with 4 or more workgroups again on half of them; in the chain, the last
  launch on half of its workgroups and two clamp mutants (L12 group 0 and L24 with the clamp switched on). Each must be
  caught (its output differs from the reference).
--timing (GPU, one GPU slot; information only) refuses unless --record is a passing --run record. It runs the chain at
480 x 272 and 640 x 368 on --make's data, checks the output against the reference again, and prints the mean
milliseconds per network run (kernel_replay "time": wall time per run, synchronised after each). A probe at a small
grid runs first; a size runs only when the whole chain, scaled by the pixel count, stays under MAX_CHAIN_MS.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.fg import layout as L  # noqa: E402
from clean.fg.reference import net2 as N  # noqa: E402

KERNELS = ["fg_net2_conv", "fg_net2_up2", "fg_net2_merge", "fg_net2_warp"]
CO = "fg_net2.co"
FILLS = (0x5A, 0xA5)
SLACK = 64
NO_CLAMP = 1 << 20
FULL_GRIDS = {"1080p_4k": (480, 272), "1440p": (640, 368)}
PROBE_GRID = (160, 96)
MAX_CHAIN_MS = 1500.0
WATCH = {"ws_bytes": 3 << 30, "system_pct": 85, "secs": 110}
F16_ONE, F16_TWO = 0x3C00, 0x4000
SPECIALS = np.array([np.nan, np.inf, -np.inf, 65504.0, -65504.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -0.0, 0.0],
                    dtype=np.float16)
FINITE_SPECIALS = np.array([60000.0, -60000.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -0.0, 0.0], dtype=np.float16)


def f16(a):
    return np.asarray(a, np.float64).astype(np.float16).astype(np.float64)


def f16_bytes(a):
    return np.ascontiguousarray(np.asarray(a, np.float64).astype("<f2")).tobytes()


def f32_bits(f):
    return int(np.array([f], np.float32).view(np.uint32)[0])


def synthetic_params(rng, wscale=2.5, bscale=1.0):
    out = {}
    for l in L.NET2:
        s = wscale / np.sqrt(9 * l.cin_used)
        out[l.name] = (f16(rng.uniform(-s, s, (l.out, 3, 3, l.cin))), f16(rng.uniform(-bscale, bscale, l.out)))
    return out


# ---- a case: data, output buffers, launches, expected buffers --------------------------------------------------------

class Case:
    def __init__(self, name, kernel):
        self.name, self.kernel = name, kernel
        self.data, self.outs, self.steps, self.expect, self.alt = {}, {}, [], {}, {}

    def buf(self, i, arr):
        self.data[i] = f16_bytes(arr)
        return "@" + i

    def out(self, i, shape):
        self.outs[i] = 2 * int(np.prod(shape))
        return "@" + i

    def step(self, kernel, n, args):
        self.steps.append((kernel, max(1, -(-n // 256)), [str(a) for a in args]))

    # launches
    def conv(self, x, w, b, y, *, in_cs, in_co=0, cin, cw, co_lo=0, co_n, out_cs=None, out_co=0, H, W, pool=0,
             leaky=1, clamp_from=0, res=0, res_mul=F16_ONE, skip="null", skip_cs=0, skip_co=0):
        n = (H // 2 if pool else H) * (W // 2 if pool else W) * co_n
        self.step("fg_net2_conv", n, [x, w, b, y, skip, in_cs, in_co, cin, cw, co_lo, co_n,
                                      co_n if out_cs is None else out_cs, out_co, H, W, pool, leaky, clamp_from, res,
                                      res_mul, skip_cs, skip_co])

    def up2(self, x, y, C, hi, wi):
        self.step("fg_net2_up2", 4 * hi * wi * C, [x, y, C, hi, wi])

    def merge(self, ib, ic, ia, x, npx, f):
        self.step("fg_net2_merge", npx * 10, [ib, ic, ia, x, npx, f32_bits(f)])

    def warp(self, ib, ic, ia, u4, u1, u3, y, W, H, f):
        self.step("fg_net2_warp", W * H * 18, [ib, ic, ia, u4, u1, u3, y, W, H, f32_bits(f)])


def conv_cpu(x, in_co, cin, w, b, co_lo, co_n, *, pool=0, leaky=1, clamp_from=0, res=0, res_mul=1.0, skip=None,
             skip_co=0):
    """What one fg_net2_conv launch computes, from the reference's operations."""
    v = N.conv3(x[:, :, in_co:in_co + cin], w, b)[:, :, co_lo:co_lo + co_n]
    if leaky:
        v = N.leaky(v)
    o = co_lo + np.arange(co_n)
    v = np.where(o >= clamp_from, N.clamp6(v), v)
    if res == 1:
        v = N.plus_zero(v)
    elif res == 2:
        v = N.fma16(res_mul, skip[:, :, skip_co:skip_co + co_n], v)
    return N.pool2(v) if pool else v


# ---- kernel cases -------------------------------------------------------------------------------------------------

def _rand(rng, shape, amp):
    return f16(rng.uniform(-amp, amp, shape))


def _sprinkle(rng, a, frac, values):
    a = a.copy()
    m = rng.random(a.shape) < frac
    a[m] = rng.choice(values, int(m.sum()))
    return f16(a)


def conv_case(name, rng, H, W, cin, cw, cout, *, in_cs=None, in_co=0, co_lo=0, co_n=None, pool=0, leaky=1,
              clamp_from=0, res=0, res_mul=F16_ONE, amp=8.0, wamp=None, specials=False, cls=None):
    c = (cls or Case)(name, "fg_net2_conv")
    in_cs = in_cs or cin
    co_n = co_n or cout
    x = _rand(rng, (H, W, in_cs), amp)
    w = _rand(rng, (cout, 3, 3, cw), wamp or 2.5 / np.sqrt(9 * cin))
    b = _rand(rng, cout, 8.0)
    skip = _rand(rng, (H, W, co_n), 8.0) if res == 2 else None
    if specials:
        x = _sprinkle(rng, x, 0.04, SPECIALS)
        w = _sprinkle(rng, w, 0.01, SPECIALS)
        b[:3] = [np.inf, -np.inf, np.nan]
        if skip is not None:
            skip = _sprinkle(rng, skip, 0.05, SPECIALS)
    xa, wa, ba = c.buf("x", x), c.buf("w", w), c.buf("b", b)
    sa = c.buf("skip", skip) if skip is not None else "null"
    Ho, Wo = (H // 2, W // 2) if pool else (H, W)
    ya = c.out("y", (Ho, Wo, co_n))
    c.conv(xa, wa, ba, ya, in_cs=in_cs, in_co=in_co, cin=cin, cw=cw, co_lo=co_lo, co_n=co_n, H=H, W=W, pool=pool,
           leaky=leaky, clamp_from=clamp_from, res=res, res_mul=res_mul, skip=sa, skip_cs=co_n)
    rm = float(np.uint16(res_mul).view(np.float16))
    c.expect["y"] = conv_cpu(x, in_co, cin, w, b, co_lo, co_n, pool=pool, leaky=leaky, clamp_from=clamp_from,
                             res=res, res_mul=rm, skip=skip)
    return c


def heads_case(name, rng, H, W, cin_each, cw, *, residual, specials=False, cls=None):
    """L13 / L25 as launched: three heads on one weight tensor, each on its own channel group of the input; with
    residual, the three write one 8-channel output and add fma(2, U4) / U1 / U3. cls: the case class (default Case;
    the fast path's runner passes its own, whose conv() emits its kernel)."""
    c = (cls or Case)(name, "fg_net2_conv")
    x = _rand(rng, (H, W, 3 * cin_each), 8.0)
    w = _rand(rng, (8, 3, 3, cw), 3.0 / np.sqrt(9 * cin_each))
    b = _rand(rng, 8, 8.0)
    us = [_rand(rng, (H, W, n), 8.0) for n in (4, 1, 3)]
    if specials:
        x = _sprinkle(rng, x, 0.04, SPECIALS)
        us = [_sprinkle(rng, u, 0.05, SPECIALS) for u in us]
    xa, wa, ba = c.buf("x", x), c.buf("w", w), c.buf("b", b)
    keep, posts = [(0, 4), (4, 1), (5, 3)], [NO_CLAMP, 0, 0]
    if residual:
        ua = [c.buf("U%d" % n, u) for n, u in zip((4, 1, 3), us)]
        ya = c.out("out", (H, W, 8))
        parts = []
        for j, ((lo, n), cf) in enumerate(zip(keep, posts)):
            c.conv(xa, wa, ba, ya, in_cs=3 * cin_each, in_co=j * cin_each, cin=cin_each, cw=cw, co_lo=lo, co_n=n,
                   out_cs=8, out_co=lo, H=H, W=W, leaky=0, clamp_from=cf, res=2,
                   res_mul=F16_TWO if j == 0 else F16_ONE, skip=ua[j], skip_cs=n)
            parts.append(conv_cpu(x, j * cin_each, cin_each, w, b, lo, n, leaky=0, clamp_from=cf, res=2,
                                  res_mul=2.0 if j == 0 else 1.0, skip=us[j]))
        c.expect["out"] = np.concatenate(parts, 2)
    else:
        for j, ((lo, n), cf, nm) in enumerate(zip(keep, posts, ("A4", "A1", "A3"))):
            ya = c.out(nm, (H, W, n))
            c.conv(xa, wa, ba, ya, in_cs=3 * cin_each, in_co=j * cin_each, cin=cin_each, cw=cw, co_lo=lo, co_n=n,
                   H=H, W=W, leaky=0, clamp_from=cf)
            c.expect[nm] = conv_cpu(x, j * cin_each, cin_each, w, b, lo, n, leaky=0, clamp_from=cf)
    return c


def conv_cases(rng, cls=None):
    k = {"cls": cls}
    cs = [conv_case("L1 16x24 pool", rng, 16, 24, 10, 16, 32, pool=1, **k),
          conv_case("L2 8x12 pool", rng, 8, 12, 32, 32, 32, pool=1, **k),
          conv_case("L3 6x10 pool", rng, 6, 10, 32, 32, 64, pool=1, **k),
          conv_case("L4 flag0 6x7", rng, 6, 7, 64, 64, 64, res=1, **k),
          conv_case("L5 flag1 6x7", rng, 6, 7, 64, 64, 64, res=2, **k),
          conv_case("L12 split 10x14", rng, 10, 14, 64, 64, 96, clamp_from=32, amp=12.0, **k),
          conv_case("L14 18ch 12x16 pool", rng, 12, 16, 18, 18, 16, pool=1, **k),
          conv_case("L15 8x18 pool", rng, 8, 18, 16, 16, 32, pool=1, **k),
          conv_case("L16 flag0 6x12", rng, 6, 12, 32, 32, 32, res=1, **k),
          conv_case("L17 flag1 6x12", rng, 6, 12, 32, 32, 32, res=2, **k),
          conv_case("L24 no clamp 10x14", rng, 10, 14, 32, 32, 48, clamp_from=NO_CLAMP, amp=12.0, **k),
          conv_case("group read 9x5", rng, 9, 5, 32, 32, 8, in_cs=96, in_co=64, co_lo=3, co_n=4, **k),
          heads_case("L13 heads 10x14", rng, 10, 14, 32, 32, residual=False, **k),
          heads_case("L25 heads 10x14", rng, 10, 14, 16, 16, residual=True, **k),
          conv_case("specials 6x8 pool", rng, 6, 8, 32, 32, 8, pool=1, specials=True, **k),
          conv_case("specials 6x9 flag1", rng, 6, 9, 16, 16, 8, res=2, specials=True, **k),
          conv_case("specials overflow 5x6", rng, 5, 6, 16, 16, 8, wamp=255.0, amp=255.0, clamp_from=4, **k),
          heads_case("specials L25 heads 6x6", rng, 6, 6, 16, 16, residual=True, specials=True, **k)]
    return cs


PROBE_A = [1.0, 2.0, 3.0, 0.25, -0.25, 1.25, 0.75, -1.0, 2.0 ** -14, 2.0 ** -24, 65504.0, 0.0]
PROBE_C = [0.0, 1.0, -1.0, 3.0, 1.0009765625, 0.99951171875, 2.0 ** -24, -2.0 ** -24, 2.0 ** -14, -2.0 ** -15, 32.0,
           -32.0, 2048.0, -2048.0, 0.5, -0.5, 65504.0, -65504.0, 1.5, -2049.0 / 2.0, 2.0 ** -10, -2.0 ** -11,
           np.inf, -np.inf, np.nan, 0.125, 6.0, -6.0, 1e-3, -1e-3, 4094.0, -4096.0]


def probe_case(rng, cls=None):
    """Adversarial probe of the float16 fused multiply-add (and through a = 1 the add) inside fg_net2_conv: zero weights
    make the conv value its bias c (one per channel), the residual is fma(a, b, c) with a per launch and b per pixel:
    products at float16 midpoints with tiny addends either way, ties, exact cancellation, zeros of both signs,
    subnormal results, overflow at 65520, inf and NaN. The alternative with two roundings, h(h(a b) + c), is saved too:
    the probe must tell it apart (a mutant it catches)."""
    c = (cls or Case)("probe fma a x b + c", "fg_net2_conv")
    b = [683.0, -683.0, 1365.0, 2048.0, -2048.0, 2047.0, 4095.0 / 2.0, 1.0, -1.0, 0.0, -0.0, 2.0 ** -24, -2.0 ** -24,
         (1023.0 / 1024.0) * 2.0 ** -14, 65504.0, -65504.0, 32752.0, 32768.0, -32768.0, 0.5, 1.5, 2.0 ** -11,
         1.0 + 2.0 ** -10, 1.0 - 2.0 ** -11, 3.0, -3.0, 6.0, -6.0, np.inf, -np.inf, np.nan, 1024.5]
    b = f16(np.concatenate([b, rng.uniform(-4, 4, 64 - len(b)) * 2.0 ** rng.integers(-24, 12, 64 - len(b))]))
    cs = f16(PROBE_C)
    H, W, cw = 8, 8, 16
    xa = c.buf("x", np.zeros((H, W, cw)))
    wa = c.buf("w", np.zeros((len(cs), 3, 3, cw)))
    ba = c.buf("b", cs)
    sk = np.broadcast_to(b.reshape(H, W, 1), (H, W, len(cs)))
    sa = c.buf("skip", sk)
    c.alt = {}
    for k, a in enumerate(PROBE_A):
        ab = int(np.float16(a).view(np.uint16))
        ya = c.out("p%d" % k, (H, W, len(cs)))
        c.conv(xa, wa, ba, ya, in_cs=cw, cin=cw, cw=cw, co_n=len(cs), H=H, W=W, leaky=0, clamp_from=NO_CLAMP, res=2,
               res_mul=ab, skip=sa, skip_cs=len(cs))
        cv = np.where(cs == 0, 0.0, cs)[None, None, :]            # an exact chunk result: a zero bias gives +0
        c.expect["p%d" % k] = N.fma16(a, sk, cv)
        with np.errstate(invalid="ignore", over="ignore"):
            c.alt["p%d" % k] = f16(f16(a * sk) + cv)
    return c


def up_cases(rng):
    cs = []
    for hi, wi, C, amp in ((1, 1, 4, 8.0), (3, 5, 2, 40.0), (5, 7, 3, 8.0), (4, 6, 96, 12.0), (34, 60, 64, 20.0),
                           (17, 30, 1, 60000.0)):
        c = Case("up %dx%dx%d" % (hi, wi, C), "fg_net2_up2")
        x = _rand(rng, (hi, wi, C), amp)
        c.up2(c.buf("x", x), c.out("y", (2 * hi, 2 * wi, C)), C, hi, wi)
        c.expect["y"] = N.up2(x)
        cs.append(c)
    c = Case("up specials 9x13x16", "fg_net2_up2")
    x = _sprinkle(rng, _rand(rng, (9, 13, 16), 3.0), 0.08, SPECIALS)
    c.up2(c.buf("x", x), c.out("y", (18, 26, 16)), 16, 9, 13)
    c.expect["y"] = N.up2(x)
    cs.append(c)
    return cs


def merge_cases(rng):
    cs = []
    for (H, W), f in (((16, 16), 0.37), ((32, 48), 1 + 2.0 ** -11), ((16, 32), 1 + 3 * 2.0 ** -11), ((16, 16), 70000.0),
                      ((16, 16), -1e-9), ((16, 48), 65519.0), ((16, 16), float("nan"))):
        c = Case("merge %dx%d f=%r" % (H, W, f), "fg_net2_merge")
        ib, ic, ia = (_sprinkle(rng, _rand(rng, (H, W, k), 9.0), 0.05, SPECIALS) for k in (4, 4, 1))
        c.merge(c.buf("Ib", ib), c.buf("Ic", ic), c.buf("Ia", ia), c.out("x", (H, W, 10)), H * W, np.float32(f))
        c.expect["x"] = N.merge_in(ib, ic, ia, np.float32(f))
        cs.append(c)
    return cs


def warp_cases(rng):
    cs = []
    for (H, W), uamp, img_specials in (((16, 16), 4.0, False), ((32, 48), 4.0, False), ((16, 32), 100.0, False),
                                       ((16, 16), 0.75, True), ((48, 32), 12.0, False)):
        c = Case("warp %dx%d u%.4g%s" % (H, W, uamp, " specials" if img_specials else ""), "fg_net2_warp")
        ib, ic, ia = (_rand(rng, (H, W, k), 9.0) for k in (4, 4, 1))
        if img_specials:
            ib, ic, ia = (_sprinkle(rng, v, 0.05, SPECIALS) for v in (ib, ic, ia))
        u4, u1, u3 = (_rand(rng, (H, W, k), uamp) for k in (4, 1, 3))
        u4 = _sprinkle(rng, u4, 0.05, np.array([0.5, -0.5, 0.0, -0.0, 2.0 ** -24, -2.0 ** -24, -0.2998046875]))
        args = [c.buf(i, v) for i, v in (("Ib", ib), ("Ic", ic), ("Ia", ia), ("U4", u4), ("U1", u1), ("U3", u3))]
        c.warp(*args, c.out("y", (H, W, 18)), W, H, np.float32(-0.8125))
        c.expect["y"] = N.warp_merge(ib, ic, ia, u4, u1, u3, np.float32(-0.8125))
        cs.append(c)
    return cs


# ---- the network chain --------------------------------------------------------------------------------------------

def chain_case(name, W2, H2, params, ib, ic, ia, f, *, mutant=None, cls=None):
    """net2 as 38 launches. mutant: None, "half_last" (the last launch on half of its workgroups), "clamp_l12" (L12's
    group 0 clamped too), "clamp_l24" (L24 clamped). cls: the case class (default Case)."""
    c = (cls or Case)(name, "chain")
    ref = N.run(ib, ic, ia, f, params)
    exp = {k: v for k, v in ref.items() if isinstance(v, np.ndarray) and k not in ("Ib", "Ic", "Ia")}
    for k in ("g0", "g1", "g2", "e0", "e1", "e2"):
        exp.pop(k)
    exp["g"] = np.concatenate([ref["g0"], ref["g1"], ref["g2"]], 2)
    exp["e"] = np.concatenate([ref["e0"], ref["e1"], ref["e2"]], 2)
    exp["r4u"], exp["gu"] = N.up2(ref["r4"]), N.up2(exp["g"])
    exp["s4u"], exp["eu"] = N.up2(ref["s4"]), N.up2(exp["e"])
    for k in ("A4", "A1", "A3"):        # saved files are named by buffer id: A1 and a1 would be one file on Windows
        exp["h" + k] = exp.pop(k)
    c.expect = exp
    ext = N.extents(W2, H2)

    def o(i):
        return c.out(i, exp[i].shape)
    A = {"Ib": c.buf("Ib", ib), "Ic": c.buf("Ic", ic), "Ia": c.buf("Ia", ia)}
    for l in L.NET2:
        w, b = params[l.name]
        A["w" + l.name[-2:]], A["b" + l.name[-2:]] = c.buf("w" + l.name[-2:], w), c.buf("b" + l.name[-2:], b)
    for i in exp:
        A[i] = o(i)

    def wb(n):
        return A["w%02d" % n], A["b%02d" % n]

    def conv(n, src, dst, cin, pool=0, **kw):
        l = L.NET2[n - 1]
        H, W = ext[src] if src in ext else exp[src].shape[:2]
        c.conv(A[src], *wb(n), A[dst], in_cs=exp[src].shape[2] if src in exp else cin, cin=cin, cw=l.cin,
               co_n=kw.pop("co_n", l.out), H=H, W=W, pool=pool, **kw)
    def block(first, src, temps, outs, cin):
        for u in range(4):
            t, r = "%s%d" % (temps, u + 1), "%s%d" % (outs, u + 1)
            conv(first + 2 * u, src, t, cin, res=1)
            conv(first + 2 * u + 1, t, r, cin, res=2, skip=A[src], skip_cs=cin)
            src = r

    # sub-net A
    c.merge(A["Ib"], A["Ic"], A["Ia"], A["x"], W2 * H2, f)
    conv(1, "x", "a1", 10, pool=1)
    conv(2, "a1", "a2", 32, pool=1)
    conv(3, "a2", "r0", 32, pool=1)
    block(4, "r0", "t", "r", 64)
    h8, w8 = ext["r4"]
    c.up2(A["r4"], A["r4u"], 64, h8, w8)
    conv(12, "r4u", "g", 64, clamp_from=0 if mutant == "clamp_l12" else 32)
    h4, w4 = ext["g0"]
    c.up2(A["g"], A["gu"], 96, h4, w4)
    h2, w2 = ext["A4"]
    for j, (dst, lo, n, cf) in enumerate((("hA4", 0, 4, NO_CLAMP), ("hA1", 4, 1, 0), ("hA3", 5, 3, 0))):
        c.conv(A["gu"], *wb(13), A[dst], in_cs=96, in_co=32 * j, cin=32, cw=32, co_lo=lo, co_n=n, H=h2, W=w2,
               leaky=0, clamp_from=cf)
    for k, C in (("4", 4), ("1", 1), ("3", 3)):
        c.up2(A["hA" + k], A["U" + k], C, h2, w2)
    # warp-merge, sub-net B
    c.warp(A["Ib"], A["Ic"], A["Ia"], A["U4"], A["U1"], A["U3"], A["y"], W2, H2, f)
    conv(14, "y", "b1", 18, pool=1)
    conv(15, "b1", "s0", 16, pool=1)
    block(16, "s0", "tb", "s", 32)
    h4, w4 = ext["s4"]
    c.up2(A["s4"], A["s4u"], 32, h4, w4)
    conv(24, "s4u", "e", 32, clamp_from=0 if mutant == "clamp_l24" else NO_CLAMP)
    h2, w2 = ext["e0"]
    c.up2(A["e"], A["eu"], 48, h2, w2)
    for j, (lo, n, cf, rm, u) in enumerate(((0, 4, NO_CLAMP, F16_TWO, "U4"), (4, 1, 0, F16_ONE, "U1"),
                                             (5, 3, 0, F16_ONE, "U3"))):
        c.conv(A["eu"], *wb(25), A["out"], in_cs=48, in_co=16 * j, cin=16, cw=16, co_lo=lo, co_n=n, out_cs=8,
               out_co=lo, H=H2, W=W2, leaky=0, clamp_from=cf, res=2, res_mul=rm, skip=A[u], skip_cs=n)
    if mutant == "half_last":
        k, g, a = c.steps[-1]
        c.steps[-1] = (k, max(1, g // 2), a)
    return c


def chain_cases(rng):
    cs = []
    p = synthetic_params(rng)
    for (W2, H2), f in (((48, 32), 0.4375), ((32, 64), -1.25)):
        ib, ic, ia = (_rand(rng, (H2, W2, k), 8.0) for k in (4, 4, 1))
        cs.append(chain_case("net2 %dx%d" % (W2, H2), W2, H2, p, ib, ic, ia, np.float32(f)))
    ib, ic, ia = (_sprinkle(rng, _rand(rng, (32, 48, k), 8.0), 0.05, FINITE_SPECIALS) for k in (4, 4, 1))
    cs.append(chain_case("net2 48x32 finite specials", 48, 32, p, ib, ic, ia, np.float32(3.0e-6)))
    return cs, p


_BEFORE_L12 = ["x", "a1", "a2", "r0", "t1", "r1", "t2", "r2", "t3", "r3", "t4", "r4", "r4u"]
_BEFORE_L24 = _BEFORE_L12 + ["g", "gu", "hA4", "hA1", "hA3", "U4", "U1", "U3", "y", "b1", "s0", "tb1", "s1", "tb2",
                             "s2", "tb3", "s3", "tb4", "s4", "s4u"]
# mutant: (the buffer it changes, the buffers written before it, which must still match)
MUTANT_TARGETS = {"half_last": ("out", None), "clamp_l12": ("g", _BEFORE_L12), "clamp_l24": ("e", _BEFORE_L24)}


def mutant_cases(rng, p):
    ib, ic, ia = (_rand(rng, (32, 48, k), 8.0) for k in (4, 4, 1))
    return [chain_case("mutant " + m, 48, 32, p, ib, ic, ia, np.float32(0.4375), mutant=m) for m in MUTANT_TARGETS]


def full_cases(p, seed=20261121):
    rng = np.random.default_rng(seed)
    cs = []
    for label, (W2, H2) in FULL_GRIDS.items():
        ib, ic, ia = (_rand(rng, (H2, W2, k), 8.0) for k in (4, 4, 1))
        cs.append(chain_case("net2 %s %dx%d" % (label, W2, H2), W2, H2, p, ib, ic, ia, np.float32(0.4375)))
    return cs


# ---- files --------------------------------------------------------------------------------------------------------

def _id(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def write_case(cdir, case):
    cid = _id(case.name)
    files = {}
    for i, b in case.data.items():
        fn = "%s.%s" % (cid, i)
        with open(os.path.join(cdir, fn), "wb") as f:
            f.write(b)
        files[i] = fn
    for i, a in case.expect.items():
        np.save(os.path.join(cdir, "%s.expect.%s.npy" % (cid, i)), np.asarray(a, np.float64).astype("<f2"))
    for i, a in case.alt.items():
        np.save(os.path.join(cdir, "%s.alt.%s.npy" % (cid, i)), np.asarray(a, np.float64).astype("<f2"))
    return {"name": case.name, "id": cid, "kernel": case.kernel, "files": files, "outs": case.outs,
            "steps": case.steps, "launch_kernels": sorted({s[0] for s in case.steps}), "alts": sorted(case.alt)}


def chain_lines(m, fill, tag, half=None, saves=True, time_runs=0):
    """kernel_replay chain lines of manifest entry m with output fill `fill`; half: a step index on half its grid."""
    name = "%s_%s" % (m["id"], tag)
    lines = ["chain " + name, "module " + CO]
    for i, fn in m["files"].items():
        lines.append("buf %s %d %s" % (i, m["sizes"][i], fn))
    for i, n in m["outs"].items():
        lines.append("buf %s %d - %d" % (i, n + SLACK, fill))
    for j, (k, g, a) in enumerate(m["steps"]):
        lines.append("step %s %d 256 %s" % (k, max(1, g // 2) if j == half else g, " ".join(a)))
    if saves:
        for i in m["outs"]:
            lines.append("save %s %s.%s" % (i, name, i))
    if time_runs:
        lines.append("time %d" % time_runs)
    return lines + [""]


def make(build, include_full=True):
    cdir = os.path.join(build, "replay-net2")
    os.makedirs(cdir, exist_ok=True)
    rng = np.random.default_rng(20261121)
    groups = {"kernel": conv_cases(rng) + [probe_case(rng)] + up_cases(rng) + merge_cases(rng) + warp_cases(rng)}
    chains, p = chain_cases(rng)
    groups["chain"] = chains
    groups["mutant"] = mutant_cases(rng, p)
    if include_full:
        groups["full"] = full_cases(p)
    man = {}
    for g, cases in groups.items():
        man[g] = []
        for c in cases:
            m = write_case(cdir, c)
            m["sizes"] = {i: len(b) for i, b in c.data.items()}
            man[g].append(m)
    probe_rng = np.random.default_rng(5)
    W2, H2 = PROBE_GRID
    ib, ic, ia = (_rand(probe_rng, (H2, W2, k), 8.0) for k in (4, 4, 1))
    pc = chain_case("net2 probe %dx%d" % (W2, H2), W2, H2, p, ib, ic, ia, np.float32(0.4375))
    m = write_case(cdir, pc)
    m["sizes"] = {i: len(b) for i, b in pc.data.items()}
    man["probe"] = [m]
    with open(os.path.join(cdir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=0)
    counts = {g: len(v) for g, v in man.items()}
    print("made", json.dumps(counts), "in", cdir)
    return man


# ---- running kernel_replay with a watchdog ------------------------------------------------------------------------

class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _MSX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def system_memory_pct():
    s = _MSX()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return int(s.dwMemoryLoad)


def _working_set(handle):
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(c), c.cb):
        return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)
    return 0, 0


def replay(build, cdir, lines, arena_mb, fname="chains.txt"):
    """kernel_replay on `lines`, killed (by its own PID) above WATCH's working set, system memory or time."""
    path = os.path.join(cdir, fname)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = str(arena_mb)
    log = os.path.join(cdir, fname + ".log")
    t0 = time.time()
    peak, killed = 0, None
    with open(log, "w") as lf:
        p = subprocess.Popen([os.path.join(build, "kernel_replay.exe"), build, path], env=env, stdout=lf,
                             stderr=subprocess.STDOUT, creationflags=0x08000000)
        while p.poll() is None:
            ws, pk = _working_set(int(p._handle))
            peak = max(peak, pk, ws)
            why = ("working set %d MB" % (ws >> 20) if ws > WATCH["ws_bytes"] else
                   "system memory %d %%" % system_memory_pct() if system_memory_pct() >= WATCH["system_pct"] else
                   "time %.0f s" % (time.time() - t0) if time.time() - t0 > WATCH["secs"] else None)
            if why:
                killed = why
                p.kill()
                p.wait()
                break
            time.sleep(0.25)
    out = open(log, encoding="utf-8", errors="replace").read()
    return {"exit": p.returncode, "secs": round(time.time() - t0, 1), "peak_ws_mb": peak >> 20, "killed": killed,
            "tail": out[-600:]}, out


def arena_for(ms):
    total = 0
    for m in ms:
        total += sum(n + 4096 for n in m["sizes"].values()) + sum(n + SLACK + 4096 for n in m["outs"].values())
    return int(min(3000, max(256, 1.5 * total / 2 ** 20 + 64)))


# ---- comparison ---------------------------------------------------------------------------------------------------

def compare(raw, expect, nbytes, fill):
    """Saved bytes against the expected float16 bits (NaN equals NaN) and the slack's fill."""
    a = np.frombuffer(raw, np.uint8)
    if a.size != nbytes + SLACK:
        return {"values": nbytes // 2, "differ": -1, "slack_ok": False, "missing": True}
    g = a[:nbytes].view(np.uint16)
    r = np.asarray(expect, "<f2").reshape(-1).view(np.uint16)
    gn = ((g & 0x7C00) == 0x7C00) & ((g & 0x3FF) != 0)
    rn = ((r & 0x7C00) == 0x7C00) & ((r & 0x3FF) != 0)
    same = (g == r) | (gn & rn)
    return {"values": int(g.size), "differ": int(np.sum(~same)), "slack_ok": bool(np.all(a[nbytes:] == fill)),
            "nan": int(np.sum(rn)), "inf": int(np.sum((r & 0x7FFF) == 0x7C00)),
            "over6": int(np.sum(np.abs(np.asarray(expect, np.float64)) > 6.0))}


def _read(path):
    return open(path, "rb").read() if os.path.exists(path) else b""


def check_entry(cdir, m, tags_fills):
    """Every output of entry m in every run tag against its expected buffer."""
    bufs, ok = {}, True
    for i, n in m["outs"].items():
        exp = np.load(os.path.join(cdir, "%s.expect.%s.npy" % (m["id"], i)))
        per = []
        for tag, fill in tags_fills:
            r = compare(_read(os.path.join(cdir, "%s_%s.%s" % (m["id"], tag, i))), exp, n, fill)
            per.append(r)
            ok = ok and r["differ"] == 0 and r["slack_ok"]
        bufs[i] = {"values": per[0]["values"], "differ": [r["differ"] for r in per],
                   "slack_ok": all(r["slack_ok"] for r in per), "nan": per[0].get("nan", 0),
                   "inf": per[0].get("inf", 0), "over6": per[0].get("over6", 0)}
    return ok, bufs


def run(build, groups=("kernel", "chain", "mutant", "full")):
    cdir = os.path.join(build, "replay-net2")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    rid = time.strftime("%H%M%S")        # saved files of this run only: a file left by an earlier run never counts
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
            if g == "kernel" and m["kernel"] not in negs and m["steps"][0][1] >= 4:
                negs[m["kernel"]] = m
                lines += chain_lines(m, FILLS[0], neg_tag, half=0)
    used = [m for g in groups for m in man.get(g, [])]
    rep, out = replay(build, cdir, lines, arena_for(used))
    names = [ln.split()[1] for ln in lines if ln.startswith("chain ")]
    not_ok = [n for n in names if "%s: ok" % n not in out]
    rep["chains"], rep["chains_not_ok"] = len(names), not_ok
    res = {"replay": rep, "cases": [], "negative_controls": {}, "mutants": {}}
    ok = rep["exit"] == 0 and rep["killed"] is None and not not_ok
    for g in groups:
        for m in man.get(g, []):
            if g == "mutant":
                # caught: the mutated launch's buffer differs (present, not missing), and everything written before it
                # still matches (half_last: every other buffer)
                target, before = MUTANT_TARGETS[m["name"].split()[-1]]
                _, bufs = check_entry(cdir, m, [tags[0]])
                before = before if before is not None else [i for i in bufs if i != target]
                caught = bufs[target]["differ"][0] > 0 or not bufs[target]["slack_ok"]
                clean = all(bufs[i]["differ"][0] == 0 and bufs[i]["slack_ok"] for i in before)
                res["mutants"][m["name"]] = {"caught": caught and clean, "target": target,
                                             "target_differ": bufs[target]["differ"][0], "before_match": clean,
                                             "differing": {i: b["differ"][0] for i, b in bufs.items()
                                                           if b["differ"][0] != 0 or not b["slack_ok"]}}
                ok = ok and caught and clean
                continue
            good, bufs = check_entry(cdir, m, tags)
            ok = ok and good
            res["cases"].append({"group": g, "case": m["name"], "kernels": m["launch_kernels"],
                                 "launches": len(m["steps"]), "ok": good, "buffers": bufs})
            if m.get("alts"):
                # the probe's mutant: the two-rounding alternative must differ from the device's output somewhere
                n = 0
                for i in m["alts"]:
                    alt = np.load(os.path.join(cdir, "%s.alt.%s.npy" % (m["id"], i)))
                    r = compare(_read(os.path.join(cdir, "%s_%s.%s" % (m["id"], tags[0][0], i))), alt, m["outs"][i],
                                tags[0][1])
                    n += max(0, r["differ"])
                res["mutants"]["probe two roundings"] = {"caught": n > 0, "differ": n}
                ok = ok and n > 0
    for k in KERNELS:
        m = negs.get(k)
        if m is None:
            res["negative_controls"][k] = None
            ok = False
            continue
        _, bufs = check_entry(cdir, m, [(neg_tag, FILLS[0])])
        caught = any(b["differ"][0] > 0 for b in bufs.values()) and all(b["differ"][0] >= 0 for b in bufs.values())
        res["negative_controls"][k] = {"case": m["name"], "half_grid": True, "caught": caught,
                                       "differ": sum(b["differ"][0] for b in bufs.values())}
        ok = ok and caught
    cases = res["cases"]
    res["summary"] = {"cases": len(cases), "cases_ok": sum(c["ok"] for c in cases),
                      "values": sum(b["values"] for c in cases for b in c["buffers"].values()),
                      "values_over_6": sum(b["over6"] for c in cases for b in c["buffers"].values()),
                      "nan_values": sum(b["nan"] for c in cases for b in c["buffers"].values()),
                      "inf_values": sum(b["inf"] for c in cases for b in c["buffers"].values()),
                      "fills": [hex(f) for f in FILLS], "pass": bool(ok)}
    return res


# ---- timing (information only) ------------------------------------------------------------------------------------

def timing(build, record, reps=10):
    rec = json.load(open(record, encoding="utf-8"))
    full_ok = [c for c in rec.get("cases", []) if c["group"] == "full" and c["ok"]]
    if not rec.get("summary", {}).get("pass") or len(full_ok) != len(FULL_GRIDS):
        raise SystemExit("timing refused: %s is not a passing --run record with the full-size grids" % record)
    cdir = os.path.join(build, "replay-net2")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    res = {"reps": reps, "max_chain_ms": MAX_CHAIN_MS, "byte_check_record": os.path.abspath(record)}
    probe = man["probe"][0]
    tag = "time_" + time.strftime("%H%M%S")
    rep, out = replay(build, cdir, chain_lines(probe, FILLS[0], tag, saves=True, time_runs=3), arena_for([probe]),
                      fname="probe.txt")
    good, _ = check_entry(cdir, probe, [(tag, FILLS[0])])
    pms = _time_of(out, probe["id"] + "_" + tag)
    res["probe"] = {"grid": list(PROBE_GRID), "ms": pms, "byte_check": good, "replay": rep}
    if pms is None or not good:
        res["error"] = "probe failed"
        return res
    sizes = {}
    for m in man["full"]:
        W2, H2 = [int(v) for v in m["name"].split()[-1].split("x")]
        predicted = pms * (W2 * H2) / (PROBE_GRID[0] * PROBE_GRID[1])
        e = {"grid": [W2, H2], "predicted_chain_ms": round(predicted, 1)}
        sizes[m["name"]] = e
        if predicted >= MAX_CHAIN_MS:
            e["skipped"] = "predicted %.0f ms (limit %.0f ms)" % (predicted, MAX_CHAIN_MS)
            continue
        rep, out = replay(build, cdir, chain_lines(m, FILLS[0], tag, saves=True, time_runs=reps), arena_for([m]),
                          fname="time_%s.txt" % m["id"])
        good, bufs = check_entry(cdir, m, [(tag, FILLS[0])])
        e["byte_check"] = good
        e["replay"] = {k: rep[k] for k in ("exit", "secs", "peak_ws_mb", "killed")}
        e["d3d12_ms"] = _time_of(out, m["id"] + "_" + tag) if good else None
        if not good:
            e["d3d12_ms_unchecked_withheld"] = True
    res["sizes"] = sizes
    res["method"] = ("kernel_replay chain: %d network runs after a first run whose every buffer was compared with the "
                     "CPU reference; mean wall time per run from submission to completion, synchronised after each run; "
                     "synthetic weights, buffers resident. Exact kernels (integer chunk arithmetic), not a fast path. "
                     "Information only." % reps)
    return res


def _time_of(out, name):
    for line in out.splitlines():
        if line.startswith("chain %s: time" % name):
            return float(line.split("time ")[1].split(" ms")[0])
    return None


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
    ap.add_argument("--groups", default="kernel,chain,mutant,full")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--record", help="--timing: the passing --run record")
    ap.add_argument("--reps", type=int, default=10)
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
              "mutants:", json.dumps({k: v["caught"] for k, v in res["mutants"].items()}))
        for c in res["cases"]:
            if not c["ok"]:
                print("FAIL", c["case"], json.dumps({i: b for i, b in c["buffers"].items()
                                                     if any(b["differ"]) or not b["slack_ok"]}))
        print("replay", json.dumps({k: res["replay"][k] for k in ("exit", "secs", "peak_ws_mb", "killed")}))
    elif a.timing:
        if not a.record:
            ap.error("--timing needs --record")
        res = timing(a.build, a.record, a.reps)
        print(json.dumps({k: v for k, v in res.items() if k != "probe"}, indent=1))
        print("probe", json.dumps({k: v for k, v in res["probe"].items() if k != "replay"}))
    if res is not None and a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    if a.run:
        return 0 if res["summary"]["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
