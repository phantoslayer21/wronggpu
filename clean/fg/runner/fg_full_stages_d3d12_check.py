# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the Full path's stage kernels (clean/dx12/kernels/fg_full_blend.hlsl, fg_out_pull.hlsl,
fg_out_push.hlsl, fg_distortion.hlsl) against the CPU reference (clean/fg/reference/stages_full.py), byte for byte.

    python clean/fg/runner/fg_full_stages_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_full_stages_d3d12_check.py --build <dir> --make [--sizes 1080p,1440p]
    python clean/fg/runner/fg_full_stages_d3d12_check.py --build <dir> --run --groups kernel,chain,mutant [--json out.json]
    python clean/fg/runner/fg_full_stages_d3d12_check.py --build <dir> --run --groups full [--json out.json]
    python clean/fg/runner/fg_full_stages_d3d12_check.py --build <dir> --timing --record r1.json,r2.json [--json t.json]

--prepare (CPU) builds the seven D3D12 kernels (clean/dx12/build_dx12.py --only; the pinned DXC through WG_TOOLCHAINS,
BUILDING.md) and kernel_replay.exe (clean/dx12/tests/kernel_replay.cpp) into <dir>.
--make (CPU) writes every case into <dir>/replay-full: input files, the kernel_replay chains and the CPU reference's
expected buffers. No GPU.
--run (GPU, inside one GPU slot) runs the chains of the chosen groups through kernel_replay on the native D3D12 device
(one process, with a memory and time watchdog) and compares the saved buffers with the CPU reference: float16 bits
(NaN equals NaN; the reference's NaN payload is not specified) and 32-bit words exactly.
- every chain runs twice, its output buffers filled with 0x5A and with 0xA5 and followed by 64 slack bytes; both runs
  must equal the reference and keep the slack, so an unwritten element or a write past the end shows in at least one
  (outputs that a stage only partly writes, push-fine's frame and pull's frame inside a larger image, start from a
  file and are compared whole);
- kernel cases: the Full blend with presets A and B at k = 1, 2, 4, 8, 16 on odd frame sizes (padding), different
  scatter and quality grids, special values (NaN, +-inf, 65504, subnormals, -0, NaN / infinite vectors, NaN weights);
  the output pull at pyramids of 9 and of more than 9 levels, with special values and with an output offset; the
  middle pull; push on levels with empty and NaN weights; push-fine; both distortion stages (low precision on / off,
  sub-rectangles, down- and upsampling, special values);
- chains: the output pyramid (pull, middle pull, push per level, push-fine) with every buffer compared, at 9 and 10
  levels; a whole Full frame (blend, pull, middle pull, push, push-fine; net2's output synthetic) for presets A and B;
  the distortion stages chained;
- negative controls: per kernel, its first kernel case with 4 or more workgroups again on half of them; mutants:
  preset A's launch (ui = 0) on preset B's data (must differ from preset B's reference) and the reverse. Each must be
  caught;
- full: every stage at 1920 x 1080 and 2560 x 1440 (k = 4, scatter and quality grids at half size), and the whole frame.
--timing (GPU, one GPU slot per size; information only) refuses unless every --record is a passing --run record and
the full cases of the timed size are in them. It runs each full-size stage chain (its byte check again on the first
run) and prints the mean milliseconds per run (kernel_replay "time": wall time per run, submission to completion,
synchronised after each run), and a 1 x 1 push launch as the per-run floor.
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

from clean.fg.reference import stages_full as S  # noqa: E402

KERNELS = ["fg_full_blend", "fg_out_pull", "fg_out_pull_mid", "fg_out_push", "fg_out_push_fine", "fg_dist_down",
           "fg_dist_fields"]
CO = "fg_full.co"
FILLS = (0x5A, 0xA5)
SLACK = 64
SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440)}
WATCH = {"ws_bytes": 3 << 30, "system_pct": 85, "secs": 80}
SPECIALS = np.array([np.nan, np.inf, -np.inf, 65504.0, -65504.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -0.0, 0.0],
                    np.float32)


def f16(a):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(a, np.float32).astype(np.float16).astype(np.float32)


def fbits(x):
    return int(np.float32(x).view(np.uint32))


def tiles(n):
    return -(-n // 16)


def groups_for(n):
    return max(1, -(-n // 256))


def _rand(rng, shape, lo, hi):
    return f16(rng.uniform(lo, hi, shape))


def _sprinkle(rng, a, frac, values):
    a = np.array(a, np.float32, copy=True)
    m = rng.random(a.shape) < frac
    a[m] = rng.choice(values, int(m.sum()))
    return f16(a)


# ---- a case: data, output buffers, launches, expected buffers --------------------------------------------------------

class Case:
    def __init__(self, name, kernel):
        self.name, self.kernel = name, kernel
        self.data, self.outs, self.kinds, self.init, self.steps, self.expect = {}, {}, {}, {}, [], {}
        self.alt = {}    # out id -> {mutant name: the buffer a wrong rounding would give}

    def buf(self, i, raw):
        self.data[i] = raw
        return "@" + i

    def h(self, i, arr):
        return self.buf(i, np.ascontiguousarray(np.asarray(arr, np.float32).astype("<f2")).tobytes())

    def f(self, i, arr):
        return self.buf(i, np.ascontiguousarray(np.asarray(arr, "<f4")).tobytes())

    def u(self, i, arr, dt):
        return self.buf(i, np.ascontiguousarray(np.asarray(arr, dt)).tobytes())

    def out(self, i, expect, kind="f16", init=None):
        """An output buffer and its expected content (float16 values, or 32-bit words with kind "u32"); init: the
        buffer's starting bytes (a partly written output), else the fill."""
        raw = np.ascontiguousarray(np.asarray(expect, np.float32).astype("<f2")).view("<u2") if kind == "f16" else \
            np.ascontiguousarray(np.asarray(expect, "<u4"))
        self.expect[i] = raw.reshape(-1)
        self.kinds[i] = kind
        self.outs[i] = int(raw.nbytes)
        if init is not None:
            self.init[i] = np.ascontiguousarray(np.asarray(init, np.float32).astype("<f2")).tobytes()
        return "@" + i

    def step(self, kernel, groups, args):
        self.steps.append((kernel, int(groups), [str(a) for a in args]))


# ---- launches (argument order of the kernels) ------------------------------------------------------------------------

def blend_launch(c, A, Wo, Ho, W1, H1, Wd, Hd, k, ui):
    Wp, Hp = S.padded(Wo, Ho, k)
    c.step("fg_full_blend", (Wp // 16) * (Hp // 16),
           [A["cp"], A["cc"], A["wimg"], A["qc"], A["ma"], A["mb"], A.get("up", "null"), A.get("uc", "null"),
            A["x0"], A["x1"], A["xw"], A["ib"], A["ic"], A["ia"], Wo, Ho, W1, H1, Wd, Hd, k, Wp // 16, ui])


def pull_launch(c, A, Wo, Ho, W2, H2, k, pitch=None, ox=0, oy=0):
    c.step("fg_out_pull", tiles(Wo) * tiles(Ho),
           [A["x0"], A["x1"], A["xw"], A["n2"], A["sig"], A["out"], A["fine"], A["lv1"], A["lv2"], A["lv3"], A["lv4"],
            Wo, Ho, W2, H2, k, tiles(Wo), Wo if pitch is None else pitch, ox, oy, fbits(S.F.rcp(np.float32(k)))])


def mid_launch(c, A, Wo, Ho):
    w4, h4 = S.level_size(Wo, Ho, 4)
    c.step("fg_out_pull_mid", tiles(w4) * tiles(h4),
           [A["lv4"], A["lv5"], A["lv6"], A["lv7"], A["lv8"], w4, h4, tiles(w4), Wo, Ho])


def push_launch(c, own, coarse, filled, Wo, Ho, l):
    wf, hf = S.level_size(Wo, Ho, l)
    wc, hc = S.level_size(Wo, Ho, l + 1)
    c.step("fg_out_push", groups_for(wf * hf), [own, coarse, filled, wf, hf, wc, hc])


def fine_launch(c, A, Wo, Ho, pitch=None, ox=0, oy=0):
    w1, h1 = S.level_size(Wo, Ho, 1)
    c.step("fg_out_push_fine", groups_for(Wo * Ho), [A["fine"], A["fd1"], A["out"], Wo, Ho, w1, h1,
                                                    Wo if pitch is None else pitch, ox, oy])


def down_launch(c, field, out, W1, H1, region, lowp):
    b, s = (0.0, 1.0) if lowp is None else lowp
    c.step("fg_dist_down", groups_for(W1 * H1), [field, out, W1, H1] + region.args() + [fbits(b), fbits(s)])


def fields_launch(c, A, W1, H1):
    c.step("fg_dist_fields", groups_for(W1 * H1), [A["dc"], A["dp"], A["zc"], A["mv"], A["fpc"], A["zd"], A["mcp"],
                                                   A["fout"], W1, H1])


def pyramid_launches(c, A, Wo, Ho):
    """Middle pull (more than 9 levels), push from the coarsest level down, push-fine."""
    Lc = S.coarsest_level(Wo, Ho)
    if Lc == 8:
        mid_launch(c, A, Wo, Ho)
    for l in range(Lc - 1, 0, -1):
        push_launch(c, A["lv%d" % l], A["lv%d" % Lc] if l == Lc - 1 else A["fd%d" % (l + 1)], A["fd%d" % l], Wo, Ho, l)
    fine_launch(c, A, Wo, Ho)


# ---- synthetic inputs ------------------------------------------------------------------------------------------------

def blend_data(rng, Wo, Ho, W1, H1, Wd, Hd, *, amp=0.02, empty=0.25, ui=False, specials=False):
    d = dict(Cp=_rand(rng, (Ho, Wo, 4), 0, 1), Cc=_rand(rng, (Ho, Wo, 4), 0, 1), Wimg=_rand(rng, (H1, W1, 2), 0, 1),
             Qc=_rand(rng, (Hd, Wd, 2), 0, 1))
    for key in ("MA", "MB"):
        w = S.pack_mv(rng.uniform(1, 600, (H1, W1)).astype(np.float32), rng.uniform(-amp, amp, (H1, W1)),
                      rng.uniform(-amp, amp, (H1, W1)))
        d[key] = np.where(rng.random((H1, W1)) < empty, np.uint64(S.EMPTY), w).astype(np.uint64)
    # blocks where both scatters are empty, so that holes occur in runs
    for _ in range(3):
        y, x = rng.integers(0, H1), rng.integers(0, W1)
        d["MA"][y:y + 4, x:x + 5] = S.EMPTY
        d["MB"][y:y + 4, x:x + 5] = S.EMPTY
    if specials:
        for key in ("Cp", "Cc"):
            d[key] = _sprinkle(rng, d[key], 0.03, SPECIALS)
        d["Wimg"] = _sprinkle(rng, d["Wimg"], 0.03, SPECIALS)
        d["Qc"] = _sprinkle(rng, d["Qc"], 0.03, SPECIALS)
        sp = [(np.nan, 0.0), (np.inf, -np.inf), (0.5, -0.5), (-0.0, 0.0), (2.0 ** -24, 60000.0), (-3.0, 7.0)]
        for i, (vx, vy) in enumerate(sp):
            d["MA" if i % 2 else "MB"][i % H1, (3 * i) % W1] = S.pack_mv(np.float32(9), np.float32(vx), np.float32(vy))
    ui_maps = None
    if ui:
        ui_maps = []
        for _ in range(2):
            a = np.where(rng.random((Ho, Wo)) < 0.35, rng.uniform(0.0, 1.0, (Ho, Wo)), 0.0)
            a[rng.random((Ho, Wo)) < 0.05] = 2.0 ** -23                   # at the threshold: neither > nor <
            rgb = rng.uniform(0, 1, (Ho, Wo, 3)) * a[..., None]
            m = f16(np.concatenate([rgb, a[..., None]], -1))
            if specials:
                m = _sprinkle(rng, m, 0.02, SPECIALS)
            ui_maps.append(m)
        ui_maps = tuple(ui_maps)
    return d, ui_maps


def blend_case(name, rng, Wo, Ho, W1, H1, Wd, Hd, k, *, ui=False, specials=False, launch_ui=None, expect_ui=None):
    """launch_ui / expect_ui override the launch's preset flag / the reference's preset (mutants)."""
    c = Case(name, "fg_full_blend")
    d, ui_maps = blend_data(rng, Wo, Ho, W1, H1, Wd, Hd, ui=ui, specials=specials)
    with np.errstate(all="ignore"):
        r = S.g4_full_blend(**d, k=k, ui=ui_maps if (expect_ui is None and ui) or expect_ui else None)
    A = {"cp": c.h("cp", d["Cp"]), "cc": c.h("cc", d["Cc"]), "wimg": c.h("wimg", d["Wimg"]), "qc": c.h("qc", d["Qc"]),
         "ma": c.u("ma", d["MA"], "<u8"), "mb": c.u("mb", d["MB"], "<u8")}
    if ui:
        A["up"], A["uc"] = c.h("up", ui_maps[0]), c.h("uc", ui_maps[1])
    for key, i in (("X0", "x0"), ("X1", "x1"), ("Xw", "xw"), ("Ib", "ib"), ("Ic", "ic"), ("Ia", "ia")):
        A[i] = c.out(i, r[key])
    blend_launch(c, A, Wo, Ho, W1, H1, Wd, Hd, k, int(ui if launch_ui is None else launch_ui))
    return c


def pull_data(rng, Wo, Ho, k, *, specials=False, hole_blocks=4):
    W2, H2 = S.padded(Wo, Ho, k)[0] // k, S.padded(Wo, Ho, k)[1] // k
    X0, X1 = _rand(rng, (Ho, Wo, 4), -0.1, 1.1), _rand(rng, (Ho, Wo, 4), -0.1, 1.1)
    Xw = (rng.random((Ho, Wo)) < 0.08).astype(np.float32)
    for _ in range(hole_blocks):
        y, x = rng.integers(0, Ho), rng.integers(0, Wo)
        Xw[y:y + rng.integers(3, 12), x:x + rng.integers(3, 30)] = 1.0
    N2 = np.concatenate([_rand(rng, (H2, W2, 4), -1.5, 1.5), _rand(rng, (H2, W2, 1), -5, 5),
                         _rand(rng, (H2, W2, 3), -9, 9)], -1)
    if specials:
        X0 = _sprinkle(rng, X0, 0.02, SPECIALS)
        X1 = _sprinkle(rng, X1, 0.02, SPECIALS)
        N2 = _sprinkle(rng, N2, 0.03, SPECIALS)
        Xw[rng.random((Ho, Wo)) < 0.01] = np.nan
    return X0, X1, Xw, N2, W2, H2


def pull_case(name, rng, Wo, Ho, k, *, specials=False, offset=None):
    c = Case(name, "fg_out_pull")
    X0, X1, Xw, N2, W2, H2 = pull_data(rng, Wo, Ho, k, specials=specials)
    with np.errstate(all="ignore"):
        p = S.g6_output_pull(X0, X1, Xw, N2, k=k)
    A = {"x0": c.h("x0", X0), "x1": c.h("x1", X1), "xw": c.h("xw", Xw), "n2": c.h("n2", N2),
         "sig": c.f("sig", S.sig_table())}
    pitch, ox, oy = None, 0, 0
    if offset is None:
        A["out"] = c.out("out", p["out"])
    else:
        ox, oy, pw, ph = offset
        pitch = pw
        bg = _rand(rng, (ph, pw, 4), -2, 2)
        exp = bg.copy()
        exp[oy:oy + Ho, ox:ox + Wo] = p["out"]
        A["out"] = c.out("out", exp, init=bg)
    A["fine"] = c.out("fine", p["fine"])
    for l in range(1, 5):
        A["lv%d" % l] = c.out("lv%d" % l, p["levels"][l])
    pull_launch(c, A, Wo, Ho, W2, H2, k, pitch, ox, oy)
    return c


def level_data(rng, Wo, Ho, l, zero_frac=0.3, nan=False):
    lw, lh = S.level_size(Wo, Ho, l)
    L = _rand(rng, (lh, lw, 4), 0, 1)
    L[..., 3] = np.where(rng.random((lh, lw)) < zero_frac, 0.0, L[..., 3])
    L[..., 3] = np.where(rng.random((lh, lw)) < 0.05, -0.0, L[..., 3])
    if nan:
        L[0, 0, 3] = np.nan
        L[-1, -1, 0] = np.inf
    return f16(L)


def mid_case(name, rng, Wo, Ho):
    c = Case(name, "fg_out_pull_mid")
    L4 = level_data(rng, Wo, Ho, 4, nan=True)
    mid = S.g6_output_pull_mid(L4, Wo, Ho)
    A = {"lv4": c.h("lv4", L4)}
    for l in range(5, 9):
        A["lv%d" % l] = c.out("lv%d" % l, mid[l])
    mid_launch(c, A, Wo, Ho)
    return c


def push_case(name, rng, Wo, Ho, l, *, nan=False):
    c = Case(name, "fg_out_push")
    own = level_data(rng, Wo, Ho, l, zero_frac=0.5, nan=nan)
    coarse = level_data(rng, Wo, Ho, l + 1, zero_frac=0.2, nan=nan)
    lw, lh = S.level_size(Wo, Ho, l)
    with np.errstate(all="ignore"):
        up = f16(S.upsample(coarse, lw, lh))
        exp = np.where(~(own[..., 3:4] <= 0), own, up)
    push_launch(c, c.h("own", own), c.h("coarse", coarse), c.out("filled", exp), Wo, Ho, l)
    return c


def fine_case(name, rng, Wo, Ho, offset=None):
    c = Case(name, "fg_out_push_fine")
    fine = (rng.random((Ho, Wo)) < 0.3).astype(np.float32)
    fd1 = level_data(rng, Wo, Ho, 1, zero_frac=0.1)
    ox, oy, pw, ph = offset if offset is not None else (0, 0, Wo, Ho)
    bg = _rand(rng, (ph, pw, 4), -2, 2)
    exp = bg.copy()
    exp[oy:oy + Ho, ox:ox + Wo] = S.g6_output_push_fine(bg[oy:oy + Ho, ox:ox + Wo], fine, fd1)
    A = {"fine": c.h("fine", fine), "fd1": c.h("fd1", fd1), "out": c.out("out", exp, init=bg)}
    fine_launch(c, A, Wo, Ho, pw, ox, oy)
    return c


def down_case(name, rng, W1, H1, region, lowp, specials=False):
    c = Case(name, "fg_dist_down")
    field = rng.uniform(-1, 1, (region.th, region.tw, 4)).astype(np.float32) * 0.05
    if specials:
        field.reshape(-1)[::7] = np.resize(np.array([np.nan, np.inf, -np.inf, 1e-40, -0.0, 7e4], np.float32),
                                           field.reshape(-1)[::7].shape)
    with np.errstate(all="ignore"):
        exp = S.r5_downsample(field, region, W1, H1, lowp)
    down_launch(c, c.f("field", field), c.out("out", exp), W1, H1, region, lowp)
    return c


def probe_case():
    """Adversarial probe of the stage kernels' float16 write and fused multiply-add (fg_dist_down: out = h(fma(scale,
    field, bias))): float16 midpoints and ties both ways, subnormal float16 results and their ties, the overflow
    boundary 65520, zeros of both signs, NaN and infinities; and with scale = 1 + 2^-12, bias = 2^-26 the field value
    1 + 2^-12, whose exact product lies 2^-24 above the float16 midpoint 1 + 2^-11 so that only a single rounding of
    a b + c lands above it (1 + 2^-10) while a rounded product plus c ties back to 1. Mutants it must catch: the product
    rounded before the addition, and a truncating float16 conversion."""
    c = Case("probe f16 write and fma", "fg_dist_down")
    e = 2.0 ** -11
    vals = [1 + 2 ** -12, 1.0 + e, 1.0 + 3 * e, 1.0 - e / 2, 1.0 + e + 2.0 ** -23, 1.0 + e - 2.0 ** -23, 2.0 ** -25,
            3 * 2.0 ** -25, 2.0 ** -24 + 2.0 ** -26, 65519.0, 65520.0, 65504.0 + 8.0, -65520.0, 0.0, -0.0, np.nan,
            np.inf, -np.inf, 1e-30, -1e-30, 2049.0, 2051.0, -2049.0, 0.1, 1.0 / 3.0, 2.0 ** -14 * (1 + e), 6e-8]
    n = 64
    field = np.zeros((n, 1, 4), np.float32)
    v = np.array(vals + [1 + 2 ** -12] * (n - len(vals)), np.float32)
    rng = np.random.default_rng(7)
    v[len(vals):] = (rng.integers(1, 2 ** 11, n - len(vals)) * 2 + 1) * 2.0 ** -11 * 2.0 ** rng.integers(-24, 15, n - len(vals))
    field[:, 0, 0], field[:, 0, 1], field[:, 0, 2], field[:, 0, 3] = v, -v, v * 0.5, v * 2
    reg = S.Region(1, n)
    lowp = (2.0 ** -26, 1 + 2 ** -12)
    with np.errstate(all="ignore"):
        exp = S.r5_downsample(field, reg, 1, n, lowp)
        b, sc = np.float32(lowp[0]), np.float32(lowp[1])
        two = np.float32(np.float32(sc * field) + b)
        # a truncating conversion of the single-rounding result: the float16 value toward zero
        fused = S.F.fma(sc, field, b)
        h16 = fused.astype(np.float16)
        back = h16.astype(np.float32)
        tz = np.where(np.abs(back) > np.abs(fused), np.nextafter(h16, np.float16(0)), h16).astype(np.float32)
    c.alt["out"] = {"two roundings": f16(two), "truncating conversion": tz}
    down_launch(c, c.f("field", field), c.out("out", exp), 1, n, reg, lowp)
    return c


def fields_data(rng, W1, H1, specials=False):
    Dc, Dp = _rand(rng, (H1, W1, 4), -0.08, 0.08), _rand(rng, (H1, W1, 4), -0.08, 0.08)
    Zc = rng.integers(0, 2 ** 32, (H1, W1), dtype=np.uint64).astype(np.uint32)
    Mv = rng.uniform(-0.05, 0.05, (H1, W1, 2)).astype(np.float32)
    Fp = _rand(rng, (H1, W1, 2), -6, 6)
    if specials:
        Dc, Dp = _sprinkle(rng, Dc, 0.02, SPECIALS), _sprinkle(rng, Dp, 0.02, SPECIALS)
        Fp = _sprinkle(rng, Fp, 0.02, SPECIALS)
        Mv.reshape(-1)[::23] = np.resize(np.array([np.nan, np.inf, -0.0, 3.0], np.float32), Mv.reshape(-1)[::23].shape)
    return Dc, Dp, Zc, Mv, Fp


def fields_case(name, rng, W1, H1, specials=False):
    c = Case(name, "fg_dist_fields")
    Dc, Dp, Zc, Mv, Fp = fields_data(rng, W1, H1, specials)
    with np.errstate(all="ignore"):
        r = S.r5_fields(Dc, Dp, Zc, Mv, Fp)
    A = {"dc": c.h("dc", Dc), "dp": c.h("dp", Dp), "zc": c.u("zc", Zc, "<u4"), "mv": c.f("mv", Mv), "fpc": c.h("fpc", Fp),
         "zd": c.out("zd", r["Zd"], "u32"), "mcp": c.out("mcp", r["Mcp"]), "fout": c.out("fout", r["Fpc"])}
    fields_launch(c, A, W1, H1)
    return c


def kernel_cases(rng):
    cs = []
    # the blend: presets A and B, every k, odd frames, scatter / quality grids of other sizes
    for k, (Wo, Ho, W1, H1, Wd, Hd) in ((1, (37, 21, 37, 21, 18, 10)), (2, (45, 27, 23, 14, 22, 13)),
                                        (4, (70, 41, 35, 21, 35, 20)), (8, (61, 35, 31, 18, 30, 17)),
                                        (16, (40, 23, 20, 12, 20, 11))):
        for ui in (False, True):
            cs.append(blend_case("blend %s k%d %dx%d" % ("B" if ui else "A", k, Wo, Ho), rng, Wo, Ho, W1, H1, Wd, Hd,
                                 k, ui=ui))
    cs.append(blend_case("blend A specials k4 48x32", rng, 48, 32, 48, 32, 24, 16, 4, specials=True))
    cs.append(blend_case("blend B specials k2 50x30", rng, 50, 30, 25, 15, 25, 15, 2, ui=True, specials=True))
    # the output pull
    cs += [pull_case("pull k2 48x40 (9 levels)", rng, 48, 40, 2),
           pull_case("pull k4 333x45 (10 levels)", rng, 333, 45, 4),
           pull_case("pull k8 70x30", rng, 70, 30, 8),
           pull_case("pull k1 40x24", rng, 40, 24, 1),
           pull_case("pull specials k4 64x48", rng, 64, 48, 4, specials=True),
           pull_case("pull offset k2 36x22", rng, 36, 22, 2, offset=(5, 3, 47, 29))]
    cs += [mid_case("pull_mid 2000x300", rng, 2000, 300), mid_case("pull_mid 333x45", rng, 333, 45),
           mid_case("pull_mid 4100x2300", rng, 4100, 2300)]
    cs += [push_case("push 300x200 l1", rng, 300, 200, 1), push_case("push 300x200 l3", rng, 300, 200, 3),
           push_case("push 1000x600 l7", rng, 1000, 600, 7), push_case("push nan 77x41 l1", rng, 77, 41, 1, nan=True),
           push_case("push 333x45 l4", rng, 333, 45, 4)]
    cs += [fine_case("push_fine 64x40", rng, 64, 40), fine_case("push_fine 37x23", rng, 37, 23),
           fine_case("push_fine offset 30x20", rng, 30, 20, offset=(4, 2, 41, 27))]
    cs += [down_case("dist_down 40x24 lowp", rng, 40, 24, S.Region(60, 40, 7, 5, 40, 24), (0.25, -1.5)),
           down_case("dist_down 40x24", rng, 40, 24, S.Region(40, 24), None),
           down_case("dist_down half 32x18", rng, 32, 18, S.Region(70, 45, 3, 4, 64, 36), None),
           down_case("dist_down up 48x30", rng, 48, 30, S.Region(20, 12, 2, 1, 12, 7), (0.0, 2.0)),
           down_case("dist_down specials 33x17", rng, 33, 17, S.Region(33, 17), (0.125, 0.5), specials=True)]
    cs.append(probe_case())
    cs += [fields_case("dist_fields 40x24", rng, 40, 24), fields_case("dist_fields 71x33", rng, 71, 33),
           fields_case("dist_fields specials 32x32", rng, 32, 32, specials=True)]
    return cs


# ---- chains ----------------------------------------------------------------------------------------------------------

def pyramid_outs(c, A, out_frame, Wo, Ho):
    """Output buffers of the pyramid after the pull (levels, filled levels, the final frame)."""
    Lc = S.coarsest_level(Wo, Ho)
    for l in range(1, 9):
        if l in out_frame["levels"]:
            A["lv%d" % l] = c.out("lv%d" % l, out_frame["levels"][l])
    for l in range(1, Lc):
        A["fd%d" % l] = c.out("fd%d" % l, out_frame["filled"][l])
    A["out"] = c.out("out", out_frame["frame"])
    A["fine"] = c.out("fine", out_frame["pull"]["fine"])


def pyramid_chain(name, rng, Wo, Ho, k, specials=False):
    c = Case(name, "chain")
    X0, X1, Xw, N2, W2, H2 = pull_data(rng, Wo, Ho, k, specials=specials, hole_blocks=8)
    with np.errstate(all="ignore"):
        o = S.g6_output(X0, X1, Xw, N2, k=k)
    A = {"x0": c.h("x0", X0), "x1": c.h("x1", X1), "xw": c.h("xw", Xw), "n2": c.h("n2", N2),
         "sig": c.f("sig", S.sig_table())}
    pyramid_outs(c, A, o, Wo, Ho)
    pull_launch(c, A, Wo, Ho, W2, H2, k)
    pyramid_launches(c, A, Wo, Ho)
    return c


def frame_inputs(rng, Wo, Ho, k, ui, d=None):
    W1, H1, Wd, Hd = Wo // 2, Ho // 2, Wo // 2, Ho // 2
    data, ui_maps = blend_data(rng, Wo, Ho, W1, H1, Wd, Hd, amp=0.01, empty=0.05, ui=ui)
    Wp, Hp = S.padded(Wo, Ho, k)
    N2 = np.concatenate([_rand(rng, (Hp // k, Wp // k, 4), -1.0, 1.0), _rand(rng, (Hp // k, Wp // k, 1), -4, 4),
                         _rand(rng, (Hp // k, Wp // k, 3), -2, 2)], -1)
    return data, ui_maps, N2, (W1, H1, Wd, Hd)


def frame_chain(name, rng, Wo, Ho, k, ui):
    """The Full frame: blend, then the output pyramid on its X0 / X1 / Xw (net2's output synthetic)."""
    c = Case(name, "chain")
    d, ui_maps, N2, (W1, H1, Wd, Hd) = frame_inputs(rng, Wo, Ho, k, ui)
    r = S.g4_full_blend(**d, k=k, ui=ui_maps)
    o = S.g6_output(r["X0"], r["X1"], r["Xw"], N2, k=k)
    A = {"cp": c.h("cp", d["Cp"]), "cc": c.h("cc", d["Cc"]), "wimg": c.h("wimg", d["Wimg"]), "qc": c.h("qc", d["Qc"]),
         "ma": c.u("ma", d["MA"], "<u8"), "mb": c.u("mb", d["MB"], "<u8"), "n2": c.h("n2", N2),
         "sig": c.f("sig", S.sig_table())}
    if ui:
        A["up"], A["uc"] = c.h("up", ui_maps[0]), c.h("uc", ui_maps[1])
    for key, i in (("X0", "x0"), ("X1", "x1"), ("Xw", "xw"), ("Ib", "ib"), ("Ic", "ic"), ("Ia", "ia")):
        A[i] = c.out(i, r[key])
    pyramid_outs(c, A, o, Wo, Ho)
    W2, H2 = N2.shape[1], N2.shape[0]
    blend_launch(c, A, Wo, Ho, W1, H1, Wd, Hd, k, int(ui))
    pull_launch(c, A, Wo, Ho, W2, H2, k)
    pyramid_launches(c, A, Wo, Ho)
    return c


def dist_chain(name, rng, W1, H1, tw, th, lowp):
    """Downsample of the current and the previous field, then the distorted fields."""
    c = Case(name, "chain")
    reg = S.Region(tw, th, 1, 2, tw - 3, th - 4)
    fc = rng.uniform(-0.05, 0.05, (th, tw, 4)).astype(np.float32)
    fp = rng.uniform(-0.05, 0.05, (th, tw, 4)).astype(np.float32)
    Dc, Dp = S.r5_downsample(fc, reg, W1, H1, lowp), S.r5_downsample(fp, reg, W1, H1, lowp)
    _, _, Zc, Mv, Fp = fields_data(rng, W1, H1)
    r = S.r5_fields(Dc, Dp, Zc, Mv, Fp)
    A = {"fc": c.f("fc", fc), "fp": c.f("fp", fp), "zc": c.u("zc", Zc, "<u4"), "mv": c.f("mv", Mv),
         "fpc": c.h("fpc", Fp), "dc": c.out("dc", Dc), "dp": c.out("dp", Dp), "zd": c.out("zd", r["Zd"], "u32"),
         "mcp": c.out("mcp", r["Mcp"]), "fout": c.out("fout", r["Fpc"])}
    down_launch(c, A["fc"], A["dc"], W1, H1, reg, lowp)
    down_launch(c, A["fp"], A["dp"], W1, H1, reg, lowp)
    fields_launch(c, A, W1, H1)
    return c


def chain_cases(rng):
    return [pyramid_chain("pyramid k2 48x40 (9 levels)", rng, 48, 40, 2),
            pyramid_chain("pyramid k4 333x45 (10 levels)", rng, 333, 45, 4),
            pyramid_chain("pyramid specials k4 300x64", rng, 300, 64, 4, specials=True),
            frame_chain("frame A k4 160x96", rng, 160, 96, 4, False),
            frame_chain("frame B k2 300x70", rng, 300, 70, 2, True),
            dist_chain("distortion 64x36", rng, 64, 36, 140, 80, (0.0, 1.0)),
            dist_chain("distortion lowp 50x30", rng, 50, 30, 100, 64, (-0.5, 1.25))]


def mutant_cases(rng):
    """Preset A's launch on preset B's data (reference: preset B) and preset B's launch on it with the reference of
    preset A: both must differ from their reference."""
    return [blend_case("mutant A launch on B", rng, 45, 27, 23, 14, 22, 13, 2, ui=True, launch_ui=0, expect_ui=True),
            blend_case("mutant B launch on A reference", rng, 45, 27, 23, 14, 22, 13, 2, ui=True, launch_ui=1,
                       expect_ui=False)]


# ---- full-size stage cases (parity and timing) --------------------------------------------------------------------

def full_cases(label, Wo, Ho, seed=20261121):
    """The full-size cases, one at a time (a generator: each is written and dropped before the next is built)."""
    rng = np.random.default_rng(seed + Wo)
    k = S.net2_k(Ho)
    W1, H1 = Wo // 2, Ho // 2
    d, ui_maps, N2, (W1, H1, Wd, Hd) = frame_inputs(rng, Wo, Ho, k, True)
    W2, H2 = N2.shape[1], N2.shape[0]
    rA = S.g4_full_blend(**d, k=k)
    rB = S.g4_full_blend(**d, k=k, ui=ui_maps)
    for key in ("x0", "x1", "hole"):
        rB.pop(key)
    o = S.g6_output(rA["X0"], rA["X1"], rA["Xw"], N2, k=k)
    for key in ("x0", "x1", "hole"):
        rA.pop(key)
    o["pull"].pop("c")
    o["pull"].pop("w")
    for key in ("Cp", "Cc"):                   # float16 values: kept as float16 from here (memory)
        d[key] = d[key].astype(np.float16)
    ui_maps = tuple(m.astype(np.float16) for m in ui_maps)
    for r in (rA, rB):
        for key in ("X0", "X1"):
            r[key] = r[key].astype(np.float16)
    sig = S.sig_table()

    def blend_inputs(c, ui):
        A = {"cp": c.h("cp", d["Cp"]), "cc": c.h("cc", d["Cc"]), "wimg": c.h("wimg", d["Wimg"]),
             "qc": c.h("qc", d["Qc"]), "ma": c.u("ma", d["MA"], "<u8"), "mb": c.u("mb", d["MB"], "<u8")}
        if ui:
            A["up"], A["uc"] = c.h("up", ui_maps[0]), c.h("uc", ui_maps[1])
        return A
    for ui, r in ((False, rA), (True, rB)):
        c = Case("full %s blend %s" % (label, "B" if ui else "A"), "fg_full_blend")
        A = blend_inputs(c, ui)
        for key, i in (("X0", "x0"), ("X1", "x1"), ("Xw", "xw"), ("Ib", "ib"), ("Ic", "ic"), ("Ia", "ia")):
            A[i] = c.out(i, r[key])
        blend_launch(c, A, Wo, Ho, W1, H1, Wd, Hd, k, int(ui))
        yield c
    del rB
    p = o["pull"]
    c = Case("full %s pull" % label, "fg_out_pull")
    A = {"x0": c.h("x0", rA["X0"]), "x1": c.h("x1", rA["X1"]), "xw": c.h("xw", rA["Xw"]), "n2": c.h("n2", N2),
         "sig": c.f("sig", sig), "out": c.out("out", p["out"]), "fine": c.out("fine", p["fine"])}
    for l in range(1, 5):
        A["lv%d" % l] = c.out("lv%d" % l, p["levels"][l])
    pull_launch(c, A, Wo, Ho, W2, H2, k)
    yield c
    c = Case("full %s pull_mid" % label, "fg_out_pull_mid")
    A = {"lv4": c.h("lv4", o["levels"][4])}
    for l in range(5, 9):
        A["lv%d" % l] = c.out("lv%d" % l, o["levels"][l])
    mid_launch(c, A, Wo, Ho)
    yield c
    c = Case("full %s push" % label, "fg_out_push")
    A = {"lv%d" % l: c.h("lv%d" % l, o["levels"][l]) for l in range(1, 9)}
    for l in range(1, 8):
        A["fd%d" % l] = c.out("fd%d" % l, o["filled"][l])
    for l in range(7, 0, -1):
        push_launch(c, A["lv%d" % l], A["lv8"] if l == 7 else A["fd%d" % (l + 1)], A["fd%d" % l], Wo, Ho, l)
    yield c
    c = Case("full %s push_fine" % label, "fg_out_push_fine")
    A = {"fine": c.h("fine", p["fine"]), "fd1": c.h("fd1", o["filled"][1]),
         "out": c.out("out", o["frame"], init=p["out"])}
    fine_launch(c, A, Wo, Ho)
    yield c
    c = Case("full %s frame A" % label, "chain")
    A = blend_inputs(c, False)
    A.update({"n2": c.h("n2", N2), "sig": c.f("sig", sig)})
    for key, i in (("X0", "x0"), ("X1", "x1"), ("Xw", "xw"), ("Ib", "ib"), ("Ic", "ic"), ("Ia", "ia")):
        A[i] = c.out(i, rA[key])
    pyramid_outs(c, A, o, Wo, Ho)
    blend_launch(c, A, Wo, Ho, W1, H1, Wd, Hd, k, 0)
    pull_launch(c, A, Wo, Ho, W2, H2, k)
    pyramid_launches(c, A, Wo, Ho)
    yield c
    del c, A, d, o, p, rA
    yield dist_chain("full %s distortion" % label, rng, W1, H1, Wo, Ho, None)


def floor_case():
    """A 1 x 1 push launch: the per-run floor of a timed chain."""
    c = Case("floor push 1x1", "fg_out_push")
    own = np.zeros((1, 1, 4), np.float32)
    coarse = np.array([[[0.5, 0.5, 0.5, 1.0]]], np.float32)
    c.step("fg_out_push", 1, [c.h("own", own), c.h("coarse", coarse),
                              c.out("filled", f16(S.upsample(coarse, 1, 1))), 1, 1, 1, 1])
    return c


# ---- files ----------------------------------------------------------------------------------------------------------

def _id(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def write_case(cdir, case):
    cid = _id(case.name)
    files, inits = {}, {}
    for i, b in case.data.items():
        fn = "%s.%s" % (cid, i)
        with open(os.path.join(cdir, fn), "wb") as f:
            f.write(b)
        files[i] = fn
    for i, b in case.init.items():
        fn = "%s.init.%s" % (cid, i)
        with open(os.path.join(cdir, fn), "wb") as f:
            f.write(b)
        inits[i] = fn
    for i, a in case.expect.items():
        np.save(os.path.join(cdir, "%s.expect.%s.npy" % (cid, i)), a)
    alts = {}
    for i, d in case.alt.items():
        for j, (mname, arr) in enumerate(d.items()):
            np.save(os.path.join(cdir, "%s.alt%d.%s.npy" % (cid, j, i)),
                    np.ascontiguousarray(np.asarray(arr, np.float32).astype("<f2")).view("<u2").reshape(-1))
            alts.setdefault(i, []).append(mname)
    return {"alts": alts, "name": case.name, "id": cid, "kernel": case.kernel, "files": files, "inits": inits, "outs": case.outs,
            "kinds": case.kinds, "sizes": {i: len(b) for i, b in case.data.items()}, "steps": case.steps,
            "launch_kernels": sorted({s[0] for s in case.steps})}


def chain_lines(m, fill, tag, half=None, saves=True, time_runs=0, ui_flip=None):
    """kernel_replay chain lines of manifest entry m with output fill `fill`; half: a step index on half its grid."""
    name = "%s_%s" % (m["id"], tag)
    lines = ["chain " + name, "module " + CO]
    for i, fn in m["files"].items():
        lines.append("buf %s %d %s" % (i, m["sizes"][i], fn))
    for i, n in m["outs"].items():
        lines.append("buf %s %d %s %d" % (i, n + SLACK, m["inits"].get(i, "-"), fill))
    for j, (k, g, a) in enumerate(m["steps"]):
        lines.append("step %s %d 256 %s" % (k, max(1, g // 2) if j == half else g, " ".join(a)))
    if saves:
        for i in m["outs"]:
            lines.append("save %s %s.%s" % (i, name, i))
    if time_runs:
        lines.append("time %d" % time_runs)
    return lines + [""]


def make(build, sizes=()):
    cdir = os.path.join(build, "replay-full")
    os.makedirs(cdir, exist_ok=True)
    rng = np.random.default_rng(20261121)
    groups = {"kernel": kernel_cases(rng), "chain": chain_cases(rng), "mutant": mutant_cases(rng)}
    man = {}
    for g, cases in groups.items():
        man[g] = [write_case(cdir, c) for c in cases]
    man["floor"] = [write_case(cdir, floor_case())]
    for label in sizes:
        Wo, Ho = SIZES[label]
        man["full_" + label] = [write_case(cdir, c) for c in full_cases(label, Wo, Ho)]
    path = os.path.join(cdir, "manifest.json")
    if os.path.exists(path):                 # keep full-size groups made by an earlier --make
        old = json.load(open(path, encoding="utf-8"))
        for g, v in old.items():
            if g.startswith("full_") and g not in man:
                man[g] = v
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=0)
    print("made", json.dumps({g: len(v) for g, v in man.items()}), "in", cdir)
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

def compare(raw, expect, kind, nbytes, fill):
    """Saved bytes against the expected buffer (float16 bits with NaN equal to NaN, or 32-bit words) and the slack."""
    a = np.frombuffer(raw, np.uint8)
    if a.size != nbytes + SLACK:
        return {"values": int(expect.size), "differ": -1, "slack_ok": False, "missing": True}
    if kind == "f16":
        g = a[:nbytes].view("<u2")
        r = expect.view("<u2")
        gn = ((g & 0x7C00) == 0x7C00) & ((g & 0x3FF) != 0)
        rn = ((r & 0x7C00) == 0x7C00) & ((r & 0x3FF) != 0)
        same = (g == r) | (gn & rn)
        extra = {"nan": int(np.sum(rn)), "inf": int(np.sum((r & 0x7FFF) == 0x7C00))}
    else:
        g = a[:nbytes].view("<u4")
        same = g == expect.view("<u4")
        extra = {"nan": 0, "inf": 0}
    return {"values": int(g.size), "differ": int(np.sum(~same)), "slack_ok": bool(np.all(a[nbytes:] == fill)), **extra}


def _read(path):
    return open(path, "rb").read() if os.path.exists(path) else b""


def check_entry(cdir, m, tags_fills):
    bufs, ok = {}, True
    for i, n in m["outs"].items():
        exp = np.load(os.path.join(cdir, "%s.expect.%s.npy" % (m["id"], i)))
        per = []
        for tag, fill in tags_fills:
            r = compare(_read(os.path.join(cdir, "%s_%s.%s" % (m["id"], tag, i))), exp, m["kinds"][i], n, fill)
            per.append(r)
            ok = ok and r["differ"] == 0 and r["slack_ok"]
        bufs[i] = {"values": per[0]["values"], "differ": [r["differ"] for r in per],
                   "slack_ok": all(r["slack_ok"] for r in per), "nan": per[0].get("nan", 0),
                   "inf": per[0].get("inf", 0)}
    return ok, bufs


def run(build, groups):
    cdir = os.path.join(build, "replay-full")
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
            if g == "kernel" and m["kernel"] not in negs and m["steps"][0][1] >= 4:
                negs[m["kernel"]] = m
                lines += chain_lines(m, FILLS[0], neg_tag, half=0)
    used = [m for g in groups for m in man.get(g, [])]
    rep, out = replay(build, cdir, lines, arena_for(used))
    names = [ln.split()[1] for ln in lines if ln.startswith("chain ")]
    not_ok = [n for n in names if "%s: ok" % n not in out]
    rep["chains"], rep["chains_not_ok"] = len(names), not_ok
    res = {"groups": list(groups), "replay": rep, "cases": [], "negative_controls": {}, "mutants": {}}
    ok = rep["exit"] == 0 and rep["killed"] is None and not not_ok
    for g in groups:
        for m in man.get(g, []):
            if g == "mutant":
                _, bufs = check_entry(cdir, m, [tags[0]])
                differ = {i: b["differ"][0] for i, b in bufs.items()}
                caught = any(v > 0 for v in differ.values()) and all(v >= 0 for v in differ.values())
                res["mutants"][m["name"]] = {"caught": caught, "differ": differ}
                ok = ok and caught
                continue
            good, bufs = check_entry(cdir, m, tags)
            ok = ok and good
            res["cases"].append({"group": g, "case": m["name"], "kernels": m["launch_kernels"],
                                 "launches": len(m["steps"]), "ok": good, "buffers": bufs})
            for i, names in m.get("alts", {}).items():
                # the probe's mutants: each wrong rounding must differ from the device's output somewhere
                for j, mname in enumerate(names):
                    alt = np.load(os.path.join(cdir, "%s.alt%d.%s.npy" % (m["id"], j, i)))
                    r = compare(_read(os.path.join(cdir, "%s_%s.%s" % (m["id"], tags[0][0], i))), alt, "f16",
                                m["outs"][i], tags[0][1])
                    res["mutants"]["probe " + mname] = {"caught": r["differ"] > 0, "differ": r["differ"]}
                    ok = ok and r["differ"] > 0
    if "kernel" in groups:
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
                      "nan_values": sum(b["nan"] for c in cases for b in c["buffers"].values()),
                      "inf_values": sum(b["inf"] for c in cases for b in c["buffers"].values()),
                      "negative_controls_caught": sum(1 for v in res["negative_controls"].values() if v and v["caught"]),
                      "mutants_caught": sum(1 for v in res["mutants"].values() if v["caught"]),
                      "fills": [hex(f) for f in FILLS], "pass": bool(ok)}
    return res


# ---- timing (information only) ------------------------------------------------------------------------------------

def _time_of(out, name):
    for line in out.splitlines():
        if line.startswith("chain %s: time" % name):
            return float(line.split("time ")[1].split(" ms")[0])
    return None


def timing(build, records, label, reps=100):
    passed = set()
    for rec_path in records:
        rec = json.load(open(rec_path, encoding="utf-8"))
        if not rec.get("summary", {}).get("pass"):
            raise SystemExit("timing refused: %s is not a passing --run record" % rec_path)
        passed |= {c["case"] for c in rec["cases"] if c["ok"]}
    cdir = os.path.join(build, "replay-full")
    man = json.load(open(os.path.join(cdir, "manifest.json"), encoding="utf-8"))
    full = man.get("full_" + label)
    if not full or any(m["name"] not in passed for m in full):
        raise SystemExit("timing refused: the full %s cases are not all in a passing --run record" % label)
    tag = "time_" + time.strftime("%H%M%S")
    floor = man["floor"][0]
    # a warm-up pass over every chain (not reported), then the floor, every stage chain, the floor again
    order = [(m, "warm_" + tag, max(5, reps // 4)) for m in full] + [(floor, tag + "_a", reps)] +         [(m, tag, reps) for m in full] + [(floor, tag + "_b", reps)]
    lines = []
    for m, t, n in order:
        lines += chain_lines(m, FILLS[0], t, saves=True, time_runs=n)
    rep, out = replay(build, cdir, lines, arena_for(man["floor"] + full), fname="time_%s.txt" % label)
    res = {"size": label, "frame": list(SIZES[label]), "k": S.net2_k(SIZES[label][1]), "reps": reps,
           "byte_check_records": [os.path.abspath(r) for r in records],
           "replay": {k: rep[k] for k in ("exit", "secs", "peak_ws_mb", "killed")}, "chains": {}}
    for m, t, n in order:
        if t.startswith("warm_"):
            continue
        name = m["name"] + {tag + "_a": " (first)", tag + "_b": " (last)"}.get(t, "")
        good, _ = check_entry(cdir, m, [(t, FILLS[0])])
        ms = _time_of(out, m["id"] + "_" + t)
        res["chains"][name] = {"launches": len(m["steps"]), "kernels": m["launch_kernels"], "byte_check": good,
                               "ms_per_run": ms if good else None}
        if not good:
            res["chains"][name]["ms_unchecked_withheld"] = True
    res["method"] = ("kernel_replay chains after a warm-up pass over all of them: %d runs after a first run whose every "
                     "output buffer was compared with the CPU reference; mean wall time per run from submission to "
                     "completion, synchronised after each run (includes the per-run submission floor, shown by the 1 x 1 "
                     "push chain timed first and last); synthetic inputs, buffers resident. Information only." % reps)
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
    ap.add_argument("--sizes", default="", help="--make: full-size groups to write (1080p,1440p)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--groups", default="kernel,chain,mutant")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--size", default="1080p", help="--timing: the full size to time")
    ap.add_argument("--record", help="--timing: passing --run records, comma separated")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.prepare:
        if not a.rocm:
            ap.error("--prepare needs --rocm")
        prepare(a.build, a.rocm)
    if a.make:
        make(a.build, [s for s in a.sizes.split(",") if s])
    res = None
    if a.run:
        groups = [g if g in ("kernel", "chain", "mutant", "floor") or g.startswith("full_") else "full_" + g
                  for g in a.groups.split(",")]
        res = run(a.build, groups)
        print(json.dumps(res["summary"]))
        print("negative controls:", json.dumps({k: v and v["caught"] for k, v in res["negative_controls"].items()}),
              "mutants:", json.dumps({k: v["caught"] for k, v in res["mutants"].items()}))
        for c in res["cases"]:
            if not c["ok"]:
                print("FAIL", c["case"], json.dumps({i: b for i, b in c["buffers"].items()
                                                     if any(b["differ"]) or not b["slack_ok"]}))
        print("replay", json.dumps({k: res["replay"][k] for k in ("exit", "secs", "peak_ws_mb", "killed",
                                                                  "chains_not_ok")}))
    elif a.timing:
        if not a.record:
            ap.error("--timing needs --record")
        res = timing(a.build, a.record.split(","), a.size, a.reps)
        print(json.dumps(res, indent=1))
    if res is not None and a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
    if a.run:
        return 0 if res["summary"]["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
