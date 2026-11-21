# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the light path (preset C) end to end on the D3D12 device against the CPU reference.

    python clean/fg/runner/fg_lite_d3d12_check.py --build <dir> --work <dir> --plan      (CPU: chains and inputs)
    python clean/fg/runner/fg_lite_d3d12_check.py --build <dir> --work <dir> --replay    (GPU: kernel_replay only)
    python clean/fg/runner/fg_lite_d3d12_check.py --build <dir> --work <dir> --verify [--json out.json]   (CPU)
    ... --plan-timing / --replay / --report-timing [--json out.json]

The build folder holds the D3D12 ports (clean/dx12/build_dx12.py --only <KERNELS>) and kernel_replay.exe. --replay is
the only GPU step: run it inside one GPU slot. Every other step is CPU only.

What is checked (--verify), per case (a sequence of game frames and calls on synthetic inputs, clean/fg/reference/
scene.py), all on the device without a host round trip between stages (one kernel_replay chain per case):
- every call's generated frame and every game frame's generation-disabled word, against the CPU reference
  (clean/fg/reference/pipeline.py LitePipeline) byte for byte;
- at the end of the sequence every internal image (colour, reference, UI, net1-grid, scatter-grid images, the 64-bit
  scatter buffers, the pyramid levels, the menu-detection result), and every game frame's net1 input cells, byte for
  byte;
- every buffer starts filled with the sentinel 0x5A (the CPU reference starts its images with the same bytes) and is
  followed by 64 slack bytes that nobody may write;
- net1: with --net1 exact (clean/fg/runner/fg_net1_gpu.py's kernels, bit-exact) the whole chain equals the pure CPU
  reference; with the fused path (fg_mconv / fg_mpair, not bit-exact: the matrix unit's chunk sums) the CPU reference
  takes the device's net1 output of each game frame, and the device's net1 output is reported against the CPU net1
  (float16 steps);
- negative controls: each kernel's largest launch on half of its grid must change that launch's outputs; two mutant
  kernels (the hole rule without its last window; the blend without the empty-vector fallback) must change the
  sequence's outputs.
Float16 NaN payloads are counted apart (the test inputs are finite; none is expected).

--plan-timing / --report-timing (information only, after --verify passed on the same build): 1920 x 1080 and
2560 x 1440 frames, a no-history game frame, a 2x game frame (one call, t = 1/2) and a 6x game frame (five calls); the
first two game frames' outputs are checked against the CPU reference first, then kernel_replay times step ranges (wall
clock from submission to completion, synchronised after each run, after one warm-up run): each stage, the
once-per-game-frame work, each call's work, and a one-launch baseline (submission and synchronisation cost).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from clean.fg.reference import net1 as N1  # noqa: E402
from clean.fg.reference import pipeline as P  # noqa: E402
from clean.fg.reference import scene as SC  # noqa: E402
from clean.fg.reference import stages_common as S  # noqa: E402
import fg_net1_mma as MMA  # noqa: E402

SENT = 0x5A
SLACK = 64
NOWIN = 0x08000000
LITE_KERNELS = ["fg_lite_fill16", "fg_lite_copy4", "fg_lite_detect", "fg_lite_menu_output", "fg_lite_downsample2",
                "fg_lite_downsample4", "fg_lite_mvec", "fg_lite_p2c_scatter", "fg_lite_p2c_unpack",
                "fg_lite_p2c_inpaint", "fg_lite_p2c_inpaint_mut", "fg_lite_p2c_pull", "fg_lite_p2c_push",
                "fg_lite_p2c_fine", "fg_lite_hole_mask", "fg_lite_quality_init", "fg_lite_net1_input",
                "fg_lite_net1_output", "fg_lite_interm", "fg_lite_weights", "fg_lite_blend", "fg_lite_blend_mut"]
NET1_KERNELS = MMA.MCONV + list(MMA.MPAIR) + MMA.EXACT + ["fg_add_f16"]
KERNELS = LITE_KERNELS + NET1_KERNELS + ["native_fill"]
MUTANTS = {"fg_lite_p2c_inpaint": "fg_lite_p2c_inpaint_mut", "fg_lite_blend": "fg_lite_blend_mut"}
# negative controls taken at a given case and call (a launch that writes: the menu output on a latched game frame)
NEG_AT = {"fg_lite_menu_output": ("menu_latch_3x", "f2_c")}
TIMING_RUNS = 20


def fb(x):
    return int(np.array(x, dtype=np.float32).view(np.uint32))


def h16(x):
    return int(np.array(x, dtype=np.float16).view(np.uint16))


def reg_args(reg):
    return [fb(v) for v in reg.floats()] if reg is not None else [0] * 10


# ---- cases ------------------------------------------------------------------------------------------------------------

def cases():
    """(name, dict): frame size, geometry overrides, settings, the game frames [(count N, scene keyword overrides)],
    scene keywords for every frame, net1 path."""
    hud = {"hudless": True}
    return [
        ("lite2_2x_menu", dict(size=(160, 90), settings=dict(menu_detection=True), scene=hud,
                               frames=[(1, {})] * 4, net1="fused")),
        ("lite2_6x_ui1", dict(size=(160, 90), scene=dict(hudless=True, ui="ui1"), frames=[(5, {})] * 3, net1="fused")),
        ("lite1_3x_ui4_inv", dict(size=(160, 90), settings=dict(lite_l1=True),
                                  scene=dict(hudless=True, ui="ui4", inverted=True), frames=[(2, {})] * 3,
                                  net1="fused")),
        ("simple_6x", dict(size=(160, 90), scene=dict(hudless=False), frames=[(5, {})] * 3, net1="fused")),
        ("menu_latch_3x", dict(size=(160, 90), settings=dict(menu_detection=True), scene=dict(hudless=True, ui="both"),
                               frames=[(2, {}), (2, {}), (2, {"menu": True}), (2, {}), (2, {})], net1="fused")),
        ("d4_pyr10_offsets", dict(size=(600, 100), geom=dict(d=4), scene=dict(hudless=True, pad=2),
                                  out=dict(out_size=(604, 104), out_offset=(2, 2)), frames=[(1, {})] * 3,
                                  net1="fused")),
        ("flags_change", dict(size=(160, 90), scene=hud,
                              frames=[(1, {}), (1, {}), (1, {"hudless": False}), (1, {}), (1, {})], net1="fused")),
        ("exact_2x", dict(size=(128, 72), scene=hud, frames=[(1, {})] * 3, net1="exact")),
        ("exact_6x_menu", dict(size=(128, 72), settings=dict(menu_detection=True), scene=hud,
                               frames=[(5, {}), (5, {}), (5, {"ui": "both", "menu": True}), (5, {})], net1="exact")),
    ]


def case_calls(cfg):
    """The case's calls in order: (game frame, CallInputs)."""
    wo, ho = cfg["size"]
    out = []
    for f, (n, over) in enumerate(cfg["frames"]):
        kw = dict(cfg.get("scene", {}))
        kw.update(over)
        kw.update(cfg.get("out", {}))
        for i in range(1, n + 1):
            out.append((f, SC.make_call(wo, ho, f, n=n, i=i, **kw)))
    return out


# ---- the device plan ---------------------------------------------------------------------------------------------------

class Planner:
    """The light path's launches for one sequence of calls (the same schedule as pipeline.LitePipeline, from its Host)."""

    def __init__(self, name, geom, settings, net1_path, params):
        self.name, self.g, self.st, self.path = name, geom, settings, net1_path
        self.host = P.Host(settings, geom)
        self.bufs = {}            # id -> [bytes, file name or None, fill]
        self.files = {}           # file name -> bytes
        self.steps = []           # (kernel, grid, args, outs, tag)
        self.roles, self.types = {}, {}
        for r, (shape, dt) in P.image_shapes(geom).items():
            if r in ("cells", "n1out"):
                continue
            self.types[r] = (shape, dt)
            self.bufs[r] = [int(np.prod(shape)) * np.dtype(dt).itemsize, None, SENT]
            self.roles[r] = r
        for l in range(1, geom.top + 1):
            w, hgt = geom.levels[l]
            self.bufs["Ps%d" % l] = [12 * w * hgt, None, SENT]
        self.bufs["detres"] = [16, None, 0]
        self.bufs["tiny"] = [16, None, SENT]
        if net1_path == "exact":
            data, acts, steps = MMA.plan_exact(geom.H0, geom.W0, params)
            self.cs, self.ps = geom.H0 * geom.W0, 1
        else:
            data, acts, steps = MMA.plan_fused(geom.H0, geom.W0, params)
            self.cs, self.ps = 1, 48
        self.n1_data, self.n1_acts, self.n1_steps = set(data), set(acts), steps
        for i, b in data.items():
            self.add_file("n1w_" + i, b)
        for i, (_, n) in acts.items():
            if i != "out":
                self.bufs["n1a_" + i] = [n, None, SENT]
        self.frame, self.callno = -1, 0
        self.n1_frames, self.out_ids, self.dis_ids = [], [], []
        self.tags = []            # per call: (frame, i)

    def add_file(self, i, data):
        fn = "%s.%s" % (self.name, i)
        self.files[fn] = data
        self.bufs[i] = [len(data), fn, SENT]

    def step(self, kernel, threads, args, outs, tag, per_lane=1):
        grid = max(1, -(-threads // (256 * per_lane)))
        self.steps.append((kernel, grid, args, outs, tag))

    def fill16(self, bid, pattern, tag):
        n16 = self.bufs[bid][0] // 2
        p = list(pattern) + [0] * (4 - len(pattern))
        self.step("fg_lite_fill16", n16, ["@" + bid, n16, p[0], p[1], p[2], p[3], len(pattern)], [bid], tag, 8)

    def r(self, role):
        return "@" + self.roles[role]

    def call(self, c):
        g, st = self.g, self.st
        first = self.host.since_last == 0
        if first:
            self.frame += 1
            f = self.frame
            self.add_file("f%d_col" % f, c.colour.tobytes())
            self.add_file("f%d_dep" % f, c.depth.tobytes())
            self.add_file("f%d_mot" % f, c.motion.tobytes())
            for k, a in (("hud", c.hudless), ("ui4", c.ui4), ("ui1", c.ui1)):
                if a is not None:
                    self.add_file("f%d_%s" % (f, k), a.tobytes())
            self.bufs["d%d" % f] = [4, None, SENT]
            self.dis_ids.append("d%d" % f)
        f = self.frame
        osz = c.out_size or (c.colour.shape[1], c.colour.shape[0])
        ox, oy = c.out_offset
        subrect = (ox, oy) != (0, 0) or tuple(osz) != (g.wo, g.ho)
        oid = "o%d" % self.callno
        self.bufs[oid] = [osz[0] * osz[1] * 8, None, SENT]
        self.out_ids.append(oid)
        ops, info = self.host.call(c.n, c.i, hudless=c.hudless is not None, ui4=c.ui4 is not None,
                                   ui1=c.ui1 is not None, reset=c.reset, out_subrect=subrect)
        tag = "f%d_c%d" % (f, c.i)
        self.tags.append((f, c.i))
        col, ctw = "@f%d_col" % f, c.colour.shape[1]
        hud = "@f%d_hud" % f if c.hudless is not None else "null"
        alpha_img, areg, ach = c.alpha_source()
        aid = ("@f%d_ui1" % f) if c.ui1 is not None else (("@f%d_ui4" % f) if c.ui4 is not None else "null")
        akind = 1 if c.ui1 is not None else 4
        aw, ah = (alpha_img.shape[1], alpha_img.shape[0]) if alpha_img is not None else (0, 0)
        hw, hh = (c.hudless.shape[1], c.hudless.shape[0]) if c.hudless is not None else (0, 0)
        steps = S.depth_steps(c.dreg)
        T1, T2 = S.thresholds(g.ho)
        nsc = g.w1 * g.h1
        for name, p in ops:
            st_tag = "%s:%s" % (tag, name) + ("_" + p["role"] if name in ("clear", "fill64") else "")
            if name == "clear":
                rid = self.roles[p["role"]]
                shape, dt = self.types[p["role"]]
                if dt == np.uint32:
                    v = int(p["value"][0])
                    self.fill16(rid, [v & 0xFFFF, v >> 16], st_tag)
                else:
                    self.fill16(rid, [h16(v) for v in p["value"]], st_tag)
            elif name == "detect":
                if p["kind"] == "alpha":
                    a = [0, aid, aw, akind, ach, areg.bx, areg.by, areg.ew, areg.eh, fb(st.alpha_threshold),
                         fb(st.fraction), "null", 0, 0, 0] + [0] * 10
                else:
                    same = 1 if (c.hreg.bx, c.hreg.by) == (c.creg.bx, c.creg.by) else 0
                    a = [1, col, ctw, 4, 0, c.creg.bx, c.creg.by, c.creg.ew, c.creg.eh, 0, 0, hud, hw, hh, same] + \
                        reg_args(c.hreg)
                a += ["@detres", "@d%d" % f, p["base"], 1 if p["hysteresis_off"] else 0]
                self.steps.append(("fg_lite_detect", 1, a, ["detres", "d%d" % f], st_tag))
            elif name == "set_disable":
                self.fill16("d%d" % f, [p["value"] & 0xFFFF, p["value"] >> 16], st_tag)
            elif name == "downsample":
                fl = p["flags"]
                bits = (1 if fl["F1"] else 0) | (2 if fl["A"] else 0) | (4 if fl["Hd"] else 0) | \
                    (8 if fl["L1"] else 0) | (16 if fl["L2"] else 0)
                a = [col, ctw, c.creg.bx, c.creg.by, hud, hw, hh] + reg_args(c.hreg) + \
                    [aid, aw, ah, akind, ach] + reg_args(areg) + \
                    [self.r("Cp"), self.r("Cc"), self.r("Uc"), self.r("Lc"), self.r("Dc"), self.r("Ld"), self.r("Um"),
                     g.wo, g.ho, g.wd, g.hd, bits]
                nb = (-(-g.wo // g.d)) * (-(-g.ho // g.d))
                self.step("fg_lite_downsample%d" % g.d, nb, a,
                          [self.roles[x] for x in ("Cc", "Uc", "Lc", "Dc", "Ld", "Um")], st_tag)
            elif name == "mvec":
                scx = np.float32(c.mvec_scale[0]) / np.float32(c.mreg.ew)
                scy = np.float32(c.mvec_scale[1]) / np.float32(c.mreg.eh)
                a = ["@f%d_dep" % f, c.depth.shape[1], c.depth.shape[0]] + reg_args(c.dreg) + \
                    ["@f%d_mot" % f, c.motion.shape[1], c.motion.shape[0]] + reg_args(c.mreg) + \
                    [self.r("Zc"), self.r("Mv"), g.w1, g.h1, int(c.depth_inverted), fb(st.s_k), fb(scx), fb(scy)]
                self.step("fg_lite_mvec", nsc, a, [self.roles["Zc"], self.roles["Mv"]], st_tag)
            elif name == "fill64":
                self.fill16(self.roles[p["role"]], [0xFFFF, 0x7FFF], st_tag)
            elif name == "p2c_scatter":
                a = [self.r("Zc"), self.r("Mv"), self.r("S"), self.r("Kc"), g.w1, g.h1] + \
                    [fb(m) for m in c.clip_to_prev] + [fb(T1), fb(T2), fb(st.s_k), int(c.depth_inverted),
                                                       fb(steps[0]), fb(steps[1])]
                self.step("fg_lite_p2c_scatter", nsc, a, [self.roles["S"], self.roles["Kc"]], st_tag)
            elif name == "p2c_unpack":
                a = [self.r(x) for x in ("S", "Zp", "Kp", "Mv", "F", "E", "Fl", "Vf")] + [g.w1, g.h1] + \
                    [fb(m) for m in c.prev_to_clip] + [fb(st.s_k), int(c.depth_inverted), fb(steps[0]), fb(steps[1]),
                                                       fb(st.kmax), fb(st.tau)]
                self.step("fg_lite_p2c_unpack", nsc, a, [self.roles[x] for x in ("F", "E", "Fl", "Vf")], st_tag)
            elif name == "p2c_inpaint":
                self.step("fg_lite_p2c_inpaint", nsc, [self.r("Fl"), self.r("G"), g.w1, g.h1], [self.roles["G"]],
                          st_tag)
            elif name == "p2c_pull":
                for l in range(1, p["top"] + 1):
                    wl, hl = g.levels[l]
                    ws, hs = g.levels[l - 1]
                    if l == 1:
                        mode, sa, sb, n = 0, self.r("G"), self.r("Vf"), 4
                    elif l == 5:
                        mode, sa, sb, n = 2, self.r("P4"), "null", 4
                    else:
                        mode, sa, sb, n = 1, "@Ps%d" % (l - 1), "null", 4 ** (l if l <= 4 else l - 4)
                    self.step("fg_lite_p2c_pull", wl * hl,
                              [mode, sa, sb, "@Ps%d" % l, self.r("P%d" % l), ws, hs, wl, hl, fb(n)],
                              ["Ps%d" % l, self.roles["P%d" % l]], st_tag)
            elif name == "p2c_push":
                for l in range(p["top"] - 1, 0, -1):
                    pw, ph = g.levels[l + 1]
                    lw, lh = g.levels[l]
                    self.step("fg_lite_p2c_push", lw * lh, [self.r("P%d" % (l + 1)), pw, ph, self.r("P%d" % l), lw, lh],
                              [self.roles["P%d" % l]], st_tag)
                pw, ph = g.levels[1]
                self.step("fg_lite_p2c_fine", nsc, [self.r("P1"), pw, ph, self.r("G"), self.r("F"), g.w1, g.h1],
                          [self.roles["F"]], st_tag)
            elif name == "hole_mask":
                self.fill16(self.roles["Hc"], [0x3C00], st_tag)
                self.step("fg_lite_hole_mask", g.wd * g.hd, [self.r("F"), g.w1, g.h1, self.r("Hc"), g.wd, g.hd],
                          [self.roles["Hc"]], st_tag)
            elif name == "quality_init":
                self.step("fg_lite_quality_init", g.wd * g.hd, [self.r("Dc"), self.r("Qc"), g.wd * g.hd],
                          [self.roles["Qc"]], st_tag)
            elif name == "net1_input":
                cid = "cells%d" % f
                self.bufs[cid] = [g.H0 * g.W0 * 96, None, SENT]
                a = [self.r(x) for x in ("Ld", "Dc", "Dp", "F", "Mv", "E", "Hc", "Hp", "Qp", "m")] + \
                    ["@" + cid, self.r("Qc"), self.r("Om"), g.wd, g.hd, g.w1, g.h1, g.W0, g.H0, g.pad_l, g.pad_t,
                     self.cs, self.ps]
                self.step("fg_lite_net1_input", 4 * g.W0 * g.H0, a, [cid, self.roles["Qc"], self.roles["Om"]], st_tag)
            elif name == "net1":
                cid, nid = "cells%d" % f, "n1o%d" % f
                self.bufs[nid] = [g.H0 * g.W0 * 32, None, SENT]
                self.n1_frames.append(f)

                def ref(a):
                    if not isinstance(a, str):
                        return a
                    k = a[1:]
                    if k == "cells":
                        return "@" + cid
                    if k == "out":
                        return "@" + nid
                    return "@n1w_" + k if k in self.n1_data else "@n1a_" + k
                for kernel, grid, args, outs, _ in self.n1_steps:
                    self.steps.append((kernel, grid, [ref(x) for x in args],
                                       [ref("@" + o)[1:] for o in outs], st_tag))
            elif name == "net1_output":
                a = ["@n1o%d" % f] + [self.r(x) for x in ("Om", "Um", "Ump", "NA", "NB", "m")] + \
                    [g.wd, g.hd, g.pad_l, g.pad_t, g.W0, g.H0, self.cs, 1 if self.ps == 1 else 16,
                     1 if p["ui_forcing"] else 0]
                self.step("fg_lite_net1_output", g.wd * g.hd, a, [self.roles[x] for x in ("NA", "NB", "m")], st_tag)
            elif name == "interm":
                for ps_, src, z, tgt in ((0, "Mv", "Zc", "MA"), (1, "F", "Zp", "MB")):
                    self.step("fg_lite_interm", nsc, [ps_, self.r(src), self.r(z), self.r(tgt), g.w1, g.h1,
                                                      fb(p["t"]), fb(st.s_k), int(c.depth_inverted), fb(T2),
                                                      fb(st.tau)], [self.roles[tgt]], st_tag)
            elif name == "weights":
                self.step("fg_lite_weights", nsc, [self.r(x) for x in ("MA", "MB", "NA", "NB", "W")] +
                          [g.w1, g.h1, g.wd, g.hd], [self.roles["W"]], st_tag)
            elif name == "prefill":
                self.step("fg_lite_copy4", osz[0] * osz[1], ["@" + oid, osz[0], 0, 0, col, ctw, 0, 0, osz[0], osz[1]],
                          [oid], st_tag)
            elif name == "lite_blend":
                a = [self.r(x) for x in ("Cp", "Cc", "MA", "MB", "W", "Up", "Uc", "Lp", "Lc")] + \
                    ["@" + oid, osz[0], ox, oy, g.wo, g.ho, g.w1, g.h1, fb(p["t"]), int(p["flag1"]),
                     int(p["flag2"])] + reg_args(S.Region(g.wo, g.ho))
                self.step("fg_lite_blend", g.wo * g.ho, a, [oid], st_tag)
            elif name == "menu_output":
                a = ["@detres", col, ctw, c.creg.bx, c.creg.by, "@" + oid, osz[0], ox, oy, g.wo, g.ho, hud, hw, hh] + \
                    reg_args(c.hreg) + [aid, aw, ah, akind, ach] + reg_args(areg)
                self.step("fg_lite_menu_output", g.wo * g.ho, a, [oid], st_tag)
            elif name == "final_copy":
                self.step("fg_lite_copy4", g.wo * g.ho, ["@" + oid, osz[0], ox, oy, col, ctw, c.creg.bx, c.creg.by,
                                                         g.wo, g.ho], [oid], st_tag)
            elif name == "swap":
                for a_, b_ in p["pairs"]:
                    self.roles[a_], self.roles[b_] = self.roles[b_], self.roles[a_]
            else:
                raise ValueError("unknown operation %r" % name)
        self.callno += 1
        return ops, info

    # chains
    def saves(self):
        """Every buffer the check reads back: outputs, disabled words, net1 cells / outputs, every role, DetRes."""
        ids = list(self.out_ids) + list(self.dis_ids) + ["detres"]
        ids += ["cells%d" % f for f in self.n1_frames] + ["n1o%d" % f for f in self.n1_frames]
        ids += sorted(set(self.roles.values()))
        return ids

    def chain_lines(self, cname, steps=None, saves=None, half=None, stop=None, sub=None, ranges=None):
        steps = self.steps if steps is None else steps
        lines = ["chain " + cname, "module " + MMA.CO_MMA, "module " + MMA.CO_NET1]
        for i, (n, fn, fill) in self.bufs.items():
            lines.append("buf %s %d %s %d" % (i, n + SLACK, os.path.basename(fn) if fn else "-", fill))
        for j, (kernel, grid, args, _, _) in enumerate(steps):
            if stop is not None and j > stop:
                break
            k = (sub or {}).get(kernel, kernel)
            gr = max(1, grid // 2) if j == half else grid
            lines.append("step %s %d 256 %s" % (k, gr, " ".join(str(a) for a in args)))
        for i in (self.saves() if saves is None else saves):
            lines.append("save %s %s.%s" % (i, cname, i))
        for runs, a, b, label in (ranges or []):
            lines.append("time %d %d %d %s" % (runs, a, b, label))
        return lines + [""]

    def arena_mb(self):
        total = sum(n + SLACK + 65536 for n, _, _ in self.bufs.values())
        return int(min(4000, max(256, 1.25 * total / 2 ** 20 + 64)))


def negative_steps(steps):
    """kernel -> the index of its launch with the largest grid (the first such), for launches of 2 or more groups."""
    best = {}
    for j, (k, grid, _, _, _) in enumerate(steps):
        if grid >= 2 and (k not in best or grid > steps[best[k]][1]):
            best[k] = j
    return best


# ---- files and replay ----------------------------------------------------------------------------------------------

def write_inputs(work, planners):
    for pl in planners:
        for fn, data in pl.files.items():
            with open(os.path.join(work, fn), "wb") as fh:
                fh.write(data)


def replay_env(arena_mb):
    e = {k: v for k, v in os.environ.items()
         if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    e["ROCM_PATH"] = r"C:\Program Files\AMD\ROCm\7.2"
    e["WG_D3D12_ARENA_MB"] = str(arena_mb)
    return e


def do_replay(build, work, timeout):
    meta = json.load(open(os.path.join(work, "plan.json")))
    t0 = time.time()
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(work, meta["chains"])],
                       env=replay_env(meta["arena_mb"]), capture_output=True, text=True, timeout=timeout,
                       creationflags=NOWIN)
    with open(os.path.join(work, "replay.log"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("exit %d seconds %.1f\n%s\n%s\n" % (r.returncode, time.time() - t0, r.stdout, r.stderr))
    return r.returncode


def read(work, fn):
    p = os.path.join(work, fn)
    return open(p, "rb").read() if os.path.exists(p) else b""


# ---- comparison ----------------------------------------------------------------------------------------------------

def compare(got, want, dtype, slack_fill=SENT):
    """got: device bytes (defined part + slack); want: the CPU reference's bytes. Float16 buffers: two NaNs with
    different payloads are counted apart (nan_payload), not as a difference."""
    n = len(want)
    res = {"bytes": n, "read": len(got) == n + SLACK}
    if not res["read"]:
        res.update(differ=-1, slack_ok=False, ok=False)
        return res
    res["slack_ok"] = bool(np.all(np.frombuffer(got[n:], np.uint8) == slack_fill))
    if dtype == np.float16 and n % 2 == 0:
        a, b = np.frombuffer(got[:n], np.uint16), np.frombuffer(want, np.uint16)
        nan_a = ((a & 0x7C00) == 0x7C00) & ((a & 0x3FF) != 0)
        nan_b = ((b & 0x7C00) == 0x7C00) & ((b & 0x3FF) != 0)
        d = a != b
        res["nan_payload"] = int(np.sum(d & nan_a & nan_b))
        res["differ"] = int(np.sum(d & ~(nan_a & nan_b)))
        idx = np.flatnonzero(d & ~(nan_a & nan_b))
    else:
        a, b = np.frombuffer(got[:n], np.uint8), np.frombuffer(want, np.uint8)
        res["differ"] = int(np.sum(a != b))
        idx = np.flatnonzero(a != b)
    if idx.size:
        res["first_differences"] = [int(i) for i in idx[:4]]
    res["ok"] = res["differ"] == 0 and res["slack_ok"]
    return res


def f16_steps(got, want):
    """Float16 step distances of device vs CPU net1 output: values over 1 / over 4 steps, the largest distance."""
    return MMA.step_report(np.frombuffer(got, np.uint16), np.frombuffer(want, np.float16).astype(np.float64))


# ---- plan / verify -------------------------------------------------------------------------------------------------

def build_case(name, cfg, params):
    g = P.Geometry(*cfg["size"], **cfg.get("geom", {}))
    st = P.Settings(**cfg.get("settings", {}))
    pl = Planner(name, g, st, cfg["net1"], params)
    calls = case_calls(cfg)
    for _, c in calls:
        pl.call(c)
    return g, st, pl, calls


def plan(build, work, only=None):
    os.makedirs(work, exist_ok=True)
    params = SC.synthetic_net1_params()
    planners, lines, meta = [], [], {"cases": [], "negatives": {}, "mutants": []}
    neg = {}                  # kernel -> (grid, planner, step): its largest launch over all cases
    for name, cfg in cases():
        if only and name not in only:
            continue
        g, st, pl, calls = build_case(name, cfg, params)
        planners.append(pl)
        lines += pl.chain_lines(name)
        for kern, j in sorted(negative_steps(pl.steps).items()):
            if kern in NEG_AT:
                cn, prefix = NEG_AT[kern]
                if name != cn:
                    continue
                j = max((i for i, s in enumerate(pl.steps) if s[0] == kern and s[4].startswith(prefix)),
                        key=lambda i: pl.steps[i][1])
            if kern not in neg or pl.steps[j][1] > neg[kern][0] or kern in NEG_AT:
                neg[kern] = (pl.steps[j][1], pl, j)
        if name == "lite2_6x_ui1" or name == "exact_6x_menu":
            for k, mk in MUTANTS.items():
                cn = "mut_%s_%s" % (name, k)
                lines += pl.chain_lines(cn, sub={k: mk}, saves=list(pl.out_ids) + sorted(set(pl.roles.values())))
                meta["mutants"].append({"case": name, "kernel": k, "mutant": mk, "chain": cn})
        meta["cases"].append({"name": name, "steps": len(pl.steps), "calls": len(calls)})
    for kern, (grid, pl, j) in sorted(neg.items()):
        outs = pl.steps[j][3]
        lines += pl.chain_lines("neg_%s_%s" % (pl.name, kern), saves=outs, half=j, stop=j)
        lines += pl.chain_lines("ref_%s_%s" % (pl.name, kern), saves=outs, stop=j)
        meta["negatives"][kern] = {"case": pl.name, "step": j, "outs": outs, "grid": grid}
    write_inputs(work, planners)
    with open(os.path.join(work, "chains.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))
    meta.update(chains="chains.txt", arena_mb=max(pl.arena_mb() for pl in planners))
    with open(os.path.join(work, "plan.json"), "w", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def cpu_run(g, st, pl, calls, params, work, chain):
    """The CPU reference over the case's calls; net1 exact (pure CPU) or the device's net1 output per game frame."""
    seen_cells = []
    if pl.path == "exact":
        inner = P.exact_net1(params)
    else:
        order = list(pl.n1_frames)

        def inner(cells):
            f = order[len(seen_cells) - 1]
            raw = read(work, "%s.n1o%d" % (chain, f))[:g.H0 * g.W0 * 32]
            return np.frombuffer(raw, np.float16).reshape(g.H0, g.W0, 16).copy()

    def net1(cells):
        seen_cells.append(cells.copy())
        return inner(cells)
    cpu = P.LitePipeline(st, g, net1, fill=SENT)
    results = [cpu.evaluate(c) for _, c in calls]
    return cpu, results, seen_cells


def layout_bytes(arr_hwc, pl):
    """CPU cells / net1 output [H0, W0, C] in the device's layout."""
    a = np.asarray(arr_hwc, np.float16)
    return (np.ascontiguousarray(a.transpose(2, 0, 1)) if pl.path == "exact" else a).tobytes()


def verify_case(name, cfg, params, work, chain=None, sub_ok=False):
    """Compares one case's chain outputs with the CPU reference; returns its record entry."""
    chain = chain or name
    g, st, pl, calls = build_case(name, cfg, params)
    cpu, results, cells = cpu_run(g, st, pl, calls, params, work, name)
    bad, rows = [], {}
    nan_payload = 0
    for k, (r, oid) in enumerate(zip(results, pl.out_ids)):
        c = compare(read(work, "%s.%s" % (chain, oid)), r["out"].tobytes(), np.float16)
        nan_payload += c.get("nan_payload", 0)
        rows[oid] = c["ok"]
        if not c["ok"]:
            bad.append({"buffer": oid, **c})
    dis_cpu = []
    k = 0
    for f in range(len(cfg["frames"])):
        n = cfg["frames"][f][0]
        dis_cpu.append(results[k]["disable"])
        k += n
    dis_gpu = []
    for f, did in enumerate(pl.dis_ids):
        raw = read(work, "%s.%s" % (chain, did))
        v = int(np.frombuffer(raw[:4], np.uint32)[0]) if len(raw) >= 4 else -1
        dis_gpu.append(v)
        if v != dis_cpu[f] or not all(b == SENT for b in raw[4:]):
            bad.append({"buffer": did, "gpu": v, "cpu": dis_cpu[f]})
    c = compare(read(work, "%s.detres" % chain), cpu.detres.tobytes(), np.uint32, slack_fill=0)
    if not c["ok"]:
        bad.append({"buffer": "detres", **c})
    images = 0
    for role, rid in pl.roles.items():
        want = cpu.img[role]
        c = compare(read(work, "%s.%s" % (chain, rid)), want.tobytes(), want.dtype)
        images += 1
        nan_payload += c.get("nan_payload", 0)
        if not c["ok"]:
            bad.append({"buffer": "image " + role, **c})
    net1 = []
    for f, cl in zip(pl.n1_frames if chain == name else [], cells):
        c = compare(read(work, "%s.cells%d" % (chain, f)), layout_bytes(cl, pl), np.float16)
        if not c["ok"]:
            bad.append({"buffer": "cells%d" % f, **c})
        raw = read(work, "%s.n1o%d" % (chain, f))[:g.H0 * g.W0 * 32]
        rep = f16_steps(raw, layout_bytes(P.exact_net1(params)(cl), pl))
        if pl.path == "exact" and (rep["over_1_step"] or rep["steps_max"] or rep["nonfinite_mismatch"]):
            bad.append({"buffer": "n1o%d" % f, "vs_cpu_net1": rep})
        net1.append({"frame": f, "vs_cpu_net1": rep})
    ops = [n for r in results for n, _ in r["ops"]]
    stats = {"calls": len(results), "history_calls": sum(1 for r in results if r["info"]["history"]),
             "disabled_frames": dis_cpu, "detections": [list(x[1:]) for x in cpu.log],
             "lite_blend_calls": ops.count("lite_blend"), "net1_runs": ops.count("net1"),
             "menu_output_calls": ops.count("menu_output")}
    return {"case": name, "frame": list(cfg["size"]), "net1": pl.path, "launches": len(pl.steps),
            "outputs_checked": len(pl.out_ids), "images_checked": images, "net1_frames": len(pl.n1_frames),
            "disabled_gpu": dis_gpu, "stats": stats, "net1_report": net1, "nan_payload": nan_payload,
            "differences": bad[:8], "pass": not bad}


def verify(build, work):
    meta = json.load(open(os.path.join(work, "plan.json")))
    log = open(os.path.join(work, "replay.log"), encoding="utf-8").read()
    params = SC.synthetic_net1_params()
    specs = dict(cases())
    res = {"replay_exit": int(log.split()[1]), "cases": [], "negatives": {}, "mutants": []}
    for c in meta["cases"]:
        res["cases"].append(verify_case(c["name"], specs[c["name"]], params, work))
    for kern, m in sorted(meta["negatives"].items()):
        diff = 0
        for o in m["outs"]:
            a = read(work, "neg_%s_%s.%s" % (m["case"], kern, o))
            b = read(work, "ref_%s_%s.%s" % (m["case"], kern, o))
            diff += int(np.sum(np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8))) if len(a) == len(b) and a \
                else -10 ** 9
        res["negatives"][kern] = {"case": m["case"], "grid": m["grid"], "differ_bytes": diff, "caught": diff > 0}
    for m in meta["mutants"]:
        e = verify_case(m["case"], specs[m["case"]], params, work, chain=m["chain"])
        res["mutants"].append({"kernel": m["kernel"], "mutant": m["mutant"], "case": m["case"],
                               "caught": not e["pass"], "differences": len(e["differences"])})
    used = set()
    for c in meta["cases"]:
        g, st, pl, _ = build_case(c["name"], specs[c["name"]], params)
        used |= set(s[0] for s in pl.steps)
    missing_neg = sorted(k for k in used if k not in res["negatives"])
    ok = res["replay_exit"] == 0 and all(c["pass"] for c in res["cases"]) and \
        all(v["caught"] for v in res["negatives"].values()) and all(m["caught"] for m in res["mutants"])
    res["summary"] = {"cases": len(res["cases"]), "cases_pass": sum(c["pass"] for c in res["cases"]),
                      "kernels_used": len(used), "negatives_caught": sum(v["caught"] for v in res["negatives"].values()),
                      "negatives": len(res["negatives"]), "kernels_without_negative": missing_neg,
                      "mutants_caught": sum(m["caught"] for m in res["mutants"]), "mutants": len(res["mutants"]),
                      "pass": bool(ok)}
    return res


# ---- timing ----------------------------------------------------------------------------------------------------------

TIMING_SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440)}


def timing_spec(size):
    return dict(size=size, scene={"hudless": True}, frames=[(1, {}), (1, {}), (5, {})], net1="fused")


def plan_timing(build, work):
    os.makedirs(work, exist_ok=True)
    params = SC.synthetic_net1_params()
    planners, lines, meta = [], [], {"sizes": {}}
    for label, size in TIMING_SIZES.items():
        name = "timing_%s" % label
        g, st, pl, calls = build_case(name, timing_spec(size), params)
        pl.steps.append(("fg_lite_fill16", 1, ["@tiny", 8, 0, 0, 0, 0, 1], ["tiny"], "baseline"))
        ranges, idx = [], {}
        for j, s in enumerate(pl.steps):
            idx.setdefault(s[4], []).append(j)
        groups = {}
        for tag, js in idx.items():
            if js != list(range(js[0], js[-1] + 1)):
                raise ValueError("stage %s is not contiguous" % tag)
            groups[tag] = (js[0], js[-1])
            ranges.append((TIMING_RUNS, js[0], js[-1], tag))
        # the once-per-game-frame part of a first call: from R3 (downsample) up to the MA clear; the per-call part:
        # from the MA clear to the blend
        def first(tag):
            return groups[tag][0]

        a1, b1 = first("f1_c1:fill64_MA"), groups["f1_c1:lite_blend"][1]
        ranges.append((TIMING_RUNS, first("f1_c1:downsample"), a1 - 1, "once_per_game_frame"))
        ranges.append((TIMING_RUNS, a1, b1, "per_call_2x_t1of2"))
        ranges.append((TIMING_RUNS, first("f2_c1:downsample"), first("f2_c1:fill64_MA") - 1, "once_6x_frame"))
        for i in range(1, 6):
            ranges.append((TIMING_RUNS, first("f2_c%d:fill64_MA" % i), groups["f2_c%d:lite_blend" % i][1],
                           "per_call_6x_i%d" % i))
        j = len(pl.steps) - 1
        ranges.append((TIMING_RUNS * 5, j, j, "baseline_one_launch"))
        saves = list(pl.out_ids) + list(pl.dis_ids) + ["n1o%d" % f for f in pl.n1_frames]
        lines += pl.chain_lines(name, saves=saves, ranges=ranges)
        planners.append(pl)
        meta["sizes"][label] = {"name": name, "size": list(size), "steps": len(pl.steps), "ranges": len(ranges),
                                "geometry": {"net1_image": [g.wd, g.hd], "net1_cells": [g.W0, g.H0],
                                             "scatter": [g.w1, g.h1], "pyramid_levels": len(g.levels)}}
    write_inputs(work, planners)
    with open(os.path.join(work, "chains.txt"), "w", newline="\n") as fh:
        fh.write("\n".join(lines))
    meta.update(chains="chains.txt", arena_mb=max(pl.arena_mb() for pl in planners))
    with open(os.path.join(work, "plan.json"), "w", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def report_timing(build, work):
    meta = json.load(open(os.path.join(work, "plan.json")))
    log = open(os.path.join(work, "replay.log"), encoding="utf-8").read()
    params = SC.synthetic_net1_params()
    res = {"replay_exit": int(log.split()[1]), "runs_per_range": TIMING_RUNS, "sizes": {}}
    for label, m in meta["sizes"].items():
        name = m["name"]
        cfg = timing_spec(tuple(m["size"]))
        g, st, pl, calls = build_case(name, cfg, params)
        # check: the first two game frames (no history, 2x) and the 6x frame against the CPU reference
        cpu, results, _ = cpu_run(g, st, pl, calls, params, work, name)
        checks = []
        for r, oid in zip(results, pl.out_ids):
            c = compare(read(work, "%s.%s" % (name, oid)), r["out"].tobytes(), np.float16)
            checks.append(c["ok"])
        times = {}
        for line in log.splitlines():
            if line.startswith("chain %s: range " % name):
                lab = line.split(" range ")[1].split(" steps ")[0]
                times[lab] = float(line.split(" time ")[1].split(" ms")[0])
        base = times.get("baseline_one_launch", 0.0)
        stage = {}
        for lab, ms in times.items():
            if lab.startswith("f1_c1:") or lab.startswith("f2_c1:"):
                stage.setdefault(lab.split(":")[1], {})[lab.split(":")[0]] = round(ms, 4)
        per6 = [times["per_call_6x_i%d" % i] for i in range(1, 6) if "per_call_6x_i%d" % i in times]
        once = times.get("once_per_game_frame")
        p2 = times.get("per_call_2x_t1of2")
        entry = {"size": m["size"], "geometry": m["geometry"], "outputs_checked": len(checks),
                 "outputs_equal_cpu": all(checks), "ms": {
                     "once_per_game_frame": once, "per_generated_frame_2x": p2,
                     "per_generated_frame_6x_each": [round(x, 4) for x in per6],
                     "per_generated_frame_6x_mean": round(float(np.mean(per6)), 4) if per6 else None,
                     "game_frame_total_2x": round(once + p2, 4) if once and p2 else None,
                     "game_frame_total_6x": round(once + sum(per6), 4) if once and per6 else None,
                     "baseline_one_launch": base},
                 "stage_ms_wall": stage}
        res["sizes"][label] = entry
    res["method"] = ("kernel_replay chain ranges: one warm-up run, then %d runs, each submitted and synchronised; wall "
                     "clock per run (includes the host's launch submission and the wait; the one-launch baseline is "
                     "that cost for a single tiny launch). Synthetic inputs (scene.py) and synthetic net1 weights, "
                     "fused net1 path, buffers resident. Information only." % TIMING_RUNS)
    res["pass"] = res["replay_exit"] == 0 and all(e["outputs_equal_cpu"] for e in res["sizes"].values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--plan-timing", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report-timing", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated case names (--plan)")
    ap.add_argument("--timeout", type=int, default=85)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = None
    if a.plan:
        m = plan(a.build, a.work, a.only.split(",") if a.only else None)
        print(json.dumps({"cases": m["cases"], "negatives": len(m["negatives"]), "mutants": len(m["mutants"]),
                          "arena_mb": m["arena_mb"]}))
    if a.plan_timing:
        m = plan_timing(a.build, a.work)
        print(json.dumps(m["sizes"]))
    if a.replay:
        rc = do_replay(a.build, a.work, a.timeout)
        print("replay exit", rc)
        return rc
    if a.verify:
        res = verify(a.build, a.work)
        print(json.dumps(res["summary"]))
        for c in res["cases"]:
            print(c["case"], "PASS" if c["pass"] else "FAIL", c["stats"], json.dumps(c["differences"][:3]))
        for k, v in res["negatives"].items():
            if not v["caught"]:
                print("negative not caught:", k, v)
        for m in res["mutants"]:
            print("mutant", m["mutant"], "caught" if m["caught"] else "NOT CAUGHT")
    if a.report_timing:
        res = report_timing(a.build, a.work)
        print(json.dumps(res, indent=1))
    if res is not None and a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, indent=1) + "\n")
    if res is not None:
        return 0 if (res.get("summary", res).get("pass")) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
