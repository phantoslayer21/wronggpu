# SPDX-License-Identifier: MIT
"""Size-aware timing and output dumps of the DEFAULT post + filter path (spec 06), launched exactly as the runtime
launches it (clean/runtime/runtime.cpp), at any render / output size. Synthetic data only.

    python clean/runner/post_sizes.py time --build <build> --rocm <ROCm> [--configs all] [--rounds 3] [--json f]
    python clean/runner/post_sizes.py time ... --render 1707x960 --output 2560x1440      # one size
    python clean/runner/post_sizes.py dump --build <build> --rocm <ROCm> --render WxH --output WxH --out res.json
    python clean/runner/post_exact.py ...   # bit-for-bit comparison of two builds (uses `dump` of each checkout)

The runtime's default (runtime_default below, the contract every post change must keep in step with runtime.cpp):
the post kernels come from post_w64.co (wave64) and the reconstruction reads dec0's pack12 coefficients. Two paths,
chosen per frame by the runtime:
  - "band" (output written as 3 x f32; the runtime's own output buffer, or an interop frame without the output copy):
    post_recon2_r_band_p12 + post_filter_c1_s4_band launched alternately over 4 bands (runner/post.py band_plan);
  - "copy" (interop output copy, on by default: the output is written as RGBA16F rows of pitch half_rgba_pitch(ow),
    pre-exposure 1): post_recon2_r_p12 (one launch) + post_filter_c1_s4_o16 when the pitch is 8 ow bytes, else
    post_filter_h.
A frame the _p12 kernels do not take (recon2_supported false) runs post_recon (base A, B, C layout) + post_filter_c1_s4
(band) or the copy filter above. Every kernel gets the exposure pair in GPU memory (expo), as in the runtime.
True fusion (runtime option post_tf, default on; tf_choice below): a frame of those banded paths whose size post_tf
takes (runner/post.py tf_supported) runs one launch instead, post_tf_p12 (copy paths) / post_tf_f32_p12 (band), or the
native-AA forms post_tf_dl_p12 / post_tf_dl_f32_p12, with C = 68 output row pairs per workgroup (36 for the
workgroups dispatched last, tf_chunks); bit-identical.

`time`: 15 configurations by default: outputs 1920x1080, 2560x1440, 3840x2160, each at DLAA (R = O), Quality (O / 1.5),
Balanced (O / 1.7), Performance (O / 2), Ultra Performance (O / 3), render sizes rounded to the nearest integer
(1440p Quality: 1707x960); D = floor((3 O + 1) / 2) per axis. Rounds are interleaved (round 1 visits every
configuration, then round 2, ...), so a GPU shared with another program hits all configurations alike; the spread over
rounds is reported. Per configuration and path:
  - total_ms: the whole default launch sequence, cold cache (a 256 MB flush kernel runs before every repeat, outside
    the timed region: closer to the game, where the network stages between two frames evict everything);
  - post_ms: the reconstruction launches alone (same sequence without the filter launches), cold;
  - filter_ms = total_ms - post_ms (what the filter adds to the sequence; for the banded path this includes the
    benefit of reading the display image from the cache);
  - filter_alone_ms: the filter launches alone, cold;
  - warm_ms: the whole sequence repeated back to back (the cache keeps part of the previous repeat's data);
  - floor_ms: memory-traffic floor at that size (post_traffic.py logic: post_traffic_r_h8 + post_traffic_f_o12 / _o8
    from post.co, every byte of the two-kernel formats moved once, coalesced, cold), floor_mb the bytes it moves, and
    mb: the bytes the default formats move at least once (pack12 inputs, 40 bytes per render sample).
Motion fields: "random" (independent N(0, 2) D-pixel motion per pixel and candidate, as post_bench.py) or "smooth"
(game-like: a smooth field, the near candidate equal to the centre one on 80 % of the pixels, far = half the field).

`dump`: runs a frame sequence (default: reset, normal, normal, reset, normal; exposures change every frame, history
fed back from frame to frame as in the runtime) through the default path and writes, per frame, SHA-256 hashes of
the canonical outputs: full history (FP16 bits, [dh, dw, 3], decoded from the path's history format), reduced
history (30-bit codes, [dh / 2, dw / 2]), the output (band: f32 bits [oh, ow, 3]; copy: FP16 bits [oh, ow, 4]), plus
the hashes of the inputs. --save DIR also writes the arrays (.npy) for diagnosis.
"""
import argparse
import ctypes
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.runner import post as PS  # noqa: E402

W64 = "post_w64.co"
W32 = "post.co"
FLUSH_BYTES = 256 << 20
DEFAULT_BANDS = 4          # runtime option post_bands default
COPYPI_PRE_EXPOSURE = 0.4375   # path "copypi": a pre-exposure != 1, as engines with eye adaptation pass
SETTLE_SECONDS = 1.5           # untimed runs of every path's sequence after a configuration's setup
OUTPUTS = (("1080p", (1920, 1080)), ("1440p", (2560, 1440)), ("4K", (3840, 2160)))
MODES = (("DLAA", 1.0), ("Q", 1.5), ("B", 1.7), ("P", 2.0), ("UP", 3.0))


def render_size(o, ratio):
    """Render size of a quality mode, rounded to the nearest integer (2560 / 1.5 -> 1707)."""
    return max(1, int(o / ratio + 0.5))


def recon_size(o):
    return (3 * int(o) + 1) // 2


def configs():
    """[(label, (rw, rh), (ow, oh))] for the 15 standard configurations."""
    out = []
    for oname, (ow, oh) in OUTPUTS:
        for mname, ratio in MODES:
            out.append((f"{oname}-{mname}", (render_size(ow, ratio), render_size(oh, ratio)), (ow, oh)))
    return out


def half_rgba_pitch(ow):
    """Row pitch (bytes) of the interop output copy (interop/convert.h half_rgba_pitch)."""
    return (ow * 8 + 255) // 256 * 256


# ---- synthetic inputs (deterministic: the same seed gives the same bytes in every checkout; do not change) ---------

def frame_inputs(seed, frame, rw, rh, dw, dh, motion="random"):
    """Device-format inputs of one frame: color RGBA f32, A / B RGBA8 codes, C RGBA16F, motions 2 x FP16 (D pixels),
    the pack12 images A8 / C4 (from A, B, C as dec0 pack12 writes them), and the per-frame scalars."""
    rng = np.random.default_rng([int(seed), int(frame), rw, rh])
    n = rw * rh
    color = np.zeros((n, 4), np.float32)
    color[:, :3] = rng.uniform(0.0, 4.0, (n, 3)).astype(np.float32)
    A = np.zeros((n, 4), np.uint8)
    A[:, :3] = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    B = np.zeros((n, 4), np.uint8)
    B[:, :3] = rng.integers(0, 256, (n, 3), dtype=np.uint8)
    sx = rng.uniform(0.3, 1.5, n).astype(np.float32)
    sy = rng.uniform(0.3, 1.5, n).astype(np.float32)
    rho = rng.uniform(-0.8, 0.8, n).astype(np.float32)
    C = np.zeros((n, 4), np.float16)
    C[:, 0] = sx * sx
    C[:, 1] = sy * sy
    C[:, 2] = sx * sy * rho
    if motion == "random":
        mot = [rng.normal(0.0, 2.0, (n, 2)).astype(np.float16) for _ in range(3)]
    elif motion == "smooth":
        yy, xx = np.meshgrid(np.arange(rh, dtype=np.float32), np.arange(rw, dtype=np.float32), indexing="ij")
        ph = float(rng.uniform(0, 6.28))
        fx = (1.37 + 3.0 * np.sin(xx * (6.2832 / 517.0) + ph) * np.cos(yy * (6.2832 / 389.0))).reshape(-1)
        fy = (-0.61 + 2.0 * np.cos(xx * (6.2832 / 701.0) - ph) * np.sin(yy * (6.2832 / 263.0))).reshape(-1)
        center = np.stack([fx, fy], -1)
        near = center.copy()
        edge = rng.random(n) < 0.2
        near[edge] += rng.normal(0.0, 3.0, (int(edge.sum()), 2)).astype(np.float32)
        mot = [near.astype(np.float16), center.astype(np.float16), (0.5 * center).astype(np.float16)]
    else:
        raise ValueError(f"unknown motion {motion!r}")
    data = {"color": color.tobytes(), "A": A.tobytes(), "B": B.tobytes(), "C": C.tobytes(),
            "m_near": mot[0].tobytes(), "m_center": mot[1].tobytes(), "m_far": mot[2].tobytes()}
    p12 = PS.pack12_inputs({"A": data["A"], "B": data["B"], "C": data["C"]})
    data["A8"], data["C4"] = p12["A"], p12["C"]
    scal = {"jitter": (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5))),
            "e": float(rng.uniform(0.8, 1.6)), "e_prev": float(rng.uniform(0.8, 1.6)), "s": float(rng.uniform(0.2, 2.0))}
    return data, scal


def initial_history(seed, dw, dh):
    rng = np.random.default_rng([int(seed), 7777, dw, dh])
    h = np.zeros((dw * dh, 4), np.float16)
    h[:, :3] = rng.uniform(0.0, 1.5, (dw * dh, 3)).astype(np.float16)
    return h.tobytes()


# ---- the runtime's default path (keep in step with clean/runtime/runtime.cpp) ---------------------------------------

def tf_cpairs(ow, oh):
    """Output row pairs per post_tf workgroup (runtime.cpp kTfCpairs, runtime option post_tf_cpairs): 68, so that a
    workgroup's 3 C + 4 = 208 D rows are 13 full steps of 16 rows (C = 52 ... 84 measured alike at all sizes)."""
    return 68


def tf_chunks(ow, oh, cpairs):
    """(k1, c2, chunks) of a post_tf launch (runtime.cpp tf_chunks): k1 chunks of cpairs output row pairs, then chunks
    of c2 = min(36, cpairs) for about the last round of resident workgroups (128 slots: 4 workgroups per WGP of an
    RX 9070 XT), which shortens the tail of the launch. Experiments: WG_TF_TAIL=0 (one size), WG_TF_C2, WG_TF_SLOTS."""
    npo, strips = PS._cdiv(oh, 2), PS._cdiv(ow, PS.TF_OUT_X)
    if os.environ.get("WG_TF_TAIL", "1") == "0":
        return 1 << 30, cpairs, PS._cdiv(npo, cpairs)
    c2 = min(int(os.environ.get("WG_TF_C2", "36")), cpairs)
    m = PS._cdiv(int(os.environ.get("WG_TF_SLOTS", "128")), strips)
    k1 = max(0, (npo - m * c2) // cpairs)
    return k1, c2, k1 + PS._cdiv(npo - k1 * cpairs, c2)


def tf_choice(rw, rh, ow, oh):
    """(kernel, NW, cpairs) of the true-fusion path for this size, or None (keep in step with runtime.cpp: frames of
    the default banded pack12 path that tf_supported takes run post_tf_p12, or post_tf_dl_p12 at native AA; RGBA16F rows
    with pi folded in on the copy paths, the _f32 form on the band path). Environment overrides for experiments
    (not runtime options): WG_TF=0 (off), WG_TF_KERNEL, WG_TF_NW, WG_TF_CPAIRS, WG_TF_DL=0."""
    if os.environ.get("WG_TF", "1") == "0":
        return None
    nw = int(os.environ.get("WG_TF_NW", "8"))
    dw, dh = recon_size(ow), recon_size(oh)
    if PS.tf_supported(rw, rh, dw, dh, nw):
        k = "post_tf_p12" if nw == 8 else f"post_tf{nw}_p12"
    elif nw == 8 and os.environ.get("WG_TF_DL", "1") != "0" and PS.tf_supported(rw, rh, dw, dh, 8, PS.TF_DL_CAP):
        k = "post_tf_dl_p12"
    else:
        return None
    k = os.environ.get("WG_TF_KERNEL", k)
    return k, nw, int(os.environ.get("WG_TF_CPAIRS", str(tf_cpairs(ow, oh))))


def runtime_default(rw, rh, ow, oh, path, bands=DEFAULT_BANDS):
    """The launches the runtime issues for one frame on the given path ("band" or "copy"). Returns
    {"recon": [launch], "filter": [launch], "hist_bytes", "out": "f32" | "rgba16f", "out_pitch", "kernels"}; a launch
    is (kernel, grid, threads, [symbolic args]) with args resolved by Frame.args. Kernels come from post_w64.co."""
    dw, dh = recon_size(ow), recon_size(oh)
    sup = PS.recon2_supported(rw, rh, dw, dh)
    pre = ["color", "A8", "B", "C4"] if sup else ["color", "A", "B", "C"]
    rargs = pre + ["hist_prev", "m_near", "m_center", "m_far", "w", "hist_out", "reduced", "display", "params", "expo"]
    if os.environ.get("WG_TF_ONLYRECON") == "1" and sup:   # experiments: the unbanded reconstruction alone
        launch = ("post_recon2_r_p12", PS.recon_grid(dw, dh), 256, rargs + [("i", 0)])
        return {"recon": [launch], "filter": [], "sequence": [launch], "hist_bytes": 8, "out": "rgba16f",
                "out_pitch": half_rgba_pitch(ow), "kernels": "post_recon2_r_p12 alone"}
    tf = tf_choice(rw, rh, ow, oh)
    if tf is not None and path in ("band", "copy", "copypi"):
        # true fusion: reconstruction + output filter in one kernel (post_tf*), the display image is not written
        kname, nw, cpairs = tf
        if path == "band":
            kname, pi, pitch, out = kname.replace("_p12", "_f32_p12", 1), 1.0, 12 * ow, "f32"
        else:
            pi = 1.0 if path == "copy" else COPYPI_PRE_EXPOSURE
            pitch, out = half_rgba_pitch(ow), "rgba16f"
        k1, c2, nch = tf_chunks(ow, oh, cpairs)
        launch = (kname, PS._cdiv(ow, PS.TF_OUT_X) * nch, 64 * nw,
                  ["color", "A8", "B", "C4", "hist_prev", "m_near", "m_center", "m_far", "w", "hist_out", "reduced",
                   "out", "params", "expo", ("i", ow), ("i", oh), "dec_scale", ("f", pi), ("i", pitch),
                   ("i", cpairs), ("i", k1), ("i", c2)])
        return {"recon": [launch], "filter": [], "sequence": [launch], "hist_bytes": 8, "out": out,
                "out_pitch": pitch, "kernels": f"{kname} (C={cpairs})"}
    fargs = ["display", "out", ("i", dw), ("i", dh), ("i", ow), ("i", oh), "dec_scale", "expo"]
    recon, filt = [], []
    if path == "band":
        if sup:
            rk, fk = "post_recon2_r_band_p12", "post_filter_c1_s4_band"
            for kind, tile0, tiles in PS.band_plan(dw, dh, ow, oh, bands, 4):
                if kind == "recon":
                    recon.append((rk, tiles, 256, rargs + [("i", tile0)]))
                else:
                    filt.append((fk, tiles, 64, fargs + [("i", tile0)]))
        else:
            recon.append(("post_recon", PS.recon_grid(dw, dh), 256, rargs))
            filt.append(("post_filter_c1_s4", PS.filter_grid(ow, oh, "post_filter_c1_s4"), 64, fargs))
        out, pitch = "f32", 12 * ow
    elif path in ("copy", "copypi"):
        # copypi: the interop output copy in a game that passes a pre-exposure pi != 1 (many engines do): the runtime
        # then always takes post_filter_h (pi folded in), whatever the pitch
        pitch = half_rgba_pitch(ow)
        pi = 1.0 if path == "copy" else COPYPI_PRE_EXPOSURE
        if sup:  # banded, as the runtime's own output: the filter bands write RGBA16F rows (post_filter_c1_s4_h_band)
            rk, fk = "post_recon2_r_band_p12", "post_filter_c1_s4_h_band"
            for kind, tile0, tiles in PS.band_plan(dw, dh, ow, oh, bands, 4):
                if kind == "recon":
                    recon.append((rk, tiles, 256, rargs + [("i", tile0)]))
                else:
                    filt.append((fk, tiles, 64, fargs + [("f", pi), ("i", pitch), ("i", tile0)]))
        else:
            recon.append(("post_recon", PS.recon_grid(dw, dh), 256, rargs))
        if sup:
            pass
        elif pitch == 8 * ow and pi == 1.0:   # pi = 1 (pre-exposure 1)
            filt.append(("post_filter_c1_s4_o16", PS._cdiv(ow, 40) * PS._cdiv(PS._cdiv(oh, 2), 8), 64, fargs))
        else:
            grid = PS._cdiv(ow, 128) * PS._cdiv(PS._cdiv(oh, 2), 16)
            filt.append(("post_filter_h", grid, 64, fargs + [("f", pi), ("i", pitch)]))
        out = "rgba16f"
    else:
        raise ValueError(f"unknown path {path!r}")
    # launch order: the banded path alternates reconstruction and filter bands
    order = []
    if sup and path in ("band", "copy", "copypi"):   # all paths banded (copy paths: post_filter_c1_s4_h_band)
        ri = fi = 0
        for kind, _, _ in PS.band_plan(dw, dh, ow, oh, bands, 4):
            if kind == "recon":
                order.append(recon[ri])
                ri += 1
            else:
                order.append(filt[fi])
                fi += 1
    else:
        order = recon + filt
    names = sorted({l[0] for l in order}, key=[l[0] for l in order].index)
    return {"recon": recon, "filter": filt, "sequence": order, "hist_bytes": 8, "out": out, "out_pitch": pitch,
            "kernels": " + ".join(names) + (f" ({bands} bands)" if sup else "")}


class Kernels:
    """Kernel handles from a code object of the build directory (post_w64.co by default)."""

    def __init__(self, dev, build, code_object=W64):
        with open(os.path.join(build, code_object), "rb") as f:
            self.module = dev.load_module(f.read())
        self.dev, self.funcs = dev, {}

    def __call__(self, name):
        if name not in self.funcs:
            self.funcs[name] = self.dev.get_function(self.module, name)
        return self.funcs[name]


class Frame:
    """Device buffers of one configuration (all input layouts, both history copies, both output formats)."""

    def __init__(self, dev, rw, rh, ow, oh):
        self.dev = dev
        self.rw, self.rh, self.ow, self.oh = rw, rh, ow, oh
        self.dw, self.dh = dw, dh = recon_size(ow), recon_size(oh)
        n, nd = rw * rh, dw * dh
        sizes = {"color": n * 16, "A": n * 4, "B": n * 4, "C": n * 8, "A8": n * 8, "C4": n * 4, "m_near": n * 4,
                 "m_center": n * 4, "m_far": n * 4, "hist0": nd * 8, "hist1": nd * 8,
                 "reduced": max((dw // 2) * (dh // 2), 1) * 4, "display": PS.display_bytes(dw, dh),
                 "out_f32": ow * oh * 12, "out_h": half_rgba_pitch(ow) * oh, "w": 16, "expo": 16}
        self.sizes = sizes
        self.bufs = {}
        try:
            for k, v in sizes.items():
                self.bufs[k] = dev.malloc(v)
        except Exception:
            self.release()
            raise
        self.cur = 0
        self.params = None
        self.dec_scale = ctypes.c_float(0.01)

    def release(self):
        for p in self.bufs.values():
            self.dev.free(p)
        self.bufs = {}

    def upload(self, data, scal, reset):
        for k in ("color", "A", "B", "C", "A8", "C4", "m_near", "m_center", "m_far"):
            self.dev.upload(self.bufs[k], data[k])
        self.dev.upload(self.bufs["w"], np.array([scal["s"], 0, 0, 0], np.float32).tobytes())
        self.dev.upload(self.bufs["expo"], np.array([scal["e"], scal["e_prev"], 0, 0], np.float32).tobytes())
        e = scal["e"]
        # as runtime.cpp: host-side e_prev = e, ratio 1 (the kernels take both from expo)
        self.params = PS.PostParams(rw=self.rw, rh=self.rh, dw=self.dw, dh=self.dh, reset=int(bool(reset)), pad0=0,
                                    pad1=0, pad2=0, rx=float(np.float32(self.rw / self.dw)),
                                    ry=float(np.float32(self.rh / self.dh)), jx=scal["jitter"][0],
                                    jy=scal["jitter"][1], e=e, e_prev=e, ratio=1.0, pad3=0.0)
        self.dec_scale = ctypes.c_float(0.01 / e)

    def args(self, spec, out_kind):
        P = ctypes.c_void_p
        res = []
        for a in spec:
            if isinstance(a, tuple):
                res.append(ctypes.c_int(a[1]) if a[0] == "i" else ctypes.c_float(a[1]))
            elif a == "hist_prev":
                res.append(P(self.bufs[f"hist{self.cur}"]))
            elif a == "hist_out":
                res.append(P(self.bufs[f"hist{1 - self.cur}"]))
            elif a == "out":
                res.append(P(self.bufs["out_f32" if out_kind == "f32" else "out_h"]))
            elif a == "params":
                res.append(self.params)
            elif a == "dec_scale":
                res.append(self.dec_scale)
            else:
                res.append(P(self.bufs[a]))
        return res


class Prepared:
    """Launches with their parameter arrays built once (the timed loop only calls hipModuleLaunchKernel)."""

    def __init__(self, dev, kern, frame, launches, out_kind):
        self.dev = dev
        self.items = []
        for name, grid, threads, spec in launches:
            args = frame.args(spec, out_kind)
            params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
            self.items.append((kern(name), int(grid), int(threads), params, args))

    def run(self):
        lib = self.dev.lib
        for f, g, t, params, _ in self.items:
            err = lib.hipModuleLaunchKernel(f, g, 1, 1, t, 1, 1, 0, None, params, None)
            if err:
                self.dev.check(err, "hipModuleLaunchKernel")


def time_sequence(dev, prep, repeats, flush=None):
    """Median ms of the prepared launches. flush (Prepared) before every repeat, outside the timed region (cold);
    without it the sequence runs `repeats` times back to back between two events (warm, mean per repeat)."""
    start, stop = dev.event(), dev.event()
    try:
        prep.run()
        dev.sync()
        if flush is None:
            vals = []
            for _ in range(3):
                dev.record(start)
                for _ in range(int(repeats)):
                    prep.run()
                dev.record(stop)
                vals.append(dev.elapsed_ms(start, stop) / int(repeats))
            return float(np.median(vals))
        times = []
        for _ in range(int(repeats)):
            flush.run()
            dev.record(start)
            prep.run()
            dev.record(stop)
            times.append(dev.elapsed_ms(start, stop))
        return float(np.median(times))
    finally:
        dev.destroy_event(start)
        dev.destroy_event(stop)


def traffic_launches(fr, out_bytes):
    P = ctypes.c_void_p
    I = ctypes.c_int
    dims = [I(v) for v in (fr.rw, fr.rh, fr.dw, fr.dh, fr.ow, fr.oh)]
    grid = PS._cdiv(fr.dw * fr.dh, 256)
    b = fr.bufs
    out = b["out_f32"] if out_bytes == 12 else b["out_h"]
    r = ("post_traffic_r_h8", grid, 256, [P(b[k]) for k in ("color", "A", "B", "C", "m_near", "m_center", "m_far",
                                                           "hist0", "hist1", "reduced", "display")] + [P(out)] + dims)
    f = ("post_traffic_f_o12" if out_bytes == 12 else "post_traffic_f_o8", grid, 256,
         [P(b["display"]), P(out)] + [I(v) for v in (fr.dw, fr.dh, fr.ow, fr.oh)])
    return [r, f]


class RawPrepared(Prepared):
    def __init__(self, dev, kern, launches):
        self.dev = dev
        self.items = []
        for name, grid, threads, args in launches:
            params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
            self.items.append((kern(name), int(grid), int(threads), params, args))


def default_bytes(rw, rh, ow, oh, out_bytes):
    """Bytes the default formats move at least once per frame: pack12 inputs (colour 16, A8 8, C4 4, three motions
    12 = 40 per render sample), history 8 read + 8 written, display 6 written + 6 read, reduced 4 per 2 x 2 block,
    output (12 or 8 per pixel)."""
    dw, dh = recon_size(ow), recon_size(oh)
    n = dw * dh
    return rw * rh * 40 + n * 16 + n * 12 + (dw // 2) * (dh // 2) * 4 + ow * oh * out_bytes


def floor_bytes(rw, rh, ow, oh, out_bytes):
    """Bytes of post_traffic_r_h8 + post_traffic_f_o<out> (44 per render sample: the unpacked A, B, C)."""
    dw, dh = recon_size(ow), recon_size(oh)
    n = dw * dh
    return rw * rh * 44 + n * 16 + n * 12 + (n // 4) * 4 + ow * oh * out_bytes


# ---- time ---------------------------------------------------------------------------------------------------------

def measure_config(dev, kern, kern32, flush, rw, rh, ow, oh, paths, repeats, motion, seed, bands, floors=True):
    fr = Frame(dev, rw, rh, ow, oh)
    try:
        data, scal = frame_inputs(seed, 1, rw, rh, fr.dw, fr.dh, motion)
        fr.upload(data, scal, reset=False)
        dev.upload(fr.bufs["hist0"], initial_history(seed, fr.dw, fr.dh))
        dev.upload(fr.bufs["hist1"], initial_history(seed + 1, fr.dw, fr.dh))
        del data
        # Settle the fresh allocations first: without this, whichever path is timed first in a configuration ran
        # ~40 % slower (new buffers are slow for a while after hipMalloc on Windows; the runtime's buffers live for the
        # whole game session, so the settled numbers are the representative ones).
        warms = [Prepared(dev, kern, fr, runtime_default(rw, rh, ow, oh, path, bands)["sequence"],
                          runtime_default(rw, rh, ow, oh, path, bands)["out"]) for path in paths]
        t_end = time.time() + SETTLE_SECONDS
        while time.time() < t_end:
            for warm in warms:
                for _ in range(10):
                    warm.run()
            dev.sync()
        res = {}
        for path in paths:
            plan = runtime_default(rw, rh, ow, oh, path, bands)
            ok = plan["out"]
            seq = Prepared(dev, kern, fr, plan["sequence"], ok)
            rec = Prepared(dev, kern, fr, plan["recon"], ok)
            fil = Prepared(dev, kern, fr, plan["filter"], ok)
            r = {"kernels": plan["kernels"]}
            r["total_ms"] = time_sequence(dev, seq, repeats, flush)
            r["post_ms"] = time_sequence(dev, rec, repeats, flush)
            r["filter_alone_ms"] = time_sequence(dev, fil, repeats, flush)
            r["warm_ms"] = time_sequence(dev, seq, repeats)
            r["filter_ms"] = r["total_ms"] - r["post_ms"]
            ob = 12 if ok == "f32" else 8
            r["mb"] = default_bytes(rw, rh, ow, oh, ob) / 1e6
            if floors:
                r["floor_ms"] = time_sequence(dev, RawPrepared(dev, kern32, traffic_launches(fr, ob)), repeats, flush)
                r["floor_mb"] = floor_bytes(rw, rh, ow, oh, ob) / 1e6
            res[path] = r
        return res
    finally:
        fr.release()


def parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def make_flush(dev, kern32):
    scratch = dev.malloc(FLUSH_BYTES)
    dev.memset(scratch, 0, FLUSH_BYTES)
    nw = FLUSH_BYTES // 4
    fl = RawPrepared(dev, kern32, [("post_traffic_flush", PS._cdiv(nw, 256), 256,
                                    [ctypes.c_void_p(scratch), ctypes.c_uint32(nw)])])
    return fl, scratch


def cmd_time(args):
    from clean.runner.hip import Hip
    dev = Hip(args.rocm)
    kern = Kernels(dev, args.build, args.code_object)
    kern32 = Kernels(dev, args.build, W32)
    flush, scratch = make_flush(dev, kern32)
    if args.render or args.output:
        if not (args.render and args.output):
            raise SystemExit("--render and --output go together")
        cfgs = [("custom", parse_size(args.render), parse_size(args.output))]
    else:
        cfgs = configs()
        if args.configs != "all":
            want = args.configs.split(",")
            cfgs = [c for c in cfgs if c[0] in want]
            if len(cfgs) != len(want):
                raise SystemExit(f"unknown configuration in {want}; known: {[c[0] for c in configs()]}")
    paths = args.paths.split(",")
    runs = {c[0]: [] for c in cfgs}
    t0 = time.time()
    try:
        # warm-up (discarded): the first measurements otherwise run while the GPU is still raising its clocks
        _, (wrw, wrh), (wow, woh) = cfgs[0]
        for _ in range(2):
            measure_config(dev, kern, kern32, flush, wrw, wrh, wow, woh, paths, args.repeats, args.motion, args.seed,
                           args.bands, False)
        for rnd in range(args.rounds):
            for label, (rw, rh), (ow, oh) in cfgs:
                runs[label].append(measure_config(dev, kern, kern32, flush, rw, rh, ow, oh, paths, args.repeats,
                                                  args.motion, args.seed, args.bands, not args.no_floor))
            print(f"# round {rnd + 1}/{args.rounds} done ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    finally:
        dev.free(scratch)
    report = {"device": dev.device_name, "code_object": args.code_object, "motion": args.motion,
              "repeats": args.repeats, "rounds": args.rounds, "bands": args.bands, "rows": []}
    keys = ["post_ms", "filter_ms", "total_ms", "filter_alone_ms", "warm_ms"] + ([] if args.no_floor else ["floor_ms"])
    hdr = (f"{'config':12s} {'render':>10s} {'D':>10s} {'output':>10s} {'path':4s} {'post':>7s} {'filter':>7s} "
           f"{'total':>7s} {'f.alone':>7s} {'warm':>7s} {'floor':>7s} {'MB':>6s} {'GB/s':>5s} {'spread':>6s} "
           f"{'ns/opx':>6s}")
    print(hdr)
    for label, (rw, rh), (ow, oh) in cfgs:
        for path in paths:
            rows = [r[path] for r in runs[label]]
            med = {k: float(np.median([r[k] for r in rows])) for k in keys}
            tot = [r["total_ms"] for r in rows]
            spread = (max(tot) - min(tot)) / med["total_ms"] if med["total_ms"] > 0 else 0.0
            row = {"label": label, "render": f"{rw}x{rh}", "recon": f"{recon_size(ow)}x{recon_size(oh)}",
                   "output": f"{ow}x{oh}", "path": path, "kernels": rows[0]["kernels"], **med,
                   "mb": rows[0]["mb"], "spread": spread, "rounds_total_ms": tot,
                   "ns_per_output_px": med["total_ms"] * 1e6 / (ow * oh)}
            if not args.no_floor:
                row["floor_mb"] = rows[0]["floor_mb"]
            report["rows"].append(row)
            print(f"{label:12s} {row['render']:>10s} {row['recon']:>10s} {row['output']:>10s} {path:4s} "
                  f"{med['post_ms']:7.3f} {med['filter_ms']:7.3f} {med['total_ms']:7.3f} {med['filter_alone_ms']:7.3f} "
                  f"{med['warm_ms']:7.3f} {med.get('floor_ms', 0):7.3f} {row['mb']:6.0f} "
                  f"{row['mb'] / med['total_ms']:5.0f} {100 * spread:5.1f}% {row['ns_per_output_px']:6.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


# ---- dump ---------------------------------------------------------------------------------------------------------

RESETS = (1, 0, 0, 1, 0)   # frame sequence: reset flags (repeated cyclically for more frames)


def canonical_outputs(dev, fr, plan):
    """{name: array} of the frame's outputs in a format-independent form."""
    dw, dh, ow, oh = fr.dw, fr.dh, fr.ow, fr.oh
    raw = dev.download(fr.bufs[f"hist{1 - fr.cur}"], dw * dh * plan["hist_bytes"])
    if plan["hist_bytes"] != 8:
        raise NotImplementedError("decode of this history format: add it here")
    hist = np.frombuffer(raw, np.uint16).reshape(dh, dw, 4)[..., :3].copy()
    red = np.frombuffer(dev.download(fr.bufs["reduced"], fr.sizes["reduced"]), np.uint32)
    red = (red[:(dw // 2) * (dh // 2)] & 0x3FFFFFFF).reshape(dh // 2, dw // 2).copy()
    if plan["out"] == "f32":
        out = np.frombuffer(dev.download(fr.bufs["out_f32"], ow * oh * 12), np.uint32).reshape(oh, ow, 3).copy()
    else:
        pitch = plan["out_pitch"]
        rows = np.frombuffer(dev.download(fr.bufs["out_h"], pitch * oh), np.uint8).reshape(oh, pitch)
        out = rows[:, :8 * ow].copy().view(np.uint16).reshape(oh, ow, 4)
    return {"history": hist, "reduced": red, "out": out}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cmd_dump(args):
    from clean.runner.hip import Hip
    dev = Hip(args.rocm)
    kern = Kernels(dev, args.build, args.code_object)
    rw, rh = parse_size(args.render)
    ow, oh = parse_size(args.output)
    fr = Frame(dev, rw, rh, ow, oh)
    res = {"render": [rw, rh], "output": [ow, oh], "recon": [fr.dw, fr.dh], "motion": args.motion, "seed": args.seed,
           "paths": {}}
    try:
        for path in args.paths.split(","):
            plan = runtime_default(rw, rh, ow, oh, path, args.bands)
            dev.upload(fr.bufs["hist0"], initial_history(args.seed, fr.dw, fr.dh))
            dev.memset(fr.bufs["hist1"], 0x7F, fr.sizes["hist1"])
            fr.cur = 0
            frames = []
            for t in range(args.frames):
                reset = RESETS[t % len(RESETS)]
                data, scal = frame_inputs(args.seed, t, rw, rh, fr.dw, fr.dh, args.motion)
                in_hash = hashlib.sha256(b"".join(data[k] for k in sorted(data)) +
                                         json.dumps(scal, sort_keys=True).encode()).hexdigest()
                fr.upload(data, scal, reset)
                del data
                for k in ("reduced", "display", "out_f32", "out_h"):
                    dev.memset(fr.bufs[k], 0x7F, fr.sizes[k])
                dev.memset(fr.bufs[f"hist{1 - fr.cur}"], 0x7F, fr.sizes["hist0"])
                Prepared(dev, kern, fr, plan["sequence"], plan["out"]).run()
                dev.sync()
                outs = canonical_outputs(dev, fr, plan)
                frames.append({"reset": reset, "inputs": in_hash, **{k: sha(v) for k, v in outs.items()}})
                if args.save:
                    os.makedirs(args.save, exist_ok=True)
                    for k, v in outs.items():
                        np.save(os.path.join(args.save, f"{path}_f{t}_{k}.npy"), v)
                fr.cur = 1 - fr.cur
            res["paths"][path] = {"kernels": plan["kernels"], "frames": frames}
    finally:
        fr.release()
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["paths"], indent=1) if args.verbose else f"wrote {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("time", "dump"):
        p = sub.add_parser(name)
        p.add_argument("--build", required=True)
        p.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
        p.add_argument("--code-object", default=W64, help="code object of the post kernels (default post_w64.co)")
        p.add_argument("--render", help="render size WxH (with --output: one configuration)")
        p.add_argument("--output", help="output size WxH")
        p.add_argument("--paths", default="band,copy", help="band, copy or both (comma-separated)")
        p.add_argument("--motion", default="random", choices=("random", "smooth"))
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--bands", type=int, default=DEFAULT_BANDS, help="bands of the banded path (runtime: 4)")
    t = sub.choices["time"]
    t.add_argument("--configs", default="all", help="'all' or comma-separated labels, e.g. 1440p-Q,4K-P")
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--repeats", type=int, default=20)
    t.add_argument("--no-floor", action="store_true")
    t.add_argument("--json")
    d = sub.choices["dump"]
    d.add_argument("--frames", type=int, default=5)
    d.add_argument("--out", required=True, help="result JSON (per-frame output hashes)")
    d.add_argument("--save", help="also save the canonical outputs (.npy) into this folder")
    d.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "dump" and not (args.render and args.output):
        ap.error("dump needs --render and --output")
    (cmd_time if args.cmd == "time" else cmd_dump)(args)


if __name__ == "__main__":
    main()
