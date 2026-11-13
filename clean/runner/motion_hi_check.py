# SPDX-License-Identifier: MIT
"""Motion vectors at output resolution (spec 03 draft 4, spec 06 section 3): GPU checks of the kernels of
kernels/post_mo.hip and kernels/encoder_mo.hip against clean/reference, and timing against the default kernels.
Synthetic data and the spec team's vectors only. Run every command under the GPU lock.

    python clean/runner/motion_hi_check.py probe --build <build> --rocm <ROCm>
    python clean/runner/motion_hi_check.py post  --build <build> --rocm <ROCm> [--frames 5]
    python clean/runner/motion_hi_check.py enc   --build <build> --rocm <ROCm> [--model <raw model folder>]
    python clean/runner/motion_hi_check.py time  --build <build> --rocm <ROCm> [--render WxH --output WxH] [--rounds 5]
    python clean/runner/motion_hi_check.py frame-time --build <build> --rocm <ROCm> [--model ...] [--rounds 5]

probe: the test kernel post_mo_probe (the texel index pair and the previous position q exactly as the reconstruction
  kernels compute them) for every D pixel of several sizes and both texel formats, against reference/motion_hi.py:
  bit for bit (a NaN equals a NaN).
post:  the kernels launched alone (as clean/runner/post_sizes.py launches the default path) on frame sequences with
  the history fed back: post_recon2_r_p12_mo (+ the unbanded filter) and the true-fusion kernels post_tf_f32_p12_mo /
  post_tf_p12_mo must agree bit for bit with each other (history, reduced history; output of the f32 form against
  reconstruction + filter), and are compared with reference/post.py (FP16 steps of the history, codes of the reduced
  history, relative error of the output). Then the two reconstruction vectors of data/motion-hi-vectors.json
  through post_recon2_r_p12_mo against the expected arrays (pixels where either texel is acceptable counted apart).
enc:   through the runtime (wg_rt_upload_motion_hi, runtime option motion_outres=original; a synthetic model): the
  three motion surfaces against the reference bit for bit, on random frames and the encoder vectors the runtime takes;
  and the features of input_encoder_mo against input_encoder fed with the same motions (the motion of each render
  pixel's texel as render-size motion, a flat depth so that near = far = centre): bit for bit on a non-reset frame.
time:  the reconstruction + filter launch alone, cold cache, rounds interleaved: the default true-fusion kernel
  (three motion surfaces) against post_tf_*_mo, and the unfused pair post_recon2_r_p12_mo + filter.
frame-time: whole frames through two runtimes (default kernels with render-size motion; motion_outres=original),
  rounds interleaved.
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from clean.reference import encoder as En  # noqa: E402
from clean.reference import motion_hi as MH  # noqa: E402
from clean.reference import post as Po  # noqa: E402
from clean.runner import post as PS  # noqa: E402
from clean.runner import post_sizes as Z  # noqa: E402

MO_CO = "post_mo_w64.co"
VECTORS = os.path.join(ROOT, "data", "motion-hi-vectors.json")


class MoParams(ctypes.Structure):
    """Must match struct WgMoParams in kernels/post_motion_outres.h."""
    _fields_ = ([(n, ctypes.c_int) for n in ("mw", "mh", "pitch", "wide")]
                + [(n, ctypes.c_float) for n in ("sx", "sy", "bx", "by", "idw", "idh")])


assert ctypes.sizeof(MoParams) == 40


def mo_params(mw, mh, dw, dh, wide, mv_scale, b_mv):
    s = MH.scale(mv_scale, (mw, mh), (dw, dh))
    return MoParams(mw=mw, mh=mh, pitch=mw * (8 if wide else 4), wide=int(bool(wide)), sx=float(s[0]), sy=float(s[1]),
                    bx=float(np.float32(b_mv[0])), by=float(np.float32(b_mv[1])),
                    idw=float(np.float32(1.0) / np.float32(dw)), idh=float(np.float32(1.0) / np.float32(dh)))


def motion_field(seed, frame, mw, mh, special=True, fp16=False):
    """Raw motion of M in pixels of M, [mh][mw][2] float32: a smooth field, per-texel detail (hair-like: neighbouring
    texels differ), and with special a block far out of range, a few NaN texels and one infinite one."""
    rng = np.random.default_rng([int(seed), int(frame), mw, mh, 4242])
    yy, xx = np.meshgrid(np.arange(mh, dtype=np.float32), np.arange(mw, dtype=np.float32), indexing="ij")
    ph = float(rng.uniform(0, 6.28))
    m = np.stack([0.9 + 2.0 * np.sin(xx * (6.2832 / 345.0) + ph) * np.cos(yy * (6.2832 / 259.0)),
                  -0.4 + 1.3 * np.cos(xx * (6.2832 / 467.0) - ph) * np.sin(yy * (6.2832 / 175.0))], -1)
    m += rng.normal(0.0, 0.6, (mh, mw, 2)) * (rng.random((mh, mw, 1)) < 0.3)
    m = m.astype(np.float32)
    if special:
        by, bx = mh // 3, mw // 2
        m[by:by + max(2, mh // 20), bx:bx + max(2, mw // 20)] += np.float32(3.0 * mw)
        idx = rng.integers(0, mw * mh, max(3, mw * mh // 5000))
        m.reshape(-1, 2)[idx] = np.nan
        m.reshape(-1, 2)[int(rng.integers(0, mw * mh))] = (np.inf, -np.inf)
    if fp16:
        with np.errstate(over="ignore"):
            m = m.astype(np.float16).astype(np.float32)
    return m


def texel_bytes(m, wide):
    """The motion buffer of kernels/wg_motion_hi.h: dense rows of two f32 (wide) or two FP16, plus 8 spare bytes."""
    with np.errstate(over="ignore"):
        a = np.ascontiguousarray(m, dtype=np.float32 if wide else np.float16)
    return a.tobytes() + b"\0" * 8


def bits_equal(a, b):
    """Bitwise equality of two float32 arrays, a NaN equal to any NaN."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))


def order16(v):
    u = np.asarray(v, np.float16).view(np.uint16).astype(np.int32)
    return np.where(u & 0x8000, -(u & 0x7FFF), u)


def fp16_counts(expected, got):
    """{values, equal, one_step, more, far (more than one step and more than 0.001), max_abs, mae_rms} of FP16 arrays."""
    e16, g16 = np.asarray(expected, np.float16), np.asarray(got, np.float16)
    st = np.abs(order16(e16) - order16(g16))
    e, g = e16.astype(np.float64), g16.astype(np.float64)
    ad = np.abs(e - g)
    return {"values": int(st.size), "equal": int((st == 0).sum()), "one_step": int((st == 1).sum()),
            "more": int((st > 1).sum()), "far": int(((st > 1) & (ad > 1e-3)).sum()),
            "max_abs": float(np.nanmax(ad)) if ad.size else 0.0,
            "mae_rms": float(np.nanmean(ad) / max(np.sqrt(np.nanmean(e * e)), 1e-30)) if ad.size else 0.0}


# ---- probe ----------------------------------------------------------------------------------------------------------

PROBE_SIZES = [((3840, 2160), (2560, 1440)), ((5760, 3240), (3840, 2160)), ((2880, 1620), (1920, 1080)),
               ((2870, 1616), (1913, 1077)), ((72, 42), (48, 28)), ((71, 41), (47, 27)), ((152, 86), (101, 57)),
               ((1500, 900), (1000, 600))]


def cmd_probe(args):
    from clean.runner.hip import Hip
    dev = Hip(args.rocm)
    kern = Z.Kernels(dev, args.build, MO_CO)
    res, bad = [], 0
    for (dw, dh), (mw, mh) in PROBE_SIZES:
        for wide in (1, 0):
            for scale, b in (((1.0, 1.0), (0.0, 0.0)), ((mw / 2.0, -mh / 2.0), (0.25, -0.125))):
                m = motion_field(args.seed, wide, mw, mh, fp16=not wide)
                if scale[0] != 1.0:   # screen-unit motion: values of the order of 2 / M
                    m = m * np.array([2.0 / mw, -2.0 / mh], np.float32)
                    if not wide:
                        m = m.astype(np.float16).astype(np.float32)
                raw = texel_bytes(m, wide)
                buf, out = dev.malloc(len(raw)), dev.malloc(dw * dh * 16)
                try:
                    dev.upload(buf, raw)
                    mo = mo_params(mw, mh, dw, dh, wide, scale, b)
                    dev.launch(kern("post_mo_probe"), PS._cdiv(dw * dh, 256), 256,
                               [ctypes.c_void_p(buf), ctypes.c_void_p(out), ctypes.c_int(dw), ctypes.c_int(dh), mo])
                    dev.sync()
                    got = np.frombuffer(dev.download(out, dw * dh * 16), np.uint32).reshape(dh, dw, 4)
                finally:
                    dev.free(buf)
                    dev.free(out)
                tx, ty = MH.recon_texels(dw, mw), MH.recon_texels(dh, mh)
                qx, qy = MH.recon_positions(m, (mw, mh), (dw, dh), scale, b)
                n_t = int((got[..., 0] != tx[None, :]).sum() + (got[..., 1] != ty[:, None]).sum())
                n_q = int((~bits_equal(got[..., 2].view(np.float32), qx)).sum() +
                          (~bits_equal(got[..., 3].view(np.float32), qy)).sum())
                bdx, bdy = MH.recon_boundary(dw, mw), MH.recon_boundary(dh, mh)
                res.append({"D": [dw, dh], "M": [mw, mh], "format": "f32" if wide else "fp16", "mv_scale": list(scale),
                            "b_mv": list(b), "pixels": dw * dh, "texel_mismatches": n_t, "q_mismatches": n_q,
                            "boundary_columns": int(bdx.sum()), "boundary_rows": int(bdy.sum()),
                            "nan_positions": int((np.isnan(qx) | np.isnan(qy)).sum())})
                bad += n_t + n_q
                print(json.dumps(res[-1]), flush=True)
    print(f"probe: {len(res)} runs, {sum(r['pixels'] for r in res)} pixels, {bad} mismatching values")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if bad else 0


# ---- post kernels alone ---------------------------------------------------------------------------------------------

class MoFrame(Z.Frame):
    """post_sizes.Frame plus the motion buffer, MoParams and the five-float exposure record of the _diet kernels."""

    def __init__(self, dev, rw, rh, ow, oh, mw=None, mh=None):
        super().__init__(dev, rw, rh, ow, oh)
        self.mw, self.mh = mw or ow, mh or oh
        for k, n in (("motion_hi", self.mw * self.mh * 8 + 8), ("expo5", 32)):
            self.sizes[k] = n
            self.bufs[k] = dev.malloc(n)
        self.mo = None

    def upload_motion(self, m, wide, mv_scale=(1.0, 1.0), b_mv=(0.0, 0.0)):
        self.dev.upload(self.bufs["motion_hi"], texel_bytes(m, wide))
        self.mo = mo_params(self.mw, self.mh, self.dw, self.dh, wide, mv_scale, b_mv)

    def upload(self, data, scal, reset):
        super().upload(data, scal, reset)
        e, ep = np.float32(scal["e"]), np.float32(scal["e_prev"])
        # exposure_pair_diet's record: e, e_prev, last e, ratio, dec_scale (kernels/exposure.hip)
        rec = np.array([e, ep, e, np.float32(float(e) / float(ep)), np.float32(0.01 / float(e)), 0, 0, 0], np.float32)
        self.dev.upload(self.bufs["expo5"], rec.tobytes())

    def args(self, spec, out_kind):
        res = []
        for a in spec:
            if a == "mo":
                res.append(self.mo)
            elif a == "expo":
                res.append(ctypes.c_void_p(self.bufs["expo5"]))
            else:
                res.extend(super().args([a], out_kind))
        return res


def mo_launches(fr, kind, pi=1.0):
    """Launch lists (kernel, grid, threads, args) of the motion-at-output-resolution path, as runtime.cpp issues them.
    kind: "recon" (post_recon2_r_p12_mo), "recon+filter" (then post_filter_c1_s4 -> 3 x f32), "recon+o16" (then
    post_filter_c1_s4_o16 -> RGBA16F rows of 8 ow bytes), "tf_f32" / "tf_h" (post_tf_f32_p12_mo / post_tf_p12_mo)."""
    rw, rh, ow, oh, dw, dh = fr.rw, fr.rh, fr.ow, fr.oh, fr.dw, fr.dh
    recon = ("post_recon2_r_p12_mo", PS.recon_grid(dw, dh), 256,
             ["color", "A8", "B", "C4", "hist_prev", "motion_hi", "w", "hist_out", "reduced", "display", "params",
              "expo", "mo"])
    fargs = ["display", "out", ("i", dw), ("i", dh), ("i", ow), ("i", oh), "dec_scale", "expo"]
    if kind == "recon":
        return [recon], "f32"
    if kind == "recon+filter":
        return [recon, ("post_filter_c1_s4", PS.filter_grid(ow, oh, "post_filter_c1_s4"), 64, fargs)], "f32"
    if kind == "recon+o16":
        return [recon, ("post_filter_c1_s4_o16", PS._cdiv(ow, 40) * PS._cdiv(PS._cdiv(oh, 2), 8), 64, fargs)], "rgba16f"
    cp = Z.tf_cpairs(ow, oh)
    k1, c2, nch = Z.tf_chunks(ow, oh, cp)
    h = kind == "tf_h"
    pitch = Z.half_rgba_pitch(ow) if h else 12 * ow
    return [("post_tf_p12_mo" if h else "post_tf_f32_p12_mo", PS._cdiv(ow, PS.TF_OUT_X) * nch, 512,
             ["color", "A8", "B", "C4", "hist_prev", "m_near", "motion_hi", "w", "hist_out", "reduced", "out", "params",
              "expo", ("i", ow), ("i", oh), "dec_scale", ("f", pi), ("i", pitch), ("i", cp), ("i", k1), ("i", c2),
              "mo"])], ("rgba16f" if h else "f32")


class TwoKernels:
    """Kernel lookup over two code objects: the motion-at-output-resolution one first, then post_w64.co (filters)."""

    def __init__(self, dev, build):
        self.mo, self.w64 = Z.Kernels(dev, build, MO_CO), Z.Kernels(dev, build, Z.W64)

    def __call__(self, name):
        return self.mo(name) if name.endswith("_mo") else self.w64(name)


def run_launches(dev, kern, fr, launches, out_kind):
    for k in ("reduced", "display", "out_f32", "out_h"):
        dev.memset(fr.bufs[k], 0x7F, fr.sizes[k])
    dev.memset(fr.bufs[f"hist{1 - fr.cur}"], 0x7F, fr.sizes["hist0"])
    Z.Prepared(dev, kern, fr, launches, out_kind).run()
    dev.sync()
    dw, dh, ow, oh = fr.dw, fr.dh, fr.ow, fr.oh
    hist = np.frombuffer(dev.download(fr.bufs[f"hist{1 - fr.cur}"], dw * dh * 8), np.float16).reshape(dh, dw, 4)
    red = np.frombuffer(dev.download(fr.bufs["reduced"], fr.sizes["reduced"]), np.uint32)
    red = (red[:(dw // 2) * (dh // 2)] & 0x3FFFFFFF).reshape(dh // 2, dw // 2).copy()
    out = {"history": hist[..., :3].copy(), "reduced": red}
    if out_kind == "f32":
        out["out"] = np.frombuffer(dev.download(fr.bufs["out_f32"], ow * oh * 12), np.float32).reshape(oh, ow, 3).copy()
    return out


def reference_frame(data, scal, reset, hist_prev, m, fr, mv_scale=(1.0, 1.0), b_mv=(0.0, 0.0), **kw):
    """reference/post.py on the device-format inputs of post_sizes.frame_inputs."""
    rw, rh = fr.rw, fr.rh
    x = {"color": np.frombuffer(data["color"], np.float32).reshape(rh, rw, 4)[..., :3],
         "A": np.frombuffer(data["A"], np.uint8).reshape(rh, rw, 4)[..., :3] / 255.0,
         "B": np.frombuffer(data["B"], np.uint8).reshape(rh, rw, 4)[..., :3] / 255.0,
         "C": np.frombuffer(data["C"], np.float16).reshape(rh, rw, 4)[..., :3].astype(np.float32),
         "history": np.asarray(hist_prev, np.float32), "exposure": scal["e"], "exposure_prev": scal["e_prev"],
         "jitter": scal["jitter"], "reset": bool(reset), "recon_size": (fr.dw, fr.dh), "motion_output_res": True,
         "motion": m, "motion_size": (fr.mw, fr.mh), "mv_scale": mv_scale, "mv_bias": b_mv}
    x.update(kw)
    return Po.reconstruct(x, scal["s"])


POST_SIZES = [("427x240 -> 640x360", (427, 240), (640, 360)), ("320x180 -> 480x270", (320, 180), (480, 270)),
              ("odd 333x187 -> 499x281", (333, 187), (499, 281)), ("odd 64x36 -> 101x57", (64, 36), (101, 57)),
              ("ratio 2: 240x135 -> 480x270", (240, 135), (480, 270)), ("ratio 3: 160x90 -> 480x270", (160, 90), (480, 270))]


def out_error(ref, got):
    r, g = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(g - r) / np.maximum(np.abs(r), 1e-3)
    return {"values": int(rel.size), "rel_gt_1e-3": int((rel > 1e-3).sum()), "rel_gt_1e-2": int((rel > 1e-2).sum()),
            "non_finite": int((~np.isfinite(g)).sum()), "max_rel": float(np.nanmax(rel))}


def cmd_post(args):
    from clean.runner.hip import Hip
    dev = Hip(args.rocm)
    kern = TwoKernels(dev, args.build)
    res, fail = {"sizes": [], "vectors": []}, 0
    for label, (rw, rh), (ow, oh) in POST_SIZES:
        fr = MoFrame(dev, rw, rh, ow, oh)
        tf = PS.tf_supported(rw, rh, fr.dw, fr.dh)
        tot = {"label": label, "D": [fr.dw, fr.dh], "tf": bool(tf), "frames": args.frames, "kernel_pairs_differ": 0,
               "history": None, "reduced": None, "out": None, "boundary_far": 0, "boundary_values": 0}
        acc_h, acc_o = [], []
        red_d = np.zeros(3, np.int64)
        try:
            dev.upload(fr.bufs["hist0"], Z.initial_history(args.seed, fr.dw, fr.dh))
            fr.cur = 0
            for t in range(args.frames):
                reset = Z.RESETS[t % len(Z.RESETS)]
                data, scal = Z.frame_inputs(args.seed, t, rw, rh, fr.dw, fr.dh, "smooth")
                scal["s"] = float(scal["s"]) * 0.05   # a small blend scale: the history term stays visible
                wide = t % 2
                m = motion_field(args.seed, t, ow, oh, fp16=not wide)
                fr.upload(data, scal, reset)
                fr.upload_motion(m, wide)
                prev = np.frombuffer(dev.download(fr.bufs[f"hist{fr.cur}"], fr.dw * fr.dh * 8),
                                     np.float16).reshape(fr.dh, fr.dw, 4)[..., :3].astype(np.float32)
                a = run_launches(dev, kern, fr, *mo_launches(fr, "recon+filter"))
                if tf:   # the fused kernels must give the same bits
                    b = run_launches(dev, kern, fr, *mo_launches(fr, "tf_f32"))
                    c = run_launches(dev, kern, fr, *mo_launches(fr, "tf_h"))
                    for k, other in (("history", b), ("reduced", b), ("out", b), ("history", c), ("reduced", c)):
                        same = (bits_equal(a[k], other[k]) if a[k].dtype == np.float32
                                else a[k].view(np.uint8) == other[k].view(np.uint8))
                        tot["kernel_pairs_differ"] += int((~same).sum())
                ref = reference_frame(data, scal, reset, prev, m, fr)
                acc_h.append(fp16_counts(ref["history"], a["history"]))
                bd = MH.recon_boundary(fr.dh, oh)[:, None] | MH.recon_boundary(fr.dw, ow)[None, :]
                st = np.abs(order16(ref["history"].astype(np.float16)) - order16(a["history"]))
                far = (st > 1) & (np.abs(ref["history"].astype(np.float64) - a["history"].astype(np.float64)) > 1e-3)
                tot["boundary_far"] += int(far[bd].sum())
                tot["boundary_values"] += int(3 * bd.sum())
                rc = np.rint(ref["reduced"].astype(np.float64) * 1023.0).astype(np.int64)
                kc = np.stack([(a["reduced"] >> (10 * i)) & 1023 for i in range(3)], -1).astype(np.int64)
                d = np.abs(rc - kc)
                red_d += np.array([(d == 0).sum(), (d == 1).sum(), (d > 1).sum()])
                acc_o.append(out_error(Po.output_filter(ref["display"], scal["e"], (ow, oh)), a["out"]))
                fr.cur = 1 - fr.cur
        finally:
            fr.release()
        tot["history"] = {k: (max if k in ("max_abs", "mae_rms") else sum)(h[k] for h in acc_h) for k in acc_h[0]}
        tot["reduced"] = {"equal": int(red_d[0]), "one_code": int(red_d[1]), "more": int(red_d[2])}
        tot["out"] = {k: (max if k == "max_rel" else sum)(o[k] for o in acc_o) for k in acc_o[0]}
        fail += tot["kernel_pairs_differ"]
        res["sizes"].append(tot)
        print(json.dumps(tot), flush=True)

    # the spec team's reconstruction vectors through post_recon2_r_p12_mo
    from clean.tests.test_motion_hi_reference import load_vectors, recon_inputs
    for c in load_vectors():
        if not c["case"].startswith("recon/"):
            continue
        A = c["arrays"]
        (rw, rh), (mw, mh), (dw, dh), (ox, oy) = c["R"], c["M"], c["D"], c["motion_origin"]
        fr = MoFrame(dev, rw, rh, mw, mh)
        assert (fr.dw, fr.dh) == (dw, dh)
        try:
            n = rw * rh
            col = np.zeros((n, 4), np.float32)
            col[:, :3] = A["color"].reshape(n, 3)
            a8 = np.zeros((n, 4), np.uint8)
            a8[:, :3] = A["A"].reshape(n, 3)
            b8 = np.zeros((n, 4), np.uint8)
            b8[:, :3] = A["B"].reshape(n, 3)
            c16 = np.zeros((n, 4), np.float16)
            c16[:, :3] = A["C"].reshape(n, 3)
            data = {"color": col.tobytes(), "A": a8.tobytes(), "B": b8.tobytes(), "C": c16.tobytes(),
                    "m_near": bytes(n * 4), "m_center": bytes(n * 4), "m_far": bytes(n * 4)}
            p12 = PS.pack12_inputs({"A": data["A"], "B": data["B"], "C": data["C"]})
            data["A8"], data["C4"] = p12["A"], p12["C"]
            scal = {"jitter": tuple(c["jitter"]), "e": c["e"], "e_prev": c["e_prev"], "s": c["s"]}
            h = np.zeros((dh, dw, 4), np.float16)
            h[..., :3] = A["history_prev"]
            dev.upload(fr.bufs["hist0"], h.tobytes())
            fr.cur = 0
            fr.upload(data, scal, c["reset"])
            tex = A["motion_texture"][oy:oy + mh, ox:ox + mw]
            fr.upload_motion(tex, 1, c["mv_scale"], c["b_mv"])
            got = run_launches(dev, kern, fr, *mo_launches(fr, "recon"))
            disp = PS.decode_planes(dev.download(fr.bufs["display"], fr.sizes["display"]), (dh, dw))
        finally:
            fr.release()
        bd = A["boundary"].astype(bool)
        b3 = np.broadcast_to(bd[..., None], (dh, dw, 3))
        ref = Po.reconstruct(recon_inputs(c), c["s"])
        k = np.float32(1.0 / 2.45)
        codes = np.stack([np.floor(np.clip(disp[..., i].astype(np.float64) * k * q + 0.5, 0, q))
                          for i, q in enumerate((2047, 2047, 1023))], -1).astype(np.int64)
        dd = np.abs(codes - A["display"].astype(np.int64))
        kc = np.stack([(got["reduced"] >> (10 * i)) & 1023 for i in range(3)], -1).astype(np.int64)
        ec = A["reduced_history"].astype(np.int64)[:kc.shape[0], :kc.shape[1]]
        rd = np.abs(kc - ec)
        row = {"case": c["case"], "kernel": "post_recon2_r_p12_mo",
               "history_vs_expected": fp16_counts(A["full_history"], got["history"]),
               "history_vs_expected_boundary": fp16_counts(A["full_history"][b3], got["history"][b3]),
               "history_vs_expected_other": fp16_counts(A["full_history"][~b3], got["history"][~b3]),
               "history_vs_reference": fp16_counts(ref["history"], got["history"]),
               "boundary_pixels": int(bd.sum()),
               "reduced_vs_expected": {"compared": int(rd.size), "expected_shape": list(A["reduced_history"].shape[:2]),
                                       "equal": int((rd == 0).sum()), "one_code": int((rd == 1).sum()),
                                       "more": int((rd > 1).sum())},
               "display_codes_vs_expected": {"equal": int((dd == 0).sum()), "one_code": int((dd == 1).sum()),
                                             "more": int((dd > 1).sum()), "max": int(dd.max())}}
        res["vectors"].append(row)
        print(json.dumps(row), flush=True)
    print(f"post: {fail} values differ between the reconstruction kernels (must be 0)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if fail else 0


# ---- input encoder through the runtime ------------------------------------------------------------------------------

def synthetic_model(path):
    from clean.runner.synthetic_model import synthetic_tensors, write_raw
    if not os.path.isdir(os.path.join(path, "raw")):
        os.makedirs(path, exist_ok=True)
        write_raw(path, synthetic_tensors(1))
    return path


def make_pipeline(args, max_render, output, options=None):
    from clean.runner.pipeline import Pipeline
    return Pipeline(args.build, args.rocm, synthetic_model(args.model), max_render, output,
                    {"depth_inverted": False, "auto_exposure": False}, backend="hip", check_packing=False,
                    model_format="raw", kernels={"options": options} if options else None)


def cmd_enc(args):
    res, fail = [], 0
    cases = [("random 320x180 -> 480x270", (320, 180), (480, 270), (1.0, 1.0), (0.0, 0.0), (0, 0), None),
             ("random 333x187 -> 499x281, screen units, b_mv", (333, 187), (499, 281), (249.5, -140.5),
              (0.25, -0.125), (0, 0), None),
             ("random ratio 2: 240x135 -> 480x270", (240, 135), (480, 270), (1.0, -1.0), (0.0, 0.0), (0, 0), None)]
    from clean.tests.test_motion_hi_reference import load_vectors
    for c in load_vectors():
        if c["case"].startswith("enc/"):
            cases.append((c["case"], tuple(c["R"]), tuple(c["M"]), tuple(c["mv_scale"]), tuple(c["b_mv"]),
                          tuple(c["motion_origin"]), c))
    for label, (rw, rh), (ow, oh), scale, b, (ox, oy), vec in cases:
        fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
        if fw > 2 * rw - 1 or fh > 2 * rh - 1:
            res.append({"case": label, "skipped": "the runtime refuses this render size (padding larger than R - 1)"})
            print(json.dumps(res[-1]), flush=True)
            continue
        rng = np.random.default_rng([args.seed, rw, rh])
        a = make_pipeline(args, (rw, rh), (ow, oh), "motion_outres=original")
        bp = make_pipeline(args, (rw, rh), (ow, oh))
        dw, dh = a.dw, a.dh
        row = {"case": label, "D": [dw, dh], "frames": 0, "surface_values": 0, "surface_mismatches": 0,
               "feature_bytes": 0, "feature_mismatches": 0, "post_last": None}
        # frame 0: both runtimes on the same render-size inputs (identical state afterwards)
        x0 = En.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh, special=False)
        f0 = {"color": x0["color"], "depth": x0["depth"], "motion": x0["motion"], "mv_scale": (1.0, 1.0),
              "jitter": x0["jitter"], "exposure": 1.0, "reset": True}
        a.frame(f0)
        bp.frame(f0)
        for t in range(1, 1 + args.frames):
            xt = En.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh, special=False)
            if vec is not None:
                tex = vec["arrays"]["motion_texture"][oy:oy + oh, ox:ox + ow]
                jit = np.array(vec["jitter"], np.float32)
            else:
                tex = motion_field(args.seed, t, ow, oh, fp16=bool(t % 2))
                if scale[0] != 1.0 and abs(scale[0]) > 2:
                    tex = (tex * np.array([2.0 / ow, -2.0 / oh], np.float32)).astype(np.float32)
                jit = xt["jitter"]
            if t % 2 and vec is None:
                tex = tex.astype(np.float16)                       # the FP16 texel format
            fa = {"color": xt["color"], "depth": xt["depth"], "motion_hi": tex, "mv_scale": scale, "mv_bias": b,
                  "jitter": jit, "exposure": 1.0, "reset": False}
            da = a.frame(fa, debug=True)
            row["post_last"] = a.kernels().get("post_last")
            m = MH.encoder_motion(np.asarray(tex, np.float32), (rw, rh), (ow, oh), (dw, dh), jit, scale, b)
            with np.errstate(over="ignore"):
                want = m.astype(np.float16).astype(np.float32)
            for k in ("m_near", "m_center", "m_far"):
                row["surface_values"] += want.size
                row["surface_mismatches"] += int((~bits_equal(da["enc0"][k], want)).sum())
            if vec is not None:
                exp = vec["arrays"]["motion"].astype(np.float32)
                row["vector_mismatches"] = row.get("vector_mismatches", 0) + int((~bits_equal(da["enc0"]["m_near"],
                                                                                              exp)).sum())
            # the same motions through input_encoder: the texel of every render pixel as render-size motion, a flat
            # depth (near = far = centre), and a render-pixel scale that gives the same FP32 factor s
            s = MH.scale(scale, (ow, oh), (dw, dh))
            sc = [None, None]
            for i, (d_, r_) in enumerate(((dw, rw), (dh, rh))):
                c0 = np.float32(float(s[i]) * r_ / d_)
                for cand in (c0, np.nextafter(c0, np.float32(np.inf)), np.nextafter(c0, np.float32(-np.inf))):
                    if np.float32(float(cand) * d_ / r_) == s[i]:
                        sc[i] = float(cand)
            if None in sc:
                row["feature_note"] = "no render-pixel scale gives the same s at this size: features not compared"
            else:
                tx = MH.encoder_texels(rw, ow, jit[0])
                ty = MH.encoder_texels(rh, oh, jit[1])
                mr = np.asarray(tex, np.float32)[ty[:, None], tx[None, :]]
                fb = {"color": xt["color"], "depth": np.zeros((rh, rw), np.float32), "motion": mr, "mv_scale": sc,
                      "mv_bias": b, "jitter": jit, "exposure": 1.0, "reset": False}
                db = bp.frame(fb, debug=True)
                fa_, fb_ = a._get("enc0.features", np.uint8), bp._get("enc0.features", np.uint8)
                row["feature_bytes"] += int(fa_.size)
                row["feature_mismatches"] += int((fa_ != fb_).sum())
                for k in ("m_near", "m_center", "m_far"):
                    row["surface_mismatches"] += int((~bits_equal(db["enc0"][k], want)).sum())
                # bring b back in step with a for the next frame: both now hold different post states, so restart
                a.frame(f0)
                bp.frame(f0)
            row["frames"] += 1
        del a, bp
        fail += row["surface_mismatches"] + row["feature_mismatches"] + row.get("vector_mismatches", 0)
        res.append(row)
        print(json.dumps(row), flush=True)
    print(f"enc: {fail} mismatches (must be 0)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 1 if fail else 0


# ---- timing ---------------------------------------------------------------------------------------------------------

def cmd_time(args):
    from clean.runner.hip import Hip
    dev = Hip(args.rocm)
    kern = TwoKernels(dev, args.build)
    kern32 = Z.Kernels(dev, args.build, Z.W32)
    flush, scratch = Z.make_flush(dev, kern32)
    rw, rh = Z.parse_size(args.render)
    ow, oh = Z.parse_size(args.output)
    fr = MoFrame(dev, rw, rh, ow, oh)
    cp = Z.tf_cpairs(ow, oh)
    k1, c2, nch = Z.tf_chunks(ow, oh, cp)

    def default_tf(h):   # the runtime's default true-fusion kernel (three motion surfaces)
        pitch = Z.half_rgba_pitch(ow) if h else 12 * ow
        return [("post_tf_p12_diet_vsit" if h else "post_tf_f32_p12_diet_vsit", PS._cdiv(ow, PS.TF_OUT_X) * nch, 512,
                 ["color", "A8", "B", "C4", "hist_prev", "m_near", "m_center", "m_far", "w", "hist_out", "reduced",
                  "out", "params", "expo", ("i", ow), ("i", oh), "dec_scale", ("f", 1.0), ("i", pitch), ("i", cp),
                  ("i", k1), ("i", c2)])], ("rgba16f" if h else "f32")

    seqs = {"default copy (post_tf_p12_diet_vsit)": default_tf(True),
            "motion_outres copy (post_tf_p12_mo)": mo_launches(fr, "tf_h"),
            "default f32 (post_tf_f32_p12_diet_vsit)": default_tf(False),
            "motion_outres f32 (post_tf_f32_p12_mo)": mo_launches(fr, "tf_f32"),
            "motion_outres unfused copy (post_recon2_r_p12_mo + post_filter_c1_s4_o16)": mo_launches(fr, "recon+o16")}
    if not PS.tf_supported(rw, rh, fr.dw, fr.dh):
        seqs = {k: v for k, v in seqs.items() if "unfused" in k}
    res = {"render": [rw, rh], "output": [ow, oh], "recon": [fr.dw, fr.dh], "rounds": args.rounds,
           "repeats": args.repeats, "ms": {k: [] for k in seqs}}
    try:
        data, scal = Z.frame_inputs(args.seed, 1, rw, rh, fr.dw, fr.dh, "smooth")
        dev.upload(fr.bufs["hist0"], Z.initial_history(args.seed, fr.dw, fr.dh))
        dev.upload(fr.bufs["hist1"], Z.initial_history(args.seed + 1, fr.dw, fr.dh))
        fr.upload(data, scal, 0)
        fr.upload_motion(motion_field(args.seed, 1, ow, oh, special=False, fp16=True), 0)
        prep = {k: Z.Prepared(dev, kern, fr, l, o) for k, (l, o) in seqs.items()}
        for _ in range(args.rounds):
            for k, p in prep.items():
                res["ms"][k].append(Z.time_sequence(dev, p, args.repeats, flush))
    finally:
        fr.release()
        dev.free(scratch)
    for k, v in res["ms"].items():
        print(f"{k:76s} median {np.median(v):.4f} ms  (rounds: {', '.join(f'{x:.4f}' for x in v)})", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def cmd_frame_time(args):
    rw, rh = Z.parse_size(args.render)
    ow, oh = Z.parse_size(args.output)
    rng = np.random.default_rng(args.seed)
    a = make_pipeline(args, (rw, rh), (ow, oh))
    b = make_pipeline(args, (rw, rh), (ow, oh), "motion_outres=original")
    dw, dh = a.dw, a.dh
    fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
    x = En.random_inputs(rng, rw=rw, rh=rh, fw=fw, fh=fh, dw=dw, dh=dh, special=False)
    tex = motion_field(args.seed, 1, ow, oh, special=False, fp16=True).astype(np.float16)
    # the stand-in's input: the texel under each render pixel's centre, in render pixels (convert.cpp)
    tx = np.minimum(((np.arange(rw) + 0.5) * ow / rw).astype(np.int64), ow - 1)
    ty = np.minimum(((np.arange(rh) + 0.5) * oh / rh).astype(np.int64), oh - 1)
    mr = tex.astype(np.float32)[ty[:, None], tx[None, :]] * np.array([rw / ow, rh / oh], np.float32)
    fa = {"color": x["color"], "depth": x["depth"], "motion": mr, "mv_scale": (1.0, 1.0), "jitter": x["jitter"],
          "exposure": 1.0, "reset": False}
    fb = {"color": x["color"], "depth": x["depth"], "motion_hi": tex, "mv_scale": (1.0, 1.0), "jitter": x["jitter"],
          "exposure": 1.0, "reset": False}
    res = {"render": [rw, rh], "output": [ow, oh], "point_ms": [], "original_ms": []}
    for _ in range(args.rounds):
        res["point_ms"].append(a.time_frame(fa, args.repeats))
        res["original_ms"].append(b.time_frame(fb, args.repeats))
    res["kernels_point"] = a.kernels().get("post_last")
    res["kernels_original"] = b.kernels().get("post_last")
    for k in ("point_ms", "original_ms"):
        print(f"{k:12s} median {np.median(res[k]):.4f} ms  (rounds: {', '.join(f'{v:.4f}' for v in res[k])})")
    print("post kernels:", res["kernels_point"], "/", res["kernels_original"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "post", "enc", "time", "frame-time"):
        p = sub.add_parser(name)
        p.add_argument("--build", required=True)
        p.add_argument("--rocm", default="")
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--json")
        p.add_argument("--frames", type=int, default=5 if name == "post" else 3)
        p.add_argument("--model", default=os.path.join(tempfile.gettempdir(), "wg_motion_hi_model"))
        p.add_argument("--render", default="1705x960")
        p.add_argument("--output", default="2560x1440")
        p.add_argument("--rounds", type=int, default=5)
        p.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args(argv)
    return {"probe": cmd_probe, "post": cmd_post, "enc": cmd_enc, "time": cmd_time,
            "frame-time": cmd_frame_time}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
