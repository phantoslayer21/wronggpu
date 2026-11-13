# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of Model L's encoder chain and decoder front, synthetic data only.

    python clean/l/runner/l_chain_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--compile]
        [--main <checkout of main>] --json out.json

--compile (CPU) builds into <dir>: l_ops.co (clean/l/kernels/l_ops.hip), l_swin.co, l_enc_values.co
(clean/kernels/encoder.hip with WG_ENC_VALUES_OUT=1) and encoder.co (the same file without it); with --main also
encoder.co of that checkout as encoder_main.co, and reports whether the two encoder.co are byte-identical (the new
switch leaves Model M's kernel unchanged). The GPU part runs in a GPU slot. Output: counts only, no timing.

Every kernel is checked in isolation: its input is the GPU's own output of the step before, and the reference is the
oracle's step on that same input (clean/l/reference/oracle.py, the block with the kernels' sigmoid GELU). Reported per
step: values that differ at all, values more than one FP8 step off, non-finite values, mean absolute error against the
reference RMS. Also the whole encoder chain (GPU features against the oracle's encode) and the decoder front.
"""
import argparse
import ctypes
import filecmp
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.l.runner.l_c32_check import metrics  # noqa: E402
from clean.reference import encoder as ME  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402
from clean.runner import enc0 as E0  # noqa: E402

FRAMES = [(40, 24), (64, 36), (97, 55)]


def compile_kernels(build, rocm, main_tree=None, clang=None):
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    clang = clang or B.find_clang(rocm)
    kdir = os.path.join(ROOT, "clean", "kernels")
    ldir = os.path.join(ROOT, "clean", "l", "kernels")
    B.run(B.device_command(clang, rocm, os.path.join(ldir, "l_ops.hip"), os.path.join(build, "l_ops.co")) +
          ["-I", kdir])
    B.run(B.device_command(clang, rocm, os.path.join(ldir, "l_swin.hip"), os.path.join(build, "l_swin.co")) +
          ["-I", kdir])
    B.run(B.device_command(clang, rocm, os.path.join(kdir, "encoder.hip"), os.path.join(build, "l_enc_values.co"),
                           defines={"WG_ENC_VALUES_OUT": 1}))
    B.run(B.device_command(clang, rocm, os.path.join(kdir, "encoder.hip"), os.path.join(build, "encoder.co")))
    res = {}
    if main_tree:
        # same output file name: the build's -cuid is derived from it, so differently named outputs always differ
        os.makedirs(os.path.join(build, "main"), exist_ok=True)
        B.run(B.device_command(clang, rocm, os.path.join(main_tree, "clean", "kernels", "encoder.hip"),
                               os.path.join(build, "main", "encoder.co")))
        res["encoder_co_identical_to_main"] = filecmp.cmp(os.path.join(build, "encoder.co"),
                                                          os.path.join(build, "main", "encoder.co"), shallow=False)
    return res


class Gpu:
    def __init__(self, build, rocm):
        from clean.runner.hip import Hip
        self.h = Hip(rocm)
        self.mods = {}
        for co in ("l_ops.co", "l_swin.co", "l_enc_values.co"):
            with open(os.path.join(build, co), "rb") as f:
                self.mods[co] = self.h.load_module(f.read())
        self.live = []

    def buf(self, data=None, nbytes=None):
        n = len(data) if data is not None else nbytes
        p = self.h.malloc(n)
        if data is not None:
            self.h.upload(p, data)
        else:
            self.h.memset(p, 0, n)
        self.live.append(p)
        return p

    def free_all(self):
        for p in self.live:
            self.h.free(p)
        self.live = []

    def launch(self, co, kernel, grid, args):
        self.h.launch(self.h.get_function(self.mods[co], kernel), grid, 256, args)
        self.h.sync()

    def get_fp8(self, p, shape):
        return fp8_decode(np.frombuffer(self.h.download(p, int(np.prod(shape))), dtype=np.uint8)).reshape(shape)


def V(p):
    return ctypes.c_void_p(p)


def run_values(g, x, enc):
    """The values kernel on frame x -> raw map [fh/2, fw/2, 80]."""
    prm = E0.frame_params(x)
    data = E0.device_inputs(x)
    ntok = (prm.fh // 2) * (prm.fw // 2)
    bufs = {k: g.buf(v) for k, v in data.items()}
    w = g.buf(LP.pack_values(enc))
    expo = g.buf(np.array([x["exposure"], x["exposure_prev"]], dtype=np.float32).tobytes())
    raw = g.buf(nbytes=ntok * 80)
    rw, rh = prm.rw, prm.rh
    mn, mf, mc = (g.buf(nbytes=rw * rh * 4) for _ in range(3))
    args = [V(bufs[k]) for k in ("color", "depth", "motion", "hist", "latent", "mhist")] + \
           [V(raw), V(mn), V(mf), V(mc), V(expo), V(w), prm]
    g.launch("l_enc_values.co", "input_encoder", -(-ntok // 128), args)
    return g.get_fp8(raw, (prm.fh // 2, prm.fw // 2, 80)), raw, ntok


def run_linear(g, kernel, x_ptr, ntok, blob, C_out, shape_hw):
    y = g.buf(nbytes=ntok * C_out)
    w = g.buf(blob)
    g.launch("l_ops.co", kernel, -(-ntok // 128), [V(x_ptr), V(y), V(w), ctypes.c_int(ntok)])
    return g.get_fp8(y, shape_hw + (C_out,)), y


def run_block(g, x, x_ptr, blob, offset):
    H, W, C = x.shape
    y = g.buf(nbytes=H * W * C)
    w = g.buf(blob)
    ox, oy = offset
    nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
    g.launch("l_swin.co", "l_swin_block_c32", -(-nwin // 8),
             [V(x_ptr), V(0), V(y), V(w), ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)])
    return g.get_fp8(y, (H, W, C)), y


def copy_channels(g, src, sstride, soff, dst, dstride, doff, n, ntok):
    chunks = n // 16
    g.launch("l_ops.co", "l_copy_channels", -(-(ntok * chunks) // 256),
             [V(src), ctypes.c_int(sstride), ctypes.c_int(soff), V(dst), ctypes.c_int(dstride), ctypes.c_int(doff),
              ctypes.c_int(n), ctypes.c_int(ntok)])


def stage(name, got, ref):
    m = metrics(got, ref)
    m["step"] = name
    return m


def run(build, rocm, seeds=(89,)):
    g = Gpu(build, rocm)
    consts = dict(T.M_BLOCK_CONSTS, gelu="sigmoid")
    t = T.TABLE
    out = []
    for seed, (rw, rh), parity in ((s, f, p) for s in seeds for f in FRAMES for p in (0, 1)):
        rng = np.random.default_rng([seed, rw, rh, parity])
        params = L.random_params(rng)
        enc = params["enc0"]
        blobs = LP.pack_model_l(params)
        x = ME.random_inputs(rng, rw=rw, rh=rh, frame=parity)
        fw, fh = x["feature_size"]
        hw = (fh // 2, fw // 2)
        steps = []
        raw, raw_p, ntok = run_values(g, x, enc)
        ref = L.encode(x, enc, parity, t, keep=True)
        steps.append(stage("values", raw, ref["maps"]["raw"]))
        h1, h1_p = run_linear(g, "l_linear_80_32", raw_p, ntok, blobs["enc0.Wa"], 32, hw)
        steps.append(stage("linear1", h1, q8(raw[..., :20] @ np.asarray(enc["Wa"], np.float64).T + enc["ba"])))
        h2, h2_p = run_linear(g, "l_linear_32_32", h1_p, ntok, blobs["enc0.Wb"], 32, hw)
        steps.append(stage("linear2", h2, q8(h1 @ np.asarray(enc["Wb"], np.float64).T + enc["bb"])))
        off = T.block_offset(t, "first", parity)
        bp = {k[4:]: v for k, v in enc.items() if k.startswith("blk.")}
        h3, h3_p = run_block(g, h2, h2_p, blobs["enc0.blk"], off)
        steps.append(stage("block", h3, q8(L.swin_block(h2, bp, off, 2, consts))))
        feat_p = g.buf(nbytes=ntok * 64)
        wproj = g.buf(blobs["enc0.project"])
        g.launch("l_ops.co", "l_project_80_32", -(-ntok // 128),
                 [V(raw_p), V(h3_p), V(feat_p), V(wproj), ctypes.c_int(ntok)])
        feat = g.get_fp8(feat_p, hw + (64,))
        z = np.concatenate([raw, h3, np.zeros(hw + (16,))], axis=-1)
        steps.append(stage("project", feat, L.project_groups(z, enc["W"], enc["b"], (32, 32, 32, 32))))
        ref_sig = L.encode(x, enc, parity, T.table(consts={"default": consts}))
        steps.append(stage("encoder_chain", feat, ref_sig["features"]))
        # decoder front: split dec1's map, block on channels 0..31, join
        dec = params["dec0"]
        Z = q8(rng.normal(0, 1.5, hw + (64,)))
        Z_p = g.buf(fp8_encode(Z).tobytes())
        y_p = g.buf(nbytes=ntok * 32)
        copy_channels(g, Z_p, 64, 0, y_p, 32, 0, 32, ntok)
        y = g.get_fp8(y_p, hw + (32,))
        steps.append(stage("split", y, Z[..., :32]))
        offd = T.block_offset(t, "second", parity)
        dbp = {k[4:]: v for k, v in dec.items() if k.startswith("blk.")}
        yb, yb_p = run_block(g, y, y_p, blobs["dec0.blk"], offd)
        front_p = g.buf(nbytes=ntok * 64)
        copy_channels(g, yb_p, 32, 0, front_p, 64, 0, 32, ntok)
        copy_channels(g, Z_p, 64, 32, front_p, 64, 32, 32, ntok)
        front = g.get_fp8(front_p, hw + (64,))
        want = L.decoder_front(Z, dec, parity, T.table(consts={"default": consts}))
        steps.append(stage("decoder_front", front, want))
        steps.append({"step": "join_exact", "identical": bool(np.array_equal(front[..., :32], yb) and
                                                              np.array_equal(front[..., 32:], Z[..., 32:]))})
        out.append({"frame": [rw, rh], "parity": parity, "steps": steps})
        g.free_all()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--clang")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--main", help="a checkout of main: compare its encoder.co with this tree's")
    ap.add_argument("--seeds", type=int, default=1, help="run the frames with this many seeds (89, 90, ...)")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.build, exist_ok=True)
    res = {}
    if a.compile or a.compile_only:
        res["compile"] = compile_kernels(a.build, a.rocm, a.main, a.clang)
        print(res["compile"], flush=True)
        if a.compile_only:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
            return 0
    res["frames"] = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
    tot = {}
    for fr in res["frames"]:
        for st in fr["steps"]:
            if "values" in st:
                t = tot.setdefault(st["step"], {"values": 0, "differ": 0, "max_steps": 0, "nonfinite": 0,
                                                "max_frac_over_1_step": 0.0, "spec09_mean_ok": True})
                t["values"] += st["values"]
                t["differ"] += st["differ"]
                t["max_steps"] = max(t["max_steps"], st["max_steps"])
                t["nonfinite"] += st["nonfinite"]
                t["max_frac_over_1_step"] = max(t["max_frac_over_1_step"], st["frac_over_1_step"])
                t["spec09_mean_ok"] &= st["mean_abs_err"] <= 0.01 * st["ref_rms"]
            elif "identical" in st:
                t = tot.setdefault(st["step"], {"cases": 0, "identical": 0})
                t["cases"] += 1
                t["identical"] += int(st["identical"])
    res["totals"] = tot
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res["totals"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
