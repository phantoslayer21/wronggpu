# SPDX-License-Identifier: MIT
"""Whole-frame GPU time of model M per output filter (runtime option output_filter), interleaved (A B C A B C ...),
on a synthetic frame: Pipeline.time_frame at 1706x960 -> 2560x1440 by default.

    python the GPU lock (WG_GPU_LOCK, BUILDING.md) -- python -m clean.runner.output_filter_time --build <dir> --rocm <ROCm>
        --model <work dir for a synthetic model with made-up filter weights>
        [--configs output_filter=analytic output_filter=original ...] [--rounds 8] [--repeats 60]
        [--copy]                                       also the post stage of the interop's output-copy form
        [--other-build <dir> --other-model <dir> --other-map <map>]   a second runtime build on another model

The synthetic model (clean/runner/synthetic_model.py, seed 1, --output-taps random) has the shapes of the real one,
so every kernel does the real amount of work; its values are random, so the frame time is not the game's (the Swin
stages may pick their clamping kernels). The comparison between filters on the same model is what this measures.
--copy times the post stage alone (wg_rt_frame_io_timed) with the output written as RGBA16F rows, the form the D3D12
interop uses in game (post_tf_p12*), again interleaved.
--other-build / --other-model: e.g. a build from before a change with a converted model, each config timed in both
builds (same rounds), to see that a path's time did not move.
"""
import argparse
import ctypes
import os

import numpy as np

from clean.reference import encoder
from clean.runner import synthetic_model as SM
from clean.runner.output_filter_original_check import OutputCopy, WgRtExtIo
from clean.runner.pipeline import DEFAULT_MAP, Pipeline


def synthetic_frame(rw, rh, seed=7):
    rng = np.random.default_rng(seed)
    s = encoder.random_inputs(rng, rw=rw, rh=rh, fw=((rw + 31) // 32) * 32, fh=((rh + 31) // 32) * 32, special=False)
    s["motion"] = (s["motion"] * 0.25).astype(np.float32)   # mostly inside the frame, as in a game
    return {"color": s["color"], "depth": s["depth"], "motion": s["motion"], "mv_scale": np.ones(2, np.float32),
            "jitter": np.array([0.25, -0.125], np.float32), "exposure": 1.0, "pre_exposure": 1.0, "reset": False}


def table(title, names, t, kern):
    base = np.median(t[names[0]])
    print(title)
    for n in names:
        print(f"  {n}: {np.median(t[n]):.3f} ms ({min(t[n]):.3f} .. {max(t[n]):.3f}), {np.median(t[n]) - base:+.3f} ms"
              f"  [{kern[n]}]")


def post_stage_ms(copy, x, repeats):
    """Median GPU ms of the post stage (reconstruction + output filter) over `repeats` frames, output-copy form."""
    lib = copy.lib
    lib.wg_rt_frame_io_timed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(WgRtExtIo),
                                         ctypes.POINTER(ctypes.c_float * 4)]
    fr = copy.ref._upload(x)
    fr.reset = 0
    b = copy.bufs
    io = WgRtExtIo(color=b["color"], depth=b["depth"], motion=b["motion"], out=b["out"], expo_src=b["expo"],
                   exposure_mode=-1, raw_inputs=0, out_half_pitch=copy.pitch)
    ms = (ctypes.c_float * 4)()
    post, whole = [], []
    for _ in range(repeats):
        if lib.wg_rt_frame_io_timed(copy.rt, ctypes.addressof(fr), ctypes.byref(io), ctypes.byref(ms)) != 0:
            raise RuntimeError(lib.wg_rt_error(copy.rt).decode())
        post.append(ms[3])
        whole.append(sum(ms))
    return float(np.median(post)), float(np.median(whole))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", default=["output_filter=analytic", "output_filter=original"])
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--output", default="2560x1440")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--other-build")
    ap.add_argument("--other-model")
    ap.add_argument("--other-map", default=DEFAULT_MAP)
    ap.add_argument("--other-configs", nargs="+", default=["output_filter=default"])
    a = ap.parse_args()
    R, O = (tuple(int(v) for v in s.split("x")) for s in (a.render, a.output))
    if os.path.exists(os.path.join(a.model, "model.wgcm")):
        raise ValueError("--model is the work directory of a synthetic model, not a converted model")
    if not os.path.isdir(os.path.join(a.model, "raw")):
        SM.write_raw(a.model, SM.synthetic_tensors(1, taps="random"))
    x = synthetic_frame(*R)
    flags = {"depth_inverted": True, "auto_exposure": False}

    def make(build, model, option, **kw):
        return Pipeline(build, a.rocm, model, R, O, flags, kernels={"options": option}, check_packing=False, **kw)

    names = list(a.configs)
    ps = {c: make(a.build, a.model, c, model_format="raw") for c in names}
    if a.other_build:   # each build with its own configs on the other model (e.g. a converted one, read only)
        for c in a.other_configs:
            names.append("other build, " + c)
            ps[names[-1]] = make(a.other_build, a.other_model, c, map_path=a.other_map)
        for c in a.configs:
            if "original" not in c:   # a converted model of an earlier map has no weights for it
                names.append("this build on the other model, " + c)
                ps[names[-1]] = make(a.build, a.other_model, c, map_path=a.other_map)
    for p in ps.values():
        p.time_frame(x, 200)   # warm up, GPU clocks
    t = {n: [] for n in names}
    for _ in range(a.rounds):
        for n in names:
            t[n].append(ps[n].time_frame(x, a.repeats))
    kern = {n: f"{ps[n].kernels()['post_last']} / {ps[n].kernels()['filter_last']}, "
               f"output_filter={ps[n].kernels()['output_filter']}" for n in names}
    table(f"whole frame, render {R[0]}x{R[1]} -> {O[0]}x{O[1]}, {a.rounds} interleaved rounds of {a.repeats} frames "
          f"(median of the rounds' medians, min .. max, difference to the first)", names, t, kern)
    for p in ps.values():
        p.lib.wg_rt_destroy(p.rt)
        p.rt = None
    if a.copy:
        pitch = (8 * O[0] + 255) // 256 * 256
        cs = {c: OutputCopy(make(a.build, a.model, c, model_format="raw"), pitch) for c in a.configs}
        for c in cs.values():
            post_stage_ms(c, x, 200)
        tp, tw = {c: [] for c in a.configs}, {c: [] for c in a.configs}
        for _ in range(a.rounds):
            for c in a.configs:
                p_ms, w_ms = post_stage_ms(cs[c], x, a.repeats)
                tp[c].append(p_ms)
                tw[c].append(w_ms)
        kern = {c: f"{cs[c].kernels()['post_last']} / {cs[c].kernels()['filter_last']}" for c in a.configs}
        table("post stage (reconstruction + output filter), output copy: RGBA16F rows as in game, each frame "
              "synchronised", list(a.configs), tp, kern)
        table("  the four stages' sum of those frames", list(a.configs), tw, kern)
        for c in cs.values():
            c.close()


if __name__ == "__main__":
    main()
