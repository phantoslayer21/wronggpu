# SPDX-License-Identifier: MIT
"""Synthetic A/B for reset input specialization. Run under the shared GPU lock.

Example: python -m clean.k.runner.k_reset_bench --build build --json results.json
Both paths use the same build, FP32 network and full spec-44 geometry. The baseline
selects the retained k_in_base on resets. Complete input/network/output timing
excludes uploads, downloads, exposure, D3D12 interop and game work. No estimated
stage subtraction is used. Alternating order limits clock/temperature bias.
"""
import argparse
import json
import time

import numpy as np

from clean.k.reference.network import random_network_params
from clean.k.runner.k_bench import KFuncs, open_device
from clean.k.runner.k_frame import KFrame


class GeneralFrame(KFrame):
    def _launches(self, d):
        parts, f = super()._launches(d)
        parts = [(part, [(name.replace("k_in_reset", "k_in_base"), grid, threads, args)
                         for name, grid, threads, args in launches]) for part, launches in parts]
        return parts, f


def time_pipeline(kf, x, n, input_only=False):
    """Host elapsed enqueue/drain time per frame after n warm-up frames; no device event
    boundaries whose timestamps can be batched across adjacent launches. Inputs already
    uploaded. input_only is explicitly isolated, and never subtracted from the total."""
    parts, _ = kf._launches(kf._frame_dict(x))
    calls = [(kf.func(name), grid, threads, args) for part, ls in parts
             if not input_only or part == "input" for name, grid, threads, args in ls]
    def enqueue():
        for fn, grid, threads, args in calls:
            kf.dev.launch(fn, grid, threads, args)
    for _ in range(n):
        enqueue()
    kf.dev.sync()
    start = time.perf_counter()
    for _ in range(n):
        enqueue()
    kf.dev.sync()
    return (time.perf_counter() - start) * 1000 / n


def run(build, rocm, rounds=5, frames=100, sizes=((1920, 1080), (2560, 1440))):
    dev = open_device("hip", build, rocm)
    fn = KFuncs(dev, build, "hip")
    params = random_network_params(83)
    result = {"device": dev.device_name, "network": "fused (FP32)", "rounds": rounds,
              "frames_per_round": frames, "cases": []}
    for W, H in sizes:
        rw, rh = round(W / 1.5), round(H / 1.5)
        rng = np.random.default_rng(41)
        flags = dict(hdr=True, mv_lowres=True, depth_inverted=True)
        x = dict(color=(rng.random((rh, rw, 4)) * 2).astype(np.float32),
                 motion=(rng.standard_normal((rh, rw, 2)) * .1).astype(np.float32),
                 depth=rng.random((rh, rw)).astype(np.float32), render_size=(rw, rh),
                 jitter=(.13, -.27), mv_scale=(1., 1.), pre_exposure=.8,
                 reset=True, E_cur=1.1, E_aux=1.3, e1=1., e2=1.)
        a = GeneralFrame(dev, fn, (W, H), (rw, rh), params, flags)
        b = KFrame(dev, fn, (W, H), (rw, rh), params, flags)
        try:
            # Full-frame comparison, including the histories consumed by later frames.
            # First frame, steady frame, explicit reset, and render-size-change reset.
            compared = 0
            for i in range(4):
                xx = dict(x, reset=i == 2)
                if i == 3:
                    xx.update(render_size=(rw-1, rh-1), color=x["color"][:-1, :-1].copy(),
                              motion=x["motion"][:-1, :-1].copy(), depth=x["depth"][:-1, :-1].copy())
                aa, bb = a.frame(xx, debug=True), b.frame(xx, debug=True)
                assert aa["f"] == bb["f"] and aa["reset"] == bb["reset"]
                for name in ("out", "x0", "N"):
                    np.testing.assert_array_equal(aa[name].view(np.uint32), bb[name].view(np.uint32))
                    assert np.isfinite(bb[name]).all(), name
                    compared += aa[name].size
                ah, bh = a.histories(), b.histories()
                for name in ah:
                    np.testing.assert_array_equal(ah[name].view(np.uint32), bh[name].view(np.uint32))
                    compared += ah[name].size
            for reset in (True, False):
                # Return both states to this render size before steady-frame timing.
                a.frame(x); b.frame(x)
                xx = dict(x, reset=reset)
                measurements = {"general": [], "specialized": []}
                for r in range(rounds):
                    order = (("general", a), ("specialized", b))
                    if r % 2:
                        order = order[::-1]
                    for name, kf in order:
                        measurements[name].append(dict(pipeline=time_pipeline(kf, xx, frames),
                                                       input_only=time_pipeline(kf, xx, frames, input_only=True)))
                summary = {name: {key: float(np.median([v[key] for v in rows])) for key in rows[0]}
                           for name, rows in measurements.items()}
                case = dict(output=[W, H], render=[rw, rh], base=[a.Bw, a.Bh], reset=reset,
                            exact_compared_values=compared, mismatches=0, median_ms=summary, rounds=measurements)
                result["cases"].append(case)
                print(json.dumps({k: v for k, v in case.items() if k != "rounds"}), flush=True)
        finally:
            a.release(); b.release()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default="")
    ap.add_argument("--json", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args()
    if a.rounds < 1 or a.frames < 20:
        ap.error("rounds >= 1 and frames >= 20 required")
    result = run(a.build, a.rocm, a.rounds, a.frames)
    with open(a.json, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
