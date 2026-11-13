# SPDX-License-Identifier: MIT
"""Brief 15 hardware checks of the native D3D12 backend (backend 3) at native AA and ratios between 1.0 and 1.5:
the output kernel post_tf4_dl_* against the cap-352 kernel and against clean/reference/post.py. Synthetic model and
frames only; no HIP (backend 3 loads no HIP runtime). Run each mode under the GPU lock (WG_GPU_LOCK, BUILDING.md) (and a no-progress watchdog;
every frame prints a line).

    python clean/dx12/tests/native_ratio_check.py force     --build DIR --model MODEL --out report.json
    python clean/dx12/tests/native_ratio_check.py reference --build DIR --model MODEL --out report.json
    python clean/dx12/tests/native_ratio_check.py time      --build DIR --model MODEL --out report.json [--repeats 60]
    python clean/dx12/tests/native_ratio_check.py diag      --build DIR --model MODEL --out report.json

DIR holds wronggpu_runtime.dll (its D3D12 manifest and DXIL are built in); MODEL a raw synthetic model folder with
output-filter weights (e.g. the 'taps' model).

force:     at ratios >= 1.5 (where the cap-352 kernel runs), the same frames through the default runtime and through
           runtime option post_tf4=force (post_tf4_dl_*): every stage output, state image and the output must be
           byte-identical; the post kernels of the two runs are checked by name. Also: at ratios below 1.5 the default
           runtime runs post_tf4_dl_*.
reference: per ratio (1.0, 1.25, 1.33, 1.5 and odd sizes) each frame's post stage against reference/post.py on the
           frame's own inputs from the debug tree (colour, dec0's pack12 A / B / C, enc0's motion surfaces, the
           exposure pair, the previous history): FP16 steps of the history, codes of the reduced history, relative error
           of the output (reference output filter "original" on the reference display). The 1.5 row runs the cap-352
           kernel and is the baseline the other rows are judged against (the reference is float64, the kernels FP32).
time:      median whole-frame GPU time (Pipeline.time_frame) at native 1920x1080 and 2560x1440, with the Quality modes and
           post_tf4=force at Quality for comparison; rounds interleaved.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from clean.reference import encoder as En  # noqa: E402
from clean.reference import post as Po  # noqa: E402
from clean.runner import post as PS  # noqa: E402
from clean.runner.pipeline import Pipeline, logical_tensors, read_model_tensors  # noqa: E402

SEED = 15015
TF4 = "post_tf4_dl_f32_p12_diet_vsit"
TF = "post_tf_f32_p12_diet_vsit"
STALE = ("b0", "b1", "b2", "b3", "b4", "dec1")      # named buffers the default kernels do not write
ON_RESET = ("state.latent_prev", "state.motion_history", "state.history_prev")
RESETS = (0, 4)


def say(*a):
    print(*a, flush=True)


def frame_input(index, render, reset):
    rw, rh = render
    rng = np.random.default_rng([SEED, index, rw, rh])
    s = En.random_inputs(rng, rw=rw, rh=rh, fw=((rw + 31) // 32) * 32, fh=((rh + 31) // 32) * 32)
    return {"color": s["color"], "depth": s["depth"], "motion": s["motion"], "mv_scale": s["mv_scale"],
            "jitter": s["jitter"], "exposure": 0.75 + (index % 24) / 24, "pre_exposure": 1.0, "reset": reset}


def frames(n, render):
    return [frame_input(i, render, i in RESETS) for i in range(n)]


def make(a, render, out, options=None):
    return Pipeline(a.build, None, a.model, render, out, {"depth_inverted": False, "auto_exposure": False},
                    backend="d3d12", model_format="raw", check_packing=False,
                    kernels={"options": options} if options else None)


def close(*ps):
    for p in ps:
        if p is not None and getattr(p, "rt", None):
            p.lib.wg_rt_destroy(p.rt)
            p.rt = None


def compare_tree(a, b, reset, prefix=""):
    """Byte identity of two debug trees apart from STALE (and the previous-frame state on a reset frame); returns the
    number of values compared."""
    n = 0
    for name in a:
        path = prefix + name
        if path in STALE or (reset and path in ON_RESET) or path == "post.display":
            continue
        x, y = a[name], b[name]
        if isinstance(x, dict):
            n += compare_tree(x, y, reset, path + ".")
        elif isinstance(x, np.ndarray):
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                raise AssertionError("differs: " + path)
            n += x.size
        elif (x is None) != (y is None) or (x is not None and not isinstance(x, (np.ndarray, dict)) and x != y):
            raise AssertionError("differs: " + path)
    return n


def dw_dh(out):
    return (3 * out[0] + 1) // 2, (3 * out[1] + 1) // 2


def expected_post(render, out, force=False):
    dw, dh = dw_dh(out)
    return TF if PS.tf_supported(render[0], render[1], dw, dh) and not force else TF4


# ---- force ----------------------------------------------------------------------------------------------------------
FORCE_CASES = [("1.5 1280x720 -> 1920x1080", (1280, 720), (1920, 1080)), ("1.5 107x60 -> 160x90", (107, 60), (160, 90)),
               ("odd 1.5 853x481 -> 1279x721", (853, 481), (1279, 721)), ("2.0 960x540 -> 1920x1080", (960, 540), (1920, 1080))]
DEFAULT_TF4_CASES = [("native 160x90", (160, 90), (160, 90)), ("1.25 128x72 -> 160x90", (128, 72), (160, 90))]


def cmd_force(a):
    rep = {"mode": "force", "cases": [], "fail": 0}
    for label, render, out in FORCE_CASES:
        p0 = make(a, render, out)
        p1 = make(a, render, out, "post_tf4=force")
        row = {"case": label, "frames": a.frames, "values": 0, "output_bytes": 0, "kernels": None, "error": None}
        try:
            for i, x in enumerate(frames(a.frames, render)):
                t0, t1 = p0.frame(x, debug=True), p1.frame(x, debug=True)
                k0, k1 = p0.kernels()["post_last"], p1.kernels()["post_last"]
                row["kernels"] = [k0, k1]
                if (k0, k1) != (TF, TF4):
                    raise AssertionError(f"post kernels {k0} / {k1}")
                row["values"] += compare_tree(t0, t1, bool(x["reset"]))
                if t0["out"].tobytes() != t1["out"].tobytes():
                    raise AssertionError("output differs")
                row["output_bytes"] += t0["out"].nbytes
                say(f"force {label} frame {i}: identical ({k0} vs {k1})")
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
            rep["fail"] += 1
            say(f"force {label}: FAIL {e}")
        finally:
            close(p0, p1)
        rep["cases"].append(row)
    for label, render, out in DEFAULT_TF4_CASES:
        p = make(a, render, out)
        try:
            p.frame(frames(1, render)[0])
            k = p.kernels()["post_last"]
        finally:
            close(p)
        ok = k == TF4
        rep["fail"] += 0 if ok else 1
        rep["cases"].append({"case": label + " (default runtime)", "kernels": [k], "error": None if ok else "not " + TF4})
        say(f"default kernel at {label}: {k}")
    return rep


# ---- reference --------------------------------------------------------------------------------------------------------
REF_CASES = [("native 160x90", (160, 90), (160, 90)), ("1.25 128x72 -> 160x90", (128, 72), (160, 90)),
             ("1.33 120x68 -> 160x90", (120, 68), (160, 90)), ("1.5 107x60 -> 160x90 (cap-352 baseline)", (107, 60), (160, 90)),
             ("odd native 1281x721", (1281, 721), (1281, 721)), ("odd 1.25 1025x577 -> 1281x721", (1025, 577), (1281, 721)),
             ("odd 1.5 854x481 -> 1281x721 (cap-352 baseline)", (854, 481), (1281, 721))]


def order16(v):
    u = np.asarray(v, np.float16).view(np.uint16).astype(np.int32)
    return np.where(u & 0x8000, -(u & 0x7FFF), u)


def pack12(p, rw, rh):
    a8 = p._get("dec0.A", np.uint8).reshape(-1)[: rw * rh * 8].reshape(rh, rw, 8)
    c4 = p._get("dec0.C", np.uint8).reshape(-1)[: rw * rh * 4].reshape(rh, rw, 4)
    A = (a8[..., 0:3] / 255.0).astype(np.float32)
    B = (a8[..., 3:6] / 255.0).astype(np.float32)
    c01 = np.ascontiguousarray(c4).view(np.float16).reshape(rh, rw, 2)
    c2 = np.ascontiguousarray(a8[..., 6:8]).view(np.float16).reshape(rh, rw, 1)
    return A, B, np.concatenate([c01, c2], -1).astype(np.float32)


def cmd_reference(a):
    s = float(logical_tensors(read_model_tensors(a.model))["post"]["s"])
    rep = {"mode": "reference", "cases": []}
    for label, render, out in REF_CASES:
        rw, rh = render
        dw, dh = dw_dh(out)
        p = make(a, render, out)
        taps = p.output_taps()
        if taps is None:
            raise SystemExit("the model has no output-filter weights (use the taps model)")
        h = {"values": 0, "equal": 0, "one_step": 0, "more": 0, "far": 0}
        red = {"values": 0, "equal": 0, "one_code": 0, "more": 0}
        o = {"values": 0, "rel_gt_1e-3": 0, "rel_gt_1e-2": 0, "non_finite": 0, "max_rel": 0.0}
        kern = None
        try:
            for i, x in enumerate(frames(a.frames, render)):
                t = p.frame(x, debug=True)
                kern = p.kernels()["post_last"]
                if kern != expected_post(render, out):
                    raise AssertionError(f"post kernel {kern}, expected {expected_post(render, out)}")
                A, B, C = pack12(p, rw, rh)
                xr = {"color": np.asarray(x["color"], np.float32), "A": A, "B": B, "C": C,
                      "m_near": t["enc0"]["m_near"], "m_far": t["enc0"]["m_far"], "m_center": t["enc0"]["m_center"],
                      "history": t["state"]["history_prev"] if not x["reset"] else np.zeros((dh, dw, 3), np.float32),
                      "exposure": t["exposure"]["e"], "exposure_prev": t["exposure"]["e_prev"],
                      "jitter": x["jitter"], "reset": bool(x["reset"]), "recon_size": (dw, dh)}
                ref = Po.reconstruct(xr, s)
                st = np.abs(order16(ref["history"]) - order16(t["post"]["history"]))
                ad = np.abs(ref["history"].astype(np.float64) - t["post"]["history"].astype(np.float64))
                h["values"] += st.size
                h["equal"] += int((st == 0).sum())
                h["one_step"] += int((st == 1).sum())
                h["more"] += int((st > 1).sum())
                h["far"] += int(((st > 1) & (ad > 1e-3)).sum())
                d = np.abs(np.rint(ref["reduced"].astype(np.float64) * 1023) - np.rint(t["post"]["reduced"].astype(np.float64) * 1023))
                red["values"] += d.size
                red["equal"] += int((d == 0).sum())
                red["one_code"] += int((d == 1).sum())
                red["more"] += int((d > 1).sum())
                ro = Po.output_filter_original(ref["display"], t["exposure"]["e"], out, taps).astype(np.float64)
                g = t["out"].astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.abs(g - ro) / np.maximum(np.abs(ro), 1e-3)
                o["values"] += rel.size
                o["rel_gt_1e-3"] += int((rel > 1e-3).sum())
                o["rel_gt_1e-2"] += int((rel > 1e-2).sum())
                o["non_finite"] += int((~np.isfinite(g)).sum())
                o["max_rel"] = max(o["max_rel"], float(np.nanmax(rel)))
                say(f"reference {label} frame {i}: history steps>1 {int((st > 1).sum())}, reduced codes>1 "
                    f"{int((d > 1).sum())}, output rel>1e-3 {int((rel > 1e-3).sum())}")
        finally:
            close(p)
        row = {"case": label, "render": render, "output": out, "post_kernel": kern, "frames": a.frames,
               "history": h, "reduced": red, "output_vs_reference": o}
        rep["cases"].append(row)
        say(json.dumps(row))
    return rep


# ---- diag -----------------------------------------------------------------------------------------------------------
def cmd_diag(a):
    """Odd native 1281x721: (1) walk independence: the same frames with post_tf_cpairs 2 and 36 (other chunk plans, so
    other step starts, staging footprints and ring slots) must give the default's bytes; (2) the far reference pixels of
    the worst frame, saved (npz) with their D / output positions for CPU analysis."""
    render = out = (1281, 721)
    dw, dh = dw_dh(out)
    s = float(logical_tensors(read_model_tensors(a.model))["post"]["s"])
    rep = {"mode": "diag", "walk": [], "far": None}
    xs = frames(3, render)
    p0 = make(a, render, out)
    trees = []
    try:
        for i, x in enumerate(xs):
            t = p0.frame(x, debug=True)
            A, B, C = pack12(p0, *render)
            trees.append((t, A, B, C))
            say(f"diag default frame {i}: {p0.kernels()['post_last']}")
        taps = p0.output_taps()
    finally:
        close(p0)
    for opt in ("post_tf_cpairs=2", "post_tf_cpairs=36", "post_tf_cpairs=20"):
        p = make(a, render, out, opt)
        row = {"options": opt, "frames_identical": 0, "differs": []}
        try:
            for i, x in enumerate(xs):
                t = p.frame(x, debug=True)
                t0 = trees[i][0]
                for name, g0, g1 in (("history", t0["post"]["history"], t["post"]["history"]),
                                     ("reduced", t0["post"]["reduced"], t["post"]["reduced"]), ("out", t0["out"], t["out"])):
                    if g0.tobytes() != g1.tobytes():
                        row["differs"].append([i, name, int((g0 != g1).sum())])
                row["frames_identical"] += int(not any(d[0] == i for d in row["differs"]))
                say(f"diag {opt} frame {i}: {'identical' if not any(d[0] == i for d in row['differs']) else row['differs']}")
        finally:
            close(p)
        rep["walk"].append(row)
    t, A, B, C = trees[2]
    x = xs[2]
    xr = {"color": np.asarray(x["color"], np.float32), "A": A, "B": B, "C": C, "m_near": t["enc0"]["m_near"],
          "m_far": t["enc0"]["m_far"], "m_center": t["enc0"]["m_center"], "history": t["state"]["history_prev"],
          "exposure": t["exposure"]["e"], "exposure_prev": t["exposure"]["e_prev"], "jitter": x["jitter"],
          "reset": bool(x["reset"]), "recon_size": (dw, dh)}
    ref = Po.reconstruct(xr, s)
    ro = Po.output_filter_original(ref["display"], t["exposure"]["e"], out, taps)
    st = np.abs(order16(ref["history"]) - order16(t["post"]["history"]))
    ad = np.abs(ref["history"].astype(np.float64) - t["post"]["history"].astype(np.float64))
    far = (st > 1) & (ad > 1e-3)
    yy, xx, cc = np.nonzero(far)
    np.savez_compressed(os.path.splitext(a.out)[0] + "-frame2.npz", ref_history=ref["history"], history=t["post"]["history"],
                        ref_display=ref["display"], ref_out=ro, out=t["out"], ref_reduced=ref["reduced"],
                        reduced=t["post"]["reduced"], history_prev=t["state"]["history_prev"], color=xr["color"],
                        A=A, B=B, C=C, m_near=xr["m_near"], m_far=xr["m_far"], m_center=xr["m_center"],
                        jitter=np.asarray(x["jitter"], np.float64), e=t["exposure"]["e"], e_prev=t["exposure"]["e_prev"], s=s)
    rep["far"] = {"count": int(far.sum()), "rows_mod_16": np.bincount(yy % 16, minlength=16).tolist(),
                  "cols_mod_60": np.bincount(xx % 60, minlength=60).tolist(), "x_range": [int(xx.min()), int(xx.max())] if xx.size else None,
                  "y_range": [int(yy.min()), int(yy.max())] if yy.size else None}
    say(json.dumps(rep["far"]))
    rep["fail"] = sum(len(r["differs"]) for r in rep["walk"])
    return rep


# ---- time -----------------------------------------------------------------------------------------------------------
TIME_CASES = [("native 1920x1080", (1920, 1080), (1920, 1080), None),
              ("native 2560x1440", (2560, 1440), (2560, 1440), None),
              ("Ultra Quality 1477x831 -> 1920x1080 (1.3)", (1477, 831), (1920, 1080), None),
              ("Ultra Quality 1969x1108 -> 2560x1440 (1.3)", (1969, 1108), (2560, 1440), None),
              ("Quality 1280x720 -> 1920x1080", (1280, 720), (1920, 1080), None),
              ("Quality 1707x960 -> 2560x1440", (1707, 960), (2560, 1440), None),
              ("Quality 1707x960 -> 2560x1440, post_tf4=force", (1707, 960), (2560, 1440), "post_tf4=force")]


def cmd_time(a):
    rep = {"mode": "time", "repeats": a.repeats, "rounds": a.rounds, "cases": []}
    res = {c[0]: [] for c in TIME_CASES}
    kern = {}
    for r in range(a.rounds):
        for label, render, out, opt in TIME_CASES:
            p = make(a, render, out, opt)
            try:
                x = frame_input(1, render, False)
                p.frame(dict(x, reset=True))
                ms = p.time_frame(x, repeats=a.repeats)
                kern[label] = p.kernels()["post_last"]
            finally:
                close(p)
            res[label].append(ms)
            say(f"time round {r} {label}: {ms:.3f} ms ({kern[label]})")
    for label, render, out, opt in TIME_CASES:
        v = sorted(res[label])
        rep["cases"].append({"case": label, "render": render, "output": out, "post_kernel": kern[label],
                             "ms_rounds": res[label], "ms_median": v[len(v) // 2]})
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("force", "reference", "time", "diag"))
    ap.add_argument("--build", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    t0 = time.time()
    rep = {"force": cmd_force, "reference": cmd_reference, "time": cmd_time, "diag": cmd_diag}[a.mode](a)
    rep["seconds"] = round(time.time() - t0, 1)
    rep["build"] = os.path.abspath(a.build)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    say(f"{a.mode} done: {a.out}")
    return 1 if rep.get("fail") else 0


if __name__ == "__main__":
    sys.exit(main())
