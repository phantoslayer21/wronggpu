# SPDX-License-Identifier: GPL-3.0-only
"""The hybrid bottleneck's wait timeout REFUSES the frame and writes no history (review P1), on native D3D12 with a
developer runtime (build.py --dev-diagnostics: the hook WG_DEV_BNECK_ABORT=<frame> writes, right after that frame's
swin_bneck6_hyb launch, the words a timed-out wait writes: the abort word and the timeout count; hyb_bneck.hlsl).

Per leg (a Model L context, frames 0 .. 4 at 128 x 72 -> 192 x 108, random inputs):
  hook: WG_DEV_BNECK_ABORT=2. Five properties (oversight 2026-11-16 04:50Z):
        1. no output write: before frame 2 the display and the output are filled with a sentinel byte; after frame 2
           (synchronised) every byte is still the sentinel;
        2. no history write: the same for both history buffers and both latent buffers;
        3. the guard lifted before the recording ends: after frame 2, with the abort word still set, a fill of a scratch
           buffer (a launch of the runtime's own list) takes effect, byte for byte;
        4. frame 3 refused with exactly one log line (the runtime's message: one line, naming the timeout);
        5. frame 4 a clean reset: info reset 1, and it writes the output, its history and its latent state again.
  control: the same frames without the hook: frame 2 writes them (the check sees writes when there are any), no frame
        is refused, frame 4 is no reset.
  release (--release-build): a release runtime with WG_DEV_BNECK_ABORT=2 set behaves as the control (the hook is not
        in a release build; check_no_dev_diagnostics.py refuses a library that carries its name).
The hook and control legs run with WG_L_FORM=hybrid and with no form variable set (the shipped default), each context
in its own process (the library's getenv reads the environment copied at load: see l_hyb_refuse_check.py).
The kernel side (a real timed-out wait sets those words) is l_hyb_bneck_check.py's swin_bneck6_hyb_tmo control.

    python clean/l/runner/l_hyb_bneck_guard_check.py --build <developer build> --models <model root>
        [--release-build <release build>] [--json out.json]
        (GPU: inside a GPU slot)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner.l_d3d12_check import FORM_VARS  # noqa: E402

RW, RH, OW, OH = 128, 72, 192, 108
ABORT = 2
WATCH = ("post.history", "state.history_prev", "dec0.latent", "state.latent_prev", "post.display", "filter.out")
SENT = 0x5A
SCRATCH, SCRATCH_SENT = "post.reduced", 0x33


def child(build, models):
    from clean.runner.pipeline import Pipeline
    p = Pipeline(build, None, models, (RW, RH), (OW, OH), {"hdr": True}, backend="d3d12",
                 map_path=os.path.join(ROOT, "data", "preset-L.v1.map.json"), check_packing=False,
                 kernels={"options": "model=L"})
    kern = p.kernels() or {}
    rng = np.random.default_rng(5)
    out = {"kernels": {k: kern.get(k) for k in ("l_form", "l_hyb_d3d12", "swin_bneck")}, "frames": []}
    for f in range(5):
        x = {"color": rng.uniform(0, 1, (RH, RW, 3)).astype(np.float32),
             "depth": rng.uniform(0, 1, (RH, RW)).astype(np.float32),
             "motion": rng.uniform(-1, 1, (RH, RW, 2)).astype(np.float32), "reset": f == 0,
             "mv_scale": (1.0, 1.0), "jitter": (0.0, 0.0)}
        fills = f in (ABORT, 4)
        if fills:
            for nm in WATCH:
                p.fill(nm, SENT)
        e = {"frame": f}
        try:
            p.frame(x)
            e["ok"] = True
        except RuntimeError as err:
            e["ok"], e["error"] = False, str(err)
        e["reset"] = int(p.info().get("reset", -1))
        if fills:
            e["untouched"] = {nm: bool(np.all(p.raw(nm) == SENT)) for nm in WATCH}
        if f == ABORT:   # the abort word is still set here (cleared before frame 4): work after the frame must run
            p.fill(SCRATCH, SCRATCH_SENT)
            e["after_frame_runs"] = bool(np.all(p.raw(SCRATCH) == SCRATCH_SENT))
        out["frames"].append(e)
    return out


def run_leg(build, models, form, hook):
    cenv = {k: v for k, v in os.environ.items() if k.upper() not in FORM_VARS and k.upper() != "WG_DEV_BNECK_ABORT"}
    if form is not None:
        cenv["WG_L_FORM"] = form
    if hook:
        cenv["WG_DEV_BNECK_ABORT"] = str(ABORT)
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", "--build", build, "--models", models]
    cp = subprocess.run(cmd, env=cenv, capture_output=True, text=True, timeout=150, creationflags=0x08000000)
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"child_failed": cp.stdout[-400:] + cp.stderr[-800:]}


def written(u):
    """A frame wrote its history, its latent state and its output: one of each double-buffered pair (the other is the
    previous frame's, read only) and the output."""
    return (not (u["post.history"] and u["state.history_prev"]) and not (u["dec0.latent"] and u["state.latent_prev"])
            and not u["filter.out"])


def judge(r, hook):
    if "child_failed" in r:
        return False, "child failed"
    fr = r["frames"]
    if r["kernels"].get("l_hyb_d3d12") != "on" or r["kernels"].get("swin_bneck") != "hyb6":
        return False, "not the hybrid route with swin_bneck6_hyb"
    if hook:
        a, refused, after = fr[ABORT], fr[ABORT + 1], fr[ABORT + 2]
        if not all(e["ok"] for e in fr[:ABORT + 1]):
            return False, "a frame before the refusal failed"
        if not all(a["untouched"].values()):
            return False, "the timed-out frame wrote " + ",".join(k for k, v in a["untouched"].items() if not v)
        if not a.get("after_frame_runs"):
            return False, "work after the timed-out frame was skipped (the guard was not lifted)"
        if refused["ok"] or "timed out" not in refused.get("error", "") or "refused" not in refused["error"]:
            return False, "the next frame was not refused"
        if len([ln for ln in refused["error"].splitlines() if ln.strip()]) != 1:
            return False, "the refusal is not exactly one log line"
        if not (after["ok"] and after["reset"] == 1 and written(after["untouched"])):
            return False, "the frame after the refusal is not a reset that writes"
        return True, ""
    if not all(e["ok"] for e in fr):
        return False, "a frame failed without the hook"
    if not fr[ABORT].get("after_frame_runs"):
        return False, "the scratch fill after frame 2 did not take effect"
    if not written(fr[ABORT]["untouched"]) or fr[4]["reset"] != 0:
        return False, "the control frame wrote nothing (or a reset without the hook)"
    return True, ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--release-build", help="also: a release runtime with the hook variable set (must be the control)")
    ap.add_argument("--json")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        print(json.dumps(child(a.build, a.models)))
        return 0
    res = {"legs": []}
    for form in ("hybrid", None):
        for hook in (True, False):
            r = run_leg(a.build, a.models, form, hook)
            good, why = judge(r, hook)
            res["legs"].append({"form": form or "(default)", "hook": hook, "pass": good, "why": why, "result": r})
    if a.release_build:   # the hook variable set, judged as the control
        r = run_leg(a.release_build, a.models, None, True)
        good, why = judge(r, False)
        res["legs"].append({"form": "(default)", "hook": True, "release": True, "pass": good, "why": why, "result": r})
    res["pass"] = all(leg["pass"] for leg in res["legs"])
    for leg in res["legs"]:
        r = leg["result"]
        print(leg["form"], "release+hook" if leg.get("release") else "hook" if leg["hook"] else "control",
              "PASS" if leg["pass"] else "FAIL", leg["why"])
        for e in r.get("frames", []):
            print("  ", e["frame"], "ok" if e["ok"] else "REFUSED", "reset", e["reset"],
                  ("untouched " + ",".join(k for k, v in e["untouched"].items() if v)) if "untouched" in e else "",
                  (e.get("error") or "")[:160])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
