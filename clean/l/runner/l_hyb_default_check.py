# SPDX-License-Identifier: GPL-3.0-only
"""Model L's shipped default on native D3D12 (0.6.5: WG_L_FORM unset = hybrid_full) is the hybrid route WITH its arena
tables, byte for byte the explicit WG_L_FORM=hybrid (oversight 2026-11-15 standing rule: every release-path check also
runs once with no model / form variable set). Found by code reading: the D3D12 device used to reserve and fill the
hybrid tables (GELU16 -> FP8, rsqrt, rcp; runtime/hyb_tables.h) only for an explicit WG_L_FORM=hybrid, so the default
route would have read whatever buffer the arena put there.

Cases (each its own Model L context, the same frames):
  1. default: no form / post variable set: create must give l_form=hybrid, l_hyb_d3d12=on, hyb_tables=on, l_post_c12=on;
  2. explicit: WG_L_FORM=hybrid: the same kernels text; every frame's output equal to case 1's, byte for byte;
  3. WG_HYB_TABLES=off with no form variable. A developer build (WG_DEV_DIAGNOSTICS, library marker
     "wg-build dev-diagnostics=1") honours the diagnostic: create must REFUSE with the runtime's one line (no other route
     runs a frame). A release build ignores it (review finding: a stray variable never refuses Model L for a player):
     create must succeed with hyb_tables=on and every frame equal to case 1's. (The refusal itself is also tested on
     the CPU recorder, clean/l/runtime/tests/test_l_runtime_frame.py test_hybrid_tables_refused_never_falls_back.)

    python clean/l/runner/l_hyb_default_check.py --build <S4 build> --models <model root> [--frames N] [--json out.json]
        (GPU: inside a GPU slot)

Every case runs in its OWN process with its environment set before the runtime library loads: the library links the
C runtime statically, and its getenv reads the environment copied at load, so a variable changed later in the same
process is invisible to it (an in-process second leg would silently repeat the first leg's environment).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner.l_d3d12_check import FORM_VARS  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

RW, RH, OW, OH = 256, 144, 384, 216


def frames(n):
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        out.append({"color": rng.uniform(0, 4, (RH, RW, 3)).astype(np.float32),
                    "depth": rng.uniform(0, 1, (RH, RW)).astype(np.float32),
                    "motion": rng.uniform(-2, 2, (RH, RW, 2)).astype(np.float32), "reset": k == 0,
                    "mv_scale": (1.0, 1.0), "jitter": (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.5, 0.5)))})
    return out


def run(build, models, env, xs):
    """(created, error, kernels, [sha256 per frame]) of one Model L context on native D3D12 under env (FORM_VARS set
    exactly as given, every other one removed), in a child process (see the module text)."""
    cenv = {k: v for k, v in os.environ.items() if k.upper() not in FORM_VARS}
    cenv.update(env)
    cp = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", "--build", build, "--models", models,
                         "--frames", str(len(xs))], env=cenv, capture_output=True, text=True, timeout=150)
    try:
        r = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "child failed: " + cp.stdout[-300:] + cp.stderr[-600:], {}, []
    return r["created"], r["error"], r["kernels"], r["shas"]


def child(build, models, xs):
    """One context in this process, under the environment it was started with; one JSON line on stdout."""
    try:
        p = Pipeline(build, None, models, (RW, RH), (OW, OH), {"hdr": True}, backend="d3d12",
                     map_path=os.path.join(ROOT, "data", "preset-L.v1.map.json"), check_packing=False,
                     kernels={"options": "model=L"})
    except RuntimeError as e:
        print(json.dumps({"created": False, "error": str(e), "kernels": {}, "shas": []}))
        return 0
    kern = p.kernels() or {}
    shas = [hashlib.sha256(np.ascontiguousarray(p.frame(dict(x))["out"]).tobytes()).hexdigest() for x in xs]
    print(json.dumps({"created": True, "error": "", "kernels": kern, "shas": shas}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    xs = frames(a.frames)
    if a.child:
        return child(a.build, a.models, xs)
    keys = ("l_form", "l_hyb_d3d12", "hyb_tables", "l_post_c12", "hyb_twins")
    res = {"cases": []}
    ok_d, err_d, kd, sd = run(a.build, a.models, {}, xs)
    want = {"l_form": "hybrid", "l_hyb_d3d12": "on", "hyb_tables": "on", "l_post_c12": "on"}
    res["cases"].append({"case": "default (no form variable)", "created": ok_d, "error": err_d[-300:],
                         "kernels": {k: kd.get(k) for k in keys}, "frames": len(sd),
                         "pass": ok_d and all(kd.get(k) == v for k, v in want.items())})
    ok_e, err_e, ke, se = run(a.build, a.models, {"WG_L_FORM": "hybrid"}, xs)
    res["cases"].append({"case": "explicit WG_L_FORM=hybrid", "created": ok_e, "error": err_e[-300:],
                         "kernels": {k: ke.get(k) for k in keys}, "frames": len(se),
                         "frames_identical_to_default": sum(x == y for x, y in zip(sd, se)),
                         "pass": ok_e and all(ke.get(k) == v for k, v in want.items()) and len(se) == len(xs)
                         and se == sd})
    ok_o, err_o, ko, so = run(a.build, a.models, {"WG_HYB_TABLES": "off"}, xs)
    lib = open(os.path.join(a.build, "wronggpu_runtime.dll"), "rb").read()
    dev = b"wg-build dev-diagnostics=1" in lib
    if dev:
        passed = (not ok_o) and "no hybrid tables" in err_o and "Model L hybrid (default)" in err_o and not so
    else:
        passed = ok_o and ko.get("hyb_tables") == "on" and so == sd and len(so) == len(xs)
    res["cases"].append({"case": "WG_HYB_TABLES=off, no form variable (%s build)" % ("developer" if dev else "release"),
                         "created": ok_o, "error": err_o[-400:],
                         "error_lines": len([ln for ln in err_o.splitlines() if ln.strip()]),
                         "hyb_tables": ko.get("hyb_tables"), "frames_identical_to_default": sum(x == y for x, y in zip(so, sd)),
                         "expect": "refused" if dev else "ignored", "pass": passed})
    res["frame_sha256_default"] = [s[:16] for s in sd]
    res["pass"] = all(c["pass"] for c in res["cases"])
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
