# SPDX-License-Identifier: MIT
"""The hybrid route never falls back (oversight 2026-11-15 18:36Z, hard rule; executor level: #225): when native D3D12
cannot take WG_L_FORM=hybrid, the runtime REFUSES at create with one line naming what is missing, and no other route
(the q7 / scalar reference, the M-form, the HIP executor) runs a frame in its place.

Cases, each on a copy of a build folder whose D3D12 manifest lacks something the hybrid route needs:
  1. one hybrid twin (swin_block_c128_hyb) removed: create must fail with the route's manifest message;
  2. the C32 decoder twin removed: create must fail likewise;
  3. control: the unmodified build creates on the hybrid route (kernels text l_form=hybrid,
     l_hyb_d3d12=on), so the refusals above are the missing kernels and nothing else.
Every case runs twice (standing rule, oversight 2026-11-15): with WG_L_FORM=hybrid and with NO form variable set (the
shipped default); both legs must give the same outcome.

Every case runs in its OWN process with its environment set before the runtime library loads: the library links the
C runtime statically, and its getenv reads the environment copied at load, so a variable changed later in the same
process is invisible to it (an in-process second leg would silently repeat the first leg's environment).

    python clean/l/runner/l_hyb_refuse_check.py --build <S4 build> --models <model root> --work <scratch dir>
        [--json out.json]        (GPU: inside a GPU slot)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner.l_d3d12_check import FORM_VARS  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

RW, RH, OW, OH = 128, 72, 192, 108


def strip_manifest(src, dst, drop):
    """A copy of build folder src at dst (files linked or copied) whose dx12/manifest.json lacks the kernels `drop`."""
    if os.path.exists(dst):   # no recursive delete here: the caller clears the work folder (guarded_delete)
        raise SystemExit(f"work folder {dst} exists: remove it first")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("lframe_model", "hyb_*", "*-chk"))
    mp = os.path.join(dst, "dx12", "manifest.json")
    m = json.load(open(mp))
    ks = m["kernels"] if isinstance(m, dict) and "kernels" in m else m
    keep = [k for k in ks if k.get("name") not in drop]
    assert len(keep) == len(ks) - len(drop), "kernels to drop not found in the manifest"
    if isinstance(m, dict) and "kernels" in m:
        m["kernels"] = keep
    else:
        m = keep
    json.dump(m, open(mp, "w"), indent=1)
    return dst


def try_create(build, models, frame=False, form="hybrid"):
    """(created, error text, kernels dict) of a Model L context on native D3D12 with WG_L_FORM=form (None: no form
    variable set, the shipped default), in a child process (see the module text)."""
    cenv = {k: v for k, v in os.environ.items() if k.upper() not in FORM_VARS}
    if form is not None:
        cenv["WG_L_FORM"] = form
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", "--build", build, "--models", models, "--work", "-"]
    cp = subprocess.run(cmd + (["--frame"] if frame else []), env=cenv, capture_output=True, text=True, timeout=150)
    try:
        r = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "child failed: " + cp.stdout[-300:] + cp.stderr[-600:], {}
    return r["created"], r["error"], r["kernels"]


def child_create(build, models, frame=False):
    """One context in this process, under its start environment: (created, error text, kernels dict)."""
    try:
        p = Pipeline(build, None, models, (RW, RH), (OW, OH), {"hdr": True}, backend="d3d12",
                     map_path=os.path.join(ROOT, "data", "preset-L.v1.map.json"), check_packing=False,
                     kernels={"options": "model=L"})
    except RuntimeError as e:
        return False, str(e), {}
    kern = p.kernels() or {}
    if frame:
        rng = np.random.default_rng(1)
        x = {"color": rng.uniform(0, 1, (RH, RW, 3)).astype(np.float32),
             "depth": rng.uniform(0, 1, (RH, RW)).astype(np.float32),
             "motion": np.zeros((RH, RW, 2), np.float32), "reset": True}
        p.frame(x)
    return True, "", kern


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--json")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--frame", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        ok, err, kern = child_create(a.build, a.models, a.frame)
        print(json.dumps({"created": ok, "error": err, "kernels": kern}))
        return 0
    res = {"cases": []}
    for leg, form in (("explicit", "hybrid"), ("default", None)):
        for name, drop in (("no_c128_twin", ["swin_block_c128_hyb"]), ("no_c32_dec_twin", ["l_swin_block_c32_dec_hyb"])):
            b = strip_manifest(a.build, os.path.join(a.work, leg, name), drop)   # one folder per leg: the first
            # leg's runtime DLL stays loaded in this process
            ok, err, _ = try_create(b, a.models, form=form)
            lines = [ln for ln in err.splitlines() if ln.strip()]
            # the refusal names the form as resolved (nothing set: the release default)
            label = "WG_L_FORM=hybrid" if form else "Model L hybrid (default)"
            refused = (not ok) and (label + " on native D3D12") in err and any(d in err for d in drop)
            res["cases"].append({"case": name, "leg": leg, "created": ok, "error": err[-400:], "error_lines": len(lines),
                                 "pass": refused})
        ok, err, kern = try_create(a.build, a.models, form=form)
        res["cases"].append({"case": "control", "leg": leg, "created": ok, "error": err[-400:],
                             "kernels": {k: kern.get(k) for k in ("l_form", "l_hyb_d3d12", "hyb_twins", "l_dec_skip",
                                                                  "l_post_c12", "hyb_tables")},
                             "pass": ok and kern.get("l_form") == "hybrid" and kern.get("l_hyb_d3d12") == "on"
                             and kern.get("hyb_tables") == "on"})
    res["pass"] = all(c["pass"] for c in res["cases"])
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
