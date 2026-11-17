# SPDX-License-Identifier: MIT
"""Model L's default route from a deployed runtime library alone (review P2: the embedded objects): the built
wronggpu_runtime.dll copied into an empty folder (no code object, no DXIL, no manifest next to it), and a Model L
context created there with no form variable set, then two frames:
  d3d12: native D3D12 (the game's executor): the hybrid route (kernels text l_form=hybrid, l_hyb_d3d12=on,
         swin_bneck=hyb6, hyb_tables=on) from the library's built-in manifest and DXIL;
  hip:   the HIP backend: the hybrid reference route (l_q7__hyb.co, l_q7w__hyb.co and the lc12 reconstruction
         l_post__lc12.co, which the library carries: build.py L_CODE_OBJECTS).
The folder is checked to hold only the library before and after (nothing read from or written next to it). Each leg
runs in its own process (the library reads its environment at load).

    python clean/l/runner/l_hyb_deployed_check.py --build <release build> --models <model root> --work <empty dir>
        [--rocm <ROCm folder>] [--json out.json]        (GPU: inside a GPU slot)
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

RW, RH, OW, OH = 128, 72, 192, 108
LIB = "wronggpu_runtime.dll"


def child(folder, models, backend, rocm):
    from clean.runner.pipeline import Pipeline
    p = Pipeline(folder, rocm if backend == "hip" else None, models, (RW, RH), (OW, OH), {"hdr": True},
                 backend=backend, map_path=os.path.join(ROOT, "data", "preset-L.v1.map.json"), check_packing=False,
                 kernels={"options": "model=L"})
    k = p.kernels() or {}
    rng = np.random.default_rng(3)
    finite = True
    for f in range(2):
        x = {"color": rng.uniform(0, 1, (RH, RW, 3)).astype(np.float32),
             "depth": rng.uniform(0, 1, (RH, RW)).astype(np.float32),
             "motion": np.zeros((RH, RW, 2), np.float32), "reset": f == 0, "mv_scale": (1.0, 1.0), "jitter": (0.0, 0.0)}
        out = p.frame(x)
        finite = finite and all(np.isfinite(np.asarray(v, np.float64)).all() for v in out.values()
                                if isinstance(v, np.ndarray))
    return {"kernels": {key: k.get(key) for key in ("model", "l_form", "l_hyb_d3d12", "swin_bneck", "hyb_tables",
                                                     "l_post_c12", "l_kernels")}, "finite": bool(finite)}


def listing(folder):
    return sorted(os.listdir(folder))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--json")
    ap.add_argument("--child", choices=("d3d12", "hip"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        print(json.dumps(child(a.work, a.models, a.child, a.rocm)))
        return 0
    res = {"legs": []}
    for backend in ("d3d12", "hip"):
        folder = os.path.join(a.work, "deployed-" + backend)
        if os.path.exists(folder):
            raise SystemExit(f"{folder} exists: use an empty --work")
        os.makedirs(folder)
        shutil.copyfile(os.path.join(a.build, LIB), os.path.join(folder, LIB))
        before = listing(folder)
        env = {k: v for k, v in os.environ.items() if k.upper() not in FORM_VARS}
        cp = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", backend, "--build", a.build,
                             "--models", a.models, "--work", folder, "--rocm", a.rocm], env=env, capture_output=True,
                            text=True, timeout=150, creationflags=0x08000000)
        try:
            r = json.loads(cp.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            r = {"child_failed": cp.stdout[-400:] + cp.stderr[-800:]}
        after = listing(folder)
        k = r.get("kernels", {})
        good = (before == [LIB] and after == [LIB] and r.get("finite") and k.get("model") == "L"
                and k.get("l_form") == "hybrid" and k.get("l_post_c12") == "on")
        if backend == "d3d12":
            good = good and k.get("l_hyb_d3d12") == "on" and k.get("swin_bneck") == "hyb6" and k.get("hyb_tables") == "on"
        else:
            good = good and "l_q7__hyb.co" in (k.get("l_kernels") or "") + str(k)
        res["legs"].append({"backend": backend, "folder_before": before, "folder_after": after, "result": r,
                            "pass": bool(good)})
        print(backend, "PASS" if good else "FAIL", json.dumps(r)[:400])
    res["pass"] = all(leg["pass"] for leg in res["legs"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
