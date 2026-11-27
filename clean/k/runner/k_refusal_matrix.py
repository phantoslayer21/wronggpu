# SPDX-License-Identifier: GPL-3.0-only
"""R1 / H2: no supported context reaches the D3D12 create refusal. Run under the shared GPU lock.

Runs test_interop_d3d12.exe's `matrix` case against a runtime build: every usual upscaler-mode render ratio x output size x
declared maximum x HDR / LDR gets a default-executor context and one frame, and each must be taken by the D3D12 executor
with no create refusal and no HIP line. Model K runs with a synthetic model (generated here from random parameters; an
installed model is never read). Model M and Model L run with converted model files the caller names (--model-dir: the
folder with model.wgcm for M, the folder with L\\model.wgcm for L); the files are only read.

    python -m clean.k.runner.k_refusal_matrix --exe test_interop_d3d12.exe --build <build dir> --out <folder> [--outputs 1080,1440,2160]
    python -m clean.k.runner.k_refusal_matrix ... --model M --model-dir <folder>   (or --model L)
"""
import argparse
import os
from pathlib import Path
import subprocess

from clean.k.reference.network import random_network_params
from clean.k.tests.test_k_runtime_gpu import write_k_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--build", required=True, help="directory with wronggpu_runtime.dll and the kernels")
    ap.add_argument("--out", required=True, help="folder for the synthetic model and the log")
    ap.add_argument("--outputs", default="1080,1440,2160", help="output heights, comma separated")
    ap.add_argument("--model", choices=("K", "M", "L"), default="K")
    ap.add_argument("--model-dir", help="M, L: the folder with the converted model (M: model.wgcm; L: L\\model.wgcm)")
    ap.add_argument("--diag-keys", action="store_true",
                    help="add executor=hip and capture_dir=<out>/capture (diagnostic keys): a release build of the library must "
                         "ignore both (one line each) and stay on Direct3D 12; the case fails on a developer build")
    ap.add_argument("--hip-switch-on", action="store_true",
                    help="add hip_switch=on (ignored for every model): one log line per context, still Direct3D 12")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    build = Path(a.build).resolve()
    if a.model == "K":
        model = out / "synthetic"
        wgcm, mapfile = write_k_model(str(model), random_network_params(11))
        base = ("model=K", "network=fused", f"kernels_dir={build}", f"model_dir={model}",
                f"model_k_dir={Path(wgcm).parent}", f"model_k_map={mapfile}")
    else:
        if not a.model_dir:
            ap.error("--model-dir is required for model M and L")
        base = (f"model={a.model}", f"kernels_dir={build}", f"model_dir={Path(a.model_dir).resolve()}")
    extra = (("hip_switch=on",) if a.hip_switch_on else ()) + \
            (("executor=hip", f"capture_dir={out / 'capture'}") if a.diag_keys else ())
    opts = ";".join(base + extra)
    env = os.environ.copy()
    env.pop("WG_RUNTIME_OPTIONS", None)
    proc = subprocess.run([str(Path(a.exe).resolve()), str(build / "wronggpu_runtime.dll"), "matrix", opts, a.outputs],
                          capture_output=True, text=True, env=env, check=False, timeout=160)
    log = proc.stdout + proc.stderr
    (out / "matrix.log").write_text(log, encoding="utf-8")
    print("\n".join(l for l in log.splitlines() if "matrix" in l and ("FAIL" in l or "contexts" in l)))
    ok = proc.returncode == 0 and log.rstrip().endswith("PASS")
    print("PASS" if ok else "FAIL (see %s)" % (out / "matrix.log"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
