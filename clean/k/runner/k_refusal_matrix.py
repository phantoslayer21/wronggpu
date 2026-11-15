# SPDX-License-Identifier: MIT
"""R1 / H2: no supported Model K context reaches the D3D12 create refusal. Run under the shared GPU lock.

Runs test_interop_d3d12.exe's `matrix` case against a runtime build with a synthetic Model K (generated here from random
parameters; an installed model is never read): every usual upscaler-mode render ratio x output size x declared maximum x HDR / LDR
gets a default-executor context and one frame, and each must be taken by the D3D12 executor with no create refusal and no
HIP line. The same case takes the options of another model (Model M, L) when its files are given by the caller.

    python -m clean.k.runner.k_refusal_matrix --exe test_interop_d3d12.exe --build <build dir> [--outputs 1080,1440,2160]
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
    ap.add_argument("--hip-switch-on", action="store_true",
                    help="add hip_switch=on (ignored for every model): one log line per context, still Direct3D 12")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = out / "synthetic"
    wgcm, mapfile = write_k_model(str(model), random_network_params(11))
    build = Path(a.build).resolve()
    opts = ";".join(("model=K", "network=fused", f"kernels_dir={build}", f"model_dir={model}",
                     f"model_k_dir={Path(wgcm).parent}", f"model_k_map={mapfile}") +
                    (("hip_switch=on",) if a.hip_switch_on else ()))
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
