# SPDX-License-Identifier: MIT
"""Regression check of the hybrid device's code-object binding (manifest "co", clean/dx12/README.md): a synthetic model
WITHOUT output taps makes the runtime load post_w64.co (not post_w64__orig.co, the build the w64 port reproduces) under
the tag w64. The w64 port must then stay on HIP while the other ports run on D3D12, and the frames must match pure HIP
byte for byte (853x481 -> 1279x721, 3 frames).

    python clean/dx12/tests/variant_check.py --build <build dir>
        [--models <work dir>]
Synthetic parameters only (clean/runner/synthetic_model.py), never real model data.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(HERE))
import compare as C  # noqa: E402
from clean.runner import output_filter_original_check as OC  # noqa: E402
from clean.runner import synthetic_model as SM  # noqa: E402

PORT = "w64/post_tf_f32_p12_diet_vsit"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--models", default=os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "build")), "dx12-m", "work", "compare_models"))
    a = ap.parse_args(argv)
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        os.environ.pop(k, None)
    if (Path(a.models) / "model.wgcm").exists():
        raise SystemExit("refusing to write synthetic parameters into a converted-model directory")
    model = Path(a.models) / "plain"
    if not (model / "raw").exists():
        SM.write_raw(str(model), SM.synthetic_tensors(OC.SEED))
    trace = os.path.join(tempfile.mkdtemp(prefix="wgvar_"), "trace.txt")
    os.environ["WG_LAUNCH_TRACE"] = trace
    a.kernels = "all"
    title, render, out_size, flags, resets = C.CASES[1]
    r = C.run_case(a, model, title, render, out_size, flags, resets, 3, trace)
    bad = sum(row["differ"] for row in r["buffers"].values()) + sum(r["tree_values_differ"].values())
    for n, row in r["buffers"].items():
        if row["differ"]:
            print("DIFF", n, row)
    for k, v in r["tree_values_differ"].items():
        print("TREE DIFF", k, v)
    counts = {}
    for line in open(trace):
        t = line.split()
        key = (t[0], "dx12" if len(t) == 4 else "hip")
        counts[key] = counts.get(key, 0) + 1
    for k in sorted(counts):
        print(f"{k[0]:40s} {k[1]:5s} x{counts[k]}")
    port_dx = counts.get((PORT, "dx12"), 0)
    others_dx = sum(v for (n, w), v in counts.items() if w == "dx12")
    ok = bad == 0 and port_dx == 0 and others_dx > 0 and r["kernels"].get("output_taps") == "none"
    print(f"variant check: {'OK' if ok else 'FAILED'} ({bad} differing, {PORT} on D3D12 x{port_dx}, "
          f"D3D12 launches {others_dx})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
