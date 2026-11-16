# SPDX-License-Identifier: MIT
"""Whole-frame parity of the native D3D12 executor with the HIP whole frame on the same synthetic inputs, for a headless run
on Linux: the D3D12 side is clean/dx12/tests/frame_host.exe under Wine (a Windows build of the runtime library), the HIP side
and the comparison run natively in Python (clean/runner/pipeline.py, as compare.py does for a Windows machine). Synthetic
model and frames only (clean/runner/synthetic_model.py, clean/runner/output_filter_original_check.py frame_input).

  python frame_parity.py prepare --out <dir> --render 1706x960 --output 2560x1440 [--frames 3] [--seed 1]
      <dir>/model (raw tensors), <dir>/in (frame inputs), <dir>/spec.txt (the host's spec, see frame_host.cpp), <dir>/case.json
  python frame_parity.py ref --out <dir> --name hip|emu --build <build> [--rocm <ROCm>] [--time N]
      the HIP (or lane-emulator) whole frame: <dir>/<name>/f<i>_<buffer>.bin, times.txt, report.txt
  (run frame_host.exe on <dir>/spec.txt with --out <dir>/d3d12 under Wine, see wine_headless_run.sh)
  python frame_parity.py compare --out <dir> --a hip --b d3d12 [--json report.json]
      per named buffer: bytes that differ; for the output image: the differences in values; the GPU times of both

Not a gate: the output standard of an RDNA 3 route is the maintainer's open decision, this only measures.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(DX12))
for p in (ROOT, DX12):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare as C  # noqa: E402  (clean/dx12/compare.py: the buffer list and the fusion rule)
from clean.runner import output_filter_original_check as OC  # noqa: E402
from clean.runner import synthetic_model as SM  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

FLAGS = {"depth_inverted": True, "auto_exposure": True}


def size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def prepare(a):
    render, output = size(a.render), size(a.output)
    os.makedirs(os.path.join(a.out, "in"), exist_ok=True)
    model = os.path.join(a.out, "model")
    os.makedirs(model, exist_ok=True)
    SM.write_raw(model, SM.synthetic_tensors(a.seed))
    lines = ["model model", f"config max_rw {render[0]}", f"config max_rh {render[1]}", f"config ow {output[0]}",
             f"config oh {output[1]}", f"config depth_inverted {int(FLAGS['depth_inverted'])}",
             f"config auto_exposure {int(FLAGS['auto_exposure'])}", "config alpha 0.18"]
    from clean.runner import variants as V
    geo = V.DEFAULT_GEOMETRY   # the values Pipeline passes (the native executor ignores the HIP launch geometry)
    lines += [f"config gemm_threads {geo['gemm_threads']}", f"config nt {geo['nt']}",
              f"config wpw64 {geo['wpw']['swin_block_c64']}", f"config wpw96 {geo['wpw']['swin_block_c96']}",
              f"config wpw128 {geo['wpw']['swin_block_c128']}"]
    for k, kernel in enumerate(("swin_block_c64", "swin_block_c64_add", "swin_block_c96", "swin_block_c96_add", "swin_block_c128")):
        lines += [f"config block_threads {k} {geo['threads'][kernel]}", f"config block_wpg {k} {geo.get('wpg', {}).get(kernel, 0)}"]
    for i in range(a.frames):
        x = OC.frame_input(i, render, reset=i == 0)
        for name, dtype in (("color", np.float32), ("depth", np.float32), ("motion", np.float32)):
            np.ascontiguousarray(x[name], dtype=dtype).tofile(os.path.join(a.out, "in", f"f{i}_{name}.bin"))
        v = lambda key, n: " ".join(repr(float(t)) for t in np.asarray(x[key], np.float64).reshape(-1)[:n])
        lines.append(f"frame {render[0]} {render[1]} {int(x['reset'])} {v('mv_scale', 2)} 0.0 0.0 {v('jitter', 2)} "
                     f"{float(x['exposure'])!r} {float(x['pre_exposure'])!r} in/f{i}_color.bin in/f{i}_depth.bin in/f{i}_motion.bin")
    lines += [f"dump {n}" for n in C.BUFFERS] + ["dump filter.out"]
    with open(os.path.join(a.out, "spec.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "case.json"), "w") as f:
        json.dump({"render": render, "output": output, "frames": a.frames, "seed": a.seed}, f)
    print(f"prepared {a.frames} frames {render} -> {output} in {a.out}")


def case(out):
    with open(os.path.join(out, "case.json")) as f:
        return json.load(f)


def reference(a):
    c = case(a.out)
    render, output = tuple(c["render"]), tuple(c["output"])
    dest = os.path.join(a.out, a.name)
    os.makedirs(dest, exist_ok=True)
    p = Pipeline(a.build, a.rocm or "", os.path.join(a.out, "model"), render, output, FLAGS,
                 backend="emu" if a.name == "emu" else "hip", model_format="raw", check_packing=False)
    try:
        for i in range(c["frames"]):
            x = OC.frame_input(i, render, reset=i == 0)
            p.frame(x)
            for name, raw in C.raw_buffers(p).items():
                raw.tofile(os.path.join(dest, f"f{i}_{name}.bin"))
            p.raw("filter.out").tofile(os.path.join(dest, f"f{i}_filter.out.bin"))
        with open(os.path.join(dest, "report.txt"), "w") as f:
            f.write("kernels " + " ".join(f"{k}={v}" for k, v in p.kernels().items()) + "\n")
        if a.time:
            ms = [p.time_frame(OC.frame_input(c["frames"] - 1, render, reset=False), a.time) for _ in range(3)]
            with open(os.path.join(dest, "times.txt"), "w") as f:
                f.write("\n".join(repr(m) for m in ms) + "\n")
            print("whole frame median ms (3 runs):", ms)
    finally:
        OC.close(p)


def kernels_of(path):
    with open(os.path.join(path, "report.txt")) as f:
        line = next(l for l in f if l.startswith("kernels "))
    return dict(kv.split("=", 1) for kv in line[len("kernels "):].split())


def load(path, i, name):
    f = os.path.join(path, f"f{i}_{name}.bin")
    return np.fromfile(f, np.uint8) if os.path.exists(f) else None


def output_image(path, i, kernels, output):
    raw = load(path, i, "filter.out")
    if kernels.get("out") == "rgba16f":
        return raw.view(np.float16).reshape(output[1], output[0], 4)[..., :3].astype(np.float32)
    return raw.view(np.float32).reshape(output[1], output[0], 3).copy()


def f16_steps(x, y):
    """The distance of two float32 images in FP16 steps (as clean/tests/test_post_emu.f16_steps)."""
    def key(v):
        b = v.astype(np.float16).view(np.uint16).astype(np.int32)
        return np.where(b & 0x8000, 0x8000 - (b & 0x7fff), 0x8000 + b)
    return np.abs(key(x) - key(y))


def times(path):
    f = os.path.join(path, "times.txt")
    if not os.path.exists(f):
        return None
    v = [float(t) for t in open(f).read().split()]
    return {"median_ms": float(np.median(v)), "min_ms": float(np.min(v)), "n": len(v)}


def compare(a):
    c = case(a.out)
    output = tuple(c["output"])
    pa, pb = os.path.join(a.out, a.a), os.path.join(a.out, a.b)
    ka, kb = kernels_of(pa), kernels_of(pb)
    unwritten = C.fusion_unwritten(kb) | C.fusion_unwritten(ka)
    stale = {f"{s}.out" for s in OC.STALE}
    report = {"a": a.a, "b": a.b, "kernels_a": ka, "kernels_b": kb, "frames": []}
    print(f"{a.a} kernels: {ka}\n{a.b} kernels: {kb}")
    for i in range(c["frames"]):
        reset = i == 0
        rows = {}
        for name in C.BUFFERS:
            x, y = load(pa, i, name), load(pb, i, name)
            if x is None or y is None:
                continue
            if name == "exposure.pair":
                x, y = x[:C.EXPOSURE_BYTES], y[:C.EXPOSURE_BYTES]
            n = min(x.size, y.size)
            skipped = name in stale or name in unwritten or (reset and name in ("state.latent_prev", "state.motion_history", "state.history_prev")) \
                or (name == "post.display" and "fused" in (ka.get("filter_last"), kb.get("filter_last"))) \
                or (name in ("dec0.A", "dec0.B", "dec0.C") and ka.get("dec0") != kb.get("dec0"))   # another decoder format
            d = int(np.count_nonzero(x[:n] != y[:n]))
            rows[name] = {"bytes": int(n), "differ": d, "skipped": bool(skipped)}
        ia, ib = output_image(pa, i, ka, output), output_image(pb, i, kb, output)
        steps = f16_steps(ia, ib)
        px = np.any(steps > 0, axis=-1)
        scale = float(np.sqrt(np.mean(ia.astype(np.float64) ** 2))) or 1.0
        out = {"nonfinite_a": int((~np.isfinite(ia)).sum()), "nonfinite_b": int((~np.isfinite(ib)).sum()),
               "pixels_differ_pct": float(px.mean() * 100), "max_f16_steps": int(steps.max()),
               "median_f16_steps_where_differ": float(np.median(steps[steps > 0])) if (steps > 0).any() else 0.0,
               "max_abs": float(np.max(np.abs(ia - ib))), "rms_diff_over_rms": float(np.sqrt(np.mean((ia.astype(np.float64) - ib) ** 2)) / scale),
               "bytes_equal": bool(np.array_equal(load(pa, i, "filter.out"), load(pb, i, "filter.out")))}
        report["frames"].append({"frame": i, "reset": reset, "buffers": rows, "output": out})
        print(f"frame {i}{' (reset)' if reset else ''}: output pixels differing {out['pixels_differ_pct']:.3f} %, max {out['max_f16_steps']} FP16 steps, "
              f"RMS diff / RMS {out['rms_diff_over_rms']:.2e}, byte-equal {out['bytes_equal']}")
        for name, r in rows.items():
            print(f"   {name:22s} {r['bytes']:>10d} B  differ {r['differ']:>9d} ({100.0 * r['differ'] / max(r['bytes'], 1):6.3f} %){'  (not counted: unwritten, stale or another format)' if r['skipped'] else ''}")
    report["times"] = {a.a: times(pa), a.b: times(pb)}
    print("GPU times:", report["times"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--render", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("ref")
    p.add_argument("--out", required=True)
    p.add_argument("--name", choices=("hip", "emu"), required=True)
    p.add_argument("--build", required=True)
    p.add_argument("--rocm", default="")
    p.add_argument("--time", type=int, default=0)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--a", default="hip")
    p.add_argument("--b", default="d3d12")
    p.add_argument("--json")
    a = ap.parse_args()
    {"prepare": prepare, "ref": reference, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
