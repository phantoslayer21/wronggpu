# SPDX-License-Identifier: MIT
"""Model L (issue #89): the native D3D12 frame with Model L's own reconstruction (runtime option l_post=own:
l_post_recon_p12, then the analytic filter w64/post_filter_c1_s4) against the HIP frame and the CPU reference, on the
frames, models and build of clean/l/runner/l_frame_check.py.

    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --prepare        # CPU, once
    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --run            # GPU: the HIP run
    python clean/l/runner/l_d3d12_own_post_check.py --build <dir> --run                 # GPU, inside a GPU slot
    python clean/l/runner/l_d3d12_own_post_check.py --build <dir> --run --stages        # GPU, inside a GPU slot
    python clean/l/runner/l_d3d12_own_post_check.py --build <dir> --compare --json out.json   # CPU

<dir>/dx12 holds the D3D12 ports (python clean/dx12/build_dx12.py --out <dir>). --run runs l_frame_check's frame
sequence for Model L on D3D12 with l_post=own ("d3d12_own"); with --stages the D3D12 fusions are off
(WG_D3D12_FUSION=none: every stage output written but b0 ... b4, "d3d12_own_stages"). --compare checks:
- against the HIP run (l_frame_check's "default", which runs l_post_recon and the same filter): every output both
  backends write, byte for byte, the post outputs (history, reduced history, display image) and the output included;
  the pack12 images unpacked against HIP's A, B, C. Outputs all zero on D3D12 and not on HIP are reported as not
  written there (the fusions keep some maps on chip), not as differing. Negative control: one output bit flipped in a
  copy of a D3D12 frame must show as one differing value;
- the "stages" run per stage against the reference on its own inputs, as l_frame_check --compare, with the post against
  Model L's reconstruction (clean/l/reference/recon.py) under Model L's post acceptance (clean/l/runner/l_post_check.py,
  per frame and over the run) and the filter against recon.output_filter; b5 as one unit (the bottleneck megakernel
  writes only b5), and where that chained reference parts from the runtime, b5 per stage on HIP's six-launch run
  (l_frame_check's "stages"), which must hold the same b5 bytes;
- the default kernels against the "stages" run, byte for byte.
Output: counts only, no timing.
"""
import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_frame_check as F  # noqa: E402
from clean.l.runner import l_post_check as LP  # noqa: E402

OPTIONS = "model=L;l_post=own"


def run(build, special=True, stages=False):
    if stages:
        os.environ["WG_D3D12_FUSION"] = "none"  # before the runtime library is loaded
    from clean.runner.pipeline import Pipeline
    config = "d3d12_own_stages" if stages else "d3d12_own"
    p = Pipeline(build, "", F.model_dir(build), F.MAX_RENDER, F.OUTPUT,
                 {"depth_inverted": False, "auto_exposure": True, "hdr": True}, backend="d3d12",
                 map_path=F.l_map(build), check_packing=False, kernels={"options": OPTIONS})
    rng = np.random.default_rng(F.SEED)
    frames = []
    for (rw, rh), reset in F.SEQUENCE:
        fr = F.frame_inputs(rng, rw, rh, reset, special)
        out = p.frame(fr, debug=True)
        raw = {n: p._get(n, np.uint8).copy() for n in ("dec0.A", "dec0.C")}
        frames.append({"inputs": fr, "out": out, "raw": raw})
    rec = {"model": "L", "config": config, "kernels": p.kernels(), "notes": p.notes(),
           "model_source": p.model_source, "dw": p.dw, "dh": p.dh, "ow": p.ow, "oh": p.oh,
           "taps": p.output_taps(), "frames": frames}
    with open(os.path.join(build, f"lframe_L_{config}.pkl"), "wb") as f:
        pickle.dump(rec, f)
    print("L", config, "frames", len(frames), "post", rec["kernels"].get("post"), "l_post", rec["kernels"].get("l_post"),
          "post_last", rec["kernels"].get("post_last"), "filter_last", rec["kernels"].get("filter_last"))


def _load(build, config):
    with open(os.path.join(build, f"lframe_L_{config}.pkl"), "rb") as f:
        return pickle.load(f)


def with_display(hip, dx, frames):
    """compare_backends leaves the display image out (Model M's fused post writes none); both runs here write it:
    values differing, per frame."""
    for m, a, b in zip(frames, hip["frames"], dx["frames"]):
        u, v = a["out"]["post"]["display"], b["out"]["post"]["display"]
        n = -1 if u is None or v is None or u.shape != v.shape else int(np.sum(u.view(np.uint16) != v.view(np.uint16)))
        m["differ"]["post.display"] = n
        m["identical"] = m["identical"] and n == 0
    return frames


def compare(build, json_path):
    with open(os.path.join(F.model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)["L"]
    hip, own, st = _load(build, "default"), _load(build, "d3d12_own"), _load(build, "d3d12_own_stages")
    res = {"kernels_d3d12": own["kernels"], "kernels_hip": hip["kernels"]}

    vs_hip = with_display(hip, own, F.compare_backends(hip, own, same_post=True))
    res["vs_hip"] = {"frames": vs_hip, "identical_frames": sum(f["identical"] for f in vs_hip),
                     "outputs_compared": sorted({n for f in vs_hip for n in f["differ"]}),
                     "differing": sorted({k for f in vs_hip for k, v in f["differ"].items() if v}),
                     "unwritten_d3d12": sorted({n for f in vs_hip for n in f["unwritten_d3d12"]})}
    bad = copy.deepcopy(own)
    o = bad["frames"][1]["out"]["out"]
    o.view(np.uint32).reshape(-1)[o.size // 2] ^= 1
    caught = with_display(hip, bad, F.compare_backends(hip, bad, same_post=True))[1]["differ"].get("out")
    res["negative_control"] = {"flipped": "out, frame 1, one bit", "out_differing": caught, "caught": caught == 1}

    st = F.fill_pack12(st)  # dec0's A, B, C from the pack12 images (the debug view has none)
    per_stage = F.compare_model(st, params, bneck_unit=True)
    post_run = {k: LP.l_post_run([m["post"][k] for m in per_stage if k in m["post"]]) for k in ("history", "display")}
    # b5 as one unit (b0 ... b5 chained in the reference from enc2.merged: the megakernel writes only b5) can part from
    # the runtime where one FP8 rounding in the chain goes the other way. Such a frame's b5 is checked per stage
    # instead: the HIP six-launch run ("stages", every block output written) has the same b5 bytes and its b5
    # passes against the reference on its own b4.
    hip_st = _load(build, "stages")
    hip_ps = None
    b5_via_hip = []
    for k, m in enumerate(per_stage):
        if m["stage_pass"].get("b5", True):
            continue
        hip_ps = hip_ps or F.compare_model(hip_st, params)
        same = own["frames"][k]["out"]["b5"]["out"].tobytes() == hip_st["frames"][k]["out"]["b5"]["out"].tobytes()
        if same and hip_ps[k]["stage_pass"]["b5"]:
            m["stage_pass"]["b5"] = True
            m["b5_checked"] = "per stage on HIP's six-launch run (same b5 bytes): pass"
            m["pass"] = all(m["stage_pass"].values())
            b5_via_hip.append(k)
    res["d3d12_per_stage"] = {"frames": per_stage, "summary": F.summary(per_stage),
                              "passed": sum(m["pass"] for m in per_stage), "post_run": post_run,
                              "b5_unit_differs_checked_per_stage_on_hip": b5_via_hip}
    fused = F.compare_backends(st, own)
    res["d3d12_default_vs_stages"] = {"frames": fused, "identical_frames": sum(f["identical"] for f in fused),
                                      "differing": sorted({k for f in fused for k, v in f["differ"].items() if v}),
                                      "unwritten_default": sorted({n for f in fused for n in f["unwritten_d3d12"]})}
    n = len(own["frames"])
    res["pass"] = (res["vs_hip"]["identical_frames"] == n and res["negative_control"]["caught"] and
                   res["d3d12_per_stage"]["passed"] == n and all(r["pass"] for r in post_run.values()) and
                   res["d3d12_default_vs_stages"]["identical_frames"] == n and
                   own["kernels"].get("l_post") == "l_post_recon")
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print("L d3d12 l_post=own:", "pass" if res["pass"] else "FAIL", "frames", n)
    print("  against the HIP run: byte-identical", res["vs_hip"]["identical_frames"], "of", n, "differing",
          res["vs_hip"]["differing"], "not written on D3D12", res["vs_hip"]["unwritten_d3d12"])
    print("  negative control:", json.dumps(res["negative_control"]))
    print("  per stage against the reference (Model L's post acceptance):", res["d3d12_per_stage"]["passed"], "of", n,
          "post over the run", json.dumps({k: r["pass"] for k, r in post_run.items()}),
          "(b5 as one unit off, checked per stage on HIP's six-launch run with the same bytes: frames",
          res["d3d12_per_stage"]["b5_unit_differs_checked_per_stage_on_hip"], ")")
    print("  default kernels against the stages run: byte-identical", res["d3d12_default_vs_stages"]["identical_frames"],
          "of", n, "differing", res["d3d12_default_vs_stages"]["differing"])
    for m in per_stage:
        if not m["pass"]:
            print("  frame", m["frame"], "failing:", [s for s, ok in m["stage_pass"].items() if not ok])
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--compare", action="store_true")
    ap.add_argument("--stages", action="store_true", help="--run with the D3D12 fusions off")
    ap.add_argument("--plain", action="store_true", help="--run: frames without special values (as l_frame_check)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.run:
        run(a.build, special=not a.plain, stages=a.stages)
        return 0
    return compare(a.build, a.json or os.path.join(a.build, "l_d3d12_own_post.json"))


if __name__ == "__main__":
    sys.exit(main())
