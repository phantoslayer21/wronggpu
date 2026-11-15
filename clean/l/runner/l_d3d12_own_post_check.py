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
(WG_D3D12_FUSION=none: every stage output written but b0 ... b4, "d3d12_own_stages"). Before every frame each run fills
its outputs with its own sentinel (l_frame_check fill_sentinels: d3d12_own 0x5A, d3d12_own_stages 0x3C; HIP's default
run 0xA5) and records every output's raw defined bytes (raw_outputs). --compare checks, with l_frame_check's raw-byte
comparison (raw_compare: a missing output, another size or any differing byte fails the frame; an output not written by
design must still hold that run's sentinel in every byte):
- against the HIP run (l_frame_check's "default", which runs l_post_recon and the same filter): every output byte for
  byte, the post outputs (full history, reduced history, display image, output) included; the D3D12 pack12 images in
  the base layout (pack12_to_base). Not written by design: b0 ... b4 on both (the bottleneck megakernel), enc1's merged
  map on D3D12 (fusion pm), on the first frame the previous frame's state on both;
- the "stages" run per stage against the reference on its own inputs, as l_frame_check --compare, with the post against
  Model L's reconstruction (clean/l/reference/recon.py) under Model L's post acceptance (clean/l/runner/l_post_check.py,
  per frame and over the run) and the filter against recon.output_filter; b5 as one unit (the bottleneck megakernel
  writes only b5), and where that chained reference parts from the runtime, b5 per stage on HIP's six-launch run
  (l_frame_check's "stages"), which must hold the same b5 bytes;
- the default kernels against the "stages" run, raw bytes;
- l_frame_check's record-level negative controls on both raw comparisons (a dropped history write, a dropped reduced
  write, a whole reduced write, a missing output): each must make the comparison fail.
Output: counts only, no timing.
"""
import argparse
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
from clean.l.runner import l_record as LR  # noqa: E402

OPTIONS = "model=L;l_post=own"
SENTINELS = {"d3d12_own": 0x5A, "d3d12_own_stages": 0x3C}


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
    for i, ((rw, rh), reset) in enumerate(F.SEQUENCE):
        fr = F.frame_inputs(rng, rw, rh, reset, special)
        F.fill_sentinels(p, SENTINELS[config], i == 0)
        out = p.frame(fr, debug=True)
        frames.append({"inputs": fr, "out": out, "raw": F.raw_outputs(p), "post_last": p.kernels().get("post_last")})
    rec = {"model": "L", "config": config, "sentinel": SENTINELS[config], "kernels": p.kernels(), "notes": p.notes(),
           "model_source": p.model_source, "dw": p.dw, "dh": p.dh, "ow": p.ow, "oh": p.oh,
           "taps": p.output_taps(), "frames": frames}
    with open(os.path.join(build, f"lframe_L_{config}.pkl"), "wb") as f:
        pickle.dump(rec, f)
    print("L", config, "frames", len(frames), "post", rec["kernels"].get("post"), "l_post", rec["kernels"].get("l_post"),
          "post_last", rec["kernels"].get("post_last"), "filter_last", rec["kernels"].get("filter_last"))


def _load(build, config):
    with open(os.path.join(build, f"lframe_L_{config}.pkl"), "rb") as f:
        return pickle.load(f)


def unwritten(k, n):
    """Outputs not written by design (raw_compare's sides): b0 ... b4 on both (the bottleneck megakernel), enc1's merged
    map with the D3D12 default kernels (fusion pm keeps it on chip), on the first frame the previous frame's state."""
    if n in F.BNECK_UNWRITTEN or (k == 0 and n in F.UNWRITTEN_ON_FIRST):
        return "both"
    if n == "enc1.merged":
        return "b"
    return None


def compare(build, json_path):
    with open(os.path.join(F.model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)["L"]
    hip, own, st = _load(build, "default"), _load(build, "d3d12_own"), _load(build, "d3d12_own_stages")
    res = {"kernels_d3d12": own["kernels"], "kernels_hip": hip["kernels"]}

    vs_hip = F.raw_compare(hip, own, hip["sentinel"], own["sentinel"], unwritten, convert_b=True)
    res["vs_hip"] = dict(F.raw_summary(vs_hip), frames=vs_hip)
    res["vs_hip"]["negative_controls"] = F.negative_controls(hip, own, hip["sentinel"], own["sentinel"], unwritten,
                                                             convert_b=True)

    st_dec = F.fill_pack12(st)  # dec0's A, B, C from the pack12 images (the debug view has none)
    per_stage = F.compare_model(st_dec, params, bneck_unit=True)
    post_run = {k: LP.l_post_run([m["post"][k] for m in per_stage if k in m["post"]]) for k in ("history", "display")}
    # b5 as one unit (b0 ... b5 chained in the reference from enc2.merged: the megakernel writes only b5) can part from
    # the runtime where one FP8 rounding in the chain goes the other way. Such a frame's b5 is checked per stage
    # instead: the HIP six-launch run ("stages", every block output written) has the same b5 bytes and its b5
    # passes against the reference on its own b4.
    hip_st = None
    b5_via_hip = []
    for k, m in enumerate(per_stage):
        if m["stage_pass"].get("b5", True):
            continue
        if hip_st is None:
            with open(os.path.join(build, "lframe_L_stages.pkl"), "rb") as f:
                hip_st = pickle.load(f)
            hip_ps = F.compare_model(hip_st, params)
        same = own["frames"][k]["raw"]["b5.out"].tobytes() == hip_st["frames"][k]["raw"]["b5.out"].tobytes()
        if same and hip_ps[k]["stage_pass"]["b5"]:
            m["stage_pass"]["b5"] = True
            m["b5_checked"] = "per stage on HIP's six-launch run (same b5 bytes): pass"
            m["pass"] = all(m["stage_pass"].values())
            b5_via_hip.append(k)
    res["d3d12_per_stage"] = {"frames": per_stage, "summary": F.summary(per_stage),
                              "passed": sum(m["pass"] for m in per_stage), "post_run": post_run,
                              "b5_unit_differs_checked_per_stage_on_hip": b5_via_hip}

    fused = F.raw_compare(st, own, st["sentinel"], own["sentinel"], unwritten)
    res["d3d12_default_vs_stages"] = dict(F.raw_summary(fused), frames=fused)
    res["d3d12_default_vs_stages"]["negative_controls"] = F.negative_controls(st, own, st["sentinel"], own["sentinel"],
                                                                              unwritten)
    n = len(own["frames"])
    caught = all(v["caught"] for r in (res["vs_hip"], res["d3d12_default_vs_stages"])
                 for v in r["negative_controls"].values())
    res["pass"] = (res["vs_hip"]["identical_frames"] == n and res["d3d12_per_stage"]["passed"] == n and
                   all(r["pass"] for r in post_run.values()) and
                   res["d3d12_default_vs_stages"]["identical_frames"] == n and caught and
                   own["kernels"].get("l_post") == "l_post_recon")
    LR.write_l_record(res, json_path)
    print("L d3d12 l_post=own:", "pass" if res["pass"] else "FAIL", "frames", n)
    print("  against the HIP run (raw bytes): identical", res["vs_hip"]["identical_frames"], "of", n,
          "outputs compared", res["vs_hip"]["outputs_compared"], "differing", res["vs_hip"]["differing"])
    print("  negative controls:", json.dumps({k: v["caught"] for k, v in res["vs_hip"]["negative_controls"].items()}))
    print("  per stage against the reference (Model L's post acceptance):", res["d3d12_per_stage"]["passed"], "of", n,
          "post over the run", json.dumps({k: r["pass"] for k, r in post_run.items()}),
          "(b5 as one unit off, checked per stage on HIP's six-launch run with the same bytes: frames", b5_via_hip, ")")
    print("  default kernels against the stages run (raw bytes): identical",
          res["d3d12_default_vs_stages"]["identical_frames"], "of", n, "differing",
          res["d3d12_default_vs_stages"]["differing"], "negative controls",
          json.dumps({k: v["caught"] for k, v in res["d3d12_default_vs_stages"]["negative_controls"].items()}))
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
