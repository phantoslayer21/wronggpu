# SPDX-License-Identifier: GPL-3.0-only
"""Model L (issue #89): which reconstruction should Model L's native D3D12 frame use by default? Both candidates against
Model L's post oracle (clean/l/reference/recon.py), on l_frame_check's frames and models (with --prepare --model: the
converter's real-weight model.wgcm, the frames synthetic).

    python clean/l/runner/l_frame_check.py --build <dir> --rocm <ROCm> --prepare [--model <model.wgcm>]   # CPU, once
    python clean/l/runner/l_default_post_check.py --build <dir> --run          # GPU, inside a GPU slot
    python clean/l/runner/l_default_post_check.py --build <dir> --compare --json out.json                # CPU

<dir>/dx12 holds the D3D12 ports (python clean/dx12/build_dx12.py --out <dir> --model-l). The two candidates were:
- "m_default" (A): Model M's fused pack12 post (w64/post_tf_f32_p12_diet_vsit, then the runtime's default for Model L on
  native D3D12, runtime option l_post=m). It writes the full and reduced history but no display image, so A is judged on
  the full history and the reduced codes. It failed every frame (clean/l/records/l_default_post_real_weights.json), and
  the runtime no longer has that route for Model L (l_post=m is refused): --run no longer runs it; --compare still
  judges an A record a run before that change left in <dir> (lframe_L_dp_m_default.pkl);
- "own_stages" (B): Model L's own post (l_post_recon_p12, then the analytic filter) with the D3D12 fusions off
  (WG_D3D12_FUSION=none, every stage output written), judged on the full history, the display image and the reduced
  codes. (l_d3d12_own_post_check.py shows the default kernels give the same bytes.) --run runs it in its own process.
--compare checks each run's post per frame on the frame's own inputs (l_frame_check compare_model: the unpacked
pack12 coefficients, the run's own previous history) against recon.reconstruct, with Model L's post acceptance
(clean/l/runner/l_post_check.py): FP16 values more than one ulp off at most 0.05 % per frame and 0.01 % over the run
(history and display), and the rest of l_post_rule (no non-finite value, at most 1 % differing, mean error at most
0.1 % of the reference RMS); the reduced codes within 1 and 99 % exact. Output: counts only, no timing; the model's
source (synthetic seed, or the file's and map's SHA-256) from l_frame_check's source.json.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_frame_check as F  # noqa: E402
from clean.l.runner import l_post_check as LP  # noqa: E402
from clean.l.runner import l_record as LR  # noqa: E402

CONFIGS = {"own_stages": ("own", True, 0x3C)}   # (A, "m_default", ran with l_post=m: no longer a runtime route)


def run_one(build, config):
    l_post, stages, sentinel = CONFIGS[config]
    if stages:
        os.environ["WG_D3D12_FUSION"] = "none"  # before the runtime library is loaded
    from clean.runner.pipeline import Pipeline
    p = Pipeline(build, "", F.model_dir(build), F.MAX_RENDER, F.OUTPUT,
                 {"depth_inverted": False, "auto_exposure": True, "hdr": True}, backend="d3d12",
                 map_path=F.l_map(build), check_packing=False, kernels={"options": f"model=L;l_post={l_post}"})
    rng = np.random.default_rng(F.SEED)
    frames = []
    for i, ((rw, rh), reset) in enumerate(F.SEQUENCE):
        fr = F.frame_inputs(rng, rw, rh, reset, True, index=i)
        F.fill_sentinels(p, sentinel, i == 0)
        out = p.frame(fr, debug=True)
        frames.append({"inputs": fr, "out": out, "raw": F.raw_outputs(p), "post_last": p.kernels().get("post_last")})
    rec = {"model": "L", "config": config, "sentinel": sentinel, "kernels": p.kernels(), "notes": p.notes(),
           "model_source": p.model_source, "dw": p.dw, "dh": p.dh, "ow": p.ow, "oh": p.oh,
           "taps": p.output_taps(), "frames": frames}
    with open(os.path.join(build, f"lframe_L_dp_{config}.pkl"), "wb") as f:
        pickle.dump(rec, f)
    print("L", config, "frames", len(frames), "post", rec["kernels"].get("post"), "l_post", rec["kernels"].get("l_post"),
          "post_last", rec["kernels"].get("post_last"))


def _load(build, config):
    with open(os.path.join(build, f"lframe_L_dp_{config}.pkl"), "rb") as f:
        return pickle.load(f)


def post_frames(rec, params):
    """Per frame: history and display against recon.reconstruct under Model L's post acceptance, reduced codes."""
    per = F.compare_model(F.fill_pack12(rec), params, bneck_unit=True)
    rows = []
    for m in per:
        row = {"frame": m["frame"]}
        for k in ("history", "display"):
            if k not in m["post"]:
                continue
            v = m["post"][k]
            row[k] = {"values": v["values"], "over_1": v["over_1"], "pct_over_1": 100.0 * v["over_1"] / v["values"],
                      "max_f16_steps": v["max_f16_steps"], "differ": v["differ"], "nonfinite": v["nonfinite"],
                      "mean_rel_rms": v["mean_rel_rms"], "pass": bool(v["pass"]),
                      "pass_over_1": v["over_1"] <= LP.L_OVER1_CASE * v["values"]}
        r = m["post"]["reduced"]
        row["reduced"] = {"values": r["values"], "differ": r["differ"], "max_code_diff": r["max_code_diff"],
                          "pass": bool(r["pass"])}
        row["pass"] = all(row[k]["pass"] for k in ("history", "display", "reduced") if k in row)
        rows.append(row)
    keys = [k for k in ("history", "display") if k in rows[0]]
    run = {k: LP.l_post_run([r[k] for r in rows]) for k in keys}
    for v in run.values():
        v["pct_over_1"] = 100.0 * v["frac_over_1"]
    worst = max(rows, key=lambda r: max(r[k]["pct_over_1"] for k in keys))
    return {"frames": rows, "run": run, "frames_passed": sum(r["pass"] for r in rows),
            "frames_passed_over_1": sum(all(r[k]["pass_over_1"] for k in keys) for r in rows),
            "worst_frame": dict({"frame": worst["frame"]}, **{f"{k}_pct_over_1": worst[k]["pct_over_1"] for k in keys}),
            "pass": all(r["pass"] for r in rows) and all(v["pass"] for v in run.values())}


def compare(build, json_path):
    with open(os.path.join(F.model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)["L"]
    src = os.path.join(F.model_dir(build), "source.json")
    res = {"model_source": json.load(open(src)) if os.path.exists(src) else None,
           "rule": {"over_1_per_frame": LP.L_OVER1_CASE, "over_1_run": LP.L_OVER1_RUN}}
    own = _load(build, "own_stages")
    res["kernels"] = {"own_stages": own["kernels"]}
    keys = ["B_own_post"]
    if os.path.exists(os.path.join(build, "lframe_L_dp_m_default.pkl")):   # an A record from before the route went
        md = _load(build, "m_default")
        res["kernels"]["m_default"] = md["kernels"]
        res["A_m_post"] = post_frames(md, params)
        keys.insert(0, "A_m_post")
    res["B_own_post"] = post_frames(own, params)
    LR.write_l_record(res, json_path)
    print("model source:", json.dumps(res["model_source"]))
    for key in keys:
        r = res[key]
        print(key, "pass" if r["pass"] else "FAIL", "frames passed", r["frames_passed"], "of", len(r["frames"]),
              "(over-1 limit only:", r["frames_passed_over_1"], ") run:",
              json.dumps({k: (v["over_1"], v["values"], round(v["pct_over_1"], 5), v["pass"]) for k, v in r["run"].items()}),
              "worst", json.dumps(r["worst_frame"]))
        for row in r["frames"]:
            print("  frame", row["frame"], *[f"{k} {row[k]['over_1']} / {row[k]['values']} "
                                            f"{row[k]['pct_over_1']:.5f} % (max {row[k]['max_f16_steps']} ulp)"
                                            for k in ("history", "display") if k in row],
                  "reduced differ", row["reduced"]["differ"], "max", row["reduced"]["max_code_diff"],
                  "pass" if row["pass"] else "FAIL")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--compare", action="store_true")
    ap.add_argument("--config", choices=sorted(CONFIGS), help="--run: one configuration (in this process)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.run:
        if a.config:
            run_one(a.build, a.config)
            return 0
        for c in CONFIGS:  # one process per configuration
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--build", a.build, "--run", "--config", c])
            if r.returncode:
                return r.returncode
        return 0
    return compare(a.build, a.json or os.path.join(a.build, "l_default_post.json"))


if __name__ == "__main__":
    sys.exit(main())
