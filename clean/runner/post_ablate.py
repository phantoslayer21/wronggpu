# SPDX-License-Identifier: MIT
"""Where does the time of the spec 06 kernels go? Timing-only ablation builds of kernels/post.hip (synthetic data;
the outputs of the ablation builds are wrong on purpose).

    python clean/runner/post_ablate.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--json out.json]

Builds build/post__<name>.co for every entry of ABLATIONS (the WG_AB_* switches at the top of kernels/post.hip) and
prints, per build, the median ms at the in-game size (R 1706x960 -> D 3840x2160 -> O 2560x1440) of post_recon,
post_recon2, post_recon2_r, post_recon3, post_recon5, post_recon6, post_recon6_eb (normal and reset frame; the last
three with their pre-pass), post_filter, post_filter_c1 and post_fused (normal and
reset frame; --kernels selects a subset), the difference to the base build
(what the removed part costs), and each kernel's VGPRs and occupancy from the compiler. --rounds N repeats the whole
sweep N times in interleaved order and reports the median per build (single sweeps vary by a few percent).
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.runner import build as B  # noqa: E402
from clean.runner import post as PS  # noqa: E402
from clean.runner.post_bench import big_inputs  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402

SKELETON = ["WG_AB_GATHER", "WG_AB_EXP", "WG_AB_STAGE", "WG_AB_MOTION", "WG_AB_SCORE", "WG_AB_HIST", "WG_AB_TRANS"]
ABLATIONS = {
    "base": [],
    "ab_gather": ["WG_AB_GATHER"],      # centre render sample only (1 of 9 taps)
    "ab_exp": ["WG_AB_EXP"],            # no exp in the tap weights
    "ab_stage": ["WG_AB_STAGE"],        # no global loads / E_in logs in the LDS staging
    "ab_motion": ["WG_AB_MOTION"],      # no motion reads (fixed candidate offsets)
    "ab_score": ["WG_AB_SCORE"],        # no candidate scoring
    "ab_hist": ["WG_AB_HIST"],          # no history taps
    "ab_trans": ["WG_AB_TRANS"],        # E() and Dec() linear (no log / exp outside the gather)
    "ab_histw": ["WG_AB_HISTW"],        # no full-history write
    "ab_reduced": ["WG_AB_REDUCED"],    # no reduced history
    "ab_display": ["WG_AB_DISPLAY"],    # no display write
    "ab_fclamp": ["WG_AB_FCLAMP"],      # filter without min/max clamp
    "ab_fdec": ["WG_AB_FDEC"],          # filter without Dec
    "ab_codes": ["WG_AB_CODES"],        # gather: W and colour only, not the six per-sample codes
    "ab_enc": ["WG_AB_ENC"],            # E() linear (no log)
    "ab_dec": ["WG_AB_DEC"],            # Dec() linear (no exp)
    "ab_blend": ["WG_AB_BLEND"],        # no step 5 (Dec, blend, NaN guard)
    "ab_head": ["WG_AB_MOTION", "WG_AB_SCORE"],                    # no motion reads, no candidate scoring
    "ab_stores": ["WG_AB_HISTW", "WG_AB_REDUCED", "WG_AB_DISPLAY"],  # no writes at all
    "ab_skeleton": SKELETON,            # only the memory traffic and the loop skeleton remain
}
KERNELS = ["post_recon", "post_recon2", "post_recon2_r", "post_recon3", "post_recon5", "post_recon6", "post_recon6_eb",
           "post_filter", "post_filter_c1", "post_fused"]


def code_object(name):
    return PS.CODE_OBJECT if name == "base" else f"post__{name}.co"


def build_all(build_dir, rocm, clang, names):
    """Build the ablation code objects; returns {name: {kernel: {"vgprs", "occupancy"}}} from the compiler."""
    clang = clang or B.find_clang(rocm)
    res = {}
    for name in names:
        out = os.path.join(build_dir, code_object(name))
        cmd = B.device_command(clang, rocm, os.path.join(B.KERNEL_DIR, "post.hip"), out, device_libs=bool(rocm),
                               report=True, defines={d: 1 for d in ABLATIONS[name]})
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"build of {name} failed:\n{r.stderr}")
        info, cur = {}, None
        for line in r.stderr.splitlines():
            m = re.search(r"Function Name: (\w+)", line)
            if m:
                cur = m.group(1)
                info[cur] = {}
            m = re.search(r"(VGPRs|Occupancy \[waves/SIMD\]): (\d+)", line)
            if m and cur:
                info[cur]["vgprs" if m.group(1) == "VGPRs" else "occupancy"] = int(m.group(2))
        res[name] = {k: info.get(k, {}) for k in KERNELS}
    return res


def time_build(r, co, w, x_norm, x_reset, disp, repeats, keys=None):
    """Median ms per timing key (kernel, or kernel + "_reset" for the reset frame) for one build."""
    post = lambda k, x: (lambda: PS.time_plans(r, [PS.prepare_post(r.dev, w, x, k)], co, repeats))  # noqa: E731
    filt = lambda k: (lambda: PS.time_plans(r, [PS.prepare_filter(r.dev, b"", disp, k)], co, repeats))  # noqa: E731
    jobs = {}
    for k in ("post_recon", "post_recon2", "post_recon2_r", "post_recon3", "post_recon5", "post_recon6",
              "post_recon6_eb"):   # post_recon5 / 6*: pre-pass included
        jobs[k] = post(k, x_norm)
        jobs[k + "_reset"] = post(k, x_reset)
    jobs["post_filter"] = filt("post_filter")
    jobs["post_filter_c1"] = filt("post_filter_c1")
    jobs["post_fused"] = lambda: PS.time_fused(r, w, x_norm, PS.KERNEL_FUSED, repeats, co)
    jobs["post_fused_reset"] = lambda: PS.time_fused(r, w, x_reset, PS.KERNEL_FUSED, repeats, co)
    return {k: job() for k, job in jobs.items() if keys is None or k in keys}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--clang", help="clang (default: from --rocm)")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--only", help="comma-separated build names (default: all)")
    ap.add_argument("--kernels", help="comma-separated timing keys to measure (default: all), e.g. "
                                      "post_recon3,post_recon3_reset")
    ap.add_argument("--json", help="also write the results to this file")
    ap.add_argument("--build-only", action="store_true", help="build and print the compiler resource report only")
    args = ap.parse_args(argv)
    names = args.only.split(",") if args.only else list(ABLATIONS)
    if "base" not in names:
        names = ["base"] + names
    resources = build_all(args.build, args.rocm, args.clang, names)
    report = {"resources": resources, "timing_ms": {}}
    for n in names:
        print(n, json.dumps(resources[n]))
    if not args.build_only:
        r = Runner(args.build, args.rocm)
        report["device"] = r.dev.device_name
        rng = np.random.default_rng(1)
        x_norm, disp = big_inputs(rng, False)
        x_reset, _ = big_inputs(rng, True)
        x_norm = dict(x_norm, output_size=disp["output_size"])
        x_reset = dict(x_reset, output_size=disp["output_size"])
        w = r.pack_weights("post", {"s": np.array(0.9)})
        runs = {n: [] for n in names}
        for _ in range(args.rounds):
            for n in names:
                runs[n].append(time_build(r, code_object(n), w, x_norm, x_reset, disp, args.repeats,
                                          args.kernels.split(",") if args.kernels else None))
        base = {k: float(np.median([t[k] for t in runs["base"]])) for k in runs["base"][0]}
        print(f"{'build':12s} " + " ".join(f"{k:>18s}" for k in base))
        for n in names:
            med = {k: float(np.median([t[k] for t in runs[n]])) for k in base}
            report["timing_ms"][n] = {"median": med, "removed_vs_base": {k: base[k] - med[k] for k in base}}
            print(f"{n:12s} " + " ".join(f"{med[k]:7.3f} ({base[k] - med[k]:+.3f})" for k in base))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
