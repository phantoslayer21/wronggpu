# SPDX-License-Identifier: MIT
"""Bit-for-bit comparison of the default post + filter path of two builds on the same synthetic frames (spec 06).

    python clean/runner/post_exact.py --rocm <ROCm> --a-repo <checkout A> --a-build <build A> \\
        --b-repo <checkout B> --b-build <build B> [--configs all | --render WxH --output WxH] [--frames 5]

Each side runs `<repo>/clean/runner/post_sizes.py dump` (its own runtime_default, i.e. the kernels that checkout's
runtime launches by default, from its own build) in a subprocess, on the deterministic synthetic frames of
post_sizes.frame_inputs: frame sequence reset, normal, normal, reset, normal (repeated), exposures changing every frame,
the full history fed back from frame to frame. Compared per frame: SHA-256 of the canonical full history (FP16 bits),
reduced history (30-bit codes) and output (band path: f32 bits; copy path: RGBA16F bits), and of the inputs (both sides
must see the same inputs). Paths "band" and "copy" (post_sizes.py docstring). Exit status 0 only if everything is
identical; on a difference the first differing frame is dumped again with --save and the number of differing values
and the first differing position are printed.

Default configurations: the 15 standard ones of post_sizes.configs() plus three odd sizes (dynamic resolution,
non-multiple-of-32 output) unless --configs or --render/--output are given. Run under the GPU lock.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.runner.post_sizes import configs  # noqa: E402

# odd sizes: dynamic-resolution steps, odd D, an output width that is not a multiple of 32 (copy path: post_filter_h)
ODD = [("odd-1", (1333, 751), (2560, 1440)), ("odd-2", (997, 563), (1913, 1077)), ("odd-3", (64, 36), (101, 57)),
       ("odd-4", (1200, 700), (1000, 600))]   # R > 2/3 D: recon2_supported false, the post_recon fallback


def dump(python, repo, build, rocm, render, output, frames, paths, motion, seed, out, save=None, extra=()):
    cmd = [python, os.path.join(repo, "clean", "runner", "post_sizes.py"), "dump", "--build", build, "--rocm", rocm,
           "--render", render, "--output", output, "--frames", str(frames), "--paths", paths, "--motion", motion,
           "--seed", str(seed), "--out", out] + list(extra)
    if save:
        cmd += ["--save", save]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"dump failed ({repo}, {build}):\n{r.stdout}\n{r.stderr}")
    with open(out) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--a-repo", required=True)
    ap.add_argument("--a-build", required=True)
    ap.add_argument("--b-repo", required=True)
    ap.add_argument("--b-build", required=True)
    ap.add_argument("--configs", default="default", help="'default' (15 + odd), 'all15', or comma-separated labels")
    ap.add_argument("--render")
    ap.add_argument("--output")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--paths", default="band,copy")
    ap.add_argument("--motions", default="random,smooth")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--work", help="folder for the dumps (default: a temporary folder)")
    ap.add_argument("--json", help="write the comparison summary here")
    args = ap.parse_args(argv)
    python = sys.executable
    if args.render or args.output:
        cfgs = [("custom", tuple(map(int, args.render.split("x"))), tuple(map(int, args.output.split("x"))))]
    else:
        allc = configs() + ODD
        if args.configs == "default":
            cfgs = allc
        elif args.configs == "all15":
            cfgs = configs()
        else:
            want = args.configs.split(",")
            cfgs = [c for c in allc if c[0] in want]
            if len(cfgs) != len(want):
                raise SystemExit(f"unknown configuration in {want}; known: {[c[0] for c in allc]}")
    work = args.work or tempfile.mkdtemp(prefix="post_exact_")
    os.makedirs(work, exist_ok=True)
    summary, ok_all = [], True
    for label, (rw, rh), (ow, oh) in cfgs:
        for motion in args.motions.split(","):
            render, output = f"{rw}x{rh}", f"{ow}x{oh}"
            res = {}
            for side, repo, build in (("a", args.a_repo, args.a_build), ("b", args.b_repo, args.b_build)):
                res[side] = dump(python, repo, build, args.rocm, render, output, args.frames, args.paths, motion,
                                 args.seed, os.path.join(work, f"{label}_{motion}_{side}.json"))
            for path in args.paths.split(","):
                fa, fb = res["a"]["paths"][path]["frames"], res["b"]["paths"][path]["frames"]
                diffs = []
                for t, (x, y) in enumerate(zip(fa, fb)):
                    if x["inputs"] != y["inputs"]:
                        raise SystemExit(f"{label} {motion} frame {t}: the two checkouts generate different inputs "
                                         "(post_sizes.frame_inputs must not change)")
                    for k in ("history", "reduced", "out"):
                        if x[k] != y[k]:
                            diffs.append((t, k))
                row = {"config": label, "render": render, "output": output, "motion": motion, "path": path,
                       "a_kernels": res["a"]["paths"][path]["kernels"], "b_kernels": res["b"]["paths"][path]["kernels"],
                       "frames": len(fa), "identical": not diffs and len(fa) == len(fb)}
                if diffs:
                    ok_all = False
                    t0 = diffs[0][0]
                    detail = {}
                    sa, sb = os.path.join(work, f"{label}_{motion}_a"), os.path.join(work, f"{label}_{motion}_b")
                    for side, repo, build, sdir in (("a", args.a_repo, args.a_build, sa),
                                                    ("b", args.b_repo, args.b_build, sb)):
                        dump(python, repo, build, args.rocm, render, output, t0 + 1, path, motion, args.seed,
                             os.path.join(work, f"{label}_{motion}_{side}_save.json"), save=sdir)
                    for _, k in [d for d in diffs if d[0] == t0]:
                        a = np.load(os.path.join(sa, f"{path}_f{t0}_{k}.npy"))
                        b = np.load(os.path.join(sb, f"{path}_f{t0}_{k}.npy"))
                        ne = np.argwhere(a != b)
                        detail[k] = {"values_differing": int(len(ne)), "of": int(a.size),
                                     "first": ne[0].tolist() if len(ne) else None}
                    row["first_diff_frame"] = t0
                    row["diffs"] = [f"f{t}:{k}" for t, k in diffs]
                    row["detail"] = detail
                summary.append(row)
                print(json.dumps(row), flush=True)
    print("ALL IDENTICAL" if ok_all else "DIFFERENCES FOUND")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"identical": ok_all, "rows": summary}, f, indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
