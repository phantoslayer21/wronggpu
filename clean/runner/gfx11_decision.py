# SPDX-License-Identifier: MIT
"""Evidence sheet for Model M on a gfx11 (RDNA 3) card, FP16 route: stages enc1 (C = 64) and b0 (C = 128, the hardest
shape), for the default gfx11 build and each variant of variants.FAMILY_VARIANTS["gfx11"]. Prints one markdown report.
It presents numbers; the output standard of an RDNA 3 build (same bytes as RDNA 4, exact, CLOSE, a new limit) is the
maintainer's decision and is not made here.

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime --variants f16w,f16wa,f16wq
    python clean/runner/gfx11_decision.py --build <build> --rocm <ROCm> --scratch <folder> --out sheet.md [--json raw.json]

Per stage and build it gives: the spec 09 measures against the float64 reference (clean/reference/swin.py, with the
sigmoid GELU the kernels use), the equivalence verdict word of clean/equiv/compare.py applied to the same pair, the
one-code flip fraction against the gfx11 emulator, the compiler's resource report, the median time of 50 launches at
the in-game render sizes, and a per-frame Swin time extrapolated from them (INFERRED). Synthetic weights and inputs
only (the timing does not depend on the values).
"""
import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from clean.equiv import compare as EQ  # noqa: E402
from clean.reference import fp8  # noqa: E402
from clean.reference import swin as S  # noqa: E402
from clean.runner import build as B  # noqa: E402
from clean.runner import family  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.tests.test_kernels_emu import spec09_metrics, ulp_distance  # noqa: E402

STAGES = ("enc1", "b0")
STAGE_KERNELS = {"enc1": ("swin_block_c64", "patch_merge_c64"), "b0": ("swin_block_c128",)}
# Stage sizes (tokens) at the render sizes of clean/README.md: enc1 at half the render size (padded to the window grid),
# b0 at a quarter of it (runner/sweep.py GAME_SIZES for the first; 1706x960 render -> 120x216 for b0).
RENDER_SIZES = {"1706x960": {"enc1": (480, 864), "b0": (120, 216)},
                "2560x1440": {"enc1": (720, 1280), "b0": (180, 320)}}
# Stage launches per frame (spec: enc1 once, b0..b5 six times, b1/b3/b5 with the shifted window offset, assumed to cost
# the same as b0) and the Swin stages this sheet does not measure.
STAGES_PER_FRAME = {"enc1": 1, "b0": 6}
UNMEASURED = ("enc2", "dec2", "dec1")
# Correctness cases: the sizes of the emulator tests (test_kernels_emu.TestStagesEmu.CASES for enc1, the same sizes and
# window offsets for b0). Several workgroups in every kernel at the last size.
CASES = {"enc1": [(8, 12, (2, 2)), (12, 20, (0, 2)), (20, 24, (2, 2)), (36, 32, (2, 2))],
         "b0": [(8, 8, (2, 2)), (12, 20, (0, 2)), (20, 24, (2, 2)), (36, 32, (2, 2))]}
# The comparator the plan names: Model M's Swin step on an RX 9070 XT, harness record of clean/l/SPEED.md (1440p Quality,
# render 1706x960). A different card, harness and kernel set (the whole Swin step: ten stages, bottleneck megakernel).
COMPARATOR = {"1706x960": 0.942}
SPEC09_LIMITS = (0.02, 0.01)   # at most 2 % more than one FP8 step off; mean abs error at most 1 % of the reference RMS


def spec09_pass(m):
    """Spec 09 for one output: no non-finite value, the two limits above. `m` is test_kernels_emu.spec09_metrics."""
    return (m["nonfinite"] == 0 and m["frac_over_1_step"] <= SPEC09_LIMITS[0]
            and m["mean_abs_err"] <= SPEC09_LIMITS[1] * m["ref_rms"])


def equivalence_verdict(got, ref, window=None, shift=None):
    """The verdict of clean/equiv/compare.py for one HWC stage output against the reference output, both given as FP8
    values. -> dict: word (MATCHES, CLOSE, DIFFERENT, or ERROR when the shapes differ), reasons, size (the tool's size
    word, only for DIFFERENT), and the numbers behind it: share equal, largest step, values over the CLOSE band (2
    steps) and over 16 steps, as shares."""
    st = EQ.compare_stage(fp8.fp8_encode(got), fp8.fp8_encode(ref), "fp8", "HWC", window=window, shift=shift)
    word, reasons = EQ.stage_verdict(st)
    out = {"word": word, "reasons": reasons, "size": EQ.size_word(st) if word == EQ.DIFFERENT else None}
    if word != EQ.ERROR:
        n = st["values"]
        out.update(equal=st["share_equal"], max_step=st["max_step"], over_band=st["over_band"] / n,
                   over_16=st["over_16"] / n)
    return out


def kernel_resources(text, kernels):
    """{kernel: resource fields} for `kernels` from the output of clang's -Rpass-analysis=kernel-resource-usage."""
    rep = B.resource_report(text)
    out = {}
    for k in kernels:
        r = rep.get(k)
        if r is None:
            raise KeyError(f"{k} is not in the resource report")
        out[k] = {"vgprs": r["VGPRs"], "spills": r["VGPRs Spill"], "waves": r["Occupancy [waves/SIMD]"],
                  "lds": r["LDS Size [bytes/block]"], "scratch": r["ScratchSize [bytes/lane]"]}
    return out


def extrapolate(ms):
    """Swin time per frame in ms from stage times {"enc1": ms, "b0": ms}. INFERRED. `measured` covers the stages
    timed here (enc1 once, b0 six times); `crude_all` adds enc2, dec2 and dec1 at the cost of one enc1 each (a
    placeholder: enc2 and dec2 run on a quarter of the tokens at C = 96, dec1 has enc1's shape)."""
    measured = sum(STAGES_PER_FRAME[s] * ms[s] for s in STAGES)
    return {"measured": measured, "crude_all": measured + len(UNMEASURED) * ms["enc1"]}


# ---- measurement (card, emulator, compiler) -----------------------------------------------------------------------------

def reference_and_emulator(emu_runner, stage, cases):
    """Per case: inputs, float64 reference output and gfx11 emulator output (shared by every build)."""
    out = []
    for i, (H, W, offset) in enumerate(cases):
        rng = np.random.default_rng(100 + i)
        p = S.random_params(stage, rng)
        x, skip = S.random_inputs(stage, H, W, rng)
        ref = S.run_stage(stage, p, x, offset, skip, gelu="sigmoid")
        emu = emu_runner.run(stage, emu_runner.pack_weights(stage, p), x, offset, skip)
        out.append({"H": H, "W": W, "offset": offset, "p": p, "x": x, "skip": skip, "ref": ref, "emu": emu})
    return out


def correctness(runner, stage, refs):
    """Spec 09 numbers, verdict and flips of one build against the shared references."""
    rows, flips, over, total = [], 0, 0, 0
    for c in refs:
        got = runner.run(stage, runner.pack_weights(stage, c["p"]), c["x"], c["offset"], c["skip"])
        for k in c["ref"]:
            m = spec09_metrics(got[k], c["ref"][k])
            window = shift = None
            if stage != "enc1" or k == "skip":  # block outputs sit on the window grid; the merged map does not
                ox, oy = c["offset"]
                window, shift = (S.WIN, S.WIN), (-oy, -ox)
            v = equivalence_verdict(got[k], c["ref"][k], window, shift)
            u = ulp_distance(got[k], c["emu"][k])
            total += u.size
            flips += int(np.sum(u > 0))
            over += int(np.sum(u > 1))
            rows.append({"H": c["H"], "W": c["W"], "output": k, "values": int(u.size), "metrics": m,
                         "spec09": spec09_pass(m), "verdict": v["word"], "reasons": v["reasons"], "size": v["size"],
                         "equal": v.get("equal"), "max_step": v.get("max_step"), "over_band": v.get("over_band"),
                         "over_16": v.get("over_16"),
                         "flip_per_million": float(np.mean(u > 0) * 1e6)})
    words = [r["verdict"] for r in rows]
    return {"cases": rows, "spec09": all(r["spec09"] for r in rows),
            "verdict": max(words, key=lambda w: EQ.RANK[w]),
            "flip_per_million": flips / total * 1e6, "over1_per_million": over / total * 1e6, "values": total}


def timing(runner, stage, repeats):
    """Median ms of `repeats` launches of the stage at each render size (inputs on the card already)."""
    out = {}
    for name, sizes in RENDER_SIZES.items():
        H, W = sizes[stage]
        rng = np.random.default_rng(5)
        p = S.random_params(stage, rng)
        x, skip = S.random_inputs(stage, H, W, rng)
        out[name] = runner.time(stage, runner.pack_weights(stage, p), x, S.window_offset(stage, 0), skip, repeats=repeats)
    return out


def compile_resources(rocm, build_dir, scratch, name):
    """Compile swin.hip as the build `name` with the resource report and return the report text (clang writes it to
    stderr; the code object goes to the scratch folder)."""
    fam = family.build_family(build_dir)
    with open(os.path.join(build_dir, family.TARGET_FILE)) as f:
        B.ARCH = json.load(f)["arch"]
    defines = {**B.FAMILY_DEFINES[fam]["swin.hip"], **({} if name == "base" else V.all_builds()[name])}
    cmd = B.device_command(B.find_clang(rocm), rocm, os.path.join(B.KERNEL_DIR, "swin.hip"),
                           os.path.join(scratch, f"resources__{name}.co"), report=True, defines=defines)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile of {name} failed:\n{r.stderr[-2000:]}")
    return r.stderr


def measure(build_dir, rocm, scratch, repeats=50, log=print):
    """Everything the sheet shows, as a dict (render_report turns it into markdown)."""
    from clean.runner.stage_runner import Runner
    if family.build_family(build_dir) != "gfx11":
        raise SystemExit("the build is not for a gfx11 target")
    builds = ["base", *V.FAMILY_VARIANTS["gfx11"]]
    emu_dir = os.path.join(scratch, "emu")
    os.makedirs(emu_dir, exist_ok=True)
    r = subprocess.run(B.emu_command(B.find_host_cxx(rocm), os.path.join(emu_dir, "wg_emu__gfx11.so"),
                                     {"WG_EMU_GFX11": 1}), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gfx11 emulator build failed:\n" + r.stderr[-2000:])
    emu = Runner(emu_dir, "", backend="emu", arch="gfx11")
    refs = {}
    for s in STAGES:
        log(f"reference and emulator: {s}")
        refs[s] = reference_and_emulator(emu, s, CASES[s])
    with open(os.path.join(build_dir, family.TARGET_FILE)) as f:
        res = {"device": None, "arch": json.load(f)["arch"], "repeats": repeats, "builds": {}}
    for name in builds:
        text = compile_resources(rocm, build_dir, scratch, name)
        runner = Runner(build_dir, rocm, variant=name)
        res["device"] = runner.dev.device_name
        entry = {"stages": {}}
        for s in STAGES:
            log(f"{name}: {s}")
            entry["stages"][s] = {"resources": kernel_resources(text, STAGE_KERNELS[s]),
                                  "correctness": correctness(runner, s, refs[s]),
                                  "time_ms": timing(runner, s, repeats)}
        entry["frame_ms"] = {size: extrapolate({s: entry["stages"][s]["time_ms"][size] for s in STAGES})
                             for size in RENDER_SIZES}
        res["builds"][name] = entry
    return res


# ---- the sheet -----------------------------------------------------------------------------------------------------------

def build_label(name):
    return "default" if name == "base" else name


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + \
           ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def render_report(res):
    """The markdown sheet from the dict of measure()."""
    builds = res["builds"]
    names = list(builds)
    L = [f"# Model M on {res['arch']} ({res['device']}): evidence sheet for the FP16 route, enc1 and b0", "",
         "Stages `enc1` (C = 64, then patch merge) and `b0` (C = 128, the hardest shape), default gfx11 build and its speed "
         f"variants ({', '.join(n for n in names if n != 'base')}). Synthetic weights and inputs. Labels: MEASURED comes "
         "from a run on this card or the compiler; INFERRED is arithmetic on measured numbers or an assumption. "
         "**This sheet presents evidence. The output standard for RDNA 3 (same bytes as RDNA 4, exact, CLOSE, a new limit) "
         "is the maintainer's decision.**", ""]
    L += ["## Correctness and distance to the emulator (MEASURED)", "",
          "Pooled over the cases listed under \"Cases\" (spec 09 columns: the worst case and output). Spec 09 passes when "
          "no value is non-finite, at most 2 % of the values are more than one FP8 step off the float64 reference, and the "
          "mean absolute error is at most 1 % of the reference RMS. Verdict: the worst word of any case and output "
          "(INFERRED use, see the notes). Flips: share of values that differ from the gfx11 emulator's exact result by "
          "one or more FP8 codes, per million.", ""]
    rows = []
    for s in STAGES:
        for n in names:
            c = builds[n]["stages"][s]["correctness"]
            ms = [r["metrics"] for r in c["cases"]]
            rows.append([s, build_label(n), sum(m["nonfinite"] for m in ms),
                         f"{max(m['frac_over_1_step'] for m in ms) * 100:.3f} %",
                         f"{max(m['mean_abs_err'] / m['ref_rms'] for m in ms) * 100:.4f} %",
                         "pass" if c["spec09"] else "**FAIL**", c["verdict"],
                         f"{c['flip_per_million']:.0f}", f"{c['over1_per_million']:.0f}"])
    L += table(["stage", "build", "non-finite", "over 1 step (worst)", "mean abs err / RMS (worst)", "spec 09",
                "verdict", "flips per million", "more than 1 step per million"], rows) + [""]
    failing = [(s, build_label(n)) for s in STAGES for n in names if not builds[n]["stages"][s]["correctness"]["spec09"]]
    if failing:
        L += ["**Spec 09 is not met on the card by:** " + ", ".join(f"{s} ({b})" for s, b in failing) +
              ". Recorded as evidence with the numbers above; no kernel was changed in this slice.", ""]
    else:
        L += ["Spec 09 is met by every listed stage and build on the card.", ""]
    L += ["## Resources (MEASURED, compiler report)", ""]
    rows = []
    for s in STAGES:
        for k in STAGE_KERNELS[s]:
            for n in names:
                r = builds[n]["stages"][s]["resources"][k]
                rows.append([s, k, build_label(n), r["vgprs"], r["spills"], r["scratch"], r["waves"], r["lds"]])
    L += table(["stage", "kernel", "build", "VGPRs", "VGPR spills", "scratch bytes/lane", "waves per SIMD",
                "LDS bytes"], rows) + ["", "## Time (MEASURED)", "",
                                       f"Median of {res['repeats']} launches of the stage's kernels, inputs on the card, ms.", ""]
    sizes = list(RENDER_SIZES)
    rows = []
    for s in STAGES:
        for n in names:
            rows.append([s, build_label(n)] + [f"{builds[n]['stages'][s]['time_ms'][z]:.3f}" for z in sizes])
    L += table(["stage", "build"] + [f"{z} render" for z in sizes], rows)
    L += ["", "Token grids at the render sizes: " + "; ".join(
        f"{z}: " + ", ".join(f"{s} {RENDER_SIZES[z][s][0]}x{RENDER_SIZES[z][s][1]}" for s in STAGES) for z in sizes) + ".", ""]
    L += ["## Swin time per frame, extrapolated (INFERRED)", "",
          f"Stage time x launches per frame: enc1 x {STAGES_PER_FRAME['enc1']}, b0 x {STAGES_PER_FRAME['b0']} "
          "(b1 to b5 assumed to cost what b0 costs; b1, b3 and b5 use the shifted window offset). "
          f"\"Measured stages\" is that sum alone: {', '.join(UNMEASURED)} are not in it. \"With the others at enc1's "
          "cost\" adds one enc1 time for each of them, a placeholder to give the order of magnitude (enc2 and dec2 run on "
          "a quarter of enc1's tokens at C = 96, so that over-counts them).", ""]
    rows = []
    for n in names:
        row = [build_label(n)]
        for z in sizes:
            f = builds[n]["frame_ms"][z]
            row += [f"{f['measured']:.2f}", f"{f['crude_all']:.2f}"]
        rows.append(row)
    L += table(["build"] + [f"{z} {c}" for z in sizes for c in ("measured stages (ms)", "with the others at enc1's cost (ms)")],
               rows)
    L += ["", "Comparator, a different card only: Model M's whole Swin step on an RX 9070 XT, " +
          "; ".join(f"{COMPARATOR[z]:.3f} ms at {z} render (1440p Quality)" for z in COMPARATOR) +
          " (clean/l/SPEED.md, harness record: Windows, Direct3D 12, per-kernel timestamps, all ten stages with the "
          "bottleneck megakernel). Not a target and not like for like with the rows above; the 2560x1440 render has no "
          "figure there.", ""]
    L += ["## Cases", ""]
    for s in STAGES:
        for n in names:
            L += [f"### {s}, {build_label(n)}", ""]
            rows = []
            for r in builds[n]["stages"][s]["correctness"]["cases"]:
                m = r["metrics"]
                rows.append([f"{r['H']}x{r['W']}", r["output"], r["values"], f"{m['frac_over_1_step'] * 100:.3f} %",
                             f"{m['mean_abs_err'] / m['ref_rms'] * 100:.4f} %", m["nonfinite"],
                             "pass" if r["spec09"] else "FAIL",
                             r["verdict"] + (f" ({r['size']})" if r["size"] else ""),
                             "-" if r["equal"] is None else f"{r['equal'] * 100:.3f} %",
                             "-" if r["max_step"] is None else r["max_step"],
                             "-" if r["over_band"] is None else f"{r['over_band'] * 100:.4f} %",
                             "-" if r["over_16"] is None else f"{r['over_16'] * 100:.4f} %",
                             f"{r['flip_per_million']:.0f}"])
            L += table(["tokens", "output", "values", "over 1 step", "mean abs err / RMS", "non-finite", "spec 09",
                        "verdict", "equal", "largest step", "over 2 steps", "over 16 steps", "flips per million"],
                       rows) + [""]
            reasons = sorted({x for r in builds[n]["stages"][s]["correctness"]["cases"] for x in r["reasons"]})
            if reasons:
                L += ["Reasons the tool gives: " + "; ".join(reasons) + ".", ""]
    L += ["## Notes", "",
          "- **Verdict word, applicability INFERRED.** clean/EQUIVALENCE-PROTOCOL.md defines MATCHES / CLOSE / DIFFERENT "
          "for our code against the original model, measured by the spec team on real weights. Applying the same rules to "
          "one stage output against our own float64 reference, on synthetic weights and inputs, is not that use: it "
          "borrows the thresholds as a ruler. Window shift and window size are passed so the tool's position-in-window "
          "classes work; the merged map of enc1 has no window structure.",
          "- **Random networks amplify rounding.** The synthetic weights do not give the flip rates of real frames (the "
          "analysis expects about 450 per million there; see clean/tests/test_gpu_enc1.py). The cases are small maps; "
          "timing uses the in-game sizes.",
          "- **The emulator is the exact per-WMMA model.** RDNA 3's F16 matrix instruction is inexact for mixed signs, so "
          "flips against the emulator are expected on this route; the speed variants are bit-identical to the default "
          "build in the emulator, so one emulator result serves all builds.",
          "- **The speed variants leave the C = 128 kernel unchanged** (swin.hip: its F16 fragments would put it over 256 "
          "registers and spill; WG_F16_A is for C < 96). The b0 rows of the variants are the default build's code, so equal "
          "numbers there are expected; time differences between them are measurement noise.",
          "- Resources are from clang 20 for gfx1100; another toolchain gives other counts.",
          "- The exact INT8 route (slices C1 to C3) is deferred; \"both routes\" in the plan means the default build and its "
          "speed variants here.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True, help="build folder for gfx1100 with --variants (build.py)")
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--scratch", required=True, help="folder for the emulator library and compiled code objects")
    ap.add_argument("--out", help="write the markdown here (default: print only)")
    ap.add_argument("--json", help="also write the raw results here")
    ap.add_argument("--repeats", type=int, default=50)
    a = ap.parse_args(argv)
    os.makedirs(a.scratch, exist_ok=True)
    res = measure(a.build, a.rocm, a.scratch, a.repeats, log=lambda m: print(m, file=sys.stderr, flush=True))
    text = render_report(res)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
