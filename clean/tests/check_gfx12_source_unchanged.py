# SPDX-License-Identifier: MIT
"""Guard: a change for another GPU family must not change what the gfx12 kernels are compiled from.

A change that is meant to leave the gfx12 code objects alone is only guaranteed to do so when the preprocessed text of
every HIP kernel compile (clang -E for gfx1201, device side) is the same as the base tree's. "The same instructions" is
not enough: a rewritten expression, a hoisted store or an extra template parameter can give the same program, and a
different compiler (another ROCm, the Windows SDK) then schedules it differently (ROCm 10.1 / clang 24 moved
instructions in swin.co and swin__bmegahg.co for exactly such edits, while clang 20 did not). Guard the text, not one
compiler's output.

What is compared. The compiles are not listed here: each tree's own clean/runner/build.py is run with its compiler calls
recorded instead of executed (--model-l --variants all --perf-variants --ablations --profile, then --motion-outres-only),
so every shipped define and flag of every object (post_w64__q5.co's WG_LZ_PROFILE=1, the K and L kernels, the variants and
the megakernels, the wave64 flags, the include folders) is preprocessed exactly as built. FG's three objects (compiled by
clean/fg/runner) and every other clean/**/*.hip in its default form are added. The two trees' texts must match token for
token (white space ignored), form by form (keyed by the code object name).

A form only the tree builds (a variant the base does not name, e.g. swin__c128w7.co) is compared with the same compile against
the base's files; if it does not preprocess for the target at all (a gfx11-only variant: #error) it is no gfx12 object and is
skipped, and the run says so. Flags are kept as built, -O3 included (__OPTIMIZE__).

Fails (exit 1) on: a differing form; a form built by the base only, or by the tree only without a base source (--allow-new-forms); any
preprocessing failure on either side, even the same one on both (--allow-unbuildable names exceptions; the only automatic one is a
swin.hip variant that stops at an #error in both trees, which build.py cannot make for the target, and the run reports it).

Run: python clean/tests/check_gfx12_source_unchanged.py --rocm <ROCm> [--base origin/main | --base-tree <folder>]
The base is a git ref of this repository (exported to --scratch, default TMPDIR) or a checked-out tree. Exit 2: setup problem.
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
THIS = os.path.abspath(__file__)

BUILD_RUNS = (["--variants", "all", "--perf-variants", "--ablations", "--profile"], ["--motion-outres-only"])
DROP_FLAGS = ("--no-gpu-bundle-output",)  # -O3 stays: it defines __OPTIMIZE__, which source text may depend on
DROP_PREFIX = ("-cuid=", "-Rpass-analysis=")
FG_SOURCES = ("fg_net1_mma.hip", "fg_net1.hip", "fg_probe.hip")


def find_clang(rocm):
    for rel in ("bin/clang", "lib/llvm/bin/clang", "llvm/bin/clang"):
        p = os.path.join(rocm, rel)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"clang not found under {rocm}")


def collect(tree, rocm, arch, out):
    """Child process: run `tree`'s build.py with the compiler calls recorded; write the commands to out + '.json'."""
    os.environ["WG_GPU_ARCH"] = arch
    sys.path.insert(0, tree)
    os.chdir(tree)
    from clean.runner import build as B
    rec = []

    def run(cmd):
        rec.append(list(cmd))
        o = cmd[cmd.index("-o") + 1]
        os.makedirs(os.path.dirname(o), exist_ok=True)
        open(o, "wb").close()  # later steps copy or read the object

    B.run = run
    B.run_post_w64 = lambda cmd, *a, **k: run(cmd)
    for extra in BUILD_RUNS:
        try:
            B.main(["--rocm", rocm, "--out", out, "--no-runtime", "--no-resource-gate", "--model-l"] + extra)
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
    from clean.runner import variants as V
    names = {V.code_object(n) for n in V.all_builds()}  # the objects `--variants` / `--ablations` / `--profile` ask for
    with open(out + ".json", "w") as f:
        json.dump({"calls": rec, "variants": sorted(names)}, f)


def build_forms(tree, rocm, arch, scratch):
    """{form key: preprocess command} of a tree: its recorded build calls, FG's objects, then every other source."""
    out = os.path.join(scratch, "rec-" + re.sub(r"\W", "_", tree))
    r = subprocess.run([sys.executable, "-I", THIS, "--collect", tree, "--rocm", rocm, "--arch", arch, "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"cannot record the build calls of {tree}: {r.stderr.strip()[-600:]}")
    with open(out + ".json") as f:
        rec = json.load(f)
    calls = rec["calls"]
    clang = find_clang(rocm)
    forms, seen_src = {}, set()
    for cmd in calls:
        name = os.path.basename(cmd[cmd.index("-o") + 1])
        key, n = name, 1
        while key in forms:
            n += 1
            key = f"{name}#{n}"
        i = cmd.index("-o")
        pre = [c for c in cmd[:i] + cmd[i + 2:] if c not in DROP_FLAGS and not c.startswith(DROP_PREFIX)]
        forms[key] = pre[:1] + ["-E", "-P"] + pre[1:]
        seen_src.add(os.path.realpath(pre[-1]))
    base = [clang, "-x", "hip", f"--offload-arch={arch}", "--cuda-device-only", "-nogpuinc",
            "-O3", f"--rocm-path={rocm}", f"--rocm-device-lib-path={os.path.join(rocm, 'amdgcn', 'bitcode')}", "-E", "-P"]
    kdir = os.path.join(tree, "clean", "kernels")
    for s in FG_SOURCES:  # clean/fg/runner compiles these with the Model M kernel folder as include folder
        p = os.path.join(tree, "clean", "fg", "kernels", s)
        if os.path.isfile(p):
            forms["fg:" + s] = base + ["-I", kdir, p]
            seen_src.add(os.path.realpath(p))
    for dirpath, _, names in sorted(os.walk(os.path.join(tree, "clean"))):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            if n.endswith(".hip") and os.path.realpath(p) not in seen_src:
                incs = [x for d in ("kernels", "k/kernels", "l/kernels", "fg/kernels")
                        for x in ("-I", os.path.join(tree, "clean", d))]
                forms["default:" + os.path.relpath(p, os.path.join(tree, "clean"))] = base + incs + [p]
    return forms


def preprocess(cmd):
    """(normalized text, None) or (None, error lines)."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, [l for l in r.stderr.splitlines() if "error" in l][:3] or r.stderr.splitlines()[:3]
    return re.sub(r"\s+", " ", r.stdout).strip(), None


# The #error that swin.hip puts in front of its gfx11-only knobs (WG_F16_*, WG_SCHED_FRAGS, WG_MERGE_REUSE). A form whose first
# error is this one is a gfx11 variant, not an object a gfx12 build makes; every other preprocessing failure fails the guard.
GFX11_ONLY = re.compile(r'error: ".*\b(is|are) for the gfx11 matrix wrappers')


def gfx11_only(errs):
    return bool(errs) and GFX11_ONLY.search(errs[0]) is not None


def text_diff(a, b, limit=12):
    ta, tb = re.split(r"(?<=[;{}]) ", a), re.split(r"(?<=[;{}]) ", b)
    d = [l for l in difflib.unified_diff(ta, tb, "base", "tree", n=0, lineterm="") if not l.startswith("@@")]
    return d[:limit]


def compare(base_forms, tree_forms, allow_new=(), allow_unbuildable=(), pre=preprocess, report=print, pair=None):
    """Returns (forms preprocessed, number of failures).

    A form only the tree builds (a variant the base does not name) is not a difference by itself: `pair(cmd)` gives the same
    compile against the base's files (None if the base lacks the source). If the tree's form stops at the gfx11-only #error
    (GFX11_ONLY), it is no gfx12 object and is skipped (reported); any other failure fails; otherwise it is compared with its
    pair, so an edit under a macro only such a variant sets is caught. Without a pair it fails unless allowed."""
    bad = checked = 0
    for key in sorted(set(base_forms) - set(tree_forms)):
        report(f"FAIL {key}: built by the base only")
        bad += 1
    pairs = {}
    for key in sorted(set(tree_forms) - set(base_forms)):
        if key in allow_new:
            continue
        pc = pair(tree_forms[key]) if pair else None
        if pc is None:
            report(f"FAIL {key}: built by the tree only")
            bad += 1
            continue
        b, eb = pre(tree_forms[key])
        if b is None:
            if gfx11_only(eb):
                report(f"skip {key}: new in the tree, stops at the gfx11-only #error (not a gfx12 object): {eb[0][:160]}")
            else:
                report(f"FAIL {key}: new in the tree and does not preprocess for the target: {eb[0][:200]}")
                bad += 1
            continue
        pairs[key] = pc
    for key in sorted(set(base_forms) & set(tree_forms)) + sorted(pairs):
        bcmd = base_forms.get(key) or pairs[key]
        a, ea = pre(bcmd)
        b, eb = pre(tree_forms[key])
        if a is None or b is None:
            if key in allow_unbuildable and a is None and b is None:
                continue
            if a is None and b is None and gfx11_only(ea) and gfx11_only(eb):
                # stops at the gfx11-only #error in both trees: not an object build.py can make for this target
                report(f"skip {key}: stops at the gfx11-only #error in both trees: {ea[0][:160]}")
                continue
            side = " and ".join(s for s, e in (("base", ea), ("tree", eb)) if e)
            report(f"FAIL {key}: does not preprocess in {side}: {(ea or eb)[0][:200]}")
            bad += 1
            continue
        checked += 1
        if a != b:
            report(f"FAIL {key}: gfx12 preprocessed text differs" + (" (paired with the base's files)" if key in pairs else ""))
            for l in text_diff(a, b):
                report("    " + l[:200])
            bad += 1
    return checked, bad


def pair_in(base, tree):
    """The compile `cmd` of the tree against the base's files (same relative paths), or None if the source is missing there."""
    def pair(cmd):
        out = [c.replace(tree, base) if tree in c else c for c in cmd]
        src = [c for c in out if c.endswith(".hip")]
        return out if src and os.path.isfile(src[-1]) else None
    return pair


def export_base(ref, dest):
    os.makedirs(dest, exist_ok=True)
    tar = subprocess.run(["git", "-C", ROOT, "archive", ref, "clean"], capture_output=True)
    if tar.returncode != 0:
        raise SystemExit(f"git archive {ref} failed: {tar.stderr.decode().strip()}")
    r = subprocess.run(["tar", "-x", "-C", dest], input=tar.stdout, capture_output=True)
    if r.returncode != 0:
        raise SystemExit("tar failed: " + r.stderr.decode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--base", default="origin/main", help="git ref of the base")
    ap.add_argument("--base-tree", help="a checked-out base tree instead of --base")
    ap.add_argument("--tree", default=ROOT, help="the tree to check (default: this repository)")
    ap.add_argument("--arch", default="gfx1201")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR"), help="working folder (created if missing)")
    ap.add_argument("--allow-new-forms", nargs="*", default=[], help="form keys (code object names, default:<source>) new in the tree")
    ap.add_argument("--allow-unbuildable", nargs="*", default=[], help="form keys that do not preprocess for the target in either tree")
    ap.add_argument("--collect", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.collect:
        collect(os.path.abspath(args.collect), args.rocm, args.arch, args.out)
        return 0
    if args.scratch:
        os.makedirs(args.scratch, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="gfx12-guard-", dir=args.scratch)
    base = os.path.abspath(args.base_tree) if args.base_tree else os.path.join(scratch, "base")
    if not args.base_tree:
        export_base(args.base, base)
    tree = os.path.abspath(args.tree)
    bf = build_forms(base, args.rocm, args.arch, scratch)
    tf = build_forms(tree, args.rocm, args.arch, scratch)
    checked, bad = compare(bf, tf, set(args.allow_new_forms), set(args.allow_unbuildable), pair=pair_in(base, tree))
    print(f"{checked} forms preprocessed for {args.arch} ({len(bf)} in the base, {len(tf)} in the tree); {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
