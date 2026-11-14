# SPDX-License-Identifier: MIT
"""Model M's code objects, base versus head: proof that a change to a shared kernel source (clean/kernels/*.hip, e.g.
a template another model reuses) leaves every Model M code object byte for byte unchanged.

    python clean/runner/m_codeobjects_compare.py --base <git ref> --work <dir> [--rocm <HIP SDK>] [--json out.json]
    python clean/runner/m_codeobjects_compare.py --base-tree <checkout> --work <dir> ...

Both trees are built with clean/runner/build.py --no-runtime --variants (the default code objects and every variant)
into <dir>/base and <dir>/head, each under the same output names: the build's -cuid comes from the output file name,
so only objects built under the same name are comparable. Every .co is then compared by SHA-256.

--base <ref> checks the ref out as a detached git worktree at <dir>/base-tree (created once and reused; this script
never deletes it). --base-tree uses an existing checkout instead. The head is this script's own tree. Run it for every
PR that touches a kernel source Model M builds, and put the result line in the PR description. Exit code 0: all
identical.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from clean.runner.hip import DEFAULT_ROCM, hip_child_environment  # noqa: E402

NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def hashes(d):
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".co"):
                p = os.path.join(root, f)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, d).replace(os.sep, "/")] = hashlib.sha256(fh.read()).hexdigest()
    return out


def build(tree, out, rocm, env):
    r = subprocess.run([sys.executable, "-B", os.path.join("clean", "runner", "build.py"), "--rocm", rocm, "--out", out,
                        "--no-runtime", "--variants"], cwd=tree, env=env, capture_output=True, text=True,
                       creationflags=NO_WINDOW)
    if r.returncode:
        sys.stderr.write((r.stdout + r.stderr)[-4000:])
        raise SystemExit(f"build failed in {tree} (exit {r.returncode})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="git ref of the base (checked out as a worktree under --work)")
    g.add_argument("--base-tree", help="an existing checkout of the base")
    ap.add_argument("--work", required=True, help="folder for the base worktree and both build outputs")
    ap.add_argument("--rocm", default=DEFAULT_ROCM)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    base_tree = a.base_tree
    if a.base:
        base_tree = os.path.join(a.work, "base-tree")
        if not os.path.isdir(base_tree):
            subprocess.run(["git", "-C", ROOT, "worktree", "add", "--detach", base_tree, a.base], check=True,
                           creationflags=NO_WINDOW)
        else:
            subprocess.run(["git", "-C", base_tree, "checkout", "--detach", a.base], check=True,
                           creationflags=NO_WINDOW)
    env, _ = hip_child_environment(None, a.rocm)
    env["TMP"] = env["TEMP"] = os.path.join(a.work, "tmp")
    os.makedirs(env["TMP"], exist_ok=True)
    outs = {"base": os.path.join(a.work, "base"), "head": os.path.join(a.work, "head")}
    build(base_tree, outs["base"], a.rocm, env)
    build(ROOT, outs["head"], a.rocm, env)
    hb, hh = hashes(outs["base"]), hashes(outs["head"])
    differ = sorted(k for k in set(hb) | set(hh) if hb.get(k) != hh.get(k))
    res = {"base": a.base or base_tree, "base_objects": len(hb), "head_objects": len(hh), "differ": differ}
    print("Model M code objects: %d base / %d head; %s" % (len(hb), len(hh),
                                                         "all identical" if not differ else "DIFFER: %s" % differ))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if not differ and hb else 1


if __name__ == "__main__":
    sys.exit(main())
