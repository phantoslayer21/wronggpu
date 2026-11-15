# SPDX-License-Identifier: MIT
"""Resource gates of the Model M Swin block kernels in the gfx11 (RDNA 3) builds: vector registers, spills and waves per SIMD
of every kernel of gfx11_gates.json, in the default gfx11 build and in each variant of variants.FAMILY_VARIANTS["gfx11"], and the
bottleneck megakernel of the persistent-kernel build (variants.FAMILY_MEGA["gfx11"], compiled from kernels/swin_v2.hip).
This table belongs to the gfx11 (HIP) builds; clean/dx12/vgpr_limits.json belongs to the Direct3D 12 ports.

A gate says: the kernel spills nothing, uses at most `vgprs` registers and keeps at least `waves` waves per SIMD. The numbers
are the compiler's own report (-Rpass-analysis=kernel-resource-usage) of clean/kernels/swin.hip (swin_v2.hip for the persistent
build) for gfx1100 with the defines of build.py; the compiler that recorded them is in the table ("clang"), and a different compiler may need other numbers.

    python clean/runner/gfx11_gates.py --rocm <ROCm> --scratch <folder>            # check the table against the compiler
    python clean/runner/gfx11_gates.py --rocm <ROCm> --scratch <folder> --write    # record the current numbers as the table
    python clean/runner/gfx11_gates.py --rocm <ROCm> --scratch <folder> --write --only bmegahg   # one build: the others stay
"""
import argparse
import json
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clean.runner import build as B  # noqa: E402
from clean.runner import variants as V  # noqa: E402

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfx11_gates.json")
TARGET = "gfx1100"
# The kernels the table gates: the block kernels (and their clamp-free `_nc` forms, which the runner launches for weights that
# prove the packs in range) of every shape, the `_add` form and the weight-resident `_r` kernels, and the patch merge and
# expand kernels. The fused kernels (`_mg`, `_ex`) are not gated yet: some of them spill in the gfx11 builds (slice E2).
KERNELS = [f"{k}{nc}" for k in ("swin_block_c64", "swin_block_c64_add", "swin_block_c64_r", "swin_block_c64_add_r",
                                "swin_block_c96", "swin_block_c96_add", "swin_block_c128") for nc in ("", "_nc")]
KERNELS += ["patch_merge_c64", "patch_merge_c96", "expand_c96", "expand_c128"]
# The persistent-kernel build: swin_bneck6 (b0 to b5 in one launch) and its clamp-free form. swin_mid (enc2, b0 to b5, dec2 in one
# launch) is not gated yet: its two C = 96 phases spill about 70 registers each in the gfx11 build (slice E3).
MEGA_KERNELS = ["swin_bneck6", "swin_bneck6_nc"]


def builds():
    return ["base", *V.FAMILY_VARIANTS["gfx11"], *V.FAMILY_MEGA["gfx11"]]


def kernels_of(name):
    return MEGA_KERNELS if name in V.FAMILY_MEGA["gfx11"] else KERNELS


def compile_report(rocm, scratch, name):
    """The resource report {kernel: {field: int}} of the build's source (swin.hip, swin_v2.hip) compiled for gfx1100, and the compiler's
    version line. The code object goes to `scratch`."""
    defines = {**B.FAMILY_DEFINES["gfx11"]["swin.hip"], **({} if name == "base" else V.all_builds()[name])}
    clang = B.find_clang(rocm)
    saved, B.ARCH = B.ARCH, TARGET
    try:
        cmd = B.device_command(clang, rocm, os.path.join(B.KERNEL_DIR, V.source(name)),
                               os.path.join(scratch, f"gates__{name}.co"), report=True, defines=defines)
    finally:
        B.ARCH = saved
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile of {name} failed:\n{r.stderr[-2000:]}")
    return B.resource_report(r.stderr), clang_version(clang)


def clang_version(clang):
    out = subprocess.run([clang, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"clang version (\S+)", out)
    return m.group(1) if m else "unknown"


def measure(rocm, scratch, names=None):
    """-> ({build: {kernel: {"vgprs", "spills", "waves", "lds"}}}, clang version) for the gated kernels."""
    out, version = {}, None
    for name in names or builds():
        rep, version = compile_report(rocm, scratch, name)
        missing = [k for k in kernels_of(name) if k not in rep]
        if missing:
            raise KeyError(f"{name}: not in the resource report: {', '.join(missing)}")
        out[name] = {k: {"vgprs": rep[k]["VGPRs"], "spills": rep[k]["VGPRs Spill"],
                         "waves": rep[k]["Occupancy [waves/SIMD]"], "lds": rep[k]["LDS Size [bytes/block]"]}
                     for k in kernels_of(name)}
    return out, version


def load(path=TABLE):
    with open(path) as f:
        return json.load(f)


def check(measured, table):
    """The ways `measured` ({build: {kernel: numbers}}) breaks `table` (the loaded JSON): a list of sentences, empty when every
    gate holds. A gate is broken by any spill, by more registers than recorded, by fewer waves per SIMD than recorded, and by a
    gated kernel or build that is missing from the measurement."""
    bad = []
    for name, kernels in table["gates"].items():
        if name not in measured:
            bad.append(f"{name}: not measured")
            continue
        for kernel, gate in kernels.items():
            m = measured[name].get(kernel)
            if m is None:
                bad.append(f"{name} {kernel}: not measured")
                continue
            if m["spills"] > 0:
                bad.append(f"{name} {kernel}: spills {m['spills']} registers (gate: none)")
            if m["vgprs"] > gate["vgprs"]:
                bad.append(f"{name} {kernel}: {m['vgprs']} registers (gate: at most {gate['vgprs']})")
            if m["waves"] < gate["waves"]:
                bad.append(f"{name} {kernel}: {m['waves']} waves per SIMD (gate: at least {gate['waves']})")
    return bad


def record(measured, version):
    """The table for `measured`: every gated kernel must spill nothing (a spilling kernel is not recorded, it is a finding)."""
    gates = {}
    for name, kernels in measured.items():
        gates[name] = {}
        for k, m in kernels.items():
            if m["spills"]:
                raise SystemExit(f"{name} {k} spills {m['spills']} registers: not recorded as a gate")
            gates[name][k] = {"vgprs": m["vgprs"], "waves": m["waves"], "lds": m["lds"]}
    return {"target": TARGET, "clang": version, "gates": gates}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--scratch", required=True, help="folder for the code objects of the compiles")
    ap.add_argument("--write", action="store_true", help="record the measured numbers as the table")
    ap.add_argument("--only", help="measure (and with --write record) this build only; the other recorded builds are kept")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    measured, version = measure(args.rocm, args.scratch, [args.only] if args.only else None)
    for name, kernels in measured.items():
        for k, m in kernels.items():
            print(f"{name:7s} {k:26s} {m['vgprs']:3d} VGPRs  {m['spills']:3d} spills  {m['waves']} waves/SIMD  LDS {m['lds']}")
    if args.write:
        table = record(measured, version)  # before the file is opened: a kernel that spills must not leave an empty table
        if args.only:
            old = load()
            if old["clang"] != version:
                raise SystemExit(f"--only keeps the recorded builds, made with clang {old['clang']}: this is {version}")
            table["gates"] = {**old["gates"], **table["gates"]}
        with open(TABLE, "w") as f:
            json.dump(table, f, indent=1)
            f.write("\n")
        print(f"recorded in {TABLE} (clang {version})")
        return
    table = load()
    if args.only:
        table = {**table, "gates": {args.only: table["gates"][args.only]}}
    bad = check(measured, table)
    print("\n".join(bad) if bad else f"every gate holds (clang {version}; recorded with {load()['clang']})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
