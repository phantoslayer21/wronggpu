# SPDX-License-Identifier: MIT
"""ISA statistics of compiled body kernels (RGA, gfx1201): VGPRs, SGPRs, LDS, scratch, and static instruction counts
(WMMA, VALU, VALU of the f16 / conversion kinds, SALU, VMEM, LDS, barriers, waitcnts), with a VGPR budget check.

    python -m clean.rr_f.gpu.isa_stats [--stages IN0,D0-F] [--step E,X] [--z f32] [--base] [--limit 128]

Static counts are per kernel text (loops counted once). Exit 1 when a kernel exceeds --limit VGPRs."""
import argparse
import re
import sys

from clean.rr_f.gpu import build, stage


def counts(txt):
    c = dict(wmma=0, valu=0, vopd=0, salu=0, vmem_ld=0, vmem_st=0, lds=0, barrier=0, waitcnt=0, cvt=0, pk16=0,
             permlane_dpp=0, readlane=0, total=0, scratch=0, fma_mix=0)
    for line in txt.splitlines():
        m = re.match(r"\s+([sv]_\w+|global_\w+|buffer_\w+|ds_\w+|scratch_\w+)", line)
        if not m:
            continue
        op = m.group(1)
        c["total"] += 1
        if op.startswith("v_wmma"):
            c["wmma"] += 1
        elif op.startswith("v_dual"):
            c["vopd"] += 1
            c["valu"] += 2
        elif op.startswith("v_"):
            c["valu"] += 1
            if op.startswith("v_fma_mix"):
                c["fma_mix"] += 1
            if "cvt" in op:
                c["cvt"] += 1
            if op.startswith("v_pk_") and "f16" in op:
                c["pk16"] += 1
            if "dpp" in line or "permlane" in op:
                c["permlane_dpp"] += 1
            if op.startswith("v_readlane") or op.startswith("v_readfirstlane"):
                c["readlane"] += 1
        elif op.startswith("s_barrier"):
            c["barrier"] += 1
        elif op.startswith("s_wait"):
            c["waitcnt"] += 1
        elif op.startswith("s_"):
            c["salu"] += 1
        elif op.startswith(("global_load", "buffer_load")):
            c["vmem_ld"] += 1
        elif op.startswith(("global_store", "buffer_store")):
            c["vmem_st"] += 1
        elif op.startswith("ds_"):
            c["lds"] += 1
        elif op.startswith("scratch_"):
            c["scratch"] += 1
    return c


def report(dxil, label, mix_check=True):
    st = build.isa_stats(dxil)
    c = counts(build.isa_text(dxil))
    keys = ("USED_VGPRs", "USED_SGPRs", "USED_LDS_BYTES", "USED_SCRATCH_BYTES")
    vals = {k: st.get(k, "?") for k in keys}
    print(f"{label:22s} vgpr {vals['USED_VGPRs']:>4} sgpr {vals['USED_SGPRs']:>4} lds {vals['USED_LDS_BYTES']:>6} "
          f"scratch {vals['USED_SCRATCH_BYTES']:>4} | wmma {c['wmma']:4d} valu {c['valu']:5d} (vopd {c['vopd']:4d}, "
          f"cvt {c['cvt']:4d}, pk16 {c['pk16']:4d}, dpp {c['permlane_dpp']:3d}) salu {c['salu']:4d} "
          f"vmem {c['vmem_ld']:3d}/{c['vmem_st']:3d} lds {c['lds']:4d} bar {c['barrier']:2d} wait {c['waitcnt']:4d} "
          f"total {c['total']:5d} fma_mix {c['fma_mix']}", flush=True)
    if c["fma_mix"] and mix_check:
        print("  WARNING: v_fma_mix in the ISA (an f16 rounding point folded away?)")
    try:
        return int(vals["USED_VGPRs"])
    except ValueError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="IN0,D0-F")
    ap.add_argument("--step", default="E,X")
    ap.add_argument("--z", default="f32")
    ap.add_argument("--base", action="store_true", help="also the base commit's kernels (body_ab's copy)")
    ap.add_argument("--limit", type=int, default=0, help="VGPR budget (0: no check)")
    ap.add_argument("--dyn", action="store_true", help="also the counting build of the specialised kernels "
                    "(RR_COUNT=1: fast paths only, loops unrolled = dynamic counts per wave)")
    # v_fma_mix check: step E kernels must have none (an f16 multiply / fma folded into a widening loses its rounding);
    # step X uses v_fma_mix_f32 legitimately (f16 start values widened inside its exact f32 sums).
    a = ap.parse_args(argv)
    bad = False
    for step in a.step.split(","):
        for name in a.stages.split(","):
            if a.base:
                from clean.rr_f.gpu import body_ab
                report(body_ab.base_kernel(body_ab.DEFAULT_BASE, name, a.z, step), f"base {name} {step}")
            v = report(stage.kernel_dxil(name, a.z, step), f"new  {name} {step}", step == "E")
            if a.dyn and stage.kernel_src(name, 1, step) != "rr_block.hlsl":
                d = stage.stage_defines(name, a.z, step, 1)
                d.update(stage.route_def(stage.route(name, 1, step))[1])
                if name == "E1" and stage.e1_direct():
                    d["RR_ADAPT"] = 2
                for cnt, lab in ((1, "dyn "), (2, "dynS")):
                    d["RR_COUNT"] = cnt
                    tag = f"cnt{cnt}_" + name.replace("-", "") + "_" + step
                    report(build.compile_kernel(stage.kernel_src(name, 1, step), "rr_block", d, tag=tag), f"{lab} {name} {step}")
            if a.limit and v is not None and v > a.limit:
                print(f"  VGPR budget exceeded: {v} > {a.limit}")
                bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
