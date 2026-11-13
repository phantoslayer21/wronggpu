# SPDX-License-Identifier: MIT
"""ISA statistics (RGA, gfx1201) of the output-stage and resolve kernels: VGPRs, LDS, scratch, static VALU / VOPD /
memory counts (isa_stats.counts), for the default build or extra defines.

    python -m clean.rr_f.gpu.outres_isa [--out K=V,...] [--res K=V,...] [--spec 0,1]
"""
import argparse

from clean.rr_f.gpu import isa_stats, output as GO, resolve_fast as GRF


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--res", default="")
    ap.add_argument("--spec", default="1")
    ap.add_argument("--which", default="out,res")
    a = ap.parse_args(argv)
    kv = lambda s: {k: int(v) if v.lstrip("-").isdigit() else v for k, v in (p.split("=") for p in s.split(",") if p)}
    if "out" in a.which:
        for n, (dx, _) in GO.kernel_set(kv(a.out)).items():
            isa_stats.report(dx, "output-" + n, mix_check=False)
    if "res" in a.which:
        for s in a.spec.split(","):
            isa_stats.report(GRF.fast_kernel(spec=int(s), extra=kv(a.res)), f"resolve-fast s{s}", mix_check=False)


if __name__ == "__main__":
    main()
