# SPDX-License-Identifier: MIT
"""Summary of a run_all --json result: per stage kind and mode (Z mode / step variant / LDS staging), the number of
cases, how many are identical to the reference, and the differing values (and, for variant E, the differences from
its step model).

    python -m clean.rr_f.gpu.summary results.json"""
import json
import sys
from collections import OrderedDict


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    rows = json.load(open(argv[0]))
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r["stage"], r["z"]), []).append(r)
    print(f"{'stage':6s} {'mode':16s} {'cases':>5s} {'exact':>5s} {'values':>9s} {'differ':>7s} {'1-step':>7s} "
          f"{'vs E-model':>10s}  outputs differing")
    for (st, z), rs in groups.items():
        n = sum(v["n"] for r in rs for v in r["outputs"].values())
        d = sum(v["diff"] for r in rs for v in r["outputs"].values())
        one = sum(v.get("diff_1code", v.get("diff_1ulp", 0)) for r in rs for v in r["outputs"].values())
        ex = sum(1 for r in rs if r["exact"])
        vm = [sum(r["vs_model"].values()) for r in rs if "vs_model" in r]
        per = {}
        for r in rs:
            for k, v in r["outputs"].items():
                per[k] = per.get(k, 0) + v["diff"]
        print(f"{st:6s} {z:16s} {len(rs):5d} {ex:5d} {n:9d} {d:7d} {one:7d} "
              f"{(str(sum(vm)) if vm else '-'):>10s}  " + " ".join(f"{k}:{v}" for k, v in per.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
