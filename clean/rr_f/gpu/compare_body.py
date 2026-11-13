# SPDX-License-Identifier: MIT
"""Byte comparison of two sequence body dumps (sequence.py --dump-body): every stage output of every frame, raw bytes
(e4m3 codes and f16 bit patterns, zero signs included), plus the variant-Y fixup counts of the first dump.

    python -m clean.rr_f.gpu.compare_body bodyY.npz bodyX.npz [--json out.json]"""
import argparse
import json
import sys

import numpy as np


def compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    keys = sorted(k for k in a.files if not k.endswith("_fixups"))
    rows, total, differ = {}, 0, 0
    for k in keys:
        if k not in b.files:
            rows[k] = "missing in second"
            differ += 1
            continue
        x, y = a[k], b[k]
        n = int((x.view(np.uint8) != y.view(np.uint8)).sum()) if x.shape == y.shape else -1
        total += x.size
        if n:
            differ += 1
            rows[k] = n
    fix = {k: int(a[k][0]) for k in a.files if k.endswith("_fixups")}
    return dict(arrays=len(keys), values=int(total), arrays_differing=differ, differing=rows,
                fixup_windows=sum(fix.values()), fixups_by_array={k: v for k, v in fix.items() if v})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = compare(a.a, a.b)
    print(f"{r['arrays']} arrays, {r['values']} values, {r['arrays_differing']} arrays differ; "
          f"fixup windows in the first run: {r['fixup_windows']}")
    for k, v in list(r["differing"].items())[:20]:
        print("  ", k, v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(r, f, indent=1)
    return 0 if r["arrays_differing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
