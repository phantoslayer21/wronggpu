# SPDX-License-Identifier: MIT
"""Re-runs the reference on every stored spec 51 vector case (read back from the binary files) and checks that the
stored expected outputs, row sums and traces are reproduced; also checks the scalar tables. Exit code 0 = all equal.

Doubles as an example of how a kernel test reads the vectors (vectors_io.load_case) and compares with the stated
tolerance (`compare_e4m3`, `compare_f16`).

Run (from the repository root):  python -m clean.rr_f.tests.check_vectors [--dir DIR]"""
import argparse
import os
import sys

import numpy as np

from clean.rr_f.reference.fmt import ALL_F16, D8_TABLE, q8, to_f16_bits
from clean.rr_f.reference.scalar import activation, attention_weight, recip_table, sig_tables
from clean.rr_f.reference.stages import run_stage
from clean.rr_f.tests.gen_vectors import file_list, sha256_file
from clean.rr_f.tests.vectors_io import DEFAULT_DIR, load_case, load_manifest, read_array, window_consts


def compare_e4m3(got, ref, zero_codes_equal=True):
    """Number of differing codes (NaN codes compare by code; 0x00 == 0x80 if zero_codes_equal)."""
    g = np.asarray(got, dtype=np.uint8).copy()
    r = np.asarray(ref, dtype=np.uint8).copy()
    if zero_codes_equal:
        g[g == 0x80] = 0
        r[r == 0x80] = 0
    return int((g != r).sum())


def compare_f16(got, ref, zero_signs_equal=True, mask=None):
    """Number of differing f16 values (bit compare; NaN == NaN; +0 == -0 if zero_signs_equal); positions where
    mask is set (e.g. the sigmoid tolerance mask) are excluded from the count."""
    g = to_f16_bits(got).astype(np.int64)
    r = to_f16_bits(ref).astype(np.int64)
    nan_g = (g & 0x7FFF) > 0x7C00
    nan_r = (r & 0x7FFF) > 0x7C00
    if zero_signs_equal:
        g = np.where((g & 0x7FFF) == 0, 0, g)
        r = np.where((r & 0x7FFF) == 0, 0, r)
    diff = (g != r) & ~(nan_g & nan_r)
    if mask is not None:
        diff = diff & ~np.broadcast_to(mask.astype(bool)[..., None], diff.shape) if diff.ndim == mask.ndim + 1 \
            else diff & ~mask.astype(bool)
    return int(diff.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    root = args.dir
    man = load_manifest(root)
    bad = 0
    # file hashes
    with open(os.path.join(root, "SHA256SUMS")) as f:
        sums = [l.split(None, 1) for l in f.read().splitlines() if l.strip()]
    nmiss = 0
    for h, rel in sums:
        p = os.path.join(root, rel.strip())
        if not os.path.exists(p):
            nmiss += 1
        elif sha256_file(p) != h:
            print("hash mismatch: %s" % rel)
            bad += 1
    if nmiss:
        print("%d files missing: run python -m clean.rr_f.tests.gen_vectors first" % nmiss)
        return 2
    if sorted(r.strip() for _, r in sums) != file_list(root):
        print("file list differs from SHA256SUMS")
        bad += 1
    # tables
    T = man["tables"]
    checks = [("d8", D8_TABLE), ("recip", recip_table()), ("sig", sig_tables()[0])]
    for flav in man["window_consts"]:
        wc = window_consts(man, flav)
        checks += [("attn_w_" + flav, attention_weight(ALL_F16, wc)), ("act_" + flav, activation(ALL_F16, wc))]
    for name, ref in checks:
        n = compare_f16(read_array(root, T[name]), ref, zero_signs_equal=False)
        bad += n
        print("table %-12s %s" % (name, "ok" if n == 0 else "%d DIFFERENT" % n))
    n = compare_e4m3(read_array(root, T["q8_of_f16"]), q8(ALL_F16), zero_codes_equal=False)
    bad += n
    print("table %-12s %s" % ("q8_of_f16", "ok" if n == 0 else "%d DIFFERENT" % n))
    # cases
    for case in man["cases"]:
        stage, inputs, W, consts, wc, shift_row, expected, inter = load_case(man, case, root)
        out, aux = run_stage(stage, inputs, W, consts, wc, shift_row, trace=True)
        # with the stored Z supplied (the 51.8 item 1 configuration)
        out2, _ = run_stage(stage, inputs, W, consts, wc, shift_row, Z_override=inter["Z"])
        nd = 0
        for k, ref in expected.items():
            if ref.dtype == np.uint8:
                nd += compare_e4m3(out[k], ref, zero_codes_equal=False) + compare_e4m3(out2[k], ref)
            else:
                nd += compare_f16(out[k], ref, zero_signs_equal=False) + compare_f16(out2[k], ref)
        nd += compare_f16(aux["Z"], inter["Z"], zero_signs_equal=False)
        nt = len(case.get("trace_windows", []))
        for k in ("q", "v", "P", "O", "y1q", "Au"):
            if k in inter:
                nd += compare_e4m3(aux["block"][k][:nt], inter[k], zero_codes_equal=False)
        for k in ("s", "Dp", "y1h", "Du", "Dd", "y2h"):
            if k in inter:
                nd += compare_f16(aux["block"][k][:nt], inter[k], zero_signs_equal=False)
        bad += nd
        print("case %-22s %s" % (case["id"], "ok" if nd == 0 else "%d DIFFERENT" % nd))
    print("ALL EQUAL" if bad == 0 else "MISMATCHES: %d" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
