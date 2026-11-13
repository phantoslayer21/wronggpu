# SPDX-License-Identifier: MIT
"""Test-vector generator for spec 51 (RR preset F window block).

For every stage kind of the 51.2 table (IN0, E1 with the 128 -> 64 adapter, E2-E5, fusion halves D4-F .. D0-F,
recurrent halves D4-R .. D1-R): synthetic weights and constants in the stated ranges, random inputs with edge values,
level extents from several render sizes (borders, repeated reflection below 8 tokens), and the reference outputs.
Also the exhaustive scalar tables (Q8 of every f16, attention weight and activation of every f16 per constant set,
sigmoid, reciprocal).

Run (from the repository root):  python -m clean.rr_f.tests.gen_vectors [--out DIR] [--seed N]
Deterministic: the same seed writes the same bytes."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

from clean.rr_f.reference.fmt import ALL_F16, D8_TABLE, q8
from clean.rr_f.reference.scalar import (activation, attention_weight, recip_table, sig_tables, SIG_NEAR_ULP32)
from clean.rr_f.reference.stages import STAGES, effective_shift, run_stage, stage_consts, stage_info, weight_shapes
from clean.rr_f.tests import synth
from clean.rr_f.tests.vectors_io import DEFAULT_DIR, f16_entry, write_array

# render token sizes (Wt, Ht): multiples of 32 (page 52.1.1); level k has Wt/2^k x Ht/2^k tokens
RENDERS = [(32, 32), (64, 96), (160, 96), (288, 256)]
TOKEN_BUDGET = 1536
SHIFT_ROWS = [(6, 6), (2, 4), (0, 0), (4, 2)]
HOT_STAGES = ["IN0", "E2", "E5", "D2-F", "D2-R"]
TRACE_WINDOWS = 2          # full intermediates are written for at most this many windows of a traced case

MATRIX_NOTE = ("51.1: the required matrix step rounds once to f16 per 32-input step; an implementation that "
               "accumulates longer chains in higher precision may differ by an occasional one-step difference in a "
               "quantised intermediate (not bit-exact then)")
ZORDER_NOTE = ("51.3 step 4 / 51.8 item 1: bit-exact when the row sums Z of intermediates/Z are supplied. With an "
               "implementation's own Z (f32 accumulation, any order) Z differs by one f16 step in about 22 % of the "
               "rows and P codes then differ by at most one code step in those rows; spec 51 states no bound for the "
               "downstream outputs in that case")
TOL_E4M3 = {"compare": "e4m3 code", "zero_codes_equal": True, "exact": True, "row_sum": ZORDER_NOTE,
            "matrix_step": MATRIX_NOTE}
TOL_HIDDEN = {"compare": "f16 bits", "zero_signs_equal": True, "exact": True, "row_sum": ZORDER_NOTE,
              "matrix_step": MATRIX_NOTE,
              "sigmoid": ("51.4: a conforming sig may give the reference or one of its two f16 neighbours where the "
                          "reference f32 value lies within about 2^-21 of an f16 midpoint; tokens whose hidden "
                          "update used such a sigmoid input are flagged in the 'sig_near_mask' file (1 = may differ)")}
TOL_Z = {"compare": "f16 bits", "exact": True,
         "note": ("reference Z = exact sum of the 64 weights rounded once (WrongGPU's choice); an f32 accumulation in "
                  "any order may differ by one f16 step (51.3 step 4)")}

LAYOUT_T = "[Y][X][C]"
NL = chr(10)


def level_extent(render, level):
    return render[0] >> level, render[1] >> level


def plan_cases():
    """List of (stage, case name, (Nx, Ny), shift_row, hot)."""
    plan = []
    for name in STAGES:
        info = stage_info(name)
        lv = info["level"]
        k = 0
        for r in RENDERS:
            Nx, Ny = level_extent(r, lv)
            if Nx * Ny > TOKEN_BUDGET:
                continue
            plan.append((name, "%s_r%dx%d" % (name, r[0], r[1]), (Nx, Ny), SHIFT_ROWS[k % 4], False))
            k += 1
        if k == 1:      # only one size fits (level 0): a second shift on the same size
            r = RENDERS[0]
            plan.append((name, "%s_r%dx%d_s2" % (name, r[0], r[1]), level_extent(r, lv), SHIFT_ROWS[1], False))
        if name in HOT_STAGES:
            r = RENDERS[0] if lv <= 1 else RENDERS[1]
            plan.append((name, "%s_hot_r%dx%d" % (name, r[0], r[1]), level_extent(r, lv), SHIFT_ROWS[3], True))
    return plan


def min_extents(name, ext):
    info = stage_info(name)
    e = list(ext)
    if info["kind"] in ("fuse", "rec"):
        e += [ext[0] // 2, ext[1] // 2]
    return min(e)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_DIR)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.out)
    if os.path.isdir(root):
        shutil.rmtree(root)            # our own generated folder
    os.makedirs(root)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    manifest = {
        "spec": "spec/51-rr-f-window-block.md draft 6",
        "generator": "clean/rr_f/tests/gen_vectors.py",
        "seed": args.seed,
        "conventions": {
            "e4m3": "uint8 code per value (byte = 128*s + 8*e + m)",
            "f16": "little-endian uint16 bit pattern per value",
            "u8": "uint8 per value (masks: 1 = flagged)",
            "tensor_layout": "[Y][X][C] (row-major; X fastest after C)",
            "window_index": ("window (i, j) of 51.6 has index w = j*nwx + i; window token t = 8*ly + lx; "
                             "nwx = ceil((Nx + sx)/8), nwy = ceil((Ny + sy)/8) with the effective shift"),
            "shift": "shift_row = the stage's SH row; recurrent halves use ((s + 4) mod 8) (effective_shift)",
            "Tb": ("Tb[h][k] is the stored 16 x 16 block of pair k in the order of 51.3 step 2: (0,0),(1,0),(0,1),"
                   "(1,1),(0,2),(1,2),(0,3),(1,3),(2,2),(3,2),(2,3),(3,3) (PROVISIONAL, QUESTIONS-51.md Q1)"),
            "E1_input": ("E1 'in' is the level-1 tensor with 128 channels as stored by IN0 (page 52); the reference "
                         "does not interpret the channel order (QUESTIONS-51.md Q2)"),
            "synthetic": ("all weights and constants are synthetic test inputs (page 57 not available); ranges in "
                          "QUESTIONS-51.md Q3"),
        },
        "tolerance_classes": {"e4m3_network": TOL_E4M3, "f16_hidden": TOL_HIDDEN, "f16_rowsum": TOL_Z},
        "files": ("binary files are not committed (clean-repo content gate): regenerate them with the generator; "
                  "SHA256SUMS lists every file, check_vectors verifies it"),
        "tables": {},
        "window_consts": {},
        "weight_sets": {},
        "cases": [],
    }

    # ---------------- window-block constant sets
    wcs = {}
    for flav in ("A", "B", "zovf"):
        wc, chk = synth.window_consts(rng, flav)
        wcs[flav] = wc
        manifest["window_consts"][flav] = {
            "values": {k: f16_entry(v) for k, v in wc.as_dict().items()},
            "converter_check": chk,
            "note": {"A": "synthetic build A", "B": "synthetic build B",
                     "zovf": "64 * w_max reaches the f16 overflow threshold: rows with Z = +inf occur"}[flav],
        }

    # ---------------- exhaustive scalar tables
    T = manifest["tables"]
    T["d8"] = write_array(root, "tables/d8.f16", D8_TABLE, "f16", "[code]", tolerance={"exact": True},
                          rule="51.1 d8 of all 256 codes (NaN for 0x7F/0xFF, -0 for 0x80)")
    T["q8_of_f16"] = write_array(root, "tables/q8_of_f16.e4m3", q8(ALL_F16), "e4m3", "[f16 bits]",
                                 tolerance={"exact": True, "zero_codes_equal": False},
                                 rule="51.1 Q8 of every f16 pattern (51.8 item 3)")
    sig, s32, near = sig_tables()
    T["sig"] = write_array(root, "tables/sig.f16", sig, "f16", "[f16 bits]",
                           tolerance={"exact": False, "allowed": "reference or one of its two f16 neighbours where "
                                      "sig_near = 1 (51.4, 51.8 item 4)"},
                           rule="51.4 sigmoid of every f16 pattern")
    T["sig_near"] = write_array(root, "tables/sig_near.u8", near.astype(np.uint8), "u8", "[f16 bits]",
                                rule="1 where the reference f32 s lies within %d f32 ulps (about 2^-21 relative) of "
                                     "an f16 rounding midpoint" % SIG_NEAR_ULP32)
    T["recip"] = write_array(root, "tables/recip.f16", recip_table(), "f16", "[f16 bits]",
                             tolerance={"exact": True}, rule="51.3 step 4 r = R(1/Z) of every f16 pattern")
    for flav, wc in wcs.items():
        T["attn_w_" + flav] = write_array(root, "tables/attn_w_%s.f16" % flav, attention_weight(ALL_F16, wc), "f16",
                                          "[f16 bits of the score s]", tolerance={"exact": True},
                                          window_consts=flav, rule="51.3 step 3 w(s) for every f16 score (51.8 item 2)")
        T["act_" + flav] = write_array(root, "tables/act_%s.f16" % flav, activation(ALL_F16, wc), "f16",
                                       "[f16 bits of x]", tolerance={"exact": True, "nan_equal": True},
                                       window_consts=flav, rule="51.3 step 8 act(x) for every f16 x (51.8 item 2)")

    # ---------------- weights (one set per stage kind, plus 'hot' sets)
    for name in STAGES:
        for hot in (False, True):
            if hot and name not in HOT_STAGES:
                continue
            key = name + ("-hot" if hot else "")
            W = synth.stage_weights(rng, name, hot=hot)
            shapes = weight_shapes(name)
            tens = {}
            for tname, (shape, typ) in shapes.items():
                assert W[tname].shape == shape
                tens[tname] = write_array(root, "weights/%s/%s.%s" % (key, tname, typ), W[tname], typ,
                                          _weight_layout(tname))
            manifest["weight_sets"][key] = {"stage": name, "hot": hot, "tensors": tens, "_W": W}

    # ---------------- cases
    summary = []
    for ci, (name, cname, ext, shift_row, hot) in enumerate(plan_cases()):
        info = stage_info(name)
        Nx, Ny = ext
        wflav = "zovf" if hot else "AB"[ci % 2]
        wc = wcs[wflav]
        sc_all = synth.stage_consts(rng, hot=hot)
        sc = {k: sc_all[k] for k in stage_consts(name)}
        wkey = name + ("-hot" if hot else "")
        W = manifest["weight_sets"][wkey]["_W"]
        inputs = synth.stage_inputs(rng, name, Nx, Ny, hot=hot)
        out, aux = run_stage(name, inputs, W, sc, wc, shift_row, trace=True)
        d = "cases/" + cname
        case = {
            "id": cname, "stage": name, "level": info["level"], "C": info["C"], "H": info["H"],
            "extent": [Nx, Ny], "coarse_extent": ([Nx // 2, Ny // 2] if info["kind"] in ("fuse", "rec") else None),
            "min_extent": min_extents(name, ext),
            "border_rule": ("single reflection (every extent >= 8)" if min_extents(name, ext) >= 8 else
                            "repeated reflection (an extent below 8: WrongGPU's choice, 51.6)"),
            "shift_row": list(shift_row), "shift": list(aux["shift"]), "windows": list(aux["grid"]),
            "weights": wkey, "window_consts": wflav, "hot": hot,
            "stage_consts": {k: f16_entry(v) for k, v in sc.items()},
            "inputs": {}, "outputs": {}, "intermediates": {},
        }
        for k, v in inputs.items():
            if k in ("hold", "Hc"):
                case["inputs"][k] = write_array(root, "%s/in_%s.f16" % (d, k), v, "f16", "[Y][X][4]")
            else:
                case["inputs"][k] = write_array(root, "%s/in_%s.e4m3" % (d, k), v, "e4m3", LAYOUT_T)
        case["outputs"]["y2"] = write_array(root, d + "/out_y2.e4m3", out["y2"], "e4m3", LAYOUT_T,
                                           tolerance="e4m3_network")
        if "merged" in out:
            case["outputs"]["merged"] = write_array(root, d + "/out_merged.e4m3", out["merged"], "e4m3", LAYOUT_T,
                                                    tolerance="e4m3_network", level=info["level"] + 1)
        if "hnew" in out:
            m = write_array(root, d + "/sig_near_hnew.u8", aux["hnew_sig_near"].astype(np.uint8), "u8", "[Y][X]")
            case["outputs"]["hnew"] = write_array(root, d + "/out_hnew.f16", out["hnew"], "f16", "[Y][X][4]",
                                                  tolerance="f16_hidden", sig_near_mask=m["file"])
        if "hmerged" in out:
            m = write_array(root, d + "/sig_near_hmerged.u8", aux["hmerged_sig_near"].astype(np.uint8), "u8",
                            "[Y][X]")
            case["outputs"]["hmerged"] = write_array(root, d + "/out_hmerged.f16", out["hmerged"], "f16", "[Y][X][4]",
                                                     tolerance="f16_hidden", sig_near_mask=m["file"],
                                                     level=info["level"] + 1)
        I = case["intermediates"]
        I["Z"] = write_array(root, d + "/int_Z.f16", aux["Z"], "f16", "[w][h][a]", tolerance="f16_rowsum",
                             use="supply to the implementation for the bit-exact comparison (51.8 item 1)")
        # full traces for the case of each stage with the fewest windows (and every hot case)
        nwin = aux["grid"][0] * aux["grid"][1]
        summary.append((name, cname, nwin))
        case["_nwin"] = nwin
        case["_aux"] = aux
        manifest["cases"].append(case)
        _stats(case, out, aux)

    # traces
    best = {}
    for case in manifest["cases"]:
        if case["hot"]:
            continue
        if case["stage"] not in best or case["_nwin"] < best[case["stage"]]["_nwin"]:
            best[case["stage"]] = case
    for case in manifest["cases"]:
        if case["hot"] or best[case["stage"]] is case:
            _write_trace(root, case)
    for case in manifest["cases"]:
        del case["_nwin"], case["_aux"]
    for ws in manifest["weight_sets"].values():
        del ws["_W"]
    with open(os.path.join(root, "manifest.json"), "w", newline=NL) as f:
        json.dump(manifest, f, indent=1)
        f.write(NL)
    write_sums(root)
    with open(os.path.join(root, ".gitignore"), "w", newline=NL) as f:
        f.write(NL.join(["# generated binaries are not committed (see manifest.json 'files')", "*", "!.gitignore",
                         "!manifest.json", "!SHA256SUMS", ""]))
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(root) for fn in fns)
    print("wrote %d cases, %d weight sets, %.1f MB in %.0f s -> %s" % (
        len(manifest["cases"]), len(manifest["weight_sets"]), total / 1e6, time.time() - t0, root))
    return manifest


def file_list(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root).replace(os.sep, "/")
            if rel not in ("manifest.json", "SHA256SUMS", ".gitignore"):
                out.append(rel)
    return sorted(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def write_sums(root):
    with open(os.path.join(root, "SHA256SUMS"), "w", newline=NL) as f:
        for rel in file_list(root):
            f.write("%s  %s%s" % (sha256_file(os.path.join(root, rel)), rel, NL))


def _weight_layout(t):
    return {
        "Wq": "[h][c][i]", "Wv": "[h][c][i]", "Tb": "[h][k][a mod 16][b mod 16]", "Wo": "[h][i][c]", "bo": "[c]",
        "W1": "[c][4C]", "b1": "[4C]", "W2": "[4C][c]", "b2": "[c]", "Wm": "[n*C + c][Cn]", "bm": "[Cn]",
        "Wa": "[128][64]", "ba": "[64]", "Wup": "[Kc][n*C + c]", "bup": "[4C]", "Wh": "[c][j]", "bh": "[j]",
    }[t]


def _write_trace(root, case):
    aux = case["_aux"]
    b = aux["block"]
    d = "cases/%s/trace" % case["id"]
    I = case["intermediates"]
    nt = min(TRACE_WINDOWS, aux["Z"].shape[0])
    case["trace_windows"] = list(range(nt))
    note = "windows 0..%d of the case (window index w as in conventions)" % (nt - 1)
    I["x_in"] = write_array(root, d + "/x_in.e4m3", aux["x_in"][:nt], "e4m3", "[w][t][c]",
                            note="block input per window; " + note)
    spec = [("q", "e4m3", "[w][h][t][i]"), ("v", "e4m3", "[w][h][t][i]"), ("s", "f16", "[w][h][a][b]"),
            ("P", "e4m3", "[w][h][a][b]"), ("O", "e4m3", "[w][h][a][i]"),
            ("Dp", "f16", "[w][t][c]"), ("y1h", "f16", "[w][t][c]"), ("y1q", "e4m3", "[w][t][c]"),
            ("Du", "f16", "[w][t][4C]"), ("Au", "e4m3", "[w][t][4C]"), ("Dd", "f16", "[w][t][c]"),
            ("y2h", "f16", "[w][t][c]")]
    for k, typ, lay in spec:
        I[k] = write_array(root, "%s/%s.%s" % (d, k, typ), b[k][:nt], typ, lay, note=note)
    if "xf" in aux:
        I["xf"] = write_array(root, d + "/xf.f16", aux["xf"][:nt], "f16", "[w][t][c]",
                              note="fusion f16 (residual base); " + note)
        I["upq"] = write_array(root, d + "/upq.e4m3", aux["upq"], "e4m3", "[Yc][Xc][n*C + c]",
                               note="quantised up-projection of every coarse token (unreflected tensor)")
    if "Dh" in aux:
        I["Dh"] = write_array(root, d + "/Dh.f16", aux["Dh"], "f16", "[Y][X][j]")
    if "u" in aux:
        I["u"] = write_array(root, d + "/u.f16", aux["u"], "f16", "[Y][X][j]", note="bilinear coarse hidden, j<3")


def _stats(case, out, aux):
    b = aux["Z"]
    st = {"Z_inf_rows": int(np.isinf(b).sum())}
    for k, v in out.items():
        if v.dtype == np.uint8:
            st[k + "_sat448"] = int(((v & 0x7F) == 0x7E).sum())
            st[k + "_nan"] = int(((v & 0x7F) == 0x7F).sum())
            st[k + "_zero"] = int(((v & 0x7F) == 0).sum())
        else:
            st[k + "_nonfinite"] = int((~np.isfinite(v)).sum())
    case["stats"] = st


if __name__ == "__main__":
    main(sys.argv[1:])
