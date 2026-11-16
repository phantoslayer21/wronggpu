# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the shared stages' FP16-form kernels (clean/l/kernels/l_q7w.hip; D3D12 ports
clean/dx12/kernels/l_q7w.hlsl) against the oracle (clean/l/reference/oracle.py run_stage_q7's parts, FP8 op model
"exact"), raw bytes.

    python clean/l/runner/l_q7w_check.py --build <dir> --compile                # CPU: <dir>/l_q7w.co
    python clean/l/runner/l_q7w_check.py --build <dir> --prepare [--real <build with lframe_model/params.pkl>
                                         --private-out <folder outside the repository>]  # CPU
    python clean/l/runner/l_q7w_check.py --build <dir> --prepare-d3d12          # CPU: D3D12 ports + kernel_replay.exe
    python clean/l/runner/l_q7w_check.py --build <dir> --run [--d3d12] --json out.json   # GPU (inside a GPU slot)

Kernels and their oracle: l_q7_block_c64 / _c96 / _c128 = FP8(swin_block(x, border "mirror")) on an FP8 map;
l_q7_block_c64_dec / _c96_dec = FP8(swin_block(E, border "coarse", skip S)); l_q7_merge_c64 / _c96 = FP8(clamp(the
Wm chain over space_to_depth(s) from bm, +-6.28125)); l_q7_expand_c128 / _c96 = depth_to_space(FP8(the We chain from
be)). Weights: Model M's synthetic stage parameters (two seeds) and, with --real, Model L's real stage weights (enc1,
enc2, b0, b5, dec2, dec1). Grids of several sizes, both window offsets of each stage, input scales, special values
(+-448, zeros, a NaN code). Every output buffer is filled with the back end's sentinel (HIP 0xA5, D3D12 0x5A) and
has 64 slack bytes after it that must keep it; the output must equal the oracle's bytes exactly. Negative control
per kernel: its first case with 4 or more workgroups launched with half of them must differ. Counts and digests only.
--real (the synthetic-values rule): the cases file and the D3D12 replay folder go only to --private-out, a folder
outside the repository (refused inside it); give the same --private-out to --run.
"""
import argparse
import ctypes
import hashlib
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.l.runner import l_record as LR  # noqa: E402
from clean.l.runner.l_q7_check import data_dir, load_cases, mask_control, mask_nan, out8, run_identity  # noqa: E402
from clean.reference import swin as MS  # noqa: E402
from clean.reference.fp8 import f16, fp8_encode, q8  # noqa: E402

HIP_SENTINEL, D3D12_SENTINEL, SLACK = 0xA5, 0x5A, 64
CO = "l_q7w.co"
BLOCKS = {"l_q7_block_c64": ("enc1", False), "l_q7_block_c96": ("enc2", False), "l_q7_block_c128": ("b0", False),
          "l_q7_block_c64_dec": ("dec1", True), "l_q7_block_c96_dec": ("dec2", True)}
MERGES = {"l_q7_merge_c64": "enc1", "l_q7_merge_c96": "enc2"}
EXPANDS = {"l_q7_expand_c96": "dec1", "l_q7_expand_c128": "dec2"}
KERNELS = tuple(BLOCKS) + tuple(MERGES) + tuple(EXPANDS)
GRIDS = [(8, 8), (12, 16), (16, 12)]
SCALES = (0.5, 3.0)


def compile_kernel(build, rocm):
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    os.makedirs(build, exist_ok=True)
    B.run(B.device_command(B.find_clang(rocm), rocm, os.path.join(ROOT, "clean", "l", "kernels", "l_q7w.hip"),
                           os.path.join(build, CO)) + ["-I", os.path.join(ROOT, "clean", "kernels")])


def _consts(stage):
    k = T.block_consts(T.TABLE, stage)
    k["mm8"] = "exact"
    return k


def _fp8_map(rng, shape, scale, special):
    v = q8(rng.normal(0, scale, shape))
    if special:
        v[0, 0, :3] = [448.0, -448.0, 0.0]
        v[1, 1, 2] = np.nan
    return v


def _b(a):
    return fp8_encode(a).tobytes()


def make_cases(params_by_set, seed=7):
    """[(kernel, label, grid, block, args, defined bytes, oracle bytes)] for every kernel and parameter set."""
    rng = np.random.default_rng(seed)
    tab = LP.q7_tables()
    cases = []
    for sname, params in params_by_set:
        for kern, (stage, dec) in BLOCKS.items():
            cfg = MS.STAGES[stage]
            p = params[stage if stage in params else stage]
            C, heads = cfg["C"], cfg["heads"]
            blob = LP.pack_q7_block(p, C, heads, 4 * C)
            for (H, W) in GRIDS:
                for scale in SCALES:
                    special = scale == SCALES[1] and (H, W) == GRIDS[0]
                    for parity in (0, 1):
                        off = MS.window_offset(stage, parity)
                        x = _fp8_map(rng, (H, W, C), scale, special)
                        if dec:
                            S = _fp8_map(rng, (H, W, C), scale, special)
                            want = L.swin_block(x, p, off, heads, _consts(stage), "coarse", skip=S)
                            args = [("buf", _b(x)), ("buf", _b(S))]
                        else:
                            want = L.swin_block(x, p, off, heads, _consts(stage), "mirror")
                            args = [("buf", _b(x)), ("null",)]
                        n = H * W * C
                        args += [("out", n), ("buf", blob), ("buf", tab), ("u32", H), ("u32", W), ("u32", off[0]),
                                 ("u32", off[1])]
                        grid = -(-(W + off[0]) // 4) * -(-(H + off[1]) // 4)
                        cases.append({"kernel": kern, "set": sname, "label": f"{H}x{W} s{scale} off{off}"
                                      + (" special" if special else ""), "grid": grid, "args": args,
                                      "defined": n, "want": out8(want)})
        for kern, stage in MERGES.items():
            p = params[stage]
            C, Co = MS.STAGES[stage]["C"], MS.STAGES[stage]["C_out"]
            blob = LP.pack_q7_linear(p["Wm"], p["bm"])
            for (H, W) in GRIDS:
                for scale in SCALES:
                    special = scale == SCALES[1] and (H, W) == GRIDS[0]
                    s = _fp8_map(rng, (H, W, C), scale, special)
                    m = L.mm8(np.asarray(p["bm"], np.float64), L.space_to_depth(s), p["Wm"])
                    want = q8(np.minimum(np.maximum(m, -T.MERGE_CLAMP), T.MERGE_CLAMP))
                    n = (H // 2) * (W // 2) * Co
                    cases.append({"kernel": kern, "set": sname, "label": f"{H}x{W} s{scale}"
                                  + (" special" if special else ""), "grid": (H // 2) * (W // 2),
                                  "args": [("buf", _b(s)), ("out", n), ("buf", blob), ("u32", H), ("u32", W)],
                                  "defined": n, "want": out8(want)})
        for kern, stage in EXPANDS.items():
            p = params[stage]
            C, Cl = MS.STAGES[stage]["C"], MS.STAGES[stage]["C_low"]
            blob = LP.pack_q7_linear(p["We"], p["be"])
            for (H, W) in GRIDS:
                for scale in SCALES:
                    special = scale == SCALES[1] and (H, W) == GRIDS[0]
                    h, w = H // 2, W // 2
                    xl = _fp8_map(rng, (h, w, Cl), scale, special)
                    want = L.depth_to_space(q8(L.mm8(np.asarray(p["be"], np.float64), xl, p["We"])))
                    n = H * W * C
                    cases.append({"kernel": kern, "set": sname, "label": f"{h}x{w} s{scale}"
                                  + (" special" if special else ""), "grid": h * w,
                                  "args": [("buf", _b(xl)), ("out", n), ("buf", blob), ("u32", h), ("u32", w)],
                                  "defined": n, "want": out8(want)})
    return cases


def storage(p):
    """Synthetic parameters in the formats the device stores (FP8 matrices, FP16 vectors and B)."""
    return {k: (q8(v) if np.ndim(v) >= 2 and k != "B" else f16(v)) for k, v in
            ((k, np.asarray(v, np.float64)) for k, v in p.items())}


def prepare(build, real=None):
    sets = [("synthetic_%d" % s, {st: storage(MS.random_params(st, np.random.default_rng(s)))
                                  for st in ("enc1", "enc2", "b0", "dec2", "dec1")}) for s in (89, 90)]
    real_sha = None
    if real:
        path = os.path.join(real, "lframe_model", "params.pkl")
        with open(path, "rb") as f:
            real_sha = hashlib.sha256(f.read()).hexdigest()
        params = pickle.load(open(path, "rb"))["L"]
        sets.append(("real", {st: params[st] for st in ("enc1", "enc2", "b0", "dec2", "dec1")}))
        sets.append(("real_b5", {"b0": params["b5"], **{st: params[st] for st in ("enc1", "enc2", "dec2", "dec1")}}))
    cases = make_cases(sets)
    if real:
        from clean.l.runner.l_private import mark_real
        mark_real(os.path.join(build, "l_q7w_cases.pkl"), "l_q7w_check --real")
    with open(os.path.join(build, "l_q7w_cases.pkl"), "wb") as f:
        pickle.dump({"cases": cases, "real_params_sha256": real_sha}, f)
    return len(cases)


def hip_run(h, mod, case, grid=None):
    args, owned, outs = [], [], []
    for a in case["args"]:
        if a[0] == "buf":
            p = h.malloc(max(len(a[1]), 4))
            h.upload(p, a[1])
            owned.append(p)
            args.append(ctypes.c_void_p(p))
        elif a[0] == "out":
            p = h.malloc(a[1] + SLACK)
            h.memset(p, HIP_SENTINEL, a[1] + SLACK)
            owned.append(p)
            outs.append((p, a[1] + SLACK))
            args.append(ctypes.c_void_p(p))
        elif a[0] == "null":
            args.append(ctypes.c_void_p(0))
        else:
            args.append(ctypes.c_int(a[1]))
    try:
        h.launch(h.get_function(mod, case["kernel"]), grid or case["grid"], 32, args)
        h.sync()
        return bytes(h.download(*outs[0]))
    finally:
        for p in owned:
            h.free(p)


def _check(got, case, sentinel):
    body, slack = mask_nan(got[:case["defined"]], "fp8") if got else b"", got[case["defined"]:]
    return {"identical": body == case["want"] and slack == bytes([sentinel]) * SLACK,
            "bytes_differ": sum(a != b for a, b in zip(body, case["want"])) if body else -1,
            "sha256": hashlib.sha256(body).hexdigest()}


def run(build, rocm, d3d12=False, private_out=None):
    from clean.runner.hip import Hip, clean_process_environment
    ddir = data_dir(build, private_out)
    prep = load_cases(os.path.join(ddir, "l_q7w_cases.pkl"), private_out)
    clean_process_environment(rocm)
    cases = prep["cases"]
    h = Hip(rocm)
    with open(os.path.join(build, CO), "rb") as f:
        mod = h.load_module(f.read())
    res = {"what": "the shared stages' FP16-form kernels against the oracle, raw bytes", "fp8_op_model": "exact",
           "identity": run_identity(build, [CO, os.path.join(ddir, "l_q7w_cases.pkl") if private_out
                                            else "l_q7w_cases.pkl"]), "real_params_sha256": prep["real_params_sha256"],
           "nan_sign_masked": "outputs compared and hashed with only a NaN's sign bit cleared (accepted port exception; "
                              "a canonical NaN store costs about 3 VALU per stored dword in the RGA ISA)",
           "nan_mask_control": mask_control(),
           "shared_dec_border": T.TABLE["shared_dec_border"],
           "shared_dec_border_source": "spec Q10b (PRESET-L-P3-Q10b-decoder-border-rule.md): E by the coarse mirror, "
                                       "S by the fine mirror (FACT); the member's own sub-pixel kept (INFERENCE, "
                                       "UNCONFIRMED U2)",
           "cases": [], "negative_controls": {}}
    negs = {}
    for i, c in enumerate(cases):
        r = _check(hip_run(h, mod, c), c, HIP_SENTINEL)
        r.update({"kernel": c["kernel"], "set": c["set"], "label": c["label"]})
        res["cases"].append(r)
        if c["kernel"] not in negs and c["grid"] >= 4:
            negs[c["kernel"]] = i
            half = hip_run(h, mod, c, grid=c["grid"] // 2)
            res["negative_controls"][c["kernel"]] = {"case": c["label"], "set": c["set"],
                                                     "caught": not _check(half, c, HIP_SENTINEL)["identical"]}
    per = {}
    for r in res["cases"]:
        k = per.setdefault(r["kernel"], [0, 0])
        k[0] += 1
        k[1] += r["identical"]
    res["per_kernel"] = {k: {"cases": v[0], "identical": v[1]} for k, v in per.items()}
    res["cases_identical"] = sum(r["identical"] for r in res["cases"])
    res["pass"] = (res["cases_identical"] == len(cases) and set(res["negative_controls"]) == set(KERNELS)
                   and all(v["caught"] for v in res["negative_controls"].values()) and res["nan_mask_control"]["pass"])
    if d3d12:
        res["d3d12"] = run_d3d12(build, cases, res, negs, ddir)
        res["pass"] = res["pass"] and res["d3d12"]["pass"]
    return res


def prepare_d3d12(build, rocm):
    from clean.l.runner.l_d3d12_check import prepare_replay
    return prepare_replay(build, rocm, list(KERNELS))


def run_d3d12(build, cases, res, negs, ddir=None):
    import subprocess
    import time
    cdir = os.path.join(ddir or build, "replay-q7w-" + time.strftime("%Y%m%d-%H%M%S"))   # the cases file's folder
    os.makedirs(cdir)
    files, lines = {}, []

    def buf(data):
        key = hashlib.sha256(data).hexdigest()[:20]
        if key not in files:
            files[key] = "b_%s.bin" % key
            with open(os.path.join(cdir, files[key]), "wb") as f:
                f.write(data)
        return files[key]

    for i, c in enumerate(cases):
        runs = [("c%04d" % i, c["grid"])]
        if negs.get(c["kernel"]) == i:
            runs.append(("negative_c%04d" % i, c["grid"] // 2))
        for name, grid in runs:
            lines += ["case " + name, "module " + CO, "kernel %s %d 32" % (c["kernel"], grid)]
            for a in c["args"]:
                if a[0] == "buf":
                    lines.append("arg buf " + buf(a[1]))
                elif a[0] == "out":
                    lines.append("arg out %d %s.o %d" % (a[1] + SLACK, name, D3D12_SENTINEL))
                elif a[0] == "null":
                    lines.append("arg null")
                else:
                    lines.append("arg u32 %d" % a[1])
            lines.append("")
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "256"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=110, creationflags=0x08000000)

    def read(n):
        p = os.path.join(cdir, n + ".o")
        return open(p, "rb").read() if os.path.exists(p) else b""

    out = {"replay_exit": r.returncode, "replay_tail": (r.stdout[-300:] + r.stderr[-300:]).strip(),
           "identity": run_identity(build, ["kernel_replay.exe"] + ["dx12/%s.dxil" % k for k in KERNELS]),
           "negative_controls": {}}
    same_hip = same_oracle = 0
    for i, (c, rc) in enumerate(zip(cases, res["cases"])):
        d = _check(read("c%04d" % i), c, D3D12_SENTINEL)
        rc["d3d12_identical_to_oracle"] = d["identical"]
        rc["d3d12_sha256"] = d["sha256"]
        same_oracle += d["identical"]
        same_hip += d["identical"] and d["sha256"] == rc["sha256"]
    for k, i in negs.items():
        out["negative_controls"][k] = {"caught": not _check(read("negative_c%04d" % i), cases[i],
                                                            D3D12_SENTINEL)["identical"]}
    out.update({"cases": len(cases), "cases_identical_to_hip": same_hip, "cases_identical_to_oracle": same_oracle})
    out["pass"] = (r.returncode == 0 and same_hip == len(cases) and same_oracle == len(cases)
                   and all(v["caught"] for v in out["negative_controls"].values()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--compile", action="store_true")
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--prepare-d3d12", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--d3d12", action="store_true")
    ap.add_argument("--real")
    ap.add_argument("--private-out", help="--real: the folder (outside the repository) for the cases file and the "
                    "replay folder; the same one for --run")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.real and not a.prepare:
        ap.error("--real: with --prepare (give --run the same --private-out)")
    if a.real and not a.private_out:
        raise SystemExit("--real is required to have --private-out <folder outside the repository>: the cases file "
                         "holds real weights and the oracle's bytes on them")
    if a.private_out and (a.real or a.run):
        from clean.l.runner.l_private import private_json
        private_json(a.json)               # a real run's record outside the repository
        data_dir(a.build, a.private_out)   # refused inside the repository, before anything is read
    if a.run:   # a real prepared file (marker) only from its private folder: checked before anything else
        from clean.l.runner.l_private import guard_cases
        guard_cases(os.path.join(data_dir(a.build, a.private_out), "l_q7w_cases.pkl"), a.private_out)
    if a.compile:
        compile_kernel(a.build, a.rocm)
        return 0
    if a.prepare:
        print("cases", prepare(data_dir(a.build, a.private_out) if a.real else a.build, a.real))
        return 0
    if a.prepare_d3d12:
        print("built", prepare_d3d12(a.build, a.rocm))
        return 0
    res = run(a.build, a.rocm, a.d3d12, a.private_out)
    for k, v in res["per_kernel"].items():
        print(f"{k:22s} HIP identical {v['identical']} of {v['cases']}  negative control caught "
              f"{res['negative_controls'].get(k, {}).get('caught')}")
    if a.d3d12:
        d = res["d3d12"]
        print("D3D12 replay exit", d["replay_exit"], "| identical to HIP", d["cases_identical_to_hip"],
              "| to the oracle", d["cases_identical_to_oracle"], "of", d["cases"])
    print("PASS" if res["pass"] else "FAIL")
    if a.json:
        LR.write_l_record(res, a.json)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
