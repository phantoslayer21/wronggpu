# SPDX-License-Identifier: MIT
"""Model L (issue #89): adversarial probe of the S2a primitive (clean/l/kernels/l_q7_op8.h and its
D3D12 twin clean/dx12/hlsl/l_q7_op8.hlsli): one 32-wide FP8 chunk summed by two matrix instructions started from 0,
used only while wsum x amax < 63 (wsum = sum |w| of the row's chunk, as the packers store it; amax = max |a| over the
SAME 32 values, from the codes), otherwise the exact scalar op8; the FP16 start joined afterwards.

Every op's result must equal the exact model (oracle MM8 "exact": the exact sum plus the start, rounded once to FP16)
bit for bit (a NaN's sign bit masked: the accepted port exception), its path (bounded / fallback) must equal the bound
computed here in float32, and HIP and D3D12 must agree. Sets: the adversarial families of l_mm8_f32_check (typical,
cancellation, dynamic_range, near_tie, start_midpoint, subnormal) and S2a's own:
- bound_at_63: wsum x amax exactly 63 (must take the fallback);
- bound_below_63: the largest wsum x amax below 63 that FP8 operands can reach (bounded; every such product is a
  multiple of 2^-18, so the largest float32 below 63, 63 - 2^-18, is not reachable: recorded);
- unbounded_inexact: wsum x amax in [64, 128) with a sum the float32 result cannot hold (64 + 2^-18), the start
  cancelling the 64: wrong if the bound let it through;
- far_half: the chunk's largest |a| in the half of K the output's lane does not load (amax must cover both lanes);
- big_inexact: a product sum of 50176 + 2^-18 with the start -50176;
- nan: NaN activation codes (amax NaN) and NaN weights (wsum NaN), both must take the fallback;
- zeros: signed-zero products, exact cancellations, +-0 starts.
Mutants (probe builds only), each must be caught: 1 the bound limit 128; 2 the first instruction's sum fed as the
second's accumulator; 3 amax from the lane's own 16 values; 4 the start fed into the accumulator.

    python clean/l/runner/l_q7_s2a_check.py --build <dir> --compile          # HIP probe + mutants (compile slot)
    python clean/l/runner/l_q7_s2a_check.py --build <dir> --prepare          # CPU
    python clean/l/runner/l_q7_s2a_check.py --build <dir> --prepare-d3d12    # D3D12 probe ports + kernel_replay.exe
    python clean/l/runner/l_q7_s2a_check.py --build <dir> --run [--d3d12] --json out.json   # GPU slot
"""
import argparse
import ctypes
import hashlib
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_mm8_f32_check as M  # noqa: E402
from clean.l.runner import l_record as LR  # noqa: E402
from clean.l.runner.l_q7_check import mask_control, run_identity  # noqa: E402
from clean.reference.fp8 import DECODE, f16, fp8_encode, q8  # noqa: E402

LIMIT = np.float32(63.0)
MUTANTS = (1, 2, 3, 4)
MUTANT_WHAT = {1: "bound limit 128 instead of 63", 2: "first instruction's sum fed as the second's accumulator",
               3: "amax from the lane's own 16 values only", 4: "start fed into the first instruction's accumulator"}
FP8_POS = np.unique(np.abs(np.asarray(DECODE, np.float64)[np.isfinite(np.asarray(DECODE, np.float64))]))
FP8_POS = FP8_POS[FP8_POS > 0]


def _signs(rng, shape):
    return rng.choice([-1.0, 1.0], shape)


def _pow2_row(target, rng):
    """32 FP8 magnitudes summing exactly to target (a multiple of 2^-9, at most 511): its binary digits, padded with
    zeros, shuffled."""
    k = int(round(target * 512))
    assert k / 512 == target and 0 < k < 512 * 512
    vals = [2.0 ** (b - 9) for b in range(18) if k >> b & 1]
    assert len(vals) <= 32
    row = np.array(vals + [0.0] * (32 - len(vals)))
    return row[rng.permutation(32)]


def _row_with_max(amax, rng):
    """32 FP8 values with max |a| exactly amax, the rest smaller, random signs."""
    small = FP8_POS[FP8_POS <= amax]
    row = rng.choice(small, 32) * _signs(rng, 32)
    row[rng.integers(32)] = amax * rng.choice([-1.0, 1.0])
    return q8(row)


def largest_below_63():
    """The largest wsum x amax below 63 reachable with FP8 operands (wsum a sum of 32 FP8 magnitudes: a multiple of
    2^-9 up to 32 x 448; amax an FP8 value), as (product, wsum, amax), and whether 63 - 2^-18 is reachable."""
    best = (0.0, 0.0, 0.0)
    for A in FP8_POS:
        k = int(np.floor((63.0 - 2.0 ** -30) / A * 512))           # wsum = k / 512 with k A < 63 x 512 strictly
        while k > 0 and (k / 512) * A >= 63.0:
            k -= 1
        if k <= 0 or k > 32 * 448 * 512:
            continue
        p = (k / 512) * A
        if p > best[0]:
            best = (p, k / 512, float(A))
    return best, best[0] == 63.0 - 2.0 ** -18


def s2a_family(name, rng, n):
    st = np.zeros(n)
    a = np.zeros((n, 32))
    w = np.zeros((n, 32))
    if name in ("bound_at_63", "bound_below_63"):
        if name == "bound_at_63":
            pairs = [(A, 63.0 / A) for A in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.75, 1.5, 3.0, 0.875, 1.75, 3.5)]
        else:
            (p, ws, A), _ = largest_below_63()
            pairs = [(A, ws)]
        for i in range(n):
            A, ws = pairs[i % len(pairs)]
            if name == "bound_below_63":
                w[i] = _row_any_sum(ws, rng)
            else:
                w[i] = _pow2_row(ws, rng) * _signs(rng, 32)
            a[i] = _row_with_max(A, rng)
            st[i] = f16(rng.normal(0, 8))
    elif name == "unbounded_inexact":
        for i in range(n):                                        # 8 w + -2^-18: a float32 tie above 64
            k1, k2 = rng.choice(32, 2, replace=False)
            a[i, k1], w[i, k1] = 8.0, rng.choice([8.0, 10.0, 12.0, 14.0]) * rng.choice([-1.0, 1.0])
            a[i, k2], w[i, k2] = 2.0 ** -9, 2.0 ** -9 * rng.choice([-1.0, 1.0])
            st[i] = -a[i, k1] * w[i, k1]
    elif name == "far_half":
        for i in range(n):
            hf = (i % 16) >> 3                                    # the diagonal lane's half (l_q7_s2a_probe)
            far = [k for k in range(32) if ((k % 16) >> 3) != hf]
            near = [k for k in range(32) if ((k % 16) >> 3) == hf]
            kb, kt = rng.choice(far), rng.choice(near)
            a[i, kb], w[i, kb] = 8.0, 8.0
            a[i, kt], w[i, kt] = 2.0 ** -9, 2.0 ** -9
            st[i] = -64.0
    elif name == "big_inexact":
        for i in range(n):
            k1, k2 = rng.choice(32, 2, replace=False)
            a[i, k1], w[i, k1] = 224.0, -224.0 if i & 1 else 224.0
            a[i, k2], w[i, k2] = 2.0 ** -9, 2.0 ** -9
            st[i] = -a[i, k1] * w[i, k1]
    elif name == "nan":
        a = q8(rng.normal(0, 1.5, (n, 32)))
        w = q8(rng.normal(0, 0.25, (n, 32)))
        st = f16(rng.normal(0, 4, n))
        for i in range(n):
            (a if i % 2 == 0 else w)[i, rng.integers(32)] = np.nan
    elif name == "zeros":
        for i in range(n):
            kind = i % 4
            if kind == 0:                                         # all products signed zeros
                a[i] = q8(rng.normal(0, 1, 32)) * (rng.random(32) < 0.5)
                w[i] = np.where(a[i] == 0, q8(rng.normal(0, 1, 32)), 0.0) * _signs(rng, 32)
                a[i] = np.where(a[i] == 0, 0.0 * _signs(rng, 32), a[i])
            elif kind == 1:                                       # exact cancellation in pairs
                v = q8(rng.normal(0, 1, 16))
                a[i, :16], w[i, :16] = v, 1.0
                a[i, 16:], w[i, 16:] = -v, 1.0
            else:                                                 # -0 products only
                a[i] = -q8(np.abs(rng.normal(0, 1, 32)))
                w[i] = 0.0
            st[i] = rng.choice([0.0, -0.0, 2.0 ** -24, -1.0])
    else:
        raise ValueError(name)
    return st, a, w


def _row_any_sum(target, rng):
    """32 FP8 magnitudes summing exactly to target (any multiple of 2^-9 up to 32 x 448), random signs."""
    k = int(round(target * 512))
    assert k / 512 == target
    vals = []
    for _ in range(31):
        if k <= 0:
            break
        cand = FP8_POS[FP8_POS * 512 <= k]
        v = cand[-1]
        vals.append(v)
        k -= int(round(v * 512))
    assert k >= 0
    if k:
        last = k / 512
        assert last in FP8_POS, "target not reachable with 32 FP8 values"
        vals.append(last)
    row = np.array(vals + [0.0] * (32 - len(vals)))
    return row[rng.permutation(32)] * _signs(rng, 32)


S2A_FAMILIES = ("bound_at_63", "bound_below_63", "unbounded_inexact", "far_half", "big_inexact", "nan", "zeros")


def bound_f32(a, w):
    """(wsum, amax, bounded) per op as the kernels compute them: wsum the packers' exact float32 sum |w|, amax the
    decoded max code (a NaN code 0x7f is the largest), bounded = float32(wsum x amax) < 63."""
    codes = fp8_encode(q8(a))
    mc = (codes & 0x7F).max(1)
    amax = np.asarray(DECODE, np.float64)[mc].astype(np.float32)
    with np.errstate(invalid="ignore"):
        wsum = np.abs(q8(w)).sum(1)
    wsum32 = wsum.astype(np.float32)
    assert np.array_equal(wsum32.astype(np.float64), wsum, equal_nan=True)
    with np.errstate(invalid="ignore", over="ignore"):
        prod = (wsum32 * amax).astype(np.float32)
        bounded = prod < LIMIT
    return wsum32, amax, bounded, prod


def prepare(build, n, seed=11):
    sets = {}
    for name in M.FAMILIES:
        sets[name] = M.family(name, np.random.default_rng([seed, M.FAMILIES.index(name)]), n)
    for j, name in enumerate(S2A_FAMILIES):
        sets[name] = s2a_family(name, np.random.default_rng([seed, 100 + j]), max(64, n // 8))
    (p, ws, A), reach = largest_below_63()
    st_all, a_all, w_all, tags = [], [], [], []
    for name, (st, a, w) in sets.items():
        m = len(st)
        pad = (-m) % 16
        st_all.append(np.concatenate([f16(np.asarray(st, np.float64)), np.zeros(pad)]))
        a_all.append(np.concatenate([q8(np.asarray(a, np.float64)), np.zeros((pad, 32))]))
        w_all.append(np.concatenate([q8(np.asarray(w, np.float64)), np.zeros((pad, 32))]))
        tags += [name] * m + ["pad"] * pad
    st, a, w = np.concatenate(st_all), np.concatenate(a_all), np.concatenate(w_all)
    wsum, amax, bounded, prod = bound_f32(a, w)
    ref = np.array([M.op_exact(st[i:i + 1], a[i], w[i:i + 1])[0] for i in range(len(st))])
    kern = np.array([M.op_kernel(st[i:i + 1], a[i], w[i:i + 1])[0] for i in range(len(st))])
    with np.errstate(invalid="ignore"):
        true_abs = np.abs(a * w).sum(1)
        exact_bound = wsum.astype(np.float64) * amax.astype(np.float64)
    cpu = {"ops": int(np.sum(np.array(tags) != "pad")),
           "bound_covers_true_abs_sum": bool(np.all(~np.isfinite(true_abs) | (true_abs <= exact_bound))),
           "bounded_with_true_abs_sum_ge_64": int(np.sum(bounded & (true_abs >= 64))),
           "op8_cpu_model_differs_from_exact": int(np.sum(_bits16(kern) != _bits16(ref))),
           "largest_reachable_bound_below_63": {"product": float(p), "wsum": float(ws), "amax": float(A),
                                                "largest_float32_below_63_reachable": bool(reach)}}
    for name in S2A_FAMILIES[:2]:
        sel = np.array(tags) == name
        cpu[name + "_products"] = sorted(set(float(x) for x in prod[sel]))[:8]
        cpu[name + "_bounded"] = int(bounded[sel].sum())
    prep = {"st": st, "a": a, "w": w, "tags": tags, "wsum": wsum, "amax": amax, "bounded": bounded, "ref": ref,
            "cpu": cpu, "a_codes": fp8_encode(a), "w_codes": fp8_encode(w),
            "w16": q8(w).astype(np.float16).view(np.uint16)}
    with open(os.path.join(build, "l_q7_s2a_cases.pkl"), "wb") as f:
        pickle.dump(prep, f)
    print("prepared", len(st), "ops", cpu, flush=True)
    return cpu


def _bits16(v):
    """FP16 bits of values, a NaN's sign bit cleared (the accepted port exception)."""
    h = f16(np.asarray(v, np.float64)).astype(np.float16).view(np.uint16)
    nan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
    return np.where(nan, h & 0x7FFF, h)


def co_name(mut):
    return "l_q7_s2a_probe%s.co" % ("" if not mut else "_m%d" % mut)


def compile_probe(build, rocm):
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    clang = B.find_clang(rocm)
    os.makedirs(build, exist_ok=True)
    src = os.path.join(ROOT, "clean", "l", "kernels", "l_q7_s2a_probe.hip")
    for mut in (0,) + MUTANTS:
        cmd = B.device_command(clang, rocm, src, os.path.join(build, co_name(mut))) + [
            "-I", os.path.join(ROOT, "clean", "kernels")] + (["-DQ7_S2A_MUTANT=%d" % mut] if mut else [])
        B.run(cmd)


def kernel_names():
    return ["l_q7_s2a_probe"] + ["l_q7_s2a_probe_m%d" % m for m in MUTANTS]


def prepare_d3d12(build, rocm):
    from clean.l.runner.l_d3d12_check import prepare_replay
    return prepare_replay(build, rocm, kernel_names())


def _inputs(prep):
    return {"a": prep["a_codes"].astype(np.uint8).tobytes(), "w": prep["w_codes"].astype(np.uint8).tobytes(),
            "w16": prep["w16"].astype(np.uint16).tobytes(), "st": prep["st"].astype(np.float32).tobytes(),
            "wsum": prep["wsum"].astype(np.float32).tobytes()}


def run_hip(build, rocm, prep):
    from clean.runner.hip import Hip
    h = Hip(rocm)
    ins = _inputs(prep)
    n = len(prep["st"])
    bufs = {k: h.malloc(len(v)) for k, v in ins.items()}
    for k, v in ins.items():
        h.upload(bufs[k], v)
    po, pp = h.malloc(4 * n), h.malloc(4 * n)
    out = {}
    for mut in (0,) + MUTANTS:
        with open(os.path.join(build, co_name(mut)), "rb") as f:
            fn = h.get_function(h.load_module(f.read()), "l_q7_s2a_probe")
        h.memset(po, 0xA5, 4 * n)
        h.memset(pp, 0xA5, 4 * n)
        h.launch(fn, n // 16, 32, [ctypes.c_void_p(bufs[k]) for k in ("a", "w", "w16", "st", "wsum")] +
                 [ctypes.c_void_p(po), ctypes.c_void_p(pp)])
        h.sync()
        out[mut] = (np.frombuffer(h.download(po, 4 * n), np.float32).astype(np.float64),
                    np.frombuffer(h.download(pp, 4 * n), np.uint32))
    for p in list(bufs.values()) + [po, pp]:
        h.free(p)
    return out


def run_d3d12(build, prep):
    cdir = os.path.join(build, "replay-s2a-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(cdir)
    ins = _inputs(prep)
    n = len(prep["st"])
    for k, v in ins.items():
        with open(os.path.join(cdir, k + ".bin"), "wb") as f:
            f.write(v)
    lines = []
    for mut, kname in zip((0,) + MUTANTS, kernel_names()):
        lines += ["case m%d" % mut, "module l_q7_s2a_probe.co", "kernel %s %d 32" % (kname, n // 16)]
        lines += ["arg buf %s.bin" % k for k in ("a", "w", "w16", "st", "wsum")]
        lines += ["arg out %d m%d.o 90" % (4 * n, mut), "arg out %d m%d.p 90" % (4 * n, mut), ""]
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "256"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=100, creationflags=0x08000000)
    out = {}
    for mut in (0,) + MUTANTS:
        po, pp = os.path.join(cdir, "m%d.o" % mut), os.path.join(cdir, "m%d.p" % mut)
        if os.path.exists(po) and os.path.exists(pp):
            out[mut] = (np.frombuffer(open(po, "rb").read()[:4 * n], np.float32).astype(np.float64),
                        np.frombuffer(open(pp, "rb").read()[:4 * n], np.uint32))
    return out, {"replay_exit": r.returncode, "replay_tail": (r.stdout[-300:] + r.stderr[-300:]).strip(),
                 "identity": run_identity(build, ["kernel_replay.exe"] + ["dx12/%s.dxil" % k for k in kernel_names()])}


def score(prep, got):
    """Per set: value mismatches against the exact model (FP16 bits, NaN sign masked) and path mismatches."""
    tags = np.array(prep["tags"])
    real = tags != "pad"
    want, path = _bits16(prep["ref"]), prep["bounded"].astype(np.uint32)
    res = {}
    for mut, (y, p) in got.items():
        bad_v = (_bits16(y) != want) & real
        bad_p = (p != path) & real
        per = {}
        for name in dict.fromkeys(tags[real]):
            sel = tags == name
            per[name] = {"ops": int(sel.sum()), "value_mismatch": int(bad_v[sel].sum()),
                         "path_mismatch": int(bad_p[sel].sum()), "bounded": int((p[sel] == 1).sum())}
        res[mut] = {"value_mismatch": int(bad_v.sum()), "path_mismatch": int(bad_p.sum()), "sets": per,
                    "output_sha256": hashlib.sha256(_bits16(y).tobytes() + p.tobytes()).hexdigest()}
    return res


def run(build, rocm, d3d12):
    with open(os.path.join(build, "l_q7_s2a_cases.pkl"), "rb") as f:
        prep = pickle.load(f)
    res = {"what": "S2a primitive (two FP8 matrix instructions from 0 under wsum x amax < 63, exact op8 fallback, "
                   "FP16 start joined afterwards) against the exact model, adversarial sets, mutants",
           "fp8_op_model": "exact", "cpu": prep["cpu"], "nan_sign_masked": True, "nan_mask_control": mask_control(),
           "identity": run_identity(build, [co_name(m) for m in (0,) + MUTANTS] + ["l_q7_s2a_cases.pkl"]),
           "mutants": MUTANT_WHAT}
    hip = run_hip(build, rocm, prep)
    res["hip"] = score(prep, hip)
    ok = (res["hip"][0]["value_mismatch"] == 0 and res["hip"][0]["path_mismatch"] == 0 and res["cpu"]
          ["bound_covers_true_abs_sum"] and res["cpu"]["bounded_with_true_abs_sum_ge_64"] == 0
          and res["nan_mask_control"]["pass"])
    res["hip_mutants_caught"] = {m: res["hip"][m]["value_mismatch"] + res["hip"][m]["path_mismatch"] > 0
                                 for m in MUTANTS}
    if d3d12:
        dx, info = run_d3d12(build, prep)
        res["d3d12_run"] = info
        res["d3d12"] = score(prep, dx) if dx else {}
        res["d3d12_identical_to_hip"] = {m: (m in res["d3d12"] and
                                             res["d3d12"][m]["output_sha256"] == res["hip"][m]["output_sha256"])
                                         for m in (0,) + MUTANTS}
        res["d3d12_mutants_caught"] = {m: m in res["d3d12"] and res["d3d12"][m]["value_mismatch"] +
                                       res["d3d12"][m]["path_mismatch"] > 0 for m in MUTANTS}
        ok = ok and info["replay_exit"] == 0 and 0 in res["d3d12"] and res["d3d12"][0]["value_mismatch"] == 0 \
            and res["d3d12"][0]["path_mismatch"] == 0 and res["d3d12_identical_to_hip"][0]
    res["pass"] = bool(ok)
    return res


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
    ap.add_argument("--cases", type=int, default=4096)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.compile:
        compile_probe(a.build, a.rocm)
        return 0
    if a.prepare:
        os.makedirs(a.build, exist_ok=True)
        prepare(a.build, a.cases)
        return 0
    if a.prepare_d3d12:
        print("built", prepare_d3d12(a.build, a.rocm))
        return 0
    from clean.runner.hip import clean_process_environment
    clean_process_environment(a.rocm)
    res = run(a.build, a.rocm, a.d3d12)
    for be in ("hip", "d3d12"):
        if be in res and res[be]:
            for m, r in sorted(res[be].items()):
                print(be, "mutant" if m else "probe", m, "value mismatch", r["value_mismatch"], "path mismatch",
                      r["path_mismatch"], flush=True)
    print("D3D12 identical to HIP", res.get("d3d12_identical_to_hip"))
    print("PASS" if res["pass"] else "FAIL")
    if a.json:
        LR.write_l_record(res, a.json)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
