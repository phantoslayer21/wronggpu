# SPDX-License-Identifier: MIT
"""Model L (issue #89): the FP32-sequential arithmetic of one FP8 matrix op (what the FP16-form kernels compute:
every product exact in float32, added one at a time in float32 in ascending k with round-to-nearest-even, the FP16
start value added last in float32, one conversion to FP16) against the oracle's exact model (clean/l/reference/oracle.py
MM8_MODELS "exact": the exact sum rounded once to FP16).

Adversarial families (each a set of single 32-term ops):
- typical: random activations and weights at typical network scales, any start;
- cancellation: pairs of large products of opposite sign with small terms between them;
- dynamic_range: products spread over the whole FP8 x FP8 range (2^-18 ... 448^2);
- near_tie: the exact sum placed on an FP16 rounding tie, plus or minus one small product;
- subnormal: FP8 subnormal operands (2^-9 ... 7 x 2^-9) with normal ones.
Every synthetic op where the two differ is recorded (the 32 FP8 codes of each operand and the start's FP16 bits) with
the difference in FP16 steps.

Synthetic values only, by default. --real (with --frames <build>: a local recorded run of the real model) adds the same
comparison on every FP8 op of Model L's own stages for that run's frames, and each op's margin: the distance of its
exact sum from the nearest FP16 rounding boundary divided by the bound on the FP32-sequential error (the two can differ
only below 1). Real-model values never reach the repository (the synthetic-values rule):
- a real run's intermediates (the probe's sample of real ops) go only to --private-out, a folder OUTSIDE the
  repository (a path inside it is refused);
- every record a real run writes (--json) is aggregate-only: counts, rates, margins and hashes, never an operand, a
  start value or a result of a real op (assert_aggregate_only refuses one that is not).

    python clean/l/runner/l_mm8_f32_check.py --json out.json [--cases 20000]                             # synthetic
    python clean/l/runner/l_mm8_f32_check.py --json out.json --real --frames <build> --private-out <dir>
    python clean/l/runner/l_mm8_f32_check.py --wmma-prepare <dir> --json x [--real --frames <build> --private-out <d>]
    python clean/l/runner/l_mm8_f32_check.py --wmma-run <dir> --json out.json [--cases-dir <d>]  # GPU (a GPU slot)
"""
import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.reference.fp8 import f16, fp8_encode, q8  # noqa: E402

# Operand-level fields of an op: allowed in a record only for synthetic sets (assert_aggregate_only)
OP_FIELDS = ("start_f16_bits", "a_fp8_codes", "w_fp8_codes", "exact", "f32seq", "exact_sum", "instruction_f32")
REAL_SETS = ("real_frames_sample",)


def private_dir(path):
    """A real run's private output folder: refused when it is inside the repository (the synthetic-values rule)."""
    from clean.l.runner.l_private import private_dir as shared
    return shared(path)


def assert_aggregate_only(res, path="record"):
    """A real run's record: no operand-level field under a real set (or anywhere under a "frames" study)."""
    if isinstance(res, dict):
        for k, v in res.items():
            real = k in REAL_SETS or k == "frames"
            if real and _has_op_fields(v):
                raise SystemExit(f"{path}.{k}: holds operand-level values of real ops; a record is aggregate-only")
            assert_aggregate_only(v, path + "." + str(k))


def _has_op_fields(v):
    if isinstance(v, dict):
        return any(k in OP_FIELDS for k in v) or any(_has_op_fields(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_op_fields(x) for x in v)
    return False


FP8_VALUES = np.unique(np.abs(np.nan_to_num(np.asarray(__import__("clean.reference.fp8", fromlist=["DECODE"]).DECODE),
                                            nan=0.0)))


def op_f32seq(start, a8, W8):
    """start [..., n] FP16 values, a8 [..., k], W8 [n, k] (k <= 32) -> FP16 [..., n]: products in float32 (exact),
    summed in float32 in ascending k, + start in float32, one conversion to FP16."""
    a8 = np.asarray(a8, np.float32)
    W8 = np.asarray(W8, np.float32)
    acc = np.zeros(np.broadcast_shapes(a8.shape[:-1] + (W8.shape[0],), np.shape(start)), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(a8.shape[-1]):
            acc = (acc + a8[..., k:k + 1] * W8[:, k]).astype(np.float32)
        acc = (acc + np.asarray(start, np.float32)).astype(np.float32)
    return f16(acc.astype(np.float64))


ABS_FAST = np.float32(63.0)   # fast path while the float32 sum of |products| stays below this (true sum < 2^6)


def _two_sum(a, b):
    """Error-free float32 sum: a + b = s + e exactly (e exact in float32)."""
    s = (a + b).astype(np.float32)
    bb = (s - a).astype(np.float32)
    e = ((a - (s - bb).astype(np.float32)).astype(np.float32) + (b - bb).astype(np.float32)).astype(np.float32)
    return s, e


def op_kernel(start, a8, W8):
    """The FP16-form kernels' arithmetic of one FP8 op (exact): products in float32 (exact); their float32 sum in
    ascending k with the float32 sum of |products| alongside. While that stays below 63 the sum is exact (every
    partial sum a multiple of 2^-18 below 2^6); otherwise the products are summed again as a TwoSum chain (s, c), c
    the exact sum of the errors (multiples of 2^-18 below 8). The FP16 start is added with error-free sums into a
    head z and a tail t (z = fl(exact total), |t| <= half an ulp of z); RN16(z), and where z is exactly an FP16
    midpoint the tail's sign picks the side (no tail: ties to even)."""
    a8 = np.asarray(a8, np.float32)
    W8 = np.asarray(W8, np.float32)
    shape = np.broadcast_shapes(a8.shape[:-1] + (W8.shape[0],), np.shape(start))
    st = np.broadcast_to(np.asarray(start, np.float32), shape)
    with np.errstate(over="ignore", invalid="ignore"):
        acc = np.zeros(shape, np.float32)
        absacc = np.zeros(shape, np.float32)
        prods = []
        for k in range(a8.shape[-1]):
            p = (a8[..., k:k + 1] * W8[:, k]).astype(np.float32)
            prods.append(p)
            acc = (acc + p).astype(np.float32)
            absacc = (absacc + np.abs(p)).astype(np.float32)
        s, c = acc, np.zeros(shape, np.float32)
        slow = ~(absacc < ABS_FAST)
        if slow.any():
            s2, c2 = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
            for p in prods:
                s2, e = _two_sum(s2, p)
                c2 = (c2 + e).astype(np.float32)
            s, c = np.where(slow, s2, s), np.where(slow, c2, c)
        u, e1 = _two_sum(s, st)
        v, e2 = _two_sum(e1, c)
        z, e3 = _two_sum(u, v)
        t = (e3 + e2).astype(np.float32)
        z, t = _two_sum(z, t)
        r16 = z.astype(np.float16)
        r = r16.astype(np.float64)
        zd = z.astype(np.float64)
        fin = np.isfinite(zd) & np.isfinite(t)
        # the FP16 neighbour of r on z's side (beyond 65504: the overflow boundary 65536 stands for infinity)
        toward = np.where(zd > r, np.float16(np.inf), np.float16(-np.inf)).astype(np.float16)
        nb = np.nextafter(np.where(np.isinf(r16), np.copysign(np.float16(65504), r16), r16), toward).astype(np.float64)
        rr = np.where(np.isinf(r), np.copysign(65536.0, r), r)
        nb = np.where(np.isinf(nb), np.copysign(65536.0, nb), nb)
        nb = np.where(np.isinf(r), np.copysign(65504.0, r), nb)   # r = inf: z on the boundary 65520, below it 65504
        mid = fin & (zd != r) & (zd == (rr + nb) / 2.0) & (t != 0)
        lo, hi = np.minimum(rr, nb), np.maximum(rr, nb)
        cand = np.where(t > 0, hi, lo)
        cand = np.where(np.abs(cand) >= 65536.0, np.copysign(np.inf, cand), cand)
        out = np.where(mid, cand, r)
        plain = (acc + st).astype(np.float32).astype(np.float16).astype(np.float64)
        return np.where(np.isfinite(acc) & np.isfinite(st), out, plain)


def op_exact(start, a8, W8):
    return L._op8(np.asarray(start, np.float64), np.asarray(a8, np.float64), np.asarray(W8, np.float64), "exact")


def f16_steps(a, b):
    ia = np.asarray(a, np.float64).astype(np.float16).view(np.int16).astype(np.int64)
    ib = np.asarray(b, np.float64).astype(np.float16).view(np.int16).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFF), ib)
    return np.abs(ia - ib)


def margin(start, a8, W8):
    """Per output: |exact sum - nearest FP16 rounding boundary| / (the FP32-sequential error bound: 32 additions, each
    off by at most half an ulp of the running sum's magnitude bound in float32). Below 1 the two may differ."""
    a8 = np.asarray(a8, np.float64)
    W8 = np.asarray(W8, np.float64)
    prods = a8[..., None, :] * W8                                     # [..., n, k]
    s = prods.sum(-1) + np.asarray(start, np.float64)
    bound_mag = np.abs(prods).sum(-1) + np.abs(np.asarray(start, np.float64))
    err = (prods.shape[-1] + 1) * bound_mag * 2.0 ** -24
    r = f16(s)
    with np.errstate(over="ignore", invalid="ignore"):
        up = np.nextafter(r.astype(np.float16), np.float16(np.inf)).astype(np.float64)
        dn = np.nextafter(r.astype(np.float16), np.float16(-np.inf)).astype(np.float64)
        dist = np.minimum(np.abs(s - (r + up) / 2), np.abs(s - (r + dn) / 2))
        return np.where(err > 0, dist / np.maximum(err, 1e-300), np.inf)


# ---- adversarial families ---------------------------------------------------------------------------------------------

def _fp8(rng, shape, lo_exp, hi_exp):
    """FP8 values log-uniform in magnitude between 2^lo_exp and 2^hi_exp, random sign."""
    mag = 2.0 ** rng.uniform(lo_exp, hi_exp, shape)
    return q8(mag * rng.choice([-1.0, 1.0], shape))


def family(name, rng, n):
    """n ops: (start [n], a [n, 32], w [n, 32]) of the named family (one output per op)."""
    if name == "typical":
        a = q8(rng.normal(0, 1.5, (n, 32)))
        w = q8(rng.normal(0, 0.25, (n, 32)))
        st = f16(rng.normal(0, 4, n))
    elif name == "cancellation":
        a = np.zeros((n, 32))
        w = np.zeros((n, 32))
        big = _fp8(rng, (n, 8), 4, 8.8)
        a[:, 0:8], w[:, 0:8] = big, _fp8(rng, (n, 8), 4, 8.8)
        a[:, 8:16], w[:, 8:16] = -a[:, 0:8], w[:, 0:8]                 # the same products, opposite sign
        a[:, 16:], w[:, 16:] = _fp8(rng, (n, 16), -9, 0), _fp8(rng, (n, 16), -9, 0)
        perm = np.argsort(rng.random((n, 32)), axis=1)
        a, w = np.take_along_axis(a, perm, 1), np.take_along_axis(w, perm, 1)
        st = f16(_fp8(rng, n, -12, 2))
    elif name == "dynamic_range":
        a = _fp8(rng, (n, 32), -9, 8.8)
        w = _fp8(rng, (n, 32), -9, 8.8)
        st = f16(rng.normal(0, 1, n) * 2.0 ** rng.uniform(-14, 15, n))
    elif name == "near_tie":
        a = q8(rng.normal(0, 1.5, (n, 32)))
        w = q8(rng.normal(0, 0.25, (n, 32)))
        a[:, 31], w[:, 31] = 0.0, 0.0
        s = (a * w).sum(1)
        r = f16(s + 8.0)                                               # a target FP16 value near the sum
        ulp = np.abs(np.nextafter(r.astype(np.float16), np.float16(np.inf)).astype(np.float64) - r)
        tie = r + ulp / 2
        st = f16(tie - s)                                              # start so that the sum sits near a tie
        a[:, 31] = _fp8(rng, n, -9, -3)
        w[:, 31] = _fp8(rng, n, -9, -3)
    elif name == "start_midpoint":
        # a large FP16 start and products summing to half its FP16 ulp plus or minus one tiny product: the float32
        # addition of the start lands on the FP16 midpoint (random synthetic operands)
        e = rng.integers(8, 15, n)                                     # start in [2^e, 2^(e+1))
        st = f16(rng.uniform(1, 2, n) * 2.0 ** e * rng.choice([-1.0, 1.0], n))
        half = 2.0 ** (e - 11)                                         # half an FP16 ulp there
        a = np.zeros((n, 32))
        w = np.zeros((n, 32))
        a[:, 0], w[:, 0] = q8(half * np.sign(st)), 1.0
        tiny = 2.0 ** rng.integers(-18, -12, n)
        a[:, 1] = q8(np.sqrt(tiny))
        w[:, 1] = q8(np.sqrt(tiny) * rng.choice([-1.0, 1.0], n))
        a[:, 2:8], w[:, 2:8] = q8(rng.normal(0, 2.0 ** -6, (n, 6))), 0.0
    elif name == "subnormal":
        a = q8(rng.integers(-7, 8, (n, 32)) * 2.0 ** -9)
        w = np.where(rng.random((n, 32)) < 0.5, q8(rng.integers(-7, 8, (n, 32)) * 2.0 ** -9),
                     q8(rng.normal(0, 2, (n, 32))))
        st = f16(rng.normal(0, 2.0 ** -12, n))
    else:
        raise ValueError(name)
    return st, a, w


FAMILIES = ("typical", "cancellation", "dynamic_range", "near_tie", "start_midpoint", "subnormal")


def compare_family(name, rng, n, keep=50, op=None):
    op = op or op_f32seq
    st, a, w = family(name, rng, n)
    got = np.array([op(st[i:i + 1], a[i], w[i:i + 1])[0] for i in range(n)])
    ref = np.array([op_exact(st[i:i + 1], a[i], w[i:i + 1])[0] for i in range(n)])
    steps = f16_steps(got, ref)
    nan_same = np.isnan(got) & np.isnan(ref)
    diff = (steps > 0) & ~nan_same
    rows = []
    for i in np.flatnonzero(diff)[:keep]:
        rows.append({"start_f16_bits": int(np.array([st[i]], np.float16).view(np.uint16)[0]),
                     "a_fp8_codes": fp8_encode(a[i]).tolist(), "w_fp8_codes": fp8_encode(w[i]).tolist(),
                     "exact": float(ref[i]), "f32seq": float(got[i]), "f16_steps": int(steps[i])})
    return {"cases": int(n), "differ": int(diff.sum()), "differ_over_1": int(((steps > 1) & ~nan_same).sum()),
            "max_f16_steps": int(steps[~nan_same].max()) if (~nan_same).any() else 0,
            "examples": rows}


# ---- a local recorded run (read at run time; nothing of it is in the repository) -----------------------------------

def frames_study(build):
    """Every FP8 op of Model L's own stages (both blocks, input linears, projections) on the recorded run's frames:
    FP32-sequential vs exact, and the smallest margin."""
    from clean.l.reference import table as T
    from clean.l.runner import l_frame_check as F
    rec = pickle.load(open(F.rec_path(build, "L", "stages"), "rb"))
    params = pickle.load(open(os.path.join(F.model_dir(build), "params.pkl"), "rb"))["L"]
    stats = {"ops": 0, "results": 0, "differ": 0, "kernel_differ": 0, "min_margin": float("inf"), "margin_below_1": 0,
             "margin_below_10": 0}
    exact = L._op8

    def spy(start, a8, W8, model):
        out = exact(start, a8, W8, model)
        k = a8.shape[-1]
        a = a8.reshape(-1, k)
        st = np.broadcast_to(np.asarray(start, np.float64), a8.shape[:-1] + (W8.shape[0],)).reshape(-1, W8.shape[0])
        ref = out.reshape(-1, W8.shape[0])
        d = f16_steps(op_f32seq(st, a, W8), ref)
        dk = f16_steps(op_kernel(st, a, W8), ref)
        m = margin(st, a, W8)
        stats["ops"] += 1
        stats["results"] += int(d.size)
        stats["differ"] += int(np.sum(d > 0))
        stats["kernel_differ"] += int(np.sum(dk > 0))
        fin = np.isfinite(m)
        if fin.any():
            stats["min_margin"] = min(stats["min_margin"], float(m[fin].min()))
        stats["margin_below_1"] += int(np.sum(m < 1))
        stats["margin_below_10"] += int(np.sum(m < 10))
        return out

    t = copy.deepcopy(T.TABLE)
    L._op8 = spy
    try:
        F.compare_model(rec, params, table=t)
    finally:
        L._op8 = exact
    stats["identity"] = rec.get("identity")
    return stats


# ---- the matrix instruction (GPU probe, a side measurement) ------------------------------------------------------------

def wmma_prepare(build, n, seed=7, real_build=None, real_max=200000, private_out=None):
    """CPU: the probe's inputs (every family, and with real_build a sample of a local recorded run's 32-term FP8 ops)
    with the exact sums and the exact model's FP16 results, saved to <build>/l_wmma_cases.pkl; with a real sample, to
    <private_out>/l_wmma_cases.pkl only (a folder outside the repository: private_dir)."""
    if real_build:
        if not private_out:
            raise SystemExit("--real: the probe's real sample needs --private-out (a folder outside the repository)")
        build = private_dir(private_out)
    sets = {}
    for name in FAMILIES:
        st, a, w = family(name, np.random.default_rng([seed, FAMILIES.index(name)]), n)
        sets[name] = (st, a, w)
    identity = None
    if real_build:
        from clean.l.reference import table as T
        from clean.l.runner import l_frame_check as F
        rec = pickle.load(open(F.rec_path(real_build, "L", "stages"), "rb"))
        params = pickle.load(open(os.path.join(F.model_dir(real_build), "params.pkl"), "rb"))["L"]
        identity = rec.get("identity")
        sample = {"st": [], "a": [], "w": []}
        exact = L._op8

        def spy(start, a8, W8, model):
            out = exact(start, a8, W8, model)
            if a8.shape[-1] == 32 and len(sample["st"]) < real_max:
                aa = a8.reshape(-1, 32)
                st_ = np.broadcast_to(np.asarray(start, np.float64), a8.shape[:-1] + (W8.shape[0],)).reshape(
                    -1, W8.shape[0])
                for i in range(min(64, aa.shape[0])):
                    j = i % W8.shape[0]
                    sample["st"].append(st_[i, j])
                    sample["a"].append(aa[i])
                    sample["w"].append(W8[j])
            return out

        L._op8 = spy
        try:
            F.compare_model(rec, params, table=copy.deepcopy(T.TABLE))
        finally:
            L._op8 = exact
        sets["real_frames_sample"] = tuple(np.array(sample[k][:real_max]) for k in ("st", "a", "w"))
    cases = {}
    for name, (st, a, w) in sets.items():
        exact_sum = (np.asarray(a, np.float64) * np.asarray(w, np.float64)).sum(1) + np.asarray(st, np.float64)
        ref16 = np.array([op_exact(st[i:i + 1], a[i], w[i:i + 1])[0] for i in range(len(st))])
        cases[name] = {"st": st, "a": a, "w": w, "exact_sum": exact_sum, "ref16": ref16}
        print("prepared", name, len(st), flush=True)
    with open(os.path.join(build, "l_wmma_cases.pkl"), "wb") as f:
        pickle.dump({"cases": cases, "identity": identity}, f)


def wmma_run(build, rocm, cases_dir=None):
    """GPU: the probe kernel (clean/l/kernels/l_wmma_probe.hip, <build>/l_wmma_probe.co) on every prepared set: per op,
    whether the instruction's float32 result equals the exact sum rounded once to float32, and whether its FP16 rounding
    equals the exact model's result."""
    import ctypes
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    cases_pkl = os.path.join(cases_dir or build, "l_wmma_cases.pkl")
    with open(cases_pkl, "rb") as f:
        prep = pickle.load(f)
    h = Hip(rocm)
    with open(os.path.join(build, "l_wmma_probe.co"), "rb") as f:
        fn = h.get_function(h.load_module(f.read()), "l_wmma_probe")
    res = {}
    for name, c in prep["cases"].items():
        st, a, w = c["st"], c["a"], c["w"]
        m = len(st)
        ab, bb = fp8_encode(a).tobytes(), fp8_encode(w).tobytes()
        sb = np.asarray(st, np.float32).tobytes()
        pa, pb, ps, po = h.malloc(len(ab)), h.malloc(len(bb)), h.malloc(len(sb)), h.malloc(4 * m)
        h.upload(pa, ab)
        h.upload(pb, bb)
        h.upload(ps, sb)
        h.launch(fn, -(-m // 16), 32, [ctypes.c_void_p(pa), ctypes.c_void_p(pb), ctypes.c_void_p(ps),
                                      ctypes.c_void_p(po), ctypes.c_int(m)])
        h.sync()
        got = np.frombuffer(h.download(po, 4 * m), np.float32).astype(np.float64)
        # S2a's use: the accumulator starts at 0 (the FP16 start is added exactly afterwards), so the instruction
        # computes the product sum alone; compare that with the exact product sum
        h.upload(ps, bytes(len(sb)))
        h.launch(fn, -(-m // 16), 32, [ctypes.c_void_p(pa), ctypes.c_void_p(pb), ctypes.c_void_p(ps),
                                      ctypes.c_void_p(po), ctypes.c_int(m)])
        h.sync()
        got0 = np.frombuffer(h.download(po, 4 * m), np.float32).astype(np.float64)
        for ptr in (pa, pb, ps, po):
            h.free(ptr)
        fin = np.isfinite(c["exact_sum"])
        psum = c["exact_sum"] - np.asarray(st, np.float64)                   # exact in float64
        p_exact = got0 == psum.astype(np.float32).astype(np.float64)
        aa0, ww0 = np.abs(np.asarray(a, np.float64)), np.abs(np.asarray(w, np.float64))
        bt0, bp0 = (aa0 * ww0).sum(1), ww0.sum(1) * aa0.max(1)
        zero_start = {"ops": int(fin.sum()), "f32_not_exact_product_sum": int(np.sum(fin & ~p_exact)),
                      "bounded_true_63": int(np.sum(fin & (bt0 < 63))),
                      "bounded_true_63_not_exact": int(np.sum(fin & (bt0 < 63) & ~p_exact)),
                      "bounded_pre_63": int(np.sum(fin & (bp0 < 63))),
                      "bounded_pre_63_not_exact": int(np.sum(fin & (bp0 < 63) & ~p_exact))}
        f32_exact = got == c["exact_sum"].astype(np.float32).astype(np.float64)
        d16 = f16_steps(f16(got), c["ref16"])
        rows = [] if name in REAL_SETS else [   # a real op's values never go into a record
            {"start_f16_bits": int(np.array([st[i]], np.float16).view(np.uint16)[0]),
             "a_fp8_codes": fp8_encode(a[i]).tolist(), "w_fp8_codes": fp8_encode(w[i]).tolist(),
             "exact_sum": float(c["exact_sum"][i]), "instruction_f32": float(got[i])}
            for i in np.flatnonzero(fin & (d16 > 0))[:20]]
        # S2a's exactness argument: while the bound stays below 63, every partial sum is a multiple of 2^-18 below 2^6,
        # exact in float32 in any order, so the instruction must give the exact sum. Two bounds: the true sum |a w| and
        # the one the kernels can afford (precomputed sum |w| of the row, times the op's max |a|).
        aa, ww = np.abs(np.asarray(a, np.float64)), np.abs(np.asarray(w, np.float64))
        b_true = (aa * ww).sum(1) + np.abs(np.asarray(st, np.float64))
        b_pre = ww.sum(1) * aa.max(1)
        res[name] = {"ops": int(fin.sum()), "f32_not_exact_sum": int(np.sum(fin & ~f32_exact)),
                     "fp16_differs_from_exact_model": int(np.sum(fin & (d16 > 0))),
                     "bounded_true_63": int(np.sum(fin & (b_true < 63))),
                     "bounded_true_63_not_exact": int(np.sum(fin & (b_true < 63) & ~f32_exact)),
                     "bounded_pre_63": int(np.sum(fin & (b_pre < 63))),
                     "bounded_pre_63_not_exact": int(np.sum(fin & (b_pre < 63) & ~f32_exact)),
                     "zero_start": zero_start, "examples": rows}
        print(f"wmma {name:18s}", {k: v for k, v in res[name].items() if k != "examples"}, flush=True)
    if prep["identity"]:
        res["real_frames_identity"] = prep["identity"]
    from clean.l.runner.l_q7_check import run_identity
    res["identity"] = run_identity(build, ["l_wmma_probe.co"])
    import hashlib
    with open(cases_pkl, "rb") as f:
        res["identity"]["l_wmma_cases_pkl_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", help="with --real: a build with a local recorded l_frame_check run of the real model")
    ap.add_argument("--real", action="store_true", help="also the local recorded run (--frames; needs --private-out)")
    ap.add_argument("--private-out", help="--real: the folder (outside the repository) for a real run's intermediates")
    ap.add_argument("--cases-dir", help="--wmma-run: the folder of l_wmma_cases.pkl (default the build folder)")
    ap.add_argument("--wmma-prepare", metavar="BUILD", help="CPU: the matrix-instruction probe's inputs into BUILD")
    ap.add_argument("--wmma-run", metavar="BUILD", help="GPU: the probe (BUILD/l_wmma_probe.co) on the prepared inputs")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    a = ap.parse_args(argv)
    if a.frames and not a.real:
        raise SystemExit("--frames reads a local recorded run of the real model: say --real (with --private-out)")
    if a.real and not (a.frames and a.private_out):
        raise SystemExit("--real needs --frames <build> and --private-out <folder outside the repository>")
    if a.real:
        private_dir(a.private_out)
    if a.wmma_prepare:
        wmma_prepare(a.wmma_prepare, a.cases, a.seed, a.frames if a.real else None, private_out=a.private_out)
        return 0
    if a.wmma_run:
        from clean.l.runner import l_record as LR
        res = {"what": "RDNA 4 FP8 matrix instruction (two 16-term steps, float32 accumulator from the start) against "
                       "the exact model, a side measurement", "fp8_op_model": "exact",
               "probe": wmma_run(a.wmma_run, a.rocm, a.cases_dir)}
        assert_aggregate_only(res)
        LR.write_l_record(res, a.json)
        return 0
    rng = np.random.default_rng(a.seed)
    res = {"what": "FP32-sequential FP8 op (the kernels' arithmetic) against the exact model", "seed": a.seed,
           "families": {}}
    for name in FAMILIES:
        for opname, op in (("f32seq", op_f32seq), ("kernel", op_kernel)):
            r = compare_family(name, np.random.default_rng([a.seed, FAMILIES.index(name)]), a.cases, op=op)
            res["families"].setdefault(name, {})[opname] = r
            print(f"{name:14s} {opname:7s} cases {r['cases']}  differ {r['differ']}  more than 1 step "
                  f"{r['differ_over_1']}  max {r['max_f16_steps']}", flush=True)
    if a.real:
        res["frames"] = frames_study(a.frames)
        print("local recorded run:", json.dumps({k: v for k, v in res["frames"].items() if k != "identity"}))
    from clean.l.runner import l_record as LR
    assert_aggregate_only(res)
    LR.write_l_record(res, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
