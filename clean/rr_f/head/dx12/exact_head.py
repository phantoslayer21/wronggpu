# SPDX-License-Identifier: MIT
"""Fast exact D0-R head (round 2): build, exactness sweep against the strict binary64 step, timing.

    python -m clean.rr_f.head.dx12.exact_head build  --work W [--variants S,X,...]
    python -m clean.rr_f.head.dx12.exact_head sweep  --work W --ref S --test X [--jobs 61] [--seed 0]
    python -m clean.rr_f.head.dx12.exact_head time --work W --variants E,XP

Variants (all with the runtime's exact row sums RR_ZEXACT=1): S = strict binary64 step (RR_STRICT_STEP 1, debug
outputs), X = the exact step on WMMA (RR_STRICT_STEP 2, debug outputs), E / XP / SP = production builds of the
default step E, of X and of S. The sweep runs one job per seed through both variants and compares every output region
(y2h, Z, the f16 HID / GATES / CANDW values, Cnet, HID, GATES, PREC, CANDW codes) byte for byte. Job modes cycle over
network-like weights, large gains, every finite e4m3 code (huge dynamic range: inexact WMMA sums, the fallback path),
every code including NaN, small integers (exact sums, f16 ties) and powers of two with cancellation.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from clean.rr_f.head.dx12 import run_dx12 as R
from clean.rr_f.head import block as B
from clean.rr_f.head import fp
from clean.rr_f.head import vectors as V

import build_dx12 as bd   # noqa: E402 (path set by run_dx12)

BASE = {"WG_BLOCK": 128, "WG_WAVE": 32, "RR_ZEXACT": 1}
VARS = {
    "S": dict(BASE, RR_STRICT_STEP=1, RR_DEBUG=1),
    "X": dict(BASE, RR_STRICT_STEP=2, RR_DEBUG=1),
    "N": dict(BASE, RR_STRICT_STEP=3, RR_DEBUG=1),
    "E": dict(BASE),
    "EN": dict(BASE, RR_HCOMPACT=0),             # step E without the compact down projection (r3-head control)
    "IPN": dict(BASE, RR_STRICT_STEP=4, RR_HCOMPACT=0),
    "XP": dict(BASE, RR_STRICT_STEP=2),
    "NP": dict(BASE, RR_STRICT_STEP=3),
    "SP": dict(BASE, RR_STRICT_STEP=1),
    "I": dict(BASE, RR_STRICT_STEP=4, RR_DEBUG=1),
    "IP": dict(BASE, RR_STRICT_STEP=4),
    "SF": dict(BASE, RR_STRICT_STEP=1, RR_DEBUG=1, RR_FIXUP=1),
    "SFP": dict(BASE, RR_STRICT_STEP=1, RR_FIXUP=1),
    "W": dict(BASE, RR_STRICT_STEP=4),           # entry rr_d0r_wplanes (ENTRY below)
}
ENTRY = {"W": "rr_d0r_wplanes"}
SRCS = ["head/dx12/rr_d0r.hlsl", "kernels/rr_step.hlsli", "kernels/rr_exact_math.hlsli", "kernels/rr_step_i.hlsli",
        "kernels/rr_step_n.hlsli"]
SIZES = {"1440pQ": (1728, 960, 1706, 960), "1440pP": (1280, 736, 1280, 720)}


def dxil_for(work, v):
    rrf = os.path.dirname(R.HEAD)
    h = hashlib.sha256(json.dumps([sorted(VARS[v].items()), ENTRY.get(v, "rr_d0r_head")]).encode())
    for s in SRCS:
        p = os.path.join(rrf, s)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return os.path.join(work, "dxil", f"head_{v}_{h.hexdigest()[:16]}.dxil")


def build(work, variants):
    os.makedirs(os.path.join(work, "dxil"), exist_ok=True)
    asm = bd.build_asm_tool(work)
    src = os.path.join(R.HERE, "rr_d0r.hlsl")
    inc = [os.path.join(R.CLEAN, "dx12", "kernels")]
    for v in variants:
        out = dxil_for(work, v)
        if not os.path.exists(out):
            t = time.time()
            bd.compile_hlsl(src, ENTRY.get(v, "rr_d0r_head"), out, VARS[v], includes=inc, asm_tool=asm)
            print(v, "built", f"{time.time() - t:.0f}s", flush=True)
        if VARS[v].get("RR_STRICT_STEP") == 1:
            continue                      # the binary64 kernels: no ISA dump (60 MB of text each)
        tag = os.path.basename(out)[:-5]
        isa = os.path.join(work, "isa")
        st = os.path.join(isa, f"{tag}.stats.csv")
        if not os.path.exists(st):
            bd.rga_isa(out, isa, tag)
        row = list(csv.DictReader(open(st, newline="")))[0]
        txt = open(os.path.join(isa, f"{tag}.isa.txt"), encoding="utf-8", errors="replace").read()
        print(v, {k: row[k] for k in ("USED_VGPRs", "SCRATCH_MEM", "ISA_SIZE")},
              "wmma", txt.count("v_wmma"), "f64", txt.count("_f64"), flush=True)


def host(work):
    return os.path.join(work, "base", "rr_host.exe") if os.path.exists(os.path.join(work, "base", "rr_host.exe")) \
        else os.path.join(work, "rr_host.exe")


def gen(seed, size):
    """One job's inputs; mode = seed % 6."""
    rng = np.random.default_rng(abs(1_000_003 * seed + 7))
    mode = seed % 6 if seed >= 0 else -seed     # negative seeds: extra modes (6: the sequence's synthetic weights)
    wt, ht, wr, hr = size
    m = V.load_manifest(R.VDIR)
    cname = "stage_edge" if mode in (2, 3) else "stage_basic"
    p = dict(R.case_params(next(c for c in m["cases"] if c["name"] == cname)), Wt=wt, Ht=ht, Wr=wr, Hr=hr,
             shift=B.recurrent_shift((2 * int(rng.integers(0, 4)), 2 * int(rng.integers(0, 4)))))
    codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], np.uint8)
    if mode == 6:                                     # gpu/sequence.py model_synthetic's D0-R weights
        from clean.rr_f.tests import synth
        W = synth.stage_weights(np.random.default_rng(5521 - seed), "IN0")
        x = fp.q8(rng.normal(0, 1.0, (ht, wt, 32)))
    elif mode == 0:
        W = V.make_weights(rng)
        x = fp.q8(rng.normal(0, 1.0, (ht, wt, 32)))
    elif mode == 1:
        W = V.make_weights(rng, gain=4.0)
        x = fp.q8(rng.normal(0, 6.0, (ht, wt, 32)))
    elif mode == 2:
        W = V.make_weights(rng, mode="allcodes")
        x = rng.choice(codes, size=(ht, wt, 32))
    elif mode == 3:
        W = V.make_weights(rng, mode="allcodes")
        x = rng.integers(0, 256, size=(ht, wt, 32)).astype(np.uint8)
        x[rng.random(x.shape) > 0.002] &= 0xF7       # NaN codes (0x7F / 0xFF) rare but present
    elif mode == 4:                                   # small integers: exact sums, f16 ties
        W = V.make_weights(rng)
        for k in ("Wq", "Wv", "Wo", "W1", "W2"):
            W[k] = fp.q8(rng.integers(-15, 16, W[k].shape).astype(np.float64))
        x = fp.q8(rng.integers(-15, 16, (ht, wt, 32)).astype(np.float64))
    else:                                             # powers of two, wide exponents: cancellation
        W = V.make_weights(rng)
        for k in ("Wq", "Wv", "Wo", "W1", "W2"):
            e = rng.integers(-9, 9, W[k].shape)
            W[k] = fp.q8(np.where(rng.random(W[k].shape) < .5, -1.0, 1.0) * np.exp2(e))
        e = rng.integers(-9, 8, (ht, wt, 32))
        x = fp.q8(np.where(rng.random((ht, wt, 32)) < .5, -1.0, 1.0) * np.exp2(e))
    hold = fp.r16(rng.normal(0, 1.0, (ht, wt, 4)))
    hold[..., 3] = 0.0
    hc = fp.r16(rng.normal(0, 1.2, (ht // 2, wt // 2, 4)))
    return p, W, x, hold, hc


def run_job(work, v, jobpath, outpath):
    r = subprocess.run([host(work), dxil_for(work, v), jobpath, outpath], capture_output=True, text=True,
                       timeout=1200)
    if r.returncode != 0:
        raise SystemExit(f"{v} {jobpath}: rc {r.returncode} " + r.stdout[-800:] + r.stderr[-800:])
    return r.stdout


def run_chain(work, chain, A, job, tmp, reps=0, lock=False):
    """Run the variants of chain ("I+SF": the integer pass, then the strict fixup) as dispatches on one arena."""
    from clean.rr_f.gpu import build as G
    words = np.frombuffer(job, "<u4")
    nargs = int(words[2]); args = words[3:3 + nargs].tolist(); groups = int(words[3 + nargs])
    ar = G.Arena(); ar.size = A.size
    ar.parts = [(off, np.frombuffer(data, np.uint8)) for off, data in A.init]
    vs = chain.split("+")
    kern = {i: (dxil_for(work, v), nargs) for i, v in enumerate(vs)}
    img, log = G.run(ar, kern, [(i, 8 if v in ENTRY else groups, args) for i, v in enumerate(vs)], lock=lock, reps=reps, workdir=tmp,
                     require_double=any(VARS[v].get("RR_STRICT_STEP") == 1 for v in vs))
    return img, log


def sweep(work, ref, test, jobs, seed0, size, tmp=None):
    d = tmp or os.path.join(work, "sweep")
    os.makedirs(d, exist_ok=True)
    tot, diff, per, marked, wins = 0, 0, {}, 0, 0
    tokens = 0
    for s in range(seed0, seed0 + jobs):
        p, W, x, hold, hc = gen(s, size)
        A, job = R.make_job(p, "stage", "dbg", W=W, x=x, hold=hold, hc=hc)
        outs = {}
        for v in (ref, test):
            img, _ = run_chain(work, v, A, job, d)
            outs[v] = {n: img[o:o + z] for n, o, z in A.reads}
        nd = 0
        for n in outs[ref]:
            if n == "flag":
                f = outs[test][n][:len(outs[test][n]) - R.GUARD].view("<u4").reshape(-1, 4)
                marked += int(np.count_nonzero(f.any(axis=1))); wins += f.shape[0]
                continue
            k = int(np.count_nonzero(outs[ref][n] != outs[test][n]))
            per[n] = per.get(n, 0) + k
            nd += k
            tot += outs[ref][n].size
        diff += nd
        tokens += p["Wt"] * p["Ht"]
        print(f"seed {s} mode {s % 6 if s >= 0 else -s} differing bytes {nd} marked windows {marked}/{wins}", flush=True)
    rep = dict(ref=ref, test=test, jobs=jobs, seed0=seed0, size=size, tokens=tokens, bytes=tot, differing=diff,
               per_region=per, marked_windows=marked, windows=wins,
               dxil={v: os.path.basename(dxil_for(work, v)) for v in (ref + "+" + test).split("+")})
    json.dump(rep, open(os.path.join(work, f"sweep_{ref}_{test}_{seed0}.json"), "w"), indent=1)
    print(json.dumps(rep))
    for f in os.listdir(d):
        if os.path.isfile(os.path.join(d, f)):
            os.remove(os.path.join(d, f))


def timing(work, variants, sizes, rounds, tmp=None):
    d = tmp or os.path.join(work, "time")
    os.makedirs(d, exist_ok=True)
    m = V.load_manifest(R.VDIR)
    basic = R.case_params(next(c for c in m["cases"] if c["name"] == "stage_basic"))
    res = {}
    for sz in sizes:
        wt, ht, wr, hr = SIZES[sz]
        rng = np.random.default_rng(1440)
        p = dict(basic, Wt=wt, Ht=ht, Wr=wr, Hr=hr, shift=B.recurrent_shift((6, 6)))
        W = V.make_weights(rng)
        x = fp.q8(rng.normal(0, 1.0, (ht, wt, 32)))
        hold = fp.r16(rng.normal(0, 1.0, (ht, wt, 4)))
        hold[..., 3] = 0.0
        hc = fp.r16(rng.normal(0, 1.2, (ht // 2, wt // 2, 4)))
        A, job = R.make_job(p, "stage", "prod", W=W, x=x, hold=hold, hc=hc, reps=0)
        for rd in range(rounds):          # interleaved rounds: the clock ramps; report the best median
            for v in variants:
                _, log = run_chain(work, v, A, job, d, reps=(5 if "SP" == v else 200))
                tl = [l for l in log.splitlines() if l.startswith("time ") or l.startswith("total")]
                key = f"{sz}/{v}"
                res.setdefault(key, []).append(tl)
                print(key, f"round {rd}", " | ".join(tl)[:300], flush=True)
    print(json.dumps(res))
    json.dump(res, open(os.path.join(work, "timing.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "sweep", "time"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--variants", default="S,X")
    ap.add_argument("--ref", default="S")
    ap.add_argument("--test", default="X")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="1440pQ")
    ap.add_argument("--sizes", default="1440pQ,1440pP")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tmp", help="directory for the sweep's job / output files (large)")
    a = ap.parse_args()
    if a.step == "build":
        build(a.work, a.variants.split(","))
    elif a.step == "sweep":
        sweep(a.work, a.ref, a.test, a.jobs, a.seed, SIZES[a.size], a.tmp)
    else:
        timing(a.work, a.variants.split(","), a.sizes.split(","), a.rounds, a.tmp)


if __name__ == "__main__":
    main()
