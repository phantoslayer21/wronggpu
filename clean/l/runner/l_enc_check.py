# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the input encoder's value kernel (`lev/input_encoder`: clean/kernels/encoder.hip
built with WG_ENC_VALUES_OUT=1 as l_enc_values.co) against the oracle, synthetic frames only.

    python clean/l/runner/l_enc_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--seeds N]
        --json out.json

<dir>/l_enc_values.co comes from clean/l/runner/l_c32_check.py --compile-only. The GPU part runs in a GPU slot.
Output: counts only, no timing.

Per frame: the halo map q [fh + 4][fw + 4][32] against the oracle's dithered values (clean/l/reference/oracle.py
encode, map "q"): values that differ, the largest difference in FP8 steps, and Model M's per-stage rule
(clean/l/runner/l_c32_check.py passes). Bytes 20..31 of every pixel must be zero. The motion surfaces m_near, m_far,
m_center (inside R) are compared with the oracle's outputs, which are Model M's, under Model M's rule for them (at most 1
FP16 ulp, at least 99.5 % exact; clean/tests/test_encoder_emu.py).
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.reference import encoder as ME  # noqa: E402
from clean.reference.fp8 import fp8_decode  # noqa: E402
from clean.runner import enc0 as E0  # noqa: E402

# (rw, rh): render sizes with padding to multiples of 32, odd sizes and both reset states
CASES = [(40, 24, False), (37, 19, False), (64, 32, True), (90, 50, False),
         (17, 17, False), (17, 24, True)]      # width 17 (padded 32): halo columns 32, 33 both mirror to column 0


def run_frame(hip, mod, x, enc):
    h = hip
    prm = E0.frame_params(x)
    data = E0.device_inputs(x)
    halo = T.block_halo(T.TABLE, "first")
    mw, mh = prm.fw + 2 * halo, prm.fh + 2 * halo
    ntok = (mw // 2) * (mh // 2)
    bufs = {}
    for k, v in data.items():
        b = v if isinstance(v, bytes) else np.ascontiguousarray(v).tobytes()
        bufs[k] = h.malloc(len(b))
        h.upload(bufs[k], b)
    blob = LP.pack_values(enc)
    pw = h.malloc(len(blob))
    h.upload(pw, blob)
    eb = np.array([x["exposure"], x["exposure_prev"]], dtype=np.float32).tobytes()
    pe = h.malloc(len(eb))
    h.upload(pe, eb)
    pq = h.malloc(mw * mh * 32)
    h.memset(pq, 0xEE, mw * mh * 32)                     # unwritten bytes would show
    npx = prm.rw * prm.rh * 4
    pm = [h.malloc(npx) for _ in range(3)]
    for p in pm:
        h.memset(p, 0, npx)
    args = [ctypes.c_void_p(bufs[k]) for k in ("color", "depth", "motion", "hist", "latent", "mhist")] + \
           [ctypes.c_void_p(pq)] + [ctypes.c_void_p(p) for p in pm] + [ctypes.c_void_p(pe), ctypes.c_void_p(pw), prm]
    h.launch(h.get_function(mod, "input_encoder"), -(-ntok // 128), 256, args)
    h.sync()
    q = np.frombuffer(h.download(pq, mw * mh * 32), np.uint8).reshape(mh, mw, 32)
    mot = [np.frombuffer(h.download(p, npx), np.float16).reshape(prm.rh, prm.rw, 2) for p in pm]
    for p in list(bufs.values()) + [pw, pe, pq] + pm:
        h.free(p)
    return q, mot


def run(build, rocm, seeds=(89,)):
    from clean.runner.hip import Hip
    hip = Hip(rocm)
    with open(os.path.join(build, "l_enc_values.co"), "rb") as f:
        mod = hip.load_module(f.read())
    res = {"frames": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        params = L.random_params(rng)
        for rw, rh, reset in CASES:
            x = ME.random_inputs(rng, rw=rw, rh=rh, reset=reset)
            q, mot = run_frame(hip, mod, x, params["enc0"])
            out = L.encode(x, params, parity=0, keep=True)
            ref = out["maps"]["q"]
            got = fp8_decode(q[..., :20])
            m = C32.metrics(got, ref)
            m.update({"case": [rw, rh, reset], "pad_bytes_nonzero": int(np.count_nonzero(q[..., 20:]))})
            mdiff, mok = {}, True
            for name, g in zip(("m_near", "m_far", "m_center"), mot):
                r = np.asarray(out[name], np.float32)
                fin = np.isfinite(r)
                u = C32.f16_steps(g[fin], r[fin])
                same_nonfinite = bool(np.array_equal(g[~fin].astype(np.float32), r[~fin], equal_nan=True))
                mdiff[name] = {"differ": int(np.sum(u > 0)), "max_f16_steps": int(u.max()) if u.size else 0}
                # Model M's rule for these surfaces (clean/tests/test_encoder_emu.py): at most 1 FP16 ulp, 99.5 % exact
                mok &= same_nonfinite and mdiff[name]["max_f16_steps"] <= 1 and float(np.mean(u == 0)) >= 0.995
            m["motion"] = mdiff
            # Model M's encoder criteria (clean/tests/test_encoder_emu.py): per frame the per-stage rule and at least
            # 99 % of values exact; over all frames at most 0.01 % of values differ (checked below).
            m["pass"] = (C32.passes(m) and m["frac_differ"] <= 0.01 and m["pad_bytes_nonzero"] == 0 and mok)
            res["frames"].append(m)
    res["total"] = {"frames": len(res["frames"]), "values": sum(m["values"] for m in res["frames"]),
                    "differ": sum(m["differ"] for m in res["frames"]),
                    "max_steps": max(m["max_steps"] for m in res["frames"])}
    res["total"]["frac_differ"] = res["total"]["differ"] / res["total"]["values"]
    res["pass"] = all(m["pass"] for m in res["frames"]) and res["total"]["frac_differ"] <= 1e-4
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--seeds", type=int, default=1, help="run the case list with this many seeds (89, 90, ...)")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print({"total": res["total"], "pass": res["pass"]})
    for m in res["frames"]:
        if not m["pass"]:
            print("FAIL", json.dumps(m))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
