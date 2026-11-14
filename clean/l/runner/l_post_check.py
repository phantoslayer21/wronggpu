# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the reconstruction kernel l_post_recon (clean/l/kernels/l_post.hip, HDR) against
the oracle (clean/l/reference/recon.py), synthetic data only.

    python clean/l/runner/l_post_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--seeds N]
        --json out.json

<dir>/l_post.co comes from clean/l/runner/l_c32_check.py --compile-only. The GPU part runs in a GPU slot. Output:
counts only, no timing.

Per case, Model M's acceptance for its reconstruction kernel (clean/tests/test_post_emu.py test_matches_reference):
history and display (FP16): no non-finite value, at most 0.01 % of values more than one FP16 ulp off, at most 1 %
differ at all, mean absolute error at most 0.1 % of the reference RMS; reduced codes: at most 1 code off, at most 1 %
differ. Cases: two D / render ratios (about 2.25 and exactly 3), odd sizes, jitter, reset, and extreme kernel shapes
C = (65504, 65504, 0): there every D pixel whose total weight is 0 in the float64 oracle must store the confirmed
values (history -1.4404296875, display 0); the kernel's float32 weights underflow at a smaller tap distance than the
oracle's float64 ones, so the other pixels of that case are not compared.
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

from clean.l.reference import recon as R  # noqa: E402
from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.reference import post as MP  # noqa: E402
from clean.runner import post as RP  # noqa: E402

S = 2.9796233
# (rw, rh, dw, dh, reset, kind)
CASES = [(24, 16, 54, 36, False, "random"), (31, 17, 70, 38, False, "random"), (20, 12, 60, 36, False, "random"),
         (33, 19, 99, 57, True, "random"), (17, 11, 38, 25, False, "zero_weight")]


def run_kernel(hip, mod, x, s):
    prm = RP.post_params(x)
    data = RP.post_device_inputs(x)
    dw, dh = prm.dw, prm.dh
    bufs = {}
    for k, v in data.items():
        bufs[k] = hip.malloc(len(v))
        hip.upload(bufs[k], v)
    blob = RP.pack({"s": np.array(s)})
    pw = hip.malloc(len(blob))
    hip.upload(pw, blob)
    nh, nr, nd = dw * dh * 8, max(4, (dw // 2) * (dh // 2) * 4), RP.display_bytes(dw, dh)
    ph, pr, pd = hip.malloc(nh), hip.malloc(nr), hip.malloc(nd)
    for ptr, n in ((ph, nh), (pr, nr), (pd, nd)):
        hip.memset(ptr, 0, n)
    args = [ctypes.c_void_p(bufs[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far")]
    args += [ctypes.c_void_p(pw), ctypes.c_void_p(ph), ctypes.c_void_p(pr), ctypes.c_void_p(pd), prm,
             ctypes.c_void_p(0)]
    hip.launch(hip.get_function(mod, "l_post_recon"), RP.recon_grid(dw, dh), 256, args)
    hip.sync()
    out = {"history": RP.decode_history(hip.download(ph, nh), (dh, dw)),
           "reduced": RP.decode_reduced(hip.download(pr, nr), (dh // 2, dw // 2)),
           "display": RP.decode_planes(hip.download(pd, nd), (dh, dw))}
    for ptr in list(bufs.values()) + [pw, ph, pr, pd]:
        hip.free(ptr)
    return out


def m_rule(got, ref):
    """Model M's post acceptance (see the module text) for the three outputs."""
    res = {}
    for k in ("history", "display"):
        g, r = np.asarray(got[k], np.float32), np.asarray(ref[k], np.float32)
        fin = np.isfinite(g)
        s = C32.f16_steps(np.where(fin, g, 0), r)
        d = np.abs(np.where(fin, g, 0.0).astype(np.float64) - r)
        res[k] = {"values": int(s.size), "nonfinite": int(np.sum(~fin)), "frac_over_1": float(np.mean(s > 1)),
                  "frac_diff": float(np.mean(s > 0)), "max_steps": int(s.max()) if s.size else 0,
                  "mean_rel_rms": float(d.mean() / max(np.sqrt(np.mean(np.square(r, dtype=np.float64))), 1e-30))}
        res[k]["pass"] = (res[k]["nonfinite"] == 0 and res[k]["frac_over_1"] <= 1e-4 and res[k]["frac_diff"] <= 0.01
                          and res[k]["mean_rel_rms"] <= 1e-3)
    dc = np.abs(np.rint(got["reduced"] * 1023.0) - np.rint(ref["reduced"] * 1023.0))
    res["reduced"] = {"values": int(dc.size), "differ": int(np.sum(dc > 0)), "max": int(dc.max(initial=0))}
    res["reduced"]["pass"] = res["reduced"]["max"] <= 1 and (res["reduced"]["differ"] <= 0.01 * max(dc.size, 1))
    res["pass"] = all(res[k]["pass"] for k in ("history", "display", "reduced"))
    return res


def run(build, rocm, seeds=(89,)):
    from clean.runner.hip import Hip
    hip = Hip(rocm)
    with open(os.path.join(build, "l_post.co"), "rb") as f:
        mod = hip.load_module(f.read())
    res = {"cases": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for rw, rh, dw, dh, reset, kind in CASES:
            x = MP.random_inputs(rng, rw, rh, dw, dh, reset=reset)
            x["exposure"], x["exposure_prev"] = np.float32(rng.uniform(0.5, 2.0)), np.float32(rng.uniform(0.5, 2.0))
            if kind == "zero_weight":
                x["C"] = np.broadcast_to(np.array([65504.0, 65504.0, 0.0], np.float32), np.shape(x["C"])).copy()
            got = run_kernel(hip, mod, x, S)
            ref = R.reconstruct(x, S)
            m = {"case": [rw, rh, dw, dh, reset, kind]}
            if kind == "zero_weight":
                zero = ref["_lin"]["W"] == 0
                m["zero_pixels"] = int(zero.sum())
                m["history_ok"] = bool((got["history"][zero] == np.float32(-1.4404296875)).all())
                m["display_ok"] = bool((got["display"][zero] == 0).all())
                m["pass"] = m["zero_pixels"] > 0 and m["history_ok"] and m["display_ok"]
            else:
                m.update(m_rule(got, ref))
            res["cases"].append(m)
    res["pass"] = all(c["pass"] for c in res["cases"])
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
    for c in res["cases"]:
        print(json.dumps(c))
    print({"pass": res["pass"]})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
