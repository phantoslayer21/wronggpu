# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the reconstruction kernel l_post_recon (clean/l/kernels/l_post.hip, HDR) against
the oracle (clean/l/reference/recon.py), synthetic data only.

    python clean/l/runner/l_post_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--seeds N]
        --json out.json

<dir>/l_post.co comes from clean/l/runner/l_c32_check.py --compile-only. The GPU part runs in a GPU slot. Output:
counts only, no timing.

Harness: every output buffer starts filled with a non-zero sentinel byte and is followed by sentinel slack. Every
defined output byte is compared: the full history's RGB (FP16) and alpha (must be 0), the display planes' pixels, and
the reduced codes' three 10-bit fields and top 2 bits (must be 0); no byte of any slack may change. A negative
control (the same case with the jitter moved by half a render pixel) must fail the criteria.

Criteria. Values: Model M's acceptance for its own reconstruction kernel (clean/tests/test_post_emu.py
test_matches_reference): no non-finite value, at most 0.01 % of FP16 values more than one ulp off, at most 1 % differ,
mean absolute error at most 0.1 % of the reference RMS; reduced codes at most 1 off, at most 1 % differ. These are
Model M's thresholds (float32 kernel against a float64 oracle); they are to be tightened when a kernel-order model of
the reconstruction exists. The oracle's colour gather runs in the model's precision (confirmed by the spec side), so
the kernel and the oracle agree on which weights underflow: no pixel is excluded.
Special colours (the encoder's random_inputs special values: NaN, 1e3 ... 1e6, small negatives, +inf): the same
criteria; the ill-conditioned pixels (a large fallback colour cancelled in the blend) are also reported apart.
Known answers (C = (65504, 65504, 0), D = R, every target offset from a texel centre by the jitter): offset 0.1, total
weight 0 everywhere: history -1.4404296875, display 0, reduced code 0; offset 0.05, total weight > 0 but every FP16
weight 0: the output is the candidate history (here the reset fallback: the linear colour times e).
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
SENTINEL = 0xA5
SLACK = 64
# (rw, rh, dw, dh, reset, kind)
CASES = [(24, 16, 54, 36, False, "random"), (31, 17, 70, 38, False, "random"), (20, 12, 60, 36, False, "random"),
         (33, 19, 99, 57, True, "random"), (17, 11, 38, 25, False, "extreme_c"),
         (16, 12, 16, 12, False, "offset_0.1"), (16, 12, 16, 12, True, "offset_0.05"),
         (24, 16, 54, 36, False, "special"), (31, 17, 70, 38, True, "special"), (33, 19, 99, 57, False, "special")]


def special_colours(rng, color):
    """The special colours of clean/reference/encoder.py random_inputs (special=True): NaN, 1e3 ... 1e6, small
    negatives, +inf in red, each on 1/200 of the pixels."""
    col = np.asarray(color, np.float32).copy()
    n = col.shape[0] * col.shape[1]
    flat = col.reshape(-1, 3)
    idx = rng.choice(n, size=max(4, n // 50), replace=False)
    flat[idx[0::4]] = np.nan
    flat[idx[1::4]] = rng.uniform(1e3, 1e6, size=(len(idx[1::4]), 3))
    flat[idx[2::4]] = -rng.uniform(0.0, 0.05, size=(len(idx[2::4]), 3))
    flat[idx[3::4], 0] = np.inf
    return col


def make_case(rng, rw, rh, dw, dh, reset, kind):
    x = MP.random_inputs(rng, rw, rh, dw, dh, reset=reset)
    x["exposure"], x["exposure_prev"] = np.float32(rng.uniform(0.5, 2.0)), np.float32(rng.uniform(0.5, 2.0))
    if kind == "special":
        x["color"] = special_colours(rng, x["color"])
        return x
    if kind != "random":
        x["C"] = np.broadcast_to(np.array([65504.0, 65504.0, 0.0], np.float32), np.shape(x["C"])).copy()
    if kind.startswith("offset_"):
        x["jitter"] = np.array([float(kind.split("_")[1]), 0.0], np.float32)
    return x


def run_kernel(hip, mod, x, s):
    """Launch l_post_recon with sentinel-filled outputs; returns the raw output buffers (with slack) and sizes."""
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
    sizes = {"history": dw * dh * 8, "reduced": max(4, (dw // 2) * (dh // 2) * 4), "display": RP.display_bytes(dw, dh)}
    outs = {}
    for k, n in sizes.items():
        outs[k] = hip.malloc(n + SLACK)
        hip.memset(outs[k], SENTINEL, n + SLACK)
    args = [ctypes.c_void_p(bufs[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far")]
    args += [ctypes.c_void_p(pw), ctypes.c_void_p(outs["history"]), ctypes.c_void_p(outs["reduced"]),
             ctypes.c_void_p(outs["display"]), prm, ctypes.c_void_p(0)]
    hip.launch(hip.get_function(mod, "l_post_recon"), RP.recon_grid(dw, dh), 256, args)
    hip.sync()
    raw = {k: np.frombuffer(hip.download(outs[k], n + SLACK), np.uint8) for k, n in sizes.items()}
    for ptr in list(bufs.values()) + [pw] + list(outs.values()):
        hip.free(ptr)
    return raw, sizes, (dw, dh)


def decode(raw, sizes, dims):
    """Every defined byte: history RGB + alpha, display pixels, reduced fields + top bits; slack integrity."""
    dw, dh = dims
    hist = raw["history"][:sizes["history"]].view(np.float16).reshape(dh, dw, 4)
    red = raw["reduced"][:sizes["reduced"]].view(np.uint32)[:(dw // 2) * (dh // 2)].reshape(dh // 2, dw // 2)
    disp = RP.decode_planes(raw["display"][:sizes["display"]].tobytes(), (dh, dw))
    return {"history": hist[..., :3].astype(np.float32),
            "alpha_zero": bool((hist[..., 3].view(np.uint16) == 0).all()),
            "reduced_codes": np.stack([red & 1023, (red >> 10) & 1023, (red >> 20) & 1023], -1).astype(np.int64),
            "reduced_top_zero": bool(((red >> 30) == 0).all()),
            "display": disp,
            "slack_untouched": all(bool((raw[k][sizes[k]:] == SENTINEL).all()) for k in sizes)}


def m_rule(got, ref):
    res = {}
    for k in ("history", "display"):
        g, r = np.asarray(got[k], np.float32), np.asarray(ref[k], np.float32)
        fin = np.isfinite(g)
        s = C32.f16_steps(np.where(fin, g, 0), r)
        d = np.abs(np.where(fin, g, 0.0).astype(np.float64) - r)
        res[k] = {"values": int(s.size), "nonfinite": int(np.sum(~fin)), "frac_over_1": float(np.mean(s > 1)),
                  "differ": int(np.sum(s > 0)), "frac_diff": float(np.mean(s > 0)), "max_steps": int(s.max()),
                  "mean_rel_rms": float(d.mean() / max(np.sqrt(np.mean(np.square(r, dtype=np.float64))), 1e-30))}
        res[k]["pass"] = (res[k]["nonfinite"] == 0 and res[k]["frac_over_1"] <= 1e-4 and res[k]["frac_diff"] <= 0.01
                          and res[k]["mean_rel_rms"] <= 1e-3)
    rc = np.rint(np.asarray(ref["reduced"], np.float64) * 1023.0).astype(np.int64)
    dc = np.abs(got["reduced_codes"] - rc)
    res["reduced"] = {"values": int(dc.size), "differ": int(np.sum(dc > 0)), "max": int(dc.max(initial=0))}
    res["reduced"]["pass"] = res["reduced"]["max"] <= 1 and res["reduced"]["differ"] <= 0.01 * max(dc.size, 1)
    res["pass"] = all(res[k]["pass"] for k in ("history", "display", "reduced"))
    return res


def check_case(hip, mod, x, kind, s=S):
    raw, sizes, dims = run_kernel(hip, mod, x, s)
    got = decode(raw, sizes, dims)
    ref = R.reconstruct(x, s)
    m = m_rule(got, ref)
    m.update({"alpha_zero": got["alpha_zero"], "reduced_top_zero": got["reduced_top_zero"],
              "slack_untouched": got["slack_untouched"]})
    ok = m["pass"] and got["alpha_zero"] and got["reduced_top_zero"] and got["slack_untouched"]
    W = ref["_lin"]["W"]
    m["weight_zero_pixels"] = int((W == 0).sum())
    if kind == "special":
        # Ill-conditioned pixels: H^ = Hl + (C - Hl) beta' cancels a large Hl (a large fallback colour), amplifying
        # every rounding of Hl by |Hl| |1 - beta'| / |H^| (here >= 2^8). Reported apart; the criteria above cover all.
        lin = ref["_lin"]
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            amp = (np.abs(lin["H_lin"]) * np.abs(1.0 - lin["beta_h"][..., None])
                   / np.maximum(np.abs(lin["H_hat"]), 1e-30)).max(-1)
        ill = np.nan_to_num(amp, nan=0.0) >= 256.0
        st = C32.f16_steps(np.where(np.isfinite(got["history"]), got["history"], 0), ref["history"])
        m["ill_conditioned"] = {"pixels": int(ill.sum()), "max_steps": int(st[ill].max(initial=0)),
                                "over_1": int((st[ill] > 1).sum())}
        m["well_conditioned"] = {"max_steps": int(st[~ill].max(initial=0)), "over_1": int((st[~ill] > 1).sum()),
                                 "values": int(st[~ill].size)}
    if kind == "offset_0.1":
        known = ((got["history"] == np.float32(-1.4404296875)).all() and (got["display"] == 0).all()
                 and (got["reduced_codes"] == 0).all() and m["weight_zero_pixels"] == W.size)
        m["known_answer"] = bool(known)
        ok = ok and known
    if kind == "offset_0.05":
        Hl = np.asarray(x["color"], np.float64) * float(np.asarray(x["exposure"]))
        known = bool((W > 0).all()) and np.array_equal(got["history"], MP.f16(MP.enc(Hl)))
        m["known_answer"] = bool(known)
        ok = ok and known
    m["pass"] = bool(ok)
    return m


def run(build, rocm, seeds=(89,)):
    from clean.runner.hip import Hip
    hip = Hip(rocm)
    with open(os.path.join(build, "l_post.co"), "rb") as f:
        mod = hip.load_module(f.read())
    res = {"cases": []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for rw, rh, dw, dh, reset, kind in CASES:
            x = make_case(rng, rw, rh, dw, dh, reset, kind)
            m = check_case(hip, mod, x, kind)
            m["case"] = [rw, rh, dw, dh, reset, kind]
            res["cases"].append(m)
    # Negative control: the first random case with the jitter moved by half a render pixel must fail.
    rng = np.random.default_rng(seeds[0])
    x = make_case(rng, *CASES[0])
    raw, sizes, dims = run_kernel(hip, mod, dict(x, jitter=np.asarray(x["jitter"], np.float32) + 0.5), S)
    neg = m_rule(decode(raw, sizes, dims), R.reconstruct(x, S))
    res["negative_control_fails"] = not neg["pass"]
    res["pass"] = all(c["pass"] for c in res["cases"]) and res["negative_control_fails"]
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
        print(json.dumps({k: c[k] for k in c if k not in ("history", "display", "reduced")}),
              {k: (c[k]["differ"], c[k].get("max_steps", c[k].get("max"))) for k in ("history", "display", "reduced")})
    print({"negative_control_fails": res["negative_control_fails"], "pass": res["pass"]})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
