# SPDX-License-Identifier: GPL-3.0-only
"""Q11 part C: bit-exact probe of the reconstruction's fused-rounding forms C1 and C2 (handoff
preset-M/from-spec/PRESET-M-C1C2-machine-code.md) as l_post__morig.co compiles them, on adversarial inputs.

    python clean/l/runner/l_post_c12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --json out.json

<dir>/l_post_c12_probe.co (WG_LP_C1 = 1, WG_CR_C2 = 1) and <dir>/l_post_c12_probe__mut.co (WG_LP_C1 = 0: item 3's
exponent; WG_CR_C2 = 2: w1 with 2.5 f2 fused) come from clean/runner/build.py. The GPU part runs in a GPU slot.

C1: eps = fused_half(h(u v), C, fused_half(h(v v), B, h(h(u u) A))), FP16 in and out. C2 (float32, one rounding per
operation): f2 = f f, f3 = f f2, w0 = fused(-(f + f3), 1/2, f2), w1 = fused(f3, 1.5, -RN(2.5 f2)) + 1,
w3 = RN(RN(f3 - f2) 1/2), w2 = ((1 - w0) - w1) - w3.
The NumPy model rounds each fused operation once, exactly: the product of two FP16 (or float32) values is exact in
float64; the float64 sum's own rounding error (TwoSum) moves the sum one float64 step towards the exact value before
the single rounding to FP16 (float32), which gives the correctly rounded result (every FP16 / float32 rounding boundary
is a float64 number). NaNs agree when both are NaN (any payload).
Pass: the probe build equals the model in every value; the mutant build differs from the model in both C1 and C2.
Inputs: C1 offsets u, v as the kernel forms them (h(d + i), d in (-1, 1), i = -1 ... 2) plus random FP16 values in
(-2.5, 2.5), coefficients in spec 06's ranges (A, B <= 0, C of either sign) and wide random FP16 bit patterns; C2
fractions: 0, uniform [0, 1), the kernel's smallest nonzero fractions (2^-25 ...) and the values just below 1.
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


def _round_once(a, b, out_dtype):
    """RN_out(a + b) with one rounding (a, b float64 arrays whose values are exact)."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)                     # TwoSum: s + err = a + b exactly
    with np.errstate(invalid="ignore"):
        s2 = np.where((err != 0) & np.isfinite(s), np.nextafter(s, s + np.sign(err) * np.inf), s)
    with np.errstate(over="ignore", invalid="ignore"):
        return s2.astype(out_dtype).astype(np.float64)


def _h(v):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(v, np.float64).astype(np.float16).astype(np.float64)


def eps_model(u, v, kA, kB, kC):
    """C1, the machine-code form (float64 in, FP16 values out)."""
    with np.errstate(over="ignore", invalid="ignore"):
        uu, vv, uv = _h(u * u), _h(v * v), _h(u * v)
        ta = _h(uu * kA)
        s = _round_once(vv * kB, ta, np.float16)
        return _round_once(uv * kC, s, np.float16)


def eps_item3(u, v, kA, kB, kC):
    """item 3's form (Model L's kernel): for the report only."""
    with np.errstate(over="ignore", invalid="ignore"):
        uu, vv, uv = _h(u * u), _h(v * v), _h(u * v)
        s = _h(_h(uu * kA) + _h(vv * kB))
        return _round_once(uv * kC, s, np.float16)


def cr_model(f):
    """C2 (float32 values in [0, 1), float32 values out as float64) -> [n, 4]."""
    F = np.float32
    f = np.asarray(f, F)
    f2 = (f * f).astype(F)
    f3 = (f * f2).astype(F)
    t = (-(f + f3).astype(F)).astype(np.float64)
    w0 = _round_once(t * 0.5, f2.astype(np.float64), F).astype(F)
    r = (F(2.5) * f2).astype(F)
    w1 = (_round_once(f3.astype(np.float64) * 1.5, -r.astype(np.float64), F).astype(F) + F(1.0)).astype(F)
    w3 = ((f3 - f2).astype(F) * F(0.5)).astype(F)
    w2 = (((F(1.0) - w0).astype(F) - w1).astype(F) - w3).astype(F)
    return np.stack([w0, w1, w2, w3], -1).astype(np.float64)


def eps_inputs(rng, n):
    """[n, 5] FP16 values (u, v, A, B, C): four structured groups of n / 5 and n / 5 wide random bit patterns."""
    k = n // 5

    def kern(m):  # offsets as the kernel forms them: h(h(d) + i), d in (-1, 1), i = -1 ... 2
        return _h(_h(rng.uniform(-1.0, 1.0, size=m)) + rng.integers(-1, 3, size=m))

    def coef(m):  # spec 06 ranges: A, B <= 0, C of either sign
        return (_h(_h(rng.uniform(0.0, 40.0, size=m)) * -0.721325), _h(_h(rng.uniform(0.0, 40.0, size=m)) * -0.721325),
                _h(_h(rng.uniform(-40.0, 40.0, size=m)) * -1.44265))

    rnd = lambda m: _h(rng.uniform(-2.5, 2.5, size=m))  # noqa: E731
    u = np.concatenate([kern(k), rnd(k), kern(k), rnd(k)])
    v = np.concatenate([kern(k), rnd(k), rnd(k), kern(k)])
    cA, cB, cC = (np.asarray(t) for t in coef(4 * k))
    m = n - 4 * k
    bits = rng.integers(0, 1 << 16, size=(5, m), dtype=np.uint64).astype(np.uint16)
    wide = bits.view(np.float16).astype(np.float64)
    wide[:2] = np.where(np.abs(wide[:2]) < 4.0, wide[:2], _h(rng.uniform(-2.5, 2.5, size=wide[:2].shape)))
    x = np.concatenate([np.stack([u, v, cA, cB, cC], -1), wide.T])
    edge = np.array([[0.0, 0.0, -0.0, 0.0, 0.0], [2.5, -2.5, -65504.0, -65504.0, 65504.0],
                     [6.1e-5, 6.1e-5, -1.0, -1.0, 1.0], [1.0, 1.0, -2.0 ** -24, -2.0 ** -24, 2.0 ** -24],
                     [1.0, 2.0, -np.inf, -1.0, 1.0], [2.0, 1.0, -1.0, -1.0, np.inf], [1.0, 1.0, np.nan, -1.0, 1.0],
                     [2.0, 2.0, -40000.0, -40000.0, 0.0], [0.5, -0.5, -0.0, -0.0, -0.0]])
    return _h(np.concatenate([edge, x])[:n])


def cr_inputs(rng, n):
    F = np.float32
    k = n // 3
    a = rng.uniform(0.0, 1.0, size=k).astype(F)
    small = (2.0 ** rng.uniform(-25.0, -1.0, size=k)).astype(F)
    top = (F(1.0) - (rng.integers(1, 1 << 20, size=n - 2 * k) * F(2.0 ** -24))).astype(F)
    fixed = np.array([0.0, 2.0 ** -25, 0.5, 0.25, 0.75, 1.0 - 2.0 ** -24], F)
    return np.concatenate([fixed, a, small, top])[:n].astype(F)


def same(got, ref):
    g, r = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    eq = (g == r) & (np.signbit(g) == np.signbit(r))
    return eq | (np.isnan(g) & np.isnan(r))


def run_probe(hip, mod, ein, fin):
    n = ein.shape[0]
    xb = ein.astype(np.float16).view(np.uint16).tobytes()
    pi, po = hip.malloc(len(xb)), hip.malloc(2 * n)
    hip.upload(pi, xb)
    hip.launch(hip.get_function(mod, "l_probe_eps"), -(-n // 256), 256,
               [ctypes.c_void_p(pi), ctypes.c_void_p(po), ctypes.c_int(n)])
    hip.sync()
    eps = np.frombuffer(hip.download(po, 2 * n), np.uint16).view(np.float16).astype(np.float64)
    hip.free(pi)
    hip.free(po)
    m = fin.size
    fb = fin.astype(np.float32).tobytes()
    pf, pw = hip.malloc(len(fb)), hip.malloc(16 * m)
    hip.upload(pf, fb)
    hip.launch(hip.get_function(mod, "l_probe_cr"), -(-m // 256), 256,
               [ctypes.c_void_p(pf), ctypes.c_void_p(pw), ctypes.c_int(m)])
    hip.sync()
    w = np.frombuffer(hip.download(pw, 16 * m), np.float32).reshape(m, 4).astype(np.float64)
    hip.free(pf)
    hip.free(pw)
    return eps, w


def run(build, rocm, n_eps=4_000_000, n_cr=1_000_000, seed=11):
    from clean.runner.hip import Hip
    hip = Hip(rocm)
    rng = np.random.default_rng(seed)
    ein, fin = eps_inputs(rng, n_eps), cr_inputs(rng, n_cr)
    e_ref = eps_model(*ein.T)
    w_ref = cr_model(fin)
    e_i3 = eps_item3(*ein.T)
    res = {"inputs": {"eps": int(ein.shape[0]), "cr": int(fin.size), "seed": seed},
           "model_item3_vs_machine_differ_frac": float(np.mean(~same(e_i3, e_ref)))}
    for tag, co in (("probe", "l_post_c12_probe.co"), ("mutant", "l_post_c12_probe__mut.co")):
        with open(os.path.join(build, co), "rb") as f:
            mod = hip.load_module(f.read())
        eps, w = run_probe(hip, mod, ein, fin)
        de = ~same(eps, e_ref)
        dw = ~same(w, w_ref).all(-1)
        r = {"eps_differ": int(de.sum()), "cr_differ": int(dw.sum())}
        if de.any():
            j = int(np.argmax(de))
            r["eps_first"] = {"in": ein[j].tolist(), "got": float(eps[j]), "model": float(e_ref[j])}
        if dw.any():
            j = int(np.argmax(dw))
            r["cr_first"] = {"f": float(fin[j]), "got": w[j].tolist(), "model": w_ref[j].tolist()}
        res[tag] = r
    res["probe"]["pass"] = res["probe"]["eps_differ"] == 0 and res["probe"]["cr_differ"] == 0
    res["mutant"]["caught"] = res["mutant"]["eps_differ"] > 0 and res["mutant"]["cr_differ"] > 0
    res["pass"] = res["probe"]["pass"] and res["mutant"]["caught"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    res = run(a.build, a.rocm)
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
