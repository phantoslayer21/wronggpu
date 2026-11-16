# SPDX-License-Identifier: MIT
"""Model L (issue #89): candidate S2a acceptance tests on a local recorded run (design study, CPU only; only counts are output).

The matrix instruction's float32 sum of a 32-wide FP8 chunk is exact when every partial sum is representable. With
every product a multiple of g (the chunk's granularity) and sum |a w| < 2^24 g, every partial sum in any order is a
multiple of g below 2^24 g: exact. The shipped test takes g = 2^-18 (the smallest FP8 x FP8 granularity) and the bound
wsum x amax < 63 (63 = 2^24 x 2^-18 x 63 / 64). Candidates, each an upper bound of sum |a w| (B) against a limit (L):
  B: cur  = wsum x amax (shipped)                    h2 = sum over the two instructions' 16-K halves of wsum_h x amax_h
     q4   = sum over the four 8-K groups (lane fragments) of wsum_q x amax_q
     dual = asum x wmax (asum = sum |a| of the token's chunk, wmax = max |w| of the row's chunk)
     min2 = min(h2, dual)   min4 = min(q4, dual)   true = sum |a w| itself (the best any magnitude bound can do)
  L: 63 (g = 2^-18), or "g": 63 x 2^(18 + la + lw), la / lw = the lowest set bit's exponent over the token's / the row's
     nonzero codes of the chunk (every product a w is a multiple of 2^(la + lw)).
For every candidate: outputs and wave steps (32 outputs that run together in a block kernel) that would take the
exact fallback. Scores: a = the query, w = the key (l_q7.hip). The study only counts; each test that is adopted gets
its own adversarial probe (l_q7_s2a_check.py) before any kernel uses it.

    python clean/l/runner/l_q7_s2a_bounds.py --build <l_frame_check build> --json out.json
"""
import argparse
import copy
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.runner import l_record as LR  # noqa: E402
from clean.l.runner.l_q7_s2a_fallbacks import _amax  # noqa: E402

BOUNDS = ("cur", "h2", "q4", "dual", "min2", "min4", "true")
LIMITS = ("63", "g")


def lsb_exp(v):
    """Exponent of the lowest set significand bit of FP8 values (exact), +99 for 0, -99 for NaN."""
    x = np.abs(np.asarray(v, np.float64))
    m, e = np.frexp(np.where(np.isfinite(x) & (x > 0), x, 1.0))
    n = np.rint(m * 16).astype(np.int64)                    # 8 ... 15: 4 significand bits (FP8 values are exact)
    tz = np.where(n % 2 == 1, 0, np.where(n % 4 == 2, 1, np.where(n % 8 == 4, 2, 3)))
    out = (e - 4 + tz).astype(np.int64)
    out = np.where(x == 0, 99, out)
    return np.where(np.isnan(x), -99, out)


def group_max(x, groups):
    return np.stack([np.where(np.isnan(x[..., g]), np.inf, np.abs(x[..., g])).max(-1) for g in groups], -1)


H2 = [slice(0, 16), slice(16, 32)]
Q4 = [slice(0, 8), slice(8, 16), slice(16, 24), slice(24, 32)]


def criteria(a, W, chunk=2048):
    """a [..., 32] (token chunks), W [n, 32] (rows) -> {name: ok [..., n]} (in chunks of tokens: memory)."""
    lead = a.shape[:-1]
    flat = a.reshape(-1, a.shape[-1])
    parts = [_criteria(flat[i:i + chunk], W) for i in range(0, max(1, flat.shape[0]), chunk)]
    return {k: np.concatenate([p[k] for p in parts]).reshape(lead + (W.shape[0],)) for k in parts[0]}


def _criteria(a, W):
    with np.errstate(invalid="ignore", over="ignore"):
        aa, ww = np.abs(a), np.abs(W)
        f32 = np.float32
        am = _amax(a)                                                     # [...]
        ws = ww.sum(-1).astype(f32)                                       # [n]
        b = {"cur": am[..., None] * ws}
        for name, gr in (("h2", H2), ("q4", Q4)):
            amg = group_max(a, gr).astype(f32)                            # [..., G]
            wsg = np.stack([ww[:, g].sum(-1) for g in gr], -1).astype(f32)   # [n, G]
            b[name] = (amg[..., None, :] * wsg).sum(-1)                   # float64 here; the kernels' float32 rounding
        asum = np.where(np.isnan(a), np.inf, aa).sum(-1).astype(f32)      # is covered by the 63 / 64 margin
        wmax = np.where(np.isnan(W), np.inf, ww).max(-1).astype(f32)
        b["dual"] = asum[..., None] * wmax
        b["min2"] = np.minimum(b["h2"], b["dual"])
        b["min4"] = np.minimum(b["q4"], b["dual"])
        b["true"] = np.where(np.isnan(a[..., None, :] * W), np.inf, np.abs(a[..., None, :] * W)).sum(-1)
        la = lsb_exp(a).min(-1)                                           # [...]
        lw = lsb_exp(W).min(-1)                                           # [n]
        lim_g = 63.0 * np.exp2((18 + la[..., None] + lw).clip(-200, 200).astype(np.float64))
        nan = np.isnan(a).any(-1)[..., None] | np.isnan(W).any(-1)
    out = {}
    for k, v in b.items():
        out[k + "@63"] = (v < 63.0) & ~nan
        out[k + "@g"] = (v < lim_g) & ~nan
    return out


def wave_steps(bad):
    if bad.ndim < 2 or bad.shape[-2] != 16 or bad.shape[-1] % 16:
        return None
    return bad.reshape(bad.shape[:-2] + (16, bad.shape[-1] // 16, 2, 8)).any(axis=(-4, -2))


def study(build):
    from clean.l.reference import table as T
    from clean.l.runner import l_frame_check as F
    rec = pickle.load(open(F.rec_path(build, "L", "stages"), "rb"))
    params = pickle.load(open(os.path.join(F.model_dir(build), "params.pkl"), "rb"))["L"]
    stats, label = {}, ["?"]

    def add(key, oks, steps_fn):
        s = stats.setdefault(label[0] + ":" + key, {})
        for name, ok in oks.items():
            r = s.setdefault(name, [0, 0, 0, 0])
            r[0] += ok.size
            r[1] += int((~ok).sum())
            st = steps_fn(~ok)
            if st is not None:
                r[2] += st.size
                r[3] += int(st.sum())

    op8, score, stage_q7, enc, dec, blk = L._op8, L._score_op, L.run_stage_q7, L.encode, L.decode, L._block_tokens_q7

    def spy_op8(start, a8, W8, model):
        if label[0].endswith("/block") or not label[0].startswith(("enc0", "dec0")):
            add("rows%d" % W8.shape[0], criteria(np.asarray(a8, np.float64), np.asarray(W8, np.float64)), wave_steps)
        return op8(start, a8, W8, model)

    def score_steps(bad):
        if bad.shape[-1] != 16 or bad.shape[-2] != 16:
            return None
        return bad.reshape(bad.shape[:-1] + (2, 8)).any(axis=(-3, -2))

    def spy_score(B, u, model):
        uu = np.asarray(u, np.float64)                                    # [..., h, T, 32]
        oks = criteria_scores(uu)
        add("scores", oks, score_steps)
        return score(B, u, model)

    def wrap(name, f):
        def g(*a, **k):
            old = label[0]
            label[0] = name(old, *a) if callable(name) else name
            try:
                return f(*a, **k)
            finally:
                label[0] = old
        return g

    L._op8, L._score_op = spy_op8, spy_score
    L.run_stage_q7 = wrap(lambda old, stage, *a: stage, stage_q7)
    L.encode, L.decode = wrap("enc0", enc), wrap("dec0", dec)
    L._block_tokens_q7 = wrap(lambda old, *a: old + "/block" if old in ("enc0", "dec0") else old, blk)
    try:
        F.compare_model(rec, params, table=copy.deepcopy(T.TABLE))
    finally:
        L._op8, L._score_op, L.run_stage_q7, L.encode, L.decode, L._block_tokens_q7 = \
            op8, score, stage_q7, enc, dec, blk
    out = {}
    for site, crit in stats.items():
        out[site] = {name: {"outputs": n, "fallback_fraction": f / n if n else 0.0,
                            "wave_step_fallback_fraction": fs / ns if ns else None}
                     for name, (n, f, ns, fs) in crit.items()}
    return {"sites": out, "frames": len(rec["frames"]), "identity": rec.get("identity")}


def criteria_scores(u, chunk=256):
    """Scores: for query i and key j, a = u_i (the query's chunk), w = u_j (the key's): {name: ok [..., h, i, j]}
    (in chunks of windows x heads: memory)."""
    lead = u.shape[:-2]
    flat = u.reshape((-1,) + u.shape[-2:])
    parts = [_criteria_scores(flat[i:i + chunk]) for i in range(0, max(1, flat.shape[0]), chunk)]
    return {k: np.concatenate([p[k] for p in parts]).reshape(lead + u.shape[-2:-1] * 2) for k in parts[0]}


def _criteria_scores(u):
    res = {}
    a = u[..., :, None, :]                                                # [..., h, T, 1, 32]
    W = u[..., None, :, :]                                                # [..., h, 1, T, 32]
    with np.errstate(invalid="ignore", over="ignore"):
        f32 = np.float32
        nan = np.isnan(a).any(-1) | np.isnan(W).any(-1)
        absf = lambda x: np.where(np.isnan(x), np.inf, np.abs(x))         # noqa: E731
        am = absf(a).max(-1).astype(f32)
        ws = absf(W).sum(-1).astype(f32)
        b = {"cur": am * ws}
        for name, gr in (("h2", H2), ("q4", Q4)):
            b[name] = sum(absf(a[..., g]).max(-1).astype(f32).astype(np.float64) *
                          absf(W[..., g]).sum(-1).astype(f32) for g in gr)
        b["dual"] = absf(a).sum(-1).astype(f32) * absf(W).max(-1).astype(f32)
        b["min2"] = np.minimum(b["h2"], b["dual"])
        b["min4"] = np.minimum(b["q4"], b["dual"])
        b["true"] = absf(a * W).sum(-1)
        la, lw = lsb_exp(a).min(-1), lsb_exp(W).min(-1)
        lim_g = 63.0 * np.exp2((18 + la + lw).clip(-200, 200).astype(np.float64))
    for k, v in b.items():
        res[k + "@63"] = (v < 63.0) & ~nan
        res[k + "@g"] = (v < lim_g) & ~nan
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--json")
    ap.add_argument("--ldr", action="store_true", help="the build's LDR run (lframe_L_stages_ldr.pkl)")
    a = ap.parse_args(argv)
    if a.ldr:
        from clean.l.runner import l_frame_check as F
        F.LDR = True
    res = {"what": "S2a acceptance-test candidates on a local recorded run: outputs / wave steps taking the exact fallback "
                   "(CPU replay of a recorded l_frame_check run; a design study)", "fp8_op_model": "exact",
           "study": study(a.build)}
    names = [b + "@" + lim for lim in LIMITS for b in BOUNDS]
    print("site".ljust(22) + "".join(n.rjust(10) for n in names), "  (wave steps with a fallback, %)")
    for site, crit in res["study"]["sites"].items():
        row = []
        for n in names:
            w = crit[n]["wave_step_fallback_fraction"]
            row.append(("%.2f" % (100 * w)) if w is not None else ("o%.2f" % (100 * crit[n]["fallback_fraction"])))
        print(site.ljust(22) + "".join(x.rjust(10) for x in row))
    if a.json:
        LR.write_l_record(res, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
