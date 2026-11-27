# SPDX-License-Identifier: GPL-3.0-only
"""Model L (issue #89): how often the S2a bound sends an FP8 op to the exact fallback on a local recorded run, per layer.
CPU only: the recorded run of an l_frame_check build is replayed through the oracle (as the frame
check does), and every 32-wide FP8 op chunk is classified with the kernels' own bound: bounded when
float32(wsum) x float32(amax) < 63, wsum = sum |w| of the weight row's chunk (the packers' table), amax = max |a| of
the token's chunk (from the codes: a NaN code counts as the largest, so NaN fails). Scores: wsum = sum |u_key|,
amax = max |u_query| (l_q7.hip). Labels: the stage (enc0_block / dec0_block for Model L's own blocks, else the shared
stage), then the op kind (the weight rows of the call, or "scores").

    python clean/l/runner/l_q7_s2a_fallbacks.py --build <l_frame_check build> --json out.json
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

LIMIT = np.float32(63.0)


def _amax(a):
    """max |a| over the last axis, a NaN as +inf (the code 0x7f is the largest)."""
    x = np.abs(np.asarray(a, np.float64))
    return np.where(np.isnan(x), np.inf, x).max(-1).astype(np.float32)


def count(build):
    from clean.l.reference import table as T
    from clean.l.runner import l_frame_check as F
    rec = pickle.load(open(F.rec_path(build, "L", "stages"), "rb"))
    params = pickle.load(open(os.path.join(F.model_dir(build), "params.pkl"), "rb"))["L"]
    stats, label = {}, ["?"]

    def add(key, outputs, fails, steps=0, step_fails=0):
        s = stats.setdefault(label[0], {}).setdefault(key, [0, 0, 0, 0])
        s[0] += int(outputs)
        s[1] += int(fails)
        s[2] += int(steps)
        s[3] += int(step_fails)

    def wave_steps(bad):
        """The block kernels' wave steps of a window's op [..., 16 tokens, n rows]: one step = the 16 tokens x rows
        r0 + j and r0 + 8 + j of a 16-row tile (l_q7.hip / l_q7w.hip: lane = token + 16 x half); a step pays the
        scalar fallback if any of its 32 outputs fails. (0, 0) for other shapes."""
        if bad.ndim < 2 or bad.shape[-2] != 16 or bad.shape[-1] % 16:
            return 0, 0
        b = bad.reshape(bad.shape[:-2] + (16, bad.shape[-1] // 16, 2, 8)).any(axis=(-4, -2))
        return b.size, int(b.sum())

    op8, score, stage_q7, enc, dec = L._op8, L._score_op, L.run_stage_q7, L.encode, L.decode

    def spy_op8(start, a8, W8, model):
        a = np.asarray(a8, np.float64)
        W = np.asarray(W8, np.float64)
        with np.errstate(invalid="ignore"):
            ws = np.abs(W).sum(-1).astype(np.float32)                       # [n]
            am = _amax(a)                                                    # [...]
            ok = (am[..., None] * ws).astype(np.float32) < LIMIT
        add("rows%d" % W.shape[0], ok.size, np.sum(~ok), *wave_steps(~ok))
        return op8(start, a8, W8, model)

    def spy_score(B, u, model):
        uu = np.asarray(u, np.float64)                                       # [..., h, T, 32]
        with np.errstate(invalid="ignore"):
            su = np.abs(uu).sum(-1).astype(np.float32)                       # key sums [..., h, T]
            am = _amax(uu)                                                   # query max [..., h, T]
            ok = (am[..., :, None] * su[..., None, :]).astype(np.float32) < LIMIT   # [..., h, query, key]
        bad = ~ok
        steps = (0, 0)
        if bad.shape[-1] == 16 and bad.shape[-2] == 16:   # lane = query + 16 x half, step jj = keys 8 half + jj
            b = bad.reshape(bad.shape[:-1] + (2, 8)).any(axis=(-3, -2))
            steps = (b.size, int(b.sum()))
        add("scores", ok.size, np.sum(bad), *steps)
        return score(B, u, model)

    def wrap(name, f):
        def g(*a, **k):
            old = label[0]
            label[0] = name if isinstance(name, str) else name(*a)
            try:
                return f(*a, **k)
            finally:
                label[0] = old
        return g

    L._op8, L._score_op = spy_op8, spy_score
    L.run_stage_q7 = wrap(lambda stage, *a: stage, stage_q7)
    L.encode, L.decode = wrap("enc0_block", enc), wrap("dec0_block", dec)
    try:
        F.compare_model(rec, params, table=copy.deepcopy(T.TABLE))
    finally:
        L._op8, L._score_op, L.run_stage_q7, L.encode, L.decode = op8, score, stage_q7, enc, dec
    out, tot = {}, [0, 0, 0, 0]
    for st, kinds in stats.items():
        out[st] = {}
        for k, (n, f, ns, fs) in sorted(kinds.items()):
            out[st][k] = {"outputs": n, "fallbacks": f, "fallback_fraction": f / n if n else 0.0,
                          "wave_steps": ns, "wave_steps_with_fallback": fs,
                          "wave_step_fallback_fraction": fs / ns if ns else None}
            for i, v in enumerate((n, f, ns, fs)):
                tot[i] += v
    return {"layers": out, "outputs": tot[0], "fallbacks": tot[1],
            "fallback_fraction": tot[1] / tot[0] if tot[0] else 0.0, "wave_steps": tot[2],
            "wave_steps_with_fallback": tot[3], "wave_step_fallback_fraction": tot[3] / tot[2] if tot[2] else 0.0,
            "frames": len(rec["frames"]), "identity": rec.get("identity")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {"what": "S2a bound on a local recorded run: FP8 op outputs per layer that take the exact fallback (CPU replay of a "
                   "recorded l_frame_check run, the kernels' bound in float32)", "fp8_op_model": "exact",
           "counts": count(a.build)}
    c = res["counts"]
    for st, kinds in c["layers"].items():
        for k, v in kinds.items():
            ws = v["wave_step_fallback_fraction"]
            print(f"{st:12s} {k:8s} outputs {v['outputs']:>12d} fallbacks {v['fallbacks']:>9d} "
                  f"({100 * v['fallback_fraction']:.3f} %)  wave steps with a fallback "
                  + ("-" if ws is None else f"{100 * ws:.2f} %"))
    print("total", c["outputs"], "fallbacks", c["fallbacks"], f"({100 * c['fallback_fraction']:.3f} %)",
          "wave steps", c["wave_steps"], "with a fallback", c["wave_steps_with_fallback"],
          f"({100 * c['wave_step_fallback_fraction']:.2f} %)")
    if a.json:
        LR.write_l_record(res, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
