# SPDX-License-Identifier: MIT
"""Exactness probes of the fast output-stage arithmetic (kernels/rr_output_probe.hlsl) on the GPU.

  mode 0: X (exp2), L (log2), H (f32 -> f16 RNE) candidates against rr_output.hlsl's helpers for all 2^32 f32 inputs;
  mode 1: f16 multiply / add / fma (third operand sampled) and the f32 M / A of f16-valued operands for all 2^32 pairs
          of f16 values;
  triples: the f32 M, A, F against hardware a * b, a + b, fma(a, b, c) on sampled operands (random bit patterns and
           operands whose exact result lies near the smallest normal).

    python -m clean.rr_f.gpu.output_probe --json <report>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import build

NAMES = ["X_exp2", "X_exp2_ftz", "L_log2", "L_log2_ftz", "H_hw", "HM_f16", "HA_f16", "HF_f16_sampled", "M_f16vals",
         "A_f16vals", "HF_one_nan_operand", "HF_several_nan_operands", "S_without_ftz",
         "log2_nonzero_below_2^-30", "R_iteration", "R_one_step"]
TINY = np.float32(2.0 ** -126)


def triples(rng, n_random=1 << 20, n_tiny=1 << 20):
    u = rng.integers(0, 1 << 32, (n_random, 3), dtype=np.uint64).astype(np.uint32)
    t = [u.view(np.float32)]
    # products near the smallest normal: b = TINY / a scaled by 1 +- a few ulps, c = -(a b) + small
    a = rng.uniform(1, 2, n_tiny).astype(np.float32) * np.float32(2.0) ** rng.integers(-60, 60, n_tiny).astype(np.float32)
    a *= np.where(rng.random(n_tiny) < .5, -1, 1).astype(np.float32)
    with np.errstate(all="ignore"):
        b = (np.float64(TINY) / a.astype(np.float64) * (1 + rng.integers(-40, 40, n_tiny) * 2.0 ** -25
                                                         + rng.normal(0, 2.0 ** -20, n_tiny) * (rng.random(n_tiny) < .3))
             ).astype(np.float32)
        b *= np.where(rng.random(n_tiny) < .5, -1, 1).astype(np.float32)
        p = a.astype(np.float64) * b.astype(np.float64)
        c = (-p * (1 + rng.integers(-8, 8, n_tiny) * 2.0 ** -24)).astype(np.float32)
        c = np.where(rng.random(n_tiny) < .3, np.float32(0), c)
        c = np.where(rng.random(n_tiny) < .2, (rng.integers(-3, 4, n_tiny) * TINY).astype(np.float32), c)
    t.append(np.stack([a, b, c], 1).astype(np.float32))
    # denormal operands
    d = rng.integers(0, 1 << 23, (1 << 16, 3), dtype=np.uint64).astype(np.uint32) | (
        rng.integers(0, 2, (1 << 16, 3), dtype=np.uint64).astype(np.uint32) << 31)
    norm = rng.integers(0, 1 << 32, (1 << 16, 3), dtype=np.uint64).astype(np.uint32)
    mix = np.where(rng.random((1 << 16, 3)) < .5, d, norm)
    t.append(mix.view(np.float32))
    # zeros, infinities, quiet / signalling NaNs, smallest normals, denormals, 1, largest finite
    sign, inf, mant = 1 << 31, 255 << 23, (1 << 23) - 1
    edge = np.array([0, sign, inf, sign | inf, inf | (1 << 22), sign | inf | (1 << 22), inf | 1, 1 << 23,
                     sign | (1 << 23), 1, mant, 127 << 23, (254 << 23) | mant, 1 << 22], np.uint32).view(np.float32)
    g = np.array(np.meshgrid(edge, edge, edge)).reshape(3, -1).T
    t.append(g.astype(np.float32))
    # sums near the smallest normal (A): c = -a + k 2^-149
    a2 = (rng.uniform(1, 4, 1 << 16) * 2.0 ** -126).astype(np.float32) * np.where(rng.random(1 << 16) < .5, -1, 1).astype(np.float32)
    k = rng.integers(-(1 << 24), 1 << 24, 1 << 16)
    c2 = (-(a2.astype(np.float64)) + k * 2.0 ** -149).astype(np.float32)
    t.append(np.stack([a2, np.ones_like(a2), c2], 1).astype(np.float32))
    return np.ascontiguousarray(np.concatenate(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("set WG_RR_BUILD")
    k = {n: (build.compile_kernel("rr_output_probe.hlsl", "rr_oprobe_" + n, {"WG_BLOCK": 256}), 4)
         for n in ("exhaustive", "triples")}
    ar = build.Arena()
    outs = []
    disp = []
    for mode in (0, 1):
        o = ar.add(np.concatenate([np.zeros(16, np.uint32), np.full(16, 0xffffffff, np.uint32)]))
        outs.append(o)
        disp.append(("exhaustive", 65536, [0, o, 0, mode]))
    tr = triples(np.random.default_rng(a.seed))
    ti = ar.add(tr)
    to = ar.reserve(len(tr) * 32)
    disp.append(("triples", (len(tr) + 255) // 256, [ti, to, len(tr), 2]))
    img, log = build.run(ar, k, disp, require_double=True, timeout=3000,
                         workdir=str(Path(build.BUILD) / "output-probe"))
    rep = {}
    for mode, o in enumerate(outs):
        cnt = img[o:o + 64].view("<u4")
        first = img[o + 64:o + 128].view("<u4")
        for j in range(16):
            if j < len(NAMES) and (j < 5 or j >= 12) == (mode == 0):
                rep[NAMES[j]] = dict(inputs=2 ** 32, mismatches=int(cnt[j]),
                                     first=None if first[j] == 0xffffffff else hex(int(first[j])))
    r = img[to:to + len(tr) * 32].view("<u4").reshape(-1, 8)
    trip = {}
    for name, ref, new in (("M_vs_mul", 0, 1), ("A_vs_add", 2, 3), ("F_vs_fma", 4, 5), ("F_vs_fma_fenced", 4, 6)):
        bad = np.nonzero(r[:, ref] != r[:, new])[0]
        nan_both = (np.isnan(r[bad, ref].view(np.float32)) & np.isnan(r[bad, new].view(np.float32)))
        rv, nv = r[bad, ref], r[bad, new]
        band = ((rv & 0x7fffffff) == 0x00800000) & ((nv & 0x7fffffff) == 0) & ((rv ^ nv) & 0x80000000 == 0)
        trip_band = int(band.sum())
        trip[name] = dict(band_mn_vs_zero_same_sign=trip_band, cases=int(len(tr)), mismatches=int(len(bad)), of_which_nan_payload=int(nan_both.sum()),
                          examples=[dict(a=hex(int(tr[i, 0].view(np.uint32))), b=hex(int(tr[i, 1].view(np.uint32))),
                                         c=hex(int(tr[i, 2].view(np.uint32))), ref=hex(int(r[i, ref])),
                                         new=hex(int(r[i, new]))) for i in bad[:12]])
    rep["triples"] = trip
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
