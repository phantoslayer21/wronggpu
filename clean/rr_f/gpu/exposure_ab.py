# SPDX-License-Identifier: MIT
"""A/B of the exposure partial pass: RR_EX_WAVE 1 (one wave per 256-pixel block, f32 products under the no-underflow
guard) against RR_EX_WAVE 0 (the 256-lane LDS tree, binary64 products), partial sums and the finish result bit for
bit, on random and adversarial f32 colour buffers (zeros, tiny, subnormal, inf, NaN, huge, negative values; weights
that force the binary64 path).

    python -m clean.rr_f.gpu.exposure_ab --json <report>"""
import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import struct

import numpy as np

from . import build
from . import exposure as GX


def colour(rng, w, h, mode):
    c = rng.random((h, w, 4)).astype(np.float32) * np.float32(4.0)
    if mode >= 1:                                   # sprinkle adversarial values
        special = np.array([0.0, -0.0, 2.0 ** -110, 2.0 ** -127, -2.0 ** -120, 1e-38, np.inf, -np.inf, np.nan, 3e38,
                            -5.0, 2.0 ** -100, 2.0 ** -101, 65504.0], np.float32)
        m = rng.random(c.shape) < (0.002 if mode == 1 else 0.3)
        c[m] = rng.choice(special, int(m.sum()))
    if mode == 3:                                   # one tiny value per block only (guard fails in few waves)
        c[:] = np.abs(c)
        c.reshape(-1)[::997] = np.float32(2.0 ** -110)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    rng = np.random.default_rng(5858)
    kern = {"p0": (build.compile_kernel("rr_exposure.hlsl", "rr_exposure_partial", {"WG_BLOCK": 256, "RR_EX_WAVE": 0},
                                        tag="rr_exposure_partial_lds"), 23),
            "p1": (build.compile_kernel("rr_exposure.hlsl", "rr_exposure_partial", {"WG_BLOCK": GX.EX_BLOCK["partial"]}), 23),
            "fin": (build.compile_kernel("rr_exposure.hlsl", "rr_exposure_finish", {"WG_BLOCK": 256}), 23)}
    k0 = GX.symbols()
    ar = build.Arena()
    disp, recs = [], []
    cases = []
    for (w, h) in ((1706, 960), (333, 77), (256, 1), (31, 9)):
        for mode in range(4):
            cases.append((w, h, mode, k0))
    cases.append((640, 360, 0, replace(k0, X_LWrb=np.float32(2.0 ** -30))))    # weight below the guard
    cases.append((640, 360, 1, replace(k0, X_LWg=np.float32(0.0))))
    # the guard's edges (products exactly 2^-126 / just below it): weights at 2^-26 and one ulp under it, colours with
    # the 2^-100 / 2^-101 / subnormal specials (mode 0)
    for lw in (2.0 ** -26, float(np.nextafter(np.float32(2.0 ** -26), np.float32(0))), -(2.0 ** -26)):
        cases.append((640, 360, 0, replace(k0, X_LWrb=np.float32(lw), X_LWg=np.float32(lw))))
    for (w, h, mode, k) in cases:
        c = colour(rng, w, h, mode)
        blocks = (w * h + 255) // 256
        cp = ar.add(c)
        outs = []
        for name in ("p0", "p1"):
            pp = ar.reserve(blocks * 8, fill=0xEE)
            out = ar.reserve(100, fill=0xA5)
            ilo, ihi = struct.unpack("<II", struct.pack("<d", 1.0 / (w * h)))
            args = [cp, pp, out, c.shape[1], c.shape[2], 0, 0, w, h, blocks, ilo, ihi,
                    int(np.float32(1.0).view(np.uint32)), 1]
            args += [int(np.float32(getattr(k, f.name)).view(np.uint32)) for f in fields(k)]
            disp += [(name, blocks, args), ("fin", 1, args)]
            outs.append((pp, out))
        recs.append((w, h, mode, blocks, outs))
    img, log = build.run(ar, kern, disp, require_double=True, workdir=str(Path(build.BUILD) / "exposure-ab"))
    rows, ok = [], True
    for (w, h, mode, blocks, ((pa, oa), (pb, ob))) in recs:
        da = int((img[pa:pa + blocks * 8] != img[pb:pb + blocks * 8]).sum())
        db = int((img[oa:oa + 100] != img[ob:ob + 100]).sum())
        rows.append(dict(w=w, h=h, mode=mode, blocks=blocks, partial_bytes_differ=da, result_bytes_differ=db))
        ok = ok and da == 0 and db == 0
    Path(a.json).write_text(json.dumps(dict(passed=ok, cases=rows), indent=1), encoding="utf-8")
    for r in rows:
        print(r)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
