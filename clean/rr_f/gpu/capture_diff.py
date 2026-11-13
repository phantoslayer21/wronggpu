# SPDX-License-Identifier: MIT
"""Image difference of two captured-frame replays (proxy/tools/rr_capture_replay.py output folders): per frame the
raw-byte equality of the RR output, PSNR (dB) of the tonemapped 8-bit output (the replay tool's tonemap: Reinhard,
sRGB), the largest absolute difference of the tonemapped values (0..255) and of the raw f16 output, and the worst frame.

    python -m clean.rr_f.gpu.capture_diff <reference dir> <test dir> [--json out]"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxy", "tools"))
import rr_capture_replay as CR   # noqa: E402  (tonemap, OUT_ORIGIN)


def frames(d):
    host = os.path.join(d, "work", "host")
    metas = [json.loads(line.replace("\\", "/")) for line in open(os.path.join(host, "meta.jsonl"), encoding="utf-8")
             if '"kind":"frame"' in line.replace(" ", "")]
    return host, metas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--json")
    ap.add_argument("--size", default="2560x1440")
    a = ap.parse_args(argv)
    wo, ho = (int(v) for v in a.size.split("x"))
    hr, mr = frames(a.ref)
    ht, mt = frames(a.test)
    ox, oy = CR.OUT_ORIGIN
    rows = []
    for i, (m0, m1) in enumerate(zip(mr, mt)):
        b0 = open(os.path.join(hr, f"frame_{i:03d}.out"), "rb").read()
        b1 = open(os.path.join(ht, f"frame_{i:03d}.out"), "rb").read()
        tw, th = m0["out_tex"][0], m0["out_tex"][1]
        r0 = np.frombuffer(b0, np.float16).reshape(th, tw, 4)[oy:oy + ho, ox:ox + wo, :3].astype(np.float32)
        r1 = np.frombuffer(b1, np.float16).reshape(th, tw, 4)[oy:oy + ho, ox:ox + wo, :3].astype(np.float32)
        t0 = CR.tonemap(r0).astype(np.float64)
        t1 = CR.tonemap(r1).astype(np.float64)
        mse = float(np.mean((t0 - t1) ** 2))
        fin = np.isfinite(r0) & np.isfinite(r1)
        rows.append(dict(frame=i, evaluate=m0.get("file", "")[-10:-4], bytes_identical=b0 == b1,
                         psnr_db=math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
                         max_abs_tonemapped=float(np.max(np.abs(t0 - t1))),
                         max_abs_raw=float(np.max(np.abs(r0[fin] - r1[fin]))) if fin.any() else 0.0,
                         pixels_differ=int(np.any(t0 != t1, axis=2).sum()),
                         # % of pixels whose largest tonemapped channel difference exceeds k / 255
                         pct_over={k: 100.0 * float(np.mean(np.max(np.abs(t0 - t1), axis=2) > k)) for k in (1, 2, 4, 8)}))
    worst = min(rows, key=lambda r: r["psnr_db"])
    rep = dict(ref=a.ref, test=a.test, frames=rows, all_identical=all(r["bytes_identical"] for r in rows),
               worst_frame=worst["frame"], worst_psnr_db=worst["psnr_db"],
               mean_psnr_db=float(np.mean([min(r["psnr_db"], 999.0) for r in rows])),
               max_abs_tonemapped=max(r["max_abs_tonemapped"] for r in rows),
               max_abs_raw=max(r["max_abs_raw"] for r in rows))
    for r in rows:
        print(r)
    print({k: v for k, v in rep.items() if k != "frames"})
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
