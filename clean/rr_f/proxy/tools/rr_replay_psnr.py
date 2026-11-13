# SPDX-License-Identifier: MIT
"""Image difference of two capture replays (rr_capture_replay.py output folders, same capture, --passes 1): per frame
PSNR in dB of the tonemapped output (the replay's tonemap without the 8-bit rounding: x * exposure, Reinhard, sRGB;
peak 1), the max abs error of the tonemapped values and of the linear output, and the count of differing values.

    python clean/rr_f/proxy/tools/rr_replay_psnr.py <reference replay folder> <test replay folder> [--exposure 1]
"""
import argparse
import json
import os

import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rr_capture_replay as R  # noqa: E402


def frames(folder):
    work = os.path.join(folder, "work", "host")
    metas = [json.loads(l.replace("\\", "/")) for l in open(os.path.join(work, "meta.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for i, hm in enumerate(metas):
        p = os.path.join(work, f"frame_{i:03d}.out")
        if not os.path.exists(p):
            break
        tw, th = hm["out_tex"][0], hm["out_tex"][1]
        out.append((np.fromfile(p, np.float16).reshape(th, tw, 4), hm))
    return out


def tone(x, e):
    x = np.asarray(x, np.float64) * e
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), 0.0)
    x = x / (1.0 + x)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--wo", type=int, default=2560)
    ap.add_argument("--ho", type=int, default=1440)
    a = ap.parse_args()
    A, B = frames(a.ref), frames(a.test)
    ox, oy = R.OUT_ORIGIN
    rows = []
    for i, ((ra, _), (rb, _)) in enumerate(zip(A, B)):
        x = ra[oy:oy + a.ho, ox:ox + a.wo, :3].astype(np.float32)
        y = rb[oy:oy + a.ho, ox:ox + a.wo, :3].astype(np.float32)
        tx, ty = tone(x, a.exposure), tone(y, a.exposure)
        mse = float(np.mean((tx - ty) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        fin = np.isfinite(x) & np.isfinite(y)
        rows.append(dict(frame=i, psnr_db=psnr, max_abs_tonemapped=float(np.abs(tx - ty).max()),
                         max_abs_linear=float(np.abs(x - y)[fin].max()) if fin.any() else 0.0,
                         differing_values=int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))))
    for r in rows:
        print(f"frame {r['frame']}: PSNR {r['psnr_db']:.2f} dB, max abs tonemapped {r['max_abs_tonemapped']:.5f}, "
              f"linear {r['max_abs_linear']:.5g}, differing {r['differing_values']}")
    w = min(rows, key=lambda r: r["psnr_db"])
    print(f"worst frame {w['frame']}: {w['psnr_db']:.2f} dB; mean PSNR {np.mean([r['psnr_db'] for r in rows]):.2f} dB; "
          f"max abs tonemapped {max(r['max_abs_tonemapped'] for r in rows):.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
