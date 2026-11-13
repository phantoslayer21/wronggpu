# SPDX-License-Identifier: MIT
"""Quality gate of a non-exact variant on the captured frames (owner's rule 2026-10-23; clean/rr-f-r8-head): per frame
the tonemapped PSNR against the exact default and % of pixels over 1/2/4/8 (of 255), and the temporal metrics of both
runs: the static-region frame-to-frame change (mean |O_t - O_t-1| over pixels whose motion is below 0.25 output
pixels) and the motion-compensated residual (mean |O_t - warp(O_t-1)|, bilinear warp along the game's motion
vectors; units and sign picked once as the ones that minimise the reference's residual), test / reference in %.
Crops: the three 64 x 64 output blocks with the largest mean difference on the worst frame, 2x nearest, as one JPEG
sheet (reference | test | |difference| x 8 per row).

    python -m clean.rr_f.gpu.temporal_gate <capture> <reference replay dir> <test replay dir> --json out.json
        [--crops sheet.jpg]"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proxy", "tools"))
import rr_capture_replay as CR   # noqa: E402


def outputs(d, wo, ho):
    host = os.path.join(d, "work", "host")
    metas = [json.loads(line.replace("\\", "/")) for line in open(os.path.join(host, "meta.jsonl"), encoding="utf-8")
             if '"kind":"frame"' in line.replace(" ", "")]
    ox, oy = CR.OUT_ORIGIN
    res = []
    for i, m in enumerate(metas):
        tw, th = m["out_tex"][0], m["out_tex"][1]
        r = np.fromfile(os.path.join(host, f"frame_{i:03d}.out"), np.float16).reshape(th, tw, 4)
        res.append(CR.tonemap(r[oy:oy + ho, ox:ox + wo, :3].astype(np.float32)).astype(np.float32))
    return res


def motion(frame, wo, ho):
    h, data = frame["inputs"][CR.BT["MOTION"]]
    x, y, w, hh = CR.input_rect(h)
    v = CR.decode(h, data)[y:y + hh, x:x + w, :2]
    yi = (np.arange(ho) * hh // ho).clip(0, hh - 1)
    xi = (np.arange(wo) * w // wo).clip(0, w - 1)
    return v[yi][:, xi], (w, hh)


def warp(img, mx, my):
    ho, wo = img.shape[:2]
    gy, gx = np.mgrid[0:ho, 0:wo].astype(np.float32)
    sx = np.clip(gx + mx, 0, wo - 1.001)
    sy = np.clip(gy + my, 0, ho - 1.001)
    x0, y0 = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    a, b = img[y0, x0], img[y0, x0 + 1]
    c, d = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--json")
    ap.add_argument("--crops")
    ap.add_argument("--size", default="2560x1440")
    a = ap.parse_args(argv)
    wo, ho = (int(v) for v in a.size.split("x"))
    R, T = outputs(a.ref, wo, ho), outputs(a.test, wo, ho)
    frames = CR.load_capture(a.capture)[:len(R)]
    rows = []
    for i, (r, t) in enumerate(zip(R, T)):
        dm = np.max(np.abs(r - t), axis=2)
        mse = float(np.mean((r - t) ** 2))
        rows.append(dict(frame=i, psnr_db=math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse),
                         pct_over={k: 100.0 * float(np.mean(dm > k)) for k in (1, 2, 4, 8)}, max_abs=float(dm.max())))
    # temporal metrics: motion units / sign chosen on the reference
    mv = [motion(f, wo, ho) for f in frames]
    resets = [int(f["meta"]["constants"].get("reset", 0)) for f in frames]
    cands = []
    for unit in ("uv", "px"):
        for sgn in (1.0, -1.0):
            tot = 0.0
            for i in range(1, len(R)):
                (m, (w, hh)) = mv[i]
                kx, ky = (wo, ho) if unit == "uv" else (wo / w, ho / hh)
                tot += float(np.mean(np.abs(R[i] - warp(R[i - 1], sgn * kx * m[..., 0], sgn * ky * m[..., 1]))))
            cands.append((tot, unit, sgn))
    _, unit, sgn = min(cands)
    temporal = []
    for i in range(1, len(R)):
        (m, (w, hh)) = mv[i]
        kx, ky = (wo, ho) if unit == "uv" else (wo / w, ho / hh)
        mx, my = sgn * kx * m[..., 0], sgn * ky * m[..., 1]
        static = np.hypot(mx, my) < 0.25
        e = dict(pair=f"{i - 1}-{i}", reset=resets[i], static_px_pct=100.0 * float(static.mean()))
        for tag, S in (("ref", R), ("test", T)):
            ch = np.abs(S[i] - S[i - 1]).mean(axis=2)
            e[f"static_change_{tag}"] = float(ch[static].mean()) if static.any() else 0.0
            e[f"mc_residual_{tag}"] = float(np.mean(np.abs(S[i] - warp(S[i - 1], mx, my))))
        temporal.append(e)
    ok = [e for e in temporal if not e["reset"]]
    summ = dict(motion_units=unit, motion_sign=sgn,
                static_change_ref=float(np.mean([e["static_change_ref"] for e in ok])),
                static_change_test=float(np.mean([e["static_change_test"] for e in ok])),
                mc_residual_ref=float(np.mean([e["mc_residual_ref"] for e in ok])),
                mc_residual_test=float(np.mean([e["mc_residual_test"] for e in ok])))
    summ["static_change_pct"] = 100.0 * (summ["static_change_test"] / summ["static_change_ref"] - 1.0)
    summ["mc_residual_pct"] = 100.0 * (summ["mc_residual_test"] / summ["mc_residual_ref"] - 1.0)
    fin = [min(r["psnr_db"], 999.0) for r in rows]
    worst = int(np.argmin(fin))
    rep = dict(frames=rows, worst_frame=worst, worst_psnr_db=rows[worst]["psnr_db"], mean_psnr_db=float(np.mean(fin)),
               worst_pct_over={k: max(r["pct_over"][k] for r in rows) for k in (1, 2, 4, 8)}, temporal=temporal,
               temporal_summary=summ)
    if a.crops:
        from PIL import Image
        d = np.abs(R[worst] - T[worst]).mean(axis=2)
        b = d[:ho // 64 * 64, :wo // 64 * 64].reshape(ho // 64, 64, wo // 64, 64).mean(axis=(1, 3))
        idx = np.argsort(b.ravel())[::-1][:3]
        tiles = []
        for k in idx:
            by, bx = divmod(int(k), b.shape[1])
            y0, x0 = by * 64, bx * 64
            sl = (slice(y0, y0 + 64), slice(x0, x0 + 64))
            df = np.clip(np.abs(R[worst][sl] - T[worst][sl]) * 8, 0, 255)
            row = np.concatenate([R[worst][sl], np.full((64, 2, 3), 255.0), T[worst][sl], np.full((64, 2, 3), 255.0), df], 1)
            tiles.append(row.repeat(2, 0).repeat(2, 1))
            rep.setdefault("crop_blocks", []).append(dict(x=x0, y=y0, mean_abs=float(b.ravel()[k])))
        sheet = np.concatenate([np.concatenate([t, np.full((4, t.shape[1], 3), 255.0)], 0) for t in tiles], 0)
        Image.fromarray(sheet.astype(np.uint8)).save(a.crops, quality=92)
    for r in rows:
        print(f"frame {r['frame']}: {r['psnr_db']:.2f} dB, % > 1/2/4/8 "
              + " / ".join(f"{r['pct_over'][k]:.3f}" for k in (1, 2, 4, 8)) + f", max {r['max_abs']:.0f}")
    for e in temporal:
        print(e)
    print({k: v for k, v in rep.items() if k not in ("frames", "temporal")})
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
