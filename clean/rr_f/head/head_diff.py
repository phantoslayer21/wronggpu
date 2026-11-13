# SPDX-License-Identifier: MIT
"""Head outputs of runtime arena dumps (rr_test_host frame dumps, WG_RR_ALIAS=0): extract and compare (track r3-head).

    python -m clean.rr_f.head.head_diff extract <host dir> <out.npz> [--delete]   (frame_NNN.arena + meta.jsonl)
    python -m clean.rr_f.head.head_diff compare <a.npz> <b.npz>
    python -m clean.rr_f.head.head_diff stats <a.npz> <b.npz> [out.json]   (value-difference percentiles, r8-head)

Regions per frame: Cnet [Ht][Wt][4] f16, PREC [Hr][Wr][4] f16, GATES / CANDW [Hr][Wr][4] u8 (work buffers, read in the
frame they are written), HID [Hr][Wr][4] s8 (state, layout slot 8). compare prints per frame and region the differing
values, the tokens (pixels) with any difference, and the largest difference (f16 steps or codes)."""
import json
import os
import sys

import numpy as np


def regions(img, meta):
    wr, hr = meta["render"]
    wt, ht = 32 * -(-wr // 32), 32 * -(-hr // 32)
    mp = dict(e.rsplit(" ", 1) for e in meta["map"].split("|") if " " in e)
    mp = {k: int(v) for k, v in mp.items()}

    def view(off, shape, dt):
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        return np.frombuffer(img[off:off + n], dt).reshape(shape).copy()
    return dict(cnet=view(mp["cnet"], (ht, wt, 4), np.uint16), prec=view(mp["prec"], (hr, wr, 4), np.uint16),
                gates=view(mp["gates"], (hr, wr, 4), np.uint8), candw=view(mp["candw"], (hr, wr, 4), np.uint8),
                hid=view(meta["layout"][8], (hr, wr, 4), np.int8))


def extract(host, out, delete=False):
    metas = [json.loads(l.replace("\\", "/")) for l in open(os.path.join(host, "meta.jsonl"))]
    res = {}
    for m in metas:
        if m.get("kind") != "frame":
            continue
        p = os.path.join(host, "frame_%03d.arena" % m["frame"])
        img = np.fromfile(p, np.uint8)
        for k, v in regions(img, m).items():
            res[f"f{m['frame']}_{k}"] = v
        del img
        if delete:
            os.remove(p)
    np.savez_compressed(out, **res)
    print(f"{out}: {len(res)} arrays")


def ordinal16(u):
    u = u.astype(np.int64)
    return np.where(u & 0x8000, -(u & 0x7fff), u & 0x7fff)


def compare(a, b):
    A, Bz = np.load(a), np.load(b)
    frames = sorted({int(k[1:].split("_")[0]) for k in A.files})
    tot = dict(values=0, tokens=0, frames=0, n=0)
    for f in frames:
        row, toks = [], None
        for k in ("cnet", "prec", "gates", "candw", "hid"):
            x, y = A[f"f{f}_{k}"], Bz[f"f{f}_{k}"]
            d = x != y
            if k in ("cnet", "prec"):
                mx = int(np.abs(ordinal16(x) - ordinal16(y)).max())
            else:
                mx = int(np.abs(x.astype(int) - y.astype(int)).max())
            t = d.any(axis=-1)
            if k != "cnet":
                toks = t if toks is None else toks | t
            tot["values"] += int(d.sum())
            tot["n"] += d.size
            row.append(f"{k} {int(d.sum())} (max {mx})")
        nt = int(toks.sum())
        tot["tokens"] += nt
        tot["frames"] += int(nt > 0 or "cnet 0" not in row[0])
        print(f"frame {f}: " + ", ".join(row) + f"; render pixels with a difference {nt}")
    print(f"total: {tot['values']} of {tot['n']} head values differ, {tot['tokens']} pixels, {tot['frames']} of "
          f"{len(frames)} frames")

def stats(a, b, out=None):
    """Value differences per region (b - a) over all frames and per frame: % differing, max, p99, p99.9 of |d| in the
    stored units (Cnet / PREC: f16 steps and absolute value; GATES / CANDW: codes of 1/255; HID: codes of 1/127)."""
    A, Bz = np.load(a), np.load(b)
    frames = sorted({int(k[1:].split("_")[0]) for k in A.files})
    rep = {}
    for k in ("cnet", "prec", "gates", "candw", "hid"):
        rows, allv = [], []
        for f in frames:
            x, y = A[f"f{f}_{k}"], Bz[f"f{f}_{k}"]
            if k in ("cnet", "prec"):
                d = np.abs(ordinal16(x) - ordinal16(y)).ravel()
                xv, yv = x.view(np.float16).astype(np.float64), y.view(np.float16).astype(np.float64)
                fin = np.isfinite(xv) & np.isfinite(yv)
                av = float(np.abs(xv - yv)[fin].max()) if fin.any() else 0.0
            else:
                d = np.abs(x.astype(np.int64) - y.astype(np.int64)).ravel()
                av = None
            allv.append(d)
            rows.append(dict(frame=f, pct_differ=100.0 * float(np.mean(d != 0)), max=int(d.max()),
                             p99=float(np.percentile(d, 99)), p999=float(np.percentile(d, 99.9)), max_abs_value=av))
        d = np.concatenate(allv)
        rep[k] = dict(pct_differ=100.0 * float(np.mean(d != 0)), max=int(d.max()), p99=float(np.percentile(d, 99)),
                      p999=float(np.percentile(d, 99.9)), mean=float(d.mean()), frames=rows)
        print(k, {q: v for q, v in rep[k].items() if q != "frames"},
              "per-frame %differ", [round(r["pct_differ"], 3) for r in rows])
    if out:
        open(out, "w", encoding="utf-8").write(json.dumps(rep, indent=1))
    return rep



if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], "--delete" in sys.argv)
    elif sys.argv[1] == "stats":
        stats(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        compare(sys.argv[2], sys.argv[3])
