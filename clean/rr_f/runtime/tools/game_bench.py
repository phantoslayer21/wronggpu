# SPDX-License-Identifier: MIT
"""Game-setup GPU timing of the RR runtime, warmed up (clean/rr-f-speed5).

    python -m clean.rr_f.runtime.tools.game_bench --runtime-dir <dir with wronggpu_rr_runtime.dll + rr_test_host.exe>
        --model <model.wgcm> --out <work dir> [--capture <capture folder>] [--render WxH] [--passes 150]
        [--window 8] [--json <report>] [--settings N]

The sequence bench (gpu/sequence.py --bench) times the standard configuration (alpha upscaling, an RGBA16F layer, the
diagnostic stores, synthetic temporal images), so changes that only act in the game's configuration do not show there,
and the captured-frame replay (proxy/tools/rr_capture_replay.py) uploads every frame's inputs between evaluates, so the
GPU idles and the clock never ramps (its times run about 2x a warmed bench). This bench runs the game's real call
setup - the capture's creation (packing, HDR, depth, motion), input formats, routing, transparency layer, flags and
sizes, prepared by the replay tool itself (no GPU) - through the real runtime and the proxy's Lifecycle, with every
frame's inputs resident (rr_test_host perfcap): the frames are evaluated in order <passes> times back to back (up to
2 frames queued, as a game), and the runtime logs its own per-pass GPU timestamps every <window> frames
(WG_RR_TIMING_WINDOW). Warm-up: the first third of the windows is discarded; the run is "stable" when the best frame
time of the last third is within 1.5% of the middle third's. Reported: per stage the best window mean (as the sequence
bench's best per dispatch) and the median, the best and median window frame time, over the last two thirds.

--render WxH resamples (nearest) every input of the capture's render size to another render size at the same output
size (e.g. 1280x720 for 1440p Performance); motion vectors keep their scale (the replay hands them as a fraction of
the render size), so the timing sees the game's formats and flags at that size (content is resampled, not re-rendered).
Takes the GPU lock itself (the GPU lock (WG_GPU_LOCK, BUILDING.md)): never wrap it in another gpu_lock call. Runtime options (WG_RR_BODY,
WG_RR_INPUT_MLP, WG_RR_HEAD, ...) come from the environment.
"""
import argparse
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "proxy", "tools"))
import rr_capture_replay as RP  # noqa: E402

DEFAULT_CAPTURE = os.environ.get("WG_RR_CAPTURE", "")   # local capture folder (clean/rr_f/proxy/CAPTURE.md), not in the repository
PASSES = ("pre", "input", "body", "head", "output", "resolve", "store", "in.dec", "in.feat", "in.rep", "in.mlp")
# texel bytes by DXGI format, as rr_test_host.cpp bpp()
_BPP = {2: 16, 10: 8, 16: 8, 41: 4, 40: 4, 28: 4, 29: 4, 24: 4, 26: 4, 34: 4, 87: 4, 54: 2, 49: 2, 61: 1}


def resample_frame(src, dst, rw, rh):
    """Rewrites a frame file (RRF2, tests/rr_test_host.cpp read_frame) at render size rw x rh: every role of the old
    render size is resampled (nearest); other roles are kept."""
    b = open(src, "rb").read()
    o = 0

    def take(n):
        nonlocal o
        v = b[o:o + n]
        o += n
        return v
    head = bytearray(take(4 * 6 + 4 * 4 + 64 * 2 + 8))
    magic, orw, orh = struct.unpack_from("<3I", head, 0)
    assert magic == 0x32465252, src
    struct.pack_into("<2I", head, 4, rw, rh)
    n = struct.unpack("<I", take(4))[0]
    out = [bytes(head), struct.pack("<I", n)]
    for _ in range(n):
        role, fmt, w, h, ox, oy, st, conv = struct.unpack("<8I", take(32))
        sc, bi = take(4), take(4)
        bpp = _BPP.get(fmt)
        if bpp is None:
            raise SystemExit(f"game_bench: no texel size for DXGI format {fmt} (role {role})")
        data = take(w * h * bpp)
        if (w, h) == (orw, orh):
            px = np.frombuffer(data, np.uint8).reshape(h, w, bpp)
            ys = (np.arange(rh) * h) // rh
            xs = (np.arange(rw) * w) // rw
            data = np.ascontiguousarray(px[ys][:, xs]).tobytes()
            w, h = rw, rh
        out += [struct.pack("<8I", role, fmt, w, h, ox, oy, st, conv), sc, bi, data]
    assert o == len(b), (src, o, len(b))
    open(dst, "wb").write(b"".join(out))


def parse_windows(log_path, window):
    rx = re.compile(r"timing, (\d+) frames from frame (\d+): GPU ([\d.]+) ms per frame \((.*?)\)")
    wins = []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = rx.search(line)
        if not m or int(m.group(1)) != window:
            continue
        parts = dict((k, float(v)) for k, v in (p.strip().split(" ") for p in m.group(4).split(",")))
        wins.append(dict(first=int(m.group(2)), frame=float(m.group(3)), **parts))
    return wins


def summarize(wins):
    n = len(wins)
    if n < 6:
        raise SystemExit(f"game_bench: only {n} timing windows (more passes needed)")
    mid, last = wins[n // 3: 2 * n // 3], wins[2 * n // 3:]
    use = wins[n // 3:]
    bm, bl = min(w["frame"] for w in mid), min(w["frame"] for w in last)
    rep = dict(windows=n, used=len(use), first_window_ms=wins[0]["frame"], stable=abs(bl - bm) <= 0.015 * bm,
               best_mid_ms=bm, best_last_ms=bl,
               frame=dict(best_ms=min(w["frame"] for w in use), median_ms=statistics.median(w["frame"] for w in use)),
               per_stage={k: dict(best_ms=min(w[k] for w in use), median_ms=statistics.median(w[k] for w in use))
                          for k in PASSES if k in use[0]})
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--render", help="WxH: resample the inputs to this render size (same output size)")
    ap.add_argument("--output", help="WxH: output size of the game setup (default: the capture's), e.g. 1920x1080 with "
                    "--render 1280x720 for 1080p Quality; inputs of the capture's render size are resampled to --render")
    ap.add_argument("--passes", type=int, default=150)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--json")
    ap.add_argument("--settings", type=int, default=0, help="WG_RR_SET_* bits handed to the runtime per frame")
    a = ap.parse_args(argv)
    dll = os.path.join(a.runtime_dir, "wronggpu_rr_runtime.dll")
    host = os.path.join(a.runtime_dir, "rr_test_host.exe")
    prep = os.path.join(a.out, "prep")
    summ = RP.run(a.capture, a.model, prep, runtime=dll, host=host, gpu=False, log=lambda s: None, settings=a.settings)
    cr = summ["create"]
    files = [os.path.join(prep, "work", "in", f"frame_{n:06d}.bin") for n in summ["evaluates"]]
    render = tuple(cr["render"])
    outsz = list(cr["output"])
    if a.output:
        outsz[0:2] = [int(v) for v in a.output.lower().split("x")]
    if a.render:
        render = tuple(int(v) for v in a.render.lower().split("x"))
        rs = os.path.join(a.out, f"in-{render[0]}x{render[1]}")
        os.makedirs(rs, exist_ok=True)
        new = []
        for f in files:
            d = os.path.join(rs, os.path.basename(f))
            resample_frame(f, d, *render)
            new.append(d)
        files = new
    hostdir = os.path.join(a.out, "host")
    os.makedirs(hostdir, exist_ok=True)
    job = [f"runtime {dll}", f"outdir {hostdir}", f"output {RP.OUT_FORMAT} 1", f"env WG_RR_TIMING_WINDOW {a.window}",
           "create {} {} {} {} {} {} {} {} {} {}".format(*outsz, *render, cr["packing"], cr["hdr"],
                                                         cr["depth_inverted"], cr["mv_jittered"], cr["hw_depth"],
                                                         cr["alpha"]),
           f"perfcap {a.passes} " + " ".join(files), "destroy", "end"]
    jf = os.path.join(a.out, "job.txt")
    with open(jf, "w") as fh:
        fh.write("\n".join(job) + "\n")
    log_path = os.path.join(a.out, "wronggpu-rr.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    env = dict(os.environ, WG_RR_MODEL=a.model, WG_RR_LOG=log_path)
    env.pop("WG_RR_ALIAS", None)
    t0 = time.time()
    r = subprocess.run(RP._locked([host, jf]), capture_output=True,
                       text=True, env=env, timeout=7200)
    with open(os.path.join(a.out, "host.log"), "w") as fh:
        fh.write(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"rr_test_host failed ({r.returncode}): {r.stdout[-1500:]} {r.stderr[-1500:]}")
    rep = summarize(parse_windows(log_path, a.window))
    opts = {k: v for k, v in os.environ.items() if k.startswith("WG_RR_") and k not in ("WG_RR_BUILD", "WG_RR_JOBS")}
    rep.update(settings=a.settings, capture=a.capture, render=render, output=outsz, passes=a.passes, window=a.window,
               frames=len(files), options=opts, runtime=dll, wall_s=round(time.time() - t0, 1),
               host=[l for l in r.stdout.splitlines() if l.startswith("perfcap")])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    st = rep["per_stage"]
    print(f"game setup {render[0]}x{render[1]} -> {outsz[0]}x{outsz[1]}, {len(files)} frames x "
          f"{a.passes} passes, {rep['windows']} windows ({rep['used']} used), first window {rep['first_window_ms']:.2f} "
          f"ms, {'stable' if rep['stable'] else 'NOT STABLE'} (best mid {rep['best_mid_ms']:.3f} / last "
          f"{rep['best_last_ms']:.3f})")
    print("best   " + " ".join(f"{k} {v['best_ms']:.3f}" for k, v in st.items()) +
          f" | frame {rep['frame']['best_ms']:.3f}")
    print("median " + " ".join(f"{k} {v['median_ms']:.3f}" for k, v in st.items()) +
          f" | frame {rep['frame']['median_ms']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
