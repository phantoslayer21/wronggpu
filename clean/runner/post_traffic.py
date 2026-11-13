# SPDX-License-Identifier: MIT
"""Memory-traffic floor of the spec 06 stages at the in-game size, against the real kernels, warm and cold cache.

    python clean/runner/post_traffic.py --build build --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--json traffic.json]

The post_traffic_* kernels (kernels/post.hip) move exactly the bytes of the reconstruction and the output filter for a
choice of image formats, fully coalesced and with no arithmetic: R 1706x960 inputs (colour f32 RGBA, A, B, C, three
motions: 44 bytes per render sample), the full history read and written at D 3840x2160 (8, 6 or 4 bytes per pixel),
the display image (3 FP16 planes, written by the reconstruction, read by the filter; absent in the fused layouts), the
reduced history (4 bytes per 2 x 2 block) and the output at O 2560x1440 (12 bytes f32 RGB or 8 bytes RGBA16F). Their
time is the floor any kernel with those formats can reach on this GPU. Each layout and each reference pair of real
kernels is timed

  - warm: launched back to back (what post_variants.py / post_bench.py measure; the 64 MB last-level cache keeps part
    of the previous launch's data), and
  - cold: after a flush kernel that streams a 256 MB scratch buffer (closer to the game, where the network stages
    between two post launches evict everything).

Printed per layout: MB moved, ms warm / cold, GB/s. Reference kernels (all exact unless noted):
  - post_recon2_r + post_filter_c1_s4 (in-game default);
  - post_recon2_r_h6 (packed 6-byte history) + post_filter_c1_s4 / post_filter_c1_s4_o16 (FP16 output);
  - post_recon2_r_hp (planar 6-byte history), post_recon2_r_p12 (dec0 pack12 coefficients) and
    post_recon2_r_hp_p12 (both), each + post_filter_c1_s4;
  - the banded two-kernel path, BAND_COUNTS bands: post_recon2_r_band + post_filter_c1_s4_band / _s2_band, and
    every other banded reconstruction (_hp_band, _band_p12, _hp_band_p12) + post_filter_c1_s4_band;
  - post_recon2_r_out (output mode D = O = 2560x1440, no filter);
  - approximations and fused: post_recon4_fastest + post_filter_c1_s4, post_fused, every post_fused3* variant.
The default pair, the h6 + o16 pair, the _hp / _p12 pairs, post_recon2_r_out and the banded runs are also timed from
post_w64.co (wave64 build of the current source, rebuilt on every run; --no-w64 skips it).
"""
import argparse
import ctypes
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.runner import post as PS  # noqa: E402
from clean.runner.post_bench import IN_GAME, big_inputs  # noqa: E402
from clean.runner.stage_runner import Runner  # noqa: E402

FLUSH_BYTES = 256 << 20
BAND_COUNTS = (2, 4, 8, 16)   # banded two-kernel path: bands per frame
# layout name -> (reconstruction-side kernel, filter-side kernel or None, history bytes, display, output bytes)
LAYOUTS = {
    "two-kernel, history 8 B (current formats)": ("post_traffic_r_h8", "post_traffic_f_o12", 8, True, 12),
    "two-kernel, history 6 B": ("post_traffic_r_h6", "post_traffic_f_o12", 6, True, 12),
    "two-kernel, history 4 B": ("post_traffic_r_h4", "post_traffic_f_o12", 4, True, 12),
    "fused, history 8 B, out f32": ("post_traffic_x_h8_o12", None, 8, False, 12),
    "fused, history 6 B, out f32": ("post_traffic_x_h6_o12", None, 6, False, 12),
    "fused, history 4 B, out f32": ("post_traffic_x_h4_o12", None, 4, False, 12),
    "fused, history 6 B, out f16": ("post_traffic_x_h6_o8", None, 6, False, 8),
    "fused, history 4 B, out f16": ("post_traffic_x_h4_o8", None, 4, False, 8),
}


def layout_bytes(rw, rh, dw, dh, ow, oh, hist, disp, out):
    """Bytes moved by one frame of the layout (each byte once)."""
    n = dw * dh
    b = rw * rh * 44 + 2 * n * hist + (n // 4) * 4 + ow * oh * out
    if disp:
        b += 2 * n * 6  # written by the reconstruction, read by the filter
    return b


def traffic_launches(sizes, layout, bufs):
    """Launches (name, grid, threads, args) of one frame of the layout."""
    rw, rh, dw, dh, ow, oh = sizes
    rk, fk, _, _, _ = LAYOUTS[layout]
    P = ctypes.c_void_p
    grid = PS._cdiv(dw * dh, PS.RECON_THREADS)
    out = [(rk, grid, PS.RECON_THREADS,
            [P(bufs[k]) for k in ("color", "A", "B", "C", "m_near", "m_center", "m_far", "hist_prev", "hist_out",
                                  "reduced", "display", "out")] + [ctypes.c_int(v) for v in sizes])]
    if fk:
        out.append((fk, grid, PS.RECON_THREADS,
                     [P(bufs["display"]), P(bufs["out"])] + [ctypes.c_int(v) for v in (dw, dh, ow, oh)]))
    return out


def time_launches(runner, launches, repeats, flush=None, plans=(), code_object=None):
    """Median ms of the launches (name, grid, threads, args) back to back. flush (function, grid, threads, args) runs
    before each repeat, outside the timed region. Releases the given plans."""
    dev = runner.dev
    fl = [(PS._function(runner, n, code_object), g, t, a) for n, g, t, a in launches]
    start, stop = dev.event(), dev.event()
    try:
        for f, g, t, a in fl:
            dev.launch(f, g, t, a)
        dev.sync()
        times = []
        for _ in range(int(repeats)):
            if flush is not None:
                dev.launch(*flush)
            dev.record(start)
            for f, g, t, a in fl:
                dev.launch(f, g, t, a)
            dev.record(stop)
            times.append(dev.elapsed_ms(start, stop))
        return float(np.median(times))
    finally:
        dev.destroy_event(start)
        dev.destroy_event(stop)
        for plan in plans:
            for ptr in plan["bufs"].values():
                dev.free(ptr)
            plan["bufs"].clear()


def time_plans(runner, plans, repeats, flush=None, code_object=None):
    return time_launches(runner, [l for plan in plans for l in plan["launches"]], repeats, flush, plans, code_object)


def run(r, sizes, repeats=50, rounds=3, reference=True, report=None, flush_bytes=FLUSH_BYTES, w64=False):
    rw, rh, dw, dh, ow, oh = sizes
    dev = r.dev
    n = dw * dh
    nbytes = {"color": rw * rh * 16, "A": rw * rh * 4, "B": rw * rh * 4, "C": rw * rh * 8, "m_near": rw * rh * 4,
              "m_center": rw * rh * 4, "m_far": rw * rh * 4, "hist_prev": n * 8, "hist_out": n * 8,
              "reduced": max(n // 4, 1) * 4, "display": n * 6, "out": ow * oh * 12}
    bufs = {k: dev.malloc(v) for k, v in nbytes.items()}
    scratch = dev.malloc(flush_bytes)
    report = report if report is not None else {}
    try:
        for k, v in nbytes.items():
            dev.memset(bufs[k], 0x11, v)
        dev.memset(scratch, 0, flush_bytes)
        nw = flush_bytes // 4
        flush = (PS._function(r, "post_traffic_flush"), PS._cdiv(nw, PS.RECON_THREADS), PS.RECON_THREADS,
                 [ctypes.c_void_p(scratch), ctypes.c_uint32(nw)])
        jobs = {}
        for name, (_, _, hist, disp, outb) in LAYOUTS.items():
            mb = layout_bytes(rw, rh, dw, dh, ow, oh, hist, disp, outb) / 1e6
            report.setdefault("layouts", {})[name] = {"MB": mb}
            for cold in (False, True):
                jobs[(name, cold)] = (lambda name=name, cold=cold: time_launches(
                    r, traffic_launches(sizes, name, bufs), repeats, flush if cold else None))
        if reference:
            rng = np.random.default_rng(1)
            x_norm, disp = big_inputs(rng, False)
            w = r.pack_weights("post", {"s": np.array(0.9)})
            pair = lambda rk, fk: (lambda: [PS.prepare_post(dev, w, x_norm, rk),  # noqa: E731
                                            PS.prepare_filter(dev, b"", disp, fk)])
            refs = {
                "post_recon2_r + post_filter_c1_s4 (in-game default)": pair("post_recon2_r", "post_filter_c1_s4"),
                "post_recon2_r_h6 + post_filter_c1_s4 (exact)": pair("post_recon2_r_h6", "post_filter_c1_s4"),
                "post_recon2_r_hp + post_filter_c1_s4 (exact, planar 6 B history)":
                    pair("post_recon2_r_hp", "post_filter_c1_s4"),
                "post_recon2_r_p12 + post_filter_c1_s4 (exact, pack12 input)":
                    pair("post_recon2_r_p12", "post_filter_c1_s4"),
                "post_recon2_r_hp_p12 + post_filter_c1_s4 (exact, planar + pack12)":
                    pair("post_recon2_r_hp_p12", "post_filter_c1_s4"),
                "post_recon2_r_h6 + post_filter_c1_s4_o16 (exact)": pair("post_recon2_r_h6", "post_filter_c1_s4_o16"),
                "post_recon4_fastest + post_filter_c1_s4":
                    lambda: [PS.prepare_post(dev, w, x_norm, "post_recon4_fastest"),
                             PS.prepare_filter(dev, b"", disp, "post_filter_c1_s4")],
                **{k: (lambda k=k: [PS.prepare_fused(dev, w, dict(x_norm, output_size=disp["output_size"]), k)])
                   for k in (PS.KERNEL_FUSED,) + tuple(PS.FUSED3_VARIANTS)},
            }
            # Output mode D = O (recon = output): reconstruction at 2560x1440 writing the output, no filter.
            x_out = dict(x_norm, recon_size=(ow, oh), history=x_norm["history"][:oh, :ow])
            if PS.recon2_supported(rw, rh, ow, oh):
                refs["post_recon2_r_out (D = O, no filter)"] = lambda: [PS.prepare_post(dev, w, x_out,
                                                                                        PS.KERNEL_RECON_OUT)]
            # Banded two-kernel path (exact): reconstruction and filter alternately over n horizontal bands.
            x_band = dict(x_norm, output_size=disp["output_size"])
            banded = []
            for rk, fk in [(PS.BAND_RECON, f) for f in PS.BAND_FILTERS] + [
                    (k, "post_filter_c1_s4_band") for k in list(PS.BAND_RECONS)[1:]]:
                for n in BAND_COUNTS:
                    name = f"banded n={n}: {rk} + {fk} (exact)"
                    refs[name] = lambda n=n, fk=fk, rk=rk: [PS.prepare_banded(dev, w, x_band, n, fk, rk)]
                    banded.append(name)
            for name, mk in refs.items():
                for cold in (False, True):
                    jobs[(name, cold)] = (lambda mk=mk, cold=cold: time_plans(r, mk(), repeats,
                                                                              flush if cold else None))
            if w64:   # the same kernels from post_w64.co (whole file built in wave64)
                for name in ["post_recon2_r + post_filter_c1_s4 (in-game default)",
                             "post_recon2_r_h6 + post_filter_c1_s4_o16 (exact)",
                             "post_recon2_r_hp + post_filter_c1_s4 (exact, planar 6 B history)",
                             "post_recon2_r_p12 + post_filter_c1_s4 (exact, pack12 input)",
                             "post_recon2_r_hp_p12 + post_filter_c1_s4 (exact, planar + pack12)",
                             "post_recon2_r_out (D = O, no filter)"] + banded:
                    if name not in refs:
                        continue
                    for cold in (False, True):
                        jobs[(name + " [wave64]", cold)] = (
                            lambda mk=refs[name], cold=cold: time_plans(r, mk(), repeats, flush if cold else None,
                                                                        PS.CODE_OBJECT_W64))
        runs = {k: [] for k in jobs}
        for _ in range(rounds):
            for k, job in jobs.items():
                runs[k].append(job())
        med = {k: float(np.median(v)) for k, v in runs.items()}
        print(f"{'layout / kernels':55s} {'MB':>6s} {'warm ms':>8s} {'cold ms':>8s} {'GB/s cold':>9s}")
        for (name, cold), ms in med.items():
            if cold:
                continue
            row = report.setdefault("layouts" if name in LAYOUTS else "reference", {}).setdefault(name, {})
            row["warm_ms"], row["cold_ms"] = ms, med[(name, True)]
            mb = row.get("MB")
            gbs = f"{mb / row['cold_ms']:9.0f}" if mb else " " * 9
            print(f"{name:55s} {mb if mb else 0:6.0f} {ms:8.3f} {row['cold_ms']:8.3f} {gbs}")
    finally:
        for p in list(bufs.values()) + [scratch]:
            dev.free(p)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-reference", action="store_true", help="only the traffic layouts, not the real kernels")
    ap.add_argument("--no-w64", action="store_true", help="skip the wave64 (post_w64.co) reference timings")
    ap.add_argument("--json", help="also write the results to this file")
    args = ap.parse_args(argv)
    r = Runner(args.build, args.rocm)
    w64 = not args.no_w64 and not args.no_reference
    if w64:   # always rebuilt: an older post_w64.co may lack kernels of this post.hip
        PS.build_wave64(args.build, args.rocm)
    (rw, rh), (ow, oh) = IN_GAME["R"], IN_GAME["O"]
    from clean.reference.post import recon_size_for
    sizes = (rw, rh, recon_size_for(ow), recon_size_for(oh), ow, oh)
    report = {"device": getattr(r.dev, "device_name", "?"), "sizes": sizes}
    run(r, sizes, args.repeats, args.rounds, not args.no_reference, report, w64=w64)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
