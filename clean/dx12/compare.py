# SPDX-License-Identifier: MIT
"""Byte-for-byte check of the D3D12 kernel ports: the same synthetic Model M frames through the pure HIP backend and
through the hybrid D3D12 backend (runtime backend 2, default), or native D3D12 (backend 3, --backend d3d12), comparing every named buffer of the
runtime (wg_rt_buffer: every stage output, every state image, the inputs and the output) after every frame, plus the
decoded stage outputs of Pipeline.frame(debug=True). Exit status 0 only if everything is identical AND every listed
kernel really ran on D3D12 (counted in the launch trace).

    python clean/dx12/compare.py --build <build dir> --kernels exposure_auto_diet
        [--rocm "C:/Program Files/AMD/ROCm/7.2"] [--models <work dir>] [--frames 4] [--quick] [--json report.json]

Cases: the default path at 1706x960 -> 2560x1440 (depth inverted, auto-exposure; resets on the first and the fourth
frame) and an odd smaller size 853x481 -> 1279x721 (auto-exposure, reset first). --quick: the small case only.
--include-4k adds 1920x1080 -> 3840x2160 Performance geometry, with resets on frames 0 and 3.
Synthetic parameters only (clean/runner/synthetic_model.py, made-up output-filter weights); never real model data.
Buffers no default kernel writes (clean/runner/output_filter_original_check.py STALE: b0-b4, dec1; the display image when
the post kernel filters itself), unused pack12 B/C capacity, and the previous-frame state on reset frames are listed
but not counted. --probe-unused-capacity poisons the entire C allocation differently on each backend before frames
1 onward, requires identical active C4 bytes, and checks that the unused allocation tail retains each sentinel.
Frame 0 is compared normally; the runtime exposes named buffers only after its first frame.
--motion-outres: motion vectors at output resolution (runtime option motion_outres=original, wg_rt_upload_motion_hi): the
cases 1705x960->2560x1440 (Quality), 1920x1080->3840x2160 (Performance), 1280x720->1920x1080 (Quality) and the odd one,
each frame with a synthetic FP16 motion texture of the output size (smooth field plus per-texel noise, positions off
the reconstruction both ways), mv_scale / mv_bias other than 1 / 0 and pre-exposures 1, 1.375 and 0.00077; the kernels
input_encoder_mo and post_tf_f32_p12_mo (code objects encoder_mo.co, post_mo_w64__orig.co) in place of the default
encoder and post kernels.
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clean.runner import output_filter_original_check as OC  # noqa: E402
from clean.runner import synthetic_model as SM  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402

BUFFERS = ["exposure.pair", "enc0.features", "enc0.m_near", "enc0.m_far", "enc0.m_center", "enc1.skip", "enc1.merged",
           "enc2.skip", "enc2.merged"] + [f"b{i}.out" for i in range(6)] + [
           "dec2.out", "dec1.out", "dec0.A", "dec0.B", "dec0.C", "dec0.latent", "post.history", "post.reduced",
           "post.display", "filter.out", "state.latent_prev", "state.motion_history", "state.history_prev",
           "input.color", "input.depth", "input.motion"]
EXPOSURE_BYTES = 20   # exposure.pair is named with 12 bytes; with post_diet it holds e, e_prev, e, ratio, 0.01 / e
CASES = [("default 1706x960->2560x1440", (1706, 960), (2560, 1440), {"depth_inverted": True, "auto_exposure": True},
          (0, 3)),
         ("odd 853x481->1279x721", (853, 481), (1279, 721), {"depth_inverted": False, "auto_exposure": True}, (0,))]
PERFORMANCE_4K = ("performance 1920x1080->3840x2160", (1920, 1080), (3840, 2160),
                  {"depth_inverted": True, "auto_exposure": True}, (0, 3))
# --motion-outres: motion at output resolution (the game shapes: Hogwarts Legacy's 1705x960 -> 2560x1440)
MO_CASES = [("motion-outres quality 1705x960->2560x1440", (1705, 960), (2560, 1440),
             {"depth_inverted": True, "auto_exposure": True}, (0, 3)),
            ("motion-outres odd 853x481->1279x721", (853, 481), (1279, 721),
             {"depth_inverted": False, "auto_exposure": True}, (0,))]
MO_4K = [("motion-outres performance 1920x1080->3840x2160", (1920, 1080), (3840, 2160),
          {"depth_inverted": True, "auto_exposure": True}, (0, 3)),
         ("motion-outres quality 1280x720->1920x1080", (1280, 720), (1920, 1080),
          {"depth_inverted": True, "auto_exposure": True}, (0, 3))]
MO_KERNELS = {"input_encoder": "emo/input_encoder_mo", "w64/post_tf_f32_p12_diet_vsit": "pmo/post_tf_f32_p12_mo"}
UNFUSED_FRAME_KERNELS = {"exposure_auto_diet", "input_encoder", "swin_block_c64_nc", "patch_merge_c64",
                         "swin_block_c96_nc", "patch_merge_c96", "bneck/swin_bneck6_nc", "swin_block_c96_add_ex_nc",
                         "expand_c96", "d1p12/swin_block_c64_add_dec0_nc", "w64/post_tf_f32_p12_diet_vsit"}
# The native backend (--backend d3d12) runs the D3D12 fusions of WG_D3D12_FUSION (default: runtime.cpp's
# kDefaultD3d12Fusion, mirrored here), which replace some launches (kernels/fused_swin.hlsl, fused_bneck.hlsl, fused_dec.hlsl).
DEFAULT_D3D12_FUSION = "pm,xe,mb,rv,xo,xd"


def d3d12_fusion():
    f = os.environ.get("WG_D3D12_FUSION", DEFAULT_D3D12_FUSION)
    return set(f.split(",")) - {"none", ""}


def frame_kernels(backend, fusion=None):
    k = set(UNFUSED_FRAME_KERNELS)
    f = d3d12_fusion() if fusion is None else fusion
    if backend != "d3d12":
        return k
    rv = "_rev" if "rv" in f else ""
    if "pm" in f:
        k = (k - {"patch_merge_c64", "swin_block_c96_nc"}) | {"swin_block_c96_pm" + rv + "_nc"}
    if "xe" in f:
        k = (k - {"expand_c96", "swin_block_c96_add_ex_nc"}) | {"swin_block_c96_add_xe" + rv + "_nc"}
    if "mb" in f:
        k = (k - {"patch_merge_c96", "bneck/swin_bneck6_nc"}) | {"bneck/swin_bneck6_mb_nc"}
    if {"xo", "xe", "mb"} <= f:   # xo: dec2's forward xe kernel overlapping the bottleneck's mbo form
        k = (k - {"swin_block_c96_add_xe" + rv + "_nc", "bneck/swin_bneck6_mb_nc"}) | {
            "swin_block_c96_add_xe_xo_nc", "bneck/swin_bneck6_mbo_nc"}
        if "xd" in f:   # xd: the xo kernel publishing e1 rows, d1p12 overlapping it (kernels/fused_dec.hlsl), only on
            # frames of at most 1.2 M render pixels (runtime.cpp kXdMaxPixels: the odd case); the larger cases keep the
            # xo kernel and the plain d1p12
            k = k | {"swin_block_c96_add_xe_xd_nc", "d1p12/swin_block_c64_add_dec0_xd_nc"}
    return k


def fusion_unwritten(kernels):
    """Maps that only the launches a fusion replaces read, not written by the fused kernels (listed, not counted):
    enc1.merged (pm), dec2.out (xe). `kernels`: the runtime's kernel report (d3d12_fusion=...)."""
    f = set(kernels.get("d3d12_fusion", "none").split(","))
    return ({"enc1.merged"} if "pm" in f else set()) | ({"dec2.out"} if "xe" in f else set())


FRAME_KERNELS = frame_kernels("d3d12", set(DEFAULT_D3D12_FUSION.split(",")) - {"none", ""})   # the default frame
DEBUG_KERNELS = {"color_divide", "scale_rgb"}
DEFAULT_MODELS = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "build")), "dx12-m", "work", "compare_models")


def selected_cases(include_4k=False, quick=False, motion_outres=False):
    if quick and include_4k:
        raise ValueError("--quick and --include-4k cannot be combined")
    if motion_outres:
        return MO_CASES[1:] if quick else MO_CASES + (MO_4K if include_4k else [])
    return CASES[1:] if quick else CASES + ([PERFORMANCE_4K] if include_4k else [])


def required_kernels(backend, manifest, motion_outres=False):
    """The selected FP32-output frame, not every variant the manifest can represent."""
    available = {k["name"] for k in manifest["kernels"]}
    k = frame_kernels(backend)
    if motion_outres:
        k = {MO_KERNELS.get(n, n) for n in k}
    return k | (DEBUG_KERNELS if backend == "d3d12" else DEBUG_KERNELS & available)


def motion_texture(index, out_size):
    """Synthetic raw motion of the output-size region M for frame `index` ([oh][ow][2] FP16): a smooth field of up to
    about 40 texels plus per-texel noise, so that previous positions leave the reconstruction on every side."""
    ow, oh = out_size
    rng = np.random.default_rng([OC.SEED, 4242, index])
    yy, xx = np.meshgrid(np.arange(oh, dtype=np.float64) / oh, np.arange(ow, dtype=np.float64) / ow, indexing="ij")
    ph = rng.uniform(0, 2 * np.pi, 4)
    mx = 40.0 * np.sin(2 * np.pi * xx + ph[0]) * np.cos(np.pi * yy + ph[1]) + rng.normal(0, 1.5, (oh, ow))
    my = 25.0 * np.cos(2 * np.pi * yy + ph[2]) * np.sin(np.pi * xx + ph[3]) + rng.normal(0, 1.5, (oh, ow))
    return np.stack([mx, my], -1).astype(np.float16)


def motion_outres_input(x, index, out_size):
    """OC.frame_input's frame with the output-size motion (pipeline.py x["motion_hi"]): mv_scale converts it to pixels
    of M, mv_bias is added first; the pre-exposure of every third frame is Hogwarts Legacy's 0.00077."""
    x = dict(x)
    x["motion_hi"] = motion_texture(index, out_size)
    x["mv_scale"] = (0.75, -1.25)
    x["mv_bias"] = (0.25, -0.125)
    if index % 3 == 2:
        x["pre_exposure"] = 0.00077
    return x


def raw_buffers(p):
    out = {}
    for name in BUFFERS:
        ptr, n = ctypes.c_size_t(), ctypes.c_size_t()
        if p.lib.wg_rt_buffer(p.rt, name.encode(), ctypes.byref(ptr), ctypes.byref(n)) != 0:
            continue
        nb = EXPOSURE_BYTES if name == "exposure.pair" else n.value
        buf = np.empty(nb, np.uint8)
        p._check(p.lib.wg_rt_download(p.rt, ptr.value, buf.ctypes.data, nb), name)
        out[name] = buf
    return out


def buffer_ranges(p, raw):
    """Written bytes from original FORMATS.md and the selected decoder, not the allocation's maximum capacity.

    runtime.cpp retains C at 8 bytes/pixel for its non-pack12 route. The selected pack12 decoder writes C4 at
    pixel*4 and post.hlsl reads the same C4 image. B is unused. No floating-point values are excluded here.
    """
    ranges = {name: {"downloaded_bytes": int(value.size), "checked_range": [0, int(value.size)],
                     "not_written_ranges": []} for name, value in raw.items()}
    kernels = p.kernels()
    if "pack12" in kernels.get("dec0", "") and "_p12" in kernels.get("post_last", ""):
        rw, rh = p.info()["render_size"]
        for name, active in (("dec0.C", int(rw) * int(rh) * 4), ("dec0.B", 0)):
            if name not in ranges:
                continue
            capacity = ranges[name]["downloaded_bytes"]
            if active > capacity:
                raise RuntimeError(f"{name}: active range exceeds reported capacity")
            ranges[name].update(checked_range=[0, active], not_written_ranges=[[active, capacity]],
                                reason="pack12 C4 image; retained C8 allocation tail" if active else "pack12 B is unused")
    return ranges


def poison_pack12_c(p, value):
    if "pack12" not in p.kernels().get("dec0", ""):
        raise RuntimeError("unused-capacity probe requires the selected pack12 decoder")
    ptr, size = ctypes.c_size_t(), ctypes.c_size_t()
    p._check(p.lib.wg_rt_buffer(p.rt, b"dec0.C", ctypes.byref(ptr), ctypes.byref(size)), "dec0.C")
    data = np.full(size.value, value, dtype=np.uint8)
    p.lib.wg_rt_upload.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    p._check(p.lib.wg_rt_upload(p.rt, ptr.value, data.ctypes.data, size.value), "poison dec0.C")
    return {"byte": value, "uploaded_bytes": int(size.value)}


def prepare_capacity_probe(frame, hip, candidate):
    if frame == 0:
        return {"frame": 0, "executed": False, "passed": None,
                "reason": "Named buffers are unavailable before the first frame; active C4 is still compared."}
    return {"frame": frame, "executed": True, "hip": poison_pack12_c(hip, 0xA0 + frame % 16),
            "candidate": poison_pack12_c(candidate, 0x50 + frame % 16)}


def first_difference(a, b, base=0):
    if a.size != b.size:
        return {"size_hip": int(a.size), "size_candidate": int(b.size)}
    differences = np.flatnonzero(a != b)
    if not differences.size:
        return None
    at = int(differences[0])
    return {"offset": base + at, "hip_hex": a[at:at + 16].tobytes().hex(),
            "candidate_hex": b[at:at + 16].tobytes().hex()}


def capacity_probe_counts(a, b, active, hip_poison, candidate_poison):
    return {"active_bytes_differ": int(np.count_nonzero(a[:active] != b[:active])),
            "hip_tail_bytes_changed": int(np.count_nonzero(a[active:] != hip_poison)),
            "candidate_tail_bytes_changed": int(np.count_nonzero(b[active:] != candidate_poison))}


def capacity_probe_sensitivity():
    """Actual sentinel-check arithmetic must reject a missing active store and an illegal tail store."""
    a, b = np.array([1, 2, 3, 4, 0xA0, 0xA0], np.uint8), np.array([1, 2, 3, 4, 0x50, 0x50], np.uint8)
    checks = {"different_untouched_tails": not any(capacity_probe_counts(a, b, 4, 0xA0, 0x50).values())}
    missing = b.copy()
    missing[3] = 0x50
    checks["missing_active_store_detected"] = capacity_probe_counts(a, missing, 4, 0xA0, 0x50)["active_bytes_differ"] == 1
    overrun = b.copy()
    overrun[4] = 7
    checks["tail_store_detected"] = capacity_probe_counts(a, overrun, 4, 0xA0, 0x50)["candidate_tail_bytes_changed"] == 1
    first = prepare_capacity_probe(0, None, None)
    checks["first_frame_has_no_buffer_access"] = first["executed"] is False and first["passed"] is None
    if not all(checks.values()):
        raise RuntimeError("capacity comparator mutation sensitivity failed")
    return checks


def tree_diff(a, b, prefix=""):
    """Differing values per leaf of two Pipeline.frame(debug=True) trees."""
    d = {}
    for k in a:
        x, y, path = a[k], b[k], prefix + k
        if isinstance(x, dict):
            d.update(tree_diff(x, y, path + "."))
        elif isinstance(x, np.ndarray):
            d[path] = int(x.size) if x.shape != y.shape else int(
                (x.view(np.uint8) != y.view(np.uint8)).reshape(x.shape + (-1,)).any(-1).sum())
        elif x is None or y is None:
            d[path] = 0 if x is y else 1
        else:
            d[path] = 0 if x == y else 1
    return d


def run_case(args, model, title, render, out_size, flags, resets, frames, trace):
    os.environ["WG_DX12_KERNELS"] = args.kernels
    mo = getattr(args, "motion_outres", False)
    mk = lambda backend: Pipeline((getattr(args, "candidate_build", None) or args.build) if backend != "hip" else args.build,
                                  args.rocm, str(model), render, out_size, flags, backend=backend,
                                  model_format="raw", check_packing=False,
                                  kernels={"options": "motion_outres=original"} if mo else None)
    hip = mk("hip")
    dx = mk(getattr(args, "backend", "dx12"))
    rows, tree, capacity_probes = {}, {}, []
    try:
        for i in range(frames):
            probe = None
            if getattr(args, "probe_unused_capacity", False):
                probe = prepare_capacity_probe(i, hip, dx)
            x = OC.frame_input(i, render, reset=i in resets)
            if mo:
                x = motion_outres_input(x, i, out_size)
            ta, tb = hip.frame(x, debug=True), dx.frame(x, debug=True)
            ra, rb = raw_buffers(hip), raw_buffers(dx)
            ranges, candidate_ranges = buffer_ranges(hip, ra), buffer_ranges(dx, rb)
            if ranges != candidate_ranges:
                raise RuntimeError("HIP/candidate buffer active ranges disagree")
            reset = bool(ta["state"]["info"]["reset"])
            skip = set(f"{s}.out" for s in OC.STALE) | ({"state." + n for n in ("latent_prev", "motion_history",
                                                                                 "history_prev")} if reset else set())
            if not OC.display_written(hip):
                skip.add("post.display")
            skip |= fusion_unwritten(dx.kernels())
            for name in BUFFERS:
                if name not in ra:
                    continue
                end = ranges[name]["checked_range"][1]
                r = rows.setdefault(name, {"bytes": end, "range": ranges[name], "differ": 0, "frames_differ": 0,
                                           "skipped": 0, "raw_capacity_differ": 0, "not_written_differ": 0,
                                           "first_differences": []})
                full_diff = int(np.count_nonzero(ra[name] != rb[name]))
                n = int(np.count_nonzero(ra[name][:end] != rb[name][:end]))
                r["raw_capacity_differ"] += full_diff
                r["not_written_differ"] += full_diff - n
                if full_diff and len(r["first_differences"]) < 8:
                    r["first_differences"].append({"frame": i,
                        "checked": first_difference(ra[name][:end], rb[name][:end]),
                        "not_written": first_difference(ra[name][end:], rb[name][end:], end)})
                if name in skip or end == 0:
                    r["skipped"] += 1
                    continue
                r["differ"] += n
                r["frames_differ"] += n > 0
            if probe is not None and probe["executed"]:
                start = ranges["dec0.C"]["checked_range"][1]
                probe["active_range"] = [0, start]
                probe["unused_range"] = [start, int(ra["dec0.C"].size)]
                probe.update(capacity_probe_counts(ra["dec0.C"], rb["dec0.C"], start,
                                                  probe["hip"]["byte"], probe["candidate"]["byte"]))
                probe["passed"] = not any(probe[k] for k in
                    ("active_bytes_differ", "hip_tail_bytes_changed", "candidate_tail_bytes_changed"))
            if probe is not None:
                capacity_probes.append(probe)
            for k, v in tree_diff(ta, tb).items():
                if k.startswith("post.display") and "post.display" in skip:
                    continue
                if any(k == s or k.startswith(s + ".") for s in OC.STALE) or (reset and k in OC.ON_RESET) or                         k in skip:
                    continue
                tree[k] = tree.get(k, 0) + v
            print(f"  frame {i}{' (reset)' if reset else ''}: {sum(rows[n]['differ'] for n in rows)} differing bytes so far",
                  flush=True)
        kernels = dx.kernels()
        if mo and (kernels.get("motion_last") != "output" or hip.kernels().get("motion_last") != "output"):
            raise RuntimeError("--motion-outres: the frames did not run the output-resolution motion kernels")
    finally:
        OC.close(hip, dx)
    return {"case": title, "render": render, "output": out_size, "frames": frames, "buffers": rows,
            "tree_values_differ": {k: v for k, v in tree.items() if v}, "kernels": kernels,
            "unused_capacity_probes": capacity_probes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="build dir: code objects, runtime library, dx12/manifest.json")
    ap.add_argument("--candidate-build", help="optional separate native/hybrid build; --build remains the HIP oracle")
    ap.add_argument("--kernels", required=True, help="WG_DX12_KERNELS for the hybrid run (names as launched, or all)")
    ap.add_argument("--backend", choices=("dx12", "d3d12"), default="dx12", help="candidate device; HIP stays the oracle")
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    ap.add_argument("--models", default=DEFAULT_MODELS, help="work dir for the synthetic model")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="only the small odd-size case")
    ap.add_argument("--include-4k", action="store_true", help="also compare 1920x1080 -> 3840x2160 Performance geometry")
    ap.add_argument("--probe-unused-capacity", action="store_true", help="poison C before frames 1 onward, verify C4 and tail")
    ap.add_argument("--motion-outres", action="store_true",
                    help="motion at output resolution (motion_outres=original) cases instead of the default ones")
    ap.add_argument("--json")
    ap.add_argument("--trace", help="new trace path; defaults to a unique directory inside the build")
    a = ap.parse_args(argv)
    if a.frames < 1:
        ap.error("--frames must be positive")
    if a.probe_unused_capacity and a.frames < 2:
        ap.error("--probe-unused-capacity needs at least two frames; frame 0 makes named buffers available")
    if a.quick and a.include_4k:
        ap.error("--quick and --include-4k cannot be combined")
    for k in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH", "HIP_CLANG_PATH"):
        os.environ.pop(k, None)
    model = Path(a.models) / "taps"
    if (Path(a.models) / "model.wgcm").exists():
        raise SystemExit("refusing to write synthetic parameters into a converted-model directory")
    if not (model / "raw").exists():
        SM.write_raw(str(model), SM.synthetic_tensors(OC.SEED, taps=SM.output_taps("random", OC.SEED)))
    trace = a.trace or os.path.join(tempfile.mkdtemp(prefix="wgcmp_", dir=a.build), "launch_trace.txt")
    if os.path.exists(trace):
        raise SystemExit("refusing to append to an existing trace: " + trace)
    Path(trace).parent.mkdir(parents=True, exist_ok=True)
    os.environ["WG_LAUNCH_TRACE"] = trace   # read once per process, at the first launch
    want = None
    if a.kernels not in ("all", "none", ""):
        want = set(a.kernels.split(","))
    elif a.kernels == "all":
        with open(os.path.join(a.candidate_build or a.build, "dx12", "manifest.json")) as f:
            want = required_kernels(a.backend, json.load(f), a.motion_outres)
    report, bad = {"backend": a.backend, "kernels": a.kernels, "trace": trace, "cases": [],
                   "required_kernels": sorted(want or ()), "include_4k": a.include_4k,
                   "motion_outres": a.motion_outres,
                   "capacity_comparator_mutations": capacity_probe_sensitivity()}, 0
    for title, render, out_size, flags, resets in selected_cases(a.include_4k, a.quick, a.motion_outres):
        print(f"{title}: {a.frames} frames, D3D12 kernels {a.kernels}", flush=True)
        r = run_case(a, model, title, render, out_size, flags, resets, a.frames, trace)
        report["cases"].append(r)
        for name, row in r["buffers"].items():
            note = f"  (not written by this path: {row['skipped']} frame(s) not counted)" if row["skipped"] else ""
            print(f"  {name:22s} {row['bytes']:>11d} bytes  differing {row['differ']:>10d}"
                  f"  in {row['frames_differ']} frame(s){note}")
            bad += row["differ"]
        for k, v in r["tree_values_differ"].items():
            print(f"  debug tree {k}: {v} values differ")
            bad += v
        bad += sum(not probe["passed"] for probe in r["unused_capacity_probes"] if probe["executed"])
    # every requested kernel must have run on D3D12
    counts = {}
    if os.path.exists(trace):
        with open(trace) as f:
            for line in f:
                t = line.split()
                if len(t) == 4 and t[3] == ("d3d12" if a.backend == "d3d12" else "dx12"):
                    counts[t[0]] = counts.get(t[0], 0) + 1
    missing = sorted(k for k in (want or ()) if not counts.get(k))
    report["dx12_launches"] = counts
    report["missing"] = missing
    print("D3D12 launches: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "none"))
    if missing:
        print("NOT RUN ON D3D12: " + ", ".join(missing))
    ok = bad == 0 and not missing
    print(f"compare: {'IDENTICAL' if ok else 'DIFFERENT'} ({bad} differing bytes / values)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
