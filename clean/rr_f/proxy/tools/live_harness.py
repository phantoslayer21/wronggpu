# SPDX-License-Identifier: MIT
"""Live-test harness driver (clean/rr-f-live2): builds the release hook ASI (--binding 2.7 or 2.14: the interface version the game ships), the
stand-in interface and tests/test_live_harness.cpp, prepares the captured frames (CAPTURE.md) as .lvf files, stages
per mode (rr_mode=exact / accurate / fast; clean/rr-f-s10-modes: ONE runtime library for all three modes, --rt; the older
two-library layout, --acc / --fast, still works) and runs the scenarios on the GPU
under the GPU lock (WG_GPU_LOCK, BUILDING.md); with --replay-acc / --replay-fast compares every 'match' output byte for byte with the offline
replay's (tools/rr_capture_replay.py, rr_test_host.exe: output rectangle at (1, 1) of a (Wo + 3) x (Ho + 2) texture).

    python clean/rr_f/proxy/tools/live_harness.py (--rt <runtime dir> | --acc <runtime dir> --fast <runtime dir>) --model <model.wgcm>
        --out <work dir> [--capture DIR] [--modes exact,accurate,fast] [--scenarios match,soak,inject]
        [--replay-acc DIR] [--replay-fast DIR] [--replay-exact DIR] [--soak N] [--no-build] [--binding 2.7|2.14]

The model is only handed over by path (wronggpu-rr.ini rr_model); this tool never opens it.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build as pbuild  # noqa: E402

# Optional GPU lock (WG_GPU_LOCK: a script run as `python <lock> --timeout <s> -- <command>`, e.g. to keep
# GPU jobs of several shells apart); unset: the command runs directly.
GPU_LOCK = os.environ.get("WG_GPU_LOCK", "")


def _locked(cmd, timeout="3600"):
    return ([sys.executable, GPU_LOCK, "--timeout", str(timeout), "--"] if GPU_LOCK else []) + list(cmd)
BINDING = os.path.join(PROXY, "binding", "rr-f-2.7.json")
CAPTURE = os.environ.get("WG_RR_CAPTURE", "")   # a capture folder of clean/rr_f/proxy (CAPTURE.md); local data, not in the repository


def build(out, binding=BINDING):
    n, gen, built = pbuild.build_hook(binding, os.path.join(out, "hook"), False, None)
    iface_dll, implib, iface_name = pbuild.fake_interface(binding, os.path.join(out, "iface"), n,
                                                          list(n["file_version"]))
    exe = os.path.join(out, "test_live_harness.exe")
    pbuild.sh(f'cl {pbuild.CLFLAGS} /I"{gen}" "{os.path.join(PROXY, "tests", "test_live_harness.cpp")}" "{implib}" '
              f'/Fo"{out}\\\\" /Fe"{exe}" /link d3d12.lib dxgi.lib version.lib psapi.lib', out)
    return dict(asi=built, asi_name=n["proxy_file"], iface=iface_dll, iface_name=iface_name, exe=exe)


def lvf(meta, folder, path):
    c, o, out = meta["constants"], meta["options"], meta["output"]
    cb = struct.pack("<80f", *(c["view_to_clip"] + c["clip_to_view"] + c["clip_to_lens_clip"] + c["clip_to_prev_clip"]
                               + c["prev_clip_to_clip"]))
    cb += struct.pack("<2f2f2f3f3f3f3f5f", *c["jitter"], *c["mv_scale"], *c["pinhole_offset"], *c["camera_pos"],
                      *c["camera_up"], *c["camera_right"], *c["camera_fwd"], c["near_plane"], c["far_plane"],
                      c["fov_y"], c["aspect"], c["mv_invalid_value"])
    cb += struct.pack("<8Bf", c["depth_inverted"], c["camera_motion_included"], c["mv_3d"], c["reset"],
                      c["orthographic"], c["mv_dilated"], c["mv_jittered"], 0, c.get("min_depth_separation", 0.0))
    assert len(cb) == 424
    ob = struct.pack("<IIIfffBBBBI16f16fB3x6II", o["mode"], o["output_width"], o["output_height"], o["sharpness"],
                     o["pre_exposure"], o["exposure_scale"], o["colour_is_hdr"], o["indicator_invert_x"],
                     o["indicator_invert_y"], 0, o["normal_roughness_packing"], *o["world_to_view"],
                     *o["view_to_world"], o.get("alpha_upscaling", 2), *(o.get("presets") or [0] * 6), 0)
    assert len(ob) == 192
    b = struct.pack("<IIIIIIII", 0x3146564C, meta["frame_index"], meta["preference_flags"], meta["api_minor"],
                    out["format"], out["rect"][2], out["rect"][3], out["state"])
    b += struct.pack("<II", c["version"], len(cb)) + cb + struct.pack("<II", o["version"], len(ob)) + ob
    ins = sorted(meta["inputs"], key=lambda i: i["type"])
    b += struct.pack("<I", len(ins))
    for i in ins:
        e = i["extent"]
        raw = os.path.join(folder, i["file"]).encode()
        b += struct.pack("<13I", i["type"], i["format"], i["width"], i["height"], i["row_pitch"], i["tag_format"],
                         i["state"], i["lifetime"], e["left"], e["top"], e["width"], e["height"], len(raw)) + raw
    with open(path, "wb") as fh:
        fh.write(b)


def prepare(capture, out):
    d = os.path.join(out, "frames")
    os.makedirs(d, exist_ok=True)
    folders = sorted(f for f in os.listdir(capture) if os.path.exists(os.path.join(capture, f, "frame.json")))
    for k, f in enumerate(folders):
        meta = json.load(open(os.path.join(capture, f, "frame.json")))
        lvf(meta, os.path.join(capture, f), os.path.join(d, f"frame_{k:03d}.lvf"))
    return d, len(folders)


def stage(out, mode, b, acc, fast, model):
    s = os.path.join(out, f"stage-{mode}")
    shutil.rmtree(s, ignore_errors=True)
    os.makedirs(s)
    shutil.copy(b["exe"], s)
    shutil.copy(b["iface"], os.path.join(s, b["iface_name"]))
    shutil.copy(b["asi"], os.path.join(s, b["asi_name"]))
    shutil.copy(os.path.join(acc, "wronggpu_rr_runtime.dll"), os.path.join(s, "wronggpu_rr_runtime.dll"))
    if fast:   # the two-library layout (before clean/rr-f-s10-modes); the single library needs no second file
        shutil.copy(os.path.join(fast, "wronggpu_rr_runtime.dll"), os.path.join(s, "wronggpu_rr_runtime_fast.dll"))
    with open(os.path.join(s, "wronggpu-rr.ini"), "w") as fh:
        fh.write(f"; live harness\nrr_mode={mode}\nrr_model={model}\n")
    return s


def run(s, frames, out, scen, n, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("WG_RR_")}
    # WG_RR_COMPILE_WAIT_MS (e.g. 900000) survives: a mode whose kernels the driver has not cached yet (first run of the
    # exact kernels) would otherwise pass frames through while they compile
    if "WG_RR_COMPILE_WAIT_MS" in os.environ:
        env["WG_RR_COMPILE_WAIT_MS"] = os.environ["WG_RR_COMPILE_WAIT_MS"]
    env.update(extra_env or {})
    o = os.path.join(out, scen)
    shutil.rmtree(o, ignore_errors=True)
    os.makedirs(o)
    for f in ("wronggpu-rr.log",):
        if os.path.exists(os.path.join(s, f)):
            os.remove(os.path.join(s, f))
    r = subprocess.run(_locked([os.path.join(s, "test_live_harness.exe"),
                        frames, o, scen, str(n)]), cwd=s, env=env, capture_output=True, text=True, timeout=7200)
    shutil.copy(os.path.join(s, "wronggpu-rr.log"), os.path.join(o, "wronggpu-rr.log")) \
        if os.path.exists(os.path.join(s, "wronggpu-rr.log")) and os.path.isdir(o) else None
    with open(os.path.join(o, "stdout.txt"), "w") as fh:
        fh.write(r.stdout + "\n--- stderr\n" + r.stderr)
    last = [l for l in r.stdout.splitlines() if l.startswith("summary") or l.startswith("RESULT") or "FAIL" in l]
    return r.returncode, last, o


def compare(o, replay, count):
    """match outputs vs the offline replay's host outputs (cropped from (1, 1))."""
    res = []
    for i in range(count):
        a = np.fromfile(os.path.join(o, f"frame_{i:03d}.out"), np.uint16)
        meta = [json.loads(l.replace("\\", "/")) for l in open(os.path.join(replay, "work", "host", "meta.jsonl"))
                if l.strip()][i]
        tw, th = meta["out_tex"][0], meta["out_tex"][1]
        r = np.fromfile(os.path.join(replay, "work", "host", f"frame_{i:03d}.out"), np.uint16).reshape(th, tw, 4)
        h, w = th - 2, tw - 3
        r = r[1:1 + h, 1:1 + w]
        a = a.reshape(h, w, 4)
        if np.array_equal(a, r):
            res.append("=")
        else:
            fa, fr = a.view(np.float16).astype(np.float32), r.view(np.float16).astype(np.float32)
            d = np.abs(fa[..., :3] - fr[..., :3])
            res.append(f"X(px {int((a != r).any(-1).sum())}, max {np.nanmax(d):.4g})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", help="the single runtime folder holding all three modes (clean/rr-f-s10-modes)")
    ap.add_argument("--acc")
    ap.add_argument("--fast")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture", default=CAPTURE)
    ap.add_argument("--modes", default="accurate,fast")
    ap.add_argument("--replay-exact")
    ap.add_argument("--scenarios", default="match,soak,inject")
    ap.add_argument("--replay-acc")
    ap.add_argument("--replay-fast")
    ap.add_argument("--soak", type=int, default=600)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--binding", default="2.7", help="2.7 or 2.14 (the game's interface version)")
    a = ap.parse_args()
    if a.rt:
        a.acc, a.fast = a.rt, None
    elif not (a.acc and a.fast):
        ap.error("give --rt, or both --acc and --fast")
    os.makedirs(a.out, exist_ok=True)
    bfile = os.path.join(a.out, "build.json")
    if a.no_build:
        b = json.load(open(bfile))
    else:
        b = build(a.out, os.path.join(PROXY, "binding", f"rr-f-{a.binding}.json"))
        json.dump(b, open(bfile, "w"))
    frames, nf = prepare(a.capture, a.out)
    ok = True
    for mode in a.modes.split(","):
        s = stage(a.out, mode, b, a.acc, a.fast, a.model)
        for scen in a.scenarios.split(","):
            n = {"match": 1, "soak": a.soak, "inject": 24}[scen]
            env = ({"WG_RR_TEST_FAIL": "3,4", "WG_RR_TEST_CRASH": "12", "WG_RR_COMPILE_WAIT_MS": "0"}
                   if scen == "inject" else None)
            rc, last, o = run(s, frames, os.path.join(a.out, mode), scen, n, env)
            print(f"[{mode} {scen}] rc={rc} " + " | ".join(last[-3:]))
            ok = ok and rc == 0
            rp = {"accurate": a.replay_acc, "fast": a.replay_fast, "exact": a.replay_exact}[mode]
            if scen == "match" and rp and rc == 0:
                cmp_ = compare(o, rp, nf)
                print(f"[{mode} match vs replay {rp}] {''.join(c if c == '=' else ' ' + c + ' ' for c in cmp_)}")
                ok = ok and all(c == "=" for c in cmp_)
            if scen == "match":
                for f in os.listdir(o):
                    if f.endswith(".out"):
                        os.remove(os.path.join(o, f))
    print("ALL OK" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
