# SPDX-License-Identifier: MIT
"""Model L (issue #89): the fallback route of l_post=own on native D3D12 (frames whose render footprint the pack12 kernels
do not take, runtime.cpp recon2_supported false: the unpacked coefficient store l_coef, l_post_recon, then the filter
w64/post_filter_c1_s4) against HIP, byte for byte, chained as the frame runs it.

    python clean/l/runner/l_d3d12_fallback_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

<dir>: l_ops.co, l_post.co, post_w64.co, swin.co (clean/l/runner/l_frame_check.py --prepare builds them all), the D3D12
ports in <dir>/dx12 and kernel_replay.exe (clean/l/runner/l_d3d12_check.py prepare_replay). --run: one GPU slot.

Per frame (FRAMES: recon2_supported false for each, asserted; D and O not multiples of the kernels' tiles, so every
launch has edge tiles: D columns vs the 32-column reconstruction tile, D row pairs vs its 8, O columns vs the filter's 40,
O row pairs vs its 8; render sizes not multiples of 32):
1. l_coef (l_ops.co): a decoder map y2 [fh][fw][32] (FP16; values past the clamp, tanh's saturation and inside its
   polynomial) -> A, B, latent (4 bytes a pixel), C (8 bytes a pixel);
2. l_post_recon (l_post.co): HIP's A, B, C of step 1 with the frame's colour, history and motion surfaces (random, or
   with NaN / infinity / 60000 colours, NaN / infinity / -1 history and far motion: "specials"), the exposures by value
   or from the exposure pair -> full history, reduced codes, display image;
3. w64/post_filter_c1_s4 (post_w64.co): HIP's display image of step 2 -> the FP32 output, dec_scale from the argument or
   the exposure pair.
Each step's D3D12 port gets the same inputs as HIP's kernel; since every step's outputs are compared, the chain is the
route's. The rules are clean/l/runner/l_d3d12_ops_check.py's: raw downloaded bytes; every output buffer filled
beforehand with its back end's own sentinel (HIP 0xA5, D3D12 0x5A) and followed by 64 slack bytes that no back end may
write (the display's pitch padding too); a negative control (one case again with half its workgroups) must be caught;
Model M's swin_block_c96_nc through the same engine as the control. Output: counts only, no timing.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_d3d12_ops_check as OPS  # noqa: E402
from clean.l.runner import l_post_check as LPC  # noqa: E402
from clean.runner import post as RP  # noqa: E402

SLACK = 64
# (rw, rh, ow, oh, kind, expo, reset); D = (3 O + 1) / 2
FRAMES = [(150, 90, 121, 73, "random", False, False),       # D 182 x 110
          (100, 60, 77, 45, "specials", True, False),       # D 116 x 68
          (96, 60, 64, 40, "random", False, True),          # D 96 x 60 = R
          (130, 70, 99, 57, "offset_0.1", True, False),     # D 149 x 86
          (11, 8, 7, 5, "specials", False, False)]          # D 11 x 8


def _cdiv(a, b):
    return -(-a // b)


def _f32_bits(v):
    return int(np.array([v], np.float32).view(np.uint32)[0])


def _defined(n, slack=SLACK):
    return np.concatenate([np.ones(n, bool), np.zeros(slack, bool)])


def coef_case(rng, rw, rh):
    from clean.runner import dec0
    fw, fh = _cdiv(rw, 32) * 32, _cdiv(rh, 32) * 32
    npx = rw * rh
    S, T = float(rng.uniform(0.5, 4.0)), float(rng.uniform(0.3, 3.0))
    blob = dec0.pack({"W": np.zeros((68, 64), np.float32), "b": np.zeros(68, np.float32), "S": S, "T": T})
    P = bytes(dec0.DecParams(rw=rw, rh=rh, fw=fw, fh=fh, reset=0, pad0=0, dw=2 * rw, dh=2 * rh, kx=0.5, ky=0.5,
                             jx=0.25, jy=-0.25, pjx=0.0, pjy=0.0))
    sizes = (npx * 4, npx * 4, npx * 8, npx * 4)   # A, B, C, latent
    return {"kernel": "l_coef", "co": "l_ops.co", "grid": _cdiv((fh // 2) * (fw // 2), 128), "block": 256,
            "args": [("buf", OPS._coef_map(rng, fw * fh))] + [("out", n + SLACK) for n in sizes] +
                    [("buf", blob), ("bytes", P)],
            "defined": [_defined(n) for n in sizes], "sizes": sizes}


def post_case(rng, x, coef_out, coef_sizes, expo):
    prm = RP.post_params(x)
    dw, dh = prm.dw, prm.dh
    data = RP.post_device_inputs(x)
    for k, o, n in zip(("A", "B", "C"), coef_out[:3], coef_sizes[:3]):
        assert len(data[k]) == n
        data[k] = o[:n]
    ev = None
    if expo:
        ev = np.array([prm.e, 0.0 if rng.random() < 0.5 else prm.e_prev], np.float32)
        prm.e, prm.e_prev = float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.5, 2.0))
    n_hist, n_red, n_disp = dw * dh * 8, max(4, (dw // 2) * (dh // 2) * 4), RP.display_bytes(dw, dh)
    pitch = RP.display_pitch(dw)
    disp_def = np.zeros((3, dh, pitch, 2), bool)
    disp_def[:, :, :dw, :] = True
    red_def = np.zeros(n_red, bool)
    red_def[:(dw // 2) * (dh // 2) * 4] = True
    args = [("buf", data[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far")]
    args += [("buf", RP.pack({"s": np.array(LPC.S)})), ("out", n_hist + SLACK), ("out", n_red + SLACK),
             ("out", n_disp + SLACK), ("bytes", bytes(prm))]
    args += [("buf", ev.tobytes())] if expo else [("null",)]
    return {"kernel": "l_post_recon", "co": "l_post.co", "grid": RP.recon_grid(dw, dh), "block": 256, "args": args,
            "defined": [_defined(n_hist), np.concatenate([red_def, np.zeros(SLACK, bool)]),
                        np.concatenate([disp_def.reshape(-1), np.zeros(SLACK, bool)])],
            "n_disp": n_disp, "e": float(ev[0]) if expo else float(prm.e)}


def filter_case(rng, display, dw, dh, ow, oh, e, expo):
    n_out = ow * oh * 12
    args = [("buf", display), ("out", n_out + SLACK), ("u32", dw), ("u32", dh), ("u32", ow), ("u32", oh),
            ("u32", _f32_bits(np.float32(0.01 / (e if e != 0.0 else 1.0))))]
    args += [("buf", np.array([e, 1.0, 0.0, 0.0, 0.0], np.float32).tobytes())] if expo else [("null",)]
    return {"kernel": "w64/post_filter_c1_s4", "co": "post_w64.co", "tag": "w64", "hip_kernel": "post_filter_c1_s4",
            "grid": _cdiv(ow, 40) * _cdiv(_cdiv(oh, 2), 8), "block": 64, "args": args, "defined": [_defined(n_out)]}


def compare(got, want, dmask):
    a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
    o = {"bytes": len(want)}
    if a.size != b.size:
        return dict(o, identical=False, differ=-1, unwritten_bad=-1)
    o["defined_bytes"] = int(dmask.sum())
    o["differ"] = int(np.sum((a != b) & dmask))
    o["unwritten_bad"] = int(np.sum(~dmask & ((b != OPS.HIP_SENTINEL) | (a != OPS.D3D12_SENTINEL))))
    o["identical"] = o["differ"] == 0 and o["unwritten_bad"] == 0
    return o


def run(build, rocm, seeds):
    g = C32.Gpu(build, rocm)
    mods = {}

    def module(co):
        if co not in mods:
            with open(os.path.join(build, co), "rb") as f:
                mods[co] = g.hip.load_module(f.read())
        return mods[co]

    cdir = os.path.join(build, "replay-fallback")
    os.makedirs(cdir, exist_ok=True)
    lines, todo, neg, frames = [], [], None, []

    def add(name, case, step, label):
        nonlocal lines, neg
        want = OPS.hip_run(g, module(case["co"]), case)
        lines += OPS.replay_lines(name, case, cdir)
        todo.append((name, case, want, step, label))
        if neg is None and step == "l_post_recon" and case["grid"] >= 4:
            neg = (name, case, want)
            lines += OPS.replay_lines(name, case, cdir, grid=case["grid"] // 2, prefix="negative_")
        return want

    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for fi, (rw, rh, ow, oh, kind, expo, reset) in enumerate(FRAMES):
            dw, dh = (3 * ow + 1) // 2, (3 * oh + 1) // 2
            assert not RP.recon2_supported(rw, rh, dw, dh), (rw, rh, dw, dh)
            label = f"R {rw}x{rh} O {ow}x{oh} D {dw}x{dh} {kind}" + (" expo" if expo else "") + (" reset" if reset else "")
            frames.append({"label": label, "recon2_supported": False,
                           "edge": {"D_cols_mod_32": dw % 32, "D_rowpairs_mod_8": _cdiv(dh, 2) % 8,
                                    "O_cols_mod_40": ow % 40, "O_rowpairs_mod_8": _cdiv(oh, 2) % 8,
                                    "R_mod_32": [rw % 32, rh % 32]}})
            base = f"s{si}_f{fi}"
            coef = coef_case(rng, rw, rh)
            coef_out = add(f"l_coef_{base}", coef, "l_coef", label)
            x = LPC.make_case(rng, rw, rh, dw, dh, reset, "random" if kind == "specials" else kind)
            if kind == "specials":
                x = OPS._post_specials(rng, x)
            post = post_case(rng, x, coef_out, coef["sizes"], expo)
            post_out = add(f"l_post_recon_{base}", post, "l_post_recon", label)
            filt = filter_case(rng, post_out[2][:post["n_disp"]], dw, dh, ow, oh, post["e"], expo)
            add(f"w64_post_filter_c1_s4_{base}", filt, "w64/post_filter_c1_s4", label)
        for ci, case in enumerate(OPS.control_cases("swin_block_c96_nc", rng)):
            add(f"control_swin_block_c96_nc_s{si}_{ci}", case, "control swin_block_c96_nc", "Model M control")
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=150, creationflags=0x08000000)

    def read(path):
        return open(path, "rb").read() if os.path.exists(path) else b""

    def judge(name, case, want, prefix=""):
        defined = case.get("defined")
        outs = [compare(read(os.path.join(cdir, f"{prefix}{name}.o{oi}")), w,
                        defined[oi] if defined else np.ones(len(w), bool)) for oi, w in enumerate(want)]
        return all(o["identical"] for o in outs), outs

    res = {"replay_exit": r.returncode, "replay_tail": r.stdout[-600:] + r.stderr[-300:], "frames": frames, "cases": []}
    for name, case, want, step, label in todo:
        ok, outs = judge(name, case, want)
        res["cases"].append({"case": name, "step": step, "label": label, "identical": ok, "outputs": outs})
    caught = None
    if neg:
        ok, outs = judge(neg[0], neg[1], neg[2], prefix="negative_")
        caught = not ok and all(o["differ"] >= 0 for o in outs)
        res["negative_control"] = {"case": neg[0], "half_grid": True, "caught": caught,
                                   "differ": sum(max(o["differ"], 0) for o in outs)}
    steps = sorted({c["step"] for c in res["cases"]})
    res["per_step"] = {s: {"cases": sum(c["step"] == s for c in res["cases"]),
                           "identical": sum(c["identical"] for c in res["cases"] if c["step"] == s),
                           "bytes": sum(o["bytes"] for c in res["cases"] if c["step"] == s for o in c["outputs"]),
                           "differ": sum(max(o["differ"], 0) for c in res["cases"] if c["step"] == s
                                         for o in c["outputs"])} for s in steps}
    res["pass"] = (r.returncode == 0 and caught is True and
                   all(v["cases"] > 0 and v["identical"] == v["cases"] for v in res["per_step"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not a.run:
        return 0
    res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k not in ("cases", "replay_tail", "frames")})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
