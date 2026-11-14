# SPDX-License-Identifier: MIT
"""The analytic-filter D3D12 ports of Model M's fused output kernels (clean/dx12/kernels/post.hlsl, WG_LZ_ANALYTIC)
against HIP's post_w64.co on the same synthetic inputs, byte for byte; the original-filter port (post_w64__orig.co,
already proven on whole frames by clean/dx12/compare.py) as the Model M control on the same harness.

    python clean/dx12/tests/post_tf_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

<dir> holds post_w64.co and post_w64__orig.co (clean/runner/build.py's builds of kernels/post.hip), the D3D12 ports
(clean/dx12/build_dx12.py --only w64/post_tf_f32_p12_diet_vsit,w64/post_tf4_dl_f32_p12_diet_vsit,native_fill: both
builds of each name) and kernel_replay.exe (clean/l/runner/l_d3d12_check.py prepare_replay). --run runs in one GPU slot.
The engine and rules are clean/l/runner/l_d3d12_ops_check.py's: the RAW downloaded bytes of every output (the full
history with its alpha, the reduced codes, the FP32 output image) are compared; each output buffer starts filled with
its back end's own sentinel (HIP 0xA5, D3D12 0x5A) and 64 slack bytes follow it, which no back end may write; a negative
control (one case again on D3D12 with half its workgroups) must be caught. Output: counts only, no timing.

Kernels:
- w64/post_tf_f32_p12_diet_vsit (analytic port) vs HIP post_tf_f32_p12_diet_vsit of post_w64.co;
- w64/post_tf4_dl_f32_p12_diet_vsit (analytic port; D3D12 only, NW = 4 row pairs per step, 256 lanes) vs HIP
  post_tf_f32_p12_diet_vsit (frames that its cap of 352 takes) or post_tf_dl_f32_p12 (native AA: D = 1.5 R), whose
  every output pixel is the same function (the exposure pair holds the ratio and dec_scale that the non-diet kernel
  computes itself);
- control: w64/post_tf_f32_p12_diet_vsit (original-filter port, five FP16 weights as arguments) vs post_w64__orig.co.
  The same name as the first: the D3D12 device runs the port of the code object the case loads under "w64".
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_d3d12_ops_check as OPS  # noqa: E402
from clean.reference import post as MP  # noqa: E402
from clean.runner import post as RP  # noqa: E402
from clean.runner import post_sizes as PZ  # noqa: E402

SLACK = 64
CPAIRS = 68   # runtime.cpp kTfCpairs
# (rw, rh, ow, oh, reset): Quality (D = 2.25 R), Performance (D = 3 R), odd sizes, reset, a wide frame of 64 strips whose
# chunk plan has k1 > 0 chunks of 68 row pairs and then chunks of 36
TF_FRAMES = [(64, 36, 96, 54, False), (48, 27, 96, 54, False), (50, 29, 99, 57, True), (70, 41, 105, 61, False),
             (1280, 187, 2560, 280, False)]
# post_tf4: Quality frames (HIP post_tf_f32_p12_diet_vsit takes them too) and native AA (D = 1.5 R: HIP post_tf_dl)
TF4_FRAMES = [(64, 36, 96, 54, False), (50, 29, 99, 57, True), (64, 36, 64, 36, False), (61, 33, 61, 33, False),
              (96, 54, 96, 54, True)]
# FP16 bits of the original filter's five weights for the control (spec 06 draft 4 gives them from the model file; any
# FP16 set exercises the port)
ORIG_W = [np.float16(v).view(np.uint16) for v in (-0.0908, 0.1129, 0.6400, 0.4092, -0.0706)]


def _f32_bits(v):
    return int(np.float32(v).view(np.uint32))


def tf_case(kernel, rng, frame, co, hip_kernel, block, hip_block, taps, expo_pair):
    rw, rh, ow, oh, reset = frame
    dw, dh = MP.recon_size_for(ow), MP.recon_size_for(oh)
    x = MP.random_inputs(rng, rw, rh, dw, dh, reset=reset)
    prm = RP.post_params(x)
    data = RP.pack12_inputs(RP.post_device_inputs(x))
    k1, c2, nch = PZ.tf_chunks(ow, oh, CPAIRS)
    e, ep = prm.e, prm.e_prev
    dec_scale = float(np.float32(0.01 / e))
    ratio = float(np.float32(np.float64(e) / np.float64(ep)))
    n_hist, n_red, n_out = dw * dh * 8, max(4, (dw // 2) * (dh // 2) * 4), ow * oh * 12
    red_def = np.zeros(n_red, bool)
    red_def[:(dw // 2) * (dh // 2) * 4] = True
    slack = np.zeros(SLACK, bool)
    defined = [np.concatenate([np.ones(n_hist, bool), slack]), np.concatenate([red_def, slack]),
               np.concatenate([np.ones(n_out, bool), slack])]
    args = [("buf", data[k]) for k in ("color", "A")] + [("null",)]
    args += [("buf", data[k]) for k in ("C", "hist_prev", "m_near", "m_center", "m_far")]
    args += [("buf", RP.pack({"s": np.array(rng.uniform(0.5, 4.0))})), ("out", n_hist + SLACK), ("out", n_red + SLACK),
             ("out", n_out + SLACK), ("bytes", bytes(prm))]
    args += [("buf", np.array([e, ep, 0.0, ratio, dec_scale], np.float32).tobytes())] if expo_pair else [("null",)]
    args += [("u32", ow), ("u32", oh), ("u32", _f32_bits(dec_scale)), ("u32", _f32_bits(1.0)), ("u32", 12 * ow),
             ("u32", CPAIRS), ("u32", k1), ("u32", c2)]
    if taps:
        args += [("u32", int(w)) for w in ORIG_W]
    return {"kernel": kernel, "co": co, "tag": "w64", "hip_kernel": hip_kernel, "grid": PZ.PS._cdiv(ow, 40) * nch,
            "block": block, "hip_block": hip_block, "args": args, "defined": defined,
            "label": f"{rw}x{rh}->{ow}x{oh} (D {dw}x{dh})" + (" reset" if reset else "") + (" expo" if expo_pair else "")}


def tf_cases(kernel, rng):
    return [tf_case(kernel, rng, f, "post_w64.co", "post_tf_f32_p12_diet_vsit", 512, 512, False, i % 2 == 0)
            for i, f in enumerate(TF_FRAMES)]


def tf4_cases(kernel, rng):
    out = []
    for i, f in enumerate(TF4_FRAMES):
        rw, rh, ow, oh, _ = f
        dw, dh = MP.recon_size_for(ow), MP.recon_size_for(oh)
        hk = "post_tf_f32_p12_diet_vsit" if RP.tf_supported(rw, rh, dw, dh) else "post_tf_dl_f32_p12"
        assert RP.tf_supported(rw, rh, dw, dh, 4, RP.TF4_DL_CAP), f
        # the non-diet HIP kernel computes the ratio and takes dec_scale from its argument: the exposure pair is given
        out.append(tf_case(kernel, rng, f, "post_w64.co", hk, 256, 512, False, True if hk == "post_tf_dl_f32_p12"
                           else i % 2 == 0))
    return out


def control_cases(kernel, rng):
    return [dict(tf_case(kernel, rng, f, "post_w64__orig.co", "post_tf_f32_p12_diet_vsit", 512, 512, True, i % 2 == 0),
                 kernel="w64/post_tf_f32_p12_diet_vsit") for i, f in enumerate(TF_FRAMES[:4])]


KINDS = {"w64/post_tf_f32_p12_diet_vsit": tf_cases, "w64/post_tf4_dl_f32_p12_diet_vsit": tf4_cases,
         "orig_w64/post_tf_f32_p12_diet_vsit": control_cases}

# ---- the direct forms (rs=direct: the colour from a texture, the output into a texture) -------------------------------
# No HIP kernel of their own: form 3 is form 0 followed by the output store (post.hlsl). With color_pi 1, an RGBA32F
# colour texture holding the colour buffer's values and an RGBA32F output, the direct port's output texels must be
# HIP's FP32 output (form 0) with alpha 1, its history and reduced codes HIP's, and its out_p buffer untouched.
def orig_direct_cases(kernel, rng):
    """Control: the original filter's direct port (post_w64__orig.co) against HIP's original form 0."""
    return [dict(c, kernel="w64/post_tf_f32_p12_diet_vsit_direct") for c in control_cases(kernel, rng)]


DIRECT = {"w64/post_tf_f32_p12_diet_vsit_direct": (tf_cases, "w64/post_tf_f32_p12_diet_vsit"),
          "w64/post_tf4_dl_f32_p12_diet_vsit_direct": (tf4_cases, "w64/post_tf4_dl_f32_p12_diet_vsit"),
          "orig_w64/post_tf_f32_p12_diet_vsit_direct": (orig_direct_cases, "w64/post_tf_f32_p12_diet_vsit")}
TEX_FILL = 0x5A5A5A5A


def direct_run(build, rocm, seeds):
    import subprocess
    from clean.l.runner import l_c32_check as C32
    g = C32.Gpu(build, rocm)
    mods = {}
    cdir = os.path.join(build, "replay-direct")
    os.makedirs(cdir, exist_ok=True)
    lines, todo, neg = [], [], None
    for dk, (gen, _) in DIRECT.items():
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            for ci, case in enumerate(gen(dk, rng)):
                if case["co"] not in mods:
                    with open(os.path.join(build, case["co"]), "rb") as f:
                        mods[case["co"]] = g.hip.load_module(f.read())
                name = f"{dk.replace('/', '_')}_s{si}_{ci}"
                want = OPS.hip_run(g, mods[case["co"]], case)
                rw, rh, ow, oh = (int(v) for v in case["label"].split()[0].replace("->", "x").split("x"))
                # kernel_replay direct lines: color RGBA32F (the colour buffer's values), depth and motion of zeros (not
                # read by these kernels), the output RGBA32F filled with the sentinel word; render rectangle at (0, 0)
                case = dict(case, direct=[
                    ("color", OPS.DXGI_R32G32B32A32_FLOAT, rw, rh, case["args"][0][1]),
                    ("depth", OPS.DXGI_R32_FLOAT, rw, rh, bytes(rw * rh * 4)),
                    ("motion", OPS.DXGI_R32G32_FLOAT, rw, rh, bytes(rw * rh * 8)),
                    ("output", OPS.DXGI_R32G32B32A32_FLOAT, ow, oh,
                     np.full(ow * oh * 4, TEX_FILL, np.uint32).tobytes())], direct_params=(0, 0, rw, rh))
                body = OPS.replay_lines(name, case, cdir)
                lines += body[:-1] + [f"direct_out {name}.tex", ""]
                todo.append((name, dk, case, want, ow, oh))
                if neg is None and case["grid"] >= 4:
                    neg = (name, case, ow, oh)
                    nl = OPS.replay_lines(name, case, cdir, grid=case["grid"] // 2, prefix="negative_")
                    lines += nl[:-1] + [f"direct_out negative_{name}.tex", ""]
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=150, creationflags=0x08000000)

    def read(path):
        return open(path, "rb").read() if os.path.exists(path) else b""

    def judge(name, case, want, ow, oh, prefix=""):
        """(identical, counts): history and reduced as l_d3d12_ops_check (defined bytes equal, slack keeps the
        sentinels); out_p untouched by the direct port; texture RGB = HIP's FP32 output, alpha = 1.0."""
        o = {}
        bad = 0
        for oi in (0, 1):
            got = read(os.path.join(cdir, f"{prefix}{name}.o{oi}"))
            a, b = np.frombuffer(got, np.uint8), np.frombuffer(want[oi], np.uint8)
            m = case["defined"][oi]
            d = int(np.sum((a != b) & m)) if a.size == b.size else -1
            u = int(np.sum(~m & ((b != OPS.HIP_SENTINEL) | (a != OPS.D3D12_SENTINEL)))) if a.size == b.size else -1
            o[f"o{oi}"] = {"bytes": len(want[oi]), "differ": d, "unwritten_bad": u}
            bad += d != 0 or u != 0
        outp = np.frombuffer(read(os.path.join(cdir, f"{prefix}{name}.o2")), np.uint8)
        o["out_p_untouched"] = bool(outp.size and np.all(outp == OPS.D3D12_SENTINEL))
        tex = np.frombuffer(read(os.path.join(cdir, f"{prefix}{name}.tex")), np.uint32)
        ref = np.frombuffer(want[2][:ow * oh * 12], np.uint32).reshape(-1, 3)
        if tex.size == ow * oh * 4:
            t = tex.reshape(-1, 4)
            o["tex"] = {"texels": ow * oh, "rgb_bytes": ow * oh * 12, "rgb_words_differ": int(np.sum(t[:, :3] != ref)),
                        "alpha_not_one": int(np.sum(t[:, 3] != 0x3F800000))}
            bad += o["tex"]["rgb_words_differ"] != 0 or o["tex"]["alpha_not_one"] != 0
        else:
            o["tex"] = {"missing": True}
            bad += 1
        bad += not o["out_p_untouched"]
        return bad == 0, o

    res = {"replay_exit": r.returncode, "replay_tail": r.stdout[-600:] + r.stderr[-300:], "cases": []}
    for name, dk, case, want, ow, oh in todo:
        ok, o = judge(name, case, want, ow, oh)
        res["cases"].append({"case": name, "kernel": dk, "label": case["label"], "identical": ok, "outputs": o})
    if neg:
        name, case, ow, oh = neg
        want = next(t[3] for t in todo if t[0] == name)
        ok, o = judge(name, case, want, ow, oh, prefix="negative_")
        res["negative_control"] = {"case": name, "half_grid": True, "caught": not ok,
                                   "tex_rgb_words_differ": o["tex"].get("rgb_words_differ")}
    res["per_kernel"] = {k: {"cases": sum(c["kernel"] == k for c in res["cases"]),
                             "identical": sum(c["identical"] for c in res["cases"] if c["kernel"] == k),
                             "tex_rgb_bytes": sum(c["outputs"]["tex"].get("rgb_bytes", 0) for c in res["cases"]
                                                  if c["kernel"] == k)}
                         for k in DIRECT}
    res["pass"] = (r.returncode == 0 and bool(neg) and res["negative_control"]["caught"]
                   and all(v["cases"] > 0 and v["identical"] == v["cases"] for v in res["per_kernel"].values()))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kernels", default=",".join(KINDS))
    ap.add_argument("--direct", action="store_true", help="the direct forms instead (see DIRECT)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not a.run:
        return 0
    kernels = [k for k in a.kernels.split(",") if k]
    seeds = tuple(range(89, 89 + a.seeds))
    res = direct_run(a.build, a.rocm, seeds) if a.direct else OPS.run(a.build, a.rocm, seeds, kernels, kinds=KINDS)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k not in ("cases", "replay_tail")})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
