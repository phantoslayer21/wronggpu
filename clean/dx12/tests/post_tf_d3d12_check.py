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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kernels", default=",".join(KINDS))
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if not a.run:
        return 0
    kernels = [k for k in a.kernels.split(",") if k]
    res = OPS.run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)), kernels, kinds=KINDS)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k not in ("cases", "replay_tail")})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
