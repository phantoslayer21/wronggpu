# SPDX-License-Identifier: MIT
"""Model L (issue #89): the D3D12 ports of the l_ops.co kernels, the halo input encoder and the reconstruction kernel
against their HIP kernels on the same synthetic inputs, byte for byte (clean/l/kernels/l_ops.hip, l_post.hip; ports
clean/dx12/kernels/l_ops_*.hlsl, l_post.hlsl).

    python clean/l/runner/l_d3d12_ops_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_d3d12_ops_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] [--kernels a,b] --json out.json

--prepare (CPU) builds into <dir>: l_ops.co (clean/l/runner/l_c32_check.py compile_kernels), the D3D12 ports
(clean/dx12/build_dx12.py --only, with WG_TOOLCHAINS / WG_DXC_DIR as BUILDING.md describes) and kernel_replay.exe.
--run (GPU, one GPU slot) runs every case on HIP in this process and the same inputs through kernel_replay on the native
D3D12 device in a child process, then compares every output buffer. The rules (as clean/l/runner/l_d3d12_check.py):
- the RAW downloaded bytes of both back ends are compared (nothing is decoded and encoded again);
- each output buffer starts filled with its back end's own non-zero sentinel (HIP 0xA5, D3D12 0x5A), so a byte that
  either path, or both, leaves unwritten differs; where a case marks bytes that no back end may write (the
  reconstruction's display pitch padding and the slack after its buffers), each back end must have kept its own
  sentinel there;
- a negative control (one case again on D3D12 with half its workgroups) must be caught;
- Model M's swin_block_c96_nc (swin.co, its port already proven) runs through the same engine as a control
  (--no-control skips it).
Output: counts only, no timing.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner.l_d3d12_check import D3D12_SENTINEL, HIP_SENTINEL, prepare_replay  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402

# (rows, cols) of the pixel maps; for the halo maps of the encoder these are (fh + 4, fw + 4) sizes
MAPS = [(16, 24), (36, 68), (8, 8), (68, 100), (5, 7)]


def _fp8_map(rng, n, scale=1.5):
    return fp8_encode(q8(rng.normal(0, scale, n))).tobytes()


def _linear_blob(rng):
    """enc0.Win1 / enc0.Win2 layout: bias tile vector (32 floats), then 4 fragments of 256 FP8 bytes."""
    bias = (rng.normal(0, 0.5, 32)).astype(np.float32).tobytes()
    frags = fp8_encode(q8(rng.normal(0, 0.25, 4 * 256))).tobytes()
    return bias + frags


def linear_cases(kernel, rng):
    out_bytes_per = 32 if kernel == "l_linear_32_32_relu" else 64
    cases = []
    for (rows, cols) in MAPS:
        n = rows * cols
        cases.append({"kernel": kernel, "grid": -(-n // 128), "block": 256,
                      "args": [("buf", _fp8_map(rng, n * 32)), ("out", n * out_bytes_per), ("buf", _linear_blob(rng)),
                               ("u32", n)]})
    return cases


def control_cases(kernel, rng):
    """Control: Model M's swin_block_c96_nc (swin.co), whose D3D12 port is already proven byte for byte (whole frames,
    clean/dx12/compare.py), through this harness's engine: if it is identical here too, the harness is sound."""
    from clean.l.runner.l_d3d12_check import _params_c96
    cases = []
    for (H, W, (ox, oy)) in C32.CASES:
        blob = _params_c96(rng)
        if not np.frombuffer(blob[:4], np.uint32)[0] & 1:
            continue                                   # only the _nc form is ported
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        cases.append({"kernel": kernel, "co": "swin.co", "grid": -(-nwin // 8), "block": 256,
                      "args": [("buf", _fp8_map(rng, H * W * 96)), ("null",), ("out", H * W * 96), ("buf", blob),
                               ("u32", H), ("u32", W), ("u32", ox), ("u32", oy)]})
    return cases


# (th, tw) token grids of the projections; the pixel maps are [2 th][2 tw]. 3 x 5 and 2 x 9: partly filled waves.
TOKENS = [(8, 12), (18, 34), (4, 4), (34, 50), (3, 5), (2, 9)]


def _project_blob(rng, rows, frags):
    """Projection blob: bias tile vector (`rows` floats), then `frags` fragments of 256 FP8 bytes."""
    return rng.normal(0, 0.5, rows).astype(np.float32).tobytes() + fp8_encode(q8(rng.normal(0, 0.2, frags * 256))).tobytes()


def project_cases(kernel, rng):
    """l_project_gather: x [2 th][2 tw][32] -> features [th][tw][64] (blob enc0.project: 64 bias rows, 4 x 8 fragments).
    l_dec_project: z [th][tw][64] -> y [2 th][2 tw][32] (blob dec0.project: 128 bias rows, 8 x 4 fragments)."""
    cases = []
    for (th, tw) in TOKENS:
        ntok = th * tw
        npx = 4 * ntok
        if kernel == "l_project_gather":
            args = [("buf", _fp8_map(rng, npx * 32)), ("out", ntok * 64), ("buf", _project_blob(rng, 64, 32))]
        else:
            args = [("buf", _fp8_map(rng, ntok * 64)), ("out", npx * 32), ("buf", _project_blob(rng, 128, 32))]
        cases.append({"kernel": kernel, "grid": -(-ntok // 128), "block": 256, "args": args + [("u32", tw), ("u32", th)]})
    return cases


# (rw, rh, fw, fh) of the coefficient step: R inside the padded grid F (even). Odd rw / rh (a token's second column or
# row outside R), F wider than R by whole tokens, and 5 x 3 / 7 x 5 (partly filled waves).
COEF_FRAMES = [(24, 16, 24, 16), (67, 35, 68, 36), (7, 5, 8, 6), (100, 68, 100, 68), (5, 3, 6, 4), (13, 7, 20, 12)]


def _coef_map(rng, n):
    """FP16 decoder map [n][32]: mostly N(0, 3), with values beyond the clamp (+-20 on the two spread inputs), beyond
    tanh's saturation (9.01) and inside its polynomial range (|t| < 0.6), so every path of l_coef runs."""
    v = rng.normal(0, 3.0, (n, 32))
    pick = rng.random((n, 32))
    v = np.where(pick < 0.08, rng.uniform(20, 60, (n, 32)) * rng.choice([-1, 1], (n, 32)), v)
    v = np.where((pick >= 0.08) & (pick < 0.2), rng.uniform(-0.6, 0.6, (n, 32)), v)
    v = np.where((pick >= 0.2) & (pick < 0.25), rng.uniform(8, 10, (n, 32)) * rng.choice([-1, 1], (n, 32)), v)
    return v.astype(np.float16).tobytes()


def coef_cases(kernel, rng):
    """l_coef: y2 [fh][fw][32] FP16 -> A, B, latent [rh][rw] (4 bytes each), C [rh][rw] (8 bytes); blob dec0.coef
    (Model M's dec0 blob with W = 0, b = 0: S, T at byte 320); LDecParams (56 bytes) by value."""
    from clean.runner import dec0
    cases = []
    for (rw, rh, fw, fh) in COEF_FRAMES:
        npx = rw * rh
        S, T = float(rng.uniform(0.5, 4.0)), float(rng.uniform(0.3, 3.0))
        blob = dec0.pack({"W": np.zeros((68, 64), np.float32), "b": np.zeros(68, np.float32), "S": S, "T": T})
        P = bytes(dec0.DecParams(rw=rw, rh=rh, fw=fw, fh=fh, reset=0, pad0=0, dw=2 * rw, dh=2 * rh, kx=0.5, ky=0.5,
                                 jx=0.25, jy=-0.25, pjx=0.0, pjy=0.0))
        assert len(P) == 56
        ntok = (fh // 2) * (fw // 2)
        cases.append({"kernel": kernel, "grid": -(-ntok // 128), "block": 256,
                      "args": [("buf", _coef_map(rng, fw * fh)), ("out", npx * 4), ("out", npx * 4), ("out", npx * 8),
                               ("out", npx * 4), ("buf", blob), ("bytes", P)]})
    return cases


# (rw, rh, reset) of the halo encoder (clean/l/runner/l_enc_check.py's frames): padding to multiples of 32, odd sizes,
# both reset states, and width 17 (padded 32), where halo columns 32 and 33 both mirror onto render column 0.
ENC_FRAMES = [(40, 24, False), (37, 19, False), (64, 32, True), (90, 50, False), (17, 17, False), (17, 24, True)]


def enc_cases(kernel, rng):
    """lev/input_encoder (l_enc_values.co, HIP name input_encoder): a synthetic frame (clean/reference/encoder.py
    random_inputs) -> the halo map q [fh + 4][fw + 4][32] and the motion surfaces m_near, m_far, m_center [rh][rw]
    (FP16 x 2); blob enc0.values (l_pack.pack_values); WgEncParams (116 bytes) by value."""
    from clean.l.reference import oracle as L
    from clean.l.runner import l_pack as LP
    from clean.reference import encoder as ME
    from clean.runner import enc0 as E0
    enc = L.random_params(rng)["enc0"]
    cases = []
    for (rw, rh, reset) in ENC_FRAMES:
        x = ME.random_inputs(rng, rw=rw, rh=rh, reset=reset)
        prm = E0.frame_params(x)
        data = E0.device_inputs(x)
        mw, mh = prm.fw + 4, prm.fh + 4
        ntok = (mw // 2) * (mh // 2)
        npx = prm.rw * prm.rh * 4
        frame = [("buf", v if isinstance(v, bytes) else np.ascontiguousarray(v).tobytes())
                 for v in (data[k] for k in ("color", "depth", "motion", "hist", "latent", "mhist"))]
        expo = np.array([x["exposure"], x["exposure_prev"]], dtype=np.float32).tobytes()
        cases.append({"kernel": kernel, "hip_kernel": "input_encoder", "co": "l_enc_values.co", "tag": "lev",
                      "grid": -(-ntok // 128), "block": 256,
                      "args": frame + [("out", mw * mh * 32), ("out", npx), ("out", npx), ("out", npx), ("buf", expo),
                                       ("buf", LP.pack_values(enc)), ("bytes", bytes(prm))]})
    return cases


# (rw, rh, dw, dh, reset, kind, expo) of the reconstruction: clean/l/runner/l_post_check.py's cases (Model M's ratios,
# odd sizes, reset, extreme kernel shapes, the zero-weight known answer "offset_0.1" (spec part 3 Q1: total weight 0
# everywhere) and the FP16 underflow known answer "offset_0.05" (Q3: total weight > 0, every FP16 weight 0)), plus
# R = D (footprint over the LDS cap: the unstaged path), ratio 1.5, the exposures read from the expo pointer, and
# non-finite / out-of-range inputs ("specials": NaN and infinite colours and history texels, motions far outside D).
POST_CASES = [(24, 16, 54, 36, False, "random", False), (31, 17, 70, 38, False, "random", False),
              (20, 12, 60, 36, False, "random", False), (33, 19, 99, 57, True, "random", False),
              (17, 11, 38, 25, False, "extreme_c", False), (16, 12, 16, 12, False, "offset_0.1", False),
              (16, 12, 16, 12, True, "offset_0.05", False), (40, 24, 40, 24, False, "random", False),
              (42, 26, 63, 39, False, "random", True), (26, 18, 52, 36, False, "specials", False)]
POST_SLACK = 64   # sentinel bytes after each output: neither back end may write them


def _post_specials(rng, x):
    c = np.array(x["color"], np.float32)
    pick = rng.random(c.shape[:2])
    c[pick < 0.04] = np.nan
    c[(pick >= 0.04) & (pick < 0.08)] = np.inf
    c[(pick >= 0.08) & (pick < 0.12)] = 60000.0
    x["color"] = c
    h = np.array(x["history"], np.float32)
    pick = rng.random(h.shape[:2])
    h[pick < 0.03] = np.nan
    h[(pick >= 0.03) & (pick < 0.06)] = np.inf
    h[(pick >= 0.06) & (pick < 0.09)] = -1.0
    x["history"] = h.astype(np.float16)
    for k in ("m_near", "m_far"):
        m = np.array(x[k], np.float32)
        far = rng.random(m.shape[:2]) < 0.2
        m[far] = rng.choice([-1, 1], (int(far.sum()), 2)) * 200.0
        x[k] = m.astype(np.float16)
    return x


def post_cases(kernel, rng):
    """l_post_recon (l_post.co): color, A, B, C, hist_prev, m_near, m_center, m_far, w (post blob, w[0] = s) ->
    full history [dh][dw] (RGBA16F), reduced codes [dh / 2][dw / 2], display (3 FP16 planes of pitch
    display_pitch(dw)); WgPostParams (64 bytes) by value; expo (null, or the two exposures). Bytes no back end writes
    (the display rows' pitch padding, the slack after each buffer) must keep each back end's own sentinel."""
    from clean.l.runner import l_post_check as LPC
    from clean.runner import post as RP
    cases = []
    for (rw, rh, dw, dh, reset, kind, expo) in POST_CASES:
        x = LPC.make_case(rng, rw, rh, dw, dh, reset, "random" if kind == "specials" else kind)
        if kind == "specials":
            x = _post_specials(rng, x)
        prm = RP.post_params(x)
        data = RP.post_device_inputs(x)
        if expo:   # the exposures come from the pointer; the by-value fields hold other values
            ev = np.array([prm.e, 0.0 if rng.random() < 0.5 else prm.e_prev], np.float32)
            prm.e, prm.e_prev = float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.5, 2.0))
        n_hist, n_red = dw * dh * 8, max(4, (dw // 2) * (dh // 2) * 4)
        n_disp = RP.display_bytes(dw, dh)
        pitch = RP.display_pitch(dw)
        disp_def = np.zeros((3, dh, pitch, 2), bool)
        disp_def[:, :, :dw, :] = True
        red_def = np.zeros(n_red, bool)
        red_def[:(dw // 2) * (dh // 2) * 4] = True
        defined = [np.concatenate([np.ones(n_hist, bool), np.zeros(POST_SLACK, bool)]),
                   np.concatenate([red_def, np.zeros(POST_SLACK, bool)]),
                   np.concatenate([disp_def.reshape(-1), np.zeros(POST_SLACK, bool)])]
        args = [("buf", data[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far")]
        args += [("buf", RP.pack({"s": np.array(LPC.S)})), ("out", n_hist + POST_SLACK), ("out", n_red + POST_SLACK),
                 ("out", n_disp + POST_SLACK), ("bytes", bytes(prm))]
        args += [("buf", ev.tobytes())] if expo else [("null",)]
        assert len(bytes(prm)) == 64
        cases.append({"kernel": kernel, "co": "l_post.co", "grid": RP.recon_grid(dw, dh), "block": 256, "args": args,
                      "defined": defined, "label": f"{rw}x{rh}->{dw}x{dh} {kind}" + (" expo" if expo else "")
                      + (" reset" if reset else "")})
    return cases


KINDS = {"l_linear_32_32_relu": linear_cases, "l_linear_32_32_f16": linear_cases,
         "l_project_gather": project_cases, "l_dec_project": project_cases, "l_coef": coef_cases,
         "lev/input_encoder": enc_cases, "l_post_recon": post_cases}
CONTROL_KINDS = {"swin_block_c96_nc": control_cases}
CO = "l_ops.co"


def prepare(build, rocm, kernels):
    C32.compile_kernels(build, rocm, with_swin=True)   # swin.co (the control); l_ops.co: with #115's compile_kernels
    return prepare_replay(build, rocm, kernels)


def hip_run(g, mod, case):
    """One case on HIP: the outputs' downloaded bytes, as they are. Outputs start filled with HIP_SENTINEL."""
    h = g.hip
    args, owned, outs = [], [], []
    for a in case["args"]:
        if a[0] == "buf":
            p = h.malloc(max(len(a[1]), 4))
            h.upload(p, a[1])
            owned.append(p)
            args.append(ctypes.c_void_p(p))
        elif a[0] == "out":
            p = h.malloc(a[1])
            h.memset(p, HIP_SENTINEL, a[1])
            owned.append(p)
            outs.append((p, a[1]))
            args.append(ctypes.c_void_p(p))
        elif a[0] == "null":
            args.append(ctypes.c_void_p(0))
        elif a[0] == "u32":
            args.append(ctypes.c_int(a[1]))
        elif a[0] == "bytes":
            args.append((ctypes.c_ubyte * len(a[1])).from_buffer_copy(a[1]))
    try:
        h.launch(h.get_function(mod, case.get("hip_kernel", case["kernel"])), case["grid"], case["block"], args)
        h.sync()
        return [bytes(h.download(p, n)) for p, n in outs]
    finally:
        for p in owned:
            h.free(p)


def replay_lines(name, case, cdir, grid=None, prefix=""):
    lines = [f"case {prefix}{name}", f"module {case.get('co', CO)}" + (f" {case['tag']}" if case.get("tag") else ""),
             f"kernel {case['kernel']} {grid or case['grid']} {case['block']}"]
    oi = 0
    for i, a in enumerate(case["args"]):
        if a[0] in ("buf", "bytes"):
            f = f"{name}.a{i}"
            if not prefix:
                with open(os.path.join(cdir, f), "wb") as fh:
                    fh.write(a[1])
            lines.append(f"arg {a[0]} {f}")
        elif a[0] == "out":
            lines.append(f"arg out {a[1]} {prefix}{name}.o{oi} {D3D12_SENTINEL}")
            oi += 1
        elif a[0] == "null":
            lines.append("arg null")
        elif a[0] == "u32":
            lines.append(f"arg u32 {a[1]}")
    return lines + [""]


def run(build, rocm, seeds, kernels):
    g = C32.Gpu(build, rocm)
    mods = {}

    def module(co):
        if co not in mods:
            with open(os.path.join(build, co), "rb") as f:
                mods[co] = g.hip.load_module(f.read())
        return mods[co]

    kinds = dict(KINDS, **CONTROL_KINDS)
    cdir = os.path.join(build, "replay-ops")
    os.makedirs(cdir, exist_ok=True)
    lines, hip_out, neg = [], {}, None
    for kernel in kernels:
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            for ci, case in enumerate(kinds[kernel](kernel, rng)):
                name = f"{kernel.replace('/', '_')}_s{si}_{ci}"   # lev/input_encoder: no folder in the file name
                hip_out[name] = (kernel, hip_run(g, module(case.get("co", CO)), case), case.get("defined"),
                                 case.get("label"))
                for oi, data in enumerate(hip_out[name][1]):   # kept next to the D3D12 outputs, for a look at a difference
                    with open(os.path.join(cdir, f"{name}.hip.o{oi}"), "wb") as fh:
                        fh.write(data)
                lines += replay_lines(name, case, cdir)
                if neg is None and case["grid"] >= 4:
                    neg = name
                    lines += replay_lines(name, case, cdir, grid=case["grid"] // 2, prefix="negative_")
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=150, creationflags=0x08000000)
    res = {"replay_exit": r.returncode, "replay_tail": r.stdout[-600:] + r.stderr[-300:], "cases": []}

    def read(path):
        return open(path, "rb").read() if os.path.exists(path) else b""

    def compare(got, want, dmask):
        """Raw bytes of one output: identical where a back end writes (dmask; None = every byte); where neither may
        write, each keeps its own sentinel."""
        a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
        o = {"bytes": len(want), "hip_sentinel_bytes": int(np.sum(b == HIP_SENTINEL)),
             "d3d12_sentinel_bytes": int(np.sum(a == D3D12_SENTINEL)) if a.size else -1}
        if a.size != b.size:
            return dict(o, identical=False, differ=-1)
        m = np.ones(b.size, bool) if dmask is None else dmask
        o["defined_bytes"] = int(m.sum())
        o["differ"] = int(np.sum((a != b) & m))
        o["unwritten_bad"] = int(np.sum(~m & ((b != HIP_SENTINEL) | (a != D3D12_SENTINEL))))
        o["identical"] = o["differ"] == 0 and o["unwritten_bad"] == 0
        return o

    for name, (kernel, wants, defined, label) in hip_out.items():
        c = {"case": name, "kernel": kernel, "outputs": []}
        if label:
            c["label"] = label
        for oi, want in enumerate(wants):
            got = read(os.path.join(cdir, f"{name}.o{oi}"))
            c["outputs"].append(compare(got, want, defined[oi] if defined else None))
        c["identical"] = all(o["identical"] for o in c["outputs"])
        res["cases"].append(c)
    caught = None
    if neg:
        wants, defined = hip_out[neg][1], hip_out[neg][2]
        outs = [compare(read(os.path.join(cdir, f"negative_{neg}.o{oi}")), w, defined[oi] if defined else None)
                for oi, w in enumerate(wants)]
        caught = all(o["differ"] >= 0 for o in outs) and any(not o["identical"] for o in outs)
        res["negative_control"] = {"case": neg, "half_grid": True, "caught": caught,
                                   "differ": sum(max(o["differ"], 0) for o in outs)}
    res["per_kernel"] = {k: {"cases": sum(c["kernel"] == k for c in res["cases"]),
                             "identical": sum(c["identical"] for c in res["cases"] if c["kernel"] == k),
                             "bytes": sum(o["bytes"] for c in res["cases"] if c["kernel"] == k for o in c["outputs"]),
                             "differ": sum(max(o["differ"], 0) for c in res["cases"] if c["kernel"] == k
                                           for o in c["outputs"])}
                         for k in kernels}
    res["pass"] = r.returncode == 0 and caught is True and all(v["cases"] > 0 and v["identical"] == v["cases"]
                                                               for v in res["per_kernel"].values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kernels", default=",".join(KINDS))
    ap.add_argument("--no-control", action="store_true", help="skip Model M's swin_block_c96_nc control")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    kernels = [k for k in a.kernels.split(",") if k] + ([] if a.no_control else list(CONTROL_KINDS))
    os.makedirs(a.build, exist_ok=True)
    if a.prepare:
        print("built", prepare(a.build, a.rocm, kernels))
    if a.run:
        res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)), kernels)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        print({k: v for k, v in res.items() if k not in ("cases", "replay_tail")})
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
