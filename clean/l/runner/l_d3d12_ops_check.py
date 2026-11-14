# SPDX-License-Identifier: MIT
"""Model L (issue #89): the D3D12 ports of the l_ops.co kernels against their HIP kernels on the same synthetic inputs,
byte for byte (clean/l/kernels/l_ops.hip; ports clean/dx12/kernels/l_ops_*.hlsl).

    python clean/l/runner/l_d3d12_ops_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_d3d12_ops_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] [--kernels a,b] --json out.json

--prepare (CPU) builds into <dir>: l_ops.co (clean/l/runner/l_c32_check.py compile_kernels), the D3D12 ports
(clean/dx12/build_dx12.py --only, with WG_TOOLCHAINS / WG_DXC_DIR as BUILDING.md describes) and kernel_replay.exe.
--run (GPU, one GPU slot) runs every case on HIP in this process and the same inputs through kernel_replay on the native
D3D12 device in a child process, then compares every output buffer. The rules (as clean/l/runner/l_d3d12_check.py):
- the RAW downloaded bytes of both back ends are compared (nothing is decoded and encoded again);
- each output buffer starts filled with its back end's own non-zero sentinel (HIP 0xA5, D3D12 0x5A), so a byte that
  either path, or both, leaves unwritten differs;
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


KINDS = {"l_linear_32_32_relu": linear_cases, "l_linear_32_32_f16": linear_cases,
         "l_project_gather": project_cases, "l_dec_project": project_cases}
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
        h.launch(h.get_function(mod, case["kernel"]), case["grid"], case["block"], args)
        h.sync()
        return [bytes(h.download(p, n)) for p, n in outs]
    finally:
        for p in owned:
            h.free(p)


def replay_lines(name, case, cdir, grid=None, prefix=""):
    lines = [f"case {prefix}{name}", f"module {case.get('co', CO)}",
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
                name = f"{kernel}_s{si}_{ci}"
                hip_out[name] = (kernel, hip_run(g, module(case.get("co", CO)), case))
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

    for name, (kernel, wants) in hip_out.items():
        c = {"case": name, "kernel": kernel, "outputs": []}
        for oi, want in enumerate(wants):
            got = read(os.path.join(cdir, f"{name}.o{oi}"))
            a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
            c["outputs"].append({"bytes": len(want), "identical": got == want,
                                 "differ": int(np.sum(a != b)) if a.size == b.size else -1,
                                 "hip_sentinel_bytes": int(np.sum(b == HIP_SENTINEL)),
                                 "d3d12_sentinel_bytes": int(np.sum(a == D3D12_SENTINEL)) if a.size else -1})
        c["identical"] = all(o["identical"] for o in c["outputs"])
        res["cases"].append(c)
    caught = None
    if neg:
        wants = hip_out[neg][1]
        gots = [read(os.path.join(cdir, f"negative_{neg}.o{oi}")) for oi in range(len(wants))]
        caught = all(len(gt) == len(w) for gt, w in zip(gots, wants)) and any(gt != w for gt, w in zip(gots, wants))
        res["negative_control"] = {"case": neg, "half_grid": True, "caught": caught,
                                   "differ": sum(int(np.sum(np.frombuffer(gt, np.uint8) != np.frombuffer(w, np.uint8)))
                                                 for gt, w in zip(gots, wants) if len(gt) == len(w))}
    res["per_kernel"] = {k: {"cases": sum(c["kernel"] == k for c in res["cases"]),
                             "identical": sum(c["identical"] for c in res["cases"] if c["kernel"] == k),
                             "bytes": sum(o["bytes"] for c in res["cases"] if c["kernel"] == k for o in c["outputs"])}
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
