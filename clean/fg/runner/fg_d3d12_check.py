# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): net1's GPU kernels (clean/fg/kernels/fg_net1.hip) against their D3D12 ports
(clean/dx12/kernels/fg_*.hlsl) byte for byte, and against the CPU reference (clean/fg/reference/net1.py) bit for bit.

    python clean/fg/runner/fg_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] [--kernels fg_conv_f16] [--json out.json]

--prepare (CPU) builds into <dir>: fg_net1.co (ROCm clang), the D3D12 ports (clean/dx12/build_dx12.py --only; the
pinned DXC through WG_TOOLCHAINS, BUILDING.md) and kernel_replay.exe (clean/dx12/tests/kernel_replay.cpp).
--run (GPU, inside one GPU slot) runs every case on HIP in this process and the same inputs through kernel_replay on
the native D3D12 device in a child process, then:
- compares the RAW downloaded bytes of both back ends; each output buffer starts filled with its back end's own
  sentinel (HIP 0xA5, D3D12 0x5A) and has 64 slack bytes after it that neither may write, so an unwritten byte or a
  write past the end differs;
- a negative control per kernel (its first case with 4 or more workgroups, again on D3D12 with half of them) must be
  caught;
- compares the HIP output with the CPU reference, value for value (float16 bits; NaN equals NaN, the reference's NaN
  payload is not specified).
Cases: fg_conv_f16: every net1 layer shape (cin, cout, kernel size, chunk order, activation) on small grids of several
sizes, and special values (NaN, +-inf, subnormals, +-65504, -0) in the inputs and weights; fg_maxpool2_f16: random maps
of several sizes (odd ones too) and every 2 x 2 window over ten special values; fg_upscale_f16: net1's level pairs,
other sizes, an output extent of 1, equal sizes, special values; fg_add_f16: random pairs of several lengths and every
pair of thirteen special values. Output: counts only, no timing. The whole network: fg_net1_gpu.py.
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

from clean.fg.reference import net1 as N  # noqa: E402

HIP_SENTINEL, D3D12_SENTINEL = 0xA5, 0x5A
SLACK = 64
CO = "fg_net1.co"
ORDER_CODE = {"ascending": 0, "block_kx_ky": 1, "kx_block_ky": 2}
GRIDS = [(5, 7), (1, 1), (8, 8), (3, 17), (16, 12)]


def _layers():
    """(layer, cin, cout, k, order, act) of net1's 17 convolutions, from the wiring and the layout."""
    from clean.fg import layout as L
    by = {l.name: l for l in L.NET1}
    out = []
    for s in N.NET1_WIRING:
        if s["op"] == "conv":
            l = by[s["layer"]]
            out.append((l.name, l.cin, l.out, 3 if l.taps == 9 else 1, s["chunks"], s["act"]))
    return out


def _f16(a):
    return np.asarray(a, dtype=np.float16)


SPECIALS = np.array([np.nan, np.inf, -np.inf, 65504.0, -65504.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -0.0, 0.0],
                    dtype=np.float16)


def conv_cases(kernel, rng, specials=False):
    cases = []
    for li, (name, cin, cout, k, order, act) in enumerate(_layers()):
        for gi, (H, W) in enumerate(GRIDS):
            if cin * cout > 32768 and H * W > 64:      # the widest layers on the smaller grids only
                continue
            x = _f16(rng.uniform(-2, 2, (cin, H, W)))
            w = _f16(rng.uniform(-0.2, 0.2, (cout, cin, k, k)))
            b = _f16(rng.uniform(-0.5, 0.5, cout))
            cases.append(_conv_case(kernel, x, w, b, k, order, act, f"{name} {H}x{W}"))
    if specials:
        for k, order in ((3, "block_kx_ky"), (3, "kx_block_ky"), (1, "ascending")):
            for act in ("leaky", "none"):
                cin, cout, H, W = 32, 8, 6, 9
                x = _f16(rng.uniform(-2, 2, (cin, H, W)))
                w = _f16(rng.uniform(-1, 1, (cout, cin, k, k)))
                b = _f16(rng.uniform(-1, 1, cout))
                fx = rng.random(x.shape) < 0.04
                x[fx] = rng.choice(SPECIALS, int(fx.sum()))
                fw = rng.random(w.shape) < 0.02
                w[fw] = rng.choice(SPECIALS, int(fw.sum()))
                b[0], b[1], b[2] = np.inf, -np.inf, np.nan
                big = _f16(rng.uniform(100, 255, (cout, cin, k, k)))   # sums that overflow to +-inf
                w[5] = big[5]
                w[6] = -big[6]
                cases.append(_conv_case(kernel, x, w, b, k, order, act, f"specials k{k} {order} {act}"))
    return cases


def _conv_case(kernel, x, w, b, k, order, act, label):
    cin, H, W = x.shape
    cout = w.shape[0]
    n = cout * H * W
    return {"kernel": kernel, "grid": -(-n // 256), "block": 256, "label": label,
            "args": [("buf", x.tobytes()), ("buf", w.tobytes()), ("buf", b.tobytes()), ("out", 2 * n + SLACK),
                     ("u32", cin), ("u32", cout), ("u32", H), ("u32", W), ("u32", k), ("u32", ORDER_CODE[order]),
                     ("u32", 1 if act == "leaky" else 0)],
            "defined": 2 * n,
            "cpu": lambda: N.conv16(x.astype(np.float64), w.astype(np.float64).transpose(0, 2, 3, 1),
                                    b.astype(np.float64), chunks=order, act=act)}


POOL_SPECIALS = np.array([1.0, -2.0, np.inf, -np.inf, np.nan, 0.0, -0.0, 65504.0, -65504.0, 2.0 ** -24],
                         dtype=np.float16)


def pool_cases(kernel, rng):
    """Random maps (odd sizes drop their last row / column), and every window of four taps over POOL_SPECIALS (10^4
    windows side by side in one 2-row map)."""
    cases = []
    for C, H, W in ((3, 7, 9), (32, 8, 8), (64, 5, 5), (1, 2, 2), (16, 3, 17), (128, 4, 6)):
        x = _f16(rng.uniform(-4, 4, (C, H, W)))
        cases.append(_pool_case(kernel, x, f"{C}x{H}x{W}"))
    v = POOL_SPECIALS
    n = len(v)
    idx = np.arange(n ** 4)
    taps = [v[(idx // n ** p) % n] for p in range(4)]          # (0,0), (0,1), (1,0), (1,1) of each window
    x = np.zeros((1, 2, 2 * n ** 4), np.float16)
    x[0, 0, 0::2], x[0, 0, 1::2], x[0, 1, 0::2], x[0, 1, 1::2] = taps
    cases.append(_pool_case(kernel, x, "every special window"))
    return cases


def _pool_case(kernel, x, label):
    C, H, W = x.shape
    n = C * (H // 2) * (W // 2)
    return {"kernel": kernel, "grid": -(-n // 256), "block": 256, "label": label,
            "args": [("buf", x.tobytes()), ("out", 2 * n + SLACK), ("u32", C), ("u32", H), ("u32", W)],
            "defined": 2 * n, "cpu": lambda: N.maxpool2(x.astype(np.float64))}


def _ratio_bits(n_in, n_out):
    """The align-corners ratio (n_in - 1) / (n_out - 1) as single-precision bits (correctly rounded; 0 for n_out 1), as
    net1.upscale_to computes it."""
    r = np.float32(n_in - 1) / np.float32(n_out - 1) if n_out > 1 else np.float32(0)
    return int(np.array([r], np.float32).view(np.uint32)[0])


def upscale_cases(kernel, rng):
    """net1's level pairs (the coarse level is the fine one halved with floor), other odd sizes, an output extent of 1,
    equal sizes, and special values (NaN, +-inf, subnormals, +-65504, -0) in the input."""
    cases = []
    pairs = [((17, 33), (35, 67)), ((1, 2), (2, 4)), ((4, 8), (8, 16)), ((8, 16), (17, 33)), ((3, 4), (7, 9)),
             ((1, 1), (2, 3)), ((5, 2), (11, 4)), ((2, 3), (2, 3)), ((1, 3), (1, 7)), ((1, 1), (1, 1))]
    for (hi, wi), (ho, wo) in pairs:
        for C in (1, 32):
            x = _f16(rng.uniform(-3, 3, (C, hi, wi)))
            cases.append(_upscale_case(kernel, x, ho, wo, f"{C}x{hi}x{wi} -> {ho}x{wo}"))
    x = _f16(rng.uniform(-3, 3, (16, 9, 13)))
    f = rng.random(x.shape) < 0.08
    x[f] = rng.choice(SPECIALS, int(f.sum()))
    cases.append(_upscale_case(kernel, x, 19, 27, "specials 9x13 -> 19x27"))
    return cases


def _upscale_case(kernel, x, ho, wo, label):
    C, hi, wi = x.shape
    n = C * ho * wo
    return {"kernel": kernel, "grid": -(-n // 256), "block": 256, "label": label,
            "args": [("buf", x.tobytes()), ("out", 2 * n + SLACK), ("u32", C), ("u32", hi), ("u32", wi), ("u32", ho),
                     ("u32", wo), ("u32", _ratio_bits(wi, wo)), ("u32", _ratio_bits(hi, ho))],
            "defined": 2 * n, "cpu": lambda: N.upscale_to(x.astype(np.float64), wo, ho)}


ADD_SPECIALS = np.array([np.nan, np.inf, -np.inf, 65504.0, -65504.0, 2.0 ** -24, -2.0 ** -24, -0.0, 0.0, 1.0, -1.0,
                         32768.0, 0.000999], dtype=np.float16)


def add_cases(kernel, rng):
    """Random pairs of several lengths, and every pair over ADD_SPECIALS (overflow, inf - inf, -0 + -0, subnormal
    sums)."""
    cases = []
    for n in (1, 255, 256, 1000, 4096 * 3 + 7):
        a = _f16(rng.uniform(-100, 100, n))
        b = _f16(rng.uniform(-100, 100, n))
        cases.append(_add_case(kernel, a, b, f"n {n}"))
    v = ADD_SPECIALS
    a = np.repeat(v, len(v)).astype(np.float16)
    b = np.tile(v, len(v)).astype(np.float16)
    cases.append(_add_case(kernel, a, b, "every special pair"))
    return cases


def _add_case(kernel, a, b, label):
    n = a.size
    return {"kernel": kernel, "grid": -(-n // 256), "block": 256, "label": label,
            "args": [("buf", a.tobytes()), ("buf", b.tobytes()), ("out", 2 * n + SLACK), ("u32", n)],
            "defined": 2 * n, "cpu": lambda: N.add16(a.astype(np.float64), b.astype(np.float64))}


KINDS = {"fg_conv_f16": lambda kernel, rng: conv_cases(kernel, rng, specials=True),
         "fg_maxpool2_f16": pool_cases, "fg_upscale_f16": upscale_cases, "fg_add_f16": add_cases}


# ---- prepare --------------------------------------------------------------------------------------------------------

def prepare(build, rocm, kernels):
    from clean.l.runner.l_d3d12_check import prepare_replay
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    os.makedirs(build, exist_ok=True)
    clean_process_environment(rocm)
    clang = B.find_clang(rocm)
    B.run(B.device_command(clang, rocm, os.path.join(ROOT, "clean", "fg", "kernels", "fg_net1.hip"),
                           os.path.join(build, CO)) + ["-I", os.path.join(ROOT, "clean", "kernels")])
    return prepare_replay(build, rocm, kernels)


# ---- run ------------------------------------------------------------------------------------------------------------

def hip_run(h, mod, case):
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
        elif a[0] == "u32":
            args.append(ctypes.c_int(a[1]))
    try:
        h.launch(h.get_function(mod, case["kernel"]), case["grid"], case["block"], args)
        h.sync()
        return [bytes(h.download(p, n)) for p, n in outs]
    finally:
        for p in owned:
            h.free(p)


def replay_lines(name, case, cdir, grid=None, prefix=""):
    lines = [f"case {prefix}{name}", f"module {CO}", f"kernel {case['kernel']} {grid or case['grid']} {case['block']}"]
    oi = 0
    for i, a in enumerate(case["args"]):
        if a[0] == "buf":
            f = f"{name}.a{i}"
            if not prefix:
                with open(os.path.join(cdir, f), "wb") as fh:
                    fh.write(a[1])
            lines.append(f"arg buf {f}")
        elif a[0] == "out":
            lines.append(f"arg out {a[1]} {prefix}{name}.o{oi} {D3D12_SENTINEL}")
            oi += 1
        elif a[0] == "u32":
            lines.append(f"arg u32 {a[1]}")
    return lines + [""]


def compare_raw(got, want, defined):
    """Raw bytes: identical everywhere; in the slack after `defined` bytes each back end keeps its own sentinel."""
    a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
    if a.size != b.size:
        return {"bytes": int(b.size), "differ": -1, "slack_ok": False}
    return {"bytes": int(b.size), "differ": int(np.sum(a[:defined] != b[:defined])),
            "slack_ok": bool(np.all(a[defined:] == D3D12_SENTINEL) and np.all(b[defined:] == HIP_SENTINEL))}


def compare_cpu(hip_bytes, ref):
    """HIP float16 output against the CPU reference: bits equal, or both NaN."""
    g = np.frombuffer(hip_bytes, np.uint16)
    r = np.asarray(ref, np.float64).astype(np.float16).reshape(-1).view(np.uint16)
    gn = ((g & 0x7C00) == 0x7C00) & ((g & 0x3FF) != 0)
    rn = ((r & 0x7C00) == 0x7C00) & ((r & 0x3FF) != 0)
    same = (g == r) | (gn & rn)
    return {"values": int(g.size), "differ": int(np.sum(~same)), "nan": int(np.sum(gn)),
            "inf": int(np.sum((g & 0x7FFF) == 0x7C00))}


def run(build, rocm, seeds, kernels):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    with open(os.path.join(build, CO), "rb") as f:
        mod = h.load_module(f.read())
    cdir = os.path.join(build, "replay-fg")
    os.makedirs(cdir, exist_ok=True)
    lines, hip_out, negs = [], {}, {}     # negs: per kernel, its first case with 4 or more workgroups
    for kernel in kernels:
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            for ci, case in enumerate(KINDS[kernel](kernel, rng)):
                name = f"{kernel}_s{si}_{ci}"
                outs = hip_run(h, mod, case)
                hip_out[name] = (kernel, case, outs)
                lines += replay_lines(name, case, cdir)
                if kernel not in negs and case["grid"] >= 4:
                    negs[kernel] = name
                    lines += replay_lines(name, case, cdir, grid=case["grid"] // 2, prefix="negative_")
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=150, creationflags=0x08000000)
    res = {"replay_exit": r.returncode, "replay_tail": r.stdout[-400:] + r.stderr[-300:], "cases": []}

    def read(path):
        return open(path, "rb").read() if os.path.exists(path) else b""

    ok = r.returncode == 0
    for name, (kernel, case, outs) in hip_out.items():
        raw = compare_raw(read(os.path.join(cdir, f"{name}.o0")), outs[0], case["defined"])
        cpu = compare_cpu(outs[0][:case["defined"]], case["cpu"]())
        # a byte either back end leaves unwritten differs (0xA5 against 0x5A), so differ == 0 also proves every byte
        # was written by both; equal bytes that happen to be 0xA5 are ordinary data
        good = raw["differ"] == 0 and raw["slack_ok"] and cpu["differ"] == 0
        ok = ok and good
        res["cases"].append({"case": name, "kernel": kernel, "label": case["label"], "raw": raw, "cpu": cpu,
                             "ok": good})
    res["negative_controls"] = {}
    for kernel in kernels:                # every kernel needs its own caught negative control
        neg = negs.get(kernel)
        if neg is None:
            res["negative_controls"][kernel] = None
            ok = False
            continue
        _, case, outs = hip_out[neg]
        nraw = compare_raw(read(os.path.join(cdir, f"negative_{neg}.o0")), outs[0], case["defined"])
        caught = nraw["differ"] != 0 or not nraw["slack_ok"]
        res["negative_controls"][kernel] = {"case": neg, "half_grid": True, "caught": caught, "differ": nraw["differ"]}
        ok = ok and caught
    res["summary"] = {"cases": len(res["cases"]), "cases_ok": sum(c["ok"] for c in res["cases"]),
                      "values": sum(c["cpu"]["values"] for c in res["cases"]),
                      "nan_values": sum(c["cpu"]["nan"] for c in res["cases"]),
                      "inf_values": sum(c["cpu"]["inf"] for c in res["cases"]), "pass": bool(ok)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--kernels", default=",".join(KINDS))
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    kernels = a.kernels.split(",")
    if a.prepare:
        prepare(a.build, a.rocm, kernels)
    if a.run:
        res = run(a.build, a.rocm, list(range(1, a.seeds + 1)), kernels)
        text = json.dumps(res, indent=1)
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
        print(json.dumps(res["summary"]), "negative controls:", json.dumps(res["negative_controls"]))
        for c in res["cases"]:
            if not c["ok"]:
                print("FAIL", c["case"], c["label"], json.dumps(c["raw"]), json.dumps(c["cpu"]))
        print("replay exit", res["replay_exit"])
        return 0 if res["summary"]["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
