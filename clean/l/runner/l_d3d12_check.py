# SPDX-License-Identifier: MIT
"""Model L (issue #89): each D3D12 port against its HIP kernel on the same synthetic inputs, byte for byte.

    python clean/l/runner/l_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_d3d12_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--seeds N] --json out.json

--prepare (CPU) builds into <dir>: the HIP code objects (l_swin.co), the D3D12 ports (clean/dx12/build_dx12.py
--only <the Model L kernels>, needs WG_TOOLCHAINS / WG_DXC_DIR as BUILDING.md describes) and
clean/dx12/tests/kernel_replay.exe (host clang++ of the HIP SDK).
--run (GPU, one GPU slot) runs every case on HIP in this process, writes the same inputs for kernel_replay, runs it on
the native D3D12 device in a child process, and compares the outputs: the RAW downloaded bytes of both (nothing is
decoded and encoded again), each output buffer filled beforehand with its own non-zero sentinel (HIP 0xA5, D3D12
0x5A), so a byte either path leaves unwritten shows up as a difference. Output: counts only, no timing.

Kernels: l_swin_block_c32_nc (the D3D12 port has only the `_nc` form, as Model M's ports).
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

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402
from clean.runner import packing as P  # noqa: E402

D3D12_KERNELS = ["l_swin_block_c32_nc", "native_fill"]   # native_fill: the device's memset
# Control: Model M's swin_block_c96_nc, whose port is already proven byte for byte (clean/dx12/compare.py on whole
# frames): if it matches here too, the harness is sound.
CONTROL_KERNELS = ["swin_block_c96_nc"]
CASES = C32.CASES + [(96, 160, (2, 2)), (240, 432, (0, 2))]


def prepare(build, rocm):
    C32.compile_kernels(build, rocm, with_swin=True)
    return prepare_replay(build, rocm, D3D12_KERNELS + CONTROL_KERNELS)


def prepare_replay(build, rocm, kernels):
    """The D3D12 ports `kernels` (and native_fill, the device's memset) into <build>/dx12, and kernel_replay.exe."""
    from clean.runner.hip import hip_child_environment
    env, _ = hip_child_environment(rocm_dir=rocm)
    names = list(kernels) + ([] if "native_fill" in kernels else ["native_fill"])
    subprocess.check_call([sys.executable, os.path.join(ROOT, "clean", "dx12", "build_dx12.py"), "--out", build,
                           "--only", ",".join(names), "-j", "1"], env=env, cwd=ROOT)
    rt = os.path.join(ROOT, "clean", "runtime")
    conv = os.path.join(ROOT, "clean", "converter", "src")
    exe = os.path.join(build, "kernel_replay.exe")
    cmd = [os.path.join(rocm, "bin", "clang++.exe"), "-std=c++17", "-O2", "-D_CRT_SECURE_NO_WARNINGS",
           "-fms-runtime-lib=static", os.path.join(ROOT, "clean", "dx12", "tests", "kernel_replay.cpp"),
           os.path.join(rt, "device.cpp"), os.path.join(rt, "device_d3d12.cpp"), os.path.join(rt, "embedded.cpp"),
           os.path.join(conv, "json.cpp"), os.path.join(conv, "sha256.cpp"), "-o", exe]
    subprocess.check_call(cmd, env=env, cwd=build)
    return exe


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def _params_c32(rng):
    p = L.random_block_params(rng, 32, 2, 128)
    blob = P.pack_block(p, 32, 2, 128, P.resolve_d_row("gfx12"))
    return blob + bytes((-len(blob)) % P.ALIGN)


def _params_c96(rng):
    from clean.reference import swin as MS
    return P.pack_stage("enc2", MS.random_params("enc2", rng))[:P.stage_sections("enc2")[1]]


KINDS = {"l_swin_block_c32_nc": ("l_swin.co", 32, _params_c32), "swin_block_c96_nc": ("swin.co", 96, _params_c96)}

# Each back end's output buffer starts filled with its own non-zero sentinel byte, different from the other's: a byte
# that one path (or both) never writes then differs between the two outputs instead of comparing equal as zeros.
HIP_SENTINEL, D3D12_SENTINEL = 0xA5, 0x5A


def hip_block_raw(g, co, kernel, xb, blob, H, W, off, C):
    """The block on HIP; returns the output buffer's downloaded bytes as they are (never decoded and encoded again, so
    for example the two FP8 NaN codes stay distinct). The output starts filled with HIP_SENTINEL."""
    h = g.hip
    n = H * W * C
    px, pw, po = h.malloc(len(xb)), h.malloc(len(blob)), h.malloc(n)
    try:
        h.upload(px, xb)
        h.upload(pw, blob)
        h.memset(po, HIP_SENTINEL, n)
        ox, oy = off
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        args = [ctypes.c_void_p(px), ctypes.c_void_p(0), ctypes.c_void_p(po), ctypes.c_void_p(pw),
                ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)]
        h.launch(h.get_function(g.mods[co], kernel), -(-nwin // 8), 256, args)
        h.sync()
        return bytes(h.download(po, n))
    finally:
        for p in (px, pw, po):
            h.free(p)


def run(build, rocm, seeds, kernels):
    g = C32.Gpu(build, rocm)
    cdir = os.path.join(build, "replay")
    os.makedirs(cdir, exist_ok=True)
    lines, hip_out = [], {}
    neg = None  # (case name, kernel) of the negative control
    for kernel in kernels:
        co, C, gen = KINDS[kernel]
        for si, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            for ci, (H, W, off) in enumerate(CASES):
                name = f"{kernel}_s{si}_{ci}"
                blob = gen(rng)
                if not np.frombuffer(blob[:4], np.uint32)[0] & 1:
                    continue                                    # only the _nc forms are ported
                x = q8(rng.normal(0, 1.5, (H, W, C)))
                xb = fp8_encode(x).tobytes()
                hip_out[name] = (kernel, hip_block_raw(g, co, kernel, xb, blob, H, W, off, C))
                _write(os.path.join(cdir, name + ".x"), xb)
                _write(os.path.join(cdir, name + ".w"), blob)
                ox, oy = off
                nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
                lines += [f"case {name}", f"module {co}", f"kernel {kernel} {-(-nwin // 8)} 256",
                          f"arg buf {name}.x", "arg null", f"arg out {H * W * C} {name}.out {D3D12_SENTINEL}", f"arg buf {name}.w",
                          f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {ox}", f"arg u32 {oy}", ""]
                if neg is None and nwin > 8:
                    # Negative control: this case again on D3D12 with half its workgroups. The windows never processed
                    # keep the D3D12 sentinel, so the comparison must find them: the check can fail.
                    neg = (name, kernel)
                    lines += [f"case negative_{name}", f"module {co}", f"kernel {kernel} {max(1, -(-nwin // 8) // 2)} 256",
                              f"arg buf {name}.x", "arg null", f"arg out {H * W * C} negative_{name}.out {D3D12_SENTINEL}",
                              f"arg buf {name}.w", f"arg u32 {H}", f"arg u32 {W}", f"arg u32 {ox}", f"arg u32 {oy}", ""]
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=150, creationflags=0x08000000)
    res = {"replay_exit": r.returncode, "replay_tail": r.stdout[-600:] + r.stderr[-300:], "cases": []}
    for name, (kernel, want) in hip_out.items():
        path = os.path.join(cdir, name + ".out")
        got = open(path, "rb").read() if os.path.exists(path) else b""
        a = np.frombuffer(got, np.uint8) if got else np.zeros(0, np.uint8)
        b = np.frombuffer(want, np.uint8)
        res["cases"].append({"case": name, "kernel": kernel, "bytes": len(want),
                             "identical": got == want,
                             "differ": int(np.sum(a != b)) if a.size == b.size else -1,
                             # bytes still equal to their buffer's sentinel (a value the kernel may also write)
                             "hip_sentinel_bytes": int(np.sum(b == HIP_SENTINEL)),
                             "d3d12_sentinel_bytes": int(np.sum(a == D3D12_SENTINEL)) if a.size else -1})
    res["per_kernel"] = {k: {"cases": sum(c["kernel"] == k for c in res["cases"]),
                             "identical": sum(c["identical"] for c in res["cases"] if c["kernel"] == k)}
                         for k in kernels}
    caught = None
    if neg:
        path = os.path.join(cdir, "negative_" + neg[0] + ".out")
        got = open(path, "rb").read() if os.path.exists(path) else b""
        want = hip_out[neg[0]][1]
        a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
        caught = len(got) == len(want) and got != want
        res["negative_control"] = {"case": neg[0], "kernel": neg[1], "half_grid": True, "caught": caught,
                                   "differ": int(np.sum(a != b)) if a.size == b.size else -1,
                                   "d3d12_sentinel_bytes": int(np.sum(a == D3D12_SENTINEL)) if a.size else -1}
    res["pass"] = r.returncode == 0 and caught is not False and all(v["cases"] > 0 and v["identical"] == v["cases"]
                                                                    for v in res["per_kernel"].values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-control", action="store_true", help="skip Model M's swin_block_c96_nc control")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    os.makedirs(a.build, exist_ok=True)
    if a.prepare:
        print("built", prepare(a.build, a.rocm))
    if a.run:
        kernels = ["l_swin_block_c32_nc"] + ([] if a.no_control else CONTROL_KERNELS)
        res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)), kernels)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        print({k: v for k, v in res.items() if k not in ("cases", "replay_tail")})
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
