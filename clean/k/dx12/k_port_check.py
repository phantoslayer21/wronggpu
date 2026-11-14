# SPDX-License-Identifier: MIT
"""Model K (issue #15): byte-for-byte check of a K kernel's D3D12 port against its HIP kernel, one family at a time.

    python clean/k/dx12/k_port_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare [--family expo]
    python clean/k/dx12/k_port_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run
        [--family expo,warm] [--seeds N] --json out.json

--prepare (CPU) builds into <dir>: the HIP code objects the families need (clean/k/runner/k_build.py's build list, and
Model M's frame.co for the control), the D3D12 ports (clean/dx12/build_dx12.py --only), and kernel_replay.exe.
--run (GPU, one GPU slot) runs every case on HIP in this process and on D3D12 through kernel_replay on the same input
bytes. The rules are the Model L checks' (clean/l/runner/l_fold_check.py):
- the RAW downloaded bytes are compared, nothing is decoded and encoded again;
- every output buffer has 64 bytes of slack after the bytes the kernel defines and starts filled with its run's own
  sentinel (HIP 0xA5, D3D12 0x5A): every defined byte identical, every slack byte its own run's sentinel;
- special values (zero, negative zero, subnormal, the largest and smallest FP16, beyond the FP16 range, infinity, NaN)
  are among the inputs of every family that reads values;
- a negative control (the port run on an input that differs from the one HIP ran: the wrong result) must be caught;
  the half-grid control (the port again with half its workgroups) is used by the families whose grid has more than one
  workgroup;
- Model M's scale_rgb (frame.co, its port already proven) runs through the same replay as a control.
The families: narrow = the narrow network stages (k_block.co: k_stage_e2 .. k_stage2_d0_hidden; the stage runner's
launch plans, clean/k/dx12/k_plan_bridge.py), on random stage weights and inputs with special values (NaN, infinity,
+-65504, zero, -0, subnormal) in every other case; expo = k_expo_scalars against k_io.co (HIP); warm = k_warm (D3D12 only, no HIP kernel: it is checked
against its own definition, the dummy word is written exactly when the key equals a lane's loaded value).
One known limit, shared with Model M's ports: D3D12 flushes FP32 denormals (inputs and results) to zero, HIP keeps them.
k_expo_scalars only meets one with a pre-exposure above about 8.5e37 (kappa = RN(1 / pi) * E), so the inputs here keep
every FP32 input and result out of the denormal range (the FP16 specials are in: they are FP32-normal or zero).
Output: counts and verdicts only, no timing.
"""
import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

HIP_SENTINEL, D3D12_SENTINEL = 0xA5, 0x5A
SLACK = 64
CONTROL = "scale_rgb"   # Model M, frame.co
# family -> (HIP code objects, D3D12 kernels)
NARROW_KERNELS = ["k_stage_e2", "k_stage_e0", "k_stage_e1", "k_stage_d2", "k_stage_d1", "k_stage2_e0", "k_stage2_e1",
                  "k_stage2_d1", "k_stage2_d0_hp", "k_stage2_d0_hidden"]
FAMILY_KERNELS = {"expo": (["k_io.co"], ["k_expo_scalars"]), "warm": ([], ["k_warm"]),
                  "narrow": (["k_block.co"], NARROW_KERNELS)}
CONTROL_CO = "frame.co"


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def f32_bits(x):
    return struct.pack("<f", x)


# ---- build ---------------------------------------------------------------------------------------------------------

def prepare(build, rocm, families):
    from clean.k.runner import k_build
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment, hip_child_environment
    os.makedirs(build, exist_ok=True)
    clean_process_environment(rocm)
    clang = B.find_clang(rocm)
    want = {co for f in families for co in FAMILY_KERNELS[f][0]}
    for co, src, defines in k_build.BUILDS:
        if co in want:
            extra = ["-I", k_build.M_KERNEL_DIR] + (["-mwavefrontsize64"] if src == "k_io.hip" else [])
            B.run(B.device_command(clang, rocm, os.path.join(k_build.K_KERNEL_DIR, src), os.path.join(build, co),
                                   defines=defines, extra=extra))
    B.run(B.device_command(clang, rocm, os.path.join(ROOT, "clean", "kernels", "frame.hip"),
                           os.path.join(build, CONTROL_CO)))
    env, _ = hip_child_environment(rocm_dir=rocm)
    names = sorted({k for f in families for k in FAMILY_KERNELS[f][1]} | {CONTROL, "native_fill"})
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


# ---- replay --------------------------------------------------------------------------------------------------------

def replay(build, cdir, lines, tag):
    path = os.path.join(cdir, f"cases_{tag}.txt")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "256"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, path], env=env, capture_output=True, text=True,
                       timeout=150, creationflags=0x08000000)
    return r.returncode, r.stdout[-500:] + r.stderr[-300:]


def compare(got, want, defined, sent_got, sent_want):
    """Raw bytes: identical over the first `defined` bytes; after them each buffer keeps its own sentinel."""
    a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
    if a.size != b.size or a.size < defined:
        return {"bytes": int(b.size), "differ": -1, "slack_bad": -1, "identical": False}
    idx = np.flatnonzero(a[:defined] != b[:defined])
    d = int(idx.size)
    bad = int(np.sum(a[defined:] != sent_got) + np.sum(b[defined:] != sent_want))
    r = {"bytes": int(b.size), "differ": d, "slack_bad": bad, "identical": d == 0 and bad == 0}
    if d:
        w = int(idx[0]) // 4 * 4   # the first differing 32-bit word: D3D12 value, HIP value (hex)
        r["first_word"] = [w, a[w:w + 4].tobytes()[::-1].hex(), b[w:w + 4].tobytes()[::-1].hex()]
    return r


class Hip:
    def __init__(self, build, rocm, cos):
        from clean.runner.hip import Hip as H
        self.h = H(rocm)
        self.mods = {}
        for co in cos:
            with open(os.path.join(build, co), "rb") as f:
                self.mods[co] = self.h.load_module(f.read())

    def run(self, co, kernel, grid, block, args, bufs):
        """bufs: [(initial bytes (padded to its size with HIP_SENTINEL), size)]; returns every buffer's bytes. args:
        a list in which an int i stands for the pointer of bufs[i]... given as ('buf', i), or a ctypes value."""
        h = self.h
        ptrs = []
        try:
            for init, size in bufs:
                p = h.malloc(max(size, 4))
                h.memset(p, HIP_SENTINEL, size)
                if init:
                    h.upload(p, init)
                ptrs.append(p)
            real = [ctypes.c_void_p(ptrs[a[1]]) if isinstance(a, tuple) else a for a in args]
            h.launch(h.get_function(self.mods[co], kernel), grid, block, real)
            h.sync()
            return [bytes(h.download(p, s)) for p, (_, s) in zip(ptrs, bufs)]
        finally:
            for p in ptrs:
                h.free(p)


# ---- family expo: k_expo_scalars ----------------------------------------------------------------------------------

F16_SPECIAL = [0.0, -0.0, 6e-8, 5.9e-8, 3e-8, 1e-8, 6.1e-5, 65504.0, 65519.0, 65520.0, 1e6, -1.0, -65504.0, 0.5, 1.0,
               2.0, 0.00077, 1.0 / 0.00077, 3.0e38, float("inf"), float("-inf"), float("nan"), 1.0e-45]
E_VALUES = [0.25, 0.75, 1.0, 1.25, 1.5, 2.0, 3.5, 8.0, 100.0, 0.0625, 1000.0]


def expo_cases(rng):
    """(args dict, st bytes, src bytes, measured bytes): every mode combination, with normal and special values."""
    cases = []
    n = 0
    for hdr in (1, 0):
        for src_mode in (0, 1, 2):
            for aux in (0, 1):
                for first, reset in ((0, 0), (1, 0), (0, 1)):
                    n += 1
                    special = n % 3 == 0
                    pick = (lambda: float(rng.choice(F16_SPECIAL))) if special else (lambda: float(rng.choice(E_VALUES)))
                    e1, e2 = (1.0, 1.0) if n % 4 else (float(rng.choice([0.5, 4.0, 1000.0])),) * 2
                    inv_pi = float(rng.choice([1.0, 1.0 / 0.00077, 2.0, 1.0 / 1024.0])) if n % 5 else float(
                        rng.choice([0.0, float("inf"), float("nan"), 1e-30]))
                    a = dict(value=pick(), src_mode=src_mode, aux=aux, first=first, hdr=hdr, reset=reset, e1=e1, e2=e2,
                             inv_pi=inv_pi)
                    st = struct.pack("<4f", pick(), pick(), pick(), pick())
                    cases.append((a, st, f32_bits(pick()), f32_bits(pick())))
    return cases


def expo_struct(a):
    from clean.k.runner.k_io import KExpoArgs
    return bytes(KExpoArgs(**a))


def expo_sizes():
    return {"st": 16 + SLACK, "src": 4 + 16, "measured": 4 + 16, "xs": 16 + SLACK}


def run_expo(build, rocm, seeds, cdir, hip):
    """Returns (results list, replay lines for the chains, bookkeeping)."""
    from clean.k.runner.k_io import KExpoArgs
    sizes = expo_sizes()
    items, lines = [], []
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for ci, (a, st, src, meas) in enumerate(expo_cases(rng)):
            name = f"expo_s{si}_{ci}"
            args = expo_struct(a)
            outs = hip.run("k_io.co", "k_expo_scalars", 1, 32,
                           [("buf", 0), ("buf", 1), ("buf", 2), ("buf", 3), KExpoArgs(**a)],
                           [(st, sizes["st"]), (src, sizes["src"]), (meas, sizes["measured"]), (None, sizes["xs"])])
            write(os.path.join(cdir, name + ".st"), st)
            write(os.path.join(cdir, name + ".src"), src)
            write(os.path.join(cdir, name + ".meas"), meas)
            write(os.path.join(cdir, name + ".args"), args)
            lines += chain_lines(name, "", 1, sizes, "k_expo_scalars", "k_io.co")
            items.append({"case": name, "hip": outs, "args": a})
    # negative control: HIP runs value 1.25, the port is given 1.5 (a different FP16 exposure: kappa and lambda differ)
    ctl = dict(value=1.25, src_mode=0, aux=0, first=1, hdr=1, reset=0, e1=1.0, e2=1.0, inv_pi=1.0)
    st0 = struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
    src0 = meas0 = f32_bits(1.0)
    outs = hip.run("k_io.co", "k_expo_scalars", 1, 32, [("buf", 0), ("buf", 1), ("buf", 2), ("buf", 3), KExpoArgs(**ctl)],
                   [(st0, sizes["st"]), (src0, sizes["src"]), (meas0, sizes["measured"]), (None, sizes["xs"])])
    write(os.path.join(cdir, "expo_ctl.st"), st0)
    write(os.path.join(cdir, "expo_ctl.src"), src0)
    write(os.path.join(cdir, "expo_ctl.meas"), meas0)
    write(os.path.join(cdir, "expo_ctl.args"), expo_struct(dict(ctl, value=1.5)))
    lines += chain_lines("expo_ctl", "negative_", 1, sizes, "k_expo_scalars", "k_io.co")
    return items, lines, {"case": "expo_ctl", "hip": outs}


def chain_lines(name, prefix, grid, sizes, kernel, co):
    p = prefix + name
    return [f"chain {p}", f"module {co}",
            f"buf st {sizes['st']} {name}.st {D3D12_SENTINEL}", f"buf src {sizes['src']} {name}.src {D3D12_SENTINEL}",
            f"buf meas {sizes['measured']} {name}.meas {D3D12_SENTINEL}", f"buf xs {sizes['xs']} - {D3D12_SENTINEL}",
            f"step {kernel} {grid} 32 @st @src @meas @xs B:{name}.args", f"save st {p}.out_st", f"save xs {p}.out_xs", ""]


# ---- family warm: k_warm ------------------------------------------------------------------------------------------

def warm_struct(ptr_slots, ns, line, key):
    """KWarmParams: r[12], dummy (13 pointers, 8 bytes each: patched by the replay), n[12], line, key."""
    return b"".join([bytes(8 * 13), struct.pack("<12I", *ns), struct.pack("<II", line, key)])


def warm_cases(rng):
    """(name, ranges [(bytes)], line, key rule, n list): one lane's XOR of its words matches the key (the dummy is
    written) or no lane's does (it keeps the sentinel)."""
    out = []
    for ci in range(6):
        nr = 1 + ci % 3
        line = [4, 16, 64, 256][ci % 4]
        ranges = []
        for _ in range(nr):
            nwords = int(rng.integers(8, 200))
            ranges.append(rng.integers(0, 2 ** 32, nwords, dtype=np.uint64).astype(np.uint32).tobytes())
        out.append((f"warm_{ci}", ranges, line, ci % 2 == 0))
    return out


def run_warm(build, rocm, seeds, cdir, hip):
    """D3D12 only: results are checked here against the kernel's definition."""
    lines, items = [], []
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed + 1000)
        for name, ranges, line, hit in warm_cases(rng):
            name = f"s{si}_{name}"
            ns = [len(r) for r in ranges] + [0] * (12 - len(ranges))
            # lane L of the one workgroup (256 lanes) loads the words at byte offsets L * line + k * 256 * line
            acc = []
            for lane in range(256):
                v = 0
                for r in ranges:
                    for off in range(lane * line, len(r), 256 * line):
                        v ^= struct.unpack_from("<I", r, off)[0]
                acc.append(v)
            if hit:
                key = acc[int(rng.integers(0, 256))] if any(acc) else 0
                # a key one lane produces; every lane with that value stores it
            else:
                used = set(acc)
                key = next(k for k in range(1, 2 ** 32) if k not in used)
            write(os.path.join(cdir, name + ".par"), warm_struct(None, ns, line, key))
            for i, r in enumerate(ranges):
                write(os.path.join(cdir, f"{name}.r{i}"), r)
            patch = ",".join(f"{8 * i}=@r{i}" for i in range(len(ranges))) + f",{8 * 12}=@dummy"
            lines += [f"chain {name}", "module k_block_wide.co"]
            lines += [f"buf r{i} {len(r)} {name}.r{i} 0" for i, r in enumerate(ranges)]
            lines += [f"buf dummy {4 + SLACK} - {D3D12_SENTINEL}",
                      f"step k_warm 1 256 S:{name}.par:{patch}", f"save dummy {name}.out", ""]
            items.append({"case": name, "key": key, "hit": bool(hit), "acc": acc})
    return items, lines


# ---- family narrow: the narrow network stages (k_block.co) ---------------------------------------------------------

# kernel -> (stage, stage-runner suffix, d0 form); the d0 kernels are the network's own forms (hidden history, compact
# pixel head) with the hidden history pointer appended to the stage runner's launch
NARROW = {"k_stage_e2": ("e2", "", None), "k_stage_e0": ("e0", "", None), "k_stage_e1": ("e1", "", None),
          "k_stage_d2": ("d2", "", None), "k_stage_d1": ("d1", "", None), "k_stage2_e0": ("e0", "@w2", None),
          "k_stage2_e1": ("e1", "@w2", None), "k_stage2_d1": ("d1", "@w2", None),
          "k_stage2_d0_hp": ("d0", "@w2", "hp"), "k_stage2_d0_hidden": ("d0", "@w2", "hidden")}
# block-level sizes (W, H) and window offsets (spec 44: even): grids with fewer than a workgroup's windows among them
NARROW_SIZES = [(8, 8), (24, 16), (17, 13), (40, 24), (9, 33), (12, 20)]
NARROW_OFFSETS = [(0, 0), (2, 2), (4, 0), (0, 6), (6, 4)]
# special inputs: level 1 without NaN (infinity, +-65504, zero, -0, the smallest subnormal), level 2 with NaN as well
F16_SPECIALS = [np.array([np.inf, -np.inf, 65504.0, -65504.0, 0.0, -0.0, 6e-8, 1e-4, -3e4], np.float32),
                np.array([np.nan, np.inf, -np.inf, 65504.0, -65504.0, 0.0, -0.0, 6e-8, 1e-4, -3e4], np.float32)]


def narrow_inputs(stage, size, rng, specials):
    from clean.k.reference.layout import STAGE, cdiv
    st = STAGE[stage]
    W, H = size
    if st["kind"] == "dec":
        x = rng.standard_normal((cdiv(H, 2), cdiv(W, 2), st["C_prev"])).astype(np.float32)
        skip = rng.standard_normal((H, W, st["C"])).astype(np.float32)
    else:
        x = rng.standard_normal((H, W, 16 if st.get("embed") else st["C"])).astype(np.float32)
        skip = None
    if specials:
        for a in (x, skip):
            if a is not None:
                pick = rng.random(a.shape) < 0.01
                a[pick] = rng.choice(F16_SPECIALS[specials - 1], int(pick.sum()))
    return x, skip


def itemsize(name):
    return 4 if name == "out" else 2


def run_narrow(build, rocm, seeds, cdir, hip, only=None):
    from clean.k.reference.layout import STAGE
    from clean.k.reference.network import random_stage_params
    from clean.k.runner import k_launch as KL
    from clean.k.runner import k_stages
    from clean.k.dx12 import k_plan_bridge as PB
    tokens = PB.manifest_args(build)
    items, lines, neg = [], [], None
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for kernel, (stage, suffix, d0) in NARROW.items():
            if only and kernel not in only:
                continue
            st = STAGE[stage]
            for ci in range(len(NARROW_SIZES)):
                size, offset = NARROW_SIZES[ci], NARROW_OFFSETS[(ci + si) % len(NARROW_OFFSETS)]
                special = ci % 3   # 0 none, 1 specials without NaN, 2 with NaN
                name = f"nar_s{si}_{kernel}_{ci}"
                blob = k_stages.pack(stage, random_stage_params(st, rng))
                x, skip = narrow_inputs(stage, size, rng, special)
                pdev = PB.PlanDev(hip.h)
                plan = k_stages.prepare(pdev, stage + suffix, blob, x, offset, skip)
                try:
                    if d0:   # the network's d0 forms: k_block.hip's k_stage2_d0_hidden / _hp take the hidden history
                        W, H = size
                        hid = pdev.malloc(W * H * 8)
                        plan["bufs"]["hidden"], plan["sizes"]["hidden"] = hid, W * H * 8
                        k0, grid, block, args = plan["launches"][0]
                        assert k0 == "k_stage2_d0", k0
                        plan["launches"] = [(kernel, grid, block, args + [ctypes.c_void_p(hid)])]
                        plan["outputs"] = [("hidden", (H, W, 4))] + (
                            [("head", (2 * H, 2 * W, 4))] if d0 == "hp" else [("head", (H, W, 40))])
                    outputs = [n for n, _ in plan["outputs"]]
                    for n, p in plan["bufs"].items():
                        if p in pdev.inputs:
                            write(os.path.join(cdir, f"{name}.{n}.in"), bytes(hip.h.download(p, pdev.sizes[p])))
                    for k, grid, block, a in plan["launches"]:
                        co = KL.KERNEL_CODE_OBJECTS[k]
                        hip.h.launch(hip.h.get_function(hip.mods[co], k), grid, block, a)
                    hip.h.sync()
                    sizes = {n: pdev.sizes[plan["bufs"][n]] for n in outputs}
                    got = {n: bytes(hip.h.download(plan["bufs"][n], sizes[n] + SLACK)) for n in outputs}
                    defined = {n: min(int(np.prod(sh)) * itemsize(n), sizes[n]) for n, sh in plan["outputs"]}
                    lines += PB.chain_lines(name, plan, pdev, lambda k: tokens[k.split(":")[-1]], cdir, ["k_block.co"],
                                            outputs)
                    if neg is None and plan["launches"][0][1] >= 4:   # the dropped windows hold output pixels
                        neg = name
                        lines += PB.chain_lines(name, plan, pdev, lambda k: tokens[k.split(":")[-1]], cdir,
                                                ["k_block.co"], outputs, prefix="negative_", grid_div=2)
                    items.append({"case": name, "kernel": kernel, "stage": stage, "size": list(size),
                                  "offset": list(offset), "specials": special, "outputs": outputs, "hip": got,
                                  "defined": defined})
                finally:
                    for p in plan["bufs"].values():
                        pdev.free(p)
    return items, lines, neg


def nan_equivalent_differ(got, want, defined, width=2):
    """Differing bytes over the first `defined` bytes after ignoring the elements (2 bytes: FP16) that are NaN in both:
    IEEE 754 does not define a NaN's sign or payload, and the two devices may generate or propagate a different one."""
    dt = np.uint16 if width == 2 else np.uint32
    a = np.frombuffer(got[:defined], dt)
    b = np.frombuffer(want[:defined], dt)
    exp, mant = (0x7C00, 0x03FF) if width == 2 else (0x7F800000, 0x007FFFFF)
    nan = lambda v: ((v & exp) == exp) & ((v & mant) != 0)
    return int(np.sum((a != b) & ~(nan(a) & nan(b)))), int(np.sum((a != b) & nan(a) & nan(b)))


def check_narrow(state, read):
    items, _, neg = state
    out = []
    for it in items:
        per = {n: compare(read(f"{it['case']}.{n}.d3d12"), it["hip"][n], it["defined"][n], D3D12_SENTINEL, HIP_SENTINEL)
               for n in it["outputs"]}
        raw = all(v["identical"] for v in per.values())
        nan_only = 0
        if not raw and it["specials"] == 2:   # NaN inputs: a NaN's sign / payload is not defined (see nan_equivalent_differ)
            for n, v in per.items():
                if not v["identical"] and v["slack_bad"] == 0:
                    d, same = nan_equivalent_differ(read(f"{it['case']}.{n}.d3d12"), it["hip"][n], it["defined"][n])
                    v["differ_outside_nan"], v["nan_elements_differing_in_sign"] = d, same
                    nan_only += same
        ok_case = raw or all(v["identical"] or (v["slack_bad"] == 0 and v.get("differ_outside_nan", 1) == 0)
                             for v in per.values())
        out.append({"case": it["case"], "kernel": it["kernel"], "outputs": per, "raw_identical": raw,
                    "identical": ok_case, "nan_sign_only_elements": nan_only})
    caught, differ = None, 0
    if neg:
        it = next(i for i in items if i["case"] == neg)
        per = [compare(read(f"negative_{neg}.{n}.d3d12"), it["hip"][n], it["defined"][n], D3D12_SENTINEL, HIP_SENTINEL)
               for n in it["outputs"]]
        caught = all(p["differ"] >= 0 for p in per) and any(not p["identical"] for p in per)
        differ = sum(max(p["differ"], 0) for p in per)
    kernels = sorted({o["kernel"] for o in out})
    by_kernel = {k: [sum(o["identical"] for o in out if o["kernel"] == k), sum(1 for o in out if o["kernel"] == k)]
                 for k in kernels}
    ok = bool(out) and all(o["identical"] for o in out) and caught is True
    return {"cases": len(out), "identical": sum(o["identical"] for o in out), "by_kernel": by_kernel,
            "raw_byte_identical": sum(o["raw_identical"] for o in out),
            "cases_differing_only_in_nan_sign": sum(1 for o in out if not o["raw_identical"] and o["identical"]),
            "nan_elements_differing_in_sign": sum(o["nan_sign_only_elements"] for o in out),
            "special_value_cases": [sum(1 for i in items if i["specials"] == 1), sum(1 for i in items if i["specials"] == 2)],
            "negative_control": {"half_grid": True, "case": neg, "caught": caught, "differ": differ},
            "failures": [o for o in out if not o["identical"]][:4]}, ok


# ---- run -----------------------------------------------------------------------------------------------------------

def run(build, rocm, families, seeds, json_out):
    # a new folder per run: an output of an earlier run must never stand in for one this run did not write
    cdir = os.path.join(build, "replay-k-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(cdir)
    res = {"families": families, "seeds": seeds, "results": {}}
    cos = sorted({co for f in families for co in FAMILY_KERNELS[f][0]} | {CONTROL_CO})
    hip = Hip(build, rocm, cos)
    lines = []
    state = {}
    if "expo" in families:
        state["expo"] = run_expo(build, rocm, seeds, cdir, hip)
        lines += state["expo"][1]
    if "warm" in families:
        state["warm"] = run_warm(build, rocm, seeds, cdir, hip)
        lines += state["warm"][1]
    if "narrow" in families:
        state["narrow"] = run_narrow(build, rocm, seeds, cdir, hip)
        lines += state["narrow"][1]
    # the control: Model M's scale_rgb, same replay, against HIP
    rng = np.random.default_rng(seeds[0])
    cn = 1000
    data = rng.normal(0, 3, cn).astype(np.float32)
    data[:6] = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30]
    pi = np.float32(0.00077)
    write(os.path.join(cdir, "ctl.in"), data.tobytes())
    h = hip.run(CONTROL_CO, CONTROL, -(-cn // 256), 256, [("buf", 0), ctypes.c_float(pi), ctypes.c_int(cn)],
                [(data.tobytes(), cn * 4 + SLACK)])
    lines += ["chain control_scale_rgb", f"module {CONTROL_CO}", f"buf b {cn * 4 + SLACK} ctl.in {D3D12_SENTINEL}",
              f"step {CONTROL} {-(-cn // 256)} 256 @b {struct.unpack('<I', struct.pack('<f', pi))[0]} {cn}",
              "save b ctl.out", ""]
    code, tail = replay(build, cdir, lines, "k")
    res["replay_exit"], res["replay_tail"] = code, tail

    def read(n):
        p = os.path.join(cdir, n)
        return open(p, "rb").read() if os.path.exists(p) else b""

    ok = code == 0
    if "expo" in state:
        out = []
        for it in state["expo"][0]:
            n = it["case"]
            st = compare(read(n + ".out_st"), it["hip"][0], 16, D3D12_SENTINEL, HIP_SENTINEL)
            xs = compare(read(n + ".out_xs"), it["hip"][3], 16, D3D12_SENTINEL, HIP_SENTINEL)
            out.append({"case": n, "st": st, "xs": xs, "identical": st["identical"] and xs["identical"]})
        ctl_case = state["expo"][2]
        neg_xs = compare(read("negative_expo_ctl.out_xs"), ctl_case["hip"][3], 16, D3D12_SENTINEL, HIP_SENTINEL)
        neg_st = compare(read("negative_expo_ctl.out_st"), ctl_case["hip"][0], 16, D3D12_SENTINEL, HIP_SENTINEL)
        caught = (not neg_xs["identical"]) or (not neg_st["identical"])
        res["results"]["expo"] = {"cases": len(out), "identical": sum(o["identical"] for o in out),
                                  "negative_control": {"different_input": True, "caught": caught,
                                                       "differ": max(neg_xs["differ"], 0) + max(neg_st["differ"], 0)},
                                  "failures": [o for o in out if not o["identical"]][:5]}
        ok = ok and caught and all(o["identical"] for o in out) and bool(out)
    if "warm" in state:
        good, fails = 0, []
        for it in state["warm"][0]:
            got = read(it["case"] + ".out")
            word = struct.unpack_from("<I", got, 0)[0] if len(got) >= 4 else None
            slack_ok = got[4:] == bytes([D3D12_SENTINEL]) * (len(got) - 4)
            want_hit = it["hit"] and it["key"] in set(it["acc"])
            if want_hit:
                right = word == it["key"] and slack_ok
            else:
                right = got[:4] == bytes([D3D12_SENTINEL]) * 4 and slack_ok
            good += bool(right)
            if not right:
                fails.append(it["case"])
        res["results"]["warm"] = {"cases": len(state["warm"][0]), "as_defined": good, "failures": fails[:5]}
        ok = ok and good == len(state["warm"][0]) and good > 0
    if "narrow" in state:
        res["results"]["narrow"], good = check_narrow(state["narrow"], read)
        ok = ok and good
    ctl_got = read("ctl.out")
    ctl = compare(ctl_got, h[0], 1000 * 4, D3D12_SENTINEL, HIP_SENTINEL)
    res["control"] = {"kernel": CONTROL, "identical": bool(ctl["identical"]), "differ": ctl["differ"],
                      "first_word": ctl.get("first_word"), "slack_bad": ctl["slack_bad"]}
    ok = ok and ctl["identical"]
    res["pass"] = bool(ok)
    if json_out:
        with open(json_out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "replay_tail"}, indent=1))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--family", default="expo,warm")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    families = a.family.split(",")
    bad = [f for f in families if f not in FAMILY_KERNELS]
    if bad:
        ap.error(f"unknown family {bad} (choose from {sorted(FAMILY_KERNELS)})")
    if a.prepare:
        prepare(a.build, a.rocm, families)
    if a.run:
        return run(a.build, a.rocm, families, list(range(a.seeds)), a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
