# SPDX-License-Identifier: MIT
"""Model K's native RDNA 3 stage kernels (kernels/k_block_g11.hlsl, names <port>__g11) against the RDNA 4 ports run
through the shortcut, on the same input bytes. WrongGPU's own code. Developer tool of the native RDNA 3 work.

  --prepare  (CPU) writes <cases>: random stage weights (the RDNA 4 blob, k_pack.py) and inputs (special values in two
             of every three cases, as k_port_check.py's narrow family), and two kernel_replay case files over the same
             input files: shortcut.txt (the ports, k_stage_e2 ...) and native.txt (k_stage_e2__g11 ...), one chain per
             case and form, each output saved as <form>_<case>.<buffer>.d3d12, plus cases.json (outputs, defined
             bytes, the NumPy reference of every case without special values, in <case>.<buffer>.ref.npy). One
             negative control: a native chain with half its grid (dropped windows hold outputs). --time N adds
             `time N` to every chain (the timing round: large maps only, --timing).
  --emulate  (CPU) runs both case files through the emulation of the kernel text (clean/dx12/emu: the ports on the
             gfx12 model, the native kernels on the gfx11 model; exact sums, so the two must agree everywhere) and
             writes the outputs where kernel_replay would: a dry run of the files and of --compare.
  --compare  reads the outputs and checks, per case: native = shortcut byte for byte over the defined bytes (a NaN
             whose two forms differ only in the sign is counted apart), the 64 bytes of slack still hold the sentinel,
             and each form against NumPy within spec 49's stage limits (mean 2.5e-3, max 3e-2 of the reference RMS;
             cases without special values); the negative control must differ. Exit 1 on any failure; --json writes
             the counts.

GPU runs (GPU-TESTS.md of the native RDNA 3 work): kernel_replay built with WG_DEV_GFX11, the K = 16 patched
vkd3d-proton, shortcut.txt with DXIL_SPIRV_CONFIG=wmma_rdna3_raw_fp8 WG_DEV_GFX11_SHORTCUT=1, native.txt without
them (the device then reads <build>/dx12-gfx11, which needs the native set and native_fill).
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean", "dx12"))

SENTINEL, SLACK = 0x5A, 64
LIMITS = dict(mean=2.5e-3, max=3e-2)
TOKEN = "s184:p19.b32"
# port name -> (stage, stage-runner suffix, d0 form)
KERNELS = {"k_stage_e2": ("e2", "", None), "k_stage_e0": ("e0", "", None), "k_stage_e1": ("e1", "", None),
           "k_stage_d2": ("d2", "", None), "k_stage_d1": ("d1", "", None), "k_stage2_e0": ("e0", "@w2", None),
           "k_stage2_e1": ("e1", "@w2", None), "k_stage2_d1": ("d1", "@w2", None),
           "k_stage2_d0_hp": ("d0", "@w2", "hp"), "k_stage2_d0_hidden": ("d0", "@w2", "hidden"),
           "k_stage_e2__rodd": ("e2", "", None), "k_stage_d2__rodd": ("d2", "", None),
           "k_stage2_d1__rodd": ("d1", "@w2", None)}
SIZES = [(8, 8), (24, 16), (17, 13), (40, 24), (9, 33), (12, 20)]    # block-level (W, H), as k_port_check.py
OFFSETS = [(0, 0), (2, 2), (4, 0), (0, 6), (6, 4)]
TIMING_SIZES = {0: (640, 384), 1: (320, 192), 2: (160, 96)}          # the 1440p Quality frame's levels


def native_name(port):
    base, rodd = (port[:-6], "__rodd") if port.endswith("__rodd") else (port, "")
    return base + rodd + "__g11"


class MemDev:
    """Memory only (the plan builder's device): byte offsets in one arena; uploads are recorded."""

    def __init__(self):
        self.top, self.data, self.sizes, self.inputs = 4096, {}, {}, set()

    def malloc(self, nbytes):
        p = self.top
        self.top = (self.top + int(nbytes) + SLACK + 255) & ~255
        self.sizes[p] = int(nbytes)
        return p

    def upload(self, ptr, data):
        self.inputs.add(ptr)
        self.data[ptr] = bytes(data)

    def memset(self, ptr, value, nbytes):
        pass

    def free(self, ptr):
        pass


def _plan(stage, suffix, d0, blob, x, offset, skip):
    from clean.k.runner import k_stages
    dev = MemDev()
    plan = k_stages.prepare(dev, stage + suffix, blob, x, offset, skip)
    if d0:   # the d0 forms take the hidden history; the compact head is [2H][2W][4]
        H, W = skip.shape[:2]
        hid = dev.malloc(W * H * 8)
        plan["bufs"]["hidden"] = hid
        if d0 == "hp":
            plan["bufs"]["head"] = dev.malloc(2 * H * 2 * W * 4 * 2)
        k0, grid, block, args = plan["launches"][0]
        args[0].head = plan["bufs"]["head"]
        plan["launches"] = [(k0, grid, block, args + [ctypes.c_void_p(hid)])]
        plan["outputs"] = [("hidden", (H, W, 4))] + ([("head", (2 * H, 2 * W, 4))] if d0 == "hp" else
                                                    [("head", (H, W, 40))]) + [o for o in plan["outputs"]
                                                                               if o[0] == "block"]
    return dev, plan


def prepare(cases, seeds, timing=False, time_runs=0):
    from clean.k.dx12 import k_plan_bridge as PB
    from clean.k.dx12.k_port_check import narrow_inputs
    from clean.k.reference import network as N
    from clean.k.reference.layout import STAGE
    from clean.k.runner import k_stages
    os.makedirs(cases, exist_ok=True)
    lines = {"shortcut": [], "native": []}
    index, neg = [], None
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        params = N.random_network_params(seed)
        for kernel, (stage, suffix, d0) in KERNELS.items():
            st = STAGE[stage]
            sizes = [TIMING_SIZES[st["level"]]] if timing else SIZES
            for ci, size in enumerate(sizes):
                offset = OFFSETS[(ci + si) % len(OFFSETS)]
                special = 0 if timing else ci % 3
                name = f"s{si}_{kernel}_{ci}"
                blob = k_stages.pack(stage, params[stage])
                x, skip = narrow_inputs(stage, size, rng, special)
                dev, plan = _plan(stage, suffix, d0, blob, x, offset, skip)
                outputs = [n for n, _ in plan["outputs"]]
                for n, p in plan["bufs"].items():
                    if p in dev.inputs:
                        with open(os.path.join(cases, f"{name}.{n}.in"), "wb") as f:
                            f.write(dev.data[p])
                k0, grid, block, args = plan["launches"][0]
                tokens = [TOKEN] + (["p"] if d0 else [])
                extra_time = [f"time {time_runs}"] if time_runs else []
                for form, kname in (("shortcut", kernel), ("native", native_name(kernel))):
                    one = dict(plan, launches=[(kname, grid, block, args)])
                    ls = PB.chain_lines(name, one, dev, lambda k: tokens, cases, ["k_block.co"], outputs,
                                        prefix=f"{form}_")
                    lines[form] += ls[:-1] + extra_time + [""]
                if neg is None and grid >= 4 and not timing:
                    neg = name
                    one = dict(plan, launches=[(native_name(kernel), grid, block, args)])
                    lines["native"] += PB.chain_lines(name, one, dev, lambda k: tokens, cases, ["k_block.co"],
                                                      outputs, prefix="negative_", grid_div=2)
                item = {"case": name, "kernel": kernel, "stage": stage, "size": list(size), "offset": list(offset),
                        "specials": special, "grid": grid,
                        "outputs": {n: {"shape": list(sh), "bytes": dev.sizes[plan["bufs"][n]],
                                        "defined": min(int(np.prod(sh)) * 2, dev.sizes[plan["bufs"][n]])}
                                    for n, sh in plan["outputs"]}}
                if not special and not timing:
                    ref = N.run_stage(st, params[stage], x, offset, skip, size) if skip is not None else \
                        N.run_stage(st, params[stage], x, offset)
                    for n in outputs:
                        if n in ref and not (d0 == "hp" and n == "head"):
                            np.save(os.path.join(cases, f"{name}.{n}.ref.npy"), np.asarray(ref[n], np.float32))
                index.append(item)
    for form, ls in lines.items():
        with open(os.path.join(cases, f"{form}.txt"), "w") as f:
            f.write("\n".join(ls))
    with open(os.path.join(cases, "cases.json"), "w") as f:
        json.dump({"cases": index, "negative": neg}, f, indent=1)
    return index


# ---- the CPU dry run: the case files through the emulation of the kernel text --------------------------------------
def _parse_chains(path):
    chains, cur = [], None
    for line in open(path).read().splitlines():
        w = line.split()
        if not w:
            continue
        if w[0] == "chain":
            cur = {"name": w[1], "bufs": [], "steps": [], "saves": []}
            chains.append(cur)
        elif w[0] == "buf":
            cur["bufs"].append((w[1], int(w[2]), w[3], int(w[4]) if len(w) > 4 else 0))
        elif w[0] == "step":
            cur["steps"].append((w[1], int(w[2]), int(w[3]), w[4:]))
        elif w[0] == "save":
            cur["saves"].append((w[1], w[2]))
    return chains


def emulate(cases, work):
    from emu import hlsl_emu as H
    ports_src = os.path.join(ROOT, "clean", "k", "dx12", "kernels", "k_block.hlsl")
    native_src = os.path.join(ROOT, "clean", "k", "dx12", "kernels", "k_block_g11.hlsl")
    groups = [["k_stage_e2", "k_stage2_e0", "k_stage2_e1", "k_stage_d2", "k_stage2_d1", "k_stage_d1"],
              ["k_stage_e0"], ["k_stage_e1"], ["k_stage2_d0_hp", "k_stage2_d0_hidden"],
              ["k_stage_e2__rodd", "k_stage_d2__rodd", "k_stage2_d1__rodd"]]
    kernels = {}
    for g in groups:
        lib, ent = H.build(ports_src, g, work, gfx=12)
        kernels.update({n: (lib, ent[n]) for n in g})
        nat = [native_name(n) for n in g]
        lib, ent = H.build(native_src, nat, work, gfx=11)
        kernels.update({n: (lib, ent[n]) for n in nat})
    dev = H.HlslEmu(kernels, arena_bytes=256 << 20)
    for form in ("shortcut", "native"):
        for ch in _parse_chains(os.path.join(cases, f"{form}.txt")):
            dev.reset()
            ptr = {}
            for bid, size, src, fill in ch["bufs"]:
                p = dev.malloc(size)
                dev.memset(p, fill, size)
                if src != "-":
                    with open(os.path.join(cases, src), "rb") as f:
                        dev.upload(p, f.read())
                ptr[bid] = p
            for kname, grid, block, args in ch["steps"]:
                words = [0, grid]
                for a in args:
                    if a == "null":
                        words.append(0)
                    elif a.startswith("@"):
                        b, _, off = a[1:].partition("+")
                        words.append(ptr[b] + int(off or 0))
                    elif a.startswith("S:"):
                        f, _, patch = a[2:].partition(":")
                        data = bytearray(open(os.path.join(cases, f), "rb").read())
                        for pt in filter(None, patch.split(",")):
                            o, _, r = pt.partition("=")
                            b, _, off = r[1:].partition("+")
                            data[int(o):int(o) + 8] = int(ptr[b] + int(off or 0)).to_bytes(8, "little")
                        np19 = 19   # s184:p19.b32: 19 pointers (one root constant each), then 32 bytes
                        words += [int.from_bytes(data[8 * i:8 * i + 4], "little") for i in range(np19)]
                        words += list(np.frombuffer(bytes(data[8 * np19:8 * np19 + 32]), np.uint32))
                    else:
                        words.append(int(a) & 0xffffffff)
                lib, entry = dev.kernels[kname]
                c = (ctypes.c_uint * len(words))(*[int(w) for w in words])
                lib.hemu_set_consts(c, len(words))
                lib.hemu_set_arena(dev.arena.ctypes.data, len(dev.arena))
                if lib.hemu_run(entry.encode(), grid, block) != 0 or lib.hemu_last_error() or lib.hemu_oob_count():
                    raise RuntimeError(f"{ch['name']}: {kname} failed in the emulation")
            for bid, fn in ch["saves"]:
                size = next(s for b, s, _, _ in ch["bufs"] if b == bid)
                with open(os.path.join(cases, fn), "wb") as f:
                    f.write(dev.download(ptr[bid], size))


# ---- the comparison -------------------------------------------------------------------------------------------------
def _sign_only(a, b):
    nan = lambda v: ((v & 0x7C00) == 0x7C00) & ((v & 0x03FF) != 0)
    return (a != b) & nan(a) & nan(b) & ((a ^ b) == np.uint16(0x8000))


def compare(cases):
    meta = json.load(open(os.path.join(cases, "cases.json")))
    res = {"cases": 0, "outputs": 0, "differ": 0, "sign_only": 0, "slack": 0, "limits": 0, "missing": 0,
           "worst": {}, "failures": []}
    read = lambda fn: open(os.path.join(cases, fn), "rb").read() if os.path.exists(os.path.join(cases, fn)) else None
    for c in meta["cases"]:
        res["cases"] += 1
        for n, o in c["outputs"].items():
            res["outputs"] += 1
            nat, sc = read(f"native_{c['case']}.{n}.d3d12"), read(f"shortcut_{c['case']}.{n}.d3d12")
            if nat is None or sc is None:
                res["missing"] += 1
                res["failures"].append(f"{c['case']}.{n}: output missing")
                continue
            d = o["defined"]
            a, b = np.frombuffer(nat[:d], np.uint16), np.frombuffer(sc[:d], np.uint16)
            so = _sign_only(a, b)
            bad = int(np.sum((a != b) & ~so))
            res["sign_only"] += int(np.sum(so))
            if bad:
                res["differ"] += 1
                res["failures"].append(f"{c['case']}.{n}: {bad} of {a.size} values differ from the shortcut")
            for form, raw in (("native", nat), ("shortcut", sc)):
                if any(v != SENTINEL for v in raw[o["bytes"]:o["bytes"] + SLACK]):
                    res["slack"] += 1
                    res["failures"].append(f"{c['case']}.{n}: {form} wrote into the slack")
            rp = os.path.join(cases, f"{c['case']}.{n}.ref.npy")
            if os.path.exists(rp):
                ref = np.load(rp).astype(np.float64)
                for form, raw in (("native", nat), ("shortcut", sc)):
                    got = np.frombuffer(raw[:d], np.float16).astype(np.float64).reshape(ref.shape)
                    rms = np.sqrt(np.mean(ref ** 2))
                    err = np.abs(got - ref)
                    m, x = (float(err.mean() / rms), float(err.max() / rms)) if np.all(np.isfinite(got)) else (
                        float("inf"), float("inf"))
                    w = res["worst"].setdefault(form, [0.0, 0.0])
                    w[0], w[1] = max(w[0], m), max(w[1], x)
                    if m > LIMITS["mean"] or x > LIMITS["max"]:
                        res["limits"] += 1
                        res["failures"].append(f"{c['case']}.{n}: {form} outside the limits ({m:.3g}, {x:.3g})")
    neg = meta.get("negative")
    if neg:
        c = next(c for c in meta["cases"] if c["case"] == neg)
        same = all(read(f"negative_{neg}.{n}.d3d12") == read(f"shortcut_{neg}.{n}.d3d12") for n in c["outputs"])
        res["negative_caught"] = not same
        if same:
            res["failures"].append(f"negative control {neg}: the half grid was not caught")
    res["pass"] = not res["failures"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="the folder of case files and outputs")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--timing", action="store_true", help="--prepare: the 1440p Quality levels only, no specials")
    ap.add_argument("--time", type=int, default=0, help="--prepare: `time N` after every chain")
    ap.add_argument("--work", help="--emulate: build folder of the emulation (default <cases>/emu)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.prepare:
        idx = prepare(a.cases, [int(s) for s in a.seeds.split(",")], a.timing, a.time)
        print(f"prepared {len(idx)} cases in {a.cases}")
    if a.emulate:
        emulate(a.cases, a.work or os.path.join(a.cases, "emu"))
        print("emulated both forms")
    if a.compare:
        r = compare(a.cases)
        print(json.dumps({k: v for k, v in r.items() if k != "failures"}, indent=1))
        for f in r["failures"][:40]:
            print("FAIL", f)
        if a.json:
            json.dump(r, open(a.json, "w"), indent=1)
        return 0 if r["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
