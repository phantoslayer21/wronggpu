# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): net1 on gfx12's matrix instructions (tier B of the fast path), HIP and D3D12.

    python clean/fg/runner/fg_net1_mma.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_net1_mma.py --build <dir> --rocm ... --run [--model <model.wgcm>] [--json out.json]
    python clean/fg/runner/fg_net1_mma.py --build <dir> --rocm ... --timing [--json out.json]

The network (NET1_WIRING) on NHWC float16 buffers in 27 launches: the 17 convolutions as fg_mconv (one
v_wmma_f32_16x16x16_f16 per spec chunk in the layer's chunk order, the float16 accumulator from the bias as C and the
result rounded to float16 after every chunk; leaky ReLU, and for net1_01..05 the 2 x 2 max-pool, in the epilogue),
five NHWC upscales and five adds (fg_add_f16).

Gates (the port standard) -- HIP against D3D12, byte for byte:
- every saved buffer (a_k, p_k, b5, d5, u_k, s_k, e_k, d_k, out) compared raw, each filled with its back end's
  sentinel (HIP 0xA5, D3D12 0x5A) and followed by 64 slack bytes nobody may write;
- a negative control per kernel: the first launch of that kernel (with 4 or more workgroups) on half of its grid on
  D3D12 must change its output.
Reported only (oversight 2026-11-14: the gate is the generated frame against the oracle): every buffer and the
unfolded output against the CPU reference (net1.run), as the largest distance in float16 steps and the number of
values more than 1 and more than 4 steps off (non-finite disagreements counted apart).
Cases: synthetic weights at 64 x 64 and 70 x 90 pixels, special values in the input and biases, and with --model the
weights of a converter output file (execute-only: the files written for the replay are deleted after the run).
--timing (information only): a 512 x 512 probe times every launch first; 1920 x 1080 and 2560 x 1440 input pixels run
only when every launch, scaled by the cell count, stays under MAX_LAUNCH_MS.
"""
import argparse
import ctypes
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from clean.fg.reference import net1 as N  # noqa: E402
import fg_d3d12_check as K  # noqa: E402
import fg_net1_gpu as G  # noqa: E402

CO_MMA, CO_NET1 = "fg_net1_mma.co", "fg_net1.co"
MCONV = ["fg_mconv%d" % mt for mt in (1, 2, 4)]
KERNELS = MCONV + ["fg_upscale_hwc_f16", "fg_add_f16"]
MODULE_OF = dict([(k, CO_MMA) for k in MCONV + ["fg_upscale_hwc_f16"]] + [("fg_add_f16", CO_NET1)])
MAX_LAUNCH_MS = 100.0
TIMING_SIZES = G.TIMING_SIZES
PROBE = G.PROBE


# ---- the fast plan ------------------------------------------------------------------------------------------------------

def swizzle(w, order):
    """Canonical weights [out, k, k, in] -> A fragments in chunk order: [out / 16][chunks][32 lanes][8] float16; lane l
    holds row 16 mt + l % 16 and input channels 16 cb + 8 (l // 16) + 0..7 of the chunk's tap."""
    w = np.asarray(w, np.float16)
    cout, k, _, cin = w.shape
    nb = cin // 16
    chunks = []
    for q in range(nb * k * k):
        if order == "ascending":
            cb, ky, kx = q, 0, 0
        elif order == "block_kx_ky":
            cb, t = divmod(q, k * k)
            kx, ky = divmod(t, k)
        else:
            kx, r = divmod(q, nb * k)
            cb, ky = divmod(r, k)
        chunks.append(w[:, ky, kx, 16 * cb:16 * cb + 16])          # [out, 16]
    a = np.stack(chunks, 1)                                         # [out, nq, 16]
    a = a.reshape(cout // 16, 16, len(chunks), 2, 8)                # [mt, m, q, h, j]
    return np.ascontiguousarray(a.transpose(0, 2, 3, 1, 4)).reshape(-1).tobytes()   # [mt, q, h, m, j] = lane 16h + m


def mconv_mt(cout):
    return 4 if cout % 64 == 0 else (2 if cout % 32 == 0 else 1)


def plan(H0, W0, params):
    """(data, acts, steps) of net1 on cells [H0, W0, 48]: data {id: bytes}; acts {id: (shape HWC, bytes)}; steps
    [(kernel, grid, args, outs, op)] with args "@id" or integers."""
    data, acts, steps = {}, {}, []
    shape = {"cells": (H0, W0, N.CELL)}
    pooled = {s["in"]: s["out"] for s in N.NET1_WIRING if s["op"] == "pool"}
    for s in N.NET1_WIRING:
        op = s["op"]
        if op == "pool":
            continue                                                # fused into the convolution before it
        if op == "conv":
            w, b = params[s["layer"]]
            cout, k, _, cin = np.shape(w)
            wid, bid = "wf_" + s["layer"], "b_" + s["layer"]
            data[wid] = swizzle(w, s["chunks"])
            data[bid] = np.asarray(b, np.float16).tobytes()
            H, W, C = shape[s["in"]]
            if C != cin:
                raise ValueError("%s: %d input channels, %s has %d" % (s["layer"], cin, s["in"], C))
            mt = mconv_mt(cout)
            tx, ty = -(-W // 16), -(-H // 16)
            shape[s["out"]] = (H, W, cout)
            outs = [s["out"]]
            pool = s["out"] in pooled
            if pool:
                shape[pooled[s["out"]]] = (H // 2, W // 2, cout)
                outs.append(pooled[s["out"]])
            args = ["@" + s["in"], "@" + wid, "@" + bid, "@" + s["out"], "@" + (outs[-1]), cin, cout, H, W, k,
                    K.ORDER_CODE[s["chunks"]], 1 if s["act"] == "leaky" else 0, 1 if pool else 0, tx, ty]
            steps.append(("fg_mconv%d" % mt, tx * ty * (cout // (16 * mt)), args, outs, "conv"))
        elif op == "upscale":
            hi, wi, C = shape[s["in"]]
            ho, wo, _ = shape[s["like"]]
            shape[s["out"]] = (ho, wo, C)
            n = ho * wo * C
            steps.append(("fg_upscale_hwc_f16", -(-n // 256),
                          ["@" + s["in"], "@" + s["out"], C, hi, wi, ho, wo, K._ratio_bits(wi, wo), K._ratio_bits(hi, ho)],
                          [s["out"]], "upscale"))
        elif op == "add":
            a, b = s["in"]
            shape[s["out"]] = shape[a]
            n = int(np.prod(shape[a]))
            steps.append(("fg_add_f16", -(-n // 256), ["@" + a, "@" + b, "@" + s["out"], n], [s["out"]], "add"))
    for s in steps:
        for o in s[3]:
            acts[o] = (shape[o], 2 * int(np.prod(shape[o])))
    return data, acts, steps


def cells_hwc(cells):
    return np.ascontiguousarray(np.asarray(cells, np.float64).astype(np.float16).transpose(1, 2, 0)).tobytes()


# ---- HIP ----------------------------------------------------------------------------------------------------------------

class HipNet:
    def __init__(self, h, mods, data, acts, steps):
        self.h, self.steps, self.acts = h, steps, acts
        self.fn = {k: h.get_function(mods[MODULE_OF[k]], k) for k in set(s[0] for s in steps)}
        self.ptr = {}
        for i, b in data.items():
            self.ptr[i] = h.malloc(max(len(b), 4))
            h.upload(self.ptr[i], b)
        for i, (_, n) in acts.items():
            self.ptr[i] = h.malloc(n + K.SLACK)
            h.memset(self.ptr[i], K.HIP_SENTINEL, n + K.SLACK)
        self.args = [[ctypes.c_void_p(self.ptr[a[1:]]) if isinstance(a, str) else ctypes.c_uint32(a) for a in s[2]]
                     for s in steps]

    def launch(self, only=None):
        for i, s in enumerate(self.steps):
            if only is None or i == only:
                self.h.launch(self.fn[s[0]], s[1], 256, self.args[i])

    def run(self):
        self.launch()
        self.h.sync()
        return {i: bytes(self.h.download(self.ptr[i], n + K.SLACK)) for i, (_, n) in self.acts.items()}

    time_steps = G.HipNet.time_steps
    time_network = G.HipNet.time_network
    free = G.HipNet.free


def load_modules(h, build):
    return {co: h.load_module(open(os.path.join(build, co), "rb").read()) for co in (CO_MMA, CO_NET1)}


# ---- D3D12 chains --------------------------------------------------------------------------------------------------------

def chain_lines(name, files, acts, steps, half=None, stop=None, saves=None, time_runs=0):
    lines = ["chain " + name, "module " + CO_MMA, "module " + CO_NET1]
    for i, f in files.items():
        lines.append("buf %s %d %s" % (i, os.path.getsize(f["path"]), f["name"]))
    for i, (_, n) in acts.items():
        lines.append("buf %s %d - %d" % (i, n + K.SLACK, K.D3D12_SENTINEL))
    for j, (kernel, grid, args, _, _) in enumerate(steps):
        if stop is not None and j > stop:
            break
        g = max(1, grid // 2) if j == half else grid
        lines.append("step %s %d 256 %s" % (kernel, g, " ".join(str(a) for a in args)))
    for i in (saves if saves is not None else acts):
        lines.append("save %s %s.%s" % (i, name, i))
    if time_runs:
        lines.append("time %d" % time_runs)
    return lines + [""]


# ---- report against the CPU reference ------------------------------------------------------------------------------------

def step_report(got_bits, want):
    """got: float16 bits; want: float64 values (float16-valued). Largest finite distance in float16 steps, counts over
    1 and over 4 steps, non-finite disagreements."""
    w = np.asarray(want, np.float64).astype(np.float16).view(np.uint16).reshape(-1)
    g = np.asarray(got_bits, np.uint16).reshape(-1)

    def key(b):
        b = b.astype(np.int64)
        return np.where(b & 0x8000, -(b & 0x7FFF), b & 0x7FFF)
    fin = ((g & 0x7C00) != 0x7C00) & ((w & 0x7C00) != 0x7C00)
    d = np.where(fin, np.abs(key(g) - key(w)), 0)
    gn = ((g & 0x7C00) == 0x7C00) & ((g & 0x3FF) != 0)
    wn = ((w & 0x7C00) == 0x7C00) & ((w & 0x3FF) != 0)
    nonfin_bad = ~fin & ~((g == w) | (gn & wn))
    return {"values": int(g.size), "steps_max": int(d.max()) if d.size else 0, "over_1_step": int(np.sum(d > 1)),
            "over_4_steps": int(np.sum(d > 4)), "nonfinite_mismatch": int(np.sum(nonfin_bad))}


# ---- run --------------------------------------------------------------------------------------------------------------------

def run(build, rocm, model):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    mods = load_modules(h, build)
    cdir = os.path.join(build, "replay-mma")
    os.makedirs(cdir, exist_ok=True)
    lines, runs, written = [], [], []
    need_mb = 256
    try:
        for cname, values, params in G.cases(model):
            cells = N.fold(values)
            _, gh, gw = values.shape
            H0, W0 = cells.shape[1:]
            ref = N.run(cells, params)
            name = cname
            data, acts, steps = plan(H0, W0, params)
            data = dict(cells=cells_hwc(cells), **data)
            net = HipNet(h, mods, data, acts, steps)
            try:
                hip = net.run()
            finally:
                net.free()
            files = G.write_files(cdir, name, data)
            written += [f["name"] for f in files.values()] + ["%s.%s" % (name, i) for i in acts]
            lines += chain_lines(name, files, acts, steps)
            negs = {}
            for kernel in sorted(set(s[0] for s in steps)):
                j = next((j for j, s in enumerate(steps) if s[0] == kernel and s[1] >= 4), None)
                if j is None:
                    continue
                nn = "negative_%s_%s" % (name, kernel)
                negs[kernel] = (nn, j, steps[j][3])
                lines += chain_lines(nn, files, acts, steps, half=j, stop=j, saves=steps[j][3])
                written += ["%s.%s" % (nn, o) for o in steps[j][3]]
            need_mb = max(need_mb, G.arena_mb(data, acts))
            runs.append({"name": name, "case": cname, "acts": acts, "steps": steps, "hip": hip,
                         "ref": ref, "negs": negs, "dims": (H0, W0, gh, gw)})
        r, secs = G.replay(build, cdir, lines, need_mb, timeout=150)

        def read(fn):
            p = os.path.join(cdir, fn)
            return open(p, "rb").read() if os.path.exists(p) else b""

        res = {"replay_exit": r.returncode, "replay_s": round(secs, 1), "replay_tail": r.stdout[-400:] + r.stderr[-200:],
               "cases": []}
        ok = r.returncode == 0
        for c in runs:
            raw_bad, rep = [], {}
            for out, (shape, n) in c["acts"].items():
                raw = K.compare_raw(read("%s.%s" % (c["name"], out)), c["hip"][out], n)
                if raw["differ"] != 0 or not raw["slack_ok"]:
                    raw_bad.append({"buffer": out, **raw})
                got = np.frombuffer(c["hip"][out][:n], np.uint16).reshape(shape)       # HWC
                rep[out] = step_report(got.transpose(2, 0, 1), c["ref"][out])
            negs = {}
            for kernel, (nn, j, outs) in c["negs"].items():
                diff = 0
                for o in outs:
                    _, n = c["acts"][o]
                    nraw = K.compare_raw(read("%s.%s" % (nn, o)), c["hip"][o], n)
                    diff += max(nraw["differ"], 0) + (0 if nraw["slack_ok"] else 1)
                negs[kernel] = {"step": j, "caught": diff != 0, "differ": diff}
            H0, W0, gh, gw = c["dims"]
            shape, n = c["acts"]["out"]
            o_hwc = np.frombuffer(c["hip"]["out"][:n], np.uint16).reshape(shape).astype(np.uint16)
            o = o_hwc.transpose(2, 0, 1).view(np.float16).astype(np.float64)
            px = N.unfold(o, gh, gw)
            rpx = N.unfold(c["ref"]["out"], gh, gw)
            out_rep = step_report(px.astype(np.float16).view(np.uint16), rpx)
            per_value = [step_report(px[t].astype(np.float16).view(np.uint16), rpx[t]) for t in range(px.shape[0])]
            good = not raw_bad and all(v["caught"] for v in negs.values()) and len(negs) >= 3
            ok = ok and good
            res["cases"].append({"case": c["case"], "input_pixels": [gw, gh], "hip_equals_d3d12": not raw_bad,
                                 "raw_differences": raw_bad[:5], "negative_controls": negs,
                                 "vs_cpu_output": out_rep, "vs_cpu_output_per_value": per_value, "vs_cpu_buffers": rep,
                                 "output_sha256": hashlib.sha256(px.astype("<f2").tobytes()).hexdigest(), "ok": good})
        res["summary"] = {"cases": len(res["cases"]), "cases_ok": sum(c["ok"] for c in res["cases"]),
                          "hip_equals_d3d12_all": all(c["hip_equals_d3d12"] for c in res["cases"]), "pass": bool(ok)}
        return res
    finally:
        G.remove_written(cdir, written)


# ---- timing (information only) ---------------------------------------------------------------------------------------------

def timing(build, rocm, reps=10):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    mods = load_modules(h, build)
    rng = np.random.default_rng(7)
    params = G.synthetic_params(rng)
    cdir = os.path.join(build, "replay-mma")
    os.makedirs(cdir, exist_ok=True)
    res = {"reps": reps, "max_launch_ms": MAX_LAUNCH_MS, "modes": {}}

    def setup(w, hgt):
        H0, W0 = -(-hgt // 2), -(-w // 2)
        cells = rng.uniform(0, 1, (N.CELL, H0, W0))
        cells[4 * N.VALUES:] = 0
        data, acts, steps = plan(H0, W0, params)
        return dict(cells=cells_hwc(cells), **data), acts, steps, H0 * W0

    for acc in ("tier_b",):
        mode = {}
        data, acts, steps, probe_cells = setup(*PROBE)
        net = HipNet(h, mods, data, acts, steps)
        try:
            net.launch()
            h.sync()
            per = net.time_steps(3)
        finally:
            net.free()
        worst = int(np.argmax(per))
        mode["probe"] = {"pixels": list(PROBE), "max_step_ms": round(per[worst], 3), "max_step": worst,
                         "sum_of_steps_ms": round(sum(per), 3)}
        for label, (w, hgt) in TIMING_SIZES.items():
            cells_n = (-(-hgt // 2)) * (-(-w // 2))
            predicted = per[worst] * cells_n / probe_cells
            entry = {"pixels": [w, hgt], "predicted_max_launch_ms": round(predicted, 2)}
            mode[label] = entry
            if predicted >= MAX_LAUNCH_MS:
                entry["skipped"] = "a launch is predicted at %.0f ms (limit %.0f ms)" % (predicted, MAX_LAUNCH_MS)
                continue
            data, acts, steps, _ = setup(w, hgt)
            net = HipNet(h, mods, data, acts, steps)
            try:
                net.launch()
                h.sync()
                st = net.time_steps(2)
                entry["max_launch_ms_measured"] = round(max(st), 3)
                entry["per_op_ms"] = {op: round(sum(t for t, s in zip(st, steps) if s[4] == op), 3)
                                      for op in ("conv", "upscale", "add")}
                entry["hip_ms"] = round(net.time_network(reps), 3)
                hip_out = net.run()["out"]
            finally:
                net.free()
            name = "timing_%s_%s" % (acc, label)
            files = G.write_files(cdir, name, data)
            written = [f["name"] for f in files.values()] + ["%s.out" % name]
            try:
                r, _ = G.replay(build, cdir, chain_lines(name, files, acts, steps, saves=["out"], time_runs=reps),
                                G.arena_mb(data, acts), timeout=120)
                entry["d3d12_exit"] = r.returncode
                for line in r.stdout.splitlines():
                    if line.startswith("chain %s: time" % name):
                        entry["d3d12_ms"] = float(line.split("time ")[1].split(" ms")[0])
                p = os.path.join(cdir, "%s.out" % name)
                got = open(p, "rb").read() if (r.returncode == 0 and os.path.exists(p)) else b""
                entry["out_hip_equals_d3d12"] = bool(got) and K.compare_raw(got, hip_out, acts["out"][1])["differ"] == 0
            finally:
                G.remove_written(cdir, written)
        res["modes"][acc] = mode
    res["method"] = ("HIP: device events around %d back-to-back network runs, mean per run (per-op: events around each "
                     "launch). D3D12: kernel_replay chain wall time from submission to completion, synchronised after each "
                     "of %d runs, mean per run. Synthetic weights, buffers resident, warm-up first. Information only."
                     % (reps, reps))
    return res


def prepare(build, rocm):
    from clean.l.runner.l_d3d12_check import prepare_replay
    from clean.runner import build as BLD
    from clean.runner.hip import clean_process_environment
    os.makedirs(build, exist_ok=True)
    clean_process_environment(rocm)
    clang = BLD.find_clang(rocm)
    for src, co in (("fg_net1_mma.hip", CO_MMA), ("fg_net1.hip", CO_NET1)):
        BLD.run(BLD.device_command(clang, rocm, os.path.join(ROOT, "clean", "fg", "kernels", src),
                                   os.path.join(build, co)) + ["-I", os.path.join(ROOT, "clean", "kernels")])
    return prepare_replay(build, rocm, KERNELS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.prepare:
        prepare(a.build, a.rocm)
    res = None
    if a.run:
        res = run(a.build, a.rocm, a.model)
        print(json.dumps(res["summary"]))
        for c in res["cases"]:
            print(c["case"], "ok" if c["ok"] else "FAIL", "| hip == d3d12:", c["hip_equals_d3d12"],
                  "| negatives caught:", sum(v["caught"] for v in c["negative_controls"].values()), "of",
                  len(c["negative_controls"]), "| output vs cpu:", json.dumps(c["vs_cpu_output"]))
            for d in c["raw_differences"]:
                print("   raw difference", json.dumps(d))
        print("replay exit", res["replay_exit"], res["replay_tail"][-200:] if res["replay_exit"] else "")
    elif a.timing:
        res = timing(a.build, a.rocm)
        print(json.dumps(res, indent=1))
    if res is not None and a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(res, indent=1) + "\n")
    if a.run:
        return 0 if res["summary"]["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
