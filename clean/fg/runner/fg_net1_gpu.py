# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): net1's 32 steps (clean/fg/reference/net1.py NET1_WIRING) chained on the GPU, on HIP
and on the native D3D12 device, against the CPU reference bit for bit.

    python clean/fg/runner/fg_net1_gpu.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/fg/runner/fg_net1_gpu.py --build <dir> --rocm ... --run [--model <model.wgcm>] [--json out.json]
    python clean/fg/runner/fg_net1_gpu.py --build <dir> --rocm ... --timing [--json out.json]

--prepare (CPU) is fg_d3d12_check.py's: fg_net1.co, the four D3D12 ports (build_dx12.py --only) and kernel_replay.exe.
--run (GPU, one GPU slot): every case runs the whole network on HIP in this process (all buffers on the device, one
launch per step, no host round trip) and through one kernel_replay "chain" per case on D3D12, then:
- every one of the 32 step outputs is compared as RAW bytes between the back ends; each starts filled with its back
  end's sentinel (HIP 0xA5, D3D12 0x5A) and has 64 slack bytes after it that no step may write;
- every step output on HIP is compared with the CPU reference's buffer (float16 bits; NaN equals NaN);
- negative controls: per kernel, a D3D12 chain whose first step of that kernel (with 4 or more workgroups) runs on half
  of its workgroups must differ in that step's output;
- the network output (unfolded to [4, G, G]) gets a SHA-256 on both sides.
Cases: synthetic weights on 64 x 64 pixels (the smallest input with a cell on every level), on 70 x 90 pixels (odd
level sizes: floor halving), and with special values (NaN, +-inf, subnormals, +-65504, -0) in the input and biases;
with --model, also the weights of a converter output file of the preset-FG map on clean/fg/tools/net1_digest.py's fixed
input. Weights from --model are execute-only: the per-case files this tool writes for kernel_replay (input, weights,
outputs) are deleted after the run, and only counts and digests are printed.
--timing (GPU, one GPU slot; information only): net1 on synthetic weights at 1920 x 1080 and 2560 x 1440 input pixels
(960 x 540 and 1280 x 720 cells), mean milliseconds per network run on each back end. A probe at 512 x 512 pixels times
every step first; the large sizes run only when every single launch, scaled by the cell count, stays under
MAX_LAUNCH_MS (a long single dispatch could trip the display driver's timeout).
"""
import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from clean.fg import layout as L  # noqa: E402
from clean.fg.reference import net1 as N  # noqa: E402
import fg_d3d12_check as K  # noqa: E402

KERNELS = ["fg_conv_f16", "fg_maxpool2_f16", "fg_upscale_f16", "fg_add_f16"]
KERNEL_OF = {"conv": "fg_conv_f16", "pool": "fg_maxpool2_f16", "upscale": "fg_upscale_f16", "add": "fg_add_f16"}
SPECIALS = K.SPECIALS
FINITE_SPECIALS = np.array([65504.0, -65504.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -20, -0.0, 0.0], dtype=np.float16)
MAX_LAUNCH_MS = 250.0
TIMING_SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440)}
PROBE = (512, 512)


# ---- the network as buffers and launches ------------------------------------------------------------------------------

def synthetic_params(rng):
    """{layer: (canonical weights [out, k, k, in], bias)} float16, scaled so that values stay in range."""
    out = {}
    for l in L.NET1:
        k = 3 if l.taps == 9 else 1
        s = 1.5 / np.sqrt(l.cin * k * k)
        out[l.name] = (rng.uniform(-s, s, (l.out, k, k, l.cin)).astype(np.float16),
                       rng.uniform(-0.1, 0.1, l.out).astype(np.float16))
    return out


def plan(H0, W0, params):
    """Buffers and launches of net1 on cells [48, H0, W0].

    Returns (data, acts, steps): data {id: bytes} (cells, weights natural [out][in][k][k], biases), acts {id: (shape,
    defined bytes)} the step outputs, steps [(kernel, grid, args, out id, op)] with args "@id" or integers."""
    data, acts, steps = {}, {}, []
    shape = {"cells": (N.CELL, H0, W0)}
    for s in N.NET1_WIRING:
        op = s["op"]
        if op == "conv":
            w, b = params[s["layer"]]
            cout, k, _, cin = w.shape
            wid, bid = "w_" + s["layer"], "b_" + s["layer"]
            data[wid] = np.ascontiguousarray(np.asarray(w, np.float16).transpose(0, 3, 1, 2)).tobytes()
            data[bid] = np.asarray(b, np.float16).tobytes()
            C, H, W = shape[s["in"]]
            if C != cin:
                raise ValueError("%s: %d input channels, buffer %s has %d" % (s["layer"], cin, s["in"], C))
            out = (cout, H, W)
            args = ["@" + s["in"], "@" + wid, "@" + bid, "@" + s["out"], cin, cout, H, W, k,
                    K.ORDER_CODE[s["chunks"]], 1 if s["act"] == "leaky" else 0]
        elif op == "pool":
            C, H, W = shape[s["in"]]
            out = (C, H // 2, W // 2)
            args = ["@" + s["in"], "@" + s["out"], C, H, W]
        elif op == "upscale":
            C, hi, wi = shape[s["in"]]
            _, ho, wo = shape[s["like"]]
            out = (C, ho, wo)
            args = ["@" + s["in"], "@" + s["out"], C, hi, wi, ho, wo, K._ratio_bits(wi, wo), K._ratio_bits(hi, ho)]
        elif op == "add":
            a, b = s["in"]
            if shape[a] != shape[b]:
                raise ValueError("add of %s %s and %s %s" % (a, shape[a], b, shape[b]))
            out = shape[a]
            args = ["@" + a, "@" + b, "@" + s["out"], int(np.prod(out))]
        else:
            raise ValueError("unknown step %r" % op)
        shape[s["out"]] = out
        n = int(np.prod(out))
        acts[s["out"]] = (out, 2 * n)
        steps.append((KERNEL_OF[op], -(-n // 256), args, s["out"], op))
    return data, acts, steps


def cells_bytes(cells):
    return np.asarray(cells, np.float64).astype(np.float16).tobytes()


def output_digest(out_bytes, H0, W0, gh, gw):
    """SHA-256 of the unfolded output [4, gh, gw] (float16 bytes as written), the same with every NaN as 0x7E00 (the
    reference's NaN payload is not specified), and the number of non-finite values."""
    o = np.frombuffer(out_bytes, np.float16).astype(np.float64).reshape(16, H0, W0)
    return pixel_digest(N.unfold(o, gh, gw))


def pixel_digest(px):
    px = np.asarray(px, np.float64).astype("<f2")
    raw = hashlib.sha256(px.tobytes()).hexdigest()
    bits = px.view("<u2").copy()
    nan = np.isnan(px)
    bits[nan] = 0x7E00
    return raw, hashlib.sha256(bits.tobytes()).hexdigest(), int(np.count_nonzero(~np.isfinite(px)))


# ---- HIP --------------------------------------------------------------------------------------------------------------

class HipNet:
    """One network's buffers on the HIP device; launch() runs the steps (optionally timed)."""

    def __init__(self, h, mod, data, acts, steps):
        self.h, self.steps, self.acts = h, steps, acts
        self.fn = {k: h.get_function(mod, k) for k in KERNELS}
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

    def time_steps(self, reps):
        """Mean device milliseconds of every launch over `reps` runs (events around each launch)."""
        ms = [0.0] * len(self.steps)
        for _ in range(reps):
            for i in range(len(self.steps)):
                e0, e1 = self.h.event(), self.h.event()
                self.h.record(e0)
                self.launch(only=i)
                self.h.record(e1)
                ms[i] += self.h.elapsed_ms(e0, e1)
                self.h.destroy_event(e0)
                self.h.destroy_event(e1)
        return [m / reps for m in ms]

    def time_network(self, reps):
        """Mean device milliseconds per whole network run (events around `reps` back-to-back runs)."""
        e0, e1 = self.h.event(), self.h.event()
        self.h.record(e0)
        for _ in range(reps):
            self.launch()
        self.h.record(e1)
        ms = self.h.elapsed_ms(e0, e1) / reps
        self.h.destroy_event(e0)
        self.h.destroy_event(e1)
        return ms

    def free(self):
        for p in self.ptr.values():
            self.h.free(p)
        self.ptr = {}


# ---- D3D12 chains -----------------------------------------------------------------------------------------------------

def chain_lines(name, files, acts, steps, half=None, stop=None, saves=None, time_runs=0):
    """A kernel_replay chain: data buffers from `files` {id: file}, step outputs filled with the D3D12 sentinel (and
    the slack). half: the index of a step to launch on half of its workgroups; stop: the last step to run."""
    lines = ["chain " + name, "module " + K.CO]
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


def write_files(cdir, name, data):
    files = {}
    for i, b in data.items():
        fn = "%s.%s" % (name, i)
        path = os.path.join(cdir, fn)
        with open(path, "wb") as f:
            f.write(b)
        files[i] = {"name": fn, "path": path}
    return files


def replay(build, cdir, lines, arena_mb, timeout):
    path = os.path.join(cdir, "chains.txt")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = str(arena_mb)
    t0 = time.time()
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, path], env=env, capture_output=True,
                       text=True, timeout=timeout, creationflags=0x08000000)
    return r, time.time() - t0


def remove_written(cdir, written):
    """Deletes the files this tool wrote into cdir (one by one; never the folder)."""
    for fn in written:
        p = os.path.join(cdir, fn)
        if os.path.isfile(p):
            os.remove(p)


def arena_mb(data, acts):
    total = sum(len(b) + 4096 for b in data.values()) + sum(n + K.SLACK + 4096 for _, n in acts.values())
    return int(min(4000, max(256, 1.5 * total / 2 ** 20 + 64)))


# ---- run: correctness -------------------------------------------------------------------------------------------------

def cases(model):
    rng = np.random.default_rng(20261114)
    out = []
    p = synthetic_params(rng)
    v = rng.uniform(0, 1, (10, 64, 64))
    out.append(("synthetic_64", v, p))
    v = rng.uniform(-1, 1, (10, 70, 90))
    out.append(("synthetic_70x90", v, synthetic_params(rng)))
    v = rng.uniform(0, 1, (10, 64, 64)).astype(np.float16)
    f = rng.random(v.shape) < 0.003
    v[f] = rng.choice(SPECIALS, int(f.sum()))
    ps = synthetic_params(rng)
    w, b = ps["net1_06"]
    b = b.copy()
    b[0], b[1], b[2], b[3] = np.inf, -np.inf, np.nan, 65504.0
    ps["net1_06"] = (w, b)
    w, b = ps["net1_02"]
    w = w.copy()
    w[3, 0, 0, :] = 200.0                       # sums that overflow to +inf
    w[4, 0, 0, :] = -200.0
    ps["net1_02"] = (w, b)
    out.append(("specials_64", v.astype(np.float64), ps))
    # finite specials only (a NaN anywhere reaches every output through the coarse levels)
    v = rng.uniform(0, 1, (10, 64, 64)).astype(np.float16)
    f = rng.random(v.shape) < 0.01
    v[f] = rng.choice(FINITE_SPECIALS, int(f.sum()))
    out.append(("finite_specials_64", v.astype(np.float64), synthetic_params(rng)))
    if model:
        from clean.fg.reference import forward as F
        from clean.fg.tools import net1_digest as D
        out.append(("model", D.fixed_input(), F.params_from_model_file(model)))
    return out


def run(build, rocm, model):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    with open(os.path.join(build, K.CO), "rb") as f:
        mod = h.load_module(f.read())
    cdir = os.path.join(build, "replay-net1")
    os.makedirs(cdir, exist_ok=True)
    lines, runs, written, need_mb = [], [], [], 256
    try:
        for name, values, params in cases(model):
            t0 = time.time()
            cells = N.fold(values)
            _, gh, gw = values.shape
            H0, W0 = cells.shape[1:]
            data, acts, steps = plan(H0, W0, params)
            data = dict(cells=cells_bytes(cells), **data)
            net = HipNet(h, mod, data, acts, steps)
            try:
                hip = net.run()
            finally:
                net.free()
            ref = N.run(cells, params)
            files = write_files(cdir, name, data)
            written += [f["name"] for f in files.values()] + ["%s.%s" % (name, i) for i in acts]
            lines += chain_lines(name, files, acts, steps)
            negs = {}
            for kernel in KERNELS:
                j = next(j for j, s in enumerate(steps) if s[0] == kernel and s[1] >= 4)
                nn = "negative_%s_%s" % (name, kernel)
                negs[kernel] = (nn, j, steps[j][3])
                lines += chain_lines(nn, files, acts, steps, half=j, stop=j, saves=[steps[j][3]])
                written.append("%s.%s" % (nn, steps[j][3]))
            need_mb = max(need_mb, arena_mb(data, acts))
            runs.append({"name": name, "acts": acts, "steps": steps, "hip": hip, "ref": ref, "negs": negs,
                         "dims": (H0, W0, gh, gw), "cpu_s": time.time() - t0})
        r, secs = replay(build, cdir, lines, need_mb, timeout=120)

        def read(fn):
            p = os.path.join(cdir, fn)
            return open(p, "rb").read() if os.path.exists(p) else b""

        res = {"replay_exit": r.returncode, "replay_s": round(secs, 1), "replay_tail": r.stdout[-600:] + r.stderr[-300:],
               "cases": []}
        ok = r.returncode == 0
        for c in runs:
            rows, bad = [], 0
            for j, (kernel, grid, _, out, op) in enumerate(c["steps"]):
                _, n = c["acts"][out]
                raw = K.compare_raw(read("%s.%s" % (c["name"], out)), c["hip"][out], n)
                cpu = K.compare_cpu(c["hip"][out][:n], c["ref"][out])
                good = raw["differ"] == 0 and raw["slack_ok"] and cpu["differ"] == 0
                bad += not good
                rows.append({"step": j, "op": op, "out": out, "values": cpu["values"], "raw_differ": raw["differ"],
                             "slack_ok": raw["slack_ok"], "cpu_differ": cpu["differ"], "nan": cpu["nan"],
                             "inf": cpu["inf"], "ok": good})
            H0, W0, gh, gw = c["dims"]
            n_out = c["acts"]["out"][1]
            gpu_sha, gpu_can, nonfinite = output_digest(c["hip"]["out"][:n_out], H0, W0, gh, gw)
            d3d_sha, _, _ = output_digest(read("%s.out" % c["name"])[:n_out] or b"\0" * n_out, H0, W0, gh, gw)
            cpu_sha, cpu_can, _ = pixel_digest(N.unfold(c["ref"]["out"], gh, gw))
            negs = {}
            for kernel, (nn, j, out) in c["negs"].items():
                _, n = c["acts"][out]
                nraw = K.compare_raw(read("%s.%s" % (nn, out)), c["hip"][out], n)
                negs[kernel] = {"step": j, "out": out, "differ": nraw["differ"],
                                "caught": nraw["differ"] != 0 or not nraw["slack_ok"]}
            # HIP and D3D12: the same bytes; the CPU reference: the same with NaN payloads made equal
            good = bad == 0 and gpu_sha == d3d_sha and gpu_can == cpu_can and all(v["caught"] for v in negs.values())
            ok = ok and good
            res["cases"].append({"case": c["name"], "input_pixels": [gw, gh], "cells": [W0, H0], "steps": rows,
                                 "steps_ok": len(rows) - bad, "values": sum(x["values"] for x in rows),
                                 "nan": sum(x["nan"] for x in rows), "inf": sum(x["inf"] for x in rows),
                                 "output_sha256": {"hip": gpu_sha, "d3d12": d3d_sha, "cpu": cpu_sha},
                                 "output_sha256_nan_canonical": {"hip": gpu_can, "cpu": cpu_can},
                                 "output_nonfinite": nonfinite, "negative_controls": negs, "cpu_s": round(c["cpu_s"], 1),
                                 "ok": good})
        res["summary"] = {"cases": len(res["cases"]), "cases_ok": sum(c["ok"] for c in res["cases"]),
                          "step_outputs": sum(len(c["steps"]) for c in res["cases"]),
                          "values": sum(c["values"] for c in res["cases"]),
                          "nan_values": sum(c["nan"] for c in res["cases"]),
                          "inf_values": sum(c["inf"] for c in res["cases"]), "pass": bool(ok)}
        return res
    finally:
        remove_written(cdir, written)


# ---- timing (information only) ----------------------------------------------------------------------------------------

def timing(build, rocm, reps=5):
    from clean.runner.hip import Hip, clean_process_environment
    clean_process_environment(rocm)
    h = Hip(rocm)
    with open(os.path.join(build, K.CO), "rb") as f:
        mod = h.load_module(f.read())
    rng = np.random.default_rng(7)
    params = synthetic_params(rng)
    cdir = os.path.join(build, "replay-net1")
    os.makedirs(cdir, exist_ok=True)
    res = {"reps": reps, "max_launch_ms": MAX_LAUNCH_MS}

    def setup(w, hgt):
        H0, W0 = -(-hgt // 2), -(-w // 2)
        cells = rng.uniform(0, 1, (N.CELL, H0, W0))
        cells[4 * N.VALUES:] = 0
        data, acts, steps = plan(H0, W0, params)
        return dict(cells=cells_bytes(cells), **data), acts, steps, H0 * W0

    data, acts, steps, probe_cells = setup(*PROBE)
    net = HipNet(h, mod, data, acts, steps)
    try:
        net.launch()
        h.sync()                                   # warm-up
        per = net.time_steps(2)
    finally:
        net.free()
    worst = int(np.argmax(per))
    res["probe"] = {"pixels": list(PROBE), "max_step_ms": round(per[worst], 3), "max_step": worst,
                    "network_ms_sum_of_steps": round(sum(per), 3)}
    sizes = {}
    for label, (w, hgt) in TIMING_SIZES.items():
        cells_n = (-(-hgt // 2)) * (-(-w // 2))
        predicted = per[worst] * cells_n / probe_cells
        entry = {"pixels": [w, hgt], "predicted_max_launch_ms": round(predicted, 1)}
        sizes[label] = entry
        if predicted >= MAX_LAUNCH_MS:
            entry["skipped"] = "a single launch is predicted at %.0f ms (limit %.0f ms)" % (predicted, MAX_LAUNCH_MS)
            continue
        data, acts, steps, _ = setup(w, hgt)
        net = HipNet(h, mod, data, acts, steps)
        try:
            net.launch()
            h.sync()                               # warm-up
            entry["max_launch_ms_measured"] = round(max(net.time_steps(1)), 2)
            entry["hip_ms"] = round(net.time_network(reps), 2)
            hip_out = net.run()["out"]
        finally:
            net.free()
        name = "timing_" + label
        files = write_files(cdir, name, data)
        written = [f["name"] for f in files.values()] + ["%s.out" % name]
        try:
            r, secs = replay(build, cdir, chain_lines(name, files, acts, steps, saves=["out"], time_runs=reps),
                             arena_mb(data, acts), timeout=120)
            tail = r.stdout
            entry["d3d12_exit"] = r.returncode
            for line in tail.splitlines():
                if line.startswith("chain %s: time" % name):
                    entry["d3d12_ms"] = float(line.split("time ")[1].split(" ms")[0])
            got = open(os.path.join(cdir, "%s.out" % name), "rb").read() if r.returncode == 0 else b""
            n = acts["out"][1]
            entry["out_raw_equal"] = K.compare_raw(got, hip_out, n)["differ"] == 0 if got else False
            if r.returncode != 0:
                entry["d3d12_tail"] = tail[-400:] + r.stderr[-200:]
        finally:
            remove_written(cdir, written)
    res["sizes"] = sizes
    res["method"] = ("HIP: device events around %d back-to-back network runs, mean per run. D3D12: kernel_replay chain "
                     "wall time from submission to completion, synchronised after each of %d runs, mean per run. "
                     "Synthetic weights, buffers resident, warm-up run first. Information only." % (reps, reps))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--model", default=None, help="a converter output file of the preset-FG map (execute-only)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.prepare:
        K.prepare(a.build, a.rocm, KERNELS)
    res = None
    if a.run:
        res = run(a.build, a.rocm, a.model)
        print(json.dumps(res["summary"]))
        for c in res["cases"]:
            print(c["case"], "ok" if c["ok"] else "FAIL", "steps_ok %d/%d" % (c["steps_ok"], len(c["steps"])),
                  "negative controls caught:", sum(v["caught"] for v in c["negative_controls"].values()),
                  "output hip == d3d12:", c["output_sha256"]["hip"] == c["output_sha256"]["d3d12"],
                  "== cpu (NaN canonical):", len(set(c["output_sha256_nan_canonical"].values())) == 1,
                  "nonfinite outputs:", c["output_nonfinite"])
            for s in c["steps"]:
                if not s["ok"]:
                    print("  FAIL step", json.dumps(s))
        print("replay exit", res["replay_exit"])
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
