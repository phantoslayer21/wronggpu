# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the FP16-form width-32 block kernels (clean/l/kernels/l_q7.hip) against the oracle
(clean/l/reference/oracle.py swin_block with the decoder / encoder block constants, form "q7", FP8 op model "exact"),
byte for byte.

    python clean/l/runner/l_q7_check.py --build <dir> --compile                 # CPU: <dir>/l_q7.co
    python clean/l/runner/l_q7_check.py --build <dir> --prepare [--real <build with lframe_model/params.pkl>]  # CPU
    python clean/l/runner/l_q7_check.py --build <dir> --prepare-d3d12           # CPU: D3D12 ports + kernel_replay.exe
    python clean/l/runner/l_q7_check.py --build <dir> --run [--d3d12] --json out.json   # GPU (inside a GPU slot)

Cases: grid sizes (multiples of 4) x both window offsets of the block's frame parities x input scales; synthetic block
weights (several seeds) and, with --real, the real model's enc0_block / dec0_block. Encoder kernel: input the FP16 halo
map [H + 4][W + 4][32], output the FP8 map. Decoder kernel: input the FP8 projection map and the FP8 skip map, output
the FP16 map. Every output buffer is filled with the sentinel 0xA5 and has 64 slack bytes after it, which must keep the
sentinel; the output must equal the oracle's bytes exactly. Special values: inputs with the largest FP16 / FP8
magnitudes, zeros and a NaN member. Negative control: the same launch with half the workgroups must differ.
With --d3d12 the same cases run on the native D3D12 device (the ports in clean/dx12/kernels/l_q7.hlsl, kernel_replay,
output buffers filled with the D3D12 sentinel 0x5A): raw bytes identical to the HIP run's and to the oracle's, slack
kept, and its own half-grid control.
Output: counts and digests only.
--form hyb / hybmut (Q11 part C, Model L's HYBRID evaluation form; HIP only): l_q7__hyb.co (l_q7.hip Q7_HYB: float32
chains, one FP16 rounding each) against the oracle with FP8 op model "f32chain", raw bytes, the same cases (synthetic
weights) and half-grid control; negative control: the exact (q7) oracle's bytes must differ in some case. hybmut:
l_q7__hybmut.co (W2 rounded after every slice) must differ from the f32chain oracle in some case (the mutant caught).
Prepare once with --form hyb (cases file l_q7_cases_hyb.pkl, used by both).
"""
import copy
import argparse
import ctypes
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.l.runner import l_record as LR  # noqa: E402
from clean.reference.fp8 import f16, fp8_encode, q8  # noqa: E402

SENTINEL, D3D12_SENTINEL, SLACK = 0xA5, 0x5A, 64
KERNELS = ("l_q7_block_c32_enc", "l_q7_block_c32_dec")
GRIDS = [(8, 12), (16, 8), (12, 20), (32, 32)]
SCALES = (0.5, 3.0, 40.0)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_identity(build, files):
    """At run time: this checkout's git commit (git rev-parse HEAD, run here), whether its tracked files differ from
    it, and the SHA-256 of the given files in build."""
    import subprocess

    def git(*a):
        r = subprocess.run(["git", "-C", ROOT] + list(a), capture_output=True, text=True,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        if r.returncode != 0 or (a[0] == "rev-parse" and not r.stdout.strip()):
            raise RuntimeError(f"run identity: git {a[0]} failed ({r.returncode}): {r.stderr.strip()}")
        return r.stdout.strip()
    ident = {"git_commit": git("rev-parse", "HEAD"),
             "git_tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}
    for f in files:
        ident[f.replace(".", "_") + "_sha256"] = _sha256(os.path.join(build, f))
    return ident


def compile_kernel(build, rocm):
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    clang = B.find_clang(rocm)
    os.makedirs(build, exist_ok=True)
    B.run(B.device_command(clang, rocm, os.path.join(ROOT, "clean", "l", "kernels", "l_q7.hip"),
                           os.path.join(build, "l_q7.co")) + ["-I", os.path.join(ROOT, "clean", "kernels")])


class Gpu:
    def __init__(self, build, rocm, co="l_q7.co"):
        from clean.runner.hip import Hip
        self.h = Hip(rocm)
        with open(os.path.join(build, co), "rb") as f:
            self.mod = self.h.load_module(f.read())
        tab = LP.q7_tables()
        self.tab = self.h.malloc(len(tab))
        self.h.upload(self.tab, tab)

    def launch(self, kernel, xb, sb, out_bytes, blob, H, W, off, half=False):
        h = self.h
        bufs = [h.malloc(len(xb)), h.malloc(max(len(sb), 1)), h.malloc(out_bytes + SLACK), h.malloc(len(blob))]
        px, ps, po, pw = bufs
        h.upload(px, xb)
        if sb:
            h.upload(ps, sb)
        h.upload(pw, blob)
        h.memset(po, SENTINEL, out_bytes + SLACK)
        ox, oy = off
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        args = [ctypes.c_void_p(px), ctypes.c_void_p(ps if sb else 0), ctypes.c_void_p(po), ctypes.c_void_p(pw),
                ctypes.c_void_p(self.tab), ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)]
        h.launch(h.get_function(self.mod, kernel), max(1, nwin // 2) if half else nwin, 32, args)
        h.sync()
        out = h.download(po, out_bytes + SLACK)
        for p in bufs:
            h.free(p)
        return out


def out8(v):
    """The kernels' output bytes of FP8 values: a NaN as the positive quiet NaN code 0x7f."""
    b = fp8_encode(q8(v))
    return np.where((b & 0x7F) == 0x7F, np.uint8(0x7F), b).astype(np.uint8).tobytes()


def out16(v):
    """The kernels' output bytes of FP16 values: a NaN as the positive quiet NaN 0x7e00."""
    h = f16(v).astype(np.float16).view(np.uint16)
    return np.where(np.isnan(f16(v)), np.uint16(0x7E00), h).astype(np.uint16).tobytes()


FORM_CO = {"q7": "l_q7.co", "hyb": "l_q7__hyb.co", "hybmut": "l_q7__hybmut.co"}


def form_table(form):
    t = copy.deepcopy(T.TABLE)
    if form in ("hyb", "hybmut"):
        t["mm8"] = "f32chain"
        for name in t["consts"]:
            t["consts"][name] = dict(t["consts"][name])
    return t


def cases_file(build, form):
    return os.path.join(build, "l_q7_cases.pkl" if form == "q7" else "l_q7_cases_hyb.pkl")


def _k(t, name):
    k = T.block_consts(t, name)
    k["mm8"] = t.get("mm8", "exact")
    return k


def enc_case(p, x_halo, off, t=T.TABLE):
    """Oracle: q8 of the encoder block on the halo map (border "input")."""
    k = _k(t, "enc0_block")
    y = L.swin_block(x_halo, p, off, 2, k, "input", 2)
    return out8(y)


def dec_case(p, P, S, off, t=T.TABLE):
    """Oracle: the decoder block on the projection map (border "coarse") with the skip, FP16 output."""
    k = _k(t, "dec0_block")
    y = L.swin_block(P, p, off, 2, k, "coarse", skip=S)
    return out16(y)


def inputs(rng, H, W, scale, special):
    x = f16(rng.normal(0, scale, (H + 4, W + 4, 32)))
    P = q8(rng.normal(0, scale, (H, W, 32)))
    S = q8(rng.normal(0, scale, (H, W, 32)))
    if special:
        x[0, 0, :4] = [65504.0, -65504.0, 0.0, -0.0]
        x[3, 5, 7] = np.nan
        P[1, 1, :3] = [448.0, -448.0, 0.0]
        S[1, 2, :2] = [448.0, 448.0]
        P[2, 3, 4] = np.nan
    return x, P, S


def mask_nan(body, kind):
    """Output bytes with only a NaN's sign bit cleared (the accepted port exception: the D3D12 driver generates NaNs
    with the sign set, HIP without; a canonical NaN in the kernels would cost VALU per store). kind "fp8" or "fp16"."""
    if kind == "fp8":
        b = np.frombuffer(body, np.uint8)
        return np.where((b & 0x7F) == 0x7F, np.uint8(0x7F), b).astype(np.uint8).tobytes()
    h = np.frombuffer(body, np.uint16)
    nan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
    return np.where(nan, h & 0x7FFF, h).astype(np.uint16).tobytes()


def mask_control():
    """Proof that mask_nan clears ONLY a NaN's sign bit: a NaN with the sign flipped compares equal; a non-NaN value
    with the sign flipped, a NaN payload change and a quiet-bit change (FP16) still differ. Every check records it."""
    f8 = lambda a, b: mask_nan(bytes(a), "fp8") == mask_nan(bytes(b), "fp8")       # noqa: E731
    h16 = lambda a, b: (mask_nan(np.array(a, np.uint16).tobytes(), "fp16")         # noqa: E731
                        == mask_nan(np.array(b, np.uint16).tobytes(), "fp16"))
    r = {"fp8_nan_sign_equal": f8([0x7F], [0xFF]),
         "fp8_nonnan_sign_differs": all(not f8([c], [c ^ 0x80]) for c in range(256) if (c & 0x7F) != 0x7F),
         "fp16_nan_sign_equal": h16([0x7E00, 0x7C01], [0xFE00, 0xFC01]),
         "fp16_nonnan_sign_differs": all(not h16([h], [h ^ 0x8000]) for h in range(0x8000)
                                         if not ((h & 0x7C00) == 0x7C00 and (h & 0x3FF))),
         "fp16_nan_payload_differs": not h16([0x7E00], [0x7E01]) and not h16([0x7E01], [0xFE02]),
         "fp16_nan_quiet_bit_differs": not h16([0x7E01], [0x7C01])}
    r["pass"] = all(r.values())
    return r


def control_blobs(blob):
    """(FP8 rows zeroed, every sum |w| table +inf, +inf tables and FP16 rows zeroed) variants of a C32 q7 blob."""
    o = LP.q7_block_offsets(32, 2, 128)
    z8 = blob[:o["Wqk"]] + bytes(o["end8"] - o["Wqk"]) + blob[o["end8"]:]
    inf = np.full((o["end"] - o["end16"]) // 4, np.inf, np.float32).tobytes()
    allfb = blob[:o["end16"]] + inf + blob[o["end"]:]
    allfb_z16 = blob[:o["end8"]] + bytes(o["end16"] - o["end8"]) + inf + blob[o["end"]:]
    return z8, allfb, allfb_z16


def section_controls(g, c, H, W):
    z8, allfb, allfb_z16 = control_blobs(c["blob"])
    run = lambda b: mask_nan(g.launch("l_q7_block_c32_enc", c["xb"], c["sb"], c["nbytes"], b, H, W,   # noqa: E731
                                      tuple(c["offset"]))[:c["nbytes"]], "fp8")
    return {"fp8_section_zeroed_differs": run(z8) != c["want"],
            "all_fallback_identical": run(allfb) == c["want"],
            "all_fallback_fp16_zeroed_differs": run(allfb_z16) != c["want"]}


def compare(got, want, out_bytes, kind="fp8"):
    body, slack = mask_nan(got[:out_bytes], kind), got[out_bytes:]
    return {"identical": body == want, "slack_kept": slack == bytes([SENTINEL]) * SLACK,
            "bytes_differ": sum(a != b for a, b in zip(body, want))}


def prepare(build, real=None, form="q7"):
    """CPU: every case's inputs and the oracle's bytes, saved to <build>/l_q7_cases.pkl (the GPU run only launches)."""
    blocks = [("synthetic_%d" % s, {k: np.asarray(v, np.float64) for k, v in
                                    L.random_block_params(np.random.default_rng(s), 32, 2, 128).items()})
              for s in (89, 90)]
    real_sha = None
    if real:
        real_sha = _sha256(os.path.join(real, "lframe_model", "params.pkl"))
        params = pickle.load(open(os.path.join(real, "lframe_model", "params.pkl"), "rb"))["L"]
        blocks += [("real_enc0_block", params["enc0_block"]), ("real_dec0_block", params["dec0_block"])]
    rng = np.random.default_rng(7)
    cases = []
    for bname, p in blocks:
        blob = LP.pack_q7_block(p, 32, 2, 128)
        for (H, W) in GRIDS:
            for scale in SCALES:
                special = scale == SCALES[1] and (H, W) == GRIDS[0]
                x, P, S = inputs(rng, H, W, scale, special)
                for kind, offs in (("enc", T.TABLE["window_offsets"]["first"]),
                                   ("dec", T.TABLE["window_offsets"]["second"])):
                    for off in offs:
                        tf = form_table(form)
                        if kind == "enc":
                            xb, sb, nbytes = x.astype(np.float16).tobytes(), b"", H * W * 32
                            want = enc_case(p, x, off, tf)
                            want_exact = enc_case(p, x, off) if form != "q7" else None
                        else:
                            xb, sb, nbytes = fp8_encode(P).tobytes(), fp8_encode(S).tobytes(), H * W * 64
                            want = dec_case(p, P, S, off, tf)
                            want_exact = dec_case(p, P, S, off) if form != "q7" else None
                        cases.append({"block": bname, "kernel": kind, "grid": [H, W], "offset": list(off),
                                      "scale": scale, "special": special, "xb": xb, "sb": sb, "nbytes": nbytes,
                                      "blob": blob, "want": want, "want_exact": want_exact})
                print(bname, H, W, scale, "prepared", flush=True)
    with open(cases_file(build, form), "wb") as f:
        pickle.dump({"cases": cases, "real_params_sha256": real_sha, "form": form}, f)
    return len(cases)


def run(build, rocm, d3d12=False, form="q7"):
    """GPU: launch every prepared case and compare with the oracle's bytes (with d3d12, also the D3D12 ports)."""
    with open(cases_file(build, form), "rb") as f:
        prep = pickle.load(f)
    cases = prep["cases"]
    co = FORM_CO[form]
    g = Gpu(build, rocm, co)
    res = {"what": "FP16-form width-32 block kernels against the oracle, raw bytes", "cases": [], "form": form,
           "fp8_op_model": "exact" if form == "q7" else "f32chain",
           "identity": run_identity(build, [co, os.path.basename(cases_file(build, form))]),
           "real_params_sha256": prep["real_params_sha256"],
           "nan_sign_masked": "outputs compared and hashed with only a NaN's sign bit cleared (accepted port exception; "
                              "a canonical NaN store costs about 3 VALU per stored dword in the RGA ISA)",
           "nan_mask_control": mask_control()}
    for c in cases:
        H, W = c["grid"]
        got = g.launch("l_q7_block_c32_" + c["kernel"], c["xb"], c["sb"], c["nbytes"], c["blob"], H, W,
                       tuple(c["offset"]))
        kind = "fp8" if c["kernel"] == "enc" else "fp16"
        r = compare(got, c["want"], c["nbytes"], kind)
        r.update({k: c[k] for k in ("block", "kernel", "grid", "offset", "scale", "special")})
        r["output_sha256"] = hashlib.sha256(mask_nan(got[:c["nbytes"]], kind)).hexdigest()
        if c.get("want_exact") is not None:
            r["differs_from_exact_oracle"] = mask_nan(got[:c["nbytes"]], kind) != c["want_exact"]
        res["cases"].append(r)
        if "negative_control" not in res and c["kernel"] == "enc" and tuple(c["grid"]) == GRIDS[2]:
            half = g.launch("l_q7_block_c32_enc", c["xb"], c["sb"], c["nbytes"], c["blob"], H, W,
                            tuple(c["offset"]), half=True)
            res["negative_control"] = {"case": "%s enc %dx%d" % (c["block"], H, W),
                                       "half_grid_differs": half[:c["nbytes"]] != c["want"]}
            # Q7F-2 section controls: the matrix-instruction path reads the FP8 rows (zeroed: the output differs); with
            # every sum |w| table set to +inf each output takes the exact fallback (FP16 rows), and the output must stay
            # identical; with the FP16 rows zeroed as well that forced fallback must differ
            if form == "q7":   # the matrix-instruction sections exist in the q7 build only
                res["section_control"] = section_controls(g, c, H, W)
    res["cases_identical"] = sum(r["identical"] and r["slack_kept"] for r in res["cases"])
    if form == "hybmut":   # the mutant must be caught: some case differs from the f32chain oracle
        res["mutant_cases_differing"] = len(res["cases"]) - res["cases_identical"]
        res["pass"] = res["mutant_cases_differing"] > 0
        return res
    res["pass"] = (res["cases_identical"] == len(res["cases"]) and res["negative_control"]["half_grid_differs"]
                   and res["nan_mask_control"]["pass"] and all(res.get("section_control", {}).values()))
    if form == "hyb":      # negative control: the exact (q7) oracle's bytes differ somewhere
        res["cases_differing_from_exact_oracle"] = sum(bool(r["differs_from_exact_oracle"]) for r in res["cases"])
        res["pass"] = res["pass"] and res["cases_differing_from_exact_oracle"] > 0
    if d3d12:
        res["d3d12"] = run_d3d12(build, cases, res)
        res["pass"] = res["pass"] and res["d3d12"]["pass"]
    return res


def prepare_d3d12(build, rocm):
    """CPU: the D3D12 ports (clean/dx12/kernels/l_q7.hlsl, build_dx12.py --only) and kernel_replay.exe into build."""
    from clean.l.runner.l_d3d12_check import prepare_replay
    return prepare_replay(build, rocm, list(KERNELS))


def run_d3d12(build, cases, res):
    """GPU: every prepared case on the native D3D12 device (kernel_replay, a child process), output buffers filled with
    the D3D12 sentinel 0x5A; raw bytes against the HIP run's and the oracle's, and the half-grid control."""
    import subprocess
    import time
    cdir = os.path.join(build, "replay-q7-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(cdir)
    with open(os.path.join(cdir, "tab.bin"), "wb") as f:
        f.write(LP.q7_tables())
    blobs, lines, neg = {}, [], None
    for i, c in enumerate(cases):
        name = "c%03d" % i
        if c["block"] not in blobs:
            blobs[c["block"]] = "blob_%s.bin" % c["block"]
            with open(os.path.join(cdir, blobs[c["block"]]), "wb") as f:
                f.write(c["blob"])
        with open(os.path.join(cdir, name + ".x"), "wb") as f:
            f.write(c["xb"])
        if c["sb"]:
            with open(os.path.join(cdir, name + ".s"), "wb") as f:
                f.write(c["sb"])
        H, W = c["grid"]
        ox, oy = c["offset"]
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        runs = [(name, nwin, blobs[c["block"]])]
        if neg is None and c["kernel"] == "enc" and tuple(c["grid"]) == GRIDS[2]:
            neg = i
            runs.append(("negative_" + name, max(1, nwin // 2), blobs[c["block"]]))
            for tag, vb in zip(("z8", "allfb", "allfbz16"), control_blobs(c["blob"])):
                with open(os.path.join(cdir, "blob_%s.bin" % tag), "wb") as f:
                    f.write(vb)
                runs.append(("%s_%s" % (tag, name), nwin, "blob_%s.bin" % tag))
        for cname, grid, bname in runs:
            lines += ["case " + cname, "module l_q7.co", "kernel l_q7_block_c32_%s %d 32" % (c["kernel"], grid),
                      "arg buf %s.x" % name, ("arg buf %s.s" % name) if c["sb"] else "arg null",
                      "arg out %d %s.o %d" % (c["nbytes"] + SLACK, cname, D3D12_SENTINEL),
                      "arg buf " + bname, "arg buf tab.bin",
                      "arg u32 %d" % H, "arg u32 %d" % W, "arg u32 %d" % ox, "arg u32 %d" % oy, ""]
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() == "HIP_PLATFORM"}
    env["WG_D3D12_ARENA_MB"] = "256"
    r = subprocess.run([os.path.join(build, "kernel_replay.exe"), build, os.path.join(cdir, "cases.txt")], env=env,
                       capture_output=True, text=True, timeout=100, creationflags=0x08000000)

    def read(n):
        p = os.path.join(cdir, n + ".o")
        return open(p, "rb").read() if os.path.exists(p) else b""

    out = {"replay_exit": r.returncode, "replay_tail": (r.stdout[-300:] + r.stderr[-300:]).strip(),
           "identity": run_identity(build, ["kernel_replay.exe", "dx12/l_q7_block_c32_enc.dxil",
                                            "dx12/l_q7_block_c32_dec.dxil"])}
    same_hip = same_oracle = 0
    for i, (c, rc) in enumerate(zip(cases, res["cases"])):
        got = read("c%03d" % i)
        body = mask_nan(got[:c["nbytes"]], "fp8" if c["kernel"] == "enc" else "fp16") if got else b""
        slack = got[c["nbytes"]:]
        ok_slack = slack == bytes([D3D12_SENTINEL]) * SLACK
        rc["d3d12_identical_to_oracle"] = body == c["want"] and ok_slack
        rc["d3d12_output_sha256"] = hashlib.sha256(body).hexdigest()
        rc["d3d12_bytes_differ_from_oracle"] = sum(a != b for a, b in zip(body, c["want"])) if body else -1
        same_oracle += rc["d3d12_identical_to_oracle"]
        same_hip += rc["d3d12_output_sha256"] == rc["output_sha256"] and ok_slack
    c = cases[neg]
    half = read("negative_c%03d" % neg)[:c["nbytes"]]
    ctl = {t: mask_nan(read("%s_c%03d" % (t, neg))[:c["nbytes"]], "fp8") for t in ("z8", "allfb", "allfbz16")}
    out.update({"cases": len(cases), "cases_identical_to_hip": same_hip, "cases_identical_to_oracle": same_oracle,
                "negative_control": {"case": "%s enc %dx%d" % (c["block"], *c["grid"]),
                                     "half_grid_differs": half != c["want"]},
                "section_control": {"fp8_section_zeroed_differs": ctl["z8"] != c["want"],
                                    "all_fallback_identical": ctl["allfb"] == c["want"],
                                    "all_fallback_fp16_zeroed_differs": ctl["allfbz16"] != c["want"]}})
    out["pass"] = (r.returncode == 0 and same_hip == len(cases) and same_oracle == len(cases)
                   and out["negative_control"]["half_grid_differs"] and all(out["section_control"].values()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--compile", action="store_true")
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--prepare-d3d12", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--d3d12", action="store_true", help="with --run: also the D3D12 ports through kernel_replay")
    ap.add_argument("--real")
    ap.add_argument("--form", choices=sorted(FORM_CO), default="q7")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.compile:
        compile_kernel(a.build, a.rocm)
        return 0
    if a.prepare_d3d12:
        print("built", prepare_d3d12(a.build, a.rocm))
        return 0
    if a.prepare:
        if a.form == "hybmut":
            ap.error("prepare with --form hyb (its cases serve hybmut too)")
        print("cases", prepare(a.build, a.real, a.form))
        return 0
    from clean.runner.hip import clean_process_environment
    clean_process_environment(a.rocm)
    if a.form != "q7" and a.d3d12:
        ap.error("--form hyb / hybmut: HIP only")
    res = run(a.build, a.rocm, a.d3d12, a.form)
    print("form", a.form, "| HIP cases identical", res["cases_identical"], "of", len(res["cases"]),
          "| negative control caught", res["negative_control"]["half_grid_differs"],
          "| differing from exact oracle", res.get("cases_differing_from_exact_oracle"),
          "| mutant cases differing", res.get("mutant_cases_differing"))
    if a.d3d12:
        d = res["d3d12"]
        print("D3D12 replay exit", d["replay_exit"], "| identical to HIP", d["cases_identical_to_hip"],
              "| to the oracle", d["cases_identical_to_oracle"], "of", d["cases"], "| negative control caught",
              d["negative_control"]["half_grid_differs"])
        if d["replay_exit"]:
            print(d["replay_tail"])
    print("PASS" if res["pass"] else "FAIL")
    if a.json:
        LR.write_l_record(res, a.json)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
