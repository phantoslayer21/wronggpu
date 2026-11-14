# SPDX-License-Identifier: MIT
"""Model L (issue #89): GPU check of the C = 32 Swin block kernel (clean/l/kernels/l_swin.hip), synthetic data only.

    python clean/l/runner/l_c32_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" [--compile]
        [--with-swin-co] --json out.json

--compile builds <dir>/l_swin.co (and with --with-swin-co Model M's swin.co from clean/kernels/swin.hip) on the CPU.
The GPU part runs in a GPU slot. Output: counts only, no timing.

Checks:
1. l_swin_block_c32 against the oracle (clean/l/reference/oracle.py swin_block with the kernels' default sigmoid
   GELU), per case: no non-finite value, at most 2 % of values more than one FP8 step off, mean absolute error at most
   1 % of the reference RMS (Model M's per-stage rule, clean/tests/test_kernels_emu.py), and the share of values that
   differ at all.
2. l_swin_block_c32_nc against l_swin_block_c32 where the weights' flags allow it: byte-identical.
3. With swin.co: l_swin_block_c64 (this file's build of the template) against Model M's swin_block_c64: byte-identical.
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

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_pack as LP  # noqa: E402
from clean.reference import swin as MS  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402
from clean.runner import packing as P  # noqa: E402

CASES = [(9, 13, (2, 2)), (16, 8, (0, 2)), (30, 54, (0, 0)), (31, 17, (2, 0)), (48, 64, (2, 2)), (5, 7, (0, 0))]


def compile_kernels(build, rocm, with_swin=False, clang=None):
    from clean.runner import build as B
    from clean.runner.hip import clean_process_environment
    clean_process_environment(rocm)
    clang = clang or B.find_clang(rocm)
    kdir = os.path.join(ROOT, "clean", "kernels")
    B.run(B.device_command(clang, rocm, os.path.join(ROOT, "clean", "l", "kernels", "l_swin.hip"),
                           os.path.join(build, "l_swin.co")) + ["-I", kdir])
    if with_swin:
        B.run(B.device_command(clang, rocm, os.path.join(kdir, "swin.hip"), os.path.join(build, "swin.co")))


def ulp_distance(a, b):
    ca, cb = fp8_encode(a).astype(int), fp8_encode(b).astype(int)
    oa = np.where(ca & 0x80, -(ca & 0x7F), ca)
    ob = np.where(cb & 0x80, -(cb & 0x7F), cb)
    return np.abs(oa - ob)


def metrics(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    finite = np.isfinite(got)
    d = np.abs(np.where(finite, got, 0.0) - ref)
    u = ulp_distance(np.where(finite, got, 0.0), ref)
    return {"nonfinite": int(np.sum(~finite)), "frac_over_1_step": float(np.mean(u > 1)),
            "frac_differ": float(np.mean(u > 0)), "values": int(u.size), "differ": int(np.sum(u > 0)),
            "max_steps": int(u.max()) if u.size else 0, "mean_abs_err": float(d.mean()),
            "ref_rms": float(np.sqrt(np.mean(ref * ref)))}


def passes(m):
    return m["nonfinite"] == 0 and m["frac_over_1_step"] <= 0.02 and m["mean_abs_err"] <= 0.01 * m["ref_rms"]


class Gpu:
    def __init__(self, build, rocm):
        from clean.runner.hip import Hip
        self.hip = Hip(rocm)
        self.mods = {}
        for co in ("l_swin.co", "swin.co"):
            path = os.path.join(build, co)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.mods[co] = self.hip.load_module(f.read())

    def block(self, co, kernel, x, blob, offset, C):
        """x: FP8-exact [H, W, C] -> kernel output [H, W, C] (float32 values)."""
        h = self.hip
        H, W = x.shape[:2]
        xb = fp8_encode(x).tobytes()
        px, pw, po = h.malloc(len(xb)), h.malloc(len(blob)), h.malloc(len(xb))
        h.upload(px, xb)
        h.upload(pw, blob)
        h.memset(po, 0, len(xb))
        ox, oy = offset
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        args = [ctypes.c_void_p(px), ctypes.c_void_p(0), ctypes.c_void_p(po), ctypes.c_void_p(pw),
                ctypes.c_int(H), ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)]
        h.launch(h.get_function(self.mods[co], kernel), -(-nwin // 8), 256, args)
        h.sync()
        out = np.frombuffer(h.download(po, len(xb)), dtype=np.uint8).reshape(H, W, C)
        for p in (px, pw, po):
            h.free(p)
        return fp8_decode(out)


CONSTS = dict(T.M_BLOCK_CONSTS, gelu="sigmoid")


def make_case(rng, H, W):
    """Synthetic block parameters, their blob and an FP8 input map [H, W, 32] (CPU only)."""
    p = L.random_block_params(rng, 32, 2, 128)
    return p, LP.pack_block(p, 32, 2, 128), q8(rng.normal(0, 1.5, (H, W, 32)))


def run(build, rocm, seeds=(89,)):
    g = Gpu(build, rocm)
    rngs = [np.random.default_rng(s) for s in seeds]
    res = {"c32": [], "c32_nc_identical": None, "c64_template_identical": None}
    nc_same = []
    for rng, (H, W, off) in ((r, c) for r in rngs for c in CASES):
        p, blob, x = make_case(rng, H, W)
        got = g.block("l_swin.co", "l_swin_block_c32", x, blob, off, 32)
        ref = q8(L.swin_block(x, p, off, 2, CONSTS))
        m = metrics(got, ref)
        m.update({"case": [H, W, list(off)], "pass": passes(m)})
        res["c32"].append(m)
        safe = bool(np.frombuffer(blob[:4], np.uint32)[0] & 1)
        if safe:
            nc = g.block("l_swin.co", "l_swin_block_c32_nc", x, blob, off, 32)
            nc_same.append(nc.tobytes() == got.tobytes())
    res["c32_nc_cases"] = len(nc_same)
    res["c32_nc_identical"] = all(nc_same) if nc_same else None
    if "swin.co" in g.mods:
        same = []
        for rng, (H, W, off) in ((r, c) for r in rngs for c in CASES):
            p = MS.random_params("enc1", rng)
            blob = P.pack_stage("enc1", p)
            x = q8(rng.normal(0, 1.5, (H, W, 64)))
            a = g.block("swin.co", "swin_block_c64", x, blob, off, 64)
            b = g.block("l_swin.co", "l_swin_block_c64", x, blob, off, 64)
            same.append(a.tobytes() == b.tobytes())
        res["c64_template_identical"] = all(same)
        res["c64_cases"] = len(same)
    res["c32_total"] = {"cases": len(res["c32"]), "values": sum(m["values"] for m in res["c32"]),
                        "differ": sum(m["differ"] for m in res["c32"]),
                        "max_steps": max(m["max_steps"] for m in res["c32"])}
    res["pass"] = (all(m["pass"] for m in res["c32"]) and res["c32_nc_identical"] in (True, None)
                   and res["c64_template_identical"] in (True, None))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--clang")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--with-swin-co", action="store_true")
    ap.add_argument("--seeds", type=int, default=1, help="run the case list with this many seeds (89, 90, ...)")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.build, exist_ok=True)
    if a.compile or a.compile_only:
        compile_kernels(a.build, a.rocm, a.with_swin_co, a.clang)
        if a.compile_only:
            return 0
    res = run(a.build, a.rocm, tuple(range(89, 89 + a.seeds)))
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    for m in res["c32"]:
        print(json.dumps(m))
    print({k: v for k, v in res.items() if k != "c32"})
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
