# SPDX-License-Identifier: GPL-3.0-only
"""Kernel-level check of the HYBRID form's shared-stage kernels (S3: clean/dx12/kernels/hyb_swin.hlsl, hyb_gemm.hlsl;
HIP twins clean/kernels/swin_hyb.hip -> swin_hyb.co): random parameters in the hybrid values (l_hyb_pack.py), FP8
input maps with special codes in every other case, both window offsets of each table kind.

  1. HIP (swin_hyb.co) == D3D12 raw bytes for every case; each output starts with its own sentinel; a half-grid D3D12
     run of the first case of every kernel is the negative control (must differ).
  2. Informational: the output's FP8 codes against the oracle's hybrid form (form q7, mm8 f32chain): blocks by
     oracle.swin_block (encoder / bottleneck border "mirror"; decoder E by "coarse", the skip map by the fine mirror),
     merges FP8(clamp(mm8)), expands FP8(mm8): counts of differing codes (both NaN = same). The per-stage check
     (SW_HYB_DUMP, the written tolerance) gives the pass / fail; a gross count here means a wrong index or mirror.

    python clean/l/runner/l_hyb_shared_check.py --build <dir> [--prepare] [--run] [--seeds N] [--kernels a,b]
        [--json out.json] [--no-oracle]
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

from clean.l.reference import oracle as O  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_d3d12_check as LD  # noqa: E402
from clean.l.runner.l_hyb_pack import pack_block_hyb, pack_expand_hyb, pack_merge_hyb  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402

# name: (kind, C, heads, mlp, extra) - kind block / add / merge / expand; extra = C_out (merge) or C_low (expand)
KERNELS = {
    "swin_block_c64_hyb": ("block", 64, 2, 256, None),
    "swin_block_c64_add_hyb": ("add", 64, 2, 256, None),
    "swin_block_c96_hyb": ("block", 96, 4, 384, None),
    "swin_block_c96_add_hyb": ("add", 96, 4, 384, None),
    "swin_block_c128_hyb": ("block", 128, 4, 512, None),
    "patch_merge_c64_hyb": ("merge", 64, 0, 0, 96),
    "patch_merge_c96_hyb": ("merge", 96, 0, 0, 128),
    "expand_c96_hyb": ("expand", 64, 0, 0, 96),
    "expand_c128_hyb": ("expand", 96, 0, 0, 128),
}
GRIDS = [(48, 40), (18, 10), (8, 8), (4, 6)]   # even sides (merges, decoder blocks); partial windows; 48 x 40: 2+ groups


def _fp8_map(rng, shape, special, scale):
    y = fp8_encode(q8(rng.normal(0, scale, shape))).astype(np.uint8)
    if special:
        pick = rng.random(y.shape) < 0.05
        y[pick] = rng.choice(LD.SPECIAL_FP8, int(pick.sum()))
    return y


def cases(rng, seeds, names):
    """(name, launch args (list of ("buf", bytes) / ("null",) / ("out", n) / ("u32", v)), grid, oracle inputs)"""
    for seed in range(seeds):
        for name in names:
            kind, C, heads, mlp, extra = KERNELS[name]
            for gi, (H, W) in enumerate(GRIDS):
                for parity in (0, 1):
                    special = (gi + parity + seed) % 2 == 1
                    scale = 40.0 if special else 1.5
                    if kind in ("block", "add"):
                        tkind = "second" if C == 128 else "first"
                        off = T.block_offset(T.TABLE, tkind, parity)
                        p = O.random_block_params(rng, C, heads, mlp)
                        blob = pack_block_hyb(p, C, heads, mlp)
                        x = _fp8_map(rng, (H, W, C), special, scale)
                        s = _fp8_map(rng, (H, W, C), special, scale) if kind == "add" else None
                        ox, oy = off
                        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
                        args = [("buf", x.tobytes()), ("buf", s.tobytes()) if s is not None else ("null",),
                                ("out", H * W * C), ("buf", blob), ("u32", H), ("u32", W), ("u32", ox), ("u32", oy)]
                        yield name, args, -(-nwin // 8), {"kind": kind, "p": p, "x": x, "s": s, "off": off, "C": C,
                                                         "heads": heads, "H": H, "W": W}
                    elif kind == "merge":
                        Co = extra
                        Wm = rng.normal(0, 1.0 / np.sqrt(4 * C), (Co, 4 * C))
                        bm = rng.normal(0, 0.1, Co)
                        x = _fp8_map(rng, (H, W, C), special, scale)
                        args = [("buf", x.tobytes()), ("out", (H // 2) * (W // 2) * Co),
                                ("buf", pack_merge_hyb(Wm, bm, C, Co)), ("u32", H), ("u32", W)]
                        yield name, args, -(-((H // 2) * (W // 2)) // 256), {"kind": kind, "x": x, "Wm": Wm, "bm": bm}
                    else:
                        Cl = extra
                        Hl, Wl = H // 2, W // 2
                        We = rng.normal(0, 1.0 / np.sqrt(Cl), (4 * C, Cl))
                        be = rng.normal(0, 0.1, 4 * C)
                        y = _fp8_map(rng, (Hl, Wl, Cl), special, scale)
                        args = [("buf", y.tobytes()), ("out", H * W * C), ("buf", pack_expand_hyb(We, be, C, Cl)),
                                ("u32", Hl), ("u32", Wl)]
                        yield name, args, -(-(Hl * Wl) // 256), {"kind": kind, "x": y, "We": We, "be": be}


def oracle_codes(o):
    """The oracle's FP8 codes of a case (the hybrid form: q7, f32chain)."""
    k = dict(T.block_consts(T.TABLE, "enc1"))
    k["mm8"] = "f32chain"
    if o["kind"] in ("block", "add"):
        x = fp8_decode(o["x"]).astype(np.float64)
        if o["kind"] == "add":
            y = O.swin_block(x, o["p"], o["off"], o["heads"], k, T.TABLE["shared_dec_border"],
                             skip=fp8_decode(o["s"]).astype(np.float64))
        else:
            y = O.swin_block(x, o["p"], o["off"], o["heads"], k, "mirror")
        return fp8_encode(q8(y)).astype(np.uint8)
    x = fp8_decode(o["x"]).astype(np.float64)
    if o["kind"] == "merge":
        m = O.mm8(O.f16(o["bm"]), O.space_to_depth(x), q8(o["Wm"]), "f32chain")
        return fp8_encode(q8(np.minimum(np.maximum(m, -T.MERGE_CLAMP), T.MERGE_CLAMP))).astype(np.uint8)
    e = q8(O.mm8(O.f16(o["be"]), x, q8(o["We"]), "f32chain"))
    return fp8_encode(O.depth_to_space(e)).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.path.join(r"C:\Program Files\AMD\ROCm", "7.2"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--kernels", default=",".join(KERNELS))
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    names = [n for n in a.kernels.split(",") if n]
    for n in names:
        if n not in KERNELS:
            raise SystemExit(f"unknown kernel {n}")
    if a.prepare:
        from clean.runner import build as B
        LD.prepare_replay(a.build, a.rocm, list(KERNELS))
        B.run(B.device_command(B.find_clang(a.rocm), a.rocm, os.path.join(ROOT, "clean", "kernels", "swin_hyb.hip"),
                               os.path.join(a.build, "swin_hyb.co"), extra=["-I", os.path.join(ROOT, "clean", "kernels")]))
    if not a.run:
        return 0
    from clean.runner.hip import Hip
    h = Hip(a.rocm)
    mod = h.load_module(open(os.path.join(a.build, "swin_hyb.co"), "rb").read())
    rng = np.random.default_rng(11)
    cdir = os.path.join(a.build, "hyb_shared")
    os.makedirs(cdir, exist_ok=True)
    lines, hip_out, meta, negs = [], {}, [], {}
    for n, (name, args, grid, o) in enumerate(cases(rng, a.seeds, names)):
        ptrs, cargs, out_ptr, nout, rl = [], [], None, 0, [f"case c{n}", f"kernel {name} {grid} 256"]
        for j, arg in enumerate(args):
            if arg[0] == "buf":
                p = h.malloc(len(arg[1]))
                h.upload(p, arg[1])
                ptrs.append(p)
                cargs.append(ctypes.c_void_p(p))
                fn = f"a{n}_{j}.bin"
                with open(os.path.join(cdir, fn), "wb") as f:
                    f.write(arg[1])
                rl.append(f"arg buf {fn}")
            elif arg[0] == "null":
                cargs.append(ctypes.c_void_p(0))
                rl.append("arg null")
            elif arg[0] == "out":
                nout = arg[1]
                out_ptr = h.malloc(nout)
                h.memset(out_ptr, LD.HIP_SENTINEL, nout)
                ptrs.append(out_ptr)
                cargs.append(ctypes.c_void_p(out_ptr))
                rl.append(f"arg out {nout} o{n}.bin {LD.D3D12_SENTINEL}")
            else:
                cargs.append(ctypes.c_int(arg[1]))
                rl.append(f"arg u32 {arg[1]}")
        h.launch(h.get_function(mod, name), grid, 256, cargs)
        h.sync()
        hip_out[n] = bytes(h.download(out_ptr, nout))
        for p in ptrs:
            h.free(p)
        lines += rl + [""]
        if name not in negs and grid > 1:   # negative control: half the workgroups
            negs[name] = n
            lines += [f"case neg{n}", f"kernel {name} {grid // 2} 256"] + \
                     [x.replace(f"o{n}.bin", f"neg{n}.bin") for x in rl[2:]] + [""]
        meta.append((n, name, o))
    with open(os.path.join(cdir, "cases.txt"), "w") as f:
        f.write("\n".join(lines))
    r = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, os.path.join(cdir, "cases.txt")],
                       env=dict(os.environ, WG_L_FORM="hybrid"), capture_output=True, text=True, timeout=150,
                       creationflags=0x08000000)
    if r.returncode:
        raise SystemExit(f"kernel_replay failed: {r.stdout[-800:]}{r.stderr[-400:]}")
    res = {"cases": [], "per_kernel": {}}
    # the default leg (standing rule): no form variable set == WG_L_FORM=hybrid, every output byte
    res["default_leg"] = LD.replay_default_leg(a.build, os.path.join(cdir, "cases.txt"), dict(os.environ, WG_L_FORM="hybrid"))
    for n, name, o in meta:
        d = open(os.path.join(cdir, f"o{n}.bin"), "rb").read()
        eq = d == hip_out[n]
        c = {"case": n, "kernel": name, "identical": eq,
             "differing_bytes": int(np.count_nonzero(np.frombuffer(d, np.uint8) != np.frombuffer(hip_out[n], np.uint8)))}
        if not a.no_oracle:
            want = oracle_codes(o).ravel()
            got = np.frombuffer(d, np.uint8)
            bad = (got != want) & ~(((got & 0x7F) == 0x7F) & ((want & 0x7F) == 0x7F))
            c["oracle_fp8_differ"], c["oracle_fp8_total"] = int(bad.sum()), int(want.size)
        res["cases"].append(c)
        k = res["per_kernel"].setdefault(name, {"identical": 0, "cases": 0, "oracle_differ": 0, "oracle_total": 0})
        k["identical"] += eq
        k["cases"] += 1
        k["oracle_differ"] += c.get("oracle_fp8_differ", 0)
        k["oracle_total"] += c.get("oracle_fp8_total", 0)
    for name, n in negs.items():
        res["per_kernel"][name]["negative_control_caught"] = open(os.path.join(cdir, f"neg{n}.bin"), "rb").read() != hip_out[n]
    res["pass_bytes"] = all(k["identical"] == k["cases"] and k.get("negative_control_caught", False)
                            for k in res["per_kernel"].values()) and res["default_leg"]["pass"]
    print(json.dumps(res["per_kernel"], indent=1))
    print("pass_bytes", res["pass_bytes"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass_bytes"] else 1


if __name__ == "__main__":
    sys.exit(main())
