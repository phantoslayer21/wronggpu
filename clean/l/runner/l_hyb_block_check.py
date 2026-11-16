# SPDX-License-Identifier: MIT
"""Block-level check of the HYBRID C32 kernels (the first leg of the hybrid check design; the per-stage check with the
SW_HYB_DUMP variant follows): random parameters in the hybrid values (l_hyb_pack.py), the encoder (FP16 halo map in)
and decoder (FP8 map in, FP16 out) blocks on the L block grids, with special values in every other case.

  1. HIP (l_swin_hyb.co) == D3D12 (l_swin_block_c32_*_hyb) raw bytes; each output starts with its own sentinel; a
     half-grid D3D12 run is the negative control (must differ).
  2. Encoder only, informational: the block output against the oracle's FP16 form with f32chain chains (border
     "input", halo 2): counts of identical / differing FP8 codes (the per-stage check gives the pass / fail).

    python clean/l/runner/l_hyb_block_check.py --build <dir> [--prepare] [--run] [--seeds N] [--json out.json]
        [--mutant]   also the order mutants l_swin_block_c32_*_hyb_mut (SW_HYB_MUT_ORDER=1, same build): each mutant
                     kernel must differ from HIP in at least one case, or the command fails (exit 1)
        [--mutant-unchanged]  self-check of that leg: the mutant leg run on the unmodified kernels; the command must
                     then FAIL (it does unless the exit condition is broken)
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
from clean.l.runner.l_hyb_pack import pack_block_hyb  # noqa: E402
from clean.reference.fp8 import fp8_decode, fp8_encode, q8  # noqa: E402

KERNELS = {"l_swin_block_c32_enc_hyb": "enc", "l_swin_block_c32_dec_hyb": "dec"}


def cases(rng, seeds):
    for seed in range(seeds):
        for kernel, kind in KERNELS.items():
            tkind = "first" if kind == "enc" else "second"
            h = T.block_halo(T.TABLE, tkind)
            for gi, (H, W) in enumerate(LD.L_BLOCK_GRIDS):
                for parity in (0, 1):
                    p = O.random_block_params(rng, 32, 2, 128)
                    blob = pack_block_hyb(p, 32, 2, 128)
                    off = T.block_offset(T.TABLE, tkind, parity)
                    special = (gi + parity + seed) % 2 == 1
                    scale = 60.0 if special else 1.5
                    if kind == "enc":
                        x = rng.normal(0, scale, (H + 2 * h, W + 2 * h, 32)).astype(np.float16)
                        if special:
                            pick = rng.random(x.shape) < 0.05
                            x[pick] = rng.choice(LD.SPECIAL_F16, int(pick.sum()))
                        yield kernel, H, W, off, x.tobytes(), H * W * 32, blob, p, None
                    else:
                        y = fp8_encode(q8(rng.normal(0, scale, (H, W, 32)))).astype(np.uint8)
                        if special:
                            pick = rng.random(y.shape) < 0.05
                            y[pick] = rng.choice(LD.SPECIAL_FP8, int(pick.sum()))
                        s = fp8_encode(q8(rng.normal(0, scale, (H, W, 32)))).astype(np.uint8)   # the skip S
                        if special:
                            pick = rng.random(s.shape) < 0.05
                            s[pick] = rng.choice(LD.SPECIAL_FP8, int(pick.sum()))
                        yield kernel, H, W, off, y.tobytes(), H * W * 64, blob, p, s.tobytes()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.path.join(r"C:\Program Files\AMD\ROCm", "7.2"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mutant", action="store_true",
                    help="also the order mutants: each mutant kernel must differ from HIP in at least one case (exit 1 "
                         "otherwise)")
    ap.add_argument("--mutant-unchanged", action="store_true",
                    help="self-check of the mutant leg: run it on the UNMODIFIED kernels; the command must then fail")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.prepare:
        from clean.runner import build as B
        LD.prepare_replay(a.build, a.rocm, list(KERNELS))
        B.run(B.device_command(B.find_clang(a.rocm), a.rocm, os.path.join(ROOT, "clean", "l", "kernels", "l_swin_hyb.hip"),
                               os.path.join(a.build, "l_swin_hyb.co"),
                               extra=["-I", os.path.join(ROOT, "clean", "kernels")]))
    if not a.run:
        return 0
    from clean.runner.hip import Hip
    h = Hip(a.rocm)
    mod = h.load_module(open(os.path.join(a.build, "l_swin_hyb.co"), "rb").read())
    rng = np.random.default_rng(7)
    cdir = os.path.join(a.build, "hyb_block")
    os.makedirs(cdir, exist_ok=True)
    lines, hip_out, meta = [], {}, []
    for n, (kernel, H, W, off, xb, nbytes, blob, p, sb) in enumerate(cases(rng, a.seeds)):
        # HIP
        px, pw, po = h.malloc(len(xb)), h.malloc(len(blob)), h.malloc(nbytes)
        ps = h.malloc(len(sb)) if sb else 0
        if sb:
            h.upload(ps, sb)
        h.upload(px, xb)
        h.upload(pw, blob)
        h.memset(po, LD.HIP_SENTINEL, nbytes)
        ox, oy = off
        nwin = -(-(W + ox) // 4) * -(-(H + oy) // 4)
        h.launch(h.get_function(mod, kernel), -(-nwin // 8), 256,
                 [ctypes.c_void_p(px), ctypes.c_void_p(ps), ctypes.c_void_p(po), ctypes.c_void_p(pw), ctypes.c_int(H),
                  ctypes.c_int(W), ctypes.c_int(ox), ctypes.c_int(oy)])
        h.sync()
        hip_out[n] = bytes(h.download(po, nbytes))
        with open(os.path.join(cdir, f"h{n}.bin"), "wb") as f:   # kept for diagnosis
            f.write(hip_out[n])
        for q in (px, pw, po) + ((ps,) if sb else ()):
            h.free(q)
        # D3D12 case (same inputs; the decoder's FP16 output goes through the out pointer as the HIP epilogue does)
        for nm, data in ((f"x{n}.bin", xb), (f"w{n}.bin", blob)) + (((f"s{n}.bin", sb),) if sb else ()):
            with open(os.path.join(cdir, nm), "wb") as f:
                f.write(data)
        grid = -(-nwin // 8)
        sarg = f"arg buf s{n}.bin" if sb else "arg null"
        lines += [f"case c{n}", f"kernel {kernel} {grid} 256", f"arg buf x{n}.bin", sarg,
                  f"arg out {nbytes} o{n}.bin {LD.D3D12_SENTINEL}", f"arg buf w{n}.bin", f"arg u32 {H}", f"arg u32 {W}",
                  f"arg u32 {ox}", f"arg u32 {oy}", ""]
        if n == 0:   # negative control: half the workgroups
            lines += [f"case neg{n}", f"kernel {kernel} {max(1, grid // 2)} 256", f"arg buf x{n}.bin", sarg,
                      f"arg out {nbytes} neg{n}.bin {LD.D3D12_SENTINEL}", f"arg buf w{n}.bin", f"arg u32 {H}",
                      f"arg u32 {W}", f"arg u32 {ox}", f"arg u32 {oy}", ""]
        meta.append((n, kernel, H, W, off, p, xb, sb))
    with open(os.path.join(cdir, "cases.txt"), "w") as f:
        f.write("\n".join(lines))
    a.mutant = a.mutant or a.mutant_unchanged
    if a.mutant:   # the same cases on the mutant kernels, outputs m<n>.bin (the negative-control cases dropped)
        import re
        out, keep = [], True
        for ln in lines:
            if ln.startswith("case "):
                keep = not ln.startswith("case neg")
            if keep:
                if not a.mutant_unchanged:
                    ln = re.sub(r"^kernel (\S+_hyb) ", r"kernel \1_mut ", ln)
                ln = re.sub(r" o(\d+)\.bin ", r" m\1.bin ", ln)
                out.append(ln)
        with open(os.path.join(cdir, "cases_mut.txt"), "w") as f:
            f.write("\n".join(out))
    env = dict(os.environ, WG_L_FORM="hybrid")
    res = {"cases": []}
    caught_by_kernel = {}
    for label, cfile, pre in (("d3d12", "cases.txt", "o"),) + ((("mutant", "cases_mut.txt", "m"),) if a.mutant else ()):
        r = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, os.path.join(cdir, cfile)], env=env,
                           capture_output=True, text=True, timeout=150, creationflags=0x08000000)
        if r.returncode:
            raise SystemExit(f"kernel_replay ({label}) failed: {r.stdout[-800:]}{r.stderr[-400:]}")
        same = 0
        for n, kernel, H, W, off, p, xb, sb in meta:
            d = open(os.path.join(cdir, f"{pre}{n}.bin"), "rb").read()
            eq = d == hip_out[n]
            same += eq
            if label == "mutant" and not eq:
                caught_by_kernel[kernel] = caught_by_kernel.get(kernel, 0) + 1
            if label == "d3d12":
                c = {"case": n, "kernel": kernel, "H": H, "W": W, "identical": eq,
                     "differing_bytes": int(np.count_nonzero(np.frombuffer(d, np.uint8) != np.frombuffer(hip_out[n], np.uint8)))}
                if kernel.endswith("enc_hyb"):   # informational oracle comparison (FP8 codes)
                    x = np.frombuffer(xb, np.float16).reshape(H + 4, W + 4, 32).astype(np.float64)
                    k = dict(T.consts(T.TABLE)) if hasattr(T, "consts") else {"window": 4, "head_dim": 32}
                    k.update({"form": "q7", "mm8": "f32chain", "head_dim": 32, "window": 4})
                    want = fp8_encode(q8(O.swin_block(x, p, off, 2, k, border="input", halo=2))).astype(np.uint8)
                    got = np.frombuffer(d, np.uint8).reshape(H, W, 32)
                    bad = (got != want) & ~(((got & 0x7F) == 0x7F) & ((want & 0x7F) == 0x7F))
                    c["oracle_fp8_differ"] = int(bad.sum())
                    c["oracle_fp8_total"] = int(want.size)
                elif kernel.endswith("dec_hyb"):   # informational: the FP16 output against the oracle WITH the skip
                    k = {"form": "q7", "mm8": "f32chain", "head_dim": 32, "window": 4}
                    xd = fp8_decode(np.frombuffer(xb, np.uint8).reshape(H, W, 32)).astype(np.float64)
                    sk = fp8_decode(np.frombuffer(sb, np.uint8).reshape(H, W, 32)).astype(np.float64)
                    want = O.f16(O.swin_block(xd, p, off, 2, k, border="coarse", skip=sk))
                    got = np.frombuffer(d, np.float16).reshape(H, W, 32).astype(np.float64)
                    bad = ~((got == want) | (np.isnan(got) & np.isnan(want)))
                    c["oracle_f16_differ"] = int(bad.sum())
                    c["oracle_f16_total"] = int(want.size)
                res["cases"].append(c)
        res[f"{label}_identical"] = f"{same}/{len(meta)}"
        if label == "d3d12":   # the default leg: no form variable set (the shipped default) == WG_L_FORM=hybrid
            res["default_leg"] = LD.replay_default_leg(a.build, os.path.join(cdir, cfile), env)
    neg = open(os.path.join(cdir, "neg0.bin"), "rb").read()
    res["negative_control_caught"] = neg != hip_out[0]
    res["pass_bytes"] = (res["d3d12_identical"] == f"{len(meta)}/{len(meta)}" and res["negative_control_caught"]
                         and res["default_leg"]["pass"])
    if a.mutant:
        # fail closed: EVERY mutant kernel (one per kernel checked) must differ from HIP in at least one case
        kernels = sorted({kernel for _, kernel, *_ in meta})
        res["mutant_caught_by_kernel"] = {k: caught_by_kernel.get(k, 0) for k in kernels}
        res["mutant_caught_by_bytes"] = all(caught_by_kernel.get(k, 0) > 0 for k in kernels)
        res["mutant_unchanged_self_check"] = a.mutant_unchanged
        res["pass_bytes"] = res["pass_bytes"] and res["mutant_caught_by_bytes"]
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["pass_bytes"] else 1


if __name__ == "__main__":
    sys.exit(main())
