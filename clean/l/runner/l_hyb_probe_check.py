# SPDX-License-Identifier: GPL-3.0-only
"""GPU check of the hybrid block form's element-wise helpers on native D3D12 (kernels/l_hyb_probe.hlsl,
hlsl/wg_hyb.hlsli) against the oracle, for every FP16 input of each single-input helper and an adversarial float32 set
for the chain-end double rounding (float32 -> FP16 -> FP8). Runs with WG_L_FORM=hybrid so the device fills the tables.

    python clean/l/runner/l_hyb_probe_check.py --build <dir> [--rocm <official HIP SDK>] [--prepare] [--run] [--json out.json]
        [--mutant gelu|rsq|chainfold]   (a deliberately wrong expectation: the check must FAIL; chainfold = one
        rounding float32 -> FP8 at the chain end, as a folded conversion would give)

--prepare builds the probe's DXIL and kernel_replay.exe (host compile); --run runs the probe (one GPU slot)."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import hyb_tables as T  # noqa: E402
from clean.l.reference import oracle as O  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402

OFF_GELU, OFF_E, OFF_RSQ, OFF_RCP, OFF_CLAMP, OFF_CHAIN = 0, 65536, 196608, 327680, 458752, 720896


def chain_inputs(seed=1):
    """float32 values at and around FP8 midpoints (and the FP16 values next to them), the saturation edge, specials."""
    rng = np.random.default_rng(seed)
    codes = np.arange(0, 0x7E, dtype=np.uint8)                     # positive finite e4m3, 0 .. 448 - 1 step
    from clean.reference.fp8 import fp8_decode
    v = fp8_decode(codes).astype(np.float64)
    mids = (v[:-1] + v[1:]) / 2                                    # every midpoint between neighbours
    out = []
    for m in mids:
        m32 = np.float32(m)
        out += [m32, np.nextafter(m32, np.float32(0)), np.nextafter(m32, np.float32(np.inf))]
        h = np.float16(m)                                          # FP16 values next to the midpoint
        for d in (-2, -1, 1, 2):
            hb = (int(np.array([h], np.float16).view(np.uint16)[0]) + d) & 0xFFFF
            hv = np.array([hb], np.uint16).view(np.float16)[0].astype(np.float32)
            out += [hv, np.nextafter(hv, np.float32(0)), np.nextafter(hv, np.float32(np.inf))]
            # the float32 midpoint between two FP16 values next to the FP8 midpoint
            hb2 = (hb + 1) & 0xFFFF
            hv2 = np.array([hb2], np.uint16).view(np.float16)[0].astype(np.float32)
            mid16 = np.float32((np.float64(hv) + np.float64(hv2)) / 2)
            out += [mid16, np.nextafter(mid16, np.float32(0)), np.nextafter(mid16, np.float32(np.inf))]
    out += [np.float32(x) for x in (448.0, 464.0, 480.0, 1e6, 65504.0, 65520.0, 0.0, -0.0, 1e-30, 2.0 ** -10, 2.0 ** -9,
                                    3 * 2.0 ** -11, np.inf, -np.inf, np.nan)]
    out += list(rng.standard_normal(20000).astype(np.float32) * np.float32(4.0) ** rng.integers(-8, 5, 20000))
    x = np.array(out, np.float32)
    x = np.concatenate([x, -x])
    x = x[: len(x) - len(x) % 8]
    return x


def expected(chain):
    H = T.H
    want = {}
    want["gelu"] = T.gelu16_q8()
    with np.errstate(invalid="ignore", over="ignore"):
        u = np.fmin(np.fmax(T.f16(H * O.Q7_SCORE_SCALE), -O.Q7_SCORE_CLAMP), O.Q7_SCORE_CLAMP)
    want["e"] = T.softmax_e_ref(u)
    valid_rsq = (H >= 2.0 ** -13) & np.isfinite(H)
    with np.errstate(divide="ignore", invalid="ignore"):
        want["rsq"] = T._rsq_ref(H).astype(np.float16).view(np.uint16)
        rc = T.f16(O._rcp32(H))
    valid_rcp = (H >= 2.0 ** -14) & np.isfinite(H) & (rc >= 2.0 ** -14)
    want["rcp"] = rc.astype(np.float16).view(np.uint16)
    with np.errstate(invalid="ignore"):
        cl = np.where(np.isnan(H), H, np.fmin(np.fmax(H, -6.28125), 6.28125)).astype(np.float32)
    want["clamp"] = cl.view(np.uint32)
    with np.errstate(invalid="ignore", over="ignore"):
        want["chain"] = np.asarray(fp8_encode(q8(chain.astype(np.float16).astype(np.float64))), np.uint8)
    return want, valid_rsq, valid_rcp


def compare(raw, chain, mutant=None):
    want, vr, vc = expected(chain)
    if mutant == "gelu":
        want["gelu"] = want["gelu"].copy()
        want["gelu"][0x3C00] ^= 1
    if mutant == "chainfold":
        with np.errstate(invalid="ignore", over="ignore"):
            want["chain"] = np.asarray(fp8_encode(q8(chain.astype(np.float64))), np.uint8)
    if mutant == "rsq":
        want["rsq"] = want["rsq"].copy()
        want["rsq"][0x3C00] ^= 1
    got = {
        "gelu": np.frombuffer(raw, np.uint8, 65536, OFF_GELU),
        "e": np.frombuffer(raw, np.uint16, 65536, OFF_E),
        "rsq": np.frombuffer(raw, np.uint16, 65536, OFF_RSQ),
        "rcp": np.frombuffer(raw, np.uint16, 65536, OFF_RCP),
        "clamp": np.frombuffer(raw, np.uint32, 65536, OFF_CLAMP),
        "chain": np.frombuffer(raw, np.uint8, len(chain), OFF_CHAIN),
    }
    res = {}
    for k in got:
        g, w = got[k], want[k]
        mask = vr if k == "rsq" else (vc if k == "rcp" else np.ones(len(w), bool))
        if k == "clamp":   # NaN: any NaN
            gn = (g & 0x7F800000) == 0x7F800000
            same = (g == w) | (gn & ((g & 0x7FFFFF) != 0) & ((w & 0x7F800000) == 0x7F800000) & ((w & 0x7FFFFF) != 0))
        elif k in ("chain", "gelu"):   # FP8 NaN: either code (the hardware conversion writes 0xFF for every NaN)
            same = (g == w) | (((g & 0x7F) == 0x7F) & ((w & 0x7F) == 0x7F))
        else:
            same = g == w
        bad = np.nonzero(mask & ~same)[0]
        res[k] = {"compared": int(mask.sum()), "differ": int(bad.size), "first": [int(i) for i in bad[:8]]}
    res["pass"] = all(v["differ"] == 0 for v in res.values() if isinstance(v, dict))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.path.join(r"C:\Program Files\AMD\ROCm", "7.2"), help="the official HIP SDK")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--mutant", choices=("gelu", "rsq", "chainfold"))
    ap.add_argument("--hip", action="store_true", help="the HIP twin (clean/l/kernels/l_hyb_probe.hip) instead")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    os.makedirs(a.build, exist_ok=True)
    if a.prepare and a.hip:
        from clean.runner import build as B
        B.run(B.device_command(B.find_clang(a.rocm), a.rocm, os.path.join(ROOT, "clean", "l", "kernels", "l_hyb_probe.hip"),
                               os.path.join(a.build, "l_hyb_probe.co"),
                               extra=["-I", os.path.join(ROOT, "clean", "kernels")]))
    elif a.prepare:
        from clean.l.runner.l_d3d12_check import prepare_replay
        prepare_replay(a.build, a.rocm, ["l_hyb_probe"])
    chain = chain_inputs()
    cdir = os.path.join(a.build, "hyb_probe_hip" if a.hip else "hyb_probe")
    os.makedirs(cdir, exist_ok=True)
    src = os.path.join(cdir, "chain.bin")
    out = os.path.join(cdir, "out.bin")
    if a.run and a.hip:
        import ctypes
        from clean.runner.hip import Hip
        h = Hip(a.rocm)
        fn = h.get_function(h.load_module(open(os.path.join(a.build, "l_hyb_probe.co"), "rb").read()), "l_hyb_probe")
        nbytes = OFF_CHAIN + len(chain)
        ps, pd = h.malloc(chain.nbytes), h.malloc(nbytes)
        h.upload(ps, chain.tobytes())
        h.memset(pd, 165, nbytes)
        h.launch(fn, 32, 256, [ctypes.c_void_p(ps), ctypes.c_void_p(pd), ctypes.c_uint(len(chain))])
        h.sync()
        with open(out, "wb") as f:
            f.write(h.download(pd, nbytes))
    elif a.run:
        chain.tofile(src)
        nbytes = OFF_CHAIN + len(chain)
        with open(os.path.join(cdir, "cases.txt"), "w") as f:
            f.write("case hyb_probe\nkernel l_hyb_probe 32 256\n"
                    f"arg buf chain.bin\narg out {nbytes} out.bin 165\narg u32 {len(chain)}\n")
        env = dict(os.environ)
        env["WG_L_FORM"] = "hybrid"
        r = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, os.path.join(cdir, "cases.txt")],
                           env=env, capture_output=True, text=True, timeout=150, creationflags=0x08000000)
        if r.returncode:
            raise SystemExit(f"kernel_replay failed ({r.returncode}):\n{r.stdout[-1500:]}{r.stderr[-800:]}")
    raw = open(out, "rb").read()
    res = compare(raw, chain, a.mutant)
    res["chain_inputs"] = int(len(chain))
    res["mutant"] = a.mutant
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if (res["pass"] != bool(a.mutant)) else 1


if __name__ == "__main__":
    sys.exit(main())
