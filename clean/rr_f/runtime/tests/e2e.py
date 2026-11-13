# SPDX-License-Identifier: MIT
"""End-to-end test of WrongGPU's RR runtime (wronggpu_rr_runtime.dll) through its owned adapter.

    python -m clean.rr_f.runtime.tests.e2e [--out DIR] [--model synthetic|admission|wgcm] [--wgcm FILE] [--calls 61]
                                           [--skip-gpu] [--jobs 8] [--perf 600]

1. Stage folder (the proxy's directory layout): the runtime library + a model file (a synthetic model in the runtime's
   own container, tools/pack_model.py, or the converter's model.wgcm, which the runtime packs itself).
2. Scenarios: synthetic inputs in real D3D12 textures (rr_test_host.exe plays the proxy, with the proxy's own
   Lifecycle).
   A: 61 evaluate calls on one instance with alpha upscaling, output 48x30 at origin (1, 1): render 29x19 -> 40x26 ->
      29x19 (two render-size changes, each followed by a reset frame), a game reset flag, one evaluate with a required
      input missing (must fail before any pass and change no state), the colour as a lifetime-0 copy, cameras usable /
      unusable (specular motion by reconstruction, by SV, none), guides on and off, non-finite input values, non-zero
      input origins, a roughness texture whose padding columns fall outside it, binary16 / binary32 input formats,
      a hit distance with the linear conversion (an f32 copy) on some calls; the resolve's transparency layer
      (RGBA16F / RGBA8 / RGB10A2, appearing at call 3: TH created), the opacity layer on some calls (OH created), the
      input alpha (R16F / R8) or the colour's alpha, a separate output alpha (R16F) from call 25.
   B: 14 calls without alpha upscaling, output 47x29 (odd: We x He > Wo x Ho) in RGBA16F: plain resolve, a layer on
      calls 4-9 (extended without alpha), bounded stores on calls 6-11, a game reset at call 8.
   Every frame is run twice: with the work buffers not aliased (WG_RR_ALIAS=0, every intermediate read back) and as in
   production (aliased by lifetime); the outputs and every state image of the two runs must be identical.
   Then create / evaluate / destroy cycles (leak check) and timing runs (the runtime logs GPU time per 300 frames).
3. Every frame is checked against the CPU reference pipeline (clean/rr_f/reference, head/), stage by stage, each
   stage fed with the GPU's own inputs to it, with the temporal state the GPU carried from the previous frame (dumped
   arena) - the pages' rules: PH2 pool, input stage, body (51.8, exact row sums), resolve (54b, transparency layer,
   opacity, alpha upscaling, the TH / OH / AH histories, stores into the game's outputs) bit-exact; head 55.3; output
   stage 54 qualification; exposure 58. A whole-frame chain (CPU from the frame's start state only, through the
   output stage) is reported as well.
GPU runs go through the optional GPU lock (WG_GPU_LOCK). Results: <out>/e2e*/report.json and report.md."""
import argparse
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(RT)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_OUT = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "build")), "rr-f-rt-int", "out")
DEFAULT_WGCM = os.environ.get("WG_RR_MODEL", "")   # a synthetic model file (clean/rr_f/runtime/tools/pack_model.py)
# Optional GPU lock (WG_GPU_LOCK: a script run as `python <lock> --timeout <s> -- <command>`, e.g. to keep
# GPU jobs of several shells apart); unset: the command runs directly.
GPU_LOCK = os.environ.get("WG_GPU_LOCK", "")


def _locked(cmd, timeout="3600"):
    return ([sys.executable, GPU_LOCK, "--timeout", str(timeout), "--"] if GPU_LOCK else []) + list(cmd)

# WgRrRole (wg_rr_adapter.h)
ROLE = dict(COLOUR=0, OUTPUT=1, DEPTH=2, MOTION=3, DIFFUSE_ALBEDO=4, SPECULAR_ALBEDO=5, NORMALS=6, ROUGHNESS=7,
            SPECULAR_MOTION=8, SPECULAR_HIT_DIST=9, DISOCCLUSION=10, G1=11, DOF_GUIDE=12, BEFORE_TRANSPARENCY=13,
            LAYER=14, OPACITY=15, INPUT_ALPHA=16, OUTPUT_ALPHA=17)
# page-52 input roles <- adapter roles
IS_ROLE = dict(C="COLOUR", Z="DEPTH", V="MOTION", SV="SPECULAR_MOTION", Dd="DIFFUSE_ALBEDO", Ds="SPECULAR_ALBEDO",
               N="NORMALS", R="ROUGHNESS", Q="DISOCCLUSION", B="BEFORE_TRANSPARENCY", G1="G1", G2="DOF_GUIDE",
               Hd="SPECULAR_HIT_DIST")
DX = dict(RGBA32F=2, RGBA16F=10, RGB10A2=24, RGBA8=28, RG32F=16, RG16F=34, R32F=41, R16F=54, R8=61)
FMT_CH = {2: 4, 10: 4, 24: 4, 28: 4, 16: 2, 34: 2, 41: 1, 54: 1, 61: 1}
ALL_SHADER = 0x40 | 0x80                   # D3D12_RESOURCE_STATE_(NON_)PIXEL_SHADER_RESOURCE
OUT_ORIGIN = (1, 1)                        # rr_test_host's output origin
RENDER_A, RENDER_B = (29, 19), (40, 26)
SCEN = dict(A=dict(output=(48, 30), alpha=1, out_fmt=DX["RGBA32F"]), B=dict(output=(47, 29), alpha=0, out_fmt=DX["RGBA16F"]))


# ------------------------------------------------------------------------------------------------- formats
def storage(d, fmt):
    """float data [h][w][c] (c <= the format's channels) -> the texture's storage array; and the values the API read
    conversion gives (the reference's f32 view) / the resolve's 54b.N surface."""
    h, w = d.shape[:2]
    ch = FMT_CH[fmt]
    full = np.zeros((h, w, ch), np.float64)
    full[..., :d.shape[2]] = d
    if fmt in (DX["RGBA16F"], DX["RG16F"], DX["R16F"]):
        s = full.astype(np.float16)
        return s, s.astype(np.float32), ("f16", s)
    if fmt in (DX["RGBA8"], DX["R8"]):
        codes = np.clip(np.rint(np.nan_to_num(full) * 255), 0, 255).astype(np.uint8)
        return codes, (codes.astype(np.float64) / 255).astype(np.float32), ("unorm8", codes)
    if fmt == DX["RGB10A2"]:
        codes = np.clip(np.rint(np.nan_to_num(full) * np.array([1023, 1023, 1023, 3])), 0, [1023, 1023, 1023, 3]).astype(np.uint16)
        packed = (codes[..., 0].astype(np.uint32) | (codes[..., 1].astype(np.uint32) << 10) |
                  (codes[..., 2].astype(np.uint32) << 20) | (codes[..., 3].astype(np.uint32) << 30))
        vals = (codes.astype(np.float64) / np.array([1023, 1023, 1023, 3])).astype(np.float32)
        return packed, vals, ("rgb10a2", codes)
    s = full.astype(np.float32)
    return s, s, ("f32", s)


# ------------------------------------------------------------------------------------------------- scenarios
def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def scenario(name, n_calls, seed=6061):
    """The evaluate calls of scenario A or B: list of dicts (inputs of one call)."""
    from clean.rr_f.tests import synth_input as SI
    rng = np.random.default_rng(seed + (0 if name == "A" else 7))
    V2C = SI.perspective(rng)
    W2V0 = SI.world_to_view(rng, far=20.0)
    origins = dict(C=(1, 2), Z=(0, 0), V=(2, 1), SV=(0, 0), Dd=(1, 0), Ds=(0, 1), N=(2, 2), R=(3, 0), Q=(0, 0),
                   B=(0, 0), G1=(4, 1), G2=(0, 3), Hd=(1, 1))
    calls = []
    for i in range(n_calls):
        if name == "A":
            size = RENDER_A if (i < 22 or i >= 44) else RENDER_B
        else:
            size = RENDER_A
        wr, hr = size
        reset = 1 if (name == "A" and i == 12) or (name == "B" and i == 8) else 0
        c = dict(index=i, render=size, reset=reset, settings=0, depth_linear=0,
                 jitter=(np.float32(halton(i + 1, 2) - 0.5), np.float32(halton(i + 1, 3) - 0.5)),
                 mv_scale=(np.float32(0.9 / wr), np.float32(-1.1 / hr)), fail_role=None, layer_origin=(0, 0))
        W2V = W2V0.copy()
        W2V[3, :3] += np.float32(0.05) * i
        c["V2C"], c["W2V"] = V2C.astype(np.float32), W2V.astype(np.float32)
        if name == "A" and 26 <= i < 30:   # unusable camera matrices (all-zero world-to-view): SV = V fallback (62.4)
            c["W2V"] = np.zeros((4, 4), np.float32)
        if name == "A":
            smv = "Hd" if (i < 10 or i >= 26) else ("SV" if i < 20 else None)
            guides = [] if 20 <= i < 26 else ["Q", "B", "G1", "G2"] if i >= 5 else ["G1"]
        else:
            smv, guides = "Hd", ["Q", "G1"]
        edge = 0.01 if 40 <= i < 43 else 0.0
        roles = ["C", "Z", "V", "Dd", "Ds", "N", "R"] + guides + ([smv] if smv else [])
        tex = {}
        for r in roles:
            ch = {"C": 4, "Z": 1, "V": 2, "SV": 2, "Dd": 3, "Ds": 3, "N": 3, "R": 1, "Q": 1, "B": 3, "G1": 1, "G2": 1,
                  "Hd": 1}[r]
            src = {"SV": "V", "B": "C", "G1": "C", "G2": "C"}.get(r, r)
            o = origins[src]
            margin = (1, 0) if r == "R" else (2, 1)   # roughness: padding reads (52.3.4) partly outside the texture
            w, h = wr + o[0] + margin[0], hr + o[1] + margin[1]
            shape = (h, w, ch)
            if r in ("C", "B"):
                d = SI.texture(rng, shape, -0.2, 6.0, edge, SI.EDGES)
            elif r == "Z":
                d = SI.texture(rng, shape, 0.05, 1.0, edge, [0.0, 1.0, np.nan])
            elif r in ("V", "SV"):
                d = SI.texture(rng, shape, -6.0, 6.0, edge, [np.nan, np.inf, 0.0])
            elif r in ("Dd", "Ds"):
                d = SI.texture(rng, shape, -0.1, 1.2, edge, SI.EDGES)
            elif r == "N":
                d = SI.texture(rng, shape, -1.1, 1.1, edge, SI.EDGES)
            elif r == "R":
                d = SI.texture(rng, shape, 0.0, 1.1, edge, SI.EDGES)
            elif r == "Q":
                d = SI.texture(rng, shape, 0.0, 1.0, edge, [np.nan, np.inf])
            elif r in ("G1", "G2"):
                d = SI.texture(rng, shape, -3.0, 5.0, edge, SI.EDGES)
            else:
                d = SI.texture(rng, shape, -1.0, 60.0, edge, [0.0, np.nan, np.inf])
            fmt = {1: DX["R32F"], 2: DX["RG32F"], 3: DX["RGBA32F"], 4: DX["RGBA32F"]}[ch]
            if r == "Dd" and i % 3 == 1:
                fmt = DX["RGBA16F"]                          # binary16 inputs through the SRV (exact values)
            if r == "V" and i % 4 == 2:
                fmt = DX["RG16F"]
            stor, vals, _ = storage(d, fmt)
            conv = (0, 1.0, 0.0)
            if r == "Hd" and 30 <= i < 40:                   # the linear conversion of WgRrImage (an f32 copy)
                conv = (1, 2.0, 0.5)
                vals = (vals * np.float32(2.0) + np.float32(0.5)).astype(np.float32)
            state = 0 if r == "Dd" else ALL_SHADER          # one input declared COMMON: transitions are exercised
            tex[r] = dict(store=stor, values=vals[..., :ch] if ch < vals.shape[2] else vals, origin=origins[r],
                          format=fmt, state=state, conv=conv)
        c["tex"] = tex
        extra = {}
        lay_on = (3 <= i and not (20 <= i < 25)) if name == "A" else (4 <= i < 10)
        if lay_on:
            lfmt = [DX["RGBA16F"], DX["RGBA8"], DX["RGB10A2"]][i % 3] if name == "A" else DX["RGBA16F"]
            c["layer_origin"] = (1, 1)
            d = rng.uniform(0, 1, (hr + 2, wr + 2, 4))
            if lfmt == DX["RGBA16F"]:
                d[..., :3] *= 3.0
            stor, _, surf = storage(d, lfmt)
            extra["LAYER"] = dict(store=stor, surf=surf, origin=(1, 1), format=lfmt)
            if name == "A" and (15 <= i < 20 or 45 <= i < 53):
                stor, _, surf = storage(rng.uniform(0, 1, (hr + 1, wr + 1, 4)), DX["RGBA8"])
                extra["OPACITY"] = dict(store=stor, surf=surf, origin=(1, 0), format=DX["RGBA8"])
        if name == "A" and 10 <= i < 40:
            afmt = DX["R16F"] if i % 2 else DX["R8"]
            stor, _, surf = storage(rng.uniform(0, 1, (hr + 1, wr + 2, 1)), afmt)
            extra["INPUT_ALPHA"] = dict(store=stor, surf=surf, origin=(2, 1), format=afmt)
        if name == "A" and i >= 25:
            wo, ho = SCEN["A"]["output"]
            stor = np.full((ho + 2, wo + 1, 1), np.float16(0.0078125), np.float16)
            extra["OUTPUT_ALPHA"] = dict(store=stor, surf=("f16", stor), origin=(0, 0), format=DX["R16F"])
        if name == "B" and 6 <= i < 12:
            c["settings"] = 2                                   # bounded stores (WG_RR_SET_BOUNDED_STORES)
        c["extra"] = extra
        calls.append(c)
    if name == "A" and n_calls > 33:
        calls[33]["fail_role"] = ROLE["DIFFUSE_ALBEDO"]   # a failed evaluate (required input detached)
    if name == "A" and n_calls > 50:
        calls[50]["transient_role"] = ROLE["COLOUR"]      # colour handed over as a lifetime-0 copy (62.1)
    return calls


def write_frame(path, c):
    out = bytearray()
    out += struct.pack("<IIIIII", 0x32465252, c["render"][0], c["render"][1], c["reset"], c["settings"], c["depth_linear"])
    out += struct.pack("<4f", *c["jitter"], *c["mv_scale"])
    out += np.asarray(c["V2C"], np.float32).tobytes() + np.asarray(c["W2V"], np.float32).tobytes()
    out += struct.pack("<II", *c["layer_origin"])
    roles = [(ROLE[IS_ROLE[r]], t) for r, t in c["tex"].items()] + [(ROLE[k], t) for k, t in c["extra"].items()]
    out += struct.pack("<I", len(roles))
    for rid, t in roles:
        h, w = t["store"].shape[:2]
        conv = t.get("conv", (0, 1.0, 0.0))
        state = t.get("state", ALL_SHADER if rid != ROLE["OUTPUT_ALPHA"] else 0)
        out += struct.pack("<IIIIIIIIff", rid, t["format"], w, h, t["origin"][0], t["origin"][1], state, conv[0],
                           conv[1], conv[2])
        out += np.ascontiguousarray(t["store"]).tobytes()
    with open(path, "wb") as fh:
        fh.write(out)


# ------------------------------------------------------------------------------------------------- GPU run
def run_gpu(out, stage, runs, perf_frames, name="e2e", leak=6, head=None, body="Z", inp="exact", front=True, mips=1):
    """runs: [(scenario name, calls)]; every scenario twice (WG_RR_ALIAS=0 and aliased)."""
    work = os.path.join(out, name)
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "in"))
    job = [f"runtime {os.path.join(stage, 'wronggpu_rr_runtime.dll')}"]
    if head:                                   # WG_RR_HEAD for every instance of this run (e.g. fast)
        job.append(f"env WG_RR_HEAD {head}")
    if body:                                   # WG_RR_BODY (default Z: the exact verification mode, so that the body
        job.append(f"env WG_RR_BODY {body}")   # and the front-end MLP are checked bit-exact; "default": the runtime's)
    if inp:                                    # WG_RR_INPUT (default exact: since clean/rr-f-speed8 the runtime's default
        job.append(f"env WG_RR_INPUT {inp}")   # input features are relaxed, which the bit-exact input check flags)
    # clean/rr-f-s11-tail: the runtime's default records the next frame's PH2 pool beside the resolve, so a dump after
    # frame i holds pool(PH1(i)) where the stage-by-stage check expects frame i's own front pool: the checks run with the
    # front order (WG_RR_TAILPREP=0; same outputs and state otherwise, r11-tail seqcmp)
    if front:
        job.append("env WG_RR_TAILPREP 0")
    if mips > 1:                               # 0.6.0 follow-up: the game's inputs with mips (mip 0 is read): the checks
        job.append(f"env WG_RR_TEST_MIPS {mips}")  # below then prove the same bits as with one mip
    for sname, calls in runs:
        sc = SCEN[sname]
        for alias in (0, 1):
            job += [f"outdir {os.path.join(work, f'run{sname}{alias}')}", f"env WG_RR_ALIAS {'0' if not alias else '-'}",
                    f"output {sc['out_fmt']} 1",
                    f"create {sc['output'][0]} {sc['output'][1]} {RENDER_A[0]} {RENDER_A[1]} 0 1 0 1 1 {sc['alpha']}"]
            for c in calls:
                p = os.path.join(work, "in", f"{sname}_{c['index']:03d}.bin")
                if not alias:
                    write_frame(p, c)
                if c["fail_role"] is not None:
                    job.append(f"fail {p} {c['fail_role']}")
                elif c.get("transient_role") is not None:
                    job.append(f"transient {p} {c['transient_role']}")
                    job.append(f"retag {p} {c['transient_role']}")  # bug hunt B: a copy released before any evaluate
                else:
                    job.append(f"frame {p}")
            job.append("destroy")
    job.append("env WG_RR_ALIAS -")
    # bug hunt E: an array texture as the colour input is refused before any pass (an instance of its own, after the
    # scenarios, so their frames and analysis are unchanged)
    sc0 = SCEN[runs[0][0]]
    job += [f"outdir {os.path.join(work, 'arrayin')}", f"output {sc0['out_fmt']} 1",
            f"create {sc0['output'][0]} {sc0['output'][1]} {RENDER_A[0]} {RENDER_A[1]} 0 1 0 1 1 {sc0['alpha']}",
            f"arrayin {os.path.join(work, 'in', runs[0][0] + '_000.bin')} {ROLE['COLOUR']}", "destroy"]
    # bug hunt G: an output without unordered access (the runtime's texture + copy): nothing outside the store's
    # rectangle changes; scenario B's odd output size (47 x 29) also covers the even-rounded store grid
    for sname in sorted({r[0] for r in runs}):
        scb = SCEN[sname]
        job += [f"outdir {os.path.join(work, 'border' + sname)}", f"output {scb['out_fmt']} 1",
                f"create {scb['output'][0]} {scb['output'][1]} {RENDER_A[0]} {RENDER_A[1]} 0 1 0 1 1 {scb['alpha']}",
                f"border {os.path.join(work, 'in', sname + '_000.bin')}", "destroy"]
    if leak:
        job.append(f"leak {leak} {os.path.join(work, 'in', runs[0][0] + '_000.bin')}")
    if perf_frames:
        job.append(f"perf 2560 1440 1706 960 {perf_frames} exact plain")
        job.append(f"perf 2560 1440 1706 960 {perf_frames} exact ext")
        job.append(f"perf 2560 1440 1706 960 {2 * perf_frames} fast plain")
        job += ["env WG_RR_ALIAS 0", "perf 2560 1440 1706 960 30 exact plain", "perf 2560 1440 1706 960 30 exact ext",
                "env WG_RR_ALIAS -"]   # memory without the lifetime aliasing (the work buffers one after the other)
    job.append("end")
    jf = os.path.join(work, "job.txt")
    with open(jf, "w") as fh:
        fh.write("\n".join(job) + "\n")
    host = os.path.join(out, "rr_test_host.exe")
    t0 = time.time()
    r = subprocess.run(_locked([host, jf]), capture_output=True, text=True,
                       timeout=7200)
    with open(os.path.join(work, "host.log"), "w") as fh:
        fh.write(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"rr_test_host failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    print(f"GPU run {time.time() - t0:.1f} s", flush=True)
    return work


# ------------------------------------------------------------------------------------------------- reference
_M = {}


def model(stage):
    if "m" not in _M:
        with open(os.path.join(stage, "model_logical.pkl"), "rb") as fh:
            _M["m"] = pickle.load(fh)
    return _M["m"]


def parse_map(s):
    m, tex, surf = {}, {}, {}
    for line in s.split("|"):
        if not line:
            continue
        k, *v = line.split()
        if k.startswith("tex."):
            tex[k[4:]] = [int(x) for x in v]
        elif k.startswith("surf."):
            surf[k[5:]] = [int(x) for x in v]
        else:
            m[k] = int(v[0])
    return m, tex, surf


def view(img, off, shape, dtype):
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    return img[off:off + n].copy().view(dtype).reshape(shape)


def f16(a):
    return np.asarray(a).view(np.float16).astype(np.float64)


def steps16(a, b):
    """distance in binary16 steps (sign-magnitude order), NaN == NaN -> 0, NaN vs number -> large."""
    def key(x):
        u = np.asarray(x, np.float16).view(np.uint16).astype(np.int64)
        return np.where(u & 0x8000, -(u & 0x7fff), u)
    a16, b16 = np.asarray(a, np.float16), np.asarray(b, np.float16)
    d = np.abs(key(a16) - key(b16))
    na, nb = np.isnan(a16), np.isnan(b16)
    d = np.where(na & nb, 0, d)
    d = np.where(na ^ nb, 1 << 20, d)
    return d


def same_bits(a, b):
    from clean.rr_f.gpu.resolve import same_bits as sb
    return sb(np.ascontiguousarray(a), np.ascontiguousarray(b))


def scalar_words(k, S, frame):
    """Words 0 .. K_TEX-1 of the input stage's constant block as clean/rr_f/gpu/input_stage.py packs them (the
    reference the runtime's packing, clean/rr_f/host/rr_input_host.h, must equal)."""
    from clean.rr_f.gpu import input_stage as GIS
    return GIS.scalar_words(k, S, {r for r in ("C", "Z", "V", "SV", "Dd", "Ds", "N", "R", "Q", "B", "G1", "G2", "Hd")
                                   if frame.attached(r)})


def tolerance_close(a, b, rel=2e-3, ab=1e-6):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= ab + rel * np.abs(b)) | (a == b)
    return ok


def resolve_symbols(G, K):
    from clean.rr_f.reference import resolve as RR
    k32 = lambda n: np.float32(np.asarray(K[n]).reshape(-1)[0])   # noqa: E731
    return RR.ResolveSymbols(**{n: np.float16(np.asarray(K[n], np.float64).reshape(-1)[0]) for n in
                                ("RW_A", "RW_B", "RW_C", "RW_D", "RW_0")},
                             K1=np.float32(G["D0-R"]["K1"][0]),
                             **{n: k32(n) for n in ("K2n", "K2x2n", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13",
                                                    "C_HDEC", "T_EFL", "TBW")})


def frame_reference(args):
    """Stage-by-stage CPU reference of one completed frame. Returns the frame's report row."""
    (work, stage, sname, idx, meta, prev_meta, call, cp_prev, first) = args
    from clean.rr_f.reference import exposure as XR
    from clean.rr_f.reference import input_stage as IS
    from clean.rr_f.reference import network as NET
    from clean.rr_f.reference import output as OR
    from clean.rr_f.reference import resolve as RR
    from clean.rr_f.reference.scalar import WindowConsts
    from clean.rr_f.gpu import exposure as GX
    from clean.rr_f.gpu import input_stage as GIS
    from clean.rr_f.gpu import stage as GST
    from clean.rr_f.gpu.output import tolerated_half
    from clean.rr_f.tests.test_output import symbols as out_symbols
    m = model(stage)
    G, K = m["groups"], m["constants"]
    k32 = lambda n: np.float32(np.asarray(K[n]).reshape(-1)[0])   # noqa: E731
    sc = SCEN[sname]
    run = os.path.join(work, f"run{sname}0")
    img = np.fromfile(os.path.join(run, f"frame_{idx:03d}.arena"), np.uint8)
    prev = np.fromfile(os.path.join(run, f"frame_{idx - 1:03d}.arena"), np.uint8) if idx > 0 else None
    L = meta["layout"]
    lay = dict(model=L[0], ch=(L[1], L[2]), ph1=L[3], ph2=L[4], ex=(L[5], L[6]), hid=L[8], pm=(L[9], L[10]), pe=L[11],
               pn=L[12], ah=(L[16], L[17]), th=(L[18], L[19]), oh=(L[20], L[21]))
    wmap, tmap, smap = parse_map(meta["map"])
    wr, hr = meta["render"]
    O = sc["output"]
    wo, ho = O
    we, he = wo + wo % 2, ho + ho % 2
    wh, hh = 3 * we // 2, 3 * he // 2
    wt, ht = 32 * -(-wr // 32), 32 * -(-hr // 32)
    par, fc, reset = meta["parity"], meta["counter"], bool(meta["reset"])
    fresh = first
    rebuild = bool(meta["rebuild"]) or fresh
    row = dict(scenario=sname, frame=idx, render=[wr, hr], reset=reset, parity=par, counter=fc,
               rebuild=bool(meta["rebuild"]), checks={})
    chk = row["checks"]

    def ok(name, passed, **info):
        chk[name] = dict(passed=bool(passed), **info)

    # ---- start-of-frame state (what the GPU carried; zero-filled on creation / render-size change, 53.8 / 53.9)
    PL = prev_meta["layout"] if prev_meta is not None else None

    def pstate(name, shape, dtype, zero):
        if zero or prev is None:
            return np.zeros(shape, dtype)
        poff = dict(ch0=PL[1], ch1=PL[2], ph1=PL[3], ph2=PL[4], ex0=PL[5], ex1=PL[6], hid=PL[8], pm0=PL[9], pm1=PL[10],
                    pe=PL[11], pn=PL[12], ah0=PL[16], ah1=PL[17], th0=PL[18], th1=PL[19], oh0=PL[20], oh1=PL[21])[name]
        if poff == 0:
            return None
        return view(prev, poff, shape, dtype)
    s_ph1 = pstate("ph1", ((hh + 1) // 2, (wh + 1) // 2, 4), np.uint16, fresh)
    s_ph2 = pstate("ph2", ((hh + 3) // 4, (wh + 3) // 4, 4), np.uint16, fresh)
    s_ex = pstate(f"ex{1 - par}", (4,), np.float32, fresh)
    s_ch = pstate(f"ch{par}", (hh, wh, 4), np.uint16, rebuild)
    s_hid = pstate("hid", (hr, wr, 4), np.int8, rebuild)
    s_pmr = pstate(f"pm{par}", (hr, wr, 4), np.uint16, rebuild)
    s_pmw = pstate(f"pm{1 - par}", (hr, wr, 4), np.uint16, rebuild)
    s_pe = pstate("pe", (hr, wr, 2), np.uint16, rebuild)
    s_pn = pstate("pn", (hr, wr, 2), np.uint16, rebuild)
    # TH / OH: created (zero-filled) on their first frame with a layer / opacity layer, kept afterwards (also across
    # render-size changes); AH: from creation with alpha upscaling (zero-filled)
    hist_state = {}
    for k_, chn in (("th", 4), ("oh", 4), ("ah", 1)):
        if k_ == "ah" and fresh:
            hist_state[k_] = [np.zeros((he, we, 1), np.float16) for _ in range(2)] if sc["alpha"] else None
            continue
        pair = [pstate(f"{k_}{i}", (he, we, chn), np.float16, False) for i in range(2)] if prev is not None else [None]
        hist_state[k_] = None if any(p is None for p in pair) else pair

    # ---- 1. copies of the game's inputs the kernels cannot read directly: linear conversion (input stage), the
    #         resolve's layer / opacity / alpha source in a 54b.N read format; every other input read through its SRV
    bad, roles = 0, []
    for r, (off, w, h, ch, ox, oy, direct) in tmap.items():
        if direct:
            continue
        roles.append(r)
        src = np.ascontiguousarray(call["tex"][r]["values"][:h, :w, :ch], np.float32)
        got = view(img, off, (h, w, ch), np.float32)
        bad += int(np.count_nonzero(got.view(np.uint32) != src.view(np.uint32)))
    for key, k_role in (("layer", "LAYER"), ("opac", "OPACITY"), ("asrc", None)):
        if key not in smap:
            continue
        w, h, fmt_, ch, ox, oy = smap[key][:6]
        if key == "asrc":
            role = smap[key][6]
            if role == ROLE["INPUT_ALPHA"]:
                kind, data = call["extra"]["INPUT_ALPHA"]["surf"]
                want = RR.read(RR.Surface(data, kind))[..., :1].astype(np.float32)
            else:
                want = np.ascontiguousarray(call["tex"]["C"]["values"][..., 3:4], np.float32)
            if fmt_ == 1:
                want = want.astype(np.float16)
        else:
            kind, data = call["extra"][k_role]["surf"]
            if kind == "rgb10a2":
                c_ = data.astype(np.uint32)
                want = (c_[..., 0] | (c_[..., 1] << 10) | (c_[..., 2] << 20) | (c_[..., 3] << 30)).astype(np.uint32)
                want = want.reshape(h, w, 1)
            elif kind == "unorm8":
                want = np.ascontiguousarray(data, np.uint8)
            elif kind == "f16":
                want = np.ascontiguousarray(data, np.float16)
            else:
                want = np.ascontiguousarray(data, np.float32)
        roles.append(key)
        got = view(img, wmap[key], want.shape, want.dtype)
        okb, nb = same_bits(got, want)
        bad += nb
    ok("copies", bad == 0, mismatches=bad, roles=roles,
       direct=sorted(r for r, v in tmap.items() if v[6]))

    # ---- 2. PH2 quarter pool (54b.5): bit-exact; unchanged on reset frames
    g_ph2 = view(img, lay["ph2"], ((hh + 3) // 4, (wh + 3) // 4, 4), np.uint16)
    rsym = resolve_symbols(G, K)
    if reset:
        ok("ph2_pool", np.array_equal(g_ph2, s_ph2), rule="reset frame: PH2 not produced (53.2)")
    else:
        want = RR.quarter_pool(s_ph1, s_ph2, output_size=O, reset=False, symbols=rsym)
        okb, nb = same_bits(g_ph2, want)
        ok("ph2_pool", okb, rule="bit-exact (54b.5)", mismatches=nb, values=int(want.size))

    # ---- 3. exposure (58) with the previous frame's result as Eprev
    ex = view(img, lay["ex"][par], (9,), np.uint32)
    gE, gEp, gInv, gX3 = (np.float32(v) for v in ex[:4].view(np.float32))
    colour = call["tex"]["C"]["values"]
    Eprev = np.float32(s_ex[0]) if s_ex is not None else np.float32(0)
    XS = XR.ExposureSymbols(*(k32(n) for n in ("X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn",
                                                "X_NRM")))
    xr = XR.auto_exposure(colour, symbols=XS, render_size=(wr, hr), Eprev=Eprev, reset=reset,
                          origin=call["tex"]["C"]["origin"])
    q = GX.qualify(img, [("exposure", lay["ex"][par], xr, GX.exact_mean(colour, (wr, hr), call["tex"]["C"]["origin"], XS),
                          reset, XS)], XS)[0]
    rules = {k: v for k, v in q["rules"].items() if k != "guard"}   # the runtime's result slot has no guard pattern
    exact = bool(np.array_equal(ex[:4], np.float32([xr.E, xr.Ep, xr.invE, xr.X3]).view(np.uint32)))
    ok("exposure", all(rules.values()), rule="page 58 (gpu/exposure.qualify)", exact=exact, rules=rules,
       E=[float(gE), float(xr.E)], X3=[float(gX3), float(xr.X3)])

    # ---- 4. input stage (52) with the GPU's exposure block and PH2, the carried PH1 / HID / PM
    S_vals = {n: k32(n) for n in IS.SYMBOLS_F32}
    S_vals.update({n: np.float32(np.asarray(K[n], np.float64).reshape(-1)[0]) for n in IS.SYMBOLS_F16})
    b2 = m.get("constants_b2")
    if b2:
        S_vals.update({n: np.float32(np.asarray(b2[n]).reshape(-1)[0]) for n in IS.SYMBOLS_B2})
    S = IS.InputSymbols(S_vals, bool(b2))
    tex = {r: IS.Texture(t["values"], t["origin"]) for r, t in call["tex"].items()}
    frame = IS.InputFrame(tex, PH1=s_ph1, PH2=g_ph2, HID=s_hid, PM_read=s_pmr.view(np.float16),
                          PM_write=s_pmw.view(np.float16), Pe=s_pe.view(np.float16), Pn=s_pn.view(np.float16))
    SH = np.asarray(K["SH"], np.uint32).reshape(2, 11, 2)
    if reset or cp_prev is None:
        V2Cp, W2Vp = call["V2C"], call["W2V"]
    else:
        V2Cp, W2Vp = cp_prev
    params = IS.FrameParams(render_size=(wr, hr), output_size=O, jitter=tuple(np.float32(v) for v in meta["jitter"]),
                            prev_jitter=tuple(np.float32(v) for v in meta["prev_jitter"]), mv_scale_raw=call["mv_scale"],
                            mv_jittered=True, E=gE, Ep=gEp, X3=gX3, reset=reset, par=par, fc=fc, SH=SH, hw_depth=True,
                            depth_inverted=False, nrm_dec=False, rgh_packed=False, V2C=call["V2C"], W2V=call["W2V"],
                            V2Cp=V2Cp, W2Vp=W2Vp)
    kk = IS.frame_constants(params)
    words_ref = scalar_words(kk, S, frame)
    words_gpu = view(img, wmap["blk"], (256,), np.uint32)
    idxs = [i for i in range(100) if i not in (10, 12, 37)]      # E, X3, rcp(Ep) come from the GPU exposure block
    wbad = [i for i in idxs if words_ref[i] != words_gpu[i]]
    ok("input_constants", not wbad, differing_words=wbad, camera_usable=bool(kk.camera_usable),
       m2_mode=GIS.m2_mode(frame, kk))
    feats = IS.compute_features(frame, kk, S, G["IN0"]["D"])
    wc = WindowConsts(**{n: float(np.asarray(K[n], np.float64).reshape(-1)[0]) for n in
                         ("A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1", "ACT_C2", "ACT_C3")})
    x0_ref, _ = IS.front_end(feats["codes"], IS.fe_tensors(G["IN0"]), wc)
    g_codes = view(img, wmap["codes"], (ht, wt, 64), np.uint8)
    g_hold = view(img, wmap["hold"], (ht, wt, 4), np.uint16)
    g_x0 = view(img, wmap["x0"], (ht, wt, 32), np.uint8)
    g_pm = view(img, lay["pm"][1 - par], (hr, wr, 4), np.uint16)
    g_pe = view(img, lay["pe"], (hr, wr, 2), np.uint16)
    g_pn = view(img, lay["pn"], (hr, wr, 2), np.uint16)
    b16 = lambda a: np.asarray(a, np.float16).view(np.uint16)   # noqa: E731
    from clean.rr_f.reference.fmt import to_f16_bits
    diffs = dict(codes=int(np.count_nonzero(g_codes != feats["codes"])),
                 hold=int(np.count_nonzero(g_hold != to_f16_bits(feats["hold"]))),
                 x0=int(np.count_nonzero(g_x0 != x0_ref)),
                 PM=int(np.count_nonzero(g_pm != b16(feats["surfaces"]["PM"]))),
                 Pe=int(np.count_nonzero(g_pe != b16(feats["surfaces"]["Pe"]))),
                 Pn=int(np.count_nonzero(g_pn != b16(feats["surfaces"]["Pn"]))))
    ok("input_stage", not any(diffs.values()), rule="bit-exact (52)", **diffs)

    # ---- 5. network body (51) and head (55), from the GPU's x0 and hold
    W, consts = {}, {}
    order = NET.ORDER
    for name in order:
        g = G[name]
        W[name] = {k: (np.asarray(v, np.uint8) if np.asarray(v).dtype == np.uint8 else np.asarray(v, np.float64))
                   for k, v in g.items() if not k.startswith("FE.") and k not in ("D", "g", "s1", "s2", "K1")}
        c = dict(ra=float(K[f"ra.{name}"][0]), m=float(K[f"m.{name}"][0]))
        if name.endswith("-F"):
            c["f"] = float(K[f"f.{name[:2]}"][0])
        if "g" in g:
            c["g"] = float(np.asarray(g["g"]).reshape(-1)[0])
        consts[name] = c
    rows_ = ("IN0", "E1", "E2", "E3", "E4", "E5", "D4", "D3", "D2", "D1", "D0")
    shifts = {n: (int(SH[par, i, 0]), int(SH[par, i, 1])) for i, n in enumerate(rows_)}
    s1, s2 = np.float32(G["D0-R"]["s1"][0]), np.float32(G["D0-R"]["s2"][0])
    dump_dir = os.environ.get("WG_E2E_DUMP_HEAD")    # head/trace_e.py inputs: d0r_stage's arguments per frame
    if dump_dir:
        import pickle
        from clean.rr_f.reference import network as _netmod
        _d0r = _netmod.d0r_stage

        def _dump_d0r(x, W_, k, ra, m, g, shift, hold, hc, s1_, s2_, wr_, hr_, *r, **kw):
            with open(os.path.join(dump_dir, f"head_{sname}_{idx:03d}.pkl"), "wb") as fh:
                pickle.dump(dict(x=x, W=W_, k=k, ra=ra, m=m, g=g, shift=shift, hold=hold, hc=hc, s1=s1_, s2=s2_,
                                 wr=wr_, hr=hr_), fh)
            return _d0r(x, W_, k, ra, m, g, shift, hold, hc, s1_, s2_, wr_, hr_, *r, **kw)
        _netmod.d0r_stage = _dump_d0r
    body = NET.run_body(g_x0, f16(g_hold), W, consts, wc, shifts, s1=s1, s2=s2, render_size=(wr, hr))
    sdiff = {}
    from clean.rr_f.reference.stages import stage_info
    for name in order[:-1]:
        lvl = 0 if name in ("IN0", "D0-F") else int(name[1])
        nx, ny = wt >> lvl, ht >> lvl
        ref = body["stages"][name]
        info = stage_info(name)
        got = dict(y2=view(img, wmap[f"{name}.y2"], (ny, nx, info["C"]), np.uint8))
        if info["Cn"]:
            got["merged"] = view(img, wmap[f"{name}.merged"], (ny // 2, nx // 2, info["Cn"]), np.uint8)
        if info["hidden"]:
            got["hnew"] = view(img, wmap[f"{name}.hnew"], (ny, nx, 4), np.uint16)
        if info["merged_hidden"]:
            got["hmerged"] = view(img, wmap[f"{name}.hmerged"], (ny // 2, nx // 2, 4), np.uint16)
        for kname, v in got.items():
            sdiff[f"{name}.{kname}"] = GST.compare(kname, v, ref[kname])["diff"]
    nz = {k: v for k, v in sdiff.items() if v}
    ok("network_body", not nz, rule="bit-exact (51.8 item 1, exact row sums)", stage_outputs=len(sdiff), differing=nz)
    hd = body["head"]
    g_cnet = f16(view(img, wmap["cnet"], (ht, wt, 4), np.uint16))
    g_prec_u = view(img, wmap["prec"], (hr, wr, 4), np.uint16)
    g_prec = f16(g_prec_u)
    g_gates = view(img, wmap["gates"], (hr, wr, 4), np.uint8)
    g_candw = view(img, wmap["candw"], (hr, wr, 4), np.uint8)
    g_hid = view(img, lay["hid"], (hr, wr, 4), np.int8)
    st_c = steps16(g_cnet, hd["Cnet"])
    st_p = steps16(g_prec, hd["PREC"])
    cg = np.abs(g_gates.astype(int) - np.asarray(hd["GATES_code"]).astype(int))
    cc = np.abs(g_candw.astype(int) - np.asarray(hd["CANDW_code"]).astype(int))
    ch_ = np.abs(g_hid.astype(int) - np.asarray(hd["HID_code"]).astype(int))
    head_ok = int(st_c.max()) <= 1 and int(st_p.max()) <= 1 and int(cg.max()) <= 1 and int(cc.max()) <= 1 and int(ch_.max()) <= 1
    ok("head", head_ok, rule="55.3 tolerances: Cnet / PREC <= 1 f16 step, GATES / CANDW / HID codes <= 1",
       cnet_exact=int(np.sum(st_c == 0)), cnet_n=int(st_c.size), prec_exact=int(np.sum(st_p == 0)), prec_n=int(st_p.size),
       gates_exact=int(np.sum(cg == 0)), candw_exact=int(np.sum(cc == 0)), hid_exact=int(np.sum(ch_ == 0)),
       codes_n=int(cg.size), max=[int(st_c.max()), int(st_p.max()), int(cg.max()), int(cc.max()), int(ch_.max())])

    # ---- 6. output stage (54) from the GPU's head buffers, PM[1-par] / Pe / Pn of this frame, the carried CH[par]
    oc = out_symbols(TR=np.asarray(K["TR"], np.float32), K1=np.float32(G["D0-R"]["K1"][0]),
                     **{n: k32(n) for n in ("K2n", "K2x2n", "K3", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13")})
    oin = dict(Cnet=g_cnet, G=OR.decode_unorm8(g_gates)[..., :3], P=g_prec[..., :3], W4=OR.decode_unorm8(g_candw),
               PM=f16(g_pm), Pe=f16(g_pe), Pn=f16(g_pn), Hprev=f16(s_ch), output_size=O,
               jitter=tuple(np.float32(v) for v in meta["jitter"]), E=gE, Ep=gEp, invE=gInv, reset=reset, constants=oc)
    o = OR.output_stage(**oin)
    g_ch = view(img, lay["ch"][1 - par], (hh, wh, 4), np.uint16).view(np.float16)
    g_q = view(img, wmap["q"], (hh, wh, 3), np.uint16).view(np.float16)
    g_ph1 = view(img, lay["ph1"], ((hh + 1) // 2, (wh + 1) // 2, 4), np.uint16)
    out_rows = {}
    for kname, v, t in (("CH", g_ch, o.CH), ("Q", g_q, o.Q)):
        good = tolerated_half(v, t)
        out_rows[kname] = int(np.count_nonzero(~good))
        out_rows[kname + "_not_bit_identical"] = int(np.count_nonzero(~same_mask(v, t)))
    dph = np.abs(g_ph1.astype(int) - o.PH1.astype(int))
    out_rows["PH1_over_1"] = int(np.count_nonzero(dph > 1))
    passed = out_rows["CH"] * 1000 <= g_ch.size and out_rows["Q"] * 1000 <= g_q.size and out_rows["PH1_over_1"] == 0
    ok("output_stage", passed, rule="54 qualification: CH / Q within 2 f16 steps or 2^-9 (<= 0.1 % outside), PH1 +-1 code",
       outside=out_rows, values=int(g_ch.size + g_q.size + g_ph1.size))

    # ---- 7. resolve (54b): the reference on the GPU's Q / PM[1-par] / PREC / GATES / exposure and the carried
    #         TH / OH / AH; the output textures as the game sees them (background outside the stores)
    tw, th_, tf = meta["out_tex"]
    if tf == DX["RGBA32F"]:
        bg = np.full((th_, tw, 4), np.frombuffer(bytes([0x3c] * 4), np.float32)[0], np.float32)
        out_img = np.fromfile(os.path.join(run, f"frame_{idx:03d}.out"), np.float32).reshape(th_, tw, 4)
        cout = RR.Surface(bg, "f32", OUT_ORIGIN)
    else:
        bg = np.full((th_, tw, 4), np.frombuffer(bytes([0x3c] * 2), np.float16)[0], np.float16)
        out_img = np.fromfile(os.path.join(run, f"frame_{idx:03d}.out"), np.float16).reshape(th_, tw, 4)
        cout = RR.Surface(bg, "f16", OUT_ORIGIN)
    alpha = bool(sc["alpha"])
    ex_ = call["extra"]
    aout = None
    if alpha and "OUTPUT_ALPHA" in ex_:
        a = ex_["OUTPUT_ALPHA"]
        aout = RR.Surface(np.ascontiguousarray(a["store"]), "f16", a["origin"])
    surf = lambda key: RR.Surface(np.ascontiguousarray(ex_[key]["surf"][1]), ex_[key]["surf"][0], ex_[key]["origin"])  # noqa: E731
    layer = RR.Surface(np.ascontiguousarray(ex_["LAYER"]["surf"][1]), ex_["LAYER"]["surf"][0], call["layer_origin"]) \
        if "LAYER" in ex_ else None
    opac = surf("OPACITY") if ("OPACITY" in ex_ and layer is not None) else None
    alpha_in = surf("INPUT_ALPHA") if (alpha and "INPUT_ALPHA" in ex_) else None
    colour_in = RR.Surface(np.ascontiguousarray(call["tex"]["C"]["values"], np.float32), "f32", call["tex"]["C"]["origin"]) \
        if alpha else None
    H = RR.ResolveHistories(O, alpha_upscaling=alpha, TH=hist_state["th"], OH=hist_state["oh"],
                            AH=(hist_state["ah"] if hist_state["ah"] is not None else
                                [np.zeros((he, we, 1), np.float16) for _ in range(2)]) if alpha else None,
                            ah_created=fresh and alpha)
    res = RR.resolve(f16(g_q.view(np.uint16)), output_size=O, render_size=(wr, hr), E=gE, Ep=gEp,
                     jitter=tuple(np.float32(v) for v in meta["jitter"]), reset=reset, parity=par, symbols=rsym,
                     colour_out=cout, histories=H, alpha_out=aout, layer=layer, opacity=opac, alpha_in=alpha_in,
                     colour_in=colour_in, motion=f16(g_pm), P=g_prec, G=g_gates[..., :3],
                     bounded_stores=bool(call["settings"] & 2))
    rchecks = {}
    okb, nb = same_bits(out_img, res.colour.data)
    rchecks["output"] = dict(values=int(out_img.size), mismatches=nb)
    if aout is not None:
        a_img = np.fromfile(os.path.join(run, f"frame_{idx:03d}.aout"), np.float16).reshape(aout.data.shape)
        okb, nb = same_bits(a_img, res.alpha.data)
        rchecks["output_alpha"] = dict(values=int(a_img.size), mismatches=nb)
    for k_, chn, attr in (("th", 4, "TH"), ("oh", 4, "OH"), ("ah", 1, "AH")):
        want = getattr(res.histories, attr)
        if want is None:
            if lay[k_][0]:
                rchecks[attr] = dict(created_without_reference=True, mismatches=1)
            continue
        if not lay[k_][0]:
            rchecks[attr] = dict(missing=True, mismatches=1)
            continue
        for i in range(2):
            got = view(img, lay[k_][i], (he, we, chn), np.float16)
            okb, nb = same_bits(got, want[i])
            rchecks[f"{attr}[{i}]"] = dict(values=int(got.size), mismatches=nb)
    rok = all(v["mismatches"] == 0 for v in rchecks.values())
    ok("resolve", rok, rule="bit-exact (54b, reference/resolve.py)", variant=res.variant, flags=wmap.get("rflags"),
       checks=rchecks)

    # ---- 8. whole-frame chain: CPU from the frame's start state only (its own exposure, PH2, input, body, head,
    #         output stage); informative
    try:
        params_c = IS.FrameParams(**{**params.__dict__, "E": xr.E, "Ep": xr.Ep, "X3": xr.X3})
        ph2_c = s_ph2 if reset else RR.quarter_pool(s_ph1, s_ph2, output_size=O, reset=False, symbols=rsym)
        frame_c = IS.InputFrame(tex, PH1=s_ph1, PH2=ph2_c, HID=s_hid, PM_read=s_pmr.view(np.float16),
                                PM_write=s_pmw.view(np.float16), Pe=s_pe.view(np.float16), Pn=s_pn.view(np.float16))
        kc = IS.frame_constants(params_c)
        fc_ = IS.compute_features(frame_c, kc, S, G["IN0"]["D"])
        x0c, _ = IS.front_end(fc_["codes"], IS.fe_tensors(G["IN0"]), wc)
        bc = NET.run_body(x0c, fc_["hold"], W, consts, wc, shifts, s1=s1, s2=s2, render_size=(wr, hr))
        hc_ = bc["head"]
        oc_in = dict(oin, Cnet=hc_["Cnet"], G=OR.decode_unorm8(np.asarray(hc_["GATES_code"], np.uint8))[..., :3],
                     P=np.asarray(hc_["PREC"])[..., :3], W4=OR.decode_unorm8(np.asarray(hc_["CANDW_code"], np.uint8)),
                     PM=np.asarray(fc_["surfaces"]["PM"], np.float64), Pe=np.asarray(fc_["surfaces"]["Pe"], np.float64),
                     Pn=np.asarray(fc_["surfaces"]["Pn"], np.float64), E=xr.E, Ep=xr.Ep, invE=xr.invE)
        oc_ = OR.output_stage(**oc_in)
        hid_d = np.abs(g_hid.astype(int) - np.asarray(hc_["HID_code"]).astype(int))
        row["chain"] = dict(q_within_1pct=float(np.mean(tolerance_close(g_q, oc_.Q, rel=1e-2, ab=1e-4))),
                            x0_equal=float(np.mean(g_x0 == x0c)), hid_codes_within_1=float(np.mean(hid_d <= 1)),
                            ch_within_tol=float(np.mean(tolerated_half(g_ch, oc_.CH))))
    except Exception as e:      # the chain is informative; it never decides pass/fail
        row["chain"] = dict(error=repr(e))
    row["passed"] = all(v["passed"] for v in chk.values())
    return row


def same_mask(a, b):
    a, b = np.asarray(a, np.float16), np.asarray(b, np.float16)
    return (a.view(np.uint16) == b.view(np.uint16)) | (np.isnan(a) & np.isnan(b))


def lifecycle_checks(metas, calls, work, sname):
    """Page 53 bookkeeping as the runtime saw it, and the failed call's state invariance."""
    rows = []
    par = fc = 0
    pending = True
    built = RENDER_A
    for m, c in zip(metas, calls):
        expect_reset = pending or bool(c["reset"])
        size = tuple(m["render"])
        rebuild = size != built
        if c["fail_role"] is not None:
            rows.append(dict(frame=m["frame"], failed=True, rc=m["rc"], passes=m["passes"],
                             ok=m["rc"] != 0 and m["passes"] == 0 and m["jitter_consumed"] == 0))
            if rebuild:
                built = size
                pending = True
            continue
        if rebuild:
            built = size
        good = (m["parity"] == par and m["counter"] == fc and bool(m["reset"]) == expect_reset and m["rc"] == 0
                and m["passes"] == 1 and bool(m["rebuild"]) == rebuild)
        rows.append(dict(frame=m["frame"], parity=m["parity"], counter=m["counter"], reset=bool(m["reset"]),
                         rebuild=bool(m["rebuild"]), ok=good))
        if m["rc"] == 0:
            par ^= 1
            fc = (fc + 1) % 8
            if expect_reset:
                pending = False
            if rebuild:
                pending = True
    # the failed call changed no state: its dump's state equals the previous dump's
    fails = [m for m, c in zip(metas, calls) if c["fail_role"] is not None]
    inv = []
    for m in fails:
        i = m["frame"]
        a = np.fromfile(os.path.join(work, f"run{sname}0", f"frame_{i:03d}.arena"), np.uint8)
        b = np.fromfile(os.path.join(work, f"run{sname}0", f"frame_{i - 1:03d}.arena"), np.uint8)
        end = m["layout"][13]
        inv.append(dict(frame=i, state_unchanged=bool(np.array_equal(a[:end], b[:end]))))
    return rows, inv


def alias_checks(work, sname, metas0, metas1):
    """The production run (work buffers aliased by lifetime) against the unaliased run: outputs and every state image
    (the arena up to the end of the render-size items) identical after every frame."""
    rows = []
    for m0, m1 in zip(metas0, metas1):
        i = m0["frame"]
        d0, d1 = os.path.join(work, f"run{sname}0"), os.path.join(work, f"run{sname}1")
        end = m0["layout"][13]
        same_layout = m0["layout"][:14] == m1["layout"][:14] and m0["layout"][16:22] == m1["layout"][16:22]
        a0 = np.fromfile(os.path.join(d0, f"frame_{i:03d}.arena"), np.uint8)[:end]
        a1 = np.fromfile(os.path.join(d1, f"frame_{i:03d}.arena"), np.uint8)[:end]
        outs = all(open(os.path.join(d0, f"frame_{i:03d}{e}"), "rb").read() ==
                   open(os.path.join(d1, f"frame_{i:03d}{e}"), "rb").read()
                   for e in (".out", ".aout") if os.path.exists(os.path.join(d0, f"frame_{i:03d}{e}")))
        rows.append(dict(frame=i, layout=same_layout, state=bool(np.array_equal(a0, a1)), outputs=outs,
                         aliased=bool(m1["layout"][23]), work_unaliased=m0["layout"][22], work_aliased=m1["layout"][22],
                         ok=same_layout and bool(np.array_equal(a0, a1)) and outs and bool(m1["layout"][23])))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--calls", type=int, default=61)
    ap.add_argument("--calls-b", type=int, default=14)
    ap.add_argument("--skip-gpu", action="store_true", help="analyse an earlier run")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--perf", type=int, default=600)
    ap.add_argument("--leak", type=int, default=6)
    ap.add_argument("--body", default="Z", help="WG_RR_BODY for the GPU run (default Z: exact body + exact front-end MLP, "
                    "checked bit-exact; 'default': the runtime's default step E, which the exact body check flags)")
    ap.add_argument("--head", default="exact", help="WG_RR_HEAD for the GPU run (exact / fast / strict; default exact: since "
                    "clean/rr-f-speed8 the runtime's default fast head is relaxed (RR_HVARL), which the 55.3 check flags; "
                    "'default': the runtime's)")
    ap.add_argument("--input", default="exact", help="WG_RR_INPUT for the GPU run (default exact; 'default': the runtime's)")
    ap.add_argument("--name", help="work folder name under --out (default e2e[-model])")
    ap.add_argument("--only", help="comma-separated frame indices to analyse")
    ap.add_argument("--model", choices=("synthetic", "admission", "wgcm"), default="synthetic",
                    help="admission: the admitted synthetic known subset; wgcm: the converter's model.wgcm (--wgcm)")
    ap.add_argument("--wgcm", default=DEFAULT_WGCM)
    ap.add_argument("--mips", type=int, default=1, help="the game's input textures with this many mips (0.6.0 follow-up; "
                    "mip 0 holds the data and is what the runtime reads: every check must pass as with 1)")
    a = ap.parse_args(argv)
    suffix = "" if a.model == "synthetic" else "-" + a.model
    stage = os.path.join(a.out, "stage" + suffix)
    if not a.skip_gpu:
        from clean.rr_f.runtime.tools import build as B
        B.stage_folder(a.out, stage, seed=5100, admission=a.model == "admission",
                       wgcm=a.wgcm if a.model == "wgcm" else None)
    runs = [("A", scenario("A", a.calls))]
    if a.calls_b:
        runs.append(("B", scenario("B", a.calls_b)))
    name = a.name or "e2e" + suffix
    work = os.path.join(a.out, name)
    if not a.skip_gpu:
        work = run_gpu(a.out, stage, runs, a.perf, name, a.leak, None if a.head == "default" else a.head,
                       None if a.body == "default" else a.body, None if a.input == "default" else a.input, mips=a.mips)
    rows, life, inv, al = [], [], [], []
    jobs = []
    metas = {}
    for sname, calls in runs:
        m0 = [json.loads(l.replace(chr(92), "/")) for l in open(os.path.join(work, f"run{sname}0", "meta.jsonl"))]
        m1 = [json.loads(l.replace(chr(92), "/")) for l in open(os.path.join(work, f"run{sname}1", "meta.jsonl"))]
        metas[sname] = (m0, m1)
        cp = None
        first = True
        for m, c in zip(m0, calls):
            if m["completed"]:
                if a.only is None or str(m["frame"]) in a.only.split(","):
                    jobs.append((work, stage, sname, m["frame"], m, m0[m["frame"] - 1] if m["frame"] else None, c, cp,
                                 first))
                cp = (c["V2C"], c["W2V"])
                first = False
        lr, li = lifecycle_checks(m0, calls, work, sname)
        life += [dict(r, scenario=sname) for r in lr]
        inv += [dict(r, scenario=sname) for r in li]
        al += [dict(r, scenario=sname) for r in alias_checks(work, sname, m0, m1)]
    t0 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(a.jobs) as pool:
            rows = pool.map(frame_reference, jobs, chunksize=1)
    else:
        rows = [frame_reference(j) for j in jobs]
    print(f"CPU reference {time.time() - t0:.0f} s", flush=True)
    report = dict(frames=rows, lifecycle=life, failed_call_state=inv, aliasing=al, model=a.model)
    p = os.path.join(work, f"run{runs[-1][0]}1", "leak.json")
    if os.path.exists(p):
        report["leak"] = json.load(open(p))
    p = os.path.join(work, f"run{runs[-1][0]}1", "perf.json")
    if os.path.exists(p):
        report["perf"] = [json.loads(l) for l in open(p) if l.strip()]
    hl = os.path.join(work, "host.log")
    report["host"] = open(hl).read().splitlines()[:1] if os.path.exists(hl) else []
    log = os.path.join(stage, "wronggpu-rr.log")
    report["log"] = open(log, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(log) else []
    checks = sorted({k for r in rows for k in r["checks"]})
    summary = {k: dict(passed=sum(1 for r in rows if r["checks"].get(k, {}).get("passed")),
                       of=sum(1 for r in rows if k in r["checks"])) for k in checks}
    report["summary"] = summary
    report["passed"] = (all(r["passed"] for r in rows) and all(x["ok"] for x in life) and
                        all(x["state_unchanged"] for x in inv) and all(x["ok"] for x in al))
    with open(os.path.join(work, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    lines = ["| check | frames passed |", "|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['passed']} / {v['of']} |")
    lines.append(f"| lifecycle (par / fc / reset / rebuild / failed call) | {sum(x['ok'] for x in life)} / {len(life)} |")
    lines.append(f"| failed call leaves state unchanged | {sum(x['state_unchanged'] for x in inv)} / {len(inv)} |")
    lines.append(f"| aliased run = unaliased run (outputs, state) | {sum(x['ok'] for x in al)} / {len(al)} |")
    with open(os.path.join(work, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in rows:
        if not r["passed"]:
            print("FAIL", r["scenario"], "frame", r["frame"], {k: v for k, v in r["checks"].items() if not v["passed"]})
    for x in al:
        if not x["ok"]:
            print("ALIAS FAIL", x)
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
