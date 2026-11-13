# SPDX-License-Identifier: MIT
"""Build, run and check the D3D12 D0-R kernel (rr_d0r.hlsl) against the spec-55 test vectors, and time it.

    python -m clean.rr_f.head.dx12.run_dx12 build   --work <dir>        # kernels (4 variants) + rr_host.exe + RGA stats
    python -m clean.rr_f.head.dx12.run_dx12 prepare --work <dir>        # job files for every vector case + timing jobs
    python -m clean.rr_f.head.dx12.run_dx12 gpu --work <dir>
    python -m clean.rr_f.head.dx12.run_dx12 compare --work <dir>        # report.json + summary

(run from the worktree root with NumPy on PYTHONPATH). Only the gpu step uses the GPU; it runs every job once (the
timing jobs warm the GPU for 3 s, then time 200 dispatches).

Kernel variants: prod (production arithmetic, no extra writes), dbg (prod + diagnostic writes of y2h, Z and the f16
values of HID / GATES / CANDW), dbgz (dbg with the reference row sums Z supplied: 51.8 item 1, first run), head (no
block: y2h given, for the head-only cases). Comparison rules: the manifest (55.3 / 55.5 tolerances, codes = the store
rule applied to the kernel's own f16 values, CANDW exact requirements), plus: y2h bit for bit with the reference Z,
prod outputs identical to dbg outputs, the bytes after every image and every unwritten byte untouched.
"""
import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = os.path.dirname(HERE)
CLEAN = os.path.dirname(os.path.dirname(HEAD))
ROOT = os.path.dirname(CLEAN)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(CLEAN, "dx12"))

from clean.rr_f.head import block as B      # noqa: E402
from clean.rr_f.head import fp              # noqa: E402
from clean.rr_f.head import vectors as V    # noqa: E402
from clean.rr_f.head import head as Hd       # noqa: E402

VDIR = os.path.join(HEAD, "tests", "vectors")
f32 = np.float32
SENT = 0xA5          # sentinel byte of every output region before the run
GUARD = 256          # sentinel bytes after every output region
VARIANTS = {"prod": {}, "dbg": {"RR_DEBUG": 1}, "dbgz": {"RR_DEBUG": 1, "RR_ZSUP": 1},
            "head": {"RR_HEAD_ONLY": 1, "RR_DEBUG": 1}, "headprod": {"RR_HEAD_ONLY": 1}}
ARGS = ["p_x", "p_wf", "p_bias", "p_tb", "p_hold", "p_hc", "p_tab", "p_rho", "p_cnet", "p_hid", "p_gates", "p_prec",
        "p_candw", "p_dy", "p_dimg", "p_dz", "p_zs", "Wt", "Ht", "Wr", "Hr", "sx", "sy", "nwx", "k_as", "k_acl",
        "k_acln", "k_p1", "k_p0", "k_actcl", "k_actcln", "k_c1", "k_c2", "k_c3", "k_ra", "k_m", "k_g", "k_s1",
        "p_flag", "p_wpl"]
KNAMES = {"k_as": "A_SCALE", "k_acl": "A_CLAMP", "k_acln": "A_CLAMPn", "k_p1": "A_P1", "k_p0": "A_P0",
          "k_actcl": "ACT_CL", "k_actcln": "ACT_CLn", "k_c1": "ACT_C1", "k_c2": "ACT_C2", "k_c3": "ACT_C3",
          "k_ra": "ra", "k_m": "m"}
TIME_SIZE = (1728, 960, 1706, 960)   # 1440p Quality: render 1706 x 960, tensor rounded up to multiples of 32


# ------------------------------------------------------------------------------------------------ build
def build(work):
    import build_dx12 as bd
    os.makedirs(work, exist_ok=True)
    asm = bd.build_asm_tool(work)
    src = os.path.join(HERE, "rr_d0r.hlsl")
    inc = [os.path.join(CLEAN, "dx12", "kernels")]
    stats = {}
    for tag, d in VARIANTS.items():
        out = os.path.join(work, f"rr_{tag}.dxil")
        bd.compile_hlsl(src, "rr_d0r_head", out, {"WG_BLOCK": 128, "WG_WAVE": 32, **d}, includes=inc, asm_tool=asm)
        bd.rga_isa(out, os.path.join(work, "isa"), f"rr_{tag}")
        import csv
        with open(os.path.join(work, "isa", f"rr_{tag}.stats.csv"), newline="") as fh:
            row = list(csv.DictReader(fh))[0]
        stats[tag] = {k: int(row[k]) for k in ("USED_VGPRs", "USED_SGPRs", "USED_LDS_BYTES", "SCRATCH_MEM", "ISA_SIZE")}
        with open(os.path.join(work, "isa", f"rr_{tag}.isa.txt"), encoding="utf-8", errors="replace") as fh:
            isa = fh.read()
        stats[tag]["v_wmma"] = isa.count("v_wmma_f32_16x16x16_fp8_fp8")
        stats[tag]["v_fma_mix"] = isa.count("v_fma_mix")
        stats[tag]["stores"] = isa.count("buffer_store") + isa.count("tbuffer_store")
        print(tag, stats[tag], flush=True)
    exe = os.path.join(work, "rr_host.exe")
    r = subprocess.run(f'"{bd.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul 2>nul && cl /nologo /O2 /EHsc /std:c++17 '
                       f'"{os.path.join(HERE, "rr_host.cpp")}" /Fe"{exe}" /Fo"{os.path.join(work, "rr_host.obj")}" '
                       f'/link d3d12.lib dxgi.lib', shell=True, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(exe):
        raise SystemExit("host build failed:\n" + r.stdout + r.stderr)
    with open(os.path.join(work, "build.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


# ------------------------------------------------------------------------------------------------ layouts
def f16bits(v):
    return fp.bits16(np.asarray(v, np.float64))


def frags(W, kslices, ntiles):
    """A / B fragments of W [K][N] (e4m3 codes): [ntile][kslice][lane 32][8 bytes]; lane l, byte i =
    W[16 ks + 8 (l >> 4) + i][16 nt + (l & 15)]."""
    W = np.asarray(W, np.uint8)
    out = np.empty((ntiles, kslices, 32, 8), np.uint8)
    lane = np.arange(32)
    r, hf = lane & 15, lane >> 4
    for nt in range(ntiles):
        for ks in range(kslices):
            for i in range(8):
                out[nt, ks, :, i] = W[16 * ks + 8 * hf + i, 16 * nt + r]
    return out


def weight_buffer(W):
    parts = []
    for name in ("Wq", "Wv"):                                   # [h][ft][ks]
        for h in range(2):
            parts.append(frags(W[name][h], 2, 2))
    for h in range(2):                                          # Wo [h][ct][ks]
        parts.append(frags(W["Wo"][h], 2, 2))
    parts.append(frags(W["W1"], 2, 8))                          # [nt][ks]
    parts.append(frags(W["W2"], 8, 2))                          # [ct][ks]
    buf = np.concatenate([p.reshape(-1) for p in parts])
    assert buf.size == 56 * 256
    return buf.tobytes()


def bias_buffers(W):
    b = np.zeros(192, f32)
    b[0:32] = np.asarray(W["bo"], np.float64).astype(f32)
    b[32:160] = np.asarray(W["b1"], np.float64).astype(f32)
    b[160:192] = np.asarray(W["b2"], np.float64).astype(f32)
    tb = np.empty((2, 4, 4, 32, 8), f32)
    lane = np.arange(32)
    for h in range(2):
        Bm = B.bias_matrix(np.asarray(W["Tb"][h], np.float64))   # [query a][key b]
        for w in range(4):
            for J in range(4):
                for i in range(8):
                    tb[h, w, J, :, i] = Bm[16 * w + (lane & 15), 16 * J + 8 * (lane >> 4) + i]
    return b.tobytes(), tb.tobytes()


_TAB = {}


def tables(s2):
    key = float(s2)
    if key not in _TAB:
        _TAB[key] = _tables(s2)
    return _TAB[key]


def _tables(s2):
    m = V.load_manifest(VDIR)
    t = next(c for c in m["cases"] if c["kind"] == "tables")["arrays"]
    sig = V.load_array(VDIR, t["sig"])
    th = V.load_array(VDIR, t["tanh16"])
    # 55-ERRATUM-2 (clean/rr-f-flow): CANDW's correctly rounded binary16 2^t and 1/S tables follow (TAB_EXP2, TAB_RCP16)
    tab = np.concatenate([f16bits(sig), f16bits(th), fp.bits16(fp.exp2_16_table()),
                          fp.bits16(fp.rcp16_table())]).astype(np.uint16).tobytes()
    v = fp.all_f16().astype(f32)
    with np.errstate(all="ignore"):
        a = v * f32(s2)                                          # f32(y17 s2), one rounding (subnormals kept)
    rho = fp.tanh32(a).astype(f32)
    return tab, rho.tobytes()


class Arena:
    def __init__(self):
        self.size = 256
        self.init = []
        self.reads = []
        self.ptr = {}

    def alloc(self, name, nbytes, data=None, sentinel=False, read=False):
        off = self.size
        self.ptr[name] = off
        total = nbytes + (GUARD if sentinel else 0)
        if data is not None:
            assert len(data) == nbytes, (name, len(data), nbytes)
            self.init.append((off, data))
        elif sentinel:
            self.init.append((off, bytes([SENT]) * total))
        if read:
            self.reads.append((name, off, total))
        self.size = (off + total + 255) // 256 * 256
        return off

    def job(self, consts, groups, reps=0, warm_ms=0):
        w = [0x314a5252, self.size, len(consts), *consts, groups, reps, warm_ms, len(self.init)]
        out = bytearray(struct.pack(f"<{len(w)}I", *w))
        for off, data in self.init:
            out += struct.pack("<II", off, len(data)) + data + b"\0" * ((-len(data)) % 4)
        out += struct.pack("<I", len(self.reads))
        for _, off, n in self.reads:
            out += struct.pack("<II", off, n)
        return bytes(out)


def make_job(case, kind, variant, W=None, x=None, y2h=None, hold=None, hc=None, Zref=None, reps=0, warm_ms=0):
    """One job: arena image + constants. case: dict with Wt, Ht, Wr, Hr, shift, constants (f16 values), g, s1, s2."""
    wt, ht, wr, hr = case["Wt"], case["Ht"], case["Wr"], case["Hr"]
    sx, sy = case["shift"]
    A = Arena()
    if kind == "stage":
        A.alloc("x", wt * ht * 32, np.asarray(x, np.uint8).tobytes())
        A.alloc("wf", 56 * 256, weight_buffer(W))
        bb, tb = bias_buffers(W)
        A.alloc("bias", 768, bb)
        A.alloc("tb", 32768, tb)
    else:
        A.alloc("x", wt * ht * 64, f16bits(y2h).astype(np.uint16).tobytes())
    A.alloc("hold", wt * ht * 8, f16bits(hold).astype(np.uint16).tobytes())
    A.alloc("hc", (wt // 2) * (ht // 2) * 8, f16bits(hc).astype(np.uint16).tobytes())
    tab, rho = tables(case["s2"])
    A.alloc("tab", len(tab), tab)
    A.alloc("rho", len(rho), rho)
    rd = reps == 0
    A.alloc("cnet", wt * ht * 8, sentinel=rd, read=rd)
    A.alloc("hid", wr * hr * 4, sentinel=rd, read=rd)
    A.alloc("gates", wr * hr * 4, sentinel=rd, read=rd)
    A.alloc("prec", wr * hr * 8, sentinel=rd, read=rd)
    A.alloc("candw", wr * hr * 4, sentinel=rd, read=rd)
    nwx = -(-(wt + sx) // 8)
    nwy = -(-(ht + sy) // 8)
    dbg = variant in ("dbg", "dbgz", "head")
    if dbg:
        A.alloc("dy", wt * ht * 64, sentinel=True, read=rd)
        A.alloc("dimg", 3 * wr * hr * 8, sentinel=True, read=rd)
        A.alloc("dz", nwx * nwy * 2 * 64 * 2, sentinel=True, read=rd)
    A.alloc("flag", nwx * nwy * 16, sentinel=rd, read=rd)   # RR_STRICT_STEP 4 marks (per window and wave)
    A.alloc("wpl", 32 * 1536 + 32 * 16)                       # RR_STRICT_STEP 4 weight planes (rr_d0r_wplanes)
    if variant == "dbgz":
        A.alloc("zs", nwx * nwy * 2 * 64 * 2, f16bits(Zref).astype(np.uint16).tobytes())
    k = case["constants"]
    # bit 8 of sx: the 18-channel reading of 55-ERRATUM-2 (rr_d0r.hlsl RR_HEAD18), whenever the reference uses it
    h18 = 0x100 if len(Hd.head_channels()["candw"]) == 4 else 0
    vals = {"Wt": wt, "Ht": ht, "Wr": wr, "Hr": hr, "sx": sx | h18, "sy": sy, "nwx": nwx,
            "k_g": int(f16bits([case["g"]])[0]), "k_s1": int(np.array([case["s1"]], f32).view(np.uint32)[0])}
    for a, n in KNAMES.items():
        vals[a] = int(f16bits([k[n]])[0]) if n in k else 0
    consts = []
    for a in ARGS:
        if a.startswith("p_"):
            consts.append(A.ptr.get(a[2:], 0))
        else:
            consts.append(vals[a])
    return A, A.job(consts, nwx * nwy, reps, warm_ms)


def case_params(c):
    p = {"Wt": c["tensor_extent"]["Wt"], "Ht": c["tensor_extent"]["Ht"], "Wr": c["render_size"]["Wr"],
         "Hr": c["render_size"]["Hr"], "g": V.scalar_value(c["scalars"]["g"]),
         "s1": V.scalar_value(c["scalars"]["s1"]), "s2": V.scalar_value(c["scalars"]["s2"]),
         "shift": tuple(c["shift"]["recurrent_shift"]) if "shift" in c else (0, 0),
         "constants": {n: V.scalar_value(e) for n, e in c.get("constants", {}).items()}}
    return p


def prepare(work):
    m = V.load_manifest(VDIR)
    jobs = []
    for c in m["cases"]:
        if c["kind"] not in ("stage", "head"):
            continue
        p = case_params(c)
        ld = lambda e: V.load_array(VDIR, e)
        I = c["inputs"]
        if c["kind"] == "stage":
            W = {n: ld(I[n]) for n in ("Wq", "Wv", "Wo", "W1", "W2", "Tb", "bo", "b1", "b2")}
            Zref = ld(c["diagnostics"]["Z"])
            for variant in ("prod", "dbg", "dbgz"):
                A, job = make_job(p, "stage", variant, W=W, x=ld(I["x"]), hold=ld(I["hold"]), hc=ld(I["Hc"]),
                                  Zref=Zref)
                jobs.append(_write(work, c["name"], variant, A, job, p))
        else:
            for variant in ("head", "headprod"):
                A, job = make_job(p, "head", variant, y2h=ld(I["y2h"]), hold=ld(I["hold"]), hc=ld(I["Hc"]))
                jobs.append(_write(work, c["name"], variant, A, job, p))
    # timing at 1440p Quality: synthetic data in the distribution of stage_basic (random weights, x ~ q8(N(0, 1)))
    basic = case_params(next(c for c in m["cases"] if c["name"] == "stage_basic"))
    rng = np.random.default_rng(1440)
    wt, ht, wr, hr = TIME_SIZE
    p = dict(basic, Wt=wt, Ht=ht, Wr=wr, Hr=hr, shift=B.recurrent_shift((6, 6)))
    W = V.make_weights(rng)
    x = fp.q8(rng.normal(0, 1.0, (ht, wt, 32)))
    hold = fp.r16(rng.normal(0, 1.0, (ht, wt, 4)))
    hold[..., 3] = 0.0
    hc = fp.r16(rng.normal(0, 1.2, (ht // 2, wt // 2, 4)))
    A, job = make_job(p, "stage", "prod", W=W, x=x, hold=hold, hc=hc, reps=200, warm_ms=3000)
    jobs.append(_write(work, "time_1440pQ", "prod", A, job, p))
    y2h = fp.r16(rng.normal(0, 2.0, (ht, wt, 32)))
    ph = dict(p, shift=(0, 0))
    A, job = make_job(ph, "head", "headprod", y2h=y2h, hold=hold, hc=hc, reps=200, warm_ms=3000)
    jobs.append(_write(work, "time_1440pQ_headonly", "headprod", A, job, ph))
    with open(os.path.join(work, "jobs.json"), "w") as fh:
        json.dump(jobs, fh, indent=1)
    print(f"{len(jobs)} jobs in {work}")


def _write(work, name, variant, A, job, p):
    d = os.path.join(work, "jobs")
    os.makedirs(d, exist_ok=True)
    jp = os.path.join(d, f"{name}.{variant}.job")
    with open(jp, "wb") as fh:
        fh.write(job)
    return {"name": name, "variant": variant, "job": jp, "out": jp[:-4] + ".out", "log": jp[:-4] + ".log",
            "reads": [[n, off, sz] for n, off, sz in A.reads],
            "params": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in p.items()
                       if k != "constants"}}


# ------------------------------------------------------------------------------------------------ gpu
def gpu(work):
    with open(os.path.join(work, "jobs.json")) as fh:
        jobs = json.load(fh)
    exe = os.path.join(work, "rr_host.exe")
    for j in jobs:
        dx = os.path.join(work, f"rr_{j['variant']}.dxil")
        r = subprocess.run([exe, dx, j["job"], j["out"]], capture_output=True, text=True, timeout=600)
        with open(j["log"], "w") as fh:
            fh.write(r.stdout + r.stderr + f"\nexit {r.returncode}\n")
        print(j["name"], j["variant"], r.returncode, (r.stdout.strip().splitlines() or [""])[0], flush=True)


# ------------------------------------------------------------------------------------------------ compare
def read_out(j):
    with open(j["out"], "rb") as fh:
        raw = fh.read()
    out, at = {}, 0
    for n, _, sz in j["reads"]:
        out[n] = raw[at:at + sz]
        at += sz
    return out


def body(b, n):
    """(the region's first n bytes as uint8, guard bytes untouched?)"""
    a = np.frombuffer(b, np.uint8)
    return a[:n], bool(np.all(a[n:] == SENT))


def as_f16(a, shape):
    return fp.from_bits16(a.view(np.uint16)).reshape(shape)


def steps(got, ref):
    return fp.step_diff16(got, ref)


def summarize_steps(got, ref, tol):
    d = steps(got, ref)
    return {"values": int(d.size), "exact": int(np.sum(d == 0)), "one_step": int(np.sum(d == 1)),
            "max_steps": int(d.max()) if d.size else 0, "over_tol": int(np.sum(d > tol))}


def compare_case(c, js):
    p = case_params(c)
    wt, ht, wr, hr = p["Wt"], p["Ht"], p["Wr"], p["Hr"]
    ld = lambda e: V.load_array(VDIR, e)
    O = c["outputs"]
    ref = {n: ld(O[n]) for n in O}
    rep = {"case": c["name"], "variants": {}}
    if c["kind"] == "stage":
        y2h_ref = ld(c["diagnostics"]["y2h"])
        Zref = ld(c["diagnostics"]["Z"])
    else:
        y2h_ref = ld(c["inputs"]["y2h"])
    outs = {}
    for j in js:
        o = read_out(j)
        outs[j["variant"]] = o
        r = {}
        guards = {}
        cn, guards["cnet"] = body(o["cnet"], wt * ht * 8)
        cnet = as_f16(cn, (ht, wt, 4))
        r["Cnet"] = summarize_steps(cnet, ref["Cnet"], 1)
        r["Cnet"]["unwritten"] = int(np.sum(np.all(cn.reshape(-1, 8) == SENT, axis=1)))
        hb, guards["hid"] = body(o["hid"], wr * hr * 4)
        gb, guards["gates"] = body(o["gates"], wr * hr * 4)
        pb, guards["prec"] = body(o["prec"], wr * hr * 8)
        wb, guards["candw"] = body(o["candw"], wr * hr * 4)
        hid_code = hb.view(np.int8).reshape(hr, wr, 4)
        gates_code = gb.reshape(hr, wr, 4)
        candw_code = wb.reshape(hr, wr, 4)
        prec = as_f16(pb, (hr, wr, 4))
        r["PREC"] = summarize_steps(prec, ref["PREC"], 1)
        r["codes_vs_reference"] = {"HID": int(np.sum(hid_code != ref["HID_code"])),
                                   "GATES": int(np.sum(gates_code != ref["GATES_code"])),
                                   "CANDW": int(np.sum(candw_code != ref["CANDW_code"]))}
        r["unwritten_texels"] = {n: int(np.sum(np.all(a.reshape(-1, 4) == SENT, axis=1)))
                                 for n, a in (("HID", hb), ("GATES", gb), ("CANDW", wb))}
        r["unwritten_texels"]["PREC"] = int(np.sum(np.all(pb.reshape(-1, 8) == SENT, axis=1)))
        if "dimg" in o:
            di, guards["dimg"] = body(o["dimg"], 3 * wr * hr * 8)
            img = as_f16(di, (3, hr, wr, 4))
            hidv, gatv, canv = img[0], img[1], img[2]
            r["HID"] = summarize_steps(hidv, ref["HID"], 1)
            r["GATES"] = summarize_steps(gatv, ref["GATES"], 1)
            r["CANDW"] = summarize_steps(canv, ref["CANDW"], 3)
            # codes = the store rule applied to the kernel's own f16 values
            r["codes_vs_own_values"] = {"HID": int(np.sum(fp.snorm8_store(hidv) != hid_code)),
                                        "GATES": int(np.sum(fp.unorm8_store(gatv) != gates_code)),
                                        "CANDW": int(np.sum(fp.unorm8_store(canv) != candw_code))}
            # CANDW exact requirements
            cb = f16bits(canv).reshape(hr, wr, 4)
            if len(Hd.head_channels()["candw"]) == 4:                # 55-ERRATUM-2: four logits y10 .. y13
                z = np.stack([y2h_ref[:hr, :wr, n] for n in Hd.head_channels()["candw"]], -1)
                eq = np.isfinite(z).all(-1) & (z == z[..., :1]).all(-1)
                nanq = np.isnan(z).any(-1) | (z == np.inf).any(-1) | (z == -np.inf).all(-1)
                r["CANDW_exact"] = {"equal_logits": int(eq.sum()),
                                    "equal_logits_bad": int((eq & ~np.all(canv == 0.25, axis=-1)).sum()),
                                    "nan_rows": int(nanq.sum()),
                                    "nan_rows_bad": int((nanq & ~np.all(np.isnan(canv), axis=-1)).sum())}
            else:
                ca, cb_ = Hd.head_channels()["candw"]                 # page 55 erratum: y10, y12
                y20 = y2h_ref[:hr, :wr, ca]
                y24 = y2h_ref[:hr, :wr, cb_]
                pair_bad = (cb[..., 0] != cb[..., 1]) | (cb[..., 2] != cb[..., 3])
                eq = np.isfinite(y20) & (y20 == y24)
                eq_bad = eq & ~np.all(canv == 0.25, axis=-1)
                with np.errstate(all="ignore"):
                    d = (y20.astype(np.float64) - y24).astype(f32)
                inf_pos = np.isposinf(d)
                inf_neg = np.isneginf(d)
                nanc = np.isnan(d)
                inf_bad = (inf_pos & ~((canv[..., 0] == 0.5) & (canv[..., 2] == 0))) | \
                          (inf_neg & ~((canv[..., 0] == 0) & (canv[..., 2] == 0.5))) | \
                          (nanc & ~np.all(np.isnan(canv), axis=-1))
                r["CANDW_exact"] = {"pair_not_identical": int(pair_bad.sum()), "equal_logits": int(eq.sum()),
                                    "equal_logits_bad": int(eq_bad.sum()),
                                    "infinite_or_nan_d": int((inf_pos | inf_neg | nanc).sum()), "infinite_or_nan_bad": int(inf_bad.sum())}
            dy, guards["dy"] = body(o["dy"], wt * ht * 64)
            y2h = as_f16(dy, (ht, wt, 32))
            r["y2h"] = summarize_steps(y2h, y2h_ref, 0)
            if c["kind"] == "stage":
                dz, guards["dz"] = body(o["dz"], Zref.size * 2)
                Z = as_f16(dz, Zref.shape)
                r["Z"] = summarize_steps(Z, Zref, 0)
                r["Z"]["rows_inf_kernel"] = int(np.isinf(Z).sum())
        r["guards_untouched"] = guards
        rep["variants"][j["variant"]] = r
    # the production build writes the same bytes as the debug build (prod vs dbg; headprod vs head)
    for a, b in (("prod", "dbg"), ("headprod", "head")):
        if a in outs and b in outs:
            rep[f"{a}_equals_{b}"] = all(outs[a][n] == outs[b][n] for n in ("cnet", "hid", "gates", "prec", "candw"))
    return rep


def verdict(rep):
    """exact / tolerance / failing for one case (the variant that the spec's test runs: own Z for stages)."""
    v = rep["variants"].get("dbg") or rep["variants"].get("head")
    keys = ["Cnet", "HID", "GATES", "PREC", "CANDW"]
    over = sum(v[k]["over_tol"] for k in keys)
    inexact = sum(v[k]["values"] - v[k]["exact"] for k in keys)
    hard = (sum(v["codes_vs_own_values"].values()) + v["CANDW_exact"].get("pair_not_identical", 0) +
            v["CANDW_exact"]["equal_logits_bad"] + v["CANDW_exact"].get("infinite_or_nan_bad", 0) +
            v["CANDW_exact"].get("nan_rows_bad", 0) +
            sum(1 for g in v["guards_untouched"].values() if not g) + v["Cnet"]["unwritten"] +
            sum(v["unwritten_texels"].values()))
    for a in ("prod_equals_dbg", "headprod_equals_head"):
        if a in rep and not rep[a]:
            hard += 1
    if "dbgz" in rep["variants"] and rep["variants"]["dbgz"]["y2h"]["values"] != rep["variants"]["dbgz"]["y2h"]["exact"]:
        hard += 1
    return "failing" if over or hard else ("tolerance" if inexact else "exact")


def compare(work):
    with open(os.path.join(work, "jobs.json")) as fh:
        jobs = json.load(fh)
    m = V.load_manifest(VDIR)
    report = {"cases": [], "timing": {}}
    for c in m["cases"]:
        if c["kind"] not in ("stage", "head"):
            continue
        js = [j for j in jobs if j["name"] == c["name"]]
        rep = compare_case(c, js)
        rep["verdict"] = verdict(rep)
        report["cases"].append(rep)
    for j in jobs:
        if j["name"].startswith("time_"):
            with open(j["log"]) as fh:
                line = next((l for l in fh if l.startswith("TIME")), "")
            report["timing"][j["name"]] = line.strip()
    with open(os.path.join(work, "build.json")) as fh:
        report["build"] = json.load(fh)
    with open(os.path.join(work, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    for rep in report["cases"]:
        v = rep["variants"].get("dbg") or rep["variants"].get("head")
        line = " ".join(f"{k}:{v[k]['values'] - v[k]['exact']}/{v[k]['over_tol']}" for k in ("Cnet", "HID", "GATES", "PREC", "CANDW"))
        extra = ""
        if "dbgz" in rep["variants"]:
            z = rep["variants"]["dbgz"]
            extra = (f" | y2h(own Z) diff {v['y2h']['values'] - v['y2h']['exact']} Z diff {v['Z']['values'] - v['Z']['exact']}"
                     f" | y2h(ref Z) diff {z['y2h']['values'] - z['y2h']['exact']}")
        print(f"{rep['case']:13s} {rep['verdict']:9s} inexact/over-tol {line}{extra}")
    print(json.dumps(report["timing"], indent=1))
    return 0 if all(r["verdict"] != "failing" for r in report["cases"]) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "prepare", "gpu", "compare"])
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    w = os.path.abspath(a.work)
    return {"build": build, "prepare": prepare, "gpu": gpu, "compare": compare}[a.step](w) or 0


if __name__ == "__main__":
    sys.exit(main())
