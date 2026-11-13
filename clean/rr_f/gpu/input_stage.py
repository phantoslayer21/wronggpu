# SPDX-License-Identifier: MIT
"""Spec 52 input stage on D3D12 (kernels/rr_input.hlsl, rr_input_mlp.hlsl): host packing of the per-frame constant
block, uploads, dispatches, readback and comparison with the NumPy reference (reference/input_stage.py).

    python -m clean.rr_f.gpu.input_stage --json <report> [--step X|E|S] [--probe] [--cases all|n,n]

Comparison rules: feature codes, x0 codes, hold and the motion surfaces bit for bit (e4m3 zero codes are compared
exactly here: the input codes are not in-network values); the f32 features of the debug buffer bit for bit.
Run under WG_RR_BUILD (private build directory); every GPU call goes through the GPU lock (gpu/build.py)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import build, layout
from ..reference import f32 as F
from ..reference import input_stage as IS
from ..reference.fmt import from_f16_bits, to_f16_bits

# constant block layout (kernels/rr_input.hlsl K_*)
NWORDS = 256
K = dict(WR=0, HR=1, WT=2, HT=3, WH=4, HH=5, SX=6, SY=7, FC=8, FLAGS=9, E=10, X3=12, JX=13, JY=14, PJX=15,
         PJY=16, MOX=17, MOY=18, MSHX=19, MSHY=20, RWHWR=21, RHHHR=22, RWRWH=23, RHRHH=24, INVWR=25, INVHR=26,
         INVWH=27, INVHH=28, HCSX=29, HCSY=30, HOFFX=31, HOFFY=32, ZFAR=33, V22=34, V32=35, V23=36, RCPEP=37,
         MPOS=40, MREP=56, SYM=72, TEX=100, PH1=208, PH2=211, HID=214, PMR=215, PMW=216, PE=217, PN=218, D=219,
         UN10=220, SN8=221, CODES=222, HOLD=223, DEBUG=224, DIAG=225, EXPO=226, LOGT=227, EXPT=228, MARK=229)
SYMS = ("K4", "K5", "K7", "K7n", "K8", "K10", "K12", "C_HDEC", "C_FCLIP", "C_AMAX", "C_GCL", "C_GCLn", "C_LWrb",
        "C_LWg", "C_MDL", "C_MDLn", "Gm", "Gd", "Gh", "Gs", "C_SV1", "C_SV2", "C_SV3", "C_SV4", "C_SV4n")
FLAG = dict(reset=1, rev=2, hwd=4, nrmdec=8, rghpk=16, b2=128, dis=256, B=512, G1=1024, G2=2048, v33lt1=4096)
DEBUG_WORDS = 96
REPLAY_GROUPS = 512          # the replay's fixed grid (its lanes stride over the list of marked tokens)
FEAT_BLOCK = 128             # lanes per workgroup of the features kernels (one token per lane)


def f32w(v):
    return int(np.asarray(v, np.float32).reshape(()).view(np.uint32))


def tables():
    un10 = IS.decode_unorm(np.arange(1024), 10).astype(np.float32)
    sn8 = IS.decode_snorm(np.arange(256).astype(np.uint8).view(np.int8).astype(np.int64), 8).astype(np.float32)
    return un10, sn8


def ff_tables():
    """The float-float log2 / exp2 tables of the kernel's fast path (gpu/ffmath.py): [256][4] and [65][2] f32."""
    from . import ffmath
    logt = np.zeros((256, 4), np.float32)
    logt[:, :3] = ffmath.LOGT
    return logt, np.ascontiguousarray(ffmath.EXPT, np.float32)


def ff_offsets(ar):
    if not hasattr(ar, "ff_tables"):
        logt, expt = ff_tables()
        ar.ff_tables = (ar.add(logt), ar.add(expt))
    return ar.ff_tables


def m2_mode(frame, k):
    return m2_mode_of({r for r in IS.ROLES if frame.attached(r)}, k)


def m2_mode_of(attached, k):
    """52.7.1 / 52.8: 0 m2 = mO, 1 from SV, 2 the reconstruction."""
    if "SV" in attached:
        return 1
    if "Hd" not in attached or k.specular_from_v or not k.camera_usable:
        return 0
    return 2


def scalar_words(k, S, attached):
    """Words 0 .. K_TEX-1 of the constant block: sizes, shifts, flags, per-frame f32 values, camera maps, symbols.
    The host-side C++ packing (clean/rr_f/host/rr_input_host.h) must give the same words."""
    words = np.zeros(K["TEX"], np.uint32)
    flags = 0
    flags |= FLAG["reset"] if k.reset else 0
    flags |= FLAG["rev"] if k.rev else 0
    flags |= FLAG["hwd"] if k.hw_depth else 0
    flags |= FLAG["nrmdec"] if k.nrm_dec else 0
    flags |= FLAG["rghpk"] if k.rgh_packed else 0
    flags |= m2_mode_of(attached, k) << 5
    flags |= FLAG["b2"] if S.b2 else 0
    for role, name in (("Q", "dis"), ("B", "B"), ("G1", "G1"), ("G2", "G2")):
        flags |= FLAG[name] if role in attached else 0
    if k.V2C is not None and k.V2C[3][3] < 1:
        flags |= FLAG["v33lt1"]
    for n, v in (("WR", k.Wr), ("HR", k.Hr), ("WT", k.Wt), ("HT", k.Ht), ("WH", k.Wh), ("HH", k.Hh),
                 ("SX", k.sx), ("SY", k.sy), ("FC", k.fc), ("FLAGS", flags)):
        words[K[n]] = v
    fl = dict(E=k.E, X3=k.X3, JX=k.jx, JY=k.jy, PJX=k.pjx, PJY=k.pjy, MOX=k.mo[0], MOY=k.mo[1],
              MSHX=k.msh[0], MSHY=k.msh[1], RWHWR=k.rWhWr, RHHHR=k.rHhHr, RWRWH=k.rWrWh, RHRHH=k.rHrHh,
              INVWR=k.invWr, INVHR=k.invHr, INVWH=k.invWh, INVHH=k.invHh, HCSX=k.hc_scale[0],
              HCSY=k.hc_scale[1], HOFFX=k.hoff[0], HOFFY=k.hoff[1], ZFAR=k.zfar, RCPEP=F.rcp(k.Ep))
    if k.V2C is not None:
        fl.update(V22=k.V2C[2][2], V32=k.V2C[3][2], V23=k.V2C[2][3])
    for n, v in fl.items():
        words[K[n]] = f32w(v)
    if k.camera_usable:
        words[K["MPOS"]:K["MPOS"] + 16] = np.asarray(k.Mpos, np.float32).reshape(-1).view(np.uint32)
        words[K["MREP"]:K["MREP"] + 16] = np.asarray(k.Mrep, np.float32).reshape(-1).view(np.uint32)
    for i, n in enumerate(SYMS):
        if n in S.values:
            words[K["SYM"] + i] = f32w(S.values[n])
    return words


class InputCall:
    """Arena state of one input-stage call (features kernel + MLP kernel)."""

    def __init__(self, arena, frame, params, S, IN0, wc, debug=True, diag=True, k=None, exposure=0, links=None, fold=0):
        """exposure: arena offset of the exposure pass's result (E, Ep, invE, X3), or 0 for the values of params.
        links: optional {"HID", "PMR", "PMW", "PE", "PN", "PH1", "PH2": arena offset} of temporal images produced
        earlier in the same arena (the previous frame's head HID / input-stage PM / output-stage PH1 / pooled PH2; the
        surfaces to write); those are not uploaded (the frame's arrays give shapes only)."""
        links = links or {}
        self.fold = fold          # front-end MLP folded into IN0 (gpu/stage.py infold): no MLP dispatch, no x0 tensor
        self.k = k = k or IS.frame_constants(params)
        self.frame, self.S = frame, S
        ar = arena
        words = np.zeros(NWORDS, np.uint32)
        attached = {r for r in IS.ROLES if frame.attached(r)}
        words[:K["TEX"]] = scalar_words(k, S, attached)
        self.mode = m2_mode_of(attached, k)
        # textures (reader origins of 52.1.2)
        self.tex = {}
        for r, role in enumerate(IS.ROLES):
            if not frame.attached(role):
                continue
            t = frame.textures[role]
            data = np.ascontiguousarray(t.data, np.float32)
            off = ar.add(data)
            ox, oy = IS.origin_of(frame, role)
            b = K["TEX"] + 8 * r
            words[b:b + 7] = [off, data.shape[1], data.shape[0], data.shape[2], ox, oy, 1]
            self.tex[role] = off
        Wr, Hr = k.Wr, k.Hr
        ph1 = np.ascontiguousarray(frame.PH1, np.uint16)
        ph2 = np.ascontiguousarray(frame.PH2, np.uint16)
        # temporal images as stored (codes) and their decoded f32 copies, made by rr_input_decode in this call
        self.decode = []
        for name, img, kind, src in (("PH1", ph1, 0, links.get("PH1") or ar.add(ph1)),
                                     ("PH2", ph2, 0, links.get("PH2") or ar.add(ph2)),
                                     ("HID", frame.HID, 1, links.get("HID") or ar.add(np.ascontiguousarray(frame.HID,
                                                                                                         np.int8)))):
            n = img.shape[0] * img.shape[1]
            dst = ar.reserve(n * 16, fill=0xEE)
            self.decode.append(ar.add(np.array([kind, n, src, dst], np.uint32)))
            self.decode_n = getattr(self, "decode_n", []) + [n]
            if name == "HID":
                words[K["HID"]] = dst
            else:
                words[K[name]:K[name] + 3] = [dst, img.shape[1], img.shape[0]]
        words[K["PMR"]] = links.get("PMR") or ar.add(np.ascontiguousarray(frame.PM_read, np.float16))
        self.out = {}
        for n, prior in (("PMW", frame.PM_write), ("PE", frame.Pe), ("PN", frame.Pn)):
            words[K[n]] = links.get(n) or ar.add(np.ascontiguousarray(prior, np.float16))
            self.out[n] = (int(words[K[n]]), prior.shape, np.float16)
        words[K["D"]] = ar.add(np.ascontiguousarray(IN0["D"], np.uint16))
        un10, sn8 = tables()
        words[K["UN10"]] = ar.add(un10)
        words[K["SN8"]] = ar.add(sn8)
        words[K["LOGT"]], words[K["EXPT"]] = ff_offsets(ar)
        nt = k.Wt * k.Ht
        words[K["CODES"]] = ar.reserve(nt * 64, fill=0xEE)
        words[K["HOLD"]] = ar.reserve(nt * 8, fill=0xEE)
        self.out["codes"] = (int(words[K["CODES"]]), (k.Ht, k.Wt, 64), np.uint8)
        self.out["hold"] = (int(words[K["HOLD"]]), (k.Ht, k.Wt, 4), np.float16)
        if debug:
            words[K["DEBUG"]] = ar.reserve(nt * DEBUG_WORDS * 4, fill=0xEE)
            self.out["debug"] = (int(words[K["DEBUG"]]), (k.Ht, k.Wt, DEBUG_WORDS), np.float32)
        words[K["MARK"]] = ar.reserve((nt + 1) * 4)               # replay list: count + token indices
        if diag:
            words[K["DIAG"]] = ar.reserve(16)
            self.out["diag"] = (int(words[K["DIAG"]]), (4,), np.uint32)
        words[K["EXPO"]] = exposure
        self.pK = ar.add(words)
        self.words = words
        # front-end MLP
        FE = IS.fe_tensors(IN0) if "FE.W1_0" in IN0 else IN0["FE"]
        w1 = np.stack([layout.frags(np.asarray(FE[f"W1_{br}"], np.uint8).T) for br in (0, 1)])   # [2][2][4]
        w2 = np.stack([layout.frags(np.asarray(FE[f"W2_{br}"], np.uint8).T) for br in (0, 1)])   # [2][2][2]
        self.pW1, self.pW2 = ar.add(w1), ar.add(w2)
        self.pB1 = ar.add(np.concatenate([layout.f32(FE["b1_0"]), layout.f32(FE["b1_1"])]))
        self.pB2 = ar.add(layout.f32(FE["b2"]))
        self.pX0 = 0 if fold else ar.reserve(nt * 32, fill=0xEE)
        if not fold:
            self.out["x0"] = (self.pX0, (k.Ht, k.Wt, 32), np.uint8)
        from .stage import wc_dwords
        self.pMlpDiag = ar.reserve(16) if diag and not fold else 0
        if self.pMlpDiag:
            self.out["mlp_fallbacks"] = (self.pMlpDiag, (1,), np.uint32)
        self.mlp_args = [int(words[K["CODES"]]), self.pW1, self.pW2, self.pB1, self.pB2, self.pX0, nt] +             wc_dwords(wc) + [self.pMlpDiag]
        self.feat_args = [self.pK]

    def dispatches(self, kf="input-features", km="input-mlp", kr="input-replay", kd="input-decode",
                   kc="input-clear"):
        nt = self.k.Wt * self.k.Ht
        dec = [(kd, (n + 255) // 256, [h]) for h, n in zip(self.decode, self.decode_n)]
        fb = FEAT_BLOCK
        return dec + [(kc, 1, self.feat_args), (kf, (nt + fb - 1) // fb, self.feat_args),
                      (kr, min((nt + fb - 1) // fb, REPLAY_GROUPS), self.feat_args)] + \
            ([] if self.fold else [(km, nt // 64, self.mlp_args)])

    def in0_links(self):
        """IN0's argument bindings ({arg index: arena offset}): x0 and hold, or with the folded MLP the codes, hold and
        the MLP's weights in IN0's argument slots unused by the in0 kind (rr_block_c32.hlsl RR_INFOLD)."""
        if not self.fold:
            return {0: self.pX0, 2: self.out["hold"][0]}
        return {0: self.out["codes"][0], 1: self.pW1, 2: self.out["hold"][0], 3: self.pB2, 15: self.pW2, 16: self.pB1}

    def outputs(self, img):
        res = {}
        for n, (off, shape, dt) in self.out.items():
            nb = int(np.prod(shape)) * np.dtype(dt).itemsize
            res[n] = img[off:off + nb].copy().view(dt).reshape(shape)
        return res


# Opt-in near-exact features kernels (tier B; never the default): name -> defines over the fast pass's
LITE_VARIANTS = {
    # plain hardware f32 math: hardware log2 / exp2 / sqrt, no fences, no zero-result tests, no certification, no
    # replay marks (the replay finds an empty list)
    "lite": {"RR_IN_FAST": 29, "RR_IN_HCERT": 0, "RR_IN_L2CERT": 0, "RR_IN_LITE": 1},
}


def features_kernel(variant="exact", srv=0):
    """The features fast pass: "exact" (default: certified, marks the replay's tokens) or an opt-in LITE_VARIANTS name."""
    d = {"WG_BLOCK": FEAT_BLOCK, "RR_IN_EXACT": 0, "RR_IN_LDST": 1}
    if not srv:
        d["RR_IN_DBG"] = 1                       # harness: debug words + K_DIAG counters (a14 C31; the runtime builds 0)
    tag = "rr_input_features_fast"
    if variant != "exact":                       # "name" or "name+K=V+..." (extra defines: timing experiments)
        name, *extra = variant.split("+")
        d.update(LITE_VARIANTS[name] if name != "exact" else {})
        d.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in extra})
        tag = "rr_input_features_" + variant.replace("+", "_").replace("=", "")
    if srv:
        d["RR_IN_SRV"] = 1
        tag += "_srv"
    return build.compile_kernel("rr_input.hlsl", "rr_input_features", d, tag=tag)


def kernels(step="X", mark_all=False, mutation=0):
    """mark_all: test variant whose fast pass marks every token, so that the replay computes all of them.
    step X: variant X with the fast path for exact sums (RR_STEP 3, the same results); X1: plain variant X."""
    sd = {"E": 0, "X": 3, "X1": 1, "S": 2}[step]
    return {"input-features": (build.compile_kernel("rr_input.hlsl", "rr_input_features",
                                                    {"WG_BLOCK": FEAT_BLOCK, "RR_IN_EXACT": 0, "RR_IN_MARK_ALL": int(mark_all),
                                                     "RR_IN_LDST": 1, "RR_IN_DBG": 1, **({"RR_IN_HCERT_MUT": 1} if mutation == "hcert" else {})},
                                                    tag="rr_input_features_fast" + ("_all" if mark_all else "") +
                                                    ("_mut" if mutation == "hcert" else "")), 1),
            "input-decode": (build.compile_kernel("rr_input.hlsl", "rr_input_decode", {"WG_BLOCK": 256}), 1),
            "input-clear": (build.compile_kernel("rr_input.hlsl", "rr_input_mark_clear", {"WG_BLOCK": 64}), 1),
            "input-replay": (build.compile_kernel("rr_input.hlsl", "rr_input_features",
                                                  {"WG_BLOCK": FEAT_BLOCK, "RR_IN_EXACT": 1, "RR_IN_REPLAY": 1, "RR_IN_LDST": 1, "RR_IN_DBG": 1},
                                                  tag="rr_input_features_replay"), 1),
            "input-mlp": (build.compile_kernel("rr_input_mlp.hlsl", "rr_input_mlp",
                                               {"WG_BLOCK": 128, "RR_STEP": sd, "RR_STEP3_MUT": int(mutation == "mlp")},
                                               tag=f"rr_input_mlp_{step}" + ("_mut" if mutation == "mlp" else "")), 13)}


def bits16(a):
    return np.asarray(a, np.float16).view(np.uint16)


def compare(got, ref):
    """Exact comparison of one input-stage call. ref: reference.input_stage result dict."""
    r = {}
    codes = got["codes"]
    d = codes != ref["codes"]
    r["codes"] = dict(n=int(codes.size), diff=int(d.sum()),
                      by_feature={int(i): int(c) for i, c in enumerate(d.reshape(-1, 64).sum(0)) if c})
    hb = bits16(got["hold"])
    he = to_f16_bits(ref["hold"])
    r["hold"] = dict(n=int(hb.size), diff=int((hb != he).sum()))
    if "x0" in got:           # not with the MLP folded into IN0 (IN0's outputs check it)
        r["x0"] = dict(n=int(got["x0"].size), diff=int((got["x0"] != ref["x0"]).sum()))
    for n, key in (("PMW", "PM"), ("PE", "Pe"), ("PN", "Pn")):
        g, e = bits16(got[n]), bits16(ref["surfaces"][key])
        r[key] = dict(n=int(g.size), diff=int((g != e).sum()))
    if "debug" in got:
        feat = got["debug"][..., :64]
        ok = (feat.view(np.uint32) == ref["feat"].view(np.uint32)) | (np.isnan(feat) & np.isnan(ref["feat"]))
        # features whose code the fast pass certified from an approximation (kernel RR_IN_HCERT; debug words 72 / 73
        # list them): their debug f32 value is the approximation; their codes are compared exactly above
        lo, hi = got["debug"][..., 72].view(np.uint32), got["debug"][..., 73].view(np.uint32)
        approx = np.zeros(feat.shape, bool)
        for k in range(64):
            approx[..., k] = (((lo if k < 32 else hi) >> (k % 32)) & 1).astype(bool)
        r["feat_f32"] = dict(n=int(feat.size), diff=int((~ok & ~approx).sum()), approximate=int(approx.sum()),
                             approximate_not_identical=int((~ok & approx).sum()),
                             by_feature={int(i): int(c) for i, c in enumerate((~ok & ~approx).reshape(-1, 64).sum(0))
                                         if c})
        mv = got["debug"][..., 64:72]
        exp = np.stack([ref["mO"][0], ref["mO"][1], ref["mN"][0], ref["mN"][1], ref["mF"][0], ref["mF"][1],
                        ref["m2"][0], ref["m2"][1]], -1).astype(np.float32)
        okm = (mv.view(np.uint32) == exp.view(np.uint32)) | (np.isnan(mv) & np.isnan(exp))
        r["motion_f32"] = dict(n=int(mv.size), diff=int((~okm).sum()))
    if "mlp_fallbacks" in got:
        r["mlp_exact5_wave_steps"] = int(got["mlp_fallbacks"][0])
    if "diag" in got:
        r["ziv_unproven"] = int(got["diag"][0])
        r["replay_fallbacks"] = int(got["diag"][1])
        r["replayed_tokens"] = int(got["diag"][2])
    r["exact"] = all(v["diff"] == 0 for v in r.values() if isinstance(v, dict))
    return r


# ---------------------------------------------------------------------------------------------------------------
# cases

def case_list():
    from ..tests.test_input_stage import FEATURE_CASES
    extra = [dict(Wr=96, Hr=70, smv="Hd", b2=False), dict(Wr=129, Hr=64, smv="SV", edge=0.02, mv_jittered=True),
             dict(Wr=200, Hr=113, reset=True, smv="Hd", b2=True), dict(Wr=64, Hr=64, origins=False),
             dict(Wr=77, Hr=41, rgh_packed=True, nrm_dec=True, guides=["B", "G1", "G2", "Q"], smv="Hd",
                  hw_depth=False, edge=0.03)]
    return list(FEATURE_CASES) + extra


def make(i, kw):
    from ..tests import synth_input as SI
    rng = np.random.default_rng(5300 + i)
    kw = dict(kw)
    kw.setdefault("Wr", int(rng.integers(9, 31)))
    kw.setdefault("Hr", int(rng.integers(7, 30)))
    return SI.make_case(rng, **kw)


# ---------------------------------------------------------------------------------------------------------------
# probe of the scalar helpers

PROBE_OPS = dict(quant=0, log2=1, exp2=2, sqrt=3, rcp=4, mul=5, fma=6, add=7, div=8, f16=9, un10=10, sn8=11)


def probe_inputs(rng):
    from ..tests.test_input_stage import special_f32
    sets = {}
    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    f16v = F.f32(from_f16_bits(bits))
    ds = np.arange(128, dtype=np.uint32)
    a = np.tile(f16v, 128)
    d = np.repeat(ds, 65536)
    sets["quant"] = (a, d.view(np.float32), np.zeros_like(a))
    r = np.concatenate([special_f32(rng, 200000), (rng.normal(0, 1, 200000) * np.exp2(rng.integers(-26, 17, 200000))),
                        np.float32([65519.99, 65520.0, -65520.0, 480.0, 448.0])]).astype(np.float32)
    sets["quant_f32"] = (r, rng.integers(0, 128, r.size).astype(np.uint32).view(np.float32), np.zeros_like(r))
    # log2: every f32 in [1, 2) and [0.5, 1) with stride 4, plus random magnitudes and specials
    m = (np.arange(0, 1 << 23, 4, dtype=np.uint32) | 0x3f800000).view(np.float32)
    m2 = (np.arange(1, 1 << 23, 4, dtype=np.uint32) | 0x3f000000).view(np.float32)
    lr = np.abs(special_f32(rng, 400000))
    sets["log2"] = (np.concatenate([m, m2, lr, np.float32([0, -0.0, -1, np.inf, np.nan, 1, 2, 0.5])]),)
    t = np.concatenate([rng.uniform(-130, 130, 600000), rng.uniform(-1, 1, 600000),
                        np.float32([0, -0.0, 128, -126, -126.00001, 127.99999, np.nan, np.inf, -np.inf])])
    sets["exp2"] = (t.astype(np.float32),)
    sets["sqrt"] = (np.abs(special_f32(rng, 500000)),)
    sets["rcp"] = (special_f32(rng, 500000),)
    a, b, c = special_f32(rng, 500000), special_f32(rng, 500000), special_f32(rng, 500000)
    for op in ("mul", "add", "div"):
        sets[op] = (a, b)
    sets["fma"] = (a, b, c)
    near = (np.float32(2.0 ** -63) * rng.uniform(0.999, 1.001, 200000)).astype(np.float32)
    sets["mul_boundary"] = (near, near[::-1].copy())
    sets["f16"] = (special_f32(rng, 300000),)
    sets["un10"] = (np.arange(1024, dtype=np.uint32).view(np.float32),)
    sets["sn8"] = (np.arange(256, dtype=np.uint32).view(np.float32),)
    sets["rcp_binade"] = ((np.arange(1 << 23, dtype=np.uint32) | 0x3f800000).view(np.float32),)
    return sets


def probe_expected(name, args):
    if name in ("quant", "quant_f32"):
        return IS.quantise(args[0], args[1].view(np.uint32).astype(np.int64)).astype(np.uint32)
    op = name.split("_")[0]
    if op == "un10":
        return IS.decode_unorm(args[0].view(np.uint32).astype(np.int64), 10).view(np.uint32)
    if op == "sn8":
        return IS.decode_snorm(args[0].view(np.uint32).astype(np.uint8).view(np.int8).astype(np.int64), 8).view(np.uint32)
    f = dict(log2=F.log2, exp2=F.exp2, sqrt=F.sqrt, rcp=F.rcp, mul=F.mul, add=F.add, div=F.div, fma=F.fma)
    if op == "f16":
        return to_f16_bits(F.f16rt(args[0]).astype(np.float64)).astype(np.uint32)
    return np.asarray(f[op](*args), np.float32).view(np.uint32)


def run_probe(json_rows):
    rng = np.random.default_rng(5299)
    sets = probe_inputs(rng)
    ar = build.Arena()
    kern = {"probe": (build.compile_kernel("rr_input.hlsl", "rr_input_probe", {"WG_BLOCK": 256}), 1)}
    disp, recs = [], []
    for name, args in sets.items():
        n = args[0].size
        arrs = [np.ascontiguousarray(a, np.float32) for a in args] + [np.zeros(n, np.float32)] * (3 - len(args))
        offs = [ar.add(a) for a in arrs]
        out = ar.reserve(n * 4 + 16)
        op = PROBE_OPS[name.split("_")[0]]
        hdr = ar.add(np.array([op, n, offs[0], offs[1], offs[2], out, *ff_offsets(ar)], np.uint32))
        disp.append(("probe", (n + 255) // 256, [hdr]))
        recs.append((name, args, out, n))
    print(f"probe: {len(disp)} sets, {ar.size / 1e6:.0f} MB", flush=True)
    img, log = build.run(ar, kern, disp, require_double=True, workdir=str(Path(build.BUILD) / "input-probe"))
    ok = True
    for name, args, out, n in recs:
        got = img[out:out + n * 4].copy().view(np.uint32)
        ziv = int(img[out + n * 4:out + n * 4 + 4].copy().view(np.uint32)[0])
        slow = int(img[out + n * 4 + 4:out + n * 4 + 8].copy().view(np.uint32)[0])
        exp = probe_expected(name, args)
        if name.startswith("quant"):
            bad = got != exp
        elif name == "f16":
            nan16 = lambda v: ((v & 0x7c00) == 0x7c00) & ((v & 0x3ff) != 0)
            bad = (got != exp) & ~(nan16(got) & nan16(exp))
        else:
            gf, ef = got.view(np.float32), exp.view(np.float32)
            bad = (got != exp) & ~(np.isnan(gf) & np.isnan(ef))
        row = dict(probe=name, n=int(n), diff=int(bad.sum()), ziv_unproven=ziv, fast_path_fallbacks=slow)
        if bad.any():
            i = np.flatnonzero(bad)[:5]
            row["examples"] = [[float(args[0][j]), float(args[1][j]) if len(args) > 1 else None,
                                hex(int(got[j])), hex(int(exp[j]))] for j in i]
            ok = False
        json_rows.append(row)
        print(row, flush=True)
    return ok


XCHECK = {  # op: list of (first pattern, end pattern)
    "log2": (0, [(0x00800000, 0x7F800000)]),
    "exp2": (1, [(0x00000001, 0x43000000), (0x80000001, 0xC2FC0001)]),
    "rcp": (2, [(0x00800000, 0x7F800000)]),
    "sqrt": (3, [(0x00800000, 0x7F800000)]),
}


def run_xcheck(json_rows, ops=("log2", "exp2", "rcp", "sqrt"), chunk=1 << 24):
    """Every f32 input of the listed ranges: the float-float fast paths against the binary64 paths, the fma-corrected
    reciprocal against the binary64 quotient, all on the GPU (results stay on the GPU; counts are read back)."""
    ar = build.Arena()
    kern = {"xcheck": (build.compile_kernel("rr_input.hlsl", "rr_input_xcheck", {"WG_BLOCK": 256}), 1)}
    plog, pexp = ff_offsets(ar)
    disp, recs = [], []
    for name in ops:
        op, ranges = XCHECK[name]
        out = ar.reserve(32)
        for lo, hi in ranges:
            for b in range(lo, hi, chunk):
                n = min(chunk, hi - b)
                hdr = ar.add(np.array([op, b, n, out, plog, pexp], np.uint32))
                disp.append(("xcheck", (n + 255) // 256, [hdr]))
        recs.append((name, out))
    print(f"xcheck: {len(disp)} dispatches", flush=True)
    img, log = build.run(ar, kern, disp, require_double=True, workdir=str(Path(build.BUILD) / "input-xcheck"))
    ok = True
    for name, out in recs:
        w = img[out:out + 20].copy().view(np.uint32)
        row = dict(xcheck=name, inputs=int(w[4]), differ=int(w[0]), fast_flags=int(w[1]), binary64_unproven=int(w[2]),
                   largest_differing_pattern=hex(int(w[3]) - 1) if w[3] else None)
        ok = ok and row["differ"] == 0
        json_rows.append(row)
        print(row, flush=True)
    return ok


def run_hwerr(json_rows, chunk=1 << 24):
    """Every f32 input of the used domains: distance (in f32 steps) of the hardware exp2 / log2 (as compiled in the
    kernels: FN(in_ftz(exp2(t))), FN(log2(x))) from the correctly rounded ones, and the inputs where only one is
    non-finite or zero."""
    ar = build.Arena()
    kern = {"hwerr": (build.compile_kernel("rr_input.hlsl", "rr_input_hwerr", {"WG_BLOCK": 256, "RR_IN_EXACT": 1}), 1)}
    plog, pexp = ff_offsets(ar)
    disp, recs = [], []
    for name, op, lo, hi in (("exp2_pos", 0, 0x00000000, 0x43000000), ("exp2_neg", 0, 0x80000000, 0xC2FC0001),
                             ("log2", 1, 0x00800000, 0x7F800000)):
        out = ar.reserve(32)
        for b in range(lo, hi, chunk):
            n = min(chunk, hi - b)
            disp.append(("hwerr", (n + 255) // 256, [ar.add(np.array([op, b, n, out, plog, pexp], np.uint32))]))
        recs.append((name, out))
    img, log = build.run(ar, kern, disp, require_double=True, workdir=str(Path(build.BUILD) / "input-hwerr"),
                         timeout=36000)
    for name, out in recs:
        w = img[out:out + 20].copy().view(np.uint32)
        row = dict(hwerr=name, inputs=int(w[2]), max_steps=int(w[0]), special_mismatch=int(w[1]),
                   exact_unproven=int(w[3]), at_pattern=hex(int(w[4]) - 1) if w[4] else None)
        json_rows.append(row)
        print(row, flush=True)
    return True


def run_mlp_diff(json_rows, mutation=False, ntok=1 << 18):
    """Front-end MLP on random e4m3 feature codes (every finite code, and a set of small codes that give exact sums
    and binary16 ties) with the synthetic FE weights: variant X with the exact-sum fast path (RR_STEP 3) against plain
    variant X (RR_STEP 1), x0 bit for bit. mutation: the fast path's negative control (must differ)."""
    from ..tests import synth, synth_input as SI
    from .stage import wc_dwords
    rng = np.random.default_rng(5252)
    ar = build.Arena()
    kern = {}
    for name, sd, mut in (("x1", 1, 0), ("x3", 3, int(mutation))):
        kern[name] = (build.compile_kernel("rr_input_mlp.hlsl", "rr_input_mlp",
                                           {"WG_BLOCK": 128, "RR_STEP": sd, "RR_STEP3_MUT": mut},
                                           tag=f"rr_input_mlp_diff{sd}{mut}"), 13)
    disp, outs = [], []
    finite = np.array([c for c in range(256) if (c & 0x7f) != 0x7f], np.uint8)
    for mode in range(3):
        if mode == 0:
            codes = rng.choice(finite, (ntok, 64))
        elif mode == 1:                                  # small exponents: exact sums, binary16 ties
            codes = (rng.integers(0, 2, (ntok, 64)) * 128 + rng.integers(40, 72, (ntok, 64))).astype(np.uint8)
        else:                                            # mixed: half small, half full range
            codes = np.where(rng.random((ntok, 64)) < 0.5, rng.choice(finite, (ntok, 64)),
                             (rng.integers(48, 64, (ntok, 64))).astype(np.uint8)).astype(np.uint8)
        IN0 = SI.in0_group(rng)
        wc, _ = synth.window_consts(rng, "A")
        FE = IS.fe_tensors(IN0)
        w1 = np.stack([layout.frags(np.asarray(FE[f"W1_{br}"], np.uint8).T) for br in (0, 1)])
        w2 = np.stack([layout.frags(np.asarray(FE[f"W2_{br}"], np.uint8).T) for br in (0, 1)])
        pW1, pW2 = ar.add(w1), ar.add(w2)
        pB1 = ar.add(np.concatenate([layout.f32(FE["b1_0"]), layout.f32(FE["b1_1"])]))
        pB2 = ar.add(layout.f32(FE["b2"]))
        pc = ar.add(np.ascontiguousarray(codes))
        res = []
        for name in ("x1", "x3"):
            px0 = ar.reserve(ntok * 32, fill=0xEE)
            dg = ar.reserve(16)
            disp.append((name, ntok // 64, [pc, pW1, pW2, pB1, pB2, px0, ntok] + wc_dwords(wc) + [dg]))
            res.append((px0, dg))
        outs.append((mode, res))
    img, log = build.run(ar, kern, disp, workdir=str(Path(build.BUILD) / "input-mlpdiff"))
    ok = True
    for mode, ((a, _), (b, dg)) in outs:
        xa, xb = img[a:a + ntok * 32], img[b:b + ntok * 32]
        diff = int((xa != xb).sum())
        row = dict(mlp_diff_mode=mode, tokens=ntok, x0_codes=ntok * 32, differing=diff,
                   exact5_wave_steps=int(img[dg:dg + 4].view(np.uint32)[0]), wave_steps=ntok // 16 * 12)
        ok = ok and (diff == 0) != mutation
        json_rows.append(row)
        print(row, flush=True)
    return ok


def run_f2hcheck(json_rows, chunk=1 << 24):
    """f2h_ags (the driver's RNE conversion) against wg_f2h for every f32 pattern (NaNs excepted)."""
    ar = build.Arena()
    kern = {"f2h": (build.compile_kernel("rr_input.hlsl", "rr_input_f2hcheck", {"WG_BLOCK": 256}), 1)}
    out = ar.reserve(16)
    disp = [("f2h", chunk // 256, [ar.add(np.array([b, chunk, out], np.uint32))]) for b in range(0, 1 << 32, chunk)]
    img, _ = build.run(ar, kern, disp, workdir=str(Path(build.BUILD) / "input-f2h"), timeout=36000)
    w = img[out:out + 12].view(np.uint32)
    row = dict(f2hcheck="all f32", checked=int(w[1]) + (1 << 32 if int(w[1]) == 0 else 0), differing=int(w[0]),
               largest=hex(int(w[2]) - 1) if w[2] else None)
    json_rows.append(row)
    print(row, flush=True)
    return row["differing"] == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--step", default="X", choices=("E", "X", "X1", "S"))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cases", default="all")
    ap.add_argument("--xcheck", action="store_true", help="exhaustive fast-path cross-check (all f32 inputs)")
    ap.add_argument("--mark-all", action="store_true", help="every token through the exact replay (test variant)")
    ap.add_argument("--hwerr", action="store_true", help="exhaustive error of the hardware exp2 / log2 (then exit)")
    ap.add_argument("--mlp-diff", action="store_true", help="MLP fast path against plain variant X (then exit)")
    ap.add_argument("--f2hcheck", action="store_true", help="driver f32 -> f16 conversion for all f32 (then exit)")
    ap.add_argument("--mutation", choices=("hcert", "mlp"),
                    help="negative control (must fail): hcert = certified feature codes without their error bound, "
                         "mlp = every MLP sum taken as exact")
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    report = dict(step=a.step, cases=[], probe=[], xcheck=[], hwerr=[])
    if a.f2hcheck:
        ok = run_f2hcheck(report["xcheck"])
        Path(a.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print("PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1
    if a.mlp_diff:
        ok = run_mlp_diff(report["xcheck"], mutation=a.mutation == "mlp")
        Path(a.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print("PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1
    if a.hwerr:
        run_hwerr(report["hwerr"])
        Path(a.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        return 0
    ok = True
    if a.xcheck:
        ok = run_xcheck(report["xcheck"]) and ok
    if a.probe:
        ok = run_probe(report["probe"]) and ok
    cl = case_list()
    sel = range(len(cl)) if a.cases == "all" else [int(v) for v in a.cases.split(",")]
    ar = build.Arena()
    kern = kernels(a.step, mark_all=a.mark_all, mutation=a.mutation)
    calls, disp = [], []
    t0 = time.time()
    for i in sel:
        frame, params, S, IN0, consts, wc = make(i, cl[i])
        ref = IS.input_stage(frame, params, S, IN0, consts, wc)
        call = InputCall(ar, frame, params, S, IN0, wc)
        disp += call.dispatches()
        calls.append((i, cl[i], call, ref))
    print(f"prepared {len(calls)} cases in {time.time() - t0:.0f} s, arena {ar.size / 1e6:.1f} MB", flush=True)
    img, log = build.run(ar, kern, disp, require_double=True, workdir=str(Path(build.BUILD) / "input-stage"))
    for i, kw, call, ref in calls:
        res = compare(call.outputs(img), ref)
        res.update(case=i, config={k: v for k, v in kw.items()}, render=[call.k.Wr, call.k.Hr],
                   tensor=[call.k.Wt, call.k.Ht], m2_mode=call.mode, m2_src=ref["m2_src"])
        report["cases"].append(res)
        ok = ok and res["exact"]
        print(f"case {i:2d} {str(kw)[:60]:60s} {'EXACT' if res['exact'] else 'DIFF '} " +
              " ".join(f"{n}:{v['diff']}" for n, v in res.items() if isinstance(v, dict) and "diff" in v) +
              f" ziv:{res.get('ziv_unproven')}", flush=True)
    report["passed"] = ok
    report["host_log"] = log
    report["shaders"] = {n: hashlib.sha256(Path(v[0]).read_bytes()).hexdigest() for n, v in kern.items()}
    Path(a.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
