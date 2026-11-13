# SPDX-License-Identifier: MIT
"""Test-vector generator and reader for the D0-R stage / output head (spec 55, with the page-51 block).

    python -m clean.rr_f.head.vectors [--out DIR] [--check]

writes DIR (default clean/rr_f/head/tests/vectors): manifest.json and one or more `.hex` files per array.
`--check` regenerates into memory and compares with the files in DIR (exit 1 on any difference).

All weights, constants, scalars and inputs are SYNTHETIC test inputs chosen here (the real converter-supplied
constants are not available to the clean team). The file format is described in manifest.json ("format").
"""
import argparse
import json
import os
import sys

import numpy as np

from . import block as B
from . import exactfp as X
from . import fp
from . import head as Hd

f64 = np.float64
f32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "tests", "vectors")

PER_LINE = 16
MAX_ELEMS = {"u8": 60000, "i8": 60000, "e4m3": 60000, "f16": 36000, "u16": 36000, "f32": 20000}
WIDTH = {"u8": 2, "i8": 2, "e4m3": 2, "f16": 4, "u16": 4, "f32": 8}

FORMAT = {
    "files": "each array is stored in one or more text files (parts, concatenated in the listed order); a file holds "
             "whitespace-separated lower-case hexadecimal words, 16 per line, in C (row-major) order of the stated "
             "shape; no 0x prefix",
    "dtypes": {
        "e4m3": "2 hex digits: the e4m3 code byte (page 51.1)",
        "u8": "2 hex digits: unsigned 8-bit code",
        "i8": "2 hex digits: two's-complement signed 8-bit code (ff = -1)",
        "f16": "4 hex digits: IEEE binary16 bit pattern (NaN written as 7e00 / fe00)",
        "f32": "8 hex digits: IEEE binary32 bit pattern",
    },
    "layout": "tensors are [Y][X][channel] with absolute token coordinates; window-indexed arrays use the window "
              "order j-major, i-minor (window (i, j) covers X = 8i - sx + lx, Y = 8j - sy + ly) and t = 8*ly + lx",
    "weights": "page 51.2 logical tensors: Wq, Wv [H][C][32]; Tb [H][12][16][16] (block k = k-th pair of "
               "tb_pair_order); Wo [H][32][C]; bo [C]; W1 [C][4C]; b1 [4C]; W2 [4C][C]; b2 [C]",
    "f16_steps": "distance on the ordered line of binary16 values; +0 and -0 are the same point; +-inf follow "
                 "+-65504; NaN matches only NaN (any NaN bit pattern)",
}


# ------------------------------------------------------------------------------------------------ file io
def _words(arr, dtype):
    a = np.asarray(arr)
    if dtype == "f16":
        u = fp.bits16(a).astype(np.uint64)
    elif dtype == "f32":
        u = np.asarray(a, f32).view(np.uint32).astype(np.uint64)
    elif dtype == "i8":
        u = np.asarray(a, np.int8).view(np.uint8).astype(np.uint64)
    else:
        u = np.asarray(a, np.uint8 if dtype in ("u8", "e4m3") else np.uint16).astype(np.uint64)
    w = WIDTH[dtype]
    return [format(int(v), "0%dx" % w) for v in u.reshape(-1)]


def encode(arr, dtype):
    """Text parts for an array."""
    words = _words(arr, dtype)
    n = MAX_ELEMS[dtype]
    parts = []
    for p0 in range(0, max(len(words), 1), n):
        chunk = words[p0:p0 + n]
        lines = [" ".join(chunk[i:i + PER_LINE]) for i in range(0, len(chunk), PER_LINE)]
        parts.append("\n".join(lines) + "\n")
    return parts


def decode(texts, dtype, shape):
    words = " ".join(texts).split()
    u = np.array([int(w, 16) for w in words], np.uint64)
    if dtype == "f16":
        a = u.astype(np.uint16).view(np.float16).astype(f64)
    elif dtype == "f32":
        a = u.astype(np.uint32).view(f32)
    elif dtype == "i8":
        a = u.astype(np.uint8).view(np.int8)
    elif dtype in ("u8", "e4m3"):
        a = u.astype(np.uint8)
    else:
        a = u.astype(np.uint16)
    return a.reshape(shape)


class Writer:
    def __init__(self):
        self.files = {}      # relative path -> text

    def array(self, rel, arr, dtype, layout=None, **extra):
        parts = encode(arr, dtype)
        names = []
        for i, t in enumerate(parts):
            name = rel + (".hex" if len(parts) == 1 else ".part%d.hex" % i)
            self.files[name] = t
            names.append(name)
        d = {"dtype": dtype, "shape": list(np.shape(arr)), "files": names}
        if layout:
            d["layout"] = layout
        d.update(extra)
        return d


def load_array(vdir, entry):
    texts = []
    for name in entry["files"]:
        with open(os.path.join(vdir, name), encoding="ascii") as fh:
            texts.append(fh.read())
    return decode(texts, entry["dtype"], tuple(entry["shape"]))


def scalar_entry(v, dtype):
    if dtype == "f16":
        return {"dtype": "f16", "bits": format(int(fp.bits16(np.array([v]))[0]), "04x"), "value": float(v)}
    return {"dtype": "f32", "bits": format(int(np.array([v], f32).view(np.uint32)[0]), "08x"), "value": float(v)}


def scalar_value(e):
    b = int(e["bits"], 16)
    if e["dtype"] == "f16":
        return float(fp.from_bits16(np.array([b]))[0])
    return f32(np.array([b], np.uint32).view(f32)[0])


# ------------------------------------------------------------------------------------------------ synthetic data
def const_set(kind):
    """Synthetic window-block constants (f16). 'base': non-power-of-two values, p in [1.15, 1.65];
    'zovf': p up to 1.875 (w_max = 8192), so that 64 * w_max overflows f16 and rows with several weights near the
    top give Z = +inf."""
    r = lambda v: float(fp.r16(np.array([v]))[0])
    k = dict(A_SCALE=r(0.0884), A_CLAMP=r(0.5), A_CLAMPn=r(-0.4688), A_P1=r(0.7505), A_P0=r(1.4),
             ACT_CL=r(3.0), ACT_CLn=r(-3.254), ACT_C1=r(0.0723), ACT_C2=r(0.3965), ACT_C3=r(0.5))
    if kind == "zovf":
        k.update(A_SCALE=r(0.125), A_P0=r(1.55), A_P1=r(0.9))
    return k


def make_weights(rng, C=32, H=2, gain=1.0, mode="normal"):
    if mode == "allcodes":       # every finite e4m3 code with equal probability (huge dynamic range)
        codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], np.uint8)
        pick = lambda shape: rng.choice(codes, size=shape)
        Tbv = fp.r16(rng.normal(0, 4.0, (H, 12, 16, 16)))
        return {"Wq": pick((H, C, 32)), "Wv": pick((H, C, 32)), "Tb": Tbv, "Wo": pick((H, 32, C)),
                "bo": fp.r16(rng.normal(0, 8, C)), "W1": pick((C, 4 * C)), "b1": fp.r16(rng.normal(0, 8, 4 * C)),
                "W2": pick((4 * C, C)), "b2": fp.r16(rng.normal(0, 8, C))}
    s = C ** -0.5
    return {
        "Wq": fp.q8(rng.normal(0, 1.7 * s * gain, (H, C, 32))),
        "Wv": fp.q8(rng.normal(0, s, (H, C, 32))),
        "Tb": fp.r16(rng.normal(0, 0.5, (H, 12, 16, 16))),
        "Wo": fp.q8(rng.normal(0, (32 * H) ** -0.5, (H, 32, C))),
        "bo": fp.r16(rng.normal(0, 0.1, C)),
        "W1": fp.q8(rng.normal(0, s, (C, 4 * C))),
        "b1": fp.r16(rng.normal(0, 0.1, 4 * C)),
        "W2": fp.q8(rng.normal(0, 1.5 * (4 * C) ** -0.5, (4 * C, C))),
        "b2": fp.r16(rng.normal(0, 0.3, C)),
    }


def f16v(v):
    return float(fp.r16(np.array([v]))[0])


def head_edge_y2h(rng, ht, wt):
    """y2h for the head-only edge case: every token gets random values in all channels, then the used channels are
    overwritten with edge values from curated lists (each list cycled over the tokens with its own stride)."""
    y = fp.r16(rng.normal(0, 3, (ht, wt, 32)))
    inf, nan = np.inf, np.nan
    tanh_half = fp.all_f16()[fp.tanh16_table() == 0.5]           # R(tanh(v)) = 0.5: HID +-64 half-way codes
    sig_half = fp.all_f16()[fp.sig_table() == 0.5]               # sig rounds to 0.5: code 128
    sig_edges = [0.0, -0.0, inf, -inf, nan, 65504, -65504, 2 ** -24, -2 ** -24, 17.0, -17.0, 16.0, -16.0,
                 f16v(8.32), f16v(-8.32), f16v(11.09), f16v(-11.09), f16v(24.0), f16v(-24.0), f16v(-103.0),
                 f16v(-104.0), 1.0, -1.0, 0.5, -0.5, f16v(1e-3), f16v(-1e-3)] + list(sig_half[::40])
    tanh_edges = [0.0, -0.0, inf, -inf, nan, 65504, -65504, 2 ** -24, -2 ** -24, 2 ** -14, f16v(4.5), f16v(-4.5),
                  f16v(4.16), f16v(-4.16), f16v(3.8), 1.0, -1.0, f16v(0.0039), f16v(-0.0039)] + list(tanh_half) + list(-tanh_half)
    a = [f16v(v) for v in (0.0, 1.0, -1.0, 3.0, -2.5, 1024.0, 2 ** -14, 0.0999, 65504.0, -65504.0)]
    cand_pairs = []
    for v in a:
        cand_pairs.append((v, v))                                  # d = 0 -> four values 0.25
    for v in (0.0, 1.0, -3.0, 100.0):                              # tiny differences: tie cases of page 54.4
        for kk in (1, 2, 3, 5, 8, 16, 40, 64, 128):
            nb = float(fp.from_bits16(fp.bits16(np.array([v])) + kk)[0]) if v >= 0 else \
                float(fp.from_bits16(fp.bits16(np.array([v])) + kk)[0])
            cand_pairs.append((v, nb))
            cand_pairs.append((nb, v))
    for dd in (2 ** -12, 2 ** -10, 2 ** -8, 2 ** -6, 0.0078, 0.0156, 0.03125):
        cand_pairs.append((f16v(dd), 0.0))
        cand_pairs.append((0.0, f16v(dd)))
    cand_pairs += [(inf, inf), (-inf, -inf), (inf, -inf), (-inf, inf), (inf, 0.0), (0.0, inf), (-inf, 5.0),
                   (5.0, -inf), (nan, 0.0), (0.0, nan), (nan, nan), (nan, inf),
                   (1024.0, 2 ** -14), (65504.0, 2 ** -24), (65504.0, -65504.0), (-65504.0, 65504.0),
                   (2 ** -24, -(2 ** -24)), (20.0, -20.0), (-60.0, 60.0)]
    prec_edges = sig_edges
    rho_edges = [0.0, -0.0, inf, -inf, nan, 65504, -65504, 2 ** -24, f16v(0.3), f16v(-0.3), f16v(9.0), f16v(-9.0),
                 f16v(1e-4), 1.0, -1.0]
    n = ht * wt
    flat = y.reshape(n, 32)
    pick = lambda lst, i, stride: lst[(i * stride) % len(lst)]
    # the edge values go to the channels the head reads (page 55 erratum numbering, head.head_channels); the list
    # positions and strides are keyed by the page's channel numbers so the value sequences are unchanged
    nat = Hd.p55_natural if Hd.P55_ERRATUM else (lambda c: c)
    for i in range(n):
        for c, stride in ((0, 1), (1, 3), (4, 5), (5, 7), (8, 1), (9, 2), (12, 4)):
            if i % 3 != 2:
                flat[i, nat(c)] = pick(sig_edges, i + c, stride)
        for c, stride in ((28, 1), (2, 3)):
            if i % 4 != 3:
                flat[i, nat(c)] = pick(tanh_edges, i + c, stride)
        if i % 5 != 4:
            flat[i, nat(13)] = pick(prec_edges, i, 1)
            flat[i, nat(16)] = pick(prec_edges, i + 7, 3)
            flat[i, nat(17)] = pick(rho_edges, i, 1)
        y20, y24 = cand_pairs[i % len(cand_pairs)]
        if i % 7 != 6:
            flat[i, nat(20)], flat[i, nat(24)] = y20, y24
    return flat.reshape(ht, wt, 32)


# ------------------------------------------------------------------------------------------------ cases
CASES = [
    # name, kind, Wt, Ht, Wr, Hr, SH row, const set, weight mode, x mode, h0, h1, S1CAP, seed, note
    dict(name="stage_basic", kind="stage", wt=32, ht=32, wr=32, hr=32, row=(2, 6), consts="base", wmode="normal",
         xmode="normal", h0=1.37, h1=-0.3, s1cap=20.5, seed=101,
         note="random weights and inputs, full render size, recurrent shift (6, 2)"),
    dict(name="stage_border", kind="stage", wt=64, ht=32, wr=57, hr=19, row=(0, 0), consts="base", wmode="normal",
         xmode="normal", h0=0.7, h1=0.45, s1cap=20.5, seed=102,
         note="non-square tensor, render size smaller than the tensor (tokens X >= Wr / Y >= Hr write nothing), "
              "recurrent shift (4, 4); generic hold[.][3] != 0"),
    dict(name="stage_shift", kind="stage", wt=32, ht=64, wr=31, hr=64, row=(6, 4), consts="base", wmode="normal",
         xmode="wide", h0=4.0, h1=-1.1, s1cap=20.5, seed=103,
         note="recurrent shift (2, 0); s1 capped (expf(4) > S1CAP); wider input distribution"),
    dict(name="stage_zovf", kind="stage", wt=32, ht=32, wr=30, hr=29, row=(4, 2), consts="zovf", wmode="zovf",
         xmode="normal", h0=-0.62, h1=1.9, s1cap=20.5, seed=104,
         note="constants with 64 * w_max >= 65520: rows with Z = +inf (r = +0, zero P codes, zero value mixing)"),
    dict(name="stage_edge", kind="stage", wt=32, ht=32, wr=32, hr=32, row=(0, 6), consts="base", wmode="allcodes",
         xmode="allcodes", h0=2.25, h1=0.125, s1cap=6.0, seed=105,
         note="every finite e4m3 code in weights and inputs: f16 overflow to +-inf / NaN inside the block, Q8 "
              "saturation, +-inf and NaN channel values reaching the head"),
    dict(name="stage_nan", kind="stage", wt=32, ht=32, wr=27, hr=32, row=(2, 2), consts="base", wmode="normal",
         xmode="nan", h0=1.37, h1=-0.3, s1cap=20.5, seed=106,
         note="every input channel of every token NaN (both NaN codes): every PREC, GATES and CANDW value NaN, "
              "codes 0; HID codes 0; PREC values 0 and 1 are not s1^2"),
    dict(name="head_edge", kind="head", wt=32, ht=32, wr=29, hr=30, row=None, consts=None, wmode=None, xmode=None,
         h0=1.37, h1=-0.3, s1cap=20.5, seed=107,
         note="head only: y2h given; edge channel values (+-0, +-inf, NaN, +-65504, subnormals, saturation "
              "thresholds, sig = 0.5, R(tanh) = +-0.5), CANDW pairs (equal, tie cases, infinite of equal and "
              "opposite sign, NaN, inexact d)"),
    dict(name="head_random", kind="head", wt=32, ht=32, wr=32, hr=32, row=None, consts=None, wmode=None,
         xmode=None, h0=f16v(1.0137), h1=f16v(-0.7651), s1cap=20.5, seed=108,
         note="head only: random y2h, hold, Hc; non-power-of-two s1, s2 (rounding order of the PREC products)"),
]


def _inputs(rng, case):
    wt, ht = case["wt"], case["ht"]
    hold = fp.r16(rng.normal(0, 1.0, (ht, wt, 4)))
    if case["name"] != "stage_border":
        hold[..., 3] = 0.0                                   # IN0's hnew[3] = 0 (page 51.4 encoder rule)
    hc = fp.r16(rng.normal(0, 1.2, (ht // 2, wt // 2, 4)))
    return hold, hc


def build_case(case, w, vdir_rel):
    rng = np.random.default_rng(case["seed"])
    name = case["name"]
    wt, ht, wr, hr = case["wt"], case["ht"], case["wr"], case["hr"]
    h0, h1 = f16v(case["h0"]), f16v(case["h1"])
    s1cap = f32(case["s1cap"])
    s1, s2 = Hd.derive_scalars(h0, h1, s1cap)
    g = f16v(0.9414 if case["seed"] % 2 else 1.2969)
    entry = {"name": name, "kind": case["kind"], "description": case["note"],
             "tensor_extent": {"Wt": wt, "Ht": ht}, "render_size": {"Wr": wr, "Hr": hr},
             "scalars": {"g": scalar_entry(g, "f16"), "h0": scalar_entry(h0, "f16"), "h1": scalar_entry(h1, "f16"),
                         "S1CAP": scalar_entry(s1cap, "f32"), "s1": scalar_entry(s1, "f32"),
                         "s2": scalar_entry(s2, "f32")},
             "inputs": {}, "outputs": {}, "diagnostics": {}}
    p = lambda n: "%s/%s/%s" % (vdir_rel, name, n) if vdir_rel else "%s/%s" % (name, n)
    L = "[Y][X][c]"
    diag = {}
    if case["kind"] == "stage":
        k = const_set(case["consts"])
        ok, wmax, ovf = B.check_attention_constants(k)
        assert ok
        if case["wmode"] == "allcodes":
            W = make_weights(rng, mode="allcodes")
        elif case["wmode"] == "zovf":
            W = make_weights(rng, gain=0.3)        # about a third of the rows get Z = +inf
        else:
            W = make_weights(rng)
        if case["xmode"] == "allcodes":
            codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], np.uint8)
            x = rng.choice(codes, size=(ht, wt, 32))
        elif case["xmode"] == "nan":
            x = rng.choice(np.array([0x7F, 0xFF], np.uint8), size=(ht, wt, 32))
        else:
            x = fp.q8(rng.normal(0, 2.5 if case["xmode"] == "wide" else 1.0, (ht, wt, 32)))
        hold, hc = _inputs(rng, case)
        ra, m = f16v(0.8516 if case["seed"] % 2 else 1.1719), f16v(0.6797 if case["seed"] % 2 else 0.4473)
        shift = B.recurrent_shift(case["row"])
        out = Hd.d0r_stage(x, W, k, ra, m, g, shift, hold, hc, s1, s2, wr, hr, diag)
        y2h = diag["y2h"]
        entry["shift"] = {"SH_row": list(case["row"]), "recurrent_shift": list(shift),
                          "rule": "((s + 4) mod 8) per component (page 51.6)"}
        entry["constants"] = {n: scalar_entry(k[n], "f16") for n in B.CONST_NAMES}
        entry["constants"]["ra"] = scalar_entry(ra, "f16")
        entry["constants"]["m"] = scalar_entry(m, "f16")
        entry["converter_check"] = {"attention_range_ok": ok, "w_max": wmax, "z_can_overflow": bool(ovf)}
        entry["inputs"]["x"] = w.array(p("x"), x, "e4m3", L, what="D0-F output tensor (block input)")
        for n in ("Wq", "Wv", "Wo", "W1", "W2"):
            entry["inputs"][n] = w.array(p(n), W[n], "e4m3")
        for n in ("Tb", "bo", "b1", "b2"):
            entry["inputs"][n] = w.array(p(n), W[n], "f16")
        entry["inputs"]["Tb"]["tb_pair_order"] = [list(t) for t in B.TB_PAIRS]
        Xw, _ = diag["windows"]
        entry["diagnostics"]["y2h"] = w.array(p("y2h"), y2h, "f16", L, compare=_cmp("y2h"))
        entry["diagnostics"]["Z"] = w.array(p("Z"), diag["Z"], "f16", "[window][head][query token]",
                                            windows=int(Xw.shape[0]),
                                            what="row sums Z of 51.3 step 4 (exact sum rounded once)")
        nz = int(np.sum(np.isinf(diag["Z"])))
        entry["diagnostics"]["Z"]["rows_with_Z_inf"] = nz
    else:
        y2h = head_edge_y2h(rng, ht, wt) if case["name"] == "head_edge" else fp.r16(rng.normal(0, 2.0, (ht, wt, 32)))
        hold, hc = _inputs(rng, case)
        out = Hd.output_images(y2h, s1, s2, wr, hr)
        out["Cnet"] = Hd.hidden_state(y2h, hold, hc, g)
        entry["inputs"]["y2h"] = w.array(p("y2h"), y2h, "f16", L, what="block output before e4m3 quantisation")
    entry["inputs"]["hold"] = w.array(p("hold"), hold, "f16", L, what="own hidden: IN0's level-0 hnew, same frame")
    entry["inputs"]["Hc"] = w.array(p("Hc"), hc, "f16", L, what="coarse hidden: D1-R's hnew (level 1), same frame")
    entry["outputs"]["Cnet"] = w.array(p("Cnet"), out["Cnet"], "f16", L, compare=_cmp("Cnet"))
    for n in ("HID", "GATES", "PREC", "CANDW"):
        entry["outputs"][n] = w.array(p(n), out[n], "f16", "[Y][X][value]", compare=_cmp(n))
    entry["outputs"]["HID_code"] = w.array(p("HID_code"), out["HID_code"], "i8", "[Y][X][value]",
                                           compare=_cmp("HID_code"))
    for n in ("GATES_code", "CANDW_code"):
        entry["outputs"][n] = w.array(p(n), out[n], "u8", "[Y][X][value]", compare=_cmp(n))
    entry["counts"] = _counts(out)
    return entry


def _counts(out):
    c = {}
    for n in ("Cnet", "HID", "GATES", "PREC", "CANDW"):
        v = out[n]
        c[n] = {"nan": int(np.isnan(v).sum()), "inf": int(np.isinf(v).sum()), "zero": int((v == 0).sum()),
                "total": int(v.size)}
    return c


def _cmp(n):
    if n == "y2h":
        return {"rule": "f16_steps", "max": 0,
                "note": "bit-exact when the implementation uses the reference row sums Z (diagnostic Z); with its "
                        "own Z order, page 51.3 step 4 accepts one-step Z differences and their consequences "
                        "(51.8 item 1). Zero signs not compared."}
    if n in ("Cnet", "HID", "GATES", "PREC"):
        return {"rule": "f16_steps", "max": 1, "target": "bit-exact",
                "note": "page 55.3 test-vector tolerance (Cnet: 55.5 'hidden state' with the same tolerance); "
                        "NaN must be NaN; zero signs not compared"}
    if n == "CANDW":
        return {"rule": "f16_steps", "max": 3, "provisional": True,
                "exact": "per token: the two values of each pair identical; equal finite CANDW logits (natural "
                         "y10 == y12, page 55's y20 == y24) -> four values "
                         "0.25; an infinite logit (or two of opposite sign) -> (1/2, 1/2, 0, 0) or (0, 0, 1/2, "
                         "1/2); two infinite logits of the same sign or a NaN logit -> four NaN",
                "note": "page 55.3 (3 f16 steps provisional, 55.6 item 2)"}
    if n == "HID_code":
        return {"rule": "store_of_own_values", "store": "snorm8 (page 53.3 / 55.3): NaN -> 0, clamp [-1, 1], "
                "round(c * 127), +-0.5 -> +-64", "max_code_diff": 1,
                "note": "codes are checked by applying the store rule to the implementation's own f16 values; the "
                        "listed codes are the reference's (within one code of an implementation within tolerance)"}
    return {"rule": "store_of_own_values", "store": "unorm8 (page 55.3): NaN -> 0, clamp [0, 1], round(c * 255), "
            "0.5 -> 128", "max_code_diff": 1,
            "note": "codes are checked by applying the store rule to the implementation's own f16 values"}


def build_scalars(w, vdir_rel):
    """Converter rule of 55.3 for s1, s2 on a list of (h0, h1, S1CAP)."""
    rows = []
    for h0, h1, cap in [(1.37, -0.3, 20.5), (0.0, 0.0, 20.5), (4.0, -1.1, 20.5), (2.9961, 2.9961, 20.0),
                        (11.0, -9.0, 1000.0), (100.0, 1.0, 20.5), (65504.0, 0.5, 3.75), (-65504.0, -65504.0, 20.5),
                        (-87.0, -86.5, 20.5), (-90.0, 1.0, 20.5), (-7.5, 88.5, 1.0e30), (0.0999, -0.0999, 1.1052), (1.0137, -0.7651, 20.5)]:
        h0v, h1v = f16v(h0), f16v(h1)
        capv = f32(cap)
        try:
            s1, s2 = Hd.derive_scalars(h0v, h1v, capv)
            rows.append({"h0": scalar_entry(h0v, "f16"), "h1": scalar_entry(h1v, "f16"),
                         "S1CAP": scalar_entry(capv, "f32"), "s1": scalar_entry(s1, "f32"),
                         "s2": scalar_entry(s2, "f32")})
        except ValueError:
            rows.append({"h0": scalar_entry(h0v, "f16"), "h1": scalar_entry(h1v, "f16"),
                         "S1CAP": scalar_entry(capv, "f32"), "refused": "s2 not finite"})
    return {"name": "converter_scalars", "kind": "scalars",
            "description": "s1 = min(expf(h0), S1CAP), s2 = expf(h1), expf correctly rounded; refused when s2 is "
                           "not finite (55.3). Exact (bit-for-bit).", "rows": rows}


def build_tables(w, vdir_rel):
    p = lambda n: "%s/tables/%s" % (vdir_rel, n) if vdir_rel else "tables/%s" % n
    allv = fp.all_f16()
    t = {"name": "tables", "kind": "tables",
         "description": "exhaustive scalar rules, index = the f16 input bit pattern (0000 .. ffff) or the code; "
                        "all exact (bit-for-bit)", "arrays": {}}
    t["arrays"]["sig"] = w.array(p("sig"), fp.sig_table(), "f16", "[f16 bits of v]",
                                 what="R(sig(v)) of page 51.4 / 55.1")
    t["arrays"]["tanh16"] = w.array(p("tanh16"), fp.tanh16_table(), "f16", "[f16 bits of v]",
                                    what="R(tanh(v)), correctly rounded (HID)")
    t["arrays"]["unorm8_store"] = w.array(p("unorm8_store"), fp.unorm8_store(allv), "u8", "[f16 bits of v]",
                                          what="GATES / CANDW code of the f16 value v")
    t["arrays"]["snorm8_store"] = w.array(p("snorm8_store"), fp.snorm8_store(allv), "i8", "[f16 bits of v]",
                                          what="HID code of the f16 value v")
    t["arrays"]["unorm8_read"] = w.array(p("unorm8_read"), fp.unorm8_read(np.arange(256)), "f16", "[code]",
                                         what="read-back: f16 nearest to k/255")
    t["arrays"]["q8"] = w.array(p("q8"), fp.q8(allv), "e4m3", "[f16 bits of v]", what="Q8 of page 51.1")
    return t


def generate():
    w = Writer()
    man = {"spec": "pages 55 (D0-R head) and 51 (window block at level 0), drafts 7 / 6",
           "generator": "python -m clean.rr_f.head.vectors",
           "synthetic": "all weights, constants (window-block constants, ra, m, g, h0, h1, S1CAP) and inputs are "
                        "synthetic test inputs chosen by this generator; none comes from model data",
           "format": FORMAT,
           "assumptions": "see clean/rr_f/head/QUESTIONS-55.md (provisional readings)",
           "channels": "block-output channel picks of 55.2 / 55.3 in the page 55 erratum's natural numbering: "
                       "gate logits 0, 1, 2, 3; HID tanh(y14), tanh(y16); GATES y4, y5, y6; PREC y7, y8, y9; "
                       "CANDW y10, y12",
           "cases": []}
    for case in CASES:
        man["cases"].append(build_case(case, w, ""))
    man["cases"].append(build_scalars(w, ""))
    man["cases"].append(build_tables(w, ""))
    w.files["manifest.json"] = json.dumps(man, indent=1, allow_nan=False) + "\n"
    return w.files


def load_manifest(vdir=DEFAULT_OUT):
    with open(os.path.join(vdir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = generate()
    if a.check:
        bad = []
        for rel, text in files.items():
            path = os.path.join(a.out, rel)
            if not os.path.exists(path) or open(path, encoding="ascii").read() != text:   # CRLF checkouts ok
                bad.append(rel)
        print("checked %d files: %d differ" % (len(files), len(bad)))
        for b in bad[:20]:
            print("  differs:", b)
        return 1 if bad else 0
    for rel, text in files.items():
        path = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
    total = sum(len(t) for t in files.values())
    print("wrote %d files, %d bytes, to %s" % (len(files), total, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
