# SPDX-License-Identifier: MIT
"""Synthetic cases for the converter tests (spec 08 sections 4-9). Written for this project."""

import copy
import random
import struct

from synth import (SHARED_CACHE, Case, NonFinite, f_source_entry, filler_text, finalize_map, half_bits,
                   keyed_fnv_state, make_f_variant, place, random_finite_halves, sha256, source_entry,
                   version_resource)
from synth import build_t_text as synth_build_t
from synth import t_source_entry as t_source_entry_

SQRT96 = "0x411cc471"
SQRT8 = "0x403504f3"


def lin(stride=1, perm=None):
    a = {"stride": stride}
    if perm:
        a["perm"] = perm
    return a


def tensor(name, shape, dtype, src, op, base, axes):
    return {"name": name, "shape": shape, "dtype": dtype, "src": src, "op": op, "base": base, "axes": axes}


def last_ok_pattern(op, start, stop):
    """Largest count n such that patterns start..start+n-1 all decode (they are monotonic in |x|)."""
    cache = SHARED_CACHE
    n = 0
    for h in range(start, stop):
        try:
            cache.decode(op, h)
        except NonFinite:
            break
        n += 1
    return n


# ---------------------------------------------------------------- the feature case

def feature_case(seed=7):
    """Every op over every relevant input (all finite binary16 patterns, all u16 values), perms,
    negative and zero strides, odd and even-but-not-16 placements, decoys and duplicates."""
    rng = random.Random(seed)
    # W: all 65536 binary16 patterns in order, then 4096 random finite values.
    w = bytearray(struct.pack("<65536H", *range(65536)))
    w += random_finite_halves(rng, 4096, max_exp_field=0x16)
    w = bytes(w)
    R = 65536  # first element of the random region
    d_values = list(range(65536))  # every u16 value once, shuffled so that D is not a copy of W's start
    rng.shuffle(d_values)
    d = struct.pack("<65536H", *d_values)
    k = struct.pack("<II", 0x3FA00000, 0x80000000)  # 1.25 and -0.0

    sc_a = {"name": "f16_scaled", "mul": SQRT96, "div": SQRT96}
    sc_b = {"name": "f16_scaled", "mul": SQRT8, "div": "0x3f800000"}
    sc_c = {"name": "f16_scaled", "mul": "0xbfc00000", "div": "0x40e00000"}  # mul -1.5, div 7
    sc_d = {"name": "f16_scaled", "mul": "0x3dcccccd", "div": "0x40400000"}  # mul 0.1f, div 3
    exp_op = {"name": "exp"}
    exp_max = {"name": "exp", "max": "0x447a0000"}  # 1000.0

    n_pos = 0x7C00  # finite positive patterns 0x0000..0x7BFF
    allhalf = []
    for sign, base in (("pos", 0), ("neg", 0x8000)):
        allhalf.append(tensor("f16_" + sign, [n_pos], "float32", "W", {"name": "f16"}, base, [lin(1)]))
        allhalf.append(tensor("fp8_" + sign, [n_pos], "float32", "W", {"name": "fp8"}, base, [lin(1)]))
        for nm, op in (("sca", sc_a), ("scb", sc_b), ("scc", sc_c), ("scd", sc_d)):
            n = last_ok_pattern(op, base, base + n_pos)
            allhalf.append(tensor(nm + "_" + sign, [n], "float32", "W", op, base, [lin(1)]))
    allhalf.append(tensor("exp_neg", [n_pos], "float32", "W", exp_op, 0x8000, [lin(1)]))
    allhalf.append(tensor("exp_pos", [last_ok_pattern(exp_op, 0, n_pos)], "float32", "W", exp_op, 0, [lin(1)]))
    allhalf.append(tensor("expmax_pos", [n_pos], "float32", "W", exp_max, 0, [lin(1)]))
    allhalf.append(tensor("exp_inf_max", [1], "float32", "W", exp_max, 0x7C00, [lin(0)]))  # +inf -> max
    allhalf.append(tensor("exp_minf", [1], "float32", "W", exp_op, 0xFC00, [lin(0)]))      # -inf -> 0

    ints = [
        tensor("shr9", [65536], "uint8", "D", {"name": "u16_shr", "shift": 9}, 0, [lin(1)]),
        tensor("shr8", [256, 256], "uint8", "D", {"name": "u16_shr", "shift": 8}, 0, [lin(256), lin(1)]),
        tensor("shr15", [4096], "uint8", "D", {"name": "u16_shr", "shift": 15}, 15, [lin(16)]),
        tensor("shr40", [16], "uint8", "D", {"name": "u16_shr", "shift": 40}, 65520, [lin(1)]),
        tensor("k_both", [2], "float32", "K", {"name": "u32_as_f32"}, 0, [lin(1)]),
        tensor("sigma1", [1], "float32", "K", {"name": "u32_as_f32"}, 0, [lin(0)]),
        tensor("sigma2", [1], "float32", "K", {"name": "u32_as_f32"}, 1, [lin(0)]),
    ]

    layout = [
        # Perm on a middle axis (like dec0 W) and on the last axis with a block shorter than the axis.
        tensor("perm_mid", [4, 17, 8], "float32", "W", {"name": "fp8"}, R, [lin(17 * 8), lin(8, "p17"), lin(1, "p4")]),
        tensor("perm_last", [3, 17], "float32", "W", {"name": "f16"}, R + 700, [lin(32), lin(1, "p17")]),
        tensor("perm_first", [8, 5], "float32", "W", sc_b, R + 900, [lin(5, "p4"), lin(1)]),
        tensor("neg_strides", [5, 6], "float32", "W", {"name": "f16"}, R + 1029, [lin(-6), lin(-1)]),
        tensor("broadcast", [3, 4], "float32", "W", {"name": "fp8"}, R + 1100, [lin(0), lin(1)]),
        tensor("transposed", [6, 5], "float32", "W", sc_a, R + 1200, [lin(1), lin(6)]),
        tensor("rank4", [2, 3, 4, 5], "float32", "W", {"name": "fp8"}, R + 1300, [lin(60), lin(20), lin(5), lin(1)]),
        tensor("T", [8, 16, 16], "uint8", "D", {"name": "u16_shr", "shift": 9}, 300, [lin(256, "p4"), lin(16), lin(1, "p17b")]),
        tensor("S", [1], "float32", "W", exp_max, R + 1500, [lin(0)]),
        tensor("scalar_f16", [1], "float32", "W", {"name": "f16"}, R + 4095, [lin(0)]),
    ]

    perms = {
        "p17": [0, 1, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 9],
        "p4": [2, 0, 3, 1],
        "p17b": [15, 3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],  # 16 entries; name is arbitrary
    }
    sources = {"W": w, "D": d, "K": k}
    key = sha256(b"test key label" + w)
    gaps = [3, 10, 25]
    m = {
        "format": "wronggpu-converter-map",
        "version": 1,
        "model": "synthetic feature case",
        "channel_order": "natural",
        "an_unknown_top_level_field": {"ignored": True},
        "sources": {
            "W": source_entry("W", "fake parameter block", "f16le", w),
            "D": source_entry("D", "fake dither table", "u16le", d),
            "K": source_entry("K", "fake constants", "u32le", k, locate="pair", gaps=gaps, keyed=True, key=key),
        },
        "key": {"label": "test key label", "from": "W"},
        "perms": perms,
        "stages": [
            {"stage": "allhalf", "digest": "", "tensors": allhalf},
            {"stage": "ints", "digest": "", "tensors": ints},
            {"stage": "layout", "digest": "", "tensors": layout},
        ],
    }
    for t in layout + allhalf:
        t["op"] = dict(t["op"], note="informational") if t["name"] in ("perm_mid", "S") else t["op"]

    size = 3_300_000
    buf = bytearray(rng.randbytes(size))
    # Decoys first in search order: W's first 64 bytes at a multiple of 16 with different content
    # after them (FNV hit, SHA-256 miss), and an almost complete copy with its last byte changed.
    place(buf, 4096, w[:64])
    decoy = bytearray(w)
    decoy[-1] ^= 0x01
    place(buf, 16 * 1000, decoy)
    # Real copies: one at an odd offset and one at an even offset that is not a multiple of 16. The
    # even one must be found (pass 2 before pass 3) although it is later in the file.
    w_odd = 300_001
    w_even = size - len(w) - 98  # even, not a multiple of 16
    assert w_even % 2 == 0 and w_even % 16 != 0 and w_odd % 2 == 1
    place(buf, w_odd, w)
    place(buf, w_even, w)
    # D at an odd offset, with a decoy (first 64 bytes) before it.
    d_off = 700_003
    place(buf, 600_000, d[:64])
    place(buf, d_off, d)
    # K: decoy (first half right, second half wrong at the real gap), then the real pair at gap 25.
    k_decoy = 900_000
    place(buf, k_decoy, k[:4])
    place(buf, k_decoy + 25, b"\x00\x00\x00\x00")
    k_off = 1_000_001
    place(buf, k_off, k[:4])
    place(buf, k_off + 25, k[4:])
    # A version resource for the log.
    place(buf, 2_000_000, version_resource(310, 99, 1, 0))

    case = Case("feature")
    case.file = bytes(buf)
    case.map = m
    case.sources = sources
    case.expected, case.key = finalize_map(m, sources)
    assert case.key == key
    case.placements = {"W": w_even, "D": d_off, "K": k_off}
    return case


# ---------------------------------------------------------------- a small case for the error tests

def mini_case(seed=3, version=None):
    rng = random.Random(seed)
    w = bytes(random_finite_halves(rng, 4096, max_exp_field=0x14))
    d = bytes(rng.randbytes(4096))
    k = struct.pack("<ff", 0.75, -1.5)
    key = sha256(b"mini label" + w)
    m = {
        "format": "wronggpu-converter-map",
        "version": 1,
        "model": "synthetic mini case",
        "channel_order": "natural",
        "sources": {
            "W": source_entry("W", "fake parameter block", "f16le", w),
            "D": source_entry("D", "fake dither table", "u16le", d),
            "K": source_entry("K", "fake constants", "u32le", k, locate="pair", gaps=[10], keyed=True, key=key),
        },
        "key": {"label": "mini label", "from": "W"},
        "perms": {"p4": [3, 1, 0, 2]},
        "stages": [
            {"stage": "enc0", "digest": "", "tensors": [
                tensor("W", [8, 12], "float32", "W", {"name": "fp8"}, 0, [lin(12), lin(1, "p4")]),
                tensor("b", [8], "float32", "W", {"name": "f16"}, 96, [lin(1)]),
                tensor("T", [8, 16, 16], "uint8", "D", {"name": "u16_shr", "shift": 9}, 0, [lin(256), lin(16), lin(1)]),
                tensor("sigma1", [1], "float32", "K", {"name": "u32_as_f32"}, 0, [lin(0)]),
                tensor("sigma2", [1], "float32", "K", {"name": "u32_as_f32"}, 1, [lin(0)]),
            ]},
            {"stage": "enc1", "digest": "", "tensors": [
                tensor("gamma1", [16], "float32", "W", {"name": "f16_scaled", "mul": SQRT96, "div": SQRT96}, 200, [lin(1)]),
                tensor("B", [2, 4, 4], "float32", "W", {"name": "f16_scaled", "mul": SQRT8, "div": "0x3f800000"}, 300, [lin(16), lin(4), lin(1)]),
            ]},
            {"stage": "dec0", "digest": "", "tensors": [
                tensor("S", [1], "float32", "W", {"name": "exp", "max": "0x41000000"}, 401, [lin(0)]),
            ]},
            {"stage": "post", "digest": "", "tensors": [
                tensor("s", [1], "float32", "W", {"name": "exp"}, 400, [lin(0)]),
            ]},
        ],
    }
    wb = bytearray(w)
    place(wb, 800, struct.pack("<H", half_bits(0.5)))
    place(wb, 802, struct.pack("<H", half_bits(2.5)))  # e^2.5 = 12.2 > 8: capped
    w = bytes(wb)
    m["sources"]["W"] = source_entry("W", "fake parameter block", "f16le", w)
    key = sha256(b"mini label" + w)
    m["sources"]["K"] = source_entry("K", "fake constants", "u32le", k, locate="pair", gaps=[10], keyed=True, key=key)
    sources = {"W": w, "D": d, "K": k}

    size = 64 * 1024
    buf = bytearray(rng.randbytes(size))
    w_off, d_off, k_off = 16 * 100, 16 * 1000 + 6, 30_001
    place(buf, w_off, w)
    place(buf, d_off, d)
    place(buf, k_off, k[:4])
    place(buf, k_off + 10, k[4:])
    if version:
        place(buf, 60_000, version_resource(*version))
    case = Case("mini")
    case.file = bytes(buf)
    case.map = m
    case.sources = sources
    case.expected, case.key = finalize_map(m, sources)
    case.placements = {"W": w_off, "D": d_off, "K": k_off}
    return case


# ---------------------------------------------------------------- two presets in one file (Model L, #89)

def _pair_map(rng, preset, label, w_halves, extra_stage):
    """One preset's synthetic map and sources: W (f16le), D (u16le), K (keyed pair), and for preset L one stage more
    (a stand-in for its extra blocks: an fp8 matrix, a bias, a scaled norm scale)."""
    w = bytes(random_finite_halves(rng, w_halves, max_exp_field=0x14))
    d = bytes(rng.randbytes(2048))
    k = struct.pack("<ff", rng.choice([0.5, 0.75, 1.25]), rng.choice([-1.5, -2.0, 3.0]))
    key = sha256(label.encode() + w)
    m = {
        "format": "wronggpu-converter-map",
        "version": 1,
        "model": "synthetic preset-%s case" % preset,
        "channel_order": "natural",
        "sources": {
            "W": source_entry("W", "fake preset-%s parameter block" % preset, "f16le", w),
            "D": source_entry("D", "fake dither table", "u16le", d),
            "K": source_entry("K", "fake constants", "u32le", k, locate="pair", gaps=[12], keyed=True, key=key),
        },
        "key": {"label": label, "from": "W"},
        "perms": {},
        "stages": [
            {"stage": "enc0", "digest": "", "tensors": [
                tensor("W", [8, 12], "float32", "W", {"name": "fp8"}, 0, [lin(12), lin(1)]),
                tensor("b", [8], "float32", "W", {"name": "f16"}, 96, [lin(1)]),
                tensor("T", [4, 16, 16], "uint8", "D", {"name": "u16_shr", "shift": 9}, 0, [lin(256), lin(16), lin(1)]),
                tensor("sigma1", [1], "float32", "K", {"name": "u32_as_f32"}, 0, [lin(0)]),
                tensor("sigma2", [1], "float32", "K", {"name": "u32_as_f32"}, 1, [lin(0)]),
            ]},
            {"stage": "enc1", "digest": "", "tensors": [
                tensor("gamma1", [16], "float32", "W", {"name": "f16_scaled", "mul": SQRT96, "div": SQRT96}, 200, [lin(1)]),
            ]},
        ],
    }
    if extra_stage:
        m["stages"].insert(0, {"stage": "inenc_extra", "digest": "", "tensors": [
            tensor("W1", [32, 20], "float32", "W", {"name": "fp8"}, 1200, [lin(20), lin(1)]),
            tensor("b1", [32], "float32", "W", {"name": "f16"}, 1840, [lin(1)]),
            tensor("gamma", [32], "float32", "W", {"name": "f16_scaled", "mul": SQRT8, "div": "0x3f800000"}, 1872, [lin(1)]),
        ]})
    return m, {"W": w, "D": d, "K": k}


def preset_pair_case(seed=31):
    """A synthetic library file with two presets' parameter blocks: preset L's W block directly before preset M's, each
    with its own D and K elsewhere, and maps for both (the L map names its preset; the M map has no "preset" field, as
    the shipped one). Returns (case_l, case_m, file_l_only, file_m_only); each case's file holds both blocks. Our own
    sizes and bytes, nothing taken from a real file."""
    rng = random.Random(seed)
    ml, sl = _pair_map(rng, "L", "pair label L", 2304, True)
    ml["preset"] = "L"
    mm, sm = _pair_map(rng, "M", "pair label M", 2048, False)

    def layout(with_l, with_m):
        buf = bytearray(rng_file.randbytes(96 * 1024))
        at = {}
        if with_l:
            at["L"] = {"W": 1600, "D": 40_000, "K": 60_001}
        if with_m:
            at["M"] = {"W": 1600 + len(sl["W"]), "D": 50_000 + 6, "K": 70_003}  # M's W directly after L's
        for p, src in (("L", sl), ("M", sm)):
            if p in at:
                place(buf, at[p]["W"], src["W"])
                place(buf, at[p]["D"], src["D"])
                place(buf, at[p]["K"], src["K"][:4])
                place(buf, at[p]["K"] + 12, src["K"][4:])
        return bytes(buf), at

    rng_file = random.Random(seed + 1)
    both, at = layout(True, True)
    l_only, _ = layout(True, False)
    m_only, _ = layout(False, True)
    out = []
    for name, m, s in (("pair-L", ml, sl), ("pair-M", mm, sm)):
        c = Case(name)
        c.file = both
        c.map = m
        c.sources = s
        c.expected, c.key = finalize_map(m, s)
        c.placements = at[name[-1]]
        out.append(c)
    return out[0], out[1], l_only, m_only


# ---------------------------------------------------------------- source F (spec 08 sections 4.3, 4.4; map version 4)

def f_elements(rng, n=5):
    """n finite binary32 bit patterns of our own (random numbers, not anyone's weights)."""
    return [struct.unpack("<I", struct.pack("<f", rng.uniform(-1.0, 1.0)))[0] for _ in range(n)]


def f_mini_case(present=0, f_offset=70_000, other_offset=None, decoy_offset=None, at_end=False, elements0=None,
                seed=21):
    """mini_case as a version-4 map: plus the optional source F (two variants whose blocks start with the same 64
    bytes, as in the shipped map) and the output group `filter` with `output_taps` [5].

    present: the variant whose block is in the file (None: no block), at f_offset (or at the very end of the file).
    other_offset: the other variant's block is placed there too. decoy_offset: the first 64 bytes of a block there
    (FNV-1a hit, SHA-256 miss). elements0: the five elements of variant 1 instead of random ones.
    The map is the same for every choice of present and of the offsets (same seed)."""
    base = mini_case()
    rng = random.Random(seed)
    m = copy.deepcopy(base.map)
    m["version"] = 4
    m["model"] = "synthetic mini case with source F"
    head = filler_text(random.Random(seed + 1000), 700)
    e0, e1 = f_elements(rng), f_elements(rng)
    variants = [make_f_variant(rng, elements0 or e0, 6000, head, upper=True),
                make_f_variant(rng, e1, 5200, head, upper=None)]
    m["sources"]["F"] = f_source_entry("fake filter text block", variants, base.key)
    fv = m["sources"]["F"]["variants"]
    assert fv[0]["fnv1a64"] == fv[1]["fnv1a64"] and fv[0]["sha256"] != fv[1]["sha256"]
    m["stages"].append({"stage": "filter", "digest": "", "tensors": [
        tensor("output_taps", [5], "float32", "F", {"name": "u32_as_f32"}, 0, [lin(1)])]})
    buf = bytearray(base.file) + rng.randbytes(64 * 1024)
    sources = dict(base.sources)
    case = Case("f-mini")
    case.placements = dict(base.placements)
    case.variants = variants
    if decoy_offset is not None:
        place(buf, decoy_offset, variants[0]["block"][:64])
    if present is not None:
        block = variants[present]["block"]
        if at_end:
            f_offset = len(buf) - len(block)
        place(buf, f_offset, block)
        if other_offset is not None:
            place(buf, other_offset, variants[1 - present]["block"])
        sources["F"] = block
        case.placements["F"] = f_offset
        case.found = {"filter": present}
    case.file = bytes(buf)
    case.map = m
    case.sources = sources
    case.expected, case.key = finalize_map(m, sources, {"F": [v["block"] for v in variants]})
    assert case.key == base.key
    return case


# ---------------------------------------------------------------- preset-K cases (spec 48, map versions 2 and 3)

QSCALE = "0x3e8293ee"   # RN32(RN32(1/RN32(ln 2)) * RN32(1/RN32(sqrt 32)))
PSCALE = "0x3fb8aa3b"   # RN32(1/RN32(ln 2))
CS_OP = {"name": "capped_softmax", "cap": "0x41700000", "div": "0x4428c000", "sub": "0x40600000"}


def k_map(label, t_entry, o_entry, stages, version=3, model="synthetic preset-K case"):
    srcs = {"T": t_entry}
    if o_entry is not None:
        srcs["O"] = o_entry
    return {"format": "wronggpu-converter-map", "version": version, "model": model, "channel_order": "natural",
            "sources": srcs, "key": {"label": label, "from": "T"}, "perms": {}, "stages": stages}


def k_feature_case(t5_path, seed=11):
    """Every feature of map version 3: Base64 segments listed in an order different from their order in T, each
    with its own padding (none, one or two '='), junk with '=' between them; f16 of every finite binary16 pattern,
    f16_scaled with the two preset-K scales; negative strides; capped_softmax on the 48-T5 vectors, on extreme inputs
    (all magnitudes, +-0, subnormals, values beyond the cap) and on a [2][8][8][8][8] table read with negative
    strides; i32 from O (negative values, both extremes) forwards and backwards. T at an even offset that is not a
    multiple of 16 after a decoy, O at an odd offset after a decoy."""
    import json
    rng = random.Random(seed)
    with open(t5_path) as f:
        t5 = json.load(f)
    # W: all 65536 binary16 patterns, the 48-T5 x vectors (512), then random finite values.
    w = bytearray(struct.pack("<65536H", *range(65536)))
    V = len(w) // 2
    w += struct.pack("<512H", *[int(x, 16) for row in t5["x"] for x in row])
    R = len(w) // 2
    w += random_finite_halves(rng, 14000, max_exp_field=0x16)  # |x| < 256: no overflow in f16_scaled
    w = bytes(w)
    # Chunks with byte lengths of every residue mod 3 (so all three padding forms occur); the total stays even.
    cuts = [0, 1000, 1000 + 3001, 1000 + 3001 + 30002, 100003, 100003 + 7, len(w)]
    order = [2, 0, 1, 5, 3, 4]
    o = struct.pack("<16i", 3, -1, 0, 7, -2147483648, 2147483647, 5, 2, 1, 6, 4, 0, -7, 1, 2, 3)

    n_pos = 0x7C00
    plain = [
        tensor("f16_pos", [n_pos], "float32", "W", {"name": "f16"}, 0, [lin(1)]),
        tensor("f16_neg", [n_pos], "float32", "W", {"name": "f16"}, 0x8000, [lin(1)]),
        tensor("Wq", [4, 32, 96], "float32", "W", {"name": "f16_scaled", "mul": QSCALE, "div": "0x3f800000"},
               R, [lin(32 * 96), lin(96), lin(1)]),
        tensor("B", [2, 8, 8, 8, 8], "float32", "W", {"name": "f16_scaled", "mul": PSCALE, "div": "0x3f800000"},
               R + 12287, [lin(-4096), lin(-8), lin(-1), lin(-512), lin(-64)]),
        tensor("g1", [96], "float32", "W", {"name": "f16_scaled", "mul": SQRT96, "div": "0x3f800000"}, R + 5, [lin(2)]),
    ]
    # Inputs for the extreme capped_softmax groups: 64 patterns spread over the whole finite range.
    ext = R + 12400  # outside every other tensor (the f16_scaled ones would overflow)
    wb = bytearray(w)
    # +-0, +-smallest subnormal, largest subnormal, +-65504, +-cap (15), just above +cap, just inside -cap.
    extreme = [half_bits(v) for v in (0.0, -0.0, 2.0 ** -24, -2.0 ** -24, 1023 * 2.0 ** -24, 65504.0, -65504.0,
                                      15.0, -15.0, 15.0 + 2.0 ** -7, -(15.0 - 2.0 ** -7))]
    extreme += [rng.randrange(0, 0x7C00) | (rng.getrandbits(1) << 15) for _ in range(64 * 3 - len(extreme))]
    place(wb, 2 * ext, struct.pack("<%dH" % len(extreme), *extreme))
    w = bytes(wb)
    chunks = [w[a:b] for a, b in zip(cuts, cuts[1:])]
    assert {len(c) % 3 for c in chunks} == {0, 1, 2} and len(w) % 2 == 0
    text, segs = synth_build_t(rng, chunks, order, lead=3, tail=9)
    key = sha256(b"k feature label" + text)
    cs = [
        tensor("t5", [8, 8, 8], "float32", "W", CS_OP, V, [lin(64), lin(8), lin(1)]),
        tensor("extreme", [3, 8, 8], "float32", "W", CS_OP, ext, [lin(64), lin(8), lin(1)]),
        tensor("M", [2, 8, 8, 8, 8], "float32", "W", CS_OP, R + 4100, [lin(4096), lin(64), lin(512), lin(-8), lin(-1)]),
        tensor("M_t", [1, 8, 8], "float32", "W", CS_OP, R + 100, [lin(0), lin(1), lin(8)]),  # transposed group
    ]
    net = [
        tensor("window_offsets", [8, 2], "int32", "O", {"name": "i32"}, 0, [lin(2), lin(1)]),
        tensor("o_rev", [16], "int32", "O", {"name": "i32"}, 15, [lin(-1)]),
    ]
    m = k_map("k feature label", t_source_entry_("fake text block", text, segs),
              source_entry("O", "fake offset table", "i32le", o, keyed=True, key=key),
              [{"stage": "plain", "digest": "", "tensors": plain},
               {"stage": "cs", "digest": "", "tensors": cs},
               {"stage": "net", "digest": "", "tensors": net}])
    m["an_unknown_top_level_field"] = [1, 2]

    size = 1_200_000
    buf = bytearray(rng.randbytes(size))
    place(buf, 4096, text[:64])                       # FNV hit, SHA miss
    t_off = 200_006                                   # even, not a multiple of 16
    place(buf, t_off, text)
    o_decoy = bytearray(o)
    o_decoy[63] ^= 1
    place(buf, 900_000, bytes(o_decoy))               # FNV miss (all 64 bytes hashed)
    o_off = 1_000_003
    place(buf, o_off, o)
    place(buf, 1_100_000, version_resource(310, 98, 2, 0))
    case = Case("k-feature")
    case.file = bytes(buf)
    case.map = m
    case.sources = {"T": text, "O": o}
    case.expected, case.key = finalize_map(m, case.sources)
    case.placements = {"T": t_off, "O": o_off}
    case.t5 = t5
    return case


def k_mini_case(seed=5, version=3, file_version=None):
    """A small version-3 (or 2: no O, no net) map for the error tests: three segments, a capped_softmax table, the
    window-offset table."""
    rng = random.Random(seed)
    w = bytes(random_finite_halves(rng, 2000, max_exp_field=0x14))
    chunks = [w[:1001], w[1001:2500], w[2500:]]
    text, segs = synth_build_t(rng, chunks, [0, 1, 2], lead=5, tail=7)
    o = struct.pack("<16i", *[rng.randrange(0, 8) for _ in range(16)])
    key = sha256(b"k mini label" + text)
    stages = [
        {"stage": "e1", "digest": "", "tensors": [
            tensor("g1", [64], "float32", "W", {"name": "f16_scaled", "mul": "0x41000000", "div": "0x3f800000"}, 10, [lin(1)]),
            tensor("M", [2, 8, 8], "float32", "W", CS_OP, 1999, [lin(-64), lin(-8), lin(-1)]),
        ]},
        {"stage": "e2", "digest": "", "tensors": [
            tensor("B", [1, 8, 8], "float32", "W", {"name": "f16_scaled", "mul": PSCALE, "div": "0x3f800000"}, 500, [lin(0), lin(8), lin(1)]),
        ]},
    ]
    if version >= 3:
        stages.append({"stage": "net", "digest": "", "tensors": [
            tensor("window_offsets", [8, 2], "int32", "O", {"name": "i32"}, 0, [lin(2), lin(1)])]})
    m = k_map("k mini label", t_source_entry_("fake text block", text, segs),
              source_entry("O", "fake offset table", "i32le", o, keyed=True, key=key) if version >= 3 else None,
              stages, version=version)
    size = 64 * 1024
    buf = bytearray(rng.randbytes(size))
    t_off, o_off = 16 * 64, 40_000 + 16 * 3
    place(buf, t_off, text)
    if version >= 3:
        place(buf, o_off, o)
    if file_version:
        place(buf, 60_000, version_resource(*file_version))
    case = Case("k-mini")
    case.file = bytes(buf)
    case.map = m
    case.sources = {"T": text, "O": o} if version >= 3 else {"T": text}
    case.expected, case.key = finalize_map(m, case.sources)
    case.placements = {"T": t_off, "O": o_off} if version >= 3 else {"T": t_off}
    return case


def flip_bit(data, offset, bit=0):
    b = bytearray(data)
    b[offset] ^= 1 << bit
    return bytes(b)
