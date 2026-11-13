# SPDX-License-Identifier: MIT
"""Independent reference for spec 57 (map version 5, preset F) and builder of synthetic version-5 cases.

Written for this project. Python standard library only, independent of the C++ converter: every decode operation
uses exact integer (rational) arithmetic from synth.py (rn16, rn32, q8, a 50-digit decimal exp), the square root is
an exact integer square root, and the hashes use hashlib and a direct FNV-1a loop.

A synthetic case is a fake file (random bytes with our own source blocks placed at chosen offsets) plus a version-5
map whose hashes and digests are computed here from those blocks. No model data is involved anywhere.
"""

import copy
import json
import math
import random
import struct

from synth import (NonFinite, f32_value, fnv1a64, half_value, keyed_fnv_state, keyed_sha256, q8, rn16, rn32,
                   round_nd, sha256, exp_rn32 as exp_rn32_rat)


class Rejected(Exception):
    """A value check of spec 57 failed ("not supported yet", exit code 14)."""


class Damaged(Exception):
    """The map does not fit (a defensive bounds failure; "converter data damaged")."""


DTYPE_SIZE = {"e4m3": 1, "float16": 2, "uint16": 2, "float32": 4, "uint32": 4}
DTYPE_CODE = {"float32": 0, "uint8": 1, "int32": 2, "float16": 3, "uint16": 4, "uint32": 5, "e4m3": 6}
ELEMENT_SIZE = {"f16le": 2, "u16le": 2, "u32le": 4}
KEY_LABEL = "synthetic preset-F test key"


# ---------------------------------------------------------------- exact values

def is_half_finite(h):
    return (h >> 10) & 0x1F != 0x1F


def rat_lt(a, b):
    return a[0] * b[1] < b[0] * a[1]


def rat_le(a, b):
    return a[0] * b[1] <= b[0] * a[1]


def half_bits_of(r, negative_zero=False):
    """binary16 bit pattern of an exactly representable rational (or 'inf' for overflow, handled by caller)."""
    num, den = r
    if num == 0:
        return 0x8000 if negative_zero else 0
    sign = 0x8000 if num < 0 else 0
    num = abs(num)
    # value = num/den with den a power of 2
    for e in range(-14, 16):
        unit = (1, 1 << (10 - e)) if 10 - e >= 0 else (1 << (e - 10), 1)
        # mantissa k = value / 2^(e-10)
        k_num = num * unit[1]
        k_den = den * unit[0]
        if k_num % k_den:
            continue
        k = k_num // k_den
        if e == -14 and k < 1024:
            return sign | k
        if 1024 <= k < 2048:
            return sign | ((e + 15) << 10) | (k - 1024)
    raise ValueError("not a binary16 value")


def f32_bits_of(r, negative_zero=False):
    if r[0] == 0:
        return 0x80000000 if negative_zero else 0
    return struct.unpack("<I", struct.pack("<f", r[0] / r[1]))[0]  # exact value: int/int is correctly rounded


def rn16_checked(r):
    return rn16(r)  # raises NonFinite on overflow


def e4m3_code(h):
    v = half_value(h)
    if isinstance(v, str):
        raise Rejected("e4m3 of a non-finite value")
    sign = 128 if h & 0x8000 else 0
    q = q8(v)
    a = (abs(q[0]), q[1])
    if a[0] == 0:
        return sign
    e = a[0].bit_length() - a[1].bit_length()
    if (a[0] << max(0, -e)) < (a[1] << max(0, e)):
        e -= 1
    if e < -6:
        m = a[0] * (1 << 9) // a[1]
        assert m * a[1] == a[0] * (1 << 9)
        return sign | m
    # a = (1 + m/8) 2^e  ->  m = a * 2^(3-e) - 8
    num = a[0] * (1 << (3 - e)) if 3 - e >= 0 else a[0]
    den = a[1] if 3 - e >= 0 else a[1] * (1 << (e - 3))
    assert num % den == 0
    return sign | ((e + 7) << 3) | (num // den - 8)


def sqrt_rn32(x):
    """Correctly rounded binary32 square root of a non-negative rational x = (num, den), den a power of two."""
    num, den = x
    if num == 0:
        return (0, 1)
    if num < 0:
        raise NonFinite("sqrt of a negative value")
    # sqrt(num/den) = sqrt(num * den * 4^K) / (den * 2^K) with K large; floor plus a sticky bit.
    K = 200
    N = num * den * (1 << (2 * K))
    s = math.isqrt(N)
    sticky = s * s != N
    d = den * (1 << K)
    if sticky:  # any value strictly between s and s + 1 rounds the same (boundaries are far coarser)
        return rn32((2 * s + 1, 2 * d))
    return rn32((s, d))


def exp32(h):
    """RN32(e^value(h)) as a rational, or 'inf' on overflow."""
    v = half_value(h)
    try:
        return exp_rn32_rat(v)
    except NonFinite:
        return "inf"


def next_f32_up(bits):
    """The binary32 bit pattern one ulp above a finite non-negative value."""
    return bits + 1


# ---------------------------------------------------------------- the operations of spec 57 section 6

def op_score_bias(S, H, a_bits, wd_bits, pairs):
    """S: list of H*4096 binary16 patterns. Returns the list of H*12*256 binary16 patterns of Tb."""
    for h in S:
        if not is_half_finite(h):
            raise Rejected("score_bias: non-finite stored value")
    try:
        A = f32_value(a_bits)
        Wd = f32_value(wd_bits)
        f2 = rn32((sqrt_rn32(Wd)[0] * A[0], sqrt_rn32(Wd)[1] * A[1]))
    except NonFinite:
        raise Rejected("score_bias: non-finite intermediate")
    T = [0] * (H * 4096)
    for hh in range(H):
        base = hh * 4096
        for a in range(64):
            for b in range(64):
                x = half_value(S[base + a * 64 + b])
                y = half_value(S[base + b * 64 + a])
                try:
                    s1 = rn32((x[0] * y[1] + y[0] * x[1], x[1] * y[1]))
                    s2 = rn32((s1[0] * A[0], s1[1] * A[1]))
                    s3 = rn32((s2[0] * f2[0], s2[1] * f2[1]))
                    t = rn16(s3)
                except NonFinite:
                    raise Rejected("score_bias: non-finite value")
                # IEEE sign rules: x + y is -0 only when both are -0; a product's sign is the XOR of its operands'
                # signs (also when it rounds to zero); sqrt(-0) = -0. So sign(T) = sign(x + y) XOR [Wd is -0].
                s1neg = s1[0] < 0 if s1[0] != 0 else bool(S[base + a * 64 + b] & S[base + b * 64 + a] & 0x8000)
                neg0 = s1neg != (wd_bits == 0x80000000)
                T[base + a * 64 + b] = half_bits_of(t, negative_zero=neg0)
    out = []
    for hh in range(H):
        for (I, J) in pairs:
            for r in range(16):
                for s in range(16):
                    out.append(T[hh * 4096 + (16 * I + r) * 64 + 16 * J + s])
    return out


def op_const(name, negate, words):
    """Returns the output bit pattern (binary32 for const_u32_as_f32, binary16 otherwise)."""
    if not words:
        raise Damaged("no site element")
    w = words[0]
    if any(x != w for x in words):
        raise Rejected("sites disagree")
    if name == "const_u32_as_f32":
        return w
    if name == "const_f16x2":
        if w >> 16 != w & 0xFFFF:
            raise Rejected("halves differ")
        return w & 0xFFFF
    if name == "const_f16lo":
        if w >> 16:
            raise Rejected("high half not zero")
        return w & 0xFFFF
    if name == "const_f32_to_f16":
        try:
            v = f32_value(w)
        except NonFinite:
            raise Rejected("non-finite")
        neg_zero = bool(w >> 31) != bool(negate)
        if negate:
            v = (-v[0], v[1])
        try:
            y = rn16(v)
        except NonFinite:
            raise Rejected("RN16 overflow")
        neg0 = neg_zero if y[0] == 0 else False
        return half_bits_of(y, negative_zero=neg0 if y[0] == 0 else False)
    raise ValueError(name)


def op_attn_wmax(lo, hi, p1, p0):
    """binary16 patterns in; (WMAX pattern, may_overflow) out, or Rejected."""
    for h in (lo, hi, p1, p0):
        if not is_half_finite(h):
            raise Rejected("attention constant not finite")
    vlo, vhi, vp1, vp0 = (half_value(x) for x in (lo, hi, p1, p0))
    if rat_lt(vhi, vlo):
        raise Rejected("bounds reversed")
    best = None
    for c16 in range(65536):
        if not is_half_finite(c16):
            continue
        c = half_value(c16)
        if rat_lt(c, vlo) or rat_lt(vhi, c):
            continue
        cc = (c[0] * c[0], c[1] * c[1])
        try:
            inner = rn16((vp1[0] * cc[1] - cc[0] * vp1[1], vp1[1] * cc[1]))
            prod = (c[0] * inner[0], c[1] * inner[1])
            p = rn16((prod[0] * vp0[1] + vp0[0] * prod[1], prod[1] * vp0[1]))
        except NonFinite:
            raise Rejected("attention polynomial overflows")
        y = (32 * (p[0] - p[1]), p[1])  # 32 (p - 1)
        if rat_lt(y, (1, 1)) or not rat_lt(y, (31, 1)):
            raise Rejected("attention polynomial out of range")
        n = y[0] // y[1]
        frac = (y[0] - n * y[1], y[1])
        # w = 2^(n-15) (1 + frac)
        w = ((frac[1] + frac[0]) * (1 << n), frac[1] * (1 << 15)) if n >= 0 else None
        if best is None or rat_lt(best, w):
            best = w
    return half_bits_of(best), best[0] * 64 >= 65520 * best[1]


def value_of(dtype, data):
    """Exact value of element 0: a rational (num, den) or 'nan'/'inf'/'-inf'."""
    if dtype == "float16":
        return half_value(struct.unpack_from("<H", data)[0])
    if dtype == "float32":
        bits = struct.unpack_from("<I", data)[0]
        try:
            return f32_value(bits)
        except NonFinite:
            return "nan" if bits & 0x7FFFFF else ("-inf" if bits >> 31 else "inf")
    if dtype == "e4m3":
        c = data[0]
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            return "nan"
        v = (m, 512) if e == 0 else ((8 + m) * (1 << max(0, e - 10)), 1 << max(0, 10 - e))
        return (-v[0], v[1]) if s else v
    if dtype == "uint16":
        return (struct.unpack_from("<H", data)[0], 1)
    return (struct.unpack_from("<I", data)[0], 1)


def element_values(dtype, data):
    sz = DTYPE_SIZE[dtype]
    return [value_of(dtype, data[i:i + sz]) for i in range(0, len(data), sz)]


MIN_NORMAL = {"float16": (1, 1 << 14), "float32": (1, 1 << 126), "e4m3": (1, 64)}


def require_ok(dtype, data, conds):
    for v in element_values(dtype, data):
        for c in conds:
            if c == "finite":
                ok = not isinstance(v, str)
            elif c == "not_subnormal":
                ok = not isinstance(v, str) and (v[0] == 0 or dtype not in MIN_NORMAL or
                                                 not rat_lt((abs(v[0]), v[1]), MIN_NORMAL[dtype]))
            elif c == "even":
                ok = not isinstance(v, str) and v[0] % v[1] == 0 and (v[0] // v[1]) % 2 == 0
            elif c == "below_8":
                ok = not isinstance(v, str) and rat_lt(v, (8, 1)) or v == "-inf"
            else:
                raise ValueError(c)
            if not ok:
                return False
    return True


def finite_output(dtype, data):
    if dtype not in ("float16", "float32"):
        return True
    return all(not isinstance(v, str) for v in element_values(dtype, data))


def check_ok(rel, a, b):
    if isinstance(a, str) or isinstance(b, str):
        return False
    if rel == "ge":
        return rat_le(b, a)
    if rel == "gt":
        return rat_lt(b, a)
    if rel == "le":
        return rat_le(a, b)
    if rel == "ratio_below_f16_overflow":
        if b[0] <= 0:
            return False
        try:
            r = rn32((b[1], b[0]))
        except NonFinite:
            return False
        rbits = f32_bits_of(r)
        rplus = struct.unpack("<f", struct.pack("<I", rbits + 1))[0]
        if math.isinf(rplus):
            return False
        rp = rplus.as_integer_ratio()
        try:
            prod = rn32((a[0] * rp[0], a[1] * rp[1]))
        except NonFinite:
            return False
        return rat_lt((abs(prod[0]), prod[1]), (65520, 1))
    raise ValueError(rel)


def const_value(c):
    """A JSON check constant (int or float) as an exact rational."""
    if isinstance(c, int):
        return (c, 1)
    n, d = float(c).as_integer_ratio()
    return (n, d)


# ---------------------------------------------------------------- whole-map decoding

def source_elements(src, data):
    """The element list of one found source (57 section 4): a window has one element, the value at its field."""
    size = ELEMENT_SIZE[src["element"]]
    fmt = "<H" if size == 2 else "<I"
    if src["locate"] == "window":
        return [struct.unpack_from(fmt, data, src["field"])[0]]
    return [struct.unpack_from(fmt, data, i)[0] for i in range(0, len(data), size)]


def axis_values(dim, axis, perms):
    vals = []
    for i in range(dim):
        p = i
        if "perm" in axis:
            q = perms[axis["perm"]]
            b = len(q)
            p = b * (i // b) + q[i % b]
        vals.append(None if ("limit" in axis and p >= axis["limit"]) else axis["stride"] * p)
    return vals


def index_list(t, perms):
    """Per output element (row-major): the offset sum, or None for an element that is zero (limit)."""
    out = [0]
    for dim, axis in zip(t["shape"], t["axes"]):
        vals = axis_values(dim, axis, perms)
        out = [None if (o is None or v is None) else o + v for o in out for v in vals]
    return out


def decode_group(m, g, elems, variants):
    """Returns [(name, bytes)] for one group; raises Rejected or Damaged."""
    out = []
    by_name = {}
    perms = m.get("perms", {})
    for t in g["tensors"]:
        op = t["op"]
        name = op["name"]
        dt = t["dtype"]
        if name == "score_bias":
            H = t["shape"][0]
            S = elems["W"][t["base"]: t["base"] + H * 4096]
            if len(S) != H * 4096:
                raise Damaged("score_bias outside W")
            vals = op_score_bias(S, H, elems[op["a_src"]][0], elems[op["w_src"]][0], op["pairs"])
            data = b"".join(struct.pack("<H", v) for v in vals)
        elif name.startswith("const_"):
            words = []
            for site in t["sites"]:
                el = elems[site["src"]]
                lst = site["elements"] if "elements" in site else site["elements_by_variant"][variants[site["src"]]]
                for e in lst:
                    if e >= len(el):
                        raise Damaged("site outside")
                    words.append(el[e])
            bits = op_const(name, op.get("negate", False), words)
            data = struct.pack("<I" if name == "const_u32_as_f32" else "<H", bits)
        elif name == "attn_wmax":
            ins = [struct.unpack("<H", by_name[op[k]])[0] for k in ("clamp_lo", "clamp_hi", "p1", "p0")]
            w, _ = op_attn_wmax(*ins)
            data = struct.pack("<H", w)
        else:
            idx = index_list(t, perms)
            src = elems[t["src"]]
            alsos = [(elems[a["src"]], a["base"]) for a in t.get("also", [])]
            cap = None
            if name == "exp" and "max_src" in op:
                try:
                    cap = f32_value(elems[op["max_src"]][0])
                except NonFinite:
                    raise Rejected("cap not finite")
            parts = []
            sz = DTYPE_SIZE[dt]
            for o in idx:
                if o is None:
                    parts.append(b"\0" * sz)
                    continue
                e = t["base"] + o
                if e >= len(src):
                    raise Damaged("index outside")
                x = src[e]
                for (asrc, abase) in alsos:
                    if abase + o >= len(asrc):
                        raise Damaged("also outside")
                    if asrc[abase + o] != x:
                        raise Rejected("also differs")
                if name == "e4m3":
                    parts.append(bytes([e4m3_code(x)]))
                elif name == "f16":
                    if not is_half_finite(x):
                        raise Rejected("f16 non-finite")
                    parts.append(struct.pack("<H", x))
                elif name == "u16":
                    parts.append(struct.pack("<H", x))
                elif name in ("u32", "u32_as_f32"):
                    parts.append(struct.pack("<I", x))
                elif name == "exp":
                    if not is_half_finite(x):
                        raise Rejected("exp of non-finite")
                    y = exp32(x)
                    if cap is not None and (y == "inf" or rat_lt(cap, y)):
                        y = cap
                    if y == "inf":
                        raise Rejected("exp overflow")
                    parts.append(struct.pack("<I", f32_bits_of(y)))
                else:
                    raise ValueError(name)
            data = b"".join(parts)
        if not t.get("allow_nonfinite", False) and not finite_output(dt, data):
            raise Rejected("non-finite output " + t["name"])
        if not require_ok(dt, data, t.get("require", [])):
            raise Rejected("require fails " + t["name"])
        by_name[t["name"]] = data
        out.append((t["name"], data))
    for c in g.get("checks", []):
        a = value_of(next(x["dtype"] for x in g["tensors"] if x["name"] == c["a"]), by_name[c["a"]])
        if isinstance(c["b"], dict):
            b = const_value(c["b"]["const"])
        else:
            b = value_of(next(x["dtype"] for x in g["tensors"] if x["name"] == c["b"]), by_name[c["b"]])
        if not check_ok(c["rel"], a, b):
            raise Rejected("check fails")
    return out


def group_enabled(g, variants):
    return all(variants.get(s) == v for s, v in g.get("requires_variant", {}).items())


def decode_all(m, found, variants):
    """found: {source name: bytes as found}; variants: {source: variant index}. Returns (outputs, key).
    outputs: {group: [(tensor, bytes)]} for the enabled groups."""
    key = sha256(m["key"]["label"].encode() + found["W"])
    elems = {n: source_elements(s, found[n]) for n, s in m["sources"].items()}
    outs = {}
    for g in m["groups"]:
        if group_enabled(g, variants):
            outs[g["group"]] = decode_group(m, g, elems, variants)
    return outs, key


def group_digest(key, tensors):
    return keyed_sha256(key, b"".join(d for _, d in tensors)).hex()


# ---------------------------------------------------------------- building cases

def hashes(data, key=None):
    if key is None:
        return "%016x" % fnv1a64(data[:64]), sha256(data).hex()
    return "%016x" % fnv1a64(data[:64], keyed_fnv_state(key)), keyed_sha256(key, data).hex()


def set_source_hashes(m, blocks, key):
    """blocks: {source: bytes} or, for a source with variants, {source: [bytes per variant]}."""
    for n, s in m["sources"].items():
        if "variants" in s:
            for v, b in zip(s["variants"], blocks[n]):
                v["length"] = len(b)
                v["fnv1a64"], v["sha256"] = hashes(b, key)
        else:
            b = blocks[n]
            s["length"] = len(b)
            s["fnv1a64"], s["sha256"] = hashes(b, None if n == "W" else key)


def finalize(m, blocks):
    """Sets every source hash and every group digest. A group gated on a variant gets the digest it has for that
    variant; every other group gets its digest with variant 0 of every source (all variants must agree on them)."""
    key = sha256(m["key"]["label"].encode() + blocks["W"])
    set_source_hashes(m, blocks, key)
    variant_sources = [n for n, s in m["sources"].items() if "variants" in s]
    expected = {}
    for g in m["groups"]:
        rv = g.get("requires_variant", {})
        variants = {n: rv.get(n, 0) for n in variant_sources}
        found = {n: (blocks[n][variants[n]] if n in variants else blocks[n]) for n in m["sources"]}
        elems = {n: source_elements(s, found[n]) for n, s in m["sources"].items()}
        try:
            tensors = decode_group(m, g, elems, variants)
            g["digest"] = group_digest(key, tensors)
        except (Rejected, Damaged):
            tensors = None
            g["digest"] = "00" * 32  # never compared: the conversion stops before
        expected[g["group"]] = tensors
    return key, expected


class Case5:
    def __init__(self, name):
        self.name = name
        self.map = None
        self.blocks = None
        self.file = None
        self.key = None
        self.placements = {}   # source -> offset (variant source: offset of the placed variant)
        self.variant = {}      # variant source -> index placed in the file
        self.expected = None   # {group: [(tensor, bytes)]} for this file (enabled groups only)


def window(rng, field, value_bytes, length=16):
    b = bytearray(rng.randbytes(length))
    b[field: field + len(value_bytes)] = value_bytes
    return bytes(b)


def build_file(rng, blocks, m, layout, size, variant_choice=None, decoys=(), extra=()):
    """layout: list of (source, offset); a variant source places blocks[source][variant_choice[source]].
    decoys: list of (offset, bytes) placed first (damaged copies, etc.); extra: (offset, bytes) placed last."""
    buf = bytearray(rng.randbytes(size))
    for off, data in decoys:
        buf[off: off + len(data)] = data
    for n, off in layout:
        b = blocks[n]
        if isinstance(b, list):
            b = b[variant_choice[n]]
        buf[off: off + len(b)] = b
    for off, data in extra:
        buf[off: off + len(data)] = data
    return bytes(buf)


def h16(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


def f32b(v):
    return struct.pack("<f", v)


def feature_case_map():
    """Our own version-5 map that uses every feature of spec 57: windows (u16 and u32), blocks, code images with and
    without variants, every op, perm, limit, also, sites with elements and elements_by_variant, require, checks,
    requires_variant. Hashes and digests are filled in by finalize()."""
    nu = [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
          30, 31]
    win = lambda what, el: {"what": what, "element": el, "field": 8, "length": 16, "locate": "window", "keyed": True}
    m = {
        "format": "wronggpu-converter-map", "version": 5, "model": "synthetic preset-F feature case",
        "channel_order": "natural",
        "sources": {
            "W": {"what": "synthetic parameter block", "element": "f16le", "locate": "block"},
            "D": {"what": "synthetic u16 table", "element": "u16le", "locate": "block", "keyed": True},
            "SH": {"what": "synthetic shift table", "element": "u32le", "locate": "block", "keyed": True},
            "BS_A": win("word A", "u32le"),
            "BS_W": win("word Wd", "u32le"),
            "CAP": win("cap word", "u32le"),
            "H.x": win("half word", "u16le"),
            "IMG-A": {"what": "synthetic image", "element": "u32le", "encoding": "none", "locate": "block",
                      "keyed": True},
            "IMG-V": {"what": "synthetic image with variants", "element": "u32le", "encoding": "none",
                      "locate": "block", "keyed": True, "variants": [{}, {}]},
            "IMG-B": {"what": "second copy region", "element": "u32le", "encoding": "none", "locate": "block",
                      "keyed": True},
        },
        "key": {"label": KEY_LABEL, "from": "W"},
        "perms": {"nu": nu, "rev4": [3, 2, 1, 0]},
        "groups": [],
    }
    lin = lambda s: {"stride": s}
    g1 = {"group": "G1", "tensors": [
        # e4m3 of every finite binary16 pattern (W elements 0..63487), sign of zero included.
        {"name": "q", "shape": [62, 1024], "dtype": "e4m3", "op": {"name": "e4m3"}, "src": "W", "base": 0,
         "axes": [lin(1024), lin(1)]},
        # e4m3 with the input-order permutation and a limit (rows nu(k) >= 43 are zero).
        {"name": "fe", "shape": [64, 8], "dtype": "e4m3", "op": {"name": "e4m3"}, "src": "W", "base": 70000,
         "axes": [{"stride": 1, "perm": "nu", "limit": 43}, lin(43)]},
        {"name": "b", "shape": [4, 5], "dtype": "float16", "op": {"name": "f16"}, "src": "W", "base": 71000,
         "axes": [{"stride": 5, "perm": "rev4"}, lin(1)]},
        {"name": "dz", "shape": [3, 4], "dtype": "float16", "op": {"name": "f16"}, "src": "W", "base": 71100,
         "axes": [{"stride": 4, "limit": 0}, lin(1)]},
        {"name": "D", "shape": [256], "dtype": "uint16", "op": {"name": "u16"}, "src": "D", "base": 0,
         "axes": [lin(1)]},
        {"name": "hx", "shape": [1], "dtype": "float16", "op": {"name": "f16"}, "src": "H.x", "base": 0,
         "axes": [lin(1)]},
        {"name": "e0", "shape": [1], "dtype": "float32", "op": {"name": "exp", "max_src": "CAP"}, "src": "W",
         "base": 71200, "axes": [lin(1)]},
        {"name": "e1", "shape": [40], "dtype": "float32", "op": {"name": "exp"}, "src": "W", "base": 71300,
         "axes": [lin(1)]},
        {"name": "e2", "shape": [3], "dtype": "float32", "op": {"name": "exp", "max_src": "CAP"}, "src": "W",
         "base": 71400, "axes": [lin(1)]},
        {"name": "Tb", "shape": [2, 12, 16, 16], "dtype": "float16",
         "op": {"name": "score_bias", "a_src": "BS_A", "w_src": "BS_W",
                "pairs": [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 2], [3, 2], [2, 3],
                          [3, 3]]},
         "src": "W", "base": 72000, "axes": []},
        {"name": "w2", "shape": [2, 6], "dtype": "uint32", "op": {"name": "u32"}, "src": "IMG-A", "base": 100,
         "axes": [lin(6), lin(1)], "also": [{"src": "IMG-B", "base": 4}]},
        {"name": "TR", "shape": [5, 4], "dtype": "float32", "op": {"name": "u32_as_f32"}, "src": "IMG-A",
         "base": 200, "axes": [lin(4), lin(1)], "also": [{"src": "IMG-B", "base": 104}, {"src": "IMG-A", "base": 300}]},
    ]}
    g2 = {"group": "constants", "tensors": [
        {"name": "A_CLAMP", "shape": [1], "dtype": "float16", "op": {"name": "const_f16x2"},
         "sites": [{"src": "IMG-A", "elements": [10, 11]}, {"src": "IMG-V", "elements_by_variant": [[5], [7, 8]]}]},
        {"name": "A_CLAMPn", "shape": [1], "dtype": "float16", "op": {"name": "const_f16x2"},
         "sites": [{"src": "IMG-A", "elements": [12]}]},
        {"name": "A_P1", "shape": [1], "dtype": "float16", "op": {"name": "const_f16lo"},
         "sites": [{"src": "IMG-A", "elements": [13, 14, 15]}]},
        {"name": "A_P0", "shape": [1], "dtype": "float16", "op": {"name": "const_f16lo"},
         "sites": [{"src": "IMG-A", "elements": [16]}]},
        {"name": "RW_A", "shape": [1], "dtype": "float16", "op": {"name": "const_f32_to_f16", "negate": True},
         "sites": [{"src": "IMG-A", "elements": [17, 18]}]},
        {"name": "RW_B", "shape": [1], "dtype": "float16", "op": {"name": "const_f32_to_f16"},
         "sites": [{"src": "IMG-A", "elements": [19]}]},
        {"name": "X_KEY", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-A", "elements": [20]}], "require": ["finite", "not_subnormal"]},
        {"name": "X_MFL", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-A", "elements": [21]}, {"src": "IMG-V", "elements_by_variant": [[], [9]]}],
         "require": ["finite", "not_subnormal"]},
        {"name": "X_CL", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-A", "elements": [22]}]},
        {"name": "X_CLn", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-A", "elements": [23]}]},
        {"name": "SUB", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-A", "elements": [24]}]},
        {"name": "TBW", "shape": [1], "dtype": "float32", "op": {"name": "u32_as_f32"}, "src": "BS_A", "base": 0,
         "axes": [lin(1)]},
        {"name": "SH", "shape": [2, 3, 2], "dtype": "uint32", "op": {"name": "u32"}, "src": "SH", "base": 0,
         "axes": [lin(6), lin(2), lin(1)], "require": ["even", "below_8"]},
        {"name": "WMAX", "shape": [1], "dtype": "float16",
         "op": {"name": "attn_wmax", "clamp_lo": "A_CLAMPn", "clamp_hi": "A_CLAMP", "p1": "A_P1", "p0": "A_P0"}},
    ], "checks": [
        {"rel": "gt", "a": "X_KEY", "b": {"const": 0}},
        {"rel": "ge", "a": "X_MFL", "b": {"const": 0.0001220703125}},
        {"rel": "le", "a": "X_CLn", "b": "X_CL"},
        {"rel": "ratio_below_f16_overflow", "a": "X_KEY", "b": "X_MFL"},
    ]}
    g3 = {"group": "only_v1", "requires_variant": {"IMG-V": 1}, "tensors": [
        {"name": "C", "shape": [1], "dtype": "float32", "op": {"name": "const_u32_as_f32"},
         "sites": [{"src": "IMG-V", "elements_by_variant": [[], [30, 31]]}]},
        {"name": "tail", "shape": [8], "dtype": "uint32", "op": {"name": "u32"}, "src": "IMG-V", "base": 40,
         "axes": [lin(1)]},
    ]}
    m["groups"] = [g1, g2, g3]
    return m


def feature_blocks(rng):
    """Synthetic content for feature_case_map."""
    nW = 72000 + 2 * 4096 + 64
    w = [0] * nW
    finite = [h for h in range(65536) if is_half_finite(h)]
    assert len(finite) == 63488
    w[:63488] = finite
    for i in range(63488, nW):
        w[i] = rng.choice(finite)
    # exp inputs: overflow (capped), large, small, zero, negative zero, subnormal
    w[71200] = h16(100.0)                       # e^100 overflows binary32: the cap applies
    exps = [h16(x) for x in (0.0, -0.0, 1.0, -1.0, 2.5, 10.0, -10.0, 50.0, -50.0, 88.0, -87.0, -103.0, -104.0,
                             0.693359375, 5.9604644775390625e-08, -5.9604644775390625e-08, 6.103515625e-05, 20.0,
                             -20.0, 11.0)]
    exps += [rng.choice(finite[:15360]) for _ in range(20)]  # magnitudes below 1
    w[71300:71340] = exps
    w[71400:71403] = [h16(1.0), h16(3.0), h16(-1.0)]
    # score-bias tensor S [2][64][64]: moderate values so that nothing overflows with the chosen A and Wd
    small = [h for h in finite if (h >> 10) & 0x1F <= 20]  # |x| < 64, both signs, subnormals and zeros included
    for i in range(72000, 72000 + 8192):
        w[i] = rng.choice(small)
    for h in range(2):  # sign of zero: (-0) + (-0) = -0, x + (-x) = +0
        base = 72000 + h * 4096
        for a, b in ((3, 40), (17, 17), (63, 0)):
            w[base + a * 64 + b] = w[base + b * 64 + a] = 0x8000
        for a, b in ((5, 9), (33, 34)):
            w[base + a * 64 + b] = h16(1.5)
            w[base + b * 64 + a] = h16(-1.5)
    W = b"".join(struct.pack("<H", x) for x in w)
    D = b"".join(struct.pack("<H", rng.getrandbits(16)) for _ in range(300))
    SH = b"".join(struct.pack("<I", rng.choice((0, 2, 4, 6))) for _ in range(12)) + b"\x07" * 16
    img = [rng.getrandbits(32) for _ in range(400)]
    img_b = [rng.getrandbits(32) for _ in range(200)]
    for k in range(12):  # w2: IMG-A[100..111] == IMG-B[4..15]
        img_b[4 + k] = img[100 + k]
    for k in range(20):  # TR: IMG-A[200..219] == IMG-B[104..123] == IMG-A[300..319]; finite binary32 values
        v = struct.unpack("<I", f32b(rng.uniform(-4, 4)))[0]
        img[200 + k] = img_b[104 + k] = img[300 + k] = v
    hx = h16(-0.75)
    hclamp = h16(0.5)
    img[10] = img[11] = (hclamp << 16) | hclamp
    img[12] = (h16(-0.5) << 16) | h16(-0.5)
    img[13] = img[14] = img[15] = h16(0.875)
    img[16] = h16(1.375)
    img[17] = img[18] = struct.unpack("<I", f32b(0.1))[0]
    img[19] = struct.unpack("<I", f32b(1e-6))[0]
    img[20] = struct.unpack("<I", f32b(65.0))[0]          # X_KEY
    img[21] = struct.unpack("<I", f32b(0.25))[0]          # X_MFL
    img[22] = struct.unpack("<I", f32b(2.0))[0]
    img[23] = struct.unpack("<I", f32b(-2.0))[0]
    img[24] = 0x000116c2                                   # a subnormal binary32 (allowed without not_subnormal)
    v0 = [rng.getrandbits(32) for _ in range(64)]
    v1 = [rng.getrandbits(32) for _ in range(80)]
    v0[5] = img[10]
    v1[7] = v1[8] = img[10]
    v1[9] = img[21]
    v1[30] = v1[31] = struct.unpack("<I", f32b(-3.5))[0]
    pack32 = lambda xs: b"".join(struct.pack("<I", x) for x in xs)
    blocks = {
        "W": W, "D": D, "SH": SH,
        "BS_A": window(rng, 8, f32b(0.125)), "BS_W": window(rng, 8, f32b(4.0)),
        "CAP": window(rng, 8, f32b(1000.0)), "H.x": window(rng, 8, struct.pack("<H", hx)),
        "IMG-A": pack32(img), "IMG-V": [pack32(v0), pack32(v1)], "IMG-B": pack32(img_b),
    }
    return blocks


def feature_cases(seed=57):
    """Two files (IMG-V variant 0 and variant 1) of the feature map, with decoys and odd alignments."""
    rng = random.Random(seed)
    m = feature_case_map()
    blocks = feature_blocks(rng)
    key, _ = finalize(m, blocks)
    cases = []
    W = blocks["W"]
    for v in (0, 1):
        c = Case5("feature-v%d" % v)
        c.map = copy.deepcopy(m)
        c.blocks = blocks
        c.key = key
        lay = [("W", 4096 + 2), ("D", 300001), ("SH", 300656), ("BS_A", 310001), ("BS_W", 310100),
               ("CAP", 310203), ("H.x", 310300), ("IMG-A", 320016), ("IMG-V", 330000 + 5 * v),
               ("IMG-B", 340010)]
        # Decoys: a damaged copy of IMG-A earlier (same first 64 bytes, a later byte changed) and a damaged copy of
        # D at a 16-aligned offset before it; an intact second copy of SH later (the first copy is used).
        bad_a = bytearray(blocks["IMG-A"])
        bad_a[200] ^= 1
        bad_d = bytearray(blocks["D"])
        bad_d[-1] ^= 0x80
        decoys = [(200000, bytes(bad_a)), (290000, bytes(bad_d))]
        extra = [(380000, blocks["SH"])]
        c.file = build_file(rng, blocks, m, lay, 400000, variant_choice={"IMG-V": v}, decoys=decoys, extra=extra)
        c.placements = dict(lay)
        c.variant = {"IMG-V": v}
        found = {n: (b[v] if isinstance(b, list) else b) for n, b in blocks.items()}
        c.expected, _ = decode_all(c.map, found, {"IMG-V": v})
        cases.append(c)
    return cases


def check_t5_vectors(path):
    """The spec's 57-T5 vectors against this reference. Returns (checked, failures)."""
    with open(path) as f:
        vec = json.load(f)
    fails = []
    n = 0
    codes = bytes.fromhex(vec["e4m3"]["codes_hex"])
    for h in range(65536):
        n += 1
        try:
            got = e4m3_code(h)
        except Rejected:
            got = None
        want = None if (codes[h] == 0xFF and not is_half_finite(h)) else codes[h]
        if got != want:
            fails.append("e4m3 %04x" % h)
    for c in vec["score_bias"]:
        n += 1
        S = list(struct.unpack("<%dH" % (len(bytes.fromhex(c["S_f16le"])) // 2), bytes.fromhex(c["S_f16le"])))
        try:
            got = op_score_bias(S, c["H"], int(c["A_bits"], 16), int(c["Wd_bits"], 16), c["pairs"])
            got = b"".join(struct.pack("<H", x) for x in got).hex()
        except Rejected:
            got = "error"
        if got != c["expect"]:
            fails.append("score_bias " + c["case"])
    for c in vec["const"]:
        n += 1
        try:
            bits = op_const(c["op"], c.get("negate", False), [int(x, 16) for x in c["site_values"]])
            width = 8 if c["op"] == "const_u32_as_f32" else 4
            out = struct.pack("<I" if width == 8 else "<H", bits)
            dt = "float32" if width == 8 else "float16"
            got = ("%0" + str(width) + "x") % bits if finite_output(dt, out) else "error"
        except Rejected:
            got = "error"
        if got != c["expect"]:
            fails.append("const %s %s" % (c["op"], c["site_values"]))
    for c in vec["attn_wmax"]:
        n += 1
        try:
            w, _ = op_attn_wmax(*(int(c[k], 16) for k in ("clamp_lo", "clamp_hi", "p1", "p0")))
            got = "%04x" % w
        except Rejected:
            got = "error"
        if got != c["expect"]:
            fails.append("attn_wmax " + c["case"])
    for c in vec["require"]:
        n += 1
        data = bytes.fromhex(c["values_hex"])
        ok = finite_output(c["dtype"], data) and require_ok(c["dtype"], data, c["require"])
        if ("ok" if ok else "error") != c["expect"]:
            fails.append("require " + c["case"])
    for c in vec["checks"]:
        n += 1
        a = f32_value(int(c["a_bits"], 16))
        b = const_value(c["b"]["const"]) if "const" in c["b"] else f32_value(int(c["b"]["tensor_bits"], 16))
        if ("ok" if check_ok(c["rel"], a, b) else "error") != c["expect"]:
            fails.append("checks %s %s" % (c["rel"], c["a_bits"]))
    return n, fails
