# SPDX-License-Identifier: MIT
"""Independent reference for spec 08 and builder of synthetic test cases.

Written for this project. Python standard library only. Everything here is computed independently of
the C++ converter: decode operations use exact integer (rational) arithmetic and, for exp,
50-digit decimal arithmetic; hashes use hashlib and a direct FNV-1a loop.

A synthetic case is a fake "DLL" (random bytes with our own fake source blocks W, D, K placed at
chosen offsets) plus a map whose hashes and digests are computed here from those blocks.

For the optional source F of spec 08 sections 4.3 and 4.4 this file has its own block decoder (written from the
seven steps of section 4.4, independently of the C++ one), a small greedy encoder of our own for the test blocks,
and a generator of our own filler text with hexadecimal fields in it.
"""

import copy
import hashlib
import json
import random
import struct
from decimal import Decimal, localcontext

FNV_BASIS = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


class NonFinite(Exception):
    pass


class OutOfRange(Exception):
    pass


class SourceDamaged(Exception):
    """A block of a source with variants matched its hashes but cannot be used (spec 08 section 4.3)."""


# ---------------------------------------------------------------- hashes

def fnv1a64(data, s0=FNV_BASIS):
    s = s0
    for b in data:
        s = ((s ^ b) * FNV_PRIME) & MASK64
    return s


def keyed_fnv_state(key):
    return FNV_BASIS ^ int.from_bytes(key[:8], "little")


def sha256(data):
    return hashlib.sha256(data).digest()


def keyed_sha256(key, data):
    return hashlib.sha256(key + data).digest()


# ---------------------------------------------------------------- exact numbers
#
# A real number is a pair (num, den) of Python integers with den > 0 (not necessarily reduced).
# Everything below is exact integer arithmetic; only the final conversion to bytes goes through a
# Python float, and only for values that are exactly representable in binary32.

def half_value(h):
    """value(h) as (num, den), or a string 'inf', '-inf', 'nan'."""
    sign = -1 if h & 0x8000 else 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0x1F:
        return "nan" if m else ("-inf" if sign < 0 else "inf")
    if e == 0:
        return (sign * m, 1 << 24)
    sh = e - 25
    return (sign * ((m + 1024) << sh), 1) if sh >= 0 else (sign * (m + 1024), 1 << -sh)


def f32_value(bits):
    """binary32 bit pattern -> (num, den); raises NonFinite for inf/NaN."""
    sign = -1 if bits >> 31 else 1
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    if e == 0xFF:
        raise NonFinite("inf/nan bit pattern")
    if e == 0:
        return (sign * m, 1 << 149)
    sh = e - 150
    return (sign * ((m + (1 << 23)) << sh), 1) if sh >= 0 else (sign * (m + (1 << 23)), 1 << -sh)


def round_nd(x, frac_bits, emin):
    """Round x = (num, den) to a binary format with frac_bits fraction bits and minimum normal
    exponent emin: nearest, ties to even significand, no upper exponent limit. Returns (num, den)."""
    num, den = x
    if num == 0:
        return (0, 1)
    neg = num < 0
    num = -num if neg else num
    e = num.bit_length() - den.bit_length()  # floor(log2(num/den)) is e or e - 1
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1
    e = max(e, emin)
    shift = frac_bits - e  # q = num/den * 2^shift
    if shift >= 0:
        n, r = divmod(num << shift, den)
        d = den
    else:
        d = den << -shift
        n, r = divmod(num, d)
    if 2 * r > d or (2 * r == d and n & 1):
        n += 1
    if neg:
        n = -n
    return (n, 1 << shift) if shift >= 0 else (n << -shift, 1)


def magnitude_above(x, limit_num, limit_den=1):
    return abs(x[0]) * limit_den > limit_num * x[1]


F32_MAX = ((1 << 24) - 1) << 104
F16_MAX = 65504


def rn32(x):
    r = round_nd(x, 23, -126)
    if magnitude_above(r, F32_MAX):
        raise NonFinite("binary32 overflow")
    return r


def rn16(x):
    r = round_nd(x, 10, -14)
    if magnitude_above(r, F16_MAX):
        raise NonFinite("binary16 overflow")
    return r


def q8(x):
    r = round_nd(x, 3, -6)
    if magnitude_above(r, 448):
        r = (448 if r[0] > 0 else -448, 1)
    return r


def f32_bytes(x, negative_zero=False):
    """Little-endian binary32 bytes of an exactly representable value."""
    if x[0] == 0:
        return struct.pack("<f", -0.0 if negative_zero else 0.0)
    return struct.pack("<f", x[0] / x[1])  # int / int is correctly rounded; here it is exact


def exp_rn32(x):
    with localcontext() as ctx:
        ctx.prec = 50
        e = (Decimal(x[0]) / Decimal(x[1])).exp()  # x[1] is a power of 2: the quotient is exact
    num, den = e.as_integer_ratio()
    return rn32((num, den))


# ---------------------------------------------------------------- block compression (spec 08 section 4.4)

class BlockError(Exception):
    pass


def lz4_block_decode(block, decoded_length):
    """The seven steps of spec 08 section 4.4. Raises BlockError for every malformed block."""
    out = bytearray()
    pos = 0
    n = len(block)

    def take(count):
        nonlocal pos
        if count > n - pos:
            raise BlockError("the input ends early")
        piece = block[pos: pos + count]
        pos += count
        return piece

    def extend(value):
        while True:
            b = take(1)[0]
            value += b
            if b != 255:
                return value

    while pos < n:
        t = take(1)[0]                                    # step 1
        literals, match = t >> 4, t & 15
        if literals == 15:                                # step 2
            literals = extend(literals)
        out += take(literals)                             # step 3
        if len(out) > decoded_length:
            raise BlockError("output too long")
        if pos == n:                                      # step 4
            break
        d = int.from_bytes(take(2), "little")             # step 5
        if d == 0 or d > len(out):
            raise BlockError("bad distance")
        if match == 15:                                   # step 6
            match = extend(match)
        match += 4
        if len(out) + match > decoded_length:
            raise BlockError("output too long")
        for _ in range(match):                            # step 7
            out.append(out[-d])
    if len(out) != decoded_length:
        raise BlockError("wrong output size")
    return bytes(out)


def _lz4_sequence(out, literals, distance=None, match=None):
    def extension(v):
        while v >= 255:
            out.append(255)
            v -= 255
        out.append(v)
    m = 0 if match is None else match - 4
    out.append((min(len(literals), 15) << 4) | min(m, 15))
    if len(literals) >= 15:
        extension(len(literals) - 15)
    out += literals
    if match is not None:
        out += distance.to_bytes(2, "little")
        if m >= 15:
            extension(m - 15)


def lz4_block_encode(data, literal_only=False):
    """Test-side encoder of our own (not a copy of any library): greedy, the last position of every 4-byte string
    in a dict, matches of at least 4 bytes at distances up to 65535, and a last sequence of at least 5 literals.
    literal_only: the whole input as one literal run."""
    out = bytearray()
    n = len(data)
    if literal_only or n < 13:
        _lz4_sequence(out, data)
        return bytes(out)
    seen = {}
    anchor = i = 0
    limit = n - 5  # matches end here; the bytes after it are the last literals
    while i + 4 <= limit:
        quad = data[i: i + 4]
        j = seen.get(quad)
        seen[quad] = i
        if j is None or i - j > 65535:
            i += 1
            continue
        length = 4
        while i + length < limit and data[j + length] == data[i + length]:
            length += 1
        _lz4_sequence(out, data[anchor: i], i - j, length)
        for p in range(i + 1, min(i + length, limit - 3)):
            seen[data[p: p + 4]] = p
        i += length
        anchor = i
    _lz4_sequence(out, data[anchor:])
    return bytes(out)


def check_lz4_vectors(path):
    """08-T7 against this reference: returns (number of cases, list of failing case names)."""
    with open(path) as f:
        d = json.load(f)
    bad = []
    for c in d["cases"]:
        block = bytes.fromhex(c["block"])
        try:
            out = lz4_block_decode(block, c["decoded_length"])
            ok = c["result"] == "ok" and hashlib.sha256(out).hexdigest() == c["decoded_sha256"]
            if ok and c.get("decoded") is not None:
                ok = out == bytes.fromhex(c["decoded"])
        except BlockError:
            ok = c["result"] == "error"
        if not ok:
            bad.append(c["name"])
    return len(d["cases"]), bad


# ---------------------------------------------------------------- source F (spec 08 section 4.3)

HEX_DIGITS = b"0123456789abcdefABCDEF"
FILLER_WORDS = ("lantern", "orchard", "pebble", "quiet", "river", "saddle", "timber", "umbrella", "violet", "window",
                "yonder", "zephyr", "table", "kernel", "margin", "north", "output", "weight", "row", "column")


def filler_text(rng, size):
    """`size` bytes of our own made-up text: words, numbers and punctuation in lines (it compresses like text)."""
    out = bytearray()
    while len(out) < size:
        line = []
        for _ in range(rng.randint(3, 9)):
            line.append(rng.choice(FILLER_WORDS) if rng.random() < 0.8 else str(rng.randrange(0, 5000)))
        out += (" ".join(line) + rng.choice((".", ";", ":", ",")) + "\n").encode()
    return bytes(out[:size])


def make_f_variant(rng, elements, size, head=b"", upper=True, positions=None):
    """One variant of a synthetic source F: text of exactly `size` bytes that starts with `head`, holds each of the
    32-bit `elements` as eight hexadecimal digits (between a space and a ';', so that a shifted field is not
    hexadecimal), and its compressed block. positions: the byte offset of each field (default: random places after
    the head, not in ascending order). upper: digit case; False gives lower case, None alternates per field."""
    text = bytearray(head + filler_text(rng, size - len(head)))
    if positions is None:
        slots = rng.sample(range((len(head) + 12) // 12 + 1, (size - 12) // 12), len(elements))  # 12-byte cells
        positions = [12 * s for s in slots]
    for e, (p, x) in enumerate(zip(positions, elements)):
        digits = "%08X" % x
        if upper is False or (upper is None and e % 2):
            digits = digits.lower()
        assert len(head) <= p - 1 and p + 9 <= size
        text[p - 1: p + 9] = b" " + digits.encode() + b";"
    text = bytes(text)
    assert len(text) == size
    block = lz4_block_encode(text)
    assert lz4_block_decode(block, size) == text
    return {"text": text, "block": block, "fields": list(positions), "elements": list(elements)}


def f_variant_entry(v, key):
    return {"length": len(v["block"]),
            "fnv1a64": "0x%016x" % fnv1a64(v["block"][:64], keyed_fnv_state(key)),
            "sha256": keyed_sha256(key, v["block"]).hex(),
            "decoded_length": len(v["text"]),
            "decoded_sha256": keyed_sha256(key, v["text"]).hex(),
            "fields": list(v["fields"])}


def f_source_entry(what, variants, key):
    """sources.F of spec 08 section 5 for our own variants."""
    return {"what": what, "element": "hex32", "locate": "block", "keyed": True, "encoding": "lz4-block",
            "optional": True, "variants": [f_variant_entry(v, key) for v in variants]}


def variant_elements(src, key, block):
    """Reference for section 4.3: (variant index, the elements as little-endian u32 bytes) for the block that is in
    the file, or None if it is no variant of the map. Raises SourceDamaged if a matching block cannot be used."""
    if block is None:
        return None
    for i, v in enumerate(src["variants"]):
        if len(block) != v["length"] or keyed_sha256(key, block).hex() != v["sha256"]:
            continue
        try:
            text = lz4_block_decode(block, v["decoded_length"])
        except BlockError as exc:
            raise SourceDamaged(str(exc))
        if keyed_sha256(key, text).hex() != v["decoded_sha256"]:
            raise SourceDamaged("decoded_sha256")
        out = b""
        for f in v["fields"]:
            digits = text[f: f + 8]
            if len(digits) != 8 or any(c not in HEX_DIGITS for c in digits):
                raise SourceDamaged("field is not hexadecimal")
            out += struct.pack("<I", int(digits, 16))
        return i, out
    return None


# ---------------------------------------------------------------- decode (spec 08 sections 5.1, 6)

ELEMENT_SIZE = {"f16le": 2, "u16le": 2, "u32le": 4, "i32le": 4}


def read_element(src_bytes, element, e):
    size = ELEMENT_SIZE[element]
    return int.from_bytes(src_bytes[size * e: size * e + size], "little")


def op_key(op):
    return tuple(sorted((k, str(v)) for k, v in op.items() if k != "note"))


class OpCache:
    """Memoised exact decode of one element for one op."""

    def __init__(self):
        self.memo = {}

    def decode(self, op, x, key=None):
        k = (key or op_key(op), x)
        hit = self.memo.get(k)
        if hit is None:
            try:
                hit = ("ok", decode_element(op, x))
            except NonFinite as exc:
                hit = ("nonfinite", str(exc))
            except OutOfRange as exc:
                hit = ("range", str(exc))
            self.memo[k] = hit
        if hit[0] == "nonfinite":
            raise NonFinite(hit[1])
        if hit[0] == "range":
            raise OutOfRange(hit[1])
        return hit[1]


def decode_element(op, x):
    name = op["name"]
    if name == "u16_shr":
        y = x >> op["shift"]
        if y > 255:
            raise OutOfRange("u16_shr result above 255")
        return bytes([y])
    if name == "u32_as_f32":
        f32_value(x)  # raises NonFinite for inf/NaN patterns
        return struct.pack("<I", x)
    if name == "i32":
        return struct.pack("<I", x)  # the int32 bit pattern, unchanged
    v = half_value(x)
    neg_x = bool(x & 0x8000)
    if name == "f16":
        if isinstance(v, str):
            raise NonFinite("f16 of " + v)
        return f32_bytes(v, negative_zero=neg_x)
    if name == "fp8":
        if isinstance(v, str):
            raise NonFinite("fp8 of " + v)
        return f32_bytes(q8(v))  # a zero result is +0
    if name == "f16_scaled":
        if isinstance(v, str):
            raise NonFinite("f16_scaled of " + v)
        mul_bits, div_bits = int(op["mul"], 16), int(op["div"], 16)
        mul, div = f32_value(mul_bits), f32_value(div_bits)
        if div[0] == 0:
            raise NonFinite("division by zero")
        p = rn32((v[0] * mul[0], v[1] * mul[1]))
        h = rn16(p)
        q = (h[0] * div[1], h[1] * div[0])
        if q[1] < 0:
            q = (-q[0], -q[1])
        y = rn32(q)
        # IEEE rounding keeps the sign of a zero result: sign(x) * sign(mul) * sign(div).
        return f32_bytes(y, negative_zero=bool((x >> 15 ^ mul_bits >> 31 ^ div_bits >> 31) & 1))
    if name == "exp":
        if v == "nan":
            raise NonFinite("exp of nan")
        if v == "-inf":
            y = (0, 1)
        elif v == "inf":
            y = None
        else:
            try:
                y = exp_rn32(v)
            except NonFinite:
                y = None
        if "max" in op:
            mx = f32_value(int(op["max"], 16))
            if y is None or y[0] * mx[1] > mx[0] * y[1]:
                y = mx
        if y is None:
            raise NonFinite("exp overflow")
        return f32_bytes(y)
    raise ValueError("unknown op " + name)


SHARED_CACHE = OpCache()


def perm_index(i, table):
    b = len(table)
    return b * (i // b) + table[i % b]


def capped_softmax_exp(op, x):
    """Steps 1-3 of spec 48 section 6 for one binary16 pattern x: E = RN32(e^g) as (num, den)."""
    v = half_value(x)
    if isinstance(v, str):
        raise NonFinite("capped_softmax of " + v)
    cap, div, sub = (f32_value(int(op[k], 16)) for k in ("cap", "div", "sub"))
    lt = lambda a, b: a[0] * b[1] < b[0] * a[1]  # noqa: E731 (dens > 0)
    neg_cap = (-cap[0], cap[1])
    u = neg_cap if lt(v, neg_cap) else (cap if lt(cap, v) else v)
    u = rn32(u)
    a = rn32((u[0] * u[0], u[1] * u[1]))
    if div[0] == 0:
        raise NonFinite("division by zero")
    a = rn32((a[0] * div[1], a[1] * div[0]) if div[0] > 0 else (-a[0] * div[1], -a[1] * div[0]))
    a = rn32((a[1] - a[0], a[1]))
    a = rn32((a[0] * u[0], a[1] * u[1]))
    g = rn32((a[0] * sub[1] - sub[0] * a[1], a[1] * sub[1]))
    return exp_rn32(g)


def capped_softmax_group(op, xs, cache):
    """Steps 4-5 of spec 48 section 6 for 64 binary16 patterns (s = 8 i + j): 64 binary32 values as bytes."""
    E = [cache.decode_exp(op, x) for x in xs]
    s = (0, 1)
    for e in E:
        s = rn32((s[0] * e[1] + e[0] * s[1], s[1] * e[1]))
    if s[0] == 0:
        raise NonFinite("zero sum")
    out = b""
    for e in E:
        h = rn16(e)
        y = rn16(rn32((h[0] * s[1], h[1] * s[0])))  # s > 0
        out += f32_bytes(y)
    return out


def _decode_exp(self, op, x):
    k = ("cs-exp", op_key(op), x)
    hit = self.memo.get(k)
    if hit is None:
        hit = self.memo[k] = capped_softmax_exp(op, x)
    return hit


OpCache.decode_exp = _decode_exp


def decode_base64_segments(text, segments):
    """W of spec 48 section 4: every segment decoded (RFC 4648, standard alphabet, '=' padding) and concatenated.
    Raises OutOfRange for a segment that is not valid Base64 or an odd total (installer errors)."""
    import base64
    import binascii
    out = bytearray()
    for off, length in segments:
        seg = text[off: off + length]
        if len(seg) != length or length % 4:
            raise OutOfRange("segment outside T")
        try:
            out += base64.b64decode(seg, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise OutOfRange("bad Base64: %s" % exc)
    if len(out) % 2:
        raise OutOfRange("odd decoded length")
    return bytes(out)


def arrays_of(m, sources, key=None):
    """The element arrays tensors index: {name: (element, bytes)}. Versions 1 and 4: the sources themselves;
    versions 2 and 3: W from the Base64 source T, and O. A source with variants (F, version 4) is an array only if
    sources[name] is the block of one of its variants; its elements are what the hexadecimal fields write."""
    arrays = {}
    for name, src in m["sources"].items():
        if src["element"] == "base64":
            arrays["W"] = ("f16le", decode_base64_segments(sources[name], src["segments"]))
        elif src["element"] == "hex32":
            found = variant_elements(src, key, sources.get(name))
            if found is not None:
                arrays[name] = ("u32le", found[1])
        else:
            arrays[name] = (src["element"], sources[name])
    return arrays


def variant_source_of(m, st):
    """The source with variants that every tensor of this stage reads, or None."""
    names = {t["src"] for t in st["tensors"]}
    hexed = {n for n in names if m["sources"].get(n, {}).get("element") == "hex32"}
    return next(iter(hexed)) if hexed and len(names) == 1 else None


def decode_tensor(m, t, arrays, cache):
    element, data = arrays[t["src"]]
    count = len(data) // ELEMENT_SIZE[element]
    shape = t["shape"]
    axes = t["axes"]
    out = bytearray()
    key = op_key(t["op"])
    group = t["op"]["name"] == "capped_softmax"
    xs = []
    total = 1
    for n in shape:
        total *= n
    for flat in range(total):
        idx = []
        rem = flat
        for n in reversed(shape):
            idx.append(rem % n)
            rem //= n
        idx.reverse()
        e = t["base"]
        for a, i in enumerate(idx):
            p = axes[a].get("perm")
            pi = perm_index(i, m["perms"][p]) if p is not None else i
            e += axes[a]["stride"] * pi
        if not 0 <= e < count:
            raise OutOfRange("index outside source")
        x = read_element(data, element, e)
        if group:
            xs.append(x)
            if len(xs) == 64:
                out += capped_softmax_group(t["op"], xs, cache)
                xs = []
        else:
            out += cache.decode(t["op"], x, key)
    return bytes(out)


def decode_all(m, sources):
    """Returns {stage: [(name, bytes), ...]} and fills in digests if asked by the caller. A stage that reads a
    source with variants of which none is in `sources` is left out (spec 08 section 4.3)."""
    key = sha256(m["key"]["label"].encode() + sources[m["key"]["from"]])
    cache = SHARED_CACHE
    arrays = arrays_of(m, sources, key)
    result = {}
    digests = {}
    for st in m["stages"]:
        if variant_source_of(m, st) is not None and st["tensors"][0]["src"] not in arrays:
            continue
        tensors = []
        for t in st["tensors"]:
            tensors.append((t["name"], decode_tensor(m, t, arrays, cache)))
        result[st["stage"]] = tensors
        digests[st["stage"]] = keyed_sha256(key, b"".join(b for _, b in tensors)).hex()
    return result, digests, key


# ---------------------------------------------------------------- case building

def source_entry(name, what, element, data, locate="block", gaps=None, keyed=False, key=None):
    e = {"what": what, "element": element, "length": len(data), "locate": locate}
    if gaps is not None:
        e["gaps"] = gaps
    if keyed:
        e["keyed"] = True
        s0 = keyed_fnv_state(key)
        e["fnv1a64"] = "0x%016x" % fnv1a64(data[: min(64, len(data))] if locate == "block" else data, s0)
        e["sha256"] = keyed_sha256(key, data).hex()
    else:
        e["fnv1a64"] = "0x%016x" % fnv1a64(data[: min(64, len(data))])
        e["sha256"] = sha256(data).hex()
    return e


def t_source_entry(what, text, segments):
    """sources.T of spec 48 section 5 for a Base64 text block."""
    e = source_entry("T", what, "base64", text)
    e["segments"] = [list(s) for s in segments]
    return e


def build_t_text(rng, chunks, order=None, gap=(4, 16), lead=0, tail=0):
    """T text from binary chunks: each chunk Base64-encoded (own padding) at its own place, with junk text
    (including '=', newlines and Base64 letters) between them. order: the order of the chunks in the text (the
    map lists the segments in chunk order). Returns (text, segments in chunk order)."""
    import base64
    order = list(range(len(chunks))) if order is None else order
    junk_chars = b'"=,:\n {}[]ABCxyz019+/'
    text = bytearray(bytes(rng.choice(junk_chars) for _ in range(lead)))
    segs = [None] * len(chunks)
    for n, k in enumerate(order):
        if n:
            text += bytes(rng.choice(junk_chars) for _ in range(rng.randint(*gap)))
        enc = base64.b64encode(chunks[k])
        segs[k] = (len(text), len(enc))
        text += enc
    text += bytes(rng.choice(junk_chars) for _ in range(tail))
    return bytes(text), segs


def check_capped_softmax_vectors(path):
    """48-T5 against this reference: returns the number of mismatching y."""
    with open(path) as f:
        d = json.load(f)
    op = {"name": "capped_softmax", "cap": d["cap"], "div": d["div"], "sub": d["sub"]}
    bad = 0
    for xs, ys in zip(d["x"], d["y"]):
        got = capped_softmax_group(op, [int(x, 16) for x in xs], SHARED_CACHE)
        want = b"".join(struct.pack("<I", int(y, 16)) for y in ys)
        bad += sum(got[4 * i: 4 * i + 4] != want[4 * i: 4 * i + 4] for i in range(len(ys)))
    return bad


def k_shipped_like_case(map_path, seed=1, t_offset=None, o_offset=None, file_extra=1 << 20, max_exp_field=0x16):
    """A case with exactly the layout of the shipped preset-K map (segments, stages, tensors, bases, strides, ops)
    but with our own random content in T and O, and hashes and digests recomputed for that content. Each segment
    gets its own Base64 padding; the decoded total equals the map's W (the largest index the map uses is in range
    only if the total is right)."""
    import base64
    rng = random.Random(seed)
    with open(map_path, "rb") as f:
        m = json.load(f)
    st = m["sources"]["T"]
    segs = st["segments"]
    # Decoded bytes per segment: 3/4 of its length minus its padding; the map's index range fixes the total.
    max_index = 0
    for stage in m["stages"]:
        for t in stage["tensors"]:
            if t["src"] == "W":
                max_index = max(max_index, t["base"] + sum(max(0, a["stride"]) * (n - 1) for a, n in zip(t["axes"], t["shape"])))
    upper = sum(n // 4 * 3 for _, n in segs)
    pads = upper - 2 * (max_index + 1)
    assert 0 <= pads <= 2 * len(segs)
    text = bytearray(rng.choice(b'"=,:\n {}[]AQz09+/') for _ in range(st["length"]))
    w = bytes(random_finite_halves(rng, max_index + 1, max_exp_field=max_exp_field))  # |x| < 2^(max_exp_field - 14)
    pos = 0
    for off, n in segs:
        p = min(2, pads)
        pads -= p
        piece = w[pos: pos + n // 4 * 3 - p]  # pieces of odd length split elements; W is their concatenation
        pos += len(piece)
        enc = base64.b64encode(piece)
        assert len(enc) == n
        text[off: off + n] = enc
    assert pos == len(w)
    text = bytes(text)
    o = struct.pack("<16i", *[rng.randrange(0, 8) for _ in range(16)])
    key = sha256(m["key"]["label"].encode() + text)
    m["sources"]["T"] = t_source_entry(st["what"], text, segs)
    m["sources"]["O"] = source_entry("O", m["sources"]["O"]["what"], "i32le", o, keyed=True, key=key)
    sources = {"T": text, "O": o}
    size = len(text) + 64 + file_extra
    buf = random_bytes(rng, size)
    t_offset = t_offset if t_offset is not None else 4096
    o_offset = o_offset if o_offset is not None else t_offset + len(text) + 512
    place(buf, t_offset, text)
    place(buf, o_offset, o)
    case = Case("k-shipped-like")
    case.file = bytes(buf)
    case.map = m
    case.sources = sources
    case.expected, case.key = finalize_map(m, sources)
    case.placements = {"T": t_offset, "O": o_offset}
    return case


def finalize_map(m, sources, variant_blocks=None):
    """Computes source hashes (already set by caller) and stage digests; returns expected outputs.
    variant_blocks: {source: [the block of each variant, in map order]} for a source with variants; the digest of
    a stage that reads it becomes the list with one entry per variant (spec 08 section 5)."""
    out, digests, key = decode_all(m, sources)
    for st in m["stages"]:
        vs = variant_source_of(m, st)
        if vs is None:
            st["digest"] = digests[st["stage"]]
            continue
        st["digest"] = []
        for block in variant_blocks[vs]:
            _, elements = variant_elements(m["sources"][vs], key, block)
            try:
                data = b"".join(decode_tensor(m, t, {vs: ("u32le", elements)}, SHARED_CACHE) for t in st["tensors"])
            except NonFinite:  # a test variant with a NaN: the converter refuses it before any digest is compared
                data = b""
            st["digest"].append(keyed_sha256(key, data).hex())
    return out, key


class Case:
    def __init__(self, name):
        self.name = name
        self.file = None          # bytes of the fake DLL
        self.map = None           # dict
        self.sources = None       # {name: bytes}
        self.expected = None      # {stage: [(name, bytes)]}
        self.key = None
        self.placements = {}
        self.found = {}           # {stage: index of the variant in the file} for stages with a digest list
        self.variants = None      # the variants of source F (make_f_variant), if the map has it


def place(buf, offset, data):
    buf[offset: offset + len(data)] = data


def random_bytes(rng, n):
    out = bytearray()
    while len(out) < n:
        out += rng.randbytes(min(1 << 20, n - len(out)))
    return out


def random_finite_halves(rng, n, max_exp_field=0x18):
    """n random binary16 bit patterns, finite, |x| < 2^(max_exp_field - 14)."""
    out = bytearray()
    for _ in range(n):
        e = rng.randrange(0, max_exp_field + 1)
        h = (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)
        out += struct.pack("<H", h)
    return out


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def version_resource(major, minor, build, rev):
    """A minimal VS_VERSIONINFO blob (header, key, padding, VS_FIXEDFILEINFO), for the display test."""
    key = "VS_VERSION_INFO\0".encode("utf-16-le")
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x00010000, (major << 16) | minor, (build << 16) | rev,
                        (major << 16) | minor, (build << 16) | rev, 0x3F, 0, 0x40004, 2, 0, 0, 0)
    body = key + b"\0\0" + fixed
    return struct.pack("<HHH", 6 + len(body), len(fixed), 0) + body


def shipped_like_case(map_path, seed=1, w_offset=None, d_offset=None, k_offset=None, file_extra=1 << 20,
                      f_present=0, f_shift=0):
    """A case with exactly the layout of a shipped map (stages, tensors, bases, strides, perms, ops)
    but with our own random source content, and hashes/digests recomputed for that content.
    For the optional source F (map version 4): our own text of the map's decoded_length with our own numbers at the
    map's field offsets, compressed by our encoder; f_present is the variant whose block is in the file (None: no
    block); the block is put at a multiple of 16 plus f_shift."""
    rng = random.Random(seed)
    with open(map_path, "rb") as f:
        shipped = json.load(f)
    m = copy.deepcopy(shipped)
    lw = m["sources"]["W"]["length"]
    ld = m["sources"]["D"]["length"]
    w = random_finite_halves(rng, lw // 2, max_exp_field=0x16)  # |x| < 256: no overflow in any op
    # The exp scalars get small arguments so that e^x stays well inside binary32.
    for st in m["stages"]:
        for t in st["tensors"]:
            if t["op"]["name"] == "exp":
                v = rng.uniform(-4.0, 4.0)
                if "max" in t["op"]:
                    v = 3.0 if seed % 2 else 1.5  # e^3 > 8 is capped by max = 8, e^1.5 is not
                place(w, 2 * t["base"], struct.pack("<H", half_bits(v)))
    w = bytes(w)
    d = bytes(random_bytes(rng, ld))
    k = struct.pack("<ff", rng.uniform(0.1, 2.0), rng.uniform(-3.0, 3.0))
    sources = {"W": w, "D": d, "K": k}
    key = sha256(m["key"]["label"].encode() + w)
    gap = m["sources"]["K"]["gaps"][0]
    for name, data in (("W", w), ("D", d)):
        src = m["sources"][name]
        m["sources"][name] = source_entry(name, src["what"], src["element"], data)
    src = m["sources"]["K"]
    m["sources"]["K"] = source_entry("K", src["what"], src["element"], k, locate="pair",
                                     gaps=src["gaps"], keyed=True, key=key)
    size = lw + ld + 64 + file_extra
    buf = random_bytes(rng, size)
    w_offset = w_offset if w_offset is not None else 4096
    d_offset = d_offset if d_offset is not None else w_offset + lw + 512
    k_offset = k_offset if k_offset is not None else d_offset + ld + 1000
    place(buf, w_offset, w)
    place(buf, d_offset, d)
    place(buf, k_offset, k[:4])
    place(buf, k_offset + gap, k[4:])
    case = Case("shipped-like")
    case.placements = {"W": w_offset, "D": d_offset, "K": k_offset}
    variant_blocks = None
    if "F" in m["sources"]:
        src = m["sources"]["F"]
        head = filler_text(random.Random(seed + 500), 700)  # the same start: the same first 64 block bytes
        case.variants = []
        for i, v in enumerate(src["variants"]):
            elements = [struct.unpack("<I", struct.pack("<f", rng.uniform(-1.0, 1.0)))[0] for _ in v["fields"]]
            case.variants.append(make_f_variant(rng, elements, v["decoded_length"], head, upper=(None, True, False)[(seed + i) % 3],
                                                positions=v["fields"]))
        m["sources"]["F"] = f_source_entry(src["what"], case.variants, key)
        variant_blocks = {"F": [v["block"] for v in case.variants]}
        if f_present is not None:
            block = case.variants[f_present]["block"]
            f_offset = (k_offset + gap + 4096 + 15) // 16 * 16 + f_shift
            assert f_offset + len(block) <= len(buf)
            place(buf, f_offset, block)
            sources["F"] = block
            case.placements["F"] = f_offset
            case.found = {st["stage"]: f_present for st in m["stages"] if variant_source_of(m, st) == "F"}
    case.file = bytes(buf)
    case.map = m
    case.sources = sources
    case.expected, case.key = finalize_map(m, sources, variant_blocks)
    return case
