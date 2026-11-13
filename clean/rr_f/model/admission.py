# SPDX-License-Identifier: MIT
"""Pure CPU admission of original synthetic RR-F prepared metadata.

Authority: published specs 51/55/58 and project-owned prepared-known-subset v1.
No paths, serialized model containers, learned data, GPU resources or converter
offsets enter this interface. Descriptor admission is not payload validation.
"""
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import math
import re
import struct
from types import MappingProxyType

import numpy as np

CONTRACT_ID = "wronggpu.rr_f.prepared_known_subset.v1"
SOURCE_HEAD = "4e61d0747211161483c7b2218347e1f892947158"
SOURCE_SHA256 = MappingProxyType({
    "spec/50-rr-f-overview.md": "dc91895b02c8dce6018873230cc0ddfe59b70488bc22dbd83c729cb08b112f81",
    "spec/51-rr-f-window-block.md": "91f39b47ab820c8c5736f35803a2c2a8dcc694537e54d45520a8c3b6fa7b0d3a",
    "spec/55-rr-f-final-head.md": "0a5c32b0081c43a092baa39010d8ded37eacb34eda6a38d1b5ddee4390fc4c7a",
    "spec/58-rr-f-auto-exposure.md": "f417fe88d0dbb2311694e52b38eb9479cf3ed62f749bdfdf9f87a0b6e6d2d295",
})
SOURCE_GATES = (
    "52: original IN0 frontend absent", "56: complete inventory/transforms absent",
    "57: supported model builds, extraction, ARCH and S1CAP authority absent",
    "60-63: complete player integration contracts absent",
    "58: declared X domain does not establish operational finiteness",
)
ADMITTED = "ADMISSIBLE_SYNTHETIC_KNOWN_SUBSET"
REJECTED = "REJECTED_KNOWN_SUBSET"
BLOCKED = "BLOCKED_PLAYER_MODEL"
_SINGLES = ("IN0", "E1", "E2", "E3", "E4", "E5")
_LEVELS = ("D4", "D3", "D2", "D1", "D0")
_STAGES = _SINGLES + tuple(d + half for d in _LEVELS for half in ("-F", "-R"))
_WINDOW = ("A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1", "ACT_C2", "ACT_C3")
_SYMBOLS = tuple(s + "." + k for s in _STAGES for k in ("ra", "m")) + tuple(d + ".f" for d in _LEVELS) + _WINDOW
_EXPOSURE = ("X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn", "X_NRM")
_TOP = {"contract_id", "source_head", "request", "provenance", "complete_player_model", "source_frontend", "tensors", "symbols", "shifts", "head_derivation", "exposure_symbols"}
_FRONTEND = {"kind": "synthetic_IN0_boundary", "supplies": ["IN0 block x e4m3 [Ht,Wt,32]", "IN0 hold f16 [Ht,Wt,4]"], "claims_original_frontend": False}


def _catalog():
    result = {}
    channels = (32, 64, 96, 128, 160, 160)
    heads = (2, 2, 4, 4, 8, 8)
    def add(stage, name, shape, precision, axes):
        result[stage + "." + name] = (shape, precision, tuple(axes.split()))
    for stage in _STAGES:
        level = _SINGLES.index(stage) if stage in _SINGLES else int(stage[1])
        c, h = channels[level], heads[level]
        for name in ("Wq", "Wv"):
            add(stage, name, (h, c, 32), "e4m3", "head input_channel head_feature")
        add(stage, "Tb", (h, 12, 16, 16), "f16", "head stored_block query_inner key_inner")
        add(stage, "Wo", (h, 32, c), "e4m3", "head head_feature output_channel")
        for name in ("bo", "b2"):
            add(stage, name, (c,), "f16", "output_channel")
        add(stage, "W1", (c, 4*c), "e4m3", "input_channel mlp_channel")
        add(stage, "b1", (4*c,), "f16", "mlp_channel")
        add(stage, "W2", (4*c, c), "e4m3", "mlp_channel output_channel")
        if stage in _SINGLES or (stage.endswith("-R") and level > 0):
            n = 3 if stage in _SINGLES else 4
            add(stage, "Wh", (c, n), "e4m3", "input_channel state_logit")
            add(stage, "bh", (n,), "f16", "state_logit")
        if stage in ("E1", "E2", "E3", "E4"):
            add(stage, "Wm", (4*c, channels[level+1]), "e4m3", "child_then_channel next_channel")
            add(stage, "bm", (channels[level+1],), "f16", "next_channel")
        if stage == "E1":
            add(stage, "Wa", (128, 64), "e4m3", "grouped_input_channel output_channel")
            add(stage, "ba", (64,), "f16", "output_channel")
        if stage.endswith("-F"):
            add(stage, "Wup", (channels[level+1], 4*c), "e4m3", "coarse_channel child_then_channel")
            add(stage, "bup", (4*c,), "f16", "child_then_channel")
    for stage in _SINGLES + _LEVELS:
        add(stage, "g", (1,), "f16", "scalar")
    for name in ("s1", "s2"):
        add("D0-R", name, (1,), "f32", "scalar")
    return MappingProxyType(result)


TENSOR_CATALOG = _catalog()


@dataclass(frozen=True)
class AdmissionIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class AdmissionResult:
    decision: str
    issues: tuple
    payload_checked: bool = False
    attention_w_max: object = None
    z_may_overflow: object = None
    exposure_declared_domain_valid: bool = False

    @property
    def player_ready(self):
        return False

    @property
    def source_gates(self):
        return SOURCE_GATES

    @property
    def exposure_operational_status(self):
        return "UNRESOLVED_X_OPERATIONAL_DOMAIN"

    def as_dict(self):
        return {**asdict(self), "player_ready": False, "source_gates": list(SOURCE_GATES),
                "exposure_operational_status": self.exposure_operational_status,
                "contract_id": CONTRACT_ID, "source_head": SOURCE_HEAD}


class _Invalid(Exception):
    def __init__(self, code, path, message):
        self.issue = AdmissionIssue(code, path, message)


def _require(condition, code, path, message):
    if not condition:
        raise _Invalid(code, path, message)


def _fields(obj, fields, code, path):
    _require(type(obj) is dict, code, path, "expected a plain metadata object")
    _require(all(type(k) is str for k in obj), code, path, "keys must be strings")
    _require(set(obj) == set(fields), code, path, "missing or unknown fields")


def _indexed(rows, keys, code, path, id_key="id"):
    _require(type(rows) is list and len(rows) == len(keys), code, path, "wrong collection length")
    out = {}
    for row in rows:
        _require(type(row) is dict and type(row.get(id_key)) is str, code, path, "invalid identifier")
        key = row[id_key]
        _require(key in keys and key not in out, code, path, "unknown or duplicate identifier")
        out[key] = row
    return out


def _bits(bits, precision, code, path):
    width = {"e4m3": 2, "f16": 4, "f32": 8}[precision]
    _require(type(bits) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", bits) is not None,
             code, path, "expected exact lowercase hexadecimal bits")
    n = int(bits, 16)
    if precision == "e4m3":
        _require(n not in (127, 255), "FINITE_PREPARED", path, "e4m3 NaN is not a prepared value")
        return n
    value = struct.unpack(">e" if precision == "f16" else ">f", bytes.fromhex(bits))[0]
    _require(math.isfinite(value), code, path, "expected a finite scalar")
    return value


@lru_cache(maxsize=65536)
def _exp_f32_bits(h):
    """Correctly round exp of a finite binary16 value, using Decimal intervals.

    Decimal.exp is correctly rounded. Enclosing its result by adjacent Decimal
    values certifies the binary32 rounding cell; increase precision if needed.
    The early bounds are deliberately far outside the binary32 transition range.
    """
    if h > 128:
        return 0x7f800000
    if h < -256:
        return 0
    # 200 digits also represent every binary32 midpoint exactly (including
    # the half-minimum-subnormal midpoint), so boundary arithmetic is exact.
    for precision in (200, 400, 800):
        with localcontext() as ctx:
            ctx.prec = precision
            value = Decimal.from_float(h).exp()
            lower, upper = value.next_minus(), value.next_plus()
            overflow = Decimal(2)**128 - Decimal(2)**103
            if lower > overflow:
                return 0x7f800000
            if upper < Decimal(2)**-150:
                return 0
            lo, hi = 0, 0x7f7fffff
            while lo < hi:
                mid = (lo + hi + 1) // 2
                v = Decimal.from_float(struct.unpack(">f", struct.pack(">I", mid))[0])
                if v <= value:
                    lo = mid
                else:
                    hi = mid - 1
            a = Decimal.from_float(struct.unpack(">f", struct.pack(">I", lo))[0])
            b = (Decimal(2)**128 if lo == 0x7f7fffff else
                 Decimal.from_float(struct.unpack(">f", struct.pack(">I", lo+1))[0]))
            midpoint = (a+b)/2
            if upper < midpoint:
                return lo
            if lower > midpoint:
                return lo + 1
    raise ArithmeticError("unable to certify binary32 exponential rounding")


def validate_metadata(metadata, *, bindings=None):
    """Validate v1 metadata and, optionally, all 197 original synthetic arrays.

    bindings is an exact-ID dict of existing native NumPy arrays: uint8 e4m3,
    float16 or float32. Arrays must match the explicit constant-fill recipe,
    including signed-zero bits. Nothing is allocated on a GPU or read from disk.
    Malformed plain metadata returns rejection instead of implicit coercion.
    Complete-player requests are unconditionally blocked before other checks.
    """
    if type(metadata) is dict and type(metadata.get("request")) is str and metadata["request"] == "complete_player_model":
        return AdmissionResult(BLOCKED, (AdmissionIssue("PLAYER_MODEL_BLOCKED", "request", "; ".join(SOURCE_GATES)),))
    w_max = overflow = None
    x_valid = False
    try:
        _fields(metadata, _TOP, "STRUCTURE", "$")
        _require(type(metadata["contract_id"]) is str and metadata["contract_id"] == CONTRACT_ID,
                 "CONTRACT_VERSION", "contract_id", "unsupported admission contract")
        _require(type(metadata["source_head"]) is str and metadata["source_head"] == SOURCE_HEAD,
                 "SOURCE_VERSION", "source_head", "source specification revision mismatch")
        _require(type(metadata["request"]) is str and metadata["request"] == "synthetic_known_subset",
                 "SCOPE", "request", "unknown admission scope")
        _require(type(metadata["provenance"]) is str and metadata["provenance"] == "original_synthetic" and metadata["complete_player_model"] is False,
                 "PROVENANCE", "provenance", "only original synthetic known-subset input is admitted")
        frontend = metadata["source_frontend"]
        _fields(frontend, _FRONTEND, "PROVENANCE", "source_frontend")
        _require(type(frontend["kind"]) is str and frontend["kind"] == _FRONTEND["kind"] and
                 type(frontend["supplies"]) is list and all(type(v) is str for v in frontend["supplies"]) and
                 frontend["supplies"] == _FRONTEND["supplies"] and frontend["claims_original_frontend"] is False,
                 "PROVENANCE", "source_frontend", "synthetic IN0 boundary must be explicit")
        tensors = _indexed(metadata["tensors"], TENSOR_CATALOG, "TENSOR_SET", "tensors")
        fills = {}
        for name, (shape, precision, axes) in TENSOR_CATALOG.items():
            row = tensors[name]
            _fields(row, ("id", "shape", "precision", "logical_axes", "synthetic_fill_bits"), "NO_PHYSICAL_MAPPING", name)
            _require(type(row["shape"]) is list and all(type(v) is int for v in row["shape"]) and
                     tuple(row["shape"]) == shape and type(row["precision"]) is str and row["precision"] == precision and
                     type(row["logical_axes"]) is list and all(type(v) is str for v in row["logical_axes"]) and
                     tuple(row["logical_axes"]) == axes, "TENSOR_SET", name, "logical descriptor mismatch")
            _bits(row["synthetic_fill_bits"], precision,
                  "HEAD_DERIVATION" if name in ("D0-R.s1", "D0-R.s2") else "FINITE_PREPARED", name)
            fills[name] = int(row["synthetic_fill_bits"], 16)
        scalars = _indexed(metadata["symbols"], _SYMBOLS, "SCALAR_SET", "symbols")
        values = {}
        for name, row in scalars.items():
            _fields(row, ("id", "precision", "bits"), "SCALAR_SET", name)
            _require(type(row["precision"]) is str and row["precision"] == "f16", "SCALAR_SET", name, "symbol must have f16 encoding")
            values[name] = _bits(row["bits"], "f16", "FINITE_PREPARED", name)
        _require(values["A_CLAMPn"] <= values["A_CLAMP"] and values["ACT_CLn"] <= values["ACT_CL"],
                 "CLAMP_DOMAIN", "symbols", "clamp endpoints must be ordered independent finite values")
        from ..reference.scalar import WindowConsts, check_attention_consts
        with np.errstate(over="ignore", invalid="ignore"):
            attention = check_attention_consts(WindowConsts(**{name: values[name] for name in _WINDOW}))
        _require(attention["ok"], "ATTENTION_RANGE", "symbols", "clamped binary16 score leaves the specified attention range")
        w_max, overflow = attention["w_max"], attention["z_may_overflow"]
        shifts = metadata["shifts"]
        _require(type(shifts) is list and len(shifts) == 2, "SHIFT_SET", "shifts", "both parity blocks required")
        parities = set()
        for block in shifts:
            _fields(block, ("parity", "rows"), "SHIFT_SET", "shifts")
            p = block["parity"]
            _require(type(p) is int and p in (0, 1) and p not in parities, "SHIFT_SET", "shifts", "invalid or duplicate parity")
            parities.add(p)
            rows = _indexed(block["rows"], _SINGLES + _LEVELS, "SHIFT_SET", "shifts", "stage")
            for name, row in rows.items():
                _fields(row, ("stage", "xy"), "SHIFT_SET", name)
                _require(type(row["xy"]) is list and len(row["xy"]) == 2 and
                         all(type(v) is int and v in (0, 2, 4, 6) for v in row["xy"]),
                         "SHIFT_SET", name, "base shifts must be even integers below eight")
        head = metadata["head_derivation"]
        _fields(head, ("h0_bits", "h1_bits", "synthetic_S1CAP_f32_bits"), "HEAD_DERIVATION", "head_derivation")
        h0 = _bits(head["h0_bits"], "f16", "HEAD_DERIVATION", "h0")
        h1 = _bits(head["h1_bits"], "f16", "HEAD_DERIVATION", "h1")
        cap = _bits(head["synthetic_S1CAP_f32_bits"], "f32", "HEAD_DERIVATION", "synthetic_S1CAP")
        e0, s2 = _exp_f32_bits(h0), _exp_f32_bits(h1)
        e0_value = struct.unpack(">f", struct.pack(">I", e0))[0]
        s1 = int(head["synthetic_S1CAP_f32_bits"], 16) if cap <= e0_value else e0
        _require(s2 != 0x7f800000 and fills["D0-R.s1"] == s1 and fills["D0-R.s2"] == s2,
                 "HEAD_DERIVATION", "head_derivation", "head scalars do not match correctly rounded derivation")
        exposure = _indexed(metadata["exposure_symbols"], _EXPOSURE, "X_SYMBOL_DOMAIN", "exposure_symbols")
        x = {}
        for name, row in exposure.items():
            _fields(row, ("id", "precision", "bits"), "X_SYMBOL_DOMAIN", name)
            _require(type(row["precision"]) is str and row["precision"] == "f32", "X_SYMBOL_DOMAIN", name, "X symbols must be f32")
            x[name] = _bits(row["bits"], "f32", "X_SYMBOL_DOMAIN", name)
            bits = int(row["bits"], 16) & 0x7fffffff
            _require(bits == 0 or bits >= 0x00800000, "X_SYMBOL_DOMAIN", name, "subnormal symbol is outside declared domain")
        _require(x["X_LMAX"] >= 0 and all(x[n] > 0 for n in ("X_KEY", "X_MFL", "X_EFL")) and
                 x["X_CLn"] <= x["X_CL"], "X_SYMBOL_DOMAIN", "exposure_symbols", "declared domain inequality violated")
        _require(Fraction(x["X_KEY"]) / Fraction(x["X_MFL"]) < 65520,
                 "X_SYMBOL_DOMAIN", "exposure_symbols", "exact ratio reaches binary16 overflow threshold")
        x_valid = True
        if bindings is not None:
            _require(type(bindings) is dict and all(type(k) is str for k in bindings) and set(bindings) == set(TENSOR_CATALOG),
                     "PAYLOAD_LENGTH", "bindings", "all 197 exact binding IDs required")
            for name, (shape, precision, _) in TENSOR_CATALOG.items():
                array = bindings[name]
                dtype = {"e4m3": np.dtype("uint8"), "f16": np.dtype("float16"), "f32": np.dtype("float32")}[precision]
                _require(type(array) is np.ndarray and array.shape == shape and array.dtype == dtype,
                         "PAYLOAD_LENGTH", name, "array shape or native dtype does not match descriptor")
                unsigned = {"e4m3": np.uint8, "f16": np.uint16, "f32": np.uint32}[precision]
                _require(bool(np.all(array.view(unsigned) == fills[name])), "PAYLOAD_CONTENT", name,
                         "array does not match the explicitly original synthetic fill recipe")
        return AdmissionResult(ADMITTED, (), bindings is not None, w_max, overflow, x_valid)
    except _Invalid as invalid:
        return AdmissionResult(REJECTED, (invalid.issue,), False, w_max, overflow, x_valid)
