# SPDX-License-Identifier: MIT
"""Loading a converted preset-F model file (the converter's `model.wgcm`, container version 2).

Authority: spec 56 (logical tensors 56.4), spec 57 (output groups 57.7.1, symbol table 57.7.2, digests and the
read-back rule 57.7.3) and clean/converter/OUTPUT-FORMAT.md (WrongGPU's own container; 57.7.3 leaves the container
to the runtime, as 08 section 7.2). Python standard library only; NumPy is used only by the conversion helpers
that hand arrays to the NumPy reference stages, and only when it is installed.

Read-back rule (57.7.3, WrongGPU's choice): the file keeps per group the digest the converter checked and the key
it used. The loader recomputes the keyed SHA-256 of every group present, `constants_b2` included, and refuses the
file on any mismatch. With the map, every group's stored digest must also be the map's. `constants_b2` present
selects build variant B2 of page 52.7.2, absent selects B1 (also for a file written before the group existed).

The loader never logs or prints tensor values; its messages name groups, tensors and offsets only.
"""
from dataclasses import dataclass, field
import hashlib
import json
import struct

CONTRACT_ID = "wronggpu.rr_f.model_file.v1"
ADMITTED = "ADMITTED_PLAYER_MODEL"
REJECTED = "REJECTED_MODEL_FILE"

DTYPES = {0: "float32", 1: "uint8", 2: "int32", 3: "float16", 4: "uint16", 5: "uint32", 6: "e4m3"}
DTYPE_SIZE = {"float32": 4, "uint8": 1, "int32": 4, "float16": 2, "uint16": 2, "uint32": 4, "e4m3": 1}

SINGLES = ("IN0", "E1", "E2", "E3", "E4", "E5")
LEVELS = (4, 3, 2, 1, 0)
PARTS = SINGLES + tuple("D%d-%s" % (k, h) for k in LEVELS for h in ("F", "R"))
# 56.2.1: (C, H, Cn, Kc) per stage part.
GEOMETRY = {
    "IN0": (32, 2, None, None), "E1": (64, 2, 96, None), "E2": (96, 4, 128, None), "E3": (128, 4, 160, None),
    "E4": (160, 8, 160, None), "E5": (160, 8, None, None),
    "D4": (160, 8, None, 160), "D3": (128, 4, None, 160), "D2": (96, 4, None, 128), "D1": (64, 2, None, 96),
    "D0": (32, 2, None, 64),
}
WINDOW_SYMBOLS = ("A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1", "ACT_C2", "ACT_C3")
F32_SYMBOLS = ("C_AMAX", "C_FCLIP", "C_GCL", "C_GCLn", "C_HDEC", "C_LWrb", "C_LWg", "C_MDL", "C_MDLn", "C_SV1",
               "C_SV2", "C_SV3", "Gm", "Gd", "K2n", "K2x2n", "K3", "K4", "K5", "K7", "K7n", "K8", "K10", "K11", "K12",
               "K13", "T_EFL", "TBW", "X_CL", "X_CLn", "X_EFL", "X_KEY", "X_LMAX", "X_LWrb", "X_LWg", "X_MFL", "X_NRM",
               "S1CAP")
F16_SYMBOLS = WINDOW_SYMBOLS + ("WMAX", "Gh", "Gs", "RW_A", "RW_B", "RW_C", "RW_D", "RW_0") + \
    tuple("%s.%s" % (k, p) for p in PARTS for k in ("ra", "m")) + tuple("f.D%d" % k for k in LEVELS)


def _part_geometry(part):
    return GEOMETRY[part.split("-")[0]]


def expected_inventory():
    """{group: {tensor: (shape, dtype)}} of pages 56.4 and 57.7, written from the page text (not from a map)."""
    inv = {}
    for part in PARTS:
        C, H, Cn, Kc = _part_geometry(part)
        t = {
            "Wq": ((H, C, 32), "e4m3"), "Wv": ((H, C, 32), "e4m3"), "Tb": ((H, 12, 16, 16), "float16"),
            "Wo": ((H, 32, C), "e4m3"), "bo": ((C,), "float16"), "W1": ((C, 4 * C), "e4m3"),
            "b1": ((4 * C,), "float16"), "W2": ((4 * C, C), "e4m3"), "b2": ((C,), "float16"),
        }
        if part in ("E1", "E2", "E3", "E4"):  # the combined axis n*C + c split into [4][C] (57.7.1)
            t["Wm"] = ((4, C, Cn), "e4m3")
            t["bm"] = ((Cn,), "float16")
        if part == "E1":
            t["Wa"] = ((4, 32, 64), "e4m3")
            t["ba"] = ((64,), "float16")
        if part.endswith("-F"):
            t["Wup"] = ((Kc, 4, C), "e4m3")
            t["bup"] = ((4, C), "float16")
        if part in SINGLES:
            t["Wh"] = ((C, 3), "e4m3")
            t["bh"] = ((3,), "float16")
        elif part.endswith("-R") and part != "D0-R":
            t["Wh"] = ((C, 4), "e4m3")
            t["bh"] = ((4,), "float16")
        if part in SINGLES or part.endswith("-R"):
            t["g"] = ((1,), "float16")
        if part == "IN0":
            for b in (0, 1):
                t["FE.W1_%d" % b] = ((64, 32), "e4m3")
                t["FE.b1_%d" % b] = ((32,), "float16")
                t["FE.W2_%d" % b] = ((32, 32), "e4m3")
            t["FE.b2"] = ((32,), "float16")
            t["D"] = ((2048,), "uint16")
        if part == "D0-R":
            for n in ("s1", "s2", "K1"):
                t[n] = ((1,), "float32")
        inv[part] = t
    c = {n: ((1,), "float16") for n in F16_SYMBOLS}
    c.update({n: ((1,), "float32") for n in F32_SYMBOLS})
    c["TR"] = ((513, 4), "float32")
    c["SH"] = ((2, 11, 2), "uint32")
    inv["constants"] = c
    inv["constants_b2"] = {"C_SV4": ((1,), "float32"), "C_SV4n": ((1,), "float32")}
    return inv


OPTIONAL_GROUPS = ("constants_b2",)


class ModelFileError(ValueError):
    def __init__(self, code, path, message):
        super().__init__("%s: %s: %s" % (code, path, message))
        self.code, self.path, self.message = code, path, message


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple
    data: bytes

    def values(self):
        """Element values as Python numbers (float16/float32 as floats, e4m3 as code integers)."""
        n = len(self.data) // DTYPE_SIZE[self.dtype]
        fmt = {"float32": "f", "float16": "e", "uint16": "H", "uint32": "I", "int32": "i", "uint8": "B",
               "e4m3": "B"}[self.dtype]
        return list(struct.unpack("<%d%s" % (n, fmt), self.data))

    def bits(self):
        n = len(self.data) // DTYPE_SIZE[self.dtype]
        fmt = {1: "B", 2: "H", 4: "I"}[DTYPE_SIZE[self.dtype]]
        return list(struct.unpack("<%d%s" % (n, fmt), self.data))


@dataclass(frozen=True)
class Group:
    name: str
    digest: bytes
    tensors: dict  # name -> Tensor, in file order


@dataclass(frozen=True)
class ModelFile:
    container_version: int
    key: bytes
    map_sha256: bytes
    groups: dict  # name -> Group, in file order


def read_model_file(path_or_bytes):
    """Parses a container and checks it: whole-file hash, index bounds, 64-byte alignment, sizes, and the keyed
    digest of every group with the stored key (57.7.3). Raises ModelFileError."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        try:
            with open(path_or_bytes, "rb") as f:
                data = f.read()
        except OSError as exc:
            raise ModelFileError("UNREADABLE", str(path_or_bytes), exc.strerror or "cannot read") from None
    if len(data) < 88 + 32 or data[:8] != b"WGCMODEL":
        raise ModelFileError("NOT_A_MODEL_FILE", "header", "not a converter model file")
    body, trailer = data[:-32], data[-32:]
    if hashlib.sha256(body).digest() != trailer:
        raise ModelFileError("FILE_DAMAGED", "trailer", "the whole-file hash does not match")
    version, ngroups = struct.unpack_from("<II", data, 8)
    if version != 2:
        raise ModelFileError("CONTAINER_VERSION", "header", "container version %d is not a preset-F file" % version)
    key, map_sha = data[16:48], data[48:80]
    (index_size,) = struct.unpack_from("<Q", data, 80)
    end = 88 + index_size
    if end > len(body):
        raise ModelFileError("FILE_DAMAGED", "index", "index runs past the data")
    pos = 88

    def need(n, where):
        if pos + n > end:
            raise ModelFileError("FILE_DAMAGED", where, "index entry runs past the index")

    def name_at(where):
        nonlocal pos
        need(2, where)
        (n,) = struct.unpack_from("<H", data, pos)
        need(2 + n, where)
        s = data[pos + 2: pos + 2 + n].decode("utf-8", "strict")
        pos += 2 + n
        return s

    groups = {}
    for _ in range(ngroups):
        gname = name_at("group")
        need(36, gname)
        digest = data[pos: pos + 32]
        (nt,) = struct.unpack_from("<I", data, pos + 32)
        pos += 36
        tensors = {}
        h = hashlib.sha256(key)
        for _ in range(nt):
            tname = name_at(gname)
            where = gname + "/" + tname
            need(2, where)
            code, rank = data[pos], data[pos + 1]
            pos += 2
            if code not in DTYPES:
                raise ModelFileError("FILE_DAMAGED", where, "unknown dtype code %d" % code)
            need(4 * rank + 16, where)
            shape = struct.unpack_from("<%dI" % rank, data, pos)
            pos += 4 * rank
            off, size = struct.unpack_from("<QQ", data, pos)
            pos += 16
            count = 1
            for d in shape:
                count *= d
            if off % 64 or off < end or off + size > len(body) or size != count * DTYPE_SIZE[DTYPES[code]]:
                raise ModelFileError("FILE_DAMAGED", where, "bad data range")
            if tname in tensors:
                raise ModelFileError("FILE_DAMAGED", where, "duplicate tensor")
            blob = data[off: off + size]
            h.update(blob)
            tensors[tname] = Tensor(tname, DTYPES[code], tuple(shape), blob)
        if h.digest() != digest:
            raise ModelFileError("DIGEST_MISMATCH", gname, "the group's data does not give its stored digest")
        if gname in groups:
            raise ModelFileError("FILE_DAMAGED", gname, "duplicate group")
        groups[gname] = Group(gname, digest, tensors)
    if pos != end:
        raise ModelFileError("FILE_DAMAGED", "index", "index size does not match its entries")
    return ModelFile(version, key, map_sha, groups)


@dataclass(frozen=True)
class AdmittedModel:
    """A loaded, checked preset-F model. Tensor access is by group and name (page 56.4 meanings)."""
    file: ModelFile
    variant: str  # "B1" or "B2" (page 52.7.2): constants_b2 absent or present
    attention_sum_may_overflow: bool

    def tensor(self, group, name):
        return self.file.groups[group].tensors[name]

    def symbol(self, name):
        """A scalar of the symbol table (57.7.2) as a Python float (or int for SH elements via tensor())."""
        for g in ("constants", "constants_b2"):
            grp = self.file.groups.get(g)
            if grp and name in grp.tensors:
                return grp.tensors[name].values()[0]
        raise KeyError(name)

    def shifts(self, parity):
        """Base shifts (sx, sy) of the eleven stages for frame parity 0 or 1 (SH, 51.6 / 53.7)."""
        sh = self.tensor("constants", "SH").values()
        names = SINGLES + tuple("D%d" % k for k in LEVELS)
        return {n: (sh[parity * 22 + 2 * i], sh[parity * 22 + 2 * i + 1]) for i, n in enumerate(names)}

    def stage_constants(self, part):
        """ra, m (all parts), f (fusion halves) and g (parts with a hidden-state update) as floats (51.2)."""
        c = {"ra": self.symbol("ra." + part), "m": self.symbol("m." + part)}
        if part.endswith("-F"):
            c["f"] = self.symbol("f." + part[:2])
        if part in SINGLES or part.endswith("-R"):
            c["g"] = self.tensor(part, "g").values()[0]
        return c

    def window_constants(self):
        return {n: self.symbol(n) for n in WINDOW_SYMBOLS}

    def stage_weights_numpy(self, part):
        """The logical weights of one part as the NumPy reference stages take them (clean/rr_f/reference):
        e4m3 as uint8 codes, f16 as float64 values, with the split axes of 57.7.1 joined again (same bytes)."""
        import numpy as np  # only here: the loader itself needs no NumPy
        out = {}
        for name, t in self.file.groups[part].tensors.items():
            if t.dtype == "e4m3":
                a = np.frombuffer(t.data, dtype=np.uint8).reshape(t.shape)
            elif t.dtype == "float16":
                a = np.frombuffer(t.data, dtype="<f2").astype(np.float64).reshape(t.shape)
            elif t.dtype == "float32":
                a = np.frombuffer(t.data, dtype="<f4").astype(np.float64).reshape(t.shape)
            else:
                a = np.frombuffer(t.data, dtype="<u2" if t.dtype == "uint16" else "<u4").reshape(t.shape)
            if name in ("Wm", "Wa"):          # [4][C][Cn] -> [4C][Cn]
                a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])
            elif name == "Wup":               # [Kc][4][C] -> [Kc][4C]
                a = a.reshape(a.shape[0], a.shape[1] * a.shape[2])
            elif name == "bup":               # [4][C] -> [4C]
                a = a.reshape(-1)
            out[name] = a
        return out


@dataclass(frozen=True)
class FileAdmission:
    decision: str
    issues: tuple = ()
    model: object = None
    contract_id: str = CONTRACT_ID

    @property
    def player_ready(self):
        return self.decision == ADMITTED


def _check_inventory(mf):
    inv = expected_inventory()
    for gname, tensors in inv.items():
        if gname not in mf.groups:
            if gname in OPTIONAL_GROUPS:
                continue
            raise ModelFileError("MISSING_GROUP", gname, "group missing")
        got = mf.groups[gname].tensors
        if set(got) != set(tensors):
            missing = sorted(set(tensors) - set(got))
            extra = sorted(set(got) - set(tensors))
            raise ModelFileError("TENSOR_SET", gname, "missing %s, unexpected %s" % (missing, extra))
        for tname, (shape, dtype) in tensors.items():
            t = got[tname]
            if t.shape != shape or t.dtype != dtype:
                raise ModelFileError("TENSOR_SET", gname + "/" + tname,
                                     "shape %s %s, expected %s %s" % (t.shape, t.dtype, shape, dtype))
    for gname in mf.groups:
        if gname not in inv:
            raise ModelFileError("UNKNOWN_GROUP", gname, "group not of preset F")


def _check_against_map(mf, map_obj):
    if map_obj.get("format") != "wronggpu-converter-map" or map_obj.get("version") != 5:
        raise ModelFileError("MAP", "map", "not a version-5 converter map")
    for g in map_obj["groups"]:
        name = g["group"]
        if name not in mf.groups:
            if g.get("requires_variant"):
                continue  # produced only for its variant (57.7.2: constants_b2)
            raise ModelFileError("MISSING_GROUP", name, "group of the map missing")
        if mf.groups[name].digest.hex() != g["digest"].lower():
            raise ModelFileError("DIGEST_MISMATCH", name, "stored digest is not the map's")
        if [t["name"] for t in g["tensors"]] != list(mf.groups[name].tensors):
            raise ModelFileError("TENSOR_SET", name, "tensor order differs from the map")
    names = {g["group"] for g in map_obj["groups"]}
    for gname in mf.groups:
        if gname not in names:
            raise ModelFileError("UNKNOWN_GROUP", gname, "group not in the map")


def _attention_record(mf):
    """Recompute 57.6.4's Boolean from the stored WMAX (64 * WMAX >= 65520); no constant is logged."""
    w = mf.groups["constants"].tensors["WMAX"].values()[0]
    return 64.0 * w >= 65520.0


def admit_model_file(path_or_bytes, map_path=None, map_obj=None):
    """Load a converted preset-F model file for the runtime. Returns FileAdmission with an AdmittedModel, or a
    rejection naming the first problem (never a value)."""
    try:
        mf = read_model_file(path_or_bytes)
        _check_inventory(mf)
        if map_path is not None or map_obj is not None:
            if map_obj is None:
                with open(map_path, "rb") as f:
                    map_obj = json.load(f)
            _check_against_map(mf, map_obj)
        variant = "B2" if "constants_b2" in mf.groups else "B1"
        return FileAdmission(ADMITTED, (), AdmittedModel(mf, variant, _attention_record(mf)))
    except ModelFileError as e:
        return FileAdmission(REJECTED, ((e.code, e.path, e.message),), None)
    except (UnicodeDecodeError, struct.error, KeyError, json.JSONDecodeError) as e:
        return FileAdmission(REJECTED, (("FILE_DAMAGED", "file", type(e).__name__),), None)


def catalog_bindings(model):
    """The 197 arrays of the synthetic known-subset catalog (admission.TENSOR_CATALOG names and shapes) from an
    admitted model file, for the existing NumPy body harnesses: e4m3 uint8 codes, f16 as float16, f32 as float32."""
    import numpy as np
    from .admission import TENSOR_CATALOG
    out = {}
    for key, (shape, precision, _axes) in TENSOR_CATALOG.items():
        stage, name = key.split(".", 1)
        group = stage if name != "g" or stage in SINGLES else stage + "-R"
        t = model.tensor(group, name)
        dt = {"e4m3": np.uint8, "f16": "<f2", "f32": "<f4"}[precision]
        out[key] = np.frombuffer(t.data, dtype=dt).reshape(shape).copy()
    return out
