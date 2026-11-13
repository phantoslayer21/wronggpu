# SPDX-License-Identifier: MIT
"""Model file of WrongGPU's RR runtime (`wronggpu_rr_model.bin`, read by clean/rr_f/runtime/src/rr_model.cpp).

The converter (page 57) emits the logical tensors of page 56.4 and the symbol table of 57.7.2; "how the runtime stores
the groups (its model-file container) is the runtime's own choice" (57.7.3). This container holds the logical model
already arranged in the kernels' buffer layouts (clean/rr_f/gpu/layout.py, head/dx12/run_dx12.py, gpu/input_stage.py;
56.4.5: any re-arrangement must not change a value) plus every scalar the kernels take as an argument, so the runtime
only uploads one device image and reads a name -> offset / bits table.

Sources of a logical model (all produce the same dict, `logical`):
  --synthetic SEED   original synthetic model (tests/synth.py, tests/synth_input.py fixtures; own invented values)
  --admission        the admitted synthetic known subset of model/admission.py (synthetic-base.json, constant
                     fills), completed with synthetic front-end / output / resolve data the contract does not cover
  --converter DIR    the converter reference's output folder (model.npz + manifest.json, clean/rr_f/converter)
  --wgcm FILE        the converter's real output, model.wgcm (clean/converter, container version 2; checked by
                     model/model_file.py admit_model_file). The runtime reads model.wgcm itself (src/rr_pack.cpp packs it
                     into the same device image); here it gives the tests the logical model and the Python packing

File layout (little-endian):
  0   8  magic b"WGRRMOD1"
  8   4  container version (1)
  12  4  flags: bit 0 constants_b2 present (page 52.7.2 build B2), bit 1 synthetic model, bit 2 admission subset
  16  4  number of blob entries NB
  20  4  number of scalar entries NS
  24  8  device image size (bytes, multiple of 256)
  32  32 SHA-256 over the logical model (group order of 57.7.1, tensors in name order, row-major bytes) - provenance
  64  NB x 64: name (48 bytes, NUL padded ASCII), u64 offset in the device image, u64 size
  ..  NS x 52: name (48 bytes), u32 bits
  ..  padding to 256, the device image
  end-32  SHA-256 of every byte before it
"""
import argparse
import hashlib
import json
import os
import pickle
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from clean.rr_f.gpu import layout                                  # noqa: E402
from clean.rr_f.reference import input_stage as IS                 # noqa: E402
from clean.rr_f.reference.fmt import to_f16_bits                   # noqa: E402
from clean.rr_f.reference.scalar import WindowConsts, sig_tables   # noqa: E402
from clean.rr_f.reference.stages import stage_info                 # noqa: E402

MAGIC = b"WGRRMOD1"
VERSION = 1
STAGES = ("IN0", "E1", "E2", "E3", "E4", "E5", "D4-F", "D4-R", "D3-F", "D3-R", "D2-F", "D2-R", "D1-F", "D1-R",
          "D0-F", "D0-R")
BLOCK_STAGES = STAGES[:-1]                      # rr_block.hlsl; D0-R is the head kernel (page 55)
SH_ROWS = ("IN0", "E1", "E2", "E3", "E4", "E5", "D4", "D3", "D2", "D1", "D0")
WINDOW = ("A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1", "ACT_C2", "ACT_C3")
OUTPUT_K = ("K2n", "K2x2n", "K3", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13")
EXPOSURE = ("X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn", "X_NRM")
RESOLVE = ("RW_A", "RW_B", "RW_C", "RW_D", "RW_0")
RESOLVE_F32 = ("T_EFL", "TBW")   # 54b.7 binary32 symbols of the extended resolve
INPUT_SYMS = ("K4", "K5", "K7", "K7n", "K8", "K10", "K12", "C_HDEC", "C_FCLIP", "C_AMAX", "C_GCL", "C_GCLn", "C_LWrb",
              "C_LWg", "C_MDL", "C_MDLn", "Gm", "Gd", "Gh", "Gs", "C_SV1", "C_SV2", "C_SV3", "C_SV4", "C_SV4n")
F16_SYMBOLS = set(WINDOW) | {"Gh", "Gs"} | set(RESOLVE)


def f16bits(v):
    return int(to_f16_bits(np.asarray([v], np.float64))[0])


def f32bits(v):
    return int(np.asarray(v, np.float32).reshape(()).view(np.uint32))


def h2pair(a, b):
    return f16bits(a) | (f16bits(b) << 16)


# ---------------------------------------------------------------------------------------------------------------
# logical models

def _f16(a):
    return np.asarray(a, np.float64).astype(np.float16)


def synthetic_logical(seed=5100):
    """An original synthetic logical model: random weights in the 56.4 shapes, plausible invented symbols."""
    from clean.rr_f.tests import synth, synth_input
    from clean.rr_f.tests.test_output import symbols as output_symbols
    from clean.rr_f.gpu.exposure import symbols as exposure_symbols
    rng = np.random.default_rng(seed)
    wc, _ = synth.window_consts(rng, "A")
    groups = {}
    consts = {}
    for name in STAGES:
        base = "IN0" if name == "D0-R" else name           # D0-R: the IN0-shaped block without hidden projection
        W = synth.stage_weights(rng, base)
        if name == "D0-R":
            W = {k: v for k, v in W.items() if k not in ("Wh", "bh")}
        g = {k: (np.asarray(v, np.uint8) if v.dtype == np.uint8 else _f16(v)) for k, v in W.items()}
        sc = synth.stage_consts(rng)
        consts[f"ra.{name}"] = _f16([sc["ra"]])
        consts[f"m.{name}"] = _f16([sc["m"]])
        if name.endswith("-F"):
            consts[f"f.{name[:2]}"] = _f16([sc["f"]])
        if name in ("IN0", "E1", "E2", "E3", "E4", "E5") or name.endswith("-R"):
            g["g"] = _f16([sc["g"]])
        groups[name] = g
    fe = synth_input.fe_weights(rng)
    for k, v in fe.items():
        groups["IN0"]["FE." + k] = np.asarray(v, np.uint8) if v.dtype == np.uint8 else _f16(v)
    groups["IN0"]["D"] = rng.integers(0, 65536, 2048).astype(np.uint16)
    groups["D0-R"]["s1"] = np.array([1.25], np.float32)
    groups["D0-R"]["s2"] = np.array([0.75], np.float32)
    groups["D0-R"]["K1"] = np.array([1 / 16], np.float32)
    for n in WINDOW:
        consts[n] = _f16([getattr(wc, n)])
    S = synth_input.symbols(rng)
    for n in IS.SYMBOLS_F32:
        consts[n] = np.array([S.values[n]], np.float32)
    for n in IS.SYMBOLS_F16:
        consts[n] = _f16([S.values[n]])
    # output stage (54): the shared K4..K12 are the input stage's (one symbol each, 57.7.2)
    table = np.tile(np.array([.125, .75, .125, 0], np.float32), (513, 1))
    table[:, 3] = np.linspace(0, .5, 513, dtype=np.float32)
    o = output_symbols(TR=table, K3=.75, K1=1 / 16, K2n=-1.5, K2x2n=-3.0, K11=.25, K13=2 ** -20,
                       K4=S.K4, K5=S.K5, K7=S.K7, K8=S.K8, K10=S.K10, K12=S.K12)
    for n in OUTPUT_K:
        consts[n] = np.array([getattr(o, n)], np.float32)
    consts["TR"] = np.asarray(o.TR, np.float32)
    x = exposure_symbols()
    for n in EXPOSURE:
        consts[n] = np.array([getattr(x, n)], np.float32)
    for n, v in zip(RESOLVE, (-0.0625, -0.0625, 0.1875, 0.3125, 0.625)):
        consts[n] = _f16([v])
    consts["T_EFL"] = np.array([2.0 ** -10], np.float32)
    consts["TBW"] = np.array([0.3125], np.float32)
    sh = rng.choice(np.array([0, 2, 4, 6], np.uint32), size=(2, 11, 2)).astype(np.uint32)
    consts["SH"] = sh
    return dict(groups=groups, constants=consts, constants_b2=None, kind="synthetic", seed=seed)


def admission_logical():
    """The admitted synthetic known subset (model/admission.py): every one of its 197 tensors and its symbols, as
    admitted; the parts its contract does not cover (front end, dither, input / output / resolve symbols, SH is
    covered) come from the synthetic model with seed 5100. Raises if admission rejects the base."""
    from clean.rr_f.model import admission
    base_path = os.path.join(ROOT, "clean", "rr_f", "model", "contracts", "synthetic-base.json")
    with open(base_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = {}
    for row in meta["tensors"]:
        shape, precision, _ = admission.TENSOR_CATALOG[row["id"]]
        bits = int(row["synthetic_fill_bits"], 16)
        unsigned = {"e4m3": np.uint8, "f16": np.uint16, "f32": np.uint32}[precision]
        native = {"e4m3": np.uint8, "f16": np.float16, "f32": np.float32}[precision]
        arrays[row["id"]] = np.full(shape, bits, unsigned).view(native)
    res = admission.validate_metadata(meta, bindings=arrays)
    if res.decision != admission.ADMITTED:
        raise SystemExit(f"admission rejected the synthetic base: {res.issues}")
    m = synthetic_logical(5100)
    for tid, a in arrays.items():
        stage, name = tid.split(".", 1)
        if stage in ("D4", "D3", "D2", "D1", "D0"):        # g of a decoder level: used by its recurrent half
            m["groups"][stage + "-R"][name] = a
        else:
            m["groups"][stage][name] = a
    for row in meta["symbols"]:
        sid, bits = row["id"], int(row["bits"], 16)
        v = np.array([bits], np.uint16).view(np.float16)
        if "." in sid:
            stage, k = sid.split(".")
            key = f"{k}.{stage}" if k in ("ra", "m") else f"f.{stage}"
        else:
            key = sid
        m["constants"][key] = v
    for row in meta["exposure_symbols"]:
        m["constants"][row["id"]] = np.array([int(row["bits"], 16)], np.uint32).view(np.float32)
    sh = np.zeros((2, 11, 2), np.uint32)
    for blk in meta["shifts"]:
        for r in blk["rows"]:
            sh[blk["parity"], SH_ROWS.index(r["stage"])] = r["xy"]
    m["constants"]["SH"] = sh
    m["kind"] = "admission"
    m["admission"] = res.as_dict()
    return m


def converter_logical(folder):
    """The converter reference's output (clean/rr_f/converter write_output): model.npz + manifest.json."""
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    z = np.load(os.path.join(folder, "model.npz"))
    groups, consts, b2 = {s: {} for s in STAGES}, {}, None
    for key in z.files:
        group, name = key.split("/", 1) if "/" in key else key.split(".", 1)
        meta = manifest.get("tensors", {}).get(key, {})
        a = z[key]
        if meta.get("dtype") == "float16" and a.dtype != np.float16:
            a = a.view(np.float16)
        if group == "constants":
            consts[name] = a
        elif group == "constants_b2":
            b2 = b2 or {}
            b2[name] = a
        else:
            groups[group][name] = a
    return dict(groups=groups, constants=consts, constants_b2=b2, kind="converter", manifest=manifest)


def wgcm_logical(path, map_path=None):
    """The converter's model.wgcm (model/model_file.py admit_model_file): the logical dict of this module (e4m3 as uint8
    codes, binary16 as float16, the split axes of 57.7.1 joined again: the same bytes)."""
    from clean.rr_f.model import model_file as MF
    adm = MF.admit_model_file(path, map_path=map_path)
    if not adm.player_ready:
        raise SystemExit(f"model file refused: {adm.issues}")
    mf = adm.model.file
    dt = {"e4m3": np.uint8, "uint8": np.uint8, "float16": "<f2", "float32": "<f4", "uint16": "<u2", "uint32": "<u4",
          "int32": "<i4"}

    def arr(t):
        a = np.frombuffer(t.data, dtype=dt[t.dtype]).reshape(t.shape).copy()
        if t.name in ("Wm", "Wa"):
            a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])
        elif t.name == "Wup":
            a = a.reshape(a.shape[0], a.shape[1] * a.shape[2])
        elif t.name == "bup":
            a = a.reshape(-1)
        return a
    groups = {g: {n: arr(t) for n, t in mf.groups[g].tensors.items()} for g in STAGES}
    consts = {n: arr(t) for n, t in mf.groups["constants"].tensors.items()}
    b2 = ({n: arr(t) for n, t in mf.groups["constants_b2"].tensors.items()} if "constants_b2" in mf.groups else None)
    return dict(groups=groups, constants=consts, constants_b2=b2, kind="wgcm", path=str(path),
                variant=adm.model.variant)


def logical_digest(m):
    h = hashlib.sha256()
    for gname in STAGES + ("constants", "constants_b2"):
        g = m["groups"].get(gname) if gname in STAGES else m.get(gname)
        if not g:
            continue
        h.update(gname.encode())
        for name in sorted(g):
            a = np.ascontiguousarray(g[name])
            h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.digest()


# ---------------------------------------------------------------------------------------------------------------
# device image

class Image:
    def __init__(self):
        self.parts, self.size, self.entries = [], 0, {}

    def add(self, name, data):
        data = np.ascontiguousarray(data).view(np.uint8).ravel()
        off = (self.size + 255) // 256 * 256
        self.parts.append((off, data))
        self.size = off + max(len(data), 4)
        self.entries[name] = (off, len(data))
        return off

    def bytes(self):
        n = (self.size + 255) // 256 * 256
        img = np.zeros(n, np.uint8)
        for off, d in self.parts:
            img[off:off + len(d)] = d
        return img.tobytes()


def _head_tables(s2):
    from clean.rr_f.head.dx12 import run_dx12 as head
    return head.tables(s2)


def pack(m):
    """logical model -> (container bytes, entries, scalars)."""
    from clean.rr_f.head.dx12 import run_dx12 as head
    G, K = m["groups"], m["constants"]
    img, sc = Image(), {}
    sc_f16 = lambda v: f16bits(np.asarray(v, np.float64).reshape(-1)[0])
    wc = WindowConsts(**{n: float(np.asarray(K[n], np.float64).reshape(-1)[0]) for n in WINDOW})
    from clean.rr_f.gpu.stage import wc_dwords
    for i, d in enumerate(wc_dwords(wc)):
        sc[f"wc{i}"] = d
    img.add("sig", layout.f16_bytes(sig_tables()[0]))
    for name in BLOCK_STAGES:
        info = stage_info(name)
        W = {k: (np.asarray(v, np.uint8) if np.asarray(v).dtype == np.uint8 else np.asarray(v, np.float64))
             for k, v in G[name].items() if not k.startswith("FE.") and k not in ("D", "g")}
        P = layout.pack_weights(W, info["C"], info["H"])
        for k, a in P.items():
            img.add(f"{name}.{k}", a)
        lv = name[:2]
        f = float(np.asarray(K[f"f.{lv}"]).reshape(-1)[0]) if name.endswith("-F") else 0.0
        g = float(np.asarray(G[name]["g"]).reshape(-1)[0]) if "g" in G[name] else 0.0
        sc[f"{name}.ra_m"] = h2pair(float(K[f"ra.{name}"][0]), float(K[f"m.{name}"][0]))
        sc[f"{name}.f_g"] = h2pair(f, g)
        # exact body step Z: host checks (gpu/stage.py y_bounds / z_params; rr_pack.cpp z_host); zok 0: step X
        from clean.rr_f.gpu import stage as GS
        y_ok, y_thr = GS.y_bounds(W, l1=True)
        zt0, zt1 = GS.z_params(W)
        sc[f"{name}.zok"] = 1 if y_ok else 0
        sc[f"{name}.zthr"] = int(np.float32(min(y_thr, 3.0e38)).view(np.uint32)) if y_ok else 0
        sc[f"{name}.zt0"] = int(zt0)
        sc[f"{name}.zt1"] = int(zt1)
    # head (page 55): D0-R block weights, biases, score-bias tiles, tables
    H = G["D0-R"]
    Wh = {k: (np.asarray(v, np.uint8) if np.asarray(v).dtype == np.uint8 else np.asarray(v, np.float64))
          for k, v in H.items() if k not in ("g", "s1", "s2", "K1")}
    img.add("head.wf", np.frombuffer(head.weight_buffer(Wh), np.uint8))
    bb, tb = head.bias_buffers(Wh)
    img.add("head.bias", np.frombuffer(bb, np.uint8))
    img.add("head.tb", np.frombuffer(tb, np.uint8))
    s1, s2 = np.float32(H["s1"][0]), np.float32(H["s2"][0])
    tab, rho = _head_tables(s2)
    img.add("head.tab", np.frombuffer(tab, np.uint8))
    img.add("head.rho", np.frombuffer(rho, np.uint8))
    for a, n in head.KNAMES.items():
        if n in ("ra", "m"):
            sc[f"head.{a}"] = sc_f16(K[f"{n}.D0-R"])
        else:
            sc[f"head.{a}"] = sc_f16(K[n])
    sc["head.k_g"] = sc_f16(H["g"])
    sc["head.k_s1"] = f32bits(s1)
    sc["K1"] = f32bits(np.float32(H["K1"][0]))
    # input stage (52): front-end MLP fragments (gpu/input_stage.py), dither, read tables
    IN0 = G["IN0"]
    w1 = np.stack([layout.frags(np.asarray(IN0[f"FE.W1_{br}"], np.uint8).T) for br in (0, 1)])
    w2 = np.stack([layout.frags(np.asarray(IN0[f"FE.W2_{br}"], np.uint8).T) for br in (0, 1)])
    img.add("in.W1", w1)
    img.add("in.W2", w2)
    img.add("in.B1", np.concatenate([layout.f32(np.asarray(IN0["FE.b1_0"], np.float64)),
                                     layout.f32(np.asarray(IN0["FE.b1_1"], np.float64))]))
    img.add("in.B2", layout.f32(np.asarray(IN0["FE.b2"], np.float64)))
    img.add("in.D", np.ascontiguousarray(IN0["D"], np.uint16))
    un10 = IS.decode_unorm(np.arange(1024), 10).astype(np.float32)
    sn8 = IS.decode_snorm(np.arange(256).astype(np.uint8).view(np.int8).astype(np.int64), 8).astype(np.float32)
    img.add("in.UN10", un10)
    img.add("in.SN8", sn8)
    b2 = m.get("constants_b2")
    for n in INPUT_SYMS:
        src = b2 if n in ("C_SV4", "C_SV4n") else K
        if src is None or n not in src:
            sc[f"sym.{n}"] = 0
        else:
            sc[f"sym.{n}"] = f32bits(np.float32(np.asarray(src[n], np.float64).reshape(-1)[0]))
    # output stage (54)
    for n in OUTPUT_K:
        sc[f"out.{n}"] = f32bits(np.float32(np.asarray(K[n]).reshape(-1)[0]))
    img.add("out.TR", np.asarray(K["TR"], np.float32))
    for n in EXPOSURE:
        sc[f"exp.{n}"] = f32bits(np.float32(np.asarray(K[n]).reshape(-1)[0]))
    for n in RESOLVE:
        sc[f"res.{n}"] = sc_f16(K[n])
    for n in RESOLVE_F32:
        sc[f"res.{n}"] = f32bits(np.float32(np.asarray(K[n]).reshape(-1)[0]))
    sc["sym.C_HDEC"] = f32bits(np.float32(np.asarray(K["C_HDEC"]).reshape(-1)[0]))
    SH = np.asarray(K["SH"], np.uint32).reshape(2, 11, 2)
    for p in range(2):
        for s in range(11):
            sc[f"SH.{p}.{s}"] = int(SH[p, s, 0]) | (int(SH[p, s, 1]) << 8)
    flags = (1 if b2 else 0) | (2 if m["kind"] in ("synthetic", "admission") else 0) | (4 if m["kind"] == "admission" else 0)
    dev = img.bytes()
    names = sorted(img.entries)
    out = bytearray()
    out += MAGIC + struct.pack("<IIII", VERSION, flags, len(names), len(sc)) + struct.pack("<Q", len(dev))
    out += logical_digest(m)
    for n in names:
        nb = n.encode("ascii")
        assert len(nb) < 48, n
        off, size = img.entries[n]
        out += nb.ljust(48, b"\0") + struct.pack("<QQ", off, size)
    for n in sorted(sc):
        nb = n.encode("ascii")
        assert len(nb) < 48, n
        out += nb.ljust(48, b"\0") + struct.pack("<I", sc[n] & 0xffffffff)
    out += b"\0" * ((-len(out)) % 256)
    out += dev
    out += hashlib.sha256(bytes(out)).digest()
    return bytes(out), img.entries, sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", type=int, metavar="SEED")
    src.add_argument("--admission", action="store_true")
    src.add_argument("--converter", metavar="DIR")
    src.add_argument("--wgcm", metavar="FILE")
    ap.add_argument("--out", help="the runtime's own container (not needed with --wgcm: the runtime reads model.wgcm)")
    ap.add_argument("--logical-out", help="also pickle the logical model (tests: the CPU reference reads it)")
    a = ap.parse_args(argv)
    if a.synthetic is not None:
        m = synthetic_logical(a.synthetic)
    elif a.admission:
        m = admission_logical()
    elif a.wgcm:
        m = wgcm_logical(a.wgcm)
    else:
        m = converter_logical(a.converter)
    data, entries, sc = pack(m)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "wb") as fh:
            fh.write(data)
    if a.logical_out:
        with open(a.logical_out, "wb") as fh:
            pickle.dump(m, fh)
    print(f"{a.out or '(no container written)'}: {len(data)} bytes, {len(entries)} blobs, {len(sc)} scalars, kind {m['kind']}, "
          f"sha256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
