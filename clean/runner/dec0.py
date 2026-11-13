# SPDX-License-Identifier: MIT
"""Stage "dec0" (spec 05 coefficient decoder) for the spec 90 runner: weight packing, input conversion, launch plan.

Layouts: clean/FORMATS.md, "Coefficient decoder (dec0)". Kernel: clean/kernels/decoder.hip (`coef_decoder`).
"""
import ctypes

import numpy as np

from clean.reference import decoder as R
from clean.reference.fp8 import DECODE, fp8_encode
from clean.runner.packing import frag_array, pack_tile_vector, resolve_d_row

CODE_OBJECT = "decoder.co"
KERNELS = ("coef_decoder",)
THREADS = 256            # kDecThreads
TOKENS_PER_GROUP = 128   # 8 waves x 16 tokens

OFF_BIAS, OFF_SCAL, OFF_TABLES, OFF_FRAGS = 0, 320, 512, 5632
N_MT, N_KB = 5, 4
BLOB_BYTES = OFF_FRAGS + N_MT * N_KB * 256    # 10752
PAD = R.N_OUT                                   # index of the zero row used for padding


class DecParams(ctypes.Structure):
    """Must match struct WgDecParams in kernels/decoder.hip."""
    _fields_ = [(n, ctypes.c_int) for n in ("rw", "rh", "fw", "fh", "reset", "pad0")] + \
               [(n, ctypes.c_float) for n in ("dw", "dh", "kx", "ky", "jx", "jy", "pjx", "pjy")]


assert ctypes.sizeof(DecParams) == 56  # static_assert in kernels/decoder.hip


def rows_decoder(drow):
    """Output rows: accumulator slot (tile mt, lane half h, element v) holds output 34 h + 8 mt + v (PAD past 33)."""
    Rw = np.full(16 * N_MT, PAD, dtype=np.int64)
    for mt in range(N_MT):
        for h in range(2):
            for v in range(8):
                i = 8 * mt + v
                if i < 2 * R.N_PER_PIXEL:
                    Rw[16 * mt + drow[h][v]] = 2 * R.N_PER_PIXEL * h + i
    return Rw


def kord_decoder():
    """K slot (h, j) of block kb holds channel 32 (kb // 2) + 16 h + 8 (kb % 2) + j (blocks 0-1: channels 0..31)."""
    K = np.empty(16 * N_KB, dtype=np.int64)
    for kb in range(N_KB):
        for h in range(2):
            for j in range(8):
                K[16 * kb + 8 * h + j] = 32 * (kb // 2) + 16 * h + 8 * (kb % 2) + j
    return K


def tables(S, T):
    """The five 256-entry tables of FORMATS.md ("Coefficient decoder"), as a [5, 256] uint32 array (float32 tables
    by their bit patterns), indexed by an FP8 code (the last one by an 8-bit latent code):
    0 A:   bits 0-7 code of A = unorm8(RN16(sigma(a))), bits 16-31 FP16 C0(a) = RN16(RN32(sx(a)^2));
    1 Sx:  f32 sx(a) = RN32(S RN16(sigma(a)));
    2 Rho: f32 rho(a);
    3 ZG:  bits 0-15 FP16 z(a) = RN16(0.5 + 0.5 th(a / 2)), bits 16-31 FP16 g(a) = th(a);
    4 Sn:  f32 snorm value of the code."""
    a = np.nan_to_num(DECODE, nan=0.0).astype(np.float32)    # NaN codes never occur (q8 saturates)
    g16 = R.sigmoid16(a)
    sx = (np.float32(S) * g16).astype(np.float32)
    f16bits = lambda v: np.asarray(v, np.float32).astype(np.float16).view(np.uint16).astype(np.uint32)  # noqa: E731
    t_a = R.unorm8_code(g16).astype(np.uint32) | (f16bits(R.rn16(sx * sx)) << 16)
    z = R.rn16(np.float32(0.5) + np.float32(0.5) * R.tanh16(a / 2))
    t_zg = f16bits(z) | (f16bits(R.tanh16(a)) << 16)
    t_sn = R.snorm8(np.arange(256, dtype=np.int64).astype(np.uint8).view(np.int8))
    f32bits = lambda v: np.asarray(v, np.float32).view(np.uint32)  # noqa: E731
    return np.stack([t_a, f32bits(sx), f32bits(R.rho32(a, T)), t_zg, f32bits(t_sn)]).astype(np.uint32)


def check_tensors(t):
    shapes = {"W": (R.N_OUT, R.N_IN), "b": (R.N_OUT,), "S": (), "T": ()}
    for k, s in shapes.items():
        if k not in t:
            raise KeyError(f"dec0: missing tensor {k}")
        if tuple(np.shape(t[k])) != s:
            raise ValueError(f"dec0: tensor {k} has shape {np.shape(t[k])}, expected {s}")


def tables_v1(S, T):
    """The tables of the first kernel version (kernels/baseline/decoder_v1.hip; timing comparisons only): float32
    RN16(sigma(a)), th(a), th(a / 2), rho(a), snorm value."""
    a = np.nan_to_num(DECODE, nan=0.0).astype(np.float32)
    return np.stack([R.sigmoid16(a), R.tanh16(a), R.tanh16(a / 2), R.rho32(a, T),
                     R.snorm8(np.arange(256, dtype=np.int64).astype(np.uint8).view(np.int8))]).astype(np.float32)


def pack_v1(tensors, d_row="gfx12"):
    """The weight blob of the first kernel version (same layout, version-1 tables); stage "dec0@v1" only."""
    blob = bytearray(pack(tensors, d_row))
    tab = tables_v1(float(np.asarray(tensors["S"])), float(np.asarray(tensors["T"]))).tobytes()
    blob[OFF_TABLES:OFF_TABLES + len(tab)] = tab
    return bytes(blob)


def pack(tensors, d_row="gfx12"):
    """Logical tensors of spec 90 (W, b, S, T) -> the dec0 weight blob (FORMATS.md)."""
    check_tensors(tensors)
    drow = resolve_d_row(d_row)
    Rw = rows_decoder(drow)
    W = np.concatenate([np.asarray(tensors["W"], np.float64), np.zeros((1, R.N_IN))], axis=0)
    b = np.concatenate([np.asarray(tensors["b"], np.float32), np.zeros(1, np.float32)])
    frags = frag_array(W[Rw][:, kord_decoder()], np.arange(16 * N_MT), np.arange(16 * N_KB))   # [5, 4, 256]
    S, T = float(np.asarray(tensors["S"])), float(np.asarray(tensors["T"]))
    scal = np.array([S, T], dtype=np.float32).tobytes().ljust(OFF_TABLES - OFF_SCAL, b"\0")
    blob = pack_tile_vector(b, Rw, drow) + scal + tables(S, T).tobytes() + frags.tobytes()
    assert len(blob) == BLOB_BYTES
    return blob


def frame_params(x):
    rw, rh, fw, fh, dw, dh = R.check_inputs(x)
    f = lambda a, i: float(np.asarray(a, dtype=np.float32).reshape(-1)[i])  # noqa: E731
    return DecParams(rw=rw, rh=rh, fw=fw, fh=fh, reset=int(bool(x["reset"])), pad0=0, dw=dw, dh=dh,
                     kx=float(np.float32(dw / rw)), ky=float(np.float32(dh / rh)),
                     jx=f(x["jitter"], 0), jy=f(x["jitter"], 1), pjx=f(x["jitter_prev"], 0),
                     pjy=f(x["jitter_prev"], 1))


def device_inputs(x):
    lat = np.clip(np.rint(np.asarray(x["latent_prev"], np.float64) * 127.0), -127, 127).astype(np.int8)
    return {
        "features": fp8_encode(np.asarray(x["features"], np.float32)).tobytes(),
        "motion": np.ascontiguousarray(np.asarray(x["motion"], np.float32).astype(np.float16)).tobytes(),
        "lprev": np.ascontiguousarray(lat).tobytes(),
    }


def _unorm3(raw, shape):
    return (np.frombuffer(raw, np.uint8).reshape(shape[0], shape[1], 4)[..., :3] / 255.0).astype(np.float32)


def _fp16x3(raw, shape):
    return np.frombuffer(raw, np.float16).reshape(shape[0], shape[1], 4)[..., :3].astype(np.float32)


def _snorm4(raw, shape):
    return (np.frombuffer(raw, np.int8).reshape(shape) / 127.0).astype(np.float32)


def prepare(dev, kernel, weights, x):
    if not isinstance(x, dict):
        raise TypeError("dec0: x must be the dict of spec 90")
    weights = bytes(weights)
    if len(weights) != BLOB_BYTES:
        raise ValueError(f"dec0: weight blob has {len(weights)} bytes, expected {BLOB_BYTES}")
    prm = frame_params(x)
    rw, rh, fw, fh = prm.rw, prm.rh, prm.fw, prm.fh
    bufs, sizes = {}, {}
    plan = {"bufs": bufs, "sizes": sizes, "launches": []}

    def alloc(name, nbytes, payload=None):
        bufs[name] = dev.malloc(nbytes)
        sizes[name] = nbytes
        if payload is not None:
            dev.upload(bufs[name], payload)

    try:
        alloc("w", len(weights), weights)
        for k, v in device_inputs(x).items():
            alloc(k, len(v), v)
        npix = rw * rh
        for k, n in (("A", 4), ("B", 4), ("C", 8), ("latent", 4)):
            alloc(k, npix * n)
        ntok = (fh // 2) * (fw // 2)
        P = ctypes.c_void_p
        args = [P(bufs[k]) for k in ("features", "motion", "lprev", "A", "B", "C", "latent", "w")] + [prm]
        plan["launches"].append((kernel, -(-ntok // TOKENS_PER_GROUP), THREADS, args))
        plan["outputs"] = [("A", (rh, rw, 3)), ("B", (rh, rw, 3)), ("C", (rh, rw, 3)), ("latent", (rh, rw, 4))]
        plan["decoders"] = {"A": _unorm3, "B": _unorm3, "C": _fp16x3, "latent": _snorm4}
    except Exception:
        for ptr in bufs.values():
            dev.free(ptr)
        raise
    return plan


def _pack12_finish(out):
    """Images of variant "pack12" -> the usual outputs: A8 [rh, rw, 8] bytes, C4 [rh, rw, 4] bytes."""
    a8, c4 = out.pop("A8"), out.pop("C4")
    out["A"] = (a8[..., 0:3] / 255.0).astype(np.float32)
    out["B"] = (a8[..., 3:6] / 255.0).astype(np.float32)
    c = np.ascontiguousarray(c4).view(np.float16)
    c2 = np.ascontiguousarray(a8[..., 6:8]).view(np.float16)
    out["C"] = np.concatenate([c, c2], axis=-1).astype(np.float32)
    return {k: out[k] for k in ("A", "B", "C", "latent")}


def prepare_variant(dev, weights, x, variant):
    """Stage "dec0@<variant>" (perf_variants.py): the same launch with the kernel of that build variant. "pack12":
    images A8 (8 bytes per pixel: A0 A1 A2 B0 B1 B2, C2 as FP16) and C4 (4 bytes: C0 C1 as FP16), decoded into the
    usual A, B, C grids afterwards."""
    from clean.runner import perf_variants as PV
    plan = prepare(dev, f"{PV.code_object('dec0', variant)}:{KERNELS[0]}", weights, x)
    if not PV.VARIANTS["dec0"][variant][1].get("WG_DEC_PACK12"):
        return plan
    bufs, sizes = plan["bufs"], plan["sizes"]
    rh, rw = plan["outputs"][0][1][:2]
    for k in ("A", "B", "C"):
        dev.free(bufs.pop(k))
        sizes.pop(k)
    try:
        for k, n in (("A8", 8), ("C4", 4)):
            bufs[k] = dev.malloc(n * rh * rw)
            sizes[k] = n * rh * rw
    except Exception:
        for ptr in bufs.values():
            dev.free(ptr)
        raise
    args = plan["launches"][0][3]
    args[3], args[4], args[5] = ctypes.c_void_p(bufs["A8"]), ctypes.c_void_p(0), ctypes.c_void_p(bufs["C4"])
    raw = lambda n: lambda r, shape: np.frombuffer(r, np.uint8).reshape(shape[0], shape[1], n)  # noqa: E731
    plan["outputs"] = [("A8", (rh, rw)), ("C4", (rh, rw)), ("latent", (rh, rw, 4))]
    plan["decoders"] = dict(plan["decoders"], A8=raw(8), C4=raw(4))
    plan["finish"] = _pack12_finish
    return plan
