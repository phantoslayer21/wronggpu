# SPDX-License-Identifier: MIT
"""Stage "enc0" (spec 03 input encoder) for the spec 90 runner: weight packing, input conversion, launch plan.

Layouts: clean/FORMATS.md, "Input encoder (enc0)". Kernel: clean/kernels/encoder.hip (`input_encoder`).
"""
import ctypes

import numpy as np

from clean.reference.encoder import N_FEATURES, N_VALUES, check_inputs
from clean.reference.fp8 import fp8_decode
from clean.runner.packing import frag_array, pack_tile_vector, resolve_d_row, rows_layout_l

CODE_OBJECT = "encoder.co"
KERNELS = ("input_encoder",)
THREADS = 256            # kEncThreads
TOKENS_PER_GROUP = 128   # 8 waves x 16 tokens

OFF_BIAS, OFF_SIGMA, OFF_DITHER, OFF_FRAGS = 0, 256, 512, 2560
N_KBLOCKS = 8
BLOB_BYTES = OFF_FRAGS + (N_FEATURES // 16) * N_KBLOCKS * 256   # 10752
PAD = 4 * N_VALUES        # index of the zero column used for padding slots


class EncParams(ctypes.Structure):
    """Must match struct WgEncParams in kernels/encoder.hip."""
    INTS = ("rw", "rh", "fw", "fh", "hw", "hh", "frame", "reset", "depth_inverted", "pad0")
    FLOATS = ("dw", "dh", "kx", "ky", "ikx", "iky", "sx", "sy", "bx", "by", "jx", "jy", "jpx", "jpy",
              "dis_scale", "idw", "idh", "irw", "irh")
    _fields_ = [(n, ctypes.c_int) for n in INTS] + [(n, ctypes.c_float) for n in FLOATS]


assert ctypes.sizeof(EncParams) == 116  # static_assert in kernels/encoder.hip


def kord_encoder():
    """K order of the projection: K[16 kb + 8 h + j] = column of W (or PAD) in slot (h, j) of K block kb.

    Sub-pixel p uses blocks 2p and 2p + 1; its owner lane half is o = p // 2 (see kernels/encoder.hip)."""
    K = np.full(16 * N_KBLOCKS, PAD, dtype=np.int64)
    for p in range(4):
        o = p // 2
        for h in range(2):
            for j in range(8):
                K[16 * (2 * p) + 8 * h + j] = N_VALUES * p + (j if h == o else 8 + j)
                if h == o and j < 4:
                    K[16 * (2 * p + 1) + 8 * h + j] = N_VALUES * p + 16 + j
    return K


def check_tensors(t):
    shapes = {"W": (N_FEATURES, 4 * N_VALUES), "b": (N_FEATURES,), "T": (8, 16, 16), "sigma1": (), "sigma2": ()}
    for k, s in shapes.items():
        if k not in t:
            raise KeyError(f"enc0: missing tensor {k}")
        if tuple(np.shape(t[k])) != s:
            raise ValueError(f"enc0: tensor {k} has shape {np.shape(t[k])}, expected {s}")
    T = np.asarray(t["T"], dtype=np.float64)
    if not np.all((T == np.round(T)) & (T >= 0) & (T <= 127)):
        raise ValueError("enc0: T must hold integers 0..127")


def pack(tensors, d_row="gfx12"):
    """Logical tensors of spec 90 (W, b, T, sigma1, sigma2) -> the enc0 weight blob (FORMATS.md)."""
    check_tensors(tensors)
    drow = resolve_d_row(d_row)
    RL = rows_layout_l(N_FEATURES, drow)
    W = np.asarray(tensors["W"], dtype=np.float64)
    Wx = np.concatenate([W, np.zeros((N_FEATURES, 1))], axis=1)[:, kord_encoder()]   # [64, 128], padding = 0
    frags = frag_array(Wx[RL], np.arange(N_FEATURES), np.arange(16 * N_KBLOCKS))      # [4, 8, 256]
    sig = np.array([np.asarray(tensors["sigma1"]), np.asarray(tensors["sigma2"])], dtype=np.float32).tobytes()
    blob = (pack_tile_vector(tensors["b"], RL, drow) + sig.ljust(OFF_DITHER - OFF_SIGMA, b"\0")
            + np.asarray(tensors["T"], dtype=np.float64).astype(np.uint8).tobytes() + frags.tobytes())
    assert len(blob) == BLOB_BYTES
    return blob


def frame_params(x):
    rw, rh, fw, fh, dw, dh = check_inputs(x)
    hh, hw = np.shape(x["reduced_history"])[:2]
    f = lambda a, i: float(np.asarray(a, dtype=np.float64).reshape(-1)[i])  # noqa: E731
    return EncParams(
        rw=rw, rh=rh, fw=fw, fh=fh, hw=hw, hh=hh, frame=int(x["frame"]), reset=int(bool(x["reset"])),
        depth_inverted=int(bool(x["depth_inverted"])), pad0=0,
        dw=dw, dh=dh, kx=dw / rw, ky=dh / rh, ikx=rw / dw, iky=rh / dh,
        sx=f(x["mv_scale"], 0) * dw / rw, sy=f(x["mv_scale"], 1) * dh / rh, bx=f(x["mv_bias"], 0),
        by=f(x["mv_bias"], 1), jx=f(x["jitter"], 0), jy=f(x["jitter"], 1), jpx=f(x["jitter_prev"], 0),
        jpy=f(x["jitter_prev"], 1), dis_scale=540.0 / rh, idw=1.0 / dw, idh=1.0 / dh, irw=1.0 / rw, irh=1.0 / rh)


def device_inputs(x, color_f16=False, motion_f16=False):
    """Spec 90 input arrays -> byte images in the kernel's formats (FORMATS.md). color_f16 / motion_f16: colour as
    4 x FP16, motion vectors as 2 x FP16 (build variants "colorh", "motionh", "in16" only; values rounded to FP16)."""
    rh, rw = np.shape(x["color"])[:2]
    color = np.zeros((rh, rw, 4), dtype=np.float32)
    color[..., :3] = np.asarray(x["color"], dtype=np.float32)
    if color_f16:
        color = color.astype(np.float16)
    hist = np.clip(np.rint(np.asarray(x["reduced_history"], dtype=np.float64) * 1023.0), 0, 1023).astype(np.uint32)
    hist = hist[..., 0] | (hist[..., 1] << 10) | (hist[..., 2] << 20)
    lat = np.clip(np.rint(np.asarray(x["latent"], dtype=np.float64) * 127.0), -127, 127).astype(np.int8)
    mh = np.asarray(x["motion_history"], dtype=np.float32).astype(np.float16)
    return {
        "color": color.tobytes(),
        "depth": np.ascontiguousarray(x["depth"], dtype=np.float32).tobytes(),
        "motion": np.ascontiguousarray(x["motion"], dtype=np.float16 if motion_f16 else np.float32).tobytes(),
        "hist": np.ascontiguousarray(hist).tobytes(),
        "latent": np.ascontiguousarray(lat).tobytes(),
        "mhist": np.ascontiguousarray(mh).tobytes(),
    }


def _decode_f16x2(raw, shape):
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)


def _decode_fp8(raw, shape):
    return fp8_decode(np.frombuffer(raw, dtype=np.uint8)).reshape(shape)


def prepare(dev, kernel, weights, x, color_f16=False, motion_f16=False):
    """Allocate and upload everything; returns a plan for stage_runner (bufs, sizes, launches, outputs)."""
    if not isinstance(x, dict):
        raise TypeError("enc0: x must be the dict of spec 90")
    weights = bytes(weights)
    if len(weights) != BLOB_BYTES:
        raise ValueError(f"enc0: weight blob has {len(weights)} bytes, expected {BLOB_BYTES}")
    prm = frame_params(x)
    rw, rh, fw, fh = prm.rw, prm.rh, prm.fw, prm.fh
    if rw < 4 or rh < 2 or prm.hw < 2 or prm.hh < 2:
        # the kernel reads 4-column depth windows and 2 x 2 texel blocks (kernels/encoder.hip, wg_mem.h)
        raise ValueError(f"enc0: render size {(rw, rh)} or reduced history {(prm.hw, prm.hh)} too small "
                         "(need rw >= 4, rh >= 2, D/2 >= 2 x 2)")
    data = device_inputs(x, color_f16, motion_f16)
    bufs, sizes = {}, {}
    plan = {"bufs": bufs, "sizes": sizes, "launches": [], "decoders": {}}

    def alloc(name, nbytes, payload=None):
        bufs[name] = dev.malloc(nbytes)
        sizes[name] = nbytes
        if payload is not None:
            dev.upload(bufs[name], payload)

    try:
        alloc("w", len(weights), weights)
        # exposure pair [e, e_prev] as given (the kernel replaces 0 by 1); in the runtime it is GPU-written state
        alloc("expo", 8, np.array([x["exposure"], x["exposure_prev"]], dtype=np.float32).tobytes())
        for k, v in data.items():
            alloc(k, len(v), v)
        ntok = (fh // 2) * (fw // 2)
        alloc("features", ntok * N_FEATURES)
        for k in ("m_near", "m_far", "m_center"):
            alloc(k, rw * rh * 4)
        P = ctypes.c_void_p
        args = [P(bufs[k]) for k in ("color", "depth", "motion", "hist", "latent", "mhist", "features", "m_near",
                                     "m_far", "m_center", "expo", "w")] + [prm]
        grid = -(-ntok // TOKENS_PER_GROUP)
        plan["launches"].append((kernel, grid, THREADS, args))
        plan["outputs"] = [("features", (fh // 2, fw // 2, N_FEATURES))] + \
                          [(k, (rh, rw, 2)) for k in ("m_near", "m_far", "m_center")]
        plan["decoders"] = {"features": _decode_fp8, "m_near": _decode_f16x2, "m_far": _decode_f16x2,
                            "m_center": _decode_f16x2}
    except Exception:
        for ptr in bufs.values():
            dev.free(ptr)
        raise
    return plan


def prepare_variant(dev, weights, x, variant):
    """Stage "enc0@<variant>" (perf_variants.py): the same launch with the kernel of that build variant."""
    from clean.runner import perf_variants as PV
    color_f16, motion_f16 = PV.input_f16("enc0", variant)
    return prepare(dev, f"{PV.code_object('enc0', variant)}:{KERNELS[0]}", weights, x, color_f16, motion_f16)
