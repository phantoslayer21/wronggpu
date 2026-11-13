# SPDX-License-Identifier: MIT
"""Stage "exposure" (spec 07 auto-exposure) for the spec 90 runner: input conversion and launch plan (no weights).

Layouts: clean/FORMATS.md, "Auto-exposure (exposure)". Kernels: clean/kernels/exposure.hip.
"""
import ctypes

import numpy as np

from clean.reference.exposure import sample_grid

CODE_OBJECT = "exposure.co"
KERNELS = ("exposure_partial", "exposure_final")
THREADS = 256
ROWS_PER_GROUP = 4   # kRowsPerGroup


class ExpParams(ctypes.Structure):
    """Must match struct WgExpParams in kernels/exposure.hip."""
    _fields_ = [(n, ctypes.c_int) for n in ("rw", "rh", "nx", "ny", "nparts", "pad0")] + \
               [(n, ctypes.c_float) for n in ("inv_pi", "K", "inv_n")]


assert ctypes.sizeof(ExpParams) == 36  # static_assert in kernels/exposure.hip


def pack(tensors, d_row="gfx12"):
    """The stage has no weights (spec 90); an empty blob is accepted by run()."""
    return b""


def frame_params(x):
    color = np.asarray(x["color"])
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"exposure: color must be [rh, rw, 3], got {color.shape}")
    rh, rw = color.shape[:2]
    nx, ny, _, _ = sample_grid(rw, rh)
    pi = float(x.get("pre_exposure", 1.0))
    alpha = float(x.get("alpha", 0.18))
    nparts = -(-ny // ROWS_PER_GROUP)
    with np.errstate(divide="ignore"):
        inv_pi = float(np.float32(1.0) / np.float32(pi))
    return ExpParams(rw=rw, rh=rh, nx=nx, ny=ny, nparts=nparts, pad0=0, inv_pi=inv_pi,
                     K=alpha / (1.0 - alpha), inv_n=1.0 / (nx * ny))


def prepare(dev, kernels, weights, x):
    if not isinstance(x, dict):
        raise TypeError("exposure: x must be the dict of spec 90")
    prm = frame_params(x)
    rh, rw = prm.rh, prm.rw
    color = np.zeros((rh, rw, 4), dtype=np.float32)
    color[..., :3] = np.asarray(x["color"], dtype=np.float32)
    bufs, sizes = {}, {}
    plan = {"bufs": bufs, "sizes": sizes, "launches": []}
    try:
        for name, nbytes in (("color", color.nbytes), ("partials", 4 * prm.nparts), ("e", 4)):
            bufs[name] = dev.malloc(nbytes)
            sizes[name] = nbytes
        dev.upload(bufs["color"], color.tobytes())
        P = ctypes.c_void_p
        k_partial, k_final = kernels
        plan["launches"] += [(k_partial, prm.nparts, THREADS, [P(bufs["color"]), P(bufs["partials"]), prm]),
                             (k_final, 1, THREADS, [P(bufs["partials"]), P(bufs["e"]), prm])]
        plan["outputs"] = [("e", ())]
        plan["decoders"] = {"e": lambda raw, shape: float(np.frombuffer(raw, np.float32)[0])}
    except Exception:
        for ptr in bufs.values():
            dev.free(ptr)
        raise
    return plan


def prepare_variant(dev, weights, x, variant):
    """Stage "exposure@<variant>" (perf_variants.py): the same launches with the kernels of that build variant."""
    from clean.runner import perf_variants as PV
    co = PV.code_object("exposure", variant)
    return prepare(dev, tuple(f"{co}:{k}" for k in KERNELS), weights, x)
