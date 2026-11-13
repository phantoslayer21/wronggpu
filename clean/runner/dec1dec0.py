# SPDX-License-Identifier: MIT
"""Stage "dec1dec0" (performance milestone, prototype): the Swin stage dec1 (spec 02) followed by dec0 (spec 05), for
timing and checking the fused kernel `swin_block_c64_add_dec0[_nc]` (kernels/dec1_dec0.hip) against the separate one.

    w = r.pack_weights("dec1dec0", {"dec1": dec1_tensors, "dec0": dec0_tensors})
    out = r.run("dec1dec0", w, x)          # expand -> swin_block_c64_add[_nc] -> coef_decoder (the map Z in memory)
    out = r.run("dec1dec0@fused", w, x)    # expand -> swin_block_c64_add_dec0[_nc] (Z stays on chip)
    out = r.run("dec1dec0@fused_t", w, x)  # same, built with ordinary A, B, C stores (perf_variants.FUSION)
    x = {"y": [H/2, W/2, 96] dec2 output, "skip": [H, W, 64], "offset": (ox, oy)} + the dec0 inputs of spec 90
        (motion, latent_prev, jitter, jitter_prev, reset, feature_size, recon_size; no "features"), H x W = F/2.
    out = {"A", "B", "C", "latent"} as for dec0.

Both use the default Swin launch geometry (variants.DEFAULT_GEOMETRY: 256 lanes, one window per wave); the fused
kernel requires it.
"""
import ctypes

import numpy as np

from clean.reference.fp8 import fp8_encode
from clean.reference.swin import STAGES
from clean.runner import dec0
from clean.runner import variants as V
from clean.runner.packing import pack_stage, stage_sections

FUSED_CODE_OBJECT = "dec1_dec0.co"
FUSED_KERNEL = "swin_block_c64_add_dec0"
ALIGN = 256


def _pad(b):
    return b + bytes(-len(b) % ALIGN)


def pack(tensors, d_row="gfx12"):
    """{"dec1": spec 90 tensors of dec1, "dec0": of dec0} -> [u32 offset of the dec0 blob, padding][dec1 blob][dec0]."""
    b1 = pack_stage("dec1", tensors["dec1"], d_row)
    b0 = dec0.pack(tensors["dec0"], d_row)
    head = ALIGN
    off0 = head + len(_pad(b1))
    return _pad(int(off0).to_bytes(4, "little")) + _pad(b1) + b0


def prepare(dev, weights, x, fused=False, code_object=FUSED_CODE_OBJECT):
    weights = bytes(weights)
    off0 = int.from_bytes(weights[:4], "little")
    w1, w0 = weights[ALIGN:off0], weights[off0:]
    cfg = STAGES["dec1"]
    C, CL = cfg["C"], cfg["C_low"]
    y = np.asarray(x["y"], np.float32)
    skip = np.asarray(x["skip"], np.float32)
    H, W = skip.shape[:2]
    if skip.shape != (H, W, C) or y.shape != (H // 2, W // 2, CL) or H % 4 or W % 4:
        raise ValueError(f"dec1dec0: y {y.shape} / skip {skip.shape} do not match [H/2, W/2, {CL}] / [H, W, {C}]")
    ox, oy = (int(v) for v in x["offset"])
    xd = dict(x, features=np.zeros((H, W, 64), np.float32))
    prm = dec0.frame_params(xd)
    if (prm.fh // 2, prm.fw // 2) != (H, W):
        raise ValueError(f"dec1dec0: feature_size {(prm.fw, prm.fh)} does not match the map {W} x {H}")
    rw, rh = prm.rw, prm.rh
    geo = V.DEFAULT_GEOMETRY
    _, tail_off = stage_sections("dec1")
    safe = (int.from_bytes(w1[:4], "little") & 1) == 1
    bufs, sizes = {}, {}
    plan = {"bufs": bufs, "sizes": sizes, "launches": []}

    def alloc(name, nbytes, payload=None):
        bufs[name] = dev.malloc(nbytes)
        sizes[name] = nbytes
        if payload is not None:
            dev.upload(bufs[name], payload)

    try:
        alloc("w1", len(w1), w1)
        alloc("w0", len(w0), w0)
        alloc("y", y.size, fp8_encode(y).tobytes())
        alloc("skip", skip.size, fp8_encode(skip).tobytes())
        alloc("e", H * W * C)
        for k, v in dec0.device_inputs(xd).items():
            if k != "features":
                alloc(k, len(v), v)
        for k, n in (("A", 4), ("B", 4), ("C", 8), ("latent", 4)):
            alloc(k, rw * rh * n)
        P, I = ctypes.c_void_p, ctypes.c_int
        gemm, nt = geo["gemm_threads"], geo["nt"]
        Hl, Wl = H // 2, W // 2
        plan["launches"].append(("expand_c96", -(-(Hl * Wl) // (gemm // 32 * 16 * nt)), gemm,
                                 [P(bufs["y"]), P(bufs["e"]), P(bufs["w1"] + tail_off), I(Hl), I(Wl)]))
        nwin = ((W + ox + 3) // 4) * ((H + oy + 3) // 4)
        threads, wpw = geo["threads"]["swin_block_c64_add"], geo["wpw"]["swin_block_c64_add"]
        grid = -(-nwin // ((threads // 32) * wpw))
        dec_args = [P(bufs["motion"]), P(bufs["lprev"]), P(bufs["A"]), P(bufs["B"]), P(bufs["C"]),
                    P(bufs["latent"]), P(bufs["w0"]), prm]
        if fused:
            if threads != 256 or wpw != 1:
                raise ValueError("dec1dec0@fused needs 256-lane workgroups with one window per wave")
            plan["launches"].append((f"{code_object}:{FUSED_KERNEL}{'_nc' if safe else ''}", grid, 256,
                                     [P(bufs["e"]), P(bufs["skip"]), P(bufs["w1"]), I(H), I(W), I(ox), I(oy)]
                                     + dec_args))
        else:
            alloc("z", H * W * C)
            plan["launches"].append(("swin_block_c64_add" + ("_nc" if safe else ""), grid, threads,
                                     [P(bufs["e"]), P(bufs["skip"]), P(bufs["z"]), P(bufs["w1"]), I(H), I(W), I(ox),
                                      I(oy)]))
            plan["launches"].append(("coef_decoder", -(-(H * W) // dec0.TOKENS_PER_GROUP), dec0.THREADS,
                                     [P(bufs["z"]), P(bufs["motion"]), P(bufs["lprev"]), P(bufs["A"]), P(bufs["B"]),
                                      P(bufs["C"]), P(bufs["latent"]), P(bufs["w0"]), prm]))
        plan["outputs"] = [("A", (rh, rw, 3)), ("B", (rh, rw, 3)), ("C", (rh, rw, 3)), ("latent", (rh, rw, 4))]
        plan["decoders"] = {"A": dec0._unorm3, "B": dec0._unorm3, "C": dec0._fp16x3, "latent": dec0._snorm4}
    except Exception:
        for ptr in bufs.values():
            dev.free(ptr)
        raise
    return plan
