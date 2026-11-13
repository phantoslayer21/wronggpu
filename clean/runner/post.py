# SPDX-License-Identifier: MIT
"""Stages "post" (spec 06 output reconstruction) and "filter" (spec 06 output filter) for the spec 90 runner: weight
packing, input conversion and launch plans; and the fused reconstruction + filter kernels (post_fused*), which are not
a spec 90 stage: run_fused / time_fused below drive them through a Runner.

Layouts: clean/FORMATS.md, "Output reconstruction and output filter". Kernels: clean/kernels/post.hip
(`post_recon`, `post_filter`, `post_fused*`).
"""
import ctypes

import numpy as np

from clean.reference.post import exposure_or_one, recon_size_for

CODE_OBJECT = "post.co"
KERNEL_RECON = "post_recon"
KERNEL_FILTER = "post_filter"
# Filter build variants in the same code object: kernel name -> output row pairs per lane (S). post_filter is the
# default used by run/time; post_bench.py times all of them.
FILTER_VARIANTS = {"post_filter": 8, "post_filter_s4": 4, "post_filter_s16": 16,
                   "post_filter_c1": 8, "post_filter_c1_s4": 4, "post_filter_c1_s16": 16,
                   "post_filter_quality9": 4}
# Output columns per workgroup: 128 (4 columns x 32 lanes, all channels per lane), 40 for the one-channel-per-lane
# _c1 kernels (10 column groups x 3 channels per wave).
FILTER_TILE_XS = {k: (40 if "_c1" in k else 128) for k in FILTER_VARIANTS}
FILTER_TILE_XS["post_filter_quality9"] = 40
# Filter kernels with an RGBA16F output (8 bytes per pixel, alpha 1; the FP16 rounding of the f32 output): name -> S.
FILTER_O16_VARIANTS = {"post_filter_c1_s4_o16": 4, "post_filter_quality9_o16": 4}
FILTER_TILE_XS.update({k: 40 for k in FILTER_O16_VARIANTS})
# Output mode D = O (runtime option recon = output; FORMATS.md): post_recon2_r_out reconstructs at D = O and writes the
# final output (RGBA16F) instead of the display image: bit-identical to post_recon2_r + post_decode_o16 (the identity
# output filter as a kernel). prepare_post(..., KERNEL_RECON_OUT) returns "history", "reduced" and "out".
KERNEL_RECON_OUT = "post_recon2_r_out"
KERNEL_DECODE = "post_decode_o16"
# Banded two-kernel path (prepare_banded): post_recon2_r_band and post_filter_c1_s<S>_band launched alternately over
# horizontal bands, so that a band of the display image is still in the caches when the filter reads it. Same
# arithmetic as post_recon2_r + post_filter_c1_s<S> (bit-identical). Filter kernel name -> S.
BAND_RECON = "post_recon2_r_band"
# banded reconstruction kernel -> the unbanded kernel of the same layouts (8-byte / planar history, pack12 input)
BAND_RECONS = {BAND_RECON: "post_recon2_r", "post_recon2_r_hp_band": "post_recon2_r_hp",
               "post_recon2_r_band_p12": "post_recon2_r_p12", "post_recon2_r_hp_band_p12": "post_recon2_r_hp_p12"}
BAND_FILTERS = {"post_filter_c1_s4_band": 4, "post_filter_c1_s2_band": 2}
# Reconstruction kernels with the 6-byte full history (RGB16F packed, as post_fused; same values as post_recon).
RECON_HIST6 = ("post_recon2_r_h6",)
# Reconstruction kernels with the planar 6-byte full history (RG16F plane, then B16F plane; same values).
RECON_PLANAR = ("post_recon2_r_hp", "post_recon2_r_hp_band", "post_recon2_r_hp_p12", "post_recon2_r_hp_band_p12")
# Reconstruction kernels reading dec0's "pack12" coefficients (pack12_inputs); they take an int tile0 after expo.
RECON_PACK12 = ("post_recon2_r_p12", "post_recon2_r_hp_p12", "post_recon2_r_band_p12", "post_recon2_r_hp_band_p12")
# Fused reconstruction + filter: kernel name -> (6-byte full history, FP16 x 4 output). post_fused is the default.
FUSED_VARIANTS = {"post_fused": (True, False), "post_fused_o16": (True, True), "post_fused_h8": (False, False),
                  "post_fused_h8_o16": (False, True)}
KERNEL_FUSED = "post_fused"
# post_fused3*: fused reconstruction + output filter scheduled for latency (staging prefetch, batched history loads),
# history 6 bytes (RGB16F, exact) or 4 bytes (compact, approximation), output f32 or RGBA16F (_o16). Arguments and grid
# as post_fused. name -> (history bytes per pixel, FP16 output, True if bit-identical to post_recon + post_filter (the
# _o16 output: its FP16 rounding) or what is approximated).
FUSED3_VARIANTS = {
    "post_fused3_h6": (6, False, True),
    "post_fused3_h6_o16": (6, True, True),
    "post_fused3_fast_h6_o16": (6, True, "post_recon4_fastest arithmetic (packed FP16 gather, bilinear history, skip4)"),
    "post_fused3_h4_o16": (4, True, "compact 4-byte history (E as 11/11/10-bit fixed point over [0, 3])"),
    "post_fused3_fast_h4_o16": (4, True, "post_recon4_fastest arithmetic + compact 4-byte history"),
}
# Drop-in replacements of post_recon (same arguments, layouts and grid; bit-identical results): kernel name ->
# (3 x 3 gather rows rolled, history taps as spans). Precondition: recon2_supported.
RECON2_VARIANTS = {"post_recon2": (False, True), "post_recon2_r": (True, True), "post_recon2_ns": (False, False),
                   "post_recon2_r_ns": (True, False), "post_recon2_r_h6": (True, True),   # _h6: RECON_HIST6
                   "post_recon2_r_hp": (True, True),                                       # _hp: RECON_PLANAR
                   "post_recon2_r_p12": (True, True), "post_recon2_r_hp_p12": (True, True)}   # _p12: RECON_PACK12
# post_recon3: as post_recon2_r plus batched history loads and the fallback colour from LDS (bit-identical);
# post_recon3_h: plus packed-FP16 history tap weights and a different rounding order in the gather (within spec 09).
RECON3_VARIANTS = {"post_recon3": True, "post_recon3_h": False}   # name -> bit-identical to post_recon
# SPEED APPROXIMATIONS of post_recon (not spec 06-exact; judged by the whole-pipeline rule of spec 09 draft 2). Same
# arguments, layouts and grid; built on post_recon3. name -> what is approximated.
RECON4_VARIANTS = {
    "post_recon4_bil": "history sampled bilinearly (4 taps) instead of the 12-tap Catmull-Rom, both candidates",
    "post_recon4_crbil": "Catmull-Rom for the stronger candidate, bilinear for the other",
    "post_recon4_skip16": "second candidate dropped when its weight is below 1/16 (spec: 1/255), weight to the first",
    "post_recon4_skip4": "second candidate dropped when its weight is below 1/4",
    "post_recon4_f16": "per-sample codes (scores, g, betas) accumulated and history tap weights in packed FP16",
    "post_recon4_f16x": "whole 3x3 gather (exponent, exp, weights, colour, codes) and tap weights in packed FP16",
    "post_recon4_fast": "f16 + crbil + skip16",
    "post_recon4_fastest": "f16x + bil + skip4",
}
# Pre-pass + reconstruction: post_prep (render resolution, one 32-byte record per sample) then the kernel below.
# name -> bit-identical to post_recon (post_recon5_bil: bilinear history, an approximation).
KERNEL_PREP = "post_prep"
PREP_RECORD_BYTES = 32
RECON5_VARIANTS = {"post_recon5": True, "post_recon5_bil": False}
# post_recon6: post_recon5 with a cheaper per-pixel head (FP32 motions in LDS, the m_center bilinear shared by the
# two pixels of a lane, 24-bit address multiplies); same pre-pass and arguments. name -> bit-identical to post_recon;
# the others are SPEED APPROXIMATIONS (whole-pipeline rule): _bil bilinear history, _eb blend in the E domain.
RECON6_VARIANTS = {"post_recon6": True, "post_recon6_bil": False, "post_recon6_eb": False,
                   "post_recon6_eb_bil": False}
PREP_VARIANTS = {**RECON5_VARIANTS, **RECON6_VARIANTS}   # reconstruction kernels that read post_prep records
# Timing-only kernels of runner/post_traffic.py (memory-traffic floor per image layout; outputs are garbage).
TRAFFIC_KERNELS = ("post_traffic_r_h8", "post_traffic_r_h6", "post_traffic_r_h4", "post_traffic_x_h8_o12",
                   "post_traffic_x_h6_o12", "post_traffic_x_h4_o12", "post_traffic_x_h6_o8", "post_traffic_x_h4_o8",
                   "post_traffic_f_o12", "post_traffic_f_o8", "post_traffic_flush")
KERNELS = ((KERNEL_RECON,) + tuple(RECON2_VARIANTS) + tuple(RECON3_VARIANTS) + tuple(RECON4_VARIANTS)
           + (KERNEL_PREP,) + tuple(PREP_VARIANTS) + tuple(FILTER_VARIANTS) + tuple(FILTER_O16_VARIANTS)
           + (KERNEL_RECON_OUT, KERNEL_DECODE) + tuple(BAND_RECONS) + RECON_PACK12[:2]
           + tuple(BAND_FILTERS) + tuple(FUSED_VARIANTS)
           + tuple(FUSED3_VARIANTS) + ("post_filter_quality9_h",)
           + TRAFFIC_KERNELS)

RECON_THREADS, RECON_TILE_X, RECON_TILE_PAIRS = 256, 32, 8          # kPostThreads, kTileX, kTilePairs
FILTER_THREADS, FILTER_TILE_X, FILTER_STRIPS = 64, 128, 2            # kFiltThreads, 4 kFiltColGroups, kFiltStrips
FUSED_THREADS, FUSED_OUT_X, FUSED_OUT_Y = 192, 40, 96                # kFuThreads, kFuOutX, kFuOutY
BLOB_BYTES = 16   # s (f32) + padding


class PostParams(ctypes.Structure):
    """Must match struct WgPostParams in kernels/post.hip."""
    _fields_ = ([(n, ctypes.c_int) for n in ("rw", "rh", "dw", "dh", "reset", "pad0", "pad1", "pad2")]
                + [(n, ctypes.c_float) for n in ("rx", "ry", "jx", "jy", "e", "e_prev", "ratio", "pad3")])


assert ctypes.sizeof(PostParams) == 64  # static_assert in kernels/post.hip


def _cdiv(a, b):
    return -(-a // b)


def pack(tensors):
    """pack_weights("post", {"s": 0-d array}) -> 16-byte blob: s as f32, zero padding."""
    if "s" not in tensors:
        raise KeyError("post: missing tensor s")
    if np.size(tensors["s"]) != 1:
        raise ValueError(f"post: s must be a scalar, got shape {np.shape(tensors['s'])}")
    return np.array([float(np.asarray(tensors["s"]).reshape(-1)[0]), 0, 0, 0], dtype=np.float32).tobytes()


def _scalar(a, i=0):
    return float(np.asarray(a, dtype=np.float64).reshape(-1)[i])


def check_post_inputs(x):
    if not isinstance(x, dict):
        raise TypeError("post: x must be the dict of spec 90")
    rh, rw = np.shape(x["color"])[:2]
    dw, dh = (int(v) for v in x["recon_size"])
    shapes = {"color": (rh, rw, 3), "A": (rh, rw, 3), "B": (rh, rw, 3), "C": (rh, rw, 3), "history": (dh, dw, 3),
              "m_near": (rh, rw, 2), "m_center": (rh, rw, 2), "m_far": (rh, rw, 2)}
    for k, s in shapes.items():
        if tuple(np.shape(x[k])) != s:
            raise ValueError(f"post: {k} has shape {np.shape(x[k])}, expected {s}")
    if rw < 1 or rh < 1 or dw < 1 or dh < 1:
        raise ValueError("post: empty image")
    return rw, rh, dw, dh


def post_params(x):
    rw, rh, dw, dh = check_post_inputs(x)
    e, e_prev = exposure_or_one(x["exposure"]), exposure_or_one(x["exposure_prev"])
    return PostParams(rw=rw, rh=rh, dw=dw, dh=dh, reset=int(bool(x["reset"])), pad0=0, pad1=0, pad2=0,
                      rx=rw / dw, ry=rh / dh, jx=_scalar(x["jitter"], 0), jy=_scalar(x["jitter"], 1), e=e,
                      e_prev=e_prev, ratio=e / e_prev, pad3=0.0)


def display_pitch(dw):
    """Row pitch (FP16 elements) of the display image planes: dw rounded up to a multiple of 64 (display_pitch in
    kernels/post.hip)."""
    return (int(dw) + 63) & ~63


def display_bytes(dw, dh):
    """Size of the display image buffer (three FP16 planes of display_pitch(dw) x dh)."""
    return 3 * display_pitch(dw) * int(dh) * 2


def _unorm8(a):
    """Values on the k/255 grid -> codes (round to nearest, clamp to 0..255)."""
    return np.clip(np.rint(np.asarray(a, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _rgba(a, dtype, fill=0):
    a = np.asarray(a)
    out = np.full(a.shape[:2] + (4,), fill, dtype=dtype)
    out[..., :a.shape[2]] = a.astype(dtype)
    return np.ascontiguousarray(out)


def post_device_inputs(x):
    """Spec 90 input arrays -> byte images in the kernel's formats (FORMATS.md)."""
    f16 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float16)  # noqa: E731 (inputs are FP16-exact)
    return {
        "color": _rgba(np.asarray(x["color"], dtype=np.float32), np.float32).tobytes(),
        "A": _rgba(_unorm8(x["A"]), np.uint8).tobytes(),
        "B": _rgba(_unorm8(x["B"]), np.uint8).tobytes(),
        "C": _rgba(f16(x["C"]), np.float16).tobytes(),
        "hist_prev": _rgba(f16(x["history"]), np.float16).tobytes(),
        "m_near": np.ascontiguousarray(f16(x["m_near"])).tobytes(),
        "m_center": np.ascontiguousarray(f16(x["m_center"])).tobytes(),
        "m_far": np.ascontiguousarray(f16(x["m_far"])).tobytes(),
    }


def pack12_inputs(data):
    """A, B, C byte images of post_device_inputs -> dec0's "pack12" layout (FORMATS.md): "A" = image A8 (codes
    A0 A1 A2 B0 B1 B2, C2 as FP16; 8 bytes per pixel), "C" = image C4 (C0, C1 as FP16; 4 bytes per pixel)."""
    a = np.frombuffer(data["A"], np.uint8).reshape(-1, 4)
    b = np.frombuffer(data["B"], np.uint8).reshape(-1, 4)
    c = np.frombuffer(data["C"], np.uint8).reshape(-1, 8)
    a8 = np.concatenate([a[:, :3], b[:, :3], c[:, 4:6]], 1)
    return dict(data, A=np.ascontiguousarray(a8).tobytes(), C=np.ascontiguousarray(c[:, :4]).tobytes())


def decode_history(raw, shape):
    return np.frombuffer(raw, dtype=np.float16).reshape(shape[:2] + (4,))[..., :3].astype(np.float32)


def history_bytes(dw, dh, hist6):
    """Full-history buffer size: 6 bytes per pixel + 2 bytes of slack (unaligned 8-byte reads), or 8 bytes."""
    return int(dw) * int(dh) * 6 + 8 if hist6 else int(dw) * int(dh) * 8


def encode_history(h, hist6):
    """[dh, dw, 3] FP16-exact values -> the full-history buffer bytes of the given format."""
    h = np.asarray(h, dtype=np.float32).astype(np.float16)
    if hist6:
        return np.ascontiguousarray(h).tobytes() + bytes(8)
    return _rgba(h, np.float16).tobytes()


H4_CODE = (np.float32(2047.0 / 3.0), np.float32(2047.0 / 3.0), np.float32(1023.0 / 3.0))   # kH4Code*


def encode_history4(h):
    """[dh, dw, 3] values -> the compact 4-byte history (h4_pack in kernels/post.hip: codes of clamp(v, 0, 3))."""
    h = np.asarray(h, dtype=np.float32)
    c = [np.rint(np.clip(np.nan_to_num(h[..., i], nan=0.0), 0.0, 3.0).astype(np.float32) * H4_CODE[i]).astype(np.uint32)
         for i in range(3)]
    return np.ascontiguousarray(c[0] | (c[1] << 11) | (c[2] << 22)).astype("<u4").tobytes()


def decode_history4(raw, shape):
    v = np.frombuffer(raw, dtype="<u4")[:shape[0] * shape[1]].reshape(shape[0], shape[1])
    return np.stack([(v & 0x7FF) * np.float32(3.0 / 2047.0), ((v >> 11) & 0x7FF) * np.float32(3.0 / 2047.0),
                     (v >> 22) * np.float32(3.0 / 1023.0)], -1).astype(np.float32)


def decode_history_fmt(raw, shape, hist6):
    if not hist6:
        return decode_history(raw, shape)
    n = shape[0] * shape[1] * 3
    return np.frombuffer(raw, dtype=np.float16)[:n].reshape(shape[0], shape[1], 3).astype(np.float32)


def encode_history_planar(h):
    """[dh, dw, 3] FP16-exact values -> planar 6-byte history: R, G plane (2 halves per pixel), then B plane, + 8."""
    h = np.asarray(h, dtype=np.float32).astype(np.float16)
    return np.ascontiguousarray(h[..., :2]).tobytes() + np.ascontiguousarray(h[..., 2]).tobytes() + bytes(8)


def decode_history_planar(raw, shape):
    n = shape[0] * shape[1]
    v = np.frombuffer(raw, dtype=np.float16)
    rg = v[:2 * n].reshape(shape[0], shape[1], 2)
    b = v[2 * n:3 * n].reshape(shape[0], shape[1], 1)
    return np.concatenate([rg, b], -1).astype(np.float32)


def decode_out_f16(raw, shape):
    return np.frombuffer(raw, dtype=np.float16).reshape(shape[0], shape[1], 4)[..., :3].astype(np.float32)


def decode_planes(raw, shape):
    h, w = shape[:2]
    planes = np.frombuffer(raw, dtype=np.float16).reshape(3, h, display_pitch(w))[:, :, :w]
    return np.ascontiguousarray(planes.transpose(1, 2, 0)).astype(np.float32)


def encode_planes(img):
    """[dh, dw, 3] -> the display buffer bytes (three FP16 planes with row pitch display_pitch(dw), padding 0)."""
    h, w = img.shape[:2]
    planes = np.zeros((3, h, display_pitch(w)), dtype=np.float16)
    planes[:, :, :w] = np.asarray(img, dtype=np.float32).astype(np.float16).transpose(2, 0, 1)
    return planes.tobytes()


def decode_reduced(raw, shape):
    t = np.frombuffer(raw, dtype=np.uint32)[:shape[0] * shape[1]].reshape(shape[:2])  # buffer >= 4 bytes
    codes = np.stack([t & 1023, (t >> 10) & 1023, (t >> 20) & 1023], axis=-1)
    return (codes / 1023.0).astype(np.float32)


def decode_f32(raw, shape):
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


class _Plan:
    def __init__(self, dev):
        self.dev = dev
        self.plan = {"bufs": {}, "sizes": {}, "launches": [], "outputs": [], "decoders": {}}

    def alloc(self, name, nbytes, payload=None):
        self.plan["bufs"][name] = self.dev.malloc(nbytes)
        self.plan["sizes"][name] = nbytes
        if payload is not None:
            self.dev.upload(self.plan["bufs"][name], payload)
        return self.plan["bufs"][name]

    def release(self):
        for ptr in self.plan["bufs"].values():
            self.dev.free(ptr)
        self.plan["bufs"].clear()


def recon_grid(dw, dh):
    return _cdiv(dw, RECON_TILE_X) * _cdiv(_cdiv(dh, 2), RECON_TILE_PAIRS)


RECON2_STAGE_CAP = 336   # kR2Cap


def offsets32_ok(rw, rh, dw, dh):
    """True if the 32-bit byte offsets of post_recon2_body (kernels/post.hip: mul24 products on a uniform base) cannot
    overflow: full history 8 dw dh, display planes 3 x 2 display_pitch(dw) dh, fallback colour 16 rw rh bytes all at
    most 2^32, and the mul24 row factors (8 dw, 2 pitch, 16 rw) below 2^24. True up to far beyond 8K output (D of an
    8K output: 597 MB of history); runtime.cpp recon2_supported checks the same."""
    pitch = (dw + 63) & ~63
    two32, two24 = 1 << 32, 1 << 24
    return (8 * dw * dh <= two32 and 6 * pitch * dh <= two32 and 16 * rw * rh <= two32 and
            16 * pitch < two24 and 16 * rw < two24)


def recon2_supported(rw, rh, dw, dh):
    """True if the render footprint of every post_recon2 workgroup (32 D columns x 8 row pairs) fits its LDS staging
    area: (floor(31 rx) + 4) (floor(14 ry) + 5) <= 336, rx = rw / dw; true for D >= about 1.43 R per axis (every FSR
    mode); dw >= 4 (the span history taps, history_cr_span, which the post_recon2* kernels take unconditionally;
    an output narrower than 3 pixels); and the sizes fit the kernels' 32-bit byte offsets (offsets32_ok). Otherwise
    use post_recon."""
    return (dw >= 4 and (int(31 * rw / dw) + 4) * (int(14 * rh / dh) + 5) <= RECON2_STAGE_CAP and
            offsets32_ok(rw, rh, dw, dh))


TF_STAGE_CAP = 352   # kTfCap (kernels/post.hip, post_tf*)
TF_OUT_X = 40         # output columns per post_tf workgroup (strip)
TF_CAPS = {4: 224, 8: 352, 16: 576}   # staging cap per NW (tf_cap<NW> in kernels/post.hip)
TF_DL_CAP = 644       # post_tf_dl*: native AA (D = 1.5 R)
TF4_DL_CAP = 448      # post_tf4_dl_* (D3D12 only, NW = 4): native AA and every ratio between 1.0 and 1.5 (Brief 15)


def tf_supported(rw, rh, dw, dh, nw=8, cap=None):
    """True if post_tf* (true fusion: reconstruction + output filter in one kernel, NW = nw row pairs per step) takes
    this size: recon2_supported (the runtime runs post_tf only on frames of the pack12 banded path; it also gives
    dw >= 4 and the 32-bit offsets), and the render footprint of every step (64 D columns x nw row pairs) fits the
    staging area, (floor(63 rx) + 4) (floor((2 nw - 2) ry) + 5) <= cap, rx = rw / dw. Default cap: 352 for NW = 8
    (post_tf_p12, true for D >= 2.25 R: the Quality, Balanced, Performance and Ultra Performance modes); TF_DL_CAP = 644
    for post_tf_dl_p12 (native AA, D = 1.5 R). runtime.cpp tf_supported checks the same (NW = 8)."""
    return (recon2_supported(rw, rh, dw, dh) and
            (int(63 * rw / dw) + 4) * (int((2 * nw - 2) * rh / dh) + 5) <= (cap or TF_CAPS.get(nw, TF_STAGE_CAP)))


def tf_grid(ow, oh, cpairs):
    """post_tf* grid with one chunk size (k1 >= the chunk count): ceil(ow / 40) strips x ceil(ceil(oh / 2) / cpairs)
    chunks (post_sizes.tf_chunks for the runtime's two chunk sizes)."""
    return _cdiv(ow, TF_OUT_X) * _cdiv(_cdiv(oh, 2), cpairs)


def recon6_supported(rw, rh, dw, dh):
    """post_recon6*: recon2_supported and R < D on both axes (true for every upscaling mode; the kernel does not clamp
    its LDS indices, which needs D rows / columns closer than one render pixel)."""
    return recon2_supported(rw, rh, dw, dh) and rw < dw and rh < dh


def prepare_post(dev, weights, x, kernel=KERNEL_RECON):
    """Allocate and upload everything for post_recon (or a RECON2_VARIANTS kernel); returns a plan for
    stage_runner."""
    weights = bytes(weights)
    if len(weights) != BLOB_BYTES:
        raise ValueError(f"post: weight blob has {len(weights)} bytes, expected {BLOB_BYTES}")
    prm = post_params(x)
    dw, dh = prm.dw, prm.dh
    data = post_device_inputs(x)
    if kernel in RECON_PACK12:
        data = pack12_inputs(data)
    hist6, planar = kernel in RECON_HIST6, kernel in RECON_PLANAR
    if hist6:
        data["hist_prev"] = encode_history(x["history"], True)
    if planar:
        data["hist_prev"] = encode_history_planar(x["history"])
    pl = _Plan(dev)
    try:
        b = {"w": pl.alloc("w", len(weights), weights)}
        for k, v in data.items():
            b[k] = pl.alloc(k, len(v), v)
        b["history"] = pl.alloc("history", history_bytes(dw, dh, hist6 or planar))   # 6-byte formats: 6 n + 8
        b["reduced"] = pl.alloc("reduced", max((dw // 2) * (dh // 2), 1) * 4)
        b["display"] = pl.alloc("display", display_bytes(dw, dh))
        P = ctypes.c_void_p
        args = [P(b[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far", "w",
                                  "history", "reduced", "display")] + [prm, P(0)]   # expo: null
        if kernel != KERNEL_RECON and not recon2_supported(prm.rw, prm.rh, dw, dh):
            raise ValueError(f"{kernel}: R {prm.rw}x{prm.rh} too large for D {dw}x{dh} (recon2_supported)")
        if kernel in RECON6_VARIANTS and not recon6_supported(prm.rw, prm.rh, dw, dh):
            raise ValueError(f"{kernel}: needs R {prm.rw}x{prm.rh} smaller than D {dw}x{dh} per axis")
        if (kernel in RECON3_VARIANTS or kernel in RECON4_VARIANTS or kernel in PREP_VARIANTS) and dw < 4:
            raise ValueError(f"{kernel}: needs D at least 4 columns wide")
        if kernel in PREP_VARIANTS:   # pre-pass, then the reconstruction from its records
            n = prm.rw * prm.rh
            b["prep"] = pl.alloc("prep", n * PREP_RECORD_BYTES)
            pl.plan["launches"].append((KERNEL_PREP, _cdiv(n, RECON_THREADS), RECON_THREADS,
                                        [P(b[k]) for k in ("color", "A", "B", "C", "prep")] + [prm, P(0)]))
            args = [P(b[k]) for k in ("prep", "color", "hist_prev", "m_near", "m_center", "m_far", "w", "history",
                                      "reduced", "display")] + [prm, P(0)]
        if kernel in RECON_PACK12:
            args = args + [ctypes.c_int(0)]   # tile0
        out_mode = kernel == KERNEL_RECON_OUT
        if out_mode:   # D = O: the output (RGBA16F) in place of the display image, dec_scale after the params
            b["out"] = pl.alloc("out", dw * dh * 8)
            args = [P(b[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far", "w",
                                      "history", "reduced", "out")]
            args += [prm, ctypes.c_float(0.01 / exposure_or_one(x["exposure"])), P(0)]
        pl.plan["launches"].append((kernel, recon_grid(dw, dh), RECON_THREADS, args))
        last = ("out", (dh, dw, 3)) if out_mode else ("display", (dh, dw, 3))
        pl.plan["outputs"] = [("history", (dh, dw, 3)), ("reduced", (dh // 2, dw // 2, 3)), last]
        hdec = (decode_history_planar if planar else
                (lambda raw, shape: decode_history_fmt(raw, shape, True)) if hist6 else decode_history)
        pl.plan["decoders"] = {"history": hdec, "reduced": decode_reduced, "display": decode_planes,
                               "out": decode_out_f16}
    except Exception:
        pl.release()
        raise
    return pl.plan


def time_prep(runner, weights, x, repeats=50, code_object=None):
    """Median GPU time (ms) of the post_prep pre-pass alone (from a post_recon5 plan)."""
    plan = prepare_post(runner.dev, weights, x, "post_recon5")
    plan["launches"] = plan["launches"][:1]
    return time_plans(runner, [plan], code_object, repeats)


def filter_grid(ow, oh, kernel=KERNEL_FILTER):
    s = FILTER_VARIANTS[kernel] if kernel in FILTER_VARIANTS else FILTER_O16_VARIANTS[kernel]
    return _cdiv(ow, FILTER_TILE_XS[kernel]) * _cdiv(_cdiv(oh, 2), FILTER_STRIPS * s)


def prepare_filter(dev, weights, x, kernel=KERNEL_FILTER):
    """Allocate and upload everything for post_filter (or one of its FILTER_VARIANTS / FILTER_O16_VARIANTS: "out"
    decoded from RGBA16F); returns a plan for stage_runner."""
    if not isinstance(x, dict):
        raise TypeError("filter: x must be the dict of spec 90")
    ow, oh = (int(v) for v in x["output_size"])
    disp = np.asarray(x["display"], dtype=np.float32)
    dh, dw = disp.shape[:2]
    if disp.shape != (dh, dw, 3) or (dw, dh) != (recon_size_for(ow), recon_size_for(oh)) or ow < 1 or oh < 1:
        raise ValueError(f"filter: display {disp.shape} does not match D = floor((3 O + 1) / 2) for O = {ow}x{oh}")
    e = exposure_or_one(x["exposure"])
    planes = encode_planes(disp)
    pl = _Plan(dev)
    try:
        src = pl.alloc("display", len(planes), planes)
        o16 = kernel in FILTER_O16_VARIANTS
        out = pl.alloc("out", ow * oh * (8 if o16 else 12))
        I = ctypes.c_int
        args = [ctypes.c_void_p(src), ctypes.c_void_p(out), I(dw), I(dh), I(ow), I(oh), ctypes.c_float(0.01 / e),
                ctypes.c_void_p(0)]   # expo: null (the runtime passes its exposure pair)
        pl.plan["launches"].append((kernel, filter_grid(ow, oh, kernel), FILTER_THREADS, args))
        pl.plan["outputs"] = [("out", (oh, ow, 3))]
        pl.plan["decoders"] = {"out": decode_out_f16 if o16 else decode_f32}
    except Exception:
        pl.release()
        raise
    return pl.plan


def prepare_decode(dev, x):
    """post_decode_o16 (the identity output filter, D = O): x = {"display": [dh, dw, 3] FP16-exact E values,
    "exposure": e}; returns a plan whose output "out" is [dh, dw, 3] (decoded from RGBA16F)."""
    disp = np.asarray(x["display"], dtype=np.float32)
    dh, dw = disp.shape[:2]
    planes = encode_planes(disp)
    pl = _Plan(dev)
    try:
        src = pl.alloc("display", len(planes), planes)
        out = pl.alloc("out", dw * dh * 8)
        args = [ctypes.c_void_p(src), ctypes.c_void_p(out), ctypes.c_int(dw), ctypes.c_int(dh),
                ctypes.c_float(0.01 / exposure_or_one(x["exposure"])), ctypes.c_void_p(0)]
        pl.plan["launches"].append((KERNEL_DECODE, _cdiv(dw * dh, RECON_THREADS), RECON_THREADS, args))
        pl.plan["outputs"] = [("out", (dh, dw, 3))]
        pl.plan["decoders"] = {"out": decode_out_f16}
    except Exception:
        pl.release()
        raise
    return pl.plan


# ---- banded two-kernel path ------------------------------------------------------------------------------------------
# post_recon2_r_band and post_filter_c1_s<S>_band (BAND_RECON, BAND_FILTERS at the top) launched alternately over
# horizontal bands.


def band_plan(dw, dh, ow, oh, n, s=4):
    """Launch list for n bands: [("recon" | "filter", tile0, tiles), ...] in launch order. Reconstruction band b
    covers recon tile rows [b R / n, (b + 1) R / n) (R = ceil(ceil(dh / 2) / 8) rows of 16 D rows); after it, every
    filter tile row (2 s output row pairs, 6 s D rows) whose input rows 3p - 2 ... 3p + 4 (clamped to D) are complete
    and that has not run yet."""
    tx = _cdiv(dw, RECON_TILE_X)
    rrows = _cdiv(_cdiv(dh, 2), RECON_TILE_PAIRS)
    fx = _cdiv(ow, 40)
    frows = _cdiv(_cdiv(oh, 2), FILTER_STRIPS * s)
    n = max(1, min(int(n), rrows))
    out, fdone = [], 0
    for b in range(n):
        r0, r1 = b * rrows // n, (b + 1) * rrows // n
        if r1 > r0:
            out.append(("recon", r0 * tx, (r1 - r0) * tx))
        d_done = min(16 * r1, dh)        # D rows 0 ... d_done - 1 complete
        f1 = fdone
        while f1 < frows:
            last_pair = min((f1 + 1) * FILTER_STRIPS * s, _cdiv(oh, 2)) - 1
            need = min(3 * last_pair + 4, dh - 1)
            if need >= d_done:
                break
            f1 += 1
        if b == n - 1:
            f1 = frows
        if f1 > fdone:
            out.append(("filter", fdone * fx, (f1 - fdone) * fx))
            fdone = f1
    return out


def prepare_banded(dev, weights, x, n, filter_kernel="post_filter_c1_s4_band", recon_kernel=BAND_RECON):
    """Plan of the banded path (recon_kernel: post_recon2_r_band or post_recon2_r_hp_band, + filter_kernel over n
    bands) for the spec 90 "post" dict x (plus optional "output_size"). Outputs "history", "reduced", "out" (f32), as
    post_recon2_r (or _hp) + post_filter_c1_s<S>."""
    s = BAND_FILTERS[filter_kernel]
    plan = prepare_post(dev, weights, x, BAND_RECONS[recon_kernel])
    try:
        prm_args = plan["launches"][0][3][:14]   # without the tile0 of the pack12 kernels
        dw, dh = x["recon_size"]
        ow, oh = (int(v) for v in x["output_size"]) if "output_size" in x else output_size_for(dw, dh)
        pl = _Plan(dev)
        pl.plan = plan
        out = pl.alloc("out", ow * oh * 12)
        P = ctypes.c_void_p
        e = exposure_or_one(x["exposure"])
        fargs = [P(plan["bufs"]["display"]), P(out), ctypes.c_int(dw), ctypes.c_int(dh), ctypes.c_int(ow),
                 ctypes.c_int(oh), ctypes.c_float(0.01 / e), P(0)]
        launches = []
        for kind, tile0, tiles in band_plan(dw, dh, ow, oh, n, s):
            if kind == "recon":
                launches.append((recon_kernel, tiles, RECON_THREADS, prm_args + [ctypes.c_int(tile0)]))
            else:
                launches.append((filter_kernel, tiles, FILTER_THREADS, fargs + [ctypes.c_int(tile0)]))
        plan["launches"] = launches
        plan["outputs"] = [("history", (dh, dw, 3)), ("reduced", (dh // 2, dw // 2, 3)), ("out", (oh, ow, 3))]
        plan["decoders"] = dict(plan["decoders"], out=decode_f32)
    except Exception:
        for ptr in plan["bufs"].values():
            dev.free(ptr)
        raise
    return plan


# ---- fused reconstruction + output filter ---------------------------------------------------------------------------

def output_size_for(dw, dh):
    """O with D = floor((3 O + 1) / 2) per axis (unique)."""
    def one(d):
        for o in (2 * d // 3, 2 * d // 3 + 1, 2 * d // 3 - 1):
            if o >= 1 and recon_size_for(o) == d:
                return o
        raise ValueError(f"no output size gives D = {d}")
    return one(int(dw)), one(int(dh))


FUSED_STAGE_CAP = 336   # kFuStageCap


def fused_supported(rw, rh, dw, dh):
    """True if every band footprint of the fused kernels fits their LDS staging area (kFuStageCap render pixels):
    at most floor(63 rx) + 4 columns times floor(4 ry) + 5 rows, rx = rw / dw, ry = rh / dh. Holds for D >= about
    1.43 R per axis, so for every FSR mode (D = 1.5 O, O >= R). Otherwise use post_recon + post_filter."""
    ncol = int(63 * rw / dw) + 4
    nrow = int(4 * rh / dh) + 5
    return ncol * nrow <= FUSED_STAGE_CAP


def fused_grid(ow, oh):
    return _cdiv(ow, FUSED_OUT_X) * _cdiv(oh, FUSED_OUT_Y)


def prepare_fused(dev, weights, x, kernel=KERNEL_FUSED):
    """Allocate and upload everything for a post_fused* or post_fused3* kernel. x: the spec 90 "post" dict, plus
    optionally "output_size" (default: the O that gives recon_size). The filter uses the post exposure. Outputs:
    "history" (decoded from the kernel's format; for _h4 the input history is quantized to that format first),
    "reduced" (as stage post) and "out" ([oh, ow, 3] float32; FP16 values for the _o16 kernels)."""
    if kernel in FUSED3_VARIANTS:
        fmt, o16 = FUSED3_VARIANTS[kernel][:2]
        hist6 = fmt == 6
    else:
        (hist6, o16), fmt = FUSED_VARIANTS[kernel], None
    weights = bytes(weights)
    if len(weights) != BLOB_BYTES:
        raise ValueError(f"post: weight blob has {len(weights)} bytes, expected {BLOB_BYTES}")
    prm = post_params(x)
    dw, dh = prm.dw, prm.dh
    ow, oh = (int(v) for v in x["output_size"]) if "output_size" in x else output_size_for(dw, dh)
    if (dw, dh) != (recon_size_for(ow), recon_size_for(oh)):
        raise ValueError(f"post_fused: D {dw}x{dh} does not match O {ow}x{oh}")
    if not fused_supported(prm.rw, prm.rh, dw, dh):
        raise ValueError(f"post_fused: R {prm.rw}x{prm.rh} too large for D {dw}x{dh} (fused_supported); use the "
                         "post_recon + post_filter path")
    data = post_device_inputs(x)
    data["hist_prev"] = encode_history4(x["history"]) if fmt == 4 else encode_history(x["history"], hist6)
    pl = _Plan(dev)
    try:
        b = {"w": pl.alloc("w", len(weights), weights)}
        for k, v in data.items():
            b[k] = pl.alloc(k, len(v), v)
        b["history"] = pl.alloc("history", dw * dh * 4 if fmt == 4 else history_bytes(dw, dh, hist6))
        b["reduced"] = pl.alloc("reduced", max((dw // 2) * (dh // 2), 1) * 4)
        b["out"] = pl.alloc("out", ow * oh * (8 if o16 else 12))
        P = ctypes.c_void_p
        args = [P(b[k]) for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far", "w",
                                  "history", "reduced", "out")]
        args += [prm, ctypes.c_int(ow), ctypes.c_int(oh), ctypes.c_float(0.01 / exposure_or_one(x["exposure"])), P(0)]
        pl.plan["launches"].append((kernel, fused_grid(ow, oh), FUSED_THREADS, args))
        pl.plan["outputs"] = [("history", (dh, dw, 3)), ("reduced", (dh // 2, dw // 2, 3)), ("out", (oh, ow, 3))]
        pl.plan["decoders"] = {"history": decode_history4 if fmt == 4 else
                               (lambda raw, shape: decode_history_fmt(raw, shape, hist6)),
                               "reduced": decode_reduced, "out": decode_out_f16 if o16 else decode_f32}
    except Exception:
        pl.release()
        raise
    return pl.plan


CODE_OBJECT_W64 = "post_w64.co"   # kernels/post.hip built with -mwavefrontsize64 (build_wave64)


def build_wave64(build_dir, rocm_dir="", clang=None):
    """Build post_w64.co: the same kernels in wave64 mode (on RDNA 4 a wave64 FP32 FMA/MUL/ADD issues in one pass,
    in wave32 only through VOPD pairs). Same compiler command as build.py plus -mwavefrontsize64."""
    import os
    import subprocess
    from clean.runner import build as B
    clang = clang or B.find_clang(rocm_dir)
    out = os.path.join(build_dir, CODE_OBJECT_W64)
    cmd = B.device_command(clang, rocm_dir, os.path.join(B.KERNEL_DIR, "post.hip"), out,
                           device_libs=bool(rocm_dir), extra=["-mwavefrontsize64"])
    subprocess.run(cmd, check=True)
    return out


def _function(runner, name, code_object=None):
    """Kernel handle from the runner's post code object, or from another code object (e.g. post_w64.co) of its
    build directory (hip backend; the emulator has one implementation per kernel name)."""
    if code_object is None or code_object == CODE_OBJECT or getattr(runner, "backend", "hip") != "hip":
        return runner._func(name)
    import os
    cache = runner.__dict__.setdefault("_post_extra", {})
    if code_object not in cache:
        with open(os.path.join(runner.build_dir, code_object), "rb") as f:
            cache[code_object] = (runner.dev.load_module(f.read()), {})
    module, funcs = cache[code_object]
    if name not in funcs:
        funcs[name] = runner.dev.get_function(module, name)
    return funcs[name]


def run_plan(runner, plan, code_object=None):
    """Execute a plan of prepare_post / prepare_filter / prepare_fused and return its decoded outputs; releases it."""
    dev = runner.dev
    try:
        for name, _ in plan["outputs"]:
            dev.memset(plan["bufs"][name], 0x7F, plan["sizes"][name])
        for name, grid, threads, args in plan["launches"]:
            dev.launch(_function(runner, name, code_object), grid, threads, args)
        dev.sync()
        return {name: plan["decoders"][name](dev.download(plan["bufs"][name], plan["sizes"][name]), shape)
                for name, shape in plan["outputs"]}
    finally:
        for ptr in plan["bufs"].values():
            dev.free(ptr)
        plan["bufs"].clear()


def time_plans(runner, plans, code_object=None, repeats=50):
    """Median GPU time (ms) of all launches of the given plans back to back (inputs already on the device);
    releases the plans."""
    dev = runner.dev
    launches = [(_function(runner, n, code_object), g, t, a) for plan in plans for n, g, t, a in plan["launches"]]
    start, stop = dev.event(), dev.event()
    try:
        for f, g, t, a in launches:
            dev.launch(f, g, t, a)
        dev.sync()
        times = []
        for _ in range(int(repeats)):
            dev.record(start)
            for f, g, t, a in launches:
                dev.launch(f, g, t, a)
            dev.record(stop)
            times.append(dev.elapsed_ms(start, stop))
        return float(np.median(times))
    finally:
        dev.destroy_event(start)
        dev.destroy_event(stop)
        for plan in plans:
            for ptr in plan["bufs"].values():
                dev.free(ptr)
            plan["bufs"].clear()


def run_fused(runner, weights, x, kernel=KERNEL_FUSED, code_object=None):
    """Run a fused kernel through a stage_runner.Runner (hip or emu backend): {"history", "reduced", "out"}."""
    return run_plan(runner, prepare_fused(runner.dev, weights, x, kernel), code_object)


def time_fused(runner, weights, x, kernel=KERNEL_FUSED, repeats=50, code_object=None):
    """Median GPU time (ms) of one fused launch, inputs already on the device."""
    return time_plans(runner, [prepare_fused(runner.dev, weights, x, kernel)], code_object, repeats)


def time_two_kernel(runner, weights, x, display_like, repeats=50, code_object=None, filter_kernel=KERNEL_FILTER):
    """Median GPU time (ms) of post_recon followed by the output filter (the A/B baseline of the fused kernels).
    display_like: the filter input dict (its display buffer stands in for post_recon's output; same size)."""
    return time_plans(runner, [prepare_post(runner.dev, weights, x),
                               prepare_filter(runner.dev, b"", display_like, filter_kernel)], code_object, repeats)
