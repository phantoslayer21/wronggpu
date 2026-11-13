# SPDX-License-Identifier: MIT
"""Offline RR on captured game frames: feeds a capture of the probe build (CAPTURE.md) through WrongGPU's RR runtime
and writes PNGs of the input colour and of our RR output, so the image quality can be judged without a game run.

    python clean/rr_f/proxy/tools/rr_capture_replay.py <capture folder> --model <model.wgcm> --out <dir>
        [--runtime <wronggpu_rr_runtime.dll>] [--host <rr_test_host.exe>] [--frames FIRST:LAST] [--head strict|fast]
        [--exposure X] [--settings N] [--previews] [--no-gpu] [--crop X,Y,W,H] [convention switches]

1. Reads every complete frame folder (frame_<n>\\frame.json present) in evaluate order and decodes its input files
   (header + rows, CAPTURE.md) by DXGI format.
2. Plays the release proxy's part for each frame (pages 61.3 / 61.4 / 62.2): creation decisions from the first frame
   (options packing / HDR, constants depth inverted / jittered, hardware or linear depth), the routing of buffer types
   to the runtime's roles, the render size from the colour's extent, motion-vector scale, camera (constants V2C,
   options W2V), jitter and reset, and converts formats the test host cannot create into the values the API read
   conversion gives (62.5: R32 typeless depth read as float, typeless families as their float / unorm member, 24-bit
   and 16-bit unorm depth as float).
3. Runs the frames in order on ONE runtime instance (temporal state carried) with the runtime's test host
   (clean/rr-f-rt-int: rr_test_host.exe, the proxy's own Lifecycle) on the GPU under the GPU lock (WG_GPU_LOCK, BUILDING.md). The model is
   only handed to the runtime by path (WG_RR_MODEL); this tool never opens it.
4. Writes per frame: <out>\\frame_<n>_input.png (the colour input, its extent), <out>\\frame_<n>_rr.png (our output),
   <out>\\frame_<n>_side.png (input upscaled bilinearly to the output size | RR output) and
   <out>\\frame_<n>_crop.png (the centre quarter at output pixels: input nearest-upscaled | RR output), all with one
   tonemap (x * exposure, Reinhard x / (1 + x) per channel, sRGB encoding; NaN / inf shown magenta); with --previews
   also every input (depth, normals, albedos, motion, guides). <out>\\summary.json: per frame result code and simple
   sanity numbers (finite fraction, mean luminance of input and output, PSNR of the tonemapped output against the
   bilinear input).

--no-gpu stops after step 2 (frame files and job written, inputs decoded and previewed).

Convention switches (quality investigation; the replay only: the runtime and the release proxy keep their defaults).
Each changes ONE input convention against what the release proxy does (62.2 / 62.3 / 61.3); the default of every
switch is the release proxy's behaviour. CONVENTIONS lists them with the spec rule that defines the default.
    --layer spec|drop|additive|straight|invert   type 39 as the layer (62.2, 54b.0.2) / not routed / alpha forced 0 /
                                                 rgb multiplied by alpha (not premultiplied) / alpha := 1 - alpha
    --g1 spec|drop                               type 58 (or 41) as G1 (62.2, 52.8) / not routed
    --normals stored|view|flat|zero              as stored (62.3) / xyz rotated by the options' world-to-view /
                                                 diagnostics: xyz := (0, 0, 1) / xyz := 0
    --roughness stored|invert|square|sqrt|half   4th normal channel as stored (52.3.4) / 1 - r / r^2 / sqrt(r) / 0.5
    --albedo unorm|srgb|swap|grey|one            unorm read (52.1.2) / sRGB-to-linear decode / diffuse <-> specular /
                                                 diagnostics: both albedos replaced by their luminance / by 1
    --hitdist stored|drop|SCALE                  type 42 as read (62.4) / not routed (smv off) / multiplied by SCALE
    --colour-scale K                             colour, layer and colour guides multiplied by K (pre-exposure test;
                                                 preset F's auto exposure, page 58, is always on)
    --depth-inverted capture|0|1                 creation decision (61.3) / forced
    --mv-jittered capture|0|1                    creation decision (61.3) / forced
    --mv-scale SX,SY                             factors on the game's motion-vector scale (61.4 step 4), e.g. -1,-1
    --jitter-sign SX,SY                          factors on the game's jitter (61.4 step 5, 62.3), e.g. -1,-1
    --colour-probe none|grey|swap_rb|rotate|zero_r|zero_g|zero_b
                                                 diagnostic: colour input as luminance only / R and B exchanged /
                                                 (r, g, b) := (g, b, r) / one channel set to 0
"""
import argparse
import glob
import json
import math
import os
import struct
import subprocess
import sys
import time
import zlib

import numpy as np

_RT_OUT = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "build")), "rr-f-rt-int", "out")
DEFAULT_RUNTIME = os.path.join(_RT_OUT, "wronggpu_rr_runtime.dll")
DEFAULT_HOST = os.path.join(_RT_OUT, "rr_test_host.exe")
# Optional GPU lock (WG_GPU_LOCK: a script run as `python <lock> --timeout <s> -- <command>`, e.g. to keep
# GPU jobs of several shells apart); unset: the command runs directly.
GPU_LOCK = os.environ.get("WG_GPU_LOCK", "")


def _locked(cmd, timeout="3600"):
    return ([sys.executable, GPU_LOCK, "--timeout", str(timeout), "--"] if GPU_LOCK else []) + list(cmd)
MAGIC = b"WGRRCAP1"
HEADER_BYTES = 64

# WgRrRole (include/wg_rr_adapter.h)
ROLE = dict(COLOUR=0, OUTPUT=1, DEPTH=2, MOTION=3, DIFFUSE_ALBEDO=4, SPECULAR_ALBEDO=5, NORMALS=6, ROUGHNESS=7,
            SPECULAR_MOTION=8, SPECULAR_HIT_DIST=9, DISOCCLUSION=10, G1=11, DOF_GUIDE=12, BEFORE_TRANSPARENCY=13,
            LAYER=14, OPACITY=15, INPUT_ALPHA=16, OUTPUT_ALPHA=17)
# public buffer types (include/rr_proxy_abi.h)
BT = dict(DEPTH=0, MOTION=1, COLOUR_IN=3, COLOUR_OUT=4, NORMALS=5, ROUGHNESS=6, DIFFUSE_ALBEDO=7, SPECULAR_ALBEDO=8,
          SPECULAR_MOTION=10, DISOCCLUSION=11, NORMAL_ROUGHNESS=14, INPUT_ALPHA=34, COLOUR_BEFORE_PARTICLES=39,
          COLOUR_BEFORE_TRANSPARENCY=40, COLOUR_BEFORE_FOG=41, SPECULAR_HIT_DISTANCE=42, LINEAR_DEPTH=49,
          TRANSPARENCY_LAYER=51, TRANSPARENCY_OPACITY=52, SSS_GUIDE=58, DOF_GUIDE=64, OUTPUT_ALPHA=67)
TYPE_NAME = {0: "depth", 1: "motion", 3: "colour", 5: "normals", 6: "roughness", 7: "diffuse_albedo",
             8: "specular_albedo", 10: "specular_motion", 11: "disocclusion", 14: "normal_roughness", 29: "bias_hint",
             34: "input_alpha", 39: "colour_before_particles", 40: "colour_before_transparency",
             41: "colour_before_fog", 42: "hit_distance", 49: "linear_depth", 51: "transparency_layer",
             52: "transparency_opacity", 58: "sss_guide", 64: "dof_guide"}
SHADER_READ = 0x40 | 0x80   # the host creates its textures in (NON_)PIXEL_SHADER_RESOURCE (no UAV flag on inputs)
OUT_FORMAT = 10             # RGBA16F output, as game A's
OUT_ORIGIN = (1, 1)         # rr_test_host's output texture: (Wo + 3) x (Ho + 2), output rectangle at (1, 1)

# DXGI formats: (bytes per texel, channels, kind); kind: f32 f16 unorm8 unorm16 snorm8 snorm16 rgb10a2 r11g11b10
# d24 (24-bit unorm in the low bits of 32) bgra8 srgb8 bgra_srgb8 rgb9e5
FMT = {
    1: (16, 4, "f32"), 2: (16, 4, "f32"), 5: (12, 3, "f32"), 6: (12, 3, "f32"), 9: (8, 4, "f16"), 10: (8, 4, "f16"),
    11: (8, 4, "unorm16"), 13: (8, 4, "snorm16"), 15: (8, 2, "f32"), 16: (8, 2, "f32"),
    19: (8, 1, "f32"), 20: (8, 1, "f32"), 21: (8, 1, "f32"),          # R32G8X24 family: depth in the first 4 bytes
    23: (4, 4, "rgb10a2"), 24: (4, 4, "rgb10a2"), 26: (4, 3, "r11g11b10"), 67: (4, 3, "rgb9e5"),
    27: (4, 4, "unorm8"), 28: (4, 4, "unorm8"), 29: (4, 4, "srgb8"), 31: (4, 4, "snorm8"),
    33: (4, 2, "f16"), 34: (4, 2, "f16"), 35: (4, 2, "unorm16"), 37: (4, 2, "snorm16"),
    39: (4, 1, "f32"), 40: (4, 1, "f32"), 41: (4, 1, "f32"),
    44: (4, 1, "d24"), 45: (4, 1, "d24"), 46: (4, 1, "d24"),
    48: (2, 2, "unorm8"), 49: (2, 2, "unorm8"), 51: (2, 2, "snorm8"),
    53: (2, 1, "f16"), 54: (2, 1, "f16"), 55: (2, 1, "unorm16"), 56: (2, 1, "unorm16"), 58: (2, 1, "snorm16"),
    60: (1, 1, "unorm8"), 61: (1, 1, "unorm8"), 63: (1, 1, "snorm8"), 65: (1, 1, "unorm8"),
    87: (4, 4, "bgra8"), 88: (4, 4, "bgra8"), 90: (4, 4, "bgra8"), 91: (4, 4, "bgra_srgb8"),
}
# formats rr_test_host can create and upload as they are (its bpp table); the typeless members map to these
HOST_DIRECT = {2, 10, 16, 41, 28, 29, 24, 26, 34, 87, 54, 49, 61}
TYPELESS_TO = {1: 2, 9: 10, 15: 16, 23: 24, 27: 28, 33: 34, 39: 41, 60: 61, 48: 49, 90: 87, 40: 41}
DEPTH_UNORM16 = {53, 55, 56}   # 62.5: the R16 typeless family read as depth is D16 (unorm)


class CaptureError(Exception):
    pass


# ------------------------------------------------------------------------------------------------- reading
def read_input_file(path):
    """One captured input: (header dict, raw bytes of the rows)."""
    with open(path, "rb") as fh:
        b = fh.read()
    if len(b) < HEADER_BYTES or b[:8] != MAGIC:
        raise CaptureError(f"{path}: not a capture input file")
    v = struct.unpack_from("<14I", b, 8)
    hb = v[0]
    h = dict(header_bytes=hb, type=v[1], format=v[2], width=v[3], height=v[4], row_pitch=v[5], tag_format=v[6],
             state=v[7], extent=dict(left=v[8], top=v[9], width=v[10], height=v[11]), lifetime=v[12])
    data = b[hb:]
    if len(data) != h["row_pitch"] * h["height"]:
        raise CaptureError(f"{path}: {len(data)} data bytes, header says {h['height']} rows of {h['row_pitch']}")
    return h, data


def texels(h, data):
    """The rows as a [height][width][bytes per texel] uint8 array (the row pitch's padding dropped)."""
    if h["format"] not in FMT:
        raise CaptureError(f"type {h['type']}: DXGI format {h['format']} is not supported by this tool")
    bpt = FMT[h["format"]][0]
    if h["row_pitch"] < h["width"] * bpt:
        raise CaptureError(f"type {h['type']}: row pitch {h['row_pitch']} < width {h['width']} x {bpt}")
    a = np.frombuffer(data, np.uint8).reshape(h["height"], h["row_pitch"])
    return np.ascontiguousarray(a[:, :h["width"] * bpt]).reshape(h["height"], h["width"], bpt)


def _f11(bits, mbits):
    """unsigned small floats of R11G11B10 (5-bit exponent, mbits mantissa) -> float32"""
    e = (bits >> mbits) & 31
    m = (bits & ((1 << mbits) - 1)).astype(np.float64)
    out = np.where(e == 0, m / (1 << mbits) * 2.0 ** -14, (1 + m / (1 << mbits)) * 2.0 ** (e.astype(np.float64) - 15))
    out = np.where(e == 31, np.where(m == 0, np.inf, np.nan), out)
    return out.astype(np.float32)


def srgb_to_linear(c):
    c = c.astype(np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def decode(h, data, depth=False):
    """Values as the API's read conversion gives them (62.5): float32 [height][width][channels]."""
    t = texels(h, data)
    bpt, ch, kind = FMT[h["format"]]
    hh, ww = t.shape[:2]
    if kind == "f32":
        v = t[..., :4 * ch].copy().view(np.float32).reshape(hh, ww, ch)
    elif kind == "f16":
        v = t.copy().view(np.float16).reshape(hh, ww, ch).astype(np.float32)
        if depth and h["format"] in DEPTH_UNORM16:
            v = t.copy().view(np.uint16).reshape(hh, ww, ch).astype(np.float32) / 65535.0
    elif kind == "unorm16":
        v = t.copy().view(np.uint16).reshape(hh, ww, ch).astype(np.float32) / 65535.0
    elif kind == "snorm16":
        v = np.maximum(t.copy().view(np.int16).reshape(hh, ww, ch).astype(np.float32) / 32767.0, -1.0)
    elif kind == "unorm8":
        v = t.astype(np.float32) / 255.0
    elif kind == "snorm8":
        v = np.maximum(t.view(np.int8).astype(np.float32) / 127.0, -1.0)
    elif kind == "srgb8":
        v = np.concatenate([srgb_to_linear(t[..., :3] / 255.0), t[..., 3:].astype(np.float32) / 255.0], axis=2)
    elif kind == "bgra8":
        v = t[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    elif kind == "bgra_srgb8":
        s = t[..., [2, 1, 0, 3]]
        v = np.concatenate([srgb_to_linear(s[..., :3] / 255.0), s[..., 3:].astype(np.float32) / 255.0], axis=2)
    elif kind == "rgb10a2":
        u = t.copy().view(np.uint32).reshape(hh, ww)
        v = np.stack([(u & 1023) / 1023.0, ((u >> 10) & 1023) / 1023.0, ((u >> 20) & 1023) / 1023.0,
                      ((u >> 30) & 3) / 3.0], axis=2).astype(np.float32)
    elif kind == "r11g11b10":
        u = t.copy().view(np.uint32).reshape(hh, ww)
        v = np.stack([_f11(u & 2047, 6), _f11((u >> 11) & 2047, 6), _f11((u >> 22) & 1023, 5)], axis=2)
    elif kind == "rgb9e5":   # R9G9B9E5_SHAREDEXP (game B2 colour / output): 9-bit mantissas, shared exponent, bias 15
        u = t.copy().view(np.uint32).reshape(hh, ww)
        sc = np.ldexp(1.0, ((u >> 27) & 31).astype(np.int32) - 24)
        v = np.stack([(u & 511) * sc, ((u >> 9) & 511) * sc, ((u >> 18) & 511) * sc], axis=2).astype(np.float32)
    elif kind == "d24":
        u = t.copy().view(np.uint32).reshape(hh, ww, 1)
        v = ((u & 0xFFFFFF).astype(np.float64) / 16777215.0).astype(np.float32)
    else:
        raise CaptureError(f"format {h['format']}: no decoder")
    return v


def for_host(h, data, depth=False):
    """(DXGI format, tight texel bytes) the test host can create: the captured bytes as they are when the host knows
    the format (typeless resources as their typed member: the same bits, read as the runtime's own view would), else
    the API-converted values as RGBA32F / RG32F / R32F."""
    f = h["format"]
    if depth:
        if f in (39, 40, 41, 19, 20, 21):       # 32-bit float depth: the bits, read as R32_FLOAT
            t = texels(h, data)
            return 41, np.ascontiguousarray(t[..., :4]).tobytes()
        v = decode(h, data, depth=True)[..., :1]
        return 41, np.ascontiguousarray(v, np.float32).tobytes()
    g = TYPELESS_TO.get(f, f)
    if g in HOST_DIRECT and FMT[g][0] == FMT[f][0]:
        return g, texels(h, data).tobytes()
    v = decode(h, data)
    ch = v.shape[2]
    if ch == 3:
        v = np.concatenate([v, np.ones(v.shape[:2] + (1,), np.float32)], axis=2)
        ch = 4
    return {1: 41, 2: 16, 4: 2}[ch], np.ascontiguousarray(v, np.float32).tobytes()


def load_capture(folder, first=None, last=None):
    """Complete frames (frame.json present) in evaluate order: [dict(meta, inputs={type: (header, data)}, dir)]."""
    frames = []
    for d in glob.glob(os.path.join(folder, "frame_*")):
        j = os.path.join(d, "frame.json")
        if not os.path.isfile(j):
            continue
        with open(j, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        n = meta["evaluate"]
        if (first is not None and n < first) or (last is not None and n > last):
            continue
        inputs = {}
        for e in meta["inputs"]:
            h, data = read_input_file(os.path.join(d, e["file"]))
            if h["type"] != e["type"] or h["format"] != e["format"]:
                raise CaptureError(f"{d}: {e['file']} does not match frame.json")
            inputs[h["type"]] = (h, data)
        frames.append(dict(meta=meta, inputs=inputs, dir=d))
    frames.sort(key=lambda f: f["meta"]["evaluate"])
    if not frames:
        raise CaptureError(f"{folder}: no complete frame folder (frame_*\\frame.json)")
    return frames


# ------------------------------------------------------------------------------------------------- conventions
# name: (default = the release proxy's behaviour, allowed values or a kind, spec rule of the default)
CONVENTIONS = {
    "layer": ("spec", ("spec", "drop", "additive", "straight", "invert"),
              "62.2 / 54b.0.2 / 54b.3: 39 replaces 51 as the layer, premultiplied RGBA, col*(1-a)+L"),
    "g1": ("spec", ("spec", "drop"), "62.2 / 52.8: 41 else 58 is the input-stage guide G1"),
    "normals": ("stored", ("stored", "view", "flat", "zero"), "62.3: normals as stored by the game; 52.7.2 uses them with "
                "camera-relative world positions"),
    "roughness": ("stored", ("stored", "invert", "square", "sqrt", "half"), "52.3.4: rough = 4th channel of N (rghPacked)"),
    "albedo": ("unorm", ("unorm", "srgb", "swap", "grey", "one"), "52.1.2: the API read conversion of the format (unorm: code/255)"),
    "hitdist": ("stored", "hitdist", "62.2 / 62.4: type 42 channel 0 as read; attached sets smv"),
    "colour_scale": (1.0, "float", "58.4 / 63.2: pre-exposure and exposure scale not used; F's own exposure (58)"),
    "depth_inverted": ("capture", ("capture", "0", "1"), "61.3: only the constants' byte == 1 at creation"),
    "mv_jittered": ("capture", ("capture", "0", "1"), "61.3: only the constants' byte == 1 at creation"),
    "mv_scale": ((1.0, 1.0), "pair", "61.4 step 4: ms = (scale.x * Wr, scale.y * Hr)"),
    "jitter_sign": ((1.0, 1.0), "pair", "61.4 step 5 / 62.3 / 52.8: jitter in render pixels, no sign conversion"),
    "colour_probe": ("none", ("none", "grey", "swap_rb", "rotate", "zero_r", "zero_g", "zero_b"),
                     "diagnostic, not a convention: the colour input replaced "
                     "by its luminance (R = G = B) / its R and B exchanged"),
}


def conventions(**kw):
    """A complete convention set: the defaults (the release proxy's behaviour) with the given overrides."""
    c = {k: v[0] for k, v in CONVENTIONS.items()}
    for k, v in kw.items():
        if k not in c:
            raise CaptureError(f"unknown convention {k}")
        kind = CONVENTIONS[k][1]
        if kind == "float":
            v = float(v)
        elif kind == "pair":
            v = tuple(float(x) for x in (v.split(",") if isinstance(v, str) else v))
            if len(v) != 2:
                raise CaptureError(f"{k}: two values expected")
        elif kind == "hitdist":
            if v not in ("stored", "drop"):
                v = float(v)
        elif str(v) not in kind:
            raise CaptureError(f"{k}: {v!r} is not one of {kind}")
        else:
            v = str(v)
        c[k] = v
    return c


def changed(conv):
    """The conventions that differ from the defaults, as 'name=value' strings."""
    d = conventions()
    return [f"{k}={conv[k]}" for k in CONVENTIONS if conv[k] != d[k]]


def _rgba(v):
    if v.shape[2] == 3:
        v = np.concatenate([v, np.ones(v.shape[:2] + (1,), np.float32)], axis=2)
    return v


def modified(frame, t, role, conv):
    """The values of input type t for this role under the conventions, as float32 [h][w][c], or None when the
    conventions leave the captured bytes as they are (then for_host hands them on unchanged)."""
    h, data = frame["inputs"][t]
    k = conv["colour_scale"]
    if role == ROLE["COLOUR"] and (k != 1.0 or conv["colour_probe"] != "none"):
        v = _rgba(decode(h, data))
        if conv["colour_probe"] == "grey":
            y = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
            v[..., 0], v[..., 1], v[..., 2] = y, y, y
        elif conv["colour_probe"] == "swap_rb":
            v[..., [0, 2]] = v[..., [2, 0]]
        elif conv["colour_probe"] == "rotate":
            v[..., [0, 1, 2]] = v[..., [1, 2, 0]]
        elif conv["colour_probe"].startswith("zero_"):
            v[..., "rgb".index(conv["colour_probe"][-1])] = 0.0
        v[..., :3] *= np.float32(k)
        return v
    if role == ROLE["BEFORE_TRANSPARENCY"] and k != 1.0:
        v = _rgba(decode(h, data))
        v[..., :3] *= np.float32(k)
        return v
    if role == ROLE["G1"] and t == BT["COLOUR_BEFORE_FOG"] and k != 1.0:
        v = decode(h, data).copy()
        v[..., :3] *= np.float32(k)
        return v
    if role == ROLE["LAYER"] and (conv["layer"] != "spec" or k != 1.0):
        v = _rgba(decode(h, data))
        v[..., :3] *= np.float32(k)
        if conv["layer"] == "additive":
            v[..., 3] = 0.0
        elif conv["layer"] == "straight":
            v[..., :3] *= v[..., 3:4]
        elif conv["layer"] == "invert":
            v[..., 3] = 1.0 - v[..., 3]
        return v
    if role == ROLE["NORMALS"] and (conv["normals"] != "stored" or conv["roughness"] != "stored"):
        v = decode(h, data).copy()
        if conv["normals"] == "view":   # row vectors (62.4): [x y z 0] W2V
            w = np.array(frame["meta"]["options"]["world_to_view"], np.float64).reshape(4, 4)[:3, :3]
            v[..., :3] = (v[..., :3].astype(np.float64) @ w).astype(np.float32)
        elif conv["normals"] in ("flat", "zero"):
            v[..., :3] = 0.0
            v[..., 2] = 1.0 if conv["normals"] == "flat" else 0.0
        if v.shape[2] >= 4 and conv["roughness"] != "stored":
            r = np.clip(v[..., 3], 0.0, None)
            v[..., 3] = {"invert": 1.0 - r, "square": r * r, "sqrt": np.sqrt(r),
                         "half": np.full_like(r, 0.5)}[conv["roughness"]]
        return v
    if role in (ROLE["DIFFUSE_ALBEDO"], ROLE["SPECULAR_ALBEDO"]) and conv["albedo"] in ("srgb", "grey", "one"):
        v = _rgba(decode(h, data))
        if conv["albedo"] == "srgb":
            v[..., :3] = srgb_to_linear(np.clip(v[..., :3], 0.0, 1.0))
        elif conv["albedo"] == "grey":
            y = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
            v[..., 0], v[..., 1], v[..., 2] = y, y, y
        else:
            v[..., :3] = 1.0
        return v
    if role == ROLE["SPECULAR_HIT_DIST"] and isinstance(conv["hitdist"], float):
        v = decode(h, data)[..., :1].copy()
        v *= np.float32(conv["hitdist"])
        return v
    return None


# ------------------------------------------------------------------------------------------------- the proxy's part
def route(frame, conv=None):
    """62.2 routing as the release proxy does it: [(role, buffer type)], depth_is_linear, layer origin.
    conv (conventions()) can drop the layer, G1 or the hit distance and swap the albedos (replay only)."""
    conv = conv or conventions()
    m, have = frame["meta"], frame["inputs"]
    opts = m["options"]
    packing = opts["normal_roughness_packing"]
    alpha = opts["version"] >= 2 and opts["alpha_upscaling"] == 1 and m["api_minor"] >= 14
    need = [BT["COLOUR_IN"], BT["MOTION"], BT["DIFFUSE_ALBEDO"], BT["SPECULAR_ALBEDO"]]
    need += [BT["NORMAL_ROUGHNESS"]] if packing == 1 else [BT["NORMALS"], BT["ROUGHNESS"]]
    missing = [t for t in need if t not in have]
    if BT["LINEAR_DEPTH"] not in have and BT["DEPTH"] not in have:
        missing.append(BT["DEPTH"])
    if missing:
        raise CaptureError(f"evaluate {m['evaluate']}: required input type(s) {missing} not captured")
    lin = BT["LINEAR_DEPTH"] in have
    sw = conv["albedo"] == "swap"
    uses = [(ROLE["COLOUR"], BT["COLOUR_IN"]), (ROLE["DEPTH"], BT["LINEAR_DEPTH"] if lin else BT["DEPTH"]),
            (ROLE["MOTION"], BT["MOTION"]),
            (ROLE["DIFFUSE_ALBEDO"], BT["SPECULAR_ALBEDO"] if sw else BT["DIFFUSE_ALBEDO"]),
            (ROLE["SPECULAR_ALBEDO"], BT["DIFFUSE_ALBEDO"] if sw else BT["SPECULAR_ALBEDO"])]
    if packing == 1:
        uses.append((ROLE["NORMALS"], BT["NORMAL_ROUGHNESS"]))
    else:
        uses += [(ROLE["NORMALS"], BT["NORMALS"]), (ROLE["ROUGHNESS"], BT["ROUGHNESS"])]
    for role, t in (("SPECULAR_MOTION", "SPECULAR_MOTION"), ("SPECULAR_HIT_DIST", "SPECULAR_HIT_DISTANCE"),
                    ("DISOCCLUSION", "DISOCCLUSION")):
        if BT[t] in have and not (t == "SPECULAR_HIT_DISTANCE" and conv["hitdist"] == "drop"):
            uses.append((ROLE[role], BT[t]))
    if conv["g1"] == "drop":
        pass
    elif BT["COLOUR_BEFORE_FOG"] in have:          # 41 replaces 58
        uses.append((ROLE["G1"], BT["COLOUR_BEFORE_FOG"]))
    elif BT["SSS_GUIDE"] in have:
        uses.append((ROLE["G1"], BT["SSS_GUIDE"]))
    if BT["DOF_GUIDE"] in have:
        uses.append((ROLE["DOF_GUIDE"], BT["DOF_GUIDE"]))
    if BT["COLOUR_BEFORE_TRANSPARENCY"] in have:
        uses.append((ROLE["BEFORE_TRANSPARENCY"], BT["COLOUR_BEFORE_TRANSPARENCY"]))
    has39, has51 = BT["COLOUR_BEFORE_PARTICLES"] in have, BT["TRANSPARENCY_LAYER"] in have
    if conv["layer"] == "drop":
        has39 = has51 = False
    if has39 or has51:                             # 39 replaces 51, also alone
        uses.append((ROLE["LAYER"], BT["COLOUR_BEFORE_PARTICLES"] if has39 else BT["TRANSPARENCY_LAYER"]))
        if BT["TRANSPARENCY_OPACITY"] in have:
            uses.append((ROLE["OPACITY"], BT["TRANSPARENCY_OPACITY"]))
    if alpha and BT["INPUT_ALPHA"] in have:
        uses.append((ROLE["INPUT_ALPHA"], BT["INPUT_ALPHA"]))
    layer_origin = (0, 0)
    if has51:                                      # the layer origin is always type 51's
        e = have[BT["TRANSPARENCY_LAYER"]][0]["extent"]
        layer_origin = (e["left"], e["top"])
    return uses, lin, layer_origin


def input_rect(h):
    e = h["extent"]
    if e["width"] == 0 or e["height"] == 0:
        return 0, 0, h["width"], h["height"]
    return e["left"], e["top"], e["width"], e["height"]


def render_size(frame):
    _, _, w, h = input_rect(frame["inputs"][BT["COLOUR_IN"]][0])
    return w, h


def output_size(frame):
    o = frame["meta"]["options"]
    if o["given"] and 0 < o["output_width"] < 0xFFFFFFFF and 0 < o["output_height"] < 0xFFFFFFFF:
        return o["output_width"], o["output_height"]
    out = frame["meta"].get("output")
    if out:
        return out["rect"][2], out["rect"][3]
    raise CaptureError(f"evaluate {frame['meta']['evaluate']}: no output size (no options, no output tag)")


def creation(frame, conv=None):
    """61.3 creation decisions from the first frame (the runtime's WgRrCreate as the test host's create line);
    conv can force depth inverted / motion vectors jittered (replay only)."""
    conv = conv or conventions()
    m = frame["meta"]
    o, c = m["options"], m["constants"]
    if c is None:
        raise CaptureError(f"evaluate {m['evaluate']}: no constants")
    wo, ho = output_size(frame)
    rw, rh = render_size(frame)
    alpha = o["version"] >= 2 and o["alpha_upscaling"] == 1 and m["api_minor"] >= 14
    _, lin, _ = route(frame, conv)
    di = (1 if c["depth_inverted"] == 1 else 0) if conv["depth_inverted"] == "capture" else int(conv["depth_inverted"])
    mj = (1 if c["mv_jittered"] == 1 else 0) if conv["mv_jittered"] == "capture" else int(conv["mv_jittered"])
    return dict(output=(wo, ho), render=(rw, rh), packing=o["normal_roughness_packing"],
                hdr=1 if o["colour_is_hdr"] == 1 else 0, depth_inverted=di,
                mv_jittered=mj, hw_depth=0 if lin else 1, alpha=1 if alpha else 0)


def checks(frame, create):
    """The proxy's rejections before a pass (61.4 steps 3 / 4, 62.3), as warnings."""
    m = frame["meta"]
    c = m["constants"]
    w = []
    rw, rh = render_size(frame)
    wo, ho = create["output"]
    if not (math.ceil(wo / 3) <= rw <= wo and math.ceil(ho / 3) <= rh <= ho):
        w.append(f"render size {rw}x{rh} outside 1/3..1 of the output {wo}x{ho} (the proxy returns 25)")
    if c["camera_motion_included"] == 0:
        w.append("camera motion not included (the proxy returns 25)")
    mh = frame["inputs"][BT["MOTION"]][0]
    _, _, mw, mhh = input_rect(mh)
    if mw > rw or mhh > rh:
        w.append("output-resolution motion vectors (the proxy returns 25)")
    return w


def frame_file(frame, path, settings=0, conv=None, no_reset=False):
    """The test host's frame input (RRF2, runtime/tests/e2e.py write_frame): one evaluate, under the conventions
    conv (conventions(); default: the release proxy's). no_reset: the capture's reset flag dropped (--passes)."""
    conv = conv or conventions()
    m = frame["meta"]
    c, o = m["constants"], m["options"]
    uses, lin, lo = route(frame, conv)
    rw, rh = render_size(frame)
    roles = []
    for role, t in uses:
        h, data = frame["inputs"][t]
        v = modified(frame, t, role, conv)
        if v is None:
            fmt, tex = for_host(h, data, depth=role == ROLE["DEPTH"])
        else:
            fmt, tex = {1: 41, 2: 16, 4: 2}[v.shape[2]], np.ascontiguousarray(v, np.float32).tobytes()
        x, y, _, _ = input_rect(h)
        roles.append((role, fmt, h["width"], h["height"], x, y, tex))
    jit = [float(np.float32(a * b)) for a, b in zip(c["jitter"], conv["jitter_sign"])]
    ms = [float(np.float32(a * b)) for a, b in zip(c["mv_scale"], conv["mv_scale"])]
    out = bytearray()
    out += struct.pack("<IIIIII", 0x32465252, rw, rh, 1 if c["reset"] == 1 and not no_reset else 0, settings,
                       1 if lin else 0)
    out += struct.pack("<4f", *jit, *ms)
    out += struct.pack("<16f", *c["view_to_clip"]) + struct.pack("<16f", *o["world_to_view"])
    out += struct.pack("<II", *lo)
    out += struct.pack("<I", len(roles))
    for role, fmt, w, h, x, y, tex in roles:
        out += struct.pack("<IIIIIIIIff", role, fmt, w, h, x, y, SHADER_READ, 0, 1.0, 0.0) + tex
    with open(path, "wb") as fh:
        fh.write(out)
    return [(role, t) for role, t in uses]


# ------------------------------------------------------------------------------------------------- images
def tonemap(rgb, exposure=1.0):
    """float RGB -> uint8 sRGB: x * exposure, Reinhard per channel, sRGB encoding; NaN / inf -> magenta."""
    x = np.asarray(rgb, np.float64) * exposure
    bad = ~np.isfinite(x).all(axis=2)
    x = np.where(np.isfinite(x), np.maximum(x, 0.0), 0.0)
    x = x / (1.0 + x)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    o = np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)
    o[bad] = (255, 0, 255)
    return o


def write_png(path, img):
    """uint8 [h][w][3] (or [h][w]) -> PNG (own minimal writer: zlib + CRC chunks)."""
    img = np.asarray(img, np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(png)


def read_png(path):
    """The images write_png writes (8-bit RGB, filter 0) -> uint8 [h][w][3]."""
    with open(path, "rb") as fh:
        b = fh.read()
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(b):
        n = struct.unpack(">I", b[pos:pos + 4])[0]
        tag, data = b[pos + 4:pos + 8], b[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif tag == b"IDAT":
            idat += data
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, 3)


def bilinear(img, wo, ho):
    """Bilinear resample (pixel centres, edges clamped) of float [h][w][c] to [ho][wo][c]."""
    hi, wi = img.shape[:2]
    xs = np.clip((np.arange(wo) + 0.5) * wi / wo - 0.5, 0, wi - 1)
    ys = np.clip((np.arange(ho) + 0.5) * hi / ho - 0.5, 0, hi - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, wi - 1), np.minimum(y0 + 1, hi - 1)
    fx, fy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    a = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    b = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def nearest(img, wo, ho):
    hi, wi = img.shape[:2]
    xs = np.minimum(((np.arange(wo) + 0.5) * wi / wo).astype(int), wi - 1)
    ys = np.minimum(((np.arange(ho) + 0.5) * hi / ho).astype(int), hi - 1)
    return img[ys][:, xs]


def side_by_side(a, b, gap=4):
    h = max(a.shape[0], b.shape[0])
    out = np.full((h, a.shape[1] + gap + b.shape[1], 3), 32, np.uint8)
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1] + gap:] = b
    return out


def luminance(rgb):
    x = np.where(np.isfinite(rgb), rgb, np.nan)
    return float(np.nanmean(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]))


def colour_input(frame):
    h, data = frame["inputs"][BT["COLOUR_IN"]]
    x, y, w, hh = input_rect(h)
    return decode(h, data)[y:y + hh, x:x + w, :3]


def preview(t, h, data):
    """A viewable picture of one input (uint8 RGB) and how it was mapped."""
    v = decode(h, data, depth=t in (BT["DEPTH"], BT["LINEAR_DEPTH"]))
    x, y, w, hh = input_rect(h)
    v = v[y:y + hh, x:x + w]
    ch = v.shape[2]
    fin = np.isfinite(v)
    if t in (BT["COLOUR_IN"], BT["COLOUR_BEFORE_PARTICLES"], BT["COLOUR_BEFORE_TRANSPARENCY"],
             BT["COLOUR_BEFORE_FOG"], BT["TRANSPARENCY_LAYER"]):
        return tonemap(v[..., :3]), "tonemapped colour"
    if t in (BT["NORMALS"], BT["NORMAL_ROUGHNESS"]):
        return np.clip(np.rint((np.nan_to_num(v[..., :3]) * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8), "xyz * 0.5 + 0.5"
    if t in (BT["DIFFUSE_ALBEDO"], BT["SPECULAR_ALBEDO"]) and ch >= 3:
        return np.clip(np.rint(np.nan_to_num(v[..., :3]) * 255), 0, 255).astype(np.uint8), "rgb"
    if t in (BT["MOTION"], BT["SPECULAR_MOTION"]) and ch >= 2:
        m = np.nan_to_num(v[..., :2]) * np.array([w, hh], np.float32)   # in pixels when the scale is 1, 1
        s = max(float(np.percentile(np.abs(m), 99)), 1e-6)
        rgb = np.stack([0.5 + 0.5 * m[..., 0] / s, 0.5 + 0.5 * m[..., 1] / s, np.full(m.shape[:2], 0.5)], axis=2)
        return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8), f"x, y / {s:.3g} (99th percentile)"
    c0 = np.where(fin[..., 0], v[..., 0], np.nan)
    if not np.isfinite(c0).any():
        return np.zeros(c0.shape + (3,), np.uint8), "no finite value"
    lo, hi = float(np.nanpercentile(c0, 1)), float(np.nanpercentile(c0, 99))
    g = np.clip((np.nan_to_num(c0, nan=lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.repeat(np.rint(g * 255).astype(np.uint8)[..., None], 3, axis=2), f"channel 0, {lo:.4g}..{hi:.4g}"


def psnr(a, b):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = float(d.mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


# ------------------------------------------------------------------------------------------------- run
def crop_rect(wo, ho, crop=None):
    """The crop of <n>_crop.png in output pixels (x, y, w, h), clipped to the output: the given one, else the centre
    quarter. Both halves of the crop image show exactly this rectangle (input nearest-upscaled | RR output)."""
    if crop is None:
        return wo // 4, ho // 4, max(wo // 2, 1), max(ho // 2, 1)
    x, y, w, h = (int(v) for v in crop)
    x, y = min(max(x, 0), wo - 1), min(max(y, 0), ho - 1)
    return x, y, max(1, min(w, wo - x)), max(1, min(h, ho - y))


def crop_image(cin, tout, exposure=1.0, crop=None):
    """<n>_crop.png: the same output-pixel rectangle (crop_rect) of the input colour cin (render size,
    nearest-upscaled to the output) | of the tonemapped RR output tout."""
    ho, wo = tout.shape[:2]
    cx, cy, cw, ch = crop_rect(wo, ho, crop)
    near = tonemap(nearest(cin, wo, ho), exposure)
    return side_by_side(near[cy:cy + ch, cx:cx + cw], tout[cy:cy + ch, cx:cx + cw])


def run(capture, model, out, runtime=DEFAULT_RUNTIME, host=DEFAULT_HOST, first=None, last=None, head="default",
        exposure=1.0, settings=0, previews=False, gpu=True, log=print, conv=None, crop=None, raw_frames=(), passes=1):
    """passes > 1: the frame list is fed that many times in a row on the same runtime instance; every pass after the
    first drops the capture's reset flag (temporal history carried across the wrap; the wrap itself is a camera jump
    the frame's motion vectors do not describe). Outputs of pass k > 1 are named frame_<n>_p<k>_*."""
    conv = conv or conventions()
    frames = load_capture(capture, first, last)
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "work")
    os.makedirs(os.path.join(work, "in"), exist_ok=True)
    create = creation(frames[0], conv)
    ch_ = changed(conv)
    log("conventions: " + (", ".join(ch_) + " (replay variant; the release proxy's defaults otherwise)" if ch_
                           else "the release proxy's (defaults)"))
    evals = [f["meta"]["evaluate"] for f in frames]
    gaps = [b - a for a, b in zip(evals, evals[1:]) if b - a != 1]
    log(f"capture {capture}: {len(frames)} frame(s), evaluates {evals[0]}..{evals[-1]}"
        + (f"; NOT consecutive (gaps {sorted(set(gaps))}): the temporal state skips the evaluates between" if gaps else ""))
    log(f"create: output {create['output'][0]}x{create['output'][1]} render {create['render'][0]}x{create['render'][1]} "
        f"packing {create['packing']} hdr {create['hdr']} depth_inverted {create['depth_inverted']} "
        f"mv_jittered {create['mv_jittered']} hw_depth {create['hw_depth']} alpha {create['alpha']}")
    job = [f"runtime {runtime}", f"outdir {os.path.join(work, 'host')}", f"output {OUT_FORMAT} 1",
           "create {} {} {} {} {} {} {} {} {} {}".format(*create["output"], *create["render"], create["packing"],
                                                         create["hdr"], create["depth_inverted"], create["mv_jittered"],
                                                         create["hw_depth"], create["alpha"])]
    routed = {}
    warnings = {}
    for f in frames:
        n = f["meta"]["evaluate"]
        w = checks(f, create)
        if w:
            warnings[n] = w
            for x in w:
                log(f"  evaluate {n}: warning: {x}")
        p = os.path.join(work, "in", f"frame_{n:06d}.bin")
        routed[n] = frame_file(f, p, settings, conv)
        if passes > 1:
            frame_file(f, os.path.join(work, "in", f"frame_{n:06d}_noreset.bin"), settings, conv, no_reset=True)
        job.append(f"frame {p}" + ("" if os.environ.get("WG_REPLAY_DUMP") else " nodump"))   # WG_REPLAY_DUMP=1: arena dumps
        if previews:
            for t, (h, data) in sorted(f["inputs"].items()):
                img, how = preview(t, h, data)
                write_png(os.path.join(out, f"frame_{n:06d}_in{t:02d}_{TYPE_NAME.get(t, 'type')}.png"), img)
    for k in range(2, passes + 1):
        log(f"pass {k}: evaluates {evals[0]}..{evals[-1]} again, reset flag dropped")
        for f in frames:
            job.append(f"frame {os.path.join(work, 'in', 'frame_%06d_noreset.bin' % f['meta']['evaluate'])} nodump")
    job += ["destroy", "end"]
    jf = os.path.join(work, "job.txt")
    with open(jf, "w") as fh:
        fh.write("\n".join(job) + "\n")
    roles_txt = ", ".join(f"{k}<-{t}" for k, t in sorted(
        {[r for r, v in ROLE.items() if v == role][0]: t for role, t in routed[evals[0]]}.items(), key=lambda x: ROLE[x[0]]))
    log(f"routing (first frame): {roles_txt}")
    summary = dict(capture=os.path.abspath(capture), runtime=runtime, host=host, head=head, exposure=exposure,
                   settings=settings, create=create, conventions={k: conv[k] for k in CONVENTIONS},
                   changed=ch_, evaluates=evals, warnings=warnings, passes=passes, frames=[])
    if not gpu:
        log("--no-gpu: frame files and job written, no GPU run")
        summary["gpu"] = False
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        return summary
    env = dict(os.environ, WG_RR_MODEL=model, WG_RR_LOG=os.path.join(out, "wronggpu-rr.log"))
    if not os.environ.get("WG_REPLAY_DUMP"):   # dump runs may set WG_RR_ALIAS=0 (head buffers kept to the end)
        env.pop("WG_RR_ALIAS", None)
    if head != "default":                # "default" = the runtime's default head (fast); "strict" = the exact head
        env["WG_RR_HEAD"] = "exact" if head == "strict" else head
    else:
        env.pop("WG_RR_HEAD", None)
    t0 = time.time()
    r = subprocess.run(_locked([host, jf]), capture_output=True, text=True,
                       env=env, timeout=7200)
    with open(os.path.join(work, "host.log"), "w") as fh:
        fh.write(r.stdout + "\n--- stderr\n" + r.stderr)
    if r.returncode != 0:
        raise CaptureError(f"rr_test_host failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    log(f"GPU run {time.time() - t0:.1f} s ({len(frames)} frames, head {head})")
    with open(os.path.join(work, "host", "meta.jsonl")) as fh:
        metas = [json.loads(l.replace("\\", "/")) for l in fh if l.strip()]   # the host writes paths unescaped
    if len(metas) != len(frames) * passes:
        raise CaptureError(f"the host ran {len(metas)} of {len(frames) * passes} frames "
                           f"(see {os.path.join(work, 'host.log')})")
    wo, ho = create["output"]
    for i, (k, f) in enumerate((k, f) for k in range(1, passes + 1) for f in frames):
        n = f["meta"]["evaluate"]
        tag = f"{n:06d}" if k == 1 else f"{n:06d}_p{k}"
        hm = metas[i]
        raw = np.fromfile(os.path.join(work, "host", f"frame_{i:03d}.out"), np.float16)
        tw, th = hm["out_tex"][0], hm["out_tex"][1]
        o = raw.reshape(th, tw, 4)[OUT_ORIGIN[1]:OUT_ORIGIN[1] + ho, OUT_ORIGIN[0]:OUT_ORIGIN[0] + wo].astype(np.float32)
        if conv["colour_scale"] != 1.0:     # the game's side of a pre-exposure: the output divided by the scale again
            o[..., :3] /= np.float32(conv["colour_scale"])
        if n in raw_frames:
            np.save(os.path.join(out, f"frame_{tag}_rr.npy"), o[..., :3].astype(np.float16))
        cin = colour_input(f)
        tin, tout = tonemap(cin, exposure), tonemap(o[..., :3], exposure)
        up = tonemap(bilinear(cin, wo, ho), exposure)
        if k == 1:
            write_png(os.path.join(out, f"frame_{tag}_input.png"), tin)
        write_png(os.path.join(out, f"frame_{tag}_rr.png"), tout)
        write_png(os.path.join(out, f"frame_{tag}_side.png"), side_by_side(up, tout))
        write_png(os.path.join(out, f"frame_{tag}_crop.png"), crop_image(cin, tout, exposure, crop))
        fin = float(np.isfinite(o[..., :3]).all(axis=2).mean())
        rec = dict(evaluate=n, run_pass=k, rc=hm["rc"], completed=hm["completed"], reset=hm["reset"], game_reset=hm["game_reset"],
                   output_finite_fraction=fin, input_mean_luminance=luminance(cin), output_mean_luminance=luminance(o[..., :3]),
                   psnr_vs_bilinear_input_db=psnr(tout, up))
        summary["frames"].append(rec)
        log(f"  evaluate {n}{'' if k == 1 else ' pass %d' % k}: rc {rec['rc']} completed {rec['completed']} reset {rec['reset']} finite {fin:.4f} "
            f"luminance in {rec['input_mean_luminance']:.4g} out {rec['output_mean_luminance']:.4g} "
            f"PSNR vs bilinear {rec['psnr_vs_bilinear_input_db']:.2f} dB")
    summary["gpu"] = True
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, allow_nan=True)
    log(f"PNGs and summary.json in {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="the capture folder (wronggpu-rr-capture with frame_<n> folders)")
    ap.add_argument("--model", help="model.wgcm for the runtime (path only; required unless --no-gpu)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime", default=DEFAULT_RUNTIME)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--frames", help="FIRST:LAST evaluate numbers (inclusive)")
    ap.add_argument("--head", choices=("default", "strict", "exact", "fast", "guard"), default="default")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--settings", type=int, default=0, help="WG_RR_SET_* bits handed to the runtime per frame")
    ap.add_argument("--previews", action="store_true", help="also a PNG of every input of every frame")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--crop", help="X,Y,W,H in output pixels for <n>_crop.png (default: the centre quarter)")
    ap.add_argument("--raw", default="", help="evaluates (comma list) whose RR output is also saved as float16 .npy")
    ap.add_argument("--passes", type=int, default=1,
                    help="feed the frame list this many times on one runtime instance; passes after the first drop "
                         "the capture's reset flag (history carried across the wrap)")
    g = ap.add_argument_group("convention switches (replay only; defaults = the release proxy's behaviour)")
    for k, (d, kind, rule) in CONVENTIONS.items():
        g.add_argument("--" + k.replace("_", "-"), dest="conv_" + k, default=None,
                       help=f"default {d if not isinstance(d, tuple) else ','.join(str(x) for x in d)}; "
                            + (f"one of {', '.join(kind)}; " if isinstance(kind, tuple) else "")
                            + f"spec: {rule}")
    a = ap.parse_args()
    first = last = None
    if a.frames:
        p = a.frames.split(":")
        first = int(p[0]) if p[0] else None
        last = int(p[1]) if len(p) > 1 and p[1] else first
    if not a.no_gpu and not a.model:
        ap.error("--model is required for a GPU run")
    try:
        conv = conventions(**{k: getattr(a, "conv_" + k) for k in CONVENTIONS if getattr(a, "conv_" + k) is not None})
        crop = [int(v) for v in a.crop.split(",")] if a.crop else None
        if crop is not None and len(crop) != 4:
            ap.error("--crop needs X,Y,W,H")
        raw = {int(v) for v in a.raw.split(",") if v.strip()}
        run(a.capture, a.model, a.out, a.runtime, a.host, first, last, a.head, a.exposure, a.settings, a.previews,
            not a.no_gpu, conv=conv, crop=crop, raw_frames=raw, passes=max(1, a.passes))
    except CaptureError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
