# SPDX-License-Identifier: MIT
"""Seeded probe generator of the equivalence tests (clean/EQUIVALENCE-PROTOCOL.md, part 3; part "Tools", item 2).

Writes synthetic game inputs, one directory per probe case, the same bytes on every machine for the same parameters:
only integer hashing, IEEE additions, multiplications and divisions in float64, and correctly rounded casts are used
(no transcendental functions, no library random-number streams), so both sides feed identical inputs to their model.

    python -m clean.equiv.probes --out DIR [--model m|rr] [--render 1707x960] [--output 2560x1440] [--frames 64]
                                 [--seed 1] [--cases a,b,...] [--jitter-phases N] [--no-jitter]
    python -m clean.equiv.probes --verify DIR       # regenerate in memory from DIR/manifest.json, compare every hash
    python -m clean.equiv.probes --list [--model rr]

DIR/manifest.json records the generator version, every parameter, the input conventions, per frame the jitter, reset
flag, pre-exposure and every file with its dtype, shape and SHA-256, and one digest over all files. Comparing the
digest is enough to know both sides hold the same bytes.
"""
import argparse
import hashlib
import io
import json
import os
import sys
from fractions import Fraction

import numpy as np

VERSION = 2          # 2: the RR colour-guide cases (page 52), the specular-motion case, camera matrices in RR cases

CONVENTIONS = {
    "color": "linear HDR RGB, float16, H x W x 3, already multiplied by the frame's pre_exposure",
    "depth": "float32, H x W; 0 = near, 1 = far; with depth_inverted the stored value is 1 - depth (1 = near)",
    "mv": "float16, H x W x 2 (x, y) in render pixels: the previous-frame position minus the current position of the "
          "surface seen in the pixel; the jitter is not included",
    "jitter": "the sample's offset from the render pixel centre in render pixels, (x, y), each in [-0.5, 0.5): "
              "Halton bases 2 and 3 minus 0.5, phase (frame mod jitter_phases) + 1",
    "sampling": "every scene is defined in output-pixel coordinates and point-sampled at render pixel centre + jitter, "
                "scaled by output / render size (the 'point' probe is defined in render pixels)",
    "subrect": "when the case has a subrect, every input texture is larger than the render size and the frame sits at "
               "origin (x, y); outside it colour holds 1000, depth 0 and motion (100, 100)",
    "reset": "reset = the game asks for the history to start over on this frame",
    "colour_guides": "page 52 functional names; color_before_transparency (RGB of RGBA16F, or R11G11B10F packed in a "
                     "uint32: R bits 0-10, G 11-21, B 22-31), sss_guide and dof_guide (channel 0 of R16F, or R8_UNORM "
                     "as uint8 codes), color_before_fog (red channel of RGBA16F); all read with the colour input's "
                     "origin; an absent guide is the neutral 0 (reference case: rr_guides_all); 'declared_origin' is "
                     "the origin the game would declare for the guide and must not move what is read",
    "rr_guides": "albedo_diffuse, albedo_specular (float16 x 3), normals (float16 x 3, unit length, view space, +z "
                 "towards the camera), roughness (float16), hit_distance (float16, output-pixel units), "
                 "disocclusion_mask (uint8 0 / 1: 1 where background that was covered on the previous frame is now "
                 "visible), transparency_layer (float16 x 3) and transparency_opacity (float16); packed roughness = "
                 "normal_roughness (float16 x 4: normal x, y, z, roughness) instead of normals + roughness",
}

# Colour guides of page 52 (functional names), Brief 5 (e): name -> (formats, the first is the main one; what the
# runtime reads). Every colour guide is read with the colour input's origin.
COLOUR_GUIDES = {
    "color_before_transparency": (("RGBA16F", "R11G11B10F"), "RGB"),
    "sss_guide": (("R16F", "R8_UNORM"), "channel 0"),
    "color_before_fog": (("RGBA16F",), "red channel (replaces the subsurface-scattering guide; both attached: fog wins)"),
    "dof_guide": (("R16F", "R8_UNORM"), "channel 0"),
}
VALUE_CLASSES = {
    "typical": "values in [0.05, 0.95)",
    "negative": "values in [-4, -0.01] (negatives use their own scale; results clamp)",
    "zero": "all values 0 (the same as the neutral value of an absent guide)",
    "small": "values in about [1.2e-5, 2.4e-4): binary16 subnormals (below 2^-14) and small normals",
    "large": "values in [1024, 61024)",
    "nan": "typical values with NaN in the block x in [ow/4, ow/2), y in [oh/4, oh/2) of the output (NaN reads as 0)",
}
GUIDE_OWN_ORIGIN = (5, 3)       # declared origin of a guide in the own-origin cases (x, y); must not change the output
GUIDE_PAD_FILL = {"float": 7.0, "unorm": 255}
CAMERA = {
    "convention": "row-major 4x4, column vectors: clip = proj . view . world; D3D clip depth [0, 1]; the camera is still "
                  "in every RR case (the scene's motion is the foreground's), so the previous matrices equal the current",
    "view": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    "proj_focal": 2.0, "proj_near": 0.125, "proj_far": 1024.0,
}


def _proj(render):
    f, n, fa = CAMERA["proj_focal"], CAMERA["proj_near"], CAMERA["proj_far"]
    a = render[0] / render[1]
    return [[f / a, 0.0, 0.0, 0.0], [0.0, f, 0.0, 0.0], [0.0, 0.0, fa / (fa - n), -n * fa / (fa - n)],
            [0.0, 0.0, 1.0, 0.0]]


def _fmt_word(fmt):
    return fmt.lower().replace("_", "")


def colour_guide_cases():
    """-> [(name, behaviour, spec)], spec = {"guides": {guide: (format, value class, declared origin)},
    "expect_same_output_as": case or None}."""
    out = []
    for g, (fmts, use) in COLOUR_GUIDES.items():
        main = fmts[0]
        for cls, what in VALUE_CLASSES.items():
            out.append((f"rr_cg_{g}_{_fmt_word(main)}_{cls}", f"{g} attached ({main}, {what}): colour-guide routing",
                        {"guides": {g: (main, cls, None)}}))
        for f2 in fmts[1:]:
            out.append((f"rr_cg_{g}_{_fmt_word(f2)}_typical", f"{g} attached ({f2}, typical values)",
                        {"guides": {g: (f2, "typical", None)}}))
        out.append((f"rr_cg_{g}_own_origin", f"{g} ({main}, typical) with a declared own origin {GUIDE_OWN_ORIGIN} "
                    "different from the colour's: read with the colour's origin, so the output must not change",
                    {"guides": {g: (main, "typical", GUIDE_OWN_ORIGIN)},
                     "expect_same_output_as": f"rr_cg_{g}_{_fmt_word(main)}_typical"}))
    out.append(("rr_cg_fog_and_sss", "color_before_fog and sss_guide both attached: fog wins, so the output must equal "
                "the fog-only case",
                {"guides": {"color_before_fog": ("RGBA16F", "typical", None), "sss_guide": ("R16F", "typical", None)},
                 "expect_same_output_as": "rr_cg_color_before_fog_rgba16f_typical"}))
    out.append(("rr_opt_spec_mv", "optional: as rr_guides_all with specular motion vectors (channels 0-1, read with "
                "the motion vectors' origin); without them the specular hit distance triggers the second-motion "
                "reconstruction, which uses the camera matrices of the manifest", {"guides": {}, "spec_mv": True}))
    return out


def small_float_bits(v, mbits):
    """Unsigned small float of R11G11B10F (5 exponent bits, bias 15, mbits mantissa bits): round to nearest even,
    negatives -> 0, above the largest finite -> the largest finite, NaN -> a NaN code. -> uint32 bit fields."""
    v = np.asarray(v, np.float64)
    nan = np.isnan(v)
    a = np.where(nan | (v < 0), 0.0, v)
    top = (2.0 - 2.0 ** -mbits) * 2.0 ** 15
    _, e = np.frexp(np.where(a > 0, a, 1.0))
    ex = np.maximum(e - 1, -14)
    sp = np.ldexp(1.0, ex - mbits)
    r = np.minimum(np.round(a / sp) * sp, top)
    _, e2 = np.frexp(np.where(r > 0, r, 1.0))
    E = e2 - 1
    sub = r < 2.0 ** -14
    field = np.where(sub, 0, E + 15)
    mant = np.where(sub, r / 2.0 ** (-14 - mbits), (r / np.ldexp(1.0, E) - 1.0) * 2 ** mbits)
    bits = (field.astype(np.uint32) << np.uint32(mbits)) | np.rint(mant).astype(np.uint32)
    return np.where(nan, np.uint32((31 << mbits) | (1 << (mbits - 1))), bits).astype(np.uint32)


def small_float_value(bits, mbits):
    b = np.asarray(bits, np.int64)
    f, m = b >> mbits, b & ((1 << mbits) - 1)
    v = np.where(f == 0, m * 2.0 ** (-14 - mbits), (1.0 + m / 2.0 ** mbits) * np.ldexp(1.0, f - 15))
    return np.where(f == 31, np.where(m == 0, np.inf, np.nan), v)


def pack_r11g11b10f(rgb):
    """[..., 3] float64 -> uint32 [...]: R in bits 0-10, G in bits 11-21, B in bits 22-31."""
    r, g, b = small_float_bits(rgb[..., 0], 6), small_float_bits(rgb[..., 1], 6), small_float_bits(rgb[..., 2], 5)
    return (r | (g << np.uint32(11)) | (b << np.uint32(22))).astype(np.uint32)


RR_GUIDES = ("albedo_diffuse", "albedo_specular", "normals", "roughness", "hit_distance", "disocclusion_mask",
             "transparency_layer", "transparency_opacity")

SPEEDS = (("0", Fraction(0)), ("0.25", Fraction(1, 4)), ("1", Fraction(1)), ("4", Fraction(4)))


def case_list(model="m"):
    """-> [(name, behaviour it isolates)] in a fixed order."""
    c = [("flat_step", "flat colour, then one step in brightness: exposure response, history adaptation"),
         ("point", "one bright render pixel, still, then moving 1 px per frame: footprint, ghosting decay"),
         ("pan", "textured scene, camera moving 1 output px per frame: steady motion (reference for other probes)")]
    for o, on in (("h", "horizontal"), ("v", "vertical"), ("d", "diagonal")):
        for sn, _ in SPEEDS:
            c.append((f"edge_{o}_v{sn}", f"{on} edge moving {sn} output px per frame: edges, motion handling"))
    c += [("checker", "1-output-px checkerboard under the jitter sequence: detail accumulation"),
          ("stripes", "1-output-px vertical stripes under the jitter sequence: detail accumulation"),
          ("disocclusion", "textured foreground moving over a still background: disocclusion, history rejection"),
          ("bad_mv", "camera moving, motion vectors wrong by (+3, -2) render px in one region: robustness"),
          ("reset_k", "camera moving, reset requested on the middle frame: history invalidation"),
          ("cut_no_reset", "the scene changes on the middle frame without a reset: history invalidation"),
          ("exposure_dark", "the pan scene 1/1024 as bright: value range"),
          ("exposure_bright", "the pan scene 1024 times as bright: value range"),
          ("pre_exposure_4", "the pan scene with pre-exposure 4: pre-exposure handling"),
          ("depth_inverted", "the disocclusion scene with inverted depth: input conventions"),
          ("subrect_origin", "the pan scene inside larger textures at origin (17, 9): input conventions")]
    if model == "rr":
        c.append(("rr_guides_all", "every guide present (separate roughness): guide routing reference"))
        for g in RR_GUIDES:
            c.append((f"rr_no_{g}", f"as rr_guides_all without {g}: guide routing"))
        c.append(("rr_roughness_packed", "as rr_guides_all with roughness packed into the normals: guide routing"))
        c += [(n, b) for n, b, _ in colour_guide_cases()]
    return c


# ---------------------------------------------------------------------------------------------------- hashing

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)


def _mix(z):
    with np.errstate(over="ignore"):
        z = z + _GOLD
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        return z ^ (z >> np.uint64(31))


def u01(seed, stream, ix, iy):
    """Deterministic uniform [0, 1) per integer lattice point (ix, iy may be negative)."""
    ix = np.asarray(ix, np.int64).astype(np.uint64)
    iy = np.asarray(iy, np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        k = _mix(np.uint64(seed) * _C1 + np.uint64(stream))
        h = _mix(k ^ _mix(ix ^ _mix(iy + _GOLD)))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def value_noise(seed, stream, x, y, cell):
    """Smooth noise in [0, 1) on a lattice of `cell` output pixels (cell a power of two: exact divisions)."""
    gx, gy = np.floor(x / cell), np.floor(y / cell)
    fx, fy = x / cell - gx, y / cell - gy
    sx, sy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    ix, iy = gx.astype(np.int64), gy.astype(np.int64)
    v00, v10 = u01(seed, stream, ix, iy), u01(seed, stream, ix + 1, iy)
    v01, v11 = u01(seed, stream, ix, iy + 1), u01(seed, stream, ix + 1, iy + 1)
    top = v00 + (v10 - v00) * sx
    bot = v01 + (v11 - v01) * sx
    return top + (bot - top) * sy


def texture(seed, stream, x, y):
    """RGB texture in [0.05, 0.95): two octaves of value noise per channel."""
    ch = [0.05 + 0.9 * (0.75 * value_noise(seed, stream * 8 + c, x, y, 8.0) +
                        0.25 * value_noise(seed, stream * 8 + 4 + c, x, y, 2.0)) for c in range(3)]
    return np.stack(ch, axis=-1)


def halton(i, b):
    f, r = Fraction(1), Fraction(0)
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


# ---------------------------------------------------------------------------------------------------- scenes

class Ctx:
    def __init__(self, seed, render, output, frames, jitter_phases, jitter=True):
        self.seed = seed
        self.rw, self.rh = render
        self.ow, self.oh = output
        self.frames = frames
        self.phases = jitter_phases
        self.jitter_on = jitter
        self.k = frames // 2
        self.j, self.i = np.mgrid[0:self.rh, 0:self.rw]          # render pixel row, column

    def jitter(self, t):
        if not self.jitter_on:
            return (Fraction(0), Fraction(0))
        p = t % self.phases + 1
        return (halton(p, 2) - Fraction(1, 2), halton(p, 3) - Fraction(1, 2))

    def samples(self, t):
        jx, jy = self.jitter(t)
        X = (self.i + 0.5 + float(jx)) * (self.ow / self.rw)
        Y = (self.j + 0.5 + float(jy)) * (self.oh / self.rh)
        return X, Y

    def mv_from_velocity(self, vx, vy):
        """Velocity of the surface in output px per frame -> the motion-vector convention (render px)."""
        return np.stack([np.full((self.rh, self.rw), -float(vx) * self.rw / self.ow),
                         np.full((self.rh, self.rw), -float(vy) * self.rh / self.oh)], axis=-1)


def _flat(c, val):
    return np.full((c.rh, c.rw, 3), val)


def scene_pan(c, t, stream=1, vx=1, vy=0):
    X, Y = c.samples(t)
    col = texture(c.seed, stream, X - vx * t, Y - vy * t)
    depth = 0.5 + 0.25 * value_noise(c.seed, stream * 8 + 7, X - vx * t, Y - vy * t, 16.0)
    return {"color": col, "depth": depth, "mv": c.mv_from_velocity(vx, vy)}


def scene_disocclusion(c, t, v=2):
    X, Y = c.samples(t)
    w = c.ow / 4.0
    x0 = c.ow / 8.0
    y_in = (Y >= c.oh / 4.0) & (Y < 3.0 * c.oh / 4.0)

    def covered(tt):
        return y_in & (X >= x0 + v * tt) & (X < x0 + v * tt + w)

    fg = covered(t)
    col = np.where(fg[..., None], texture(c.seed, 3, X - v * t, Y), texture(c.seed, 2, X, Y))
    depth = np.where(fg, 0.3, 0.9)
    mv = np.where(fg[..., None], c.mv_from_velocity(v, 0), 0.0)
    out = {"color": col, "depth": depth, "mv": mv, "_fg": fg}
    out["_disoccluded"] = (covered(t - 1) & ~fg) if t > 0 else np.zeros_like(fg)
    out["_X"], out["_Y"] = X, Y
    return out


def scene(case, c, t):
    """-> dict of float64 arrays (and per-frame scalars) before casting."""
    s, k = c.seed, c.k
    meta = {"reset": False, "pre_exposure": 1.0}
    if case == "flat_step":
        d = {"color": _flat(c, 0.18 if t < k else 0.36), "depth": np.full((c.rh, c.rw), 0.5),
             "mv": c.mv_from_velocity(0, 0)}
    elif case == "point":
        px = c.rw // 3 + max(0, t - k + 1) if t >= k else c.rw // 3
        col = _flat(c, 0.02)
        col[c.rh // 2, px % c.rw] = 64.0
        mv = c.mv_from_velocity(0, 0)
        if t >= k:
            mv[c.rh // 2, px % c.rw] = (-1.0, 0.0)               # render px: moved +1 since the previous frame
        d = {"color": col, "depth": np.full((c.rh, c.rw), 0.5), "mv": mv}
    elif case in ("pan", "reset_k", "exposure_dark", "exposure_bright", "pre_exposure_4", "subrect_origin"):
        d = scene_pan(c, t)
        if case == "reset_k":
            meta["reset"] = t == k
        elif case == "exposure_dark":
            d["color"] = d["color"] / 1024.0
        elif case == "exposure_bright":
            d["color"] = d["color"] * 1024.0
        elif case == "pre_exposure_4":
            meta["pre_exposure"] = 4.0
            d["color"] = d["color"] * 4.0
    elif case == "cut_no_reset":
        d = scene_pan(c, t, stream=1 if t < k else 5)
    elif case.startswith("edge_"):
        _, o, sp = case.split("_")
        v = dict(SPEEDS)[sp[1:]]
        X, Y = c.samples(t)
        if o == "h":
            on = Y >= c.oh / 2.0 + 0.3 + float(v * t)
            mv = c.mv_from_velocity(0, v)
        elif o == "v":
            on = X >= c.ow / 2.0 + 0.3 + float(v * t)
            mv = c.mv_from_velocity(v, 0)
        else:
            on = (X - float(v * t)) + Y >= (c.ow + c.oh) / 2.0 + 0.3
            mv = c.mv_from_velocity(v, 0)
        d = {"color": np.where(on[..., None], 1.0, 0.05) * np.ones(3), "depth": np.full((c.rh, c.rw), 0.5), "mv": mv}
    elif case in ("checker", "stripes"):
        X, Y = c.samples(t)
        par = (np.floor(X) + (np.floor(Y) if case == "checker" else 0)) % 2
        d = {"color": np.where(par[..., None] == 0, 0.9, 0.1) * np.ones(3), "depth": np.full((c.rh, c.rw), 0.5),
             "mv": c.mv_from_velocity(0, 0)}
    elif case in ("disocclusion", "depth_inverted") or case.startswith("rr_"):
        d = scene_disocclusion(c, t)
    elif case == "bad_mv":
        d = scene_pan(c, t)
        reg = (c.i >= c.rw // 4) & (c.i < c.rw // 2) & (c.j >= c.rh // 4) & (c.j < c.rh // 2)
        d["mv"] = d["mv"] + np.where(reg[..., None], np.array([3.0, -2.0]), 0.0)
    else:
        raise KeyError(f"unknown probe case {case!r}")
    return d, meta


def frame_arrays(case, c, t, model):
    """-> (ordered dict name -> final array, per-frame record without files)."""
    d, meta = scene(case, c, t)
    depth = d["depth"]
    if case == "depth_inverted":
        depth = 1.0 - depth
    out = {"color": d["color"].astype(np.float16), "depth": depth.astype(np.float32), "mv": d["mv"].astype(np.float16)}
    if model == "rr" and case.startswith("rr_"):
        X, Y, fg = d["_X"], d["_Y"], d["_fg"]
        s = c.seed
        nb = np.broadcast_to(np.array([0.0, 0.0, 1.0]), fg.shape + (3,))
        nf = np.broadcast_to(np.array([0.6, 0.0, 0.8]), fg.shape + (3,))
        rough = 0.05 + 0.9 * value_noise(s, 41, X, Y, 8.0)
        g = {"albedo_diffuse": 0.1 + 0.8 * texture(s, 6, X, Y),
             "albedo_specular": 0.04 + 0.2 * texture(s, 7, X, Y),
             "normals": np.where(fg[..., None], nf, nb),
             "roughness": rough,
             "hit_distance": 1.0 + 64.0 * value_noise(s, 42, X, Y, 16.0),
             "disocclusion_mask": d["_disoccluded"],
             "transparency_layer": np.broadcast_to(np.array([0.8, 0.4, 0.2]), fg.shape + (3,)),
             "transparency_opacity": None}
        q = (X >= c.ow / 2.0) & (X < 3.0 * c.ow / 4.0) & (Y >= c.oh / 8.0) & (Y < 3.0 * c.oh / 8.0)
        g["transparency_opacity"] = np.where(q, 0.5, 0.0)
        g["transparency_layer"] = np.where(q[..., None], g["transparency_layer"], 0.0)
        drop = case[len("rr_no_"):] if case.startswith("rr_no_") else None
        for name in RR_GUIDES:
            if name == drop:
                continue
            if case == "rr_roughness_packed" and name in ("normals", "roughness"):
                continue
            out[name] = g[name].astype(np.uint8 if name == "disocclusion_mask" else np.float16)
        if case == "rr_roughness_packed":
            out["normal_roughness"] = np.concatenate([g["normals"], rough[..., None]], axis=-1).astype(np.float16)
        spec = dict((n, sp) for n, _, sp in colour_guide_cases()).get(case)
        if spec:
            for gname, (fmt, cls, origin) in spec["guides"].items():
                out[gname] = colour_guide(gname, fmt, cls, origin, c, d)
            if spec.get("spec_mv"):
                out["specular_motion_vectors"] = (0.5 * d["mv"]).astype(np.float16)
    if case == "subrect_origin":
        ox, oy = SUBRECT_ORIGIN
        tw, th = c.rw + ox + SUBRECT_PAD[0], c.rh + oy + SUBRECT_PAD[1]
        fills = {"color": 1000.0, "depth": 0.0, "mv": 100.0}
        for name in list(out):
            a = out[name]
            big = np.full((th, tw) + a.shape[2:], fills.get(name, 0.0), a.dtype)
            big[oy:oy + c.rh, ox:ox + c.rw] = a
            out[name] = big
    jx, jy = c.jitter(t)
    rec = {"frame": t, "jitter": [float(jx), float(jy)], "reset": bool(meta["reset"]),
           "pre_exposure": meta["pre_exposure"]}
    return out, rec


SUBRECT_ORIGIN = (17, 9)
SUBRECT_PAD = (5, 7)


def colour_guide(name, fmt, cls, origin, c, d):
    """One colour guide of page 52 in its texture format (RGBA16F -> float16 [H, W, 4]; R11G11B10F -> uint32 [H, W];
    R16F -> float16 [H, W]; R8_UNORM -> uint8 [H, W]). With a declared own origin (x, y) the texture is that much
    larger on the right / bottom (filled with 7.0, or 255 for unorm) and the guide's data stays at the colour's origin."""
    X, Y, s = d["_X"], d["_Y"], c.seed
    stream = {"color_before_transparency": 9, "sss_guide": 10, "color_before_fog": 11, "dof_guide": 12}[name]
    if name in ("color_before_transparency", "color_before_fog"):
        t = texture(s, stream, X, Y)                                   # [H, W, 3] in [0.05, 0.95)
    else:
        t = 0.05 + 0.9 * value_noise(s, stream * 8, X, Y, 8.0)        # [H, W]
    v = {"typical": t, "negative": -(0.01 + (3.99 / 0.9) * (t - 0.05)), "zero": 0.0 * t,
         "small": t * 2.0 ** -12, "large": 1024.0 + (60000.0 / 0.9) * (t - 0.05),
         "nan": t}[cls]
    if cls == "nan":
        blk = (X >= c.ow / 4.0) & (X < c.ow / 2.0) & (Y >= c.oh / 4.0) & (Y < c.oh / 2.0)
        v = np.where(blk[..., None] if v.ndim == 3 else blk, np.nan, v)
    if fmt == "RGBA16F":
        a = np.concatenate([v, np.ones(v.shape[:2] + (1,))], axis=-1).astype(np.float16)
    elif fmt == "R11G11B10F":
        a = pack_r11g11b10f(v)
    elif fmt == "R16F":
        a = v.astype(np.float16)
    elif fmt == "R8_UNORM":
        a = np.rint(np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        raise KeyError(fmt)
    if origin:
        ox, oy = origin
        fill = GUIDE_PAD_FILL["unorm"] if a.dtype == np.uint8 else GUIDE_PAD_FILL["float"]
        if a.dtype == np.uint32:
            fill = int(pack_r11g11b10f(np.full(3, 7.0)))
        big = np.full((a.shape[0] + oy, a.shape[1] + ox) + a.shape[2:], fill, a.dtype)
        big[:a.shape[0], :a.shape[1]] = a
        a = big
    return a


# ---------------------------------------------------------------------------------------------------- output

def _npy_bytes(a):
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(a), allow_pickle=False)
    return b.getvalue()


def default_phases(render, output):
    return max(8, round(8 * (output[0] * output[1]) / (render[0] * render[1])))


def generate(params, out_dir=None):
    """Builds every case; writes the files and manifest.json when out_dir is given. -> manifest dict."""
    p = dict(params)
    model = p["model"]
    names = [n for n, _ in case_list(model)]
    sel = p.get("cases") or names
    unknown = [n for n in sel if n not in names]
    if unknown:
        raise KeyError(f"unknown case(s) for model {model}: {', '.join(unknown)}")
    beh = dict(case_list(model))
    c = Ctx(p["seed"], p["render"], p["output"], p["frames"], p["jitter_phases"], p["jitter"])
    cases, lines = [], []
    for name in [n for n in names if n in sel]:
        crec = {"name": name, "behaviour": beh[name], "frames": []}
        if name == "depth_inverted":
            crec["depth_inverted"] = True
        spec = dict((n, sp) for n, _, sp in colour_guide_cases()).get(name)
        if model == "rr" and name.startswith("rr_"):
            crec["camera"] = dict(view=CAMERA["view"], proj=_proj(p["render"]), view_prev=CAMERA["view"],
                                  proj_prev=_proj(p["render"]), convention=CAMERA["convention"])
        if spec:
            crec["colour_guides"] = {g: {"format": f, "values": cl, "use": COLOUR_GUIDES[g][1],
                                         "declared_origin": list(o) if o else [0, 0],
                                         "read_origin": "the colour input's origin (0, 0)"}
                                     for g, (f, cl, o) in spec["guides"].items()}
            crec["absent_reference"] = "rr_guides_all"
            if spec.get("expect_same_output_as"):
                crec["expect_same_output_as"] = spec["expect_same_output_as"]
            if spec.get("spec_mv"):
                crec["specular_motion_vectors"] = {"format": "RG16F", "read_origin": "the motion vectors' origin (0, 0)"}
        if name == "subrect_origin":
            crec["subrect"] = {"origin": list(SUBRECT_ORIGIN), "size": [c.rw, c.rh],
                               "texture": [c.rw + SUBRECT_ORIGIN[0] + SUBRECT_PAD[0],
                                           c.rh + SUBRECT_ORIGIN[1] + SUBRECT_PAD[1]]}
        if out_dir:
            os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        for t in range(c.frames):
            arrs, rec = frame_arrays(name, c, t, model)
            rec["files"] = {}
            for fname, a in arrs.items():
                rel = f"{name}/f{t:04d}_{fname}.npy"
                blob = _npy_bytes(a)
                h = hashlib.sha256(blob).hexdigest()
                rec["files"][fname] = {"file": rel, "dtype": str(a.dtype), "shape": list(a.shape), "sha256": h}
                lines.append(f"{rel}:{h}")
                if out_dir:
                    with open(os.path.join(out_dir, rel), "wb") as f:
                        f.write(blob)
            crec["frames"].append(rec)
        cases.append(crec)
    man = {"generator": "clean/equiv/probes.py", "version": VERSION,
           "params": {"model": model, "seed": p["seed"], "render": list(p["render"]), "output": list(p["output"]),
                      "frames": p["frames"], "jitter_phases": p["jitter_phases"], "jitter": p["jitter"],
                      "cases": list(sel) if p.get("cases") else None},
           "conventions": CONVENTIONS, "cases": cases,
           "digest": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    if out_dir:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1)
    return man


def verify(out_dir):
    """Regenerates from the directory's manifest and compares the manifest's and the files' hashes.
    -> (ok, list of problem strings)."""
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    if man.get("version") != VERSION:
        return False, [f"manifest version {man.get('version')}, this generator {VERSION}"]
    pr = man["params"]
    params = {"model": pr["model"], "seed": pr["seed"], "render": tuple(pr["render"]), "output": tuple(pr["output"]),
              "frames": pr["frames"], "jitter_phases": pr["jitter_phases"], "jitter": pr["jitter"],
              "cases": pr["cases"]}
    want = generate(params)
    bad = []
    if want["digest"] != man["digest"]:
        bad.append("manifest digest differs from a fresh generation")
    for wc in want["cases"]:
        for fr in wc["frames"]:
            for info in fr["files"].values():
                path = os.path.join(out_dir, info["file"])
                if not os.path.exists(path):
                    bad.append(f"missing {info['file']}")
                    continue
                with open(path, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() != info["sha256"]:
                        bad.append(f"bytes differ: {info['file']}")
    return not bad, bad


def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--verify")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", choices=("m", "rr"), default="m")
    ap.add_argument("--render", type=_size, default=(1707, 960))
    ap.add_argument("--output", type=_size, default=(2560, 1440))
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases")
    ap.add_argument("--jitter-phases", type=int)
    ap.add_argument("--no-jitter", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for n, b in case_list(a.model):
            print(f"{n:<28}{b}")
        return 0
    if a.verify:
        ok, bad = verify(a.verify)
        for b in bad[:50]:
            print(b)
        print("probe inputs " + ("verified: every file matches a fresh generation" if ok else
                                 f"DIFFER: {len(bad)} problem(s)"))
        return 0 if ok else 1
    if not a.out:
        ap.error("--out, --verify or --list")
    params = {"model": a.model, "seed": a.seed, "render": a.render, "output": a.output, "frames": a.frames,
              "jitter_phases": a.jitter_phases or default_phases(a.render, a.output), "jitter": not a.no_jitter,
              "cases": a.cases.split(",") if a.cases else None}
    man = generate(params, a.out)
    nfiles = sum(len(fr["files"]) for cs in man["cases"] for fr in cs["frames"])
    print(f"{len(man['cases'])} case(s), {nfiles} file(s) in {a.out}; digest {man['digest']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
