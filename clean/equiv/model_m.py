# SPDX-License-Identifier: MIT
"""Model M tools of the equivalence tests (Brief 5; clean/EQUIVALENCE-PROTOCOL.md part 1, "Tools" item 1).

Our side's values, written in the specification's storage formats, for the spec team to compare with the original.
The spec team runs these tools on its machine; the clean side runs them only on synthetic models and inputs.

    # (a) chained stage dumps through spec 90's Pipeline.frame(debug=True), every part-1 boundary
    python -m clean.equiv.model_m dump   --build B --rocm R --model MODEL --render 1706x960 --output 2560x1440 \\
        --inputs FRAMES [--frames 2] (--full | --tiles tiles.json) [--state] [input conventions, below] \\
        [--backend hip|d3d12|dx12|emu] [--kernels post=post_recon2_r,...] --out DIR

    # (b) isolated: one stage from input files in the same formats, outputs in the same formats
    python -m clean.equiv.model_m stage  --build B --rocm R --model MODEL --spec stage_inputs.json --out DIR

    # (c) replay of a recorded input set on the D3D12 executor, final output per frame as RGBA16F
    python -m clean.equiv.model_m replay --build B --rocm R --model MODEL --render WxH --output WxH --inputs FRAMES \\
        --out DIR [--backend d3d12] [--then "python cmp.py {path} {frame}"] [--delete-after]

    # helper: the stage-input file of a stage from our own chained dump (self-check: isolated == chained)
    python -m clean.equiv.model_m inputs --dump DIR --frame 0 --stage dec2 --out stage_inputs.json

FRAMES (the input reader, --reader): "npz" (default) = a folder of one .npz per frame in sorted name order with the keys
of clean/runner/frame_ab.py (color [rh, rw, 3], depth [rh, rw], motion [rh, rw, 2], mv_scale [2], jitter [2]; optional
mv_bias, exposure, pre_exposure, reset); or "module:function" returning an iterable of spec 90 frame dicts.

Input conventions (dump and replay; defaults = spec 90 as the frames give it; recorded in every manifest / description):
  --mv-units frame|render|output|uv  what the raw motion is measured in. frame (default): the frame's own mv_scale (1, 1
                                     when absent); render: render pixels (scale 1); output: output pixels (scale render /
                                     output size); uv: fractions of the render size (scale = render size). The result is
                                     spec 90's mv_scale (raw motion -> render pixels).
  --mv-sign X,Y                      +-1 per axis, multiplies the scale (default 1,1): flips a capture whose motion points
                                     the other way than spec 03's convention. Write a leading minus with "=":
                                     --mv-sign=-1,1 (the same for --jitter-sign and --mv-bias).
  --mv-bias X,Y                      spec 90's mv_bias, added before scaling (default: the frame's, else 0,0).
  --jitter-units render|output|uv    the jitter's unit (default render pixels, spec 90); converted to render pixels.
  --jitter-sign X,Y                  +-1 per axis (default 1,1).
  --depth normal|inverted            depth direction (default normal: 0 near, 1 far; --depth-inverted = inverted).
  --auto-exposure off|on             (default off: the frame's exposure is used).
  --subrect X,Y[,W,H]                the frame sits at origin (X, Y) of larger input textures: colour, depth and motion are
                                     cut to it (default: none, origin 0,0 and the whole frame); without W,H the size is
                                     --render clipped to the texture.

Files: every array is a .npy in the storage format of its spec page (BOUNDARIES below: FP8 E4M3 codes as uint8, FP16
as float16, the 8-bit unsigned grid as uint8 codes k, the 8-bit signed grid as int8 codes k, the 10-bit grid as uint16
codes k, the filter output as float32), natural order [H, W, C]; next to it a .json description (boundary, spec page,
format, dtype, shape, layout, frame, frame counter n, parity, reset, render / feature / reconstruction / output size,
the Swin window offsets, tile). A dump also writes compare-manifest.json for clean/equiv/compare.py.

Tiles (--tiles): {"units": "render pixels", "tiles": [[x, y, w, h], ...]}. Each boundary writes one file per tile,
<boundary>.t<k>.npy, holding the rectangle mapped to its own grid: feature-grid outputs (stride s = feature width /
grid width) [x // s, ceil((x + w) / s)), render-size outputs as given, reconstruction- and output-size outputs scaled
by their size / the render size (floor / ceil). The exposure value is always written whole.
"""
import argparse
import glob
import importlib
import json
import math
import os
import shlex
import subprocess
import sys

import numpy as np

from clean.reference.fp8 import fp8_decode, fp8_encode, is_fp8_exact

GENERATOR = "clean/equiv/model_m.py"
VERSION = 1

# name, path in the debug tree, spec page, storage format, grid
BOUNDARIES = [
    ("exposure.e", ("exposure", "e"), "07", "fp16", "scalar"),
    ("enc0.features", ("enc0", "features"), "03", "fp8", "feature"),
    ("enc0.m_near", ("enc0", "m_near"), "03", "fp16", "render"),
    ("enc0.m_far", ("enc0", "m_far"), "03", "fp16", "render"),
    ("enc0.m_center", ("enc0", "m_center"), "03", "fp16", "render"),
    ("enc1.skip", ("enc1", "skip"), "02", "fp8", "feature"),
    ("enc1.merged", ("enc1", "merged"), "02", "fp8", "feature"),
    ("enc2.skip", ("enc2", "skip"), "02", "fp8", "feature"),
    ("enc2.merged", ("enc2", "merged"), "02", "fp8", "feature"),
] + [(f"b{i}.out", (f"b{i}", "out"), "02", "fp8", "feature") for i in range(6)] + [
    ("dec2.out", ("dec2", "out"), "02", "fp8", "feature"),
    ("dec1.out", ("dec1", "out"), "02", "fp8", "feature"),
    ("dec0.A", ("dec0", "A"), "05", "unorm8", "render"),
    ("dec0.B", ("dec0", "B"), "05", "unorm8", "render"),
    ("dec0.C", ("dec0", "C"), "05", "fp16", "render"),
    ("dec0.latent", ("dec0", "latent"), "05", "snorm8", "render"),
    ("post.history", ("post", "history"), "06", "fp16", "recon"),
    ("post.reduced", ("post", "reduced"), "06", "unorm10", "recon"),
    ("post.display", ("post", "display"), "06", "fp16", "recon"),
    ("filter.out", ("filter", "out"), "06", "fp32", "output"),
]
STATE_BOUNDARIES = [("state.motion_history", ("state", "motion_history"), "10", "fp16", "render")]

# Outputs of each stage run in isolation (spec 90 return keys) -> storage format
STAGE_OUTPUTS = {"exposure": {"e": "fp16"},
                 "enc0": {"features": "fp8", "m_near": "fp16", "m_far": "fp16", "m_center": "fp16"},
                 "enc1": {"skip": "fp8", "merged": "fp8"}, "enc2": {"skip": "fp8", "merged": "fp8"},
                 **{f"b{i}": {"out": "fp8"} for i in range(6)},
                 "dec2": {"out": "fp8"}, "dec1": {"out": "fp8"},
                 "dec0": {"A": "unorm8", "B": "unorm8", "C": "fp16", "latent": "snorm8"},
                 "post": {"history": "fp16", "reduced": "unorm10", "display": "fp16"},
                 "filter": {"out": "fp32"}}
SWIN = ["enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"]
NO_WEIGHTS = ("exposure", "filter")
# Inputs of the Swin stages and the filter in our own chained dump (helper "inputs")
SWIN_INPUTS = {"enc1": {"x": "enc0.features"}, "enc2": {"x": "enc1.merged"}, "b0": {"x": "enc2.merged"},
               **{f"b{i}": {"x": f"b{i - 1}.out"} for i in range(1, 6)},
               "dec2": {"x": "b5.out", "skip": "enc2.skip"}, "dec1": {"x": "dec2.out", "skip": "enc1.skip"}}

_GRID_SCALE = {"unorm8": 255, "snorm8": 127, "unorm10": 1023}
_GRID_DTYPE = {"unorm8": np.uint8, "snorm8": np.int8, "unorm10": np.uint16}


class EncodeError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------- formats

def encode(values, fmt):
    """float32 values (spec 90: exactly on the format's grid) -> the stored codes. Raises when a value is off-grid."""
    v = np.asarray(values)
    if fmt == "fp8":
        if not is_fp8_exact(v):
            raise EncodeError("values are not FP8 E4M3 exact")
        return fp8_encode(v)
    if fmt == "fp16":
        h = v.astype(np.float16)
        same = (h.astype(np.float64) == v.astype(np.float64)) | (np.isnan(h) & np.isnan(v))
        if not np.all(same):
            raise EncodeError(f"{int(same.size - np.count_nonzero(same))} value(s) are not FP16 exact")
        return h
    if fmt == "fp32":
        return v.astype(np.float32)
    if fmt in _GRID_SCALE:
        sc = _GRID_SCALE[fmt]
        k = np.rint(v.astype(np.float64) * sc)
        if not np.all(np.isfinite(v)) or np.any(np.abs(k - v.astype(np.float64) * sc) > 1e-3):
            raise EncodeError(f"values are not on the {fmt} grid")
        info = np.iinfo(_GRID_DTYPE[fmt])
        if np.any(k < info.min) or np.any(k > info.max):
            raise EncodeError(f"codes outside the {fmt} range")
        return k.astype(_GRID_DTYPE[fmt])
    raise EncodeError(f"unknown format {fmt!r}")


def decode(codes, fmt):
    """Stored codes -> float32 values as spec 90 passes them to the stages."""
    a = np.asarray(codes)
    if fmt == "fp8":
        return fp8_decode(a)
    if fmt in ("fp16", "fp32"):
        return a.astype(np.float32)
    if fmt in _GRID_SCALE:
        return (a.astype(np.float64) / _GRID_SCALE[fmt]).astype(np.float32)
    raise EncodeError(f"unknown format {fmt!r}")


# ---------------------------------------------------------------------------------------------------- tiles

def load_tiles(path):
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    if t.get("units", "render pixels") != "render pixels":
        raise ValueError("tiles: only \"units\": \"render pixels\" is defined")
    tiles = [tuple(int(v) for v in r) for r in t["tiles"]]
    if not tiles or any(len(r) != 4 or r[2] <= 0 or r[3] <= 0 or r[0] < 0 or r[1] < 0 for r in tiles):
        raise ValueError("tiles: a non-empty list of [x, y, w, h] with w, h > 0 and x, y >= 0")
    return tiles


def map_rect(rect, grid, shape_hw, sizes):
    """Render-pixel rectangle -> (x0, y0, x1, y1) in a boundary's own grid (clamped to it)."""
    x, y, w, h = rect
    H, W = shape_hw
    rw, rh = sizes["render_size"]
    if grid == "feature":
        fw, fh = sizes["feature_size"]
        sx, sy = fw // W, fh // H
        x0, y0, x1, y1 = x // sx, y // sy, -(-(x + w) // sx), -(-(y + h) // sy)
    elif grid == "render":
        x0, y0, x1, y1 = x, y, x + w, y + h
    else:
        gw, gh = sizes["recon_size"] if grid == "recon" else sizes["output_size"]
        x0, y0 = math.floor(x * gw / rw), math.floor(y * gh / rh)
        x1, y1 = math.ceil((x + w) * gw / rw), math.ceil((y + h) * gh / rh)
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


# ---------------------------------------------------------------------------------------------------- inputs

def read_npz_frames(path):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz frames in {path}")
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            x = {k: z[k] for k in z.files}
        for k in ("reset",):
            if k in x:
                x[k] = bool(np.asarray(x[k]).reshape(-1)[0])
        for k in ("exposure", "pre_exposure"):
            if k in x:
                x[k] = float(np.asarray(x[k]).reshape(-1)[0])
        yield x


def _pair(text, name, allowed=None):
    v = tuple(float(t) for t in str(text).split(","))
    if len(v) != 2 or (allowed and any(x not in allowed for x in v)):
        raise ValueError(f"{name}: two comma-separated values" + (f" from {sorted(allowed)}" if allowed else ""))
    return v


class Conventions:
    """Input conventions of a recorded set, applied to every frame before the pipeline (see the module docstring)."""
    MV_UNITS = ("frame", "render", "output", "uv")
    JITTER_UNITS = ("render", "output", "uv")

    def __init__(self, mv_units="frame", mv_sign=(1, 1), mv_bias=None, jitter_units="render", jitter_sign=(1, 1),
                 depth="normal", auto_exposure="off", subrect=None):
        if mv_units not in self.MV_UNITS or jitter_units not in self.JITTER_UNITS:
            raise ValueError("unknown motion or jitter units")
        if depth not in ("normal", "inverted") or auto_exposure not in ("off", "on"):
            raise ValueError("depth: normal | inverted; auto_exposure: off | on")
        if any(v not in (1, -1) for v in tuple(mv_sign) + tuple(jitter_sign)):
            raise ValueError("signs are +1 or -1")
        if subrect is not None and (len(subrect) not in (2, 4) or min(subrect) < 0 or
                                    (len(subrect) == 4 and min(subrect[2:]) <= 0)):
            raise ValueError("subrect: X,Y or X,Y,W,H with X, Y >= 0 and W, H > 0")
        self.mv_units, self.mv_sign, self.mv_bias = mv_units, tuple(mv_sign), mv_bias
        self.jitter_units, self.jitter_sign = jitter_units, tuple(jitter_sign)
        self.depth, self.auto_exposure = depth, auto_exposure
        self.subrect = tuple(int(v) for v in subrect) if subrect is not None else None

    def flags(self):
        return {"depth_inverted": self.depth == "inverted", "auto_exposure": self.auto_exposure == "on"}

    def record(self):
        return {"mv_units": self.mv_units, "mv_sign": list(self.mv_sign),
                "mv_bias": list(self.mv_bias) if self.mv_bias is not None else "frame's, else 0,0",
                "jitter_units": self.jitter_units, "jitter_sign": list(self.jitter_sign), "depth": self.depth,
                "auto_exposure": self.auto_exposure,
                "subrect": list(self.subrect) if self.subrect else "none (origin 0,0, whole frame)"}

    def apply(self, x, render, output):
        x = dict(x)
        if self.subrect:
            ox, oy = self.subrect[:2]
            th, tw = np.asarray(x["color"]).shape[:2]
            w, h = self.subrect[2:] if len(self.subrect) == 4 else (min(render[0], tw - ox), min(render[1], th - oy))
            if ox + w > tw or oy + h > th or w <= 0 or h <= 0:
                raise ValueError(f"subrect {self.subrect} does not fit textures of {tw}x{th}")
            for k in ("color", "depth", "motion"):
                if k in x:
                    x[k] = np.ascontiguousarray(np.asarray(x[k])[oy:oy + h, ox:ox + w])
        rh, rw = np.asarray(x["color"]).shape[:2]
        ow, oh = output
        unit = {"render": (1.0, 1.0), "output": (rw / ow, rh / oh), "uv": (float(rw), float(rh))}
        if self.mv_units == "frame":
            sc = tuple(float(v) for v in np.asarray(x.get("mv_scale", (1.0, 1.0)), np.float64).reshape(-1)[:2])
        else:
            sc = unit[self.mv_units]
        x["mv_scale"] = np.array([sc[0] * self.mv_sign[0], sc[1] * self.mv_sign[1]])
        if self.mv_bias is not None:
            x["mv_bias"] = np.array(self.mv_bias, np.float64)
        j = np.asarray(x["jitter"], np.float64).reshape(-1)[:2]
        ju = unit[self.jitter_units]
        x["jitter"] = np.array([j[0] * ju[0] * self.jitter_sign[0], j[1] * ju[1] * self.jitter_sign[1]])
        return x


def adapt(frames, conv, render, output):
    for x in frames:
        yield conv.apply(x, render, output)


def conventions_from_args(a):
    return Conventions(mv_units=a.mv_units, mv_sign=_pair(a.mv_sign, "--mv-sign", {1.0, -1.0}),
                       mv_bias=_pair(a.mv_bias, "--mv-bias") if a.mv_bias else None, jitter_units=a.jitter_units,
                       jitter_sign=_pair(a.jitter_sign, "--jitter-sign", {1.0, -1.0}),
                       depth="inverted" if a.depth_inverted else a.depth, auto_exposure=a.auto_exposure,
                       subrect=tuple(int(t) for t in a.subrect.split(",")) if a.subrect else None)


def frame_reader(spec, path):
    if spec in (None, "npz"):
        return read_npz_frames(path)
    mod, fn = spec.split(":", 1)
    return iter(getattr(importlib.import_module(mod), fn)(path))


# ---------------------------------------------------------------------------------------------------- writing

def _save(out_dir, stem, arr, desc):
    np.save(os.path.join(out_dir, stem + ".npy"), arr, allow_pickle=False)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(desc, f, indent=1)
    return os.path.getsize(os.path.join(out_dir, stem + ".npy"))


def frame_sizes(pipe, info):
    rw, rh = info["render_size"]
    return {"render_size": [rw, rh], "feature_size": [-(-rw // 32) * 32, -(-rh // 32) * 32],
            "recon_size": [int(pipe.dw), int(pipe.dh)], "output_size": [int(pipe.ow), int(pipe.oh)]}


def _tree_get(tree, path):
    v = tree
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            return None
    return v


def dump(pipe, frames, out_dir, tiles=None, full=False, max_frames=None, state=False, backend=None,
         conventions=None):
    """(a): run frames through the pipeline with debug=True and write every boundary. -> manifest dict."""
    if not full and not tiles:
        raise ValueError("dump: choose full images (full=True) or a tile list")
    os.makedirs(out_dir, exist_ok=True)
    bounds = BOUNDARIES + (STATE_BOUNDARIES if state else [])
    man = {"generator": GENERATOR, "version": VERSION, "source": "chained: spec 90 Pipeline.frame(debug=True)",
           "backend": backend, "kernels": _kernels(pipe), "tiles": [list(t) for t in tiles] if tiles else None,
           "input_conventions": conventions, "frames": [], "missing": []}
    cmp_entries, total = [], 0
    for i, x in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        tree = pipe.frame(x, debug=True)
        info = pipe.info()
        sizes = frame_sizes(pipe, info)
        fdir = os.path.join(out_dir, f"f{i:04d}")
        os.makedirs(fdir, exist_ok=True)
        meta = {"frame": i, "n": int(info["n"]), "parity": int(info["parity"]), "reset": bool(info["reset"]),
                **sizes, "window_offsets": {k: list(v) for k, v in info["offsets"].items()},
                "jitter": [float(v) for v in np.asarray(x["jitter"], np.float64).reshape(-1)[:2]],
                "jitter_prev": [float(v) for v in info["jitter_prev"]]}
        frec = dict(meta, files=[])
        for name, path, page, fmt, grid in bounds:
            val = _tree_get(tree, path)
            if val is None:
                man["missing"].append({"frame": i, "boundary": name,
                                       "why": "not produced by this kernel choice (e.g. a fused post kernel writes no "
                                              "display image; use --kernels post=post_recon2_r)"})
                continue
            codes = encode(np.asarray(val, np.float32).reshape(-1) if grid == "scalar" else val, fmt)
            base = {"boundary": name, "spec_page": page, "format": fmt, "grid": grid, **meta}
            parts = [(None, codes)] if (full or grid == "scalar") else []
            if tiles and grid != "scalar":
                for k, rect in enumerate(tiles):
                    x0, y0, x1, y1 = map_rect(rect, grid, codes.shape[:2], sizes)
                    parts.append(((k, rect, (x0, y0, x1, y1)), codes[y0:y1, x0:x1]))
            for tile, arr in parts:
                stem = name if tile is None else f"{name}.t{tile[0]:02d}"
                desc = dict(base, dtype=str(arr.dtype), shape=list(arr.shape), layout="HWC"[:arr.ndim] if
                            grid != "scalar" else "W", tile=None if tile is None else
                            {"index": tile[0], "render_rect": list(tile[1]),
                             "rect": [tile[2][0], tile[2][1], tile[2][2] - tile[2][0], tile[2][3] - tile[2][1]]})
                total += _save(fdir, stem, np.ascontiguousarray(arr), desc)
                rel = f"f{i:04d}/{stem}.npy"
                frec["files"].append(rel)
                cmp_entries.append({"name": f"{stem} f{i}", "file": rel, "format": fmt,
                                    "layout": desc["layout"], "border": 8 if tile is None else 0})
        man["frames"].append(frec)
    man["bytes"] = total
    _write_json(os.path.join(out_dir, "manifest.json"), man)
    _write_json(os.path.join(out_dir, "compare-manifest.json"),
                {"model": "Model M", "run": "chained",
                 "note": "chained dump: intermediates get numbers only; set \"final\": true on the outputs to judge "
                         "(filter.out is marked), or use the stage tool's outputs for isolated verdicts",
                 "stages": [dict(e, final=e["name"].startswith("filter.out")) for e in cmp_entries]})
    return man


def _kernels(pipe):
    try:
        return pipe.kernels()
    except Exception:  # a test double or an older runtime library
        return None


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------------------------------------------- isolated

def _load_input(base_dir, ref):
    """An input reference: "file.npy" (format from its .json description) or {"file": ..., "format": ...}."""
    if isinstance(ref, str):
        ref = {"file": ref}
    path = os.path.join(base_dir, ref["file"])
    fmt = ref.get("format")
    if fmt is None:
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            fmt = json.load(f)["format"]
    return decode(np.load(path, allow_pickle=False), fmt)


def stage_x(spec, base_dir):
    """Stage-input file -> (x, offset, skip) for Runner.run (spec 90)."""
    stage = spec["stage"]
    ins = {k: _load_input(base_dir, r) for k, r in spec.get("inputs", {}).items()}
    sc = dict(spec.get("scalars", {}))
    for k in ("feature_size", "recon_size", "output_size", "jitter", "jitter_prev", "mv_scale", "mv_bias"):
        if k in sc:
            sc[k] = tuple(sc[k])
    if stage in SWIN:
        if spec.get("offset") is not None:
            off = tuple(spec["offset"])
        else:
            from clean.reference.swin import window_offset
            off = window_offset(stage, int(spec["parity"]))
        return ins["x"], off, ins.get("skip")
    if stage == "exposure":
        x = {"color": ins["color"], "pre_exposure": float(sc.get("pre_exposure", 1.0)), "alpha": float(sc.get("alpha", 0.18))}
        return x, None, None
    x = dict(ins)
    x.update(sc)
    return x, None, None


def run_stage(runner, spec_path, out_dir, weights=None, model_dir=None):
    """(b): one stage from a stage-input file; writes <stage>.<output>.npy / .json. -> list of written stems."""
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    stage = spec["stage"]
    if stage not in STAGE_OUTPUTS:
        raise ValueError(f"unknown stage {stage!r} (known: {', '.join(STAGE_OUTPUTS)})")
    x, off, skip = stage_x(spec, os.path.dirname(os.path.abspath(spec_path)))
    if weights is None:
        weights = b"" if stage in NO_WEIGHTS else stage_weights(runner, stage, model_dir)
    out = runner.run(stage, weights, x, off, skip=skip) if skip is not None else runner.run(stage, weights, x, off)
    os.makedirs(out_dir, exist_ok=True)
    stems = []
    meta = dict(spec.get("meta", {}))
    for key, fmt in STAGE_OUTPUTS[stage].items():
        val = out.get(key)
        if val is None:
            continue
        codes = encode(np.asarray(val, np.float32).reshape(-1) if np.ndim(val) == 0 else val, fmt)
        stem = f"{stage}.{key}"
        _save(out_dir, stem, codes, {"boundary": stem, "format": fmt, "dtype": str(codes.dtype),
                                     "shape": list(codes.shape), "layout": "HWC"[:codes.ndim] if codes.ndim > 1
                                     else "W", "source": "isolated: spec 90 Runner.run", "stage": stage,
                                     "offset": list(off) if off else None, **meta})
        stems.append(stem)
    return stems


def stage_weights(runner, stage, model_dir):
    from clean.runner.pipeline import logical_tensors, read_model_tensors
    return runner.pack_weights(stage, logical_tensors(read_model_tensors(model_dir))[stage])


def stage_inputs_from_dump(dump_dir, frame, stage):
    """Stage-input file content for a Swin stage or the filter from our own chained dump (inputs referenced by path
    relative to the dump folder)."""
    fdir = f"f{frame:04d}"
    with open(os.path.join(dump_dir, fdir, "enc0.features.json"), encoding="utf-8") as f:
        d = json.load(f)
    meta = {k: d[k] for k in ("frame", "n", "parity", "reset", "render_size", "feature_size", "recon_size",
                              "output_size")}
    if stage in SWIN:
        return {"stage": stage, "inputs": {k: f"{fdir}/{b}.npy" for k, b in SWIN_INPUTS[stage].items()},
                "offset": d["window_offsets"][stage], "parity": d["parity"], "meta": meta}
    if stage == "filter":
        e = decode(np.load(os.path.join(dump_dir, fdir, "exposure.e.npy")), "fp16")
        return {"stage": "filter", "inputs": {"display": f"{fdir}/post.display.npy"},
                "scalars": {"exposure": float(e[0]), "output_size": d["output_size"]}, "meta": meta}
    raise ValueError(f"inputs from a dump: Swin stages and filter only (got {stage!r}); write the stage-input file "
                     "for enc0 / dec0 / post / exposure with the spec 90 keys")


# ---------------------------------------------------------------------------------------------------- replay

def to_rgba16f(out):
    """[oh, ow, 3] float32 -> [oh, ow, 4] float16, alpha 1."""
    o = np.asarray(out, np.float32)
    return np.concatenate([o, np.ones(o.shape[:2] + (1,), np.float32)], axis=-1).astype(np.float16)


def replay(pipe, frames, out_dir, then=None, delete_after=False, max_frames=None, backend=None, conventions=None):
    """(c): every frame's final output as RGBA16F. then(path, index) runs after each frame is written; with
    delete_after the file is removed after it. -> summary dict (frames, peak bytes on disk)."""
    os.makedirs(out_dir, exist_ok=True)
    peak, live, n, results = 0, 0, 0, []
    for i, x in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        img = to_rgba16f(pipe.frame(x)["out"])
        stem = f"f{i:04d}.rgba16f"
        live += _save(out_dir, stem, img, {"frame": i, "format": "RGBA16F", "dtype": "float16",
                                           "shape": list(img.shape), "layout": "HWC", "backend": backend,
                                           "input_conventions": conventions,
                                           "source": "replay: spec 90 Pipeline.frame (final output, alpha 1)"})
        peak = max(peak, live)
        path = os.path.join(out_dir, stem + ".npy")
        if then is not None:
            results.append(then(path, i))
        if delete_after:
            live -= os.path.getsize(path)
            os.remove(path)
            os.remove(os.path.join(out_dir, stem + ".json"))
        n += 1
    return {"frames": n, "peak_bytes": peak, "kept": not delete_after, "results": results,
            "input_conventions": conventions}


def command_callback(template):
    """--then "cmd {path} {frame}": runs the command (no shell) per frame; a non-zero exit stops the replay."""
    def run(path, i):
        toks = shlex.split(template, posix=os.name != "nt")
        if os.name == "nt":   # non-POSIX splitting keeps the quotes of a quoted argument
            toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]
        args = [t.format(path=path, frame=i) for t in toks]
        rc = subprocess.run(args).returncode
        if rc != 0:
            raise RuntimeError(f"--then command failed on frame {i} (exit {rc})")
        return rc
    return run


# ---------------------------------------------------------------------------------------------------- CLI

def _size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def _pipeline(a, conv):
    from clean.runner.pipeline import Pipeline
    kernels = dict(kv.split("=", 1) for kv in a.kernels.split(",")) if a.kernels else None
    return Pipeline(a.build, a.rocm, a.model, a.render, a.output, conv.flags(), backend=a.backend, kernels=kernels)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "stage", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--build", required=True)
        p.add_argument("--rocm", default=r"C:\Program Files\AMD\ROCm\7.2")
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        if name in ("dump", "replay"):
            p.add_argument("--render", type=_size, required=True, help="maximum render size WxH")
            p.add_argument("--output", type=_size, required=True)
            p.add_argument("--inputs", required=True)
            p.add_argument("--reader", default="npz")
            p.add_argument("--frames", type=int)
            p.add_argument("--mv-units", choices=Conventions.MV_UNITS, default="frame")
            p.add_argument("--mv-sign", default="1,1")
            p.add_argument("--mv-bias")
            p.add_argument("--jitter-units", choices=Conventions.JITTER_UNITS, default="render")
            p.add_argument("--jitter-sign", default="1,1")
            p.add_argument("--depth", choices=("normal", "inverted"), default="normal")
            p.add_argument("--depth-inverted", action="store_true", help="the same as --depth inverted")
            p.add_argument("--auto-exposure", choices=("off", "on"), nargs="?", const="on", default="off")
            p.add_argument("--subrect", help="X,Y or X,Y,W,H")
            p.add_argument("--kernels")
            p.add_argument("--backend", default="hip" if name == "dump" else "d3d12")
    d = sub.choices["dump"]
    d.add_argument("--full", action="store_true")
    d.add_argument("--tiles")
    d.add_argument("--state", action="store_true", help="also the motion history (state after the frame)")
    sub.choices["stage"].add_argument("--spec", required=True)
    sub.choices["stage"].add_argument("--backend", default="hip")
    r = sub.choices["replay"]
    r.add_argument("--then")
    r.add_argument("--delete-after", action="store_true")
    i = sub.add_parser("inputs")
    i.add_argument("--dump", required=True)
    i.add_argument("--frame", type=int, default=0)
    i.add_argument("--stage", required=True)
    i.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "inputs":
        spec = stage_inputs_from_dump(a.dump, a.frame, a.stage)
        rel = os.path.relpath(os.path.abspath(a.dump), os.path.dirname(os.path.abspath(a.out)))
        for k, v in spec.get("inputs", {}).items():
            spec["inputs"][k] = os.path.join(rel, v).replace("\\", "/")
        _write_json(a.out, spec)
        print(f"stage-input file {a.out}")
        return 0
    if a.cmd == "stage":
        from clean.runner.stage_runner import Runner
        stems = run_stage(Runner(a.build, a.rocm, backend=a.backend), a.spec, a.out, model_dir=a.model)
        print(f"{len(stems)} output(s) in {a.out}: {', '.join(stems)}")
        return 0
    conv = conventions_from_args(a)
    frames = adapt(frame_reader(a.reader, a.inputs), conv, a.render, a.output)
    pipe = _pipeline(a, conv)
    if a.cmd == "dump":
        man = dump(pipe, frames, a.out, tiles=load_tiles(a.tiles) if a.tiles else None, full=a.full,
                   max_frames=a.frames, state=a.state, backend=a.backend, conventions=conv.record())
        print(f"{len(man['frames'])} frame(s), {sum(len(f['files']) for f in man['frames'])} file(s), "
              f"{man['bytes']} bytes in {a.out}; missing: {len(man['missing'])}")
        return 0
    s = replay(pipe, frames, a.out, then=command_callback(a.then) if a.then else None, delete_after=a.delete_after,
               max_frames=a.frames, backend=a.backend, conventions=conv.record())
    print(f"{s['frames']} frame(s) replayed; peak {s['peak_bytes']} bytes on disk; outputs "
          f"{'kept' if s['kept'] else 'deleted after each step'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
