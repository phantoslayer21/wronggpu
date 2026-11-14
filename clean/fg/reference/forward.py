# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): CPU reference of both networks' forward pass, as far as it is delivered.

Written for this project. The layers are settled: their shapes (clean/fg/layout.py), the weights in natural order and
the biases the converter map gives per stage (data/fg.v1.map.json, revision v1.1; clean/fg/reference/natural.py),
float16 convolutions with bias, ReLU, 2x pooling, bilinear 2x upscale, element-wise combines, and net1's 2 x 2 fold
(clean/fg/reference/ops.py). net1's blob 2 is a 1 x 1 convolution. How the layers are connected is not delivered, so a
network is run from a wiring the caller supplies. When the real wiring arrives it is passed in as data; the layers do
not change.

Required arguments for what is OPEN (clean/fg/FORMATS.md):
- `wiring`: the list of steps (which layer reads which buffers, where the pooling, upscales, combines and channel
  splits sit, ReLU after each convolution, the pooling kind). Format: see run_graph.
- `pad`: the border rule of the 3 x 3 convolutions (ops.PAD_MODES).
- `fold_order`: the order of the values inside net1's 48-value cells.

Parameters are {layer name: (weights [out, k, k, in] float16, bias [out] float16)} with the layout's layer names
(net1_01 .. net1_17, net2_01 .. net2_25): from blobs (params_from_blobs: stored weights put in natural order by the
map's own offsets) or from the converter's raw output (params_from_raw: stages dl1.lNN / dl2.lNN, tensors W and b).
"""
import os

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import blobs as _blobs
from clean.fg.reference import natural as _natural
from clean.fg.reference import ops

LAYERS = {l.name: l for l in list(L.NET1) + list(L.NET2)}


# ---- parameters -------------------------------------------------------------------------------------------------

def params_from_blobs(blob_bytes, m=None, *, check=True):
    """{layer name: (canonical weights, bias)} from {blob name: bytes} (layout.blobs() names): blobs.split_all, then the
    map's natural view (m: the parsed map; default data/fg.v1.map.json)."""
    m = m if m is not None else _natural.load_map()
    return _natural.canonical_params(_blobs.split_all(blob_bytes, check=check), m)


map_stage_name = _natural.stage_name


def params_from_raw(raw_dir, m=None):
    """{layer name: (canonical weights, bias)} from the converter's --raw output of the preset-FG map
    (raw/<stage>/W.bin in the map's natural shape, raw/<stage>/b.bin; float32 little endian). Every value must be exact
    in float16."""
    m = m if m is not None else _natural.load_map()
    out = {}
    for name, l in LAYERS.items():
        st = os.path.join(raw_dir, map_stage_name(l))
        shape = _natural.weight_shape(l, m)
        w = np.fromfile(os.path.join(st, "W.bin"), dtype="<f4")
        b = np.fromfile(os.path.join(st, "b.bin"), dtype="<f4")
        if w.size != int(np.prod(shape)) or b.size != l.out:
            raise ValueError("%s: %d weights / %d biases, the map says %s / %d" % (name, w.size, b.size, shape, l.out))
        w16, b16 = w.astype("<f2"), b.astype("<f2")
        if not (np.array_equal(w16.astype("<f4"), w) and np.array_equal(b16.astype("<f4"), b)):
            raise ValueError("%s: a value is not exact in float16" % name)
        out[name] = (_natural.to_canonical(w16.reshape(shape)), b16)
    return out


# ---- one layer ----------------------------------------------------------------------------------------------------

def layer_weights(name, params):
    """(canonical weights [out, k, k, in], bias [out]) of one layer as float64, checked against the layout."""
    if name not in LAYERS:
        raise KeyError("unknown layer %r" % name)
    l = LAYERS[name]
    w, b = params[name]
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if w.shape != ops.weight_shape(l) or b.shape != (l.out,):
        raise ValueError("%s: weights %s / bias %s, the layout says %s / (%d,)" %
                         (name, w.shape, b.shape, ops.weight_shape(l), l.out))
    return w, b


def run_layer(name, x, params, *, pad, relu_after):
    """One convolution layer of either network on x [cin, H, W] (float16 values); ops.conv2d does the arithmetic."""
    w, b = layer_weights(name, params)
    return ops.conv2d(x, w, b, pad=pad, relu_after=relu_after)


# ---- a network from its wiring ------------------------------------------------------------------------------------

STEP_FIELDS = {
    "conv": {"layer", "in", "out", "relu"},
    "pool": {"in", "out", "kind"},
    "upscale": {"in", "out"},
    "add": {"in", "out"},
    "concat": {"in", "out"},
    "slice": {"in", "out", "channels"},
}


def check_wiring(wiring, inputs):
    """Validates a wiring against the names of the given inputs: known step kinds, exactly their fields, every buffer
    written once before it is read, every layer used at most once. Returns the buffer names in write order."""
    have = list(inputs)
    used = set()
    for i, s in enumerate(wiring):
        kind = s.get("op")
        if kind not in STEP_FIELDS:
            raise ValueError("step %d: unknown op %r" % (i, kind))
        fields = set(s) - {"op"}
        if fields != STEP_FIELDS[kind]:
            raise ValueError("step %d (%s): fields %s, want %s" % (i, kind, sorted(fields), sorted(STEP_FIELDS[kind])))
        ins = s["in"] if isinstance(s["in"], list) else [s["in"]]
        for n in ins:
            if n not in have:
                raise ValueError("step %d (%s): buffer %r is read before it is written" % (i, kind, n))
        if kind in ("add", "concat") and len(ins) < 2:
            raise ValueError("step %d (%s): needs at least two inputs" % (i, kind))
        if kind in ("conv", "pool", "upscale", "slice") and (isinstance(s["in"], list) or len(ins) != 1):
            raise ValueError("step %d (%s): takes one input" % (i, kind))
        if kind == "conv":
            if s["layer"] not in LAYERS:
                raise ValueError("step %d: unknown layer %r" % (i, s["layer"]))
            if s["layer"] in used:
                raise ValueError("step %d: layer %s used twice" % (i, s["layer"]))
            if not isinstance(s["relu"], bool):
                raise ValueError("step %d: relu must be true or false" % i)
            used.add(s["layer"])
        if s["out"] in have:
            raise ValueError("step %d (%s): buffer %r written twice" % (i, kind, s["out"]))
        have.append(s["out"])
    return have


def run_graph(wiring, inputs, params, *, pad):
    """Runs a wiring: a list of steps, each a dict
        {"op": "conv", "layer": <layout layer name>, "in": <buffer>, "out": <buffer>, "relu": true|false}
        {"op": "pool", "in": <buffer>, "out": <buffer>, "kind": "max"|"avg"}
        {"op": "upscale", "in": <buffer>, "out": <buffer>}            (bilinear 2x, align corners)
        {"op": "add", "in": [<buffer>, ...], "out": <buffer>}         (element-wise, float16 after each sum)
        {"op": "concat", "in": [<buffer>, ...], "out": <buffer>}      (channels)
        {"op": "slice", "in": <buffer>, "out": <buffer>, "channels": [first, end]}
    on the input buffers {name: [c, H, W]}. Returns every buffer {name: array}."""
    check_wiring(wiring, inputs)
    buf = {k: ops.f16(v) for k, v in inputs.items()}
    for s in wiring:
        kind = s["op"]
        if kind == "conv":
            y = run_layer(s["layer"], buf[s["in"]], params, pad=pad, relu_after=s["relu"])
        elif kind == "pool":
            y = ops.pool2x(buf[s["in"]], kind=s["kind"])
        elif kind == "upscale":
            y = ops.upscale2x_bilinear(buf[s["in"]])
        elif kind == "add":
            y = buf[s["in"][0]]
            for n in s["in"][1:]:
                y = ops.add(y, buf[n])
        elif kind == "concat":
            y = ops.concat(*[buf[n] for n in s["in"]])
        else:  # slice
            a, e = s["channels"]
            src = buf[s["in"]]
            if not (0 <= a < e <= src.shape[0]):
                raise ValueError("slice %s outside %d channels" % (s["channels"], src.shape[0]))
            y = src[a:e].copy()
        buf[s["out"]] = y
    return buf


def net1_forward(features, params, *, wiring, pad, fold_order, output):
    """net1 on its per-pixel features [10, H, W] (ops.net1_features): fold 2 x 2 into 48-value cells (fold_order:
    OPEN, required), run the wiring on the input buffer "cells", and unfold the buffer named `output` (16 values per
    cell) back to 4 values per pixel -> [4, H, W]."""
    if features.shape[0] != ops.NET1_FEATURES:
        raise ValueError("net1 takes %d features per pixel" % ops.NET1_FEATURES)
    cells = ops.fold2x2(features, order=fold_order)
    buf = run_graph(wiring, {"cells": cells}, params, pad=pad)
    out = buf[output]
    if out.shape[0] != 16:
        raise ValueError("net1's output has 16 values per cell, got %d" % out.shape[0])
    return ops.unfold2x2(out, 4, order=fold_order)


def net2_forward(inputs, params, *, wiring, pad, outputs):
    """net2 on its input buffers {name: [c, H, W]}: runs the wiring and returns {name: buffer} for `outputs`."""
    buf = run_graph(wiring, inputs, params, pad=pad)
    return {n: buf[n] for n in outputs}
