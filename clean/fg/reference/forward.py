# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): CPU reference of both networks' forward pass, as far as it is delivered.

Written for this project. The layers are settled: their shapes (clean/fg/layout.py), the weights and biases the
converter map gives per stage, float16 convolutions with bias, ReLU, 2x pooling, bilinear 2x upscale, element-wise
combines, and net1's 2 x 2 fold (clean/fg/reference/ops.py). How they are connected is not delivered, so the network
is run from a wiring the caller supplies. When the real wiring arrives it is passed in as data; the layers do not
change.

Required arguments for what is OPEN (clean/fg/FORMATS.md):
- `wiring`: the list of steps (which layer reads which buffers, where the pooling, upscales, combines and channel
  splits sit, ReLU after each convolution, the pooling kind). Format: see run_graph.
- `weight_view`: a function (storage weights [n] float16, layer) -> natural weights [out, k, k, in]. The converter
  map v1 gives the weights as stored (the matrix unit's own tiling), so the natural view comes with map v1.1.
  `view_by_axes` makes one for a plain axis order, for tests and for a map revision that is a plain reorder.
- `pad`: the border rule of the 3 x 3 convolutions (ops.PAD_MODES).
- `blob2`: net1's blob 2 is a 1 x 1 or a 2 x 2 convolution. "1x1" runs; "2x2" raises OpenItem (its stride and its
  grouping of the 32 input values into 4 taps x 8 channels are not delivered).

Parameters are {layer name: (storage weights [n] float16, bias [out] float16)} with the layout's layer names
(net1_01 .. net1_17, net2_01 .. net2_25): from synthetic or converted blobs (params_from_blobs) or from the
converter's raw output (params_from_raw: stages dl1.lNN / dl2.lNN of the preset-FG map).
"""
import os

import numpy as np

from clean.fg import layout as L
from clean.fg.reference import blobs as _blobs
from clean.fg.reference import ops
from clean.fg.reference.ops import OpenItem

LAYERS = {l.name: l for l in list(L.NET1) + list(L.NET2)}
BLOB2_FORMS = ("1x1", "2x2")


# ---- parameters -------------------------------------------------------------------------------------------------

def params_from_blobs(blob_bytes, *, check=True):
    """{layer name: (weights, bias)} from {blob name: bytes} (layout.blobs() names), via blobs.split_all."""
    return _blobs.split_all(blob_bytes, check=check)


def map_stage_name(layer):
    """The preset-FG map's stage of a layout layer: net1_NN -> dl1.lNN, net2_NN -> dl2.lNN."""
    net, idx = layer.name.split("_")
    return "dl%s.l%s" % (net[-1], idx)


def params_from_raw(raw_dir):
    """{layer name: (weights, bias)} from the converter's --raw output (raw/<stage>/<tensor>.bin, float32 little
    endian; tensors W_storage and b (net1) or b_storage (net2)). Every value must be exact in float16."""
    out = {}
    for name, l in LAYERS.items():
        st = os.path.join(raw_dir, map_stage_name(l))
        w = np.fromfile(os.path.join(st, "W_storage.bin"), dtype="<f4")
        bname = "b.bin" if name.startswith("net1") else "b_storage.bin"
        b = np.fromfile(os.path.join(st, bname), dtype="<f4")
        if w.size != l.weight_elements or b.size != l.out:
            raise ValueError("%s: %d weights / %d biases, the layout says %d / %d" %
                             (name, w.size, b.size, l.weight_elements, l.out))
        w16, b16 = w.astype("<f2"), b.astype("<f2")
        if not (np.array_equal(w16.astype("<f4"), w) and np.array_equal(b16.astype("<f4"), b)):
            raise ValueError("%s: a value is not exact in float16" % name)
        out[name] = (w16, b16)
    return out


def view_by_axes(order):
    """A weight_view for weights stored as a plain row-major array whose axes, slowest first, are `order`: a
    permutation of "out", "ky", "kx", "in" given as a string, e.g. "out in ky kx". Returns [out, k, k, in]."""
    axes = order.split()
    if sorted(axes) != ["in", "kx", "ky", "out"]:
        raise ValueError("order must name out, ky, kx and in once each")

    def view(w, layer):
        cout, k, _, cin = ops.weight_shape(layer)
        size = {"out": cout, "ky": k, "kx": k, "in": cin}
        a = np.asarray(w).reshape([size[n] for n in axes])
        return a.transpose([axes.index(n) for n in ("out", "ky", "kx", "in")])
    return view


# ---- one layer ----------------------------------------------------------------------------------------------------

def layer_weights(name, params, *, weight_view, blob2):
    """(natural weights [out, k, k, in], bias [out]) of one layer, checked against the layout."""
    if name not in LAYERS:
        raise KeyError("unknown layer %r" % name)
    l = LAYERS[name]
    if name == "net1_02":
        if blob2 not in BLOB2_FORMS:
            raise ValueError("blob2 must be one of %s (OPEN; no default)" % (BLOB2_FORMS,))
        if blob2 == "2x2":
            raise OpenItem("net1 blob 2 as a 2 x 2 convolution: stride and the 4 taps x 8 channels grouping")
        shape = (l.out, 1, 1, l.k)
    else:
        shape = ops.weight_shape(l)
    w, b = params[name]
    if np.asarray(w).size != l.weight_elements or np.asarray(b).shape != (l.out,):
        raise ValueError("%s: parameter sizes do not fit the layout" % name)
    if name == "net1_02":
        nat = np.asarray(w, dtype=np.float64).reshape(l.out, l.k)[:, None, None, :]   # 1 x 1: out x in only
    else:
        nat = np.asarray(weight_view(w, l), dtype=np.float64)
    if nat.shape != shape:
        raise ValueError("%s: weight_view gave %s, want %s" % (name, nat.shape, shape))
    return nat, np.asarray(b, dtype=np.float64)


def run_layer(name, x, params, *, weight_view, pad, relu_after, blob2=None):
    """One convolution layer of either network on x [cin, H, W] (float16 values); ops.conv2d does the arithmetic."""
    w, b = layer_weights(name, params, weight_view=weight_view, blob2=blob2)
    return ops.conv2d(x, w, b, pad=pad, relu_after=relu_after)


# ---- a network from its wiring ------------------------------------------------------------------------------------

STEP_FIELDS = {
    "conv": ({"layer", "in", "out", "relu"}, set()),
    "pool": ({"in", "out", "kind"}, set()),
    "upscale": ({"in", "out"}, set()),
    "add": ({"in", "out"}, set()),
    "concat": ({"in", "out"}, set()),
    "slice": ({"in", "out", "channels"}, set()),
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
        need, _ = STEP_FIELDS[kind]
        fields = set(s) - {"op"}
        if fields != need:
            raise ValueError("step %d (%s): fields %s, want %s" % (i, kind, sorted(fields), sorted(need)))
        ins = s["in"] if isinstance(s["in"], list) else [s["in"]]
        for n in ins:
            if n not in have:
                raise ValueError("step %d (%s): buffer %r is read before it is written" % (i, kind, n))
        if kind in ("add", "concat") and len(ins) < 2:
            raise ValueError("step %d (%s): needs at least two inputs" % (i, kind))
        if kind in ("conv", "pool", "upscale", "slice") and len(ins) != 1:
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


def run_graph(wiring, inputs, params, *, weight_view, pad, blob2=None):
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
            y = run_layer(s["layer"], buf[s["in"]], params, weight_view=weight_view, pad=pad, relu_after=s["relu"],
                          blob2=blob2)
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


def net1_forward(features, params, *, wiring, weight_view, pad, blob2, fold_order, output):
    """net1 on its per-pixel features [10, H, W] (ops.net1_features): fold 2 x 2 into 48-value cells (fold_order:
    OPEN, required), run the wiring on the input buffer "cells", and unfold the buffer named `output` (16 values per
    cell) back to 4 values per pixel -> [4, H, W]."""
    if features.shape[0] != ops.NET1_FEATURES:
        raise ValueError("net1 takes %d features per pixel" % ops.NET1_FEATURES)
    cells = ops.fold2x2(features, order=fold_order)
    buf = run_graph(wiring, {"cells": cells}, params, weight_view=weight_view, pad=pad, blob2=blob2)
    out = buf[output]
    if out.shape[0] != 16:
        raise ValueError("net1's output has 16 values per cell, got %d" % out.shape[0])
    return ops.unfold2x2(out, 4, order=fold_order)


def net2_forward(inputs, params, *, wiring, weight_view, pad, outputs):
    """net2 on its input buffers {name: [c, H, W]}: runs the wiring and returns {name: buffer} for `outputs`."""
    buf = run_graph(wiring, inputs, params, weight_view=weight_view, pad=pad)
    return {n: buf[n] for n in outputs}
