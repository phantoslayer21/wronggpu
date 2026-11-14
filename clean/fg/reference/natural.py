# SPDX-License-Identifier: MIT
"""Frame generation (issue #96): the natural order of the weights, taken from the converter map itself.

Written for this project. The preset-FG map (data/fg.v1.map.json, revision v1.1) gives every weight tensor `W` in
natural order through explicit per-axis offsets into its blob: shape [out, in, 3, 3] for 3 x 3 layers and [out, in] for
1 x 1 layers. This module turns those offsets into gather indices, so stored weights (blobs.split_*) become natural
weights with exactly the converter's element mapping, and converts natural weights into ops.py's canonical
[out, k, k, in]. Nothing here restates the storage order: the map is the only source of it.
"""
import json
import os

import numpy as np

from clean.fg import layout as L

DEFAULT_MAP = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data",
                                            "fg.v1.map.json"))


def load_map(path=DEFAULT_MAP):
    with open(path, "rb") as f:
        m = json.load(f)
    if m.get("preset") != "FG" or m.get("channel_order") != "natural":
        raise ValueError("%s: not a natural-order preset-FG map" % path)
    return m


def stage_name(layer):
    """The map's stage of a layout layer: net1_NN -> dl1.lNN, net2_NN -> dl2.lNN."""
    net, idx = layer.name.split("_")
    return "dl%s.l%s" % (net[-1], idx)


def _stage(m, layer):
    name = stage_name(layer)
    for st in m["stages"]:
        if st["stage"] == name:
            return st
    raise KeyError("map has no stage %s" % name)


def _tensor(st, name):
    for t in st["tensors"]:
        if t["name"] == name:
            return t
    raise KeyError("stage %s has no tensor %s" % (st["stage"], name))


def gather_index(tensor):
    """Element index into the tensor's source for every element of the tensor (row-major over its shape): base plus,
    per axis, offsets[i] or stride * i (the converter's index rule without perms; the FG map has none)."""
    idx = np.array(tensor["base"], dtype=np.int64)
    for ax, n in zip(tensor["axes"], tensor["shape"]):
        if "perm" in ax:
            raise ValueError("perms are not used by the preset-FG map")
        o = np.asarray(ax["offsets"], dtype=np.int64) if "offsets" in ax else ax["stride"] * np.arange(n, dtype=np.int64)
        if o.shape != (n,):
            raise ValueError("axis length does not match the shape")
        idx = idx[..., None] + o
    return idx


def to_canonical(natural):
    """Natural [out, in, 3, 3] or [out, in] -> ops.py's canonical [out, k, k, in]."""
    a = np.asarray(natural)
    if a.ndim == 4:
        return a.transpose(0, 2, 3, 1)
    if a.ndim == 2:
        return a[:, None, None, :]
    raise ValueError("natural weights are 4-D (3 x 3) or 2-D (1 x 1)")


def natural_weights(layer, stored, m):
    """Natural weights (map shape) of one layer from its stored weight span (blobs.split_net1 / split_net2)."""
    t = _tensor(_stage(m, layer), "W")
    idx = gather_index(t)
    stored = np.asarray(stored).reshape(-1)
    if idx.min() < 0 or idx.max() >= stored.size:
        raise ValueError("%s: the map reads outside the stored weights" % layer.name)
    return stored[idx]


def natural_bias_index(layer, m):
    """Element indices of the bias in its source (net1: into the whole blob; net2: into the bias blob)."""
    return gather_index(_tensor(_stage(m, layer), "b"))


def weight_shape(layer, m):
    """The map's natural shape of a layer's W."""
    return tuple(_tensor(_stage(m, layer), "W")["shape"])


def canonical_params(stored_params, m):
    """{layer: (stored weights, bias)} (blobs.split_all) -> {layer: (canonical weights [out, k, k, in] float16, bias)}."""
    out = {}
    for l in list(L.NET1) + list(L.NET2):
        w, b = stored_params[l.name]
        out[l.name] = (to_canonical(natural_weights(l, w, m)), np.asarray(b))
    return out
