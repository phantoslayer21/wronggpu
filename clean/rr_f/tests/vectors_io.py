# SPDX-License-Identifier: MIT
"""Reading and writing the spec 51 test vectors (binary files + JSON manifest).

File types: `.e4m3` = uint8 codes, `.f16` = little-endian uint16 bit patterns, `.u8` = uint8 (masks).
Every array entry in the manifest is {"file", "dtype", "shape", "layout"}; paths are relative to the manifest."""
import json
import os

import numpy as np

from clean.rr_f.reference.fmt import from_f16_bits, to_f16_bits
from clean.rr_f.reference.scalar import WindowConsts

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")


def write_array(root, rel, arr, dtype, layout=None, **extra):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.asarray(arr)
    if dtype == "e4m3" or dtype == "u8":
        data = a.astype(np.uint8)
    elif dtype == "f16":
        data = to_f16_bits(a).astype("<u2")
    else:
        raise ValueError(dtype)
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(data).tobytes())
    e = {"file": rel.replace(os.sep, "/"), "dtype": dtype, "shape": list(a.shape)}
    if layout:
        e["layout"] = layout
    e.update(extra)
    return e


def read_array(root, entry):
    path = os.path.join(root, entry["file"])
    shape = tuple(entry["shape"])
    if entry["dtype"] in ("e4m3", "u8"):
        return np.fromfile(path, dtype=np.uint8).reshape(shape)
    if entry["dtype"] == "f16":
        return from_f16_bits(np.fromfile(path, dtype="<u2").reshape(shape))
    raise ValueError(entry["dtype"])


def f16_entry(v):
    return {"value": float(v), "f16": "0x%04X" % int(to_f16_bits(np.array([v]))[0])}


def entry_value(e):
    return from_f16_bits(np.array([int(e["f16"], 16)], dtype=np.uint16))[0]


def load_manifest(root=DEFAULT_DIR):
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def window_consts(manifest, name):
    d = manifest["window_consts"][name]["values"]
    return WindowConsts(**{k: float(entry_value(v)) for k, v in d.items()})


def load_case(manifest, case, root=DEFAULT_DIR):
    """Returns (stage, inputs, weights, consts, wc, shift_row, expected outputs, intermediates)."""
    inputs = {k: read_array(root, e) for k, e in case["inputs"].items()}
    ws = manifest["weight_sets"][case["weights"]]
    weights = {k: read_array(root, e) for k, e in ws["tensors"].items()}
    consts = {k: float(entry_value(v)) for k, v in case["stage_consts"].items()}
    wc = window_consts(manifest, case["window_consts"])
    expected = {k: read_array(root, e) for k, e in case["outputs"].items()}
    inter = {k: read_array(root, e) for k, e in case.get("intermediates", {}).items()}
    return case["stage"], inputs, weights, consts, wc, tuple(case["shift_row"]), expected, inter
