# SPDX-License-Identifier: MIT
"""Model K parameters from the converter's output (spec 48 section 10 / spec 08 section 10: with --raw, every tensor as
`<dir>/raw/<stage>/<name>.bin`, row-major little-endian float32), with the names and shapes of the preset-K map
(`data/preset-K.v1.map.json`, version 3: the eleven stages and the group "net" with the window-offset table of
spec 44, int32). The clean team never has these files; the spec team uses this to run the harness and the runner on
real parameters (41-T2, 48-T6).

    params = load_raw(model_dir)                    # {stage: {name: float32 array}}, converter shapes
    w = Runner(..., model="K").pack_weights("e2", params["e2"])     # pack_weights accepts the converter shapes
"""
import json
import os

import numpy as np

from clean.k.reference.layout import STAGE_NAMES
from clean.k.reference.params import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MAP_PATH = os.path.join(ROOT, "data", "preset-K.v1.map.json")


def map_tensors(map_path=MAP_PATH, with_dtype=False):
    """{output group: [(name, shape)]} in map order: the eleven stages and "net" (spec 48 draft 3: the window-offset
    table, int32). with_dtype=True: [(name, shape, dtype)]."""
    with open(map_path) as f:
        m = json.load(f)
    return {s["stage"]: [(t["name"], tuple(t["shape"])) + ((t.get("dtype", "float32"),) if with_dtype else ())
                         for t in s["tensors"]] for s in m["stages"]}


def load_raw(model_dir, map_path=MAP_PATH, check=True):
    """Read raw/<stage>/<name>.bin for every tensor of the map. model_dir may hold raw/ or be raw/ itself. With
    check=True every stage is also normalized (names and shapes as spec 41 needs them) and every value is checked to be
    finite and exactly representable in FP16."""
    root = os.path.join(model_dir, "raw")
    if not os.path.isdir(root):
        root = model_dir
    out = {}
    for stage, tensors in map_tensors(map_path, with_dtype=True).items():
        if stage not in STAGE_NAMES and stage != "net":
            raise ValueError(f"map group {stage!r} is not a model K stage")
        p = {}
        for name, shape, dtype in tensors:
            path = os.path.join(root, stage, name + ".bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"model data missing: {path}")
            if dtype == "int32":
                a = np.fromfile(path, dtype="<i4")
                if a.size != int(np.prod(shape)):
                    raise ValueError(f"model data damaged: {path} has {a.size} values, expected {int(np.prod(shape))}")
                p[name] = a.reshape(shape)
                continue
            a = np.fromfile(path, dtype="<f4")
            if a.size != int(np.prod(shape)):
                raise ValueError(f"model data damaged: {path} has {a.size} values, expected {int(np.prod(shape))}")
            a = a.reshape(shape)
            if check:
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"model data damaged: {path} has non-finite values")
                if not np.array_equal(a.astype(np.float16).astype(np.float32), a):
                    raise ValueError(f"model data: {path} has values that are not FP16")
            p[name] = a
        if check and stage != "net":
            normalize(stage, p)
        out[stage] = p
    if check:
        from clean.k.reference.layout import check_offset_table
        check_offset_table(out["net"]["window_offsets"])
    missing = [s for s in STAGE_NAMES if s not in out]
    if missing:
        raise ValueError(f"the map has no stage {missing}")
    return out


def write_raw(model_dir, params, map_path=MAP_PATH):
    """Write params ({stage: tensors}, canonical or converter shapes) in the converter's --raw layout (tests)."""
    for stage, tensors in map_tensors(map_path, with_dtype=True).items():
        p = params[stage] if stage == "net" else normalize(stage, params[stage])
        d = os.path.join(model_dir, "raw", stage)
        os.makedirs(d, exist_ok=True)
        for name, shape, dtype in tensors:
            np.asarray(p[name], dtype="<i4" if dtype == "int32" else "<f4").reshape(shape).tofile(
                os.path.join(d, name + ".bin"))
