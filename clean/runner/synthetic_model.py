# SPDX-License-Identifier: MIT
"""Write a SYNTHETIC model directory in the layout of the converter's --raw output (spec 08 sections 7 and 10):
<dir>/raw/<stage>/<name>.bin, row-major little-endian float32 (uint8 for enc0 T), every tensor of spec 08 section 7.2.

Random values with the right shapes and value grids, for testing the runtime and pipeline.py without model data.
Never model data.

    python clean/runner/synthetic_model.py --out <dir> [--seed 1] [--output-taps analytic|random|w0,w1,w2,w3,w4]
                                           [--wgcm]

--output-taps adds the optional group `filter` / `output_taps [5]` of spec 08 draft 2 (raw/filter/output_taps.bin):
the output filter's weights for an even output index. The values are made up here ("analytic": spec 06's analytic
weights rounded to float32; "random": those plus a seeded perturbation, sum 1; or five numbers); the original's
weights only ever come from a player's converted model. --wgcm also writes model.wgcm (the container of
clean/converter/OUTPUT-FORMAT.md) and a synthetic map file next to it (write_wgcm).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clean.reference import decoder as Dc  # noqa: E402
from clean.reference import encoder as En  # noqa: E402
from clean.reference import post as Po  # noqa: E402
from clean.reference import swin as Sw  # noqa: E402
from clean.reference.fp8 import f16  # noqa: E402

SWIN_ORDER = ["enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"]
MAP_ORDER = ["enc0"] + SWIN_ORDER + ["dec0", "post"]   # stage order of the map (spec 08 section 7)
FILTER_STAGE, FILTER_TAPS = "filter", "output_taps"     # the optional group of spec 08 draft 2 (after "post")
SYNTHETIC_MAP = "synthetic.map.json"                    # written by write_wgcm next to model.wgcm


def output_taps(kind="analytic", seed=1):
    """Five made-up output-filter weights w(-2) ... w(+2) for an even output index, float32 [5].
    "analytic": spec 06's analytic set (reference/post.py filter_weights) rounded to float32; "random": the analytic
    set plus a seeded perturbation, renormalized to sum 1 (in float64, then rounded); a sequence of five numbers or
    the text "w0,w1,w2,w3,w4": those values."""
    if isinstance(kind, str) and "," in kind:
        kind = [float(v) for v in kind.split(",")]
    if not isinstance(kind, str):
        w = np.asarray(kind, np.float64).reshape(-1)
        if w.shape != (5,) or not np.isfinite(w).all():
            raise ValueError("output taps: five finite numbers")
        return w.astype(np.float32)
    w = Po.filter_weights(np.arange(1))[1][0]
    if kind == "random":
        w = w + np.random.default_rng([seed, 0x7A95]).normal(0.0, 0.05, 5)
        w = w / w.sum()
    elif kind != "analytic":
        raise ValueError(f"output taps: unknown kind {kind!r} (analytic, random, or five numbers)")
    return w.astype(np.float32)


def synthetic_tensors(seed=1, taps=None):
    """{stage: {name: array}} in the converter's raw shapes and dtypes. taps (see output_taps; None = no group) adds
    the optional group "filter"."""
    rng = np.random.default_rng(seed)
    out = {}
    p = En.random_params(rng)
    out["enc0"] = {"W": p["W"], "b": p["b"], "T": p["T"].astype(np.uint8),
                   "sigma1": np.asarray(p["sigma1"], np.float32).reshape(1),
                   "sigma2": np.asarray(p["sigma2"], np.float32).reshape(1)}
    for s in SWIN_ORDER:
        q = Sw.random_params(s, rng)
        out[s] = {k: np.asarray(v, np.float32) for k, v in q.items()}
        for k in ("gamma1", "gamma2", "B", "bp", "b1", "b2", "bm", "be"):   # FP16-exact like the converter's output
            if k in out[s]:
                out[s][k] = f16(out[s][k]).astype(np.float32)
    d = Dc.random_params(rng)
    out["dec0"] = {"W": d["W"].reshape(4, 17, 64), "b": d["b"].reshape(4, 17),
                   "S": np.asarray(d["S"], np.float32).reshape(1), "T": np.asarray(d["T"], np.float32).reshape(1)}
    out["post"] = {"s": np.array([rng.uniform(0.5, 2.0)], np.float32)}
    if taps is not None:
        out[FILTER_STAGE] = {FILTER_TAPS: output_taps(taps, seed)}
    return out


def stage_order(tensors):
    """The stages of a tensor set in map order: MAP_ORDER, then the optional group "filter" when it is there."""
    return MAP_ORDER + ([FILTER_STAGE] if FILTER_STAGE in tensors else [])


def write_raw(model_dir, tensors):
    for stage in stage_order(tensors):
        d = os.path.join(model_dir, "raw", stage)
        os.makedirs(d, exist_ok=True)
        for name, a in tensors[stage].items():
            a = np.ascontiguousarray(a, dtype=np.uint8 if a.dtype == np.uint8 else "<f4")
            a.tofile(os.path.join(d, name + ".bin"))


def read_raw(model_dir):
    """Read a raw model directory back into {stage: {name: array}} (shapes from synthetic_tensors)."""
    shapes = synthetic_tensors(0)
    root = os.path.join(model_dir, "raw") if os.path.isdir(os.path.join(model_dir, "raw")) else model_dir
    out = {}
    for stage, ts in shapes.items():
        out[stage] = {}
        for name, a in ts.items():
            dt = np.uint8 if a.dtype == np.uint8 else "<f4"
            out[stage][name] = np.fromfile(os.path.join(root, stage, name + ".bin"), dtype=dt).astype(
                np.uint8 if dt == np.uint8 else np.float32).reshape(a.shape)
    taps = os.path.join(root, FILTER_STAGE, FILTER_TAPS + ".bin")
    if os.path.exists(taps):   # the optional group
        out[FILTER_STAGE] = {FILTER_TAPS: np.fromfile(taps, dtype="<f4").astype(np.float32).reshape(5)}
    return out


def write_wgcm(model_dir, tensors, key=None, other_filter_variants=1, filter_variant=None, map_file=None,
               list_filter=True, order=None):
    """Write <model_dir>/model.wgcm (clean/converter/OUTPUT-FORMAT.md) from a tensor set, and a synthetic map file
    <model_dir>/synthetic.map.json that carries what the runtime reads from a map: the stage list with the keyed
    digests. Returns the map's path (Pipeline(map_path=...)). Test aid only: the real file comes from the converter.

    The group "filter" gets a digest LIST in the map, as spec 08 draft 2 section 5 gives a group whose source has
    variants: `other_filter_variants` made-up entries plus this file's own, which sits at index `filter_variant`
    (default: last). The map always lists the group, also when the tensor set has none (as the shipped map does).
    map_file: no synthetic map is written; the file records that it was converted with this existing map (its
    digests will not match: for tests of how the runtime reads a given map).
    list_filter=False: a map as before spec 08 draft 2 ("version 1"): thirteen stages, no entry for the group.
    order: the stages of another model (e.g. Model L's map order) for the file and the map, in place of Model M's."""
    key = hashlib.sha256(b"WrongGPU synthetic model key").digest() if key is None else bytes(key)
    stage_list = list(order) if order is not None else MAP_ORDER
    if order is None:
        order = stage_order(tensors)
    else:
        order = [s for s in stage_list if s in tensors] + ([FILTER_STAGE] if FILTER_STAGE in tensors else [])
    data, index, digests = bytearray(), bytearray(), {}

    def put_str(text):
        b = text.encode()
        return len(b).to_bytes(2, "little") + b

    blobs = []   # (stage, name, dtype code, shape, bytes)
    for stage in order:
        h = hashlib.sha256(key)
        entries = []
        for name, a in tensors[stage].items():
            u8 = a.dtype == np.uint8
            raw = np.ascontiguousarray(a, dtype=np.uint8 if u8 else "<f4").tobytes()
            h.update(raw)
            entries.append((name, 1 if u8 else 0, list(a.shape), raw))
        digests[stage] = h.digest()
        blobs.append((stage, entries))
    # index size first (offsets depend on it): every entry's size is known without the offsets
    isize = sum(len(put_str(stage)) + 32 + 4 + sum(len(put_str(n)) + 2 + 4 * len(sh) + 16 for n, _, sh, _ in entries)
                for stage, entries in blobs)
    pos = 88 + isize
    for stage, entries in blobs:
        index += put_str(stage) + digests[stage] + len(entries).to_bytes(4, "little")
        for name, code, shape, raw in entries:
            pad = -pos % 64
            data += bytes(pad)
            pos += pad
            index += put_str(name) + bytes([code, len(shape)]) + b"".join(int(d).to_bytes(4, "little") for d in shape)
            index += pos.to_bytes(8, "little") + len(raw).to_bytes(8, "little")
            data += raw
            pos += len(raw)
    assert len(index) == isize
    stages = [{"stage": s, "digest": digests[s].hex()} for s in stage_list if s in digests]
    others = [hashlib.sha256(b"another variant %d" % i).hexdigest() for i in range(int(other_filter_variants))]
    own = [digests[FILTER_STAGE].hex()] if FILTER_STAGE in tensors else []
    at = len(others) if filter_variant is None else int(filter_variant)
    if list_filter:
        stages.append({"stage": FILTER_STAGE, "digest": others[:at] + own + others[at:]})
    map_bytes = json.dumps({"format": "wronggpu-converter-map", "version": 4 if list_filter else 1,
                            "model": "synthetic (tests)", "stages": stages}, indent=1).encode()
    os.makedirs(model_dir, exist_ok=True)
    map_path = map_file or os.path.join(model_dir, SYNTHETIC_MAP)
    if map_file:
        with open(map_file, "rb") as f:
            map_bytes = f.read()
    else:
        with open(map_path, "wb") as f:
            f.write(map_bytes)
    body = (b"WGCMODEL" + (1).to_bytes(4, "little") + len(order).to_bytes(4, "little") + key +
            hashlib.sha256(map_bytes).digest() + isize.to_bytes(8, "little") + bytes(index) + bytes(data))
    with open(os.path.join(model_dir, "model.wgcm"), "wb") as f:
        f.write(body + hashlib.sha256(body).digest())
    return map_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output-taps", help="add the group filter / output_taps: analytic, random, or w0,w1,w2,w3,w4")
    ap.add_argument("--wgcm", action="store_true", help="also write model.wgcm and its synthetic map")
    args = ap.parse_args(argv)
    tensors = synthetic_tensors(args.seed, args.output_taps)
    write_raw(args.out, tensors)
    print(f"synthetic raw model written to {os.path.join(args.out, 'raw')}")
    if args.wgcm:
        print(f"model.wgcm written to {args.out}; map: {write_wgcm(args.out, tensors)}")


if __name__ == "__main__":
    main()
