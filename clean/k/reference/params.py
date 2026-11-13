# SPDX-License-Identifier: MIT
"""Model K parameter tensors: the converter's names and shapes (spec 48 section 7.2) and the canonical form used by
the reference and the packer.

Canonical form (what clean/k/reference/network.py and clean/k/runner/k_pack.py read), per stage:
  g1 [C], g2 [C], Wv [H, 32, C], Wo [C, D], bo [C], W1 [4C, C], b1 [4C], W2 [C, 4C], b2 [C]
  softmax stages: Wq, Wk [H, 32, C], B [H, 64, 64];  fixed-mixing stages: M [H, 64, 64]   ([h, t, s], t = query)
  e0: We [32, 16], be [32];  e0..e4: Wm [C_next, 4C], bm [C_next];  d4..d0: Wx [4C, C_prev], bx [4C];
  d0: Wh [40, 32], bh [40]
`normalize` accepts the converter's shapes as well (B and M [H, 8, 8, 8, 8] with axes (h, row_t, col_t, row_s,
col_s); Wm [C_next, 4, C]; Wx [4, C, C_prev]; bx [4, C]) and the older name gamma1 for g1. Both index orders are the
same row-major flattening, so normalizing is a reshape.
"""
import numpy as np

from clean.k.reference.layout import EMBED_C, HEAD_W, INPUT_C, OUTPUT_C, STAGE


def expected_shapes(stage):
    """{name: canonical shape} of one stage."""
    st = STAGE[stage]
    C, H, D, HID = st["C"], st["heads"], st["D"], st["HID"]
    s = {"g1": (C,), "g2": (C,), "Wv": (H, HEAD_W, C), "Wo": (C, D), "bo": (C,), "W1": (HID, C), "b1": (HID,),
         "W2": (C, HID), "b2": (C,)}
    if st["mix"] == "fixed":
        s["M"] = (H, 64, 64)
    else:
        s.update({"Wq": (H, HEAD_W, C), "Wk": (H, HEAD_W, C), "B": (H, 64, 64)})
    if st.get("embed"):
        s.update({"We": (EMBED_C, INPUT_C), "be": (EMBED_C,)})
    if st["kind"] == "enc":
        s.update({"Wm": (st["C_next"], 4 * C), "bm": (st["C_next"],)})
    if st["kind"] == "dec":
        s.update({"Wx": (4 * C, st["C_prev"]), "bx": (4 * C,)})
    if st.get("head"):
        s.update({"Wh": (OUTPUT_C, C), "bh": (OUTPUT_C,)})
    return s


def normalize(stage, tensors):
    """Converter (spec 48) or canonical tensors -> canonical float32 arrays; raises ValueError on a missing tensor or
    a shape that is neither form. Extra tensors are ignored."""
    t = dict(tensors)
    if "g1" not in t and "gamma1" in t:
        t["g1"] = t["gamma1"]
    out = {}
    for name, shape in expected_shapes(stage).items():
        if name not in t:
            raise ValueError(f"{stage}: tensor {name} missing")
        a = np.asarray(t[name], dtype=np.float32)
        if a.shape != shape:
            if a.size != int(np.prod(shape)) or not _compatible(name, a.shape, shape):
                raise ValueError(f"{stage}: {name} has shape {a.shape}, expected {shape}")
            a = a.reshape(shape)
        out[name] = a
    return out


def _compatible(name, got, want):
    """The converter's shapes that flatten to the canonical ones."""
    if name in ("B", "M"):
        return got == (want[0], 8, 8, 8, 8)
    if name == "Wm":
        return got == (want[0], 4, want[1] // 4)
    if name == "Wx":
        return got == (4, want[0] // 4, want[1])
    if name == "bx":
        return got == (4, want[0] // 4)
    return False
