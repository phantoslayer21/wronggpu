# SPDX-License-Identifier: MIT
"""Preset-F network-body wiring from specs 50.2/50.6, 51 and 55.

Inputs are the IN0 block's already prepared tokens and current-frame state stream.
Page 52's input frontend and pages 54/54b's output passes are outside this module.
No state-stream tensor is treated as previous-frame history. Constants, shifts and
weights are explicit logical inputs; this is not a converter or game interface.
"""
import numpy as np

from .stages import run_stage
from ..head.block import recurrent_shift
from ..head.head import d0r_stage

ENCODERS = ("IN0", "E1", "E2", "E3", "E4", "E5")
DECODER_LEVELS = (4, 3, 2, 1, 0)
ORDER = ENCODERS + tuple(name for level in DECODER_LEVELS for name in (f"D{level}-F", f"D{level}-R"))


def regroup_children(tokens):
    """51.5 E1 adapter input: four children in row-major order, channel last."""
    h, w, c = tokens.shape
    if h % 2 or w % 2:
        raise ValueError("E1 regrouping requires an even token extent")
    return np.concatenate([tokens[dy::2, dx::2] for dy in (0, 1) for dx in (0, 1)], axis=-1)


def run_body(tokens, state0, weights, constants, window_constants, shifts, *, s1, s2, render_size=None):
    """Return all stage outputs and final-head surfaces for one prepared frame.

    tokens: e4m3 uint8 [Ht,Wt,32], state0: f16 values [Ht,Wt,4].
    weights/constants are keyed by the sixteen names in ORDER. Base shifts are
    keyed by IN0/E1..E5 and D4..D0; recurrent halves apply (s+4) mod 8 exactly once.
    s1/s2 are supplied logical final-head scalars, never guessed converter data.
    render_size is (width,height); padding tokens remain part of Cnet, while the
    four head images are cropped to the render size per 55.3.
    """
    tokens = np.asarray(tokens)
    state0 = np.asarray(state0, dtype=np.float64)
    if tokens.dtype != np.uint8 or tokens.ndim != 3 or tokens.shape[2] != 32:
        raise ValueError("tokens must be uint8 e4m3 codes with 32 channels")
    ht, wt = tokens.shape[:2]
    if ht <= 0 or wt <= 0 or ht % 32 or wt % 32:
        raise ValueError("token extents must be positive multiples of 32")
    if state0.shape != (ht, wt, 4):
        raise ValueError("state0 must match the token extent and have four channels")
    wr, hr = (wt, ht) if render_size is None else render_size
    if not (0 < wr <= wt and 0 < hr <= ht):
        raise ValueError("render extent must lie inside the token extent")
    missing = set(ORDER) - (set(weights) & set(constants))
    if missing:
        raise ValueError(f"missing logical stage data: {sorted(missing)}")
    shift_keys = ENCODERS + tuple(f"D{k}" for k in DECODER_LEVELS)
    for key in shift_keys:
        if key not in shifts or len(shifts[key]) != 2 or any(v not in (0, 2, 4, 6) for v in shifts[key]):
            raise ValueError(f"invalid or missing base shift: {key}")

    outputs, state_by_level, skip_by_level = {}, {}, {}
    x, hold = tokens, state0
    for level, name in enumerate(ENCODERS):
        inputs = {"in" if name == "E1" else "x": x, "hold": hold}
        stage, _ = run_stage(name, inputs, weights[name], constants[name], window_constants, shifts[name])
        outputs[name] = stage
        skip_by_level[level], state_by_level[level] = stage["y2"], stage["hnew"]
        if name == "IN0":
            x = regroup_children(stage["y2"])
        elif level < 5:
            x = stage["merged"]
        if level < 5:
            hold = stage["hmerged"]

    coarse_tokens, coarse_state = outputs["E5"]["y2"], outputs["E5"]["hnew"]
    for level in DECODER_LEVELS:
        fusion, recurrent, key = f"D{level}-F", f"D{level}-R", f"D{level}"
        inputs = {"cin": coarse_tokens, "skip": skip_by_level[level]}
        stage, _ = run_stage(fusion, inputs, weights[fusion], constants[fusion], window_constants, shifts[key])
        outputs[fusion] = stage
        if level:
            inputs = {"x": stage["y2"], "hold": state_by_level[level], "Hc": coarse_state}
            stage, _ = run_stage(recurrent, inputs, weights[recurrent], constants[recurrent], window_constants, shifts[key])
            outputs[recurrent] = stage
            coarse_tokens, coarse_state = stage["y2"], stage["hnew"]
        else:
            c = constants[recurrent]
            head = d0r_stage(stage["y2"], weights[recurrent], window_constants.as_dict(),
                             c["ra"], c["m"], c["g"], recurrent_shift(shifts[key]),
                             state_by_level[0], coarse_state, s1, s2, wr, hr)
            outputs[recurrent] = head
    return {"stages": outputs, "head": outputs["D0-R"]}
