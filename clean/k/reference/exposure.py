# SPDX-License-Identifier: MIT
"""Model K: which exposure value is which (spec 42 draft 3 section 4), the host rule in front of the input stage
(instage.py) and the output stage (outstage.py), which take E_cur, E_prev, E_aux, e1, e2 as given.

HDR context:
  E_cur  = the game's exposure (texture texel or value); 1 if the game passes none. The auto-exposure value never
           takes its place.
  E_aux  = the auto-exposure value (spec 07) when auto-exposure runs; otherwise E_cur.
  E_prev = the previous frame's E_cur; E_cur itself on a history reset (and on the first frame).
  e1 = e2 = the game's exposure scale if it is positive and finite, else 1 (the per-feature factor that multiplies it
           is taken as 1).
LDR context: every value and factor 1.
Auto-exposure runs (HDR only) when the game asks for it at creation. Each value is then read as FP16 with 0 -> 1 by
the stages. So kappa = e1 E_cur / pi, lambda = e2 E_aux / (e1 E_cur), rho = (e1 E_cur) / (e1 E_prev).

The cases (HDR): no texture + auto-exposure -> E_cur = E_prev = 1, E_aux = measured (lambda = measured);
texture, no auto-exposure -> E_cur = E_aux = texture (lambda = 1); texture + auto-exposure -> E_cur = texture,
E_aux = measured; neither -> all 1.

This is the rule in NumPy, the one place that states it for the references; a caller (the whole-frame runner, the
runtime's exposure kernel) applies it to get the values the two stages take.
"""
import math


def exposure_scale(scale):
    """e1 = e2: the game's exposure scale if positive and finite, else 1."""
    try:
        s = float(scale)
    except (TypeError, ValueError):
        return 1.0
    return s if math.isfinite(s) and s > 0.0 else 1.0


def exposure_values(hdr, game_exposure=None, measured=None, scale=None):
    """{E_cur, E_aux, e1, e2} of one frame (E_prev is the previous frame's E_cur, kept by the caller).

    hdr: the HDR context flag; game_exposure: the exposure the game passes this frame (its texture's texel or a value;
    None: none); measured: the spec 07 auto-exposure value when auto-exposure runs this frame (None: it does not)."""
    if not hdr:
        return {"E_cur": 1.0, "E_aux": 1.0, "e1": 1.0, "e2": 1.0}
    E_cur = 1.0 if game_exposure is None else float(game_exposure)
    E_aux = E_cur if measured is None else float(measured)
    e = exposure_scale(scale)
    return {"E_cur": E_cur, "E_aux": E_aux, "e1": e, "e2": e}


def apply_rule(x, hdr):
    """x (a frame dict) with E_cur, E_aux, e1, e2 from the rule when it gives the rule's inputs instead: game_exposure
    (None or a value), measured (None or the auto-exposure value), exposure_scale (optional). A dict that already
    has E_cur is returned unchanged (the stage tests give the values directly)."""
    if "E_cur" in x and x["E_cur"] is not None:
        return x
    d = dict(x)
    d.update(exposure_values(hdr, x.get("game_exposure"), x.get("measured"), x.get("exposure_scale")))
    return d
