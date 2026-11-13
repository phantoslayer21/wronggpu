# SPDX-License-Identifier: MIT
"""Comparison tool of the equivalence tests (clean/EQUIVALENCE-PROTOCOL.md, part "Tools", item 3).

Reads two sets of values (ours, and the original's saved on the spec team's side), computes the verdicts MATCHES /
CLOSE / DIFFERENT by the protocol's rules, and prints the verdict table first, then the numbers of parts 1-5. It never
prints, logs or returns a value of either side: every report field is a count, a share, a statistic of the step
differences, a value-range class label (for example "|value| in [2^-3, 2^-2)"), a location index or a field name.

    python -m clean.equiv.compare --manifest M.json --ours DIR --orig DIR [--json report.json]

The manifest (JSON) lists what to compare; file names are relative to --ours and to --orig (the same names on both):

    {"model": "Model M", "run": "isolated" | "chained",
     "close_band":    {"enc0.*": {"fp8": 2, "fp16": 3}, "post.history": 4, "*": {"unorm8": 1}},   # optional
     "stages":        [{"name": "enc0.features", "file": "enc0.npy", "format": "fp8", "layout": "FHWC",
                        "border": 8, "window": [8, 8], "shift": [4, 4],
                        "close_steps": 2, "final": false, "verdict": true}, ...],  # compute order; last three optional
     "temporal":      [{"name": "history", "case": "steady", "file": "...", "format": "fp16", "layout": "FHWC"}],
     "probes":        [{"name": "edge_h_v1", "file": "...", "format": "fp16", "layout": "FHWC"}],
     "perturbations": [{"name": "colour +10 %", "before": "b.npy", "after": "a.npy", "format": "fp16",
                        "layout": "HWC", "center": [y, x]}],
     "sequences":     [{"name": "profile 1", "file": "answers.json"}]}

Formats: clean/equiv/steps.py. Layout letters: F (frame), H (row), W (column), C (channel), each at most once, in the
array's axis order; compute order inside a stage is frame, row, column, channel.

Rules (the protocol's table "What every test answers"; tool choices marked *):
  MATCHES    all values equal; or >= 99.99 % equal, every other value 1 step off, and no bias: the counts of +1 and -1
             differences n+ and n- satisfy |n+ - n-| <= 3.29 sqrt(n+ + n-) (two-sided 0.1 % level)*
  CLOSE      >= 99 % equal, every difference within the stage's CLOSE band (default 2 steps; none non-finite), |mean signed difference| <= 0.01 steps (mean
             over all compared values), and no class of a family (channel; border band vs interior; value range;
             window position) holding more than 3 times its share of the values among the differences; a class is only
             judged with at least 10 differences in it*
  DIFFERENT  anything else, with the reasons and its size: the share off by more than the band and by more than 16
             steps, the largest difference, and a size word*: "minor" (none over 16 steps and under 1 % over the band),
             "massive" (at least 10 % over 16 steps), "major" otherwise
  CLOSE band (spec team's table, relayed): per stage and format, in steps. Lookup: the entry's "close_steps"; else the
             first "close_band" key matching the stage name (exact name first, then shell patterns such as "b*" or "*"
             in manifest order) whose value is a number or has the entry's format; else 2.
  Non-finite values: NaN against NaN and an infinity against the same infinity are equal; the share non-finite on
             either side is reported separately; FP8 steps count across zero with +-0 one value, and the NaN code is
             non-finite, never a step (spec team confirmation 2026-11-04).
  Temporal images (part 2): worst per-frame verdict; DIFFERENT also when the mean absolute difference of the last 100
  frames is more than 1.1 times that of frames 11 to 110 (needs at least 110 frames, else reported as not evaluated).
  Probes (part 3): worst per-frame verdict.
  Perturbations (part 4): MATCHES when the change images are equal; CLOSE when correlation >= 0.99, ratio of total
  absolute change in [0.95, 1.05], the same 99 % radius (whole output pixels, rounded up) and under 0.1 % of values with
  opposite signs; DIFFERENT otherwise (size word*: "minor" when correlation >= 0.95 and ratio in [0.8, 1.25], "massive"
  when correlation < 0.5 or the ratio is outside [0.5, 2] or a radius is missing, "major" otherwise).
  Sequences (part 5): MATCHES when every answer is the same, DIFFERENT otherwise.
  Channels: every stage, temporal image and probe with a C axis of 2 or more channels also gets one verdict row per
  channel, judged by the same rules on that channel alone (so one wrong channel of three cannot hide under the 3-times
  rule).
  Chained runs: intermediate stages get every number but no verdict ("numbers only", outside the model verdict);
  verdicts go to isolated stages and final outputs. In a chained manifest mark the final outputs "final": true (or
  any entry "verdict": true / false to override). Temporal images, probes, perturbations and sequences keep verdicts.
  Model: the worst verdict of every row; ERROR (a comparison that could not run, e.g. differing shapes) outranks all.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from clean.equiv import steps as S

MATCHES, CLOSE, DIFFERENT, ERROR = "MATCHES", "CLOSE", "DIFFERENT", "ERROR"
NUMBERS_ONLY = "numbers only"      # chained intermediate: no verdict, not part of the model verdict
RANK = {MATCHES: 0, CLOSE: 1, DIFFERENT: 2, ERROR: 3}
DEFAULT_BAND = 2
BUCKETS = ("0", "1", "2", "3-4", "5-8", "9-16", ">16")
_EDGES = np.array([0, 1, 2, 4, 8, 16])


class Rules:
    match_equal = (9999, 10000)   # share equal for MATCHES with 1-step ties, as an integer ratio
    close_equal = (99, 100)
    close_max_step = 2
    close_bias = 0.01
    concentration = 3.0
    min_class = 10
    bias_z = 3.29
    trend_ratio = 1.1
    trend_early = (10, 110)       # frames 11..110 (0-based slice)
    trend_late = 100
    pert_corr = 0.99
    pert_ratio = (0.95, 1.05)
    pert_opposite = (1, 1000)     # under 0.1 %
    pert_radius_share = 0.99


def with_band(rules, band):
    """The rules with another CLOSE band (steps)."""
    if band is None or band == rules.close_max_step:
        return rules
    return type("RulesBand", (rules,), {"close_max_step": int(band)})


def band_for(man, name, fmt):
    """CLOSE band of a stage from the manifest's "close_band" table (see the module docstring); None = default."""
    import fnmatch
    table = man.get("close_band") or {}
    keys = ([name] if name in table else []) + [k for k in table if k != name and fnmatch.fnmatchcase(name, k)]
    for k in keys:
        v = table[k]
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, dict) and fmt in v:
            return int(v[fmt])
    return None


def worst(verdicts):
    vs = [v for v in verdicts if v in RANK]
    return max(vs, key=RANK.__getitem__) if vs else NUMBERS_ONLY


# ---------------------------------------------------------------------------------------------------- layout

def canonical(a, layout):
    """Array with axes in `layout` order -> 4-D array (F, H, W, C), and which of F/H/W/C were present."""
    a = np.asarray(a)
    layout = layout or ""
    if layout == "" and a.ndim == 1:
        layout = "W"
    if len(layout) != a.ndim or len(set(layout)) != len(layout) or set(layout) - set("FHWC"):
        raise S.FormatError(f"layout {layout!r} does not fit an array with {a.ndim} axes (letters F, H, W, C)")
    present = [c for c in "FHWC" if c in layout]
    t = np.transpose(a, [layout.index(c) for c in present])
    shape = [t.shape[present.index(c)] if c in present else 1 for c in "FHWC"]
    return t.reshape(shape), set(layout)


# ---------------------------------------------------------------------------------------------------- one stage

def _families(shape4, present, border, window, shift, range_key, range_label):
    """Class families of part 1 item 5: name -> (key array (F,H,W,C), label function)."""
    F, H, W, C = shape4
    fam = {}
    if C > 1:
        fam["channel"] = (np.broadcast_to(np.arange(C).reshape(1, 1, 1, C), shape4), lambda k: f"channel {k}")
    if "H" in present and "W" in present and H > 2 * border and W > 2 * border and border > 0:
        y = np.arange(H).reshape(1, H, 1, 1)
        x = np.arange(W).reshape(1, 1, W, 1)
        band = (y < border) | (y >= H - border) | (x < border) | (x >= W - border)
        fam["border band"] = (np.broadcast_to(band.astype(np.int64), shape4),
                              lambda k, b=border: f"border band ({b} px)" if k else "interior")
    fam["value range"] = (range_key, range_label)
    if window and "H" in present and "W" in present:
        wh, ww = window
        sy, sx = shift or (0, 0)
        yy = (np.arange(H) - sy) % H
        xx = (np.arange(W) - sx) % W
        nwy, nwx = -(-H // wh), -(-W // ww)
        wy, wx = yy // wh, xx // ww
        bw = ((wy == 0) | (wy == nwy - 1)).reshape(1, H, 1, 1) | ((wx == 0) | (wx == nwx - 1)).reshape(1, 1, W, 1)
        fam["window"] = (np.broadcast_to(bw.astype(np.int64), shape4),
                         lambda k: "border windows" if k else "interior windows")
        ey = ((yy % wh == 0) | (yy % wh == wh - 1)).reshape(1, H, 1, 1)
        ex = ((xx % ww == 0) | (xx % ww == ww - 1)).reshape(1, 1, W, 1)
        fam["position in window"] = (np.broadcast_to((ey | ex).astype(np.int64), shape4),
                                     lambda k: "window edge rows / columns" if k else "window interior")
    return fam


def compare_stage(ours, orig, fmt, layout="", border=8, window=None, shift=None, rules=Rules):
    """Part-1 numbers of one stage output. Returns a dict of counts and statistics only (no values)."""
    ours, orig = np.asarray(ours), np.asarray(orig)
    if ours.shape != orig.shape:
        return {"error": f"shapes differ: ours {list(ours.shape)}, original {list(orig.shape)}"}
    try:
        o1, nf1 = S.ordinals(ours, fmt)
        o2, nf2 = S.ordinals(orig, fmt)
        rk, rlabel = S.range_classes(orig, fmt)
        o1, present = canonical(o1, layout)
        o2, _ = canonical(o2, layout)
        nf1, _ = canonical(nf1, layout)
        nf2, _ = canonical(nf2, layout)
        rk, _ = canonical(rk, layout)
    except S.FormatError as e:
        return {"error": str(e)}
    n = int(o1.size)
    if n == 0:
        return {"error": "no values"}
    fin = (nf1 == S.NF_NONE) & (nf2 == S.NF_NONE)
    eq = np.where(fin, o1 == o2, nf1 == nf2)
    nfm = ~eq & ~fin                                  # a non-finite value on one side, or different non-finite kinds
    diff = np.where(fin, o1 - o2, 0)
    absd = np.abs(diff)
    n_eq = int(np.count_nonzero(eq))
    n_d = n - n_eq
    n_nfm = int(np.count_nonzero(nfm))
    n_fin = int(np.count_nonzero(fin))
    hist = np.bincount(np.searchsorted(_EDGES, absd[fin], side="left"), minlength=len(BUCKETS)).astype(np.int64)
    hist[-1] += n_nfm
    st = {
        "format": fmt, "shape_fhwc": list(o1.shape), "values": n, "equal": n_eq, "share_equal": n_eq / n,
        "histogram": dict(zip(BUCKETS, (int(v) for v in hist))),
        "nonfinite_ours": int(np.count_nonzero(nf1 != S.NF_NONE)),
        "nonfinite_orig": int(np.count_nonzero(nf2 != S.NF_NONE)),
        "nonfinite_either": int(np.count_nonzero((nf1 != S.NF_NONE) | (nf2 != S.NF_NONE))),
        "nonfinite_mismatch": n_nfm,
        "over_2": int(np.count_nonzero(absd > 2)) + n_nfm,
        "band": int(rules.close_max_step),
        "over_band": int(np.count_nonzero(absd > rules.close_max_step)) + n_nfm,
        "over_16": int(np.count_nonzero(absd > 16)) + n_nfm,
    }
    if n_nfm:
        st["max_step"] = None                       # non-finite on one side
        loc = np.argmax(nfm.reshape(-1))
    else:
        st["max_step"] = int(absd.max())
        loc = np.argmax(absd.reshape(-1))
    st["max_step_range"] = rlabel(int(rk.reshape(-1)[loc])) if n_d else None
    st["mean_signed"] = float(int(diff.sum()) / n_fin) if n_fin else 0.0
    st["mean_abs"] = math.inf if n_nfm else (float(int(absd.sum()) / n_fin) if n_fin else 0.0)
    st["positive"] = int(np.count_nonzero(diff > 0))
    st["negative"] = int(np.count_nonzero(diff < 0))
    C = o1.shape[3]
    fin_c = fin.reshape(-1, C)
    diff_c = diff.reshape(-1, C)
    st["mean_signed_per_channel"] = [float(int(diff_c[:, c].sum()) / max(1, int(np.count_nonzero(fin_c[:, c]))))
                                     for c in range(C)]
    # item 5: where the differences are
    fams = _families(o1.shape, present, border, window, shift, rk, rlabel)
    st["classes"], st["concentrated"] = {}, []
    ne = ~eq
    for name, (keys, label) in fams.items():
        k = keys.reshape(-1).astype(np.int64)
        kmin = int(k.min())
        nk = np.bincount(k - kmin)
        dk = np.bincount(k[ne.reshape(-1)] - kmin, minlength=len(nk))
        rows = []
        for i in np.nonzero(nk)[0]:
            row = {"class": label(int(i) + kmin), "values": int(nk[i]), "differences": int(dk[i])}
            rows.append(row)
            if n_d and dk[i] >= rules.min_class and dk[i] * n > rules.concentration * n_d * nk[i]:
                st["concentrated"].append({"family": name, **row})
        st["classes"][name] = rows
    # item 6: first difference in compute order (frame, row, column, channel); index only
    if n_d:
        f, y, x, c = (int(v) for v in np.unravel_index(int(np.argmax(ne.reshape(-1))), o1.shape))
        st["first_difference"] = {"frame": f, "row": y, "column": x, "channel": c}
    else:
        st["first_difference"] = None
    v, reasons = stage_verdict(st, rules)
    st["verdict"], st["reasons"] = v, reasons
    st["size"] = size_word(st) if v == DIFFERENT else None
    return st


def _unbiased(n_pos, n_neg, rules=Rules):
    return abs(n_pos - n_neg) <= rules.bias_z * math.sqrt(n_pos + n_neg)


def stage_verdict(st, rules=Rules):
    """-> (verdict, reasons). Reasons name the failed CLOSE conditions (DIFFERENT) or why not MATCHES (CLOSE)."""
    if "error" in st:
        return ERROR, [st["error"]]
    n, n_eq = st["values"], st["equal"]
    if n_eq == n:
        return MATCHES, []
    a, b = rules.match_equal
    one_step = st["nonfinite_mismatch"] == 0 and st["max_step"] == 1
    unbiased = _unbiased(st["positive"], st["negative"], rules)
    if one_step and n_eq * b >= a * n and unbiased:
        return MATCHES, []
    why_not_match = []
    if not one_step:
        why_not_match.append("differences over 1 step")
    if n_eq * b < a * n:
        why_not_match.append("under 99.99 % equal")
    if not unbiased:
        why_not_match.append("1-step differences biased to one sign")
    fails = []
    a, b = rules.close_equal
    if n_eq * b < a * n:
        fails.append("under 99 % equal")
    if st["nonfinite_mismatch"]:
        fails.append("non-finite on one side")
    elif st["max_step"] > rules.close_max_step:
        fails.append(f"differences over {rules.close_max_step} steps")
    if abs(st["mean_signed"]) > rules.close_bias:
        fails.append(f"mean signed difference beyond +-{rules.close_bias} steps")
    for c in st["concentrated"]:
        fails.append(f"differences concentrated in {c['family']}: {c['class']}")
    if fails:
        return DIFFERENT, fails
    return CLOSE, why_not_match


def size_word(st):
    n = st["values"]
    if st["over_16"] == 0 and st.get("over_band", st["over_2"]) * 100 < n:
        return "minor"
    if st["over_16"] * 10 >= n:
        return "massive"
    return "major"


# ---------------------------------------------------------------------------------------------------- series

def compare_series(ours, orig, fmt, layout, trend=False, border=8, window=None, shift=None, rules=Rules):
    """Parts 2 and 3: per-frame comparison of an array with an F axis. Full part-1 numbers for frames 1, 10, 100,
    300, 600 (1-based, those that exist) and for the worst frame; share equal and mean absolute difference per frame."""
    ours, orig = np.asarray(ours), np.asarray(orig)
    if ours.shape != orig.shape:
        return {"error": f"shapes differ: ours {list(ours.shape)}, original {list(orig.shape)}"}
    if "F" not in (layout or ""):
        return {"error": "a temporal image or probe needs an F (frame) axis"}
    fa = layout.index("F")
    sub = layout.replace("F", "")
    nfr = ours.shape[fa]
    frames, full = [], {}
    first_frame, first_st = None, None
    for i in range(nfr):
        st = compare_stage(np.take(ours, i, axis=fa), np.take(orig, i, axis=fa), fmt, sub, border, window, shift, rules)
        if "error" in st:
            return {"error": f"frame {i + 1}: {st['error']}"}
        frames.append({"share_equal": st["share_equal"], "mean_abs": st["mean_abs"], "verdict": st["verdict"]})
        if (i + 1) in (1, 10, 100, 300, 600):
            full[i + 1] = st
        if first_frame is None and st["first_difference"] is not None:
            first_frame = i + 1
            first_st = st
    out = {"frames": nfr, "per_frame": frames, "numbers_at_frames": full}
    verdicts = [f["verdict"] for f in frames]
    v = worst(verdicts)
    wi = max(range(nfr), key=lambda i: (RANK[verdicts[i]], frames[i]["mean_abs"]))
    out["worst_frame"] = wi + 1
    if (wi + 1) not in full:
        full[wi + 1] = compare_stage(np.take(ours, wi, axis=fa), np.take(orig, wi, axis=fa), fmt, sub, border, window,
                                     shift, rules)
    reasons = list(full[wi + 1]["reasons"]) if v != MATCHES else []
    if first_frame is not None:
        out["first_difference"] = {"frame": first_frame, **{k: first_st["first_difference"][k]
                                                            for k in ("row", "column", "channel")}}
    else:
        out["first_difference"] = None
    out["trend"] = None
    if trend:
        e0, e1 = rules.trend_early
        if nfr < e1:
            out["trend"] = {"evaluated": False, "why": f"{nfr} frames; needs at least {e1}"}
        else:
            early = float(np.mean([f["mean_abs"] for f in frames[e0:e1]]))
            late = float(np.mean([f["mean_abs"] for f in frames[-rules.trend_late:]]))
            grows = late > rules.trend_ratio * early if math.isfinite(early) else False
            ratio = (late / early) if early > 0 else (None if late == 0 else math.inf)
            out["trend"] = {"evaluated": True, "early_mean_abs": early, "late_mean_abs": late, "ratio": ratio,
                            "grows": bool(grows)}
            if grows:
                reasons = (reasons if v == DIFFERENT else []) + ["difference grows over time"]
                v = DIFFERENT
    out["verdict"], out["reasons"] = v, reasons
    if v == DIFFERENT:
        ws = full[wi + 1]
        out["size"] = size_word(ws) if ws["verdict"] == DIFFERENT else "minor"
    else:
        out["size"] = None
    return out


# ---------------------------------------------------------------------------------------------------- perturbation

def _radius(change4, center, share):
    mag = np.abs(change4).sum(axis=(0, 3))            # per output pixel
    tot = float(mag.sum())
    if tot == 0:
        return None
    H, W = mag.shape
    cy, cx = center
    yy, xx = np.meshgrid(np.arange(H) - cy, np.arange(W) - cx, indexing="ij")
    d = np.sqrt(yy.astype(np.float64) ** 2 + xx.astype(np.float64) ** 2).reshape(-1)
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(mag.reshape(-1)[order])
    i = int(np.searchsorted(cum, share * tot, side="left"))
    return int(math.ceil(d[order][min(i, d.size - 1)]))


def compare_perturbation(ours_before, ours_after, orig_before, orig_after, fmt, layout, center, rules=Rules):
    """Part 4: compares the change each side makes to its own output (after minus before), never the outputs."""
    arrs = [np.asarray(a) for a in (ours_before, ours_after, orig_before, orig_after)]
    if len({a.shape for a in arrs}) != 1:
        return {"error": "the four images do not have the same shape"}
    try:
        vals = [canonical(S.decoded(a, fmt), layout)[0] for a in arrs]
    except S.FormatError as e:
        return {"error": str(e)}
    with np.errstate(invalid="ignore", over="ignore"):
        co = vals[1] - vals[0]
        cr = vals[3] - vals[2]
    if not (np.all(np.isfinite(co)) and np.all(np.isfinite(cr))):
        return {"error": "a change image is not finite"}
    n = int(co.size)
    st = {"values": n}
    if np.array_equal(co, cr):
        st.update(correlation=1.0, ratio=1.0, opposite=0, verdict=MATCHES, reasons=[], size=None)
        st["radius_ours"] = st["radius_orig"] = _radius(co, center, rules.pert_radius_share)
        return st
    a, b = co.reshape(-1), cr.reshape(-1)
    sa, sb = float(a.std()), float(b.std())
    st["correlation"] = float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else None
    ta, tb = float(np.abs(a).sum()), float(np.abs(b).sum())
    st["ratio"] = ta / tb if tb > 0 else None
    st["radius_ours"] = _radius(co, center, rules.pert_radius_share)
    st["radius_orig"] = _radius(cr, center, rules.pert_radius_share)
    st["opposite"] = int(np.count_nonzero(np.sign(a) * np.sign(b) < 0))
    fails = []
    if st["correlation"] is None or st["correlation"] < rules.pert_corr:
        fails.append(f"correlation under {rules.pert_corr}")
    lo, hi = rules.pert_ratio
    if st["ratio"] is None or not lo <= st["ratio"] <= hi:
        fails.append(f"ratio of total change outside [{lo}, {hi}]")
    if st["radius_ours"] is None or st["radius_ours"] != st["radius_orig"]:
        fails.append("99 % radius differs")
    pa, pb = rules.pert_opposite
    if st["opposite"] * pb >= pa * n:
        fails.append("opposite signs in 0.1 % of values or more")
    if fails:
        r, c = st["ratio"], st["correlation"]
        if c is not None and c >= 0.95 and r is not None and 0.8 <= r <= 1.25 and st["radius_ours"] is not None:
            size = "minor"
        elif c is None or c < 0.5 or r is None or not 0.5 <= r <= 2.0 or None in (st["radius_ours"], st["radius_orig"]):
            size = "massive"
        else:
            size = "major"
        st.update(verdict=DIFFERENT, reasons=fails, size=size)
    else:
        st.update(verdict=CLOSE, reasons=["change images not equal"], size=None)
    return st


# ---------------------------------------------------------------------------------------------------- sequences

def compare_sequence(ours_calls, orig_calls):
    """Part 5: two lists of answer records (dicts, one per call). No CLOSE: every answer must be the same."""
    n = max(len(ours_calls), len(orig_calls))
    first, ndiff = None, 0
    for i in range(n):
        a = ours_calls[i] if i < len(ours_calls) else None
        b = orig_calls[i] if i < len(orig_calls) else None
        if a == b:
            continue
        ndiff += 1
        if first is None:
            if a is None or b is None:
                field = "(call missing on " + ("our" if a is None else "the original's") + " side)"
            else:
                keys = list(b.keys()) + [k for k in a.keys() if k not in b]
                field = next(k for k in keys if a.get(k, _MISSING) != b.get(k, _MISSING))
            first = {"call": i, "field": field}
    st = {"calls_ours": len(ours_calls), "calls_orig": len(orig_calls), "differing_calls": ndiff,
          "first_difference": first}
    st["verdict"] = MATCHES if ndiff == 0 else DIFFERENT
    st["reasons"] = [] if ndiff == 0 else ["answers differ"]
    st["size"] = None if ndiff == 0 else f"{ndiff} of {n} calls"
    return st


_MISSING = object()


# ---------------------------------------------------------------------------------------------------- channels

def compare_channels(ours, orig, layout, fn):
    """Each channel on its own (protocol question 5, project decision 2026-11-04): fn(ours_c, orig_c, sub_layout)
    is compare_stage or compare_series bound to the entry's other arguments. -> list of per-channel results, empty when
    the layout has no C axis or only one channel. A wrong channel then cannot hide under the 3-times rule."""
    layout = layout or ""
    if "C" not in layout:
        return []
    ours, orig = np.asarray(ours), np.asarray(orig)
    ax = layout.index("C")
    if ours.shape != orig.shape or ours.shape[ax] < 2:
        return []
    sub = layout.replace("C", "")
    out = []
    for c in range(ours.shape[ax]):
        st = fn(np.take(ours, c, axis=ax), np.take(orig, c, axis=ax), sub)
        if st.get("first_difference"):
            st["first_difference"] = dict(st["first_difference"], channel=c)
        st["channel"] = c
        out.append(st)
    return out


def _channel_rows(parent, chans):
    rows = []
    for st in chans:
        st.setdefault("verdict", ERROR)
        if "error" in st:
            st.setdefault("reasons", [st["error"]])
        r = _row("channel", f"{parent} / ch {st['channel']}", st)
        r["parent"], r["channel"] = parent, st["channel"]
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------------- manifest

def _load(d, name):
    return np.load(os.path.join(d, name), allow_pickle=False)


def run_manifest(man, ours_dir, orig_dir, rules=Rules):
    rows = []
    chained = man.get("run") == "chained"
    first_diff = None

    def entry_rules(e):
        band = e.get("close_steps")
        if band is None:
            band = band_for(man, e["name"], e.get("format"))
        return with_band(rules, band)

    for e in man.get("stages", []):
        chans = []
        er = entry_rules(e)

        def stage():
            a, b = _load(ours_dir, e["file"]), _load(orig_dir, e["file"])
            fn = lambda x, y, lay: compare_stage(x, y, e["format"], lay, e.get("border", 8), e.get("window"),
                                                 e.get("shift"), er)
            chans.extend(compare_channels(a, b, e.get("layout", ""), fn))
            return fn(a, b, e.get("layout", ""))
        st = _guard(stage)
        st.setdefault("verdict", ERROR)
        if "error" in st:
            st["reasons"] = [st["error"]]
        crow = _channel_rows(e["name"], chans)
        judged = e.get("verdict", (not chained) or bool(e.get("final")))
        if not judged and st["verdict"] != ERROR:   # chained intermediate: numbers, no verdict
            for x in [st] + [r["numbers"] for r in crow]:
                x["verdict"], x["size"], x["reasons"] = NUMBERS_ONLY, None, []
            for r in crow:
                r["verdict"], r["size"], r["reasons"] = NUMBERS_ONLY, None, []
        if first_diff is None and st.get("first_difference"):
            first_diff = {"stage": e["name"], **st["first_difference"]}
        rows.append(_row("stage", e["name"], st, "final output" if chained and e.get("final") else None))
        rows.extend(crow)
    for kind, trend in (("temporal", True), ("probes", False)):
        for e in man.get(kind, []):
            name = e["name"] + (f" / {e['case']}" if e.get("case") else "")
            chans = []
            er = entry_rules(e)

            def series():
                a, b = _load(ours_dir, e["file"]), _load(orig_dir, e["file"])
                fn = lambda x, y, lay: compare_series(x, y, e["format"], lay, trend, e.get("border", 8),
                                                      e.get("window"), e.get("shift"), er)
                chans.extend(compare_channels(a, b, e.get("layout", "FHWC"), fn))
                return fn(a, b, e.get("layout", "FHWC"))
            rows.append(_row("temporal" if kind == "temporal" else "probe", name, _guard(series)))
            rows.extend(_channel_rows(name, chans))
    for e in man.get("perturbations", []):
        st = _guard(lambda: compare_perturbation(_load(ours_dir, e["before"]), _load(ours_dir, e["after"]),
                                                 _load(orig_dir, e["before"]), _load(orig_dir, e["after"]),
                                                 e["format"], e.get("layout", "HWC"), tuple(e["center"]), rules))
        rows.append(_row("perturbation", e["name"], st))
    for e in man.get("sequences", []):
        def seq():
            with open(os.path.join(ours_dir, e["file"]), encoding="utf-8") as f:
                a = json.load(f)
            with open(os.path.join(orig_dir, e["file"]), encoding="utf-8") as f:
                b = json.load(f)
            return compare_sequence(a, b)
        rows.append(_row("sequence", e["name"], _guard(seq)))
    return {"model": man.get("model", "?"), "run": man.get("run", "?"),
            "model_verdict": worst(r["verdict"] for r in rows), "first_difference": first_diff, "rows": rows}


def _guard(fn):
    try:
        st = fn()
    except (OSError, ValueError, KeyError) as ex:  # a missing file or a broken manifest entry: the row did not run
        return {"error": f"{type(ex).__name__}: {ex}", "verdict": ERROR, "reasons": [f"{type(ex).__name__}"]}
    if "error" in st:
        st.setdefault("verdict", ERROR)
        st.setdefault("reasons", [st["error"]])
    return st


def _row(kind, name, st, note=None):
    return {"kind": kind, "name": name, "verdict": st.get("verdict", ERROR), "size": st.get("size"),
            "reasons": st.get("reasons", []), "note": note, "numbers": st}


# ---------------------------------------------------------------------------------------------------- printing

def _pct(k, n):
    return f"{100.0 * k / n:.4f} %" if n else "-"


def _stage_lines(st, indent="    "):
    if "error" in st:
        return [indent + "not compared: " + st["error"]]
    n = st["values"]
    h = st["histogram"]
    L = [indent + f"values {n}, equal {_pct(st['equal'], n)}; steps " +
         " ".join(f"{k}:{h[k]}" for k in BUCKETS),
         indent + f"non-finite: either side {_pct(st['nonfinite_either'], n)}, one side only "
                  f"{st['nonfinite_mismatch']}; over the band ({st.get('band', 2)} steps) "
                  f"{_pct(st.get('over_band', st['over_2']), n)}, over 16 steps "
                  f"{_pct(st['over_16'], n)}"]
    if st["equal"] < n:
        mx = "non-finite on one side" if st["max_step"] is None else f"{st['max_step']} step(s)"
        nd = st["positive"] + st["negative"]
        L.append(indent + f"largest difference {mx} (original's {st['max_step_range']}); mean signed "
                          f"{st['mean_signed']:+.6f} steps; of the differences {_pct(st['positive'], nd)} positive, "
                          f"{_pct(st['negative'], nd)} negative")
        pc = st["mean_signed_per_channel"]
        if len(pc) > 1:
            top = sorted(range(len(pc)), key=lambda c: -abs(pc[c]))[:4]
            L.append(indent + "mean signed per channel, largest: " +
                     ", ".join(f"channel {c} {pc[c]:+.6f}" for c in top))
        nd_all = n - st["equal"]
        for fam, rows in st["classes"].items():
            ranked = sorted(rows, key=lambda r: -(r["differences"] / nd_all) / (r["values"] / n))[:3]
            L.append(indent + f"{fam}: " + "; ".join(
                f"{r['class']} {_pct(r['differences'], nd_all)} of differences / {_pct(r['values'], n)} of values"
                for r in ranked))
        fd = st["first_difference"]
        L.append(indent + f"first difference: frame {fd['frame']}, row {fd['row']}, column {fd['column']}, "
                          f"channel {fd['channel']}")
    return L


def _tail(r):
    return r["verdict"] + (f" ({r['size']})" if r["size"] else "") + (f"  [{r['note']}]" if r["note"] else "")


def _ranges(idx):
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ", ".join(out)


def _table_rows(rows):
    """Verdict table lines; the channel rows of one parent are grouped by verdict (every channel is listed)."""
    out, i = [], 0
    while i < len(rows):
        r = rows[i]
        if r["kind"] != "channel":
            out.append((r["kind"], r["name"], _tail(r)))
            i += 1
            continue
        j = i
        while j < len(rows) and rows[j]["kind"] == "channel" and rows[j]["parent"] == r["parent"]:
            j += 1
        groups = {}
        for x in rows[i:j]:
            groups.setdefault(_tail(x), []).append(x["channel"])
        for tail, chs in sorted(groups.items(), key=lambda g: g[1][0]):
            lab = "ch" if len(chs) == 1 else "channels"
            out.append(("channel", f"{r['parent']} / {lab} {_ranges(chs)}", tail))
        i = j
    return out


def format_report(rep):
    L = [f"EQUIVALENCE REPORT  {rep['model']}  ({rep['run']} run)  numbers only, no values of either side", "",
         "VERDICTS", f"  {'model':<14}{'':<40}{rep['model_verdict']}"]
    for kind, name, tail in _table_rows(rep["rows"]):
        L.append(f"  {kind:<14}{name[:39]:<40}{tail}")
    fd = rep.get("first_difference")
    if fd:
        L += ["", f"FIRST DIFFERENCE (compute order): stage {fd['stage']}, frame {fd['frame']}, row {fd['row']}, "
                  f"column {fd['column']}, channel {fd['channel']}"]
    L += ["", "NUMBERS"]
    for r in rep["rows"]:
        st = r["numbers"]
        if r["kind"] == "channel":
            quiet = r["verdict"] == NUMBERS_ONLY and st.get("equal") == st.get("values")
            if r["verdict"] != MATCHES and not quiet:
                L.append(f"  channel {r['name']}: {r['verdict']}" + (f" ({r['size']})" if r["size"] else "") +
                         ("; reasons: " + "; ".join(r["reasons"]) if r["reasons"] else ""))
                if "error" in st:
                    L.append("    not compared: " + st["error"])
                elif "values" in st:
                    L += _stage_lines(st)
                else:
                    L.append(f"    frames {st['frames']}, worst frame {st['worst_frame']}")
            continue
        L.append(f"  {r['kind']} {r['name']}: {r['verdict']}" + (f" ({r['size']})" if r["size"] else ""))
        if r["reasons"]:
            L.append("    reasons: " + "; ".join(r["reasons"]))
        if "error" in st:
            L.append("    not compared: " + st["error"])
        elif r["kind"] == "stage":
            L += _stage_lines(st)
        elif r["kind"] in ("temporal", "probe"):
            L.append(f"    frames {st['frames']}, worst frame {st['worst_frame']}")
            if st["first_difference"]:
                fd = st["first_difference"]
                L.append(f"    first difference: frame {fd['frame']}, row {fd['row']}, column {fd['column']}, "
                         f"channel {fd['channel']}")
            t = st.get("trend")
            if t:
                if t["evaluated"]:
                    ratio = "-" if t["ratio"] is None else f"{t['ratio']:.4f}"
                    L.append(f"    trend: mean |difference| frames 11-110 {t['early_mean_abs']:.6f}, last 100 "
                             f"{t['late_mean_abs']:.6f} steps, ratio {ratio}" + (" (grows)" if t["grows"] else ""))
                else:
                    L.append(f"    trend: not evaluated ({t['why']})")
            for fr, fs in sorted(st["numbers_at_frames"].items()):
                L.append(f"    frame {fr}: {fs['verdict']}")
                L += _stage_lines(fs, "      ")
            L.append("    per frame (share equal %, mean |difference| steps): " + " ".join(
                f"{100 * f['share_equal']:.3f}/{f['mean_abs']:.4f}" for f in st["per_frame"]))
        elif r["kind"] == "perturbation":
            c = st.get("correlation")
            ra = st.get("ratio")
            L.append(f"    correlation {'-' if c is None else f'{c:.5f}'}, ratio "
                     f"{'-' if ra is None else f'{ra:.5f}'}, 99 % radius ours {st.get('radius_ours')} / original "
                     f"{st.get('radius_orig')} px, opposite signs {_pct(st.get('opposite', 0), st['values'])}")
        elif r["kind"] == "sequence":
            L.append(f"    calls ours {st['calls_ours']}, original {st['calls_orig']}, differing {st['differing_calls']}")
            if st["first_difference"]:
                L.append(f"    first differing call: index {st['first_difference']['call']}, field "
                         f"{st['first_difference']['field']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--json", help="also write the report (numbers only) as JSON")
    a = ap.parse_args(argv)
    with open(a.manifest, encoding="utf-8") as f:
        man = json.load(f)
    rep = run_manifest(man, a.ours, a.orig)
    sys.stdout.write(format_report(rep))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1, default=lambda o: None)
    return {MATCHES: 0, CLOSE: 0, NUMBERS_ONLY: 0}.get(rep["model_verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
