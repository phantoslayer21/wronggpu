# SPDX-License-Identifier: MIT
"""Spec 55: the recurrent half of decoder D0 (D0-R) and the network's output head, vectorised NumPy reference.

Entry points:
* `derive_scalars(h0, h1, s1cap)`: the converter rule of 55.3 for s1, s2 (f32).
* `hidden_state(y2h, hold, hc, g)`: 55.2 / 51.4, the level-0 hidden state Cnet [Ht][Wt][4] (f16).
* `output_images(y2h, s1, s2, wr, hr)`: 55.3, the four images at the render size (f16 values and stored codes).
* `d0r_stage(...)`: block (page 51 at level 0) + both of the above.

Tensors are [Y][X][channel]; f16 values are float64 arrays holding binary16 values (see fp.py).
"""
import numpy as np

from . import block as B
from . import fp

f64 = np.float64
f32 = np.float32

# Block-output channels of D0-R used by the head (55.2, 55.3). Page 55 wrote every explicit channel number in an internal
# ordering; spec erratum page 55-rr-f-erratum-channel-numbers ("page 55 erratum") gives the natural (converter) numbers:
# internal c -> natural 16 ((c mod 4) div 2) + 8 (c div 16) + 2 ((c mod 16) div 4) + (c mod 2), applied as one
# simultaneous renaming. The erratum numbering is the default (P55_ERRATUM); the page's numbering stays available
# (erratum=False / head_channels(False)) for comparison only. Measured first on the gate logits (55.2: page 0, 1, 4, 5
# -> 0, 1, 2, 3; results/2026-10-21-rr-bluefix.md, QUESTIONS-BLUEFIX.md).
P55_ERRATUM = True
PAGE55_CH = {
    "gate": (0, 1, 4, 5),          # 55.2 Dh_0..3 (Dh_3 = z)
    "hid": (28, 2),                # 55.3 HID values 0, 1 = tanh(y_a); values 2, 3 = tanh(y_b)
    "gates": (8, 9, 12),           # 55.3 GATES values 0, 1, 2
    "prec": (13, 16, 17),          # 55.3 PREC: Px logit, Py logit, rho logit
    "candw": (20, 24),             # 55.3 CANDW: d = f32(y_a - y_b)
}


def p55_natural(c):
    """Page-55 (internal) block-output channel c -> natural (converter) channel (the erratum's bijection)."""
    return 16 * ((c % 4) // 2) + 8 * (c // 16) + 2 * ((c % 16) // 4) + (c % 2)


ERRATUM55_CH = {k: tuple(p55_natural(c) for c in v) for k, v in PAGE55_CH.items()}

# Functional erratum 55-ERRATUM-2 (spec side, confirmed against the original; clean/rr-f-flow): page 55.3's HID and
# CANDW pairs were packed pairs. HID = R(tanh y14), R(tanh y15), R(tanh y16), R(tanh y17); CANDW W4.r..a from the four
# logits y10, y11, y12, y13 (pair1 (MV1) <- y10 .. pair4 (MV4) <- y13) by the binary16 chain of candw4_values. The
# head then reads natural channels 0 .. 17. Default; erratum2=False keeps page 55's paired reading for comparison.
P55_ERRATUM2 = True
ERRATUM2_CH = dict(ERRATUM55_CH, hid=(14, 15, 16, 17), candw=(10, 11, 12, 13))
LOG2E16 = 1.4423828125            # R(log2 e) in binary16 (bits 0x3dc5)


def head_channels(erratum=None, erratum2=None):
    """The head's channel picks: 55-ERRATUM-2 (default), the first erratum's natural numbers (erratum2=False) or the
    page's numbers (erratum=False)."""
    if not (P55_ERRATUM if erratum is None else erratum):
        return PAGE55_CH
    return ERRATUM2_CH if (P55_ERRATUM2 if erratum2 is None else erratum2) else ERRATUM55_CH


GATE_LOGIT_CH = ERRATUM55_CH["gate"]          # (0, 1, 2, 3)
SPEC_55_2_GATE_CH = PAGE55_CH["gate"]         # (0, 1, 4, 5), comparison only
UNUSED_CH = tuple(c for c in range(32) if c not in {x for v in ERRATUM2_CH.values() for x in v})   # 18 .. 31


# ------------------------------------------------------------------------------------------------ converter
def derive_scalars(h0, h1, s1cap):
    """s1 = min(expf(h0), S1CAP), s2 = expf(h1), expf correctly rounded; h0, h1 f16 values, S1CAP f32.
    Raises ValueError when s2 is not finite (the converter refuses the model)."""
    e0 = fp.expf32(np.array([h0], f32))[0]
    s1 = f32(min(float(e0), float(f32(s1cap))))       # e0 is never NaN (h0 finite); e0 = +inf gives S1CAP
    s2 = fp.expf32(np.array([h1], f32))[0]
    if not np.isfinite(s2):
        raise ValueError("s2 is not finite: the converter refuses the model (55.3)")
    return s1, f32(s2)


# ------------------------------------------------------------------------------------------------ 55.2
def upsample_coarse(hc, ht, wt):
    """Bilinear coarse-hidden values u [Ht][Wt][C'] of page 51.4 (corner order A, B, Cc, Dc, rounding as stated),
    coarse reads reflected with the coarse extent (51.6)."""
    my, mx = hc.shape[:2]
    X = np.arange(wt)
    Y = np.arange(ht)
    xm, ym = (X - 1) // 2, (Y - 1) // 2               # floor((X - 1/2)/2)
    xp, yp = xm + 1, ym + 1
    wxp = np.where(X % 2 == 0, 0.75, 0.25)[None, :, None]
    wyp = np.where(Y % 2 == 0, 0.75, 0.25)[:, None, None]
    wxm, wym = 1.0 - wxp, 1.0 - wyp
    rxm, rxp = B.reflect(xm, mx), B.reflect(xp, mx)
    rym, ryp = B.reflect(ym, my), B.reflect(yp, my)
    A = hc[rym][:, rxp]       # (Y-, X+)
    Bv = hc[rym][:, rxm]      # (Y-, X-)
    Cc = hc[ryp][:, rxm]      # (Y+, X-)
    Dc = hc[ryp][:, rxp]      # (Y+, X+)
    t = fp.mul16(fp.mul16(A, wxp), wym)
    t = fp.fma16(fp.mul16(Bv, wxm), wym, t)
    t = fp.fma16(fp.mul16(Cc, wxm), wyp, t)
    return fp.fma16(fp.mul16(Dc, wxp), wyp, t)


def hidden_state(y2h, hold, hc, g, gate_ch=None, erratum=None):
    """Cnet [Ht][Wt][4] of 55.2: the decoder recurrent-half rule of 51.4 with Dh_j = y2h channel gate_ch[j]
    (default: head_channels(erratum)["gate"])."""
    ht, wt = y2h.shape[:2]
    if gate_ch is None:
        gate_ch = head_channels(erratum)["gate"]
    Dh = y2h[..., list(gate_ch)]
    sg = fp.sig16(Dh)
    z = sg[..., 3]
    omz = fp.add16(1.0, -z)
    u = upsample_coarse(hc, ht, wt)
    out = np.empty((ht, wt, 4), f64)
    for j in range(3):
        gate = fp.mul16(sg[..., j], g)
        hh = fp.mul16(fp.mul16(u[..., j], gate), z)
        out[..., j] = fp.fma16(hold[..., j], omz, hh)
    out[..., 3] = fp.mul16(hold[..., 3], omz)
    return out


# ------------------------------------------------------------------------------------------------ 55.3
def candw_values(y20, y24):
    """(w, w') of CANDW for f16 logits y20, y24: d = f32(y20 - y24), w = R(sigx(d)/2), w' = R(sigx(-d)/2)."""
    with np.errstate(all="ignore"):
        d = (np.asarray(y20, f64) - np.asarray(y24, f64)).astype(f32)     # exact difference, rounded once
    return fp.logistic_half16(d), fp.logistic_half16(-d)


def candw4_values(z0, z1, z2, z3):
    """The four CANDW values of 55-ERRATUM-2 for f16 logits (y10, y11, y12, y13), every step rounded to binary16:
    m = max of the four (a NaN operand is ignored unless all are NaN); d = R(z - m); t = R(d R(log2 e)); e = R(2^t)
    (correctly rounded); S = R(R(e0 + e2) + R(e1 + e3)); q = R(1/S) (correctly rounded); W = R(e q)."""
    z = [np.asarray(v, f64) for v in (z0, z1, z2, z3)]
    with np.errstate(all="ignore"):
        m = np.fmax(np.fmax(z[0], z[1]), np.fmax(z[2], z[3]))
        e = [fp.lookup(fp.exp2_16_table(), fp.r16(fp.r16(v - m) * LOG2E16)) for v in z]
        S = fp.r16(fp.r16(e[0] + e[2]) + fp.r16(e[1] + e[3]))
        q = fp.lookup(fp.rcp16_table(), S)
        return [fp.r16(v * q) for v in e]


def prec_values(y13, y16, y17, s1, s2):
    """The three PREC values of 55.3 (f16) and the f32 intermediates (Px, Py, rho)."""
    s1 = f32(s1)
    s2 = f32(s2)
    with np.errstate(all="ignore"):
        Px = fp.sig16(y13).astype(f32) * s1
        Py = fp.sig16(y16).astype(f32) * s1
        rho = fp.tanh32(np.asarray(y17, f64).astype(f32) * s2)
        p0 = fp.r16_from_f32(Px * Px)
        p1 = fp.r16_from_f32(Py * Py)
        p2 = fp.r16_from_f32((Px * Py) * rho)
    return (p0, p1, p2), (Px, Py, rho)


def output_images(y2h, s1, s2, wr, hr, erratum=None, erratum2=None):
    """The four images of 55.3 for the tokens with X < Wr, Y < Hr. Returns a dict of f16 values ([Hr][Wr][4],
    float64) and stored forms: HID codes int8, GATES / CANDW codes uint8, PREC f16 values. Channel picks:
    head_channels(erratum) (page numbers y28, y2, y8, y9, y12, y13, y16, y17, y20, y24 -> natural y14, y16, y4, y5,
    y6, y7, y8, y9, y10, y12)."""
    y = y2h[:hr, :wr]
    ch = head_channels(erratum, erratum2)
    c = lambda n: y[..., n]
    if len(ch["hid"]) == 4:      # 55-ERRATUM-2
        hid = np.stack([fp.tanh16(c(n)) for n in ch["hid"]], -1)
    else:
        ha, hb = ch["hid"]
        hid = np.stack([fp.tanh16(c(ha)), fp.tanh16(c(ha)), fp.tanh16(c(hb)), fp.tanh16(c(hb))], -1)
    zero = np.zeros(y.shape[:2], f64)
    gates = np.stack([fp.sig16(c(n)) for n in ch["gates"]] + [zero], -1)
    (p0, p1, p2), _ = prec_values(*(c(n) for n in ch["prec"]), s1, s2)
    prec = np.stack([p0, p1, p2, zero], -1)
    if len(ch["candw"]) == 4:    # 55-ERRATUM-2
        candw = np.stack(candw4_values(*(c(n) for n in ch["candw"])), -1)
    else:
        w, w2 = candw_values(*(c(n) for n in ch["candw"]))
        candw = np.stack([w, w, w2, w2], -1)
    return {
        "HID": hid, "HID_code": fp.snorm8_store(hid),
        "GATES": gates, "GATES_code": fp.unorm8_store(gates),
        "PREC": prec,
        "CANDW": candw, "CANDW_code": fp.unorm8_store(candw),
    }


# ------------------------------------------------------------------------------------------------ stage
def d0r_stage(x, W, k, ra, m, g, shift, hold, hc, s1, s2, wr, hr, diag=None, erratum=None):
    """D0-R: window block (page 51.3, level 0, shift = the recurrent-half shift) on the D0-F output x [Ht][Wt][32]
    (e4m3 codes), then Cnet (55.2) and the output images (55.3). hold [Ht][Wt][4] f16 (IN0's hnew),
    hc [Ht/2][Wt/2][4] f16 (D1-R's hnew)."""
    ht, wt = x.shape[:2]
    if not (0 < wr <= wt and 0 < hr <= ht):
        raise ValueError("render size must lie inside the tensor extent")
    y2h = B.run_block_stage(x, W, k, ra, m, shift, diag)
    if diag is not None:
        diag["y2h"] = y2h
    cnet = hidden_state(y2h, hold, hc, g, erratum=erratum)
    imgs = output_images(y2h, s1, s2, wr, hr, erratum=erratum)
    imgs["Cnet"] = cnet
    return imgs
