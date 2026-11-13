# SPDX-License-Identifier: MIT
"""Stages of spec 51 (51.2, 51.4, 51.5, 51.6): IN0 (block and hidden state; the front end of page 52 is an input),
E1 (adapter 128 -> 64) .. E5, decoder fusion halves D4-F .. D0-F and recurrent halves D4-R .. D1-R.

Tensors are [Y][X][C]: e4m3 as uint8 codes, f16 as float64 values. Hidden states are f16 [Y][X][4]."""
import numpy as np

from .block import Grid, reflect, window_block
from .fmt import add16, d8, fma16, mm, mul16, q8, round_f16
from .scalar import sig16, sig_near

# name: (level, C, H, kind, extra)  extra = Cn for encoder merges, Kc for fusion halves
STAGES = {
    "IN0": (0, 32, 2, "in0", None),
    "E1": (1, 64, 2, "enc", 96),
    "E2": (2, 96, 4, "enc", 128),
    "E3": (3, 128, 4, "enc", 160),
    "E4": (4, 160, 8, "enc", 160),
    "E5": (5, 160, 8, "enc", None),
    "D4-F": (4, 160, 8, "fuse", 160),
    "D3-F": (3, 128, 4, "fuse", 160),
    "D2-F": (2, 96, 4, "fuse", 128),
    "D1-F": (1, 64, 2, "fuse", 96),
    "D0-F": (0, 32, 2, "fuse", 64),
    "D4-R": (4, 160, 8, "rec", None),
    "D3-R": (3, 128, 4, "rec", None),
    "D2-R": (2, 96, 4, "rec", None),
    "D1-R": (1, 64, 2, "rec", None),
}


def stage_info(name):
    level, C, H, kind, extra = STAGES[name]
    return dict(level=level, C=C, H=H, kind=kind,
                Cn=extra if kind in ("enc",) else None,
                Kc=extra if kind == "fuse" else None,
                adapter=(name == "E1"),
                hidden=("enc" if kind in ("in0", "enc") else "rec" if kind == "rec" else None),
                merged_hidden=(name == "IN0" or (kind == "enc" and extra is not None)),
                n_hidden_proj=(3 if kind in ("in0", "enc") else 4 if kind == "rec" else 0))


def weight_shapes(name):
    """Logical weight tensors of a stage (51.2): name -> (shape, type)."""
    i = stage_info(name)
    C, H = i["C"], i["H"]
    s = {
        "Wq": ((H, C, 32), "e4m3"), "Wv": ((H, C, 32), "e4m3"), "Tb": ((H, 12, 16, 16), "f16"),
        "Wo": ((H, 32, C), "e4m3"), "bo": ((C,), "f16"),
        "W1": ((C, 4 * C), "e4m3"), "b1": ((4 * C,), "f16"), "W2": ((4 * C, C), "e4m3"), "b2": ((C,), "f16"),
    }
    if i["Cn"]:
        s["Wm"] = ((4 * C, i["Cn"]), "e4m3")
        s["bm"] = ((i["Cn"],), "f16")
    if i["adapter"]:
        s["Wa"] = ((128, 64), "e4m3")
        s["ba"] = ((64,), "f16")
    if i["Kc"]:
        s["Wup"] = ((i["Kc"], 4 * C), "e4m3")
        s["bup"] = ((4 * C,), "f16")
    if i["n_hidden_proj"]:
        s["Wh"] = ((C, i["n_hidden_proj"]), "e4m3")
        s["bh"] = ((i["n_hidden_proj"],), "f16")
    return s


def stage_consts(name):
    """Per-stage constants used by a stage call (51.2): ra, m always; f for fusion halves; g for stages with a
    hidden-state update (a decoder level's g is used by its recurrent half)."""
    i = stage_info(name)
    c = ["ra", "m"]
    if i["kind"] == "fuse":
        c.append("f")
    if i["hidden"]:
        c.append("g")
    return c


def effective_shift(name, row):
    """51.6: single stages and fusion halves use the SH row as stored; recurrent halves ((s + 4) mod 8)."""
    sx, sy = row
    if STAGES[name][3] == "rec":
        return ((sx + 4) % 8, (sy + 4) % 8)
    return (sx, sy)


# ---------------------------------------------------------------------------------------------------------------
# tails

def merge_windows(grid, y2w, Wm, bm):
    """51.4 merge on the window's tokens: returns coarse tensor codes [Ny/2][Nx/2][Cn]."""
    assert grid.Nx % 2 == 0 and grid.Ny % 2 == 0
    Nw = y2w.shape[0]
    C = y2w.shape[-1]
    t = y2w.reshape(Nw, 8, 8, C)                                  # [w, ly, lx, c]
    kids = [t[:, dy::2, dx::2, :] for dy in (0, 1) for dx in (0, 1)]   # child n = 2*dy + dx, each [w, 4, 4, C]
    z = np.concatenate(kids, axis=-1)                             # z[n*C + c]
    merged = q8(mm(d8(z), d8(Wm), bm))                             # [w, 4, 4, Cn]
    Mx, My = grid.Nx // 2, grid.Ny // 2
    out = np.zeros((My, Mx, merged.shape[-1]), dtype=np.uint8)
    cnt = np.zeros((My, Mx), dtype=np.int64)
    for w in range(Nw):
        jw, iw = divmod(w, grid.nwx)
        x0 = (8 * iw - grid.sx) // 2
        y0 = (8 * jw - grid.sy) // 2
        for ry in range(4):
            for rx in range(4):
                Xc, Yc = x0 + rx, y0 + ry
                if 0 <= Xc < Mx and 0 <= Yc < My:
                    out[Yc, Xc] = merged[w, ry, rx]
                    cnt[Yc, Xc] += 1
    assert np.all(cnt == 1)
    return out


def hidden_projection(y2, Wh, bh):
    """Dh[t][j] = MM(d8(y2[t][.]); Wh[.][j]; bh[j])."""
    return mm(d8(y2), d8(Wh), bh)


def hidden_encoder(y2, hold, Wh, bh, g):
    """51.4 encoder rule (IN0, E1-E5). Returns (hnew [Y][X][4], Dh)."""
    Dh = hidden_projection(y2, Wh, bh)                             # [Y, X, 3]
    G = mul16(sig16(Dh), g)
    hnew = np.zeros(hold.shape, dtype=np.float64)
    hnew[..., :3] = mul16(hold[..., :3], G)
    hnew[..., 3] = 0.0
    return hnew, Dh


def merge_hidden(hnew):
    """51.4 merged hidden: per coarse token add16(add16(q0, q1), add16(q2, q3)), q_n = R(h_n / 4)."""
    Ny, Nx = hnew.shape[:2]
    assert Nx % 2 == 0 and Ny % 2 == 0
    q = mul16(hnew, 0.25)
    q0, q1 = q[0::2, 0::2], q[0::2, 1::2]
    q2, q3 = q[1::2, 0::2], q[1::2, 1::2]
    return add16(add16(q0, q1), add16(q2, q3))


def bilinear_coarse(Hc, Nx, Ny):
    """u_j of 51.4 (recurrent halves) for every fine token (X, Y) of an Nx x Ny tensor, j = 0..2.
    Returns u [Ny][Nx][3] (f16)."""
    My, Mx = Hc.shape[:2]
    X = np.arange(Nx)
    Y = np.arange(Ny)
    xm = np.where(X % 2 == 0, X // 2 - 1, (X - 1) // 2)           # floor((X - 1/2) / 2)
    ym = np.where(Y % 2 == 0, Y // 2 - 1, (Y - 1) // 2)
    xp, yp = xm + 1, ym + 1
    wxp = np.where(X % 2 == 0, 0.75, 0.25)
    wyp = np.where(Y % 2 == 0, 0.75, 0.25)
    wxm, wym = 1.0 - wxp, 1.0 - wyp
    rxm, rxp = reflect(xm, Mx), reflect(xp, Mx)
    rym, ryp = reflect(ym, My), reflect(yp, My)
    H3 = Hc[..., :3]
    A = H3[rym[:, None], rxp[None, :]]                             # (Y-, X+)
    B = H3[rym[:, None], rxm[None, :]]                             # (Y-, X-)
    Cc = H3[ryp[:, None], rxm[None, :]]                            # (Y+, X-)
    Dc = H3[ryp[:, None], rxp[None, :]]                            # (Y+, X+)
    WXP = wxp[None, :, None]
    WXM = wxm[None, :, None]
    WYP = wyp[:, None, None]
    WYM = wym[:, None, None]
    t = mul16(mul16(A, WXP), WYM)
    t = fma16(mul16(B, WXM), WYM, t)
    t = fma16(mul16(Cc, WXM), WYP, t)
    t = fma16(mul16(Dc, WXP), WYP, t)
    return t


def hidden_recurrent(y2, hold, Hc, Wh, bh, g):
    """51.4 recurrent-half rule (D4-R .. D1-R). Returns (hnew [Y][X][4], Dh, u)."""
    Ny, Nx = y2.shape[:2]
    Dh = hidden_projection(y2, Wh, bh)                             # [Y, X, 4]
    sg = sig16(Dh)
    z = sg[..., 3]
    omz = add16(1.0, -z)
    u = bilinear_coarse(Hc, Nx, Ny)
    gate = mul16(sg[..., :3], g)
    hh = mul16(mul16(u, gate), z[..., None])
    hnew = np.empty(hold.shape, dtype=np.float64)
    hnew[..., :3] = fma16(hold[..., :3], omz[..., None], hh)
    hnew[..., 3] = mul16(hold[..., 3], omz)
    return hnew, Dh, u


# ---------------------------------------------------------------------------------------------------------------
# stage call

def run_stage(name, inputs, W, consts, wc, shift_row, Z_override=None, trace=False):
    """One stage call.

    inputs:  IN0: x [N0y][N0x][32] codes, hold [N0y][N0x][4] f16
             E1: in [Ny][Nx][128] codes (level-1 tokens, the four level-0 children as stored), hold
             E2..E5: x [Ny][Nx][C], hold
             Dk-F: cin [My][Mx][Kc] codes (level k+1), skip [Ny][Nx][C] codes
             Dk-R: x [Ny][Nx][C] codes (the fusion half's y2), hold [Ny][Nx][4], Hc [My][Mx][4]
    consts:  dict of f16 per-stage constants (ra, m, f, g as used, see stage_consts)
    shift_row: (sx, sy) of the stage's SH row; recurrent halves apply (s + 4) mod 8 (51.6).
    Z_override: optional f16 [nw][H][64] row sums (window order w = j*nwx + i).
    Returns (outputs dict, aux dict)."""
    info = stage_info(name)
    C = info["C"]
    sx, sy = effective_shift(name, shift_row)
    kind = info["kind"]
    if kind == "fuse":
        skip = inputs["skip"]
        Ny, Nx = skip.shape[:2]
    elif info["adapter"]:
        Ny, Nx = inputs["in"].shape[:2]
    else:
        Ny, Nx = inputs["x"].shape[:2]
    grid = Grid(Nx, Ny, sx, sy)
    aux = {"shift": (sx, sy), "grid": (grid.nwx, grid.nwy)}

    # ---- front (51.3 input, 51.5)
    if kind == "fuse":
        cin = inputs["cin"]
        My, Mx = cin.shape[:2]
        assert (2 * Mx, 2 * My) == (Nx, Ny), "fusion half: fine extent must be twice the coarse extent"
        upq = q8(mm(d8(cin), d8(W["Wup"]), W["bup"]))              # [My, Mx, 4C], quantised before use
        skw = grid.gather(skip)                                    # skip read at the fine level
        Xc = np.floor_divide(grid.X, 2)
        Yc = np.floor_divide(grid.Y, 2)
        n = 2 * np.mod(grid.Y, 2) + np.mod(grid.X, 2)              # child index inside the coarse parent
        upw = upq[reflect(Yc, My), reflect(Xc, Mx)]                # [nw, 64, 4C], coarse read reflected with M
        sel = np.take_along_axis(upw.reshape(grid.nw, 64, 4, C), n[..., None, None], axis=2)[:, :, 0, :]
        xf = fma16(d8(skw), consts["f"], d8(sel))
        x = q8(xf)
        xr = xf
        if trace:
            aux["xf"] = xf
            aux["upq"] = upq
    elif info["adapter"]:
        inw = grid.gather(inputs["in"])
        x = q8(mm(d8(inw), d8(W["Wa"]), W["ba"]))
        xr = d8(x)
    else:
        x = grid.gather(inputs["x"])
        xr = d8(x)

    # ---- core (51.3)
    y2w, binfo = window_block(x, xr, W, consts["ra"], consts["m"], wc, Z_override=Z_override, trace=trace)
    aux["Z"] = binfo["Z"]
    if trace:
        aux["block"] = binfo
        aux["x_in"] = x
    y2 = grid.scatter(y2w, 0)
    out = {"y2": y2}

    # ---- tails (51.4)
    if info["Cn"]:
        out["merged"] = merge_windows(grid, y2w, W["Wm"], W["bm"])
    if info["hidden"] == "enc":
        hnew, Dh = hidden_encoder(y2, inputs["hold"], W["Wh"], W["bh"], consts["g"])
        out["hnew"] = hnew
        aux["Dh"] = Dh
        aux["hnew_sig_near"] = np.any(sig_near(Dh), axis=-1)
        if info["merged_hidden"]:
            out["hmerged"] = merge_hidden(hnew)
            nr = aux["hnew_sig_near"]
            aux["hmerged_sig_near"] = nr[0::2, 0::2] | nr[0::2, 1::2] | nr[1::2, 0::2] | nr[1::2, 1::2]
    elif info["hidden"] == "rec":
        Hc = inputs["Hc"]
        assert (2 * Hc.shape[1], 2 * Hc.shape[0]) == (Nx, Ny)
        hnew, Dh, u = hidden_recurrent(y2, inputs["hold"], Hc, W["Wh"], W["bh"], consts["g"])
        out["hnew"] = hnew
        aux["Dh"] = Dh
        aux["u"] = u
        aux["hnew_sig_near"] = np.any(sig_near(Dh), axis=-1)
    return out, aux
