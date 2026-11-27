# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): CPU reference of the light candidate blend (preset C), written for this project.

One output pixel per frame pixel (x, y) < (Wo, Ho), u, v over Wo x Ho (approximate division):
- the vectors mA (MA, toward the current frame) and mB (MB, toward the previous frame) at index
  rint(fma(u, Wm, -0.5) - 2^-23) of the scatter grid, (0, 0) and empty flag 1 where nothing landed;
- P0 / Q0: the previous / current colour image at (x, y); cP / cQ: bilinear reads of them at uvP = (u, v) + mB and
  uvQ = (u, v) + mA; w: bilinear read of the weight image at (u, v);
- flag 2 (refined candidates): a candidate whose source sits on UI (its alpha after filtering) while both co-located
  pixels are UI-free (alpha exactly 0) is replaced, faded in by smoothstep(g), by the reference image at the source plus
  the pixel's own detail (P0 - reference at (u, v)) when that detail is small (max <= 0.15) and the motion is at least
  two pixels;
- a = fma(eB, cQ - cP, cP), b = fma(eA, cP - cQ, cQ), rP = fma(a - P0, w.x, P0), rQ = fma(b - Q0, w.y, Q0),
  out = fma(rQ - rP, t, rP);
- flag 1 (UI composite): U = fma(Uc - Up, t, Up), out.rgb = fma(out.rgb, 1 - U.a, U.rgb);
- written as (r, g, b, 1) at the output's base offset.
"""
import numpy as np

from clean.fg.reference import stages_common as S
from clean.fg.reference.stages_common import F32, adiv, finite, fma, fmax, sat

ALPHA_MID = F32(0.5)
ALPHA_SPAN = S.fbits(0x3EE66666)        # 0.45
DETAIL_MAX = S.fbits(0x3E19999A)        # 0.15


def _in_box(px, py, wo, ho):
    with np.errstate(invalid="ignore"):
        return (px >= F32(0.0)) & (px < F32(wo - 1)) & (py >= F32(0.0)) & (py < F32(ho - 1))


def _correct(c, alpha, uv, u, v, empty, P0, Q0, L, lreg, own, wo, ho):
    """The refined-path correction of one candidate c [h, w, 3]; own = P0 (for cP) or Q0 (for cQ). L: the reference
    image (Lp or Lc) read through lreg (a Region of its own size)."""
    with np.errstate(invalid="ignore", over="ignore"):
        ok = np.all(finite(c), -1) & ~empty & finite(uv[..., 0]) & finite(uv[..., 1]) & (wo >= 2) & (ho >= 2)
        ok &= _in_box(fma(uv[..., 0], F32(wo), F32(-0.5)), fma(uv[..., 1], F32(ho), F32(-0.5)), wo, ho)
        ok &= finite(u) & finite(v) & _in_box(fma(u, F32(wo), F32(-0.5)), fma(v, F32(ho), F32(-0.5)), wo, ho)
        s = adiv((alpha - ALPHA_MID).astype(np.float32), ALPHA_SPAN)
        s = np.where(s <= F32(0.0), F32(0.0), np.where(s >= F32(1.0), F32(1.0), s)).astype(np.float32)
        g = ((s * s).astype(np.float32) * (F32(3.0) - (s + s).astype(np.float32)).astype(np.float32)).astype(np.float32)
        ok &= g > F32(0.0)
        dx = ((uv[..., 0] - u).astype(np.float32) * F32(wo)).astype(np.float32)
        dy = ((uv[..., 1] - v).astype(np.float32) * F32(ho)).astype(np.float32)
        ok &= S.sq(dx, dy) >= F32(4.0)
        ok &= np.all(finite(P0[..., :3]), -1) & np.all(finite(Q0[..., :3]), -1)
        ok &= (P0[..., 3] == F32(0.0)) & (Q0[..., 3] == F32(0.0))
        X1, Y1 = lreg.coords(uv[..., 0], uv[..., 1])
        d1 = S.read_linear(L, X1, Y1)[..., :3]
        X0, Y0 = lreg.coords(u, v)
        d0 = S.read_linear(L, X0, Y0)[..., :3]
        dl = (own[..., :3] - d0).astype(np.float32)
        ok &= np.all(finite(d1), -1) & np.all(finite(d0), -1) & np.all(finite(dl), -1)
        ad = np.abs(dl)
        ok &= fmax(fmax(ad[..., 0], ad[..., 1]), ad[..., 2]) <= DETAIL_MAX
        corr = fma(g[..., None], (sat((d1 + dl).astype(np.float32)) - c).astype(np.float32), c)
    return np.where(ok[..., None], corr, c).astype(np.float32)


def lite_blend(Cp, Cc, MA, MB, Wimg, wm, hm, t, flag1, flag2, Up, Uc, Lp, Lc, out, ox=0, oy=0):
    """The light blend: writes out[oy + y, ox + x] (float16 [.., 4]) for the frame extent of Cp / Cc."""
    ho, wo = Cc.shape[:2]
    u, v = S.grid_uv(wo, ho)
    u, v = np.broadcast_to(u, (ho, wo)), np.broadcast_to(v, (ho, wo))
    mA, eA = S.vec_lookup(MA, wm, hm, u, v)
    mB, eB = S.vec_lookup(MB, wm, hm, u, v)
    P0, Q0 = Cp.astype(np.float32), Cc.astype(np.float32)
    t = F32(t)
    with np.errstate(invalid="ignore", over="ignore"):
        uvP = np.stack([(u + mB[..., 0]).astype(np.float32), (v + mB[..., 1]).astype(np.float32)], -1)
        uvQ = np.stack([(u + mA[..., 0]).astype(np.float32), (v + mA[..., 1]).astype(np.float32)], -1)
        sP = S.read_linear(Cp, uvP[..., 0], uvP[..., 1])
        sQ = S.read_linear(Cc, uvQ[..., 0], uvQ[..., 1])
        cP, cQ = sP[..., :3], sQ[..., :3]
        w = S.read_linear(Wimg, u, v)
        if flag2:
            lreg = S.Region(wo, ho)
            ncP = _correct(cP, sP[..., 3], uvP, u, v, eB, P0, Q0, Lp, lreg, P0, wo, ho)
            ncQ = _correct(cQ, sQ[..., 3], uvQ, u, v, eA, P0, Q0, Lc, lreg, Q0, wo, ho)
            cP, cQ = ncP, ncQ
        fB = np.where(eB, F32(1.0), F32(0.0))[..., None]
        fA = np.where(eA, F32(1.0), F32(0.0))[..., None]
        a = fma(fB, (cQ - cP).astype(np.float32), cP)
        b = fma(fA, (cP - cQ).astype(np.float32), cQ)
        rP = fma((a - P0[..., :3]).astype(np.float32), w[..., 0:1], P0[..., :3])
        rQ = fma((b - Q0[..., :3]).astype(np.float32), w[..., 1:2], Q0[..., :3])
        o = fma((rQ - rP).astype(np.float32), t, rP)
        if flag1:
            up, uc = Up.astype(np.float32), Uc.astype(np.float32)
            U = fma((uc - up).astype(np.float32), t, up)
            o = fma(o, (F32(1.0) - U[..., 3]).astype(np.float32)[..., None], U[..., :3])
    res = np.concatenate([o, np.ones((ho, wo, 1), np.float32)], -1)
    out[oy:oy + ho, ox:ox + wo] = S.h(res)
