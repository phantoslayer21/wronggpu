# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): constructed inputs for the light path's tests and GPU check. Every value is
synthetic: a gradient-and-noise background panning with the camera, a nearer block moving on its own, a static UI
panel (or a full-screen menu) blended over the hudless colour, a depth plane with the block in front, and the two
clip-space matrices of a pure camera translation (with a small depth-dependent w)."""
import numpy as np

from clean.fg.reference import pipeline as P
from clean.fg.reference import stages_common as S


def matrices(tx, ty, wz=0.0625):
    """ClipToPrevClip and PrevClipToClip of a translation by (tx, ty) in NDC, w = 1 + wz z (M[i] as supplied)."""
    a = np.zeros(16, np.float32)
    a[0] = a[5] = a[15] = 1.0
    a[11] = wz
    b = a.copy()
    a[12], a[13] = tx, ty
    b[12], b[13] = -tx, -ty
    return a, b


def make_call(wo, ho, frame, n=1, i=1, hudless=True, ui="none", menu=False, inverted=False, seed=7, pad=0,
              out_offset=(0, 0), out_size=None, reset=False, mvec_scale=(1.0, 1.0), specials=False):
    """CallInputs of game frame `frame` (index i of n). ui: "none", "ui4", "ui1" or "both"; menu: a full-screen
    overlay of alpha 0.9 over 85 % of the frame. pad: texels of extra texture border around every input region (base
    offset = pad). specials: a few colour texels at +-inf / large values (finite-check paths)."""
    rng = np.random.default_rng(seed * 1000 + frame)
    tw, th = wo + 2 * pad, ho + 2 * pad
    yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
    pan = 6.0 * frame
    base = np.stack([0.5 + 0.4 * np.sin((xx + pan) * 0.11) * np.cos(yy * 0.07),
                     0.5 + 0.3 * np.cos((xx + pan) * 0.05 + yy * 0.09),
                     0.4 + 0.3 * np.sin((yy - pan) * 0.13)], -1)
    base += rng.uniform(-0.05, 0.05, base.shape)
    bx0, by0 = int(tw * 0.3 + 2 * frame), int(th * 0.35 + frame)
    bw, bh = max(4, tw // 5), max(4, th // 4)
    block = (xx >= bx0) & (xx < bx0 + bw) & (yy >= by0) & (yy < by0 + bh)
    base[block] = np.array([0.9, 0.2, 0.15]) + rng.uniform(-0.03, 0.03, (int(block.sum()), 3))
    hud = np.concatenate([base, np.ones((th, tw, 1))], -1).astype(np.float32)
    alpha = np.zeros((th, tw), np.float32)
    uicol = np.zeros((th, tw, 3), np.float32)
    if ui != "none":
        px0, py0 = int(tw * 0.62), int(th * 0.08)
        panel = (xx >= px0) & (xx < px0 + tw // 4) & (yy >= py0) & (yy < py0 + th // 5)
        alpha[panel] = 0.75
        uicol[panel] = (0.1, 0.8, 0.9)
        txt = panel & (((xx.astype(int) // 3) + (yy.astype(int) // 2)) % 3 == 0)
        alpha[txt] = 1.0
        uicol[txt] = (1.0, 1.0, 1.0)
    if menu:
        m = (xx >= tw * 0.05) & (xx < tw * 0.95) & (yy >= th * 0.03) & (yy < th * 0.97)
        alpha[m] = 0.9
        uicol[m] = (0.05, 0.05, 0.2)
    colour = hud.copy()
    colour[..., :3] = uicol * alpha[..., None] + hud[..., :3] * (1 - alpha[..., None])
    if specials:
        k = rng.choice(th * tw, 6, replace=False)
        colour.reshape(-1, 4)[k[:2], 0] = np.inf
        colour.reshape(-1, 4)[k[2:4], 1] = 3.0e4
        colour.reshape(-1, 4)[k[4:], 2] = -np.inf
    # depth: plane (D = 1 - z in 0.3 .. 0.6) with the block nearer (D 0.012 .. 0.02)
    dplane = 0.3 + 0.3 * (yy / th)
    D = np.where(block, 0.012 + 0.008 * (xx / tw), dplane)
    z = D if inverted else 1.0 - D
    depth = z.astype(np.float32)[..., None]
    # motion (current -> previous), normalised units of the motion texture extent: camera pan plus the block's own
    tx = 12.0 / wo                 # NDC shift of the pan: 6 frame pixels per game frame (tx / 2 in normalised units)
    cam = np.array([tx * 0.5, 0.0], np.float32)
    mot = np.broadcast_to(cam, (th, tw, 2)).copy()
    mot[block] = (-4.0 / wo, -2.0 / ho)
    mot += rng.uniform(-0.2, 0.2, mot.shape).astype(np.float32) / np.array([wo, ho], np.float32)
    mot *= np.array([wo, ho], np.float32)          # supplied in pixels of the motion extent (motion scale 1)
    a, b = matrices(tx, 0.0)
    reg = S.Region(tw, th, pad, pad, wo, ho)
    kw = {}
    if hudless:
        kw.update(hudless=hud.astype(np.float16), hreg=reg)
    if ui in ("ui4", "both"):
        kw.update(ui4=np.concatenate([uicol * alpha[..., None], alpha[..., None]], -1).astype(np.float16), u4reg=reg)
    if ui in ("ui1", "both"):
        kw.update(ui1=alpha[..., None].astype(np.float16), u1reg=reg)
    return P.CallInputs(colour.astype(np.float16), reg, depth, reg, mot.astype(np.float32), reg, a, b, n=n, i=i,
                        depth_inverted=inverted, mvec_scale=mvec_scale, reset=reset, out_size=out_size,
                        out_offset=out_offset, **kw)


def synthetic_net1_params(seed=11):
    """Synthetic net1 weights (scaled so values stay in range), as the net1 GPU tools use."""
    from clean.fg.layout import NET1
    rng = np.random.default_rng(seed)
    out = {}
    for l in NET1:
        k = 3 if l.taps == 9 else 1
        s = 1.5 / np.sqrt(l.cin * k * k)
        out[l.name] = (rng.uniform(-s, s, (l.out, k, k, l.cin)).astype(np.float16),
                       rng.uniform(-0.1, 0.1, l.out).astype(np.float16))
    return out
