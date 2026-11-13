# SPDX-License-Identifier: MIT
"""Synthetic test inputs for spec 51: window-block constant sets, per-stage constants, logical weights and stage
inputs. Converter-supplied constants and weights are test inputs (page 57 is not available); their ranges are our
own choice except where spec 51 states one (attention constants: p of every clamped score in [1+1/32, 1+31/32);
shifts even and below 8)."""
import numpy as np

from clean.rr_f.reference.fmt import f16, q8, round_f16
from clean.rr_f.reference.scalar import WindowConsts, check_attention_consts
from clean.rr_f.reference.stages import stage_info, weight_shapes

# e4m3 edge codes: +-0, smallest / largest subnormal, smallest normal, 1, large, +-448
EDGE_SMALL = np.array([0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x3F, 0xBF], dtype=np.uint8)
EDGE_LARGE = np.array([0x76, 0xF6, 0x7E, 0xFE], dtype=np.uint8)     # +-224, +-448
F16_EDGE = np.array([0.0, -0.0, 2.0 ** -24, -(2.0 ** -24), 2.0 ** -14, -(2.0 ** -14), 1.0, -1.0, 65504.0, -65504.0])


def f16s(x):
    return float(f16(x))


def window_consts(rng, flavour):
    """flavour 'A', 'B' (two plausible builds), 'zovf' (64 * w_max above the f16 range: rows can overflow)."""
    for _ in range(1000):
        if flavour == "zovf":
            hi = rng.uniform(0.56, 0.6)
            lo = -rng.uniform(0.56, 0.6)
            cl = max(hi, -lo)
            p1 = 3.0 * cl * cl * rng.uniform(1.0, 1.05)
            amp = cl * (p1 - cl * cl)
            p0 = 1.0 + (31.0 / 32.0 - 0.012) - amp    # top of the range nearly reached, wide spread below it
            scale = rng.uniform(0.01, 0.02)
        else:
            hi = rng.uniform(0.3, 0.8)
            lo = -rng.uniform(0.3, 0.8)
            cl = max(hi, -lo)
            p1 = 3.0 * cl * cl * rng.uniform(1.0, 1.3)
            amp = cl * abs(p1 - cl * cl)
            p0 = rng.uniform(1.0 + 1 / 32 + amp + 0.02, 1.0 + 31 / 32 - amp - 0.02) if amp < 0.44 else 1.5
            scale = rng.uniform(0.01, 0.08)
        act_cl = rng.uniform(2.0, 3.5)
        act_cln = -act_cl * rng.uniform(0.9, 1.1)
        c1 = 0.5 / act_cl ** 2 * rng.uniform(0.9, 1.1)
        c2 = 2.0 * c1 * act_cl * rng.uniform(0.95, 1.05)
        c3 = 0.5 * rng.uniform(0.98, 1.02)
        wc = WindowConsts(A_SCALE=f16s(scale), A_CLAMP=f16s(hi), A_CLAMPn=f16s(lo), A_P1=f16s(p1), A_P0=f16s(p0),
                          ACT_CL=f16s(act_cl), ACT_CLn=f16s(act_cln), ACT_C1=f16s(c1), ACT_C2=f16s(c2),
                          ACT_C3=f16s(c3))
        chk = check_attention_consts(wc)
        if not chk["ok"]:
            continue
        if flavour == "zovf" and not chk["z_may_overflow"]:
            continue
        if flavour != "zovf" and chk["z_may_overflow"]:
            continue
        return wc, chk
    raise RuntimeError("no valid window constants found")


def stage_consts(rng, hot=False):
    lo, hi = (0.25, 1.5) if not hot else (1.0, 6.0)
    return {
        "ra": f16s(rng.uniform(lo, hi)),
        "m": f16s(rng.uniform(lo, hi)),
        "f": f16s(rng.uniform(0.5, 1.5) * (1.5 if hot else 1.0)),
        "g": f16s(rng.uniform(0.5, 2.0)),
    }


def e4m3(rng, shape, sigma, edge_small=0.01, edge_large=0.0):
    c = q8(rng.normal(0.0, sigma, size=shape))
    if edge_small:
        msk = rng.random(shape) < edge_small
        c = np.where(msk, rng.choice(EDGE_SMALL, size=shape), c)
    if edge_large:
        msk = rng.random(shape) < edge_large
        c = np.where(msk, rng.choice(EDGE_LARGE, size=shape), c)
    return c.astype(np.uint8)


def f16_arr(rng, shape, sigma, edge=0.0, edges=F16_EDGE):
    v = round_f16(rng.normal(0.0, sigma, size=shape))
    if edge:
        msk = rng.random(shape) < edge
        v = np.where(msk, rng.choice(edges, size=shape), v)
    return v


def stage_weights(rng, name, hot=False):
    """Random logical weights of a stage (51.2 table). hot: larger weights (saturation, f16 overflow)."""
    info = stage_info(name)
    C = info["C"]
    k = 8.0 if hot else 1.0
    el = 0.002 if hot else 0.0
    W = {}
    for tname, (shape, typ) in weight_shapes(name).items():
        if typ == "e4m3":
            fan_in = shape[-2]
            kk = 1.0 if tname in ("Wq", "Wv", "Wa", "Wup") else k   # hot: large output / MLP weights only
            W[tname] = e4m3(rng, shape, kk / np.sqrt(fan_in), edge_small=0.005,
                            edge_large=(0.0 if tname in ("Wq", "Wv", "Wa", "Wup") else el))
        elif tname == "Tb":
            W[tname] = f16_arr(rng, shape, 2.0, edge=0.002)
        elif tname == "bh":
            W[tname] = f16_arr(rng, shape, 2.0 * k)
        else:
            W[tname] = f16_arr(rng, shape, 0.1 * k, edge=0.01,
                               edges=np.array([0.0, -0.0, 2.0 ** -24, -(2.0 ** -24), 2.0 ** -14]))
    return W


def hidden(rng, shape, hot=False):
    edges = F16_EDGE if hot else F16_EDGE[:8]
    return f16_arr(rng, shape + (4,), 1.0, edge=0.02, edges=edges)


def stage_inputs(rng, name, Nx, Ny, hot=False):
    """Inputs of one stage call on a level tensor of Nx x Ny tokens (coarse extents Nx/2 x Ny/2)."""
    info = stage_info(name)
    C = info["C"]
    el = 0.0 if hot else 0.001       # hot cases: no +-224 / +-448 inputs, so that only some rows overflow Z
    sig = 1.0
    kind = info["kind"]
    if kind == "fuse":
        return {"cin": e4m3(rng, (Ny // 2, Nx // 2, info["Kc"]), sig, 0.02, el),
                "skip": e4m3(rng, (Ny, Nx, C), sig, 0.02, el)}
    if info["adapter"]:
        return {"in": e4m3(rng, (Ny, Nx, 128), sig, 0.02, el), "hold": hidden(rng, (Ny, Nx), hot)}
    d = {"x": e4m3(rng, (Ny, Nx, C), sig, 0.02, el), "hold": hidden(rng, (Ny, Nx), hot)}
    if kind == "rec":
        d["Hc"] = hidden(rng, (Ny // 2, Nx // 2), hot)
    return d
