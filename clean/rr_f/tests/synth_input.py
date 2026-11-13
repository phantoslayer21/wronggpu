# SPDX-License-Identifier: MIT
"""Synthetic inputs of the spec 52 input stage: symbol values, IN0 tensors, textures, temporal state, cameras.

Every value here is our own synthetic choice (no converter output, no player data). Symbol values are plausible
magnitudes chosen so that every clamp and branch of page 52 is reachable with random textures."""
import numpy as np

from clean.rr_f.reference import f32 as F
from clean.rr_f.reference import input_stage as IS
from clean.rr_f.reference.fmt import from_f16_bits, q8, round_f16, to_f16_bits
from clean.rr_f.tests import synth


def symbols(rng=None, b2=False, flavour=0):
    base = dict(K4=1.5, K5=1e-3, K7=0.8, K7n=-0.8, K10=3.5, C_HDEC=3.0, C_FCLIP=6.0, C_AMAX=0.875,
                C_GCL=2.5, C_GCLn=-2.25, C_LWrb=0.25, C_LWg=0.5, C_MDL=3.0, C_MDLn=-2.75, Gm=0.5, Gd=1.25,
                C_SV1=0.01, C_SV2=100.0, C_SV3=1e-6)
    if flavour == 1:
        base.update(K4=0.75, K7=1.25, K7n=-1.3, C_FCLIP=2.0, C_AMAX=1.25, Gm=2.0, Gd=0.75, C_MDL=1.5,
                    C_MDLn=-1.0, C_SV1=0.05, C_SV2=8.0, C_SV3=1e-3)
    v = {k: np.float32(x) for k, x in base.items()}
    v["K8"] = F.rcp(v["K7"])
    v["K12"] = F.rcp(v["K4"])
    v["Gh"] = np.float32(np.float16(0.75 if flavour == 0 else 1.5))
    v["Gs"] = np.float32(np.float16(1.5 if flavour == 0 else -0.625))
    if b2:
        v["C_SV4"] = np.float32(64.0)
        v["C_SV4n"] = np.float32(-48.0)
    return IS.InputSymbols(v, b2)


def symbol_groups(S):
    """The same symbols as converter-style groups (name -> array [1])."""
    c = {n: np.array([S.values[n]], np.float32) for n in IS.SYMBOLS_F32}
    c.update({n: np.array([S.values[n]], np.float16) for n in IS.SYMBOLS_F16})
    b2 = {n: np.array([S.values[n]], np.float32) for n in IS.SYMBOLS_B2} if S.b2 else None
    return c, b2


def fe_weights(rng, zero_rows=True, scale=1.0):
    W = {}
    for br in (0, 1):
        w = synth.e4m3(rng, (64, 32), scale / np.sqrt(42), edge_small=0.005)
        if zero_rows:
            for kk in range(64):
                if IS.nu(kk) >= 43:
                    w[kk] = 0
        W[f"W1_{br}"] = w
        W[f"b1_{br}"] = synth.f16_arr(rng, (32,), 0.2)
        W[f"W2_{br}"] = synth.e4m3(rng, (32, 32), scale / np.sqrt(64), edge_small=0.005)
    W["b2"] = synth.f16_arr(rng, (32,), 0.2)
    return W


def in0_group(rng, hot=False):
    W = synth.stage_weights(rng, "IN0", hot=hot)
    for n, v in fe_weights(rng).items():
        W["FE." + n] = v
    W["D"] = rng.integers(0, 65536, 2048).astype(np.uint16)
    return W


def shift_table(rng):
    return rng.choice(np.array([0, 2, 4, 6], np.uint32), size=(2, 11, 2)).astype(np.uint32)


def perspective(rng, reverse=False, ortho=False):
    zn, zf = np.float32(rng.uniform(0.05, 1.0)), np.float32(rng.uniform(50, 5000))
    f = rng.uniform(0.8, 2.5)
    a = rng.uniform(0.6, 2.2)
    M = np.zeros((4, 4), np.float32)
    if ortho:
        M[0][0], M[1][1] = 1 / rng.uniform(5, 50), 1 / rng.uniform(5, 50)
        M[2][2] = 1 / (zf - zn) if not reverse else 1 / (zn - zf)
        M[3][2] = -zn / (zf - zn) if not reverse else zf / (zf - zn)
        M[3][3] = 1
        return M
    M[0][0], M[1][1] = f / a, f
    if reverse:
        M[2][2] = zn / (zn - zf)
        M[3][2] = -zf * zn / (zn - zf)
    else:
        M[2][2] = zf / (zf - zn)
        M[3][2] = -zn * zf / (zf - zn)
    M[2][3] = 1
    if rng.random() < 0.3:                      # jittered projection: small translation terms
        M[2][0], M[2][1] = rng.uniform(-1e-3, 1e-3, 2)
    return M.astype(np.float32)


def world_to_view(rng, far=1000.0):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    M = np.eye(4)
    M[:3, :3] = q
    M[3, :3] = rng.uniform(-far, far, 3)
    return M.astype(np.float32)


def perturb(rng, M, eps):
    return (M + rng.normal(0, eps, M.shape) * (np.abs(M) + 0.1)).astype(np.float32)


def camera_kind(rng, kind):
    """kind: perspective, reverse, ortho, identity, zero, random, nonfinite, static."""
    if kind == "identity":
        I = np.eye(4, dtype=np.float32)
        return I, I, I, I
    if kind == "zero":
        Z = np.zeros((4, 4), np.float32)
        return Z, Z, Z, Z
    if kind == "random":
        return tuple(rng.normal(0, 3, (4, 4)).astype(np.float32) for _ in range(4))
    V2C = perspective(rng, reverse=(kind == "reverse"), ortho=(kind == "ortho"))
    W2V = world_to_view(rng)
    if kind == "static":
        return V2C, W2V, V2C.copy(), W2V.copy()
    V2Cp = perturb(rng, V2C, 1e-3)
    W2Vp = W2V.copy()
    W2Vp[:3, :3] = perturb(rng, W2V[:3, :3], 2e-3)
    W2Vp[3, :3] = W2V[3, :3] + rng.normal(0, 0.02, 3).astype(np.float32)
    if kind == "nonfinite":
        W2Vp = W2Vp.copy()
        W2Vp[3][rng.integers(0, 3)] = np.float32(np.inf)
    return V2C, W2V, V2Cp, W2Vp


def texture(rng, shape, lo, hi, edge=0.0, edge_values=None):
    t = rng.uniform(lo, hi, shape).astype(np.float32)
    if edge and edge_values is not None:
        msk = rng.random(shape) < edge
        t = np.where(msk, rng.choice(np.asarray(edge_values, np.float32), size=shape), t)
    return t.astype(np.float32)


EDGES = [0.0, -0.0, np.inf, -np.inf, np.nan, 1e30, -1e30, 2.0 ** -130, 65504.0, 70000.0, -70000.0]


def make_case(rng, Wr=45, Hr=37, Wo=None, Ho=None, reset=False, rev=False, hw_depth=True, nrm_dec=False,
              rgh_packed=False, guides=None, smv=None, b2=False, camera="perspective", edge=0.0, origins=True,
              flavour=0, mv_jittered=False, dis=None, specular_from_v=False, par=None, fc=None, hot=False,
              tex_margin=None, tiny=False):
    """One synthetic frame. smv: None (no specular role), 'SV' or 'Hd'. Returns
    (frame, params, S, IN0 group, consts, wc)."""
    Wo = Wo or int(Wr * rng.uniform(1.0, 1.6))
    Ho = Ho or int(Hr * rng.uniform(1.0, 1.6))
    Wh, Hh = IS.history_size((Wo, Ho))
    S = symbols(rng, b2=b2, flavour=flavour)
    tex = {}
    roles = ["C", "Z", "V", "Dd", "Ds", "N"] + ([] if rgh_packed else ["R"])
    guides = guides if guides is not None else [g for g in ("Q", "B", "G1", "G2") if rng.random() < 0.5]
    roles += [g for g in guides if g != "Q"]
    if dis is None:
        dis = "Q" in guides
    if dis:
        roles.append("Q")
    if smv == "SV":
        roles.append("SV")
    elif smv == "Hd":
        roles.append("Hd")
    if rng.random() < 0.3 and smv == "SV":
        roles.append("Hd")                      # SV takes precedence over Hd (52.8)
    roles.sort(key=lambda r: r not in ("C", "V"))         # sources of borrowed origins first
    for role in roles:
        ch = {"C": 4, "Z": 1, "V": 2, "SV": 2, "Dd": 3, "Ds": 3, "N": 4 if rgh_packed else 3, "R": 1, "Q": 1,
              "B": 3, "G1": 1, "G2": 1, "Hd": 1}[role]
        o = (int(rng.integers(0, 9)), int(rng.integers(0, 7))) if origins else (0, 0)
        mx = tex_margin if tex_margin is not None else (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
        so = {"SV": "V", "B": "C", "G1": "C", "G2": "C"}.get(role)
        ro = tex[so].origin if so else o                 # the origin the reader applies (52.1.2)
        shape = (Hr + ro[1] + mx[1] + 2, Wr + ro[0] + mx[0] + 2, ch)
        if role in ("C", "B"):
            data = texture(rng, shape, -0.2, 6.0 if not hot else 4e4, edge, EDGES)
        elif role == "Z":
            data = texture(rng, shape, 0.0, 1.0, edge, [0.0, 1.0, np.nan, np.inf, -1.0, 2.0 ** -130])
            if not hw_depth:
                data = texture(rng, shape, 0.1, 400.0, edge, [0.0, np.inf, np.nan, -3.0])
            # repeated values for ties
            data = np.where(rng.random(shape) < 0.1, np.float32(0.5), data).astype(np.float32)
        elif role in ("V", "SV"):
            data = texture(rng, shape, -12.0, 12.0, edge, [np.nan, np.inf, -np.inf, 1e20, -1e20, 0.0])
        elif role in ("Dd", "Ds"):
            data = texture(rng, shape, -0.1, 1.2, edge, EDGES)
        elif role == "N":
            data = texture(rng, shape, 0.0 if nrm_dec else -1.1, 1.1, edge, EDGES)
        elif role == "R":
            data = texture(rng, shape, 0.0, 1.1, edge, EDGES)
        elif role == "Q":
            data = texture(rng, shape, 0.0, 1.0, edge, [np.nan, np.inf, -1.0])
        elif role in ("G1", "G2"):
            data = texture(rng, shape, -3.0, 5.0, edge, EDGES)
        elif role == "Hd":
            data = texture(rng, shape, -1.0, 200.0, edge, [0.0, -0.0, np.nan, np.inf, -np.inf, 100.0, 1e30])
        if tiny and role in ("C", "Dd", "Ds", "B", "G1", "G2", "N", "Hd"):
            # values near the smallest normal f32: products and fma results that underflow (flush rules, 52.3)
            msk = rng.random(data.shape) < 0.3
            data = np.where(msk, (data * np.float32(2.0 ** -120)).astype(np.float32), data).astype(np.float32)
        tex[role] = IS.Texture(data, o)
    ph1 = rng.integers(0, 1024, (-(-Hh // 2), -(-Wh // 2), 4)).astype(np.uint16)
    ph1[..., 3] = 3
    ph2 = rng.integers(0, 1024, (-(-Hh // 4), -(-Wh // 4), 4)).astype(np.uint16)
    ph2[..., 3] = 3
    hid = rng.integers(-128, 128, (Hr, Wr, 4)).astype(np.int8)
    pm = round_f16(rng.uniform(-10, 10, (Hr, Wr, 4))).astype(np.float16)
    if edge:
        msk = rng.random(pm.shape) < edge
        pm = np.where(msk, rng.choice(np.array([np.nan, np.inf, -np.inf, 65504.0], np.float16), pm.shape), pm)
    prior = lambda c: round_f16(rng.uniform(-3, 3, (Hr, Wr, c))).astype(np.float16)
    frame = IS.InputFrame(tex, PH1=ph1, PH2=ph2, HID=hid, PM_read=pm, PM_write=prior(4), Pe=prior(2), Pn=prior(2))
    V2C, W2V, V2Cp, W2Vp = camera_kind(rng, camera)
    SH = shift_table(rng)
    E = np.float32(np.float16(rng.uniform(0.25, 4.0)))
    params = IS.FrameParams(
        render_size=(Wr, Hr), output_size=(Wo, Ho),
        jitter=(np.float32(rng.uniform(-0.5, 0.5)), np.float32(rng.uniform(-0.5, 0.5))),
        prev_jitter=(np.float32(rng.uniform(-0.5, 0.5)), np.float32(rng.uniform(-0.5, 0.5))),
        mv_scale_raw=(np.float32(rng.uniform(0.5, 1.5) / Wr), np.float32(-rng.uniform(0.5, 1.5) / Hr)),
        mv_jittered=mv_jittered, E=E, Ep=np.float32(1.0) if reset else np.float32(np.float16(rng.uniform(0.25, 4))),
        X3=np.float32(0) if reset else np.float32(rng.uniform(-1, 1)), reset=reset,
        par=int(rng.integers(0, 2)) if par is None else par, fc=int(rng.integers(0, 8)) if fc is None else fc,
        SH=SH, hw_depth=hw_depth, depth_inverted=rev, nrm_dec=nrm_dec, rgh_packed=rgh_packed,
        V2C=V2C, W2V=W2V, V2Cp=V2Cp, W2Vp=W2Vp, specular_from_v=specular_from_v)
    IN0 = in0_group(rng, hot=hot)
    sc = synth.stage_consts(rng)
    consts = {"ra": sc["ra"], "m": sc["m"], "g": sc["g"]}
    wc, _ = synth.window_consts(rng, "A")
    return frame, params, S, IN0, consts, wc
