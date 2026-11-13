# SPDX-License-Identifier: MIT
"""Model K input stage (spec 42) and output stage (spec 43): launch arguments, host-side scalars and launch plans.

Kernels: clean/k/kernels/k_io.hip (code object k_io.co). Frame inputs (dicts as clean/k/reference/instage.py and
outstage.py take them): color [.., .., 3 or 4] float32 (RGB(A); alpha 1 if missing), motion [.., .., 2], depth [.., ..]
([MVlo]), subrect bases (default 0), render_size, output_size, flags hdr / mv_lowres / depth_inverted, jitter,
mv_scale, mv_offset, pre_exposure, reset, E_cur / E_prev / E_aux (FP16 values), e1, e2.
"""
import ctypes
import math

import numpy as np

from clean.k.reference.fp import F32
from clean.k.reference.layout import base_size, cdiv
from clean.k.reference.outstage import ceil16, filter_level, filter_table

CODE_OBJECT = "k_io.co"
EXACT_CODE_OBJECT = "k_io_exact.co"  # the same with the bit-exact input stage (WGK_IN_EXACT)
KERNELS = ("k_out_color", "k_out_pixel", "k_out_hidden", "k_in_color", "k_in_base", "k_in_reset")
THREADS = 128
HDR, MVLO, INV, RESET = 1, 2, 4, 8


class KFrameIn(ctypes.Structure):
    """xs: 0 (kappa, rho, lam and KOutUni.kinv from this struct), or the device address of the exposure scalars that
    k_expo_scalars writes (the per-frame runtime's GPU-resident exposure)."""
    _fields_ = [(n, ctypes.c_void_p) for n in ("color", "motion", "depth", "xs")] + \
               [(n, ctypes.c_int) for n in ("cpitch", "mpitch", "zpitch", "cbx", "cby", "mbx", "mby", "zbx", "zby",
                                            "W", "H", "rw", "rh", "Bw", "Bh", "flags")] + \
               [(n, ctypes.c_float) for n in ("jx", "jy", "sx", "sy", "mox", "moy", "kappa", "rho", "lam", "s")] +                [("mch", ctypes.c_int)] + [(n, ctypes.c_float) for n in ("rmsx", "rmsy", "rmox", "rmoy")] +                [("pad_", ctypes.c_int)]


class KOutUni(ctypes.Structure):
    """Frame-uniform FP32 values of the output stage (out_uniforms)."""
    _fields_ = [(n, ctypes.c_float) for n in ("invW", "invH", "rhox", "rhoy", "rho2x", "rho2y", "jmx", "jmy", "spx",
                                              "spy", "invW16", "invH16", "kinv")] + \
               [("kux", ctypes.c_uint16), ("kvy", ctypes.c_uint16)]


class KOutParams(ctypes.Structure):
    _fields_ = [("f", KFrameIn)] + \
               [(n, ctypes.c_void_p) for n in ("N", "L", "Hc", "table", "exp2tab", "out", "Lout", "Hcout", "Hhout")] + \
               [("opitch", ctypes.c_int), ("omode", ctypes.c_int)] + \
               [(n, ctypes.c_void_p) for n in ("ctap", "ctapT")] + [("u", KOutUni)]


class KInParams(ctypes.Structure):
    _fields_ = [("f", KFrameIn)] + [(n, ctypes.c_void_p) for n in ("Hc", "Hh", "x0", "cpre", "cpre2")]


class KInColorParams(ctypes.Structure):
    """Tiled input and uniquely owned output taps; existing input ABI is unchanged."""
    _fields_ = [("input", KInParams), ("ctap", ctypes.c_void_p), ("ctapT", ctypes.c_void_p)]


class KExpoArgs(ctypes.Structure):
    """Host arguments of k_expo_scalars (the per-frame runtime's GPU exposure scalars, clean/k/runtime/runtime_k.cpp)."""
    _fields_ = [("value", ctypes.c_float)] + [(n, ctypes.c_int) for n in ("pair", "use_src", "first", "hdr", "reset")] + \
               [(n, ctypes.c_float) for n in ("e1", "e2", "inv_pi")]


# k_out_pixel's game-output formats (KOutParams.omode)
OUT_RGBA32F, OUT_RGBA16F, OUT_RGB32F = 0, 1, 2

assert ctypes.sizeof(KFrameIn) == 160 and ctypes.sizeof(KOutUni) == 56 and ctypes.sizeof(KOutParams) == 312 and \
    ctypes.sizeof(KInParams) == 200 and ctypes.sizeof(KExpoArgs) == 36
assert ctypes.sizeof(KInColorParams) == 216 and KInColorParams.ctap.offset == 200 and KInColorParams.ctapT.offset == 208


def out_uniforms(fin):
    """The output stage's frame-uniform values (spec 43 section 1 ratios and the other values that depend only on the
    frame), each FP32 operation as the kernel would do it (NumPy float32 rounds every operation once, as the GPU's
    correctly rounded operations do), so k_out_pixel reads them instead of computing them per pixel."""
    f = np.float32
    with np.errstate(all="ignore"):
        invW, invH = f(1) / f(fin.W), f(1) / f(fin.H)
        rhox, rhoy = f(fin.rw) * invW, f(fin.rh) * invH
        rho2x, rho2y = f(0.5) * rhox, f(0.5) * rhoy
        jmx, jmy = f(fin.jx) - f(0.5), f(fin.jy) - f(0.5)
        sx, sy = f(fin.sx), f(fin.sy)
        spx, spy = (sx / rhox, sy / rhoy) if fin.flags & MVLO else (sx, sy)
        invW16, invH16 = f(1) / f(ceil16(fin.W)), f(1) / f(ceil16(fin.H))
        kinv = f(1) / f(fin.kappa)
        kux, kvy = (np.float16(f(-8) / r).view(np.uint16) for r in (rho2x, rho2y))
    return KOutUni(invW=invW, invH=invH, rhox=rhox, rhoy=rhoy, rho2x=rho2x, rho2y=rho2y, jmx=jmx, jmy=jmy, spx=spx,
                   spy=spy, invW16=invW16, invH16=invH16, kinv=kinv, kux=int(kux), kvy=int(kvy))


def compact_head(head):
    """The compact pixel head [2 Bh, 2 Bw, 4] of a head map [Bh, Bw, 40] (any dtype, values copied): slot k of base
    position (x, y) at row 2 y + (k >> 1), column 2 x + (k & 1), holding head channels 10 k, 10 k + 1, 10 k + 9 and a
    zero: what k_stage2_d0_hp / k_stage2h_d0_hp write and k_out_pixel_c reads (clean/k/kernels/k_block.hip)."""
    head = np.asarray(head)
    h, w, _ = head.shape
    out = np.zeros((2 * h, 2 * w, 4), head.dtype)
    for k in range(4):
        for i, c in enumerate((10 * k, 10 * k + 1, 10 * k + 9)):
            out[(k >> 1)::2, (k & 1)::2, i] = head[..., c]
    return out


def output_scratch_bytes(rw, rh):
    """Scratch of the output stage at render size rw x rh: k_out_color's stored tap (4 FP16) and tap value T (FP16)."""
    return {"ctap": rw * rh * 8, "ctapT": rw * rh * 2}


def input_scratch_bytes(rw, rh):
    """Scratch of the input stage at render size rw x rh: k_in_color's (Y, Co, Cg, phi) and (HDR weight, own-sample
    channel) per pixel."""
    return {"cpre": rw * rh * 16, "cpre2": rw * rh * 8}

_EXP2 = None


def exp2_table():
    """h(2^E) for every binary16 bit pattern E (float16 [65536]): the output stage's g = h(2^E), exact."""
    global _EXP2
    if _EXP2 is None:
        e = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
        with np.errstate(all="ignore"):
            _EXP2 = np.exp2(e).astype(np.float16)
    return _EXP2


def scalars(x):
    """Host-side FP32 scalars (specs 42 section 4 / 43 section 3.7): kappa, rho (history scale), lambda (luma scale)."""
    def ev(e):
        e = F32(np.float16(e))
        return F32(1) if e == 0 else e
    hdr = bool(x["hdr"])
    e1, e2 = F32(x.get("e1", 1.0)), F32(x.get("e2", 1.0))
    with np.errstate(all="ignore"):
        ce = e1 * ev(x["E_cur"])
        if hdr:
            kappa = (F32(1) / F32(x.get("pre_exposure", 1.0))) * ce
            kappa = F32(1) if kappa == 0 else kappa
            rho = ce / (e1 * ev(x["E_prev"]))
        else:
            kappa, rho = F32(1), F32(1)
        lam = (e2 * ev(x.get("E_aux", x["E_cur"]))) / ce
    return float(kappa), float(rho), float(lam)


def _rgba(color):
    c = np.asarray(color, dtype=np.float32)
    if c.shape[-1] == 4:
        return np.ascontiguousarray(c)
    out = np.ones(c.shape[:2] + (4,), np.float32)
    out[..., :3] = c[..., :3]
    return out


def frame_in(x, bufs):
    """KFrameIn for the frame dict x; bufs: device addresses of "color", "motion", "depth"."""
    W, H = (int(v) for v in x["output_size"])
    rw, rh = (int(v) for v in x["render_size"])
    from clean.k.reference.layout import DEFAULT_S
    s = float(x.get("s", DEFAULT_S))  # spec 44's base-size factor
    Bw, Bh = tuple(x["base"]) if x.get("base") is not None else base_size(W, H, s)  # "base": test-only override
    color, motion = np.asarray(x["color"]), np.asarray(x["motion"])
    depth = np.asarray(x["depth"]) if x.get("depth") is not None else np.zeros((1, 1), np.float32)
    flags = (HDR if x["hdr"] else 0) | (MVLO if x["mv_lowres"] else 0) | (INV if x["depth_inverted"] else 0) | \
            (RESET if x["reset"] else 0)
    kappa, rho, lam = scalars(x)
    cb, mb, zb = (tuple(int(v) for v in x.get(k, (0, 0))) for k in ("color_base", "motion_base", "depth_base"))
    jx, jy = (float(np.float32(v)) for v in x["jitter"])
    sx, sy = (float(np.float32(v)) for v in x["mv_scale"])
    mox, moy = (float(np.float32(v)) for v in x.get("mv_offset", (0, 0)))
    return KFrameIn(color=bufs["color"], motion=bufs["motion"], depth=bufs["depth"], cpitch=color.shape[1],
                    mpitch=motion.shape[1], zpitch=depth.shape[1], cbx=cb[0], cby=cb[1], mbx=mb[0], mby=mb[1],
                    zbx=zb[0], zby=zb[1], W=W, H=H, rw=rw, rh=rh, Bw=Bw, Bh=Bh, flags=flags, jx=jx, jy=jy, sx=sx,
                    sy=sy, mox=mox, moy=moy, kappa=kappa, rho=rho, lam=lam, s=float(np.float32(s)))


def upload_frame(alloc, x):
    """Allocate and upload the frame textures; alloc(name, nbytes, data) -> device address."""
    color = _rgba(x["color"])
    motion = np.ascontiguousarray(np.asarray(x["motion"], dtype=np.float32))
    depth = np.ascontiguousarray(np.asarray(x["depth"], dtype=np.float32)) if x.get("depth") is not None else \
        np.zeros((1, 1), np.float32)
    return {"color": alloc("color", color.nbytes, color.tobytes()),
            "motion": alloc("motion", motion.nbytes, motion.tobytes()),
            "depth": alloc("depth", depth.nbytes, depth.tobytes())}


def f16_bytes(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32).astype(np.float16)).tobytes()


def output_launches(fin, bufs, sizes):
    """k_out_color + k_out_pixel + k_out_hidden. bufs: N, L, Hc, table, exp2tab, out, Lout, Hcout, Hhout and the scratch
    ctap, ctapT (output_scratch_bytes); sizes: opitch (texels), omode (OUT_*: game-output format)."""
    prm = KOutParams(f=fin, N=bufs["N"], L=bufs["L"], Hc=bufs["Hc"], table=bufs["table"], exp2tab=bufs["exp2tab"],
                     out=bufs["out"], Lout=bufs["Lout"], Hcout=bufs["Hcout"], Hhout=bufs["Hhout"],
                     opitch=sizes.get("opitch", fin.W), omode=sizes.get("omode", OUT_RGBA32F), ctap=bufs["ctap"],
                     ctapT=bufs["ctapT"], u=out_uniforms(fin))
    tiles = cdiv(ceil16(fin.W), 64) * (ceil16(fin.H) // 2)  # k_out_pixel: workgroups of 64 x 2 pixels
    return [("k_out_color", cdiv(fin.rw * fin.rh, THREADS), THREADS, [prm]),
            ("k_out_pixel", tiles, THREADS, [prm]),
            ("k_out_hidden", cdiv(fin.Bw * fin.Bh, THREADS), THREADS, [prm])]


def input_tiled_supported(fin):
    """Conservative footprint/coordinate guard for the 24x24 producer-local tile.

    Unsupported frames retain BOTH original input launches. Runtime mirror:
    clean/k/runtime/runtime_k.cpp. Standalone stage calls remain unfused by default.
    """
    return (fin.s == 0.5 and 0 < fin.W <= 16384 and 0 < fin.H <= 16384 and
            0 < fin.Bw <= 4096 and 0 < fin.Bh <= 4096 and
            0 < fin.rw <= .68 * fin.W and 0 < fin.rh <= .68 * fin.H and
            math.isfinite(fin.jx) and math.isfinite(fin.jy) and abs(fin.jx) <= .5 and abs(fin.jy) <= .5)


def input_launches(fin, bufs, exact=False, tiled=False):
    """k_in_color (per render pixel) + k_in_base (k_in_reset on history resets). bufs: Hc, Hh, x0 and the scratch cpre, cpre2
    (input_scratch_bytes). exact: the bit-exact build (k_io_exact.co) instead of the default fast arithmetic.
    tiled: GPU-only fusion from complete-frame callers; unsupported geometry preserves the original pair."""
    prm = KInParams(f=fin, Hc=bufs["Hc"], Hh=bufs["Hh"], x0=bufs["x0"], cpre=bufs["cpre"], cpre2=bufs["cpre2"])
    if tiled and input_tiled_supported(fin):
        prm.cpre = prm.cpre2 = None  # fused producer reads raw colour; global scratch is not consumed
        co = EXACT_CODE_OBJECT if exact else "k_io.co"
        return [(co + ":k_in_tiled", cdiv(fin.Bw, 8) * cdiv(fin.Bh, 8), 256, [prm])]
    co = EXACT_CODE_OBJECT + ":" if exact else ""
    base_kernel = "k_in_reset" if fin.flags & RESET else "k_in_base"
    return [(co + "k_in_color", cdiv(fin.rw * fin.rh, THREADS), THREADS, [prm]),
            (co + base_kernel, cdiv(4 * fin.Bw * fin.Bh, THREADS), THREADS, [prm])]  # one lane per display position


def input_tiled_color_supported(fin):
    """Full render coverage for unique producer ownership; reset keeps separate taps."""
    return (input_tiled_supported(fin) and not (fin.flags & RESET) and
            4 * fin.Bw >= fin.W and 4 * fin.Bh >= fin.H)


def complete_input_launches(fin, bufs, exact=False, tiled=False):
    """Return input launches and whether they also prepare ctap/T for this frame.

    Standalone input/output plans stay unchanged. Exact-input and fallback frames
    keep separate output-color preparation in the complete-frame caller.
    """
    launches = input_launches(fin, bufs, exact=exact, tiled=tiled)
    if tiled and not exact and input_tiled_color_supported(fin):
        _, grid, threads, args = launches[0]
        prm = KInColorParams(input=args[0], ctap=bufs["ctap"], ctapT=bufs["ctapT"])
        return [(CODE_OBJECT + ":k_in_tiled_color", grid, threads, [prm])], True
    return launches, False


def table_for(x):
    """The frame's filter-shape set (spec 43 section 7): x["table"] if given, else generated for the frame's level."""
    if x.get("table") is not None:
        return np.asarray(x["table"], dtype=np.float16)
    W, H = x["output_size"]
    rw, rh = x["render_size"]
    return filter_table(filter_level(rw, rh, W, H))
