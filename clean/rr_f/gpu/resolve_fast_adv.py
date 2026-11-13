# SPDX-License-Identifier: MIT
"""Adversarial differential of rr_resolve_fast (+ fixup) against the verified rr_resolve_main (independent check of
the resolve speed track): game sizes (many tiles per persistent workgroup, the cross-tile prefetch), odd / strip /
tiny sizes, resources smaller than the render size (the outside rule of the RF_SPEC loader), NaN / Inf / subnormal
inputs, extreme exposures, jitters, motion, TBW, sharp precision values (tap weights below 2^-100). Every output
value compared bit for bit (NaNs one class, as GR.same_bits; NaN sign / payload differences counted separately); the
fixup bitmap must be zero again after the fixup.

    python -m clean.rr_f.gpu.resolve_fast_adv <group> [--seeds N] [--count] --json out.json
    (RF_MUT=<n> in the environment: a negative control build of the fast kernel)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import build
from . import resolve as GR
from . import resolve_fast as RF
from ..reference import resolve as RR

SPECIALS16 = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 2.0 ** -24, -2.0 ** -24, 2.0 ** -14, 2.0 ** -14 - 2.0 ** -24,
                       65504.0, -65504.0, 6e-5, 1.0, -1.0, 3 * 2.0 ** -24, -(2.0 ** -20)], np.float32).astype(np.float16)


def spec16(rng, a, frac):
    a = a.copy()
    flat = a.reshape(-1)
    k = int(frac * flat.size)
    if k:
        idx = rng.choice(flat.size, k, replace=False)
        flat[idx] = SPECIALS16[rng.integers(0, len(SPECIALS16), k)]
    return a


def make_case2(ar, rng, *, out, render, flags, table, E=1.25, Ep=1.0, jitter=(0.125, -0.25), tbw=None,
               sp=None, p_shrink=(0, 0), a_shrink=(0, 0), a_origin=(0, 0), l_shrink=(0, 0), l_param=(1, 1),
               l_upload_origin=(1, 1), p_range=(0.05, 3.0), p_signs=False, mv_mode="rand", q_mode="rand",
               th_mode="rand", lay_range=(-0.2, 1.2), bounded_wo=None, k13=None):
    from ..tests.test_resolve import symbols
    sym = symbols()
    if tbw is not None:
        sym = RR.ResolveSymbols(**{**{f: getattr(sym, f) for f in sym.__dataclass_fields__}, "TBW": np.float32(tbw)})
    if k13 is not None:   # the K13 screen (0 or normal; r3-outres): game-model-like and edge values of K13
        sym = RR.ResolveSymbols(**{**{f: getattr(sym, f) for f in sym.__dataclass_fields__}, "K13": np.float32(k13)})
    sp = sp or {}
    sz = RR.sizes(out)
    wr, hr = render
    res = {}
    if q_mode == "rand":
        q = rng.uniform(-0.5, 3, (sz.Hh, sz.Wh, 3)).astype(np.float16)
    elif q_mode == "extreme":
        q = rng.choice(np.array([65504.0, -65504.0, 60000.0, -1.0, 0.0, 2.0 ** -24], np.float16), (sz.Hh, sz.Wh, 3))
    else:
        q = (rng.uniform(-0.5, 3, (sz.Hh, sz.Wh, 3)) * 2.0 ** -14).astype(np.float16)    # subnormal-heavy
    res["Q"] = GR.Upload(ar, spec16(rng, q, sp.get("Q", 0.0)), "f16")
    lw, lh = wr + 2 - l_shrink[0], hr + 1 - l_shrink[1]
    lay = rng.uniform(lay_range[0], lay_range[1], (lh, lw, 4)).astype(np.float16)
    res["LAYER"] = GR.Upload(ar, spec16(rng, lay, sp.get("L", 0.0)), "f16", origin=l_upload_origin)
    aw, ah_ = wr - a_shrink[0], hr - a_shrink[1]
    c = rng.integers(0, 1024, (ah_, aw, 4)).astype(np.uint16)
    c[..., 3] &= 3
    res["ASRC"] = GR.Upload(ar, c, "rgb10a2", origin=a_origin, packed=True)
    if mv_mode == "rand":
        mvv = rng.uniform(-3, 3, (hr, wr, 4)).astype(np.float16)
    elif mv_mode == "huge":
        mvv = rng.choice(np.array([65504.0, -65504.0, 30000.0, -2.0 ** -24, 0.0, 1e4], np.float16), (hr, wr, 4))
    elif mv_mode == "zero":
        mvv = np.zeros((hr, wr, 4), np.float16)
    else:                                                    # sub-pixel, subnormal-ish
        mvv = (rng.uniform(-1, 1, (hr, wr, 4)) * 2.0 ** -16).astype(np.float16)
    res["MV"] = GR.Upload(ar, spec16(rng, mvv, sp.get("MV", 0.0)), "f16")
    ph, pw = hr - p_shrink[1], wr - p_shrink[0]
    p = rng.uniform(p_range[0], p_range[1], (ph, pw, 4))
    if p_signs:
        p *= rng.choice([-1.0, 1.0], p.shape)
    res["P"] = GR.Upload(ar, spec16(rng, p.astype(np.float16), sp.get("P", 0.0)), "f16")
    res["G"] = GR.Upload(ar, rng.integers(0, 256, (hr, wr, 4)).astype(np.uint8), "unorm8")
    if th_mode == "rand":
        th = rng.uniform(-0.1, 1.5, (sz.He, sz.We, 4)).astype(np.float16)
        ahv = rng.uniform(0, 1, (sz.He, sz.We, 1)).astype(np.float16)
    elif th_mode == "zero":
        th = np.zeros((sz.He, sz.We, 4), np.float16)
        ahv = np.zeros((sz.He, sz.We, 1), np.float16)
    else:                                                    # extreme
        th = rng.choice(np.array([65504.0, -65504.0, 0.0, 2.0 ** -24, -1.0, 1e4], np.float16), (sz.He, sz.We, 4))
        ahv = rng.choice(np.array([65504.0, -65504.0, 0.0, 2.0 ** -24, 1.0, 2.0], np.float16), (sz.He, sz.We, 1))
    res["THr"] = GR.Upload(ar, spec16(rng, th, sp.get("TH", 0.0)), "f16")
    res["AHr"] = GR.Upload(ar, spec16(rng, ahv, sp.get("AH", 0.0)), "f16")
    outs = []
    for _ in range(2):
        o = {}
        o["OUT"] = GR.Upload(ar, np.zeros((sz.He + 1, sz.We + 2, 4), np.float16), "f16", fill=0xA5)
        o["AOUT"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 1), np.float32), "f32", fill=0x5A)
        o["THw"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 4), np.float16), "f16", fill=0xEE)
        o["AHw"] = GR.Upload(ar, np.zeros((sz.He, sz.We, 1), np.float16), "f16", fill=0xEE)
        outs.append(o)
    blocks = []
    for i, o in enumerate(outs):
        ups = {**res, **o}
        pr = GR.params(sym, sizes=sz, render=render, flags=flags, E=np.float32(E), Ep=np.float32(Ep),
                       jitter=tuple(np.float32(j) for j in jitter), layer_origin=(0, 0), alpha_channel=3, res=ups)
        pr[13], pr[14] = l_param[0] & 0xffffffff, l_param[1] & 0xffffffff
        if bounded_wo is not None:
            pr[6], pr[7] = bounded_wo
        if i == 1:
            pr[37] = RF.fixup_bitmap(ar, sz)
            pr[38] = table
        blocks.append(ar.add(pr))
    return sz, blocks, outs


STD = "ext|layer|alpha|aout"
NAN32 = float("nan")
GAME = "ext|layer"   # the game's configuration (no alpha upscaling): RF_SPEC 2 (RF_ADV_SPEC=2 in the environment)
GROUPS = {
    "gamecfg": [("1440pQ", (2560, 1440), (1706, 960), GAME, dict(sp=dict(Q=0.001, L=0.001, TH=0.001, MV=0.0005))),
                ("1440pQ-specials", (2560, 1440), (1706, 960), GAME, dict(sp=dict(Q=0.05, L=0.05, TH=0.05, MV=0.02))),
                ("1440pQ-reset", (2560, 1440), (1706, 960), GAME + "|treset", {}),
                ("1440pQ-bounded", (2560, 1440), (1706, 960), GAME + "|bounded", dict(bounded_wo=(2001, 1111))),
                ("4kP", (3840, 2160), (1920, 1080), GAME, dict(jitter=(-0.4375, 0.3125))),
                ("l-smaller", (640, 360), (427, 240), GAME, dict(l_shrink=(6, 4), l_param=(0, 0))),
                ("l-mix", (641, 359), (427, 239), GAME, dict(l_param=(-1, 3), l_shrink=(3, 0), sp=dict(L=0.02))),
                ("th-zero", (640, 360), (427, 240), GAME, dict(th_mode="zero")),
                ("q-subnormal", (640, 360), (427, 240), GAME, dict(q_mode="subnormal")),
                ("strip-wide", (4100, 9), (2733, 6), GAME, {}),
                ("1x1", (1, 1), (1, 1), GAME, {}),
                ("odd-1279x721", (1279, 721), (853, 481), GAME, dict(sp=dict(Q=0.005, L=0.005)))],
    "k13": [(f"K13={k!r}/{m}", (640, 360), (427, 240), STD, dict(k13=k, **kw))
            for k in (1e-10, -1e-10, 2.0 ** -126, -(2.0 ** -126), 2.0 ** -25, 1e-30, 0.0)
            for m, kw in (("rand", {}), ("th-zero", dict(th_mode="zero")), ("q-subnormal", dict(q_mode="subnormal")),
                          ("specials", dict(sp=dict(Q=0.2, L=0.2, P=0.2, TH=0.2, AH=0.2, MV=0.1))))],
    "game1440": [("1440pQ", (2560, 1440), (1706, 960), STD, dict(sp=dict(Q=0.001, L=0.001, P=0.001, TH=0.001, AH=0.001, MV=0.0005))),
                 ("1440pQ-reset", (2560, 1440), (1706, 960), STD + "|treset|areset", {}),
                 ("1440pQ-bounded", (2560, 1440), (1706, 960), STD + "|bounded", dict(bounded_wo=(2001, 1111)))],
    "game4k": [("4kP", (3840, 2160), (1920, 1080), STD, dict(sp=dict(Q=0.001, L=0.001, P=0.001, TH=0.001, AH=0.001), jitter=(-0.4375, 0.3125)))],
    "game1080": [("1080pQ", (1920, 1080), (1280, 720), STD, dict(sp=dict(Q=0.002, L=0.002, P=0.002, TH=0.002, AH=0.002, MV=0.001))),
                 ("1080pUP", (1920, 1080), (640, 360), STD, {}),
                 ("1080pDLAA", (1920, 1080), (1920, 1080), STD, {})],
    "shapes": [("seq-40x27-34x23", (40, 27), (34, 23), STD, dict(sp=dict(Q=0.02, L=0.02, P=0.02, TH=0.02, AH=0.02))),
               ("seq-40x27-29x19", (40, 27), (29, 19), STD, {}),
               ("odd-1279x721", (1279, 721), (853, 481), STD, dict(sp=dict(Q=0.005, P=0.005))),
               ("strip-wide", (4100, 9), (2733, 6), STD, {}),
               ("strip-tall", (17, 4100), (12, 2733), STD, {}),
               ("1x1", (1, 1), (1, 1), STD, {}),
               ("2x1", (2, 1), (1, 1), STD, {}),
               ("16x16", (16, 16), (11, 11), STD, {}),
               ("17x17", (17, 17), (12, 12), STD, {}),
               ("33x31", (33, 31), (33, 31), STD, {}),
               ("render-odd-scale", (997, 523), (331, 174), STD, {}),
               ("render-gt-output", (64, 48), (80, 60), STD, {})],
    "outside": [("p-smaller", (640, 360), (427, 240), STD, dict(p_shrink=(5, 3))),
                ("a-origin-neg", (640, 360), (427, 240), STD, dict(a_origin=(-2, -1))),
                ("a-origin-pos-smaller", (640, 360), (427, 240), STD, dict(a_origin=(3, 2), a_shrink=(0, 0))),
                ("l-param-neg", (640, 360), (427, 240), STD, dict(l_param=(-3, -2))),
                ("l-smaller", (640, 360), (427, 240), STD, dict(l_shrink=(6, 4), l_param=(0, 0))),
                ("all-outside-mix", (641, 359), (427, 239), STD, dict(p_shrink=(2, 7), a_origin=(1, -3), l_param=(-1, 3), l_shrink=(3, 0), sp=dict(P=0.02, L=0.02)))],
    "exposure": [(f"E={e!r}/Ep={ep!r}", (320, 180), (213, 120), STD, dict(E=e, Ep=ep)) for e, ep in (
        (2.0 ** -24, 1.0), (np.float32(2.0 ** -24) * np.float32(1 - 2.0 ** -24), 1.0), (2.0 ** -23, 2.0 ** -23), (0.0, 1.0),
        (float("inf"), 1.0), (NAN32, 1.0), (1e-40, 1.0), (2.0 ** 24, 1.0), (2.0 ** 24 * 1.0000001, 1.0), (3.0e38, 1.0),
        (1.0, 0.0), (1.0, NAN32), (1.0, float("inf")), (1.0, 2.0 ** -126), (0.7, 2.0 ** 60), (2.0 ** -20, 2.0 ** 20))],
    "jitter": [(f"j={j!r}", (320, 180), (213, 120), STD, dict(jitter=j)) for j in (
        (0.5, -0.5), (-0.5, 0.5), (0.49999997, -0.49999997), (3.7, -2.2), (-11.25, 7.5), (NAN32, 0.0),
        (float("inf"), 0.0), (2.0 ** -24, 0.0), (1e-30, 0.0), (0.0, -0.0), (0.25, 0.25), (-0.0, 0.333333))],
    "prange": [(f"P in (0,{hi})", (640, 360), (427, 240), STD, dict(p_range=(0.0, hi))) for hi in (1, 3, 6, 10, 15, 20, 30, 60, 120, 1000)],
    "nanprobe": [("specials-heavy", (640, 360), (427, 240), STD, dict(sp=dict(Q=0.2, L=0.2, P=0.2, TH=0.2, AH=0.2, MV=0.1)))],
    "content": [("specials-heavy", (640, 360), (427, 240), STD, dict(sp=dict(Q=0.2, L=0.2, P=0.2, TH=0.2, AH=0.2, MV=0.1))),
                ("specials-heavy-reset", (640, 360), (427, 240), STD + "|treset|areset", dict(sp=dict(Q=0.2, L=0.2, P=0.2, MV=0.1))),
                ("mv-huge", (640, 360), (427, 240), STD, dict(mv_mode="huge")),
                ("mv-zero", (640, 360), (427, 240), STD, dict(mv_mode="zero")),
                ("mv-subpix", (640, 360), (427, 240), STD, dict(mv_mode="subpix")),
                ("p-huge-signs", (640, 360), (427, 240), STD, dict(p_range=(0.0, 65504.0), p_signs=True)),
                ("p-tiny-signs", (640, 360), (427, 240), STD, dict(p_range=(0.0, 2.0 ** -14), p_signs=True)),
                ("p-zero", (640, 360), (427, 240), STD, dict(p_range=(0.0, 0.0))),
                ("q-extreme", (640, 360), (427, 240), STD, dict(q_mode="extreme")),
                ("q-subnormal", (640, 360), (427, 240), STD, dict(q_mode="subnormal")),
                ("th-zero", (640, 360), (427, 240), STD, dict(th_mode="zero")),
                ("th-extreme", (640, 360), (427, 240), STD, dict(th_mode="extreme")),
                ("layer-extreme", (640, 360), (427, 240), STD, dict(lay_range=(-65504.0, 65504.0))),
                ("layer-tiny", (640, 360), (427, 240), STD, dict(lay_range=(-2.0 ** -14, 2.0 ** -14))),
                ("tbw-0", (320, 180), (213, 120), STD, dict(tbw=0.0)),
                ("tbw-1", (320, 180), (213, 120), STD, dict(tbw=1.0)),
                ("tbw-tiny", (320, 180), (213, 120), STD, dict(tbw=2.0 ** -24)),
                ("tbw-denorm", (320, 180), (213, 120), STD, dict(tbw=1e-40)),
                ("tbw-near1", (320, 180), (213, 120), STD, dict(tbw=1 - 2.0 ** -24)),
                ("tbw-neg", (320, 180), (213, 120), STD, dict(tbw=-0.5)),
                ("noalpha3", (320, 180), (213, 120), STD + "|areset", {}),
                ("nolayer3", (320, 180), (213, 120), STD + "|treset", {})],
}


def raw_nan_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind != "f":
        return 0
    u = np.uint32 if a.itemsize == 4 else np.uint16
    m = np.isnan(a) & np.isnan(b) & (a.view(u) != b.view(u))
    if m.any():
        idx = np.argwhere(m)[:4]
        print("    NaN sign / payload:", [(tuple(int(t) for t in i), hex(int(a.view(u)[tuple(i)])), hex(int(b.view(u)[tuple(i)]))) for i in idx])
    return int(np.count_nonzero(m))


def run_group(name, seeds, json_path, count):
    kernels = {"ref": (RF.verified_kernel(), GR.NARGS), "fast": (RF.fast_kernel(count, spec=int(os.environ.get("RF_ADV_SPEC", "1")), extra={"RF_MUT": int(os.environ["RF_MUT"])} if os.environ.get("RF_MUT") else None), RF.NARGS),
               "fixup": (RF.fixup_kernel(False), RF.NARGS)}
    rows, all_ok = [], True
    for seed in range(seeds):
        ar = build.Arena()
        disp, recs = [], []
        table = RF.exp2_table(ar, disp, kernels)
        rng = np.random.default_rng(777000 + 31 * seed + sum(map(ord, name)))
        for cname, out, render, fl, opt in GROUPS[name]:
            flags = RF.parse_flags(fl)
            sz, blocks, outs = make_case2(ar, rng, out=out, render=render, flags=flags, table=table, **opt)
            cnt = ar.add(np.zeros(4, np.uint32))
            disp.append(("ref", (sz.We * sz.He + GR.BLOCK - 1) // GR.BLOCK, [blocks[0], 0, 0]))
            disp.append(("fast", RF.fast_groups(sz), [blocks[1], cnt if count else 0, 0]))
            disp.append(("fixup", RF.fixup_groups(sz), [blocks[1], 0, 0]))
            recs.append((cname, outs, cnt, sz, blocks[1]))
        print(f"{name} seed {seed}: arena {ar.size / 1e6:.0f} MB, {len(disp)} dispatches", flush=True)
        img, log = build.run(ar, kernels, disp, require_double=True, workdir=str(Path(build.BUILD) / f"adv-{name}"),
                             timeout=7200)
        for cname, outs, cnt, sz, blk in recs:
            r = {}
            ok = True
            for key in ("OUT", "AOUT", "THw", "AHw"):
                a, b = outs[0][key].read(img), outs[1][key].read(img)
                okk, bad = GR.same_bits(a, b)
                r[key] = dict(values=int(a.size), mismatches=bad, nan_payload_diffs=raw_nan_diff(a, b),
                              nans=int(np.count_nonzero(np.isnan(a))) if a.dtype.kind == "f" else 0)
                ok &= okk
            # the fixup bitmap must be all zero again after the fixup
            bm_words = (sz.We * sz.He + 31) // 32
            bmo = int(img[blk:blk + 4 * 40].view(np.uint32)[37])
            left = int(np.count_nonzero(img[bmo:bmo + 4 * bm_words].view(np.uint32)))
            r["bitmap_left"] = left
            ok &= left == 0
            if count:
                r["slow_pixels"] = int(img[cnt:cnt + 4].view(np.uint32)[0])
            all_ok &= ok
            rows.append(dict(seed=seed, case=cname, passed=ok, outputs=r))
            print(f"  {cname:28s} {'ok' if ok else 'MISMATCH'} " + " ".join(
                f"{k}:{v['mismatches']}/{v['values']}(nan {v['nans']}, payload {v['nan_payload_diffs']})"
                for k, v in r.items() if isinstance(v, dict)) + f" bm_left {left}" +
                (f" slow {r['slow_pixels']}" if count else ""), flush=True)
    rep = dict(group=name, passed=all_ok, cases=rows)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=list(GROUPS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    raise SystemExit(run_group(a.group, a.seeds, a.json, a.count))
