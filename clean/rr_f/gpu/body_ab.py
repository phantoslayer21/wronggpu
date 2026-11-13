# SPDX-License-Identifier: MIT
"""A/B of body stage kernels at a game size: the base kernels (a copy of clean/rr_f/kernels at the base commit,
--base, default <WG_RR_BUILD>/../base_kernels/clean/rr_f/kernels) against the current kernels, on the same inputs,
in one host call (interleaved timing: every timed run executes base and new back to back per stage), with a raw
byte comparison of every output buffer (y2, merged, hnew, hmerged; also the bytes a kernel must not write).

    python -m clean.rr_f.gpu.body_ab [--stages IN0,D0-F] [--render 1706x960] [--step E] [--z f32] [--reps 20]
                                     [--data rand|synth|same|zero|smallw|adv|adv2|bigw|advbig|adv2big] [--census]
                                     [--only new|base] [--nolock]

adv / adv2 put special values into every input (adv2: NaN / inf sparse, so most windows stay finite), bigw
large weight codes (overflow, saturation), advbig / adv2big both; --census counts the special values of the
outputs (which paths the data reached).

Exit 0 when every compared output is byte-identical."""
import argparse
import os
import sys
import time

import numpy as np

from clean.rr_f.gpu import bench, build, stage
from clean.rr_f.reference.fmt import f16
from clean.rr_f.reference.stages import stage_info
from clean.rr_f.tests import synth
from clean.rr_f.tests.vectors_io import load_manifest, window_consts

DEFAULT_BASE = os.path.join(os.path.dirname(build.BUILD.rstrip("\\/")), "base_kernels", "clean", "rr_f", "kernels")


def base_kernel(base_dir, name, zmode, step):
    """The base commit's kernel of a stage (compiled from the copy, cached under its own hash)."""
    old = build.KDIR
    build.KDIR = base_dir
    try:
        tag = "base_" + name.replace("-", "") + "_z" + zmode + "_" + step
        return build.compile_kernel("rr_block.hlsl", "rr_block", stage.stage_defines(name, zmode, step, 1), tag=tag)
    finally:
        build.KDIR = old


class _Out:
    def __init__(self, out):
        self.out = out


def d0r_calls(a, ar, kernels, disp, rng, W, wc, Wt, Ht, Wr, Hr, shift):
    """D0-R (block + head of spec 55) at the game size: base copy and current source of rr_d0r.hlsl."""
    import struct
    from pathlib import Path
    from clean.rr_f.head.dx12 import run_dx12 as head
    from clean.rr_f.head.block import recurrent_shift
    params = dict(Wt=Wt, Ht=Ht, Wr=Wr, Hr=Hr, shift=recurrent_shift(shift),
                  constants={**wc.as_dict(), "ra": f16(0.75), "m": f16(0.5)}, g=f16(1.25), s1=1.25, s2=.75)
    Wh = {k: W[k] for k in ("Wq", "Wv", "Tb", "Wo", "bo", "W1", "b1", "W2", "b2")}
    x = bench.fast_e4m3(rng, (Ht, Wt, 32))
    hold, hc = bench.fast_hidden(rng, (Ht, Wt)), bench.fast_hidden(rng, (Ht // 2, Wt // 2))
    if a.data in ("adv", "advbig", "adv2", "adv2big"):
        adv = adversarial(rng, {"x": x, "hold": hold, "hc": hc}, pnan=1 / 600 if a.data.startswith("adv2") else 1.0)
        x, hold, hc = adv["x"], adv["hold"], adv["hc"]
    calls = {}
    for var in ("base", "new"):
        if a.only and var != a.only:
            continue
        ha, job = head.make_job(params, "stage", "prod", W=Wh, x=x, hold=hold, hc=hc)
        init = np.zeros(ha.size, np.uint8)
        for off, data in ha.init:
            init[off:off + len(data)] = np.frombuffer(data, np.uint8)
        base = ar.add(init)
        nargs = struct.unpack_from("<I", job, 8)[0]
        args = list(struct.unpack_from(f"<{nargs}I", job, 12))
        for i, nm in enumerate(head.ARGS):
            if nm.startswith("p_") and args[i]:
                args[i] += base
        src_dir = os.path.join(os.path.dirname(a.base), "head", "dx12") if var == "base" else head.HERE
        src = Path(src_dir) / "rr_d0r.hlsl"
        import hashlib
        key = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        dxil = str(Path(build.BUILD) / "dxil" / f"ab_d0r_{var}_{key}.dxil")
        if not os.path.exists(dxil):
            build.build_dx12.compile_hlsl(str(src), "rr_d0r_head", dxil, {"WG_BLOCK": 128, "WG_WAVE": 32},
                                         includes=(str(Path(head.CLEAN) / "dx12/kernels"),),
                                         asm_tool=build.build_dx12.build_asm_tool(build.BUILD))
        kid = var + "_D0R"
        kernels[kid] = (dxil, nargs)
        groups = ((Wt + params["shift"][0] + 7) // 8) * ((Ht + params["shift"][1] + 7) // 8)
        disp.append((kid, groups, args))
        sizes = {"cnet": Wt * Ht * 8, "hid": Wr * Hr * 4, "gates": Wr * Hr * 4, "prec": Wr * Hr * 8,
                 "candw": Wr * Hr * 4}
        calls[var] = _Out({k: (base + ha.ptr[k], (n,), "e4m3") for k, n in sizes.items()})
    return calls


def adversarial(rng, d, pnan=1.0):
    """Special values where a stage can receive them: e4m3 NaN codes (0x7f / 0xff), +-448, subnormal codes, +-0; f16
    hidden values +-inf, NaN, -0, subnormals, +-65504 (a few percent of the elements each)."""
    for k, v in d.items():
        r = rng.random(v.shape)
        if v.dtype == np.uint8:
            s = rng.integers(0, 2, v.shape, dtype=np.uint8) * 128
            v = np.where(r < .03 * pnan, s | 0x7f, v)
            v = np.where((r >= .03) & (r < .06), s | 0x7e, v)
            v = np.where((r >= .06) & (r < .09), s | rng.integers(1, 8, v.shape, dtype=np.uint8), v)
            v = np.where((r >= .09) & (r < .11), s, v).astype(np.uint8)
        else:
            sg = np.where(rng.random(v.shape) < .5, -1.0, 1.0)
            v = np.where(r < .02 * pnan, sg * np.inf, v)
            v = np.where((r >= .02) & (r < .02 + .02 * pnan), np.nan, v)
            v = np.where((r >= .04) & (r < .06), -0.0, v)
            v = np.where((r >= .06) & (r < .09), sg * rng.integers(1, 1024, v.shape) * 2.0 ** -24, v)
            v = np.where((r >= .09) & (r < .11), sg * 65504.0, v)
        d[k] = v
    return d


def make_inputs(rng, name, Nx, Ny, data):
    if data in ("adv", "advbig"):
        return adversarial(rng, bench.inputs_for(rng, name, Nx, Ny))
    if data in ("adv2", "adv2big"):        # sparse NaN / inf (1/600 of the rate of adv): most windows stay finite
        return adversarial(rng, bench.inputs_for(rng, name, Nx, Ny), pnan=1 / 600)
    if data == "synth":
        return synth.stage_inputs(rng, name, Nx, Ny)
    d = bench.inputs_for(rng, name, Nx, Ny)
    if data == "zero":
        return {k: np.zeros_like(v) for k, v in d.items()}
    if data == "same":    # one token vector everywhere (the saturating stand-in model's x0 looks like this)
        for k, v in d.items():
            d[k] = np.broadcast_to(v[:1, :1], v.shape).copy()
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="IN0,D0-F")
    ap.add_argument("--render", default="1706x960")
    ap.add_argument("--step", default="E")
    ap.add_argument("--z", default="f32")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--data", default="rand", choices=("rand", "synth", "same", "zero", "smallw", "adv", "bigw", "advbig", "adv2", "adv2big"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--only", choices=("new", "base"))
    ap.add_argument("--shift", default="4,4")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nolock", action="store_true")
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--model", action="store_true", help="weights / constants of the sequence's default model file")
    ap.add_argument("--warm", type=int, default=100, help="untimed-in-report copies of the first dispatch first")
    ap.add_argument("--census", action="store_true", help="print special-value counts of every output (NaN / +-448 "
                    "codes, f16 inf / NaN / subnormal): shows which paths adversarial data reaches")
    ap.add_argument("--variants", help="routes (gpu/stage.py ROUTES) timed as new variants of every stage, e.g. "
                                       "gen,fast0,fast1 (default: the stage's own route, 'new')")
    a = ap.parse_args(argv)
    variants = a.variants.split(",") if a.variants else ["new"]
    rw, rh = (int(v) for v in a.render.split("x"))
    Wt, Ht = -(-rw // 32) * 32, -(-rh // 32) * 32
    rng = np.random.default_rng(a.seed)
    wc = window_consts(load_manifest(), "A")
    shift = tuple(int(v) for v in a.shift.split(","))
    mdl = None
    if a.model:
        from clean.rr_f.gpu import sequence
        mdl = sequence.model_from_file(sequence.DEFAULT_MODEL, sequence.DEFAULT_MAP)
        wc = mdl.wc
    ar = build.Arena()
    kernels, disp, rows = {}, [], []
    t0 = time.time()
    names = a.stages.split(",")
    if a.only != "base":
        for v in variants:
            old_route = os.environ.get("WG_RR_ROUTE")
            if v != "new":
                os.environ["WG_RR_ROUTE"] = ",".join(f"{n}={v}" for n in names if n != "D0-R")
            try:
                stage.precompile([(n, a.z, a.step, 1) for n in names if n != "D0-R"])
            finally:
                if old_route is None:
                    os.environ.pop("WG_RR_ROUTE", None)
                else:
                    os.environ["WG_RR_ROUTE"] = old_route
    for name in names:
        lv = 0 if name == "D0-R" else stage_info(name)["level"]
        Nx, Ny = Wt >> lv, Ht >> lv
        W = synth.stage_weights(rng, "IN0" if name == "D0-R" else name)
        if a.data in ("zero", "smallw"):    # power probe: zero / tiny weight codes (the stand-in model's look)
            for k, v in W.items():
                if v.dtype == np.uint8:
                    W[k] = np.zeros_like(v) if a.data == "zero" else (v & 0x87)
        if a.data in ("bigw", "advbig", "adv2big"):     # large weight codes (2^7 .. 448): f32 / f16 overflow, Q8 saturation
            for k, v in W.items():
                if v.dtype == np.uint8:
                    W[k] = ((v & 0x80) | 0x70 | (v & 0x0e)).astype(np.uint8)
        consts = {"ra": f16(0.75), "m": f16(0.5), "f": f16(1.0), "g": f16(1.25)}
        if name == "D0-R":                         # the head kernel (clean/rr_f/head/dx12/rr_d0r.hlsl, production)
            rows.append((name, Nx, Ny, d0r_calls(a, ar, kernels, disp, rng, W, wc, Nx, Ny, rw, rh, shift)))
            continue
        if mdl is not None:
            W, consts = mdl.weights[name], mdl.constants[name]
        inp = make_inputs(rng, name, Nx, Ny, a.data)
        calls = {}
        for var in ["base"] + variants:
            if a.only and var != a.only and not (a.only == "new" and var != "base"):
                continue
            old_env = os.environ.get("WG_RR_GENERIC")
            if var == "base":                      # the base layouts (E1: the regrouped 128-channel input)
                os.environ["WG_RR_GENERIC"] = "1"
            try:
                sc = stage.StageCall(name, inp, W, consts, wc, shift, arena=ar, step=a.step)   # E1 layout of the step
            finally:
                if old_env is None:
                    os.environ.pop("WG_RR_GENERIC", None)
                else:
                    os.environ["WG_RR_GENERIC"] = old_env
            kid = var + "_" + name.replace("-", "")
            if var == "base":
                kernels[kid] = (base_kernel(a.base, name, a.z, a.step), len(sc.args))
                if stage_info(name)["adapter"] and stage.e1_direct() and not a.only:
                    # the base needs the regroup dispatch: time it (from a level-0 copy) as part of the base
                    lv0 = ar.add(stage.ungroup_children(inp["in"]))
                    kernels["base_regroup"] = (build.compile_kernel("rr_body_helpers.hlsl", "rr_regroup_e1",
                                                                    {"WG_BLOCK": 256}), 4)
                    disp.append(("base_regroup", (2 * Nx * 2 * Ny * 2 + 255) // 256, [lv0, sc.args[0], 2 * Nx, 2 * Ny]))
                    rows.append(("regroup", Nx, Ny, {"base": None}))
                disp.append((kid, sc.groups(), sc.args))
            else:
                old_route = os.environ.get("WG_RR_ROUTE")
                if var != "new":
                    os.environ["WG_RR_ROUTE"] = f"{name}={var}"
                try:
                    dx, args, groups = stage.kernel_call(sc, a.z, a.step)
                finally:
                    if old_route is None:
                        os.environ.pop("WG_RR_ROUTE", None)
                    else:
                        os.environ["WG_RR_ROUTE"] = old_route
                kernels[kid] = (dx, len(args))
                disp.append((kid, groups, args))
            calls[var] = sc
        rows.append((name, Nx, Ny, calls))
    if a.warm:
        disp = [disp[0]] * a.warm + disp
    print(f"render {rw}x{rh} -> level-0 tokens {Wt}x{Ht}; data {a.data}; step {a.step}; arena {ar.size / 1e6:.0f} MB, "
          f"prepared in {time.time() - t0:.0f} s", flush=True)
    init_img = ar.image()
    img, log = build.run(ar, kernels, disp, lock=not a.nolock, reps=a.reps,
                         workdir=os.path.join(build.BUILD, "runs", a.tag))
    if os.environ.get("WG_RR_DBGCNT"):           # debug counters of RR_DBG_CNT builds (arena dwords 4..6)
        print("debug counters", img[16:28].view(np.uint32).tolist(), flush=True)
    times = {}
    for line in log.splitlines():
        if line.startswith("time "):
            p = line.split()
            times[int(p[1])] = (float(p[4]), float(p[7]))
    ok = True
    di = a.warm
    tot = {v: 0.0 for v in ["base"] + variants}
    print(f"{'stage':6s} {'variant':7s} {'best ms':>8s} {'mean ms':>8s}  outputs")
    for name, Nx, Ny, calls in rows:
        for var, sc in calls.items():
            b, m = times.get(di, (float("nan"), float("nan")))
            if sc is None:                          # the base's regroup dispatch
                tot[var] += b
                di += 1
                print(f"{name:6s} {var:7s} {b:8.3f} {m:8.3f}  (regroup dispatch of the base)", flush=True)
                continue
            tot[var] += b
            di += 1
            cmp = ""
            if var != "base" and "base" in calls:
                parts = []
                for k, (off, shape, typ) in sc.out.items():
                    boff = calls["base"].out[k][0]
                    nb = int(np.prod(shape)) * (1 if typ == "e4m3" else 2)
                    d = int(np.count_nonzero(img[off:off + nb] != img[boff:boff + nb]))
                    ok &= d == 0
                    parts.append(f"{k}:{d}/{nb}")
                cmp = "bytes differing " + " ".join(parts)
                if a.census:
                    cs = []
                    for k, (off, shape, typ) in sc.out.items():
                        nb = int(np.prod(shape)) * (1 if typ == "e4m3" else 2)
                        if typ == "e4m3":
                            cb = img[off:off + nb] & 0x7f
                            cs.append(f"{k}[nan {int(np.count_nonzero(cb == 0x7f))} max {int(np.count_nonzero(cb == 0x7e))}]")
                        else:
                            h = img[off:off + nb].view("<u2") & 0x7fff
                            cs.append(f"{k}[nan {int(np.count_nonzero(h > 0x7c00))} inf {int(np.count_nonzero(h == 0x7c00))} "
                                      f"sub {int(np.count_nonzero((h > 0) & (h < 0x400)))}]")
                    cmp += "  census " + " ".join(cs)
            print(f"{name:6s} {var:7s} {b:8.3f} {m:8.3f}  {cmp}", flush=True)
    # bytes no kernel may write: everything outside the output buffers keeps its initial value
    keep = np.ones(len(init_img), bool)
    for _, _, _, calls in rows:
        for sc in calls.values():
            for off, shape, typ in (sc.out.values() if sc is not None else ()):
                keep[off:off + int(np.prod(shape)) * (1 if typ == "e4m3" else 2)] = False
            if sc is not None and a.step[0] in "YZI":   # the fixup marks / count of steps with a fixup dispatch
                keep[sc.pYflag:sc.pYflag + (sc.groups() + 1) * 4] = False
    n = min(len(init_img), len(img))
    stray = int(np.count_nonzero((img[:n] != init_img[:n]) & keep[:n]))
    ok &= stray == 0
    print(f"bytes outside the output buffers changed: {stray}")
    print("total  " + "  ".join(f"{v} {t:.3f}" for v, t in tot.items()) + f"  ({'IDENTICAL' if ok else 'DIFFERENT'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
