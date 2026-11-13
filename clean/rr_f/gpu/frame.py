# SPDX-License-Identifier: MIT
"""Connected preset-F frames on one D3D12 arena: exposure (page 58) -> input stage (page 52: features, front-end
MLP) -> the sixteen-stage body with the E1 regroup -> the D0-R head (pages 51, 55). The body's IN0 reads the input
stage's x0 and state-stream seed straight from the GPU (no prepared tokens are uploaded); the input stage reads E,
Ep and X3 from the exposure pass's result buffer. With --frames 2 a second frame follows in the same submission:
its input stage reads the first frame's head HID and the first frame's PM[1-par] as its PM[par], and writes into
the first frame's PM[par] / Pe / Pn images (parity toggled, counter advanced, pj and CP of the first frame, 53.2).

    python -m clean.rr_f.gpu.frame --json <report> [--size 29x19] [--frames 1|2] [--mutation input|exposure|history]

CPU references (exposure, input stage, body, head) are evaluated with the values the GPU actually produced upstream
(exposure result, previous head HID), for comparison only. Exit 0 only when every exposure result equals its
reference, every input stage is bit exact, every body/head passes network.analyze's rules and every binding is the
intended producer. A mutation must give exit 1 (a numerical failure), not a crash."""
import argparse
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from . import build, input_stage as GI, network as GN
from . import exposure as GX
from ..reference import exposure as RX
from ..reference import input_stage as IS
from ..reference import network as RN
from ..tests import synth, synth_input as SI


def model(seed, size, b2=False):
    rng = np.random.default_rng(seed)
    frame, params, S, IN0, c_in0, wc = SI.make_case(rng, Wr=size[0], Hr=size[1], smv="Hd", b2=b2,
                                                    guides=["B", "G1", "Q"])
    weights = {name: synth.stage_weights(rng, "IN0" if name == "D0-R" else name) for name in RN.ORDER}
    weights["IN0"] = {n: IN0[n] for n in ("Wq", "Wv", "Tb", "Wo", "bo", "W1", "b1", "W2", "b2", "Wh", "bh")}
    constants = {name: synth.stage_consts(rng) for name in RN.ORDER}
    constants["IN0"] = dict(c_in0)
    return rng, frame, params, S, IN0, wc, weights, constants


def shifts_of(params):
    sh = np.asarray(params.SH)[params.par]
    shifts = {name: (int(sh[i][0]), int(sh[i][1])) for i, name in enumerate(RN.ENCODERS)}
    shifts.update({f"D{k}": (int(sh[6 + 4 - k][0]), int(sh[6 + 4 - k][1])) for k in range(5)})
    return shifts


def exposure_call(arena, frame, params, Eprev):
    """The two exposure dispatches on the colour input (its origin, the render size)."""
    k = GX.symbols()
    C = frame.textures["C"]
    c = np.ascontiguousarray(C.data, np.float32)
    w, h = params.render_size
    blocks = (w * h + 255) // 256
    expected = RX.auto_exposure(c, symbols=k, render_size=(w, h), Eprev=Eprev, reset=params.reset,
                                origin=C.origin)
    cp, pp = arena.add(c), arena.reserve(blocks * 8)
    out = arena.reserve(100, fill=0xA5)
    ilo, ihi = struct.unpack("<II", struct.pack("<d", 1.0 / (w * h)))
    args = [cp, pp, out, c.shape[1], c.shape[2], *C.origin, w, h, blocks, ilo, ihi,
            int(np.float32(Eprev).view(np.uint32)), int(params.reset)]
    args += [int(getattr(k, f.name).view(np.uint32)) for f in fields(k)]
    kern = {f"exposure-{n}": (build.compile_kernel("rr_exposure.hlsl", "rr_exposure_" + n, {"WG_BLOCK": GX.EX_BLOCK[n]}), 23)
            for n in ("partial", "finish")}
    record = ("exposure", out, expected, GX.exact_mean(c, (w, h), C.origin, k), bool(params.reset), k)
    return kern, [("exposure-partial", blocks, args), ("exposure-finish", 1, args)], out, (expected, record)


def add_frame(arena, kernels, dispatches, frame, params, S, IN0, wc, weights, constants, Eprev, step, mutation,
              links=None):
    kx, dx, expo_off, (ex_ref, ex_record) = exposure_call(arena, frame, params, Eprev)
    kernels.update(kx)
    dispatches += dx
    params.E, params.Ep, params.X3 = ex_ref.E, ex_ref.Ep, ex_ref.X3
    expo_link = arena.reserve(64) if mutation == "exposure" else expo_off
    call = GI.InputCall(arena, frame, params, S, IN0, wc, exposure=expo_link, links=links)
    kernels.update(GI.kernels(step))
    dispatches += call.dispatches()
    ref_in = IS.input_stage(frame, params, S, IN0, constants["IN0"], wc)
    x0_link = arena.reserve(ref_in["x0"].size) if mutation == "input" else call.pX0
    shifts = shifts_of(params)
    data = (ref_in["x0"], ref_in["hold"], weights, constants, wc, shifts)
    _, _, _, calls, reference, base, ha, bindings, regroup = GN.prepare(
        step=step, data=data, render=params.render_size, arena=arena, kernels=kernels, dispatches=dispatches,
        in0_links={0: x0_link, 2: call.out["hold"][0]})
    in0_args = [b for b in bindings if b["stage"] == "IN0"][0]["input_offsets"]
    connected = in0_args[0] == call.pX0 and in0_args[2] == call.out["hold"][0] and expo_link == expo_off
    return dict(frame=frame, params=params, call=call, calls=calls, base=base, ha=ha, regroup=regroup,
                shifts=shifts, expo_off=expo_off, ex_ref=ex_ref, ex_record=ex_record, connected=connected,
                bindings=dict(x0=call.pX0, hold=call.out["hold"][0], in0_inputs=in0_args, exposure=expo_off,
                              exposure_read=expo_link, hid_read=int(call.words[GI.K["HID"]]),
                              pm_read=int(call.words[GI.K["PMR"]]), pm_write=int(call.words[GI.K["PMW"]])))


def analyze_frame(img, f, S, IN0, wc, weights, constants, prev_hid=None, overwritten=()):
    """Reference of one frame from the GPU's upstream values, then the comparisons."""
    params, frame, call = f["params"], f["frame"], f["call"]
    e = img[f["expo_off"]:f["expo_off"] + 16].copy().view(np.float32)
    ex = f["ex_ref"]
    equal = bool(np.array_equal(e[[0, 1, 3]].view(np.uint32), np.float32([ex.E, ex.Ep, ex.X3]).view(np.uint32)))
    # page 58's tolerance (E one binary16 step, invE / X3 from rcp / log2 within 1 ulp): gpu/exposure.qualify
    expo_rules = GX.qualify(img, [f["ex_record"]], f["ex_record"][5])[0]
    params.E, params.Ep, params.X3 = e[0], e[1], e[3]
    if prev_hid is not None:
        frame.HID = prev_hid
    ref_in = IS.input_stage(frame, params, S, IN0, constants["IN0"], wc)
    reference = RN.run_body(ref_in["x0"], ref_in["hold"], weights, constants, wc, f["shifts"], s1=1.25, s2=.75,
                            render_size=params.render_size)
    in_report = GI.compare(call.outputs(img), ref_in)
    for key in overwritten:                 # shared images rewritten by a later frame of the same submission
        in_report[key] = dict(in_report[key], diff=0, skipped="overwritten by the next frame; checked there")
    in_report["exact"] = all(v["diff"] == 0 for v in in_report.values() if isinstance(v, dict) and "diff" in v)
    body = GN.analyze(img, f["calls"], reference, f["base"], f["ha"], f["regroup"], render=params.render_size)
    passed = bool(in_report["exact"] and body["passed"] and f["connected"] and expo_rules["passed"])
    return dict(passed=passed, exposure=dict(E=float(e[0]), Ep=float(e[1]), invE=float(e[2]), X3=float(e[3]),
                                             equal_to_reference=equal, page58_rules=expo_rules["rules"],
                                             within_page58_tolerance=expo_rules["passed"]),
                input_stage=in_report, body=body, connected=f["connected"], bindings=f["bindings"],
                m2_src=ref_in["m2_src"], par=params.par, fc=params.fc, reset=params.reset), ref_in


def head_hid(img, f):
    wr, hr = f["params"].render_size
    off = f["base"] + f["ha"].ptr["hid"]
    return img[off:off + wr * hr * 4].copy().view(np.int8).reshape(hr, wr, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--size", default="29x19")
    ap.add_argument("--seed", type=int, default=5521)
    ap.add_argument("--frames", type=int, default=1, choices=(1, 2))
    ap.add_argument("--step", default="X", choices=("E", "X"))
    ap.add_argument("--b2", action="store_true")
    ap.add_argument("--mutation", choices=("input", "exposure", "history"))
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    size = tuple(int(v) for v in a.size.split("x"))
    rng, frame1, params1, S, IN0, wc, weights, constants = model(a.seed, size, a.b2)
    params1.reset = False
    arena, kernels, dispatches = build.Arena(), {}, []
    Eprev = np.float32(np.float16(rng.uniform(0.5, 2.0)))
    f1 = add_frame(arena, kernels, dispatches, frame1, params1, S, IN0, wc, weights, constants, Eprev, a.step,
                   None if a.frames == 2 and a.mutation == "history" else a.mutation)
    frames = [f1]
    if a.frames == 2:
        frame2, _, _, _, _, _ = SI.make_case(rng, Wr=size[0], Hr=size[1], Wo=params1.output_size[0],
                                             Ho=params1.output_size[1], smv="Hd", guides=["B", "G1", "Q"])
        params2 = replace(params1, par=1 - params1.par, fc=(params1.fc + 1) % 8, reset=False,
                          prev_jitter=params1.jitter,
                          jitter=(np.float32(rng.uniform(-.5, .5)), np.float32(rng.uniform(-.5, .5))),
                          V2Cp=params1.V2C, W2Vp=params1.W2V, V2C=SI.perturb(rng, params1.V2C, 1e-4),
                          W2V=params1.W2V.copy())
        # the second frame's temporal inputs are the first frame's GPU outputs
        frame2.PM_read = frame1.PM_write                      # shape only (linked)
        frame2.PM_write, frame2.Pe, frame2.Pn = frame1.PM_read, frame1.Pe, frame1.Pn
        frame2.HID = frame1.HID
        frame2.PH1, frame2.PH2 = frame1.PH1, frame1.PH2
        hid_link = arena.reserve(size[0] * size[1] * 4) if a.mutation == "history" else f1["base"] + f1["ha"].ptr["hid"]
        links = dict(HID=hid_link, PMR=f1["bindings"]["pm_write"], PMW=f1["bindings"]["pm_read"],
                     PE=f1["call"].out["PE"][0], PN=f1["call"].out["PN"][0])
        f2 = add_frame(arena, kernels, dispatches, frame2, params2, S, IN0, wc, weights, constants,
                       np.float32(f1["ex_ref"].E), a.step, None, links=links)
        f2["links"] = links
        frames.append(f2)
    print(f"Prepared {len(dispatches)} dispatches, {arena.size / 1e6:.1f} MB", flush=True)
    img, log = build.run(arena, kernels, dispatches, require_double=True,
                         workdir=str(Path(build.BUILD) / ("frame-" + (a.mutation or "normal"))))
    rows = []
    r1, ref1 = analyze_frame(img, f1, S, IN0, wc, weights, constants,
                             overwritten=("Pe", "Pn") if a.frames == 2 else ())
    rows.append(r1)
    if a.frames == 2:
        f2 = frames[1]
        frame2 = f2["frame"]
        # actual upstream values: frame 1's PM written by the GPU (exact) and its head's HID codes
        out1 = f1["call"].outputs(img)
        frame2.PM_read = out1["PMW"]
        # Pe / Pn priors: frame 1's values (its reference; the GPU buffers now hold frame 2's writes)
        frame2.PM_write, frame2.Pe, frame2.Pn = frame1.PM_read, ref1["surfaces"]["Pe"], ref1["surfaces"]["Pn"]
        r2, _ = analyze_frame(img, f2, S, IN0, wc, weights, constants, prev_hid=head_hid(img, f1))
        r2["temporal_links"] = dict(hid=f2["links"]["HID"] == f1["base"] + f1["ha"].ptr["hid"],
                                    pm=f2["links"]["PMR"] == f1["bindings"]["pm_write"])
        r2["passed"] = r2["passed"] and all(r2["temporal_links"].values())
        rows.append(r2)
    passed = all(r["passed"] for r in rows)
    report = dict(passed=passed, mutation=a.mutation, size=size, seed=a.seed, step=a.step, frames=rows,
                  dispatches=[d[0] for d in dispatches], host_log=log,
                  shaders={n: hashlib.sha256(Path(v[0]).read_bytes()).hexdigest() for n, v in kernels.items()},
                  scope="synthetic frames: exposure -> input stage -> body -> head in one arena/submission"
                        + ("; frame 2 reads frame 1's head HID and PM" if a.frames == 2 else "")
                        + "; no output stage, PH1/PH2 producers, player model or game")
    Path(a.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for i, r in enumerate(rows):
        print(f"frame {i + 1}: par {r['par']} fc {r['fc']} input", {k: v["diff"] for k, v in r["input_stage"].items()
                                                                 if isinstance(v, dict) and "diff" in v},
              "body passed:", r["body"]["passed"], "exposure equal:", r["exposure"]["equal_to_reference"],
              "within page 58:", r["exposure"]["within_page58_tolerance"],
              "connected:", r["connected"], r.get("temporal_links", ""), flush=True)
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
