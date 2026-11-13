# SPDX-License-Identifier: MIT
"""Synthetic 16-stage preset-F body on one D3D12 arena, including the independent D0-R head.

This is a stage-integration qualification, not the missing page-52 frontend,
page-54/54b output pipeline, temporal runtime, converter, or OptiScaler bridge.
Run with WG_RR_BUILD set to a private build directory under the clean root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from . import build, stage
from ..reference import network
from ..reference.stages import stage_info
from ..head import fp
from ..head.dx12 import run_dx12 as head
from ..head.block import recurrent_shift
from ..head.head import head_channels
from ..tests import synth


def fixture(seed):
    rng = np.random.default_rng(seed)
    wc, _ = synth.window_consts(rng, "A")
    weights = {name: synth.stage_weights(rng, "IN0" if name == "D0-R" else name) for name in network.ORDER}
    constants = {name: synth.stage_consts(rng) for name in network.ORDER}
    shifts = {name: (2, 6) for name in network.ENCODERS}
    shifts.update({f"D{k}": (0, 2) for k in range(5)})
    tokens = synth.e4m3(rng, (32, 32, 32), .1)
    state0 = rng.uniform(0, 1, (32, 32, 4)).astype(np.float16).astype(np.float64)
    return tokens, state0, weights, constants, wc, shifts


def unbound_input(array):
    """Allocate only shape information for a GPU-produced input, never oracle data."""
    return np.full_like(array, 0x7f if array.dtype == np.uint8 else np.nan)


def prepare(seed=551, step="X", mutation=None, *, data=None, render=(29, 19), arena=None, kernels=None,
            dispatches=None, in0_links=None):
    """data: optional (tokens, state0, weights, constants, wc, shifts) in place of the synthetic fixture;
    in0_links: optional {arg index: arena offset} binding IN0's x (0) and hold (2) to earlier GPU outputs (the
    input stage), in which case no IN0 input payload is uploaded."""
    tokens, state0, weights, constants, wc, shifts = fixture(seed) if data is None else data
    reference = network.run_body(tokens, state0, weights, constants, wc, shifts, s1=1.25, s2=.75, render_size=render)
    refs = reference["stages"]
    arena = build.Arena() if arena is None else arena
    kernels = {} if kernels is None else kernels
    dispatches = [] if dispatches is None else dispatches
    calls, bindings = {}, []
    ht, wt = tokens.shape[:2]
    stage.precompile([(name, "exact", step, 1) for name in network.ORDER[:-1]])
    regroup = build.compile_kernel("rr_body_helpers.hlsl", "rr_regroup_e1", {"WG_BLOCK": 256})
    kernels["regroup"] = (regroup, 4)
    regroup_offset = None
    for name in network.ORDER[:-1]:
        info = stage_info(name)
        level = info["level"]
        links = {}
        if name == "IN0":
            inputs = {"x": tokens, "hold": state0}
            shift = shifts[name]
        elif name.startswith("E"):
            parent = "IN0" if level == 1 else f"E{level-1}"
            inputs = {"in" if level == 1 else "x": network.regroup_children(refs[parent]["y2"]) if level == 1 else refs[parent]["merged"],
                      "hold": refs[parent]["hmerged"]}
            links[2] = calls[parent].out["hmerged"][0]
            if level == 1 and stage.e1_direct(step=step):
                links[0] = calls[parent].out["y2"][0]          # E1 reads IN0's y2 directly (no regroup)
            elif level == 1:
                regroup_offset = arena.reserve(tokens.size, fill=0xEE)
                dispatches.append(("regroup", (tokens.size // 16 + 255) // 256,
                                   [calls[parent].out["y2"][0], regroup_offset, wt, ht]))
                links[0] = regroup_offset
            else:
                links[0] = calls[parent].out["merged"][0]
            shift = shifts[name]
        elif name.endswith("-F"):
            parent = "E5" if level == 4 else f"D{level+1}-R"
            enc = "IN0" if level == 0 else f"E{level}"
            inputs = {"cin": refs[parent]["y2"], "skip": refs[enc]["y2"]}
            links = {0: calls[enc].out["y2"][0], 1: calls[parent].out["y2"][0]}
            shift = shifts[f"D{level}"]
        else:
            parent = "E5" if level == 4 else f"D{level+1}-R"
            fusion, enc = f"D{level}-F", f"E{level}"
            inputs = {"x": refs[fusion]["y2"], "hold": refs[enc]["hnew"], "Hc": refs[parent]["hnew"]}
            links = {0: calls[fusion].out["y2"][0], 2: calls[enc].out["hnew"][0], 3: calls[parent].out["hnew"][0]}
            shift = shifts[f"D{level}"]
        if name != "IN0" or in0_links:
            inputs = {key: unbound_input(value) for key, value in inputs.items()}
        if name == "IN0" and in0_links:
            links.update(in0_links)
        call = stage.StageCall(name, inputs, weights[name], constants[name], wc, shift, arena=arena, step=step)
        for index, pointer in links.items():
            if mutation == "encoder" and name == "E3" and index == 0:
                continue
            call.args[index] = pointer
        calls[name] = call
        kernels[name] = (stage.kernel_for(name, "exact", step=step, ldsw=1), len(call.args))
        dispatches.append((name, call.groups(), call.args))
        bindings.append(dict(stage=name, input_offsets=call.args[:4], outputs={key: value[0] for key, value in call.out.items()}))

    c = constants["D0-R"]
    params = dict(Wt=wt, Ht=ht, Wr=render[0], Hr=render[1], shift=recurrent_shift(shifts["D0"]),
                  constants={**wc.as_dict(), "ra": c["ra"], "m": c["m"]}, g=c["g"], s1=1.25, s2=.75)
    ha, job = head.make_job(params, "stage", "dbg", W=weights["D0-R"], x=unbound_input(refs["D0-F"]["y2"]),
                            hold=unbound_input(refs["IN0"]["hnew"]), hc=unbound_input(refs["D1-R"]["hnew"]))
    init = np.zeros(ha.size, np.uint8)
    for offset, data in ha.init:
        init[offset:offset+len(data)] = np.frombuffer(data, np.uint8)
    base = arena.add(init)
    nargs = struct.unpack_from("<I", job, 8)[0]
    args = list(struct.unpack_from(f"<{nargs}I", job, 12))
    for index, name in enumerate(head.ARGS):
        if name.startswith("p_") and args[index]:
            args[index] += base
    args[0] = arena.reserve(tokens.size) if mutation == "head" else calls["D0-F"].out["y2"][0]
    args[4] = calls["IN0"].out["hnew"][0]
    args[5] = calls["D1-R"].out["hnew"][0]
    dxil = str(Path(build.BUILD) / "dxil" / "body_d0r_debug.dxil")
    build.build_dx12.compile_hlsl(str(Path(head.HERE) / "rr_d0r.hlsl"), "rr_d0r_head", dxil,
                                 {"WG_BLOCK": 128, "WG_WAVE": 32, "RR_DEBUG": 1},
                                 includes=(str(Path(head.CLEAN) / "dx12/kernels"),), asm_tool=build.build_dx12.build_asm_tool(build.BUILD))
    kernels["D0-R"] = (dxil, nargs)
    groups = ((params["Wt"] + params["shift"][0] + 7) // 8) * ((params["Ht"] + params["shift"][1] + 7) // 8)
    dispatches.append(("D0-R", groups, args))
    bindings.append(dict(stage="D0-R", input_offsets=[args[0], args[4], args[5]], outputs={key: base+value for key,value in ha.ptr.items() if key in ("cnet","hid","gates","prec","candw")}))
    return arena, kernels, dispatches, calls, reference, base, ha, bindings, regroup_offset


def analyze(img, calls, reference, base, ha, regroup_offset, render=(29, 19)):
    rows = {name: {key: stage.compare(key, value, reference["stages"][name][key]) for key,value in call.outputs(img).items()}
            for name,call in calls.items()}
    wr, hr = render
    ht, wt = reference["stages"]["IN0"]["y2"].shape[:2]
    regroup_bad = 0                                      # no regroup dispatch when E1 reads IN0's y2 directly
    if regroup_offset is not None:
        regroup = img[regroup_offset:regroup_offset+wt*ht*32].reshape(ht//2,wt//2,128)
        regroup_bad = int(np.count_nonzero(regroup != network.regroup_children(calls["IN0"].outputs(img)["y2"])))
    def read(name, shape, dtype):
        n = int(np.prod(shape))*np.dtype(dtype).itemsize
        off = base+ha.ptr[name]
        return img[off:off+n].copy().view(dtype).reshape(shape)
    values = read("dimg", (3,hr,wr,4), "<f2").astype(np.float64)
    actual = dict(Cnet=read("cnet",(ht,wt,4),"<f2").astype(np.float64), PREC=read("prec",(hr,wr,4),"<f2").astype(np.float64),
                  HID=values[0],GATES=values[1],CANDW=values[2])
    head_rows = {name:head.summarize_steps(value,reference["head"][name],3 if name=="CANDW" else 1) for name,value in actual.items()}
    code_bad = {}
    for name,key,dtype,encode in (("HID","hid",np.int8,fp.snorm8_store),("GATES","gates",np.uint8,fp.unorm8_store),("CANDW","candw",np.uint8,fp.unorm8_store)):
        code_bad[name] = int(np.count_nonzero(read(key,(hr,wr,4),dtype) != encode(actual[name])))
    guards = {name:bool(np.all(img[base+offset+total-head.GUARD:base+offset+total]==head.SENT)) for name,offset,total in ha.reads}
    cb=actual["CANDW"]
    bits=cb.astype(np.float16).view(np.uint16)
    # Check exact head rules against the kernel's own pre-head values, separately
    # from the end-to-end comparison with the independent CPU reference.
    y=read("dy",(ht,wt,32),"<f2").astype(np.float64)[:hr,:wr]
    if len(head_channels()["candw"])==4:     # 55-ERRATUM-2: four logits y10 .. y13, no pairs
        z=np.stack([y[...,n] for n in head_channels()["candw"]],-1)
        pair_bad=0
        eq=np.isfinite(z).all(-1) & (z==z[...,:1]).all(-1)
        equal_bad=int(np.count_nonzero(eq & ~np.all(cb==.25,axis=-1)))
        special=np.isnan(z).any(-1) | (z==np.inf).any(-1) | (z==-np.inf).all(-1)
        special_bad=int(np.count_nonzero(special & ~np.all(np.isnan(cb),axis=-1)))
    else:
        pair_bad=int(np.count_nonzero((bits[...,0]!=bits[...,1]) | (bits[...,2]!=bits[...,3])))
        ca,cb_=head_channels()["candw"]          # page 55 erratum: y10, y12
        eq=np.isfinite(y[...,ca]) & (y[...,ca]==y[...,cb_])
        equal_bad=int(np.count_nonzero(eq & ~np.all(cb==.25,axis=-1)))
        with np.errstate(all="ignore"):
            delta=(y[...,ca]-y[...,cb_]).astype(np.float32)
        special=np.isinf(delta) | np.isnan(delta)
        special_bad=int(np.count_nonzero(
            (np.isposinf(delta) & ~((cb[...,0]==.5) & (cb[...,2]==0))) |
            (np.isneginf(delta) & ~((cb[...,0]==0) & (cb[...,2]==.5))) |
            (np.isnan(delta) & ~np.all(np.isnan(cb),axis=-1))))
    passed = not regroup_bad and all(value["diff"]==0 for row in rows.values() for value in row.values()) and all(value["over_tol"]==0 for value in head_rows.values()) and not any(code_bad.values()) and all(guards.values()) and not (pair_bad or equal_bad or special_bad)
    return dict(passed=passed,stages=rows,head=head_rows,head_store_mismatches=code_bad,head_guards=guards,candw_pair_mismatches=pair_bad,
                candw_equal_logits=int(np.count_nonzero(eq)),candw_equal_logits_bad=equal_bad,
                candw_special_logits=int(np.count_nonzero(special)),candw_special_logits_bad=special_bad,regroup_mismatches=regroup_bad)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed",type=int,default=551)
    ap.add_argument("--step",choices=("E","X"),default="X")
    ap.add_argument("--mutation",nargs="?",const="head",choices=("head","encoder"),help="break the D0-R or E3 input connection; qualification must fail")
    ap.add_argument("--json",required=True)
    a=ap.parse_args()
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private output directory")
    ar,kernels,dispatches,calls,ref,base,ha,bindings,regroup=prepare(a.seed,a.step,a.mutation)
    print(f"Prepared {len(dispatches)} dispatches, {ar.size} bytes",flush=True)
    img,log=build.run(ar,kernels,dispatches,workdir=str(Path(build.BUILD)/(f"body-mutant-{a.mutation}" if a.mutation else "body")))
    report=analyze(img,calls,ref,base,ha,regroup)
    report.update(seed=a.seed,step=a.step,mutation=a.mutation,dispatches=[d[0] for d in dispatches],bindings=bindings,
                  shaders={name:hashlib.sha256(Path(k[0]).read_bytes()).hexdigest() for name,k in kernels.items()},host_log=log,
                  scope="one prepared synthetic frame; no input/output frontend or temporal runtime")
    Path(a.json).write_text(json.dumps(report,indent=2))
    print("PASS" if report["passed"] else "FAIL",flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
