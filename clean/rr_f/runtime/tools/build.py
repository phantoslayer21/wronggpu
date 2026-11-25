# SPDX-License-Identifier: MIT
"""Build WrongGPU's RR runtime: the kernel library (DXIL, exact arithmetic mode of clean/dx12/build_dx12.py), the
model-independent tables, the runtime library wronggpu_rr_runtime.dll, the D3D12 test host rr_test_host.exe;
optionally a stage folder with the runtime and a model file (the proxy's directory layout).

    python -m clean.rr_f.runtime.tools.build [--out DIR] [--stage DIR (--model-seed N | --model-admission |
                                                                       --model-wgcm FILE)]

Default output: <WG_BUILD_ROOT>\\rr-f-rt-int\\out (WG_BUILD_ROOT defaults to <repo>\\build; WG_RR_BUILD is set to <out>\\kbuild for the kernel
cache of clean/rr_f/gpu/build.py). CPU only (no GPU use).

The kernels are the merged speed branch's (clean/rr-f-speed1): exposure (RR_EX_SRV: the colour read from the game's
resource), input stage (decode, replay-list clear, certified fast features and the exact replay of the listed tokens
with RR_IN_SRV: the game's inputs read through SRVs, front-end MLP step X with the exact-sum fast path), the body stages
(step X, exact row sums; the struct routes of gpu/stage.py: rr_block_c32 / fast / wide, E1 reading IN0's y2 directly),
the D0-R head (exact step on integer WMMA + weight planes + strict fixup of marked windows, exact row sums: the
strict head's bits; head.strict: the binary64 step alone, WG_RR_HEAD=strict; head.fast: the default step, the runtime default
WG_RR_HEAD=fast), the output stage
(fast path prep / fast / fallback on the binary16 state images; output.main / pool with RR_RT kept for WG_RR_PATHS=
verified), the PH2 pool through exact tables (pool.tables / pool.main; resolve.pool kept), the resolve (resolve.fast +
resolve.fixup with the exp2 table of resolve.init; resolve.main kept), all with exact elementary functions, and the
runtime's own passes (rr_runtime.hlsl). Every fast kernel gives the same bits as the kernel it replaces.

clean/rr-f-s10-modes: ONE library holds the kernel sets of all three modes (rr_mode = exact | accurate | fast; the runtime
picks the set per instance, src/rr_runtime.cpp). The default (accurate) kernels are the primary set (stage.*, head.fast,
output.*, resolve.fast*); the alternatives sit beside them under their own names: head.fast.rx (relaxed head, RR_HVARL 7),
stageex.* (exact body step E, the front-end MLP not folded), output.{prep,fast,fallback,prepu}.f (OF_HWRCP +
OF_REPROJ_MAX), resolve.fast.x / resolve.fast2.x (correctly rounded exp2 / reciprocal). Each is built exactly as the
separate builds were (WG_RR_EZMODE=exact WG_RR_HEAD_DEFS={} WG_RR_RF_DEFINES={} / WG_RR_HEAD_DEFS={"RR_HVARL":7}
WG_RR_OUT_DEFINES={...}), so the mode's kernels are the same DXIL bytes (checked against kernels.json of those builds)."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.dirname(HERE)
RRF = os.path.dirname(RT)
ROOT = os.path.dirname(os.path.dirname(RRF))
DEFAULT_OUT = os.path.join(os.environ.get("WG_BUILD_ROOT", os.path.join(ROOT, "build")), "rr-f-rt-int", "out")

EXACT_OUT_DEFINES = {"OFX_FUSE": 1, "OF_UNIF": 1}   # exact, same bits (output_diff PASS): fused 16-tap loop + uniform scratch words
FAST_OUT_DEFINES = {"OF_HWRCP": 1, "OF_REPROJ_MAX": 1,   # Fast mode's output stage (clean/rr-f-s10-modes; tier B, NOT exact)
                    "OF_UNIF": 1, "OFX_FUSE": 1, "OFX_NOCHK": 1}   # fused 16-tap loop + uniform scratch + no subnormal-band check (fast mode only)
# clean/rr-f-wave4 R4.3 (same bits as without; rr_output_fast.hlsli, written arguments there): extra defines of output.fast.f
# alone (output.prep.f / output.fallback.f keep their DXIL): OFX_PH1W (the PH1 pool inside each wave64: no groupshared
# copy, no barrier), OFX_LD3 (the motion pair / eroded texels as one b64 / b32 load), OFX_QST (two neighbour Q texels as
# one 12-byte store). WG_RR_OUTF_FASTDEFS (JSON object, build time) replaces them; the runtime kernel output.fast.f0 is
# the same kernel without them (WG_RR_W4 bit 2 = 0 selects it).
FAST_OUT_FAST_ONLY = {"OFX_PH1W": 1, "OFX_LD3": 1, "OFX_QST": 1}
# clean/rr-f-wave4c W4C.3: output.fast.f_ph1w = output.fast.f0 plus OFX_PH1W alone
FAST_OUT_PH1W_ONLY = {"OFX_PH1W": 1}


class _env:
    """Temporarily set environment variables (restored on exit)."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


STAGES = ("IN0", "E1", "E2", "E3", "E4", "E5", "D4-F", "D4-R", "D3-F", "D3-R", "D2-F", "D2-R", "D1-F", "D1-R", "D0-F")


# clean/rr-f-s9-kernels: also the body kernels and their helpers (never included by the head; edits to them no longer
# invalidate the head kernels' cache keys)
OWN_KEY_SKIP = ("rr_input.hlsl", "rr_input_mlp.hlsl", "rr_block.hlsl", "rr_block_c32.hlsl", "rr_block_fast.hlsl",
                "rr_block_wide.hlsl", "rr_fast.hlsli", "rr_mlp_step.hlsli", "rr_dflow.hlsli",   # (s12-dflow)
                # clean/rr-f-wave4b: the wave64 forms (the head's own file is never included by the other head kernels;
                # listing them keeps every existing head kernel's cache key)
                "rr_d0r_w64.hlsl", "rr_block_c64_w64.hlsl",
                # clean/rr-f-wave4c: the wave64 form of the wide stages (never included by the head)
                "rr_block_wide_w64.hlsl",
                "rr_in_math.hlsli",   # a14-frame (request of track C): included by rr_input.hlsl only
                # clean/rr-f-s10-tail: the output / resolve / pool / exposure kernels are never included by the head
                "rr_output.hlsl", "rr_output_fast.hlsli", "rr_output_probe.hlsl", "rr_resolve.hlsl", "rr_resolve_fast.hlsl",
                "rr_resolve_fast_main.hlsli", "rr_resolve_fast_math.hlsli", "rr_pool.hlsl", "rr_exposure.hlsl")


def kernels(out):
    """[(name, nargs, dxil path)] of every kernel the runtime uses."""
    os.environ["WG_RR_BUILD"] = os.path.join(out, "kbuild")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clean.rr_f.gpu import build, stage
    import build_dx12 as bd
    asm = bd.build_asm_tool(build.BUILD)
    from clean.rr_f.gpu import input_stage as GI, output as GO, pool as GP, resolve_fast as GRF, exposure as GX
    # step E's row sums: exact (default) or WG_RR_EZMODE=f32 (opt-in tier-B image comparisons); its other opt-in
    # variants come from WG_RR_EDEFS (gpu/stage.py)
    # clean/rr-f-speed8 (owner's rule 2026-10-23: network arithmetic may be relaxed, history logic stays exact): the
    # default body is step E with f32 row sums and the relaxed elementwise maths (RR_VARL + RR_B32; quality gate in
    # results/2026-10-23-speed8.md). Exact network body: WG_RR_EZMODE=exact (WG_RR_EDEFS then defaults to none).
    ez = os.environ.get("WG_RR_EZMODE", "f32")
    if "WG_RR_EDEFS" not in os.environ:
        os.environ["WG_RR_EDEFS"] = "RR_VARL=1,RR_B32=1" if ez == "f32" else ""
    # clean/rr-f-s9-kernels: with the relaxed body (RR_B32) the front-end MLP (variant E, same bits as input.mlp.e) is folded
    # into stagee.IN0 (rr_block_c32.hlsl RR_INFOLD; the runtime then skips input.mlp.e). Off: WG_RR_INFOLD=0. Opt-in (NOT
    # variant E, network arithmetic relaxed like the body): RR_FOLD_B32=1 in WG_RR_EDEFS (folded) or
    # WG_RR_MLP_DEFS={"RR_MLP_B32": 1} (separate dispatch). The exact body (WG_RR_EZMODE=exact) keeps the fold off.
    b32 = "RR_B32=1" in os.environ["WG_RR_EDEFS"]
    os.environ.setdefault("WG_RR_INFOLD", "1" if b32 else "0")
    stage.precompile([(n, "exact", s, 1) for s in ("X", "Z", "XF") for n in STAGES] + [(n, ez, "E", 1) for n in STAGES])
    res = [("stage." + n, 37, stage.kernel_for(n, "exact", step="X", ldsw=1)) for n in STAGES]
    # the exact body step Z (bit-identical to X; rr_block.hlsl, E1 on the regrouped input) and its step-X fixup:
    # 37 arguments + the per-window marks, the start-value bound and the class words (gpu/stage.py StageCall.args)
    res += [("stagez." + n, 41, stage.kernel_for(n, "exact", step="Z", ldsw=1)) for n in STAGES]
    res += [("stagefix." + n, 41, stage.kernel_for(n, "exact", step="XF", ldsw=1)) for n in STAGES]
    # the default step E on the struct routes (opt-in WG_RR_BODY=E; E1 reads IN0's y2 directly, as for step X)
    assert stage.e1_direct(1, "E") == stage.e1_direct(1, "X")
    res += [("stagee." + n, 37, stage.kernel_for(n, ez, step="E", ldsw=1)) for n in STAGES]
    # clean/rr-f-wave4 R4.2b: stagee.E1 / D1-F / D0-F share the exact body's LDS overlay (rr_block_fast.hlsl / rr_block_c32.hlsl
    # RR_OVL); stagee0.* = the same kernels without it (RR_OVL 0, the kernels of fc16c9d; runtime WG_RR_W4 bit 1 = 0)
    res += [("stagee0." + n, 37, stage.kernel_for(n, ez, step="E", ldsw=1, extra={"RR_OVL": 0})) for n in ("E1", "D1-F", "D0-F")]
    # clean/rr-f-wave4b R4b.2: the level-1 stages in wave64 (rr_block_c64_w64.hlsl, design D; the same bits as stagee.E1 /
    # D1-F / D1-R; runtime WG_RR_W4B bits 3 / 2 / 1). Relaxed body only (the exact body keeps wave32).
    if b32:
        res += [("stagee." + n + ".w64", 37, stage.kernel_c64_w64(n, ez, step="E", ldsw=1)) for n in stage.W4B_C64_STAGES]
    else:   # (a non-relaxed library: the names stay, with stagee.<n>'s own DXIL)
        res += [("stagee." + n + ".w64", 37, stage.kernel_for(n, ez, step="E", ldsw=1)) for n in stage.W4B_C64_STAGES]
    # clean/rr-f-wave4c W4C.1: the wide stages (E2, D2-F, D2-R, E3, D3-F, D3-R, D4-F, D4-R) in wave64 (rr_block_wide_w64.hlsl, the
    # same bits as stagee.<n>; runtime WG_RR_W4C bits 0 / 1 / 2 = level 2 / level 3 / level 4 decoder). E4 / E5 keep wave32.
    if b32:
        res += [("stagee." + n + ".w64", 37, stage.kernel_wide_w64(n, ez, step="E", ldsw=1)) for n in stage.W4C_WIDE_STAGES]
    else:
        res += [("stagee." + n + ".w64", 37, stage.kernel_for(n, ez, step="E", ldsw=1)) for n in stage.W4C_WIDE_STAGES]
    # clean/rr-f-wave4c W4C.2: stagee.IN0.w4c / stagee.D0-F.w4c = stagee.IN0 / D0-F (rr_block_c32.hlsl wave64) with the deferred Q8 decision
    # (RR_DEFER), the Dd / Wo weight prefetch and, for IN0, the interleaved front-end MLP (stage.W4C_C32_DEFS); same bits. Runtime
    # WG_RR_W4C bits 3 / 4. The non-relaxed library repeats stagee.<n>'s DXIL under the names.
    for n in ("IN0", "D0-F"):
        res.append(("stagee." + n + ".w4c", 37, stage.kernel_for(n, ez, step="E", ldsw=1, extra=stage.W4C_C32_DEFS[n]) if b32
                    else stage.kernel_for(n, ez, step="E", ldsw=1)))
    # clean/rr-f-s12-dflow: staged.* = stagee.* in the dataflow form (rr_dflow.hlsli: row counters instead of the UAV
    # barriers; opt-in WG_RR_DFLOW=1). Only the rr_block_wide.hlsl stages have that form; the others repeat stagee.*'s
    # DXIL (kRrDflowMask: the stages that have it)
    global DFMASK
    DFMASK = 0
    import concurrent.futures as _cf
    # clean/rr-f-a14-frame E4: the level-0/1 stages (rr_block_c32 / rr_block_fast, A1b hooks) too; WG_RR_DFL01=0 at build
    # time: the wide stages only (as before)
    _dsrc = stage.DFLOW_SRCS if os.environ.get("WG_RR_DFL01", "1") == "1" else ("rr_block_wide.hlsl",)
    _dfw = [n for n in STAGES if stage.kernel_src(n, 1, "E") in _dsrc]
    with _cf.ThreadPoolExecutor(int(os.environ.get("WG_RR_JOBS", "8"))) as _ex:
        list(_ex.map(lambda n: stage.kernel_for(n, ez, step="E", ldsw=1, dflow=1), _dfw))
    for i, n in enumerate(STAGES):
        if n in _dfw:
            DFMASK |= 1 << i
            res.append(("staged." + n, 42, stage.kernel_for(n, ez, step="E", ldsw=1, dflow=1)))
        else:
            res.append(("staged." + n, 37, stage.kernel_for(n, ez, step="E", ldsw=1)))
    # clean/rr-f-s10-modes: the exact body (mode exact) = the "acc" build's stagee.*: exact elementwise maths (no RR_VARL /
    # RR_B32), the front-end MLP not folded into IN0 (the runtime then dispatches input.mlp.e). Built under its own
    # environment (stage.py reads WG_RR_EDEFS / WG_RR_INFOLD when it compiles).
    # clean/rr-f-lw-fold: stageex.IN0 folds the front-end MLP too (rr_block_c32.hlsl RR_INFOLD 1 without RR_B32: the
    # variant-E MLP of input.mlp.e, the same operations per token, so x0 and everything after keep their bits); the
    # runtime then skips input.mlp.e in mode exact as well (kRrInFoldX). WG_RR_EXFOLD=0: the separate dispatch.
    global EXFOLD
    EXFOLD = 1 if os.environ.get("WG_RR_EXFOLD", "1") == "1" and stage.route("IN0", 1, "E") == "c32" else 0
    with _env(WG_RR_EDEFS=""):
        stage.precompile([(n, "exact", "E", 1) for n in STAGES])
        res += [("stageex." + n, 37, stage.kernel_for(n, "exact", step="E", ldsw=1, fold=EXFOLD if n == "IN0" else 0))
                for n in STAGES]
        # clean/rr-f-s13x-x-flow: stagedx.* = stageex.* in the dataflow form (rr_dflow.hlsli has no arithmetic: the same
        # bits as stageex.*, only the start times change). Only the rr_block_wide.hlsl stages of the exact body have
        # that form; the others repeat stageex.*'s DXIL (kRrDflowMaskX)
        global DFMASKX
        DFMASKX = 0
        _dfx = [n for n in STAGES if stage.kernel_src(n, 1, "E") == "rr_block_wide.hlsl"]
        with _cf.ThreadPoolExecutor(int(os.environ.get("WG_RR_JOBS", "8"))) as _ex:
            list(_ex.map(lambda n: stage.kernel_for(n, "exact", step="E", ldsw=1, dflow=1), _dfx))
        for i, n in enumerate(STAGES):
            if n in _dfx:
                DFMASKX |= 1 << i
                res.append(("stagedx." + n, 42, stage.kernel_for(n, "exact", step="E", ldsw=1, dflow=1)))
            else:
                res.append(("stagedx." + n, 37, stage.kernel_for(n, "exact", step="E", ldsw=1,
                                                                 fold=EXFOLD if n == "IN0" else 0)))
    ck = build.compile_kernel
    res.append(("regroup", 4, ck("rr_body_helpers.hlsl", "rr_regroup_e1", {"WG_BLOCK": 256})))
    for n in ("partial", "finish"):
        res.append((f"exposure.{n}", 23, ck("rr_exposure.hlsl", f"rr_exposure_{n}", {"WG_BLOCK": GX.EX_BLOCK[n], "RR_EX_SRV": 1},
                                            tag=f"rr_exposure_{n}_srv")))
    res.append(("input.decode", 1, ck("rr_input.hlsl", "rr_input_decode", {"WG_BLOCK": 256})))
    res.append(("input.clear", 1, ck("rr_input.hlsl", "rr_input_mark_clear", {"WG_BLOCK": 64})))
    fb = GI.FEAT_BLOCK
    # clean/rr-mp1-input: I1 RR_IN_KCB 1 (the constant block through scalar loads of root SRV 6, which the runtime rebinds
    # to the frame's input block; same words) in input.features, input.features.lite and input.replay; I2 RR_IN_FZT 1 (the
    # fast pass's zero-result tests as r == 0.0f; same output bits, written proof) in input.features
    res.append(("input.features", 1, ck("rr_input.hlsl", "rr_input_features",
                                        dict({"WG_BLOCK": fb, "RR_IN_EXACT": 0, "RR_IN_MARK_ALL": 0, "RR_IN_LDST": 0,
                                              "RR_IN_SRV": 1, "RR_IN_SRVC": 1, "RR_IN_KCB": 1, "RR_IN_FZT": 1},
                                             **json.loads(os.environ.get("WG_RR_FEAT_DEFS", "{}"))),
                                        tag="rr_input_features_fast_srv")))   # WG_RR_FEAT_DEFS: timing builds only
    # opt-in near-exact features (WG_RR_INPUT=lite; tier B, not the default)
    # clean/rr-f-speed8: the runtime default (WG_RR_INPUT=exact selects input.features), built as "lite" plus the hardware
    # reciprocal (RR_IN_FAST 31); WG_RR_LITE_VARIANT=lite gives round 7's lite kernel
    # clean/rr-f-s9-inx: the log2 / exp2 tables are read from memory (RR_IN_LDST 0) in every features kernel: the same
    # table words, without the 4.6 KB of LDS per workgroup that capped the features pass at 9 waves / SIMD (now 13)
    res.append(("input.features.lite", 1, GI.features_kernel(os.environ.get("WG_RR_LITE_VARIANT",
                                                                            "lite+RR_IN_FAST=31+RR_IN_LDST=0+RR_IN_KCB=1"),
                                                             srv=1)))
    # clean/rr-f-s9-inx: the split input (history samples exact through the fast pass + replay, the non-history colour /
    # guide features with input lite's maths; rr_input.hlsl RR_IN_NHL)
    res.append(("input.features.split", 1, ck("rr_input.hlsl", "rr_input_features",
                                              dict({"WG_BLOCK": fb, "RR_IN_EXACT": 0, "RR_IN_MARK_ALL": 0, "RR_IN_LDST": 0,
                                                    "RR_IN_SRV": 1, "RR_IN_SRVC": 1, "RR_IN_NHL": 1},
                                                   **json.loads(os.environ.get("WG_RR_FEAT_DEFS", "{}"))),
                                              tag="rr_input_features_split_srv")))
    res.append(("input.replay", 1, ck("rr_input.hlsl", "rr_input_features",
                                      dict({"WG_BLOCK": fb, "RR_IN_EXACT": 1, "RR_IN_REPLAY": 1, "RR_IN_LDST": 0,
                                            "RR_IN_SRV": 1, "RR_IN_RSPREAD": 1, "RR_IN_KCB": 1},
                                           **json.loads(os.environ.get("WG_RR_REPLAY_DEFS", "{}"))),
                                      tag="rr_input_features_replay_srv")))   # WG_RR_REPLAY_DEFS: experiments
    res.append(("input.mlp", 13, ck("rr_input_mlp.hlsl", "rr_input_mlp", {"WG_BLOCK": 128, "RR_STEP": 3,
                                                                          "RR_STEP3_MUT": 0}, tag="rr_input_mlp_X")))
    # opt-in WG_RR_INPUT_MLP=E (clean/rr-f-speed5): the front-end MLP on variant E (RR_STEP 0; gpu/input_stage.py step E)
    # clean/rr-f-s9-kernels: WG_RR_MLP_DEFS (JSON object) = extra defines of input.mlp.e, e.g. {"RR_MLP_B32": 1} (opt-in:
    # the front-end MLP with the body's relaxed network arithmetic; rr_input_mlp.hlsl)
    mld = json.loads(os.environ.get("WG_RR_MLP_DEFS", "{}"))
    res.append(("input.mlp.e", 13, ck("rr_input_mlp.hlsl", "rr_input_mlp", dict({"WG_BLOCK": 128, "RR_STEP": 0,
                                                                                 "RR_STEP3_MUT": 0}, **mld),
                                      tag="rr_input_mlp_E" + "".join(f"_{k}{v}" for k, v in sorted(mld.items())))))
    for n in ("main", "pool"):
        res.append((f"output.{n}", GO.NARGS, ck("rr_output.hlsl", f"rr_output_{n}", {"WG_BLOCK": 32, "RR_RT": 1},
                                                tag=f"rr_output_{n}_rt")))
    for n, (dxil, nargs) in GO.kernel_set(defines=EXACT_OUT_DEFINES, fast=True, names=GO.FAST_KERNELS + ("prepu",), spec=True).items():   # prep, fast, fallback, prepu
        res.append((f"output.{n}", nargs, dxil))
    # clean/rr-f-s10-modes: Fast mode's output stage (OF_HWRCP + OF_REPROJ_MAX + OFX_FUSE + OFX_NOCHK; tier B, NOT exact), same four kernels
    for n, (dxil, nargs) in GO.kernel_set(defines=FAST_OUT_DEFINES, fast=True, names=GO.FAST_KERNELS + ("prepu",), spec=True,
                                          fast_only=json.loads(os.environ.get("WG_RR_OUTF_FASTDEFS", json.dumps(FAST_OUT_FAST_ONLY)))).items():
        res.append((f"output.{n}.f", nargs, dxil))
    # clean/rr-f-wave4: output.fast.f before R4.3 (A/B kernel of the runtime's WG_RR_W4 mask)
    res.append(("output.fast.f0", GO.NARGS, GO.kernel_set(defines=FAST_OUT_DEFINES, fast=True, names=("fast",), spec=True)["fast"][0]))
    # clean/rr-f-wave4c W4C.3 (same bits; rr_output_fast.hlsli): output.fast.f with OFX_PH1W alone (the PH1 pool inside each
    # wave64, no groupshared tile copy: LDS 10,304 -> 7,200 bytes) without OFX_LD3 / OFX_QST, which Wave 4a mixed into the same
    # kernel. Runtime WG_RR_W4C bit 5 (when WG_RR_W4 bit 2 is off).
    res.append(("output.fast.f_ph1w", GO.NARGS, GO.kernel_set(defines=FAST_OUT_DEFINES, fast=True, names=("fast",), spec=True,
                                                              fast_only=json.loads(os.environ.get("WG_RR_OUTF_PH1W_DEFS", json.dumps(FAST_OUT_PH1W_ONLY))))["fast"][0]))
    for n in ("main", "pool"):
        res.append((f"resolve.{n}", 3, ck("rr_resolve.hlsl", f"rr_resolve_{n}", {"WG_BLOCK": 64, "RR_ELEM_EXACT": 1,
                                                                                "RR_MUT": 0},
                                          tag=f"rr_resolve_{n}_e1m0")))
    # the fast resolve reading every format (RF_SPEC 0: the game's configurations differ from the sequence's standard
    # one, for which RF_SPEC 1 is specialised), its fixup and the exp2 table of resolve.init
    # clean/rr-f-r7-spec: the fast resolve takes the hardware exp2 and reciprocal (RF_HW, RF_HWRCP; erratum 51-58 and
    # 54b "within 1 ulp", FTZ kept; 8 captured frames >= 115 dB, at most 2 pixels differ by 1/255). The correctly
    # rounded build: WG_RR_RF_DEFINES={} (any JSON object replaces these defines). clean/rr-f-s11-tail: only rr_mode=fast
    # runs these two (owner 2026-10-28: accurate and exact use the correctly rounded resolve.fast.x / resolve.fast2.x).
    rfd =json.loads(os.environ.get("WG_RR_RF_DEFINES", '{"RF_HW": 1, "RF_HWRCP": 1}'))
    res.append(("resolve.fast", GRF.NARGS, GRF.fast_kernel(spec=0, extra=rfd)))
    res.append(("resolve.fixup", GRF.NARGS, GRF.fixup_kernel()))
    res.append(("resolve.init", GRF.NARGS, GRF.init_kernel()))
    # the fast resolve specialised for the game's configuration (RF_SPEC 2: extended + layer, no alpha upscaling)
    res.append(("resolve.fast2", GRF.NARGS, GRF.fast_kernel(spec=2, extra=rfd)))
    # clean/rr-f-s10-modes: mode exact's resolve (correctly rounded exp2 / reciprocal: WG_RR_RF_DEFINES={})
    res.append(("resolve.fast.x", GRF.NARGS, GRF.fast_kernel(spec=0, extra={})))
    res.append(("resolve.fast2.x", GRF.NARGS, GRF.fast_kernel(spec=2, extra={})))
    # clean/rr-f-300 R2: the plain configuration (RF_SPEC 3: not extended, no layer / opacity / alpha upscaling; same bits as
    # resolve.fast / resolve.fast.x for those dispatches), hardware exp2 / reciprocal for Fast (rfd), correctly rounded for the .x
    res.append(("resolve.fast3", GRF.NARGS, GRF.fast_kernel(spec=3, extra=rfd)))
    res.append(("resolve.fast3.x", GRF.NARGS, GRF.fast_kernel(spec=3, extra={})))
    for n, (dxil, nargs) in GP.kernels(exact=True).items():       # pool-tables, pool-main (exact tables)
        res.append((n.replace("-", "."), nargs, dxil))

    def own(src, entry, defines, includes, tag):
        h = hashlib.sha256()
        for d in [os.path.dirname(src), bd.HLSL_DIR] + list(includes):
            for f in sorted(os.listdir(d)):
                # the input-stage kernels are never included by these kernels: not part of their cache key
                if f.endswith((".hlsl", ".hlsli")) and f not in OWN_KEY_SKIP:
                    with open(os.path.join(d, f), "rb") as fh:
                        h.update(f.encode() + fh.read())
        h.update(json.dumps([os.path.basename(src), entry, sorted(defines.items())]).encode())
        path = os.path.join(build.BUILD, "dxil", f"{tag}_{h.hexdigest()[:16]}.dxil")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            bd.compile_hlsl(src, entry, path, defines, includes=tuple(includes), asm_tool=asm)
        return path

    head = os.path.join(RRF, "head", "dx12", "rr_d0r.hlsl")
    # rr_f/kernels too: the head includes rr_step_i.hlsli / rr_exact_math.hlsli from there (part of the cache key)
    hinc = [os.path.join(ROOT, "clean", "dx12", "kernels"), os.path.join(RRF, "kernels")]
    hz = {"WG_BLOCK": 128, "WG_WAVE": 32, "RR_ZEXACT": 1}
    # default: the exact step on integer WMMA (RR_STRICT_STEP 4) with its weight planes and the strict fixup of the
    # windows it marks (same bits as the strict head, clean/rr_f/head/dx12/exact_head.py)
    res.append(("head", 40, own(head, "rr_d0r_head", dict(hz, RR_STRICT_STEP=4), hinc, "rr_d0r_head_int_zexact")))
    res.append(("head.wpl", 40, own(head, "rr_d0r_wplanes", dict(hz, RR_STRICT_STEP=4), hinc, "rr_d0r_head_wplanes")))
    res.append(("head.fixup", 40, own(head, "rr_d0r_head", dict(hz, RR_STRICT_STEP=1, RR_FIXUP=1), hinc,
                                      "rr_d0r_head_strict_fixup")))
    res.append(("head.strict", 40, own(head, "rr_d0r_head", dict(hz, RR_STRICT_STEP=1), hinc,
                                       "rr_d0r_head_strict_zexact")))
    res.append(("head.debug", 40, own(head, "rr_d0r_head", dict(hz, RR_STRICT_STEP=1, RR_DEBUG=1), hinc,
                                      "rr_d0r_head_strict_zexact_debug")))
    # head.fast also writes the output stage's render records (RR_OUT_PREP; clean/rr-f-r6-tail), 8 more arguments
    # opt-in WG_RR_HEAD_DEFS (JSON object, clean/rr-f-r8-head): extra defines of head.fast, e.g. {"RR_HVARL": 1} = the
    # head with the body's relaxed arithmetic (tier B, NOT exact; rr_d0r.hlsl RR_HVARL). clean/rr-f-speed8: RR_HVARL 7 is
    # the default; WG_RR_HEAD_DEFS={} builds the exact fast head (same bits as head.exact)
    # clean/rr-f-s9-merged (owner's rule 2026-10-23, superseded): head.fast is the EXACT head (same bits as head.exact),
    # now Exact mode's head; WG_RR_HEAD_DEFS={"RR_HVARL": 7} builds it relaxed. Policy since 2026-11-01 (confirmed
    # 2026-11-04): the head is exact in Exact mode; Accurate and Fast run the gate-checked relaxed head.fast.rx below.
    hfd = dict(hz, RR_OUT_PREP=1, **json.loads(os.environ.get("WG_RR_HEAD_DEFS", '{}')))
    res.append(("head.fast", 48, own(head, "rr_d0r_head", hfd, hinc, "rr_d0r_head_zexact_oprep")))
    # clean/rr-f-s10-modes: Fast mode's relaxed head (RR_HVARL 7; tier B, NOT exact) beside the exact one
    # clean/rr-f-lw-fold: the relaxed head keeps RR_S9 61 (without bit 64, the reading-specific CANDW / HID chains: measured
    # slower in head.fast.rx, +0.002 / +0.009 ms at 1080p / 1440p Q; the same DXIL as before). WG_RR_HEADRX_DEFS (JSON
    # object, A/B builds only) = extra defines of head.fast.rx
    # clean/rr-mp1-head (owner decision 2026-11-01): accurate mode runs head.fast.rx too (registry mode row, gate-checked);
    # head.fast (exact) remains mode exact's fast head.
    # clean/rr-f-wave4 R4.1a (same bits as without; rr_d0r.hlsl): RR_HSTM 1 (each record whole from the hf0 lane: b128 / b64 /
    # b32 stores) and RR_HQ8S 1 (step 1's q / v^T tiles quantised per head: 99 -> 95 VGPRs, under the 96 cliff, 0 spills).
    # head.fast.rx0 = head.fast.rx of fc16c9d (runtime WG_RR_W4 bit 0 = 0). WG_RR_HEADRX_DEFS replaces nothing: it adds to both
    hrx = dict(hz, RR_OUT_PREP=1, RR_HVARL=7, RR_S9=61, **json.loads(os.environ.get("WG_RR_HEADRX_DEFS", "{}")))
    res.append(("head.fast.rx", 48, own(head, "rr_d0r_head", dict(hrx, RR_HSTM=1, RR_HQ8S=1,
                                                                  **json.loads(os.environ.get("WG_RR_HEADRX_W4_DEFS", "{}"))),
                                        hinc, "rr_d0r_head_zexact_oprep")))
    res.append(("head.fast.rx0", 48, own(head, "rr_d0r_head", hrx, hinc, "rr_d0r_head_zexact_oprep")))
    # clean/rr-f-wave4b R4b.1 (same bytes as head.fast.rx; head/dx12/rr_d0r_w64.hlsl, written arguments there): the relaxed head
    # in wave64 (design D of rr_block_c32.hlsl: one wave64 = two token tiles of the window, the weight fragments shared).
    # Runtime WG_RR_W4B bit 0; head.fast.rx stays as it is (bit 0 = 0 selects it).
    res.append(("head.fast.rx.w64", 48, own(os.path.join(RRF, "head", "dx12", "rr_d0r_w64.hlsl"), "rr_d0r_head",
                                            dict(hrx, WG_WAVE=64, RR_HSTM=1, RR_HQ8S=1,
                                                 **json.loads(os.environ.get("WG_RR_HEADRX_W4_DEFS", "{}"))),
                                            hinc, "rr_d0r_head_w64_zexact_oprep")))
    rk = os.path.join(RT, "kernels", "rr_runtime.hlsl")
    for name, entry, blk in (("rt.ingest", "rr_rt_ingest", 64), ("rt.fill", "rr_rt_fill", 64),
                             ("rt.store", "rr_rt_store", 64), ("rt.selftest", "rr_rt_selftest", 32),
                             ("rt.pass", "rr_rt_pass", 64)):
        res.append((name, 24, own(rk, entry, {"WG_BLOCK": blk, "WG_WAVE": 32}, [], entry)))
    # clean/rr-f-a14-frame E24: the read-only cache warm-up pass (WG_RR_TOUCH at creation; not dispatched by default)
    res.append(("rt.touch", 24, own(os.path.join(RT, "kernels", "rr_touch.hlsl"), "rr_rt_touch", {"WG_BLOCK": 256, "WG_WAVE": 32}, [], "rr_rt_touch")))
    res.append(("rt.blkcopy", 24, own(os.path.join(RT, "kernels", "rr_touch.hlsl"), "rr_rt_blkcopy", {"WG_BLOCK": 256, "WG_WAVE": 32}, [], "rr_rt_blkcopy")))
    # clean/rr-f-wave5b (fast mode only; runtime WG_RR_W5, default 0x7f; NOT exact, network/tail arithmetic relaxed, history logic exact):
    #   bit 0 head.fast.rx.f.w64 (RR_HFAST_ALU=2: bounded f32 ALU head tail, per-head Q8 reducing VGPR 96->84, branchless Q8, no dead CANDW)
    #   bit 1 output.fast.w5 (OFX_W5_ALU=1 on output.fast.f0 base: native f16, FMA bilinear, 16-tap half exp2, 3-cmp max, no fallback pass)
    #   bit 2 resolve.fast2.w5 / resolve.fast3.w5 (RFX_W5_ALU=1: f32 uniforms, uint2 half4 taps, FP16 3x3 clamp before E, no fixup pass)
    #   bit 3 input.features.w5 (RR_IN_W5_ALU=1, wave64, 0 B LDS, vectorized tex loads, DLANE + MDD)
    #   bits 4..6 stagef.* (gpu/stage.py W5_KERNELS: pure f32 RR_B32=1 + branchless clamped RR_Q8_FAST=1, zero narrow-EXEC WMMAs).
    # Appended after every wave-4c kernel; all 172 wave-4c kernels above keep bit-identical cache keys and DXIL.
    res.append(("head.fast.rx.f.w64", 48, own(os.path.join(RRF, "head", "dx12", "rr_d0r_w64.hlsl"), "rr_d0r_head",
                                              dict(hrx, WG_WAVE=64, RR_HSTM=1, RR_HQ8S=1, RR_HFAST_ALU=2,
                                                   **json.loads(os.environ.get("WG_RR_HEADRX_W4_DEFS", "{}"))),
                                              hinc, "rr_d0r_head_w64_f_zexact_oprep")))
    res.append(("output.fast.w5", GO.NARGS,
                GO.kernel_set(defines=dict(FAST_OUT_DEFINES, OFX_W5_ALU=1), fast=True, names=("fast",), spec=True)["fast"][0]))
    res.append(("resolve.fast2.w5", GRF.NARGS, GRF.fast_kernel(spec=2, extra=dict(rfd, RFX_W5_ALU=1))))
    res.append(("resolve.fast3.w5", GRF.NARGS, GRF.fast_kernel(spec=3, extra=dict(rfd, RFX_W5_ALU=1))))
    res.append(("input.features.w5", 1,
                GI.features_kernel("lite+RR_IN_FAST=31+RR_IN_LDST=0+RR_IN_KCB=1+RR_IN_W5_ALU=1+RR_IN_M24=1+RR_IN_DLANE=1+RR_IN_MDD=1+WG_WAVE=64",
                                   srv=1)))
    for name, st, kind, defs in stage.W5_KERNELS:
        # (a non-relaxed library repeats stagee.<n>'s DXIL under the names, as the wave-4 kernels)
        res.append((name, 37, stage.kernel_w5(st, ez, step="E", ldsw=1) if b32 else stage.kernel_for(st, ez, step="E", ldsw=1)))
    return res


DFMASK = 0   # set by kernels(): the stages whose staged.* kernel has the dataflow form (clean/rr-f-s12-dflow)
EXFOLD = 0   # set by kernels(): stageex.IN0 built with the folded front-end MLP (clean/rr-f-lw-fold)
DFMASKX = 0  # set by kernels(): the stages whose stagedx.* kernel has the dataflow form (clean/rr-f-s13x-x-flow)


def embed(ks, path, e1_direct=False, infold=0):
    lines = ['// generated by clean/rr_f/runtime/tools/build.py: DXIL of the runtime\'s kernels', '#include "rr_kernels.h"', '',
             '// stage.E1 reads IN0\'s y2 directly (gpu/stage.py e1_direct, RR_ADAPT 2): no regroup dispatch',
             f'const int kRrE1Direct = {1 if e1_direct else 0};',
             '// stagee.IN0 computes the front-end MLP itself from the codes (gpu/stage.py infold, rr_block_c32.hlsl RR_INFOLD)',
             f'const int kRrInFold = {1 if infold else 0};',
             '// stageex.IN0 (the body of mode exact) computes the front-end MLP itself too (clean/rr-f-lw-fold, same bits)',
             f'const int kRrInFoldX = {1 if EXFOLD else 0};',
             '// staged.<stage> has the dataflow form (rr_dflow.hlsli) for the stages of this mask (clean/rr-f-s12-dflow)',
             f'const unsigned kRrDflowMask = {DFMASK}u;',
             '// stagedx.<stage> (the exact body) has the dataflow form for the stages of this mask (clean/rr-f-s13x-x-flow)',
             f'const unsigned kRrDflowMaskX = {DFMASKX}u;',
             '']
    table = []
    seen = {}   # clean/rr-f-s12-dflow: identical DXIL embedded once (staged.* of the non-wide stages repeat stagee.*)
    for i, (name, nargs, dxil) in enumerate(ks):
        data = open(dxil, "rb").read()
        if data not in seen:
            body = ",".join(str(b) for b in data)
            lines.append(f"static const unsigned char k{i}[] = {{{body}}};")
            seen[data] = f"k{i}"
        table.append(f'  {{"{name}", {nargs}u, {seen[data]}, {len(data)}u}},  // {os.path.basename(dxil)}')
    lines.append("const RrKernelBlob kRrKernels[] = {")
    lines += table
    lines.append("};")
    lines.append(f"const uint32_t kRrKernelCount = {len(ks)}u;")
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")


def tables(path):
    """The model-independent tables (src/rr_tables.h) from the clean reference: the binary16 sigmoid of 51.4 and the
    head's sigmoid / tanh tables, exactly the arrays tools/pack_model.py packs."""
    if os.path.exists(path):
        return path
    import numpy as np
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from clean.rr_f.gpu import layout
    from clean.rr_f.head.dx12 import run_dx12 as head
    from clean.rr_f.reference.scalar import sig_tables
    sig = np.asarray(layout.f16_bytes(sig_tables()[0]), "<u2")
    assert sig.size == 65536
    tab = np.frombuffer(head.tables(np.float32(1.0))[0], "<u2")
    lines = ["// generated by clean/rr_f/runtime/tools/build.py from the clean reference (reference/scalar.py "
             "sig_tables, head/dx12/run_dx12.py tables)", '#include "rr_tables.h"', "namespace wgrr_rt {"]
    lines.append("const uint16_t kRrSigTable[65536] = {" + ",".join(str(int(v)) for v in sig) + "};")
    lines.append(f"const uint16_t kRrHeadTab[{tab.size}] = {{" + ",".join(str(int(v)) for v in tab) + "};")
    lines.append(f"const uint32_t kRrHeadTabCount = {tab.size}u;")
    lines.append("}  // namespace wgrr_rt")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


def cl(cmd, what):
    import build_dx12 as bd
    full = f'"{bd.VSDEVCMD}" -arch=x64 -host_arch=x64 >nul && {cmd}'
    r = subprocess.run(full, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"building {what} failed:\n{r.stdout}{r.stderr}")
    return r.stdout


def build(out):
    os.makedirs(out, exist_ok=True)
    ks = kernels(out)
    from clean.rr_f.gpu import stage
    gen = os.path.join(out, "gen")
    os.makedirs(gen, exist_ok=True)
    emb = os.path.join(gen, "rr_kernels_embedded.cpp")
    embed(ks, emb, e1_direct=stage.e1_direct(1, "X"), infold=stage.infold("E", 1))
    tab = tables(os.path.join(gen, "rr_tables_embedded.cpp"))
    src = os.path.join(RT, "src")
    obj = os.path.join(out, "obj")
    os.makedirs(obj, exist_ok=True)
    dll = os.path.join(out, "wronggpu_rr_runtime.dll")
    files = " ".join(f'"{os.path.join(src, f)}"' for f in ("rr_runtime.cpp", "rr_model.cpp", "rr_pack.cpp", "rr_modes.cpp"))
    cl(f'cl /nologo /Brepro /O2 /EHsc /std:c++17 /fp:strict /W3 /MT /LD /I"{src}" {files} "{emb}" "{tab}" '
       f'/Fo"{obj}\\\\" /Fe"{dll}" /link /BREPRO d3d12.lib dxgi.lib bcrypt.lib gdi32.lib', "runtime")
    host = os.path.join(out, "rr_test_host.exe")
    hsrc = os.path.join(RT, "tests", "rr_test_host.cpp")
    hobj = os.path.join(out, "hobj")
    os.makedirs(hobj, exist_ok=True)
    cl(f'cl /nologo /O2 /EHsc /std:c++17 /fp:strict /W3 /MT /I"{src}" "{hsrc}" /Fo"{hobj}\\\\" /Fe"{host}" '
       f'/link d3d12.lib dxgi.lib', "test host")
    manifest = {name: dict(nargs=n, dxil=os.path.basename(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
                for name, n, p in ks}
    with open(os.path.join(out, "kernels.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return dll, host


def stage_folder(out, stage, seed=None, admission=False, wgcm=None):
    """A folder as the proxy's directory would hold it: the runtime and its model file (the converter's model.wgcm,
    or the runtime's own container of a synthetic model); model_logical.pkl for the tests' CPU reference."""
    from clean.rr_f.runtime.tools import pack_model
    os.makedirs(stage, exist_ok=True)
    shutil.copyfile(os.path.join(out, "wronggpu_rr_runtime.dll"), os.path.join(stage, "wronggpu_rr_runtime.dll"))
    for n in ("model.wgcm", "wronggpu_rr_model.bin", "wronggpu-rr.log"):
        p = os.path.join(stage, n)
        if os.path.exists(p):
            os.remove(p)
    if wgcm:
        shutil.copyfile(wgcm, os.path.join(stage, "model.wgcm"))
        pack_model.main(["--wgcm", wgcm, "--logical-out", os.path.join(stage, "model_logical.pkl")])
    else:
        argv = ["--admission"] if admission else ["--synthetic", str(seed)]
        argv += ["--out", os.path.join(stage, "wronggpu_rr_model.bin"), "--logical-out",
                 os.path.join(stage, "model_logical.pkl")]
        pack_model.main(argv)
    return stage


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--stage")
    ap.add_argument("--model-seed", type=int, default=5100)
    ap.add_argument("--model-admission", action="store_true")
    ap.add_argument("--model-wgcm")
    a = ap.parse_args(argv)
    dll, host = build(a.out)
    print("runtime", dll, hashlib.sha256(open(dll, "rb").read()).hexdigest())
    print("host", host)
    if a.stage:
        stage_folder(a.out, a.stage, a.model_seed, a.model_admission, a.model_wgcm)
        print("stage", a.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
