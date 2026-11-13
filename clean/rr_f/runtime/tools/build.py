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

FAST_OUT_DEFINES = {"OF_HWRCP": 1, "OF_REPROJ_MAX": 1,   # Fast mode's output stage (clean/rr-f-s10-modes)
                    "OF_UNIF": 1}   # clean/rr-f-lw-fold: uniform scratch words (exact, same bits; faster in output.fast.f only)


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
    for n, (dxil, nargs) in GO.kernel_set(fast=True, names=GO.FAST_KERNELS + ("prepu",), spec=True).items():   # prep, fast, fallback, prepu
        res.append((f"output.{n}", nargs, dxil))
    # clean/rr-f-s10-modes: Fast mode's output stage (OF_HWRCP + OF_REPROJ_MAX; tier B, NOT exact), same four kernels
    for n, (dxil, nargs) in GO.kernel_set(defines=FAST_OUT_DEFINES, fast=True, names=GO.FAST_KERNELS + ("prepu",), spec=True).items():
        res.append((f"output.{n}.f", nargs, dxil))
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
    res.append(("head.fast.rx", 48, own(head, "rr_d0r_head", dict(hz, RR_OUT_PREP=1, RR_HVARL=7, RR_S9=61,
                                                                  **json.loads(os.environ.get("WG_RR_HEADRX_DEFS", "{}"))),
                                        hinc, "rr_d0r_head_zexact_oprep")))
    rk = os.path.join(RT, "kernels", "rr_runtime.hlsl")
    for name, entry, blk in (("rt.ingest", "rr_rt_ingest", 64), ("rt.fill", "rr_rt_fill", 64),
                             ("rt.store", "rr_rt_store", 64), ("rt.selftest", "rr_rt_selftest", 32),
                             ("rt.pass", "rr_rt_pass", 64)):
        res.append((name, 24, own(rk, entry, {"WG_BLOCK": blk, "WG_WAVE": 32}, [], entry)))
    # clean/rr-f-a14-frame E24: the read-only cache warm-up pass (WG_RR_TOUCH at creation; not dispatched by default)
    res.append(("rt.touch", 24, own(os.path.join(RT, "kernels", "rr_touch.hlsl"), "rr_rt_touch", {"WG_BLOCK": 256, "WG_WAVE": 32}, [], "rr_rt_touch")))
    res.append(("rt.blkcopy", 24, own(os.path.join(RT, "kernels", "rr_touch.hlsl"), "rr_rt_blkcopy", {"WG_BLOCK": 256, "WG_WAVE": 32}, [], "rr_rt_blkcopy")))
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
