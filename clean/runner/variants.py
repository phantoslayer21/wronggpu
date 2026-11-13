# SPDX-License-Identifier: MIT
"""Build variants of the Swin kernels (compile-time knobs of kernels/swin.hip, and of the second design
kernels/swin_v2.hip: variants named "v2*") and their launch geometry.

A variant is a dict of -D macros. `geometry()` derives what the runner needs to launch it. The same variants can be
timed on the GPU with sweep.py: the variants produce bit-identical outputs (except those in INEXACT, which reorder FP32
sums), they differ only in speed.
"""

# macro -> default (must match the defaults at the top of kernels/swin.hip)
SWIN_KERNELS = {  # block kernel -> (threads macro, windows-per-wave macro)
    "swin_block_c64": ("WG_THREADS_C64", "WG_WPW_C64"),
    "swin_block_c64_add": ("WG_THREADS_C64ADD", "WG_WPW_C64"),
    "swin_block_c96": ("WG_THREADS_C96", "WG_WPW_C96"),
    "swin_block_c96_add": ("WG_THREADS_C96ADD", "WG_WPW_C96"),
    "swin_block_c128": ("WG_THREADS_C128", "WG_WPW_C128"),
}
DEFAULTS = {"WG_WPW_C64": 1, "WG_WPW_C96": 1, "WG_WPW_C128": 1, "WG_GEMM_THREADS": 256, "WG_NT": 2,
            "WG_THREADS_C64": 256, "WG_THREADS_C64ADD": 256, "WG_THREADS_C96": 256, "WG_THREADS_C96ADD": 256,
            "WG_THREADS_C128": 256, "WG_MINW_C64": 1, "WG_MINW_C96": 5, "WG_MINW_C128": 1, "WG_MLP_GROUP": 2, "WG_FORCE_CLAMP": 0, "WG_FUSE": 0, "WG_NORM_DOT4": 0, "WG_PROFILE": 0, "WG_AV8": 0, "WG_SMAX16": 0, "WG_GELU_MODE": 2, "WG_GELU_MIX": 1, "WG_MLP_PF": 0, "WG_UNIT_BYTES": 16384, "WG_SAT_MINMAX": 1, "WG_NUMV_C96": 0}


def all_threads(n):
    """Every block kernel with n lanes per workgroup (including the dec1 block, whose default is 512)."""
    return {m: n for m, _ in SWIN_KERNELS.values()}


VARIANTS = {
    "base": {},
    # hard VGPR cap 192 on the C = 96 block kernels (enc2, dec2): 8 waves per SIMD instead of 6, with some spills
    "v192": {"WG_NUMV_C96": 192, "WG_MINW_C96": 8},
    "satmm": {"WG_SAT_MINMAX": 1},                         # NaN-propagating one-instruction clamp in wg_sat448
    "t128": all_threads(128),                              # 4-wave block workgroups
    "t512": all_threads(512),                              # 16 waves per workgroup: half the weight traffic
    "c64w2": {"WG_WPW_C64": 2},                            # two C=64 windows per wave
    "nt1": {"WG_NT": 1},                                   # merge / expand: fewer registers per wave
    "nt3": {"WG_NT": 3},
    "g128": {"WG_GEMM_THREADS": 128},                      # merge / expand: smaller workgroups
    # pair: two hidden tiles per MLP step, their W1 chains interleaved (bit-identical; measured: no gain).
    "pair": {"WG_MLP_GROUP": 2},
    # Exact A/B: the clamp-free packs (taken when the weights prove them safe, see packing.block_bounds) against the
    # always-clamped packs. Bit-identical by construction.
    "forceclamp": {"WG_FORCE_CLAMP": 1},
    # Occupancy floors (amdgpu_waves_per_eu): the register file allows 4 / 5 / 6 / 7 / 8 waves per SIMD at <= 256 / 192 / 160
    # / 144 / 128 VGPRs. Natural register use: C=64 159 (166 for the _add block: 5 waves), C=96 188 (pinned at 5),
    # C=128 233 (4 waves). The floors force the allocator under the limit, at the price of some spilled registers
    # (checked with build.py --report). Bit-identical to base.
    "occ64_6": {"WG_MINW_C64": 6},                         # dec1 block 166 -> 160 VGPRs (5 -> 6 waves), 8 spills
    "occ64_7": {"WG_MINW_C64": 7},
    "occ64_8": {"WG_MINW_C64": 8},
    "occ96_6": {"WG_MINW_C96": 6},
    "occ128_5": {"WG_MINW_C128": 5},
    "occ128_6": {"WG_MINW_C128": 6},
    # Block workgroup size of the C=128 (bottleneck) kernel alone: 4 waves (finer launch granularity, twice the weight
    # traffic per window) / 16 waves.
    "c128t128": {"WG_THREADS_C128": 128},
    "c128t512": {"WG_THREADS_C128": 512},
    # Patch merge fused into the encoder blocks and expand fused into the decoder blocks (kernels *_mg / *_ex, always
    # compiled; this flag only makes the runner launch them). Bit-identical to base.
    "fuse": {"WG_FUSE": 1},
    # Attention variants (judged by the whole-frame rule): A.V on the FP8 WMMA with FP8 probabilities and V; soft cap and
    # softmax in packed FP16.
    "av8": {"WG_AV8": 1},
    "smax16": {"WG_SMAX16": 1},
    # GELU implementations (see wg_platform.h); the default is the fitted sigmoid form (mode 2).
    "geluexact": {"WG_GELU_MODE": 0},                      # exact: 0.001 % FP8 flips
    "gelu16": {"WG_GELU_MODE": 3},                         # packed FP16 polynomial, no transcendental: 2-6 % FP8 flips
    # First RMSNorm statistics with v_dot4_f32_fp8_fp8 (FP32 summation order only; blocks with plain FP8 input)
    "dot4norm": {"WG_NORM_DOT4": 1},
    # Mixed hidden path: k of the 4 value pairs of each tile by the packed polynomial, the rest by the sigmoid form. The
    # transcendental unit (exp, rcp: 3.9 slots each when alone) may run beside the vector ALU, so mixing the two balances
    # the units; whether it does is what rates.py's exp/fma interleave probe measures.
    "gelumix1": {"WG_GELU_MODE": 4, "WG_GELU_MIX": 1},
    "gelumix2": {"WG_GELU_MODE": 4, "WG_GELU_MIX": 2},
    "gelumix3": {"WG_GELU_MODE": 4, "WG_GELU_MIX": 3},
    # LDS prefetch in the MLP tile loop (see WG_MLP_PF in swin.hip): hides the two LDS round trips per tile. Bit-identical.
    "pf1": {"WG_MLP_PF": 1},
    "pf2": {"WG_MLP_PF": 2},
    "vec16": {"WG_GELU_MODE": 3, "WG_SMAX16": 1},          # all packed-FP16 vector paths together (GELU polynomial + softmax)
    "gelu725": {"WG_GELU_MODE": 1},                        # A&S 7.1.25 erfc: 0.05-0.14 % FP8 flips, none > 1 step
}

# Variants whose output differs from base (a different GELU than the default). The sweep reports their mismatches but
# does not treat them as an error.
INEXACT = {"geluexact", "gelu725", "gelu16", "gelumix1", "gelumix2", "gelumix3", "av8", "smax16", "vec16", "dot4norm"}

# Timing-only builds: part of the block is switched off, so the output is WRONG on purpose. They answer "where does
# the time go": the difference to base is what the removed part costs. Built with build.py --ablations.
ABLATIONS = {
    "ab_gelu": {"WG_ABLATE_GELU": 1},        # GELU = identity
    "ab_mlp": {"WG_ABLATE_MLP": 1},          # no MLP at all (steps 9-10)
    "ab_attn": {"WG_ABLATE_ATTN": 1},        # no softmax / A.V
    "ab_stage": {"WG_ABLATE_STAGING": 1},    # weights from LDS unit 0 only: no global weight loads, no barriers
    "ab_w1": {"WG_ABLATE_W1": 1},            # no W1 matrix multiplies (the MLP's first layer)
    "ab_w2": {"WG_ABLATE_W2": 1},            # no W2 matrix multiplies; GELU and the hidden packing stay fully alive
    "ab_nomat": {"WG_ABLATE_W1": 1, "WG_ABLATE_W2": 1},   # neither W1 nor W2: matrix cost = base - ab_nomat
    "ab_pack": {"WG_ABLATE_PACK": 1},        # hidden values: 6 adds instead of the clamps + FP8 conversions
    "ab_hidden": {"WG_ABLATE_GELU": 1, "WG_ABLATE_PACK": 1},   # no GELU, no FP8 conversion of the hidden values
    "ab_lds": {"WG_ABLATE_LDS": 1},          # MLP weight fragments from registers: no LDS reads there
    "ab_norm": {"WG_ABLATE_NORM": 1},        # both RMSNorms without the sum of squares and rsqrt
    "ab_load": {"WG_ABLATE_LOAD": 1},        # block input from registers instead of global loads
    "ab_store": {"WG_ABLATE_STORE": 1},      # no final store
    "ab_bias": {"WG_ABLATE_BIAS": 1},        # all bias vectors zero: no loads, no adds
    "ab_cap": {"WG_ABLATE_CAP": 1},          # attention: no position bias, no soft cap
    "ab_smax": {"WG_ABLATE_SMAX": 1},        # attention: no softmax reductions / normalization
    "ab_qkv": {"WG_ABLATE_QKV": 1},          # no Q/K and V projection matrix multiplies
    "ab_wp": {"WG_ABLATE_WP": 1},            # no attention output projection (Wp) matrix multiplies
    "ab_floor": {"WG_ABLATE_FLOOR": 1},      # gather -> store only: the memory / launch floor of the block kernel
    "ab_qpack": {"WG_ABLATE_QPACK": 1},      # FP8 conversions of norm outputs, q/k and o: 6 adds instead
}


# ---- second design: kernels/swin_v2.hip (same kernels, arguments and formats; its own block-kernel knobs) ----------
V2_SOURCE = "swin_v2.hip"
V2_KERNELS = {  # block kernel -> (threads macro, windows-per-wave macro)
    "swin_block_c64": ("WG_V2_THREADS_C64", "WG_V2_WPW_C64"),
    "swin_block_c64_add": ("WG_V2_THREADS_C64ADD", "WG_V2_WPW_C64"),
    "swin_block_c96": ("WG_V2_THREADS_C96", "WG_V2_WPW_C96"),
    "swin_block_c96_add": ("WG_V2_THREADS_C96ADD", "WG_V2_WPW_C96"),
    "swin_block_c128": ("WG_V2_THREADS_C128", "WG_V2_WPW_C128"),
}
# macro -> default (must match the defaults at the top of kernels/swin_v2.hip); merge / expand come from swin.hip
V2_DEFAULTS = {"WG_V2_THREADS_C64": 512, "WG_V2_THREADS_C64ADD": 512, "WG_V2_THREADS_C96": 512,
               "WG_V2_THREADS_C96ADD": 512, "WG_V2_THREADS_C128": 512,
               "WG_V2_HDR_GLOBAL": 0,
               "WG_V2_WPW_C64": 2, "WG_V2_WPW_C96": 1,
               "WG_V2_WPW_C128": 1, "WG_V2_GELU": 3, "WG_V2_GTAB_BITS": 8, "WG_V2_PIPE": 1, "WG_V2_UNIT_BYTES": 16384,
               "WG_V2_RESIDENT_C64": 1, "WG_V2_SCHED_FENCE": 1,
               "WG_V2_BASEMATH": 0, "WG_V2_MEGA_THREADS": 256, "WG_V2_MEGA_WPW": 1,
               "WG_V2_MEGA_HDR_GLOBAL": 1,
               "WG_GEMM_THREADS": 256, "WG_NT": 2}

V2_VARIANTS = {
    "v2": {},                                               # GELU = max(x,0) - c(|x|), c from a 16 KB FP16 LDS table
    "v2exact": {"WG_V2_GELU": 0, "WG_GELU_MODE": 0},        # exact GELU: for the real-data exactness check
    "v2gsig": {"WG_V2_GELU": 2},                            # GELU = gelusig (accepted under spec 09 draft 2)
    "v2g7": {"WG_V2_GTAB_BITS": 7},                         # correction table of 8 KB (about gelusig's accuracy)
    "v2gbyte": {"WG_V2_GELU": 4},                           # q8(GELU) read from a 16 KB FP8 table: fastest, least exact
    "v2nopipe": {"WG_V2_PIPE": 0},                          # MLP without the W1 / GELU software pipeline
    "v2c64t256": {"WG_V2_THREADS_C64": 256, "WG_V2_THREADS_C64ADD": 256},  # resident C=64 at 8 waves per workgroup
    "v2c64w1": {"WG_V2_WPW_C64": 1},                        # C=64: one window per wave (twice the weight copies)
    "v2c64w4": {"WG_V2_WPW_C64": 4},                        # C=64: four windows per wave (half the weight copies)
    "v2c64s": {"WG_V2_RESIDENT_C64": 0, "WG_V2_THREADS_C64": 256, "WG_V2_THREADS_C64ADD": 256,
               "WG_V2_WPW_C64": 1},                         # C=64 streamed like the unsplit C=96/128 kernel
    # C=96/128 (default: one window per wave, 512-lane workgroups sharing each staged weight unit among 16 windows)
    "v2t256": {"WG_V2_THREADS_C96": 256, "WG_V2_THREADS_C96ADD": 256, "WG_V2_THREADS_C128": 256},  # 8 windows/group
    "v2w2": {"WG_V2_THREADS_C96": 256, "WG_V2_THREADS_C96ADD": 256, "WG_V2_THREADS_C128": 256,
             "WG_V2_WPW_C96": 2, "WG_V2_WPW_C128": 2},      # 256 lanes, 2 interleaved windows per wave
    "v2lds32": {"WG_V2_THREADS_C96": 256, "WG_V2_THREADS_C96ADD": 256, "WG_V2_THREADS_C128": 256,
                "WG_V2_HDR_GLOBAL": 1, "WG_V2_UNIT_BYTES": 8192, "WG_V2_GTAB_BITS": 7},  # <= 32 KB LDS per group
    "v2nofence": {"WG_V2_SCHED_FENCE": 0},                  # no scheduling fences (compiler decides)
}
V2_ABLATIONS = {  # timing only, output wrong on purpose (v2prof: correct output, adds cycle counters; profile_v2.py)
    "v2prof": {"WG_V2_PROFILE": 1},                         # per-phase cycle counts (instrumented, slightly slower)
    "bmegaprof": {"WG_V2_BASEMATH": 1, "WG_V2_PROFILE": 1},  # same for the bottleneck megakernel (stage "bneck")
    "v2ab_mlp": {"WG_V2_ABLATE_MLP": 1},                    # no MLP
    "v2ab_hidden": {"WG_V2_ABLATE_HIDDEN": 1},              # no GELU and no FP8 conversion of the hidden values
    "v2ab_attn": {"WG_V2_ABLATE_ATTN": 1},                  # no softmax / A.V
}
# Bottleneck megakernel builds: swin_v2.hip with WG_V2_BASEMATH (every kernel computes exactly as swin.hip with the
# same arithmetic macros, so the whole build is bit-identical to that swin.hip build) plus swin_bneck6 /
# swin_bneck6_nc, the six b-blocks in one persistent launch (stage "bneck" of stage_runner.Runner, bench_bneck.py).
# The single-stage kernels of these builds are the v2 kernels with base arithmetic. 256 lanes, 8 windows per item
# (512-lane forms measured 0.55-0.68 ms against 0.40-0.42: removed).
V2_MEGA = {
    "bmegahg": {"WG_V2_BASEMATH": 1},                                      # header vectors from global: 32 KB LDS (best)
    "bmega": {"WG_V2_BASEMATH": 1, "WG_V2_MEGA_HDR_GLOBAL": 0},
    "bmegahg_t128": {"WG_V2_BASEMATH": 1, "WG_V2_MEGA_THREADS": 128},     # 4-window items: finer work units for the tail
    "bmegahg_t512": {"WG_V2_BASEMATH": 1, "WG_V2_MEGA_THREADS": 512},     # 16-window items: weights shared by 16 waves            # header staged in LDS: 40 KB
    "bmegahg16": {"WG_V2_BASEMATH": 1, "WG_GELU_MODE": 3, "WG_SMAX16": 1},  # vec16 arithmetic (packed FP16 GELU, softmax)
}
# The swin.hip build each megakernel build is bit-identical to (bench_bneck.py, sweep.py exactness).
MEGA_REFERENCE = {"bmegahg": "base", "bmegahg_t128": "base", "bmegahg_t512": "base", "bmega": "base", "bmegahg16": "vec16", "bmegaprof": "base"}
V2_VARIANTS.update(V2_MEGA)
VARIANTS.update(V2_VARIANTS)
ABLATIONS.update(V2_ABLATIONS)
# v2 differs from swin.hip in the order of some FP32 sums (and, by default, in the GELU): rare FP8 rounding differences.
INEXACT |= set(V2_VARIANTS) - {n for n in V2_MEGA if MEGA_REFERENCE[n] == "base"}


def is_v2(name):
    return name in V2_VARIANTS or name in V2_ABLATIONS


def source(name):
    """Kernel source file of a build (relative to clean/kernels)."""
    return V2_SOURCE if is_v2(name) else "swin.hip"

# Instrumented builds (exact results, slower): every wave records the time spent in each phase of the block. Built with
# build.py --profile and read with profile_stages.py. WG_PROFILE 1 drains the memory pipeline at every phase boundary
# (clean attribution, serializes the phases); 2 does not (stalls land in the phase that first uses the data).
PROFILES = {
    "prof": {"WG_PROFILE": 1},                       # shader cycle counter, drained phases
    "prof2": {"WG_PROFILE": 2},                      # shader cycle counter, no draining
    "profrt": {"WG_PROFILE": 1, "WG_PROF_CLOCK": 0},  # realtime clock (s_sendmsg_rtn) instead of the cycle counter
}
PROF_WORDS = 20  # words per instrumented wave (WG_PROF_WORDS in swin.hip)


def is_profile(name):
    return name in PROFILES


def all_builds():
    return {**VARIANTS, **ABLATIONS, **PROFILES}


def is_ablation(name):
    return name in ABLATIONS


def macros(name):
    return {**DEFAULTS, **all_builds()[name]}


def geometry(defs, src="swin.hip"):
    """Launch geometry of a variant from its macro values: lanes per workgroup and windows per wave of each block
    kernel, plus the merge / expand shape."""
    kernels, base = (V2_KERNELS, V2_DEFAULTS) if src == V2_SOURCE else (SWIN_KERNELS, DEFAULTS)
    m = {**base, **defs}
    geo = {"threads": {k: m[t] for k, (t, _) in kernels.items()},
           "wpw": {k: m[w] for k, (_, w) in kernels.items()},
           "gemm_threads": m["WG_GEMM_THREADS"], "nt": m["WG_NT"], "force_clamp": bool(m.get("WG_FORCE_CLAMP", 0)),
           "profile": int(m.get("WG_PROFILE", 0)), "fuse": bool(m.get("WG_FUSE", 0))}
    # windows per workgroup (the grid is ceil(nwin / wpg))
    geo["wpg"] = {k: geo["threads"][k] // 32 * geo["wpw"][k] for k in kernels}
    if src == V2_SOURCE and m.get("WG_V2_BASEMATH", 0):
        # bottleneck megakernel: lanes per workgroup, windows per wave, windows per work item
        # default grid: the resident capacity of a 64-CU RX 9070 XT (32 WGPs, 128 KB LDS each; 256 lanes: 4 x 32 KB
        # groups per WGP with the header in global memory, 3 x 40 KB with it in LDS). Any grid is correct (more
        # groups than fit only wait for the queue to empty); bench_bneck.py --groups.
        threads = m["WG_V2_MEGA_THREADS"]
        groups = (128 if m["WG_V2_MEGA_HDR_GLOBAL"] else 96) * 256 // threads
        geo["mega"] = {"threads": threads, "wpw": m["WG_V2_MEGA_WPW"],
                       "ipw": threads // 32 * m["WG_V2_MEGA_WPW"], "groups": groups}
        # persistent "mid" kernel (enc2 + merge, b0..b5, expand + dec2): 256 lanes, 8 windows per item; about 205
        # VGPRs, so 3 groups per WGP (96 on the 9070 XT)
        geo["mid"] = {"threads": 256, "ipw": 8, "groups": 96}
    return geo


def variant_geometry(name):
    return geometry(all_builds()[name], source(name))


def is_exact(name):
    return name not in INEXACT


MANIFEST = "variants.json"


def code_object(name):
    return "swin.co" if name == "base" else f"swin__{name}.co"


DEFAULT_GEOMETRY = geometry({})
