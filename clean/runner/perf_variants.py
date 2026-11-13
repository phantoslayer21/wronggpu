# SPDX-License-Identifier: MIT
"""Build variants of the per-pixel stages enc0, dec0 and exposure (performance milestone, clean/PERF.md).

Each variant is a code object built from one kernel source with some -D macros; perf_bench.py times them at the
in-game size and checks that the non-ablation variants give the same outputs as "base". The runner addresses a
variant as stage "<stage>@<variant>" (stage_runner.EXTRA_STAGES), e.g. "enc0@v1".

    base      the default build (what the runtime uses)
    v1        the first version of the kernel (kernels/baseline/*_v1.hip), as measured by the spec team before this
              milestone, for a same-machine comparison
    others    alternatives of the default whose speed cannot be predicted without the GPU
    ab_*      timing-only ablations: part of the work removed, the output is wrong on purpose; the time difference to
              "base" is what that part costs

Builds: python clean/runner/build.py ... --perf-variants  (writes perf_variants.json next to the code objects).
"""

# stage -> (kernel source of the default, code object of the default)
STAGES = {"enc0": ("encoder.hip", "encoder.co"), "dec0": ("decoder.hip", "decoder.co"),
          "exposure": ("exposure.hip", "exposure.co")}

# stage -> variant -> (source relative to clean/kernels, defines, ablation)
VARIANTS = {
    "enc0": {
        "base": ("encoder.hip", {}, False),
        "v1": ("baseline/encoder_v1.hip", {}, False),
        "lds": ("encoder.hip", {"WG_ENC_FRAG_GLOBAL": 0}, False),        # A fragments staged in LDS (1668645 default)
        "w12": ("encoder.hip", {"WG_ENC_WAVES": 12}, False),              # register target for 12 waves per SIMD
        "ntstore": ("encoder.hip", {"WG_ENC_NT_STORE": 1}, False),        # non-temporal m_far / m_center stores
        "colorh": ("encoder.hip", {"WG_ENC_COLOR_F16": 1}, False),        # colour read as RGBA16F (input rounded to FP16)
        "motionh": ("encoder.hip", {"WG_ENC_MOTION_F16": 1}, False),      # motion vectors read as 2 x FP16
        "in16": ("encoder.hip", {"WG_ENC_COLOR_F16": 1, "WG_ENC_MOTION_F16": 1}, False),  # both
        "nopin": ("encoder.hip", {"WG_ENC_PIN": 0}, False),               # compiler's own schedule before the projection
        "seq": ("encoder.hip", {"WG_ENC_SEQ": 1}, False),                 # no overlap between a lane's two pixels
        "ab_hist": ("encoder.hip", {"WG_ENC_AB_HIST": 1}, True),
        "ab_lat": ("encoder.hip", {"WG_ENC_AB_LAT": 1}, True),
        "ab_dis": ("encoder.hip", {"WG_ENC_AB_DIS": 1}, True),
        "ab_depth": ("encoder.hip", {"WG_ENC_AB_DEPTH": 1}, True),
        "ab_quant": ("encoder.hip", {"WG_ENC_AB_QUANT": 1}, True),
        "ab_proj": ("encoder.hip", {"WG_ENC_AB_PROJ": 1}, True),
        "ab_store": ("encoder.hip", {"WG_ENC_AB_STORE": 1}, True),
    },
    "dec0": {
        "base": ("decoder.hip", {}, False),
        "v1": ("baseline/decoder_v1.hip", {}, False),                     # needs the version-1 tables (dec0.py)
        "early": ("decoder.hip", {"WG_DEC_EARLY": 1}, False),             # previous-latent loads before the projection
        "fragglobal": ("decoder.hip", {"WG_DEC_FRAG_GLOBAL": 1}, False),  # A fragments from global; only tables in LDS
        "nt": ("decoder.hip", {"WG_DEC_NT": 1}, False),                   # non-temporal A, B, C stores (= base now)
        "temporal": ("decoder.hip", {"WG_DEC_NT": 0}, False),             # ordinary stores (previous default)
        "pack12": ("decoder.hip", {"WG_DEC_PACK12": 1}, False),           # A, B, C in 12 instead of 16 bytes (format!)
        "ab_prev": ("decoder.hip", {"WG_DEC_AB_PREV": 1}, True),          # no previous-latent read (and its use)
        "ab_soft": ("decoder.hip", {"WG_DEC_AB_SOFT": 1}, True),          # no softmax
        "ab_proj": ("decoder.hip", {"WG_DEC_AB_PROJ": 1}, True),          # no projection (WMMA, LDS fragments)
        "ab_store": ("decoder.hip", {"WG_DEC_AB_STORE": 1}, True),        # only the latent is stored
    },
    "exposure": {
        "base": ("exposure.hip", {}, False),
        "v1": ("baseline/exposure_v1.hip", {}, False),
    },
}

# Fused dec1 + dec0 (kernels/dec1_dec0.hip, runner/dec1dec0.py): stage "dec1dec0@<name>" -> (code object, defines).
# "fused" is the default build of dec1_dec0.hip; "fused_t" uses ordinary instead of non-temporal A, B, C stores (a
# wave of the fused kernel writes a 4 x 4-token window, i.e. 8 short row pieces, instead of 2 whole rows).
FUSION = {"fused": ("dec1_dec0.co", {}), "fused_t": ("dec1_dec0__temporal.co", {"WG_DEC_NT": 0})}

MANIFEST = "perf_variants.json"


def code_object(stage, variant):
    return STAGES[stage][1] if variant == "base" else f"{STAGES[stage][1][:-3]}__{variant}.co"


def is_ablation(stage, variant):
    return VARIANTS[stage][variant][2]


def input_f16(stage, variant):
    """(colour, motion) read as FP16 by this variant: compare it with the default build on FP16-rounded inputs."""
    d = VARIANTS[stage][variant][1]
    return d.get("WG_ENC_COLOR_F16", 0) == 1, d.get("WG_ENC_MOTION_F16", 0) == 1


def round_inputs(x, color, motion):
    """The input dict with the colour and/or motion vectors rounded to FP16 (what an input_f16 variant reads)."""
    import numpy as np
    r = lambda a: np.asarray(a, np.float32).astype(np.float16).astype(np.float32)  # noqa: E731
    return dict(x, **({"color": r(x["color"])} if color else {}), **({"motion": r(x["motion"])} if motion else {}))
