# SPDX-License-Identifier: MIT
"""Model K launch arguments and launch builders (kernels: clean/k/kernels/*.hip).

A launch is (kernel, grid, threads, [ctypes args]) as in clean/runner/stage_runner.py. Kernel names are bare
("k_lin_64x192") for the default code objects or "<code object>:<kernel>" for build variants; KERNEL_CODE_OBJECTS maps
the bare names to their code objects.
"""
import ctypes

from clean.k.reference.layout import STAGE, cdiv, window_grid
from clean.k.runner import k_pack

THREADS = 128
LIN_CO = {"g2": "k_linear.co", "g1": "k_linear__g1.co"}  # tokens per wave: 32 (default) or 16
LIN_G = {"g2": 2, "g1": 1}
FLAG_BIAS, FLAG_ACT, FLAG_RES, FLAG_RELU = 1, 2, 4, 8
BORDER = {"zero": 0, "clamp": 1, "mirror": 2}


class KLinParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "w", "bias", "res", "y")] + \
               [(n, ctypes.c_int) for n in ("in_w", "in_h", "out_w", "out_h", "ncols", "nchunks", "flags", "vw", "fw",
                                            "fh", "ox", "oy")]


class KBlkParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("x", "skip", "y", "merged", "head", "g1", "g2", "wqkv", "mix", "wo",
                                               "bo", "w1", "b1", "w2", "b2", "wa", "ba", "wz", "bz")] + \
               [(n, ctypes.c_int) for n in ("w", "h", "ox", "oy", "nwx", "nwy", "cw", "ch")]


class KWinParams(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in ("src", "dst")] + \
               [(n, ctypes.c_int) for n in ("w", "h", "c", "ox", "oy", "margin", "mode", "nwx", "nwy", "count")]


assert ctypes.sizeof(KLinParams) == 88 and ctypes.sizeof(KBlkParams) == 184 and ctypes.sizeof(KWinParams) == 56


def nwt(cout):
    t = cout // 16
    return next(c for c in (8, 6, 5, 4, 3, 2, 1) if t % c == 0)


def lin_kernel(mode, k, n):
    return {"plain": f"k_lin_{k}x{n}", "merge": f"k_merge_{k}x{n}", "expand": f"k_expand_{k}x{n}",
            "expandp": f"k_expandp_{k}x{n}"}[mode]


def floor2(a):
    return a // 2  # Python's // is floor division


def padded_size(fine_size, offset):
    """(8 nwx, 8 nwy): the padded map of a decoder stage's window grid."""
    nwx, nwy = window_grid(*fine_size, *offset)
    return 8 * nwx, 8 * nwy


def expandp_geometry(fine_size, offset):
    """(vw, vh): virtual coarse positions floor(r / 2) over the padded rows / columns (spec 41 draft 3)."""
    pw, ph = padded_size(fine_size, offset)
    ox, oy = offset
    vw = floor2(pw - ox - 1) - floor2(-ox) + 1
    vh = floor2(ph - oy - 1) - floor2(-oy) + 1
    return vw, vh


def block_kernel(C, heads, mix):
    return f"k_block_c{C}_h{heads}_{mix}"


def lin_launch(mode, k, n, x, w, y, in_size, out_size, bias=None, res=None, relu=False, act=False, variant="g2"):
    """One linear launch. in_size / out_size: (W, H) of the input / output map. Plain maps have in = out. expandp:
    out_size = ((W, H) of the fine map, (ox, oy) of the decoder stage); res is the skip map (fine, mirrored)."""
    vw = 0
    fw = fh = ox = oy = 0
    if mode == "expand":
        ncols = in_size[0] * in_size[1]
    elif mode == "expandp":  # out_size = (fine size, offset)
        (fw, fh), (ox, oy) = out_size
        vw, vh = expandp_geometry((fw, fh), (ox, oy))
        ncols = vw * vh
        out_size = padded_size((fw, fh), (ox, oy))
    else:
        ncols = out_size[0] * out_size[1]
    npad = k_pack.pad16(n)
    nchunks = npad // 16 // nwt(npad)
    flags = (FLAG_BIAS if bias else 0) | (FLAG_ACT if act else 0) | (FLAG_RES if res else 0) | \
            (FLAG_RELU if relu else 0)
    prm = KLinParams(x=x, w=w, bias=bias or 0, res=res or 0, y=y, in_w=in_size[0], in_h=in_size[1],
                     out_w=out_size[0], out_h=out_size[1], ncols=ncols, nchunks=nchunks, flags=flags, vw=vw, fw=fw,
                     fh=fh, ox=ox, oy=oy)
    name = lin_kernel(mode, k, n)
    if variant != "g2":
        name = f"{LIN_CO[variant]}:{name}"
    grid = cdiv(ncols, 64 * LIN_G[variant]) * nchunks
    return (name, grid, THREADS, [prm])


def lin_blob_launch(mode, k, n, x, blob, y, in_size, out_size, **kw):
    """lin_launch with the weights and bias of a linear blob at device address blob."""
    s = k_pack.linear_sections(n, k)
    return lin_launch(mode, k, n, x, blob + s["w"], y, in_size, out_size, bias=blob + s["b"], **kw)


# stages with a two-windows-per-workgroup build (k_stage2_*); d1's passes its block input through the block output map
# (k_block.hip XG), so it is used only when the stage writes one (bufs["block"], always in the network)
WINDOWS2_STAGES = ("e0", "e1", "d1", "d0")
WINDOWS2_NEEDS_BLOCK = ("d1",)
# Stages with a wide build (k_block_wide.hip, k_stagew<NG>_<name>: one window per NG x 4 waves) and their NG values.
WIDE_BUILDS = {"e3": (2, 4), "e4": (2, 4), "m": (4,), "d4": (2, 4), "d3": (2, 4)}
WIDE_DEFAULT = {"e3": 2, "e4": 2, "m": 4, "d4": 2, "d3": 2}  # the network's choice (k_net.DEFAULT_PATHS)
# the FP16-accumulating network's choice (k_net.DEFAULT_PATHS16, variant "fused16"; e3, d3: k_stageh_*)
WIDE_DEFAULT16 = {"e4": 4, "m": 4, "d4": 4}
# FP32 network: e4 and d4 with NG = 4 instead of 2 when the stage has at most this many windows (few windows: more waves
# per window shorten the latency-bound stage; 1440p output, 24 windows: -6 us per frame; 4K output, 40: +10 us)
WIDE4_SMALL = ("e4", "d4")
WIDE4_SMALL_MAX_WINDOWS = 24


def wide_ng(stage, ng, nwin, f16=False):
    """The NG the network runs a wide stage with (runtime_k.cpp wide_ng): WIDE4_SMALL at <= WIDE4_SMALL_MAX_WINDOWS."""
    if not f16 and ng == 2 and stage in WIDE4_SMALL and nwin <= WIDE4_SMALL_MAX_WINDOWS:
        return 4
    return ng


def _block_params(C, heads, mix, x, y, blob, size, offset, **extra):
    s = k_pack.block_sections(C, heads, mix)
    w, h = size
    ox, oy = offset
    nwx, nwy = window_grid(w, h, ox, oy)
    prm = KBlkParams(x=x, y=y or 0, g1=blob + s["g1"], g2=blob + s["g2"], wqkv=blob + s["qkv"],
                     mix=blob + s["mix"], wo=blob + s["wo"], bo=blob + s["bo"], w1=blob + s["w1"], b1=blob + s["b1"],
                     w2=blob + s["w2"], b2=blob + s["b2"], w=w, h=h, ox=ox, oy=oy, nwx=nwx, nwy=nwy, **extra)
    return prm, nwx * nwy


def block_launch(C, heads, mix, x, y, blob, size, offset, padded=False):
    """One block launch on a map of size (W, H) with window offset (ox, oy); blob: device address of a block blob.
    padded: x is the padded window-grid map of a decoder stage (padded_size), else the map itself (mirrored)."""
    prm, grid = _block_params(C, heads, mix, x, y, blob, size, offset)
    name = block_kernel(C, heads, mix)
    return (name.replace("k_block_", "k_blockp_") if padded else name, grid, THREADS, [prm])


def fusable(stage, offset):
    """Whether the fused stage kernel applies: merging and decoder stages need an even offset (spec 44 always gives
    one)."""
    st = STAGE[stage]
    return st["kind"] == "mid" or (offset[0] % 2 == 0 and offset[1] % 2 == 0)


def stage_fused_launch(stage, blob, bufs, sizes, offset, windows=1, wide=0, f16=False):
    """One launch for the whole stage (kernel k_stage_<name>; m: the plain block). bufs: "in", "skip" (dec),
    "block" (block output, or None: not written; d0 in the network), "merged" (enc), "head" (d0); sizes: "level" (W, H)
    of the block level, "in" (W, H) of the input map. windows=2: two windows per workgroup (k_stage2_*, e0, e1, d1, d0
    only; other stages ignore it). wide=NG: the wide kernel k_stagew<NG>_<name> (WIDE_BUILDS), 128 NG lanes per
    window. f16: the FP16-accumulating form (spec 49 speed approximation, opt-in): the kernel name's prefix + "h"
    (k_stageh_<name> with m included, k_stage2h_*, k_stagew<NG>h_*)."""
    st = STAGE[stage]
    if not fusable(stage, offset):
        raise ValueError(f"{stage}: the fused stage kernel needs an even window offset, got {offset}")
    if wide and wide not in WIDE_BUILDS.get(stage, ()):
        raise ValueError(f"{stage}: no wide build with NG = {wide}")
    h = "h" if f16 else ""
    if st["kind"] == "mid" and not wide and f16:  # k_stageh_m: the block's arguments
        launch = block_launch(st["C"], st["heads"], st["mix"], bufs["in"], bufs["block"],
                              blob + k_pack.stage_sections(st)["block"], sizes["level"], offset)
        return (f"k_stageh_{stage}",) + launch[1:]
    if st["kind"] == "mid" and not wide:
        return block_launch(st["C"], st["heads"], st["mix"], bufs["in"], bufs["block"],
                            blob + k_pack.stage_sections(st)["block"], sizes["level"], offset)
    sec = k_pack.stage_sections(st)
    extra = {}
    if st.get("embed"):
        ls = k_pack.linear_sections(32, 16)
        extra.update(wa=blob + sec["embed"] + ls["w"], ba=blob + sec["embed"] + ls["b"])
    if st["kind"] == "dec":
        ls = k_pack.linear_sections(4 * st["C"], st["C_prev"])
        extra.update(wa=blob + sec["expand"] + ls["w"], ba=blob + sec["expand"] + ls["b"], skip=bufs["skip"],
                     cw=sizes["in"][0], ch=sizes["in"][1])
    if st["kind"] == "enc":
        ls = k_pack.linear_sections(st["C_next"], 4 * st["C"])
        extra.update(wz=blob + sec["merge"] + ls["w"], bz=blob + sec["merge"] + ls["b"], merged=bufs["merged"])
    if st.get("head"):
        ls = k_pack.linear_sections(40, st["C"])
        extra.update(wz=blob + sec["head"] + ls["w"], bz=blob + sec["head"] + ls["b"], head=bufs["head"])
    prm, grid = _block_params(st["C"], st["heads"], st["mix"], bufs["in"], bufs.get("block"), blob + sec["block"],
                              sizes["level"], offset, **extra)
    if wide:
        return (f"k_stagew{wide}{h}_{stage}", grid, THREADS * wide, [prm])
    if windows == 2 and stage in WINDOWS2_STAGES and (stage not in WINDOWS2_NEEDS_BLOCK or bufs.get("block")):
        return (f"k_stage2{h}_{stage}", cdiv(grid, 2), THREADS, [prm])
    return (f"k_stage{h}_{stage}", grid, THREADS, [prm])


def win_launch(kernel, src, dst, size, c, offset, margin=0, border="mirror"):
    w, h = size
    ox, oy = offset
    nwx, nwy = window_grid(w, h, ox, oy)
    s = 8 + 2 * margin
    members = s * s if kernel == "k_win_gather" else 64
    count = nwx * nwy * members * (c // 8)
    prm = KWinParams(src=src, dst=dst, w=w, h=h, c=c, ox=ox, oy=oy, margin=margin, mode=BORDER[border], nwx=nwx,
                     nwy=nwy, count=count)
    return (kernel, cdiv(count, THREADS), THREADS, [prm])


def stage_launches(stage, blob, bufs, sizes, offset, variant="g2"):
    """Launches of one model K stage. blob: device address of the stage blob; bufs: device addresses of
    "in" (stage input), "skip" (dec), "tmp" (embedding output, or the decoder's padded block input of
    padded_size(level, offset)), "block" (block output), "merged" (enc),
    "head" (d0); sizes: (W, H) of "in" and of the block level ("level"). Returns [launch]."""
    st = STAGE[stage]
    sec = k_pack.stage_sections(st)
    C, lv = st["C"], sizes["level"]
    L = []
    src = bufs["in"]
    if st.get("embed"):
        L.append(lin_blob_launch("plain", 16, 32, src, blob + sec["embed"], bufs["tmp"], lv, lv, relu=True,
                                 variant=variant))
        src = bufs["tmp"]
    if st["kind"] == "dec":
        L.append(lin_blob_launch("expandp", st["C_prev"], 4 * C, src, blob + sec["expand"], bufs["tmp"],
                                 sizes["in"], (lv, offset), res=bufs["skip"], variant=variant))
        src = bufs["tmp"]
    L.append(block_launch(C, st["heads"], st["mix"], src, bufs["block"], blob + sec["block"], lv, offset,
                          padded=st["kind"] == "dec"))
    if st["kind"] == "enc":
        out = (cdiv(lv[0], 2), cdiv(lv[1], 2))
        L.append(lin_blob_launch("merge", 4 * C, st["C_next"], bufs["block"], blob + sec["merge"], bufs["merged"],
                                 lv, out, variant=variant))
    if st.get("head"):
        L.append(lin_blob_launch("plain", C, 40, bufs["block"], blob + sec["head"], bufs["head"], lv, lv,
                                 variant=variant))
    return L


def kernel_code_objects():
    """{bare kernel name: code object} of every model K kernel (the default builds)."""
    from clean.k.reference.layout import STAGES, linear_shapes
    m = {}
    for mode, k, n in linear_shapes():
        m[lin_kernel(mode, k, n)] = LIN_CO["g2"]
        if mode == "expand":
            m[lin_kernel("expandp", k, n)] = LIN_CO["g2"]
    for st in STAGES:
        m[block_kernel(st["C"], st["heads"], st["mix"])] = "k_block.co"
        m[block_kernel(st["C"], st["heads"], st["mix"]).replace("k_block_", "k_blockp_")] = "k_block.co"
        if st["kind"] != "mid":
            m[f"k_stage_{st['name']}"] = "k_block.co"
        m[f"k_stageh_{st['name']}"] = "k_block.co"
    m["k_stage2_d0_hidden"] = "k_block.co"
    m["k_stage2h_d0_hidden"] = "k_block.co"
    m["k_stage2_d0_hp"] = "k_block.co"    # hidden epilogue + compact pixel head (complete frames)
    m["k_stage2h_d0_hp"] = "k_block.co"
    for s in WINDOWS2_STAGES:
        m[f"k_stage2_{s}"] = "k_block.co"
        m[f"k_stage2h_{s}"] = "k_block.co"
    for s, ngs in WIDE_BUILDS.items():
        for ng in ngs:
            m[f"k_stagew{ng}_{s}"] = "k_block_wide.co"
            m[f"k_stagew{ng}h_{s}"] = "k_block_wide.co"
    m.update({"k_win_gather": "k_window.co", "k_win_scatter": "k_window.co", "k_act_table": "k_act.co", "k_fp_probe": "k_act.co", "k_exp2h_probe": "k_act.co",
              "k_act_table2": "k_act.co", "k_out_color": "k_io.co", "k_out_pixel": "k_io.co", "k_out_pixel_c": "k_io.co", "k_out_hidden": "k_io.co",
              "k_in_color": "k_io.co", "k_in_base": "k_io.co", "k_in_reset": "k_io.co"})
    return m


KERNEL_CODE_OBJECTS = kernel_code_objects()
