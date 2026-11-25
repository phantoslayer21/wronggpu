# SPDX-License-Identifier: MIT
"""Model L's runtime frame on the CPU (#89): the runtime built with Model L (define WG_MODEL_L, clean/l/runtime) runs
on the recording device of clean/tests/runtime_frame_trace.cpp (launches recorded with their arguments, not executed;
no GPU, no HIP). The interface is clean/l/FORMATS.md.

- The 0.6.5 release routes (WG_L_FORM unset = hybrid_full): on HIP the scalar reference (the FP16-form kernels with
  float32 chains, l_q7__hyb.co / l_q7w__hyb.co, for Model L's two blocks and the ten shared stages) and the C1/C2
  reconstruction (lc12/); on native D3D12 the fast route (every Swin / merge / expand kernel the manifest's "_hyb"
  twin on the unfused M-form layout, the pack12 store, lc12/ reconstruction). Both decoder blocks read the encoder
  block's map as their skip. The M-form L route is refused (ReleaseDefault); no test here runs it.
- Launch for launch (HIP): Model L's frame is Model M's frame with steps 2 to 5 replaced: the values into the halo
  map, the two per-pixel linears, the encoder block and the gather projection in place of input_encoder; the ten
  FP16-form shared stages in place of Model M's; the decoder projection, the decoder block and l_coef in place of
  coef_decoder; the C1/C2 reconstruction in place of post_recon. The grids come from FORMATS.md and the kernels, the
  window offsets from the default tables (clean/l/reference/table.py, clean/reference/swin.py). With an empty build
  folder and with a full build's optional code objects: Model L on HIP runs without pack12 (l_coef writes the base
  coefficient format).
- Arguments: every map goes from the launch that writes it to the one that reads it (q -> x1 -> x2 -> x3 -> enc0's
  features -> enc1; dec1's output -> y -> y2 -> l_coef), no launch writes what it reads, every buffer holds its map
  (allocation sizes), the extents (npix, tw, th, H, W), the window offsets of both parities, the render size, padded
  grid and reset in the encoder's and decoder's parameters, and the history (the latent l_coef writes is what the next
  frame's encoder reads). Sizes aligned, non-aligned and changing from frame to frame, and a game reset.
- A failing launch stops its frame; the next frame runs whole.
- Model L's code objects are loaded (l_enc_values.co under the tag lev, l_ops.co, l_swin.co, l_post.co, the hybrid
  builds l_q7__hyb.co / l_q7w__hyb.co, l_post__lc12.co under the tag lc12), Model M's encoder and decoder not. Model M runs unchanged in the build with Model L.
- Model data: only the converter's model.wgcm, checked against the preset-L map (a synthetic file and map here). A raw
  folder is refused, so is a file whose shared stages are Model M's (their digests are not the map's), and a missing
  file. The paths that keep dec1's map on chip, need Model M's encoder or the pack12 decoder are refused.
- Native D3D12: the shared stages, the pack12 store and the C1/C2 reconstruction, every argument by its buffer or
  blob (the twins' blobs: Model M's packer on the hybrid values); the route stays unfused whatever the fusion option
  or the manifest offers; each port the route needs refused by name when missing.

Built with the host C++ compiler (clean/tests/host_cxx.py; skipped without one).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_frame.py
"""
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import numpy as np

import test_l_runtime_load as LL
from clean.l.reference import oracle as L
from clean.l.reference import table as T
from clean.l.runner import l_hyb_stage_pack as HSP
from clean.l.runner import l_pack as LP
from clean.reference import swin as SW
from clean.runner import packing as P
from clean.runner import synthetic_model as SM
from clean.tests import host_cxx
from clean.tests import test_runtime_frame_trace as FT

L_SOURCES = [os.path.join("clean", "l", "runtime", "l_pack.cpp"), os.path.join("clean", "l", "runtime", "l_model.cpp")]
SEED = 6
RW, RH = 80, 48  # the maximum render size of most runs: not a multiple of 32 (F = 96 x 64)
SHARED = ("enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1")
_EXE = None
_MODEL = None


def trace_exe():
    global _EXE
    if _EXE is None:
        _EXE = host_cxx.build("runtime_frame_trace_l", FT.SOURCES + L_SOURCES, ["WG_NO_MODEL_K", "WG_MODEL_L"])
    return _EXE


def write_l(model_dir, tensors, map_file=None):
    """<model_dir>/L/model.wgcm in the preset-L map's stage order; with map_file, the file records that map (whose
    digests it then must match), else a synthetic map of its own is written. Returns the map's path."""
    return SM.write_wgcm(os.path.join(model_dir, "L"), tensors, order=list(LL.specs()), map_file=map_file)


def model_dir():
    """A synthetic Model M in raw/ and a synthetic Model L as L/model.wgcm; its map under the preset-L map's file
    name in maps/ (the runtime finds Model L's map by that name next to map_path)."""
    global _MODEL
    if _MODEL is None:
        d = tempfile.mkdtemp(prefix="wg_lf_")
        SM.write_raw(d, SM.synthetic_tensors(1))
        synthetic_map = write_l(d, LL.raw_model(SEED))
        os.makedirs(os.path.join(d, "maps"))
        shutil.copyfile(synthetic_map, os.path.join(d, "maps", "preset-L.v1.map.json"))
        _MODEL = d
    return _MODEL


def map_path():
    return os.path.join(model_dir(), "maps", "preset-L.v1.map.json")


def run(options="", frames=4, build_files=(), env=None, model=None, exe=None, flags=(), rw=RW, rh=RH, hdr=1,
        manifest=None, tf_default=False, w3_default=False):
    """The trace program's output lines (WG_* removed from the environment, then env applied). hdr: WgRtConfig.hdr
    (Model L's reconstruction kernel is HDR; Model M does not read the flag). manifest: [(kernel, code object)] for
    <build>/dx12/manifest.json (the D3D12 ports a build has; backend 3). tf_default: leave WG_L_POST_TF unset (Model L's
    fused post, the shipped default); otherwise it is 0, the separate launches the other tests trace. w3_default: leave
    WG_L_W3 unset (the wave 3 kernels of the default mask 0x17, when the manifest has them); otherwise it is 0, the wave 2
    kernels (the names every other test checks)."""
    build = tempfile.mkdtemp(prefix="wg_lfb_")
    for n in build_files:
        open(os.path.join(build, n), "wb").close()
    if manifest is not None:
        os.makedirs(os.path.join(build, "dx12"))
        with open(os.path.join(build, "dx12", "manifest.json"), "w") as fh:
            json.dump({"format": 1, "kernels": [{"name": k, "co": co} for k, co in manifest]}, fh)
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
    if not tf_default:   # the separate launches (WG_L_POST_TF=0: recon + filter + scale_rgb), unless the test is of the shipped defaults
        e["WG_L_POST_TF"] = "0"
    if not w3_default:   # the wave 2 kernels (WG_L_W3=0), unless the test is of the wave 3 selection
        e["WG_L_W3"] = "0"
    e.update(env or {})
    try:
        r = subprocess.run([exe or trace_exe(), build, model or model_dir(), str(rw), str(rh), "120", "72", str(frames),
                            options, "--format=0", "--map=" + map_path(), "--hdr=%d" % hdr, *flags],
                           capture_output=True, text=True,
                           env=e, timeout=600)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if r.returncode != 0:
        raise AssertionError("runtime_frame_trace failed (exit %d)\n%s%s" % (r.returncode, r.stdout, r.stderr))
    return r.stdout.splitlines()


def blob_digests(route="hip"):
    """{blob name: SHA-256 hex} of the blobs Model L's launches read on `route`, packed in Python from the test model's
    parameters: the independent names of the weight buffers (the recorder logs what create uploads, with its hash).
    hip (the scalar hybrid_full reference): Model L's own blobs (clean/l/runner/l_pack.py pack_model_l) with the
    FP16-form blobs in place of the two block blobs (pack_q7_block: "enc0_block.q7", "dec0_block.q7"), the ten shared
    stages' "<s>.q7" / "<s>.q7lin" and the tables "q7.tab"; d3d12 (the fast hybrid route): pack_model_l of the
    parameters in the hybrid values (clean/l/runner/l_hyb_stage_pack.py hyb_params), the ten shared stages' blobs
    packed from them by Model M's packer (clean/runner/packing.py pack_stage)."""
    params = L.random_params(np.random.default_rng(SEED))
    if route == "d3d12":
        hyb = HSP.hyb_params(params)
        blobs = LP.pack_model_l(hyb)
        blobs.update({s: P.pack_stage(s, hyb[s], "gfx12", model="M") for s in SHARED})   # the twins' stage blobs
    else:
        blobs = LP.pack_model_l(params)
        del blobs["enc0_block"], blobs["dec0_block"]
        for s, C, heads, lin in Q7_STAGES:
            blobs[s + ".q7"] = LP.pack_q7_block(params[s], C, heads, 4 * C)
            if lin:
                blobs[s + ".q7lin"] = LP.pack_q7_linear(params[s][lin[0]], params[s][lin[1]])
        blobs["q7.tab"] = LP.q7_tables()
    # the post blob: s as float32, zero padding to 16 bytes (Model M's layout; the test model's s, LL.raw_model)
    blobs["post"] = struct.pack("<f", 1.25) + bytes(12)
    return {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}


# The FP16-form blobs (clean/l/runtime/l_pack.cpp pack_q7_model): stage, C, heads, the merge / expansion tensors
Q7_STAGES = (("enc0_block", 32, 2, None), ("dec0_block", 32, 2, None), ("enc1", 64, 2, ("Wm", "bm")),
             ("enc2", 96, 4, ("Wm", "bm"))) + tuple((b, 128, 4, None) for b in ("b0", "b1", "b2", "b3", "b4", "b5")) + \
    (("dec2", 96, 4, ("We", "be")), ("dec1", 64, 2, ("We", "be")))

# 0.6.5 (oversight 2026-11-15 18:36Z): Model L = hybrid_full (WG_L_FORM unset). Its two routes:
#   HIP: the scalar reference (l_q7__hyb.co, l_q7w__hyb.co: the FP16-form kernels with float32 chains), the C1/C2
#        reconstruction (lc12/l_post_recon[_ldr], l_post__lc12.co);
#   native D3D12: the fast route, the manifest's "_hyb" twins on the unfused M-form layout, the pack12 store and the
#        C1/C2 reconstruction (lc12/l_post_recon_p12[_ldr]; frames the pack12 kernels do not take: l_coef and
#        lc12/l_post_recon[_ldr]).
# Both decoder blocks read the encoder block's map as their skip S (RN16(E + S)).
ENC_BLOCK = {"hip": "l_q7_block_c32_enc", "d3d12": "l_swin_block_c32_enc_hyb"}
DEC_BLOCK = {"hip": "l_q7_block_c32_dec", "d3d12": "l_swin_block_c32_dec_hyb"}
L_KERNELS = ("lev/input_encoder", "l_linear_32_32_relu", "l_linear_32_32_f16", "l_enc_block", "l_project_gather",
             "l_dec_project", "l_dec_block", "l_coef", "l_post_recon")
# what neither release route runs: the M-form L blocks and the native fusions on them (lc: the decoder block with
# l_coef_pack12 folded in; ll: the two linears in one kernel; le: the encoder block with the linears as its prologue)
FOLD = "l_swin_block_c32_dec_coef12_nc"
LIN = "l_linear_32_32_fused"
ENCLIN = "l_swin_block_c32_enc_lin_nc"
# (speed lever 3: the hybrid route's encoder block takes the linears from the q20 map itself, so the fused linears and
# the q20 input encoder are no longer M-form only: lev/input_encoder_q20 is the native route's encoder, LIN unused)
M_FORM_L = ("l_swin_block_c32_enc", "l_swin_block_c32_enc_nc", "l_swin_block_c32_dec", "l_swin_block_c32_dec_nc",
            FOLD, ENCLIN)
# the native-D3D12 hybrid route's fused twins (speed lever 3): the encoder block with the two input linears (q from the
# q20 map), the decoder block with the pack12 coefficient store (pack12 frames); required, refused by name if missing
ENC_LIN_HYB = "l_swin_block_c32_enc_lin_hyb"
DEC_COEF_HYB = "l_swin_block_c32_dec_coef12_hyb"
# the hybrid bottleneck in one launch (speed lever 4: kernels/hyb_bneck.hlsl), b0 ... b5 on the native route
BNECK_HYB = "swin_bneck6_hyb"
FUSED_REQUIRED = (ENC_LIN_HYB, DEC_COEF_HYB, "lev/input_encoder_q20", "lev/input_encoder_direct_q20", BNECK_HYB)
NO_FOLD = {"WG_D3D12_FUSION": "pm,xe,mb,rv,xo,xd"}
NO_LIN = {"WG_D3D12_FUSION": "pm,xe,mb,rv,xo,xd,lc"}
NO_ENCLIN = {"WG_D3D12_FUSION": "pm,xe,mb,rv,xo,xd,lc,ll"}
# the D3D12 route's ten shared stages: the twins in launch order (unfused: dec2's and dec1's own blocks and expands)
TWIN_STAGES_UNFUSED = ("swin_block_c64_hyb", "patch_merge_c64_hyb", "swin_block_c96_hyb", "patch_merge_c96_hyb",
                       "swin_bneck6_hyb", "expand_c128_hyb", "swin_block_c96_add_hyb", "expand_c96_hyb",
                       "swin_block_c64_add_hyb")
# WG_L_EXFUSE (default on when the manifest has the kernels): dec2 and dec1 with their expands fused into the block
# (hyb_swin.hlsl swin_block_c96_add_ex_hyb / swin_block_c64_add_ex_hyb; the expanded map is never stored)
EX_HYB = ("swin_block_c96_add_ex_hyb", "swin_block_c64_add_ex_hyb")
# Model L wave 3 (WG_L_W3, default mask 0x17 = 0x01 | 0x02 | 0x04 | 0x10): the kernel of the wave 2 route -> its wave 3 form
# (the same arguments, grids and bytes; hyb_swin.hlsl, hyb_bneck.hlsl, l_swin.hlsl, l_ops_project_w64.hlsl)
W3_DEFAULT = {"l_swin_block_c32_enc_lin_hyb": "l_swin_block_c32_enc_lin_hyb_w3",
              "l_swin_block_c32_dec_coef12_hyb": "l_swin_block_c32_dec_coef12_hyb_w3",
              "swin_block_c64_hyb": "swin_block_c64_hyb_w3", "swin_block_c64_add_ex_hyb": "swin_block_c64_add_ex_hyb_w3",
              "swin_block_c96_hyb": "swin_block_c96_hyb_w3", "swin_block_c96_add_ex_hyb": "swin_block_c96_add_ex_hyb_w3",
              "swin_bneck6_hyb": "swin_bneck6_hyb_w3", "l_project_gather": "l_project_gather_w64",
              "l_dec_project": "l_dec_project_w64"}
TWIN_STAGES = ("swin_block_c64_hyb", "patch_merge_c64_hyb", "swin_block_c96_hyb", "patch_merge_c96_hyb",
               "swin_bneck6_hyb") + EX_HYB


def same_frames(lines):
    """A trace's lines for comparing two runs of the release route: the build folder's name dropped, and the kernel
    line's fusion reports (d3d12_fusion, l_lin_fused, l_dec_fold, l_enc_lin: whether the manifest has kernels that
    do not change the route)."""
    out = []
    for x in lines:
        x = re.sub(r"wg_lfb_\w+", "B", x)
        if x.startswith("kernels "):
            x = re.sub(r" (d3d12_fusion|l_lin_fused|l_dec_fold|l_enc_lin)=\S+", "", x)
        out.append(x)
    return out


def route_of(t):
    """The route a trace ran: d3d12 (the hybrid twins) or hip (the FP16-form reference)."""
    return "d3d12" if "l_hyb_d3d12" in t.kernels else "hip"


def latent_out(coef):
    """The new latent a frame's coefficient store wrote (l_coef / l_coef_pack12)."""
    return coef.args[4]


def block_args(route, x, skip, out, w, tab, fh, fw, offsets):
    """A Model L block's arguments: x, skip, out, w, H, W, ox, oy (the twins); the FP16-form blocks take the tables
    after w."""
    return [x, skip, out, w] + ([tab] if route == "hip" else []) + [str(fh), str(fw)] + offsets


def expected_args(t, f, digests, route):
    """Every Model L launch's arguments from the buffers' names (the runtime's wg_rt_buffer names after the frame) and
    the weight blobs' addresses (the upload holding each blob's bytes), independently of the launches themselves:
    {kernel role (base): [argument]}, None where the parameters E / D / P stand (checked field by field elsewhere)."""
    n = f["names"]
    w = {}
    for blob, digest in digests.items():
        w[blob] = t.uploaded(digest)
        if w[blob] is None:
            raise AssertionError("no single upload holds blob %s" % blob)
    info = f["info"]
    fw, fh = cdiv(info["rw"], 32) * 32, cdiv(info["rh"], 32) * 32
    npix, tw, th = str((fh + 4) * (fw + 4)), str(fw // 2), str(fh // 2)
    eo = [str(v) for v in T.block_offset(T.TABLE, "first", info["parity"])]
    do = [str(v) for v in T.block_offset(T.TABLE, "second", info["parity"])]
    hip = route == "hip"
    tab = w.get("q7.tab")
    fused = any(ln.kernel == ENC_LIN_HYB for ln in f["launches"])
    q_map = n["l.narrow"] if fused else n["l.wide"]
    return {
        # the input encoder's values: Model M's encoder inputs (the colour as the network reads it, here the game's,
        # pre-exposure 1), the state it reads, q into the wide map, the motion surfaces, the exposure pair
        "lev/input_encoder": [n["input.color"], n["input.depth"], n["input.motion"], n["post.reduced"],
                              n["state.latent_prev"], n["state.motion_history"], q_map, n["enc0.m_near"],
                              n["enc0.m_far"], n["enc0.m_center"], n["exposure.pair"], w["enc0.values"], None],
        "l_linear_32_32_relu": [n["l.wide"], n["l.narrow"], w["enc0.Win1"], npix],  # q -> x1
        "l_linear_32_32_f16": [n["l.narrow"], n["l.wide"], w["enc0.Win2"], npix],   # x1 -> x2
        # x2 -> x3 (no skip)
        "l_enc_block": block_args(route, n["l.wide"], "0", n["l.x3"], w["enc0_block.q7" if hip else "enc0_block"],
                                  tab, fh, fw, eo),
        "l_project_gather": [n["l.x3"], n["enc0.features"], w["enc0.project"], tw, th],  # x3 -> enc1's input
        "l_dec_project": [n["dec1.out"], n["l.narrow"], w["dec0.project"], tw, th],      # dec1's output -> y
        # y with the skip S = x3 -> y2
        "l_dec_block": block_args(route, n["l.narrow"], n["l.x3"], n["l.wide"],
                                  w["dec0_block.q7" if hip else "dec0_block"], tab, fh, fw, do),
        # the native hybrid route (speed lever 3): q in the narrow map's 20-byte form (lev/input_encoder_q20: the same
        # arguments, q -> l.narrow), the encoder block with the linears as its prologue (x2 neither written nor read),
        # the decoder block with the pack12 store as its epilogue (y2 not stored; A / C its pack12 images)
        **({"l_enc_block_lin": [n["l.narrow"], "0", n["l.x3"], w["enc0_block"], str(fh), str(fw)] + eo +
                                [w["enc0.Win1"], w["enc0.Win2"]],
            "l_dec_block_coef": [n["l.narrow"], n["l.x3"], "0", w["dec0_block"], str(fh), str(fw)] + do +
                                [n["dec0.A"], n["dec0.C"], n["dec0.latent"], w["dec0.coef"], str(info["rw"]),
                                 str(info["rh"])]} if fused else {}),
        # (l_coef_pack12: the same arguments; A and C its pack12 images)
        "l_coef": [n["l.wide"], n["dec0.A"], n["dec0.B"], n["dec0.C"], n["dec0.latent"], w["dec0.coef"], None],
        # the reconstruction (Model M's post_recon arguments): colour, A, B, C, the previous history, this frame's
        # motion surfaces, the post blob, the new history, the reduced history, the display image, P, exposure pair
        "l_post_recon": [n["input.color"], n["dec0.A"], n["dec0.B"], n["dec0.C"], n["state.history_prev"],
                         n["enc0.m_near"], n["enc0.m_center"], n["enc0.m_far"], w["post"], n["post.history"],
                         n["post.reduced"], n["post.display"], None, n["exposure.pair"]],
    }


def base(kernel):
    """A Model L launch's role: the blocks of both routes as l_enc_block / l_dec_block, the pack12 coefficient store
    as l_coef (whose arguments it shares), every form of the reconstruction (C1/C2 lc12/, pack12, LDR) as
    l_post_recon."""
    if kernel == ENC_LIN_HYB:
        return "l_enc_block_lin"
    if kernel == DEC_COEF_HYB:
        return "l_dec_block_coef"
    if kernel == "lev/input_encoder_q20":
        return "lev/input_encoder"
    if kernel in ENC_BLOCK.values():
        return "l_enc_block"
    if kernel in DEC_BLOCK.values():
        return "l_dec_block"
    k = kernel[len("lc12/"):] if kernel.startswith("lc12/") else kernel
    post = ("l_post_recon", "l_post_recon_p12", "l_post_recon_ldr", "l_post_recon_p12_ldr")
    return "l_coef" if k == "l_coef_pack12" else "l_post_recon" if k in post else kernel


def model_m_post(names):
    """The launches of Model M's post route among a frame's kernel names: its reconstruction kernels (post_recon*,
    post_tf*, post_fused*, the pmo/ motion-at-output forms and the banded forms), which Model L never runs."""
    return [n for n in names if n.split("/")[-1].startswith(("post_recon", "post_tf", "post_fused", "post_prep"))
            or n.startswith("pmo/") or n.endswith("_band")]


def check_roles(t, f, digests, route, l_post=True):
    """Raises AssertionError naming every Model L launch argument that is not the expected buffer or blob, every role
    not launched, and any M-form L block or fusion launched."""
    exp = expected_args(t, f, digests, route)
    seen, bad = set(), []
    for ln in f["launches"]:
        k = base(ln.kernel)
        if ln.kernel in M_FORM_L:
            bad.append("%s launched (the M-form L route is not offered)" % ln.kernel)
        if k not in exp:
            continue
        seen.add(k)
        want = exp[k]
        if len(ln.args) != len(want):
            bad.append("%s: %d arguments, expected %d" % (k, len(ln.args), len(want)))
            continue
        bad += ["%s argument %d: %s, expected %s" % (k, i, a, e) for i, (a, e) in enumerate(zip(ln.args, want))
                if e is not None and a != e]
    names = [ln.kernel for ln in f["launches"]]
    need = L_KERNELS
    if ENC_LIN_HYB in names:   # the native hybrid route (lever 3): no separate linears; the fold on pack12 frames
        need = tuple(k for k in L_KERNELS if k not in ("l_linear_32_32_relu", "l_linear_32_32_f16", "l_enc_block"))
        need += ("l_enc_block_lin",)
        if DEC_COEF_HYB in names:
            need = tuple(k for k in need if k not in ("l_dec_block", "l_coef")) + ("l_dec_block_coef",)
    missing = [k for k in need if k not in seen and (l_post or k != "l_post_recon")]
    if bad or missing:
        raise AssertionError("; ".join(bad + ["not launched: " + k for k in missing]))


def shared_stages(t, f, route, digests=None):
    """The ten shared stages between l_project_gather and l_dec_project, argument for argument (AssertionError):
    HIP the FP16-form kernels (l_q7_stages: each block one 4 x 4 window per workgroup of 32 lanes, its "<s>.q7" blob
    and the tables; the merges and expansions their "<s>.q7lin" blob), D3D12 the twins on the unfused layout (Model M's
    block arguments and grids). The maps by the runtime's names; the offsets from the reference table
    (clean/reference/swin.py window_offset); with digests, every FP16-form blob by its bytes."""
    ls, n, info = f["launches"], f["names"], f["info"]
    fw, fh = cdiv(info["rw"], 32) * 32, cdiv(info["rh"], 32) * 32
    H2, W2, H4, W4, H8, W8 = fh // 2, fw // 2, fh // 4, fw // 4, fh // 8, fw // 8
    names = [ln.kernel for ln in ls]
    a, b = names.index("l_project_gather"), names.index("l_dec_project")
    got = ls[a + 1:b]
    par = info["parity"]
    bouts = [n["b%d.out" % i] for i in range(6)]

    def off(stage):
        return [str(v) for v in SW.window_offset(stage, par)]

    w = (lambda blob: t.uploaded(digests[blob])) if digests else (lambda blob: None)
    if route == "hip":
        tab = w("q7.tab")

        def blk(k, x, skip, out, stage, H, W):
            ox, oy = SW.window_offset(stage, par)
            return (k, cdiv(W + ox, 4) * cdiv(H + oy, 4), 32, [x, skip, out, w(stage + ".q7"), tab, str(H), str(W)] +
                    off(stage))

        def lin(k, x, out, stage, h, w_, grid):
            return (k, grid, 32, [x, out, w(stage + ".q7lin"), str(h), str(w_)])

        e2 = got[10].args[1] if len(got) > 10 and got[10].args else None   # the expansions' maps (no name)
        e1 = got[12].args[1] if len(got) > 12 and got[12].args else None
        want = [blk("l_q7_block_c64", n["enc0.features"], "0", n["enc1.skip"], "enc1", H2, W2),
                lin("l_q7_merge_c64", n["enc1.skip"], n["enc1.merged"], "enc1", H2, W2, (H2 // 2) * (W2 // 2)),
                blk("l_q7_block_c96", n["enc1.merged"], "0", n["enc2.skip"], "enc2", H4, W4),
                lin("l_q7_merge_c96", n["enc2.skip"], n["enc2.merged"], "enc2", H4, W4, (H4 // 2) * (W4 // 2))]
        x = n["enc2.merged"]
        for i in range(6):
            want.append(blk("l_q7_block_c128", x, "0", bouts[i], "b%d" % i, H8, W8))
            x = bouts[i]
        want += [lin("l_q7_expand_c128", bouts[5], e2, "dec2", H8, W8, H8 * W8),
                 blk("l_q7_block_c96_dec", e2, n["enc2.skip"], n["dec2.out"], "dec2", H4, W4),
                 lin("l_q7_expand_c96", n["dec2.out"], e1, "dec1", H4, W4, H4 * W4),
                 blk("l_q7_block_c64_dec", e1, n["enc1.skip"], n["dec1.out"], "dec1", H2, W2)]
        if e2 in (None, "0") or e1 in (None, "0") or e1 == e2 or {e1, e2} & set(n.values()):
            raise AssertionError("the expansions' maps: %s, %s" % (e2, e1))
    else:
        def blk(k, x, skip, out, stage, H, W):
            ox, oy = SW.window_offset(stage, par)
            return (k, cdiv(cdiv(W + ox, 4) * cdiv(H + oy, 4), 8), 256, [x, skip, out, w(stage), str(H), str(W)] +
                    off(stage))

        want = [blk("swin_block_c64_hyb", n["enc0.features"], "0", n["enc1.skip"], "enc1", H2, W2),
                ("patch_merge_c64_hyb", None, None, None),
                blk("swin_block_c96_hyb", n["enc1.merged"], "0", n["enc2.skip"], "enc2", H4, W4),
                ("patch_merge_c96_hyb", None, None, None)]
        # b0 ... b5: one launch (speed lever 4) of 128 workgroups; enc2's merged map in, b5.out out (tmp and the
        # sync words: the runtime's scratch, not named), the six blobs, the even (b0) and odd (b1) blocks' offsets
        want.append((BNECK_HYB, 128, 256, [n["enc2.merged"], bouts[5], None, None] + [w("b%d" % i) for i in range(6)] +
                     [str(H8), str(W8)] + off("b0") + off("b1") + ["128"]))
        if EX_HYB[0] in names:   # the fused decoder blocks (WG_L_EXFUSE): x = the low-resolution map, aux null
            want += [blk(EX_HYB[0], bouts[5], n["enc2.skip"], n["dec2.out"], "dec2", H4, W4),
                     blk(EX_HYB[1], n["dec2.out"], n["enc1.skip"], n["dec1.out"], "dec1", H2, W2)]
            want = [(k, g, b_, a_ + ["0"] if k in EX_HYB else a_) for k, g, b_, a_ in want]
        else:
            want += [("expand_c128_hyb", None, None, None),
                     blk("swin_block_c96_add_hyb", None, n["enc2.skip"], n["dec2.out"], "dec2", H4, W4),
                     ("expand_c96_hyb", None, None, None),
                     blk("swin_block_c64_add_hyb", None, n["enc1.skip"], n["dec1.out"], "dec1", H2, W2)]
    bad = []
    if [ln.kernel for ln in got] != [k for k, _, _, _ in want]:
        raise AssertionError("shared stages: %s, expected %s" % ([ln.kernel for ln in got], [k for k, *_ in want]))
    for i, (ln, (k, grid, block, args)) in enumerate(zip(got, want)):
        if grid is not None and (ln.grid, ln.block) != (grid, block):
            bad.append("%s (launch %d): grid %s x %s, expected %s x %s" % (k, i, ln.grid, ln.block, grid, block))
        if args is not None:
            if len(ln.args) != len(args):
                bad.append("%s (launch %d): %d arguments, expected %d" % (k, i, len(ln.args), len(args)))
                continue
            bad += ["%s (launch %d) argument %d: %s, expected %s" % (k, i, j, a_, e) for j, (a_, e) in
                    enumerate(zip(ln.args, args)) if e is not None and a_ != e]
    if route == "d3d12" and EX_HYB[0] not in names:   # the maps the twin expansions write (no name, arguments not recorded)
        e2, e1 = got[6].args[0], got[8].args[0]
        if e2 == e1 or {e2, e1} & ({"0"} | set(n.values())):
            bad.append("the expanded maps: %s, %s" % (e2, e1))
    if bad:
        raise AssertionError("; ".join(bad))


def declared_ports():
    """[(kernel, code object)] of every D3D12 port declared in clean/dx12/kernels (the "@wg-kernel" lines that
    clean/dx12/build_dx12.py builds into the manifest): what a full build's manifest names."""
    out = []
    for path in sorted(glob.glob(os.path.join(host_cxx.ROOT, "clean", "dx12", "kernels", "*.hlsl"))):
        with open(path, encoding="utf-8") as fh:
            for m in re.finditer(r"@wg-kernel\s+(\S+)\s+co=(\S+)", fh.read()):
                if m.groups() not in out:
                    out.append(m.groups())
    return out


# The hybrid route's D3D12 kernels the runtime checks at create besides the twins and the reconstruction
# (runtime.cpp l_native_missing), with their code objects
L_D3D12 = [("lev/input_encoder", "l_enc_values.co"), ("l_linear_32_32_relu", "l_ops.co"),
           ("l_linear_32_32_f16", "l_ops.co"), ("l_project_gather", "l_ops.co"), ("l_dec_project", "l_ops.co"),
           ("l_coef_pack12", "l_ops.co")]
HYB_TWINS = ("l_swin_block_c32_enc_hyb", "l_swin_block_c32_dec_hyb", "swin_block_c64_hyb", "patch_merge_c64_hyb",
             "swin_block_c96_hyb", "patch_merge_c96_hyb", "swin_block_c128_hyb", "expand_c128_hyb",
             "swin_block_c96_add_hyb", "expand_c96_hyb", "swin_block_c64_add_hyb")
LC12 = ("lc12/l_post_recon", "lc12/l_post_recon_p12", "lc12/l_post_recon_ldr", "lc12/l_post_recon_p12_ldr")


def full_ports():
    """The declared ports plus any of Model L's not declared yet: a build where every port Model L needs has landed.
    (Model M's wave64 post ports come in both filter builds, keyed by name and code object: Model L's file has no
    output-filter weights, so it takes the analytic one, post_w64.co.)"""
    have = declared_ports()
    return have + [kc for kc in L_D3D12 if kc[0] not in [k for k, _ in have]]


class Ports(unittest.TestCase):
    def test_analytic_post_declared(self):
        """The analytic-filter build of Model M's fused pack12 post (the post Model L's native frame runs) is among the
        declared ports."""
        have = declared_ports()
        for k in ("w64/post_tf_f32_p12_diet_vsit", "w64/post_tf4_dl_f32_p12_diet_vsit"):
            self.assertIn((k, "post_w64.co"), have)

    def test_release_route_declared(self):
        """Every port of the release route (the hybrid twins, the C1/C2 reconstruction, Model L's own kernels and its
        filters) is declared: a full build's manifest has them all."""
        have = [k for k, _ in declared_ports()]
        need = [k for k, _ in L_D3D12] + list(HYB_TWINS) + list(LC12) + ["l_coef", "w64/post_filter_c1_s4",
                                                                          "l_post_filter_ldr"]
        self.assertEqual([k for k in need if k not in have], [])


def cdiv(a, b):
    return -(-a // b)


def expected_l(m_frame, parity, fh, fw):
    """Model M's launches of one frame (on the unbanded post_recon route) with Model L's steps 2 to 5 on the HIP
    reference route (clean/l/FORMATS.md, "Kernels" and "Launch order of one frame"; the FP16-form kernels): kernel,
    grid, block. Model M's input encoder -> Model L's values, linears, encoder block and gather; Model M's ten shared
    stages -> the FP16-form stages (l_q7_stages); its decoder -> Model L's projection, decoder block and l_coef; its
    post_recon -> the C1/C2 reconstruction (same arguments and grid)."""
    npix, tokens = (fh + 4) * (fw + 4), (fh // 2) * (fw // 2)
    H2, W2, H4, W4, H8, W8 = fh // 2, fw // 2, fh // 4, fw // 4, fh // 8, fw // 8

    def nwin(H, W, ox, oy):
        return cdiv(W + ox, 4) * cdiv(H + oy, 4)

    def lblock(k, offsets):
        return (k, nwin(fh, fw, *T.block_offset(T.TABLE, offsets, parity)), 32)

    def sblock(k, stage, H, W):
        return (k, nwin(H, W, *SW.window_offset(stage, parity)), 32)

    stages = [sblock("l_q7_block_c64", "enc1", H2, W2), ("l_q7_merge_c64", (H2 // 2) * (W2 // 2), 32),
              sblock("l_q7_block_c96", "enc2", H4, W4), ("l_q7_merge_c96", (H4 // 2) * (W4 // 2), 32)] + \
        [sblock("l_q7_block_c128", "b%d" % i, H8, W8) for i in range(6)] + \
        [("l_q7_expand_c128", H8 * W8, 32), sblock("l_q7_block_c96_dec", "dec2", H4, W4),
         ("l_q7_expand_c96", H4 * W4, 32), sblock("l_q7_block_c64_dec", "dec1", H2, W2)]
    out, in_stages = [], False
    for k, g, b in m_frame:
        if k == "input_encoder":
            out += [("lev/input_encoder", cdiv(((fh + 4) // 2) * ((fw + 4) // 2), 128), 256),
                    ("l_linear_32_32_relu", cdiv(npix, 128), 256), ("l_linear_32_32_f16", cdiv(npix, 128), 256),
                    lblock("l_q7_block_c32_enc", "first"), ("l_project_gather", cdiv(tokens, 128), 256)] + stages
            in_stages = True   # Model M's shared stages follow, up to its decoder
        elif k == "coef_decoder":
            out += [("l_dec_project", cdiv(tokens, 128), 256), lblock("l_q7_block_c32_dec", "second"),
                    ("l_coef", cdiv(tokens, 128), 256)]
            in_stages = False
        elif in_stages:
            continue
        elif k.split("/")[-1] == "post_recon":  # Model L's C1/C2 reconstruction, same arguments and grid
            out.append(("lc12/l_post_recon", g, b))
        else:
            out.append((k, g, b))
    return out


class Frame(unittest.TestCase):
    """The HIP release route (the scalar hybrid_full reference) launch for launch against Model M's frame."""

    def check(self, build_files=(), m_env=None):
        # Model M on the route Model L's reconstruction takes: the unbanded post_recon, then the unbanded filter
        fh, fw = cdiv(RH, 32) * 32, cdiv(RW, 32) * 32
        cm, lm, fm, km = FT.parse(run(build_files=build_files, env=dict(m_env or {}, WG_POST_KERNEL="post_recon")))
        cl, ll, fl, kl = FT.parse(run("model=L", build_files=build_files))
        self.assertEqual(cm, "create ok model M")
        self.assertEqual(cl, "create ok model L")
        self.assertEqual(len(fl), 4)
        for i, (m, l_) in enumerate(zip(fm, fl)):
            names = [k for k, _, _ in m]
            self.assertIn("input_encoder", names)
            self.assertIn("coef_decoder", names)  # Model M's base decoder: the one Model L's l_coef stands in for
            self.assertEqual(sum(k.split("/")[-1] == "post_recon" for k in names), 1)
            self.assertEqual(l_, expected_l(m, i & 1, fh, fw), "frame %d" % i)
        l_mods = ["l_enc_values.co as lev", "l_ops.co", "l_swin.co", "l_post.co", "l_q7__hyb.co", "l_q7w__hyb.co",
                  "l_post__lc12.co as lc12"]
        self.assertEqual([x for x in ll if x not in l_mods], [x for x in lm if x not in (km["enc0"], km["dec0"])])
        for want in l_mods:
            self.assertIn(want, ll)
        self.assertEqual(kl.get("model"), "L")
        self.assertEqual(kl.get("l_form"), "hybrid")
        self.assertEqual(kl.get("l_post_c12"), "on")
        self.assertEqual(kl.get("post_pack"), "none")
        self.assertEqual(kl.get("model_map"), "same")  # read from model.wgcm, checked against its map
        return fl

    def test_empty_build(self):
        self.check()

    def test_shipping_build(self):
        # l_coef writes Model M's base coefficient format: Model L on HIP runs without pack12 (and so without the fused
        # pack12 dec1 + dec0), compared with Model M without it (WG_POST_PACK=none)
        fl = self.check(FT.SHIPPING, {"WG_POST_PACK": "none"})
        names = [k for k, _, _ in fl[0]]
        self.assertTrue(all(not n.startswith(("d0p12/", "d1p12/")) and "_p12" not in n for n in names), names)
        self.assertTrue(any(n.startswith("w64/") for n in names))  # the wave64 post (not a pack12 kernel) is kept

    def test_model_m_unchanged(self):
        """Model M in the build with Model L: the same trace as in the build without it (clean/tests)."""
        for files in ((), FT.SHIPPING):
            self.assertEqual(run(frames=2, build_files=files), run(frames=2, build_files=files, exe=FT.trace_exe()))


class Arguments(unittest.TestCase):
    """The recorder's view of every Model L launch (clean/l/FORMATS.md, "Maps" and "Kernels") on the release routes."""

    def check_frame_fused(self, t, f, roles=True, l_post=True):
        """check_frame for the native hybrid route with its fused twins (speed lever 3): q (20-byte form) -> the encoder
        block with the linears -> x3; y with the skip x3 -> the decoder block with the pack12 store (pack12 frames) or
        the decoder block -> y2 -> l_coef (other frames). Returns (lev, coef) with coef the launch holding A / C /
        latent at the l_coef positions 1 / 3 / 4 (a stand-in object for the folded store)."""
        info, ls = f["info"], f["launches"]
        rw, rh = info["rw"], info["rh"]
        fw, fh = cdiv(rw, 32) * 32, cdiv(rh, 32) * 32
        npix, tw, th = (fh + 4) * (fw + 4), fw // 2, fh // 2
        names = [ln.kernel for ln in ls]
        self.assertEqual([n for n in names if n in M_FORM_L], [])
        self.assertEqual([n for n in names if n in ("l_linear_32_32_relu", "l_linear_32_32_f16", LIN)], [])
        k = {base(ln.kernel): (i, ln) for i, ln in enumerate(ls)}
        _, lev = k["lev/input_encoder"]
        self.assertEqual(lev.kernel, "lev/input_encoder_q20")
        _, enc = k["l_enc_block_lin"]
        gi, gather = k["l_project_gather"]
        di, dproj = k["l_dec_project"]
        q, x3, feat = lev.args[6], enc.args[2], gather.args[1]
        self.assertEqual(enc.args[0], q)       # the block reads q itself
        self.assertEqual(enc.args[1], "0")
        self.assertEqual(gather.args[0], x3)
        self.assertEqual(ls[gi + 1].kernel, "swin_block_c64_hyb")
        self.assertEqual(ls[gi + 1].args[0], feat)
        dec1 = ls[di - 1]
        self.assertIn(dec1.kernel, ("swin_block_c64_add_hyb", EX_HYB[1]))
        y = dproj.args[1]
        self.assertEqual(dproj.args[0], dec1.args[2])
        fold = "l_dec_block_coef" in k
        if fold:
            _, dec = k["l_dec_block_coef"]
            self.assertEqual((dec.args[0], dec.args[1], dec.args[2]), (y, x3, "0"))
            A, C, L = dec.args[8], dec.args[9], dec.args[10]
            self.assertEqual(dec.args[12:14], [str(rw), str(rh)])

            class _Coef:
                args = [None, A, None, C, L]
            coef = _Coef()
            self.assertGreaterEqual(t.extent(A), 8 * rw * rh)
            self.assertGreaterEqual(t.extent(C), 4 * rw * rh)
            self.assertGreaterEqual(t.extent(L), 4 * rw * rh)
        else:
            _, dec = k["l_dec_block"]
            _, coef = k["l_coef"]
            self.assertEqual((dec.args[0], dec.args[1]), (y, x3))
            self.assertEqual(coef.args[0], dec.args[2])
        for a_, b_ in ((q, x3), (x3, feat), (dec1.args[2], y), (x3, y)):
            self.assertNotEqual(a_, b_)
        self.assertGreaterEqual(t.extent(q), 20 * npix)   # the q20 map: 16 + 4 bytes a pixel (two planes)
        self.assertGreaterEqual(t.extent(x3), 32 * fh * fw)
        self.assertEqual(gather.args[3:5], [str(tw), str(th)])
        self.assertEqual(dproj.args[3:5], [str(tw), str(th)])
        for blk, kind in ((enc, "first"), (dec, "second")):
            ox, oy = T.block_offset(T.TABLE, kind, info["parity"])
            self.assertEqual(blk.args[4:8], [str(fh), str(fw), str(ox), str(oy)])
            nwin = cdiv(fw + ox, 4) * cdiv(fh + oy, 4)
            self.assertEqual((blk.grid, blk.block), (cdiv(nwin, 8), 256))
        e = lev.params(12)
        self.assertEqual(e[:4], [rw, rh, fw, fh])
        self.assertEqual(e[5], info["reset"])
        post = [ln for ln in ls if base(ln.kernel) == "l_post_recon"]
        self.assertEqual(len(post), 1 if l_post else 0)
        if l_post:
            self.assertTrue(post[0].kernel.startswith("lc12/"), post[0].kernel)
        digests = blob_digests("d3d12") if roles else None
        shared_stages(t, f, "d3d12", digests)
        if roles:
            check_roles(t, f, digests, "d3d12", l_post)
        return lev, coef

    def check_frame(self, t, f, roles=True, l_post=True):
        route = route_of(t)
        if route == "d3d12" and any(ln.kernel == ENC_LIN_HYB for ln in f["launches"]):
            return Arguments.check_frame_fused(self, t, f, roles, l_post)
        info, ls = f["info"], f["launches"]
        rw, rh = info["rw"], info["rh"]
        fw, fh = cdiv(rw, 32) * 32, cdiv(rh, 32) * 32
        npix, tw, th = (fh + 4) * (fw + 4), fw // 2, fh // 2
        names = [ln.kernel for ln in ls]
        self.assertEqual([n for n in names if n in M_FORM_L], [])   # no M-form L block, no fusion on it
        k = {base(ln.kernel): (i, ln) for i, ln in enumerate(ls)}
        _, lev = k["lev/input_encoder"]
        _, relu = k["l_linear_32_32_relu"]
        _, f16 = k["l_linear_32_32_f16"]
        _, enc = k["l_enc_block"]
        gi, gather = k["l_project_gather"]
        di, dproj = k["l_dec_project"]
        _, dec = k["l_dec_block"]
        _, coef = k["l_coef"]
        self.assertEqual((enc.kernel, dec.kernel), (ENC_BLOCK[route], DEC_BLOCK[route]))
        h = 5 if route == "hip" else 4   # the block's H (the FP16-form blocks take the tables after w)
        q, x3, feat = lev.args[6], enc.args[2], gather.args[1]
        # the chain of maps (no launch reads what it writes)
        x1, x2 = relu.args[1], f16.args[1]
        self.assertEqual(relu.args[0], q)
        self.assertEqual(f16.args[0], x1)
        self.assertEqual(relu.args[3], str(npix))
        self.assertEqual(f16.args[3], str(npix))
        self.assertEqual(enc.args[0], x2)
        self.assertEqual(enc.args[1], "0")  # the encoder block: no skip
        self.assertEqual(gather.args[0], x3)
        self.assertEqual(ls[gi + 1].kernel, "l_q7_block_c64" if route == "hip" else "swin_block_c64_hyb")
        self.assertEqual(ls[gi + 1].args[0], feat)  # enc1 reads Model L's features
        dec1 = ls[di - 1]   # dec1's block, right before the decoder
        self.assertIn(dec1.kernel, ("l_q7_block_c64_dec",) if route == "hip" else ("swin_block_c64_add_hyb", EX_HYB[1]))
        y, y2 = dproj.args[1], dec.args[2]
        self.assertEqual(dproj.args[0], dec1.args[2])  # the decoder reads dec1's output
        self.assertEqual(dec.args[0], y)
        self.assertEqual(dec.args[1], x3)  # the decoder block's skip S: the encoder block's map
        self.assertEqual(coef.args[0], y2)
        for a, b in ((q, x1), (x1, x2), (x2, x3), (x3, feat), (dec1.args[2], y), (y, y2), (x3, y), (x3, y2)):
            self.assertNotEqual(a, b)
        self.assertEqual(q, x2)  # x2 over q (q is dead once x1 is written)
        # every buffer holds its map (FORMATS.md "Maps")
        maps = ((q, 32 * npix), (x1, 32 * npix), (x2, 64 * npix), (x3, 32 * fh * fw), (feat, 64 * tw * th),
                (y, 32 * fh * fw), (y2, 64 * fh * fw))
        p12 = coef.kernel == "l_coef_pack12"   # pack12 image A: 8 bytes a pixel, C 4; Model M's base: A 4, C 8
        coefs = ((coef.args[1], (8 if p12 else 4) * rw * rh), (coef.args[2], 4 * rw * rh),
                 (coef.args[3], (4 if p12 else 8) * rw * rh), (coef.args[4], 4 * rw * rh))
        for buf, need in maps + coefs:
            self.assertGreaterEqual(t.extent(buf), need, buf)
        # extents, offsets, parameters
        self.assertEqual(gather.args[3:5], [str(tw), str(th)])
        self.assertEqual(dproj.args[3:5], [str(tw), str(th)])
        for blk, kind in ((enc, "first"), (dec, "second")):
            ox, oy = T.block_offset(T.TABLE, kind, info["parity"])
            self.assertEqual(blk.args[h:h + 4], [str(fh), str(fw), str(ox), str(oy)])
            nwin = cdiv(fw + ox, 4) * cdiv(fh + oy, 4)
            self.assertEqual((blk.grid, blk.block), (nwin, 32) if route == "hip" else (cdiv(nwin, 8), 256))
        if route == "hip":
            self.assertEqual(enc.args[4], dec.args[4])  # the FP16-form tables
            self.assertTrue(enc.args[4].startswith("b"))
        e = lev.params(12)
        self.assertEqual(e[:4], [rw, rh, fw, fh])
        self.assertEqual(e[5], info["reset"])
        self.assertEqual(coef.params(6), [rw, rh, fw, fh, info["reset"]])
        post = [ln for ln in ls if base(ln.kernel) == "l_post_recon"]
        self.assertEqual(len(post), 1 if l_post else 0)
        if l_post:
            self.assertTrue(post[0].kernel.startswith("lc12/"), post[0].kernel)   # the C1/C2 reconstruction
            self.assertEqual(post[0].params(12), [rw, rh, (3 * 120 + 1) // 2, (3 * 72 + 1) // 2, info["reset"]])
            self.assertEqual(post[0].grid, cdiv((3 * 120 + 1) // 2, 32) * cdiv(cdiv((3 * 72 + 1) // 2, 2), 8))
            nxt = ls[ls.index(post[0]) + 1].kernel   # the output filter next (LDR: Model L's LDR output stage)
            self.assertTrue(nxt.endswith("post_filter_c1_s4") or nxt == "l_post_filter_ldr", nxt)
        self.assertTrue(lev.args[11].startswith("b"))  # the values blob: an allocation
        digests = blob_digests(route) if roles else None
        shared_stages(t, f, route, digests)
        if roles:  # every argument against the named buffers and blobs
            check_roles(t, f, digests, route, l_post)
        return lev, coef

    def test_sizes_reset_history(self):
        sizes = "--sizes=96x96,96x96,80x48,80x48,80x48,64x64"  # aligned, non-aligned, a smaller aligned size
        t = FT.Trace(run("model=L", frames=6, rw=96, rh=96, flags=(sizes, "--reset=4")))
        self.assertEqual(t.create, "create ok model L")
        self.assertEqual(t.unknown(), [])
        # first frame, a size change, the game's reset, a size change
        self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 1, 0, 1, 1])
        prev = None
        for f in t.frames:
            self.assertIsNone(f["error"])
            lev, coef = self.check_frame(t, f)
            if prev:  # the latent l_coef wrote and the near motion the encoder wrote: what this frame's encoder reads
                self.assertEqual(lev.args[4], prev[1].args[4])
                self.assertEqual(lev.args[5], prev[0].args[7])
            prev = (lev, coef)

    def test_steady_frames(self):
        t = FT.Trace(run("model=L", frames=4))
        self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 0, 0])
        self.assertEqual(len({f["info"]["parity"] for f in t.frames}), 2)
        prev = None
        for f in t.frames:
            lev, coef = self.check_frame(t, f)
            if prev:
                self.assertEqual(lev.args[4], prev[1].args[4])
                self.assertNotEqual(lev.args[4], prev[0].args[4])  # the two latent buffers swap
            prev = (lev, coef)

    def test_failed_launch_stops_the_frame(self):
        for kernel, before in (("l_coef", "l_q7_block_c32_dec"), ("l_q7_block_c128", "l_q7_merge_c96"),
                               ("lc12/l_post_recon", "l_coef")):
            with self.subTest(kernel=kernel):
                t = FT.Trace(run("model=L", frames=3, flags=("--fail=%s@1" % kernel,)))
                ok0, bad, ok2 = t.frames
                self.assertEqual(bad["failed"], kernel)
                self.assertIn("injected launch failure: " + kernel, bad["error"])
                names = [ln.kernel for ln in bad["launches"]]
                self.assertEqual(names, [ln.kernel for ln in ok0["launches"]][:len(names)])
                self.assertEqual(names[-1], before)  # stopped right before the failing launch
                self.assertIsNone(ok2["error"])
                self.check_frame(t, ok2)


class Roles(unittest.TestCase):
    """Buffer identities: each Model L launch's weight blob, inputs and outputs are the named ones, and a mix-up of two
    of them fails even where the map chain, sizes and argument kinds stay as they were; on both release routes."""

    MIXUPS = (  # (kernel role, argument, kernel role, argument) swapped in a recorded frame
        ("l_linear_32_32_relu", 2, "l_linear_32_32_f16", 2),        # enc0.Win1 <-> enc0.Win2
        ("l_enc_block", 3, "l_dec_block", 3),                       # the encoder's block blob <-> the decoder's
        ("l_enc_block", 1, "l_dec_block", 1),                       # no skip <-> the decoder's skip S (x3)
        ("l_dec_block", 0, "l_dec_block", 1),                       # y <-> the skip S
        ("l_project_gather", 2, "l_dec_project", 2),                # enc0.project <-> dec0.project
        ("lev/input_encoder", 11, "l_coef", 5),                     # enc0.values <-> dec0.coef
        ("l_coef", 1, "l_coef", 2),                                 # A <-> B
        ("l_coef", 3, "l_coef", 4),                                 # C <-> the new latent
        ("lev/input_encoder", 0, "lev/input_encoder", 1),           # colour <-> depth
        ("lev/input_encoder", 4, "lev/input_encoder", 5),           # previous latent <-> motion history
        ("lev/input_encoder", 7, "lev/input_encoder", 8),           # m_near <-> m_far
        ("l_linear_32_32_relu", 0, "l_linear_32_32_relu", 1),       # q's map <-> x1's map
        ("l_post_recon", 4, "l_post_recon", 9),                     # history read <-> history written
        ("l_post_recon", 8, "l_coef", 5),                           # post blob <-> dec0.coef
        ("l_post_recon", 1, "l_post_recon", 3),                     # A <-> C
    )
    Q7_MIXUPS = (("l_enc_block", 4, "l_dec_block", 3),)              # the tables <-> the decoder's block blob
    # the native route's fused twins (speed lever 3): the same mix-ups on their arguments (the encoder block with the
    # linears: q, -, x3, block blob, H, W, ox, oy, Win1, Win2; the decoder block with the store: y, S, -, block blob,
    # H, W, ox, oy, A, C, latent, dec0.coef, rw, rh)
    FUSED_MIXUPS = (
        ("l_enc_block_lin", 8, "l_enc_block_lin", 9),               # enc0.Win1 <-> enc0.Win2
        ("l_enc_block_lin", 3, "l_dec_block_coef", 3),              # the encoder's block blob <-> the decoder's
        ("l_enc_block_lin", 1, "l_dec_block_coef", 1),              # no skip <-> the decoder's skip S (x3)
        ("l_dec_block_coef", 0, "l_dec_block_coef", 1),             # y <-> the skip S
        ("l_project_gather", 2, "l_dec_project", 2),                # enc0.project <-> dec0.project
        ("lev/input_encoder", 11, "l_dec_block_coef", 11),          # enc0.values <-> dec0.coef
        ("l_dec_block_coef", 8, "l_dec_block_coef", 9),             # A <-> C
        ("l_dec_block_coef", 9, "l_dec_block_coef", 10),            # C <-> the new latent
        ("lev/input_encoder", 0, "lev/input_encoder", 1),           # colour <-> depth
        ("lev/input_encoder", 4, "lev/input_encoder", 5),           # previous latent <-> motion history
        ("lev/input_encoder", 7, "lev/input_encoder", 8),           # m_near <-> m_far
        ("l_enc_block_lin", 0, "l_enc_block_lin", 2),               # q's map <-> x3's map
        ("l_post_recon", 4, "l_post_recon", 9),                     # history read <-> history written
        ("l_post_recon", 8, "l_dec_block_coef", 11),                # post blob <-> dec0.coef
        ("l_post_recon", 1, "l_post_recon", 3),                     # A <-> C
    )

    @classmethod
    def setUpClass(cls):
        sizes = ("--sizes=96x64,80x48,80x48",)
        cls.traces = {"hip": FT.Trace(run("model=L", frames=3, rw=96, rh=64, flags=sizes)),
                      "d3d12": FT.Trace(run("model=L", frames=3, rw=96, rh=64, build_files=FT.SHIPPING,
                                            flags=("--backend=3",) + sizes, manifest=full_ports()))}

    def test_every_argument_named(self):
        for route, t in self.traces.items():
            with self.subTest(route=route):
                self.assertEqual(route_of(t), route)
                self.assertEqual(t.unknown(), [])
                digests = blob_digests(route)
                addrs = [t.uploaded(d) for d in digests.values()]
                self.assertNotIn(None, addrs)
                self.assertEqual(len(set(addrs)), len(addrs))  # distinct blobs, each in its own allocation
                for f in t.frames:
                    self.assertEqual(len(f["names"]) >= 30, True, sorted(f["names"]))
                    check_roles(t, f, digests, route)
                    shared_stages(t, f, route, digests)

    def test_mixups_fail(self):
        for route, t0 in self.traces.items():
            digests = blob_digests(route)
            mixups = self.FUSED_MIXUPS if route == "d3d12" else self.MIXUPS + self.Q7_MIXUPS
            for ka, ia, kb, ib in mixups:
                with self.subTest(route=route, mixup=(ka, ia, kb, ib)):
                    t = copy.deepcopy(t0)
                    f = t.frames[1]
                    ls = {base(ln.kernel): ln for ln in f["launches"]}
                    a, b = ls[ka], ls[kb]
                    a.args[ia], b.args[ib] = b.args[ib], a.args[ia]
                    with self.assertRaises(AssertionError):
                        check_roles(t, f, digests, route)

    def test_shared_stage_mixups_fail(self):
        """The shared stages' check catches a swapped map or blob (HIP: the FP16-form blobs by their bytes)."""
        for route, t0 in self.traces.items():
            digests = blob_digests(route)
            # (launch, argument) pairs after l_project_gather; the d3d12 route runs b0 ... b5 as one launch (speed lever
            # 4: positions 0 c64, 2 c96, 4 the bottleneck, 6 c96_add, 8 c64_add), swapped inside it too: its input and
            # output maps, b0's and b5's blobs
            swaps = (((0, 2, 2, 2), (4, 0, 5, 0), (11, 1, 13, 1), (4, 3, 5, 3), (0, 3, 13, 3)) if route != "d3d12" else
                     ((0, 2, 2, 2), (4, 0, 4, 1), (6, 1, 8, 1), (4, 4, 4, 9), (0, 3, 8, 3), (2, 2, 4, 1)))
            for i, ia, j, ib in swaps:
                with self.subTest(route=route, swap=(i, ia, j, ib)):
                    t = copy.deepcopy(t0)
                    f = t.frames[1]
                    names = [ln.kernel for ln in f["launches"]]
                    st = f["launches"][names.index("l_project_gather") + 1:]
                    a, b = st[i], st[j]
                    if not a.args or not b.args:   # a twin merge / expand: arguments not recorded
                        continue
                    a.args[ia], b.args[ib] = b.args[ib], a.args[ia]
                    if a.args == b.args or a.args[ia] == b.args[ib]:
                        continue
                    with self.assertRaises(AssertionError):
                        shared_stages(t, f, route, digests)

    def test_swapped_weights_in_the_runtime(self):
        """The review's example in the runtime itself: enc0.Win1 and enc0.Win2 swapped in its source. The map chain,
        sizes, grids and argument kinds stay as they were (the frame checks without the roles pass), and the role
        check fails on exactly the two linears' weights."""
        with open(os.path.join(host_cxx.ROOT, "clean", "runtime", "runtime.cpp"), encoding="utf-8") as fh:
            src = fh.read()
        right = 'w1 = wdev["enc0.Win1"], w2 = wdev["enc0.Win2"]'
        self.assertEqual(src.count(right), 1)
        tmp = tempfile.mkdtemp(prefix="wg_lmut_")
        mutant = os.path.join(tmp, "runtime.cpp")
        with open(mutant, "w", encoding="utf-8") as fh:
            fh.write(src.replace(right, 'w1 = wdev["enc0.Win2"], w2 = wdev["enc0.Win1"]'))
        sources = [mutant if s.replace("\\", "/").endswith("clean/runtime/runtime.cpp") else s
                   for s in FT.SOURCES] + L_SOURCES
        exe = host_cxx.build("runtime_frame_trace_lmut", sources, ["WG_NO_MODEL_K", "WG_MODEL_L"],
                             include_dirs=[os.path.join(host_cxx.ROOT, "clean", "runtime")])
        for route, kw in (("hip", {}), ("d3d12", dict(build_files=FT.SHIPPING, flags=("--backend=3",),
                                                      manifest=full_ports()))):
            with self.subTest(route=route):
                t = FT.Trace(run("model=L", frames=2, exe=exe, **kw))
                self.assertEqual(route_of(t), route)
                digests = blob_digests(route)
                for f in t.frames:
                    Arguments.check_frame(self, t, f, roles=False)
                    with self.assertRaises(AssertionError) as cm:
                        check_roles(t, f, digests, route)
                    msg = str(cm.exception)
                    if route == "d3d12":   # the native route's encoder block takes both blobs (speed lever 3)
                        self.assertIn("l_enc_block_lin argument 8", msg)
                        self.assertIn("l_enc_block_lin argument 9", msg)
                    else:
                        self.assertIn("l_linear_32_32_relu argument 2", msg)
                        self.assertIn("l_linear_32_32_f16 argument 2", msg)
                    self.assertEqual(msg.count("argument"), 2, msg)


class D3D12(unittest.TestCase):
    """Model L on native D3D12 (backend 3), the release route: Model M's native frame layout, unfused, with every Swin /
    merge / expand kernel its hybrid twin, Model L's encoder and decoder, the pack12 coefficient store and the C1/C2
    reconstruction (no Model M post route); each missing D3D12 port refused by name, never replaced."""

    def l_run(self, ports, frames=3, flags=(), **kw):
        return run("model=L", frames=frames, build_files=FT.SHIPPING, flags=("--backend=3",) + tuple(flags),
                   manifest=ports, **kw)

    def test_declared_build_creates(self):
        got = self.l_run(declared_ports(), frames=1)
        self.assertEqual(got[0], "create ok model L")

    def test_each_port_named(self):
        for k, _ in L_D3D12:
            with self.subTest(port=k):
                got = self.l_run([kc for kc in full_ports() if kc[0] != k], frames=1)
                self.assertEqual(got, ["ERR create: Model L on native D3D12: not yet ported to D3D12: " + k])

    def test_m_form_ports_not_needed(self):
        """The M-form L blocks, dec1's M-form block and the fused L kernels are not the release route's: a manifest
        without them creates and traces the same frames."""
        drop = ("l_swin_block_c32_enc_nc", "l_swin_block_c32_dec_nc", "swin_block_c64_add_ex_nc", FOLD, LIN, ENCLIN)
        a = same_frames(self.l_run(full_ports(), frames=2, rw=96, rh=64))
        b = same_frames(self.l_run([kc for kc in full_ports() if kc[0] not in drop], frames=2, rw=96, rh=64))
        self.assertEqual(a[0], "create ok model L")
        self.assertEqual(b, a)

    def check_d3d12_frame(self, t, f, sfx=""):
        """The Arguments checks (map chain, extents, offsets, parameters, every argument's buffer or blob, the shared
        stages, the C1/C2 reconstruction and the filter after it), then what the native frame adds: the twins only (no
        M-form Swin, merge or expand kernel), the pack12 store's images and lc12/l_post_recon_p12 right after it (a
        frame the pack12 kernels do not take: l_coef and lc12/l_post_recon), no Model M post kernel."""
        info = f["info"]
        rw, rh = info["rw"], info["rh"]
        names = [ln.kernel for ln in f["launches"]]
        fold = DEC_COEF_HYB in names   # pack12 frames: the decoder block with the store (speed lever 3)
        p12 = fold or "l_coef_pack12" in names
        self.assertEqual(int(fold) + int("l_coef_pack12" in names) + int("l_coef" in names), 1)   # one store
        self.assertNotIn("l_coef_pack12", names)   # (the native route folds every pack12 store)
        self.assertTrue(all(not n.startswith(("d1p12/", "d0p12/")) and n != "input_encoder" for n in names), names)
        stages = [n for n in names if n.split("/")[-1].startswith(("swin_", "l_swin_", "patch_merge", "expand"))]
        twins = TWIN_STAGES if EX_HYB[0] in names else TWIN_STAGES_UNFUSED   # (WG_L_EXFUSE on / off)
        self.assertEqual(stages, [ENC_LIN_HYB] + list(twins) + [DEC_COEF_HYB if fold else "l_swin_block_c32_dec_hyb"])
        self.assertEqual([n for n in names if "l_q7" in n], [])
        lev, coef = Arguments.check_frame(self, t, f)
        if p12:
            self.assertGreaterEqual(t.extent(coef.args[1]), 8 * rw * rh)  # pack12 image A: 8 bytes per pixel
            self.assertGreaterEqual(t.extent(coef.args[3]), 4 * rw * rh)  # pack12 image C: 4 bytes per pixel
        i = names.index(DEC_COEF_HYB if fold else "l_coef")
        self.assertEqual(names[i + 1], ("lc12/l_post_recon_p12" if p12 else "lc12/l_post_recon") + sfx, names[i + 1:])
        self.assertEqual(model_m_post(names), [])
        return lev, coef

    def test_frame(self):
        """Sizes aligned, non-aligned and changing, a game reset; history handed from frame to frame. The route is the
        unfused one whatever the fusion option asks or the manifest offers."""
        # upscaling sizes the pack12 post path takes (D = 180 x 108): aligned, non-aligned, a change of size
        sizes = "--sizes=96x64,96x64,80x48,80x48,80x48,77x45"
        lines = self.l_run(full_ports(), frames=6, rw=96, rh=64, flags=(sizes, "--reset=4"))
        t = FT.Trace(lines)
        self.assertEqual(t.create, "create ok model L")
        self.assertEqual(t.unknown(), [])
        self.assertEqual(t.kernels.get("swin_fused"), "dec2,dec1")   # WG_L_EXFUSE (default on): dec2, dec1 with expands fused
        self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 1, 0, 1, 1])
        self.assertEqual(len({f["info"]["parity"] for f in t.frames}), 2)
        prev = None
        for f in t.frames:
            self.assertIsNone(f["error"])
            self.assertIn(DEC_COEF_HYB, [ln.kernel for ln in f["launches"]])
            lev, coef = self.check_d3d12_frame(t, f)
            if prev:
                self.assertEqual(lev.args[4], latent_out(prev[1]))  # the latent the last frame's store wrote
                self.assertEqual(lev.args[5], prev[0].args[7])  # its near motion
            prev = (lev, coef)
        without = lambda *ks: [kc for kc in full_ports() if kc[0] not in ks]  # noqa: E731
        for env, ports in ((NO_FOLD, full_ports()), (NO_LIN, full_ports()), (NO_ENCLIN, full_ports()),
                           ({"WG_D3D12_FUSION": "none"}, full_ports()),
                           (None, without(FOLD, LIN, ENCLIN))):   # the M-form fusions' kernels: not needed
            with self.subTest(fusion=env, ports=len(ports)):
                got = self.l_run(ports, frames=6, rw=96, rh=64, flags=(sizes, "--reset=4"), env=env)
                self.assertEqual(same_frames(got), same_frames(lines))

    def test_failed_launch_stops_the_frame(self):
        for kernel, before in ((DEC_COEF_HYB, "l_dec_project"), (BNECK_HYB, "patch_merge_c96_hyb"),
                               ("lc12/l_post_recon_p12", DEC_COEF_HYB)):
            with self.subTest(kernel=kernel):
                t = FT.Trace(self.l_run(full_ports(), frames=3, flags=("--fail=%s@1" % kernel,)))
                ok0, bad, ok2 = t.frames
                self.assertEqual(bad["failed"], kernel)
                names = [ln.kernel for ln in bad["launches"]]
                self.assertEqual(names, [ln.kernel for ln in ok0["launches"]][:len(names)])
                self.assertEqual(names[-1], before)
                self.assertIsNone(ok2["error"])
                self.check_d3d12_frame(t, ok2)

    def test_model_m_unchanged(self):
        """Model M on backend 3 traces the same in the build with Model L as in the build without it."""
        a = run(frames=2, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=full_ports())
        b = run(frames=2, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=full_ports(),
                exe=FT.trace_exe())
        self.assertEqual([re.sub(r"wg_lfb_\w+", "B", x) for x in a], [re.sub(r"wg_lfb_\w+", "B", x) for x in b])
        self.assertTrue(a[0].startswith("create ok model M"), a[0])

    def test_hip_unchanged(self):
        """On HIP Model L keeps the base coefficient format (l_coef) and refuses pack12."""
        self.assertTrue(run("model=L", build_files=FT.SHIPPING, env={"WG_POST_PACK": "p12"})[0].startswith(
            "ERR create: Model L: post_pack=p12"))

    # Model L's C1/C2 reconstruction on native D3D12, HDR and LDR colour: its D3D12 ports, then the output filter
    OWN = {1: [("lc12/l_post_recon_p12", "l_post__lc12.co"), ("lc12/l_post_recon", "l_post__lc12.co"),
               ("l_coef", "l_ops.co"), ("w64/post_filter_c1_s4", "post_w64.co")],
           0: [("lc12/l_post_recon_p12_ldr", "l_post__lc12.co"), ("lc12/l_post_recon_ldr", "l_post__lc12.co"),
               ("l_coef", "l_ops.co"), ("l_post_filter_ldr", "l_post.co")]}

    def own_ports(self):
        have = full_ports()
        return have + [kc for hdr in self.OWN for kc in self.OWN[hdr] if kc not in have]

    def test_own_post(self):
        """HDR and LDR colour: lc12/l_post_recon_p12[_ldr] on the frames l_coef_pack12 stores, with post_recon's
        arguments (the role checks), then the unbanded filter (HDR w64/post_filter_c1_s4 with the exposure pair; LDR
        l_post_filter_ldr) from its display image into the output; no Model M post kernel. A frame the pack12 kernels
        do not take (render 180 x 108 = D) runs l_coef and lc12/l_post_recon[_ldr] (the fallback route)."""
        sizes = "--sizes=96x64,96x64,180x108,80x48,77x45,180x108"
        for hdr in (1, 0):
            sfx = "" if hdr else "_ldr"
            with self.subTest(hdr=hdr):
                t = FT.Trace(run("model=L", frames=6, build_files=FT.SHIPPING, rw=180, rh=108, hdr=hdr,
                                 flags=("--backend=3", sizes, "--reset=4"), manifest=self.own_ports()))
                self.assertEqual(t.create, "create ok model L")
                self.assertEqual(t.unknown(), [])
                self.assertEqual(t.kernels.get("l_post"), "l_post_recon" + sfx)
                self.assertEqual(t.kernels.get("l_post_c12"), "on")
                prev = None
                for f in t.frames:
                    self.assertIsNone(f["error"])
                    info, ls = f["info"], f["launches"]
                    names = [ln.kernel for ln in ls]
                    p12 = (info["rw"], info["rh"]) != (180, 108)  # pack12 frames
                    self.assertIn(DEC_COEF_HYB if p12 else "l_coef", names)   # (lever 3: the store folded)
                    self.assertNotIn("l_coef" if p12 else DEC_COEF_HYB, names)
                    self.assertNotIn("l_coef_pack12", names)
                    self.assertEqual(model_m_post(names), [])
                    post = [ln for ln in ls if ln.kernel == ("lc12/l_post_recon_p12" if p12 else "lc12/l_post_recon") +
                            sfx]
                    self.assertEqual(len(post), 1, names)
                    lev, coef = self.check_d3d12_frame(t, f, sfx)  # every argument's role, the filter after the post
                    flt = ls[ls.index(post[0]) + 1]
                    self.assertEqual(flt.kernel, "w64/post_filter_c1_s4" if hdr else "l_post_filter_ldr")
                    self.assertEqual((flt.grid, flt.block), (cdiv(120, 40) * cdiv(cdiv(72, 2), 8), 64))
                    self.assertEqual(flt.args[:2], [f["names"]["post.display"], f["names"]["filter.out"]])
                    self.assertEqual(flt.args[2:6], [str((3 * 120 + 1) // 2), str((3 * 72 + 1) // 2), "120", "72"])
                    if hdr:
                        self.assertEqual(flt.args[7], f["names"]["exposure.pair"])
                    if p12:  # pack12 image A (the store's argument 1) is what the reconstruction reads
                        self.assertEqual(coef.args[1], post[0].args[1])
                        self.assertGreaterEqual(t.extent(coef.args[1]), 8 * info["rw"] * info["rh"])
                    if prev:
                        self.assertEqual(lev.args[4], latent_out(prev[1]))  # the latent the last frame's store wrote
                    prev = (lev, coef)

    def test_color_early_option(self):
        """Runtime option l_color_early=on (the direct-resource route's colour staging after the input encoder): reported
        in the kernel line; on the buffer route (no staging) the frames launch exactly what they launch without it."""
        args = dict(frames=3, build_files=FT.SHIPPING, rw=96, rh=64, flags=("--backend=3",), manifest=self.own_ports())
        on = FT.Trace(run("model=L,l_color_early=on", **args))
        off = FT.Trace(run("model=L", **args))
        self.assertEqual(on.create, "create ok model L")
        self.assertEqual(on.kernels.get("l_color_early"), "on")
        self.assertIsNone(off.kernels.get("l_color_early"))
        self.assertEqual([[(ln.kernel, ln.args) for ln in f["launches"]] for f in on.frames],
                         [[(ln.kernel, ln.args) for ln in f["launches"]] for f in off.frames])

    def test_no_model_m_post_route(self):
        """Model L never runs Model M's post: every route traced (HDR and LDR colour; HIP and native D3D12; pack12 and
        fallback frames) launches Model L's C1/C2 reconstruction and no Model M post kernel; l_post=m is refused; where
        Model L's own route cannot run (a port missing, another post / filter route asked for) creation is refused with
        the reason, never a different algorithm."""
        def first(options, **kw):
            return run(options, frames=1, build_files=FT.SHIPPING, **kw)[0]
        d3d12 = dict(flags=("--backend=3",), manifest=self.own_ports())
        # the runtime option: m refused, own a no-op, anything else refused
        self.assertEqual(first("model=L;l_post=m", **d3d12), "ERR create: l_post=m: Model L has no Model M post route "
                                                             "(its reconstruction is l_post_recon[_ldr] only)")
        self.assertEqual(first("model=L;l_post=yes", **d3d12),
                         "ERR create: l_post must be own (Model L's reconstruction; there is no other)")
        self.assertEqual(first("l_post=own", **d3d12), "ERR create: l_post=own is a Model L option (its own "
                                                        "reconstruction)")
        # a missing port of Model L's route: refused by name (HDR and LDR), never another post (the C1/C2 ports: all
        # four checked first, by the hybrid_full form)
        for hdr, kcs in self.OWN.items():
            for k, _ in kcs:
                with self.subTest(hdr=hdr, port=k):
                    got = first("model=L", hdr=hdr, flags=("--backend=3",),
                                manifest=[kc for kc in self.own_ports() if kc[0] != k])
                    if k.startswith("lc12/"):
                        self.assertEqual(got, "ERR create: WG_L_POST=c12 on native D3D12: not in the kernel manifest: "
                                         + k)
                    else:
                        self.assertEqual(got, "ERR create: Model L on native D3D12 (%s): not yet ported to D3D12: %s; "
                                              "Model L has no other reconstruction (no Model M post route)"
                                         % ("HDR" if hdr else "LDR", k))
        # another of Model M's filter routes asked for: refused (the RGBA16F filter output)
        got = first("model=L", env={"WG_FILTER_KERNEL": "post_filter_c1_s4_o16"}, **d3d12)
        self.assertTrue(got.startswith("ERR create: "), got)
        self.assertNotIn("create ok", got)
        # every route: Model L's C1/C2 reconstruction, no Model M post kernel (HIP: base coefficients and
        # lc12/l_post_recon[_ldr]; native D3D12: pack12 frames and the fallback frame)
        sizes = "--sizes=96x64,180x108,80x48"
        for hdr in (1, 0):
            sfx = "" if hdr else "_ldr"
            for backend in ("hip", "d3d12"):
                with self.subTest(hdr=hdr, backend=backend):
                    kw = dict(d3d12) if backend == "d3d12" else {}
                    kw["flags"] = kw.get("flags", ()) + (sizes,)
                    t = FT.Trace(run("model=L", frames=3, build_files=FT.SHIPPING, rw=180, rh=108, hdr=hdr, **kw))
                    self.assertEqual(t.create, "create ok model L")
                    self.assertEqual(t.kernels.get("l_post"), "l_post_recon" + sfx)
                    for f in t.frames:
                        self.assertIsNone(f["error"])
                        names = [ln.kernel for ln in f["launches"]]
                        self.assertEqual(model_m_post(names), [], names)
                        own = [n for n in names if base(n) == "l_post_recon"]
                        self.assertEqual(len(own), 1, names)
                        self.assertTrue(own[0].startswith("lc12/") and own[0].endswith(sfx), own)
                        if not hdr:  # LDR: Model L's LDR output stage, no exposure-decoding filter
                            self.assertIn("l_post_filter_ldr", names)
                            self.assertFalse([n for n in names if n.endswith("post_filter_c1_s4")], names)

        def hip(options):
            return [re.sub(r"wg_lfb_\w+", "B", x) for x in run(options, frames=2, build_files=FT.SHIPPING)]
        self.assertEqual(hip("model=L;l_post=own"), hip("model=L"))

    def test_ldr_filter_selection(self):
        """LDR colour: the output stage is l_post_filter_ldr (the default filter's geometry, 3 x f32). Another filter
        selection has no LDR form and is refused at create with the reason, on HIP (an alternate geometry, an RGBA16F
        filter output, output_filter=quality9) and on native D3D12, never run with another filter's grid or format (the
        output copy's l_post_filter_ldr_h is behind the same check). HDR keeps those selections. The default LDR frame
        launches l_post_filter_ldr on the default filter's grid, from the display image into the output."""
        def first(options="model=L", **kw):
            return run(options, frames=1, build_files=FT.SHIPPING, **kw)[0]
        why = ("ERR create: Model L, LDR colour: its output stage is l_post_filter_ldr (the default filter's geometry, "
               "40 columns x 4 row pairs, 3 x f32 RGB; RGBA16F rows for the output copy); the selected filter (")
        for env, opts, tail in (({"WG_FILTER_KERNEL": "post_filter"}, "model=L", "post_filter) has no LDR form"),
                                ({"WG_FILTER_KERNEL": "post_filter_c1_s16"}, "model=L",
                                 "post_filter_c1_s16) has no LDR form"),
                                ({"WG_FILTER_KERNEL": "post_filter_c1_s4_o16"}, "model=L",
                                 "post_filter_c1_s4_o16, RGBA16F output) has no LDR form"),
                                ({}, "model=L;output_filter=quality5",
                                 "post_filter_c1_s4, output_filter=quality5) has no LDR form")):
            with self.subTest(env=env, options=opts):
                self.assertEqual(first(opts, hdr=0, env=env), why + tail)
                hdr_got = first(opts, hdr=1, env=env)   # HDR: Model M's output stage takes the filter kernels
                if env:
                    self.assertEqual(hdr_got, "create ok model L")
                else:   # (the output_filter options need Model M's own post paths: refused for Model L before)
                    self.assertTrue(hdr_got.startswith("ERR create: output_filter="), hdr_got)
        self.assertTrue(first("model=L;output_filter=quality9", hdr=0).startswith("ERR create: "))   # refused before
        got = first("model=L;output_filter=quality9", hdr=0, flags=("--backend=3",), manifest=self.own_ports())
        self.assertTrue(got.startswith("ERR create: Model L"), got)
        # the default: l_post_filter_ldr on the default filter's grid (O 120 x 72: 3 x 5 workgroups of 64 lanes)
        for backend in ("hip", "d3d12"):
            with self.subTest(default=backend):
                kw = dict(flags=("--backend=3",), manifest=self.own_ports()) if backend == "d3d12" else {}
                t = FT.Trace(run("model=L", frames=2, build_files=FT.SHIPPING, hdr=0, **kw))
                self.assertEqual(t.create, "create ok model L")
                for f in t.frames:
                    flt = [ln for ln in f["launches"] if ln.kernel == "l_post_filter_ldr"]
                    self.assertEqual(len(flt), 1)
                    self.assertEqual((flt[0].grid, flt[0].block), (cdiv(120, 40) * cdiv(cdiv(72, 2), 8), 64))
                    self.assertEqual(flt[0].args[:2], [f["names"]["post.display"], f["names"]["filter.out"]])
                    self.assertEqual(flt[0].args[2:6], [str((3 * 120 + 1) // 2), str((3 * 72 + 1) // 2), "120", "72"])


class Refusals(unittest.TestCase):
    def test_paths(self):
        self.assertTrue(run("model=L", env={"WG_DEC1_DEC0": "fused"})[0].startswith(
            "ERR create: Model L: WG_DEC1_DEC0=fused"))
        self.assertTrue(run("model=L", env={"WG_RECON_SIZE": "output"})[0].startswith(
            "ERR create: Model L: recon=output"))
        self.assertTrue(run("model=L;motion_outres=original", build_files=("encoder_mo.co",))[0]
                        .startswith("ERR create: Model L: motion_outres=original"))
        self.assertTrue(run("model=L", build_files=FT.SHIPPING, env={"WG_POST_PACK": "p12"})[0].startswith(
            "ERR create: Model L: post_pack=p12"))
        # Model L's reconstruction is its own kernel: Model M's post kernels are not chosen for it
        self.assertTrue(run("model=L", env={"WG_POST_KERNEL": "post_recon2_r"})[0].startswith(
            "ERR create: Model L: its reconstruction is l_post_recon"))
        # LDR colour: Model L's LDR reconstruction (test_no_model_m_post_route traces it)
        self.assertEqual(run("model=L", hdr=0)[0], "create ok model L")

    def test_model_data(self):
        d = tempfile.mkdtemp(prefix="wg_lf_")
        try:
            SM.write_raw(d, SM.synthetic_tensors(1))
            # no Model L file
            self.assertEqual(run("model=L", model=d), ["ERR create: model data missing: " + d + "/L/model.wgcm"])
            # raw tensors, even complete and of the right shapes: refused (nothing shows they are Model L's)
            LL.write_raw(os.path.join(d, "L"), LL.raw_model(SEED))
            self.assertTrue(run("model=L", model=d, flags=("--format=1",))[0].startswith(
                "ERR create: Model L reads only the converter's model.wgcm"))
            # Model L's own stages with Model M's shared stages (same names and shapes): their digests are not the
            # map's, refused
            mixed = LL.raw_model(SEED)
            m = SM.synthetic_tensors(1)
            for st in SHARED:
                mixed[st] = m[st]
            write_l(d, mixed, map_file=map_path())
            got = run("model=L", model=d)
            self.assertEqual(len(got), 1)
            self.assertTrue(got[0].startswith("ERR create: model data damaged: "), got[0])
            self.assertIn("does not match the map", got[0])
            # the same file with Model L's shared stages loads
            write_l(d, LL.raw_model(SEED), map_file=map_path())
            self.assertEqual(run("model=L", model=d, frames=1)[0], "create ok model L")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class FusedPost(unittest.TestCase):
    """Model L's true-fusion post (WG_L_POST_TF, default on; dx12/kernels/l_post_tf.hlsl) on native D3D12, HDR colour, the staged
    route: l_post_recon_p12 + post_filter_c1_s4 + scale_rgb in one launch (w64_lc12/l_post_tf_p12), and with pi != 1 the
    pre-exposure division folded into the encoder and the post (lev/input_encoder_q20_pf + w64_lc12/l_post_tf_p12_pf:
    no color_divide). Off (WG_L_POST_TF=0), with a missing port, for LDR colour or for a footprint the kernel does not
    stage: the separate launches."""

    TF = "w64_lc12/l_post_tf_p12"
    SEPARATE = {"w64/post_filter_c1_s4", "scale_rgb", "color_divide", "lc12/l_post_recon_p12", "lc12/l_post_recon"}

    @staticmethod
    def bits(x):
        return str(struct.unpack("<I", struct.pack("<f", x))[0])

    def trace(self, pe=None, ports=None, env=None, hdr=1, frames=3, rw=RW, rh=RH):
        flags = ("--backend=3",) + (("--pre-exposure=%s" % pe,) if pe is not None else ())
        t = FT.Trace(run("model=L", frames=frames, build_files=FT.SHIPPING, flags=flags, manifest=ports or full_ports(),
                         env=env, hdr=hdr, rw=rw, rh=rh, tf_default=True))
        self.assertEqual(t.create, "create ok model L")
        self.assertEqual(t.unknown(), [])
        return t

    @staticmethod
    def names(f):
        return [ln.kernel for ln in f["launches"]]

    def check_fused(self, t, pi, fold):
        """Every frame: the fused post (the _pf form when folded), its arguments, none of the separate launches."""
        post_name = self.TF + ("_pf" if fold else "")
        enc_name = "lev/input_encoder_q20" + ("_pf" if fold else "")
        self.assertEqual(t.kernels["l_post_tf"], "on")
        self.assertEqual(t.kernels["post_last"], post_name)
        self.assertEqual(t.kernels["filter_last"], "fused")
        for f in t.frames:
            self.assertIsNone(f["error"])
            names = self.names(f)
            self.assertEqual(names.count(post_name), 1, names)
            self.assertEqual(names.count(enc_name), 1, names)
            self.assertEqual(sorted(set(names) & self.SEPARATE), [], names)
            self.assertEqual(len([n for n in names if n.startswith("w64_lc12/l_post_tf")]), 1, names)
            self.assertNotIn("lev/input_encoder_q20" if fold else "lev/input_encoder_q20_pf", names)
            post = next(ln for ln in f["launches"] if ln.kernel == post_name)
            # the post_tf chunk plan (runtime.cpp tf_chunks): output 120 x 72, post_tf_cpairs 68
            npo, strips, cp = 36, cdiv(120, 40), 68
            c2 = min(36, cp)
            k1 = max(0, (npo - cdiv(128, strips) * c2) // cp)
            nch = k1 + cdiv(npo - k1 * cp, c2)
            self.assertEqual((post.grid, post.block), (strips * nch, 512))
            a = post.args
            self.assertEqual(len(a), 22, a)
            self.assertEqual(a[11], f["names"]["filter.out"])      # the output
            self.assertEqual(a[13], f["names"]["exposure.pair"])
            self.assertEqual(a[14:16], ["120", "72"])
            self.assertEqual(a[16], self.bits(float(np.float32(0.01 / 1.0))))   # dec_scale
            self.assertEqual(a[17], self.bits(pi))                               # pi: the scale folded in (1: none)
            self.assertEqual(a[19:22], [str(cp), str(k1), str(c2)])

    def test_default_is_the_fused_post(self):
        self.check_fused(self.trace(), 1.0, False)

    def test_pre_exposure_folds_the_division(self):
        for pi in (1.25, 2.0, 0.5, 64.0):
            with self.subTest(pi=pi):
                self.check_fused(self.trace(pe=pi), pi, True)

    def test_division_window_invalid_keeps_color_divide(self):
        """A pi whose division window is invalid (B > 125: no fast form): color_divide stays, the post is the plain form
        (it still folds scale_rgb), no scale_rgb launch."""
        pi = 1e38
        t = self.trace(pe=pi)
        self.assertEqual(t.kernels["post_last"], self.TF)
        for f in t.frames:
            names = self.names(f)
            self.assertEqual(names.count("color_divide"), 1, names)
            self.assertNotIn("scale_rgb", names)
            self.assertNotIn("lev/input_encoder_q20_pf", names)
            self.assertEqual(names.count(self.TF), 1)
            post = next(ln for ln in f["launches"] if ln.kernel == self.TF)
            self.assertEqual(post.args[17], self.bits(float(np.float32(pi))))

    def test_missing_pf_ports_keep_color_divide(self):
        ports = [kc for kc in full_ports() if kc[0] not in ("w64_lc12/l_post_tf_p12_pf", "lev/input_encoder_q20_pf")]
        t = self.trace(pe=2.0, ports=ports)
        for f in t.frames:
            names = self.names(f)
            self.assertEqual(names.count("color_divide"), 1, names)
            self.assertEqual(names.count(self.TF), 1, names)
            self.assertNotIn("scale_rgb", names)

    def test_toggle_off_runs_the_separate_launches(self):
        for pi in (None, 2.0):
            with self.subTest(pi=pi):
                t = FT.Trace(run("model=L", frames=2, build_files=FT.SHIPPING,
                                 flags=("--backend=3",) + (() if pi is None else ("--pre-exposure=%s" % pi,)),
                                 manifest=full_ports(), env={"WG_L_POST_TF": "0"}, tf_default=True))
                self.assertEqual(t.create, "create ok model L")
                self.assertEqual(t.kernels["l_post_tf"], "off")
                for f in t.frames:
                    names = self.names(f)
                    self.assertEqual([n for n in names if "l_post_tf" in n or n.endswith("_pf")], [])
                    self.assertEqual(names.count("lc12/l_post_recon_p12"), 1, names)
                    self.assertEqual(names.count("w64/post_filter_c1_s4"), 1, names)
                    self.assertEqual(names.count("scale_rgb"), 0 if pi is None else 1, names)
                    self.assertEqual(names.count("color_divide"), 0 if pi is None else 1, names)

    def test_bad_toggle_refused(self):
        got = run("model=L", frames=1, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=full_ports(),
                  env={"WG_L_POST_TF": "2"}, tf_default=True)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("ERR create: WG_L_POST_TF"), got[0])

    def test_without_the_port_the_separate_route(self):
        ports = [kc for kc in full_ports() if kc[0] not in ("w64_lc12/l_post_tf_p12", "w64_lc12/l_post_tf_p12_pf")]
        t = self.trace(ports=ports, frames=1)
        self.assertEqual(t.kernels["l_post_tf"], "off")
        self.assertIn("lc12/l_post_recon_p12", self.names(t.frames[0]))

    def test_ldr_and_a_footprint_over_the_staging_are_not_fused(self):
        t = self.trace(hdr=0, frames=1)
        self.assertEqual([n for n in self.names(t.frames[0]) if "l_post_tf" in n], [])
        # render = D (180 x 108 for the 120 x 72 output): the 4 x 4 footprint of a step is over 512 staged texels
        t = self.trace(frames=1, rw=180, rh=108)
        self.assertEqual([n for n in self.names(t.frames[0]) if "l_post_tf" in n], [])
        self.assertIn("lc12/l_post_recon", self.names(t.frames[0]))   # (pack12 does not fit D = R: l_coef + l_post_recon)


class ReleaseDefault(unittest.TestCase):
    """0.6.5 (oversight 2026-11-15 18:36Z): Model L = hybrid_full when WG_L_FORM is unset. On native D3D12 the fast
    hybrid route (the manifest's "_hyb" twins on the unfused layout, the C1/C2 reconstruction lc12/*); on HIP the scalar
    reference route (l_q7__hyb.co, l_q7w__hyb.co, l_post__lc12.co). q7 is an opt-in reference; the M-form is refused.
    A hybrid route that cannot be taken is refused at create, by name, in one line: no fallback to the q7 reference,
    the M-form or HIP. (The game-side rule, no move to the HIP executor when the D3D12 executor refuses a context at
    create, is #225's for every model: clean/runtime/interop/executor_select.h dx12_refused_line.)"""

    def d3d12(self, ports, env=None, frames=1, **kw):
        return run("model=L", frames=frames, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=ports, env=env,
                   **kw)

    @staticmethod
    def launched(lines):
        return {ln.split()[1] for ln in lines if ln.startswith("launch ")}

    @staticmethod
    def kernels_text(lines):
        return next(ln for ln in lines if ln.startswith("kernels "))

    def test_exfuse_toggle(self):
        """WG_L_EXFUSE=0, or a manifest without the fused decoder blocks: the unfused route (standalone expands and decoder
        blocks, swin_fused=none); any other value is refused at create."""
        unfused = {"expand_c128_hyb", "expand_c96_hyb", "swin_block_c96_add_hyb", "swin_block_c64_add_hyb"}
        for label, got in (("env 0", self.d3d12(full_ports(), env={"WG_L_EXFUSE": "0"})),
                           ("no ports", self.d3d12([kc for kc in full_ports() if kc[0] not in EX_HYB]))):
            with self.subTest(case=label):
                self.assertEqual(got[0], "create ok model L")
                text = self.kernels_text(got)
                self.assertIn("swin_fused=none", text)
                self.assertIn("l_exfuse=off", text)
                names = self.launched(got)
                self.assertTrue(unfused <= names, sorted(unfused - names))
                self.assertEqual(set(EX_HYB) & names, set())
        got = self.d3d12(full_ports(), env={"WG_L_EXFUSE": "2"})
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("ERR create: WG_L_EXFUSE"), got[0])

    def test_default_is_hybrid_full_on_d3d12(self):
        for env in (None, {"WG_L_FORM": "hybrid"}):
            with self.subTest(env=env):
                got = self.d3d12(full_ports(), env=env, tf_default=True, w3_default=True)
                self.assertEqual(got[0], "create ok model L")
                text = self.kernels_text(got)
                ntwins = sum(k.endswith("_hyb") and not k.endswith("_q7c_hyb") for k, _ in full_ports())
                for want in ("l_form=hybrid", "l_hyb_d3d12=on", "l_post_c12=on", "hyb_twins=%d" % ntwins,
                             "swin_fused=dec2,dec1", "l_exfuse=on", "hyb_tables=on", "l_w3=23 w3_kernels=%d" % len(W3_DEFAULT)):
                    self.assertIn(want, text)
                names = self.launched(got)
                # the wave 3 default mask 0x17: each kernel of W3_DEFAULT in place of its wave 2 form
                twin_of = {w3: w2 for w2, w3 in W3_DEFAULT.items()}
                used = (set(HYB_TWINS) - {"l_swin_block_c32_enc_hyb", "l_swin_block_c32_dec_hyb",
                                          "swin_block_c128_hyb", "swin_block_c96_add_hyb", "swin_block_c64_add_hyb",
                                          "expand_c128_hyb", "expand_c96_hyb"} | {ENC_LIN_HYB, DEC_COEF_HYB, BNECK_HYB}
                        | set(EX_HYB))
                used = {W3_DEFAULT.get(k, k) for k in used}
                self.assertTrue(used <= names, sorted(used - names))
                self.assertEqual(set(W3_DEFAULT) & names, set())   # no wave 2 form of a routed kernel
                self.assertTrue(set(W3_DEFAULT.values()) <= names, sorted(set(W3_DEFAULT.values()) - names))
                self.assertIn("w64_lc12/l_post_tf_p12", names)   # the fused post (WG_L_POST_TF default on)
                blocks = [n for n in names if n.split("/")[-1].startswith(("swin_", "l_swin_", "patch_merge", "expand"))]
                self.assertEqual([n for n in blocks if not n.endswith("_hyb") and n not in twin_of], [])   # no M-form block, merge, expand
                self.assertEqual([n for n in names if "l_q7" in n], [])               # no q7 reference kernel
                self.assertEqual([n for n in names if n.startswith("l_post_recon")], [])   # only the C1/C2 post

    def test_hybrid_route_refused_never_falls_back(self):
        """Each twin and each C1/C2 port missing: create refused in one line naming it; nothing created, nothing
        launched (no fallback to the q7 reference, the M-form or HIP)."""
        for missing in HYB_TWINS + LC12 + FUSED_REQUIRED:
            for env in (None, {"WG_L_FORM": "hybrid"}):
                with self.subTest(missing=missing, env=env):
                    got = self.d3d12([kc for kc in full_ports() if kc[0] != missing], env=env, frames=2)
                    self.assertEqual(len(got), 1, got[:3])
                    self.assertTrue(got[0].startswith("ERR create: ") and missing in got[0], got[0])
                    # the form as resolved: nothing set = the release default (review finding)
                    form = "Model L hybrid (default)" if env is None else "WG_L_FORM=hybrid"
                    self.assertTrue(form in got[0] or "WG_L_POST=c12" in got[0], got[0])

    def test_hybrid_geometry_overrides_refused(self):
        """A Swin / GEMM launch-geometry override is refused at create for every hybrid context: the HIP route and the
        native-D3D12 route without direct I/O (review P1: nt=4 sized the merge / expand grids for 512 tokens per workgroup
        while the shaders kept 256; wpg 16 halved the block grid), explicit and default form, before any allocation."""
        for geo in ("nt=4", "gemm_threads=128", "swin_threads=128", "wpw64=2", "wpw96=2", "wpw128=2", "wpg0=16",
                    "wpg4=16"):
            for backend, kw in ((0, {}), (3, dict(build_files=FT.SHIPPING, manifest=full_ports()))):
                for env in (None, {"WG_L_FORM": "hybrid"}):
                    with self.subTest(geo=geo, backend=backend, env=env):
                        got = run("model=L", frames=2, flags=("--backend=%d" % backend, "--geo=" + geo), env=env, **kw)
                        self.assertEqual(len(got), 1, got[:3])
                        self.assertTrue(got[0].startswith("ERR create: ") and "launch-geometry override" in got[0], got[0])
        # the base geometry spelled out is not an override
        got = run("model=L", frames=1, flags=("--backend=3", "--geo=nt=2", "--geo=wpg4=8"), build_files=FT.SHIPPING,
                  manifest=full_ports())
        self.assertTrue(got[0].startswith("create ok model L"), got[0])

    def test_default_route_loads_only_deployed_objects(self):
        """Every Model L code object the default (hybrid) route loads on the HIP backend is built into a release runtime
        library (clean/runner/build.py L_EMBED_FILES): a deployed library alone creates the default (review P2: l_q7__hyb,
        l_q7w__hyb and l_post__lc12 were read from the build folder only)."""
        from clean.runner import build as B
        B.set_model_l(True)
        deployed = set(B.EMBED_FILES)
        for env in (None, {"WG_L_FORM": "hybrid"}):
            with self.subTest(env=env):
                got = run("model=L", frames=1, env=env)
                self.assertTrue(got[0].startswith("create ok model L"), got[0])
                loads = [x.split()[1] for x in got if x.startswith("load ")]
                self.assertTrue([n for n in loads if n.startswith("l_q7")], loads)
                self.assertEqual([n for n in loads if n.startswith("l_") and n not in deployed], [])
        self.assertFalse([n for n in deployed if B.is_check_object(n)])

    def test_release_build_refuses_check_routes(self):
        """The check routes (the mutant, the HIP twins) are not offered by a release build (the recorder is built
        without WG_DEV_DIAGNOSTICS): refused at create in one line, nothing created."""
        for env in ({"WG_L_FORM": "hybrid_mutant"}, {"WG_L_FORM": "hybrid_twin"}):
            with self.subTest(env=env):
                got = run("model=L", frames=1, env=env)
                self.assertEqual(len(got), 1, got[:3])
                self.assertTrue(got[0].startswith("ERR create: ") and "developer builds only" in got[0], got[0])
        got = run("", frames=1, env={"WG_M_FORM": "hybrid_mutant"})
        self.assertTrue(got[0].startswith("ERR create: ") and "developer builds only" in got[0], got[0])

    def test_release_hybrid_route_ignores_the_diagnostic_hooks(self):
        """A release build's hybrid route is never switched by WG_AB_FUSION / WG_AB_BLOCK (or the barrier / kernel
        substitution hooks): the frames trace as without them, and one note says the keys were ignored."""
        def l_run(**kw):
            return run("model=L", frames=4, rw=96, rh=64, build_files=FT.SHIPPING, flags=("--backend=3",),
                       manifest=full_ports(), **kw)
        base = l_run()
        for env in ({"WG_AB_FUSION": "none", "WG_AB_BLOCK": "1"}, {"WG_D3D12_NOBAR": "swin_bneck6_hyb"},
                    {"WG_D3D12_KSUB": "swin_bneck6_hyb=swin_block_c128_hyb"},
                    # an invalid value is not even read: it cannot refuse the release hybrid create (review P2)
                    {"WG_AB_FUSION": "no-such-fusion", "WG_AB_BLOCK": "x"}, {"WG_AB_KSUB": "=", "WG_AB_NOBAR": ","}):
            with self.subTest(env=env):
                got = l_run(env=env)
                self.assertTrue(any(x.startswith("kernels ") and " diag_hooks=ignored" in x for x in got))
                self.assertEqual(same_frames([x.replace(" diag_hooks=ignored", "") for x in got]), same_frames(base))

    def test_hybrid_tables_refused_never_falls_back(self):
        """The device refuses the hybrid tables (the recorder's --fail-hyb-tables, as native D3D12 when their arena
        range is held): create refused in one line naming the resolved form, nothing launched; with the tables the
        same build creates (the CPU recorder implements Device::hyb_reserve: review finding P1)."""
        for env in (None, {"WG_L_FORM": "hybrid"}):
            with self.subTest(env=env):
                got = run("model=L", frames=2, build_files=FT.SHIPPING, flags=("--backend=3", "--fail-hyb-tables"),
                          manifest=full_ports(), env=env)
                form = "Model L hybrid (default)" if env is None else "WG_L_FORM=hybrid"
                self.assertEqual(len(got), 1, got[:3])
                self.assertTrue(got[0].startswith("ERR create: " + form + " on native D3D12: no hybrid tables"), got[0])
                ok = self.d3d12(full_ports(), env=env)
                self.assertEqual(ok[0], "create ok model L")
                self.assertIn("hyb_tables=on", self.kernels_text(ok))
        # the HIP reference route and Model L's q7 reference never ask for the tables
        for flags, env in (((), None), (("--backend=3",), {"WG_L_FORM": "q7"})):
            with self.subTest(flags=flags, env=env):
                got = run("model=L", frames=1, build_files=FT.SHIPPING, flags=flags + ("--fail-hyb-tables",),
                          manifest=full_ports() if flags else None, env=env)
                self.assertEqual(got[0], "create ok model L")

    def test_hybrid_bneck_timeout_refuses_and_writes_no_history(self):
        """The bottleneck's bounded wait (swin_bneck6_hyb) REFUSES a frame and lets nothing of it through (review P1):
        the whole hybrid frame runs under the device's frame guard, read again right after the bottleneck. In the frame
        whose wait times out (the recorder's --guard-timeout), every launch after the bottleneck is skipped (the post
        that writes the history and the output among them); the next frame is refused in one line, launching nothing;
        the frame after it starts over (reset 1) with the abort word cleared first. Control: without the timeout,
        nothing is skipped or refused. Default form and explicit form."""
        for env in (None, {"WG_L_FORM": "hybrid"}):
            for timeout in (None, 1):
                with self.subTest(env=env, timeout=timeout):
                    flags = ("--backend=3",) + (("--guard-timeout=%d" % timeout,) if timeout is not None else ())
                    got = run("model=L", frames=5, rw=96, rh=64, build_files=FT.SHIPPING, flags=flags,
                              manifest=full_ports(), env=env)
                    self.assertEqual(got[0], "create ok model L")
                    blocks, cur = [], None
                    for x in got:
                        if x.startswith("frame "):
                            cur = [x]
                            blocks.append(cur)
                        elif cur is not None and not x.startswith(("name ", "kernels ")):
                            cur.append(x)
                    self.assertEqual(len(blocks), 5)
                    dev = [[x for x in b[1:] if x.startswith(("launch ", "skipped ", "guard_", "ERR "))] for b in blocks]
                    for f, d in enumerate(dev):
                        refused = timeout is not None and f == timeout + 1
                        if refused:
                            self.assertEqual(len(d), 1, d[:4])
                            self.assertTrue(d[0].startswith("ERR frame %d: " % f) and "timed out" in d[0]
                                            and "refused" in d[0] and "starts over" in d[0], d[0])
                            continue
                        clear = timeout is not None and f == timeout + 2
                        if clear:
                            self.assertEqual(d[0], "guard_clear", d[:2])
                        g = d[1:] if clear else d
                        self.assertNotIn("guard_clear", g)
                        self.assertTrue(g[0].startswith("guard_begin ") and "(skipping)" not in g[0], g[:2])
                        self.assertEqual(g[-1], "guard_end")
                        i = [x.startswith("launch swin_bneck6_hyb ") for x in g].index(True)
                        self.assertEqual(g[i + 1].split(" (")[0], g[0])   # the guard read again, same word
                        after = g[i + 2:-1]
                        self.assertTrue([x for x in after if "post" in x], after)
                        if timeout is not None and f == timeout:
                            self.assertTrue(g[i + 1].endswith("(skipping)"), g[i + 1])
                            self.assertEqual([x for x in after if not x.startswith("skipped ")], [])
                        else:
                            self.assertFalse([x for x in g if x.startswith("skipped ") or "(skipping)" in x])
                        reset = int(blocks[f][0].split(" reset ")[1].split()[0])
                        self.assertEqual(reset, 1 if f == 0 or clear else 0, blocks[f][0])

    def test_m_form_refused(self):
        for flags in ((), ("--backend=3",)):
            with self.subTest(flags=flags):
                got = run("model=L", frames=1, build_files=FT.SHIPPING, flags=flags,
                          manifest=full_ports() if flags else None, env={"WG_L_FORM": "m"})
                self.assertEqual(got, ["ERR create: WG_L_FORM=m refused: the M-form is not offered for Model L "
                                       "(default hybrid_full; q7 = reference)"])

    def test_q7_is_opt_in_reference(self):
        got = self.d3d12(full_ports(), env={"WG_L_FORM": "q7"})
        self.assertEqual(got[0], "create ok model L")
        names = self.launched(got)
        self.assertTrue(any(n.startswith("l_q7_") for n in names), sorted(names))
        self.assertEqual([n for n in names if n.endswith("_hyb") or n.startswith("lc12/")], [])

    def test_hip_default_is_the_reference_hybrid(self):
        got = run("model=L", frames=1, build_files=FT.SHIPPING)
        self.assertEqual(got[0], "create ok model L")
        self.assertIn("load l_q7__hyb.co", got)
        self.assertIn("load l_q7w__hyb.co", got)
        text = self.kernels_text(got)
        self.assertIn("l_form=hybrid", text)
        self.assertIn("l_post_c12=on", text)
        self.assertIn("lc12/l_post_recon", self.launched(got))


class WaveThree(unittest.TestCase):
    """Model L wave 3 (WG_L_W3, native D3D12 hybrid route): a mask of levers, each its own kernels in the manifest (the wave 2
    kernels stay), the same launches otherwise: kernel for kernel, grid, block and arguments identical to the wave 2 frame
    (WG_L_W3=0), only the names of the routed kernels differ. Default 0x17; a manifest without a lever's kernels runs the wave
    2 kernel; any other value is refused at create. (The kernels' bytes are the GPU verifier's: no GPU here.)"""
    SIZES = ("--sizes=96x64,96x64,80x48", "--reset=2")

    def d3d12(self, w3, ports=None, frames=3, **kw):
        env = dict(kw.pop("env", None) or {})
        if w3 is not None:
            env["WG_L_W3"] = w3
        return run("model=L", frames=frames, build_files=FT.SHIPPING, flags=("--backend=3",) + self.SIZES,
                   manifest=full_ports() if ports is None else ports, env=env, rw=96, rh=64, w3_default=w3 is None, **kw)

    @staticmethod
    def frames(lines):
        t = FT.Trace(lines)
        assert t.create == "create ok model L", t.create
        return t, [[(ln.kernel, ln.grid, ln.block, tuple(ln.args)) for ln in f["launches"]] for f in t.frames]

    def routed(self, mask):
        """The wave 2 -> wave 3 kernel names a mask selects (the default-mask table W3_DEFAULT plus the ablation forms)."""
        out = {}
        if mask & 0x01:
            out.update({k: v for k, v in W3_DEFAULT.items() if k.startswith("l_swin_block_c32")})
        elif mask & 0x08:
            out.update({k: k + "_gp" for k in W3_DEFAULT if k.startswith("l_swin_block_c32")})
        if mask & 0x02:
            out.update({k: v for k, v in W3_DEFAULT.items() if k.startswith("swin_block_c64")})
        if mask & 0x04:
            sfx = "_w3p" if mask & 0x20 else "_w3"
            out.update({k: k + sfx for k in ("swin_block_c96_hyb", "swin_block_c96_add_ex_hyb")})
            out["swin_bneck6_hyb"] = "swin_bneck6_hyb_w3"
        if mask & 0x10:
            out.update({k: v for k, v in W3_DEFAULT.items() if k.startswith("l_") and k.endswith(("gather", "project"))})
        return out

    def test_masks_route_only_the_names(self):
        _, base = self.frames(self.d3d12("0"))
        for mask in (None, "0x17", "1", "2", "4", "8", "0x10", "0x24", "0x3f", "0x18"):
            with self.subTest(mask=mask):
                lines = self.d3d12(mask)
                t, got = self.frames(lines)
                m = 0x17 if mask is None else int(mask, 0)
                table = self.routed(m)
                self.assertEqual(len(got), len(base))
                hit = set()
                for gf, bf in zip(got, base):
                    self.assertEqual(len(gf), len(bf))
                    for g, b in zip(gf, bf):
                        self.assertEqual(g[0], table.get(b[0], b[0]), (g, b))
                        self.assertEqual(g[1:], b[1:], (g, b))   # grid, block, arguments: unchanged
                        if b[0] in table:
                            hit.add(b[0])
                self.assertEqual(hit, set(table))   # every routed kernel is launched by the frame
                self.assertEqual(t.unknown(), [])
                text = next(ln for ln in lines if ln.startswith("kernels "))
                self.assertIn("l_w3=%d w3_kernels=%d" % (m, len(table)), text)

    def test_zero_is_the_wave_2_frame(self):
        _, a = self.frames(self.d3d12("0"))
        names = {k for f in a for k, *_ in f}
        self.assertEqual(names & set(W3_DEFAULT.values()), set())
        self.assertTrue(set(W3_DEFAULT) <= names, sorted(set(W3_DEFAULT) - names))

    def test_missing_kernels_fall_back_to_wave_2(self):
        """A manifest without a lever's kernels: that lever's launches keep the wave 2 kernel (same bytes), the rest are routed."""
        drop = {"swin_block_c64_hyb_w3", "l_project_gather_w64", "swin_bneck6_hyb_w3"}
        ports = [kc for kc in full_ports() if kc[0] not in drop]
        _, got = self.frames(self.d3d12(None, ports=ports))
        _, base = self.frames(self.d3d12("0"))
        table = {k: v for k, v in W3_DEFAULT.items() if v not in drop}
        for gf, bf in zip(got, base):
            self.assertEqual([(table.get(b[0], b[0]),) + b[1:] for b in bf], gf)
        self.assertEqual({k for f in got for k, *_ in f} & drop, set())
        none = [kc for kc in full_ports() if kc[0] not in W3_DEFAULT.values()]
        _, got = self.frames(self.d3d12(None, ports=none))
        self.assertEqual(got, base)

    def test_bad_values_refused(self):
        for bad in ("64", "0x40", "-1", "x", "1,2", "0x1g", "1.5"):
            with self.subTest(value=bad):
                got = self.d3d12(bad, frames=1)
                self.assertEqual(len(got), 1)
                self.assertTrue(got[0].startswith('ERR create: WG_L_W3: unknown value "%s"' % bad), got[0])


if __name__ == "__main__":
    unittest.main()
