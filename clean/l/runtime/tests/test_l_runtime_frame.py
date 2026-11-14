# SPDX-License-Identifier: MIT
"""Model L's runtime frame on the CPU (#89): the runtime built with Model L (define WG_MODEL_L, clean/l/runtime) runs
on the recording device of clean/tests/runtime_frame_trace.cpp (launches recorded with their arguments, not executed;
no GPU, no HIP). The interface is clean/l/FORMATS.md.

- Launch for launch: Model L's frame is Model M's frame with step 2 (encoder) and step 4 (decoder) replaced: the
  values into the halo map, the two per-pixel linears, the encoder block and the gather projection in place of
  input_encoder; the decoder projection, the decoder block and l_coef in place of coef_decoder. The grids come from
  FORMATS.md, the window offsets from the default table (clean/l/reference/table.py), the clamp-free block kernel
  (_nc) from bit 0 of the Python packer's block blobs. With an empty build folder and with a full build's optional
  code objects: Model L runs without pack12 (l_coef writes the base coefficient format).
- Arguments: every map goes from the launch that writes it to the one that reads it (q -> x1 -> x2 -> x3 -> enc0's
  features -> enc1; dec1's output -> y -> y2 -> l_coef), no launch writes what it reads, every buffer holds its map
  (allocation sizes), the extents (npix, tw, th, H, W), the window offsets of both parities, the render size, padded
  grid and reset in the encoder's and decoder's parameters, and the history (the latent l_coef writes is what the next
  frame's encoder reads). Sizes aligned, non-aligned and changing from frame to frame, and a game reset.
- A failing launch stops its frame; the next frame runs whole.
- Model L's code objects are loaded (l_enc_values.co under the tag lev, l_ops.co, l_swin.co), Model M's encoder and
  decoder not. Model M runs unchanged in the build with Model L.
- Model data: only the converter's model.wgcm, checked against the preset-L map (a synthetic file and map here). A raw
  folder is refused, so is a file whose shared stages are Model M's (their digests are not the map's), and a missing
  file. The paths that keep dec1's map on chip, need Model M's encoder or the pack12 decoder are refused.

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
from clean.l.runner import l_pack as LP
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
        manifest=None):
    """The trace program's output lines (WG_* removed from the environment, then env applied). hdr: WgRtConfig.hdr
    (Model L's reconstruction kernel is HDR; Model M does not read the flag). manifest: [(kernel, code object)] for
    <build>/dx12/manifest.json (the D3D12 ports a build has; backend 3)."""
    build = tempfile.mkdtemp(prefix="wg_lfb_")
    for n in build_files:
        open(os.path.join(build, n), "wb").close()
    if manifest is not None:
        os.makedirs(os.path.join(build, "dx12"))
        with open(os.path.join(build, "dx12", "manifest.json"), "w") as fh:
            json.dump({"format": 1, "kernels": [{"name": k, "co": co} for k, co in manifest]}, fh)
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
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


def blob_digests():
    """{blob name: SHA-256 hex} of Model L's blobs packed in Python (clean/l/runner/l_pack.py) from the test model's
    parameters: the independent names of the weight buffers (the recorder logs what create uploads, with its hash)."""
    blobs = LP.pack_model_l(L.random_params(np.random.default_rng(SEED)))
    # the post blob: s as float32, zero padding to 16 bytes (Model M's layout; the test model's s, LL.raw_model)
    blobs["post"] = struct.pack("<f", 1.25) + bytes(12)
    return {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}


L_KERNELS = ("lev/input_encoder", "l_linear_32_32_relu", "l_linear_32_32_f16", "l_swin_block_c32_enc",
             "l_project_gather", "l_dec_project", "l_swin_block_c32_dec", "l_coef", "l_post_recon")
# native D3D12, fusion lc (the default): the decoder block with l_coef_pack12 as its epilogue, in place of the pair
FOLD = "l_swin_block_c32_dec_coef12_nc"
FOLD_BASE = "l_swin_block_c32_dec_coef12"
# the default fusions without lc: the pair (the fold's fallback route)
NO_FOLD = {"WG_D3D12_FUSION": "pm,xe,mb,rv,xo,xd"}


def latent_out(coef):
    """The new latent a frame's coefficient store wrote (l_coef / l_coef_pack12, or the folded block)."""
    return coef.args[10] if coef.kernel == FOLD else coef.args[4]


def expected_args(t, f, digests):
    """Every Model L launch's arguments from the buffers' names (the runtime's wg_rt_buffer names after the frame) and
    the weight blobs' addresses (the upload holding each blob's bytes), independently of the launches themselves:
    {kernel (without _nc): [argument]}, None where the parameters E / D / P stand (checked field by field
    elsewhere)."""
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
    return {
        # the input encoder's values: Model M's encoder inputs (the colour as the network reads it, here the game's,
        # pre-exposure 1), the state it reads, q into the wide map, the motion surfaces, the exposure pair
        "lev/input_encoder": [n["input.color"], n["input.depth"], n["input.motion"], n["post.reduced"],
                              n["state.latent_prev"], n["state.motion_history"], n["l.wide"], n["enc0.m_near"],
                              n["enc0.m_far"], n["enc0.m_center"], n["exposure.pair"], w["enc0.values"], None],
        "l_linear_32_32_relu": [n["l.wide"], n["l.narrow"], w["enc0.Win1"], npix],  # q -> x1
        "l_linear_32_32_f16": [n["l.narrow"], n["l.wide"], w["enc0.Win2"], npix],   # x1 -> x2
        "l_swin_block_c32_enc": [n["l.wide"], "0", n["l.x3"], w["enc0_block"], str(fh), str(fw)] + eo,  # x2 -> x3
        "l_project_gather": [n["l.x3"], n["enc0.features"], w["enc0.project"], tw, th],  # x3 -> enc1's input
        "l_dec_project": [n["dec1.out"], n["l.narrow"], w["dec0.project"], tw, th],      # dec1's output -> y
        "l_swin_block_c32_dec": [n["l.narrow"], "0", n["l.wide"], w["dec0_block"], str(fh), str(fw)] + do,  # -> y2
        "l_coef": [n["l.wide"], n["dec0.A"], n["dec0.B"], n["dec0.C"], n["dec0.latent"], w["dec0.coef"], None],
        # the folded form: the block's arguments (no map out), then the pack12 store's images, latent, blob, rectangle
        FOLD_BASE: [n["l.narrow"], "0", "0", w["dec0_block"], str(fh), str(fw)] + do +
                   [n["dec0.A"], n["dec0.C"], n["dec0.latent"], w["dec0.coef"], str(info["rw"]), str(info["rh"])],
        # the reconstruction (Model M's post_recon arguments): colour, A, B, C, the previous history, this frame's
        # motion surfaces, the post blob, the new history, the reduced history, the display image, P, exposure pair
        "l_post_recon": [n["input.color"], n["dec0.A"], n["dec0.B"], n["dec0.C"], n["state.history_prev"],
                         n["enc0.m_near"], n["enc0.m_center"], n["enc0.m_far"], w["post"], n["post.history"],
                         n["post.reduced"], n["post.display"], None, n["exposure.pair"]],
    }


def base(kernel):
    """A Model L launch's kernel without its clamp-free suffix; the pack12 coefficient store (native D3D12) as l_coef,
    whose arguments it shares."""
    k = kernel.replace("_nc", "")
    return "l_coef" if k == "l_coef_pack12" else "l_post_recon" if k == "l_post_recon_p12" else k


def check_roles(t, f, digests, l_post=True):
    """Raises AssertionError naming every Model L launch argument that is not the expected buffer or blob."""
    exp = expected_args(t, f, digests)
    seen, bad = set(), []
    for ln in f["launches"]:
        k = base(ln.kernel)
        if k not in exp:
            continue
        seen.add(k)
        want = exp[k]
        if len(ln.args) != len(want):
            bad.append("%s: %d arguments, expected %d" % (k, len(ln.args), len(want)))
            continue
        bad += ["%s argument %d: %s, expected %s" % (k, i, a, e) for i, (a, e) in enumerate(zip(ln.args, want))
                if e is not None and a != e]
    fold = FOLD_BASE in seen
    need = [k for k in L_KERNELS if not (fold and k in ("l_swin_block_c32_dec", "l_coef"))] + ([FOLD_BASE] if fold else [])
    if fold and ({"l_swin_block_c32_dec", "l_coef"} & seen):
        bad.append("the folded decoder block and the pair in one frame")
    missing = [k for k in need if k not in seen and (l_post or k != "l_post_recon")]
    if bad or missing:
        raise AssertionError("; ".join(bad + ["not launched: " + k for k in missing]))


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


# Model L's D3D12 kernels the runtime checks at create (runtime.cpp l_native_missing), with their code objects
L_D3D12 = [("lev/input_encoder", "l_enc_values.co"), ("l_linear_32_32_relu", "l_ops.co"),
           ("l_linear_32_32_f16", "l_ops.co"), ("l_swin_block_c32_enc_nc", "l_swin.co"), ("l_project_gather", "l_ops.co"),
           ("swin_block_c64_add_ex_nc", "swin.co"), ("l_dec_project", "l_ops.co"), ("l_swin_block_c32_dec_nc", "l_swin.co"),
           ("l_coef_pack12", "l_ops.co")]


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


def cdiv(a, b):
    return -(-a // b)


def nc_flags():
    blobs = LP.pack_model_l(L.random_params(np.random.default_rng(SEED)))
    return [bool(int.from_bytes(blobs[n][:4], "little") & 1) for n in ("enc0_block", "dec0_block")]


def expected_l(m_frame, parity, fh, fw):
    """Model M's launches of one frame (on the unbanded post_recon route) with Model L's steps 2, 4 and 5
    (clean/l/FORMATS.md, "Kernels" and "Launch order of one frame"): kernel, grid, block."""
    nc = nc_flags()
    npix, tokens = (fh + 4) * (fw + 4), (fh // 2) * (fw // 2)

    def block(kind, offsets, i):
        ox, oy = T.block_offset(T.TABLE, offsets, parity)
        nwin = cdiv(fw + ox, 4) * cdiv(fh + oy, 4)
        return ("l_swin_block_c32_%s%s" % (kind, "_nc" if nc[i] else ""), cdiv(nwin, 8), 256)

    out = []
    for k, g, b in m_frame:
        if k == "input_encoder":
            out += [("lev/input_encoder", cdiv(((fh + 4) // 2) * ((fw + 4) // 2), 128), 256),
                    ("l_linear_32_32_relu", cdiv(npix, 128), 256), ("l_linear_32_32_f16", cdiv(npix, 128), 256),
                    block("enc", "first", 0), ("l_project_gather", cdiv(tokens, 128), 256)]
        elif k == "coef_decoder":
            out += [("l_dec_project", cdiv(tokens, 128), 256), block("dec", "second", 1),
                    ("l_coef", cdiv(tokens, 128), 256)]
        elif k.split("/")[-1] == "post_recon":  # Model L's reconstruction, same arguments and grid
            out.append(("l_post_recon", g, b))
        else:
            out.append((k, g, b))
    return out


class Frame(unittest.TestCase):
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
        l_mods = ["l_enc_values.co as lev", "l_ops.co", "l_swin.co", "l_post.co"]
        self.assertEqual([x for x in ll if x not in l_mods], [x for x in lm if x not in (km["enc0"], km["dec0"])])
        for want in l_mods:
            self.assertIn(want, ll)
        self.assertEqual(kl.get("model"), "L")
        self.assertEqual(kl.get("post_pack"), "none")
        self.assertEqual(kl.get("model_map"), "same")  # read from model.wgcm, checked against its map
        return fl

    def test_empty_build(self):
        self.check()

    def test_shipping_build(self):
        # l_coef writes Model M's base coefficient format: Model L runs without pack12 (and so without the fused pack12
        # dec1 + dec0), compared with Model M without it (WG_POST_PACK=none)
        fl = self.check(FT.SHIPPING, {"WG_POST_PACK": "none"})
        names = [k for k, _, _ in fl[0]]
        self.assertTrue(all(not n.startswith(("d0p12/", "d1p12/")) and "_p12" not in n for n in names), names)
        self.assertTrue(any(n.startswith("w64/") for n in names))  # the wave64 post (not a pack12 kernel) is kept

    def test_model_m_unchanged(self):
        """Model M in the build with Model L: the same trace as in the build without it (clean/tests)."""
        for files in ((), FT.SHIPPING):
            self.assertEqual(run(frames=2, build_files=files), run(frames=2, build_files=files, exe=FT.trace_exe()))


class Arguments(unittest.TestCase):
    """The recorder's view of every Model L launch (clean/l/FORMATS.md, "Maps" and "Kernels")."""

    def check_frame(self, t, f, roles=True, l_post=True):
        info, ls = f["info"], f["launches"]
        rw, rh = info["rw"], info["rh"]
        fw, fh = cdiv(rw, 32) * 32, cdiv(rh, 32) * 32
        npix, tw, th = (fh + 4) * (fw + 4), fw // 2, fh // 2
        k = {base(ln.kernel): (i, ln) for i, ln in enumerate(ls)}
        _, lev = k["lev/input_encoder"]
        _, relu = k["l_linear_32_32_relu"]
        _, f16 = k["l_linear_32_32_f16"]
        _, enc = k["l_swin_block_c32_enc"]
        gi, gather = k["l_project_gather"]
        di, dproj = k["l_dec_project"]
        fold = FOLD_BASE in k
        if fold:  # one launch: the block with the coefficient store as its epilogue (no y2 map)
            _, dec = k[FOLD_BASE]
            coef = dec
        else:
            _, dec = k["l_swin_block_c32_dec"]
            _, coef = k["l_coef"]
        q, x1, x2, x3, feat = lev.args[6], relu.args[1], f16.args[1], enc.args[2], gather.args[1]
        # the chain of maps (no launch reads what it writes)
        self.assertEqual(relu.args[0], q)
        self.assertEqual(f16.args[0], x1)
        self.assertEqual(enc.args[0], x2)
        self.assertEqual(enc.args[1], "0")  # xadd: none
        self.assertEqual(gather.args[0], x3)
        self.assertTrue(ls[gi + 1].kernel.startswith("swin_block_c64"))
        self.assertEqual(ls[gi + 1].args[0], feat)  # enc1 reads Model L's features
        dec1 = [ln for ln in ls[:di] if ln.kernel.startswith("swin_block_c64")][-1]
        y, y2 = dproj.args[1], dec.args[2]
        self.assertEqual(dproj.args[0], dec1.args[2])  # the decoder reads dec1's output
        self.assertEqual(dec.args[0], y)
        self.assertEqual(dec.args[1], "0")
        if fold:
            self.assertEqual(y2, "0")  # the map is not written
        else:
            self.assertEqual(coef.args[0], y2)
        for a, b in ((q, x1), (x1, x2), (x2, x3), (x3, feat), (dec1.args[2], y), (y, y2)):
            self.assertNotEqual(a, b)
        # every buffer holds its map (FORMATS.md "Maps")
        maps = ((q, 32 * npix), (x1, 32 * npix), (x2, 64 * npix), (x3, 32 * fh * fw), (feat, 64 * tw * th),
                (y, 32 * fh * fw)) + (() if fold else ((y2, 64 * fh * fw),))
        if fold:  # pack12 image A (8 bytes a pixel), image C (4), the new latent
            coefs = ((coef.args[8], 8 * rw * rh), (coef.args[9], 4 * rw * rh), (coef.args[10], 4 * rw * rh))
        else:
            coefs = ((coef.args[1], 4 * rw * rh), (coef.args[2], 4 * rw * rh), (coef.args[3], 8 * rw * rh),
                     (coef.args[4], 4 * rw * rh))  # A, B, C, the new latent: Model M's formats
        for buf, need in maps + coefs:
            self.assertGreaterEqual(t.extent(buf), need, buf)
        # extents, offsets, parameters
        self.assertEqual(relu.args[3], str(npix))
        self.assertEqual(f16.args[3], str(npix))
        self.assertEqual(gather.args[3:5], [str(tw), str(th)])
        self.assertEqual(dproj.args[3:5], [str(tw), str(th)])
        for blk, kind in ((enc, "first"), (dec, "second")):
            ox, oy = T.block_offset(T.TABLE, kind, info["parity"])
            self.assertEqual(blk.args[4:8], [str(fh), str(fw), str(ox), str(oy)])
        e = lev.params(12)
        self.assertEqual(e[:4], [rw, rh, fw, fh])
        self.assertEqual(e[5], info["reset"])
        if fold:
            self.assertEqual(coef.args[12:14], [str(rw), str(rh)])
        else:
            self.assertEqual(coef.params(6), [rw, rh, fw, fh, info["reset"]])
        post = [ln for ln in ls if base(ln.kernel) == "l_post_recon"]
        self.assertEqual(len(post), 1 if l_post else 0)  # (native D3D12: Model M's fused pack12 post by default)
        if l_post:
            self.assertEqual(post[0].params(12), [rw, rh, (3 * 120 + 1) // 2, (3 * 72 + 1) // 2, info["reset"]])
            self.assertEqual(post[0].grid, cdiv((3 * 120 + 1) // 2, 32) * cdiv(cdiv((3 * 72 + 1) // 2, 2), 8))
            self.assertTrue(ls[ls.index(post[0]) + 1].kernel.endswith("post_filter_c1_s4"))  # Model M's filter next
        self.assertTrue(lev.args[11].startswith("b"))  # the values blob: an allocation
        if roles:  # every argument against the named buffers and blobs
            check_roles(t, f, blob_digests(), l_post)
        return lev, coef

    def test_sizes_reset_history(self):
        sizes = "--sizes=96x96,96x96,80x48,80x48,80x48,64x64"  # aligned, non-aligned, a smaller aligned size
        t = FT.Trace(run("model=L", frames=6, rw=96, rh=96, flags=(sizes, "--reset=4")))
        self.assertEqual(t.create, "create ok model L")
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
        t = FT.Trace(run("model=L", frames=3, flags=("--fail=l_coef@1",)))
        ok0, bad, ok2 = t.frames
        self.assertEqual(bad["failed"], "l_coef")
        self.assertIn("injected launch failure: l_coef", bad["error"])
        names = [ln.kernel for ln in bad["launches"]]
        self.assertEqual(names, [ln.kernel for ln in ok0["launches"]][:len(names)])
        self.assertTrue(names[-1].startswith("l_swin_block_c32_dec"))  # stopped right before l_coef
        self.assertIsNone(ok2["error"])
        self.check_frame(t, ok2)


class Roles(unittest.TestCase):
    """Buffer identities: each Model L launch's weight blob, inputs and outputs are the named ones, and a mix-up of two
    of them fails even where the map chain, sizes and argument kinds stay as they were."""

    MIXUPS = (  # (kernel, argument, kernel, argument) swapped in a recorded frame
        ("l_linear_32_32_relu", 2, "l_linear_32_32_f16", 2),        # enc0.Win1 <-> enc0.Win2
        ("l_swin_block_c32_enc", 3, "l_swin_block_c32_dec", 3),     # enc0_block <-> dec0_block
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

    @classmethod
    def setUpClass(cls):
        cls.trace = FT.Trace(run("model=L", frames=3, rw=96, rh=64, flags=("--sizes=96x64,80x48,80x48",)))

    def test_every_argument_named(self):
        t, digests = self.trace, blob_digests()
        addrs = [t.uploaded(d) for d in digests.values()]
        self.assertNotIn(None, addrs)
        self.assertEqual(len(set(addrs)), len(addrs))  # nine distinct blobs, each in its own allocation
        for f in t.frames:
            self.assertEqual(len(f["names"]) >= 30, True, sorted(f["names"]))
            check_roles(t, f, digests)

    def test_mixups_fail(self):
        digests = blob_digests()
        for ka, ia, kb, ib in self.MIXUPS:
            with self.subTest(mixup=(ka, ia, kb, ib)):
                t = copy.deepcopy(self.trace)
                f = t.frames[1]
                ls = {ln.kernel.replace("_nc", ""): ln for ln in f["launches"]}
                a, b = ls[ka], ls[kb]
                a.args[ia], b.args[ib] = b.args[ib], a.args[ia]
                with self.assertRaises(AssertionError):
                    check_roles(t, f, digests)

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
        t = FT.Trace(run("model=L", frames=2, exe=exe))
        digests = blob_digests()
        for f in t.frames:
            Arguments.check_frame(self, t, f, roles=False)
            with self.assertRaises(AssertionError) as cm:
                check_roles(t, f, digests)
            msg = str(cm.exception)
            self.assertIn("l_linear_32_32_relu argument 2", msg)
            self.assertIn("l_linear_32_32_f16 argument 2", msg)
            self.assertEqual(msg.count("argument"), 2, msg)


class D3D12(unittest.TestCase):
    """Model L on native D3D12 (backend 3): Model M's native frame with Model L's encoder and decoder, the pack12
    coefficient store for Model M's pack12 post path; each missing D3D12 port refused by name, never replaced."""

    def l_run(self, ports, frames=3, flags=(), **kw):
        return run("model=L", frames=frames, build_files=FT.SHIPPING, flags=("--backend=3",) + tuple(flags),
                   manifest=ports, **kw)

    def test_missing_ports_named(self):
        have = [k for k, _ in declared_ports()]
        missing = [k for k, _ in L_D3D12 if k not in have]
        got = self.l_run(declared_ports(), frames=1)
        if not missing:
            self.assertEqual(got[0], "create ok model L")
            return
        self.assertEqual(got, ["ERR create: Model L on native D3D12: not yet ported to D3D12: " + ", ".join(missing)])

    def test_each_port_named(self):
        for k, _ in L_D3D12:
            with self.subTest(port=k):
                got = self.l_run([kc for kc in full_ports() if kc[0] != k], frames=1)
                self.assertEqual(got, ["ERR create: Model L on native D3D12: not yet ported to D3D12: " + k])

    def check_d3d12_frame(self, t, f):
        """The #111 checks (map chain, extents, offsets, parameters, every argument's buffer or blob), then what the
        native frame adds: the pack12 store's images, dec1's own block, Model M's pack12 post path after it."""
        info = f["info"]
        rw, rh = info["rw"], info["rh"]
        names = [ln.kernel for ln in f["launches"]]
        fold = FOLD in names
        if fold:
            self.assertNotIn("l_coef_pack12", names)
            self.assertNotIn("l_swin_block_c32_dec_nc", names)
            self.assertEqual(names[names.index(FOLD) - 1], "l_dec_project")
        else:
            self.assertIn("l_coef_pack12", names)
        self.assertNotIn("l_coef", names)
        self.assertTrue(all(not n.startswith(("d1p12/", "d0p12/")) and n != "input_encoder" for n in names), names)
        dec1 = [ln for ln in f["launches"] if ln.kernel.startswith("swin_block_c64_add")]
        self.assertEqual([ln.kernel for ln in dec1], ["swin_block_c64_add_ex_nc"])  # dec1's own block, not fused
        lev, coef = Arguments.check_frame(self, t, f, l_post=False)
        a, c = (coef.args[8], coef.args[9]) if fold else (coef.args[1], coef.args[3])
        self.assertGreaterEqual(t.extent(a), 8 * rw * rh)  # pack12 image A: 8 bytes per pixel
        self.assertGreaterEqual(t.extent(c), 4 * rw * rh)  # pack12 image C: 4 bytes per pixel
        i = names.index(FOLD if fold else "l_coef_pack12")
        self.assertTrue(names[i + 1].startswith("w64/post_tf") and "_p12" in names[i + 1], names[i + 1:])
        return lev, coef

    def test_frame(self):
        """Sizes aligned, non-aligned and changing, a game reset; history handed from frame to frame."""
        # upscaling sizes the pack12 post path takes (D = 180 x 108): aligned, non-aligned, a change of size
        sizes = "--sizes=96x64,96x64,80x48,80x48,80x48,77x45"
        t = FT.Trace(self.l_run(full_ports(), frames=6, rw=96, rh=64, flags=(sizes, "--reset=4")))
        self.assertEqual(t.create, "create ok model L")
        self.assertEqual(t.unknown(), [])
        self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 1, 0, 1, 1])
        self.assertEqual(len({f["info"]["parity"] for f in t.frames}), 2)
        for env, ports, fold in ((None, full_ports(), True), (NO_FOLD, full_ports(), False),
                                 (None, [kc for kc in full_ports() if kc[0] != FOLD], False)):
            with self.subTest(fusion=env, fold_in_manifest=any(kc[0] == FOLD for kc in ports)):
                # the fold (fusion lc, the default) and its fallback, the pair: without lc, or without the kernel
                t = FT.Trace(self.l_run(ports, frames=6, rw=96, rh=64, flags=(sizes, "--reset=4"), env=env))
                self.assertEqual(t.create, "create ok model L")
                self.assertEqual(t.unknown(), [])
                self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 1, 0, 1, 1])
                self.assertEqual(len({f["info"]["parity"] for f in t.frames}), 2)
                prev = None
                for f in t.frames:
                    self.assertIsNone(f["error"])
                    self.assertEqual(FOLD in [ln.kernel for ln in f["launches"]], fold)
                    lev, coef = self.check_d3d12_frame(t, f)
                    if prev:
                        self.assertEqual(lev.args[4], latent_out(prev[1]))  # the latent the last frame's store wrote
                        self.assertEqual(lev.args[5], prev[0].args[7])  # its near motion
                    prev = (lev, coef)

    def test_failed_launch_stops_the_frame(self):
        for env, kernel, before in ((None, FOLD, "l_dec_project"), (NO_FOLD, "l_coef_pack12", "l_swin_block_c32_dec")):
            with self.subTest(kernel=kernel):
                t = FT.Trace(self.l_run(full_ports(), frames=3, flags=("--fail=%s@1" % kernel,), env=env))
                ok0, bad, ok2 = t.frames
                self.assertEqual(bad["failed"], kernel)
                names = [ln.kernel for ln in bad["launches"]]
                self.assertEqual(names, [ln.kernel for ln in ok0["launches"]][:len(names)])
                self.assertTrue(names[-1].startswith(before))
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

    # runtime option l_post=own: Model L's own reconstruction on native D3D12 (its D3D12 ports, then the analytic filter)
    OWN = [("l_post_recon_p12", "l_post.co"), ("l_post_recon", "l_post.co"), ("l_coef", "l_ops.co"),
           ("w64/post_filter_c1_s4", "post_w64.co")]

    def own_ports(self):
        have = full_ports()
        return have + [kc for kc in self.OWN if kc not in have]

    def test_own_post(self):
        """l_post=own: l_post_recon_p12 on the frames l_coef_pack12 stores, with post_recon's arguments (the #111 role
        checks), then the unbanded filter w64/post_filter_c1_s4 from its display image into the output; no fused post.
        A frame the pack12 kernels do not take (render 180 x 108 = D) runs l_coef and l_post_recon."""
        sizes = "--sizes=96x64,96x64,180x108,80x48,77x45,180x108"
        t = FT.Trace(run("model=L;l_post=own", frames=6, build_files=FT.SHIPPING, rw=180, rh=108,
                         flags=("--backend=3", sizes, "--reset=4"), manifest=self.own_ports()))
        self.assertEqual(t.create, "create ok model L")
        self.assertEqual(t.unknown(), [])
        self.assertEqual(t.kernels.get("l_post"), "l_post_recon")
        prev = None
        for f in t.frames:
            self.assertIsNone(f["error"])
            info, ls = f["info"], f["launches"]
            names = [ln.kernel for ln in ls]
            p12 = (info["rw"], info["rh"]) != (180, 108)  # pack12 frames: the fold (fusion lc, the default)
            self.assertIn(FOLD if p12 else "l_coef", names)
            self.assertFalse({"l_coef_pack12", "l_coef" if p12 else FOLD} & set(names), names)
            self.assertFalse([n for n in names if "post_tf" in n or "post_recon2" in n or n.endswith("_band")], names)
            post = [ln for ln in ls if ln.kernel == ("l_post_recon_p12" if p12 else "l_post_recon")]
            self.assertEqual(len(post), 1, names)
            lev, coef = Arguments.check_frame(self, t, f)  # the roles of every argument, the filter after the post
            flt = ls[ls.index(post[0]) + 1]
            self.assertEqual(flt.kernel, "w64/post_filter_c1_s4")
            self.assertEqual((flt.grid, flt.block), (cdiv(120, 40) * cdiv(cdiv(72, 2), 8), 64))
            self.assertEqual(flt.args[:2], [f["names"]["post.display"], f["names"]["filter.out"]])
            self.assertEqual(flt.args[2:6], [str((3 * 120 + 1) // 2), str((3 * 72 + 1) // 2), "120", "72"])
            self.assertEqual(flt.args[7], f["names"]["exposure.pair"])
            if p12:  # pack12 image A (the fold's argument 8) is what the reconstruction reads
                self.assertEqual(coef.args[8], post[0].args[1])
                self.assertGreaterEqual(t.extent(coef.args[8]), 8 * info["rw"] * info["rh"])
            if prev:
                self.assertEqual(lev.args[4], latent_out(prev[1]))  # the latent the last frame's store wrote
            prev = (lev, coef)

    def test_own_post_refusals(self):
        def first(options, **kw):
            return run(options, frames=1, build_files=FT.SHIPPING, **kw)[0]
        d3d12 = dict(flags=("--backend=3",), manifest=self.own_ports())
        self.assertEqual(first("model=L;l_post=yes", **d3d12),
                         "ERR create: l_post must be m or own (Model L on native D3D12)")
        self.assertTrue(first("model=L;l_post=own", hdr=0, **d3d12).startswith(
            "ERR create: Model L: l_post=own needs HDR colour"))
        self.assertEqual(first("l_post=own", **d3d12), "ERR create: l_post=own is a Model L option (its own "
                                                        "reconstruction)")
        for k, _ in self.OWN:
            with self.subTest(port=k):
                got = first("model=L;l_post=own", flags=("--backend=3",),
                            manifest=[kc for kc in self.own_ports() if kc[0] != k])
                self.assertEqual(got, "ERR create: Model L on native D3D12 (l_post=own): not yet ported to D3D12: " + k)
        # the default stays Model M's fused pack12 post (test_frame), named so; on HIP Model L runs its own post either way
        self.assertEqual(FT.Trace(run("model=L", frames=1, build_files=FT.SHIPPING, **d3d12)).kernels.get("l_post"),
                         "m_pack12")

        def hip(options):
            return [re.sub(r"wg_lfb_\w+", "B", x) for x in run(options, frames=2, build_files=FT.SHIPPING)]
        self.assertEqual(hip("model=L;l_post=own"), hip("model=L"))


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
        # LDR colour: Model L's LDR reconstruction kernel is not written yet (refused, never Model M's post instead)
        self.assertTrue(run("model=L", hdr=0)[0].startswith(
            "ERR create: Model L: LDR colour (WgRtConfig.hdr = 0) needs Model L's LDR reconstruction kernel"))

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


if __name__ == "__main__":
    unittest.main()
