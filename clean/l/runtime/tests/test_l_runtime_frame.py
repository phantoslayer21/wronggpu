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
import os
import shutil
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


def run(options="", frames=4, build_files=(), env=None, model=None, exe=None, flags=(), rw=RW, rh=RH):
    """The trace program's output lines (WG_* removed from the environment, then env applied)."""
    build = tempfile.mkdtemp(prefix="wg_lfb_")
    for n in build_files:
        open(os.path.join(build, n), "wb").close()
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
    e.update(env or {})
    try:
        r = subprocess.run([exe or trace_exe(), build, model or model_dir(), str(rw), str(rh), "120", "72", str(frames),
                            options, "--format=0", "--map=" + map_path(), *flags], capture_output=True, text=True,
                           env=e, timeout=600)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if r.returncode != 0:
        raise AssertionError("runtime_frame_trace failed (exit %d)\n%s%s" % (r.returncode, r.stdout, r.stderr))
    return r.stdout.splitlines()


def cdiv(a, b):
    return -(-a // b)


def nc_flags():
    blobs = LP.pack_model_l(L.random_params(np.random.default_rng(SEED)))
    return [bool(int.from_bytes(blobs[n][:4], "little") & 1) for n in ("enc0_block", "dec0_block")]


def expected_l(m_frame, parity, fh, fw):
    """Model M's launches of one frame with Model L's steps 2 and 4 (clean/l/FORMATS.md, "Kernels" and "Launch order
    of one frame"): kernel, grid, block."""
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
        else:
            out.append((k, g, b))
    return out


class Frame(unittest.TestCase):
    def check(self, build_files=(), m_env=None):
        fh, fw = cdiv(RH, 32) * 32, cdiv(RW, 32) * 32
        cm, lm, fm, km = FT.parse(run(build_files=build_files, env=m_env))
        cl, ll, fl, kl = FT.parse(run("model=L", build_files=build_files))
        self.assertEqual(cm, "create ok model M")
        self.assertEqual(cl, "create ok model L")
        self.assertEqual(len(fl), 4)
        for i, (m, l_) in enumerate(zip(fm, fl)):
            names = [k for k, _, _ in m]
            self.assertIn("input_encoder", names)
            self.assertIn("coef_decoder", names)  # Model M's base decoder: the one Model L's l_coef stands in for
            self.assertEqual(l_, expected_l(m, i & 1, fh, fw), "frame %d" % i)
        l_mods = ["l_enc_values.co as lev", "l_ops.co", "l_swin.co"]
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

    def check_frame(self, t, f):
        info, ls = f["info"], f["launches"]
        rw, rh = info["rw"], info["rh"]
        fw, fh = cdiv(rw, 32) * 32, cdiv(rh, 32) * 32
        npix, tw, th = (fh + 4) * (fw + 4), fw // 2, fh // 2
        k = {ln.kernel.replace("_nc", ""): (i, ln) for i, ln in enumerate(ls)}
        _, lev = k["lev/input_encoder"]
        _, relu = k["l_linear_32_32_relu"]
        _, f16 = k["l_linear_32_32_f16"]
        _, enc = k["l_swin_block_c32_enc"]
        gi, gather = k["l_project_gather"]
        di, dproj = k["l_dec_project"]
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
        self.assertEqual(coef.args[0], y2)
        for a, b in ((q, x1), (x1, x2), (x2, x3), (x3, feat), (dec1.args[2], y), (y, y2)):
            self.assertNotEqual(a, b)
        # every buffer holds its map (FORMATS.md "Maps")
        maps = ((q, 32 * npix), (x1, 32 * npix), (x2, 64 * npix), (x3, 32 * fh * fw), (feat, 64 * tw * th),
                (y, 32 * fh * fw), (y2, 64 * fh * fw))
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
        e, d = lev.params(12), coef.params(6)
        self.assertEqual(e[:4], [rw, rh, fw, fh])
        self.assertEqual(e[5], info["reset"])
        self.assertEqual(d, [rw, rh, fw, fh, info["reset"]])
        self.assertTrue(lev.args[11].startswith("b"))  # the values blob: an allocation
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
