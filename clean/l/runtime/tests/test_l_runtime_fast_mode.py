# SPDX-License-Identifier: MIT
"""Model L's quality preset l_mode (runtime option l_mode=hybrid|fast, env WG_L_MODE) on the CPU: the runtime built with
Model L on the recording device of clean/tests/runtime_frame_trace.cpp (launches recorded with their arguments, not
executed; no GPU).

- hybrid is the default and is untouched: nothing set, l_mode=hybrid, WG_L_MODE=hybrid and WG_L_MODE= (empty) trace the same
  frames as each other, and a manifest without the fast-only kernels changes nothing for them (the hybrid route never
  names a fast-only kernel, so it loads nothing of them).
- the option and the environment: a wrong value is refused with its source named; the runtime option wins over WG_L_MODE
  and WG_RUNTIME_OPTIONS over the configuration string; fast is native D3D12 only and has no WG_L_FORM next to it.
- fast on native D3D12: the M-form route on Model L's own values (Model M's packers without the hybrid values:
  clean/runner/packing.py pack_stage / l_pack.pack_model_l), Model M's "_nc" Swin kernels with Model M's fusions, Model L's
  encoder block with the linears (le), its decoder block with the skip S = the encoder block's map and the pack12 store
  (lc, l_swin_block_c32_dec_coef12_add_nc; unfused: l_swin_block_c32_dec_add_nc + l_coef_pack12), the C1/C2
  reconstruction; no hybrid twin, no no-skip decoder block. Every kernel the route needs is refused by name when missing.

Built with the host C++ compiler (clean/tests/host_cxx.py; skipped without one).

Run (repository root):
    python -m unittest discover -s clean/l/runtime/tests -t clean/l/runtime/tests -p test_l_runtime_fast_mode.py
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

import test_l_runtime_frame as F
from clean.l.reference import oracle as L
from clean.l.runner import l_pack as LP
from clean.runner import packing as P
from clean.runner import variants as V
from clean.tests import test_runtime_frame_trace as FT

FAST_ONLY = ("l_swin_block_c32_dec_add_nc", "l_swin_block_c32_dec_coef12_add_nc")
FOLD_ADD = "l_swin_block_c32_dec_coef12_add_nc"
DEC_ADD = "l_swin_block_c32_dec_add_nc"
ENC_LIN = "l_swin_block_c32_enc_lin_nc"
M_FORM_SHARED = ("swin_block_c64_nc", "swin_block_c96_nc", "swin_block_c96_pm_nc", "swin_block_c96_pm_rev_nc",
                 "swin_block_c96_add_ex_nc", "swin_block_c64_add_ex_nc", "swin_block_c96_add_xe_nc",
                 "swin_block_c96_add_xe_rev_nc", "swin_block_c96_add_xe_xo_nc", "patch_merge_c64", "patch_merge_c96",
                 "bneck/swin_bneck6_nc", "bneck/swin_bneck6_mb_nc", "bneck/swin_bneck6_mbo_nc")
# what runtime.cpp l_native_missing asks of the fast route (after the release route's own checks: lc12 reconstruction)
FAST_NEEDS = ("lev/input_encoder", "lev/input_encoder_q20", "l_linear_32_32_relu", "l_linear_32_32_f16",
              "l_swin_block_c32_enc_nc", ENC_LIN, "l_project_gather", "l_dec_project", DEC_ADD, FOLD_ADD, "l_coef_pack12",
              "swin_block_c64_nc", "patch_merge_c64", "swin_block_c96_nc", "patch_merge_c96", "bneck/swin_bneck6_nc",
              "swin_block_c96_add_ex_nc", "swin_block_c64_add_ex_nc")
SIZES = "--sizes=96x64,96x64,80x48,80x48,80x48,77x45"


def l_run(options="model=L", ports=None, frames=3, flags=(), backend=3, rw=F.RW, rh=F.RH, hdr=1, env=None,
          tf_default=False, variants=True):
    """The trace program's lines (the arguments of test_l_runtime_frame.run). variants: the build folder holds the
    variants.json and swin__bmegahg.co of a full build (the bottleneck in one launch by default, Model M's geometry
    variants; without them the six block launches, which D3D12 has no port of)."""
    ports = F.full_ports() if ports is None else ports
    build = tempfile.mkdtemp(prefix="wg_lfb_")
    for n in FT.SHIPPING + (("swin__bmegahg.co",) if variants else ()):
        open(os.path.join(build, n), "wb").close()
    if variants:
        with open(os.path.join(build, V.MANIFEST), "w") as fh:
            json.dump({n: {"file": V.code_object(n), "defines": V.all_builds().get(n, {}),
                           "geometry": V.variant_geometry(n), "ablation": False} for n in ("base", "bmegahg")}, fh)
    os.makedirs(os.path.join(build, "dx12"))
    with open(os.path.join(build, "dx12", "manifest.json"), "w") as fh:
        json.dump({"format": 1, "kernels": [{"name": k, "co": co} for k, co in ports]}, fh)
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("WG_")}
    if not tf_default:   # the separate launches (WG_L_POST_TF=0), unless the test is of the shipped defaults
        e["WG_L_POST_TF"] = "0"
    e.update(env or {})
    try:
        r = subprocess.run([F.trace_exe(), build, F.model_dir(), str(rw), str(rh), "120", "72", str(frames), options,
                            "--format=0", "--map=" + F.map_path(), "--hdr=%d" % hdr, "--backend=%d" % backend, *flags],
                           capture_output=True, text=True, env=e, timeout=600)
    finally:
        shutil.rmtree(build, ignore_errors=True)
    if r.returncode != 0:
        raise AssertionError("runtime_frame_trace failed (exit %d)\n%s%s" % (r.returncode, r.stdout, r.stderr))
    return r.stdout.splitlines()


def without(*ks):
    return [kc for kc in F.full_ports() if kc[0] not in ks]


def names(frame):
    return [ln.kernel for ln in frame["launches"]]


def fast_digests():
    """{blob name: SHA-256 hex} of the blobs Model L's fast route reads: Model L's own launches (pack_model_l on the
    parameters as they are) and the ten shared stages by Model M's packer, on the parameters as they are (no hybrid
    values), the post blob as every route's."""
    params = L.random_params(np.random.default_rng(F.SEED))
    blobs = LP.pack_model_l(params)
    blobs.update({s: P.pack_stage(s, params[s], "gfx12", model="M") for s in F.SHARED})
    import hashlib
    import struct
    blobs["post"] = struct.pack("<f", 1.25) + bytes(12)
    return {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}


class Mode(unittest.TestCase):
    """l_mode, WG_L_MODE: the default, the values, the precedence, the backends."""

    def test_hybrid_is_the_default(self):
        a = F.same_frames(l_run("model=L", frames=4, rw=96, rh=64, flags=(SIZES,), variants=False))
        self.assertEqual(a[0], "create ok model L")
        kernels = [x for x in a if x.startswith("kernels ")][0]
        self.assertIn("l_hyb_d3d12=on", kernels)
        self.assertNotIn("l_mode=", kernels)   # the hybrid default's report is the release's
        for options, env in (("model=L,l_mode=hybrid", None), ("model=L", {"WG_L_MODE": "hybrid"}),
                             ("model=L", {"WG_L_MODE": ""}), ("model=L,l_mode=", None),
                             ("model=L,l_mode=hybrid", {"WG_L_MODE": "fast"})):
            with self.subTest(options=options, env=env):
                self.assertEqual(F.same_frames(l_run(options, frames=4, rw=96, rh=64, flags=(SIZES,), env=env, variants=False)), a)

    def test_hybrid_needs_no_fast_kernel(self):
        """The hybrid route names no fast-only kernel: a manifest without them traces the same frames (they are not
        loaded for it), and no hybrid frame launches one."""
        a = F.same_frames(l_run("model=L", frames=3, rw=96, rh=64, variants=False))
        b = F.same_frames(l_run("model=L", ports=without(*FAST_ONLY), frames=3, rw=96, rh=64, variants=False))
        self.assertEqual(a, b)
        t = FT.Trace(l_run("model=L", frames=3, rw=96, rh=64, variants=False))
        self.assertEqual([n for f in t.frames for n in names(f) if n in FAST_ONLY], [])

    def test_warm_up_scope(self):
        """The device's warm-up (a pipeline state for every kernel of the build) leaves out what the mode never launches:
        hybrid (the default) the two fast-only decoder blocks and nothing else; fast the hybrid twins, the FP16-form q7
        blocks and the no-skip decoder blocks, none of the kernels its own route launches. Model M: nothing."""
        ports = [k for k, _ in F.full_ports()]
        t = FT.Trace(l_run("model=L", frames=1, rw=96, rh=64, variants=False, flags=("--warm-skip",)))
        self.assertEqual(sorted(t.warm_skip), sorted(FAST_ONLY))
        t = FT.Trace(l_run("model=L", ports=without(*FAST_ONLY), frames=1, rw=96, rh=64, variants=False,
                           flags=("--warm-skip",)))
        self.assertEqual(t.warm_skip, [])
        t = FT.Trace(l_run("model=L,l_mode=fast", frames=1, rw=96, rh=64, flags=("--warm-skip",)))
        twins = [k for k in ports if k.endswith("_hyb")]
        self.assertTrue(twins)
        want = set(twins) | {k for k in ports if k.startswith("l_q7_")} | {"l_swin_block_c32_dec_nc",
                                                                           "l_swin_block_c32_dec_coef12_nc"}
        self.assertEqual(sorted(t.warm_skip), sorted(want))
        launched = {n for f in t.frames for n in names(f)}
        self.assertEqual(launched & set(t.warm_skip), set())
        self.assertEqual(set(FAST_NEEDS) & set(t.warm_skip), set())
        t = FT.Trace(F.run("", frames=1, build_files=FT.SHIPPING, flags=("--backend=3", "--warm-skip"),
                           manifest=F.full_ports()))
        self.assertEqual(t.warm_skip, [])

    def test_values_are_checked(self):
        got = l_run("model=L,l_mode=bogus", frames=1)
        self.assertEqual(got, ['ERR create: l_mode must be hybrid or fast (got "bogus")'])
        got = l_run("model=L", frames=1, env={"WG_L_MODE": "Fast"})
        self.assertEqual(got, ['ERR create: WG_L_MODE must be hybrid or fast (got "Fast")'])
        got = l_run("model=L", frames=1, env={"WG_RUNTIME_OPTIONS": "l_mode=exact"})
        self.assertEqual(got, ['ERR create: l_mode must be hybrid or fast (got "exact")'])

    def test_precedence(self):
        """The option over WG_L_MODE; WG_RUNTIME_OPTIONS over the configuration string (as every runtime option)."""
        def mode(options, env=None):
            t = FT.Trace(l_run(options, frames=1, rw=96, rh=64, env=env))
            self.assertEqual(t.create, "create ok model L")
            return t.kernels.get("l_mode", "hybrid")
        self.assertEqual(mode("model=L,l_mode=fast"), "fast")
        self.assertEqual(mode("model=L", {"WG_L_MODE": "fast"}), "fast")
        self.assertEqual(mode("model=L,l_mode=hybrid", {"WG_L_MODE": "fast"}), "hybrid")
        self.assertEqual(mode("model=L,l_mode=fast", {"WG_L_MODE": "hybrid"}), "fast")
        self.assertEqual(mode("model=L,l_mode=hybrid", {"WG_RUNTIME_OPTIONS": "l_mode=fast"}), "fast")
        self.assertEqual(mode("model=L,l_mode=fast", {"WG_RUNTIME_OPTIONS": "l_mode=hybrid"}), "hybrid")
        self.assertEqual(mode("model=L,l_mode=fast;l_mode=hybrid"), "hybrid")   # the later key wins

    def test_fast_is_native_d3d12_only(self):
        got = l_run("model=L,l_mode=fast", frames=1, backend=0)
        self.assertEqual(got, ["ERR create: l_mode=fast runs on native D3D12 (backend 3) only (l_mode=hybrid is the "
                               "default on every backend)"])
        # hybrid is the default of every backend: HIP creates with l_mode=hybrid as without it
        self.assertEqual(l_run("model=L,l_mode=hybrid", frames=1, backend=0)[0], "create ok model L")

    def test_fast_has_no_form_next_to_it(self):
        for form in ("hybrid", "q7"):
            with self.subTest(form=form):
                got = l_run("model=L,l_mode=fast", frames=1, env={"WG_L_FORM": form})
                self.assertEqual(got, ["ERR create: l_mode=fast cannot be combined with WG_L_FORM=%s (the fast route is "
                                       "its own form: unset WG_L_FORM)" % form])

    def test_model_m_ignores_the_option(self):
        """l_mode is Model L's: Model M creates and traces as without it."""
        a = F.run("", frames=2, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=F.full_ports())
        b = F.run("l_mode=fast", frames=2, build_files=FT.SHIPPING, flags=("--backend=3",), manifest=F.full_ports())
        self.assertEqual([re.sub(r"wg_lfb_\w+", "B", x) for x in a], [re.sub(r"wg_lfb_\w+", "B", x) for x in b])
        self.assertTrue(a[0].startswith("create ok model M"), a[0])


class Fast(unittest.TestCase):
    """The fast route's dispatch plan on native D3D12."""

    def trace(self, ports=None, frames=3, env=None, tf=False, **kw):
        t = FT.Trace(l_run("model=L,l_mode=fast", ports=ports, frames=frames, env=env, tf_default=tf, **kw))
        self.assertEqual(t.create, "create ok model L", t.create)
        self.assertEqual(t.unknown(), [])
        return t

    def check_frame(self, t, f, fold=True, enc_lin=True):
        n, ls, nm = names(f), f["launches"], f["names"]
        self.assertTrue(all(not k.endswith(("_hyb", "_q7c_hyb")) and "l_q7" not in k for k in n), n)
        self.assertNotIn("l_swin_block_c32_dec_coef12_nc", n)    # the no-skip decoder blocks are never launched
        self.assertNotIn("l_swin_block_c32_dec_nc", n)
        self.assertEqual(any(k in n for k in M_FORM_SHARED), True, n)   # Model M's _nc kernels on Model L's values
        # the decoder block: the skip S is the encoder block's map l_x3 (xadd), argument 1
        dec = [ln for ln in ls if ln.kernel in (FOLD_ADD, DEC_ADD)]
        self.assertEqual([ln.kernel for ln in dec], [FOLD_ADD if fold else DEC_ADD])
        self.assertEqual(dec[0].args[1], nm["l.x3"])
        enc = [ln for ln in ls if ln.kernel in (ENC_LIN, "l_swin_block_c32_enc_nc")]
        self.assertEqual([ln.kernel for ln in enc], [ENC_LIN if enc_lin else "l_swin_block_c32_enc_nc"])
        self.assertEqual(enc[0].args[2], nm["l.x3"])             # (the encoder block writes l_x3: out, argument 2)
        self.assertLess(n.index(enc[0].kernel), n.index("l_project_gather"))
        self.assertLess(n.index("l_dec_project"), n.index(dec[0].kernel))
        self.assertEqual(dec[0].args[0], nm["l.narrow"])          # y: l_dec_project's output
        if fold:   # the pack12 store is the block's epilogue: A, C, latent, dec0.coef, rw, rh
            self.assertEqual(dec[0].args[8:12], [nm["dec0.A"], nm["dec0.C"], nm["dec0.latent"], dec[0].args[11]])
            self.assertNotIn("l_coef_pack12", n)
        else:      # the pair: the block's FP16 map y2 (out, argument 2) is l_coef_pack12's input
            self.assertEqual(dec[0].args[2], nm["l.wide"])
            self.assertEqual(n[n.index(DEC_ADD) + 1], "l_coef_pack12")
        return n

    def test_frame(self):
        """Sizes aligned, non-aligned and changing, a game reset, both window parities; the history handed on."""
        t = self.trace(frames=6, rw=96, rh=64, flags=(SIZES, "--reset=4"))
        self.assertEqual(t.kernels.get("l_mode"), "fast")
        self.assertEqual(t.kernels.get("l_form"), "fast")
        self.assertEqual(t.kernels.get("l_dec_skip"), "l_x3")
        self.assertNotIn("l_hyb_d3d12", t.kernels)
        self.assertEqual([f["info"]["reset"] for f in t.frames], [1, 0, 1, 0, 1, 1])
        self.assertEqual(len({f["info"]["parity"] for f in t.frames}), 2)
        prev = None
        for f in t.frames:
            self.assertIsNone(f["error"])
            n = self.check_frame(t, f)
            self.assertEqual(n[n.index(FOLD_ADD) + 1], "lc12/l_post_recon_p12", n)   # the C1/C2 reconstruction (tf off)
            lev = [ln for ln in f["launches"] if ln.kernel == "lev/input_encoder_q20"][0]
            fold = [ln for ln in f["launches"] if ln.kernel == FOLD_ADD][0]
            if prev:
                self.assertEqual(lev.args[4], prev.args[10])   # the latent the last frame's store wrote
            prev = fold

    def test_fused_post_default(self):
        """With the fused post at its default (WG_L_POST_TF unset): the hybrid route's kernels (l_post_tf_p12, and with a
        pre-exposure the _pf pair with the folded division), after the decoder block's store."""
        t = self.trace(frames=2, tf=True)
        self.assertEqual(t.kernels.get("l_post_tf"), "on")
        for f in t.frames:
            n = names(f)
            self.assertIn("w64_lc12/l_post_tf_p12", n)
            self.assertEqual(n[n.index(FOLD_ADD) + 1], "w64_lc12/l_post_tf_p12", n)
            self.assertNotIn("lc12/l_post_recon_p12", n)
            self.assertNotIn("w64/post_filter_c1_s4", n)
            self.assertNotIn("scale_rgb", n)
        t = self.trace(frames=2, tf=True, flags=("--pre-exposure=2.0",))
        for f in t.frames:
            n = names(f)
            self.assertIn("w64_lc12/l_post_tf_p12_pf", n)
            self.assertIn("lev/input_encoder_q20_pf", n)
            self.assertNotIn("color_divide", n)

    def test_unfused_fallbacks(self):
        """Without lc / ll / le the pair and the linears run, still with the skip (the fallback routes)."""
        # (the option lists the fusions that stay: NO_FOLD drops lc, ll and le, NO_LIN drops ll and le, NO_ENCLIN le)
        for env, fold, enc_lin in ((F.NO_FOLD, False, False), (F.NO_LIN, True, False), (F.NO_ENCLIN, True, False),
                                   ({"WG_D3D12_FUSION": "none"}, False, False)):
            with self.subTest(fusion=env):
                t = self.trace(frames=3, rw=96, rh=64, env=env)
                for f in t.frames:
                    self.check_frame(t, f, fold=fold, enc_lin=enc_lin)

    def test_weights_are_models_own(self):
        """Every blob create uploads is Model M's packer's on Model L's values (never the hybrid values), under the
        names the kernels read."""
        t = self.trace(frames=1, rw=96, rh=64)
        digests = fast_digests()
        for blob, digest in digests.items():
            self.assertIsNotNone(t.uploaded(digest), "no single upload holds blob " + blob)
        hybrid = F.blob_digests("d3d12")
        for blob in ("enc0_block", "dec0_block", "enc1", "b0", "dec1"):   # the gamma / weights differ in the hybrid values
            self.assertNotEqual(digests[blob], hybrid[blob], blob)
            self.assertIsNone(t.uploaded(hybrid[blob]), blob)
        f = t.frames[0]
        fold = [ln for ln in f["launches"] if ln.kernel == FOLD_ADD][0]
        self.assertEqual(fold.args[3], t.uploaded(digests["dec0_block"]))
        self.assertEqual(fold.args[11], t.uploaded(digests["dec0.coef"]))
        enc = [ln for ln in f["launches"] if ln.kernel == ENC_LIN][0]
        self.assertEqual(enc.args[3], t.uploaded(digests["enc0_block"]))
        self.assertEqual(enc.args[8:10], [t.uploaded(digests["enc0.Win1"]), t.uploaded(digests["enc0.Win2"])])

    def test_bottleneck_needs_the_bmegahg_build(self):
        """Model M's six-launch bottleneck has no D3D12 port (swin_block_c128_nc): without the build's variants the
        fast route is refused at create, never a frame that cannot launch."""
        got = l_run("model=L,l_mode=fast", frames=1, variants=False)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("ERR create: Model L fast on native D3D12: no standalone swin_block_c128_nc "
                                          "port"), got)

    def test_each_kernel_named(self):
        for k in FAST_NEEDS:
            with self.subTest(port=k):
                got = l_run("model=L,l_mode=fast", ports=without(k), frames=1)
                if k in ("lev/input_encoder_q20",):
                    self.assertTrue(got[0].startswith("ERR create: Model L on native D3D12: not yet ported to D3D12: "), got)
                else:
                    self.assertEqual(got, ["ERR create: Model L on native D3D12: not yet ported to D3D12: " + k])

    def test_hybrid_twins_not_needed(self):
        """The fast route launches no twin: a manifest without them creates and traces the same frames."""
        drop = F.HYB_TWINS + (F.ENC_LIN_HYB, F.DEC_COEF_HYB, F.BNECK_HYB, *F.EX_HYB)
        a = F.same_frames(l_run("model=L,l_mode=fast", frames=2, rw=96, rh=64))
        b = F.same_frames(l_run("model=L,l_mode=fast", ports=without(*drop), frames=2, rw=96, rh=64))
        self.assertEqual(a[0], "create ok model L")
        self.assertEqual(b, a)

    def test_failed_launch_stops_the_frame(self):
        for kernel, before in ((FOLD_ADD, "l_dec_project"), ("lc12/l_post_recon_p12", FOLD_ADD)):
            with self.subTest(kernel=kernel):
                t = FT.Trace(l_run("model=L,l_mode=fast", frames=3, flags=("--fail=%s@1" % kernel,), tf_default=False))
                ok0, bad, ok2 = t.frames
                self.assertEqual(bad["failed"], kernel)
                self.assertEqual(names(bad)[-1], before)
                self.assertIsNone(ok2["error"])
                self.check_frame(t, ok2)

    def test_ldr_and_fallback_post(self):
        """LDR colour and a frame the pack12 kernels do not take run the same decoder block (skip) and the lc12 forms."""
        ports = F.full_ports() + [kc for hdr in F.D3D12.OWN for kc in F.D3D12.OWN[hdr] if kc not in F.full_ports()]
        t = FT.Trace(l_run("model=L,l_mode=fast", ports=ports, frames=2, rw=96, rh=64, hdr=0))
        self.assertEqual(t.create, "create ok model L")
        for f in t.frames:
            n = names(f)
            self.assertIn(FOLD_ADD, n)
            self.assertIn("lc12/l_post_recon_p12_ldr", n)


if __name__ == "__main__":
    unittest.main()
