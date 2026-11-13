# SPDX-License-Identifier: MIT
"""Per-frame runtime (clean/runtime, spec 10 core) through clean/runner/pipeline.py, in the host lane emulator.
Synthetic model and frames only.

- Packing: the C++ packers give the same bytes as Runner.pack_weights (both WMMA layouts).
- Wiring: on a frame sequence (first frame, normal frames, a game reset, a render-size change), every stage output
  of the runtime equals the per-stage Runner output (same kernels) computed from the runtime's own stage inputs, with
  window offsets, frame counter, previous jitter and reset taken from spec 10 / spec 02 independently.
- State: double-buffered latent, motion history and full history are handed from frame to frame (spec 10 section 4);
  exposure pair (section 3).
- Failure: missing or damaged model data, no HIP runtime, render size above the maximum (spec 10 section 8).

Run: python -m unittest discover -s clean/tests -t .
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import warnings

import numpy as np

from clean.reference import encoder as En
from clean.reference import exposure as Ex
from clean.reference.swin import window_offset
from clean.runner import build
from clean.runner.pipeline import SWIN_ORDER, Pipeline, logical_tensors
from clean.reference import post as Po  # noqa: E402
from clean.runner.synthetic_model import read_raw, synthetic_tensors, write_raw
from clean.tests.test_kernels_emu import EmuTest, build_dir

_RT = None


def runtime_dir():
    """Emulator library and runtime library in one directory."""
    global _RT
    if _RT is None:
        d = build_dir()
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        r = subprocess.run(build.runtime_command(cxx, os.path.join(d, build.runtime_library_name())),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("runtime build failed:\n" + r.stderr)
        _RT = d
    return _RT


def model_dir(seed=1):
    d = tempfile.mkdtemp(prefix="wg_model_")
    write_raw(d, synthetic_tensors(seed))
    return d


def make_pipeline(mdir, max_size=(24, 20), out_size=(16, 12), flags=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return Pipeline(runtime_dir(), "", mdir, max_size, out_size,
                        flags or {"depth_inverted": False, "auto_exposure": True}, backend="emu", **kw)


def frame_inputs(rng, rw, rh, reset=False, exposure=1.0):
    x = En.random_inputs(rng, rw=rw, rh=rh, fw=-(-rw // 32) * 32, fh=-(-rh // 32) * 32)
    return {"color": x["color"], "depth": x["depth"], "motion": x["motion"], "mv_scale": x["mv_scale"],
            "jitter": x["jitter"], "exposure": exposure, "pre_exposure": 1.25, "reset": reset}


class RuntimeTest(EmuTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mdir = model_dir()
        cls.tensors = logical_tensors(read_raw(cls.mdir))

    def stage_run(self, stage, blob, x, offset=None, skip=None):
        return self.runner.run(stage, blob, x, offset, skip)

    def check_frame(self, p, fr, out, prev, expect):
        """Every stage of this frame vs the Runner on the runtime's own inputs; expect: n, parity, reset, jp."""
        info = out["state"]["info"]
        self.assertEqual((info["n"], info["parity"], info["reset"]), (expect["n"], expect["f"], expect["reset"]))
        np.testing.assert_allclose(info["jitter_prev"], expect["jp"], rtol=0, atol=0)
        rh, rw = fr["color"].shape[:2]
        fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
        # spec 10 draft 2: the network reads color / pi (IEEE float32 division); exposure uses the color and pi
        pi = np.float32(fr["pre_exposure"])
        cnet = fr["color"] if pi == 1 else (fr["color"] / pi).astype(np.float32)
        D = (p.dw, p.dh)

        # exposure (auto): spec 07 value; pair [e, e_prev]
        e = Ex.exposure(fr["color"], fr["pre_exposure"], 0.18)
        self.assertEqual(out["exposure"]["e"], e)
        self.assertEqual(out["exposure"]["e_prev"], e if prev is None else prev["exposure"]["e"])

        # state handed over from the previous frame (spec 10 section 4)
        if prev is not None and not expect["size_change"]:
            np.testing.assert_array_equal(out["state"]["latent_prev"], prev["dec0"]["latent"])
            np.testing.assert_array_equal(out["state"]["motion_history"], prev["enc0"]["m_near"])
            np.testing.assert_array_equal(out["state"]["history_prev"], prev["post"]["history"])

        # enc0
        reduced = prev["post"]["reduced"] if prev is not None else np.zeros((p.dh // 2, p.dw // 2, 3), np.float32)
        x0 = {"color": cnet, "depth": fr["depth"], "motion": fr["motion"], "mv_scale": fr["mv_scale"],
              "mv_bias": np.zeros(2, np.float32), "jitter": fr["jitter"], "jitter_prev": np.array(expect["jp"]),
              "exposure": out["exposure"]["e"], "exposure_prev": out["exposure"]["e_prev"],
              "reset": expect["reset"], "frame": expect["n"], "depth_inverted": False, "feature_size": (fw, fh),
              "recon_size": D, "reduced_history": reduced, "latent": out["state"]["latent_prev"],
              "motion_history": out["state"]["motion_history"]}
        if expect["reset"]:   # not read on reset; any finite content
            x0["latent"] = np.zeros_like(x0["latent"])
            x0["motion_history"] = np.zeros_like(x0["motion_history"])
            x0["reduced_history"] = np.zeros_like(reduced)
        ref = self.stage_run("enc0", p.stage_blob("enc0"), x0)
        for k in ref:
            np.testing.assert_array_equal(out["enc0"][k], ref[k], err_msg="enc0 " + k)

        # Swin stages with the spec 02 offsets for parity f
        x, skips = out["enc0"]["features"], {}
        for s in SWIN_ORDER:
            off = window_offset(s, expect["f"])
            self.assertEqual(info["offsets"][s], off, s)
            if s in ("dec2", "dec1"):
                ref = self.stage_run(s, p.stage_blob(s), x, off, skips["enc2" if s == "dec2" else "enc1"])
            else:
                ref = self.stage_run(s, p.stage_blob(s), x, off)
            for k in ref:
                np.testing.assert_array_equal(out[s][k], ref[k], err_msg=f"{s} {k}")
            if s in ("enc1", "enc2"):
                skips[s] = out[s]["skip"]
                x = out[s]["merged"]
            else:
                x = out[s]["out"]

        # dec0
        xd = {"features": out["dec1"]["out"], "motion": out["enc0"]["m_near"],
              "latent_prev": out["state"]["latent_prev"] if not expect["reset"]
              else np.zeros((rh, rw, 4), np.float32),
              "jitter": fr["jitter"], "jitter_prev": np.array(expect["jp"]), "reset": expect["reset"],
              "feature_size": (fw, fh), "recon_size": D}
        ref = self.stage_run("dec0", p.stage_blob("dec0"), xd)
        for k in ref:
            np.testing.assert_array_equal(out["dec0"][k], ref[k], err_msg="dec0 " + k)

        # post and filter (spec 06) on the runtime's own inputs
        xp = {"color": cnet, "exposure": out["exposure"]["e"], "exposure_prev": out["exposure"]["e_prev"],
              "A": out["dec0"]["A"], "B": out["dec0"]["B"], "C": out["dec0"]["C"],
              "history": out["state"]["history_prev"] if not expect["reset"]
              else np.zeros((p.dh, p.dw, 3), np.float32),
              "m_near": out["enc0"]["m_near"], "m_center": out["enc0"]["m_center"], "m_far": out["enc0"]["m_far"],
              "jitter": fr["jitter"], "reset": expect["reset"], "recon_size": D}
        ref = self.stage_run("post", p.stage_blob("post"), xp)
        for k in ref:
            np.testing.assert_array_equal(out["post"][k], ref[k], err_msg="post " + k)
        if getattr(p, "recon_output", False):
            # experimental recon=output (D = O): the filter is the identity, Dec(display) / e (float32 kernel vs
            # float64 reference: a few ulp)
            ref_out = Po.dec(out["post"]["display"]) / Po.exposure_or_one(out["exposure"]["e"]) * float(pi)
            np.testing.assert_allclose(out["filter"]["out"], ref_out, rtol=2e-5, atol=1e-9)
            np.testing.assert_array_equal(out["out"], out["filter"]["out"])
            return
        ref = self.stage_run("filter", b"", {"display": out["post"]["display"], "exposure": out["exposure"]["e"],
                                            "output_size": (p.ow, p.oh)})
        ref_out = ref["out"] if pi == 1 else (ref["out"] * pi).astype(np.float32)   # filter result times pi
        np.testing.assert_array_equal(out["filter"]["out"], ref_out)
        np.testing.assert_array_equal(out["out"], ref_out)

    def test_frame_sequence(self):
        p = make_pipeline(self.mdir)
        rng = np.random.default_rng(42)
        # (render size, game reset, expected effective reset, size change)
        seq = [((20, 18), False, True, False), ((20, 18), False, False, False), ((20, 18), True, True, False),
               ((20, 18), False, False, False), ((24, 20), False, True, True)]
        prev, prev_jitter, n, f = None, None, 0, 0
        for k, (R, game_reset, eff_reset, size_change) in enumerate(seq):
            with self.subTest(frame=k):
                fr = frame_inputs(rng, *R, reset=game_reset)
                out = p.frame(fr, debug=True)
                jp = tuple(float(v) for v in (fr["jitter"] if eff_reset else prev_jitter))
                expect = {"n": n, "f": f, "reset": eff_reset, "jp": jp, "size_change": size_change}
                self.check_frame(p, fr, out, prev, expect)
                self.assertEqual(out["out"].shape, (12, 16, 3))
                self.assertTrue(np.all(np.isfinite(out["out"])))
            prev, prev_jitter, n, f = out, np.asarray(fr["jitter"], np.float32), (n + 1) % 8, 1 - f

    def test_given_exposure(self):
        p = make_pipeline(self.mdir, flags={"depth_inverted": True, "auto_exposure": False})
        rng = np.random.default_rng(7)
        a = p.frame(frame_inputs(rng, 20, 18, exposure=0.0), debug=True)
        self.assertEqual((a["exposure"]["e"], a["exposure"]["e_prev"]), (0.0, 0.0))
        b = p.frame(frame_inputs(rng, 20, 18, exposure=2.5), debug=True)
        self.assertEqual((b["exposure"]["e"], b["exposure"]["e_prev"]), (2.5, 0.0))

    def test_pre_exposure_invalid_means_one(self):
        """pi <= 0 or not finite counts as 1 (spec 10 draft 2)."""
        rng = np.random.default_rng(13)
        frs = [frame_inputs(rng, 20, 18) for _ in range(2)]
        outs = []
        for pi in (1.0, 0.0, -2.0, float("nan"), float("inf")):
            p = make_pipeline(self.mdir, flags={"depth_inverted": False, "auto_exposure": False})
            outs.append([p.frame(dict(f, pre_exposure=pi, exposure=1.7))["out"] for f in frs])
        for o in outs[1:]:
            for a, b in zip(outs[0], o):
                np.testing.assert_array_equal(a, b)

    def test_frame_counter_wraps(self):
        p = make_pipeline(self.mdir)
        fr = frame_inputs(np.random.default_rng(3), 20, 18)
        ns = []
        for _ in range(9):
            p._submit(dict(fr, reset=True))        # reset does not change n (spec 10 section 3)
            ns.append(p.info()["n"])
        self.assertEqual(ns, [0, 1, 2, 3, 4, 5, 6, 7, 0])


    def test_default_kernels_whole_frame(self):
        """At a size where post_recon2_r applies (D about 2.25 R, as in game), the default kernels (banded
        post_recon2_r_band + post_filter_c1_s4_band, 4 bands) give exactly the per-stage Runner results (post_recon,
        post_filter)."""
        p = make_pipeline(self.mdir, max_size=(20, 18), out_size=(30, 27))
        k = p.kernels()
        self.assertEqual((k["post"], k["filter"], k["post_bands"], k["exposure"]),
                         ("post_recon2_r_band", "post_filter_c1_s4_band", "4", "base"))
        self.assertEqual((k["swin"], k["enc0"], k["dec0"]), ("swin.co", "encoder.co", "decoder.co"))
        self.assertEqual(k["swin_fused"], "dec1")   # fused expand in dec1 by default (bit-identical)
        rng = np.random.default_rng(11)
        prev, prev_jitter = None, None
        for i, reset in enumerate((False, False, True)):
            with self.subTest(frame=i):
                fr = frame_inputs(rng, 20, 18, reset=reset)
                fr["pre_exposure"] = 1.0 if i == 1 else 1.25
                out = p.frame(fr, debug=True)
                self.assertEqual(p.kernels()["post_last"], "post_recon2_r_band")
                eff_reset = reset or i == 0
                jp = tuple(float(v) for v in (fr["jitter"] if eff_reset else prev_jitter))
                self.check_frame(p, fr, out, prev, {"n": i, "f": i % 2, "reset": eff_reset, "jp": jp,
                                                    "size_change": False})
            prev, prev_jitter = out, np.asarray(fr["jitter"], np.float32)

    def test_kernel_choice(self):
        """Kernels chosen by name (config or environment) give the same frames as the defaults; post falls back to
        post_recon where the chosen kernel's footprint does not fit; bad names are reported at create."""
        rng = np.random.default_rng(12)
        frs = [frame_inputs(rng, 20, 18) for _ in range(2)]
        size = dict(max_size=(20, 18), out_size=(30, 27))
        ref = make_pipeline(self.mdir, **size)
        want = [ref.frame(f)["out"] for f in frs]
        alt = make_pipeline(self.mdir, kernels={"post": "post_recon", "filter": "post_filter", "exposure": "base",
                                                "fuse": "none"},
                            **size)
        k = alt.kernels()
        self.assertEqual((k["post"], k["filter"], k["exposure"]), ("post_recon", "post_filter", "base"))
        for f, w in zip(frs, want):
            np.testing.assert_array_equal(alt.frame(f)["out"], w)
        # pre-pass + reconstruction (post_prep, post_recon5): bit-identical too
        for k in ("post_recon5", "post_recon6"):
            r5 = make_pipeline(self.mdir, kernels={"post": k}, **size)
            for f, w in zip(frs, want):
                np.testing.assert_array_equal(r5.frame(f)["out"], w, err_msg=k)
            self.assertEqual(r5.kernels()["post_last"], k)
        os.environ["WG_FILTER_KERNEL"] = "post_filter_s16"
        try:
            env = make_pipeline(self.mdir, **size)
        finally:
            del os.environ["WG_FILTER_KERNEL"]
        self.assertEqual(env.kernels()["filter"], "post_filter_s16")
        for f, w in zip(frs, want):
            np.testing.assert_array_equal(env.frame(f)["out"], w)
        # D smaller than about 1.43 R: post_recon2_r cannot stage the footprint, the runtime uses post_recon (the
        # banded default: post_recon + the unbanded filter)
        for kern in (None, {"post": "post_recon2_r"}):
            small = make_pipeline(self.mdir, kernels=kern)
            ref_small = make_pipeline(self.mdir, kernels={"post": "post_recon", "filter": "post_filter"})
            np.testing.assert_array_equal(small.frame(frs[0])["out"], ref_small.frame(frs[0])["out"])
            self.assertEqual(small.kernels()["post_last"], "post_recon")
        self.assertEqual((small.kernels()["post"], make_pipeline(self.mdir).kernels()["post"]),
                         ("post_recon2_r", "post_recon2_r_band"))
        # runtime options: parsed; graphs and stream priorities are HIP-only (reported off in the emulator)
        opt = make_pipeline(self.mdir, kernels={"options": "graph=1;stream_priority=high"}, **size)
        self.assertEqual((opt.kernels()["graph"], opt.kernels()["stream_priority"]), ("0", "default"))
        np.testing.assert_array_equal(opt.frame(frs[0])["out"], want[0])
        for bad, msg in (({"filter": "post_filter_x"}, "filter kernel"), ({"post": "post_bogus"}, "neither"),
                         ({"dec0": "v1"}, "version-1"), ({"dec0": "pack12"}, "together"),
                         ({"enc0": "colorh"}, "input format")):
            with self.subTest(kernels=bad):
                with self.assertRaisesRegex((RuntimeError, ValueError), msg):
                    make_pipeline(self.mdir, kernels=bad, check_packing=False)


class RuntimeFusedPostTest(EmuTest):
    def test_fused_post_whole_frame(self):
        """post_fused3_h6 (one launch for reconstruction + filter, 6-byte history) gives exactly the frames of the
        default post_recon2_r + post_filter_c1_s4 path (output, history values, reduced history), also with pre-exposure
        != 1; post_fused3_h6_o16 gives the same output rounded to FP16; the 4-byte history variant runs; RGBA16F output
        is refused with pre-exposure != 1."""
        mdir = model_dir(14)
        size = dict(max_size=(20, 18), out_size=(30, 27))
        rng = np.random.default_rng(8)
        frs = [frame_inputs(rng, 20, 18, reset=(i == 2)) for i in range(4)]
        ref = make_pipeline(mdir, **size)
        want = [ref.frame(f, debug=True) for f in frs]
        h6 = make_pipeline(mdir, kernels={"post": "post_fused3_h6"}, **size)
        k = h6.kernels()
        self.assertEqual((k["post"], k["filter"], k["history_bytes"], k["out"]), ("post_fused3_h6", "fused", "6", "f32"))
        for i, (f, w) in enumerate(zip(frs, want)):
            got = h6.frame(f, debug=True)
            np.testing.assert_array_equal(got["out"], w["out"], err_msg=f"frame {i}")
            np.testing.assert_array_equal(got["post"]["history"], w["post"]["history"], err_msg=f"frame {i}")
            np.testing.assert_array_equal(got["post"]["reduced"], w["post"]["reduced"], err_msg=f"frame {i}")
        frs1 = [dict(f, pre_exposure=1.0) for f in frs]
        ref1 = make_pipeline(mdir, **size)
        want1 = [ref1.frame(f)["out"] for f in frs1]
        o16 = make_pipeline(mdir, kernels={"post": "post_fused3_h6_o16"}, **size)
        self.assertEqual(o16.kernels()["out"], "rgba16f")
        for f, w in zip(frs1, want1):
            np.testing.assert_array_equal(o16.frame(f)["out"], w.astype(np.float16).astype(np.float32))
        # two-kernel path with fewer bytes (clean/post 6834679): 6-byte history (exact), RGBA16F filter output
        r6 = make_pipeline(mdir, kernels={"post": "post_recon2_r_h6"}, **size)
        self.assertEqual((r6.kernels()["history_bytes"], r6.kernels()["post_wave"]), ("6", "32"))  # emulator: wave32
        for i, (f, w) in enumerate(zip(frs, want)):
            got = r6.frame(f, debug=True)
            np.testing.assert_array_equal(got["out"], w["out"], err_msg=f"h6 frame {i}")
            np.testing.assert_array_equal(got["post"]["history"], w["post"]["history"], err_msg=f"h6 frame {i}")
        f16 = make_pipeline(mdir, kernels={"filter": "post_filter_c1_s4_o16"}, **size)
        self.assertEqual(f16.kernels()["out"], "rgba16f")
        for f, w in zip(frs1, want1):
            np.testing.assert_array_equal(f16.frame(f)["out"], w.astype(np.float16).astype(np.float32))
        h4 = make_pipeline(mdir, kernels={"post": "post_fused3_h4_o16"}, **size)
        self.assertEqual(h4.kernels()["history_bytes"], "4")
        h4.frame(frs1[0])   # runs (an approximation: its values are not checked here)
        with self.assertRaisesRegex(RuntimeError, "pre-exposure"):
            make_pipeline(mdir, kernels={"post": "post_fused3_h6_o16"}, **size).frame(frs[0])


class RuntimeBandPostTest(EmuTest):
    def test_banded_post_whole_frame(self):
        """The banded two-kernel path (post_recon2_r_band + post_filter_c1_s4_band / _s2_band over n bands, also with
        the planar 6-byte history post_recon2_r_hp_band) gives exactly the frames of the default path (output and
        history values), with pre-exposure != 1 as well."""
        mdir = model_dir(16)
        size = dict(max_size=(20, 18), out_size=(30, 27))
        rng = np.random.default_rng(9)
        frs = [frame_inputs(rng, 20, 18, reset=(i == 2)) for i in range(4)]
        ref = make_pipeline(mdir, kernels={"post": "post_recon2_r"}, **size)   # unbanded
        want = [ref.frame(f, debug=True) for f in frs]
        for kern in (None, {"post": "post_recon2_r_band", "options": "post_bands=3"},
                     {"post": "post_recon2_r_band", "filter": "post_filter_c1_s2_band", "options": "post_bands=2"},
                     {"post": "post_recon2_r_hp_band", "options": "post_bands=3"},
                     {"post": "post_recon2_r_hp"}):
            p = make_pipeline(mdir, kernels=kern, **size)
            for i, (f, w) in enumerate(zip(frs, want)):
                got = p.frame(f, debug=True)
                np.testing.assert_array_equal(got["out"], w["out"], err_msg=f"{kern} frame {i}")
                np.testing.assert_array_equal(got["post"]["history"], w["post"]["history"], err_msg=f"{kern} frame {i}")
        k = make_pipeline(mdir, kernels={"post": "post_recon2_r_band"}, **size).kernels()
        self.assertEqual((k["filter"], k["post_bands"]), ("post_filter_c1_s4_band", "4"))
        with self.assertRaisesRegex(RuntimeError, "banded"):
            make_pipeline(mdir, kernels={"post": "post_recon2_r", "filter": "post_filter_c1_s4_band"},
                          check_packing=False, **size)
        # an unbanded filter or recon=output turn the banded default off
        k = make_pipeline(mdir, kernels={"filter": "post_filter_c1"}, check_packing=False, **size).kernels()
        self.assertEqual((k["post"], k["filter"], k["post_bands"]), ("post_recon2_r", "post_filter_c1", "0"))


class RuntimeDec1Dec0Test(EmuTest):
    def test_fused_dec1_dec0_whole_frame(self):
        """WG_DEC1_DEC0=fused (swin_block_c64_add_dec0: dec1's block with dec0 as its epilogue) gives the same dec0
        outputs and frames as the separate kernels; dec1's expand then is not fused (swin_fused drops dec1)."""
        mdir = model_dir(17)
        rng = np.random.default_rng(10)
        frs = [frame_inputs(rng, 20, 18, reset=(i == 2)) for i in range(3)]
        ref = make_pipeline(mdir)
        fz = make_pipeline(mdir, kernels={"dec1_dec0": "fused"})
        k = fz.kernels()
        self.assertEqual((k["dec1_dec0"], k["swin_fused"]), ("fused", "none"))
        for i, f in enumerate(frs):
            a, b = ref.frame(f, debug=True), fz.frame(f, debug=True)
            for key in ("A", "B", "C", "latent"):
                np.testing.assert_array_equal(b["dec0"][key], a["dec0"][key], err_msg=f"frame {i} dec0 {key}")
            np.testing.assert_array_equal(b["out"], a["out"], err_msg=f"frame {i}")


class RuntimePack12Test(EmuTest):
    def test_pack12_with_p12_post(self):
        """dec0 pack12 (12-byte coefficients) with the _p12 post kernels (the default when the pack12 decoder is
        built: banded post_recon2_r_band_p12; also by the post kernel's name or WG_POST_PACK=p12) gives exactly the
        frames of the base kernels; frames the _p12 kernels do not take (D < about 1.43 R) run decoder.co and
        post_recon."""
        d = tempfile.mkdtemp(prefix="wg_p12_")
        for f in ("wg_emu.so", build.runtime_library_name()):
            shutil.copy(os.path.join(runtime_dir(), f), d)
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu__pack12.so"), {"WG_DEC_PACK12": 1}),
                           capture_output=True, text=True)   # HIP: decoder__pack12.co (tagged "d0p12")
        self.assertEqual(r.returncode, 0, r.stderr)
        mdir = model_dir(18)
        rng = np.random.default_rng(14)
        frs = [frame_inputs(rng, 20, 18, reset=(i == 2)) for i in range(3)]
        ref = make_pipeline(mdir, max_size=(20, 18), out_size=(30, 27))   # no pack12 decoder there: base dec0
        self.assertEqual(ref.kernels()["post_pack"], "none")
        want = [ref.frame(f)["out"] for f in frs]
        mk = lambda kern, out=(30, 27): Pipeline(  # noqa: E731
            d, "", mdir, (20, 18), out, {"depth_inverted": False, "auto_exposure": True}, backend="emu", kernels=kern)
        for kern in ({"post": "post_recon2_r_p12"}, {"post": "post_recon2_r_band_p12", "options": "post_bands=3"},
                     {"pack": "p12"}, None):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p = mk(kern)
            k = p.kernels()
            self.assertEqual((k["dec0"], k["post_pack"]), ("decoder__pack12.co", "p12"))
            for i, (f, w) in enumerate(zip(frs, want)):
                np.testing.assert_array_equal(p.frame(f)["out"], w, err_msg=f"{kern} frame {i}")
                self.assertEqual(p.kernels()["post_last"], k["post"])
        self.assertEqual(k["post"], "post_recon2_r_band_p12")
        # generic fallback: D < about 1.43 R
        ref = make_pipeline(mdir, max_size=(20, 18), out_size=(16, 12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = mk(None, (16, 12))
        for i, f in enumerate(frs):
            np.testing.assert_array_equal(p.frame(f)["out"], ref.frame(f)["out"], err_msg=f"fallback frame {i}")
            self.assertEqual(p.kernels()["post_last"], "post_recon")
        with self.assertRaisesRegex(RuntimeError, "together"):
            make_pipeline(mdir, kernels={"dec0": "pack12"}, check_packing=False)
        for bad, msg in (({"pack": "p12", "post": "post_recon"}, "needs"), ({"pack": "p13"}, "unknown"),
                         ({"pack": "p12", "dec1_dec0": "fused"}, "pack12")):
            with self.assertRaisesRegex(RuntimeError, msg):
                mk(bad)
        # not the default with another post / dec0 / dec1 + dec0 choice, or WG_POST_PACK=none
        for kern in ({"post": "post_recon2_r"}, {"dec0": "base"}, {"dec1_dec0": "fused"}, {"pack": "none"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                k = mk(kern).kernels()
            self.assertEqual((k["post_pack"], k["dec0"]), ("none", "decoder.co"), kern)
        # explicit pack12 without the pack12 decoder: reported at create
        with self.assertRaisesRegex(RuntimeError, "decoder__pack12"):
            make_pipeline(mdir, kernels={"pack": "p12"}, check_packing=False)


class RuntimeSwinV2Test(EmuTest):
    def test_swin_v2_whole_frame(self):
        """The runtime with the second Swin design (variant v2: C=64 blocks of 512 lanes and 2 windows per wave)
        gives exactly the per-stage v2 kernels' outputs on its own stage inputs."""
        from clean.tests.test_swin_v2_emu import runner as v2_runner, v2_build_dir
        d = v2_build_dir()
        shutil.copy(os.path.join(runtime_dir(), build.runtime_library_name()), d)
        mdir = model_dir(9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p = Pipeline(d, "", mdir, (24, 20), (16, 12), {"depth_inverted": False, "auto_exposure": True},
                         backend="emu", variant="v2")
        self.assertEqual(p.kernels()["swin"], "swin__v2.co")
        r = v2_runner("v2")
        rng = np.random.default_rng(21)
        for i in range(2):
            out = p.frame(frame_inputs(rng, 20, 18), debug=True)
            x, skips = out["enc0"]["features"], {}
            for s in SWIN_ORDER:
                off = window_offset(s, i % 2)
                skip = skips.get("enc2" if s == "dec2" else "enc1") if s in ("dec2", "dec1") else None
                ref = r.run(s, p.stage_blob(s), x, off, skip)
                for k in ref:
                    np.testing.assert_array_equal(out[s][k], ref[k], err_msg=f"frame {i} {s} {k}")
                if s in ("enc1", "enc2"):
                    skips[s], x = out[s]["skip"], out[s]["merged"]
                else:
                    x = out[s]["out"]


class RuntimeStageVariantTest(EmuTest):
    def test_per_stage_swin_variants(self):
        """Swin stages run from other build variants (tagged code objects, their own geometry: t128 = 128-lane block
        workgroups, nt1 = merge / expand with one token group per wave, fuse = merge / expand fused into the blocks;
        all bit-identical builds) give the same frames as the base build."""
        from clean.runner import variants as V
        d = tempfile.mkdtemp(prefix="wg_stagevar_")
        for f in ("wg_emu.so", build.runtime_library_name()):
            shutil.copy(os.path.join(runtime_dir(), f), d)
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        names = ["t128", "nt1", "fuse"]
        for n in names:
            r = subprocess.run(build.emu_command(cxx, os.path.join(d, f"wg_emu__{n}.so"), V.all_builds()[n]),
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({n: {"file": V.code_object(n), "defines": V.all_builds().get(n, {}),
                           "geometry": V.variant_geometry(n), "ablation": False} for n in ["base"] + names}, f)
        mdir = model_dir(12)
        rng = np.random.default_rng(4)
        frs = [frame_inputs(rng, 20, 18) for _ in range(2)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mk = lambda k: Pipeline(d, "", mdir, (24, 20), (16, 12), {"depth_inverted": False, "auto_exposure": True},
                                    backend="emu", kernels=k)  # noqa: E731
            ref = mk(None)
            want = [ref.frame(f, debug=True) for f in frs]
            alt = mk({"swin.enc1": "t128", "swin.b3": "t128", "swin.dec2": "nt1", "swin.dec1": "nt1"})
        self.assertEqual(alt.kernels()["swin_stages"], "enc1:t128,b3:t128,dec2:nt1,dec1:nt1")
        for f, w in zip(frs, want):
            got = alt.frame(f, debug=True)
            for s in SWIN_ORDER:
                for k in w[s]:
                    np.testing.assert_array_equal(got[s][k], w[s][k], err_msg=f"{s} {k}")
            np.testing.assert_array_equal(got["out"], w["out"])
        os.environ["WG_SWIN_STAGE_VARIANTS"] = "b0=t128"
        try:
            env = mk(None)
        finally:
            del os.environ["WG_SWIN_STAGE_VARIANTS"]
        self.assertEqual(env.kernels()["swin_stages"], "b0:t128")
        np.testing.assert_array_equal(env.frame(frs[0])["out"], want[0]["out"])
        # fused merge / expand (variant fuse, bit-identical): per stage, and as the global Swin build
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fz = [mk({"swin.enc1": "fuse", "swin.enc2": "fuse", "swin.dec2": "fuse", "swin.dec1": "fuse"}),
                  mk({"swin": "fuse"})]
        for p in fz:
            for f, w in zip(frs, want):
                got = p.frame(f, debug=True)
                for s in SWIN_ORDER:
                    for k in w[s]:
                        np.testing.assert_array_equal(got[s][k], w[s][k], err_msg=f"fuse {s} {k}")
        with self.assertRaisesRegex(RuntimeError, "not built"):
            mk({"swin.b0": "gelu16"})


class RuntimeBneckTest(EmuTest):
    def test_bneck_megakernel_default(self):
        """With the bmegahg build present, b0 ... b5 run as one swin_bneck6 launch by default; the b5 output, the
        decoder stages and the frame equal the six block launches (WG_SWIN_BNECK=none)."""
        from clean.runner import variants as V
        d = tempfile.mkdtemp(prefix="wg_bneck_")
        for f in ("wg_emu.so", build.runtime_library_name()):
            shutil.copy(os.path.join(runtime_dir(), f), d)
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu__bmegahg.so"), V.all_builds()["bmegahg"],
                                             V.source("bmegahg")), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({n: {"file": V.code_object(n), "defines": V.all_builds().get(n, {}),
                           "geometry": V.variant_geometry(n), "ablation": False} for n in ("base", "bmegahg")}, f)
        mdir = model_dir(15)
        rng = np.random.default_rng(5)
        frs = [frame_inputs(rng, 20, 18) for _ in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mk = lambda k: Pipeline(d, "", mdir, (24, 20), (16, 12), {"depth_inverted": False, "auto_exposure": True},
                                    backend="emu", kernels=k)  # noqa: E731
            six, mega = mk({"bneck": "none"}), mk(None)
        self.assertEqual((six.kernels()["swin_bneck"], mega.kernels()["swin_bneck"]), ("none", "bmegahg"))
        for i, f in enumerate(frs):
            a, b = six.frame(f, debug=True), mega.frame(f, debug=True)
            for s in ("b5", "dec2", "dec1"):
                np.testing.assert_array_equal(b[s]["out"], a[s]["out"], err_msg=f"frame {i} {s}")
            np.testing.assert_array_equal(b["out"], a["out"], err_msg=f"frame {i}")


class RuntimeMidTest(EmuTest):
    def test_swin_mid_whole_frame(self):
        """WG_SWIN_MID=1 (swin_mid of the bmegahg build: enc2 + merge, b0 ... b5, expand + dec2 in one launch) gives
        the enc2 / b5 / dec2 / dec1 outputs and frames of the default path (bottleneck megakernel, dec1 with its expand
        fused) and of six block launches; with dec1_dec0=fused the dec0 outputs and frames equal dec1_dec0 alone."""
        from clean.runner import variants as V
        d = tempfile.mkdtemp(prefix="wg_mid_")
        for f in ("wg_emu.so", build.runtime_library_name()):
            shutil.copy(os.path.join(runtime_dir(), f), d)
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu__bmegahg.so"), V.all_builds()["bmegahg"],
                                             V.source("bmegahg")), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(d, V.MANIFEST), "w") as f:
            json.dump({n: {"file": V.code_object(n), "defines": V.all_builds().get(n, {}),
                           "geometry": V.variant_geometry(n), "ablation": False} for n in ("base", "bmegahg")}, f)
        mdir = model_dir(19)
        rng = np.random.default_rng(12)
        frs = [frame_inputs(rng, 20, 18, reset=(i == 2)) for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mk = lambda k: Pipeline(d, "", mdir, (24, 20), (16, 12), {"depth_inverted": False, "auto_exposure": True},
                                    backend="emu", kernels=k)  # noqa: E731
            dflt, six, mid = mk(None), mk({"bneck": "none"}), mk({"mid": "1"})
            d1d0, mid_d1d0 = mk({"dec1_dec0": "fused"}), mk({"mid": "1", "dec1_dec0": "fused"})
            for bad in ({"mid": "1", "bneck": "bmegahg"}, {"mid": "1", "swin.b2": "bmegahg"}):
                with self.assertRaisesRegex((RuntimeError, ValueError), "together"):
                    mk(bad)
        k = mid.kernels()
        self.assertEqual((k["swin_mid"], k["swin_bneck"], k["swin_fused"], dflt.kernels()["swin_mid"]),
                         ("bmegahg", "none", "dec1", "none"))
        self.assertEqual((mid_d1d0.kernels()["swin_mid"], mid_d1d0.kernels()["dec1_dec0"]), ("bmegahg", "fused"))
        for i, f in enumerate(frs):
            a, b, c = dflt.frame(f, debug=True), six.frame(f, debug=True), mid.frame(f, debug=True)
            for s, key in (("enc2", "skip"), ("b5", "out"), ("dec2", "out"), ("dec1", "out")):
                np.testing.assert_array_equal(c[s][key], a[s][key], err_msg=f"frame {i} {s} vs default")
                np.testing.assert_array_equal(c[s][key], b[s][key], err_msg=f"frame {i} {s} vs six launches")
            np.testing.assert_array_equal(c["out"], a["out"], err_msg=f"frame {i}")
            a, c = d1d0.frame(f, debug=True), mid_d1d0.frame(f, debug=True)
            np.testing.assert_array_equal(c["dec2"]["out"], a["dec2"]["out"], err_msg=f"frame {i} dec2 (dec1_dec0)")
            for key in ("A", "B", "C", "latent"):
                np.testing.assert_array_equal(c["dec0"][key], a["dec0"][key], err_msg=f"frame {i} dec0 {key}")
            np.testing.assert_array_equal(c["out"], a["out"], err_msg=f"frame {i} (dec1_dec0)")


class RuntimeModelTest(EmuTest):
    def test_packing_equals_pack_weights(self):
        mdir = model_dir(5)
        for d_row in ("gfx12", "gfx11"):
            make_pipeline(mdir, d_row=d_row)   # check_packing raises on any difference

    def test_damaged_model_is_rejected(self):
        base = model_dir(6)
        cases = {"missing": lambda d: os.remove(os.path.join(d, "raw", "b3", "W1.bin")),
                 "short": lambda d: os.truncate(os.path.join(d, "raw", "dec0", "W.bin"), 100),
                 "nan": lambda d: np.full(64, np.nan, "<f4").tofile(os.path.join(d, "raw", "enc1", "bp.bin")),
                 "T>127": lambda d: np.full(2048, 200, np.uint8).tofile(os.path.join(d, "raw", "enc0", "T.bin"))}
        for name, damage in cases.items():
            with self.subTest(case=name):
                d = tempfile.mkdtemp(prefix="wg_bad_")
                shutil.copytree(base, d, dirs_exist_ok=True)
                damage(d)
                with self.assertRaisesRegex(RuntimeError, "model data"):
                    make_pipeline(d, check_packing=False)

    def test_no_hip_runtime_fails_cleanly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(RuntimeError, "HIP runtime"):
                Pipeline(runtime_dir(), "/nonexistent", model_dir(7), (24, 20), (16, 12), {}, backend="hip",
                         check_packing=False)

    def test_render_size_limits(self):
        p = make_pipeline(model_dir(8))
        rng = np.random.default_rng(0)
        with self.assertRaisesRegex(RuntimeError, "maximum"):
            p.frame(frame_inputs(rng, 28, 18))
        with self.assertRaisesRegex(RuntimeError, "padding"):
            p.frame(frame_inputs(rng, 12, 12))


class ReconOutputTest(RuntimeTest):
    """Experimental recon=output (D = O): every stage still matches the per-stage Runner at that D, the output is the
    identity decode, the result is finite and settles on a static scene; turned on by the config or WG_RECON_SIZE."""

    # the inherited sequence tests run at the spec D in RuntimeTest already
    test_frame_sequence = test_given_exposure = test_pre_exposure_invalid_means_one = None
    test_frame_counter_wraps = None

    def run_sequence(self, p):
        rng = np.random.default_rng(11)
        self.assertEqual((p.dw, p.dh), (p.ow, p.oh))
        prev, prev_jitter, n, f = None, None, 0, 0
        for k, (R, game_reset, eff_reset) in enumerate([((20, 18), False, True), ((20, 18), False, False),
                                                        ((20, 18), False, False)]):
            fr = frame_inputs(rng, *R, reset=game_reset)
            out = p.frame(fr, debug=True)
            jp = tuple(float(v) for v in (fr["jitter"] if eff_reset else prev_jitter))
            self.check_frame(p, fr, out, prev, {"n": n, "f": f, "reset": eff_reset, "jp": jp, "size_change": False})
            self.assertEqual(out["out"].shape, (p.oh, p.ow, 3))
            self.assertTrue(np.all(np.isfinite(out["out"])))
            prev, prev_jitter, n, f = out, np.asarray(fr["jitter"], np.float32), (n + 1) % 8, 1 - f

    def test_config_option(self):
        p = make_pipeline(self.mdir, out_size=(30, 27), flags={"depth_inverted": False, "auto_exposure": True,
                                                               "recon": "output"})
        self.run_sequence(p)

    def test_environment_variable_and_stable(self):
        os.environ["WG_RECON_SIZE"] = "output"
        try:
            p = make_pipeline(self.mdir, out_size=(30, 27))
        finally:
            del os.environ["WG_RECON_SIZE"]
        self.run_sequence(p)

    def test_static_scene_settles_like_spec(self):
        """Fresh context, static scene (same inputs, no motion, no jitter): with the synthetic (random-weight) model the
        first frames change a lot in either mode, then the steps shrink; at D = O they must shrink at least about as
        well as at the spec D (measured: last steps 1.3, 0.29, 0.20 at D = O vs 1.7, 1.1, 0.31 at the spec D)."""
        late = {}
        for mode in ("spec", "output"):
            p = make_pipeline(self.mdir, out_size=(30, 27), check_packing=False,
                              flags={"depth_inverted": False, "auto_exposure": True, "recon": mode})
            fr = frame_inputs(np.random.default_rng(12), 20, 18)
            fr["motion"] = np.zeros_like(fr["motion"])
            fr["jitter"] = np.zeros(2, np.float32)
            outs = [p.frame(fr)["out"] for _ in range(10)]
            self.assertTrue(all(np.all(np.isfinite(o)) for o in outs), mode)
            scale = float(np.mean(np.abs(outs[-1]))) + 1e-6
            rel = [float(np.mean(np.abs(outs[i + 1] - outs[i]))) / scale for i in range(len(outs) - 1)]
            self.assertLess(rel[-1], 0.05 * max(rel[:3]), f"{mode}: {rel}")
            late[mode] = float(np.mean(rel[-3:]))
        self.assertLess(late["output"], 2.0 * late["spec"] + 0.05, late)

    def test_direct_fp16_output(self):
        """D = O with pi = 1 and the runtime's own output buffer: post_recon2_r_out writes the output (RGBA16F) in one
        launch; equal to the post_recon2 + post_copy path (3 x f32, used with pi != 1 or a 3 x f32 caller buffer)
        rounded to FP16, and the full history is the same."""
        rng = np.random.default_rng(13)
        frs = [dict(frame_inputs(rng, 20, 18), pre_exposure=1.0) for _ in range(3)]
        p16 = make_pipeline(self.mdir, out_size=(30, 27), flags={"depth_inverted": False, "auto_exposure": True,
                                                                 "recon": "output"})
        pf = make_pipeline(self.mdir, out_size=(30, 27), kernels={"post": "post_recon2"},   # bit-identical to _r
                           flags={"depth_inverted": False, "auto_exposure": True, "recon": "output"})
        for i, f in enumerate(frs):
            a = p16.frame(f, debug=True)
            self.assertEqual((p16.kernels()["post_last"], p16.kernels()["out"]), ("post_recon2_r_out", "rgba16f"))
            b = pf.frame(f, debug=True)   # not post_recon2_r: the post_copy path
            self.assertEqual(pf.kernels()["out"], "f32")
            np.testing.assert_array_equal(a["post"]["history"], b["post"]["history"], err_msg=f"frame {i}")
            np.testing.assert_allclose(a["out"], b["out"].astype(np.float16).astype(np.float32), rtol=1e-3, atol=1e-6,
                                       err_msg=f"frame {i}")

    def test_default_is_spec(self):
        p = make_pipeline(self.mdir, out_size=(30, 27))
        self.assertEqual((p.dw, p.dh), ((3 * 30 + 1) // 2, (3 * 27 + 1) // 2))


if __name__ == "__main__":
    unittest.main()
