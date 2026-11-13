# SPDX-License-Identifier: MIT
"""Runtime with external inputs and output (the path the D3D12 interop's HIP executor uses), in the host lane
emulator: frames on external device buffers give exactly the same output as the runtime's own buffers, the exposure
read from an external float equals the same value given directly, and the own-buffer test path is disabled."""
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from clean.runner import build  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402
from clean.tests.test_runtime import frame_inputs, make_pipeline, model_dir, runtime_dir  # noqa: E402
from clean.tests.test_kernels_emu import EmuTest  # noqa: E402

P = ctypes.POINTER
vp, up, sz = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t


class External:
    """A second runtime from the reference pipeline's configuration, created with wg_rt_create_external."""

    def __init__(self, ref):
        lib = self.lib = ref.lib
        lib.wg_rt_create_external.argtypes = [vp, P(vp), ctypes.c_char_p, ctypes.c_int]
        lib.wg_rt_alloc.argtypes = [vp, sz, P(up)]
        lib.wg_rt_upload.argtypes = [vp, up, vp, sz]
        lib.wg_rt_frame_external.argtypes = [vp, vp, up, up, up, up, ctypes.c_int, up]
        self.ow, self.oh, self.max_rw, self.max_rh = ref.ow, ref.oh, ref.max_rw, ref.max_rh
        self.ref = ref
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(2048)
        if lib.wg_rt_create_external(ctypes.byref(ref._cfg), ctypes.byref(rt), err, len(err)) != 0:
            raise RuntimeError(err.value.decode())
        self.rt = rt
        R, O = self.max_rw * self.max_rh, self.ow * self.oh
        self.bufs = {k: self.alloc(n) for k, n in
                     {"color": R * 16, "depth": R * 4, "motion": R * 8, "out": O * 12, "expo": 16}.items()}

    def alloc(self, n):
        p = up()
        assert self.lib.wg_rt_alloc(self.rt, n, ctypes.byref(p)) == 0
        return p.value

    def upload(self, name, a):
        a = np.ascontiguousarray(a)
        assert self.lib.wg_rt_upload(self.rt, self.bufs[name], a.ctypes.data, a.nbytes) == 0

    def frame(self, x, mode=-1, expo=None):
        rh, rw = x["color"].shape[:2]
        rgba = np.zeros((rh, rw, 4), np.float32)
        rgba[..., :3] = x["color"]
        self.upload("color", rgba)
        self.upload("depth", np.asarray(x["depth"], np.float32))
        self.upload("motion", np.asarray(x["motion"], np.float32))
        if expo is not None:
            self.upload("expo", np.array([expo], np.float32))
        fr = self.ref._upload(x)   # builds the WgRtFrame (and uploads into the reference runtime, unused here)
        b = self.bufs
        rc = self.lib.wg_rt_frame_external(self.rt, ctypes.addressof(fr), b["color"], b["depth"], b["motion"],
                                           b["out"], mode, b["expo"])
        assert rc == 0, self.lib.wg_rt_error(self.rt).decode()
        out = np.empty(self.oh * self.ow * 3, np.float32)
        assert self.lib.wg_rt_download(self.rt, b["out"], out.ctypes.data, out.nbytes) == 0
        return out.reshape(self.oh, self.ow, 3)

    def __del__(self):
        if getattr(self, "rt", None):
            self.lib.wg_rt_destroy(self.rt)


class ExternalIoTest(EmuTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mdir = model_dir()

    def test_external_equals_own_buffers(self):
        rng = np.random.default_rng(5)
        ref = make_pipeline(self.mdir, flags={"depth_inverted": False, "auto_exposure": True})
        ext = External(make_pipeline(self.mdir, flags={"depth_inverted": False, "auto_exposure": True},
                                     check_packing=False))
        for t in range(3):
            x = frame_inputs(rng, 21, 17, reset=(t == 0))
            want = ref.frame(x)["out"]
            got = ext.frame(x, mode=2)
            np.testing.assert_array_equal(got, want, err_msg=f"frame {t}")

    def test_exposure_from_external_float(self):
        rng = np.random.default_rng(6)
        flags = {"depth_inverted": False, "auto_exposure": False}
        a = External(make_pipeline(self.mdir, flags=flags, check_packing=False))
        b = External(make_pipeline(self.mdir, flags=flags, check_packing=False))
        for t in range(2):
            x = frame_inputs(rng, 20, 18, reset=(t == 0), exposure=0.75 + 0.5 * t)
            np.testing.assert_array_equal(a.frame(x, mode=0), b.frame(x, mode=1, expo=x["exposure"]))

    def test_own_input_path_disabled(self):
        ext = External(make_pipeline(self.mdir, check_packing=False))
        c = np.zeros((10, 10, 3), np.float32)
        d = np.zeros((10, 10), np.float32)
        m = np.zeros((10, 10, 2), np.float32)
        self.assertNotEqual(ext.lib.wg_rt_upload_inputs(ext.rt, 10, 10, c.ctypes.data, d.ctypes.data,
                                                        m.ctypes.data), 0)
        self.assertIn("external", ext.lib.wg_rt_error(ext.rt).decode())


if __name__ == "__main__":
    unittest.main()


class WgRtExtIo(ctypes.Structure):
    _fields_ = [("color", up), ("depth", up), ("motion", up), ("out", up), ("expo_src", up),
                ("exposure_mode", ctypes.c_int), ("raw_inputs", ctypes.c_int), ("color_pitch", ctypes.c_int),
                ("depth_pitch", ctypes.c_int), ("motion_pitch", ctypes.c_int), ("motion_channels", ctypes.c_int),
                ("ms", ctypes.c_float * 2), ("mo", ctypes.c_float * 2), ("out_half_pitch", ctypes.c_int)]


def pitch(width, bpp):
    return (width * bpp + 255) // 256 * 256


class CopyLayoutTest(EmuTest):
    """The interop's copy paths on the runtime side: raw inputs (the game's textures copied as they are, converted by
    input_convert) give the same frames as the dense inputs; RGBA16F output rows equal the FP16 rounding of the dense
    output (pi folded in), at the spec D and with recon=output."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mdir = model_dir()

    def frame_io(self, ext, x, io):
        ext.lib.wg_rt_frame_io.argtypes = [vp, vp, P(WgRtExtIo)]
        fr = ext.ref._upload(x)
        assert ext.lib.wg_rt_frame_io(ext.rt, ctypes.addressof(fr), ctypes.byref(io)) == 0, \
            ext.lib.wg_rt_error(ext.rt).decode()

    def download(self, ext, ptr, n, dtype):
        a = np.empty(n, dtype)
        assert ext.lib.wg_rt_download(ext.rt, ptr, a.ctypes.data, a.nbytes) == 0
        return a

    def inputs(self, rng, rw, rh, t):
        x = frame_inputs(rng, rw, rh, reset=(t == 0))
        x["pre_exposure"] = 1.0   # raw inputs: exposure reads color / pi with 1 / pi = 1 (identical only for pi = 1)
        x["color"] = x["color"].astype(np.float16).astype(np.float32)
        raw = rng.uniform(-2, 2, (rh, rw, 4)).astype(np.float16)
        ms, mo = np.array([4.0, -2.0], np.float32), np.array([0.5, 0.25], np.float32)
        x["motion"] = (raw[..., :2].astype(np.float32) * ms + mo).astype(np.float32)   # exact (power-of-2 scale)
        x["mv_scale"] = np.ones(2, np.float32)
        return x, raw, ms, mo

    def test_raw_inputs_equal_dense(self):
        flags = {"depth_inverted": False, "auto_exposure": True}
        dense = External(make_pipeline(self.mdir, flags=flags, check_packing=False))
        rawx = External(make_pipeline(self.mdir, flags=flags, check_packing=False))
        rng = np.random.default_rng(21)
        rw, rh = 21, 17
        cp, dp, mp = pitch(rw, 8), pitch(rw, 4), pitch(rw, 8)
        c16, d32, m16 = rawx.alloc(cp * rh), rawx.alloc(dp * rh), rawx.alloc(mp * rh)
        for t in range(3):
            x, raw, ms, mo = self.inputs(rng, rw, rh, t)
            want = dense.frame(x, mode=2)
            crow = np.zeros((rh, cp // 2), np.float16)
            crow[:, :4 * rw].reshape(rh, rw, 4)[..., :3] = x["color"]
            crow[:, :4 * rw].reshape(rh, rw, 4)[..., 3] = 1
            drow = np.zeros((rh, dp // 4), np.float32)
            drow[:, :rw] = x["depth"]
            mrow = np.zeros((rh, mp // 2), np.float16)
            mrow[:, :4 * rw] = raw.reshape(rh, 4 * rw)
            for ptr, a in ((c16, crow), (d32, drow), (m16, mrow)):
                assert rawx.lib.wg_rt_upload(rawx.rt, ptr, a.ctypes.data, a.nbytes) == 0
            io = WgRtExtIo(color=c16, depth=d32, motion=m16, out=rawx.bufs["out"], expo_src=rawx.bufs["expo"],
                           exposure_mode=2, raw_inputs=1, color_pitch=cp, depth_pitch=dp, motion_pitch=mp,
                           motion_channels=4, out_half_pitch=0)
            io.ms[0], io.ms[1], io.mo[0], io.mo[1] = float(ms[0]), float(ms[1]), float(mo[0]), float(mo[1])
            self.frame_io(rawx, x, io)
            got = self.download(rawx, rawx.bufs["out"], rawx.oh * rawx.ow * 3, np.float32).reshape(want.shape)
            np.testing.assert_array_equal(got, want, err_msg=f"frame {t}")

    def check_half_output(self, flags, out_size, kernels=None, dense=False, pi=None, filter_last=None, bdir=None,
                          post_last=None):
        """Byte identity of the output copy: the RGBA16F rows equal the FP16 rounding (to nearest even) of the f32
        output of the default path, which the output pass writes into the game's RGBA16F texture."""
        def mk(kern):
            if bdir is None:
                return External(make_pipeline(self.mdir, out_size=out_size, flags=flags, check_packing=False,
                                              kernels=kern))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return External(Pipeline(bdir, "", self.mdir, (24, 20), out_size, flags, backend="emu",
                                         kernels=kern, check_packing=False))
        # reference: the base kernels in f32 (the output pass's input); b: the kernels under test, RGBA16F rows
        a, b = mk({"pack": "none", "post": "post_recon", "filter": "post_filter"}), mk(kernels)
        ow, oh = b.ow, b.oh
        op = 8 * ow if dense else pitch(ow, 8)
        half = b.alloc(op * oh)
        rng = np.random.default_rng(22)
        for t in range(3):
            x = frame_inputs(rng, 20, 18, reset=(t == 0))   # pre_exposure 1.25: pi folded into the half output
            if pi is not None:
                x["pre_exposure"] = pi
            want = a.frame(x, mode=2)
            b.upload("color", np.concatenate([x["color"], np.zeros(x["color"].shape[:2] + (1,), np.float32)], -1))
            b.upload("depth", np.asarray(x["depth"], np.float32))
            b.upload("motion", np.asarray(x["motion"], np.float32))
            bb = b.bufs
            io = WgRtExtIo(color=bb["color"], depth=bb["depth"], motion=bb["motion"], out=half, expo_src=bb["expo"],
                           exposure_mode=2, out_half_pitch=op)
            self.frame_io(b, x, io)
            rows = self.download(b, half, op * oh // 2, np.float16).reshape(oh, op // 2)[:, :4 * ow].reshape(oh, ow, 4)
            np.testing.assert_array_equal(rows[..., :3].view(np.uint16), want.astype(np.float16).view(np.uint16),
                                          err_msg=f"frame {t}")
            self.assertTrue(np.all(rows[..., 3] == 1))
            k = dict(kv.split("=", 1) for kv in b.lib.wg_rt_kernels(b.rt).decode().split())
            if filter_last:
                self.assertEqual(k["filter_last"], filter_last)
            if post_last:
                self.assertEqual(k["post_last"], post_last)

    def test_half_output_spec_d(self):
        self.check_half_output({"depth_inverted": False, "auto_exposure": True}, (16, 12))

    def test_half_output_default_kernels(self):
        """At an in-game ratio (D about 2.4 R) the defaults (banded post) leave the banded path for the output copy:
        post_recon2_r + post_filter_c1_s4_o16 with dense rows and pi = 1, else post_filter_h (pi folded in, padded
        rows); the unbanded kernels chosen explicitly give the same bytes."""
        f = {"depth_inverted": False, "auto_exposure": True}
        self.check_half_output(f, (32, 27), dense=True, pi=1.0, filter_last="post_filter_c1_s4_o16")
        self.check_half_output(f, (30, 27), pi=1.25, filter_last="post_filter_h")
        self.check_half_output(f, (30, 27), dense=True, pi=1.0, filter_last="post_filter_c1_s4_o16")
        self.check_half_output(f, (32, 27), kernels={"post": "post_recon2_r"}, dense=True, pi=1.0,
                               filter_last="post_filter_c1_s4_o16")
        self.check_half_output(f, (32, 27), kernels={"filter": "post_filter"}, dense=True, pi=1.0,
                               filter_last="post_filter_h")

    def test_half_output_pack12_default(self):
        """With the pack12 decoder built, the default (banded post with pack12) takes post_recon2_r_p12 +
        post_filter_c1_s4_o16 for the output copy: the same bytes."""
        d = tempfile.mkdtemp(prefix="wg_p12x_")
        for f in ("wg_emu.so", build.runtime_library_name()):
            shutil.copy(os.path.join(runtime_dir(), f), d)
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        r = subprocess.run(build.emu_command(cxx, os.path.join(d, "wg_emu__pack12.so"), {"WG_DEC_PACK12": 1}),
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        f = {"depth_inverted": False, "auto_exposure": True}
        self.check_half_output(f, (32, 27), dense=True, pi=1.0, filter_last="post_filter_c1_s4_o16", bdir=d,
                               post_last="post_recon2_r_p12")
        self.check_half_output(f, (30, 27), pi=1.25, filter_last="post_filter_h", bdir=d,
                               post_last="post_recon2_r_p12")

    def test_half_output_recon_output(self):
        self.check_half_output({"depth_inverted": False, "auto_exposure": True, "recon": "output"}, (30, 27))
