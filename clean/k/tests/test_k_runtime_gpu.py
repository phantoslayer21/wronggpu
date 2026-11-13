# SPDX-License-Identifier: MIT
"""Model K in the per-frame runtime library (wronggpu_runtime.dll, runtime option model=K) on the GPU, through the
external-I/O entry points the D3D12 interop's HIP executor uses (wg_rt_create_external, wg_rt_alloc, wg_rt_frame_io),
against clean/k/runner/k_frame.py KFrame on the same inputs: the same kernels, so every frame's output must be the
same bits.

- A synthetic model K (random FP16 parameters, clean/k/reference/network.py random_network_params) written as the
  converter's model.wgcm (clean/converter/OUTPUT-FORMAT.md) with a matching map, loaded and checked by the runtime.
- Sequences with a reset, steady frames, a render-size change and a game reset; HDR and LDR, depth inverted or not,
  the three exposure sources (value, the game's texel on the GPU, auto-exposure on the GPU), pre-exposure != 1, both
  output layouts of the interop (dense RGB float32; RGBA16F rows with alpha 1) and the input-copy path (raw RGBA16F
  colour, R32F depth, FP16 motion converted on the GPU).
- 1440p Quality (2560 x 1440 from 1706 x 960): the runtime's timing marks (as timing_log=1 in game) per part.

GPU only: WG_K_GPU_BUILD=<full build dir (clean/runner/build.py, with the runtime library)>, ROCM_PATH=<HIP SDK>;
run under the machine's GPU lock. Run: python -m unittest clean.k.tests.test_k_runtime_gpu
"""
import ctypes
import hashlib
import json
import os
import struct
import sys
import unittest

import numpy as np

from clean.k.reference.layout import STAGE_NAMES
from clean.k.reference.network import random_network_params
from clean.k.reference.params import normalize
from clean.k.runner.k_frame import KFrame
from clean.k.runner.k_model import MAP_PATH, map_tensors

P = ctypes.POINTER
vp, up, sz = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t


def gpu_build():
    b = os.environ.get("WG_K_GPU_BUILD")
    if not b:
        raise unittest.SkipTest("GPU only: set WG_K_GPU_BUILD (and ROCM_PATH)")
    from clean.runner.build import runtime_library_name
    if not os.path.exists(os.path.join(b, runtime_library_name())):
        raise unittest.SkipTest("no runtime library in WG_K_GPU_BUILD (build with clean/runner/build.py)")
    return b


# ---- a synthetic converted model K ---------------------------------------------------------------------------------

def write_k_model(out_dir, params, seed=0):
    """model.wgcm (OUTPUT-FORMAT.md, container version 1) of params ({stage: tensors}, plus net/window_offsets) with
    the tensor names, order and shapes of the preset-K map, and a map with the matching stage digests. Returns
    (model.wgcm path, map path)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    groups = []
    for stage, tensors in map_tensors(MAP_PATH, with_dtype=True).items():
        canon = params[stage] if stage == "net" else normalize(stage, params[stage])
        items = []
        for name, shape, dtype in tensors:
            if dtype == "int32":
                items.append((name, 2, shape, np.ascontiguousarray(np.asarray(canon[name], "<i4").reshape(shape))))
            else:
                a = np.asarray(canon[name], "<f4").reshape(shape)  # the same row-major order in both shape forms
                items.append((name, 0, shape, np.ascontiguousarray(a)))
        h = hashlib.sha256(key)
        for _, _, _, a in items:
            h.update(a.tobytes())
        groups.append((stage, h.digest(), items))
    smap = {"name": "synthetic preset-K map (clean/k/tests/test_k_runtime_gpu.py)",
            "stages": [{"stage": s, "digest": d.hex()} for s, d, _ in groups]}
    map_path = os.path.join(out_dir, "map.json")
    mbytes = json.dumps(smap, indent=1).encode()
    with open(map_path, "wb") as f:
        f.write(mbytes)

    def name(s):
        b = s.encode()
        return struct.pack("<H", len(b)) + b

    def index(offsets):
        ix, k = b"", 0
        for stage, digest, items in groups:
            ix += name(stage) + digest + struct.pack("<I", len(items))
            for tname, dt, shape, a in items:
                ix += name(tname) + struct.pack("<BB", dt, len(shape)) + b"".join(struct.pack("<I", n) for n in shape)
                ix += struct.pack("<QQ", offsets[k], a.nbytes)
                k += 1
        return ix

    n = sum(len(items) for _, _, items in groups)
    isize = len(index([0] * n))
    offsets, data, pos = [], b"", 88 + isize
    for _, _, items in groups:
        for _, _, _, a in items:
            pad = (-pos) % 64
            data += b"\0" * pad
            pos += pad
            offsets.append(pos)
            data += a.tobytes()
            pos += a.nbytes
    body = b"WGCMODEL" + struct.pack("<II", 1, len(groups)) + key + hashlib.sha256(mbytes).digest() + \
        struct.pack("<Q", isize) + index(offsets) + data
    path = os.path.join(out_dir, "K", "model.wgcm")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(body + hashlib.sha256(body).digest())
    return path, map_path


# ---- the runtime library -------------------------------------------------------------------------------------------

class WgRtExtIo(ctypes.Structure):
    _fields_ = [("color", up), ("depth", up), ("motion", up), ("out", up), ("expo_src", up),
                ("exposure_mode", ctypes.c_int), ("raw_inputs", ctypes.c_int), ("color_pitch", ctypes.c_int),
                ("depth_pitch", ctypes.c_int), ("motion_pitch", ctypes.c_int), ("motion_channels", ctypes.c_int),
                ("ms", ctypes.c_float * 2), ("mo", ctypes.c_float * 2), ("out_half_pitch", ctypes.c_int)]


def pitch256(nbytes):
    return (nbytes + 255) // 256 * 256


class KRuntimeExt:
    """wg_rt_create_external with model=K and the slot buffers of the interop (FORMATS.md "Per-frame runtime")."""

    def __init__(self, build, model_dir, map_path, out_size, max_render, hdr, depth_inverted, auto_exposure,
                 options=""):
        from clean.runner.build import runtime_library_name
        from clean.runner.pipeline import WgRtConfig, WgRtFrame
        self.WgRtFrame = WgRtFrame
        rocm = os.environ.get("ROCM_PATH", "")
        if sys.platform == "win32" and rocm and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.join(rocm, "bin"))
        lib = self.lib = ctypes.CDLL(os.path.abspath(os.path.join(build, runtime_library_name())))
        lib.wg_rt_create_external.argtypes = [P(WgRtConfig), P(vp), ctypes.c_char_p, ctypes.c_int]
        lib.wg_rt_destroy.argtypes = [vp]
        lib.wg_rt_error.argtypes = [vp]
        lib.wg_rt_error.restype = ctypes.c_char_p
        lib.wg_rt_alloc.argtypes = [vp, sz, P(up)]
        lib.wg_rt_upload.argtypes = [vp, up, vp, sz]
        lib.wg_rt_download.argtypes = [vp, up, vp, sz]
        lib.wg_rt_frame_io.argtypes = [vp, P(WgRtFrame), P(WgRtExtIo)]
        lib.wg_rt_frame_io_timed.argtypes = [vp, P(WgRtFrame), P(WgRtExtIo), P(ctypes.c_float * 4)]
        lib.wg_rt_timing_labels.argtypes = [vp]
        lib.wg_rt_timing_labels.restype = ctypes.c_char_p
        lib.wg_rt_kernels.argtypes = [vp]
        lib.wg_rt_kernels.restype = ctypes.c_char_p
        lib.wg_rt_buffer.argtypes = [vp, ctypes.c_char_p, P(up), P(sz)]
        lib.wg_rt_sync.argtypes = [vp]
        self.W, self.H = out_size
        self.max_rw, self.max_rh = max_render
        cfg = WgRtConfig()
        opts = f"model=K;model_k_map={map_path}" + (";" + options if options else "")
        self._keep = [os.fsencode(build), os.fsencode(rocm), os.fsencode(model_dir), opts.encode()]
        cfg.build_dir, cfg.rocm_dir, cfg.model_dir, cfg.runtime_options = self._keep
        cfg.backend = 0
        cfg.max_rw, cfg.max_rh, cfg.ow, cfg.oh = self.max_rw, self.max_rh, self.W, self.H
        cfg.depth_inverted, cfg.hdr, cfg.auto_exposure = int(depth_inverted), int(hdr), int(auto_exposure)
        cfg.model_format = 2
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(4096)
        if lib.wg_rt_create_external(ctypes.byref(cfg), ctypes.byref(rt), err, len(err)) != 0:
            raise RuntimeError("runtime create failed: " + err.value.decode(errors="replace"))
        self.rt = rt
        R, O = self.max_rw * self.max_rh, self.W * self.H
        self.half_pitch = pitch256(8 * self.W)
        self.bufs = {k: self.alloc(n) for k, n in {
            "color": max(R * 16, pitch256(8 * self.max_rw) * self.max_rh), "depth": R * 4,
            "motion": max(R * 8, pitch256(8 * self.max_rw) * self.max_rh),
            "out": max(O * 12, self.half_pitch * self.H), "expo": 256}.items()}

    def check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: " + self.lib.wg_rt_error(self.rt).decode(errors="replace"))

    def alloc(self, n):
        p = up()
        self.check(self.lib.wg_rt_alloc(self.rt, n, ctypes.byref(p)), "alloc")
        return p.value

    def upload(self, ptr, a):
        a = np.ascontiguousarray(a)
        self.check(self.lib.wg_rt_upload(self.rt, ptr, a.ctypes.data, a.nbytes), "upload")

    def download(self, ptr, n, dtype):
        a = np.empty(n, dtype)
        self.check(self.lib.wg_rt_download(self.rt, ptr, a.ctypes.data, a.nbytes), "download")
        return a

    def buffer(self, name, dtype):
        p, n = up(), sz()
        self.check(self.lib.wg_rt_buffer(self.rt, name.encode(), ctypes.byref(p), ctypes.byref(n)), name)
        return self.download(p.value, n.value // np.dtype(dtype).itemsize, dtype)

    def time_parts(self, x, mode, half, frames=100, warmup=300):
        """The inputs already in the slot buffers (frame() ran on x before): `frames` steady frames back to back with
        the timing marks, median ms of the four parts."""
        b = self.bufs
        io = WgRtExtIo(color=b["color"], depth=b["depth"], motion=b["motion"], out=b["out"], expo_src=b["expo"],
                       exposure_mode=mode, motion_channels=4, out_half_pitch=self.half_pitch if half else 0)
        fr = self.WgRtFrame()
        fr.rw, fr.rh = x["render_size"]
        for i in range(2):
            fr.mv_scale[i], fr.mv_bias[i], fr.jitter[i] = 1.0, 0.0, float(x["jitter"][i])
        fr.exposure, fr.pre_exposure, fr.reset = float(x["E_cur"]), float(x["pre_exposure"]), 0
        times = []
        for k in range(warmup + frames):
            t = (ctypes.c_float * 4)()
            self.check(self.lib.wg_rt_frame_io_timed(self.rt, ctypes.byref(fr), ctypes.byref(io), ctypes.byref(t)),
                       "frame")
            if k >= warmup:
                times.append(list(t))
        return np.median(np.asarray(times), axis=0)

    def frame(self, x, mode, half=False, raw=None, timed=False):
        """One frame on the slot buffers. x: the KFrame dict (color RGBA [rh, rw, 4], motion, depth, ...); mode:
        exposure source 0 value, 1 the f32 in the expo buffer (x["E_cur"]), 2 auto-exposure; half: RGBA16F rows;
        raw: (color RGBA16F rows, FP16 motion rows, ms, mo, motion channels) for the input-copy path. Returns RGB float32 [H, W, 3]
        (half: the FP16 values as float32, and alpha) and the part times when timed."""
        rw, rh = x["render_size"]
        b = self.bufs
        io = WgRtExtIo(color=b["color"], depth=b["depth"], motion=b["motion"], out=b["out"], expo_src=b["expo"],
                       exposure_mode=mode, motion_channels=4, out_half_pitch=self.half_pitch if half else 0)
        if raw is None:
            self.upload(b["color"], np.asarray(x["color"], np.float32))
            self.upload(b["motion"], np.asarray(x["motion"], np.float32))
        else:
            crow, mrow, ms, mo, io.motion_channels = raw
            self.upload(b["color"], crow)
            self.upload(b["motion"], mrow)
            io.raw_inputs, io.color_pitch, io.depth_pitch, io.motion_pitch = 1, crow.shape[1] * 2, rw * 4, \
                mrow.shape[1] * 2
            io.ms[0], io.ms[1], io.mo[0], io.mo[1] = (float(v) for v in (*ms, *mo))
        self.upload(b["depth"], np.asarray(x["depth"], np.float32))
        if mode == 1:
            self.upload(b["expo"], np.array([x["E_cur"]], np.float32))
        fr = self.WgRtFrame()
        fr.rw, fr.rh = rw, rh
        for i in range(2):
            fr.mv_scale[i] = float(x["mv_scale"][i])
            fr.mv_bias[i] = float(x.get("mv_offset", (0.0, 0.0))[i])
            fr.jitter[i] = float(x["jitter"][i])
        fr.exposure = float(x["E_cur"]) if mode == 0 else 0.5  # mode 1 / 2: the value is not used
        fr.pre_exposure = float(x["pre_exposure"])
        fr.reset = int(x["reset"])  # (True / False, or 2: start over as newly created)
        ms = None
        if timed:
            t = (ctypes.c_float * 4)()
            self.check(self.lib.wg_rt_frame_io_timed(self.rt, ctypes.byref(fr), ctypes.byref(io), ctypes.byref(t)),
                       "frame")
            ms = list(t)
        else:
            self.check(self.lib.wg_rt_frame_io(self.rt, ctypes.byref(fr), ctypes.byref(io)), "frame")
        if half:
            rows = self.download(b["out"], self.half_pitch * self.H // 2, np.float16).reshape(self.H, -1)
            px = rows[:, :4 * self.W].reshape(self.H, self.W, 4)
            out = (px[..., :3].astype(np.float32), px[..., 3].astype(np.float32))
        else:
            out = (self.download(b["out"], self.W * self.H * 3, np.float32).reshape(self.H, self.W, 3), None)
        return out, ms

    def __del__(self):
        if getattr(self, "rt", None):
            self.lib.wg_rt_destroy(self.rt)
            self.rt = None


def inputs(rng, rw, rh, hdr, reset=False, jitter=None):
    color = np.zeros((rh, rw, 4), np.float32)  # alpha 0, as the interop's input pass writes it
    color[..., :3] = rng.random((rh, rw, 3)) * (6.0 if hdr else 1.0)
    color[rng.random((rh, rw)) < 0.01, 0] = -0.25  # negative values (max(RGB, 0) in specs 42, 43)
    return dict(color=color, motion=(rng.standard_normal((rh, rw, 2)) * 1.5).astype(np.float32),
                depth=rng.random((rh, rw)).astype(np.float32), render_size=(rw, rh),
                jitter=tuple(float(np.float32(v)) for v in (jitter if jitter is not None else rng.uniform(-0.5, 0.5, 2))),
                mv_scale=(1.0, 1.0), mv_offset=(0.0, 0.0), pre_exposure=1.0, reset=reset,
                E_cur=1.0, E_aux=None, e1=1.0, e2=1.0)


class TestKRuntimeGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = gpu_build()
        from clean.k.tests.test_k_kernels_emu import device
        cls.dev, cls.fn = device()
        cls.params = random_network_params(11)
        cls.mdir = os.path.join(cls.build, "k_runtime_gpu_test")
        cls.wgcm, cls.map = write_k_model(cls.mdir, cls.params)

    def run_sequence(self, flags, frames, out_size=(200, 120), max_render=(150, 90), mode=0, half=False, raw=False,
                     seed=0, network="fused", options=""):
        """frames: [(rw, rh, reset, E, pre_exposure)]; every frame's runtime output against KFrame's (network: the
        runtime option network= and KFrame's variant)."""
        rng = np.random.default_rng(seed)
        rt = KRuntimeExt(self.build, self.mdir, self.map, out_size, max_render, flags["hdr"], flags["depth_inverted"],
                         mode == 2, ";".join(o for o in ("" if network == "fused" else f"network={network}", options) if o))
        kf = KFrame(self.dev, self.fn, out_size, max_render, self.params, dict(flags, mv_lowres=True), network)
        try:
            for i, (rw, rh, reset, E, pi) in enumerate(frames):
                x = inputs(rng, rw, rh, flags["hdr"], reset)
                x["pre_exposure"] = pi
                x["E_cur"] = float(np.float32(E))
                if i == 2:  # a motion scale and offset (the runtime passes WgRtFrame.mv_scale / mv_bias to K)
                    x["mv_scale"], x["mv_offset"] = (0.75, 1.25), (0.125, -0.25)
                rawio = None
                if raw:  # the game's textures as the interop copies them: RGBA16F colour, FP16 motion, m = ms raw + mo
                    x["color"] = x["color"].astype(np.float16).astype(np.float32)
                    cp = pitch256(8 * rw) // 2
                    crow = np.zeros((rh, cp), np.float16)
                    crow[:, :4 * rw] = x["color"].reshape(rh, 4 * rw)
                    mch = 2 if raw == 2 else 4  # raw=2: RG16F motion, else RGBA16F
                    mraw = rng.uniform(-2, 2, (rh, rw, mch)).astype(np.float16)
                    ms, mo = np.array([4.0, -2.0], np.float32), np.array([0.5, 0.25], np.float32)
                    x["motion"] = (mraw[..., :2].astype(np.float32) * ms + mo).astype(np.float32)
                    mrow = np.zeros((rh, pitch256(2 * mch * rw) // 2), np.float16)
                    mrow[:, :mch * rw] = mraw.reshape(rh, mch * rw)
                    rawio = (crow, mrow, ms, mo, mch)
                (got, alpha), _ = rt.frame(x, mode, half=half, raw=rawio)
                if mode == 2:  # auto-exposure: the runtime's value (spec 07 on the GPU) is KFrame's E_cur
                    x["E_cur"] = float(rt.buffer("exposure.pair", np.float32)[0])
                x["E_aux"] = x["E_cur"]
                want = kf.frame(x)
                ref = want["out"][..., :3]
                info = (i, rw, rh, flags, mode, half, raw)
                if half:
                    np.testing.assert_array_equal(got.astype(np.float16).view(np.uint16),
                                                  ref.astype(np.float16).view(np.uint16), err_msg=f"frame {info}")
                    self.assertTrue(np.all(alpha == 1.0))
                else:
                    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32), err_msg=f"frame {info}")
                self.assertTrue(np.all(np.isfinite(got)), f"frame {info}")
                # histories and network output as well (bit for bit)
                hist = kf.histories()
                for name, key in (("k.Hc", "Hc"), ("k.L", "L"), ("k.Hh", "Hh")):
                    h = rt.buffer(name, np.float16).astype(np.float32).reshape(-1)[:hist[key].size]
                    np.testing.assert_array_equal(h, hist[key].reshape(-1), err_msg=f"{name} frame {info}")
                info_k = dict(kv.split("=", 1) for kv in rt.lib.wg_rt_kernels(rt.rt).decode().split())
                self.assertEqual(info_k["model"], "K")
                self.assertEqual(info_k["network"], network)
                self.assertEqual(want["reset"], i == 0 or reset or frames[i - 1][:2] != (rw, rh))
        finally:
            kf.release()
            del rt

    SEQ = [(150, 90, False, 1.0, 1.0), (150, 90, False, 1.0, 1.0), (150, 90, False, 1.5, 1.0),
           (133, 80, False, 1.5, 1.0), (133, 80, False, 0.75, 1.0), (133, 80, True, 0.75, 1.0),
           (133, 80, False, 0.0, 1.0)]

    def test_hdr_value(self):
        self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ, mode=0)

    def test_hdr_texel_pre_exposure_half(self):
        seq = [(w, h, r, E, 1.0 + 0.25 * i) for i, (w, h, r, E, _) in enumerate(self.SEQ)]
        self.run_sequence({"hdr": True, "depth_inverted": False}, seq, mode=1, half=True, seed=1)

    def test_hdr_auto_exposure(self):
        seq = [(w, h, r, E, 2.0) for (w, h, r, E, _) in self.SEQ]
        self.run_sequence({"hdr": True, "depth_inverted": True}, seq, mode=2, half=True, seed=2)

    def test_ldr(self):
        self.run_sequence({"hdr": False, "depth_inverted": False}, self.SEQ, mode=0, seed=3)

    def test_raw_inputs(self):
        """The interop's input copies read directly (RGBA16F colour, R32F depth, FP16 motion with ms, mo)."""
        self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ[:5], mode=2, half=True, raw=True, seed=4)

    def test_raw_inputs_rg16_motion(self):
        self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ[:4], mode=0, half=True, raw=2, seed=6)

    def test_network_fused16(self):
        """network=fused16 (FP16 accumulation, spec 49 speed approximation, opt-in) against KFrame's "fused16"."""
        self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ[:4], mode=0, network="fused16")

    def test_head_full(self):
        """k_head=full (the 40-channel head map and k_out_pixel) against KFrame's default plan with the compact pixel
        head: the same bits, both networks; and the buffer "k.N" exists only with k_head=full."""
        for network in ("fused", "fused16"):
            self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ[:4], mode=2, half=True, network=network,
                              options="k_head=full")
        from clean.k.reference.layout import base_size
        bw, bh = base_size(200, 120)
        for options, head in (("", "compact"), ("k_head=full", "full")):
            rt = KRuntimeExt(self.build, self.mdir, self.map, (200, 120), (150, 90), True, True, False, options)
            try:
                rt.frame(inputs(np.random.default_rng(8), 150, 90, True, True), 0)
                info_k = dict(kv.split("=", 1) for kv in rt.lib.wg_rt_kernels(rt.rt).decode().split())
                self.assertEqual(info_k["head"], head)
                p, n = up(), sz()
                rc = rt.lib.wg_rt_buffer(rt.rt, b"k.N", ctypes.byref(p), ctypes.byref(n))
                if head == "full":
                    self.assertEqual(rc, 0)
                    self.assertEqual(n.value, bw * bh * 80)
                    self.assertNotEqual(rt.lib.wg_rt_buffer(rt.rt, b"k.Np", ctypes.byref(p), ctypes.byref(n)), 0)
                    self.assertIn("k_head=compact", rt.lib.wg_rt_error(rt.rt).decode())
                else:
                    self.assertNotEqual(rc, 0)
                    self.assertIn("k_head=full", rt.lib.wg_rt_error(rt.rt).decode())
                    self.assertEqual(rt.buffer("k.Np", np.uint16).size, bw * bh * 16)
            finally:
                del rt

    def test_reset2_as_new(self):
        """WgRtFrame.reset = 2 (a parked upscaler taken over by a new context): after earlier frames of other inputs and
        sizes, the frames from the reset = 2 one on are those of a newly created runtime, bit for bit (output,
        histories, exposure, frame info); auto-exposure, so the first-frame exposure is part of it. Both networks."""
        from clean.runner.pipeline import WgRtFrameInfo

        def info(rt):
            i = WgRtFrameInfo()
            rt.lib.wg_rt_info.argtypes = [vp, P(WgRtFrameInfo)]
            rt.check(rt.lib.wg_rt_info(rt.rt, ctypes.byref(i)), "info")
            return (i.index, i.n, i.parity, i.reset, i.rw, i.rh, i.jitter_prev[0], i.jitter_prev[1])

        def state(rt):
            return [rt.buffer(nm, np.uint8).tobytes() for nm in ("k.Hc", "k.L", "k.Hh", "exposure.pair", "k.scalars")]

        for network in ("fused", "fused16"):
            opts = "" if network == "fused" else "network=fused16"
            old_rng, new_rng = np.random.default_rng(20), np.random.default_rng(21)
            before = [inputs(old_rng, w, h, True, False) for (w, h) in ((150, 90), (150, 90), (133, 80))]
            for k, x in enumerate(before):
                x["pre_exposure"] = 1.0 + 0.5 * k
            after = [inputs(new_rng, 150, 90, True, False) for _ in range(3)]
            for k, x in enumerate(after):
                x["pre_exposure"] = 0.75 + 0.25 * k
            used = KRuntimeExt(self.build, self.mdir, self.map, (200, 120), (150, 90), True, True, True, opts)
            fresh = KRuntimeExt(self.build, self.mdir, self.map, (200, 120), (150, 90), True, True, True, opts)
            try:
                for x in before:
                    used.frame(x, 2, half=True)
                for k, x in enumerate(after):
                    (a, _), _ = used.frame(dict(x, reset=2 if k == 0 else False), 2, half=True)
                    (b, _), _ = fresh.frame(x, 2, half=True)
                    what = f"{network} frame {k} after reset = 2"
                    np.testing.assert_array_equal(a.astype(np.float16).view(np.uint16),
                                                  b.astype(np.float16).view(np.uint16), err_msg=what)
                    self.assertEqual(info(used), info(fresh), what)
                    for name, sa, sb in zip(("k.Hc", "k.L", "k.Hh", "exposure.pair", "k.scalars"), state(used),
                                            state(fresh)):
                        self.assertEqual(sa, sb, f"{name}: {what}")
                self.assertEqual(info(used)[0], 2)  # frames since create counted from the reset = 2 frame
            finally:
                del used, fresh

    def test_diagnostics_same_output(self):
        """k_diag=input2+copyin (timing diagnostics: the input stage twice, inputs copied first) changes no bit."""
        self.run_sequence({"hdr": True, "depth_inverted": True}, self.SEQ[:4], mode=2, half=True,
                          options="k_diag=input2+copyin")

    def test_odd_output_size(self):
        seq = [(97, 61, False, 1.0, 1.0), (97, 61, False, 1.25, 1.0), (101, 64, False, 1.25, 1.0)]
        self.run_sequence({"hdr": True, "depth_inverted": True}, seq, out_size=(203, 117), max_render=(101, 64),
                          mode=0, seed=5)

    def test_1440p_quality_timing(self):
        """2560 x 1440 from 1706 x 960 (HDR, auto-exposure, RGBA16F rows: the in-game path); first frames checked
        against KFrame, then the median of the four parts over 100 steady frames (the timing marks of timing_log=1;
        the warm-up lets the GPU clocks settle). WG_K_RT_TIMING_JSON=<file>: also KFrame's part times there.
        WG_K_RT_NETWORK=fused16: the runtime option network=fused16 (KFrame variant "fused16")."""
        W, H, rw, rh = 2560, 1440, 1706, 960
        rng = np.random.default_rng(9)
        network = os.environ.get("WG_K_RT_NETWORK", "fused")
        rt = KRuntimeExt(self.build, self.mdir, self.map, (W, H), (rw, rh), True, True, True,
                         "" if network == "fused" else f"network={network}")
        kf = KFrame(self.dev, self.fn, (W, H), (rw, rh), self.params, {"hdr": True, "mv_lowres": True,
                                                                     "depth_inverted": True}, network)
        try:
            x = inputs(rng, rw, rh, True, True)
            for i in range(2):
                (got, _), _ = rt.frame(x, 2, half=True)
                x["E_cur"] = float(rt.buffer("exposure.pair", np.float32)[0])
                x["E_aux"] = x["E_cur"]
                want = kf.frame(x)
                np.testing.assert_array_equal(got.astype(np.float16).view(np.uint16),
                                              want["out"][..., :3].astype(np.float16).view(np.uint16))
                x["reset"] = False
            med = rt.time_parts(x, 2, True)
            labels = rt.lib.wg_rt_timing_labels(rt.rt).decode().split(",")
            line = f"model K runtime (network={network}) 1440p Quality (2560x1440 from 1706x960), median of 100 frames after 300 warm-up " \
                   "frames: " + ", ".join(f"{n} {t:.3f} ms" for n, t in zip(labels, med)) + \
                   f", total {med.sum():.3f} ms"
            print("\n" + line, flush=True)
            out = os.environ.get("WG_K_RT_TIMING_JSON")
            if out:
                with open(out, "w") as f:
                    json.dump({"labels": labels, "median_ms": [float(v) for v in med], "total_ms": float(med.sum()),
                               "kframe_parts_ms": kf.time_frame(dict(x, reset=False), repeats=40)}, f, indent=1)
            self.assertEqual(labels, ["exposure", "input", "network", "output"])
        finally:
            kf.release()
            del rt

    def test_missing_model_message(self):
        from clean.runner.build import runtime_library_name
        from clean.runner.pipeline import WgRtConfig
        lib = ctypes.CDLL(os.path.abspath(os.path.join(self.build, runtime_library_name())))
        lib.wg_rt_create_external.argtypes = [P(WgRtConfig), P(vp), ctypes.c_char_p, ctypes.c_int]
        cfg = WgRtConfig()
        keep = [os.fsencode(self.build), os.fsencode(os.environ.get("ROCM_PATH", "")),
                os.fsencode(os.path.join(self.mdir, "nothing")), b"model=K"]
        cfg.build_dir, cfg.rocm_dir, cfg.model_dir, cfg.runtime_options = keep
        cfg.max_rw, cfg.max_rh, cfg.ow, cfg.oh = 64, 64, 128, 128
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(4096)
        self.assertNotEqual(lib.wg_rt_create_external(ctypes.byref(cfg), ctypes.byref(rt), err, len(err)), 0)
        self.assertIn("model K: no converted model", err.value.decode())

    def test_unknown_network_message(self):
        with self.assertRaisesRegex(RuntimeError, "model K: unknown network variant fused32"):
            KRuntimeExt(self.build, self.mdir, self.map, (128, 128), (64, 64), True, True, False, "network=fused32")


if __name__ == "__main__":
    unittest.main()
