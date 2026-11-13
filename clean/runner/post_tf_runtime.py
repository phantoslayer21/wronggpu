# SPDX-License-Identifier: MIT
"""Whole-frame check of the runtime's true-fusion path (post_tf*) on the GPU, synthetic model and inputs only.

Two runtimes of the same build per output path, one with the runtime option post_tf=1 (the default: one post_tf launch
per frame) and one with post_tf=0 (the banded post_recon2_r_band_p12 + post_filter_c1_s4(_h)_band sequence), run the
same frames and must give the same bytes:
  - "copy": the interop's output copy (wg_rt_frame_io with out_half_pitch: RGBA16F rows, pi folded in; post_tf_p12 /
    post_tf_dl_p12);
  - "own": the runtime's own output buffer (3 x f32, then scale_rgb when pi != 1; post_tf_f32_p12 / post_tf_dl_f32_p12).
The frames change the render size (dynamic resolution: Quality, Balanced, Performance, native AA, Ultra Performance and
an odd size), reset, and switch the pre-exposure between 1 and != 1; the full history is fed back from frame to frame,
so any history or reduced-history difference shows up in later outputs.

    python clean/runner/post_tf_runtime.py --build <build with the runtime library> --rocm <ROCm> --model <dir>
        [--output 2560x1440] [--frames 14]

<dir> holds a synthetic model in the raw layout (runner/synthetic_model.py write_raw; written there if missing).
Run under the GPU lock.
"""
import argparse
import ctypes
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np  # noqa: E402

from clean.reference import encoder as En  # noqa: E402
from clean.runner.pipeline import Pipeline  # noqa: E402
from clean.runner.synthetic_model import synthetic_tensors, write_raw  # noqa: E402

up, vp = ctypes.c_size_t, ctypes.c_void_p


class WgRtExtIo(ctypes.Structure):   # wg_runtime.h (as tests/test_runtime_external.py)
    _fields_ = [("color", up), ("depth", up), ("motion", up), ("out", up), ("expo_src", up),
                ("exposure_mode", ctypes.c_int), ("raw_inputs", ctypes.c_int), ("color_pitch", ctypes.c_int),
                ("depth_pitch", ctypes.c_int), ("motion_pitch", ctypes.c_int), ("motion_channels", ctypes.c_int),
                ("ms", ctypes.c_float * 2), ("mo", ctypes.c_float * 2), ("out_half_pitch", ctypes.c_int)]


class Ext:
    """A runtime created with wg_rt_create_external from a Pipeline's configuration (the interop's HIP executor)."""

    def __init__(self, ref, out_pitch):
        lib = self.lib = ref.lib
        P = ctypes.POINTER
        lib.wg_rt_create_external.argtypes = [vp, P(vp), ctypes.c_char_p, ctypes.c_int]
        lib.wg_rt_alloc.argtypes = [vp, ctypes.c_size_t, P(up)]
        lib.wg_rt_upload.argtypes = [vp, up, vp, ctypes.c_size_t]
        lib.wg_rt_frame_io.argtypes = [vp, vp, P(WgRtExtIo)]
        self.ref = ref
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(2048)
        if lib.wg_rt_create_external(ctypes.byref(ref._cfg), ctypes.byref(rt), err, len(err)) != 0:
            raise RuntimeError(err.value.decode())
        self.rt = rt
        R = ref.max_rw * ref.max_rh
        self.pitch = out_pitch
        self.bufs = {k: self.alloc(n) for k, n in {"color": R * 16, "depth": R * 4, "motion": R * 8,
                                                     "out": out_pitch * ref.oh, "expo": 16}.items()}

    def alloc(self, n):
        p = up()
        assert self.lib.wg_rt_alloc(self.rt, n, ctypes.byref(p)) == 0
        return p.value

    def upload(self, name, a):
        a = np.ascontiguousarray(a)
        assert self.lib.wg_rt_upload(self.rt, self.bufs[name], a.ctypes.data, a.nbytes) == 0

    def frame(self, x):
        rh, rw = x["color"].shape[:2]
        rgba = np.zeros((rh, rw, 4), np.float32)
        rgba[..., :3] = x["color"]
        self.upload("color", rgba)
        self.upload("depth", np.asarray(x["depth"], np.float32))
        self.upload("motion", np.asarray(x["motion"], np.float32))
        fr = self.ref._upload(x)
        b = self.bufs
        io = WgRtExtIo(color=b["color"], depth=b["depth"], motion=b["motion"], out=b["out"], expo_src=b["expo"],
                       exposure_mode=2, out_half_pitch=self.pitch)
        if self.lib.wg_rt_frame_io(self.rt, ctypes.addressof(fr), ctypes.byref(io)) != 0:
            raise RuntimeError(self.lib.wg_rt_error(self.rt).decode())
        a = np.empty(self.pitch * self.ref.oh, np.uint8)
        assert self.lib.wg_rt_download(self.rt, b["out"], a.ctypes.data, a.nbytes) == 0
        return a.reshape(self.ref.oh, self.pitch)[:, :8 * self.ref.ow]

    def kernels(self):
        return dict(kv.split("=", 1) for kv in self.lib.wg_rt_kernels(self.rt).decode().split())

    def __del__(self):
        if getattr(self, "rt", None):
            self.lib.wg_rt_destroy(self.rt)


class Own:
    """A Pipeline on its own buffers (3 x f32 output)."""

    def __init__(self, ref):
        self.ref = ref

    def frame(self, x):
        return self.ref.frame(x)["out"].view(np.uint32)

    def kernels(self):
        return dict(kv.split("=", 1) for kv in self.ref.lib.wg_rt_kernels(self.ref.rt).decode().split())


def frame_inputs(rng, rw, rh, reset, pi):
    x = En.random_inputs(rng, rw=rw, rh=rh, fw=-(-rw // 32) * 32, fh=-(-rh // 32) * 32)
    return {"color": x["color"], "depth": x["depth"], "motion": x["motion"], "mv_scale": x["mv_scale"],
            "jitter": x["jitter"], "exposure": float(rng.uniform(0.5, 2.0)), "pre_exposure": pi, "reset": reset}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default=os.environ.get("ROCM_PATH", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="2560x1440")
    ap.add_argument("--frames", type=int, default=14)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--paths", default="copy,own")
    a = ap.parse_args()
    if not os.path.isdir(os.path.join(a.model, "raw")):
        write_raw(a.model, synthetic_tensors(1))
    ow, oh = (int(v) for v in a.output.lower().split("x"))
    max_rw, max_rh = ow, oh   # up to native AA; smaller frames: dynamic resolution
    pitch = (ow * 8 + 255) // 256 * 256
    flags = {"depth_inverted": False, "auto_exposure": True}
    sizes = [(int(ow / 1.5 + 0.5), int(oh / 1.5 + 0.5)), (int(ow / 1.5 + 0.5), int(oh / 1.5 + 0.5)),
             (int(ow / 1.7 + 0.5), int(oh / 1.7 + 0.5)), (int(ow / 2 + 0.5), int(oh / 2 + 0.5)), (ow, oh),
             (int(ow / 3 + 0.5), int(oh / 3 + 0.5)), (int(ow / 1.5 + 0.5) - 37, int(oh / 1.5 + 0.5) - 21)]
    ok = True
    for path in a.paths.split(","):
        runs = {}
        for name, opts in (("tf", "post_tf=1"), ("band", "post_tf=0")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref = Pipeline(a.build, a.rocm, a.model, (max_rw, max_rh), (ow, oh), flags, backend="hip",
                               check_packing=False, model_format="raw", kernels={"options": opts})
            runs[name] = Ext(ref, pitch) if path == "copy" else Own(ref)
        for t in range(a.frames):
            rw, rh = sizes[t % len(sizes)]
            pi = 1.0 if t % 3 else 1.375
            outs, kern = {}, {}
            for name, r in runs.items():
                rng = np.random.default_rng([a.seed, t])
                x = frame_inputs(rng, rw, rh, reset=(t == 0 or t == 8), pi=pi)
                outs[name] = r.frame(x)
                k = r.kernels()
                kern[name] = (k["post_last"], k["filter_last"])
            same = np.array_equal(outs["tf"], outs["band"])
            ok &= same and kern["tf"][0].startswith("post_tf") and kern["band"][0] == "post_recon2_r_band_p12"
            print(f"{path} frame {t}: render {rw}x{rh} pi {pi}: tf {kern['tf']} band {kern['band']} "
                  f"{'identical' if same else 'DIFFERENT (%d values)' % int((outs['tf'] != outs['band']).sum())}",
                  flush=True)
        print(runs["tf"].kernels())
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
