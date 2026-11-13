# SPDX-License-Identifier: MIT
"""Whole-frame test interface of spec/90-test-interface.md ("Whole frame (spec 10 core)") over the C++ per-frame
runtime (clean/runtime, built by clean/runner/build.py as libwg_runtime.so / wronggpu_runtime.dll on Windows, which also holds the D3D12 interop).

    from clean.runner.pipeline import Pipeline
    p = Pipeline(build_dir, rocm_dir, model_dir, max_render_size=(1706, 960), output_size=(2560, 1440),
                 flags={"depth_inverted": True, "auto_exposure": False})
    out = p.frame(x)                  # {"out": [oh, ow, 3]}
    out = p.frame(x, debug=True)      # + every stage output of this frame under the spec 90 stage names
    ms = p.time_frame(x)              # median GPU ms of one whole frame

model_dir is the converter's output folder: the runtime loads model.wgcm when it is there (checking the file's
integrity, that it was converted with map_path, and every stage's keyed digest against the map), else the --raw
layout raw/<stage>/<name>.bin (model_format="raw" forces it). It packs every stage in C++; at construction the pipeline
also packs every stage with the functions behind Runner.pack_weights and checks that the bytes are identical
(check_packing=True).

Kernel choice per stage (A/B timing and 10-T1 runs without a new build of the runtime): kernels={"swin": ...,
"swin.<stage>": ..., "enc0": ..., "dec0": ..., "exposure": ..., "post": ..., "filter": ...}, any subset; or the
environment variables WG_SWIN_VARIANT, WG_SWIN_STAGE_VARIANTS ("enc2=smax16,b0=gelu16"), WG_ENC0_VARIANT,
WG_DEC0_VARIANT, WG_EXPOSURE_VARIANT, WG_POST_KERNEL, WG_FILTER_KERNEL (also read by the runtime DLL in game).
"swin.<stage>" runs that Swin stage (enc1 ... dec1) from another build variant of variants.py with its geometry. Values: Swin and enc0 / dec0 / exposure variant names of variants.py and
perf_variants.py (build.py --variants all / --perf-variants builds them), post and filter kernel names of
kernels/post.hip (e.g. post_recon, post_recon2_r, post_recon3, post_recon4_fast; post_filter, post_filter_c1_s4).
Defaults: the fastest exact variants (clean/runtime/wg_runtime.h). p.kernels() reports what is in use, e.g.

    p = Pipeline(..., kernels={"post": "post_recon4_fast", "filter": "post_filter"})
    p.kernels()   # {"swin": "swin.co", "enc0": "encoder.co", ..., "post": "post_recon4_fast", ...}

Extra keyword arguments (not needed by the spec team): backend="emu" runs everything in the host lane emulator;
backend="dx12" the hybrid D3D12 device (the kernels with a D3D12 port in <build_dir>/dx12 on D3D12, WG_DX12_KERNELS;
clean/dx12/README.md); backend="d3d12" selects native D3D12 (backend 3; unsupported routes fail, no HIP fallback);
variant selects a Swin build variant (variants.py; same as kernels["swin"]); d_row the WMMA accumulator layout
(see selftest.py).
"""
import ctypes
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from clean.reference.fp8 import fp8_decode  # noqa: E402
from clean.runner import dec0, enc0, post  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.runner.build import runtime_library_name  # noqa: E402
from clean.runner.packing import pack_stage, resolve_d_row  # noqa: E402
from clean.runner.synthetic_model import read_raw  # noqa: E402

DEFAULT_MAP = os.path.join(_ROOT, "data", "preset-M.v1.map.json")
MODEL_FORMATS = {"auto": 0, "raw": 1, "wgcm": 2}
SWIN_ORDER = ["enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"]
# Model-keyed (#89): the Swin stage order per model (Model M: SWIN_ORDER); a model adds its row next to its stage
# table (clean/reference/swin.py MODEL_STAGES).
MODEL_SWIN_ORDER = {"M": SWIN_ORDER, "L": SWIN_ORDER}  # L: Model M's ten Swin stages, same order (#89)


def swin_order(model="M"):
    """The Swin stages of `model` in frame order (Model M: SWIN_ORDER)."""
    try:
        return MODEL_SWIN_ORDER[model]
    except KeyError:
        known = ", ".join(sorted(MODEL_SWIN_ORDER))
        raise ValueError(f"no Swin stage order for model {model!r}; known: {known}") from None


SWIN_CHANNELS = {"enc1": (64, 96), "enc2": (96, 128), "dec2": (96, None), "dec1": (64, None)}


class WgRtConfig(ctypes.Structure):
    _fields_ = [("build_dir", ctypes.c_char_p), ("rocm_dir", ctypes.c_char_p), ("model_dir", ctypes.c_char_p),
                ("backend", ctypes.c_int), ("max_rw", ctypes.c_int), ("max_rh", ctypes.c_int),
                ("ow", ctypes.c_int), ("oh", ctypes.c_int), ("depth_inverted", ctypes.c_int),
                ("auto_exposure", ctypes.c_int), ("alpha", ctypes.c_float),
                ("swin_threads", ctypes.c_int), ("gemm_threads", ctypes.c_int), ("nt", ctypes.c_int),
                ("wpw64", ctypes.c_int), ("wpw96", ctypes.c_int), ("wpw128", ctypes.c_int),
                ("swin_code_object", ctypes.c_char_p), ("d_row", ctypes.c_int * 16),
                ("emu_library", ctypes.c_char_p), ("map_path", ctypes.c_char_p), ("model_format", ctypes.c_int),
                ("block_threads", ctypes.c_int * 5),
                ("enc0_variant", ctypes.c_char_p), ("dec0_variant", ctypes.c_char_p),
                ("exposure_variant", ctypes.c_char_p), ("post_kernel", ctypes.c_char_p),
                ("filter_kernel", ctypes.c_char_p), ("block_wpg", ctypes.c_int * 5),
                ("swin_force_clamp", ctypes.c_int), ("swin_stage_variants", ctypes.c_char_p),
                ("swin_fuse", ctypes.c_int), ("swin_fuse_stages", ctypes.c_char_p), ("post_wave", ctypes.c_int),
                ("runtime_options", ctypes.c_char_p), ("recon_output", ctypes.c_int),
                ("swin_bneck", ctypes.c_char_p), ("dec1_dec0", ctypes.c_char_p), ("swin_mid", ctypes.c_char_p),
                ("post_pack", ctypes.c_char_p), ("hdr", ctypes.c_int)]

# Per-stage kernel choice (Pipeline(kernels=...)): key -> WgRtConfig field / environment variable. None or missing =
# the environment variable, else the runtime's default (the fastest exact variant; see clean/runtime/wg_runtime.h).
KERNEL_KEYS = {"enc0": ("enc0_variant", "WG_ENC0_VARIANT"), "dec0": ("dec0_variant", "WG_DEC0_VARIANT"),
               "exposure": ("exposure_variant", "WG_EXPOSURE_VARIANT"), "post": ("post_kernel", "WG_POST_KERNEL"),
               "filter": ("filter_kernel", "WG_FILTER_KERNEL"),
               "fuse": ("swin_fuse_stages", "WG_SWIN_FUSE_STAGES"),
               "options": ("runtime_options", "WG_RUNTIME_OPTIONS"),
               "bneck": ("swin_bneck", "WG_SWIN_BNECK"),
               "dec1_dec0": ("dec1_dec0", "WG_DEC1_DEC0"),
               "mid": ("swin_mid", "WG_SWIN_MID"), "pack": ("post_pack", "WG_POST_PACK")}


class WgRtFrame(ctypes.Structure):
    _fields_ = [("rw", ctypes.c_int), ("rh", ctypes.c_int), ("mv_scale", ctypes.c_float * 2),
                ("mv_bias", ctypes.c_float * 2), ("jitter", ctypes.c_float * 2), ("exposure", ctypes.c_float),
                ("pre_exposure", ctypes.c_float), ("reset", ctypes.c_int)]


class WgRtFrameInfo(ctypes.Structure):
    _fields_ = [("index", ctypes.c_int), ("n", ctypes.c_int), ("parity", ctypes.c_int), ("reset", ctypes.c_int),
                ("rw", ctypes.c_int), ("rh", ctypes.c_int), ("jitter_prev", ctypes.c_float * 2),
                ("offsets", (ctypes.c_int * 2) * 10)]


def read_wgcm(path):
    """Tensors of a model.wgcm (clean/converter/OUTPUT-FORMAT.md) as {stage: {name: array}}; no hash checks (the
    runtime does them)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"WGCMODEL":
        raise ValueError(f"{path} is not a WrongGPU model file")
    nstages = int.from_bytes(data[12:16], "little")
    pos = 88
    out = {}

    def u(n):
        nonlocal pos
        v = int.from_bytes(data[pos:pos + n], "little")
        pos += n
        return v

    def s():
        nonlocal pos
        n = u(2)
        pos += n
        return data[pos - n:pos].decode()

    for _ in range(nstages):
        stage = s()
        pos += 32
        out[stage] = {}
        for _ in range(u(4)):
            name, dtype, rank = s(), u(1), u(1)
            shape = [u(4) for _ in range(rank)]
            off, size = u(8), u(8)
            dt = np.uint8 if dtype else "<f4"
            a = np.frombuffer(data[off:off + size], dtype=dt).reshape(shape)
            out[stage][name] = a.astype(np.uint8 if dtype else np.float32)
    return out


def read_model_tensors(model_dir):
    if os.path.isdir(os.path.join(model_dir, "raw")):
        return read_raw(model_dir)
    return read_wgcm(os.path.join(model_dir, "model.wgcm"))


def logical_tensors(raw):
    """Converter raw tensors -> the spec 90 pack_weights tensors of each stage."""
    out = {}
    for stage, t in raw.items():
        if stage == "enc0":
            out[stage] = {"W": t["W"], "b": t["b"], "T": t["T"].astype(np.float32),
                          "sigma1": np.asarray(t["sigma1"][0], np.float32), "sigma2": np.asarray(t["sigma2"][0],
                                                                                                  np.float32)}
        elif stage == "dec0":
            out[stage] = {"W": t["W"].reshape(68, 64), "b": t["b"].reshape(68),
                          "S": np.asarray(t["S"][0], np.float32), "T": np.asarray(t["T"][0], np.float32)}
        elif stage == "post":
            out[stage] = {"s": np.asarray(t["s"][0], np.float32)}
        else:
            out[stage] = dict(t)
    return out


class Pipeline:
    def __init__(self, build_dir: str, rocm_dir: str, model_dir: str, max_render_size: tuple, output_size: tuple,
                 flags: dict, backend: str = "hip", variant: str = None, d_row=None, check_packing: bool = True,
                 alpha: float = 0.18, map_path: str = DEFAULT_MAP, model_format: str = "auto", kernels: dict = None):
        self.build_dir = build_dir
        lib_path = os.path.join(build_dir, runtime_library_name())
        if backend != "d3d12" and sys.platform == "win32" and rocm_dir and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.join(rocm_dir, "bin"))
        self.lib = lib = ctypes.CDLL(os.path.abspath(lib_path))
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        P = ctypes.POINTER
        lib.wg_rt_create.argtypes = [P(WgRtConfig), P(vp), ctypes.c_char_p, ctypes.c_int]
        lib.wg_rt_destroy.argtypes = [vp]
        lib.wg_rt_error.argtypes = [vp]
        lib.wg_rt_error.restype = ctypes.c_char_p
        lib.wg_rt_upload_inputs.argtypes = [vp, ctypes.c_int, ctypes.c_int, vp, vp, vp]
        lib.wg_rt_frame.argtypes = [vp, P(WgRtFrame)]
        lib.wg_rt_frame_timed.argtypes = [vp, P(WgRtFrame), P(ctypes.c_float)]
        lib.wg_rt_sync.argtypes = [vp]
        lib.wg_rt_info.argtypes = [vp, P(WgRtFrameInfo)]
        lib.wg_rt_buffer.argtypes = [vp, ctypes.c_char_p, P(ctypes.c_size_t), P(sz)]
        lib.wg_rt_download.argtypes = [vp, ctypes.c_size_t, vp, sz]
        lib.wg_rt_stage_blob.argtypes = [vp, ctypes.c_char_p, P(vp), P(sz)]
        lib.wg_rt_model_source.argtypes = [vp]
        lib.wg_rt_model_source.restype = ctypes.c_char_p
        lib.wg_rt_kernels.argtypes = [vp]
        lib.wg_rt_kernels.restype = ctypes.c_char_p
        self._has_notes = hasattr(lib, "wg_rt_notes")   # not in a runtime library from before spec 06 draft 4
        if self._has_notes:
            lib.wg_rt_notes.argtypes = [vp]
            lib.wg_rt_notes.restype = ctypes.c_char_p
        if variant is None:
            variant = (kernels or {}).get("swin") or os.environ.get("WG_SWIN_VARIANT") or "base"
        stage_keys = {"swin." + s for s in SWIN_ORDER}
        unknown = set(kernels or {}) - set(KERNEL_KEYS) - {"swin", "post_wave"} - stage_keys
        if unknown:
            raise ValueError(f"unknown kernel keys {sorted(unknown)}; known: swin, swin.<stage>, "
                             f"{', '.join(KERNEL_KEYS)}")
        self.variant = variant
        lib.wg_rt_recon_size.argtypes = [vp, P(ctypes.c_int), P(ctypes.c_int)]
        if hasattr(lib, "wg_rt_upload_motion_hi"):   # motion at output resolution (not in older runtime libraries)
            lib.wg_rt_upload_motion_hi.argtypes = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int, vp,
                                                   P(ctypes.c_float), P(ctypes.c_float)]

        self.d_row = resolve_d_row(d_row if d_row is not None else os.environ.get("WG_D_ROW", "gfx12"))
        geo, code_object = V.DEFAULT_GEOMETRY, V.code_object(variant)
        manifest = os.path.join(build_dir, V.MANIFEST)
        if os.path.exists(manifest):
            with open(manifest) as f:
                entry = json.load(f).get(variant)
            if entry is None:
                raise ValueError(f"variant {variant!r} was not built (see {manifest})")
            geo, code_object = entry["geometry"], entry["file"]
            if geo.get("profile"):
                raise ValueError(f"variant {variant!r} is an instrumented profiling build (Runner.profile only)")
        self.max_rw, self.max_rh = (int(v) for v in max_render_size)
        self.ow, self.oh = (int(v) for v in output_size)
        self.dw, self.dh = (3 * self.ow + 1) // 2, (3 * self.oh + 1) // 2
        flags = dict(flags or {})
        self.auto_exposure = bool(flags.get("auto_exposure", False))
        cfg = WgRtConfig()
        self._keep = [os.fsencode(build_dir), os.fsencode(rocm_dir or ""), os.fsencode(model_dir),
                      os.fsencode(code_object), os.fsencode(self._emu_library(build_dir, variant)),
                      os.fsencode(map_path or "")]
        (cfg.build_dir, cfg.rocm_dir, cfg.model_dir, cfg.swin_code_object, cfg.emu_library,
         cfg.map_path) = self._keep
        for key, (field, _) in KERNEL_KEYS.items():
            if kernels and kernels.get(key):
                self._keep.append(kernels[key].encode())
                setattr(cfg, field, self._keep[-1])
        per_stage = ",".join(f"{s}={kernels['swin.' + s]}" for s in SWIN_ORDER if (kernels or {}).get("swin." + s))
        if (kernels or {}).get("post_wave"):   # 32: post kernels from post.co (default: post_w64.co when built)
            cfg.post_wave = int(kernels["post_wave"])
        if per_stage:   # per-stage Swin build variants (else WG_SWIN_STAGE_VARIANTS, read by the runtime)
            self._keep.append(per_stage.encode())
            cfg.swin_stage_variants = self._keep[-1]
        cfg.model_format = MODEL_FORMATS[model_format]
        cfg.backend = {"hip": 0, "emu": 1, "dx12": 2, "d3d12": 3}[backend]
        cfg.max_rw, cfg.max_rh, cfg.ow, cfg.oh = self.max_rw, self.max_rh, self.ow, self.oh
        cfg.depth_inverted = int(bool(flags.get("depth_inverted", False)))
        cfg.hdr = int(bool(flags.get("hdr", False)))   # model K's [HDR] variant (runtime option model=K)
        cfg.recon_output = int(flags.get("recon") == "output")   # experimental D = O (also WG_RECON_SIZE=output)
        cfg.auto_exposure = int(self.auto_exposure)
        cfg.alpha = float(alpha)
        cfg.gemm_threads, cfg.nt = geo["gemm_threads"], geo["nt"]
        cfg.swin_force_clamp = int(bool(geo.get("force_clamp", False)))
        cfg.swin_fuse = int(bool(geo.get("fuse", False)))
        for k, kernel in enumerate(("swin_block_c64", "swin_block_c64_add", "swin_block_c96", "swin_block_c96_add",
                                    "swin_block_c128")):
            cfg.block_threads[k] = geo["threads"][kernel]
            cfg.block_wpg[k] = geo.get("wpg", {}).get(kernel, 0)
        cfg.wpw64, cfg.wpw96, cfg.wpw128 = (geo["wpw"]["swin_block_c64"], geo["wpw"]["swin_block_c96"],
                                            geo["wpw"]["swin_block_c128"])
        for i, v in enumerate(np.asarray(self.d_row).reshape(-1)):
            cfg.d_row[i] = int(v)
        self._cfg = cfg
        rt, err = ctypes.c_void_p(), ctypes.create_string_buffer(2048)
        if lib.wg_rt_create(ctypes.byref(cfg), ctypes.byref(rt), err, len(err)) != 0:
            raise RuntimeError("runtime create failed: " + err.value.decode(errors="replace"))
        self.rt = rt
        self.model_source = lib.wg_rt_model_source(rt).decode()
        # D in use: spec 01, or O with the experimental recon_output / WG_RECON_SIZE=output (read by the runtime)
        dw, dh = ctypes.c_int(), ctypes.c_int()
        lib.wg_rt_recon_size(rt, ctypes.byref(dw), ctypes.byref(dh))
        self.dw, self.dh = dw.value, dh.value
        self.recon_output = (self.dw, self.dh) == (self.ow, self.oh)
        if check_packing:
            self.check_packing(model_dir)

    def __del__(self):
        if getattr(self, "rt", None):
            self.lib.wg_rt_destroy(self.rt)
            self.rt = None

    # ---- spec 90 API -------------------------------------------------------------------------------------------------

    def frame(self, x: dict, debug: bool = False) -> dict:
        self._submit(x)
        self._check(self.lib.wg_rt_sync(self.rt), "frame")
        out = {"out": self._output()}
        if debug:
            out.update(self.debug_outputs())
        return out

    def time_frame(self, x: dict, repeats: int = 30) -> float:
        """Median GPU time in ms of one whole frame (after one warm-up frame; inputs already on the GPU)."""
        fr = self._upload(x)
        self._check(self.lib.wg_rt_frame(self.rt, ctypes.byref(fr)), "frame")
        fr.reset = 0
        times = []
        ms = ctypes.c_float()
        for _ in range(int(repeats)):
            self._check(self.lib.wg_rt_frame_timed(self.rt, ctypes.byref(fr), ctypes.byref(ms)), "frame")
            times.append(ms.value)
        return float(np.median(times))

    # ---- extras -----------------------------------------------------------------------------------------------------

    def info(self) -> dict:
        i = WgRtFrameInfo()
        self.lib.wg_rt_info(self.rt, ctypes.byref(i))
        return {"index": i.index, "n": i.n, "parity": i.parity, "reset": bool(i.reset), "render_size": (i.rw, i.rh),
                "jitter_prev": (i.jitter_prev[0], i.jitter_prev[1]),
                "offsets": {s: (i.offsets[k][0], i.offsets[k][1]) for k, s in enumerate(SWIN_ORDER)}}

    def kernels(self) -> dict:
        """The kernels in use: {"swin": code object, "enc0": code object, "dec0": code object, "exposure": variant,
        "post": kernel, "filter": kernel, "post_last": post kernel of the last frame (post_recon when the frame's
        render size is too large for the chosen one)}."""
        return dict(kv.split("=", 1) for kv in self.lib.wg_rt_kernels(self.rt).decode().split())

    def stage_blob(self, stage: str) -> bytes:
        p, n = ctypes.c_void_p(), ctypes.c_size_t()
        self._check(self.lib.wg_rt_stage_blob(self.rt, stage.encode(), ctypes.byref(p), ctypes.byref(n)), "blob")
        return ctypes.string_at(p, n.value)

    def notes(self) -> list:
        """The runtime's notes of create for a log (wg_rt_notes), e.g. that the output filter is "analytic" because
        the model file has no output filter weights."""
        if not self._has_notes:
            return []
        return [n for n in self.lib.wg_rt_notes(self.rt).decode().split("\n") if n]

    def output_taps(self):
        """The model's output-filter weights (group filter / output_taps of spec 08 draft 2) as the runtime loaded
        them, float32 [5]; None when the model data has no such group."""
        p, n = ctypes.c_void_p(), ctypes.c_size_t()
        if self.lib.wg_rt_stage_blob(self.rt, b"filter", ctypes.byref(p), ctypes.byref(n)) != 0:
            return None
        return np.frombuffer(ctypes.string_at(p, n.value), np.float32).copy()

    def check_packing(self, model_dir):
        """The C++ packers must give the same bytes as Runner.pack_weights (clean/runner/packing.py etc.)."""
        tensors = logical_tensors(read_model_tensors(model_dir))
        for stage in ["enc0"] + SWIN_ORDER + ["dec0", "post"]:
            if stage == "enc0":
                ref = enc0.pack(tensors[stage], self.d_row)
            elif stage == "dec0":
                ref = dec0.pack(tensors[stage], self.d_row)
            elif stage == "post":
                ref = post.pack(tensors[stage])
            else:
                ref = pack_stage(stage, tensors[stage], self.d_row)
            got = self.stage_blob(stage)
            if got != ref:
                raise RuntimeError(f"runtime packing of {stage} differs from pack_weights")

    def debug_outputs(self) -> dict:
        info = self.info()
        rw, rh = info["render_size"]
        fw, fh = -(-rw // 32) * 32, -(-rh // 32) * 32
        u32 = np.uint32

        def fp8(name, shape):
            return fp8_decode(self._get(name, np.uint8)).reshape(shape)

        def h2(name):
            return self._get(name, np.float16).astype(np.float32).reshape(rh, rw, 2)

        def h4(name, h, w):
            return self._get(name, np.float16).astype(np.float32).reshape(h, w, 4)[..., :3]

        def unorm3(name):
            return (self._get(name, np.uint8).reshape(rh, rw, 4)[..., :3] / 255.0).astype(np.float32)

        def snorm4(name):
            return (self._get(name, np.int8).reshape(rh, rw, 4) / 127.0).astype(np.float32)

        pair = self._get("exposure.pair", np.float32)
        d = {"exposure": {"e": float(pair[0]), "e_prev": float(pair[1])},
             "enc0": {"features": fp8("enc0.features", (fh // 2, fw // 2, 64)), "m_near": h2("enc0.m_near"),
                      "m_far": h2("enc0.m_far"), "m_center": h2("enc0.m_center")}}
        size = {"enc1": (fh // 2, fw // 2), "enc2": (fh // 4, fw // 4), "dec2": (fh // 4, fw // 4),
                "dec1": (fh // 2, fw // 2)}
        for s in SWIN_ORDER:
            if s in ("enc1", "enc2"):
                (H, W), (C, Co) = size[s], SWIN_CHANNELS[s]
                d[s] = {"skip": fp8(s + ".skip", (H, W, C)), "merged": fp8(s + ".merged", (H // 2, W // 2, Co))}
            elif s in ("dec2", "dec1"):
                (H, W), (C, _) = size[s], SWIN_CHANNELS[s]
                d[s] = {"out": fp8(s + ".out", (H, W, C))}
            else:
                d[s] = {"out": fp8(s + ".out", (fh // 8, fw // 8, 128))}
        if "pack12" in self.kernels()["dec0"]:   # A, C in the pack12 images (FORMATS.md), B unused
            d["dec0"] = {"A": None, "B": None, "C": None, "latent": snorm4("dec0.latent")}
        else:
            d["dec0"] = {"A": unorm3("dec0.A"), "B": unorm3("dec0.B"), "C": h4("dec0.C", rh, rw),
                         "latent": snorm4("dec0.latent")}
        red = self._get("post.reduced", u32).reshape(self.dh // 2, self.dw // 2)
        d["post"] = {"history": self._history("post.history"),
                     "reduced": (np.stack([(red >> (10 * c)) & 1023 for c in range(3)], -1) / 1023.0).astype(
                         np.float32),
                     # the fused post kernels write no display image
                     "display": None if self._fused() else
                     post.decode_planes(self._get("post.display", np.uint8).tobytes(), (self.dh, self.dw))}
        d["filter"] = {"out": self._output()}
        d["state"] = {"latent_prev": snorm4("state.latent_prev"), "motion_history": h2("state.motion_history"),
                      "history_prev": self._history("state.history_prev"), "info": info}
        return d

    # ---- internals ---------------------------------------------------------------------------------------------------

    def _fused(self):
        return self.kernels()["post"].startswith("post_fused")

    def _output(self):
        """filter.out: 3 x f32, or RGBA16F for the post_fused*_o16 kernels -> [oh, ow, 3] float32."""
        if self.kernels()["out"] == "rgba16f":
            return self._get("filter.out", np.float16).reshape(self.oh, self.ow, 4)[..., :3].astype(np.float32)
        return self._get("filter.out", np.float32).reshape(self.oh, self.ow, 3).copy()

    def _history(self, name):
        """Full history in the post kernel's format (8: RGBA16F, 6: RGB16F, 4: compact) -> [dh, dw, 3] float32."""
        k = self.kernels()
        nb = int(k["history_bytes"])
        raw = self._get(name, np.uint8)
        if k.get("history_format") == "hp":   # planar: R, G (2 x FP16) per pixel, then the B plane (FP16)
            n = self.dh * self.dw
            rg = np.frombuffer(raw.tobytes(), np.float16, 2 * n).reshape(self.dh, self.dw, 2)
            bb = np.frombuffer(raw.tobytes(), np.float16, n, 4 * n).reshape(self.dh, self.dw, 1)
            return np.concatenate([rg, bb], -1).astype(np.float32)
        if nb == 4:
            return post.decode_history4(raw.tobytes(), (self.dh, self.dw))
        return post.decode_history_fmt(raw.tobytes(), (self.dh, self.dw), nb == 6)

    @staticmethod
    def _emu_library(build_dir, variant):
        """Emulator backend: the library of the Swin build variant when there is one (build.py --emu --variants)."""
        lib = os.path.join(build_dir, f"wg_emu__{variant}.so")
        return lib if variant != "base" and os.path.exists(lib) else os.path.join(build_dir, "wg_emu.so")

    def _check(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"runtime {what} failed: " + self.lib.wg_rt_error(self.rt).decode(errors="replace"))

    def _upload(self, x):
        color = np.ascontiguousarray(x["color"], dtype=np.float32)
        rh, rw = color.shape[:2]
        if color.shape != (rh, rw, 3):
            raise ValueError(f"color must be [rh, rw, 3], got {color.shape}")
        depth = np.ascontiguousarray(x["depth"], dtype=np.float32)
        # Motion at output resolution (spec 03 draft 4, spec 06 section 3; kernels={"options": "motion_outres=original"}):
        # x["motion_hi"] = the motion of the region M [mh][mw][2] (float32, or float16 for the FP16 texel format),
        # x["mv_scale"] converts it to pixels of M and x["mv_bias"] is b_mv; x["motion"] (render size) may then be
        # absent. Without "motion_hi" a previous one is switched off.
        hi = x.get("motion_hi")
        motion = np.ascontiguousarray(x["motion"] if "motion" in x or hi is None else np.zeros((rh, rw, 2)),
                                      dtype=np.float32)
        if depth.shape != (rh, rw) or motion.shape != (rh, rw, 2):
            raise ValueError("depth must be [rh, rw] and motion [rh, rw, 2]")
        self._check(self.lib.wg_rt_upload_inputs(self.rt, rw, rh, color.ctypes.data, depth.ctypes.data,
                                                 motion.ctypes.data), "upload")
        if hi is not None:
            hi = np.ascontiguousarray(hi, dtype=np.float16 if np.asarray(hi).dtype == np.float16 else np.float32)
            if hi.ndim != 3 or hi.shape[2] != 2:
                raise ValueError("motion_hi must be [mh, mw, 2]")
            f2 = ctypes.c_float * 2
            sc = f2(*[float(v) for v in np.asarray(x["mv_scale"], np.float64).reshape(-1)[:2]])
            bb = f2(*[float(v) for v in np.asarray(x.get("mv_bias", (0.0, 0.0)), np.float64).reshape(-1)[:2]])
            self._check(self.lib.wg_rt_upload_motion_hi(self.rt, hi.shape[1], hi.shape[0], int(hi.dtype == np.float32),
                                                        hi.ctypes.data, sc, bb), "upload motion_hi")
        elif getattr(self, "_hi_on", False):
            self._check(self.lib.wg_rt_upload_motion_hi(self.rt, 0, 0, 0, None, None, None), "motion_hi off")
        self._hi_on = hi is not None
        fr = WgRtFrame()
        fr.rw, fr.rh = rw, rh
        for i in range(2):
            fr.mv_scale[i] = float(np.asarray(x["mv_scale"], np.float64).reshape(-1)[i])
            fr.mv_bias[i] = float(np.asarray(x.get("mv_bias", (0.0, 0.0)), np.float64).reshape(-1)[i])
            fr.jitter[i] = float(np.asarray(x["jitter"], np.float64).reshape(-1)[i])
        fr.exposure = float(x.get("exposure", 1.0))
        fr.pre_exposure = float(x.get("pre_exposure", 1.0))
        fr.reset = int(bool(x.get("reset", False)))
        return fr

    def _submit(self, x):
        fr = self._upload(x)
        self._check(self.lib.wg_rt_frame(self.rt, ctypes.byref(fr)), "frame")

    def _get(self, name, dtype):
        p, n = ctypes.c_size_t(), ctypes.c_size_t()
        self._check(self.lib.wg_rt_buffer(self.rt, name.encode(), ctypes.byref(p), ctypes.byref(n)), name)
        buf = np.empty(n.value, dtype=np.uint8)
        self._check(self.lib.wg_rt_download(self.rt, p.value, buf.ctypes.data, n.value), name)
        return buf.view(dtype)
