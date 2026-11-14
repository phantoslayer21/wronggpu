# SPDX-License-Identifier: MIT
"""Test interface of spec/90-test-interface.md: run the gfx1201 stage kernels from Python (NumPy + ctypes).

Stages: the Swin stages of spec 02, and the stages in EXTRA_STAGES: enc0 (input encoder, spec 03, runner/enc0.py),
dec0 (coefficient decoder, spec 05, runner/dec0.py), exposure (auto-exposure, spec 07, runner/exposure.py, no
weights), post / filter (spec 06; kernels/post.hip, runner/post.py).

    from clean.runner.stage_runner import Runner          # or: sys.path.append("clean/runner"); import stage_runner
    r = Runner(build_dir, rocm_dir)                         # build_dir: output of clean/runner/build.py
    w = r.pack_weights("enc1", tensors)
    out = r.run("enc1", w, x, (ox, oy))                     # {"skip": ..., "merged": ...}
    ms = r.time("enc1", w, x, (ox, oy))

Extra keyword arguments (not needed by the spec team): backend="emu" runs the kernels in the host emulator built by
`build.py --emu`; d_row selects the WMMA accumulator layout assumed by the weight packing (see selftest.py).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ctypes  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

from clean.reference.fp8 import fp8_decode, fp8_encode  # noqa: E402
from clean.reference.swin import MODEL_STAGES, STAGES, stage_table, window_offset  # noqa: E402,F401
from clean.runner import dec0, dec1dec0, enc0, exposure, post  # noqa: E402
from clean.runner import perf_variants as PV  # noqa: E402
from clean.runner import variants as V  # noqa: E402
from clean.runner.packing import (MID_STAGES, pack_bneck, pack_mid, pack_stage, split_bneck, split_mid,  # noqa: E402
                                  stage_sections)

# Kernel names. Launch geometry (workgroup sizes, windows per wave, token groups per wave) comes from the build
# variant (variants.py, recorded in build_dir/variants.json); the defaults below match the unmodified kernels.
BLOCK_KERNELS = {64: "swin_block_c64", 96: "swin_block_c96", 128: "swin_block_c128"}
MERGE_KERNELS = {64: "patch_merge_c64", 96: "patch_merge_c96"}
EXPAND_KERNELS = {128: "expand_c128", 96: "expand_c96"}
# fused kernels (variant `fuse`): encoder blocks that also do the patch merge, decoder blocks that also do the expand
MERGE_FUSED = {64: "swin_block_c64_mg", 96: "swin_block_c96_mg"}
EXPAND_FUSED = {64: "swin_block_c64_add_ex", 96: "swin_block_c96_add_ex"}


def _module_stage(m):
    """(pack, prepare) of a stage module with CODE_OBJECT, KERNELS, pack(tensors, d_row), prepare(dev, kernels, ...)."""
    kernels = m.KERNELS[0] if len(m.KERNELS) == 1 else m.KERNELS
    return m.pack, lambda dev, weights, x: m.prepare(dev, kernels, weights, x)


# Stages with their own weight packing and launch plans: stage -> (pack(tensors, d_row) or None,
# prepare(dev, weights, x)).
EXTRA_STAGES = {"enc0": _module_stage(enc0), "dec0": _module_stage(dec0), "exposure": _module_stage(exposure),
                "post": (lambda t, d_row: post.pack(t), post.prepare_post), "filter": (None, post.prepare_filter)}
# Filter build variants for timing (post_bench.py): stage "filter:<kernel name>", same inputs and outputs as "filter".
EXTRA_STAGES.update({f"filter:{k}": (None, functools.partial(post.prepare_filter, kernel=k))
                     for k in post.FILTER_VARIANTS})
# Build variants of enc0 / dec0 / exposure for timing (perf_bench.py, perf_variants.py): stage "<stage>@<variant>",
# same inputs and outputs as the stage. Their kernels are addressed as "<code object>:<kernel>".
EXTRA_STAGES.update({f"{st}@{v}": (dec0.pack_v1 if (st, v) == ("dec0", "v1") else EXTRA_STAGES[st][0],
                                    functools.partial(m.prepare_variant, variant=v))
                     for st, m in (("enc0", enc0), ("dec0", dec0), ("exposure", exposure))
                     for v in PV.VARIANTS[st] if v != "base"})
# dec1 followed by dec0, separate or fused (kernels/dec1_dec0.hip; runner/dec1dec0.py): prototype for timing.
EXTRA_STAGES["dec1dec0"] = (dec1dec0.pack, functools.partial(dec1dec0.prepare, fused=False))
EXTRA_STAGES.update({f"dec1dec0@{v}": (dec1dec0.pack, functools.partial(dec1dec0.prepare, fused=True, code_object=co))
                     for v, (co, _) in PV.FUSION.items()})
# Kernels outside the Swin code object (self.module), loaded on first use.
KERNEL_CODE_OBJECTS = {k: m.CODE_OBJECT for m in (enc0, dec0, exposure, post) for k in m.KERNELS}
# Model K (spec 40/41, clean/k): stages "k.<stage>" (or plain stage names with Runner(..., model="K")).
from clean.k.runner import k_stages  # noqa: E402
KERNEL_CODE_OBJECTS.update(k_stages.KERNEL_CODE_OBJECTS)
# Models: the ones with a reference stage table (clean/reference/swin.py MODEL_STAGES: "M") and model K (clean/k)
MODELS = tuple(MODEL_STAGES) + ("K",)
EMU_LIBRARY = "wg_emu.so"


def _cdiv(a, b):
    return -(-a // b)


class Runner:
    def __init__(self, build_dir: str, rocm_dir: str, backend: str = "hip", d_row=None, variant: str = "base",
                 model: str = "M"):
        self.model = str(model).upper()
        if self.model not in MODELS:
            raise ValueError(f"unknown model {model!r}; known: {', '.join(MODELS)}")
        self.build_dir = build_dir
        self.variant = variant
        self.backend = backend
        self._extra_modules = {}
        self.d_row = d_row if d_row is not None else os.environ.get("WG_D_ROW", "gfx12")
        self.geo, code_object = V.DEFAULT_GEOMETRY, V.code_object(variant)
        manifest = os.path.join(build_dir, V.MANIFEST)
        if os.path.exists(manifest):
            with open(manifest) as f:
                entry = json.load(f).get(variant)
            if entry is None:
                raise ValueError(f"variant {variant!r} was not built (see {manifest})")
            self.geo, code_object = entry["geometry"], entry["file"]
        elif variant != "base":
            raise ValueError(f"variant {variant!r} requested but {manifest} does not exist (build.py --variants)")
        if backend == "hip":
            from clean.runner.hip import Hip
            self.dev = Hip(rocm_dir)
            with open(os.path.join(build_dir, code_object), "rb") as f:
                self.module = self.dev.load_module(f.read())
        elif backend == "emu":
            from clean.runner.emu import Emu
            self.dev = Emu(os.path.join(build_dir, EMU_LIBRARY if variant == "base" else f"wg_emu__{variant}.so"))
            self.module = None
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self._funcs = {}
        self.mega_groups = None  # megakernel grid (None: the variant's default, variants.py geometry["mega"])
        self.mid_groups = None   # swin_mid grid (None: the variant's default, variants.py geometry["mid"])

    # ---- public API (spec 90) --------------------------------------------------------------------------------------

    def _k_stage(self, stage):
        """Model K stage name without the "k." prefix, or None for a model M stage."""
        if stage.startswith("k."):
            return stage[2:]
        return stage if self.model == "K" else None

    def pack_weights(self, stage: str, tensors: dict) -> bytes:
        if self._k_stage(stage) is not None:
            return k_stages.pack(self._k_stage(stage), tensors, self.d_row)
        if stage == "bneck":  # {"b0": ..., "b5": ...} (or the six in order) -> the six b-stage blobs joined
            return pack_bneck(tensors, self.d_row, self.model)
        if stage == "mid":  # {"enc2": ..., "b0": ..., ..., "b5": ..., "dec2": ...} -> the eight stage blobs joined
            return pack_mid(tensors, self.d_row, self.model)
        if stage in EXTRA_STAGES:
            packer = EXTRA_STAGES[stage][0]
            if packer is None:
                raise ValueError(f"stage {stage!r} has no weights")
            return packer(tensors, self.d_row)
        return pack_stage(stage, tensors, self.d_row, self.model)

    def run(self, stage: str, weights: bytes, x, offset=None, skip=None) -> dict:
        return self._run_plan(self._prepare(stage, weights, x, offset, skip))

    def _run_plan(self, plan) -> dict:
        try:
            for name, _ in plan["outputs"]:
                # 0x7F bytes are NaN in FP8 and FP16: unwritten values show up
                self.dev.memset(plan["bufs"][name], 0x7F, plan["sizes"][name])
            self._execute(plan)
            self.dev.sync()
            if "sync" in plan["bufs"]:  # megakernel: its sync words must be zero again for the next launch
                sync = np.frombuffer(self.dev.download(plan["bufs"]["sync"], plan["sizes"]["sync"]), dtype=np.uint32)
                if np.any(sync):
                    raise RuntimeError(f"swin_bneck6 left {int(np.count_nonzero(sync))} sync words non-zero")
            decoders = plan.get("decoders", {})
            out = {}
            for name, shape in plan["outputs"]:
                raw = self.dev.download(plan["bufs"][name], plan["sizes"][name])
                dec = decoders.get(name)
                out[name] = dec(raw, shape) if dec else fp8_decode(np.frombuffer(raw, dtype=np.uint8)).reshape(shape)
            return plan["finish"](out) if "finish" in plan else out  # optional conversion of the plan's outputs
        finally:
            self._release(plan)

    def time(self, stage: str, weights: bytes, x, offset=None, skip=None, repeats: int = 50) -> float:
        """Median GPU time in ms of the stage's kernels (inputs already on the device)."""
        plan = self._prepare(stage, weights, x, offset, skip)
        start, stop = self.dev.event(), self.dev.event()
        try:
            self._execute(plan)  # warm-up
            self.dev.sync()
            times = []
            for _ in range(int(repeats)):
                self.dev.record(start)
                self._execute(plan)
                self.dev.record(stop)
                times.append(self.dev.elapsed_ms(start, stop))
            return float(np.median(times))
        finally:
            self.dev.destroy_event(start)
            self.dev.destroy_event(stop)
            self._release(plan)

    def time_kernels(self, stage: str, weights: bytes, x, offset=None, skip=None, repeats: int = 50,
                     stride: int = 0) -> dict:
        """Median GPU time in ms of each kernel launch of the stage, in launch order: {kernel name: ms}. With an
        instrumented build, `stride` is the sampling stride of the instrumented waves (0 = none instrumented)."""
        plan = self._prepare(stage, weights, x, offset, skip)
        if "profile" in plan:
            plan["profile"]["stride"].value = int(stride)
        start, stop = self.dev.event(), self.dev.event()
        try:
            self._execute(plan)  # warm-up
            self.dev.sync()
            result = {}
            for name, grid, threads, args in plan["launches"]:
                times = []
                for _ in range(int(repeats)):
                    self.dev.record(start)
                    self.dev.launch(self._func(name), grid, threads, args)
                    self.dev.record(stop)
                    times.append(self.dev.elapsed_ms(start, stop))
                result[name] = float(np.median(times))
            return result
        finally:
            self.dev.destroy_event(start)
            self.dev.destroy_event(stop)
            self._release(plan)

    def profile(self, stage: str, weights: bytes, x, offset, skip=None, stride: int = 1):
        """Run the stage once with an instrumented build (variant prof / prof2) and return (words, info): words is a
        uint32 array [waves, 16] with the per-wave phase times (see WG_PROFILE in kernels/swin.hip), info describes the
        launch. Only waves whose index is a multiple of `stride` are instrumented (the other rows stay zero): every clock
        read perturbs the kernel, so large launches should be sampled. Needs build.py --profile and
        Runner(..., variant="prof")."""
        plan = self._prepare(stage, weights, x, offset, skip)
        try:
            if "profile" not in plan:
                raise ValueError('not an instrumented build: use Runner(..., variant="prof") (build.py --profile)')
            plan["profile"]["stride"].value = int(stride)
            self._execute(plan)  # warm-up
            self.dev.sync()
            self.dev.memset(plan["bufs"]["prof"], 0, plan["sizes"]["prof"])
            self._execute(plan)
            self.dev.sync()
            raw = self.dev.download(plan["bufs"]["prof"], plan["sizes"]["prof"])
            info = {k: v for k, v in plan["profile"].items() if k != "stride"}
            info["stride"] = int(stride)
            return np.frombuffer(raw, dtype=np.uint32).reshape(-1, V.PROF_WORDS).copy(), info
        finally:
            self._release(plan)

    # ---- whole bottleneck (b0..b5), six launches or the megakernel (variants bmega*) ---------------------------------

    def run_bneck(self, weights, x, parity=0, mega=True, groups=None) -> np.ndarray:
        """Run the six bottleneck blocks b0..b5 on x [H, W, 128]: one swin_bneck6 launch (mega=True, needs a bmega*
        build) or six block-kernel launches (mega=False, any build). weights: a "bneck" blob (pack_weights("bneck",
        ...)) or the six packed b-stage blobs; parity: window-offset parity (spec 02). Returns b5's output.
        run("bneck", ...) is the same with the build's default (the megakernel when the build has one)."""
        plan = self._prepare_bneck(weights, x, parity, mega, groups)
        return self._run_plan(plan)["out"]

    def time_bneck(self, weights, x, parity=0, mega=True, groups=None, repeats: int = 50) -> float:
        """Median GPU time in ms of the whole bottleneck (see run_bneck), inputs already on the device."""
        plan = self._prepare_bneck(weights, x, parity, mega, groups)
        start, stop = self.dev.event(), self.dev.event()
        try:
            self._execute(plan)  # warm-up
            self.dev.sync()
            times = []
            for _ in range(int(repeats)):
                self.dev.record(start)
                self._execute(plan)
                self.dev.record(stop)
                times.append(self.dev.elapsed_ms(start, stop))
            return float(np.median(times))
        finally:
            self.dev.destroy_event(start)
            self.dev.destroy_event(stop)
            self._release(plan)

    def _prepare_bneck(self, weights, x, parity, mega, groups):
        weights = split_bneck(weights)
        x = np.asarray(x, dtype=np.float32)
        H, W = x.shape[:2]
        if x.shape != (H, W, 128) or H % 4 or W % 4:
            raise ValueError(f"x must be [H, W, 128] with H, W multiples of 4, got {x.shape}")
        offs = [window_offset(f"b{k}", parity) for k in range(6)]
        if any(offs[k] != offs[k % 2] for k in range(6)):
            raise ValueError("b-block window offsets must alternate between two values")
        force_clamp = bool(self.geo.get("force_clamp", False))
        safe = [(int.from_bytes(w[:4], "little") & 1) == 1 and not force_clamp for w in weights]
        bufs, sizes = {}, {}
        plan = {"bufs": bufs, "sizes": sizes, "launches": [], "outputs": [("out", (H, W, 128))]}

        def alloc(name, nbytes, data=None):
            bufs[name] = self.dev.malloc(nbytes)
            sizes[name] = nbytes
            if data is not None:
                self.dev.upload(bufs[name], data)

        P, I = ctypes.c_void_p, ctypes.c_int
        try:
            for k, w in enumerate(weights):
                alloc(f"w{k}", len(w), w)
            alloc("x", x.size, fp8_encode(x).tobytes())
            alloc("tmp", H * W * 128)
            alloc("out", H * W * 128)
            # ping-pong as in the megakernel: x -> tmp -> out -> tmp -> out -> tmp -> out
            src = [bufs["x"]] + [bufs["tmp"] if k % 2 else bufs["out"] for k in range(1, 6)]
            dst = [bufs["out"] if k % 2 else bufs["tmp"] for k in range(6)]
            if mega:
                geo = self.geo.get("mega")
                if geo is None:
                    raise ValueError(f"variant {self.variant!r} has no bottleneck megakernel (build a bmega* variant)")
                nsync = 16 + 6 * ((H + 6) // 4)  # kMegaSyncHead + 6 blocks x window rows (swin_v2.hip)
                alloc("sync", 4 * nsync)
                self.dev.memset(bufs["sync"], 0, 4 * nsync)
                groups = int(groups or self.mega_groups or geo["groups"])
                (oxe, oye), (oxo, oyo) = offs[0], offs[1]
                plan["launches"].append(
                    ("swin_bneck6" + ("_nc" if all(safe) else ""), groups, geo["threads"],
                     [P(bufs["x"]), P(bufs["out"]), P(bufs["tmp"]), P(bufs["sync"])]
                     + [P(bufs[f"w{k}"]) for k in range(6)]
                     + [I(H), I(W), I(oxe), I(oye), I(oxo), I(oyo), I(groups)]))
            else:
                kname = BLOCK_KERNELS[128]
                threads = self.geo["threads"][kname]
                wpg = self.geo["wpg"][kname] if "wpg" in self.geo else threads // 32 * self.geo["wpw"][kname]
                for k in range(6):
                    ox, oy = offs[k]
                    nwin = ((W + ox + 3) // 4) * ((H + oy + 3) // 4)
                    plan["launches"].append(
                        (kname + ("_nc" if safe[k] else ""), _cdiv(nwin, wpg), threads,
                         [P(src[k]), P(0), P(dst[k]), P(bufs[f"w{k}"]), I(H), I(W), I(ox), I(oy)]))
        except Exception:
            self._release(plan)
            raise
        return plan

    def run_mid(self, weights, x, parity=0, mega=True, groups=None) -> np.ndarray:
        """Run enc2 (block and patch merge), b0..b5 and dec2 (expand and block) on x = enc1's merged output
        [H, W, 96]: one swin_mid launch (mega=True, needs a bmega* build) or ten launches (mega=False). Returns dec2's
        output [H, W, 96]. run("mid", ...) is the same with the build's default."""
        return self._run_plan(self._prepare_mid(weights, x, parity, mega, groups))["out"]

    def time_mid(self, weights, x, parity=0, mega=True, groups=None, repeats: int = 50) -> float:
        plan = self._prepare_mid(weights, x, parity, mega, groups)
        start, stop = self.dev.event(), self.dev.event()
        try:
            self._execute(plan)  # warm-up
            self.dev.sync()
            times = []
            for _ in range(int(repeats)):
                self.dev.record(start)
                self._execute(plan)
                self.dev.record(stop)
                times.append(self.dev.elapsed_ms(start, stop))
            return float(np.median(times))
        finally:
            self.dev.destroy_event(start)
            self.dev.destroy_event(stop)
            self._release(plan)

    def _prepare_mid(self, weights, x, parity, mega, groups):
        blobs = dict(zip(MID_STAGES, split_mid(weights)))
        x = np.asarray(x, dtype=np.float32)
        H, W = x.shape[:2]
        if x.shape != (H, W, 96) or H % 8 or W % 8:
            raise ValueError(f"x must be [H, W, 96] with H, W multiples of 8, got {x.shape}")
        Hb, Wb = H // 2, W // 2
        offs = {s: tuple(window_offset(s, parity)) for s in MID_STAGES}
        force_clamp = bool(self.geo.get("force_clamp", False))
        safe = {s: (int.from_bytes(b[:4], "little") & 1) == 1 and not force_clamp for s, b in blobs.items()}
        bufs, sizes = {}, {}
        plan = {"bufs": bufs, "sizes": sizes, "launches": [], "outputs": [("out", (H, W, 96))]}

        def alloc(name, nbytes, data=None):
            bufs[name] = self.dev.malloc(nbytes)
            sizes[name] = nbytes
            if data is not None:
                self.dev.upload(bufs[name], data)

        P, I = ctypes.c_void_p, ctypes.c_int
        try:
            for s_, b in blobs.items():
                alloc("w_" + s_, len(b), b)
            alloc("x", x.size, fp8_encode(x).tobytes())
            alloc("skip", H * W * 96)
            alloc("A", Hb * Wb * 128)
            alloc("B", Hb * Wb * 128)
            alloc("out", H * W * 96)
            if mega:
                geo = self.geo.get("mid")
                if geo is None:
                    raise ValueError(f"variant {self.variant!r} has no swin_mid kernel (build a bmega* variant)")
                nsync = 16 + 8 * ((H + 6) // 4)  # kMegaSyncHead + 8 phases x window rows (swin_v2.hip)
                alloc("sync", 4 * nsync)
                self.dev.memset(bufs["sync"], 0, 4 * nsync)
                groups = int(groups or self.mid_groups or geo["groups"])
                o = [v for s_ in ("enc2", "b0", "b1", "dec2") for v in offs[s_]]
                plan["launches"].append(
                    ("swin_mid" + ("_nc" if all(safe.values()) else ""), groups, geo["threads"],
                     [P(bufs["x"]), P(bufs["out"]), P(bufs["skip"]), P(bufs["A"]), P(bufs["B"]), P(bufs["sync"])]
                     + [P(bufs["w_" + s_]) for s_ in MID_STAGES]
                     + [I(H), I(W)] + [I(v) for v in o] + [I(groups)]))
            else:  # the launches of run("enc2"), run("b0") ... run("b5"), run("dec2"), unfused
                geo = self.geo
                gemm, nt = geo["gemm_threads"], geo["nt"]
                alloc("e", H * W * 96)

                def block(stage, C, add, src, xadd, dst, h, w):
                    kname = BLOCK_KERNELS[C] + ("_add" if add else "")
                    threads = geo["threads"][kname]
                    wpg = geo["wpg"][kname] if "wpg" in geo else threads // 32 * geo["wpw"][kname]
                    ox, oy = offs[stage]
                    nwin = ((w + ox + 3) // 4) * ((h + oy + 3) // 4)
                    plan["launches"].append(
                        (kname + ("_nc" if safe[stage] else ""), _cdiv(nwin, wpg), threads,
                         [P(src), P(xadd), P(dst), P(bufs["w_" + stage]), I(h), I(w), I(ox), I(oy)]))

                block("enc2", 96, False, bufs["x"], 0, bufs["skip"], H, W)
                plan["launches"].append(
                    (MERGE_KERNELS[96], _cdiv(Hb * Wb, gemm // 32 * 16 * nt), gemm,
                     [P(bufs["skip"]), P(bufs["A"]), P(bufs["w_enc2"] + stage_sections("enc2")[1]), I(H), I(W)]))
                for k in range(6):
                    src, dst = (bufs["B"], bufs["A"]) if k % 2 else (bufs["A"], bufs["B"])
                    block(f"b{k}", 128, False, src, 0, dst, Hb, Wb)
                plan["launches"].append(
                    (EXPAND_KERNELS[128], _cdiv(Hb * Wb, gemm // 32 * 16 * nt), gemm,
                     [P(bufs["A"]), P(bufs["e"]), P(bufs["w_dec2"] + stage_sections("dec2")[1]), I(Hb), I(Wb)]))
                block("dec2", 96, True, bufs["e"], bufs["skip"], bufs["out"], H, W)
        except Exception:
            self._release(plan)
            raise
        return plan

    # ---- internals ---------------------------------------------------------------------------------------------------

    def _module(self, code_object):
        """code_object None: the Swin code object of the selected variant (self.module)."""
        if code_object is None or self.backend != "hip":
            return self.module
        if code_object not in self._extra_modules:
            with open(os.path.join(self.build_dir, code_object), "rb") as f:
                self._extra_modules[code_object] = self.dev.load_module(f.read())
        return self._extra_modules[code_object]

    def _func(self, name):
        if name not in self._funcs:
            if ":" in name:  # "<code object>:<kernel>" (build variants)
                co, kernel = name.split(":", 1)
                self._funcs[name] = self.dev.get_function(self._module(co), kernel)
            else:
                self._funcs[name] = self.dev.get_function(self._module(KERNEL_CODE_OBJECTS.get(name)), name)
        return self._funcs[name]

    def _prepare(self, stage, weights, x, offset, skip):
        if self._k_stage(stage) is not None:
            return k_stages.prepare(self.dev, self._k_stage(stage), weights, x, offset, skip)
        if stage == "bneck":  # b0..b5; offset = b0's window offset (it gives the parity)
            parity = [p for p in (0, 1) if tuple(window_offset("b0", p, self.model)) == tuple(int(v) for v in offset)]
            if not parity:
                raise ValueError(f"bneck offset {offset} is not a b0 window offset")
            return self._prepare_bneck(weights, x, parity[0], "mega" in self.geo, None)
        if stage == "mid":  # enc2 (+ merge), b0..b5, dec2 (+ expand); offset = enc2's window offset
            parity = [p for p in (0, 1) if tuple(window_offset("enc2", p, self.model)) == tuple(int(v) for v in offset)]
            if not parity:
                raise ValueError(f"mid offset {offset} is not an enc2 window offset")
            return self._prepare_mid(weights, x, parity[0], "mid" in self.geo, None)
        if stage in EXTRA_STAGES:
            return EXTRA_STAGES[stage][1](self.dev, weights, x)
        stages = stage_table(self.model)
        if stage not in stages:
            raise ValueError(f"unknown stage {stage!r}")
        cfg = stages[stage]
        C = cfg["C"]
        ox, oy = (int(v) for v in offset)
        if not (0 <= ox < 4 and 0 <= oy < 4):
            raise ValueError(f"window offset {offset} out of range")
        x = np.asarray(x, dtype=np.float32)
        if cfg["kind"] == "dec":
            if skip is None:
                raise ValueError(f"{stage} needs skip")
            skip = np.asarray(skip, dtype=np.float32)
            H, W = skip.shape[:2]
            if skip.shape != (H, W, C) or x.shape != (H // 2, W // 2, cfg["C_low"]):
                raise ValueError(f"{stage}: x {x.shape} / skip {skip.shape} do not match [H/2, W/2, "
                                 f"{cfg['C_low']}] / [H, W, {C}]")
        else:
            H, W = x.shape[:2]
            if x.shape != (H, W, C):
                raise ValueError(f"{stage}: x has shape {x.shape}, expected [H, W, {C}]")
        if H % 4 or W % 4:
            raise ValueError(f"H and W must be multiples of 4, got {H}x{W}")

        weights = bytes(weights)
        block_size, tail_off = stage_sections(stage, self.model)
        bufs, sizes = {}, {}

        def alloc(name, nbytes, data=None):
            bufs[name] = self.dev.malloc(nbytes)
            sizes[name] = nbytes
            if data is not None:
                self.dev.upload(bufs[name], data)

        plan = {"bufs": bufs, "sizes": sizes, "launches": []}
        try:
            alloc("w", len(weights), weights)
            wptr = bufs["w"]
            kname = BLOCK_KERNELS[C] + ("_add" if cfg["kind"] == "dec" else "")
            geo = self.geo
            swin_block, gemm_block, nt = geo["threads"][kname], geo["gemm_threads"], geo["nt"]
            wpw = geo["wpw"][kname]
            # fused merge / expand: 256 lanes, one window per wave, even window offsets (all real ones are)
            fused = bool(geo.get("fuse")) and cfg["kind"] in ("enc", "dec") and ox % 2 == 0 and oy % 2 == 0
            if fused:
                swin_block, wpw = 256, 1
            nwin = ((W + ox + 3) // 4) * ((H + oy + 3) // 4)
            # windows per workgroup: explicit in the geometry of variants that do not run whole windows per wave
            wpg = geo["wpg"][kname] if ("wpg" in geo and not fused) else (swin_block // 32) * wpw
            block_grid = _cdiv(nwin, wpg)
            # the weights' first word says whether every inner FP8 pack is proven in range (no clamp needed)
            safe = (int.from_bytes(weights[:4], "little") & 1) == 1 and not geo.get("force_clamp", False)
            if fused:
                kname = (EXPAND_FUSED if cfg["kind"] == "dec" else MERGE_FUSED)[C]
            launch_name = kname + ("_nc" if safe else "")
            profile = int(geo.get("profile", 0))
            prof_args = []
            if profile:
                waves = block_grid * (swin_block // 32)
                alloc("prof", waves * V.PROF_WORDS * 4)
                stride = ctypes.c_uint32(0)  # profile() / time_kernels(stride=) set it; 0 = no wave instrumented
                prof_args = [ctypes.c_void_p(bufs["prof"]), stride]
                plan["profile"] = {"waves": waves, "grid": block_grid, "threads": swin_block, "kernel": launch_name,
                                   "windows": nwin, "wpw": wpw, "stride": stride, "fused": fused}
            P, I = ctypes.c_void_p, ctypes.c_int
            if cfg["kind"] == "dec" and fused:
                alloc("y", x.size, fp8_encode(x).tobytes())
                alloc("skip_in", skip.size, fp8_encode(skip).tobytes())
                alloc("out", H * W * C)
                plan["launches"].append(
                    (launch_name, block_grid, swin_block,
                     [P(bufs["y"]), P(bufs["skip_in"]), P(bufs["out"]), P(wptr), I(H), I(W), I(ox), I(oy), P(0)] + prof_args))
                plan["outputs"] = [("out", (H, W, C))]
            elif cfg["kind"] == "dec":
                Hl, Wl = H // 2, W // 2
                alloc("y", x.size, fp8_encode(x).tobytes())
                alloc("skip_in", skip.size, fp8_encode(skip).tobytes())
                alloc("e", H * W * C)
                alloc("out", H * W * C)
                plan["launches"] += [
                    (EXPAND_KERNELS[cfg["C_low"]], _cdiv(Hl * Wl, gemm_block // 32 * 16 * nt), gemm_block,
                     [P(bufs["y"]), P(bufs["e"]), P(wptr + tail_off), I(Hl), I(Wl)]),
                    (launch_name, block_grid, swin_block,
                     [P(bufs["e"]), P(bufs["skip_in"]), P(bufs["out"]), P(wptr), I(H), I(W), I(ox), I(oy)] + prof_args),
                ]
                plan["outputs"] = [("out", (H, W, C))]
            else:
                alloc("x", x.size, fp8_encode(x).tobytes())
                outname = "skip" if cfg["kind"] == "enc" else "out"
                alloc(outname, H * W * C)
                if fused:
                    Co = cfg["C_out"]
                    alloc("merged", (H // 2) * (W // 2) * Co)
                    plan["launches"].append(
                        (launch_name, block_grid, swin_block,
                         [P(bufs["x"]), P(0), P(bufs[outname]), P(wptr), I(H), I(W), I(ox), I(oy), P(bufs["merged"])]
                         + prof_args))
                    plan["outputs"] = [(outname, (H, W, C)), ("merged", (H // 2, W // 2, Co))]
                else:
                    plan["launches"].append(
                        (launch_name, block_grid, swin_block,
                         [P(bufs["x"]), P(0), P(bufs[outname]), P(wptr), I(H), I(W), I(ox), I(oy)] + prof_args))
                    plan["outputs"] = [(outname, (H, W, C))]
                if cfg["kind"] == "enc" and not fused:
                    Co = cfg["C_out"]
                    alloc("merged", (H // 2) * (W // 2) * Co)
                    plan["launches"].append(
                        (MERGE_KERNELS[C], _cdiv((H // 2) * (W // 2), gemm_block // 32 * 16 * nt), gemm_block,
                         [P(bufs["skip"]), P(bufs["merged"]), P(wptr + tail_off), I(H), I(W)]))
                    plan["outputs"].append(("merged", (H // 2, W // 2, Co)))
        except Exception:
            self._release(plan)
            raise
        return plan

    def _execute(self, plan):
        for name, grid, threads, args in plan["launches"]:
            self.dev.launch(self._func(name), grid, threads, args)

    def _release(self, plan):
        for ptr in plan["bufs"].values():
            self.dev.free(ptr)
        plan["bufs"].clear()
