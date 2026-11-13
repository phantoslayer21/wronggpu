# SPDX-License-Identifier: MIT
"""A connected preset-F frame sequence on one resident D3D12 arena: every page of the frame, temporal state across
frames, resets and a render-size change, with the weights and symbols of a converted model file.

Per frame (one command-list submission per frame on the resident arena; after each frame the host writes a copy
of the arena for the comparisons, build.run checkpoints):

  PH2 quarter pool (54b.5, frames without reset; PH1 of the previous frame; exact tables built once per model,
  gpu/pool.py)
  -> exposure (58; the previous frame's E is read from its result buffer, reset frames read the cleared EX)
  -> input stage (52; reads PH1 / PH2 / HID / PM[par] / the exposure result; writes PM[1-par] / Pe / Pn in place)
  -> sixteen-stage body + D0-R head (51, 55; IN0 reads the input stage's x0 and seed)
  -> output stage (54; packed head outputs, E / Ep / invE from the exposure result; reads PM[1-par] / Pe / Pn /
     CH[par] as binary16 and widens them exactly (no widening pass); the fast path (gpu/output.py, prep / fast /
     fallback, bit-identical to rr_output_main) is the default; writes CH[1-par], Q and PH1)
  -> extended resolve (54b; transparency layer + alpha upscaling + separate output alpha; reads Q, PM[1-par],
     PREC, GATES, TH[par] / AH[par], the exposure result; writes the output colour / alpha and TH / AH[1-par]).

The host-side frame plan (parity, counter, reset, pj, CP, the render-size rebuild with zero-filled images) is the
lifecycle reference (reference/lifecycle.py). Every stage is compared with its CPU reference evaluated on the values
the GPU actually produced upstream in the same frame and on the temporal images the GPU left from the previous frame
(read from the previous checkpoint at the intended producer's buffer, never at the buffer a stage was bound to):
exposure within page 58's tolerance, PH2 / input stage / resolve bit-exact, body bit-exact with step X
and the head within page 55's tolerances (network.analyze), output stage under gpu/output.qualify's rules (exact
traces, CH / Q within two binary16 steps, PH1 within one code; the count of bit-identical values is reported).

    python -m clean.rr_f.gpu.sequence --json <report> [--model <model.wgcm> | --synthetic] [--mutation ...]
    python -m clean.rr_f.gpu.sequence --bench 1706x960:2560x1440 --json <report> [--model ...] [--step E|X]

--resolve fast|verified picks the resolve kernels (default fast: rr_resolve_fast + rr_resolve_fixup, bit-identical to
the verified rr_resolve_main). --mutation hid|pm|ch|exposure|ph1 rebinds one temporal input of frame 2 to the wrong buffer; the run must then fail
numerically (exit 1). --bench builds one frame without reset at a game size on synthetic temporal images (no references) and
times it (per dispatch, and the span from its first to its last dispatch) on the resident arena; the first,
untimed run leaves the frame's own outputs as the next run's temporal images.
"""
import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

import numpy as np

from . import build, stage
from . import exposure as GX
from . import input_stage as GI
from . import network as GN
from . import output as GO
from . import pool as GP
from . import resolve as GR
from . import resolve_fast as GRF
from ..head.block import recurrent_shift
from ..head.dx12 import run_dx12 as head
from ..model import model_file as MF
from ..reference import exposure as RX
from ..reference import input_stage as IS
from ..reference import lifecycle as LC
from ..reference import network as RN
from ..reference import output as RO
from ..reference import resolve as RR
from ..reference.scalar import WindowConsts
from ..reference.stages import stage_info, weight_shapes
from ..tests import synth, synth_input as SI

HEAD_KEYS = ("Wq", "Wv", "Tb", "Wo", "bo", "W1", "b1", "W2", "b2")
X_NAMES = ("X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn", "X_NRM")
OUT_K = ("K2n", "K2x2n", "K3", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13")
RES_F32 = ("K2n", "K2x2n", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13", "C_HDEC", "T_EFL", "TBW")
F_EXPO = 256          # rr_resolve.hlsl: E / Ep from the exposure result at word 15
OUT_FLAGS = 1 | 2 | 4 | 32  # rr_output.hlsl packed_head: bit 0 packed head, bit 1 exposure result link, bits 2 / 5
                            # binary16 motion / history inputs (read and widened by the output kernel: no widening pass)


# ------------------------------------------------------------------------------------------------------- model

@dataclass
class Model:
    source: str
    S: object             # reference.input_stage.InputSymbols
    IN0: dict             # the whole IN0 group (block tensors, FE.*, D, Wh, bh)
    weights: dict         # per network.ORDER name
    constants: dict       # ra, m, f, g per stage
    wc: WindowConsts
    SH: np.ndarray        # uint32 [2][11][2]
    s1: np.float32
    s2: np.float32
    xsym: object          # reference.exposure.ExposureSymbols
    osym: object          # reference.output.OutputConstants
    rsym: object          # reference.resolve.ResolveSymbols
    b2: bool
    digest: str = ""


def model_from_file(path, map_path=None):
    """The runtime's view of a converted model file (model/model_file.admit_model_file)."""
    adm = MF.admit_model_file(path, map_path=map_path)
    if not adm.player_ready:
        raise SystemExit(f"model file refused: {adm.issues}")
    m = adm.model
    sym = m.symbol
    c = {n: np.array([sym(n)], np.float32) for n in IS.SYMBOLS_F32}
    c.update({n: np.array([sym(n)], np.float16) for n in IS.SYMBOLS_F16})
    b2 = {n: np.array([sym(n)], np.float32) for n in IS.SYMBOLS_B2} if m.variant == "B2" else None
    S = IS.InputSymbols.from_groups(c, b2)
    IN0 = m.stage_weights_numpy("IN0")
    weights = {}
    for name in RN.ORDER:
        w = m.stage_weights_numpy(name)
        keys = HEAD_KEYS if name == "D0-R" else tuple(weight_shapes(name))
        weights[name] = {k: w[k] for k in keys}
    constants = {name: m.stage_constants(name) for name in RN.ORDER}
    wc = WindowConsts(**m.window_constants())
    SH = np.array(m.tensor("constants", "SH").values(), np.uint32).reshape(2, 11, 2)
    d0r = lambda n: np.float32(m.tensor("D0-R", n).values()[0])  # noqa: E731
    K1 = d0r("K1")
    TR = np.frombuffer(m.tensor("constants", "TR").data, "<f4").reshape(513, 4).copy()
    xsym = RX.ExposureSymbols(*(np.float32(sym(n)) for n in X_NAMES))
    osym = RO.OutputConstants(K1=K1, TR=TR, **{n: np.float32(sym(n)) for n in OUT_K})
    rsym = RR.ResolveSymbols(K1=K1, **{n: np.float16(sym(n)) for n in ("RW_A", "RW_B", "RW_C", "RW_D", "RW_0")},
                             **{n: np.float32(sym(n)) for n in RES_F32})
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Model(f"model file {path} (variant {m.variant})", S, IN0, weights, constants, wc, SH, d0r("s1"),
                 d0r("s2"), xsym, osym, rsym, m.variant == "B2", digest)


def model_synthetic(seed=5521, b2=False):
    """Own synthetic values with network-like magnitudes (tests/synth*.py), for a non-saturating run."""
    rng = np.random.default_rng(seed)
    S = SI.symbols(rng, b2=b2)
    IN0 = SI.in0_group(rng)
    weights = {n: synth.stage_weights(rng, "IN0" if n == "D0-R" else n) for n in RN.ORDER}
    weights["IN0"] = {k: IN0[k] for k in weight_shapes("IN0")}
    weights["D0-R"] = {k: weights["D0-R"][k] for k in HEAD_KEYS}
    constants = {n: synth.stage_consts(rng) for n in RN.ORDER}
    wc, _ = synth.window_consts(rng, "A")
    SH = SI.shift_table(rng)
    v = S.values
    table = np.tile(np.array([.125, .75, .125, .25], np.float32), (513, 1))
    table[:, 3] = np.linspace(0, .5, 513, dtype=np.float32)
    k = dict(K2n=np.float32(-1.25), K2x2n=np.float32(-0.5), K3=np.float32(0.5), K4=v["K4"], K5=v["K5"],
             K7=v["K7"], K8=v["K8"], K10=v["K10"], K11=np.float32(0.5), K12=v["K12"], K13=np.float32(2.0 ** -20))
    K1 = np.float32(1 / 16)
    osym = RO.OutputConstants(K1=K1, TR=table, **k)
    rk = {n: k[n] for n in RES_F32 if n in k}
    rsym = RR.ResolveSymbols(RW_A=-0.0625, RW_B=-0.03125, RW_C=0.375, RW_D=0.125, RW_0=0.625, K1=K1,
                             C_HDEC=v["C_HDEC"], T_EFL=np.float32(2.0 ** -10), TBW=np.float32(0.3125), **rk)
    return Model(f"own synthetic values (seed {seed})", S, IN0, weights, constants, wc, SH, np.float32(1.25),
                 np.float32(0.75), GX.symbols(), osym, rsym, b2)


# ------------------------------------------------------------------------------------------------------ helpers

def read(img, off, shape, dt):
    n = int(np.prod(shape)) * np.dtype(dt).itemsize
    return img[off:off + n].copy().view(dt).reshape(shape)


def guarded(arena, nbytes, data=None):
    """Zero-filled image (or data) with a 64-byte 0xA5 guard behind it (output.qualify checks the guard)."""
    body = np.zeros(nbytes, np.uint8) if data is None else np.ascontiguousarray(data).view(np.uint8).ravel()
    return arena.add(np.concatenate([body, np.full(64, 0xA5, np.uint8)]))


class ShapeArena(build.Arena):
    """Arena that does not store shape-only placeholder inputs (all their pointers are rebound to GPU outputs):
    while `shapes_only` is set, an upload of a registered placeholder array, or of the binary16 NaN pattern that
    a NaN placeholder becomes, returns one shared dummy offset."""

    def __init__(self):
        super().__init__()
        self.shapes_only, self.placeholders = False, set()
        self.dummy = super().reserve(256)

    def add(self, data, align=256):
        if self.shapes_only:
            if id(data) in self.placeholders:
                return self.dummy
            a = np.asarray(data)
            if a.dtype == np.dtype('<u2') and a.size >= 64 and np.all(a == 0x7E00):
                return self.dummy
        return super().add(data, align)


def bound(off, shape, fmt, ch, origin=(0, 0)):
    """A resolve resource descriptor for a buffer that already lives in the arena (no upload)."""
    u = GR.Upload.__new__(GR.Upload)
    h, w = shape
    u.shape, u.fmt, u.origin, u.packed = (h, w, ch), fmt, tuple(origin), False
    u.off, u.nbytes = off, h * w * ch * {"f32": 4, "f16": 2, "unorm8": 1, "rgb10a2": 2}[fmt]
    u.desc = [off, w, h, w, {"f32": 0, "f16": 1, "unorm8": 2, "rgb10a2": 4}[fmt], ch, origin[0], origin[1]]
    return u


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def jitter(i):
    """Jitter sequence in render pixels (exact binary32 values, never +-0.5)."""
    return (np.float32(np.float16(halton(i + 1, 2) - 0.5)), np.float32(np.float16(halton(i + 1, 3) - 0.5)))


@dataclass
class Expo:
    E: np.float32
    Ep: np.float32
    X3: np.float32


def shifts_of(SH, par):
    sh = np.asarray(SH)[par]
    shifts = {name: (int(sh[i][0]), int(sh[i][1])) for i, name in enumerate(RN.ENCODERS)}
    shifts.update({f"D{k}": (int(sh[6 + 4 - k][0]), int(sh[6 + 4 - k][1])) for k in range(5)})
    return shifts


def body_shapes(Wt, Ht):
    """Shape-only inputs of every stage call (the inputs are bound to earlier GPU outputs)."""
    out = {}
    for name in RN.ORDER[:-1]:
        i = stage_info(name)
        lv, C = i["level"], i["C"]
        Nx, Ny = Wt >> lv, Ht >> lv
        hold = lambda nx, ny: np.full((ny, nx, 4), np.nan)  # noqa: E731
        if i["kind"] == "fuse":
            out[name] = {"cin": np.zeros((Ny // 2, Nx // 2, i["Kc"]), np.uint8),
                         "skip": np.zeros((Ny, Nx, C), np.uint8)}
        elif i["adapter"]:
            out[name] = {"in": np.zeros((Ny, Nx, 128), np.uint8), "hold": hold(Nx, Ny)}
        else:
            out[name] = {"x": np.zeros((Ny, Nx, C), np.uint8), "hold": hold(Nx, Ny)}
            if i["kind"] == "rec":
                out[name]["Hc"] = hold(Nx // 2, Ny // 2)
    return out


# ------------------------------------------------------------------------------------------------------ builder

class Builder:
    """Host side of the sequence: one arena, all frames' dispatches, a checkpoint after every frame."""

    def __init__(self, model, output_size, first_render, *, step="X", zmode="exact", debug=True, seed=20261020,
                 input_exact=True, resolve_exact=True, steady=False, output_fast=True, resolve_kernel="fast",
                 pool_kernel="tables", head_mode="default", input_variant="exact"):
        """steady (timing only): the first frame is a frame without reset on synthetic temporal images.
        output_fast: the output stage's fast path (gpu/output.py dispatch_list), else rr_output_main.
        resolve_kernel: "fast" (rr_resolve_fast + rr_resolve_fixup, the default: the same bits as the verified kernel)
        or "verified" (rr_resolve_main); resolve_exact=False: both with the opt-in hardware elementary functions.
        pool_kernel: "tables" (gpu/pool.py rr_pool_main, the default) or "resolve-fast" (rr_resolve_pool_fast; needs
        the fast resolve and exact elementary functions); both bit-exact."""
        self.m, self.O, self.step, self.zmode, self.debug = model, tuple(output_size), step, zmode, debug
        self.head_mode = head_mode     # "default": step E head (55.3 tolerances); "exact": the exact head
        #                                (RR_STRICT_STEP 4 + weight planes + strict fixup, exact row sums: the runtime's)
        self.output_fast = output_fast
        self.steady = steady
        self.rng = np.random.default_rng(seed)
        self.ar = ShapeArena()
        self.kernels, self.disp, self.checkpoints, self.frames = {}, [], [], []
        self.lc = LC.Lifecycle(first_render, output_size, alpha_upscaling=True)
        self.sz = sz = RR.sizes(output_size)
        self.resolve_fast = resolve_kernel == "fast"
        self.pool_fast = (pool_kernel == "resolve-fast" and self.resolve_fast and resolve_exact)  # rr_resolve_pool_fast
        self.fixup_bitmap = GRF.fixup_bitmap(self.ar, sz) if self.resolve_fast else 0
        # the exp2 table of the fast resolve and pool: filled once per arena by rr_rf_init_table, before the first frame
        self.exp2_table = GRF.exp2_table(self.ar, self.disp, self.kernels) if self.resolve_fast else 0
        (self.h1, self.w1), (self.h2, self.w2) = RR.pooled_sizes(output_size)
        ar = self.ar
        rng = self.rng
        def codes(h, w):
            c = rng.integers(0, 1024, (h, w, 4)).astype(np.uint16)
            c[..., 3] = 3
            return c

        def hist(ch):
            shape = (sz.He, sz.We, ch)
            return rng.uniform(0, 1, shape).astype(np.float16) if steady else np.zeros(shape, np.float16)

        # creation: zero-filled (53); steady: synthetic history content
        self.PH1 = guarded(ar, self.h1 * self.w1 * 8, codes(self.h1, self.w1) if steady else None)
        self.PH2 = ar.add(codes(self.h2, self.w2)) if steady else ar.reserve(self.h2 * self.w2 * 8)
        self.TH = [GR.Upload(ar, hist(4), "f16") for _ in range(2)]
        self.AH = [GR.Upload(ar, hist(1), "f16") for _ in range(2)]
        self.EX1 = ar.add(np.array([1, 1, 1, 0], np.float32))  # the cleared EX (CLEAR_EX on reset frames)
        self.prev_expo = ar.add(np.array([1.25, 1, 0.8, 0], np.float32)) if steady else None
        self.prev_head = None                                 # (render size, hid offset) of the previous frame
        self.th_live = steady
        self.ah_created = not steady                          # AH exists from creation (alpha upscaling)
        if steady:
            self.lc.pending_reset_generation = None           # timing only: a frame of a running sequence
        self.render = None
        self.camera = SI.camera_kind(self.rng, "perspective")[:2]
        self.input_variant = input_variant          # opt-in near-exact features kernel (bench only), gpu/input_stage.py
        self._kernels(input_exact, resolve_exact)
        # per-model setup (once, before the first frame): the exact tables of the PH2 pool (gpu/pool.py)
        self.pool_tab, td = GP.table_dispatch(ar, model.rsym)
        self.disp.append(td)
        self.setup = len(self.disp)

    def _kernels(self, input_exact, resolve_exact):
        k = self.kernels
        for n in ("partial", "finish"):
            k[f"exposure-{n}"] = (build.compile_kernel("rr_exposure.hlsl", "rr_exposure_" + n, {"WG_BLOCK": GX.EX_BLOCK[n]}), 23)
        k.update(GI.kernels("X" if self.step in ("Y", "Z", "Z2", "I") else self.step))   # the input MLP keeps its own step
        if not input_exact:
            k["input-features"] = (build.compile_kernel("rr_input.hlsl", "rr_input_features",
                                                        {"WG_BLOCK": GI.FEAT_BLOCK, "RR_IN_FAST": 1, "RR_IN_EXACT": 0,
                                                         "RR_IN_LDST": 1}, tag="rr_input_features_fast1"), 1)
        if self.input_variant != "exact":   # opt-in near-exact features (tier B): bench only
            k["input-features"] = (GI.features_kernel(self.input_variant), 1)
        k.update(GP.kernels(resolve_exact))
        k.update({"output-" + n: v for n, v in GO.kernel_set(fast=self.output_fast).items()})
        e = int(resolve_exact)
        if self.pool_fast:
            k["resolve-pool-fast"] = (GRF.pool_fast_kernel(), GRF.NARGS)
        if self.resolve_fast:          # resolve_exact False: the opt-in hardware elementary functions (RF_HW)
            k["resolve-fast"] = (GRF.fast_kernel(spec=int(os.environ.get("WG_RR_RF_SPEC", "1")), extra=None if resolve_exact else {"RF_HW": 1}), GRF.NARGS)
            k["resolve-fixup"] = (GRF.fixup_kernel(hw=not resolve_exact), GRF.NARGS)
        for n in ("main", "pool"):
            k["resolve-" + n] = (build.compile_kernel("rr_resolve.hlsl", "rr_resolve_" + n,
                                                      {"WG_BLOCK": GR.BLOCK, "RR_ELEM_EXACT": e, "RR_MUT": 0},
                                                      tag=f"rr_resolve_{n}_e{e}m0"), GR.NARGS)
        stage.precompile([(n, self.zmode, self.step, 1) for n in RN.ORDER[:-1]])
        for n in RN.ORDER[:-1]:
            k[n] = (stage.kernel_for(n, self.zmode, step=self.step, ldsw=1), None)
        k["regroup"] = (build.compile_kernel("rr_body_helpers.hlsl", "rr_regroup_e1", {"WG_BLOCK": 256}), 4)
        variant = "dbg" if self.debug else "prod"
        src = Path(head.HERE) / "rr_d0r.hlsl"
        srcs = src.read_bytes() + (Path(head.CLEAN) / "rr_f/kernels/rr_step_i.hlsli").read_bytes()

        def head_dxil(name, entry, extra):
            defs = {"WG_BLOCK": 128, "WG_WAVE": 32, **head.VARIANTS[variant], **extra}
            key = hashlib.sha256(srcs + json.dumps([sorted(defs.items()), entry]).encode()).hexdigest()[:16]
            dxil = str(Path(build.BUILD) / "dxil" / f"seq_{name}_{variant}_{key}.dxil")
            if not os.path.exists(dxil):          # cached by source + defines (parallel runs share the build)
                build.build_dx12.compile_hlsl(str(src), entry, dxil, defs,
                                             includes=(str(Path(head.CLEAN) / "dx12/kernels"),),
                                             asm_tool=build.build_dx12.build_asm_tool(build.BUILD))
            return dxil
        self.head_variant = variant
        if self.head_mode == "exact":
            ex = {"RR_ZEXACT": 1, "RR_STRICT_STEP": 4}
            k["D0-R-W"] = (head_dxil("d0rw", "rr_d0r_wplanes", ex), None)
            k["D0-R"] = (head_dxil("d0ri", "rr_d0r_head", ex), None)
            k["D0-R-F"] = (head_dxil("d0rf", "rr_d0r_head", {"RR_ZEXACT": 1, "RR_STRICT_STEP": 1, "RR_FIXUP": 1}),
                           None)
        else:
            k["D0-R"] = (head_dxil("d0r", "rr_d0r_head", {}), None)

    # -- render-size images (53: recreated zero-filled on every render-size change)
    def _rebuild(self, wr, hr):
        ar, sz = self.ar, self.sz
        self.render = (wr, hr)
        if self.steady and not self.frames:
            rng = self.rng
            mv = lambda c: rng.uniform(-3, 3, (hr, wr, c)).astype(np.float16)  # noqa: E731
            self.PM = [ar.add(mv(4)), ar.add(mv(4))]
            self.Pe, self.Pn = ar.add(mv(2)), ar.add(mv(2))
            self.CH = [guarded(ar, 0, rng.uniform(0, 2, (sz.Hh, sz.Wh, 4)).astype(np.float16)) for _ in range(2)]
            self.HID0 = ar.add(rng.integers(-128, 128, (hr, wr, 4)).astype(np.int8))
        else:
            self.PM = [ar.reserve(hr * wr * 8), ar.reserve(hr * wr * 8)]
            self.Pe, self.Pn = ar.reserve(hr * wr * 4), ar.reserve(hr * wr * 4)
            self.CH = [guarded(ar, sz.Hh * sz.Wh * 8), guarded(ar, sz.Hh * sz.Wh * 8)]
            self.HID0 = ar.reserve(hr * wr * 4)
        self.prev_head = None

    def _body(self, call, shifts, render, Wt, Ht):
        """The sixteen stage calls and the head, inputs bound to the input stage's x0 / seed and to each other."""
        m, ar = self.m, self.ar
        shapes = body_shapes(Wt, Ht)
        calls = {}
        regroup = None
        for name in RN.ORDER[:-1]:
            i = stage_info(name)
            level = i["level"]
            links = {}
            if name == "IN0":
                links = call.in0_links()
                shift = shifts[name]
            elif name.startswith("E"):
                parent = "IN0" if level == 1 else f"E{level - 1}"
                links[2] = calls[parent].out["hmerged"][0]
                if level == 1 and stage.e1_direct(step=self.step):
                    links[0] = calls["IN0"].out["y2"][0]       # E1 reads IN0's y2 directly (no regroup)
                elif level == 1:
                    regroup = ar.reserve(Wt * Ht * 32, fill=0xEE)
                    self.disp.append(("regroup", (Wt * Ht * 2 + 255) // 256,
                                      [calls["IN0"].out["y2"][0], regroup, Wt, Ht]))
                    links[0] = regroup
                else:
                    links[0] = calls[parent].out["merged"][0]
                shift = shifts[name]
            elif name.endswith("-F"):
                parent = "E5" if level == 4 else f"D{level + 1}-R"
                enc = "IN0" if level == 0 else f"E{level}"
                links = {0: calls[enc].out["y2"][0], 1: calls[parent].out["y2"][0]}
                shift = shifts[f"D{level}"]
            else:
                parent = "E5" if level == 4 else f"D{level + 1}-R"
                links = {0: calls[f"D{level}-F"].out["y2"][0], 2: calls[f"E{level}"].out["hnew"][0],
                         3: calls[parent].out["hnew"][0]}
                shift = shifts[f"D{level}"]
            ar.placeholders = {id(v) for v in shapes[name].values()}
            ar.shapes_only = True
            try:
                sc = stage.StageCall(name, shapes[name], m.weights[name], m.constants[name], m.wc, shift, arena=ar,
                                     step=self.step)
            finally:
                ar.shapes_only = False
            for index, pointer in links.items():
                sc.args[index] = pointer
            calls[name] = sc
            for kid, kern, groups, args in stage.dispatches_for(sc, self.zmode, self.step, 1, kid=name):
                self.kernels[kid] = kern                     # variant Y: the Y dispatch and its fixup
                self.disp.append((kid, groups, args))
        c = m.constants["D0-R"]
        wr, hr = render
        params = dict(Wt=Wt, Ht=Ht, Wr=wr, Hr=hr, shift=recurrent_shift(shifts["D0"]),
                      constants={**m.wc.as_dict(), "ra": c["ra"], "m": c["m"]}, g=c["g"], s1=m.s1, s2=m.s2)
        ha, job = head.make_job(params, "stage", self.head_variant, W=m.weights["D0-R"],
                                x=np.zeros((Ht, Wt, 32), np.uint8), hold=np.zeros((Ht, Wt, 4)),
                                hc=np.zeros((Ht // 2, Wt // 2, 4)))
        init = np.zeros(ha.size, np.uint8)
        for offset, data in ha.init:
            init[offset:offset + len(data)] = np.frombuffer(data, np.uint8)
        base = ar.add(init)
        nargs = struct.unpack_from("<I", job, 8)[0]
        args = list(struct.unpack_from(f"<{nargs}I", job, 12))
        for index, name in enumerate(head.ARGS):
            if name.startswith("p_") and args[index]:
                args[index] += base
        args[0] = calls["D0-F"].out["y2"][0]
        args[4] = calls["IN0"].out["hnew"][0]
        args[5] = calls["D1-R"].out["hnew"][0]
        self.kernels["D0-R"] = (self.kernels["D0-R"][0], nargs)
        groups = ((Wt + params["shift"][0] + 7) // 8) * ((Ht + params["shift"][1] + 7) // 8)
        if self.head_mode == "exact":
            for n in ("D0-R-W", "D0-R-F"):
                self.kernels[n] = (self.kernels[n][0], nargs)
            self.disp.append(("D0-R-W", 8, args))   # 32 weight pairs (28 + the 4 merged W2 pairs), 4 per group
            self.disp.append(("D0-R", groups, args))
            self.disp.append(("D0-R-F", groups, args))
        else:
            self.disp.append(("D0-R", groups, args))
        return calls, base, ha, regroup

    def add_frame(self, render, *, game_reset=False, mutation=None):
        m, ar, sz, idx = self.m, self.ar, self.sz, len(self.frames)
        wr, hr = render
        Wt, Ht = 32 * -(-wr // 32), 32 * -(-hr // 32)
        V2C, W2V = self.camera
        W2V = W2V.copy()
        W2V[3, :3] += np.float32(0.05 * idx)                   # the camera moves a little every frame
        camera = LC.CameraPair(tuple(np.asarray(V2C, np.float32).ravel()), tuple(np.asarray(W2V, np.float32).ravel()))
        plan = self.lc.begin_frame(render, jitter=jitter(idx), camera=camera, game_reset=game_reset,
                                   layer_attached=True)
        if plan.rebuild is not None or self.render is None:
            self._rebuild(wr, hr)
        par, reset = plan.parity, bool(plan.reset)
        start = len(self.disp)
        rec = dict(index=idx, render=render, Wt=Wt, Ht=Ht, plan=plan, par=par, reset=reset, game_reset=game_reset,
                   rebuild=plan.rebuild is not None or idx == 0, start=start, mutation=mutation)
        mut = lambda name: mutation == name and idx == 2       # noqa: E731
        # ---- PH2 pass (54b.5) at the start of a frame without reset
        if not reset:
            ph1_src = self.PH1
            pp = GR.params(m.rsym, sizes=sz, render=(1, 1), flags=0,
                           res=dict(PH1=bound(ph1_src, (self.h1, self.w1), "rgb10a2", 4),
                                    PQ=bound(self.PH2, (self.h2, self.w2), "rgb10a2", 4)))
            if self.pool_fast:
                pp[38] = self.exp2_table
                self.disp.append(("resolve-pool-fast", (self.w2 * self.h2 + 255) // 256, [ar.add(pp), 0, 0]))
            else:
                self.disp.append(GP.pool_dispatch(ar.add(pp), self.pool_tab, self.w2, self.h2))
        rec["pooled"] = not reset
        # ---- synthetic game inputs of this frame
        frame, _, _, _, _, _ = SI.make_case(self.rng, Wr=wr, Hr=hr, Wo=self.O[0], Ho=self.O[1], smv="Hd",
                                            guides=["B", "G1", "Q"], b2=m.b2, camera="perspective")
        rec["textures"] = frame.textures
        # ---- exposure (58)
        C = frame.textures["C"]
        c = np.ascontiguousarray(C.data, np.float32)
        blocks = (wr * hr + 255) // 256
        expo = ar.reserve(100, fill=0xA5)
        prev = self.EX1 if (reset or self.prev_expo is None) else self.prev_expo
        rec["expo_prev_intended"] = prev
        if mut("exposure"):
            prev = self.EX1
        ilo, ihi = struct.unpack("<II", struct.pack("<d", 1.0 / (wr * hr)))
        xargs = [ar.add(c), ar.reserve(blocks * 8), expo, c.shape[1], c.shape[2], *C.origin, wr, hr, blocks, ilo,
                 ihi, prev, 2 | int(reset)]
        xargs += [int(np.float32(getattr(m.xsym, f.name)).view(np.uint32)) for f in fields(m.xsym)]
        self.disp += [("exposure-partial", blocks, xargs), ("exposure-finish", 1, xargs)]
        rec["expo"], rec["expo_prev"] = expo, prev
        # ---- input stage (52)
        hid_intended = self.prev_head[1] if self.prev_head is not None else self.HID0
        links = dict(HID=hid_intended, PMR=self.PM[par], PMW=self.PM[1 - par], PE=self.Pe, PN=self.Pn,
                     PH1=self.PH1, PH2=self.PH2)
        rec["intended"] = dict(links)
        if mut("hid"):
            links["HID"] = self.HID0 if self.prev_head is not None else ar.reserve(hr * wr * 4)
        if mut("pm"):
            links["PMR"] = self.PM[1 - par]
        if mut("ph1"):
            links["PH1"] = ar.reserve(self.h1 * self.w1 * 8)
        rec["links"] = links
        z = lambda *s: np.zeros(s, np.float16)                  # noqa: E731
        frame.PH1 = np.zeros((self.h1, self.w1, 4), np.uint16)
        frame.PH2 = np.zeros((self.h2, self.w2, 4), np.uint16)
        frame.HID = np.zeros((hr, wr, 4), np.int8)
        frame.PM_read, frame.PM_write, frame.Pe, frame.Pn = z(hr, wr, 4), z(hr, wr, 4), z(hr, wr, 2), z(hr, wr, 2)
        mv_scale = (np.float32(1.0 / wr), np.float32(-1.0 / hr))
        params = IS.params_from_plan(plan, exposure=Expo(np.float32(1), np.float32(1), np.float32(0)), SH=m.SH,
                                     mv_scale_raw=mv_scale)
        rec["frame"], rec["params"] = frame, params
        call = GI.InputCall(ar, frame, params, m.S, m.IN0, m.wc, debug=self.debug, diag=self.debug,
                            exposure=expo, links=links, fold=stage.infold(self.step))
        self.disp += call.dispatches()
        rec["call"] = call
        # ---- body + head (51, 55)
        shifts = shifts_of(m.SH, par)
        calls, base, ha, regroup = self._body(call, shifts, render, Wt, Ht)
        self.last_head = (base, ha, Wt, Ht)
        rec.update(shifts=shifts, calls=calls, base=base, ha=ha, regroup=regroup)
        hp = lambda key: base + ha.ptr[key]                    # noqa: E731
        # ---- the output kernel reads PM[1-par] / Pe / Pn / CH[par] as binary16 and widens them exactly (OUT_FLAGS bits 2, 5)
        ch_read = self.CH[1 - par] if mut("ch") else self.CH[par]
        rec["ch_read_intended"], rec["ch_write"] = self.CH[par], self.CH[1 - par]
        w = dict(PM=self.PM[1 - par], Pe=self.Pe, Pn=self.Pn, Hprev=ch_read)
        # ---- output stage (54)
        a = dict(Cnet=np.zeros((Ht, Wt, 4)), G=np.zeros((hr, wr, 3)), P=np.zeros((hr, wr, 3)),
                 W4=np.zeros((hr, wr, 4)), PM=np.zeros((hr, wr, 4)), Pe=np.zeros((hr, wr, 2)),
                 Pn=np.zeros((hr, wr, 2)), Hprev=np.zeros((sz.Hh, sz.Wh, 4)), output_size=self.O,
                 jitter=plan.current_jitter, E=np.float32(1), Ep=np.float32(1), invE=np.float32(1), reset=reset,
                 constants=m.osym)
        binds = dict(Cnet=hp("cnet"), G=hp("gates"), P=hp("prec"), W4=hp("candw"), PM=w["PM"], Pe=w["Pe"],
                     Pn=w["Pn"], Hprev=w["Hprev"])
        oargs, offs, sizes = GO.prepare(ar, a, packed=True, input_bindings=binds)
        oargs[10], oargs[12] = self.CH[1 - par], self.PH1
        offs = list(offs)
        offs[1], offs[3] = self.CH[1 - par], self.PH1
        oargs[21] = expo
        # bits 2 / 5 binary16 motion / history (OUT_FLAGS, both paths); fast path bit 3 (timing builds, debug=False):
        # no linear / trace outputs
        oargs[37] = OUT_FLAGS | ((0 if self.debug else 8) if self.output_fast else 0)
        self.disp += GO.dispatch_list(oargs, fast=self.output_fast, prefix="output-")
        rec.update(out_args=oargs, out_offs=offs, out_sizes=sizes, out_binds=binds)
        # ---- resolve (54b): layer + alpha upscaling + separate output alpha
        layer = RR.Surface(self.rng.uniform(0, 1, (hr + 1, wr + 2, 4)).astype(np.float16), "f16", (1, 1))
        codes = self.rng.integers(0, 1024, (hr, wr, 4)).astype(np.uint16)
        codes[..., 3] &= 3
        colour_in = RR.Surface(codes, "rgb10a2")
        dest = RR.Surface(np.zeros((sz.He, sz.We, 4), np.float16), "f16")
        aout = RR.Surface(np.zeros((sz.He, sz.We, 1), np.float32), "f32")
        created = not self.th_live
        flags = GR.F_EXT | GR.F_LAYER | GR.F_ALPHA | GR.F_AOUT | F_EXPO
        if reset or created or self.ah_created:
            flags |= GR.F_TRESET
        if reset or self.ah_created:
            flags |= GR.F_ARESET
        ups = dict(Q=bound(offs[2], (sz.Hh, sz.Wh), "f16", 3), OUT=GR.surface_upload(ar, dest),
                   AOUT=GR.surface_upload(ar, aout), LAYER=GR.surface_upload(ar, layer),
                   ASRC=GR.surface_upload(ar, colour_in), MV=bound(self.PM[1 - par], (hr, wr), "f16", 4),
                   P=bound(hp("prec"), (hr, wr), "f16", 4), G=bound(hp("gates"), (hr, wr), "unorm8", 4),
                   THr=self.TH[par], THw=self.TH[1 - par], AHr=self.AH[par], AHw=self.AH[1 - par])
        p = GR.params(m.rsym, sizes=sz, render=(wr, hr), flags=flags, jitter=plan.current_jitter,
                      layer_origin=layer.origin, alpha_channel=3, res=ups)
        p[15] = expo
        if self.resolve_fast:
            p[37], p[38] = self.fixup_bitmap, self.exp2_table
            po = ar.add(p)
            self.disp += [("resolve-fast", GRF.fast_groups(sz), [po, 0, 0]),
                          ("resolve-fixup", GRF.fixup_groups(sz), [po, 0, 0])]
        else:
            self.disp.append(("resolve-main", (sz.We * sz.He + GR.BLOCK - 1) // GR.BLOCK, [ar.add(p), 0, 0]))
        rec.update(layer=layer, colour_in=colour_in, dest=dest, aout=aout, ups=ups, flags=flags,
                   th_live_before=self.th_live, ah_created_before=self.ah_created)
        self.th_live, self.ah_created = True, False
        # ---- lifecycle: every pass acknowledged, frame completed (the GPU run is checked afterwards)
        for ps in plan.passes:
            if ps == LC.Pass.INPUT:
                self.lc.input_consumed_jitter(plan)
            self.lc.acknowledge_pass(plan, ps)
        self.lc.complete_frame(plan)
        rec["hid_prev"] = self.prev_head[1] if self.prev_head is not None else None
        self.prev_expo = expo
        self.prev_head = (render, hp("hid"))
        rec["end"] = len(self.disp)
        rec["dispatch_names"] = [d[0] for d in self.disp[start:]]
        self.checkpoints.append(len(self.disp))
        self.frames.append(rec)
        return rec


# ------------------------------------------------------------------------------------------------------ analysis

def head_outputs(img, rec):
    wr, hr = rec["render"]
    hp = lambda key: rec["base"] + rec["ha"].ptr[key]        # noqa: E731
    return dict(Cnet=read(img, hp("cnet"), (rec["Ht"], rec["Wt"], 4), "<f2"),
                GATES_code=read(img, hp("gates"), (hr, wr, 4), np.uint8),
                PREC=read(img, hp("prec"), (hr, wr, 4), "<f2"),
                CANDW_code=read(img, hp("candw"), (hr, wr, 4), np.uint8),
                HID_code=read(img, hp("hid"), (hr, wr, 4), np.int8))


def same(a, b):
    return GR.same_bits(np.ascontiguousarray(a), np.ascontiguousarray(b))


def analyze_frame(b, rec, img, prev):
    """All comparisons of one frame: img = arena after the frame, prev = arena before it (previous checkpoint)."""
    m, sz, O = b.m, b.sz, b.O
    wr, hr = rec["render"]
    par, reset = rec["par"], rec["reset"]
    it = rec["intended"]
    out = dict(frame=rec["index"], render=rec["render"], parity=par, counter=rec["plan"].counter, reset=reset,
               game_reset=rec["game_reset"], rebuild=rec["rebuild"], dispatches=rec["dispatch_names"])
    stages = {}
    t0 = time.time()
    # exposure
    C = rec["textures"]["C"]
    c = np.ascontiguousarray(C.data, np.float32)
    e = read(img, rec["expo"], (4,), "<f4")
    intended_prev = rec["expo_prev_intended"]
    Eprev = np.float32(read(prev, intended_prev, (1,), "<f4")[0])
    expected = RX.auto_exposure(c, symbols=m.xsym, render_size=(wr, hr), Eprev=Eprev, reset=reset, origin=C.origin)
    xrow = GX.qualify(img, [("exposure", rec["expo"], expected, GX.exact_mean(c, (wr, hr), C.origin, m.xsym), reset,
                             m.xsym)], m.xsym)[0]
    xexact = bool(np.array_equal(e.view(np.uint32), np.float32([expected.E, expected.Ep, expected.invE,
                                                                expected.X3]).view(np.uint32)))
    stages["exposure"] = dict(passed=xrow["passed"], exact=xexact, rule="page 58 tolerance", rules=xrow["rules"],
                              E=float(e[0]), Ep=float(e[1]), X3=float(e[3]))
    E, Ep, invE, X3 = (np.float32(v) for v in e)
    # PH2 quarter pool
    PH1_prev = read(prev, it["PH1"], (b.h1, b.w1, 4), "<u2")
    PH2_now = read(img, b.PH2, (b.h2, b.w2, 4), "<u2")
    if rec["pooled"]:
        want = RR.quarter_pool(PH1_prev, read(prev, b.PH2, (b.h2, b.w2, 4), "<u2"), output_size=O, reset=False,
                               symbols=m.rsym)
        ok, bad = same(PH2_now, want)
        stages["ph2_pool"] = dict(passed=ok, exact=ok, values=int(want.size), mismatches=bad)
    # input stage
    frame = rec["frame"]
    frame.PH1, frame.PH2 = PH1_prev, PH2_now
    frame.HID = read(prev, it["HID"], (hr, wr, 4), np.int8)
    frame.PM_read = read(prev, it["PMR"], (hr, wr, 4), "<f2")
    frame.PM_write = read(prev, it["PMW"], (hr, wr, 4), "<f2")
    frame.Pe, frame.Pn = read(prev, it["PE"], (hr, wr, 2), "<f2"), read(prev, it["PN"], (hr, wr, 2), "<f2")
    params = rec["params"]
    params.E, params.Ep, params.X3 = E, Ep, X3
    ref_in = IS.input_stage(frame, params, m.S, m.IN0, m.constants["IN0"], m.wc)
    rin = GI.compare(rec["call"].outputs(img), ref_in)
    stages["input"] = dict(passed=rin["exact"], exact=rin["exact"], m2_src=str(ref_in.get("m2_src")),
                           **{k: v for k, v in rin.items() if isinstance(v, dict)},
                           **{k: v for k, v in rin.items() if k in ("ziv_unproven", "replay_fallbacks",
                                                                       "replayed_tokens", "mlp_exact5_wave_steps")})
    # body + head
    reference = RN.run_body(ref_in["x0"], ref_in["hold"], m.weights, m.constants, m.wc, rec["shifts"], s1=m.s1,
                            s2=m.s2, render_size=(wr, hr))
    body = GN.analyze(img, rec["calls"], reference, rec["base"], rec["ha"], rec["regroup"], render=(wr, hr))
    stage_exact = all(v["diff"] == 0 for row in body["stages"].values() for v in row.values())
    stages["body"] = dict(passed=stage_exact and not body["regroup_mismatches"], exact=stage_exact,
                          rule="bit-exact (step X)", per_stage={n: {k: v["diff"] for k, v in row.items()}
                                                                for n, row in body["stages"].items()},
                          regroup_mismatches=body["regroup_mismatches"])
    head_exact = all(v.get("exact", v["values"]) == v["values"] for v in body["head"].values()
                     if isinstance(v, dict) and "values" in v)
    stages["head"] = dict(passed=body["passed"], exact=head_exact, rule="page 55 tolerances (network.analyze)",
                          summary=body["head"], store_mismatches=body["head_store_mismatches"],
                          guards=body["head_guards"])
    hg = head_outputs(img, rec)
    finite = {k: float(np.isfinite(v.astype(np.float64)).mean()) for k, v in hg.items() if v.dtype.kind == "f"}
    finite["y2_IN0_nonzero"] = float((rec["calls"]["IN0"].outputs(img)["y2"] & 0x7F).astype(bool).mean())
    # (no widening pass: the output kernel reads the binary16 images, OUT_FLAGS bits 2, 5)
    # output stage: the intended inputs (head outputs, this frame's PM[1-par] / Pe / Pn, the previous CH[par])
    a = dict(Cnet=hg["Cnet"].astype(np.float64), G=RO.decode_unorm8(hg["GATES_code"][..., :3]),
             P=hg["PREC"][..., :3].astype(np.float64), W4=RO.decode_unorm8(hg["CANDW_code"]),
             PM=read(img, it["PMW"], (hr, wr, 4), "<f2").astype(np.float64),
             Pe=read(img, it["PE"], (hr, wr, 2), "<f2").astype(np.float64),
             Pn=read(img, it["PN"], (hr, wr, 2), "<f2").astype(np.float64),
             Hprev=read(prev, rec["ch_read_intended"], (sz.Hh, sz.Wh, 4), "<f2").astype(np.float64),
             output_size=O, jitter=tuple(np.float32(v) for v in rec["plan"].current_jitter), E=E, Ep=Ep, invE=invE,
             reset=reset, constants=m.osym)
    expected, trace = GO.oracle_with_trace(a)
    orow, _ = GO.qualify(img, [("output", a, rec["out_offs"], rec["out_sizes"], expected, trace)], reset_pairs=())
    orow = orow[0]
    lin, ch, q, ph, otrace = GO.read_outputs(img, a, rec["out_offs"], rec["out_sizes"])
    # digests of every output-stage buffer (comparison of runs, e.g. --output-path fast / main)
    out["output_sha256"] = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16] for k, v in
                            (("linear", lin), ("CH", ch), ("Q", q), ("PH1", ph), ("trace", otrace))}
    exact_counts = {}
    for key, got, want in (("CH", ch, expected.CH), ("Q", q, expected.Q), ("PH1", ph, expected.PH1)):
        okb, bad = same(got, np.asarray(want, got.dtype))
        exact_counts[key] = dict(values=int(got.size), not_bit_identical=bad)
    stages["output"] = dict(passed=orow["passed"], exact=all(v["not_bit_identical"] == 0 for v in exact_counts.values()),
                            rule="gpu/output.qualify (exact traces; CH/Q 2 binary16 steps; PH1 1 code)",
                            comparison=orow["comparison"], exact_rules=orow["exact"], bit_identical=exact_counts)
    finite["Q"] = float(np.isfinite(q.astype(np.float64)).mean())
    # resolve
    Qg = q.astype(np.float64)
    H = RR.ResolveHistories(O, alpha_upscaling=True,
                            TH=[b.TH[i].read(prev) for i in range(2)] if rec["th_live_before"] else None,
                            AH=[b.AH[i].read(prev) for i in range(2)], ah_created=rec["ah_created_before"])
    res = RR.resolve(Qg, output_size=O, render_size=(wr, hr), E=E, Ep=Ep,
                     jitter=tuple(np.float32(v) for v in rec["plan"].current_jitter), reset=reset, parity=par,
                     symbols=m.rsym, colour_out=rec["dest"], histories=H, alpha_out=rec["aout"], layer=rec["layer"],
                     colour_in=rec["colour_in"], motion=a["PM"], P=a["P"], G=hg["GATES_code"][..., :3])
    rchecks = {}
    for label, up, want in (("colour", rec["ups"]["OUT"], res.colour.data), ("alpha_out", rec["ups"]["AOUT"], res.alpha.data),
                            ("TH[1-par]", b.TH[1 - par], res.histories.TH[1 - par]),
                            ("AH[1-par]", b.AH[1 - par], res.histories.AH[1 - par])):
        ok, bad = same(up.read(img), want)
        rchecks[label] = dict(values=int(np.asarray(want).size), mismatches=bad,
                              sha256=hashlib.sha256(np.ascontiguousarray(up.read(img)).tobytes()).hexdigest()[:16])
    rok = all(v["mismatches"] == 0 for v in rchecks.values())
    stages["resolve"] = dict(passed=rok, exact=rok, variant=res.variant, flags=rec["flags"], checks=rchecks)
    finite["colour_out"] = float(np.isfinite(rec["ups"]["OUT"].read(img).astype(np.float64)).mean())
    # how much the values vary (a saturated network gives few distinct values)
    distinct = {k: int(np.unique(v.reshape(-1, v.shape[-1]), axis=0).shape[0]) for k, v in
                (('x0', ref_in['x0'].reshape(-1, 32)[:, :, None]), ('HID', hg['HID_code']), ('GATES', hg['GATES_code']),
                 ('PREC', hg['PREC']), ('CANDW', hg['CANDW_code']), ('Q', q), ('PH1', ph),
                 ('colour_out', rec['ups']['OUT'].read(img)))}
    out["distinct_values"] = distinct
    out["stages"] = stages
    out["finite_fraction"] = finite
    out["passed"] = all(s["passed"] for s in stages.values())
    out["analysis_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------------------------------------------ main

def load_model(a):
    if a.synthetic:
        return model_synthetic(b2=a.b2)
    return model_from_file(a.model, a.map or None)      # --map "": no map check (e.g. the F-B1 game model)


DEFAULT_MODEL = os.environ.get("WG_RR_MODEL", "")   # a synthetic model file (clean/rr_f/runtime/tools/pack_model.py)
DEFAULT_MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "rr-f-synth", "synth-r0.map.json")


def sequence(a):
    m = load_model(a)
    O = tuple(int(v) for v in a.output.split("x"))
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.renders.split(",")]
    plan = [(sizes[0], False), (sizes[0], False), (sizes[0], False), (sizes[0], True), (sizes[0], False),
            (sizes[1], False), (sizes[1], False), (sizes[1], False), (sizes[0], False), (sizes[0], False)][:a.frames]
    t0 = time.time()
    b = Builder(m, O, sizes[0], step=a.step, zmode="exact", output_fast=a.output_path == "fast",
                resolve_kernel=a.resolve, pool_kernel=a.pool, head_mode=a.head)
    for render, game_reset in plan:
        b.add_frame(render, game_reset=game_reset, mutation=a.mutation)
    print(f"Prepared {len(b.frames)} frames, {len(b.disp)} dispatches, arena {b.ar.size / 1e6:.1f} MB "
          f"({time.time() - t0:.0f} s)", flush=True)
    wd = str(Path(build.BUILD) / ("seq-" + (a.tag or ((a.mutation or "normal") + ("-synth" if a.synthetic else "")))))
    initial = b.ar.image()
    img, log = build.run(b.ar, b.kernels, b.disp, require_double=True, workdir=wd, checkpoints=b.checkpoints)
    paths = list(build.run.last_checkpoints)
    rows = []
    prev = initial
    dump = {}
    for rec, path in zip(b.frames, paths):
        cur = np.fromfile(path, np.uint8)
        if a.dump_body:                                  # raw body outputs, for byte comparisons between steps
            for name, sc in rec["calls"].items():
                for key, arr in sc.outputs(cur).items():
                    dump[f"f{rec['index']}_{name}_{key}"] = arr
                dump[f"f{rec['index']}_{name}_fixups"] = np.array([sc.fixups(cur)])
            for key, arr in head_outputs(cur, rec).items():   # the head's outputs (GPU-vs-GPU head compares)
                dump[f"f{rec['index']}_head_{key}"] = np.ascontiguousarray(arr)
        r = analyze_frame(b, rec, cur, prev)
        rows.append(r)
        st = r["stages"]
        print(f"frame {r['frame']}: render {r['render']} par {r['parity']} reset {r['reset']} "
              + " ".join(f"{k}={'ok' if v['passed'] else 'FAIL'}{'' if v['exact'] else '~'}" for k, v in st.items())
              + f" ({r['analysis_s']} s)", flush=True)
        prev = cur
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    if a.dump_body:
        np.savez_compressed(a.dump_body, **dump)
    passed = all(r["passed"] for r in rows)
    report = dict(passed=passed, mutation=a.mutation, resolve=a.resolve, model=m.source, model_sha256=m.digest, output_size=O,
                  renders=sizes, step=a.step, frames=rows, host_log=log,
                  shaders={n: hashlib.sha256(Path(k[0]).read_bytes()).hexdigest() for n, k in b.kernels.items()},
                  sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(build.KDIR) / "rr_output.hlsl", Path(build.KDIR) / "rr_resolve.hlsl",
                            Path(build.KDIR) / "rr_resolve_fast.hlsl", Path(build.KDIR) / "rr_resolve_fast_main.hlsli",
                            Path(build.KDIR) / "rr_resolve_fast_math.hlsli",
                            Path(build.KDIR) / "rr_exposure.hlsl", Path(build.KDIR) / "rr_input.hlsl")},
                  scope="synthetic game inputs; weights / symbols from the stated model; one submission per frame "
                        "on a resident arena; no game, no proxy, no real model")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                            encoding="utf-8")
    print("PASS" if passed else "FAIL", flush=True)
    if a.mutation:
        failed = [r["frame"] for r in rows if not r["passed"]]
        print(f"mutation {a.mutation}: frames failing {failed}", flush=True)
    return 0 if passed else 1


GROUP_OF = {"resolve-pool": "PH2 pool", "pool-main": "PH2 pool", "resolve-pool-fast": "PH2 pool",
            "exposure-partial": "exposure", "exposure-finish": "exposure",
            "input-decode": "input", "input-clear": "input", "input-features": "input", "input-replay": "input", "input-mlp": "input",
            "regroup": "body", "D0-R": "head", "D0-R-W": "head", "D0-R-F": "head", "widen": "widen", "output-main": "output", "output-pool": "output",
            "output-prep": "output", "output-fast": "output", "output-fallback": "output",
            "resolve-main": "resolve", "resolve-fast": "resolve", "resolve-fixup": "resolve", "init": "resolve"}


def bench(a):
    m = load_model(a)
    render_s, out_s = a.bench.split(":")
    render = tuple(int(v) for v in render_s.split("x"))
    O = tuple(int(v) for v in out_s.split("x"))
    t0 = time.time()
    b = Builder(m, O, render, step=a.step, zmode=a.zmode, debug=False, input_exact=not a.fast_input,
                resolve_exact=not a.fast_resolve, steady=True, output_fast=a.output_path == "fast",
                resolve_kernel=a.resolve, pool_kernel=a.pool, head_mode=a.head, input_variant=a.input_variant)
    b.add_frame(render)
    first = b.setup                      # per-model / per-arena setup dispatches (tables) are not part of the frame
    print(f"Prepared 1 steady frame at render {render} -> output {O}: {len(b.disp)} dispatches, arena "
          f"{b.ar.size / 1e9:.2f} GB ({time.time() - t0:.0f} s)", flush=True)
    if b.ar.size >= 1 << 32:
        raise SystemExit("arena above 4 GB")
    wd = str(Path(build.BUILD) / f"seq-bench-{render[0]}x{render[1]}-{a.tag or a.step}")
    img, log = build.run(b.ar, b.kernels, b.disp, require_double=True, workdir=wd, reps=a.reps, time_first=first,
                         timeout=3000)
    if os.environ.get("WG_RR_DBGCNT"):           # debug counters of RR_DBG_CNT routes (arena dwords 4..6, all reps)
        print("debug counters", img[16:28].view(np.uint32).tolist(), flush=True)
    if b.head_mode == "exact":           # windows the exact head's integer pass marked (recomputed by the fixup)
        hb, ha, wt_, ht_ = b.last_head
        nwin = ((wt_ + 7) // 8 + 1) * ((ht_ + 7) // 8 + 1)
        fl = img[hb + ha.ptr["flag"]: hb + ha.ptr["flag"] + nwin * 16].view("<u4").reshape(-1, 4)
        xa = [d for d in b.disp if d[0] == "D0-R"][-1][2][0]
        xc = img[xa: xa + wt_ * ht_ * 32]
        if os.environ.get("WG_RR_HEAD_DUMP"):
            np.save(os.environ["WG_RR_HEAD_DUMP"], xc.reshape(ht_, wt_, 32))
        print(f"head x: NaN codes {int(np.count_nonzero((xc & 127) == 127))}, zero codes "
              f"{int(np.count_nonzero((xc & 127) == 0))} of {xc.size}, max exponent field {int(((xc >> 3) & 15).max())}")
        print(f"exact head: {int(np.count_nonzero(fl.any(axis=1)))} of {nwin} window slots marked "
              f"(flag words: {[(int(u), int(c)) for u, c in zip(*np.unique(fl, return_counts=True))][:10]})", flush=True)
    per, total = [], None
    for line in log.splitlines():
        f = line.split()
        if line.startswith("time "):
            per.append(dict(index=int(f[1]), kernel=f[2], best_ms=float(f[4]), mean_ms=float(f[7])))
        elif line.startswith("total "):
            total = dict(best_ms=float(f[2]), mean_ms=float(f[5]))
    groups = {}
    for p in per:
        g = GROUP_OF.get(p["kernel"], "body")
        groups.setdefault(g, dict(best_ms=0.0, mean_ms=0.0, dispatches=0))
        groups[g]["best_ms"] += p["best_ms"]
        groups[g]["mean_ms"] += p["mean_ms"]
        groups[g]["dispatches"] += 1
    report = dict(render=render, output=O, step=a.step, zmode=a.zmode, input_exact=not a.fast_input,
                  input_variant=a.input_variant, resolve=a.resolve,
                  resolve_exact=not a.fast_resolve, model=m.source, reps=a.reps, per_dispatch=per, per_stage=groups,
                  frame_span=total, sum_of_dispatches=dict(best_ms=sum(p["best_ms"] for p in per),
                                                           mean_ms=sum(p["mean_ms"] for p in per)))
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    for g, v in groups.items():
        print(f"{g:10s} {v['best_ms']:8.3f} ms best {v['mean_ms']:8.3f} ms mean ({v['dispatches']} dispatches)")
    print(f"frame span {total}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--synthetic", action="store_true", help="own synthetic network-like values instead of a file")
    ap.add_argument("--b2", action="store_true", help="with --synthetic: build variant B2")
    ap.add_argument("--output", default="40x27")
    ap.add_argument("--renders", default="29x19,34x23", help="render size before / after the size change")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--step", default="Z", choices=("E", "X", "Y", "Z", "Z2", "I"),
                    help="body step (default Z: the exact step, bit-identical to X, the runtime's default since r2)")
    ap.add_argument("--dump-body", help="npz of every stage's raw outputs per frame (and Y fixup counts)")
    ap.add_argument("--zmode", default="exact", choices=("exact", "f32"))
    ap.add_argument("--mutation", choices=("hid", "pm", "ch", "exposure", "ph1"))
    ap.add_argument("--bench", help="render:output, e.g. 1706x960:2560x1440")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--head", default="default", choices=("default", "exact"),
                    help="exact: the runtime's exact head (integer step + strict fixup; bit-exact to the reference)")
    ap.add_argument("--tag", help="work directory name under WG_RR_BUILD")
    ap.add_argument("--fast-input", action="store_true", help="bench: hardware log2/exp2 in the input features")
    ap.add_argument("--input-variant", default="exact",
                    help="bench: opt-in near-exact input features kernel (gpu/input_stage.py LITE_VARIANTS; "
                         "name+K=V adds defines for timing experiments)")
    ap.add_argument("--fast-resolve", action="store_true", help="bench: hardware elementary functions in the resolve")
    ap.add_argument("--output-path", default="fast", choices=("fast", "main"),
                    help="output stage: fast path (prep / fast / fallback / pool, default) or rr_output_main + pool")
    ap.add_argument("--resolve", default="fast", choices=("fast", "verified"),
                    help="resolve kernels: fast (rr_resolve_fast + fixup, same bits) or verified (rr_resolve_main)")
    ap.add_argument("--pool", default="tables", choices=("tables", "resolve-fast"),
                    help="PH2 pool: rr_pool_main through exact tables (default) or rr_resolve_pool_fast")
    a = ap.parse_args(argv)
    if "WG_RR_BUILD" not in os.environ:
        ap.error("WG_RR_BUILD must name a private build directory")
    return bench(a) if a.bench else sequence(a)


if __name__ == "__main__":
    raise SystemExit(main())
