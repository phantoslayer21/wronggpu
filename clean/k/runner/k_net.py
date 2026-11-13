# SPDX-License-Identifier: MIT
"""The whole model K network (spec 41 stages, wiring and offsets of spec 44) on one device: task K4's harness core.

    net = KNet(dev, func, wb, hb, params)        # dev: clean.runner.hip.Hip or clean.runner.emu.Emu
    net.upload_input(x0)                         # [Hb, Wb, 16] network input (level 0)
    net.run(f)                                   # all eleven stages, frame index f (spec 44 offsets)
    out = net.download("OUT", (hb, wb, 40))
    times = net.time_stages(f, repeats=30)       # {stage: median ms}, plus "total"

Variants with the wiring layout.FUSED_IO (FUSED_VARIANTS) choose a path per stage: "fused" (one k_stage_* kernel,
128 lanes per window), ("wide", NG) (one k_stagew<NG>_* kernel, 128 NG lanes per window; levels 3..5) or "split"
(the separate kernels, through buffer T):
  "fused" (default)  wide kernels at levels 3..5 (DEFAULT_PATHS), two windows per workgroup in e0 and d0 (k_stage2_*,
                     bit-identical to one window and faster since round 2: half the weight reads per token),
                     k_stage_* elsewhere
  "fused_w1"         the same with one window per workgroup in e0 and d0 (the plan before round 2)
  "fused2"           = "fused" (the earlier name of the two-window plan)
  "fused1"           k_stage_* for every stage (the plan before the wide kernels)
  "auto"             k_stage_* with e3, e4, d4, d3 split
  "wide2" / "wide4"  NG = 2 / 4 at levels 3..5 where built (m: 4)
These accumulate in FP32 (spec 41 as written). Opt-in FP16 accumulation (spec 49 speed approximation, the kernels'
"h" forms; clean/k/README.md "Speed approximations"): "<variant>@f16" for each of them ("split" stages stay FP32), and
"fused16", the fastest (DEFAULT_PATHS16, two windows per workgroup in e0 and d0; "fused16_w1": one).
"g2" / "g1": the separate kernels (embedding / expand, block, merge / head; linear kernels with 32 / 16 tokens per
wave) for every stage, wiring layout.stage_io.
Buffers: clean/k/reference/layout.py buffer_plan (X0, S0..S4, A, B, OUT, FP16 maps); each stage's weights are one
stage blob (clean/k/runner/k_pack.py).
"""
import numpy as np

from clean.k.reference.layout import (FUSED_IO, INPUT_C, OUTPUT_C, STAGES, buffer_plan, level_size, stage_offsets,
                                      cdiv, window_grid)
from clean.k.runner import k_launch as KL
from clean.k.runner import k_pack


DEFAULT_PATHS = {s: ("wide", ng) for s, ng in KL.WIDE_DEFAULT.items()}
# FP32-accumulating variants (spec 41 as written)
FUSED_VARIANTS = {
    "fused": DEFAULT_PATHS,
    "fused2": DEFAULT_PATHS,
    "fused_w1": DEFAULT_PATHS,
    "fused1": {},
    "auto": {"e3": "split", "e4": "split", "d4": "split", "d3": "split"},
    "wide2": {s: ("wide", 2 if 2 in ngs else max(ngs)) for s, ngs in KL.WIDE_BUILDS.items()},
    "wide4": {s: ("wide", 4) for s in KL.WIDE_BUILDS},
}
# FP16-accumulating variants (spec 49 speed approximation, opt-in; clean/k/README.md "Speed approximations"):
# "<variant>@f16" runs the same paths with the FP16 forms of the fused and wide kernels ("split" stages stay FP32);
# "fused16" is the fastest of them (DEFAULT_PATHS16: wide with NG = 4 for e4, m, d4, k_stageh_* for e3 and d3; measured
# at 1440p, s = 0.5, against the FP16 forms of the FP32 plan and four other mixes, clean/k/README.md).
DEFAULT_PATHS16 = {s: ("wide", ng) for s, ng in KL.WIDE_DEFAULT16.items()}
F16_VARIANTS = {f"{v}@f16" for v in FUSED_VARIANTS} | {"fused16", "fused16_w1"}
FUSED_VARIANTS.update({f"{v}@f16": p for v, p in list(FUSED_VARIANTS.items())})
FUSED_VARIANTS["fused16"] = DEFAULT_PATHS16
FUSED_VARIANTS["fused16_w1"] = DEFAULT_PATHS16
# variants with two windows per workgroup in e0 and d0 (k_stage2_*, k_stage2h_*; KL.WINDOWS2_STAGES)
WINDOWS2_VARIANTS = {"fused", "fused2", "fused@f16", "fused2@f16", "fused16"}


def stage_path(variant, stage):
    """The path of one stage in a FUSED_VARIANTS variant: "fused", "split" or ("wide", NG)."""
    return FUSED_VARIANTS[variant].get(stage, "fused")


def stage_flops(st, wb, hb):
    """Matrix FLOPs (2 per multiply-add) of one stage, counting every computed window member (mirrored included)."""
    lw, lh = level_size(wb, hb, st["level"])
    C, D, H = st["C"], st["D"], st["heads"]
    nwx, nwy = cdiv(lw + 4, 8), cdiv(lh + 4, 8)  # typical offset 4
    t = 64 * nwx * nwy
    qkv = D if st["mix"] == "fixed" else 3 * D
    mixing = (1 if st["mix"] == "fixed" else 2) * 64 * 32 * H
    f = 2 * t * (C * qkv + mixing + D * C + 8 * C * C)
    n = lw * lh
    if st.get("embed"):
        f += 2 * n * 16 * 32
    if st["kind"] == "enc":
        f += 2 * cdiv(lw, 2) * cdiv(lh, 2) * 4 * C * st["C_next"]
    if st["kind"] == "dec":
        f += 2 * cdiv(lw, 2) * cdiv(lh, 2) * st["C_prev"] * 4 * C
    if st.get("head"):
        f += 2 * n * C * 48
    return f


class KNet:
    def __init__(self, dev, func, wb, hb, params, variant="fused", d_row="gfx12"):
        """func(name) -> kernel handle (Runner._func or KFuncs); params: {stage: tensors}."""
        self.dev, self.func, self.wb, self.hb, self.variant = dev, func, int(wb), int(hb), variant
        from clean.k.reference.network import offset_table
        self.offset_table = offset_table(params)  # spec 44 draft 2: from the converted model
        self.bufs, self.sizes = {}, {}
        try:
            fused = variant in FUSED_VARIANTS
            split = [st["name"] for st in STAGES if fused and stage_path(variant, st["name"]) == "split"]
            for name, nbytes in buffer_plan(self.wb, self.hb, "fused" if fused else "split", split).items():
                self._alloc(name, nbytes)
            self.blobs = {}
            for st in STAGES:
                blob = k_pack.pack_stage(st["name"], params[st["name"]], d_row)
                self.blobs[st["name"]] = self._alloc("w_" + st["name"], len(blob))
                dev.upload(self.blobs[st["name"]], blob)
        except Exception:
            self.release()
            raise
        self._plans = {}

    def _alloc(self, name, nbytes):
        self.bufs[name] = self.dev.malloc(max(int(nbytes), 256))
        self.sizes[name] = max(int(nbytes), 256)
        return self.bufs[name]

    def release(self):
        for p in self.bufs.values():
            self.dev.free(p)
        self.bufs.clear()

    def memory_bytes(self):
        return sum(self.sizes.values())

    # ---- launches -------------------------------------------------------------------------------------------------

    def stage_plan(self, f):
        """{stage: [launch]} for frame index f (spec 44 offsets depend on f mod 8)."""
        key = f % 8
        if key not in self._plans:
            off = stage_offsets(f, self.offset_table)
            B = self.bufs
            plan = {}
            for st in STAGES:
                lv = level_size(self.wb, self.hb, st["level"])
                insize = level_size(self.wb, self.hb, st["level"] + 1) if st["kind"] == "dec" else lv
                if self.variant in FUSED_VARIANTS:
                    io = FUSED_IO[st["name"]]
                    bufs = {k: B[v] for k, v in io.items()}
                    bufs.setdefault("block", None)
                    sizes = {"in": insize, "level": lv}
                    path = stage_path(self.variant, st["name"])
                    if path == "split":
                        bufs["tmp"] = B.get("T")
                        plan[st["name"]] = KL.stage_launches(st["name"], self.blobs[st["name"]], bufs, sizes,
                                                             off[st["name"]], "g2")
                        continue
                    wide = path[1] if isinstance(path, tuple) else 0
                    if wide and self.variant in ("fused", "fused2", "fused_w1"):  # the runtime's "fused" (wide_ng)
                        wide = KL.wide_ng(st["name"], wide, int(np.prod(window_grid(*lv, *off[st["name"]]))))
                    plan[st["name"]] = [KL.stage_fused_launch(st["name"], self.blobs[st["name"]], bufs, sizes,
                                                              off[st["name"]],
                                                              2 if self.variant in WINDOWS2_VARIANTS else 1, wide,
                                                              f16=self.variant in F16_VARIANTS)]
                    continue
                if st["kind"] == "dec":
                    insize = level_size(self.wb, self.hb, st["level"] + 1)
                    bufs = {"in": B["B"], "skip": B[st["skip"]], "tmp": B["A"], "block": B["B"]}
                    if st.get("head"):
                        bufs["head"] = B["OUT"]
                elif st["kind"] == "enc":
                    insize = lv
                    bufs = {"in": B["X0"] if st.get("embed") else B["A"], "tmp": B["A"], "block": B[st["skip"]],
                            "merged": B["A"]}
                else:
                    insize = lv
                    bufs = {"in": B["A"], "block": B["B"]}
                plan[st["name"]] = KL.stage_launches(st["name"], self.blobs[st["name"]], bufs,
                                                     {"in": insize, "level": lv}, off[st["name"]], self.variant)
            self._plans[key] = plan
        return self._plans[key]

    def _launch(self, launches):
        for name, grid, threads, args in launches:
            self.dev.launch(self.func(name), grid, threads, args)

    def upload_input(self, x0):
        x0 = np.asarray(x0, dtype=np.float32)
        if x0.shape != (self.hb, self.wb, INPUT_C):
            raise ValueError(f"network input must be [{self.hb}, {self.wb}, {INPUT_C}], got {x0.shape}")
        self.dev.upload(self.bufs["X0"], x0.astype(np.float16).tobytes())

    def run(self, f=0):
        for launches in self.stage_plan(f).values():
            self._launch(launches)

    def download(self, name, shape):
        raw = self.dev.download(self.bufs[name], int(np.prod(shape)) * 2)
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)

    def output(self):
        return self.download("OUT", (self.hb, self.wb, OUTPUT_C))

    def block_output(self, stage):
        """(buffer name, shape) holding a stage's block output after the stage has run (d0 fused: not written)."""
        st = next(s for s in STAGES if s["name"] == stage)
        lw, lh = level_size(self.wb, self.hb, st["level"])
        if self.variant in FUSED_VARIANTS:
            name = FUSED_IO[stage].get("block")
        else:
            name = st["skip"] if st["kind"] == "enc" else "B"
        return name, (lh, lw, st["C"])

    # ---- timing ---------------------------------------------------------------------------------------------------

    def time_stages(self, f=0, repeats=30):
        """Median GPU ms per stage (events between consecutive stages of whole-network runs) and of the whole run."""
        plan = self.stage_plan(f)
        names = list(plan)
        ev = [self.dev.event() for _ in range(len(names) + 1)]
        try:
            self.run(f)  # warm-up
            self.dev.sync()
            per = {n: [] for n in names}
            total = []
            for _ in range(int(repeats)):
                self.dev.record(ev[0])
                for i, n in enumerate(names):
                    self._launch(plan[n])
                    self.dev.record(ev[i + 1])
                self.dev.sync()
                for i, n in enumerate(names):
                    per[n].append(self.dev.elapsed_ms(ev[i], ev[i + 1]))
                total.append(self.dev.elapsed_ms(ev[0], ev[-1]))
            out = {n: float(np.median(v)) for n, v in per.items()}
            out["total"] = float(np.median(total))
            return out
        finally:
            for e in ev:
                self.dev.destroy_event(e)

    def time_kernels(self, f=0, repeats=30):
        """Median GPU ms of every launch on its own: {stage: [(kernel, ms)]}."""
        plan = self.stage_plan(f)
        a, b = self.dev.event(), self.dev.event()
        try:
            self.run(f)
            self.dev.sync()
            out = {}
            for n, launches in plan.items():
                out[n] = []
                for launch in launches:
                    t = []
                    for _ in range(int(repeats)):
                        self.dev.record(a)
                        self._launch([launch])
                        self.dev.record(b)
                        self.dev.sync()
                        t.append(self.dev.elapsed_ms(a, b))
                    out[n].append((launch[0], float(np.median(t))))
            return out
        finally:
            self.dev.destroy_event(a)
            self.dev.destroy_event(b)


def stage_info(wb, hb, table, f=0):
    """{stage: dict(level size, windows, GFLOP)} for reports."""
    off = stage_offsets(f, table)
    info = {}
    for st in STAGES:
        lw, lh = level_size(wb, hb, st["level"])
        nwx, nwy = window_grid(lw, lh, *off[st["name"]])
        info[st["name"]] = {"level": st["level"], "size": [lw, lh], "C": st["C"], "mix": st["mix"],
                            "windows": nwx * nwy, "gflop": stage_flops(st, wb, hb) / 1e9}
    return info
