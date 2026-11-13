# SPDX-License-Identifier: MIT
"""Model K, one whole frame on one device: input stage (spec 42) -> the eleven network stages (spec 41 / 44) ->
output stage (spec 43), with the history images kept between frames (spec 42 / 43 inputs Hc, Hh, L, double-buffered)
and the frame index of spec 44 (0 on every reset frame, then +1 per frame; it selects the window offsets of m, e4, d4
from the converted model's table).

    kf = KFrame(dev, func, output_size, max_render_size, params, flags)    # flags: hdr, mv_lowres, depth_inverted
    out = kf.frame(x)          # x: color, motion, depth, render_size, jitter, mv_scale, mv_offset, pre_exposure,
                               #    reset, E_cur, E_aux (FP16 values), e1, e2 -> {"out": [H, W, 4] RGBA float32}
    ms = kf.time_frame(x)      # {"input", "network", "output", "total"}: median GPU ms

reference_frame(state, x, params, flags) runs the same frame with the NumPy references (state: the histories and the
frame index, updated in place), for tests.
"""
import ctypes

import numpy as np

from clean.k.reference import instage, network, outstage
from clean.k.reference.layout import base_size
from clean.k.reference.outstage import ceil16
from clean.k.runner import k_io
from clean.k.runner.k_net import KNet


class KFrame:
    def __init__(self, dev, func, output_size, max_render_size, params, flags, variant="fused", base=None,
                 exact_input=False, s=None, input_tiled=None, compact_head=None):
        """exact_input: the bit-exact input stage (k_io_exact.co) instead of the default fast one (spec 42 section 9
        accepts >= 99.9 % within one FP16 step).
        compact_head: complete frames with a G2 d0 pass the head to the output stage as the compact pixel head
        (k_stage2_d0_hp / k_stage2h_d0_hp -> k_out_pixel_c; the same values, see clean/k/kernels/k_block.hip) instead
        of the 40-channel map. Default: on the GPU, off on the host emulator. Frames run with debug=True always write
        the 40-channel map (their "N")."""
        self.dev, self.func = dev, func
        self.exact_input = exact_input
        # Only complete HIP frames opt into the local producer. The host emulator
        # and standalone input-stage interface retain the canonical two launches.
        from clean.runner.hip import Hip
        self.input_tiled = isinstance(dev, Hip) and (input_tiled is None or bool(input_tiled))
        self.compact_head = isinstance(dev, Hip) if compact_head is None else bool(compact_head)
        self.W, self.H = (int(v) for v in output_size)
        self.max_rw, self.max_rh = (int(v) for v in max_render_size)
        self.flags = {k: bool(flags.get(k, False)) for k in ("hdr", "mv_lowres", "depth_inverted")}
        from clean.k.reference.layout import DEFAULT_S
        self.s = float(DEFAULT_S if s is None else s)  # spec 44's base-size factor
        self.Bw, self.Bh = tuple(base) if base is not None else base_size(self.W, self.H, self.s)
        self.W16, self.H16 = ceil16(self.W), ceil16(self.H)
        self.net = KNet(dev, func, self.Bw, self.Bh, params, variant)
        self.bufs, self.sizes = {}, {}
        try:
            n16 = {"Hc": self.W16 * self.H16 * 4, "L": 4 * self.W16 * self.H16, "Hh": self.Bw * self.Bh * 4}
            for name, n in n16.items():
                for p in (0, 1):
                    self._alloc(f"{name}{p}", n * 2, zero=True)
            mw, mh = (self.max_rw, self.max_rh) if self.flags["mv_lowres"] else (self.W, self.H)
            self._alloc("color", self.max_rw * self.max_rh * 16)
            self._alloc("motion", mw * mh * 8)
            self._alloc("depth", self.max_rw * self.max_rh * 4)
            for name, nbytes in list(k_io.input_scratch_bytes(self.max_rw, self.max_rh).items()) + \
                    list(k_io.output_scratch_bytes(self.max_rw, self.max_rh).items()):
                self._alloc(name, nbytes)
            self._alloc("out", self.W * self.H * 16)
            self._alloc("exp2tab", 65536 * 2)
            dev.upload(self.bufs["exp2tab"], k_io.exp2_table().tobytes())
            self._alloc("table", 2048 * 16)
        except Exception:
            self.release()
            raise
        self.table_level = None
        self.parity = 0
        self.f = 0
        self.first = True
        self.E_prev = None
        self.last_render = None

    def _alloc(self, name, nbytes, zero=False):
        nbytes = max(int(nbytes), 256)
        self.bufs[name] = self.dev.malloc(nbytes)
        self.sizes[name] = nbytes
        if zero:
            self.dev.memset(self.bufs[name], 0, nbytes)
        return self.bufs[name]

    def release(self):
        for p in self.bufs.values():
            self.dev.free(p)
        self.bufs.clear()
        self.net.release()

    def _frame_dict(self, x):
        """The frame dict for both stages, with this frame's reset, previous exposure and base size."""
        rw, rh = (int(v) for v in x["render_size"])
        if rw > self.max_rw or rh > self.max_rh:
            raise ValueError(f"render size {(rw, rh)} above the maximum {(self.max_rw, self.max_rh)}")
        reset = bool(x.get("reset")) or self.first or self.last_render != (rw, rh)
        d = dict(x)
        # E_prev: the previous frame's E_cur, unless the frame gives it (spec 49's test sequences); E_cur on a reset
        e_prev = x.get("E_prev") if x.get("E_prev") is not None else self.E_prev
        d.update(self.flags, output_size=(self.W, self.H), base=(self.Bw, self.Bh), reset=reset, s=self.s,
                 E_prev=x["E_cur"] if reset or e_prev is None else e_prev)
        return d

    def _launches(self, d, full_head=False):
        """[(part, [launch])] of one frame (uploads the frame textures and the filter table). full_head: the
        40-channel head map even when the frame would use the compact pixel head."""
        dev, B = self.dev, self.bufs
        color = k_io._rgba(d["color"])
        dev.upload(B["color"], color.tobytes())
        dev.upload(B["motion"], np.ascontiguousarray(np.asarray(d["motion"], np.float32)).tobytes())
        if self.flags["mv_lowres"]:
            dev.upload(B["depth"], np.ascontiguousarray(np.asarray(d["depth"], np.float32)).tobytes())
        level = outstage.filter_level(*d["render_size"], self.W, self.H)
        if level != self.table_level:
            dev.upload(B["table"], outstage.filter_table(level).tobytes())
            self.table_level = level
        fin = k_io.frame_in(d, {"color": B["color"], "motion": B["motion"], "depth": B["depth"]})
        p, q = self.parity, 1 - self.parity
        f = 0 if d["reset"] else self.f
        nb = self.net.bufs
        input_plan, color_fused = k_io.complete_input_launches(
            fin, {"Hc": B[f"Hc{p}"], "Hh": B[f"Hh{p}"], "x0": nb["X0"],
                  "cpre": B["cpre"], "cpre2": B["cpre2"], "ctap": B["ctap"], "ctapT": B["ctapT"]},
            exact=self.exact_input, tiled=self.input_tiled)
        parts = [("input", input_plan)]
        parts.append(("network", [l for ls in self.net.stage_plan(f).values() for l in ls]))
        parts.append(("output", k_io.output_launches(
            fin, {"N": nb["OUT"], "L": B[f"L{p}"], "Hc": B[f"Hc{p}"], "table": B["table"], "exp2tab": B["exp2tab"],
                  "out": B["out"], "Lout": B[f"L{q}"], "Hcout": B[f"Hc{q}"], "Hhout": B[f"Hh{q}"],
                  "ctap": B["ctap"], "ctapT": B["ctapT"]}, {})))
        # Output colour taps depend only on current colour/exposure (spec 43 4.2/5.3).
        # Eligible tiles prepare them while loading input color; otherwise use the separate prepass.
        # Standalone input/output stage launch plans retain their original kernels.
        color_plan = parts[2][1].pop(0)
        if not color_fused:
            parts[0][1].insert(0, color_plan)
        # The G2 d0 kernels (FP32, and the FP16-accumulating form of "fused16") have the exact hidden-history
        # epilogue. Standalone networks and G1/split paths retain k_out_hidden.
        # With that epilogue nothing but k_out_pixel reads the head, and it reads three of a slot's ten channels: the
        # compact pixel head (k_stage2*_d0_hp writes it into the OUT buffer, k_out_pixel_c reads it) carries just those.
        for i, (name, grid, threads, args) in enumerate(parts[1][1]):
            if name in ("k_stage2_d0", "k_stage2h_d0"):
                compact = self.compact_head and not full_head
                parts[1][1][i] = (name + ("_hp" if compact else "_hidden"), grid, threads,
                                   args + [ctypes.c_void_p(B[f"Hh{q}"])])
                out = [launch for launch in parts[2][1] if launch[0] != "k_out_hidden"]
                if compact:
                    out = [("k_out_pixel_c",) + launch[1:] if launch[0] == "k_out_pixel" else launch for launch in out]
                parts[2] = (parts[2][0], out)
                break
        return parts, f

    def _advance(self, d, f):
        self.parity = 1 - self.parity
        self.f = f + 1
        self.first = False
        self.E_prev = d["E_cur"]
        self.last_render = tuple(int(v) for v in d["render_size"])

    def frame(self, x, debug=False):
        d = self._frame_dict(x)
        parts, f = self._launches(d, full_head=debug)  # debug returns the 40-channel head map
        for _, ls in parts:
            for name, grid, threads, args in ls:
                self.dev.launch(self.func(name), grid, threads, args)
        self.dev.sync()
        raw = self.dev.download(self.bufs["out"], self.W * self.H * 16)
        out = {"out": np.frombuffer(raw, np.float32).reshape(self.H, self.W, 4).copy(), "f": f, "reset": d["reset"]}
        if debug:
            out["x0"] = self.net.download("X0", (self.Bh, self.Bw, 16))
            out["N"] = self.net.output()
            out["frame"] = d
        self._advance(d, f)
        return out

    def histories(self):
        """The histories the next frame reads: {"Hc", "L", "Hh"} (float32 arrays of FP16 values)."""
        p = self.parity
        def get(name, shape):
            raw = self.dev.download(self.bufs[f"{name}{p}"], int(np.prod(shape)) * 2)
            return np.frombuffer(raw, np.float16).reshape(shape).astype(np.float32)
        return {"Hc": get("Hc", (self.H16, self.W16, 4)), "L": get("L", (2 * self.H16, 2 * self.W16)),
                "Hh": get("Hh", (self.Bh, self.Bw, 4))}

    def time_frame(self, x, repeats=30):
        """Median GPU ms of the frame parts; input includes the output colour prepass.
        Network includes hidden-history output for the G2 d0 paths ("fused", "fused16").

        The same frame is repeated and the frame state does not advance.
        """
        d = self._frame_dict(x)
        parts, _ = self._launches(d)
        ev = [self.dev.event() for _ in range(len(parts) + 1)]
        try:
            res = {name: [] for name, _ in parts}
            res["total"] = []
            for rep in range(int(repeats) + 1):
                self.dev.record(ev[0])
                for i, (name, ls) in enumerate(parts):
                    for ln, grid, threads, args in ls:
                        self.dev.launch(self.func(ln), grid, threads, args)
                    self.dev.record(ev[i + 1])
                self.dev.sync()
                if rep == 0:
                    continue  # warm-up
                for i, (name, _) in enumerate(parts):
                    res[name].append(self.dev.elapsed_ms(ev[i], ev[i + 1]))
                res["total"].append(self.dev.elapsed_ms(ev[0], ev[-1]))
            return {k: float(np.median(v)) for k, v in res.items()}
        finally:
            for e in ev:
                self.dev.destroy_event(e)

    def time_sustained(self, x, n=300, sample_every=10):
        """GPU time per frame under sustained load, as in a game where the GPU is never idle (the frame state does not
        advance). Warm-up: n frames back to back (the clocks settle). Phase 1: the queue drained, a host clock started,
        n frames enqueued back to back with no host synchronisation, drained again: per_frame = host time / n (GPU
        events deep in a long queue get batched timestamps, so they are not used for this; the host enqueues ~16
        launches per frame, much faster than the GPU runs them, so the GPU stays busy and the clock measures it). Phase 2,
        right after: another n frames, every sample_every-th preceded by a host sync (the queue drained: events deep in
        a long queue get batched timestamps, measured as a 16 ms output part inside a 2 ms frame) and timed per part;
        "input" / "network" / "output" are the medians of those samples."""
        d = self._frame_dict(x)
        parts, _ = self._launches(d)
        names = [p for p, _ in parts]

        def enqueue(ls):
            for ln, grid, threads, args in ls:
                self.dev.launch(self.func(ln), grid, threads, args)

        samples = list(range(0, n, sample_every))
        sev = {i: [self.dev.event() for _ in range(len(parts) + 1)] for i in samples}
        try:
            import time
            for i in range(n):  # warm-up
                for _, ls in parts:
                    enqueue(ls)
            self.dev.sync()
            t0 = time.perf_counter()
            for i in range(n):  # phase 1
                for _, ls in parts:
                    enqueue(ls)
            self.dev.sync()
            per_frame = (time.perf_counter() - t0) * 1000.0 / n
            for i in range(n):  # phase 2 (the queue still full from phase 1 at its start)
                if i in sev:
                    self.dev.sync()
                for k, (_, ls) in enumerate(parts):
                    if i in sev:
                        self.dev.record(sev[i][k])
                    enqueue(ls)
                if i in sev:
                    self.dev.record(sev[i][-1])
            self.dev.sync()
            out = {"per_frame": per_frame}
            for k, name in enumerate(names):
                out[name] = float(np.median([self.dev.elapsed_ms(sev[i][k], sev[i][k + 1]) for i in samples[1:]]))
            return out
        finally:
            for e in [e for es in sev.values() for e in es]:
                self.dev.destroy_event(e)

def new_reference_state(output_size, base=None):
    W, H = output_size
    Bw, Bh = tuple(base) if base is not None else base_size(W, H)
    W16, H16 = ceil16(W), ceil16(H)
    return {"Hc": np.zeros((H16, W16, 4), np.float16), "L": np.zeros((2 * H16, 2 * W16), np.float16),
            "Hh": np.zeros((Bh, Bw, 4), np.float16), "f": 0, "first": True, "E_prev": None, "last_render": None,
            "base": (Bw, Bh), "output_size": (W, H)}


def reference_frame(state, x, params, flags):
    """The NumPy references for one frame; state (new_reference_state) is updated like KFrame's."""
    rw, rh = (int(v) for v in x["render_size"])
    reset = bool(x.get("reset")) or state["first"] or state["last_render"] != (rw, rh)
    d = dict(x)
    d.update({k: bool(flags.get(k, False)) for k in ("hdr", "mv_lowres", "depth_inverted")},
             output_size=state["output_size"], base=state["base"], reset=reset,
             E_prev=x["E_cur"] if reset or (x.get("E_prev") is None and state["E_prev"] is None)
             else (x["E_prev"] if x.get("E_prev") is not None else state["E_prev"]))
    f = 0 if reset else state["f"]
    x0 = instage.input_stage(dict(d, Hc=state["Hc"], Hh=state["Hh"]))
    outs = network.run_network(x0, params, f)
    N = outs["d0"]["head"]
    o = outstage.output_stage(dict(d, N=N, L=state["L"], Hc=state["Hc"]))
    state.update(Hc=o["Hc"], L=o["L"], Hh=o["Hh"], f=f + 1, first=False, E_prev=x["E_cur"], last_render=(rw, rh))
    return {"out": o["out"], "x0": x0, "N": N, "f": f, "reset": reset}
