# SPDX-License-Identifier: MIT
"""Owned CPU resource scaffold for published specs 50/53/54/54b/55/58.

NumPy storage, not a full evaluate implementation. Input commits accept
externally prepared synthetic data (page 52 is not implemented here). PH2 pooling
and the output resolve with its conditional histories TH/OH/AH run the page-54b
reference (reference/resolve.py). No converter or GPU work is supplied.
Lifecycle owns every ticket, pass index, counter, reset latch, jitter and camera.
Partial successful writes survive fail_frame; frame completion is never inferred.
"""
import operator

import numpy as np

from .exposure import auto_exposure
from .lifecycle import Lifecycle, Pass
from .output import OutputResult, decode_unorm8, output_stage
from .resolve import ResolveHistories, quarter_pool, resolve


EXCLUSIONS = (
    "IN0/input mathematics (52) only through run_input; write_prepared_motion stays a prepared-data store",
    "network intermediate allocation/execution; only Cnet is materialized",
    "allocation-failure recovery, GPU resources, converter and game integration",
)


def _half(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind != "f":
        raise ValueError(f"{name} must be a floating array with shape {shape}")
    with np.errstate(all="ignore"):
        stored = value.astype(np.float16)
    if not np.array_equal(value, stored.astype(value.dtype), equal_nan=True):
        raise ValueError(f"{name} must contain already-rounded binary16 values")
    return stored


def _codes(value, shape, dtype, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"{name} must have shape {shape} and dtype {np.dtype(dtype)}")
    if dtype == np.uint16 and (np.any(value[..., :3] > 1023)
                               or np.any(value[..., 3] > 3)):
        raise ValueError(f"{name} requires RGB10A2 channel codes")
    # Snapshot every source before any destination is written. In particular,
    # simultaneous head channel swaps must not depend on commit copy order.
    return value.copy()


class ResourceState:
    """Serial host images with explicit stage ownership and immediate stores.

    Sizes use (width, height), arrays [height, width, channel]. PH1/PH2 are
    unpacked uint16 [R10,G10,B10,A2] codes, matching reference.output. All images
    begin zero-filled; the scheduled CLEAR_EX pass alone sets EX to one.

    buffer() returns detached immutable-byte-backed inspection snapshots. No
    view/base/flag/metadata operation on a snapshot can mutate owned storage.
    resource_identity() exposes an opaque token stable until reallocation.
    Methods validate all supplied buffers before making any stores.
    Successful commit methods acknowledge the corresponding Lifecycle pass;
    write_prepared_motion deliberately does not, so callers can report an INPUT
    failure after a partial image write or the provisional jitter-use point.

    Missing-pass acknowledgements, if a harness supplies them to lifecycle,
    are caller reports only. This class has no successful full-frame API.
    """

    exclusions = EXCLUSIONS

    def __init__(self, render_size, output_size, *, alpha_upscaling=False):
        self.lifecycle = Lifecycle(render_size, output_size, alpha_upscaling=alpha_upscaling)
        # TH / OH / AH (54b.6): AH exists from creation with alpha upscaling; TH / OH appear on first use.
        self.resolve_histories = ResolveHistories(tuple(output_size), alpha_upscaling=alpha_upscaling)
        self._images, self._identities = {}, {}
        self.exposure_result = None
        self._rebuild_render()
        wh, hh = self.lifecycle.history_size
        self._allocate("PH1", ((hh + 1) // 2, (wh + 1) // 2, 4), np.uint16)
        self._allocate("PH2", ((hh + 3) // 4, (wh + 3) // 4, 4), np.uint16)
        for i in range(2):
            self._allocate(f"EX[{i}]", (1, 1, 1), np.float16)
        self._rebuild_scratch()

    def _allocate(self, name, shape, dtype):
        self._images[name] = np.zeros(shape, dtype=dtype)
        self._identities[name] = object()

    def _rebuild_render(self):
        wr, hr = self.lifecycle.render_size
        wh, hh = self.lifecycle.history_size
        for name, channels, dtype in (("HID", 4, np.int8), ("Pe", 2, np.float16),
                                     ("Pn", 2, np.float16), ("GATES", 4, np.uint8),
                                     ("PREC", 4, np.float16), ("CANDW", 4, np.uint8)):
            self._allocate(name, (hr, wr, channels), dtype)
        for i in range(2):
            self._allocate(f"PM[{i}]", (hr, wr, 4), np.float16)
            self._allocate(f"CH[{i}]", (hh, wh, 4), np.float16)

    def _rebuild_scratch(self):
        # Cnet/Q never carry temporal data: fresh workspaces for every call.
        wt, ht = self.lifecycle.tensor_size
        wh, hh = self.lifecycle.history_size
        self._allocate("Cnet", (ht, wt, 4), np.float16)
        self._allocate("Q", (hh, wh, 3), np.float16)

    def buffer(self, name):
        """Immutable snapshot; later pass writes do not change this snapshot.

        The complete NumPy base chain ends in detached immutable bytes. Array
        metadata is local to the snapshot, and writeability cannot be enabled.
        """
        value = self._images[name]
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)

    def resource_identity(self, name):
        """Opaque allocation identity, stable across writes and retention."""
        return self._identities[name]

    def begin_frame(self, render_size, *, layer_attached=False, **kwargs):
        """layer_attached: a layer (transparency layer or colour-before-particles) is attached (54b.2)."""
        plan = self.lifecycle.begin_frame(render_size, layer_attached=layer_attached, **kwargs)
        if plan.rebuild is not None:
            self._rebuild_render()
        self._rebuild_scratch()
        self.exposure_result = None
        return plan

    def clear_exposure(self, plan):
        self.lifecycle.require_pass(plan, Pass.CLEAR_EX)
        for i in range(2):
            self._images[f"EX[{i}]"].fill(1)
        return self.lifecycle.acknowledge_pass(plan, Pass.CLEAR_EX)

    def run_exposure(self, plan, colour, *, symbols, origin=(0, 0)):
        """Execute owned 58 reference and store EX[par] during this pass."""
        self.lifecycle.require_pass(plan, Pass.EXPOSURE)
        previous = np.float32(self._images[f"EX[{plan.exposure_read}]"][0, 0, 0])
        result = auto_exposure(colour, symbols=symbols, render_size=plan.render_size,
                               Eprev=previous, reset=plan.reset, origin=origin)
        self._images[f"EX[{plan.exposure_write}]"][0, 0, 0] = result.stored
        self.exposure_result = result
        self.lifecycle.acknowledge_pass(plan, Pass.EXPOSURE)
        return result

    def run_ph2(self, plan, *, symbols):
        """54b.5 at the start of a non-reset frame: PH2 from the PH1 written by the previous frame."""
        self.lifecycle.require_pass(plan, Pass.PH2)
        value = quarter_pool(self._images["PH1"], self._images["PH2"], output_size=plan.output_size,
                             reset=bool(plan.reset), symbols=symbols)
        np.copyto(self._images["PH2"], value)
        return self.lifecycle.acknowledge_pass(plan, Pass.PH2)

    def run_resolve(self, plan, *, symbols, colour_out, alpha_out=None, layer=None, cbp=None, opacity=None,
                    alpha_in=None, colour_in=None, motion_origin=(0, 0), bounded_stores=False):
        """54b.1-54b.4 on this frame's Q, PM[1-par], PREC, GATES and exposure; updates TH/OH/AH.

        colour_out / alpha_out are reference.resolve.Surface destinations; returns the ResolveResult with
        the stored surfaces. The planned variant (begin_frame's layer_attached) must match the attachments."""
        layer_now = layer is not None or cbp is not None
        stage = Pass.EXTENDED_RESOLVE if (layer_now or self.lifecycle.alpha_upscaling) else Pass.PLAIN_RESOLVE
        self.lifecycle.require_pass(plan, stage)
        if self.exposure_result is None:
            raise RuntimeError("run_exposure must supply this call's exposure values")
        e = self.exposure_result
        result = resolve(self._images["Q"], output_size=plan.output_size, render_size=plan.render_size,
                         E=e.E, Ep=e.Ep, jitter=plan.current_jitter, reset=bool(plan.reset),
                         parity=plan.parity, symbols=symbols, colour_out=colour_out,
                         histories=self.resolve_histories, alpha_out=alpha_out, layer=layer, cbp=cbp,
                         opacity=opacity, alpha_in=alpha_in, colour_in=colour_in,
                         motion=self._images[f"PM[{plan.history_write}]"], motion_origin=motion_origin,
                         P=self._images["PREC"][..., :3], G=self._images["GATES"][..., :3],
                         bounded_stores=bounded_stores)
        self.resolve_histories = result.histories
        self.lifecycle.acknowledge_pass(plan, stage)
        return result

    def commit_prepared_ph2(self, plan, codes):
        """Store external PH2 codes; does NOT compute or validate 54b pooling."""
        self.lifecycle.require_pass(plan, Pass.PH2)
        value = _codes(codes, self._images["PH2"].shape, np.uint16, "PH2")
        np.copyto(self._images["PH2"], value)
        return self.lifecycle.acknowledge_pass(plan, Pass.PH2)

    def write_prepared_motion(self, plan, PM, Pe, Pn, *, origin=(0, 0)):
        """Store prepared per-token motion at origin-shifted texels (53.1).

        Source arrays cover exactly the render area. Out-of-image writes are
        discarded, leaving untouched texels intact. This supplies no page-52
        motion computation, input-stage completion, or jitter-use inference.
        """
        self.lifecycle.require_pass(plan, Pass.INPUT)
        wr, hr = plan.render_size
        values = [_half(v, (hr, wr, c), n) for v, c, n in
                  ((PM, 4, "PM"), (Pe, 2, "Pe"), (Pn, 2, "Pn"))]
        if len(origin) != 2 or any(isinstance(v, (bool, np.bool_)) for v in origin):
            raise ValueError("motion origin requires two integers")
        try:
            ox, oy = (operator.index(v) for v in origin)
        except TypeError:
            raise ValueError("motion origin requires two integers") from None
        x0, x1 = max(0, -ox), min(wr, wr - ox)
        y0, y1 = max(0, -oy), min(hr, hr - oy)
        if x1 > x0 and y1 > y0:
            for name, value in zip((f"PM[{plan.history_write}]", "Pe", "Pn"), values):
                self._images[name][y0 + oy:y1 + oy, x0 + ox:x1 + ox] = value[y0:y1, x0:x1]

    def run_input(self, plan, textures, *, S, IN0, consts, wc, SH, mv_scale_raw, **options):
        """Execute the page-52 input stage (reference/input_stage.py) on the owned temporal images: reads PM[par],
        HID, PH1, PH2 and this call's exposure values, reports the jitter use (53.5), stores PM[1-par], Pe and Pn
        (origin-shifted writes, discards) and acknowledges INPUT. Returns the stage result (x0, hold, y2, hnew,
        hmerged, ... for the network)."""
        from . import input_stage as IS
        self.lifecycle.require_pass(plan, Pass.INPUT)
        if self.exposure_result is None:
            raise RuntimeError("run_exposure must supply this call's exposure values")
        params = IS.params_from_plan(plan, exposure=self.exposure_result, SH=SH, mv_scale_raw=mv_scale_raw,
                                     **options)
        frame = IS.InputFrame(dict(textures), PH1=self._images["PH1"], PH2=self._images["PH2"],
                              HID=self._images["HID"], PM_read=self._images[f"PM[{plan.history_read}]"],
                              PM_write=self._images[f"PM[{plan.history_write}]"], Pe=self._images["Pe"],
                              Pn=self._images["Pn"])
        result = IS.input_stage(frame, params, S, IN0, consts, wc)
        self.lifecycle.input_consumed_jitter(plan)
        for name, key in ((f"PM[{plan.history_write}]", "PM"), ("Pe", "Pe"), ("Pn", "Pn")):
            np.copyto(self._images[name], result["surfaces"][key])
        self.lifecycle.acknowledge_pass(plan, Pass.INPUT)
        return result

    def acknowledge_prepared_input(self, plan):
        """Caller reports INPUT complete; jitter consumption must be reported
        separately on lifecycle at its actual use point, even before failure.
        """
        return self.lifecycle.acknowledge_pass(plan, Pass.INPUT)

    def commit_head(self, plan, head):
        """Commit existing owned head result dictionary, including stored codes.

        Accepts decoded binary16 arrays from head.head or physical float16
        storage. Code channels are already quantized; no invented head math.
        """
        self.lifecycle.require_pass(plan, Pass.NETWORK_HEAD)
        values = {}
        for name, key, dtype in (("HID", "HID_code", np.int8),
                                 ("GATES", "GATES_code", np.uint8),
                                 ("CANDW", "CANDW_code", np.uint8)):
            values[name] = _codes(head[key], self._images[name].shape, dtype, key)
        for name in ("PREC", "Cnet"):
            values[name] = _half(head[name], self._images[name].shape, name)
        if np.any(values["GATES"][..., 3] != 0) or np.any(values["PREC"][..., 3] != 0):
            raise ValueError("GATES/PREC unused fourth channel must be zero")
        for name, value in values.items():
            np.copyto(self._images[name], value)
        return self.lifecycle.acknowledge_pass(plan, Pass.NETWORK_HEAD)

    def commit_output(self, plan, result):
        """Commit page-54 output to CH[1-par], PH1 and current-frame Q."""
        self.lifecycle.require_pass(plan, Pass.OUTPUT)
        if not isinstance(result, OutputResult):
            raise TypeError("result must be OutputResult")
        name = f"CH[{plan.history_write}]"
        ch = _half(result.CH, self._images[name].shape, "CH")
        q = _half(result.Q, self._images["Q"].shape, "Q")
        ph1 = _codes(result.PH1, self._images["PH1"].shape, np.uint16, "PH1")
        if np.any(ch[..., 3] != 0) or np.any(ph1[..., 3] != 3):
            raise ValueError("CH fourth channel must be zero and PH1 alpha must be one")
        for key, value in ((name, ch), ("Q", q), ("PH1", ph1)):
            np.copyto(self._images[key], value)
        return self.lifecycle.acknowledge_pass(plan, Pass.OUTPUT)

    def run_output(self, plan, *, constants):
        """Run owned page 54 using committed head/motion/exposure resources.

        Stops before absent 54b resolve. A caller's external INPUT report is
        not evidence that the missing frontend mathematics were executed.
        """
        self.lifecycle.require_pass(plan, Pass.OUTPUT)
        if self.exposure_result is None:
            raise RuntimeError("run_exposure must supply this call's exposure values")
        e = self.exposure_result
        result = output_stage(
            self._images["Cnet"], decode_unorm8(self._images["GATES"][..., :3]),
            self._images["PREC"][..., :3], decode_unorm8(self._images["CANDW"]),
            self._images[f"PM[{plan.history_write}]"], self._images["Pe"],
            self._images["Pn"], self._images[f"CH[{plan.history_read}]"],
            output_size=plan.output_size, jitter=plan.current_jitter,
            E=e.E, Ep=e.Ep, invE=e.invE, reset=plan.reset, constants=constants)
        self.commit_output(plan, result)
        return result
