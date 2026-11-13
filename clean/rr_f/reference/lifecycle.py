# SPDX-License-Identifier: MIT
"""Host-only lifecycle model from spec 53 draft 6 and 50.2, 2026-10-18.

This is an original internal coordinator, not a game ABI or an executor. Plans
describe required work; acknowledgements are caller reports, not proof of GPU
execution. No image mathematics is supplied (resources.py runs 54/54b). The caller owns
images and must retain partial writes after failure. CP-on-resize and jitter
timing implement the explicitly provisional WrongGPU choices in 53.13.
"""

from dataclasses import dataclass
from enum import Enum


class Pass(str, Enum):
    PH2 = "ph2_pooling"
    CLEAR_EX = "clear_both_ex_to_one"
    EXPOSURE = "exposure"
    INPUT = "input"
    NETWORK_HEAD = "network_and_head"
    OUTPUT = "output_stage"
    PLAIN_RESOLVE = "plain_resolve"
    EXTENDED_RESOLVE = "extended_resolve"


@dataclass(frozen=True)
class CameraPair:
    """Opaque row/column convention: caller supplies two flattened f32 matrices.

    Values, including NaNs/infinities, are preserved without usability checks.
    Matrix interpretation and fallback belong to the unavailable input contract.
    """

    view_to_clip: tuple
    world_to_view: tuple

    def __post_init__(self):
        for name in ("view_to_clip", "world_to_view"):
            value = tuple(getattr(self, name))
            if len(value) != 16:
                raise ValueError("camera matrices must each contain 16 f32 values")
            object.__setattr__(self, name, value)


ZERO_CAMERA = CameraPair((0.0,) * 16, (0.0,) * 16)
REBUILT_IMAGES = ("HID", "PM[0]", "PM[1]", "Pe", "Pn", "GATES", "PREC",
                  "CANDW", "CH[0]", "CH[1]")
RETAINED_IMAGES = ("PH1", "PH2", "EX[0]", "EX[1]", "TH", "OH", "AH")


class ZeroExtentError(ValueError):
    """Rejected before preparation; the pending-reset latch is the only change."""


class RecreateRequired(ValueError):
    """A changed output size requires a new coordinator/instance (53.10)."""


@dataclass(frozen=True)
class RebuildPlan:
    """Required resource work, not an allocation performed by this module.

    Every render-size change recreates these images zero-filled and rebuilds
    tensor buffers, even if the padded tensor extent does not change. Conditional
    history activation/creation is deliberately unspecified (missing 54b).
    """

    generation: int
    render_size: tuple
    tensor_size: tuple
    history_size: tuple
    zero_fill_images: tuple = REBUILT_IMAGES
    retained_images: tuple = RETAINED_IMAGES
    rebuild_tensor_buffers: bool = True


@dataclass(frozen=True)
class FramePlan:
    sequence: int
    render_size: tuple
    output_size: tuple
    tensor_size: tuple
    history_size: tuple
    parity: int
    counter: int
    reset: bool
    entry_reset_generation: int | None
    previous_jitter: tuple
    current_jitter: tuple
    current_camera: CameraPair
    effective_camera: CameraPair
    passes: tuple
    rebuild: RebuildPlan | None

    @property
    def history_read(self):
        """CH/PM and applicable conditional histories; reset readers bypass."""
        return self.parity

    @property
    def history_write(self):
        return 1 - self.parity

    @property
    def exposure_read(self):
        return 1 - self.parity

    @property
    def exposure_write(self):
        return self.parity


@dataclass(frozen=True)
class LifecycleEvent:
    kind: str
    sequence: int
    stage: Pass | None = None
    detail: str = ""


def _size(value):
    value = tuple(value)
    if len(value) != 2 or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("size must be two nonnegative integers")
    return value


def _tensor(size):
    return tuple((n + 31) // 32 * 32 for n in size)


class Lifecycle:
    """Serial, bounded host bookkeeping for one instance.

    Creation requires positive sizes and mandates caller zero-fill of all created
    images. begin_frame models the atomic ensure-state transition and emits its
    resource work as a plan. Rebuild metadata then stands after later failure.
    Allocation/preparation failures are outside this model: their resource
    recovery contract is unavailable. No image write, rollback, or inferred
    successful pass occurs. Use only for serial host traces; this is not a
    thread-safe resource allocator or command scheduler.
    """

    def __init__(self, render_size, output_size, *, alpha_upscaling=False):
        self.render_size = _size(render_size)
        self.output_size = _size(output_size)
        if 0 in self.render_size + self.output_size:
            raise ValueError("creation sizes must be positive")
        self.alpha_upscaling = bool(alpha_upscaling)
        self.parity = self.counter = 0
        self.previous_jitter = (0.0, 0.0)
        self.previous_camera = ZERO_CAMERA
        self.resource_generation = 0
        self._reset_generation = 1
        self.pending_reset_generation = 1
        self._sequence = 0
        self._active = None
        self._next_pass = 0
        self._jitter_consumed = False

    @property
    def pending_reset(self):
        return self.pending_reset_generation is not None

    @property
    def tensor_size(self):
        return _tensor(self.render_size)

    @property
    def history_size(self):
        return tuple(3 * (n + n % 2) // 2 for n in self.output_size)

    def _request_reset(self):
        self._reset_generation += 1
        self.pending_reset_generation = self._reset_generation

    def begin_frame(self, render_size, *, jitter, camera, game_reset=False,
                    transparency_attached=False, layer_attached=False, output_size=None):
        """layer_attached (alias transparency_attached): a layer - the transparency layer or
        colour-before-particles (erratum 50-53, 54b.2) - is attached; selects the extended resolve."""
        layer_attached = bool(layer_attached or transparency_attached)
        if self._active is not None:
            raise RuntimeError("finish the active frame before beginning another")
        render = _size(render_size)
        output = self.output_size if output_size is None else _size(output_size)
        # Zero rejection precedes rebuild, counters, camera, jitter, and sequence.
        if 0 in render + output:
            self._request_reset()
            raise ZeroExtentError("zero extent rejected; reset now pending")
        if output != self.output_size:
            raise RecreateRequired("output-size change requires a new instance")
        jitter = tuple(jitter)
        if len(jitter) != 2 or not isinstance(camera, CameraPair):
            raise ValueError("supply a jitter pair and CameraPair")
        entry_reset = self.pending_reset_generation
        reset = bool(game_reset or entry_reset is not None)
        rebuild = None
        if render != self.render_size:
            rebuild = RebuildPlan(self.resource_generation + 1, render,
                                  _tensor(render), self.history_size)
            self.render_size = render
            self.resource_generation = rebuild.generation
            self._request_reset()
        passes = ([Pass.CLEAR_EX] if reset else [Pass.PH2]) + [
            Pass.EXPOSURE, Pass.INPUT, Pass.NETWORK_HEAD, Pass.OUTPUT,
            Pass.EXTENDED_RESOLVE if layer_attached or self.alpha_upscaling
            else Pass.PLAIN_RESOLVE]
        self._sequence += 1
        plan = FramePlan(self._sequence, render, output, _tensor(render),
                         self.history_size, self.parity, self.counter, reset,
                         entry_reset, self.previous_jitter, jitter, camera,
                         camera if reset else self.previous_camera,
                         tuple(passes), rebuild)
        self._active = plan
        self._next_pass = 0
        self._jitter_consumed = False
        return plan

    def _check(self, plan):
        if self._active is not plan:
            raise RuntimeError("frame ticket is not active on this instance")

    def input_consumed_jitter(self, plan):
        """Report the precise provisional 53.5 consumption point, even if the
        remainder of INPUT subsequently fails. No successful INPUT is inferred.
        """
        self._check(plan)
        if (self._next_pass >= len(plan.passes)
                or plan.passes[self._next_pass] != Pass.INPUT
                or self._jitter_consumed):
            raise RuntimeError("jitter consumption must occur once during INPUT")
        self.previous_jitter = plan.current_jitter
        self._jitter_consumed = True
        return LifecycleEvent("input_consumed_jitter", plan.sequence, Pass.INPUT)

    def require_pass(self, plan, stage):
        """Validate the current pass before a caller mutates its owned resources.

        This does not advance the ticket or imply that any image work succeeded.
        """
        self._check(plan)
        if (self._next_pass >= len(plan.passes)
                or stage != plan.passes[self._next_pass]):
            raise RuntimeError("pass acknowledgement is out of order")

    def acknowledge_pass(self, plan, stage):
        self.require_pass(plan, stage)
        if stage == Pass.INPUT and not self._jitter_consumed:
            raise RuntimeError("report INPUT jitter consumption before completion")
        self._next_pass += 1
        return LifecycleEvent("pass_succeeded", plan.sequence, stage)

    def complete_frame(self, plan):
        self._check(plan)
        if self._next_pass != len(plan.passes):
            raise RuntimeError("all required passes, including resolve, must succeed")
        self.parity = 1 - self.parity
        self.counter = (self.counter + 1) % 8
        self.previous_camera = plan.current_camera
        if (plan.reset and plan.entry_reset_generation is not None
                and self.pending_reset_generation == plan.entry_reset_generation):
            self.pending_reset_generation = None
        self._active = None
        return LifecycleEvent("frame_completed", plan.sequence)

    def fail_frame(self, plan, reason):
        """Release the ticket; retain partial resource writes, pj, and resets.

        CP and counters have not yet changed. reason is diagnostic only; it does
        not invent an error code or resource recovery contract for the host ABI.
        """
        self._check(plan)
        stage = plan.passes[self._next_pass] if self._next_pass < len(plan.passes) else None
        self._active = None
        return LifecycleEvent("frame_failed", plan.sequence, stage, str(reason))
