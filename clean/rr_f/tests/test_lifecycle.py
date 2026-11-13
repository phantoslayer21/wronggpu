# SPDX-License-Identifier: MIT
"""Original synthetic host traces derived from spec 53.12(4), not GPU tests."""

import math
import unittest

from clean.rr_f.reference.lifecycle import (
    CameraPair, Lifecycle, Pass, RecreateRequired, ZeroExtentError, ZERO_CAMERA,
)


def camera(n):
    return CameraPair((float(n),) * 16, (float(-n),) * 16)


RESET_PASSES = (Pass.CLEAR_EX, Pass.EXPOSURE, Pass.INPUT, Pass.NETWORK_HEAD,
                Pass.OUTPUT, Pass.PLAIN_RESOLVE)
NORMAL_PASSES = (Pass.PH2, Pass.EXPOSURE, Pass.INPUT, Pass.NETWORK_HEAD,
                 Pass.OUTPUT, Pass.PLAIN_RESOLVE)


def begin(state, n, size=(8, 6), **kwargs):
    return state.begin_frame(size, jitter=(n / 8, -n / 8), camera=camera(n), **kwargs)


def report(state, plan, stages):
    for stage in stages:
        if stage == Pass.INPUT:
            state.input_consumed_jitter(plan)
        state.acknowledge_pass(plan, stage)


def succeed(state, plan):
    report(state, plan, plan.passes)
    state.complete_frame(plan)


def state_values(state):
    return (state.parity, state.counter, state.previous_jitter,
            state.previous_camera, state.pending_reset)


class Spec53LifecycleTraces(unittest.TestCase):
    def test_creation_reset_success_and_eight_frame_wrap(self):
        s = Lifecycle((8, 6), (13, 9))
        self.assertEqual(state_values(s), (0, 0, (0, 0), ZERO_CAMERA, True))
        self.assertEqual(s.history_size, (21, 15))
        self.assertEqual(s.tensor_size, (32, 32))
        for n in range(1, 10):
            f = begin(s, n)
            self.assertEqual((f.parity, f.counter), ((n - 1) % 2, (n - 1) % 8))
            self.assertEqual(f.passes, RESET_PASSES if n == 1 else NORMAL_PASSES)
            self.assertEqual(f.reset, n == 1)
            self.assertEqual(f.effective_camera, camera(1 if n == 1 else n - 1))
            self.assertEqual((f.history_read, f.history_write,
                              f.exposure_read, f.exposure_write),
                             ((n - 1) % 2, n % 2, n % 2, (n - 1) % 2))
            self.assertEqual(f.previous_jitter, ((n - 1) / 8, -(n - 1) / 8))
            succeed(s, f)
            self.assertEqual(state_values(s),
                             (n % 2, n % 8, (n / 8, -n / 8), camera(n), False))

    def test_first_call_resize_then_repeated_resize_has_three_resets(self):
        # 53.9: first-call resize resets both first and second calls. A second
        # resize during call two schedules a third reset; none resets counters.
        s = Lifecycle((8, 6), (13, 9))
        for n, size in ((1, (9, 6)), (2, (7, 5)), (3, (7, 5))):
            f = begin(s, n, size)
            self.assertTrue(f.reset)
            self.assertEqual(f.passes, RESET_PASSES)
            self.assertEqual(f.tensor_size, (32, 32))
            self.assertEqual(f.rebuild is not None, n < 3)
            if f.rebuild:
                self.assertEqual(f.rebuild.generation, n)
                self.assertEqual(f.rebuild.zero_fill_images,
                                 ("HID", "PM[0]", "PM[1]", "Pe", "Pn", "GATES",
                                  "PREC", "CANDW", "CH[0]", "CH[1]"))
                self.assertEqual(f.rebuild.retained_images,
                                 ("PH1", "PH2", "EX[0]", "EX[1]", "TH", "OH", "AH"))
                self.assertTrue(f.rebuild.rebuild_tensor_buffers)
                self.assertNotEqual(f.entry_reset_generation, s.pending_reset_generation)
            succeed(s, f)
            self.assertEqual(state_values(s),
                             (n % 2, n, (n / 8, -n / 8), camera(n), n < 3))
        f = begin(s, 4, (7, 5))
        self.assertFalse(f.reset)
        self.assertEqual(s.resource_generation, 2)
        s.fail_frame(f, "stop synthetic trace")

    def test_resize_change_call_not_reset_and_failure_keeps_rebuild(self):
        s = Lifecycle((8, 6), (13, 9))
        succeed(s, begin(s, 1))
        f = begin(s, 2, (33, 7))
        self.assertFalse(f.reset)
        self.assertEqual(f.passes, NORMAL_PASSES)
        self.assertEqual(f.effective_camera, camera(1))
        self.assertEqual((s.render_size, s.tensor_size), ((33, 7), (64, 32)))
        self.assertEqual((f.rebuild.history_size, f.rebuild.generation), ((21, 15), 1))
        report(s, f, NORMAL_PASSES[:3])
        s.fail_frame(f, "network failed after input")
        self.assertEqual(state_values(s), (1, 1, (.25, -.25), camera(1), True))
        retry = begin(s, 3, (33, 7))
        self.assertIsNone(retry.rebuild)
        self.assertEqual((retry.reset, retry.previous_jitter, retry.effective_camera),
                         (True, (.25, -.25), camera(3)))
        succeed(s, retry)
        self.assertEqual(state_values(s), (0, 2, (.375, -.375), camera(3), False))

    def test_failure_before_each_pass_retains_only_consumed_jitter(self):
        # This literal pass sequence is the spec's required ordering. The
        # failures cover initial/pending reset and ordinary calls independently.
        for pending in (False, True):
            stages = RESET_PASSES if pending else NORMAL_PASSES
            for failed_index, failed_stage in enumerate(stages):
                with self.subTest(pending=pending, failed_stage=failed_stage):
                    s = Lifecycle((8, 6), (13, 9))
                    if not pending:
                        succeed(s, begin(s, 1))
                    f = begin(s, 2)
                    report(s, f, stages[:failed_index])
                    event = s.fail_frame(f, "synthetic stage failure")
                    self.assertEqual((event.kind, event.stage), ("frame_failed", failed_stage))
                    expected_jitter = (.25, -.25) if failed_index >= 3 else (
                        (0, 0) if pending else (.125, -.125))
                    self.assertEqual(state_values(s),
                                     (0 if pending else 1, 0 if pending else 1,
                                      expected_jitter, ZERO_CAMERA if pending else camera(1), pending))
                    retry = begin(s, 3)
                    self.assertEqual(retry.reset, pending)
                    self.assertEqual(retry.previous_jitter, expected_jitter)
                    self.assertEqual(retry.effective_camera, camera(3) if pending else camera(1))
                    succeed(s, retry)
                    self.assertFalse(s.pending_reset)

    def test_game_only_reset_failure_does_not_latch_reset_or_commit_camera(self):
        s = Lifecycle((8, 6), (13, 9))
        succeed(s, begin(s, 1))
        f = begin(s, 2, game_reset=True)
        self.assertEqual(f.effective_camera, camera(2))
        self.assertIsNone(f.entry_reset_generation)
        report(s, f, RESET_PASSES[:-1])
        s.fail_frame(f, "resolve failure")
        self.assertEqual(state_values(s), (1, 1, (.25, -.25), camera(1), False))
        retry = begin(s, 3)
        self.assertEqual((retry.reset, retry.effective_camera), (False, camera(1)))
        succeed(s, retry)
        self.assertEqual(state_values(s), (0, 2, (.375, -.375), camera(3), False))

    def test_failure_inside_input_after_jitter_consumption(self):
        s = Lifecycle((8, 6), (13, 9))
        f = begin(s, 1)
        report(s, f, RESET_PASSES[:2])
        event = s.input_consumed_jitter(f)
        self.assertEqual(event.kind, "input_consumed_jitter")
        s.fail_frame(f, "input failed after consuming jitter")
        self.assertEqual(state_values(s), (0, 0, (.125, -.125), ZERO_CAMERA, True))
        retry = begin(s, 2)
        self.assertEqual(retry.previous_jitter, (.125, -.125))
        succeed(s, retry)

    def test_successful_game_reset_preserves_entry_jitter_and_counter_progress(self):
        s = Lifecycle((8, 6), (13, 9))
        succeed(s, begin(s, 1))
        succeed(s, begin(s, 2))
        f = begin(s, 3, game_reset=True)
        self.assertEqual((f.parity, f.counter, f.previous_jitter, f.effective_camera),
                         (0, 2, (.25, -.25), camera(3)))
        self.assertEqual(f.passes, RESET_PASSES)
        self.assertIsNone(f.rebuild)
        succeed(s, f)
        self.assertEqual(state_values(s), (1, 3, (.375, -.375), camera(3), False))

    def test_resize_pending_reset_failure_keeps_generation_and_previous_camera(self):
        s = Lifecycle((8, 6), (13, 9))
        succeed(s, begin(s, 1))
        succeed(s, begin(s, 2, (7, 5)))
        pending_generation = s.pending_reset_generation
        f = begin(s, 3, (7, 5))
        self.assertTrue(f.reset)
        self.assertEqual(f.effective_camera, camera(3))
        report(s, f, RESET_PASSES[:-1])
        s.fail_frame(f, "pending-reset resolve failed")
        self.assertEqual(s.pending_reset_generation, pending_generation)
        self.assertEqual(state_values(s), (0, 2, (.375, -.375), camera(2), True))
        f = begin(s, 4, (7, 5))
        self.assertEqual(f.passes, RESET_PASSES)
        self.assertEqual(f.previous_jitter, (.375, -.375))
        succeed(s, f)
        self.assertEqual(state_values(s), (1, 3, (.5, -.5), camera(4), False))

    def test_zero_render_or_output_rejects_before_resize_and_latches_reset(self):
        for render, output in (((0, 6), (13, 9)), ((9, 0), (13, 9)),
                               ((34, 6), (0, 9)), ((34, 6), (13, 0))):
            with self.subTest(render=render, output=output):
                s = Lifecycle((8, 6), (13, 9))
                succeed(s, begin(s, 1))
                with self.assertRaises(ZeroExtentError):
                    begin(s, 2, render, output_size=output)
                self.assertEqual(state_values(s), (1, 1, (.125, -.125), camera(1), True))
                self.assertEqual((s.render_size, s.resource_generation), ((8, 6), 0))
                retry = begin(s, 3)
                self.assertEqual((retry.sequence, retry.reset), (2, True))
                self.assertEqual(retry.passes, RESET_PASSES)
                succeed(s, retry)
                self.assertEqual(state_values(s), (0, 2, (.375, -.375), camera(3), False))

    def test_extended_resolve_is_required_and_global_parity_continues(self):
        s = Lifecycle((8, 6), (13, 9))
        f = begin(s, 1, transparency_attached=True)
        self.assertEqual(f.passes[-1], Pass.EXTENDED_RESOLVE)
        report(s, f, f.passes[:-1])
        with self.assertRaises(RuntimeError):
            s.complete_frame(f)
        s.acknowledge_pass(f, Pass.EXTENDED_RESOLVE)
        s.complete_frame(f)
        succeed(s, begin(s, 2))  # conditional histories not used, parity advances
        f = begin(s, 3, transparency_attached=True)
        self.assertEqual((f.history_read, f.history_write, f.reset), (0, 1, False))
        succeed(s, f)
        a = Lifecycle((8, 6), (13, 9), alpha_upscaling=True)
        self.assertEqual(begin(a, 1).passes[-1], Pass.EXTENDED_RESOLVE)

    def test_unusable_camera_committed_as_received_and_recreation_is_new(self):
        s = Lifecycle((8, 6), (13, 9))
        bad = CameraPair((math.nan,) * 16, (math.inf,) * 16)
        f = s.begin_frame((8, 6), jitter=(.5, -.5), camera=bad)
        succeed(s, f)
        self.assertIs(s.previous_camera, bad)
        with self.assertRaises(RecreateRequired):
            begin(s, 2, output_size=(14, 9))
        self.assertEqual((s.parity, s.counter, s.pending_reset), (1, 1, False))
        replacement = Lifecycle((8, 6), (14, 9))
        self.assertEqual(state_values(replacement), (0, 0, (0, 0), ZERO_CAMERA, True))

    def test_serial_tickets_pass_order_and_explicit_input_event(self):
        s = Lifecycle((8, 6), (13, 9))
        other = Lifecycle((8, 6), (13, 9))
        f = begin(s, 1)
        for operation in (lambda: begin(s, 2), lambda: other.fail_frame(f, "wrong owner"),
                          lambda: s.acknowledge_pass(f, Pass.INPUT),
                          lambda: s.input_consumed_jitter(f), lambda: s.complete_frame(f)):
            with self.assertRaises(RuntimeError):
                operation()
        report(s, f, RESET_PASSES[:2])
        with self.assertRaises(RuntimeError):
            s.acknowledge_pass(f, Pass.INPUT)
        s.input_consumed_jitter(f)
        with self.assertRaises(RuntimeError):
            s.input_consumed_jitter(f)
        s.acknowledge_pass(f, Pass.INPUT)
        report(s, f, RESET_PASSES[3:])
        s.complete_frame(f)
        with self.assertRaises(RuntimeError):
            s.complete_frame(f)


if __name__ == "__main__":
    unittest.main()
