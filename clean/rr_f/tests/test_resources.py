# SPDX-License-Identifier: MIT
"""Synthetic storage traces, not full evaluate or missing-pass correctness.

The harness explicitly reports a fictitious resolve to test next-frame resource
selection. Those reports are not execution evidence for page 52 or page 54b.
"""
from dataclasses import replace
import unittest

import numpy as np

from clean.rr_f.head.head import head_channels, output_images
from clean.rr_f.reference.lifecycle import CameraPair, Pass, RecreateRequired, ZeroExtentError
from clean.rr_f.reference.output import OutputResult
from clean.rr_f.reference.resources import ResourceState
from clean.rr_f.tests.test_exposure import symbols as exposure_symbols
from clean.rr_f.tests.test_output import symbols as output_symbols


PERSISTENT = ("CH[0]", "CH[1]", "HID", "PM[0]", "PM[1]", "Pe", "Pn",
              "GATES", "PREC", "CANDW", "PH1", "PH2", "EX[0]", "EX[1]")
RETAINED = ("PH1", "PH2", "EX[0]", "EX[1]")


def begin(s, n=1, size=None, **kwargs):
    return s.begin_frame(size or s.lifecycle.render_size,
                         jitter=(n / 32, -n / 32),
                         camera=CameraPair((float(n),) * 16, (float(-n),) * 16), **kwargs)


def head(s, n):
    shape = s.buffer("HID").shape
    values = dict(HID_code=np.full(shape, n, np.int8),
                  GATES_code=np.full(shape, n, np.uint8),
                  CANDW_code=np.full(shape, n, np.uint8),
                  PREC=np.full(shape, n, np.float16),
                  Cnet=np.full(s.buffer("Cnet").shape, n, np.float16))
    values["GATES_code"][..., 3] = 0
    values["PREC"][..., 3] = 0
    return values


def output(s, n):
    ch = np.full(s.buffer("CH[0]").shape, n, np.float16)
    ch[..., 3] = 0
    ph1 = np.full(s.buffer("PH1").shape, n, np.uint16)
    ph1[..., 3] = 3
    return OutputResult(ch, np.full(s.buffer("Q").shape, n, np.float16), ph1)


def initial_pass(s, p, n):
    if p.reset:
        s.clear_exposure(p)
    else:
        ph2 = np.full(s.buffer("PH2").shape, n, np.uint16)
        ph2[..., 3] = 3
        s.commit_prepared_ph2(p, ph2)


def expose(s, p, value=1):
    wr, hr = p.render_size
    return s.run_exposure(p, np.full((hr, wr, 3), value, np.float32),
                          symbols=exposure_symbols())


def motion(s, p, n, **kwargs):
    wr, hr = p.render_size
    s.write_prepared_motion(p, np.full((hr, wr, 4), n, np.float16),
                            np.full((hr, wr, 2), n + 1, np.float16),
                            np.full((hr, wr, 2), n + 2, np.float16), **kwargs)


def through_output(s, p, n=1):
    initial_pass(s, p, n)
    expose(s, p, n)
    motion(s, p, n)
    s.lifecycle.input_consumed_jitter(p)
    s.acknowledge_prepared_input(p)
    s.commit_head(p, head(s, n))
    s.commit_output(p, output(s, n))


def simulate_external_resolve_for_trace(s, p):
    # Deliberately only a lifecycle report, never an image or 54b implementation.
    s.lifecycle.acknowledge_pass(p, Pass.PLAIN_RESOLVE)
    s.lifecycle.complete_frame(p)


class ResourceTests(unittest.TestCase):
    def test_inspection_flags_bases_and_metadata_cannot_mutate_owned_state(self):
        s = ResourceState((3, 2), (5, 3))
        for name in PERSISTENT + ("Cnet", "Q"):
            with self.subTest(name=name):
                snapshot = s.buffer(name)
                shape, dtype = snapshot.shape, snapshot.dtype
                allocation = s.resource_identity(name)
                base = snapshot
                while isinstance(base, np.ndarray):
                    with self.assertRaises(ValueError):
                        base.setflags(write=True)
                    with self.assertRaises(ValueError):
                        base.flat[0] = 42
                    base = base.base
                self.assertIsInstance(base, bytes)
                self.assertTrue(memoryview(base).readonly)
                with self.assertRaises(TypeError):
                    memoryview(base)[0] = 42
                # Metadata belongs to the detached snapshot, not the resource.
                snapshot.shape = (snapshot.size,)
                snapshot.dtype = np.uint8
                snapshot.base.shape = (1, snapshot.base.size)
                current = s.buffer(name)
                self.assertEqual((current.shape, current.dtype), (shape, dtype))
                self.assertFalse(np.any(current))
                self.assertIs(s.resource_identity(name), allocation)
        ex_before = s.buffer("EX[0]")
        p = begin(s)
        s.clear_exposure(p)
        self.assertFalse(np.any(ex_before))  # snapshots do not become live views
        np.testing.assert_array_equal(s.buffer("EX[0]"), 1)

    def test_head_swap_from_inspection_snapshots_preserves_both_inputs(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s)
        initial_pass(s, p, 1)
        expose(s, p)
        s.lifecycle.input_consumed_jitter(p)
        s.acknowledge_prepared_input(p)
        first = head(s, 1)
        first["CANDW_code"][..., :3] = 2
        first["CANDW_code"][..., 3] = 0
        s.commit_head(p, first)
        s.commit_output(p, output(s, 1))
        simulate_external_resolve_for_trace(s, p)
        p = begin(s, 2)
        initial_pass(s, p, 2)
        expose(s, p)
        s.lifecycle.input_consumed_jitter(p)
        s.acknowledge_prepared_input(p)
        second = head(s, 3)
        second["GATES_code"] = s.buffer("CANDW")
        second["CANDW_code"] = s.buffer("GATES")
        s.commit_head(p, second)
        np.testing.assert_array_equal(s.buffer("GATES"), first["CANDW_code"])
        np.testing.assert_array_equal(s.buffer("CANDW"), first["GATES_code"])

    def test_creation_shapes_formats_and_zeroes(self):
        s = ResourceState((7, 5), (9, 7))
        expected = {"CH[0]": ((12, 15, 4), np.float16),
                    "HID": ((5, 7, 4), np.int8), "PM[1]": ((5, 7, 4), np.float16),
                    "Pe": ((5, 7, 2), np.float16), "Pn": ((5, 7, 2), np.float16),
                    "GATES": ((5, 7, 4), np.uint8), "PREC": ((5, 7, 4), np.float16),
                    "CANDW": ((5, 7, 4), np.uint8), "PH1": ((6, 8, 4), np.uint16),
                    "PH2": ((3, 4, 4), np.uint16), "EX[0]": ((1, 1, 1), np.float16),
                    "Cnet": ((32, 32, 4), np.float16), "Q": ((12, 15, 3), np.float16)}
        for name, (shape, dtype) in expected.items():
            a = s.buffer(name)
            self.assertEqual(a.shape, shape)
            self.assertEqual(a.dtype, dtype)
            self.assertFalse(a.flags.writeable)
            self.assertFalse(np.any(a))
        p = begin(s)
        self.assertTrue(p.reset)
        self.assertEqual((p.history_read, p.history_write, p.exposure_read, p.exposure_write),
                         (0, 1, 1, 0))

    def test_ten_frames_sentinel_parity_and_immediate_exposure(self):
        s = ResourceState((3, 2), (5, 3))
        identities = {k: s.resource_identity(k) for k in PERSISTENT}
        last_exposure = np.float32(1)
        for n in range(1, 11):
            p = begin(s, n)
            if n > 1:
                np.testing.assert_array_equal(s.buffer(f"CH[{p.history_read}]")[..., :3], n - 1)
                np.testing.assert_array_equal(s.buffer(f"PM[{p.history_read}]"), n - 1)
                np.testing.assert_array_equal(s.buffer("HID"), n - 1)
            initial_pass(s, p, n)
            e = expose(s, p, n)
            self.assertEqual(e.Ep, last_exposure)
            self.assertEqual(s.buffer(f"EX[{p.exposure_write}]")[0, 0, 0], e.stored)
            self.assertEqual(s.lifecycle.counter, (n - 1) % 8)
            motion(s, p, n)
            s.lifecycle.input_consumed_jitter(p)
            s.acknowledge_prepared_input(p)
            s.commit_head(p, head(s, n))
            s.commit_output(p, output(s, n))
            with self.assertRaisesRegex(RuntimeError, "including resolve"):
                s.lifecycle.complete_frame(p)
            simulate_external_resolve_for_trace(s, p)
            last_exposure = e.E
            for key in PERSISTENT:
                self.assertIs(s.resource_identity(key), identities[key])
        self.assertEqual((s.lifecycle.parity, s.lifecycle.counter), (0, 2))

    def test_reset_clears_only_ex_at_scheduled_pass(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s)
        through_output(s, p, 4)
        simulate_external_resolve_for_trace(s, p)
        old = {k: s.buffer(k).copy() for k in PERSISTENT}
        p = begin(s, 2, game_reset=True)
        for k in PERSISTENT:
            np.testing.assert_array_equal(s.buffer(k), old[k])
        with self.assertRaises(RuntimeError):
            s.commit_prepared_ph2(p, s.buffer("PH2").copy())
        s.clear_exposure(p)
        for k in PERSISTENT:
            np.testing.assert_array_equal(s.buffer(k), 1 if k.startswith("EX") else old[k])
        expose(s, p, 2)
        s.lifecycle.fail_frame(p, "after reset exposure")
        self.assertFalse(s.lifecycle.pending_reset)  # game flag alone is not latched

    def test_resize_rebuilds_even_same_tensor_extent_and_retains_pools_ex(self):
        s = ResourceState((7, 5), (9, 7))
        p = begin(s)
        through_output(s, p, 5)
        simulate_external_resolve_for_trace(s, p)
        previous_camera, previous_jitter = s.lifecycle.previous_camera, s.lifecycle.previous_jitter
        for n, size in enumerate(((8, 6), (6, 4)), 2):
            old = {k: s.resource_identity(k) for k in PERSISTENT}
            snapshots = {k: s.buffer(k).copy() for k in RETAINED}
            old_cnet = s.resource_identity("Cnet")
            p = begin(s, n, size)
            self.assertEqual(p.tensor_size, (32, 32))
            self.assertEqual(p.reset, n == 3)  # failed resize schedules next reset
            for k in PERSISTENT:
                if k in RETAINED:
                    self.assertIs(s.resource_identity(k), old[k])
                    np.testing.assert_array_equal(s.buffer(k), snapshots[k])
                else:
                    self.assertIsNot(s.resource_identity(k), old[k])
                    self.assertFalse(np.any(s.buffer(k)))
            self.assertIsNot(s.resource_identity("Cnet"), old_cnet)
            self.assertEqual(s.lifecycle.previous_camera, previous_camera)
            self.assertEqual(s.lifecycle.previous_jitter, previous_jitter)
            s.lifecycle.fail_frame(p, "after rebuild")
        self.assertTrue(s.lifecycle.pending_reset)
        self.assertEqual((s.lifecycle.parity, s.lifecycle.counter), (1, 1))

    def test_first_frame_resize_and_failed_first_exposure_remain_reset(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s, size=(4, 2))
        self.assertTrue(p.reset)
        through_output(s, p)
        simulate_external_resolve_for_trace(s, p)
        p = begin(s, 2)
        self.assertTrue(p.reset)
        s.clear_exposure(p)
        e = expose(s, p, 65504)
        self.assertLess(e.stored, np.finfo(np.float16).tiny)
        self.assertGreater(e.stored, 0)
        s.lifecycle.fail_frame(p, "after subnormal exposure store")
        self.assertEqual(s.buffer(f"EX[{p.exposure_write}]")[0, 0, 0], e.stored)
        p = begin(s, 3)
        self.assertTrue(p.reset)
        s.clear_exposure(p)
        for k in ("EX[0]", "EX[1]"):
            np.testing.assert_array_equal(s.buffer(k), 1)

    def test_failed_first_call_and_subnormal_readback_chain(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s)
        s.clear_exposure(p)
        e = expose(s, p, 65504)
        s.lifecycle.fail_frame(p, "first call failed after exposure")
        self.assertTrue(s.lifecycle.pending_reset)
        self.assertEqual((s.lifecycle.parity, s.lifecycle.counter), (0, 0))
        self.assertEqual(s.buffer("EX[0]")[0, 0, 0], e.stored)
        p = begin(s, 2)
        self.assertTrue(p.reset)
        through_output(s, p, 3)
        simulate_external_resolve_for_trace(s, p)
        p = begin(s, 3)
        initial_pass(s, p, 1)
        e = expose(s, p, 65504)
        s.lifecycle.input_consumed_jitter(p)
        s.acknowledge_prepared_input(p)
        s.commit_head(p, head(s, 1))
        s.commit_output(p, output(s, 1))
        simulate_external_resolve_for_trace(s, p)
        p = begin(s, 4)
        initial_pass(s, p, 1)
        next_e = expose(s, p, 1)
        self.assertEqual(next_e.Ep, np.float32(e.stored))
        self.assertLess(next_e.Ep, np.finfo(np.float16).tiny)
        self.assertGreater(next_e.Ep, 0)

    def test_tensor_boundary_resize_and_per_call_scratch_replacement(self):
        s = ResourceState((31, 5), (65, 7))
        for n, size in enumerate(((31, 5), (33, 5), (31, 5)), 1):
            cnet, q = s.resource_identity("Cnet"), s.resource_identity("Q")
            pools = (s.resource_identity("PH1"), s.resource_identity("PH2"))
            p = begin(s, n, size)
            self.assertIsNot(cnet, s.resource_identity("Cnet"))
            self.assertIsNot(q, s.resource_identity("Q"))
            self.assertEqual(s.buffer("Cnet").shape, (32, 64 if n == 2 else 32, 4))
            self.assertEqual(s.buffer("HID").shape, (5, size[0], 4))
            self.assertIs(s.resource_identity("PH1"), pools[0])
            self.assertIs(s.resource_identity("PH2"), pools[1])
            s.lifecycle.fail_frame(p, "no network or resolve execution")

    def test_failure_after_every_stage_retains_writes_and_not_counters_camera(self):
        for stop in ("ph2", "exposure", "motion", "jitter", "input", "head", "output"):
            with self.subTest(stop=stop):
                s = ResourceState((3, 2), (5, 3))
                p = begin(s)
                through_output(s, p, 1)
                simulate_external_resolve_for_trace(s, p)
                camera = s.lifecycle.previous_camera
                p = begin(s, 2)
                actions = (("ph2", lambda: initial_pass(s, p, 9)),
                           ("exposure", lambda: expose(s, p, 9)),
                           ("motion", lambda: motion(s, p, 9)),
                           ("jitter", lambda: s.lifecycle.input_consumed_jitter(p)),
                           ("input", lambda: s.acknowledge_prepared_input(p)),
                           ("head", lambda: s.commit_head(p, head(s, 9))),
                           ("output", lambda: s.commit_output(p, output(s, 9))))
                for label, action in actions:
                    action()
                    if label == stop:
                        break
                snapshot = {k: s.buffer(k).copy() for k in PERSISTENT}
                s.lifecycle.fail_frame(p, "synthetic partial failure")
                self.assertEqual((s.lifecycle.parity, s.lifecycle.counter), (1, 1))
                self.assertEqual(s.lifecycle.previous_camera, camera)
                self.assertFalse(s.lifecycle.pending_reset)
                p2 = begin(s, 3)
                self.assertEqual((p2.history_read, p2.history_write), (1, 0))
                for k in PERSISTENT:
                    np.testing.assert_array_equal(s.buffer(k), snapshot[k])
                consumed = stop in ("jitter", "input", "head", "output")
                self.assertEqual(p2.previous_jitter, ((2 if consumed else 1) / 32,
                                                     -(2 if consumed else 1) / 32))

    def test_origin_discard_preserves_prior_texels_after_partial_failure(self):
        s = ResourceState((3, 2), (5, 3))
        for n in (1, 2):
            p = begin(s, n)
            through_output(s, p, n)
            simulate_external_resolve_for_trace(s, p)
        p = begin(s, 3)
        initial_pass(s, p, 3)
        expose(s, p)
        motion(s, p, 9, origin=(1, 1))
        np.testing.assert_array_equal(s.buffer("PM[1]")[0], 1)
        np.testing.assert_array_equal(s.buffer("PM[1]")[1, 0], 1)
        np.testing.assert_array_equal(s.buffer("PM[1]")[1, 1:], 9)
        np.testing.assert_array_equal(s.buffer("Pe")[0], 3)
        np.testing.assert_array_equal(s.buffer("Pe")[1, 1:], 10)
        snapshot = s.buffer("PM[1]").copy()
        motion(s, p, 8, origin=(99, 99))
        np.testing.assert_array_equal(s.buffer("PM[1]"), snapshot)
        s.lifecycle.input_consumed_jitter(p)
        s.lifecycle.fail_frame(p, "input failed after motion and jitter")
        np.testing.assert_array_equal(s.buffer("PM[1]"), snapshot)

    def test_wrong_stage_stale_ticket_and_invalid_input_never_mutate(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s)
        with self.assertRaises(RuntimeError):
            s.commit_head(p, head(s, 8))
        s.clear_exposure(p)
        expose(s, p)
        with self.assertRaises(RuntimeError):
            s.acknowledge_prepared_input(p)  # no implicit jitter use
        with self.assertRaises(ValueError):
            s.write_prepared_motion(p, np.ones((2, 3, 4), np.float16),
                                    np.full((2, 3, 2), 0.1), np.zeros((2, 3, 2)))
        self.assertFalse(np.any(s.buffer("PM[1]")))
        s.lifecycle.fail_frame(p, "validation checks")
        p2 = begin(s, 2)
        with self.assertRaises(RuntimeError):
            s.clear_exposure(p)
        s.clear_exposure(p2)

    def test_head_and_output_validation_before_any_store(self):
        s = ResourceState((3, 2), (5, 3))
        p = begin(s)
        initial_pass(s, p, 1)
        expose(s, p)
        s.lifecycle.input_consumed_jitter(p)
        s.acknowledge_prepared_input(p)
        for key, bad in (("HID_code", np.zeros((2, 3, 4), np.uint8)),
                         ("Cnet", np.zeros((2, 3, 4), np.float16)),
                         ("PREC", np.full((2, 3, 4), 0.1, np.float32))):
            data = head(s, 6)
            data[key] = bad
            with self.assertRaises(ValueError):
                s.commit_head(p, data)
            self.assertFalse(np.any(s.buffer("HID")))
        s.commit_head(p, head(s, 6))
        valid = output(s, 6)
        for bad in (replace(valid, CH=valid.CH.astype(np.float32) + np.float32(0.1)),
                    replace(valid, Q=valid.Q[:1]),
                    replace(valid, PH1=np.full(valid.PH1.shape, 1024, np.uint16)),
                    replace(valid, PH1=valid.PH1.astype(np.uint32))):
            with self.assertRaises(ValueError):
                s.commit_output(p, bad)
            self.assertFalse(np.any(s.buffer("CH[1]")))
            self.assertFalse(np.any(s.buffer("PH1")))
        s.commit_output(p, valid)
        valid.CH.fill(7)
        np.testing.assert_array_equal(s.buffer("CH[1]")[..., :3], 6)

    def test_actual_head_and_output_interface_reach_only_resolve_boundary(self):
        s = ResourceState((2, 2), (2, 2))
        p = begin(s)
        s.clear_exposure(p)
        expose(s, p)
        motion(s, p, 0)
        s.lifecycle.input_consumed_jitter(p)
        s.acknowledge_prepared_input(p)
        y = np.zeros((32, 32, 32), np.float64)
        # y_a = 0.5, y_b = -0.5 in the head's HID channels: one channel each (55-ERRATUM), or two each, values 0, 1 = y_a
        # and 2, 3 = y_b (55-ERRATUM-2, the default since clean/rr_f/head/head.py gained it; head.py output_images
        # reads both layouts)
        hid = list(head_channels()["hid"])
        y[..., hid[:len(hid) // 2]], y[..., hid[len(hid) // 2:]] = 0.5, -0.5
        result = output_images(y, np.float32(1), np.float32(1), 2, 2)
        result["Cnet"] = np.ones((32, 32, 4), np.float64)
        s.commit_head(p, result)
        np.testing.assert_array_equal(s.buffer("HID"), result["HID_code"])
        out = s.run_output(p, constants=output_symbols())
        np.testing.assert_array_equal(s.buffer("CH[1]"), out.CH)
        np.testing.assert_array_equal(s.buffer("PH1"), out.PH1)
        self.assertTrue(np.any(s.buffer("Q")))
        s.lifecycle.require_pass(p, Pass.PLAIN_RESOLVE)
        with self.assertRaises(RuntimeError):
            s.lifecycle.complete_frame(p)
        self.assertEqual(s.lifecycle.counter, 0)
        s.lifecycle.fail_frame(p, "54b deliberately unavailable")
        np.testing.assert_array_equal(s.buffer("CH[1]"), out.CH)

    def test_exclusions_and_zero_output_resize_rejection_preserve_resources(self):
        a = ResourceState((3, 2), (5, 3), alpha_upscaling=True)
        self.assertEqual(begin(a).passes[-1], Pass.EXTENDED_RESOLVE)
        self.assertEqual(a.resolve_histories.AH[0].shape, (4, 6, 1))
        s = ResourceState((3, 2), (5, 3))
        original = {k: s.resource_identity(k) for k in PERSISTENT}
        p = begin(s, layer_attached=True)
        self.assertEqual(p.passes[-1], Pass.EXTENDED_RESOLVE)
        s.lifecycle.fail_frame(p, "layer frame abandoned")
        with self.assertRaises(RecreateRequired):
            begin(s, output_size=(6, 3))
        with self.assertRaises(ZeroExtentError):
            begin(s, size=(0, 2))
        for k in PERSISTENT:
            self.assertIs(s.resource_identity(k), original[k])
        self.assertTrue(s.lifecycle.pending_reset)
        self.assertTrue(s.exclusions)


if __name__ == "__main__":
    unittest.main()
