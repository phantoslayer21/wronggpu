# SPDX-License-Identifier: MIT
"""Frame sequences through the lifecycle (pages 52.8, 53): ResourceState.run_input against the input stage called
with explicitly tracked state (52.10 item 9 shape): first call (pending reset), a normal frame reading the previous
frame's PM, a render-size change (zero-filled images, no reset on the change frame), the following pending reset,
a call failing after its input stage (pj replaced, counters and CP kept, partial writes kept) and the next call."""
import unittest

import numpy as np

from clean.rr_f.reference import input_stage as IS
from clean.rr_f.reference.lifecycle import CameraPair, Pass
from clean.rr_f.reference.resources import ResourceState
from clean.rr_f.tests import synth_input as SI
from clean.rr_f.tests.test_exposure import symbols as exposure_symbols


class TestInputLifecycle(unittest.TestCase):
    def test_sequence(self):
        rng = np.random.default_rng(5500)
        out_size = (40, 33)
        _, _, S, IN0, consts, wc = SI.make_case(rng, Wr=27, Hr=20, Wo=40, Ho=33)
        SH = SI.shift_table(rng)
        raw = (np.float32(0.02), np.float32(-0.03))
        state = ResourceState((27, 20), out_size)
        # independent bookkeeping of the temporal state this test expects
        par, fc, pj, cp = 0, 0, (0.0, 0.0), None
        pm = {0: np.zeros((20, 27, 4), np.float16), 1: np.zeros((20, 27, 4), np.float16)}
        pe, pn = np.zeros((20, 27, 2), np.float16), np.zeros((20, 27, 2), np.float16)
        pending = True
        render = (27, 20)
        steps = [("first", (27, 20), False, True), ("normal", (27, 20), False, True),
                 ("size-change", (30, 22), False, True), ("pending-reset", (30, 22), False, True),
                 ("fail-after-input", (30, 22), False, False), ("after-failure", (30, 22), False, True),
                 ("game-reset", (30, 22), True, True)]
        for name, size, game_reset, completes in steps:
            frame, _, _, _, _, _ = SI.make_case(rng, Wr=size[0], Hr=size[1], Wo=40, Ho=33, smv="Hd")
            V2C, W2V, _, _ = SI.camera_kind(rng, "perspective")
            cam = CameraPair(tuple(V2C.reshape(-1).tolist()), tuple(W2V.reshape(-1).tolist()))
            jitter = (float(np.float32(rng.uniform(-.5, .5))), float(np.float32(rng.uniform(-.5, .5))))
            if size != render:
                render = size
                pm = {0: np.zeros((size[1], size[0], 4), np.float16), 1: np.zeros((size[1], size[0], 4), np.float16)}
                pe, pn = np.zeros((size[1], size[0], 2), np.float16), np.zeros((size[1], size[0], 2), np.float16)
                reset = game_reset or pending                  # the change frame keeps its own reset status
                pending_after = True
            else:
                reset = game_reset or pending
                pending_after = False
            plan = state.begin_frame(size, jitter=jitter, camera=cam, game_reset=game_reset)
            self.assertEqual((plan.parity, plan.counter, plan.reset), (par, fc, reset), name)
            self.assertEqual(tuple(plan.previous_jitter), pj, name)
            if Pass.CLEAR_EX in plan.passes:
                state.clear_exposure(plan)
            else:
                # stand-in PH2 codes (the 54b producer is another track's); the input stage reads what is stored
                codes = rng.integers(0, 1024, state.buffer("PH2").shape).astype(np.uint16)
                codes[..., 3] = 3
                state.commit_prepared_ph2(plan, codes)
            colour = frame.textures["C"]
            ex = state.run_exposure(plan, colour.data, symbols=exposure_symbols(), origin=colour.origin)
            hid, ph1, ph2 = state.buffer("HID"), state.buffer("PH1"), state.buffer("PH2")
            res = state.run_input(plan, frame.textures, S=S, IN0=IN0, consts=consts, wc=wc, SH=SH, mv_scale_raw=raw)
            # the same stage with the explicitly tracked state
            prevcam = (V2C, W2V) if reset else cp
            params = IS.FrameParams(render_size=size, output_size=out_size, jitter=tuple(np.float32(jitter)),
                                    prev_jitter=tuple(np.float32(pj)), mv_scale_raw=raw, mv_jittered=False,
                                    E=ex.E, Ep=ex.Ep, X3=ex.X3, reset=reset, par=par, fc=fc, SH=SH, V2C=V2C,
                                    W2V=W2V, V2Cp=prevcam[0], W2Vp=prevcam[1])
            explicit = IS.InputFrame(frame.textures, PH1=ph1, PH2=ph2, HID=hid, PM_read=pm[par], PM_write=pm[1 - par],
                                     Pe=pe, Pn=pn)
            ref = IS.input_stage(explicit, params, S, IN0, consts, wc)
            for key in ("codes", "x0", "y2", "hnew", "hmerged"):
                self.assertTrue(np.array_equal(res[key], ref[key], equal_nan=True), (name, key))
            pm[1 - par], pe, pn = ref["surfaces"]["PM"], ref["surfaces"]["Pe"], ref["surfaces"]["Pn"]
            for key, img in ((f"PM[{1 - par}]", pm[1 - par]), ("Pe", pe), ("Pn", pn), (f"PM[{par}]", pm[par])):
                self.assertTrue(np.array_equal(state.buffer(key).view(np.uint16), img.view(np.uint16)), (name, key))
            pj = tuple(np.float32(jitter).tolist())          # replaced once the input stage used it (53.5)
            if completes:
                for p in plan.passes[plan.passes.index(Pass.INPUT) + 1:]:
                    state.lifecycle.acknowledge_pass(plan, p)
                state.lifecycle.complete_frame(plan)
                par, fc, cp = 1 - par, (fc + 1) % 8, (V2C, W2V)
                if reset:
                    pending = False
                if pending_after:
                    pending = True
            else:
                state.lifecycle.fail_frame(plan, "test failure after the input stage")
                if pending_after:
                    pending = True
        self.assertEqual((par, fc), (state.lifecycle.parity, state.lifecycle.counter))


if __name__ == "__main__":
    unittest.main()
