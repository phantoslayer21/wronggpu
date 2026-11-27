# SPDX-License-Identifier: GPL-3.0-only
"""The Model K port check's launch-plan bridge (clean/k/dx12/k_plan_bridge.py), CPU only: pointer fields of the manifest's
struct argument token, the pointers' buffer and offset, and the kernel_replay chain steps made from a stage launch.

Run: python -m unittest clean.k.tests.test_k_plan_bridge
"""
import ctypes
import os
import tempfile
import unittest

from clean.k.dx12 import k_plan_bridge as PB
from clean.k.runner.k_launch import KBlkParams


class PlanBridge(unittest.TestCase):
    def test_struct_pointer_offsets(self):
        self.assertEqual(PB.struct_pointer_offsets("s184:p19.b32"), [8 * i for i in range(19)])
        self.assertEqual(PB.struct_pointer_offsets("s40:p2.b8.x8.p1"), [0, 8, 32])
        self.assertEqual(PB.struct_pointer_offsets("s160:p13.b56"), [8 * i for i in range(13)])

    def test_locate_and_ref(self):
        bufs = {"in": (0x1000, 256), "weights": (0x4000, 1024)}
        self.assertEqual(PB.locate(0x1000, bufs), ("in", 0))
        self.assertEqual(PB.locate(0x4000 + 96, bufs), ("weights", 96))
        self.assertEqual(PB.locate(0x1000 + 256 + PB.SLACK - 1, bufs), ("in", 256 + PB.SLACK - 1))
        with self.assertRaises(ValueError):
            PB.locate(0x3000, bufs)
        self.assertEqual(PB.ref("in", 0), "@b_in")
        self.assertEqual(PB.ref("weights", 96), "@b_weights+96")

    def test_chain_step_of_a_block_launch(self):
        assert ctypes.sizeof(KBlkParams) == 184
        bufs = {"in": (0x1000, 256), "weights": (0x4000, 1024), "block": (0x9000, 512)}
        p = KBlkParams(x=0x1000, y=0x9000, wqkv=0x4000 + 96, w=24, h=16)
        with tempfile.TemporaryDirectory() as d:
            line = PB.chain_step("k_stage_e2", 12, 128, [p], ["s184:p19.b32"], bufs, d, "case.l0")
            self.assertTrue(line.startswith("step k_stage_e2 12 128 S:case.l0.a0:"))
            patch = dict(item.split("=") for item in line.split(":", 2)[2].split(","))
            self.assertEqual(patch["0"], "@b_in")                  # x
            self.assertEqual(patch["16"], "@b_block")              # y
            self.assertEqual(patch["56"], "@b_weights+96")         # wqkv
            self.assertEqual(len(patch), 3)                        # null pointers are not patched
            with open(os.path.join(d, "case.l0.a0"), "rb") as f:
                self.assertEqual(f.read(), bytes(p))
            # scalars, a null pointer, a pointer into a buffer and a struct by value
            line = PB.chain_step("k", 1, 32, [ctypes.c_void_p(0), ctypes.c_void_p(0x4000 + 8), ctypes.c_int(-3),
                                              ctypes.c_float(1.0), (ctypes.c_ubyte * 4)(1, 2, 3, 4)],
                                 ["p", "p", "u", "u", "b4"], bufs, d, "case.l1")
            self.assertEqual(line, "step k 1 32 null @b_weights+8 4294967293 1065353216 B:case.l1.a4")

    def test_compact_head_rows_cover_pixel_and_hidden_channels(self):
        rows = PB.head_rows()
        self.assertEqual(len(rows), 40)
        used = {10 * slot + offset for slot in range(4) for offset in (0, 1, 3, 4, 5, 6, 7, 9)}
        self.assertEqual(sorted(rows[ch] for ch in used), list(range(32)))
        self.assertTrue(all(rows[ch] == -1 for ch in set(range(40)) - used))
        for half in range(2):
            slot = 10 * half
            channels = (slot, slot + 1, slot + 9, slot + 20, slot + 21, slot + 29)
            self.assertEqual([rows[ch] for ch in channels], list(range(8 * half, 8 * half + 6)))

    def test_argument_count_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                PB.chain_step("k", 1, 32, [ctypes.c_int(1)], ["u", "u"], {}, d, "x")


if __name__ == "__main__":
    unittest.main()
