# SPDX-License-Identifier: MIT
"""CPU inspection of the real body arena and dispatch graph; no GPU or compiler runs."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from clean.rr_f.gpu import network as gpu
from clean.rr_f.reference import network
from clean.rr_f.reference.stages import stage_info


class BodyDispatchGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.allocated_inputs = {}
        real_stage_call = gpu.stage.StageCall

        # Instrument the actual constructor before prepare overwrites its pointers.
        # All weight packing, allocation, reference computation and head jobs are real.
        def record_call(name, inputs, *args, **kwargs):
            call = real_stage_call(name, inputs, *args, **kwargs)
            cls.allocated_inputs[name] = (dict(inputs), tuple(call.args[:4]))
            return call

        with ExitStack() as stack:
            stack.enter_context(patch.object(gpu.stage, "precompile"))
            stack.enter_context(patch.object(gpu.stage, "kernel_for", return_value="unused.dxil"))
            stack.enter_context(patch.object(gpu.build, "compile_kernel", return_value="unused.dxil"))
            stack.enter_context(patch.object(gpu.build.build_dx12, "compile_hlsl"))
            stack.enter_context(patch.object(gpu.build.build_dx12, "build_asm_tool", return_value="unused.exe"))
            stack.enter_context(patch.object(gpu.stage, "StageCall", side_effect=record_call))
            cls.prepared = gpu.prepare()
        cls.arena, _, cls.dispatches, cls.calls, _, cls.base, cls.ha, _, cls.regroup = cls.prepared
        cls.image = cls.arena.image()
        cls.dispatch_by_name = {name: (groups, args) for name, groups, args in cls.dispatches}

    def test_dispatch_order_and_all_producer_bindings(self):
        # E1 on rr_block_fast.hlsl reads IN0's y2 directly (gpu/stage.py e1_direct): no regroup dispatch
        direct = gpu.stage.e1_direct(step="X")
        self.assertEqual(self.regroup is None, direct)
        expected_order = ["IN0", *([] if direct else ["regroup"]), *network.ORDER[1:]]
        self.assertEqual([d[0] for d in self.dispatches], expected_order)
        offset = lambda name, output: self.calls[name].out[output][0]
        for name in network.ORDER[1:-1]:
            with self.subTest(stage=name):
                level = stage_info(name)["level"]
                if name.startswith("E"):
                    parent = "IN0" if level == 1 else f"E{level-1}"
                    first = offset("IN0", "y2") if direct else self.regroup
                    expected = {0: first if level == 1 else offset(parent, "merged"),
                                2: offset(parent, "hmerged")}
                elif name.endswith("-F"):
                    parent = "E5" if level == 4 else f"D{level+1}-R"
                    enc = "IN0" if level == 0 else f"E{level}"
                    expected = {0: offset(enc, "y2"), 1: offset(parent, "y2")}
                else:
                    parent = "E5" if level == 4 else f"D{level+1}-R"
                    expected = {0: offset(f"D{level}-F", "y2"), 2: offset(f"E{level}", "hnew"),
                                3: offset(parent, "hnew")}
                groups, args = self.dispatch_by_name[name]
                self.assertEqual(groups, self.calls[name].groups())
                for index, expected_pointer in expected.items():
                    self.assertEqual(args[index], expected_pointer)
                    self.assertNotEqual(args[index], self.allocated_inputs[name][1][index])
        head_args = self.dispatch_by_name["D0-R"][1]
        for index, producer, output in ((0, "D0-F", "y2"), (4, "IN0", "hnew"), (5, "D1-R", "hnew")):
            self.assertEqual(head_args[index], offset(producer, output))
        if not direct:
            groups, args = self.dispatch_by_name["regroup"]
            self.assertEqual(args, [offset("IN0", "y2"), self.regroup, 32, 32])
            self.assertEqual(groups, 8)

    def test_every_unbound_stage_input_is_poison_in_arena(self):
        indices = {"x": 0, "in": 0, "skip": 0, "cin": 1, "hold": 2, "Hc": 3}
        checked = 0
        for name, (inputs, pointers) in self.allocated_inputs.items():
            if name == "IN0":
                continue
            for key, array in inputs.items():
                with self.subTest(stage=name, input=key):
                    off = pointers[indices[key]]
                    if array.dtype == np.uint8:
                        actual = self.image[off:off + array.size]
                        self.assertTrue(np.all(actual == 0x7f))
                    else:
                        actual = self.image[off:off + array.size * 2].copy().view("<f2")
                        self.assertTrue(np.all(np.isnan(actual)))
                    checked += 1
        self.assertEqual(checked, 32)

    def test_head_pointer_relocation_and_unbound_poison(self):
        args = self.dispatch_by_name["D0-R"][1]
        for index, name in enumerate(gpu.head.ARGS):
            if not name.startswith("p_") or index in (0, 4, 5):
                continue
            expected = self.base + self.ha.ptr[name[2:]] if name[2:] in self.ha.ptr else 0
            self.assertEqual(args[index], expected, name)
        for name, size, dtype in (("x", 32*32*32, np.uint8), ("hold", 32*32*4, "<f2"),
                                  ("hc", 16*16*4, "<f2")):
            off = self.base + self.ha.ptr[name]
            raw = self.image[off:off + size*np.dtype(dtype).itemsize].copy().view(dtype)
            self.assertTrue(np.all(raw == 0x7f) if name == "x" else np.all(np.isnan(raw)), name)


    def test_candw_equal_logits_rejects_tolerated_one_step_error(self):
        # Fabricated readback isolates the validator, without invoking a GPU.
        # Keep every checked field valid, then perturb only CANDW by one f16 step.
        img = self.image.copy()
        reference = self.prepared[4]
        reference = {**reference, "head": dict(reference["head"])}
        reference["head"]["CANDW"] = np.full((19, 29, 4), .25, np.float64)

        def write(off, value, dtype):
            data = np.asarray(value, dtype=dtype).view(np.uint8).ravel()
            img[off:off+data.size] = data

        for name, call in self.calls.items():
            for key, (off, _, typ) in call.out.items():
                write(off, reference["stages"][name][key], np.uint8 if typ == "e4m3" else "<f2")
        if self.regroup is not None:      # (no regroup buffer when E1 reads IN0's y2 directly)
            write(self.regroup, network.regroup_children(reference["stages"]["IN0"]["y2"]), np.uint8)
        ho = reference["head"]
        for name, key in (("cnet", "Cnet"), ("prec", "PREC")):
            write(self.base+self.ha.ptr[name], ho[key], "<f2")
        values = np.stack([ho["HID"], ho["GATES"], ho["CANDW"]])
        write(self.base+self.ha.ptr["dimg"], values, "<f2")
        write(self.base+self.ha.ptr["dy"], np.zeros((32,32,32)), "<f2")
        for name, key, encode in (("hid", "HID", gpu.fp.snorm8_store),
                                  ("gates", "GATES", gpu.fp.unorm8_store),
                                  ("candw", "CANDW", gpu.fp.unorm8_store)):
            codes = encode(ho[key])
            write(self.base+self.ha.ptr[name], codes, codes.dtype)
        baseline = gpu.analyze(img, self.calls, reference, self.base, self.ha, self.regroup)
        self.assertTrue(baseline["passed"])
        values[2] = np.nextafter(np.float16(.25), np.float16(np.inf))
        write(self.base+self.ha.ptr["dimg"], values, "<f2")
        write(self.base+self.ha.ptr["candw"], gpu.fp.unorm8_store(values[2]), np.uint8)
        mutated = gpu.analyze(img, self.calls, reference, self.base, self.ha, self.regroup)
        self.assertEqual(mutated["head"]["CANDW"]["over_tol"], 0)
        self.assertEqual(mutated["head_store_mismatches"]["CANDW"], 0)
        self.assertEqual(mutated["candw_pair_mismatches"], 0)
        self.assertEqual(mutated["candw_equal_logits_bad"], 19*29)
        self.assertFalse(mutated["passed"])

if __name__ == "__main__":
    unittest.main()
