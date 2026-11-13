# SPDX-License-Identifier: MIT
"""CPU structural checks of real body/head -> output arena preparation; no GPU."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from clean.rr_f.gpu import pipeline as gpu


class ShapeOnly:
    """A bound head input may expose dimensions but cannot supply CPU payload."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.size = int(np.prod(shape))

    def __array__(self, *args, **kwargs):
        raise AssertionError("Bound GPU head data must not be converted or uploaded")


class ConnectedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ExitStack() as stack:
            stack.enter_context(patch.object(gpu.network.stage, 'precompile'))
            stack.enter_context(patch.object(gpu.network.stage, 'kernel_for', return_value='unused.dxil'))
            stack.enter_context(patch.object(gpu.build, 'compile_kernel', return_value='unused.dxil'))
            stack.enter_context(patch.object(gpu.build.build_dx12, 'compile_hlsl'))
            stack.enter_context(patch.object(gpu.build.build_dx12, 'build_asm_tool', return_value='unused.exe'))
            cls.body = gpu.network.prepare()
            original, kernels, dispatches, _, reference, cls.base, cls.ha, _, _ = cls.body
            cls.original_image = original.image()
            cls.inputs = gpu.output_inputs(reference['head'])
            cls.branches = {}
            for mutation in (False, True):
                # Independent arenas share existing immutable upload arrays; only
                # new output allocations differ. Preparation itself is real.
                arena = gpu.build.Arena()
                arena.parts, arena.size = list(original.parts), original.size
                ks, ds = dict(kernels), list(dispatches)
                a = dict(cls.inputs)
                for name in gpu.HEAD_LINKS:
                    a[name] = ShapeOnly(a[name].shape)
                before = len(arena.parts)
                appended = gpu.append_output(arena, ks, ds, a, cls.base, cls.ha, mutation)
                cls.branches[mutation] = arena, ks, ds, appended, before

    def test_actual_dispatch_consumes_head_outputs_with_packed_formats(self):
        arena, _, dispatches, (args, _, _, expected, bound), _ = self.branches[False]
        # 16 body / head dispatches (+ the regroup unless E1 reads IN0's y2 directly) + output main / pool
        self.assertEqual(len(dispatches), 18 + (not gpu.network.stage.e1_direct(step="X")))
        self.assertEqual([d[0] for d in dispatches[-3:]], ['D0-R', 'output-main', 'output-pool'])
        head_args = next(d[2] for d in dispatches if d[0] == 'D0-R')
        # These indices are the head ABI, independently of HEAD_LINKS metadata.
        self.assertEqual(args[:4], [head_args[8], head_args[10], head_args[11], head_args[12]])
        self.assertEqual(args[:4], [self.base+self.ha.ptr[k] for k in ('cnet','gates','prec','candw')])
        self.assertEqual(args[:4], [bound[k] for k in ('Cnet','G','P','W4')])
        self.assertEqual(expected, bound)
        self.assertEqual(args[37], 1)                 # packed_head: packed head inputs
        self.assertEqual(len(args), 39)               # + the fast path's scratch area (output.prepare packed)
        self.assertEqual(args[14:19], [29,19,32,45,30])
        self.assertEqual(dispatches[-2][1], (45*30+31)//32)
        self.assertEqual(dispatches[-1][1], (23*15+31)//32)
        self.assertIs(dispatches[-1][2], args)
        self.assertIs(dispatches[-2][2], args)

    def test_bound_head_arrays_are_shape_only_and_producers_unchanged(self):
        for mutation, (arena, _, dispatches, _, before) in self.branches.items():
            with self.subTest(mutation=mutation):
                # Four motion/history inputs, table, five outputs and the fast path's scratch area. The
                # mutant adds exactly one zero Cnet allocation, not a CPU reference upload.
                self.assertEqual(len(arena.parts)-before, 11+int(mutation))
                np.testing.assert_array_equal(arena.image()[:len(self.original_image)], self.original_image)
                self.assertEqual(dispatches[:-2], self.body[2])

    def test_cnet_disconnect_changes_only_selected_head_edge(self):
        _, _, normal_dispatches, normal, _ = self.branches[False]
        arena, _, mutant_dispatches, mutant, _ = self.branches[True]
        normal_args, _, _, expected, _ = normal
        args, _, _, mutant_expected, bound = mutant
        self.assertEqual(mutant_expected, expected)
        self.assertNotEqual(bound, expected)
        self.assertNotEqual(args[0], normal_args[0])
        self.assertEqual(args[1:4], normal_args[1:4])
        self.assertEqual(bound['Cnet'], args[0])
        self.assertEqual(mutant_dispatches[:-2], normal_dispatches[:-2])
        nbytes = self.inputs['Cnet'].size*2
        self.assertTrue(np.all(arena.image()[args[0]:args[0]+nbytes] == 0))
        self.assertEqual({k for k in bound if bound[k] != expected[k]}, {'Cnet'})

    def test_readback_uses_packed_half_and_byte_layouts_then_half_unorm_decode(self):
        image = self.branches[False][0].image()
        cnet = ((np.arange(32*32*4)%2048)/32).astype('<f2').reshape(32,32,4)
        prec = ((np.arange(19*29*4)%1024)/16).astype('<f2').reshape(19,29,4)
        gates = (np.arange(19*29*4)%256).astype(np.uint8).reshape(19,29,4)
        candw = (255-np.arange(19*29*4)%256).astype(np.uint8).reshape(19,29,4)
        for name, value in (('cnet',cnet),('prec',prec),('gates',gates),('candw',candw)):
            raw = value.view(np.uint8).ravel()
            off = self.base+self.ha.ptr[name]
            image[off:off+raw.size] = raw
        head = gpu.head_from_gpu(image, self.base, self.ha)
        for key, value in (('Cnet',cnet),('PREC',prec),('GATES_code',gates),('CANDW_code',candw)):
            self.assertEqual(head[key].dtype, value.dtype)
            np.testing.assert_array_equal(head[key], value)
            self.assertFalse(np.shares_memory(head[key], image))
        a = gpu.output_inputs(head)
        np.testing.assert_array_equal(a['G'], (gates.astype(np.float64)/255).astype(np.float16)[...,:3])
        np.testing.assert_array_equal(a['W4'], (candw.astype(np.float64)/255).astype(np.float16))
        np.testing.assert_array_equal(a['P'], prec[...,:3])
        self.assertEqual(a['Cnet'].shape, (32,32,4))
        # At least one code distinguishes required half decode from direct f32.
        direct = gates.astype(np.float32)/np.float32(255)
        self.assertTrue(np.any(a['G'].astype(np.float32) != direct[...,:3]))


if __name__ == '__main__':
    unittest.main()
