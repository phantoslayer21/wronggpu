# SPDX-License-Identifier: MIT
"""CPU-only regressions for the synthetic D3D12 output validator; never launches GPU."""
import unittest
from unittest.mock import patch

import numpy as np
from clean.rr_f.gpu import output as gpu
from clean.rr_f.reference import output as ref
from clean.rr_f.tests.test_output import inputs, symbols


class QualificationTests(unittest.TestCase):
    def test_storage_equality_preserves_sign_of_zero(self):
        self.assertFalse(gpu.same_storage(np.array([0],np.float16),np.array([-0.0],np.float16)))
        self.assertTrue(gpu.same_storage(np.array([0],np.float16),np.array([0],np.float16)))

    def test_impossible_rgb10_code_rejected_even_within_one_code_tolerance(self):
        a=inputs(); a['reset']=True; a['Cnet'][...,:3]=255; a['constants']=symbols(K11=1)
        captured=[]; pool=ref.pool_history
        def capture(linear,c):
            captured.append(linear.copy()); return pool(linear,c)
        with patch.object(ref,'pool_history',side_effect=capture):
            expected,trace=gpu.oracle_with_trace(a)
        self.assertTrue(np.all(expected.PH1[...,:3]==1023))
        arena=gpu.build.Arena(); _,offs,sizes=gpu.prepare(arena,a)
        img=arena.image()
        for offset,value in zip(offs,[captured[0].astype('<f4'),expected.CH,expected.Q,expected.PH1,trace]):
            data=np.ascontiguousarray(value).view(np.uint8).ravel(); img[offset:offset+data.size]=data
        records=[('saturated',a,offs,sizes,expected,trace)]
        rows,_=gpu.qualify(img,records,reset_pairs=()); self.assertTrue(rows[0]['passed'])
        corrupt=expected.PH1.copy(); corrupt[...,:3]=1024
        img[offs[3]:offs[3]+sizes[3]]=corrupt.view(np.uint8).ravel()
        rows,_=gpu.qualify(img,records,reset_pairs=())
        self.assertTrue(rows[0]['comparison']['PH1']['passed'])
        self.assertTrue(rows[0]['comparison']['PH1_from_GPU_linear']['passed'])
        self.assertFalse(rows[0]['exact']['PH1_RGB_range'])
        self.assertFalse(rows[0]['passed'])

    def test_nonfinite_classification_cannot_pass_half_tolerance(self):
        pairs=[(np.inf,65504),(65504,np.inf),(-np.inf,np.inf),(np.nan,0),(0,np.nan)]
        for actual,expected in pairs:
            with self.subTest(actual=actual,expected=expected):
                self.assertFalse(gpu.tolerated_half(actual,expected))
        self.assertTrue(gpu.tolerated_half(np.nan,np.nan))
        self.assertTrue(gpu.tolerated_half(np.inf,np.inf))

    def test_underflow_regression_is_exact_oracle_case(self):
        v,expected=gpu.arithmetic_cases()
        bits=v.view(np.uint32)
        for sign in (0,0x80000000):
            which=np.flatnonzero((bits[:,0]==(0x00800001|sign))&(bits[:,1]==0x3f7ffffd)&(bits[:,2]==0))
            self.assertEqual(len(which),1)
            self.assertEqual(int(expected[which[0],0]),sign)
            self.assertEqual(int(expected[which[0],1]),sign)

    def test_gpu_input_binding_uploads_no_placeholder_head_data(self):
        a=inputs(); arena=gpu.build.Arena()
        bindings={n:arena.reserve(64,fill=0xCD) for n in ('Cnet','G','P','W4')}
        for n in bindings: a[n][:]=np.nan
        before=len(arena.parts)
        args,_,_=gpu.prepare(arena,a,packed=True,input_bindings=bindings)
        self.assertEqual(args[:4],[bindings[n] for n in ('Cnet','G','P','W4')])
        # Four motion/history inputs, one TR table, five outputs, the fast path's scratch; no head upload.
        self.assertEqual(len(arena.parts)-before,11)
        self.assertEqual(args[37],1)
        self.assertNotEqual(args[38],0)


if __name__=='__main__': unittest.main()
