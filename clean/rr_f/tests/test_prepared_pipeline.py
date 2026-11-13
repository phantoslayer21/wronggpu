# SPDX-License-Identifier: MIT
"""Prepared-input integration only; missing input construction/resolve are not stubbed as success."""
import unittest
import numpy as np

from clean.rr_f.gpu.network import fixture
from clean.rr_f.gpu.exposure import symbols as exposure_symbols
from clean.rr_f.reference import exposure, network, output
from clean.rr_f.tests.test_output import symbols as output_symbols


class PreparedPipeline(unittest.TestCase):
    def test_exposure_and_real_head_to_history_outputs(self):
        tokens,state,weights,constants,wc,shifts=fixture(551)
        # These tokens and motion textures are explicitly supplied synthetic inputs.
        # They do not model page52's transformation of the colour/guide images.
        colour=np.full((2,2,3),.5,np.float32)
        ex=exposure.auto_exposure(colour,symbols=exposure_symbols(),render_size=(2,2),Eprev=0,reset=True)
        body=network.run_body(tokens,state,weights,constants,wc,shifts,s1=1.25,s2=.75,render_size=(2,2))
        head=body['head']
        inputs=dict(Cnet=head['Cnet'],G=output.decode_unorm8(head['GATES_code'])[...,:3],
                    P=head['PREC'][...,:3],W4=output.decode_unorm8(head['CANDW_code']),
                    PM=np.zeros((2,2,4),np.float16),Pe=np.zeros((2,2,2),np.float16),
                    Pn=np.zeros((2,2,2),np.float16),Hprev=np.zeros((3,3,4),np.float16),
                    output_size=(2,2),jitter=(.125,-.125),E=ex.E,Ep=ex.Ep,invE=ex.invE,
                    reset=True,constants=output_symbols())
        actual=output.output_stage(**inputs)
        self.assertEqual(actual.CH.shape,(3,3,4))
        self.assertEqual(actual.Q.shape,(3,3,3))
        self.assertEqual(actual.PH1.shape,(2,2,4))
        self.assertEqual(actual.CH.dtype,np.float16)
        self.assertEqual(actual.Q.dtype,np.float16)
        self.assertTrue(np.all(np.isfinite(actual.CH)))
        self.assertTrue(np.all(np.isfinite(actual.Q)))
        self.assertTrue(np.all(actual.CH[...,3]==0))
        self.assertTrue(np.all(actual.PH1[...,3]==3))
        # Real head stores feed the UNORM readback seam; previous-history contents
        # remain irrelevant on reset even after all these independently built stages.
        inputs['Hprev']=np.full((3,3,4),np.nan,np.float16)
        poisoned=output.output_stage(**inputs)
        for field in ('CH','Q','PH1'):
            np.testing.assert_array_equal(getattr(actual,field),getattr(poisoned,field))


if __name__=='__main__': unittest.main()
