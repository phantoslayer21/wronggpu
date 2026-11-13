# SPDX-License-Identifier: MIT
"""CPU-only tests of the exposure result validator; fabricated readbacks, no GPU."""
from dataclasses import replace
from fractions import Fraction
import struct
import unittest

import numpy as np

from clean.rr_f.gpu import exposure as gpu
from clean.rr_f.reference.exposure import auto_exposure


class ExposureComparison(unittest.TestCase):
    def test_exact_zero_sign_and_nan_policy(self):
        self.assertTrue(gpu.within_ulp(np.float32(0), np.float32(0), 0))
        self.assertTrue(gpu.within_ulp(np.float32(-0.), np.float32(-0.), 0))
        self.assertFalse(gpu.within_ulp(np.float32(0), np.float32(-0.), 0))
        self.assertFalse(gpu.within_ulp(np.float32(-0.), np.float32(0), 0))
        # Approximate comparisons intentionally identify signed zeros.
        self.assertTrue(gpu.within_ulp(np.float32(-0.), np.float32(0), 1))
        nan1 = np.uint32(0x7fc00001).view(np.float32)
        nan2 = np.uint32(0xffc00100).view(np.float32)
        self.assertTrue(gpu.within_ulp(nan1, nan2, 0))
        for finite in (np.float32(0), np.float32(1), np.float32(np.inf)):
            self.assertFalse(gpu.within_ulp(nan1, finite, 0))
            self.assertFalse(gpu.within_ulp(finite, nan1, 0))

    def test_infinity_classification_and_neighbor_distance(self):
        for sign in (1, -1):
            inf = np.float32(sign*np.inf)
            self.assertTrue(gpu.within_ulp(inf, inf, 0))
            self.assertFalse(gpu.within_ulp(inf, -inf, 1))
            near = np.float32(sign*np.finfo(np.float32).max)
            self.assertFalse(gpu.within_ulp(inf, near, 1))
            self.assertFalse(gpu.within_ulp(near, inf, 1))
        one = np.float32(1)
        adjacent = np.nextafter(one, np.float32(np.inf))
        two_steps = np.nextafter(adjacent, np.float32(np.inf))
        self.assertFalse(gpu.within_ulp(adjacent, one, 0))
        self.assertTrue(gpu.within_ulp(adjacent, one, 1))
        self.assertFalse(gpu.within_ulp(two_steps, one, 1))

    def test_negative_zero_validator_mutation_and_per_case_symbols(self):
        default = gpu.symbols()
        custom = replace(default, X_KEY=np.float32(.125), X_MFL=np.float32(.25),
                         X_NRM=np.float32(2.0**-126))
        expected = auto_exposure(np.zeros((1,1,3), np.float32), symbols=custom,
                                 render_size=(1,1), Eprev=np.float32(.75), reset=False)
        self.assertEqual(int(expected.X3.view(np.uint32)), 0x80000000)
        image = np.full(100, 0xA5, np.uint8)
        values = np.array([expected.E, expected.Ep, expected.invE, expected.X3,
                           expected.E_raw, expected.mean], np.float32)
        image[:24] = values.view(np.uint8)
        image[24:32] = np.frombuffer(struct.pack('<d', 0), np.uint8)
        image[32:36] = np.frombuffer(struct.pack('<I', int(expected.stored.view(np.uint16))), np.uint8)
        records = [('negative-zero', 0, expected, Fraction(0), False, custom)]
        original = gpu.qualify(image, records, default)[0]
        self.assertTrue(original['passed'], original['rules'])
        # Exactly one field differs. +0 is not the required sign-kept FTZ result.
        image[12:16] = 0
        mutant = gpu.qualify(image, records, default)[0]
        self.assertFalse(mutant['passed'])
        self.assertFalse(mutant['rules']['X3'])
        self.assertTrue(all(v for name,v in mutant['rules'].items() if name != 'X3'))


if __name__ == '__main__':
    unittest.main()
