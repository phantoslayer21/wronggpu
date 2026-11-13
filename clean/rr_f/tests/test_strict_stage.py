# SPDX-License-Identifier: MIT
"""Independent layout mapping and strict qualification verdict regressions."""
import unittest
import numpy as np
from clean.rr_f.gpu import layout, strict_stage


class StrictStageTests(unittest.TestCase):
    def test_every_operand_coordinate_mapping(self):
        # Coordinates encode unique integers; repeat the test on each byte so
        # the layout check is not ambiguous after uint8 truncation.
        coords = np.arange(16 * 32).reshape(16, 32)
        for shift in (0, 8):
            codes = ((coords >> shift) & 255).astype(np.uint8)
            fragments = layout.frags(codes)[0]
            for row in range(16):
                for k in range(32):
                    lane = 16 * ((k % 16) // 8) + row
                    self.assertEqual(fragments[k // 16, lane, k % 8], codes[row, k])

    def test_every_accumulator_coordinate_mapping(self):
        matrix = np.arange(256).reshape(16, 16)
        packed = strict_stage.acc_pack(matrix)
        for lane in range(32):
            for v in range(8):
                self.assertEqual(packed[lane, v], matrix[8 * (lane >> 4) + v, lane & 15])
        np.testing.assert_array_equal(strict_stage.acc_unpack(packed), matrix)


if __name__ == '__main__':
    unittest.main()
