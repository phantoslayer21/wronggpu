# SPDX-License-Identifier: GPL-3.0-only
"""The FP16-form block kernels' blob and tables (clean/l/runner/l_pack.py pack_q7_block, q7_tables): layout, values
and the agreement of the tables with the oracle's functions for every FP16 input.

    python -m unittest clean.l.tests.test_l_q7_pack
"""
import unittest

import numpy as np

from clean.l.reference import oracle as L
from clean.l.runner import l_pack as LP
from clean.reference.fp8 import f16, fp8_decode, q8


class Q7PackTest(unittest.TestCase):
    def test_blob_layout_and_values(self):
        rng = np.random.default_rng(3)
        p = {k: np.asarray(v, np.float64) for k, v in L.random_block_params(rng, 32, 2, 128).items()}
        blob = LP.pack_q7_block(p, 32, 2, 128)
        o = LP.q7_block_offsets(32, 2, 128)
        self.assertEqual(len(blob) % LP.ALIGN, 0)
        self.assertEqual((o["end8"], o["end16"], o["end"]), (15872, 44544, 46336))
        for name in ("Wqk", "Wv", "Wp", "W1", "W2"):                # the sum |w| tables (Q7F-2), exact
            W = p[name].reshape(-1, p[name].shape[-1])
            got = np.frombuffer(blob[o[name + "_wsum"]:o[name + "_wsum"] + 4 * W.size // 32], np.float32)
            want = np.abs(q8(W)).reshape(W.shape[0], -1, 32).sum(-1).T.reshape(-1)      # chunk-major [K/32][rows]
            np.testing.assert_array_equal(got.astype(np.float64), want)
        self.assertEqual((o["Wqk"], o["W2"]), (1536, 11776))     # the constants of clean/l/kernels/l_q7.hip
        self.assertEqual((o["Wqk16"], o["Wv16"], o["Wp16"], o["W116"], o["W216"]),
                         (15872, 19968, 24064, 28160, 36352))   # and of its FP16 section (Q7F-1)

        def f16s(name, n):
            return np.frombuffer(blob[o[name]:o[name] + 2 * n], np.float16).astype(np.float64)

        def fp8s(name, shape):
            n = int(np.prod(shape))
            return fp8_decode(np.frombuffer(blob[o[name]:o[name] + n], np.uint8)).astype(np.float64).reshape(shape)

        np.testing.assert_array_equal(f16s("g1", 32), L.gamma_q7(p["gamma1"], 32))
        np.testing.assert_array_equal(f16s("g2", 32), L.gamma_q7(p["gamma2"], 32))
        np.testing.assert_array_equal(f16s("bp", 32), f16(p["bp"]))
        np.testing.assert_array_equal(f16s("b1", 128), f16(p["b1"]))
        np.testing.assert_array_equal(f16s("B", 512).reshape(2, 16, 16), f16(p["B"]))
        for name in ("Wqk", "Wv", "Wp", "W1", "W2"):
            np.testing.assert_array_equal(fp8s(name, p[name].shape), q8(p[name]))
            n = p[name].size
            h = np.frombuffer(blob[o[name + "16"]:o[name + "16"] + 2 * n], np.float16).astype(np.float64)
            np.testing.assert_array_equal(h.reshape(p[name].shape), q8(p[name]))   # the FP16 copy is exact
        lin = LP.pack_q7_linear(p["W1"], p["b1"])
        lo = LP.q7_linear_offsets(128, 32)
        np.testing.assert_array_equal(np.frombuffer(lin[lo["W16"]:lo["W16"] + 2 * 128 * 32], np.float16)
                                      .astype(np.float64).reshape(128, 32), q8(p["W1"]))
        with self.assertRaises(ValueError):
            LP.pack_q7_block(dict(p, W1=p["W1"][:64]), 32, 2, 128)

    @staticmethod
    def _fp16_copy_matches(copy_bits, codes):
        """Adversarial probe of the FP16 weight copy (Q7F-1): for every FP8 E4M3 code, the copy's bits are the FP16
        bits of the code's exact value (+-0 with its sign, the subnormal codes, the largest 448), and the two NaN codes
        give NaN. Model L has E4M3 weights only (no E5M2 anywhere in the network)."""
        vals = fp8_decode(codes).astype(np.float64)
        want = vals.astype(np.float16).view(np.uint16)
        h = np.asarray(copy_bits, np.uint16)
        nan = np.isnan(vals)
        hnan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
        return bool(np.all(h[~nan] == want[~nan]) and np.all(hnan[nan]))

    def test_fp16_copy_exact_for_every_fp8_code(self):
        codes = np.arange(256, dtype=np.uint8)
        W = fp8_decode(codes).astype(np.float64).reshape(8, 32)
        lin = LP.pack_q7_linear(W, np.zeros(8))
        o = LP.q7_linear_offsets(8, 32)
        h = np.frombuffer(lin[o["W16"]:o["W16"] + 2 * 256], np.uint16)
        self.assertTrue(self._fp16_copy_matches(h, codes))
        self.assertEqual((int(h[0x80]), int(h[0x01]), int(h[0x7E])), (0x8000, 0x1800, 0x5F00))   # -0, 2^-9, 448
        # mutant the probe must catch: a bit-shift conversion (rebias the exponent, shift the mantissa) that is right
        # for normal codes but not for the subnormal ones
        c = codes.astype(np.uint32)
        mutant = (((c & 0x80) << 8) | (((c & 0x7F) << 7) + (8 << 10))).astype(np.uint16)
        self.assertFalse(self._fp16_copy_matches(mutant, codes))

    def test_tables_equal_the_oracle_for_every_fp16_input(self):
        tab = np.frombuffer(LP.q7_tables(), np.float16).astype(np.float64)
        self.assertEqual(tab.size, 2 * 65536)
        x = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
        with np.errstate(all="ignore"):
            rs, rc = f16(L._rsqrt32(x)), f16(L._rcp32(x))
        for got, want in ((tab[:65536], rs), (tab[65536:], rc)):
            same = (got == want) | (np.isnan(got) & np.isnan(want))
            self.assertTrue(same.all())
        self.assertEqual(tab[0x5000], f16(L._rsqrt32(np.array([32.0])))[0])   # S' = 32 -> RN16(1 / sqrt 32)
        self.assertEqual(tab[65536 + 0x4000], 0.5)                             # 1 / 2


if __name__ == "__main__":
    unittest.main()
