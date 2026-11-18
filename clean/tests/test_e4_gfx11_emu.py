# SPDX-License-Identifier: MIT
"""The encoder (kernels/encoder.hip), the decoder (kernels/decoder.hip) and the fused dec1 + dec0 (kernels/dec1_dec0.hip) with
the gfx11 (RDNA 3) matrix wrappers of kernels/wg_platform.h, in the host lane emulator's gfx11 mode (-DWG_EMU_GFX11) with the
gfx11 accumulator row table: the same cases and the same limits as test_encoder_emu, test_decoder_emu and test_dec1_dec0_emu.
The emulator's matrix instruction sums exactly, so the stricter limits of those tests hold here (the card: test_gpu_e4).
"""
import unittest

import numpy as np

from clean.tests import test_dec1_dec0_emu as TD1
from clean.tests import test_decoder_emu as TD
from clean.tests import test_encoder_emu as TE
from clean.tests.test_kernels_emu_gfx11 import Gfx11Case, build_dir


class TestEncoderEmuGfx11(Gfx11Case, TE.TestEncoderEmu):
    def test_the_row_table_matters(self):
        """Sanity of the comparison: weights packed for the gfx12 accumulator rows give another result."""
        rng = np.random.default_rng(21)
        x = TE.E.random_inputs(rng, rw=40, rh=24)
        p = TE.E.random_params(rng)
        a = self.runner.run("enc0", self.runner.pack_weights("enc0", p), x, None)
        wrong = self.Runner(build_dir(), "", backend="emu", arch="gfx11", d_row="gfx12")
        b = wrong.run("enc0", wrong.pack_weights("enc0", p), x, None)
        self.assertGreater(float(np.mean(a["features"] != b["features"])), 0.2)


class TestDecoderEmuGfx11(Gfx11Case, TD.TestDecoderEmu):
    def test_the_row_table_matters(self):
        rng = np.random.default_rng(31)
        x = TD.Dc.random_inputs(rng)
        p = TD.Dc.random_params(rng)
        a = self.run_dec0(x, p)
        wrong = self.Runner(build_dir(), "", backend="emu", arch="gfx11", d_row="gfx12")
        b = self.run_dec0(x, p, runner=wrong)
        self.assertGreater(float(np.mean(a["A"] != b["A"])), 0.2)


class TestDec1Dec0EmuGfx11(Gfx11Case, TD1.TestDec1Dec0Emu):
    pass


if __name__ == "__main__":
    unittest.main()
