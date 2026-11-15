# SPDX-License-Identifier: MIT
"""The resource gates of the gfx11 Swin block kernels (runner/gfx11_gates.py, gfx11_gates.json): no spills, no more registers
and no fewer waves per SIMD than recorded, for every gated kernel in the default gfx11 build and in each gfx11 variant.

The first group runs everywhere (the check on made-up numbers). The second compiles clean/kernels/swin.hip (swin_v2.hip for the
persistent-kernel build) for gfx1100 once per build and needs the compiler: set WG_ROCM to a ROCm folder (no GPU is used).

    WG_ROCM=<ROCm> TMPDIR=<scratch> python -m unittest clean.tests.test_gfx11_gates
"""
import copy
import os
import tempfile
import unittest

from clean.runner import gfx11_gates as G
from clean.runner import variants as V


def table_measurement(table):
    """A measurement that sits exactly on the gates of `table`."""
    return {name: {k: {"vgprs": g["vgprs"], "spills": 0, "waves": g["waves"], "lds": g["lds"]} for k, g in kernels.items()}
            for name, kernels in table["gates"].items()}


class TestGateTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = G.load()

    def test_table_has_every_build_and_kernel(self):
        self.assertEqual(sorted(self.table["gates"]), sorted(G.builds()))
        for name, kernels in self.table["gates"].items():
            self.assertEqual(sorted(kernels), sorted(G.kernels_of(name)), name)

    def test_every_kernel_has_a_position_in_the_table_of_its_shape(self):
        """The shapes of E1: c96, c128, the `_add` form and the resident kernels."""
        names = set(self.table["gates"]["base"])
        for want in ("swin_block_c96", "swin_block_c128", "swin_block_c64_add", "swin_block_c96_add", "swin_block_c64_r",
                     "swin_block_c64_add_r"):
            self.assertIn(want, names)

    def test_the_persistent_build_gates_the_bottleneck_megakernel(self):
        """swin_bneck6 of the persistent-kernel build (swin_v2.hip): no spill, 6 waves per SIMD. swin_mid is not gated yet."""
        self.assertEqual(sorted(self.table["gates"]["bmegahg"]), ["swin_bneck6", "swin_bneck6_nc"])
        self.assertGreaterEqual(self.table["gates"]["bmegahg"]["swin_bneck6"]["waves"], 6)

    def test_the_encoder_is_gated(self):
        """E4: input_encoder (encoder.hip): its own build in the table, no spill by construction (a spilling kernel is not
        recorded)."""
        gates = self.table["gates"]
        self.assertEqual(sorted(gates["enc0"]), ["input_encoder"])
        self.assertEqual(G.SOURCE_BUILDS["enc0"][0], "encoder.hip")

    def test_recorded_numbers_are_plausible(self):
        """Registers fit the 256 of a wave and the waves are the 1 to 16 a SIMD holds (a sanity check on the file)."""
        for name, kernels in self.table["gates"].items():
            for k, g in kernels.items():
                self.assertTrue(0 < g["vgprs"] <= 256, (name, k))
                self.assertTrue(1 <= g["waves"] <= 16, (name, k))


class TestCheck(unittest.TestCase):
    """check(): what breaks a gate, on made-up numbers (the recorded table with one number moved)."""

    @classmethod
    def setUpClass(cls):
        cls.table = G.load()

    def measured(self):
        return table_measurement(self.table)

    def test_a_measurement_on_the_gates_passes(self):
        self.assertEqual(G.check(self.measured(), self.table), [])

    def test_a_better_measurement_passes(self):
        m = self.measured()
        m["base"]["swin_block_c128"]["vgprs"] -= 8
        m["base"]["swin_block_c128"]["waves"] += 1
        self.assertEqual(G.check(m, self.table), [])

    def test_a_spill_fails(self):
        m = self.measured()
        m["f16wq"]["swin_block_c96"]["spills"] = 30
        bad = G.check(m, self.table)
        self.assertEqual(len(bad), 1)
        self.assertIn("f16wq swin_block_c96", bad[0])
        self.assertIn("spills 30", bad[0])

    def test_fewer_waves_per_simd_fails(self):
        m = self.measured()
        m["base"]["swin_block_c128"]["waves"] -= 1
        bad = G.check(m, self.table)
        self.assertEqual(len(bad), 1)
        self.assertIn("waves per SIMD", bad[0])

    def test_more_registers_fails(self):
        m = self.measured()
        m["f16w"]["swin_block_c64_r"]["vgprs"] += 1
        bad = G.check(m, self.table)
        self.assertEqual(len(bad), 1)
        self.assertIn("registers (gate: at most", bad[0])

    def test_a_missing_kernel_or_build_fails(self):
        m = self.measured()
        del m["base"]["swin_block_c64_add"]
        del m["f16wa"]
        bad = G.check(m, self.table)
        self.assertTrue(any("base swin_block_c64_add: not measured" in b for b in bad))
        self.assertTrue(any("f16wa: not measured" in b for b in bad))

    def test_record_refuses_a_spilling_kernel(self):
        m = self.measured()
        m["base"]["swin_block_c128"]["spills"] = 94
        with self.assertRaises(SystemExit):
            G.record(m, "x")

    def test_record_then_check_round_trips(self):
        m = self.measured()
        rec = G.record(m, "x")
        self.assertEqual(rec["gates"], copy.deepcopy(self.table["gates"]))
        self.assertEqual(G.check(m, rec), [])


class TestCompiler(unittest.TestCase):
    """The recorded gates against the compiler's report for the sources of this tree."""

    @classmethod
    def setUpClass(cls):
        rocm = os.environ.get("WG_ROCM")
        if not rocm:
            raise unittest.SkipTest("set WG_ROCM to a ROCm folder to compile swin.hip for gfx1100")
        cls.table = G.load()
        with tempfile.TemporaryDirectory(prefix="wg_gates_") as scratch:
            cls.measured, cls.version = G.measure(rocm, scratch)

    def test_every_gate_holds(self):
        print(f"gfx11 gates: clang {self.version} (table recorded with {self.table['clang']}), "
              f"{sum(len(k) for k in self.measured.values())} kernel builds")
        self.assertEqual(G.check(self.measured, self.table), [])

    def test_the_variants_are_the_family_variants(self):
        self.assertEqual(G.builds(), ["base", *V.FAMILY_VARIANTS["gfx11"], *V.FAMILY_MEGA["gfx11"], *G.SOURCE_BUILDS])


if __name__ == "__main__":
    unittest.main()
