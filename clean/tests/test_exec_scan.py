# SPDX-License-Identifier: MIT
"""clean/runner/exec_scan.py: v_wmma and v_permlanex16 under a lane-divergent exec mask.

Part 1 runs the pass over small hand-written functions (every instruction 4 bytes, branches resolved from labels), so each
rule of the pass is tested on a case that has to be flagged and a look-alike that must not be.

Part 2 scans the gfx11 code objects of a build (WG_BUILD, objdump from WG_ROCM): the kernels the runner launches on the card
must have no matrix or exchange instruction under a lane mask. Skipped without a gfx11 build:

    python clean/runner/build.py --arch gfx1100 --rocm <ROCm> --out <build> --no-runtime
    WG_BUILD=<build> WG_ROCM=<ROCm> python -m unittest clean.tests.test_exec_scan
"""
import os
import unittest

from clean.runner import exec_scan as E
from clean.runner import family


def asm(*items, name="k"):
    """Disassembly text of one function. Items: "label:" lines, or "mnemonic operands" (a branch names its label last)."""
    pcs, labels, pc = [], {}, 0
    for it in items:
        if it.endswith(":"):
            labels[it[:-1]] = pc
        else:
            pcs.append((pc, it))
            pc += 4
    lines = [f"{0:016x} <{name}>:"]
    for pc, it in pcs:
        mn, _, ops = it.partition(" ")
        dword = 0xBF800000
        if mn.startswith(("s_branch", "s_cbranch")):
            target = labels[ops.strip()]
            ops = "0"
            dword = 0xBFA00000 | (((target - pc - 4) // 4) & 0xFFFF)
        lines.append(f"\t{mn} {ops}".rstrip() + f"   // {pc:012X}: {dword:08X}")
    return "\n".join(lines) + "\n"


def flagged(*items):
    found, _ = E.scan_text(asm(*items))
    return [(mn, origin) for _addr, mn, origin in found.get("k", [])]


WMMA = "v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]"
PERM = "v_permlanex16_b32 v1, v0, s2, s3 op_sel:[1,0]"


class TestExecScanRules(unittest.TestCase):
    def test_plain_code_is_clean(self):
        self.assertEqual(flagged(WMMA, PERM, "s_endpgm"), [])

    def test_matrix_in_a_branch_under_saveexec_is_flagged(self):
        self.assertEqual(flagged("v_cmp_gt_i32 vcc_lo, s0, v1", "s_and_saveexec_b32 s1, vcc_lo", "s_cbranch_execz out",
                                 WMMA, "out:", "s_or_b32 exec_lo, exec_lo, s1", "s_endpgm"), [(WMMA.split()[0], 4)])

    def test_exchange_in_a_branch_under_saveexec_is_flagged(self):
        self.assertEqual([m for m, _ in flagged("s_and_saveexec_b32 s1, vcc_lo", PERM, "s_or_b32 exec_lo, exec_lo, s1",
                                                 "s_endpgm")], ["v_permlanex16_b32"])

    def test_matrix_after_the_mask_is_restored_is_clean(self):
        self.assertEqual(flagged("s_and_saveexec_b32 s1, vcc_lo", "s_cbranch_execz out", "v_add_f32 v0, v0, v1", "out:",
                                 "s_or_b32 exec_lo, exec_lo, s1", WMMA, "s_endpgm"), [])

    def test_restore_with_mov_from_the_saved_register(self):
        self.assertEqual(flagged("s_mov_b32 s5, exec_lo", "s_and_b32 exec_lo, exec_lo, vcc_lo", "v_add_f32 v0, v0, v1",
                                 "s_mov_b32 exec_lo, s5", WMMA, "s_endpgm"), [])
        self.assertEqual(len(flagged("s_mov_b32 s5, exec_lo", "s_and_b32 exec_lo, exec_lo, vcc_lo", WMMA,
                                     "s_mov_b32 exec_lo, s5", "s_endpgm")), 1)

    def test_else_branch_of_if_else_is_flagged_and_the_join_is_clean(self):
        items = ["s_and_saveexec_b32 s1, vcc_lo", "s_xor_b32 s1, exec_lo, s1", "s_cbranch_execz else", "v_add_f32 v0, v0, v1",
                 "else:", "s_and_not1_saveexec_b32 s1, s1", "s_cbranch_execz join", WMMA, "join:",
                 "s_or_b32 exec_lo, exec_lo, s1", PERM, "s_endpgm"]
        self.assertEqual([m for m, _ in flagged(*items)], [WMMA.split()[0]])

    def test_uniform_branches_do_not_change_the_mask(self):
        self.assertEqual(flagged("s_cmp_lt_i32 s0, s1", "s_cbranch_scc1 out", WMMA, "out:", PERM, "s_endpgm"), [])
        self.assertEqual(flagged("s_cbranch_vccz out", WMMA, "out:", "s_endpgm"), [])

    def test_matrix_inside_a_divergent_loop_is_flagged(self):
        self.assertEqual(len(flagged("s_mov_b32 s1, 0", "top:", "v_cmp_lt_u32 s0, v8, v9", "s_or_b32 s1, s0, s1", WMMA,
                                     "s_and_not1_b32 exec_lo, exec_lo, s1", "s_cbranch_execnz top",
                                     "s_or_b32 exec_lo, exec_lo, s1", "s_endpgm")), 1)

    def test_matrix_after_a_divergent_loop_is_clean(self):
        self.assertEqual(flagged("s_mov_b32 s1, 0", "top:", "v_cmp_lt_u32 s0, v8, v9", "s_or_b32 s1, s0, s1",
                                 "v_add_f32 v0, v0, v1", "s_and_not1_b32 exec_lo, exec_lo, s1", "s_cbranch_execnz top",
                                 "s_or_b32 exec_lo, exec_lo, s1", WMMA, PERM, "s_endpgm"), [])

    def test_loop_entered_under_a_mask_stays_partial_after_it(self):
        self.assertEqual(len(flagged("s_and_saveexec_b32 s9, vcc_lo", "s_mov_b32 s1, 0", "top:", "s_or_b32 s1, s0, s1",
                                     "s_and_not1_b32 exec_lo, exec_lo, s1", "s_cbranch_execnz top",
                                     "s_or_b32 exec_lo, exec_lo, s1", WMMA, "s_endpgm")), 1)

    def test_v_cmpx_makes_the_mask_partial(self):
        self.assertEqual(len(flagged("v_cmpx_gt_i32 exec_lo, 0, v1", WMMA, "s_endpgm")), 1)

    def test_mask_set_back_to_all_ones(self):
        self.assertEqual(flagged("s_and_b32 exec_lo, exec_lo, vcc_lo", "s_mov_b32 exec_lo, -1", WMMA, "s_endpgm"), [])

    def test_a_partial_path_joining_a_full_one_is_partial(self):
        self.assertEqual(len(flagged("s_cbranch_scc1 skip", "s_and_saveexec_b32 s1, vcc_lo", "skip:", WMMA, "s_endpgm")), 1)

    def test_a_full_path_joining_a_partial_one_is_partial(self):
        self.assertEqual(len(flagged("s_and_saveexec_b32 s1, vcc_lo", "s_cbranch_scc1 skip", "s_or_b32 exec_lo, exec_lo, s1",
                                     "skip:", WMMA, "s_endpgm")), 1)

    def test_if_else_through_v_cmpx_and_the_else_saveexec_joins_back_to_the_mask_before_the_if(self):
        """The shape the compiler gives `if (c) {...} else {...}` over a per-lane test: the join restores the mask the if was
        entered with, not the one the else side saved."""
        items = ["s_mov_b32 s1, exec_lo", "v_cmpx_ne_u32 exec_lo, 0, v20", "s_xor_b32 s1, exec_lo, s1", "s_cbranch_execz else",
                 "v_add_f32 v0, v0, v1", "else:", "s_and_not1_saveexec_b32 s0, s1", "s_cbranch_execz join",
                 "v_add_f32 v0, v0, v2", "join:", "s_or_b32 exec_lo, exec_lo, s0"]
        self.assertEqual(flagged(*items, WMMA, PERM, "s_endpgm"), [])
        self.assertEqual(len(flagged(*items[:6], "s_and_not1_saveexec_b32 s0, s1", WMMA, "s_endpgm")), 1)

    def test_the_origin_is_the_instruction_that_narrowed_the_mask(self):
        found, _ = E.scan_text(asm("v_add_f32 v0, v0, v1", "s_and_saveexec_b32 s1, vcc_lo", WMMA, "s_endpgm"))
        self.assertEqual([(a, o) for a, _m, o in found["k"]], [(8, 4)])


# the kernels the runner launches on the card (stage_runner.py BLOCK_KERNELS and the others), by code object
RUN_ON_THE_CARD = {
    "swin.co": ("swin_block_c64", "swin_block_c64_nc", "swin_block_c64_add", "swin_block_c64_add_nc", "swin_block_c64_r",
                "swin_block_c64_r_nc", "swin_block_c64_add_r", "swin_block_c64_add_r_nc", "swin_block_c96", "swin_block_c96_nc",
                "swin_block_c96_add", "swin_block_c96_add_nc", "swin_block_c128", "swin_block_c128_nc", "patch_merge_c64",
                "patch_merge_c96", "expand_c96", "expand_c128"),
    "swin__bmegahg.co": ("patch_merge_c64", "patch_merge_c96", "expand_c96", "expand_c128", "swin_block_c64",
                         "swin_block_c64_nc", "swin_block_c64_add", "swin_block_c64_add_nc", "swin_block_c96",
                         "swin_block_c96_nc", "swin_block_c96_add", "swin_block_c96_add_nc", "swin_block_c128",
                         "swin_block_c128_nc", "swin_bneck6", "swin_bneck6_nc"),
    "encoder.co": ("input_encoder",),
    "decoder.co": ("coef_decoder",),
}


class TestExecScanBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = os.environ.get("WG_BUILD")
        if not cls.build or family.build_family(cls.build) != "gfx11":
            raise unittest.SkipTest("no gfx11 build in WG_BUILD")
        rocm = os.environ.get("WG_ROCM", "")
        cls.objdump = os.path.join(rocm, "llvm", "bin", "llvm-objdump")
        if not os.path.exists(cls.objdump):
            cls.objdump = None  # the pass's default: ROCM_PATH, else /usr/lib64/rocm
        if not os.path.exists(cls.objdump or os.path.join(os.environ.get("ROCM_PATH", "/usr/lib64/rocm"), "llvm", "bin",
                                                          "llvm-objdump")):
            raise unittest.SkipTest("no llvm-objdump")

    def test_no_matrix_or_exchange_instruction_under_a_lane_mask_in_the_kernels_the_card_runs(self):
        for co, kernels in RUN_ON_THE_CARD.items():
            path = os.path.join(self.build, co)
            if not os.path.exists(path):
                continue
            found, counts = E.scan_object(path, self.objdump)
            for k in kernels:
                with self.subTest(code_object=co, kernel=k):
                    self.assertGreater(counts.get(k, 0), 0, "the kernel has no matrix instruction: is the name right?")
                    self.assertEqual([(hex(a), m, hex(o)) for a, m, o in found.get(k, [])][:3], [])

    def test_the_scan_sees_the_deliberately_divergent_probe(self):
        path = os.path.join(self.build, "probe.co")
        if not os.path.exists(path):
            self.skipTest("no probe.co")
        found, _ = E.scan_object(path, self.objdump)
        self.assertIn("probe_xchg16_div", found)


if __name__ == "__main__":
    unittest.main()
