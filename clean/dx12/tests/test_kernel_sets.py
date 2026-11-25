# SPDX-License-Identifier: MIT
"""The D3D12 kernel sets, from the sources' declarations (clean/dx12/build_dx12.py kernel_set; nothing is compiled):
the cheap check before a build, in seconds. kernel_sets.json lists them: "model_m" = what a build without --model-l
compiles (Model M's kernel set; Model L's and Model K's kernels never enter it), "model_l" = what --model-l adds
(#89), "model_k" = what --model-k adds (clean/k/dx12/kernels, #15).

A kernel added, removed or moved between the sets changes kernel_sets.json in the same change; regenerate it with
    python clean/dx12/tests/test_kernel_sets.py --write
and review the difference."""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DX12 = os.path.dirname(HERE)
sys.path.insert(0, DX12)
import build_dx12 as B  # noqa: E402

SETS = os.path.join(DX12, "kernel_sets.json")


def current():
    m = B.kernel_set(False)
    full = B.kernel_set(True)
    k = B.kernel_set(False, True)
    return {"model_m": [list(x) for x in m], "model_l": [list(x) for x in full if x not in set(m)],
            "model_k": [list(x) for x in k if x not in set(m)]}


class TestKernelSets(unittest.TestCase):
    def test_sets_match_the_list(self):
        with open(SETS) as f:
            want = json.load(f)
        got = current()
        for name in ("model_m", "model_l", "model_k"):
            w, g = set(map(tuple, want[name])), set(map(tuple, got[name]))
            self.assertEqual(g, w, "%s: added %s, removed %s (update kernel_sets.json: python "
                                   "clean/dx12/tests/test_kernel_sets.py --write)" % (name, sorted(g - w), sorted(w - g)))

    def test_model_m_has_no_model_l_kernel(self):
        m = B.kernel_set(False)
        self.assertFalse([k for k in m if k[1].startswith("l_") or k[0].startswith("l_") or k[0] in B.MODEL_L_ONLY])
        self.assertTrue(m)

    def test_model_l_adds_only_model_l_kernels(self):
        m, full = set(B.kernel_set(False)), set(B.kernel_set(True))
        self.assertTrue(m <= full)
        extra = full - m
        self.assertTrue(extra)
        # Model L's kernels: its code objects, the l_* direct-route kernels, MODEL_L_ONLY, the hybrid twins (*_hyb) and the
        # fast-mode twins (*_fast): Model L's route only; build_dx12.py is_model_l)
        self.assertFalse([k for k in extra if not (k[1].startswith("l_") or k[0].startswith("l_") or k[0] in B.MODEL_L_ONLY
                                                   or k[0].endswith("_hyb") or k[0].endswith("_fast"))])


    def test_model_k_adds_only_model_k_kernels(self):
        """--model-k adds Model K's kernels to Model M's set and nothing else; with --model-l it is all three sets."""
        m, k = set(B.kernel_set(False)), set(B.kernel_set(False, True))
        self.assertTrue(m <= k)
        extra = k - m
        self.assertTrue(extra)
        self.assertFalse([x for x in extra if not (x[1].startswith("k_") or x[0].startswith("k_") or x[0].startswith("kx/"))])
        self.assertEqual(set(B.kernel_set(True, True)), set(B.kernel_set(True)) | extra)

    def test_no_k_kernel_in_the_other_sets(self):
        """Neither Model M's nor Model L's set holds a Model K kernel (a K port is declared in clean/k/dx12/kernels)."""
        for full in (B.kernel_set(False), B.kernel_set(True)):
            self.assertFalse([x for x in full if x[1].startswith("k_") or x[0].startswith("k_") or x[0].startswith("kx/")])

    def test_struct_argument_tokens(self):
        """s<N>:<fields>: pointers are one root constant each, copied bytes one per four, skipped bytes none."""
        self.assertEqual(B.struct_dwords("s184:p19.b32", "t"), 27)
        self.assertEqual(B.struct_dwords("s160:p13.b56", "t"), 27)
        self.assertEqual(B.struct_dwords("s32:p1.x8.b16", "t"), 5)
        self.assertEqual(B.arg_dwords("p,u,s184:p19.b32,b36", "t"), 1 + 1 + 27 + 9)
        for bad in ("s16:p1", "s16:p1.b4", "s12:p1.b3", "s8:q1", "s0:p0", "s8", "s8:p1.x4", "s12:p1.b4.x"):
            with self.assertRaises(SystemExit, msg=bad):
                B.struct_dwords(bad, "t")

    def test_frame_gen_kernels_are_in_no_set(self):
        """Frame generation's ports (#96, co=fg_*.co or named fg_*) are declared but enter neither set: only --only
        builds them, so the Model M and Model L builds stay as they are."""
        decls = []
        for fn in sorted(os.listdir(B.KERNEL_DIR)):
            if fn.endswith(".hlsl"):
                decls += B.kernel_decls(os.path.join(B.KERNEL_DIR, fn))
        fg = [(d["name"], d["co"]) for d in decls if B.is_frame_gen(d)]
        self.assertTrue(fg, "no frame-generation kernel declared")
        full = set(B.kernel_set(True))
        self.assertFalse([k for k in fg if k in full])


    def test_fast_twins_take_their_base_kernels_arguments(self):
        """A "<kernel>_fast" twin (Model L's l_mode=fast) is launched by the runtime in place of its base kernel with the same
        grid, block and arguments: the declarations agree in block, wave, args and (for a twin of a kernel with a launch
        shape of its own) group; the twin has no HIP code object, and is in Model L's set only."""
        decls = {}
        for fn in sorted(os.listdir(B.KERNEL_DIR)):
            if fn.endswith(".hlsl"):
                for d in B.kernel_decls(os.path.join(B.KERNEL_DIR, fn)):
                    decls.setdefault(d["name"], []).append(d)
        twins = [n for n in decls if n.endswith("_fast")]
        self.assertTrue(twins)
        full, m = set(B.kernel_set(True)), set(B.kernel_set(False))
        for n in twins:
            base = n[:-5]
            self.assertIn(base, decls, n)
            (t,) = decls[n]
            self.assertEqual(t["co"], "-", n)
            self.assertTrue(B.is_model_l(t), n)
            self.assertIn((n, "-"), full - m, n)
            for b in decls[base]:
                for key in ("block", "wave", "args", "group", "gscale"):
                    self.assertEqual(t.get(key), b.get(key), (n, key))


if __name__ == "__main__":
    if "--write" in sys.argv:
        with open(SETS, "w", newline="\n") as f:
            json.dump(current(), f, indent=1)
            f.write("\n")
        print("wrote", SETS)
    else:
        unittest.main()
