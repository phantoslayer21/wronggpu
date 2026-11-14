# SPDX-License-Identifier: MIT
"""The D3D12 kernel sets, from the sources' declarations (clean/dx12/build_dx12.py kernel_set; nothing is compiled):
the cheap check before a build, in seconds. kernel_sets.json lists them: "model_m" = what a build without --model-l
compiles (Model M's kernel set; Model L's kernels never enter it), "model_l" = what --model-l adds (#89).

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
    return {"model_m": [list(k) for k in m], "model_l": [list(k) for k in full if k not in set(m)]}


class TestKernelSets(unittest.TestCase):
    def test_sets_match_the_list(self):
        with open(SETS) as f:
            want = json.load(f)
        got = current()
        for name in ("model_m", "model_l"):
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
        self.assertFalse([k for k in extra if not (k[1].startswith("l_") or k[0].startswith("l_") or k[0] in B.MODEL_L_ONLY)])


if __name__ == "__main__":
    if "--write" in sys.argv:
        with open(SETS, "w", newline="\n") as f:
            json.dump(current(), f, indent=1)
            f.write("\n")
        print("wrote", SETS)
    else:
        unittest.main()
