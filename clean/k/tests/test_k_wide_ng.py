# SPDX-License-Identifier: MIT
"""Model K: the NG = 4 forms of the wide stages e3 and d3 (k_stagew4_e3, k_stagew4_d3) cannot be reached (host checks).

The runtime launches a wide stage as k_stagew<NG>_<name> with NG from wide_ng (clean/k/runtime/runtime_k.cpp), in one
place for the frame and the softmax self-check (stage_kernel_base, which network() and smx_check / smx_case_name call)
and one for the list of kernels a context can launch (d3d12_kernels()). For the FP32
(exact) network, the only one the D3D12 devices run, wide_ng gives e3 and d3 NG = 2 whatever the window count, the
render size, the output size or any option; the FP16 networks (network=fused16 / fused8) take other kernels and
are refused on the native device. So the NG = 4 forms of e3 and d3 are not part of any context's kernel set and are not
declared in the D3D12 build. This test pins that:
- the source of wide_ng: e3 and d3 return 2 before any window-count rule, and no option or size reaches them;
- the only places that build a k_stagew name use wide_ng's result;
- the declared D3D12 kernels (clean/dx12/kernel_sets.json) hold k_stagew2_e3 and k_stagew2_d3 and no NG = 4 form of e3 or
  d3, and the Python mirror of the rule (clean/k/runner/k_launch.py wide_ng) agrees for every window count.
If a later change makes the NG = 4 form of e3 or d3 reachable, this test fails until the form is ported and checked.

Run: python -m unittest clean.k.tests.test_k_wide_ng
"""
import json
import os
import re
import unittest

from clean.k.runner import k_launch as KL

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RT = os.path.join(ROOT, "clean", "k", "runtime", "runtime_k.cpp")


def source():
    with open(RT, encoding="utf-8") as f:
        return f.read()


def wide_ng_body(src):
    m = re.search(r"int wide_ng\(const std::string& s, bool f16, int nwin\) \{\n(.*?)\n\}\n", src, re.S)
    assert m, "wide_ng not found in runtime_k.cpp"
    return m.group(1)


class TestWideNg(unittest.TestCase):
    def test_wide_ng_source(self):
        body = wide_ng_body(source()).splitlines()
        body = [ln.strip() for ln in body]
        # the FP16 networks first: e4, m, d4 only (e3 and d3 take k_stageh_*, no wide kernel)
        self.assertEqual(body[0], 'if (f16) return s == "e4" || s == "m" || s == "d4" ? 4 : 0;')
        # FP32: only e4 and d4 look at the window count; e3 and d3 are 2 without one
        self.assertIn('if (s == "e3" || s == "d3") return 2;', body)
        self.assertEqual([ln for ln in body if "nwin" in ln and "int wide_ng" not in ln],
                         ['if (s == "e4" || s == "d4") return nwin <= 24 ? 4 : 2;'])
        self.assertEqual(body[-1], "return 0;")
        self.assertEqual(len(body), 5)

    def test_wide_names_come_from_wide_ng(self):
        src = source()
        # the two sites that build a wide kernel name use wide_ng's result only: stage_kernel_base (the frame and the
        # softmax self-check) and the kernel list
        sites = re.findall(r'"k_stagew" \+ std::to_string\((\*?\w+)\)', src)
        self.assertEqual(sorted(sites), ["*ng", "ng"])
        self.assertEqual(re.findall(r"\*ng = wide_ng\(([^;]*)\);", src), ["st.name, f16, nwin"])
        self.assertEqual(re.findall(r"const int ng = wide_ng\(([^;]*)\);", src), ["n, false, k_window_count(lw, lh, o, o)"])
        # network() names its stage through stage_kernel_base with the frame's window count; the self-check with FP32
        self.assertIn("stage_kernel_base(st, f16_, p.y != 0, p.nwx * p.nwy, &ng, &w2)", src)
        self.assertEqual(len(re.findall(r"stage_kernel_base\(st, false, true,", src)), 2)
        self.assertNotIn("k_stagew4_e3", src)
        self.assertNotIn("k_stagew4_d3", src)

    def test_python_mirror(self):
        for stage in ("e3", "d3"):
            self.assertEqual(KL.WIDE_DEFAULT[stage], 2)
            for nwin in range(0, 2000):
                self.assertEqual(KL.wide_ng(stage, KL.WIDE_DEFAULT[stage], nwin), 2, (stage, nwin))
        self.assertNotIn("e3", KL.WIDE4_SMALL)
        self.assertNotIn("d3", KL.WIDE4_SMALL)

    def test_declared_kernel_set(self):
        with open(os.path.join(ROOT, "clean", "dx12", "kernel_sets.json"), encoding="utf-8") as f:
            names = {n for n, _ in json.load(f)["model_k"]}
        self.assertTrue({"k_stagew2_e3", "k_stagew2_d3"} <= names)
        self.assertFalse({"k_stagew4_e3", "k_stagew4_d3"} & names)
        # the NG = 4 forms the runtime does reach are declared: e4 and d4 at up to 24 windows, and m
        self.assertTrue({"k_stagew4_e4", "k_stagew4_d4", "k_stagew4_m"} <= names)


if __name__ == "__main__":
    unittest.main()
