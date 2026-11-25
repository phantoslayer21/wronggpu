# SPDX-License-Identifier: MIT
"""Developer builds refuse a post_tf launch whose A8 or history pointer is not 8-byte aligned.

The D3D12 post_tf ports (clean/dx12/kernels/post.hlsl ld64_a8 / ld128_a8) clear the low 3 bits of every A8 and history
address, which is the requested address only when both pointers are 8-byte aligned (the runtime's own allocations are
256-byte aligned; an interior pointer such as A + 4 would be read 4 bytes early, silently). In builds with
WG_DEV_DIAGNOSTICS runtime.cpp checks both before every post_tf launch and fails naming the pointer. No GPU:
- clean/runtime/tests/test_post_tf_align.cpp, built with the host C++ compiler: each pointer at base + 1, 2, 4 or 6 is
  refused naming it; null, 8- and 256-byte aligned pointers pass. Negative controls: the check removed from
  post_tf_align.h (red), and a pointer left out of its list (red).
- source pins: one check before each post_tf launch (the ordinary and the motion-at-output form), with the pointers the
  launch passes (A, hist[cur]), ending in fail(); the header and both checks inside WG_DEV_DIAGNOSTICS only, so a release
  build has none of it.

Run: python -m unittest clean.tests.test_post_tf_align
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RT = os.path.join(ROOT, "clean", "runtime", "runtime.cpp")
HDR = os.path.join(ROOT, "clean", "runtime", "post_tf_align.h")
SRC = os.path.join(ROOT, "clean", "runtime", "tests", "test_post_tf_align.cpp")
DEV = "#ifdef WG_DEV_DIAGNOSTICS"


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


class TestPostTfAlign(unittest.TestCase):
    def _run(self, header_text=None):
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no host C++ compiler (CXX, clang++ or g++)")
        with tempfile.TemporaryDirectory(dir=os.environ.get("WG_TEST_TMP")) as d:
            inc = ROOT
            if header_text is not None:   # a mutated post_tf_align.h in a copy of the tree layout the source includes
                inc = d
                os.makedirs(os.path.join(d, "clean", "runtime"))
                with open(os.path.join(d, "clean", "runtime", "post_tf_align.h"), "w") as f:
                    f.write(header_text)
            exe = os.path.join(d, "t.exe")
            r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", inc, SRC, "-o", exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return subprocess.run([exe], capture_output=True, text=True)

    def test_misaligned_pointers_are_refused(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(r.stdout.count("refused naming it"), 2 * 4)

    def test_negative_control_check_removed(self):
        h = read(HDR)
        self.assertIn("if (e.v & 7u) return e.name;", h)
        r = self._run(h.replace("if (e.v & 7u) return e.name;", "(void)e;"))
        self.assertNotEqual(r.returncode, 0, "the test passed with the check removed")

    def test_negative_control_pointer_dropped(self):
        h = read(HDR)
        self.assertIn('{a8, "A8"}, ', h)
        r = self._run(h.replace('{a8, "A8"}, ', ""))
        self.assertNotEqual(r.returncode, 0, "the test passed with a pointer left out of the check")
        self.assertIn("FAIL A8 + 4", r.stdout)

    def test_runtime_wiring(self):
        src = read(RT)
        blocks = [m for m in re.finditer(re.escape(DEV) + r"\n(.*?)\n#endif", src, re.S)]
        calls = [m.start() for m in re.finditer(r"post_tf_misaligned\(", src)]
        self.assertEqual(len(calls), 2, "one check before each post_tf launch (ordinary, motion at output resolution)")
        for c in calls:
            self.assertTrue(any(b.start(1) <= c < b.end(1) for b in blocks), "a check outside WG_DEV_DIAGNOSTICS")
        self.assertTrue(any('#include "post_tf_align.h"' in b.group(1) for b in blocks), "the header outside the macro")
        self.assertNotIn('#include "post_tf_align.h"', re.sub(re.escape(DEV) + r"\n.*?\n#endif", "", src, flags=re.S))
        for launch in (r'launch\(\("pmo/" \+ post_last\)', r"launch\(\(post_pre \+ post_last\)\.c_str\(\), cdiv\(ow, 40\)"):
            m = re.search(launch, src)
            self.assertIsNotNone(m, launch)
            before = src[max(0, m.start() - 600):m.start()]
            self.assertIn("post_tf_misaligned(A, hist[cur])", before, "no check right before " + launch)
            self.assertRegex(before, r"return fail\(.*\n?.*is not 8-byte aligned")
            call = src[m.start():m.start() + 300]
            self.assertIn("&cnet, &A, &post_b, &C, &hist[cur]", call, "the launch no longer passes A and hist[cur] as checked")


if __name__ == "__main__":
    unittest.main()
