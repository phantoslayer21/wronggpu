# SPDX-License-Identifier: MIT
"""Developer builds refuse a Model K stage parameter block whose pointers are not 16-byte aligned (K-ENC review fix 1).

The native RDNA 3 kernels' KN_A16 form clears the low 4 bits of every 16-byte arena address (k_block_g11.hlsli), which is
the requested address only when every KBlkParams pointer is 16-byte aligned. In builds with WG_DEV_GFX11 or
WG_DEV_DIAGNOSTICS (the builds that can launch those kernels) runtime_k.cpp checks every stage's parameters at create and
every k_stage* launch, and fails naming the pointer. No GPU:
- clean/runtime/tests/test_k_blk_align.cpp, built with the host C++ compiler against KBlkParams as runtime_k.cpp declares
  it: every pointer field with a base + 1, 4, 8 or 12 is refused naming that field; null, 16- and 256-byte aligned
  pointers pass. Negative controls: the check removed from k_blk_align.h (red), and a field dropped from its list (red).
- source pins: the create check and the launch check exist, call k_blk_misaligned and fail(); they are outside the
  softmax self-check regions (whose text the self-check digest covers) and inside the developer macros only, so a release
  build has none of it.

Run: python -m unittest clean.tests.test_k_blk_align
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RT = os.path.join(ROOT, "clean", "k", "runtime", "runtime_k.cpp")
HDR = os.path.join(ROOT, "clean", "k", "runtime", "k_blk_align.h")
SRC = os.path.join(ROOT, "clean", "runtime", "tests", "test_k_blk_align.cpp")
DEV = "#if defined(WG_DEV_GFX11) || defined(WG_DEV_DIAGNOSTICS)"


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def kblk_struct(src):
    m = re.search(r"struct KBlkParams \{\n(.*?)\n\};", src, re.S)
    assert m, "KBlkParams not found in runtime_k.cpp"
    ptrs = re.search(r"DevPtr ([^;]+);", m.group(1)).group(1)
    names = [n.strip() for n in ptrs.split(",")]
    fields = ", ".join('{"%s", offsetof(KBlkParams, %s)}' % (n, n) for n in names)
    return m.group(0) + "\n#include <cstddef>\n#define KBLK_FIELDS " + fields + "\n", names


class TestKBlkAlign(unittest.TestCase):
    def _run(self, header_text=None):
        cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
        if not cxx:
            self.skipTest("no host C++ compiler (CXX, clang++ or g++)")
        with tempfile.TemporaryDirectory(dir=os.environ.get("WG_TEST_TMP")) as d:
            struct, _ = kblk_struct(read(RT))
            with open(os.path.join(d, "kblk_struct.h"), "w") as f:
                f.write(struct)
            src = SRC
            if header_text is not None:   # a mutated k_blk_align.h: a copy of the tree layout the test source includes
                os.makedirs(os.path.join(d, "clean", "k", "runtime"))
                os.makedirs(os.path.join(d, "clean", "runtime", "tests"))
                with open(os.path.join(d, "clean", "k", "runtime", "k_blk_align.h"), "w") as f:
                    f.write(header_text)
                src = os.path.join(d, "clean", "runtime", "tests", "test_k_blk_align.cpp")
                shutil.copy(SRC, src)
            exe = os.path.join(d, "t.exe")
            r = subprocess.run([cxx, "-std=c++17", "-O1", "-I", d, src, "-o", exe], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return subprocess.run([exe], capture_output=True, text=True)

    def test_misaligned_pointers_are_refused(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout)
        _, names = kblk_struct(read(RT))
        self.assertEqual(len(names), 19)
        self.assertEqual(r.stdout.count("refused naming it"), 4 * len(names))

    def test_negative_control_check_removed(self):
        h = read(HDR)
        self.assertIn("if (e.v & 15u) return e.name;", h)
        r = self._run(h.replace("if (e.v & 15u) return e.name;", "(void)e;"))
        self.assertNotEqual(r.returncode, 0, "the test passed with the check removed")

    def test_negative_control_field_dropped(self):
        h = read(HDR)
        self.assertIn('{p.wz, "wz"}, ', h)
        r = self._run(h.replace('{p.wz, "wz"}, ', ""))
        self.assertNotEqual(r.returncode, 0, "the test passed with a pointer field left out of the check")
        self.assertIn("FAIL wz + 4", r.stdout)

    def test_runtime_wiring(self):
        src = read(RT)
        regions = [m.span() for m in re.finditer(r"// SMX-SELFCHECK-SOURCE BEGIN.*?// SMX-SELFCHECK-SOURCE END", src, re.S)]
        blocks = [m for m in re.finditer(re.escape(DEV) + r"\n(.*?)\n#endif", src, re.S)]
        calls = [m.start() for m in re.finditer(r"k_blk_misaligned\(", src)]
        self.assertEqual(len(calls), 2, "one check at create, one at every stage launch")
        for c in calls:
            self.assertTrue(any(b.start(1) <= c < b.end(1) for b in blocks), "a check outside the developer macros")
            self.assertFalse(any(a <= c < b for a, b in regions), "a check inside a self-check digest region")
        self.assertTrue(any('#include "k_blk_align.h"' in b.group(1) for b in blocks), "the header outside the macros")
        create = src[src.index("bool KRuntime::create("):src.index("bool KRuntime::smx_check(")]
        self.assertLess(create.index("k_blk_misaligned("), create.index("if (!smx_select("), "create checks before launching")
        self.assertIn("block_params(st, wdev_[st.name], net_[io.in]", create)
        launch = src[src.index("bool KRuntime::launch("):src.index("bool KRuntime::upload_inputs(")]
        self.assertIn('std::strncmp(kernel, "k_stage", 7) == 0', launch)
        for part in (create, launch):
            self.assertRegex(part, r"return fail\(.*\n?.*is not 16-byte aligned")


if __name__ == "__main__":
    unittest.main()
