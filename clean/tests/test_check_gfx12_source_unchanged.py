# SPDX-License-Identifier: MIT
"""The gfx12 source guard (clean/tests/check_gfx12_source_unchanged.py) catches what it promises to: an edit that only a
shipped define combination reaches (post_w64__q5.co's WG_LZ_PROFILE=1), a preprocessing failure on either side (also the
same one on both), and a --scratch folder that does not exist yet. Needs a ROCm clang (ROCM_PATH or WG_ROCM; skipped without).

Run: python -m unittest clean.tests.test_check_gfx12_source_unchanged
"""
import os
import shutil
import tempfile
import unittest

from clean.tests import check_gfx12_source_unchanged as G

ROCM = os.environ.get("WG_ROCM") or os.environ.get("ROCM_PATH") or ""
ROOT = G.ROOT


def _have_clang():
    try:
        G.find_clang(ROCM)
        return True
    except (SystemExit, TypeError):
        return False


@unittest.skipUnless(ROCM and _have_clang(), "no ROCm clang (set ROCM_PATH or WG_ROCM)")
class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wg_gfx12_guard_", dir=os.environ.get("TMPDIR"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _form(self, name, text, opt="-O3"):
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            f.write(text)
        return [G.find_clang(ROCM), "-x", "hip", "--offload-arch=gfx1201", "--cuda-device-only", "-nogpuinc", "-E", "-P", opt,
                f"--rocm-path={ROCM}", f"--rocm-device-lib-path={os.path.join(ROCM, 'amdgcn', 'bitcode')}", p]

    def _run(self, base, tree, **kw):
        lines = []
        checked, bad = G.compare({"k": base}, {"k": tree}, report=lines.append, **kw)
        return checked, bad, lines

    def test_equal_text_passes(self):
        _, bad, _ = self._run(self._form("a1.hip", "int f() { return 1; }\n"), self._form("a2.hip", "int  f()\n{ return 1; }\n"))
        self.assertEqual(bad, 0)

    def test_different_text_fails(self):
        _, bad, _ = self._run(self._form("b1.hip", "int f() { return 1; }\n"), self._form("b2.hip", "int f() { return 2; }\n"))
        self.assertEqual(bad, 1)

    def test_same_preprocessing_error_on_both_sides_fails(self):
        # the same early #error, three diagnostic lines alike, and a real difference after it: must not count as equal
        head = '#error "shared failure"\n'
        _, bad, lines = self._run(self._form("c1.hip", head + "int f() { return 1; }\n"),
                                  self._form("c2.hip", head + "int f() { return 2; }\n"))
        self.assertEqual(bad, 1)
        self.assertIn("does not preprocess", lines[0])

    def test_error_on_one_side_fails(self):
        _, bad, _ = self._run(self._form("d1.hip", "int f() { return 1; }\n"), self._form("d2.hip", '#error "x"\n'))
        self.assertEqual(bad, 1)

    def test_listed_unbuildable_form_is_tolerated_only_when_listed(self):
        a, b = self._form("e1.hip", '#error "x"\n'), self._form("e2.hip", '#error "x"\n')
        self.assertEqual(self._run(a, b)[1], 1)
        self.assertEqual(self._run(a, b, allow_unbuildable={"k"})[1], 0)

    def test_form_in_one_tree_only_fails_unless_allowed(self):
        f = self._form("f1.hip", "int f() { return 1; }\n")
        self.assertEqual(G.compare({}, {"k": f}, report=lambda _: None)[1], 1)
        self.assertEqual(G.compare({}, {"k": f}, allow_new={"k"}, report=lambda _: None)[1], 0)

    def test_optimize_dependent_text_is_compared(self):
        body = "#ifdef __OPTIMIZE__\nint f() { return %d; }\n#else\nint f() { return 0; }\n#endif\n"
        _, bad, _ = self._run(self._form("o1.hip", body % 1), self._form("o2.hip", body % 2))
        self.assertEqual(bad, 1)
        # the same two files preprocessed without -O3 look equal: that is the false green the flag removes
        _, bad, _ = self._run(self._form("o1.hip", body % 1, opt="-O0"), self._form("o2.hip", body % 2, opt="-O0"))
        self.assertEqual(bad, 0)

    def test_recorded_commands_keep_the_optimisation_flag(self):
        forms = G.build_forms(ROOT, ROCM, "gfx1201", os.path.join(self.tmp, "s0"))
        for key in ("swin.co", "post_w64__q5.co", "k_io.co", "fg:fg_net1.hip", "default:kernels/mlpbench.hip"):
            if key in forms:
                self.assertIn("-O3", forms[key], key)
        self.assertIn("-O3", forms["swin.co"])

    def test_variant_the_base_does_not_name_is_paired_with_the_base_files(self):
        base, tree = os.path.join(self.tmp, "base"), os.path.join(self.tmp, "tree1")
        for root, ret in ((base, 1), (tree, 2)):
            os.makedirs(root)
            with open(os.path.join(root, "k.hip"), "w") as f:
                f.write("#if WG_MINW == 7\nint f() { return %d; }\n#else\nint f() { return 0; }\n#endif\n" % ret)
        cmd = lambda root: self._form("unused.hip", "")[:-1] + ["-DWG_MINW=7", os.path.join(root, "k.hip")]
        lines = []
        _, bad = G.compare({}, {"v.co": cmd(tree)}, report=lines.append, pair=G.pair_in(base, tree))
        self.assertEqual(bad, 1, lines)   # buildable on gfx12, text differs under the macro only the variant sets
        with open(os.path.join(tree, "k.hip"), "w") as f:
            f.write("#if WG_MINW == 7\nint f() { return 1; }\n#else\nint f() { return 0; }\n#endif\n")
        self.assertEqual(G.compare({}, {"v.co": cmd(tree)}, report=lambda _: None, pair=G.pair_in(base, tree))[1], 0)

    def _new_variant(self, name, message):
        base, tree = os.path.join(self.tmp, name + "b"), os.path.join(self.tmp, name + "t")
        for root in (base, tree):
            os.makedirs(root)
            with open(os.path.join(root, "k.hip"), "w") as f:
                f.write('#ifdef WG_G11\n#error "%s"\n#endif\nint f() { return 1; }\n' % message)
        cmd = self._form("unused.hip", "")[:-1] + ["-DWG_G11=1", os.path.join(tree, "k.hip")]
        lines = []
        bad = G.compare({}, {"v.co": cmd}, report=lines.append, pair=G.pair_in(base, tree))[1]
        return bad, lines

    def test_new_variant_that_stops_at_the_gfx11_only_error_is_skipped(self):
        bad, lines = self._new_variant("g1", "WG_F16_* are for the gfx11 matrix wrappers (wg_platform.h)")
        self.assertEqual(bad, 0, lines)
        self.assertTrue(lines and lines[0].startswith("skip v.co"))

    def test_new_variant_that_fails_for_another_reason_fails(self):
        # any other preprocessing error of a tree-only form is a failure, not a skip
        bad, lines = self._new_variant("g2", "gfx11 only")
        self.assertEqual(bad, 1, lines)

    def test_shared_error_is_skipped_only_when_it_is_the_gfx11_only_error(self):
        g11 = '#error "WG_MERGE_REUSE is for the gfx11 matrix wrappers (wg_platform.h)"\n'
        a, b = self._form("h1.hip", g11 + "int f() { return 1; }\n"), self._form("h2.hip", g11 + "int f() { return 2; }\n")
        self.assertEqual(G.compare({"swin__mgx.co": a}, {"swin__mgx.co": b}, report=lambda _: None)[1], 0)
        other = '#error "WG_F16_Q makes the F16 operands that WG_F16_A or WG_F16_M keep: set one of them"\n'
        a, b = self._form("h3.hip", other), self._form("h4.hip", other)
        self.assertEqual(G.compare({"swin.co": a}, {"swin.co": b}, report=lambda _: None)[1], 1)

    def _clean_copy(self):
        dst = os.path.join(self.tmp, "tree")
        os.makedirs(dst)
        shutil.copytree(os.path.join(ROOT, "clean"), os.path.join(dst, "clean"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.co", "*.so"))
        return dst

    def test_edit_under_the_q5_define_only_is_caught(self):
        # output_quality5.h is included only under WG_LZ_PROFILE == 1: neither the default form of post.hip nor its wave64 form
        # sees it; the shipped post_w64__q5.co form does
        tree = self._clean_copy()
        path = os.path.join(tree, "clean", "kernels", "output_quality5.h")
        with open(path) as f:
            text = f.read()
        self.assertIn("0x1.46f5560000000p-1f", text)
        with open(path, "w") as f:
            f.write(text.replace("0x1.46f5560000000p-1f", "0x1.46f5570000000p-1f"))
        rc = G.main(["--rocm", ROCM, "--base-tree", ROOT, "--tree", tree, "--scratch", os.path.join(self.tmp, "s1")])
        self.assertEqual(rc, 1)
        forms = G.build_forms(ROOT, ROCM, "gfx1201", os.path.join(self.tmp, "s1"))
        self.assertTrue(any("-DWG_LZ_PROFILE=1" in c for c in forms["post_w64__q5.co"]))

    def test_unchanged_tree_passes_and_missing_scratch_folder_is_created(self):
        scratch = os.path.join(self.tmp, "does", "not", "exist")
        # dx12/tests/gfx11_hip_ref.hip is a host program for the gfx11 helper tests (hip_runtime.h, not a gfx12 object)
        gfx11_only = [f for f in ("default:dx12/tests/gfx11_hip_ref.hip",) if os.path.isfile(os.path.join(ROOT, "clean", f[8:]))]
        self.assertEqual(G.main(["--rocm", ROCM, "--base-tree", ROOT, "--scratch", scratch,
                                 "--allow-unbuildable", *gfx11_only]), 0)
        self.assertTrue(os.path.isdir(scratch))


if __name__ == "__main__":
    unittest.main()
