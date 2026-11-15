# SPDX-License-Identifier: MIT
"""The HIP device compile command carries a fixed compilation-unit id; nothing is executed, the paths are dummies
(approved for each test as with --force-sdk, so the SDK policy boundary in device_command lets them through)."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from clean.runner import build, sdk

CLANG = "dummy-clang"
ROCM = os.path.join("dummy-rocm", "7.2")
SRC = "src.hip"


def cuid_flags(cmd):
    return [a for a in cmd if a.startswith("-cuid")]


class DeviceCommandCuid(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(sdk, "_approved", set()))
        stack.enter_context(patch.object(sdk, "_spellings", {}))
        stack.enter_context(patch.object(sdk, "_bannered", set()))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))   # the policy banner
        sdk.approve(ROCM)
        sdk.approve(CLANG)

    def test_unapproved_dummies_are_refused(self):
        """Without the approval the boundary refuses the dummy SDK (the reason for setUp)."""
        with patch.object(sdk, "_approved", set()), self.assertRaises(SystemExit):
            build.device_command(CLANG, ROCM, SRC, "swin.co")

    def test_cuid_flag_is_in_the_command(self):
        cmd = build.device_command(CLANG, ROCM, SRC, os.path.join("out", "post_w64.co"))
        self.assertEqual(cuid_flags(cmd), ["-cuid=wg_post_w64"])

    def test_cuid_does_not_depend_on_the_output_folder(self):
        a = build.device_command(CLANG, ROCM, SRC, os.path.join("one", "folder", "post_w64.co"))
        b = build.device_command(CLANG, ROCM, SRC, os.path.join("another", "place", "wg", "post_w64.co"))
        self.assertEqual(cuid_flags(a), cuid_flags(b))
        self.assertEqual(len(cuid_flags(a)), 1)

    def test_cuid_has_no_path_in_it(self):
        flag = cuid_flags(build.device_command(CLANG, ROCM, SRC, os.path.join("a b", "c-d", "post.w64__orig.co")))[0]
        value = flag.split("=", 1)[1]
        self.assertTrue(value.startswith("wg_"))
        self.assertEqual(value, "wg_post_w64__orig")
        self.assertNotIn(os.sep, value)
        self.assertNotIn("/", value)
        self.assertNotIn(" ", value)

    def test_cuid_follows_the_output_name(self):
        a = cuid_flags(build.device_command(CLANG, ROCM, SRC, "swin.co"))
        b = cuid_flags(build.device_command(CLANG, ROCM, SRC, "post.co"))
        self.assertEqual(a, ["-cuid=wg_swin"])
        self.assertNotEqual(a, b)

    def test_cuid_is_also_in_the_compile_check_command(self):
        cmd = build.device_command(CLANG, ROCM, SRC, "swin.co", device_libs=False)
        self.assertEqual(cuid_flags(cmd), ["-cuid=wg_swin"])
        self.assertIn("-nogpulib", cmd)


if __name__ == "__main__":
    unittest.main()
