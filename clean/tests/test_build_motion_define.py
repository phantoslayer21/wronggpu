# SPDX-License-Identifier: MIT
"""Issue #55: `build.py --motion-outres-only --extra-archs ...` starts one child build per extra architecture; the
child's command line must carry the parent's --define (and --report) like the ordinary branch's child does. No compiler
and no GPU: the subprocess call and the parent's own kernel builds are stubs, nothing is executed and nothing is written."""
import contextlib
import io
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from clean.runner import build


def child_commands(argv):
    """The child commands `build.main` starts for the extra architectures, for the given extra arguments."""
    started = []

    def fake_run(cmd, *args, **kwargs):
        started.append((list(cmd), dict(kwargs.get("env") or {})))
        return subprocess.CompletedProcess(cmd, 0)

    with tempfile.TemporaryDirectory() as out, patch.object(build, "EXTRA_ARCHS", []), \
            patch.object(build, "build_motion_outres") as parent, patch.object(build.subprocess, "run", fake_run), \
            contextlib.redirect_stdout(io.StringIO()):
        # (--rocm is given: the default would be the ROCM_PATH of the environment this runs in)
        build.main(["--out", out, "--motion-outres-only", "--rocm", "dummy-rocm", "--no-device-libs", "--clang", "dummy-clang",
                    "--extra-archs", "gfx1200,gfx1201"] + argv)
        parent.assert_called_once()
    return started


def values_of(cmd, flag):
    return [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == flag]


class MotionOutresChild(unittest.TestCase):
    def test_every_define_reaches_the_child(self):
        started = child_commands(["--define", "WG_GELU_MODE=1", "--define", "OTHER=2"])
        self.assertEqual(len(started), 1)  # one extra architecture besides the primary one
        cmd, env = started[0]
        self.assertIn("--motion-outres-only", cmd)
        self.assertEqual(values_of(cmd, "--define"), ["WG_GELU_MODE=1", "OTHER=2"])
        self.assertIn(env.get("WG_GPU_ARCH"), ("gfx1200", "gfx1201"))

    def test_report_reaches_the_child(self):
        cmd = child_commands(["--report"])[0][0]
        self.assertIn("--report", cmd)
        self.assertEqual(values_of(cmd, "--define"), [])

    def test_without_them_the_child_has_neither(self):
        cmd = child_commands([])[0][0]
        self.assertNotIn("--report", cmd)
        self.assertNotIn("--define", cmd)

    def test_the_options_that_were_forwarded_before_still_are(self):
        cmd = child_commands(["--soft-fp8", "--no-resource-gate", "--define", "X=1"])[0][0]
        for flag in ("--no-device-libs", "--soft-fp8", "--no-resource-gate"):
            self.assertIn(flag, cmd)
        self.assertEqual(values_of(cmd, "--clang"), ["dummy-clang"])


if __name__ == "__main__":
    unittest.main()
