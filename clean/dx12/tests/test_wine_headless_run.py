# SPDX-License-Identifier: MIT
"""wine_headless_run.sh refuses every setup in which Wine could open a window or use a memory-backed scratch folder
(it exits 9 before it starts anything). No Wine is needed: the refusals come first."""
import os
import subprocess
import tempfile
import unittest

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine_headless_run.sh")


def run(prefix_reg, scratch):
    with tempfile.TemporaryDirectory(dir=os.path.dirname(RUN)) as d:   # not /tmp: the script refuses it
        pfx = os.path.join(d, "pfx")
        os.makedirs(pfx)
        with open(os.path.join(pfx, "user.reg"), "w") as f:
            f.write(prefix_reg)
        env = {"PATH": os.environ["PATH"], "WG_WINE_PREFIX": pfx, "WG_WINE_SCRATCH": scratch(d)}
        return subprocess.run([RUN, "--", "true"], env=env, capture_output=True, text=True)


NULL = '[Software\\\\Wine\\\\Drivers]\n"Graphics"="null"\n'


class TestWineHeadlessRun(unittest.TestCase):
    def test_a_prefix_with_a_graphics_driver_is_refused(self):
        r = run('[Software\\\\Wine\\\\Drivers]\n"Graphics"="x11,wayland"\n', lambda d: os.path.join(d, "s"))
        self.assertEqual(r.returncode, 9, r.stderr)
        self.assertIn("null graphics", r.stderr)

    def test_a_prefix_without_the_setting_is_refused(self):
        self.assertEqual(run("", lambda d: os.path.join(d, "s")).returncode, 9)

    def test_memory_backed_and_session_scratch_is_refused(self):
        for bad in ("/tmp/wg_x", "/run/user/1000/wg_x"):
            r = run(NULL, lambda d: bad)
            self.assertEqual(r.returncode, 9, bad)
            self.assertIn("refusing scratch", r.stderr)

    def test_missing_settings_are_refused(self):
        r = subprocess.run([RUN, "--", "true"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
