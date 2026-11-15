# SPDX-License-Identifier: MIT
"""Release step (R1): the FINAL package holds no developer diagnostics (executor=hip, the frame capture).

    python tools/release/check_package_diagnostics.py <final zip | package folder | launcher exe | runtime dll> [...]

Runs clean/tests/check_no_dev_diagnostics.py on each path: every runtime library found (also inside a zip, and inside the
launcher executable, which embeds the payload's files as they are) must say "wg-build dev-diagnostics=0"; a developer build,
a library without the marker, or a path with no runtime library fails (exit 1). Run it on the package that is published,
after it is made: the payload packager (clean/installer/tools/make_payload.py) refuses a developer runtime at build time, this
checks what was actually produced.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "clean", "tests"))
import check_no_dev_diagnostics  # noqa: E402

if __name__ == "__main__":
    sys.exit(check_no_dev_diagnostics.main(sys.argv[1:]))
