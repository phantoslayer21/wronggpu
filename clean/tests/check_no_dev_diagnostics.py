# SPDX-License-Identifier: MIT
"""R1: a release build or a package never holds the developer diagnostics (executor=hip, the frame capture).

The runtime library says which kind of build it is: the text "wg-build dev-diagnostics=0" (release) or
"wg-build dev-diagnostics=1" (a developer build made with --dev-diagnostics; clean/runtime/interop/backend.cpp
wgBuildMarker). This checks libraries for it:

    python clean/tests/check_no_dev_diagnostics.py <runtime dll, or a folder searched for wronggpu_runtime*.dll> [...]

Exit status 0 only when every library found says dev-diagnostics=0. A library that has no marker (an older build), says
dev-diagnostics=1, or a path with no library at all fails: a package check that finds nothing must not pass. The release
script runs this on the packaged library (the same function is used by clean/tests/test_dev_diagnostics_off.py in CI).
"""
import os
import re
import sys

MARKER = re.compile(rb"wg-build dev-diagnostics=([01])")


def check_bytes(data):
    """(True, text) when the library data says dev-diagnostics=0 once and never =1."""
    found = MARKER.findall(data)
    if not found:
        return False, "no 'wg-build dev-diagnostics=' marker (a build older than R1, or not the runtime library)"
    if any(v == b"1" for v in found):
        return False, "built with WG_DEV_DIAGNOSTICS (dev-diagnostics=1)"
    return True, "dev-diagnostics=0"


def libraries(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for dirpath, _, names in os.walk(path):
        for n in names:
            if n.lower().startswith("wronggpu_runtime") and n.lower().endswith(".dll"):
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def check_path(path):
    """[(library, ok, text)]; an empty list is a failure of the caller to find a library."""
    results = []
    for lib in libraries(path):
        with open(lib, "rb") as f:
            ok, text = check_bytes(f.read())
        results.append((lib, ok, text))
    return results


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = False
    for path in argv:
        results = check_path(path)
        if not results:
            print(f"FAIL {path}: no wronggpu_runtime library found")
            bad = True
        for lib, ok, text in results:
            print(("ok   " if ok else "FAIL ") + f"{lib}: {text}")
            bad = bad or not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
