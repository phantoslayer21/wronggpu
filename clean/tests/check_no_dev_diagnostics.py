# SPDX-License-Identifier: MIT
"""R1: a release build or a package never holds the developer diagnostics (executor=hip, the frame capture).

The runtime library says which kind of build it is: the text "wg-build dev-diagnostics=0" (release) or
"wg-build dev-diagnostics=1" (a developer build made with --dev-diagnostics; clean/runtime/interop/backend.cpp
wgBuildMarker). This checks libraries and packages for it:

    python clean/tests/check_no_dev_diagnostics.py <path> [...]

A path is a runtime library (wronggpu_runtime*.dll), a folder (searched for libraries, zip files and executables), a zip
file (every member), or an executable (the launcher embeds the payload's files as they are, so its bytes hold the marker).
Exit status 0 only when every runtime library found says dev-diagnostics=0, no other file says =1, and each path gave
evidence of at least one "dev-diagnostics=0" library: a library with no marker (an older build), a developer build, or a
path with nothing to check fails. This is the gate of the payload packager (clean/installer/tools/make_payload.py, which
refuses the payload) and of the release script on the final package (tools/release/check_package_diagnostics.py); the same
functions are used by clean/tests/test_dev_diagnostics_off.py in CI.
"""
import os
import re
import sys
import zipfile

MARKER = re.compile(rb"wg-build dev-diagnostics=([01])")


def is_runtime_name(name):
    n = os.path.basename(name).lower()
    return n.startswith("wronggpu_runtime") and n.endswith(".dll")


def check_bytes(data):
    """(True, text) when the library data says dev-diagnostics=0 once and never =1 (a runtime library)."""
    found = MARKER.findall(data)
    if not found:
        return False, "no 'wg-build dev-diagnostics=' marker (a build older than R1, or not the runtime library)"
    if any(v == b"1" for v in found):
        return False, "built with WG_DEV_DIAGNOSTICS (dev-diagnostics=1)"
    return True, "dev-diagnostics=0"


def check_artifact(name, data):
    """(ok, text, evidence) for one file by its name: a runtime library must carry the =0 marker; any other file only
    must not say =1. evidence: the file says dev-diagnostics=0 (so a package check knows it looked at a runtime)."""
    found = MARKER.findall(data)
    zero = any(v == b"0" for v in found)
    if is_runtime_name(name):
        ok, text = check_bytes(data)
        return ok, text, ok
    if any(v == b"1" for v in found):
        return False, "carries a developer runtime (dev-diagnostics=1)", False
    return True, "dev-diagnostics=0" if zero else "nothing to check", zero


def artifacts(path):
    """[(label, name, data)] of one path: the file itself, a folder's libraries / zips / executables, a zip's members."""
    out = []
    if os.path.isdir(path):
        for dirpath, _, names in os.walk(path):
            for n in sorted(names):
                low = n.lower()
                if is_runtime_name(n) or low.endswith((".zip", ".exe", ".asi")):
                    out += artifacts(os.path.join(dirpath, n))
        return out
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                out.append((path + "!" + info.filename, info.filename, z.read(info)))
        return out
    with open(path, "rb") as f:
        out.append((path, path, f.read()))
    return out


def check_path(path):
    """[(label, ok, text, evidence)]; an empty list or no evidence is a failure of the caller to find a library."""
    results = []
    for label, name, data in artifacts(path):
        ok, text, evidence = check_artifact(name, data)
        if ok and text == "nothing to check":
            continue  # a file that is neither a runtime nor says anything: not listed
        results.append((label, ok, text, evidence))
    return results


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    bad = False
    for path in argv:
        results = check_path(path)
        if not any(e for _, _, _, e in results):
            print(f"FAIL {path}: no runtime library with a 'dev-diagnostics=0' marker found")
            bad = True
        for label, ok, text, _ in results:
            print(("ok   " if ok else "FAIL ") + f"{label}: {text}")
            bad = bad or not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
