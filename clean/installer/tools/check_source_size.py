#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every source file of the installer / launcher stays below the spec side's scan limit (tools/nv_leak_scan.py: a file
over 256 KB is reported). The size is counted as git stores the file (LF line ends), as the scan reads it.

    check_source_size.py [<folder> ...] [--limit <bytes>]

Default folder: the installer folder this script is in (src, tests, tools, CMakeLists.txt, README.md). Exit 1 when a
file is at or over the limit. Written for this project; Python standard library only.
"""
import argparse
import os
import sys

LIMIT = 256 * 1024
SKIP_DIRS = {"build", "__pycache__", ".git"}
TEXT = (".cpp", ".h", ".hpp", ".c", ".py", ".txt", ".md", ".rc", ".manifest", ".sh", ".ps1", ".json", ".cmake")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*", default=[here])
    ap.add_argument("--limit", type=int, default=LIMIT)
    a = ap.parse_args()
    bad, count, largest = [], 0, (0, "")
    for folder in a.folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("build-"))
            for fn in sorted(filenames):
                if not fn.lower().endswith(TEXT) and fn != "CMakeLists.txt":
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "rb") as f:
                    size = len(f.read().replace(b"\r\n", b"\n"))
                count += 1
                largest = max(largest, (size, path))
                if size >= a.limit:
                    bad.append((size, path))
    for size, path in bad:
        print("source-size: %s is %d bytes (limit: below %d)" % (path, size, a.limit))
    print("source-size: %d files checked, largest %s (%d bytes), %d at or over the limit" % (count, largest[1], largest[0], len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
