# SPDX-License-Identifier: MIT
"""Vendored-copy check (CI): OptiScaler WrongGPU edition holds a byte-identical copy of
clean/wgshared/include/wg_model_entries.h (the order of WrongGPU's model entries in the version list its FFX DLL gives a
host; OptiScaler stores the player's entry as a position in that list). The two must never differ.

This check fails when
  - clean/wgshared/include/wg_model_entries.h no longer has the SHA-256 pinned below (a change to the order needs the
    fork's copy updated first, then FORK_COMMIT and HEADER_SHA256 here), or
  - the fork's copy at FORK_COMMIT is not byte-identical to it (fetched from GitHub; --offline skips this part).
Run from the repository root: python tools/ci/check_vendored_model_entries.py [--offline]
"""
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

HEADER = 'clean/wgshared/include/wg_model_entries.h'
HEADER_SHA256 = 'dbc42cf67378a610c35935829683f3928a98e83f1bfb49b53e1f36152da9ab4f'
FORK_REPO = 'phantoslayer21/OptiScaler'
FORK_COMMIT = 'f31d091df2989f70c1e4ab296942de69240af227'
FORK_PATH = 'OptiScaler/include/wronggpu/wg_model_entries.h'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main(argv):
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    ours = (root / HEADER).read_bytes()
    problems = []

    if sha256(ours) != HEADER_SHA256:
        problems.append('%s has SHA-256 %s, pinned %s: update the fork\'s copy (%s), then FORK_COMMIT and '
                        'HEADER_SHA256 in this script' % (HEADER, sha256(ours), HEADER_SHA256, FORK_PATH))

    if '--offline' not in argv:
        url = 'https://raw.githubusercontent.com/%s/%s/%s' % (FORK_REPO, FORK_COMMIT, FORK_PATH)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                theirs = r.read()
        except OSError as e:
            problems.append('could not fetch the fork\'s copy (%s): %s' % (url, e))
        else:
            if theirs != ours:
                problems.append('the fork\'s copy at %s (SHA-256 %s) differs from %s (SHA-256 %s)'
                                % (FORK_COMMIT[:12], sha256(theirs), HEADER, sha256(ours)))

    for p in problems:
        print('FAIL: ' + p)
    if problems:
        return 1
    print('vendored wg_model_entries.h: %s matches the pin%s' % (HEADER, '' if '--offline' in argv else
                                                                 ' and the fork\'s copy at ' + FORK_COMMIT[:12]))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
