# SPDX-License-Identifier: MIT
"""Line-ending check (CI): every tracked file is stored as committed, without line-ending conversion.

The repository's .gitattributes says `* -text`: git never converts line endings, so the converter maps and the test
vectors keep the exact bytes their SHA-256 digests describe. This check fails when
  - .gitattributes no longer starts the rules with `* -text`, or
  - a tracked text file holds a CR LF pair (the repository is LF only; a CRLF file needs a deliberate exception here), or
  - git reports any file as a text file under conversion (`attr/text` or `attr/text=auto` in `git ls-files --eol`).
Run from the repository root: python tools/ci/check_line_endings.py
"""
import subprocess
import sys
from pathlib import Path

# repository-relative paths of files that are meant to hold CR LF. The preset-L converter map is kept byte for byte as
# the spec side delivered it (CR LF; data/ is spec-side output, never edited by the clean team; .gitattributes: -text).
CRLF_ALLOWED = {"data/preset-L.v1.map.json"}


def main():
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    problems = []
    rules = [l.strip() for l in (root / '.gitattributes').read_text(encoding='utf-8').splitlines()
             if l.strip() and not l.lstrip().startswith('#')]
    if not rules or rules[0].split()[:2] != ['*', '-text']:
        problems.append('.gitattributes: the first rule must be "* -text" (found %r)' % (rules[0] if rules else None))
    eol = subprocess.check_output(['git', 'ls-files', '--eol', '-z'], cwd=root).decode('utf-8').split('\0')
    checked = 0
    for entry in eol:
        if not entry:
            continue
        meta, path = entry.split('\t', 1)
        fields = meta.split()
        index_eol, attr = fields[0], fields[2] if len(fields) > 2 else ''
        checked += 1
        if attr.startswith('attr/text') or attr.startswith('attr/auto'):
            problems.append('%s: converted by git (%s)' % (path, attr))
        if index_eol in ('i/crlf', 'i/mixed') and path not in CRLF_ALLOWED:
            problems.append('%s: CR LF in the committed file (%s)' % (path, index_eol))
    for p in problems:
        print(p)
    print('line endings: %d files checked, %d problem(s)' % (checked, len(problems)))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
