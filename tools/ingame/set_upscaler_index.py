# SPDX-License-Identifier: MIT
"""Set UpscalerIndex in the [WrongGPU] section of the game's OptiScaler.ini (our install's file, backed up by
game_backup.py; restored by game_restore.py). 0 = WrongGPU Model M, 1 = WrongGPU Model L. Only that one line changes; bytes and line endings otherwise kept.
usage: set_upscaler_index.py <OptiScaler.ini> <index>"""
import re
import sys

path, idx = sys.argv[1], sys.argv[2].encode()
raw = open(path, 'rb').read()
nl = b'\r\n' if b'\r\n' in raw else b'\n'
lines = raw.split(nl)
sec = None
hits = []
for i, l in enumerate(lines):
    s = l.strip()
    if s.startswith(b'[') and s.endswith(b']'):
        sec = s
    elif sec == b'[WrongGPU]' and re.match(rb'^UpscalerIndex\s*=', s):
        hits.append(i)
if len(hits) != 1:
    raise SystemExit('expected one UpscalerIndex line in [WrongGPU], found %d' % len(hits))
old = lines[hits[0]]
lines[hits[0]] = b'UpscalerIndex=' + idx
open(path, 'wb').write(nl.join(lines))
print('line %d: %r -> %r' % (hits[0] + 1, old, lines[hits[0]]))
