# SPDX-License-Identifier: MIT
"""After an in-game run, for any exe folder (the game root itself or a subfolder such as <root>\\bin\\x64): put back every file of the
backup (copy2), remove the files the update added under <exe dir>\\wronggpu-setup\\previous\\ (listed as 'added' by
the manifest check, file by file, then their now-empty folders), and bring %LOCALAPPDATA%\\WrongGPU (model\\ excluded)
back to localappdata-before.json. Other added files are listed, not removed.
usage: game_restore.py <game root> <exe dir> <run folder (with backup\\)> <verify-after-run.txt>"""
import hashlib
import json
import os
import shutil
import sys

root, exe_dir, a4, verify_txt = sys.argv[1:5]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
copied = json.load(open(os.path.join(a4, 'backup', 'game-backup-list.json')))
for r in copied:
    shutil.copy2(os.path.join(a4, 'backup', 'game', r), os.path.join(exe_dir, r))
print('restored from backup: %d files' % len(copied))
rel_exe = os.path.relpath(exe_dir, root).replace(os.sep, '/')
prefix = ('' if rel_exe == '.' else rel_exe + '/') + 'wronggpu-setup/previous/'
added = [l.split(None, 1)[1].strip() for l in open(verify_txt, encoding='utf-8') if l.startswith('added')]
removed, dirs = [], set()
for rel in added:
    if not rel.startswith(prefix):
        print('NOT REMOVED (outside %s): %s' % (prefix, rel))
        continue
    p = os.path.join(root, rel.replace('/', os.sep))
    os.remove(p)
    removed.append(rel)
    d = os.path.dirname(p)
    while os.path.normcase(d) != os.path.normcase(os.path.join(exe_dir, 'wronggpu-setup')):
        dirs.add(d)
        d = os.path.dirname(d)
for d in sorted(dirs, key=len, reverse=True):
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
print('removed added files: %d; empty folders removed: %d' % (len(removed), sum(1 for d in dirs if not os.path.exists(d))))
lad = os.path.join(os.environ['LOCALAPPDATA'], 'WrongGPU')
before = json.load(open(os.path.join(a4, 'backup', 'localappdata-before.json')))
now = {}
for dp, dn, fn in os.walk(lad):
    if os.path.relpath(dp, lad).split(os.sep)[0] == 'model':
        continue
    for n in fn:
        now[os.path.relpath(os.path.join(dp, n), lad)] = os.path.join(dp, n)
changed = [r for r in before if r not in now or sha(now[r]) != before[r]['sha256']]
extra = sorted(set(now) - set(before))
for r in changed:
    dst = os.path.join(lad, r)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(a4, 'backup', 'lad', r), dst)
print('%%LOCALAPPDATA%%\\WrongGPU: restored %d changed file(s): %s; files added since the backup (left, listed): %s'
      % (len(changed), changed, extra))
