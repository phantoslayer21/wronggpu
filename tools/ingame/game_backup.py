# SPDX-License-Identifier: MIT
"""Backups before an in-game run (copies, nothing moved or changed):
  - every file the game's WrongGPU install record names (copy entries), the record folder wronggpu-setup\\ (with its
    backup\\, which holds the owner's earlier OptiScaler dxgi.dll: copied as a file, never opened), OptiScaler.ini,
    wronggpu*.ini / logs in the exe folder -> <out>\\game\\<relative path> (copy2: timestamps kept);
  - %LOCALAPPDATA%\\WrongGPU except model\\ (list + SHA-256 in localappdata-before.json, copies in <out>\\lad\\).
usage: game_backup.py <exe folder> <out folder>"""
import hashlib
import json
import os
import shutil
import sys

exe_dir, out = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
rels = set()
for l in open(os.path.join(exe_dir, 'wronggpu-setup', 'install.record'), encoding='utf-8').read().splitlines():
    f = l.split('\t')
    if f[0] == 'copy':
        rels.add(f[4])
for dp, dn, fn in os.walk(os.path.join(exe_dir, 'wronggpu-setup')):
    for n in fn:
        rels.add(os.path.relpath(os.path.join(dp, n), exe_dir))
for n in os.listdir(exe_dir):
    low = n.lower()
    if os.path.isfile(os.path.join(exe_dir, n)) and (low.startswith('wronggpu') or low == 'optiscaler.ini'):
        rels.add(n)
for sub in ('WrongGPU', os.path.join('OptiScaler', 'custom', 'WrongGPU')):
    d = os.path.join(exe_dir, sub)
    for dp, dn, fn in os.walk(d):
        for n in fn:
            rels.add(os.path.relpath(os.path.join(dp, n), exe_dir))
copied = []
for r in sorted(rels):
    src = os.path.join(exe_dir, r)
    if not os.path.isfile(src):
        continue
    dst = os.path.join(out, 'game', r)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(r)
lad = os.path.join(os.environ['LOCALAPPDATA'], 'WrongGPU')
lad_list = {}
for dp, dn, fn in os.walk(lad):
    rel_dp = os.path.relpath(dp, lad)
    if rel_dp.split(os.sep)[0] == 'model':
        continue
    for n in fn:
        p = os.path.join(dp, n)
        r = os.path.relpath(p, lad)
        lad_list[r] = {'size': os.path.getsize(p), 'sha256': sha(p)}
        dst = os.path.join(out, 'lad', r)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
json.dump(lad_list, open(os.path.join(out, 'localappdata-before.json'), 'w'), indent=1, sort_keys=True)
json.dump(copied, open(os.path.join(out, 'game-backup-list.json'), 'w'), indent=1)
print('game files backed up: %d; %%LOCALAPPDATA%%\\WrongGPU files (model\\ excluded): %d' % (len(copied), len(lad_list)))
