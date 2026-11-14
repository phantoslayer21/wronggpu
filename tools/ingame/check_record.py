# SPDX-License-Identifier: MIT
"""Check a game's WrongGPU install record against the folder: every recorded 'copy' file's SHA-256 against the
record's installed hash (files owned by wg / wg-optibundle only), and list the files in the exe folder's top level and
OptiScaler\\ that the record doesn't name. Vendor-named files (nvngx*, *dlss*, sl.*, nvapi*) are never opened.
usage: check_record.py <exe folder>"""
import fnmatch
import hashlib
import os
import sys

d = sys.argv[1]
rec = open(os.path.join(d, 'wronggpu-setup', 'install.record'), encoding='utf-8').read().splitlines()
owner = {}
copies = []
for l in rec:
    f = l.split('\t')
    if f[0] == 'owner' and len(f) >= 4:
        owner[(f[1], f[2])] = f[3]
    elif f[0] == 'copy':
        copies.append((f[4], f[3]))
    elif f[0] == 'disable':
        copies.append((f[3], None))
VENDOR = ['nvngx*', '*dlss*', 'sl.*', 'nvapi*']
bad = 0
named = set()
for rel, inst in copies:
    named.add(rel.lower())
    p = os.path.join(d, rel)
    own = owner.get(('copy', rel), owner.get(('disable', rel), '?'))
    if inst is None:
        print('%-8s %-14s %s present=%s (moved aside by the install; backup in wronggpu-setup\\backup)' % ('disable', own, rel, os.path.exists(p)))
        continue
    if not os.path.exists(p):
        print('MISSING  %-14s %s' % (own, rel)); bad += 1; continue
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    ok = h == inst
    bad += not ok
    print('%-8s %-14s %s' % ('ok' if ok else 'CHANGED', own, rel))
print('--- not named by the record (top level and OptiScaler\\, names only):')
for sub in ('', 'OptiScaler'):
    base = os.path.join(d, sub)
    for n in sorted(os.listdir(base)):
        rel = os.path.join(sub, n) if sub else n
        if os.path.isfile(os.path.join(base, n)) and rel.lower() not in named:
            tag = 'vendor' if any(fnmatch.fnmatch(n.lower(), v) for v in VENDOR) else ''
            print('  %s %s' % (rel, tag))
print('record check: %d problem(s)' % bad)
