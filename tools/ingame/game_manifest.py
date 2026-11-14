# SPDX-License-Identifier: MIT
"""Manifest of a game folder for an in-game run, and its verification afterwards.

    python game_manifest.py make   <game folder> <manifest.json>
    python game_manifest.py verify <game folder> <manifest.json>      (exit 1 on any difference)
    python game_manifest.py scan   <game folder>                      (foreign-mod check only)

Every file: relative path, size, mtime. SHA-256 for files up to 256 MiB outside the game's archive folders, EXCEPT
vendor-named files (nvngx*, *dlss*, sl.*, nvapi*, NvLowLatency*), which are recorded by name, size and mtime only and
never opened (owner's rule). scan lists names that point at another or older mod (version.dll, winmm.dll, dxgi.dll,
OptiScaler*, *.asi, nvngx.dll, d3d12.dll in bin/x64 ...) so the tester can check them against the WrongGPU install
record before anything runs. Read-only on the game folder.
"""
import fnmatch
import hashlib
import json
import os
import sys

VENDOR = ['nvngx*', '*dlss*', 'sl.*', 'nvapi*', 'nvlowlatency*', 'nvperf*']
SKIP_HASH_DIRS = ['archive', 'r6/cache', 'tools']
HASH_MAX = 256 * 1024 * 1024
SUSPECT = ['version.dll', 'winmm.dll', 'dxgi.dll', 'd3d12.dll', 'dinput8.dll', 'optiscaler*', '*.asi', 'nvngx.dll',
           'nvapi64.dll', 'wronggpu*', 'amd_fidelityfx*', 'libxess*', 'reshade*', 'dxvk*', 'winhttp.dll']


def vendor(name):
    n = name.lower()
    return any(fnmatch.fnmatch(n, p) for p in VENDOR)


def walk(root):
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for n in sorted(fn):
            p = os.path.join(dp, n)
            yield os.path.relpath(p, root).replace('\\', '/'), p


def entry(rel, p):
    st = os.stat(p)
    e = {'size': st.st_size, 'mtime': int(st.st_mtime)}
    low = rel.lower()
    if vendor(os.path.basename(rel)):
        e['sha256'] = 'not read (vendor file)'
    elif st.st_size <= HASH_MAX and not any(low.startswith(d + '/') for d in SKIP_HASH_DIRS):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        e['sha256'] = h.hexdigest()
    return e


def make(root, out):
    m = {rel: entry(rel, p) for rel, p in walk(root)}
    json.dump({'root': root, 'files': m}, open(out, 'w', encoding='utf-8'), indent=0, sort_keys=True)
    print('manifest: %d files -> %s' % (len(m), out))


def verify(root, man):
    old = json.load(open(man, encoding='utf-8'))['files']
    new = {rel: entry(rel, p) for rel, p in walk(root)}
    added = sorted(set(new) - set(old))
    gone = sorted(set(old) - set(new))
    changed = sorted(r for r in set(old) & set(new) if old[r] != new[r])
    for tag, lst in (('added', added), ('removed', gone), ('changed', changed)):
        for r in lst:
            print('%-8s %s' % (tag, r))
    print('verify: %d files; added %d, removed %d, changed %d' % (len(new), len(added), len(gone), len(changed)))
    return 1 if (added or gone or changed) else 0


def scan(root):
    hits = [rel for rel, _ in walk(root) if any(fnmatch.fnmatch(os.path.basename(rel).lower(), p) for p in SUSPECT)
            and not rel.lower().startswith('archive/')]
    for h in hits:
        print('check: ' + h)
    print('scan: %d name(s) to check against the WrongGPU install record' % len(hits))


if __name__ == '__main__':
    cmd, root = sys.argv[1], sys.argv[2]
    if cmd == 'make':
        make(root, sys.argv[3])
    elif cmd == 'verify':
        sys.exit(verify(root, sys.argv[3]))
    else:
        scan(root)
