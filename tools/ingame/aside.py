# SPDX-License-Identifier: MIT
"""Move files out of a game tree for a test and put them back byte for byte (e.g. another mod's files).
The files' contents are never parsed: only hashed (SHA-256), stat'ed and renamed (same volume: a rename keeps bytes,
timestamps and attributes).

    python aside.py out     <aside folder> <file> [<file> ...]   -> moves, writes <aside folder>\\MANIFEST.json
    python aside.py restore <aside folder>                       -> moves back to the original paths
    python aside.py verify  <aside folder>                       -> checks the original paths against the manifest
"""
import ctypes
import hashlib
import json
import os
import sys
from ctypes import wintypes

k32 = ctypes.WinDLL('kernel32', use_last_error=True)


class FILETIME(ctypes.Structure):
    _fields_ = [('lo', wintypes.DWORD), ('hi', wintypes.DWORD)]


class WIN32_FILE_ATTRIBUTE_DATA(ctypes.Structure):
    _fields_ = [('attrs', wintypes.DWORD), ('ctime', FILETIME), ('atime', FILETIME), ('mtime', FILETIME),
                ('size_hi', wintypes.DWORD), ('size_lo', wintypes.DWORD)]


def info(path):
    d = WIN32_FILE_ATTRIBUTE_DATA()
    if not k32.GetFileAttributesExW(path, 0, ctypes.byref(d)):
        raise OSError(ctypes.get_last_error(), 'GetFileAttributesExW', path)
    ft = lambda f: (f.hi << 32) | f.lo
    h = hashlib.sha256()
    with open(path, 'rb') as f:   # hashed only, never parsed
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return {'size': (d.size_hi << 32) | d.size_lo, 'sha256': h.hexdigest(), 'attributes': d.attrs,
            'creation_filetime': ft(d.ctime), 'last_write_filetime': ft(d.mtime)}


def out(folder, files):
    os.makedirs(folder, exist_ok=False)
    man = []
    for i, src in enumerate(files):
        src = os.path.abspath(src)
        e = {'original_path': src, 'aside_name': '%02d_%s' % (i, os.path.basename(src))}
        e.update(info(src))
        dst = os.path.join(folder, e['aside_name'])
        os.rename(src, dst)          # same volume: bytes, timestamps and attributes kept
        after = info(dst)
        e['aside_check'] = 'ok' if all(after[x] == e[x] for x in ('size', 'sha256', 'creation_filetime', 'last_write_filetime', 'attributes')) else 'MISMATCH %r' % after
        man.append(e)
        print('moved %s -> %s (%s)' % (src, dst, e['aside_check']))
    json.dump(man, open(os.path.join(folder, 'MANIFEST.json'), 'w', encoding='utf-8'), indent=2)
    print('manifest: %s' % os.path.join(folder, 'MANIFEST.json'))


def restore(folder):
    man = json.load(open(os.path.join(folder, 'MANIFEST.json'), encoding='utf-8'))
    for e in man:
        src = os.path.join(folder, e['aside_name'])
        if os.path.exists(e['original_path']):
            raise SystemExit('refusing: %s exists again' % e['original_path'])
        os.rename(src, e['original_path'])
        print('restored %s' % e['original_path'])


def verify(folder):
    man = json.load(open(os.path.join(folder, 'MANIFEST.json'), encoding='utf-8'))
    bad = 0
    for e in man:
        now = info(e['original_path'])
        diff = [x for x in ('size', 'sha256', 'creation_filetime', 'last_write_filetime', 'attributes') if now[x] != e[x]]
        bad += bool(diff)
        print('%s %s%s' % ('ok      ' if not diff else 'MISMATCH', e['original_path'], '' if not diff else ' ' + repr(diff)))
    print('verify: %d file(s), %d mismatch(es)' % (len(man), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'out':
        out(sys.argv[2], sys.argv[3:])
    elif cmd == 'restore':
        restore(sys.argv[2])
    else:
        sys.exit(verify(sys.argv[2]))
