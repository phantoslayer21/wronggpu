# SPDX-License-Identifier: MIT
"""A game's graphics settings for an in-game run: back the file up byte for byte, change only the named keys, put the
backup back and check its SHA-256.

    python settings_edit.py backup  <settings file> <backup file>
    python settings_edit.py stage   <settings file> <format> <key=value> [<key=value> ...]
    python settings_edit.py restore <settings file> <backup file>

Formats:
  ini     lines key=value (the first line of each key changes; the rest of the file is kept byte for byte), e.g.
          Resident Evil Requiem's config.ini
  cp2077  Cyberpunk 2077's UserSettings.json: data[].options[] with name / value; values true / false / numbers are
          written as JSON values, everything else as strings
Every named key must exist; nothing is added."""
import hashlib
import json
import os
import re
import shutil
import sys

sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def parse_pairs(pairs):
    out = {}
    for p in pairs:
        if '=' not in p:
            raise SystemExit('use key=value: %s' % p)
        k, v = p.split('=', 1)
        out[k] = v
    return out


def stage_ini(path, want):
    raw = open(path, 'rb').read()
    nl = b'\r\n' if b'\r\n' in raw else b'\n'
    lines = raw.split(nl)
    done = {}
    for i, l in enumerate(lines):
        m = re.match(rb'^([A-Za-z0-9_.]+)=(.*)$', l)
        if m and m.group(1).decode() in want and m.group(1).decode() not in done:
            k = m.group(1).decode()
            lines[i] = (k + '=' + want[k]).encode()
            done[k] = (m.group(2).decode(), want[k])
    missing = sorted(set(want) - set(done))
    if missing:
        raise SystemExit('keys not found: %s' % missing)
    open(path, 'wb').write(nl.join(lines))
    return done


def json_value(v):
    if v in ('true', 'false'):
        return v == 'true'
    try:
        return int(v)
    except ValueError:
        return v


def stage_cp2077(path, want):
    d = json.load(open(path, encoding='utf-8'))
    done = {}
    for g in d.get('data', []):
        for o in g.get('options', []):
            if o.get('name') in want:
                new = json_value(want[o['name']])
                done[o['name']] = (o.get('value'), new)
                o['value'] = new
    missing = sorted(set(want) - set(done))
    if missing:
        raise SystemExit('keys not found: %s' % missing)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(d, f, indent=4)
    return done


def main(argv):
    cmd = argv[1]
    if cmd == 'backup':
        f, b = argv[2], argv[3]
        if os.path.exists(b):
            raise SystemExit('backup exists already: %s' % b)
        shutil.copy2(f, b)
        print('backup %s sha256 %s' % (b, sha(b)))
    elif cmd == 'stage':
        f, fmt, want = argv[2], argv[3], parse_pairs(argv[4:])
        done = {'ini': stage_ini, 'cp2077': stage_cp2077}[fmt](f, want)
        print('staged: %s' % done)
    elif cmd == 'restore':
        f, b = argv[2], argv[3]
        shutil.copy2(b, f)
        print('restored; sha256 %s (backup %s) %s' % (sha(f), sha(b), 'ok' if sha(f) == sha(b) else 'MISMATCH'))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv)
