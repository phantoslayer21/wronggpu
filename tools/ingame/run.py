# SPDX-License-Identifier: MIT
"""One in-game run of WrongGPU with a record and a byte-for-byte restore: a game folder's manifest, backups, foreign
files moved aside or hashed, the candidate installed through the launcher's command line, a model choice (e.g. Model L,
issue #89), screenshots, the runtime's own proof lines and timing, then everything put back and verified.

    python run.py --config <game>.json preflight              checks only (always safe)
    python run.py --config <game>.json prepare  [--dry-run]   manifest, backups, aside, update, model choice, settings
    python run.py --config <game>.json use-m | use-l          the model choice (the game closed): runtime_options and
                                                              OptiScaler's [WrongGPU] UpscalerIndex
    python run.py --config <game>.json launch   [--dry-run]   GPU flag (released by game_hold.py when the game exits),
                                                              VRAM log, the launcher's `play`
    python run.py --config <game>.json hold                   reports the game's pid once game_hold.py holds it
    python run.py --config <game>.json shot <label>           screenshot of the desktop into <out>\\shots
    python run.py --config <game>.json collect [--logs <dir>] our logs (the game closed), proof lines, verdict
    python run.py --config <game>.json restore  [--dry-run]   everything back, the placed model removed, verified
    python run.py --config <game>.json all --dry-run          every phase as a dry run

The config (examples/*.json, see README.md) names every path: the game, the exe folder, the output folder, the
candidate launcher and its payload, the model file and its SHA-256, the files moved aside or only hashed, the settings
file and its keys, the GPU flag. Nothing here is specific to one PC.

Game-folder rules: only our files and the files the config names are touched; other mods' files are never opened
(hashed at most); vendor files (nvngx*, *dlss*, sl.*, nvapi*) are recorded by name only; OptiScaler.log is never read.
The model goes into the game's data folder (<exe dir>\\WrongGPU\\model\\<model>\\model.wgcm), which the front end looks
at first. Not %LOCALAPPDATA%: inside a packaged (MSIX) app, new files there go to the package's private copy, which the
game never sees. It stays there until the game has exited; restore removes it and the folders it created."""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NOWIN = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_config(path):
    """The game's config (JSON). %VARIABLES% are expanded; exe_dir is relative to game_root, other relative paths are
    relative to the config file. A value still in <angle brackets> is an unfilled placeholder (an error)."""
    with open(path, encoding='utf-8') as f:
        c = json.load(f)
    base = os.path.dirname(os.path.abspath(path))

    def ab(p, rel_to=base):
        if not p:
            return p
        if '<' in p and '>' in p:
            raise SystemExit('%s: fill in the placeholder %r' % (path, p))
        p = os.path.expandvars(p)
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(rel_to, p))
    for k in ('game_root', 'out', 'launcher_package', 'payload', 'aside_parent', 'gpu_flag'):
        if k in c:
            c[k] = ab(c[k])
    c['exe_dir'] = ab(c.get('exe_dir') or '.', c['game_root'])
    c['model']['file'] = ab(c['model']['file'])
    if c.get('settings'):
        c['settings']['file'] = ab(c['settings']['file'])
    return c


def _config_path(argv):
    if '--config' not in argv or argv.index('--config') + 1 >= len(argv):
        raise SystemExit('--config <game>.json is required (see tools/ingame/examples)')
    return argv[argv.index('--config') + 1]


CFG = load_config(_config_path(sys.argv)) if __name__ == '__main__' else None
if CFG:
    G = CFG['game_root']
    X = CFG['exe_dir']
    EXE = os.path.join(X, CFG['exe'])
    A = CFG['out']
    PKG_EXE = CFG['launcher_package']
    PAYLOAD = CFG['payload']
    MODEL_NAME = CFG['model'].get('name', 'L')
    MODEL_SRC = CFG['model']['file']
    MODEL_SHA = CFG['model']['sha256'].lower()
    MODEL_DST = os.path.join(X, 'WrongGPU', 'model', MODEL_NAME, 'model.wgcm')
    FLAG = CFG['gpu_flag']
    SANDBOX = os.path.dirname(FLAG)
    ASIDE_FILES = [os.path.join(G, p) for p in CFG.get('aside', [])]
    ASIDE_PARENT = CFG.get('aside_parent') or os.path.join(os.path.dirname(G), '_wronggpu_aside')
    FOREIGN_HASH = [os.path.join(G, p) for p in CFG.get('hash_only', [])]
    OUR_LOGS = {  # our own logs only (OptiScaler.log is OptiScaler's: never copied or read)
        'wronggpu-fsr.log': os.path.join(X, 'OptiScaler', 'custom', 'WrongGPU', 'ffx', 'wronggpu-fsr.log'),
        'wronggpu-diagnostics.jsonl': os.path.join(X, 'OptiScaler', 'custom', 'WrongGPU', 'ffx',
                                                   'wronggpu-diagnostics-amd_fidelityfx_dx12.jsonl'),
        'wronggpu-rr.log': os.path.join(X, 'wronggpu-rr.log'),
    }
    # the installed payload against the build's payload (installed name <- payload name)
    PAYLOAD_MAP = {'wronggpu_runtime.dll': 'wronggpu_runtime.dll',
                   r'OptiScaler\custom\WrongGPU\ffx\amd_fidelityfx_upscaler_dx12.dll': 'amd_fidelityfx_dx12.dll'}
    OWNER = CFG.get('flag_owner', 'ingame-run') + ' (%s, %s)' % (CFG.get('name', CFG['exe']), MODEL_NAME)
    GAME_EXE = CFG['exe'].lower()
    PROC = os.path.splitext(CFG['exe'])[0]
    LAUNCHERS = tuple(x.lower() for x in CFG.get('launchers', []))
    DISK_FREE_GB = CFG.get('disk_free_gb', {})
    S = CFG.get('settings') or {}
    os.makedirs(A, exist_ok=True)
    # the runtime's timing lines and the launch trace (one line per D3D12 kernel launch, capped; the runtime sets
    # WG_LAUNCH_TRACE from the option, since a store launcher starts the game without our environment)
    TRACE = os.path.join(A, 'launch-trace.txt')
    OPTS_EXTRA = 'timing_log=1;timing_stages=1;launch_trace=%s;launch_trace_max=%d' % (
        TRACE, int(CFG.get('launch_trace_max', 200000)))


def settings_cmd(verb):
    """settings_edit.py backup / stage / restore of the config's settings file (None: no settings file)."""
    if not S:
        return None
    tool = os.path.join(HERE, 'settings_edit.py')
    backup = os.path.join(A, os.path.basename(S['file']) + '.before')
    if verb == 'stage':
        return [tool, 'stage', S['file'], S['format']] + ['%s=%s' % kv for kv in S['keys'].items()]
    return [tool, verb, S['file'], backup]


def clean_env():
    e = {k: v for k, v in os.environ.items()
         if not (any(s in k.upper() for s in ('HIP', 'LLVM', 'ROCM', 'HSA')) and k.upper() not in ('HIP_PLATFORM', 'HIP_PATH_72'))
         and not k.upper().startswith('WG_')}
    if CFG.get('rocm_path'):
        e['ROCM_PATH'] = CFG['rocm_path']
    return e


class Run:
    def __init__(self, dry):
        self.dry = dry
        self.a = os.path.join(A, 'dry-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')) if dry else A
        os.makedirs(self.a, exist_ok=True)
        self.state_path = os.path.join(A, 'state.json')
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.problems = []

    def log(self, s):
        line = '%s %s%s' % (datetime.datetime.now().strftime('%H:%M:%S'), 'DRY ' if self.dry else '', s)
        print(line, flush=True)
        with open(os.path.join(self.a, 'session.log'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    def need(self, ok, what):
        self.log(('ok    ' if ok else 'FAIL  ') + what)
        if not ok:
            self.problems.append(what)
        return ok

    def save(self, **kv):
        if self.dry:
            return
        self.state.update(kv)
        json.dump(self.state, open(self.state_path, 'w'), indent=1)

    def py(self, name, args, out=None, check=True):
        """Run a Python tool (no window, below normal); output to <a>\\<name>.txt. In a dry run: printed only."""
        cmd = [sys.executable] + args
        if self.dry:
            self.log('would run: ' + ' '.join('"%s"' % c if ' ' in c else c for c in cmd))
            return 0
        path = out or os.path.join(self.a, name + '.txt')
        with open(path, 'w', encoding='utf-8') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, creationflags=NOWIN, env=clean_env())
        self.log('%s exit %d -> %s' % (name, r.returncode, path))
        if check and r.returncode != 0:
            raise SystemExit('%s failed (exit %d), see %s' % (name, r.returncode, path))
        return r.returncode

    def wg(self, name, verb_args):
        """The candidate launcher's command line (--out / --log into <a>)."""
        cmd = [PKG_EXE] + verb_args + ['--out=' + os.path.join(self.a, name + '.json'),
                                       '--log=' + os.path.join(self.a, name + '.log')]
        if self.dry:
            self.log('would run: ' + ' '.join('"%s"' % c if ' ' in c else c for c in cmd))
            return 0
        r = subprocess.run(cmd, creationflags=NOWIN, env=clean_env())
        self.log('launcher %s exit %d' % (' '.join(verb_args[:1]), r.returncode))
        return r.returncode


def tasks():
    out = subprocess.run(['tasklist', '/fo', 'csv', '/nh'], capture_output=True, text=True).stdout
    return {l.split('","')[0].strip('"').lower(): int(l.split('","')[1]) for l in out.splitlines() if l.count('","') >= 2}


def memory_used():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [('l', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                                                                       ('a', 'b', 'c', 'd', 'e', 'f', 'g')]
    m = MS()
    m.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.load


def ini_runtime_options(text):
    m = re.search(r'^runtime_options\s*=\s*(.*)$', text, re.M)
    return m.group(1).strip() if m else None


TOOLS = ('game_manifest.py', 'check_record.py', 'aside.py', 'vram_log.py', 'game_backup.py', 'game_restore.py',
         'game_hold.py', 'set_upscaler_index.py', 'settings_edit.py')
tool = lambda n: os.path.join(HERE, n)


# ---- phases -----------------------------------------------------------------------------------------------------------

def preflight(r):
    t = tasks()
    for p in (GAME_EXE,) + LAUNCHERS:
        r.need(p not in t, 'not running: ' + p)
    r.need(not os.path.exists(FLAG) or OWNER.split(' ')[0] in open(FLAG, encoding='utf-8', errors='replace').read(),
           'GPU flag free (or ours): ' + FLAG)
    req = glob.glob(os.path.join(SANDBOX, 'GPU-REQUEST*')) + glob.glob(os.path.join(SANDBOX, 'GPU-SLOTS-PAUSED*'))
    r.need(not req, 'no GPU request / pause file: %s' % req)
    mu = memory_used()
    r.need(mu < 80, 'memory used %d %% (< 80)' % mu)
    for d, gb in sorted(DISK_FREE_GB.items()):
        free = shutil.disk_usage(d).free / 2**30
        r.need(free > gb, '%s free %.1f GB (> %s)' % (d, free, gb))
    r.need(os.path.isfile(EXE), 'game: ' + EXE)
    r.need(os.path.isfile(PKG_EXE), 'candidate launcher: ' + PKG_EXE)
    for inst, pay in PAYLOAD_MAP.items():
        r.need(os.path.isfile(os.path.join(PAYLOAD, pay)), 'payload file: ' + pay)
    r.need(os.path.isfile(MODEL_SRC) and sha(MODEL_SRC) == MODEL_SHA,
           'model %s file %s sha256 %s' % (MODEL_NAME, MODEL_SRC, MODEL_SHA))
    r.need(not os.path.exists(MODEL_DST) or sha(MODEL_DST) == MODEL_SHA or r.state.get('model_placed'),
           'no other file at ' + MODEL_DST)
    for f in ASIDE_FILES:
        r.need(os.path.isfile(f) or r.state.get('aside'), 'aside candidate present: ' + f)
    ini = os.path.join(X, 'wronggpu.ini')
    r.need(os.path.isfile(ini) and ini_runtime_options(open(ini, encoding='utf-8', errors='replace').read()) is not None,
           'runtime_options line in ' + ini)
    for n in TOOLS:
        r.need(os.path.isfile(tool(n)), 'tool ' + n)
    if S:
        r.need(os.path.isfile(S['file']), 'settings file ' + S['file'])
        r.need(S.get('format') in ('ini', 'cp2077'), 'settings format %r (ini or cp2077)' % S.get('format'))
    # foreign-mod scan and the install record (read-only; vendor files by name only)
    for name, args in (('scan', [tool('game_manifest.py'), 'scan', G]),
                       ('record-check', [tool('check_record.py'), X])):
        out = subprocess.run([sys.executable] + args, capture_output=True, text=True, creationflags=NOWIN).stdout
        open(os.path.join(r.a, 'preflight-%s.txt' % name), 'w', encoding='utf-8').write(out)
        r.log('%s: %s' % (name, (out.strip().splitlines() or ['(no output)'])[-1][:200]))
    return not r.problems


def prepare(r):
    if not preflight(r):
        raise SystemExit('preflight: %s' % r.problems)
    if r.state.get('prepared'):
        raise SystemExit('already prepared (state.json); run restore first')
    # 0. (the GPU flag is taken per game run by launch and released by game_hold.py when the game exits)
    # 1. manifest, record, backups (copies; nothing changed)
    r.py('manifest-before', [tool('game_manifest.py'), 'make', G, os.path.join(r.a, 'manifest-before.json')])
    r.py('record-before', [tool('check_record.py'), X])
    r.py('backup', [tool('game_backup.py'), X, os.path.join(r.a, 'backup')])
    if S:
        r.py('settings-backup', settings_cmd('backup'))
    foreign_hashes(r, 'before')
    r.save(backup=True)
    # 2. the config's foreign files aside (hash + stat + rename; contents never opened)
    aside_dir = os.path.join(ASIDE_PARENT, 'aside-' + datetime.datetime.now().strftime('%Y%m%d-%H%M'))
    if ASIDE_FILES:
        r.py('aside-out', [tool('aside.py'), 'out', aside_dir] + ASIDE_FILES)
        r.save(aside=aside_dir)
    else:
        r.log('no files to move aside for this game')
    # 3. the candidate through the launcher's update verb
    rc = r.wg('update', ['update', EXE])
    if rc != 0:
        raise SystemExit('update failed (exit %d): restore next' % rc)
    r.save(updated=True)
    for inst, pay in PAYLOAD_MAP.items():
        if r.dry:
            r.log('would compare %s with payload %s' % (inst, pay))
        else:
            r.need(sha(os.path.join(X, inst)) == sha(os.path.join(PAYLOAD, pay)), 'installed %s = payload %s' % (inst, pay))
    # 4. the model choice: runtime_options model=<model> (executor kept) and OptiScaler's UpscalerIndex
    if not r.dry and not os.path.isfile(os.path.join(X, 'wronggpu.ini')):
        raise SystemExit('no wronggpu.ini in %s after the update: restore next' % X)
    set_options(r, MODEL_NAME, OPTS_EXTRA)
    # 5. the model file where the front end and the runtime look first (the game's WrongGPU\model\<model>)
    place_model(r)
    # 6. the game's graphics settings named in the config
    if S:
        r.py('settings-stage', settings_cmd('stage'))
    r.save(prepared=True)
    r.log("prepared: next launch, then start the game in its launcher, then hold")


def set_options(r, model='L', extra=''):
    """runtime_options = model=<model>;<the other items, timing items replaced by extra> in the exe folder's
    wronggpu.ini, and OptiScaler.ini [WrongGPU] UpscalerIndex (0 = WrongGPU Model M, 1 = WrongGPU Model L: the host's
    version choice, which the front end appends after runtime_options)."""
    ini = os.path.join(X, 'wronggpu.ini')
    raw = open(ini, 'rb').read()
    nl = b'\r\n' if b'\r\n' in raw else b'\n'
    lines = raw.split(nl)
    idx = [i for i, l in enumerate(lines) if re.match(rb'^runtime_options\s*=', l)]
    if len(idx) != 1:
        raise SystemExit('wronggpu.ini: %d runtime_options lines' % len(idx))
    old = lines[idx[0]].split(b'=', 1)[1].decode()
    items = [i for i in re.split(r'[;,]', old) if i.strip() and not re.match(r'\s*(model|timing_log|timing_stages)=', i)]
    new = ';'.join(['model=' + model] + items + [e for e in extra.split(';') if e])
    r.log('wronggpu.ini runtime_options: %r -> %r' % (old, new))
    if not r.dry:
        lines[idx[0]] = b'runtime_options=' + new.encode()
        open(ini, 'wb').write(nl.join(lines))
        r.save(ini_model=model)
    r.py('upscaler-index', [tool('set_upscaler_index.py'), os.path.join(X, 'OptiScaler.ini'),
                            '1' if model == 'L' else '0'])


def reinstall(r):
    """Within a prepared session: the launcher's update with the (rebuilt) candidate, the payload check, then the
    model choice again (the update may rewrite both ini files)."""
    if not r.dry and not r.state.get('prepared'):
        raise SystemExit('not prepared')
    if GAME_EXE in tasks():
        raise SystemExit('the game is running')
    rc = r.wg('update2', ['update', EXE])
    if rc != 0:
        raise SystemExit('update failed (exit %d)' % rc)
    for inst, pay in PAYLOAD_MAP.items():
        if not r.dry:
            r.need(sha(os.path.join(X, inst)) == sha(os.path.join(PAYLOAD, pay)), 'installed %s = payload %s' % (inst, pay))
    set_options(r, MODEL_NAME, OPTS_EXTRA)


def foreign_hashes(r, when):
    """SHA-256 + size + mtime of the foreign files (the config's hash_only: e.g. a script loader the game needs), before and after;
    'after' must equal 'before'."""
    if not FOREIGN_HASH:
        return
    res = {p: {'sha256': sha(p), 'size': os.path.getsize(p), 'mtime': os.path.getmtime(p)} if os.path.isfile(p) else None
           for p in FOREIGN_HASH}
    path = os.path.join(r.a, 'foreign-%s.json' % when)
    if r.dry:
        r.log('foreign files (%s): %s' % (when, {os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p):
                                                 (v or {}).get('sha256', 'absent')[:16] for p, v in res.items()}))
        return
    json.dump(res, open(path, 'w'), indent=1)
    if when == 'after':
        before = json.load(open(os.path.join(r.a, 'foreign-before.json')))
        r.need(before == res, 'foreign files unchanged (%d)' % len(res))


def place_model(r):
    """Copy the model file to MODEL_DST; the folders it creates are recorded (restore removes them, deepest
    first, only when empty)."""
    created, d = [], os.path.dirname(MODEL_DST)
    while not os.path.isdir(d):
        created.append(d)
        d = os.path.dirname(d)
    if r.dry:
        r.log('would copy %s -> %s (folders created: %s)' % (MODEL_SRC, MODEL_DST, created))
        return
    os.makedirs(os.path.dirname(MODEL_DST), exist_ok=True)
    shutil.copy2(MODEL_SRC, MODEL_DST)
    r.need(sha(MODEL_DST) == MODEL_SHA, 'placed %s sha256 %s' % (MODEL_DST, MODEL_SHA))
    r.save(model_placed=MODEL_DST, model_dirs_created=created)


def launch(r):
    if not r.dry and not r.state.get('prepared'):
        raise SystemExit('not prepared')
    vcsv = os.path.join(r.a, 'vram.csv')
    vcmd = [sys.executable, tool('vram_log.py'), PROC, vcsv, '--interval', '5', '--max-min', '40']
    if r.dry:
        r.log('would start (background): ' + ' '.join(vcmd))
    else:
        subprocess.Popen(vcmd, stdout=open(os.path.join(r.a, 'vram.txt'), 'w'), stderr=subprocess.STDOUT,
                         creationflags=NOWIN)
        r.log('VRAM log started -> ' + vcsv)
    # the GPU flag for this one game run; game_hold.py releases it in `finally` when the game exits (or does not start)
    rec = os.path.join(r.a, 'hold-%s.json' % datetime.datetime.now().strftime('%H%M%S'))
    hcmd = [sys.executable, tool('game_hold.py'), FLAG, OWNER.split(' ')[0], rec, '--exe', GAME_EXE]
    if r.dry:
        r.log('would take %s as %r, start (background) %s, then play' % (FLAG, OWNER, ' '.join(hcmd)))
        r.wg('play', ['play', EXE])
        return
    fd = os.open(FLAG, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # fails if anyone holds the GPU
    try:
        os.write(fd, ('%s %s\n' % (OWNER, datetime.datetime.now().isoformat(timespec='seconds'))).encode())
    finally:
        os.close(fd)
    try:
        subprocess.Popen(hcmd, stdout=open(rec[:-5] + '.txt', 'w'), stderr=subprocess.STDOUT, creationflags=NOWIN)
        r.log('flag taken; game_hold.py releases it when the game exits (%s)' % rec)
        rc = r.wg('play', ['play', EXE])
        if rc != 0:
            raise SystemExit('play failed (exit %d)' % rc)
    except BaseException:
        release_flag(r)  # game_hold.py would also release it after its start timeout
        raise
    r.save(hold_record=rec)
    r.log("next: start the game in its launcher; then: run.py hold (reports the game pid)")


def release_flag(r):
    try:
        if os.path.exists(FLAG) and open(FLAG, encoding='utf-8', errors='replace').read().startswith(OWNER.split(' ')[0]):
            os.remove(FLAG)
            r.log('flag released')
    except OSError as e:
        r.log('flag release failed: %s' % e)


def hold(r):
    """Report the game pid once game_hold.py holds it (the holder itself releases the flag at the game's exit)."""
    if r.dry:
        r.log('would wait for game_hold.py to report the game pid')
        return
    rec = r.state.get('hold_record')
    t0 = time.time()
    st = {}
    while time.time() - t0 < 320:
        st = json.load(open(rec)) if rec and os.path.exists(rec) else {}
        if st.get('status') != 'waiting' and st:
            break
        time.sleep(2)
    r.log('hold: %s' % st)
    if st.get('game_pid'):
        r.save(game_pid=st['game_pid'])


def shot(r, label):
    d = os.path.join(r.a, 'shots')
    os.makedirs(d, exist_ok=True)
    n = len(glob.glob(os.path.join(d, '*.png'))) + 1
    path = os.path.join(d, '%02d-%s.png' % (n, re.sub(r'[^A-Za-z0-9_-]', '_', label)))
    from PIL import ImageGrab
    img = ImageGrab.grab(all_screens=True)
    if r.dry:
        r.log('screenshot works: %dx%d (dry run: saved to %s)' % (img.size[0], img.size[1], path))
    img.save(path)
    r.log('screenshot %s (%dx%d)' % (path, img.size[0], img.size[1]))


def proof_patterns():
    """(key, regex over our logs); the run passes only with the required ones (required_keys)."""
    sub = re.escape(MODEL_NAME)
    return [
    ('model_option', r'model=' + sub),
    ('model_loaded', r'(model|loaded)[^\n]*[\\/]' + sub + r'[\\/]model\.wgcm'),
    ('model_digest', MODEL_SHA[:16]),
    ('l_kernels', r'l_(swin_block|ops|post|coef|linear|project|enc_values|dec_project|post_recon)\w*'),
    ('executor_dx12', r'executor dx12'),
    ('executor_hip', r'executor hip|the HIP executor (runs|takes)'),
    ('refused', r'refus|not part of this build|Model L runs on the HIP backend'),
    ('dispatches', r'destroyed after (\d+) dispatches'),
    ('failed', r'(\d+) failed|dispatch failed|evaluate failed'),
    ('network_ready', r'network runtime ready[^\n]*'),
    ]


def required_keys():
    if MODEL_NAME == 'L':  # Model L must load its own file, launch its own kernels and run on D3D12
        return ('model_option', 'model_loaded', 'l_kernels', 'executor_dx12', 'network_ready')
    return ('model_option', 'executor_dx12', 'network_ready')


def proof(text):
    res = {}
    for key, rx in proof_patterns():
        hits = [m.group(0) for m in re.finditer(rx, text)]
        res[key] = {'count': len(hits), 'first': hits[0][:300] if hits else '', 'lines': []}
        if hits:
            res[key]['lines'] = [l[:400] for l in text.splitlines() if re.search(rx, l)][:5]
    disp = [int(m) for m in re.findall(r'destroyed after (\d+) dispatches', text)]
    res['dispatch_total'] = sum(disp)
    verdict = []
    for k in required_keys():
        if not res[k]['count']:
            verdict.append('missing: ' + k)
    if res['executor_hip']['count']:
        verdict.append('HIP executor named (the run must be on D3D12)')
    if res['refused']['count']:
        verdict.append('a refusal line')
    if not disp or not sum(disp):
        verdict.append('no dispatches')
    res['verdict'] = 'PASS' if not verdict else 'NOT PROVEN: ' + '; '.join(verdict)
    return res


def collect(r, logs=None):
    d = logs or os.path.join(r.a, 'logs')
    if not logs:
        if r.dry:
            for name, src in OUR_LOGS.items():
                r.log('would copy (PowerShell Copy-Item) %s -> %s' % (src, os.path.join(d, name)))
            return
        if GAME_EXE in tasks():
            raise SystemExit('the game is still running: close it first (CloseMainWindow)')
        os.makedirs(d, exist_ok=True)
        for name, src in OUR_LOGS.items():
            ps = "Copy-Item -LiteralPath '%s' -Destination '%s' -ErrorAction Stop" % (src, os.path.join(d, name))
            rc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', ps], creationflags=NOWIN).returncode
            r.log('copied %s: exit %d' % (name, rc))
    text = ''
    for n in sorted(os.listdir(d)):
        if n.endswith('.log'):
            text += open(os.path.join(d, n), encoding='utf-8', errors='replace').read() + '\n'
    res = proof(text)
    vcsv = os.path.join(r.a if not logs else d, 'vram.csv')
    if os.path.isfile(vcsv):
        import csv
        rows = [x for x in csv.DictReader(open(vcsv)) if x.get('game_dedicated_MiB') not in ('', '0.0', None)]
        v = [float(x['game_dedicated_MiB']) for x in rows]
        if v:
            res['vram'] = {'samples': len(v), 'from': rows[0]['time'], 'to': rows[-1]['time'], 'first': v[0], 'last': v[-1],
                           'min': min(v), 'max': max(v)}
    res['model_file'] = {'path': MODEL_DST, 'sha256': sha(MODEL_DST) if os.path.isfile(MODEL_DST) else None,
                           'expected': MODEL_SHA}
    res['shots'] = sorted(glob.glob(os.path.join(r.a, 'shots', '*.png')))
    tr = os.path.join(d if logs else r.a, 'launch-trace.txt')
    if os.path.isfile(tr):
        counts = {}
        for l in open(tr, encoding='utf-8', errors='replace'):
            f = l.split()
            if f:
                counts[f[0]] = counts.get(f[0], 0) + 1
        lk = {k: v for k, v in counts.items() if k.startswith(('l_', 'lev/')) or k == 'swin_block_c64_add_ex_nc'}
        res['launch_trace'] = {'file': tr, 'lines': sum(counts.values()), 'kernels': len(counts), 'model_l_launches': lk,
                               'all': dict(sorted(counts.items()))}
        r.log('launch trace: %d lines, %d kernels; Model L kernels: %s' % (sum(counts.values()), len(counts), lk))
    if not logs:
        release_flag(r)  # the game has exited (checked above); normally game_hold.py has released it already
    out = os.path.join(r.a, 'proof.json')
    json.dump(res, open(out, 'w'), indent=1)
    for k, _ in proof_patterns():
        r.log('%-15s %4d  %s' % (k, res[k]['count'], res[k]['first'][:160]))
    r.log('dispatches %d; VRAM %s' % (res['dispatch_total'], res.get('vram')))
    r.log('verdict: %s (%s)' % (res['verdict'], out))


def restore(r):
    st = r.state
    if r.dry:
        r.log('would require the game closed, then (in this order, each only if prepare did it):')
    elif GAME_EXE in tasks():
        raise SystemExit('the game is still running: close it first')
    if st.get('backup') or r.dry:
        r.py('verify-after-run', [tool('game_manifest.py'), 'verify', G, os.path.join(A, 'manifest-before.json')],
             check=False)
        r.py('game-restore', [tool('game_restore.py'), G, X, A, os.path.join(A, 'verify-after-run.txt')])
    if st.get('model_placed') or r.dry:
        if r.dry:
            r.log('would remove %s (and the folders place_model created, when empty)' % MODEL_DST)
        else:
            placed = st['model_placed'] if isinstance(st['model_placed'], str) else MODEL_DST
            if os.path.isfile(placed):
                os.remove(placed)
            for d in st.get('model_dirs_created', []):  # deepest first
                if os.path.isdir(d) and not os.listdir(d):
                    os.rmdir(d)
            r.log('removed the placed model (%s) and created folders %s' % (placed, st.get('model_dirs_created', [])))
            r.save(model_placed=False, model_dirs_created=[])
    if st.get('aside') or r.dry:
        a = st.get('aside', os.path.join(ASIDE_PARENT, 'aside-<time>'))
        r.py('aside-restore', [tool('aside.py'), 'restore', a])
        r.py('aside-verify', [tool('aside.py'), 'verify', a])
        r.save(aside=None, aside_restored=a)
    if st.get('backup') or r.dry:
        if S:
            r.py('settings-restore', settings_cmd('restore'))
        foreign_hashes(r, 'after')
        r.py('verify-final', [tool('game_manifest.py'), 'verify', G, os.path.join(A, 'manifest-before.json')],
             check=False)
        r.py('record-after', [tool('check_record.py'), X])
    if r.dry:
        r.log('would release ' + FLAG + ' if it is still ours')
    else:
        release_flag(r)
    r.save(prepared=False, flag=False, backup=False, updated=False, ini_model=None)
    r.log("restored; check verify-final.txt (expected: only the game's and OptiScaler's own logs changed) and "
          'record-after.txt')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='the game config (JSON; examples/*.json)')
    ap.add_argument('phase', choices=['preflight', 'prepare', 'place-model', 'reinstall', 'use-m', 'use-l', 'launch',
                                      'hold', 'shot', 'collect', 'restore', 'all'])
    ap.add_argument('label', nargs='?', default='game')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--logs', help='collect: logs already copied (proof extraction only)')
    a = ap.parse_args()
    if a.phase == 'all' and not a.dry_run:
        raise SystemExit('all is a dry run only (use --dry-run)')
    r = Run(a.dry_run or a.phase == 'preflight')
    r.log('phase %s (%s)' % (a.phase, r.a))
    if a.phase == 'preflight':
        ok = preflight(r)
        r.log('preflight %s' % ('PASS' if ok else 'FAIL: %s' % r.problems))
        return 0 if ok else 1
    if a.phase == 'all':
        for f in (prepare, launch, hold):
            try:
                f(r)
            except SystemExit as e:
                r.log('stopped: %s' % e)
                r.problems.append(str(e))
        shot(r, 'dry-run')
        collect(r)
        restore(r)
        r.log('dry run %s' % ('PASS' if not r.problems else 'FAIL: %s' % r.problems))
        return 0 if not r.problems else 1
    if a.phase in ('use-m', 'use-l'):  # the comparison run in the same session and settings (the game must be closed)
        if GAME_EXE in tasks():
            raise SystemExit('the game is running')
        set_options(r, 'M' if a.phase == 'use-m' else 'L', OPTS_EXTRA)
    elif a.phase == 'shot':
        shot(r, a.label)
    elif a.phase == 'collect':
        collect(r, a.logs)
    else:
        {'prepare': prepare, 'place-model': place_model, 'reinstall': reinstall, 'launch': launch, 'hold': hold,
         'restore': restore}[a.phase](r)
    return 1 if r.problems else 0


if __name__ == '__main__':
    sys.exit(main())
