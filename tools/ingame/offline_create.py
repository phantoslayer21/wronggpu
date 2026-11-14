# SPDX-License-Identifier: MIT
"""Offline create + run test of Model L on the D3D12 executor before any game run (issue #89), and the synthetic
timing harness (test_interop_d3d12.exe hipbench) for Model L and Model M.

  python offline_create.py build  --work <dir>
        compile the harness from this checkout (MSVC, CPU only)
  python offline_create.py create --work <dir> --runtime-dir <dir> --model <L model.wgcm> --model-sha <sha256> [-- <wrap>]
        one short run: model=L, executor=dx12, the converted Model L file copied into <work>\\data\\L\\model.wgcm,
        600 frames 1280x720 -> 1920x1080; launch trace; PASS / FAIL in <work>\\create-L.json
  python offline_create.py timing --work <dir> --runtime-dir <dir> --model ... --model-sha ... [--m-model-dir <dir>]
                                  [--rounds 3] [--shapes "1080p Quality,..."] [-- <wrap>]
        Model L and Model M interleaved per shape, one run each; record <work>\\l-timing-harness.json

--runtime-dir is a built runtime folder (wronggpu_runtime.dll and the kernels, build.py --model-l). Everything after
a lone "--" is a command prefix that wraps each GPU run (e.g. a GPU-slot tool: `-- python gpu_slot.py --cap-s 170 --`);
without it the harness runs directly. --extra adds runtime options to Model L's runs (e.g. l_post=own), --tmp sets
TMP / TEMP for the runs, --rocm-path ROCM_PATH.

The model data folder is under --work, not %LOCALAPPDATA%: inside a packaged (MSIX) app new files there are private
to the package. Model M uses --m-model-dir (default %LOCALAPPDATA%\\WrongGPU\\model, read only)."""
import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SHAPES = [('1080p Quality', 1280, 720, 1920, 1080), ('1080p Ultra Quality', 1477, 831, 1920, 1080),
          ('1080p DLAA', 1920, 1080, 1920, 1080), ('1440p Quality', 1707, 960, 2560, 1440),
          ('1440p Ultra Quality', 1969, 1108, 2560, 1440), ('1440p DLAA', 2560, 1440, 2560, 1440)]
NOWIN = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
L_KERNELS = ('l_', 'lev/', 'swin_block_c64_add_ex_nc')
VSWHERE = os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'), 'Microsoft Visual Studio',
                       'Installer', 'vswhere.exe')
A = None      # the parsed arguments
WRAP = []     # the command prefix after "--"


def harness():
    return os.path.join(A.work, 'harness', 'test_interop_d3d12.exe')


def env_base():
    e = {k: v for k, v in os.environ.items()
         if not (any(s in k.upper() for s in ('HIP', 'LLVM', 'ROCM', 'HSA')) and k.upper() not in ('HIP_PLATFORM', 'HIP_PATH_72'))
         and not k.upper().startswith('WG_')}
    e.update(AMD_COMGR_CACHE='0', WG_TEST_DEFAULT_EXECUTOR='1')
    if A.rocm_path:
        e['ROCM_PATH'] = A.rocm_path
    if A.tmp:
        os.makedirs(A.tmp, exist_ok=True)
        e.update(TMP=A.tmp, TEMP=A.tmp)
    return e


def build():
    exe = harness()
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    vs = subprocess.check_output([A.vswhere, '-latest', '-find', r'Common7\Tools\VsDevCmd.bat'], text=True).splitlines()[0]
    src = os.path.join(A.src, 'clean', 'runtime', 'interop', 'tests', 'test_interop_d3d12.cpp')
    bat = os.path.join(A.work, 'harness', 'build.bat')
    open(bat, 'w').write('@call "%s" -arch=x64 -host_arch=x64 >nul\r\ncd /d "%s"\r\n'
                         'cl /nologo /std:c++17 /O2 /EHsc /D_CRT_SECURE_NO_WARNINGS "%s" /Fe:test_interop_d3d12.exe '
                         'd3d12.lib dxgi.lib\r\n' % (vs, os.path.dirname(exe), src))
    r = subprocess.run(['cmd.exe', '/d', '/c', bat], capture_output=True, text=True, creationflags=NOWIN)
    print(r.stdout[-3000:], r.stderr[-2000:])
    print('harness', 'built' if os.path.isfile(exe) else 'NOT built', exe)
    return 0 if os.path.isfile(exe) else 1


def need_run_args():
    miss = [n for n in ('runtime_dir', 'model', 'model_sha') if not getattr(A, n)]
    if miss:
        raise SystemExit('missing: %s' % ', '.join('--' + m.replace('_', '-') for m in miss))
    if not os.path.isfile(harness()):
        raise SystemExit('no harness at %s (run build first)' % harness())


def l_data():
    d = os.path.join(A.work, 'data')
    os.makedirs(os.path.join(d, 'L'), exist_ok=True)
    dst = os.path.join(d, 'L', 'model.wgcm')
    if not os.path.isfile(dst):
        shutil.copy2(A.model, dst)
    got = hashlib.sha256(open(dst, 'rb').read()).hexdigest()
    if got != A.model_sha.lower():
        raise SystemExit('L model digest %s != %s' % (got, A.model_sha))
    return d


def options(model, data, stages):
    o = 'kernels_dir=%s;timing_log=1;timing_stages=%d;executor=dx12' % (A.runtime_dir, 1 if stages else 0)
    if model == 'L':
        return 'model=L;model_dir=%s;' % data + o + (';' + A.extra if A.extra else '')
    return 'model=M;model_dir=%s;' % A.m_model_dir + o


def run(name, model, shape, frames, stages, trace=None):
    label, rw, rh, ow, oh = shape
    e = dict(env_base(), WG_TEST_RENDER_WIDTH=str(rw), WG_TEST_RENDER_HEIGHT=str(rh), WG_TEST_OUTPUT_WIDTH=str(ow),
             WG_TEST_OUTPUT_HEIGHT=str(oh))
    if trace:
        e.update(WG_LAUNCH_TRACE=trace, WG_LAUNCH_TRACE_MAX='60000')
    log = os.path.join(A.work, '%s.log' % name)
    dll = os.path.join(A.runtime_dir, 'wronggpu_runtime.dll')
    with open(log, 'w', encoding='utf-8') as f:
        rc = subprocess.call(WRAP + [harness(), dll, 'hipbench', options(model, l_data(), stages), str(frames)], env=e,
                             stdout=f, stderr=subprocess.STDOUT, timeout=900, creationflags=NOWIN)
    text = open(log, encoding='utf-8', errors='replace').read()
    tot = [float(x) for x in re.findall(r'TOTAL\(t3-t0\) ([0-9.]+)', text)]
    net = re.findall(r'network GPU ([0-9.]+) \(max [0-9.]+\) ms \[([^\]]*)\]', text)
    ex = re.findall(r'network runtime ready: executor (\w+)', text)
    took = re.findall(r'could not take[^\n]*', text)
    errs = re.findall(r'(?:ERROR|error|refus)[^\n]*', text)
    ok = rc == 0 and len(tot) >= 2 and ex and ex[-1] == 'dx12' and not took and re.search(r'^PASS\s*$', text, re.M)
    return {'name': name, 'model': model, 'shape': label, 'render': [rw, rh], 'output': [ow, oh], 'rc': rc, 'ok': bool(ok),
            'executor': ex[-1] if ex else None, 'total_windows_ms': tot, 'network_windows': net,
            'value_ms': statistics.median(tot[1:]) if len(tot) >= 2 else None, 'errors': errs[:5], 'took': took[:3],
            'log': log}


def create():
    need_run_args()
    trace = os.path.join(A.work, 'create-trace.txt')
    if os.path.exists(trace):
        os.remove(trace)
    res = run('create-L', 'L', SHAPES[0], 600, True, trace)
    counts = {}
    if os.path.isfile(trace):
        for l in open(trace, encoding='utf-8', errors='replace'):
            f = l.split()
            if f:
                counts[f[0]] = counts.get(f[0], 0) + 1
    lk = {k: v for k, v in counts.items() if k.startswith(L_KERNELS)}
    text = open(res['log'], encoding='utf-8', errors='replace').read()
    res['launch_trace'] = {'file': trace, 'kernels': len(counts), 'model_l_launches': lk}
    res['lines'] = [l.strip()[:400] for l in text.splitlines()
                    if re.search(r'network runtime ready|network kernels|model=L|L[\\/]model\.wgcm|ERROR|refus|PASS|FAIL', l)][:20]
    res['verdict'] = 'PASS' if res['ok'] and lk else 'FAIL'
    json.dump(res, open(os.path.join(A.work, 'create-L.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('verdict', 'rc', 'executor', 'value_ms', 'errors', 'took', 'launch_trace', 'lines')},
                     indent=1))
    return 0 if res['verdict'] == 'PASS' else 1


def build_commit():
    try:
        return subprocess.check_output(['git', '-C', A.src, 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def timing():
    need_run_args()
    only = A.shapes.split(',') if A.shapes else None
    shapes = [s for s in SHAPES if not only or s[0] in only]
    runs = []
    for rnd in range(A.rounds):
        for shape in shapes:
            for model in ('L', 'M'):
                r = run('t%d_%s_%s' % (rnd, model, shape[0].replace(' ', '_')), model, shape, 1500, False)
                runs.append(r)
                print('%s %-20s round %d: %s ms ok=%s' % (model, shape[0], rnd, r['value_ms'], r['ok']), flush=True)
    stages = {}
    for shape in shapes:
        for model in ('L', 'M'):
            r = run('stages_%s_%s' % (model, shape[0].replace(' ', '_')), model, shape, 1500, True)
            stages['%s %s' % (model, shape[0])] = r['network_windows'][1:]
    out = {'definition': 'TOTAL(t3-t0) of the runtime timing log, test_interop_d3d12.exe hipbench 1500 frames, executor '
                         'dx12 (native D3D12), first 300-frame window dropped, median of the other windows per run, '
                         'median over rounds interleaved L / M per shape',
           'runtime': os.path.join(A.runtime_dir, 'wronggpu_runtime.dll'), 'build_commit': build_commit(),
           'l_model_sha256': A.model_sha.lower(), 'm_model': os.path.join(A.m_model_dir, 'model.wgcm'), 'shapes': {},
           'stage_windows': stages, 'runs': runs}
    for shape in shapes:
        for model in ('L', 'M'):
            v = [r['value_ms'] for r in runs if r['model'] == model and r['shape'] == shape[0] and r['ok']]
            out['shapes']['%s %s' % (model, shape[0])] = {'median_ms': statistics.median(v) if v else None,
                                                         'rounds_ms': v, 'n': len(v)}
    p = os.path.join(A.work, 'l-timing-harness.json')
    json.dump(out, open(p, 'w'), indent=1)
    print(json.dumps(out['shapes'], indent=1))
    print('record', p)
    return 0


def main(argv):
    global A, WRAP
    if '--' in argv:
        i = argv.index('--')
        argv, WRAP = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['build', 'create', 'timing'])
    ap.add_argument('--work', required=True, help='output folder (harness, model data copy, logs, records)')
    ap.add_argument('--src', default=REPO, help='the checkout with the harness source (default: this one)')
    ap.add_argument('--runtime-dir', help='built runtime folder (wronggpu_runtime.dll + kernels, with Model L)')
    ap.add_argument('--model', help='the converted Model L file (model.wgcm)')
    ap.add_argument('--model-sha', help="the Model L file's SHA-256")
    ap.add_argument('--m-model-dir', default=os.path.join(os.environ.get('LOCALAPPDATA', ''), 'WrongGPU', 'model'),
                    help="Model M's data folder (timing; read only)")
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--shapes', help='comma list of shape names (default: all six)')
    ap.add_argument('--extra', help="more runtime options for Model L's runs, e.g. l_post=own")
    ap.add_argument('--tmp', help='TMP / TEMP for the runs')
    ap.add_argument('--rocm-path', help='ROCM_PATH for the runs')
    ap.add_argument('--vswhere', default=VSWHERE)
    A = ap.parse_args(argv)
    A.work = os.path.abspath(A.work)
    os.makedirs(A.work, exist_ok=True)
    return {'build': build, 'create': create, 'timing': timing}[A.cmd]()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
