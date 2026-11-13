# SPDX-License-Identifier: MIT
"""Build or prove the packaged native DLL through its real opaque C ABI.

--prepare is CPU-only. Run mode must be invoked under the GPU lock (WG_GPU_LOCK, BUILDING.md) and the
root-owned activity monitor. No performance claim is made by this diagnostic.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


COMMON = ('swin_block_c64_nc', 'patch_merge_c64', 'swin_block_c96_nc', 'patch_merge_c96',
          'bneck/swin_bneck6_nc', 'swin_block_c96_add_ex_nc', 'expand_c96',
          'd1p12/swin_block_c64_add_dec0_nc')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    clean = Path(__file__).resolve().parents[2]
    rt = clean / 'runtime'
    sources = [Path(__file__).with_name('native_direct_dll_test.cpp')]
    sources += [rt / s for s in ('device.cpp', 'device_dx12.cpp', 'device_d3d12.cpp', 'embedded.cpp',
                                'model.cpp', 'no_model_k.cpp', 'runtime.cpp', 'interop/convert.cpp',
                                'interop/native_candidate.cpp')]
    sources += [clean / 'converter/src' / s for s in ('json.cpp', 'sha256.cpp')]
    dependencies = sorted(set(sources + list(rt.rglob('*.h')) + list((clean/'converter/src').glob('*.h')) +
                              [clean/'fsr/include/wronggpu_upscaler.h', Path(__file__).resolve()]))
    before = {str(p): sha(p) for p in dependencies}
    exe = out / 'native_direct_dll_test.exe'
    command = [str(Path(a.rocm)/'bin/clang++.exe'), '-std=c++17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
               '-DWG_NO_MODEL_K', '-fms-runtime-lib=static', *map(str, sources), '-o', str(exe)]
    env = os.environ.copy()
    tmp = out/'tmp'
    tmp.mkdir()
    env['TEMP'] = env['TMP'] = str(tmp)
    result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, errors='replace')
    (out/'build.stdout.log').write_text(result.stdout)
    (out/'build.stderr.log').write_text(result.stderr)
    after = {str(p): sha(p) for p in dependencies}
    changed = [p for p in before if before[p] != after[p]]
    report = {'command': command, 'returncode': result.returncode, 'source_sha256_before': before,
              'source_sha256': after, 'changed_during_build': changed, 'gpu_run': False,
              'exe': str(exe), 'exe_sha256': sha(exe) if exe.exists() else None,
              'pass': result.returncode == 0 and not changed}
    (out/'build.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'pass': report['pass'], 'exe': str(exe), 'changed': changed}, indent=2))
    if result.returncode:
        print(result.stderr[-6000:], file=sys.stderr)
    return 0 if report['pass'] else 1


def trace_scopes(lines, frames):
    scopes = []
    errors = []
    outside = Counter()
    initialization = []
    frames_started = False
    current = None
    for index, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) != 4 or fields[-1] != 'd3d12':
            errors.append({'line': index, 'problem': 'non-native or malformed record', 'text': line})
            continue
        name = fields[0]
        if name in ('native_direct_begin', 'native_staged_begin'):
            frames_started = True
            if current is not None:
                errors.append({'line': index, 'problem': 'nested scope'})
            current = {'route': 'direct' if name == 'native_direct_begin' else 'staged', 'kernels': Counter()}
        elif name in ('native_direct_end', 'native_staged_end'):
            route = 'direct' if name == 'native_direct_end' else 'staged'
            if current is None or current['route'] != route:
                errors.append({'line': index, 'problem': 'unmatched scope end'})
            else:
                scopes.append(current)
                current = None
        elif current is not None:
            current['kernels'][name] += 1
        else:
            outside[name] += 1
            initialization.append(fields)
            if frames_started:
                errors.append({'line': index, 'problem': 'outside work after frames started', 'text': line})
    if current is not None:
        errors.append({'problem': 'unfinished scope'})
    for scope in scopes:
        suffix = '_direct' if scope['route'] == 'direct' else ''
        expected = (*COMMON, 'input_encoder'+suffix, 'w64/post_tf_f32_p12_diet_vsit'+suffix)
        exposure = ('exposure_auto_diet'+suffix, 'exposure_pair_diet'+suffix)
        k = scope['kernels']
        if any(k[name] != 1 for name in expected) or sum(k[name] for name in exposure) != 1:
            errors.append({'problem': 'stage count differs', 'scope': scope})
        if suffix:
            extra = set(k)-set(expected)-set(exposure)
            if extra:
                errors.append({'problem': 'unexpected direct work', 'extra': sorted(extra)})
        elif any(k[name] != 1 for name in ('native_staged_input_conversion', 'native_staged_output_conversion')):
            errors.append({'problem': 'staged converter missing', 'scope': scope})
    if [s['route'] for s in scopes] != ['direct', 'staged']*frames:
        errors.append({'problem': 'wrong frame scope order/count'})
    # Each borrowed NativeDevice first runs its driver-channel probe. Runtime
    # creation then zeroes expo_sync and bneck_sync through native_fill; both
    # ranges fit one 256-thread group. The DLL candidate and linked staged
    # reference are created in that order, before either dispatches a frame.
    # Match the entire bounded prefix, including launch geometry and order:
    # these initialization fills must never excuse work between/after frames.
    expected_initialization = [
        ['native_driver_probe', '1', '32', 'd3d12'],
        ['native_fill', '1', '256', 'd3d12'],
        ['native_fill', '1', '256', 'd3d12'],
    ] * 2
    if initialization != expected_initialization:
        errors.append({'problem': 'initialization prefix differs', 'expected': expected_initialization,
                       'actual': initialization})
    return scopes, errors, outside


def run(a):
    if not a.exe or not a.build or not a.model:
        raise SystemExit('run requires --exe, --build, --model')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bundle = Path(a.build).resolve()
    dll = bundle/'wronggpu_runtime.dll'
    manifest = bundle/'dx12/manifest.json'
    build_evidence = Path(a.exe).resolve().parent/'build.json'
    compiled = json.loads(build_evidence.read_text())
    if not compiled.get('pass') or compiled.get('exe_sha256') != sha(a.exe):
        raise SystemExit('executable does not match a stable successful CPU build')
    files = [Path(a.exe).resolve(), dll, manifest, bundle/'variants.json', build_evidence, Path(__file__).resolve()]
    files += [manifest.parent/k['file'] for k in json.loads(manifest.read_text())['kernels']]
    before = {str(p): sha(p) for p in files}
    env = os.environ.copy()
    for name in ('HIP_PATH', 'ROCM_PATH', 'LLVM_PATH', 'HIP_DEVICE_LIB_PATH', 'HIP_CLANG_PATH',
                 'WG_DX12_KERNELS', 'WG_D3D12_PROFILE'):
        env.pop(name, None)
    trace = out/'dispatch.trace'
    env['WG_LAUNCH_TRACE'] = str(trace)
    # The EXE and packaged DLL may each use a static CRT environment snapshot.
    # Set these before process creation so both copies see identical settings.
    env['WG_DX12_DEBUG'] = '1'
    env['WG_D3D12_BATCH'] = '0'
    env['TEMP'] = env['TMP'] = str(out)
    command = [a.exe, str(dll), str(bundle), str(Path(a.model).resolve()), a.queue,
               *map(str, a.shape), str(a.frames), a.output_format]
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True, errors='replace', timeout=900)
    (out/'stdout.log').write_text(result.stdout)
    (out/'stderr.log').write_text(result.stderr)
    after = {str(p): sha(p) for p in files}
    scopes, errors, outside = trace_scopes(trace.read_text().splitlines() if trace.exists() else [], a.frames)
    stdout = result.stdout
    gates = {'exit_zero': result.returncode == 0,
             'final_pass': 'PASS packaged direct DLL ' in stdout,
             'opaque_C_ABI_exports_verified': stdout.count('DLL_EXPORT ') == 5 and stdout.count('verified_owner=1') == 5,
             'packaged_DLL_hash_verified': 'DLL_SHA256 '+before[str(dll)] in stdout and
                                           'DLL_UNCHANGED '+before[str(dll)] in stdout,
             'artifact_hashes_unchanged': before == after,
             'all_frames_byte_exact': stdout.count('PASS DLL frame=') == a.frames,
             'all_direct_dispatches_proved': not errors,
             'fresh_module_snapshots': all('MODULES '+s in stdout for s in ('fresh', 'loaded-dll', 'created', 'after-destroy')),
             'no_HIP_modules': 'amdhip' not in stdout.lower() and 'hiprtc' not in stdout.lower(),
             'debug_states_clean': 'DEBUG enabled' in stdout and 'DEBUG clean' in stdout and
                                   'DEBUG unavailable' not in stdout and 'DEBUG ERROR' not in stdout,
             'ready_ordering': 'PASS DLL delayed input-ready fence held final completion' in stdout,
             'mutation_detected': 'mutation=1' in stdout,
             'null_error_paths': 'PASS packaged DLL C ABI null/error paths' in stdout}
    report = {'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
              'returncode': result.returncode, 'gates': gates, 'pass': all(gates.values()),
              'sha256_before': before, 'sha256_after': after, 'trace_scopes': scopes, 'trace_errors': errors,
              'outside_trace_counts': outside,
              'domain': 'Original synthetic model/scene; opaque loaded DLL direct resources versus linked staged same-D3D12 oracle; typed output bytes and temporal frames.',
              'limits': 'No C ABI internal stage readback is exposed. Internal stage/history proof is a separate static harness; no game qualification or timing claim.'}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'pass': report['pass'], 'gates': gates, 'report': str(out/'report.json')}, indent=2))
    if not report['pass']:
        print(result.stderr[-6000:], file=sys.stderr)
    return 0 if report['pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--out', required=True, help='New private directory; never overwrites evidence')
    p.add_argument('--rocm', default=r'C:\Program Files\AMD\ROCm\7.2')
    p.add_argument('--exe')
    p.add_argument('--build')
    p.add_argument('--model')
    p.add_argument('--queue', choices=('direct', 'compute'), default='direct')
    p.add_argument('--shape', nargs=4, type=int, default=(853, 481, 1279, 721))
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--output-format', choices=('half', 'float'), default='half')
    a = p.parse_args()
    if a.frames < 6 or min(a.shape) < 1:
        p.error('positive shapes and at least six frames required')
    return build(a) if a.prepare else run(a)


if __name__ == '__main__':
    sys.exit(main())
