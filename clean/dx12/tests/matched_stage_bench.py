# SPDX-License-Identifier: MIT
"""Reproducible real-frame stage capture/replay; synthetic Model M only.

--prepare compiles a private executable and generates two original OC input frames
and private synthetic model parameters. It never runs GPU work. --run captures
frame 1 of that stream on native D3D12 and replays identical per-stage allocation
snapshots on HIP/native. Run --run under the shared GPU lock and activity wrapper.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def environment(out):
    env = os.environ.copy()
    for k in ('HIP_PATH', 'ROCM_PATH', 'LLVM_PATH', 'HIP_DEVICE_LIB_PATH', 'HIP_CLANG_PATH',
              'WG_D3D12_PROFILE', 'WG_LAUNCH_TRACE', 'WG_D3D12_BATCH', 'WG_RUNTIME_OPTIONS',
              'WG_POST_KERNEL', 'WG_SWIN_VARIANT', 'WG_RECON_SIZE', 'WG_DX12_KERNELS',
              'WG_ENC0_VARIANT', 'WG_DEC0_VARIANT', 'WG_EXPOSURE_VARIANT', 'WG_FILTER_KERNEL',
              'WG_SWIN_STAGE_VARIANTS', 'WG_SWIN_FUSE_STAGES', 'WG_POST_WAVE', 'WG_SWIN_BNECK',
              'WG_DEC1_DEC0', 'WG_SWIN_MID', 'WG_POST_PACK', 'WG_POST_BANDS', 'WG_SWIN_RESIDENT', 'WG_POST_TF'):
        env.pop(k, None)
    (out / 'tmp').mkdir(exist_ok=True)
    env['TMP'] = env['TEMP'] = str(out / 'tmp')
    return env


def prepare_clock(a, out):
    """Honor Device::read_alt's architecture-qualified packaging; no GPU calls."""
    original = Path(a.clock_co).resolve()
    original_hash = digest(original)
    command = [str(Path(a.rocm) / 'bin/llvm-readobj.exe'), '--notes', str(original)]
    notes = subprocess.check_output(command, text=True, env=environment(out))
    if 'amdhsa.target:   amdgcn-amd-amdhsa--gfx1201' not in notes:
        raise ValueError('clock probe must positively identify its gfx1201 AMDGPU target')
    clock_dir = out / 'clock'
    clock_dir.mkdir(exist_ok=True)
    physical = clock_dir / 'gfx1201.clk.co'
    if physical.exists():
        if digest(physical) != original_hash:
            raise ValueError(f'preserve differing clock payload: {physical}')
    else:
        shutil.copyfile(original, physical)
    if digest(physical) != original_hash:
        raise ValueError('clock payload copy hash mismatch')
    (clock_dir / 'llvm-readobj-notes.txt').write_text(notes)
    return {'original': str(original), 'original_sha256': original_hash,
            'logical_load_path': str(clock_dir / 'clk.co'), 'physical_path': str(physical),
            'physical_sha256': digest(physical), 'arch': 'gfx1201', 'inspection_command': command,
            'lookup_contract': 'Device::read_alt requests gfx1201.clk.co for logical clk.co'}


def prepare(a, out):
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from clean.runner import output_filter_original_check as OC
    from clean.runner import synthetic_model as SM
    from clean.runner.pipeline import WgRtFrame
    rt = ROOT / 'clean/runtime'
    sources = [HERE / 'matched_stage_bench.cpp']
    sources += [rt / s for s in ('device.cpp', 'device_dx12.cpp', 'device_d3d12.cpp',
                                'embedded.cpp', 'model.cpp', 'no_model_k.cpp', 'runtime.cpp')]
    sources += [ROOT / 'clean/converter/src' / s for s in ('json.cpp', 'sha256.cpp')]
    exe = out / 'matched_stage_bench.exe'
    if a.rebuild and exe.exists():
        prior = out / 'prior-builds' / digest(exe)
        prior.mkdir(parents=True, exist_ok=True)
        if not (prior / exe.name).exists():
            shutil.copyfile(exe, prior / exe.name)
        old_metadata = out / 'matched-stage-build.json'
        if old_metadata.exists() and not (prior / old_metadata.name).exists():
            shutil.copyfile(old_metadata, prior / old_metadata.name)
    command = [str(Path(a.rocm) / 'bin/clang++.exe'), '-std=c++17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
               '-DWG_NO_MODEL_K', '-fms-runtime-lib=static', *map(str, sources), '-o', str(exe)]
    subprocess.check_call(command, cwd=out, env=environment(out))
    cpu_self_test = subprocess.check_output([str(exe), 'self-test'], text=True, cwd=out, env=environment(out))
    render, output = {'1440p': ((1706, 960), (2560, 1440)),
                      'odd': ((853, 481), (1279, 721)),
                      '4k-performance': ((1920, 1080), (3840, 2160))}[a.shape]
    inputs = out / ('inputs-' + a.shape)
    if inputs.exists() and not a.rebuild:
        raise FileExistsError(f'Preserve existing input evidence; select a new --out: {inputs}')
    inputs.mkdir(exist_ok=True)
    for i in range(0 if a.rebuild else 2):
        x = OC.frame_input(i, render, reset=i == 0)
        for name in ('color', 'depth', 'motion'):
            np.ascontiguousarray(x[name], dtype=np.float32).tofile(inputs / f'{i}-{name}.bin')
        fr = WgRtFrame()
        fr.rw, fr.rh = render
        fr.mv_scale[:] = x['mv_scale']
        fr.mv_bias[:] = [0, 0]
        fr.jitter[:] = x['jitter']
        fr.exposure = x['exposure']
        fr.pre_exposure = x['pre_exposure']
        fr.reset = int(x['reset'])
        (inputs / f'{i}-frame.bin').write_bytes(bytes(fr))
    input_metadata = {
        'rw': render[0], 'rh': render[1], 'ow': output[0], 'oh': output[1],
        'depth_inverted': int(a.shape != 'odd'), 'seed': OC.SEED, 'frame_indices': [0, 1],
        'generator': 'clean.runner.output_filter_original_check.frame_input',
        'frame_struct_bytes': ctypes.sizeof(WgRtFrame),
        'files': {p.name: digest(p) for p in inputs.glob('*.bin')},
    }
    if not a.rebuild:
        (inputs / 'inputs.json').write_text(json.dumps(input_metadata, indent=2))
    else:
        old = json.loads((inputs / 'inputs.json').read_text())
        if old != input_metadata:
            raise ValueError('existing input metadata/hashes differ on --rebuild')
    model = out / 'synthetic-model'
    if model.exists() and not a.rebuild:
        raise FileExistsError(f'Preserve existing model evidence; select a new --out: {model}')
    if not a.rebuild:
        SM.write_raw(model, SM.synthetic_tensors(OC.SEED, taps=SM.output_taps('random', OC.SEED)))
    model_hashes = {str(p.relative_to(model)): digest(p) for p in model.rglob('*') if p.is_file()}
    if a.rebuild:
        old = json.loads((out / 'matched-stage-build.json').read_text())
        if old['model_sha256'] != model_hashes:
            raise ValueError('existing synthetic model hashes differ on --rebuild')
    build = Path(a.build).resolve()
    manifest = json.loads((build / 'dx12/manifest.json').read_text())
    payloads = {str(build / 'dx12' / k['file']): digest(build / 'dx12' / k['file']) for k in manifest['kernels']}
    for k in manifest['kernels']:
        p = build / k['co']
        if p.is_file():
            payloads[str(p)] = digest(p)
        for arch in ('gfx1200', 'gfx1201'):
            # These architecture-qualified files, not the logical .co name,
            # are what the current HIP Device actually reads.
            for physical in (p.with_name(arch + '.' + p.name), p.parent / arch / p.name):
                if physical.is_file():
                    payloads[str(physical)] = digest(physical)
    clock = prepare_clock(a, out)
    evidence = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': command,
                'source_sha256': {str(p): digest(p) for p in sources}, 'exe_sha256': digest(exe),
                'payload_sha256': payloads, 'manifest_sha256': digest(build / 'dx12/manifest.json'),
                'model_sha256': model_hashes,
                'gpu_run': False, 'shape': a.shape, 'build': str(build), 'source_root': str(ROOT), 'clock_probe': clock,
                'cpu_self_test': cpu_self_test.strip()}
    (out / 'matched-stage-build.json').write_text(json.dumps(evidence, indent=2))
    print(f'CPU-only preparation complete: {exe}', flush=True)


def execute(command, log, env, out):
    print('RUN ' + subprocess.list2cmdline(command), flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=out, env=env)
    print(log.read_text(encoding='utf-8'), end='', flush=True)
    if p.returncode:
        raise RuntimeError(f'Exit {p.returncode}: {log}')


def summarize(results):
    table = []
    for result in results:
        row = {'kernel': result['kernel'], 'grid': result['grid'], 'block': result['block']}
        row['clock_snapshots_valid'] = bool(result['clock_snapshots']) and all(s['valid'] for s in result['clock_snapshots'])
        for mode, metric, tag in [('event', 'gpu_ms_per_launch', 'gpu'), ('stream', 'wall_ms_per_launch', 'wall')]:
            invalid = [i for i, s in enumerate(result['samples']) if s['mode'] == mode and not s[tag + '_valid']]
            row[tag + '_invalid_sample_indices'] = invalid
            if invalid:
                for backend in ('hip', 'd3d12'):
                    row[f'{backend}_{tag}_median_ms'] = None
                row[f'{tag}_delta_ms'] = row[f'{tag}_ratio'] = row[f'{tag}_paired_delta_ms'] = None
                continue
            med = {}
            for backend in ('hip', 'd3d12'):
                values = [s[metric] for s in result['samples'] if s['mode'] == mode and s['backend'] == backend]
                med[backend] = statistics.median(values)
                row[f'{backend}_{tag}_median_ms'] = med[backend]
            row[f'{tag}_delta_ms'] = med['d3d12'] - med['hip']
            row[f'{tag}_ratio'] = med['d3d12'] / med['hip']
            pairs = {}
            for s in result['samples']:
                if s['mode'] == mode:
                    pairs.setdefault((s['round'], s['pair']), {})[s['backend']] = s[metric]
            row[f'{tag}_paired_delta_ms'] = [p['d3d12'] - p['hip'] for p in pairs.values()]
        table.append(row)
    return table


def run(a, out):
    build = Path(a.build).resolve()
    metadata = json.loads((out / 'matched-stage-build.json').read_text())
    if metadata['shape'] != a.shape:
        raise ValueError('--shape differs from prepared input evidence')
    exe = out / 'matched_stage_bench.exe'
    if digest(exe) != metadata['exe_sha256']:
        raise ValueError('executable differs from prepared hash')
    if digest(build / 'dx12/manifest.json') != metadata['manifest_sha256']:
        raise ValueError('manifest differs from prepared hash')
    for p, expected in metadata['payload_sha256'].items():
        if digest(p) != expected:
            raise ValueError(f'payload changed: {p}')
    env = environment(out)
    clock = metadata['clock_probe']
    clock_co = Path(clock['physical_path'])
    if digest(clock_co) != clock['physical_sha256'] or digest(a.clock_co) != clock['original_sha256']:
        raise ValueError('clock probe differs from prepared, architecture-qualified payload')
    env['WG_STAGE_CLOCK_CO'] = clock['logical_load_path']
    capture = out / ('capture-' + a.shape)
    if not a.replay_only:
        if capture.exists():
            raise FileExistsError(f'Preserve prior capture; --replay-only or new --out: {capture}')
        capture.mkdir()
        execute([str(exe), 'capture', str(build), str(out / 'synthetic-model'), str(out / ('inputs-' + a.shape)), str(capture)],
                out / 'capture.log', env, out)
    stages = json.loads((capture / 'capture.json').read_text())['stages']
    results = []
    for stage in stages:
        case = json.loads((capture / stage / 'case.json').read_text())
        if a.kernels and case['kernel'] not in a.kernels.split(','):
            continue
        report = out / f'{stage}-{a.run_tag}.json'
        if report.exists():
            raise FileExistsError(f'Preserve prior results; use a unique --run-tag: {report}')
        execute([str(exe), 'replay', str(build), str(capture / stage), str(report), a.rocm, str(a.warm), str(a.rounds),
                 str(a.repeats), str(a.batch), str(a.minimum_window_seconds)], out / f'{stage}-{a.run_tag}.log', env, out)
        results.append(json.loads(report.read_text()))
    table = summarize(results)
    summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'table': table, 'stages': results,
               'telemetry': 'Use external gpu_evidence report; this benchmark never asserts isolation.',
               'qualification': 'Diagnostic standalone-stage attribution, not a complete-path gain or shader-body timing.',
               'build_metadata': metadata, 'clock_co': str(clock_co), 'clock_co_sha256': digest(clock_co)}
    (out / f'summary-{a.run_tag}.json').write_text(json.dumps(summary, indent=2))
    for row in table:
        print(f"{row['kernel']:40s} HIP {row['hip_gpu_median_ms']} native {row['d3d12_gpu_median_ms']} "
              f"delta {row['gpu_delta_ms']} ms invalid={row['gpu_invalid_sample_indices']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True, help='frozen complete HIP/native payload build')
    p.add_argument('--out', required=True, help='private build and evidence directory')
    p.add_argument('--rocm', default=r'C:\Program Files\AMD\ROCm\7.2')
    p.add_argument('--prepare', action='store_true', help='CPU-only compile/input/model generation')
    p.add_argument('--rebuild', action='store_true', help='with --prepare, recompile while preserving verified original inputs/model')
    p.add_argument('--run', action='store_true', help='GPU work: use shared lock + activity wrapper')
    p.add_argument('--replay-only', action='store_true')
    p.add_argument('--shape', choices=('1440p', 'odd', '4k-performance'), default='1440p')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--repeats', type=int, default=24)
    p.add_argument('--batch', type=int, choices=(4, 16), default=16)
    p.add_argument('--warm', type=float, default=3)
    p.add_argument('--minimum-window-seconds', type=float, default=1.0,
                   help='each paired marker window runs at least this long AND satisfies --repeats; all samples retained')
    p.add_argument('--kernels', help='comma-separated launch names; captures full frame, then replays selection')
    p.add_argument('--run-tag', default='paired-r0')
    p.add_argument('--clock-co', default=os.environ.get('WG_CLOCK_CO', ''), help='original synthetic cycle/realtime probe')
    a = p.parse_args()
    if not a.prepare and not a.run:
        p.error('choose --prepare and/or --run')
    if not (math.isfinite(a.warm) and a.warm >= 3 and math.isfinite(a.minimum_window_seconds) and
            a.minimum_window_seconds >= 1 and a.rounds >= 1 and a.repeats >= 1):
        p.error('warm >= 3 seconds, minimum window >= 1 second, positive rounds/repeats required')
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.prepare:
        prepare(a, out)
    if a.run:
        run(a, out)


if __name__ == '__main__':
    main()
