# SPDX-License-Identifier: MIT
"""Build direct-texture/staged supplied-device tests. CPU/build work only.

GPU execution is a separate native_direct_test.exe command under the GPU lock (WG_GPU_LOCK, BUILDING.md).
The build directory supplies kernels/manifests and original synthetic models.
This compiles original Model M host code. Both execution paths select native
D3D12; the executable verifies fresh module snapshots contain no HIP runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--rocm', default=r'C:\Program Files\AMD\ROCm\7.2')
    p.add_argument('--model-l', action='store_true',
                   help='also Model L (WG_MODEL_L, clean/l/runtime): the test then runs it with its last argument L')
    a = p.parse_args()
    clean = Path(__file__).resolve().parents[2]
    rt = clean / 'runtime'
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = [Path(__file__).with_name('native_direct_test.cpp')]
    src += [rt / s for s in ('device.cpp', 'device_dx12.cpp', 'device_d3d12.cpp', 'embedded.cpp', 'model.cpp', 'no_model_k.cpp',
                             'runtime.cpp', 'interop/convert.cpp', 'interop/native_candidate.cpp')]
    src += [clean / 'converter/src' / s for s in ('json.cpp', 'sha256.cpp')]
    if a.model_l:
        src += [clean / 'l/runtime' / s for s in ('l_model.cpp', 'l_pack.cpp')]
    exe = out / 'native_direct_test.exe'
    if exe.exists():
        p.error('output executable already exists; select a new private build directory')
    cmd = [str(Path(a.rocm) / 'bin/clang++.exe'), '-std=c++17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
           '-DWG_NO_MODEL_K', *(['-DWG_MODEL_L'] if a.model_l else []), '-fms-runtime-lib=static', *map(str, src),
           '-o', str(exe)]
    dependencies=src+[rt/s for s in ('device.h','device_d3d12.h','runtime_core.h','wg_runtime.h','model.h','embedded.h',
                                    'interop/native_candidate.h','interop/convert.h','interop/d3d12_include.h')]
    dependencies += [clean/'fsr/include/wronggpu_upscaler.h']
    before={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in dependencies}
    tmp=out / 'tmp'
    tmp.mkdir(exist_ok=True)
    env=os.environ.copy()
    env['TMP']=env['TEMP']=str(tmp)
    subprocess.check_call(cmd, cwd=out, env=env)
    after={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in dependencies}
    changed=[s for s in before if before[s]!=after[s]]
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=clean.parent,text=True).strip()
    evidence = {'command': cmd,'source_checkout':str(clean.parent),'source_commit':head,
                'source_sha256_before':before,'source_sha256':after,'changed_during_build':changed,
                'exe': str(exe), 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'gpu_run': False}
    (out / 'native-direct-build.json').write_text(json.dumps(evidence, indent=2))
    if changed:
        raise SystemExit('Source changed during compilation; preserved artifact is not qualified: '+', '.join(changed))
    print(exe)


if __name__ == '__main__':
    main()
