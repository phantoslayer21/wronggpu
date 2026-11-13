# SPDX-License-Identifier: MIT
"""Build the private synthetic supplied-device candidate. CPU/build work only.

GPU execution is a separate native_game_test.exe command under the GPU lock (WG_GPU_LOCK, BUILDING.md).
The build directory supplies kernels/manifests and original synthetic models.
This compiles only original Model M host code; HIP is dynamically loaded only if
the executable's explicitly selected differential oracle is `hip`.
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
    a = p.parse_args()
    clean = Path(__file__).resolve().parents[2]
    rt = clean / 'runtime'
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = [Path(__file__).with_name('native_game_test.cpp')]
    src += [rt / s for s in ('device.cpp', 'device_dx12.cpp', 'device_d3d12.cpp', 'embedded.cpp', 'model.cpp', 'no_model_k.cpp',
                             'runtime.cpp', 'interop/convert.cpp', 'interop/native_candidate.cpp')]
    src += [clean / 'converter/src' / s for s in ('json.cpp', 'sha256.cpp')]
    exe = out / 'native_game_test.exe'
    cmd = [str(Path(a.rocm) / 'bin/clang++.exe'), '-std=c++17', '-O2', '-D_CRT_SECURE_NO_WARNINGS',
           '-DWG_NO_MODEL_K', '-fms-runtime-lib=static', *map(str, src), '-o', str(exe)]
    tmp=out / 'tmp'
    tmp.mkdir(exist_ok=True)
    env=os.environ.copy()
    env['TMP']=env['TEMP']=str(tmp)
    subprocess.check_call(cmd, cwd=out, env=env)
    evidence = {'command': cmd, 'source_sha256': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in src},
                'exe': str(exe), 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(), 'gpu_run': False}
    (out / 'native-game-build.json').write_text(json.dumps(evidence, indent=2))
    print(exe)


if __name__ == '__main__':
    main()
