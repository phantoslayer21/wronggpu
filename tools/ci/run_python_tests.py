# SPDX-License-Identifier: MIT
"""Run the CPU Python test set (CI, and locally): every clean/**/test_*.py except the modules that need a GPU device.

Each module runs in its own process (package folders with `python -m unittest <module>`, folders without __init__.py
with `discover` in that folder). Skips are fine (each module states its reason: the host lane emulator needs Linux, a
compiler or a file that is not in the repository, an opt-in GPU variable); a failure or an error fails the run.
`clean/wgshared/tests/*.py` are scripts that the wgshared ctest runs, so they are not run here.
Run from the repository root: python tools/ci/run_python_tests.py [--jobs N]
"""
import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time
from pathlib import Path

GPU_MODULES = {   # load a D3D12 / HIP device (BUILDING.md, section 6)
    'clean/k/tests/test_k_hidden_guard.py', 'clean/k/tests/test_k_input_uniforms.py',
    'clean/k/tests/test_k_interop_timing.py', 'clean/k/tests/test_k_norm_r16.py',
    'clean/k/tests/test_k_output_arithmetic.py', 'clean/k/tests/test_k_output_selector.py',
    'clean/k/tests/test_k_runtime_gpu.py', 'clean/k/tests/test_k_tiled_input.py',
    'clean/rr_f/proxy/tests/test_capture_replay.py', 'clean/rr_f/runtime/tests/test_model_pack.py',
}
RUN_ELSEWHERE = {'clean/wgshared/tests/test_dummy_extension.py', 'clean/wgshared/tests/test_gen_registry.py'}


def run_one(root, rel, timeout):
    d = (root / rel).parent
    if (d / '__init__.py').exists():
        args = [sys.executable, '-B', '-m', 'unittest', '-q', rel[:-3].replace('/', '.')]
    else:
        args = [sys.executable, '-B', '-m', 'unittest', 'discover', '-q', '-s', str(d), '-t', str(d), '-p', Path(rel).name]
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout if isinstance(e.stdout, str) else '') + (e.stderr if isinstance(e.stderr, str) else '')
        rc = 'timeout'
    ran = re.findall(r'Ran (\d+) tests?', out)
    skipped = re.findall(r'skipped=(\d+)', out)
    return rel, rc, int(ran[-1]) if ran else 0, int(skipped[-1]) if skipped else 0, time.time() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--timeout', type=int, default=2400)
    a = ap.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    files = sorted(p.relative_to(root).as_posix() for p in (root / 'clean').rglob('test_*.py'))
    todo = [f for f in files if f not in GPU_MODULES and f not in RUN_ELSEWHERE]
    print('python tests: %d files, %d run here, %d GPU modules left out, %d run by ctest'
          % (len(files), len(todo), len(GPU_MODULES & set(files)), len(RUN_ELSEWHERE & set(files))), flush=True)
    failed, total_ran, total_skipped = [], 0, 0
    with cf.ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for rel, rc, ran, skipped, secs, out in ex.map(lambda f: run_one(root, f, a.timeout), todo):
            total_ran += ran
            total_skipped += skipped
            print('%-60s exit %-7s ran %4d skipped %3d  %5.0f s' % (rel, rc, ran, skipped, secs), flush=True)
            if rc != 0:
                failed.append(rel)
                print(out[-6000:], flush=True)
    print('python tests: %d ran, %d skipped, %d module(s) failed%s' % (total_ran, total_skipped, len(failed),
          (': ' + ', '.join(failed)) if failed else ''), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
