# SPDX-License-Identifier: MIT
"""Run the CPU Python test set (CI, and locally): every clean/**/test_*.py except the modules that need a GPU device.

Each module runs in its own process (package folders with `python -m unittest <module>`, folders without __init__.py
with `discover` in that folder). Skips are fine (each module states its reason: the host lane emulator needs Linux, a
compiler or a file that is not in the repository, an opt-in GPU variable); a failure or an error fails the run.
`clean/wgshared/tests/*.py` are scripts that the wgshared ctest runs, so they are not run here.

CI runs the set in parallel jobs, which together run exactly the same tests as one run without options:
  --group proxy   only the modules that need the Ray Reconstruction plugin's CPU build (WG_RR_PROXY_BUILD), run in the
                  job that builds the plugin;
  --shard K/N     part K of N of all other modules. The modules in SPLIT_BY_TEST (long, independent test methods) are
                  split into SPLIT_CHUNKS lists of test ids; units are spread over the N parts by their measured time
                  (WEIGHTS; a module without an entry counts DEFAULT_WEIGHT). Times only change the balance: every
                  shard computes the whole plan and checks that each module (or each test id of a split module) is in
                  exactly one part.
  --plan N        print the whole plan for N shards (and the proxy group) and exit.
Run from the repository root: python tools/ci/run_python_tests.py [--jobs N] [--group proxy | --shard K/N | --plan N]
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
PROXY_GROUP = {'clean/rr_f/proxy/tests/test_optimal.py'}   # reads test_proxy_core.exe from WG_RR_PROXY_BUILD
SPLIT_BY_TEST = {'clean/rr_f/tests/test_stages.py'}
SPLIT_CHUNKS = 6
# seconds per module in CI (run 43531456640, 3 modules at a time); balance only
WEIGHTS = {
    'clean/rr_f/tests/test_stages.py': 812, 'clean/rr_f/tests/test_vectors.py': 240,
    'clean/tests/test_post_tf_geometry.py': 145, 'clean/rr_f/head/tests/test_stage.py': 98,
    'clean/rr_f/tests/test_scalar.py': 80, 'clean/rr_f/tests/test_input_stage.py': 72,
    'clean/k/tests/test_k_runtime_pack.py': 51, 'clean/rr_f/tests/test_network.py': 48,
    'clean/rr_f/tests/test_colour_symmetry.py': 46, 'clean/k/tests/test_k_runtime_tables.py': 42,
    'clean/rr_f/head/tests/test_scalar_rules.py': 41, 'clean/rr_f/tests/test_gpu_pipeline.py': 40,
    'clean/rr_f/tests/test_gpu_network.py': 40, 'clean/rr_f/tests/test_prepared_pipeline.py': 39,
    'clean/rr_f/tests/test_input_lifecycle.py': 39, 'clean/rr_f/tests/test_resolve.py': 30,
    'clean/tests/test_encoder_dither.py': 27, 'clean/rr_f/tests/test_block.py': 25,
    'clean/tests/test_model_filter.py': 17, 'clean/tests/test_runtime_frame_trace.py': 14,
}
DEFAULT_WEIGHT = 5


def module_args(root, rel):
    d = (root / rel).parent
    if (d / '__init__.py').exists():
        return [sys.executable, '-B', '-m', 'unittest', '-q', rel[:-3].replace('/', '.')]
    return [sys.executable, '-B', '-m', 'unittest', 'discover', '-q', '-s', str(d), '-t', str(d), '-p', Path(rel).name]


def test_ids(root, rel):
    """The test ids of a module in a package folder, sorted."""
    code = ('import sys, unittest\n'
            'def walk(s):\n'
            '    for t in s:\n'
            '        yield from (walk(t) if isinstance(t, unittest.TestSuite) else [t.id()])\n'
            'print("\\n".join(sorted(walk(unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])))))\n')
    r = subprocess.run([sys.executable, '-B', '-c', code, rel[:-3].replace('/', '.')], cwd=root, capture_output=True,
                       text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit('cannot list the tests of %s:\n%s' % (rel, (r.stdout + r.stderr)[-4000:]))
    return [l for l in r.stdout.splitlines() if l.strip()]


def run_one(root, unit, timeout):
    """unit: (label, rel, test ids or None)."""
    label, rel, ids = unit
    args = module_args(root, rel) if ids is None else [sys.executable, '-B', '-m', 'unittest', '-q'] + ids
    t0 = time.time()
    try:
        r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries the captured output as bytes, even with text=True
        out = ''.join(s.decode(errors='replace') if isinstance(s, bytes) else (s or '') for s in (e.stdout, e.stderr))
        rc = 'timeout'
    ran = re.findall(r'Ran (\d+) tests?', out)
    skipped = re.findall(r'skipped=(\d+)', out)
    return label, rc, int(ran[-1]) if ran else 0, int(skipped[-1]) if skipped else 0, time.time() - t0, out


def plan(root, todo, n):
    """Split todo (minus the proxy group) into n parts: [[unit, ...], ...] and the proxy group's units."""
    units = []
    for rel in todo:
        if rel in PROXY_GROUP:
            continue
        if rel in SPLIT_BY_TEST and n > 1:
            ids = test_ids(root, rel)
            chunks = [ids[i::SPLIT_CHUNKS] for i in range(SPLIT_CHUNKS)]
            w = WEIGHTS.get(rel, DEFAULT_WEIGHT) / SPLIT_CHUNKS
            units += [(w, ('%s [%d/%d]' % (rel, i + 1, SPLIT_CHUNKS), rel, c)) for i, c in enumerate(chunks) if c]
        else:
            units.append((WEIGHTS.get(rel, DEFAULT_WEIGHT), (rel, rel, None)))
    parts, load = [[] for _ in range(n)], [0.0] * n
    for w, u in sorted(units, key=lambda x: (-x[0], x[1][0])):   # longest first, each to the least loaded part
        k = min(range(n), key=lambda i: (load[i], i))
        parts[k].append(u)
        load[k] += w
    # every module, or every test id of a split module, in exactly one part
    seen = {}
    for k, p in enumerate(parts):
        for label, rel, ids in p:
            for key in ([rel] if ids is None else [rel + '::' + t for t in ids]):
                if key in seen:
                    raise SystemExit('plan error: %s in parts %d and %d' % (key, seen[key] + 1, k + 1))
                seen[key] = k
    covered = {key.split('::')[0] for key in seen}
    missing = [rel for rel in todo if rel not in PROXY_GROUP and rel not in covered]
    if missing:
        raise SystemExit('plan error: not in any part: %s' % missing)
    proxy = [(rel, rel, None) for rel in todo if rel in PROXY_GROUP]
    return parts, load, proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--timeout', type=int, default=2400)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--shard', help='K/N: part K of N of the modules outside the proxy group')
    g.add_argument('--group', choices=['proxy'], help='only the modules that need the plugin build')
    g.add_argument('--plan', type=int, metavar='N', help='print the plan for N shards and exit')
    a = ap.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    files = sorted(p.relative_to(root).as_posix() for p in (root / 'clean').rglob('test_*.py'))
    todo = [f for f in files if f not in GPU_MODULES and f not in RUN_ELSEWHERE]
    print('python tests: %d files, %d run here, %d GPU modules left out, %d run by ctest'
          % (len(files), len(todo), len(GPU_MODULES & set(files)), len(RUN_ELSEWHERE & set(files))), flush=True)
    if a.plan:
        parts, load, proxy = plan(root, todo, a.plan)
        for k, p in enumerate(parts):
            print('shard %d/%d: %d units, about %.0f s of test time' % (k + 1, a.plan, len(p), load[k]))
            for label, rel, ids in p:
                print('  ' + label + ('' if ids is None else ' (%d tests)' % len(ids)))
        print('proxy group: ' + ', '.join(u[0] for u in proxy))
        return 0
    if a.shard:
        m = re.fullmatch(r'(\d+)/(\d+)', a.shard)
        k, n = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        if not 1 <= k <= n:
            raise SystemExit('--shard K/N with 1 <= K <= N')
        parts, load, _ = plan(root, todo, n)
        units = parts[k - 1]
        print('python tests: shard %d/%d, %d units (about %.0f s of test time)' % (k, n, len(units), load[k - 1]),
              flush=True)
    elif a.group == 'proxy':
        units = [(rel, rel, None) for rel in todo if rel in PROXY_GROUP]
        print('python tests: proxy group, %d module(s)' % len(units), flush=True)
    else:
        units = [(rel, rel, None) for rel in todo]
    failed, total_ran, total_skipped = [], 0, 0
    with cf.ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for label, rc, ran, skipped, secs, out in ex.map(lambda u: run_one(root, u, a.timeout), units):
            total_ran += ran
            total_skipped += skipped
            print('%-60s exit %-7s ran %4d skipped %3d  %5.0f s' % (label, rc, ran, skipped, secs), flush=True)
            if rc != 0:
                failed.append(label)
                print(out[-6000:], flush=True)
    print('python tests: %d ran, %d skipped, %d module(s) failed%s' % (total_ran, total_skipped, len(failed),
          (': ' + ', '.join(failed)) if failed else ''), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
