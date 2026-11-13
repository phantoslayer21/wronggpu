# SPDX-License-Identifier: MIT
"""Run the Ray Reconstruction plugin's CPU test executables (CI, and locally), as clean/rr_f/proxy/tools/build.py runs
them, after `python clean/rr_f/proxy/tools/build.py --out <dir> --no-tests`. That build still builds and runs the
WARP-based probe and hook tests (build.py: the probe test's fake game host and the hook tests' WARP sequence); it skips
only the Python unit tests and the core executables, and this script runs the core executables. The plugin's other
test executables create a Direct3D 12 device and are not run here (BUILDING.md, section 7).
Usage (from the repository root): python tools/ci/run_proxy_core_tests.py <out dir of the plugin build>
"""
import glob
import os
import subprocess
import sys


def main():
    out = sys.argv[1]
    core = os.path.join(out, 'core')
    data = os.path.join('clean', 'rr_f', 'proxy', 'tests', 'data')
    runs = [('core', [os.path.join(core, 'test_proxy_core.exe')]),
            ('probe core', [os.path.join(core, 'test_probe_core.exe')]),
            ('setupapi identity', [os.path.join(core, 'test_setupapi_identity.exe')])]
    runs += [('RR provider ' + os.path.basename(f), [os.path.join(core, 'test_ngx_rr.exe'), f])
             for f in sorted(glob.glob(os.path.join(data, '*-rr-frames.json')))]
    runs += [('replay ' + os.path.basename(f), [os.path.join(core, 'test_replay_profile.exe'), f])
             for f in sorted(glob.glob(os.path.join(data, '*-traffic.json')))]
    bad = 0
    for name, args in runs:
        r = subprocess.run(args, capture_output=True, text=True, timeout=1800)
        last = (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']
        print('%-45s exit %d  %s' % (name, r.returncode, last[0][:150]), flush=True)
        if r.returncode != 0:
            bad += 1
            print((r.stdout + r.stderr)[-4000:], flush=True)
    print('plugin CPU tests: %d run, %d failed' % (len(runs), bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
