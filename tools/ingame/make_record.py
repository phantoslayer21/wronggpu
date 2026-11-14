# SPDX-License-Identifier: MIT
"""The in-game timing record of a run (e.g. Model L against Model M in one session), from the runtime's own timing
lines in our front-end log only ("timing, last 300 dispatches: network GPU <mean> (max <max>) ms [enc0 .., Swin ..,
dec0 .., post ..] (<n> frames); TOTAL(t3-t0) <mean> (max <max>) ms"): the runtime's D3D12 timestamps in the game's
command list, each line the mean over its 300-dispatch window. Split by game process and by scene (wall-clock
boundaries you take from run.py's session log and screenshots). No other source of numbers.

    python make_record.py --log <wronggpu-fsr.log> --scenes <scenes.json> --out <record.json>
                          [--meta <meta.json>] [--checkout <dir>] [--gpu-name-like <text>]

scenes.json: a list of {"label", "model", "pid", "scene", "from", "to"} (times HH:MM:SS; a window counts when it
starts in [from, to)). meta.json: free fields copied into the record (game, settings, model files, notes such as which
post the run used). --checkout: the build's source checkout (its HEAD goes in as build_commit). --gpu-name-like: the
display adapter whose driver version goes in (Win32_VideoController name filter, e.g. "*9070*")."""
import argparse
import json
import re
import statistics
import subprocess
import sys

LINE = re.compile(r'^\[(\d\d:\d\d:\d\d)\.\d+ (\d+)\].*timing, last 300 dispatches: network GPU ([0-9.]+) \(max ([0-9.]+)\) ms '
                  r'\[enc0 ([0-9.]+), Swin ([0-9.]+), dec0 ([0-9.]+), post ([0-9.]+)\] \((\d+) frames\); '
                  r'TOTAL\(t3-t0\) ([0-9.]+) \(max ([0-9.]+)\) ms')
DESTROYED = re.compile(r'^\[[0-9:.]+ (\d+)\].*destroyed after (\d+) dispatches')
FAILED = re.compile(r'^\[[0-9:.]+ (\d+)\].*(?:dispatch failed|evaluate failed|(\d+) failed)')


def p95(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def stats(v):
    return {'median_ms': round(statistics.median(v), 3), 'p95_ms': round(p95(v), 3), 'max_ms': round(max(v), 3),
            'min_ms': round(min(v), 3)}


def parse(log):
    rows, disp, failed = [], {}, {}
    for l in open(log, encoding='utf-8', errors='replace'):
        m = LINE.match(l)
        if m:
            g = [float(x) for x in m.group(3, 4, 5, 6, 7, 8)] + [int(m.group(9))] + [float(x) for x in m.group(10, 11)]
            rows.append({'time': m.group(1), 'pid': m.group(2), 'net': g[0], 'net_max': g[1], 'enc0': g[2], 'Swin': g[3],
                         'dec0': g[4], 'post': g[5], 'frames': g[6], 'total': g[7], 'total_max': g[8]})
            continue
        m = DESTROYED.match(l)
        if m:
            disp[m.group(1)] = disp.get(m.group(1), 0) + int(m.group(2))
            continue
        m = FAILED.match(l)
        if m:
            failed[m.group(1)] = failed.get(m.group(1), 0) + (int(m.group(2)) if m.group(2) else 1)
    return rows, disp, failed


def gpu_driver(like):
    if not like:
        return None
    ps = ("(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like '%s' } | Select-Object -First 1 | "
          "ForEach-Object { $_.Name + ' | driver ' + $_.DriverVersion })" % like.replace("'", "''"))
    return subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True,
                          creationflags=0x08000000).stdout.strip() or None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--log', required=True, help='our front-end log (wronggpu-fsr.log) copied by run.py collect')
    ap.add_argument('--scenes', required=True, help='scenes.json (see the docstring)')
    ap.add_argument('--out', required=True, help='the record to write (JSON)')
    ap.add_argument('--meta', help='meta.json: fields copied into the record')
    ap.add_argument('--checkout', help="the build's source checkout (build_commit)")
    ap.add_argument('--gpu-name-like', help='display adapter name filter for the driver version')
    a = ap.parse_args(argv)
    scenes = json.load(open(a.scenes, encoding='utf-8'))
    rows, disp, failed = parse(a.log)
    commit = None
    if a.checkout:
        commit = subprocess.run(['git', '-C', a.checkout, 'rev-parse', 'HEAD'], capture_output=True,
                                text=True).stdout.strip() or None
    out = {
        'timer': "the runtime's D3D12 timestamps in the game's command list (timing_log=1, timing_stages=1): each sample "
                 "is one 'timing, last 300 dispatches' line = the mean over 300 frames; stage spans [0] start of the "
                 "network -> [1] first Swin kernel -> [2] dec0 -> [3] post -> [4] after the last kernel; network_total = "
                 "[0]->[4]; TOTAL_t3_t0 = the frame's full D3D12 span",
        'statistics': 'over the 300-frame windows of each scene: median, p95, min of the window means; max_ms = the '
                      'largest window mean; the largest single frame per window is in the log (max ...) and given as '
                      'worst_frame_network_ms / worst_frame_total_ms',
    }
    if a.meta:
        out.update(json.load(open(a.meta, encoding='utf-8')))
    out.update({'gpu_driver': gpu_driver(a.gpu_name_like), 'build_commit': commit, 'dispatches_per_process': disp,
                'failed_dispatches_per_process': failed, 'log': a.log, 'scenes': {}})
    for s in scenes:
        w = [r for r in rows if r['pid'] == str(s['pid']) and s['from'] <= r['time'] < s['to']]
        if not w:
            print('no timing windows for scene %r' % s['label'])
            continue
        out['scenes'][s['label']] = {
            'model': s['model'], 'pid': str(s['pid']), 'scene': s.get('scene', ''), 'windows': len(w),
            'frames': sum(r['frames'] for r in w), 'from': w[0]['time'], 'to': w[-1]['time'],
            'enc0': stats([r['enc0'] for r in w]), 'Swin': stats([r['Swin'] for r in w]),
            'dec0': stats([r['dec0'] for r in w]), 'post': stats([r['post'] for r in w]),
            'network_total': stats([r['net'] for r in w]), 'TOTAL_t3_t0': stats([r['total'] for r in w]),
            'worst_frame_network_ms': max(r['net_max'] for r in w), 'worst_frame_total_ms': max(r['total_max'] for r in w)}
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, indent=1)
    for k, s in out['scenes'].items():
        print('%-22s n=%3d  net %s  enc0 %s Swin %s dec0 %s post %s' % (
            k, s['windows'], s['network_total']['median_ms'], s['enc0']['median_ms'], s['Swin']['median_ms'],
            s['dec0']['median_ms'], s['post']['median_ms']))
    print('driver:', out['gpu_driver'], '| commit:', commit, '| dispatches:', disp, '| failed:', failed)
    print('record', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
