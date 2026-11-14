# SPDX-License-Identifier: MIT
"""Video memory over time during an in-game run: every <interval> s, the dedicated GPU memory of the game process and of the
whole adapter (Windows performance counters, read with typeperf; no GPU work of our own).

    python vram_log.py <process name without .exe> <out.csv> [--interval 5] [--max-min 30]

Waits for the process, then samples until it exits (or --max-min). CSV: time, seconds since start, game MiB, adapter MiB.
"""
import argparse
import csv
import io
import subprocess
import time

NW = 0x08000000 | 0x00004000


def pids(name):
    out = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq %s.exe' % name, '/FO', 'CSV', '/NH'], capture_output=True,
                         text=True, creationflags=NW).stdout
    return [int(r[1]) for r in csv.reader(io.StringIO(out)) if len(r) > 1 and r[1].isdigit()]


def sample(pid):
    ctr = [r'\GPU Process Memory(pid_%d_*)\Dedicated Usage' % pid, r'\GPU Adapter Memory(*)\Dedicated Usage']
    out = subprocess.run(['typeperf'] + ctr + ['-sc', '1'], capture_output=True, text=True, creationflags=NW).stdout
    rows = [r for r in csv.reader(io.StringIO(out)) if r]
    if len(rows) < 2:
        return None, None
    head, vals = rows[0], rows[1]
    game = sum(float(v) for h, v in zip(head[1:], vals[1:]) if 'GPU Process Memory' in h and v.strip())
    adapter = max([float(v) for h, v in zip(head[1:], vals[1:]) if 'GPU Adapter Memory' in h and v.strip()] or [0.0])
    return game / 2**20, adapter / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('out')
    ap.add_argument('--interval', type=float, default=5)
    ap.add_argument('--max-min', type=float, default=30)
    a = ap.parse_args()
    while not pids(a.name):
        time.sleep(2)
    pid = pids(a.name)[0]
    t0 = time.time()
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'seconds', 'game_dedicated_MiB', 'adapter_dedicated_MiB', 'pid'])
        while pid in pids(a.name) and time.time() - t0 < a.max_min * 60:
            g, ad = sample(pid)
            w.writerow([time.strftime('%H:%M:%S'), round(time.time() - t0, 1), None if g is None else round(g, 1),
                        None if ad is None else round(ad, 1), pid])
            f.flush()
            time.sleep(a.interval)
    print('vram log done: %s' % a.out)


if __name__ == '__main__':
    main()
