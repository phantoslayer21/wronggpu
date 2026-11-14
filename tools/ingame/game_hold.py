# SPDX-License-Identifier: MIT
"""The GPU flag for ONE game run (a session flag must never outlive the game). Started by run.py launch after it
took the flag: waits up to --start-s for the game process, holds while it runs, and
releases the flag in `finally` on every path (game exit, start timeout, exception, Ctrl-C, kill of the game). The flag
is removed only when its text is still ours.
usage: game_hold.py <flag file> <owner text prefix> <record.json> --exe <game.exe> [--start-s 300] [--max-s 3600]"""
import argparse
import ctypes
import json
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument('flag')
ap.add_argument('owner')
ap.add_argument('record')
ap.add_argument('--exe', required=True, help="the game's process image name, e.g. game.exe")
ap.add_argument('--start-s', type=int, default=300)
ap.add_argument('--max-s', type=int, default=3600)
a = ap.parse_args()
state = {'status': 'waiting', 'exe': a.exe, 'holder_pid': os.getpid(), 'started_unix': time.time()}


def save():
    json.dump(state, open(a.record, 'w'), indent=1)


def find_pid():
    out = subprocess.run(['tasklist', '/fo', 'csv', '/nh', '/fi', 'IMAGENAME eq ' + a.exe], capture_output=True,
                         text=True, creationflags=0x08000000).stdout
    for l in out.splitlines():
        f = l.split('","')
        if len(f) >= 2 and f[0].strip('"').lower() == a.exe.lower():
            return int(f[1])
    return None


k = ctypes.WinDLL('kernel32', use_last_error=True)
k.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
k.OpenProcess.restype = ctypes.c_void_p
k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
k.WaitForSingleObject.restype = ctypes.c_uint32
k.CloseHandle.argtypes = [ctypes.c_void_p]
try:
    save()
    t0 = time.time()
    pid = None
    while time.time() - t0 < a.start_s and not pid:
        pid = find_pid()
        if not pid:
            time.sleep(2)
    if not pid:
        state['status'] = 'game did not start within %d s' % a.start_s
    else:
        state.update(status='held', game_pid=pid, acquired_unix=time.time())
        save()
        h = k.OpenProcess(0x100000, 0, pid)  # SYNCHRONIZE
        if h:
            r = k.WaitForSingleObject(h, a.max_s * 1000)
            k.CloseHandle(h)
            state['status'] = 'game exited' if r == 0 else 'max hold time reached (%d s)' % a.max_s
        else:
            state['status'] = 'game exited before the hold'
finally:
    released = False
    try:
        if os.path.exists(a.flag) and open(a.flag, encoding='utf-8', errors='replace').read().startswith(a.owner):
            os.remove(a.flag)
            released = True
    finally:
        state.update(flag_released=released, released_unix=time.time())
        save()
