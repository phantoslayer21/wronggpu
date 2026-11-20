# SPDX-License-Identifier: MIT
"""PSO-creation sweep over a folder of built D3D12 kernels (<dir>/manifest.json + the DXIL files): creates each kernel's
pipeline with the runtime's root signature on the card (pso_sweep.exe under Wine, wine_headless_run.sh) and prints a table.
Each kernel is also classified from its DXIL disassembly (dxc -dumpbin): does it contain an FP8 matrix op or an FP8
conversion of the driver channel (code words of wg_ags.hlsli)? On RDNA 3 a PSO with one is expected to fail, any other to
succeed; a surprise returns 1.
    python3 pso_sweep.py --dxil-dir <build>/dx12 --exe pso_sweep.exe --run wine_headless_run.sh []
Environment: WG_DXC_DIR (dxc.exe, as build_dx12.py), plus whatever --run needs."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import build_dx12 as B  # noqa: E402

# phase 1 code words 0x50000000 | 1 << 24 | imm << 8 | op: FP8 matrix op (op 0x28, format 0x7), FP8 -> F32 convert (op 0x32,
# source type 0x1a), F32 -> FP8 convert (op 0x32, source type 0x26, the destination type 0x1a is the data value)
FP8_MMA = 0x50000000 | 1 << 24 | 0x7 << 8 | 0x28
FP8_UNPACK = 0x50000000 | 1 << 24 | 0x1A << 8 | 0x32
CONVERT_F32 = 0x50000000 | 1 << 24 | 0x26 << 8 | 0x32


def has_fp8(dxil):
    r = subprocess.run([os.path.join(B.DXC_DIR, "dxc.exe"), "-dumpbin", dxil], capture_output=True, text=True)
    text = r.stdout
    consts = lambda v: re.search(r"(?<![\w.-])%d(?!\d)" % v, text) is not None
    if consts(FP8_MMA) or consts(FP8_UNPACK):
        return True
    # F32 -> F16 converts use the same code word with data 0x24; an FP8 target has the data value 26 next to it
    return consts(CONVERT_F32) and re.search(r"i32 %d, i32 26," % CONVERT_F32, text) is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dxil-dir", required=True)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--run", required=True, help="the headless Wine runner script")
    ap.add_argument("--work", help="folder for the list file (default: next to the DXIL folder)")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    with open(os.path.join(a.dxil_dir, "manifest.json")) as f:
        kernels = json.load(f)["kernels"]
    work = a.work or os.path.dirname(os.path.abspath(a.dxil_dir))
    lst = os.path.join(work, "pso_sweep.list")
    got = {}
    todo = [k for k in kernels if k.get("rs", "buffer") != "tex"]   # the runtime has no root signature of the tex kind
    log = ""
    while todo:   # a driver crash ends the process: the kernel being created is recorded as CRASH, the rest run again
        with open(lst, "w") as f:
            for k in todo:
                f.write(f"{k['file']} {k.get('rs', 'buffer')} {os.path.abspath(os.path.join(a.dxil_dir, k['file']))}\n")
        r = subprocess.run([a.run, "--", os.path.abspath(a.exe), lst], capture_output=True, text=True, timeout=a.timeout)
        log = r.stdout + r.stderr
        for line in r.stdout.splitlines():
            m = re.match(r"PSO (\S+) (\S+) hr=(0x[0-9a-f]+) (OK|FAILED)", line)
            if m:
                got[m.group(1)] = (m.group(3), m.group(4) == "OK")
        todo = [k for k in todo if k["file"] not in got]
        if "SWEEP DONE" in r.stdout:
            break
        if not todo or "PSO " not in r.stdout and "D3D12CreateDevice" in log:
            print(log[-600:])
            return 2
        got[todo[0]["file"]] = ("CRASH", False)   # the first kernel without a line
        todo = todo[1:]
    bad = 0
    print(f"{'kernel':44s} {'rs':7s} {'fp8':4s} {'hr':12s} result")
    for k in kernels:
        fp8 = has_fp8(os.path.join(a.dxil_dir, k["file"]))
        if k.get("rs") == "tex":
            print(f"{k['name']:44s} tex     {'yes' if fp8 else 'no':4s} {'-':12s} not tried (no root signature of this kind in the runtime)")
            continue
        hr, ok = got.get(k["file"], ("-", False))
        surprise = ok == fp8 and hr != "CRASH"
        bad += surprise
        print(f"{k['name']:44s} {k.get('rs', 'buffer'):7s} {'yes' if fp8 else 'no':4s} {hr:12s} {'OK' if ok else ('CRASH' if hr == 'CRASH' else 'FAILED')}"
              f"{'   <-- unexpected' if surprise else ''}")
    n_ok = sum(1 for v in got.values() if v[1])
    n_crash = sum(1 for v in got.values() if v[0] == "CRASH")
    print(f"{len(kernels)} kernels: {n_ok} created, {len(got) - n_ok - n_crash} failed, {n_crash} crashed the driver, "
          f"{len(kernels) - len(got)} not tried, {bad} unexpected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
