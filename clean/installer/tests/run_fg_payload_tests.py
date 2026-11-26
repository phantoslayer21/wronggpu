#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Frame generation's files in the payload step (tools/make_payload.py --fg, wg_core/fg.h): the release stage folder's layout
(plugins\\WrongGPU-FG.asi, fg-dxil\\<name>.dxil and the kernel build's manifest.json) and the same kernels under wronggpu-setup\\
or plugins\\wronggpu-setup\\ embed as plugins\\WrongGPU-FG.asi and fg-dxil\\<name>.dxil (kFgPayload, resource ids 2001 and up,
never the payload's or the RR files'); an OptiScaler file, an NVIDIA name, a developer runtime, a file of any other name, one
kernel name twice, a folder without the plugin or without kernels each stop the build; without --fg the table is empty.

Synthetic files only (written here), in --work. Written for this project; Python standard library only. Run by ctest
("fg-payload").

    python run_fg_payload_tests.py --work <dir>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mockpe  # noqa: E402

MAKE_PAYLOAD = os.path.join(os.path.dirname(HERE), "tools", "make_payload.py")
passed, failures = 0, []


def check(name, cond, detail=""):
    global passed
    if cond:
        passed += 1
        print("  PASS ", name)
    else:
        failures.append(name)
        print("  FAIL ", name, str(detail)[:1500])


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))


def plugin(product="WrongGPU"):
    return mockpe.make_dll({"ProductName": product, "FileDescription": "frame generation plugin (test stand-in)"}, (0, 7, 0, 0),
                           extra=b"synthetic, never loaded", dll_name="WrongGPU-FG.asi")


def run(work, name, fg=None):
    """make_payload.py on a minimal payload (the FSR front end stand-in) with --fg <fg>: (exit code, output, payload_table.h)."""
    out = os.path.join(work, "out-" + name)
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, MAKE_PAYLOAD, "--payload", os.path.join(work, "payload"), "--map", os.path.join(work, "map.json"),
           "--version", "0.7.0-test", "--out", out] + (["--fg", fg] if fg else [])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    table = ""
    if os.path.isfile(os.path.join(out, "payload_table.h")):
        with open(os.path.join(out, "payload_table.h"), encoding="utf-8") as f:
            table = f.read()
    rc_text = ""
    if os.path.isfile(os.path.join(out, "resources.rc")):
        with open(os.path.join(out, "resources.rc"), encoding="utf-8") as f:
            rc_text = f.read()
    return p.returncode, p.stdout.decode("utf-8", "replace"), table, rc_text


def fg_rows(table):
    m = re.search(r"kFgPayload\[\] = \{(.*?)\};", table, re.S)
    rows = re.findall(r'\{(\d+), L"([^"]*)", L"([0-9a-f]*)", (\d+)ull\}', m.group(1)) if m else []
    count = re.search(r"kFgPayloadCount = (\d+);", table)
    return [(int(i), rel.replace("\\\\", "\\")) for i, rel, _, _ in rows if int(i)], int(count.group(1)) if count else -1


def folder(work, name, files):
    d = os.path.join(work, "fg-" + name)
    shutil.rmtree(d, ignore_errors=True)
    for rel, data in files.items():
        write(os.path.join(d, rel), data)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    shutil.rmtree(work, ignore_errors=True)
    write(os.path.join(work, "payload", "amd_fidelityfx_dx12.dll"), b"MZ synthetic FSR front end stand-in")
    write(os.path.join(work, "map.json"), b'{"synthetic": true}\n')
    kernel = b"DXBC synthetic kernel stand-in (never loaded)\n" * 4
    print("[fg-payload] make_payload.py --fg")

    rc, out, table, rc_text = run(work, "none")
    rows, count = fg_rows(table)
    check("without --fg: kFgPayload holds only its end mark, kFgPayloadCount 0", rc == 0 and count == 0 and rows == [] and "kFgPayload[]" in table, out)

    stage = folder(work, "stage", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/fg_exec_pack.dxil": kernel, "fg-dxil/fg_exec_capture.dxil": kernel + b"b",
                                   "fg-dxil/manifest.json": b'{"format": 1, "kernels": []}\n'})
    rc, out, table, rc_text = run(work, "stage", stage)
    rows, count = fg_rows(table)
    rels = [r for _, r in rows]
    check("the stage folder's layout: plugins\\WrongGPU-FG.asi and fg-dxil\\<name>.dxil embedded, manifest.json read past (not shipped)",
          rc == 0 and count == 3 and rels == ["fg-dxil\\fg_exec_capture.dxil", "fg-dxil\\fg_exec_pack.dxil", "plugins\\WrongGPU-FG.asi"] and
          "manifest" not in table, (rc, out, rels))
    check("their resource ids: 2001 and up (never the payload's 1001+ or the RR files' 1601+), each an RCDATA of its file",
          [i for i, _ in rows] == [2001, 2002, 2003] and all(("%d RCDATA" % i) in rc_text for i in (2001, 2002, 2003)) and "manifest.json" not in rc_text,
          rc_text[-600:])
    check("the payload's own table is unchanged by --fg (the front end at 1001)", '{1001, L"amd_fidelityfx_dx12.dll"' in table, table[:400])

    for name, prefix in (("setup", "wronggpu-setup/fg-dxil/"), ("plugins-setup", "plugins/wronggpu-setup/fg-dxil/")):
        d = folder(work, name, {"plugins/WrongGPU-FG.asi": plugin(), prefix + "k1.dxil": kernel})
        rc, out, table, _ = run(work, name, d)
        rows, count = fg_rows(table)
        check("kernels under %s: embedded as fg-dxil\\k1.dxil" % prefix.replace("/", "\\"),
              rc == 0 and count == 2 and [r for _, r in rows] == ["fg-dxil\\k1.dxil", "plugins\\WrongGPU-FG.asi"], (rc, out, rows))

    refusals = [
        ("twice", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "wronggpu-setup/fg-dxil/k.dxil": kernel}, "are both"),
        ("optiscaler-name", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "plugins/OptiScaler.asi": b"MZ x"}, "OptiScaler"),
        ("optiscaler-product", {"plugins/WrongGPU-FG.asi": plugin("OptiScaler"), "fg-dxil/k.dxil": kernel}, "OptiScaler"),
        ("nvidia-name", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "fg-dxil/nvngx_dlssg.dxil": kernel}, "must not use"),
        ("dlss-name", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "dlss/x.dxil": kernel}, "must not use"),
        ("dev-runtime", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel + b"wg-build dev-diagnostics=1"}, "developer"),
        ("other-file", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "readme.txt": b"hello"}, "is not plugins\\WrongGPU-FG.asi"),
        ("other-asi", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/k.dxil": kernel, "plugins/WrongGPU.asi": plugin()}, "is not plugins\\WrongGPU-FG.asi"),
        ("no-plugin", {"fg-dxil/k.dxil": kernel}, "needs plugins\\WrongGPU-FG.asi"),
        ("no-kernels", {"plugins/WrongGPU-FG.asi": plugin(), "fg-dxil/manifest.json": b"{}"}, "no kernel files"),
    ]
    for name, files, words in refusals:
        d = folder(work, name, files)
        rc, out, table, _ = run(work, name, d)
        check("refused (%s): the build stops, no table written" % name, rc != 0 and words in out and not table, (rc, out[-400:]))

    shutil.rmtree(work, ignore_errors=True)
    print("\n%d checks passed, %d failed" % (passed, len(failures)))
    for f in failures:
        print("  FAILED:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
