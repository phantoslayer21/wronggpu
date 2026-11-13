# SPDX-License-Identifier: MIT
"""RR guard: Model M is not affected by WrongGPU RR's coexistence guard (CPU only).

    python clean/rr_f/proxy/tools/guard_modelm.py --fsr-build <Model M ctest build dir> --rr-out <build.py --out dir>
                                                  [--exclude REGEX] [--config Release]

Runs every test of Model M's own ctest list (clean/fsr: WrongGPU.asi, the front end, the custom slot, the shared
settings), unchanged, once on its own and once per guard mode with WrongGPU RR loaded into the test process before
its main and the guard tripped (tests/guard_inject.cpp starts the program suspended and loads tests/guard_scene.cpp):
none (control: WrongGPU RR loaded, nothing tripped), off, take, owncopy, passthrough. Each guard mode must end exactly
as the control did (passed / skipped / failed), and the scene must have seen WrongGPU RR's line for that mode. A test
whose control differs from its plain run is listed as a note: that difference comes from WrongGPU RR being in the
process at all (e.g. a unit test that must create the process's shared settings block itself), not from the guard.
The stand-ins come from
build.py's guard stage of the 2.14 table (<rr-out>/2.14/hook-test/guard/stage). wronggpu-rr.ini (take) and
wronggpu-rr.log are written next to each test program and removed afterwards (only files this run created). Tests that
use a hardware GPU are excluded by default (fsr_overlay, fsr_dll_d3d12_*).
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build  # noqa: E402

MODES = ("none", "off", "take", "owncopy", "passthrough")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsr-build", required=True)
    ap.add_argument("--rr-out", default=build.DEFAULT_OUT)
    ap.add_argument("--config", default="Release")
    ap.add_argument("--exclude", default=r"^(fsr_overlay|fsr_dll_d3d12.*)$")
    a = ap.parse_args()
    hook_test = os.path.join(a.rr_out, "2.14", "hook-test")
    stage = os.path.join(hook_test, "guard", "stage")
    gen = os.path.join(hook_test, "gen")
    if not os.path.isfile(os.path.join(stage, "wgtest_other_rr.dll")):
        raise SystemExit(f"no guard stage in {stage}: run build.py first")
    tools = os.path.join(a.rr_out, "guard-modelm")
    os.makedirs(tools, exist_ok=True)
    scene = os.path.join(tools, "guard_scene.dll")
    inject = os.path.join(tools, "guard_inject.exe")
    build.sh(f'cl {build.CLFLAGS} /LD /I"{gen}" {build.src("tests", "guard_scene.cpp")} /Fo"{tools}\\\\" /Fe"{scene}"',
             tools)
    build.sh(f'cl {build.CLFLAGS} {build.src("tests", "guard_inject.cpp")} /Fo"{tools}\\\\" /Fe"{inject}"', tools)
    listing = build.sh(f'ctest --test-dir "{a.fsr_build}" -C {a.config} --show-only=json-v1', tools)
    tests = json.loads(listing[listing.index("{"):])["tests"]
    exclude = re.compile(a.exclude)
    rows, failures, presence = [], 0, []
    for t in tests:
        props = {p["name"]: p["value"] for p in t.get("properties", [])}
        if exclude.search(t["name"]) or props.get("DISABLED") or "command" not in t:
            continue
        cmd, wd = t["command"], props.get("WORKING_DIRECTORY", os.path.dirname(t["command"][0]))
        skip = int(props.get("SKIP_RETURN_CODE", -1)) if "SKIP_RETURN_CODE" in props else None
        exe_dir = os.path.dirname(cmd[0])
        ini, log = os.path.join(exe_dir, "wronggpu-rr.ini"), os.path.join(exe_dir, "wronggpu-rr.log")
        if os.path.exists(ini) or os.path.exists(log):
            raise SystemExit(f"{exe_dir} already has wronggpu-rr.ini / wronggpu-rr.log: not touched, stopping")

        def outcome(rc):
            return "skipped" if skip is not None and rc == skip else "passed" if rc == 0 else f"failed ({rc})"

        plain = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=600)
        row = [t["name"], outcome(plain.returncode)]
        control = None  # the "none" run: WrongGPU RR in the process, the guard not tripped
        for mode in MODES:
            if mode == "take":
                with open(ini, "w") as f:
                    f.write("rr_other_plugin = take\n")
            try:
                r = subprocess.run([inject, scene, stage, mode] + cmd, cwd=wd, capture_output=True, text=True, timeout=600)
            finally:
                if os.path.exists(ini):
                    os.remove(ini)
            got = outcome(r.returncode)
            seen = f"guard_inject: ok {mode}" in r.stdout
            if mode == "none":
                control = got
                if got != row[1]:  # WrongGPU RR's presence itself, before any guard decision
                    presence.append((t["name"], row[1], got, (r.stdout + r.stderr).strip().splitlines()[-3:]))
            # a guard mode must end exactly as the control does (the guard changes nothing for Model M)
            same = got == control and seen
            failures += not same
            row.append(got if same else f"{got}{'' if seen else ', scene: ' + r.stderr.strip()[-200:]} <-- DIFFERS")
        if os.path.exists(log):
            os.remove(log)
        rows.append(row)
    print("test".ljust(34) + "plain".ljust(12) + "".join(m.ljust(14) for m in MODES))
    for row in rows:
        print(row[0].ljust(34) + row[1].ljust(12) + "".join(c.ljust(14) for c in row[2:]))
    print(f"{len(rows)} Model M test(s) x {len(MODES) - 1} guard mode(s) against the control: {failures} difference(s)")
    for name, p, c, tail in presence:
        print(f"note: {name}: {p} alone, {c} with WrongGPU RR in the process before any guard decision (control); "
              f"the guard modes match the control. Last lines: {' | '.join(tail)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
