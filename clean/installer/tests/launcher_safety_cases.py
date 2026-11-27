# SPDX-License-Identifier: GPL-3.0-only
"""Process-level launcher safety regressions, using generated fixture games."""
import os
import shutil
import subprocess

from run_installer_tests import read, write


def run(t, opti, synth_f, m):
    print("\n[launcher safety] scope, settings merge, and installation races")
    real_like = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU")
    sentinel = os.path.join(real_like, "managed-games.txt")
    write(sentinel, "scratch profile sentinel\r\n")
    rc = subprocess.run([t.exe, "get", "--global"], env=t.env, capture_output=True).returncode
    t.check("testing launcher refuses startup without --data", rc == 10, rc)
    work = os.path.join(t.work, "scoped")
    os.makedirs(work, exist_ok=True)
    root, b, exe = m.game(t, "Safety outside")
    before = m.snapshot(root)
    rc = subprocess.run([t.exe, "install", exe, "--force", "--noscan", "--data=" + work],
                        env=t.env, capture_output=True).returncode
    t.check("testing launcher refuses a fixture game outside --data", rc != 0 and m.snapshot(root) == before, rc)
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety merge and backup")
    merge_package = os.path.join(t.work, "safety-opti-package")
    shutil.copytree(opti, merge_package)
    package_ini = os.path.join(merge_package, "OptiScaler.ini")
    with open(package_ini, "ab") as f:
        f.write(b"\n[Spoofing]\nDxgi=auto\n[FrameGen]\nFGOutput=auto\n")
    args = ("--opti=" + merge_package, "--noscan", "--force", "--experimental")
    rc = t.run("install", exe, *args)
    cache = os.path.join(t.data, "model", "model.wgcm")
    t.check("--data holds the model and managed games list",
            rc == 0 and os.path.isfile(cache) and b.lower() in m.managed_list(t).lower() and
            read(sentinel) == b"scratch profile sentinel\r\n" and not os.path.exists(os.path.join(real_like, "model")),
            (rc, t.j, t.logtail()))
    if rc == 0:
        ini = os.path.join(b, "OptiScaler.ini")
        for verb in ("update", "repair"):
            old = (read(ini).replace(b"Dxgi=auto", b"Dxgi=true")
                   .replace(b"LogLevel=2", b"LogLevel=0").replace(b"FGOutput=auto", b"FGOutput=dlssg"))
            write(ini, old)
            t.env["WRONGGPU_TEST_RENAME_AT"] = "seed-recorded"
            t.env["WRONGGPU_TEST_RENAME_FILE"] = ini
            try:
                rc = t.run(verb, exe, *args)
            finally:
                t.env.pop("WRONGGPU_TEST_RENAME_AT")
                t.env.pop("WRONGGPU_TEST_RENAME_FILE")
            backup = os.path.join(b, "wronggpu-setup", "backup", "OptiScaler.ini")
            t.check(verb + " keeps edited INI values and backs up its pinned original",
                    rc == 0 and b"Dxgi=true" in read(ini) and b"LogLevel=0" in read(ini) and
                    b"FGOutput=auto" in read(ini) and read(backup) == old and not os.path.exists(ini + ".race-test"), (rc, t.j, t.logtail()))
            if rc:
                break
        t.run("uninstall", exe, "--noscan")
    shutil.rmtree(root)

    for verb in ("install", "repair", "update"):
        root, b, exe = m.game(t, "Safety new target " + verb)
        if verb != "install":
            t.run("install", exe, *args)
        candidate = os.path.join(b, "wronggpu-notices.txt")
        t.env["WRONGGPU_TEST_CREATE_AT"] = "seed-recorded"
        t.env["WRONGGPU_TEST_CREATE_FILE"] = candidate
        try:
            rc = t.run(verb, exe, *args)
        finally:
            t.env.pop("WRONGGPU_TEST_CREATE_AT")
            t.env.pop("WRONGGPU_TEST_CREATE_FILE")
        t.check(verb + " refuses a destination created after the backup and keeps its bytes",
                rc != 0 and os.path.isfile(candidate) and b"made by the player" in read(candidate), (rc, t.j, t.logtail()))
        shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety direct rename")
    original = os.path.join(b, "amd_fidelityfx_dx12.dll")
    renamed = os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")
    before = read(original)
    t.env["WRONGGPU_TEST_CREATE_AT"] = "seed-recorded"
    t.env["WRONGGPU_TEST_CREATE_FILE"] = renamed
    try:
        rc = t.raw([t.exe, "/S", "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/DIRECTFSR",
                    "/DIR=" + root, "/LOG=" + t.log])
    finally:
        t.env.pop("WRONGGPU_TEST_CREATE_AT")
        t.env.pop("WRONGGPU_TEST_CREATE_FILE")
    if rc != 10:  # the no-direct build rejects /DIRECTFSR
        t.check("direct install cannot rename over a destination that appeared after backup",
                rc != 0 and read(original) == before and os.path.isfile(renamed) and
                b"made by the player" in read(renamed), (rc, t.logtail()))
    shutil.rmtree(root)

    if rc != 10:
        root, b, exe = m.game(t, "Safety restore rename")
        renamed = os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")
        direct = [t.exe, "/S", "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/DIRECTFSR", "/DIR=" + root, "/LOG=" + t.log]
        first = t.raw(direct)
        t.env["WRONGGPU_TEST_RENAME_AT"] = "rename-restore-locked"
        t.env["WRONGGPU_TEST_RENAME_FILE"] = renamed
        try:
            repaired = t.raw(direct)
        finally:
            t.env.pop("WRONGGPU_TEST_RENAME_AT")
            t.env.pop("WRONGGPU_TEST_RENAME_FILE")
        t.check("repair pins the prior FFX original through restoration",
                first == 0 and repaired == 0 and read(renamed) == before and not os.path.exists(renamed + ".race-test"),
                (first, repaired, t.logtail()))
        t.run("uninstall", exe, "--noscan")
        shutil.rmtree(root)

    rr_root, rr_dir, rr_exe = m.game(t, "Safety RR incompatible", extra={
        "bin/x64/nvngx_dlssd.dll": b"MZ fixture RR library",
        "bin/x64/sl.dlss_d.dll": b"MZ fixture RR interface",
    })
    rr_data = os.path.join(t.data, "model", "F", "model.wgcm")
    rr_state = os.path.join(rr_dir, "wronggpu-setup", "settings.ini")
    if t.run("install", rr_exe, *args) == 0:
        write(rr_data, b"another map's cache")
        with open(rr_state, "ab") as f:
            f.write(b"rr_enabled=1\r\n")
        before = read(rr_state)
        rc = t.run("set", rr_exe, "rr_mode=exact", "--noscan")
        t.check("an incompatible RR cache never rewrites saved RR keys",
                rc == 0 and m.state_kv(rr_dir).get("rr_enabled") == "1" and
                m.state_kv(rr_dir).get("rr_mode") != "exact", (rc, before, t.j, t.logtail()))
        os.remove(rr_data)
        t.run("uninstall", rr_exe, "--noscan")
    shutil.rmtree(rr_root)
