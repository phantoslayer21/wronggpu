# SPDX-License-Identifier: GPL-3.0-only
"""Process-level launcher safety regressions, using generated fixture games."""
import ctypes
from ctypes import wintypes
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
    for args in (("get", "--global", "--data="), ("/S", "/HIPCHECK", "/DATA="), ("--data=",)):
        rc = subprocess.run([t.exe, *args], env=t.env, capture_output=True).returncode
        t.check("launcher rejects an explicit empty data root " + " ".join(args), rc == 10, rc)
    work = os.path.join(t.work, "scoped")
    os.makedirs(work, exist_ok=True)
    root, b, exe = m.game(t, "Safety outside")
    before = m.snapshot(root)
    rc = subprocess.run([t.exe, "install", exe, "--force", "--noscan", "--data=" + work],
                        env=t.env, capture_output=True).returncode
    t.check("testing launcher refuses a fixture game outside --data", rc != 0 and m.snapshot(root) == before, rc)
    shutil.rmtree(root)
    outside_package = subprocess.run(
        [t.exe, "install", "--opti=" + opti, "--noscan", "--data=" + work],
        env=t.env, capture_output=True, timeout=30)
    t.check("scoped CLI refuses an OptiScaler package outside the active root", outside_package.returncode == 2,
            outside_package.returncode)

    for name in ("setup.log", "optiscaler", "Programs", "StartMenu", "localappdata", "Documents"):
        target = os.path.join(t.work, "linked-data-fixture", name)
        child = os.path.join(work, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if name == "setup.log":
            write(target, b"untouched fixture log")
            link_cmd = ["cmd.exe", "/d", "/c", "mklink", "/H", child, target]
        else:
            os.makedirs(target, exist_ok=True)
            write(os.path.join(target, "sentinel.txt"), b"untouched fixture directory")
            link_cmd = ["cmd.exe", "/d", "/c", "mklink", "/J", child, target]
        subprocess.run(link_cmd, env=t.env, capture_output=True, check=True, timeout=30)
        try:
            rc = subprocess.run([t.exe, "get", "--global", "--data=" + work], env=t.env,
                                capture_output=True, timeout=30).returncode
            t.check("scoped startup rejects a linked " + name, rc == 10, rc)
        finally:
            if name == "setup.log":
                os.unlink(child)
            else:
                os.rmdir(child)

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
        saved = read(ini)
        create = ctypes.windll.kernel32.CreateFileW
        create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
        create.restype = wintypes.HANDLE
        close = ctypes.windll.kernel32.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        scan = create(ini, 0x80000000, 3, None, 3, 0, None)  # antivirus-style read handle denies delete sharing
        if scan == ctypes.c_void_p(-1).value:
            raise OSError("fixture sharing lock could not be acquired")
        try:
            rc = t.run("set", exe, "fps_overlay=on", "--noscan")
        finally:
            close(scan)
        t.check("an antivirus-style sharing lock leaves OptiScaler.ini and warns on apply",
                rc == 0 and read(ini) == saved and "couldn't be updated" in str(t.j), (rc, t.j, t.logtail()))
        game_log = os.path.join(b, "wronggpu-setup", "setup.log")
        write(game_log, b"stale fixture log")
        for verb in ("update", "repair"):
            old = (read(ini).replace(b"Dxgi=auto", b"Dxgi=true")
                   .replace(b"LogLevel=2", b"LogLevel=0").replace(b"FGOutput=auto", b"FGOutput=dlssg") +
                   ("\r\n; player's " + verb + " fixture edit\r\n").encode())
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
                    b"FGOutput=dlssg" in read(ini) and read(backup) == old and not os.path.exists(ini + ".race-test"), (rc, t.j, t.logtail()))
            if rc == 0:
                t.check(verb + " replaces the stale game setup.log", b"stale fixture log" not in read(game_log), game_log)
            if rc:
                break
        if rc == 0:
            change = t.run("set", exe, "opti.Menu.ShortcutKey=0x24", "--noscan")
            repair = t.run("repair", exe, *args)
            reset = t.run("opti", exe, "--reset", "--noscan")
            t.check("opti --reset can still restore the initial value after repair",
                    change == repair == reset == 0 and b"ShortcutKey=auto" in read(ini),
                    (change, repair, reset, t.j, t.logtail()))
        t.run("uninstall", exe, "--noscan")
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety post-install edit")
    edited = os.path.join(b, "OptiScaler.ini")
    hooks = {
        "WRONGGPU_TEST_RENAME_AT": "installed", "WRONGGPU_TEST_RENAME_FILE": edited,
        "WRONGGPU_TEST_CREATE_AT": "installed", "WRONGGPU_TEST_CREATE_FILE": edited,
    }
    t.env.update(hooks)
    try:
        rc = t.run("install", exe, *args)
    finally:
        for key in hooks:
            t.env.pop(key)
    t.check("a post-install INI edit skips profile write without failing the install",
            rc == 0 and b"made by the player" in read(edited) and "changed after installation" in str(t.j),
            (rc, t.j, t.logtail()))
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety keep new restorer")
    candidate = os.path.join(b, "wronggpu-setup", "uninstall.exe")
    t.env["WRONGGPU_TEST_CREATE_AT"] = "seed-recorded"
    t.env["WRONGGPU_TEST_CREATE_FILE"] = candidate
    try:
        rc = t.run("install", exe, *args)
    finally:
        t.env.pop("WRONGGPU_TEST_CREATE_AT")
        t.env.pop("WRONGGPU_TEST_CREATE_FILE")
    t.check("a restorer path created first by another program survives rollback",
            rc != 0 and os.path.isfile(candidate) and b"made by the player" in read(candidate), (rc, t.j, t.logtail()))
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety known restorer")
    prior = os.path.join(b, "wronggpu-setup", "uninstall.exe")
    old_restorer = m.mockpe.make_dll({"ProductName": "WrongGPU", "OriginalFilename": "WrongGPU-Restore.exe"})
    write(prior, old_restorer)
    rc = t.run("install", exe, *args)
    t.check("a recognized restorer left without an install record is replaced",
            rc == 0 and read(prior) != old_restorer, (rc, t.j, t.logtail()))
    if rc == 0:
        result = subprocess.run([prior, "/S", "/DATA="], env=t.env, capture_output=True, timeout=30)
        t.check("the restorer refuses an explicit empty data root", result.returncode == 10, result.returncode)
        t.run("uninstall", exe, "--noscan")
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety hard link")
    linked = os.path.join(b, "wronggpu-notices.txt")
    other_name = os.path.join(b, "fixture-player-notes.txt")
    write(other_name, b"player's shared file")
    os.link(other_name, linked)
    rc = t.run("install", exe, *args)
    t.check("an existing hard-linked target is kept and the message identifies the link",
            rc != 0 and "hard link" in str(t.j) and read(linked) == read(other_name) == b"player's shared file",
            (rc, t.j, t.logtail()))
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety staged write")
    ini = os.path.join(b, "OptiScaler.ini")
    preinstall = b"; before install\r\n"
    write(ini, preinstall)
    t.env["WRONGGPU_TEST_FAULT"] = "install-text-write"
    try:
        rc = t.run("install", exe, *args, "--switch")
    finally:
        t.env.pop("WRONGGPU_TEST_FAULT")
    t.check("a failed staged text write leaves the original intact",
            rc != 0 and read(ini) == preinstall and "rolling back" in t.logtail(), (rc, t.j, t.logtail()))
    shutil.rmtree(root)

    root, b, exe = m.game(t, "Safety edition merge")
    v5 = os.path.join(t.work, "edition-fixture-v5")
    v6 = os.path.join(t.work, "edition-fixture-v6")
    for folder, dll, defaults in (
        (v5, b"MZ OptiScaler fixture v5", "OLD\r\nRemoveOnUpdate=legacy"),
        (v6, b"MZ OptiScaler fixture v6", "NEW\r\nAddOnUpdate=current"),
    ):
        shutil.copytree(opti, folder)
        write(os.path.join(folder, "OptiScaler.dll"), dll)
        cfg = m.OPTI_INI.replace("\n", "\r\n") + (
            "[Spoofing]\r\nDxgi=auto\r\n[FrameGen]\r\nFGOutput=auto\r\nFGType=auto\r\n"
            "[Package]\r\nDefault=" + defaults + "\r\n")
        write(os.path.join(folder, "OptiScaler.ini"), cfg)
    install = t.run("install", exe, "--opti=" + v5, "--noscan", "--force", "--experimental")
    ini = os.path.join(b, "OptiScaler.ini")
    if install == 0:
        write(ini, read(ini).replace(b"Dxgi=auto", b"Dxgi=true")
              .replace(b"LogLevel=2", b"LogLevel=0").replace(b"FGOutput=auto", b"FGOutput=dlssg"))
        changed = t.run("update", exe, "--opti=" + v6, "--noscan", "--force", "--experimental")
        merged = read(ini)
        t.check("v5 to v6 uses new package defaults and replays the player's edits",
                changed == 0 and all(v in merged for v in (
                    b"Default=NEW", b"AddOnUpdate=current", b"Dxgi=true", b"LogLevel=0", b"FGOutput=dlssg"))
                and b"RemoveOnUpdate" not in merged, (changed, t.j, merged))
        t.run("uninstall", exe, "--noscan")
    else:
        t.check("install v5 fixture for edition merge", False, (install, t.j))
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
        root, b, exe = m.game(t, "Safety reopen rollback")
        before_publish = m.snapshot(root)
        t.env["WRONGGPU_TEST_FAULT"] = "install-binary-reopen"
        try:
            failed = t.raw([t.exe, "/S", "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/DIRECTFSR",
                            "/DIR=" + root, "/LOG=" + t.log])
        finally:
            t.env.pop("WRONGGPU_TEST_FAULT")
        t.check("reopening failure after binary publication restores the game byte for byte",
                failed != 0 and m.snapshot(root) == before_publish and "rolling back" in t.logtail(),
                (failed, m.diff(before_publish, m.snapshot(root)), t.logtail()))
        shutil.rmtree(root)

        root, b, exe = m.game(t, "Safety partial rename")
        original = os.path.join(b, "amd_fidelityfx_dx12.dll")
        renamed = os.path.join(b, "amd_fidelityfx_dx12.wronggpu-original.dll")
        write(renamed, b"existing rename destination")
        before_rename = read(original)
        t.env["WRONGGPU_TEST_FAULT"] = "install-rename"
        try:
            failed = t.raw([t.exe, "/S", "/FORCE", "/NOSCAN", "/EXPERIMENTAL", "/DIRECTFSR",
                            "/DIR=" + root, "/LOG=" + t.log])
        finally:
            t.env.pop("WRONGGPU_TEST_FAULT")
        t.check("if the move fails after removing a destination, rollback restores both originals",
                failed != 0 and read(original) == before_rename and
                read(renamed) == b"existing rename destination", (failed, t.logtail()))
        shutil.rmtree(root)

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
        t.check("an incompatible cache preserves active RR settings and reports the pending mode",
                rc == 0 and m.state_kv(rr_dir).get("rr_enabled") == "1" and
                m.state_kv(rr_dir).get("rr_mode") != "exact" and
                m.state_kv(rr_dir).get("rr_mode_pending") == "exact" and "pending" in str(t.j), (rc, before, t.j, t.logtail()))
        os.remove(rr_data)
        t.run("uninstall", rr_exe, "--noscan")
    shutil.rmtree(rr_root)

    root, b, exe = m.game(t, "Safety RR config path", exe="Borderlands4.exe")
    outside = os.path.join(os.path.dirname(t.work), os.path.basename(t.work) + "-outside", "Engine.ini")
    foreign = b"[SystemSettings]\r\nr.NGX.DLSS.DenoiserMode=1\r\n"
    write(outside, foreign)
    installed = t.run("install", exe, *args)
    if installed == 0:
        record = os.path.join(b, "wronggpu-setup", "install.record")
        with open(record, "ab") as f:
            f.write(("rrconfig\t" + outside + "\t-\tabsent\t-\r\n").encode())
        rc = t.run("set", exe, "rr_config_fix=off", "--noscan")
        t.check("a stored RR config path outside the fixture is refused and untouched",
                rc != 0 and read(outside) == foreign, (rc, t.j, t.logtail()))
    else:
        t.check("install fixture for recorded RR config path", False, (installed, t.j))
    shutil.rmtree(root)
    os.unlink(outside)
    os.rmdir(os.path.dirname(outside))
