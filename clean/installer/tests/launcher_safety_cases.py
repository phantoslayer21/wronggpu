# SPDX-License-Identifier: GPL-3.0-only
"""Process-level launcher safety regressions, using generated fixture games."""
import ctypes
import glob
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
        m.state_set(rr_dir, "rr_mode_pending", "off")
        rc = t.run("set", rr_exe, "rr_mode=exact", "--noscan")
        t.check("explicit RR set queues even when the saved profile already has that value",
                rc == 0 and m.state_kv(rr_dir).get("rr_mode_pending") == "exact", (rc, t.j))
        rc = t.run("set", rr_exe, "fps_overlay=on", "--noscan")
        t.check("an unrelated setting leaves the explicit pending RR mode alone",
                rc == 0 and m.state_kv(rr_dir).get("rr_mode_pending") == "exact", (rc, t.j))
        os.unlink(os.path.join(rr_dir, "wronggpu-setup", "profile.ini"))
        rc = t.run("set", rr_exe, "rr_mode=accurate", "--noscan")
        t.check("explicit RR set queues without a saved profile",
                rc == 0 and m.state_kv(rr_dir).get("rr_mode_pending") == "accurate", (rc, t.j))
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
    review_regressions(t, opti, m)



def review_regressions(t, opti, m):
    print("\n[launcher safety] rollback recovery, file metadata, and explicit roots")
    fixtures = []

    def fixture(name, **kwargs):
        root, folder, exe = m.game(t, name, **kwargs)
        fixtures.append(root)
        return root, folder, exe

    package = os.path.join(t.work, "round-two-opti")
    edition = os.path.join(t.work, "round-two-edition")
    for folder, version, default in ((package, b"5", b"auto"), (edition, b"6", b"0x1234")):
        shutil.copytree(opti, folder)
        write(os.path.join(folder, "OptiScaler.dll"), b"MZ fixture edition " + version)
        write(os.path.join(folder, "OptiScaler.ini"), m.OPTI_INI.encode() +
              b"[Plugins]\nPath=auto\n[Spoofing]\nDxgi=auto\nSpoofedDeviceId=" + default + b"\n")
        write(os.path.join(folder, "OptiScaler", "companion.dll"), b"fixture companion binary")
    flags = ("--noscan", "--force", "--experimental")
    args = ("--opti=" + package, *flags)
    def settings_path(folder):
        return os.path.join(folder, "OptiScaler.ini")

    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    def short_path(path):
        buf = ctypes.create_unicode_buffer(2048)
        length = get_short(path, buf, len(buf))
        return buf.value if 0 < length < len(buf) else path
    if short_path(t.work).lower() != t.work.lower():
        for short_root in (True, False):
            env = dict(t.env)
            if short_root:
                env["WRONGGPU_TEST_FIXTURE_ROOT"] = short_path(t.work)
                data = short_path(t.data)
            else:
                env["LOCALAPPDATA"] = short_path(env["LOCALAPPDATA"])
                env["APPDATA"] = short_path(env["APPDATA"])
                data = t.data
            rc = subprocess.run([t.exe, "get", "--global", "--data=" + data], env=env,
                                capture_output=True, timeout=30).returncode
            t.check("fixture startup accepts short names for " + ("fixture and data" if short_root else "profile paths"),
                    rc == 0, rc)

    path_package = m.opti_package(t, "round-two-path-package")
    with open(os.path.join(path_package, "OptiScaler.ini"), "ab") as f:
        f.write(b"[Plugins]\nPath=auto\n")
    asi_loader = m.mockpe.make_dll({"ProductName": "Ultimate ASI Loader", "CompanyName": "ThirteenAG"})
    for suffix in (" #2", " ; notes"):
        _, b, exe = fixture("Safety special path" + suffix, extra={"bin/x64/dinput8.dll": asi_loader})
        expected = {b"Path": os.path.join(b, "wronggpu-plugins").encode(),
                    b"FfxDx12SRPath": os.path.join(b, "wronggpu_ffx_dx12.dll").encode()}
        for verb in ("install", "repair", "update"):
            if verb != "install" and os.path.isfile(settings_path(b)):
                with open(settings_path(b), "ab") as f:
                    f.write(b"\n; player-edited file to merge\n")
            rc = t.run(verb, exe, "--opti=" + path_package, *flags)
            lines = dict(line.split(b"=", 1) for line in read(settings_path(b)).splitlines() if b"=" in line) if rc == 0 else {}
            t.check(verb + " keeps complete INI paths with" + suffix,
                    rc == 0 and all(lines.get(key) == value for key, value in expected.items()), (rc, lines, t.j))
        t.run("uninstall", exe, "--noscan")

    _, b, exe = fixture("Safety library duplicate")
    _, _, other = fixture("Safety separate install")
    assert t.run("install", exe, *args) == 0, t.j
    profile = os.path.join(b, "wronggpu-setup", "profile.ini")
    os.unlink(profile)
    cfg = settings_path(b)
    original = read(cfg)
    write(cfg, original + b"\n[Menu]\nShowFps=false\n")
    rc = t.run("scan", "--noscan")
    listed = any(x.get("exe", "").lower() == exe.lower() and "duplicate" in x.get("ini_warning", "")
                 for x in t.j.get("games", []))
    t.check("duplicate INI keeps a managed game in the Library with a warning",
            rc == 0 and listed and "duplicate" in t.logtail(10000).lower(), (rc, t.j, t.logtail()))
    rc = t.run("opti", exe, "--noscan")
    t.check("the OptiScaler tab reports the same read warning", rc == 0 and "duplicate" in t.j.get("ini_warning", ""), (rc, t.j))
    rc = t.run("install", other, *args)
    t.check("the duplicate INI does not block another install", rc == 0, (rc, t.j, t.logtail()))
    write(cfg, original)
    t.run("uninstall", exe, "--noscan")
    t.run("uninstall", other, "--noscan")

    _, b, exe = fixture("Safety opt-in journal", exe="Spider-Man2.exe")
    cfg = settings_path(b)
    assert t.run("install", exe, *args) == 0, t.j
    assert t.run("set", exe, "sm2_rr_experimental=on", "--noscan") == 0, t.j
    def opt_in_lines():
        return [line for line in m.record_text(b).splitlines() if "\tSpoofedDeviceId\t" in line and line.startswith("inikey\t")]
    for i in range(5):
        rc = t.run("repair", exe, *args)
        t.check("repair %d leaves a single opt-in journal entry" % (i + 1),
                rc == 0 and len(opt_in_lines()) == 1, (rc, t.j, opt_in_lines()))
    rc = t.run("update", exe, "--opti=" + edition, *flags)
    t.check("edition update rebases the opt-in entry once",
            rc == 0 and len(opt_in_lines()) == 1 and "\t=0x1234\t=0x2684" in opt_in_lines()[0],
            (rc, t.j, opt_in_lines()))
    for switch, expected in (("off", b"0x1234"), ("on", b"0x2684"), ("off", b"0x1234")):
        rc = t.run("set", exe, "sm2_rr_experimental=" + switch, "--noscan")
        t.check("opt-in %s restores the correct edition's value" % switch,
                rc == 0 and b"SpoofedDeviceId=" + expected in read(cfg) and len(opt_in_lines()) == (switch == "on"),
                (rc, t.j, opt_in_lines()))
    assert t.run("set", exe, "opti.Menu.ShortcutKey=0x24", "--noscan") == 0, t.j
    write(cfg, read(cfg).replace(b"ShortcutKey=0x24", b"ShortcutKey=auto"))
    rc = t.run("repair", exe, "--opti=" + edition, *flags)
    t.check("merge drops journal entries whose written value equals their original",
            rc == 0 and b"ShortcutKey=auto" in read(cfg) and
            not any(line.startswith("inikey\t") and "\tShortcutKey\t" in line for line in m.record_text(b).splitlines()),
            (rc, t.j, m.record_text(b)))
    t.run("uninstall", exe, "--noscan")

    _, b, exe = fixture("Safety protected metadata")
    cfg = settings_path(b)
    content = m.OPTI_INI.encode()
    write(cfg, content)
    write(cfg + ":fixture", b"named stream")
    os.utime(cfg, (1590000000, 1590000000))
    timestamp = os.stat(cfg).st_mtime_ns
    attrs = 0x1 | 0x2 | 0x4  # read-only, hidden, system
    set_attrs = ctypes.windll.kernel32.SetFileAttributesW
    get_attrs = ctypes.windll.kernel32.GetFileAttributesW
    set_attrs.argtypes = (wintypes.LPCWSTR, wintypes.DWORD)
    get_attrs.argtypes = (wintypes.LPCWSTR,)
    def acl():
        result = subprocess.run(["icacls.exe", cfg], env=t.env, capture_output=True, check=True)
        return result.stdout
    subprocess.run(["icacls.exe", cfg, "/inheritance:d"], env=t.env, capture_output=True, check=True)
    explicit_acl = acl()
    assert set_attrs(cfg, attrs)
    def metadata_ok():
        return (os.path.isfile(cfg) and get_attrs(cfg) & attrs == attrs and read(cfg + ":fixture") == b"named stream"
                and acl() == explicit_acl)
    for verb, more in (("install", (*args, "--switch")),
                      ("set", ("fps_overlay=on", "--noscan")),
                      ("set", ("opti.Menu.ShortcutKey=0x24", "--noscan")),
                      ("update", ("--opti=" + edition, *flags))):
        rc = t.run(verb, exe, *more)
        t.check(verb + " preserves protected INI attributes and its named stream",
                rc == 0 and metadata_ok(), (rc, t.j, t.logtail()))
    rc = t.run("uninstall", exe, "--noscan")
    t.check("uninstall restores original INI bytes, attributes, stream and last-write time",
            rc == 0 and metadata_ok() and read(cfg) == content and os.stat(cfg).st_mtime_ns == timestamp,
            (rc, t.j, t.logtail()))
    if os.path.isfile(cfg):
        set_attrs(cfg, 0x80)  # the metadata assertions above used the restored attributes

    for guarded in (True, False):
        label = "guarded" if guarded else "unguarded"
        _, b, exe = fixture("Safety staged binary restore " + label)
        companion = os.path.join(b, "OptiScaler", "companion.dll")
        player_binary = b"fixture player's original binary"
        write(companion, player_binary)
        os.utime(companion, (1590000000, 1590000000))
        timestamp = os.stat(companion).st_mtime_ns
        assert t.run("install", exe, *args) == 0, t.j
        if not guarded:
            record = os.path.join(b, "wronggpu-setup", "install.record")
            contents = read(record)
            normal = b"owner\tcopy\tOptiScaler\\companion.dll\twg-optibundle"
            assert normal in contents
            write(record, contents.replace(normal, b"owner\tcopy\tOptiScaler\\companion.dll\tfixture"))
        installed = read(companion)
        t.env["WRONGGPU_TEST_FAULT"] = "restore-binary-write"
        try:
            rc = t.run("uninstall", exe, "--noscan")
        finally:
            del t.env["WRONGGPU_TEST_FAULT"]
        t.check("mid-write failure leaves the " + label + " binary whole and its original restorable",
                rc != 0 and read(companion) == installed and os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")),
                (rc, t.j, t.logtail()))
        rc = t.run("uninstall", exe, "--noscan")
        t.check(label + " binary restore preserves the original timestamp and logs the restore on retry",
                rc == 0 and read(companion) == player_binary and os.stat(companion).st_mtime_ns == timestamp and
                "restored OptiScaler\\companion.dll" in t.logtail(10000), (rc, t.j, t.logtail()))

    _, b, exe = fixture("Safety rollback sharing lock")
    companion = os.path.join(b, "OptiScaler", "companion.dll")
    source = os.path.join(package, "OptiScaler", "companion.dll")
    hooks = {
        "WRONGGPU_TEST_FAULT": "install-binary-reopen",
        "WRONGGPU_TEST_LOCK_ROLLBACK": "1",
        "WRONGGPU_TEST_CREATE_AT": "seed-recorded",
        "WRONGGPU_TEST_CREATE_FILE": companion,
        "WRONGGPU_TEST_CREATE_FROM": source,
    }
    t.env.update(hooks)
    try:
        rc = t.run("install", exe, *args)
    finally:
        for key in hooks:
            del t.env[key]
    record = m.record_text(b) if os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")) else ""
    t.check("a sharing lock leaves a pruned recovery record with publication identities",
            rc != 0 and "Cannot undo the creation" in t.logtail(10000) and "\npublished\t" in record and
            "companion.dll" not in record, (rc, t.j, record, t.logtail()))
    rc = t.run("uninstall", exe, "--noscan")
    t.check("uninstall after the locked rollback preserves matching bytes never installed by the launcher",
            rc == 0 and read(companion) == read(source), (rc, t.j, t.logtail()))

    for name, before, should_keep in (
        ("unknown executable", m.mockpe.make_dll({"ProductName": "Fixture", "OriginalFilename": "fixture.exe"}), True),
        ("legacy version info", m.mockpe.make_dll({"ProductName": "WrongGPU", "OriginalFilename": "WrongGPU-Setup.exe"}), False),
        ("shipped restorer", read(t.legacy_restorer) if t.legacy_restorer else None, False),
    ):
        if before is None:
            continue
        _, b, exe = fixture("Safety " + name)
        restorer = os.path.join(b, "wronggpu-setup", "uninstall.exe")
        write(restorer, before)
        identity = os.stat(restorer).st_ino
        rc = t.run("install", exe, *args)
        kept = glob.glob(os.path.join(b, "WrongGPU-originals", "*", "wronggpu-setup", "uninstall.exe"))
        t.check("fresh install handles " + name,
                rc == 0 and read(restorer) != before and bool(kept) == should_keep and
                (not should_keep or (read(kept[0]) == before and os.stat(kept[0]).st_ino == identity and
                                     "unrecognized uninstaller was kept" in str(t.j))),
                (rc, t.j, kept, t.logtail()))
        t.run("uninstall", exe, "--noscan")
        if should_keep:
            t.check("uninstall leaves the kept unknown executable available", len(kept) == 1 and read(kept[0]) == before, kept)

    _, b, exe = fixture("Safety linked companion")
    link = os.path.join(package, "OptiScaler", "linked.dll")
    outside_package = os.path.join(t.work, "companion-outside-package.dll")
    write(outside_package, b"outside package")
    os.link(outside_package, link)
    try:
        rc = t.run("install", exe, *args)
        t.check("scoped package rejects a linked companion before copying it",
                rc != 0 and not os.path.exists(os.path.join(b, "OptiScaler", "linked.dll")) and "linked.dll" in str(t.j), (rc, t.j))
    finally:
        os.unlink(link)

    _, b, exe = fixture("Safety stale staging files")
    assert t.run("install", exe, *args) == 0, t.j
    cfg = settings_path(b)
    stale = cfg + ".wronggpu-tmp-4294967295-1"
    original_sibling = cfg + ".wronggpu-old-4294967295-1"
    write(stale, b"stale staged bytes")
    write(original_sibling, b"original to keep")
    rc = t.run("set", exe, "fps_overlay=on", "--noscan")
    t.check("the next staged write cleans an abandoned temporary and keeps the saved original",
            rc == 0 and not os.path.exists(stale) and read(original_sibling) == b"original to keep", (rc, t.j, t.logtail()))
    saved = read(cfg)
    t.env["WRONGGPU_TEST_REPLACE_RACE"] = cfg
    try:
        rc = t.run("set", exe, "fps_overlay=off", "--noscan")
    finally:
        del t.env["WRONGGPU_TEST_REPLACE_RACE"]
    kept = [path for path in glob.glob(cfg + ".wronggpu-old-*") if path != original_sibling]
    t.check("a failed rename-back reports the path retaining the original",
            rc != 0 and len(kept) == 1 and read(kept[0]) == saved and
            kept[0] in t.j.get("message", "") and read(cfg) == b"; competing fixture file\r\n",
            (rc, t.j, kept, t.logtail()))
    write(cfg, saved)
    t.run("uninstall", exe, "--noscan")

    scoped_self_uninstall(t, opti, fixture)
    games = os.path.normcase(os.path.realpath(os.path.join(t.work, "games")))
    for root in fixtures:
        path = os.path.realpath(root)
        if os.path.normcase(os.path.dirname(path)) != games:
            raise RuntimeError("refusing to clean up a fixture outside the games directory: " + path)
        shutil.rmtree(path)


def scoped_self_uninstall(t, opti, fixture):
    programs = os.path.join(t.work, "round-two-registered")
    installed = os.path.join(programs, "WrongGPU.exe")
    portable = os.path.join(t.work, "round-two-portable", "WrongGPU.exe")
    menu = os.path.join(t.work, "round-two-menu")
    args = ("--opti=" + opti, "--noscan", "--force", "--experimental")
    _, b, exe = fixture("Safety off-list game")
    assert t.run("install", exe, *args) == 0, t.j
    # This per-game entry is deliberately not in the explicit root's managed list.
    managed = os.path.join(t.data, "managed-games.txt")
    write(managed, "\n".join(line for line in read(managed).decode().splitlines() if line.lower() != b.lower()) + "\n")
    foreign = t.entries_for(b)
    assert foreign, t.reg_keys()
    assert t.run("self-install", "--dir=" + programs, "--startmenu=" + menu) == 0, t.j
    os.makedirs(os.path.dirname(portable), exist_ok=True)
    shutil.copyfile(t.exe, portable)
    rc = t.run("self-uninstall", exe=portable)
    t.check("portable self-uninstall leaves the other registered launcher and off-list games intact",
            rc == 0 and not os.path.exists(portable) and os.path.isfile(installed) and
            os.path.isfile(os.path.join(menu, "WrongGPU.lnk")) and "WrongGPU" in t.reg_keys() and
            t.entries_for(b) == foreign and os.path.isfile(os.path.join(b, "wronggpu-setup", "install.record")),
            (rc, t.j, t.reg_keys()))
    create = ctypes.windll.kernel32.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = ctypes.windll.kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    lock = create(installed, 0x80000000, 3, None, 3, 0, None)  # reads can run, rename and delete cannot
    assert lock != ctypes.c_void_p(-1).value
    try:
        rc = t.run("self-uninstall", exe=installed)
    finally:
        close(lock)
    t.check("failed program retirement returns an error before removing registration",
            rc != 0 and "try again" in str(t.j) and "WrongGPU" in t.reg_keys() and os.path.isfile(installed), (rc, t.j))
    rc = t.run("self-uninstall", exe=installed)
    t.check("retry retires the registered launcher and removes only its registration",
            rc == 0 and "WrongGPU" not in t.reg_keys() and t.entries_for(b) == foreign, (rc, t.j, t.reg_keys()))
    t.run("uninstall", exe, "--noscan")
