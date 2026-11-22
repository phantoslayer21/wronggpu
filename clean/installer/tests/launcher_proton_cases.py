# SPDX-License-Identifier: MIT
"""Engine.ini for Ray Reconstruction on the native Linux launcher (S3): the game runs under Proton, so the line goes into the game's
own Wine prefix. A Steam game: <library>/steamapps/compatdata/<appid>/pfx/drive_c/users/steamuser/{Documents,AppData/Local}, refused in
plain words (nothing written) until Steam has made it. A game that is not from Steam: the Windows folder the player points at once
(the windows-folder verb), remembered. Only a fake Steam tree under the suite's work folder is read or written; the suite's
Documents / LOCALAPPDATA test roots are taken away for these runs, as the real launcher has none. For run_launcher_tests.py
(--native-linux), after test_rr_config_fix. Written for this project."""
import os
import shutil

import mockpe
import run_installer_tests as suite
from run_installer_tests import read, write


def _game(t, root, exe_name, appid=None):
    """A game folder with the layout an install needs and, with an appid, its appmanifest in the library above it."""
    shutil.rmtree(root, ignore_errors=True)
    b = os.path.join(root, "bin", "x64")
    write(os.path.join(b, exe_name), "mock game exe " * 200 + exe_name)
    write(os.path.join(b, "amd_fidelityfx_dx12.dll"), suite.AMD_V1)
    write(os.path.join(b, "nvngx_dlss.dll"), t.model.file)
    write(os.path.join(b, "steam_api64.dll"), "untouched game file")
    write(os.path.join(root, "Engine", "Plugins", "Runtime", "Vendor", "RR", "Binaries", "ThirdParty", "Win64", "nvngx_dlssd.dll"),
          mockpe.make_dll({"ProductName": "RR library (test stand-in)"}, (310, 5, 0, 0), extra=b"synthetic, never loaded", dll_name="nvngx_dlssd.dll"))
    if appid:
        steamapps = os.path.dirname(os.path.dirname(root))
        write(os.path.join(steamapps, "appmanifest_%s.acf" % appid),
              '"AppState"\n{\n\t"appid"\t\t"%s"\n\t"installdir"\t\t"%s"\n}\n' % (appid, os.path.basename(root)))
    return b, os.path.join(b, exe_name)


def _tree(p):
    out = []
    for d, _, fs in os.walk(p):
        out += [os.path.relpath(os.path.join(d, f), p) for f in fs] + [os.path.relpath(d, p) + "/" for _ in [0]]
    return sorted(out)


def run(t, opti):
    print("\n[proton] native Linux: Engine.ini goes into the game's Proton prefix (or the Windows folder the player chose)")
    base = os.path.join(t.work, "proton-steam")
    shutil.rmtree(base, ignore_errors=True)
    home_docs = os.path.join(t.env["HOME"], "Documents")
    saved_env = t.env
    t.env = {k: v for k, v in saved_env.items() if k not in ("WRONGGPU_TEST_DOCUMENTS", "LOCALAPPDATA")}
    try:
        steamapps = os.path.join(base, "steamapps")
        root = os.path.join(steamapps, "common", "Borderlands 4")
        b, exe = _game(t, root, "Borderlands4.exe", appid="4242")
        compat = os.path.join(steamapps, "compatdata")
        rc = t.run("install", exe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.check("proton: install into a Steam game folder", rc == 0, (rc, t.j.get("message")))
        # 1. Steam has made no Proton folder: refused in plain words, nothing written anywhere
        rc = t.run("set", exe, "rr_config_fix=on")
        warn = " ".join(t.j.get("warnings", []))
        t.check("proton: no Proton folder yet: the choice is refused with a plain sentence (start the game once so Steam makes it), nothing written",
                rc == 0 and "Steam has not made this game's Proton folder yet" in warn and "Start the game once" in warn and not os.path.exists(compat) and
                not os.path.exists(home_docs), (rc, t.j.get("warnings"), t.j.get("notes")))
        rc = t.run("get", exe, "rr_config_fix")
        info = t.j.get("rr_config", {})
        t.check("proton: get says no file is known, and why", rc == 0 and info.get("file") == "" and "Proton folder" in info.get("file_why", "") and info.get("fixed") is False, info)
        # 2. the prefix exists (the game was started once): the line goes into its Documents, the file read-only as before
        user = os.path.join(compat, "4242", "pfx", "drive_c", "users", "steamuser")
        cfg = os.path.join(user, "Documents", "My Games", "Borderlands 4", "Saved", "Config", "Windows")
        write(os.path.join(cfg, "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")
        ini = os.path.join(cfg, "Engine.ini")
        rc = t.run("set", exe, "rr_config_fix=on")
        data = read(ini) if os.path.isfile(ini) else b""
        t.check("proton: with the prefix there: Engine.ini is in the prefix's Documents with the line, read-only, nothing in this account's Documents",
                rc == 0 and data.startswith(b"[SystemSettings]\r\n") and len(data) == 45 and not os.access(ini, os.W_OK) and not os.path.exists(home_docs),
                (rc, data, t.j.get("warnings")))
        rc = t.run("get", exe, "rr_config_fix")
        info = t.j.get("rr_config", {})
        t.check("proton: get names the file in the prefix, fixed", rc == 0 and info.get("fixed") is True and info.get("file", "").replace("\\", "/").lower().endswith(
                "compatdata/4242/pfx/drive_c/users/steamuser/documents/my games/borderlands 4/saved/config/windows/engine.ini"), info)
        # 3. uninstall takes the same line out of the same file
        rc = t.run("uninstall", exe)
        t.check("proton: uninstall takes the line out of the same file (the file it made is gone, the game's settings folder stays)",
                rc == 0 and not os.path.exists(ini) and os.listdir(cfg) == ["GameUserSettings.ini"], (rc, os.listdir(cfg)))
        # 4. a game that is not from Steam: it asks once for the Windows folder and remembers it
        plain_root = os.path.join(t.work, "proton-epic", "Borderlands 4")
        pb, pexe = _game(t, plain_root, "Borderlands4.exe")
        rc = t.run("install", pexe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        rc = t.run("set", pexe, "rr_config_fix=on")
        warn = " ".join(t.j.get("warnings", []))
        t.check("proton: a game that is not from Steam: refused in plain words, asking for its Windows folder once",
                rc == 0 and "not a Steam game" in warn and "once" in warn and not os.path.exists(home_docs), (rc, t.j.get("warnings")))
        wine_user = os.path.join(t.work, "proton-wine-user")
        shutil.rmtree(wine_user, ignore_errors=True)
        rc = t.run("windows-folder", pexe, wine_user)
        t.check("proton: windows-folder with a folder that is not there is refused, nothing remembered", rc != 0 and "is not a folder" in t.j.get("error", ""), (rc, t.j))
        os.makedirs(wine_user)
        rc = t.run("windows-folder", pexe, wine_user)
        t.check("proton: windows-folder with neither Documents nor AppData in it is refused", rc != 0 and "no Documents or AppData" in t.j.get("error", ""), (rc, t.j))
        os.makedirs(os.path.join(wine_user, "Documents"))
        rc = t.run("windows-folder", pexe, wine_user)
        t.check("proton: windows-folder remembers the Windows folder", rc == 0 and t.j.get("windows_folder", {}).get("folder", "").replace("\\", "/").endswith("proton-wine-user"), (rc, t.j))
        rc = t.run("windows-folder", pexe)
        t.check("proton: it is remembered by the next run of the launcher", rc == 0 and t.j.get("windows_folder", {}).get("folder", "").replace("\\", "/").endswith("proton-wine-user"), t.j)
        pcfg = os.path.join(wine_user, "Documents", "My Games", "Borderlands 4", "Saved", "Config", "Windows")
        rc = t.run("set", pexe, "rr_config_fix=on")
        t.check("proton: the remembered folder before the game made its settings folder: nothing written, added after the first start",
                rc == 0 and any("will be added after you start the game once" in n for n in t.j.get("notes", [])) and not os.path.exists(os.path.join(pcfg, "Engine.ini")),
                (rc, t.j.get("notes"), t.j.get("warnings")))
        write(os.path.join(pcfg, "GameUserSettings.ini"), b"[ScalabilityGroups]\r\n")
        rc = t.run("set", pexe, "rr_config_fix=on")
        t.check("proton: the settings folder is there: Engine.ini written in the remembered folder", rc == 0 and os.path.isfile(os.path.join(pcfg, "Engine.ini")) and
                len(read(os.path.join(pcfg, "Engine.ini"))) == 45, (rc, t.j.get("warnings")))
        rc = t.run("uninstall", pexe)
        t.check("proton: uninstall takes it out of the remembered folder too", rc == 0 and not os.path.exists(os.path.join(pcfg, "Engine.ini")), rc)
        rc = t.run("windows-folder", pexe, "--clear")
        t.check("proton: --clear forgets it", rc == 0 and t.j.get("windows_folder", {}).get("folder") == "", t.j)
    finally:
        t.env = saved_env
