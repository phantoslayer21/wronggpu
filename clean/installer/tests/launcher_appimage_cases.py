# SPDX-License-Identifier: MIT
"""Uninstall WrongGPU in the native Linux launcher (S5): the AppImage file ($APPIMAGE) and WrongGPU's data folder go after the
player's confirmation, and nothing else. The suite never points APPIMAGE at the program under test: it uses a fake file with an
AppImage's header, in the work folder, and a fake data folder (the suite's LOCALAPPDATA test root). Every refusal leaves both
untouched; a game that still has WrongGPU stops it; files next to the AppImage, in the data folder's parent and in the games
stay. For run_launcher_tests.py (--native-linux). Written for this project."""
import os
import shutil

import launcher_proton_cases as pc
from native_linux import host
from run_installer_tests import read, write

HEAD = b"\x7fELF\x02\x01\x01\x00AI\x02\x00\x00\x00\x00\x00"


def run(t, opti):
    print("\n[appimage] native Linux: Uninstall WrongGPU deletes the AppImage and WrongGPU's data folder, nothing else")
    base = os.path.join(t.work, "appimage-case")
    shutil.rmtree(host(base), ignore_errors=True)
    apps = os.path.join(base, "Applications")
    image = os.path.join(apps, "WrongGPU-0.6.5-x86_64.AppImage")
    sibling = os.path.join(apps, "OtherApp.AppImage")
    write(image, HEAD + b"fake payload")
    write(sibling, HEAD + b"another program's AppImage")
    data = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU")  # the core's AppDataPath() in this run
    neighbour = os.path.join(t.env["LOCALAPPDATA"], "OtherProgram", "keep.txt")
    write(neighbour, b"not WrongGPU's")
    write(os.path.join(data, "launcher.ini"), b"[settings]\r\n")
    write(os.path.join(data, "model", "M", "model.wgcm"), b"converted data")
    appdir = os.path.dirname(host(t.exe))  # this program stands in for the one inside the AppImage's mount
    saved = t.env

    def with_env(**kw):
        e = dict(saved)
        for k in ("APPIMAGE", "APPDIR"):
            e.pop(k, None)
        e.update({k: v for k, v in kw.items() if v is not None})
        return e

    def untouched():
        return os.path.isfile(image) and os.path.isfile(os.path.join(data, "model", "M", "model.wgcm")) and os.path.isfile(neighbour)
    try:
        # refusals: nothing is deleted whatever the reason
        for tag, env, want in (
                ("no $APPIMAGE", with_env(APPDIR=appdir), "not running from an AppImage"),
                ("a name that is not WrongGPU's", with_env(APPIMAGE=host(sibling), APPDIR=appdir), "not WrongGPU"),
                ("a program that is not inside it", with_env(APPIMAGE=host(image), APPDIR=os.path.join(host(base), "elsewhere")), "not inside"),
                ("no AppImage header", None, "not an AppImage file")):
            if env is None:
                bare = os.path.join(apps, "WrongGPU-bare.AppImage")
                write(bare, b"just text")
                env = with_env(APPIMAGE=host(bare), APPDIR=appdir)
            t.env = env
            rc = t.run("uninstall-appimage", "--confirm")
            t.check("appimage: refused for " + tag + ", in words, nothing deleted", rc != 0 and want in t.j.get("error", "") and untouched(), (rc, t.j))
        good = with_env(APPIMAGE=host(image), APPDIR=appdir)
        t.env = good
        rc = t.run("uninstall-appimage")
        t.check("appimage: without --confirm it says what would go and deletes nothing", rc == 0 and t.j.get("appimage", {}).get("removed") is False and
                t.j["appimage"].get("file", "").endswith("WrongGPU-0.6.5-x86_64.AppImage") and t.j["appimage"].get("data", "").endswith("WrongGPU") and untouched(), (rc, t.j))
        # a game that still has WrongGPU stops it: both stay (its record reads the data)
        gb, gexe = pc._game(t, os.path.join(base, "games", "Some Game"), "Game.exe")
        t.env = saved
        rc = t.run("install", gexe, "--opti=" + opti, "--noscan", "--force", "--experimental")
        t.env = good
        rc = t.run("uninstall-appimage", "--confirm", "--noscan")
        t.check("appimage: a game still has WrongGPU: refused, nothing deleted", rc != 0 and "still has WrongGPU" in t.j.get("error", "") and untouched(), (rc, t.j))
        t.env = saved
        rc = t.run("uninstall", gexe)
        game_files = sorted(os.listdir(os.path.dirname(gexe)))
        t.env = good
        rc = t.run("uninstall-appimage", "--confirm", "--noscan")
        t.check("appimage: confirmed: the AppImage file and the data folder are gone, exit 0", rc == 0 and t.j.get("appimage", {}).get("removed") is True and
                not os.path.exists(image) and not os.path.exists(data), (rc, t.j, os.path.exists(data)))
        t.check("appimage: nothing else went: another AppImage next to it, another program's data folder, the game's files",
                os.path.isfile(sibling) and os.path.isfile(neighbour) and os.path.isdir(apps) and os.path.isfile(gexe) and
                sorted(os.listdir(os.path.dirname(gexe))) == game_files, (os.listdir(apps), os.path.isfile(neighbour)))
        rc = t.run("uninstall-appimage", "--confirm", "--noscan")
        t.check("appimage: a second run finds no file and refuses", rc != 0 and not os.path.exists(data), (rc, t.j))
    finally:
        t.env = saved
