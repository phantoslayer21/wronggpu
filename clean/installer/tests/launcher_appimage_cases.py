# SPDX-License-Identifier: MIT
"""Uninstall WrongGPU in the native Linux launcher (S5): the AppImage file ($APPIMAGE) and WrongGPU's data folder go after the
player's confirmation, and nothing else. The suite never points APPIMAGE at the program under test: it uses a fake file with an
AppImage's header, in the work folder, a fake data folder (the suite's LOCALAPPDATA test root) and a made-up process folder
(WRONGGPU_TEST_PROC_ROOT) holding what the real AppImage mount leaves: a mount table line for $APPDIR, a process started from the
AppImage file and its /dev/fuse descriptor naming the mount. Every refusal leaves both
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
    other = os.path.join(apps, "WrongGPU-old.AppImage")  # a real WrongGPU AppImage, but not the one that is running
    write(other, HEAD + b"an older WrongGPU AppImage")
    data = os.path.join(t.env["LOCALAPPDATA"], "WrongGPU")  # the core's AppDataPath() in this run
    neighbour = os.path.join(t.env["LOCALAPPDATA"], "OtherProgram", "keep.txt")
    write(neighbour, b"not WrongGPU's")
    write(os.path.join(data, "launcher.ini"), b"[settings]\r\n")
    write(os.path.join(data, "model", "M", "model.wgcm"), b"converted data")
    appdir = os.path.realpath(os.path.dirname(host(t.exe)))  # this program stands in for the one inside the AppImage's mount
    saved = t.env
    st = os.stat(appdir)
    proc = os.path.join(host(base), "proc")

    def make_proc(server_exe=None, connection=None):
        """The process folder after a runtime mounted $APPDIR: the mount's own device number is its FUSE connection number."""
        shutil.rmtree(proc, ignore_errors=True)
        os.makedirs(os.path.join(proc, "self"))
        with open(os.path.join(proc, "self", "mountinfo"), "w") as f:
            f.write("22 1 0:5 / /sys rw - sysfs sysfs rw\n1003 60 %d:%d / %s ro,nosuid,nodev - fuse.WrongGPU-0.6.5-x86_64.AppImage WrongGPU-0.6.5-x86_64.AppImage ro\n" %
                    (os.major(st.st_dev), os.minor(st.st_dev), appdir.replace(" ", "\\040")))
        if server_exe:
            os.makedirs(os.path.join(proc, "4242", "fdinfo"))
            os.symlink(server_exe, os.path.join(proc, "4242", "exe"))
            with open(os.path.join(proc, "4242", "fdinfo", "5"), "w") as f:
                f.write("pos:\t0\nflags:\t02100002\nmnt_id:\t43\nino:\t136\nfuse_connection:\t%d\n" % (os.minor(st.st_dev) if connection is None else connection))
    make_proc(host(image))

    def with_env(**kw):
        e = dict(saved)
        for k in ("APPIMAGE", "APPDIR"):
            e.pop(k, None)
        e["WRONGGPU_TEST_PROC_ROOT"] = proc
        e.update({k: v for k, v in kw.items() if v is not None})
        return e

    def untouched():
        return os.path.isfile(image) and os.path.isfile(other) and os.path.isfile(os.path.join(data, "model", "M", "model.wgcm")) and os.path.isfile(neighbour)
    try:
        # refusals: nothing is deleted whatever the reason
        for tag, env, want in (
                ("no $APPIMAGE", with_env(APPDIR=appdir), "not running from an AppImage"),
                ("a name that is not WrongGPU's", with_env(APPIMAGE=host(sibling), APPDIR=appdir), "not WrongGPU"),
                ("a program that is not inside it", with_env(APPIMAGE=host(image), APPDIR=os.path.join(host(base), "elsewhere")), "not inside"),
                ("a valid WrongGPU AppImage that is not the one running (the real $APPDIR)", with_env(APPIMAGE=host(other), APPDIR=appdir), "cannot prove"),
                ("a mount that no process started from the file serves", with_env(APPIMAGE=host(image), APPDIR=appdir, WRONGGPU_TEST_PROC_ROOT=os.path.join(host(base), "proc-empty")), "not a mounted AppImage"),
                ("no AppImage header", None, "not an AppImage file")):
            if env is None:
                bare = os.path.join(apps, "WrongGPU-bare.AppImage")
                write(bare, b"just text")
                env = with_env(APPIMAGE=host(bare), APPDIR=appdir)
            t.env = env
            rc = t.run("uninstall-appimage", "--confirm")
            t.check("appimage: refused for " + tag + ", in words, nothing deleted", rc != 0 and want in t.j.get("error", "") and untouched(), (rc, t.j))
        good = with_env(APPIMAGE=host(image), APPDIR=appdir)
        # a mount that something else serves: the case of the folder is the disk's
        t.env = with_env(APPIMAGE=host(image), APPDIR=appdir.upper())
        rc = t.run("uninstall-appimage", "--confirm")
        t.check("appimage: $APPDIR in another letter case is not the folder: refused, nothing deleted", rc != 0 and "not inside" in t.j.get("error", "") and untouched(), (rc, t.j))
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
        # links in the data folder are removed, never followed: what they lead to stays
        keep_dir = os.path.join(base, "outside-data")
        write(os.path.join(keep_dir, "keep.txt"), b"keep")
        write(os.path.join(keep_dir, "launcher.ini"), b"[settings]\r\n")
        write(image, HEAD + b"fake payload")
        os.symlink(host(keep_dir), host(data))  # the data folder itself is a link
        rc = t.run("uninstall-appimage", "--confirm", "--noscan")
        t.check("appimage: a data folder that is a link: the link goes, its target and what is in it stay",
                rc == 0 and not os.path.lexists(host(data)) and not os.path.exists(image) and
                read(os.path.join(keep_dir, "keep.txt")) == b"keep" and os.path.isfile(os.path.join(keep_dir, "launcher.ini")), (rc, t.j))
        nested_dir = os.path.join(base, "outside-nested")
        write(os.path.join(nested_dir, "keep.txt"), b"keep")
        write(os.path.join(base, "outside-file.txt"), b"keep")
        write(image, HEAD + b"fake payload")
        write(os.path.join(data, "launcher.ini"), b"[settings]\r\n")
        write(os.path.join(data, "model", "M", "model.wgcm"), b"converted data")
        os.symlink(host(nested_dir), os.path.join(host(data), "model", "linkdir"))
        os.symlink(host(os.path.join(base, "outside-file.txt")), os.path.join(host(data), "linkfile"))
        rc = t.run("uninstall-appimage", "--confirm", "--noscan")
        t.check("appimage: links inside the data folder are removed, their targets stay",
                rc == 0 and not os.path.lexists(host(data)) and not os.path.exists(image) and
                read(os.path.join(nested_dir, "keep.txt")) == b"keep" and read(os.path.join(base, "outside-file.txt")) == b"keep", (rc, t.j))
    finally:
        t.env = saved
