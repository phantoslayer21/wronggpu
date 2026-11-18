# SPDX-License-Identifier: MIT
"""The native Linux launcher's Codex review fixes (launcher step 4): each finding as a real UI state, with the complete string.

    run_linux_ui_codexfix_tests.py --exe <WrongGPU> [--work <dir>]

Same setup as run_linux_ui_tests.py (a home of its own, a fake Steam library, SDL's offscreen driver, a made-up process folder for
the AppImage proof). Needs a WRONGGPU_TESTING build of the Linux launcher. Written for this project."""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_linux_ui_tests as base  # noqa: E402

check = base.check
IMAGE = "WrongGPU-test-x86_64.AppImage"
CAVEAT = "System should follow the desktop's light or dark setting, but WrongGPU can't read it on Linux yet, so it uses light for now."
HEAD = b"\x7fELF\x02\x01\x01\x00AI\x02\x00\x00\x00\x00\x00fake"
LIST = b"wronggpu-ignore-files 1\r\n"


def shell(p):
    return "'" + str(p).replace("'", "'\\''") + "'"


def dump(exe, work, home, tag, *, image=None, proven=True, appdir=None, env_extra=None, args=(), setup=None, dirname="Apps"):
    """One ui-dump. image: the AppImage file to make (named IMAGE in `dirname`); proven: the made-up process folder backs it."""
    env, run_dir = base.environment(work, home, tag)
    path = None
    appdir = Path(appdir) if appdir else Path(exe).resolve().parent
    if image:
        path = run_dir / dirname / IMAGE
        path.parent.mkdir(parents=True)
        path.write_bytes(HEAD)
        env.update(APPIMAGE=str(path), APPDIR=str(appdir))
        if proven:
            base.fake_proc(run_dir / "proc", appdir, path)
            env["WRONGGPU_TEST_PROC_ROOT"] = str(run_dir / "proc")
    env.update(env_extra or {})
    if setup:
        setup(run_dir)
    out = work / tag
    r = base.run(exe, ["ui-dump", f"--dump={out / 'dump.json'}", *args], env, 300)
    check(r.returncode == 0, f"{tag}: ui-dump exits 0 (got {r.returncode}: {r.stderr[-300:]})")
    return json.loads((out / "dump.json").read_text()), path, run_dir


def names(d, page="settings"):
    return [c["text"] for c in d[page]["controls"]]


def launcher_row(d):
    return next((n for n in names(d) if n.startswith("Launcher install")), "")


def sentence(image, data, keep=False, games=False):
    return (f"The file: {image}. Uninstall deletes that file and WrongGPU's data folder ({data}) after you confirm. "
            + ("Your ignore-files.txt list is kept. " if keep else "") + "Nothing else is deleted."
            + (" Uninstall WrongGPU from your games first." if games else ""))


def question(image, data, keep):
    return ("Uninstall WrongGPU?\n\nThis deletes two things and cannot be undone:\n\n• The program file:\n  " + str(image) +
            "\n• WrongGPU's data folder (the converted Model M and Ray Reconstruction data, the games' list, settings and logs):\n  " + str(data) +
            "\n\n" + ("Your ignore-files.txt list is kept. " if keep else "") + "Nothing else is deleted. Your games and their files are not touched.")


def put_list(folder, link=False):
    folder.mkdir(parents=True, exist_ok=True)
    if link:
        (folder / "my-list.txt").write_bytes(LIST)
        (folder / "ignore-files.txt").symlink_to("my-list.txt")
    else:
        (folder / "ignore-files.txt").write_bytes(LIST)


def test_uninstall_words(exe, work, home):
    print("findings 2, 3, 4: the warning, the folder and the list in the Settings row and the dialog")
    games = []

    def with_game(run_dir):  # a game WrongGPU is installed in, on the managed list, in no library the launcher scans
        game = run_dir / "elsewhere" / "Ghost Game"
        (game / "wronggpu-setup").mkdir(parents=True)
        (game / "wronggpu-setup" / "install.record").write_bytes(b"x")
        put_list(run_dir / "data" / "WrongGPU")
        (run_dir / "data" / "WrongGPU" / "managed-games.txt").write_text("Z:" + str(game) + "\n")
        games.append(game)

    d, image, run = dump(exe, work, home, "w-plain", image=True, setup=lambda r: put_list(r / "data" / "WrongGPU"))
    data = run / "data" / "WrongGPU"
    want = f"Launcher install: Runs from an AppImage. {sentence(image, data, keep=True)} Settings and logs: {data}"
    check(launcher_row(d) == want, f"the row, whole ({launcher_row(d)})")
    check(d["appimage_question"] == question(image, data, True), f"the live dialog's text, whole ({d['appimage_question']})")
    d, image, run = dump(exe, work, home, "w-none", image=True)
    check(launcher_row(d) == f"Launcher install: Runs from an AppImage. {sentence(image, run / 'data' / 'WrongGPU')} Settings and logs: {run / 'data' / 'WrongGPU'}",
          f"no list: the row has no list sentence and no games sentence ({launcher_row(d)})")
    check(d["appimage_question"] == question(image, run / "data" / "WrongGPU", False), "no list: the dialog's text has no list sentence")
    # 2: a game outside the scanned libraries still blocks the uninstall, so the row says so (the Library does not list it)
    d, image, run = dump(exe, work, home, "w-game", image=True, setup=with_game)
    data = run / "data" / "WrongGPU"
    check(len(d["rows"]) == 2 and not any("Ghost" in r for r in d["rows"]), f"the Library does not list that game ({d['rows']})")
    check(launcher_row(d) == f"Launcher install: Runs from an AppImage. {sentence(image, data, keep=True, games=True)} Settings and logs: {data}",
          f"the row warns about the game the uninstall would refuse for ({launcher_row(d)[-160:]})")
    # 3: --data names another folder for the launcher's own files; the uninstall still deletes the shared folder
    other = work / "other-data"
    shutil.rmtree(other, ignore_errors=True)
    d, image, run = dump(exe, work, home, "w-override", image=True, args=(f"--data={other}",), setup=lambda r: put_list(other))
    data = run / "data" / "WrongGPU"
    check(launcher_row(d) == f"Launcher install: Runs from an AppImage. {sentence(image, data)} Settings and logs: {other}",
          f"--data: the row names the folder the uninstall deletes, no list kept there, and the logs label its own ({launcher_row(d)[-200:]})")
    check(d["appimage_question"] == question(image, data, False), "--data: the dialog names that folder too")
    shutil.rmtree(other, ignore_errors=True)
    d, image, run = dump(exe, work, home, "w-override2", image=True, args=(f"--data={other}",), setup=lambda r: put_list(r / "data" / "WrongGPU"))
    data = run / "data" / "WrongGPU"
    check(launcher_row(d) == f"Launcher install: Runs from an AppImage. {sentence(image, data, keep=True)} Settings and logs: {other}",
          "--data: the list in the shared folder is the one that is kept")
    shutil.rmtree(other, ignore_errors=True)
    # 4: a list that is a link: its target goes with the folder, so nothing is promised
    d, image, run = dump(exe, work, home, "w-link", image=True, setup=lambda r: put_list(r / "data" / "WrongGPU", link=True))
    data = run / "data" / "WrongGPU"
    check(launcher_row(d) == f"Launcher install: Runs from an AppImage. {sentence(image, data)} Settings and logs: {data}", "a linked list: the row promises nothing")
    check(d["appimage_question"] == question(image, data, False), "a linked list: the dialog promises nothing")


def test_label(exe, work, home):
    print("finding 5: the side label and the row when the AppImage cannot be proven")
    appdir = Path(exe).resolve().parent
    d, _, _ = dump(exe, work, home, "l-extract", image=True, proven=False, env_extra={"APPIMAGE_EXTRACT_AND_RUN": "1"})
    check("WrongGPU 0.6.5, AppImage" in names(d), f"extract-and-run: the side label says AppImage ({[n for n in names(d) if 'WrongGPU 0' in n]})")
    row = launcher_row(d)
    check(row.startswith("Launcher install: Runs from an AppImage. Uninstall WrongGPU is not available here. "), f"extract-and-run: the row says so, with no uninstall ({row[:140]})")
    check(not any(n.startswith("Uninstall WrongGPU") for n in names(d)), "extract-and-run: no Uninstall button")
    d, _, _ = dump(exe, work, home, "l-inherited", image=True, proven=False, appdir=work)  # another AppImage's variables, this program is not inside it
    check("WrongGPU 0.6.5, AppImage" not in names(d) and "WrongGPU 0.6.5, not installed" in names(d), f"inherited variables: the label stays ({[n for n in names(d) if 'WrongGPU 0' in n]})")
    check(launcher_row(d).startswith("Launcher install: Not an AppImage. Uninstall WrongGPU is not available here. "), "inherited variables: the row says it is no AppImage")
    d, _, _ = dump(exe, work, home, "l-proven", image=True)
    check("WrongGPU 0.6.5, AppImage" in names(d) and "Uninstall WrongGPU…" in names(d), "proven: the label and the button")
    del appdir


def test_command(exe, work, home):
    print("finding 1: the Windows-folder command, and Copy as a click")
    me = Path(exe).resolve()

    def command(tag, **kw):
        d, image, run = dump(exe, work, home, tag, **kw)
        c = d["windows_folder_copy"]
        game = c["exe"].replace("Z:", "", 1).replace("\\", "/")
        return c, image, game

    c, image, game = command("c-proven", image=True)
    check(c["clipboard"] == f"{shell(image)} windows-folder {shell(game)} '<that folder>'", f"proven AppImage: its file heads the command ({c['clipboard']})")
    check(c["toast"] == "The command is on the clipboard. Paste it into a terminal, and put your folder where it says <that folder>, inside the quotes.", "Copy's toast, whole")
    c, image, game = command("c-quote", image=True, dirname="it's here")
    check(c["clipboard"] == f"{shell(image)} windows-folder {shell(game)} '<that folder>'" and "it'\\''s here" in c["clipboard"], f"an apostrophe in the AppImage's folder is quoted ({c['clipboard'][:120]})")
    deep = "/".join(["d" * 200] * 3)  # a path of more than 600 characters
    c, image, game = command("c-long", image=True, dirname=deep)
    check(len(str(image)) > 600 and c["clipboard"] == f"{shell(image)} windows-folder {shell(game)} '<that folder>'", f"a long AppImage path is not cut ({len(str(image))} characters)")
    c, _, game = command("c-self")
    check(c["clipboard"] == f"{shell(me)} windows-folder {shell(game)} '<that folder>'", f"no AppImage: this program's own file heads it ({c['clipboard'][:120]})")
    c, _, game = command("c-other", env_extra={"APPIMAGE": str(work / "Other.AppImage")})
    check(c["clipboard"] == f"{shell(me)} windows-folder {shell(game)} '<that folder>'", f"another AppImage's variable, no proof: this program's own file ({c['clipboard'][:120]})")
    c, image, game = command("c-unproven", image=True, proven=False)
    check(c["clipboard"].startswith(shell(me) + " windows-folder "), "an AppImage file the system does not tie to this process is not used")


def test_misc(exe, work, home):
    print("findings 7: the pictures and the Settings rows named in the review")
    d, _, run = dump(exe, work, home, "m-shots", image=True, args=(f"--shots={work / 'm-shots'}",))
    out = work / "m-shots"
    details = (out / "first-run-details.png.texts").read_text(encoding="utf-8").replace("\u00a0", " ").splitlines()  # (the sheet keeps its words together with no-break spaces)
    plain = (out / "first-run.png.texts").read_text(encoding="utf-8").replace("\u00a0", " ").splitlines()
    line = "Graphics card detected: AMD Radeon RX 9070 XT"
    check(any(t.endswith("\t" + line) for t in details) and not any(t.endswith("\t" + line) for t in plain) and any(t.endswith("\tHide details") for t in details),
          "the Show details picture is really open (a detail row, 'Hide details')")
    d, _, _ = dump(exe, work, home, "m-hip", image=True, setup=lambda r: ((r / "libhip.so").write_bytes(b"x"), (r / "data" / "WrongGPU").mkdir(parents=True, exist_ok=True),
                                                                          (r / "data" / "WrongGPU" / "hip-runtime.txt").write_text(str(r / "libhip.so") + "\n")))
    hip = " ".join(n for n in names(d) if n.startswith("HIP runtime"))
    check(hip.startswith("HIP runtime: Found") and "Not used in games: Model M runs through Proton's Direct3D 12." in hip, f"the HIP row, found ({hip})")
    (work / "m-light").mkdir()
    d, _, _ = dump(exe, work, home, "m-light", image=True, setup=lambda r: ((r / "data" / "WrongGPU").mkdir(parents=True, exist_ok=True),
                                                                          (r / "data" / "WrongGPU" / "launcher.ini").write_bytes(b"first_run_done=1\r\nappearance=light\r\n")))
    app = " ".join(n for n in names(d) if n.startswith("Appearance:") and n not in ("Appearance: System", "Appearance: Light", "Appearance: Dark"))
    check(app == "Appearance: The launcher stays light. " + CAVEAT, f"Light chosen: the caveat follows ({app})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = Path(a.work or "linux-ui-codexfix-tests").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    home = base.make_home(work)
    for fn in (test_uninstall_words, test_label, test_command, test_misc):
        fn(Path(a.exe), work, home)
    if base.failures:
        print(f"{len(base.failures)} check(s) failed")
        return 1
    print("linux ui codex-fix tests: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
