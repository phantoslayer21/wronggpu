# SPDX-License-Identifier: MIT
"""The native Linux launcher's wording fixes from the UI review (launcher step 4): the exact strings a player reads.

    run_linux_ui_wording_tests.py --exe <WrongGPU> [--work <dir>]

Same setup as run_linux_ui_tests.py (a home of its own, a fake Steam library, SDL's offscreen driver, nothing of the player's read or
written). Needs a WRONGGPU_TESTING build of the Linux launcher. Written for this project."""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_linux_ui_tests as base  # noqa: E402

check = base.check
NO_CARD = "No graphics card found. WrongGPU's models need an RX 9000-series GPU; games use AMD FSR."
HIP = "Optional. Model M runs through Proton's Direct3D 12 without it, and no game setting uses it."
IMAGE = "WrongGPU-test-x86_64.AppImage"


def dump_of(exe, work, home, tag, ids=("0x1002", "0x7550"), image=False, ignore=False, shots=False):
    env, run_dir = base.environment(work, home, tag)
    env["WRONGGPU_TEST_DRM_ROOT"] = "Z:" + str(base.fake_drm(run_dir / "drm", *ids))
    path = None
    if image:
        path = run_dir / "Apps" / IMAGE
        path.parent.mkdir(parents=True)
        path.write_bytes(b"\x7fELF\x02\x01\x01\x00AI\x02\x00\x00\x00\x00\x00fake")
        appdir = Path(exe).resolve().parent
        base.fake_proc(run_dir / "proc", appdir, path)
        env.update(APPIMAGE=str(path), APPDIR=str(appdir), WRONGGPU_TEST_PROC_ROOT=str(run_dir / "proc"))
    if ignore:
        (run_dir / "data" / "WrongGPU").mkdir(parents=True, exist_ok=True)
        (run_dir / "data" / "WrongGPU" / "ignore-files.txt").write_bytes(b"wronggpu-ignore-files 1\r\n")
    out = work / tag
    args = ["ui-dump", f"--dump={out / 'dump.json'}"] + ([f"--shots={out}"] if shots else [])
    r = base.run(exe, args, env, 300)
    check(r.returncode == 0, f"{tag}: ui-dump exits 0 (got {r.returncode}: {r.stderr[-300:]})")
    return json.loads((out / "dump.json").read_text()), path, out


def test_first_run(exe, work, home):
    print("first-run sheet: no card, and the Show details pictures")
    for tag, ids, want in (("none", (None, None), NO_CARD), ("rdna3", ("0x1002", "0x744c"), None), ("rdna4", ("0x1002", "0x7550"), None)):
        d, _, out = dump_of(exe, work, home, "fr-" + tag, ids, shots=True)
        lines = d["first_run_lines"]
        if want:
            check(want in lines and not any(l.startswith("Graphics card detected: none") for l in lines), f"{tag}: the no-card line is exactly '{want}' ({lines[:3]})")
        else:
            check(any(l.startswith("Graphics card detected: AMD Radeon") for l in lines), f"{tag}: the card line is unchanged ({lines[:3]})")
        pic = out / "first-run-details.png"
        check(pic.is_file() and base.png_size(pic) == (1180, 760), f"{tag}: the picture of the sheet with Show details open is there")


def test_settings(exe, work, home):
    print("Settings: the Launcher row, the side label, Appearance, HIP")
    d, image, _ = dump_of(exe, work, home, "set-ignore", image=True, ignore=True)
    names = [c["text"] for c in d["settings"]["controls"]]
    joined = "\n".join(names)
    data = str(work / "set-ignore" / "data" / "WrongGPU")
    check(f"WrongGPU 0.6.5, AppImage" in joined or any(n.endswith(", AppImage") for n in names), f"the side label says AppImage ({[n for n in names if 'WrongGPU 0' in n]})")
    check(not any("not installed" in n.lower() for n in names), "no 'not installed' on the AppImage's Settings")
    row = next((n for n in names if n.startswith("Launcher install")), "")
    check("Runs from an AppImage" in row, f"the status is 'Runs from an AppImage' ({row[:120]})")
    check(row.count("Runs from") == 1, "the row does not repeat 'Runs from'")
    check(f"The file: {image}. Uninstall deletes that file and WrongGPU's data folder ({data}) after you confirm." in row, f"the row's sentence names both paths ({row})")
    check("Your ignore-files.txt list is kept. Nothing else is deleted." in row, f"the row says the ignore list is kept when there is one ({row[-120:]})")
    check("Uninstall WrongGPU from your games first." not in row, "no 'games first' sentence while no game has WrongGPU")
    appearance = " ".join(n for n in names if n.startswith("Appearance:") and n not in ("Appearance: System", "Appearance: Light", "Appearance: Dark"))
    caveat = "System should follow the desktop's light or dark setting, but WrongGPU can't read it on Linux yet, so it uses light for now."
    check(appearance.endswith("The launcher stays dark. " + caveat) or "The launcher stays dark." in appearance and caveat in appearance,
          f"the Appearance caveat shows with Dark chosen, the default ({appearance})")
    hip = " ".join(n for n in names if n.startswith("HIP runtime"))
    check(HIP in hip, f"the HIP row says Model M runs through Proton's Direct3D 12 ({hip})")
    d2, _, _ = dump_of(exe, work, home, "set-noignore", image=True)
    row2 = next((n for n in (c["text"] for c in d2["settings"]["controls"]) if n.startswith("Launcher install")), "")
    check("ignore-files.txt" not in row2 and "Nothing else is deleted." in row2, f"no ignore-list sentence when there is no list ({row2[-100:]})")
    # the words by their inputs: with and without the list, with and without a game that has WrongGPU
    words = d["appimage_words"]
    check(len(words) == 4 and ("Uninstall WrongGPU from your games first." in words[1] and "Uninstall WrongGPU from your games first." not in words[0]),
          f"the 'games first' sentence appears only when a game has WrongGPU ({words[:2]})")
    check(words[2].endswith("Nothing else is deleted.") and "Your ignore-files.txt list is kept." in words[2] and "ignore-files" not in words[0], "the ignore-list sentence follows the list")
    text = d["appimage_uninstall_text"]
    check("Your ignore-files.txt list is kept." in text[1] and "ignore-files" not in text[0], f"the uninstall question says the list is kept, only when there is one ({text[1][-140:]})")
    check("Nothing else is deleted. Your games and their files are not touched." in text[1], "the question still says nothing else is touched")
    # System chosen: the caveat is the row's sentence
    env, run_dir = base.environment(work, home, "set-system")
    (run_dir / "data" / "WrongGPU").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "WrongGPU" / "launcher.ini").write_bytes(b"first_run_done=1\r\nappearance=system\r\n")
    r = base.run(exe, ["ui-dump", f"--dump={work / 'set-system' / 'dump.json'}"], env, 300)
    dd = json.loads((work / "set-system" / "dump.json").read_text())
    system = " ".join(c["text"] for c in dd["settings"]["controls"] if c["text"].startswith("Appearance:") and c["text"] not in ("Appearance: System", "Appearance: Light", "Appearance: Dark"))
    check(r.returncode == 0 and caveat in system and "The launcher stays" not in system, f"with System chosen the row's sentence is the caveat ({system})")


def test_windows_folder_card(exe, work, home):
    print("the Windows-folder card: a command the player can run, with Copy")
    for tag, image, shown in (("plain", "/home/player/Apps/WrongGPU-0.6.5-x86_64.AppImage", "'/home/player/Apps/WrongGPU-0.6.5-x86_64.AppImage'"),
                              ("quote", "/home/player/it's here/WrongGPU-0.6.5-x86_64.AppImage", "'/home/player/it'\\''s here/WrongGPU-0.6.5-x86_64.AppImage'")):
        env, _ = base.environment(work, home, "wf-" + tag)
        env["APPIMAGE"] = image
        out = work / ("wf-" + tag)
        r = base.run(exe, ["ui-render", f"--out-dir={out}", f"--fixture={HERE / 'ui_fixture.ini'}", "--size=1180x1500", "--scales=1",
                           "--frames=game-rrconfig-nofolder-other,game-rrconfig-nofolder-steam"], env, 600)
        check(r.returncode == 0, f"{tag}: ui-render exits 0 (got {r.returncode}: {r.stderr[-300:]})")
        texts = (out / "game-rrconfig-nofolder-other@100.png.texts").read_text(encoding="utf-8").splitlines()
        joined = " ".join(texts)
        check("It isn't a Steam game, so WrongGPU can't tell where its Proton or Wine folder is." in joined and "Then run this once in a terminal:" in joined,
              f"{tag}: the card says why and what to do ({joined[:100]})")
        check("Wine prefix" not in joined and "Run: WrongGPU windows-folder" not in joined, f"{tag}: no 'Wine prefix' and no bare WrongGPU command")
        line = next((t.split("\t")[-1] for t in texts if "Point WrongGPU at the Windows folder" in t), "")
        check(f"{shown} windows-folder '" in line and line.endswith("Borderlands4.exe' '<that folder>'"),
              f"{tag}: the command starts with the AppImage path in quotes, then the verb, the game's exe and the folder ({line})")
        check(any(t.split("\t")[2:] == ["Copy the windows-folder command"] for t in texts), f"{tag}: the card has a Copy button")
        steam = " ".join((out / "game-rrconfig-nofolder-steam@100.png.texts").read_text(encoding="utf-8").splitlines())
        check("Point WrongGPU at the Windows folder" not in steam and "Steam has not made this game's Proton folder yet" in steam, f"{tag}: a Steam game without its Proton folder gets no command card")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    work = Path(a.work or "linux-ui-wording-tests").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    home = base.make_home(work)
    for fn in (test_first_run, test_settings, test_windows_folder_card):
        fn(Path(a.exe), work, home)
    if base.failures:
        print(f"{len(base.failures)} check(s) failed")
        return 1
    print("linux ui wording tests: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
