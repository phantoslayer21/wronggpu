# SPDX-License-Identifier: MIT
"""The launcher's window on Linux (the one WrongGPU binary: src/app on the SDL window layer of src/platform/posix).

    run_linux_ui_tests.py --exe <WrongGPU> [--work <dir>]

Every run uses a home of its own under --work, its own XDG folders and registry file, a fake Steam library with two stub games
(PE stand-ins from mockpe.py: nothing of the player's is read or written), and SDL's offscreen video driver (no window opens).
  ui-dump    the pages as JSON: the structure the launcher tests expect (first run, Library, Settings, a page per game; each widget's
             class, text, id, place, state), the games read from the fake library, the PNGs of the pages
  ui-render  the fixture screenshots (Library, Settings, a game's page) at two scales: PNG files of the sizes asked for
  live       the real window under the offscreen driver with scripted SDL events (win32_window_sdl.cpp, WRONGGPU_UI_SCRIPT): a click
             on Settings, a scroll, a click on the Library's search field and typing in it, Ctrl+A and Backspace; the page each step
             left is read back as JSON; the window then closes on WM_CLOSE and the program exits 0
Needs a WRONGGPU_TESTING build of the Linux launcher.
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mockpe  # noqa: E402

ROLES = {"Button", "Switch", "Segment", "Item", "Select", "Link", "MenuItem", "Edit", "ScrollBar", "Dialog", "Menu", "Progress", "Text"}
failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("  FAIL:", what)
    return cond


def png_size(path):
    b = Path(path).read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n" or b[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", b[16:24])


def make_home(work):
    """A home with a Steam library of two games, each with a DLSS stand-in (what makes the launcher list a game)."""
    home = work / "home"
    steam = home / ".local" / "share" / "Steam" / "steamapps"
    games = [("1091500", "Cyberpunk 2077", "bin/x64/Cyberpunk2077.exe"), ("1086940", "Baldur's Gate 3", "bin/bg3.exe")]
    (steam / "common").mkdir(parents=True)
    apps = "".join(f'\t\t\t"{i}"\t\t"1"\n' for i, _, _ in games)
    (steam / "libraryfolders.vdf").write_text('"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t\t"apps"\n\t\t{\n%s\t\t}\n\t}\n}\n' % (home / ".local/share/Steam", apps))
    for appid, name, exe in games:
        (steam / f"appmanifest_{appid}.acf").write_text('"AppState"\n{\n\t"appid"\t\t"%s"\n\t"name"\t\t"%s"\n\t"installdir"\t\t"%s"\n\t"StateFlags"\t\t"4"\n}\n' % (appid, name, name))
        d = steam / "common" / name / Path(exe).parent
        d.mkdir(parents=True)
        (steam / "common" / name / exe).write_bytes(mockpe.make_dll({"ProductName": "stub game"}, (1, 0, 0, 0), dll_name="game.exe"))
        (d / "nvngx_dlss.dll").write_bytes(mockpe.make_dll({"ProductName": "DLSS stand-in"}, (310, 5, 0, 0), dll_name="nvngx_dlss.dll"))
    return home


def environment(work, home, name):
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "WRONGGPU_UI_SCRIPT")}
    run = work / name
    for d in ("data", "config", "cache", "state", "tmp"):
        (run / d).mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), XDG_DATA_HOME=str(run / "data"), XDG_CONFIG_HOME=str(run / "config"), XDG_CACHE_HOME=str(run / "cache"),
               XDG_STATE_HOME=str(run / "state"), TMPDIR=str(run / "tmp"), WRONGGPU_REGISTRY_FILE=str(run / "registry.ini"),
               SDL_VIDEO_DRIVER="offscreen", SDL_AUDIODRIVER="dummy")
    return env, run


def run(exe, args, env, timeout):
    r = subprocess.run([str(exe), *args], env=env, capture_output=True, text=True, timeout=timeout)
    return r


def test_dump(exe, work, home):
    print("ui-dump")
    env, run_dir = environment(work, home, "dump")
    out = work / "dump"
    r = run(exe, ["ui-dump", f"--dump={out / 'dump.json'}", f"--shots={out / 'shots'}"], env, 300)
    check(r.returncode == 0, f"ui-dump exits 0 (got {r.returncode}: {r.stderr[-300:]})")
    d = json.loads((out / "dump.json").read_text())
    for k in ("version", "first_run", "list", "settings", "games", "banner", "keyboard"):
        check(k in d, f"dump has '{k}'")
    for page in ("first_run", "list", "settings"):
        ctl = d[page]["controls"]
        check(len(ctl) > 5, f"{page} has widgets")
        for c in ctl:
            if not check(c["class"] in ROLES and isinstance(c["id"], int) and all(isinstance(c[k], int) for k in "xywh")
                         and isinstance(c["enabled"], bool), f"widget shape in {page}: {c}"):
                break
    names = {c["text"] for c in d["list"]["controls"]}
    classes = {(c["class"], c["text"]) for c in d["list"]["controls"]}
    for want in ("Settings", "Minimize", "Maximize", "Close"):
        check(want in names, f"Library has '{want}'")
    check(("Edit", "Search games") in classes, "Library has the Search games field")
    check(any(n.startswith("Library") for n in names), "Library has its tab")
    games = d["games"] if isinstance(d["games"], list) else list(d["games"].values())
    check(len(games) == 2, f"two games listed from the fake library (got {len(games)})")
    for i in range(2):
        p = out / "shots" / f"game-{i}.png"
        check(p.is_file() and png_size(p) == (1180, 760), f"game page {i} PNG is 1180x760")
    for p in ("first-run", "list", "settings"):
        check(png_size(out / "shots" / f"{p}.png") == (1180, 760), f"{p}.png is 1180x760")
    return d


def test_render(exe, work, home):
    print("ui-render")
    env, _ = environment(work, home, "render")
    out = work / "render"
    r = run(exe, ["ui-render", f"--out-dir={out}", f"--fixture={HERE / 'ui_fixture.ini'}", "--size=1180x760", "--scales=1,2",
                  "--frames=library,settings,game"], env, 600)
    check(r.returncode == 0, f"ui-render exits 0 (got {r.returncode}: {r.stderr[-300:]})")
    for frame in ("library", "settings", "game"):
        check(png_size(out / f"{frame}@100.png") == (1180, 760), f"{frame}@100.png is 1180x760")
        check(png_size(out / f"{frame}@200.png") == (2360, 1520), f"{frame}@200.png is 2360x1520")
        check((out / f"{frame}@100.png").stat().st_size > 20000, f"{frame}@100.png is not blank")


SCRIPT = """# a click on Settings, a scroll, a click on the Library's search field and typing in it
until games 2
dump {o}/d0.json
shot {o}/s0.png
clickname Settings
wait 500
dump {o}/d1.json
wheel -5
wait 800
dump {o}/d2.json
clickname Library
wait 500
clickname Search games
wait 300
text bal
wait 400
dump {o}/d3.json
shot {o}/s3.png
key ctrl+a
key backspace
wait 400
dump {o}/d4.json
close
"""


def test_live(exe, work, home):
    print("live window (offscreen SDL, scripted events)")
    env, run_dir = environment(work, home, "live")
    (run_dir / "data" / "WrongGPU").mkdir(parents=True, exist_ok=True)
    (run_dir / "data" / "WrongGPU" / "launcher.ini").write_bytes(b"first_run_done=1\r\n")  # no first-run sheet over the page
    out = work / "live"
    out.mkdir(exist_ok=True)
    script = work / "live-script.txt"
    script.write_text(SCRIPT.format(o=out))
    env["WRONGGPU_UI_SCRIPT"] = str(script)
    try:
        r = run(exe, [], env, 180)
    except subprocess.TimeoutExpired:
        check(False, "the window closed on its own within 180 s")
        return
    check(r.returncode == 0, f"the program exits 0 after WM_CLOSE (got {r.returncode}: {r.stderr[-300:]})")
    st = [json.loads((out / f"d{i}.json").read_text()) if (out / f"d{i}.json").is_file() else {} for i in range(5)]
    check(st[0].get("page_name") == "library" and st[0].get("games") == 2, f"starts on the Library with two games ({st[0].get('page_name')}, {st[0].get('games')})")
    check(st[1].get("page_name") == "settings", f"a click on Settings opens Settings ({st[1].get('page_name')})")
    check(st[2].get("scroll", 0) > 100, f"the wheel scrolls the page (scroll {st[2].get('scroll')})")
    check(st[3].get("page_name") == "library" and st[3].get("search") == "bal", f"typing reaches the search field ({st[3].get('page_name')}, '{st[3].get('search')}')")
    check(any(c["class"] == "Item" and c["text"].startswith("Baldur") for c in st[3].get("controls", [])) or
          any("Baldur" in c["text"] for c in st[3].get("controls", [])), "the Library is filtered to Baldur's Gate 3")
    check(not any("Cyberpunk" in c["text"] for c in st[3].get("controls", [])), "the Library no longer shows Cyberpunk 2077")
    check(st[4].get("search") == "", f"Ctrl+A, Backspace empties the field ('{st[4].get('search')}')")
    check(png_size(out / "s0.png") is not None and png_size(out / "s3.png") is not None, "the live frames were saved as PNGs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--only", default="dump,render,live")
    a = ap.parse_args()
    work = Path(a.work or "linux-ui-tests").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    real_data = Path.home() / ".local" / "share" / "WrongGPU"
    had = real_data.exists()
    home = make_home(work)
    for name, fn in (("dump", test_dump), ("render", test_render), ("live", test_live)):
        if name in a.only.split(","):
            fn(Path(a.exe), work, home)
    check(had or not real_data.exists(), "nothing was written to the real ~/.local/share/WrongGPU")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("linux ui tests: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
