# SPDX-License-Identifier: MIT
"""The launcher's window on Linux (the one WrongGPU binary: src/app on the SDL window layer of src/platform/posix).

    run_linux_ui_tests.py --exe <WrongGPU> [--work <dir>]

Every run uses a home of its own under --work, its own XDG folders and registry file, a fake Steam library with two stub games
(PE stand-ins from mockpe.py: nothing of the player's is read or written), and SDL's offscreen video driver (no window opens).
  ui-dump    the pages as JSON: the structure the launcher tests expect (first run, Library, Settings, a page per game; each widget's
             class, text, id, place, state), the games read from the fake library, the PNGs of the pages
  ui-render  the fixture screenshots (Library, Settings, a game's page) at two scales: PNG files of the sizes asked for
  jpeg       ui-render of a game whose Steam cache holds a synthetic JPEG hero: its colours are on the page (needs --jpeg-tool)
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
import zlib
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


def png_pixels(path):
    """(width, height, rows of RGB tuples-as-bytes) of an 8-bit RGB or RGBA PNG, from the standard library alone."""
    b = Path(path).read_bytes()
    pos, idat, w = 8, b"", 0
    while pos < len(b):
        n, kind = struct.unpack(">I4s", b[pos:pos + 8])
        data = b[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert depth == 8 and ctype in (2, 6), "8-bit RGB or RGBA only"
        elif kind == b"IDAT":
            idat += data
        pos += 12 + n
    bpp = 4 if ctype == 6 else 3
    raw, stride, rows, prev = zlib.decompress(idat), w * bpp, [], bytearray(w * bpp)
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            up = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1: line[i] = (line[i] + a) & 255
            elif f == 2: line[i] = (line[i] + up) & 255
            elif f == 3: line[i] = (line[i] + (a + up) // 2) & 255
            elif f == 4:
                pa, pb, pc = abs(up - c), abs(a - c), abs(a + up - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else up if pb <= pc else c)) & 255
        rows.append(bytes(line)); prev = line
    return w, h, rows, bpp


def count_near(png, want, tol):
    """How many pixels of the PNG are within `tol` of the RGB colour `want`."""
    _, _, rows, bpp = png
    n = 0
    for r in rows:
        for x in range(0, len(r), bpp):
            if abs(r[x] - want[0]) <= tol and abs(r[x + 1] - want[1]) <= tol and abs(r[x + 2] - want[2]) <= tol:
                n += 1
    return n


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


def test_jpeg_art(exe, work, home, jpeg_tool):
    """Steam's art is JPEG: a fixture game whose cache holds a synthetic hero (jpeg-linux-tests --make: red, green, blue and white
    blocks) shows it on its page; the same render without the file shows none of those colours."""
    print("ui-render with JPEG art")
    art_home = work / "home-art"
    shutil.copytree(home, art_home)
    cache = art_home / ".local" / "share" / "Steam" / "appcache" / "librarycache" / "1091500"
    cache.mkdir(parents=True)
    r = subprocess.run([str(jpeg_tool), "--make", str(cache / "library_hero.jpg"), "1920", "620"], capture_output=True, text=True)
    check(r.returncode == 0, "the synthetic hero is written")
    shots = {}
    for name, h in (("render-noart", home), ("render-jpeg", art_home)):
        env, _ = environment(work, h, name)
        out = work / name
        r = run(exe, ["ui-render", f"--out-dir={out}", f"--fixture={HERE / 'ui_fixture.ini'}", "--size=1180x760", "--scales=1", "--frames=game"], env, 600)
        check(r.returncode == 0, f"ui-render ({name}) exits 0 (got {r.returncode}: {r.stderr[-300:]})")
        shots[name] = png_pixels(out / "game@100.png")
    # the page's own light background is near the white block, so the green block (top right of the picture) is the check
    green = (32, 200, 40)
    with_art, without = count_near(shots["render-jpeg"], green, 24), count_near(shots["render-noart"], green, 24)
    check(with_art > 20000 and without < 500, f"the hero's green block is on the page with the art ({with_art} pixels) and not without ({without})")


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
    ap.add_argument("--only", default="dump,render,jpeg,live")
    ap.add_argument("--jpeg-tool", default=None, help="jpeg-linux-tests, which writes the synthetic JPEG art")
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
    if "jpeg" in a.only.split(",") and a.jpeg_tool:
        test_jpeg_art(Path(a.exe), work, home, a.jpeg_tool)
    check(had or not real_data.exists(), "nothing was written to the real ~/.local/share/WrongGPU")
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("linux ui tests: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
