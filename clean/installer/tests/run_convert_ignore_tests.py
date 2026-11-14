#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The ignore lists of the converter's file search (src/wg_core/convert_ignore.h) through the launcher's command line:
`model convert` and `rr-data convert` with the well-known lists (%LOCALAPPDATA%\\WrongGPU\\ignore-files.txt and the
launcher's own data folder's), --ignore-list= and --ignore=.

Written for this project; Python standard library only. Runs WrongGPU.exe (a WRONGGPU_TESTING build made with the
installer suite's test map) against GENERATED library folders only: synthetic model files from the converter's test
generator, the synthetic preset-F file, names and bytes of our own. Only the two convert verbs run: no
install, no window, no GPU; --noscan always (the tester's own store libraries are never read).

    python run_convert_ignore_tests.py --launcher <WrongGPU.exe> --work <dir> [--synth-f <folder of synth-common-v0.bin>]

The candidate sets of the auto search are the installer suite's "several DLLs" sets ([13]: the picked file and a game's,
with and without the output-filter block, in both orders; two without it) and the same with library games only, plus a
file that does not convert first. Each set is converted with no list at all, then with an empty list in every place a
list is read: the winner, the model file's bytes and model-source.txt must be the same each time.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["WRONGGPU_TESTING"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_installer_tests as suite  # noqa: E402  (the synthetic model cases, OLD_DLSS)
from run_installer_tests import read, write  # noqa: E402

DLL = "nvngx_dlss.dll"
RR_DLL = "nvngx_dlssd.dll"  # the RR model file's name (a synthetic stand-in here)


class T:
    def __init__(self, exe, work):
        self.exe, self.work = exe, work
        self.passed, self.failures, self.skipped = 0, [], []
        self.data = os.path.join(work, "data")  # --data: the launcher's own data folder (its ignore-files.txt is read too)
        self.log = os.path.join(work, "launcher-test.log")
        self.out = os.path.join(work, "out.json")
        self.lad = os.path.join(work, "localappdata")
        self.env = dict(os.environ, LOCALAPPDATA=self.lad, ProgramData=os.path.join(work, "programdata"))
        for k in [k for k in self.env if k.upper().startswith("WG_") or k.upper() in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH")]:
            self.env.pop(k)
        os.makedirs(self.lad, exist_ok=True)
        self.j, self.stdout = {}, ""

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print("  PASS ", name)
        else:
            self.failures.append(name + (": " + str(detail)[:1500] if detail else ""))
            print("  FAIL ", name, str(detail)[:1500])

    def skip(self, name, why):
        self.skipped.append("%s (%s)" % (name, why))
        print("  SKIP ", name, "-", why)

    def run(self, *args):
        """WrongGPU.exe <args> --noscan --data --log --out; the JSON result in self.j; returns the exit code."""
        if os.path.exists(self.out):
            os.remove(self.out)
        if os.path.exists(self.log):
            os.remove(self.log)
        cmd = [self.exe] + list(args) + ["--noscan", "--log=" + self.log, "--out=" + self.out, "--data=" + self.data]
        p = subprocess.run(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        self.stdout = p.stdout.decode("utf-8", "replace")
        try:
            with open(self.out, encoding="utf-8") as f:
                self.j = json.load(f)
        except (OSError, ValueError):
            self.j = {}
        return p.returncode

    def logtext(self):
        return read(self.log).decode("utf-8", "replace") if os.path.isfile(self.log) else ""

    # the shared data: %LOCALAPPDATA%\WrongGPU (the well-known list) and its model folders
    def shared(self, *rel):
        return os.path.join(self.lad, "WrongGPU", *rel)

    def clear_models(self):
        """The converted model and RR data gone (each run converts again); the lists next to them stay."""
        for rel in (("model", "model.wgcm"), ("model", "model-source.txt"), ("model", "F", "model.wgcm"), ("model", "F", "model-source.txt")):
            p = self.shared(*rel)
            if os.path.isfile(p):
                os.remove(p)

    def set_lists(self, shared=None, own=None):
        """The two well-known lists: None removes the file, a string is written as it is."""
        for path, text in ((self.shared("ignore-files.txt"), shared), (os.path.join(self.data, "ignore-files.txt"), own)):
            if text is None:
                if os.path.isfile(path):
                    os.remove(path)
            else:
                write(path, text.encode("utf-8") if isinstance(text, str) else text)


def library(t, name, files):
    """A generated library folder: one sub-folder per game, each with its DLL (name -> bytes, in that order)."""
    lib = os.path.join(t.work, "libs", name)
    shutil.rmtree(lib, ignore_errors=True)
    paths = []
    for game, (rel, data) in files:
        p = os.path.join(lib, game, rel)
        write(p, data)
        paths.append(p)
    return lib, paths


def model_state(t):
    src = t.shared("model", "model-source.txt")
    m = t.shared("model", "model.wgcm")
    return {"code": None, "model": read(m) if os.path.isfile(m) else b"",
            "source": read(src).decode("utf-8", "replace") if os.path.isfile(src) else ""}


def named_log_has(message, prefix, count):
    """The log a player's message names ("... is in the log: <file>.") exists and holds at least `count` lines with `prefix`."""
    at = message.find("is in the log: ")
    if at < 0:
        return False
    rest = message[at + len("is in the log: "):]
    end = rest.lower().find(".log")
    path = rest[:end + 4] if end >= 0 else ""
    return bool(path) and os.path.isfile(path) and read(path).decode("utf-8", "replace").count(prefix) >= count


def source_lines(text):
    """model-source.txt without its time stamp (converted\\t...)."""
    return [l for l in text.replace("\r", "").split("\n") if l and not l.startswith("converted\t")]


def convert_model(t, lib=None, model=None, extra=(), keep=False):
    """model convert; keep: the model converted earlier stays (the reuse cases), else it is removed first."""
    if not keep:
        t.clear_models()
    args = ["model", "convert"] + (["--library=" + lib] if lib else []) + (["--model=" + model] if model else []) + list(extra)
    code = t.run(*args)
    s = model_state(t)
    s["code"] = code
    s["skipped"] = t.j.get("skipped", [])
    s["warnings"] = t.j.get("ignore_warnings", [])
    s["error"] = t.j.get("error", "")
    s["notes"] = t.j.get("notes", [])
    s["outcome"] = t.j.get("outcome", "")
    return s


def source_of(s):
    for l in source_lines(s["source"]):
        if l.startswith("source\t"):
            return l.split("\t", 1)[1]
    return ""


# ---------------------------------------------------------------------------------------------------- the tests

def test_empty_lists_same_winner(t):
    print("\n[1] an empty list never changes which file wins: the existing candidate sets, no list vs an empty list everywhere")
    with_f, with_f2, none = suite.cases.f_mini_case(present=0), suite.cases.f_mini_case(present=1), suite.cases.f_mini_case(present=None)
    none2 = none.file + b"another synthetic file without the block"
    picked = os.path.join(t.work, "elsewhere", "picked", DLL)
    # (label, the picked file's bytes or None, [(game, bytes)], the expected winner: "picked" or a game's name)
    sets = [
        ("[13] picked file without the block, game's file with it", none.file, [("Game", with_f.file)], "Game"),
        ("[13] picked file with the block, game's file without it", with_f.file, [("Game", none.file)], "picked"),
        ("[13] two files without the block", none.file, [("Game", none2)], "picked"),
        ("library: A without, B with", None, [("A Game", none.file), ("B Game", with_f.file)], "B Game"),
        ("library: A with, B without", None, [("A Game", with_f.file), ("B Game", none.file)], "A Game"),
        ("library: A with, B with variant 2", None, [("A Game", with_f.file), ("B Game", with_f2.file)], "A Game"),
        ("library: two without the block", None, [("A Game", none.file), ("B Game", none2)], "A Game"),
        ("library: a file that does not convert first, then one with the block", None, [("A Game", suite.OLD_DLSS), ("B Game", with_f.file)], "B Game"),
        ("picked file only", with_f.file, [], "picked"),
    ]
    empties = [("no list anywhere", None, None, ()),
               ("%LOCALAPPDATA%\\WrongGPU\\ignore-files.txt empty", "", None, ()),
               ("the launcher's own data folder's ignore-files.txt: comments, version line, blank lines", None,
                "# nothing placed\r\nwronggpu-ignore-files 1\r\n\r\n", ()),
               ("both well-known lists empty and an empty --ignore-list=", "\ufeff", "", ("--ignore-list=" + os.path.join(t.work, "empty-list.txt"),))]
    write(os.path.join(t.work, "empty-list.txt"), b"")
    for label, pick, games, winner in sets:
        lib, paths = library(t, "same-winner", [(g, (DLL, data)) for g, data in games])
        if pick is not None:
            write(picked, pick)
        want = picked if winner == "picked" else os.path.join(lib, winner, DLL)
        results = []
        for elabel, shared, own, extra in empties:
            t.set_lists(shared, own)
            s = convert_model(t, lib if games else None, picked if pick is not None else None, extra)
            results.append((elabel, s))
        base = results[0][1]
        t.check(label + ": converted from the expected file with no list", base["code"] == 0 and source_of(base).lower() == want.lower(),
                (base["code"], source_of(base), base["error"], t.logtext()[-1500:]))
        for elabel, s in results[1:]:
            t.check("   ...%s: the same winner, model file and model-source.txt, nothing skipped, no warning" % elabel,
                    s["code"] == base["code"] and s["model"] == base["model"] and source_lines(s["source"]) == source_lines(base["source"]) and
                    s["skipped"] == [] and s["warnings"] == [], (s["code"], source_of(s), s["skipped"], s["warnings"]))
    t.set_lists(None, None)


def test_model_lists(t):
    print("\n[2] model convert: the lists used, every candidate ignored, a hash alone, a bad line, path normalisation, links")
    with_f, with_f2 = suite.cases.f_mini_case(present=0), suite.cases.f_mini_case(present=1)
    lib, (a, b, c) = library(t, "lists", [("A Game", (DLL, with_f.file)), ("B Game", (DLL, with_f2.file)), ("C Game", (DLL, with_f.file + b" synthetic tail"))])
    shared_list = t.shared("ignore-files.txt")
    own_list = os.path.join(t.data, "ignore-files.txt")

    t.set_lists(None, None)
    s = convert_model(t, lib)
    t.check("no list: the first library game's file wins (A)", s["code"] == 0 and source_of(s).lower() == a.lower(), (s["code"], source_of(s), s["error"]))

    # the well-known list in %LOCALAPPDATA%\WrongGPU: an integrator's installer names the file it placed
    t.set_lists("wronggpu-ignore-files 1\r\n# placed by another program's installer\r\n%s\r\n" % a, None)
    s = convert_model(t, lib)
    log = t.logtext()
    t.check("%LOCALAPPDATA%\\WrongGPU\\ignore-files.txt: A skipped, B converted",
            s["code"] == 0 and source_of(s).lower() == b.lower(), (s["code"], source_of(s), s["error"], log[-1500:]))
    t.check("...the result names the skipped file, the list (with its line) and the path",
            len(s["skipped"]) == 1 and s["skipped"][0].lower().startswith(a.lower() + ": on the ignore list " + shared_list.lower() + ", line 3") and
            "(path " in s["skipped"][0], s["skipped"])
    t.check("...the log and the console name it too", ("model: skipped " + a).lower() in log.lower() and ("skipped: " + a).lower() in t.stdout.lower(),
            (log[-1500:], t.stdout[-800:]))
    t.check("...the skipped file is not tried (no converter line for it) and not in model-source.txt",
            ("model: rejected " + a).lower() not in log.lower() and a.lower() not in s["source"].lower(), s["source"])

    # the launcher's own data folder's list (a portable launcher's WrongGPU-data; here the --data folder)
    t.set_lists(None, "%s\r\n" % a)
    s = convert_model(t, lib)
    t.check("the launcher's own data folder's ignore-files.txt: A skipped, B converted, the list named",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1 and own_list.lower() in s["skipped"][0].lower(), (s["code"], source_of(s), s["skipped"]))

    # the command line's lists add to the well-known ones
    cli_list = os.path.join(t.work, "cli-list.txt")
    write(cli_list, ("%s\n" % b).encode())
    t.set_lists("%s\r\n" % a, None)
    s = convert_model(t, lib, extra=("--ignore-list=" + cli_list,))
    t.check("--ignore-list= adds to the well-known list: A and B skipped, C converted",
            s["code"] == 0 and source_of(s).lower() == c.lower() and len(s["skipped"]) == 2 and cli_list.lower() in s["skipped"][1].lower(), (s["code"], source_of(s), s["skipped"]))
    s = convert_model(t, lib, extra=("--ignore=" + b,))
    t.check("--ignore=<path> adds to the well-known list: A and B skipped, C converted, '--ignore' named",
            s["code"] == 0 and source_of(s).lower() == c.lower() and len(s["skipped"]) == 2 and "--ignore" in s["skipped"][1], (s["code"], source_of(s), s["skipped"]))

    # every candidate ignored
    before = model_state(t)
    s = convert_model(t, lib, extra=("--ignore=" + b, "--ignore=" + c))
    t.check("every candidate ignored: exit 5 (no model), nothing converted or written", s["code"] == 5 and s["model"] == b"" and s["source"] == "",
            (s["code"], len(s["model"]), s["error"]))
    t.check("...a clear message: every file found is on an ignore list, and all three are named",
            "Every DLSS model file found (3) is on an ignore list, so none was converted" in s["error"] and len(s["skipped"]) == 3 and
            "Every DLSS model file found (3) is on an ignore list" in t.stdout, (s["error"], s["skipped"], t.stdout[-600:]))
    t.check("...the message itself names the first file skipped with its list and line, and points to the log for the other two",
            ("Skipped: " + s["skipped"][0] + ".") in s["error"] and (shared_list + ", line 1").lower() in s["error"].lower() and
            "2 other files were skipped too" in s["error"] and named_log_has(s["error"], "model: skipped ", len(s["skipped"])) and
            "remove that line" in s["error"], (s["error"], s["skipped"]))
    del before

    # a hash alone is not a list entry (it would skip every identical copy, the player's own too): warned, nothing skipped
    sha_a = hashlib.sha256(with_f.file).hexdigest()
    t.set_lists("sha256=%s\r\n" % sha_a, None)
    s = convert_model(t, lib)
    t.check("a hash alone in a list: warned (list, line, why), skipped; nothing is left out, A converts",
            s["code"] == 0 and source_of(s).lower() == a.lower() and s["skipped"] == [] and len(s["warnings"]) == 1 and
            (shared_list + ", line 1").lower() in s["warnings"][0].lower() and "hash alone is not accepted in a list" in s["warnings"][0], (s["code"], source_of(s), s["skipped"], s["warnings"]))
    # --ignore=sha256=<hex> (the command line only): the file is skipped wherever it is, under any name
    t.set_lists(None, None)
    s = convert_model(t, lib, extra=("--ignore=sha256=" + sha_a,))
    t.check("--ignore=sha256=: A (that hash) skipped, B converted; the reason names the argument and the hash",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1 and ("SHA-256 " + sha_a) in s["skipped"][0] and "--ignore=sha256=" in s["skipped"][0],
            (s["code"], source_of(s), s["skipped"]))
    copy = os.path.join(t.work, "elsewhere", "renamed", DLL)
    write(copy, with_f.file)
    s = convert_model(t, lib, model=copy, extra=("--ignore=sha256=" + sha_a,))
    t.check("...the same bytes chosen with --model= from another folder: skipped too, B converted",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 2 and s["skipped"][0].lower().startswith(copy.lower()), (s["code"], source_of(s), s["skipped"]))

    # a path + hash entry matches only when BOTH match; an identical copy under another path is still converted
    t.set_lists("%s  sha256=%s\r\n" % (a, sha_a), None)
    s = convert_model(t, lib)
    t.check("a path + hash entry (both match): A skipped, B converted; the reason names the path and the hash",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1 and "(path %s; SHA-256 %s)" % (a, sha_a) in s["skipped"][0], (s["code"], source_of(s), s["skipped"]))
    s = convert_model(t, lib, model=copy)
    t.check("...the same bytes at another path (--model=): NOT skipped by the path + hash entry, converted (it is the first candidate)",
            s["code"] == 0 and source_of(s).lower() == copy.lower() and len(s["skipped"]) == 1 and s["skipped"][0].lower().startswith(a.lower()), (s["code"], source_of(s), s["skipped"]))
    t.set_lists("%s  sha256=%s\r\n" % (a, hashlib.sha256(with_f2.file).hexdigest()), None)
    s = convert_model(t, lib)
    t.check("a path + hash entry whose hash is not the file's: nothing skipped, A converts", s["code"] == 0 and source_of(s).lower() == a.lower() and s["skipped"] == [],
            (s["code"], source_of(s), s["skipped"]))
    # two identical stand-ins in two folders, only one listed by path + hash: the other is still converted
    twin_lib, (t1, t2) = library(t, "twins", [("Twin One", (DLL, with_f.file)), ("Twin Two", (DLL, with_f.file))])
    t.set_lists("%s  sha256=%s\r\n" % (t1, sha_a), None)
    s = convert_model(t, twin_lib)
    t.check("two identical files, one listed by path + hash: the listed one is skipped, the identical other one is converted",
            s["code"] == 0 and source_of(s).lower() == t2.lower() and len(s["skipped"]) == 1 and s["skipped"][0].lower().startswith(t1.lower()), (s["code"], source_of(s), s["skipped"]))
    t.set_lists("%s\r\n" % t1, None)
    s = convert_model(t, twin_lib, extra=("--ignore=" + t2,))
    t.check("...and both listed (by path, one from the command line): exit 5, nothing converted", s["code"] == 5 and s["model"] == b"" and len(s["skipped"]) == 2, (s["code"], s["error"], s["skipped"]))

    # a folder entry is warned about like any bad line (it would ignore nothing)
    t.set_lists("%s\r\n%s\r\n" % (os.path.dirname(a), a), None)
    s = convert_model(t, lib)
    t.check("a folder in a list: warned (list, line, folder), the file line after it still used (A skipped, B converted)",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["warnings"]) == 1 and "folder" in s["warnings"][0] and (shared_list + ", line 1").lower() in s["warnings"][0].lower(),
            (s["code"], source_of(s), s["warnings"]))

    # a bad line: warned (with its list and line), skipped; the rest of the list is used
    t.set_lists("this is not a path\r\nsha256=12345\r\n%s\r\n" % a, None)
    s = convert_model(t, lib)
    t.check("bad lines: each warned with the list and the line, the good line still used (A skipped, B converted)",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["warnings"]) == 2 and (shared_list + ", line 1").lower() in s["warnings"][0].lower() and
            ", line 2" in s["warnings"][1] and len(s["skipped"]) == 1, (s["code"], source_of(s), s["warnings"], s["skipped"]))
    t.check("...the warnings are in the log and on the console", "model: warning: " in t.logtext() and "ignore list warning: " in t.stdout, t.stdout[-800:])

    # path normalisation: case, slashes, '.' and '..'
    odd = a.lower().replace("\\", "/").replace("/a game/", "/a game/./bin/../")
    t.set_lists("%s\r\n" % odd, None)
    s = convert_model(t, lib)
    t.check("a listed path in lower case with '/', '.' and '..': A skipped, B converted (%s)" % odd,
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1, (s["code"], source_of(s), s["skipped"], s["warnings"]))
    t.set_lists('"%s"\r\n' % a.upper(), None)
    s = convert_model(t, lib)
    t.check("a listed path in upper case, in quotes: A skipped", s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1,
            (s["code"], source_of(s), s["skipped"], s["warnings"]))

    # a junction to the listed file's folder: the candidate's fully resolved path is the listed file
    junction = os.path.join(t.work, "junction-to-a")
    if os.path.lexists(junction):
        os.rmdir(junction)
    r = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", junction, os.path.dirname(a)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode == 0 and os.path.isfile(os.path.join(junction, DLL)):
        t.set_lists("%s\r\n" % a, None)
        s = convert_model(t, lib, model=os.path.join(junction, DLL))
        t.check("--model= through a junction to the listed file's folder: skipped (resolved to the listed file), B converted",
                s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 2 and s["skipped"][0].lower().startswith(junction.lower()),
                (s["code"], source_of(s), s["skipped"]))
        t.set_lists("%s\r\n" % os.path.join(junction, DLL), None)
        s = convert_model(t, lib)
        t.check("a path listed through the junction: the file itself (A) skipped", s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1,
                (s["code"], source_of(s), s["skipped"]))
        os.rmdir(junction)  # the link only
    else:
        t.skip("junction cases", "mklink /J failed: %s" % r.stderr.decode("utf-8", "replace").strip())

    # a symbolic link in a library game, named as the DLL, pointing at the listed file kept elsewhere
    real = os.path.join(t.work, "elsewhere", "placed", "real-file.dll")
    write(real, with_f.file)
    lib2 = os.path.join(t.work, "libs", "symlink")
    shutil.rmtree(lib2, ignore_errors=True)
    link = os.path.join(lib2, "A Game", DLL)
    os.makedirs(os.path.dirname(link))
    write(os.path.join(lib2, "B Game", DLL), with_f2.file)
    try:
        os.symlink(real, link)
        made = True
    except OSError as e:
        made, why = False, str(e)
    if made:
        t.set_lists("%s\r\n" % real, None)
        s = convert_model(t, lib2)
        t.check("a symbolic link in a game pointing at the listed file: skipped, B converted",
                s["code"] == 0 and source_of(s).lower() == os.path.join(lib2, "B Game", DLL).lower() and len(s["skipped"]) == 1 and s["skipped"][0].lower().startswith(link.lower()),
                (s["code"], source_of(s), s["skipped"]))
        os.remove(link)  # the link only
    else:
        t.skip("symbolic link case", "this account may not make a symbolic link: " + why)

    # hard link: the same file under another path
    hard = os.path.join(t.work, "elsewhere", "hard", DLL)
    if os.path.exists(hard):
        os.remove(hard)
    os.makedirs(os.path.dirname(hard), exist_ok=True)
    os.link(a, hard)
    t.set_lists("%s\r\n" % a, None)
    s = convert_model(t, lib, model=hard)
    t.check("--model= a hard link to the listed file: skipped as the same file, B converted",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 2 and "the same file" in s["skipped"][0], (s["code"], source_of(s), s["skipped"]))
    os.remove(hard)

    # UNC: the drive's administrative share on this PC (read only; nothing is shared or changed). Not reachable: said, skipped.
    unc = "\\\\localhost\\%s$%s" % (a[0], a[2:]) if len(a) > 2 and a[1] == ":" else ""
    if unc and os.path.isfile(unc):
        t.set_lists("%s\r\n" % a, None)
        s = convert_model(t, lib, model=unc)
        t.check("--model= the listed file through \\\\localhost\\%s$ (UNC): skipped as the same file, B converted" % a[0],
                s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 2 and s["skipped"][0].lower().startswith(unc.lower()), (s["code"], source_of(s), s["skipped"]))
        t.set_lists("%s\r\n" % unc, None)
        s = convert_model(t, lib)
        t.check("a UNC entry for the file: the file found by its local path (A) skipped", s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1,
                (s["code"], source_of(s), s["skipped"]))
        t.set_lists("\\\\?\\UNC\\localhost\\%s$%s\r\n" % (a[0], a[2:]), None)
        s = convert_model(t, lib)
        t.check("a \\\\?\\UNC\\ entry for the file: A skipped", s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1, (s["code"], source_of(s), s["skipped"]))
    else:
        t.skip("UNC cases", "%s is not reachable here" % (unc or "a local drive path"))

    # the command line's own mistakes stop it before anything is converted
    t.set_lists(None, None)
    s = convert_model(t, lib, extra=("--ignore-list=" + os.path.join(t.work, "no-such-list.txt"),))
    t.check("--ignore-list= of a file that isn't there: exit 10, nothing converted", s["code"] == 10 and s["model"] == b"" and "no-such-list.txt" in s["error"],
            (s["code"], s["error"]))
    s = convert_model(t, lib, extra=("--ignore=not a path",))
    t.check("--ignore= that is not a path or a hash: exit 10, nothing converted", s["code"] == 10 and s["model"] == b"" and "not a full path" in s["error"],
            (s["code"], s["error"]))
    s = convert_model(t, lib, extra=("--ignore", b))
    t.check("a bare --ignore (no '='): exit 10, nothing converted; it does not take the next argument as its value",
            s["code"] == 10 and s["model"] == b"" and "--ignore needs its entry after an equals sign" in s["error"], (s["code"], s["error"]))
    s = convert_model(t, lib, extra=("--ignore-list",))
    t.check("a bare --ignore-list followed by another option: exit 10, nothing converted; it does not take that option as its file",
            s["code"] == 10 and s["model"] == b"" and "--ignore-list needs its file" in s["error"], (s["code"], s["error"]))
    s = convert_model(t, lib, extra=("--ignore-list", cli_list))
    t.check("a bare --ignore-list followed by a file (no '='): exit 10, nothing converted; the file is not taken as its value",
            s["code"] == 10 and s["model"] == b"" and "--ignore-list needs its file" in s["error"], (s["code"], s["error"]))
    # --ignore / --ignore-list are applied by model convert and rr-data convert only: elsewhere exit 10 and a message
    for verb_args in (["scan"], ["model", "status"], ["rr-data", "status"], ["hipcheck"]):
        for arg in ("--ignore=" + b, "--ignore-list=" + cli_list):
            t.clear_models()
            code = t.run(*(verb_args + [arg]))
            t.check("%s %s: exit 10, not run, the message says where they apply" % (" ".join(verb_args), arg.split("=")[0]),
                    code == 10 and "apply only to" in t.j.get("error", "") and "apply only to" in t.stdout, (code, t.j.get("error"), t.stdout[-300:]))
    # the chosen file alone, on a list: the message names the argument (or the list and the line)
    s = convert_model(t, None, model=a, extra=("--ignore=" + a,))
    t.check("--model= of a file named by --ignore= with nothing else found: exit 5, nothing converted; the message names the file and the --ignore= argument",
            s["code"] == 5 and s["model"] == b"" and len(s["skipped"]) == 1 and
            ("The file you chose (%s) is not converted: it is named by --ignore=%s" % (a, a)).lower() in s["error"].lower(), (s["code"], s["error"], s["skipped"]))
    t.set_lists("# placed by another installer\r\n%s\r\n" % a, None)
    s = convert_model(t, lib, model=a)
    t.check("--model= of a listed file with other files found: B converted; the player's line names the list and line 2 and how to undo a mistake",
            s["code"] == 0 and source_of(s).lower() == b.lower() and
            any(("The file you chose (%s) is not converted: it is on the ignore list %s, line 2" % (a, shared_list)).lower() in n.lower() and "remove that line" in n
                for n in s["notes"]), (s["code"], source_of(s), s["notes"]))
    t.check("...the source file's SHA-256 is recorded in model-source.txt", ("sha256\t" + hashlib.sha256(with_f2.file).hexdigest()) in s["source"], s["source"])
    t.set_lists(None, None)


def synth_f_dll(folder):
    """The synthetic preset-F source of the test build's F map (as run_launcher_tests.py synth_f_dll): synthetic bytes only."""
    import struct
    n = 6386372 // 2
    tags = [struct.pack("<e", float(v)) for v in range(16)]
    return b"".join(tags[e & 15] for e in range(n)) + read(os.path.join(folder, "synth-common-v0.bin"))


def convert_rr(t, lib=None, dll=None, extra=()):
    t.clear_models()
    args = ["rr-data", "convert"] + (["--library=" + lib] if lib else []) + (["--dll=" + dll] if dll else []) + list(extra)
    code = t.run(*args)
    rd = t.j.get("rr_data", {})
    return {"code": code, "source": rd.get("source", "") if code == 0 else "", "valid": rd.get("valid"), "skipped": t.j.get("skipped", []),
            "warnings": t.j.get("ignore_warnings", []), "error": t.j.get("error", ""),
            "data": read(t.shared("model", "F", "model.wgcm")) if os.path.isfile(t.shared("model", "F", "model.wgcm")) else b""}


def test_rr_lists(t, synth_f):
    print("\n[3] rr-data convert: an empty list (same winner), the lists used, every candidate ignored, a hash alone, the chosen file")
    if not synth_f or not os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        t.skip("rr-data cases", "no synthetic preset-F file (--synth-f)")
        return
    src = synth_f_dll(synth_f)
    src2 = src + b"synthetic tail of a second RR stand-in"
    lib, (a, b) = library(t, "rr", [("A Game", (RR_DLL, src)), ("B Game", (RR_DLL, src2))])
    t.set_lists(None, None)
    base = convert_rr(t, lib)
    t.check("no list: the first library game's RR file wins (A)", base["code"] == 0 and base["source"].lower() == a.lower() and base["valid"] is True,
            (base["code"], base["source"], base["error"], t.logtext()[-1500:]))
    for elabel, shared, own in (("an empty %LOCALAPPDATA% list", "", None), ("a comments-only list in the data folder", None, "# none\r\n")):
        t.set_lists(shared, own)
        s = convert_rr(t, lib)
        t.check("   ...%s: the same winner and the same RR data, nothing skipped" % elabel,
                s["code"] == 0 and s["source"] == base["source"] and s["data"] == base["data"] and s["skipped"] == [] and s["warnings"] == [], (s["code"], s["source"], s["skipped"]))
    t.set_lists("%s\r\n" % a, None)
    s = convert_rr(t, lib)
    log = t.logtext()
    t.check("the well-known list names A: A skipped (named with the list and line), B converted",
            s["code"] == 0 and s["source"].lower() == b.lower() and len(s["skipped"]) == 1 and (t.shared("ignore-files.txt") + ", line 1").lower() in s["skipped"][0].lower() and
            ("RR data: skipped " + a).lower() in log.lower(), (s["code"], s["source"], s["skipped"], log[-1500:]))
    s = convert_rr(t, lib, extra=("--ignore=" + b,))
    t.check("every candidate ignored: exit 5, no RR data written, a clear message",
            s["code"] == 5 and s["data"] == b"" and "every Ray Reconstruction model file found (2) is on an ignore list" in s["error"] and len(s["skipped"]) == 2,
            (s["code"], s["error"], s["skipped"]))
    t.check("...the message names the first file skipped with its list and line, and points to the log for the other",
            ("Skipped: " + s["skipped"][0]) in s["error"] and (t.shared("ignore-files.txt") + ", line 1").lower() in s["error"].lower() and
            "1 other file was skipped too" in s["error"] and named_log_has(s["error"], "RR data: skipped ", len(s["skipped"])), (s["error"], s["skipped"]))
    t.set_lists(None, None)
    s = convert_rr(t, lib, extra=("--ignore=sha256=" + hashlib.sha256(src).hexdigest(),))
    t.check("--ignore=sha256=: A skipped by its SHA-256, B converted", s["code"] == 0 and s["source"].lower() == b.lower() and "SHA-256 " in "".join(s["skipped"]),
            (s["code"], s["source"], s["skipped"]))
    t.set_lists("%s  sha256=%s\r\n" % (a, hashlib.sha256(src).hexdigest()), None)
    s = convert_rr(t, lib)
    t.check("a path + hash entry (both match): A skipped, B converted", s["code"] == 0 and s["source"].lower() == b.lower() and "; SHA-256 " in "".join(s["skipped"]),
            (s["code"], s["source"], s["skipped"]))
    twin_lib, (t1, t2) = library(t, "rr-twins", [("Twin One", (RR_DLL, src)), ("Twin Two", (RR_DLL, src))])
    t.set_lists("%s  sha256=%s\r\n" % (t1, hashlib.sha256(src).hexdigest()), None)
    s = convert_rr(t, twin_lib)
    t.check("two identical RR files, one listed by path + hash: the identical other one is converted", s["code"] == 0 and s["source"].lower() == t2.lower() and len(s["skipped"]) == 1,
            (s["code"], s["source"], s["skipped"]))
    t.set_lists("sha256=%s\r\n" % hashlib.sha256(src).hexdigest(), None)
    s = convert_rr(t, lib)
    t.check("a hash alone in a list: warned, nothing skipped (A converts)", s["code"] == 0 and s["source"].lower() == a.lower() and s["skipped"] == [] and len(s["warnings"]) == 1,
            (s["code"], s["source"], s["skipped"], s["warnings"]))
    t.set_lists("%s\r\nnot a path\r\n" % a, None)
    s = convert_rr(t, dll=a)
    t.check("--dll= of a listed file: exit 5, nothing converted (never another file instead); the message names the list and the line; the bad line warned",
            s["code"] == 5 and s["data"] == b"" and ("the file you chose (%s) is not converted: it is on the ignore list %s, line 1" % (a, t.shared("ignore-files.txt"))).lower() in s["error"].lower() and
            "remove that line" in s["error"] and len(s["warnings"]) == 1, (s["code"], s["error"], s["warnings"]))
    t.set_lists(None, None)
    s = convert_rr(t, dll=a, extra=("--ignore=" + a,))
    t.check("--dll= of a file named by --ignore=: the message names the argument", s["code"] == 5 and ("named by --ignore=" + a).lower() in s["error"].lower(), (s["code"], s["error"]))
    s = convert_rr(t, lib)
    t.check("the RR data's model-source.txt records the source file's SHA-256",
            s["code"] == 0 and ("sha256\t" + hashlib.sha256(src).hexdigest()) in read(t.shared("model", "F", "model-source.txt")).decode(), read(t.shared("model", "F", "model-source.txt")))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strip_sha(path):
    """An earlier version's model-source.txt: the same file without its sha256 line."""
    text = read(path).decode("utf-8")
    write(path, "".join(l for l in text.splitlines(True) if not l.startswith("sha256\t")).encode("utf-8"))


def test_model_reuse(t):
    print("\n[4] model data converted earlier: its recorded source checked against the lists before it is reused")
    with_f, with_f2, none = suite.cases.f_mini_case(present=0), suite.cases.f_mini_case(present=1), suite.cases.f_mini_case(present=None)
    lib, (a, _b) = library(t, "reuse", [("A Game", (DLL, with_f.file)), ("B Game", (DLL, with_f2.file))])
    lib_b, (b,) = library(t, "reuse-allowed", [("B Game", (DLL, with_f2.file))])
    lib_n, (n,) = library(t, "reuse-allowed-without-weights", [("N Game", (DLL, none.file))])
    src_file = t.shared("model", "model-source.txt")
    shared_list = t.shared("ignore-files.txt")
    unrelated = os.path.join(t.work, "elsewhere", "unrelated", DLL)

    def first():
        t.set_lists(None, None)
        s = convert_model(t, lib)
        assert s["code"] == 0 and source_of(s).lower() == a.lower(), (s["code"], source_of(s), s["error"])
        return s

    s = first()
    t.check("a new conversion records its source's SHA-256 in model-source.txt", ("sha256\t" + sha(with_f.file)) in s["source"], s["source"])
    s = convert_model(t, None, keep=True)
    t.check("no list: the model data on this PC is reused as before", s["code"] == 0 and s["outcome"].startswith("reused"), (s["code"], s["outcome"]))
    t.set_lists("%s\r\n" % unrelated, None)
    s = convert_model(t, None, keep=True)
    t.check("a list that doesn't name its source (SHA-256 on record): reused", s["code"] == 0 and s["outcome"].startswith("reused"), (s["code"], s["outcome"]))

    before = first()
    t.set_lists(None, None)
    s = convert_model(t, lib_b, keep=True, extra=("--ignore=sha256=" + sha(with_f.file),))
    t.check("its source's SHA-256 named by --ignore=sha256=, an allowed file found: not reused, converted again from the allowed file (B)",
            s["code"] == 0 and source_of(s).lower() == b.lower() and s["model"] != before["model"] and ("sha256\t" + sha(with_f2.file)) in s["source"] and
            s["outcome"].lower().startswith(("replaced: it was converted from %s, which is named by --ignore=sha256=%s (SHA-256 " % (a, sha(with_f.file))).lower()),
            (s["code"], source_of(s), s["outcome"]))
    first()
    t.set_lists("%s  sha256=%s\r\n" % (a, sha(with_f.file)), None)
    s = convert_model(t, lib_b, keep=True)
    t.check("its recorded path and recorded SHA-256 both match a path + hash entry: not reused, converted again from B",
            s["code"] == 0 and source_of(s).lower() == b.lower() and s["outcome"].startswith("replaced: it was converted from") and "; SHA-256 " in s["outcome"],
            (s["code"], source_of(s), s["outcome"]))
    first()
    t.set_lists("%s  sha256=%s\r\n" % (a, sha(with_f2.file)), None)
    s = convert_model(t, None, keep=True)
    t.check("a path + hash entry for its source path with another hash (an identical-path file with other bytes): the model data is reused",
            s["code"] == 0 and s["outcome"].startswith("reused"), (s["code"], s["outcome"]))
    t.set_lists("%s  sha256=%s\r\n" % (unrelated, sha(with_f.file)), None)
    s = convert_model(t, None, keep=True)
    t.check("a path + hash entry with its source's hash but another path (a copy placed elsewhere): the model data converted from the legitimate file is reused",
            s["code"] == 0 and s["outcome"].startswith("reused"), (s["code"], s["outcome"]))
    # the recorded source is matched as it was recorded: its path made a link to a listed file (same bytes) after the conversion
    placed = os.path.join(t.work, "elsewhere", "placed", DLL)
    write(placed, with_f.file)
    for kind in ("hard link", "symbolic link"):
        first()
        os.remove(a)
        try:
            os.link(placed, a) if kind == "hard link" else os.symlink(placed, a)
        except OSError as e:
            write(a, with_f.file)
            t.skip("model reuse: source path made a %s" % kind, "not made here (%s)" % e)
            continue
        for entry in ("%s\r\n" % placed, "%s  sha256=%s\r\n" % (placed, sha(with_f.file))):
            t.set_lists(entry, None)
            s = convert_model(t, None, keep=True)
            t.check("its source path made a %s to a listed file with the same bytes after the conversion (%s): the model data is still reused" %
                    (kind, "path + hash entry" if "sha256=" in entry else "path entry"), s["code"] == 0 and s["outcome"].startswith("reused"), (s["code"], s["outcome"], s["error"]))
        os.remove(a)
        write(a, with_f.file)
    os.remove(placed)
    first()
    t.set_lists("%s\r\n" % a, None)
    s = convert_model(t, lib_n, keep=True)
    t.check("its source's path listed, the only allowed file has no output-filter weights: converted again from it all the same",
            s["code"] == 0 and source_of(s).lower() == n.lower() and s["outcome"].startswith("replaced: it was converted from") and "(path " in s["outcome"],
            (s["code"], source_of(s), s["outcome"], s["error"]))
    before = first()
    t.set_lists("%s\r\n" % a, None)
    s = convert_model(t, None, keep=True)
    t.check("its source listed, no allowed file: exit 5, not used; the message says why (the list, the line) and that nothing else was found",
            s["code"] == 5 and ("is not used: it was converted from %s, which is on the ignore list %s, line 1" % (a, shared_list)).lower() in s["error"].lower() and
            "No other DLSS model file that WrongGPU may convert was found" in s["error"] and
            "choose the nvngx_dlss.dll that came with one of your own games" in s["error"], (s["code"], s["error"]))
    t.check("...nothing was written over it (the launcher only decides; other games' runtime files are not touched)", s["model"] == before["model"], len(s["model"]))

    first()
    strip_sha(src_file)
    t.set_lists("%s\r\n" % unrelated, None)
    s = convert_model(t, lib_b, keep=True)
    t.check("an earlier conversion without a recorded SHA-256, a list in use, an allowed file found: converted again from it",
            s["code"] == 0 and source_of(s).lower() == b.lower() and "is not on record" in s["outcome"] and ("sha256\t" + sha(with_f2.file)) in s["source"],
            (s["code"], source_of(s), s["outcome"]))
    before = first()
    strip_sha(src_file)
    t.set_lists("%s\r\n" % unrelated, None)
    s = convert_model(t, None, keep=True)
    t.check("...no allowed file found: kept (exit 0) with a warning for the player",
            s["code"] == 0 and s["outcome"].startswith("kept: the SHA-256 of the file it was converted from is not on record") and s["model"] == before["model"] and
            any("did not record the SHA-256" in x for x in s["notes"]), (s["code"], s["outcome"], s["notes"]))
    first()
    strip_sha(src_file)
    t.set_lists(None, None)
    s = convert_model(t, None, keep=True)
    t.check("...no list at all: reused as before (an empty list changes nothing)", s["code"] == 0 and s["outcome"].startswith("reused") and s["notes"] == [], (s["code"], s["outcome"], s["notes"]))
    t.set_lists(None, None)


def prepare_rr(t, lib=None):
    code = t.run("rr-data", "prepare", *(["--library=" + lib] if lib else []))
    p = t.shared("model", "F", "model-source.txt")
    text = read(p).decode("utf-8", "replace") if os.path.isfile(p) else ""
    src = next((l.split("\t", 1)[1] for l in text.replace("\r", "").split("\n") if l.startswith("source\t")), "")
    return {"code": code, "reused": t.j.get("reused"), "notes": t.j.get("notes", []), "error": t.j.get("error", ""), "source": src, "text": text,
            "data": read(t.shared("model", "F", "model.wgcm")) if os.path.isfile(t.shared("model", "F", "model.wgcm")) else b""}


def test_rr_reuse(t, synth_f):
    print("\n[5] RR data converted earlier: checked against the lists before an install reuses it (rr-data prepare: the install's step)")
    if not synth_f or not os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        t.skip("RR reuse cases", "no synthetic preset-F file (--synth-f)")
        return
    src = synth_f_dll(synth_f)
    src2 = src + b"synthetic tail of a second RR stand-in"
    lib, (a,) = library(t, "rr-reuse", [("A Game", (RR_DLL, src))])
    lib_b, (b,) = library(t, "rr-reuse-allowed", [("B Game", (RR_DLL, src2))])
    src_file = t.shared("model", "F", "model-source.txt")
    unrelated = os.path.join(t.work, "elsewhere", "unrelated", RR_DLL)

    def first():
        t.set_lists(None, None)
        s = convert_rr(t, lib)
        assert s["code"] == 0 and s["source"].lower() == a.lower(), (s["code"], s["error"])

    first()
    s = prepare_rr(t)
    t.check("no list: the RR data on this PC is reused", s["code"] == 0 and s["reused"] is True and s["source"].lower() == a.lower(), (s["code"], s["reused"], s["error"]))
    t.set_lists("%s\r\n" % unrelated, None)
    s = prepare_rr(t)
    t.check("a list that doesn't name its source: reused", s["code"] == 0 and s["reused"] is True, (s["code"], s["reused"], s["error"]))
    t.set_lists("%s  sha256=%s\r\n" % (a, sha(src)), None)
    s = prepare_rr(t, lib_b)
    t.check("its recorded path and SHA-256 both match a path + hash entry, an allowed file found: converted again from it (B), its SHA-256 recorded",
            s["code"] == 0 and s["reused"] is False and s["source"].lower() == b.lower() and ("sha256\t" + sha(src2)) in s["text"], (s["code"], s["reused"], s["source"], s["error"]))
    first()
    t.set_lists("%s  sha256=%s\r\n" % (unrelated, sha(src)), None)
    s = prepare_rr(t, lib_b)
    t.check("a path + hash entry with its source's hash but another path: the RR data is reused", s["code"] == 0 and s["reused"] is True and s["source"].lower() == a.lower(),
            (s["code"], s["reused"], s["source"], s["error"]))
    # the recorded source is matched as it was recorded: its path made a link to a listed file (same bytes) after the conversion
    placed = os.path.join(t.work, "elsewhere", "placed-rr", RR_DLL)
    write(placed, src)
    for kind in ("hard link", "symbolic link"):
        first()
        os.remove(a)
        try:
            os.link(placed, a) if kind == "hard link" else os.symlink(placed, a)
        except OSError as e:
            write(a, src)
            t.skip("RR reuse: source path made a %s" % kind, "not made here (%s)" % e)
            continue
        for entry in ("%s\r\n" % placed, "%s  sha256=%s\r\n" % (placed, sha(src))):
            t.set_lists(entry, None)
            s = prepare_rr(t)
            t.check("its source path made a %s to a listed file with the same bytes after the conversion (%s): the RR data is still reused" %
                    (kind, "path + hash entry" if "sha256=" in entry else "path entry"), s["code"] == 0 and s["reused"] is True and s["source"].lower() == a.lower(),
                    (s["code"], s["reused"], s["error"]))
        os.remove(a)
        write(a, src)
    os.remove(placed)
    first()
    before = read(t.shared("model", "F", "model.wgcm"))
    t.set_lists("%s\r\n" % a, None)
    s = prepare_rr(t)
    t.check("its source's path listed, no allowed file: exit 5, not used, the reason names the list; nothing written over it",
            s["code"] == 5 and ("the Ray Reconstruction data on this PC is not used: it was converted from " + a).lower() in s["error"].lower() and
            t.shared("ignore-files.txt").lower() in s["error"].lower() and
            "choose the Ray Reconstruction model file (nvngx_dlssd.dll) that came with one of your own games" in s["error"] and s["data"] == before, (s["code"], s["error"]))
    first()
    strip_sha(src_file)
    t.set_lists("%s\r\n" % unrelated, None)
    s = prepare_rr(t, lib_b)
    t.check("no SHA-256 on record, a list in use, an allowed file found: converted again from it", s["code"] == 0 and s["reused"] is False and s["source"].lower() == b.lower(),
            (s["code"], s["reused"], s["source"], s["error"]))
    first()
    strip_sha(src_file)
    t.set_lists("%s\r\n" % unrelated, None)
    s = prepare_rr(t)
    t.check("...no allowed file: kept, with the player's line", s["code"] == 0 and s["reused"] is True and any("did not record the SHA-256" in x for x in s["notes"]),
            (s["code"], s["reused"], s["notes"]))
    first()
    strip_sha(src_file)
    t.set_lists(None, None)
    s = prepare_rr(t)
    t.check("...no list at all: reused as before, no line", s["code"] == 0 and s["reused"] is True and s["notes"] == [], (s["code"], s["reused"], s["notes"]))
    t.set_lists(None, None)


class Held:
    """Holds a file open with no sharing (as an installer writing it could); `release_after`: let go from a thread."""
    def __init__(self, path, release_after=None):
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateFileW.restype = wintypes.HANDLE
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self.k = k
        self.h = k.CreateFileW(path, 0x80000000 | 0x40000000, 0, None, 3, 0x80, None)  # GENERIC_READ|WRITE, no sharing, OPEN_EXISTING
        if self.h in (None, wintypes.HANDLE(-1).value):
            raise OSError("could not hold %s (error %d)" % (path, ctypes.get_last_error()))
        self.timer = None
        if release_after is not None:
            import threading
            self.timer = threading.Timer(release_after, self.close)
            self.timer.start()

    def close(self):
        if self.h:
            self.k.CloseHandle(self.h)
            self.h = None


def test_fail_closed(t, synth_f):
    print("\n[6] a well-known list that can't be read: fail closed (exit 10, nothing converted or reused) after a short wait")
    with_f, with_f2 = suite.cases.f_mini_case(present=0), suite.cases.f_mini_case(present=1)
    lib, (a, b) = library(t, "closed", [("A Game", (DLL, with_f.file)), ("B Game", (DLL, with_f2.file))])
    shared_list = t.shared("ignore-files.txt")
    t.set_lists("%s\r\n" % a, None)
    h = Held(shared_list)
    t0 = time.time()
    s = convert_model(t, lib)
    took = time.time() - t0
    h.close()
    t.check("%LOCALAPPDATA% list held open by another program: exit 10, nothing converted, the file and the reason named",
            s["code"] == 10 and s["model"] == b"" and shared_list.lower() in s["error"].lower() and "another program holds it open" in s["error"] and
            "nothing was converted" in s["error"], (s["code"], s["error"]))
    t.check("...and it says what to do: close the program that holds it, or fix the file or delete it",
            "Close the program that holds it" in s["error"] and "or delete it" in s["error"], s["error"])
    t.check("...after the retries (about 5 x 200 ms), not at once and not forever (%.2f s)" % took, 0.75 <= took < 10, took)
    h = Held(shared_list, release_after=0.3)
    s = convert_model(t, lib)
    h.close()
    t.check("a list let go during the retries (an installer finishing its write): read, A skipped, B converted",
            s["code"] == 0 and source_of(s).lower() == b.lower() and len(s["skipped"]) == 1, (s["code"], source_of(s), s["error"]))
    h = Held(shared_list)
    s = convert_model(t, None, keep=True)
    h.close()
    t.check("...also before a reuse: the model data on this PC is not reused while the list can't be read (exit 10)", s["code"] == 10 and "can't be read" in s["error"],
            (s["code"], s["error"], s["outcome"]))
    t.set_lists(None, None)
    os.makedirs(os.path.join(t.data, "ignore-files.txt"))
    s = convert_model(t, lib)
    os.rmdir(os.path.join(t.data, "ignore-files.txt"))
    t.check("the data folder's ignore-files.txt is a folder: exit 10, nothing converted", s["code"] == 10 and s["model"] == b"" and "a folder" in s["error"], (s["code"], s["error"]))
    # a list that can't be decoded (written by another program as UTF-16 without a BOM): named, and what to do said
    t.set_lists(("%s\r\n" % a).encode("utf-16-le"), None)
    s = convert_model(t, lib)
    t.check("a well-known list that can't be decoded: exit 10, nothing converted; the message names the file and says fix it or delete it",
            s["code"] == 10 and s["model"] == b"" and shared_list.lower() in s["error"].lower() and "NUL bytes" in s["error"] and "or delete it" in s["error"],
            (s["code"], s["error"]))
    t.set_lists(None, None)
    # --ignore-list= goes through the lists' bounded reader once: a file of 16 MB or more is refused before it is read
    big = os.path.join(t.work, "big-list.txt")
    with open(big, "wb") as f:
        f.truncate(16 << 20)
    s = convert_model(t, lib, extra=("--ignore-list=" + big,))
    os.remove(big)
    t.check("--ignore-list= of a file of 16 MB: exit 10, nothing converted, the size named",
            s["code"] == 10 and s["model"] == b"" and "16 MB or larger" in s["error"] and "big-list.txt" in s["error"], (s["code"], s["error"]))
    if synth_f and os.path.isfile(os.path.join(synth_f, "synth-common-v0.bin")):
        rlib, _ = library(t, "closed-rr", [("A Game", (RR_DLL, synth_f_dll(synth_f)))])
        t.set_lists("# nothing\r\n", None)
        h = Held(shared_list)
        s = convert_rr(t, rlib)
        h.close()
        t.check("rr-data convert with the list held open: exit 10, no RR data written", s["code"] == 10 and s["data"] == b"" and "can't be read" in s["error"], (s["code"], s["error"]))
    t.set_lists(None, None)


def shared_names(t):
    d = t.shared()
    return sorted(os.listdir(d)) if os.path.isdir(d) else None


def wait_trash_gone(t, seconds=30):
    for _ in range(seconds * 5):
        if not [x for x in os.listdir(t.lad) if ".removed-" in x or ".cleanup-" in x]:
            return True
        time.sleep(0.2)
    return False


def test_keep_list(t):
    print("\n[7] remove-data and the launcher's uninstall keep ignore-files.txt, and say so")
    kept_text = "kept ignore-files.txt (your/integrators' ignore list)"
    list_bytes = b"wronggpu-ignore-files 1\r\n# placed by another program's installer\r\nC:\\Games\\Some Game\\nvngx_dlssd.dll\r\n"
    t.set_lists(None, None)
    write(t.shared("model", "model.wgcm"), b"synthetic model data")
    write(t.shared("ignore-files.txt"), list_bytes)
    rc = t.run("remove-data")
    t.check("remove-data: exit 0, everything else gone, ignore-files.txt kept byte for byte",
            rc == 0 and shared_names(t) == ["ignore-files.txt"] and read(t.shared("ignore-files.txt")) == list_bytes, (rc, shared_names(t), t.j))
    t.check("...and it says so (console and result)", kept_text in t.stdout and t.j.get("kept", "").startswith(kept_text), (t.stdout[-500:], t.j.get("kept")))
    wait_trash_gone(t)
    os.remove(t.shared("ignore-files.txt"))
    write(t.shared("model", "model.wgcm"), b"synthetic model data")
    rc = t.run("remove-data")
    wait_trash_gone(t)
    t.check("remove-data without a list: the folder goes as before, nothing said about a list", rc == 0 and shared_names(t) is None and kept_text not in t.stdout and
            t.j.get("kept") == "", (rc, shared_names(t), t.stdout[-300:]))

    # the launcher installed for this user (the test build's own Apps & features root), then uninstalled
    programs = os.path.join(t.work, "programs", "WrongGPU")
    rc = t.run("self-install", "--dir=" + programs, "--startmenu=-")
    if rc != 0:
        t.check("self-install for the uninstall case", False, (rc, t.j))
        return
    write(t.shared("model", "model.wgcm"), b"synthetic model data")
    write(t.shared("ignore-files.txt"), list_bytes)
    write(os.path.join(t.data, "ignore-files.txt"), list_bytes)  # the launcher's own data folder's list (here the --data folder)
    rc = t.run("self-uninstall", "--startmenu=-")
    t.check("self-uninstall: exit 0, the shared folder's ignore-files.txt kept byte for byte, the rest gone",
            rc == 0 and shared_names(t) == ["ignore-files.txt"] and read(t.shared("ignore-files.txt")) == list_bytes, (rc, shared_names(t), t.j, t.stdout[-600:]))
    t.check("...the data folder's ignore-files.txt kept too",
            os.path.isfile(os.path.join(t.data, "ignore-files.txt")) and read(os.path.join(t.data, "ignore-files.txt")) == list_bytes, os.listdir(t.data))
    t.check("...and it says so for both", (kept_text + ": " + t.shared("ignore-files.txt")).lower() in t.stdout.lower() and
            (kept_text + ": " + os.path.join(t.data, "ignore-files.txt")).lower() in t.stdout.lower(), t.stdout[-800:])
    t.check("...the installed copy goes (the hidden cleanup)", wait_trash_gone(t) and not os.path.exists(os.path.join(programs, "WrongGPU.exe")), os.listdir(t.lad))
    os.remove(t.shared("ignore-files.txt"))
    os.remove(os.path.join(t.data, "ignore-files.txt"))
    test_list_never_lost(t, list_bytes, programs)


def test_list_never_lost(t, list_bytes, programs):
    """The list can't go back as it was (WRONGGPU_TEST_LIST_PUTBACK, test builds: between the folder's move and the put-back,
    another program makes the folder again with its own list ("exists"), or a file is where the folder should be ("blocked")).
    remove-data and the launcher's uninstall never delete it: kept as ignore-files.txt.kept-1 beside the new list, or left in
    the moved-aside folder (nothing deletes that folder's tree), and the player's result says where."""
    print("\n[8] the list can't go back as it was: never deleted, kept beside the new one or in the moved folder, and said")
    other = b"wronggpu-ignore-files 1\r\n# another program's new list (test)\r\n"

    def trash_dirs():
        return [os.path.join(t.lad, x) for x in os.listdir(t.lad) if ".removed-" in x or ".cleanup-" in x]

    def said():
        return " ".join([t.stdout, t.j.get("kept", "") or "", t.j.get("error", "") or ""])

    for verb in ("remove-data", "self-uninstall"):
        for how in ("exists", "blocked"):
            if verb == "self-uninstall":
                rc = t.run("self-install", "--dir=" + programs, "--startmenu=-")
                if rc != 0:
                    t.check("self-install for the %s case" % how, False, (rc, t.j))
                    continue
            write(t.shared("model", "model.wgcm"), b"synthetic model data")
            write(t.shared("ignore-files.txt"), list_bytes)
            t.env["WRONGGPU_TEST_LIST_PUTBACK"] = how
            try:
                rc = t.run(verb, "--startmenu=-") if verb == "self-uninstall" else t.run(verb)
            finally:
                t.env.pop("WRONGGPU_TEST_LIST_PUTBACK", None)
            if how == "exists":
                kept = t.shared("ignore-files.txt.kept-1")
                t.check("%s, another program made the list again meanwhile: exit 0, the player's list kept byte for byte as ignore-files.txt.kept-1, "
                        "the other program's list untouched, the rest gone" % verb,
                        rc == 0 and shared_names(t) == ["ignore-files.txt", "ignore-files.txt.kept-1"] and read(kept) == list_bytes and
                        read(t.shared("ignore-files.txt")) == other, (rc, shared_names(t), t.j, t.stdout[-600:]))
                t.check("...and the result says where it is kept", ("kept your ignore-files.txt as " + kept).lower() in said().lower(), said()[-800:])
                t.check("...the moved-aside folder still goes", wait_trash_gone(t), os.listdir(t.lad))
                os.remove(kept)
                os.remove(t.shared("ignore-files.txt"))
            else:
                trash = trash_dirs()
                left = os.path.join(trash[0], "ignore-files.txt") if len(trash) == 1 else ""
                t.check("%s, the list can't be moved or copied out of the moved folder: exit 0, it stays there byte for byte, the rest of that folder "
                        "goes" % verb, rc == 0 and left and os.path.isfile(left) and read(left) == list_bytes and os.listdir(trash[0]) == ["ignore-files.txt"],
                        (rc, trash, os.listdir(trash[0]) if trash else None, t.j, t.stdout[-600:]))
                t.check("...and the result says where it is kept", left and ("it is kept at " + left).lower() in said().lower(), said()[-800:])
                time.sleep(4)  # a hidden cleanup starts with a 3 s wait: none may delete it
                t.check("...nothing deletes it afterwards (no cleanup of that folder's tree)", left and os.path.isfile(left) and read(left) == list_bytes, trash_dirs())
                if left:
                    os.remove(left)
                    os.rmdir(trash[0])
                if os.path.isfile(t.shared()):
                    os.remove(t.shared())  # the test's file where the folder should be
            if verb == "self-uninstall":
                t.check("...the installed copy goes", not os.path.exists(os.path.join(programs, "WrongGPU.exe")), programs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--synth-f", default=os.environ.get("WGC_SYNTH_BINS", ""))
    a = ap.parse_args()
    work = os.path.abspath(a.work)
    for d in ("libs", "data", "localappdata", "elsewhere", "programs"):
        shutil.rmtree(os.path.join(work, d), ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    t = T(os.path.abspath(a.launcher), work)
    t0 = time.time()
    test_empty_lists_same_winner(t)
    test_model_lists(t)
    test_rr_lists(t, a.synth_f)
    test_model_reuse(t)
    test_rr_reuse(t, a.synth_f)
    test_fail_closed(t, a.synth_f)
    test_keep_list(t)
    print("\n%d checks passed, %d failed, %d skipped (%.0f s)" % (t.passed, len(t.failures), len(t.skipped), time.time() - t0))
    for f in t.failures:
        print("  FAILED:", f)
    for s in t.skipped:
        print("  SKIPPED:", s)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
