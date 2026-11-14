#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Model L (#89) by name through the launcher's command line: `model status --preset=L` and `model convert --preset=L`
pick Model L's converter map (data/preset-L.v1.map.json, the spec side's revision v1.1) and its model folder
(%LOCALAPPDATA%\\WrongGPU\\model\\L, the runtime's folder for model=L); a file without Model L's data is refused with
Model L's words; Model M stays as it was, byte for byte.

Written for this project; Python standard library only. Runs WrongGPU.exe (WRONGGPU_TESTING builds made from the
installer suite's prepared payload and test map) against generated files only: synthetic model files from the
converter's test generator. Only the model verbs run: no install, no window, no GPU; --noscan always.

    python run_model_l_tests.py --write-test-map <file.json>
        the test map for --launcher-synth's build (-DWRONGGPU_MAP_L=<file.json>): the exact layout of
        data/preset-L.v1.map.json with digests of our own random content
    python run_model_l_tests.py --launcher <exe> [--launcher-synth <exe>] [--reference-launcher <exe>] --work <dir>
        --launcher: built with the shipped preset-L map (the default WRONGGPU_MAP_L)
        --launcher-synth: built with the test map above (the conversion itself)
        --reference-launcher: a launcher built from the base branch with the same payload and test map; Model M's file
        from both must be the same bytes
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

os.environ["WRONGGPU_TESTING"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_installer_tests as suite  # noqa: E402  (cases, read_container, read, write)
from run_installer_tests import read, write  # noqa: E402
from synth import shipped_like_case  # noqa: E402  (converter tests: a shipped map's exact layout, our own content)

L_MAP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "data", "preset-L.v1.map.json"))
L_SEED, L_W_OFFSET = 31, 4096 + 6
NO_L = "This DLSS file has no preset-L model."
NOT_MODEL = "This is not a DLSS model file."  # the converter's reason for a file smaller than Model L's parameter block


def l_case():
    """The synthetic Model L source of the test map (the same every time)."""
    return shipped_like_case(L_MAP, seed=L_SEED, w_offset=L_W_OFFSET)


def map_bytes(case):
    return json.dumps(case.map, indent=1).encode("utf-8")


def m_case():
    """The installer suite's synthetic Model M source and the map its prepared test map is (run_installer_tests main)."""
    return suite.cases.f_mini_case(present=0)


class T:
    def __init__(self, exe, work):
        self.exe, self.work = exe, work
        self.passed, self.failures = 0, []
        self.log = os.path.join(work, "launcher-test.log")
        self.out = os.path.join(work, "out.json")
        self.lad = os.path.join(work, "localappdata")
        shutil.rmtree(self.lad, ignore_errors=True)
        os.makedirs(self.lad)
        self.env = dict(os.environ, LOCALAPPDATA=self.lad, ProgramData=os.path.join(work, "programdata"))
        for k in [k for k in self.env if k.upper().startswith("WG_") or k.upper() in ("HIP_PATH", "ROCM_PATH", "LLVM_PATH", "HIP_DEVICE_LIB_PATH")]:
            self.env.pop(k)
        self.j, self.stdout = {}, ""

    def check(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print("  PASS ", name)
        else:
            self.failures.append(name + (": " + str(detail)[:1500] if detail else ""))
            print("  FAIL ", name, str(detail)[:1500])

    def run(self, *args):
        """WrongGPU.exe <args> --noscan --data --log --out; the JSON result in self.j; returns the exit code."""
        for p in (self.out, self.log):
            if os.path.exists(p):
                os.remove(p)
        cmd = [self.exe] + list(args) + ["--noscan", "--log=" + self.log, "--out=" + self.out, "--data=" + os.path.join(self.work, "data")]
        p = subprocess.run(cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        self.stdout = p.stdout.decode("utf-8", "replace")
        try:
            with open(self.out, encoding="utf-8") as f:
                self.j = json.load(f)
        except (OSError, ValueError):
            self.j = {}
        return p.returncode

    def model(self, *rel):
        return os.path.join(self.lad, "WrongGPU", "model", *rel)

    def m_files(self):
        """Model M's files (the model folder itself, without the per-model sub-folders) with their bytes."""
        d = self.model()
        return {n: read(os.path.join(d, n)) for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n))} if os.path.isdir(d) else {}


def holds(path, case):
    """The model file holds the reference tensors of `case`: the key, the stages in map order, each digest, every tensor."""
    try:
        key, _, stages = suite.read_container(path)
    except Exception:  # noqa: BLE001
        return False
    want = [st for st in case.map["stages"] if st["stage"] in case.expected]
    if key != case.key or [s[0] for s in stages] != [st["stage"] for st in want]:
        return False
    for (name, digest, tensors), st in zip(stages, want):
        # a stage with a digest list (Model M's output filter, map version 4): the variant the file holds
        expected = st["digest"] if isinstance(st["digest"], str) else st["digest"][case.found[name]]
        if digest.hex() != expected or [t[3] for t in tensors] != [b for _, b in case.expected[name]]:
            return False
    return True


def test_shipped_map(t):
    """The launcher built with the shipped preset-L map: Model L by name reads that map (v1.1) and Model L's folder."""
    print("\n[1] the shipped preset-L map (v1.1): chosen by name, a file without Model L refused")
    with open(L_MAP, "rb") as f:
        shipped = f.read()
    m = json.loads(shipped)
    t.check("data/preset-L.v1.map.json is revision v1.1: \"preset\": \"L\", map version 1, a 13-entry dec0_values",
            m.get("preset") == "L" and m["version"] == 1 and len(m["perms"]["dec0_values"]) == 13, (m.get("preset"), m["version"]))
    rc = t.run("model", "status", "--preset=L")
    mj = t.j.get("model", {})
    t.check("model status --preset=L: exit 0, model L, its file in model\\L, not valid yet",
            rc == 0 and t.j.get("preset") == "L" and mj.get("file", "").lower() == t.model("L", "model.wgcm").lower() and mj.get("valid") is False, (rc, t.j))
    t.check("...the converter map in use is the shipped preset-L map (its SHA-256), loaded without an error",
            mj.get("map_sha256") == hashlib.sha256(shipped).hexdigest() and not mj.get("map_error"), mj)
    rc = t.run("model", "status", "--preset=l")
    t.check("--preset=l (lower case): the same model", rc == 0 and t.j.get("preset") == "L", (rc, t.j))
    rc = t.run("model", "status")
    mm = t.j.get("model", {})
    t.check("model status without --preset: Model M, the model folder itself, another map",
            rc == 0 and t.j.get("preset") == "M" and mm.get("file", "").lower() == t.model("model.wgcm").lower() and
            mm.get("map_sha256") not in ("", hashlib.sha256(shipped).hexdigest()), (rc, t.j))
    # Model M converted first: what a refused Model L conversion must leave as it is
    mc = m_case()
    msrc = os.path.join(t.work, "m-source", "nvngx_dlss.dll")
    write(msrc, mc.file)
    rc = t.run("model", "convert", "--model=" + msrc)
    t.check("Model M converts from the suite's synthetic file (as before)", rc == 0 and holds(t.model("model.wgcm"), mc), (rc, t.j, t.stdout[-400:]))
    before = t.m_files()
    rc = t.run("model", "convert", "--preset=L", "--model=" + msrc)
    err = t.j.get("error", "")
    t.check("model convert --preset=L of Model M's small stand-in: exit 5, refused with Model L's words and the converter's reason for that file",
            rc == 5 and err.startswith("Model L can't be prepared: no DLSS model file with Model L's data was found.") and
            ("The file you chose (" + msrc + "): " + NOT_MODEL).lower() in err.lower() and "310.5 or newer" in err, (rc, err, t.stdout[-400:]))
    t.check("...nothing written for Model L", not os.path.exists(t.model("L")), os.listdir(t.model()))
    t.check("...Model M's files unchanged, byte for byte", t.m_files() == before, sorted(t.m_files()))
    big = os.path.join(t.work, "random-source", "nvngx_dlss.dll")
    write(big, hashlib.sha256(b"model L, random").digest() * (m["sources"]["W"]["length"] // 32 + 4096))
    rc = t.run("model", "convert", "--preset=L", "--model=" + big)
    t.check("a random file larger than Model L's parameter block: exit 5, the converter's reason is Model L's (\"no preset-L model\")",
            rc == 5 and ("The file you chose (" + big + "): " + NO_L).lower() in t.j.get("error", "").lower(), (rc, t.j.get("error")))
    for bad, why in (("Q", "Unknown model \"Q\""), ("ML", "Unknown model \"ML\""), ("", "Unknown model \"\"")):
        rc = t.run("model", "convert", "--preset=" + bad, "--model=" + msrc)
        t.check("--preset=%s: a bad command line (exit 10), named, nothing converted" % bad,
                rc == 10 and t.j.get("error", "").startswith(why) and t.m_files() == before and not os.path.exists(t.model("L")), (rc, t.j))
    rc = t.run("model", "status", "--preset=K")
    t.check("--preset=K in a build without Preset K: exit 10, the models of this build named", rc == 10 and "M or L" in t.j.get("error", ""), (rc, t.j))


def test_conversion(t, ref):
    """The launcher built with the test map: Model L converts into model\\L, bit-exact; Model M is untouched."""
    print("\n[2] Model L converted by name (a test map of the shipped layout, our own content)")
    lc, mc = l_case(), m_case()
    rc = t.run("model", "status", "--preset=L")
    t.check("the build's Model L map is the test map of the shipped layout", rc == 0 and t.j.get("model", {}).get("map_sha256") == hashlib.sha256(map_bytes(lc)).hexdigest(), t.j)
    msrc = os.path.join(t.work, "m-source", "nvngx_dlss.dll")
    write(msrc, mc.file)
    lsrc = os.path.join(t.work, "l-source", "nvngx_dlss.dll")
    write(lsrc, lc.file)
    rc = t.run("model", "convert", "--model=" + msrc)
    t.check("Model M first (no --preset): converted, bit-exact", rc == 0 and holds(t.model("model.wgcm"), mc), (rc, t.j))
    m_before = t.m_files()
    rc = t.run("model", "convert", "--preset=L", "--model=" + lsrc)
    t.check("model convert --preset=L: exit 0, converted from the chosen file", rc == 0 and t.j.get("outcome", "").startswith("converted from " + lsrc), (rc, t.j, t.stdout[-600:]))
    t.check("...model\\L\\model.wgcm holds Model L's tensors, bit-exact with the reference (every stage, digest and tensor)", holds(t.model("L", "model.wgcm"), lc))
    src = read(t.model("L", "model-source.txt")).decode("utf-8", "replace") if os.path.isfile(t.model("L", "model-source.txt")) else ""
    t.check("...model\\L\\model-source.txt names the file and the map (its SHA-256), no output-filter line (Model L has none)",
            ("source\t" + lsrc + "\r\n") in src and ("map\t" + hashlib.sha256(map_bytes(lc)).hexdigest()) in src and "filter\t" not in src, src)
    t.check("...Model M's files unchanged, byte for byte", t.m_files() == m_before, sorted(t.m_files()))
    rc = t.run("model", "status", "--preset=L")
    t.check("model status --preset=L: ready", rc == 0 and t.j.get("model", {}).get("valid") is True, t.j)
    l_bytes = read(t.model("L", "model.wgcm"))
    rc = t.run("model", "convert", "--preset=L", "--model=" + msrc)
    t.check("Model L again (a file without L chosen): the valid file on this PC is reused, unchanged",
            rc == 0 and t.j.get("outcome", "").startswith("reused") and read(t.model("L", "model.wgcm")) == l_bytes, (rc, t.j))
    rc = t.run("model", "convert", "--preset=M", "--model=" + msrc)
    t.check("--preset=M: Model M's file reused, byte for byte as before", rc == 0 and t.m_files() == m_before and t.j.get("preset") == "M", (rc, t.j))
    shutil.rmtree(t.model("L"))
    rc = t.run("model", "convert", "--preset=L", "--model=" + msrc)
    t.check("Model L with only Model M's file given: exit 5, refused with Model L's words; Model M untouched",
            rc == 5 and t.j.get("error", "").startswith("Model L can't be prepared:") and NOT_MODEL in t.j.get("error", "") and not os.path.exists(t.model("L", "model.wgcm")) and t.m_files() == m_before, (rc, t.j))
    if ref:
        r = T(ref, os.path.join(t.work, "reference"))
        rc = r.run("model", "convert", "--model=" + msrc)
        t.check("the base branch's launcher converts Model M from the same file into the same bytes (model.wgcm)",
                rc == 0 and r.m_files().get("model.wgcm") == m_before.get("model.wgcm") and m_before.get("model.wgcm"), (rc, r.j))


def upscaler_index(text):
    """OptiScaler.ini's UpscalerIndex WrongGPU manages: [WrongGPU] (OptiScaler 10's slot) when there, else [FSR]."""
    sec, found = None, {}
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if s.startswith("["):
            sec = s.strip("[]").lower()
        elif s.lower().startswith("upscalerindex") and "=" in s:
            found[sec] = s.split("=", 1)[1].strip()
    return found.get("wronggpu", found.get("fsr"))


def recorded_hash(b, name):
    """The hash the install record holds for a recorded file (its copy entry: action existed orig inst rel)."""
    for line in read(os.path.join(b, "wronggpu-setup", "install.record")).decode("utf-8", "replace").replace("\r", "").split("\n"):
        f = line.split("\t")
        if len(f) == 5 and f[0] == "copy" and f[4].lower() == name.lower():
            return f[3].lower()
    return None


def test_choice(t):
    """The launcher's Model L choice (a build with WRONGGPU_MODEL_L on and the test map): set <game> model=L/M on an installed
    game. L only with Model L's data on this PC; it writes wronggpu.ini's model item and OptiScaler's UpscalerIndex, the
    record's hashes follow (the file-ownership rule); M puts both files back byte for byte; uninstall restores the game."""
    print("\n[3] the model choice: set <game> model=L / model=M (a build with the choice on)")
    import run_launcher_tests as rl  # (its game and OptiScaler fixtures: names and synthetic bytes only)
    t.model = m_case()  # rl.game() writes t.model.file as the game's DLSS file (this hides T.model, the path helper)
    root, b, exe = rl.game(t, "Choice Game", validated=True)
    before = suite.snapshot(root)
    opti = rl.opti_package(t)
    rc = t.run("install", exe, "--opti=" + opti, "--force")
    t.check("install: exit 0", rc == 0, (rc, t.j.get("message"), t.stdout[-400:]))
    if rc != 0:
        return
    wini, oini = os.path.join(b, "wronggpu.ini"), os.path.join(b, "OptiScaler.ini")
    m_w, m_o = read(wini), read(oini)
    rc = t.run("get", exe, "model")
    t.check("get model: m (Model M, the default)", rc == 0 and t.j.get("settings", {}).get("model", {}).get("profile") == "m", t.j.get("settings"))
    t.check("profile.ini of the Model M game: no model line", b"model=" not in read(os.path.join(b, "wronggpu-setup", "profile.ini")))
    rc = t.run("set", exe, "model=L")
    t.check("set model=L without Model L's data: exit 5, refused by name, nothing changed",
            rc == 5 and "Model L can't be chosen" in t.j.get("message", "") + t.j.get("error", "") and read(wini) == m_w and read(oini) == m_o,
            (rc, t.j.get("message"), t.j.get("error")))
    lsrc = os.path.join(t.work, "l-source", "nvngx_dlss.dll")
    write(lsrc, l_case().file)
    rc = t.run("model", "convert", "--preset=L", "--model=" + lsrc)
    t.check("Model L's data converted (model convert --preset=L)", rc == 0, (rc, t.j))
    rc = t.run("set", exe, "model=L")
    w = read(wini).decode("utf-8", "replace")
    t.check("set model=L: exit 0; wronggpu.ini's last model item is model=L", rc == 0 and suite_options_model(w) == "L", (rc, t.j.get("message"), w))
    t.check("...OptiScaler's UpscalerIndex names Model L's entry (1: after Model M in a build without Preset K)",
            upscaler_index(read(oini).decode("utf-8", "replace")) == "1", read(oini).decode("utf-8", "replace"))
    t.check("...the record's hashes follow both files (the file-ownership rule: they are WrongGPU's writes)",
            recorded_hash(b, "wronggpu.ini") == hashlib.sha256(read(wini)).hexdigest() and recorded_hash(b, "OptiScaler.ini") == hashlib.sha256(read(oini)).hexdigest(),
            (recorded_hash(b, "wronggpu.ini"), recorded_hash(b, "OptiScaler.ini")))
    t.check("...profile.ini says model=l", b"model=l\r\n" in read(os.path.join(b, "wronggpu-setup", "profile.ini")))
    rc = t.run("set", exe, "model=M")
    t.check("set model=M: wronggpu.ini and OptiScaler.ini back byte for byte as the install wrote them",
            rc == 0 and read(wini) == m_w and read(oini) == m_o, (rc, read(wini)[-200:], read(oini)[-200:]))
    t.check("...profile.ini has no model line again", b"model=" not in read(os.path.join(b, "wronggpu-setup", "profile.ini")))
    # a game saved on L whose data goes away: a repair keeps the player's choice, the files say Model M with a warning
    rc = t.run("set", exe, "model=L")
    os.remove(os.path.join(t.lad, "WrongGPU", "model", "L", "model.wgcm"))  # (t.model is the case here, not the path helper)
    rc = t.run("repair", exe, "--opti=" + opti, "--force")
    msgs = " ".join(t.j.get("warnings", []) + t.j.get("notes", []))
    t.check("repair of a game set to L without Model L's data: exit 0, the files say Model M, the player is told how to convert it",
            rc == 0 and suite_options_model(read(wini).decode("utf-8", "replace")) == "M" and "Model L's data on this PC is not ready" in msgs and
            "model convert --preset=L" in msgs, (rc, t.j.get("message"), msgs[:600]))
    t.check("...the choice stays in profile.ini (model=l)", b"model=l\r\n" in read(os.path.join(b, "wronggpu-setup", "profile.ini")))
    rc = t.run("uninstall", exe)
    t.check("uninstall: the game folder byte for byte as before the install", rc == 0 and suite.snapshot(root) == before, (rc, suite.diff(before, suite.snapshot(root))))


def test_no_choice(t):
    """A build without the choice (the default): model=L refused by name; nothing changes."""
    print("\n[4] a build without the model choice (the default)")
    import run_launcher_tests as rl
    t.model = m_case()
    root, b, exe = rl.game(t, "No Choice Game", validated=True)
    rc = t.run("install", exe, "--opti=" + rl.opti_package(t), "--force")
    t.check("install: exit 0", rc == 0, (rc, t.j.get("message")))
    if rc != 0:
        return
    w, o = read(os.path.join(b, "wronggpu.ini")), read(os.path.join(b, "OptiScaler.ini"))
    rc = t.run("set", exe, "model=L")
    t.check("set model=L: a bad command line (exit 10), Model L named, nothing changed",
            rc == 10 and t.j.get("error", "").startswith("Model L isn't part of this build of WrongGPU") and
            read(os.path.join(b, "wronggpu.ini")) == w and read(os.path.join(b, "OptiScaler.ini")) == o, (rc, t.j))
    rc = t.run("get", exe)
    t.check("get: no model setting listed", rc == 0 and "model" not in t.j.get("settings", {}), list(t.j.get("settings", {})))
    t.run("uninstall", exe)


def suite_options_model(text):
    """The last model item of wronggpu.ini's runtime_options ("M", "L", ... or None)."""
    model = None
    for line in text.replace("\r", "").split("\n"):
        s = line.strip()
        if s.lower().startswith("runtime_options") and "=" in s:
            for item in s.split("=", 1)[1].split(";"):
                k, _, v = item.strip().partition("=")
                if k.strip().lower() == "model":
                    model = v.strip().split()[0] if v.strip() else ""
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-test-map", help="write the test map (the shipped preset-L layout, our own digests) and stop")
    ap.add_argument("--launcher", help="WrongGPU.exe built with the shipped preset-L map")
    ap.add_argument("--launcher-synth", help="WrongGPU.exe built with --write-test-map's map (-DWRONGGPU_MAP_L=)")
    ap.add_argument("--reference-launcher", help="WrongGPU.exe of the base branch (same payload and test map)")
    ap.add_argument("--launcher-choice", help="WrongGPU.exe built with -DWRONGGPU_MODEL_L=ON and the test map (the model choice)")
    ap.add_argument("--work", default=os.path.join(HERE, "..", "build-tests", "model-l"))
    a = ap.parse_args()
    if a.write_test_map:
        write(a.write_test_map, map_bytes(l_case()))
        print("test map: " + a.write_test_map)
        return 0
    if not a.launcher and not a.launcher_synth and not a.launcher_choice:
        ap.error("--launcher, --launcher-synth or --launcher-choice is needed")
    work = os.path.abspath(a.work)
    passed, failures = 0, []
    for exe, part, fn in ((a.launcher, "shipped", test_shipped_map), (a.launcher_synth, "synth", lambda t: test_conversion(t, a.reference_launcher)),
                          (a.launcher_choice, "choice", test_choice), (a.launcher_synth, "no-choice", test_no_choice)):
        if not exe:
            print("\n(%s: no launcher given, skipped)" % part)
            continue
        d = os.path.join(work, part)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        t = T(os.path.abspath(exe), d)
        fn(t)
        passed += t.passed
        failures += t.failures
    print("\n%d checks passed, %d failed" % (passed, len(failures)))
    for f in failures:
        print("  FAILED: " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
