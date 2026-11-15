#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The launcher's Model K choice (CMake WRONGGPU_MODEL_K, a development build) through its command line: `set <game>
model=K|M` on an installed fixture game. K only with Model K's data on this PC; it writes wronggpu.ini's model item and
OptiScaler's UpscalerIndex (1, Model K's entry right after Model M's), the install record's hashes follow, M puts both files
back byte for byte, an L control (a build with the Model L choice too) is unchanged, uninstall restores the game.

Written for this project; Python standard library only. Runs WrongGPU.exe (a WRONGGPU_TESTING build made from the installer
suite's prepared payload and maps, with -DWRONGGPU_PRESET_K=ON -DWRONGGPU_MODEL_K=ON -DWRONGGPU_MODEL_L=ON -DWRONGGPU_DIRECT_ROUTE=OFF
and the two test maps below) against generated files only: synthetic model files from the converter's test generator. No
game, no GPU, no window; --noscan always.

    python run_installer_tests.py --prepare-only --work <dir>        (the payload and test-map.json for -DWRONGGPU_PAYLOAD_DIR / -DWRONGGPU_MAP)
    python run_model_k_tests.py --write-test-maps <dir>              (test-map-k.json for -DWRONGGPU_MAP_K=, test-map-l.json for -DWRONGGPU_MAP_L=)
    python run_model_k_tests.py --launcher <WrongGPU.exe> --work <dir>
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_model_l_tests as ml  # noqa: E402  (T, the Model L case and the install helpers; it sets WRONGGPU_TESTING)
from run_installer_tests import read, write  # noqa: E402
from run_model_l_tests import m_case, l_case, recorded_hash, suite, suite_options_model, upscaler_index, no_stamp, stamp  # noqa: E402


def k_case():
    """The synthetic Model K source of the test map (the same every time)."""
    return suite.cases.k_mini_case(file_version=(310, 98, 2, 0))


def profile_text(b):
    return read(os.path.join(b, "wronggpu-setup", "profile.ini"))


def test_choice(t):
    print("\n[1] the model choice: set <game> model=K / model=M / model=L (a build with the K and L choices)")
    import run_launcher_tests as rl  # (its game and OptiScaler fixtures: names and synthetic bytes only)
    t.model = m_case()  # rl.game() writes t.model.file as the game's DLSS file (this hides T.model, the path helper)
    root, b, exe = rl.game(t, "K Choice Game", validated=True)
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
    rc = t.run("set", exe, "model=Q")
    t.check("set model=Q: refused, the allowed values named (m, k, l); nothing changed",
            rc != 0 and "allowed: m, k, l" in t.j.get("message", "") + t.j.get("error", "") and read(wini) == m_w and read(oini) == m_o,
            (rc, t.j.get("message"), t.j.get("error")))
    rc = t.run("set", exe, "model=K")
    t.check("set model=K without Model K's data: exit 5, refused by name, nothing changed",
            rc == 5 and "Model K can't be chosen" in t.j.get("message", "") + t.j.get("error", "") and
            "model convert --preset=K" in t.j.get("message", "") + t.j.get("error", "") and read(wini) == m_w and read(oini) == m_o and
            b"model=" not in profile_text(b), (rc, t.j.get("message"), t.j.get("error")))
    ksrc = os.path.join(t.work, "k-source", "nvngx_dlss.dll")
    write(ksrc, k_case().file)
    rc = t.run("model", "convert", "--preset=K", "--model=" + ksrc)
    t.check("Model K's data converted (model convert --preset=K)", rc == 0 and ml.holds(t.model_path("K", "model.wgcm"), k_case()), (rc, t.j))
    rc = t.run("set", exe, "model=K")
    w = read(wini).decode("utf-8", "replace")
    t.check("set model=K: exit 0; wronggpu.ini's last model item is model=K", rc == 0 and suite_options_model(w) == "K", (rc, t.j.get("message"), w))
    o = read(oini).decode("utf-8", "replace")
    t.check("...OptiScaler's UpscalerIndex names Model K's entry (1: right after Model M)", upscaler_index(o) == "1", o)
    t.check("...the record's hashes follow both files (the file-ownership rule: they are WrongGPU's writes)",
            recorded_hash(b, "wronggpu.ini") == hashlib.sha256(read(wini)).hexdigest() and recorded_hash(b, "OptiScaler.ini") == hashlib.sha256(read(oini)).hexdigest(),
            (recorded_hash(b, "wronggpu.ini"), recorded_hash(b, "OptiScaler.ini")))
    t.check("...profile.ini says model=k", b"model=k\r\n" in profile_text(b))
    k_w, k_o = read(wini), read(oini)
    rc = t.run("set", exe, "model=K")
    t.check("set model=K again: only model_chosen changes (stamped again: the newest choice)",
            rc == 0 and no_stamp(read(wini)) == no_stamp(k_w) and read(oini) == k_o and stamp(read(wini)) > stamp(k_w) > 0, rc)
    rc = t.run("get", exe, "model")
    t.check("get model: k", rc == 0 and t.j.get("settings", {}).get("model", {}).get("profile") == "k", t.j.get("settings"))
    rc = t.run("set", exe, "model=M")
    t.check("set model=M: wronggpu.ini (but its model_chosen line) and OptiScaler.ini back byte for byte as the install wrote them",
            rc == 0 and no_stamp(read(wini)) == no_stamp(m_w) and read(oini) == m_o and stamp(read(wini)) > 0, (rc, read(wini)[-200:], read(oini)[-200:]))
    t.check("...profile.ini has no model line again", b"model=" not in profile_text(b))
    # the Model L control: unchanged by the K choice (its entry follows Model K's, so it is 2)
    lsrc = os.path.join(t.work, "l-source", "nvngx_dlss.dll")
    write(lsrc, l_case().file)
    rc = t.run("model", "convert", "--preset=L", "--model=" + lsrc)
    t.check("L control: Model L's data converted", rc == 0, (rc, t.j))
    rc = t.run("set", exe, "model=L")
    t.check("L control: set model=L writes model=L and index 2 (after Model K's entry)",
            rc == 0 and suite_options_model(read(wini).decode("utf-8", "replace")) == "L" and upscaler_index(read(oini).decode("utf-8", "replace")) == "2",
            (rc, t.j.get("message"), read(wini), read(oini)))
    rc = t.run("set", exe, "model=K")
    t.check("L to K: model=K and index 1", rc == 0 and suite_options_model(read(wini).decode("utf-8", "replace")) == "K" and
            upscaler_index(read(oini).decode("utf-8", "replace")) == "1", (rc, t.j.get("message")))
    rc = t.run("set", exe, "model=L")
    rc2 = t.run("set", exe, "model=M")
    t.check("K to L to M: back byte for byte as the install wrote them (but model_chosen)",
            rc == 0 and rc2 == 0 and no_stamp(read(wini)) == no_stamp(m_w) and read(oini) == m_o, (rc, rc2))
    # a game saved on K whose data goes away: a repair keeps the player's choice, the files say Model M with a warning
    rc = t.run("set", exe, "model=K")
    os.remove(os.path.join(t.lad, "WrongGPU", "model", "K", "model.wgcm"))
    rc = t.run("repair", exe, "--opti=" + opti, "--force")
    msgs = " ".join(t.j.get("warnings", []) + t.j.get("notes", []))
    t.check("repair of a game set to K without Model K's data: exit 0, the files say Model M, the player is told how to convert it",
            rc == 0 and suite_options_model(read(wini).decode("utf-8", "replace")) == "M" and "Model K's data on this PC is not ready" in msgs and
            "model convert --preset=K" in msgs, (rc, t.j.get("message"), msgs[:600]))
    t.check("...the choice stays in profile.ini (model=k)", b"model=k\r\n" in profile_text(b))
    rc = t.run("uninstall", exe)
    # A build with Preset K seeds the plugin's own settings file wronggpu-asi.ini on every OptiScaler install; an uninstall leaves
    # it in place and lists it (optiscaler.h SeedAsiSelection), as it did before the K choice. Nothing else may differ.
    left = suite.diff(before, suite.snapshot(root))
    t.check("uninstall: the game folder byte for byte as before the install, except the plugin's own wronggpu-asi.ini",
            rc == 0 and left.startswith("bin/x64/wronggpu-asi.ini: absent -> ") and ";" not in left, (rc, left))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-test-maps", help="write test-map-k.json and test-map-l.json into this folder and stop")
    ap.add_argument("--launcher", help="WrongGPU.exe built with the K, L and Preset K choices and the test maps")
    ap.add_argument("--work", default=os.path.join(HERE, "..", "build-tests", "model-k"))
    a = ap.parse_args()
    if a.write_test_maps:
        write(os.path.join(a.write_test_maps, "test-map-k.json"), json.dumps(k_case().map, indent=1))
        write(os.path.join(a.write_test_maps, "test-map-l.json"), ml.map_bytes(l_case()))
        print("test maps: " + a.write_test_maps)
        return 0
    if not a.launcher:
        ap.error("--launcher or --write-test-maps is needed")
    d = os.path.join(os.path.abspath(a.work), "choice")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    t = ml.T(os.path.abspath(a.launcher), d)
    t.model_path = t.model  # (test_choice replaces t.model with the Model M case, as the Model L script does)
    test_choice(t)
    print("\n%d checks passed, %d failed" % (t.passed, len(t.failures)))
    for f in t.failures:
        print("  FAILED: " + f)
    return 1 if t.failures else 0


if __name__ == "__main__":
    sys.exit(main())
