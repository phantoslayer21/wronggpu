# SPDX-License-Identifier: MIT
"""The extension path, registry half: a new RR mode and a new game-interface family through the tables only.

    python clean/wgshared/tests/test_dummy_extension.py

tests/data/ext-dummy.json adds mode "dummy" (the kernel-variant words of "accurate"), no option, and family "2.99"
(layout A10-2.14). Generating with --extend must give WG_RR_MODE_DUMMY = 3, WG_RR_MODE_COUNT 4 (the RR plugin's menu
box lists the registry's modes, so "Dummy" appears there; i20: RR modes are no entries of the host's version list),
the mode row with accurate's words, the ini comment listing the new value, the family in REGISTRY.md, and no change
to any existing id, value or entry. Nothing outside the two json files is edited. The RR half of the proof (the
plugin's dummy-family test) reads the same file.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WGSHARED = os.path.dirname(HERE)
GEN = os.path.join(WGSHARED, "tools", "gen_registry.py")
EXT = os.path.join(HERE, "data", "ext-dummy.json")

checks = failures = 0



def cleanup(tmp):
    """Removes the test's own temporary folder file by file (no recursive delete)."""
    for dirpath, dirnames, filenames in os.walk(tmp, topdown=False):
        for name in filenames:
            os.remove(os.path.join(dirpath, name))
        os.rmdir(dirpath)

def check(cond, what):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
        print("FAIL:", what)


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    tmp = tempfile.mkdtemp(prefix="wg_dummy_ext_")
    try:
        merged = os.path.join(tmp, "merged.json")
        p = subprocess.run([sys.executable, "-B", GEN, "--extend", EXT, "--out-dir", tmp, "--dump-merged", merged,
                            "--previous", os.path.join(HERE, "data", "registry.prev.json")], capture_output=True, text=True)
        check(p.returncode == 0, f"generation with the extension succeeds ({p.stderr.strip()})")
        h = read(os.path.join(tmp, "include", "wg_registry_gen.h"))
        base = read(os.path.join(WGSHARED, "include", "wg_registry_gen.h"))
        check("#define WG_RR_MODE_DUMMY 3" in h, "WG_RR_MODE_DUMMY = 3")
        check("#define WG_RR_MODE_COUNT 4" in h, "WG_RR_MODE_COUNT 4")
        check('kWgRrModeNames[WG_RR_MODE_COUNT] = {"exact", "accurate", "fast", "dummy"};' in h, "mode names")
        check("#define WG_ENTRY_COUNT 2" in h, "still the two SR entries")
        check("0x5747000000000104" not in h and "WrongGPU RR dummy" not in h, "no version-list entry for the new mode")
        rows = re.findall(r'^  \{"(\w+)",\s+(.*)\},$', h, re.M)
        words = {name: re.sub(r"\s+", " ", rest) for name, rest in rows}
        check(words.get("dummy") == words.get("accurate") and "dummy" in words, "dummy mode row = accurate's variant words")
        check("#define WG_OPT_COUNT 12" in h, "no option added")
        # every line of the committed header except the comment, hash, counts and the extended tables is unchanged
        keep = [l for l in base.split("\n") if l.startswith("#define WG_OPT_") or l.startswith("  {0x5747") or
                l.startswith("#define WG_SR_MODEL_") or l.startswith("#define WG_VARIANT_") or
                (l.startswith("  {") and "WG_KIND_" in l and '"rr_mode"' not in l)]
        check(all(l in h.split("\n") for l in keep), "existing ids, values, entries, option rows and variant words unchanged")
        check('  {1, "rr_mode", WG_KIND_CHOICE, WG_RR_MODE_ACCURATE, 0, 3, ' in h, "rr_mode row: same default, range grown to 0..3")
        ini = read(os.path.join(tmp, "ini", "wronggpu.default.ini"))
        check("Values: exact | accurate | fast | dummy." in ini, "ini comment lists the new mode")
        md = read(os.path.join(tmp, "REGISTRY.md"))
        check("WrongGPU RR dummy" not in md and "| exact \\| accurate \\| fast \\| dummy |" in md,
              "REGISTRY.md: rr_mode lists the new value, no entry row")
        check(re.search(r"\| 2\.99 \| 2\.99\.0 \.\. 2\.99\.99 \| A10-2\.14 \| confirmed \| rr-f-2\.99\.json \|", md) is not None,
              "REGISTRY.md family 2.99 row (layout A10-2.14)")
        m = json.load(open(merged, encoding="utf-8"))
        check([f["family"] for f in m.get("families", [])] == ["2.99"], "merged registry carries family 2.99")
        check(m["options"][1]["values"] == ["exact", "accurate", "fast", "dummy"], "merged rr_mode values")
        # the committed outputs are untouched by an extension run
        check(subprocess.run([sys.executable, "-B", GEN, "--check"], capture_output=True).returncode == 0,
              "committed outputs unchanged")
        # a mode naming a variant word the registry does not know is refused (a new variant needs kernels first)
        bad = json.load(open(EXT, encoding="utf-8"))
        bad["modes"][0]["body"] = "turbo"
        bad_path = os.path.join(tmp, "ext-bad.json")
        json.dump(bad, open(bad_path, "w", encoding="utf-8"))
        p = subprocess.run([sys.executable, "-B", GEN, "--extend", bad_path, "--out-dir", os.path.join(tmp, "bad")],
                           capture_output=True, text=True)
        check(p.returncode == 2 and "unknown body variant word 'turbo'" in p.stderr, "unknown variant word refused")
    finally:
        cleanup(tmp)
    print(f"test_dummy_extension: {checks} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
