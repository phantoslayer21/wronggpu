# SPDX-License-Identifier: MIT
"""Tests of clean/wgshared/tools/gen_registry.py.

    python clean/wgshared/tests/test_gen_registry.py

- the committed outputs equal a fresh generation;
- the generated header reproduces the initial i15 interface header (tests/data/wg_registry_gen.initial.h) exactly,
  except the header comment and WG_REGISTRY_HASH;
- ids are append-only against the previous registry (tests/data/registry.prev.json: release 0.6.0's), entry ids
  ...0001/...0002 kept and Model L's ...0003 only appended after them;
- every refusal rule fires (id gap, duplicate id, renumbering, removed / reordered values, changed entry label or API
  prefix, a first label without the API prefix, a 4.x prefix, an unknown variant word, overlapping families, an
  unknown layout, test rows into the committed folder);
- --extend merges rows without touching existing ids; the hash is FNV-1a-32 of registry.json with LF line ends.
"""
import copy
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WGSHARED = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WGSHARED, "tools"))
import gen_registry as gen  # noqa: E402

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


def body(text):
    """The header without its leading comment block and without the hash line."""
    lines = text.replace("\r\n", "\n").split("\n")
    end = next(i for i, l in enumerate(lines) if l.strip() == "*/")
    return [l for l in lines[end + 1:] if not l.startswith("#define WG_REGISTRY_HASH ")]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def refused(reg, previous=None, expect=None):
    tmp = tempfile.mkdtemp(prefix="wg_gen_")
    try:
        p = os.path.join(tmp, "registry.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(reg, f)
        try:
            gen.generate(p, (), previous)
        except gen.RegistryError as e:
            return expect is None or re.search(expect, str(e)) is not None
        return False
    finally:
        cleanup(tmp)


def main():
    reg_path = gen.REGISTRY
    reg = load(reg_path)
    prev = os.path.join(HERE, "data", "registry.prev.json")

    # 1. committed outputs = fresh generation
    rc = gen.main(["--check"])
    check(rc == 0, "committed generated files equal a fresh generation (run tools/gen_registry.py)")

    # 2. the initial interface header, reproduced
    outs, merged, h = gen.generate(reg_path, (), prev)
    with open(os.path.join(HERE, "data", "wg_registry_gen.initial.h"), encoding="utf-8") as f:
        initial = f.read()
    # i20: the RR entries left the version list (the RR plugin's menu box) and rr_exact_allowed was appended; every
    # slot id of the initial header and the SR entries stay as they were
    kept = [l for l in body(initial) if (l.startswith("#define WG_OPT_") and "WG_OPT_COUNT" not in l)
            or l.startswith("  {0x5747000000000001ull") or l.startswith("  {0x5747000000000002ull")]
    check(len(kept) == 12 and all(l in body(outs["header"]) for l in kept),
          "every slot id and SR entry of the initial i15 header kept")
    check("0x5747000000000101ull" not in outs["header"], "no RR entry in the version list")
    for needle in ["#define WG_OPT_SR_MODEL 0", "#define WG_OPT_ENTRY_SOURCE 9", "#define WG_OPT_RR_EXACT_ALLOWED 10",
                   "#define WG_OPT_RR_HIT_DISTANCE_FILTER 11", "#define WG_OPT_COUNT 12", "#define WG_ENTRY_COUNT 3",
                   "#define WG_RR_MODE_EXACT 0", "#define WG_RR_MODE_ACCURATE 1", "#define WG_RR_MODE_FAST 2",
                   "static const uint32_t kWgContractSlots[WG_CONTRACT_COUNT] = {2, 3, 4, 5, 6, 11};",
                   '#define WG_API_LEVEL_PREFIX "3.1.0"',
                   'static const char* const kWgRrModeNames[WG_RR_MODE_COUNT] = {"exact", "accurate", "fast"};']:
        check(needle in outs["header"], f"header has {needle!r}")

    # 3. hash
    with open(reg_path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n")
    check(h == gen.fnv1a32(raw), "WG_REGISTRY_HASH is FNV-1a-32 of registry.json (LF)")
    check(f"#define WG_REGISTRY_HASH 0x{h:08X}u" in outs["header"], "hash line in the header")
    check(gen.fnv1a32(b"a") == 0xE40C292C and gen.fnv1a32(b"") == 0x811C9DC5, "FNV-1a-32 reference values")

    # 4. entries
    entries = gen.entries_of(merged)
    # only the SR models: the RR modes are a box of their own in the RR plugin's menu tab (menu list "panel"); Model L
    # (#89) appended after M and K (a front end offers it only when built with it: clean/fsr WG_MODEL_L)
    check([e["id"] for e in entries] == [0x5747000000000001, 0x5747000000000002, 0x5747000000000003], "entry ids")
    check(entries[0]["label"] == "WrongGPU Model M##3.1.0" and entries[1]["label"] == "WrongGPU Model K##3.1.0",
          "the two released entries keep their names")
    check(entries[2]["label"] == "WrongGPU Model L##3.1.0", "Model L's entry")
    check([e["folder_label"] for e in entries] == ["WrongGPU M", "WrongGPU K", "WrongGPU L"], "folder labels")
    check(merged["options"][1]["menu"]["list"] == "panel", "rr_mode is offered in the panel, not the version list")

    # 5. refusals
    r = copy.deepcopy(reg); r["options"][9]["id"] = 11
    check(refused(r, expect="without gaps"), "refuses an id gap")
    r = copy.deepcopy(reg); r["options"][9]["id"] = 8
    check(refused(r, expect="duplicate option id 8"), "refuses a duplicate id")
    r = copy.deepcopy(reg); r["options"][2]["key"], r["options"][3]["key"] = r["options"][3]["key"], r["options"][2]["key"]
    check(refused(r, prev, expect="key changed"), "refuses renumbering (two keys swapped)")
    r = copy.deepcopy(reg); del r["options"][11]; del r["options"][10]; del r["options"][9]
    check(refused(r, prev, expect="removed"), "refuses a removed option")
    r = copy.deepcopy(reg); r["options"][1]["values"] = ["accurate", "exact", "fast"]
    r["modes"] = [r["modes"][1], r["modes"][0], r["modes"][2]]
    check(refused(r, prev, expect="prefix"), "refuses reordered values")
    r = copy.deepcopy(reg); r["options"][0]["menu"]["label"] = "WrongGPU M{value}"
    check(refused(r, prev, expect="changed or moved"), "refuses a changed entry label")
    r = copy.deepcopy(reg); r["options"][0]["menu"]["list"] = "panel"; r["options"][1]["menu"] = {"list": "versions",
                                                                                                "label": "WrongGPU RR {value}"}
    check(refused(r, prev, expect="changed or moved"), "refuses entries that change place (models gone from the front)")
    r = copy.deepcopy(reg); r["options"][1]["menu"]["list"] = "sidebar"
    check(refused(r, expect="not one of versions, panel"), "refuses an unknown menu list")
    check(not gen.append_only_errors(load(prev), reg), "Model L appended after the released entries (M and K keep 0 and 1)")
    r = copy.deepcopy(reg); r["options"][0]["values"] = ["M", "L", "K"]; r["models"] = [r["models"][0], r["models"][2],
                                                                                          r["models"][1]]
    check(refused(r, prev, expect="changed or moved"), "refuses a new model put before a released one")
    r = copy.deepcopy(reg); r["options"][0]["values"] = ["M", "K"]; r["models"] = r["models"][:2]
    check(not gen.append_only_errors(load(prev), r), "the last entry may leave the end of the list")
    r = copy.deepcopy(reg); r["api_level_prefix"] = "v3"
    check(refused(r, expect="not an a.b.c version"), "refuses an API prefix that is not a version")
    r = copy.deepcopy(reg); r["api_level_prefix"] = "4.0.0"
    check(refused(r, expect="3.x.y"), "refuses a 4.x API prefix")
    r = copy.deepcopy(reg); r["options"][0]["menu"]["label"] = "WrongGPU 3.1.5 Model {value}"
    check(refused(r, expect="has an a.b.c number before ##"), "refuses a label with a version number in its shown text")
    r = copy.deepcopy(reg); r["modes"][2]["head"] = "turbo"
    check(refused(r, expect="unknown head variant word 'turbo'"), "refuses an unknown variant word")
    r = copy.deepcopy(reg); r["modes"].append({"id": "x", "option": "rr_mode", "head": "exact", "input": "exact", "body": "exact",
                                              "output": "default", "resolve": "exact"})
    check(refused(r, expect="must equal the modes rows"), "refuses a mode row the option's values do not list")
    r = copy.deepcopy(reg); r["options"][1]["default"] = "medium"
    check(refused(r, expect="default"), "refuses a default outside the values")
    r = copy.deepcopy(reg); r["families"] = [
        {"family": "2.98", "min": [2, 98, 0], "max": [2, 99, 5], "layout": "A10-2.14"},
        {"family": "2.99", "min": [2, 99, 0], "max": [2, 99, 99], "layout": "A10-2.14"}]
    check(refused(r, expect="overlap"), "refuses overlapping families")
    r = copy.deepcopy(reg); r["families"] = [{"family": "2.99", "min": [2, 99, 0], "max": [2, 99, 99], "layout": "A10-3.0"}]
    check(refused(r, expect="unknown layout"), "refuses an unknown family layout")
    check(gen.main(["--extend", os.path.join(HERE, "data", "ext-dummy.json")]) == 2,
          "refuses extension rows into the committed folder")

    # 6. --extend keeps existing ids
    tmp = tempfile.mkdtemp(prefix="wg_gen_ext_")
    try:
        ext = os.path.join(HERE, "data", "ext-dummy.json")
        check(gen.main(["--extend", ext, "--out-dir", tmp, "--previous", prev]) == 0, "--extend generation succeeds, append-only")
        with open(os.path.join(tmp, "include", "wg_registry_gen.h"), encoding="utf-8") as f:
            exth = f.read()
        base_defines = [l for l in body(initial) if (l.startswith("#define WG_OPT_") and "WG_OPT_COUNT" not in l)
                        or l.startswith("  {0x5747000000000001ull") or l.startswith("  {0x5747000000000002ull")]
        check(all(l in exth.split("\n") for l in base_defines), "every original slot id and entry line kept under --extend")
        check("#define WG_RR_MODE_DUMMY 3" in exth and "#define WG_RR_MODE_COUNT 4" in exth, "dummy mode appended")
    finally:
        cleanup(tmp)

    # 7. the RR plugin's family index (WG_FAMILIES_INDEX = a fixture of its shape): validated, but the outputs do not
    #    depend on it (the committed files stay equal when the RR side's table is present after a merge)
    sample = os.path.join(HERE, "data", "families.sample.json")
    saved = os.environ.get("WG_FAMILIES_INDEX")
    os.environ["WG_FAMILIES_INDEX"] = sample
    try:
        outs2, _, _ = gen.generate(reg_path, (), prev)
        check(outs2 == outs, "outputs independent of the family index")
        check(gen.main(["--check"]) == 0, "committed outputs still equal with the family index present")
        tmp = tempfile.mkdtemp(prefix="wg_gen_fam_")
        try:
            check(gen.main(["--extend", os.path.join(HERE, "data", "ext-dummy.json"), "--out-dir", tmp]) == 0,
                  "dummy family 2.99 fits next to 2.6 / 2.7 / 2.8 / 2.10 / 2.14")
        finally:
            cleanup(tmp)
        r = copy.deepcopy(reg); r["families"] = [{"family": "2.14b", "min": [2, 14, 5], "max": [2, 14, 9], "layout": "A10-2.14"}]
        check(refused(r, expect="overlap"), "refuses an extension row inside an indexed family")
    finally:
        if saved is None:
            del os.environ["WG_FAMILIES_INDEX"]
        else:
            os.environ["WG_FAMILIES_INDEX"] = saved

    print(f"test_gen_registry: {checks} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
