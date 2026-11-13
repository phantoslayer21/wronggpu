# SPDX-License-Identifier: MIT
"""Generate WrongGPU's registry outputs from clean/wgshared/registry/registry.json (the one table).

    python clean/wgshared/tools/gen_registry.py                      write the committed outputs
    python clean/wgshared/tools/gen_registry.py --check              exit 1 when a committed output differs
    python clean/wgshared/tools/gen_registry.py --extend X.json --out-dir D
                                                                     merge an extension (tests), write into D
    python clean/wgshared/tools/gen_registry.py --previous P.json    also refuse anything that is not append-only vs P

Outputs (all committed, all regenerated from the registry; never edit them by hand):
  include/wg_registry_gen.h      slot ids, value tables, defaults, readers, entries, contract rows, mode rows, models
  include/wg_registry_ini_gen.h  kWgDefaultIniText: the [wronggpu] section written when no wronggpu.ini exists
  ini/wronggpu.default.ini       the same text as a file
  REGISTRY.md                    the human table

Refusals (exit 2, one line per problem): option id gaps or duplicates, duplicate keys, a choice default that is not
one of its values, value lists that disagree with the model / mode rows that feed them, a mode naming a variant word
the registry does not list, an API-level prefix that is not a 3.x.y version or a version-list entry whose label
does not end with ##<prefix>, duplicate entry ids, overlapping or unknown-layout game-interface families, and (--previous)
any renumbered, removed or reordered option, value, text option, model, mode or entry.

WG_REGISTRY_HASH = FNV-1a-32 of registry.json with LF line ends (then of each extension file, in order).
Plain Python 3, standard library only.
"""
import argparse
import copy
import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
WGSHARED = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(WGSHARED))
REGISTRY = os.path.join(WGSHARED, "registry", "registry.json")
OUTPUTS = {
    "header": os.path.join("include", "wg_registry_gen.h"),
    "ini_header": os.path.join("include", "wg_registry_ini_gen.h"),
    "ini": os.path.join("ini", "wronggpu.default.ini"),
    "md": "REGISTRY.md",
}


class RegistryError(Exception):
    pass


def fnv1a32(data, h=0x811C9DC5):
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def read_bytes_lf(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def ident(text):
    return re.sub(r"[^A-Za-z0-9]", "_", text).upper()


def camel(key):
    return "".join(p[:1].upper() + p[1:] for p in key.split("_"))


def c_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------------------------------------------
# merge + validate


def merge(reg, ext):
    """Merge an extension (tests) into the registry: everything is appended, no existing id changes."""
    reg = copy.deepcopy(reg)
    options = reg["options"]
    by_key = {o["key"]: o for o in options}
    for o in ext.get("options", []):
        options.append(copy.deepcopy(o))
        by_key[o["key"]] = options[-1]
    reg["text_options"].extend(copy.deepcopy(ext.get("text_options", [])))
    for v in ext.get("variants", []):
        group = next((g for g in reg["variants"] if g["group"] == v["group"]), None)
        if group is None:
            reg["variants"].append(copy.deepcopy(v))
        else:
            group["words"].extend(w for w in v["words"] if w not in group["words"])
    for kind in ("models", "modes"):
        for row in ext.get(kind, []):
            reg[kind].append(copy.deepcopy(row))
            opt = by_key.get(row.get("option"))
            if opt is not None and opt.get("type") == "choice" and row["id"] not in opt["values"]:
                opt["values"].append(row["id"])
    reg.setdefault("gi_layouts", []).extend(l for l in ext.get("gi_layouts", []) if l not in reg["gi_layouts"])
    reg.setdefault("families", []).extend(copy.deepcopy(ext.get("families", [])))
    return reg


def load_families_index(reg):
    # WG_FAMILIES_INDEX (tests) replaces the path of the RR plugin's family table.
    path = os.environ.get("WG_FAMILIES_INDEX") or (os.path.join(REPO, reg.get("families_index", "")) if reg.get("families_index") else "")
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("families", []), path
    return [], path


def entries_of(reg):
    base = int(reg.get("entry_id_base", "0x5747000000000000"), 16)
    out = []
    for o in reg["options"]:
        menu = o.get("menu")
        if not menu or menu.get("list") != "versions":
            continue
        for i, v in enumerate(o["values"]):
            out.append({
                "id": base | (o["id"] << 8) | (i + 1),
                "option": o,
                "value_index": i,
                "value": v,
                "label": menu["label"].replace("{value}", v) + "##" + reg["api_level_prefix"],
                "label_tail": menu["label"].replace("{value}", v),
                "folder_label": menu.get("folder_label", menu["label"]).replace("{value}", v),
            })
    return out


def validate(reg, previous=None):
    errors = []
    err = errors.append
    kinds, readers, latches = reg["kinds"], reg["readers"], reg["latches"]
    prefix = reg.get("api_level_prefix", "")
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", prefix)
    if not m:
        err(f"api_level_prefix {prefix!r} is not an a.b.c version (the host reads the API level from the first name)")
    elif int(m.group(1)) != 3:
        err(f"api_level_prefix {prefix} must stay a 3.x.y level (below 4.0.0; the host must never take WrongGPU for FSR 4)")
    ids = [o.get("id") for o in reg["options"]]
    seen = set()
    for i in ids:
        if i in seen:
            err(f"duplicate option id {i}")
        seen.add(i)
    if sorted(i for i in ids if isinstance(i, int)) != list(range(len(ids))):
        err(f"option ids must be 0..{len(ids) - 1} without gaps (append-only slots), got {ids}")
    if ids != sorted(ids, key=lambda x: (not isinstance(x, int), x)):
        err("options must be listed in id order")
    if len(ids) > 64:
        err("more than 64 options: the shared block has 64 slots (WG_SS_SLOT_COUNT)")
    keys = [o["key"] for o in reg["options"]] + [t["key"] for t in reg["text_options"]]
    for k in set(keys):
        if keys.count(k) > 1:
            err(f"duplicate key {k}")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
            err(f"key {k!r}: lower-case letters, digits and '_' only")
    by_key = {o["key"]: o for o in reg["options"]}
    for o in reg["options"]:
        k = o["key"]
        if o.get("type") not in kinds:
            err(f"{k}: type {o.get('type')!r} not one of {kinds}")
            continue
        for r in o.get("readers", []):
            if r not in readers:
                err(f"{k}: unknown reader {r}")
        if o.get("latch") not in latches:
            err(f"{k}: latch {o.get('latch')!r} not one of {latches}")
        if o["type"] == "choice":
            vals = o.get("values", [])
            if not vals or len(set(vals)) != len(vals):
                err(f"{k}: values must be a non-empty list without repeats")
            if o.get("default") not in vals:
                err(f"{k}: default {o.get('default')!r} is not one of {vals}")
            if len(vals) > 254:
                err(f"{k}: at most 254 values (entry id low byte)")
        elif o["type"] == "onoff":
            if o.get("default") not in (0, 1):
                err(f"{k}: onoff default must be 0 or 1")
        else:
            if not (o.get("min", 0) <= o.get("default", 0) <= o.get("max", 0)):
                err(f"{k}: int default outside min..max")
        if o.get("menu") and o["type"] != "choice":
            err(f"{k}: only a choice option can be offered in the host's version list")
        # versions: entries of the host's version list; panel: a box of its own in WrongGPU's menu panel (RR plugin)
        if o.get("menu") and o["menu"].get("list") not in ("versions", "panel"):
            err(f"{k}: menu list {o['menu'].get('list')!r} not one of versions, panel")
    for t in reg["text_options"]:
        for r in t.get("readers", []):
            if r not in readers:
                err(f"text option {t['key']}: unknown reader {r}")
    groups = {g["group"]: g["words"] for g in reg["variants"]}
    for g, words in groups.items():
        if len(set(words)) != len(words):
            err(f"variant group {g}: repeated word")
    for kind, source in (("models", "models"), ("modes", "modes")):
        rows = reg[kind]
        rids = [r["id"] for r in rows]
        if len(set(rids)) != len(rids):
            err(f"duplicate {kind[:-1]} id")
        for opt in reg["options"]:
            if opt.get("values_from") == source:
                mine = [r["id"] for r in rows if r.get("option") == opt["key"]]
                if mine != opt["values"]:
                    err(f"{opt['key']}: values {opt['values']} must equal the {kind} rows {mine} (same order)")
        for r in rows:
            if r.get("option") not in by_key:
                err(f"{kind[:-1]} {r['id']}: unknown option {r.get('option')!r}")
    for mode in reg["modes"]:
        for g, words in groups.items():
            if g not in mode:
                err(f"mode {mode['id']}: no word for variant group {g}")
            elif mode[g] not in words:
                err(f"mode {mode['id']}: unknown {g} variant word {mode[g]!r} (known: {words}); a new variant needs its "
                    "kernels and the word added to the registry's variants and the runtime's word table")
        for f in mode:
            if f not in groups and f not in ("id", "option", "doc"):
                err(f"mode {mode['id']}: unknown field {f!r}")
    for model in reg["models"]:
        if not model.get("runtime_option"):
            err(f"model {model['id']}: runtime_option missing")
    entries = entries_of(reg)
    eids = [e["id"] for e in entries]
    if len(set(eids)) != len(eids):
        err("duplicate version-list entry ids")
    # every label is "<text>##<a.b.c>": the host's menu hides what follows "##" but still reads the API level there
    # (owner 2026-10-31: no version number on show); the text itself must not contain an a.b.c number
    for e in entries:
        if m and not e["label"].endswith("##" + prefix):
            err(f"version-list entry {e['label']!r} does not end with ##{prefix} (the hidden API level)")
        if re.search(r"\d+\.\d+\.\d+", e["label"].split("##")[0]):
            err(f"version-list entry {e['label']!r} has an a.b.c number before ##: the host would read it as the API level")
    # game-interface families (index on the RR side + extension rows)
    base_families, _ = load_families_index(reg)
    fams = base_families + reg.get("families", [])
    names = [f.get("family") for f in fams]
    for n in set(names):
        if names.count(n) > 1:
            err(f"duplicate game-interface family {n}")
    layouts = reg.get("gi_layouts", [])
    for f in fams:
        if f.get("layout") not in layouts:
            err(f"family {f.get('family')}: unknown layout {f.get('layout')!r} (known: {layouts})")
        if not (isinstance(f.get("min"), list) and isinstance(f.get("max"), list) and f["min"] <= f["max"]):
            err(f"family {f.get('family')}: min/max must be [a,b,c] lists with min <= max")
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            try:
                if a["min"] <= b["max"] and b["min"] <= a["max"]:
                    err(f"families {a['family']} and {b['family']} overlap")
            except (KeyError, TypeError):
                pass
    if previous is not None:
        errors.extend(append_only_errors(previous, reg))
    if errors:
        raise RegistryError("\n".join(errors))
    return entries


def append_only_errors(old, new):
    errs = []
    oo, no = old["options"], new["options"]
    for i, o in enumerate(oo):
        if i >= len(no):
            errs.append(f"option {o['key']} (id {o['id']}) was removed")
            continue
        n = no[i]
        for f in ("id", "key", "type"):
            if o.get(f) != n.get(f):
                errs.append(f"option id {o['id']}: {f} changed from {o.get(f)!r} to {n.get(f)!r} (slots are append-only)")
        if o.get("type") == "choice" and n.get("values", [])[:len(o["values"])] != o["values"]:
            errs.append(f"option {o['key']}: values {o['values']} must stay a prefix of {n.get('values')}")
    for kind in ("text_options", "models", "modes"):
        ok = [r.get("key", r.get("id")) for r in old.get(kind, [])]
        nk = [r.get("key", r.get("id")) for r in new.get(kind, [])]
        if nk[:len(ok)] != ok:
            errs.append(f"{kind}: {ok} must stay a prefix of {nk}")
    if old.get("api_level_prefix") != new.get("api_level_prefix"):
        errs.append("api_level_prefix changed (the host's API level of the first name)")
    # The host keeps the chosen entry as an index into the list: an entry may change neither its place nor its label,
    # and entries may only go from the end of the list (owner 2026-10-31: the RR modes left the list for their own box;
    # the SR models before them keep their indexes)
    oe = [(e["id"], e["label"]) for e in entries_of(old)]
    ne = [(e["id"], e["label"]) for e in entries_of(new)]
    if oe[:len(ne)] != ne[:len(oe)]:
        for i, (eid, label) in enumerate(oe[:len(ne)]):
            if ne[i] != (eid, label):
                errs.append(f"version-list entry {eid:#018x} {label!r} changed or moved")
    return errs


# ---------------------------------------------------------------------------------------------------------------
# outputs


def gen_header(reg, entries, hash_value, sources):
    o = []
    w = o.append
    w("/* GENERATED by clean/wgshared/tools/gen_registry.py from clean/wgshared/registry/registry.json. Do not edit.")
    if len(sources) > 1:
        w(" * EXTENDED (test output, never committed into include/): " + ", ".join(sources[1:]) + ".")
    w(" * Regenerate with python clean/wgshared/tools/gen_registry.py; tests/test_gen_registry.py fails when the committed")
    w(" * copy differs from a fresh generation. WG_REGISTRY_HASH = FNV-1a-32 of registry.json (LF line ends); a hash")
    w(" * mismatch between DLLs is log-only. Slot ids are append-only, never renumbered.")
    w(" */")
    w("#ifndef WG_REGISTRY_GEN_H")
    w("#define WG_REGISTRY_GEN_H")
    w("")
    w("#include <stdint.h>")
    w("")
    w("#ifdef __cplusplus")
    w('extern "C" {')
    w("#endif")
    w("")
    w(f"#define WG_REGISTRY_SCHEMA {c_str(reg['schema'])}")
    w(f"#define WG_REGISTRY_HASH 0x{hash_value:08X}u")
    w(f"#define WG_API_LEVEL_PREFIX {c_str(reg['api_level_prefix'])}")
    w("")
    w("/* ---- option slots (shared block index) ---- */")
    for opt in reg["options"]:
        w(f"#define WG_OPT_{ident(opt['key'])} {opt['id']}")
    w(f"#define WG_OPT_COUNT {len(reg['options'])}")
    w("")
    w("/* ---- option kinds ---- */")
    for i, k in enumerate(reg["kinds"]):
        w(f"#define WG_KIND_{ident(k)} {i}")
    w("")
    w("/* ---- readers (bit mask) ---- */")
    for i, r in enumerate(reg["readers"]):
        w(f"#define WG_READER_{ident(r)} {1 << i}u")
    w("")
    w("/* ---- latch points ---- */")
    for i, l in enumerate(reg["latches"]):
        w(f"#define WG_LATCH_{ident(l)} {i}")
    w("")
    w("/* ---- choice values ---- */")
    choices = [opt for opt in reg["options"] if opt["type"] == "choice"]
    for opt in choices:
        k = ident(opt["key"])
        for i, v in enumerate(opt["values"]):
            w(f"#define WG_{k}_{ident(v)} {i}")
        w(f"#define WG_{k}_COUNT {len(opt['values'])}")
        w("")
    w("typedef struct WgOptionDef {")
    w("  uint32_t id;           /* slot */")
    w("  const char* key;       /* ini key, log word */")
    w("  uint32_t kind;         /* WG_KIND_* */")
    w("  int32_t default_value; /* choice index / 0-1 / int */")
    w("  int32_t min_value, max_value;")
    w("  uint32_t readers;      /* WG_READER_* mask */")
    w("  uint32_t live;         /* 1: applied without re-creation */")
    w("  uint32_t latch;        /* WG_LATCH_* */")
    w("  uint32_t contract;     /* 1: row of the plugin's 60.9 contract table */")
    w("  uint32_t opt_in;       /* 1: changes output versus the original (60.9 flag bit 1) */")
    w("  const char* ini_section;")
    w('  const char* env;       /* process environment override, "" = none */')
    w("  const char* const* value_names; /* kind choice: names[max_value + 1] */")
    w("  const char* doc;")
    w("} WgOptionDef;")
    w("")
    for opt in choices:
        k = ident(opt["key"])
        names = ", ".join(c_str(v) for v in opt["values"])
        w(f"static const char* const kWg{camel(opt['key'])}Names[WG_{k}_COUNT] = {{{names}}};")
    w("")
    w("static const WgOptionDef kWgOptions[WG_OPT_COUNT] = {")
    for opt in reg["options"]:
        k = ident(opt["key"])
        if opt["type"] == "choice":
            default = f"WG_{k}_{ident(opt['default'])}"
            lo, hi = 0, len(opt["values"]) - 1
            names = f"kWg{camel(opt['key'])}Names"
        elif opt["type"] == "onoff":
            default, lo, hi, names = str(opt["default"]), 0, 1, "0"
        else:
            default, lo, hi, names = str(opt["default"]), opt["min"], opt["max"], "0"
        rd = " | ".join(f"WG_READER_{ident(r)}" for r in reg["readers"] if r in opt.get("readers", [])) or "0u"
        w(f"  {{{opt['id']}, {c_str(opt['key'])}, WG_KIND_{ident(opt['type'])}, {default}, {lo}, {hi}, {rd}, "
          f"{1 if opt.get('live') else 0}, WG_LATCH_{ident(opt['latch'])}, {1 if opt.get('contract') else 0}, "
          f"{1 if opt.get('opt_in') else 0}, {c_str(opt.get('ini_section', 'wronggpu'))}, {c_str(opt.get('env', ''))}, "
          f"{names}, {c_str(opt.get('doc', ''))}}},")
    w("};")
    w("")
    contract = [opt["id"] for opt in reg["options"] if opt.get("contract")]
    w("/* ---- the plugin's 60.9 contract rows, in contract order (index = today's SettingId) ---- */")
    w(f"#define WG_CONTRACT_COUNT {len(contract)}")
    w(f"static const uint32_t kWgContractSlots[WG_CONTRACT_COUNT] = {{{', '.join(str(c) for c in contract)}}};")
    w("")
    w("/* ---- text options (ini only; copied into the block's text area at seeding) ---- */")
    w("typedef struct WgTextOptionDef { const char* key; const char* default_value; uint32_t readers; const char* env; const char* doc; } WgTextOptionDef;")
    w(f"#define WG_TEXT_OPT_COUNT {len(reg['text_options'])}")
    w("static const WgTextOptionDef kWgTextOptions[WG_TEXT_OPT_COUNT] = {")
    for t in reg["text_options"]:
        rd = " | ".join(f"WG_READER_{ident(r)}" for r in reg["readers"] if r in t.get("readers", [])) or "0u"
        w(f"  {{{c_str(t['key'])}, {c_str(t.get('default', ''))}, {rd}, {c_str(t.get('env', ''))}, {c_str(t.get('doc', ''))}}},")
    w("};")
    w("")
    w("/* ---- version-list entries (what OptiScaler shows under the WrongGPU folder), in list order ---- */")
    w("typedef struct WgEntryDef { uint64_t id; uint32_t slot; int32_t value; const char* label; const char* folder_label; } WgEntryDef;")
    w(f"#define WG_ENTRY_ID_BASE 0x{int(reg.get('entry_id_base', '0x5747000000000000'), 16):016x}ull")
    w(f"#define WG_ENTRY_COUNT {len(entries)}")
    w("static const WgEntryDef kWgEntries[WG_ENTRY_COUNT] = {")
    for e in entries:
        k = ident(e["option"]["key"])
        w(f"  {{0x{e['id']:016x}ull, WG_OPT_{k}, WG_{k}_{ident(e['value'])}, {c_str(e['label_tail'] + '##')} WG_API_LEVEL_PREFIX, "
          f"{c_str(e['folder_label'])}}},")
    w("};")
    w("")
    w("/* ---- RR modes: kernel-variant words per group (the runtime maps words to its variant bits) ---- */")
    for g in reg["variants"]:
        for i, word in enumerate(g["words"]):
            w(f"#define WG_VARIANT_{ident(g['group'])}_{ident(word)} {i}")
    groups = [g["group"] for g in reg["variants"]]
    w(f"typedef struct WgModeRow {{ const char* name; uint32_t {', '.join(groups)}; }} WgModeRow;")
    mode_opt = reg["modes"][0]["option"] if reg["modes"] else "rr_mode"
    w(f"static const WgModeRow kWgModeRows[WG_{ident(mode_opt)}_COUNT] = {{")
    cells = [[c_str(mode["id"]) + ","] + [f"WG_VARIANT_{ident(g)}_{ident(mode[g])}" + ("," if j < len(groups) - 1 else "")
                                          for j, g in enumerate(groups)] for mode in reg["modes"]]
    widths = [max(len(row[c]) for row in cells) for c in range(len(groups) + 1)]
    for row in cells:
        text = " ".join(cell.ljust(widths[c]) if c < len(row) - 1 else cell for c, cell in enumerate(row))
        w(f"  {{{text}}},")
    w("};")
    w("")
    w("/* ---- SR models ---- */")
    w("typedef struct WgModelRow { const char* name; const char* runtime_option; } WgModelRow;")
    model_opt = reg["models"][0]["option"] if reg["models"] else "sr_model"
    rows = ", ".join(f"{{{c_str(mdl['id'])}, {c_str(mdl['runtime_option'])}}}" for mdl in reg["models"])
    w(f"static const WgModelRow kWgModelRows[WG_{ident(model_opt)}_COUNT] = {{{rows}}};")
    w("")
    w("#ifdef __cplusplus")
    w("}")
    w("#endif")
    w("#endif")
    return "\n".join(o) + "\n"


LATCH_WORDS = {"next-frame": "at the next frame", "rr-create": "when the game creates RR", "sr-create": "at the next SR create",
               "start": "at game start"}


def wrap_comment(note):
    return ["# " + line for line in textwrap.wrap(note[2:], 114, break_long_words=False, break_on_hyphens=False)]


def ini_lines(reg):
    lines = [
        "# WrongGPU settings: the one settings file of every WrongGPU DLL (<game exe folder>\\WrongGPU\\wronggpu.ini).",
        "# Delete this file to restore the defaults. The [wronggpu] section is generated from WrongGPU's registry",
        "# (clean/wgshared/registry/registry.json); while OptiScaler shows WrongGPU's entries, the entry chosen in its",
        "# menu is written back here, so the next start without OptiScaler keeps it. A process environment variable",
        "# named on a key's comment overrides the key for that process.",
        "[wronggpu]",
    ]
    for opt in reg["options"]:
        if opt["type"] == "choice":
            values, default = " | ".join(opt["values"]), opt["default"]
        elif opt["type"] == "onoff":
            values, default = "0 | 1", str(opt["default"])
        else:
            values, default = f"{opt['min']} .. {opt['max']}", str(opt["default"])
        note = f"# {opt['key']}: {opt.get('doc', '')}. Values: {values}. Default: {default}. Read by: {', '.join(opt.get('readers', []))};"
        note += f" applied {LATCH_WORDS[opt['latch']]}."
        if opt.get("env"):
            note += f" Environment: {opt['env']}."
        if opt.get("opt_in"):
            note += " Opt-in: changes the output."
        lines.extend(wrap_comment(note))
        lines.append(f"{opt['key']} = {default}")
    for t in reg["text_options"]:
        note = f"# {t['key']}: {t['doc']}." if t.get("doc") else f"# {t['key']}: as above."
        note += f" Read by: {', '.join(t.get('readers', []))}."
        if t.get("env"):
            note += f" Environment: {t['env']}."
        lines.extend(wrap_comment(note))
        lines.append(f"{t['key']} = {t.get('default', '')}".rstrip())
    return lines


def gen_ini(reg):
    return "\n".join(ini_lines(reg)) + "\n"


def gen_ini_header(reg, hash_value):
    o = [
        "/* GENERATED by clean/wgshared/tools/gen_registry.py from clean/wgshared/registry/registry.json. Do not edit.",
        " * The [wronggpu] section of wronggpu.ini written when the file does not exist (CRLF line ends, as the DLLs write",
        " * their ini files). Same text as clean/wgshared/ini/wronggpu.default.ini.",
        " */",
        "#ifndef WG_REGISTRY_INI_GEN_H",
        "#define WG_REGISTRY_INI_GEN_H",
        "",
        f"#define WG_REGISTRY_INI_HASH 0x{hash_value:08X}u",
        "static const char kWgDefaultIniText[] =",
    ]
    lines = ini_lines(reg)
    for i, line in enumerate(lines):
        o.append("    " + c_str(line)[:-1] + '\\r\\n"' + (";" if i == len(lines) - 1 else ""))
    o += ["", "#endif"]
    return "\n".join(o) + "\n"


def gen_md(reg, entries, hash_value, families, families_path):
    o = ["# WrongGPU registry (generated)", "",
         "Generated by `clean/wgshared/tools/gen_registry.py` from `clean/wgshared/registry/registry.json`; do not edit.",
         f"Registry hash `0x{hash_value:08X}` (FNV-1a-32). API level prefix of the version-list names: `{reg['api_level_prefix']}`.",
         "How to add a mode, a model, a game-interface family or an API adapter: `EXTENDING.md`.", "",
         "## Options (shared-block slots)", "",
         "| slot | key | type | values | default | readers | live / latch | env | contract | opt-in | what |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for opt in reg["options"]:
        vals = " \\| ".join(opt["values"]) if opt["type"] == "choice" else ("0 \\| 1" if opt["type"] == "onoff" else f"{opt['min']}..{opt['max']}")
        o.append(f"| {opt['id']} | `{opt['key']}` | {opt['type']} | {vals} | {opt['default']} | {', '.join(opt.get('readers', []))} | "
                 f"{'live' if opt.get('live') else 'latched'} / {opt['latch']} | {opt.get('env', '') or '-'} | "
                 f"{'yes' if opt.get('contract') else '-'} | {'yes' if opt.get('opt_in') else '-'} | {opt.get('doc', '')} |")
    o += ["", "## Text options (ini only; copied into the block's text area at seeding)", "",
          "| key | default | readers | env | what |", "|---|---|---|---|---|"]
    for t in reg["text_options"]:
        o.append(f"| `{t['key']}` | {t.get('default', '') or '-'} | {', '.join(t.get('readers', []))} | {t.get('env', '') or '-'} | {t.get('doc', '')} |")
    o += ["", "## OptiScaler version-list entries (in list order; the first name carries the API level)", "",
          "| index | id | label | folder (one-folder-per-choice fallback) | writes |", "|---|---|---|---|---|"]
    for i, e in enumerate(entries):
        o.append(f"| {i} | `0x{e['id']:016x}` | {e['label']} | {e['folder_label']} | `{e['option']['key']}={e['value']}` |")
    o += ["", "## SR models", "", "| model | runtime option | kernels | build flag |", "|---|---|---|---|"]
    for mdl in reg["models"]:
        o.append(f"| {mdl['id']} | `{mdl['runtime_option']}` | {mdl.get('kernels', '')} | {mdl.get('build_flag', '-')} |")
    groups = [g["group"] for g in reg["variants"]]
    o += ["", "## RR modes (kernel-variant words)", "", "| mode | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for mode in reg["modes"]:
        o.append(f"| {mode['id']} | " + " | ".join(mode[g] for g in groups) + " |")
    o += ["", "Variant words per group (index = the generated `WG_VARIANT_*` value): " +
          "; ".join(f"{g['group']}: " + ", ".join(g["words"]) for g in reg["variants"]) + "."]
    o += ["", "## Game-interface families", "",
          f"Kept by the RR plugin in `{reg.get('families_index', '')}` (one row per family; the generator validates it when "
          f"present: no overlapping version ranges, known layouts). Known structure layouts: "
          f"{', '.join(reg.get('gi_layouts', []))}."]
    # Only rows that come with an extension are listed here (tests), so this file never depends on the RR plugin's table.
    if families:
        o += ["", "Extension rows:", "", "| family | versions | layout | status | table |", "|---|---|---|---|---|"]
        for f in families:
            o.append(f"| {f['family']} | {'.'.join(map(str, f['min']))} .. {'.'.join(map(str, f['max']))} | {f['layout']} | "
                     f"{f.get('status', '')} | {f.get('table', '')} |")
    return "\n".join(o) + "\n"


def generate(reg_path=REGISTRY, extends=(), previous=None):
    with open(reg_path, encoding="utf-8") as f:
        reg = json.load(f)
    h = fnv1a32(read_bytes_lf(reg_path))
    sources = [os.path.basename(reg_path)]
    for e in extends:
        with open(e, encoding="utf-8") as f:
            reg = merge(reg, json.load(f))
        h = fnv1a32(read_bytes_lf(e), h)
        sources.append(os.path.basename(e))
    prev = None
    if previous:
        with open(previous, encoding="utf-8") as f:
            prev = json.load(f)
    entries = validate(reg, prev)
    _, families_path = load_families_index(reg)
    families = reg.get("families", [])  # extension rows only (REGISTRY.md must not depend on the RR plugin's table)
    return {
        "header": gen_header(reg, entries, h, sources),
        "ini_header": gen_ini_header(reg, h),
        "ini": gen_ini(reg),
        "md": gen_md(reg, entries, h, families, families_path),
    }, reg, h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", default=REGISTRY)
    ap.add_argument("--extend", action="append", default=[], help="extension json (tests): merged after the registry")
    ap.add_argument("--previous", help="earlier registry.json: refuse anything that is not append-only")
    ap.add_argument("--out-dir", default=WGSHARED)
    ap.add_argument("--check", action="store_true", help="compare with the files in --out-dir, write nothing")
    ap.add_argument("--dump-merged", help="write the merged registry json here")
    a = ap.parse_args(argv)
    if a.extend and os.path.abspath(a.out_dir) == os.path.abspath(WGSHARED) and not a.check:
        print("gen_registry: --extend output must go to its own --out-dir (the committed outputs never carry test rows)",
              file=sys.stderr)
        return 2
    try:
        outs, reg, h = generate(a.registry, a.extend, a.previous)
    except RegistryError as e:
        for line in str(e).splitlines():
            print("gen_registry: refused: " + line, file=sys.stderr)
        return 2
    if a.dump_merged:
        with open(a.dump_merged, "w", encoding="utf-8", newline="\n") as f:
            json.dump(reg, f, indent=1)
    differ = 0
    for name, rel in OUTPUTS.items():
        path = os.path.join(a.out_dir, rel)
        if a.check:
            have = read_bytes_lf(path).decode("utf-8") if os.path.isfile(path) else None
            if have != outs[name]:
                print(f"gen_registry: {rel} differs from a fresh generation")
                differ += 1
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(outs[name])
    print(f"gen_registry: hash 0x{h:08X}, {len(reg['options'])} options, {len(reg['modes'])} modes, "
          f"{len(reg['models'])} models, {len(entries_of(reg))} entries" + (" (check)" if a.check else f" -> {a.out_dir}"))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
