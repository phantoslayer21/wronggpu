# SPDX-License-Identifier: MIT
"""Forwarding-table generator of the RR proxy (spec 60.1, 60.3, 60.9, annexes A9 and A10).

    python gen_forwarding.py <binding.json> <out_dir> [--probe] [--hook] [--output-name <file.dll|file.asi>]

Reads a name-binding table (binding/rr-f-2.7.json, binding/rr-f-2.14.json) and the files it names:
binding/rr-f-sl-interposer-exports.json (the crossed spec data of annex A9.3: per Streamline version, every export
name with its ordinal), binding/rr-functions.json (the three RR feature-function names of A9.5, by role) and
binding/interposer.json (the game's interposer file name). Writes:

  wg_proxy.def          module-definition file: every export of the original under its own name and ordinal
                        (A9.2: ordinal = 1-based rank in ASCII order, base 1, no gaps), mapped to a forwarding stub
                        or, for the calls the proxy implements (A9.4, typed per A10), to the proxy's role function;
                        plus WrongGPU_GetSettingsContract at the ordinal above the original's highest (60.9)
  wg_forward_stubs.asm  ml64 stubs: jmp through wg_fwd_table[i]; an empty slot resolves the original's export
                        by name first (wg_fwd_resolve), preserving every argument register. Argument-agnostic.
                        Probe build: every stub goes through wg_fwd_note (logs the call, then the same jump)
  wg_binding.inc        C++ data for proxy_dll.cpp: names, role names, feature-function names, GUID bytes,
                        file names, version magic
  wg_binding.json       a manifest of what was generated (counts, ordinals, digests of the inputs)
  wg_version.rc         the VERSIONINFO resource of the built proxy: file and product version = the table's file_version
                        (the game's interposer version plus .0, e.g. 2.7.1.0), which other mods in the game read
                        from the interposer file; WrongGPU's own company, product and description strings (release or
                        probe), the interposer file name as original file name

--probe generates the probe build (WG_RR_PROBE, never the release build): all 19 API roles are typed role
functions (A10) so that each call is logged with its result; graphics and Vulkan entry points stay argument-agnostic
stubs that log the call before the jump. --output-name sets the built file name (default: the table's
proxy_file_name, or probe_file_name with --probe), e.g. the game's interposer file name.

--hook generates the hook-mode plugin (HOOK-MODE.md; with --probe its probe variant): an OptiScaler ASI that
patches the import-address-table slots of the game's interface DLL instead of replacing the file. It writes
wg_binding.inc (as above, plus WG_BINDING_HOOK, the interposer file version the plugin requires, and the roles it
redirects: the release build's intercepted roles, every API role in the probe), wg_hook.def (InitializeASI and the
settings-contract export), wg_version.rc (WrongGPU's own version and strings: the plugin does not stand in for
the interposer) and wg_binding.json; no forwarding stubs. Output name: the table's hook_file_name /
hook_probe_file_name (.asi).

The generator refuses a table that still contains a TODO marker, an export list whose ordinals are not the ASCII
rank, a wrong export count or group count for its version, an API export without a role, a role name that is not
an export, duplicate names, the 2.14.1-only role in a 2.7.x table, or a missing feature-function name. It never
reads a DLL.
"""
import copy
import hashlib
import json
import os
import re
import sys
import uuid

SCHEMA = "wronggpu.rr-proxy.binding.v3"
FF_SCHEMA = "wronggpu.rr-proxy.rr-functions.v1"
INTERPOSER_SCHEMA = "wronggpu.rr-proxy.interposer.v1"
KINDS = ("release",)
OWN_EXPORT = "WrongGPU_GetSettingsContract"
OWN_SYMBOL = "wg_get_settings_contract"
# Streamline API roles of 60.3.1 / A9.4, in a fixed order (the order of the role table).
ROLES = ["init", "shutdown", "is-supported", "is-loaded", "set-loaded", "requirements", "version", "get-function",
         "allocate", "free", "set-device", "set-vulkan-info", "tag", "tag-for-frame", "constants", "new-frame-token",
         "evaluate", "upgrade-interface", "native-interface"]
ROLES_2_14_ONLY = {"tag-for-frame"}
# A9.4: the calls the proxy implements itself (observe / copy / answer RR, then forward the rest); the other three
# API calls (set-vulkan-info, upgrade-interface, native-interface) are forwarded by stubs like every graphics and
# Vulkan entry point. The probe build types all of them (A10) so that every API call is logged with its result.
INTERCEPTED = ["init", "shutdown", "is-supported", "is-loaded", "set-loaded", "requirements", "version",
               "get-function", "allocate", "free", "set-device", "tag", "tag-for-frame", "constants",
               "new-frame-token", "evaluate"]
EXPORT_COUNT = {7: 261, 8: 262, 14: 262}  # 60.1, A9.3 (2.8.0: spec answer SL-2.8, same list as 2.14.1)
API_FUNCTION_COUNT = {7: 18, 8: 19, 14: 19}  # 60.1, A9.3
GRAPHICS_COUNT = 14                    # A9.3 Direct3D / DXGI entry points
VULKAN_COUNT = 229                     # A9.3 vk* entry points
API_PREFIX, VULKAN_PREFIX = "sl", "vk"
FEATURE_FUNCTIONS = ["rr-optimal", "rr-state", "rr-options"]
GUIDS_REQUIRED = ["viewport", "tag"]
GUIDS_OPTIONAL = ["tag_conversion"]   # null = the proxy does not recognise the tag extension (Q-B6)
# A9.6 structure GUIDs the probe build checks (frame_token, A10.1) or names in its log; optional.
GUIDS_NAMED = ["frame_token", "resource", "constants", "rr_options", "rr_optimal", "rr_state", "requirements",
               "feature_version", "adapter_info", "preferences", "precision_info", "subresource_range",
               "resource_allocation_desc"]
NAME_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STEM = "{stem}"
# VERSIONINFO strings of every built proxy (WrongGPU's own; the install tool's double-install guard looks for
# "WrongGPU" in ProductName and FileDescription).
VERSION_COMPANY = "WrongGPU"
VERSION_PRODUCT = "WrongGPU RR proxy"
VERSION_DESCRIPTION = "WrongGPU ray reconstruction proxy ({kind})"
VERSION_INTERNAL = "wg_rr_proxy"
HOOK_PRODUCT = "WrongGPU RR hook"
HOOK_DESCRIPTION = "WrongGPU ray reconstruction hook-mode plugin ({kind}, game interface {api})"
HOOK_INTERNAL = "wg_rr_hook"
HOOK_ENTRY = "InitializeASI"  # the OptiScaler ASI entry WrongGPU.asi exports as well (clean/fsr/asi)


class BindingError(Exception):
    pass


def _todos(obj, path="$"):
    if isinstance(obj, str):
        return [path] if "TODO" in obj else []
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in _todos(v, f"{path}[{i}]")]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() if k not in ("comment", "note") for p in _todos(v, f"{path}.{k}")]
    return []


def role_ident(role):
    return role.replace("-", "_")


def ascii_key(s):
    return s.encode("ascii")


def is_api(name):
    return name.startswith(API_PREFIX) and name[2:3].isupper()


def is_vulkan(name):
    return name.startswith(VULKAN_PREFIX) and name[2:3].isupper()


def guid_bytes(text):
    """GUID text -> the 16 bytes of the in-memory GUID structure (Data1/2/3 little-endian, Data4 as written)."""
    try:
        return list(uuid.UUID(text).bytes_le)
    except (ValueError, AttributeError, TypeError):
        raise BindingError(f"invalid GUID: {text!r}")


def _read(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise BindingError(f"cannot read {path}: {e.strerror}")
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def load(path):
    """Read a binding table and resolve its export list. Returns (binding with 'exports' = [{ordinal, name}],
    digests)."""
    b, digest = _read(path)
    digests = {"binding_sha256": digest}
    if "exports_file" in b and "exports" not in b:
        ep = os.path.join(os.path.dirname(os.path.abspath(path)), b["exports_file"])
        data, edigest = _read(ep)
        key = b.get("exports_key")
        if key not in data or not isinstance(data[key], list):
            raise BindingError(f"{b['exports_file']} has no export list {key!r}")
        b = copy.deepcopy(b)
        b["exports"] = data[key]
        digests["exports_sha256"] = edigest
    base = os.path.dirname(os.path.abspath(path))
    if "rr_feature_functions_file" in b and "rr_feature_functions" not in b:
        data, d = _read(os.path.join(base, b["rr_feature_functions_file"]))
        if data.get("schema") != FF_SCHEMA:
            raise BindingError(f"{b['rr_feature_functions_file']}: schema must be {FF_SCHEMA}")
        roles, names = data.get("roles"), data.get("names")
        if not (isinstance(roles, list) and isinstance(names, list) and len(roles) == len(names)):
            raise BindingError(f"{b['rr_feature_functions_file']}: roles and names must be lists of one length")
        b = copy.deepcopy(b)
        b["rr_feature_functions"] = dict(zip(roles, names))
        digests["rr_functions_sha256"] = d
    if "interposer_file" in b and "interposer_file_name" not in b:
        data, d = _read(os.path.join(base, b["interposer_file"]))
        if data.get("schema") != INTERPOSER_SCHEMA:
            raise BindingError(f"{b['interposer_file']}: schema must be {INTERPOSER_SCHEMA}")
        b = copy.deepcopy(b)
        b["interposer_file_name"] = data.get("file_name")
        digests["interposer_sha256"] = d
    return b, digests


def _magic(v):
    if isinstance(v, str) and re.fullmatch(r"0x[0-9a-fA-F]{1,4}", v):
        return int(v, 16)
    if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 0xFFFF:
        return v
    return None


def bare_dll(v):
    return isinstance(v, str) and len(v) > 4 and v.lower().endswith(".dll") and "/" not in v and "\\" not in v


def bare_asi(v):
    return isinstance(v, str) and len(v) > 4 and v.lower().endswith(".asi") and "/" not in v and "\\" not in v


def validate(b, probe=False, output_name=None, hook=False):
    """Return the normalised binding; raise BindingError with every problem found. probe: the probe build
    (every API role typed and logged); output_name: the built file name instead of the table's; hook: the hook-mode
    plugin (.asi)."""
    errs = []
    if hook:
        for k in ("hook_file_name", "hook_probe_file_name"):
            if not bare_asi(b.get(k)):
                errs.append(f"{k} must be a bare .asi file name")
        if output_name is not None and not bare_asi(output_name):
            errs.append("the hook plugin's output name must be a bare .asi file name")
    if b.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    if b.get("kind") not in KINDS:
        errs.append(f"kind must be one of {KINDS}")
    todos = _todos(b)
    if todos:
        raise BindingError("binding incomplete, TODO at: " + ", ".join(todos))
    m = re.fullmatch(r"2\.(7|8|14)\.\d+", str(b.get("api_version", "")))
    if not m:
        raise BindingError("api_version must be 2.7.x, 2.8.x or 2.14.x")
    minor = int(m.group(1))
    base = b.get("ordinal_base")
    if not isinstance(base, int) or isinstance(base, bool) or base != 1:
        raise BindingError("ordinal_base must be 1 (A9.2: ordinals are the 1-based ASCII rank)")
    entries = b.get("exports")
    if not isinstance(entries, list) or not entries:
        raise BindingError("exports must be a non-empty list of {ordinal, name}")
    if not all(isinstance(e, dict) and set(e) == {"ordinal", "name"} for e in entries):
        raise BindingError("every export entry must be exactly {ordinal, name}")
    exports = [e["name"] for e in entries]
    bad = [n for n in exports if not isinstance(n, str) or not NAME_RX.match(n)]
    if bad:
        raise BindingError(f"export names that are not plain identifiers: {bad[:5]}")
    dups = sorted({n for n in exports if exports.count(n) > 1})
    if dups:
        errs.append(f"duplicate export names: {dups[:5]}")
    if OWN_EXPORT in exports:
        errs.append(f"{OWN_EXPORT} collides with an export of the original")
    if len(exports) != EXPORT_COUNT[minor]:
        errs.append(f"{len(exports)} exports, A9.3 says {EXPORT_COUNT[minor]} for 2.{minor}")
    ordered = sorted(exports, key=ascii_key)  # ASCII (byte-wise) order, A9.2
    want_ord = {n: base + i for i, n in enumerate(ordered)}
    wrong = [(e["name"], e["ordinal"]) for e in entries
             if not isinstance(e["ordinal"], int) or isinstance(e["ordinal"], bool) or e["ordinal"] != want_ord[e["name"]]]
    if wrong:
        errs.append(f"ordinals that are not the 1-based ASCII rank (A9.2): {wrong[:5]}")
    api = [n for n in ordered if is_api(n)]
    vk = [n for n in ordered if is_vulkan(n)]
    gfx = [n for n in ordered if not is_api(n) and not is_vulkan(n)]
    if (len(api), len(gfx), len(vk)) != (API_FUNCTION_COUNT[minor], GRAPHICS_COUNT, VULKAN_COUNT):
        errs.append(f"export groups API/graphics/Vulkan = {len(api)}/{len(gfx)}/{len(vk)}, A9.3 says "
                    f"{API_FUNCTION_COUNT[minor]}/{GRAPHICS_COUNT}/{VULKAN_COUNT}")
    roles = b.get("roles", {})
    unknown = sorted(set(roles) - set(ROLES))
    if unknown:
        errs.append(f"unknown roles: {unknown}")
    want = [r for r in ROLES if not (minor == 7 and r in ROLES_2_14_ONLY)]
    missing = [r for r in want if r not in roles]
    if missing:
        errs.append(f"missing roles: {missing}")
    if minor == 7 and any(r in roles for r in ROLES_2_14_ONLY):
        errs.append("tag-for-frame exists only in 2.14.1")
    rnames = list(roles.values())
    if len(set(rnames)) != len(rnames):
        errs.append("two roles share one export name")
    for r, n in roles.items():
        if n not in exports:
            errs.append(f"role {r}: {n!r} is not an export of the original")
    unroled = [n for n in api if n not in rnames]
    if unroled:
        errs.append(f"API exports without a role (A9.4): {unroled}")
    ff = b.get("rr_feature_functions", {})
    if not isinstance(ff, dict):
        ff = {}
    if sorted(ff) != sorted(FEATURE_FUNCTIONS):
        errs.append(f"rr_feature_functions must name exactly the roles {FEATURE_FUNCTIONS}")
    for k in FEATURE_FUNCTIONS:
        if not isinstance(ff.get(k), str) or not NAME_RX.match(ff.get(k)):
            errs.append(f"rr_feature_functions.{k} missing")
    if len({ff.get(k) for k in FEATURE_FUNCTIONS}) != 3:
        errs.append("the three feature-function names must differ")
    gd = b.get("guids", {})
    gb = {}
    unknown_g = sorted(set(gd) - set(GUIDS_REQUIRED + GUIDS_OPTIONAL + GUIDS_NAMED))
    if unknown_g:
        errs.append(f"unknown guids: {unknown_g}")
    for k in GUIDS_REQUIRED + GUIDS_OPTIONAL + GUIDS_NAMED:
        if k not in GUIDS_REQUIRED and gd.get(k) is None:
            continue
        try:
            gb[k] = guid_bytes(gd.get(k))
        except BindingError as e:
            errs.append(f"guids.{k}: {e}")
    if len({bytes(v) for v in gb.values()}) != len(gb):
        errs.append("structure GUIDs must differ")
    mode = b.get("binding_mode")
    if mode not in ("name", "ordinal"):
        errs.append("binding_mode must be name or ordinal")
    magic = _magic(b.get("version_magic"))
    if magic is None:
        errs.append("version_magic must be a 16-bit integer")
    rv = b.get("release_version")
    if not (isinstance(rv, list) and len(rv) == 3 and all(isinstance(x, int) and x >= 0 for x in rv)):
        errs.append("release_version must be three non-negative integers")
    # Version number other mods read from the interposer file (the table's file_version, with its comment): four
    # 16-bit parts, the game's interposer version of the table plus .0 (2.7.1.0 for 2.7.1, 2.14.1.0 for 2.14.1).
    fv = b.get("file_version")
    if not (isinstance(fv, list) and len(fv) == 4 and
            all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 0xFFFF for x in fv)):
        errs.append("file_version must be four integers 0..65535 (VERSIONINFO)")
    elif ".".join(str(x) for x in fv[:3]) != b["api_version"] or fv[3] != 0:
        errs.append("file_version must be the api_version plus .0 (the game's interposer version)")
    for k in ("proxy_file_name", "probe_file_name", "original_renamed_file_name", "interposer_file_name"):
        if not bare_dll(b.get(k)):
            errs.append(f"{k} must be a bare .dll file name")
    if output_name is not None and not hook and not bare_dll(output_name):
        errs.append("the output name must be a bare .dll file name")
    if hook:
        pf = output_name or b.get("hook_probe_file_name" if probe else "hook_file_name", "")
    else:
        pf = output_name or b.get("probe_file_name" if probe else "proxy_file_name", "")
    of = b.get("original_renamed_file_name", "")
    if (any(STEM in str(x) for x in (pf, b.get("proxy_file_name"), b.get("probe_file_name"),
                                     b.get("interposer_file_name")))
            or str(of).count(STEM) > 1 or (STEM in str(of) and not str(of).startswith(STEM))):
        errs.append(f"{STEM} may appear once, at the start of original_renamed_file_name only")
    if str(pf).lower() == str(of).lower() or str(of).lower() == STEM + ".dll":
        errs.append("the renamed original must have another file name than the proxy")
    if errs:
        raise BindingError("; ".join(errs))
    role_of = {n: r for r, n in roles.items()}
    file_version = list(fv)  # version number other mods read from the interposer file
    intercepted = [r for r in (ROLES if probe else INTERCEPTED) if r in roles]
    forwarded = [n for n in ordered if role_of.get(n) not in intercepted]
    return {
        "minor": minor, "api_version": b["api_version"], "ordered": ordered, "ordinals": want_ord,
        "role_of": role_of, "roles": roles, "forwarded": forwarded, "own_ordinal": base + len(ordered),
        "ff": ff, "guids": gb, "mode": mode, "magic": magic, "release": rv,
        "proxy_file": pf, "original_file": of, "interposer_file": b["interposer_file_name"], "kind": b["kind"],
        "probe": bool(probe), "intercepted": intercepted, "file_version": file_version, "hook": bool(hook),
        "groups": {"api": len(api), "graphics": len(gfx), "vulkan": len(vk)},
    }


def emit_def(n):
    lines = [f'LIBRARY "{n["proxy_file"][:-4]}"', "EXPORTS"]
    fwd_index = {name: i for i, name in enumerate(n["forwarded"])}
    for name in n["ordered"]:
        role = n["role_of"].get(name)
        target = f"wg_role_{role_ident(role)}" if role in n["intercepted"] else f"wg_fwd_{fwd_index[name]}"
        lines.append(f"  {name}={target} @{n['ordinals'][name]}")
    lines.append(f"  {OWN_EXPORT}={OWN_SYMBOL} @{n['own_ordinal']}")
    return "\n".join(lines) + "\n"


def emit_asm(n):
    slow_fn = "wg_fwd_note" if n["probe"] else "wg_fwd_resolve"
    out = ["; generated by clean/rr_f/proxy/tools/gen_forwarding.py - do not edit",
           "EXTERN wg_fwd_table:QWORD", f"EXTERN {slow_fn}:PROC", ".code", "",
           f"; Slow path: r11d = slot index. Saves the argument registers, calls {slow_fn} (resolves the slot"
           + ("; probe build: also logs the call" if n["probe"] else "") + "), restores, jumps to the returned address.",
           "wg_fwd_slow PROC FRAME", "  sub rsp, 88h", "  .allocstack 88h", "  .endprolog",
           "  mov [rsp+20h], rcx", "  mov [rsp+28h], rdx", "  mov [rsp+30h], r8", "  mov [rsp+38h], r9",
           "  movdqa [rsp+40h], xmm0", "  movdqa [rsp+50h], xmm1", "  movdqa [rsp+60h], xmm2",
           "  movdqa [rsp+70h], xmm3", "  mov ecx, r11d", f"  call {slow_fn}",
           "  mov rcx, [rsp+20h]", "  mov rdx, [rsp+28h]", "  mov r8, [rsp+30h]", "  mov r9, [rsp+38h]",
           "  movdqa xmm0, [rsp+40h]", "  movdqa xmm1, [rsp+50h]", "  movdqa xmm2, [rsp+60h]",
           "  movdqa xmm3, [rsp+70h]", "  add rsp, 88h", "  jmp rax", "wg_fwd_slow ENDP", ""]
    for i, _ in enumerate(n["forwarded"]):
        if n["probe"]:  # every call is noted (logged) before the jump
            out += [f"wg_fwd_{i} PROC", f"  mov r11d, {i}", "  jmp wg_fwd_slow", f"wg_fwd_{i} ENDP", ""]
            continue
        out += [f"wg_fwd_{i} PROC", f"  mov rax, qword ptr [wg_fwd_table + {8 * i}]", "  test rax, rax",
                f"  jz slow_{i}", "  jmp rax", f"slow_{i}:", f"  mov r11d, {i}", "  jmp wg_fwd_slow",
                f"wg_fwd_{i} ENDP", ""]
    out.append("END")
    return "\n".join(out) + "\n"


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_inc(n):
    of = n["original_file"]
    stem = of.startswith(STEM)
    o = ["// generated by clean/rr_f/proxy/tools/gen_forwarding.py - do not edit", "#pragma once",
         f"#define WG_BINDING_API_MINOR {n['minor']}u",
         f"#define WG_BINDING_FORWARD_COUNT {len(n['forwarded'])}u",
         f"#define WG_BINDING_EXPORT_COUNT {len(n['ordered'])}u",
         f"#define WG_BINDING_OWN_ORDINAL {n['own_ordinal']}u",
         f"#define WG_BINDING_BY_ORDINAL {1 if n['mode'] == 'ordinal' else 0}",
         f"#define WG_BINDING_VERSION_MAGIC 0x{n['magic']:04x}u",
         f"#define WG_BINDING_PROBE {1 if n['probe'] else 0}",
         f"#define WG_BINDING_HAS_TAG_CONVERSION {1 if 'tag_conversion' in n['guids'] else 0}",
         "// The renamed original: WG_BINDING_ORIGINAL_FROM_STEM = the proxy's own file stem + kWgOriginalFile.",
         f"#define WG_BINDING_ORIGINAL_FROM_STEM {1 if stem else 0}",
         f"static const wchar_t kWgOriginalFile[] = L{c_str(of[len(STEM):] if stem else of)};",
         f"static const char kWgProxyFile[] = {c_str(n['proxy_file'])};",
         "// The game's interposer file name (binding/interposer.json): the name the install tool gives the proxy.",
         f"static const char kWgInterposerFile[] = {c_str(n['interposer_file'])};",
         f"static const unsigned kWgReleaseVersion[3] = {{{', '.join(str(x) for x in n['release'])}}};",
         "static const char* const kWgForwardNames[WG_BINDING_FORWARD_COUNT] = {"]
    o += [f"  {c_str(x)}," for x in n["forwarded"]] + ["};"]
    o.append("// Export names of the 19 roles in the order of gen_forwarding.ROLES (\"\" = absent in this version).")
    o.append("static const char* const kWgRoleNames[19] = {")
    o += [f"  {c_str(n['roles'].get(r, ''))},  // {r}" for r in ROLES] + ["};"]
    o.append("// Every export, ASCII order, with its ordinal and its target (role index or -1, forward slot or -1).")
    o.append("struct WgExportInfo { const char* name; unsigned ordinal; int role; int forward_slot; };")
    o.append("static const WgExportInfo kWgExports[WG_BINDING_EXPORT_COUNT] = {")
    fwd_index = {name: i for i, name in enumerate(n["forwarded"])}
    for name in n["ordered"]:
        role = n["role_of"].get(name)
        ri = ROLES.index(role) if role else -1
        o.append(f"  {{{c_str(name)}, {n['ordinals'][name]}u, {ri}, {fwd_index.get(name, -1)}}},")
    o.append("};")
    o.append("// RR feature-function names (A9.5, binding/rr-functions.json), matched exactly by <get-function>(1001).")
    for k, sym in (("rr-optimal", "Optimal"), ("rr-state", "State"), ("rr-options", "Options")):
        o.append(f"static const char kWgFn{sym}[] = {c_str(n['ff'][k])};")
    for k in GUIDS_REQUIRED + GUIDS_OPTIONAL + ["frame_token"]:
        g = n["guids"].get(k, [0] * 16)
        o.append(f"static const unsigned char kWgGuid_{k}[16] = {{{', '.join(str(x) for x in g)}}};")
    o.append(f"#define WG_BINDING_HAS_FRAME_TOKEN_GUID {1 if 'frame_token' in n['guids'] else 0}")
    o.append("// A9.6 structure GUIDs by name (probe log); in-memory byte order.")
    o.append("struct WgGuidName { const char* name; unsigned char b[16]; };")
    named = [k for k in GUIDS_REQUIRED + GUIDS_OPTIONAL + GUIDS_NAMED if k in n["guids"]]
    o.append(f"#define WG_BINDING_GUID_NAME_COUNT {len(named)}u")
    o.append("static const WgGuidName kWgGuidNames[WG_BINDING_GUID_NAME_COUNT] = {")
    o += [f"  {{{c_str(k)}, {{{', '.join(str(x) for x in n['guids'][k])}}}}}," for k in named] + ["};"]
    if n.get("hook"):
        o += emit_hook_inc(n)
    return "\n".join(o) + "\n"


def emit_hook_inc(n):
    """Extra data of the hook-mode plugin: the interposer version it requires and the roles it redirects."""
    hooked = [ROLES.index(r) for r in n["intercepted"]]
    fv = n["file_version"]
    return ["// ---- hook-mode plugin (HOOK-MODE.md)",
            "#define WG_BINDING_HOOK 1",
            f"static const char kWgHookFile[] = {c_str(n['proxy_file'])};",
            "// The interposer file version the plugin requires (the module's VERSIONINFO, major.minor compared; the",
            "// table's file_version, the version number other mods read from the interposer file).",
            f"static const unsigned kWgInterposerFileVersion[4] = {{{', '.join(str(x) for x in fv)}}};",
            "// Role indices (gen_forwarding.ROLES) whose import-address-table slots the plugin redirects.",
            f"#define WG_BINDING_HOOK_ROLE_COUNT {len(hooked)}u",
            f"static const int kWgHookRoles[WG_BINDING_HOOK_ROLE_COUNT] = {{{', '.join(str(x) for x in hooked)}}};"]


def emit_hook_def(n):
    return "\n".join([f'LIBRARY "{n["proxy_file"]}"', "EXPORTS", f"  {HOOK_ENTRY}",
                      f"  {OWN_EXPORT}={OWN_SYMBOL}"]) + "\n"


def emit_hook_rc(n):
    """VERSIONINFO of the hook-mode plugin: WrongGPU's release version, WrongGPU's own strings."""
    rv = list(n["release"]) + [0]
    comma, dotted = ",".join(str(x) for x in rv), ".".join(str(x) for x in rv)
    strings = [("CompanyName", VERSION_COMPANY),
               ("FileDescription", HOOK_DESCRIPTION.format(kind="probe" if n["probe"] else "release",
                                                           api=n["api_version"])),
               ("FileVersion", dotted), ("InternalName", HOOK_INTERNAL),
               ("OriginalFilename", n["proxy_file"]), ("ProductName", HOOK_PRODUCT),
               ("ProductVersion", dotted)]
    o = ["// generated by clean/rr_f/proxy/tools/gen_forwarding.py - do not edit", "#include <winver.h>", "",
         "VS_VERSION_INFO VERSIONINFO", f" FILEVERSION {comma}", f" PRODUCTVERSION {comma}",
         " FILEFLAGSMASK VS_FFI_FILEFLAGSMASK", " FILEFLAGS 0", " FILEOS VOS_NT_WINDOWS32", " FILETYPE VFT_DLL",
         " FILESUBTYPE VFT2_UNKNOWN", "BEGIN", ' BLOCK "StringFileInfo"', " BEGIN", '  BLOCK "040904b0"', "  BEGIN"]
    o += [f'   VALUE "{k}", {rc_str(v)}' for k, v in strings]
    o += ["  END", " END", ' BLOCK "VarFileInfo"', " BEGIN", '  VALUE "Translation", 0x409, 1200', " END", "END"]
    return "\r\n".join(o) + "\r\n"


def rc_str(s):
    return '"' + s.replace('"', '""') + '"'


def emit_rc(n):
    """VERSIONINFO of the built proxy (Microsoft's public resource-script syntax, US English / Unicode)."""
    fv = n["file_version"]
    comma, dotted = ",".join(str(x) for x in fv), ".".join(str(x) for x in fv)
    strings = [("CompanyName", VERSION_COMPANY),
               ("FileDescription", VERSION_DESCRIPTION.format(kind="probe" if n["probe"] else "release")),
               ("FileVersion", dotted), ("InternalName", VERSION_INTERNAL),
               ("OriginalFilename", n["interposer_file"]), ("ProductName", VERSION_PRODUCT),
               ("ProductVersion", dotted)]
    o = ["// generated by clean/rr_f/proxy/tools/gen_forwarding.py - do not edit", "#include <winver.h>", "",
         "// FILEVERSION / PRODUCTVERSION: the game's interposer version of this build (version number other mods",
         "// read from the interposer file).",
         "VS_VERSION_INFO VERSIONINFO", f" FILEVERSION {comma}", f" PRODUCTVERSION {comma}",
         " FILEFLAGSMASK VS_FFI_FILEFLAGSMASK", " FILEFLAGS 0", " FILEOS VOS_NT_WINDOWS32", " FILETYPE VFT_DLL",
         " FILESUBTYPE VFT2_UNKNOWN", "BEGIN", ' BLOCK "StringFileInfo"', " BEGIN", '  BLOCK "040904b0"', "  BEGIN"]
    o += [f'   VALUE "{k}", {rc_str(v)}' for k, v in strings]
    o += ["  END", " END", ' BLOCK "VarFileInfo"', " BEGIN", '  VALUE "Translation", 0x409, 1200', " END", "END"]
    return "\r\n".join(o) + "\r\n"


def generate_from(b, out_dir, digests=None, source="(in memory)", probe=False, output_name=None, hook=False):
    n = validate(b, probe, output_name, hook)
    os.makedirs(out_dir, exist_ok=True)
    if hook:
        files = {"wg_hook.def": emit_hook_def(n), "wg_binding.inc": emit_inc(n), "wg_version.rc": emit_hook_rc(n)}
    else:
        files = {"wg_proxy.def": emit_def(n), "wg_forward_stubs.asm": emit_asm(n), "wg_binding.inc": emit_inc(n),
                 "wg_version.rc": emit_rc(n)}
    manifest = {"binding": source, **(digests or {}), "kind": n["kind"],
                "api_version": n["api_version"], "exports": len(n["ordered"]), "groups": n["groups"],
                "forwarded": len(n["forwarded"]), "probe": n["probe"], "hook": n["hook"],
                "proxy_file": n["proxy_file"],
                "interposer_file": n["interposer_file"], "original_file": n["original_file"],
                "intercepted": sorted(n["intercepted"]), "feature_functions": "named",
                "file_version": ".".join(str(x) for x in n["file_version"]),
                "own_export": {"name": OWN_EXPORT, "ordinal": n["own_ordinal"]},
                "ordinals": n["ordinals"]}
    files["wg_binding.json"] = json.dumps(manifest, indent=1) + "\n"
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(text.encode("utf-8"))
    return n


def generate(binding_path, out_dir, probe=False, output_name=None, hook=False):
    b, digests = load(binding_path)
    return generate_from(b, out_dir, digests, os.path.basename(binding_path), probe, output_name, hook)


def main(argv):
    args = list(argv[1:])
    probe = "--probe" in args
    if probe:
        args.remove("--probe")
    hook = "--hook" in args
    if hook:
        args.remove("--hook")
    output_name = None
    if "--output-name" in args:
        i = args.index("--output-name")
        if i + 1 >= len(args):
            print(__doc__)
            return 2
        output_name = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        print(__doc__)
        return 2
    try:
        n = generate(args[0], args[1], probe, output_name, hook)
    except BindingError as e:
        print(f"gen_forwarding: {e}", file=sys.stderr)
        return 1
    print(f"gen_forwarding: {len(n['ordered'])} exports ({len(n['forwarded'])} forwarded by stubs), "
          f"own export ordinal {n['own_ordinal']}, API 2.{n['minor']}{' probe' if probe else ''}"
          f"{' hook' if hook else ''}, "
          f"{n['proxy_file']} -> {args[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
