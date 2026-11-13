# SPDX-License-Identifier: MIT
"""Unit tests of the forwarding-table generator (tools/gen_forwarding.py) on the real tables of annex A9
(spec 60.1 / 60.3 / 60.9, A9.1-A9.5, A10), the names files of binding/ and the probe build's generation."""
import copy
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROXY = os.path.dirname(HERE)
BINDING = os.path.join(PROXY, "binding")
sys.path.insert(0, os.path.join(PROXY, "tools"))
import gen_forwarding as gf  # noqa: E402

DEF_RX = re.compile(r"^  (\w+)=(\w+) @(\d+)$", re.M)
# A9.3: the non-Vulkan exports of 2.7.x in ordinal order (2.14.1 inserts slSetTagForFrame at 30).
NON_VULKAN_2_7 = [
    "CreateDXGIFactory", "CreateDXGIFactory1", "CreateDXGIFactory2", "D3D11CreateDevice",
    "D3D11CreateDeviceAndSwapChain", "D3D12CreateDevice", "D3D12CreateRootSignatureDeserializer",
    "D3D12CreateVersionedRootSignatureDeserializer", "D3D12EnableExperimentalFeatures", "D3D12GetDebugInterface",
    "D3D12GetInterface", "D3D12SerializeRootSignature", "D3D12SerializeVersionedRootSignature",
    "DXGIGetDebugInterface1", "slAllocateResources", "slEvaluateFeature", "slFreeResources", "slGetFeatureFunction",
    "slGetFeatureRequirements", "slGetFeatureVersion", "slGetNativeInterface", "slGetNewFrameToken", "slInit",
    "slIsFeatureLoaded", "slIsFeatureSupported", "slSetConstants", "slSetD3DDevice", "slSetFeatureLoaded",
    "slSetTag", "slSetVulkanInfo", "slShutdown", "slUpgradeInterface"]
# A9.4: export name per role.
ROLE_NAMES = {
    "init": "slInit", "shutdown": "slShutdown", "is-supported": "slIsFeatureSupported",
    "is-loaded": "slIsFeatureLoaded", "set-loaded": "slSetFeatureLoaded", "requirements": "slGetFeatureRequirements",
    "version": "slGetFeatureVersion", "get-function": "slGetFeatureFunction", "allocate": "slAllocateResources",
    "free": "slFreeResources", "set-device": "slSetD3DDevice", "set-vulkan-info": "slSetVulkanInfo",
    "tag": "slSetTag", "tag-for-frame": "slSetTagForFrame", "constants": "slSetConstants",
    "new-frame-token": "slGetNewFrameToken", "evaluate": "slEvaluateFeature",
    "upgrade-interface": "slUpgradeInterface", "native-interface": "slGetNativeInterface"}
FORWARDED_API = {"slSetVulkanInfo", "slUpgradeInterface", "slGetNativeInterface"}


def names_file(name):
    with open(os.path.join(BINDING, name), "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def rr_names():
    """The three feature-function names by role, read from binding/rr-functions.json (never written in a test)."""
    d = names_file("rr-functions.json")
    return dict(zip(d["roles"], d["names"]))


def interposer_name():
    return names_file("interposer.json")["file_name"]


def table(minor):
    return os.path.join(BINDING, f"rr-f-2.{minor}.json")


def load(minor):
    b, _ = gf.load(table(minor))
    return b


def raw_exports(key):
    with open(os.path.join(BINDING, "rr-f-sl-interposer-exports.json"), "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))[key]


class Generator(unittest.TestCase):
    def setUp(self):
        self.b7 = load(7)
        self.b14 = load(14)

    def gen(self, b, probe=False, output_name=None):
        with tempfile.TemporaryDirectory() as d:
            n = gf.generate_from(b, os.path.join(d, "out"), probe=probe, output_name=output_name)
            files = {}
            for name in os.listdir(os.path.join(d, "out")):
                with open(os.path.join(d, "out", name), "rb") as f:
                    files[name] = f.read().decode()
            return n, files

    def test_tables_resolve_the_spec_export_lists(self):
        self.assertEqual(len(raw_exports("2.7.1")), 261)
        self.assertEqual(len(raw_exports("2.14.1")), 262)
        self.assertEqual(self.b7["exports"], raw_exports("2.7.1"))
        self.assertEqual(self.b14["exports"], raw_exports("2.14.1"))
        self.assertEqual(self.b7["api_version"], "2.7.1")
        self.assertEqual(self.b14["api_version"], "2.14.1")

    def test_counts_and_ordinals(self):
        for b, total, api in ((self.b7, 261, 18), (self.b14, 262, 19)):
            n, files = self.gen(b)
            self.assertEqual(len(n["ordered"]), total)
            self.assertEqual(len(n["roles"]), api)
            self.assertEqual(n["groups"], {"api": api, "graphics": 14, "vulkan": 229})
            # A9.2: ordinal = 1-based ASCII rank, base 1, no gaps; identical to the spec's list
            self.assertEqual([n["ordinals"][x] for x in n["ordered"]], list(range(1, total + 1)))
            self.assertEqual({e["name"]: e["ordinal"] for e in b["exports"]}, n["ordinals"])
            # own export above the original's highest ordinal, by name
            self.assertEqual(n["own_ordinal"], total + 1)
            self.assertIn(f"WrongGPU_GetSettingsContract=wg_get_settings_contract @{total + 1}", files["wg_proxy.def"])
            exports = DEF_RX.findall(files["wg_proxy.def"])
            self.assertEqual(len(exports), total + 1)
            self.assertEqual(len({o for _, _, o in exports}), total + 1)
            for name, target, o in exports[:-1]:
                self.assertEqual(int(o), n["ordinals"][name])

    def test_a9_3_non_vulkan_table(self):
        n7, _ = self.gen(self.b7)
        n14, _ = self.gen(self.b14)
        for i, name in enumerate(NON_VULKAN_2_7):
            self.assertEqual(n7["ordinals"][name], i + 1, name)
            self.assertEqual(n14["ordinals"][name], i + 1 + (1 if i + 1 >= 30 else 0), name)
        self.assertEqual(n14["ordinals"]["slSetTagForFrame"], 30)
        self.assertNotIn("slSetTagForFrame", n7["ordinals"])
        vk7 = [x for x in n7["ordered"] if x.startswith("vk")]
        vk14 = [x for x in n14["ordered"] if x.startswith("vk")]
        self.assertEqual(vk7, vk14)
        self.assertEqual(len(vk7), 229)
        self.assertEqual([n7["ordinals"][x] for x in vk7], list(range(33, 262)))
        self.assertEqual([n14["ordinals"][x] for x in vk14], list(range(34, 263)))

    def test_tag_for_frame_shifts_ordinals(self):
        n7, _ = self.gen(self.b7)
        n14, _ = self.gen(self.b14)
        new = self.b14["roles"]["tag-for-frame"]
        self.assertEqual(set(n14["ordered"]) - set(n7["ordered"]), {new})
        for name in n7["ordered"]:
            shift = 1 if name.encode() > new.encode() else 0
            self.assertEqual(n14["ordinals"][name], n7["ordinals"][name] + shift, name)
        self.assertTrue(any(x.encode() > new.encode() for x in n7["ordered"]))  # the shift is exercised

    def test_roles_are_a9_4(self):
        for b, minor in ((self.b7, 7), (self.b14, 14)):
            want = {r: x for r, x in ROLE_NAMES.items() if minor == 14 or r != "tag-for-frame"}
            self.assertEqual(b["roles"], want)

    def test_routing_of_exports(self):
        for b in (self.b7, self.b14):
            n, files = self.gen(b)
            d = dict((name, target) for name, target, _ in DEF_RX.findall(files["wg_proxy.def"]))
            for role, name in b["roles"].items():
                if role in gf.INTERCEPTED:
                    self.assertEqual(d[name], "wg_role_" + role.replace("-", "_"))
                    self.assertNotIn(name, FORWARDED_API)
                else:
                    self.assertTrue(d[name].startswith("wg_fwd_"), role)
                    self.assertIn(name, FORWARDED_API)
            # every Direct3D / DXGI and vk* entry point goes to a stub (forwarded unchanged)
            for name in n["ordered"]:
                if not gf.is_api(name):
                    self.assertTrue(d[name].startswith("wg_fwd_"), name)
            stubs = [t for t in d.values() if t.startswith("wg_fwd_")]
            self.assertEqual(len(stubs), len(n["forwarded"]))
            self.assertEqual(sorted(int(t[7:]) for t in stubs), list(range(len(stubs))))
            self.assertEqual(len(stubs), 14 + 229 + 3)
            # one argument-agnostic stub per slot in the asm, each jumping through its own table entry
            asm = files["wg_forward_stubs.asm"]
            for i in range(len(stubs)):
                self.assertIn(f"wg_fwd_{i} PROC\n  mov rax, qword ptr [wg_fwd_table + {8 * i}]", asm)
            self.assertEqual(asm.count(" PROC"), len(stubs) + 1)
            # the C++ table lists the forwarded names in slot order (resolved by name, A9.2)
            inc = files["wg_binding.inc"]
            fwd = re.search(r"kWgForwardNames\[WG_BINDING_FORWARD_COUNT\] = \{\n(.*?)\n\};", inc, re.S).group(1)
            self.assertEqual(re.findall(r'"(\w+)"', fwd), n["forwarded"])

    def test_every_game_import_is_exported(self):
        with open(os.path.join(BINDING, "game-imports.json"), "rb") as f:
            games = json.loads(f.read().decode("utf-8"))["games"]
        self.assertEqual([len(g["imports"]) for g in games], [19, 13, 13])  # A9.1; 2.8.0: the names RE Requiem resolves
        gens = {7: self.gen(self.b7), 8: self.gen(load(8)), 14: self.gen(self.b14)}
        for g in games:
            minor = int(g["interposer_version"].split(".")[1])
            n, files = gens[minor]
            exported = {name for name, _, _ in DEF_RX.findall(files["wg_proxy.def"])}
            for name in g["imports"]:
                self.assertIn(name, exported, f"{g['game']} imports {name}")
            # A9.4: an imported API call is implemented by the proxy unless A9.4 marks it "forward" (game A imports
            # slUpgradeInterface); the graphics imports are forwarded
            for name in g["imports"]:
                role = n["role_of"].get(name)
                if gf.is_api(name) and name not in FORWARDED_API:
                    self.assertIn(role, gf.INTERCEPTED, name)
                else:
                    self.assertIn(name, n["forwarded"])
            v = int(g["sdk_version_value"], 16)
            self.assertEqual(v & 0xFFFF, 0xFEDC)  # A9.1 magic
        # games that import the interface statically never use tag-for-frame; RE Requiem (2.8.0, resolved by name at
        # run time, reached through the host's routes) uses only tag-for-frame
        static = [g for g in games if g["interposer_version"] != "2.8.0"]
        self.assertNotIn("slSetTagForFrame", {x for g in static for x in g["imports"]})

    def test_inc_contents(self):
        n, files = self.gen(self.b14)
        inc = files["wg_binding.inc"]
        self.assertIn("#define WG_BINDING_API_MINOR 14u", inc)
        self.assertIn("#define WG_BINDING_VERSION_MAGIC 0xfedcu", inc)
        self.assertIn("#define WG_BINDING_BY_ORDINAL 0", inc)
        self.assertIn("#define WG_BINDING_PROBE 0", inc)
        # A9.5: the real names from binding/rr-functions.json, by role
        names = rr_names()
        for role, sym in (("rr-optimal", "Optimal"), ("rr-state", "State"), ("rr-options", "Options")):
            self.assertIn(f'static const char kWgFn{sym}[] = "{names[role]}";', inc)
        self.assertIn(f'static const char kWgInterposerFile[] = "{interposer_name()}";', inc)
        self.assertIn("#define WG_BINDING_HAS_FRAME_TOKEN_GUID 1", inc)
        m = re.search(r"kWgGuid_frame_token\[16\] = \{([^}]*)\}", inc)
        self.assertEqual(bytes(int(x) for x in m.group(1).split(",")).hex(), "350f0a8384db7141a80459b206499b18")
        self.assertIn("#define WG_BINDING_ORIGINAL_FROM_STEM 1", inc)
        self.assertIn('static const wchar_t kWgOriginalFile[] = L".wg-original.dll";', inc)
        self.assertIn("#define WG_BINDING_HAS_TAG_CONVERSION 0", inc)
        # A9.6 GUID bytes in the in-memory GUID layout (first three fields little-endian)
        m = re.search(r"kWgGuid_viewport\[16\] = \{([^}]*)\}", inc)
        self.assertEqual([int(x) for x in m.group(1).split(",")],
                         [0x35, 0x64, 0x1b, 0x17, 0x3c, 0x9b, 0xc8, 0x4f, 0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4])
        m = re.search(r"kWgGuid_tag\[16\] = \{([^}]*)\}", inc)
        self.assertEqual(bytes(int(x) for x in m.group(1).split(",")).hex(), "ad5a6a4c45b46c4987ff1af3845be653")
        roles = re.search(r"kWgRoleNames\[19\] = \{\n(.*?)\n\};", inc, re.S).group(1).splitlines()
        self.assertEqual(len(roles), 19)
        for line, role in zip(roles, gf.ROLES):
            self.assertIn(f"// {role}", line)
        n7, files7 = self.gen(self.b7)
        self.assertIn('  "",  // tag-for-frame', files7["wg_binding.inc"])
        self.assertIn("#define WG_BINDING_VERSION_MAGIC 0xfedcu", files7["wg_binding.inc"])
        # A9.6 structure names for the probe log
        self.assertIn("#define WG_BINDING_GUID_NAME_COUNT 15u", inc)
        self.assertIn('{"constants", {', inc)

    def test_deterministic(self):
        _, a = self.gen(self.b14)
        _, b = self.gen(self.b14)
        self.assertEqual(a, b)
        shuffled = copy.deepcopy(self.b14)
        shuffled["exports"] = list(reversed(shuffled["exports"]))
        _, c = self.gen(shuffled)
        self.assertEqual(a, c)

    def refused(self, mutate, needle, base=None):
        b = copy.deepcopy(base or self.b14)
        mutate(b)
        with self.assertRaises(gf.BindingError) as e:
            gf.validate(b)
        self.assertIn(needle, str(e.exception))

    def test_refusals(self):
        ex = lambda b: b["exports"]  # noqa: E731
        self.refused(lambda b: ex(b).pop(0), "A9.3 says 262")
        self.refused(lambda b: ex(b).append({"ordinal": 263, "name": "vkExtra"}), "A9.3 says 262")
        self.refused(lambda b: ex(b)[0].__setitem__("ordinal", 2), "ASCII rank")
        self.refused(lambda b: [e.__setitem__("ordinal", e["ordinal"] + 1) for e in ex(b)], "ASCII rank")
        self.refused(lambda b: ex(b)[0].__setitem__("name", ex(b)[1]["name"]), "duplicate")
        self.refused(lambda b: ex(b)[0].__setitem__("name", "bad name"), "not plain identifiers")
        self.refused(lambda b: ex(b)[0].__setitem__("extra", 1), "exactly {ordinal, name}")
        self.refused(lambda b: ex(b)[0].__setitem__("name", "WrongGPU_GetSettingsContract"), "collides")
        self.refused(lambda b: ex(b)[40].__setitem__("name", "slNewCall"), "without a role")
        self.refused(lambda b: ex(b)[40].__setitem__("name", "slNewCall"), "export groups")
        self.refused(lambda b: b["roles"].pop("evaluate"), "missing roles")
        self.refused(lambda b: b["roles"].__setitem__("evaluate", "slMissing"), "not an export")
        self.refused(lambda b: b["roles"].__setitem__("evaluate", b["roles"]["init"]), "share one export")
        self.refused(lambda b: b["roles"].__setitem__("frobnicate", "CreateDXGIFactory"), "unknown roles")
        self.refused(lambda b: b.__setitem__("rr_feature_functions", {"rr-optimal": "a", "rr-state": "a",
                                                                      "rr-options": "c"}), "must differ")
        self.refused(lambda b: b.__setitem__("rr_feature_functions", {"rr-optimal": "a"}), "rr-state missing")
        self.refused(lambda b: b.pop("rr_feature_functions"), "rr_feature_functions must name exactly")
        self.refused(lambda b: b["rr_feature_functions"].__setitem__("rr-extra", "x"), "must name exactly")
        self.refused(lambda b: b.__setitem__("interposer_file_name", "dir/x.dll"), "interposer_file_name")
        self.refused(lambda b: b.__setitem__("probe_file_name", "probe.exe"), "probe_file_name")
        self.refused(lambda b: b["guids"].__setitem__("frobnicate", b["guids"]["tag"]), "unknown guids")
        self.refused(lambda b: b["guids"].__setitem__("tag", "not-a-guid"), "invalid GUID")
        self.refused(lambda b: b["guids"].pop("viewport"), "guids.viewport")
        self.refused(lambda b: b["guids"].__setitem__("tag", b["guids"]["viewport"]), "GUIDs must differ")
        self.refused(lambda b: b.__setitem__("binding_mode", "magic"), "binding_mode")
        self.refused(lambda b: b.__setitem__("version_magic", 70000), "16-bit")
        self.refused(lambda b: b.__setitem__("version_magic", "fedc"), "16-bit")
        self.refused(lambda b: b.__setitem__("ordinal_base", 0), "ordinal_base")
        self.refused(lambda b: b.__setitem__("api_version", "2.9.0"), "2.7.x, 2.8.x or 2.14.x")
        self.refused(lambda b: b.__setitem__("kind", "synthetic-test"), "kind")
        self.refused(lambda b: b.__setitem__("kind", "test-names"), "kind")  # the real names are in the tables now
        self.refused(lambda b: b.__setitem__("original_renamed_file_name", "wg_rr_proxy.dll"), "another file name")
        self.refused(lambda b: b.__setitem__("original_renamed_file_name", "{stem}.dll"), "another file name")
        self.refused(lambda b: b.__setitem__("original_renamed_file_name", "x{stem}.dll"), "{stem}")
        self.refused(lambda b: b.__setitem__("proxy_file_name", "{stem}.dll"), "{stem}")
        self.refused(lambda b: b.__setitem__("proxy_file_name", "..\\x.dll"), "bare .dll")
        self.refused(lambda b: b["roles"].__setitem__("init", "TODO(x)"), "TODO at: $.roles.init")
        # the 2.14.1-only role in a 2.7.x table
        self.refused(lambda b: b["roles"].__setitem__("tag-for-frame", "slSetTag"), "only in 2.14.1", self.b7)
        # the 2.14.1 export list in a 2.7.x table
        self.refused(lambda b: b.__setitem__("exports", copy.deepcopy(self.b14["exports"])), "A9.3 says 261", self.b7)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(table(14), "rb") as f:
                b = json.loads(f.read().decode("utf-8"))
            for k in ("exports_file", "rr_feature_functions_file", "interposer_file"):
                b[k] = os.path.join(BINDING, b[k])
            b["version_magic"] = "TODO"
            with open(bad, "wb") as f:
                f.write(json.dumps(b).encode())
            out = os.path.join(d, "out")
            self.assertEqual(gf.main(["x", bad, out]), 1)
            self.assertFalse(os.path.exists(out))  # nothing written for a refused table
            self.assertEqual(gf.main(["x", table(7), out]), 0)
            self.assertEqual(sorted(os.listdir(out)),
                             ["wg_binding.inc", "wg_binding.json", "wg_forward_stubs.asm", "wg_proxy.def",
                              "wg_version.rc"])
            with open(os.path.join(out, "wg_binding.json"), "rb") as f:
                m = json.loads(f.read().decode())
            self.assertEqual((m["exports"], m["forwarded"], m["own_export"]["ordinal"]), (261, 246, 262))
            self.assertEqual(m["feature_functions"], "named")
            self.assertEqual((m["proxy_file"], m["interposer_file"], m["probe"]),
                             ("wg_rr_proxy.dll", interposer_name(), False))
            self.assertEqual(len(m["rr_functions_sha256"]), 64)
            # --probe and --output-name
            out2 = os.path.join(d, "out2")
            self.assertEqual(gf.main(["x", table(14), out2, "--probe", "--output-name", interposer_name()]), 0)
            with open(os.path.join(out2, "wg_binding.json"), "rb") as f:
                m2 = json.loads(f.read().decode())
            self.assertEqual((m2["proxy_file"], m2["probe"]), (interposer_name(), True))
            self.assertEqual(gf.main(["x", table(14), out2, "--output-name"]), 2)
            self.assertEqual(len(m["exports_sha256"]), 64)


class NamesAndProbe(unittest.TestCase):
    def setUp(self):
        self.b7 = load(7)
        self.b14 = load(14)

    def gen(self, b, probe=False, output_name=None):
        with tempfile.TemporaryDirectory() as d:
            n = gf.generate_from(b, os.path.join(d, "out"), probe=probe, output_name=output_name)
            files = {}
            for name in os.listdir(os.path.join(d, "out")):
                with open(os.path.join(d, "out", name), "rb") as f:
                    files[name] = f.read().decode()
            return n, files

    def test_names_files(self):
        # A9.5 / A10.3: three distinct identifiers in role order; the tables resolve them
        d = names_file("rr-functions.json")
        self.assertEqual(d["roles"], ["rr-optimal", "rr-state", "rr-options"])
        self.assertEqual(len(set(d["names"])), 3)
        for b in (self.b7, self.b14):
            self.assertEqual(b["rr_feature_functions"], rr_names())
            self.assertEqual(b["interposer_file_name"], interposer_name())
            self.assertEqual(b["probe_file_name"], "wg_rr_proxy_probe.dll")
        self.assertTrue(gf.bare_dll(interposer_name()))
        # each name stands alone on its own line (the leak-scan allow rule for binding/)
        for fname, values in (("rr-functions.json", d["names"]), ("interposer.json", [interposer_name()])):
            with open(os.path.join(BINDING, fname), encoding="utf-8") as f:
                lines = f.read().splitlines()
            for v in values:
                holding = [l for l in lines if v in l]
                self.assertEqual(len(holding), 1, v)
                self.assertRegex(holding[0], r'^\s*"' + re.escape(v) + r'",?$')

    def test_names_nowhere_else_in_clean(self):
        # the four names appear only in binding/ (code refers to them by role)
        clean = os.path.dirname(os.path.dirname(PROXY))
        needles = [x.lower() for x in list(rr_names().values()) + [interposer_name()]]
        hits = []
        for root, dirs, files in os.walk(clean):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
            if os.path.abspath(root) == os.path.abspath(BINDING):
                continue
            for fn in files:
                path = os.path.join(root, fn)
                try:
                    with open(path, "rb") as f:
                        text = f.read().decode("utf-8", "replace").lower()
                except OSError:
                    continue
                hits += [(path, x) for x in needles if x in text]
        self.assertEqual(hits, [])

    def test_probe_generation(self):
        for b, minor in ((self.b7, 7), (self.b14, 14)):
            n, files = self.gen(b, probe=True)
            self.assertTrue(n["probe"])
            self.assertEqual(n["proxy_file"], "wg_rr_proxy_probe.dll")
            d = dict((name, target) for name, target, _ in DEF_RX.findall(files["wg_proxy.def"]))
            # every API role is a typed role function (A10), so each call is logged with its result
            for role, name in b["roles"].items():
                self.assertEqual(d[name], "wg_role_" + role.replace("-", "_"), name)
            self.assertEqual(len(n["forwarded"]), 14 + 229)
            for name in n["ordered"]:
                if not gf.is_api(name):
                    self.assertTrue(d[name].startswith("wg_fwd_"), name)
            asm = files["wg_forward_stubs.asm"]
            self.assertIn("EXTERN wg_fwd_note:PROC", asm)
            self.assertNotIn("wg_fwd_resolve", asm)
            for i in range(len(n["forwarded"])):
                self.assertIn(f"wg_fwd_{i} PROC\n  mov r11d, {i}\n  jmp wg_fwd_slow", asm)
            self.assertIn("#define WG_BINDING_PROBE 1", files["wg_binding.inc"])
            # ordinals and the own export are the release build's
            nr, fr = self.gen(b)
            self.assertEqual(n["ordinals"], nr["ordinals"])
            self.assertEqual(sorted(DEF_RX.findall(files["wg_proxy.def"]), key=lambda x: int(x[2]))[-1][2],
                             str(nr["own_ordinal"]))

    def test_hook_generation(self):
        for b, minor, release_rc, iface_fv in ((self.b7, 7, "0.6.0.0", "2.7.1.0"),
                                               (self.b14, 14, "0.6.0.0", "2.14.1.0")):
            for probe in (False, True):
                with tempfile.TemporaryDirectory() as d:
                    nn = gf.generate_from(b, os.path.join(d, "out"), probe=probe, hook=True)
                    files = {}
                    for name in os.listdir(os.path.join(d, "out")):
                        with open(os.path.join(d, "out", name), "rb") as f:
                            files[name] = f.read().decode()
                # hook outputs: InitializeASI + the contract export only, WrongGPU's own VERSIONINFO, no stubs
                self.assertTrue(nn["hook"])
                self.assertEqual(nn["proxy_file"], b["hook_probe_file_name"] if probe else b["hook_file_name"])
                self.assertTrue(nn["proxy_file"].endswith(".asi"))
                self.assertIn("wg_hook.def", files)
                self.assertNotIn("wg_forward_stubs.asm", files)
                self.assertNotIn("wg_proxy.def", files)
                deff = files["wg_hook.def"]
                self.assertIn("InitializeASI", deff)
                self.assertIn("WrongGPU_GetSettingsContract=wg_get_settings_contract", deff)
                inc = files["wg_binding.inc"]
                self.assertIn("#define WG_BINDING_HOOK 1", inc)
                self.assertIn("kWgInterposerFileVersion", inc)
                self.assertIn("kWgHookRoles", inc)
                # the roles redirected: the probe hooks every API role; the release hooks the intercepted ones
                m = re.search(r"WG_BINDING_HOOK_ROLE_COUNT (\d+)u", inc)
                self.assertTrue(m)
                want = (19 if minor == 14 else 18) if probe else (15 if minor == 7 else 16)
                self.assertEqual(int(m.group(1)), want)
                rc = files["wg_version.rc"]
                self.assertIn(f"\n FILEVERSION {release_rc.replace('.', ',')}\r", rc)  # WrongGPU's release version
                vals = dict(re.findall(r'^   VALUE "(\w+)", "([^"]*)"\r$', rc, re.M))
                self.assertEqual(vals["ProductName"], "WrongGPU RR hook")
                self.assertIn("WrongGPU ray reconstruction hook-mode plugin", vals["FileDescription"])
                self.assertEqual(vals["OriginalFilename"], nn["proxy_file"])
        # the hook names live in the tables with a .asi extension
        for b in (self.b7, self.b14):
            self.assertTrue(gf.bare_asi(b["hook_file_name"]) and gf.bare_asi(b["hook_probe_file_name"]))
        with self.assertRaises(gf.BindingError):  # a non-.asi hook output name is refused
            gf.validate(self.b14, output_name="x.dll", hook=True)

    def test_version_resource(self):
        # VERSIONINFO of every build: file / product version = the game interposer version of the table (other mods
        # read it from the interposer file), WrongGPU's own strings, release or probe in the description
        for b, dotted in ((self.b7, "2.7.1.0"), (self.b14, "2.14.1.0")):
            for probe in (False, True):
                n, files = self.gen(b, probe=probe)
                rc = files["wg_version.rc"]
                comma = dotted.replace(".", ",")
                self.assertIn(f"\n FILEVERSION {comma}\r", rc)
                self.assertIn(f"\n PRODUCTVERSION {comma}\r", rc)
                values = dict(re.findall(r'^   VALUE "(\w+)", "([^"]*)"\r$', rc, re.M))
                self.assertEqual(values, {
                    "CompanyName": "WrongGPU",
                    "FileDescription": f"WrongGPU ray reconstruction proxy ({'probe' if probe else 'release'})",
                    "FileVersion": dotted, "ProductVersion": dotted, "InternalName": "wg_rr_proxy",
                    "OriginalFilename": interposer_name(), "ProductName": "WrongGPU RR proxy"})
                self.assertIn("WrongGPU", values["ProductName"])  # the install tool's double-install guard
                self.assertIn('VALUE "Translation", 0x409, 1200', rc)
                self.assertEqual(n["file_version"], [int(x) for x in dotted.split(".")])
        # the numbers live in the table, with the comment saying what they are for
        for b, fv in ((self.b7, [2, 7, 1, 0]), (self.b14, [2, 14, 1, 0])):
            self.assertEqual(b["file_version"], fv)
            self.assertIn("version number other mods read from the interposer file", b["file_version_comment"])
        for bad in ([2, 7, 1], [2, 7, 1, 70000], [2, 7, 2, 0], [2, 7, 1, 1], "2.7.1.0", None):
            with self.assertRaises(gf.BindingError):  # missing, not VERSIONINFO-sized, or not the interposer version
                gf.validate(dict(self.b7, file_version=bad))

    def test_output_name(self):
        n, files = self.gen(self.b14, output_name=interposer_name())
        self.assertEqual(n["proxy_file"], interposer_name())
        self.assertTrue(files["wg_proxy.def"].startswith(f'LIBRARY "{interposer_name()[:-4]}"\n'))
        n, files = self.gen(self.b14)
        self.assertTrue(files["wg_proxy.def"].startswith('LIBRARY "wg_rr_proxy"\n'))
        with self.assertRaises(gf.BindingError):
            gf.validate(self.b14, output_name="x/y.dll")
        with self.assertRaises(gf.BindingError):  # the proxy may not take the renamed original's name
            gf.validate(dict(self.b14, original_renamed_file_name="same.dll"), output_name="same.dll")


if __name__ == "__main__":
    unittest.main()
