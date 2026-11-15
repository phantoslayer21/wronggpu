// SPDX-License-Identifier: MIT
// Unit tests of the launcher's part of wg_core (build/launcher-design.md section 7.1) that need no game, no payload and
// no GPU: settings validation and rendering (runtime_options as one line without spaces, enum rejection, the last value
// wins, runtime keys on lines of their own), ini round trips that keep comments, order and unknown keys, outside-edit
// detection, the profile file, record v1 -> v2 (owners, payload line, owned-files.json, a 0.5.x reader still reads
// it), the log-tail parser on synthetic logs, the job and result files, the update reply, the game environment, Ray
// Reconstruction (its settings, the RR settings file round trip, which games use it, the RR log).
// Written for this project. Run by ctest ("launcher-core").
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "app/ui_palette.h"
#include "wg_core/detect.h"
#include "wg_core/gamerules.h"
#include "wg_core/install.h"
#include "wg_core/launcher.h"
#include "wg_core/logs.h"
#include "wg_core/logtail.h"
#include "wg_core/optiedition.h"
#include "wg_core/pe.h"
#include "wg_core/profile.h"
#include "wg_core/record.h"
#include "wg_core/rr.h"
#include "wg_core/rrconfig.h"
#include "wg_core/settings.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
void require(bool ok, const std::string& what) { require(ok, what.c_str()); }
wstring TempFolder() {
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    wstring dir = Join(tmp, L"launcher-core-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
string OptionsOf(const string& ini) { string v, s; IniGetFrontEnd(ini, "runtime_options", &v, &s); return v; }

// A minimal PE32+ DLL with a version resource of `strings` and `extra` bytes after it (a port of tests/mockpe.py make_dll
// without exports or imports, from Microsoft's public PE/COFF and VS_VERSIONINFO documentation): never loaded, only read.
void Le(string& b, uint64_t v, int n) { for (int i = 0; i < n; ++i) b += char((v >> (8 * i)) & 0xFF); }
string U16z(const wstring& s) { string r; for (wchar_t c : s) Le(r, (uint16_t)c, 2); Le(r, 0, 2); return r; }
void Pad(string& b, size_t a) { while (b.size() % a) b += '\0'; }
string VNode(const wstring& key, const string& value, bool text, std::vector<string> children = {}) {
    string body;
    Le(body, 0, 2); Le(body, 0, 2); Le(body, text ? 1 : 0, 2);
    body += U16z(key);
    Pad(body, 4);
    body += value;
    if (!children.empty()) { Pad(body, 4); for (auto& c : children) { Pad(c, 4); body += c; } }
    string out;
    Le(out, body.size(), 2); Le(out, text ? value.size() / 2 : value.size(), 2);
    return out + body.substr(4);
}
string MockDll(const std::vector<std::pair<wstring, wstring>>& strings, const string& extra) {
    string fixed;
    for (uint32_t v : {0xFEEF04BDu, 0x00010000u, 0x00010000u, 0u, 0x00010000u, 0u, 0x3Fu, 0u, 0x40004u, 2u, 0u, 0u, 0u}) Le(fixed, v, 4);
    std::vector<string> entries;
    for (auto& s : strings) entries.push_back(VNode(s.first, U16z(s.second), true));
    string tr; Le(tr, 0x409, 2); Le(tr, 1200, 2);
    const string vi = VNode(L"VS_VERSION_INFO", fixed, false,
                            {VNode(L"StringFileInfo", "", false, {VNode(L"040904B0", "", false, entries)}), VNode(L"VarFileInfo", "", false, {VNode(L"Translation", tr, false)})});
    const uint32_t fa = 0x200, sa = 0x1000;
    string rsrc;
    auto dirHead = [&] { Le(rsrc, 0, 4); Le(rsrc, 0, 4); Le(rsrc, 0, 2); Le(rsrc, 0, 2); Le(rsrc, 0, 2); Le(rsrc, 1, 2); };
    dirHead(); Le(rsrc, 16, 4); Le(rsrc, 0x80000000u | 0x18, 4);
    dirHead(); Le(rsrc, 1, 4); Le(rsrc, 0x80000000u | 0x30, 4);
    dirHead(); Le(rsrc, 0x409, 4); Le(rsrc, 0x48, 4);
    Le(rsrc, sa + 0x58, 4); Le(rsrc, vi.size(), 4); Le(rsrc, 0, 4); Le(rsrc, 0, 4);
    rsrc.resize(0x58, '\0');
    rsrc += vi;
    const size_t rsrcSize = rsrc.size();
    string content = rsrc; Pad(content, 16);
    string raw = content; Pad(raw, fa);
    const uint32_t vsize = (uint32_t)content.size(), image = sa + ((vsize + sa - 1) / sa) * sa;
    string h = "MZ"; h.resize(0x3C, '\0'); Le(h, 0x40, 4);
    h += string("PE\0\0", 4);
    Le(h, 0x8664, 2); Le(h, 1, 2); Le(h, 0, 4); Le(h, 0, 4); Le(h, 0, 4); Le(h, 240, 2); Le(h, 0x2022, 2);
    Le(h, 0x20B, 2); Le(h, 14, 1); Le(h, 0, 1); Le(h, 0, 4); Le(h, raw.size(), 4); Le(h, 0, 4); Le(h, 0, 4); Le(h, sa, 4);
    Le(h, 0x180000000ull, 8); Le(h, sa, 4); Le(h, fa, 4);
    for (int v : {6, 0, 0, 0, 6, 0}) Le(h, v, 2);
    Le(h, 0, 4); Le(h, image, 4); Le(h, 0x200, 4); Le(h, 0, 4); Le(h, 2, 2); Le(h, 0x160, 2);
    Le(h, 0x100000, 8); Le(h, 0x1000, 8); Le(h, 0x100000, 8); Le(h, 0x1000, 8); Le(h, 0, 4); Le(h, 16, 4);
    for (int d = 0; d < 16; ++d) { Le(h, d == 2 ? sa : 0, 4); Le(h, d == 2 ? rsrcSize : 0, 4); }
    string name = ".rsrc"; name.resize(8, '\0');
    h += name; Le(h, vsize, 4); Le(h, sa, 4); Le(h, raw.size(), 4); Le(h, fa, 4); Le(h, 0, 4); Le(h, 0, 4); Le(h, 0, 2); Le(h, 0, 2); Le(h, 0x40000040u, 4);
    h.resize(fa, '\0');
    return h + raw + extra;
}
}  // namespace

int main() {
    try {
        // ---- 0.6.1: Collect logs in uninstall.exe (no converter data) never claims the map is invalid ----------------
        {
            const wstring cl = TempFolder() + L"-collect";
            RemoveTree(cl);
            EnsureDir(cl);
            const wstring model = Join(cl, L"model.wgcm");
            const wstring tail = L" bytes, not verified here (uninstall.exe has no converter data; use Collect logs in the WrongGPU launcher to verify)";
            require(ModelDataUnverified(model) == L"missing", "uninstall.exe's Model data: missing");
            const string bytes(1234, 'm');
            WriteFileAtomic(model, bytes.data(), bytes.size());
            require(ModelDataUnverified(model) == L"present, 1234" + tail, "uninstall.exe's Model data: present, its size, not verified here");
            HANDLE lock = CreateFileW(model.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);  // no sharing: unreadable
            const wstring locked = ModelDataUnverified(model);
            if (lock != INVALID_HANDLE_VALUE) CloseHandle(lock);
            require(lock != INVALID_HANDLE_VALUE && locked.rfind(L"present but unreadable (", 0) == 0 && locked.back() == L')' && locked.size() > 26,
                    "uninstall.exe's Model data: present but unreadable (the reason)");
            DeleteFileW(model.c_str());
            EnsureDir(model);
            require(ModelDataUnverified(model) == L"present but unreadable (a folder, not a file)", "uninstall.exe's Model data: a folder in its place");
            // which process says what: uninstall.exe's table (as restore_main.cpp makes it) has no converter data; WrongGPU.exe's
            // has (its lines are the 0.6.0 code, unchanged: the suite's collect comparison checks them byte for byte)
            static const EmbeddedData restorer{nullptr, 0, 0, L"", 0, L"", 0, L"0.6.1"};
            static const EmbeddedData launcherLike{nullptr, 0, 901, L"00", 902, L"00", 0, L"0.6.1", L"0.6.1", L"0.6.1"};
            Context rc, lc;
            rc.data = &restorer;
            lc.data = &launcherLike;
            require(!HasConverterData(rc, false) && !HasConverterData(rc, true) && HasConverterData(lc, false), "the converter data: uninstall.exe none, WrongGPU.exe its map");
            RemoveTree(cl);
        }
        // ---- Model L (#89): the model by name (model convert/status --preset=); its converter map and model folder --------
        {
            static const EmbeddedData withL{nullptr, 0, 900, L"00", 0, L"", 0, L"0.7.0", L"0.7.0", L"0.7.0", nullptr, 0, 0, L"", L"", 903, L"00"};
            static const EmbeddedData noL{nullptr, 0, 900, L"00", 0, L"", 0, L"0.7.0"};
            Context lc, nc;
            lc.data = &withL;
            nc.data = &noL;
            wstring err;
            require(SelectedPreset(lc) == L'M' && ModelDir(lc) == ModelBaseDir(), "Model M by default: the model folder itself");
            lc.mapError = L"an earlier load";
            require(SelectPreset(lc, L"l", err) && lc.presetL && SelectedPreset(lc) == L'L' && ModelDir(lc) == Join(ModelBaseDir(), L"L") &&
                        ModelFile(lc) == Join(Join(ModelBaseDir(), L"L"), L"model.wgcm") && lc.mapError.empty(),
                    "--preset=l: Model L, its folder <model>\\L (the runtime's folder for model=L); the earlier map state is dropped");
            require(SelectPreset(lc, L"L", err) && lc.presetL, "the same model again: nothing changes");
            require(SelectPreset(lc, L"M", err) && !lc.presetL && SelectedPreset(lc) == L'M' && ModelDir(lc) == ModelBaseDir(), "back to Model M: the model folder itself");
            require(!SelectPreset(lc, L"Q", err) && err.rfind(L"Unknown model \"Q\"", 0) == 0 && SelectedPreset(lc) == L'M', "an unknown name is refused; nothing changes");
            require(!SelectPreset(lc, L"ML", err) && !SelectPreset(lc, L"", err) && SelectedPreset(lc) == L'M', "a model name is one letter");
            require(!SelectPreset(nc, L"L", err) && err == L"This build of WrongGPU has no converter data for Model L." && !nc.presetL && ModelDir(nc) == ModelBaseDir(),
                    "a build without Model L's map refuses L by name; nothing changes");
#if !WRONGGPU_PRESET_K
            require(!SelectPreset(lc, L"K", err) && err.find(L"M or L") != wstring::npos, "K is not a model of a build without Preset K");
#endif
        }
#if !WRONGGPU_MODEL_L
        // ---- a build without the Model L choice (the default): no model setting anywhere; model=L refused by name -----------
        {
            Profile mp = DefaultProfile();
            wstring mwhy;
            require(!FindSetting("model") && !SetSetting(mp, "model", "L", &mwhy) && mwhy == kNoModelChoice,
                    "no Model L choice in this build: set model=L is refused by name");
            const string lfile = "[fsr]\r\nruntime_options = timing_log=1;model=L\r\n";
            string lout;
            require(RenderWronggpuIni(mp, lfile, lout, &mwhy) && lout == lfile && ProfileText(mp).find("model=") == string::npos,
                    "no Model L choice: a model item in wronggpu.ini is left as it is, profile.ini has no model line");
            const string slotIni = "[WrongGPU]\r\nUpscalerIndex=1\r\n";
            RenderOptiIni(mp, slotIni, lout);
            require(lout == slotIni, "no Model L choice: OptiScaler's UpscalerIndex is left as it is");
        }
#endif
        // ---- validation -----------------------------------------------------------------------------------------
        string n; wstring why;
        const SettingDef* exec = FindSetting("executor");
        const SettingDef* filter = FindSetting("output_filter");
        const SettingDef* sparkle = FindSetting("sparkle_guard");
        require(exec && filter && sparkle && FindSetting("wronggpu") && FindSetting("pre_exposure") && FindSetting("fps_overlay"), "the Basic tier has its six settings");
        require(ValidateSetting(*exec, "D3D12", &n, &why) && n == "dx12", "executor d3d12 -> dx12");
        require(!ValidateSetting(*exec, "vulkan", &n, &why) && why.find(L"auto, dx12, hip") != wstring::npos, "a bad executor is refused and the allowed values named");
        require(!ValidateSetting(*filter, "quality7", &n, &why), "output_filter: only the runtime's values");
        require(ValidateSetting(*filter, "quality9", &n, &why) && n == "quality9", "output_filter quality9");
        require(ValidateSetting(*sparkle, "0.25", &n, &why) && n == "0.25", "sparkle_guard rate");
        require(ValidateSetting(*sparkle, "0", &n, &why) && n == "off" && ValidateSetting(*sparkle, "on", &n, &why) && n == "on", "sparkle_guard 0 = off, on");
        require(!ValidateSetting(*sparkle, "1.5", &n, &why) && !ValidateSetting(*sparkle, "-0.1", &n, &why) && !ValidateSetting(*sparkle, "abc", &n, &why), "sparkle_guard rate must be in (0, 1)");
        Profile p = DefaultProfile();
        require(!SetSetting(p, "nosuchkey", "1", &why) && !SetSetting(p, "pre_exposure", "bright", &why), "unknown keys and bad values are refused by set");
        require(SetSetting(p, "Executor", " HIP ", &why) && p.get("executor") == "hip", "set trims and lowers");

        // ---- rendering wronggpu.ini ---------------------------------------------------------------------------
        const string front = "# WrongGPU settings\r\n[fsr]\r\n# who upscales\r\nupscaler = network\r\nruntime_dll = wronggpu_runtime.dll\r\n"
                             "runtime_options = model=M ; the model\r\nfoo_unknown = 7\r\n[overlay]\r\nexecutor = keep-me\r\n";
        p = DefaultProfile();
        string out;
        require(RenderWronggpuIni(p, front, out, &why) && out == front, "the default profile changes nothing (defaults are never written)");
        SetSetting(p, "executor", "dx12", nullptr);
        SetSetting(p, "output_filter", "quality5", nullptr);
        SetSetting(p, "sparkle_guard", "0.3", nullptr);
        require(RenderWronggpuIni(p, front, out, &why), "render");
        require(OptionsOf(out) == "model=M;executor=dx12;output_filter=quality5;sparkle_guard=0.3", "runtime_options: unmanaged items first, managed ones appended in order");
        require(out.find("runtime_options=model=M;executor=dx12;output_filter=quality5;sparkle_guard=0.3\r\n") != string::npos, "runtime_options is one line with no spaces");
        require(out.find("# who upscales\r\n") != string::npos && out.find("foo_unknown = 7") != string::npos && out.find("[overlay]\r\nexecutor = keep-me") != string::npos,
                "comments, unknown keys and other sections are kept");
        string again;
        require(RenderWronggpuIni(p, out, again, &why) && again == out, "rendering twice gives the same file (round trip)");
        auto eff = EffectiveSettings(out, "");
        require(eff["executor"] == "dx12" && eff["output_filter"] == "quality5" && eff["sparkle_guard"] == "0.3" && eff["wronggpu"] == "on", "the file reads back as the profile");
        // a runtime key on a line of its own wins (the front end appends it); rendering removes such lines
        const string loose = "[fsr]\r\nruntime_options = executor=hip;output_filter=analytic;executor=dx12\r\noutput_filter = quality9\r\n";
        eff = EffectiveSettings(loose, "");
        require(eff["executor"] == "dx12", "the last value of a key in runtime_options wins");
        require(eff["output_filter"] == "quality9", "a runtime key on its own line comes after runtime_options and wins");
        Profile q = DefaultProfile();
        SetSetting(q, "output_filter", "original", nullptr);
        require(RenderWronggpuIni(q, loose, out, &why) && OptionsOf(out) == "output_filter=original" && out.find("output_filter = quality9") == string::npos,
                "rendering folds the loose line away and writes one value");
        // WrongGPU off
        Profile off = DefaultProfile();
        SetSetting(off, "wronggpu", "off", nullptr);
        require(RenderWronggpuIni(off, front, out, &why) && out.find("upscaler=amd") != string::npos && out.find("upscaler = network") == string::npos, "off writes upscaler=amd in place");
        require(RenderWronggpuIni(DefaultProfile(), out, again, &why) && EffectiveSettings(again, "")["wronggpu"] == "on", "on again: upscaler=network");
        // a profile value that does not validate is never written
        Profile bad = DefaultProfile();
        bad.v["executor"] = "warp";
        require(!RenderWronggpuIni(bad, front, out, &why) && why.find(L"warp") != wstring::npos, "an invalid profile value stops the render");
        // empty file: nothing for defaults, an [fsr] section for the rest
        require(RenderWronggpuIni(DefaultProfile(), "", out, &why) && out.empty(), "defaults into an empty file: nothing");
        require(RenderWronggpuIni(p, "", out, &why) && OptionsOf(out) == "executor=dx12;output_filter=quality5;sparkle_guard=0.3", "a new [fsr] section");

        // ---- outside edits ------------------------------------------------------------------------------------------
        Profile prof = DefaultProfile();
        SetSetting(prof, "output_filter", "quality5", nullptr);
        const string edited = "[fsr]\r\nruntime_options = output_filter=quality9;pre_exposure=scale\r\n";
        auto edits = OutsideEdits(prof, edited, "");
        require(edits.size() == 1 && edits[0].key == "output_filter" && edits[0].fileValue == "quality9" && edits[0].profileValue == "quality5",
                "a value that differs from the default and the profile is an outside edit");
        require(OutsideEdits(prof, "[fsr]\r\nruntime_options = output_filter=default\r\n", "").empty(), "a value equal to its default is ignored");
        require(OutsideEdits(prof, "[fsr]\r\nruntime_options = output_filter=quality5\r\n", "").empty(), "the profile's own value is not an outside edit");

        // ---- OptiScaler.ini -------------------------------------------------------------------------------------------
        const string opti = "[Menu]\r\n; the overlay\r\nShowFps=auto\r\nShortcutKey=auto\r\n[FSR]\r\nUpscalerIndex=0\r\n";
        Profile fps = DefaultProfile();
        require(RenderOptiIni(fps, opti, out) && out == opti, "fps_overlay keep: OptiScaler.ini unchanged");
        SetSetting(fps, "fps_overlay", "on", nullptr);
        require(RenderOptiIni(fps, opti, out) && out.find("ShowFps=true\r\n") != string::npos && out.find("; the overlay\r\n") != string::npos && out.find("ShortcutKey=auto") != string::npos,
                "fps_overlay on: ShowFps=true, the rest kept");
        require(EffectiveSettings("", out)["fps_overlay"] == "on", "ShowFps reads back");
        std::vector<wstring> notes;
        const string noKey = "[Menu]\r\nShortcutKey=auto\r\n";
        require(RenderOptiIni(fps, noKey, out, &notes) && out == noKey && notes.size() == 1, "a key the OptiScaler build does not have is not added");
        require(IniHasKey(opti, "menu", "showfps") && !IniHasKey(opti, "FSR", "ShowFps"), "IniHasKey: section and key without case, only in that section");

        // ---- import (adoption) ----------------------------------------------------------------------------------------
        const Profile imp = ImportProfile("[fsr]\r\nupscaler = standin\r\nruntime_options = model=M;output_filter=quality9;timing_log=1\r\n", "", &notes);
        require(imp.get("wronggpu") == "on" && imp.get("output_filter") == "quality9" && imp.get("executor") == "auto", "import: standin counts as on, the player's filter kept");

        // ---- profile.ini -------------------------------------------------------------------------------------------------
        const wstring dir = TempFolder();
        const wstring pp = ProfilePath(dir);
        Profile saved = DefaultProfile();
        SetSetting(saved, "pre_exposure", "original", nullptr);
        saved.v["future_key"] = "x";
        SaveProfile(pp, saved);
        Profile loaded;
        require(LoadProfile(pp, loaded) && loaded.get("pre_exposure") == "original" && loaded.get("future_key") == "x" && loaded.get("executor") == "auto",
                "profile.ini round trip keeps unknown keys");
        { const string broken = "[profile]\r\nexecutor=warp\r\n"; WriteFileAtomic(pp, broken.data(), broken.size()); }
        std::vector<wstring> problems;
        require(LoadProfile(pp, loaded, &problems) && loaded.get("executor") == "auto" && problems.size() == 1, "a bad profile value falls back to the default, reported");

        // ---- record v1 -> v2 -------------------------------------------------------------------------------------------
        const wstring game = Join(dir, L"game");
        EnsureDir(StateDir(game));
        const string v1 = "WRONGGPU-RECORD 1\r\nversion\t0.5.1\r\ngame\t" + ToUtf8(game) + "\r\nexe\t" + ToUtf8(Join(game, L"Game.exe")) +
                          "\r\nstatus\texperimental\r\nregkey\tWrongGPU-12345678\r\nmodel\tm.wgcm\r\n"
                          "dir\t0\t-\t-\tplugins\r\ncopy\t0\t-\taaa\twronggpu_ffx_dx12.dll\r\ncopy\t0\t-\tbbb\twronggpu_runtime.dll\r\n"
                          "copy\t0\t-\tccc\tplugins\\WrongGPU.asi\r\ncopy\t0\t-\tddd\tdxgi.dll\r\ncopy\t1\teee\tfff\tOptiScaler.ini\r\n"
                          "copy\t0\t-\t111\twronggpu.ini\r\ndisable\t1\t999\t-\twinmm.dll\r\n";
        WriteFileAtomic(RecordPath(game), v1.data(), v1.size());
        Context ctx;
        static const EmbeddedData noPayload{nullptr, 0, 0, L"", 0, L"", 0, L"0.5.5"};
        ctx.data = &noPayload;
        Record r1;
        require(LoadRecord(RecordPath(game), r1) && r1.format == 1 && r1.e.size() == 8, "v1 record read");
        require(UpgradeRecord(ctx, game, &notes), "v1 -> v2");
        require(IsFile(RecordPath(game) + L".v1"), "the v1 record is kept as install.record.v1");
        string rawv1; ReadAll(RecordPath(game) + L".v1", rawv1);
        require(rawv1 == v1, "install.record.v1 is the old file byte for byte");
        Record r2;
        require(LoadRecord(RecordPath(game), r2) && r2.format == 2 && r2.e.size() == 8 && r2.regkey == L"WrongGPU-12345678", "v2 record read back");
        auto owner = [&](const wchar_t* rel) { for (auto& e : r2.e) if (IEq(e.rel, rel)) return e.owner; return wstring(L"?"); };
        require(owner(L"wronggpu_ffx_dx12.dll") == L"wg" && owner(L"wronggpu_runtime.dll") == L"wg" && owner(L"plugins\\WrongGPU.asi") == L"wg" && owner(L"plugins") == L"wg",
                "owners: our payload is wg");
        require(owner(L"dxgi.dll") == L"wg-optibundle" && owner(L"OptiScaler.ini") == L"wg-optibundle", "owners: OptiScaler's files of the bundle are wg-optibundle");
        require(owner(L"winmm.dll") == L"moved" && owner(L"wronggpu.ini") == L"wg", "owners: a file moved aside is moved");
        require(r2.payloadVersion == L"0.5.1" && r2.runtimeSha == L"bbb" && r2.frontEndSha == L"aaa" && r2.pluginSha == L"ccc" && r2.proxy == L"dxgi.dll",
                "payload and proxy lines from the v1 entries");
        require(!UpgradeRecord(ctx, game, nullptr), "a v2 record is not upgraded again");
        string rawv2; ReadAll(RecordPath(game), rawv2);
        require(rawv2.rfind("WRONGGPU-RECORD 2\r\n", 0) == 0, "v2 header");
        // a 0.5.x reader (header prefix, five-field action lines) still sees every action line of a v2 record
        size_t actions = 0;
        for (size_t a = 0; a < rawv2.size();) {
            size_t e = rawv2.find('\n', a); if (e == string::npos) e = rawv2.size();
            const string line = rawv2.substr(a, e - a); a = e + 1;
            size_t tabs = 0; for (char c : line) tabs += c == '\t';
            const string act = line.substr(0, line.find('\t'));
            if ((act == "copy" || act == "disable" || act == "rename" || act == "dir") && tabs == 4) ++actions;
        }
        require(actions == 8, "v2 keeps the five-field action lines of v1");
        // a line of a later version survives a rewrite
        string withExtra = rawv2 + "coexist\tReShade\tfp\twarn\r\n";
        WriteFileAtomic(RecordPath(game), withExtra.data(), withExtra.size());
        Record r3; LoadRecord(RecordPath(game), r3); SaveRecord(RecordPath(game), r3);
        string rawv3; ReadAll(RecordPath(game), rawv3);
        require(rawv3.find("coexist\tReShade\tfp\twarn\r\n") != string::npos, "unknown v2 lines are written back");
        const string js = OwnedFilesJson(r2);
        require(js.find("\"format\": \"wronggpu-owned-files/1\"") != string::npos && js.find("\"owner\": \"wg-optibundle\"") != string::npos &&
                js.find("plugins\\\\WrongGPU.asi") != string::npos && js.find("\"original_sha256\": \"999\"") != string::npos, "owned-files.json");
        // VerifyInstall
        { const string b = "runtime"; WriteFileAtomic(Join(game, L"wronggpu_runtime.dll"), b.data(), b.size()); }
        auto vp = VerifyInstall(game, r2);
        require(std::find(vp.begin(), vp.end(), wstring(L"changed: wronggpu_runtime.dll")) != vp.end() &&
                std::find(vp.begin(), vp.end(), wstring(L"missing: wronggpu_ffx_dx12.dll")) != vp.end(), "VerifyInstall: changed and missing files");

        // ---- the payload line names the RR plugin build (rc6 L6): another RR plugin build of the same version is another build ----
        {
            static const PayloadFile pfiles[] = {{1, L"wronggpu_runtime.dll", L"r1", 1}, {2, L"amd_fidelityfx_dx12.dll", L"f1", 1}};
            static const PayloadFile prr[] = {{3, L"plugins\\WrongGPU-RR.asi", L"p1", 1}};
            EmbeddedData pd{pfiles, 2, 0, L"", 0, L"", 0, L"0.6.0"};
            pd.rrFiles = prr; pd.rrCount = 1;
            Context pc; pc.data = &pd;
            const wstring pg = Join(dir, L"payload-line");
            Record m; m.version = L"0.6.0"; m.game = pg; m.exe = Join(pg, L"Game.exe");
            Entry rre; rre.action = L"copy"; rre.rel = L"plugins\\WrongGPU-RR.asi"; rre.inst = L"p1"; m.e.push_back(rre);
            SetPayloadLine(pc, m);
            require(m.rrPluginSha == L"p1" && m.runtimeSha == L"r1" && !PayloadDiffers(pc, m), "payload line: the RR plugin build this payload has for the game");
            SaveRecord(RecordPath(pg), m);
            string rawp; ReadAll(RecordPath(pg), rawp);
            require(rawp.find("payload\t0.6.0\tr1\tf1\t-\tp1\r\n") != string::npos, "payload line: the RR plugin's hash is its sixth field");
            Record back;
            require(LoadRecord(RecordPath(pg), back) && back.rrPluginSha == L"p1" && back.pluginSha.empty() && !PayloadDiffers(pc, back), "payload line: read back");
            back.rrPluginSha = L"p0";
            require(PayloadDiffers(pc, back), "payload line: another RR plugin build of the same version differs");
            back.rrPluginSha.clear();
            back.e[0].inst = L"p0";
            require(PayloadDiffers(pc, back), "payload line before rc6 (no sixth field): the RR plugin entry's hash decides");
            back.e[0].inst = L"p1";
            require(!PayloadDiffers(pc, back), "payload line before rc6: the same RR plugin is the same build");
            Record norr; norr.version = L"0.6.0"; norr.game = pg; norr.exe = m.exe;
            SetPayloadLine(pc, norr);
            SaveRecord(RecordPath(pg), norr);
            ReadAll(RecordPath(pg), rawp);
            require(norr.rrPluginSha.empty() && rawp.find("payload\t0.6.0\tr1\tf1\t-\r\n") != string::npos && !PayloadDiffers(pc, norr),
                    "payload line: an install without the RR files keeps five fields");
        }

        // ---- a moved game folder (rc6 L1): the record names the old folder; it is the new folder's record all the same ------
        {
            const wstring oldLad = EnvVar(L"LOCALAPPDATA"), lad = Join(dir, L"moved-localappdata");
            auto file = [](const wstring& p, const string& text) { EnsureDir(Parent(p)); WriteFileAtomic(p, text.data(), text.size()); };
            file(Join(lad, L"WrongGPU\\launcher.ini"), "[launcher]\r\n");  // a launcher uses the shared data: an uninstall leaves it
            SetEnvironmentVariableW(L"LOCALAPPDATA", lad.c_str());
            const wstring oldDir = Join(dir, L"OldLib\\steamapps\\common\\Moved Game"), newDir = Join(dir, L"NewLib\\steamapps\\common\\Moved Game");
            const string rt = "runtime", ini = "[FSR]\r\nFfxDx12SRPath = " + ToUtf8(oldDir) + "\\wronggpu_ffx_dx12.dll\r\nUpscalerIndex=0\r\n[Plugins]\r\nPath=" +
                                               ToUtf8(oldDir) + "\\plugins\r\n";
            file(Join(newDir, L"Game.exe"), "MZ");
            file(Join(newDir, L"wronggpu_runtime.dll"), rt);
            file(Join(newDir, L"OptiScaler.ini"), ini);
            Record m; m.version = L"0.6.0"; m.game = oldDir; m.exe = Join(oldDir, L"Game.exe"); m.status = L"experimental"; m.model = L"m.wgcm";
            Entry a; a.action = L"copy"; a.rel = L"wronggpu_runtime.dll"; a.inst = Sha256Bytes(rt.data(), rt.size()); a.owner = L"wg"; m.e.push_back(a);
            Entry b; b.action = L"copy"; b.rel = L"OptiScaler.ini"; b.inst = Sha256Bytes(ini.data(), ini.size()); b.owner = L"wg-optibundle"; m.e.push_back(b);
            IniKeyChange k; k.file = L"OptiScaler.ini"; k.section = "Plugins"; k.key = "Path"; k.original = "auto"; k.ours = ToUtf8(oldDir) + "\\plugins"; m.inikeys.push_back(k);
            SaveRecord(RecordPath(newDir), m);
            AddManagedGame(oldDir);
            Record r;
            require(LoadRecord(RecordPath(newDir), r) && IEq(r.game, newDir) && IEq(r.exe, Join(newDir, L"Game.exe")) && IEq(r.movedFrom, oldDir),
                    "moved game: the record read from the new folder names it (game, exe), movedFrom the old one");
            require(r.inikeys.size() == 1 && r.inikeys[0].ours == ToUtf8(newDir) + "\\plugins" && r.inikeys[0].original == "auto", "moved game: the journal's paths follow");
            // another drive letter: only the changed part of the path is replaced
            Record drive = m; drive.game = L"Q:\\Games\\Moved Game"; drive.exe = L"Q:\\Games\\Moved Game\\Game.exe"; drive.model = L"Q:\\model.wgcm";
            SaveRecord(RecordPath(newDir), drive);
            Record dr;
            require(LoadRecord(RecordPath(newDir), dr) && IEq(dr.exe, Join(newDir, L"Game.exe")) && dr.model == L"Q:\\model.wgcm" && dr.inikeys[0].ours == ToUtf8(oldDir) + "\\plugins",
                    "moved game: a drive letter change; the model path and values of other folders stay");
            Record own = m, same;
            own.game = newDir; own.exe = Join(newDir, L"Game.exe");
            SaveRecord(RecordPath(newDir), own);
            require(LoadRecord(RecordPath(newDir), same) && same.movedFrom.empty() && same.game == newDir, "moved game: a record of its own folder is not rebased");
            // written back: the record, OptiScaler.ini's paths, the managed list (the old folder is gone)
            SaveRecord(RecordPath(newDir), m);
            Context mc; mc.logFile = Join(dir, L"moved.log");
            mc.data = &noPayload;  // an empty payload table (the scan and EntryFor read it)
            require(LoadRecord(RecordPath(newDir), r) && SaveRebasedRecord(mc, newDir, r) && r.movedFrom.empty(), "moved game: the rebased record is written");
            Record savedRec;
            string rawRec, iniNow;
            ReadAll(RecordPath(newDir), rawRec);
            require(rawRec.find("game\t" + ToUtf8(newDir) + "\r\n") != string::npos && LoadRecord(RecordPath(newDir), savedRec) && savedRec.movedFrom.empty(), "moved game: the saved record names the new folder");
            require(ReadAll(Join(newDir, L"OptiScaler.ini"), iniNow) && iniNow.find("FfxDx12SRPath = " + ToUtf8(newDir) + "\\wronggpu_ffx_dx12.dll\r\n") != string::npos &&
                        iniNow.find("Path=" + ToUtf8(newDir) + "\\plugins\r\n") != string::npos && iniNow.find("UpscalerIndex=0\r\n") != string::npos,
                    "moved game: OptiScaler.ini's paths into the old folder point into the new one");
            require(savedRec.e[1].inst == Sha256Bytes(iniNow.data(), iniNow.size()), "moved game: the record's OptiScaler.ini hash follows the rewritten file");
            auto managed = [&](const wstring& f) { for (auto& l : ManagedGames()) if (IEq(FullPath(l), FullPath(f))) return true; return false; };
            require(managed(newDir) && !managed(oldDir), "moved game: the new folder is on the managed list in the old one's place");
            // Uninstall (and uninstall.exe) of a record that still names the old folder: no refusal, the new folder's files go
            SaveRecord(RecordPath(newDir), m);
            {  // (owned-files.json is what WrongGPU makes of the record; the test put an older record back, so it is made again)
                Record again;
                LoadRecord(RecordPath(newDir), again);
                WriteOwnedFiles(newDir, again);
            }
            Ui quiet{[](int, int, const wstring&) {}, [](const wstring&) {}};
            bool refused = false;
            try { DoRestore(mc, newDir, quiet, false, false); } catch (const SetupError&) { refused = true; }
            require(!refused && !IsFile(Join(newDir, L"wronggpu_runtime.dll")) && !IsDir(StateDir(newDir)) && IsFile(Join(newDir, L"Game.exe")),
                    "moved game: the uninstall works in the new folder");

            // ---- a hand-placed WrongGPU DLL under the game's FSR name (a direct install) goes only when the game's
            // original is there to put back (the backup, or the kept *.wronggpu-original.dll); else the uninstall refuses ----
            {
                const FfxSlot& sl = kSlots[0];
                const string orig = "the game's own FidelityFX DLL (synthetic)", installed = "WrongGPU's FSR DLL as installed (synthetic)";
                const string hand = MockDll({{L"CompanyName", L"WrongGPU project"}, {L"ProductName", L"WrongGPU"}, {L"FileDescription", L"WrongGPU front end (synthetic)"}},
                                            "a newer build placed by hand");
                auto direct = [&](const wstring& name, bool backup, bool keptCopy) {
                    const wstring d = Join(dir, L"Direct Slot\\" + name);
                    file(Join(d, L"Game.exe"), "MZ");
                    file(Join(d, sl.name), hand);
                    if (keptCopy) file(Join(d, sl.original), orig);
                    if (backup) file(BackupPath(d, sl.name), orig);
                    Record m; m.version = L"0.4.5"; m.game = d; m.exe = Join(d, L"Game.exe"); m.status = L"experimental"; m.model = L"m.wgcm";
                    Entry r; r.action = L"rename"; r.rel = sl.original; r.inst = Sha256Bytes(orig.data(), orig.size()); r.owner = L"moved"; m.e.push_back(r);
                    Entry c; c.action = L"copy"; c.rel = sl.name; c.existed = true; c.orig = r.inst; c.inst = Sha256Bytes(installed.data(), installed.size()); c.owner = L"wg";
                    m.e.push_back(c);
                    SaveRecord(RecordPath(d), m);
                    return d;
                };
                wstring info;
                require(ClassifySlot(sl, [&] { const wstring p = Join(dir, L"slot-probe.dll"); file(p, hand); return p; }(), nullptr, info) == FFX_OURS,
                        "hand-placed FSR DLL: the hand-placed DLL is WrongGPU's by its version resource (the synthetic PE is read)");
                for (int how = 0; how < 2; ++how) {  // the original from the backup, then from the kept copy alone
                    const wstring d = direct(how ? L"Kept Copy" : L"Backup", how == 0, true);
                    std::vector<wstring> w;
                    bool threw = false;
                    try { w = DoRestore(mc, d, quiet, false, false); } catch (const SetupError&) { threw = true; }
                    string now;
                    require(!threw && ReadAll(Join(d, sl.name), now) && now == orig && !IsFile(Join(d, sl.original)) && !IsDir(StateDir(d)),
                            how ? "hand-placed FSR DLL: no backup, the kept copy there: the hand-placed DLL is replaced by the game's original"
                                : "hand-placed FSR DLL: the backup there: the hand-placed DLL is replaced by the game's original");
                }
                {  // neither the backup nor the kept copy: the uninstall refuses, as before 0.6.1 ("Nothing was changed")
                    const wstring d = direct(L"No Original", false, false);
                    bool threw = false;
                    wstring msg;
                    try { DoRestore(mc, d, quiet, false, false); } catch (const SetupError& ex) { threw = true; msg = ex.msg; }
                    string now;
                    require(threw && msg.find(L"Nothing was changed") != wstring::npos && ReadAll(Join(d, sl.name), now) && now == hand && IsFile(RecordPath(d)) &&
                                IsFile(Join(d, L"Game.exe")),
                            "game FSR DLL, no original anywhere: the uninstall refuses (\"Nothing was changed\"), the DLL and the record stay");
                    const wstring u = direct(L"No Original Update", false, false);
                    threw = false;
                    try { DoRestore(mc, u, quiet, false, true); } catch (const SetupError&) { threw = true; }
                    require(threw && ReadAll(Join(u, sl.name), now) && now == hand && IsFile(RecordPath(u)),
                            "game FSR DLL, no original anywhere, an update's restore: refused, nothing changed (record kept)");
                }
                {  // a stop after WrongGPU's DLL was deleted and before the game's file took its name: the second run puts it there
                    const wstring d = Join(dir, L"Direct Slot\\Name Free");
                    file(Join(d, L"Game.exe"), "MZ");
                    file(Join(d, sl.original), orig);
                    Record rm; rm.version = L"0.4.5"; rm.game = d; rm.exe = Join(d, L"Game.exe"); rm.status = L"experimental"; rm.model = L"m.wgcm";
                    Entry re; re.action = L"rename"; re.rel = sl.original; re.inst = Sha256Bytes(orig.data(), orig.size()); re.owner = L"moved"; rm.e.push_back(re);
                    SaveRecord(RecordPath(d), rm);
                    std::vector<wstring> w;
                    bool threw = false;
                    try { w = DoRestore(mc, d, quiet, false, false); } catch (const SetupError&) { threw = true; }
                    string now;
                    require(!threw && ReadAll(Join(d, sl.name), now) && now == orig && !IsFile(Join(d, sl.original)) && !AnyPartly(w),
                            "game FSR DLL: the name is free and the game's file is kept aside: it goes back under its name (no warning, nothing lost)");
                }
                {  // another file at the name: it is never replaced; the game's file is kept (WrongGPU-originals) and said
                    const wstring d = Join(dir, L"Direct Slot\\Name Taken");
                    file(Join(d, L"Game.exe"), "MZ");
                    file(Join(d, sl.original), orig);
                    file(Join(d, sl.name), "a file another program put here (synthetic)");
                    Record rm; rm.version = L"0.4.5"; rm.game = d; rm.exe = Join(d, L"Game.exe"); rm.status = L"experimental"; rm.model = L"m.wgcm";
                    Entry re; re.action = L"rename"; re.rel = sl.original; re.inst = Sha256Bytes(orig.data(), orig.size()); re.owner = L"moved"; rm.e.push_back(re);
                    SaveRecord(RecordPath(d), rm);
                    std::vector<wstring> w;
                    bool threw = false;
                    try { w = DoRestore(mc, d, quiet, false, false); } catch (const SetupError&) { threw = true; }
                    string now, keptNow;
                    bool kept = false;
                    for (auto& g : ListDirs(Join(d, kKeptDirName))) kept = kept || (ReadAll(Join(g, sl.name), keptNow) && keptNow == orig);
                    require(!threw && ReadAll(Join(d, sl.name), now) && now == "a file another program put here (synthetic)" && kept && AnyPartly(w),
                            "game FSR DLL: another program's file at the name stays, the game's own file is kept in WrongGPU-originals, the result says partly");
                }
            }
            // ---- 0.6.1: the file-ownership rule (record.h): a restore deletes or overwrites only WrongGPU's bytes; anything else
            // stays as it is, and a player's original that cannot go back is kept in WrongGPU-originals (both generations
            // survive); a stop at any point loses nothing, and a second run finishes ------------------------------------------
            {
                struct Killed {};
                auto H = [](const string& s) { return Sha256Bytes(s.data(), s.size()); };
                auto read = [](const wstring& p) { string s; if (!ReadAll(p, s)) return string("(missing)"); return s; };
                auto keptOf = [&](const wstring& d, const wstring& rel) {  // every kept generation of rel, oldest name first
                    std::vector<string> out;
                    for (auto& g : ListDirs(Join(d, kKeptDirName))) if (IsFile(Join(g, rel))) out.push_back(read(Join(g, rel)));
                    return out;
                };
                struct F { wstring rel, action; string orig, ours, now; };  // orig: there before WrongGPU ("" none); now: "" missing
                auto ownGame = [&](const wstring& name, const std::vector<F>& fs) {
                    const wstring d = Join(dir, L"Ownership\\" + name);
                    file(Join(d, L"Game.exe"), "MZ");
                    Record m; m.version = L"0.6.1"; m.format = 2; m.game = d; m.exe = Join(d, L"Game.exe"); m.status = L"experimental"; m.model = L"m.wgcm";
                    for (auto& f : fs) {
                        if (!f.now.empty()) file(Join(d, f.rel), f.now);
                        if (!f.orig.empty()) file(BackupPath(d, f.rel), f.orig);
                        Entry e; e.action = f.action; e.rel = f.rel; e.existed = !f.orig.empty() && f.action == L"copy";
                        e.orig = f.orig.empty() ? wstring() : H(f.orig);
                        if (f.action == L"copy") e.inst = H(f.ours);
                        e.owner = f.action == L"copy" ? L"wg" : L"moved";
                        m.e.push_back(e);
                    }
                    m.runtime.push_back({L".", L"wronggpu-*"});
                    SaveRecord(RecordPath(d), m);
                    return d;
                };
                std::string stopAt;
                SetRestoreStepHook([&](const char* s) { if (stopAt == s) throw Killed{}; });
                bool killed = false;
                auto restore = [&](const wstring& d, bool keepState, std::vector<wstring>* w = nullptr, wstring* err = nullptr) {
                    killed = false;
                    try { const auto r = DoRestore(mc, d, quiet, false, keepState); if (w) *w = r; return true; }
                    catch (const SetupError& e) { if (err) *err = e.msg; return false; }
                    catch (const Killed&) { killed = true; return false; }
                };
                auto partly = [](const std::vector<wstring>& w, const wstring& rel) {
                    for (auto& x : w) if (x.rfind(kPartlyRestored, 0) == 0 && x.find(rel) != wstring::npos && x.find(L"is kept in") != wstring::npos) return true;
                    return false;
                };
                const string A = "the file that was there before WrongGPU (synthetic)\r\n", W = "what WrongGPU wrote (synthetic)\r\n",
                             B = "the unchanged file changed since (synthetic)\r\n", C = "another setup's file put over it (synthetic)\r\n";
                const string P = "the player's own OptiScaler DLL (synthetic)", O = "the OptiScaler DLL WrongGPU's setup placed (synthetic)";
                for (int keep = 0; keep < 2; ++keep) {
                    const wstring how = keep ? L" (repair/update)" : L" (uninstall)";
                    const string howA = keep ? " (an update or repair's restore)" : " (an uninstall)";
                    // wronggpu.ini: A there before, WrongGPU wrote W, edited by hand to B: B stays, A is kept; neither is lost
                    {
                        const wstring d = ownGame(L"Settings A-B" + how, {{L"wronggpu.ini", L"copy", A, W, B}});
                        std::vector<wstring> w;
                        require(restore(d, keep == 1, &w) && read(Join(d, L"wronggpu.ini")) == B && keptOf(d, L"wronggpu.ini") == std::vector<string>{A} &&
                                    partly(w, L"wronggpu.ini") && AnyPartly(w) && !IsFile(RecordPath(d)),
                                "ownership: wronggpu.ini A before, edited to B: B stays, A is kept in WrongGPU-originals, the result says partly" + howA);
                        const wstring u = ownGame(L"Settings A-W" + how, {{L"wronggpu.ini", L"copy", A, W, W}});
                        w.clear();
                        require(restore(u, keep == 1, &w) && read(Join(u, L"wronggpu.ini")) == A && !IsDir(Join(u, kKeptDirName)) && !AnyPartly(w),
                                "ownership: wronggpu.ini as WrongGPU last wrote it: the original A goes back, nothing kept, no partly" + howA);
                        const wstring noneBefore = ownGame(L"Settings none-B" + how, {{L"wronggpu.ini", L"copy", "", W, B}});
                        w.clear();
                        require(restore(noneBefore, keep == 1, &w) && read(Join(noneBefore, L"wronggpu.ini")) == B && !AnyPartly(w),
                                "ownership: wronggpu.ini edited by hand, none before: left in place (never removed by its name)" + howA);
                    }
                    // OptiScaler.ini: the player's A switched over, then saved in OptiScaler's menu (B), WrongGPU's OptiScaler DLL
                    // untouched; and replaced in full by another file (C), the DLL untouched: the ini stays, A is kept, the
                    // player's own OptiScaler DLL comes back
                    for (const string& now : {B, C}) {
                        const wstring d = ownGame(wstring(now == B ? L"Opti menu" : L"Opti replaced") + how,
                                               {{L"winmm.dll", L"copy", P, O, O}, {L"OptiScaler.ini", L"copy", A, W, now}});
                        std::vector<wstring> w;
                        require(restore(d, keep == 1, &w) && read(Join(d, L"OptiScaler.ini")) == now && keptOf(d, L"OptiScaler.ini") == std::vector<string>{A} &&
                                    read(Join(d, L"winmm.dll")) == P && partly(w, L"OptiScaler.ini"),
                                string("ownership: OptiScaler.ini ") + (now == B ? "saved by OptiScaler's menu" : "replaced in full") +
                                    " beside WrongGPU's untouched OptiScaler DLL: it stays, the player's A is kept, their DLL comes back" + howA);
                    }
                    // OptiScaler.log: the player's A moved aside, then another log B written there: B stays, A is kept
                    {
                        const wstring d = ownGame(L"Log A-B" + how, {{L"OptiScaler.ini", L"copy", "", W, W}, {L"OptiScaler.log", L"disable", A, "", B}});
                        std::vector<wstring> w;
                        require(restore(d, keep == 1, &w) && read(Join(d, L"OptiScaler.log")) == B && keptOf(d, L"OptiScaler.log") == std::vector<string>{A} &&
                                    !IsFile(Join(d, L"OptiScaler.ini")) && partly(w, L"OptiScaler.log"),
                                "ownership: OptiScaler.log A moved aside, another log B there now: B stays (never deleted by its name), A is kept" + howA);
                        const wstring g2 = ownGame(L"Log A-gone" + how, {{L"OptiScaler.log", L"disable", A, "", ""}});
                        require(restore(g2, keep == 1) && read(Join(g2, L"OptiScaler.log")) == A && !IsDir(Join(g2, kKeptDirName)),
                                "ownership: OptiScaler.log A moved aside, nothing there now: A goes back" + howA);
                    }
                }
                // two generations: A kept by a first restore; the next install over B, changed to C: B is kept too, A stays
                {
                    const wstring d = ownGame(L"Generations", {{L"wronggpu.ini", L"copy", A, W, B}});
                    require(restore(d, true), "ownership: generations: the first restore");
                    const wstring nextGen = ownGame(L"Generations", {{L"wronggpu.ini", L"copy", B, W, C}});
                    std::vector<wstring> w;
                    const auto kept = restore(nextGen, false, &w) ? keptOf(nextGen, L"wronggpu.ini") : std::vector<string>{};
                    require(kept.size() == 2 && std::count(kept.begin(), kept.end(), A) == 1 && std::count(kept.begin(), kept.end(), B) == 1 &&
                                read(Join(nextGen, L"wronggpu.ini")) == C,
                            "ownership: two generations (A, then B) are both kept, neither overwritten; C stays");
                }
                // files under WrongGPU's names that the install never recorded (what its programs wrote while the game ran, or a
                // file a player or another program created there after the install): never deleted by their names, listed
                {
                    const wstring d = ownGame(L"Runtime names", {{L"wronggpu_runtime.dll", L"copy", "", W, W}});
                    file(Join(d, L"wronggpu-frame.log"), "a log WrongGPU's runtime wrote");
                    file(Join(d, L"wronggpu-notes.txt"), "the player's notes, saved after the install");
                    file(Join(d, L"OptiScaler.log"), "a log another program wrote");
                    file(Join(d, L"fakenvapi.log"), "a log another program wrote");
                    file(Join(d, L"sub\\wronggpu-other.log"), "in a folder the record does not list");
                    const auto found = UnrecordedRuntimeFiles(d, [&] { Record r; LoadRecord(RecordPath(d), r); return r; }());
                    require(found.size() == 2 && IEq(found[0], Join(d, L"wronggpu-frame.log")) && IEq(found[1], Join(d, L"wronggpu-notes.txt")),
                            "ownership: the unrecorded wronggpu-* files of the listed folders are found (the recorded one, other names and unlisted folders are not)");
                    std::vector<wstring> w;
                    const auto listed = [](const std::vector<wstring>& ws, const wchar_t* name) {
                        for (auto& x : ws) if (x.rfind(kLeftUnrecorded, 0) == 0 && x.find(name) != wstring::npos) return true;
                        return false;
                    };
                    require(restore(d, false, &w) && !IsFile(Join(d, L"wronggpu_runtime.dll")) && !IsFile(RecordPath(d)),
                            "ownership: an uninstall with unrecorded wronggpu-* files runs to its end (the recorded file goes)");
                    require(read(Join(d, L"wronggpu-frame.log")) == "a log WrongGPU's runtime wrote" && read(Join(d, L"wronggpu-notes.txt")) == "the player's notes, saved after the install" &&
                                IsFile(Join(d, L"OptiScaler.log")) && IsFile(Join(d, L"fakenvapi.log")) && IsFile(Join(d, L"sub\\wronggpu-other.log")),
                            "ownership: a file created under our prefix after the install is not deleted (no file of the folder is: logs, notes, other names, unlisted folders)");
                    require(listed(w, L"wronggpu-notes.txt") && listed(w, L"wronggpu-frame.log") && !AnyPartly(w),
                            "ownership: the uninstall's result lists the files it left in place (\"left in place\", not \"partly\": nothing was lost)");
                    // an update's or repair's restore leaves them too, and does not list them again
                    const wstring u = ownGame(L"Runtime names update", {{L"wronggpu_runtime.dll", L"copy", "", W, W}});
                    file(Join(u, L"wronggpu-notes.txt"), "the player's notes");
                    w.clear();
                    require(restore(u, true, &w) && read(Join(u, L"wronggpu-notes.txt")) == "the player's notes" && !listed(w, L"wronggpu-notes.txt"),
                            "ownership: an update's restore leaves them in place without a warning each time");
                }
                // a recorded file the install created (nothing was there) whose bytes changed since is not WrongGPU's by its name:
                // wronggpu-*.ini/log/txt, a file in a WrongGPU folder, a program file; unchanged, each goes
                {
                    const std::vector<wstring> names = {L"wronggpu-rr.ini", L"wronggpu-frame.log", L"WrongGPU\\wronggpu.ini", L"wronggpu_runtime.dll"};
                    for (int keep = 0; keep < 2; ++keep) {
                        const string howA = keep ? " (an update or repair's restore)" : " (an uninstall)";
                        for (auto& name : names) {
                            const wstring d = ownGame(L"Created changed " + wstring(keep ? L"keep " : L"") + Stem(name) + L"-" + std::to_wstring(&name - &names[0]), {{name, L"copy", "", W, B}});
                            std::vector<wstring> w;
                            bool said = false;
                            const bool ok = restore(d, keep == 1, &w);
                            for (auto& x : w) said = said || x == L"Left in place (changed since install): " + name;
                            require(ok && read(Join(d, name)) == B && said && !AnyPartly(w) && !IsFile(RecordPath(d)),
                                    "ownership: a recorded file created by the install and changed since (" + ToUtf8(name) + ") stays with its new bytes and is said" + howA);
                            const wstring g = ownGame(L"Created same " + wstring(keep ? L"keep " : L"") + Stem(name) + L"-" + std::to_wstring(&name - &names[0]), {{name, L"copy", "", W, W}});
                            require(restore(g, keep == 1) && !IsFile(Join(g, name)),
                                    "ownership: ...the same file as the install wrote it goes (" + ToUtf8(name) + ")" + howA);
                        }
                    }
                }
                // a recorded file replaced after the install by another program's file: stays, with the player's original kept
                {
                    const wstring d = ownGame(L"Replaced after install", {{L"wronggpu_runtime.dll", L"copy", "", W, C}, {L"wronggpu.ini", L"copy", A, W, C}});
                    std::vector<wstring> w;
                    require(restore(d, false, &w) && read(Join(d, L"wronggpu_runtime.dll")) == C && read(Join(d, L"wronggpu.ini")) == C &&
                                keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && partly(w, L"wronggpu.ini"),
                            "ownership: recorded files replaced after the install stay (both), the player's original of the one that had one is kept, and the result says so");
                }
                // the game root above the install folder (bin\x64 under it): a named runtime line, resolved from the record's exe
                {
                    const wstring g = Join(dir, L"Ownership\\Game root"), bx = Join(g, L"bin\\x64");
                    file(Join(bx, L"Game.exe"), "MZ");
                    Record m; m.version = L"0.6.1"; m.game = bx; m.exe = Join(bx, L"Game.exe");
                    SetRuntimeLines(m, bx, bx);
                    require(std::any_of(m.runtime.begin(), m.runtime.end(), [](const Record::RuntimeLine& r) { return r.folder == kRuntimeGameRoot; }) &&
                                std::any_of(m.runtime.begin(), m.runtime.end(), [](const Record::RuntimeLine& r) { return r.folder == L"."; }),
                            "runtime lines: the install folder and the game root above it");
                    SaveRecord(RecordPath(bx), m);
                    Record back;
                    require(LoadRecord(RecordPath(bx), back) && back.runtime.size() == m.runtime.size(), "runtime lines: the game-root line is read back");
                    file(Join(g, L"wronggpu-root.log"), "a log WrongGPU's runtime wrote in the game's working folder");
                    file(Join(g, L"OptiScaler.log"), "a log another program wrote");
                    const auto unrecorded = UnrecordedRuntimeFiles(bx, back);
                    require(unrecorded.size() == 1 && IEq(unrecorded[0], Join(g, L"wronggpu-root.log")) && IsFile(Join(g, L"wronggpu-root.log")) && IsFile(Join(g, L"OptiScaler.log")),
                            "runtime lines: the game root's wronggpu-* is found (and never deleted); OptiScaler.log there is not a candidate");
                }
                // checked again right before the change: a file that changed after the check is left, its original kept
                {
                    const wstring d = ownGame(L"Recheck", {{L"wronggpu.ini", L"copy", A, W, W}});
                    SetRestoreStepHook([&](const char* s) { if (string(s) == "kept") file(Join(d, L"wronggpu.ini"), C); });
                    std::vector<wstring> w;
                    require(restore(d, false, &w) && read(Join(d, L"wronggpu.ini")) == C && keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && partly(w, L"wronggpu.ini"),
                            "ownership: a file changed between the check and the restore is left as it is; the original is kept");
                    SetRestoreStepHook([&](const char* s) { if (stopAt == s) throw Killed{}; });
                }
                // the replace race: a swap between the check and the action. Before the file is held (the hook "kept" runs after
                // the plan, before the action): found by the hash, the file is left and said. After it is held (the hook "locked"
                // runs with the file held and checked): no program can delete, rename, replace or write it, so the action acts on
                // the file that was checked. A path found empty that gets a file a moment later (the hook "absent") is not
                // overwritten.
                {
                    auto said = [](const std::vector<wstring>& ws, const wstring& what) { for (auto& x : ws) if (x == what) return true; return false; };
                    {  // 1. a recorded file to delete, swapped for another program's bytes before it is held
                        const wstring d = ownGame(L"Swap remove", {{L"wronggpu_runtime.dll", L"copy", "", W, W}});
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "kept") file(Join(d, L"wronggpu_runtime.dll"), C); });
                        std::vector<wstring> w;
                        require(restore(d, false, &w) && read(Join(d, L"wronggpu_runtime.dll")) == C && said(w, L"Left in place (changed since install): wronggpu_runtime.dll"),
                                "swap: a recorded file swapped for other bytes between the check and the delete is not deleted, and the result says so");
                    }
                    {  // 2. the same before an original goes back over a file WrongGPU installed
                        const wstring d = ownGame(L"Swap put back", {{L"wronggpu.ini", L"copy", A, W, W}});
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "kept") file(Join(d, L"wronggpu.ini"), C); });
                        std::vector<wstring> w;
                        require(restore(d, false, &w) && read(Join(d, L"wronggpu.ini")) == C && keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && partly(w, L"wronggpu.ini"),
                                "swap: a file swapped for other bytes before the original goes back is not overwritten; the original is kept");
                    }
                    {  // 3. a path that was empty at the plan gets a file after the open found nothing: created, never overwritten
                        const wstring d = ownGame(L"Swap absent", {{L"OptiScaler.log", L"disable", A, "", ""}});
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "absent") file(Join(d, L"OptiScaler.log"), C); });
                        std::vector<wstring> w;
                        require(restore(d, false, &w) && read(Join(d, L"OptiScaler.log")) == C && keptOf(d, L"OptiScaler.log") == std::vector<string>{A} && partly(w, L"OptiScaler.log"),
                                "swap: a file created at an empty path a moment before the original goes back is not overwritten; the original is kept");
                    }
                    // 4. the file held and checked: every way to swap it fails
                    for (int back = 0; back < 2; ++back) {
                        const wstring name = back ? L"wronggpu.ini" : L"wronggpu_runtime.dll";
                        const wstring d = ownGame(back ? L"Swap held put back" : L"Swap held remove", {{name, L"copy", back ? A : string(), W, W}});
                        const wstring other = Join(d, L"another-program.tmp"), p = Join(d, name);
                        file(other, C);
                        int attacks = 0, calls = 0;
                        SetRestoreStepHook([&](const char* s) {
                            if (string(s) != "locked") return;
                            ++calls;
                            const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
                            if (DeleteFileW(p.c_str())) ++attacks;
                            if (MoveFileExW(p.c_str(), (p + L".moved").c_str(), 0)) ++attacks;
                            if (MoveFileExW(other.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING)) ++attacks;
                            for (DWORD access : {(DWORD)GENERIC_WRITE, (DWORD)DELETE}) {
                                HANDLE h = CreateFileW(p.c_str(), access, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                                if (h != INVALID_HANDLE_VALUE) { ++attacks; CloseHandle(h); }
                            }
                        });
                        std::vector<wstring> w;
                        const bool ok = restore(d, false, &w);
                        require(ok && calls == 1 && attacks == 0 && read(other) == C && !AnyPartly(w) && (back ? read(p) == A : !IsFile(p)),
                                string("swap: while the file is held and checked, deleting, renaming, replacing and writing it all fail (") + (back ? "the original goes back over it" : "it is deleted") +
                                    "); the restore then finishes on the file that was checked");
                    }
                    SetRestoreStepHook([&](const char* s) { if (stopAt == s) throw Killed{}; });
                }
                // interrupted: a stop after the record is gone (before the backup) loses nothing; a stop before it is finished by
                // a second run; a stop right after the originals were kept keeps them once
                for (int keep = 0; keep < 2; ++keep) {
                    const wstring how = keep ? L" (repair/update)" : L" (uninstall)";
                    const string howA = keep ? " (an update or repair's restore)" : " (an uninstall)";
                    {
                        const wstring d = ownGame(L"Killed after record" + how, {{L"wronggpu.ini", L"copy", A, W, B}, {L"other.dll", L"copy", P, O, O}});
                        stopAt = "record-removed";
                        const bool done = restore(d, keep == 1), stopped = killed;
                        stopAt.clear();
                        wstring err;
                        const bool second = restore(d, keep == 1, nullptr, &err);
                        require(!done && stopped && !IsFile(RecordPath(d)) && read(Join(d, L"other.dll")) == P && read(Join(d, L"wronggpu.ini")) == B &&
                                    keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && !second && err.find(L"No WrongGPU installation record") != wstring::npos,
                                "interrupted between the record's removal and the backup's: every original is at its path or kept; a second run changes nothing" + howA);
                    }
                    {
                        const wstring d = ownGame(L"Killed before record" + how, {{L"wronggpu.ini", L"copy", A, W, B}, {L"other.dll", L"copy", P, O, O}});
                        stopAt = "restored";
                        const bool done = restore(d, keep == 1);
                        stopAt.clear();
                        std::vector<wstring> w;
                        require(!done && killed && IsFile(RecordPath(d)) && restore(d, keep == 1, &w) && read(Join(d, L"other.dll")) == P &&
                                    read(Join(d, L"wronggpu.ini")) == B && keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && !IsFile(RecordPath(d)),
                                "interrupted after the restore, before the record's removal: a second run finishes; A is kept once, B stays" + howA);
                    }
                }
                // a locked file (another program holds it with read sharing only, as a running game): refused before anything
                // changes; once it is free, the restore runs
                {
                    const wstring d = ownGame(L"Locked", {{L"wronggpu.ini", L"copy", A, W, W}, {L"other.dll", L"copy", "", O, O}});
                    HANDLE h = CreateFileW(Join(d, L"other.dll").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                    wstring err;
                    const bool done = restore(d, false, nullptr, &err);
                    const bool unchanged = read(Join(d, L"wronggpu.ini")) == W && read(Join(d, L"other.dll")) == O && IsFile(RecordPath(d)) && read(BackupPath(d, L"wronggpu.ini")) == A;
                    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                    require(h != INVALID_HANDLE_VALUE && !done && err.find(L"Nothing was changed") != wstring::npos && unchanged,
                            "interrupted: a file another program holds without delete sharing: refused before anything changes (record, backup, files as they were)");
                    require(restore(d, false) && read(Join(d, L"wronggpu.ini")) == A && !IsFile(Join(d, L"other.dll")) && !IsDir(StateDir(d)),
                            "interrupted: ...once it is free, the restore runs (the original back, WrongGPU's file gone)");
                }
                // ---- #184: Roll back puts the earlier payload only over files whose bytes are the ones the record says WrongGPU
                // installed, each held (one handle) from the check to the write; anything else stops it before any change ----------
                {
                    const string N = "the newer runtime WrongGPU installed (synthetic)", E = "the earlier runtime kept for Roll back (synthetic)",
                                 N2 = "the newer front end WrongGPU installed (synthetic)", E2 = "the earlier front end kept for Roll back (synthetic)",
                                 X = "the player's own build put over it (synthetic)";
                    const wstring rt = L"wronggpu_runtime.dll", fe = L"wronggpu_ffx_dx12.dll";
                    auto rbGame = [&](const wstring& name) {
                        const wstring d = ownGame(L"Rollback " + name, {{rt, L"copy", "", N, N}, {fe, L"copy", "", N2, N2}});
                        const wstring pv = Join(StateDir(d), L"previous");
                        file(Join(pv, rt), E);
                        file(Join(pv, fe), E2);
                        file(Join(pv, L"previous.txt"), "version=0.6.0\r\nfile=" + ToUtf8(H(E)) + "|" + ToUtf8(rt) + "\r\nfile=" + ToUtf8(H(E2)) + "|" + ToUtf8(fe) + "\r\n");
                        return d;
                    };
                    auto rollback = [&](const wstring& d) { OpRequest q; q.op = Op::Rollback; q.dir = d; return RunOp(mc, q, quiet); };
                    auto inst = [&](const wstring& d, const wstring& rel) {
                        Record r;
                        if (!LoadRecord(RecordPath(d), r)) return wstring(L"(no record)");
                        for (auto& e : r.e) if (e.action == L"copy" && IEq(e.rel, rel)) return Lower(e.inst);
                        return wstring(L"(no entry)");
                    };
                    auto prevTxt = [&](const wstring& d) { return read(Join(StateDir(d), L"previous\\previous.txt")); };
                    auto untouched = [&](const wstring& d, const string& runtimeNow) {  // nothing of the roll back happened
                        return read(Join(d, rt)) == runtimeNow && read(Join(d, fe)) == N2 && inst(d, rt) == Lower(H(N)) && inst(d, fe) == Lower(H(N2)) &&
                               prevTxt(d).find("version=0.6.0") != string::npos && read(Join(StateDir(d), L"previous\\" + rt)) == E && !IsDir(Join(StateDir(d), L"previous.swap"));
                    };
                    SetRestoreStepHook(nullptr);
                    {  // the ordinary case: both files are WrongGPU's
                        const wstring d = rbGame(L"ours");
                        const OpResult r = rollback(d);
                        require(r.code == EXIT_OK && read(Join(d, rt)) == E && read(Join(d, fe)) == E2 && inst(d, rt) == Lower(H(E)) && inst(d, fe) == Lower(H(E2)) &&
                                    prevTxt(d).find("version=0.6.1") != string::npos && read(Join(StateDir(d), L"previous\\" + rt)) == N && read(Join(StateDir(d), L"previous\\" + fe)) == N2,
                                "rollback (#184): files that are WrongGPU's get the earlier version; the newer one is kept for a later update, the record names the earlier hashes");
                    }
                    {  // a payload file the player replaced since the update: kept as it is, nothing else changes, the result names it
                        const wstring d = rbGame(L"player changed");
                        file(Join(d, rt), X);
                        const OpResult r = rollback(d);
                        bool listed = false;
                        for (auto& w : r.warnings) listed = listed || w.find(rt) != wstring::npos;
                        require(r.code == EXIT_REFUSED && r.message.find(rt) != wstring::npos && r.message.find(L"Nothing was changed") != wstring::npos && listed &&
                                    untouched(d, X),
                                "rollback (#184): a payload file the player changed is not overwritten (kept as it is); nothing else changes; the result names it");
                    }
                    {  // a hard-linked payload file: never written through
                        const wstring d = rbGame(L"hard link");
                        const wstring link = Join(d, L"players-link.dll");
                        const bool made = CreateHardLinkW(link.c_str(), Join(d, rt).c_str(), nullptr) != 0;
                        const OpResult r = rollback(d);
                        require(made && r.code == EXIT_REFUSED && r.message.find(L"hard link") != wstring::npos && untouched(d, N) && read(link) == N,
                                "rollback (#184): a payload file with another name (a hard link) is not written through; both names keep their bytes");
                        DeleteFileW(link.c_str());
                    }
                    {  // a swap between the check and the write: the files are held from the check on, every way to change one fails
                        const wstring d = rbGame(L"held");
                        const wstring p = Join(d, rt), other = Join(d, L"another-program.tmp");
                        file(other, X);
                        int attacks = 0, calls = 0;
                        SetRestoreStepHook([&](const char* s) {
                            if (string(s) != "rollback-held") return;
                            ++calls;
                            const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
                            if (DeleteFileW(p.c_str())) ++attacks;
                            if (MoveFileExW(p.c_str(), (p + L".moved").c_str(), 0)) ++attacks;
                            if (MoveFileExW(other.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING)) ++attacks;
                            for (DWORD access : {(DWORD)GENERIC_WRITE, (DWORD)DELETE}) {
                                HANDLE h = CreateFileW(p.c_str(), access, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                                if (h != INVALID_HANDLE_VALUE) { ++attacks; CloseHandle(h); }
                            }
                        });
                        const OpResult r = rollback(d);
                        SetRestoreStepHook(nullptr);
                        require(r.code == EXIT_OK && calls == 1 && attacks == 0 && read(other) == X && read(p) == E && read(Join(StateDir(d), L"previous\\" + rt)) == N,
                                "rollback (#184): between the check and the write, deleting, renaming, replacing and writing a held file all fail; the roll back acts on the checked file");
                    }
                    {  // a missing payload file is created only where nothing is
                        const wstring d = rbGame(L"missing");
                        DeleteFileW(Join(d, rt).c_str());
                        const OpResult r = rollback(d);
                        require(r.code == EXIT_OK && read(Join(d, rt)) == E && read(Join(d, fe)) == E2 && inst(d, rt) == Lower(H(E)),
                                "rollback (#184): a payload file the player deleted is created again with the earlier bytes");
                        const wstring u = rbGame(L"missing then made");
                        DeleteFileW(Join(u, rt).c_str());
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "rollback-absent") file(Join(u, rt), X); });
                        const OpResult r2 = rollback(u);
                        SetRestoreStepHook(nullptr);
                        require(r2.code == EXIT_REFUSED && r2.message.find(L"appeared") != wstring::npos && untouched(u, X),
                                "rollback (#184): a file another program creates where a payload file was missing is not replaced; nothing else changes");
                    }
                    // ---- #209 review: the kept folders go file by file, by their lists, after the writes; a write that fails part way
                    // is undone through the held handles; a payload file missing at a roll back is recorded as absent ---------------
                    auto prevFile = [&](const wstring& d, const wstring& rel) { return read(Join(StateDir(d), L"previous\\" + rel)); };
                    auto said = [](const OpResult& r, const wchar_t* what) {
                        bool found = r.message.find(what) != wstring::npos;
                        for (auto& w : r.warnings) found = found || w.find(what) != wstring::npos;
                        return found;
                    };
                    auto version = [&](const wstring& d) { Record r; return LoadRecord(RecordPath(d), r) ? r.version : wstring(L"(no record)"); };
                    int faultAt = 0, writes = 0;
                    SetFaultHook([&](const char* s) { return string(s) == "rollback-write" && ++writes == faultAt; });
                    {  // 1: a file in previous\ its list does not name stays and is said (and then the newer version can't take its place)
                        const wstring d = rbGame(L"foreign in previous");
                        const wstring foreign = Join(StateDir(d), L"previous\\players-notes.txt");
                        file(foreign, X);
                        const OpResult r = rollback(d);
                        require(r.code == EXIT_OK && read(Join(d, rt)) == E && read(Join(d, fe)) == E2 && read(foreign) == X && said(r, L"players-notes.txt") &&
                                    !IsFile(Join(StateDir(d), L"previous\\" + rt)) && !IsFile(Join(StateDir(d), L"previous\\" + fe)) &&
                                    !IsFile(Join(StateDir(d), L"previous\\previous.txt")) && !IsDir(Join(StateDir(d), L"previous.swap")) && inst(d, rt) == Lower(H(E)),
                                "rollback (#209 review): previous\\ goes only by its list (each file while its bytes are the stated hash); a file it does not name stays and is said");
                    }
                    {  // 1: ...a listed file whose bytes differ stays too (previous.txt names the bytes it must have)
                        const wstring d = rbGame(L"previous swap stale");
                        const wstring stale = Join(StateDir(d), L"previous.swap");  // an earlier roll back stopped before its end
                        file(Join(stale, rt), N);
                        file(Join(stale, fe), X);  // listed with N2's hash, other bytes now
                        file(Join(stale, L"previous.txt"), "version=0.6.1\r\nfile=" + ToUtf8(H(N)) + "|" + ToUtf8(rt) + "\r\nfile=" + ToUtf8(H(N2)) + "|" + ToUtf8(fe) + "\r\n");
                        const OpResult r = rollback(d);
                        require(r.code == EXIT_OK && read(Join(d, rt)) == E && prevFile(d, rt) == N && prevFile(d, fe) == N2 && prevTxt(d).find("version=0.6.1") != string::npos &&
                                    !IsFile(Join(stale, rt)) && read(Join(stale, fe)) == X && !IsFile(Join(stale, L"previous.txt")) && said(r, L"previous.swap") &&
                                    !IsDir(Join(StateDir(d), L"previous.swap-2")),
                                "rollback (#209 review): an earlier swap folder is never cleared before the writes; afterwards it goes by its list, a listed file with other bytes "
                                "stays and is said; the new swap folder becomes previous\\");
                    }
                    {  // 2: every earlier file is staged (held and hashed) before the first write; a damaged one changes nothing
                        const wstring d = rbGame(L"source damaged");
                        file(Join(StateDir(d), L"previous\\" + fe), X);
                        const OpResult r = rollback(d);
                        require(r.code == EXIT_FAILED && said(r, L"damaged") && read(Join(d, rt)) == N && read(Join(d, fe)) == N2 && inst(d, rt) == Lower(H(N)),
                                "rollback (#209 review): an earlier file with other bytes than its list states stops the roll back before the first write");
                        const wstring u = rbGame(L"source held");
                        const wstring s = Join(StateDir(u), L"previous\\" + fe);
                        int changed = 0;
                        SetRestoreStepHook([&](const char* st) {
                            if (string(st) != "rollback-held") return;
                            HANDLE h = CreateFileW(s.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                            if (h != INVALID_HANDLE_VALUE) { ++changed; CloseHandle(h); }
                            if (DeleteFileW(s.c_str())) ++changed;
                        });
                        const OpResult r2 = rollback(u);
                        SetRestoreStepHook(nullptr);
                        require(r2.code == EXIT_OK && changed == 0 && read(Join(u, fe)) == E2,
                                "rollback (#209 review): the earlier files are held from their check to the write (writing or deleting one meanwhile fails)");
                    }
                    {  // 2: a write fault on the second file: the first and the torn second go back from their copies; record and previous\ as they were
                        const wstring d = rbGame(L"write fault");
                        const string recBefore = read(RecordPath(d)), txtBefore = prevTxt(d);
                        writes = 0; faultAt = 2;
                        const OpResult r = rollback(d);
                        faultAt = 0;
                        require(r.code == EXIT_FAILED && writes == 2 && said(r, L"undone") && read(Join(d, rt)) == N && read(Join(d, fe)) == N2 &&
                                    read(RecordPath(d)) == recBefore && prevTxt(d) == txtBefore && prevFile(d, rt) == E && prevFile(d, fe) == E2 &&
                                    !IsDir(Join(StateDir(d), L"previous.swap")),
                                "rollback (#209 review): a write that fails on the second file puts the first and the torn second back from their copies through their held "
                                "handles; the record and previous\\ stay as they were");
                        const wstring u = rbGame(L"write fault created");  // the missing file is created last: the fault hits it
                        DeleteFileW(Join(u, rt).c_str());
                        const string recU = read(RecordPath(u));
                        writes = 0; faultAt = 2;
                        const OpResult r2 = rollback(u);
                        faultAt = 0;
                        require(r2.code == EXIT_FAILED && writes == 2 && !IsFile(Join(u, rt)) && read(Join(u, fe)) == N2 && read(RecordPath(u)) == recU && prevFile(u, fe) == E2,
                                "rollback (#209 review): ...a file the roll back created goes again (through its handle), the file already written comes back from its copy");
                    }
                    {  // 3: missing at the roll back -> "absent" in the new previous.txt; the second roll back takes the recreated file out again
                        const wstring d = rbGame(L"absent twice");
                        DeleteFileW(Join(d, rt).c_str());
                        const OpResult r1 = rollback(d);
                        const wstring txt = Lower(FromUtf8(prevTxt(d)));
                        const OpResult r2 = rollback(d);
                        require(r1.code == EXIT_OK && txt.find(Lower(L"file=absent:" + H(N) + L"|" + rt)) != wstring::npos && r2.code == EXIT_OK &&
                                    !IsFile(Join(d, rt)) && read(Join(d, fe)) == N2 && inst(d, rt) == Lower(H(N)) && inst(d, fe) == Lower(H(N2)) && version(d) == L"0.6.1" &&
                                    prevFile(d, rt) == E && prevFile(d, fe) == E2,
                                "rollback (#209 review): a payload file missing at the roll back is recorded as absent; the second roll back takes the recreated file out "
                                "(its bytes the recorded ones) and the record names the newer version again");
                        const OpResult r3 = rollback(d);
                        require(r3.code == EXIT_OK && read(Join(d, rt)) == E && read(Join(d, fe)) == E2 && version(d) == L"0.6.0",
                                "rollback (#209 review): ...and a third roll back creates it again");
                        const wstring u = rbGame(L"absent then changed");
                        DeleteFileW(Join(u, rt).c_str());
                        const OpResult u1 = rollback(u);
                        file(Join(u, rt), X);  // the player put their own file there since
                        const OpResult u2 = rollback(u);
                        require(u1.code == EXIT_OK && u2.code == EXIT_REFUSED && said(u2, L"had no such file") && said(u2, L"Nothing was changed") &&
                                    read(Join(u, rt)) == X && read(Join(u, fe)) == E2 && version(u) == L"0.6.0",
                                "rollback (#209 review): ...a file there with other bytes stops the second roll back with a clear message; nothing changes");
                    }
                    SetFaultHook(nullptr);
                }
                // ---- #186: the Engine.ini restore reads, compares and writes or deletes through one handle; a missing file is
                // created only where nothing is (CREATE_NEW) ---------------------------------------------------------------------
                {
                    const wstring d = Join(dir, L"RrConfig"), ini = Join(d, L"Config\\Windows\\Engine.ini");
                    const string own = "[Core.Log]\r\nLogNet=Log\r\n", edited = own + "\r\n" + RrConfigNewFile();
                    const string Y = "[Core.Log]\r\nLogNet=Verbose\r\n; another program's Engine.ini (synthetic)\r\n";
                    EnsureDir(StateDir(d));
                    auto fixFor = [&](bool created) {
                        Record m;
                        m.rrconfig.present = true; m.rrconfig.path = ini; m.rrconfig.created = created; m.rrconfig.prevAbsent = true;
                        m.rrconfig.sectionAdded = !created; m.rrconfig.backup = !created;
                        if (!created) file(RrConfigBackupPath(d), own);
                        return m;
                    };
                    auto readOnly = [](const wstring& p) { const DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_READONLY); };
                    auto clear = [&](const wstring& p) { SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL); DeleteFileW(p.c_str()); };
                    auto attack = [&](const wstring& other, int& attacks, int& calls) {
                        return [&, other](const char* s) {
                            if (string(s) != "rrconfig-held") return;
                            ++calls;
                            const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
                            if (DeleteFileW(ini.c_str())) ++attacks;
                            if (MoveFileExW(ini.c_str(), (ini + L".moved").c_str(), 0)) ++attacks;
                            if (MoveFileExW(other.c_str(), ini.c_str(), MOVEFILE_REPLACE_EXISTING)) ++attacks;
                            for (DWORD access : {(DWORD)GENERIC_WRITE, (DWORD)DELETE}) {
                                HANDLE h = CreateFileW(ini.c_str(), access, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                                if (h != INVALID_HANDLE_VALUE) { ++attacks; CloseHandle(h); }
                            }
                        };
                    };
                    const wstring other = Join(d, L"another-program.tmp");
                    {  // the player's Engine.ini (read-only, with our line): a replacement tried after the read fails; the line goes, read-only stays
                        Record m = fixFor(false);
                        m.rrconfig.wasReadOnly = true;
                        file(ini, edited);
                        SetFileAttributesW(ini.c_str(), FILE_ATTRIBUTE_READONLY);
                        file(other, Y);
                        int attacks = 0, calls = 0;
                        SetRestoreStepHook(attack(other, attacks, calls));
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(calls == 1 && attacks == 0 && read(other) == Y && read(ini) == own && readOnly(ini) && w.empty() && !m.rrconfig.present &&
                                    !IsFile(RrConfigBackupPath(d)),
                                "Engine.ini (#186): held from the read to the write, a replacement, delete or write by another program fails (its bytes stay its own); "
                                "the player's file gets its line out, read-only as before");
                        clear(ini);
                    }
                    {  // the file WrongGPU created: deleted through the handle it was read through; a replacement tried meanwhile fails
                        Record m = fixFor(true);
                        file(ini, RrConfigNewFile());
                        file(other, Y);
                        int attacks = 0, calls = 0;
                        SetRestoreStepHook(attack(other, attacks, calls));
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(calls == 1 && attacks == 0 && read(other) == Y && !IsFile(ini) && w.empty(),
                                "Engine.ini (#186): the file WrongGPU created is deleted through the handle it was read through; a replacement tried meanwhile fails");
                    }
                    {  // replaced before the open (another program's Engine.ini now, without our line): that file is the one read, and it stays
                        Record m = fixFor(true);
                        file(ini, Y);
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(read(ini) == Y && w.empty(), "Engine.ini (#186): a file replaced by another program before the restore is the one read: it stays byte for byte");
                        clear(ini);
                    }
                    {  // gone, and another program creates one just before the player's copy goes back: not replaced, the copy is kept, said
                        Record m = fixFor(false);
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "rrconfig-absent") file(ini, Y); });
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(read(ini) == Y && w.size() == 1 && w[0].rfind(L"Left in place: ", 0) == 0 && w[0].find(RrConfigBackupPath(d)) != wstring::npos &&
                                    read(RrConfigBackupPath(d)) == own,
                                "Engine.ini (#186): a file created by another program before the absent-file restore is not replaced; the player's copy is kept and said");
                        clear(ini);
                        clear(RrConfigBackupPath(d));
                        // ...the same when the caller holds the copy (a relocation): it is kept next to the backup path, never over a file
                        Record m2 = fixFor(false);
                        clear(RrConfigBackupPath(d));
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "rrconfig-absent") file(ini, Y); });
                        notes.clear(); w.clear();
                        RevertRrConfig(mc, d, m2, notes, w, &own);
                        SetRestoreStepHook(nullptr);
                        bool keptCopy = false;
                        for (auto& f : ListFiles(Parent(RrConfigBackupPath(d)), L"Engine.ini.kept-*")) keptCopy = keptCopy || read(f) == own;
                        require(read(ini) == Y && w.size() == 1 && w[0].find(L"Engine.ini.kept-") != wstring::npos && keptCopy,
                                "Engine.ini (#186): ...with the copy in memory, it is kept in a new file beside the backup path");
                        clear(ini);
                    }
                    {  // gone, nothing in the way: the player's copy is created again, read-only as the player had it
                        Record m = fixFor(false);
                        m.rrconfig.wasReadOnly = true;
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(read(ini) == own && readOnly(ini) && w.empty() && !IsFile(RrConfigBackupPath(d)),
                                "Engine.ini (#186): gone since, nothing in the way: the player's copy is created again (read-only as before) and the backup goes");
                        clear(ini);
                    }
                    {  // a hard-linked Engine.ini is not written through: it stays, the warning says to take the line out by hand
                        Record m = fixFor(false);
                        file(ini, edited);
                        const wstring link = Join(d, L"players-link.ini");
                        const bool made = CreateHardLinkW(link.c_str(), ini.c_str(), nullptr) != 0;
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(made && read(ini) == edited && read(link) == edited && w.size() == 1 && w[0].find(L"hard link") != wstring::npos,
                                "Engine.ini (#186): a hard-linked Engine.ini is not written through; it stays and the warning says so");
                        clear(link);
                        clear(ini);
                    }
                    // ---- #209 review: Engine.ini policy (only WrongGPU's line goes, also from a file changed elsewhere) and crash
                    // safety (a corrected copy replaces it); the backup goes only while it is the recorded copy; read-only checked ------
                    string fault;
                    SetFaultHook([&](const char* s) { return (";" + fault + ";").find(";" + string(s) + ";") != string::npos; });  // (';'-separated names)
                    const wstring bak = RrConfigBackupPath(d);
                    auto leftovers = [&] { return ListFiles(Parent(ini), L"Engine.ini.wronggpu-*").size(); };
                    auto fileId = [](const wstring& p) {
                        HANDLE h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
                        if (h == INVALID_HANDLE_VALUE) return 0ull;
                        BY_HANDLE_FILE_INFORMATION i{};
                        GetFileInformationByHandle(h, &i);
                        CloseHandle(h);
                        return ((unsigned long long)i.nFileIndexHigh << 32) | i.nFileIndexLow;
                    };
                    auto hidden = [](const wstring& p) { const DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_HIDDEN); };
                    const string theirs = "[Core.Log]\r\nLogNet=Verbose\r\n";  // the player changed another section since the fix
                    {  // 4: policy: from a file the player changed elsewhere only WrongGPU's line goes; by replacement (a new file), attributes kept
                        Record m = fixFor(false);
                        m.rrconfig.backupSha = H(own);
                        file(ini, theirs + "\r\n" + RrConfigNewFile());
                        SetFileAttributesW(ini.c_str(), FILE_ATTRIBUTE_HIDDEN);
                        const auto idBefore = fileId(ini);
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(read(ini) == theirs && hidden(ini) && idBefore != 0 && fileId(ini) != idBefore && w.empty() && leftovers() == 0 && !IsFile(bak) &&
                                    !m.rrconfig.present,
                                "Engine.ini (#209 review): from a file the player changed elsewhere only WrongGPU's line goes (policy); a corrected copy replaces the file "
                                "(its attributes kept, nothing left beside it); the backup with the recorded hash goes");
                        clear(ini);
                    }
                    {  // 4: a write that stops part way leaves Engine.ini byte for byte as it was
                        Record m = fixFor(false);
                        file(ini, edited);
                        fault = "rrconfig-write";
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        fault.clear();
                        require(read(ini) == edited && leftovers() == 0 && w.size() == 1 && w[0].find(L"yourself") != wstring::npos && read(bak) == own,
                                "Engine.ini (#209 review): a write fault part way leaves Engine.ini byte for byte as it was (the new bytes go to a new file first, which "
                                "goes again); the result says to take the line out by hand; the backup stays");
                        clear(ini);
                        clear(bak);
                    }
                    {  // 4: another program changes Engine.ini between the close and the replace: its version is kept beside it (never written back), said
                        Record m = fixFor(false);
                        file(ini, edited);
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "rrconfig-replace") file(ini, Y); });
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        const wstring theirs = ini + L".wronggpu-their-change";
                        require(read(ini) == own && read(theirs) == Y && leftovers() == 1 && w.size() == 1 && w[0].find(L"changed while WrongGPU") != wstring::npos &&
                                    w[0].find(theirs) != wstring::npos && w[0].find(bak) != wstring::npos && read(bak) == own,
                                "Engine.ini (#209 review): a change by another program in the moment before the replace is not lost: it is kept beside the corrected "
                                "file (never written into it), the result names both, the backup stays");
                        clear(ini);
                        clear(theirs);
                        clear(bak);
                    }
                    {  // 4: the file WrongGPU created, changed since: it stays without WrongGPU's line (deleted only while it is WrongGPU's bytes)
                        for (const string& extra : {string("r.PlayersOwnSetting=1\r\n"), string("\r\n")}) {
                            Record m = fixFor(true);
                            const string grown = RrConfigNewFile() + extra, want = RrConfigRemoveText(grown, m.rrconfig);
                            file(ini, grown);
                            std::vector<wstring> notes, w;
                            RevertRrConfig(mc, d, m, notes, w);
                            require(want != grown && IsFile(ini) && read(ini) == want && want.find(ToUtf8(RrConfigLineW())) == string::npos && w.empty(),
                                    "Engine.ini (#209 review): a file WrongGPU created and something changed since stays, without WrongGPU's line");
                            clear(ini);
                        }
                    }
                    {  // 5: the backup goes only while it is the copy WrongGPU made (the recorded hash); changed: kept and said
                        Record m = fixFor(false);
                        m.rrconfig.backupSha = H(own);
                        const string other = "a changed backup (synthetic)\r\n";
                        file(bak, other);
                        file(ini, edited);
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(read(ini) == own && read(bak) == other && w.size() == 1 && w[0].find(bak) != wstring::npos && w[0].find(L"changed since") != wstring::npos,
                                "Engine.ini (#209 review): a backup with other bytes than the recorded hash is not deleted; it stays and the result names it");
                        clear(ini);
                        clear(bak);
                        const wstring g = ownGame(L"Record backup hash", {});
                        Record r;
                        LoadRecord(RecordPath(g), r);
                        r.rrconfig = fixFor(false).rrconfig;
                        r.rrconfig.backupSha = H(own);
                        SaveRecord(RecordPath(g), r);
                        Record back;
                        require(LoadRecord(RecordPath(g), back) && back.rrconfig.present && back.rrconfig.backup && IEq(back.rrconfig.backupSha, H(own)) &&
                                    read(RecordPath(g)).find("\nrrconfig-backup\t") != string::npos && back.extra.empty(),
                                "record (#209 review): the backup's hash is kept in a line of its own (rrconfig-backup) and read back");
                        clear(bak);
                    }
                    {  // 6: the read-only attribute not put back: said, and the backup stays
                        Record m = fixFor(false);
                        m.rrconfig.wasReadOnly = true;
                        file(ini, edited);
                        SetFileAttributesW(ini.c_str(), FILE_ATTRIBUTE_READONLY);
                        fault = "rrconfig-readonly";
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        fault.clear();
                        require(read(ini) == own && w.size() == 1 && w[0].find(L"read-only") != wstring::npos && read(bak) == own,
                                "Engine.ini (#209 review): a read-only attribute that cannot be put back is said, and the backup stays");
                        clear(ini);
                        clear(bak);
                    }
                    // ---- #209 re-check (REVIEW-209b) -------------------------------------------------------------------------
                    const wstring copy = ini + L".wronggpu-new", old = ini + L".wronggpu-old";
                    const string F = "[SystemSettings]\r\n; bytes another program put in place of the corrected copy (synthetic)\r\n",
                                 Z = "[Core.Log]\r\nLogNet=Warning\r\n; a second change, made after the replace (synthetic)\r\n";
                    {  // a: the corrected copy swapped after it was verified and closed: the live file is not trusted, left as it is, said
                        Record m = fixFor(false);
                        m.rrconfig.backupSha = H(own);
                        file(ini, edited);
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "rrconfig-replace") file(copy, F); });
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(read(ini) == F && read(old) == edited && read(bak) == own && w.size() == 1 && w[0].find(L"not the corrected file WrongGPU put there") != wstring::npos &&
                                    w[0].find(old) != wstring::npos && !IsFile(copy) && notes.empty(),
                                "Engine.ini (REVIEW-209b a): a corrected copy swapped before the replace is caught after it (the live file held and hashed): left as "
                                "it is, the file it replaced kept beside it, the backup kept, the result says so");
                        clear(ini);
                        clear(old);
                        clear(bak);
                    }
                    {  // b: a change before the replace and a second one after it: the second is never overwritten; the first kept beside it
                        Record m = fixFor(false);
                        file(ini, edited);
                        SetRestoreStepHook([&](const char* s) {
                            if (string(s) == "rrconfig-replace") file(ini, Y);
                            if (string(s) == "rrconfig-replaced") file(ini, Z);
                        });
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(read(ini) == Z && read(old) == Y && read(bak) == own && w.size() == 1 && w[0].find(L"not the corrected file WrongGPU put there") != wstring::npos,
                                "Engine.ini (REVIEW-209b b): a second change after the replace is never overwritten by the recovery; the first change is kept beside "
                                "it, the backup kept, the result says so");
                        clear(ini);
                        clear(old);
                        clear(bak);
                    }
                    {  // c: the player deleted their own settings and kept only WrongGPU's: the revert leaves an empty file, never the old backup
                        Record m = fixFor(false);
                        m.rrconfig.backupSha = H(own);
                        file(ini, RrConfigNewFile());
                        const string want = RrConfigRemoveText(RrConfigNewFile(), m.rrconfig);
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        require(want.find_first_not_of(" \t\r\n") == string::npos && IsFile(ini) && read(ini) == want && w.empty() && leftovers() == 0,
                                "Engine.ini (REVIEW-209b c): taking out WrongGPU's line from a file the player emptied of their own settings leaves it empty; the "
                                "backup from before the fix is not put back");
                        clear(ini);
                        clear(bak);
                    }
                    {  // d: the replace fails: the read-only bit goes back, and a failure to put it back is said
                        for (int roFails = 0; roFails < 2; ++roFails) {
                            Record m = fixFor(false);
                            m.rrconfig.wasReadOnly = true;
                            file(ini, edited);
                            SetFileAttributesW(ini.c_str(), FILE_ATTRIBUTE_READONLY);
                            fault = roFails ? "rrconfig-replacefile;rrconfig-readonly" : "rrconfig-replacefile";
                            std::vector<wstring> notes, w;
                            RevertRrConfig(mc, d, m, notes, w);
                            fault.clear();
                            const bool saidRo = w.size() == 1 && w[0].find(L"read-only attribute could not be set again") != wstring::npos;
                            require(read(ini) == edited && read(bak) == own && leftovers() == 0 && w.size() == 1 && w[0].find(L"Cannot replace") != wstring::npos &&
                                        (roFails ? saidRo && !readOnly(ini) : !saidRo && readOnly(ini)),
                                    roFails ? "Engine.ini (REVIEW-209b d): a failed replace whose read-only restore fails says so in the warning; the file and the backup stay"
                                            : "Engine.ini (REVIEW-209b d): a failed replace puts the read-only bit back; the file and the backup stay");
                            clear(ini);
                            clear(bak);
                        }
                    }
                    // ---- #209 re-check 2 (REVIEW-209c) -----------------------------------------------------------------------
                    {  // b: a change before the replace, then a DISTINCT file with IDENTICAL bytes swapped in after it: never written, left as it is
                        Record m = fixFor(false);
                        file(ini, edited);
                        unsigned long long swappedId = 0;
                        SetRestoreStepHook([&](const char* s) {
                            if (string(s) == "rrconfig-replace") file(ini, Y);
                            if (string(s) == "rrconfig-replaced") { file(ini, own); swappedId = fileId(ini); }  // own: the corrected bytes, another file
                        });
                        std::vector<wstring> notes, w;
                        RevertRrConfig(mc, d, m, notes, w);
                        SetRestoreStepHook(nullptr);
                        require(swappedId != 0 && fileId(ini) == swappedId && read(ini) == own && read(old) == Y && read(bak) == own && w.size() == 1 &&
                                    w[0].find(L"not the corrected file WrongGPU put there") != wstring::npos && w[0].find(old) != wstring::npos &&
                                    w[0].find(L"read-only") == wstring::npos && notes.empty(),
                                "Engine.ini (REVIEW-209c b): a distinct file with the corrected bytes swapped in after the replace is not taken for WrongGPU's (its "
                                "identity differs): left untouched, the earlier change kept beside it, the backup kept, the result says so");
                        clear(ini);
                        clear(old);
                        clear(bak);
                    }
                    {  // P2: the verification paths never touch another program's read-only file, and a read-only restore failure there is said
                        for (int roFails = 0; roFails < 2; ++roFails) {
                            Record m = fixFor(false);
                            file(ini, edited);
                            SetRestoreStepHook([&](const char* s) {
                                if (string(s) != "rrconfig-replaced") return;
                                file(ini, Z);
                                SetFileAttributesW(ini.c_str(), FILE_ATTRIBUTE_READONLY);
                            });
                            fault = roFails ? "rrconfig-verify-readonly" : "";
                            std::vector<wstring> notes, w;
                            RevertRrConfig(mc, d, m, notes, w);
                            fault.clear();
                            SetRestoreStepHook(nullptr);
                            const bool saidRo = w.size() == 1 && w[0].find(L"read-only attribute could not be put back") != wstring::npos;
                            require(read(ini) == Z && readOnly(ini) && read(old) == edited && read(bak) == own && w.size() == 1 &&
                                        w[0].find(L"not the corrected file WrongGPU put there") != wstring::npos && (roFails ? saidRo : !saidRo),
                                    roFails ? "Engine.ini (REVIEW-209c P2): on the verification's mismatch path a read-only restore failure is said in the warning; the "
                                              "file and both backups stay"
                                            : "Engine.ini (REVIEW-209c P2): the verification opens another program's read-only file without changing its attribute; "
                                              "the file and both backups stay");
                            clear(ini);
                            clear(old);
                            clear(bak);
                        }
                    }
                    SetFaultHook(nullptr);
                }
                SetRestoreStepHook(nullptr);
                // ---- PinnedFile, DeleteIfSha and CopyNew (util.h): the handle-based primitives under the restore ----------------
                {
                    const wstring base = Join(dir, L"Pinned"), p = Join(base, L"a.bin");
                    DWORD err = 0;
                    file(p, W);
                    {
                        PinnedFile f, second;
                        require(f.Open(p, false, &err) && f.IsOpen() && f.Sha256() == H(W), "pinned: a plain file is opened and hashed through the handle");
                        require(!second.Open(p, false, &err) && err == ERROR_SHARING_VIOLATION, "pinned: while held, another open for delete (a replace, a rename) is refused");
                        require(!DeleteFileW(p.c_str()) && !MoveFileExW(p.c_str(), (p + L".x").c_str(), 0) && read(p) == W, "pinned: while held, it cannot be deleted or renamed by its path; it can still be read");
                        require(!DeleteFileW(Join(base, L"missing").c_str()) && !f.Open(Join(base, L"missing"), false, &err) && err == ERROR_FILE_NOT_FOUND,
                                "pinned: a path with nothing there: ERROR_FILE_NOT_FOUND");
                    }
                    require(DeleteIfSha(p, {H(C)}, &err) == DeleteIf::Changed && read(p) == W, "pinned: DeleteIfSha with other bytes: Changed, the file stays");
                    {
                        PinnedFile held;
                        held.Open(p, false, &err);
                        require(DeleteIfSha(p, {H(W)}, &err) == DeleteIf::Failed && read(p) == W, "pinned: DeleteIfSha on a file another program holds: Failed, the file stays");
                    }
                    require(DeleteIfSha(p, {H(C), H(W)}, &err) == DeleteIf::Deleted && !IsFile(p) && DeleteIfSha(p, {H(W)}, &err) == DeleteIf::Gone,
                            "pinned: DeleteIfSha with one of the hashes: Deleted; then nothing there: Gone");
                    file(p, W);
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
                    require(DeleteIfSha(p, {H(W)}, &err) == DeleteIf::Deleted && !IsFile(p), "pinned: a read-only file is deleted through the handle");
                    // Overwrite: the bytes of src (hashed first), its attributes; a src that is not the expected bytes touches nothing
                    const wstring src = Join(base, L"src.bin");
                    file(src, "short original");
                    file(p, W);
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
                    {
                        PinnedFile f;
                        require(f.Open(p, true, &err) && !f.Overwrite(src, H(C), &err) && err == ERROR_FILE_CORRUPT && f.Sha256() == H(W),
                                "pinned: Overwrite from a file that is not the expected bytes fails and changes nothing");
                        require(f.Overwrite(src, H("short original"), &err) && f.Sha256() == H("short original"), "pinned: Overwrite puts the source's bytes through the handle (a read-only target too)");
                    }
                    require(read(p) == "short original" && !(GetFileAttributesW(p.c_str()) & FILE_ATTRIBUTE_READONLY), "pinned: ...the file is the source's bytes, shorter than before, with the source's attributes");
                    // a link is not a plain file (when the account may make one)
                    const wstring link = Join(base, L"link.bin");
                    if (CreateSymbolicLinkW(link.c_str(), p.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
                        PinnedFile f;
                        require(!f.Open(link, false, &err) && err == ERROR_NOT_SUPPORTED && DeleteIfSha(link, {H("short original")}, &err) == DeleteIf::Changed && read(p) == "short original",
                                "pinned: a symbolic link is refused (not followed): its target is untouched");
                        DeleteFileW(link.c_str());
                    }
                    // CopyNew creates only where nothing is
                    const wstring dst = Join(base, L"dst.bin");
                    require(CopyNew(src, dst, &err) && read(dst) == "short original", "CopyNew: copies where nothing is");
                    file(dst, "another file");
                    require(!CopyNew(src, dst, &err) && err == ERROR_FILE_EXISTS && read(dst) == "another file", "CopyNew: a file that is there is not overwritten (ERROR_FILE_EXISTS)");
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
                }
                // ---- the state folder, the shared data folder, hard links, the mirror, read-only files, the settings migration -----
                // (1) what WrongGPU's own folders hold: only what it wrote is deleted; a file a player saved there stays, with the folder
                {
                    auto said = [](const std::vector<wstring>& ws, const wchar_t* what) {
                        for (auto& x : ws) if (x.rfind(kLeftUnrecorded, 0) == 0 && x.find(what) != wstring::npos) return true;
                        return false;
                    };
                    const wstring d = ownGame(L"State foreign", {{L"wronggpu_runtime.dll", L"copy", "", W, W}, {L"wronggpu.ini", L"copy", A, W, W}});
                    const wstring st = StateDir(d);
                    file(Join(st, L"uninstall.exe"), "the restorer");
                    Record recd;
                    LoadRecord(RecordPath(d), recd);
                    file(Join(st, L"owned-files.json"), OwnedFilesJson(recd));  // the bytes WrongGPU makes of the record
                    file(Join(st, L"setup.log"), "log");
                    file(Join(st, L"profile.ini"), "[game]");
                    file(Join(st, L"previous\\previous.txt"), "version=0.5.1\r\nfile=" + ToUtf8(H(P)) + "|wronggpu_runtime.dll\r\nfile=" + ToUtf8(H(P)) + "|changed-by-the-player.dll\r\n");
                    file(Join(st, L"previous\\wronggpu_runtime.dll"), P);
                    file(Join(st, L"previous\\changed-by-the-player.dll"), "a player's file at a listed path of the previous version, other bytes than listed");
                    file(Join(st, L"notes.txt"), "the player's notes, saved after the install");
                    file(Join(st, L"backup\\not-ours.bin"), "a file in the backup folder that no record lists");
                    file(Join(st, L"previous\\kept-by-the-player.dll"), "in the previous version's folder, not listed in previous.txt");
                    file(Join(st, L"rr-config\\Engine.ini"), "the player's Engine.ini, kept by the fix");
                    std::vector<wstring> w;
                    require(restore(d, false, &w) && !IsFile(RecordPath(d)) && !IsFile(Join(d, L"wronggpu_runtime.dll")) && read(Join(d, L"wronggpu.ini")) == A,
                            "state folder: an uninstall with foreign files in the state folder runs to its end (the game as before)");
                    require(!IsFile(Join(st, L"uninstall.exe")) && !IsFile(Join(st, L"owned-files.json")) && !IsFile(Join(st, L"setup.log")) && !IsFile(Join(st, L"profile.ini")) &&
                                !IsFile(Join(st, L"backup\\wronggpu.ini")) && !IsFile(Join(st, L"previous\\previous.txt")) && !IsFile(Join(st, L"previous\\wronggpu_runtime.dll")),
                            "state folder: what WrongGPU wrote goes (its fixed names, the recorded backups, the previous version's listed files)");
                    require(read(Join(st, L"notes.txt")) == "the player's notes, saved after the install" && IsFile(Join(st, L"backup\\not-ours.bin")) &&
                                IsFile(Join(st, L"previous\\kept-by-the-player.dll")) && IsFile(Join(st, L"rr-config\\Engine.ini")) && IsDir(st) &&
                                IsFile(Join(st, L"previous\\changed-by-the-player.dll")),
                            "state folder: a file saved there after the install, an unlisted file in backup\\ and previous\\, a listed previous-version file whose bytes changed, and the player's Engine.ini copy stay, and so does the folder");
                    require(said(w, L"changed-by-the-player.dll"), "state folder: the changed previous-version file is listed in the result");
                    // owned-files.json that is not what WrongGPU makes of the record (the player's own content) stays too
                    const wstring d2 = ownGame(L"State owned-files changed", {{L"wronggpu_runtime.dll", L"copy", "", W, W}});
                    file(Join(StateDir(d2), L"owned-files.json"), "{\"the player's own file\": 1}");
                    std::vector<wstring> w2;
                    require(restore(d2, false, &w2) && read(Join(StateDir(d2), L"owned-files.json")) == "{\"the player's own file\": 1}" && said(w2, L"owned-files.json"),
                            "state folder: owned-files.json whose bytes are not WrongGPU's stays and is listed");
                    require(said(w, L"wronggpu-setup") && said(w, L"notes.txt") && !AnyPartly(w), "state folder: the result lists what was left");
                    // the same without a record (relocation: the restore has removed the backup): RemoveStateFolder
                    const wstring o = Join(dir, L"Ownership\\Relocated");
                    file(Join(StateDir(o), L"uninstall.exe"), "the restorer");
                    Record relocated;
                    relocated.version = L"0.6.0"; relocated.game = o; relocated.exe = Join(o, L"Game.exe");
                    file(Join(StateDir(o), L"owned-files.json"), OwnedFilesJson(relocated));
                    file(Join(StateDir(o), L"keep-me.txt"), "a player's file");
                    const auto stay = RemoveStateFolder(o, &relocated);
                    require(stay.size() == 1 && IEq(stay[0], L"keep-me.txt") && IsFile(Join(StateDir(o), L"keep-me.txt")) && !IsFile(Join(StateDir(o), L"uninstall.exe")) && !IsFile(Join(StateDir(o), L"owned-files.json")),
                            "state folder: relocation's removal deletes WrongGPU's files and returns what stays");
                    DeleteFileW(Join(StateDir(o), L"keep-me.txt").c_str());
                    require(RemoveStateFolder(o, &relocated).empty() && !IsDir(StateDir(o)), "state folder: with nothing foreign in it the folder itself goes");
                    // the shared data folder is WrongGPU's own application folder: the last game's cleanup removes it whole, whatever
                    // is in it (README "Uninstall"), by moving it aside at once; the hidden cleanup then deletes the moved folder
                    const wstring ad = Join(dir, L"Ownership\\shared-data\\WrongGPU");
                    file(Join(ad, L"model\\model.wgcm"), "model");
                    file(Join(ad, L"model\\readme-from-player.txt"), "a file in the model folder that is not WrongGPU's");
                    file(Join(ad, L"launcher.ini"), "[launcher]");
                    file(Join(ad, L"logs\\notes.txt"), "a file in the log folder");
                    file(Join(ad, L"photos.txt"), "a file in WrongGPU's own folder");
                    Context cc; cc.logFile = Join(dir, L"Ownership\\shared-cleanup.log");
                    const wstring kept = StartCleanup(cc, L"", L"", ad);
                    require(kept.empty() && !IsDir(ad) && !IsFile(Join(ad, L"photos.txt")),
                            "shared data: the last game's cleanup removes %LOCALAPPDATA%\\WrongGPU whole, with everything in it");
                }
                // (2) the plugin's settings file is rewritten only when it is demonstrably WrongGPU's (recorded, bytes match)
                {
                    const wstring d = Join(dir, L"Ownership\\Asi selection"), p = Join(d, L"wronggpu-asi.ini");
                    const string k = "; the plugin\r\nselection=wronggpu-k\r\nother=1\r\n", m = "; the plugin\r\nselection=wronggpu\r\nother=1\r\n";
                    file(p, k);
                    require(MigrateAsiSelection(d, L"") == AsiMigration::Left && read(p) == k, "asi selection: not recorded: left as it is");
                    require(MigrateAsiSelection(d, H("other bytes")) == AsiMigration::Left && read(p) == k, "asi selection: recorded with other bytes (the player changed it): left as it is");
                    require(MigrateAsiSelection(d, H(k)) == AsiMigration::Changed && read(p) == m, "asi selection: recorded and the bytes match: that one line is changed, the others stay");
                    require(MigrateAsiSelection(d, H(m)) == AsiMigration::None && read(p) == m, "asi selection: nothing selects Preset K any more: nothing to change");
                    DeleteFileW(p.c_str());
                    require(MigrateAsiSelection(d, H(k)) == AsiMigration::None, "asi selection: no file: nothing to change");
                    file(p, k);
                    HANDLE other = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);  // another program holds it
                    const AsiMigration held = MigrateAsiSelection(d, H(k));
                    if (other != INVALID_HANDLE_VALUE) CloseHandle(other);
                    require(other != INVALID_HANDLE_VALUE && held == AsiMigration::Left && read(p) == k, "asi selection: held by another program: left as it is (the write is conditional on the held file)");
                }
                // (3) hard links: a file with another name is never written through; its original is kept
                {
                    const wstring d = ownGame(L"Hard link", {{L"wronggpu.ini", L"copy", A, W, W}, {L"wronggpu_runtime.dll", L"copy", "", W, W}});
                    const wstring ini = Join(d, L"wronggpu.ini"), dll = Join(d, L"wronggpu_runtime.dll"), linkIni = Join(d, L"player-copy.ini"), linkDll = Join(d, L"player-copy.dll");
                    const bool linked = CreateHardLinkW(linkIni.c_str(), ini.c_str(), nullptr) && CreateHardLinkW(linkDll.c_str(), dll.c_str(), nullptr);
                    require(linked, "hard link: this folder allows hard links (NTFS)");
                    PinnedFile pf;
                    DWORD err = 0;
                    require(!pf.Open(ini, true, &err) && err == ERROR_TOO_MANY_LINKS && read(ini) == W, "hard link: a file with two names is refused for writing (ERROR_TOO_MANY_LINKS)");
                    require(pf.Open(ini, false, &err) && pf.Sha256() == H(W), "hard link: ...and opened for reading or deleting");
                    pf.Close();
                    std::vector<wstring> w;
                    require(restore(d, false, &w) && read(ini) == W && read(linkIni) == W && keptOf(d, L"wronggpu.ini") == std::vector<string>{A} && partly(w, L"wronggpu.ini"),
                            "hard link: the original is not written through the link (the file and its other name keep WrongGPU's bytes), the original is kept, the result says partly");
                    bool said = false;
                    for (auto& x : w) said = said || (x.find(L"hard link") != wstring::npos && x.find(L"wronggpu.ini") != wstring::npos);
                    require(said, "hard link: the warning says why");
                    require(!IsFile(dll) && read(linkDll) == W, "hard link: a recorded file with nothing to put back is unlinked (its name goes), the other name keeps the bytes");
                }
                // (4) the mirror: a copy goes only when the record lists it
                {
                    const wstring d = ownGame(L"Mirror unrecorded", {{L"wronggpu_runtime.dll", L"copy", "", W, W}});
                    file(Join(d, L"_storage_\\wronggpu_runtime.dll"), W);  // REFramework's copy, byte for byte the installed file, not in the record
                    std::vector<wstring> w;
                    bool said = false;
                    require(restore(d, false, &w) && !IsFile(Join(d, L"wronggpu_runtime.dll")) && read(Join(d, L"_storage_\\wronggpu_runtime.dll")) == W, "mirror: an unrecorded copy that matches the installed bytes is not deleted");
                    for (auto& x : w) said = said || (x.rfind(kLeftUnrecorded, 0) == 0 && x.find(L"wronggpu_runtime.dll") != wstring::npos);
                    require(said, "mirror: the result says it was left");
                    const wstring r = ownGame(L"Mirror recorded", {{L"wronggpu_runtime.dll", L"copy", "", W, W}, {L"_storage_\\wronggpu_runtime.dll", L"copy", "", W, W}});
                    require(restore(r, false) && !IsFile(Join(r, L"wronggpu_runtime.dll")) && !IsFile(Join(r, L"_storage_\\wronggpu_runtime.dll")), "mirror: a recorded copy goes with the install");
                }
                // (5) read-only files: the bit is changed only on a file that is written, and put back otherwise
                {
                    const wstring base = Join(dir, L"Pinned"), p = Join(base, L"ro.bin");
                    DWORD err = 0;
                    auto readOnly = [](const wstring& f) { return (GetFileAttributesW(f.c_str()) & FILE_ATTRIBUTE_READONLY) != 0; };
                    file(p, W);
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
                    {
                        HANDLE other = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                        PinnedFile f;
                        const bool ok = f.Open(p, true, &err);
                        if (other != INVALID_HANDLE_VALUE) CloseHandle(other);
                        require(other != INVALID_HANDLE_VALUE && !ok && err == ERROR_SHARING_VIOLATION && readOnly(p),
                                "read-only: a write open that fails (another program holds the file) leaves the read-only bit as it was");
                    }
                    {
                        PinnedFile f;
                        require(f.Open(p, true, &err) && !readOnly(p) && f.Sha256() == H(W), "read-only: opened for writing, the file is writable while held");
                    }
                    require(readOnly(p), "read-only: closed without writing, the read-only bit is back");
                    {
                        const wstring d = ownGame(L"Read-only changed", {{L"wronggpu.ini", L"copy", A, W, W}});
                        const wstring f = Join(d, L"wronggpu.ini");
                        SetRestoreStepHook([&](const char* s) { if (string(s) == "kept") { file(f, C); SetFileAttributesW(f.c_str(), FILE_ATTRIBUTE_READONLY); } });
                        std::vector<wstring> w;
                        const bool ok = restore(d, false, &w);
                        const bool ro = readOnly(f);
                        SetFileAttributesW(f.c_str(), FILE_ATTRIBUTE_NORMAL);
                        SetRestoreStepHook(nullptr);
                        require(ok && read(f) == C && ro && keptOf(d, L"wronggpu.ini") == std::vector<string>{A},
                                "read-only: a changed read-only file the restore leaves keeps its read-only bit (and its bytes); the original is kept");
                    }
                    const wstring target = Join(base, L"ro-target.bin"), link = Join(base, L"ro-link.bin");
                    file(target, W);
                    SetFileAttributesW(target.c_str(), FILE_ATTRIBUTE_READONLY);
                    if (CreateSymbolicLinkW(link.c_str(), target.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
                        PinnedFile f;
                        require(!f.Open(link, true, &err) && err == ERROR_NOT_SUPPORTED && readOnly(target), "read-only: a link is refused before any attribute is touched (the target stays read-only)");
                        DeleteFileW(link.c_str());
                    }
                    SetFileAttributesW(target.c_str(), FILE_ATTRIBUTE_NORMAL);
                    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
                }
            }
            // ---- 0.6.1: an update that now leaves RR out (the game no longer uses it, or another setup's
            // plugin keeps it out) first runs the earlier install's restore with keepState (DoInstall step 2): the earlier RR plugin
            // goes with it when it is the file installed; one changed since is left in place with a warning, never removed ----
            {
                const string rrAsi = "WrongGPU-RR.asi as installed (synthetic)", rrIni = "rr_mode=accurate\r\n";
                auto earlier = [&](const wstring& name) {
                    const wstring d = Join(dir, L"RR Update\\" + name);
                    file(Join(d, L"Game.exe"), "MZ");
                    file(Join(d, L"plugins\\WrongGPU-RR.asi"), rrAsi);
                    file(Join(d, L"wronggpu-rr.ini"), rrIni);
                    Record m; m.version = L"0.6.0"; m.format = 2; m.game = d; m.exe = Join(d, L"Game.exe"); m.status = L"experimental"; m.model = L"m.wgcm";
                    Entry dirE; dirE.action = L"dir"; dirE.rel = L"plugins"; m.e.push_back(dirE);
                    Entry a; a.action = L"copy"; a.rel = L"plugins\\WrongGPU-RR.asi"; a.inst = Sha256Bytes(rrAsi.data(), rrAsi.size()); a.owner = L"wg"; m.e.push_back(a);
                    Entry i; i.action = L"copy"; i.rel = L"wronggpu-rr.ini"; i.inst = Sha256Bytes(rrIni.data(), rrIni.size()); i.owner = L"wg"; m.e.push_back(i);
                    SaveRecord(RecordPath(d), m);
                    return d;
                };
                const wstring d = earlier(L"Unchanged");
                bool threw = false;
                try { DoRestore(mc, d, quiet, false, true); } catch (const SetupError&) { threw = true; }
                require(!threw && !IsFile(Join(d, L"plugins\\WrongGPU-RR.asi")) && !IsFile(Join(d, L"wronggpu-rr.ini")) && IsFile(Join(d, L"Game.exe")),
                        "RR update: an update's restore removes the earlier install's RR plugin and its settings (the new install then places none)");
                const wstring c = earlier(L"Changed");
                file(Join(c, L"plugins\\WrongGPU-RR.asi"), "another build put here since (synthetic)");
                std::vector<wstring> w;
                threw = false;
                try { w = DoRestore(mc, c, quiet, false, true); } catch (const SetupError&) { threw = true; }
                bool warned = false;
                for (auto& x : w) warned = warned || x == L"Left in place (changed since install): plugins\\WrongGPU-RR.asi";
                require(!threw && IsFile(Join(c, L"plugins\\WrongGPU-RR.asi")) && warned, "RR update: an RR plugin changed since the install: left in place, with a warning");
            }
            // the scan takes managed folders whose record is gone off the list, unless their drive isn't connected
            wstring offline;
            const DWORD drives = GetLogicalDrives();
            for (wchar_t c = L'Z'; c >= L'D' && offline.empty(); --c) if (!(drives & (1u << (c - L'A'))) ) offline = wstring(1, c) + L":\\Games\\Offline Game";
            AddManagedGame(oldDir);
            if (!offline.empty()) AddManagedGame(offline);
            GlobalSettings sgs;
            ScanGames(mc, sgs, true);
            require(!managed(oldDir) && (offline.empty() || managed(offline)), "moved game: the scan drops a managed folder without its record (not one on a drive that is not connected)");

            // ---- rc6 L2: an install next to a launcher stub (before rc5) is in the wrong place; a stub alone is its game ----
            const wstring cp = Join(dir, L"Stub Lib\\Cyberpunk 2077"), stub = Join(cp, L"REDprelauncher.exe"), real = Join(cp, L"bin\\x64\\Cyberpunk2077.exe");
            file(stub, "MZ stub");
            file(real, "MZ game");
            file(Join(cp, L"bin\\x64\\amd_fidelityfx_dx12.dll"), "MZ");
            const wstring hl = Join(dir, L"Stub Lib\\Hogwarts Legacy"), starter = Join(hl, L"HogwartsLegacy.exe"), hlGame = Join(hl, L"Phoenix\\Binaries\\Win64\\HogwartsLegacy.exe");
            file(starter, "MZ starter");
            file(hlGame, "MZ game");
            require(IEq(MisplacedFor(stub), real) && IEq(MisplacedFor(starter), hlGame), "MisplacedFor: a launcher stub's game, a root starter's game");
            require(MisplacedFor(real).empty() && MisplacedFor(hlGame).empty() && MisplacedFor(Join(newDir, L"Game.exe")).empty(), "MisplacedFor: none for the game itself");
            const GameEntry alone = EntryFor(mc, stub, L"", L"Added");
            require(IEq(alone.exe, real) && !alone.hasRecord && alone.realExe.empty(), "a stub without an install next to it is read as its game");
            Record sr; sr.version = L"0.6.0"; sr.game = cp; sr.exe = stub; sr.status = L"validated";
            Entry se; se.action = L"copy"; se.rel = L"wronggpu_runtime.dll"; se.inst = Sha256Bytes(rt.data(), rt.size()); se.owner = L"wg"; sr.e.push_back(se);
            file(Join(cp, L"wronggpu_runtime.dll"), rt);
            file(Join(cp, L"wronggpu.ini"), "[fsr]\r\n");
            SaveRecord(RecordPath(cp), sr);
            AddManagedGame(cp);
            const GameEntry wrong = EntryFor(mc, stub, L"", L"Managed");
            require(wrong.hasRecord && IEq(wrong.exe, stub) && IEq(wrong.realExe, real) && wrong.status == GameStatus::NeedsRepair && !wrong.problems.empty() &&
                        IsMisplacedProblem(wrong.problems[0]) && wrong.problems[0].find(L"bin\\x64\\Cyberpunk2077.exe") != wstring::npos,
                    "an install next to the stub: Needs repair, in the wrong place, the game's program named");
            sgs.addedGames = {real};
            const auto listed = ScanGames(mc, sgs, true);
            size_t cpEntries = 0;
            for (auto& g : listed) cpEntries += IEq(g.exe, stub) || IEq(g.exe, real);
            require(cpEntries == 1, "the scan lists the game once: the wrong-place install's entry, not the game a second time");
            RemoveTree(cp); RemoveManagedGame(cp);

            // ---- rc6 L4: what a failed install or update leaves in the state folder: the player's settings only ----------------
            const wstring fs = Join(dir, L"failed-state");
            require(!StateHasSettings(fs), "L4: no state folder, no settings");
            file(Join(StateDir(fs), L"uninstall.exe"), "MZ");
            file(Join(StateDir(fs), L"owned-files.json"), "{}");
            file(Join(StateDir(fs), L"backup\\x.dll"), "x");
            require(!StateHasSettings(fs), "L4: an uninstaller and a backup are no settings");
            RemoveStateKeepingSettings(fs);
            require(!IsDir(StateDir(fs)), "L4: without settings the state folder goes");
            file(Join(StateDir(fs), L"uninstall.exe"), "MZ");
            file(ProfilePath(fs), "[profile]\r\n");
            file(Join(StateDir(fs), L"previous\\previous.txt"), "version=0.5.9\r\n");
            file(Join(StateDir(fs), L"rr-config\\Engine.ini"), "[x]\r\n");
            file(Join(StateDir(fs), L"backup\\x.dll"), "x");
            require(StateHasSettings(fs), "L4: profile.ini, previous\\ and rr-config\\ are settings");
            RemoveStateKeepingSettings(fs);
            require(IsFile(ProfilePath(fs)) && IsFile(Join(StateDir(fs), L"previous\\previous.txt")) && IsFile(Join(StateDir(fs), L"rr-config\\Engine.ini")) &&
                        !IsFile(Join(StateDir(fs), L"uninstall.exe")) && !IsDir(Join(StateDir(fs), L"backup")),
                    "L4: the settings stay, the rest goes");
            SetEnvironmentVariableW(L"LOCALAPPDATA", oldLad.c_str());
        }

        // ---- log tail -----------------------------------------------------------------------------------------------------
        const string log =
            "[10:00:00.000 100] [amd_fidelityfx_dx12] ==== session start: Old.exe (process 100) ====\r\n"
            "[10:00:00.100 100] [amd_fidelityfx_dx12] settings: upscaler=amd runtime_dll=x\r\n"
            "[11:00:00.000 200] [amd_fidelityfx_dx12] ==== session start: Game.exe (process 200) ====\r\n"
            "[11:00:00.100 200] [amd_fidelityfx_dx12] settings: upscaler=network runtime_dll=wronggpu_runtime.dll fallback_to_amd=1\r\n"
            "[11:00:01.000 200] [info] executor dx12: the network runs on the game's D3D12 device\r\n"
            "[11:00:01.200 200] [info] network runtime ready: executor dx12, sync inline, max render 1280x720\r\n"
            "[11:00:02.000 200] [amd_fidelityfx_dx12] #4 OptiScaler plugin: WrongGPU selected in the host's upscaler list\r\n"
            "[11:00:03.000 200] [d3d12] own table: list 0x1 records into its own vtable 0x2 (x), which WrongGPU does not take over: reason\r\n"
            "[11:00:04.000 200] [amd_fidelityfx_dx12] [support] WG-HOOK-TABLE incident=200-1\r\n";
        LastRun lr = ParseLastRun(log);
        require(lr.found && lr.process == L"Game.exe" && lr.upscaler == L"network" && lr.ready && lr.executor == L"dx12" && lr.optiPlugin && lr.support == L"WG-HOOK-TABLE",
                "log tail: the last session's upscaler, executor, plugin and support code");
        require(lr.notes.size() == 1 && lr.notes[0].find(L"foreign table") != wstring::npos, "log tail: foreign command-list tables noted");
        require(lr.Summary().find(L"Model M (dx12)") == 0, "summary");
        LastRun none = ParseLastRun("");
        require(!none.found && none.Summary() == L"Not run yet", "no log: not run yet");
        LastRun amd = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=amd\r\n");
        require(amd.found && !amd.ready && amd.Summary().find(L"AMD FSR") == 0, "WrongGPU off: AMD FSR");
        LastRun failed = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n[2] [x] ERROR: create failed -> AMD's FSR takes over\r\n");
        require(failed.found && !failed.ready && failed.notes.size() == 1 && failed.Summary().find(L"did not start") != wstring::npos, "a failed start");
        LastRun sized = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                     "[2] [fe] [U1] WrongGPU frame: frame=0 R=1707x960 at (0,0) O=2560x1440 jitter=(0,0) motion=raw*(1,1)+(0,0)\r\n"
                                     "[3] [fe] [U1] WrongGPU frame: frame=9 R=1280x720 at (0,0) O=2560x1440 jitter=(0,0)\r\n");
        require(sized.renderSize == L"1280x720" && sized.outputSize == L"2560x1440" && sized.notes.empty(), "log tail: the last frame's render and output sizes (the window's stat tiles)");
        LastRun oldFrame = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[3] [fe] [up] WrongGPU frame: 1280x720 notes: none\r\n");
        require(oldFrame.renderSize.empty() && oldFrame.outputSize.empty(), "log tail: a frame line without R= / O= gives no sizes");

        // ---- adoption: developer / test keys never become settings (2026-10-17) ---------------------------------------------
        {
            const string dev = "[fsr]\r\nupscaler = network\r\nruntime_options=model=M;timing_log=1;output_filter=quality5;capture_dir=C:\\cap;executor=d3d12-test\r\n"
                               "log_dispatches = 1\r\nsplit_test = 0\r\nview = upscale\r\n";
            const auto ign = IgnoredImportKeys(dev);
            require(ign == std::vector<string>{"log_dispatches=1", "timing_log=1", "capture_dir=C:\\cap"}, "ignored keys: diagnostics lines off their default, non-schema runtime items (model= excepted)");
            std::vector<wstring> pn;
            const Profile ip = ImportProfile(dev, "", &pn);
            size_t ignoredNotes = 0;
            bool notImported = false;
            for (auto& note : pn) { ignoredNotes += note.rfind(kIgnoredPrefix, 0) == 0; notImported |= note.rfind(L"not imported:", 0) == 0 && note.find(L"d3d12-test") != wstring::npos; }
            require(ip.get("output_filter") == "quality5" && ip.get("executor") == "auto" && !ip.v.count("timing_log") && !ip.v.count("capture_dir") && ignoredNotes == 3 && notImported,
                    "import: the schema's keys only; every other key listed (ignored / not imported)");
            require(ProfileText(ip).find("timing_log") == string::npos && ProfileText(ip).find("capture_dir") == string::npos, "profile.ini gets no developer key");
        }

        // ---- job and result files -----------------------------------------------------------------------------------------
        OpRequest job; job.op = Op::Update; job.exe = L"C:\\Games\\X\\x.exe"; job.dir = L"C:\\Games\\X"; job.optiPackage = L"C:\\dl\\Opti.7z"; job.moveMods = true;
        job.hasProfile = true; job.profile = DefaultProfile(); SetSetting(job.profile, "executor", "hip", nullptr);
        const string jt = JobText(job, L"C:\\Users\\p\\AppData\\Local", L"C:\\log.txt");
        OpRequest back; wstring lad, lf;
        require(ParseJob(jt, back, lad, lf, &why) && back.op == Op::Update && back.exe == job.exe && back.dir == job.dir && back.moveMods && !back.removeOld &&
                back.hasProfile && back.profile.get("executor") == "hip" && lad == L"C:\\Users\\p\\AppData\\Local" && lf == L"C:\\log.txt", "job file round trip");
        {   // the player's account and whether a new install gets its own entry: carried to the worker (another account may run it)
            wstring sid; int per = 7;
            require(ParseJob(JobText(job, L"C:\\Users\\p\\AppData\\Local", L"", L"S-1-5-21-1-2-3-1001", 0), back, lad, lf, &why, &sid, &per) && sid == L"S-1-5-21-1-2-3-1001" && per == 0 &&
                        ParseJob(jt, back, lad, lf, &why, &sid, &per) && sid.empty() && per == -1, "job file: usersid and perentry round trip (absent: empty, -1)");
            OpRequest d; d.op = Op::Detach; d.dir = L"C:\\Games\\X";
            require(ParseJob(JobText(d, L"C:\\x", L""), back, lad, lf, &why) && back.op == Op::Detach, "job file: the detach operation (the launcher's uninstall, games kept)");
            OpRequest o; o.op = Op::Apply; o.dir = job.dir; o.optiKeep = o.optiOnly = true;
            require(ParseJob(JobText(o, lad, L""), back, lad, lf, &why) && back.optiKeep && back.optiOnly && !back.hasProfile &&
                        ParseJob(jt, back, lad, lf, &why) && !back.optiOnly, "job file: an OptiScaler-only apply round trip (absent: the profile is applied too)");
            // the files the player agreed to move aside travel with the hash each had when the question listed it
            OpRequest sw = job; sw.moveMods = sw.movePlugins = true;
            const wstring h1(64, L'1'), h2(64, L'2');
            sw.moveModsList = {{L"winmm.dll", h1}};
            sw.movePluginsList = {{L"plugins\\Other.asi", h2}};
            require(ParseJob(JobText(sw, lad, L""), back, lad, lf, &why) && back.moveModsList.size() == 1 && back.moveModsList[0].first == L"winmm.dll" &&
                        back.moveModsList[0].second == h1 && back.movePluginsList.size() == 1 && back.movePluginsList[0].first == L"plugins\\Other.asi" &&
                        back.movePluginsList[0].second == h2,
                    "job file: the agreed files round trip with their hashes (moved only while their bytes are those)");
            require(ParseJob("op=install\r\nexe=C:\\g\\x.exe\r\nlocalappdata=C:\\x\r\nmovemods=1\r\nmovemod.0=winmm.dll\r\n", back, lad, lf, &why) &&
                        back.moveModsList.size() == 1 && back.moveModsList[0].second.empty(),
                    "job file: an agreed file without its hash is read with none (agreed to no bytes)");
        }
        require(!ParseJob("op=install\r\nexe=relative\\x.exe\r\nlocalappdata=C:\\x\r\n", back, lad, lf, &why), "a job with a relative path is refused");
        require(!ParseJob("op=format\r\nlocalappdata=C:\\x\r\n", back, lad, lf, &why) && !ParseJob("exe=C:\\x.exe\r\n", back, lad, lf, &why), "unknown or missing operation refused");
        require(!ParseJob("op=apply\r\ndir=C:\\g\r\nlocalappdata=C:\\x\r\nprofile.executor=warp\r\n", back, lad, lf, &why), "a job with a bad setting is refused");
        OpResult res; res.code = 6; res.message = L"switch?"; res.warnings = {L"w1"}; res.notes = {L"n1", L"n2"};
        OpResult rb;
        require(ParseResult(ResultText(res), rb) && rb.code == 6 && rb.message == L"switch?" && rb.warnings.size() == 1 && rb.notes.size() == 2, "result file round trip");

        // ---- update reply -----------------------------------------------------------------------------------------------
        wstring ver, page;
        require(ParseUpdateReply("{\"tag_name\": \"v0.6.0\", \"html_url\": \"https:\\/\\/example.invalid\\/r\\/0.6.0\"}", ver, page) && ver == L"0.6.0" && page == L"https://example.invalid/r/0.6.0",
                "update reply: a release object");
        require(ParseUpdateReply("version=0.5.6\npage=https://example.invalid/x\n", ver, page) && ver == L"0.5.6", "update reply: text lines");
        require(!ParseUpdateReply("<html>", ver, page) && ParseUpdateReply("version=1.0\npage=http://plain\n", ver, page) && page.empty(), "update reply: garbage refused, non-https page dropped");
        GlobalSettings gs;
        require(!CheckForUpdate(gs, L"0.5.5").checked, "update check off by default: no request");
        gs.updateCheck = true;
        const UpdateInfo noUrl = CheckForUpdate(gs, L"0.5.5");
        require(!noUrl.checked && noUrl.message.find(L"no release address") != wstring::npos, "update check on without a URL: no request");
        require(!SetGlobal(gs, "update_url", "http://example.invalid/x", &why) && SetGlobal(gs, "update_url", "https://example.invalid/x", &why), "update URL must be https");

        // ---- game environment ---------------------------------------------------------------------------------------------
        SetEnvironmentVariableW(L"HIP_PATH", L"C:\\sdk");
        SetEnvironmentVariableW(L"ROCM_PATH", L"C:\\sdk");
        SetEnvironmentVariableW(L"WG_KEEP_ME", L"1");
        const auto block = GameEnvironment({L"AMD_LOG_LEVEL=4"});
        wstring all;
        for (wchar_t c : block) all += c ? c : L'|';
        require(all.find(L"|HIP_PATH=") == wstring::npos && all.find(L"|ROCM_PATH=") == wstring::npos && all.rfind(L"HIP_PATH=", 0) != 0, "Play: the HIP SDK variables are not passed to the game");
        require(all.find(L"WG_KEEP_ME=1") != wstring::npos && all.find(L"AMD_LOG_LEVEL=4") != wstring::npos, "Play: other variables and the extra ones are passed");

        // ---- Play: through Steam for a game with a Steam app id, else directly (the starts stubbed: no Steam, no game) ----------
        {
            const wstring lib = Join(dir, L"steamlib\\steamapps");
            auto file = [](const wstring& p, const string& text) { EnsureDir(Parent(p)); WriteFileAtomic(p, text.data(), text.size()); };
            file(Join(lib, L"appmanifest_1234560.acf"), "\"AppState\"\n{\n\t\"appid\"\t\t\"1234560\"\n\t\"name\"\t\t\"Quarry Nine\"\n\t\"installdir\"\t\t\"Quarry Nine\"\n}\n");
            file(Join(lib, L"appmanifest_777.acf"), "\"AppState\"\n{\n\t\"name\"\t\t\"Odd Layout\"\n\t\"installdir\"\t\t\"Odd Layout\"\n}\n");  // no appid key
            const wstring unreal = Join(lib, L"common\\Quarry Nine\\QuarryNine\\Binaries\\Win64\\WgPlayTestQuarry-Win64-Shipping.exe");
            const wstring odd = Join(lib, L"common\\Odd Layout\\engine\\bin\\retail\\WgPlayTestOdd.exe");     // not a known store layout
            const wstring unlisted = Join(lib, L"common\\Unlisted\\WgPlayTestUnlisted.exe");                  // no appmanifest names it
            const wstring other = Join(dir, L"Games\\Other Game\\WgPlayTestOther.exe");                         // no Steam library
            for (auto& e : {unreal, odd, unlisted, other}) file(e, "MZ");
            require(SteamAppId(unreal) == L"1234560", "Play: the app id of the appmanifest whose installdir holds the game (Unreal layout)");
            require(SteamAppId(odd) == L"777", "Play: any depth below steamapps\\common\\<installdir>; the id from the file name without an appid key");
            require(SteamAppId(unlisted).empty() && SteamAppId(other).empty() && SteamAppId(Join(lib, L"common\\WgLoose.exe")).empty(), "Play: no app id without a matching appmanifest or outside a Steam library");

            PlayWay w = ChoosePlayWay(unreal, true);
            require(w.steam && w.appid == L"1234560" && w.note.empty(), "Play: a Steam game with an app id starts through Steam");
            w = ChoosePlayWay(unreal, false);
            require(!w.steam && w.appid == L"1234560" && !w.note.empty(), "Play: directly when Windows has no steam: address handler");
            w = ChoosePlayWay(other, true);
            require(!w.steam && w.appid.empty() && w.note.empty(), "Play: a game outside Steam starts directly");
            w = ChoosePlayWay(unlisted, true);
            require(!w.steam && w.appid.empty(), "Play: a Steam folder without its app id starts directly");

            std::vector<wstring> urls, directs;
            bool urlOk = true, steamKnown = true;
            PlayStarters stub;
            stub.steamKnown = [&] { return steamKnown; };
            stub.openUrl = [&](const wstring& u, wstring* why) { urls.push_back(u); if (!urlOk && why) *why = L"no Steam"; return urlOk; };
            stub.direct = [&](const wstring& e, wstring* why) { directs.push_back(e); if (why) why->clear(); return true; };
            Context pc;
            pc.logFile = Join(dir, L"play.log");
            auto logText = [&] { string t; ReadAll(pc.logFile, t); return t; };
            PlayWay how;
            require(PlayGame(unreal, &why, &how, &pc, stub) && how.steam && how.appid == L"1234560", "Play: started through Steam");
            require(urls.size() == 1 && urls[0] == L"steam://rungameid/1234560" && directs.empty(), "Play: steam://rungameid/<appid>, no direct start");
            require(logText().find("started through Steam (app id 1234560") != string::npos, "Play: the log says it started through Steam and the app id");
            require(PlayGame(other, &why, &how, &pc, stub) && !how.steam && how.appid.empty(), "Play: a non-Steam game started directly");
            require(urls.size() == 1 && directs.size() == 1 && directs[0] == other, "Play: the direct start for a game without an app id, no steam: address");
            require(logText().find("WgPlayTestOther.exe started directly") != string::npos, "Play: the log says it started directly");
            require(PlayGame(unlisted, &why, &how, &pc, stub) && !how.steam && urls.size() == 1 && directs.size() == 2, "Play: the direct start when no app id is found");
            steamKnown = false;
            require(PlayGame(unreal, &why, &how, &pc, stub) && !how.steam && how.appid == L"1234560" && urls.size() == 1 && directs.size() == 3, "Play: the direct start without a steam: handler");
            steamKnown = true; urlOk = false;
            require(PlayGame(unreal, &why, &how, &pc, stub) && !how.steam && urls.size() == 2 && directs.size() == 4 && how.note.find(L"no Steam") != wstring::npos,
                    "Play: the direct start when starting through Steam fails, and why");
            require(logText().find("started directly (Steam app id 1234560: starting it through Steam failed") != string::npos, "Play: the log says why a Steam game started directly");
            urlOk = true;
            require(!PlayGame(Join(dir, L"Games\\Missing\\WgPlayTestMissing.exe"), &why, &how, &pc, stub) && why.find(L"not found") != wstring::npos && urls.size() == 2 && directs.size() == 4,
                    "Play: a missing exe is refused before either start");

            // rc6 L7: an Epic game starts through Epic (com.epicgames.launcher://apps/<AppName>), else directly; the starts stubbed
            const wstring oldPd = EnvVar(L"ProgramData"), pd = Join(dir, L"programdata");
            const wstring epicGame = Join(dir, L"Epic Games\\Quarry Two\\Binaries\\Win64\\WgPlayTestEpic.exe");
            file(epicGame, "MZ");
            auto jsonPath = [](const wstring& p) { string e; for (char c : ToUtf8(p)) { if (c == '\\') e += '\\'; e += c; } return e; };  // JSON escapes '\'
            file(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests\\ABC.item"),
                 "{\n\t\"DisplayName\": \"Quarry Two\",\n\t\"InstallLocation\": \"" + jsonPath(Join(dir, L"Epic Games\\Quarry Two")) + "\",\n\t\"AppName\": \"QuarryTwo_App-1.0\"\n}\n");
            file(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests\\BAD.item"),
                 "{\n\t\"InstallLocation\": \"" + jsonPath(Join(dir, L"Epic Games\\Odd")) + "\",\n\t\"AppName\": \"a b&c\"\n}\n");
            file(Join(dir, L"Epic Games\\Odd\\WgOdd.exe"), "MZ");
            SetEnvironmentVariableW(L"ProgramData", pd.c_str());
            require(EpicAppName(epicGame) == L"QuarryTwo_App-1.0" && EpicAppName(other).empty() && EpicAppName(Join(dir, L"Epic Games\\Quarry Twofold\\x.exe")).empty(),
                    "Play: the Epic AppName of the manifest whose InstallLocation holds the game (none elsewhere)");
            require(EpicAppName(Join(dir, L"Epic Games\\Odd\\WgOdd.exe")).empty(), "Play: an AppName that can't go into an address is none");
            w = ChoosePlayWay(epicGame, true, true, L"QuarryTwo_App-1.0");
            require(w.epic && !w.steam && w.epicApp == L"QuarryTwo_App-1.0" && w.note.empty(), "Play: an Epic game starts through Epic");
            w = ChoosePlayWay(epicGame, true, false, L"QuarryTwo_App-1.0");
            require(!w.epic && !w.note.empty(), "Play: directly when Windows has no com.epicgames.launcher: handler");
            w = ChoosePlayWay(unreal, true, true, L"QuarryTwo_App-1.0");
            require(w.steam && !w.epic, "Play: a Steam app id goes first");
            bool epicKnown = true;
            stub.epicKnown = [&] { return epicKnown; };
            stub.epicApp = [](const wstring& e) { return EpicAppName(e); };
            urls.clear(); directs.clear();
            require(PlayGame(epicGame, &why, &how, &pc, stub) && how.epic && !how.steam && how.epicApp == L"QuarryTwo_App-1.0" && directs.empty() && urls.size() == 1 &&
                        urls[0] == L"com.epicgames.launcher://apps/QuarryTwo_App-1.0?action=launch&silent=true",
                    "Play: com.epicgames.launcher://apps/<AppName>?action=launch&silent=true, no direct start");
            require(logText().find("started through Epic (app QuarryTwo_App-1.0") != string::npos, "Play: the log says it started through Epic and the app");
            urlOk = false;
            require(PlayGame(epicGame, &why, &how, &pc, stub) && !how.epic && how.epicApp == L"QuarryTwo_App-1.0" && urls.size() == 2 && directs.size() == 1 &&
                        how.note.find(L"through Epic failed") != wstring::npos,
                    "Play: the direct start when starting through Epic fails, and why");
            require(logText().find("started directly (Epic app QuarryTwo_App-1.0: starting it through Epic failed") != string::npos, "Play: the log says why an Epic game started directly");
            urlOk = true; epicKnown = false;
            require(PlayGame(epicGame, &why, &how, &pc, stub) && !how.epic && urls.size() == 2 && directs.size() == 2, "Play: the direct start without an Epic handler");
            epicKnown = true;
            require(PlayGame(other, &why, &how, &pc, stub) && !how.epic && how.epicApp.empty() && urls.size() == 2 && directs.size() == 3, "Play: a game outside Epic and Steam starts directly");
            SetEnvironmentVariableW(L"ProgramData", oldPd.c_str());
        }

        // ---- global settings round trip --------------------------------------------------------------------------------------
        LauncherPaths lp; lp.dataDir = Join(dir, L"data");
        GlobalSettings g1;
        g1.optiPackage = L"C:\\dl\\Opti.7z"; g1.extraFolders = {L"K:\\Games"}; g1.playButton = false; g1.updateCheck = true; g1.updateUrl = L"https://example.invalid/u";
        g1.appearance = L"light"; g1.accent = L"graphite"; g1.exactMode = true;
        SetSetting(g1.defaults, "output_filter", "analytic", nullptr);
        SaveGlobal(lp, g1);
        const GlobalSettings g2 = LoadGlobal(lp);
        require(g2.optiPackage == g1.optiPackage && g2.extraFolders == g1.extraFolders && !g2.playButton && g2.updateCheck && g2.updateUrl == g1.updateUrl &&
                g2.defaults.get("output_filter") == "analytic" && g2.appearance == L"light" && g2.accent == L"graphite" && g2.exactMode, "launcher.ini round trip");
        GlobalSettings fresh = LoadGlobal(LauncherPaths{L"", Join(dir, L"none")});
        require(!fresh.updateCheck && fresh.updateUrl.empty() && fresh.playButton && fresh.appearance == L"dark" && fresh.accent == L"green" && !fresh.exactMode,
                "defaults: update check off, no URL, Play button shown, the dark theme, the green accent, Exact mode off");
        {
            GlobalSettings x;
            require(SetGlobal(x, "exact_mode", "1", nullptr) && x.exactMode && SetGlobal(x, "exact_mode", "0", nullptr) && !x.exactMode && !SetGlobal(x, "exact_mode", "maybe", nullptr),
                    "exact_mode: 0 or 1");
        }
        {
            GlobalSettings a;
            require(SetGlobal(a, "appearance", "Light", nullptr) && a.appearance == L"light" && !SetGlobal(a, "appearance", "blue", nullptr) && a.appearance == L"light",
                    "appearance: system, light or dark only");
            require(SetGlobal(a, "accent", "Blue", nullptr) && a.accent == L"blue" && SetGlobal(a, "accent", "windows", nullptr) && !SetGlobal(a, "accent", "red", nullptr) &&
                        !SetGlobal(a, "accent", "orange", nullptr) && a.accent == L"windows",
                    "accent: green, blue, purple, pink, cyan, graphite or windows only (red and amber keep their meaning)");
        }

        // ---- the accent colours: every accent, both themes (BE-wide notes.md "Accent color"): white words on the filled
        // buttons 4.5:1, the accent's words 4.5:1 on the page, a cell, the fill and its tint, its graphics (switch track, the
        // chosen radio and swatch ring) 3:1 on the page and a cell; the working green the same; and the Windows accent made
        // into tokens by the same rules from colours that are hard to read (yellow, near white, near black, Windows' red)
        {
            using namespace wg::app::lui;
            const Col white = Hex(0xffffff);
            auto check = [&](const AccentTokens& k, bool dark, const char* what) {
                const Surfaces s = SurfacesOf(dark);
                const bool ok = Contrast(white, k.f) >= 4.5f && Contrast(k.t, s.bg) >= 4.5f && Contrast(k.t, s.cell) >= 4.5f && Contrast(k.t, s.fill) >= 4.5f &&
                                Contrast(k.t, k.tint) >= 4.5f && Contrast(k.t, k.tintHover) >= 4.5f && Contrast(k.c, s.bg) >= 3.f && Contrast(k.c, s.cell) >= 3.f;
                if (!ok) std::printf("  contrast: %s (%s): white/fill %.2f text/page %.2f text/cell %.2f text/fill %.2f text/tint %.2f graphics/page %.2f graphics/cell %.2f\n",
                                     what, dark ? "dark" : "light", Contrast(white, k.f), Contrast(k.t, s.bg), Contrast(k.t, s.cell), Contrast(k.t, s.fill),
                                     Contrast(k.t, k.tintHover), Contrast(k.c, s.bg), Contrast(k.c, s.cell));
                return ok;
            };
            bool contrastOk = true;
            for (int dark = 0; dark < 2; ++dark) {
                for (int a = 0; a < A_COUNT; ++a) contrastOk = check(AccentFor(a, dark != 0, 0x0078d4), dark != 0, "accent") && contrastOk;
                for (unsigned w : {0x0078d4u, 0xffb900u, 0xfff4cdu, 0x101010u,  // Windows blue, yellow, near white, near black
                                   0xe81123u, 0x00cc6au, 0xffffffu, 0x000000u}) // Windows red, green, white, black
                    contrastOk = check(AccentFor(A_WINDOWS, dark != 0, w), dark != 0, "windows accent") && contrastOk;
                const Surfaces s = SurfacesOf(dark != 0);
                contrastOk = contrastOk && Contrast(StatusGreen(dark != 0), s.bg) >= 3.f && Contrast(StatusGreen(dark != 0), s.cell) >= 3.f &&
                      Contrast(StatusGreenText(dark != 0), s.bg) >= 4.5f && Contrast(StatusGreenText(dark != 0), s.cell) >= 4.5f;
            }
            require(contrastOk, "accent colours: 4.5:1 for words and filled buttons, 3:1 for graphics, every accent and theme");
            // a filled button that is off, red or any accent, either theme: one rule (DisabledFilled): its label 3:1 or better
            // and under 4:1 on its faded fill, and weaker than the live label on the live fill (never reads as live)
            bool offOk = true;
            for (int dark = 0; dark < 2; ++dark) {
                const Surfaces s = SurfacesOf(dark != 0);
                std::vector<Col> fills = {Hex(dark ? 0xc9362a : 0xc42b1c)};  // the red (needs you)
                for (int a = 0; a < A_COUNT; ++a) fills.push_back(AccentFor(a, dark != 0, 0x0078d4).f);
                for (unsigned w : {0xffb900u, 0x101010u, 0xe81123u}) fills.push_back(AccentFor(A_WINDOWS, dark != 0, w).f);
                for (const Col& f : fills) {
                    const FilledOff o = DisabledFilled(f, s.cell, white, dark != 0);
                    const float c = Contrast(o.label, o.fill);
                    if (c < 3.f || c >= 4.f || c >= Contrast(white, f)) {
                        offOk = false;
                        std::printf("  disabled filled (%s): label/fill %.2f, live %.2f\n", dark ? "dark" : "light", c, Contrast(white, f));
                    }
                }
            }
            require(offOk, "disabled filled buttons: one rule for red and every accent, label 3:1 to 4:1 on its fill, weaker than live");
            require(AccentFromKey(L"graphite") == A_GRAPHITE && AccentFromKey(L"nonsense") == A_GREEN && std::wstring(AccentKey(A_CYAN)) == L"cyan",
                    "accent keys round trip; an unknown one is the default");
        }

        // ---- Ray Reconstruction (rr-notes.md) -------------------------------------------------------------------------
        {
            // the profile's two RR settings: validated, normalised, and never written into the upscaler's wronggpu.ini
            const SettingDef* rrOn = FindSetting("rr_enabled");
            const SettingDef* rrMode = FindSetting("rr_mode");
            require(rrOn && rrMode && rrOn->layer == Layer::Rr && rrMode->layer == Layer::Rr, "rr_enabled and rr_mode are profile settings of their own layer");
            require(ValidateSetting(*rrOn, "1", &n, &why) && n == "on" && ValidateSetting(*rrOn, "false", &n, &why) && n == "off", "rr_enabled 1 = on, false = off");
            require(ValidateSetting(*rrMode, "Fast", &n, &why) && n == "fast" && !ValidateSetting(*rrMode, "turbo", &n, &why), "rr_mode: exact, accurate or fast only");
            Profile rp = DefaultProfile();
            require(rp.get("rr_enabled") == "on" && rp.get("rr_mode") == "fast", "RR defaults: on, Fast (owner 2026-11-05)");
            SetSetting(rp, "rr_mode", "fast", nullptr);
            SetSetting(rp, "rr_enabled", "off", nullptr);
            require(RenderWronggpuIni(rp, front, out, &why) && out == front, "the RR settings leave the upscaler's wronggpu.ini alone");
            require(OutsideEdits(rp, front, "").empty(), "RR settings are never outside edits of the upscaler's file");

            // WrongGPU\wronggpu.ini: rendered from the profile, read back, the plugin's own keys and comments kept
            require(ParseRrSettings(RenderRrSettings(DefaultProfile(), true, "")).mode == "fast", "a new RR settings file from the default profile: rr_mode=fast written (rc5)");
            Profile acc = DefaultProfile();
            SetSetting(acc, "rr_mode", "accurate", nullptr);  // a player's Accurate (rc5: no longer the default)
            const string rrNew = RenderRrSettings(acc, true, "");
            RrFileState st = ParseRrSettings(rrNew);
            require(st.present && st.enabled && st.mode == "accurate", "a new RR settings file: on, Accurate");
            require(rrNew.find("[wronggpu]\r\n") != string::npos && rrNew.find("rr_enabled=1\r\n") != string::npos && rrNew.find("rr_exact_allowed=0\r\n") != string::npos,
                    "rr_enabled=1, rr_exact_allowed=0 in [wronggpu], CRLF");
            require(RenderRrSettings(acc, true, rrNew) == rrNew, "rendering twice gives the same file (round trip)");
            require(ParseRrSettings(RenderRrSettings(acc, false, rrNew)).enabled == false, "no RR data: rr_enabled=0 whatever the profile says");
            Profile ex = DefaultProfile();
            SetSetting(ex, "rr_mode", "exact", nullptr);
            const string exText = RenderRrSettings(ex, true, rrNew);
            require(ParseRrSettings(exText).mode == "exact" && exText.find("rr_exact_allowed=1\r\n") != string::npos, "exact: rr_mode=exact with rr_exact_allowed=1");
            require(RenderRrSettings(acc, true, exText).find("rr_exact_allowed=0\r\n") != string::npos, "back from exact: rr_exact_allowed=0 again");
            const string plugin = "# the plugin's\r\n[wronggpu]\r\nrr_mode = fast\r\nrr_hud = 1\r\n[other]\r\nrr_mode = exact\r\n";
            st = ParseRrSettings(plugin);
            require(st.mode == "fast" && st.enabled, "only [wronggpu] is read; a missing rr_enabled is on");
            const string kept = RenderRrSettings(acc, true, plugin);
            require(kept.find("# the plugin's\r\n") != string::npos && kept.find("rr_hud = 1") != string::npos && kept.find("[other]\r\nrr_mode = exact") != string::npos &&
                        ParseRrSettings(kept).mode == "accurate",
                    "the plugin's comment, keys and other sections stay; rr_mode follows the profile");
            require(RrIniText(L"C:\\Users\\P\\AppData\\Local\\WrongGPU\\model\\F\\model.wgcm") ==
                        "# WrongGPU Ray Reconstruction (written by the WrongGPU launcher; the RR plugin adds its own keys)\r\nrr_model=C:\\Users\\P\\AppData\\Local\\WrongGPU\\model\\F\\model.wgcm\r\n",
                    "wronggpu-rr.ini names the RR data");

            // the mode switched in the game (OptiScaler's menu) becomes the profile's
            const wstring rg = Join(dir, L"rrgame");
            EnsureDir(Join(rg, kRrDataDir));
            Profile fold = DefaultProfile();
            require(!FoldRrFromGame(rg, fold), "no RR settings file: nothing to fold");
            // (rc5: the file's mode differs from the default, Fast)
            { const string t = "[wronggpu]\r\nrr_enabled=1\r\nrr_mode=accurate\r\n"; WriteFileAtomic(Join(rg, kRrSettings), t.data(), t.size()); }
            require(FoldRrFromGame(rg, fold) && fold.get("rr_mode") == "accurate" && !FoldRrFromGame(rg, fold), "the game's rr_mode folds into the profile once");
            require(ReadRrSettings(rg).present && ReadRrSettings(Join(dir, L"nothing")).present == false, "ReadRrSettings: present only with the file");

            // which games: the RR files next to the exe, deeper in the game's folder, a game known to call it, none
            auto file = [](const wstring& p) { EnsureDir(Parent(p)); WriteFileAtomic(p, "x", 1); };
            const wstring a = Join(dir, L"rr-a"), b = Join(dir, L"rr-b"), c = Join(dir, L"rr-c"), d = Join(dir, L"rr-d");
            file(Join(a, L"bin\\x64\\Game.exe")); file(Join(a, L"bin\\x64\\nvngx_dlssd.dll"));  // DLSS Ray Reconstruction file
            file(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe")); file(Join(b, L"Engine\\Plugins\\RR\\Binaries\\sl.dlss_d.dll"));  // DLSS
            file(Join(c, L"HogwartsLegacy.exe"));
            file(Join(d, L"Game.exe")); file(Join(d, L"nvngx_dlss.dll"));  // DLSS (the upscaler's file alone: no RR)
            RrUse ru = DetectRrUncached(Join(a, L"bin\\x64\\Game.exe"));
            require(ru.uses && IEq(ru.why, L"nvngx_dlssd.dll"), "the RR model file next to the exe: uses RR");  // DLSS
            ru = DetectRrUncached(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe"));
            require(ru.uses && ru.why.find(L"sl.dlss_d.dll") != wstring::npos, "the RR plugin deeper in the game's folder: uses RR");  // DLSS
            require(DetectRrUncached(Join(c, L"HogwartsLegacy.exe")).uses, "a game known to call RR directly");
            require(!DetectRrUncached(Join(d, L"Game.exe")).uses && !DetectRrUncached(Join(d, L"Missing.exe")).uses, "a game without RR (and a missing exe): no");
            require(DetectRr(Join(a, L"bin\\x64\\Game.exe")).uses && !DetectRr(Join(d, L"Game.exe")).uses, "the cached detection agrees");
            // direct-RR games (2026-11-04): a game that creates RR itself (the RR library without the game interface's RR plugin) is
            // "direct"; the game interface's RR plugin anywhere in the folder makes it not direct; the known games' table decides
            // for its games (Hogwarts Legacy direct, RE Requiem through the game interface)
            require(DetectRrUncached(Join(a, L"bin\\x64\\Game.exe")).direct, "the RR library alone next to the exe: direct");
            require(!DetectRrUncached(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe")).direct, "the game interface's RR plugin: not direct");
            {
                const wstring u = Join(dir, L"rr-unreal"), cp = Join(dir, L"rr-both"), r9 = Join(dir, L"rr-re9"), late = Join(dir, L"rr-late");
                file(Join(u, L"Lego\\Binaries\\Win64\\Lego-Win64-Shipping.exe"));
                file(Join(u, L"Engine\\Plugins\\Runtime\\Vendor\\RR\\Binaries\\ThirdParty\\Win64\\nvngx_dlssd.dll"));  // DLSS RR (the Unreal plugin layout)
                RrUse ur = DetectRrUncached(Join(u, L"Lego\\Binaries\\Win64\\Lego-Win64-Shipping.exe"));
                require(ur.uses && ur.direct && ur.why.find(L"nvngx_dlssd.dll") != wstring::npos, "the Unreal plugin layout (RR library deep in Engine): direct, the library named");
                file(Join(cp, L"bin\\x64\\Game.exe")); file(Join(cp, L"bin\\x64\\nvngx_dlssd.dll")); file(Join(cp, L"bin\\x64\\sl.dlss_d.dll"));  // DLSS
                ur = DetectRrUncached(Join(cp, L"bin\\x64\\Game.exe"));
                require(ur.uses && !ur.direct && IEq(ur.why, L"sl.dlss_d.dll"), "the RR library and the game interface's RR plugin next to the exe: not direct");  // DLSS
                file(Join(late, L"bin\\x64\\Game.exe")); file(Join(late, L"bin\\x64\\nvngx_dlssd.dll")); file(Join(late, L"zz\\plugins\\sl.dlss_d.dll"));  // DLSS
                require(!DetectRrUncached(Join(late, L"bin\\x64\\Game.exe")).direct, "the library next to the exe, the game interface's RR plugin elsewhere in the folder: not direct");
                require(DetectRrUncached(Join(c, L"HogwartsLegacy.exe")).direct, "Hogwarts Legacy (known game): direct");
                file(Join(r9, L"re9.exe")); file(Join(r9, L"nvngx_dlssd.dll"));  // DLSS: the library alone, but the table says RE Requiem is not direct
                require(DetectRrUncached(Join(r9, L"re9.exe")).uses && !DetectRrUncached(Join(r9, L"re9.exe")).direct, "RE Requiem (known game): never direct");
                const wstring cpx = Join(r9, L"cp\\Cyberpunk2077.exe"), cqx = Join(r9, L"cq\\Cyberpunk2077.exe");
                file(cpx); file(Join(r9, L"cp\\nvngx_dlssd.dll"));  // DLSS
                file(cqx);
                require(DetectRrUncached(cpx).uses && !DetectRrUncached(cpx).direct && !DetectRrUncached(cqx).uses,
                        "Cyberpunk 2077: never direct; RR only by its files (its row adds no use)");
            }
            {   // the install's OptiScaler from its record: the proxy's copy entry has the edition DLL's hash
                const wstring ed = PinnedEdition().dllSha256;
                Record er; er.proxy = L"dxgi.dll";
                er.e.push_back({L"copy", L"dxgi.dll", false, L"-", ed, L"wg-optibundle"});
                require(!ed.empty() && RecordHasEdition(er), "a record whose OptiScaler DLL has the edition's hash: the edition");
                er.e.back().inst = L"0000000000000000000000000000000000000000000000000000000000000000";
                require(!RecordHasEdition(er), "another OptiScaler DLL: not the edition");
                Record nr; nr.e.push_back({L"copy", L"winmm.dll", false, L"-", ed, L"wg-optibundle"});
                require(RecordHasEdition(nr), "no proxy line: OptiScaler's DLL under a proxy name with the edition's hash");
                Record pr; pr.proxy = L"dxgi.dll"; pr.e.push_back({L"copy", L"plugins\\WrongGPU.asi", false, L"-", ed, L"wg"});
                require(!RecordHasEdition(pr), "the hash on another file than the proxy: not the edition");
                require(RrDirectNoEditionText(L"Some Game").find(L"no OptiScaler edition") != wstring::npos &&
                        RrDirectNoEditionText(L"Some Game").find(L"isn't offered in Some Game, and the game") != wstring::npos,
                        "the warning carries the words the window turns into the player's sentence (ui_words.cpp)");
            }
            ForgetRrDetection();

            // the last run's RR log: the session after the last "attached", the mode it ran, the GPU time a frame
            LastRun rl;
            ParseRrLog("[10:00:00] load: process 100 attached\r\n[10:00:01] runtime: create instance\r\nrr_mode=fast\r\n"
                       "[10:02:00] load: process 200 attached\r\n[10:02:01] rr_mode=accurate\r\n[10:02:02] runtime: create instance\r\n"
                       "[10:02:03] rr_mode accurate -> fast\r\n[10:03:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", rl);
            require(rl.rrFound && rl.rrUsed && rl.rrMode == L"fast" && rl.rrMs == L"6.8", "RR log: used, the mode switched in game, 6.8 ms a frame");
            LastRun lr2;
            ParseRrLog("load: process 1 attached\r\nruntime: create instance\r\nload: process 2 attached\r\nrr_mode=exact\r\nrr_exact_allowed=0: using accurate\r\n", lr2);
            require(lr2.rrFound && !lr2.rrUsed && lr2.rrMode == L"accurate" && lr2.rrMs.empty(), "RR log: the last session only; a refused Exact runs Accurate; no timing, no cost");
            LastRun lr3;
            ParseRrLog("nothing here\r\n", lr3);
            require(!lr3.rrFound && !lr3.rrUsed, "no session in the log: no RR line");
            LastRun lr4;  // a long session: its "attached" line is out of the 256 KB tail, its timing lines are not
            ParseRrLog("ames at 2560x1440: GPU 7.10 ms per frame\r\n[11:00:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", lr4);
            require(lr4.rrFound && lr4.rrUsed && lr4.rrMode.empty() && lr4.rrMs == L"6.8", "RR log: the attach line scrolled out of the tail: still used, its last cost");
            LastRun lr5;
            ParseRrLog("load: process 7 detached\r\n[11:00:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", lr5);
            require(!lr5.rrFound && !lr5.rrUsed, "RR log: another load line without an attach in the tail: not guessed");

            // which build of the RR plugin a game gets: by the version of the game interface it hooks (its major.minor), the
            // general plugin otherwise; never by the game's name
            require(RrPluginVersion(L"plugins\\2.8\\WrongGPU-RR.asi") == L"2.8" && RrPluginVersion(kRrPayloadPlugin).empty() &&
                    RrPluginVersion(L"plugins\\x\\WrongGPU-RR.asi").empty() && RrPluginVersion(L"plugins\\2.8.1\\WrongGPU-RR.asi").empty() &&
                    RrPluginVersion(L"other\\2.8\\WrongGPU-RR.asi").empty(), "the plugin's builds: plugins\\<major>.<minor>\\ only");
            require(IsRrPayloadPlugin(kRrPayloadPlugin) && IsRrPayloadPlugin(L"plugins\\2.14\\WrongGPU-RR.asi") && !IsRrPayloadPlugin(kRrRuntime) &&
                    !IsRrPayloadPlugin(kRrPayloadEdition), "IsRrPayloadPlugin");
            const std::vector<wstring> rels{kRrPayloadPlugin, L"plugins\\2.8\\WrongGPU-RR.asi", L"plugins\\2.7\\WrongGPU-RR.asi", kRrRuntime, kRrPayloadEdition};
            require(IEq(PickRrPlugin(rels, L"2.8.0.0"), L"plugins\\2.8\\WrongGPU-RR.asi") && IEq(PickRrPlugin(rels, L"2.7.1.0"), L"plugins\\2.7\\WrongGPU-RR.asi"),
                    "a game interface 2.8.0 / 2.7.1: the build for its version");
            require(IEq(PickRrPlugin(rels, L"2.14.1.0"), kRrPayloadPlugin) && IEq(PickRrPlugin(rels, L""), kRrPayloadPlugin) &&
                    IEq(PickRrPlugin(rels, L"2.80.0.0"), kRrPayloadPlugin), "no build for its version (or no interface file): the general plugin");
            require(PickRrPlugin({kRrRuntime}, L"2.8.0.0").empty(), "no plugin at all: none");

            // the per-game rules: RE Requiem needs REFramework, its RR needs the host's restore and comes with path tracing only
            const GameRule* re9 = GameRuleFor(L"K:\\Games\\RE\\re9.exe");
            require(re9 && re9->needsReFramework && re9->needsHostRestore && re9->rrOnlyPathTracing && GameRuleFor(L"RE9DEMO.EXE") &&
                    !GameRuleFor(L"Cyberpunk2077.exe"), "GameRuleFor: re9.exe and its demo, nothing for other games");
            require(NeedsReFrameworkText(*re9).find(L"needs REFramework installed first") != wstring::npos, "the refusal's sentence");
            // REFramework: dinput8.dll whose own words say REFramework (never the name alone), the folder as evidence
            const wstring rf = Join(dir, L"refw");
            file(Join(rf, L"dinput8.dll"));
            require(!DetectReFramework(rf).found, "another dinput8.dll: not REFramework");
            { const string t = "MZ synthetic loader stand-in: REFramework v1.5 (test)"; WriteFileAtomic(Join(rf, L"dinput8.dll"), t.data(), t.size()); }
            ReFrameworkState rs = DetectReFramework(rf);
            require(rs.found && rs.evidence == L"dinput8.dll", "REFramework's loader by its words");
            EnsureDir(Join(rf, L"reframework"));
            rs = DetectReFramework(rf);
            require(rs.found && rs.evidence.find(L"folder reframework") != wstring::npos, "its folder named as evidence");
            require(!DetectReFramework(Join(dir, L"nothing")).found, "no dinput8.dll: not found");
            // its mirror: the install folder's root DLLs only
            require(MirroredRel(L"dxgi.dll") && MirroredRel(L"wronggpu_rr_runtime.dll") && !MirroredRel(L"plugins\\WrongGPU-RR.asi") &&
                    !MirroredRel(L"OptiScaler.ini") && IsMirrorRel(MirrorRel(L"dxgi.dll")) && !IsMirrorRel(L"dxgi.dll") &&
                    MirrorRel(L"dxgi.dll") == L"_storage_\\dxgi.dll", "REFramework's mirror: root DLLs, _storage_\\<name>");
            // an install record's mirror entries are REFramework's to rewrite: never a problem of the install
            Record mr;
            Entry me; me.action = L"copy"; me.rel = MirrorRel(L"dxgi.dll"); me.inst = L"00"; mr.e.push_back(me);
            require(VerifyInstall(Join(dir, L"nothing"), mr).empty(), "VerifyInstall: a missing mirror copy is no problem");
        }
        {
            // an install an earlier build set up to start on Preset K (0.5.0 / 0.5.1): the plugin's choice first, then the
            // runtime options' last model item, then the record's model file
            const wstring pk = Join(dir, L"presetk");
            EnsureDir(pk);
            auto put = [&](const wchar_t* name, const string& text) { WriteFileAtomic(Join(pk, name), text.data(), text.size()); };
            Record noRec;
            require(!StartsOnPresetK(pk, &noRec), "Preset K: nothing there, not K");
            put(L"wronggpu.ini", "[fsr]\r\nupscaler = network\r\nruntime_options = output_filter=original;model=K\r\n");
            require(StartsOnPresetK(pk, &noRec), "Preset K: runtime_options model=K");
            put(L"wronggpu.ini", "[fsr]\r\nruntime_options = model=K;model=M\r\n");
            require(!StartsOnPresetK(pk, &noRec), "Preset K: the last model item wins (M)");
            put(L"wronggpu-asi.ini", "; WrongGPU OptiScaler plugin\r\nselection=wronggpu-k\r\n");
            require(StartsOnPresetK(pk, &noRec), "Preset K: the plugin's choice decides over wronggpu.ini");
            put(L"wronggpu-asi.ini", "selection=wronggpu\r\n");
            put(L"wronggpu.ini", "[fsr]\r\nruntime_options = model=K\r\n");
            require(!StartsOnPresetK(pk, &noRec), "Preset K: the plugin on WrongGPU (Model M) decides over model=K");
            DeleteFileW(Join(pk, L"wronggpu-asi.ini").c_str());
            put(L"wronggpu.ini", "[fsr]\r\nupscaler = network\r\n");
            Record rk; rk.model = Join(Join(ModelBaseDir(), L"K"), L"model.wgcm");
            Record rm; rm.model = Join(ModelBaseDir(), L"model.wgcm");
            require(StartsOnPresetK(pk, &rk) && !StartsOnPresetK(pk, &rm), "Preset K: the record's model file in the K cache");
        }

        // ---- the Engine.ini file check: a file it can't read is "couldn't read", never "no line" (r3 #8) -------------------
        {
            Context rc;
            rc.playerDocuments = Join(dir, L"docs");  // a test root, never this PC's Documents
            const RrConfigGame* bl4 = RrConfigGameFor(L"Borderlands4.exe");
            const wstring cfg = Join(Join(Join(Join(Join(rc.playerDocuments, L"My Games"), L"Borderlands 4"), L"Saved"), L"Config"), L"Windows");
            EnsureDir(cfg);
            { const string gus = "[ScalabilityGroups]\r\n"; WriteFileAtomic(Join(cfg, L"GameUserSettings.ini"), gus.data(), gus.size()); }
            const wstring ini = Join(cfg, L"Engine.ini");
            wstring f, rwhy;
            require(bl4 && RrConfigLineState(rc, *bl4, &f, &rwhy) == 0 && IEq(f, ini), "Engine.ini check: no file is no line (0)");
            const string utf16("\xFF\xFE[\0S\0", 6);
            WriteFileAtomic(ini, utf16.data(), utf16.size());
            rwhy.clear();
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 3 && rwhy.find(L"UTF-16") != wstring::npos, "Engine.ini check: UTF-16 is couldn't read (3), why says so");
            const string line = RrConfigNewFile();
            WriteFileAtomic(ini, line.data(), line.size());
            HANDLE held = CreateFileW(ini.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);  // another program has it, no sharing
            rwhy.clear();
            const int st = RrConfigLineState(rc, *bl4, &f, &rwhy);
            if (held != INVALID_HANDLE_VALUE) CloseHandle(held);
            require(held != INVALID_HANDLE_VALUE && st == 3 && rwhy.find(L"another program") != wstring::npos, "Engine.ini check: a file held open is couldn't read (3)");
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 2, "Engine.ini check: readable again, the line there, not read-only (2)");
            // r4 #5: saved as UTF-8 with a byte order mark (Notepad's "UTF-8 with BOM"): the section right after it is found
            const string bom = "\xEF\xBB\xBF";
            const string withLine = bom + line;
            WriteFileAtomic(ini, withLine.data(), withLine.size());
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 2, "Engine.ini check with a BOM: the line in its [SystemSettings] is there (2), not missing (r4 #5)");
            const string noLine = bom + "[SystemSettings]\r\nr.Other=1\r\n\r\n[Core.Log]\r\nLogNet=Log\r\n";
            WriteFileAtomic(ini, noLine.data(), noLine.size());
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 0, "Engine.ini check with a BOM, the section without the line: no line (0) (r4 #5)");
            // adding: into that section (no second [SystemSettings]), the BOM kept; taking it out again: byte for byte as it was
            RrConfigFix fx;
            bool changed = false;
            const string added = RrConfigAddText(noLine, fx, &changed, true);
            size_t sections = 0;
            for (size_t at = added.find("[SystemSettings]"); at != string::npos; at = added.find("[SystemSettings]", at + 1)) ++sections;
            const string lineOnly = line.substr(line.find('\n') + 1);  // the line itself, with its CRLF
            require(changed && sections == 1 && !fx.sectionAdded && added.compare(0, 3, bom) == 0 &&
                        added == bom + "[SystemSettings]\r\nr.Other=1\r\n" + lineOnly + "\r\n[Core.Log]\r\nLogNet=Log\r\n",
                    "Engine.ini add with a BOM: the line under the existing [SystemSettings], no second section, the BOM kept (r4 #5)");
            require(RrConfigRemoveText(added, fx) == noLine, "Engine.ini revert with a BOM: byte for byte as before, the BOM kept (r4 #5)");
            bool againBom = true;
            RrConfigFix fx2;
            require(RrConfigAddText(withLine, fx2, &againBom, true) == withLine && !againBom, "Engine.ini add with a BOM and the line there: nothing to change (r4 #5)");
            // ...a file with the BOM alone: the section added after it, and taken out again leaves the BOM alone (reverted as empty)
            RrConfigFix fx3;
            const string onlyBom = RrConfigAddText(bom, fx3, &changed, true);
            require(onlyBom == bom + line && fx3.sectionAdded && RrConfigRemoveText(onlyBom, fx3) == bom,
                    "Engine.ini add to a file of a BOM alone: BOM, section, line; the revert leaves the BOM alone (r4 #5)");
        }

        // rc5: the launcher next to an installed copy decides by build at the same version (DecideLauncherStart), never by
        // the version string alone. Synthetic PE files: a COFF header with a link time, a tag byte that makes the files differ
        {
            auto pe = [&](const wchar_t* name, uint32_t linkTime, char tag) {
                string b(512, '\0');
                b[0] = 'M'; b[1] = 'Z';
                b[0x3C] = 0x40;                                   // e_lfanew
                b[0x40] = 'P'; b[0x41] = 'E';                     // the signature (two zero bytes follow)
                auto byte = [](unsigned v) { return static_cast<char>(static_cast<unsigned char>(v)); };
                b[0x44] = byte(0x64); b[0x45] = byte(0x86);       // Machine x64
                for (int i = 0; i < 4; ++i) b[0x48 + i] = byte((linkTime >> (8 * i)) & 0xFF);   // TimeDateStamp
                b[0x54] = byte(0xF0);                             // SizeOfOptionalHeader
                b[0x58] = 0x0B; b[0x59] = 0x02;                   // PE32+ magic
                b[400] = tag;
                const wstring p = Join(dir, name);
                WriteFileAtomic(p, b.data(), b.size());
                return p;
            };
            const uint32_t nov2 = 1793660000u, nov5 = 1793920000u;
            const wstring self = pe(L"self.exe", nov5, 'A'), same = pe(L"same.exe", nov5, 'A'), older = pe(L"older.exe", nov2, 'B'),
                          newer = pe(L"newer.exe", nov5 + 3600, 'C'), notPe = Join(dir, L"notpe.exe");
            WriteFileAtomic(notPe, "not a PE file", 13);
            require(FileLinkTime(self) == nov5 && FileLinkTime(notPe) == 0, "FileLinkTime: the COFF header's link time; 0 for a file that isn't a PE image");
            auto d = DecideLauncherStart(self, L"0.6.0", same, L"0.6.0");
            require(d.choice == LauncherStart::OpenInstalled && d.sameVersion && d.sameBuild, "same version, same build (same bytes): the installed copy opens");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.6.0");
            require(d.choice == LauncherStart::OfferReplace && d.sameVersion && !d.sameBuild && d.installedBuilt == nov2 && d.why.find(L"older build") != wstring::npos,
                    "same version, a different and older build installed: this copy offers to replace it (never opens the older build)");
            d = DecideLauncherStart(self, L"0.6.0", newer, L"0.6.0");
            require(d.choice == LauncherStart::OpenInstalled && d.sameVersion && !d.sameBuild && d.why.find(L"newer build") != wstring::npos,
                    "same version, a newer build installed: the installed copy opens");
            // an installed copy that isn't a readable program (corrupt, cut short): this copy runs, whatever its entry's version
            for (const wchar_t* entryVersion : {L"0.6.0", L"0.6.1", L"0.4.5"}) {
                d = DecideLauncherStart(self, L"0.6.0", notPe, entryVersion);
                require(d.choice == LauncherStart::RunHere && d.installedBuilt == 0 && d.why.find(L"isn't a readable program") != wstring::npos,
                        "an unreadable installed copy (same, newer or older version in its entry): this copy runs");
            }
            d = DecideLauncherStart(self, L"0.6.0", newer, L"0.5.5");
            require(d.choice == LauncherStart::OfferReplace && !d.sameVersion, "an older version installed (even a newer file): this copy offers to replace it");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.4.5");
            require(d.choice == LauncherStart::OfferReplace && !d.sameVersion && d.why.find(L"older version") != wstring::npos,
                    "0.4.5 installed: a newer running copy never defers to it (offers to replace it)");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.6.1");
            require(d.choice == LauncherStart::OpenInstalled && !d.sameVersion, "a newer version installed (even an older file): the installed copy opens");
            require(BuildTimeText(0) == L"unknown" && BuildTimeText(nov5).size() == 20 && BuildTimeText(nov5) == L"2026-11-05 23:06 UTC", "BuildTimeText: \"YYYY-MM-DD hh:mm UTC\"");
        }

        // #243: the store access (util.h LibraryAccess) and the no-scan of each operation
        {
            const wstring nsDir = Join(dir, L"noscan");
            EnsureDir(nsDir);
            Ui nui;
            nui.step = [](int, int, const wstring&) {};
            nui.log = [](const wstring&) {};
            auto failingOp = [&](bool noScan) {  // an operation that fails at once (no record): only what RunOp does around it matters
                OpRequest r;
                r.op = Op::Uninstall;
                r.dir = Join(nsDir, L"no-such-game");
                r.noScan = noScan;
                return r;
            };
            // the no-scan is per operation: a job's noscan holds for that job only, the process's for all
            Context pc;
            pc.logFile = Join(nsDir, L"per-op.log");
            require(RunOp(pc, failingOp(true), nui).code != 0 && !pc.noLibraryScan, "a noscan operation: the flag goes back when it ends (never sticks into the next job)");
            RunOp(pc, failingOp(false), nui);
            require(!pc.noLibraryScan, "an operation without noscan: no search is held back");
            Context ac;
            ac.noScanAlways = true;
            ac.noLibraryScan = true;
            RunOp(ac, failingOp(false), nui);
            require(ac.noLibraryScan, "--noscan for the process: every operation, also one whose own noscan is off, searches nothing");
            pc.noLibraryScan = true;  // (as RunOp sets it for a noscan operation)
            const CompareDll none = RecordCompareDll(pc, nui, L"");
            string plog;
            ReadAll(pc.logFile, plog);
            require(none.path.empty() && plog.find("AMD FSR 4 search: libraries not searched") != string::npos && plog.find("candidate") == string::npos,
                    "noscan: the AMD FSR 4 search leaves the store libraries alone (said in the log)");
            pc.noLibraryScan = false;
#ifdef WRONGGPU_TESTING
            // test builds: WRONGGPU_TEST_LIBRARIES (a NON-EMPTY test library here) is the only library list; every store read is
            // asked of LibraryAccess first and logged to WRONGGPU_TEST_ACCESS_LOG; nothing outside the test folder is read
            const wstring libs = Join(nsDir, L"libraries"), access = Join(nsDir, L"access.log");
            auto put = [](const wstring& p, const string& t) { EnsureDir(Parent(p)); WriteFileAtomic(p, t.data(), t.size()); };
            put(Join(libs, L"Library Game\\amd_fidelityfx_upscaler_dx12.dll"), "MZ not AMD's DLL");
            SetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", libs.c_str());
            SetEnvironmentVariableW(L"WRONGGPU_TEST_ACCESS_LOG", access.c_str());
            auto accessLog = [&] { string t; ReadAll(access, t); DeleteFileW(access.c_str()); return t; };
            const string outsideRoot = "Q:\\NotTheTestFolder";
            const auto found = ScanLibraries();
            string al = accessLog();
            require(found.size() == 1 && found[0].source == L"Test" && TestLibraries() && al.find("library access: " + ToUtf8(libs)) != string::npos &&
                        al.find("registry") == string::npos,
                    "ScanLibraries: the test library's games only, logged; no store registry read");
            require(!LibraryRegistryAccess(L"GOG.com Games") && accessLog().find("library access refused: registry GOG.com Games") != string::npos,
                    "GOG (and Steam's registry, the art cache's Steam path): a store's registry key is refused in a test build, logged");
            // Steam: an app manifest inside the test folder is read; one of the PC's is not
            put(Join(libs, L"steamapps\\appmanifest_123.acf"), "\"AppState\"\r\n{\r\n\t\"appid\"\t\t\"123\"\r\n\t\"name\"\t\t\"Test Game\"\r\n\t\"installdir\"\t\t\"G\"\r\n}\r\n");
            put(Join(libs, L"steamapps\\common\\G\\Game.exe"), "MZ");
            require(SteamAppId(Join(libs, L"steamapps\\common\\G\\Game.exe")) == L"123" && GameDisplayName(Join(libs, L"steamapps\\common\\G\\Game.exe")) == L"Test Game" &&
                        accessLog().find("library access: " + ToUtf8(Join(libs, L"steamapps"))) != string::npos,
                    "Steam (SteamAppId, GameDisplayName): an app manifest in the test folder is read, logged");
            require(SteamAppId(FromUtf8(outsideRoot) + L"\\steamapps\\common\\G\\Game.exe").empty() &&
                        accessLog().find("library access refused: " + outsideRoot + "\\steamapps") != string::npos,
                    "Steam: an app manifest folder outside the test folder is refused, logged, not read");
            // Epic: the manifests of a ProgramData in the test folder are read; the PC's are not
            wchar_t pdWas[MAX_PATH] = {};
            GetEnvironmentVariableW(L"ProgramData", pdWas, MAX_PATH);
            const wstring pd = Join(nsDir, L"programdata");
            {
                string loc;  // (JSON: the path's backslashes escaped, as Epic writes them)
                for (char c : ToUtf8(Join(libs, L"E"))) { loc += c; if (c == '\\') loc += '\\'; }
                put(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests\\t.item"), "{\"InstallLocation\": \"" + loc + "\", \"AppName\": \"TestApp\"}");
            }
            SetEnvironmentVariableW(L"ProgramData", pd.c_str());
            const wstring epic = EpicAppName(Join(libs, L"E\\Game.exe"));
            const string eal = accessLog();
            SetEnvironmentVariableW(L"ProgramData", FromUtf8(outsideRoot).c_str());
            const wstring epicOut = EpicAppName(Join(libs, L"E\\Game.exe"));
            const string eal2 = accessLog();
            SetEnvironmentVariableW(L"ProgramData", pdWas[0] ? pdWas : nullptr);
            require(epic == L"TestApp" && eal.find("library access: " + ToUtf8(pd)) != string::npos, "Epic (EpicAppName): the manifests of a ProgramData in the test folder are read, logged");
            require(epicOut.empty() && eal2.find("library access refused: " + outsideRoot) != string::npos, "Epic: the manifests outside the test folder are refused, logged, not read");
            // the walk of a library game's folder (FindFilesNamed): inside yes, outside no; \\?\ and UNC forms as such
            require(FindFilesNamed(Join(libs, L"Library Game"), L"amd_fidelityfx_upscaler_dx12.dll").size() == 1 &&
                        FindFilesNamed(FromUtf8(outsideRoot), L"amd_fidelityfx_upscaler_dx12.dll").empty() && !LibraryAccess(L"\\\\server\\share\\games") &&
                        !LibraryAccess(L"\\\\?\\UNC\\server\\share\\games") && LibraryAccess(L"\\\\?\\" + Join(libs, L"Library Game")),
                    "FindFilesNamed and LibraryAccess: inside the test folder walked (also as \\\\?\\), outside and UNC refused");
            const string wal = accessLog();
            require(wal.find("library access refused: " + outsideRoot) != string::npos && wal.find("library access refused: \\\\server\\share\\games") != string::npos,
                    "the refused walks are logged");
            // what a path really is (recheck P1a): a junction above it, a file symlink, a hard link to an outside file, and a test
            // folder that is itself a junction are all refused, however inside their spelling looks
            {
                const wstring outside = Join(dir, L"outside-target");  // (outside the test run's folder nsDir)
                put(Join(outside, L"amd_fidelityfx_upscaler_dx12.dll"), "MZ outside");
                const wstring junction = Join(libs, L"Junction Game");
                _wsystem((L"cmd /d /c mklink /J \"" + junction + L"\" \"" + outside + L"\" >nul 2>nul").c_str());
                require(IsDir(junction) && !LibraryAccess(Join(junction, L"amd_fidelityfx_upscaler_dx12.dll")) &&
                            FindFilesNamed(junction, L"amd_fidelityfx_upscaler_dx12.dll").empty() &&
                            accessLog().find("library access refused: " + ToUtf8(junction)) != string::npos,
                        "a junction in the library (an ancestor of the file): refused, not walked, logged");
                const wstring symDir = Join(libs, L"Symlink Game"), sym = Join(symDir, L"amd_fidelityfx_upscaler_dx12.dll");
                EnsureDir(symDir);
                if (CreateSymbolicLinkW(sym.c_str(), Join(outside, L"amd_fidelityfx_upscaler_dx12.dll").c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
                    require(!LibraryAccess(sym) && FindFilesNamed(symDir, L"amd_fidelityfx_upscaler_dx12.dll").empty() &&
                                accessLog().find("library access refused: " + ToUtf8(sym)) != string::npos,
                            "a file symlink in the library to an outside file: refused, not returned by the walk, logged");
                    DeleteFileW(sym.c_str());
                } else {
                    std::printf("  skipped: the file symlink case (this account cannot create symbolic links: Windows developer mode off; the "
                                "junction and hard-link cases cover the same check)\n");
                }
                const wstring hardDir = Join(libs, L"Hardlink Game"), hard = Join(hardDir, L"amd_fidelityfx_upscaler_dx12.dll");
                EnsureDir(hardDir);
                require(CreateHardLinkW(hard.c_str(), Join(outside, L"amd_fidelityfx_upscaler_dx12.dll").c_str(), nullptr) && !LibraryAccess(hard) &&
                            FindFilesNamed(hardDir, L"amd_fidelityfx_upscaler_dx12.dll").empty(),
                        "a hard link in the library to an outside file (a second name): refused, not returned by the walk");
                DeleteFileW(hard.c_str());
                RemoveDirectoryW(junction.c_str());  // (the junction only: never what it points to)
                accessLog();
                // the test folder itself a junction: nothing is read
                const wstring realRoot = Join(dir, L"real-root"), linkRoot = Join(nsDir, L"root-junction");
                EnsureDir(Join(realRoot, L"libraries\\G"));
                _wsystem((L"cmd /d /c mklink /J \"" + linkRoot + L"\" \"" + realRoot + L"\" >nul 2>nul").c_str());
                SetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", Join(linkRoot, L"libraries").c_str());
                require(IsDir(linkRoot) && !LibraryAccess(Join(linkRoot, L"libraries\\G")) && ScanLibraries().empty(),
                        "a test folder that resolves elsewhere (itself a junction): every read refused");
                SetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", libs.c_str());
                RemoveDirectoryW(linkRoot.c_str());
                accessLog();
            }
            // the access log (recheck P1b): a read whose line cannot be written is refused; a new log starts with this run's id
            {
                SetEnvironmentVariableW(L"WRONGGPU_TEST_ACCESS_LOG", libs.c_str());  // a folder: no line can be appended
                const bool unwritable = LibraryAccess(Join(libs, L"Library Game"));
                SetEnvironmentVariableW(L"WRONGGPU_TEST_ACCESS_LOG", nullptr);
                const bool unset = LibraryAccess(Join(libs, L"Library Game"));
                SetEnvironmentVariableW(L"WRONGGPU_TEST_ACCESS_LOG", access.c_str());
                require(!unwritable && !unset, "an access log that cannot be written (or none named): the read is refused (fail closed)");
                SetEnvironmentVariableW(L"WRONGGPU_TEST_RUN_ID", L"unit-run-7");
                DeleteFileW(access.c_str());
                TestAccessStart(L"unit");
                const string started = accessLog();
                require(started.rfind("run unit-run-7\r\nprocess ", 0) == 0 && started.find(" unit start") != string::npos,
                        "a new access log starts with \"run <id>\", then the program's start line");
                SetEnvironmentVariableW(L"WRONGGPU_TEST_RUN_ID", nullptr);
            }
            // a normal operation after a noscan one finds its compare DLL in the test library (the search runs again)
            RunOp(pc, failingOp(true), nui);
            DeleteFileW(pc.logFile.c_str());
            RecordCompareDll(pc, nui, L"");
            ReadAll(pc.logFile, plog);
            require(plog.find("AMD FSR 4 search: " + ToUtf8(Join(libs, L"Library Game\\amd_fidelityfx_upscaler_dx12.dll"))) != string::npos &&
                        accessLog().find("library access: " + ToUtf8(Join(libs, L"Library Game"))) != string::npos,
                    "after a noscan job, a normal one's AMD FSR 4 search finds the test library's DLL (here: not AMD's, so not used)");
            SetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", L"");
            require(ScanLibraries().empty() && TestLibraries(), "WRONGGPU_TEST_LIBRARIES set but empty: no library at all");
            SetEnvironmentVariableW(L"WRONGGPU_TEST_LIBRARIES", nullptr);
            SetEnvironmentVariableW(L"WRONGGPU_TEST_ACCESS_LOG", nullptr);
            require(!TestLibraries(), "WRONGGPU_TEST_LIBRARIES not set: TestLibraries false (the normal search, never run here)");
#else
            require(!TestLibraries(), "a build without WRONGGPU_TESTING never takes a test library list");
#endif
        }

        RemoveTree(dir);
        std::printf("launcher-core: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::printf("launcher-core FAILED after %d checks: %s\n", checks, e.what());
        return 1;
    } catch (const SetupError& e) {
        std::printf("launcher-core FAILED after %d checks: %ls\n", checks, e.msg.c_str());
        return 1;
    }
}
