// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the per-game settings profile (profile.h). Written for this project.
#include "profile.h"

#include <cmath>
#include <cstdlib>

#include "record.h"
#include "settings.h"

namespace wg {

// The Basic tier (build/launcher-design.md section 4). Defaults are the front end's and the runtime's own: a default
// value is never written into the game's files.
const SettingDef kSettings[] = {
    {"wronggpu", Layer::Fe, Tier::Basic, "on", "on|off", L"WrongGPU",
     L"On: Model M upscales. Off: AMD's FSR upscales (WrongGPU stays installed and only logs)."},
    {"executor", Layer::Rt, Tier::Basic, "auto", "auto|dx12|hip", L"Executor",
     L"Auto: D3D12 on RX 9000 cards. HIP is used anyway for sparkle guard and frame capture."},
    {"output_filter", Layer::Rt, Tier::Basic, "default", "default|original|analytic|quality5|quality9", L"Output filter",
     L"Default and Original: native output. Quality 5 / Quality 9: WrongGPU variations."},
    {"pre_exposure", Layer::Rt, Tier::Basic, "scale", "scale|original", L"Pre-exposure",
     L"Scale (default). Original can change hair and fine detail in some games (Hogwarts Legacy)."},
    {"sparkle_guard", Layer::Rt, Tier::Basic, "off", "off|on|#rate", L"Sparkle guard",
     L"Calms single bright pixels (about +0.11 ms). Needs the HIP runtime."},
    {"fps_overlay", Layer::Opti, Tier::Basic, "keep", "keep|on|off", L"OptiScaler FPS overlay",
     L"OptiScaler's own FPS counter (only for OptiScaler builds that have the setting)."},
    // Ray Reconstruction (games that use it, rr.h): the RR plugin's settings file
    {"rr_enabled", Layer::Rr, Tier::Basic, "on", "on|off", L"Ray Reconstruction",
     L"On: WrongGPU denoises and upscales the ray-traced image when the game asks for Ray Reconstruction. Off: the game's own denoiser."},
    // rc5 (owner 2026-11-05): Fast is the default; a profile that has rr_mode keeps it (every saved profile writes it)
    {"rr_mode", Layer::Rr, Tier::Basic, "fast", "accurate|fast|exact", L"Ray Reconstruction quality",
     L"Fast (default) is the fastest. Accurate looks the same as Exact: higher quality, slower. Exact is the most precise and the slowest."},
    // A game's experimental opt-in (optiscaler.h kOptiGameKeys; only for its game, `set` only, not on the settings page):
    // Marvel's Spider-Man 2's Ray Reconstruction (OptiScaler's GPU-name spoof). Off: Model M only.
    {"sm2_rr_experimental", Layer::Opti, Tier::Advanced, "off", "on|off", L"Spider-Man 2 Ray Reconstruction (experimental)",
     L"Experimental, not recommended: the game often freezes or crashes with it. Off: Model M only."},
    // Borderlands 4 and Hogwarts Legacy (rrconfig.h; only for those games, `set` and the launcher's question, not on the
    // settings page): on, WrongGPU adds the engine setting their Ray Reconstruction needs to the game's Engine.ini; ask
    // (not decided yet: the launcher asks after the install) and off leave the file alone.
    {"rr_config_fix", Layer::Rr, Tier::Advanced, "ask", "ask|on|off", L"Ray Reconstruction engine setting",
     L"On: WrongGPU adds the one Engine.ini line this game's Ray Reconstruction needs (and takes it out again on uninstall)."},
#if WRONGGPU_MODEL_L
    // Model L (#89; only in a release whose front end and runtime run it): the model the game starts with. L needs Model L's
    // data on this PC (model\L\model.wgcm, checked against Model L's converter data); M changes nothing in the game's files.
    {"model", Layer::Model, Tier::Basic, "m", "m|l", L"Model",
     L"M: Model M (default). L: Model L, converted from a DLSS file of version 310.5 or newer that has it."},
#endif
};
const size_t kSettingCount = sizeof(kSettings) / sizeof(kSettings[0]);

const SettingDef* FindSetting(const string& key) {
    for (auto& d : kSettings) if (key == d.key) return &d;
    return nullptr;
}

static string TrimA(string t) {
    while (!t.empty() && (t.back() == '\r' || t.back() == '\n' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
    size_t i = 0;
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
    return t.substr(i);
}
static string LowerA(string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; }

bool ValidateSetting(const SettingDef& d, const string& valueIn, string* normal, wstring* why) {
    string v = LowerA(TrimA(valueIn));
    if (!strcmp(d.key, "executor") && (v == "d3d12" || v.empty())) v = v.empty() ? "auto" : "dx12";
    if (!strcmp(d.key, "wronggpu")) {
        if (v == "1" || v == "true" || v == "network") v = "on";
        else if (v == "0" || v == "false" || v == "amd") v = "off";
    }
    if (!strcmp(d.key, "sparkle_guard") && (v == "0" || v == "false" || v.empty())) v = "off";
    if (!strcmp(d.key, "rr_enabled") || !strcmp(d.key, "sm2_rr_experimental") || !strcmp(d.key, "rr_config_fix")) {
        if (v == "1" || v == "true") v = "on";
        else if (v == "0" || v == "false") v = "off";
    }
    if (!strcmp(d.key, "fps_overlay")) {
        if (v == "true" || v == "1") v = "on";
        else if (v == "false" || v == "0") v = "off";
        else if (v == "auto" || v.empty()) v = "keep";
    }
    const string allowed = d.values;
    bool rate = false;
    for (size_t a = 0; a <= allowed.size();) {
        size_t b = allowed.find('|', a);
        if (b == string::npos) b = allowed.size();
        const string item = allowed.substr(a, b - a);
        if (item == v) { if (normal) *normal = v; return true; }
        if (item == "#rate") rate = true;
        a = b + 1;
    }
    if (rate && !v.empty()) {
        char* end = nullptr;
        const double r = std::strtod(v.c_str(), &end);
        if (end && *end == 0 && std::isfinite(r) && r > 0.0 && r < 1.0) {
            char buf[32]; snprintf(buf, sizeof buf, "%g", r);
            if (normal) *normal = buf;
            return true;
        }
    }
    if (why) {
        wstring list;
        for (size_t a = 0; a <= allowed.size();) {
            size_t b = allowed.find('|', a);
            if (b == string::npos) b = allowed.size();
            string item = allowed.substr(a, b - a);
            if (item == "#rate") item = "a number between 0 and 1";
            list += (list.empty() ? L"" : L", ") + FromUtf8(item);
            a = b + 1;
        }
        *why = L"\"" + FromUtf8(valueIn) + L"\" is not a value of " + FromUtf8(d.key) + L" (allowed: " + list + L")";
    }
    return false;
}

string Profile::get(const string& key) const {
    auto it = v.find(key);
    if (it != v.end()) return it->second;
    const SettingDef* d = FindSetting(key);
    return d ? d->def : "";
}
bool Profile::isDefault(const string& key) const { const SettingDef* d = FindSetting(key); return d && get(key) == d->def; }

Profile DefaultProfile() {
    Profile p;
    for (auto& d : kSettings) p.v[d.key] = d.def;
    return p;
}

wstring ProfilePath(const wstring& installDir) { return Join(StateDir(installDir), L"profile.ini"); }

bool LoadProfile(const wstring& path, Profile& p, std::vector<wstring>* problems) {
    p = DefaultProfile();
    string raw;
    if (!ReadAll(path, raw)) return false;
    for (size_t a = 0; a < raw.size();) {
        size_t e = raw.find('\n', a);
        if (e == string::npos) e = raw.size();
        const string t = TrimA(raw.substr(a, e - a));
        a = e + 1;
        if (t.empty() || t[0] == ';' || t[0] == '#' || t[0] == '[') continue;
        const size_t eq = t.find('=');
        if (eq == string::npos) continue;
        const string k = LowerA(TrimA(t.substr(0, eq))), val = TrimA(t.substr(eq + 1));
        if (k == "format") continue;
        const SettingDef* d = FindSetting(k);
        if (!d) { p.v[k] = val; continue; }  // a key of a later launcher: kept
        string n; wstring why;
        if (ValidateSetting(*d, val, &n, &why)) p.v[k] = n;
        else if (problems) problems->push_back(L"profile.ini: " + why + L"; the default is used");
    }
    return true;
}

string ProfileText(const Profile& p) {
    string t = "; WrongGPU launcher: the settings of this game. The launcher writes wronggpu.ini (and OptiScaler.ini)\r\n"
               "; from this file; change them in the launcher or with: WrongGPU.exe set <game> key=value\r\n"
               "[profile]\r\nformat=1\r\n";
    for (auto& d : kSettings) {
        if (d.layer == Layer::Model && p.isDefault(d.key)) continue;  // Model M: profile.ini as before the model choice
        t += string(d.key) + "=" + p.get(d.key) + "\r\n";
    }
    for (auto& kv : p.v) if (!FindSetting(kv.first)) t += kv.first + "=" + kv.second + "\r\n";
    return t;
}
void SaveProfile(const wstring& path, const Profile& p) {
    const string t = ProfileText(p);
    EnsureDir(Parent(path));
    WriteFileAtomic(path, t.data(), t.size());
}

bool SetSetting(Profile& p, const string& keyIn, const string& value, wstring* why) {
    const string key = LowerA(TrimA(keyIn));
    const SettingDef* d = FindSetting(key);
    if (!d && key == "model") { if (why) *why = kNoModelChoice; return false; }  // a build without the Model L choice
    if (!d) { if (why) *why = L"unknown setting \"" + FromUtf8(keyIn) + L"\""; return false; }
    string n;
    if (!ValidateSetting(*d, value, &n, why)) return false;
    p.v[key] = n;
    return true;
}

// ---- runtime_options ----------------------------------------------------------------------------------------------
std::vector<std::pair<string, string>> SplitOptions(const string& options) {
    std::vector<std::pair<string, string>> items;
    for (size_t a = 0; a <= options.size();) {
        size_t b = options.find(';', a);
        if (b == string::npos) b = options.size();
        const string item = TrimA(options.substr(a, b - a));
        a = b + 1;
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        if (eq == string::npos) items.push_back({TrimA(item), string()});
        else items.push_back({TrimA(item.substr(0, eq)), TrimA(item.substr(eq + 1))});
        if (items.back().first.empty()) items.pop_back();
    }
    return items;
}
string JoinOptions(const std::vector<std::pair<string, string>>& items) {
    string s;
    for (auto& kv : items) s += (s.empty() ? "" : ";") + kv.first + (kv.second.empty() ? "" : "=" + kv.second);
    return s;
}

// The profile's value of a runtime key as the runtime spells it (empty: not written).
static string RuntimeValue(const string& key, const string& v) {
    if (key == "sparkle_guard") return v == "off" ? "" : v;
    return v;
}
static string ProfileFromRuntime(const string& key, const string& raw) {
    const SettingDef* d = FindSetting(key);
    string n;
    if (d && ValidateSetting(*d, raw, &n, nullptr)) return n;
    return raw;  // not valid: shown as it is
}

// Lines of a front-end INI text in order: (section, key, value) for the key=value lines the front end reads.
struct FeLine { string section, key, value; };
static std::vector<FeLine> FrontEndLines(const string& text) {
    std::vector<FeLine> r;
    string sec;
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) e = text.size();
        string t = TrimA(text.substr(a, e - a));
        a = e + 1;
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        if (t[0] == '[') { const string l = LowerA(t); sec = l == "[wronggpu]" ? "wronggpu" : l == "[fsr]" ? "fsr" : "other"; continue; }
        if (sec == "other") continue;
        const size_t eq = t.find('=');
        if (eq == string::npos) continue;
        string v = t.substr(eq + 1);
        for (size_t i = 1; i < v.size(); ++i)
            if ((v[i] == '#' || v[i] == ';') && (v[i - 1] == ' ' || v[i - 1] == '\t')) { v = v.substr(0, i); break; }
        r.push_back({sec.empty() ? "fsr" : sec, LowerA(TrimA(t.substr(0, eq))), TrimA(v)});
    }
    return r;
}

std::map<string, string> EffectiveSettings(const string& wronggpuIni, const string& optiIni) {
    std::map<string, string> eff;
    for (auto& d : kSettings) eff[d.key] = d.def;
    string options, loose;
    bool haveUpscaler = false;
    string upscaler;
    for (auto& l : FrontEndLines(wronggpuIni)) {
        if (l.key == "upscaler") { upscaler = LowerA(l.value); haveUpscaler = true; }
        else if (l.key == "runtime_options") options = l.value;
        else if (const SettingDef* d = FindSetting(l.key); d && d->layer == Layer::Rt) loose += (loose.empty() ? "" : ";") + l.key + "=" + l.value;
    }
    if (haveUpscaler) eff["wronggpu"] = upscaler == "amd" ? "off" : upscaler == "network" || upscaler == "standin" ? "on" : upscaler;
    for (auto& kv : SplitOptions(options + ";" + loose)) {
        string k = LowerA(kv.first);
        if (k == "backend") k = "executor";
        const SettingDef* d = FindSetting(k);
        if (!d || d->layer != Layer::Rt) continue;
        eff[k] = ProfileFromRuntime(k, kv.second);
    }
    if (FindSetting("model")) {  // the last model item of runtime_options (the one the front end uses): Model L or not
        string model;
        for (auto& kv : SplitOptions(options)) if (LowerA(kv.first) == "model") model = LowerA(kv.second);
        eff["model"] = model == "l" ? "l" : "m";
    }
    string fps;
    if (IniHasKey(optiIni, "Menu", "ShowFps", &fps)) {
        const string l = LowerA(fps);
        eff["fps_overlay"] = l == "true" ? "on" : l == "false" ? "off" : "keep";
    }
    string dxgi, devid;  // sm2_rr_experimental: its keys' values in the file (optiscaler.h kOptiGameKeys)
    if (IniHasKey(optiIni, "Spoofing", "Dxgi", &dxgi) && IniHasKey(optiIni, "Spoofing", "SpoofedDeviceId", &devid) &&
        LowerA(dxgi) == "true" && LowerA(devid) == "0x2684")
        eff["sm2_rr_experimental"] = "on";
    return eff;
}

bool RenderWronggpuIni(const Profile& p, const string& in, string& out, wstring* why) {
    for (auto& d : kSettings) {
        if (!ValidateSetting(d, p.get(d.key), nullptr, why)) return false;
    }
    string text = in;
    // front end: upscaler
    {
        const string want = p.get("wronggpu") == "off" ? "amd" : "network";
        string cur, sec;
        const bool has = IniGetFrontEnd(text, "upscaler", &cur, &sec);
        if (!has && want != "network") text = IniSet(text, "fsr", "upscaler", want);
        else if (has && LowerA(cur) != want) text = IniSet(text, sec, "upscaler", want);
    }
    // runtime items: the managed keys out of runtime_options and off their own lines, then the non-default ones
    string options, section;
    const bool hadOptions = IniGetFrontEnd(text, "runtime_options", &options, &section);
    std::vector<std::pair<string, string>> items;
    for (auto& kv : SplitOptions(options)) {
        string k = LowerA(kv.first);
        if (k == "backend") k = "executor";
        const SettingDef* d = FindSetting(k);
        if (d && d->layer == Layer::Rt) continue;
        items.push_back(kv);
    }
    for (auto& d : kSettings) {
        if (d.layer != Layer::Rt) continue;
        text = IniRemoveFrontEnd(text, d.key);
        const string rv = RuntimeValue(d.key, p.get(d.key));
        if (!p.isDefault(d.key) && !rv.empty()) items.push_back({d.key, rv});
    }
    // the model (WRONGGPU_MODEL_L builds): L, every model item out and model=L last (the last model item wins); M, only Model
    // L's items go back to model=M (the installer's model=M, and Model K's item of a build with it, stay as they are)
    if (FindSetting("model")) {
        auto isModel = [](const std::pair<string, string>& kv) { return LowerA(kv.first) == "model"; };
        if (p.get("model") == "l") {
            std::vector<std::pair<string, string>> kept;
            for (auto& kv : items) if (!isModel(kv)) kept.push_back(kv);
            kept.push_back({"model", "L"});
            items = kept;
        } else {
            for (auto& kv : items) if (isModel(kv) && LowerA(kv.second) == "l") kv.second = "M";
        }
    }
    const string joined = JoinOptions(items);
    // the line is rewritten only when its items change (a player's spacing and comment stay otherwise)
    if (hadOptions ? joined != JoinOptions(SplitOptions(options)) : !joined.empty()) text = IniSet(text, hadOptions ? section : "fsr", "runtime_options", joined);
    out = text;
    return true;
}

bool RenderOptiIni(const Profile& p, const string& in, string& out, std::vector<wstring>* notes) {
    out = in;
    // the model (WRONGGPU_MODEL_L builds): the entry of WrongGPU's list OptiScaler starts with. OptiScaler 10's custom
    // upscaler slot: [WrongGPU] UpscalerIndex; else [FSR] UpscalerIndex (the list of WrongGPU's FFX DLL). L: Model L's entry;
    // M: back to Model M's entry (0) only when the file names Model L's (another value, such as Model K's, stays).
    if (FindSetting("model")) {
        string cur;
        const char* sec = IniHasKey(out, "WrongGPU", "UpscalerIndex", &cur) ? "WrongGPU" : IniHasKey(out, "FSR", "UpscalerIndex", &cur) ? "FSR" : nullptr;
        if (sec) {
            cur = TrimA(cur);
            if (p.get("model") == "l" && cur != kModelLEntryIndex) out = IniSet(out, sec, "UpscalerIndex", kModelLEntryIndex);
            else if (p.get("model") != "l" && cur == kModelLEntryIndex) out = IniSet(out, sec, "UpscalerIndex", "0");
        } else if (p.get("model") == "l" && notes) {
            notes->push_back(L"Model L: this OptiScaler.ini has no UpscalerIndex key; choose WrongGPU Model L in OptiScaler's menu");
        }
    }
    const string fps = p.get("fps_overlay");
    if (fps == "keep") return true;
    string cur;
    if (!IniHasKey(out, "Menu", "ShowFps", &cur)) {
        if (notes) notes->push_back(L"OptiScaler FPS overlay: this OptiScaler build has no [Menu] ShowFps key; not changed");
        return true;
    }
    out = IniSet(out, "Menu", "ShowFps", fps == "on" ? "true" : "false");
    return true;
}

// Front-end keys of wronggpu.ini that are diagnostics, with the value the installers write (a different value is a
// developer's or a test's, never a launcher setting).
static const struct { const char* key; const char* normal; } kFrontEndDiagnostics[] = {
    {"log_dispatches", "0"}, {"split_test", "0"}, {"view", "upscale"}, {"d3d12_hooks", "auto"}, {"debug_gain", "4"}, {"warp_weight", "0.9"},
};

std::vector<string> IgnoredImportKeys(const string& wronggpuIni) {
    std::vector<string> r;
    string options;
    for (auto& l : FrontEndLines(wronggpuIni)) {
        if (l.key == "runtime_options") options += (options.empty() ? "" : ";") + l.value;  // every line: the front end reads the last, a player may have edited either
        for (auto& d : kFrontEndDiagnostics)
            if (l.key == d.key && LowerA(l.value) != d.normal) r.push_back(l.key + "=" + l.value);
    }
    for (auto& kv : SplitOptions(options)) {
        string k = LowerA(kv.first);
        if (k == "backend") k = "executor";
        if (k == "model") continue;  // the installer's own model choice (model=M), written on every install
        const SettingDef* d = FindSetting(k);
        if (d && d->layer == Layer::Rt) continue;  // a setting of the schema (imported, or reported as not valid)
        const string item = kv.first + (kv.second.empty() ? "" : "=" + kv.second);
        if (std::find(r.begin(), r.end(), item) == r.end()) r.push_back(item);
    }
    return r;
}

Profile ImportProfile(const string& wronggpuIni, const string& optiIni, std::vector<wstring>* notes) {
    Profile p = DefaultProfile();
    const auto eff = EffectiveSettings(wronggpuIni, optiIni);
    for (auto& d : kSettings) {
        auto it = eff.find(d.key);
        if (it == eff.end() || it->second == d.def) continue;
        if (!strcmp(d.key, "sm2_rr_experimental") || !strcmp(d.key, "rr_config_fix")) continue;  // never imported: the player chooses
        string n; wstring why;
        if (ValidateSetting(d, it->second, &n, &why)) {
            p.v[d.key] = n;
            if (notes) notes->push_back(L"kept your setting " + FromUtf8(d.key) + L" = " + FromUtf8(n));
        } else if (notes) notes->push_back(L"not imported: " + why);
    }
    string up, sec;
    if (notes && IniGetFrontEnd(wronggpuIni, "upscaler", &up, &sec) && LowerA(up) == "standin")
        notes->push_back(L"upscaler = standin (the default of WrongGPU 0.3.0) counts as WrongGPU on");
    if (notes)
        for (auto& k : IgnoredImportKeys(wronggpuIni))
            notes->push_back(wstring(kIgnoredPrefix) + FromUtf8(k) + L" (a developer or test key, not a launcher setting; not imported, wronggpu.ini keeps it as it is)");
    return p;
}

std::vector<OutsideEdit> OutsideEdits(const Profile& p, const string& wronggpuIni, const string& optiIni) {
    std::vector<OutsideEdit> r;
    const auto eff = EffectiveSettings(wronggpuIni, optiIni);
    for (auto& d : kSettings) {
        auto it = eff.find(d.key);
        if (it == eff.end()) continue;
        if (it->second == d.def || it->second == p.get(d.key)) continue;
        r.push_back({d.key, it->second, p.get(d.key)});
    }
    return r;
}

}  // namespace wg
