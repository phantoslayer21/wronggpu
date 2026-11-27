// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU launcher core: frame generation's per-game settings and its library (fg.h). Written for this project.
#include "fg.h"

#include "context.h"
#include "detect.h"
#include "pe.h"
#include "profile.h"
#include "record.h"
#include "settings.h"

namespace wg {

namespace {

struct FgKeyDef {
    const char* key;
    const char* values;  // '|' separated, the exact spellings
    const char* def;
};
constexpr FgKeyDef kFgKeyDefs[] = {
    {kFgKeyEnable, "off|on", "off"},
    {kFgKeyPreset, "A|B|C", "A"},
    {kFgKeyMultiplier, "auto|2|3|4|5|6", "auto"},
    {kFgKeyPresenter, "auto|hwfq|dxgi_tearing|dxgi_vsync", "auto"},
};

const FgKeyDef* FindFgKey(const string& key) {
    for (const FgKeyDef& d : kFgKeyDefs)
        if (key == d.key) return &d;
    return nullptr;
}

string TrimSpaces(const string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

std::vector<string> SplitBars(const string& s) {
    std::vector<string> r;
    for (size_t a = 0; a <= s.size();) {
        size_t b = s.find('|', a);
        if (b == string::npos) b = s.size();
        r.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return r;
}

// The leading number of `s` from `at` (digits only); false when there is none or it is too long to be a version part.
bool VersionPart(const wstring& s, size_t& at, unsigned& out) {
    size_t n = 0;
    unsigned v = 0;
    while (at < s.size() && s[at] >= L'0' && s[at] <= L'9' && n < 9) {
        v = v * 10 + unsigned(s[at] - L'0');
        ++at;
        ++n;
    }
    out = v;
    return n > 0 && n < 9;
}

}  // namespace

bool IsFgKey(const string& key) { return FindFgKey(key) != nullptr; }
const char* FgValues(const string& key) { const FgKeyDef* d = FindFgKey(key); return d ? d->values : nullptr; }
const char* FgDefault(const string& key) { const FgKeyDef* d = FindFgKey(key); return d ? d->def : nullptr; }

bool ValidateFgValue(const string& key, const string& value, string* normal, wstring* why) {
    const FgKeyDef* d = FindFgKey(key);
    if (!d) {
        if (why) *why = L"\"" + FromUtf8(key) + L"\" is not a frame generation setting";
        return false;
    }
    const string v = TrimSpaces(value);
    const std::vector<string> allowed = SplitBars(d->values);
    for (const string& a : allowed)
        if (v == a) {
            if (normal) *normal = v;
            return true;
        }
    if (why) {
        wstring list;
        for (const string& a : allowed) list += (list.empty() ? L"" : L", ") + FromUtf8(a);
        *why = L"\"" + FromUtf8(value) + L"\" is not a value of " + FromUtf8(key) + L" (allowed, exactly as written: " + list + L")";
    }
    return false;
}

FgState FgStateOf(const std::vector<std::pair<string, string>>& kv) {
    FgState s;
    for (const FgKeyDef& d : kFgKeyDefs) s.v[d.key] = d.def;
    std::set<string> seen;
    for (const auto& p : kv) {
        if (!IsFgKey(p.first) || !seen.insert(p.first).second) continue;  // a key's first line counts
        s.present = true;
        string n;
        wstring why;
        if (ValidateFgValue(p.first, p.second, &n, &why)) s.v[p.first] = n;
        else s.issues.push_back(L"settings.ini: " + why + L"; the default (" + FromUtf8(FgDefault(p.first)) + L") is used");
    }
    return s;
}

bool FgOptedIn(const Profile& p) {
    for (const FgKeyDef& d : kFgKeyDefs) {
        const auto it = p.v.find(d.key);
        if (it != p.v.end() && it->second != d.def) return true;
    }
    return false;
}

std::vector<std::pair<string, string>> FgStateKeys(const Profile& p, const std::vector<std::pair<string, string>>& now, bool dataReady) {
    std::vector<std::pair<string, string>> r;
    if (!FgOptedIn(p) && !FgStateOf(now).present) return r;  // never opted in: the file stays as it is
    for (const FgKeyDef& d : kFgKeyDefs) {
        const auto it = p.v.find(d.key);
        string v = it == p.v.end() ? string(d.def) : it->second;
        string n;
        if (!ValidateFgValue(d.key, v, &n, nullptr)) n = d.def;  // (a profile holds valid values only; never a guess)
        if (!strcmp(d.key, kFgKeyEnable) && !dataReady) n = "off";
        r.push_back({d.key, n});
    }
    return r;
}

bool FgVersionSupported(const wstring& version, wstring* why) {
    size_t at = 0;
    unsigned major = 0, minor = 0;
    if (!VersionPart(version, at, major)) return true;  // no version to go by: the converter decides from the bytes
    if (at < version.size() && version[at] == L'.') {
        ++at;
        if (!VersionPart(version, at, minor)) minor = 0;
    }
    if (major > kFgMinMajor || (major == kFgMinMajor && minor >= kFgMinMinor)) return true;
    if (why)
        *why = L"version " + version + L" has no frame-generation model (version " + std::to_wstring(kFgMinMajor) + L"." +
               std::to_wstring(kFgMinMinor) + L" or newer is needed)";
    return false;
}

FgLibrary CheckFgLibrary(const wstring& path) {
    FgLibrary l;
    l.path = path;
    if (!IsFile(path)) {
        l.why = L"not found";
        return l;
    }
    l.version = FileVersionInfo(path).fileVersion;  // (empty when the file has no version resource)
    wstring why;
    if (!FgVersionSupported(l.version, &why)) {
        l.verdict = FgLibrary::TOO_OLD;
        l.why = why;
        return l;
    }
    l.verdict = FgLibrary::CANDIDATE;
    return l;
}

std::vector<FgLibrary> GameFgLibraries(const wstring& gameDir) {
    std::vector<FgLibrary> usable, other;
    if (gameDir.empty() || !IsDir(gameDir)) return usable;
    for (const wstring& f : FindFilesNamed(gameDir, kFgLibraryName)) {
        FgLibrary l = CheckFgLibrary(f);
        (l.usable() ? usable : other).push_back(l);
    }
    usable.insert(usable.end(), other.begin(), other.end());
    return usable;
}

FgEnable CheckFgEnable(const wstring& gameDir, bool dataReady, wstring* library, wstring* why) {
    if (library) library->clear();
    std::vector<FgLibrary> libs = GameFgLibraries(gameDir);
    if (libs.empty()) {
        const wstring root = GameRoot(gameDir);
        if (!root.empty() && !IEq(root, gameDir)) libs = GameFgLibraries(root);
    }
    if (libs.empty()) {
        if (why)
            *why = L"This game has no frame generation library (nvngx_dlssg.dll): it does not offer DLSS Frame Generation, so WrongGPU's "
                   L"frame generation can't run in it. Frame generation stays off.";
        return FgEnable::NO_LIBRARY;
    }
    if (dataReady) return FgEnable::READY;
    if (!libs.front().usable()) {
        const FgLibrary& l = libs.front();
        if (why)
            *why = L"This game's frame generation library (" + l.path + L") is " +
                   (l.version.empty() ? wstring(L"too old") : L"version " + l.version) +
                   L", which has no frame-generation model: WrongGPU's frame generation is converted from version 310.1 or newer. "
                   L"Frame generation stays off. Convert its data from a game with a newer library first: WrongGPU.exe model convert "
                   L"--preset=FG --model=<that game's nvngx_dlssg.dll>.";
        return FgEnable::TOO_OLD;
    }
    if (library) *library = libs.front().path;
    return FgEnable::CONVERT;
}

bool RecordHasFg(const Record& m) {
    for (auto& e : m.e) if (e.action == L"copy" && IsFgPlugin(e.rel)) return true;
    return false;
}

bool FgPayloadAvailable(Context& ctx) {
    if (!ctx.data || !ctx.data->fgFiles) return false;
    for (size_t i = 0; i < ctx.data->fgCount; ++i)
        if (IEq(ctx.data->fgFiles[i].rel, kFgPayloadPlugin)) return true;
    return false;
}

wstring FgInstallRel(const wstring& pluginRel, const wstring& payloadRel) {
    const wstring folder = Parent(pluginRel);
    if (IEq(payloadRel, kFgPayloadPlugin)) return folder.empty() ? wstring(kFgPluginName) : Join(folder, kFgPluginName);
    const wstring dxilDir = Parent(payloadRel), name = BaseName(payloadRel);
    if (!IEq(dxilDir, kFgPayloadDxilDir) || !EndsWithI(name, L".dxil") || name.size() <= 5) return L"";
    return Join(folder.empty() ? wstring(kFgDxilDir) : Join(folder, kFgDxilDir), name);
}

wstring OptiFgConflictText(const string& optiIni) {
    wstring set;
    for (const char* key : {"FGInput", "FGOutput", "FGType", "Enabled"}) {
        string value;
        if (!IniHasKey(optiIni, "FrameGen", key, &value)) continue;
        value = TrimSpaces(value);
        string low = value;
        for (char& c : low) c = char(tolower((unsigned char)c));
        const bool off = !strcmp(key, "Enabled") ? low != "true" : (low.empty() || low == "auto" || low == "nofg");
        if (!off) set += (set.empty() ? L"" : L", ") + FromUtf8(key) + L"=" + FromUtf8(value);
    }
    if (set.empty()) return L"";
    return L"OptiScaler's own frame generation is set in this game (OptiScaler.ini [FrameGen] " + set + L"). It would take the same frames as "
           L"WrongGPU's frame generation: turn it off in OptiScaler's menu (Frame Generation) so WrongGPU's runs alone.";
}

wstring OptiFgConflict(const wstring& dir) {
    string ini;
    if (dir.empty() || !ReadAll(Join(dir, L"OptiScaler.ini"), ini)) return L"";
    return OptiFgConflictText(ini);
}

bool GameHasFgLibrary(const wstring& exeDir, const wstring& dir, const wstring& root) {
    for (const wstring& d : {exeDir, dir, root})
        if (!d.empty() && !GameFgLibraries(d).empty()) return true;
    return false;
}

}  // namespace wg
