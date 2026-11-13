// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the OptiScaler tab and the inikey journal (optikeys.h). Written for this project.
#include "optikeys.h"

#include <cstdio>

namespace wg {

namespace {
string LowerA(string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; }
bool IEqA(const string& a, const string& b) { return LowerA(a) == LowerA(b); }
bool Space(char c) { return c == ' ' || c == '\t'; }
// One physical line: [begin, end) without its '\n'; content end excludes a trailing '\r'.
struct Line { size_t begin, end, contentEnd; };
std::vector<Line> Lines(const string& t) {
    std::vector<Line> v;
    for (size_t a = 0; a < t.size();) {
        size_t e = t.find('\n', a);
        if (e == string::npos) e = t.size();
        size_t ce = e;
        if (ce > a && t[ce - 1] == '\r') --ce;
        v.push_back({a, e, ce});
        a = e + 1;
    }
    return v;
}
// Visits every key line: (section, key, span).
template <typename F> void ForEachKey(const string& t, F f) {
    string section;
    for (const Line& l : Lines(t)) {
        size_t a = l.begin, e = l.contentEnd;
        while (a < e && Space(t[a])) ++a;
        if (a >= e || t[a] == ';' || t[a] == '#') continue;
        if (t[a] == '[') {
            const size_t close = t.find(']', a);
            if (close != string::npos && close < e) section = t.substr(a + 1, close - a - 1);
            continue;
        }
        const size_t eq = t.find('=', a);
        if (eq == string::npos || eq >= e) continue;
        size_t ke = eq;
        while (ke > a && Space(t[ke - 1])) --ke;
        IniSpan s;
        s.found = true; s.lineBegin = l.begin; s.lineEnd = l.end;
        size_t vb = eq + 1, ve = e;
        while (vb < ve && Space(t[vb])) ++vb;
        while (ve > vb && Space(t[ve - 1])) --ve;
        s.valueBegin = vb; s.valueEnd = ve;
        if (!f(section, t.substr(a, ke - a), s)) return;
    }
}
}  // namespace

IniSpan IniFind(const string& text, const string& section, const string& key) {
    IniSpan r;
    ForEachKey(text, [&](const string& sec, const string& k, const IniSpan& s) {
        if (IEqA(sec, section) && IEqA(k, key)) { r = s; return false; }
        return true;
    });
    return r;
}
string IniValueAt(const string& text, const IniSpan& s) { return s.found ? text.substr(s.valueBegin, s.valueEnd - s.valueBegin) : string(); }
string IniReplaceValue(const string& text, const IniSpan& s, const string& value) {
    if (!s.found) return text;
    return text.substr(0, s.valueBegin) + value + text.substr(s.valueEnd);
}
std::vector<IniEntry> IniEntries(const string& text) {
    std::vector<IniEntry> v;
    std::set<string> seen;
    ForEachKey(text, [&](const string& sec, const string& k, const IniSpan& s) {
        if (seen.insert(LowerA(sec) + "\n" + LowerA(k)).second) v.push_back({sec, k, IniValueAt(text, s)});
        return true;
    });
    return v;
}

// ---- the keys -------------------------------------------------------------------------------------------------------------
// Editable: the player's choices that do not touch how WrongGPU is wired in. Only keys present in the installed
// OptiScaler.ini are offered (names differ between OptiScaler builds; a name this list does not know is never written).
const OptiKeyDef kOptiEditable[] = {
    {"Menu", "ShortcutKey", OptiKind::Key, L"Menu key", L"The key that opens OptiScaler's menu in the game."},
    {"Menu", "FpsOverlayType", OptiKind::Int, L"FPS overlay style", L"Which FPS overlay OptiScaler draws when its FPS overlay is on."},
    {"Menu", "Scale", OptiKind::Number, L"Menu scale", L"The size of OptiScaler's menu."},
    {"Spoofing", "DxgiFactoryWrapping", OptiKind::Bool, L"Compatibility wrapper", L"A few games start only with it (007 First Light)."},
    {"Spoofing", "Dxgi", OptiKind::Bool, L"Report a DLSS-capable graphics card", L"Lets a game offer its DLSS option on this graphics card; needed for the game's DLSS inputs."},
    {"Inputs", "EnableDlssInputs", OptiKind::Bool, L"Use the game's DLSS inputs", L"OptiScaler handles the game's DLSS upscaling."},
    {"Inputs", "EnableXeSSInputs", OptiKind::Bool, L"Use the game's XeSS inputs", L"OptiScaler handles the game's XeSS upscaling. Some games (such as Hogwarts Legacy) give XeSS too little to work with: choose DLSS or FSR there."},
    {"Inputs", "EnableFsr2Inputs", OptiKind::Bool, L"Use the game's FSR 2 inputs", L"OptiScaler handles the game's FSR 2 upscaling."},
    {"Inputs", "EnableFsr3Inputs", OptiKind::Bool, L"Use the game's FSR 3 inputs", L"OptiScaler handles the game's FSR 3 upscaling."},
    {"Inputs", "EnableFfxInputs", OptiKind::Bool, L"Use the game's FidelityFX API inputs", L"OptiScaler handles the game's FidelityFX API upscaling."},
    {"Inputs", "UseFsr2Inputs", OptiKind::Bool, L"Use the game's FSR 2 inputs", L"OptiScaler handles the game's FSR 2 upscaling."},
    {"Inputs", "UseFsr3Inputs", OptiKind::Bool, L"Use the game's FSR 3 inputs", L"OptiScaler handles the game's FSR 3 upscaling."},
    {"Inputs", "UseFfxInputs", OptiKind::Bool, L"Use the game's FidelityFX API inputs", L"OptiScaler handles the game's FidelityFX API upscaling."},
};
const size_t kOptiEditableCount = sizeof(kOptiEditable) / sizeof(kOptiEditable[0]);
// Locked: what WrongGPU's setup writes (optiscaler.h kOptiIni) and what the Settings page owns.
const OptiLockedKey kOptiLocked[] = {
    {"Upscalers", "Dx12Upscaler", L"managed by WrongGPU: the game's upscaler calls go to WrongGPU"},
    {"Libraries", "FfxDx12SRPath", L"managed by WrongGPU: WrongGPU's DLL"},
    {"FSR", "Fsr4Update", L"managed by WrongGPU: off (OptiScaler must not patch WrongGPU's DLL)"},
    {"FSR", "UpscalerIndex", L"managed by WrongGPU: the model entry"},
    {"WrongGPU", "UpscalerIndex", L"managed by WrongGPU: the model entry"},
    {"CAS", "Enabled", L"managed by WrongGPU: no extra sharpening"},
    {"Sharpness", "OverrideSharpness", L"managed by WrongGPU: no extra sharpening"},
    {"OutputScaling", "Enabled", L"managed by WrongGPU: the game's own sizes"},
    {"UpscaleRatio", "UpscaleRatioOverrideEnabled", L"managed by WrongGPU: the game's own sizes"},
    {"Plugins", "LoadAsiPlugins", L"managed by WrongGPU: its plugin"},
    {"Plugins", "Path", L"managed by WrongGPU: its plugin's folder"},
    {"Log", "LogToFile", L"managed by WrongGPU: OptiScaler.log for support"},
    {"Log", "LogLevel", L"managed by WrongGPU: OptiScaler.log for support"},
    {"Hotfix", "MipmapBiasOverride", L"managed by WrongGPU (reserved for the mip-bias setting)"},
    {"Menu", "ShowFps", L"set on the Settings page (OptiScaler FPS overlay)"},
};
const size_t kOptiLockedCount = sizeof(kOptiLocked) / sizeof(kOptiLocked[0]);

const OptiKeyDef* FindOptiKey(const string& section, const string& key) {
    for (auto& d : kOptiEditable) if (IEqA(d.section, section) && IEqA(d.key, key)) return &d;
    return nullptr;
}
bool IsOptiLocked(const string& section, const string& key, const wchar_t** why) {
    for (auto& d : kOptiLocked) if (IEqA(d.section, section) && IEqA(d.key, key)) { if (why) *why = d.why; return true; }
    return false;
}
bool SplitOptiName(const string& name, string& section, string& key) {
    const size_t dot = name.find('.');
    if (dot == string::npos || dot == 0 || dot + 1 >= name.size()) return false;
    section = name.substr(0, dot); key = name.substr(dot + 1);
    return true;
}
bool ValidateOptiValue(const OptiKeyDef& d, const string& valueIn, string* normal, wstring* why) {
    string v = valueIn;
    while (!v.empty() && Space(v.back())) v.pop_back();
    while (!v.empty() && Space(v.front())) v.erase(0, 1);
    const string l = LowerA(v);
    auto ok = [&](const string& n) { if (normal) *normal = n; return true; };
    auto bad = [&](const wchar_t* w) { if (why) *why = FromUtf8(d.section) + L"." + FromUtf8(d.key) + L": " + w; return false; };
    if (v.empty() || v.size() > 32 || v.find_first_of("\r\n;#=[]\"") != string::npos) return bad(L"not a value");
    if (l == "auto") return ok("auto");
    switch (d.kind) {
        case OptiKind::Bool:
            if (l == "true" || l == "on" || l == "1") return ok("true");
            if (l == "false" || l == "off" || l == "0") return ok("false");
            return bad(L"use true, false or auto");
        case OptiKind::Int: {
            size_t i = l[0] == '-' ? 1 : 0;
            if (i >= l.size()) return bad(L"use a whole number or auto");
            for (size_t k = i; k < l.size(); ++k) if (l[k] < '0' || l[k] > '9') return bad(L"use a whole number or auto");
            return ok(l);
        }
        case OptiKind::Number: {
            bool dot = false, digit = false;
            for (size_t k = l[0] == '-' ? 1 : 0; k < l.size(); ++k) {
                if (l[k] == '.' && !dot) dot = true;
                else if (l[k] >= '0' && l[k] <= '9') digit = true;
                else return bad(L"use a number or auto");
            }
            return digit ? ok(l) : bad(L"use a number or auto");
        }
        case OptiKind::Key: {
            unsigned long code = 0;
            if (l.size() > 2 && l[0] == '0' && l[1] == 'x') {
                for (size_t k = 2; k < l.size(); ++k) {
                    const char c = l[k];
                    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return bad(L"use a key code (decimal, or 0x hex) or auto");
                    code = code * 16 + (c <= '9' ? c - '0' : c - 'a' + 10);
                    if (code > 0xFF) break;
                }
                if (code < 1 || code > 0xFE) return bad(L"a key code is 1 to 254 (0x01 to 0xFE)");
                char b[8]; snprintf(b, sizeof b, "0x%02X", (unsigned)code);
                return ok(b);
            }
            for (char c : l) { if (c < '0' || c > '9') return bad(L"use a key code (decimal, or 0x hex) or auto"); code = code * 10 + (c - '0'); if (code > 1000) break; }
            if (code < 1 || code > 254) return bad(L"a key code is 1 to 254");
            return ok(l);
        }
    }
    return bad(L"not a value");
}

std::vector<OptiRow> OptiTabRows(const string& ini) {
    std::vector<OptiRow> rows;
    const IniSpan up = IniFind(ini, "Upscalers", "Dx12Upscaler");
    const bool slot = up.found && IEqA(IniValueAt(ini, up), "WrongGPU");  // OptiScaler 10: WrongGPU is its own entry
    for (auto& d : kOptiLocked) {
        const IniSpan s = IniFind(ini, d.section, d.key);
        if (!s.found) continue;
        OptiRow r; r.section = d.section; r.key = d.key; r.value = IniValueAt(ini, s); r.locked = true; r.why = d.why;
        if (slot && IEqA(d.section, "FSR") && IEqA(d.key, "UpscalerIndex")) r.why = L"AMD FSR's own entry, left to OptiScaler (WrongGPU is selected under [WrongGPU])";
        else if (slot && IEqA(d.section, "Libraries") && IEqA(d.key, "FfxDx12SRPath")) r.why = L"managed by WrongGPU: auto (WrongGPU is its own entry in OptiScaler's menu)";
        rows.push_back(r);
    }
    for (auto& d : kOptiEditable) {
        const IniSpan s = IniFind(ini, d.section, d.key);
        if (s.found) { OptiRow r; r.section = d.section; r.key = d.key; r.value = IniValueAt(ini, s); r.def = &d; rows.push_back(r); }
    }
    return rows;
}

bool ApplyOptiEdits(const string& in, const std::vector<std::pair<string, string>>& edits, string& out, wstring* why) {
    string t = in;
    for (auto& e : edits) {
        string section, key;
        if (!SplitOptiName(e.first, section, key)) { if (why) *why = L"name an OptiScaler key as Section.Key: " + FromUtf8(e.first); return false; }
        const wchar_t* lockedWhy = nullptr;
        if (IsOptiLocked(section, key, &lockedWhy)) { if (why) *why = FromUtf8(e.first) + L" is " + lockedWhy; return false; }
        const OptiKeyDef* d = FindOptiKey(section, key);
        if (!d) { if (why) *why = FromUtf8(e.first) + L" is not an OptiScaler key the launcher edits"; return false; }
        const IniSpan s = IniFind(t, section, key);
        if (!s.found) { if (why) *why = L"this OptiScaler build has no [" + FromUtf8(d->section) + L"] " + FromUtf8(d->key) + L" key"; return false; }
        string normal;
        if (!ValidateOptiValue(*d, e.second, &normal, why)) return false;
        t = IniReplaceValue(t, s, normal);
    }
    out = t;
    return true;
}

// ---- the journal ----------------------------------------------------------------------------------------------------------
static IniKeyChange* FindChange(std::vector<IniKeyChange>& j, const wstring& file, const string& section, const string& key) {
    for (auto& c : j) if (IEq(c.file, file) && IEqA(c.section, section) && IEqA(c.key, key)) return &c;
    return nullptr;
}
void JournalChanges(std::vector<IniKeyChange>& journal, const wstring& file, const string& before, const string& after) {
    std::map<string, string> old;
    for (auto& e : IniEntries(before)) old[LowerA(e.section) + "\n" + LowerA(e.key)] = e.value;
    for (auto& e : IniEntries(after)) {
        auto it = old.find(LowerA(e.section) + "\n" + LowerA(e.key));
        if (it == old.end() || it->second == e.value) continue;
        IniKeyChange* c = FindChange(journal, file, e.section, e.key);
        if (!c) { journal.push_back({file, e.section, e.key, false, it->second, e.value}); continue; }
        c->ours = e.value;
    }
    journal.erase(std::remove_if(journal.begin(), journal.end(), [&](const IniKeyChange& c) { return IEq(c.file, file) && !c.absent && c.original == c.ours; }),
                  journal.end());
}
string RevertJournal(const string& text, const std::vector<IniKeyChange>& journal, const wstring& file, std::vector<wstring>* notes) {
    string t = text;
    for (auto it = journal.rbegin(); it != journal.rend(); ++it) {
        const IniKeyChange& c = *it;
        if (!IEq(c.file, file)) continue;
        const IniSpan s = IniFind(t, c.section, c.key);
        if (!s.found) { if (notes) notes->push_back(L"[" + FromUtf8(c.section) + L"] " + FromUtf8(c.key) + L" is not in the file any more"); continue; }
        if (notes && IniValueAt(t, s) != c.ours)
            notes->push_back(L"[" + FromUtf8(c.section) + L"] " + FromUtf8(c.key) + L" had been changed outside the launcher (" + FromUtf8(IniValueAt(t, s)) + L")");
        if (c.absent) t = t.substr(0, s.lineBegin) + (s.lineEnd < t.size() ? t.substr(s.lineEnd + 1) : string());  // the line WrongGPU added
        else t = IniReplaceValue(t, s, c.original);
    }
    return t;
}
std::vector<OptiOutside> JournalOutside(const string& text, const std::vector<IniKeyChange>& journal, const wstring& file) {
    std::vector<OptiOutside> v;
    for (auto& c : journal) {
        if (!IEq(c.file, file)) continue;
        const IniSpan s = IniFind(text, c.section, c.key);
        const string cur = s.found ? IniValueAt(text, s) : string("(missing)");
        if (cur != c.ours) v.push_back({c.section, c.key, cur, c.ours});
    }
    return v;
}
string ReplayJournal(const string& fresh, const std::vector<IniKeyChange>& old, const wstring& file, std::vector<IniKeyChange>& journal,
                     std::vector<wstring>* notes) {
    string t = fresh;
    for (auto& c : old) {
        if (!IEq(c.file, file) || IsOptiLocked(c.section, c.key)) continue;
        const IniSpan s = IniFind(t, c.section, c.key);
        if (!s.found) { if (notes) notes->push_back(L"OptiScaler.ini: [" + FromUtf8(c.section) + L"] " + FromUtf8(c.key) + L" is not in this OptiScaler build; your value was not carried over"); continue; }
        if (IniValueAt(t, s) != c.ours) t = IniReplaceValue(t, s, c.ours);
    }
    JournalChanges(journal, file, fresh, t);
    return t;
}

}  // namespace wg
