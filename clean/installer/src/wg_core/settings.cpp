// SPDX-License-Identifier: MIT
// WrongGPU launcher core: wronggpu.ini: reading and changing keys, the defaults of earlier versions. The per-game
// profile of build/launcher-design.md section 4 comes here (M1). Written for this project.
#include "settings.h"

namespace wg {

// key=value in [section] of an INI text: the existing line replaced, else added at the end of the section (or a new
// section at the end of the file).
string IniSet(const string& text, const string& section, const string& key, const string& value) {
    std::vector<string> lines;
    for (size_t a = 0; a <= text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) { if (a < text.size()) lines.push_back(text.substr(a)); break; }
        lines.push_back(text.substr(a, e - a)); a = e + 1;
    }
    auto trim = [](string t) { while (!t.empty() && (t.back() == '\r' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
                               size_t i = 0; while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i; return t.substr(i); };
    const string head = "[" + section + "]", line = key + "=" + value;
    int sec = -1, end = (int)lines.size();
    for (int i = 0; i < (int)lines.size(); ++i) {
        const string t = trim(lines[i]);
        if (!t.empty() && t[0] == '[') { if (sec >= 0) { end = i; break; } if (t == head) sec = i; continue; }
        if (sec < 0) continue;
        const size_t eq = t.find('=');
        if (eq != string::npos && trim(t.substr(0, eq)) == key) { lines[i] = line + "\r"; sec = -2; break; }
    }
    if (sec == -1) { lines.push_back("\r"); lines.push_back(head + "\r"); lines.push_back(line + "\r"); }
    else if (sec >= 0) {
        int at = end;
        while (at - 1 > sec && trim(lines[at - 1]).empty()) --at;  // before the blank lines that end the section
        lines.insert(lines.begin() + at, line + "\r");
    }
    string out;
    for (size_t i = 0; i < lines.size(); ++i) { out += lines[i]; if (i + 1 < lines.size() || (!text.empty() && text.back() == '\n')) out += "\n"; }
    return out;
}
// The value of `key` as WrongGPU's front end reads wronggpu.ini (keys before any section, in [wronggpu] and [fsr]; the
// last one wins; a '#' or ';' after a space or tab starts a comment); false when it is not set. *section: where it is.
bool IniGetFrontEnd(const string& text, const string& key, string* value, string* section) {
    auto trim = [](string t) { while (!t.empty() && (t.back() == '\r' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
                               size_t i = 0; while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i; return t.substr(i); };
    string sec;
    bool found = false;
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) e = text.size();
        string t = trim(text.substr(a, e - a));
        a = e + 1;
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        if (t[0] == '[') { sec = Lower(FromUtf8(t)) == L"[wronggpu]" ? "wronggpu" : Lower(FromUtf8(t)) == L"[fsr]" ? "fsr" : "other"; continue; }
        if (sec == "other") continue;
        const size_t eq = t.find('=');
        if (eq == string::npos || Lower(FromUtf8(trim(t.substr(0, eq)))) != FromUtf8(key)) continue;
        string v = t.substr(eq + 1);
        for (size_t i = 1; i < v.size(); ++i)
            if ((v[i] == '#' || v[i] == ';') && (v[i - 1] == ' ' || v[i - 1] == '\t')) { v = v.substr(0, i); break; }
        *value = trim(v);
        *section = sec.empty() ? "fsr" : sec;
        found = true;
    }
    return found;
}
// Whether [section] has `key` (section and key compared without case); *value: its last value.
bool IniHasKey(const string& text, const string& section, const string& key, string* value) {
    const wstring head = L"[" + Lower(FromUtf8(section)) + L"]", lkey = Lower(FromUtf8(key));
    bool in = false, found = false;
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) e = text.size();
        wstring t = FromUtf8(text.substr(a, e - a));
        a = e + 1;
        while (!t.empty() && (t.back() == L'\r' || t.back() == L' ' || t.back() == L'\t')) t.pop_back();
        size_t i = 0; while (i < t.size() && (t[i] == L' ' || t[i] == L'\t')) ++i;
        t = t.substr(i);
        if (t.empty() || t[0] == L';' || t[0] == L'#') continue;
        if (t[0] == L'[') { in = Lower(t) == head; continue; }
        if (!in) continue;
        const size_t eq = t.find(L'=');
        if (eq == wstring::npos) continue;
        wstring k = t.substr(0, eq);
        while (!k.empty() && (k.back() == L' ' || k.back() == L'\t')) k.pop_back();
        if (Lower(k) != lkey) continue;
        found = true;
        if (value) {
            wstring v = t.substr(eq + 1);
            size_t s = 0; while (s < v.size() && (v[s] == L' ' || v[s] == L'\t')) ++s;
            *value = ToUtf8(v.substr(s));
        }
    }
    return found;
}
// Every line of `key` the front end would read (before any section, [wronggpu], [fsr]; key without case) removed.
string IniRemoveFrontEnd(const string& text, const string& key) {
    string out, sec;
    const wstring lkey = Lower(FromUtf8(key));
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        const size_t next = e == string::npos ? text.size() : e + 1;
        const string line = text.substr(a, next - a);
        a = next;
        wstring t = FromUtf8(line);
        while (!t.empty() && (t.back() == L'\r' || t.back() == L'\n' || t.back() == L' ' || t.back() == L'\t')) t.pop_back();
        size_t i = 0; while (i < t.size() && (t[i] == L' ' || t[i] == L'\t')) ++i;
        t = t.substr(i);
        if (!t.empty() && t[0] == L'[') { const wstring l = Lower(t); sec = l == L"[wronggpu]" || l == L"[fsr]" ? "fe" : "other"; out += line; continue; }
        if (sec != "other" && !t.empty() && t[0] != L'#' && t[0] != L';') {
            const size_t eq = t.find(L'=');
            if (eq != wstring::npos) {
                wstring k = t.substr(0, eq);
                while (!k.empty() && (k.back() == L' ' || k.back() == L'\t')) k.pop_back();
                if (Lower(k) == lkey) continue;
            }
        }
        out += line;
    }
    return out;
}
// Settings an earlier WrongGPU wrote into wronggpu.ini as its defaults that are wrong for this version. The front end
// writes all its defaults into wronggpu.ini at its first start, and an update keeps the player's file; a value that
// still equals what that earlier version wrote by itself is brought to today's default. Values the player changed stay.
const StaleDefault kStaleDefaults[] = {
    {L"0.3.0", "upscaler", "standin", "network", L"0.3.0 started its diagnostics stand-in instead of Model M"},
    {L"0.3.0", "log_dispatches", "1", "0", L"0.3.0 logged every frame (large logs, a little slower)"},
};
// "0.3.0" < "0.3.1" < "0.10.0"; an empty or unreadable version counts as the oldest.
bool VersionAtMost(const wstring& v, const wstring& limit) {
    auto parts = [](const wstring& s) { std::vector<int> p; int n = 0; bool any = false;
                                        for (wchar_t c : s) { if (c >= L'0' && c <= L'9') { n = n * 10 + (c - L'0'); any = true; }
                                                              else if (c == L'.') { p.push_back(n); n = 0; } else break; }
                                        if (any) p.push_back(n); return p; };
    std::vector<int> a = parts(v), b = parts(limit);
    if (a.empty()) return true;
    a.resize(3, 0); b.resize(3, 0);
    return a <= b;
}
void MigrateSettings(Context& ctx, const wstring& dir, const wstring& oldVersion, Ui& ui) {
    const wstring path = Join(dir, L"wronggpu.ini");
    string text;
    if (!ReadAll(path, text)) return;
    string changed = text;
    for (auto& s : kStaleDefaults) {
        string value, section;
        if (!VersionAtMost(oldVersion, s.upTo) || !IniGetFrontEnd(changed, s.key, &value, &section) || value != s.old) continue;
        changed = IniSet(changed, section, s.key, s.now);
        ui.log(L"wronggpu.ini: " + FromUtf8(s.key) + L" = " + FromUtf8(s.old) + L" was the default of WrongGPU " + oldVersion + L" (" + s.why +
               L"); set to " + FromUtf8(s.now) + L".");
    }
    if (changed != text) WriteFileAtomic(path, changed.data(), changed.size());
    else LogLine(ctx, L"wronggpu.ini kept as it is (written by WrongGPU " + (oldVersion.empty() ? wstring(L"?") : oldVersion) + L"; nothing outdated in it)");
}

}  // namespace wg
