// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the engine setting Ray Reconstruction needs in Borderlands 4 and Hogwarts Legacy (rrconfig.h).
// Written for this project.
#include "rrconfig.h"

#include <shlobj.h>

#include "rrconfig_line.h"
#include "rr.h"

namespace wg {

static const RrConfigGame kRrConfigGames[] = {
    {L"Borderlands4.exe", L"Borderlands 4", true, L"My Games\\Borderlands 4\\Saved\\Config", true},
    {L"HogwartsLegacy.exe", L"Hogwarts Legacy", false, L"Hogwarts Legacy\\Saved\\Config", false},
};
static const char kSection[] = "SystemSettings";

const RrConfigGame* RrConfigGameFor(const wstring& exe) {
    for (auto& g : kRrConfigGames) if (IEq(BaseName(exe), g.exe)) return &g;
    return nullptr;
}
wstring RrConfigGameExes() { return wstring(kRrConfigGames[0].exe) + L" and " + kRrConfigGames[1].exe; }

static string LineKey() { const string l = kRrConfigLine; return l.substr(0, l.find('=')); }
static string LineValue() { const string l = kRrConfigLine; return l.substr(l.find('=') + 1); }
wstring RrConfigLineW() { return FromUtf8(kRrConfigLine); }
string RrConfigNewFile() { return string("[") + kSection + "]\r\n" + kRrConfigLine + "\r\n"; }

static wstring KnownFolder(REFKNOWNFOLDERID id) {
    PWSTR p = nullptr;
    wstring r;
    if (SUCCEEDED(SHGetKnownFolderPath(id, KF_FLAG_DONT_VERIFY, nullptr, &p)) && p) r = p;
    if (p) CoTaskMemFree(p);
    return r;
}
#ifdef WRONGGPU_TESTING
static wstring Trimmed(wstring s) { while (!s.empty() && (s.back() == L'\\' || s.back() == L'/')) s.pop_back(); return s; }
#endif

wstring PlayerDocumentsDir(Context& ctx) {
#ifdef WRONGGPU_TESTING
    // a test root only (the job's, else WRONGGPU_TEST_DOCUMENTS), never this PC's Documents
    const wstring t = !ctx.playerDocuments.empty() ? ctx.playerDocuments : EnvVar(L"WRONGGPU_TEST_DOCUMENTS");
    if (t.empty() || IEq(Trimmed(FullPath(t)), Trimmed(KnownFolder(FOLDERID_Documents)))) return L"";
    return t;
#else
    if (!ctx.playerDocuments.empty()) return ctx.playerDocuments;
    return KnownFolder(FOLDERID_Documents);  // where Windows has it (OneDrive's Documents when it is redirected there)
#endif
}

static wstring LocalAppDataDir() {
    const wstring l = EnvVar(L"LOCALAPPDATA");  // the elevated worker sets the player's from its job
#ifdef WRONGGPU_TESTING
    if (l.empty() || IEq(Trimmed(FullPath(l)), Trimmed(KnownFolder(FOLDERID_LocalAppData)))) return L"";  // a test root only
#endif
    return l;
}

wstring RrConfigFile(Context& ctx, const RrConfigGame& g, wstring* why, bool* known) {
    if (known) *known = false;
    const wstring root = g.documents ? PlayerDocumentsDir(ctx) : LocalAppDataDir();
    if (root.empty()) {
        if (why) *why = g.documents ? L"the Documents folder could not be found" : L"the local application data folder could not be found";
        return L"";
    }
    const wstring config = Join(root, g.folder);
    // the sub-folder that holds the game's other settings (its GameUserSettings.ini): Windows, or WindowsNoEditor
    for (const wchar_t* sub : {L"Windows", L"WindowsNoEditor"})
        if (IsFile(Join(Join(config, sub), L"GameUserSettings.ini"))) {
            if (known) *known = true;
            return Join(Join(config, sub), L"Engine.ini");
        }
    return Join(Join(config, L"Windows"), L"Engine.ini");  // a guess: the game hasn't made its settings folder yet
}

// ---- player text ---------------------------------------------------------------------------------------------------
wstring RrConfigWhy(const RrConfigGame& g) {
    return L"Ray Reconstruction in " + wstring(g.name) + L" needs one line in Engine.ini, because the game's own menu can't turn it on.";
}
wstring RrConfigNotice(const RrConfigGame& g, bool protectedFolder, bool beforeFirstStart) {
    const wstring name = g.name;
    // the game named in the title, the recommended way first, every choice open to change (UI review #2)
    const wstring title = name + L" needs one more setting for Ray Reconstruction\n\n";
    // Fix it for me in a protected folder brings Windows' permission prompt: the offer says so, as the game page's card does (r4 #9).
    // Before the game's first start (r6 #7: when Windows asks): Fix it for me saves the choice in the game's folder at once (the
    // elevated worker: Windows asks now); the line goes in only after that start, and never by itself in a protected folder: Fix it
    // now on its page adds it, and Windows asks again then (ui_rrconfig.cpp Answer, StartDue). r7 #7: three sentences, nothing said
    // twice (the read-only reason once, no "change it on the game's page": the card there says that)
    const wstring ask = protectedFolder && !beforeFirstStart ? L" (Windows asks for permission first, because " + name + L" is installed in a protected folder)"
                                                             : wstring();
    if (protectedFolder && beforeFirstStart)
        return title + RrConfigWhy(g) + L" WrongGPU can add it after the game's first start" +
               (g.readOnly ? L" and keep the file read-only, so " + name + L" doesn't delete it" : wstring()) + L". " + name +
               L" is in a protected folder, so Windows asks for permission twice: when you click Fix it for me, and when you click Fix it now on the "
               L"game's page after that start.";
    // led by what the line is for, as the game page's card says it (r2 #8: RrConfigWhy)
    if (g.readOnly)
        return title + RrConfigWhy(g) + L" " + name + L" deletes its Engine.ini on every start, so WrongGPU can add the line and keep the file "
                       L"read-only" + ask + L". You can change this on the game's page at any time, and uninstalling WrongGPU takes it out.";
    return title + RrConfigWhy(g) + L" WrongGPU can add it for you" + ask + L". You can change this on the game's page at any time, and "
                   L"uninstalling WrongGPU takes it out.";
}

wstring RrConfigSteps(const RrConfigGame& g, const wstring& file) {
    const wstring folder = !file.empty() ? Parent(file)
                           : g.documents ? L"Documents\\" + wstring(g.folder) + L"\\Windows" : L"%LOCALAPPDATA%\\" + wstring(g.folder) + L"\\Windows";
    wstring t = L"Ray Reconstruction in " + wstring(g.name) + L": the steps\n\n"
                L"1. Open this folder in File Explorer (if it isn't there, start the game once: it makes it):\n" + folder + L"\n"
                L"2. Open Engine.ini there with Notepad (create the file if it isn't there).\n"
                L"3. Find the line [" + FromUtf8(kSection) + L"]. If the file doesn't have it, add it at the end.\n"
                L"4. Add this line under it (Copy the line puts it on the clipboard):\n" + RrConfigLineW() + L"\n";
    if (g.readOnly)
        t += L"5. Save the file. Then right-click Engine.ini, choose Properties, tick Read-only and click OK: " + wstring(g.name) +
             L" removes the file on every start otherwise.";
    else
        t += L"5. Save the file.";
    return t;
}

// ---- text edits ------------------------------------------------------------------------------------------------------
namespace {
struct Line { string text, eol; };  // a line without its line end, and its line end ("" for a last line without one)
std::vector<Line> SplitLines(const string& in) {
    std::vector<Line> out;
    size_t at = 0;
    while (at < in.size()) {
        size_t nl = in.find('\n', at);
        if (nl == string::npos) { out.push_back({in.substr(at), ""}); break; }
        const bool cr = nl > at && in[nl - 1] == '\r';
        out.push_back({in.substr(at, nl - at - (cr ? 1 : 0)), cr ? "\r\n" : "\n"});
        at = nl + 1;
    }
    return out;
}
string JoinLines(const std::vector<Line>& ls) { string t; for (auto& l : ls) t += l.text + l.eol; return t; }
string Trim(const string& s) {
    size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
    return a == string::npos ? string() : s.substr(a, b - a + 1);
}
bool IEqA(const string& a, const string& b) { return a.size() == b.size() && _stricmp(a.c_str(), b.c_str()) == 0; }
bool IsHeader(const string& t, string* name = nullptr) {
    const string s = Trim(t);
    if (s.size() < 2 || s[0] != '[') return false;
    const size_t c = s.find(']');
    if (c == string::npos) return false;
    if (name) *name = Trim(s.substr(1, c - 1));
    return true;
}
// The key of a "key=value" line ("" for comments, headers, blank lines and lines without '=').
string KeyOf(const string& t) {
    const string s = Trim(t);
    if (s.empty() || s[0] == ';' || s[0] == '#' || s[0] == '[') return "";
    const size_t eq = s.find('=');
    return eq == string::npos ? "" : Trim(s.substr(0, eq));
}
string ValueOf(const string& t) { const size_t eq = t.find('='); return eq == string::npos ? "" : Trim(t.substr(eq + 1)); }
// [first, last) of the first [SystemSettings] section's lines after its header; header index in *h (npos: none).
void FindSection(const std::vector<Line>& ls, size_t* h, size_t* end) {
    *h = string::npos; *end = ls.size();
    for (size_t i = 0; i < ls.size(); ++i) {
        string n;
        if (!IsHeader(ls[i].text, &n)) continue;
        if (*h != string::npos) { *end = i; return; }
        if (IEqA(n, kSection)) *h = i;
    }
}
string EolOf(const string& in) { return in.empty() || in.find("\r\n") != string::npos ? "\r\n" : in.find('\n') != string::npos ? "\n" : "\r\n"; }
}  // namespace

// A UTF-8 byte order mark (Notepad's "UTF-8 with BOM"): not part of the first line. Taken off before the text is read, so a
// [SystemSettings] right after it is found (r4 #5), and put back on what is written.
static const char kBom[] = "\xEF\xBB\xBF";
static bool StripBom(string& t) {
    if (t.compare(0, 3, kBom) != 0) return false;
    t.erase(0, 3);
    return true;
}

string RrConfigAddText(const string& in, RrConfigFix& fix, bool* changed, bool first) {
    *changed = false;
    string body = in;
    const string bom = StripBom(body) ? kBom : "";  // kept as it is on what is written (r4 #5)
    std::vector<Line> ls = SplitLines(body);
    const string eol = EolOf(body), key = LineKey(), value = LineValue();
    size_t h, end;
    FindSection(ls, &h, &end);
    if (h == string::npos) {  // no [SystemSettings]: added at the end
        if (!ls.empty() && ls.back().eol.empty()) ls.back().eol = eol;
        if (!ls.empty() && !Trim(ls.back().text).empty()) ls.push_back({"", eol});
        ls.push_back({"[" + string(kSection) + "]", eol});
        ls.push_back({kRrConfigLine, eol});
        fix.sectionAdded = true;
        *changed = true;
        return bom + JoinLines(ls);
    }
    for (size_t i = h + 1; i < end; ++i)
        if (IEqA(KeyOf(ls[i].text), key)) {
            const string v = ValueOf(ls[i].text);
            // the player's key (the first apply, or a value changed since): its value is what =off puts back
            if (first || (fix.prevAbsent && v != value)) { fix.prevAbsent = false; fix.prev = v; }
            if (v == value) return in;  // there already
            ls[i].text = ls[i].text.substr(0, ls[i].text.find('=') + 1) + value;
            *changed = true;
            return bom + JoinLines(ls);
        }
    // the section without the key: the line after its last non-blank line
    size_t at = h + 1;
    for (size_t i = h + 1; i < end; ++i) if (!Trim(ls[i].text).empty()) at = i + 1;
    if (ls[at - 1].eol.empty()) ls[at - 1].eol = eol;
    ls.insert(ls.begin() + (ptrdiff_t)at, Line{kRrConfigLine, eol});
    *changed = true;
    return bom + JoinLines(ls);
}

string RrConfigRemoveText(const string& in, const RrConfigFix& fix) {
    string body = in;
    const string bom = StripBom(body) ? kBom : "";  // (r4 #5)
    std::vector<Line> ls = SplitLines(body);
    const string key = LineKey();
    size_t h, end;
    FindSection(ls, &h, &end);
    if (h == string::npos) return in;
    for (size_t i = h + 1; i < end; ++i)
        if (IEqA(KeyOf(ls[i].text), key)) {
            if (fix.prevAbsent) { ls.erase(ls.begin() + (ptrdiff_t)i); --end; }
            else ls[i].text = ls[i].text.substr(0, ls[i].text.find('=') + 1) + fix.prev;
            break;
        }
    if (fix.sectionAdded) {  // the section the fix added: gone again when nothing else is in it (and the blank line before it)
        bool empty = true;
        for (size_t i = h + 1; i < end; ++i) if (!Trim(ls[i].text).empty()) empty = false;
        if (empty) {
            ls.erase(ls.begin() + (ptrdiff_t)h, ls.begin() + (ptrdiff_t)end);
            if (h > 0 && h == ls.size() && Trim(ls[h - 1].text).empty()) ls.erase(ls.begin() + (ptrdiff_t)(h - 1));
        }
    }
    return bom + JoinLines(ls);
}

// ---- the file ---------------------------------------------------------------------------------------------------------
wstring RrConfigBackupPath(const wstring& dir) { return Join(Join(StateDir(dir), L"rr-config"), L"Engine.ini"); }

static bool ReadOnly(const wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_READONLY);
}
static void SetReadOnly(const wstring& p, bool on) {
    const DWORD a = GetFileAttributesW(p.c_str());
    if (a == INVALID_FILE_ATTRIBUTES) return;
    const DWORD b = on ? (a | FILE_ATTRIBUTE_READONLY) : (a & ~(DWORD)FILE_ATTRIBUTE_READONLY);
    if (b != a && !SetFileAttributesW(p.c_str(), b)) Fail(L"Cannot change the read-only attribute of " + p + L": " + ErrText(GetLastError()));
}
static bool Utf16(const string& t) { return t.size() >= 2 && (((unsigned char)t[0] == 0xFF && (unsigned char)t[1] == 0xFE) || ((unsigned char)t[0] == 0xFE && (unsigned char)t[1] == 0xFF)); }

const wchar_t kRrConfigWaitNote[] = L"the Ray Reconstruction setting will be added after you start the game once";

int RrConfigWaiting(Context& ctx, const Record& m, const string& fixSetting) {
    const RrConfigGame* g = RrConfigGameFor(m.exe);
    if (!g || fixSetting != "on" || m.rrconfig.present) return 0;
    bool known = false;
    if (RrConfigFile(ctx, *g, nullptr, &known).empty()) return 0;
    return known ? 1 : -1;
}

bool ApplyRrConfig(Context& ctx, const wstring& dir, Record& m, std::vector<wstring>& notes, std::vector<wstring>& warnings) {
    const RrConfigGame* g = RrConfigGameFor(m.exe);
    if (!g) return false;
    // rc6: where WrongGPU's RR isn't offered (the updated Hogwarts Legacy, rr.cpp) the line is never added; one that is there
    // stays as the player chose (turning the fix off and Uninstall still take it out: RevertRrConfig)
    if (const RrUse ru = DetectRr(m.exe); ru.notOffered) {
        notes.push_back(L"Engine.ini left as it is: WrongGPU's Ray Reconstruction isn't offered in this game");
        LogLine(ctx, L"rr_config_fix: Engine.ini left as it is for " + dir + L" (WrongGPU's Ray Reconstruction isn't offered there)");
        return false;
    }
    RrConfigFix& fix = m.rrconfig;
    bool known = false;
    wstring why;
    const wstring found = RrConfigFile(ctx, *g, &why, &known);
    // added before the game made its settings folder, in a guessed sub-folder that isn't the game's (an earlier build: Hogwarts
    // Legacy uses WindowsNoEditor, the guess was Windows): taken out there, then added in the game's own (r2b #1)
    bool moved = false;
    if (fix.present && known && !IEq(fix.path, found)) {
        LogLine(ctx, L"rr_config_fix: " + fix.path + L" isn't the game's Engine.ini (" + found + L"): the line moves there");
        RevertRrConfig(ctx, dir, m, notes, warnings);
        moved = true;
    }
    const wstring file = fix.present ? fix.path : found;
    if (file.empty()) {
        warnings.push_back(L"Ray Reconstruction setting not added for " + wstring(g->name) + L": " + why + L". You can add it yourself (the release notes have the steps).");
        return moved;
    }
    // the game's settings folder isn't there yet (it makes it at its first start; the sub-folder is a guess until then): the
    // choice stays (rr_config_fix=on), nothing is created, and the next Apply adds the line once the folder is there (r2b #1)
    if ((!fix.present && !known) || (fix.present && !IsDir(Parent(file)))) {
        notes.push_back(wstring(g->name) + L": " + kRrConfigWaitNote + L" (no settings in " + Parent(Parent(file)) + L" yet)");
        LogLine(ctx, L"rr_config_fix: waiting for the game's settings folder in " + Parent(Parent(file)) + L" (nothing written)");
        return moved;
    }
    try {
        string in;
        const bool had = IsFile(file);
        if (had && !ReadAll(file, in)) Fail(L"Cannot read " + file);
        if (Utf16(in)) Fail(file + L" is saved as UTF-16, which WrongGPU doesn't edit");
        RrConfigFix next = fix;
        if (!fix.present) {
            next = RrConfigFix();
            next.present = true; next.path = file; next.created = !had; next.wasReadOnly = had && ReadOnly(file);
        }
        string out;
        bool changed = false;
        if (!had) {  // a new file (or one that went away since: the line again, which the revert takes out as it added it)
            out = RrConfigNewFile();
            changed = true;
            if (fix.present) next.sectionAdded = true;
            // (its folder is there: the game made it. No folder is made here any more, r2b #1; a record's folder field from an
            // earlier build that made them stays, so its revert still takes them out)
        } else {
            out = RrConfigAddText(in, next, &changed, !fix.present);
        }
        if (had && !fix.present) {  // the player's file as it was, kept in the game's wronggpu-setup folder
            const wstring b = RrConfigBackupPath(dir);
            EnsureDir(Parent(b));
            WriteFileAtomic(b, in.data(), in.size());
            next.backup = true;
        }
        if (changed) {
            if (had) SetReadOnly(file, false);
            WriteFileAtomic(file, out.data(), out.size());
            if (next.wasReadOnly) SetReadOnly(file, true);
            notes.push_back(wstring(g->name) + L": Engine.ini has the Ray Reconstruction setting (" + file + L")");
            LogLine(ctx, L"rr_config_fix: " + file + (had ? L" (line added)" : L" (created)"));
        }
        if (g->readOnly && !ReadOnly(file)) {
            SetReadOnly(file, true);
            next.setReadOnly = true;
            notes.push_back(wstring(g->name) + L": Engine.ini is read-only now, so the game keeps it");
            LogLine(ctx, L"rr_config_fix: " + file + L" made read-only");
        }
        const bool recordChanged = moved || !fix.present || next.sectionAdded != fix.sectionAdded || next.setReadOnly != fix.setReadOnly || next.folder != fix.folder ||
                                   next.prevAbsent != fix.prevAbsent;
        fix = next;
        return recordChanged;
    } catch (const SetupError& e) {
        warnings.push_back(L"Ray Reconstruction setting not added for " + wstring(g->name) + L": " + e.msg);
        LogLine(ctx, L"rr_config_fix ERROR: " + e.msg);
        return false;
    }
}

void RevertRrConfig(Context& ctx, const wstring& dir, Record& m, std::vector<wstring>& notes, std::vector<wstring>& warnings, const string* backupBytes) {
    RrConfigFix fix = m.rrconfig;
    if (!fix.present) return;
    const wstring& file = fix.path;
    try {
        string backup;
        if (fix.backup) {
            if (backupBytes) backup = *backupBytes;
            else if (!ReadAll(RrConfigBackupPath(dir), backup)) Fail(L"the backup of " + file + L" is missing");
        }
        if (IsFile(file)) {
            string in;
            if (!ReadAll(file, in)) Fail(L"Cannot read " + file);
            string out = RrConfigRemoveText(in, fix);
            string bare = out;
            StripBom(bare);  // a file left with its byte order mark alone is empty too (r4 #5)
            const bool empty = bare.find_first_not_of(" \t\r\n") == string::npos;
            SetReadOnly(file, false);
            if (fix.created && (in == RrConfigNewFile() || empty)) {
                ForceDelete(file);
                LogLine(ctx, L"rr_config_fix off: " + file + L" deleted (WrongGPU created it)");
            } else {
                if (!fix.created && empty && fix.backup) out = backup;  // nothing but our edit left: the player's file as it was
                if (out != in) WriteFileAtomic(file, out.data(), out.size());
                if (fix.wasReadOnly) SetReadOnly(file, true);
                LogLine(ctx, L"rr_config_fix off: " + file + L" put back");
            }
        } else if (!fix.created && fix.backup) {  // gone since: the player's file from the backup
            EnsureDir(Parent(file));
            WriteFileAtomic(file, backup.data(), backup.size());
            if (fix.wasReadOnly) SetReadOnly(file, true);
            LogLine(ctx, L"rr_config_fix off: " + file + L" restored from the backup");
        }
        if (!fix.folder.empty())  // the folders the fix made, when nothing else is in them
            for (wstring d = Parent(file); d.size() >= fix.folder.size() && IsDir(d); d = Parent(d)) {
                if (!RemoveDirectoryW(d.c_str())) break;
                if (IEq(d, fix.folder)) break;
            }
        notes.push_back(L"Engine.ini's Ray Reconstruction setting was taken out again (" + file + L")");
        DeleteFileW(RrConfigBackupPath(dir).c_str());
        RemoveDirectoryW(Parent(RrConfigBackupPath(dir)).c_str());
    } catch (const SetupError& e) {
        warnings.push_back(L"Engine.ini could not be put back (" + e.msg + L"): take the line " + RrConfigLineW() + L" out of " + file + L" yourself" +
                           (fix.setReadOnly ? L" and clear its read-only attribute" : L""));
        LogLine(ctx, L"rr_config_fix off ERROR: " + e.msg);
    }
    m.rrconfig = RrConfigFix();
}

int RrConfigLineState(Context& ctx, const RrConfigGame& g, wstring* file, wstring* readWhy) {
    wstring why;
    bool known = false;
    const wstring f = RrConfigFile(ctx, g, &why, &known);
    if (file) *file = f;
    if (f.empty()) return -1;
    // the game makes its settings folder at its first start; before that the sub-folder is only a guess (Hogwarts Legacy uses
    // WindowsNoEditor, the guess is Windows): not known yet, so a file made there by hand never counts (UI review r2 #7); known
    // once a sub-folder holds the game's GameUserSettings.ini, as for "Fix it for me" (r2b #1)
    if (!known) return -1;
    // no file: no line (0); a file that can't be read, or isn't text WrongGPU reads: said as such (3), never "no line", which
    // would send the player to Fix it again for nothing (r3 #8)
    HANDLE h = CreateFileW(f.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) return 0;
        if (readWhy) *readWhy = e == ERROR_SHARING_VIOLATION ? L"another program has it open" : e == ERROR_ACCESS_DENIED ? L"Windows denied access to it" : ErrText(e);
        return 3;
    }
    CloseHandle(h);
    string text;
    if (!ReadAll(f, text)) { if (readWhy) *readWhy = L"it couldn't be read"; return 3; }
    if (Utf16(text)) { if (readWhy) *readWhy = L"it is saved as UTF-16, which WrongGPU doesn't read"; return 3; }
    RrConfigFix fix;
    bool changed = false;
    RrConfigAddText(text, fix, &changed, true);  // adding the line would change nothing: it is there
    if (changed) return 0;
    if (!g.readOnly) return 1;
    const DWORD a = GetFileAttributesW(f.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_READONLY) ? 1 : 2;
}

}  // namespace wg
