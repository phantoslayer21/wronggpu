// SPDX-License-Identifier: MIT
// Tests of wg_state.h (the per-game state folder, its lock, the operation guard and atomic writes) on real files and real
// processes. The test runs itself as the child process (--child-...). Scratch files go to the TMP folder of the run.
//  - the settings.ini format: tolerant (BOM, comments in any UTF-8, spaces around '=', sections, unknown lines: kept and
//    written back), corrupt only for text that cannot be kept (a control character, a CR outside CRLF, not UTF-8);
//  - WrongGPU's stamp: a new file is stamped, an edited or player-made file is written on without one;
//  - the decision under the state lock (UpdateSettings): a write made while another thread holds the lock is what decide()
//    sees (forced interleaving);
//  - atomic writes: oversized input refused, a failure at every step after the temporary file exists (write, flush, close,
//    move) leaves the target as it was and no temporary file, a cleanup that fails is reported;
//  - SetSettings: no state folder, a folder this process may not write (AccessDenied, its own result), unreadable, a file it
//    cannot keep (Unparseable: refused, the bytes stay), the player's keys and comments kept;
//  - one lock name for several aliases of one folder (case, trailing backslash, 8.3 name, \\?\ form, a junction, a path longer
//    than MAX_PATH), each computed in a SEPARATE process, and mutual exclusion across processes that use different aliases;
//  - a busy lock (bounded wait, nothing written), a holder that dies in the middle of a write (abandoned: the next holder
//    re-reads the old whole file; the half-written temporary file is never read as state);
//  - the operation guard, with a real DLL (wg_state_test_dll) under WrongGPU's binary names: nothing loaded = Ok with every
//    existing binary held exclusively; a held guard makes LoadLibrary of the file fail (here and in another process);
//    deletes through the held handle; PutHeld of a new file and replaces of a held one while another process keeps loading
//    it (complete old or new binary, or a refusal; never partial; no leftovers); a binary loaded in another process or in
//    this one = GameRunning; that
//    process killed = Ok again; a binary the guard may not open = Unknown; the lock busy = Busy; a lock it may not open =
//    LockDenied.
// Written for this project.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "wg_state.h"

using namespace wgstate;

static const char* kId = "abcdef0123456789abcdef0123456789";  // the install id the test's record names
static std::wstring g_hookFile;                                // the settings file the interleaving hooks act on
static DWORD g_editorWriteError = 0, g_editorMoveError = 0;

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char* what) {
    if (ok) ++g_pass;
    else { ++g_fail; std::printf("FAIL: %s\n", what); }
}

static std::wstring Exe() {
    wchar_t b[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, b, MAX_PATH * 2);
    return std::wstring(b, n);
}
static std::wstring TempRoot() {
    wchar_t b[MAX_PATH * 2];
    const DWORD n = GetTempPathW(MAX_PATH * 2, b);
    std::wstring r = JoinPath(std::wstring(b, n), L"wg-state-test-" + std::to_wstring(GetCurrentProcessId()));
    CreateDirectoryW(r.c_str(), nullptr);
    return r;
}
static void WriteText(const std::wstring& p, const std::string& t) {
    HANDLE h = CreateFileW(ExtendedPath(p).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD put = 0;
    if (h != INVALID_HANDLE_VALUE) { WriteFile(h, t.data(), static_cast<DWORD>(t.size()), &put, nullptr); CloseHandle(h); }
}
static std::string ReadText(const std::wstring& p) { std::string t; ReadBytes(p, &t); return t; }
static bool Exists(const std::wstring& p) { return GetFileAttributesW(ExtendedPath(p).c_str()) != INVALID_FILE_ATTRIBUTES; }
static std::vector<std::wstring> List(const std::wstring& dir) {
    std::vector<std::wstring> out;
    WIN32_FIND_DATAW f;
    HANDLE h = FindFirstFileW(JoinPath(ExtendedPath(dir), L"*").c_str(), &f);
    if (h == INVALID_HANDLE_VALUE) return out;
    do {  // (the record with the install id is not counted: the tests count settings.ini and temporary files)
        if (wcscmp(f.cFileName, L".") && wcscmp(f.cFileName, L"..") && _wcsicmp(f.cFileName, L"install.record")) out.push_back(f.cFileName);
    } while (FindNextFileW(h, &f));
    FindClose(h);
    return out;
}
static HANDLE Spawn(const std::wstring& args) {
    std::wstring cmd = L"\"" + Exe() + L"\" " + args;
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return nullptr;
    CloseHandle(pi.hThread);
    return pi.hProcess;
}
static DWORD WaitExit(HANDLE p, DWORD ms = 60000) {
    DWORD code = 999;
    if (WaitForSingleObject(p, ms) == WAIT_OBJECT_0) GetExitCodeProcess(p, &code);
    CloseHandle(p);
    return code;
}
static bool WaitFile(const std::wstring& p, DWORD ms = 30000) {
    for (DWORD t = 0; t < ms; t += 20) {
        if (Exists(p)) return true;
        Sleep(20);
    }
    return false;
}
// One quoted command-line argument (backslashes before the closing quote doubled, as the C runtime's parser wants).
static std::wstring Q(const std::wstring& s) {
    size_t tail = 0;
    while (tail < s.size() && s[s.size() - 1 - tail] == L'\\') ++tail;
    return L"\"" + s + std::wstring(tail, L'\\') + L"\"";
}
// A setting's value, or "(none)" when the key is missing (a failed write or read fails the check, never crashes the test).
static std::string Val(const Settings& s, const char* key) {
    const std::string* v = Get(s, key);
    return v ? *v : std::string("(none)");
}
static unsigned long Counter(const std::wstring& game) {
    const Settings s = ReadSettings(game);
    const std::string* v = Get(s, "counter");
    return v ? std::strtoul(v->c_str(), nullptr, 10) : 0;
}
static std::string Narrow(const std::wstring& w) {
    std::string s;
    for (wchar_t c : w) s.push_back(c < 0x80 ? static_cast<char>(c) : '?');
    return s;
}

// ---- child modes
static int Child(int argc, wchar_t** argv) {
    const std::wstring mode = argv[1], game = argv[2];
    if (mode == L"--child-add" && argc >= 4) {  // n locked read-modify-write increments of one counter
        const int n = _wtoi(argv[3]);
        const std::wstring diag = argc >= 5 ? argv[4] : L"";  // why it stopped, for the parent's failure line
        auto stop = [&](int code, const std::string& why) {
            if (!diag.empty()) WriteText(diag, why);
            return code;
        };
        for (int i = 0; i < n; ++i) {
            StateLock lock(game, 30000);
            if (!lock.held()) return stop(3, "lock not held at increment " + std::to_string(i) + " (error " + std::to_string(lock.error()) + ")");
            const Settings s = ReadSettings(game);  // re-read after the lock is taken
            if (!s.why.empty()) return stop(5, "settings.ini " + s.why + " at increment " + std::to_string(i));  // never count from 0
            const std::string* v = Get(s, "counter");
            if (!v) return stop(6, "no counter in settings.ini at increment " + std::to_string(i));
            const unsigned long c = std::strtoul(v->c_str(), nullptr, 10);
            Settings w = s;
            Set(w, "counter", std::to_string(c + 1));
            Sleep(0);  // widen the window a lost update would need
            const WriteResult r = WriteAtomic(SettingsPathOf(game), RenderSettings(w));
            if (!r.ok) return stop(4, "write " + r.words() + " at increment " + std::to_string(i) + " (counter was " + std::to_string(c) + ")");
        }
        return 0;
    }
    if (mode == L"--child-lockname" && argc >= 4) {  // the lock name of this spelling, into a file
        std::wstring name;
        DWORD err = 0;
        WriteText(argv[3], LockNameFor(game, &name, &err) ? Narrow(name) : "error " + std::to_string(err));
        return 0;
    }
    if (mode == L"--child-hold" && argc >= 5) {  // holds the lock for ms
        StateLock lock(game, 30000);
        if (!lock.held()) return 3;
        WriteText(argv[4], "ready");
        Sleep(static_cast<DWORD>(_wtoi(argv[3])));
        return 0;
    }
    if (mode == L"--child-die-midwrite" && argc >= 5) {  // takes the lock, half-writes a temporary file, dies holding the lock
        StateLock* lock = new StateLock(game, 30000);
        if (!lock->held()) return 3;
        HANDLE h = CreateFileW(ExtendedPath(SettingsPathOf(game) + kTempMark + L"crash-1").c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, 0, nullptr);
        DWORD put = 0;
        if (h != INVALID_HANDLE_VALUE) { WriteFile(h, "counter=99", 10, &put, nullptr); CloseHandle(h); }  // no line end: half a file
        WriteText(argv[3], "ready");
        WaitFile(argv[4]);  // the parent keeps a handle to the mutex now
        ExitProcess(0);
    }
    if (mode == L"--child-load" && argc >= 4) {  // a "game" that loaded a WrongGPU binary (argv[2]), until it is killed
        const HMODULE m = LoadLibraryW(game.c_str());
        WriteText(argv[3], m ? "loaded" : "error " + std::to_string(GetLastError()));
        if (!m) return 3;
        Sleep(120000);
        return 0;
    }
    if (mode == L"--child-loadloop" && argc >= 6) {  // once <go> exists: loads argv[2] again and again for <ms>, then a summary
        WaitFile(argv[4]);
        const ULONGLONG end = GetTickCount64() + static_cast<ULONGLONG>(_wtoi(argv[5]));
        int loads = 0, refused = 0;
        bool v219 = false, v220 = false, vOther = false;
        std::string other;
        while (GetTickCount64() < end) {
            const HMODULE m = LoadLibraryW(game.c_str());
            if (m) {
                ++loads;
                typedef int (*Fn)();
                const Fn f = reinterpret_cast<Fn>(GetProcAddress(m, "WgStateTestDll"));
                const int v = f ? f() : 0;
                if (v == 219) v219 = true;
                else if (v == 220) v220 = true;
                else vOther = true;  // loaded but not a complete binary of either version
                FreeLibrary(m);
            } else {
                const DWORD e = GetLastError();
                // refusals: the guard holds the file (sharing violation), or the name is between the rename-away and the rename-in
                if (e == ERROR_SHARING_VIOLATION || e == ERROR_MOD_NOT_FOUND || e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND || e == ERROR_ACCESS_DENIED) ++refused;
                else if (other.find(std::to_string(e)) == std::string::npos) other += (other.empty() ? "" : ",") + std::to_string(e);  // e.g. a bad image: partial
            }
        }
        std::string values = std::string(v219 ? "219" : "") + (v220 ? (v219 ? ",220" : "220") : "");
        if (vOther) values += values.empty() ? "partial" : ",partial";
        WriteText(argv[3], "loads=" + std::to_string(loads) + " values=" + (values.empty() ? "none" : values) + " refused=" + std::to_string(refused) +
                               " other=" + (other.empty() ? "-" : other));
        return 0;
    }
    if (mode == L"--child-tryload" && argc >= 4) {  // tries to load argv[2] once: "ok" or "error <n>"
        const HMODULE m = LoadLibraryW(game.c_str());
        WriteText(argv[3], m ? std::string("ok") : "error " + std::to_string(GetLastError()));
        if (m) FreeLibrary(m);
        return 0;
    }
    return 2;
}

int wmain(int argc, wchar_t** argv) {
    if (argc >= 3 && std::wstring(argv[1]).rfind(L"--child", 0) == 0) return Child(argc, argv);
    const std::wstring root = TempRoot();
    const std::wstring game = JoinPath(root, L"Game");
    CreateDirectoryW(game.c_str(), nullptr);

    // ---- the format: tolerant, everything kept; corrupt only for text that cannot be kept
    {
        Settings s = ParseSettings("; comment\r\n\r\nrr_mode=fast\r\nrr_enabled=1\n");
        check(s.present && !s.corrupt && s.kv.size() == 2 && Val(s, "rr_mode") == "fast" && Val(s, "rr_enabled") == "1", "parse: comments, blank lines, CRLF and LF");
        struct { const char* text; const char* key; const char* value; } kept[] = {
            {"rr_mode = fast\r\n", "rr_mode", "fast"}, {"[wronggpu]\r\nrr_mode=fast\r\n", "rr_mode", "fast"},
            {"RR_MODE=fast\r\n", nullptr, nullptr}, {"rr_mode=fast\r\nrr_mode=accurate\r\n", "rr_mode", "fast"},
            {"rr_mode\r\n", nullptr, nullptr}, {"rr_mode=fa st\r\n", nullptr, nullptr}, {"=fast\r\n", nullptr, nullptr},
            {"rr_mode=f\xc3\xa4st\r\n", nullptr, nullptr}, {"; \xc3\xa4 a note\r\nrr_mode=exact\r\n", "rr_mode", "exact"},
            {"# hash comment\r\n\trr_mode =\taccurate \r\n", "rr_mode", "accurate"}, {"\xEF\xBB\xBFrr_mode=fast\r\n", "rr_mode", "fast"}};
        bool all = true;
        for (const auto& k : kept) {
            const Settings x = ParseSettings(k.text);
            const bool good = !x.corrupt && (k.key ? (Get(x, k.key) && *Get(x, k.key) == k.value) : x.kv.empty()) && RenderSettings(x) == std::string(k.text);
            if (!good) std::printf("  (kept case %s: corrupt %d, keys %zu, render %s)\n", k.text, x.corrupt, x.kv.size(), RenderSettings(x).c_str());
            all = all && good;
        }
        check(all, "parse: spaces around '=', sections, upper case, a key twice (the first counts), no '=', a value WrongGPU does not take, "
                   "non-ASCII comments and values, '#' comments, tabs, a BOM: read, and written back byte for byte");
        const std::string nul("rr_mode=fast\r\nnote=a\0b\r\n", 24);
        const char* bad[] = {"rr_mode=fast\r", "rr_mode=fa\rst\r\n", "rr_mode=fast\r\r\n", "; a\rb\n", "; \xff\xfe not UTF-8\r\n", "k=v\x01\r\n"};
        bool allBad = ParseSettings(nul).corrupt;
        for (const char* b : bad) { const Settings x = ParseSettings(b); allBad = allBad && x.corrupt && x.kv.empty() && x.lines.empty() && !x.why.empty(); }
        check(allBad, "parse: a NUL or other control character, a bare or embedded CR, bytes that are not UTF-8: corrupt (nothing WrongGPU could keep)");
        check(!ParseSettings("rr_mode=fast").corrupt, "parse: a last line without a line end is fine");
        Settings mixed = ParseSettings("\xEF\xBB\xBF; the player's note \xc3\xa4\nPlayer Key = 1\n[section]\nrr_mode = fast\nother_key=2\n");
        check(Set(mixed, "rr_mode", "exact") && Set(mixed, "new_key", "3") &&
                  RenderSettings(mixed) == "\xEF\xBB\xBF; the player's note \xc3\xa4\nPlayer Key = 1\n[section]\nrr_mode=exact\nother_key=2\nnew_key=3\n",
              "a change keeps the BOM, the comments, the unknown lines and the line ends; the changed key rewritten in place, a new one at the end");
        const std::string st = Stamped("rr_mode=fast\r\n", "settings.ini", kId);
        check(ParseSettings(st).kv.size() == 1 && StampedByWrongGPU(st, "settings.ini", kId) && StampedByWrongGPU(st, "SETTINGS.INI", kId) &&
                  !StampedByWrongGPU(st + "x=1\r\n", "settings.ini", kId) && !StampedByWrongGPU("rr_mode=fast\r\n; wronggpu-written 0123456789abcdef\r\n", "settings.ini", kId),
              "the stamp: a comment to readers, valid only right after the bytes it checks");
        check(!StampedByWrongGPU(st, "settings.ini.wgtmp-backup", kId) && !StampedByWrongGPU(st, "model-choice.ini", kId) &&
                  !StampedByWrongGPU(st, "settings.ini", "fedcba9876543210fedcba9876543210") && !StampedByWrongGPU(st, "settings.ini", ""),
              "the stamp names the file and the install: a copy under another name, from another install, or an install without an id does not match");
        // every line keeps its own line end; nothing is added at the end of a file that had none; a stamp-like comment stays
        Settings ends = ParseSettings("a=1\r\nb=2\nc=3");
        check(Set(ends, "rr_mode", "fast") && RenderSettings(ends) == "a=1\r\nb=2\nc=3\r\nrr_mode=fast",
              "mixed line ends kept line by line; a new key after a last line without a line end: that line gets one, the new one none");
        Settings noEnd = ParseSettings("; wronggpu-written reminder\nrr_mode=fast");
        check(Set(noEnd, "rr_mode", "exact") && RenderSettings(noEnd) == "; wronggpu-written reminder\nrr_mode=exact",
              "a player's comment that starts like the stamp stays; a changed key in a file without a final line end gets none");
        Settings r;
        check(Set(r, "rr_mode", "fast") && Set(r, "rr_mode", "exact") && r.kv.size() == 1 && !Set(r, "Bad Key", "x") && !Set(r, "k", "a b"), "Set: replace, refuse bad key/value");
        const Settings back = ParseSettings(RenderSettings(r));
        check(!back.corrupt && back.kv == r.kv, "render -> parse round trip");
    }

    // ---- atomic writes
    const std::wstring state = StateDirOf(game);
    {
        check(SetSettings(game, {{"rr_mode", "fast"}}).result == SetResult::NoStateFolder && DirectoryState(state) == PathState::Missing,
              "no state folder: NoStateFolder, the folder is not created");
        CreateDirectoryW(state.c_str(), nullptr);
        WriteText(JoinPath(state, L"install.record"), std::string("WRONGGPU-RECORD 2\r\nversion\t0.6.5\r\ninstall-id\t") + kId + "\r\n");
        check(InstallIdOf(game) == kId, "InstallIdOf: the record's install-id line");
        const std::wstring f = SettingsPathOf(game);
        check(SetSettings(game, {{"rr_mode", "fast"}}).ok() && Val(ReadSettings(game), "rr_mode") == "fast", "SetSettings writes the file");
        const std::string before = ReadText(f);
        WriteResult w = WriteAtomic(f, std::string(kMaxStateBytes + 1, 'x'));
        check(!w.ok && w.error == ERROR_FILE_TOO_LARGE && ReadText(f) == before, "over kMaxStateBytes: refused, the file unchanged");
        bool steps = true;
        static const char* failAt = nullptr;
        for (const char* step : {"write", "flush", "close", "move"}) {
            failAt = step;
            TestFault() = [](const char* s) { return std::strcmp(s, failAt) == 0; };
            w = WriteAtomic(f, "rr_mode=exact\r\n");
            TestFault() = nullptr;
            const bool good = !w.ok && w.error == ERROR_WRITE_FAULT && w.tempLeft.empty() && !w.cleanupError && ReadText(f) == before && List(state).size() == 1;
            if (!good) std::printf("  (failure at %s: ok %d, error %lu, left %d, files %zu)\n", step, w.ok, w.error, !w.tempLeft.empty(), List(state).size());
            steps = steps && good;
        }
        check(steps, "a failure at write / flush / close / move (after the temporary file exists): not written, the file unchanged, no temporary file left");
        // secondary failures are reported with the first one: a close that fails after a failed write; a cleanup that fails
        TestFault() = [](const char* s) { return std::strcmp(s, "write") == 0 || std::strcmp(s, "close") == 0; };
        w = WriteAtomic(f, "rr_mode=exact\r\n");
        TestFault() = nullptr;
        check(!w.ok && w.error == ERROR_WRITE_FAULT && w.closeError == ERROR_WRITE_FAULT &&
                  ReadText(f) == before && List(state).size() == 1,
              "a close that fails after a failed write: closeError reported beside the first error");
        TestFault() = [](const char* s) { return std::strcmp(s, "move") == 0 || std::strcmp(s, "delete") == 0; };
        w = WriteAtomic(f, "rr_mode=exact\r\n");
        TestFault() = nullptr;
        check(!w.ok && w.cleanupError == ERROR_WRITE_FAULT && !w.tempLeft.empty() && Exists(w.tempLeft) && IsTempName(w.tempLeft) &&
                  w.words().find("could not be deleted") != std::string::npos && ReadText(f) == before,
              "a cleanup that fails: its error and the file left are reported (words say so), the file unchanged");
        if (!w.tempLeft.empty()) DeleteFileW(w.tempLeft.c_str());
        w = WriteAtomic(JoinPath(JoinPath(root, L"missing-folder"), L"x.ini"), "x");
        check(!w.ok && w.error != 0, "a write into a missing folder fails with its error");
        // the stamp: the file WrongGPU created is stamped and stays so while only WrongGPU writes it
        check(StampedByWrongGPU(ReadText(f), "settings.ini", kId), "a file SetSettings created carries WrongGPU's stamp");
        check(SetSettings(game, {{"rr_enabled", "1"}}).ok() && StampedByWrongGPU(ReadText(f), "settings.ini", kId), "...and keeps it through WrongGPU's next write");
        // the player's edits: kept by every write, and the file is the player's from then on (no stamp)
        WriteText(f, ReadText(f) + "player_key = 7\r\n; the player's note \xc3\xa4\r\n; wronggpu-written a player's line\r\n");
        SetOutcome o = SetSettings(game, {{"rr_mode", "accurate"}});
        std::string t = ReadText(f);
        check(o.ok() && o.written && Val(ReadSettings(game), "rr_mode") == "accurate" && t.find("player_key = 7\r\n") != std::string::npos &&
                  t.find("; the player's note \xc3\xa4\r\n; wronggpu-written a player's line\r\n") != std::string::npos && !StampedByWrongGPU(t, "settings.ini", kId),
              "a file the player edited: the menu's save keeps the player's key and note as they are and writes no stamp (the player's file now)");
        WriteText(f, "\xEF\xBB\xBF# made by the player\r\nRR Mode = Fast\r\n");
        o = SetSettings(game, {{"rr_mode", "fast"}});
        t = ReadText(f);
        check(o.ok() && t.rfind("\xEF\xBB\xBF# made by the player\r\nRR Mode = Fast\r\nrr_mode=fast\r\n", 0) == 0 && !StampedByWrongGPU(t, "settings.ini", kId),
              "a file the player made (BOM, '#' comment, a key in other spelling): kept as it is, the key added, no stamp");
        // text WrongGPU cannot keep as it is: the save is refused and the bytes stay
        const std::string unkeepable = std::string("rr_mode=fast\r\nplayer_key=1\r\nnote=a\0b\r\n", 38);
        WriteText(f, unkeepable);
        check(ReadSettings(game).corrupt && ReadSettings(game).kv.empty(), "a file WrongGPU cannot keep reads as corrupt, no keys");
        o = SetSettings(game, {{"rr_mode", "accurate"}});
        check(o.result == SetResult::Unparseable && !o.written && ReadText(f) == unkeepable && o.words().find("nothing was saved") != std::string::npos &&
                  List(state).size() == 1,
              "...and a save is REFUSED (Unparseable, the reason in words), the bytes left as they are, no temporary file");
        WriteText(f, "\xff\xfeplayer bytes\r\n");
        o = SetSettings(game, {{"rr_mode", "accurate"}});
        check(o.result == SetResult::Unparseable && ReadText(f) == "\xff\xfeplayer bytes\r\n", "bytes that are not UTF-8: refused, left as they are");
        DeleteFileW(f.c_str());
        check(SetSettings(game, {{"rr_mode", "accurate"}}).ok() && StampedByWrongGPU(ReadText(f), "settings.ini", kId), "(a new file again: stamped)");
        const std::string b2 = ReadText(f);
        HANDLE h = CreateFileW(f.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        o = SetSettings(game, {{"rr_mode", "exact"}});
        CloseHandle(h);
        check(o.result == SetResult::Unreadable && o.error != 0 && ReadText(f) == b2 && List(state).size() == 1, "an unreadable file: Unreadable, left as it was");
        // a state folder this process may not write: AccessDenied (its own result, for the elevated repair)
        PSECURITY_DESCRIPTOR ro = nullptr, rw = nullptr;
        ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;0x1200a9;;;WD)", SDDL_REVISION_1, &ro, nullptr);  // read and list only
        ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;WD)", SDDL_REVISION_1, &rw, nullptr);
        const bool locked = ro && SetFileSecurityW(state.c_str(), DACL_SECURITY_INFORMATION, ro);
        o = SetSettings(game, {{"rr_mode", "exact"}});
        if (rw) SetFileSecurityW(state.c_str(), DACL_SECURITY_INFORMATION, rw);  // the owner may always change the DACL back
        if (ro) LocalFree(ro);
        if (rw) LocalFree(rw);
        check(locked && o.result == SetResult::AccessDenied && o.error == ERROR_ACCESS_DENIED && ReadText(f) == b2 && o.words().find("administrator") != std::string::npos,
              "a folder this process may not write: AccessDenied (words name the elevated repair), nothing written");
        // the state lock may not be opened (an object under its name that denies everyone): LockDenied, not contention
        {
            std::wstring lockName;
            LockNameFor(game, &lockName);
            PSECURITY_DESCRIPTOR denyAll = nullptr;
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(D;;GA;;;WD)", SDDL_REVISION_1, &denyAll, nullptr);
            SECURITY_ATTRIBUTES sa{sizeof(sa), denyAll, FALSE};
            HANDLE squat = CreateMutexW(&sa, FALSE, lockName.c_str());
            o = SetSettings(game, {{"rr_mode", "exact"}}, 300);
            OperationGuard g(game, {}, 300);
            check(squat && o.result == SetResult::LockDenied && o.error == ERROR_ACCESS_DENIED && ReadText(f) == b2 &&
                      g.result() == GuardResult::LockDenied && std::string(g.words()).find("access denied") != std::string::npos,
                  "a state lock this process may not open: LockDenied (SetSettings and the guard), not Busy; nothing written");
            if (squat) CloseHandle(squat);
            if (denyAll) LocalFree(denyAll);
        }
        WriteText(f, "counter=0\r\n");
    }

    // ---- one lock name for every alias of the folder, each in a SEPARATE process
    std::vector<std::wstring> aliases;
    {
        aliases.push_back(game);
        aliases.push_back(game + L"\\");
        std::wstring upper = game;
        for (auto& ch : upper) ch = static_cast<wchar_t>(towupper(ch));
        aliases.push_back(upper);
        wchar_t shortp[MAX_PATH * 2];
        const bool haveShort = GetShortPathNameW(game.c_str(), shortp, MAX_PATH * 2) != 0;
        if (haveShort) aliases.push_back(shortp);
        aliases.push_back(L"\\\\?\\" + game);
        const std::wstring junction = JoinPath(root, L"GameJunction");
        HANDLE mk = nullptr;
        {
            std::wstring cmd = L"cmd.exe /d /c mklink /J " + Q(junction) + L" " + Q(game) + L" >nul";
            STARTUPINFOW si{};
            si.cb = sizeof si;
            PROCESS_INFORMATION pi{};
            if (CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) { CloseHandle(pi.hThread); mk = pi.hProcess; }
        }
        if (mk) WaitExit(mk);
        const bool haveJunction = Exists(junction);
        if (haveJunction) aliases.push_back(junction);
        check(haveShort && haveJunction, "aliases made: the 8.3 short name and a junction");
        std::vector<std::string> names;
        bool all = true;
        for (size_t i = 0; i < aliases.size(); ++i) {
            const std::wstring out = JoinPath(root, L"name-" + std::to_wstring(i));
            HANDLE k = Spawn(L"--child-lockname " + Q(aliases[i]) + L" " + Q(out));
            all = all && k && WaitExit(k) == 0;
            names.push_back(ReadText(out));
            DeleteFileW(out.c_str());
        }
        bool same = all && !names.empty() && names[0].rfind("Global\\WrongGPU-state-", 0) == 0;
        for (auto& n : names) same = same && n == names[0];
        if (!same)
            for (size_t i = 0; i < names.size(); ++i) std::printf("  (alias %zu %s -> %s)\n", i, Narrow(aliases[i]).c_str(), names[i].c_str());
        check(same, "lock name: case, trailing backslash, 8.3 name, \\\\?\\ form and a junction give one name, each computed in its own process");
        // a folder deeper than MAX_PATH: the helpers work on it without the long-path setting
        std::wstring deep = root;
        while (deep.size() < MAX_PATH + 40) {
            deep = JoinPath(deep, L"a-rather-long-folder-name-for-the-path-test");
            CreateDirectoryW(ExtendedPath(deep).c_str(), nullptr);
        }
        std::wstring dn;
        CreateDirectoryW(ExtendedPath(StateDirOf(deep)).c_str(), nullptr);
        check(LockNameFor(deep, &dn) && Narrow(dn) != names[0] && SetSettings(deep, {{"k", "v"}}).ok() && Val(ReadSettings(deep), "k") == "v",
              "a game folder deeper than MAX_PATH: lock name, write and read work");
        const std::wstring dout = JoinPath(root, L"name-deep");
        HANDLE k = Spawn(L"--child-lockname " + Q(deep) + L" " + Q(dout));
        if (k) WaitExit(k);
        check(ReadText(dout) == Narrow(dn), "the deep folder's lock name is the same in another process");
        DeleteFileW(dout.c_str());
        // cleanup of the deep folders (own files only)
        DeleteFileW(ExtendedPath(SettingsPathOf(deep)).c_str());
        RemoveDirectoryW(ExtendedPath(StateDirOf(deep)).c_str());
        while (deep.size() > root.size()) {
            RemoveDirectoryW(ExtendedPath(deep).c_str());
            deep = deep.substr(0, deep.find_last_of(L'\\'));
        }
    }

    // ---- mutual exclusion across processes that use DIFFERENT aliases: 4 x 50 locked increments, no update lost
    {
        WriteText(SettingsPathOf(game), "counter=0\r\n");
        std::vector<HANDLE> kids;
        std::vector<std::wstring> diags;
        for (int i = 0; i < 4; ++i) {
            diags.push_back(JoinPath(root, L"add-diag-" + std::to_wstring(i)));
            DeleteFileW(diags.back().c_str());
            kids.push_back(Spawn(L"--child-add " + Q(aliases[(i * 2 + 1) % aliases.size()]) + L" 50 " + Q(diags.back())));
        }
        bool ok = true;
        std::vector<DWORD> codes;
        for (HANDLE k : kids) {
            codes.push_back(k ? WaitExit(k, 120000) : 998);  // 998: not started, 999: no exit in time
            ok = codes.back() == 0 && ok;
        }
        const unsigned long count = Counter(game);
        if (!ok || count != 200) {  // which way it failed: a child that stopped (and why), or increments lost
            std::printf("  (count %lu; exit codes", count);
            for (DWORD c : codes) std::printf(" %lu", static_cast<unsigned long>(c));
            std::printf(")\n");
            for (size_t i = 0; i < diags.size(); ++i)
                if (Exists(diags[i])) std::printf("  (child %zu, alias %s: %s)\n", i, Narrow(aliases[(i * 2 + 1) % aliases.size()]).c_str(), ReadText(diags[i]).c_str());
        }
        for (auto& d : diags) DeleteFileW(d.c_str());
        check(ok && count == 200, "4 processes, 4 different aliases, 50 locked increments each: 200 (one lock for all spellings)");
        check(List(state).size() == 1, "no temporary file stays after concurrent writes");
    }

    // ---- busy: a bounded wait gives up, nothing written
    {
        const std::wstring ready = JoinPath(root, L"ready-hold");
        HANDLE k = Spawn(L"--child-hold " + Q(game) + L" 3000 " + Q(ready));
        const bool up = k && WaitFile(ready);
        const std::string before = ReadText(SettingsPathOf(game));
        const SetOutcome r = SetSettings(game, {{"rr_mode", "fast"}}, 200);
        check(up && r.result == SetResult::Busy && r.error == WAIT_TIMEOUT && ReadText(SettingsPathOf(game)) == before, "lock held elsewhere: Busy after the bounded wait, nothing written");
        if (k) WaitExit(k);
        DeleteFileW(ready.c_str());
    }

    // ---- the decision under the lock (forced interleaving): another program saves while this one waits for the lock; the
    // decision is made from what that program wrote, never from a read made before
    {
        CreateDirectoryW(state.c_str(), nullptr);
        DeleteFileW(SettingsPathOf(game).c_str());
        check(SetSettings(game, {{"rr_mode", "fast"}}).ok(), "(interleaving: the file starts with fast)");
        const std::string staleRead = Val(ReadSettings(game), "rr_mode");  // a decision would rest on this read: "fast"
        HANDLE held = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        std::thread menu([&] {  // the game's menu: takes the lock, saves accurate a moment later, lets go
            StateLock lock(game, 2000);
            SetEvent(held);
            Sleep(300);
            Settings now = ReadSettings(game);
            Set(now, "rr_mode", "accurate");
            WriteAtomic(SettingsPathOf(game), Stamped(RenderSettings(now), "settings.ini", kId));
        });
        WaitForSingleObject(held, 5000);
        std::string seen;
        const SetOutcome o = UpdateSettings(game, [&](const Settings& now, std::vector<std::pair<std::string, std::string>>* out) {
            seen = Get(now, "rr_mode") ? Val(now, "rr_mode") : "";
            if (seen.empty()) out->push_back({"rr_mode", staleRead});  // only when there is none: a saved mode stays
            out->push_back({"rr_exact_allowed", seen == "exact" ? "1" : "0"});
        }, 5000);
        menu.join();
        CloseHandle(held);
        check(staleRead == "fast" && o.ok() && seen == "accurate" && Val(ReadSettings(game), "rr_mode") == "accurate" && StampedByWrongGPU(ReadText(SettingsPathOf(game)), "settings.ini", kId),
              "UpdateSettings: decide() sees the mode another thread saved while this one waited for the lock (accurate), not the older read "
              "(fast); the newer mode stays");
        const std::string before = ReadText(SettingsPathOf(game));
        const SetOutcome same = SetSettings(game, {{"rr_mode", "accurate"}, {"rr_exact_allowed", "0"}});
        check(same.ok() && !same.written && ReadText(SettingsPathOf(game)) == before, "a change to the values already there: nothing written");
    }

    // ---- a player's editor saving while WrongGPU saves (the state lock does not stop it): the file is held from the read to just
    // before the move (an editor's save meanwhile fails: Windows refuses it), and a change or a new file in the last moment
    // refuses WrongGPU's save; the player's bytes always stay
    {
        const std::wstring f = SettingsPathOf(game);
        g_hookFile = f;
        // 1. held: an editor writing the file, or replacing it with its own copy, is refused; WrongGPU's save goes through
        WriteText(f, "rr_mode=fast\r\n");
        TestFault() = [](const char* step) {
            if (std::strcmp(step, "replace-held") == 0) {
                HANDLE h = CreateFileW(ExtendedPath(g_hookFile).c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL, nullptr);
                g_editorWriteError = h == INVALID_HANDLE_VALUE ? GetLastError() : 0;
                if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
                const std::wstring copy = g_hookFile + L".editor";
                WriteText(copy, "rr_mode=exact\r\n; the editor's copy\r\n");
                g_editorMoveError = MoveFileExW(ExtendedPath(copy).c_str(), ExtendedPath(g_hookFile).c_str(), MOVEFILE_REPLACE_EXISTING) ? 0 : GetLastError();
                DeleteFileW(ExtendedPath(copy).c_str());
            }
            return false;
        };
        SetOutcome o = SetSettings(game, {{"rr_mode", "accurate"}});
        TestFault() = nullptr;
        check(g_editorWriteError == ERROR_SHARING_VIOLATION && g_editorMoveError != 0 && o.ok() && o.written && Val(ReadSettings(game), "rr_mode") == "accurate",
              "while WrongGPU holds the file, an editor's write and an editor's replace are refused by Windows (its bytes never land); WrongGPU's save completes");
        // 2. the last moment (the file let go, before the move): another program changes the file: WrongGPU's save is refused
        WriteText(f, "rr_mode=fast\r\n");
        TestFault() = [](const char* step) {
            if (std::strcmp(step, "replace-check") == 0) { Sleep(20); WriteText(g_hookFile, "rr_mode=exact\r\n; the player's save\r\n"); }
            return false;
        };
        o = SetSettings(game, {{"rr_mode", "accurate"}});
        TestFault() = nullptr;
        check(o.result == SetResult::Changed && !o.written && ReadText(f) == "rr_mode=exact\r\n; the player's save\r\n" && List(state).size() == 1 &&
                  o.words().find("nothing was saved") != std::string::npos,
              "a change in the last moment before the move: WrongGPU's save is refused (Changed, in words), the player's bytes stay, no temporary file");
        // 3. the file was not there: another program makes it before the move: refused, its bytes stay
        DeleteFileW(f.c_str());
        TestFault() = [](const char* step) {
            if (std::strcmp(step, "replace-held") == 0) WriteText(g_hookFile, "player_made=1\r\n");
            return false;
        };
        o = SetSettings(game, {{"rr_mode", "accurate"}});
        TestFault() = nullptr;
        check(o.result == SetResult::Changed && ReadText(f) == "player_made=1\r\n" && List(state).size() == 1,
              "a file made by another program while WrongGPU saved a new one: refused (Changed), the other program's file stays, no temporary file");
        DeleteFileW(f.c_str());
    }

    // ---- abandoned in the middle of a write: the next holder re-reads the old whole file
    {
        WriteText(SettingsPathOf(game), "counter=7\r\n");
        const std::wstring ready = JoinPath(root, L"ready-die"), go = JoinPath(root, L"go-die");
        HANDLE k = Spawn(L"--child-die-midwrite " + Q(game) + L" " + Q(ready) + L" " + Q(go));
        const bool up = k && WaitFile(ready);
        std::wstring name;
        LockNameFor(game, &name);
        HANDLE keep = OpenMutexW(SYNCHRONIZE, FALSE, name.c_str());  // keeps the mutex alive past the child's end
        WriteText(go, "go");
        if (k) WaitExit(k);
        {
            StateLock lock(game, 5000);
            const Settings s = ReadSettings(game);  // re-read after the lock
            check(up && keep && lock.held() && lock.abandoned() && !s.corrupt && Val(s, "counter") == "7",
                  "a holder that died mid-write: the lock is taken (abandoned), the state re-read is the old whole file");
            std::wstring temp;
            for (auto& n : List(state)) if (IsTempName(n)) temp = n;
            check(!temp.empty(), "the half-written temporary file is there and named as a temporary file (never read as state)");
            check(SetSettings(game, {{"counter", "8"}}).ok() && Val(ReadSettings(game), "counter") == "8", "a write under the abandoned-then-taken lock works");
            if (!temp.empty()) DeleteFileW(JoinPath(state, temp).c_str());
        }
        if (keep) CloseHandle(keep);
        DeleteFileW(ready.c_str());
        DeleteFileW(go.c_str());
    }

    // ---- the operation guard: Windows decides whether a WrongGPU binary is loaded (an exclusive open of a loaded image fails)
    {
        wchar_t me[MAX_PATH * 2];
        const DWORD n = GetModuleFileNameW(nullptr, me, MAX_PATH * 2);
        std::wstring exeDir(me, n);
        exeDir = exeDir.substr(0, exeDir.find_last_of(L'\\'));
        const std::wstring testDll = JoinPath(exeDir, L"wg_state_test_dll.dll"), testDll2 = JoinPath(exeDir, L"wg_state_test_dll2.dll");
        const std::wstring plugins = JoinPath(game, L"plugins");
        CreateDirectoryW(plugins.c_str(), nullptr);
        const std::wstring bin = JoinPath(plugins, L"WrongGPU-RR.asi"), bin2 = JoinPath(game, L"wronggpu_runtime.dll");
        const bool copied = CopyFileW(testDll.c_str(), bin.c_str(), FALSE) && CopyFileW(testDll.c_str(), bin2.c_str(), FALSE);
        check(copied, "two real DLLs in the game folder (the test DLL under WrongGPU's binary names)");
        const std::vector<std::wstring> bins{bin, bin2, JoinPath(game, L"wronggpu_ffx_dx12.dll") /* not there: skipped */};
        {
            OperationGuard g(game, bins, 1000);
            check(g.ok() && g.Holds(bin) && g.Holds(bin2) && !g.Holds(bins[2]), "nothing loaded: Ok, every existing binary held exclusively, a missing one skipped");
            // a held guard blocks a load of the file, in this process and in another one
            const HMODULE m = LoadLibraryW(bin.c_str());
            const DWORD le = m ? 0 : GetLastError();
            if (m) FreeLibrary(m);
            const std::wstring out = JoinPath(root, L"tryload");
            HANDLE k = Spawn(L"--child-tryload " + Q(bin) + L" " + Q(out));
            if (k) WaitExit(k);
            const std::string r = ReadText(out);
            check(!m && le == ERROR_SHARING_VIOLATION && r.rfind("error ", 0) == 0, "while the guard holds a binary, LoadLibrary of it fails (here and in another process): a game cannot load WrongGPU mid-operation");
            DeleteFileW(out.c_str());
            // the operation itself can still read a held binary (hash it, read its version) with an open that shares writes and deletes
            {
                HANDLE rd = CreateFileW(bin.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
                char head[2] = {0, 0};
                DWORD got = 0;
                const bool read = rd != INVALID_HANDLE_VALUE && ReadFile(rd, head, 2, &got, nullptr) && got == 2 && head[0] == 'M' && head[1] == 'Z';
                if (rd != INVALID_HANDLE_VALUE) CloseHandle(rd);
                check(read, "while the guard holds a binary, the operation can still read it (an open sharing writes and deletes)");
            }
            // deletes go through the held handle
            check(g.DeleteHeld(bin2) == ERROR_SUCCESS && !Exists(bin2) && !g.Holds(bin2), "DeleteHeld: the binary deleted through the guard's own handle");
            check(g.DeleteHeld(bin2) == ERROR_NOT_FOUND, "DeleteHeld of a file the guard does not hold: ERROR_NOT_FOUND");
            // a file the operation creates: renamed in complete, then held
            const std::string newBytes = ReadText(testDll2);
            check(g.PutHeld(bin2, newBytes.data(), newBytes.size()) == ERROR_SUCCESS && g.Holds(bin2) && ReadText(bin2) == newBytes,
                  "PutHeld of a new file: the complete file renamed into place and held at once");
        }
        CopyFileW(testDll.c_str(), bin2.c_str(), FALSE);
        // ---- a replace while another process keeps trying to load the file: it gets the old or the new COMPLETE binary, or a
        //      refusal, never a partial one; no .wronggpu-old and no temporary file stays
        {
            const std::string newBytes = ReadText(testDll2);
            const std::wstring out = JoinPath(root, L"loadloop"), go = JoinPath(root, L"go-loadloop");
            // the old file loaded once before the guard (the loop's first load), then the loop runs against the guard
            OperationGuard g(game, bins, 1000);
            const bool held = g.ok() && g.Holds(bin);
            HANDLE k = Spawn(L"--child-loadloop " + Q(bin) + L" " + Q(out) + L" " + Q(go) + L" 3000");
            Sleep(500);  // the loop is running (refused while the old file is held)
            WriteText(go, "go");
            std::wstring oldLeft;
            DWORD e = ERROR_SUCCESS;
            for (int i = 0; i < 20 && e == ERROR_SUCCESS; ++i) {  // replace back and forth: 220, 219, 220, ...
                const std::string& bytes = (i % 2 == 0) ? newBytes : ReadText(testDll);
                const std::string copy = bytes;
                e = g.PutHeld(bin, copy.data(), copy.size(), &oldLeft);
                Sleep(25);
            }
            const DWORD replaced = e;
            if (k) WaitExit(k, 60000);
            const std::string r = ReadText(out);  // "loads=<n> values=<v...> refused=<n> other=<codes or ->"
            bool onlyComplete = r.find("other=-") != std::string::npos;
            for (const char* bad : {"values=0", "partial"}) onlyComplete = onlyComplete && r.find(bad) == std::string::npos;
            std::printf("  (load loop during replaces: %s; last PutHeld %lu)\n", r.c_str(), replaced);
            check(held && (replaced == ERROR_SUCCESS || replaced == ERROR_SHARING_VIOLATION) && !r.empty() && onlyComplete,
                  "replaces while another process keeps loading the file: every load got a complete old or new binary, or a refusal; never a partial file");
            bool leftovers = false;
            for (auto& name : List(plugins)) leftovers = leftovers || name.find(L".wronggpu-old") != std::wstring::npos || IsTempName(name);
            check(oldLeft.empty() && !leftovers, "after the replaces: no .wronggpu-old and no temporary file left (the old files were deleted through their handles)");
            DeleteFileW(out.c_str());
            DeleteFileW(go.c_str());
        }
        // a .wronggpu-old name already taken (a file of an earlier crash): the next free name is used, the earlier file stays
        {
            const std::wstring taken = bin + L".wronggpu-old";
            WriteText(taken, "an earlier file");
            const std::string newBytes = ReadText(testDll2);
            OperationGuard g(game, bins, 1000);
            std::wstring oldLeft;
            const DWORD e = g.PutHeld(bin, newBytes.data(), newBytes.size(), &oldLeft);
            check(g.ok() && e == ERROR_SUCCESS && ReadText(bin) == newBytes && ReadText(taken) == "an earlier file" && !Exists(bin + L".wronggpu-old-1") && oldLeft.empty(),
                  "a taken .wronggpu-old name: the next free name used and cleaned up, the earlier file untouched (nothing deleted by name)");
            DeleteFileW(taken.c_str());
        }
        CopyFileW(testDll.c_str(), bin.c_str(), FALSE);
        // a binary loaded in another process: GameRunning, nothing held; that process killed: Ok
        const std::wstring ready = JoinPath(root, L"ready-load");
        HANDLE k = Spawn(L"--child-load " + Q(bin) + L" " + Q(ready));
        const bool up = k && WaitFile(ready) && ReadText(ready) == "loaded";
        {
            OperationGuard g(game, bins, 1000);
            check(up && g.result() == GuardResult::GameRunning && g.error() == ERROR_SHARING_VIOLATION && g.which() == bin && !g.Holds(bin2) &&
                      std::string(g.words()).find("in use by the game or another program") != std::string::npos,
                  "a binary loaded in another process: GameRunning (sharing violation, names the file), nothing held");
        }
        if (k) { TerminateProcess(k, 9); WaitForSingleObject(k, 30000); CloseHandle(k); }
        DeleteFileW(ready.c_str());
        check(OperationGuard(game, bins, 1000).ok(), "that process ended (killed): Ok again, nothing stale");
        // loaded in this process
        {
            const HMODULE m = LoadLibraryW(bin2.c_str());
            OperationGuard g(game, bins, 1000);
            check(m && g.result() == GuardResult::GameRunning && g.which() == bin2, "a binary loaded in this process: GameRunning");
            if (m) FreeLibrary(m);
        }
        // a binary this process may not open: Unknown (refuse), never Ok
        {
            PSECURITY_DESCRIPTOR deny = nullptr, allow = nullptr;
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(D;;FA;;;WD)", SDDL_REVISION_1, &deny, nullptr);
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;FA;;;WD)", SDDL_REVISION_1, &allow, nullptr);
            const bool set = deny && SetFileSecurityW(bin2.c_str(), DACL_SECURITY_INFORMATION, deny);
            GuardResult gr;
            DWORD ge;
            {
                OperationGuard g(game, bins, 1000);
                gr = g.result();
                ge = g.error();
            }
            if (allow) SetFileSecurityW(bin2.c_str(), DACL_SECURITY_INFORMATION, allow);  // the owner may always change the DACL back
            for (PSECURITY_DESCRIPTOR p : {deny, allow}) if (p) LocalFree(p);
            check(set && gr == GuardResult::Unknown && ge == ERROR_ACCESS_DENIED, "a binary the guard may not open (access denied): Unknown, the operation refuses");
        }
        // the state lock busy: Busy (the binaries are not even opened)
        {
            const std::wstring ready2 = JoinPath(root, L"ready-hold2");
            HANDLE h2 = Spawn(L"--child-hold " + Q(game) + L" 2000 " + Q(ready2));
            const bool up2 = h2 && WaitFile(ready2);
            OperationGuard* g = new OperationGuard(game, bins, 200);
            check(up2 && g->result() == GuardResult::Busy && !g->Holds(bin), "the state lock held elsewhere: Busy, no binary held");
            delete g;
            if (h2) WaitExit(h2);
            DeleteFileW(ready2.c_str());
        }
        check(OperationGuard(JoinPath(root, L"gone"), {}, 1000).ok(), "a game folder that is gone: nothing to guard (Ok)");
        DeleteFileW(bin.c_str());
        DeleteFileW(bin2.c_str());
        RemoveDirectoryW(plugins.c_str());
    }

    // cleanup (own files only)
    for (const auto& n : List(state)) DeleteFileW(JoinPath(state, n).c_str());
    RemoveDirectoryW(state.c_str());
    RemoveDirectoryW(JoinPath(root, L"GameJunction").c_str());  // the junction itself (its target is not touched)
    RemoveDirectoryW(game.c_str());
    RemoveDirectoryW(root.c_str());
    std::printf("wg_state: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
