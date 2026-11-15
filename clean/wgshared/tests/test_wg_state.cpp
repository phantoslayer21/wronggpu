// SPDX-License-Identifier: MIT
// Tests of wg_state.h (the per-game state folder, its lock, the loaded protocol and atomic writes) on real files and real
// processes. The test runs itself as the child process (--child-...). Scratch files go to the TMP folder of the run.
//  - the strict settings.ini format (a bare or embedded CR is corrupt);
//  - atomic writes: oversized input refused, a failure at every step after the temporary file exists (write, flush, close,
//    move) leaves the target as it was and no temporary file, a cleanup that fails is reported;
//  - SetSettings: no state folder, a folder this process may not write (AccessDenied, its own result), unreadable, corrupt;
//  - one lock name for several aliases of one folder (case, trailing backslash, 8.3 name, \\?\ form, a junction, a path longer
//    than MAX_PATH), each computed in a SEPARATE process, and mutual exclusion across processes that use different aliases;
//  - a busy lock (bounded wait, nothing written), a holder that dies in the middle of a write (abandoned: the next holder
//    re-reads the old whole file; the half-written temporary file is never read as state);
//  - the loaded protocol: QueryGameUse NotRunning / Running / Unknown (lookup errors are Unknown, never NotRunning), a
//    registered child, a killed child (no stale marker), the launcher's OperationGuard refusing on Running and Unknown, a
//    registration that waits for a running operation and gets in only after it, and one that gives up in time.
// Written for this project.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "wg_state.h"

using namespace wgstate;

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
    do { if (wcscmp(f.cFileName, L".") && wcscmp(f.cFileName, L"..")) out.push_back(f.cFileName); } while (FindNextFileW(h, &f));
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
static ULONGLONG Now() { return GetTickCount64(); }

// ---- child modes
static int Child(int argc, wchar_t** argv) {
    const std::wstring mode = argv[1], game = argv[2];
    if (mode == L"--child-add" && argc >= 4) {  // n locked read-modify-write increments of one counter
        const int n = _wtoi(argv[3]);
        for (int i = 0; i < n; ++i) {
            StateLock lock(game, 30000);
            if (!lock.held()) return 3;
            const Settings s = ReadSettings(game);  // re-read after the lock is taken
            const std::string* v = Get(s, "counter");
            const unsigned long c = v ? std::strtoul(v->c_str(), nullptr, 10) : 0;
            Settings w = s;
            Set(w, "counter", std::to_string(c + 1));
            Sleep(0);  // widen the window a lost update would need
            if (!WriteAtomic(SettingsPathOf(game), RenderSettings(w))) return 4;
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
    if (mode == L"--child-register" && argc >= 5) {  // the game's registration (wait ms), then stays loaded until killed
        const ULONGLONG t0 = Now();
        LoadedRegistration reg(game, static_cast<DWORD>(_wtoi(argv[3])));
        WriteText(argv[4], (reg.ok() ? "ok " : "no ") + std::to_string(Now() - t0) + " " + reg.why());
        if (!reg.ok()) return 0;
        Sleep(120000);
        return 0;
    }
    return 2;
}

int wmain(int argc, wchar_t** argv) {
    if (argc >= 3 && std::wstring(argv[1]).rfind(L"--child", 0) == 0) return Child(argc, argv);
    const std::wstring root = TempRoot();
    const std::wstring game = JoinPath(root, L"Game");
    CreateDirectoryW(game.c_str(), nullptr);

    // ---- the strict format
    {
        Settings s = ParseSettings("; comment\r\n\r\nrr_mode=fast\r\nrr_enabled=1\n");
        check(s.present && !s.corrupt && s.kv.size() == 2 && *Get(s, "rr_mode") == "fast" && *Get(s, "rr_enabled") == "1", "parse: comments, blank lines, CRLF and LF");
        const char* bad[] = {"rr_mode = fast\r\n", "[wronggpu]\r\nrr_mode=fast\r\n", "RR_MODE=fast\r\n", "rr_mode=fast\r\nrr_mode=accurate\r\n",
                             "rr_mode\r\n", "rr_mode=fa st\r\n", "=fast\r\n", "rr_mode=f\xc3\xa4st\r\n", "; \xc3\xa4\r\n",
                             "rr_mode=fast\r", "rr_mode=fa\rst\r\n", "rr_mode=fast\r\r\n", "; a\rb\n"};
        bool all = true;
        for (const char* b : bad) { const Settings x = ParseSettings(b); all = all && x.corrupt && x.kv.empty() && !x.why.empty(); }
        check(all, "parse: spaces, sections, upper case, a key twice, no '=', bad value, empty key, non-ASCII, a bare or embedded CR: corrupt");
        check(!ParseSettings("rr_mode=fast").corrupt, "parse: a last line without a line end is fine");
        Settings r;
        check(Set(r, "rr_mode", "fast") && Set(r, "rr_mode", "exact") && r.kv.size() == 1 && !Set(r, "Bad Key", "x") && !Set(r, "k", "a b"), "Set: replace, refuse bad key/value");
        const Settings back = ParseSettings(RenderSettings(r));
        check(!back.corrupt && back.kv == r.kv, "render -> parse round trip");
    }

    // ---- atomic writes
    const std::wstring state = StateDirOf(game);
    {
        check(SetSettings(game, {{"rr_mode", "fast"}}) == SetResult::NoStateFolder && DirectoryState(state) == PathState::Missing,
              "no state folder: NoStateFolder, the folder is not created");
        CreateDirectoryW(state.c_str(), nullptr);
        const std::wstring f = SettingsPathOf(game);
        check(SetSettings(game, {{"rr_mode", "fast"}}) == SetResult::Ok && *Get(ReadSettings(game), "rr_mode") == "fast", "SetSettings writes the file");
        const std::string before = ReadText(f);
        DWORD e = 0;
        check(!WriteAtomic(f, std::string(kMaxStateBytes + 1, 'x'), &e) && e == ERROR_FILE_TOO_LARGE && ReadText(f) == before, "over kMaxStateBytes: refused, the file unchanged");
        bool steps = true;
        for (const char* step : {"write", "flush", "close", "move"}) {
            static const char* failAt = nullptr;
            failAt = step;
            TestFault() = [](const char* s) { return std::strcmp(s, failAt) == 0; };
            std::wstring left;
            e = 0;
            const bool ok = WriteAtomic(f, "rr_mode=exact\r\n", &e, &left);
            TestFault() = nullptr;
            const bool good = !ok && e == ERROR_WRITE_FAULT && left.empty() && ReadText(f) == before && List(state).size() == 1;
            if (!good) std::printf("  (failure at %s: ok %d, error %lu, left %d, files %zu)\n", step, ok, e, !left.empty(), List(state).size());
            steps = steps && good;
        }
        check(steps, "a failure at write / flush / close / move (after the temporary file exists): false, the file unchanged, no temporary file left");
        TestFault() = [](const char* s) { return std::strcmp(s, "move") == 0 || std::strcmp(s, "delete") == 0; };
        std::wstring left;
        const bool ok = WriteAtomic(f, "rr_mode=exact\r\n", &e, &left);
        TestFault() = nullptr;
        check(!ok && !left.empty() && Exists(left) && IsTempName(left) && ReadText(f) == before, "a cleanup that fails: reported (tempLeft), the file unchanged");
        if (!left.empty()) DeleteFileW(left.c_str());
        check(!WriteAtomic(JoinPath(JoinPath(root, L"missing-folder"), L"x.ini"), "x", &e) && e != 0, "a write into a missing folder fails with its error");
        // corrupt, unreadable
        WriteText(f, "rr_mode = fast\r\n");
        check(ReadSettings(game).corrupt && ReadSettings(game).kv.empty(), "a corrupt file reads as corrupt, no keys");
        check(SetSettings(game, {{"rr_mode", "accurate"}}) == SetResult::Ok && ReadSettings(game).kv.size() == 1, "a corrupt file is replaced by the next write");
        const std::string b2 = ReadText(f);
        HANDLE h = CreateFileW(f.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD err = 0;
        const SetResult r = SetSettings(game, {{"rr_mode", "exact"}}, kDefaultWaitMs, &err);
        CloseHandle(h);
        check(r == SetResult::Unreadable && err != 0 && ReadText(f) == b2 && List(state).size() == 1, "an unreadable file: Unreadable, left as it was");
        // a state folder this process may not write: AccessDenied (its own result, for the elevated repair)
        PSECURITY_DESCRIPTOR ro = nullptr, rw = nullptr;
        ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;0x1200a9;;;WD)", SDDL_REVISION_1, &ro, nullptr);  // read and list only
        ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;WD)", SDDL_REVISION_1, &rw, nullptr);
        const bool locked = ro && SetFileSecurityW(state.c_str(), DACL_SECURITY_INFORMATION, ro);
        err = 0;
        const SetResult denied = SetSettings(game, {{"rr_mode", "exact"}}, kDefaultWaitMs, &err);
        if (rw) SetFileSecurityW(state.c_str(), DACL_SECURITY_INFORMATION, rw);  // the owner may always change the DACL back
        if (ro) LocalFree(ro);
        if (rw) LocalFree(rw);
        check(locked && denied == SetResult::AccessDenied && err == ERROR_ACCESS_DENIED && ReadText(f) == b2, "a folder this process may not write: AccessDenied, nothing written");
        check(std::string(SetResultWords(SetResult::AccessDenied)).find("administrator") != std::string::npos, "AccessDenied's words name the elevated repair");
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
        check(LockNameFor(deep, &dn) && Narrow(dn) != names[0] && SetSettings(deep, {{"k", "v"}}) == SetResult::Ok && *Get(ReadSettings(deep), "k") == "v",
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
        for (int i = 0; i < 4; ++i) kids.push_back(Spawn(L"--child-add " + Q(aliases[(i * 2 + 1) % aliases.size()]) + L" 50"));
        bool ok = true;
        for (HANDLE k : kids) ok = k && WaitExit(k, 120000) == 0 && ok;
        check(ok && Counter(game) == 200, "4 processes, 4 different aliases, 50 locked increments each: 200 (one lock for all spellings)");
        check(List(state).size() == 1, "no temporary file stays after concurrent writes");
    }

    // ---- busy: a bounded wait gives up, nothing written
    {
        const std::wstring ready = JoinPath(root, L"ready-hold");
        HANDLE k = Spawn(L"--child-hold " + Q(game) + L" 3000 " + Q(ready));
        const bool up = k && WaitFile(ready);
        const std::string before = ReadText(SettingsPathOf(game));
        DWORD err = 0;
        const SetResult r = SetSettings(game, {{"rr_mode", "fast"}}, 200, &err);
        check(up && r == SetResult::Busy && err == WAIT_TIMEOUT && ReadText(SettingsPathOf(game)) == before, "lock held elsewhere: Busy after the bounded wait, nothing written");
        if (k) WaitExit(k);
        DeleteFileW(ready.c_str());
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
            check(up && keep && lock.held() && lock.abandoned() && !s.corrupt && *Get(s, "counter") == "7",
                  "a holder that died mid-write: the lock is taken (abandoned), the state re-read is the old whole file");
            std::wstring temp;
            for (auto& n : List(state)) if (IsTempName(n)) temp = n;
            check(!temp.empty(), "the half-written temporary file is there and named as a temporary file (never read as state)");
            check(SetSettings(game, {{"counter", "8"}}) == SetResult::Ok && *Get(ReadSettings(game), "counter") == "8", "a write under the abandoned-then-taken lock works");
            if (!temp.empty()) DeleteFileW(JoinPath(state, temp).c_str());
        }
        if (keep) CloseHandle(keep);
        DeleteFileW(ready.c_str());
        DeleteFileW(go.c_str());
    }

    // ---- the loaded protocol
    {
        DWORD err = 0;
        check(QueryGameUse(game, &err) == GameUse::NotRunning && QueryGameUse(game) == GameUse::NotRunning, "no marker: NotRunning (and the query creates none)");
        check(QueryGameUse(JoinPath(root, L"gone")) == GameUse::NotRunning, "a folder that does not exist: NotRunning (nothing can run from it)");
        {
            LoadedRegistration reg(game);
            check(reg.ok() && QueryGameUse(aliases.back()) == GameUse::Running, "registered in this process: Running (asked through another alias)");
        }
        check(QueryGameUse(game) == GameUse::NotRunning, "the registration ended: NotRunning");
        // a lookup error is Unknown, never NotRunning: an object of another type under the marker's name
        {
            std::wstring markerName;
            NameFor(kLoadedPrefix, game, &markerName, nullptr);
            HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, markerName.c_str());
            err = 0;
            const GameUse u = QueryGameUse(game, &err);
            OperationGuard g(game, 1000);
            check(ev && u == GameUse::Unknown && err != 0 && g.result() == GuardResult::Unknown && std::string(g.words()).find("could not tell") != std::string::npos,
                  "a marker lookup that fails: Unknown, and the launcher's guard refuses (Unknown)");
            if (ev) CloseHandle(ev);
        }
        // a game folder this process cannot open: Unknown, the guard refuses
        {
            const std::wstring locked = JoinPath(root, L"Locked");
            CreateDirectoryW(locked.c_str(), nullptr);
            PSECURITY_DESCRIPTOR deny = nullptr, allow = nullptr, parentDeny = nullptr, parentAllow = nullptr;
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(D;;FA;;;WD)", SDDL_REVISION_1, &deny, nullptr);
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;WD)", SDDL_REVISION_1, &allow, nullptr);
            // the parent must not grant listing either (listing a folder lets its entries' attributes be read)
            const std::wstring parent = JoinPath(root, L"LockedParent");
            CreateDirectoryW(parent.c_str(), nullptr);
            const std::wstring inner = JoinPath(parent, L"Game");
            CreateDirectoryW(inner.c_str(), nullptr);
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(D;;0x1;;;WD)(A;;FA;;;WD)", SDDL_REVISION_1, &parentDeny, nullptr);
            ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;OICI;FA;;;WD)", SDDL_REVISION_1, &parentAllow, nullptr);
            const bool set = deny && parentDeny && SetFileSecurityW(inner.c_str(), DACL_SECURITY_INFORMATION, deny) &&
                             SetFileSecurityW(parent.c_str(), DACL_SECURITY_INFORMATION, parentDeny);
            err = 0;
            const GameUse u = QueryGameUse(inner, &err);
            const GuardResult gr = OperationGuard(inner, 500).result();
            if (allow) SetFileSecurityW(inner.c_str(), DACL_SECURITY_INFORMATION, allow);
            if (parentAllow) SetFileSecurityW(parent.c_str(), DACL_SECURITY_INFORMATION, parentAllow);
            for (PSECURITY_DESCRIPTOR p : {deny, allow, parentDeny, parentAllow}) if (p) LocalFree(p);
            if (!(set && u == GameUse::Unknown && err == ERROR_ACCESS_DENIED && gr == GuardResult::Unknown))
                std::printf("  (access-denied lookup: set %d, use %d, error %lu, guard %d)\n", set, static_cast<int>(u), err, static_cast<int>(gr));
            check(set && u == GameUse::Unknown && err == ERROR_ACCESS_DENIED && gr == GuardResult::Unknown,
                  "a game folder whose identity cannot be read (access denied): Unknown, the guard refuses");
            RemoveDirectoryW(inner.c_str());
            RemoveDirectoryW(parent.c_str());
            RemoveDirectoryW(locked.c_str());
        }
        // a registered game in another process; then it crashes
        const std::wstring out = JoinPath(root, L"reg-1");
        HANDLE k = Spawn(L"--child-register " + Q(game) + L" 2000 " + Q(out));
        const bool up = k && WaitFile(out) && ReadText(out).rfind("ok", 0) == 0;
        OperationGuard* g = new OperationGuard(game, 1000);
        check(up && QueryGameUse(game) == GameUse::Running && g->result() == GuardResult::GameRunning && std::string(g->words()).find("close the game") != std::string::npos,
              "a game registered in another process: Running, the guard refuses (GameRunning)");
        delete g;
        if (k) {
            TerminateProcess(k, 9);  // the game crashes
            WaitForSingleObject(k, 30000);
            CloseHandle(k);
        }
        check(up && QueryGameUse(game) == GameUse::NotRunning && OperationGuard(game, 1000).result() == GuardResult::Ok,
              "the game died: no stale marker, NotRunning, the guard lets the operation run");
        DeleteFileW(out.c_str());
    }

    // ---- registration against a running operation: no slipping in between
    {
        const std::wstring out = JoinPath(root, L"reg-2");
        ULONGLONG released = 0;
        HANDLE k = nullptr;
        {
            OperationGuard op(game, 1000);  // the launcher's operation starts (lock held, no game)
            const bool ok = op.result() == GuardResult::Ok;
            k = Spawn(L"--child-register " + Q(game) + L" 2000 " + Q(out));  // the game starts during it
            Sleep(800);
            check(ok && !Exists(out) && QueryGameUse(game) == GameUse::NotRunning, "during the operation the starting game cannot register (it waits; nothing is loaded)");
            released = Now();
        }  // the operation ends: the lock is released
        const bool got = k && WaitFile(out, 10000);
        const std::string r = ReadText(out);
        check(got && r.rfind("ok", 0) == 0 && QueryGameUse(game) == GameUse::Running, "after the operation the game registers (it waited, bounded)");
        check(released > 0 && std::strtoul(r.c_str() + 3, nullptr, 10) >= 500, "the registration came only after the operation released the lock");
        if (k) { TerminateProcess(k, 9); WaitForSingleObject(k, 30000); CloseHandle(k); }
        DeleteFileW(out.c_str());
        // a registration that cannot wait that long gives up in time and touches nothing
        const std::wstring out2 = JoinPath(root, L"reg-3");
        HANDLE k2 = nullptr;
        {
            OperationGuard op(game, 1000);
            k2 = Spawn(L"--child-register " + Q(game) + L" 300 " + Q(out2));
            const bool got2 = k2 && WaitFile(out2, 10000);
            const std::string r2 = ReadText(out2);
            check(op.result() == GuardResult::Ok && got2 && r2.rfind("no", 0) == 0 && r2.find("not free in time") != std::string::npos &&
                      std::strtoul(r2.c_str() + 3, nullptr, 10) < 1500 && QueryGameUse(game) == GameUse::NotRunning,
                  "a registration while an operation runs gives up after its bounded wait: no marker, defaults, nothing blocked");
        }
        if (k2) WaitExit(k2);
        DeleteFileW(out2.c_str());
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
