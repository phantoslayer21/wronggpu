// SPDX-License-Identifier: MIT
// Tests of wg_state.h (the per-game state folder, its lock and atomic writes) on real files and real processes: the strict
// settings.ini format, atomic writes and their cleanup, one lock name for every spelling of a folder, mutual exclusion
// across processes (concurrent read-modify-write of one counter from several child processes loses no update), a busy
// lock (bounded wait), an abandoned lock (a child that dies holding it), an unreadable file never overwritten, a corrupt
// file replaced, and no state folder created. The test runs itself as the child process (--child ...).
// Written for this project.
#include <cstdio>
#include <cstdlib>
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
    HANDLE h = CreateFileW(p.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    DWORD put = 0;
    if (h != INVALID_HANDLE_VALUE) { WriteFile(h, t.data(), static_cast<DWORD>(t.size()), &put, nullptr); CloseHandle(h); }
}
static std::string ReadText(const std::wstring& p) { std::string t; ReadBytes(p, &t); return t; }
static std::vector<std::wstring> List(const std::wstring& dir) {
    std::vector<std::wstring> out;
    WIN32_FIND_DATAW f;
    HANDLE h = FindFirstFileW(JoinPath(dir, L"*").c_str(), &f);
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
static unsigned long Counter(const std::wstring& game) {
    const Settings s = ReadSettings(game);
    const std::string* v = Get(s, "counter");
    return v ? std::strtoul(v->c_str(), nullptr, 10) : 0;
}

static bool WaitFile(const std::wstring& p, DWORD ms = 30000) {
    for (DWORD t = 0; t < ms; t += 20) {
        if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) return true;
        Sleep(20);
    }
    return false;
}

// child modes: --child-add <game> <n>: n locked read-modify-write increments; --child-hold <game> <ms> <ready file>: holds the
// lock for ms; --child-die <game> <ready file> <go file>: takes the lock and, once go exists, ends without releasing it
static int Child(int argc, wchar_t** argv) {
    const std::wstring mode = argv[1], game = argv[2];
    if (mode == L"--child-add" && argc >= 4) {
        const int n = _wtoi(argv[3]);
        for (int i = 0; i < n; ++i) {
            StateLock lock(game, 30000);
            if (!lock.held()) return 3;
            const Settings s = ReadSettings(game);
            const std::string* v = Get(s, "counter");
            const unsigned long c = v ? std::strtoul(v->c_str(), nullptr, 10) : 0;
            Settings w = s;
            Set(w, "counter", std::to_string(c + 1));
            Sleep(0);  // widen the window a lost update would need
            if (!WriteAtomic(SettingsPathOf(game), RenderSettings(w))) return 4;
        }
        return 0;
    }
    if (mode == L"--child-loaded" && argc >= 4) {  // a "game" with WrongGPU loaded: the marker open until it is killed
        LoadedMarker marker(game);
        if (!marker.held()) return 3;
        WriteText(argv[3], "ready");
        Sleep(120000);
        return 0;
    }
    if (mode == L"--child-hold" && argc >= 5) {
        StateLock lock(game, 30000);
        if (!lock.held()) return 3;
        WriteText(argv[4], "ready");
        Sleep(static_cast<DWORD>(_wtoi(argv[3])));
        return 0;
    }
    if (mode == L"--child-die" && argc >= 5) {
        StateLock* lock = new StateLock(game, 30000);  // never released: the process ends holding it
        if (!lock->held()) return 3;
        WriteText(argv[3], "ready");
        WaitFile(argv[4]);  // the parent has its own handle now
        ExitProcess(0);
    }
    return 2;
}

int wmain(int argc, wchar_t** argv) {
    if (argc >= 3 && std::wstring(argv[1]).rfind(L"--child", 0) == 0) return Child(argc, argv);
    const std::wstring root = TempRoot();

    // ---- the strict format
    {
        Settings s = ParseSettings("; comment\r\n\r\nrr_mode=fast\r\nrr_enabled=1\n");
        check(s.present && !s.corrupt && s.kv.size() == 2 && *Get(s, "rr_mode") == "fast" && *Get(s, "rr_enabled") == "1", "parse: comments, blank lines, CRLF and LF");
        const char* bad[] = {"rr_mode = fast\r\n", "[wronggpu]\r\nrr_mode=fast\r\n", "RR_MODE=fast\r\n", "rr_mode=fast\r\nrr_mode=accurate\r\n",
                             "rr_mode\r\n", "rr_mode=fa st\r\n", "=fast\r\n", "rr_mode=f\xc3\xa4st\r\n", "; \xc3\xa4\r\n"};
        bool all = true;
        for (const char* b : bad) { const Settings x = ParseSettings(b); all = all && x.corrupt && x.kv.empty() && !x.why.empty(); }
        check(all, "parse: spaces, sections, upper case, a key twice, no '=', bad value, empty key, non-ASCII: corrupt, no keys");
        Settings r;
        check(Set(r, "rr_mode", "fast") && Set(r, "rr_mode", "exact") && r.kv.size() == 1 && !Set(r, "Bad Key", "x") && !Set(r, "k", "a b"), "Set: replace, refuse bad key/value");
        const Settings back = ParseSettings(RenderSettings(r));
        check(!back.corrupt && back.kv == r.kv, "render -> parse round trip");
    }

    // ---- atomic writes and reads on real files
    const std::wstring game = JoinPath(root, L"Game");
    CreateDirectoryW(game.c_str(), nullptr);
    {
        check(ReadSettings(game).present == false && ReadSettings(game).why.empty(), "no settings.ini: not present, no error");
        check(SetSettings(game, {{"rr_mode", "fast"}}) == SetResult::NoStateFolder && !IsDirectory(StateDirOf(game)),
              "no state folder: NoStateFolder, the folder is not created");
        CreateDirectoryW(StateDirOf(game).c_str(), nullptr);
        check(SetSettings(game, {{"rr_mode", "fast"}}) == SetResult::Ok && *Get(ReadSettings(game), "rr_mode") == "fast", "SetSettings writes the file");
        check(SetSettings(game, {{"rr_enabled", "0"}}) == SetResult::Ok && *Get(ReadSettings(game), "rr_mode") == "fast" &&
                  *Get(ReadSettings(game), "rr_enabled") == "0", "a second key keeps the first");
        check(SetSettings(game, {{"rr mode", "x"}}) == SetResult::Invalid, "an invalid key: Invalid, nothing written");
        std::vector<std::wstring> names = List(StateDirOf(game));
        check(names.size() == 1 && names[0] == kSettingsFile, "no temporary file stays after writes");
        DWORD e = 0;
        check(!WriteAtomic(JoinPath(JoinPath(root, L"missing-folder"), L"x.ini"), "x", &e) && e != 0, "a write into a missing folder fails with its error");
        // corrupt: readers see no keys; the next write replaces the file
        WriteText(SettingsPathOf(game), "rr_mode = fast\r\n");
        const Settings c = ReadSettings(game);
        check(c.present && c.corrupt && c.kv.empty(), "a corrupt file reads as corrupt, no keys");
        check(SetSettings(game, {{"rr_mode", "accurate"}}) == SetResult::Ok && !ReadSettings(game).corrupt && *Get(ReadSettings(game), "rr_mode") == "accurate" &&
                  ReadSettings(game).kv.size() == 1, "a corrupt file is replaced by the next write (only the new key)");
        // unreadable (held without sharing): never overwritten
        const std::string before = ReadText(SettingsPathOf(game));
        HANDLE h = CreateFileW(SettingsPathOf(game).c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD err = 0;
        const SetResult r = SetSettings(game, {{"rr_mode", "exact"}}, kDefaultWaitMs, &err);
        CloseHandle(h);
        check(r == SetResult::Unreadable && err != 0 && ReadText(SettingsPathOf(game)) == before, "an unreadable file: Unreadable, left as it was");
        names = List(StateDirOf(game));
        check(names.size() == 1, "no temporary file stays after the refused write");
    }

    // ---- one lock name for every spelling of the folder
    {
        std::wstring a, b, c, d;
        check(LockNameFor(game, &a) && LockNameFor(game + L"\\", &b) && a == b && a.rfind(L"Local\\WrongGPU-state-", 0) == 0, "lock name: trailing backslash");
        std::wstring upper = game;
        for (auto& ch : upper) ch = static_cast<wchar_t>(towupper(ch));
        check(LockNameFor(upper, &c) && a == c, "lock name: another case");
        wchar_t shortp[MAX_PATH * 2];
        if (GetShortPathNameW(game.c_str(), shortp, MAX_PATH * 2)) check(LockNameFor(shortp, &d) && a == d, "lock name: the short (8.3) path");
        std::wstring other;
        const std::wstring g2 = JoinPath(root, L"Other");
        CreateDirectoryW(g2.c_str(), nullptr);
        check(LockNameFor(g2, &other) && other != a, "lock name: another folder, another name");
        DWORD e = 0;
        check(!LockNameFor(JoinPath(root, L"nope"), &other, &e) && e != 0, "lock name: a missing folder fails");
    }

    // ---- mutual exclusion across processes: 4 children x 50 locked increments of one counter, no update lost
    {
        WriteText(SettingsPathOf(game), "counter=0\r\n");
        std::vector<HANDLE> kids;
        for (int i = 0; i < 4; ++i) kids.push_back(Spawn(L"--child-add \"" + game + L"\" 50"));
        bool started = true, ok = true;
        for (HANDLE k : kids) { if (!k) started = false; else ok = WaitExit(k, 120000) == 0 && ok; }
        check(started && ok, "4 child processes ran their locked increments");
        check(Counter(game) == 200, "the counter is 200: the lock serialised every read-modify-write across processes");
        check(List(StateDirOf(game)).size() == 1, "no temporary file stays after concurrent writes");
    }

    // ---- busy: another process holds the lock; a bounded wait gives up (Busy), the file unchanged
    {
        const std::wstring ready = JoinPath(root, L"ready-hold");
        HANDLE k = Spawn(L"--child-hold \"" + game + L"\" 3000 \"" + ready + L"\"");
        const bool up = k && WaitFile(ready);
        const std::string before = ReadText(SettingsPathOf(game));
        DWORD err = 0;
        const SetResult r = SetSettings(game, {{"rr_mode", "fast"}}, 200, &err);
        check(up && r == SetResult::Busy && err == WAIT_TIMEOUT && ReadText(SettingsPathOf(game)) == before, "lock held elsewhere: Busy after the bounded wait, nothing written");
        if (k) WaitExit(k);
        check(SetSettings(game, {{"rr_mode", "fast"}}, 5000) == SetResult::Ok, "after the holder ends: the write goes through");
    }

    // ---- abandoned: a child dies holding the lock; the next waiter gets it
    {
        const std::wstring ready = JoinPath(root, L"ready-die");
        const std::wstring go = JoinPath(root, L"go-die");
        HANDLE k = Spawn(L"--child-die \"" + game + L"\" \"" + ready + L"\" \"" + go + L"\"");
        const bool up = k && WaitFile(ready);
        std::wstring name;
        LockNameFor(game, &name);
        HANDLE keep = OpenMutexW(SYNCHRONIZE, FALSE, name.c_str());  // keeps the mutex object alive past the child's end
        WriteText(go, "go");
        if (k) WaitExit(k);
        {
            StateLock lock(game, 5000);
            if (!(up && keep && lock.held() && lock.abandoned()))
                std::printf("  (up %d, kept handle %d, held %d, abandoned %d, error %lu)\n", up, keep != nullptr, lock.held(), lock.abandoned(), lock.error());
            check(up && keep && lock.held() && lock.abandoned(), "a holder that died: the lock is taken (abandoned), not stuck");
        }
        if (keep) CloseHandle(keep);
        StateLock again(game, 5000);
        check(again.held() && !again.abandoned(), "after that: the lock is taken normally");
    }

    // ---- the loaded marker: in this process, in another one, and a dead owner (killed) leaves none
    {
        const std::wstring g2 = JoinPath(root, L"Other");
        check(!IsGameUsingWrongGPU(game) && !IsGameUsingWrongGPU(game), "no marker: not in use (and the check creates none)");
        {
            LoadedMarker m(game);
            std::wstring upper = game;
            for (auto& ch : upper) ch = static_cast<wchar_t>(towupper(ch));
            check(m.held() && IsGameUsingWrongGPU(game) && IsGameUsingWrongGPU(upper + L"\\") && !IsGameUsingWrongGPU(g2),
                  "a marker in this process: in use (any spelling of the folder), another folder not");
        }
        check(!IsGameUsingWrongGPU(game), "the marker closed: not in use");
        check(!IsGameUsingWrongGPU(JoinPath(root, L"nope")), "a missing folder: not in use");
        const std::wstring ready = JoinPath(root, L"ready-loaded");
        HANDLE k = Spawn(L"--child-loaded \"" + game + L"\" \"" + ready + L"\"");
        const bool up = k && WaitFile(ready);
        check(up && IsGameUsingWrongGPU(game), "another process with the marker open: in use");
        if (k) {
            TerminateProcess(k, 9);  // the game crashes (no cleanup of its own)
            WaitForSingleObject(k, 30000);
            CloseHandle(k);
        }
        check(up && !IsGameUsingWrongGPU(game), "the owning process died: no stale marker, not in use");
        DeleteFileW(ready.c_str());
    }

    // cleanup (own files only)
    for (const auto& n : List(StateDirOf(game))) DeleteFileW(JoinPath(StateDirOf(game), n).c_str());
    RemoveDirectoryW(StateDirOf(game).c_str());
    RemoveDirectoryW(game.c_str());
    RemoveDirectoryW(JoinPath(root, L"Other").c_str());
    DeleteFileW(JoinPath(root, L"ready-hold").c_str());
    DeleteFileW(JoinPath(root, L"ready-die").c_str());
    DeleteFileW(JoinPath(root, L"go-die").c_str());
    RemoveDirectoryW(root.c_str());
    std::printf("wg_state: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
