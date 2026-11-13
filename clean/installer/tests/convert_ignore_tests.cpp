// SPDX-License-Identifier: MIT
// Unit tests of the converter search's ignore lists (wg_core/convert_ignore.h): the list format (version line, comments,
// a path, a path with its SHA-256, a hash alone, quotes, UTF-8 / UTF-16 LE, CRLF), bad lines (warned, skipped, the rest
// read), path matching (case, '/' and '\', '.' and '..', a hard link, a junction, a symbolic link when this account may
// make one), a hash that matches wherever the file is, the well-known list locations, and an empty list that leaves every
// candidate set exactly as it was. Synthetic files only, in the temp folder. Written for this project. Run by ctest
// ("convert-ignore").
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <thread>

#include "wg_core/convert_ignore.h"
#include "wg_core/model.h"
#include "wg_core/record.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
wstring TempFolder() {
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    wstring dir = Join(tmp, L"convert-ignore-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return FullPath(dir);
}
void File(const wstring& p, const string& text) { EnsureDir(Parent(p)); WriteFileAtomic(p, text.data(), text.size()); }
string U8(const wstring& w) { return ToUtf8(w); }
wstring Slashes(wstring s) { for (auto& c : s) if (c == L'\\') c = L'/'; return s; }
wstring Upper(wstring s) { for (auto& c : s) c = (wchar_t)towupper(c); return s; }
IgnoreLists Parse(const string& text) { IgnoreLists l; ParseIgnoreText(text, L"test.txt", l); return l; }
bool Has(const wstring& s, const wstring& part) { return s.find(part) != wstring::npos; }
// StartCleanup's folder move for the retry cases (record.h SetCleanupMoveForTest): the first g_moveFailFirst calls fail
// with g_moveError, later ones really move.
std::atomic<int> g_moveCalls{0}, g_moveFailFirst{0};
std::atomic<DWORD> g_moveError{0};
BOOL WINAPI FakeMove(LPCWSTR from, LPCWSTR to, DWORD flags) {
    if (++g_moveCalls <= g_moveFailFirst) { SetLastError(g_moveError); return FALSE; }
    return MoveFileExW(from, to, flags);
}
size_t CountIn(const string& s, const string& part) {
    size_t n = 0;
    for (size_t at = 0; (at = s.find(part, at)) != string::npos; ++at) ++n;
    return n;
}
}  // namespace

int main() {
    std::vector<wstring> links;  // removed (the link, never its target) before the folder goes
    wstring root;
    try {
        root = TempFolder();
        const wstring lib = Join(root, L"Library");
        const wstring a = Join(lib, L"Game A\\bin\\x64\\nvngx_dlss.dll"), b = Join(lib, L"Game B\\nvngx_dlss.dll"), c = Join(lib, L"Game C\\x\\nvngx_dlss.dll");
        const wstring chosen = Join(root, L"Chosen\\nvngx_dlss.dll");
        File(a, "synthetic file A");
        File(b, "synthetic file B");
        File(c, "synthetic file C");
        File(chosen, "synthetic chosen file");
        const wstring shaB = Lower(Sha256File(b));
        require(shaB.size() == 64, "the test's SHA-256");

        // ---- the format --------------------------------------------------------------------------------------------
        {
            const string text = "wronggpu-ignore-files 1\r\n# placed by an installer\r\n\r\n   \r\n" + U8(a) + "\r\n" + U8(b) + "  sha256=" + U8(Upper(shaB)) +
                                "\r\nsha256=" + string(64, 'a') + "\r\n\"" + U8(c) + "\"\r\n";
            IgnoreLists l = Parse(text);
            require(l.warnings.empty() && l.entries.size() == 4, "a list with a version line, comments, blank lines, a path, a path with its hash, a hash alone, a quoted path");
            require(l.entries[0].path == a && l.entries[0].sha256.empty() && Has(l.entries[0].where, L"line 5"), "a path entry, where it is written");
            require(l.entries[1].path == b && l.entries[1].sha256 == shaB, "a path with its SHA-256 (hex read in either case)");
            require(l.entries[2].path.empty() && l.entries[2].sha256 == wstring(64, L'a'), "a hash alone");
            require(l.entries[3].path == c, "quotes around a path");
            IgnoreLists lf = Parse("\xEF\xBB\xBF" + U8(a) + "\n" + U8(b) + "\n");
            require(lf.warnings.empty() && lf.entries.size() == 2 && lf.entries[0].path == a, "UTF-8 with a BOM, LF line ends");
            const wstring w16 = L"\xFEFF" + a + L"\r\nsha256=" + shaB + L"\r\n";
            string bytes(w16.size() * 2, '\0');
            std::memcpy(&bytes[0], w16.data(), bytes.size());
            IgnoreLists lw = Parse(string("\xFF\xFE", 2) + bytes.substr(2));
            require(lw.warnings.empty() && lw.entries.size() == 2 && lw.entries[0].path == a && lw.entries[1].sha256 == shaB, "UTF-16 LE with its BOM");
            IgnoreLists ln = Parse("wronggpu-ignore-files 7\n" + U8(a) + "\n");
            require(ln.entries.size() == 1 && ln.warnings.size() == 1 && Has(ln.warnings[0], L"newer"), "a newer format version: said once, its lines still read");
        }
        // ---- bad lines: each skipped with a warning naming the list and the line; the rest is read --------------------
        {
            const string text = "Game A\\nvngx_dlss.dll\n" + U8(a) + "\nC:nvngx_dlss.dll\nsha256=1234\n" + U8(b) + " sha256=" + string(63, 'f') + "\n" +
                                "sha256=" + string(63, 'f') + "g\n" + "C:\\Games\\*\\nvngx_dlss.dll\nwronggpu-ignore-files one\n" + U8(c) + "\n";
            IgnoreLists l = Parse(text);
            require(l.entries.size() == 2 && l.entries[0].path == a && l.entries[1].path == c, "the good lines between and after the bad ones are read");
            require(l.warnings.size() == 7, "one warning per bad line");
            require(Has(l.warnings[0], L"test.txt, line 1") && Has(l.warnings[0], L"not a full path") && Has(l.warnings[0], L"skipped"), "a relative path: warned with its list and line");
            require(Has(l.warnings[1], L"line 3"), "a drive-relative path");
            require(Has(l.warnings[2], L"line 4") && Has(l.warnings[2], L"64 hex digits"), "a short hash");
            require(Has(l.warnings[3], L"line 5"), "a path with a short hash: the whole line is skipped (not the path alone)");
            require(Has(l.warnings[4], L"line 6"), "a hash with a letter that is not hex");
            require(Has(l.warnings[5], L"line 7"), "a wildcard");
            require(Has(l.warnings[6], L"line 8") && Has(l.warnings[6], L"format version"), "a version line without a number");
            IgnoreEntry e;
            wstring why;
            require(ParseIgnoreLine(L"   # comment", e, why) && e.text.empty() && ParseIgnoreLine(L"", e, why) && e.text.empty(), "comment and blank lines: no entry");
        }
        // ---- matching -------------------------------------------------------------------------------------------------
        {
            // case, slashes, '.' and '..'
            const wstring odd = Slashes(Lower(Join(lib, L"Game A\\.\\bin\\..\\bin\\x64\\NVNGX_DLSS.DLL")));
            IgnoreLists l = Parse(U8(odd) + "\n");
            require(l.entries.size() == 1, "an odd spelling of a path is read");
            const wstring why = IgnoreReason(l, a);
            require(!why.empty() && Has(why, L"test.txt, line 1") && Has(why, L"(path "), "case, '/', '.' and '..' normalised: the file is skipped, with the list, the line and the path");
            require(!IgnoreReason(l, Upper(a)).empty(), "the candidate spelled in upper case");
            require(IgnoreReason(l, b).empty() && IgnoreReason(l, c).empty(), "other files are not skipped");
            IgnoreLists gone = Parse(U8(Join(root, L"Gone\\nvngx_dlss.dll")) + "\n");
            require(gone.entries.size() == 1 && IgnoreReason(gone, a).empty(), "a listed path that does not exist matches nothing else");
            // a hash alone: wherever the file is; a path with a hash matches by either
            IgnoreLists h = Parse("sha256=" + U8(shaB) + "\n");
            const wstring copy = Join(root, L"Elsewhere\\renamed.dll");
            File(copy, "synthetic file B");
            require(Has(IgnoreReason(h, b), L"SHA-256 " + shaB) && !IgnoreReason(h, copy).empty(), "a hash alone skips the file wherever it is, under any name");
            require(IgnoreReason(h, a).empty(), "a file with other bytes is not skipped by the hash");
            IgnoreLists ph = Parse(U8(c) + "  sha256=" + U8(shaB) + "\n");
            require(!IgnoreReason(ph, c).empty() && !IgnoreReason(ph, copy).empty() && IgnoreReason(ph, a).empty(), "a path with a hash: either matches");
            // a hard link to a listed file
            const wstring hard = Join(root, L"Hard\\nvngx_dlss.dll");
            EnsureDir(Parent(hard));
            if (CreateHardLinkW(hard.c_str(), a.c_str(), nullptr)) {
                require(Has(IgnoreReason(l, hard), L"the same file"), "a hard link to a listed file is the same file");
            } else {
                std::printf("convert-ignore: hard link not made here (error %lu): that check is skipped\n", GetLastError());
            }
            // a junction over the listed file's folder (no privilege needed)
            const wstring junction = Join(root, L"Junction");
            const wstring cmd = L"cmd.exe /d /c mklink /J \"" + junction + L"\" \"" + Parent(a) + L"\" >nul";
            if (_wsystem(cmd.c_str()) == 0 && IsDir(junction)) {
                links.push_back(junction);
                const wstring via = Join(junction, L"nvngx_dlss.dll");
                require(IsFile(via) && !IgnoreReason(l, via).empty(), "a junction to the listed file's folder: resolved to the listed file and skipped");
                IgnoreLists lj = Parse(U8(via) + "\n");
                require(!IgnoreReason(lj, a).empty(), "a path listed through a junction matches the file itself");
            } else {
                std::printf("convert-ignore: junction not made here: those checks are skipped\n");
            }
            // a symbolic link (Developer Mode or the privilege: not every account may make one)
            const wstring sym = Join(root, L"Symlink\\nvngx_dlss.dll");
            EnsureDir(Parent(sym));
            if (CreateSymbolicLinkW(sym.c_str(), a.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
                links.push_back(sym);
                require(!IgnoreReason(l, sym).empty(), "a symbolic link to the listed file: skipped");
                IgnoreLists ls = Parse(U8(sym) + "\n");
                require(!IgnoreReason(ls, a).empty(), "a listed symbolic link matches the file it points to");
            } else {
                std::printf("convert-ignore: SYMLINK-SKIPPED: this account may not make a symbolic link (error %lu): those checks are skipped\n", GetLastError());
            }
        }
        // ---- the well-known lists ---------------------------------------------------------------------------------------
        {
            const wstring lad = Join(root, L"localappdata");
            EnsureDir(lad);
            SetEnvironmentVariableW(L"LOCALAPPDATA", lad.c_str());
            const wstring shared = Join(lad, L"WrongGPU");
            std::vector<wstring> w = WellKnownIgnoreLists(shared);
            require(w.size() == 1 && IEq(w[0], Join(shared, L"ignore-files.txt")), "installed launcher: %LOCALAPPDATA%\\WrongGPU\\ignore-files.txt once");
            const wstring portable = Join(root, L"Portable\\WrongGPU-data");
            w = WellKnownIgnoreLists(portable);
            require(w.size() == 2 && IEq(w[1], Join(portable, L"ignore-files.txt")), "a portable launcher: its WrongGPU-data\\ignore-files.txt too");
            Context ctx;
            ctx.ownDataDir = portable;
            IgnoreReport rep;
            require(LoadConvertIgnore(ctx, L"model", &rep).empty() && rep.warnings.empty(), "no list anywhere: empty, no warning");
            File(Join(shared, L"ignore-files.txt"), "wronggpu-ignore-files 1\r\n" + U8(a) + "\r\nnot a path\r\n");
            File(Join(portable, L"ignore-files.txt"), "sha256=" + U8(shaB) + "\r\n");
            const wstring cli = Join(root, L"cli-list.txt");
            File(cli, U8(c) + "\n");
            ctx.ignoreListFiles = {cli};
            ctx.ignoreEntries = {chosen, L"bad entry"};
            rep = IgnoreReport();
            IgnoreLists l = LoadConvertIgnore(ctx, L"model", &rep);
            require(l.entries.size() == 4 && l.lists.size() == 3, "the well-known lists, then the command line's list and entries");
            require(rep.warnings.size() == 2 && Has(rep.warnings[0], L"line 3") && Has(rep.warnings[1], L"--ignore=bad entry"), "their bad lines are warned about");
            const std::vector<bool> skip = IgnoredMask(ctx, l, {chosen, a, b, c, Join(root, L"Elsewhere\\renamed.dll")}, L"model", rep);
            require(skip == std::vector<bool>({true, true, true, true, true}) && rep.skipped.size() == 5, "every list and entry is honoured");
            require(Has(rep.skipped[0], L"named by --ignore=" + chosen) && Has(rep.skipped[1], L"on the ignore list " + Join(shared, L"ignore-files.txt") + L", line 2") &&
                        Has(rep.skipped[2], L"WrongGPU-data") && Has(rep.skipped[3], L"cli-list.txt, line 1"),
                    "each skipped file names the list and line, or the --ignore= argument, that skips it");

            // a well-known list that is there but can't be read fails closed, after a short wait on a sharing violation
            ctx.ignoreListFiles.clear();
            ctx.ignoreEntries.clear();
            const wstring wk = Join(shared, L"ignore-files.txt");
            HANDLE held = CreateFileW(wk.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            require(held != INVALID_HANDLE_VALUE, "the test holds the list open without sharing");
            const ULONGLONG t0 = GetTickCount64();
            rep = IgnoreReport();
            IgnoreLists locked = LoadConvertIgnore(ctx, L"model", &rep);
            const ULONGLONG waited = GetTickCount64() - t0;
            require(!locked.unreadable.empty() && rep.unreadable == locked.unreadable && Has(locked.unreadable, wk) && Has(locked.unreadable, L"another program holds it open") &&
                        Has(locked.unreadable, L"nothing was converted"), "a list held open by another program: unreadable, the file and the reason named");
            // GetTickCount64 moves in ~16 ms steps: 4 sleeps of 200 ms can read a step short of 800 ms (two steps of slack)
            require(waited + 32 >= (ULONGLONG)(kIgnoreReadTries - 1) * kIgnoreReadWaitMs && waited < 5000, "...after the retries (about 5 x 200 ms), not forever");
            std::thread release([held] { Sleep(300); CloseHandle(held); });  // an installer that finishes writing during the wait
            rep = IgnoreReport();
            IgnoreLists later = LoadConvertIgnore(ctx, L"model", &rep);
            release.join();
            require(later.unreadable.empty() && later.entries.size() == 2, "a list let go during the retries is read");
            DeleteFileW(wk.c_str());
            EnsureDir(wk);  // a folder where the list should be
            rep = IgnoreReport();
            require(Has(LoadConvertIgnore(ctx, L"model", &rep).unreadable, L"a folder"), "a folder named ignore-files.txt: unreadable (fail closed)");
            RemoveDirectoryW(wk.c_str());
            DeleteFileW(Join(portable, L"ignore-files.txt").c_str());
            rep = IgnoreReport();
            require(LoadConvertIgnore(ctx, L"model", &rep).unreadable.empty(), "no list at all: nothing unreadable");
        }
        // ---- a conversion on this PC: its recorded source checked against the lists ------------------------------------
        {
            const wstring src = Join(root, L"source.txt");
            File(src, "source\t" + U8(a) + "\r\nversion\t310.5.0.0\r\nsha256\t" + U8(shaB) + "\r\n");
            require(SourceField(src, L"source") == a && SourceField(src, L"sha256") == shaB && SourceField(src, L"none").empty(), "model-source.txt fields read");
            IgnoreLists byPath = Parse(U8(a) + "\n"), byHash = Parse("sha256=" + U8(shaB) + "\n"), other = Parse(U8(c) + "\n");
            require(Has(RecordedSourceReason(byPath, a, L""), L"line 1"), "a recorded source path on a list");
            require(Has(RecordedSourceReason(byHash, Join(root, L"Gone\\nvngx_dlss.dll"), shaB), L"SHA-256"), "a recorded SHA-256 on a list (the source file itself gone)");
            require(RecordedSourceReason(other, a, shaB).empty() && RecordedSourceReason(byHash, a, L"").empty() && RecordedSourceReason(IgnoreLists(), a, shaB).empty(),
                    "not on a list; no hash recorded (the file is not hashed again); no list");
        }
        // ---- the shared folder removed: its ignore list stays (StartCleanup: the last game's uninstall, the restorer) --
        {
            const wstring appData = Join(Join(root, L"localappdata"), L"WrongGPU");
            const string listBytes = "wronggpu-ignore-files 1\r\n" + U8(a) + "\r\n";
            File(Join(appData, L"model\\model.wgcm"), "synthetic model data");
            File(Join(appData, L"ignore-files.txt"), listBytes);
            Context ctx;
            const wstring kept = StartCleanup(ctx, L"", L"", appData);
            string back;
            require(StartsWithI(kept, kKeptIgnoreListText) && Has(kept, Join(appData, L"ignore-files.txt")), "StartCleanup says it kept ignore-files.txt");
            require(ReadAll(Join(appData, L"ignore-files.txt"), back) && back == listBytes && !IsDir(Join(appData, L"model")),
                    "the list is back in place byte for byte, the rest of the folder went");
            DeleteFileW(Join(appData, L"ignore-files.txt").c_str());
            File(Join(appData, L"model\\model.wgcm"), "synthetic model data");
            require(StartCleanup(ctx, L"", L"", appData).empty() && !IsDir(appData), "no list: nothing kept, the folder goes as before");
            // 0.6.1 stress (every build, the release configuration too): the two cases above back to back, 200 times on the
            // same folder, mostly within one GetTickCount64 step of each other. Before the fix the second move aimed at the
            // first one's folder whenever both fell in the same step (ERROR_ALREADY_EXISTS) and the folder stayed.
            {
                Context sctx;
                sctx.logFile = Join(root, L"cleanup-stress.log");  // outside the folder that is moved
                const int runs = 200;
                int gone = 0;
                for (int i = 0; i < runs; ++i) {
                    File(Join(appData, L"model\\model.wgcm"), "synthetic model data");
                    File(Join(appData, L"ignore-files.txt"), listBytes);
                    const wstring k1 = StartCleanup(sctx, L"", L"", appData);
                    DeleteFileW(Join(appData, L"ignore-files.txt").c_str());
                    File(Join(appData, L"model\\model.wgcm"), "synthetic model data");
                    const wstring k2 = StartCleanup(sctx, L"", L"", appData);
                    if (StartsWithI(k1, kKeptIgnoreListText) && k2.empty() && !IsDir(appData)) ++gone;
                }
                string slog;
                ReadAll(sctx.logFile, slog);
                size_t retries = 0, deferred = 0;
                for (size_t at = 0; (at = slog.find("); trying again in ", at)) != string::npos; ++at) ++retries;
                for (size_t at = 0; (at = slog.find("cleanup deferred", at)) != string::npos; ++at) ++deferred;
                std::printf("convert-ignore: cleanup stress %d/%d (retries %zu, deferred %zu)\n", gone, runs, retries, deferred);
                require(gone == runs && deferred == 0, "cleanup stress: the list kept, then no list: the folder goes, 200 of 200 times");
            }
            // 0.6.1: the move's bounded retry (error 32 sharing violation, error 5 access denied: a scanner's open file),
            // through the test seam: fails for the first N tries, then moves; or fails every time and ends deferred
            {
                Context rctx;
                rctx.logFile = Join(root, L"cleanup-retry.log");
                struct Case { DWORD error; int failFirst; bool gone; int calls; size_t retries; const char* what; };
                const Case cases[] = {
                    {ERROR_ACCESS_DENIED, 3, true, 4, 3, "error 5 for 3 tries, then the move: the folder goes after 3 logged retries"},
                    {ERROR_ACCESS_DENIED, 1000, false, kCleanupMoveTries, kCleanupMoveTries - 1,
                     "error 5 every time: 5 tries (4 logged retries), then deferred with the error, the folder left in place"},
                    {ERROR_SHARING_VIOLATION, 2, true, 3, 2, "error 32 for 2 tries, then the move: the folder goes"},
                    {ERROR_ALREADY_EXISTS, 1000, false, 1, 0, "another error (183): no retry, deferred at once"}};
                for (const Case& rc : cases) {
                    DeleteFileW(rctx.logFile.c_str());
                    File(Join(appData, L"model\\model.wgcm"), "synthetic model data");
                    g_moveCalls = 0; g_moveFailFirst = rc.failFirst; g_moveError = rc.error;
                    SetCleanupMoveForTest(&FakeMove);
                    const ULONGLONG t0 = GetTickCount64();
                    const wstring rkept = StartCleanup(rctx, L"", L"", appData);
                    const ULONGLONG took = GetTickCount64() - t0;
                    SetCleanupMoveForTest(nullptr);
                    string rlog;
                    ReadAll(rctx.logFile, rlog);
                    const string code = "(error " + std::to_string(rc.error);
                    const size_t retries = CountIn(rlog, code + (rc.error == ERROR_SHARING_VIOLATION ? ", sharing violation); trying again" : ", access denied); trying again"));
                    const size_t deferred = CountIn(rlog, "cleanup deferred: cannot detach " + U8(appData) + " " + code + ":");
                    std::printf("convert-ignore: cleanup retry, error %lu x %d: %d tries, %zu retries logged, deferred %zu, %llu ms\n",
                                rc.error, rc.failFirst, g_moveCalls.load(), retries, deferred, (unsigned long long)took);
                    require(rkept.empty() && IsDir(appData) == !rc.gone && g_moveCalls == rc.calls && retries == rc.retries &&
                                deferred == (rc.gone ? 0u : 1u) && took < 5000, rc.what);
                }
                if (IsDir(appData)) {  // the deferred cases left it: the real move now
                    require(StartCleanup(rctx, L"", L"", appData).empty() && !IsDir(appData), "after the retry cases: the folder goes");
                }
            }
            for (int i = 0; i < 600; ++i) {  // the hidden cleanup deletes the moved-aside folders (ping -n 3 first)
                bool left = false;
                for (auto& d : ListDirs(Join(root, L"localappdata"))) left = left || Has(d, L".cleanup-");
                if (!left) break;
                Sleep(100);
            }
        }
        // ---- UNC paths (the drive's administrative share on this PC, read only; nothing is shared or changed) -------------
        {
            const wstring unc = L"\\\\localhost\\" + wstring(1, a[0]) + L"$" + a.substr(2);
            if (a.size() > 3 && a[1] == L':' && IsFile(unc)) {
                IgnoreLists l = Parse(U8(unc) + "\n");
                require(l.entries.size() == 1 && l.warnings.empty(), "a UNC path is a full path");
                require(!IgnoreReason(l, unc).empty(), "a UNC entry matches the candidate reached by the same UNC path");
                IgnoreLists l2 = Parse(U8(L"\\\\?\\UNC\\localhost\\" + wstring(1, a[0]) + L"$" + a.substr(2)) + "\n");
                require(l2.entries.size() == 1 && !IgnoreReason(l2, unc).empty(), "a \\\\?\\UNC\\ entry is read as the same UNC path");
                IgnoreLists local = Parse(U8(a) + "\n");
                // the share's path and the local path are two names of one file: the file identity matches them
                require(Has(IgnoreReason(l, a), L"the same file"), "a UNC entry matches the same file reached by its local path");
                require(Has(IgnoreReason(local, unc), L"the same file"), "a local entry matches the same file reached through the UNC path");
                IgnoreLists lh = Parse("sha256=" + U8(Lower(Sha256File(a))) + "\n");
                require(!IgnoreReason(lh, unc).empty(), "a hash entry matches the file read through its UNC path");
            } else {
                std::printf("convert-ignore: UNC-SKIPPED: %ls is not reachable here\n", unc.c_str());
            }
        }
        // ---- an empty list never changes the candidates (and so never which file wins) ------------------------------
        {
            Context ctx;
            const wstring shared = Join(Join(root, L"localappdata"), L"WrongGPU");
            GameTarget t;
            t.root = Join(lib, L"Game A");
            const std::vector<StoreGame> libs{{L"Game B", Join(lib, L"Game B"), L"", L"Folder"}, {L"Game C", Join(lib, L"Game C"), L"", L"Folder"},
                                              {L"Game A", Join(lib, L"Game A"), L"", L"Folder"}};
            const std::vector<StoreGame> rev(libs.rbegin(), libs.rend());
            struct Set { const char* name; const GameTarget* t; wstring user; std::vector<StoreGame> libs; };
            const Set sets[] = {{"chosen file, this game, three library games", &t, chosen, libs},
                                {"library games only", nullptr, L"", libs},
                                {"library games in the other order", nullptr, L"", rev},
                                {"chosen file only", nullptr, chosen, {}},
                                {"this game only", &t, L"", {}},
                                {"nothing found", nullptr, L"", {}}};
            const string emptyLists[] = {"", "\xEF\xBB\xBF", "# nothing\r\n\r\nwronggpu-ignore-files 1\r\n", "\r\n\r\n"};
            for (const string& listText : emptyLists) {
                File(Join(shared, L"ignore-files.txt"), listText);
                IgnoreReport rep;
                const IgnoreLists l = LoadConvertIgnore(ctx, L"model", &rep);
                require(l.empty() && rep.warnings.empty(), "an empty list file (blank, a BOM alone, comments and the version line): no entry, no warning");
                for (const Set& s : sets) {
                    const std::vector<ModelCandidate> cands = ModelCandidates(s.t, s.user, s.libs);
                    std::vector<wstring> paths;
                    for (auto& x : cands) paths.push_back(x.path);
                    const std::vector<bool> skip = IgnoredMask(ctx, l, paths, L"model", rep);
                    std::vector<wstring> kept;
                    for (size_t i = 0; i < paths.size(); ++i) if (!skip[i]) kept.push_back(paths[i]);
                    if (kept != paths || !rep.skipped.empty()) { std::printf("set: %s\n", s.name); require(false, "an empty list keeps every candidate in its order"); }
                    ++checks;
                }
            }
            // and a list whose entries match none of them changes nothing either
            File(Join(shared, L"ignore-files.txt"), U8(Join(root, L"Gone\\nvngx_dlss.dll")) + "\r\nsha256=" + string(64, '0') + "\r\n");
            IgnoreReport rep;
            const IgnoreLists l = LoadConvertIgnore(ctx, L"model", &rep);
            const std::vector<ModelCandidate> cands = ModelCandidates(&t, chosen, libs);
            std::vector<wstring> paths;
            for (auto& x : cands) paths.push_back(x.path);
            require(paths.size() == 4 && l.entries.size() == 2, "the full candidate set: the chosen file and three games' files");
            const std::vector<bool> skip = IgnoredMask(ctx, l, paths, L"model", rep);
            require(std::find(skip.begin(), skip.end(), true) == skip.end() && rep.skipped.empty(), "a list that names none of the candidates keeps them all");
            DeleteFileW(Join(shared, L"ignore-files.txt").c_str());
        }
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << " (after " << checks << " checks)\n";
        for (auto& l : links) { if (IsDir(l)) RemoveDirectoryW(l.c_str()); else DeleteFileW(l.c_str()); }
        return 1;
    }
    for (auto& l : links) { if (IsDir(l)) RemoveDirectoryW(l.c_str()); else DeleteFileW(l.c_str()); }
    RemoveTree(root);
    std::printf("convert-ignore: %d checks passed\n", checks);
    return 0;
}
