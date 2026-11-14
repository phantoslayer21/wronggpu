// SPDX-License-Identifier: MIT
// Unit tests of the converter search's ignore lists (wg_core/convert_ignore.h): the list format (version line, comments,
// a path, a path with its SHA-256, quotes, UTF-8 / UTF-16 LE, CRLF; a hash alone, a folder and an undecodable list refused),
// bad lines (warned, skipped, the rest read), path matching (case, '/' and '\', '.' and '..', a hard link, a junction, a
// symbolic link when this account may make one), a path + hash entry that matches only when both match (an identical copy
// elsewhere is not skipped), --ignore=sha256= that matches wherever the file is, the well-known list locations, and an empty
// list that leaves the real candidate search (AllowedModelCandidates, the one FindModel uses) exactly as it was; a recorded
// source matched as recorded (its path made a hard or symbolic link to a listed file after the conversion is still not
// listed); and a player's list never deleted by the cleanups (StartCleanup, RemoveSharedData, SelfUninstall: another
// program's new list there meanwhile, or no way to put it back; RemoveSharedData and SelfUninstall against a private registry
// hive of the test's own). Synthetic files only, in the temp folder. Written for this project. Run by ctest ("convert-ignore").
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include <thread>

#include "wg_core/convert_ignore.h"
#include "wg_core/launcher.h"
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
string Bytes(const wstring& p) { string b; return ReadAll(p, b) ? b : string("<none>"); }
// PutIgnoreListBack's test seam (convert_ignore.h SetPutBackHookForTest), run between the folder's move and the put-back:
// another program makes the folder again with its own list, or a file is where the folder should be
const string kOtherList = "wronggpu-ignore-files 1\r\n# another program's new list (test)\r\n";
void OtherProgramMakesList(const wstring& shared) { File(Join(shared, L"ignore-files.txt"), kOtherList); }
void FileWhereFolderShouldBe(const wstring& shared) { const string t = "a file where the folder should be (test)"; WriteFileAtomic(shared, t.data(), t.size()); }
// A list of --ignore= entries (the command line takes "sha256=<hex>" alone; a list does not)
IgnoreLists Cli(const std::vector<wstring>& entries) {
    IgnoreLists l;
    for (auto& s : entries) {
        IgnoreEntry e;
        wstring why;
        if (!ParseIgnoreLine(s, e, why, nullptr, true) || e.text.empty()) throw std::runtime_error("a --ignore= entry of the test is not understood: " + U8(s) + ": " + U8(why));
        e.where = L"--ignore=" + s;
        l.entries.push_back(e);
    }
    return l;
}
// The paths of the candidates the search tries (AllowedModelCandidates: what FindModel uses), and what it left out
struct Search { std::vector<wstring> tried; IgnoreReport ignored; wstring chosenSkipped; };
Search RunSearch(Context& ctx, const IgnoreLists& ign, const GameTarget* t, const wstring& user, const std::vector<StoreGame>& libs) {
    ModelState m;
    Search s;
    for (auto& c : AllowedModelCandidates(ctx, ign, t, user, libs, m)) s.tried.push_back(c.path);
    s.ignored = m.ignored;
    s.chosenSkipped = m.chosenSkipped;
    return s;
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
            require(l.entries.size() == 3 && l.warnings.size() == 1, "a list with a version line, comments, blank lines, a path, a path with its hash, a quoted path; a hash alone is warned about");
            require(l.entries[0].path == a && l.entries[0].sha256.empty() && Has(l.entries[0].where, L"line 5"), "a path entry, where it is written");
            require(l.entries[1].path == b && l.entries[1].sha256 == shaB, "a path with its SHA-256 (hex read in either case)");
            require(l.entries[2].path == c, "quotes around a path");
            require(Has(l.warnings[0], L"test.txt, line 7") && Has(l.warnings[0], L"hash alone is not accepted in a list") && Has(l.warnings[0], L"--ignore=sha256=") &&
                        Has(l.warnings[0], L"skipped"), "a hash alone in a list: skipped with a warning (list, line, why, what to write instead)");
            IgnoreEntry ce;
            wstring cwhy;
            require(ParseIgnoreLine(L"sha256=" + wstring(64, L'a'), ce, cwhy, nullptr, true) && ce.path.empty() && ce.sha256 == wstring(64, L'a') && !ce.text.empty(),
                    "a hash alone is an entry for --ignore=");
            IgnoreLists lf = Parse("\xEF\xBB\xBF" + U8(a) + "\n" + U8(b) + "\n");
            require(lf.warnings.empty() && lf.entries.size() == 2 && lf.entries[0].path == a, "UTF-8 with a BOM, LF line ends");
            const wstring w16 = L"\xFEFF" + a + L"\r\n" + b + L"  sha256=" + shaB + L"\r\n";
            string bytes(w16.size() * 2, '\0');
            std::memcpy(&bytes[0], w16.data(), bytes.size());
            IgnoreLists lw = Parse(string("\xFF\xFE", 2) + bytes.substr(2));
            require(lw.warnings.empty() && lw.entries.size() == 2 && lw.entries[0].path == a && lw.entries[1].path == b && lw.entries[1].sha256 == shaB, "UTF-16 LE with its BOM");
            IgnoreLists ln = Parse("wronggpu-ignore-files 7\n" + U8(a) + "\n");
            require(ln.entries.size() == 1 && ln.warnings.size() == 1 && Has(ln.warnings[0], L"newer"), "a newer format version: said once, its lines still read");
        }
        // ---- a list that can't be decoded is not guessed at (fail closed): UTF-16 big-endian, UTF-16 without a BOM, not UTF-8 ----
        {
            auto wide = [](const wstring& w, bool bigEndian, bool bom) {
                string s;
                if (bom) s += bigEndian ? string("\xFE\xFF", 2) : string("\xFF\xFE", 2);
                for (wchar_t ch : w) {
                    const char lo = (char)(ch & 0xFF), hi = (char)(ch >> 8);
                    s += bigEndian ? string{hi, lo} : string{lo, hi};
                }
                return s;
            };
            IgnoreLists none;
            wstring why;
            require(!ParseIgnoreText(wide(a + L"\r\n", true, true), L"be.txt", none, &why) && Has(why, L"big-endian") && none.entries.empty() && none.warnings.empty(),
                    "UTF-16 big-endian (with its BOM): not decoded, nothing added");
            why.clear();
            require(!ParseIgnoreText(wide(a + L"\r\n", false, false), L"nobom.txt", none, &why) && Has(why, L"NUL") && none.entries.empty(), "UTF-16 LE without a BOM: not decoded");
            why.clear();
            require(!ParseIgnoreText(wide(a + L"\r\n", true, false), L"nobom-be.txt", none, &why) && Has(why, L"NUL"), "UTF-16 BE without a BOM: not decoded");
            why.clear();
            require(!ParseIgnoreText(U8(a) + "\xC3\x28\r\n", L"latin.txt", none, &why) && Has(why, L"not UTF-8") && none.entries.empty(), "bytes that are not UTF-8: not decoded");
            why.clear();
            require(!ParseIgnoreText(string("\xFF\xFE", 2) + string("a\0b", 3), L"odd.txt", none, &why), "UTF-16 with a byte missing: not decoded");
            require(ParseIgnoreText("", L"empty.txt", none) && ParseIgnoreText("\xEF\xBB\xBF", L"bom.txt", none) && ParseIgnoreText(U8(L"café ") + "\n", L"u8.txt", none),
                    "an empty file, a BOM alone, UTF-8 with an accent: decoded");
            IgnoreEntry fe;
            wstring fwhy;
            require(!ParseIgnoreLine(lib, fe, fwhy) && Has(fwhy, L"folder") && fe.text.empty(), "a folder entry: not understood (it would ignore nothing)");
            IgnoreLists fl = Parse(U8(lib) + "\n" + U8(a) + "\n");
            require(fl.entries.size() == 1 && fl.warnings.size() == 1 && Has(fl.warnings[0], L"test.txt, line 1") && Has(fl.warnings[0], L"folder"),
                    "a folder in a list: warned with the list and the line like any bad line, the rest read");
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
            // --ignore=sha256=<hex> (the only hash alone): wherever the file is, under any name
            IgnoreLists h = Cli({L"sha256=" + shaB});
            const wstring copy = Join(root, L"Elsewhere\\renamed.dll");
            File(copy, "synthetic file B");
            require(Has(IgnoreReason(h, b), L"SHA-256 " + shaB) && !IgnoreReason(h, copy).empty(), "--ignore=sha256= skips the file wherever it is, under any name");
            require(IgnoreReason(h, a).empty(), "a file with other bytes is not skipped by the hash");
            // a path with a hash matches only when BOTH match: that file, with that SHA-256
            IgnoreLists pb = Parse(U8(b) + "  sha256=" + U8(shaB) + "\n");
            require(pb.entries.size() == 1 && pb.warnings.empty(), "a path with its hash is one entry");
            const wstring pbWhy = IgnoreReason(pb, b);
            require(!pbWhy.empty() && Has(pbWhy, L"test.txt, line 1") && Has(pbWhy, L"(path " + b) && Has(pbWhy, L"; SHA-256 " + shaB), "path + hash, both match: skipped, the reason names both");
            require(IgnoreReason(pb, copy).empty() && IgnoreReason(pb, a).empty(), "an identical copy under another path is NOT skipped by a path + hash entry (nor another file)");
            IgnoreLists pc = Parse(U8(c) + "  sha256=" + U8(shaB) + "\n");  // c's path with b's bytes' hash
            require(IgnoreReason(pc, c).empty() && IgnoreReason(pc, copy).empty() && IgnoreReason(pc, b).empty(),
                    "a path + hash entry whose hash is not the file's: the file is not skipped, and neither is the file with that hash elsewhere");
            // two identical stand-ins in two folders; only one is listed, by path and hash: the other is still a candidate
            const wstring twin1 = Join(root, L"Twin\\One\\nvngx_dlss.dll"), twin2 = Join(root, L"Twin\\Two\\nvngx_dlss.dll");
            File(twin1, "synthetic twin bytes");
            File(twin2, "synthetic twin bytes");
            const wstring shaT = Lower(Sha256File(twin1));
            require(shaT == Lower(Sha256File(twin2)), "the stand-ins are identical");
            IgnoreLists pt = Parse(U8(twin1) + "  sha256=" + U8(shaT) + "\n");
            require(!IgnoreReason(pt, twin1).empty() && IgnoreReason(pt, twin2).empty(), "two identical files, one listed by path + hash: the listed one is skipped, the other is not");
            {   // the real search: the other is the one that is tried
                Context sctx;
                const std::vector<StoreGame> tw{{L"One", Join(root, L"Twin\\One"), L"", L"Folder"}, {L"Two", Join(root, L"Twin\\Two"), L"", L"Folder"}};
                const Search sr = RunSearch(sctx, pt, nullptr, L"", tw);
                require(sr.tried == std::vector<wstring>{twin2} && sr.ignored.skipped.size() == 1 && sr.ignored.skipped[0].rfind(twin1 + L": ", 0) == 0,
                        "the candidate search tries the unlisted identical copy (twin 2) and names the listed one (twin 1) as skipped");
                const Search none = RunSearch(sctx, IgnoreLists(), nullptr, L"", tw);
                require(none.tried == std::vector<wstring>{twin1, twin2}, "and with no entry both are tried, in order");
            }
            // a path + hash entry for a file that now holds other bytes (a game update replaced it): not skipped
            const wstring changed = Join(root, L"Changed\\nvngx_dlss.dll");
            File(changed, "the bytes it had when it was listed");
            const wstring shaOld = Lower(Sha256File(changed));
            IgnoreLists pold = Parse(U8(changed) + "  sha256=" + U8(shaOld) + "\n");
            require(!IgnoreReason(pold, changed).empty(), "path + hash entry, the file as listed: skipped");
            File(changed, "other bytes after an update");
            require(IgnoreReason(pold, changed).empty(), "...the same path now holding other bytes: not skipped");
            // a hash is computed only when an entry that needs it applies; a file that can't be hashed then is left out (fail closed)
            {
                const wstring locked = Join(root, L"Locked\\nvngx_dlss.dll");
                File(locked, "synthetic locked file");
                HANDLE held = CreateFileW(locked.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                require(held != INVALID_HANDLE_VALUE, "the test holds a candidate open without sharing");
                require(Lower(Sha256File(locked)).empty(), "...so it can't be hashed");
                IgnoreLists byHashOnly = Cli({L"sha256=" + shaB}), byPathHash = Parse(U8(locked) + "  sha256=" + U8(shaB) + "\n"), elsewhereHash = Parse(U8(b) + "  sha256=" + U8(shaB) + "\n"),
                            byPathOnly = Parse(U8(a) + "\n");
                const wstring r1 = IgnoreReason(byHashOnly, locked), r2 = IgnoreReason(byPathHash, locked);
                require(Has(r1, L"unreadable") && Has(r1, L"--ignore=sha256=") && Has(r2, L"unreadable") && Has(r2, L"test.txt, line 1"),
                        "a candidate that can't be hashed, a hash entry applying to it (--ignore=sha256=, or a path + hash entry for its path): left out, the entry named");
                require(IgnoreReason(elsewhereHash, locked).empty() && IgnoreReason(byPathOnly, locked).empty(), "no hash entry applies (another path, or a path alone): it is not hashed, not skipped");
                CloseHandle(held);
            }
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
            File(Join(portable, L"ignore-files.txt"), U8(b) + "  sha256=" + U8(shaB) + "\r\n");
            const wstring cli = Join(root, L"cli-list.txt");
            File(cli, U8(c) + "\n");
            ctx.ignoreListFiles = {cli};
            ctx.ignoreEntries = {chosen, L"bad entry", L"sha256=" + shaB};
            rep = IgnoreReport();
            IgnoreLists l = LoadConvertIgnore(ctx, L"model", &rep);
            require(l.entries.size() == 5 && l.lists.size() == 3, "the well-known lists, then the command line's list and entries (--ignore=sha256= too)");
            require(rep.warnings.size() == 2 && Has(rep.warnings[0], L"line 3") && Has(rep.warnings[1], L"--ignore=bad entry"), "their bad lines are warned about");
            const std::vector<bool> skip = IgnoredMask(ctx, l, {chosen, a, b, c, Join(root, L"Elsewhere\\renamed.dll")}, L"model", rep);
            require(skip == std::vector<bool>({true, true, true, true, true}) && rep.skipped.size() == 5, "every list and entry is honoured");
            require(Has(rep.skipped[0], L"named by --ignore=" + chosen) && Has(rep.skipped[1], L"on the ignore list " + Join(shared, L"ignore-files.txt") + L", line 2") &&
                        Has(rep.skipped[2], L"WrongGPU-data") && Has(rep.skipped[3], L"cli-list.txt, line 1") && Has(rep.skipped[4], L"named by --ignore=sha256="),
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
            // a list that can't be decoded fails closed too (UTF-16 big-endian here), the reason named; a bad hash line only warns
            File(wk, string("\xFE\xFF\0a\0b", 6));
            rep = IgnoreReport();
            const IgnoreLists undecoded = LoadConvertIgnore(ctx, L"model", &rep);
            require(Has(undecoded.unreadable, wk) && Has(undecoded.unreadable, L"big-endian") && Has(undecoded.unreadable, L"nothing was converted") && rep.unreadable == undecoded.unreadable,
                    "a well-known list in UTF-16 big-endian: unreadable (fail closed), the file and the reason named");
            DeleteFileW(wk.c_str());
            // GetFileAttributesW failing for a reason other than "not there" (a name the system refuses) is an unreadable list too
            const wstring saved = ctx.ownDataDir;
            ctx.ownDataDir = Join(root, L"bad?name");
            rep = IgnoreReport();
            const IgnoreLists refused = LoadConvertIgnore(ctx, L"model", &rep);
            std::printf("convert-ignore: a name the system refuses: %ls\n", refused.unreadable.c_str());
            require(!refused.unreadable.empty() && Has(refused.unreadable, L"bad?name"), "a well-known list whose attributes can't be read (not \"not found\"): unreadable, fail closed");
            ctx.ownDataDir = saved;
            DeleteFileW(Join(portable, L"ignore-files.txt").c_str());
            rep = IgnoreReport();
            require(LoadConvertIgnore(ctx, L"model", &rep).unreadable.empty(), "no list at all: nothing unreadable");
        }
        // ---- a conversion on this PC: its recorded source checked against the lists ------------------------------------
        {
            const wstring src = Join(root, L"source.txt");
            File(src, "source\t" + U8(a) + "\r\nversion\t310.5.0.0\r\nsha256\t" + U8(shaB) + "\r\n");
            require(SourceField(src, L"source") == a && SourceField(src, L"sha256") == shaB && SourceField(src, L"none").empty(), "model-source.txt fields read");
            IgnoreLists byPath = Parse(U8(a) + "\n"), byHash = Cli({L"sha256=" + shaB}), other = Parse(U8(c) + "\n");
            require(Has(RecordedSourceReason(byPath, a, L""), L"line 1"), "a recorded source path on a list");
            require(Has(RecordedSourceReason(byHash, Join(root, L"Gone\\nvngx_dlss.dll"), shaB), L"SHA-256"), "a recorded SHA-256 on --ignore=sha256= (the source file itself gone)");
            require(RecordedSourceReason(other, a, shaB).empty() && RecordedSourceReason(byHash, a, L"").empty() && RecordedSourceReason(IgnoreLists(), a, shaB).empty(),
                    "not on a list; no hash recorded (the file is not hashed again); no list");
            // a path + hash entry lists a recorded source only when its recorded path AND recorded SHA-256 are both the entry's
            IgnoreLists pathHash = Parse(U8(a) + "  sha256=" + U8(shaB) + "\n");
            const wstring otherSha = wstring(64, L'1');
            require(Has(RecordedSourceReason(pathHash, a, shaB), L"line 1") && Has(RecordedSourceReason(pathHash, a, Upper(shaB)), L"SHA-256 " + shaB),
                    "recorded path and recorded SHA-256 both match the path + hash entry: listed");
            require(RecordedSourceReason(pathHash, a, otherSha).empty() && RecordedSourceReason(pathHash, c, shaB).empty() &&
                        RecordedSourceReason(pathHash, Join(root, L"Gone\\nvngx_dlss.dll"), shaB).empty(),
                    "only one of them matches (the hash, or the path): not listed, so a model converted from a legitimate identical copy is reused");
            // an old record without a SHA-256: the path text alone decides (its bytes can't be checked)
            require(Has(RecordedSourceReason(pathHash, a, L""), L"the path alone decides") && RecordedSourceReason(pathHash, c, L"").empty(),
                    "no hash on record: a path + hash entry naming the recorded path lists it, one naming another path does not");
            require(Has(RecordedSourceReason(byPath, Slashes(Upper(a)), L""), L"line 1") && Has(RecordedSourceReason(byPath, L"\\\\?\\" + a, shaB), L"line 1"),
                    "the recorded path text is normalised (case, '/', a \\\\?\\ prefix)");
        }
        // ---- a recorded source is matched as it was recorded: its path retargeted after the conversion -----------------
        // The conversion was made from `orig`; later `orig` is made a hard link or a symbolic link to `placed`, a file another
        // program placed and listed, with the same bytes. Today a candidate at `orig` is that listed file (skipped), but the
        // model data converted earlier came from the file that was at `orig` then: it is not listed, and is reused.
        {
            const wstring orig = Join(root, L"Retarget\\Game\\nvngx_dlss.dll"), placed = Join(root, L"Retarget\\Placed\\nvngx_dlss.dll");
            const string bytes = "synthetic retarget bytes (the same in both files)";
            File(placed, bytes);
            File(orig, bytes);
            const wstring shaR = Lower(Sha256File(orig));
            const IgnoreLists byPath = Parse(U8(placed) + "\n"), byPathHash = Parse(U8(placed) + "  sha256=" + U8(shaR) + "\n");
            require(RecordedSourceReason(byPath, orig, shaR).empty() && RecordedSourceReason(byPathHash, orig, shaR).empty(), "retarget: before, not listed");
            auto retargeted = [&](const char* kind) {
                std::printf("convert-ignore: recorded source retargeted through a %s: checked\n", kind);
                require(!IgnoreReason(byPath, orig).empty() && !IgnoreReason(byPathHash, orig).empty(),
                        "retarget: today a candidate at the source path is the listed file (skipped)");
                require(RecordedSourceReason(byPath, orig, shaR).empty() && RecordedSourceReason(byPathHash, orig, shaR).empty() &&
                            RecordedSourceReason(byPathHash, orig, L"").empty(),
                        "retarget: the recorded source is matched as recorded (path text and SHA-256): not listed, the data converted from it is reused");
            };
            DeleteFileW(orig.c_str());
            if (CreateHardLinkW(orig.c_str(), placed.c_str(), nullptr)) retargeted("hard link");
            else std::printf("convert-ignore: hard link not made here (error %lu): the retarget check is skipped\n", GetLastError());
            DeleteFileW(orig.c_str());
            if (CreateSymbolicLinkW(orig.c_str(), placed.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) { links.push_back(orig); retargeted("symbolic link"); }
            else std::printf("convert-ignore: SYMLINK-SKIPPED: retarget through a symbolic link not made here (error %lu)\n", GetLastError());
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
            Sleep(100);  // the folder's new name carries GetTickCount64 (~16 ms steps): another step than the first cleanup's
            require(StartCleanup(ctx, L"", L"", appData).empty() && !IsDir(appData), "no list: nothing kept, the folder goes as before");
            // PutIgnoreListBack: a player's list is never deleted. Back in place; beside another program's new list
            // (.kept-1, then .kept-2); left in the moved folder when it can't leave it; a list that is a link moved as the link
            {
                const wstring moved = Join(root, L"moved-aside"), again = Join(root, L"made-again"), blocker = Join(root, L"not-a-folder");
                const wstring from = Join(moved, L"ignore-files.txt"), to = Join(again, L"ignore-files.txt");
                Context pctx;
                pctx.logFile = Join(root, L"put-back.log");
                File(from, listBytes);
                File(to, kOtherList);
                ListKept k = PutIgnoreListBack(pctx, moved, again);
                require(!k.inPlace && k.leftIn.empty() && Has(k.said, L"kept your ignore-files.txt as " + to + L".kept-1") && Has(k.said, L"another program made " + to),
                        "put back over another program's new list: kept beside it as ignore-files.txt.kept-1, and said");
                require(Bytes(to + L".kept-1") == listBytes && Bytes(to) == kOtherList && !IsFile(from), "...byte for byte; the other list untouched; nothing left behind");
                File(from, listBytes);
                k = PutIgnoreListBack(pctx, moved, again);
                require(Has(k.said, to + L".kept-2") && Bytes(to + L".kept-2") == listBytes && Bytes(to + L".kept-1") == listBytes, "...a second time: .kept-2, .kept-1 untouched");
                File(blocker, "a file where the shared folder should be");
                File(from, listBytes);
                k = PutIgnoreListBack(pctx, moved, blocker);
                require(!k.inPlace && IEq(k.leftIn, from) && Has(k.said, L"it is kept at " + from) && Bytes(from) == listBytes,
                        "it can't be moved or copied out (a file where the folder should be): it stays in the moved folder (leftIn), and said where");
                require(Has(FromUtf8(Bytes(pctx.logFile)), L"could not put your ignore-files.txt back into " + blocker), "...and logged");
                require(PutIgnoreListBack(pctx, Join(root, L"no-such-folder"), blocker).said.empty(), "no list in the moved folder: nothing to put back, nothing said");
                DeleteFileW(from.c_str());
                // a list that is a link (a reparse point): moved as the link; what it points to is never touched
                const wstring target = Join(root, L"list-target.txt"), linkDir = Join(root, L"made-again-link"), linkTo = Join(linkDir, L"ignore-files.txt");
                File(target, listBytes);
                auto isLink = [](const wstring& p) { const DWORD x = GetFileAttributesW(p.c_str()); return x != INVALID_FILE_ATTRIBUTES && (x & FILE_ATTRIBUTE_REPARSE_POINT); };
                if (CreateSymbolicLinkW(from.c_str(), target.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) {
                    k = PutIgnoreListBack(pctx, moved, linkDir);
                    require(k.inPlace && isLink(linkTo) && Bytes(linkTo) == listBytes && !isLink(from) && !IsFile(from), "a list that is a link: put back as the link");
                    CreateSymbolicLinkW(from.c_str(), target.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE);
                    k = PutIgnoreListBack(pctx, moved, linkDir);
                    require(Has(k.said, linkTo + L".kept-1") && isLink(linkTo + L".kept-1") && Bytes(linkTo + L".kept-1") == listBytes, "...beside another list: .kept-1, still the link");
                    CreateSymbolicLinkW(from.c_str(), target.c_str(), SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE);
                    k = PutIgnoreListBack(pctx, moved, blocker);
                    require(IEq(k.leftIn, from) && isLink(from) && Bytes(target) == listBytes, "...that can't leave the moved folder: the link stays there (leftIn), its target untouched");
                    for (auto& l : {from, linkTo, linkTo + L".kept-1"}) DeleteFileW(l.c_str());
                    require(Bytes(target) == listBytes, "...removing the links never touched the file they point to");
                } else {
                    std::printf("convert-ignore: SYMLINK-SKIPPED: a list that is a link not made here (error %lu)\n", GetLastError());
                }
                DeleteFileW(blocker.c_str());
            }
            // ---- each cleanup path, when the list can't go back as it was (the seam runs between the move and the put-back)
            {
                const wstring list = Join(appData, L"ignore-files.txt"), lad = Parent(appData);
                auto fill = [&] { File(Join(appData, L"model\\model.wgcm"), "synthetic model data"); File(list, listBytes); };
                auto movedWithList = [&](const wchar_t* tag) {  // the moved-aside folder that holds the list (others may still be deleted)
                    for (auto& d : ListDirs(lad)) if (Has(d, tag) && IsFile(Join(d, L"ignore-files.txt"))) return d;
                    return wstring();
                };
                auto cleanAfterExists = [&](const char* what) {
                    require(Bytes(list + L".kept-1") == listBytes && Bytes(list) == kOtherList && !IsDir(Join(appData, L"model")), what);
                    DeleteFileW((list + L".kept-1").c_str());
                    DeleteFileW(list.c_str());
                    RemoveDirectoryW(appData.c_str());
                };
                auto checkBlocked = [&](const wstring& said, const wchar_t* tag, const char* what) {
                    const wstring d = movedWithList(tag);
                    require(!d.empty() && Has(said, L"it is kept at " + Join(d, L"ignore-files.txt")) && Bytes(Join(d, L"ignore-files.txt")) == listBytes &&
                                !IsDir(Join(d, L"model")) && ListFiles(d, L"*").size() == 1, what);
                    return d;
                };
                auto cleanAfterBlocked = [&](const wstring& d) {
                    DeleteFileW(Join(d, L"ignore-files.txt").c_str());
                    RemoveDirectoryW(d.c_str());
                    DeleteFileW(appData.c_str());  // the test's file where the folder should be
                };
                // StartCleanup (the last game's uninstall, the restorer: record.cpp)
                Sleep(100);
                fill();
                SetPutBackHookForTest(&OtherProgramMakesList);
                wstring said = StartCleanup(ctx, L"", L"", appData);
                SetPutBackHookForTest(nullptr);
                require(Has(said, L"kept your ignore-files.txt as " + list + L".kept-1"), "StartCleanup, the list made again meanwhile: kept as .kept-1, said in its result");
                cleanAfterExists("StartCleanup: .kept-1 holds the player's list, the other program's list untouched, the rest went");
                Sleep(100);
                fill();
                SetPutBackHookForTest(&FileWhereFolderShouldBe);
                said = StartCleanup(ctx, L"", L"", appData);
                SetPutBackHookForTest(nullptr);
                const wstring sd = checkBlocked(said, L".cleanup-", "StartCleanup, the list can't be put back: it stays in the moved folder (the rest went), said where");
                Sleep(4500);  // its hidden cmd.exe waits ~2 s (ping -n 3) before it deletes: never this folder's tree now
                require(Bytes(Join(sd, L"ignore-files.txt")) == listBytes, "StartCleanup: the hidden cleanup does not delete the list afterwards");
                cleanAfterBlocked(sd);
                // RemoveSharedData (a portable copy's Remove data...) and SelfUninstall (the launcher's uninstall): launcher.cpp. A
                // private registry hive of the test's own (RegLoadAppKey: a file in the temp folder, nothing in the user's registry)
                // stands in for Apps & features, so no WrongGPU installed on this PC is seen or changed.
                HKEY hive = nullptr;
                const LONG hr = RegLoadAppKeyW(Join(root, L"test-hive.dat").c_str(), &hive, KEY_ALL_ACCESS, 0, 0);
                if (hr == ERROR_SUCCESS) {
                    SetUninstallHive(hive);
                    wstring why, kept;
                    fill();
                    SetPutBackHookForTest(&OtherProgramMakesList);
                    const bool ok1 = RemoveSharedData(ctx, &why, &kept);
                    SetPutBackHookForTest(nullptr);
                    require(ok1 && Has(kept, L"kept your ignore-files.txt as " + list + L".kept-1"), "RemoveSharedData, the list made again meanwhile: done, kept as .kept-1, said");
                    cleanAfterExists("RemoveSharedData: .kept-1 holds the player's list, the other program's list untouched, the rest went");
                    fill();
                    SetPutBackHookForTest(&FileWhereFolderShouldBe);
                    const bool ok2 = RemoveSharedData(ctx, &why, &kept);
                    SetPutBackHookForTest(nullptr);
                    require(ok2, "RemoveSharedData, the list can't be put back: done");
                    cleanAfterBlocked(checkBlocked(kept, L".removed-", "RemoveSharedData: the list stays in the moved folder (the rest went), said where"));
                    LauncherPaths lp;
                    lp.dataDir = Join(root, L"own-data");
                    lp.custom = true;
                    std::vector<wstring> uiLines;
                    Ui ui{[](int, int, const wstring&) {}, [&uiLines](const wstring& s) { uiLines.push_back(s); }};
                    auto uiSaid = [&uiLines] { wstring all; for (auto& s : uiLines) all += s + L"\n"; return all; };
                    fill();
                    SetPutBackHookForTest(&OtherProgramMakesList);
                    const bool ok3 = SelfUninstall(ctx, lp, false, L"-", ui, &why);
                    SetPutBackHookForTest(nullptr);
                    require(ok3 && Has(uiSaid(), L"kept your ignore-files.txt as " + list + L".kept-1"), "SelfUninstall, the list made again meanwhile: done, kept as .kept-1, said");
                    cleanAfterExists("SelfUninstall: .kept-1 holds the player's list, the other program's list untouched, the rest went");
                    uiLines.clear();
                    fill();
                    SetPutBackHookForTest(&FileWhereFolderShouldBe);
                    const bool ok4 = SelfUninstall(ctx, lp, false, L"-", ui, &why);
                    SetPutBackHookForTest(nullptr);
                    require(ok4, "SelfUninstall, the list can't be put back: done");
                    const wstring ud = checkBlocked(uiSaid(), L".removed-", "SelfUninstall: the list stays in the moved folder (the rest went), said where");
                    Sleep(2500);
                    require(Bytes(Join(ud, L"ignore-files.txt")) == listBytes, "SelfUninstall: nothing deletes the list afterwards");
                    cleanAfterBlocked(ud);
                    SetUninstallHive(nullptr);
                    RegCloseKey(hive);
                } else {
                    std::printf("convert-ignore: HIVE-SKIPPED: no private registry hive here (error %ld): RemoveSharedData / SelfUninstall cases skipped\n", hr);
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
                IgnoreLists lh = Cli({L"sha256=" + Lower(Sha256File(a))});
                require(!IgnoreReason(lh, unc).empty(), "an --ignore=sha256= entry matches the file read through its UNC path");
            } else {
                std::printf("convert-ignore: UNC-SKIPPED: %ls is not reachable here\n", unc.c_str());
            }
        }
        // ---- an empty list never changes the candidate search (and so never which file wins) -----------------------------
        // The search is AllowedModelCandidates, the code FindModel runs before it converts (the conversion needs the embedded
        // map, which the CI build has not: this test compares what the search hands to the converter, in its order, whose first
        // entry is the file that converts first). No list anywhere is the baseline; an empty list in each place one is read.
        {
            Context ctx;
            const wstring shared = Join(Join(root, L"localappdata"), L"WrongGPU"), portable = Join(root, L"Portable\\WrongGPU-data");
            const wstring sharedList = Join(shared, L"ignore-files.txt"), portableList = Join(portable, L"ignore-files.txt");
            DeleteFileW(sharedList.c_str());
            DeleteFileW(portableList.c_str());
            GameTarget t;
            t.root = Join(lib, L"Game A");
            const std::vector<StoreGame> libs{{L"Game B", Join(lib, L"Game B"), L"", L"Folder"}, {L"Game C", Join(lib, L"Game C"), L"", L"Folder"},
                                              {L"Game A", Join(lib, L"Game A"), L"", L"Folder"}};
            const std::vector<StoreGame> rev(libs.rbegin(), libs.rend());
            struct Set { const char* name; const GameTarget* t; wstring user; std::vector<StoreGame> libs; size_t count; };
            const Set sets[] = {{"chosen file, this game, three library games", &t, chosen, libs, 4},
                                {"library games only", nullptr, L"", libs, 3},
                                {"library games in the other order", nullptr, L"", rev, 3},
                                {"chosen file only", nullptr, chosen, {}, 1},
                                {"this game only", &t, L"", {}, 1},
                                {"nothing found", nullptr, L"", {}, 0}};
            // the baseline: no list at all (nothing in either well-known place, nothing on the command line)
            ctx.ownDataDir = portable;
            IgnoreReport baseRep;
            const IgnoreLists none = LoadConvertIgnore(ctx, L"model", &baseRep);
            require(none.empty() && baseRep.warnings.empty() && none.unreadable.empty(), "no list anywhere: no entry, no warning");
            std::vector<Search> base;
            for (const Set& s : sets) {
                base.push_back(RunSearch(ctx, none, s.t, s.user, s.libs));
                std::vector<wstring> plain;
                for (auto& x : ModelCandidates(s.t, s.user, s.libs)) plain.push_back(x.path);
                if (base.back().tried != plain || base.back().tried.size() != s.count || !base.back().ignored.skipped.empty()) {
                    std::printf("set: %s\n", s.name);
                    require(false, "with no list the search hands over ModelCandidates unchanged");
                }
                ++checks;
            }
            // an empty list in each place one is read: %LOCALAPPDATA%\WrongGPU, the launcher's own data folder, both, and
            // --ignore-list=; each text that is empty (blank, a BOM alone, comments and the version line, blank lines)
            const wstring cliEmpty = Join(root, L"cli-empty.txt");
            const string emptyLists[] = {"", "\xEF\xBB\xBF", "# nothing\r\n\r\nwronggpu-ignore-files 1\r\n", "\r\n\r\n"};
            struct Place { const char* name; bool shared, own, cli; };
            const Place places[] = {{"%LOCALAPPDATA%\\WrongGPU\\ignore-files.txt", true, false, false}, {"the launcher's own data folder", false, true, false},
                                    {"both well-known lists", true, true, false}, {"--ignore-list=", false, false, true}};
            for (const string& listText : emptyLists) {
                for (const Place& pl : places) {
                    DeleteFileW(sharedList.c_str());
                    DeleteFileW(portableList.c_str());
                    ctx.ignoreListFiles.clear();
                    if (pl.shared) File(sharedList, listText);
                    if (pl.own) File(portableList, listText);
                    if (pl.cli) { File(cliEmpty, listText); ctx.ignoreListFiles = {cliEmpty}; }
                    IgnoreReport rep;
                    const IgnoreLists l = LoadConvertIgnore(ctx, L"model", &rep);
                    if (!l.empty() || !rep.warnings.empty() || !l.unreadable.empty()) { std::printf("place: %s\n", pl.name); require(false, "an empty list file: no entry, no warning, readable"); }
                    for (size_t i = 0; i < std::size(sets); ++i) {
                        const Search got = RunSearch(ctx, l, sets[i].t, sets[i].user, sets[i].libs);
                        const bool sameWinner = got.tried.empty() == base[i].tried.empty() && (got.tried.empty() || got.tried.front() == base[i].tried.front());
                        if (got.tried != base[i].tried || !sameWinner || !got.ignored.skipped.empty() || !got.chosenSkipped.empty()) {
                            std::printf("set: %s, empty list in %s\n", sets[i].name, pl.name);
                            require(false, "an empty list leaves the candidates, their order and the winner as they were");
                        }
                        ++checks;
                    }
                }
            }
            ctx.ignoreListFiles.clear();
            DeleteFileW(sharedList.c_str());
            DeleteFileW(portableList.c_str());
            // the search does change with an entry (the test can fail): the first candidate listed, the next one is the winner
            File(sharedList, U8(chosen) + "\r\n");
            IgnoreReport rep1;
            const IgnoreLists one = LoadConvertIgnore(ctx, L"model", &rep1);
            const Search s1 = RunSearch(ctx, one, &t, chosen, libs);
            require(one.entries.size() == 1 && s1.tried.size() == 3 && s1.tried.front() == base[0].tried[1] && s1.ignored.skipped.size() == 1 && !s1.chosenSkipped.empty(),
                    "a list that names the chosen file: it is left out (and said), the next candidate is the first tried");
            // and a list whose entries match none of them changes nothing either: another path, a path whose hash is not the file's
            File(sharedList, U8(Join(root, L"Gone\\nvngx_dlss.dll")) + "\r\n" + U8(a) + "  sha256=" + string(64, '0') + "\r\n");
            IgnoreReport rep2;
            const IgnoreLists nomatch = LoadConvertIgnore(ctx, L"model", &rep2);
            const Search s2 = RunSearch(ctx, nomatch, &t, chosen, libs);
            require(nomatch.entries.size() == 2 && rep2.warnings.empty() && s2.tried == base[0].tried && s2.ignored.skipped.empty(), "a list that names none of the candidates keeps them all");
            DeleteFileW(sharedList.c_str());
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
