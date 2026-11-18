// SPDX-License-Identifier: MIT
// Tests of the Linux layer itself (src/platform/posix): the Windows calls it implements, checked against what Windows does.
// The core's own tests exercise it only as far as the core goes; this covers the rest (sharing, processes, the registry file).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <windows.h>

#include <restartmanager.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "win32_posix_internal.h"
#include "wg_core/appimage.h"
#include "wg_core/checks.h"
#include "wg_core/launcher.h"
#include "wg_core/optiedition.h"
#include "../../../../../wgshared/include/wg_state.h"

namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAILED: %s (last error %u)\n", what, (unsigned)GetLastError()); }
}
std::wstring Tmp;   // a fresh folder of this run
std::wstring In(const wchar_t* rel) { return Tmp + L"\\" + rel; }
bool Put(const std::wstring& p, const char* text) {
    HANDLE h = CreateFileW(p.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD w = 0;
    const bool ok = WriteFile(h, text, (DWORD)std::strlen(text), &w, nullptr);
    CloseHandle(h);
    return ok;
}
std::string Slurp(const std::wstring& p) {
    HANDLE h = CreateFileW(p.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return "<missing>";
    char b[4096]; DWORD got = 0;
    ReadFile(h, b, sizeof(b), &got, nullptr);
    CloseHandle(h);
    return std::string(b, got);
}

void Paths() {
    check(GetFileAttributesW(L"C:\\Windows") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_PATH_NOT_FOUND, "a drive other than Z: does not exist");
    check((GetFileAttributesW(L"Z:\\") & FILE_ATTRIBUTE_DIRECTORY) != 0, "Z:\\ is the root folder");
    check((GetFileAttributesW(L"\\\\?\\Z:\\") & FILE_ATTRIBUTE_DIRECTORY) != 0, "a \\\\?\\ prefix is dropped");
    check(GetFileAttributesW(L"Z:\\no?such") == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_INVALID_NAME, "a ? in a name: invalid name");
    check(GetFileAttributesW((Tmp + L"\\nothing-here\\x").c_str()) == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_PATH_NOT_FOUND, "a missing folder: path not found");
    check(GetFileAttributesW(In(L"nothing-here").c_str()) == INVALID_FILE_ATTRIBUTES && GetLastError() == ERROR_FILE_NOT_FOUND, "a missing file: file not found");
    Put(In(L"MixedCase.TXT"), "x");
    check(GetFileAttributesW(In(L"mixedcase.txt").c_str()) != INVALID_FILE_ATTRIBUTES, "names match ignoring case, as on NTFS");
    wchar_t full[MAX_PATH * 2];
    const DWORD n = GetFullPathNameW(L"Z:\\a\\.\\b\\..\\c/d", (DWORD)(sizeof(full) / sizeof(full[0])), full, nullptr);
    check(n && std::wstring(full, n) == L"Z:\\a\\c\\d", "GetFullPathNameW folds . and .. and uses backslashes");
    const DWORD m = GetFullPathNameW(L"C:\\x\\..\\y", (DWORD)(sizeof(full) / sizeof(full[0])), full, nullptr);
    check(m && std::wstring(full, m) == L"C:\\y", "a drive other than Z: is spelled, not looked at");
    check(std::wstring(Tmp).rfind(L"Z:\\", 0) == 0, "GetTempPathW is a Z:\\ path");
}

void Files() {
    check(Put(In(L"a.txt"), "hello") && Slurp(In(L"a.txt")) == "hello", "write and read back");
    HANDLE held = CreateFileW(In(L"a.txt").c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(held != INVALID_HANDLE_VALUE, "open with no sharing");
    check(CreateFileW(In(L"a.txt").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION,
          "a second open of a file held without sharing: sharing violation");
    check(!DeleteFileW(In(L"a.txt").c_str()) && GetLastError() == ERROR_SHARING_VIOLATION, "a held file cannot be deleted");
    check(!MoveFileExW(In(L"a.txt").c_str(), In(L"b.txt").c_str(), 0) && GetLastError() == ERROR_SHARING_VIOLATION, "a held file cannot be moved");
    CloseHandle(held);
    HANDLE shared = CreateFileW(In(L"a.txt").c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    check(shared != INVALID_HANDLE_VALUE && Slurp(In(L"a.txt")) == "hello", "a reader that allows readers and deleters lets another read");
    check(MoveFileExW(In(L"a.txt").c_str(), In(L"b.txt").c_str(), 0), "and lets the file move");
    CloseHandle(shared);

    Put(In(L"c.txt"), "new");
    check(!MoveFileExW(In(L"c.txt").c_str(), In(L"b.txt").c_str(), 0) && GetLastError() == ERROR_ALREADY_EXISTS, "a move onto a file without REPLACE_EXISTING fails");
    check(MoveFileExW(In(L"c.txt").c_str(), In(L"b.txt").c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) && Slurp(In(L"b.txt")) == "new", "a move with REPLACE_EXISTING replaces");

    check(SetFileAttributesW(In(L"b.txt").c_str(), FILE_ATTRIBUTE_READONLY) && (GetFileAttributesW(In(L"b.txt").c_str()) & FILE_ATTRIBUTE_READONLY), "read-only is a missing write bit");
    check(!DeleteFileW(In(L"b.txt").c_str()) && GetLastError() == ERROR_ACCESS_DENIED, "a read-only file is not deleted");
    check(CreateFileW(In(L"b.txt").c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_ACCESS_DENIED, "nor opened for writing");
    check(SetFileAttributesW(In(L"b.txt").c_str(), FILE_ATTRIBUTE_NORMAL) && DeleteFileW(In(L"b.txt").c_str()), "normal again: deleted");

    check(CreateDirectoryW(In(L"d").c_str(), nullptr) && !CreateDirectoryW(In(L"d").c_str(), nullptr) && GetLastError() == ERROR_ALREADY_EXISTS, "CreateDirectoryW twice: already exists");
    Put(In(L"d\\one.txt"), "1"); Put(In(L"d\\Two.txt"), "2"); Put(In(L"d\\three.dat"), "3");
    WIN32_FIND_DATAW fd;
    std::vector<std::wstring> names;
    HANDLE f = FindFirstFileW(In(L"d\\*.txt").c_str(), &fd);
    if (f != INVALID_HANDLE_VALUE) { do names.push_back(fd.cFileName); while (FindNextFileW(f, &fd)); check(GetLastError() == ERROR_NO_MORE_FILES, "the end of a listing: no more files"); FindClose(f); }
    check(names.size() == 2 && names[0] == L"one.txt" && names[1] == L"Two.txt", "FindFirstFileW: the wildcard, sorted ignoring case");
    check(FindFirstFileW(In(L"d\\*.zip").c_str(), &fd) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_NOT_FOUND, "no match: file not found");
    check(FindFirstFileW(In(L"nope\\*").c_str(), &fd) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_PATH_NOT_FOUND, "no folder: path not found");

    check(CreateSymbolicLinkW(In(L"link").c_str(), In(L"d").c_str(), SYMBOLIC_LINK_FLAG_DIRECTORY), "a symbolic link");
    const DWORD la = GetFileAttributesW(In(L"link").c_str());
    check((la & FILE_ATTRIBUTE_REPARSE_POINT) && (la & FILE_ATTRIBUTE_DIRECTORY), "a link to a folder is a reparse point and a folder");
    check(RemoveDirectoryW(In(L"link").c_str()) && Slurp(In(L"d\\one.txt")) == "1", "RemoveDirectoryW on the link leaves the target");
    check(!RemoveDirectoryW(In(L"d").c_str()) && GetLastError() == ERROR_DIR_NOT_EMPTY, "a folder with files stays: directory not empty");

    HANDLE h = CreateFileW(In(L"d\\one.txt").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    BY_HANDLE_FILE_INFORMATION bi{};
    FILE_ID_INFO id{};
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && bi.nFileSizeLow == 1 && GetFileInformationByHandleEx(h, FileIdInfo, &id, sizeof(id)), "file information by handle");
    CloseHandle(h);
    check(!CloseHandle(h) && GetLastError() == ERROR_INVALID_HANDLE, "closing a handle twice: invalid handle");

    HANDLE a = CreateFileW(In(L"log.txt").c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, 0, nullptr);
    DWORD w = 0;
    WriteFile(a, "ab", 2, &w, nullptr); CloseHandle(a);
    a = CreateFileW(In(L"log.txt").c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, 0, nullptr);
    check(GetLastError() == ERROR_ALREADY_EXISTS, "OPEN_ALWAYS on a file that is there: already exists, success");
    WriteFile(a, "cd", 2, &w, nullptr); CloseHandle(a);
    check(Slurp(In(L"log.txt")) == "abcd", "FILE_APPEND_DATA appends");
}

void Times() {
    SYSTEMTIME st{2001, 9, 0, 9, 1, 46, 40, 0};   // 1e9 seconds after the Unix epoch, UTC
    FILETIME ft{};
    check(SystemTimeToFileTime(&st, &ft) && (((ULONGLONG)ft.dwHighDateTime << 32) | ft.dwLowDateTime) == (1000000000ull + 11644473600ull) * 10000000ull, "SystemTimeToFileTime");
    SYSTEMTIME back{};
    check(FileTimeToSystemTime(&ft, &back) && back.wYear == 2001 && back.wMonth == 9 && back.wDay == 9 && back.wHour == 1 && back.wMinute == 46 && back.wSecond == 40, "FileTimeToSystemTime");
    FILETIME loc{}, utc{};
    check(FileTimeToLocalFileTime(&ft, &loc) && LocalFileTimeToFileTime(&loc, &utc) && CompareFileTime(&ft, &utc) == 0, "local and UTC file times convert back and forth");
    SYSTEMTIME lst{2026, 10, 0, 2, 20, 20, 0, 0};
    FILETIME lf{}, uf{}, again{};
    SYSTEMTIME shown{};
    SystemTimeToFileTime(&lst, &lf); LocalFileTimeToFileTime(&lf, &uf); FileTimeToLocalFileTime(&uf, &again); FileTimeToSystemTime(&again, &shown);
    check(shown.wHour == 20 && shown.wMinute == 20 && shown.wDay == 2, "a local reading survives the round trip");
}

void Strings() {
    const char bad[] = {'a', (char)0xFF, 'b', 0};
    check(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, bad, -1, nullptr, 0) == 0, "bytes that are not UTF-8 are refused with MB_ERR_INVALID_CHARS");
    check(MultiByteToWideChar(CP_UTF8, 0, bad, -1, nullptr, 0) == 4, "and replaced without it");
    wchar_t w[8]{};
    const char eacute[] = "\xC3\xA9";
    check(MultiByteToWideChar(CP_UTF8, 0, eacute, 2, w, 8) == 1 && w[0] == 0xE9, "UTF-8 to wide");
    char u[8]{};
    check(WideCharToMultiByte(CP_UTF8, 0, w, 1, u, 8, nullptr, nullptr) == 2 && (unsigned char)u[0] == 0xC3, "wide to UTF-8");
    check(_wcsicmp(L"Game.EXE", L"game.exe") == 0 && _wcsicmp(L"a", L"b") < 0, "_wcsicmp");
}

void Registry() {
    HKEY k = nullptr;
    DWORD disposition = 0;
    check(RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\WgLayerTest\\Sub", 0, nullptr, 0, KEY_WRITE, nullptr, &k, &disposition) == ERROR_SUCCESS && disposition == REG_CREATED_NEW_KEY, "create a key");
    const std::wstring text = L"C:\\Games\\\u00e9";
    RegSetValueExW(k, L"Text", 0, REG_SZ, (const BYTE*)text.c_str(), (DWORD)((text.size() + 1) * 2));
    const DWORD num = 42;
    RegSetValueExW(k, L"Num", 0, REG_DWORD, (const BYTE*)&num, 4);
    RegCloseKey(k);
    check(RegOpenKeyExW(HKEY_CURRENT_USER, L"software\\wgLAYERtest\\sub", 0, KEY_READ, &k) == ERROR_SUCCESS, "key names ignore case");
    wchar_t buf[64]{}; DWORD sz = sizeof(buf), type = 0;
    check(RegQueryValueExW(k, L"TEXT", nullptr, &type, (LPBYTE)buf, &sz) == ERROR_SUCCESS && type == REG_SZ && text == buf && sz == (text.size() + 1) * sizeof(wchar_t), "a string value reads back (value names ignore case)");
    DWORD got = 0, gsz = sizeof(got);
    check(RegGetValueW(HKEY_CURRENT_USER, L"Software\\WgLayerTest\\Sub", L"Num", RRF_RT_REG_DWORD, nullptr, &got, &gsz) == ERROR_SUCCESS && got == 42, "RegGetValueW reads a DWORD");
    DWORD small = 2;
    check(RegQueryValueExW(k, L"Text", nullptr, &type, (LPBYTE)buf, &small) == ERROR_MORE_DATA && small == sz, "a short buffer: more data, with the size");
    check(RegSetValueExW(k, L"X", 0, REG_SZ, (const BYTE*)L"", 2) == ERROR_ACCESS_DENIED, "a key opened for reading refuses writes");
    RegCloseKey(k);
    check(RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\WgLayerTest") == ERROR_ACCESS_DENIED, "a key with a subkey is not deleted");
    check(RegOpenKeyExW(HKEY_CURRENT_USER, L"Software\\WgLayerTest", 0, KEY_READ, &k) == ERROR_SUCCESS, "its parent exists");
    wchar_t name[64]; DWORD len = 64;
    check(RegEnumKeyExW(k, 0, name, &len, nullptr, nullptr, nullptr, nullptr) == ERROR_SUCCESS && std::wstring(name) == L"Sub" && RegEnumKeyExW(k, 1, name, &len, nullptr, nullptr, nullptr, nullptr) == ERROR_NO_MORE_ITEMS, "enumerate subkeys");
    RegCloseKey(k);
    check(RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\WgLayerTest\\Sub") == ERROR_SUCCESS && RegDeleteKeyW(HKEY_CURRENT_USER, L"Software\\WgLayerTest") == ERROR_SUCCESS, "delete leaf first, then parent");
    check(RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft", 0, KEY_READ, &k) == ERROR_FILE_NOT_FOUND, "HKEY_LOCAL_MACHINE reads as empty");
    check(RegCreateKeyExW(HKEY_LOCAL_MACHINE, L"Software\\Nope", 0, nullptr, 0, KEY_WRITE, nullptr, &k, nullptr) == ERROR_ACCESS_DENIED, "and refuses writes");

    // an app hive is a file: what one load wrote, the next reads
    const std::wstring hive = In(L"hive.dat");
    HKEY h = nullptr, key = nullptr;
    check(RegLoadAppKeyW(hive.c_str(), &h, KEY_ALL_ACCESS, 0, 0) == ERROR_SUCCESS, "load an app hive");
    check(RegCreateKeyExW(h, L"A\\B", 0, nullptr, 0, KEY_WRITE, nullptr, &key, nullptr) == ERROR_SUCCESS && RegSetValueExW(key, L"v", 0, REG_SZ, (const BYTE*)L"x\ny=z%", 14) == ERROR_SUCCESS, "write to it");
    RegCloseKey(key); RegCloseKey(h);
    h = nullptr;
    check(RegLoadAppKeyW(hive.c_str(), &h, KEY_READ, 0, 0) == ERROR_SUCCESS, "load it again");
    std::wstring seen = L"?";
    if (RegOpenKeyExW(h, L"a\\b", 0, KEY_READ, &key) == ERROR_SUCCESS) { wchar_t v[16]{}; DWORD vs = sizeof(v); RegQueryValueExW(key, L"v", nullptr, nullptr, (LPBYTE)v, &vs); seen = v; RegCloseKey(key); }
    check(seen == L"x\ny=z%", "the value (newline, = and % too) comes back from the file");
    RegCloseKey(h);
}

void Processes() {
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    check(CreatePipe(&rd, &wr, &sa, 0), "a pipe");
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wr; si.hStdError = wr;
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"/bin/sh\" -c \"printf '%s|' \\\"a b\\\" c; exit 7\"";
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    check(CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT, nullptr, nullptr, &si, &pi), "start a program");
    CloseHandle(wr);
    check(WaitForSingleObject(pi.hProcess, 10000) == WAIT_OBJECT_0, "wait for it");
    DWORD code = 0, avail = 0, got = 0;
    char out[64]{};
    check(GetExitCodeProcess(pi.hProcess, &code) && code == 7, "its exit code");
    check(PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail == 6 && ReadFile(rd, out, sizeof(out), &got, nullptr) && std::string(out, got) == "a b|c|", "its output through the pipe, quoted arguments kept together");
    check(!ReadFile(rd, out, sizeof(out), &got, nullptr) && GetLastError() == ERROR_BROKEN_PIPE, "a pipe whose writer is gone: broken pipe");
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(rd);

    // looked up in PATH when it has no folder; stopped by TerminateProcess with the code asked for
    std::wstring sleepCmd = L"sleep 30";
    std::vector<wchar_t> sb(sleepCmd.begin(), sleepCmd.end()); sb.push_back(0);
    PROCESS_INFORMATION sp{};
    STARTUPINFOW si2{}; si2.cb = sizeof(si2);
    check(CreateProcessW(nullptr, sb.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si2, &sp), "start sleep from PATH");
    check(WaitForSingleObject(sp.hProcess, 0) == WAIT_TIMEOUT && (GetExitCodeProcess(sp.hProcess, &code), code == STILL_ACTIVE), "it runs: still active");
    check(TerminateProcess(sp.hProcess, 98) && WaitForSingleObject(sp.hProcess, 5000) == WAIT_OBJECT_0 && (GetExitCodeProcess(sp.hProcess, &code), code == 98), "TerminateProcess: the exit code asked for");
    CloseHandle(sp.hProcess); CloseHandle(sp.hThread);

    // a job: processes in it end with TerminateJobObject
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    check(job && SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li)), "a job object");
    PROCESS_INFORMATION jp{};
    check(CreateProcessW(nullptr, sb.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si2, &jp) && AssignProcessToJobObject(job, jp.hProcess), "a process in the job");
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION ai{};
    check(QueryInformationJobObject(job, JobObjectBasicAccountingInformation, &ai, sizeof(ai), nullptr) && ai.ActiveProcesses == 1, "one active process");
    check(TerminateJobObject(job, 97) && WaitForSingleObject(jp.hProcess, 5000) == WAIT_OBJECT_0, "TerminateJobObject ends it");
    QueryInformationJobObject(job, JobObjectBasicAccountingInformation, &ai, sizeof(ai), nullptr);
    check(ai.ActiveProcesses == 0 && (GetExitCodeProcess(jp.hProcess, &code), code == 97), "none left, exit code 97");
    CloseHandle(jp.hProcess); CloseHandle(jp.hThread); CloseHandle(job);

    std::wstring missing = L"Z:\\no\\such\\program.exe";
    std::vector<wchar_t> mb(missing.begin(), missing.end()); mb.push_back(0);
    PROCESS_INFORMATION mp{};
    check(!CreateProcessW(nullptr, mb.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si2, &mp) && GetLastError() == ERROR_PATH_NOT_FOUND, "a program that is not there: path not found");
    Put(In(L"prog.exe"), "MZ not a program");
    std::wstring notProg = In(L"prog.exe");
    std::vector<wchar_t> nb(notProg.begin(), notProg.end()); nb.push_back(0);
    check(!CreateProcessW(nullptr, nb.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si2, &mp), "a Windows program does not start here");
}

// ---- the review findings of the process, case, suspended-start and registry-write code ----------------------------------
std::wstring WinOf(const std::string& posix) {
    std::wstring w(posix.begin(), posix.end());
    for (auto& c : w) if (c == L'/') c = L'\\';
    return L"Z:" + w;
}
char StateOf(int pid) {   // the state letter of /proc/<pid>/stat
    std::string s;
    if (FILE* f = std::fopen(("/proc/" + std::to_string(pid) + "/stat").c_str(), "r")) { char b[512]; size_t n = std::fread(b, 1, sizeof(b) - 1, f); s.assign(b, n); std::fclose(f); }
    const size_t p = s.rfind(')');
    return p == std::string::npos || p + 2 >= s.size() ? '?' : s[p + 2];
}

void ProcessImage() {
    const std::string dirPosix(Tmp.begin() + 2, Tmp.end());
    std::string dir = dirPosix;
    for (auto& c : dir) if (c == '\\') c = '/';
    const std::string exeName = "a-program-with-a-long-name.exe";   // 29 characters: /proc/<pid>/comm keeps 15
    const std::string exePath = dir + "/" + exeName;
    std::error_code ec;
    std::filesystem::copy_file("/bin/sleep", exePath, std::filesystem::copy_options::overwrite_existing, ec);
    chmod(exePath.c_str(), 0755);
    const std::string real = std::filesystem::canonical(exePath, ec).string();
    const std::wstring winForm = WinOf(real);
    std::wstring cmd = L"\"" + In(L"a-program-with-a-long-name.exe") + L"\" 30";
    std::vector<wchar_t> cb(cmd.begin(), cmd.end()); cb.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    check(CreateProcessW(nullptr, cb.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi), "start a copy of sleep with a long .exe name");
    std::wstring seen = L"<none>";
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W pe{}; pe.dwSize = sizeof(pe);
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) if (pe.th32ProcessID == pi.dwProcessId) seen = pe.szExeFile;
    CloseHandle(snap);
    check(seen == L"a-program-with-a-long-name.exe", "the snapshot shows the whole name, not the kernel's 15 characters");
    wchar_t path[MAX_PATH * 2]; DWORD n = (DWORD)(sizeof(path) / sizeof(path[0]));
    HANDLE op = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pi.dwProcessId);
    const bool got = op && QueryFullProcessImageNameW(op, 0, path, &n);
    check(got && std::wstring(path) == winForm, "the image path is the Z:\\ form of the file");
    check(got && _wcsicmp(path, winForm.c_str()) == 0, "and it equals the path the caller holds (the GameRunning comparison)");
    if (op) CloseHandle(op);
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
}

void Suspended() {
    std::wstring cmd = L"sleep 5";
    std::vector<wchar_t> cb(cmd.begin(), cmd.end()); cb.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    check(CreateProcessW(nullptr, cb.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi), "start sleep suspended");
    check(StateOf((int)pi.dwProcessId) == 'T', "a suspended process is stopped before it runs");
    check(ResumeThread(pi.hThread) == 1, "ResumeThread: the previous suspend count was 1");
    char st = 'T';
    for (int i = 0; i < 100 && (st = StateOf((int)pi.dwProcessId)) == 'T'; ++i) Sleep(10);
    check(st != 'T', "and it runs after ResumeThread");
    check(ResumeThread(pi.hThread) == 0, "a second ResumeThread: count 0");
    check(ResumeThread(nullptr) == (DWORD)-1, "ResumeThread of a bad handle: -1");
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
}

// ---- a process started suspended: failures at the call, no inherited files, its own image, no signal handlers --------------
std::string PosixTmp() {
    std::string dir(Tmp.begin() + 2, Tmp.end());
    for (auto& c : dir) if (c == '\\') c = '/';
    return dir;
}
std::string ExeOf(int pid) {
    char link[4096];
    const ssize_t n = readlink(("/proc/" + std::to_string(pid) + "/exe").c_str(), link, sizeof(link) - 1);
    return n < 0 ? std::string() : std::string(link, (size_t)n);
}
// a copy of sleep with a name of its own, so that it is not the test's own image; the path as the kernel spells it
std::string CopyOfSleep(const char* name) {
    const std::string path = PosixTmp() + "/" + name;
    std::error_code ec;
    std::filesystem::copy_file("/bin/sleep", path, std::filesystem::copy_options::overwrite_existing, ec);
    chmod(path.c_str(), 0755);
    return std::filesystem::canonical(path, ec).string();
}
void EndProcess(PROCESS_INFORMATION& pi) {
    TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
}
std::vector<wchar_t> Cmd(const std::wstring& s) { std::vector<wchar_t> v(s.begin(), s.end()); v.push_back(0); return v; }

extern "C" void HangingHandler(int) { for (;;) pause(); }   // stands for a handler that waits for a lock another thread holds

void SuspendedStart() {
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    // early failures come back from the call, as they do without CREATE_SUSPENDED
    auto bare = Cmd(L"no-such-program-in-any-path-folder 1");
    check(!CreateProcessW(nullptr, bare.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_FILE_NOT_FOUND,
          "suspended: a program that is not in PATH: file not found");
    check(!CreateProcessW(nullptr, bare.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_FILE_NOT_FOUND,
          "not suspended: the same");
    const std::string notElf = PosixTmp() + "/not-an-elf-file";
    Put(In(L"not-an-elf-file"), "just some text, no program\n");
    chmod(notElf.c_str(), 0755);
    auto ne = Cmd(In(L"not-an-elf-file"));
    check(!CreateProcessW(nullptr, ne.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_BAD_EXE_FORMAT,
          "not suspended: a file that is no program: bad exe format");
    check(!CreateProcessW(nullptr, ne.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_BAD_EXE_FORMAT,
          "suspended: the same");
    const std::string plain = PosixTmp() + "/not-executable";
    Put(In(L"not-executable"), "\x7f" "ELF");
    chmod(plain.c_str(), 0644);
    auto nx = Cmd(In(L"not-executable"));
    check(!CreateProcessW(nullptr, nx.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_ACCESS_DENIED,
          "not suspended: a file without the execute bit: access denied");
    check(!CreateProcessW(nullptr, nx.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi) && GetLastError() == ERROR_ACCESS_DENIED,
          "suspended: the same");

    // the stopped child holds none of the parent's close-on-exec files
    int fds[2];
    check(pipe2(fds, O_CLOEXEC) == 0, "a close-on-exec pipe");
    auto sl = Cmd(L"sleep 30");
    check(CreateProcessW(nullptr, sl.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi), "start sleep suspended, a pipe open");
    close(fds[1]);
    struct pollfd pf = {fds[0], POLLIN, 0};
    const int ready = poll(&pf, 1, 1000);
    check(StateOf((int)pi.dwProcessId) == 'T', "the child is still stopped");
    check(ready == 1 && (pf.revents & POLLHUP), "the reader sees the writer gone while the child is stopped (the child did not keep the write end)");
    close(fds[0]);
    EndProcess(pi);

    // while stopped, the program is the target, not the parent
    const std::string real = CopyOfSleep("suspended-target-with-a-long-name.exe");
    auto tc = Cmd(L"\"" + In(L"suspended-target-with-a-long-name.exe") + L"\" 30");
    check(CreateProcessW(nullptr, tc.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi), "start a copy of sleep suspended");
    wchar_t path[MAX_PATH * 2]; DWORD n = (DWORD)(sizeof(path) / sizeof(path[0]));
    const bool got = QueryFullProcessImageNameW(pi.hProcess, 0, path, &n);
    check(got && std::wstring(path) == WinOf(real), "stopped: the image path is the target's Z:\\ path");
    std::wstring seen = L"<none>";
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32W pe{}; pe.dwSize = sizeof(pe);
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) if (pe.th32ProcessID == pi.dwProcessId) seen = pe.szExeFile;
    CloseHandle(snap);
    check(seen == L"suspended-target-with-a-long-name.exe", "stopped: the snapshot name is the target's name");
    ResumeThread(pi.hThread);
    for (int i = 0; i < 300 && ExeOf((int)pi.dwProcessId) != real; ++i) Sleep(10);
    n = (DWORD)(sizeof(path) / sizeof(path[0]));
    check(ExeOf((int)pi.dwProcessId) == real && QueryFullProcessImageNameW(pi.hProcess, 0, path, &n) && std::wstring(path) == WinOf(real), "resumed: the same answer");
    EndProcess(pi);

    // a signal that a handler of the parent would catch does not run that handler in the stopped child
    struct sigaction sa{}, old{};
    sa.sa_handler = HangingHandler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, &old);
    check(CreateProcessW(nullptr, tc.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi), "start it again with a handler installed");
    kill((int)pi.dwProcessId, SIGUSR1);
    ResumeThread(pi.hThread);
    bool ended = false, execd = false;
    for (int i = 0; i < 300 && !ended && !execd; ++i) {
        ended = WaitForSingleObject(pi.hProcess, 0) == WAIT_OBJECT_0;
        execd = ExeOf((int)pi.dwProcessId) == real;
        if (!ended && !execd) Sleep(10);
    }
    check(ended || execd, "SIGUSR1 while stopped: the child ends or runs the target, it does not stay in the parent's image");
    sigaction(SIGUSR1, &old, nullptr);
    EndProcess(pi);
}

void CaseFolding() {
    check(Put(In(L"\u00C9.txt"), "e"), "create \xC3\x89.txt");
    check(GetFileAttributesW(In(L"\u00E9.txt").c_str()) != INVALID_FILE_ATTRIBUTES, "GetFileAttributesW finds it spelled with the lower-case letter");
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW(In(L"\u00E9.txt").c_str(), &fd);
    check(f != INVALID_HANDLE_VALUE && std::wstring(fd.cFileName) == L"\u00C9.txt", "FindFirstFileW finds it by the lower-case name");
    if (f != INVALID_HANDLE_VALUE) FindClose(f);
    f = FindFirstFileW(In(L"\u00E9*").c_str(), &fd);
    check(f != INVALID_HANDLE_VALUE, "and by a wildcard");
    if (f != INVALID_HANDLE_VALUE) FindClose(f);
    check(CompareStringOrdinal(L"\u00C9", 1, L"\u00E9", 1, TRUE) == CSTR_EQUAL, "CompareStringOrdinal ignores the case of an accented letter");
    check(CompareStringOrdinal(L"\u00C9", 1, L"\u00E9", 1, FALSE) != CSTR_EQUAL, "and not when asked to be exact");
    check(_wcsicmp(L"\u00C9", L"\u00E9") == 0 && _wcsnicmp(L"\u00C9x", L"\u00E9y", 1) == 0, "_wcsicmp and _wcsnicmp too");
}

void RegistryWrites() {
    // a hive whose file cannot be written: its folder is a regular file
    Put(In(L"not-a-folder"), "x");
    HKEY h = nullptr, key = nullptr;
    check(RegLoadAppKeyW(In(L"not-a-folder\\hive.dat").c_str(), &h, KEY_ALL_ACCESS, 0, 0) == ERROR_SUCCESS, "load a hive whose file cannot be written");
    const LSTATUS c = RegCreateKeyExW(h, L"K", 0, nullptr, 0, KEY_ALL_ACCESS, nullptr, &key, nullptr);
    check(c != ERROR_SUCCESS, "creating a key that cannot be saved fails");
    check(RegOpenKeyExW(h, L"K", 0, KEY_READ, &key) != ERROR_SUCCESS, "and the key is not kept in memory");
    // the value case on a key that exists in memory only because the load made the root
    check(RegSetValueExW(h, L"v", 0, REG_SZ, (const BYTE*)L"x", 4) != ERROR_SUCCESS, "setting a value that cannot be saved fails");
    wchar_t buf[8]{}; DWORD sz = sizeof(buf);
    check(RegQueryValueExW(h, L"v", nullptr, nullptr, (LPBYTE)buf, &sz) == ERROR_FILE_NOT_FOUND, "and a fresh read does not see it");
    RegCloseKey(h);
    // the error code is the one for the cause: a read-only folder is access denied
    if (geteuid() != 0) {
        const std::string dir(Tmp.begin() + 2, Tmp.end());
        std::string d = dir; for (auto& ch : d) if (ch == '\\') ch = '/';
        mkdir((d + "/ro").c_str(), 0755);
        HKEY r = nullptr;
        check(RegLoadAppKeyW(In(L"ro\\hive.dat").c_str(), &r, KEY_ALL_ACCESS, 0, 0) == ERROR_SUCCESS, "load a hive in a folder");
        chmod((d + "/ro").c_str(), 0555);
        check(RegSetValueExW(r, L"v", 0, REG_SZ, (const BYTE*)L"x", 4) == ERROR_ACCESS_DENIED, "a read-only folder: access denied");
        RegCloseKey(r);
        chmod((d + "/ro").c_str(), 0755);
    }
}

void FinishedChildren() {
    std::wstring cmd = L"true";
    std::vector<wchar_t> cb(cmd.begin(), cmd.end()); cb.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    CreateProcessW(nullptr, cb.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi);
    WaitForSingleObject(pi.hProcess, 5000);
    wgposix::ChildState c;
    check(wgposix::PollChild((int)pi.dwProcessId, c) && c.exited, "a finished child is known while its handle is open");
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    check(!wgposix::PollChild((int)pi.dwProcessId, c), "and forgotten once its handle is closed");
    wchar_t* msg = nullptr;
    check(FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER, nullptr, ERROR_INVALID_NAME, 0, (LPWSTR)&msg, 0, nullptr) > 0, "FormatMessageW has text for ERROR_INVALID_NAME");
    std::free(msg);
}

void NotAvailable() {
    check(LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32) == nullptr && GetLastError() == ERROR_MOD_NOT_FOUND, "no Windows DLL loads");
    check(GetModuleHandleW(L"nothere.dll") == nullptr && GetProcAddress(GetModuleHandleW(L"nothere.dll"), "RtlGetVersion") == nullptr,
          "and GetProcAddress of a missing module is null");
    // the one module that answers: ntdll's RtlGetVersion says Windows 10 (what a Proton prefix reports), no other export exists
    typedef LONG(WINAPI * RtlGetVersionFn)(OSVERSIONINFOW*);
    RtlGetVersionFn rtl = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(GetModuleHandleW(L"NTDLL.DLL"), "RtlGetVersion"));
    OSVERSIONINFOW ver{}; ver.dwOSVersionInfoSize = sizeof(ver);
    check(rtl && rtl(&ver) == 0 && ver.dwMajorVersion == 10 && ver.dwBuildNumber >= 10240, "ntdll's RtlGetVersion reports Windows 10");
    check(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile") == nullptr, "and ntdll has no other export");
    check(FindResourceW(nullptr, MAKEINTRESOURCEW(900), MAKEINTRESOURCEW(10)) == nullptr && LockResource(LoadResource(nullptr, nullptr)) == nullptr &&
          SizeofResource(nullptr, nullptr) == 0, "a program with no embedded resources finds none (this one links no payload table)");
    check(WinHttpOpen(L"x", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, nullptr, nullptr, 0) == nullptr, "WinHTTP does not open");
    IShellLinkW* sl = nullptr;
    check(FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW, (void**)&sl)) && !sl, "no shell link object");
    SHELLEXECUTEINFOW sei{}; sei.cbSize = sizeof(sei); sei.lpFile = L"Z:\\bin\\Game.exe"; sei.lpVerb = L"open";
    check(!ShellExecuteExW(&sei) && GetLastError() == ERROR_NOT_SUPPORTED, "ShellExecuteExW does not start a Windows program");

    URL_COMPONENTS uc{}; uc.dwStructSize = sizeof(uc);
    wchar_t host[64]{}, path[64]{};
    uc.lpszHostName = host; uc.dwHostNameLength = 64; uc.lpszUrlPath = path; uc.dwUrlPathLength = 64;
    check(WinHttpCrackUrl(L"https://example.org:8443/a/b?c=1", 0, 0, &uc) && uc.nScheme == INTERNET_SCHEME_HTTPS && uc.nPort == 8443 && std::wstring(host) == L"example.org" && std::wstring(path) == L"/a/b", "WinHttpCrackUrl: scheme, host, port, path");
    URL_COMPONENTS bad{}; bad.dwStructSize = sizeof(bad);
    check(!WinHttpCrackUrl(L"ftp://example.org/", 0, 0, &bad), "WinHttpCrackUrl: another scheme is refused");
}

// S1: the native launcher names its graphics card from Linux's device list (no DXGI here). A made-up /sys/class/drm with the
// `driver` link a real card has; every card of the folder is looked at, the AMD RDNA 4 one wins.
void GpuIdentity() {
    check(wg::NativeLinux(), "NativeLinux: this layer's ntdll carries the marker");
    const wg::WinInfo sys = wg::WindowsInfo();
    check(sys.label == L"Linux" && sys.ok && !sys.text.empty() && sys.text.find(L"Windows") == std::wstring::npos, "WindowsInfo natively: Linux, the distribution, never a Windows version");
    std::string posixTmp(Tmp.begin(), Tmp.end());
    posixTmp.erase(0, 2);
    for (auto& c : posixTmp) if (c == '\\') c = '/';
    const std::filesystem::path root = std::filesystem::path(posixTmp) / "gpu-drm";
    std::filesystem::create_directories(root / "drivers" / "amdgpu");
    std::filesystem::create_directories(root / "drivers" / "i915");
    auto card = [&](const char* name, const char* vendor, const char* device, const char* driver) {
        const auto d = root / "drm" / name / "device";
        std::filesystem::create_directories(d);
        std::FILE* f = std::fopen((d / "vendor").c_str(), "w"); std::fputs(vendor, f); std::fclose(f);
        f = std::fopen((d / "device").c_str(), "w"); std::fputs(device, f); std::fclose(f);
        if (driver) std::filesystem::create_directory_symlink(root / "drivers" / driver, d / "driver");
    };
    const std::wstring drm = Tmp + L"\\gpu-drm\\drm";
    check(wg::GpuFromDrm(drm).name.empty() && !wg::GpuFromDrm(drm).amd, "no cards: no GPU");
    card("card0", "0x8086\n", "0x46a6\n", "i915");                     // an Intel iGPU
    wg::GpuInfo g = wg::GpuFromDrm(drm);
    check(!g.amd && !g.rdna4 && g.name == L"Intel graphics (code 8086:46A6)" && g.driver == L"i915", "Intel only: named by code, driver i915, not AMD");
    card("card1", "0x1002\n", "0x744c\n", "amdgpu");                    // an RX 7900 XTX (RDNA 3)
    g = wg::GpuFromDrm(drm);
    check(g.amd && !g.rdna4 && g.name == L"AMD Radeon (code 1002:744C)" && g.driver == L"amdgpu", "RDNA 3: AMD, not RDNA 4, by code, driver amdgpu");
    card("card2", "0x1002\n", "0x7550\n", "amdgpu");                    // an RX 9070 XT
    g = wg::GpuFromDrm(drm);
    check(g.amd && g.rdna4 && g.name == L"AMD Radeon RX 9070 XT" && g.driver == L"amdgpu", "RDNA 4 after an earlier non-RDNA 4 card: it wins, named, driver amdgpu");
    card("card3", "0x1002\n", "0x7590\n", nullptr);                      // a second RDNA 4 card without a driver link: the first stays
    g = wg::GpuFromDrm(drm);
    check(g.rdna4 && g.name == L"AMD Radeon RX 9070 XT", "two RDNA 4 cards: the lower number stays");
    card("card4-DP-1", "0x1002\n", "0x7551\n", nullptr);                 // a connector, ignored
    card("card5", "garbage", "0x7551\n", nullptr);                       // unreadable ids: skipped
    check(wg::GpuFromDrm(drm).name == L"AMD Radeon RX 9070 XT", "connectors and unreadable cards are ignored");
    card("card10", "0x10de\n", "0x2684\n", nullptr);                     // NVIDIA
    check(wg::GpuFromDrm(drm).rdna4, "an NVIDIA card does not displace it");
    std::filesystem::remove_all(root / "drm" / "card2");
    std::filesystem::remove_all(root / "drm" / "card3");
    g = wg::GpuFromDrm(drm);
    check(g.amd && !g.rdna4 && g.name == L"AMD Radeon (code 1002:744C)", "without the RDNA 4 cards, the first AMD card is chosen over NVIDIA");
    const std::wstring nvOnly = Tmp + L"\\gpu-drm\\nv";
    {
        const auto d = root / "nv" / "card0" / "device";
        std::filesystem::create_directories(d);
        std::FILE* f = std::fopen((d / "vendor").c_str(), "w"); std::fputs("0x10de\n", f); std::fclose(f);
        f = std::fopen((d / "device").c_str(), "w"); std::fputs("0x2684\n", f); std::fclose(f);
    }
    g = wg::GpuFromDrm(nvOnly);
    check(!g.amd && !g.rdna4 && g.name == L"NVIDIA graphics (code 10DE:2684)" && g.driver.empty(), "NVIDIA only: not AMD, no driver link: driver unknown");
    // DetectGpu on this machine follows Linux's own list: it must agree with GpuFromDrm() on /sys/class/drm
    unsetenv("WRONGGPU_TEST_DRM_ROOT");
    const wg::GpuInfo here = wg::DetectGpu(), direct = wg::GpuFromDrm();
    check(here.name == direct.name && here.rdna4 == direct.rdna4 && here.amd == direct.amd, "DetectGpu natively is GpuFromDrm of /sys/class/drm");
    // a test build can point DetectGpu at a made-up folder (the UI tests and the screenshots use it)
    setenv("WRONGGPU_TEST_DRM_ROOT", "Z:\\nonexistent-drm-root", 1);
    check(wg::DetectGpu().name.empty(), "WRONGGPU_TEST_DRM_ROOT: a folder with no cards is no GPU");
    std::string r1 = posixTmp + "/gpu-drm/nv";
    setenv("WRONGGPU_TEST_DRM_ROOT", r1.c_str(), 1);
    check(wg::DetectGpu().name == L"NVIDIA graphics (code 10DE:2684)", "WRONGGPU_TEST_DRM_ROOT: DetectGpu reads that folder");
    unsetenv("WRONGGPU_TEST_DRM_ROOT");
}

// S5: which file the native launcher will delete as its own AppImage: every refusal, and the one yes
void AppImageIdentity() {
    std::string posixTmp(Tmp.begin(), Tmp.end());
    posixTmp.erase(0, 2);
    for (auto& c : posixTmp) if (c == '\\') c = '/';
    const std::filesystem::path dir = std::filesystem::path(posixTmp) / "appimage";
    std::filesystem::create_directories(dir);
    auto make = [&](const char* name, const char* head16) {
        std::FILE* f = std::fopen((dir / name).c_str(), "wb");
        std::fwrite(head16, 1, 16, f); std::fputs("payload", f); std::fclose(f);
        return (dir / name).string();
    };
    const char good[17] = "\x7f" "ELF\x02\x01\x01\x00" "AI\x02\x00\x00\x00\x00\x00";
    const char plainElf[17] = "\x7f" "ELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00";
    const std::string app = make("WrongGPU-0.6.5-x86_64.AppImage", good);
    const std::wstring selfW = wg::SelfPath();  // "Z:\\home\\...\\posix-layer-tests"
    std::string selfHost(selfW.begin(), selfW.end());
    selfHost.erase(0, 2);
    for (auto& c : selfHost) if (c == '\\') c = '/';
    const std::string selfDir = std::filesystem::path(selfHost).parent_path().string();
    const std::string appdir = selfDir;  // this test program stands in for the program inside the AppImage's mount
    std::wstring file, why;
    auto env = [&](const char* image, const char* appDir) {
        if (image) setenv("APPIMAGE", image, 1); else unsetenv("APPIMAGE");
        if (appDir) setenv("APPDIR", appDir, 1); else unsetenv("APPDIR");
    };
    env(nullptr, appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not running from an AppImage") != std::wstring::npos, "OwnAppImage: no $APPIMAGE is refused in words");
    env((dir / "missing.AppImage").c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not a plain file") != std::wstring::npos, "OwnAppImage: a file that is not there is refused");
    env(dir.c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not a plain file") != std::wstring::npos, "OwnAppImage: a folder is refused");
    std::filesystem::create_symlink(app, dir / "WrongGPU-link.AppImage");
    env((dir / "WrongGPU-link.AppImage").c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not a plain file") != std::wstring::npos, "OwnAppImage: a link to the AppImage is refused (the link is not the file)");
    env(make("notes.AppImage", good).c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not WrongGPU") != std::wstring::npos, "OwnAppImage: another name is refused");
    env(make("WrongGPU.txt", good).c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not WrongGPU") != std::wstring::npos, "OwnAppImage: a name without .AppImage is refused");
    env(make("WrongGPU-plain.AppImage", plainElf).c_str(), appdir.c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not an AppImage file") != std::wstring::npos, "OwnAppImage: an ELF file without the AppImage mark is refused");
    env(app.c_str(), nullptr);
    check(!wg::OwnAppImage(file, &why) && why.find(L"not inside") != std::wstring::npos, "OwnAppImage: no $APPDIR: this program is not inside it");
    env(app.c_str(), (dir / "elsewhere").c_str());
    check(!wg::OwnAppImage(file, &why) && why.find(L"not inside") != std::wstring::npos, "OwnAppImage: this program is not under $APPDIR");
    env(app.c_str(), appdir.c_str());
    file.clear();
    check(wg::OwnAppImage(file, &why) && std::string(file.begin(), file.end()) == "Z:" + [&] { std::string s = app; for (auto& c : s) if (c == '/') c = '\\'; return s; }(),
          "OwnAppImage: the file named by $APPIMAGE, a plain AppImage called WrongGPU...AppImage, this program inside its $APPDIR: yes");
    unsetenv("APPIMAGE"); unsetenv("APPDIR");
}

// S6: nothing is downloaded on Linux, and what needs a download says so in words (no WinHTTP error, no wait)
void DownloadsOff() {
    check(!wg::DownloadsAvailable(), "DownloadsAvailable: off in the native launcher");
    wg::Context ctx;
    ctx.logFile = Tmp + L"\\downloads.log";
    const wg::EditionResult r = wg::GetEdition(ctx, Tmp + L"\\downloads-data", nullptr);
    check(r.path.empty() && r.error.find(L"doesn't download anything on Linux") != std::wstring::npos && r.error.find(L"choose an OptiScaler file") != std::wstring::npos &&
              r.detail == L"downloads are off on Linux", "GetEdition: no download on Linux, the player's own OptiScaler is asked for");
    wg::GlobalSettings gs;
    gs.updateCheck = true;
    gs.updateUrl = L"https://example.org/releases/latest";
    const wg::UpdateInfo u = wg::CheckForUpdate(gs, L"0.6.5");
    check(!u.checked && !u.newer && u.message.find(L"on Linux") != std::wstring::npos && u.message.find(L"AppImage") != std::wstring::npos, "CheckForUpdate: no request on Linux, said in words");
}

void Folders() {
    PWSTR p = nullptr;
    check(SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &p)) && p && std::wstring(p).rfind(L"Z:\\", 0) == 0, "LocalAppData is a Z:\\ path");
    CoTaskMemFree(p);
    check(GetEnvironmentVariableW(L"LOCALAPPDATA", nullptr, 0) > 0, "%LOCALAPPDATA% has a value even when the environment has none");
    SetEnvironmentVariableW(L"WgLayerVar", L"v1");
    wchar_t b[16]{};
    check(GetEnvironmentVariableW(L"WGLAYERVAR", b, 16) == 2 && std::wstring(b) == L"v1", "environment names ignore case");
    SetEnvironmentVariableW(L"WGLAYERVAR", nullptr);
    check(GetEnvironmentVariableW(L"WgLayerVar", b, 16) == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND, "a variable removed: not found");
}
// Attributes, times, deleting and truncating through a handle (what the core's PinnedFile does to the files it owns).
ULONGLONG Q(const FILETIME& f) { return ((ULONGLONG)f.dwHighDateTime << 32) | f.dwLowDateTime; }
ULONGLONG FileTimeOfUnix(long long sec) { return (ULONGLONG)(sec + 11644473600LL) * 10000000ull; }
bool Exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
HANDLE OpenFor(const std::wstring& p, DWORD access, DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE) {
    return CreateFileW(p.c_str(), access, share, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}
BOOL SetDisposition(HANDLE h, DWORD flags) {
    FILE_DISPOSITION_INFO_EX ex{flags};
    return SetFileInformationByHandle(h, FileDispositionInfoEx, &ex, sizeof(ex));
}

void ErrorValues() {
    static_assert(ERROR_TOO_MANY_LINKS == 1142 && ERROR_FILE_CORRUPT == 1392 && ERROR_READ_FAULT == 30, "the Win32 values");
    static_assert(FileBasicInfo == 0 && FileDispositionInfo == 4 && FileDispositionInfoEx == 21, "the FILE_INFO_BY_HANDLE_CLASS values");
    static_assert(sizeof(FILE_BASIC_INFO) == 40, "FILE_BASIC_INFO is four 8-byte times and the attributes, padded to 8");
    check(true, "constants");
}

void BasicInfo() {
    const std::wstring p = In(L"basic.txt");
    Put(p, "x");
    HANDLE h = OpenFor(p, GENERIC_READ | FILE_WRITE_ATTRIBUTES);
    FILE_BASIC_INFO b{};
    b.FileAttributes = FILE_ATTRIBUTE_READONLY;
    check(h != INVALID_HANDLE_VALUE && SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)), "basic info: read-only set through a handle");
    check((GetFileAttributesW(p.c_str()) & FILE_ATTRIBUTE_READONLY) != 0, "  and the name reports it");
    FILE_BASIC_INFO got{};
    check(GetFileInformationByHandleEx(h, FileBasicInfo, &got, sizeof(got)) && (got.FileAttributes & FILE_ATTRIBUTE_READONLY), "  and so does the handle");
    b.FileAttributes = 0;   // 0: the attributes are left as they are
    check(SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) && (GetFileAttributesW(p.c_str()) & FILE_ATTRIBUTE_READONLY), "attributes of 0 change nothing");
    b.FileAttributes = FILE_ATTRIBUTE_NORMAL;
    check(SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) && !(GetFileAttributesW(p.c_str()) & FILE_ATTRIBUTE_READONLY), "FILE_ATTRIBUTE_NORMAL clears read-only");
    check(GetFileInformationByHandleEx(h, FileBasicInfo, &got, sizeof(got)) && !(got.FileAttributes & FILE_ATTRIBUTE_READONLY), "  the handle sees it clear");
    // times: set, then 0 leaves them
    std::memset(&b, 0, sizeof(b));
    b.LastWriteTime.QuadPart = (LONGLONG)FileTimeOfUnix(1000000000);
    b.LastAccessTime.QuadPart = (LONGLONG)FileTimeOfUnix(1000000500);
    check(SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)), "basic info: times set");
    FILETIME ca{}, ac{}, wr{};
    check(GetFileTime(h, &ca, &ac, &wr) && Q(wr) == FileTimeOfUnix(1000000000) && Q(ac) == FileTimeOfUnix(1000000500), "  the write and access times are what was set");
    check(GetFileInformationByHandleEx(h, FileBasicInfo, &got, sizeof(got)) && (ULONGLONG)got.LastWriteTime.QuadPart == FileTimeOfUnix(1000000000), "  and basic info reads the write time back");
    std::memset(&b, 0, sizeof(b));
    b.LastWriteTime.QuadPart = (LONGLONG)FileTimeOfUnix(1000001000);   // access time 0: unchanged
    check(SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) && GetFileTime(h, &ca, &ac, &wr) && Q(wr) == FileTimeOfUnix(1000001000) && Q(ac) == FileTimeOfUnix(1000000500),
          "a time of 0 is left unchanged");
    std::memset(&b, 0, sizeof(b));
    check(SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) && GetFileTime(h, &ca, &ac, &wr) && Q(wr) == FileTimeOfUnix(1000001000), "an all-zero basic info changes nothing");
    check(!GetFileInformationByHandleEx(h, FileBasicInfo, &got, sizeof(got) - 8) && GetLastError() == ERROR_INSUFFICIENT_BUFFER, "basic info into a buffer that is too small");
    CloseHandle(h);
    check(!SetFileInformationByHandle(h, FileBasicInfo, &b, sizeof(b)) && GetLastError() == ERROR_INVALID_HANDLE, "a closed handle: invalid handle");
}

void Disposition() {
    // classic: the name goes when the handle closes, not before
    std::wstring p = In(L"classic.txt");
    Put(p, "abc");
    HANDLE h = OpenFor(p, GENERIC_READ | DELETE);
    FILE_DISPOSITION_INFO d{TRUE};
    check(h != INVALID_HANDLE_VALUE && SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d)), "classic disposition set");
    check(Exists(p), "  the file is still there while the handle is open");
    CloseHandle(h);
    check(!Exists(p), "  and gone once it is closed");
    Put(p, "abc");
    h = OpenFor(p, GENERIC_READ | DELETE);
    SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d));
    d.DeleteFile = FALSE;
    check(SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d)), "classic disposition taken back");
    CloseHandle(h);
    check(Exists(p), "  the file stays");
    h = OpenFor(p, GENERIC_READ);
    d.DeleteFile = TRUE;
    check(!SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d)) && GetLastError() == ERROR_ACCESS_DENIED, "a handle without DELETE access cannot ask for delete");
    CloseHandle(h);
    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
    h = OpenFor(p, GENERIC_READ | DELETE);
    check(!SetFileInformationByHandle(h, FileDispositionInfo, &d, sizeof(d)) && GetLastError() == ERROR_ACCESS_DENIED, "classic disposition on a read-only file: access denied");
    CloseHandle(h);
    check(Exists(p), "  and it stays");
    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
    DeleteFileW(p.c_str());

    // POSIX semantics: the name is free at once, the open handle still reads
    p = In(L"posix.txt");
    Put(p, "abc");
    h = OpenFor(p, GENERIC_READ | DELETE, FILE_SHARE_READ);
    check(h != INVALID_HANDLE_VALUE && SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS), "POSIX disposition set");
    check(!Exists(p) && GetLastError() == ERROR_FILE_NOT_FOUND, "  the name is gone at once");
    char buf[8]{}; DWORD got = 0;
    check(ReadFile(h, buf, sizeof(buf), &got, nullptr) && got == 3 && std::string(buf, got) == "abc", "  the handle still reads the file");
    check(Put(p, "new"), "  the name can be used again");
    CloseHandle(h);
    check(Slurp(p) == "new", "  and closing the old handle leaves the new file alone");
    DeleteFileW(p.c_str());

    // IGNORE_READONLY_ATTRIBUTE
    p = In(L"posixro.txt");
    Put(p, "abc");
    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
    h = OpenFor(p, GENERIC_READ | DELETE);
    check(!SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS) && GetLastError() == ERROR_ACCESS_DENIED && Exists(p),
          "POSIX disposition on a read-only file: access denied, file kept");
    check(SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE) && !Exists(p),
          "  with IGNORE_READONLY_ATTRIBUTE the read-only file is deleted");
    CloseHandle(h);

    // Ex without POSIX semantics is the classic delete; without the DELETE flag it takes a request back
    p = In(L"exclassic.txt");
    Put(p, "abc");
    h = OpenFor(p, GENERIC_READ | DELETE);
    check(SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE) && Exists(p), "Ex disposition without POSIX semantics: the file waits for the close");
    check(SetDisposition(h, FILE_DISPOSITION_FLAG_DO_NOT_DELETE), "  flags 0 take the request back");
    CloseHandle(h);
    check(Exists(p), "  the file stays");
    DeleteFileW(p.c_str());
}

void EndOfFile() {
    const std::wstring p = In(L"eof.txt");
    Put(p, "0123456789");
    HANDLE h = OpenFor(p, GENERIC_READ | GENERIC_WRITE);
    LARGE_INTEGER pos{}, size{};
    pos.QuadPart = 4;
    check(h != INVALID_HANDLE_VALUE && SetFilePointerEx(h, pos, nullptr, FILE_BEGIN) && SetEndOfFile(h) && GetFileSizeEx(h, &size) && size.QuadPart == 4, "SetEndOfFile at 4 cuts the file to 4 bytes");
    check(Slurp(p) == "0123", "  and keeps the first four");
    pos.QuadPart = 10;
    check(SetFilePointerEx(h, pos, nullptr, FILE_BEGIN) && SetEndOfFile(h) && GetFileSizeEx(h, &size) && size.QuadPart == 10, "SetEndOfFile at 10 extends the file to 10 bytes");
    check(Slurp(p) == std::string("0123\0\0\0\0\0\0", 10), "  and the new bytes are zero");
    CloseHandle(h);
    h = OpenFor(p, GENERIC_READ);
    check(!SetEndOfFile(h) && GetLastError() == ERROR_ACCESS_DENIED, "SetEndOfFile on a handle opened for reading: access denied");
    CloseHandle(h);
    check(!SetEndOfFile(h) && GetLastError() == ERROR_INVALID_HANDLE, "SetEndOfFile on a closed handle: invalid handle");
}

void PinnedPattern() {
    // What PinnedFile::Open does: a probe with attribute access only and every sharing mode, then the real handle with
    // read, delete and attribute access and FILE_SHARE_READ only, while the probe is still open.
    for (const bool readOnly : {false, true}) {
        const std::wstring p = In(L"pin.txt");
        Put(p, "pinned");
        if (readOnly) SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_READONLY);
        HANDLE probe = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES | FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                   FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
        check(probe != INVALID_HANDLE_VALUE, "pinning: the probe opens, even on a read-only file");
        BY_HANDLE_FILE_INFORMATION pi{}, bi{};
        check(GetFileInformationByHandle(probe, &pi) && ((pi.dwFileAttributes & FILE_ATTRIBUTE_READONLY) != 0) == readOnly, "  the probe sees the read-only bit as it is");
        FILE_BASIC_INFO b{};
        b.FileAttributes = FILE_ATTRIBUTE_NORMAL;
        check(SetFileInformationByHandle(probe, FileBasicInfo, &b, sizeof(b)), "  the probe clears it (attribute access is enough)");
        HANDLE real = CreateFileW(p.c_str(), GENERIC_READ | DELETE | FILE_WRITE_ATTRIBUTES | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
        check(real != INVALID_HANDLE_VALUE, "  the real handle opens while the probe is open (attribute access takes no part in sharing)");
        check(real != INVALID_HANDLE_VALUE && GetFileInformationByHandle(real, &bi) && bi.dwVolumeSerialNumber == pi.dwVolumeSerialNumber && bi.nFileIndexHigh == pi.nFileIndexHigh &&
                  bi.nFileIndexLow == pi.nFileIndexLow && (pi.nFileIndexHigh | pi.nFileIndexLow) != 0,
              "  both handles name the same file (volume serial and index)");
        check(CreateFileW(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr) == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION,
              "  but a writer is still kept out by FILE_SHARE_READ only");
        check(CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, 0, nullptr, OPEN_EXISTING, 0, nullptr) != INVALID_HANDLE_VALUE, "  and an attribute-only open takes no part in sharing either");
        if (real != INVALID_HANDLE_VALUE) {
            check(SetDisposition(real, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS | FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE) && !Exists(p), "  and it can delete the file");
            CloseHandle(real);
        }
        CloseHandle(probe);
    }
}

void LinksAndReparse() {
    const std::wstring a = In(L"nl-a.txt"), l = In(L"nl-b.txt"), s = In(L"nl-sym.txt");
    Put(a, "linked");
    BY_HANDLE_FILE_INFORMATION bi{};
    HANDLE h = OpenFor(a, FILE_READ_ATTRIBUTES);
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && bi.nNumberOfLinks == 1, "a file with one name reports 1 link");
    CloseHandle(h);
    check(CreateHardLinkW(l.c_str(), a.c_str(), nullptr), "a hard link");
    h = OpenFor(a, FILE_READ_ATTRIBUTES);
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && bi.nNumberOfLinks == 2, "a hard-linked file reports 2 links");
    CloseHandle(h);
    h = OpenFor(l, FILE_READ_ATTRIBUTES);
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && bi.nNumberOfLinks == 2, "  by either name");
    CloseHandle(h);
    check(CreateSymbolicLinkW(s.c_str(), a.c_str(), 0), "a symbolic link to a file");
    h = CreateFileW(s.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && (bi.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT), "FILE_FLAG_OPEN_REPARSE_POINT: a symbolic link reports the reparse attribute");
    FILE_BASIC_INFO fb{};
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandleEx(h, FileBasicInfo, &fb, sizeof(fb)) && (fb.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT), "  in basic info too");
    CloseHandle(h);
    h = OpenFor(s, FILE_READ_ATTRIBUTES);
    check(h != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h, &bi) && !(bi.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && bi.nNumberOfLinks == 2,
          "without the flag the link is followed: the target's attributes, no reparse point");
    CloseHandle(h);
    DeleteFileW(s.c_str()); DeleteFileW(l.c_str()); DeleteFileW(a.c_str());
}

void ProcessEnd() {
    void* p = std::malloc(16);
    check(GetProcessHeap() != nullptr && HeapFree(GetProcessHeap(), 0, p), "HeapFree frees what the process heap hands out");
    const pid_t pid = fork();
    if (pid == 0) ExitProcess(7);
    int status = 0;
    check(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 7, "ExitProcess(7) ends the process with exit code 7");
}
}  // namespace

// ReplaceFileW: the replacement takes the name of the replaced file; documented failure codes (1175 to 1177 are the numbers Windows gives).
void ReplaceFile() {
    check(ERROR_UNABLE_TO_REMOVE_REPLACED == 1175 && ERROR_UNABLE_TO_MOVE_REPLACEMENT == 1176 && ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 == 1177, "the replace failure codes are 1175, 1176 and 1177");
    Put(In(L"live.ini"), "old"); Put(In(L"fresh.ini"), "new");
    check(ReplaceFileW(In(L"live.ini").c_str(), In(L"fresh.ini").c_str(), In(L"live.bak").c_str(), 0, nullptr, nullptr), "ReplaceFileW with a backup name");
    check(Slurp(In(L"live.ini")) == "new" && Slurp(In(L"live.bak")) == "old" && GetFileAttributesW(In(L"fresh.ini").c_str()) == INVALID_FILE_ATTRIBUTES,
          "the replacement is at the old name, the old file at the backup name, the replacement's own name is free");
    Put(In(L"fresh2.ini"), "newer");
    check(ReplaceFileW(In(L"live.ini").c_str(), In(L"fresh2.ini").c_str(), In(L"live.bak").c_str(), REPLACEFILE_WRITE_THROUGH, nullptr, nullptr) && Slurp(In(L"live.bak")) == "new" && Slurp(In(L"live.ini")) == "newer",
          "an existing backup is overwritten");
    Put(In(L"fresh3.ini"), "third");
    check(ReplaceFileW(In(L"live.ini").c_str(), In(L"fresh3.ini").c_str(), nullptr, 0, nullptr, nullptr) && Slurp(In(L"live.ini")) == "third" && GetFileAttributesW(In(L"fresh3.ini").c_str()) == INVALID_FILE_ATTRIBUTES,
          "ReplaceFileW without a backup name");

    check(!ReplaceFileW(In(L"absent.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_FILE_NOT_FOUND && Slurp(In(L"live.bak")) == "new" ,
          "a replaced file that does not exist: file not found, the replacement stays");
    check(!ReplaceFileW(In(L"live.ini").c_str(), In(L"absent.ini").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_FILE_NOT_FOUND && Slurp(In(L"live.ini")) == "third",
          "a replacement that does not exist: file not found, the replaced file stays");
    check(!ReplaceFileW(In(L"nope\\live.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_PATH_NOT_FOUND, "a missing folder: path not found");
    int dummy = 0;
    check(!ReplaceFileW(In(L"live.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, &dummy, nullptr) && GetLastError() == ERROR_INVALID_PARAMETER, "the reserved arguments must be null");

    SetFileAttributesW(In(L"live.ini").c_str(), FILE_ATTRIBUTE_READONLY);
    check(!ReplaceFileW(In(L"live.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_ACCESS_DENIED && Slurp(In(L"live.ini")) == "third" && Slurp(In(L"live.bak")) == "new",
          "a read-only replaced file: access denied, nothing moved");
    SetFileAttributesW(In(L"live.ini").c_str(), FILE_ATTRIBUTE_NORMAL);

    HANDLE held = CreateFileW(In(L"live.ini").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    check(!ReplaceFileW(In(L"live.ini").c_str(), In(L"live.bak").c_str(), In(L"live.old").c_str(), 0, nullptr, nullptr) && GetLastError() == ERROR_UNABLE_TO_REMOVE_REPLACED
              && Slurp(In(L"live.ini")) == "third" && Slurp(In(L"live.bak")) == "new" && GetFileAttributesW(In(L"live.old").c_str()) == INVALID_FILE_ATTRIBUTES,
          "a replaced file held without delete sharing: 1175, both files keep their names");
    CloseHandle(held);

    held = CreateFileW(In(L"live.bak").c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    check(!ReplaceFileW(In(L"live.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_SHARING_VIOLATION && Slurp(In(L"live.ini")) == "third",
          "a replacement held without delete sharing: sharing violation");
    CloseHandle(held);

    SetFileAttributesW(In(L"live.bak").c_str(), FILE_ATTRIBUTE_READONLY);
    check(ReplaceFileW(In(L"live.ini").c_str(), In(L"live.bak").c_str(), nullptr, 0, nullptr, nullptr) && Slurp(In(L"live.ini")) == "new" && (GetFileAttributesW(In(L"live.ini").c_str()) & FILE_ATTRIBUTE_READONLY) == 0,
          "the replaced file's attributes stay on the new file (a read-only replacement does not make it read-only)");

    Put(In(L"h-live.ini"), "a"); Put(In(L"h-fresh.ini"), "b");
    SetFileAttributesW(In(L"h-live.ini").c_str(), FILE_ATTRIBUTE_HIDDEN);
    check((GetFileAttributesW(In(L"h-live.ini").c_str()) & FILE_ATTRIBUTE_HIDDEN) && ReplaceFileW(In(L"h-live.ini").c_str(), In(L"h-fresh.ini").c_str(), nullptr, 0, nullptr, nullptr)
              && Slurp(In(L"h-live.ini")) == "b" && (GetFileAttributesW(In(L"h-live.ini").c_str()) & FILE_ATTRIBUTE_HIDDEN),
          "the hidden attribute of the replaced file stays on the new file");
    SetFileAttributesW(In(L"h-live.ini").c_str(), FILE_ATTRIBUTE_NORMAL);
    check((GetFileAttributesW(In(L"h-live.ini").c_str()) & FILE_ATTRIBUTE_HIDDEN) == 0, "a normal attribute clears hidden");

    check(CreateDirectoryW(In(L"rd").c_str(), nullptr) && !ReplaceFileW(In(L"live.ini").c_str(), In(L"rd").c_str(), nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_ACCESS_DENIED, "a folder cannot replace a file");
    check(!ReplaceFileW(L"Z:\\no\\such\\a", L"Z:\\no\\such\\b", nullptr, 0, nullptr, nullptr) && GetLastError() == ERROR_PATH_NOT_FOUND, "Z:\\ paths are read as everywhere else");
}

// ---- 0.6.5's calls: the named mutex (the per-game state lock), events, rename by handle, the long path, security
// descriptors, the Restart Manager. The other processes are this program started again with a --child-* argument.
std::string SelfExe() {
    char b[4096];
    const ssize_t n = readlink("/proc/self/exe", b, sizeof(b) - 1);
    return n > 0 ? std::string(b, (size_t)n) : std::string();
}
std::string PosixOf(const std::wstring& w) {   // Tmp's Z:\ spelling as the POSIX path a child gets
    std::string p(w.begin(), w.end());
    if (p.rfind("Z:", 0) == 0) p.erase(0, 2);
    for (auto& c : p) if (c == '\\') c = '/';
    return p;
}
PROCESS_INFORMATION StartChild(const std::string& args) {
    const std::string self = SelfExe();
    std::wstring cmd = L"\"" + std::wstring(self.begin(), self.end()) + L"\" ";
    cmd += std::wstring(args.begin(), args.end());
    std::vector<wchar_t> b(cmd.begin(), cmd.end());
    b.push_back(0);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, b.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) pi.hProcess = nullptr;
    return pi;
}
bool WaitReady(const std::string& file, int ms = 10000) {
    for (int i = 0; i < ms / 5; ++i) {
        if (access(file.c_str(), F_OK) == 0) return true;
        Sleep(5);
    }
    return false;
}
void EndChild(PROCESS_INFORMATION& pi) {
    if (!pi.hProcess) return;
    if (WaitForSingleObject(pi.hProcess, 0) == WAIT_TIMEOUT) TerminateProcess(pi.hProcess, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
}
long long MsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count();
}
std::wstring Name(const char* base) { return L"Global\\wg-posix-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::wstring(base, base + std::strlen(base)); }
std::string Narrow(const std::wstring& w) { return std::string(w.begin(), w.end()); }

// the children
int ChildMain(int argc, char** argv) {
    const std::string mode = argv[1];
    auto ready = [](const char* f) { const int fd = open(f, O_CREAT | O_WRONLY | O_CLOEXEC, 0600); if (fd >= 0) close(fd); };
    if (mode == "--child-mutex" && argc == 6) {   // <name> <ms> <ready> release|die
        const std::wstring name(argv[2], argv[2] + std::strlen(argv[2]));
        HANDLE m = CreateMutexW(nullptr, FALSE, name.c_str());
        if (!m || WaitForSingleObject(m, 10000) != WAIT_OBJECT_0) return 2;
        ready(argv[4]);
        Sleep((DWORD)std::atoi(argv[3]));
        if (std::string(argv[5]) == "die") _exit(3);   // ends owning it
        ReleaseMutex(m);
        CloseHandle(m);
        return 0;
    }
    if ((mode == "--child-open" || mode == "--child-map") && argc == 4) {   // <file> <ready>
        const int fd = open(argv[2], O_RDONLY | O_CLOEXEC);
        if (fd < 0) return 2;
        if (mode == "--child-map") {   // only the mapping holds it, as a Wine process holds a DLL it loaded
            struct stat st;
            fstat(fd, &st);
            if (mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) == MAP_FAILED) return 3;
            close(fd);
        }
        ready(argv[3]);
        Sleep(30000);
        return 0;
    }
    if (mode == "--child-nodump" && argc == 3) {   // its /proc fd and maps become closed to its own user
        prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
        ready(argv[2]);
        Sleep(30000);
        return 0;
    }
    return 9;
}

void NamedMutex() {
    const std::wstring a = Name("a");
    HANDLE m = CreateMutexW(nullptr, FALSE, a.c_str());
    check(m && GetLastError() == 0, "mutex: a new name: created, not ERROR_ALREADY_EXISTS");
    HANDLE m2 = CreateMutexW(nullptr, FALSE, a.c_str());
    check(m2 && GetLastError() == ERROR_ALREADY_EXISTS, "mutex: the same name again while a handle is open: ERROR_ALREADY_EXISTS");
    HANDLE o = OpenMutexW(SYNCHRONIZE | MUTEX_MODIFY_STATE, FALSE, a.c_str());
    check(o != nullptr, "mutex: OpenMutexW of an open name");
    check(!OpenMutexW(SYNCHRONIZE, FALSE, Name("never").c_str()) && GetLastError() == ERROR_FILE_NOT_FOUND, "mutex: OpenMutexW of a name nobody has open: ERROR_FILE_NOT_FOUND");
    check(!CreateMutexW(nullptr, FALSE, nullptr) && GetLastError() == ERROR_NOT_SUPPORTED, "mutex: an unnamed mutex: ERROR_NOT_SUPPORTED");
    check(!CreateMutexW(nullptr, FALSE, std::wstring(300, L'x').c_str()) && GetLastError() == ERROR_INVALID_NAME, "mutex: a name too long for a file: ERROR_INVALID_NAME");
    check(!ReleaseMutex(m) && GetLastError() == ERROR_NOT_OWNER, "mutex: ReleaseMutex of a mutex nobody owns: ERROR_NOT_OWNER");
    check(WaitForSingleObject(m, 0) == WAIT_OBJECT_0, "mutex: taken");
    check(WaitForSingleObject(m2, 0) == WAIT_OBJECT_0, "mutex: taken again by its owner through another handle (recursive, per thread)");
    std::atomic<DWORD> other{0}, otherRelease{0}, otherError{0};
    long long waited = 0;
    std::thread t([&] {
        const auto t0 = std::chrono::steady_clock::now();
        other = WaitForSingleObject(o, 150);
        waited = MsSince(t0);
        otherRelease = ReleaseMutex(o);
        otherError = GetLastError();
    });
    t.join();
    check(other == WAIT_TIMEOUT && waited >= 140 && waited < 2000, "mutex: another thread of this process waits its timeout, then WAIT_TIMEOUT");
    check(!otherRelease && otherError == ERROR_NOT_OWNER, "mutex: another thread cannot release it (ERROR_NOT_OWNER)");
    check(ReleaseMutex(m2) && ReleaseMutex(m), "mutex: released as often as taken");
    check(!ReleaseMutex(m) && GetLastError() == ERROR_NOT_OWNER, "mutex: one release too many: ERROR_NOT_OWNER");
    std::thread t2([&] { other = WaitForSingleObject(o, 1000); ReleaseMutex(o); });
    t2.join();
    check(other == WAIT_OBJECT_0, "mutex: once released, another thread takes it");
    // a thread that waits while another holds it gets it when it is released, inside its wait
    WaitForSingleObject(m, 0);
    std::thread t3([&] { other = WaitForSingleObject(o, 3000); ReleaseMutex(o); });
    Sleep(100);
    ReleaseMutex(m);
    t3.join();
    check(other == WAIT_OBJECT_0, "mutex: a waiting thread gets it once the owner releases it");
    CloseHandle(m2);
    CloseHandle(m);
    HANDLE again = CreateMutexW(nullptr, FALSE, a.c_str());
    check(again && GetLastError() == ERROR_ALREADY_EXISTS, "mutex: kept by an opened handle: the creators' handles closed, it still exists");
    CloseHandle(again);
    CloseHandle(o);
    again = CreateMutexW(nullptr, FALSE, a.c_str());
    check(again && GetLastError() == 0, "mutex: every handle closed: the name is new again");
    CloseHandle(again);

    // between processes
    const std::wstring b = Name("b");
    const std::string readyB = PosixOf(In(L"ready-b"));
    PROCESS_INFORMATION pi = StartChild("--child-mutex " + Narrow(b) + " 800 " + readyB + " release");
    check(pi.hProcess && WaitReady(readyB), "mutex: (a child holds it)");
    HANDLE mb = CreateMutexW(nullptr, FALSE, b.c_str());
    check(mb && GetLastError() == ERROR_ALREADY_EXISTS, "mutex: another process's handle: ERROR_ALREADY_EXISTS here");
    auto t0 = std::chrono::steady_clock::now();
    const DWORD busy = WaitForSingleObject(mb, 200);
    const long long ms = MsSince(t0);
    check(busy == WAIT_TIMEOUT && ms >= 190 && ms < 2000, "mutex: held by another process: WAIT_TIMEOUT after the wait asked for");
    check(WaitForSingleObject(mb, 10000) == WAIT_OBJECT_0, "mutex: the other process releases it: taken (not abandoned)");
    ReleaseMutex(mb);
    EndChild(pi);
    CloseHandle(mb);

    // an owner that ends without releasing it: abandoned for whoever kept the object alive; a fresh object for a new creator
    const std::wstring c = Name("c");
    const std::string readyC = PosixOf(In(L"ready-c"));
    pi = StartChild("--child-mutex " + Narrow(c) + " 300 " + readyC + " die");
    check(pi.hProcess && WaitReady(readyC), "mutex: (a child holds it and will end owning it)");
    HANDLE keep = OpenMutexW(SYNCHRONIZE, FALSE, c.c_str());
    WaitForSingleObject(pi.hProcess, 5000);
    check(keep && WaitForSingleObject(keep, 2000) == WAIT_ABANDONED, "mutex: its owner died, a handle kept it: WAIT_ABANDONED, and it is ours");
    check(ReleaseMutex(keep), "mutex: an abandoned mutex taken is released as usual");
    check(WaitForSingleObject(keep, 0) == WAIT_OBJECT_0 && ReleaseMutex(keep), "mutex: after that release it is a normal mutex again");
    CloseHandle(keep);
    EndChild(pi);
    const std::wstring d = Name("d");
    const std::string readyD = PosixOf(In(L"ready-d"));
    pi = StartChild("--child-mutex " + Narrow(d) + " 100 " + readyD + " die");
    check(pi.hProcess && WaitReady(readyD), "mutex: (a child holds a second one and will end owning it)");
    WaitForSingleObject(pi.hProcess, 5000);
    HANDLE fresh = CreateMutexW(nullptr, FALSE, d.c_str());
    check(fresh && GetLastError() == 0 && WaitForSingleObject(fresh, 0) == WAIT_OBJECT_0, "mutex: no handle kept it: a new mutex, not abandoned (as on Windows)");
    ReleaseMutex(fresh);
    CloseHandle(fresh);
    EndChild(pi);

    // a lock file this user may not open: access denied, not a timeout
    const std::wstring e = Name("e");
    HANDLE me = CreateMutexW(nullptr, FALSE, e.c_str());
    const char* rt = getenv("XDG_RUNTIME_DIR");
    const std::string lockFile = std::string(rt && *rt ? rt : "") + "/wronggpu-mutex-Global%5C" + Narrow(e).substr(7) + ".lock";
    CloseHandle(me);
    if (geteuid() == 0 || !rt || !*rt) {
        std::printf("posix-layer: running as root or without XDG_RUNTIME_DIR: the lock-file permission check is skipped\n");
    } else {
        check(access(lockFile.c_str(), F_OK) == 0, "mutex: (its lock file is in XDG_RUNTIME_DIR)");
        chmod(lockFile.c_str(), 0);
        check(!CreateMutexW(nullptr, FALSE, e.c_str()) && GetLastError() == ERROR_ACCESS_DENIED, "mutex: a lock file this user may not open: CreateMutexW ERROR_ACCESS_DENIED");
        check(!OpenMutexW(SYNCHRONIZE, FALSE, e.c_str()) && GetLastError() == ERROR_ACCESS_DENIED, "mutex: and OpenMutexW ERROR_ACCESS_DENIED");
        chmod(lockFile.c_str(), 0600);
    }
    // this run's lock files (a product run keeps them: one per name in the runtime folder, which the session clears)
    if (rt && *rt)
        for (const char* n : {"a", "b", "c", "d", "e"}) unlink((std::string(rt) + "/wronggpu-mutex-Global%5C" + Narrow(Name(n)).substr(7) + ".lock").c_str());
}

void Events() {
    HANDLE ev = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    check(ev && WaitForSingleObject(ev, 0) == WAIT_TIMEOUT, "event: manual reset, not set: WAIT_TIMEOUT");
    std::thread t([&] { Sleep(50); SetEvent(ev); });
    check(WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0, "event: set by another thread inside the wait");
    t.join();
    check(WaitForSingleObject(ev, 0) == WAIT_OBJECT_0, "event: a manual-reset event stays set");
    CloseHandle(ev);
    HANDLE au = CreateEventW(nullptr, FALSE, TRUE, nullptr);
    check(au && WaitForSingleObject(au, 0) == WAIT_OBJECT_0 && WaitForSingleObject(au, 0) == WAIT_TIMEOUT, "event: an auto-reset event lets one wait through");
    CloseHandle(au);
    check(!CreateEventW(nullptr, TRUE, FALSE, L"named") && GetLastError() == ERROR_NOT_SUPPORTED, "event: a named event: ERROR_NOT_SUPPORTED");
    check(!SetEvent(nullptr) && GetLastError() == ERROR_INVALID_HANDLE, "event: SetEvent of no handle: ERROR_INVALID_HANDLE");
    volatile LONG n = 0;
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) ts.emplace_back([&] { for (int k = 0; k < 10000; ++k) InterlockedIncrement(&n); });
    for (auto& x : ts) x.join();
    check(n == 40000, "InterlockedIncrement: no increment lost between threads");
}

bool RenameTo(HANDLE h, const std::wstring& name, bool replace) {
    std::vector<unsigned char> buf(sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t), 0);
    FILE_RENAME_INFO* ri = reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
    ri->ReplaceIfExists = replace ? TRUE : FALSE;
    ri->FileNameLength = (DWORD)(name.size() * sizeof(wchar_t));
    std::memcpy(ri->FileName, name.c_str(), name.size() * sizeof(wchar_t));
    return SetFileInformationByHandle(h, FileRenameInfo, ri, (DWORD)buf.size()) != 0;
}

void RenameByHandle() {
    const std::wstring a = In(L"rn-a.dll"), b = In(L"rn-b.dll"), c = In(L"rn-c.dll");
    Put(a, "old");
    HANDLE h = OpenFor(a, DELETE | GENERIC_READ | FILE_WRITE_DATA, FILE_SHARE_READ);   // as the operation guard holds a binary
    check(RenameTo(h, L"\\\\?\\" + b, false) && !Exists(a) && Slurp(b) == "old", "rename by handle: to a free name (\\\\?\\ form)");
    check(SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS) && !Exists(b), "  a delete through the handle then deletes the new name");
    CloseHandle(h);
    Put(a, "old");
    h = OpenFor(a, DELETE | GENERIC_READ, FILE_SHARE_READ);
    check(RenameTo(h, L"\\??\\" + b, false) && Slurp(b) == "old", "rename by handle: a name in the NT form (\\??\\Z:\\...)");
    Put(c, "taken");
    check(!RenameTo(h, c, false) && GetLastError() == ERROR_ALREADY_EXISTS && Slurp(b) == "old" && Slurp(c) == "taken", "rename by handle: onto a file without replace: ERROR_ALREADY_EXISTS, both as they were");
    HANDLE busy = OpenFor(c, GENERIC_READ, FILE_SHARE_READ);
    check(!RenameTo(h, c, true) && GetLastError() == ERROR_ACCESS_DENIED && Slurp(c) == "taken" && Slurp(b) == "old", "rename by handle: replace over a file open without delete sharing: ERROR_ACCESS_DENIED, nothing moved");
    CloseHandle(busy);
    SetFileAttributesW(c.c_str(), FILE_ATTRIBUTE_READONLY);
    check(!RenameTo(h, c, true) && GetLastError() == ERROR_ACCESS_DENIED && Slurp(c) == "taken", "rename by handle: replace over a read-only file: ERROR_ACCESS_DENIED");
    SetFileAttributesW(c.c_str(), FILE_ATTRIBUTE_NORMAL);
    check(RenameTo(h, c, true) && Slurp(c) == "old" && !Exists(b), "rename by handle: replace over a closed file: replaced");
    // renamed by someone else meanwhile (by name; it shares delete): the handle's rename still moves its own file
    CloseHandle(h);
    h = OpenFor(c, DELETE | GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE);
    check(MoveFileExW(c.c_str(), b.c_str(), 0) && RenameTo(h, a, false) && Slurp(a) == "old" && !Exists(b) && !Exists(c), "rename by handle: after another rename by name, the file's current name is used");
    CloseHandle(h);
    h = OpenFor(a, GENERIC_READ);
    check(!RenameTo(h, b, false) && GetLastError() == ERROR_ACCESS_DENIED && Exists(a), "rename by handle: a handle without DELETE access: ERROR_ACCESS_DENIED");
    CloseHandle(h);
    h = OpenFor(a, DELETE | GENERIC_READ);
    std::vector<unsigned char> buf(sizeof(FILE_RENAME_INFO) + 8, 0);
    FILE_RENAME_INFO* ri = reinterpret_cast<FILE_RENAME_INFO*>(buf.data());
    ri->RootDirectory = h;
    ri->FileNameLength = sizeof(wchar_t);
    ri->FileName[0] = L'x';
    check(!SetFileInformationByHandle(h, FileRenameInfo, ri, (DWORD)buf.size()) && GetLastError() == ERROR_INVALID_PARAMETER, "rename by handle: a name relative to a folder handle: ERROR_INVALID_PARAMETER");
    ri->RootDirectory = nullptr;
    ri->FileNameLength = 400;
    check(!SetFileInformationByHandle(h, FileRenameInfo, ri, (DWORD)buf.size()) && GetLastError() == ERROR_INVALID_PARAMETER, "rename by handle: a name longer than the buffer: ERROR_INVALID_PARAMETER");
    // a folder this user may not write: access denied, the file where it was
    const std::wstring ro = In(L"rn-ro");
    CreateDirectoryW(ro.c_str(), nullptr);
    if (geteuid() == 0) {
        std::printf("posix-layer: running as root: the read-only folder rename check is skipped\n");
    } else {
        chmod(PosixOf(ro).c_str(), 0555);
        check(!RenameTo(h, ro + L"\\x.dll", false) && GetLastError() == ERROR_ACCESS_DENIED && Exists(a), "rename by handle: into a folder this user may not write: ERROR_ACCESS_DENIED");
        chmod(PosixOf(ro).c_str(), 0755);
    }
    check(!RenameTo(h, L"Z:\\no\\such\\folder\\x.dll", false) && GetLastError() == ERROR_PATH_NOT_FOUND && Exists(a), "rename by handle: into a folder that is not there: ERROR_PATH_NOT_FOUND");
    CloseHandle(h);
    DeleteFileW(a.c_str());
}

void LongPathAndSecurity() {
    Put(In(L"LongName.Txt"), "x");
    wchar_t out[MAX_PATH * 2];
    DWORD n = GetLongPathNameW(In(L"longname.txt").c_str(), out, MAX_PATH * 2);
    check(n && std::wstring(out, n) == In(L"LongName.Txt"), "GetLongPathNameW: the path, its names as they are on disk");
    n = GetLongPathNameW((L"\\\\?\\" + In(L"LongName.Txt")).c_str(), out, MAX_PATH * 2);
    check(n && std::wstring(out, n) == L"\\\\?\\" + In(L"LongName.Txt"), "GetLongPathNameW: a \\\\?\\ prefix stays");
    const DWORD need = GetLongPathNameW(In(L"LongName.Txt").c_str(), out, 3);
    check(need == In(L"LongName.Txt").size() + 1, "GetLongPathNameW: a short buffer: the size needed, with the NUL");
    check(!GetLongPathNameW(In(L"none.txt").c_str(), out, MAX_PATH) && GetLastError() == ERROR_FILE_NOT_FOUND, "GetLongPathNameW: a missing file: ERROR_FILE_NOT_FOUND");
    check(!GetLongPathNameW(In(L"none\\x.txt").c_str(), out, MAX_PATH) && GetLastError() == ERROR_PATH_NOT_FOUND, "GetLongPathNameW: a missing folder: ERROR_PATH_NOT_FOUND");
    // through a symbolic link: the long form keeps the link's name, the final form is where it leads (what util.cpp compares)
    CreateDirectoryW(In(L"lp-real").c_str(), nullptr);
    Put(In(L"lp-real\\f.txt"), "x");
    if (symlink(PosixOf(In(L"lp-real")).c_str(), PosixOf(In(L"lp-link")).c_str()) == 0) {
        n = GetLongPathNameW(In(L"lp-link\\f.txt").c_str(), out, MAX_PATH * 2);
        HANDLE h = OpenFor(In(L"lp-link\\f.txt"), GENERIC_READ);
        wchar_t fin[MAX_PATH * 2];
        const DWORD m = GetFinalPathNameByHandleW(h, fin, MAX_PATH * 2, 0);
        CloseHandle(h);
        check(n && std::wstring(out, n) == In(L"lp-link\\f.txt") && m && std::wstring(fin, m) == In(L"lp-real\\f.txt"),
              "GetLongPathNameW: a link is not followed (its final path is the target's)");
    }
    PSECURITY_DESCRIPTOR sd = reinterpret_cast<PSECURITY_DESCRIPTOR>(1);
    check(!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:(A;;GA;;;WD)", SDDL_REVISION_1, &sd, nullptr) && GetLastError() == ERROR_NOT_SUPPORTED && sd == nullptr,
          "security descriptors: not available (FALSE, ERROR_NOT_SUPPORTED, no descriptor)");
}

UINT HoldersOf(const std::wstring& file, RM_PROCESS_INFO* info, UINT cap, DWORD* reasons, DWORD* rc = nullptr, UINT* needed = nullptr) {
    DWORD s = 0;
    WCHAR key[CCH_RM_SESSION_KEY + 1] = {};
    if (RmStartSession(&s, 0, key) != ERROR_SUCCESS) return 999;
    LPCWSTR files[] = {file.c_str()};
    RmRegisterResources(s, 1, files, 0, nullptr, 0, nullptr);
    UINT need = 0, n = cap;
    const DWORD r = RmGetList(s, &need, &n, info, reasons);
    RmEndSession(s);
    if (rc) *rc = r;
    if (needed) *needed = need;
    return r == ERROR_SUCCESS ? n : 999;
}
bool Lists(const RM_PROCESS_INFO* info, UINT n, DWORD pid) {
    for (UINT i = 0; i < n; ++i) if (info[i].Process.dwProcessId == pid) return true;
    return false;
}

void RestartManager() {
    const std::wstring f = In(L"rm-held.dll"), g = In(L"rm-mapped.dll"), idle = In(L"rm-free.dll");
    Put(f, "MZ held");
    Put(g, "MZ mapped");
    Put(idle, "MZ free");
    RM_PROCESS_INFO info[16];
    DWORD reasons = 0;
    check(HoldersOf(idle, info, 16, &reasons) == 0, "Restart Manager: a file nobody holds: none listed");
    HANDLE mine = OpenFor(f, GENERIC_READ);
    UINT n = HoldersOf(f, info, 16, &reasons);
    check(n != 999 && Lists(info, n, GetCurrentProcessId()), "Restart Manager: a file this process holds: this process listed");
    CloseHandle(mine);
    const std::string r1 = PosixOf(In(L"ready-rm1")), r2 = PosixOf(In(L"ready-rm2"));
    PROCESS_INFORMATION po = StartChild("--child-open " + PosixOf(f) + " " + r1);
    PROCESS_INFORMATION pm = StartChild("--child-map " + PosixOf(g) + " " + r2);
    check(po.hProcess && pm.hProcess && WaitReady(r1) && WaitReady(r2), "Restart Manager: (two children hold the files)");
    n = HoldersOf(f, info, 16, &reasons);
    bool named = false;
    for (UINT i = 0; i < n && n != 999; ++i)
        if (info[i].Process.dwProcessId == po.dwProcessId) named = std::wstring(info[i].strAppName) == L"posix-layer-tests" && info[i].Process.ProcessStartTime.dwHighDateTime != 0;
    check(n != 999 && Lists(info, n, po.dwProcessId) && !Lists(info, n, pm.dwProcessId), "Restart Manager: a file another process has open: that process listed (and not the other)");
    check(named, "Restart Manager: the holder's program name and start time");
    n = HoldersOf(g, info, 16, &reasons);
    check(n != 999 && Lists(info, n, pm.dwProcessId), "Restart Manager: a file another process only has mapped (as Wine maps a DLL): listed");
    struct stat fs, is;
    stat(PosixOf(f).c_str(), &fs);
    stat(PosixOf(idle).c_str(), &is);
    check(wgposix::RmScanProcess((int)po.dwProcessId, {{fs.st_dev, fs.st_ino}}) == 1 && wgposix::RmScanProcess((int)po.dwProcessId, {{is.st_dev, is.st_ino}}) == 0,
          "Restart Manager: one process looked at: it holds the file it has open (1), not another (0)");
    DWORD rc = 0;
    UINT needed = 0;
    HoldersOf(f, info, 0, &reasons, &rc, &needed);
    check(rc == ERROR_MORE_DATA && needed >= 1, "Restart Manager: no room in the list: ERROR_MORE_DATA and the count needed");
    EndChild(po);
    EndChild(pm);
    n = HoldersOf(f, info, 16, &reasons);
    check(n != 999 && !Lists(info, n, po.dwProcessId), "Restart Manager: the holder ended: no longer listed");
    // a process of this user whose /proc entries are closed (not dumpable): unreadable (-1), and the reasons say not everything
    // was seen (on most desktops other processes of the user, the compositor for one, are closed too, so the reason alone
    // cannot single the child out: the scan of the child is checked first)
    const std::string r3 = PosixOf(In(L"ready-rm3"));
    PROCESS_INFORMATION pn = StartChild("--child-nodump " + r3);
    check(pn.hProcess && WaitReady(r3), "Restart Manager: (a child that is not dumpable)");
    check(wgposix::RmScanProcess((int)pn.dwProcessId, {{is.st_dev, is.st_ino}}) == -1, "Restart Manager: a process of this user that cannot be read: unreadable (-1), not 'holds nothing'");
    reasons = 0;
    n = HoldersOf(idle, info, 16, &reasons);
    check(n == 0 && (reasons & RmRebootReasonPermissionDenied), "Restart Manager: ...and the list's reasons say RmRebootReasonPermissionDenied (none seen is not none)");
    EndChild(pn);
    check(wgposix::RmScanProcess((int)pn.dwProcessId, {{is.st_dev, is.st_ino}}) == -2, "Restart Manager: a process that ended: not looked at (-2)");
    if (getuid() != 0) check(wgposix::RmScanProcess(1, {{is.st_dev, is.st_ino}}) == -2, "Restart Manager: another user's process (pid 1, root's): not looked at (-2)");
    DWORD s = 0;
    check(RmStartSession(&s, 0, nullptr) == ERROR_SUCCESS && RmEndSession(s) == ERROR_SUCCESS && RmEndSession(s) == ERROR_INVALID_HANDLE, "Restart Manager: a session ends once");
    UINT z = 0, zn = 0;
    check(RmGetList(s, &z, &zn, nullptr, &reasons) == ERROR_INVALID_HANDLE, "Restart Manager: an ended session: ERROR_INVALID_HANDLE");
    RM_UNIQUE_PROCESS app{};
    RmStartSession(&s, 0, nullptr);
    check(RmRegisterResources(s, 0, nullptr, 1, &app, 0, nullptr) == ERROR_NOT_SUPPORTED, "Restart Manager: registering applications: ERROR_NOT_SUPPORTED");
    RmEndSession(s);
}

// ---- a Windows program under Wine with a WrongGPU binary loaded: a write open is refused, as Windows refuses one of a
// loaded image (the operation guard's game check). The "game" is this program started again under a Windows program's name.
pid_t StartNamed(const char* argv0, const std::vector<std::string>& args) {
    const std::string self = SelfExe();
    std::vector<char*> argv{const_cast<char*>(argv0)};
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = 0;
    return posix_spawn(&pid, self.c_str(), nullptr, nullptr, argv.data(), environ) == 0 ? pid : 0;
}
void StopNamed(pid_t pid) {
    if (pid <= 0) return;
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}
DWORD WriteOpen(const std::wstring& p) {   // the guard's open: 0, or the error
    HANDLE h = CreateFileW(p.c_str(), DELETE | GENERIC_READ | FILE_WRITE_DATA, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return GetLastError();
    CloseHandle(h);
    return 0;
}
const char* kGame = "C:\\Games\\Some Game\\Game.exe";   // as Wine spells a Windows program's command line

void GameImage() {
    const std::wstring dir = In(L"img-game"), dll = dir + L"\\wronggpu_runtime.dll", ini = dir + L"\\wronggpu.ini";
    CreateDirectoryW(dir.c_str(), nullptr);
    Put(dll, "MZ stand-in runtime");
    Put(ini, "[wronggpu]\r\n");
    check(WriteOpen(dll) == 0, "game check: nobody has the DLL loaded: the guard's write open works");
    // running: a Windows program under Wine has it mapped
    const std::string ready = PosixOf(In(L"ready-img1"));
    pid_t game = StartNamed(kGame, {"--child-map", PosixOf(dll), ready});
    check(game && WaitReady(ready), "game check: (a Windows program under Wine has the DLL mapped)");
    check(WriteOpen(dll) == ERROR_SHARING_VIOLATION, "game check: the game has it loaded: a write open fails with ERROR_SHARING_VIOLATION");
    HANDLE t = CreateFileW(dll.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(t == INVALID_HANDLE_VALUE && GetLastError() == ERROR_SHARING_VIOLATION && Slurp(dll) == "MZ stand-in runtime", "game check: ...a truncating open too, whatever it shares; the file as it was");
    if (t != INVALID_HANDLE_VALUE) CloseHandle(t);
    HANDLE r = CreateFileW(dll.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    check(r != INVALID_HANDLE_VALUE, "game check: ...a read open still works (Windows lets a loaded image be read)");
    if (r != INVALID_HANDLE_VALUE) CloseHandle(r);
    {
        wgstate::OperationGuard g(Tmp, {dll, ini}, 500);
        check(g.result() == wgstate::GuardResult::GameRunning && g.which() == dll && std::string(g.words()).find("Close the game") != std::string::npos,
              "game check: the operation guard: GameRunning, the DLL named, the words say to close the game");
    }
    StopNamed(game);
    check(WriteOpen(dll) == 0, "game check: the game ended: the write open works again");
    {
        wgstate::OperationGuard g(Tmp, {dll, ini}, 500);
        check(g.ok(), "game check: the operation guard: Ok once the game ended");
    }
    // not the game: a native program with the DLL mapped (a data mapping: Windows does not refuse that either), and a file that
    // is not a Windows image mapped by the game
    const std::string ready2 = PosixOf(In(L"ready-img2")), ready3 = PosixOf(In(L"ready-img3"));
    pid_t native = StartNamed("/usr/bin/some-native-tool", {"--child-map", PosixOf(dll), ready2});
    check(native && WaitReady(ready2) && WriteOpen(dll) == 0, "game check: a native (not Wine) program has the DLL mapped: not the game, the write open works");
    StopNamed(native);
    const std::string ready6 = PosixOf(In(L"ready-img6"));
    pid_t loader = StartNamed("/opt/proton/files/bin/wine64-preloader", {"--child-map", PosixOf(dll), ready6});
    check(loader && WaitReady(ready6) && WriteOpen(dll) == ERROR_SHARING_VIOLATION,
          "game check: a Wine process named without a Windows path (wine*, or start.exe as Wine names it) with the DLL mapped: refused too");
    StopNamed(loader);
    game = StartNamed(kGame, {"--child-map", PosixOf(ini), ready3});
    check(game && WaitReady(ready3) && WriteOpen(ini) == 0, "game check: the game has a file mapped that is not a Windows image: the write open works");
    StopNamed(game);
    // unreadable: a Wine process of this user whose maps cannot be read (not dumpable): it cannot be ruled out
    const std::string ready4 = PosixOf(In(L"ready-img4")), ready5 = PosixOf(In(L"ready-img5"));
    pid_t closed = StartNamed(kGame, {"--child-nodump", ready4});
    check(closed && WaitReady(ready4), "game check: (a Wine process whose maps cannot be read)");
    int who = 0;
    struct stat st;
    stat(PosixOf(dll).c_str(), &st);
    check(wgposix::ImageInUse(st.st_dev, st.st_ino, &who) == -1 && who == closed, "game check: the scan says it cannot rule that process out (-1, its pid)");
    check(WriteOpen(dll) == ERROR_CANT_ACCESS_FILE, "game check: ...the write open fails with ERROR_CANT_ACCESS_FILE (could not tell), not a sharing violation");
    {
        wgstate::OperationGuard g(Tmp, {dll, ini}, 500);
        check(g.result() == wgstate::GuardResult::Unknown && std::string(g.words()).find("could not tell") != std::string::npos &&
                  std::string(g.words()).find("close the game") != std::string::npos,
              "game check: the operation guard: Unknown, the words say it could not tell and to close the game");
    }
    StopNamed(closed);
    pid_t closedNative = StartNamed("/usr/bin/some-native-tool", {"--child-nodump", ready5});
    check(closedNative && WaitReady(ready5) && WriteOpen(dll) == 0, "game check: an unreadable process that is not a Wine program: not counted, the write open works");
    StopNamed(closedNative);
}

// ---- FileRenameInfo without replace on a file system that lacks RENAME_NOREPLACE: link, then unlink of the old name. When
// that unlink fails the new name goes again and the call fails, so the file keeps only its old name (review fix).
void RenameFallback() {
    const auto realRename = wgposix::g_renameat2;
    const auto realUnlink = wgposix::g_unlink;
    wgposix::g_renameat2 = [](int, const char*, int, const char*, unsigned int) { errno = EINVAL; return -1; };
    const std::wstring src = In(L"fb-src"), dst = In(L"fb-dst");
    CreateDirectoryW(src.c_str(), nullptr);
    CreateDirectoryW(dst.c_str(), nullptr);
    Put(src + L"\\a.dll", "MZ a");
    Put(dst + L"\\taken.dll", "MZ taken");
    HANDLE h = OpenFor(src + L"\\a.dll", DELETE | GENERIC_READ, FILE_SHARE_READ);
    check(!RenameTo(h, dst + L"\\taken.dll", false) && GetLastError() == ERROR_ALREADY_EXISTS && Slurp(dst + L"\\taken.dll") == "MZ taken",
          "rename fallback (no RENAME_NOREPLACE): onto a file: ERROR_ALREADY_EXISTS, nothing replaced");
    check(RenameTo(h, dst + L"\\a.dll", false) && !Exists(src + L"\\a.dll") && Slurp(dst + L"\\a.dll") == "MZ a",
          "rename fallback: link then unlink: the file has only its new name");
    CloseHandle(h);
    // the source folder refuses the unlink (read-only), the target folder allows the link
    Put(src + L"\\c.dll", "MZ c");
    h = OpenFor(src + L"\\c.dll", DELETE | GENERIC_READ, FILE_SHARE_READ);
    if (geteuid() == 0) {
        std::printf("posix-layer: running as root: the read-only source folder check is skipped\n");
    } else {
        chmod(PosixOf(src).c_str(), 0555);
        const bool ok = RenameTo(h, dst + L"\\c.dll", false);
        const DWORD e = GetLastError();
        chmod(PosixOf(src).c_str(), 0755);
        check(!ok && e == ERROR_ACCESS_DENIED && Exists(src + L"\\c.dll") && !Exists(dst + L"\\c.dll"),
              "rename fallback: the old name cannot be removed: the call fails (access denied) and the file keeps only its old name");
        check(SetDisposition(h, FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS) && !Exists(src + L"\\c.dll"),
              "rename fallback: ...the handle still names the old name (a delete through it removes that one)");
    }
    CloseHandle(h);
    // neither unlink works (the old name's nor the undo's): the call still fails; both names are left (said in win32_posix.h)
    Put(src + L"\\d.dll", "MZ d");
    h = OpenFor(src + L"\\d.dll", DELETE | GENERIC_READ, FILE_SHARE_READ);
    wgposix::g_unlink = [](const char*) { errno = EIO; return -1; };
    const bool ok = RenameTo(h, dst + L"\\d.dll", false);
    wgposix::g_unlink = realUnlink;
    check(!ok && Exists(src + L"\\d.dll") && Exists(dst + L"\\d.dll"), "rename fallback: the undo fails too: the call still fails, both names are left");
    CloseHandle(h);
    wgposix::g_renameat2 = realRename;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::strncmp(argv[1], "--child-", 8) == 0) return ChildMain(argc, argv);
    // a registry file of this run's own, and a folder of its own, both under TMPDIR
    wchar_t t[MAX_PATH]{};
    GetTempPathW(MAX_PATH, t);
    Tmp = std::wstring(t) + L"wg-posix-layer-" + std::to_wstring(GetCurrentProcessId());
    CreateDirectoryW(Tmp.c_str(), nullptr);
    const std::string regFile = std::string(getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp") + "/wg-posix-layer-" + std::to_string(GetCurrentProcessId()) + ".ini";
    setenv("WRONGGPU_REGISTRY_FILE", regFile.c_str(), 1);
    Paths();
    Files();
    Times();
    Strings();
    Registry();
    Processes();
    ProcessImage();
    Suspended();
    SuspendedStart();
    CaseFolding();
    RegistryWrites();
    FinishedChildren();
    NotAvailable();
    GpuIdentity();
    AppImageIdentity();
    DownloadsOff();
    Folders();
    ErrorValues();
    BasicInfo();
    Disposition();
    EndOfFile();
    PinnedPattern();
    LinksAndReparse();
    ReplaceFile();
    ProcessEnd();
    NamedMutex();
    Events();
    RenameByHandle();
    LongPathAndSecurity();
    RestartManager();
    GameImage();
    RenameFallback();
    std::remove(regFile.c_str());
    std::error_code ec;
    std::string posixTmp(Tmp.begin(), Tmp.end());
    posixTmp.erase(0, 2);   // "Z:" goes, "\\" becomes "/"
    for (auto& c : posixTmp) if (c == '\\') c = '/';
    std::filesystem::remove_all(posixTmp, ec);
    std::printf("posix-layer: %d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
