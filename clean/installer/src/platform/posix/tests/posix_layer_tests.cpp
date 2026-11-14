// SPDX-License-Identifier: MIT
// Tests of the Linux layer itself (src/platform/posix): the Windows calls it implements, checked against what Windows does.
// The core's own tests exercise it only as far as the core goes; this covers the rest (sharing, processes, the registry file).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <windows.h>

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

void NotAvailable() {
    check(LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32) == nullptr && GetLastError() == ERROR_MOD_NOT_FOUND, "no Windows DLL loads");
    check(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion") == nullptr, "and GetProcAddress of a missing module is null");
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
}  // namespace

int main() {
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
    NotAvailable();
    Folders();
    std::remove(regFile.c_str());
    std::error_code ec;
    std::string posixTmp(Tmp.begin(), Tmp.end());
    posixTmp.erase(0, 2);   // "Z:" goes, "\\" becomes "/"
    for (auto& c : posixTmp) if (c == '\\') c = '/';
    std::filesystem::remove_all(posixTmp, ec);
    std::printf("posix-layer: %d checks, %d failed\n", checks, failures);
    return failures ? 1 : 0;
}
