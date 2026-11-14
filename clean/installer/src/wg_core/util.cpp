// SPDX-License-Identifier: MIT
// WrongGPU launcher core: small helpers (strings, paths, files, SHA-256), moved unchanged from setup.cpp. Written for
// this project.
#include "util.h"

namespace wg {

[[noreturn]] void Fail(const wstring& m, int code) { throw SetupError{m + L" [WG-SETUP-" + std::to_wstring(code) + L"]", code}; }

wstring Lower(wstring s) { for (auto& c : s) c = (wchar_t)towlower(c); return s; }
bool IEq(const wstring& a, const wstring& b) { return _wcsicmp(a.c_str(), b.c_str()) == 0; }
bool Contains(const wstring& hay, const wchar_t* needle) { return Lower(hay).find(Lower(needle)) != wstring::npos; }
bool EndsWithI(const wstring& s, const wstring& suffix) { return s.size() >= suffix.size() && IEq(s.substr(s.size() - suffix.size()), suffix); }
bool StartsWithI(const wstring& s, const wstring& prefix) { return s.size() >= prefix.size() && IEq(s.substr(0, prefix.size()), prefix); }
wstring Join(const wstring& a, const wstring& b) {
    if (a.empty()) return b;
    wstring r = a;
    if (r.back() != L'\\' && r.back() != L'/') r += L'\\';
    return r + b;
}
wstring Parent(const wstring& p) { auto i = p.find_last_of(L"\\/"); return i == wstring::npos ? wstring() : p.substr(0, i); }
wstring BaseName(const wstring& p) { auto i = p.find_last_of(L"\\/"); return i == wstring::npos ? p : p.substr(i + 1); }
wstring Stem(const wstring& p) { wstring b = BaseName(p); auto i = b.find_last_of(L'.'); return i == wstring::npos ? b : b.substr(0, i); }
wstring ShortenMiddle(const wstring& s, size_t max) {
    if (s.size() <= max || max < 8) return s;
    size_t head = (max - 1) * 2 / 5, tail = max - 1 - head;
    const size_t hs = s.find_last_of(L"\\/", head - 1);
    if (hs != wstring::npos && hs + 1 >= head / 2) head = hs + 1;  // keep "G:\Games\"
    const size_t from = s.size() - tail, ts = s.find_first_of(L"\\/", from);
    if (ts != wstring::npos && ts - from <= tail / 2) tail = s.size() - ts;  // and "\Game\BattlEye"
    return s.substr(0, head) + L"\u2026" + s.substr(s.size() - tail);
}
bool Exists(const wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
bool IsDir(const wstring& p) { DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY); }
bool IsFile(const wstring& p) { DWORD a = GetFileAttributesW(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY); }
unsigned long long FileSize(const wstring& p) {
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExW(p.c_str(), GetFileExInfoStandard, &d)) return 0;
    return ((unsigned long long)d.nFileSizeHigh << 32) | d.nFileSizeLow;
}
wstring FullPath(const wstring& p) {
    wchar_t b[MAX_PATH * 2]; DWORD n = GetFullPathNameW(p.c_str(), (DWORD)std::size(b), b, nullptr);
    return n && n < std::size(b) ? wstring(b, n) : p;
}
wstring Hex32(DWORD v) { wchar_t b[16]; swprintf(b, 16, L"0x%08X", (unsigned)v); return b; }
wstring ErrText(DWORD e) {
    wchar_t* b = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, e, 0, (LPWSTR)&b, 0, nullptr);
    wstring s = b ? b : L"Unknown error";
    if (b) LocalFree(b);
    while (!s.empty() && (s.back() == L'\r' || s.back() == L'\n' || s.back() == L' ')) s.pop_back();
    return s + L" (" + Hex32(e) + L")";
}
string ToUtf8(const wstring& w) {
    if (w.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}
wstring FromUtf8(const string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
bool ReadAll(const wstring& path, string& out) {
    // FILE_SHARE_DELETE: a read never stops another thread's move or delete of the file (the launcher's data check hashing
    // the model file while a conversion swaps it); the read goes on from the handle
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz{};
    GetFileSizeEx(h, &sz);
    out.resize((size_t)sz.QuadPart);
    size_t done = 0;
    while (done < out.size()) {
        DWORD got = 0;
        if (!ReadFile(h, out.data() + done, (DWORD)std::min<size_t>(out.size() - done, 1u << 24), &got, nullptr) || !got) break;
        done += got;
    }
    CloseHandle(h);
    return done == out.size();
}
// Case-insensitive search for an ASCII needle, as ASCII or as UTF-16LE text.
bool ContainsText(const uint8_t* d, size_t n, const char* needle) {
    const size_t k = strlen(needle);
    if (!d || !k || n < k) return false;
    auto low = [](uint8_t c) { return (uint8_t)(c >= 'A' && c <= 'Z' ? c + 32 : c); };
    const uint8_t first = low((uint8_t)needle[0]);
    for (size_t i = 0; i + k <= n; ++i) {
        if (low(d[i]) != first) continue;
        size_t j = 1;
        while (j < k && low(d[i + j]) == low((uint8_t)needle[j])) ++j;
        if (j == k) return true;
        if (i + 2 * k <= n && d[i + 1] == 0) {  // UTF-16LE candidate
            j = 1;
            while (j < k && low(d[i + 2 * j]) == low((uint8_t)needle[j]) && d[i + 2 * j + 1] == 0) ++j;
            if (j == k) return true;
        }
    }
    return false;
}
bool FileContainsText(const wstring& path, const char* needle, unsigned long long limit) {
    MappedFile f(path, limit);
    return ContainsText(f.data, f.size, needle);
}
void EnsureDir(const wstring& dir, std::vector<wstring>* created) {
    if (dir.empty() || IsDir(dir)) return;
    if (dir.size() <= 3 && dir.size() >= 2 && dir[1] == L':') return;
    EnsureDir(Parent(dir), created);
    if (!CreateDirectoryW(dir.c_str(), nullptr)) {
        DWORD e = GetLastError();
        if (e == ERROR_ALREADY_EXISTS && IsDir(dir)) return;
        if (e == ERROR_ALREADY_EXISTS) Fail(L"A file is in the way of the folder " + dir + L". Remove or rename it and run setup again.");
        Fail(L"Cannot create folder " + dir + L": " + ErrText(e));
    }
    if (created) created->push_back(dir);
}
void ForceDelete(const wstring& p) {
    SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!DeleteFileW(p.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND) Fail(L"Cannot delete " + p + L": " + ErrText(GetLastError()));
}
void WriteFileAtomic(const wstring& path, const void* data, size_t size) {
    wstring tmp = path + kTmpSuffix;
    HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) Fail(L"Cannot write " + path + L": " + ErrText(GetLastError()));
    const char* p = static_cast<const char*>(data);
    size_t done = 0;
    while (done < size) {
        DWORD wrote = 0;
        if (!WriteFile(h, p + done, (DWORD)std::min<size_t>(size - done, 1u << 24), &wrote, nullptr)) {
            DWORD e = GetLastError(); CloseHandle(h); DeleteFileW(tmp.c_str());
            Fail(L"Write failed for " + path + L": " + ErrText(e));
        }
        done += wrote;
    }
    FlushFileBuffers(h);
    CloseHandle(h);
    if (Exists(path)) SetFileAttributesW(path.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DWORD e = GetLastError(); DeleteFileW(tmp.c_str());
        Fail(L"Cannot replace " + path + L": " + ErrText(e) + L". Is the file in use?");
    }
}
void CopyChecked(const wstring& src, const wstring& dst) {
    EnsureDir(Parent(dst));
    if (Exists(dst)) SetFileAttributesW(dst.c_str(), FILE_ATTRIBUTE_NORMAL);
    if (!CopyFileW(src.c_str(), dst.c_str(), FALSE)) Fail(L"Cannot copy " + src + L" to " + dst + L": " + ErrText(GetLastError()));
}
void RemoveTree(const wstring& dir, const wstring& keepFile) {
    // Check the root too: enumerating dir\* would otherwise follow a root junction into its target.
    const DWORD attrs = GetFileAttributesW(dir.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) return;
    if (attrs & FILE_ATTRIBUTE_REPARSE_POINT) { RemoveDirectoryW(dir.c_str()); return; }
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            wstring p = Join(dir, n);
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) RemoveDirectoryW(p.c_str());
                else RemoveTree(p, keepFile);
            } else if (keepFile.empty() || !IEq(p, keepFile)) {
                SetFileAttributesW(p.c_str(), FILE_ATTRIBUTE_NORMAL);
                DeleteFileW(p.c_str());
            }
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    RemoveDirectoryW(dir.c_str());
}
// Files (not folders) in dir whose name matches the wildcard pattern.
std::vector<wstring> ListFiles(const wstring& dir, const wchar_t* pattern) {
    std::vector<wstring> r;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, pattern).c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return r;
    do { if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) r.push_back(Join(dir, fd.cFileName)); } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(r.begin(), r.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return r;
}
std::vector<wstring> ListDirs(const wstring& dir) {
    std::vector<wstring> r;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return r;
    do {
        wstring n = fd.cFileName;
        if (n == L"." || n == L"..") continue;
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) r.push_back(Join(dir, n));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(r.begin(), r.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return r;
}
// Every file named `name` below root (depth-limited, budgeted, no links), sorted by path.
std::vector<wstring> FindFilesNamed(const wstring& root, const wchar_t* name, size_t budget, int maxDepth) {
    std::vector<wstring> out;
    std::vector<std::pair<wstring, int>> stack{{root, 0}};
    size_t visited = 0;
    while (!stack.empty() && visited < budget) {
        auto [dir, depth] = stack.back(); stack.pop_back();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(Join(dir, L"*").c_str(), &fd);
        if (h == INVALID_HANDLE_VALUE) continue;
        do {
            ++visited;
            wstring n = fd.cFileName;
            if (n == L"." || n == L"..") continue;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                if (depth < maxDepth && !(fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) && !IEq(n, kStateDirName) && !IEq(n, kLegacyStateDir))
                    stack.push_back({Join(dir, n), depth + 1});
            } else if (IEq(n, name)) out.push_back(Join(dir, n));
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    std::sort(out.begin(), out.end(), [](const wstring& a, const wstring& b) { return Lower(a) < Lower(b); });
    return out;
}

// SHA-256 with the converter's FIPS 180-4 implementation (no bcrypt dependency).
wstring HexW(const wgc::Digest& d) { return FromUtf8(wgc::to_hex(d)); }
wstring Sha256Bytes(const void* data, size_t size) { return HexW(wgc::Sha256::hash(data, size)); }
wstring Sha256File(const wstring& path) {
    MappedFile f(path, ~0ull);
    if (f.data) return Sha256Bytes(f.data, f.size);
    if (IsFile(path) && FileSize(path) == 0) return Sha256Bytes("", 0);
    return L"";
}
wstring NowStamp(bool forFileName) {
    SYSTEMTIME t; GetLocalTime(&t);
    wchar_t b[40];
    if (forFileName) swprintf(b, 40, L"%04d%02d%02d-%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute);
    else swprintf(b, 40, L"%04d-%02d-%02d %02d:%02d:%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    return b;
}
wstring EnvVar(const wchar_t* name) {
    wchar_t b[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(name, b, (DWORD)std::size(b));
    return n && n < std::size(b) ? wstring(b, n) : wstring();
}
// System DLLs this exe uses only sometimes are loaded from System32 only, never from the download folder.
HMODULE LoadSystemDll(const wchar_t* name) { return LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); }
bool UnderWine() { return GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr; }
wstring PathForGamePrefix(const wstring& path) {
    if (path.empty() || !UnderWine()) return path;
    // Wine's kernel32: the Unix name of a DOS path, allocated on the process heap (UTF-8). It fails when more than the last
    // part is missing (the RR data's F\ folder before its first conversion), so the deepest folder that exists is converted
    // and the missing rest appended.
    using UnixNameFn = char*(CDECL*)(const wchar_t*);
    auto unixName = reinterpret_cast<UnixNameFn>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "wine_get_unix_file_name")));
    if (!unixName) return path;
    wstring head = path, tail;
    char* u = nullptr;
    while (!(u = unixName(head.c_str()))) {
        const size_t cut = head.find_last_of(L"\\/");
        if (cut == wstring::npos || cut < 3) return path;
        tail = head.substr(cut) + tail;
        head.resize(cut);
    }
    wstring z = L"Z:" + FromUtf8(u) + tail;
    HeapFree(GetProcessHeap(), 0, u);
    for (wchar_t& c : z)
        if (c == L'/') c = L'\\';
    return z;
}
wstring SelfPath() { wchar_t b[MAX_PATH * 2]; DWORD n = GetModuleFileNameW(nullptr, b, (DWORD)std::size(b)); return wstring(b, n); }

}  // namespace wg
