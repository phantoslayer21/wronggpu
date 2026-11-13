// SPDX-License-Identifier: MIT
// Test builds only: a test program refuses to start when its temp folder is outside the folder the build was configured
// with (CMake WRONGGPU_TEST_TEMP_ROOT -> WG_TEST_TEMP_ROOT; empty: no check), so a test never writes outside it. Checked:
// the TMP and TEMP variables (each when set) and the folder Windows hands out (GetTempPathW). Header only, no wg_core
// dependency (the wizard policy tests use it too). Written for this project.
#pragma once
#include <windows.h>

#include <cwctype>
#include <string>

namespace wg_test {

// path is root or a folder below it: full paths, either slash, any case, never by a bare prefix ("Q:\Root-x" is not below
// "Q:\Root")
inline bool UnderRoot(const std::wstring& path, const std::wstring& root) {
    auto full = [](const std::wstring& p) {
        wchar_t buf[MAX_PATH * 2] = {};
        const DWORD n = GetFullPathNameW(p.c_str(), MAX_PATH * 2, buf, nullptr);
        std::wstring s = n && n < MAX_PATH * 2 ? std::wstring(buf, n) : p;
        for (auto& c : s) c = c == L'/' ? L'\\' : (wchar_t)towlower(c);
        while (s.size() > 3 && s.back() == L'\\') s.pop_back();
        return s;
    };
    if (path.empty() || root.empty()) return false;
    const std::wstring p = full(path), r = full(root);
    return p == r || (p.size() > r.size() && p.compare(0, r.size(), r) == 0 && (p[r.size()] == L'\\' || r.back() == L'\\'));
}

// Empty when the temp folder is below root; else why the program refuses.
inline std::wstring TempOutsideRoot(const std::wstring& root) {
    if (root.empty()) return std::wstring();
    for (const wchar_t* name : {L"TMP", L"TEMP"}) {
        wchar_t v[MAX_PATH * 2] = {};
        const DWORD n = GetEnvironmentVariableW(name, v, MAX_PATH * 2);
        if (n && n < MAX_PATH * 2 && !UnderRoot(v, root))
            return std::wstring(L"refused: ") + name + L"=" + v + L" is outside " + root + L" (WRONGGPU_TEST_TEMP_ROOT)";
    }
    wchar_t t[MAX_PATH + 1] = {};
    if (!GetTempPathW(MAX_PATH + 1, t) || !UnderRoot(t, root))
        return std::wstring(L"refused: the temp folder ") + t + L" is outside " + root + L" (WRONGGPU_TEST_TEMP_ROOT)";
    return std::wstring();
}

#ifdef WG_TEST_TEMP_ROOT
#define WG_TEST_WIDEN2(x) L##x
#define WG_TEST_WIDEN(x) WG_TEST_WIDEN2(x)
inline std::wstring TempGuard() { return TempOutsideRoot(WG_TEST_WIDEN(WG_TEST_TEMP_ROOT)); }
#else
inline std::wstring TempGuard() { return std::wstring(); }
#endif
const int kTempGuardExit = 97;  // the exit code of a refused start

// Writes the refusal to stderr (also from a WIN32 program whose parent gave it one); true when the program must not start.
inline bool TempGuardRefuses() {
    const std::wstring why = TempGuard();
    if (why.empty()) return false;
    const std::wstring line = why + L"\n";
    const int n = WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int)line.size(), nullptr, 0, nullptr, nullptr);
    std::string u(n > 0 ? n : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, line.c_str(), (int)line.size(), &u[0], n, nullptr, nullptr);
    const HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD done = 0;
    if (h && h != INVALID_HANDLE_VALUE) WriteFile(h, u.data(), (DWORD)u.size(), &done, nullptr);
    return true;
}

}  // namespace wg_test
