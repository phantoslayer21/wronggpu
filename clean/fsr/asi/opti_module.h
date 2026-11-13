// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU OptiScaler ASI plugin: find OptiScaler's module in the running process, whatever name it was loaded under
// (dxgi.dll, winmm.dll, version.dll, ..., OptiScaler.dll), never the plugin itself, the game's executable or a system
// DLL. Written for this project; kernel32 only (no version.dll or psapi import: OptiScaler itself can be the process's
// version.dll).
//
// Candidates, in this order, each scanned with the plugin's own resolver (opti_resolver.h) until one resolves:
//   1. the module InitializeASI returned into (OptiScaler's loader calls it), when that is a candidate at all;
//   2. modules whose version resource names "OptiScaler" (ProductName, FileDescription, ... as UTF-16 text);
//   3. modules loaded under a name OptiScaler is installed as (kProxyNames) or whose file name contains "optiscaler".
// The first candidate the resolver recognises is OptiScaler's menu module. 0.5.0 in-game finding: the return address
// was taken inside std::call_once's lambda, so step 1 alone named the plugin and it scanned its own image.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>

#include <cstdint>
#include <cstring>
#include <cwchar>
#include <functional>
#include <string>
#include <vector>

namespace wgasi {

// The file names OptiScaler is installed under (its proxy names, as WrongGPU's installer and OptiScaler's own
// documentation use them), plus its own name.
inline const wchar_t* const* ProxyNames(size_t& n) {
    static const wchar_t* const names[] = {L"dxgi.dll",    L"winmm.dll",   L"version.dll", L"dbghelp.dll",
                                           L"wininet.dll", L"winhttp.dll", L"d3d12.dll",   L"optiscaler.dll",
                                           L"optiscaler.asi"};
    n = sizeof(names) / sizeof(names[0]);
    return names;
}

inline std::wstring LowerW(std::wstring s) {
    for (auto& c : s) c = wchar_t(towlower(c));
    return s;
}

inline std::string Utf8(const std::wstring& w) {
    if (w.empty()) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(size_t(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), int(w.size()), &s[0], n, nullptr, nullptr);
    return s;
}

// Whether the module's version resource contains the UTF-16 text "OptiScaler" (its ProductName / FileDescription /
// InternalName / OriginalFilename values). Reads the resource of the mapped module; no API of version.dll.
inline bool VersionNamesOptiScaler(HMODULE m) {
    HRSRC res = FindResourceW(m, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(16) /* RT_VERSION */);
    if (!res) return false;
    const DWORD size = SizeofResource(m, res);
    HGLOBAL h = LoadResource(m, res);
    const uint8_t* p = h ? static_cast<const uint8_t*>(LockResource(h)) : nullptr;
    if (!p || size < 20) return false;
    static const wchar_t needle[] = L"OptiScaler";
    const size_t nb = (sizeof(needle) / sizeof(wchar_t) - 1) * sizeof(wchar_t);
    for (size_t i = 0; i + nb <= size; i += 2)
        if (std::memcmp(p + i, needle, nb) == 0) return true;
    return false;
}

struct ModuleCandidate {
    HMODULE module = nullptr;
    std::wstring path;
    std::string why;
};

// The candidates for OptiScaler's module, in the order above. `caller`: InitializeASI's return address (may be null);
// `self`: the plugin's own module. Excluded: self, the process's executable, modules under the Windows folder.
inline std::vector<ModuleCandidate> OptiScalerCandidates(const void* caller, HMODULE self, std::vector<std::string>* skipped = nullptr) {
    wchar_t win[MAX_PATH] = {};
    const UINT wl = GetWindowsDirectoryW(win, MAX_PATH);
    const std::wstring winDir = LowerW(std::wstring(win, wl)) + L"\\";
    const HMODULE exe = GetModuleHandleW(nullptr);
    struct Mod { HMODULE m; std::wstring path; };
    std::vector<Mod> mods;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (snap != INVALID_HANDLE_VALUE) {
        MODULEENTRY32W e = {};
        e.dwSize = sizeof e;
        for (BOOL ok = Module32FirstW(snap, &e); ok; ok = Module32NextW(snap, &e)) mods.push_back({e.hModule, e.szExePath});
        CloseHandle(snap);
    }
    auto excluded = [&](HMODULE m, const std::wstring& path) -> const char* {
        if (m == self) return "the plugin itself";
        if (m == exe) return "the process's executable";
        if (wl && LowerW(path).compare(0, winDir.size(), winDir) == 0) return "a system DLL";
        return nullptr;
    };
    std::vector<ModuleCandidate> out;
    auto add = [&](HMODULE m, const std::wstring& path, const std::string& why) {
        for (auto& c : out)
            if (c.module == m) return;
        if (const char* no = excluded(m, path)) {
            if (skipped) skipped->push_back(Utf8(path) + " (" + why + "): skipped, " + no);
            return;
        }
        out.push_back({m, path, why});
    };
    if (caller) {
        HMODULE m = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(caller), &m) && m) {
            wchar_t file[MAX_PATH] = {};
            GetModuleFileNameW(m, file, MAX_PATH);
            add(m, file, "the caller of InitializeASI");
        }
    }
    for (auto& md : mods)
        if (!excluded(md.m, md.path) && VersionNamesOptiScaler(md.m)) add(md.m, md.path, "its version resource names OptiScaler");
    size_t nn = 0;
    const wchar_t* const* names = ProxyNames(nn);
    for (auto& md : mods) {
        const size_t slash = md.path.find_last_of(L"\\/");
        const std::wstring base = LowerW(slash == std::wstring::npos ? md.path : md.path.substr(slash + 1));
        bool named = base.find(L"optiscaler") != std::wstring::npos;
        for (size_t i = 0; i < nn && !named; ++i) named = base == names[i];
        if (named) add(md.m, md.path, "loaded as " + Utf8(base) + ", a name OptiScaler is installed under");
    }
    return out;
}

}  // namespace wgasi
